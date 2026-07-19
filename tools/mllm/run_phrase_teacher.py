#!/usr/bin/env python
"""Run Qwen3-VL or InternVL3.5-HF on phrase relationship cases.

The runner is deterministic, resumable, and writes both raw and parsed output.
It supports the propagation cases used by v16.6.0 and comparative cases used by
v16.7.0.  Teacher/model differences do not change the requested JSON schema.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import os.path as op
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.mllm.phrase_teacher_common import (
    SYSTEM_PROMPT,
    build_teacher_prompt,
    extract_json_object,
    ordered_case_images,
    read_jsonl,
    validate_teacher_payload,
    write_jsonl,
)


class TeacherOutputError(RuntimeError):
    """Carry an unparseable raw generation into the retry/audit log."""

    def __init__(self, message, raw_output, image_order):
        super().__init__(message)
        self.raw_output = raw_output
        self.image_order = image_order


def parse_args():
    parser = argparse.ArgumentParser(description="Run phrase MLLM teacher")
    parser.add_argument("--teacher", required=True, choices=["qwen3vl", "internvl35"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--order", required=True, choices=["forward", "reverse"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "int8"])
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model_and_processor(model_path: str, precision: str):
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "The MLLM environment requires a recent transformers installation"
        ) from exc

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoVisionModel.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    device_map = getattr(model, "hf_device_map", {}) or {}
    forbidden = {
        str(value).lower()
        for value in device_map.values()
        if str(value).lower() in {"cpu", "disk", "meta"}
    }
    if forbidden:
        raise RuntimeError(
            "Teacher model offloaded to forbidden devices: {}".format(sorted(forbidden))
        )
    parameter_devices = {str(parameter.device).lower() for parameter in model.parameters()}
    forbidden_parameter_devices = {
        value
        for value in parameter_devices
        if value == "cpu" or value == "meta" or value.startswith("disk")
    }
    if forbidden_parameter_devices:
        raise RuntimeError(
            "Teacher parameters found on forbidden devices: {}".format(
                sorted(forbidden_parameter_devices)
            )
        )
    return model, processor


def model_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Cannot determine teacher model device")


def infer_one(model, processor, case, order, max_new_tokens):
    images_meta = ordered_case_images(case, order)
    images = [Image.open(item["path"]).convert("RGB") for item in images_meta]
    prompt = build_teacher_prompt(case, order)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in images]
                + [{"type": "text", "text": prompt}]
            ),
        },
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = SYSTEM_PROMPT + "\n\n" + prompt
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    device = model_device(model)
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
        )
    input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    generated_only = generated[:, input_length:] if input_length else generated
    raw = processor.batch_decode(
        generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    try:
        payload = extract_json_object(raw)
        parsed = validate_teacher_payload(case, payload)
    except Exception as exc:
        raise TeacherOutputError(str(exc), raw, images_meta) from exc
    return raw, parsed, images_meta


def existing_case_ids(path: Path):
    if not path.is_file():
        return set()
    return {
        str(row.get("case_id"))
        for row in read_jsonl(str(path))
        if bool(row.get("parsed_ok", False))
    }


def attempt_counts(path: Path):
    counts = {}
    if not path.is_file():
        return counts
    for row in read_jsonl(str(path)):
        case_id = str(row.get("case_id"))
        counts[case_id] = counts.get(case_id, 0) + 1
    return counts


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_manifest(cli, output, started_at, success, failure):
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    manifest = {
        "teacher": cli.teacher,
        "model_path": str(Path(cli.model_path).expanduser().resolve()),
        "order": cli.order,
        "precision": cli.precision,
        "cases_path": str(Path(cli.cases).expanduser().resolve()),
        "cases_sha256": file_sha256(cli.cases),
        "prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT + inspect.getsource(build_teacher_prompt)).encode("utf-8")
        ).hexdigest(),
        "output_file": str(output),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(cli.max_new_tokens),
        "start_time_unix": started_at,
        "end_time_unix": time.time(),
        "success": int(success),
        "failure": int(failure),
    }
    manifest_path = output.with_suffix(output.suffix + ".run_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    cli = parse_args()
    run_started = time.time()
    cases = read_jsonl(cli.cases)
    output = Path(cli.output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and cli.overwrite:
        output.unlink()
    completed = existing_case_ids(output)
    attempts = attempt_counts(output)

    selected = cases[int(cli.start_index) :]
    if cli.max_cases > 0:
        selected = selected[: int(cli.max_cases)]

    model, processor = load_model_and_processor(cli.model_path, cli.precision)
    success = 0
    failure = 0
    for position, case in enumerate(selected, start=1):
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        started = time.time()
        row = {
            "case_id": case_id,
            "case_type": case.get("case_type", "propagation"),
            "teacher": cli.teacher,
            "order": cli.order,
            "precision": cli.precision,
            "model_path": str(Path(cli.model_path).expanduser().resolve()),
            "attempt": attempts.get(case_id, 0) + 1,
        }
        try:
            raw, parsed, image_order = infer_one(
                model, processor, case, cli.order, cli.max_new_tokens
            )
            row.update(
                {
                    "parsed_ok": True,
                    "phrases": parsed,
                    "raw_output": raw,
                    "image_order": image_order,
                    "elapsed_seconds": time.time() - started,
                }
            )
            success += 1
        except TeacherOutputError as exc:
            cause = exc.__cause__
            row.update(
                {
                    "parsed_ok": False,
                    "phrases": {},
                    "raw_output": exc.raw_output,
                    "image_order": exc.image_order,
                    "error_type": (
                        type(cause).__name__ if cause is not None else type(exc).__name__
                    ),
                    "error": str(exc),
                    "elapsed_seconds": time.time() - started,
                }
            )
            failure += 1
        except Exception as exc:
            row.update(
                {
                    "parsed_ok": False,
                    "phrases": {},
                    "raw_output": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.time() - started,
                }
            )
            failure += 1
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "[{}/{}] {} parsed={} elapsed={:.2f}s".format(
                position,
                len(selected),
                case_id,
                row["parsed_ok"],
                row["elapsed_seconds"],
            ),
            flush=True,
        )

    print(json.dumps({"success": success, "failure": failure}, indent=2))
    write_run_manifest(cli, output, run_started, success, failure)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
