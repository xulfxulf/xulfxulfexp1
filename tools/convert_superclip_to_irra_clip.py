import argparse
import os
import sys
from collections import Counter

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.clip_model import build_CLIP_from_openai_pretrained  # noqa: E402


OUTER_KEYS = ("state_dict", "model", "module")
STRIP_PREFIXES = (
    "module.",
    "model.",
    "clip.",
    "clip_model.",
    "base_model.",
    "module.model.",
    "module.clip.",
    "module.clip_model.",
    "module.base_model.",
)
CLASSIFIER_PREFIXES = (
    "classifier",
    "head",
    "heads",
    "fc",
    "logit_scale",
    "teacher",
    "queue",
    "optimizer",
    "scheduler",
    "epoch",
    "iter",
)
CLASSIFIER_INFIXES = (
    ".classifier.",
    ".head.",
    ".heads.",
    ".fc.",
    ".teacher.",
    ".queue.",
)
KEY_MODULES = (
    "visual.conv1",
    "visual.transformer",
    "token_embedding",
    "transformer",
    "ln_final",
    "text_projection",
)
POSITIONAL_KEYS = {"visual.positional_embedding", "positional_embedding"}


def load_tensor_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        state_dict = checkpoint.state_dict()
        tensor_state = {
            str(k): v.detach().cpu() for k, v in state_dict.items()
            if torch.is_tensor(v)
        }
        if not tensor_state:
            raise RuntimeError("TorchScript checkpoint has no tensor state_dict entries.")
        return tensor_state, ["<torchscript>.state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict-like object, got {type(checkpoint)!r}")

    current = checkpoint
    used = []
    for key in OUTER_KEYS:
        value = current.get(key) if isinstance(current, dict) else None
        if isinstance(value, dict):
            current = value
            used.append(key)
            break

    tensor_state = {
        str(k): v.detach().cpu() for k, v in current.items()
        if torch.is_tensor(v)
    }
    if not tensor_state:
        raise RuntimeError(
            "No tensor state_dict entries found. Top-level keys: "
            + ", ".join(str(k) for k in list(checkpoint.keys())[:50])
        )
    return tensor_state, used


def normalize_key(key):
    out = key
    changed = True
    while changed:
        changed = False
        for prefix in STRIP_PREFIXES:
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def is_classifier_or_runtime_key(key):
    return key.startswith(CLASSIFIER_PREFIXES) or any(part in key for part in CLASSIFIER_INFIXES)


def normalize_tensor(key, value):
    if key in POSITIONAL_KEYS and value.ndim == 3 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def compatible_positional(key, value, target_value):
    if key == "visual.positional_embedding":
        return value.ndim == 2 and target_value.ndim == 2 and value.shape[-1] == target_value.shape[-1]
    if key == "positional_embedding":
        return tuple(value.shape) == tuple(target_value.shape)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SuperCLIP checkpoint to a pure IRRA CLIP state_dict."
    )
    parser.add_argument("--superclip_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--img_size", nargs=2, type=int, default=[384, 128])
    parser.add_argument("--stride_size", type=int, default=16)
    args = parser.parse_args()

    clip_model, _ = build_CLIP_from_openai_pretrained(
        args.pretrain_choice,
        image_size=tuple(args.img_size),
        stride_size=args.stride_size,
    )
    target = clip_model.state_dict()

    source, used_outer = load_tensor_state_dict(args.superclip_ckpt)
    filtered = {}
    matched = []
    positional_kept_with_shape_change = []
    shape_mismatch = []
    ignored_classifier = []
    ignored_unknown = []
    normalized_collisions = Counter()

    for raw_key, raw_value in source.items():
        key = normalize_key(raw_key)
        normalized_collisions[key] += 1

        if is_classifier_or_runtime_key(key):
            ignored_classifier.append(raw_key)
            continue
        if key not in target:
            ignored_unknown.append(raw_key)
            continue

        value = normalize_tensor(key, raw_value)
        if tuple(value.shape) == tuple(target[key].shape):
            filtered[key] = value
            matched.append(key)
        elif key in POSITIONAL_KEYS and compatible_positional(key, value, target[key]):
            filtered[key] = value
            positional_kept_with_shape_change.append((key, tuple(value.shape), tuple(target[key].shape)))
        else:
            shape_mismatch.append((raw_key, key, tuple(value.shape), tuple(target[key].shape)))

    collisions = [key for key, count in normalized_collisions.items() if count > 1]
    if collisions:
        print("Warning: normalized key collisions detected:")
        for key in collisions[:50]:
            print(f"  {key}: {normalized_collisions[key]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(filtered, args.output)

    encoder_target_keys = [
        key for key in target
        if key.startswith("visual.")
        or key.startswith("transformer.")
        or key.startswith("token_embedding.")
        or key in {"positional_embedding", "ln_final.weight", "ln_final.bias", "text_projection"}
    ]
    encoder_loaded = [key for key in filtered if key in encoder_target_keys]
    loaded_key_ratio = len(encoder_loaded) / max(1, len(encoder_target_keys))
    target_param_count = sum(target[key].numel() for key in encoder_target_keys)
    loaded_param_count = sum(filtered[key].numel() for key in encoder_loaded)
    loaded_param_ratio = loaded_param_count / max(1, target_param_count)

    print(f"Input checkpoint: {args.superclip_ckpt}")
    print(f"Output checkpoint: {args.output}")
    print(f"Unwrapped outer keys: {used_outer or '<none>'}")
    print(f"Total source tensor keys: {len(source)}")
    print(f"Current CLIP target keys: {len(target)}")
    print(f"Successfully matched keys: {len(matched)}")
    print(f"Positional keys kept with shape change: {len(positional_kept_with_shape_change)}")
    print(f"Shape-mismatched keys skipped: {len(shape_mismatch)}")
    print(f"Ignored classifier/runtime keys: {len(ignored_classifier)}")
    print(f"Ignored unknown keys: {len(ignored_unknown)}")
    print(f"Encoder loaded key ratio: {loaded_key_ratio:.3f}")
    print(f"Encoder loaded parameter ratio: {loaded_param_ratio:.3f}")

    if positional_kept_with_shape_change:
        print("\nPositional keys kept for current load_param resize logic:")
        for key, src_shape, dst_shape in positional_kept_with_shape_change:
            print(f"  {key}: source {src_shape} -> target {dst_shape}")

    if shape_mismatch:
        print("\nFirst 50 shape mismatches:")
        for raw_key, key, src_shape, dst_shape in shape_mismatch[:50]:
            print(f"  {raw_key} -> {key}: source {src_shape}, target {dst_shape}")

    print("\nCritical module coverage:")
    for prefix in KEY_MODULES:
        target_count = sum(1 for key in target if key.startswith(prefix) or key == prefix)
        loaded_count = sum(1 for key in filtered if key.startswith(prefix) or key == prefix)
        print(f"  {prefix}: {loaded_count}/{target_count}")

    required = ("visual.conv1.weight", "token_embedding.weight", "ln_final.weight", "text_projection")
    missing_required = [key for key in required if key not in filtered]
    if missing_required:
        raise RuntimeError("Converted checkpoint is missing required CLIP keys: " + ", ".join(missing_required))
    if loaded_param_ratio < 0.70:
        raise RuntimeError(f"Loaded encoder parameter ratio is too low: {loaded_param_ratio:.3f} < 0.70")


if __name__ == "__main__":
    main()
