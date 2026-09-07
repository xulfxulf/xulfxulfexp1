"""Resumable serial queue. Explicit --execute starts work; no hash approval gate."""
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from .common import ROOT, read_json, utcnow, write_json
from .runtime import disk_guard
from .validation import critical_config


def queue_plan(config_path, output_dir):
    out = Path(output_dir).resolve()
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(ROOT / "run_view4.py"), "train", "--config", str(Path(config_path).resolve())]
    commands = []
    e0 = out / "E0_seed1_formal/checkpoints/best.pth"
    for experiment, seed, kind in [("E0", 1, "formal")] + [
            (e, 1, "smoke") for e in ("E1", "E2", "E3")] + [
            (e, s, "formal") for s in (1, 2, 3) for e in ("E1", "E2", "E3")]:
        name = "%s_seed%d_%s" % (experiment, seed, kind)
        cmd = base + ["--experiment", experiment, "--seed", str(seed), "--run-kind", kind,
                      "--output-dir", str(out / name)]
        if experiment != "E0":
            cmd += ["--e0-checkpoint", str(e0)]
        commands.append({"name": name, "experiment": experiment, "kind": kind, "seed": seed, "command": cmd})
    return commands


def check_other_training():
    matches = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = [s.decode(errors="replace") for s in path.read_bytes().split(b"\0") if s]
        except (OSError, ProcessLookupError):
            continue
        if any(Path(arg).name in ("train.py", "main.py") for arg in args) or (
                any(Path(arg).name == "run_view4.py" for arg in args) and "train" in args):
            matches.append(path.parent.name)
    if matches:
        raise RuntimeError("Another training process exists: " + ",".join(matches))
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if gpu.returncode or gpu.stdout.strip():
        raise RuntimeError("GPU unavailable or another compute process exists")


def _stamp():
    return utcnow().replace(":", "").replace(".", "_")


def archive_incomplete(path, queue_dir):
    """Preserve attempts that have no resumable checkpoint, instead of overwriting."""
    path = Path(path)
    if path.exists():
        archive = Path(queue_dir) / "failed_attempts"
        archive.mkdir(exist_ok=True)
        path.rename(archive / (path.name + "_" + _stamp()))


def completed_run(item, out):
    result_path = Path(out) / item["name"] / "result.json"
    if not result_path.exists():
        return False
    result = read_json(result_path)
    if result.get("status") != "complete":
        return False
    actual = tuple(result.get(k) for k in ("experiment", "seed", "run_kind", "epochs"))
    expected = (item["experiment"], item["seed"], item["kind"], 5 if item["kind"] == "formal" else 2)
    if actual != expected:
        raise ValueError("Completed result has mismatched run identity: " + item["name"])
    if not (result_path.parent / "checkpoints/best.pth").is_file():
        raise FileNotFoundError("Completed run is missing best.pth: " + item["name"])
    return True


def continuation_command(item, out):
    folder = Path(out) / item["name"]
    last = folder / "checkpoints/last.pth"
    command = list(item["command"])
    if last.is_file():
        command += ["--resume", str(last)]
    elif folder.exists():
        archive_incomplete(folder, out)
    return command


def run_queue(cfg, config_path, output_dir, execute=False, review_approval=None, resume=False):
    out = Path(output_dir).resolve()
    if out.exists() and not resume:
        raise FileExistsError("Queue directory exists. Use --resume to continue it, or choose a new directory.")
    if resume and not (out / "queue_plan.json").is_file():
        raise ValueError("--resume requires an existing queue_plan.json")
    out.mkdir(parents=True, exist_ok=True)
    commands = queue_plan(config_path, out)
    stage = "initialization"
    # One lock covers inspection and all state changes, including plan-only calls.
    with open(ROOT / ".queue.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if resume:
                old_plan = read_json(out / "queue_plan.json")
                if [(x["name"], x["experiment"], x["seed"], x["kind"]) for x in old_plan] != [
                        (x["name"], x["experiment"], x["seed"], x["kind"]) for x in commands]:
                    raise ValueError("Queue layout changed; create a new experiment group")
                if (out / "queue_config.json").is_file():
                    if critical_config(read_json(out / "queue_config.json")) != critical_config(cfg):
                        raise ValueError("Queue data/training settings changed; create a new experiment group")
            write_json(out / "queue_plan.json", commands)
            write_json(out / "queue_config.json", cfg)
            if not execute:
                if not resume:
                    write_json(out / "queue_status.json", {"status": "planned_only", "created_at": utcnow()})
                return
            # Optional user confirmation file; never tied to a code SHA.
            if review_approval and read_json(review_approval).get("approved") is not True:
                raise RuntimeError("The supplied review approval is not approved")
            from . import __version__
            with open(out / "queue_history.jsonl", "a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps({"time": utcnow(), "resume": resume,
                             "implementation_version": __version__}) + "\n")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="4", NLTK_DATA=cfg["nltk_data"])

            def run_command(name, cmd):
                nonlocal stage
                stage = name
                write_json(out / "queue_status.json", {"status": "running", "current": name, "time": utcnow()})
                check_other_training()
                with open(out / (name + "." + _stamp() + ".log"), "x", encoding="utf-8") as stream:
                    subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)

            def gate(name, folder, record_name, command):
                nonlocal stage
                stage = name
                record = folder / record_name
                if record.is_file():
                    previous = read_json(record)
                    if (previous.get("status") in ("passed", "complete")
                            and previous.get("implementation_version") == __version__):
                        return
                archive_incomplete(folder, out)
                run_command(name, command)
                if not record.is_file() or read_json(record).get("status") not in ("passed", "complete"):
                    raise RuntimeError("Gate did not produce a passing record: " + name)

            stage = "resource_precheck"
            remaining = sum(not completed_run(item, out) for item in commands)
            if remaining:
                disk_guard(out, 1800 * 1024**2, remaining_runs=remaining)
            gate("gpu_equivalence", out / "gpu_equivalence", "run_manifest.json", [
                sys.executable, str(ROOT / "run_view4.py"), "audit", "--config", str(config_path),
                "--suite", "gpu-equivalence", "--device", "cuda:0", "--output-dir", str(out / "gpu_equivalence")])
            gate("gpu_distributed", out / "gpu_distributed", "run_manifest.json", [
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                "-m", "tests.distributed_checks", "--cuda", "--nltk-data", cfg["nltk_data"],
                "--output-dir", str(out / "gpu_distributed")])
            for item in commands:
                stage = item["name"]
                if not completed_run(item, out):
                    run_command(item["name"], continuation_command(item, out))
                    if not completed_run(item, out):
                        raise RuntimeError("Run did not complete: " + item["name"])
                # Also run missing post-run gates after skipping a completed run.
                if item["experiment"] == "E0":
                    gate("e0_reference_validation", out / "e0_reference_validation", "reference_validation.json", [
                        sys.executable, str(ROOT / "run_view4.py"), "audit", "--config", str(config_path),
                        "--suite", "reference-val", "--checkpoint", str(out / item["name"] / "checkpoints/best.pth"),
                        "--device", "cuda:0", "--output-dir", str(out / "e0_reference_validation")])
                if item["name"] == "E3_seed1_smoke":
                    stage = "epoch1_parity_gate"
                    folder = out / stage
                    if (not (folder / "parity.json").is_file()
                            or read_json(folder / "parity.json").get("status") != "passed"
                            or read_json(folder / "parity.json").get("implementation_version") != __version__):
                        archive_incomplete(folder, out)
                        from .audit import compare_epoch1
                        compare_epoch1(out / "E2_seed1_smoke/checkpoints/epoch1_parity.pth",
                                       out / "E3_seed1_smoke/checkpoints/epoch1_parity.pth", folder)
            stage = "summary"
            from .report import summarize_queue
            if not (out / "summary/results.json").is_file():
                archive_incomplete(out / "summary", out)
                summarize_queue(out, out / "summary")
            write_json(out / "queue_status.json", {"status": "complete", "ended_at": utcnow()})
        except BaseException as exc:
            write_json(out / "queue_status.json", {"status": "paused_failed", "current": stage,
                       "error": repr(exc), "time": utcnow(), "prior_records_preserved": True})
            raise
