import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .common import capture_rng, rank, world, load_training_checkpoint


def configure_backends():
    """One numeric backend policy for training, evaluation and reference checks."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_pinned_environment():
    import torchvision
    if (torch.__version__.split("+")[0] != "1.13.0" or torch.version.cuda != "11.7"
            or torchvision.__version__.split("+")[0] != "0.14.0"):
        raise RuntimeError("GPU execution requires torch 1.13.0 / CUDA 11.7 / torchvision 0.14.0; "
                           "found torch=%s cuda=%s torchvision=%s" %
                           (torch.__version__, torch.version.cuda, torchvision.__version__))


def init_evaluation(device_name):
    configure_backends()
    require_pinned_environment()
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full dataset evaluation requires GPU; no CPU fallback")
    torch.cuda.set_device(device)
    return device


def init_training(cfg):
    configure_backends()
    require_pinned_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: no CPU training fallback")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if int(os.environ.get("WORLD_SIZE", "0")) != cfg["world_size"] or local_rank < 0:
        raise RuntimeError("Use torchrun --nproc_per_node=4; do not change global batch")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    return torch.device("cuda", local_rank), control


def assert_global_finite(value, name):
    flag = (~torch.isfinite(value).all()).to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise FloatingPointError("Nonfinite " + name)


def assert_model_synchronized(model, scaler, control_group):
    """Compare a few numeric statistics, not full model/file hashes.

    This is a cheap divergence diagnostic, not proof of bitwise equality. AMP
    overflow is synchronized separately on every optimizer update.
    """
    device = next(model.parameters()).device
    stats = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.no_grad():
        for value in model.state_dict().values():
            value = value.detach().double()
            stats[0] += value.sum()
            stats[1] += value.square().sum()
            if value.numel():
                stats[2] = torch.maximum(stats[2], value.abs().max())
    state = {"parameter_stats": stats.cpu().tolist(),
             "scaler": scaler.state_dict() if scaler else None}
    states = [None] * world()
    if dist.is_initialized():
        dist.all_gather_object(states, state, group=control_group)
    else:
        states[0] = state
    if any(value != states[0] for value in states):
        raise RuntimeError("Model statistics or AMP state diverged across ranks")
    return state["parameter_stats"]


class SynchronizedScaler:
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
        if not hasattr(self.scaler, "_found_inf_per_device"):
            raise RuntimeError("Pinned GradScaler adapter interface unavailable")

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self, optimizer):
        self.scaler.unscale_(optimizer)
        devices = self.scaler._found_inf_per_device(optimizer)
        if not devices:
            raise RuntimeError("No optimizer gradients found by AMP")
        flag = torch.stack(list(devices.values())).max()
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        for value in devices.values():
            value.copy_(flag)
        skipped = bool(flag.item())
        self.scaler.step(optimizer)
        self.scaler.update()
        return skipped

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        self.scaler.load_state_dict(state)


def disk_guard(directory, checkpoint_bytes, remaining_runs=1):
    free = shutil.disk_usage(directory).free
    needed = int(remaining_runs) * 2 * int(checkpoint_bytes) + 5 * 1024**3
    if free < needed:
        raise RuntimeError("Insufficient disk: free=%d required=%d; no automatic deletion" % (free, needed))


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, str(path))
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _fsync_directory(directory):
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_alias(source, destination):
    """Publish an immutable checkpoint through best.pth without copying GBs."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=str(destination.parent))
    os.close(fd)
    try:
        os.unlink(temporary)
        try:
            os.link(str(source), temporary)
        except OSError:
            shutil.copyfile(str(source), temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, str(destination))
        _fsync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _best_name(epoch):
    return "best_epoch_%03d.pth" % int(epoch)


def resolve_best_checkpoint(directory, saved_last=None, repair_alias=False):
    """last.pth is the commit record; best.pth is only its public alias.

    A crash after writing a new best but before last commits must not select
    the new, uncommitted best. Legacy runs without a versioned best are checked
    against last's best epoch; an inconsistent old run fails explicitly.
    """
    directory = Path(directory)
    if saved_last is None and (directory / "last.pth").is_file():
        saved_last = load_training_checkpoint(directory / "last.pth")
    expected = (saved_last or {}).get("best", {}).get("epoch")
    name = (saved_last or {}).get("best_checkpoint")
    if name:
        if Path(name).name != name:
            raise ValueError("Invalid best checkpoint reference")
        path = directory / name
    else:
        path = directory / "best.pth"
    if expected is None and saved_last is not None:
        raise RuntimeError("No validation-selected checkpoint has committed yet")
    if not path.is_file():
        raise FileNotFoundError("Committed best checkpoint is missing: " + str(path))
    best = load_training_checkpoint(path)
    actual = best.get("best", {}).get("epoch")
    if actual is None or best.get("next_epoch") != actual + 1:
        raise RuntimeError("Checkpoint is not its own validation-selected best")
    if expected is not None and actual != expected:
        raise RuntimeError("last/best epoch mismatch; recover an intact checkpoint before resuming")
    if saved_last is not None:
        for key in ("experiment", "seed", "run_kind"):
            if key in saved_last and best.get(key) != saved_last[key]:
                raise RuntimeError("last/best metadata mismatch: " + key)
    del best
    if repair_alias and path.name != "best.pth":
        atomic_alias(path, directory / "best.pth")
    return path


def save_checkpoint(paths, payload, generator, control_group):
    """Crash-safe best/last commit, with no file hashes.

    Write the versioned best (and optional parity snapshot) first, then last.
    Only after last commits is best.pth updated. Keep the previous version until
    that point so an interrupted transaction can always resume its old last.
    """
    local_state = capture_rng(generator)
    gathered = [None] * world() if rank() == 0 else None
    if dist.is_initialized():
        dist.gather_object(local_state, gathered, dst=0, group=control_group)
    else:
        gathered = [local_state]
    status = [None]
    if rank() == 0:
        try:
            paths = [Path(path) for path in paths]
            saved = dict(payload, rng_by_rank=gathered, world_size=world())
            last = next((p for p in paths if p.name == "last.pth"), None)
            best_path = next((p for p in paths if p.name == "best.pth"), None)
            best_epoch = saved.get("best", {}).get("epoch")
            # Generic checkpoint tests and legacy callers still work.
            if last is None:
                for path in paths:
                    atomic_torch_save(path, saved)
            else:
                last.parent.mkdir(parents=True, exist_ok=True)
                version = last.parent / _best_name(best_epoch) if best_epoch is not None else None
                saved["best_checkpoint"] = version.name if version else None
                saved["checkpoint_format"] = 2
                if best_path is not None:
                    if version is None:
                        raise ValueError("Saving best requires a validation epoch")
                    atomic_torch_save(version, saved)
                elif version is not None and not version.exists():
                    # Migrate a consistent legacy best only; never silently pick
                    # a newer alias left by an incomplete old implementation.
                    alias = resolve_best_checkpoint(last.parent, saved_last=payload)
                    atomic_alias(alias, version)
                for path in paths:
                    if path.name not in ("last.pth", "best.pth"):
                        atomic_torch_save(path, saved)
                atomic_torch_save(last, saved)   # transaction commit point
                if version is not None:
                    atomic_alias(version, last.parent / "best.pth")
                    # Only our generated older versions, never user artifacts.
                    for old in last.parent.glob("best_epoch_*.pth"):
                        if old != version:
                            old.unlink()
            status[0] = {"ok": True}
        except Exception as exc:
            status[0] = {"ok": False, "error": repr(exc)}
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0, group=control_group)
    if not status[0]["ok"]:
        raise RuntimeError("Checkpoint transaction failed: " + status[0]["error"])


def rank0_call(function, control_group):
    if dist.is_initialized():
        dist.barrier()
    result = [None]
    if rank() == 0:
        try:
            result[0] = {"ok": True, "value": function()}
        except Exception as exc:
            result[0] = {"ok": False, "error": repr(exc)}
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0, group=control_group)
        dist.barrier()
    if not result[0]["ok"]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]
