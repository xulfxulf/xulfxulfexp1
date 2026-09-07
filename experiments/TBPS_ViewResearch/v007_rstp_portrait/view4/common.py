import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "ba090bc7dcb3b2787e80f701fc34e19f9205cee4"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def seed32(*parts):
    return int.from_bytes(hashlib.sha256(canonical(list(parts))).digest()[:8], "little") % 2**32


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_bytes(path, canonical(value) + b"\n")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def capture_rng(generator=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "loader": generator.get_state() if generator is not None else None}


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"])


@contextlib.contextmanager
def preserve_rng(generator=None):
    state = capture_rng(generator)
    try:
        yield
    finally:
        restore_rng(state, generator)


@contextlib.contextmanager
def data_rng(seed):
    # Workers must not initialize CUDA or change the model's CUDA RNG.
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    random.seed(seed)
    np.random.seed(seed)
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


def reset_training_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def file_record(path):
    """Cheap metadata; deliberately not a file-content checksum."""
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def manifest(command, inputs, output_dir, seed=None):
    from . import __version__
    return {"started_at": utcnow(), "ended_at": None, "command": command,
            "upstream_commit": UPSTREAM_COMMIT, "implementation_version": __version__,
            "inputs": [file_record(p) for p in inputs],
            "output_dir": str(Path(output_dir).resolve()), "seed": seed,
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "gpu_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def load_training_checkpoint(path, map_location="cpu"):
    """Load a trusted local training checkpoint, including optimizer/RNG state.

    Explicit weights_only=False also supports PyTorch >=2.6 CPU regression tests.
    Training itself stays pinned to 1.13.0/CUDA 11.7. Never use untrusted .pth files.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
