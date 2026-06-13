import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.clip_model import build_CLIP_from_openai_pretrained  # noqa: E402


OUTER_KEYS = ("state_dict", "model", "module")


def install_numpy_pickle_compat():
    # Some newer checkpoints pickle numpy as numpy._core, while older
    # environments expose the same module as numpy.core.
    sys.modules.setdefault("numpy._core", np.core)
    if hasattr(np.core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    if hasattr(np.core, "umath"):
        sys.modules.setdefault("numpy._core.umath", np.core.umath)


def load_tensor_state_dict(path):
    install_numpy_pickle_compat()
    checkpoint = torch.load(path, map_location="cpu")
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        state_dict = checkpoint.state_dict()
        tensor_state = {
            str(k): v for k, v in state_dict.items()
            if torch.is_tensor(v)
        }
        if not tensor_state:
            raise RuntimeError("TorchScript checkpoint has no tensor state_dict entries.")
        return checkpoint, tensor_state, ["<torchscript>.state_dict"]
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
        str(k): v for k, v in current.items()
        if torch.is_tensor(v)
    }
    if not tensor_state:
        raise RuntimeError(
            "No tensor state_dict entries found. Top-level keys: "
            + ", ".join(str(k) for k in list(checkpoint.keys())[:50])
        )
    return checkpoint, tensor_state, used


def prefix_counter(keys):
    counter = Counter()
    for key in keys:
        parts = key.split(".")
        for width in (1, 2, 3):
            if len(parts) >= width:
                counter[".".join(parts[:width])] += 1
    return counter


def main():
    parser = argparse.ArgumentParser(description="Inspect a SuperCLIP checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to the original SuperCLIP checkpoint.")
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--img_size", nargs=2, type=int, default=[384, 128])
    parser.add_argument("--stride_size", type=int, default=16)
    args = parser.parse_args()

    checkpoint, state_dict, used = load_tensor_state_dict(args.ckpt)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Top-level type: {type(checkpoint).__name__}")
    if isinstance(checkpoint, dict):
        print(f"Top-level keys ({len(checkpoint)}): {list(checkpoint.keys())[:50]}")
    print(f"Unwrapped outer keys: {used or '<none>'}")
    print(f"Tensor keys: {len(state_dict)}")

    print("\nFirst 50 tensor keys:")
    for key in list(state_dict.keys())[:50]:
        value = state_dict[key]
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")

    print("\nPrefix distribution:")
    for key, count in prefix_counter(state_dict.keys()).most_common(50):
        print(f"  {key}: {count}")

    clip_model, _ = build_CLIP_from_openai_pretrained(
        args.pretrain_choice,
        image_size=tuple(args.img_size),
        stride_size=args.stride_size,
    )
    target = clip_model.state_dict()
    direct = [
        key for key, value in state_dict.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    ]
    direct_name = [key for key in state_dict if key in target]
    print("\nDirect current-CLIP key match:")
    print(f"  matching names: {len(direct_name)} / {len(target)}")
    print(f"  matching names and shapes: {len(direct)} / {len(target)}")


if __name__ == "__main__":
    main()
