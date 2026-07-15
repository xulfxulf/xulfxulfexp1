#!/usr/bin/env python3
"""Lightweight HIRE implementation audit.

Run from the repository root after extracting the overlay:
    python tools/hire/audit_hire.py

This audit does not download CLIP weights and does not require a dataset.  It
validates the mathematical core, gradients, source layout, and parser exposure.
A real one-epoch end-to-end test is provided by run_hire_smoke.sh.
"""

import importlib.util
import pathlib
import subprocess
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / "model" / "hire_components.py",
    ROOT / "model" / "hire_model.py",
    ROOT / "model" / "__init__.py",
    ROOT / "datasets" / "build.py",
    ROOT / "utils" / "options.py",
    ROOT / "utils" / "metrics.py",
    ROOT / "solver" / "build.py",
    ROOT / "processor" / "processor.py",
    ROOT / "run_hire_4090_tag.sh",
]


def load_components():
    path = ROOT / "model" / "hire_components.py"
    spec = importlib.util.spec_from_file_location("hire_components_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    if missing:
        raise RuntimeError("missing HIRE files: {}".format(missing))

    for path in REQUIRED:
        if path.suffix == ".py":
            subprocess.check_call([sys.executable, "-m", "py_compile", str(path)])

    c = load_components()
    means = torch.randn(4, 3, 16, requires_grad=True)
    variances = torch.rand(4, 3, 16, requires_grad=True) + 0.2
    mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
    )
    group = c.heterogeneity_aware_posterior(means, variances, mask)
    assert torch.isfinite(group["mean"]).all()
    assert torch.isfinite(group["variance"]).all()
    assert group["valid"].tolist() == [True, True, True, False]

    query_mean = torch.randn(4, 16, requires_grad=True)
    query_var = torch.rand(4, 16, requires_grad=True) + 0.2
    scores = c.gaussian_pairwise_score(
        query_mean, query_var, group["mean"], group["variance"]
    )
    pids = torch.arange(4)
    posterior = c.symmetric_multi_positive_nce(
        scores,
        pids,
        temperature=0.02,
        row_valid=torch.ones(4, dtype=torch.bool),
        column_valid=group["valid"],
    )
    tal = c.all_negative_tal(torch.randn(4, 4, requires_grad=True), pids)
    total = posterior + tal
    assert torch.isfinite(total)
    total.backward()

    print("HIRE audit passed")
    print("repository root:", ROOT)
    print("valid random-effects rows:", group["valid"].tolist())
    print("posterior loss:", float(posterior.detach()))
    print("TAL loss:", float(tal.detach()))
    print("Next: configure DATA_ROOT and run bash run_hire_smoke.sh")


if __name__ == "__main__":
    main()
