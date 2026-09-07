"""Two CPU ranks: compare weighted global shared-loss gradients to one process."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from research.model import discriminative_shared_values, weighted_global_mean
from model.shared_modules import AllGather


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.manual_seed(73)
    x, y = torch.randn(4, 3), torch.randn(4, 3)
    initial = torch.randn(3, 6)
    ids = torch.tensor([0, 1, 0, 2])
    weights = torch.tensor([.1, .8, .4, .9])
    for count in (0, 1, 2):
        parameter = initial.clone().requires_grad_()
        h = x[rank*2:rank*2+2] @ parameter
        support = y[rank*2:rank*2+2] @ parameter
        bank = AllGather.apply(h).reshape(4, 6)
        valid = 2 if rank == 0 else count
        values = discriminative_shared_values(h[:valid], support[:valid], bank,
            ids[rank*2:rank*2+valid], ids, .07)
        loss, report = weighted_global_mean(values, weights[rank*2:rank*2+valid])
        loss.backward()
        actual = parameter.grad.clone()
        dist.all_reduce(actual)
        actual /= 2
        reference = initial.clone().requires_grad_()
        all_h, all_y = x @ reference, y @ reference
        selected = torch.tensor([0, 1] + list(range(2, 2+count)))
        expected_values = discriminative_shared_values(all_h[selected], all_y[selected], all_h,
                                                       ids[selected], ids, .07)
        expected_loss = (expected_values * weights[selected]).sum()/weights[selected].sum()
        expected_loss.backward()
        torch.testing.assert_close(actual, reference.grad, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(report, expected_loss.detach(), rtol=2e-5, atol=1e-6)
    if rank == 0:
        print('PASS: global negatives and weighted DDP gradients, including empty-support rank', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
