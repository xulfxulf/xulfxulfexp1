import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []
    is_light = bool(getattr(args, "irra_light", False))
    is_hire = bool(getattr(args, "hire", False))

    if is_light:
        print("IRRA-light: projection heads use base learning rate; no 5x lr factor.")
    elif is_hire:
        print(
            "HIRE: CLIP backbone uses base learning rate; newly initialized "
            "token-selection, posterior, fusion, and state modules use {}x.".format(
                args.lr_factor
            )
        )
    else:
        print("Using {} times learning rate for random init module".format(args.lr_factor))

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        random_hire_module = is_hire and not key.startswith("base_model.")
        base_lr = args.lr * (args.lr_factor if random_hire_module else 1.0)
        lr = base_lr
        weight_decay = args.weight_decay

        if is_light and (
            "identity_head" in key or "state_head" in key or "single_head" in key
        ):
            lr = args.lr
            weight_decay = args.weight_decay
        elif (not is_hire) and "cross" in key:
            lr = args.lr * args.lr_factor

        if "bias" in key:
            lr = base_lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if (not is_light) and (not is_hire) and (
            "classifier" in key or "mlm_head" in key
        ):
            lr = args.lr * args.lr_factor

        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise ValueError("unsupported optimizer: {}".format(args.optimizer))
    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
