import logging
import time
import distutils.version  # Required by torch 1.9's tensorboard shim.
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def _format_identity_stats(pids, image_ids=None):
    pids = pids.detach().view(-1)
    batch_size = int(pids.numel())
    unique_pids, counts = torch.unique(pids, return_counts=True)
    num_unique = int(unique_pids.numel())
    duplicate_ids = int((counts > 1).sum().item())
    max_per_id = int(counts.max().item()) if batch_size else 0
    same_id_ordered_pairs = int((counts * (counts - 1)).sum().item())
    total_ordered_pairs = batch_size * (batch_size - 1)
    negative_ordered_pairs = total_ordered_pairs - same_id_ordered_pairs
    msg = (
        "batch_size={}, unique_ids={}, duplicate_ids={}, max_per_id={}, "
        "same_id_ordered_pairs={}, negative_ordered_pairs={}".format(
            batch_size,
            num_unique,
            duplicate_ids,
            max_per_id,
            same_id_ordered_pairs,
            negative_ordered_pairs,
        )
    )
    if image_ids is not None:
        image_ids = image_ids.detach().view(-1)
        _, image_counts = torch.unique(image_ids, return_counts=True)
        duplicate_images = int((image_counts > 1).sum().item())
        max_per_image = int(image_counts.max().item()) if batch_size else 0
        same_image_ordered_pairs = int((image_counts * (image_counts - 1)).sum().item())
        msg += ", duplicate_images={}, max_per_image={}, same_image_ordered_pairs={}".format(
            duplicate_images, max_per_image, same_image_ordered_pairs
        )
    return msg


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0}
    logger = logging.getLogger("IRRA.train")
    logger.info("start training")

    def to_scalar(value):
        if torch.is_tensor(value):
            return value.detach().float().item()
        return float(value)

    meter_names = [
        "loss",
        "sdm_loss",
        "itc_loss",
        "identity_sdm_loss",
        "identity_itc_loss",
        "identity_src_loss",
        "identity_set_loss",
        "state_itc_loss",
        "state_src_loss",
        "id_loss",
        "mlm_loss",
        "support_rho_mean",
        "support_rho_zero_ratio",
        "support_rho_mid_ratio",
        "support_rho_one_ratio",
        "img_acc",
        "txt_acc",
        "mlm_acc",
        "identity_bag_loss",
        "state_nontransitive_loss",
        "support_valid_ratio",
        "support_conflict_anchor_ratio",
        "hard_negative_valid_ratio",
        # HIRE aggregated losses and diagnostics.
        "joint_tal_loss",
        "identity_posterior_loss",
        "state_hierarchical_loss",
        "identity_set_nce",
        "uncertainty_calibration",
        "state_pair_nce",
        "residual_alignment",
        "state_safety",
        "mean_image_variance",
        "mean_text_variance",
        "mean_group_heterogeneity",
        "identity_scale",
        "state_scale",
        # HIRE-v2 anchor diagnostics.  These names intentionally avoid the
        # substring 'loss'; only sdm_loss and itc_loss enter total_loss.
        "global_sdm",
        "global_itc",
        "local_sdm",
        "local_itc",
        "observation_sdm",
        "observation_itc",
        "anchor_objective",
        "observation_objective",
        "image_local_residual_norm",
        "text_local_residual_norm",
    ]
    meters = {name: AverageMeter() for name in meter_names}
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            if (
                getattr(args, "irra_light", False)
                or getattr(args, "hire", False)
                or getattr(args, "hire_v2", False)
            ) and (n_iter == 0 or (n_iter + 1) % args.light_stat_period == 0):
                logger.info(
                    "BatchIdentityStats Epoch[{}] Iteration[{}/{}], {}".format(
                        epoch,
                        n_iter + 1,
                        len(train_loader),
                        _format_identity_stats(batch["pids"], batch.get("image_ids")),
                    )
                )

            ret = model(batch)
            loss_values = [value for key, value in ret.items() if "loss" in key]
            if not loss_values:
                raise RuntimeError("model forward returned no loss entries")
            total_loss = sum(loss_values)
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    "non-finite total loss at epoch {}, iteration {}".format(
                        epoch, n_iter + 1
                    )
                )

            batch_size = batch["images"].shape[0]
            meters["loss"].update(to_scalar(total_loss), batch_size)
            for name in meter_names:
                if name == "loss":
                    continue
                if name in ret:
                    meters[name].update(to_scalar(ret[name]), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info = "Epoch[{}] Iteration[{}/{}]".format(
                    epoch, n_iter + 1, len(train_loader)
                )
                for key, meter in meters.items():
                    if meter.count > 0:
                        info += ", {}: {:.4f}".format(key, meter.avg)
                info += ", Base Lr: {:.2e}".format(scheduler.get_lr()[0])
                logger.info(info)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        if "temperature" in ret:
            tb_writer.add_scalar("temperature", to_scalar(ret["temperature"]), epoch)
        for key, meter in meters.items():
            if meter.count > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )
        if epoch % eval_period == 0 and get_rank() == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            eval_model = model.module if args.distributed else model
            top1 = evaluator.eval(eval_model.eval())
            torch.cuda.empty_cache()
            if best_top1 < top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)

    if get_rank() == 0:
        best_epoch = arguments.get("epoch", num_epoch)
        logger.info("best R1: {} at epoch {}".format(best_top1, best_epoch))


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    return evaluator.eval(model.eval())
