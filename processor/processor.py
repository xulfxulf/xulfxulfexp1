import logging
import time
import distutils.version
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


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
        f"batch_size={batch_size}, unique_ids={num_unique}, "
        f"duplicate_ids={duplicate_ids}, max_per_id={max_per_id}, "
        f"same_id_ordered_pairs={same_id_ordered_pairs}, "
        f"negative_ordered_pairs={negative_ordered_pairs}"
    )
    if image_ids is not None:
        image_ids = image_ids.detach().view(-1)
        _, image_counts = torch.unique(image_ids, return_counts=True)
        duplicate_images = int((image_counts > 1).sum().item())
        max_per_image = int(image_counts.max().item()) if batch_size else 0
        same_image_ordered_pairs = int((image_counts * (image_counts - 1)).sum().item())
        msg += (
            f", duplicate_images={duplicate_images}, "
            f"max_per_image={max_per_image}, "
            f"same_image_ordered_pairs={same_image_ordered_pairs}"
        )
    return msg


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    def to_scalar(x):
        if torch.is_tensor(x):
            return x.detach().float().item()
        return float(x)

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "identity_sdm_loss": AverageMeter(),
        "identity_itc_loss": AverageMeter(),
        "state_itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if getattr(args, 'irra_light', False) and (
                n_iter == 0 or (n_iter + 1) % args.light_stat_period == 0
            ):
                logger.info(
                    f"BatchIdentityStats Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}], "
                    + _format_identity_stats(batch['pids'], batch.get('image_ids'))
                )

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(to_scalar(total_loss), batch_size)
            meters['sdm_loss'].update(to_scalar(ret.get('sdm_loss', 0)), batch_size)
            meters['itc_loss'].update(to_scalar(ret.get('itc_loss', 0)), batch_size)
            meters['identity_sdm_loss'].update(to_scalar(ret.get('identity_sdm_loss', 0)), batch_size)
            meters['identity_itc_loss'].update(to_scalar(ret.get('identity_itc_loss', 0)), batch_size)
            meters['state_itc_loss'].update(to_scalar(ret.get('state_itc_loss', 0)), batch_size)
            meters['id_loss'].update(to_scalar(ret.get('id_loss', 0)), batch_size)
            meters['mlm_loss'].update(to_scalar(ret.get('mlm_loss', 0)), batch_size)

            meters['img_acc'].update(to_scalar(ret.get('img_acc', 0)), batch_size)
            meters['txt_acc'].update(to_scalar(ret.get('txt_acc', 0)), batch_size)
            meters['mlm_acc'].update(to_scalar(ret.get('mlm_acc', 0)), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', to_scalar(ret['temperature']), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
