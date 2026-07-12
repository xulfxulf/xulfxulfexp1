import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--light_stat_period", default=100, type=int)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")
    parser.add_argument("--irra_light", default=False, action='store_true',
                        help="enable the clean IRRA-light two-head baseline")
    parser.add_argument("--irra_light_mode", default="single_pure",
                        choices=[
                            "single_pure",
                            "single_proj_pure",
                            "split_pure",
                            "single_id",
                            "single_proj_id",
                            "split_id",
                            "single_proj_bag",
                            "split_bag",
                            "single_proj_bag_consistency",
                            "split_bag_consistency",
                            "split_bag_safe",
                            "split_bag_state",
                            "split_bag_state_hn",
                        ],
                        help=("IRRA-light first-round mode: "
                              "single_pure=A, single_proj_pure=B, split_pure=C, "
                              "single_id=D, single_proj_id=E, split_id=F, "
                              "single_proj_bag/split_bag=v16 support-bag diagnostics, "
                              "single_proj_bag_consistency/split_bag_consistency=v16 scheme-2, "
                              "split_bag_safe/state/state_hn=v16 fast3"))
    parser.add_argument("--irra_light_identity_loss", default="sdm",
                        choices=["sdm", "itc"],
                        help="identity head alignment loss; default SDM uses same-PID positives")
    parser.add_argument("--irra_light_support_size", default=3, type=int,
                        help="same-PID different-image support count for v16 support-bag modes")
    parser.add_argument("--irra_light_support_consistency_csv", default="",
                        help="image-slot consistency csv for v16 scheme-2 support reliability")
    parser.add_argument("--irra_light_support_relation_csv", default="",
                        help="query-to-support explicit hard-contradiction csv for v16 fast3")
    parser.add_argument("--irra_light_hard_negative_csv", default="",
                        help="similar different-PID image pool csv for v16 fast3")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid, TAG-PEDES]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    if args.irra_light:
        args.MLM = False
        args.sampler = "random"
        args.loss_names = "irra_light"

    return args
