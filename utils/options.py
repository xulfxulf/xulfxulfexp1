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
    parser.add_argument("--val_dataset", default="test")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_ckpt_file", default="", help="resume from ...")

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--img_aug", default=False, action="store_true")

    ## cross modal transformer setting
    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--masked_token_rate", type=float, default=0.8)
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1)
    parser.add_argument("--lr_factor", type=float, default=5.0)
    parser.add_argument("--MLM", default=False, action="store_true")

    ######################## IRRA-light settings ########################
    parser.add_argument("--irra_light", default=False, action="store_true")
    parser.add_argument(
        "--irra_light_mode",
        default="single_pure",
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
    )
    parser.add_argument("--irra_light_identity_loss", default="sdm", choices=["sdm", "itc"])
    parser.add_argument("--irra_light_support_size", default=3, type=int)
    parser.add_argument("--irra_light_support_consistency_csv", default="")
    parser.add_argument("--irra_light_support_relation_csv", default="")
    parser.add_argument("--irra_light_hard_negative_csv", default="")

    ######################## HIRE settings ########################
    parser.add_argument(
        "--hire",
        default=False,
        action="store_true",
        help=(
            "enable the heterogeneity-aware hierarchical identity-state model: "
            "CLIP ViT-B/16 + RDE token selection + random-effects posterior + state residual"
        ),
    )
    parser.add_argument(
        "--hire_support_size",
        default=3,
        type=int,
        help="same-PID, different-image observations used to infer each dynamic identity posterior",
    )
    # The following defaults are inherited from the public RDE recipe or are
    # architectural dimensions.  They are exposed for reproducibility, not for
    # a grid search in the main experiment.
    parser.add_argument("--hire_select_ratio", default=0.3, type=float)
    parser.add_argument("--hire_tau", default=0.015, type=float)
    parser.add_argument("--hire_margin", default=0.1, type=float)
    parser.add_argument("--hire_tse_dim", default=1024, type=int)
    parser.add_argument("--hire_observation_dim", default=512, type=int)
    parser.add_argument("--hire_identity_dim", default=512, type=int)
    parser.add_argument("--hire_state_dim", default=512, type=int)
    parser.add_argument("--hire_eval_query_chunk", default=128, type=int)
    parser.add_argument("--hire_eval_gallery_chunk", default=512, type=int)

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default="sdm+id+mlm")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--id_loss_weight", type=float, default=1.0)

    ######################## vision transformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs="+", default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument(
        "--dataset_name",
        default="CUHK-PEDES",
        help="[CUHK-PEDES, ICFG-PEDES, RSTPReid, TAG-PEDES]",
    )
    parser.add_argument("--sampler", default="random", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest="training", default=True, action="store_false")

    args = parser.parse_args()

    if args.irra_light and args.hire:
        raise ValueError("--irra_light and --hire are mutually exclusive")
    if args.irra_light:
        args.MLM = False
        args.sampler = "random"
        args.loss_names = "irra_light"
    if args.hire:
        if args.pretrain_choice != "ViT-B/16":
            raise ValueError("The delivered HIRE main version is defined for CLIP ViT-B/16")
        if args.hire_support_size < 1:
            raise ValueError("HIRE requires --hire_support_size >= 1")
        args.MLM = False
        args.sampler = "random"
        args.loss_names = "hire"

    return args
