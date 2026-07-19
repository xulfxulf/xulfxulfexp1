import logging
import os.path as osp
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset
from .hire_v2_identity_dataset import HIREV2IdentityDataset
from .hire_v2_phrase_route_dataset import (
    HIREV2PhraseRouteDataset,
    PhraseTextDataset,
)
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .tagpedes import TAGPEDES


__factory = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
    "TAG-PEDES": TAGPEDES,
}

FAST3_MODES = {
    "split_bag_safe",
    "split_bag_state",
    "split_bag_state_hn",
}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    if aug:
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    return T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def collate(batch):
    keys = set(key for item in batch for key in item.keys())
    dict_batch = {key: [item[key] if key in item else None for item in batch] for key in keys}
    tensor_batch = {}
    for key, values in dict_batch.items():
        if isinstance(values[0], int):
            tensor_batch[key] = torch.tensor(values)
        elif torch.is_tensor(values[0]):
            tensor_batch[key] = torch.stack(values)
        else:
            raise TypeError("Unexpected data type {} for key {}".format(type(values[0]), key))
    return tensor_batch


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")
    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    def seed_worker(worker_id):
        worker_seed = (args.seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.training:
        train_transforms = build_transforms(args.img_size, args.img_aug, True)
        val_transforms = build_transforms(args.img_size, False, False)

        if args.MLM:
            train_set = ImageTextMLMDataset(
                dataset.train,
                train_transforms,
                text_length=args.text_length,
            )
        else:
            support_size = 0
            support_selection_policy = "cross_first"
            support_reliability_rule = "scheme2"
            support_image_only = False
            support_relation_csv = ""
            hard_negative_csv = ""
            hard_negative_size = 0
            support_consistency_csv = ""

            mode = getattr(args, "irra_light_mode", "")
            support_bag_modes = {
                "single_proj_bag",
                "split_bag",
                "single_proj_bag_consistency",
                "split_bag_consistency",
            } | FAST3_MODES
            if getattr(args, "irra_light", False) and mode in support_bag_modes:
                support_size = getattr(args, "irra_light_support_size", 3)
                if support_size < 1:
                    raise ValueError("support-bag modes require --irra_light_support_size >= 1")

            if mode in {"single_proj_bag_consistency", "split_bag_consistency"}:
                support_consistency_csv = getattr(args, "irra_light_support_consistency_csv", "")
                if not support_consistency_csv:
                    raise ValueError("consistency support-bag modes require a consistency csv")
            elif mode in FAST3_MODES:
                support_consistency_csv = getattr(args, "irra_light_support_consistency_csv", "")
                if not support_consistency_csv or not osp.isfile(support_consistency_csv):
                    raise ValueError("v16 fast3 modes require an existing consistency csv")
                support_selection_policy = "balanced"
                support_reliability_rule = "hard_only"
                support_image_only = True
                if mode in {"split_bag_state", "split_bag_state_hn"}:
                    support_relation_csv = getattr(args, "irra_light_support_relation_csv", "")
                    if not support_relation_csv or not osp.isfile(support_relation_csv):
                        raise ValueError("{} requires an existing support relation csv".format(mode))
                if mode == "split_bag_state_hn":
                    hard_negative_csv = getattr(args, "irra_light_hard_negative_csv", "")
                    if not hard_negative_csv or not osp.isfile(hard_negative_csv):
                        raise ValueError("split_bag_state_hn requires an existing hard-negative csv")
                    hard_negative_size = 1

            # HIRE uses same-PID, different-image observations only to infer a
            # latent identity posterior.  They are returned as image/text pairs
            # but never inserted as ordinary positive pairs by the model.
            if getattr(args, "hire", False):
                support_size = int(args.hire_support_size)
                support_selection_policy = "balanced"
                support_reliability_rule = "scheme2"
                support_image_only = False
                support_consistency_csv = ""
                support_relation_csv = ""
                hard_negative_csv = ""
                hard_negative_size = 0

            hire_v2_mode = getattr(args, "hire_v2_mode", "anchor")
            phrase_modes = {"identity_phrase_route", "identity_phrase_route_cmp"}
            if getattr(args, "hire_v2", False) and hire_v2_mode in phrase_modes:
                expected_version = (
                    "v16.7.0" if hire_v2_mode == "identity_phrase_route_cmp" else "v16.6.0"
                )
                expected_route_kind = (
                    "comparative" if hire_v2_mode == "identity_phrase_route_cmp" else "propagation"
                )
                train_set = HIREV2PhraseRouteDataset(
                    dataset.train,
                    train_transforms,
                    text_length=args.text_length,
                    support_size=int(args.hire_v2_support_size),
                    support_image_views=getattr(dataset, "train_image_views", None),
                    seed=int(args.seed),
                    phrase_label_file=args.hire_v2_phrase_train_labels,
                    expected_version=expected_version,
                    expected_route_kind=expected_route_kind,
                )
            elif (
                getattr(args, "hire_v2", False)
                and hire_v2_mode in {
                    "identity",
                    "identity_balanced",
                    "identity_state",
                    "identity_token_route",
                }
            ):
                train_set = HIREV2IdentityDataset(
                    dataset.train,
                    train_transforms,
                    text_length=args.text_length,
                    support_size=int(args.hire_v2_support_size),
                    support_image_views=getattr(dataset, "train_image_views", None),
                    seed=int(args.seed),
                )
            else:
                train_set = ImageTextDataset(
                    dataset.train,
                    train_transforms,
                    text_length=args.text_length,
                    support_size=support_size,
                    support_image_views=getattr(dataset, "train_image_views", None),
                    support_consistency_csv=support_consistency_csv,
                    support_selection_policy=support_selection_policy,
                    support_reliability_rule=support_reliability_rule,
                    support_relation_csv=support_relation_csv,
                    hard_negative_csv=hard_negative_csv,
                    hard_negative_size=hard_negative_size,
                    support_image_only=support_image_only,
                )

        if args.sampler == "identity":
            if args.distributed:
                logger.info("using ddp random identity sampler")
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True
                )
                train_loader = DataLoader(
                    train_set,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=generator,
                    collate_fn=collate,
                )
            else:
                logger.info(
                    "using random identity sampler: batch_size: {}, id: {}, instance: {}".format(
                        args.batch_size,
                        args.batch_size // args.num_instance,
                        args.num_instance,
                    )
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    sampler=RandomIdentitySampler(
                        dataset.train, args.batch_size, args.num_instance
                    ),
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=generator,
                    collate_fn=collate,
                )
        elif args.sampler == "random":
            logger.info("using random sampler")
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=generator,
                collate_fn=collate,
            )
        else:
            raise ValueError("unsupported sampler: {}".format(args.sampler))

        ds = dataset.val if args.val_dataset == "val" else dataset.test
        val_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], val_transforms)
        if (
            getattr(args, "hire_v2", False)
            and getattr(args, "hire_v2_mode", "")
            in {"identity_phrase_route", "identity_phrase_route_cmp"}
        ):
            phrase_split = "val" if args.val_dataset == "val" else "test"
            phrase_span_file = (
                args.hire_v2_phrase_val_spans
                if phrase_split == "val"
                else args.hire_v2_phrase_test_spans
            )
            val_txt_set = PhraseTextDataset(
                ds["caption_pids"],
                ds["captions"],
                phrase_span_file=phrase_span_file,
                split=phrase_split,
                text_length=args.text_length,
            )
        else:
            val_txt_set = TextDataset(
                ds["caption_pids"], ds["captions"], text_length=args.text_length
            )
        val_img_loader = DataLoader(
            val_img_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        val_txt_loader = DataLoader(
            val_txt_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_img_loader, val_txt_loader, num_classes

    if tranforms:
        test_transforms = tranforms
    else:
        test_transforms = build_transforms(args.img_size, False, False)
    ds = dataset.test
    test_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], test_transforms)
    if (
        getattr(args, "hire_v2", False)
        and getattr(args, "hire_v2_mode", "")
        in {"identity_phrase_route", "identity_phrase_route_cmp"}
    ):
        test_txt_set = PhraseTextDataset(
            ds["caption_pids"],
            ds["captions"],
            phrase_span_file=args.hire_v2_phrase_test_spans,
            split="test",
            text_length=args.text_length,
        )
    else:
        test_txt_set = TextDataset(
            ds["caption_pids"], ds["captions"], text_length=args.text_length
        )
    test_img_loader = DataLoader(
        test_img_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_img_loader, test_txt_loader, num_classes
