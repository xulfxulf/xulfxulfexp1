import os.path as op
import re
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


def pre_caption(caption: str, max_words: int = 77) -> str:
    """Follow TAG-PR official caption preprocessing."""
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


class TAGPEDES(BaseDataset):
    """
    TAG-PEDES

    Reference:
    Text-based Aerial-Ground Person Retrieval (AAAI 2026)

    Official code:
    https://github.com/Flame-Chasers/TAG-PR

    Official annotation format:
    train_reid.json / test_reid.json entries contain
    {'file_path', 'id', optional 'cam_id', 'captions'}.
    """
    dataset_dir = "TAG-PEDES"

    def __init__(self, root="", verbose=True):
        super(TAGPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)

        official_anno_dir = op.join(self.dataset_dir, "anno_dir")
        official_img_dir = op.join(self.dataset_dir, "images")
        self.anno_dir = official_anno_dir if op.isdir(official_anno_dir) else self.dataset_dir
        self.img_dir = official_img_dir if op.isdir(official_img_dir) else self.dataset_dir

        self.train_anno_path = op.join(self.anno_dir, "train_reid.json")
        self.test_anno_path = op.join(self.anno_dir, "test_reid.json")
        self._check_before_run()

        self.train_annos = read_json(self.train_anno_path)
        self.test_annos = read_json(self.test_anno_path)
        # TAG-PR official release defines train/test only. Use test as validation,
        # matching the IRRA-light reproduction setting where --val_dataset=test.
        self.val_annos = self.test_annos

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> TAG-PEDES Images and Captions are loaded")
            self.show_dataset_info()

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            person_id2idx = {}
            next_pid = 0
            for anno in annos:
                person_id = int(anno["id"])
                if person_id not in person_id2idx:
                    person_id2idx[person_id] = next_pid
                    next_pid += 1
                pid = person_id2idx[person_id]
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno["file_path"])
                for caption in anno["captions"]:
                    dataset.append((pid, image_id, img_path, pre_caption(caption)))
                image_id += 1
            return dataset, pid_container

        dataset = {}
        img_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        for anno in annos:
            pid = int(anno["id"])
            pid_container.add(pid)
            img_path = op.join(self.img_dir, anno["file_path"])
            img_paths.append(img_path)
            image_pids.append(pid)
            for caption in anno["captions"]:
                captions.append(pre_caption(caption))
                caption_pids.append(pid)
        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions,
        }
        return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.anno_dir):
            raise RuntimeError("'{}' is not available".format(self.anno_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.train_anno_path):
            raise RuntimeError("'{}' is not available".format(self.train_anno_path))
        if not op.exists(self.test_anno_path):
            raise RuntimeError("'{}' is not available".format(self.test_anno_path))
