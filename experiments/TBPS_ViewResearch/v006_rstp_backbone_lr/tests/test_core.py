from view4.common import load_training_checkpoint
import copy
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler

from view4.common import capture_rng, data_rng, reset_training_rng, restore_rng
from view4.config import loss_weights, lr_value
from view4.data import TrainTransform, circular_distance, flip, preprocess_text, view_of
from view4.model import build_optimizer, decorrelation, pair_mask
from view4.runtime import assert_model_synchronized, save_checkpoint
from view4.sampling import e0_plan, pk_plan
from tests.helpers import tiny_batch, tiny_model
from model.tbps_model import CLIP
from misc.data import Choose


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        reset_training_rng(1)

    def test_view_boundaries_and_double_flip(self):
        expected = {0: 2, 30: 2, 30.01: 3, 90: 3, 149.99: 3, 150: 0, 210: 0,
                    210.01: 1, 270: 1, 329.99: 1, 330: 2, 360: 2}
        for angle, view in expected.items():
            self.assertEqual(view_of(angle), view)
        for bad in (-1, 361, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                view_of(bad)
        x = torch.randn(3, 4, 7)
        for angle in range(0, 360, 5):
            y, a = flip(x, angle)
            z, b = flip(y, a)
            self.assertTrue(torch.equal(x, z))
            self.assertEqual(angle, b)
            self.assertEqual(view_of(a), {0: 0, 2: 2, 1: 3, 3: 1}[view_of(angle)])

    def test_transform_matches_official_pixels_and_rng(self):
        from PIL import Image
        from torchvision import transforms as T
        pool = [T.ColorJitter(.1, .1, .1, 0), T.RandomRotation(15),
                T.RandomResizedCrop((224, 224), (.9, 1.), antialias=True),
                T.RandomGrayscale(), T.RandomHorizontalFlip(), T.RandomErasing(scale=(.10, .20))]
        original = Choose(pool, (224, 224))
        new = TrainTransform()
        image = Image.fromarray(np.random.randint(0, 256, (60, 30, 3), dtype=np.uint8))
        for seed in range(40):
            with data_rng(seed):
                reference = original(image)
            with data_rng(seed):
                actual, _ = new(image, 90)
            self.assertTrue(torch.equal(reference, actual), seed)

    def test_text_single_word_and_rng_isolation(self):
        state = capture_rng()
        a = preprocess_text(["a person wearing red shoes", "hat"], ["red shoe pedestrian", "cap"], 2)
        self.assertEqual(tuple(a.shape), (2, 77))
        self.assertTrue(torch.equal(torch.get_rng_state(), state["torch_cpu"]))
        self.assertEqual(random.getstate(), state["python"])
        b = preprocess_text(["a person wearing red shoes", "hat"], ["red shoe pedestrian", "cap"], 2)
        self.assertTrue(torch.equal(a, b))

    def test_zero_init_head_count_and_fp32_fusion(self):
        model = tiny_model()
        h = torch.randn(4, 16)
        views = torch.arange(4)
        v, r = model.fuse(h, views)
        self.assertTrue(torch.equal(v, F.normalize(h, dim=-1)))
        self.assertTrue(torch.equal(r, torch.zeros_like(r)))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.experts.parameters()))
        d = 512
        expected = 4 * (2 * d + d * (d // 4) + d // 4 + (d // 4) * d + d)
        self.assertEqual(expected, 530944)

    @unittest.skipUnless(torch.cuda.is_available(), "torch1.13 CPU has no FP16 clamp_min; GPU gate required")
    def test_fp16_fusion_gpu(self):
        model = tiny_model().cuda()
        h = torch.randn(4, 16, device="cuda", dtype=torch.float16)
        views = torch.arange(4, device="cuda")
        v, r = model.fuse(h, views)
        self.assertEqual(v.dtype, torch.float16)
        self.assertEqual(r.dtype, torch.float32)
        self.assertTrue(torch.equal(v, F.normalize(h, dim=-1)))

    def test_pair_masks_use_raw_aug_and_distinct_image(self):
        batch = tiny_batch(8)
        self.assertEqual(int(pair_mask(batch).sum()), 4)
        batch["image_id"][1] = batch["image_id"][0]
        batch["view_aug"][3] = batch["view_aug"][2]
        batch["theta_aug"][5] = 150
        self.assertEqual(int(pair_mask(batch).sum()), 1)

    def test_decorrelation_gradient_excludes_backbone(self):
        model = tiny_model()
        for head in model.experts:
            torch.nn.init.normal_(head[-1].weight, std=.01)
        batch = tiny_batch()
        h = model.encode_image(batch["image"])
        r = model.route(h.detach(), batch["view_aug"])
        loss, stats = decorrelation(r, batch)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.visual.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encode_text.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts.parameters()))
        self.assertEqual(int(stats["ortho_pairs"]), 4)

    def test_empty_pair_zero_backward_and_zero_norm_reporting(self):
        model = tiny_model()
        batch = tiny_batch()
        r = model.route(model.encode_image(batch["image"]).detach(), batch["view_aug"])
        loss, stats = decorrelation(r, batch)
        self.assertEqual(int(stats["ortho_zero_pairs"]), 4)
        self.assertEqual(float(loss), 0.)
        batch["person_id"] = torch.arange(8)
        loss, stats = decorrelation(r, batch)
        loss.backward()
        self.assertEqual(int(stats["ortho_pairs"]), 0)
        self.assertTrue(all(p.grad is None for p in model.visual.parameters()))

    def test_optimizer_and_schedule(self):
        model = tiny_model()
        opt, audit = build_optimizer(model, lr_value(0, "E3"))
        for group, row in zip(opt.param_groups, audit):
            self.assertEqual(group["ratio"], 10 if row["name"].startswith("experts.") else 1)
            if "bias" in row["name"]:
                self.assertEqual(row["weight_decay"], 0.)
        self.assertEqual(lr_value(0, "E0"), 1e-6)
        self.assertAlmostEqual(lr_value(211, "E0"), 1e-4)
        self.assertEqual(loss_weights(1, 211, "E3")[1], 0)
        self.assertEqual(loss_weights(2, 0, "E3")[1], 0)
        self.assertEqual(loss_weights(2, 211, "E3")[1], .01)
        self.assertEqual(loss_weights(5, 211, "E2")[1], 0)

    def test_inactive_experts_and_target_pass_routing(self):
        model = tiny_model("E3")
        batch = tiny_batch(8)
        seen = [[] for _ in range(4)]
        hooks = []
        for i, head in enumerate(model.experts):
            def record(module, args, result, index=i):
                seen[index].append((len(args[0]), torch.is_grad_enabled()))
            hooks.append(head.register_forward_hook(record))
        model(batch, .3, .01)["loss_total"].backward()
        for hook in hooks:
            hook.remove()
        self.assertFalse(seen[1])
        self.assertFalse(seen[3])
        self.assertEqual(seen[0], [(4, True), (4, False), (4, True)])
        self.assertEqual(seen[2], [(4, True), (4, False), (4, True)])

    def test_loss_schedule_rejection_and_sync_statistics(self):
        model = tiny_model("E2")
        with self.assertRaises(ValueError):
            model(tiny_batch(), .5, .01)
        with self.assertRaises(ValueError):
            model(tiny_batch(), .6, 0.)
        before = assert_model_synchronized(model, None, None)
        with torch.no_grad():
            model.logit_scale.add_(.01)
        self.assertNotEqual(before, assert_model_synchronized(model, None, None))

    def test_optimizer_rng_midstep_resume(self):
        model = tiny_model("E3")
        for head in model.experts:
            torch.nn.init.normal_(head[-1].weight, std=.01)
        optimizer, _ = build_optimizer(model, 1e-4)
        def step():
            optimizer.zero_grad(set_to_none=True)
            model(tiny_batch(), .5, .01)["loss_total"].backward()
            optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            save_checkpoint([path], {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, None, None)
            step()
            expected_model = copy.deepcopy(model.state_dict())
            expected_opt = copy.deepcopy(optimizer.state_dict())
            restored = load_training_checkpoint(path, map_location="cpu")
            model.load_state_dict(restored["model"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            restore_rng(restored["rng_by_rank"][0])
            step()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]))
            for key, state in optimizer.state_dict()["state"].items():
                for field, value in state.items():
                    self.assertTrue(torch.equal(value, expected_opt["state"][key][field]))

    def test_eval_gallery_scatter_and_diagnostic_rng(self):
        from PIL import Image
        from torch.utils.data import DataLoader
        from view4.evaluation import expert_diagnostics, feature_export
        from tests.helpers import TinyVisual
        class Pixels(TinyVisual):
            def forward(self, x, return_dense=False):
                return super().forward(x.flatten(1)[:, :6], return_dense)
        model = tiny_model("E3")
        model.visual = Pixels()
        with tempfile.TemporaryDirectory() as directory:
            records = []
            for i in range(4):
                image_path = "image_%d.png" % i
                Image.fromarray(np.full((50, 25, 3), i * 60, dtype=np.uint8)).save(Path(directory) / image_path)
                records.append({"image_path": image_path, "image_id": 20 - i, "original_pid": i,
                                "theta_raw": i * 90., "view_raw": view_of(i * 90.),
                                "captions": ["a person wearing a blue shirt"]})
            state = capture_rng()
            texts, images, _ = feature_export(model, records, directory, "cpu")
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(state["torch_cpu"], torch.get_rng_state()))
            def shuffled(dataset, **kwargs):
                kwargs.pop("shuffle", None)
                return DataLoader(dataset, sampler=list(reversed(range(len(dataset)))), **kwargs)
            with patch("view4.evaluation.DataLoader", shuffled):
                texts_b, images_b, _ = feature_export(model, records, directory, "cpu")
            self.assertTrue(torch.equal(texts, texts_b))
            self.assertTrue(torch.equal(images, images_b))
            diag = expert_diagnostics(model, {"train": records}, list(range(4)), directory, "cpu")
            self.assertTrue(torch.equal(state["torch_cpu"], torch.get_rng_state()))
            for item in diag["same_input_all_heads"].values():
                self.assertIsNone(item["mean_cos2"])
                self.assertEqual(item["degenerate_fraction"], 1.)

    def test_official_forward_loss_gradient_and_updates(self):
        reference = tiny_model("E0")
        candidate = copy.deepcopy(reference)
        aopt, _ = build_optimizer(reference, 1e-4)
        bopt, _ = build_optimizer(candidate, 1e-4)
        captions = ["a person with a bag wearing blue pants"] * 8
        bt = ["the pedestrian wears blue trousers and carries a bag"] * 8
        batch = tiny_batch()
        for step, beta in enumerate((0., .5)):
            aopt.zero_grad(set_to_none=True)
            bopt.zero_grad(set_to_none=True)
            state = capture_rng()
            old_input = {"image": batch["image"], "aug1": batch["image"], "id": batch["person_id"],
                         "caption": list(captions), "caption_bt": list(bt)}
            old = CLIP.forward(reference, old_input, beta)
            (old["nitc_loss"] + old["ritc_loss"]).backward()
            a_rng = capture_rng()
            restore_rng(state)
            batch["text_tokens"] = preprocess_text(captions, bt)
            new = candidate(batch, beta, 0.)
            new["loss_total"].backward()
            torch.testing.assert_close(old["nitc_loss"], new["nitc"], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(old["ritc_loss"], new["ritc"], atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(a_rng["torch_cpu"], torch.get_rng_state()))
            for a, b in zip(reference.parameters(), candidate.parameters()):
                if a.grad is None or b.grad is None:
                    self.assertIs(a.grad, b.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
            aopt.step()
            bopt.step()
            for a, b in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_e2_e3_first_epoch_backward_and_rng_match(self):
        a = tiny_model("E2")
        b = copy.deepcopy(a)
        b.experiment = "E3"
        ao, _ = build_optimizer(a, 1e-4)
        bo, _ = build_optimizer(b, 1e-4)
        batch = tiny_batch()
        for _ in range(3):
            state = capture_rng()
            ao.zero_grad(set_to_none=True)
            a(batch, .3, 0.)["loss_total"].backward()
            arng = capture_rng()
            ao.step()
            restore_rng(state)
            bo.zero_grad(set_to_none=True)
            b(batch, .3, 0.)["loss_total"].backward()
            bo.step()
            self.assertTrue(torch.equal(arng["torch_cpu"], torch.get_rng_state()))
            for pa, pb in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(pa, pb))

    def test_sampling_e0_and_pk(self):
        planned = e0_plan(68126, 1)
        sampler = DistributedSampler(range(68126), num_replicas=4, rank=2, seed=0)
        sampler.set_epoch(0)
        self.assertEqual(planned[:, 2].ravel().tolist(), list(sampler)[:212 * 80])
        records = []
        for pid in range(12):
            for j, (angle, view) in enumerate(((0., 2), (180., 0))):
                records.append({"person_id": pid, "image_id": pid * 2 + j, "theta_raw": angle,
                                "view_raw": view, "captions": ["one", "two"]})
        plan, _ = pk_plan(records, 1, 1, steps=12, world_size=2, p=8)
        again, _ = pk_plan(records, 1, 1, steps=12, world_size=2, p=8)
        self.assertTrue(np.array_equal(plan, again))
        for step in plan:
            pids_by_rank = []
            for local in step:
                image_indices = local // 2
                pids = image_indices // 2
                pids_by_rank.append(set(pids.tolist()))
                self.assertEqual(len(set(pids.tolist())), 4)
                for pair in image_indices.reshape(-1, 2):
                    self.assertNotEqual(pair[0], pair[1])
            self.assertFalse(pids_by_rank[0] & pids_by_rank[1])

    def test_rng_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = torch.Generator().manual_seed(5)
            path = Path(directory) / "state.pth"
            save_checkpoint([path], {"marker": 1}, generator, None)
            expected = torch.rand(8)
            saved = load_training_checkpoint(path, map_location="cpu")
            restore_rng(saved["rng_by_rank"][0], generator)
            self.assertTrue(torch.equal(expected, torch.rand(8)))


if __name__ == "__main__":
    unittest.main()
