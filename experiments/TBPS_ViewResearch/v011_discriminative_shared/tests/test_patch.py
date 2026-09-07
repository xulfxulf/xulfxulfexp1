"""CPU regression tests for review fixes 1,2,3,5,7,8; no datasets/checkpoints needed."""
import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from view4 import __version__
from view4.common import read_json, write_json, load_training_checkpoint
from view4.upstream import install_checkpoint_compatibility, enable_visual_checkpointing
from view4.runtime import (atomic_torch_save, save_checkpoint, resolve_best_checkpoint,
                          configure_backends, init_training, init_evaluation, require_pinned_environment)
from view4.prepare import verify_prepared, source_records, source_paths
from view4.validation import check_resume, check_e0_checkpoint
from view4.queue import run_queue, queue_plan, continuation_command
from model.base_transformer import Transformer


def basic_cfg(root):
    return {"dataset_root": str(root / "data"), "annotation_root": str(root / "ann"),
            "orientation": str(root / "angles.json"), "clip_checkpoint": str(root / "clip.pt"),
            "prepared_dir": str(root / "prepared"), "nltk_data": str(root / "nltk"),
            "world_size": 4, "local_batch": 80, "epochs": 5, "steps_per_epoch": 212,
            "P": 160, "K": 2, "workers": 0, "alpha": .2, "delta": 60,
            "lambda_max": .01, "seeds": [1, 2, 3]}


def checkpoint_payload(epoch):
    return {"experiment": "E0", "seed": 1, "run_kind": "formal",
            "model": {"weight": torch.tensor([float(epoch)])},
            "best": {"r1": float(epoch), "epoch": epoch}, "next_epoch": epoch + 1,
            "next_step": 0}


class CheckpointActivationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        install_checkpoint_compatibility()

    def test_visual_flag_explicitly_enabled_and_text_unchanged(self):
        visual = Transformer(8, 12, 2, checkpoint=False)
        text = Transformer(8, 2, 2, checkpoint=False)
        holder = SimpleNamespace(visual=SimpleNamespace(transformer=visual), encode_text=text)
        enable_visual_checkpointing(holder)
        self.assertTrue(visual.checkpoint)
        self.assertFalse(text.checkpoint)

    def test_first_11_blocks_recompute_and_gradients_match(self):
        torch.manual_seed(18)
        ref = Transformer(8, 12, 2, dropout=.05, checkpoint=False).train()
        actual = copy.deepcopy(ref)
        actual.checkpoint = True
        counters = [0] * 12
        handles = []
        for i, block in enumerate(actual.resblocks):
            def count(module, args, index=i):
                counters[index] += 1
            handles.append(block.register_forward_pre_hook(count))
        x1 = torch.randn(3, 2, 8, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        rng = torch.get_rng_state()
        y1 = ref(x1); y1.square().mean().backward()
        torch.set_rng_state(rng)
        y2 = actual(x2); y2.square().mean().backward()
        torch.testing.assert_close(y1, y2)
        torch.testing.assert_close(x1.grad, x2.grad)
        for a, b in zip(ref.parameters(), actual.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        self.assertEqual(counters[:11], [2] * 11)
        self.assertEqual(counters[-1], 1)
        for h in handles:
            h.remove()

    def test_no_grad_and_eval_do_not_checkpoint(self):
        net = Transformer(8, 12, 2, checkpoint=True)
        x = torch.randn(3, 2, 8)
        with patch("view4.upstream.checkpoint", side_effect=AssertionError("must not checkpoint")):
            net.train()
            with torch.no_grad():
                net(x)
            net.eval()
            net(x.requires_grad_(True)).sum().backward()

    def test_wrong_visual_depth_rejected(self):
        holder = SimpleNamespace(visual=SimpleNamespace(transformer=Transformer(8, 2, 2)))
        with self.assertRaises(ValueError):
            enable_visual_checkpointing(holder)


class TransactionTests(unittest.TestCase):
    def test_best_last_commit_and_mid_epoch_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(1), None, None)
            self.assertEqual(load_training_checkpoint(d / "last.pth")["best_checkpoint"], "best_epoch_001.pth")
            self.assertEqual(resolve_best_checkpoint(d).name, "best_epoch_001.pth")
            payload = checkpoint_payload(1)
            payload.update(next_epoch=2, next_step=50)
            save_checkpoint([d / "last.pth"], payload, None, None)
            self.assertEqual(load_training_checkpoint(d / "last.pth")["next_step"], 50)
            self.assertEqual(load_training_checkpoint(d / "best.pth")["next_step"], 0)

    def test_failed_best_write_preserves_old_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(1), None, None)
            def fail(path, data):
                if Path(path).name == "best_epoch_002.pth":
                    raise OSError("disk-full injected")
                atomic_torch_save(path, data)
            with patch("view4.runtime.atomic_torch_save", side_effect=fail), self.assertRaises(RuntimeError):
                save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(2), None, None)
            self.assertEqual(load_training_checkpoint(d / "last.pth")["best"]["epoch"], 1)
            self.assertEqual(resolve_best_checkpoint(d).name, "best_epoch_001.pth")

    def test_failure_between_best_and_last_retries_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(1), None, None)
            def fail(path, data):
                if Path(path).name == "last.pth":
                    raise OSError("commit failure injected")
                atomic_torch_save(path, data)
            with patch("view4.runtime.atomic_torch_save", side_effect=fail), self.assertRaises(RuntimeError):
                save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(2), None, None)
            self.assertTrue((d / "best_epoch_002.pth").exists())
            self.assertEqual(resolve_best_checkpoint(d, repair_alias=True).name, "best_epoch_001.pth")
            save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(2), None, None)
            self.assertEqual(resolve_best_checkpoint(d).name, "best_epoch_002.pth")
            self.assertFalse((d / "best_epoch_001.pth").exists())

    def test_alias_failure_after_commit_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(1), None, None)
            with patch("view4.runtime.atomic_alias", side_effect=OSError("alias failure")), self.assertRaises(RuntimeError):
                save_checkpoint([d / "last.pth", d / "best.pth"], checkpoint_payload(2), None, None)
            self.assertEqual(load_training_checkpoint(d / "last.pth")["best"]["epoch"], 2)
            self.assertEqual(load_training_checkpoint(d / "best.pth")["best"]["epoch"], 1)
            resolve_best_checkpoint(d, repair_alias=True)
            self.assertEqual(load_training_checkpoint(d / "best.pth")["best"]["epoch"], 2)

    def test_legacy_consistent_best_migrates(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            payload = checkpoint_payload(1)
            atomic_torch_save(d / "best.pth", payload)
            atomic_torch_save(d / "last.pth", payload)
            self.assertEqual(resolve_best_checkpoint(d).name, "best.pth")
            save_checkpoint([d / "last.pth"], payload, None, None)
            self.assertEqual(resolve_best_checkpoint(d).name, "best_epoch_001.pth")

    def test_legacy_missing_best_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            atomic_torch_save(d / "last.pth", checkpoint_payload(1))
            with self.assertRaises(FileNotFoundError):
                resolve_best_checkpoint(d)

    def test_legacy_inconsistent_best_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            atomic_torch_save(d / "last.pth", checkpoint_payload(2))
            atomic_torch_save(d / "best.pth", checkpoint_payload(1))
            with self.assertRaises(RuntimeError):
                resolve_best_checkpoint(d)

    def test_generic_checkpoint_rng_roundtrip(self):
        from view4.common import restore_rng
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "state.pth"
            save_checkpoint([p], {"value": 123}, None, None)
            expected = torch.rand(3)
            state = load_training_checkpoint(p)
            restore_rng(state["rng_by_rank"][0])
            torch.testing.assert_close(expected, torch.rand(3), atol=0, rtol=0)


def make_prepared_fixture(root, legacy=False):
    cfg = basic_cfg(root)
    cfg.update(world_size=1, local_batch=2, seeds=[1], epochs=1, steps_per_epoch=1)
    data, ann, prepared = [Path(cfg[k]) for k in ("dataset_root", "annotation_root", "prepared_dir")]
    (data / "imgs").mkdir(parents=True)
    ann.mkdir(); prepared.mkdir()
    catalog, angles, counts = {}, [], {}
    iid = 0
    for split in ("train", "val", "test"):
        annotations, rows = [], []
        for j in range(2):
            name = "%s%d.jpg" % (split, j)
            (data / "imgs" / name).write_bytes(b"fixture")
            pid = 10 * (("train", "val", "test").index(split) + 1)
            captions = ["a person"]
            bt = ["a pedestrian"] if split == "train" else []
            record = {"file_path": name, "id": pid, "captions": captions, "captions_bt": bt}
            annotations.append(record)
            theta, view = (0., 2) if j == 0 else (180., 0)
            rows.append({"image_path": "imgs/" + name, "original_pid": pid, "person_id": 0,
                         "captions": captions, "captions_bt": bt, "theta_raw": theta, "view_raw": view,
                         "image_id": iid})
            angles.append({"image_path": "imgs/" + name, "body_orientation_degrees": theta,
                           "status": "ok", "pids": [pid], "splits": [split]})
            iid += 1
        catalog[split] = rows
        counts[split] = (2, 1, 2)
        write_json(ann / (split + "_reid.json"), annotations)
    ordered = {r["image_path"]: i for i, r in enumerate(sorted(sum(catalog.values(), []), key=lambda r:r["image_path"]))}
    for rows in catalog.values():
        for row in rows:
            row["image_id"] = ordered[row["image_path"]]
    write_json(prepared / "catalog.json", catalog)
    write_json(prepared / "expert_diagnostic_subset.json", [0, 1])
    write_json(cfg["orientation"], {"images": angles})
    Path(cfg["clip_checkpoint"]).write_bytes(b"dummy-trusted-weight")
    plan_folder = prepared / "plans/seed_1"
    plan_folder.mkdir(parents=True)
    for kind in ("E0", "PK"):
        np.savez_compressed(str(plan_folder / (kind + "_epoch_1.npz")), pair_indices=np.array([[[0, 1]]]))
    if legacy:
        write_json(prepared / "run_manifest.json", {"inputs": {p:"ignored legacy hash" for p in source_paths(cfg).values()}})
    else:
        write_json(prepared / "prepared_sources.json", source_records(cfg))
    return cfg, counts


class SourceValidationTests(unittest.TestCase):
    def test_current_sources_and_catalog_pass_without_image_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp))
            with patch("view4.prepare.EXPECTED", counts), patch("view4.common.hashlib.sha256", side_effect=AssertionError("no hash")):
                self.assertEqual(verify_prepared(cfg), source_records(cfg))

    def test_legacy_prepared_paths_supported_without_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp), legacy=True)
            with patch("view4.prepare.EXPECTED", counts):
                verify_prepared(cfg)

    def test_config_swapped_weight_rejected_even_old_weight_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp), legacy=True)
            alt = Path(tmp) / "other.pt"; alt.write_bytes(b"different")
            cfg["clip_checkpoint"] = str(alt)
            with patch("view4.prepare.EXPECTED", counts), self.assertRaises(RuntimeError):
                verify_prepared(cfg)

    def test_new_source_file_metadata_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp))
            Path(cfg["clip_checkpoint"]).write_bytes(b"new content")
            with patch("view4.prepare.EXPECTED", counts), self.assertRaises(RuntimeError):
                verify_prepared(cfg)

    def test_legacy_changed_angle_detected_semantically(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp), legacy=True)
            data = read_json(cfg["orientation"])
            data["images"][0]["body_orientation_degrees"] = 100.
            write_json(cfg["orientation"], data)
            with patch("view4.prepare.EXPECTED", counts), self.assertRaises(RuntimeError):
                verify_prepared(cfg)

    def test_missing_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, counts = make_prepared_fixture(Path(tmp))
            (Path(cfg["dataset_root"]) / "imgs/train0.jpg").unlink()
            with patch("view4.prepare.EXPECTED", counts), self.assertRaises(ValueError):
                verify_prepared(cfg)

    def test_e0_reusable_after_code_bugfix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _ = make_prepared_fixture(root)
            cfg["epochs"] = 5
            d = root / "E0/checkpoints"; d.mkdir(parents=True)
            saved = dict(checkpoint_payload(2), config=cfg, code_sha256="OLD-CODE",
                         input_sources=source_records(cfg))
            atomic_torch_save(d / "best.pth", saved)
            write_json(d.parent / "result.json", {"experiment":"E0", "seed":1, "run_kind":"formal",
                       "status":"complete", "epochs":5, "best_validation":{"epoch":2}, "code_sha256":"OLD-CODE"})
            result = check_e0_checkpoint(d / "best.pth", cfg, source_records(cfg))
            self.assertEqual(result["best_epoch"], 2)

    def test_ten_epoch_e0_is_reusable_but_duration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _ = make_prepared_fixture(root)
            cfg["epochs"] = 10
            d = root / "E0/checkpoints"; d.mkdir(parents=True)
            saved = dict(checkpoint_payload(2), config=cfg, input_sources=source_records(cfg))
            atomic_torch_save(d / "best.pth", saved)
            write_json(d.parent / "result.json", {"experiment":"E0", "seed":1, "run_kind":"formal",
                       "status":"complete", "epochs":10, "best_validation":{"epoch":2}})
            target = dict(cfg, epochs=5)
            result = check_e0_checkpoint(d / "best.pth", target, source_records(target))
            self.assertEqual(result["source_epochs"], 10)
            saved["config"]["epochs"] = 5
            atomic_torch_save(d / "best.pth", saved)
            with self.assertRaises(ValueError):
                check_e0_checkpoint(d / "best.pth", target, source_records(target))

    def test_resume_ignores_code_hash_but_rejects_training_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = basic_cfg(Path(tmp))
            identity = {"experiment":"E3", "seed":1, "run_kind":"formal", "world_size":4}
            saved = dict(identity, config=cfg, code_sha256="old-version", input_sources={"source":"same"})
            check_resume(saved, dict(identity, implementation_version="new-version"), cfg, {"source":"same"})
            changed = dict(cfg, alpha=.3)
            with self.assertRaises(ValueError):
                check_resume(saved, identity, changed, {"source":"same"})


class BackendTests(unittest.TestCase):
    def test_shared_configuration_disables_tf32(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        configure_backends()
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_train_eval_call_identical_backend_configuration(self):
        with patch("view4.runtime.configure_backends") as shared, patch("view4.runtime.require_pinned_environment"), \
             patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_device"), \
             patch("torch.distributed.init_process_group"), patch("torch.distributed.new_group", return_value="control"), \
             patch.dict(os.environ, {"LOCAL_RANK":"0", "WORLD_SIZE":"4"}):
            init_training({"world_size":4})
            init_evaluation("cuda:0")
            self.assertEqual(shared.call_count, 2)

    def test_wrong_target_environment_rejected(self):
        with patch("torch.__version__", "2.10.0+cpu"), self.assertRaises(RuntimeError):
            require_pinned_environment()


class QueueTests(unittest.TestCase):
    def fake_process(self, cmd, **kwargs):
        if "--output-dir" not in cmd:
            raise AssertionError(cmd)
        folder = Path(cmd[cmd.index("--output-dir") + 1]); folder.mkdir(parents=True, exist_ok=True)
        if "train" in cmd:
            exp = cmd[cmd.index("--experiment") + 1]
            seed = int(cmd[cmd.index("--seed") + 1]); kind = cmd[cmd.index("--run-kind") + 1]
            (folder / "checkpoints").mkdir(exist_ok=True)
            (folder / "checkpoints/best.pth").write_bytes(b"mocked trainer output")
            write_json(folder / "result.json", {"status":"complete", "experiment":exp, "seed":seed,
                       "run_kind":kind, "epochs":5 if kind=="formal" else 2,
                       "best_validation":{"epoch":1,"r1":70.},
                       "test_at_best_validation":{k:70. for k in ("r1","r5","r10","mAP")}})
            if exp == "E3" and kind == "smoke":
                write_json(folder.parent / "epoch1_parity_gate/parity.json", {"status":"passed", "implementation_version": __version__})
        elif "tests.distributed_checks" in cmd:
            write_json(folder / "distributed_result.json", {"status":"passed", "implementation_version": __version__})
            write_json(folder / "run_manifest.json", {"status":"passed", "implementation_version": __version__})
        else:
            suite = cmd[cmd.index("--suite")+1]
            record = "reference_validation.json" if suite=="reference-val" else "run_manifest.json"
            write_json(folder / record, {"status":"passed", "implementation_version": __version__})
        return subprocess.CompletedProcess(cmd, 0)

    def test_pause_on_gate_failure_then_resume_without_approval_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=basic_cfg(root); out=root/"queue"
            with patch("view4.queue.ROOT", root), patch("view4.queue.disk_guard"), patch("view4.queue.check_other_training"):
                with patch("view4.queue.subprocess.run", side_effect=subprocess.CalledProcessError(1,["gate"])), self.assertRaises(subprocess.CalledProcessError):
                    run_queue(cfg, root/"config.yaml", out, execute=True)
                self.assertEqual(read_json(out/"queue_status.json")["status"], "paused_failed")
                self.assertEqual(read_json(out/"queue_status.json")["current"], "gpu_equivalence")
                with patch("view4.queue.subprocess.run", side_effect=self.fake_process) as calls:
                    run_queue(cfg, root/"config.yaml", out, execute=True, resume=True)
                    self.assertEqual(sum("train" in c.args[0] for c in calls.call_args_list), 13)
                self.assertEqual(read_json(out/"queue_status.json")["status"], "complete")
                with patch("view4.queue.subprocess.run", side_effect=AssertionError("completed queue should not launch")):
                    run_queue(cfg, root/"config.yaml", out, execute=True, resume=True)

    def test_failed_run_with_last_resumes_and_without_last_is_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); item=queue_plan(root/"cfg", root/"q")[4]
            out=root/"q"; folder=out/item["name"]
            (folder/"checkpoints").mkdir(parents=True)
            (folder/"checkpoints/last.pth").write_bytes(b"state")
            cmd=continuation_command(item,out)
            self.assertIn("--resume",cmd)
            (folder/"checkpoints/last.pth").unlink()
            (folder/"failed.log").write_text("preserve")
            cmd=continuation_command(item,out)
            self.assertNotIn("--resume",cmd)
            self.assertFalse(folder.exists())
            self.assertEqual(len(list((out/"failed_attempts").rglob("failed.log"))),1)

    def test_resume_rejects_changed_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=basic_cfg(root); out=root/"q"
            with patch("view4.queue.ROOT",root):
                run_queue(cfg, root/"cfg.yaml",out)
                with self.assertRaises(ValueError):
                    run_queue(dict(cfg, alpha=.3),root/"cfg.yaml",out,resume=True)

    def test_planned_queue_can_execute_via_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=basic_cfg(root); out=root/"q"
            with patch("view4.queue.ROOT",root), patch("view4.queue.disk_guard"), patch("view4.queue.check_other_training"), \
                 patch("view4.queue.subprocess.run",side_effect=self.fake_process):
                run_queue(cfg,root/"cfg.yaml",out)
                run_queue(cfg,root/"cfg.yaml",out,execute=True,resume=True)
                self.assertEqual(read_json(out/"queue_status.json")["status"],"complete")


if __name__ == "__main__":
    unittest.main()
