import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from view4.common import data_rng, load_training_checkpoint
from view4.data import TrainTransform, eval_transform
from view4.geometry import INPUT_SIZE, CROP_RATIO, check_geometry
from view4.runtime import (atomic_torch_save, save_checkpoint, selection_payload,
                           resolve_best_checkpoint, disk_guard, guard_checkpoint_write)
from view4.validation import check_resume
from misc.build import resize_pos_embed
from model.visual_transformer import VisualTransformer


class PortraitTests(unittest.TestCase):
    def test_explicit_profile_and_transform_shapes(self):
        check_geometry({'input_resolution': [384,128]})
        for value in (None, [224,224], [128,384]):
            with self.assertRaises(ValueError):
                check_geometry({'input_resolution': value} if value is not None else {})
        image = Image.fromarray(np.zeros((180,60,3),dtype=np.uint8))
        self.assertEqual(tuple(eval_transform()(image).shape),(3,384,128))
        transform = TrainTransform()
        self.assertEqual(tuple(transform.pool[2].ratio), CROP_RATIO)
        for seed in range(6):
            with data_rng(seed):
                result, theta = transform(image,180.)
            self.assertEqual(tuple(result.shape),(3,384,128))
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(theta,0.)
            self.assertLess(theta,360.)

    def test_upstream_position_resize_and_dense_patch_count(self):
        original=torch.randn(197,8)
        resized=resize_pos_embed(original,torch.empty(193,8),24,8)
        self.assertEqual(tuple(resized.shape),(193,8))
        torch.testing.assert_close(resized[0],original[0],rtol=0,atol=0)
        visual=VisualTransformer(INPUT_SIZE,16,8,1,2,8,False).eval()
        with torch.no_grad():
            visual.positional_embedding.copy_(resized)
            h,dense=visual(torch.randn(2,3,384,128),return_dense=True)
        self.assertEqual(tuple(h.shape),(2,8))
        self.assertEqual(tuple(dense[:,1:].shape),(2,192,8))
        self.assertTrue(torch.isfinite(dense).all())

    def test_selection_payload_does_not_modify_resumable_payload(self):
        source={'model':{'weight':torch.ones(2)},'optimizer':{'moment':torch.zeros(2)},
                'config':{'compact_best':True},'rng_by_rank':['preserved']}
        selected=selection_payload(source)
        self.assertNotIn('optimizer',selected)
        self.assertIn('optimizer',source)
        self.assertEqual(selected['rng_by_rank'],source['rng_by_rank'])
        self.assertIs(selected['model'],source['model'])
        with self.assertRaisesRegex(ValueError,'last.pth'):
            check_resume(selected,{}, {}, {})

    def test_space_guard_has_no_automatic_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('view4.runtime.shutil.disk_usage') as usage:
                usage.return_value.free=1
                with self.assertRaises(RuntimeError):
                    disk_guard(directory,1800*1024**2,compact_best=True)
                with self.assertRaises(RuntimeError):
                    guard_checkpoint_write(Path(directory)/'last.pth',{'config':{'compact_best':True}})

    @unittest.skipUnless(os.name=='posix','Directory fsync and hardlink transaction are Linux-specific')
    def test_compact_best_and_full_last_transaction_recovers(self):
        def payload(epoch):
            return dict(model={'weight':torch.full((2,),float(epoch))},
                        optimizer={'state':{0:{'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)}}},
                        config={'compact_best':True}, experiment='test', seed=1,run_kind='audit',
                        best={'epoch':epoch,'r1':float(epoch)},next_epoch=epoch+1,next_step=0)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            paths=[root/'last.pth',root/'best.pth']
            save_checkpoint(paths,payload(1),None,None)
            self.assertIn('optimizer',load_training_checkpoint(root/'last.pth'))
            self.assertNotIn('optimizer',load_training_checkpoint(root/'best.pth'))
            def fail_last(path, data):
                if Path(path).name=='last.pth':
                    raise OSError('injected commit failure')
                atomic_torch_save(path,data)
            with patch('view4.runtime.atomic_torch_save',side_effect=fail_last),self.assertRaises(RuntimeError):
                save_checkpoint(paths,payload(2),None,None)
            self.assertEqual(resolve_best_checkpoint(root).name,'best_epoch_001.pth')
            save_checkpoint(paths,payload(2),None,None)
            self.assertEqual(resolve_best_checkpoint(root,repair_alias=True).name,'best_epoch_002.pth')
            best=load_training_checkpoint(root/'best.pth')
            last=load_training_checkpoint(root/'last.pth')
            torch.testing.assert_close(best['model']['weight'],last['model']['weight'])
            self.assertNotIn('optimizer',best)
            self.assertIn('optimizer',last)

    @unittest.skipUnless(os.name=='posix','Directory fsync is Linux-specific')
    def test_full_last_preserves_adamw_next_update(self):
        torch.manual_seed(7)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        inputs=torch.tensor([[1.,2.,3.]])
        def update(net, opt):
            opt.zero_grad(set_to_none=True)
            net(inputs).square().mean().backward()
            opt.step()
        update(model,optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                         config={'compact_best':True},experiment='test',seed=1,run_kind='audit',
                         best={'epoch':1,'r1':1.},next_epoch=2,next_step=0)
            save_checkpoint([root/'last.pth',root/'best.pth'],payload,None,None)
            saved=load_training_checkpoint(root/'last.pth')
            resumed=torch.nn.Linear(3,2)
            resumed.load_state_dict(saved['model'],strict=True)
            resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
            resumed_optimizer.load_state_dict(saved['optimizer'])
            update(model,optimizer)
            update(resumed,resumed_optimizer)
            for expected,actual in zip(model.parameters(),resumed.parameters()):
                torch.testing.assert_close(expected,actual,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
