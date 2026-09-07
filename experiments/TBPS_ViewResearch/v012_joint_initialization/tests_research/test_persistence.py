import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from view4.common import read_json, write_json, load_training_checkpoint
import persist_completed_run as persistence


@unittest.skipUnless(os.name == 'posix', 'Server fsync/hardlink copy transaction')
class PersistenceTests(unittest.TestCase):
    def test_complete_best_persisted_and_sources_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            ram,disk=root/'ram',root/'disk'
            ram.mkdir()
            disk.mkdir()
            source=ram/'e0_validation_run'
            (source/'checkpoints').mkdir(parents=True)
            history=[{'epoch':e,'batches':212,'validation':{'r1':60.+e,'mAP':50.+e}} for e in range(1,6)]
            best=dict(history[-1]['validation'],epoch=5)
            result=dict(status='complete',experiment='E0',seed=1,epochs=5,
                        config={'dataset':'CUHK-PEDES','input_resolution':[384,128]},
                        test_at_best_validation=None,history=history,best_validation=best)
            write_json(source/'result.json',result)
            payload=dict(model={'weight':torch.ones(2)},optimizer={},checkpoint_role='resumable',
                         next_epoch=6,next_step=0,history=history,best=best,best_checkpoint='best_epoch_005.pth')
            torch.save(payload,source/'checkpoints/last.pth')
            selected={key:value for key,value in payload.items() if key!='optimizer'}
            selected['checkpoint_role']='selection_only'
            torch.save(selected,source/'checkpoints/best.pth')
            destination=disk/'e0_validation_run'
            argv=['persist_completed_run.py','--phase','e0_validation_run','--output-dir',str(destination)]
            with patch.object(persistence,'RAM_ROOT',ram),patch.object(persistence,'DISK_ROOT',disk),patch.object(sys,'argv',argv):
                persistence.main()
                with self.assertRaises(ValueError):
                    persistence.main()
            self.assertTrue((source/'checkpoints/last.pth').exists())
            self.assertTrue((source/'checkpoints/best.pth').exists())
            self.assertFalse((destination/'checkpoints/last.pth').exists())
            self.assertTrue(torch.equal(load_training_checkpoint(destination/'checkpoints/best.pth')['model']['weight'],torch.ones(2)))
            self.assertEqual(read_json(destination/'result.json'),result)
            self.assertFalse(read_json(destination/'storage_record.json')['source_deleted'])
            self.assertEqual(os.stat(destination/'checkpoints/best.pth').st_ino,
                             os.stat(destination/'checkpoints/best_epoch_005.pth').st_ino)


if __name__=='__main__':
    unittest.main()
