from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research.model import build_model


class InitializationTests(unittest.TestCase):
    def tiny(self):
        model = nn.Module()
        model.visual = nn.Linear(2, 2)
        model.visual.freeze_conv1 = True
        model.experts = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        model.expert_queries = nn.Parameter(torch.zeros(3, 2))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def test_only_new_expert_parameters_may_be_missing(self):
        cfg = dict(seed=1, initialization='openai_clip', clip_checkpoint=str(Path('openai.pt').resolve()))
        for bad in (False, True):
            model = self.tiny()
            missing = [k for k in model.state_dict() if k.startswith('experts.') or k == 'expert_queries']
            if bad:
                missing.append('visual.weight')
            result = SimpleNamespace(missing_keys=missing, unexpected_keys=[])
            with patch('research.model.SoftViewCLIP', return_value=model), \
                    patch('research.model.install_checkpoint_compatibility'), \
                    patch('research.model.enable_visual_checkpointing'), \
                    patch('research.model.load_checkpoint', return_value=(model, result)) as loader:
                if bad:
                    with self.assertRaises(RuntimeError):
                        build_model(object(), cfg, 4, cfg['clip_checkpoint'])
                else:
                    loaded = build_model(object(), cfg, 4, cfg['clip_checkpoint'])
                    self.assertIs(loaded, model)
                    self.assertFalse(loaded.visual.freeze_conv1)
                    self.assertTrue(all(p.requires_grad for p in loaded.parameters()))
                    loader.assert_called_once()

    def test_e0_checkpoint_cannot_be_used_as_initializer(self):
        cfg = dict(seed=1, initialization='openai_clip', clip_checkpoint=str(Path('openai.pt').resolve()))
        with patch('research.model.SoftViewCLIP', return_value=self.tiny()), \
                patch('research.model.install_checkpoint_compatibility'), \
                patch('research.model.enable_visual_checkpointing'), \
                patch('research.model.load_checkpoint') as loader:
            with self.assertRaises(ValueError):
                build_model(object(), cfg, 4, str(Path('best.pth').resolve()))
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
