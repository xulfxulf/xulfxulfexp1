import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    from view4.config import load_config
    from view4.icfg_split import write_derived
    from view4.prepare import prepare
    cfg = load_config(args.config)
    root = Path(args.output_dir)
    if (root != Path('/root/autodl-tmp/TBPS_ViewResearch_ICFG_inputs_20260907') or
            root.resolve() != root or root.exists() or
            Path(cfg['annotation_root']) != root/'annotation' or Path(cfg['prepared_dir']) != root/'prepared'):
        raise ValueError('Require the new, fixed ICFG derived-input root')
    write_derived(cfg)
    prepare(cfg, cfg['prepared_dir'], ' '.join(sys.argv))


if __name__ == '__main__':
    main()
