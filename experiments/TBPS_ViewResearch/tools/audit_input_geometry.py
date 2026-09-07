"""Measure image-header geometry on a reproducible train-only sample."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, __version__ as pillow_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    indices = random.Random(args.seed).sample(range(len(catalog['train'])),
                                             min(args.sample_size, len(catalog['train'])))
    rows = []
    root = Path(args.dataset_root).resolve()
    for index in indices:
        record = catalog['train'][index]
        path = (root / record['image_path']).resolve()
        if root not in path.parents:
            raise ValueError('Image path outside dataset root')
        with Image.open(path) as opened:
            width, height = opened.size
        if min(width, height) <= 0:
            raise ValueError('Invalid image dimensions')
        rows.append(dict(train_index=index, image_id=record['image_id'],
                         width=width, height=height, aspect_h_over_w=height/width))
    aspects = np.array([row['aspect_h_over_w'] for row in rows])
    errors = {name: np.abs(np.log(aspects / ratio)) for name, ratio in
              (('square_224x224', 1.), ('portrait_384x128', 3.))}
    report = dict(train_only=True, test_images_read=0, sample_seed=args.seed,
                  sample_size=len(rows), training_image_count=len(catalog['train']),
                  aspect_quantiles=dict(zip(('min','p10','median','p90','max'),
                                        np.quantile(aspects,[0.,.1,.5,.9,1.]).tolist())),
                  mean_absolute_log_aspect_distortion={k: float(v.mean()) for k,v in errors.items()},
                  portrait_closer_fraction=float((errors['portrait_384x128'] < errors['square_224x224']).mean()),
                  note='Header geometry only; not evidence of retrieval improvement', rows=rows)
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output/'run_manifest.json').write_text(json.dumps(dict(command=sys.argv,
                    catalog=str(Path(args.catalog).resolve()), dataset_root=str(root),
                    output_dir=str(output), python=sys.version, pillow=pillow_version,
                    seed=args.seed, gpu_used=False), indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
