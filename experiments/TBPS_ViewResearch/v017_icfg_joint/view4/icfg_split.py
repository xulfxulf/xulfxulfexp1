"""Fixed train-ID holdout; quarantine ambiguous images without editing sources."""
import collections
import copy
from pathlib import Path

import numpy as np

from .common import file_record, read_json, write_json

PROTOCOL = 'icfg_train_pid310_seed1_conflict_quarantine_v1'
CONFLICT = 'test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg'


def partition(train, test, holdout_count=310, seed=1):
    train_paths = {r['file_path'] for r in train}
    test_paths = {r['file_path'] for r in test}
    train_pids = {r['id'] for r in train}
    test_pids = {r['id'] for r in test}
    if train_paths & test_paths or train_pids & test_pids or len(test_paths) != len(test):
        raise ValueError('Official train/test identity/path overlap or duplicate test image')
    by_path = collections.defaultdict(list)
    for row in train:
        by_path[row['file_path']].append(row)
    conflicts = {p: sorted({r['id'] for r in rows}) for p, rows in by_path.items()
                 if len({r['id'] for r in rows}) > 1}
    if any(len(rows) > 1 and path not in conflicts for path, rows in by_path.items()):
        raise ValueError('Unexpected repeated nonconflicting training image')
    clean = [r for r in train if r['file_path'] not in conflicts]
    pids = sorted({r['id'] for r in clean})
    if not 0 < holdout_count < len(pids):
        raise ValueError('Invalid identity holdout size')
    val_pids = set(np.random.RandomState(seed).permutation(pids)[:holdout_count].tolist())
    splits = dict(train=[], val=[], test=[])
    for source, rows in (('train', clean), ('test', test)):
        for row in rows:
            if row['split'] != source:
                raise ValueError('Official row split disagrees with its source file')
            target = 'test' if source == 'test' else 'val' if row['id'] in val_pids else 'train'
            derived = copy.deepcopy(row)
            derived.update(split=target, source_split=source)
            splits[target].append(derived)
    report = dict(protocol=PROTOCOL, seed=seed, holdout_pid_count=holdout_count,
        val_pids=sorted(val_pids), quarantined_paths=conflicts,
        quarantined_rows=len(train)-len(clean),
        counts={s: [len(rows), len({r['id'] for r in rows}), sum(len(r['captions']) for r in rows)]
                for s, rows in splits.items()}, original_files_modified=False,
        uses_test_for_selection=False)
    return splits, report


def expected_split(cfg):
    root = Path(cfg['source_annotation_root'])
    train, val, test = [read_json(root/(s+'_reid.json')) for s in ('train','val','test')]
    if len(train) != 34674 or val or len(test) != 19848:
        raise ValueError('Unexpected official ICFG source counts')
    splits, report = partition(train, test)
    if (report['quarantined_paths'] != {CONFLICT: [1,5]} or report['quarantined_rows'] != 2 or
            report['counts'] != {'train':[31289,2792,31289], 'val':[3383,310,3383], 'test':[19848,1000,19848]}):
        raise ValueError('Source conflict or deterministic holdout no longer matches the frozen protocol')
    return splits, report


def write_derived(cfg):
    out = Path(cfg['annotation_root']).resolve()
    raw = Path(cfg['source_annotation_root']).resolve()
    dataset = Path(cfg['dataset_root']).resolve()
    if out == raw or raw in out.parents or out == dataset or dataset in out.parents or out.exists():
        raise ValueError('Derived annotations must be new and outside original data/source folders')
    splits, report = expected_split(cfg)
    out.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        write_json(out/(split+'_reid.json'), rows)
    write_json(out/'split_manifest.json', dict(report, source_files=[
        file_record(raw/(s+'_reid.json')) for s in ('train','val','test')]))


def verify_derived(cfg):
    splits, report = expected_split(cfg)
    out = Path(cfg['annotation_root'])
    manifest = read_json(out/'split_manifest.json')
    if any(manifest.get(k) != v for k,v in report.items()):
        raise ValueError('Changed holdout/quarantine protocol')
    current = [file_record(Path(cfg['source_annotation_root'])/(s+'_reid.json'))
               for s in ('train','val','test')]
    if manifest['source_files'] != current:
        raise ValueError('Official source metadata changed after preparation')
    for split, rows in splits.items():
        if read_json(out/(split+'_reid.json')) != rows:
            raise ValueError('Derived annotations differ from deterministic source partition')


def original_split(split, row):
    expected = 'test' if split == 'test' else 'train'
    if row.get('source_split') != expected:
        raise ValueError('Incorrect original-split provenance for ICFG holdout')
    return expected
