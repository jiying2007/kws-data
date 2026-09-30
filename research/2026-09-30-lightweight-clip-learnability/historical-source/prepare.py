import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path('/workspace/shared/kws-data-pinned')
BUILD = pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback')
COMMIT = '2f9658ffa9568076ef547615c76861abed84f56e'
CATALOG = '27b590cdc325d48d2cf5e1293b8e431c815382be0219bc2e069d4e08896ee391'
FEATURE_SHA = '014f4f3f1a71a5de192a415f186d5bb6cc791dcf1d553055869ee913be7f47b8'
DATASETS = ['qwen3-xiaowo-reviewed-development-20260928',
            'qwen3-train-asr-development-20260928', 'qwen3-holdout-asr-development-20260928']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def main():
    if (ROOT / 'spec.json').exists():
        raise ValueError('refuse to overwrite preregistered spec')
    tool = BUILD / 'kws_feature_dump'
    if sha(tool) != FEATURE_SHA:
        raise ValueError('feature tool identity mismatch')
    command = [sys.executable, '-m', 'tools.codex_assets', 'export', '--expected-commit', COMMIT,
               '--expected-catalog-sha256', CATALOG]
    for name in DATASETS:
        command += ['--dataset', name]
    receipt_bytes = subprocess.check_output(command, cwd=DATA)
    (ROOT / 'native-export-receipt.json').write_bytes(receipt_bytes)
    receipt = json.loads(receipt_bytes)
    rows = []
    features = {}
    (ROOT / 'features').mkdir(exist_ok=False)
    for dataset in receipt['datasets']:
        for row in dataset['recordings']:
            # Native exporter owns corpus validation; this adapter only computes research features/labels.
            wav_path = DATA / row['path']
            if sha(wav_path) != row['file_sha256']:
                raise ValueError('WAV changed before feature extraction')
            output = subprocess.check_output([str(tool), str(wav_path), '32', 'logmel'])
            if sha(wav_path) != row['file_sha256']:
                raise ValueError('WAV changed during feature extraction')
            records = [json.loads(line) for line in output.decode().splitlines()]
            expected_frames = max(0, (row['frames'] - 400) // 320 + 1)
            if len(records) != expected_frames or [r['frame'] for r in records] != list(range(expected_frames)):
                raise ValueError('C feature frame count/index mismatch')
            values = np.asarray([r['features'] for r in records], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != 32 or not len(values) or not np.isfinite(values).all():
                raise ValueError('invalid C features')
            key = row['recording']
            raw = ROOT / 'features' / (key + '.jsonl')
            raw.write_bytes(output)
            features[key] = values
            native = dict(row, dataset_id=dataset['dataset_id'], feature_frames=len(values),
                          feature_jsonl_sha256=sha(raw))
            rows.append(native)
    np.savez(ROOT / 'features.npz', **features)
    save(ROOT / 'rows.json', rows)
    human_roles = {role: sum(r['review_method'] == 'human' and r['split'] == role for r in rows)
                   for role in ('train', 'development_a', 'development_b')}
    if human_roles != {'train': 12, 'development_a': 4, 'development_b': 4} or len(rows) != 42:
        raise ValueError('unexpected frozen dataset membership')
    spec = {
        'experiment_id': 'causal-two-word-clip-learnability-v1', 'evidence_scope': 'observed-development-only',
        'data_commit': COMMIT, 'catalog_sha256': CATALOG,
        'native_export_sha256': sha(ROOT / 'native-export-receipt.json'),
        'feature_runner_sha256': FEATURE_SHA,
        'feature_runner_source_commit': '0539106167f0bb7a657ce461a0ac5a508ecb3a22',
        'frontend': {'kind': 'logmel', 'dimensions': 32, 'rate_hz': 16000,
                     'window_samples': 400, 'hop_samples': 320, 'padding': 'none'},
        'feature_cache_sha256': sha(ROOT / 'features.npz'), 'rows_sha256': sha(ROOT / 'rows.json'),
        'architectures': {'A': {'parameters': 14882, 'mac_per_frame': 14352, 'cache_floats': 5952},
                          'B': {'parameters': 11954, 'mac_per_frame': 11712, 'cache_floats': 2880}},
        'seeds': [1337, 2346], 'optimizer': 'Adam', 'lr': .001, 'batch_size': 4,
        'steps': 1000, 'checkpoint_selection': 'last-step-only', 'per_run_wall_cap_seconds': 3600,
        'training_membership': {'review_method': 'human', 'split': 'train', 'count': 12},
        'evaluation_groups': ['human:train', 'human:development_a', 'human:development_b',
                              'asr:train', 'asr:development_a', 'asr:development_b'],
        'asr_training_allowed': False, 'augmentation': 'none',
        'objective': 'two-logit masked temporal max then binary cross entropy; native clip labels only',
        'decision': 'per-keyword max logit >= 0; no threshold calibration or event decoder',
        'training_order': 'private torch Generator(seed+10000); successive randperm train rows, batches4',
        'precision': 'float32 CPU', 'cpu_threads': 1, 'mkldnn': False, 'deterministic_algorithms': True,
        'research_python': sys.version, 'research_torch': torch.__version__, 'research_numpy': np.__version__,
        'formal_oci_equivalent': False,
        'runtime_difference': 'Python3.12.14 versus formal3.12.11; independently installed official CPU wheel, no formal OCI digest',
        'source_hashes': {name: sha(ROOT / name) for name in ('models.py', 'test_models.py', 'prepare.py', 'run.py')},
        'install_report_sha256': sha(ROOT / 'install-report.json'),
        'excludes': ['token alignment', 'word endpoint labels', 'continuous FAR', 'event-aligned FRR',
                     'fresh heldout', 'board resource evidence', 'product qualification', 'superiority claim'],
        'preregistered_revision': 2,
        'superseded_untrained_spec_sha256': '83c03621e8561c084f3d457ed7ef29d534fed90ba80f47d34a0a4f4248ac85db',
        'protocol_change_note': 'Untrained review revision: WAV pre/post hash and Cframe count/index verification; exact runtime version enforcement. No optimizer/data/architecture changes.'
    }
    save(ROOT / 'spec.json', spec)
    print(json.dumps({'rows': len(rows), 'human_roles': human_roles,
                      'feature_frames': sum(len(v) for v in features.values()),
                      'spec_sha256': sha(ROOT / 'spec.json')}, indent=2))

if __name__ == '__main__':
    main()
