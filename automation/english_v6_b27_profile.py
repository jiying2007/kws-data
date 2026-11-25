"""Exact, prospective B27 execution profile and complete external proof scopes.

The complete original files remain individually pinned and are reopened by the
original independent readers. No historical ZIP is repackaged inside a new
producer or source archive. This module performs no scientific operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import base64
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

HERE = Path(__file__).resolve().parent
PROFILE = HERE / 'english_v6_b27_execution_profile.json'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
DESIGN_SHA = '956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf'
Q1_LOCK_SHA = 'fa8da8d60bd7eaed2dc4dd4a179882bb09f55f65dbc70aff73ba9d8669f135ee'
Q1_REPORT_SHA = '9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f'
Q1_READER_SHA = '23d5d45b9b3630e9a561fc3ce181c9d80898e1afb33a492f917f185fac5bbe5b'
LIMITS = dict(core=128_000_000, source=256_000_000,
              primary_external_proof=768_000_000,
              replication_external_proof=1_280_000_000, source_metadata=32_000_000)
_PROOF = ContextVar('b27_complete_external_original_proof', default=None)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def document(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    need(type(value) is dict, 'explicit JSON object')
    return value


def valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def profile(require_metadata=True):
    value = document(PROFILE.read_bytes())
    need(value['schema'] == 'english-v6-b27-execution-profile-v1'
         and value['experiment_id'] == 'B27'
         and value['control_pt_sha256'] == CONTROL_PT
         and value['initial_pt_sha256'] == INITIAL_PT
         and value['control_pt_sha256'] != value['initial_pt_sha256']
         and value['training_design_sha256'] == DESIGN_SHA
         and value['limits'] == LIMITS
         and value['gains'] == ['1.0', '0.75', '0.5']
         and value['maximum_model_forwards'] == 1440
         and value['source_gate'] == dict(synthetic_positive=12, synthetic_ordinary=24,
              synthetic_paired=8, natural_originals=12, natural_speakers=8)
         and value['panels'] == ['primary', 'replication']
         and value['old_assessments_remain_terminal'] is True
         and all(value[key] is False for key in ('threshold_search', 'new_source_policy',
                                                'automatic_next_assessment_allowed')),
         'unchanged B14/source rules, new B27 TRAIN and immutable B24 comparison')
    for name, digest in value['fixed_base_sha256'].items():
        need(Path(name).name == name and valid_hash(digest)
             and sha((HERE / name).read_bytes()) == digest,
             'unchanged original mechanism ' + name)
    hashes = [value['assessment_sha256'], value['ancestry_sha256'],
              *value['panel_sha256'].values()]
    if require_metadata:
        need(all(valid_hash(item) for item in hashes), 'prospective B27 metadata not bound')
        metadata = importlib.import_module('english_v6_b27_source_metadata')
        metadata.validate()
        for name, expected in [('assessment', value['assessment_sha256']),
                               ('ancestry', value['ancestry_sha256']),
                               *value['panel_sha256'].items()]:
            need(sha((metadata.PANELS / (name + '.json')).read_bytes()) == expected,
                 'exact frozen B27 ' + name)
    else:
        need(all(item is None for item in hashes) or all(valid_hash(item) for item in hashes),
             'metadata binding is all-or-none')
    return value


def preserved(name):
    value = profile(False)
    need(name + '.py' in value['fixed_base_sha256'], 'explicit original module')
    module = importlib.import_module(name)
    need(Path(module.__file__).resolve() == HERE / (name + '.py')
         and sha(Path(module.__file__).read_bytes()) == value['fixed_base_sha256'][name + '.py'],
         'exact local original implementation')
    return module


def inherit(base, namespace):
    """Keep exact original code objects, with this explicit profile's globals."""
    preserved('english_v6_b22_replication_reuse').inherit(base, namespace)


def inventory(directory):
    root = Path(directory)
    need(root.is_absolute() and root == root.resolve() and root.is_dir()
         and all(not p.is_symlink() for p in (root, *root.parents)), 'regular resolved proof root')
    result = {}
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        need(not stat.S_ISLNK(mode) and (stat.S_ISDIR(mode) or stat.S_ISREG(mode)),
             'no symlink or special proof member')
        if path.is_dir():
            continue
        h, size = hashlib.sha256(), 0
        with path.open('rb') as handle:
            while raw := handle.read(4 * 1024 * 1024):
                h.update(raw); size += len(raw)
        result[path.relative_to(root).as_posix()] = dict(bytes=size, sha256=h.hexdigest())
    return result


def file_pins(files, maximum):
    need(type(files) is dict and files, 'complete explicit original proof file map')
    for name, pin in files.items():
        path = PurePosixPath(name)
        need(type(name) is str and name and '\\' not in name and '\0' not in name
             and not path.is_absolute() and path.as_posix() == name
             and all(part not in ('', '.', '..') for part in name.split('/')),
             'normalized original file path')
        need(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
             and type(pin['bytes']) is int and 0 <= pin['bytes'] <= maximum
             and valid_hash(pin['sha256']), 'exact original whole-file pin')
    need(sum(pin['bytes'] for pin in files.values()) <= maximum,
         'prospectively fixed complete external proof capacity')
    need(not any(name + '/' == other[:len(name) + 1] for name in files for other in files),
         'file cannot also be a directory')
    return files


def binding(panel, require_bound=True):
    need(panel in ('primary', 'replication'), 'explicit B27 panel')
    path = HERE / ('english_v6_b27_' + panel + '_proof_binding.json')
    return validate_binding(document(path.read_bytes()), panel, require_bound)


def validate_binding(value, panel, require_bound=True):
    need(panel in ('primary', 'replication'), 'explicit B27 panel')
    need(value['schema'] == 'english-v6-b27-' + panel + '-external-proof-binding-v1'
         and value['panel'] == panel and value['original_B24Q1_failure_required'] is True
         and value['old_stops_and_reserves_preserved'] is True
         and value['scientific_execution_authorized'] is False,
         'proof metadata alone never authorizes scientific execution')
    if value['status'] == 'INERT_UNTIL_COMPLETE_ORIGINAL_B27_TRAIN':
        need(all(value[key] is None for key in ('training_lock', 'training_reader_sha256',
             'training_report_sha256', 'files', 'primary_quality_lock',
             'primary_quality_reader_sha256', 'primary_quality_report_sha256', 'primary_quality_files', 'transport', 'primary_quality_transport')),
             'all original proof pins remain null before real training')
        need(not require_bound, 'actual complete original B27 TRAIN proof required')
        return value
    p = profile()
    need(value['status'] == 'BOUND_COMPLETE_ORIGINAL_TRAIN'
         and value['assessment_sha256'] == p['assessment_sha256']
         and type(value['training_lock']) is dict
         and value['training_lock']['schema'] == 'english-v6-b27-original-training-lock-v1'
         and all(valid_hash(value[key]) for key in ('training_reader_sha256', 'training_report_sha256')),
         'complete actual B27 training and prospective same assessment')
    file_pins(value['files'], LIMITS['primary_external_proof'])
    required = {'trained/TRAIN_LIVE.json', *('trained/TRAIN_' + role + '.zip'
        for role in ('models', 'producer', 'receipts', 'consumed'))}
    need(len(value['files']) == 41 and required <= set(value['files'])
         and len([name for name in value['files'] if name.startswith('qualification/no-promotion/')]) == 18
         and len([name for name in value['files'] if name.startswith('qualification/training/')]) == 18,
         'complete original TRAIN and all original qualification/training inputs')
    validate_transport(value['transport'], value['files'])
    if panel == 'primary':
        need(all(value[key] is None for key in ('primary_quality_lock', 'primary_quality_reader_sha256',
                 'primary_quality_report_sha256', 'primary_quality_files', 'primary_quality_transport')),
             'primary cannot borrow a previous assessment PASS')
    else:
        need(type(value['primary_quality_lock']) is dict
             and value['primary_quality_lock']['panel'] == 'primary'
             and value['primary_quality_lock']['schema'] == 'english-v6-b27-original-quality-lock-v1'
             and all(valid_hash(value[key]) for key in ('primary_quality_reader_sha256',
                                                        'primary_quality_report_sha256')),
             'new B27 replica stays sealed until actual complete original B27 PRIMARY PASS')
        file_pins(value['primary_quality_files'], LIMITS['replication_external_proof'])
        validate_transport(value['primary_quality_transport'], value['primary_quality_files'])
        need(sum(pin['bytes'] for pin in value['files'].values())
             + sum(pin['bytes'] for pin in value['primary_quality_files'].values())
             <= LIMITS['replication_external_proof'], 'whole replica proof bound before calls')
    return value


def validate_transport(locations, files):
    need(type(locations) is dict and set(locations) == set(files),
         'each complete external original has one exact prospective transport')
    for name, location in locations.items():
        need(type(location) is dict, 'explicit original transport')
        if name.endswith('.json'):
            need(set(location) == {'original_api_base64'} and type(location['original_api_base64']) is str,
                 'saved entire original API bytes, not an API summary')
            body = base64.b64decode(location['original_api_base64'], validate=True)
            need(len(body) == files[name]['bytes'] and sha(body) == files[name]['sha256'],
                 'exact saved original full API byte identity')
            document(body)
        else:
            need(name.endswith('.zip') and set(location) == {'artifact_id'}
                 and type(location['artifact_id']) is int and location['artifact_id'] > 0,
                 'one specifically pinned original artifact; never follow latest')
    return locations


@contextmanager
def proof_scope(directory, panel='primary'):
    from english_v6_b27_evidence_scope import scope
    value = binding(panel)
    root = Path(directory)
    need(root.is_absolute() and root == root.resolve() and not root.is_symlink(),
         'explicit regular resolved original proof directory')
    need({p.name for p in root.iterdir()} in (({'training-origin'}, {'training-origin', 'primary-quality'})
         if panel == 'primary' else ({'training-origin', 'primary-quality'},)),
         'complete scoped training root and, for replication, complete primary-quality sibling')
    need(inventory(root / 'training-origin') == value['files'],
         'all and only original externally pinned TRAIN proof bytes')
    if panel == 'replication':
        need(inventory(root / 'primary-quality') == value['primary_quality_files'],
             'all and only original independently passed PRIMARY proof bytes')
    with scope():
        token = _PROOF.set((root, panel, value))
        try:
            yield value
        finally:
            _PROOF.reset(token)


def require_proof(panel='primary'):
    context = _PROOF.get()
    need(context is not None and context[1] == panel,
         'explicit complete proof scope before source, observation or scoring')
    root, _, value = context
    need(value == binding(panel), 'same frozen proof binding throughout audit')
    reader_path = HERE / 'english_v6_b27_training_readback.py'
    need(sha(reader_path.read_bytes()) == value['training_reader_sha256'],
         'exact original B27 independent training reader')
    reader = importlib.import_module('english_v6_b27_training_readback')
    need(Path(reader.__file__).resolve() == reader_path, 'local pinned training reader')
    evidence = importlib.import_module('english_v6_b27_evidence_scope')
    # The original 41 files are a separate closed directory. A replica adds its
    # own complete primary proof as a sibling, never as a summary substitute.
    raw = evidence.verify_once('b27_complete_training', root / 'training-origin', value['files'],
        canonical(value['training_lock']), reader_path, value['training_reader_sha256'],
        value['training_report_sha256'], lambda: canonical(reader.verify_training_original(
            root / 'training-origin/trained', value['training_lock'], root / 'training-origin/qualification')))
    report = document(raw)
    result = report['result']
    need(report['schema'] == 'english-v6-b27-original-training-readback-v1'
         and result['schema'] == 'english-v6-b27-train-natural-weight-v1'
         and result['status'] == 'TRAINED_NATURAL_WEIGHT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
         and result['assessment_sha256'] == profile()['assessment_sha256']
         and result['model_pt_sha256']['control'] == CONTROL_PT
         and result['model_pt_sha256']['challenger'] not in (CONTROL_PT, INITIAL_PT)
         and all(report[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_optimizer_updates')),
         'actual B27 challenger, immutable B24 control and complete zero-call original readback')
    if panel == 'replication':
        quality_path = HERE / 'english_v6_b27_quality_readback.py'
        need(sha(quality_path.read_bytes()) == value['primary_quality_reader_sha256'],
             'exact B27 original quality reader')
        quality = importlib.import_module('english_v6_b27_quality_readback')
        raw_quality = evidence.verify_once('b27_primary_quality', root / 'primary-quality', value['primary_quality_files'],
            canonical(value['primary_quality_lock']), quality_path, value['primary_quality_reader_sha256'],
            value['primary_quality_report_sha256'], lambda: canonical(quality.verify_quality_original(
                root / 'primary-quality', value['primary_quality_lock'], root)))
        verified = document(raw_quality)
        need(verified['panel_gate'] == 'PASS' and verified['panel'] == 'primary'
             and verified['model_pt_sha256'] == result['model_pt_sha256']
             and sha(canonical(verified)) == value['primary_quality_report_sha256'],
             'only the complete same-candidate original B27 PRIMARY PASS opens replication')
        return dict(report, primary_quality=verified)
    return report


def proof_root():
    context = _PROOF.get()
    need(context is not None, 'explicit complete proof directory required')
    return context[0]
