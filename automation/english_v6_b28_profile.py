"""Prospective B28 quality identity and complete, external original proof scopes.

The 71 immutable training originals are never nested into a new producer. Five
B24 files are recovered only in a temporary audit directory from those same
original bytes. They are derivative copies, never new qualification evidence.
This module authenticates saved evidence; it performs no scientific operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import FunctionType, ModuleType

import english_v6_b27_profile as _old_profile

HERE = Path(__file__).resolve().parent
PROFILE = HERE / 'english_v6_b28_execution_profile.json'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
B27_PT = 'dd0d4bc08be1162886ab462dfb3ad7f4503150a57ee13272d01f41e80ca1069f'
DESIGN_SHA = None  # Bound prospectively by the separate, reviewed B28 registration.
LIMITS = dict(core=128_000_000, source=256_000_000, source_metadata=32_000_000,
              primary_external_proof=1_920_000_000, replication_external_proof=2_560_000_000)
CATALOG_LIMITS = dict(archive=201_326_592, api=8_000_000, total=234_881_024)
CATALOG_PREFIX = 'inputs/metadata/train360/'
HISTORICAL_ROLE_LIMITS = {
    'inputs/proof/training-origin/qualification/no-promotion/inputs/source.zip': 768_000_000,
}
TRAIN_FILES = frozenset({'TRAIN_LIVE.json', *('TRAIN_' + role + '.zip'
    for role in ('models', 'producer', 'receipts', 'consumed'))})
QUALITY_INPUTS = frozenset(('B0', 'B9', 'pair', 'source', 'screen', 'train_producer', 'source_producer',
                          'native_qwen', 'native_whisper', 'native_parakeet', 'native_xlsr', 'native_lv60'))
QUALITY_FILES = frozenset({'api/' + name + '.json' for name in ('run', 'commit', 'artifacts')} |
    {'originals/' + name + '.zip' for name in ('result', 'producer', 'consumed')} |
    {'inputs/' + name + '.zip' for name in QUALITY_INPUTS})
CATALOG_FILES = frozenset({'api/' + name + '.json' for name in ('run', 'commit', 'artifacts', 'tree')} |
    {'originals/' + name + '.zip' for name in ('producer', 'consumed', 'metadata')})
_PROOF = ContextVar('b28_complete_external_original_proof', default=None)
_TRAIN_CHILD = ContextVar('b28_exact_temporary_training_readback', default=None)
need, sha, canonical, valid_hash = (_old_profile.need, _old_profile.sha, _old_profile.canonical, _old_profile.valid_hash)
inventory, file_pins, validate_transport = (_old_profile.inventory, _old_profile.file_pins, _old_profile.validate_transport)


def document(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        need(math.isfinite(result), 'finite JSON number')
        return result
    value = json.loads(raw, object_pairs_hook=pairs, parse_float=number,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    need(type(value) is dict, 'explicit JSON object')
    return value


def profile(require_metadata=True):
    value = document(PROFILE.read_bytes())
    need(value['schema'] == 'english-v6-b28-execution-profile-v1'
         and value['experiment_id'] == 'B28'
         and value['training_design_sha256'] == DESIGN_SHA
         and value['control_pt_sha256'] == CONTROL_PT and value['initial_pt_sha256'] == INITIAL_PT
         and value['previous_failed_b27_pt_sha256'] == B27_PT
         and len({CONTROL_PT, INITIAL_PT, B27_PT}) == 3
         and value['model_comparison'] == 'B28_new_challenger_against_immutable_historical_B24_challenger'
         and value['limits'] == LIMITS and value['catalog_transport_limits'] == CATALOG_LIMITS
         and value['historical_role_limits'] == HISTORICAL_ROLE_LIMITS
         and value['panels'] == ['primary', 'replication']
         and value['gains'] == ['1.0', '0.75', '0.5']
         and type(value['maximum_model_forwards']) is int and value['maximum_model_forwards'] == 1440
         and value['source_gate'] == dict(synthetic_positive=12, synthetic_ordinary=24,
              synthetic_paired=8, natural_originals=12, natural_speakers=8)
         and value['external_training_originals'] == 71 and value['external_replication_originals'] == 89
         and value['derived_B24_originals'] == 5 and value['derived_B24_is_new_qualification_source'] is False
         and value['old_assessments_remain_terminal'] is value['original_B27_NO_PROMOTION_required'] is True
         and all(value[key] is False for key in ('scientific_execution_authorized', 'threshold_search',
              'new_source_policy', 'automatic_next_assessment_allowed')),
         'unchanged source/B14 rules; prospectively fixed B28 originals and byte ceilings')
    bases = value['fixed_base_sha256']
    required = {'english_v6_b14_decision.py', 'english_v6_b22_replication_reuse.py',
        'english_v6_b24q1_evidence_scope.py', 'english_v6_b27_profile.py',
        'english_v6_b27_primary_eval.py', 'english_v6_b27_quality_readback.py', 'english_v6_b27_capacity.py'}
    need(type(bases) is dict and required <= set(bases), 'explicit fixed original verification mechanisms')
    for name, digest in bases.items():
        need(Path(name).name == name and valid_hash(digest) and sha((HERE / name).read_bytes()) == digest,
             'unchanged original mechanism ' + name)
    need(set(value['panel_sha256']) == {'primary', 'replication'}, 'two explicit assessment panels')
    hashes = [DESIGN_SHA, value['assessment_sha256'], value['ancestry_sha256'], *value['panel_sha256'].values()]
    need(all(item is None for item in hashes) or all(valid_hash(item) for item in hashes),
         'B28 prospective design and panel identities bind together')
    if require_metadata:
        need(all(valid_hash(item) for item in hashes), 'INERT: prospective B28 quality metadata is unbound')
        train = importlib.import_module('english_v6_b28_train')
        train.design(); train.opening()
        bound = train.assessment_binding()
        need(train.DESIGN_SHA == DESIGN_SHA and bound['assessment']['sha256'] == value['assessment_sha256'],
             'same original B28 TRAIN design and independent frozen assessment')
        metadata = importlib.import_module('english_v6_b28_source_metadata')
        checked = metadata.validate_registration()
        for name, expected in [('assessment', value['assessment_sha256']), ('ancestry', value['ancestry_sha256']),
                               *value['panel_sha256'].items()]:
            need(sha((metadata.PANELS / (name + '.json')).read_bytes()) == expected,
                 'exact prospective B28 ' + name)
        need(checked['hashes'] == bound['files'], 'same complete frozen metadata registration')
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
    preserved('english_v6_b22_replication_reuse').inherit(base, namespace)


def _evidence_module():
    base = preserved('english_v6_b24q1_evidence_scope')
    module = ModuleType('_b28_original_evidence_scope')
    inherit(base, vars(module))
    module.KINDS = frozenset(('b28_complete_training', 'b28_primary_quality'))
    module._ACTIVE = ContextVar('b28_independent_evidence_cache', default=None)
    original = base.scope.__wrapped__
    module.scope = contextmanager(FunctionType(original.__code__, vars(module), 'scope',
        original.__defaults__, original.__closure__))
    return module


evidence = _evidence_module()


def proof_binding_path(panel):
    need(panel in ('primary', 'replication'), 'explicit B28 panel')
    return HERE / ('english_v6_b28_' + panel + '_proof_binding.json')


def inert_binding(panel):
    need(panel in ('primary', 'replication'), 'explicit B28 panel')
    return dict(schema='english-v6-b28-' + panel + '-external-proof-binding-v1', panel=panel,
        status='INERT_UNTIL_COMPLETE_ORIGINAL_B28_TRAIN', assessment_sha256=None,
        original_B27_failure_required=True, old_stops_and_reserves_preserved=True,
        scientific_execution_authorized=False, **dict.fromkeys(('training_lock', 'training_reader_sha256',
        'training_report_sha256', 'files', 'transport', 'primary_quality_lock', 'primary_quality_reader_sha256',
        'primary_quality_report_sha256', 'primary_quality_files', 'primary_quality_transport'), None))


def binding(panel, require_bound=True):
    path = proof_binding_path(panel)
    # Missing future registration is an explicit inert state, never an inferred lock.
    value = document(path.read_bytes()) if path.is_file() else inert_binding(panel)
    return validate_binding(value, panel, require_bound)


def original_file_limit(name):
    need(type(name) is str, 'explicit whole original file')
    if name in HISTORICAL_ROLE_LIMITS:
        # This is an already consumed B24Q1 original (inside the old41 proof),
        # not a new B28 SOURCE. Preserve its exact original role and original
        # bound; validate_binding still requires the old immutable bytes/hash.
        original = preserved('english_v6_b24q1_quality_readback')
        need(original.archive_limit('primary', 'source') == HISTORICAL_ROLE_LIMITS[name],
             'unchanged original B24Q1 historical SOURCE capacity')
        return HISTORICAL_ROLE_LIMITS[name]
    if name.startswith(CATALOG_PREFIX):
        member = name.removeprefix(CATALOG_PREFIX)
        need(member in CATALOG_FILES, 'only the seven original registered catalog transport roles')
        return CATALOG_LIMITS['api'] if member.endswith('.json') else CATALOG_LIMITS['archive']
    if name.endswith('/source.zip') or name == 'source.zip':
        return LIMITS['source']
    need(name.endswith(('.zip', '.json')), 'only saved complete original ZIP/API objects')
    return LIMITS['source_metadata'] if name.endswith('.json') else LIMITS['core']


def proof_file_pins(files, maximum):
    file_pins(files, maximum)
    for name, pin in files.items():
        need(pin['bytes'] <= original_file_limit(name), 'prospective per-original capacity ' + name)
    need(sum(pin['bytes'] for name, pin in files.items() if name.startswith(CATALOG_PREFIX))
         <= CATALOG_LIMITS['total'], 'complete original metadata catalog7 fits its independently frozen transport ceiling')
    return files


def expected_training_inputs():
    train = importlib.import_module('english_v6_b28_train')
    previous, old = train.opening(), _old_profile.binding('primary')
    metadata = importlib.import_module('english_v6_b28_source_metadata')
    catalog = metadata.catalog_registration()
    need(len(old['files']) == 41 and set(previous['original_primary_files']) == QUALITY_FILES
         and set(catalog['files']) == CATALOG_FILES, 'all old41+18 and complete original catalog7 including the original Git tree')
    result = {'inputs/proof/training-origin/' + name: pin for name, pin in old['files'].items()}
    result.update({'inputs/proof/primary-quality/' + name: pin for name, pin in previous['original_primary_files'].items()})
    result.update({'inputs/metadata/train360/' + name: pin for name, pin in catalog['files'].items()})
    return result


def validate_binding(value, panel, require_bound=True):
    inert = inert_binding(panel)
    need(type(value) is dict and set(value) == set(inert)
         and value['schema'] == inert['schema'] and value['panel'] == panel
         and value['original_B27_failure_required'] is value['old_stops_and_reserves_preserved'] is True
         and value['scientific_execution_authorized'] is False,
         'complete B28 proof registration never grants scientific execution')
    if value['status'] == inert['status']:
        need(value == inert, 'all future original evidence pins stay null before actual TRAIN')
        need(not require_bound, 'INERT: actual complete original B28 TRAIN proof required')
        return value
    p = profile()
    reader = importlib.import_module('english_v6_b28_training_readback')
    need(value['status'] == 'BOUND_COMPLETE_ORIGINAL_TRAIN'
         and value['assessment_sha256'] == p['assessment_sha256']
         and all(valid_hash(value[key]) for key in ('training_reader_sha256', 'training_report_sha256')),
         'bound complete actual B28 training and same prospective assessment')
    training = reader.validate_lock(value['training_lock'])
    need(training['model_pt_sha256']['challenger'] not in (CONTROL_PT, INITIAL_PT, B27_PT),
         'one new B28 candidate; no prior checkpoint can substitute')
    files = proof_file_pins(value['files'], LIMITS['primary_external_proof'])
    required_inputs = expected_training_inputs()
    need(len(files) == 71 and set(files) == {'trained/' + name for name in TRAIN_FILES} | set(required_inputs)
         and all(files[name] == pin for name, pin in required_inputs.items()),
         'exact new TRAIN5 plus original B27 59 and catalog7; no summary or nested replacement')
    for role, pin in training['artifacts'].items():
        need(files['trained/TRAIN_' + role + '.zip'] == dict(bytes=pin['bytes'], sha256=pin['sha256']),
             'same whole actual new TRAIN artifact ' + role)
    validate_transport(value['transport'], files)
    if panel == 'primary':
        need(all(value[key] is None for key in ('primary_quality_lock', 'primary_quality_reader_sha256',
            'primary_quality_report_sha256', 'primary_quality_files', 'primary_quality_transport')),
            'new PRIMARY cannot borrow any earlier PASS')
    else:
        need(type(value['primary_quality_lock']) is dict
             and value['primary_quality_lock'].get('schema') == 'english-v6-b28-original-quality-lock-v1'
             and value['primary_quality_lock'].get('panel') == 'primary'
             and all(valid_hash(value[key]) for key in ('primary_quality_reader_sha256', 'primary_quality_report_sha256')),
             'replication requires one new same-B28 original PRIMARY')
        proof_file_pins(value['primary_quality_files'], LIMITS['replication_external_proof'])
        need(set(value['primary_quality_files']) == QUALITY_FILES, 'all18 complete original B28 PRIMARY files')
        validate_transport(value['primary_quality_transport'], value['primary_quality_files'])
        need(sum(pin['bytes'] for pin in files.values()) + sum(pin['bytes'] for pin in value['primary_quality_files'].values())
             <= LIMITS['replication_external_proof'], 'all89 replica evidence bytes fit the prospective ceiling')
    return value


@contextmanager
def proof_scope(directory, panel='primary'):
    value = binding(panel)
    root = Path(directory)
    need(root.is_absolute() and root == root.resolve() and root.is_dir() and not root.is_symlink(),
         'explicit regular resolved complete proof directory')
    names = {path.name for path in root.iterdir()}
    need(names in ({'training-origin'}, {'training-origin', 'primary-quality'}) if panel == 'primary'
         else names == {'training-origin', 'primary-quality'}, 'only explicit complete evidence siblings')
    need(inventory(root / 'training-origin') == value['files'], 'all and only externally pinned71 originals')
    if panel == 'replication':
        need(inventory(root / 'primary-quality') == value['primary_quality_files'], 'all18 exact original B28 PRIMARY files')
    with evidence.scope():
        token = _PROOF.set((root, panel, value))
        try:
            yield value
        finally:
            _PROOF.reset(token)


def proof_root(panel=None):
    context = _PROOF.get()
    need(context is not None, 'explicit complete B28 original proof scope required')
    need(panel is None or panel == context[1], 'same explicit active B28 panel')
    return context[0]


def training_child():
    value = _TRAIN_CHILD.get()
    need(value is not None, 'only the explicitly staged exact TRAIN reader child')
    return value


def _training_readback(root, bound, reader):
    train = importlib.import_module('english_v6_b28_train')
    originals = root / 'training-origin'
    input_pins = {name.removeprefix('inputs/'): pin for name, pin in bound['files'].items() if name.startswith('inputs/')}
    need(len(input_pins) == 66 and inventory(originals / 'inputs') == input_pins, 'complete unchanged original66 TRAIN inputs')
    with tempfile.TemporaryDirectory(prefix='b28-exact-train-inputs-') as temporary:
        staged = Path(temporary).resolve() / 'inputs'
        staged.mkdir()
        for name in input_pins:
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(originals / 'inputs' / name, path)
        need(inventory(staged) == input_pins, 'complete byte-identical temporary66 inputs')
        train.retain_original_b24(staged, create=True)
        derived = inventory(staged / 'original_b24')
        need(set(derived) == TRAIN_FILES, 'exactly five B24 originals recovered from the same B27 receipts')
        expected = {**input_pins, **{'original_b24/' + name: pin for name, pin in derived.items()}}
        need(inventory(staged) == expected, 'no supplementary qualification evidence in temporary audit')
        token = _TRAIN_CHILD.set(dict(directory=originals / 'trained', inputs=staged, files=expected))
        try:
            result = reader.verify_training_original(originals / 'trained', bound['training_lock'], staged)
            train.retain_original_b24(staged)
            need(inventory(staged) == expected, 'all external and derived original bytes unchanged through readback')
            return canonical(result)
        finally:
            _TRAIN_CHILD.reset(token)


def require_proof(panel='primary'):
    context = _PROOF.get()
    need(context is not None and context[1] == panel, 'explicit complete B28 proof scope before any source or quality work')
    root, _, bound = context
    need(bound == binding(panel), 'same prospective proof registration throughout this audit')
    path = HERE / 'english_v6_b28_training_readback.py'
    need(sha(path.read_bytes()) == bound['training_reader_sha256'], 'exact independent B28 TRAIN reader')
    reader = importlib.import_module('english_v6_b28_training_readback')
    need(Path(reader.__file__).resolve() == path, 'one pinned local B28 TRAIN reader root')
    raw = evidence.verify_once('b28_complete_training', root / 'training-origin', bound['files'],
        canonical(bound['training_lock']), path, bound['training_reader_sha256'], bound['training_report_sha256'],
        lambda: _training_readback(root, bound, reader))
    report = reader.validate_report(document(raw), bound['training_lock'])
    result = report['result']
    need(result['schema'] == 'english-v6-b28-train-natural-event-v1'
         and result['status'] == 'TRAINED_NATURAL_EVENT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
         and result['assessment_sha256'] == profile()['assessment_sha256']
         and result['model_pt_sha256']['control'] == CONTROL_PT
         and result['model_pt_sha256']['challenger'] not in (CONTROL_PT, INITIAL_PT, B27_PT)
         and report['complete_original_B27_NO_PROMOTION_verified'] is True
         and report['natural_no_event_terms_recomputed'] is True
         and report['saved_training_observations']['natural_objective_returns_verified'] == 384,
         'actual original192-update B28 candidate and complete384-term independent objective readback')
    if panel == 'replication':
        path = HERE / 'english_v6_b28_quality_readback.py'
        need(sha(path.read_bytes()) == bound['primary_quality_reader_sha256'], 'exact original B28 PRIMARY reader')
        quality = importlib.import_module('english_v6_b28_quality_readback')
        need(Path(quality.__file__).resolve() == path, 'one pinned local B28 quality reader root')
        raw_quality = evidence.verify_once('b28_primary_quality', root / 'primary-quality', bound['primary_quality_files'],
            canonical(bound['primary_quality_lock']), path, bound['primary_quality_reader_sha256'],
            bound['primary_quality_report_sha256'], lambda: canonical(quality.verify_quality_original(
                root / 'primary-quality', bound['primary_quality_lock'], root)))
        verified = quality.validate_report(document(raw_quality), bound['primary_quality_lock'])
        need(verified['panel_gate'] == 'PASS' and verified['panel'] == 'primary'
             and verified['model_pt_sha256'] == result['model_pt_sha256']
             and verified['assessment_sha256'] == result['assessment_sha256'],
             'only complete independently recomputed same-candidate B28 PRIMARY PASS opens replication')
        return dict(report, primary_quality=verified)
    return report
