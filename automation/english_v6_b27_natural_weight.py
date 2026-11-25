"""One sealed B27 phase: the previously frozen natural weight 0.08 -> 0.16.

The B23 numerical fit and B24 export wrapper are reused by code identity with
private globals. Original B24 is an immutable historical control; B23 remains
the initializer. Complete original Q1 failure, prospectively frozen fresh
assessment, complete producer capture and a durable once marker precede fit.
Neither this module nor its design is an execution permit by itself.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import FunctionType, SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for _name, _digest in (
    ('english_v6_b24_b23_warm_start.py', '6259ae829a32655822350caee2608c9d162fdadc6366a3fa6ecf3fb6ab452cd7'),
    ('english_v6_b24q1_primary_failure.py', '23d5d45b9b3630e9a561fc3ce181c9d80898e1afb33a492f917f185fac5bbe5b'),
):
    if hashlib.sha256((HERE / _name).read_bytes()).hexdigest() != _digest:
        raise ValueError('exact frozen dependency before import: ' + _name)
import english_v6_b24_b23_warm_start as b24
import english_v6_b24q1_primary_failure as failure
import english_v6_b24q1_capacity as capacity
import english_v6_b24_source_ancestry as old_ancestry
import english_v6_b24_source_training_proof as old_proof

DESIGN = HERE / 'english_v6_b27_natural_weight_design.json'
DESIGN_SHA = '956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf'
BINDING = HERE / 'english_v6_b27_assessment_binding.json'
OPENING = HERE / 'english_v6_b27_opening.json'
PROPOSAL_SHA = '4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
NATURAL_WEIGHT = 0.16
STEM = 'mero43-b27-challenger192'
SCHEMA = 'english-v6-b27-train-natural-weight-v1'
CONDITION = 'only_after_actual_complete_B24Q1_primary_NO_PROMOTION'
REPO = b24.REPO
BRANCH = 'refs/heads/research/en-v6-b27-natural-weight-train-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-natural-weight-train-once.yml'
MARKER = 'english-v6-b27-natural-weight-train-consumed'
RESULT_STATUS = 'TRAINED_NATURAL_WEIGHT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
RECEIPT_RESERVE = 8_000_000
need, sha, encoded, write, gh = b24.need, b24.sha, b24.encoded, b24.write, b24.gh


def design():
    need(sha(DESIGN.read_bytes()) == DESIGN_SHA, 'exact preoptimizer B27 single-variable design')
    value = failure.document(DESIGN.read_bytes())
    for key in ('baseline_recipe', 'baseline_fit', 'baseline_wrapper', 'proposal', 'evidence',
                'bounded_proposal', 'original_training_binding', 'original_training_producer',
                'failure_reader', 'base_quality_reader', 'original_B24_quality_function'):
        pin = value[key]
        path = ROOT / pin['path']
        need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == pin['sha256'],
             'unchanged complete scientific dependency ' + key)
    before = b24.design()
    proposal = failure.document((ROOT / value['proposal']['path']).read_bytes())
    bounded = failure.document((ROOT / value['bounded_proposal']['path']).read_bytes())
    need(value['proposal']['sha256'] == PROPOSAL_SHA
         and proposal['hypothesis']['scientific_variable'] == 'NATURAL_WEIGHT'
         and proposal['hypothesis']['old_value'] == before['natural_weight'] == b24.old.NATURAL_WEIGHT == 0.08
         and proposal['hypothesis']['new_value'] == NATURAL_WEIGHT == 0.16
         and value['scientific_change'] == bounded['hypothesis']['scientific_change']
             == dict(field='natural_weight', **{'global': 'NATURAL_WEIGHT'}, before=0.08, after=0.16)
         and bounded['hypothesis']['prior_definition']['sha256'] == PROPOSAL_SHA,
         'one already frozen TRAIN-only scalar; no use of Q1 metrics to select parameters')
    need(value['initial_pt_sha256'] == before['parent_pt_sha256'] == INITIAL_PT
         and value['historical_control_pt_sha256'] == CONTROL_PT
         and value['opening_condition'] == CONDITION and value['candidate_stem'] == STEM
         and value['optimizer_updates'] == before['updates'] == 192
         and value['maximum_new_model_forwards'] == before['budget']['total_new_model_forwards'] == 1847
         and value['budget'] == before['budget'] and value['input_archives'] == before['input_archives']
         and value['common_draw_sha256'] == before['common_draw_sha256']
         and before['union_retention_conditions'] == 122 and before['gain_consistency_weight'] == 0.03
         and before['positive_retention_weight'] == before['ordinary_ceiling_weight'] == 0.08
         and value['new_training_audio'] == value['new_teacher_forwards'] == value['new_control_forwards'] == 0
         and all(value[key] is False for key in ('exact_optimizer_continuation', 'historical_numerical_replay_claimed',
             'old_B25_condition_changed', 'old_B26_condition_changed', 'quality_thresholds_changed',
             'scientific_execution_authorized', 'Q1_assessment_or_budget_changed',
             'Q1_metrics_used_to_choose_parameters', 'parameter_specific_user_approval_claimed',
             'artifact_is_execution_authority')),
         'same initializer, 88 TRAIN originals, teachers, draws, budget and original quality rules')
    return value


def scientific_recipe():
    design()
    value = copy.deepcopy(b24.design())
    value['natural_weight'] = NATURAL_WEIGHT
    return value


def assessment_binding(require_frozen=True):
    value = failure.document(BINDING.read_bytes())
    fields = {'schema', 'recipe_path', 'recipe_sha256', 'frozen', 'optimizer_allowed',
              'assessment', 'files', 'original_B24Q1_source_exclusion'}
    need(set(value) == fields and value['schema'] == 'english-v6-b27-prospective-assessment-binding-v1'
         and value['recipe_path'] == DESIGN.relative_to(ROOT).as_posix() and value['recipe_sha256'] == DESIGN_SHA
         and value['optimizer_allowed'] is False, 'metadata freeze is separate from any execution permission')
    if value['frozen'] is False:
        need(value['assessment'] is value['original_B24Q1_source_exclusion'] is None and value['files'] == {},
             'inert binding contains no invented fresh people or source history')
        need(not require_frozen, 'fresh B27 assessment and complete Q1 exclusions remain sealed')
        return value
    need(value['frozen'] is True and type(value['files']) is dict and bool(value['files'])
         and type(value['original_B24Q1_source_exclusion']) is dict, 'complete prospective metadata freeze')
    for name, digest in value['files'].items():
        capacity.safe_name(name)
        target = ROOT / name
        need(failure.valid_hash(digest) and target.is_file() and not target.is_symlink()
             and sha(target.read_bytes()) == digest, 'every frozen metadata dependency ' + name)
    pin = value['assessment']
    need(type(pin) is dict and set(pin) == {'path', 'sha256'}
         and pin['path'] == 'automation/english_v6_b27_panels/assessment.json'
         and value['files'].get(pin['path']) == pin['sha256']
         and 'automation/english_v6_b27_source_metadata.py' in value['files'],
         'fresh assessment and complete deterministic validator are explicitly frozen')
    report = failure.document((ROOT / pin['path']).read_bytes())
    need(report['freeze_before_optimizer_and_new_source'] is True
         and report['training']['design_path'] == DESIGN.relative_to(ROOT).as_posix()
         and report['training']['design_sha256'] == DESIGN_SHA, 'metadata predates this one optimizer and source draw')
    import english_v6_b27_source_metadata as metadata
    checked = metadata.validate()
    need(Path(metadata.__file__).resolve() == HERE / 'english_v6_b27_source_metadata.py'
         and checked['hashes'] == value['files']
         and checked['original_B24Q1_source_exclusion'] == value['original_B24Q1_source_exclusion'],
         'all fresh files and full original Q1 exposure exclusions independently validate')
    return value


def opening(require_enabled=True):
    value = failure.document(OPENING.read_bytes())
    fields = {'schema', 'condition', 'enabled', 'quality_lock', 'failure_adapter',
              'baseline_challenger_pt_sha256', 'actual_primary_evaluator_sha256',
              'actual_primary_producer_manifest_sha256', 'original_failure_report_sha256'}
    need(set(value) == fields and value['schema'] == 'english-v6-b27-after-original-b24q1-no-promotion-v1'
         and value['condition'] == CONDITION and value['baseline_challenger_pt_sha256'] == CONTROL_PT,
         'new B27 requires original Q1 PRIMARY failure; old B25/B26 opening conditions stay unchanged')
    if value['enabled'] is False:
        need(all(value[key] is None for key in ('quality_lock', 'failure_adapter',
             'actual_primary_evaluator_sha256', 'actual_primary_producer_manifest_sha256',
             'original_failure_report_sha256')), 'disabled opening has no invented actual-result pins')
        need(not require_enabled, 'B27 is sealed until complete original Q1 PRIMARY NO_PROMOTION')
        return value
    need(value['enabled'] is True, 'explicit separately reviewed B27 opening only')
    failure.validate_lock(value['quality_lock'])
    expected = dict(path='automation/english_v6_b24q1_primary_failure.py',
        sha256=design()['failure_reader']['sha256'], base_reader_path='automation/english_v6_b24q1_quality_readback.py',
        base_reader_sha256=failure.BASE_READER_SHA)
    need(value['failure_adapter'] == expected
         and value['actual_primary_evaluator_sha256'] == value['quality_lock']['evaluator_sha256'] == failure.PRIMARY_CODE_SHA
         and value['actual_primary_producer_manifest_sha256'] == value['quality_lock']['producer_manifest_sha256']
         and failure.valid_hash(value['original_failure_report_sha256']), 'exact failure reader, original lock and full-report identity')
    return value


def previous_failure(inputs):
    bound = opening()
    report = failure.verify_failure_original(Path(inputs) / 'no-promotion', bound['quality_lock'])
    need(report['schema'] == 'english-v6-b24q1-original-primary-no-promotion-v1'
         and report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['model_pt_sha256']['challenger'] == CONTROL_PT
         and report['assessment_sha256'] == failure.ASSESSMENT_SHA
         and report['training_assessment_sha256'] == failure.TRAINING_ASSESSMENT_SHA
         and report['training_design_sha256'] == b24.DESIGN_SHA
         and report['all_original_arrays_verified'] is report['original_source_admission_recomputed']
             is report['original_producer_git_verified'] is True
         and all(report[key] is False for key in failure.CLOSED)
         and report['old_B25_and_B26_openers_unchanged'] is True
         and sha(failure.canonical(report)) == bound['original_failure_report_sha256'],
         'entire original source, Git producer and every actual return independently reproduce the bound failure')
    return report


def source_history(inputs):
    bound = opening()
    binding = assessment_binding()
    with failure.original(Path(inputs) / 'no-promotion/originals/producer.zip', bound['quality_lock']['producer'],
                          panel='primary', role='producer') as archive:
        raw = archive.read('primary_producer/automation/english_v6_b24q1_primary_inputs.json')
        need(sha(raw) == bound['quality_lock']['inputs_sha256'], 'actual original source registration before metadata read')
        spec = failure.document(raw)
        fetched = failure.document(archive.read('FETCH.json'))
    import english_v6_b27_source_metadata as metadata
    report = metadata.verify_original_source(Path(inputs) / 'no-promotion/inputs/source.zip',
        fetched['source_metadata']['source'], spec['inputs']['source'])
    need(report == binding['original_B24Q1_source_exclusion'],
         'all actual Q1 source originals and all72 reserved or exposed people remain excluded')
    return report


@contextmanager
def original_models(directory):
    """Read pinned old bytes only; never deserialize or infer a model here."""
    pin = design()['original_B24_models']
    path = Path(directory) / 'TRAIN_models.zip'
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes'], 'original B24 model ZIP')
    raw = path.read_bytes()
    need(sha(raw) == pin['sha256'], 'exact original B24 model ZIP bytes')
    with failure.opened(raw) as archive:
        result_raw = archive.read('RESULT.json')
        result = failure.document(result_raw)
        need(sha(result_raw) == pin['result_sha256']
             and result['schema'] == 'english-v6-b24-train-b23-warm-start-v1'
             and result['identity']['run_id'] == str(pin['run_id']) and result['identity']['run_head'] == pin['run_head']
             and result['recipe_sha256'] == b24.DESIGN_SHA
             and result['phase_initialization_pt_sha256'] == result['model_pt_sha256']['control'] == INITIAL_PT
             and result['model_pt_sha256']['challenger'] == CONTROL_PT
             and result['actual_optimizer_updates'] == 192 and result['actual_new_model_forwards'] == 1847
             and archive.read('DESIGN.json') == b24.DESIGN.read_bytes()
             and sha(archive.read('COMMON_DRAW.json')) == b24.design()['common_draw_sha256'],
             'exact original B24 run, unchanged B23 initializer and original COMMON_DRAW')
        for role, prefix, wanted, stem in (
                ('control', 'control/models/', INITIAL_PT, 'mero43-b23-challenger192'),
                ('challenger', 'challenger/models/', CONTROL_PT, 'mero43-b24-challenger192')):
            files = result[role]['files']
            need(set(files) == {stem + ext for ext in ('.pt', '.npz', '.onnx')}, 'three immutable exports ' + role)
            for name, item in files.items():
                body = archive.read(prefix + name)
                need(len(body) == item['bytes'] and sha(body) == item['sha256'], 'exact saved export ' + name)
            need(files[stem + '.pt']['sha256'] == wanted, 'distinct exact B23 initializer and B24 control')
        yield archive, result


def historical_control(directory, output=None):
    with original_models(directory) as (archive, result):
        receipt = copy.deepcopy(result['challenger'])
        if output is not None:
            output = Path(output)
            need(not output.exists() and output.parent.is_dir(), 'new control export directory only')
            output.mkdir()
            for name in receipt['files']:
                (output / name).write_bytes(archive.read('challenger/models/' + name))
    return dict(receipt, immutable_historical_control=True, new_optimizer_updates=0, new_forwards=0,
        same_job_matched_control=False, original_train_run_id=design()['original_B24_models']['run_id'])


def load_initial_state(directory, frontend, torch, np):
    with original_models(directory) as (archive, _result):
        stem = 'control/models/mero43-b23-challenger192'
        raw = archive.read(stem + '.pt')
        need(sha(raw) == INITIAL_PT, 'same saved B23 initializer, never B24 or B0')
        initial = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        need(sum(v.numel() for v in initial.values()) == 43876 and all(torch.isfinite(v).all() for v in initial.values()),
             'finite exact original fixed-architecture parent')
        with np.load(io.BytesIO(archive.read(stem + '.npz')), allow_pickle=False) as bundle:
            need(set(bundle.files) == set(initial) | {'mel_filters', 'train_mean', 'train_scale'}
                 and all(np.array_equal(bundle[key], value.numpy()) for key, value in initial.items())
                 and all(np.array_equal(bundle[key], value) for key, value in
                    zip(('mel_filters', 'train_mean', 'train_scale'), frontend, strict=True)),
                 'same exact PT/NPZ state and unchanged original frontend')
    return {key: value.detach().clone() for key, value in initial.items()}


def fit_engine():
    design()
    original = b24.b23.fit
    profile = SimpleNamespace(**vars(b24.old))
    profile.NATURAL_WEIGHT = NATURAL_WEIGHT
    return FunctionType(original.__code__, dict(original.__globals__, old=profile),
                        original.__name__, original.__defaults__, original.__closure__)


def fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    original = b24.fit
    namespace = dict(original.__globals__, b23=SimpleNamespace(fit=fit_engine()), STEM=STEM)
    wrapper = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    value = wrapper(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F)
    return dict(value, natural_weight=NATURAL_WEIGHT, historical_control_pt_sha256=CONTROL_PT,
                old_B24_optimizer_updates_reexecuted=0, new_teacher_forwards=0, new_control_forwards=0)


def _reuse(name):
    """Reuse only immutable transport/once guards; no historical globals change."""
    need(name in ('guard', 'consume', 'verify_consumed'), 'three reviewed non-numerical helpers only')
    original = getattr(b24, name)
    namespace = dict(original.__globals__)
    for key in ('design', 'opening', 'assessment_binding', 'guard', 'source_preflight', 'ROOT', 'DESIGN_SHA',
                'BINDING', 'OPENING', 'REPO', 'BRANCH', 'WORKFLOW', 'MARKER', 'SCHEMA', 'gh'):
        namespace[key] = globals()[key]
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def guard(parent, env=None, event=None, git_parent=None):
    return _reuse('guard')(parent, env, event, git_parent)


def external_inputs(inputs):
    """Keep original inputs external, complete and hash-bound; never prune proof."""
    inputs = Path(inputs)
    value = capacity.inventory(inputs / 'no-promotion')
    need(set(value['files']) == capacity.PRIMARY_FILES, 'exact complete eighteen-file original Q1 proof')
    pins = {}
    for name, pin in value['files'].items():
        maximum = capacity.PRIMARY_SOURCE_LIMIT if name == 'inputs/source.zip' else capacity.CORE_LIMIT
        need(pin['bytes'] <= maximum, 'unchanged original role capacity ' + name)
        if name.startswith('inputs/'):
            pins['no-promotion/' + name] = pin
    training = capacity.inventory(inputs / 'training')
    need(set(training['files']) == set(design()['input_archives']), 'all eighteen unchanged TRAIN archives retained')
    for name, pin in training['files'].items():
        need(pin['sha256'] == design()['input_archives'][name]['sha256'] and pin['bytes'] <= capacity.CORE_LIMIT,
             'exact complete old TRAIN archive ' + name)
        pins['training/' + name] = pin
    return dict(schema='english-v6-b27-complete-external-original-inputs-v1', files=pins,
                original_Q1_input_archives=12, original_TRAIN_input_archives=18,
                complete_bytes_retained=True, input_archives_nested_into_new_producer=False)


def receipt_capacity(inputs):
    """Pre-fit envelope for four small TRAIN artifacts; large inputs stay pinned."""
    inputs = Path(inputs)
    names = {'original_b24/TRAIN_LIVE.json'} | {'original_b24/TRAIN_' + role + '.zip'
             for role in ('models', 'producer', 'receipts', 'consumed')}
    names |= {'no-promotion/api/' + name + '.json' for name in ('run', 'commit', 'artifacts')}
    names |= {'no-promotion/originals/' + name + '.zip' for name in ('result', 'producer', 'consumed')}
    files = {}
    for name in sorted(names):
        path = inputs / name
        need(path.is_file() and not path.is_symlink(), 'complete original retained receipt ' + name)
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        files[name] = dict(bytes=path.stat().st_size, sha256=digest)
    total = sum(pin['bytes'] for pin in files.values()) + RECEIPT_RESERVE
    members = len(files) + 64
    bound = capacity.zip_bound(total, members, sum(len(name.encode()) for name in files) + 64 * 1024)
    need(total <= capacity.CORE_LIMIT and bound <= capacity.CORE_LIMIT,
         'complete original receipts plus prospective metadata reserve fit unchanged128MB role cap before any budget')
    return dict(schema='english-v6-b27-prospective-training-receipt-capacity-v1', originals=files,
                new_receipt_reserve_bytes=RECEIPT_RESERVE, expanded_bound=total,
                zip_bound=bound, limit=capacity.CORE_LIMIT, discarded_original_bytes=0)


def source_preflight(inputs):
    inputs = Path(inputs)
    proof = previous_failure(inputs)
    binding = assessment_binding()
    history = source_history(inputs)
    original = old_proof.verify(inputs / 'original_b24')
    control = historical_control(inputs / 'original_b24')
    external = external_inputs(inputs)
    need((inputs / 'EXTERNAL_INPUTS.json').read_bytes() == encoded(external), 'all external originals match pre-fetch pin receipt')
    envelope = receipt_capacity(inputs)
    report, *values = b24.baseline.source_preflight(inputs / 'training')
    need(sha(encoded(values[-1])) == b24.design()['common_draw_sha256'], 'same complete original192 draws')
    report = dict(report, schema='english-v6-b27-authenticated-train-source-v1',
        original_B24Q1_failed_primary=proof, original_B24_train=original, historical_control=control,
        initialization_pt_sha256=INITIAL_PT, natural_weight=NATURAL_WEIGHT,
        original_B24Q1_source_exclusion=history, prospective_assessment_sha256=binding['assessment']['sha256'],
        complete_external_inputs=external, receipt_capacity=envelope,
        all_original_train_sources_and_teachers_unchanged=True, dev_rows_used_for_training=0,
        new_model_forwards=0, new_source_calls=0, new_dev_model_forwards=0,
        Q1_metrics_used_to_choose_parameters=False)
    return report, *values


def fetch(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), 'fresh immutable B27 input root')
    inputs.mkdir()
    proof = failure.fetch_failure_original(inputs / 'no-promotion', opening()['quality_lock'])
    need(sha(failure.canonical(proof)) == opening()['original_failure_report_sha256'], 'exact independently locked Q1 failure')
    history = source_history(inputs)
    original = inputs / 'original_b24'
    original.mkdir()
    bound = old_ancestry.training_binding()['training']
    commit = failure.document(gh('commits/' + bound['run_head']))
    live = dict(run=failure.document(gh('actions/runs/' + str(bound['run_id']))), commit=commit,
        artifacts=failure.document(gh('actions/runs/' + str(bound['run_id']) + '/artifacts?per_page=100')),
        tree=failure.document(gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')))
    old_ancestry.validate_training_live(live)
    write(original / 'TRAIN_LIVE.json', live)
    for role, pin in bound['artifacts'].items():
        raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'complete original B24 TRAIN artifact ' + role)
        (original / ('TRAIN_' + role + '.zip')).write_bytes(raw)
    old_proof.verify(original)
    (inputs / 'training').mkdir()
    receipts = {}
    for name, pin in design()['input_archives'].items():
        artifact = failure.document(gh('actions/artifacts/' + str(pin['artifact_id'])))
        need(artifact['id'] == pin['artifact_id'] and artifact['expired'] is False
             and artifact['digest'] == 'sha256:' + pin['sha256'], 'live immutable TRAIN input ' + name)
        raw = gh('actions/artifacts/' + str(pin['artifact_id']) + '/zip')
        need(sha(raw) == pin['sha256'] and len(raw) == artifact['size_in_bytes']
             and len(raw) <= capacity.CORE_LIMIT, 'exact complete unchanged TRAIN bytes ' + name)
        (inputs / 'training' / name).write_bytes(raw)
        receipts[name] = artifact
    write(inputs / 'EXTERNAL_INPUTS.json', external_inputs(inputs))
    envelope = receipt_capacity(inputs)
    write(inputs / 'FETCH.json', dict(identity=identity, artifacts=receipts, recipe_sha256=DESIGN_SHA,
        assessment_binding_sha256=sha(BINDING.read_bytes()), opening_sha256=sha(OPENING.read_bytes()),
        previous_failure=proof, original_B24Q1_source_exclusion=history, receipt_capacity=envelope,
        external_inputs_sha256=sha((inputs / 'EXTERNAL_INPUTS.json').read_bytes()), new_model_forwards=0))


def consume(inputs, parent):
    return _reuse('consume')(inputs, parent)


def verify_consumed(inputs, parent):
    identity = _reuse('verify_consumed')(inputs, parent)
    marker = failure.document((Path(inputs) / 'CONSUMPTION.json').read_bytes())
    need(marker['schema'] == SCHEMA and marker['status'] == 'TRAINING_BUDGET_CONSUMED_UNKNOWN'
         and marker['promotion_allowed'] is False, 'same independently uploaded unopened completion status')
    return identity


def execute(inputs, output, parent):
    inputs, output = Path(inputs), Path(output)
    identity = verify_consumed(inputs, parent)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'one fresh output; no training retry')
    report, rows, waves, groups, extra, extra_waves, slots = source_preflight(inputs)
    need(sha(encoded(report)) == failure.document((inputs / 'CONSUMPTION.json').read_bytes())['source_preflight_sha256'],
         'same complete source proof as the durable consumption marker')
    output.mkdir()
    write(output / 'STARTED.json', dict(schema=SCHEMA, identity=identity, budget=design()['budget'], recipe_sha256=DESIGN_SHA))
    write(output / 'SOURCE.json', report)
    write(output / 'COMMON_DRAW.json', slots)
    for name, path in (('DESIGN.json', DESIGN), ('ASSESSMENT_BINDING.json', BINDING), ('OPENING.json', OPENING)):
        (output / name).write_bytes(path.read_bytes())
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    _unused_b0, _unused_b9, frontend = b24.old.b17.load_parents(inputs / 'training', torch, np)
    base_cache = b24.old.b13.paired.make_features(rows, waves, frontend, torch, np)
    natural_cache = b24.old.b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    paths, old_audit = b24.baseline.prepare_union_paths(inputs / 'training', rows, waves, frontend, np, torch)
    teachers = b24.baseline.saved_ordinary_teachers(inputs / 'training', torch, np)
    write(output / 'UNION_EVIDENCE.json', report['union'])
    write(output / 'OLD120_POSITIVE_REFERENCES.json', old_audit)
    (output / 'control').mkdir()
    control = historical_control(inputs / 'original_b24', output / 'control/models')
    target = output / 'challenger'
    target.mkdir()
    candidate = fit(load_initial_state(inputs / 'original_b24', frontend, torch, np), slots, base_cache, natural_cache,
                    paths, teachers, frontend, target, torch, np, F)
    observations = {name: dict(bytes=(target / name).stat().st_size, sha256=sha((target / name).read_bytes())) for name in (
        'TRAIN_FORWARD_INDEX.json', 'TRAIN_FORWARD_LOGITS.npz', 'OPTIMIZER_TRACE.json',
        'POSITIVE_RETENTION_TRACE.json', 'ORDINARY_MARGIN_TRACE.json', 'CHECKPOINT_RELOAD_LOGITS.npz')}
    result = dict(schema=SCHEMA, status=RESULT_STATUS, identity=identity, recipe_sha256=DESIGN_SHA,
        design_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        assessment_sha256=assessment_binding()['assessment']['sha256'], source_sha256=sha((output / 'SOURCE.json').read_bytes()),
        opening_sha256=sha(OPENING.read_bytes()), common_draw_sha256=sha((output / 'COMMON_DRAW.json').read_bytes()),
        union_evidence_sha256=sha((output / 'UNION_EVIDENCE.json').read_bytes()),
        old120_references_sha256=sha((output / 'OLD120_POSITIVE_REFERENCES.json').read_bytes()),
        prefix_blank_reduction='amax', natural_weight=NATURAL_WEIGHT,
        original_B24Q1_NO_PROMOTION_reauthenticated=True, Q1_metrics_used_to_choose_parameters=False,
        phase_initialization_pt_sha256=INITIAL_PT, optimizer_initialization='fresh_AdamW',
        restored_optimizer_state=False, exact_optimizer_continuation=False,
        old_B23_optimizer_updates_reexecuted=0, old_B24_optimizer_updates_reexecuted=0,
        model_pt_sha256=dict(B0=b24.old.B0_PT, B9=b24.old.b10.b9.MODELS['B9_control']['pt'],
            control=CONTROL_PT, challenger=candidate['files'][STEM + '.pt']['sha256']),
        control=dict(control, output_stem='control/models/mero43-b24-challenger192'), challenger=candidate,
        saved_training_observations=observations, actual_optimizer_updates=192, actual_new_model_forwards=1847,
        new_control_optimizer_updates=0, new_control_forwards=0, new_teacher_forwards=0,
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        dev_cal_locked_forwards=0, new_tts_asr_calls=0, historical_numerical_replay_claimed=False,
        same_job_matched_control=False, architecture=b24.baseline.architecture_contract(), promotion_allowed=False,
        physical_ssc305=False, field_far_per_hour=None)
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('check')
    for mode in ('fetch', 'consume', 'execute', 'preflight'):
        item = sub.add_parser(mode)
        item.add_argument('--inputs', type=Path, required=True)
        if mode != 'preflight': item.add_argument('--prepared-parent', required=True)
        if mode == 'execute': item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'check':
        design(); assessment_binding(False); opening(False)
        print('PASS: sealed B27 one-scalar training path; actual complete Q1 failure and fresh metadata required')
    elif args.mode == 'fetch': fetch(args.inputs, args.prepared_parent)
    elif args.mode == 'consume': print(json.dumps(consume(args.inputs, args.prepared_parent), sort_keys=True))
    elif args.mode == 'preflight': print(json.dumps(source_preflight(args.inputs)[0], sort_keys=True))
    else: print(json.dumps(execute(args.inputs, args.output, args.prepared_parent), sort_keys=True))
