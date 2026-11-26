"""Sealed B28 TRAIN authority for the frozen natural suffix no-event objective.

No model is imported by this module. Complete original B27 PRIMARY failure,
all 59 external originals, fresh preoptimizer metadata, a public once marker
and the exact reviewed workflow must precede one 192-update/1847-forward fit.
Old qualifications, original sources, model identities and role caps remain
unchanged. This implementation and its unfrozen design grant no execution.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_v6_b27_natural_weight as b27
import english_v6_b27_profile as profile
import english_v6_b27_original_proof as transport
import english_v6_b27_quality_readback as quality
import english_v6_b27_evidence_scope as evidence_scope
import english_v6_b28_natural_event_fit as objective_fit

b24, capacity, failure = b27.b24, b27.capacity, b27.failure
fit = objective_fit
old_proof, original_models, historical_control = b27.old_proof, b27.original_models, b27.historical_control
need, sha, encoded, write, gh = b27.need, b27.sha, b27.encoded, b27.write, b27.gh
DESIGN = HERE / 'english_v6_b28_natural_event_design.json'
DESIGN_SHA = None  # Set only after the complete design and assessment are reviewed.
OPENING = HERE / 'english_v6_b28_opening.json'
BINDING = HERE / 'english_v6_b28_assessment_binding.json'
INITIAL_PT, CONTROL_PT, STEM = objective_fit.INITIAL_PT, objective_fit.CONTROL_PT, objective_fit.STEM
NATURAL_WEIGHT = 0.16
B27_PT = 'dd0d4bc08be1162886ab462dfb3ad7f4503150a57ee13272d01f41e80ca1069f'
B27_PROOF_SHA = '6c6334a1d8d6452e186c41fe472eca6e072a8af42a02a81d9aac68b30e7d9a28'
B27_READER_SHA = '87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa'
B27_IMPORT_ADAPTER_PATH = 'automation/english_v6_b27_primary_no_promotion_readback.py'
B27_IMPORT_ADAPTER_SHA = '9de0cb1e8619d6d3127c09cc279d2480621663feabf1208535f9099e2c271518'
B27_IMPORT_REPORT_SCHEMA = 'english-v6-b27-primary-rooted-no-promotion-readback-v1'
CONDITION = 'only_after_actual_complete_B27_primary_NO_PROMOTION'
SCHEMA = 'english-v6-b28-train-natural-event-v1'
RESULT_STATUS = 'TRAINED_NATURAL_EVENT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
REPO = b27.REPO
BRANCH = 'refs/heads/research/en-v6-b28-natural-event-train-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b28-natural-event-train-once.yml'
MARKER = 'english-v6-b28-natural-event-train-consumed'
OBSERVATIONS = ('TRAIN_FORWARD_INDEX.json', 'TRAIN_FORWARD_LOGITS.npz', 'OPTIMIZER_TRACE.json',
    'POSITIVE_RETENTION_TRACE.json', 'ORDINARY_MARGIN_TRACE.json', 'CHECKPOINT_RELOAD_LOGITS.npz',
    'NATURAL_OBJECTIVE_TRACE.json')
OLD_B24_FILES = {'TRAIN_LIVE.json'} | {'TRAIN_' + role + '.zip' for role in ('models', 'producer', 'receipts', 'consumed')}


def design(require_frozen=True):
    raw = DESIGN.read_bytes()
    value = failure.document(raw)
    if DESIGN_SHA is None:
        need(not require_frozen, 'B28 design is unbound; preparation cannot authorize TRAIN')
    else:
        need(failure.valid_hash(DESIGN_SHA) and sha(raw) == DESIGN_SHA, 'exact independently frozen B28 design')
    original = b24.design()
    contract = objective_fit.contract()
    need(value['schema'] == 'english-v6-b28-train-natural-event-design-v1'
         and value['candidate_stem'] == STEM and value['initial_pt_sha256'] == INITIAL_PT
         and value['historical_control_pt_sha256'] == CONTROL_PT and value['natural_weight'] == NATURAL_WEIGHT
         and value['optimizer_updates'] == 192 and value['maximum_new_model_forwards'] == 1847
         and value['budget'] == original['budget'] and value['input_archives'] == original['input_archives']
         and all(value[key] == expected for key, expected in contract.items()),
         'same original TRAIN bytes, B23 initialization, B24 control and exact no-event fit')
    need(value['archive_caps'] == dict(core_artifact=128000000,
         complete_external_original_B27_proof=1280000000, new_receipt_metadata_reserve=8000000)
         and capacity.CORE_LIMIT == 128000000
         and all(value[key] is False for key in ('scientific_execution_authorized',
             'B27_scores_used_to_choose_objective_or_coefficients', 'automatic_next_assessment_allowed',
             'old_openers_or_frozen_sources_modified', 'promotion_allowed', 'product_released')),
         'unchanged role capacities and preparation-only authority')
    for pin in value['frozen_dependencies']:
        capacity.safe_name(pin['path'])
        path = ROOT / pin['path']
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
             and sha(path.read_bytes()) == pin['sha256'], 'exact complete frozen dependency ' + pin['path'])
    return value


def assessment_binding(require_frozen=True, require_catalog=False):
    value = failure.document(BINDING.read_bytes())
    need(set(value) == {'schema', 'recipe_path', 'recipe_sha256', 'frozen', 'optimizer_allowed',
         'assessment', 'files', 'original_B27_source_exclusion'}
         and value['schema'] == 'english-v6-b28-prospective-assessment-binding-v1'
         and value['recipe_path'] == DESIGN.relative_to(ROOT).as_posix()
         and value['optimizer_allowed'] is False, 'separate prospective metadata, never an execution permit')
    if value['frozen'] is False:
        need(value['recipe_sha256'] in (None, DESIGN_SHA) and value['assessment'] is None
             and value['files'] == {} and value['original_B27_source_exclusion'] is None,
             'inert binding has no invented assessment, source inventory or budget')
        need(not require_frozen, 'complete fresh B28 primary and sealed replica metadata required before TRAIN')
        return value
    design()
    need(value['frozen'] is True and value['recipe_sha256'] == DESIGN_SHA
         and type(value['files']) is dict and bool(value['files'])
         and type(value['original_B27_source_exclusion']) is dict, 'complete exact preoptimizer binding')
    for name, digest in value['files'].items():
        capacity.safe_name(name)
        path = ROOT / name
        need(failure.valid_hash(digest) and path.is_file() and not path.is_symlink()
             and sha(path.read_bytes()) == digest, 'frozen metadata dependency ' + name)
    pin = value['assessment']
    need(type(pin) is dict and set(pin) == {'path', 'sha256'}
         and pin['path'] == 'automation/english_v6_b28_panels/assessment.json'
         and value['files'].get(pin['path']) == pin['sha256']
         and 'automation/english_v6_b28_source_metadata.py' in value['files'],
         'the assessment and full source validator are frozen explicitly')
    assessment = failure.document((ROOT / pin['path']).read_bytes())
    need(assessment['freeze_before_optimizer_and_new_source'] is True
         and assessment['training']['design_path'] == DESIGN.relative_to(ROOT).as_posix()
         and assessment['training']['design_sha256'] == DESIGN_SHA,
         'all fresh primary and reserve metadata predates one TRAIN optimizer')
    metadata = importlib.import_module('english_v6_b28_source_metadata')
    need(Path(metadata.__file__).resolve() == HERE / 'english_v6_b28_source_metadata.py', 'exact local source metadata authority')
    checked = metadata.validate() if require_catalog else metadata.validate_registration()
    need(checked['hashes'] == value['files']
         and checked['original_B27_source_exclusion'] == value['original_B27_source_exclusion'],
         'complete deterministic new72-person assessment and old exposure exclusions')
    return value


def opening(require_enabled=True):
    value = failure.document(OPENING.read_bytes())
    evidence = {'quality_lock', 'quality_reader_sha256', 'original_failure_report_sha256',
        'original_primary_files', 'original_primary_transport', 'original_training_binding_sha256',
        'original_B27_challenger_pt_sha256', 'quality_import_adapter_path',
        'quality_import_adapter_sha256', 'quality_import_adapter_report_sha256'}
    need(set(value) == {'schema', 'condition', 'enabled'} | evidence
         and value['schema'] == 'english-v6-b28-after-original-b27-no-promotion-v1'
         and value['condition'] == CONDITION, 'B28 opens only after its exact complete B27 terminal')
    if value['enabled'] is False:
        need(all(value[key] is None for key in evidence), 'unopened B28 has no fabricated original result identity')
        need(not require_enabled, 'complete original B27 NO_PROMOTION remains unbound')
        return value
    need(value['enabled'] is True and value['original_B27_challenger_pt_sha256'] == B27_PT
         and value['original_training_binding_sha256'] == B27_PROOF_SHA
         and sha((HERE / 'english_v6_b27_primary_proof_binding.json').read_bytes()) == B27_PROOF_SHA
         and value['quality_reader_sha256'] == B27_READER_SHA
         and sha(Path(quality.__file__).read_bytes()) == B27_READER_SHA
         and failure.valid_hash(value['original_failure_report_sha256']),
         'unchanged exact B27 model, complete external proof and original independent reader')
    need(failure.valid_hash(B27_IMPORT_ADAPTER_SHA)
         and value['quality_import_adapter_path'] == B27_IMPORT_ADAPTER_PATH
         and value['quality_import_adapter_sha256'] == B27_IMPORT_ADAPTER_SHA
         and sha((ROOT / B27_IMPORT_ADAPTER_PATH).read_bytes()) == B27_IMPORT_ADAPTER_SHA
         and failure.valid_hash(value['quality_import_adapter_report_sha256']),
         'separate exact reviewed import adapter and complete original certification, never an original-reader substitution')
    quality.validate_lock(value['quality_lock'])
    need(value['quality_lock']['panel'] == 'primary'
         and value['quality_lock']['external_proof_binding_sha256'] == B27_PROOF_SHA,
         'only actual original B27 PRIMARY terminal can open this candidate')
    profile.file_pins(value['original_primary_files'], 1280000000)
    need(set(value['original_primary_files']) == capacity.PRIMARY_FILES,
         'all18 original PRIMARY inputs, outputs and full API files required')
    profile.validate_transport(value['original_primary_transport'], value['original_primary_files'])
    bound = profile.binding('primary')
    need(len(bound['files']) == 41 and sum(pin['bytes'] for pin in bound['files'].values())
         + sum(pin['bytes'] for pin in value['original_primary_files'].values()) <= 1280000000,
         'all59 external originals fit the unchanged complete-proof capacity')
    return value


def failure_certification(bound=None):
    bound = opening() if bound is None else bound
    return dict(schema='english-v6-b28-original-b27-failure-certification-binding-v1',
        original_reader_path='automation/english_v6_b27_quality_readback.py',
        original_reader_sha256=bound['quality_reader_sha256'],
        original_report_sha256=bound['original_failure_report_sha256'],
        adapter_path=bound['quality_import_adapter_path'], adapter_sha256=bound['quality_import_adapter_sha256'],
        adapter_report_schema=B27_IMPORT_REPORT_SCHEMA,
        adapter_report_sha256=bound['quality_import_adapter_report_sha256'],
        original_standalone_continuation_succeeded=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0)


def quality_import_adapter():
    need(failure.valid_hash(B27_IMPORT_ADAPTER_SHA), 'complete original B27 import adapter remains unbound')
    path = ROOT / B27_IMPORT_ADAPTER_PATH
    need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == B27_IMPORT_ADAPTER_SHA,
         'exact complete reviewed B27 import adapter before import')
    module = importlib.import_module(path.stem)
    need(Path(module.__file__).resolve() == path and module.READER_SHA == B27_READER_SHA
         and module.REPORT_SCHEMA == B27_IMPORT_REPORT_SCHEMA,
         'separate original-root adapter preserves the untouched original reader identity')
    return module


def external_inputs(inputs):
    inputs = Path(inputs)
    bound, previous = profile.binding('primary'), opening()
    expected = {'training-origin/' + name: pin for name, pin in bound['files'].items()}
    expected.update({'primary-quality/' + name: pin for name, pin in previous['original_primary_files'].items()})
    profile.file_pins(expected, design()['archive_caps']['complete_external_original_B27_proof'])
    need(profile.inventory(inputs / 'proof') == expected, 'all59 complete original external bytes; no summary substitute')
    training = profile.inventory(inputs / 'proof/training-origin/qualification/training')
    need(set(training) == set(design()['input_archives'])
         and all(training[name]['sha256'] == pin['sha256'] and training[name]['bytes'] <= capacity.CORE_LIMIT
                 for name, pin in design()['input_archives'].items()), 'same18 complete original TRAIN archives')
    metadata = importlib.import_module('english_v6_b28_source_metadata')
    catalog = metadata.catalog_registration()
    need(len(metadata.CATALOG_FILES) == len(catalog['files']) == 7
         and profile.inventory(inputs / 'metadata/train360') == catalog['files'],
         'all seven original metadata producer, consumption, result and complete Git/API files remain complete')
    return dict(schema='english-v6-b28-complete-external-original-inputs-v1', files=expected,
        complete_original_B27_primary_files=18, complete_original_B27_training_files=41,
        original_TRAIN_input_archives=18, complete_bytes_retained=True,
        input_archives_nested_into_new_producer=False, total_bytes=sum(v['bytes'] for v in expected.values()),
        capacity=1280000000, metadata_catalog_files=catalog['files'],
        metadata_catalog_original_count=7,
        metadata_catalog_total_bytes=sum(pin['bytes'] for pin in catalog['files'].values()))


def previous_failure(inputs):
    inputs = Path(inputs)
    bound = opening()
    external_inputs(inputs)
    adapter = quality_import_adapter()
    # Existing cache mechanics always hash every original byte, and its scope
    # never survives this complete read. It cannot accept a supplied summary.
    with evidence_scope.scope():
        raw = evidence_scope.verify_once('b27_primary_quality', inputs / 'proof/primary-quality',
            bound['original_primary_files'], profile.canonical(bound['quality_lock']), Path(adapter.__file__),
            bound['quality_import_adapter_sha256'], bound['quality_import_adapter_report_sha256'],
            lambda: profile.canonical(adapter.verify_original(inputs / 'proof/primary-quality',
                bound['quality_lock'], inputs / 'proof')))
    certificate = failure.document(raw)
    adapter.validate_report(certificate, bound['quality_lock'])
    need(certificate['schema'] == B27_IMPORT_REPORT_SCHEMA
         and certificate['original_reader_sha256'] == bound['quality_reader_sha256'] == B27_READER_SHA
         and certificate['import_scope_adapter_sha256'] == bound['quality_import_adapter_sha256']
         and sha(raw) == bound['quality_import_adapter_report_sha256']
         and certificate['original_report_sha256'] == bound['original_failure_report_sha256']
         and certificate['original_standalone_continuation_succeeded'] is False,
         'full newly executed original-root certification and original report retain separate exact identities')
    report = certificate['original_report']
    need(report['schema'] == 'english-v6-b27-verified-original-quality-v1'
         and report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['model_pt_sha256']['challenger'] == B27_PT
         and report['model_pt_sha256']['control'] == CONTROL_PT
         and report['external_proof_binding_sha256'] == B27_PROOF_SHA
         and report['original_training_report_sha256'] == profile.binding('primary')['training_report_sha256']
         and all(report[key] is True for key in ('all_original_arrays_verified', 'original_source_admission_recomputed',
             'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified'))
         and all(report[key] is False for key in ('replication_open_allowed', 'candidate_engineering_validation_allowed',
             'automatic_next_assessment_allowed', 'promotion_allowed', 'product_released'))
         and all(type(report[key]) is int and report[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls'))
         and sha(profile.canonical(report)) == bound['original_failure_report_sha256'],
         'entire actual B27 source/TRAIN/producer and every return reproduce terminal NO_PROMOTION')
    return report


def source_history(inputs):
    inputs, bound = Path(inputs), opening()
    metadata = importlib.import_module('english_v6_b28_source_metadata')
    with metadata.catalog_scope(inputs / 'metadata/train360'):
        binding = assessment_binding(require_catalog=True)
        with quality.original_archive(inputs / 'proof/primary-quality/originals/producer.zip',
                bound['quality_lock']['producer'], 'primary', 'producer') as archive:
            spec_raw = archive.read('primary_producer/automation/english_v6_b27_primary_inputs.json')
            need(sha(spec_raw) == bound['quality_lock']['inputs_sha256'], 'actual original B27 source input registration')
            spec = failure.document(spec_raw)
            fetched = failure.document(archive.read('FETCH.json'))
        result = metadata.verify_original_source(inputs / 'proof/primary-quality/inputs/source.zip',
            fetched['source_metadata']['source'], spec['inputs']['source'])
        need(result == binding['original_B27_source_exclusion'], 'all original B27 PCM and exposed or sealed people excluded')
    return result


def retain_original_b24(inputs, create=False):
    """Use the five exact old members inside the fully pinned B27 receipts."""
    inputs = Path(inputs)
    pin = profile.binding('primary')['files']['trained/TRAIN_receipts.zip']
    archive_path = inputs / 'proof/training-origin/trained/TRAIN_receipts.zip'
    raw = archive_path.read_bytes()
    need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'complete original B27 TRAIN receipt artifact')
    destination = inputs / 'original_b24'
    if create:
        need(not destination.exists(), 'fresh retained original B24 evidence directory')
        destination.mkdir()
    files = {}
    with failure.opened(raw) as archive:
        prefix = 'original_b24/'
        members = {item.filename for item in archive.infolist() if not item.is_dir() and item.filename.startswith(prefix)}
        need(members == {prefix + name for name in OLD_B24_FILES}, 'all five original B24 files; no selective original-member pruning')
        for name in sorted(OLD_B24_FILES):
            body = archive.read(prefix + name)
            target = destination / name
            if create:
                with target.open('xb') as handle:
                    handle.write(body)
            need(target.is_file() and not target.is_symlink() and target.read_bytes() == body,
                 'unaltered original receipt member ' + name)
            files[name] = dict(bytes=len(body), sha256=sha(body))
    need(profile.inventory(destination) == files, 'exact complete old B24 retained evidence root')
    return files


def receipt_capacity(inputs):
    originals = retain_original_b24(inputs)
    total = sum(pin['bytes'] for pin in originals.values()) + design()['archive_caps']['new_receipt_metadata_reserve']
    bound = capacity.zip_bound(total, len(originals) + 64,
        sum(len(name.encode()) for name in originals) + 64 * 1024)
    need(total <= capacity.CORE_LIMIT and bound <= capacity.CORE_LIMIT,
         'five originals and all prospective receipt metadata fit unchanged128MB before budget')
    return dict(schema='english-v6-b28-prospective-training-receipt-capacity-v1', originals=originals,
        new_receipt_reserve_bytes=8000000, expanded_bound=total, zip_bound=bound,
        limit=capacity.CORE_LIMIT, discarded_original_bytes=0, external_proof_embedded=False)


def source_preflight(inputs):
    inputs = Path(inputs)
    proof = previous_failure(inputs)
    binding = assessment_binding()
    history = source_history(inputs)
    retain_original_b24(inputs)
    original = b27.old_proof.verify(inputs / 'original_b24')
    control = b27.historical_control(inputs / 'original_b24')
    external = external_inputs(inputs)
    need((inputs / 'EXTERNAL_INPUTS.json').read_bytes() == encoded(external), 'complete external originals match initial fetch')
    envelope = receipt_capacity(inputs)
    report, *values = b24.baseline.source_preflight(inputs / 'proof/training-origin/qualification/training')
    need(sha(encoded(values[-1])) == objective_fit.COMMON_DRAW_SHA, 'same original192 TRAIN-only draws')
    report = dict(report, schema='english-v6-b28-authenticated-train-source-v1',
        original_B27_failed_primary=proof, original_B24_train=original, historical_control=control,
        original_B27_failure_certification=failure_certification(),
        initialization_pt_sha256=INITIAL_PT, natural_weight=NATURAL_WEIGHT, **objective_fit.contract(),
        original_B27_source_exclusion=history, prospective_assessment_sha256=binding['assessment']['sha256'],
        complete_external_inputs=external, receipt_capacity=envelope,
        all_original_train_sources_and_teachers_unchanged=True, dev_rows_used_for_training=0,
        new_model_forwards=0, new_source_calls=0, new_dev_model_forwards=0,
        B27_metrics_used_to_choose_parameters=False)
    return report, *values


def _reuse(name):
    need(name in ('guard', 'consume', 'verify_consumed'), 'only three original transport/once guard helpers')
    original = getattr(b24, name)
    namespace = dict(original.__globals__)
    for key in ('design', 'opening', 'assessment_binding', 'guard', 'source_preflight', 'ROOT', 'DESIGN_SHA',
                'BINDING', 'OPENING', 'REPO', 'BRANCH', 'WORKFLOW', 'MARKER', 'SCHEMA', 'gh'):
        namespace[key] = globals()[key]
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def guard(parent, env=None, event=None, git_parent=None):
    return _reuse('guard')(parent, env, event, git_parent)


def fetch(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    need(inputs.is_absolute() and inputs == inputs.resolve() and inputs.parent.is_dir() and not inputs.exists(),
         'fresh original B28 input root')
    inputs.mkdir()
    (inputs / 'proof').mkdir()
    train_bound, bound = profile.binding('primary'), opening()
    transport.fetch_files(inputs / 'proof/training-origin', train_bound['files'], train_bound['transport'])
    transport.fetch_files(inputs / 'proof/primary-quality', bound['original_primary_files'], bound['original_primary_transport'])
    metadata = importlib.import_module('english_v6_b28_source_metadata')
    catalog = metadata.catalog_registration()
    (inputs / 'metadata').mkdir()
    transport.fetch_files(inputs / 'metadata/train360', catalog['files'], catalog['transport'])
    proof = previous_failure(inputs)
    history = source_history(inputs)
    retain_original_b24(inputs, create=True)
    b27.old_proof.verify(inputs / 'original_b24')
    b27.historical_control(inputs / 'original_b24')
    external = external_inputs(inputs)
    write(inputs / 'EXTERNAL_INPUTS.json', external)
    envelope = receipt_capacity(inputs)
    result = dict(identity=identity, recipe_sha256=DESIGN_SHA, **objective_fit.contract(),
        assessment_binding_sha256=sha(BINDING.read_bytes()), opening_sha256=sha(OPENING.read_bytes()),
        previous_failure=proof, original_B27_source_exclusion=history, receipt_capacity=envelope,
        original_B27_failure_certification=failure_certification(),
        external_inputs_sha256=sha((inputs / 'EXTERNAL_INPUTS.json').read_bytes()), new_model_forwards=0)
    write(inputs / 'FETCH.json', result)
    return result


def consume(inputs, parent):
    return _reuse('consume')(inputs, parent)


def verify_consumed(inputs, parent):
    identity = _reuse('verify_consumed')(inputs, parent)
    marker = failure.document((Path(inputs) / 'CONSUMPTION.json').read_bytes())
    need(marker['schema'] == SCHEMA and marker['status'] == 'TRAINING_BUDGET_CONSUMED_UNKNOWN'
         and marker['promotion_allowed'] is False, 'exact durable new budget; no earlier experiment is replayed')
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
    training = inputs / 'proof/training-origin/qualification/training'
    _unused_b0, _unused_b9, frontend = b24.old.b17.load_parents(training, torch, np)
    base_cache = b24.old.b13.paired.make_features(rows, waves, frontend, torch, np)
    natural_cache = b24.old.b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    paths, old_audit = b24.baseline.prepare_union_paths(training, rows, waves, frontend, np, torch)
    teachers = b24.baseline.saved_ordinary_teachers(training, torch, np)
    write(output / 'UNION_EVIDENCE.json', report['union'])
    write(output / 'OLD120_POSITIVE_REFERENCES.json', old_audit)
    (output / 'control').mkdir()
    control = b27.historical_control(inputs / 'original_b24', output / 'control/models')
    target = output / 'challenger'
    target.mkdir()
    candidate = objective_fit.fit(b27.load_initial_state(inputs / 'original_b24', frontend, torch, np),
        slots, base_cache, natural_cache, paths, teachers, frontend, target, torch, np, F)
    observations = {name: dict(bytes=(target / name).stat().st_size, sha256=sha((target / name).read_bytes()))
                    for name in OBSERVATIONS}
    result = dict(schema=SCHEMA, status=RESULT_STATUS, identity=identity, recipe_sha256=DESIGN_SHA,
        design_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        assessment_sha256=assessment_binding()['assessment']['sha256'], source_sha256=sha((output / 'SOURCE.json').read_bytes()),
        opening_sha256=sha(OPENING.read_bytes()), **objective_fit.contract(),
        natural_objective_schema=objective_fit.OBJECTIVE_SCHEMA, natural_objective_returns=384,
        union_evidence_sha256=sha((output / 'UNION_EVIDENCE.json').read_bytes()),
        old120_references_sha256=sha((output / 'OLD120_POSITIVE_REFERENCES.json').read_bytes()),
        prefix_blank_reduction='amax', natural_weight=NATURAL_WEIGHT,
        original_B27_NO_PROMOTION_reauthenticated=True, B27_metrics_used_to_choose_parameters=False,
        original_B27_failure_certification=report['original_B27_failure_certification'],
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
        same_job_matched_control=False, architecture=b24.baseline.architecture_contract(),
        source_execution_allowed=False, new_primary_allowed=False, replication_allowed=False,
        candidate_engineering_allowed=False, promotion_allowed=False, product_released=False,
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
        design(False); assessment_binding(False); opening(False)
        print('PASS: B28 implementation remains sealed until exact full original failure and new assessment')
    elif args.mode == 'fetch': fetch(args.inputs, args.prepared_parent)
    elif args.mode == 'consume': print(json.dumps(consume(args.inputs, args.prepared_parent), sort_keys=True))
    elif args.mode == 'preflight': print(json.dumps(source_preflight(args.inputs)[0], sort_keys=True))
    else: print(json.dumps(execute(args.inputs, args.output, args.prepared_parent), sort_keys=True))
