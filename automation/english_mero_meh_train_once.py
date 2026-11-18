"""Bounded public Actions execution of one new synthetic expansion experiment.

Default mode is source-only. A completed, exact synthetic-source artifact and
an exact future main predecessor must be bound before execution is enabled.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'jiying2007/kws-data'
SCHEMA = 'english-mero-meh-training-once-v1'
WORKER_PATH = 'automation/english_mero_meh_train_once.py'
CORE_PATH = 'automation/english_mero_meh_training_core.py'
TEST_PATH = 'automation/test_english_mero_meh_train_once.py'
PLAN_PATH = 'automation/english_mero_meh_train_plan.json'
WORKFLOW_PATH = '.github/workflows/english-mero-meh-train-once.yml'
HELPER_PATH = 'automation/english_hardneg_increment_once.py'
HELPER_PIN = {'bytes': 69681, 'sha256': 'ac89f4564bb3f93661beb02f01f7766616322a39096c94a4bdac84560f5a9ec1'}
ORIGINAL_PLAN_PATH = 'automation/english_hardneg_increment_plan.json'
IDENTITY_PATH = 'automation/english_source_identity.py'
INTAKE_PATH = 'automation/english_typed_training_intake.py'
BINDER_PATH = 'automation/bind_english_mero_meh_training.py'
FRESH_SOURCE = {
    'campaign_id': 'english-mero-meh-corpus-v1',
    'workflow_path': '.github/workflows/english-mero-meh-once.yml',
    'artifact_prefix': 'english-mero-meh-corpus-',
    'policy_path': 'frozen_source/automation/english_mero_meh_policy.py',
    'files_schema': 'english-mero-meh-corpus-files-v1',
    'maximum_planned_rows': 144}
LEGACY_LIBRI_IDS = ['1272', '1462', '1673', '174', '1919', '1988', '1993', '2035',
    '2078', '2086', '2277', '2412', '2428', '251', '2803', '2902', '3000', '3081',
    '3170', '3536', '3576', '3752', '3853', '422', '5338', '5536', '5694', '5895',
    '6241', '6295', '6313', '6319', '6345', '652', '777', '7850', '7976', '8297', '84', '8842']
BACKBONE_PINS = {
    'causal_embedding.py': 'bb623ca627c0394f900638c90eee3a1ec71ff6901fa59713319fd3132de84f37',
    'weak_embedding_contract.py': 'ba9567b0303b1883f4d4834fec2b499abb8697b8fe5773519f07105f5d55388c'}
ASSETS = {
    'melspectrogram.onnx': {'bytes': 1087958, 'sha256': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'},
    'embedding_model.onnx': {'bytes': 1326578, 'sha256': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'}}
MAXIMUM_SECONDS = 3600
MAX_ZIP = 512 * 1024**2
MAX_EXPANDED = 1024**3
MAX_MEMBER = 128 * 1024**2


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def get_helper():
    path = ROOT/HELPER_PATH
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('REGULAR_EXISTING_HELPER_REQUIRED')
    raw = path.read_bytes()
    if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != HELPER_PIN:
        raise ValueError('EXACT_IMMUTABLE_HELPER_REQUIRED')
    return load_module(path, 'synthetic192_immutable_io_helpers')


frozen = get_helper()
need, sha, identity = frozen.need, frozen.sha, frozen.identity
regular, loads, canonical = frozen.regular, frozen.loads, frozen.canonical
put, once, file_identity = frozen.put, frozen.once, frozen.file_identity
cleanup_group, close_interrupted_receipts = frozen.cleanup_group, frozen.close_interrupted_receipts
FALSE_FLAGS = frozen.FALSE_FLAGS


def expected_plan(worker_sha, source_pins, *, main_before=None, enabled=False,
                  corpus_source=None, fresh_execution_plan=None,
                  corpus_run_conclusion=None, corpus_combine_job=None,
                  intake_binding=None, fresh_source_pins=None):
    return {'schema': SCHEMA, 'repository': REPO, 'enabled': enabled, 'main_before': main_before,
        'worker_sha256': worker_sha, 'source_pins': source_pins, 'sources': frozen.SOURCES,
        'base_npz': frozen.BASE_NPZ, 'base_bin': frozen.BASE_BIN,
        'corpus_source': corpus_source, 'fresh_execution_plan': fresh_execution_plan,
        'corpus_run_conclusion': corpus_run_conclusion, 'corpus_combine_job': corpus_combine_job,
        'fresh_source_contract': FRESH_SOURCE,
        'intake_binding': intake_binding, 'fresh_source_pins': fresh_source_pins,
        'backbone_runtime_pins': BACKBONE_PINS,
        'backbone_assets': ASSETS, 'maximum_execution_seconds': MAXIMUM_SECONDS,
        'maximum_zip_bytes': MAX_ZIP, 'maximum_expanded_bytes': MAX_EXPANDED,
        'maximum_member_bytes': MAX_MEMBER, 'numeric_processes': 1, 'native_head_processes': 1,
        'new_tts_asr_calls': 0, 'new_compilations': 0, 'retry_count': 0,
        'checkpoint_selection': 'NEW_DEVELOPMENT_ONLY_BEFORE_CALIBRATION_AND_LOCKED_TEST',
        'current_default_model_replaced': False, 'full_pcm_new_head': 'NOT_RUN',
        'legacy_replay_scope': 'PRIOR_CAPABILITY_RETENTION_NOT_NEW_PRONUNCIATION_AUTHORITY',
        'physical_ssc305_prerequisite': False, 'human_recording_prerequisite': False,
        'trigger': 'PUBLIC_MAIN_EXACT_SUCCESSOR_PUSH_RUN_ATTEMPT1'}


def workflow_contract(raw, plan):
    text = raw.decode()
    wanted = ("if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && "
        "github.run_attempt == 1 && github.event.before == '"+plan['main_before']+"' }}") if plan['enabled'] else 'if: ${{ false }}'
    need(wanted in text and text.count('github.event.before') == int(plan['enabled']), 'WORKFLOW_EXACT_ENABLED_BINDING')
    need('include-hidden-files: true' in text and 'python-version: "3.12"' in text,
        'WORKFLOW_COMPLETE_SOURCE_AND_RUNTIME')
    need('NUMEXPR_NUM_THREADS: "1"' in text and 'PYTHONHASHSEED: "20261117"' in text,
        'WORKFLOW_REPRODUCIBLE_ENVIRONMENT')


def source_preflight(plan):
    pins = plan.get('source_pins', {})
    original = loads(regular(ROOT/ORIGINAL_PLAN_PATH))
    need(all(original['source_pins'].get(k) == v for k, v in frozen.SOURCE_PINS.items()), 'IMMUTABLE_HISTORICAL_SOURCE_CLOSURE')
    required = set(original['source_pins']) | {ORIGINAL_PLAN_PATH, HELPER_PATH, CORE_PATH, TEST_PATH, WORKFLOW_PATH, IDENTITY_PATH, INTAKE_PATH, BINDER_PATH}
    need(set(pins) == required, 'EXACT_NEW_SOURCE_CLOSURE')
    need(all(pins[k] == v for k, v in original['source_pins'].items()), 'HISTORICAL_SOURCE_PINS_UNCHANGED')
    for name, pin in pins.items():
        need(identity(regular(ROOT/name)) == pin, 'SOURCE_PIN:'+name)
    need(pins[HELPER_PATH] == HELPER_PIN, 'IMMUTABLE_IO_HELPER_PIN')
    need(type(plan.get('enabled')) is bool, 'EXPLICIT_PLAN_STATE')
    if plan['enabled']:
        need(re.fullmatch('[0-9a-f]{40}', plan.get('main_before') or ''), 'EXACT_NEXT_MAIN_BASE')
        source = plan.get('corpus_source'); cp = plan.get('fresh_execution_plan')
        need(type(source) is dict and set(source) == {'run_id', 'artifact_id', 'name', 'head_sha', 'bytes', 'sha256'}, 'ACTUAL_CORPUS_SOURCE_PIN')
        need(all(type(source[k]) is int and source[k] > 0 for k in ('run_id', 'artifact_id', 'bytes'))
            and source['bytes'] <= MAX_ZIP and re.fullmatch('[0-9a-f]{40}', source['head_sha'])
            and re.fullmatch('[0-9a-f]{64}', source['sha256']), 'CORPUS_METADATA_PIN_TYPES')
        need(source['name'] == FRESH_SOURCE['artifact_prefix']+str(source['run_id']), 'MERO_MEH_CORPUS_ARTIFACT_REQUIRED')
        need(type(cp) is dict and set(cp) == {'bytes', 'sha256'} and type(cp['bytes']) is int
            and 0 < cp['bytes'] <= MAX_MEMBER and re.fullmatch('[0-9a-f]{64}', cp['sha256']), 'FROZEN_SOURCE_EXECUTION_PLAN')
        need(plan.get('corpus_run_conclusion') in ('success', 'failure'), 'COMPLETED_SOURCE_RUN_CONCLUSION')
        job = plan.get('corpus_combine_job')
        need(type(job) is dict and set(job) == {'job_id', 'name'} and type(job['job_id']) is int
            and job['job_id'] > 0 and job['name'] == 'combine', 'EXACT_COMPLETED_COMBINE_JOB')
        binding = plan.get('intake_binding')
        core = load_module(ROOT/CORE_PATH, 'mero_meh_preflight_core')
        need(core.SOURCE_IDENTITY_CANDIDATE_ENABLED is True,
             'ACTUAL_BOUND_NEW_CORE_ENABLED')
        core.typed_intake.validate_binding(binding, core.BOUND_INTAKE_BINDING_SHA256)
        need(binding['campaign_id'] == FRESH_SOURCE['campaign_id']
            and len(binding['planned_row_ids']) <= FRESH_SOURCE['maximum_planned_rows']
            and binding['legacy_libri_ids'] == LEGACY_LIBRI_IDS
            and binding['parent_context_weak_policy'] is False,
            'EXACT_NEW_SHORT_PCM_INTAKE_SCOPE')
        need(binding['producer_run'] == {'repository': REPO, 'run_id': source['run_id'],
            'commit': source['head_sha'], 'run_attempt': 1, 'event': 'push'}
            and binding['source_artifact'] == {'id': source['artifact_id'],
                **{k: source[k] for k in ('name', 'bytes', 'sha256')}},
            'INTAKE_ACTUAL_ARTIFACT_MATCHES_TRAINER_PIN')
        fresh_pins = plan.get('fresh_source_pins')
        need(type(fresh_pins) is dict and FRESH_SOURCE['policy_path'] in fresh_pins,
             'FROZEN_SOURCE_POLICY_CLOSURE_REQUIRED')
        for name, pin in fresh_pins.items():
            frozen.safe_member(name)
            need(name.startswith('frozen_source/') and type(pin) is dict
                and set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int
                and 0 < pin['bytes'] <= MAX_MEMBER
                and type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}', pin['sha256']),
                'FROZEN_SOURCE_FILE_PIN')
        need(fresh_pins[FRESH_SOURCE['policy_path']]['sha256'] == binding['admission_policy_sha256'],
             'INTAKE_PINNED_POLICY_MATCH')
    else:
        need(plan.get('main_before') is None and plan.get('corpus_source') is None
            and plan.get('fresh_execution_plan') is None and plan.get('corpus_run_conclusion') is None
            and plan.get('corpus_combine_job') is None and plan.get('intake_binding') is None
            and plan.get('fresh_source_pins') is None, 'DISABLED_UNBOUND_PLAN')
        core = load_module(ROOT/CORE_PATH, 'mero_meh_disabled_core')
        need(core.SOURCE_IDENTITY_CANDIDATE_ENABLED is False
            and core.BOUND_INTAKE_BINDING_SHA256 is None, 'DISABLED_CORE_UNBOUND')
    need(plan == expected_plan(sha(regular(ROOT/WORKER_PATH)), pins, main_before=plan['main_before'],
        enabled=plan['enabled'], corpus_source=plan['corpus_source'], fresh_execution_plan=plan['fresh_execution_plan'],
        corpus_run_conclusion=plan['corpus_run_conclusion'], corpus_combine_job=plan['corpus_combine_job'],
        intake_binding=plan['intake_binding'], fresh_source_pins=plan['fresh_source_pins']), 'EXACT_NEW_PLAN')
    workflow_contract(regular(ROOT/WORKFLOW_PATH), plan)
    admission = loads(regular(ROOT/frozen.ADMISSION_PATH)); review = loads(regular(ROOT/frozen.REVIEW_PATH))
    frozen.validate_owner_review(review, admission)
    return admission


def execution_identity(plan, env=None, event=None, head=None, parent=None):
    env = os.environ if env is None else env
    need(plan['enabled'] is True, 'DISABLED_UNBOUND_EXPERIMENT')
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPO
        and env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main'
        and env.get('GITHUB_RUN_ATTEMPT') == '1', 'PUBLIC_MAIN_ONE_ATTEMPT_ONLY')
    run_id, commit = env.get('GITHUB_RUN_ID', ''), env.get('GITHUB_SHA', '')
    need(re.fullmatch('[0-9]+', run_id) and re.fullmatch('[0-9a-f]{40}', commit), 'ACTUAL_ACTION_IDENTITY')
    if event is None: event = loads(regular(Path(env['GITHUB_EVENT_PATH'])))
    need(event.get('before') == plan['main_before'] and event.get('after') == commit
        and event.get('repository', {}).get('private') is False
        and event.get('repository', {}).get('full_name') == REPO, 'EXACT_PUBLIC_PUSH_SUCCESSOR')
    if head is None: head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    if parent is None: parent = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD^1'], text=True, timeout=10).strip()
    need(head == commit and parent == plan['main_before'], 'EXACT_CHECKOUT_PARENT')
    return {'repository': REPO, 'run_id': run_id, 'head_sha': commit, 'main_before': plan['main_before'],
        'run_attempt': 1, 'event': 'push', 'public_repository': True}


def extract_archive(archive_path, destination, source):
    """Validate and extract every member of the actual source artifact once."""
    need(file_identity(archive_path, limit=MAX_ZIP) == {k: source[k] for k in ('bytes', 'sha256')},
         'EXACT_NEW_CORPUS_ZIP')
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and len(names) <= 20000, 'CORPUS_UNIQUE_BOUNDED_MEMBERS')
        need(sum(i.file_size for i in z.infolist()) <= MAX_EXPANDED, 'CORPUS_EXPANDED_BOUND')
        for item in z.infolist():
            frozen.safe_member(item.filename)
            mode = (item.external_attr >> 16) & 0xffff
            need(not item.is_dir() and not item.flag_bits & 1 and item.file_size <= MAX_MEMBER
                 and not stat.S_ISLNK(mode), 'REGULAR_UNENCRYPTED_BOUNDED_MEMBER')
        manifest = loads(z.read('FILES.json'))
        need(manifest.get('schema') == FRESH_SOURCE['files_schema']
            and type(manifest.get('files')) is list, 'FRESH_MANIFEST_SCHEMA')
        pins = {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in manifest['files']}
        need(len(pins) == len(manifest['files']) and set(names) == set(pins)|{'FILES.json'},
             'FRESH_EXACT_ARCHIVE_CLOSURE')
        for name, pin in pins.items():
            raw = z.read(name)
            need(identity(raw) == pin, 'FRESH_PAYLOAD:'+name)
            put(destination/name, raw)
        put(destination/'FILES.json', z.read('FILES.json'))
    return pins


@contextlib.contextmanager
def frozen_source_imports(root, pins):
    """Resolve all source helpers from authenticated archived bytes, not a cache."""
    directory = Path(root)/'frozen_source/automation'
    names = {Path(name).stem for name in pins
        if Path(name).parent.as_posix() == 'frozen_source/automation' and name.endswith('.py')}
    previous = {name: sys.modules[name] for name in names if name in sys.modules}
    old_path = list(sys.path)
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def _load_source_component(name):
    return importlib.import_module('english_mero_meh_' + name)


def byte_evidence_readback(root, source_plan, corpus, evidence, pins):
    """Reconstruct source and observer evidence from their actual retained bytes."""
    for suffix in ('generate.py', 'observe.py'):
        need('frozen_source/automation/english_mero_meh_'+suffix in pins,
             'AUTHENTICATED_BYTE_READER_REQUIRED')
    source_pins = source_plan.get('source_files_sha256')
    need(type(source_pins) is dict and source_pins, 'PRODUCER_SOURCE_PINS_REQUIRED')
    for name, expected in source_pins.items():
        need(pins.get('frozen_source/'+name, {}).get('sha256') == expected,
             'PRODUCER_FROZEN_SOURCE_MATCH:'+name)
    # The producer serializes run IDs as strings. Preserve its exact observed
    # dictionary after source_packet has authenticated it against external IDs.
    run = corpus['run']
    with frozen_source_imports(root, pins):
        generate = _load_source_component('generate')
        observe = _load_source_component('observe')
        source_status = generate.source_readback(root, source_plan, run)
        need(source_status.get('source_ready') is True, 'COMPLETE_BYTE_VERIFIED_TRAINING_SOURCE')
        packets = {provider: observe.readback(provider=provider, source_root=root,
            observer_root=Path(root)/'observers'/provider, plan=source_plan, run_identity=run)
            for provider in ('qwen', 'whisper', 'parakeet', 'phoneme')}
    providers = ('qwen', 'whisper', 'parakeet', 'phoneme')
    observations = {p: {r['id']: r for r in packets[p]['rows']} for p in providers}
    rows = [{'id': row['id'], 'observations': {p: {k: v for k, v in observations[p][row['id']].items()
        if k != 'id'} for p in providers}} for row in corpus['rows']]
    exact = [source_status.get('tts_attempts')] + [packets[p]['summary']['actual_forward_attempts'] for p in providers]
    lower = [source_status.get('tts_attempts_lower_bound', exact[0] if type(exact[0]) is int else 0)]
    lower += [packets[p]['summary']['attempts_lower_bound'] for p in providers]
    total = sum(exact) if all(type(x) is int for x in exact) else None
    expected = {'schema': 'english-mero-meh-machine-evidence-v1',
        'campaign_id': FRESH_SOURCE['campaign_id'], 'run': run, 'denominator': 144,
        'source_readback': source_status, 'observer_summaries': {p: packets[p]['summary'] for p in providers},
        'actual_total_model_calls': total, 'actual_total_model_calls_lower_bound': sum(lower),
        'maximum_total_model_calls': 720, 'missing_terminal_is_zero': False, 'retry_count': 0, 'rows': rows}
    for key, value in expected.items():
        need(canonical(evidence.get(key)) == canonical(value), 'BYTE_RECOMPUTED_MACHINE_EVIDENCE:'+key)
    need(total is None or total <= 720, 'ACTUAL_PRODUCER_CALL_BUDGET')
    return {'source_manifest_sha256': source_status['source_manifest_sha256'],
        'source_rows': len(corpus['rows']), 'observer_summaries': expected['observer_summaries'],
        'actual_source_model_calls': total, 'all_machine_observations_reconstructed': True,
        'new_model_calls': 0, 'resampling_reexecuted': False}


def source_packet(root, plan, *, expected_binding_sha256=None):
    """Recompute the bound same-PCM policy; this function imports no numeric runtime."""
    source, binding = plan['corpus_source'], plan['intake_binding']
    need(identity(regular(root/'EXECUTION_PLAN.json')) == plan['fresh_execution_plan'],
         'SOURCE_JOB_PLAN_IDENTITY')
    manifest = loads(regular(root/'FILES.json'))
    archived = {r['path']: {'bytes': r['bytes'], 'sha256': r['sha256']}
                for r in manifest['files'] if r['path'].startswith('frozen_source/')}
    need(archived == plan['fresh_source_pins'], 'COMPLETE_FROZEN_SOURCE_CLOSURE')
    for name, pin in archived.items():
        need(identity(regular(root/name)) == pin, 'ACTUAL_SOURCE_CODE:'+name)
    corpus = loads(regular(root/'CORPUS_MANIFEST.json'))
    admission = loads(regular(root/'ADMISSION.json'))
    evidence = loads(regular(root/'MACHINE_EVIDENCE.json'))
    contract = loads(regular(root/'SOURCE_IDENTITY_CONTRACT.json'))
    exclusion = loads(regular(root/'EXCLUSION_EVIDENCE.json'))
    source_plan = loads(regular(root/'EXECUTION_PLAN.json'))
    result = loads(regular(root/'RESULT.json'))
    need(source_plan.get('campaign_id') == corpus.get('campaign_id') == corpus.get('source_design_id')
        == FRESH_SOURCE['campaign_id'], 'EXACT_NEW_SOURCE_CAMPAIGN')
    need(result.get('stage') == 'combine' and result.get('status') == 'MACHINE_EVIDENCE_PACKET_RETAINED',
         'COMPLETED_SOURCE_OBSERVATION_PACKET')
    for record in (corpus, result):
        run = record.get('run', {})
        need(run.get('repository') == REPO and run.get('commit') == source['head_sha']
            and str(run.get('run_id')) == str(source['run_id']) and str(run.get('run_attempt')) == '1'
            and run.get('event') == 'push', 'CORPUS_EXECUTION_BINDING')
    core = load_module(ROOT/CORE_PATH, 'mero_meh_packet_core')
    digest = core.typed_intake.digest
    need(digest(exclusion) == binding['exclusion_evidence_sha256']
        and exclusion.get('excluded_identities') == binding['excluded_identities']
        and {44202856867, 44208480336}.issubset(set(exclusion.get('source_runs', []))),
        'COMPLETE_CONTEXT64_AND_POCKET12_EXCLUSION_EVIDENCE')
    need(len(corpus['rows']) <= FRESH_SOURCE['maximum_planned_rows'], 'NEW_CORPUS_BOUND')
    pcm_roles = {}
    for row in corpus['rows']:
        if row.get('generation_status') != 'success':
            continue
        name = row.get('pcm_path'); frozen.safe_member(name)
        raw = regular(root/name)
        need(type(row.get('pcm_samples')) is int and len(raw) == 2*row['pcm_samples']
            and sha(raw) == row.get('pcm_sha256'), 'ACTUAL_SOURCE_PCM_BYTES')
        need(row['pcm_sha256'] not in pcm_roles or pcm_roles[row['pcm_sha256']] == row['role'],
             'PCM_HASH_CANNOT_CROSS_FROZEN_ROLES')
        pcm_roles[row['pcm_sha256']] = row['role']
    byte_proof = byte_evidence_readback(root, source_plan, corpus, evidence, archived)
    policy_path = root/FRESH_SOURCE['policy_path']
    need(sha(regular(policy_path)) == binding['admission_policy_sha256'],
         'ACTUAL_ADMISSION_POLICY_SHA256')
    # Every local dependency is in the authenticated frozen_source closure.
    sys.path.insert(0, str(policy_path.parent))
    try:
        policy = load_module(policy_path, 'mero_meh_actual_admission')
        regenerated = policy.build_admission(corpus, evidence, contract)
    finally:
        sys.path.pop(0)
    need(regenerated == admission and admission.get('training_snapshot_allowed') is True,
         'RECOMPUTED_MACHINE_ADMISSION')
    proof = core.typed_intake.validate_packet(corpus, admission, contract, evidence, regenerated,
        binding, expected_binding_sha256=(core.BOUND_INTAKE_BINDING_SHA256
            if expected_binding_sha256 is None else expected_binding_sha256),
        normalize_libri=core.libri_identity)
    return {'admission': admission, 'machine_evidence': evidence,
        'recomputed_admission': regenerated, 'identity_proof': proof, 'byte_readback': byte_proof}


def extract_fresh(archive_path, destination, plan):
    extract_archive(archive_path, destination, plan['corpus_source'])
    return source_packet(destination, plan)


def verify_combine_job(job, plan):
    source, selected = plan['corpus_source'], plan['corpus_combine_job']
    need(job.get('id') == selected['job_id'] and job.get('name') == selected['name'] == 'combine'
        and job.get('run_id') == source['run_id'] and job.get('head_sha') == source['head_sha']
        and job.get('status') == 'completed' and job.get('conclusion') == 'success',
        'ACTUAL_SUCCESSFUL_COMBINE_JOB')
    return {'job_id': job['id'], 'name': job['name'], 'run_id': job['run_id'],
        'head_sha': job['head_sha'], 'status': job['status'], 'conclusion': job['conclusion']}


def result_boundary(binding):
    return {'schema': SCHEMA, **binding, 'status': 'NOT_COMPLETED', 'passed': False,
        'numerical_execution_completed': False, 'research_retention_allowed': False,
        'current_default_model_replaced': False, 'human_gold': False,
        'strong_english_fsmn_ctc': False, 'ssc305_qualification': False,
        'model_promotion_allowed': False,
        'source_design_id': FRESH_SOURCE['campaign_id'],
        'physical_ssc305_prerequisite': False, 'human_recording_prerequisite': False,
        'full_pcm_new_head': 'NOT_RUN', 'arm_new_head': 'NOT_RUN', 'ssc305': 'NOT_RUN',
        'independent_real_far': 'NOT_MEASURED', 'independent_real_frr': 'NOT_MEASURED'}


def numeric(args, plan):
    output = args.output.absolute(); binding = execution_identity(plan)
    need(os.environ.get('KWS_MERO_MEH_NUMERIC_PARENT_PID') == str(os.getppid()) and os.getpgrp() == os.getppid(), 'SUPERVISED_NUMERIC_CHILD_ONLY')
    parent = loads(regular(output/'evidence/WORKER_STARTED.json'))
    need(parent['pid'] == os.getppid() and parent['binding'] == binding, 'NUMERIC_PARENT_BINDING')
    once(output/'evidence/NUMERIC_STARTED.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'binding': binding, 'retry_count': 0})
    core = load_module(ROOT/CORE_PATH, 'synthetic192_numeric_core')
    progress = {k: 0 for k in ('new_embedding_calls_started', 'new_embedding_calls_completed',
        'new_onnx_forwards_started', 'new_onnx_forwards_completed', 'optimizer_updates_started',
        'optimizer_updates_completed', 'training_bag_forwards_started', 'training_bag_forwards_completed',
        'epochs_completed', 'fresh_head_case_evaluations', 'calibration_head_cases', 'locked_test_head_cases', 'legacy_final_head_cases')}
    def save_progress(): frozen.checkpoint(output/'evidence/NUMERIC_PROGRESS.json', progress)
    save_progress(); result = {'passed': False, 'status': 'NUMERIC_FAILED_NO_RETRY'}
    try:
        packet = source_packet(output/'inputs/fresh', plan)
        result = core.train_and_evaluate(output, frozen, progress, save_progress,
            intake_binding=plan['intake_binding'], machine_evidence=packet['machine_evidence'],
            recomputed_admission=packet['recomputed_admission'])
    except BaseException as error:
        result['error'] = type(error).__name__+':'+str(error)
        put(output/'evidence/NUMERIC_TRACEBACK.txt', traceback.format_exc().encode())
    finally:
        result['actual_progress'] = progress
        once(output/'evidence/NUMERIC_RESULT.json', result)
    return 0 if result['passed'] else 1


def child(args, plan, old_admission):
    output = args.output.absolute(); binding = execution_identity(plan)
    need(os.environ.get('KWS_MERO_MEH_PARENT_PID') == str(os.getppid()) and os.getpgrp() == os.getpid(), 'SUPERVISED_PUBLIC_WORKER_ONLY')
    started = loads(regular(output/'STARTED.json'))
    need(started['binding'] == binding and started['worker_sha256'] == plan['worker_sha256'], 'WORKER_SOURCE_BINDING')
    runner = frozen.Runner(output, started['monotonic_deadline']); result = result_boundary(binding); prepared = None
    once(output/'evidence/WORKER_STARTED.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'binding': binding, 'retry_count': 0})
    try:
        for name in ('source', 'inputs', 'cache', 'predictions', 'runtime', 'models', 'downloads', 'assets', 'fresh'):
            (output/name).mkdir()
        with runner.stage('source-binding'):
            run = loads(regular(runner.command('current-run', ['gh', 'api', 'repos/'+REPO+'/actions/runs/'+binding['run_id']], auth=True)))
            need(run.get('id') == int(binding['run_id']) and run.get('head_sha') == binding['head_sha']
                and run.get('run_attempt') == 1 and run.get('repository', {}).get('private') is False, 'CURRENT_PUBLIC_RUN')
            for name in plan['source_pins']: put(output/'source'/name, regular(ROOT/name))
            for name in (WORKER_PATH, PLAN_PATH): put(output/'source'/name, regular(ROOT/name))
        with runner.stage('authenticate-frozen-and-fresh-inputs'):
            quality = runner.download('quality'); natural = runner.download('natural')
            prepared = frozen.prepare_saved_inputs(ROOT, quality, natural, old_admission, output)
            corpus_zip = runner.download('synthetic192', pin=plan['corpus_source'],
                expected_conclusion=plan['corpus_run_conclusion'])
            corpus_run = loads(regular(output/'evidence/synthetic192-run.stdout'))
            need(corpus_run.get('path') == FRESH_SOURCE['workflow_path'], 'EXACT_NEW_MERO_MEH_SOURCE_WORKFLOW')
            need(corpus_run.get('event') == 'push' and corpus_run.get('head_branch') == 'main',
                 'ACTUAL_PUBLIC_SOURCE_MAIN_PUSH')
            combine = loads(regular(runner.command('synthetic192-combine-job', ['gh', 'api',
                'repos/'+REPO+'/actions/jobs/'+str(plan['corpus_combine_job']['job_id'])], auth=True)))
            combine = verify_combine_job(combine, plan)
            packet = extract_fresh(corpus_zip, output/'inputs/fresh', plan)
            admission = packet['admission']
            once(output/'evidence/FRESH_INPUTS_VERIFIED.json', {'source': plan['corpus_source'],
                'source_run_conclusion': plan['corpus_run_conclusion'], 'successful_combine': combine,
                'admission': file_identity(output/'inputs/fresh/ADMISSION.json'), 'tiers': admission.get('tier_counts'),
                'balanced_groups': admission.get('balanced_groups_by_role'),
                'byte_readback': packet['byte_readback'], 'new_model_calls': 0})
            # Reuse exact backbone assets already retained in the coverage source;
            # no second upstream model download or new feature recipe.
            for name, expected in BACKBONE_PINS.items():
                raw = prepared['quality'].read('sources/'+name)
                need(sha(raw) == expected, 'ORIGINAL_FEATURE_RUNTIME:'+name)
                put(output/'runtime/backbone'/name, raw)
            for name, expected in ASSETS.items():
                raw = prepared['quality'].read('models/'+name)
                need(identity(raw) == expected, 'ORIGINAL_FROZEN_BACKBONE:'+name)
                put(output/'assets'/name, raw)
        with runner.stage('numeric-runtime-preparation'):
            frozen.prepare_dependencies(runner)
            runner.command('install-frozen-onnx-runtime', [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-input', '--retries', '0', '--timeout', '30', '--only-binary=:all:', 'onnx==1.17.0', 'onnxruntime==1.20.1'], timeout=180)
        with runner.stage('synthetic-expansion-convergence-and-frozen-evaluation'):
            os.environ['KWS_MERO_MEH_NUMERIC_PARENT_PID'] = str(os.getpid())
            runner.command('synthetic192-numeric', [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--_numeric', '--output', str(output)], timeout=2700, kind='numeric')
            measured = loads(regular(output/'evidence/NUMERIC_RESULT.json'))
            need(measured['passed'] is True, 'NUMERICAL_COMPLETION')
        with runner.stage('native-cached-head-parity'):
            raw = regular(output/'inputs/coverage/runtime/head-native')
            need(raw[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', raw, 18)[0] == 62, 'PINNED_NATIVE_ELF')
            consumer = output/'runtime/head-native'; put(consumer, raw); consumer.chmod(0o500)
            stdout = runner.command('head-native', [str(consumer), str(output/'models/synthetic-head.bin'),
                str(output/'runtime/candidate-head-stream.bin')], timeout=240, kind='native', maximum_stdout=MAX_MEMBER)
            refs = [loads(line) for line in regular(output/'runtime/reference-frames.jsonl', limit=MAX_MEMBER).splitlines()]
            core = load_module(ROOT/CORE_PATH, 'synthetic192_result_core')
            parity = core.verify_native(regular(stdout, limit=MAX_MEMBER), refs)
            need(file_identity(output/'models/synthetic-head.bin') == measured['candidate']['bin'] and
                file_identity(output/'models/synthetic-head.npz') == measured['candidate']['npz'] and regular(consumer) == raw,
                'SELECTED_WEIGHTS_AND_CONSUMER_UNCHANGED')
            once(output/'evidence/HEAD_NATIVE_PARITY.json', {**parity, 'consumer': identity(raw),
                'candidate_bin': measured['candidate']['bin'], 'candidate_npz': measured['candidate']['npz']})
        need(runner.counts == {'numeric': 1, 'native': 1}, 'EXACT_TWO_EXECUTION_PROCESSES')
        gates = {**measured['gates'], 'native_saved_npz_bin_parity': True}; passed = all(gates.values())
        result.update(status='RETAINED_RESEARCH_MERO_MEH_HEAD' if passed else 'REJECTED_MERO_MEH_QUALITY_GATE',
            passed=passed, numerical_execution_completed=True, candidate=measured['candidate'],
            gates=gates, research_retention_allowed=passed, selected_epoch=measured['selected_epoch'],
            actual_epochs=measured['actual_epochs'], optimizer_updates=measured['optimizer_updates'],
            stop_reason=measured['stop_reason'], progress=measured['progress'],
            locked_test_opened=measured['locked_test_opened'], corpus_source=plan['corpus_source'])
    except BaseException as error:
        result['error'] = type(error).__name__+':'+str(error)
        put(output/'evidence/FAILURE_TRACEBACK.txt', traceback.format_exc().encode())
        if (output/'evidence/NUMERIC_PROGRESS.json').exists(): result['actual_partial_progress'] = loads(regular(output/'evidence/NUMERIC_PROGRESS.json'))
    finally:
        if prepared is not None: prepared['quality'].close(); prepared['natural'].close()
        result['process_attempts'] = runner.counts
        once(output/'evidence/RESULT.json', result)
    return 0 if result['passed'] else 1


def supervise(args, plan):
    binding = execution_identity(plan)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_SUPERVISED_OUTPUT_ONLY')
    output.mkdir(mode=0o700)
    (output / 'evidence').mkdir()
    started, process, timed_out, error, cleanup = time.monotonic(), None, False, None, {}
    deadline = started + 3600
    once(output / 'STARTED.json', {'schema': SCHEMA + '/started', 'binding': binding,
        'parent_pid': os.getpid(), 'worker_sha256': plan['worker_sha256'], 'monotonic_started': started,
        'monotonic_deadline': deadline, 'maximum_seconds': 3600, 'retry_count': 0, 'network_calls_at_start': 0})
    env = dict(os.environ, KWS_MERO_MEH_PARENT_PID=str(os.getpid()))
    argv = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--_worker', '--output', str(output)]
    def cancelled(signum, _frame):
        raise InterruptedError('SUPERVISOR_SIGNAL_' + str(signum))
    handlers = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / 'evidence/SUPERVISOR.stdout').open('xb') as out, (output / 'evidence/SUPERVISOR.stderr').open('xb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=env, cwd=str(ROOT), start_new_session=True)
            once(output / 'evidence/WORKER_LAUNCH.json', {'pid': process.pid, 'process_group': process.pid,
                 'argv': argv, 'attempt': 1, 'retry_count': 0})
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise TimeoutError('SUPERVISOR_3600_SECONDS')
                time.sleep(0.1)
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process is not None:
            try:
                cleanup = cleanup_group(process.pid)
            except BaseException as exc:
                cleanup = {'error': type(exc).__name__ + ':' + str(exc), 'descendants_found': True}
                error = error or 'PROCESS_GROUP_CLEANUP_FAILED'
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                error = error or 'WORKER_NOT_REAPED'
        receipt = {'schema': SCHEMA + '/supervisor', 'returncode': None if process is None else process.returncode,
            'timed_out': timed_out, 'error': error, 'wall_seconds': time.monotonic() - started,
            'retry_count': 0, 'cleanup': cleanup, **binding}
        worker_result = None
        if (output / 'evidence/RESULT.json').exists():
            try:
                worker_result = loads(regular(output / 'evidence/RESULT.json'))
            except (ValueError, OSError) as exc:
                receipt['error'] = 'WORKER_RESULT_UNREADABLE:' + str(exc)
        receipt['passed'] = (process is not None and process.returncode == 0 and not timed_out and
            receipt['error'] is None and not cleanup.get('descendants_found') and
            type(worker_result) is dict and worker_result.get('passed') is True and
            all(worker_result.get(k) == v for k, v in binding.items()))
        receipt['partial_receipt_errors'] = close_interrupted_receipts(output, receipt)
        if receipt['partial_receipt_errors']:
            receipt['passed'] = False
        once(output / 'evidence/SUPERVISOR_EXIT.json', receipt)
        if type(worker_result) is not dict:
            fallback = result_boundary(binding)
            fallback.update(error='SUPERVISOR_TERMINATED_BEFORE_WORKER_RESULT', supervisor=receipt,
                            numeric_or_native_process_attempts=None if receipt['partial_receipt_errors'] else
                            sum(loads(regular(p)).get('kind') in ('numeric', 'native') for p in (output / 'evidence').glob('*.started.json')),
                            optimizer_updates_completed='UNKNOWN_IF_INTERRUPTED')
            result_path = output / 'evidence/RESULT.json'
            if result_path.exists():
                fallback['preserved_damaged_worker_result'] = file_identity(result_path)
                result_path = output / 'evidence/RESULT_SUPERVISOR_FAILURE.json'
            once(result_path, fallback)
        files = {p.relative_to(output).as_posix(): file_identity(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and 'downloads' not in p.relative_to(output).parts}
        need(sum(pin['bytes'] for pin in files.values()) <= MAX_EXPANDED and all(pin['bytes'] <= MAX_MEMBER for pin in files.values()), 'TRAINING_ARTIFACT_SIZE_BOUND')
        once(output / 'MANIFEST.json', {'schema': SCHEMA + '/artifact', 'files': files,
             'downloads_excluded': True, 'supervised_success': receipt['passed'],
             'authoritative_completion_receipt': 'evidence/SUPERVISOR_EXIT.json', **dict.fromkeys(FALSE_FLAGS, False)})
    print(canonical(receipt).decode(), end='')
    return 0 if receipt['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--_numeric', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    plan = loads(regular(ROOT/PLAN_PATH)); admission = source_preflight(plan)
    if args.execute or args._worker or args._numeric:
        need(args.output is not None, 'EXPLICIT_EXECUTION_OUTPUT')
        if args._numeric: return numeric(args, plan)
        return child(args, plan, admission) if args._worker else supervise(args, plan)
    need(args.output is None, 'SOURCE_PREFLIGHT_HAS_NO_NUMERIC_OUTPUT')
    print(canonical({'status': 'SOURCE_CONTRACT_ONLY', 'enabled': plan['enabled'],
        'new_model_calls': 0, 'new_network_calls': 0, 'numeric_configuration_count': 1,
        'maximum_seconds': MAXIMUM_SECONDS, 'corpus_pin_bound': plan['corpus_source'] is not None,
        'human_recording_prerequisite': False, 'physical_ssc305_prerequisite': False}).decode())
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())

