"""One separately declared observation of terminal final144 source evidence.

Default: source contract only, no ML. Old campaign/guards/receipts are immutable.
An audit does not enable execution. A new protected first-main successor and
complete pins are required for the four bounded audio-only observers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data' / 'automation'
sys.path.insert(0, str(AUTOMATION))
import final144_runner as original
import final144_recipe as recipe

need, sha, canonical = recipe.need, recipe.sha, recipe.canonical
regular, loads, once = original.regular, original.loads, original.once
POLICY = 'english-terminal144-observation-v1'
PIN_SCHEMA = 'english-terminal144-observation-pins-v1'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
SOURCE = {'source_run_id': 43816869386,
    'source_head_sha': '8286ee743a2690d168c3e56f5b31ccc64ae490ed', 'source_run_attempt': 1,
    'source_runner_sha256': '16814d7cc89d2fe744bae51b9672aca45f360e2307d3f6f76f57fe1e2546aae2',
    'source_recipe_sha256': 'd1fe7973324bf35f9fcd1c20472786e121c18bb3e4bdbe8967953a0f88243b83',
    'source_execution_pins_sha256': '49f27437c9e3b2458f4c3bb987100acfe72395cd71455ed5fda5256cf24dab55',
    'source_model_lock_sha256': '06abd719b12f4191e1552194fc386d5a0b813318d7f8dc4dac719c589c59ac32'}
XLSR_LOCK = '71a30ee3d506cd10911a531c7a97918757ad7c924fbb01316e0f126fc0dffce1'
MAX_ZIP = 32 * 1024**2
MAX_TOTAL_ZIP = 128 * 1024**2
VALID = 'VALID_UNOBSERVED_SOURCE'
SOURCE_STATUSES = {'GENERATED_UNADMITTED', 'FAILED_NO_RETRY', 'STARTED_UNKNOWN_NO_RETRY',
                   'NOT_STARTED', 'INVALID_SOURCE_RECEIPT'}


def hex_id(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def pins_template():
    return {'schema': PIN_SCHEMA, 'enabled': False, 'next_main_before': None,
        'code_sha256': None, **SOURCE, 'source_artifacts': [],
        'terminal_evidence_canonical_sha256': None, 'source_ledger_canonical_sha256': None,
        'valid_cells': None, 'max_calls_per_provider': None, 'xlsr_model_lock_sha256': XLSR_LOCK,
        'new_tts_calls': 0, 'optimizer_steps': 0, 'no_retry': True, **recipe.FLAGS}


def validate_pins(pins, *, frozen=False):
    need(pins.get('schema') == PIN_SCHEMA and type(pins.get('enabled')) is bool,
         'OBSERVATION_PIN_SCHEMA')
    need(all(pins.get(k) == v for k, v in SOURCE.items()) and
         all(pins.get(k) is False for k in recipe.FLAGS) and pins.get('no_retry') is True and
         pins.get('new_tts_calls') == pins.get('optimizer_steps') == 0 and
         pins.get('xlsr_model_lock_sha256') == XLSR_LOCK, 'FIXED_SOURCE_AND_RESEARCH_SCOPE')
    entries = pins.get('source_artifacts')
    need(type(entries) is list and len(entries) == 9 and
         [p.get('shard_index') for p in entries] == list(range(9)), 'EXACT_NINE_SOURCE_ARTIFACT_PINS')
    for pin in entries:
        need(set(pin) == {'shard_index', 'artifact_id', 'zip_bytes', 'zip_sha256',
                         'source_plan_sha256', 'source_result_sha256', 'model_lock_sha256'} and
             type(pin['artifact_id']) is int and pin['artifact_id'] > 0 and
             type(pin['zip_bytes']) is int and 0 < pin['zip_bytes'] <= MAX_ZIP and
             all(hex_id(pin[k]) for k in ('zip_sha256', 'source_plan_sha256',
                 'source_result_sha256', 'model_lock_sha256')) and
             pin['model_lock_sha256'] == SOURCE['source_model_lock_sha256'], 'SOURCE_ARTIFACT_PIN')
    need(len({p['artifact_id'] for p in entries}) == 9 and
         sum(p['zip_bytes'] for p in entries) <= MAX_TOTAL_ZIP, 'ARTIFACT_IDS_OR_TOTAL_SIZE')
    if frozen:
        need(hex_id(pins.get('terminal_evidence_canonical_sha256')) and
             hex_id(pins.get('source_ledger_canonical_sha256')), 'UNFROZEN_SOURCE_EVIDENCE')
        cells = pins.get('valid_cells')
        need(type(cells) is list and 0 < len(cells) <= 144 and
             type(pins.get('max_calls_per_provider')) is int and
             pins['max_calls_per_provider'] == len(cells), 'UNFROZEN_VALID_COHORT_BUDGET')
        for cell in cells:
            need(set(cell) == {'cell_id', 'wav_sha256', 'pcm_sha256', 'frames'} and
                 type(cell['cell_id']) is str and hex_id(cell['wav_sha256']) and
                 hex_id(cell['pcm_sha256']) and type(cell['frames']) is int and
                 0 < cell['frames'] <= 192000, 'FROZEN_VALID_CELL')
        need(all(len({c[k] for c in cells}) == len(cells) for k in
                 ('cell_id', 'wav_sha256', 'pcm_sha256')), 'FROZEN_COHORT_NOT_UNIQUE')


def execution_guard(pins, event, env):
    """Pure guard; tests pass dictionaries, never forge the process environment."""
    validate_pins(pins, frozen=True)
    need(pins['enabled'] is True and hex_id(pins.get('next_main_before'), 40) and
         hex_id(pins.get('code_sha256')) and
         sha(regular(Path(__file__), 256 * 1024)) == pins['code_sha256'], 'EXECUTION_DISABLED_OR_CODE_UNPINNED')
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and str(env.get('GITHUB_RUN_ID', '')).isdigit() and
         env.get('GITHUB_RUN_ID') != str(SOURCE['source_run_id']) and
         event.get('before') == pins['next_main_before'] and
         event.get('after') == env.get('GITHUB_SHA') and hex_id(event.get('after'), 40) and
         event['after'] != event['before'] and event['after'] != SOURCE['source_head_sha'] and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'FIRST_NEW_PUBLIC_MAIN_ATTEMPT_ONLY')


def source_identity(simulated=False):
    # Never call original.identity/shard_plan or read the current Actions identity.
    return {'schema': original.POLICY, 'recipe_sha256': SOURCE['source_recipe_sha256'],
        'code_sha256': SOURCE['source_runner_sha256'],
        'execution_pins_sha256': SOURCE['source_execution_pins_sha256'],
        'reused_module_blobs': original.MODULES, 'github_sha': SOURCE['source_head_sha'],
        'run_id': str(SOURCE['source_run_id']), 'run_attempt': '1', 'test_backend': simulated,
        'source_labels_are_intents_only': True, 'acoustic_review': 'UNCERTAIN_MACHINE_OBSERVATIONS_ONLY',
        'retry_count': 0, **recipe.FLAGS}


def terminal_evidence(bundle):
    run = loads(regular(bundle / 'run.json', 4 * 1024**2))
    payload = loads(regular(bundle / 'jobs.json', 4 * 1024**2))
    need(run.get('id') == SOURCE['source_run_id'] and run.get('head_sha') == SOURCE['source_head_sha'] and
         run.get('run_attempt') == 1 and run.get('event') == 'push' and run.get('head_branch') == 'main' and
         run.get('status') == 'completed' and run.get('conclusion') == 'failure' and
         run.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         run.get('repository', {}).get('private') is False, 'ORIGINAL_RUN_NOT_TERMINAL_FAILURE')
    jobs = payload.get('jobs')
    need(type(jobs) is list and payload.get('total_count') == len(jobs) and
         len({j['name'] for j in jobs}) == len(jobs), 'COMPLETE_ORIGINAL_JOB_LIST')
    names = {j['name'] for j in jobs}
    observed = {n for n in names if n == 'observe' or re.fullmatch(r'observe \((qwen|whisper|parakeet|phones)\)', n)}
    need(observed == {'observe'} or observed == {'observe (' + p + ')' for p in PROVIDERS},
         'ORIGINAL_OBSERVE_JOB_EVIDENCE_REQUIRED')
    expected = {'source-contract', 'evidence', 'train-weak-intent'} | observed | {
        'generate (%02d)' % i for i in range(9)}
    need(names == expected, 'EXACT_ORIGINAL_JOB_ROSTER')
    rows = []
    for job in jobs:
        need(job.get('run_id') == SOURCE['source_run_id'] and job.get('run_attempt') == 1 and
             job.get('head_sha') == SOURCE['source_head_sha'] and job.get('status') == 'completed',
             'ORIGINAL_JOB_NOT_SAME_TERMINAL_RUN')
        if job['name'] in observed | {'train-weak-intent'}:
            steps = job.get('steps')
            need(job.get('conclusion') == 'skipped' and type(steps) is list and
                 all(s.get('conclusion') == 'skipped' for s in steps), 'ORIGINAL_OBSERVER_OR_TRAIN_CONSUMED')
        rows.append({k: job[k] for k in ('id', 'name', 'run_id', 'run_attempt', 'head_sha', 'status', 'conclusion')})
    return {'run_id': run['id'], 'head_sha': run['head_sha'], 'run_attempt': 1,
            'status': 'completed', 'conclusion': 'failure', 'jobs': sorted(rows, key=lambda j: j['name']),
            'original_observers_and_training_actually_skipped': True}


def extract_shard(archive, destination, cell_ids):
    allowed = {'started.json', 'source-plan.json', 'model-lock.json', 'result.json',
               'capture-stage.json', 'capture/plan-lock.json'}
    required = allowed - {'capture-stage.json'}
    for cid in cell_ids:
        allowed.update('capture/cells/' + cid + '/' + name for name in
                       ('started.json', 'completed.json', 'failed.json', 'audio-16k.wav', 'native-f32.wav'))
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        need(len(names) == len(set(names)) and required <= set(names) <= allowed and
             sum(m.file_size for m in members) <= 64 * 1024**2, 'SOURCE_ZIP_ROSTER_OR_SIZE')
        for member in members:
            path = PurePosixPath(member.filename)
            need(not path.is_absolute() and '..' not in path.parts and str(path) == member.filename and
                 '\\' not in member.filename and not member.is_dir() and not member.flag_bits & 1 and
                 stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG) and
                 0 < member.file_size <= 4 * 1024**2, 'SOURCE_ZIP_UNSAFE_MEMBER')
        destination.mkdir(mode=0o700)
        for member in members:
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(z.read(member))


def cohort(ledger):
    return [{k: row[k] for k in ('cell_id', 'wav_sha256', 'pcm_sha256', 'frames')}
            for row in ledger['rows'] if row['source_disposition'] == VALID]


def read_sources(bundle, pins, workspace, *, allow_simulated=False):
    """Zero-model archival readback; only a new caller-owned scratch tree is written."""
    validate_pins(pins)
    bundle, workspace = Path(bundle), Path(workspace)
    need(bundle.is_dir() and not any(p.is_symlink() for p in (bundle, *bundle.parents)) and
         not workspace.exists() and not any(p.is_symlink() for p in (workspace, *workspace.parents)),
         'SOURCE_BUNDLE_OR_FRESH_SCRATCH')
    need({p.name for p in bundle.glob('*.zip')} == {'shard-%02d.zip' % i for i in range(9)},
         'EXACT_NINE_ORIGINAL_ZIPS')
    need(sha(regular(AUTOMATION / 'final144_runner.py', 256 * 1024)) == SOURCE['source_runner_sha256'] and
         sha(canonical(recipe.recipe())) == SOURCE['source_recipe_sha256'], 'ORIGINAL_READER_OR_RECIPE_CHANGED')
    original.check_modules()
    terminal = terminal_evidence(bundle)
    terminal_sha = sha(canonical(terminal))
    if pins.get('terminal_evidence_canonical_sha256') is not None:
        need(terminal_sha == pins['terminal_evidence_canonical_sha256'], 'TERMINAL_EVIDENCE_CHANGED')
    artifacts = loads(regular(bundle / 'artifacts.json', 4 * 1024**2))
    artifacts = artifacts.get('artifacts', [])
    need(type(artifacts) is list, 'ARTIFACT_API_LIST')
    workspace.mkdir(mode=0o700)
    whole, rows, bindings, wavs, simulations = recipe.recipe(), [], [], {}, set()
    for pin in pins['source_artifacts']:
        index = pin['shard_index']
        name = 'shard-%02d' % index
        matches = [a for a in artifacts if a.get('name') == name]
        need(len(matches) == 1, 'UNIQUE_SOURCE_ARTIFACT_METADATA')
        meta = matches[0]
        run = meta.get('workflow_run', {})
        need(meta.get('id') == pin['artifact_id'] and meta.get('size_in_bytes') == pin['zip_bytes'] and
             meta.get('digest') == 'sha256:' + pin['zip_sha256'] and meta.get('expired') is False and
             run.get('id') == SOURCE['source_run_id'] and run.get('head_sha') == SOURCE['source_head_sha'] and
             run.get('head_branch') == 'main', 'SOURCE_ARTIFACT_IDENTITY')
        archive = bundle / (name + '.zip')
        raw_zip = regular(archive, MAX_ZIP)
        need(len(raw_zip) == pin['zip_bytes'] and sha(raw_zip) == pin['zip_sha256'], 'SOURCE_ZIP_BYTES')
        cells = whole['cells'][index * 16:(index + 1) * 16]
        root = workspace / name
        extract_shard(archive, root, [c['cell_id'] for c in cells])
        raw_plan = regular(root / 'source-plan.json', 2 * 1024**2)
        raw_result = regular(root / 'result.json', 2 * 1024**2)
        raw_lock = regular(root / 'model-lock.json', 1024**2)
        need(sha(raw_plan) == pin['source_plan_sha256'] and sha(raw_result) == pin['source_result_sha256'] and
             sha(raw_lock) == pin['model_lock_sha256'], 'ORIGINAL_RECEIPT_BYTES')
        plan, result, lock = loads(raw_plan), loads(raw_result), loads(raw_lock)
        started = loads(regular(root / 'started.json', 65536))
        simulated = started.get('test_backend') is True
        need(allow_simulated or not simulated, 'SIMULATED_SOURCE_REJECTED')
        simulations.add(simulated)
        identity = source_identity(simulated)
        base = {**identity, 'shard_index': index, 'shard_count': 9}
        for value in (started, result):
            need(canonical({k: value.get(k) for k in base}) == canonical(base), 'ORIGINAL_SOURCE_IDENTITY')
        need(started.get('operation') == 'generate' and started.get('call_budget') == 16 and
             result.get('status') in ({'COMPLETE', 'INCOMPLETE_NO_RETRY'} |
                                      ({'SIMULATED_COMPLETE'} if simulated else set())) and
             result.get('planned_audio') == 16 and result.get('global_planned_audio') == 144 and
             result.get('asr_calls') == result.get('phoneme_calls') == result.get('optimizer_steps') == 0 and
             result.get('source_plan_sha256') == pin['source_plan_sha256'] and
             result.get('model_lock_sha256') == pin['model_lock_sha256'], 'ORIGINAL_SOURCE_COUNTERS')
        need(lock.get('model_id') == recipe.MODEL and lock.get('revision') == recipe.REVISION,
             'ORIGINAL_TTS_MODEL_IDENTITY')
        counts = {'cells': 16, 'train': sum(c['role'] == 'train' for c in cells),
            'development': sum(c['role'] == 'development' for c in cells),
            'wake_candidates': sum(c['intent'] == 'wake-candidate' for c in cells),
            'ordinary_negative_candidates': sum(c['intent'] == 'ordinary-negative-candidate' for c in cells),
            'independent_humans': 0, 'independent_test_cells': 0}
        expected = {**whole, 'cells': cells, 'counts': counts, 'global_counts': whole['counts'],
            'global_recipe_sha256': SOURCE['source_recipe_sha256'], 'shard_index': index, 'shard_count': 9,
            'max_tts_calls': 16, 'capture_stage_budgets': [16], 'capture_identity': identity,
            'model_lock_sha256': pin['model_lock_sha256']}
        need(canonical(plan) == canonical(expected), 'ORIGINAL_PARTITION_PLAN')
        capture_lock = loads(regular(root / 'capture/plan-lock.json', 65536))
        need(capture_lock == {'policy': 'en-market-qwen-customvoice-capture-v1',
             'plan_sha256': sha(canonical(plan)), 'model_calls_admitted_to_training': False},
             'ORIGINAL_CAPTURE_PLAN_LOCK')
        stored = result.get('rows')
        need(type(stored) is list and len(stored) == 16 and [r.get('cell_id') for r in stored] ==
             [c['cell_id'] for c in cells] and all(r.get('status') in SOURCE_STATUSES and
             type(r.get('attempted')) is bool for r in stored), 'ORIGINAL_144_SOURCE_ROW_ROSTER')
        need(result['tts_attempts'] == sum(r['attempted'] for r in stored) and
             result['generated_audio'] == sum(r['status'] == 'GENERATED_UNADMITTED' for r in stored),
             'ORIGINAL_REPORTED_COUNTERS_CHANGED')
        completed_status = 'SIMULATED_COMPLETE' if simulated else 'COMPLETE'
        job = next(j for j in terminal['jobs'] if j['name'] == 'generate (%02d)' % index)
        need(job['conclusion'] != 'success' or result['status'] == completed_status,
             'SUCCESS_JOB_WITH_INCOMPLETE_SOURCE')
        if result['status'] == completed_status:
            need(result['tts_attempts'] == result['generated_audio'] == 16 and
                 result.get('capture_stages_completed') == 1 and result.get('error_type') is None,
                 'FALSE_ORIGINAL_SOURCE_COMPLETION')
        stage_path = root / 'capture-stage.json'
        need(type(result.get('capture_stages_completed')) is int and
             result['capture_stages_completed'] == int(stage_path.exists()), 'ORIGINAL_CAPTURE_STAGE_COUNT')
        if stage_path.exists():
            stage = loads(regular(stage_path, 65536))
            need(canonical({k: stage.get(k) for k in base}) == canonical(base) and
                 stage.get('policy') == 'en-market-qwen-customvoice-capture-v1' and
                 stage.get('attempts_this_call') == stage.get('total_started') == result['tts_attempts'] and
                 stage.get('total_completed') == result['generated_audio'] and stage.get('asr_calls') == 0,
                 'ORIGINAL_CAPTURE_STAGE_ACCOUNTING')
        checked = original.capture_rows(plan, root / 'capture')
        for cell, old, current in zip(cells, stored, checked):
            need(canonical({k: old.get(k) for k in cell}) == canonical(cell), 'ORIGINAL_ROW_ROLE_OR_RECIPE')
            equal = canonical(old) == canonical(current)
            disposition = VALID if equal and current['status'] == 'GENERATED_UNADMITTED' else (
                current['status'] if equal else 'INVALID_SOURCE_RECEIPT')
            row = {**cell, 'source_shard': index, 'source_status': old['status'],
                'source_attempted': old['attempted'], 'source_error_type': old.get('error_type'),
                'readback_status': current['status'], 'readback_error_type': current.get('error_type'),
                'readback_equals_original': equal, 'source_disposition': disposition,
                'wav_sha256': current.get('wav_sha256'), 'pcm_sha256': current.get('pcm_sha256'),
                'frames': current.get('frames')}
            rows.append(row)
            wavs[cell['cell_id']] = root / 'capture/cells' / cell['cell_id'] / 'audio-16k.wav'
        bindings.append({**pin, 'original_status': result['status'],
                         'recorded_tts_attempts': result['tts_attempts'], 'recorded_generated_audio': result['generated_audio']})
    need(len(simulations) == 1 and len(rows) == 144, 'MIXED_OR_INCOMPLETE_SOURCE_IDENTITY')
    duplicates = Counter(r['pcm_sha256'] for r in rows if r['source_disposition'] == VALID)
    for row in rows:
        if row['source_disposition'] == VALID and duplicates[row['pcm_sha256']] > 1:
            row['source_disposition'] = 'EXCLUDED_DUPLICATE_PCM'
    ledger = {'schema': POLICY + '/source-ledger', **SOURCE, 'source_denominator': 144,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'original_run_conclusion': 'failure',
        'terminal_evidence_canonical_sha256': terminal_sha, 'terminal_evidence': terminal,
        'source_artifacts': bindings, 'test_backend': simulations.pop(), 'rows': rows,
        'source_labels_are_intents_only': True, **recipe.FLAGS}
    frozen_cells = cohort(ledger)
    ledger['valid_cells'] = frozen_cells
    ledger['valid_count'] = len(frozen_cells)
    ledger['source_dispositions'] = dict(Counter(r['source_disposition'] for r in rows))
    if pins.get('source_ledger_canonical_sha256') is not None:
        need(sha(canonical(ledger)) == pins['source_ledger_canonical_sha256'], 'FROZEN_SOURCE_LEDGER_CHANGED')
    if pins.get('valid_cells') is not None:
        need(canonical(frozen_cells) == canonical(pins['valid_cells']) and
             pins['max_calls_per_provider'] == len(frozen_cells), 'FROZEN_COHORT_CHANGED')
    return ledger, [(row, wavs[row['cell_id']]) for row in rows if row['source_disposition'] == VALID]


def observer_identity(pins, ledger, simulated, env):
    return {'schema': POLICY, 'source_run_id': SOURCE['source_run_id'],
        'source_head_sha': SOURCE['source_head_sha'], 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
        'observer_run_id': env.get('GITHUB_RUN_ID'), 'observer_github_sha': env.get('GITHUB_SHA'),
        'observer_run_attempt': env.get('GITHUB_RUN_ATTEMPT'), 'test_backend': simulated,
        'code_sha256': sha(regular(Path(__file__), 256 * 1024)), 'pins_sha256': sha(canonical(pins)),
        'source_ledger_canonical_sha256': sha(canonical(ledger)), 'source_denominator': 144,
        'call_budget': ledger['valid_count'], 'source_labels_are_intents_only': True,
        'new_tts_calls': 0, 'optimizer_steps': 0, 'retry_count': 0, **recipe.FLAGS}


def receipt_identity(base, row):
    return {**base, **{k: row[k] for k in ('cell_id', 'wav_sha256', 'pcm_sha256', 'frames')},
            'attempt': 1}


def error_message(exc):
    """Keep a bounded diagnosis, without credentials or signed URL parameters."""
    message = str(exc)
    for key, value in os.environ.items():
        if len(value) >= 4 and re.search(r'TOKEN|PASSWORD|SECRET|CREDENTIAL|AUTHORIZATION|API_KEY', key, re.I):
            message = message.replace(value, '[REDACTED]')
    message = re.sub(r'(https?://[^\s?\"\'<>]+)\?[^\s\"\'<>]*', r'\1?[REDACTED]', message)
    message = re.sub(r'(?i)\b(token|api[_-]?key|authorization|password|secret|signature)\s*[:=]\s*'
                     r'(?:Bearer\s+)?[^\s,;]+', r'\1=[REDACTED]', message)
    return message[:2000]


def output_value(provider, value, frames):
    # Empty decoded output is still a raw observation, never an acoustic label.
    if provider == 'phones':
        tokens, actual_frames = value
        need(type(actual_frames) is int and actual_frames == frames and type(tokens) is list and
             len(tokens) <= 300 and all(type(t) is str and 0 < len(t) <= 40 and
             not any(ord(c) < 32 for c in t) for t in tokens), 'PHONE_OUTPUT')
        return {'raw_ipa_tokens': tokens}
    need(type(value) is str and len(value) <= 512 and
         not any(ord(c) < 32 for c in value), 'ASR_OUTPUT')
    return {'text': value}


def read_observation_rows(output, base, ledger):
    rows = []
    allowed = {r['cell_id'] for r in ledger['rows'] if r['source_disposition'] == VALID}
    parent = output / 'cells'
    if parent.exists():
        need(parent.is_dir() and not parent.is_symlink() and
             {p.name for p in parent.iterdir()} <= allowed, 'OBSERVER_EXTRA_CELL')
    for source_row in ledger['rows']:
        row = {**source_row, 'test_backend': base['test_backend'], 'observation_attempted': False,
               'observation_status': 'NOT_STARTED' if source_row['source_disposition'] == VALID
                                      else 'NOT_ELIGIBLE_SOURCE'}
        folder = parent / source_row['cell_id']
        if folder.exists():
            expected = receipt_identity(base, source_row)
            started = loads(regular(folder / 'started.json', 65536))
            need(started == expected, 'OBSERVER_STARTED_BINDING')
            row.update(observation_attempted=True, observation_status='STARTED_UNKNOWN_NO_RETRY')
            complete, failed = folder / 'completed.json', folder / 'failed.json'
            need(not (complete.exists() and failed.exists()), 'CONFLICTING_OBSERVER_TERMINALS')
            if complete.exists() or failed.exists():
                terminal = loads(regular(complete if complete.exists() else failed, 65536))
                need(canonical({k: terminal.get(k) for k in expected}) == canonical(expected),
                     'OBSERVER_TERMINAL_BINDING')
                if complete.exists():
                    need(terminal.get('observation_status') == 'COMPLETE' and
                         terminal.get('decoder_completed') is True, 'OBSERVER_COMPLETE_RECEIPT')
                    value = ((terminal.get('raw_ipa_tokens'), source_row['frames'])
                             if base['provider'] == 'phones' else terminal.get('text'))
                    row.update(output_value(base['provider'], value, source_row['frames']),
                               observation_status='COMPLETE', decoder_completed=True)
                else:
                    need(terminal.get('observation_status') == 'FAILED_NO_RETRY' and
                         type(terminal.get('error_type')) is str and
                         type(terminal.get('error_message')) is str and len(terminal['error_message']) <= 2000,
                         'OBSERVER_FAILED_RECEIPT')
                    row.update(observation_status='FAILED_NO_RETRY', error_type=terminal['error_type'],
                               error_message=terminal['error_message'])
        rows.append(row)
    return rows


def observe(provider, bundle, pins, output, *, backend=None):
    """A backend injection is explicitly simulated and never used by the CLI."""
    need(provider in PROVIDERS, 'OBSERVER_PROVIDER')
    simulated = backend is not None
    validate_pins(pins, frozen=True)
    if not simulated:
        event = loads(regular(Path(os.environ.get('GITHUB_EVENT_PATH', '')), 8 * 1024**2))
        execution_guard(pins, event, os.environ)
    with tempfile.TemporaryDirectory(prefix='terminal144-observe-') as private_name:
        private = Path(private_name)
        ledger, eligible = read_sources(bundle, pins, private / 'source', allow_simulated=simulated)
        output = original.new_output(Path(output))
        base = {**observer_identity(pins, ledger, simulated, os.environ), 'provider': provider}
        once(output / 'SOURCE_LEDGER.json', ledger)
        once(output / 'started.json', base)
        (output / 'cells').mkdir()
        attempts, error, detail, lock_sha = 0, None, None, None
        try:
            model_root = private / 'model'
            model_root.mkdir(mode=0o700)
            if simulated:
                lock = {'test_backend': True, 'not_a_real_model': True}
            elif provider == 'phones':
                xlsr = importlib.import_module('english_final_ipa_xlsr_once')
                predict, lock = xlsr.load_runtime(model_root)
                backend = lambda wav, _: predict(wav)
            else:
                seven = importlib.import_module('english_seven_voice_asr_once')
                backend = seven._load(provider, model_root)
                lock = {'provenance': list(seven.PROVENANCE[provider]),
                        'loader_git_blob': original.MODULES['english_seven_voice_asr_once']}
            once(output / 'model-lock.json', lock)
            lock_sha = sha(regular(output / 'model-lock.json', 1024**2))
            need(simulated or provider != 'phones' or lock_sha == pins['xlsr_model_lock_sha256'],
                 'FROZEN_XLSR_MODEL_LOCK')
            for row, wav in eligible:
                need(attempts < pins['max_calls_per_provider'], 'FROZEN_OBSERVER_CALL_BUDGET')
                raw = regular(wav, 2 * 1024**2)
                need(sha(raw) == row['wav_sha256'], 'SOURCE_CHANGED_BEFORE_OBSERVATION')
                folder = output / 'cells' / row['cell_id']
                folder.mkdir(mode=0o700)
                anonymous = private / 'audio.wav'
                with anonymous.open('xb') as stream:
                    stream.write(raw)
                receipt = receipt_identity(base, row)
                once(folder / 'started.json', receipt)
                attempts += 1
                try:
                    value = output_value(provider, backend(anonymous, folder), row['frames'])
                    once(folder / 'completed.json', {**receipt, **value,
                        'observation_status': 'COMPLETE', 'decoder_completed': True})
                except Exception as exc:
                    once(folder / 'failed.json', {**receipt, 'observation_status': 'FAILED_NO_RETRY',
                        'error_type': type(exc).__name__, 'error_message': error_message(exc)})
                finally:
                    anonymous.unlink()
        except BaseException as exc:
            error, detail = type(exc).__name__, error_message(exc)
            if not isinstance(exc, Exception):
                raise
        finally:
            rows = read_observation_rows(output, base, ledger)
            completed = sum(r['observation_status'] == 'COMPLETE' for r in rows)
            need(attempts == sum(r['observation_attempted'] for r in rows), 'OBSERVER_ATTEMPT_RECEIPTS')
            status = 'OBSERVATIONS_COMPLETE' if completed == len(eligible) and error is None else 'INCOMPLETE_NO_RETRY'
            if simulated and status == 'OBSERVATIONS_COMPLETE':
                status = 'SIMULATED_COMPLETE'
            result = {**base, 'status': status, 'attempts': attempts, 'completed': completed,
                'model_lock_sha256': lock_sha, 'error_type': error, 'error_message': detail, 'rows': rows}
            once(output / 'summary.json', result)
    return result


def audit(bundle, pins, output):
    with tempfile.TemporaryDirectory(prefix='terminal144-audit-') as private:
        ledger, _ = read_sources(bundle, pins, Path(private) / 'source')
    output = original.new_output(Path(output))
    once(output / 'SOURCE_LEDGER.json', ledger)
    result = {'schema': POLICY, 'status': 'SOURCE_EVIDENCE_ONLY', **SOURCE,
        'source_denominator': 144, 'valid_count': ledger['valid_count'], 'valid_cells': ledger['valid_cells'],
        'source_ledger_canonical_sha256': sha(canonical(ledger)),
        'terminal_evidence_canonical_sha256': ledger['terminal_evidence_canonical_sha256'],
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'model_calls': 0,
        'new_tts_calls': 0, 'optimizer_steps': 0, 'execution_enabled_by_audit': False, **recipe.FLAGS}
    once(output / 'RESULT.json', result)
    return result


def verify_observer_lock(root, provider, simulated, expected_sha=None):
    raw = regular(root / 'model-lock.json', 1024**2)
    if expected_sha is not None:
        need(hex_id(expected_sha) and sha(raw) == expected_sha, 'OBSERVER_MODEL_LOCK_BYTES')
    if simulated:
        need(loads(raw) == {'test_backend': True, 'not_a_real_model': True}, 'TEST_MODEL_LOCK')
    elif provider == 'phones':
        need(sha(raw) == XLSR_LOCK, 'FROZEN_XLSR_MODEL_LOCK')
    else:
        seven = importlib.import_module('english_seven_voice_asr_once')
        need(loads(raw) == {'provenance': list(seven.PROVENANCE[provider]),
             'loader_git_blob': original.MODULES['english_seven_voice_asr_once']}, 'FROZEN_ASR_MODEL_LOCK')
    return sha(raw)


def combine(bundle, observation_parent, pins, *, allow_simulated=False):
    validate_pins(pins, frozen=True)
    with tempfile.TemporaryDirectory(prefix='terminal144-combine-') as private:
        ledger, _ = read_sources(bundle, pins, Path(private) / 'source', allow_simulated=allow_simulated)
    values, identities, summaries, model_locks = {}, set(), {}, {}
    for provider in PROVIDERS:
        root = Path(observation_parent) / ('observer-' + provider)
        if not (root / 'started.json').exists():
            values[provider] = {'status': 'UNKNOWN_MISSING_PROVIDER_OUTPUT', 'attempts': None, 'completed': None,
                'rows': [{**r, 'observation_attempted': None,
                    'observation_status': 'UNKNOWN_MISSING_PROVIDER_OUTPUT' if r['source_disposition'] == VALID
                                          else 'NOT_ELIGIBLE_SOURCE'} for r in ledger['rows']]}
            summaries[provider] = None
            model_locks[provider] = None
            continue
        started = loads(regular(root / 'started.json', 65536))
        simulated = started.get('test_backend') is True
        need(allow_simulated or not simulated, 'SIMULATED_OBSERVER_REJECTED')
        env = {'GITHUB_RUN_ID': started.get('observer_run_id'), 'GITHUB_SHA': started.get('observer_github_sha'),
               'GITHUB_RUN_ATTEMPT': started.get('observer_run_attempt')}
        if not simulated:
            need(type(env['GITHUB_RUN_ID']) is str and env['GITHUB_RUN_ID'].isdigit() and
                 env['GITHUB_RUN_ID'] != str(SOURCE['source_run_id']) and
                 hex_id(env['GITHUB_SHA'], 40) and env['GITHUB_SHA'] != SOURCE['source_head_sha'] and
                 env['GITHUB_RUN_ATTEMPT'] == '1', 'NEW_OBSERVER_IDENTITY')
        identities.add((env['GITHUB_RUN_ID'], env['GITHUB_SHA'], env['GITHUB_RUN_ATTEMPT'], simulated))
        base = {**observer_identity(pins, ledger, simulated, env), 'provider': provider}
        need(started == base and loads(regular(root / 'SOURCE_LEDGER.json', 2 * 1024**2)) == ledger,
             'OBSERVER_SOURCE_OR_CODE_BINDING')
        rows = read_observation_rows(root, base, ledger)
        path = root / 'summary.json'
        if path.exists():
            raw = regular(path, 2 * 1024**2)
            summary = loads(raw)
            need(canonical({k: summary.get(k) for k in base}) == canonical(base) and
                 summary.get('rows') == rows and
                 summary.get('attempts') == sum(r['observation_attempted'] for r in rows) and
                 summary.get('completed') == sum(r['observation_status'] == 'COMPLETE' for r in rows) and
                 summary['attempts'] <= ledger['valid_count'], 'OBSERVER_SUMMARY_RECEIPTS')
            if summary['status'] in ('OBSERVATIONS_COMPLETE', 'SIMULATED_COMPLETE'):
                need(summary['completed'] == ledger['valid_count'] and summary.get('error_type') is None,
                     'FALSE_OBSERVER_COMPLETION')
            need(summary.get('status') in ('OBSERVATIONS_COMPLETE', 'SIMULATED_COMPLETE', 'INCOMPLETE_NO_RETRY') and
                 (simulated or summary['status'] != 'SIMULATED_COMPLETE'), 'OBSERVER_TERMINAL_STATUS')
            values[provider], summaries[provider] = summary, sha(raw)
        else:
            values[provider] = {'status': 'UNKNOWN_MISSING_PROVIDER_TERMINAL',
                'attempts': sum(r['observation_attempted'] for r in rows),
                'completed': sum(r['observation_status'] == 'COMPLETE' for r in rows), 'rows': rows}
            summaries[provider] = None
        if values[provider]['attempts']:
            expected_sha = values[provider].get('model_lock_sha256')
            need(not path.exists() or hex_id(expected_sha), 'ATTEMPTED_MODEL_LOCK_NOT_BOUND')
            model_locks[provider] = verify_observer_lock(root, provider, simulated, expected_sha)
        else:
            model_locks[provider] = None
    need(len(identities) <= 1, 'CROSS_RUN_OR_SIMULATED_OBSERVERS')
    complete = all(v['status'] in ('OBSERVATIONS_COMPLETE', 'SIMULATED_COMPLETE') for v in values.values())
    simulated = ledger['test_backend'] or any(x[3] for x in identities)
    rows = [{**row, 'observations': {p: values[p]['rows'][i] for p in PROVIDERS}}
            for i, row in enumerate(ledger['rows'])]
    return {'schema': POLICY + '/combined', **SOURCE,
        'status': ('SIMULATED_COMPLETE' if simulated else 'FOUR_OBSERVERS_COMPLETE')
                  if complete else 'INCOMPLETE_NO_RETRY',
        'test_backend': simulated,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'source_denominator': 144,
        'valid_count': ledger['valid_count'], 'source_ledger_canonical_sha256': sha(canonical(ledger)),
        'observer_summaries_sha256': summaries,
        'observer_identity': dict(zip(('run_id', 'github_sha', 'run_attempt', 'test_backend'),
                                     next(iter(identities)))) if identities else None,
        'observer_model_locks_sha256': model_locks,
        'providers': {p: {k: v.get(k) for k in ('status', 'attempts', 'completed', 'error_type', 'error_message')}
                      for p, v in values.items()},
        'new_tts_calls': 0, 'optimizer_steps': 0, 'model_calls_by_combiner': 0,
        'source_labels_are_intents_only': True, **recipe.FLAGS, 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--template', action='store_true')
    modes.add_argument('--audit', action='store_true')
    modes.add_argument('--observe', choices=PROVIDERS)
    modes.add_argument('--combine', type=Path)
    parser.add_argument('--source-bundle', type=Path)
    parser.add_argument('--pins', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), indent=2, sort_keys=True))
        return 0
    if not (args.audit or args.observe or args.combine):
        need(args.source_bundle is None and args.output is None, 'SOURCE_ONLY_ARGUMENTS')
        print(json.dumps({'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0, 'enabled': False,
                          **SOURCE, **recipe.FLAGS}, sort_keys=True))
        return 0
    need(args.pins is not None and args.source_bundle is not None and args.output is not None,
         'EXPLICIT_PINS_SOURCE_OUTPUT_REQUIRED')
    pins = loads(regular(args.pins, 256 * 1024))
    if args.audit:
        value = audit(args.source_bundle, pins, args.output)
    elif args.observe:
        value = observe(args.observe, args.source_bundle, pins, args.output)
    else:
        need(not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents)),
             'CREATE_ONLY_COMBINED_OUTPUT')
        value = combine(args.source_bundle, args.combine, pins)
        once(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('rows', 'valid_cells')}, sort_keys=True))
    return 2 if value['status'] == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
