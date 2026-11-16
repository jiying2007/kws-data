"""Disabled-by-default successor for one terminal-source weak-intent experiment.

Read back the original failed 144-cell campaign and four saved complete observers.
No generation, recognition, pronunciation admission, or model execution on import.
Only a separately pinned first public-main push may perform the fixed 96 updates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data' / 'automation'
SCHEMA = 'english-terminal144-embedding-intent-research-v1'
PINS_SCHEMA = 'english-terminal144-training-pins-v1'
SUCCESS = 'EXECUTED_TERMINAL_SOURCE_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'
PIPELINE_COMMIT = '31238cafe3d6668874d446eae5dfa7c55028fb13'
LEGACY_SHA = '92b1097ca3c98f9071bd1cda3f8905015f355502d1d34ecfa613bf95b3083726'
OBSERVER_SHA = '67dac982beed528d8a867558e50222712b73a56a93c77e65e614020a37a2dd0d'
SELECTION_SHA = '27a18716b5c0e7f29a937eeedde085917c04d4d45f30db0d9115dc3c881a9fd5'
RECIPE_SHA = 'd1fe7973324bf35f9fcd1c20472786e121c18bb3e4bdbe8967953a0f88243b83'
SOURCE_IDENTITY = {'run_id': '43816869386', 'github_sha': '8286ee743a2690d168c3e56f5b31ccc64ae490ed',
                   'run_attempt': '1', 'conclusion': 'failure'}
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
MAX_OBSERVER_ZIP = 16 * 1024**2
SOURCE_EVIDENCE_FILES = ('STARTED.json', 'SOURCE_BINDING.json', 'OBSERVER_BINDING.json',
    'SOURCE_LEDGER.json', 'COMBINED_OBSERVATIONS.json', 'SELECTION.json',
    'TRAINING_PINS.json', 'OBSERVATION_PINS.json')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def hex_id(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def regular(path, limit=16 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
         path.stat().st_size <= limit, 'REGULAR_FILE_SIZE')
    raw = path.read_bytes()
    need(len(raw) <= limit, 'FILE_SIZE_CHANGED')
    return raw


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def read_json(path, limit=16 * 1024**2):
    return loads(regular(path, limit))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha(raw)


def load_module(name, filename, expected):
    path = AUTOMATION / filename
    need(sha(regular(path, 256 * 1024)) == expected, 'FROZEN_DEPENDENCY_CHANGED:' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contracts():
    old = load_module('_terminal_training_legacy', 'english_final144_embedding_once.py', LEGACY_SHA)
    observer = load_module('_terminal_training_observer', 'terminal144_observe_once.py', OBSERVER_SHA)
    need(sha(canonical(old.SELECTION)) == SELECTION_SHA and
         sha(canonical(observer.recipe.recipe())) == RECIPE_SHA, 'FIXED_SELECTION_AND_RECIPE')
    need(sha(regular(AUTOMATION / 'final144_runner.py', 256 * 1024)) ==
         observer.SOURCE['source_runner_sha256'], 'ORIGINAL_RUNNER_UNCHANGED')
    return old, observer


def pins_template():
    """No actual observer completion or cohort identity is guessed here."""
    return {'schema': PINS_SCHEMA, 'enabled': False, 'next_main_before': None, 'code_sha256': None,
        'pipeline_commit': PIPELINE_COMMIT, 'legacy_worker_sha256': LEGACY_SHA,
        'terminal_observer_code_sha256': OBSERVER_SHA, 'recipe_sha256': RECIPE_SHA,
        'selection_policy_sha256': SELECTION_SHA, 'source_identity': SOURCE_IDENTITY,
        'observation_pins_sha256': None, 'observer_run': None, 'observer_artifacts': None,
        'combined_artifact': None, 'observer_terminal_evidence_canonical_sha256': None,
        'source_ledger_canonical_sha256': None, 'valid_count': None, 'valid_cells_sha256': None,
        'selected_rows_sha256': None, 'source_denominator': 144, 'optimizer_updates': 96,
        'seed': 20261116, 'proposal_threshold': 0.8, 'proposal_confirm_embeddings': 3,
        'proposal_cooldown_embeddings': 13, 'development_tuning': False,
        'new_tts_asr_phone_calls': 0, 'no_retry': True, 'source_labels_are_intents_only': True,
        **dict.fromkeys(FALSE_FLAGS, False)}


def validate_pins(pins, *, execution=False):
    need(type(pins) is dict and pins.get('schema') == PINS_SCHEMA and
         type(pins.get('enabled')) is bool, 'TRAINING_PIN_SCHEMA')
    template = pins_template()
    dynamic = {'enabled', 'next_main_before', 'code_sha256', 'observation_pins_sha256', 'observer_run',
               'observer_artifacts', 'combined_artifact', 'observer_terminal_evidence_canonical_sha256',
               'source_ledger_canonical_sha256', 'valid_count', 'valid_cells_sha256', 'selected_rows_sha256'}
    need(set(pins) == set(template) and all(canonical(pins[k]) == canonical(v)
         for k, v in template.items() if k not in dynamic)
         and all(pins.get(k) is False for k in FALSE_FLAGS), 'FIXED_TERMINAL_RESEARCH_RECIPE')
    run = pins.get('observer_run')
    need(type(run) is dict and set(run) == {'run_id', 'github_sha', 'run_attempt'} and
         type(run['run_id']) is int and run['run_id'] > 0 and str(run['run_id']) != SOURCE_IDENTITY['run_id'] and
         hex_id(run['github_sha'], 40) and run['github_sha'] != SOURCE_IDENTITY['github_sha'] and
         type(run['run_attempt']) is int and run['run_attempt'] == 1, 'ACTUAL_NEW_OBSERVER_RUN_PIN')
    need(hex_id(pins.get('observation_pins_sha256')) and
         hex_id(pins.get('observer_terminal_evidence_canonical_sha256')), 'ACTUAL_OBSERVER_EVIDENCE_PINS')
    artifacts = pins.get('observer_artifacts')
    need(type(artifacts) is dict and set(artifacts) == set(PROVIDERS), 'FOUR_OBSERVER_ARTIFACT_PINS')
    entries = list(artifacts.values()) + [pins.get('combined_artifact')]
    for entry in entries:
        need(type(entry) is dict and type(entry.get('artifact_id')) is int and entry['artifact_id'] > 0 and
             type(entry.get('zip_bytes')) is int and 0 < entry['zip_bytes'] <= MAX_OBSERVER_ZIP and
             hex_id(entry.get('zip_sha256')), 'ACTUAL_OBSERVER_ARTIFACT_PIN')
    need(len({e['artifact_id'] for e in entries}) == 5 and
         sum(e['zip_bytes'] for e in entries) <= 64 * 1024**2, 'OBSERVER_ARTIFACT_BOUNDS')
    for entry in artifacts.values():
        need(set(entry) == {'artifact_id', 'zip_bytes', 'zip_sha256', 'summary_sha256', 'model_lock_sha256'} and
             hex_id(entry['summary_sha256']) and hex_id(entry['model_lock_sha256']), 'SAVED_OBSERVER_SUMMARY_PINS')
    combined = pins['combined_artifact']
    need(set(combined) == {'artifact_id', 'zip_bytes', 'zip_sha256', 'member', 'member_sha256'} and
         combined['member'] == 'terminal144-observations-144.json' and hex_id(combined['member_sha256']),
         'SAVED_COMBINED_OBSERVER_PIN')
    for name in ('source_ledger_canonical_sha256', 'valid_cells_sha256', 'selected_rows_sha256'):
        need(pins.get(name) is None or hex_id(pins[name]), 'OPTIONAL_AUDIT_HASH:' + name)
    need(pins.get('valid_count') is None or type(pins['valid_count']) is int and 0 < pins['valid_count'] <= 144,
         'OPTIONAL_AUDIT_SOURCE_COUNT')
    if execution:
        need(pins['enabled'] is True and hex_id(pins.get('next_main_before'), 40) and
             hex_id(pins.get('code_sha256')) and
             sha(regular(Path(__file__), 256 * 1024)) == pins['code_sha256'], 'TRAINING_DISABLED_OR_CODE_UNPINNED')
        need(all(hex_id(pins.get(k)) for k in
             ('source_ledger_canonical_sha256', 'valid_cells_sha256', 'selected_rows_sha256')) and
             type(pins['valid_count']) is int, 'UNFROZEN_ACTUAL_SOURCE_OR_SELECTION')


def execution_guard(pins, event, env):
    """Pure guard; tests pass dictionaries, never forge the process environment."""
    validate_pins(pins, execution=True)
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and str(env.get('GITHUB_RUN_ID', '')).isdigit() and
         str(env.get('GITHUB_RUN_ID')) not in (SOURCE_IDENTITY['run_id'], str(pins['observer_run']['run_id'])) and
         event.get('before') == pins['next_main_before'] and event.get('after') == env.get('GITHUB_SHA') and
         hex_id(event.get('after'), 40) and event['after'] not in
         (event['before'], SOURCE_IDENTITY['github_sha'], pins['observer_run']['github_sha']) and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'FIRST_DISTINCT_PUBLIC_MAIN_TRAIN_ATTEMPT_ONLY')


def observer_identity(pins):
    run = pins['observer_run']
    return {'run_id': str(run['run_id']), 'github_sha': run['github_sha'], 'run_attempt': '1', 'test_backend': False}


def select_terminal(recipe, ledger, combined, old):
    """The frozen per-row predicate, with full-denominator source quarantine."""
    cells, source, joined = recipe.get('cells'), ledger.get('rows'), combined.get('rows')
    need(all(type(rows) is list and len(rows) == 144 and len({r['cell_id'] for r in rows}) == 144
             for rows in (cells, source, joined)) and
         {c['cell_id'] for c in cells} == {r['cell_id'] for r in source} == {r['cell_id'] for r in joined},
         'COMPLETE_144_TERMINAL_DISPOSITION_ROSTER')
    need(ledger.get('original_campaign_status') == combined.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY' and
         ledger.get('test_backend') is False and combined.get('test_backend') is False and
         combined.get('status') == 'FOUR_OBSERVERS_COMPLETE', 'REAL_COMPLETE_TERMINAL_OBSERVATIONS_ONLY')
    summaries = combined.get('observer_summaries_sha256')
    need(type(summaries) is dict and set(summaries) == set(PROVIDERS) and
         all(hex_id(v) for v in summaries.values()), 'FOUR_ORIGINAL_SUMMARY_IDENTITIES')
    manifest_sha = sha(canonical(summaries))
    by_source, by_joined = ({r['cell_id']: r for r in rows} for rows in (source, joined))
    pcm_counts = Counter(r['pcm_sha256'] for r in source if hex_id(r.get('pcm_sha256')))
    eligible, observations, decisions = [], {}, []
    for cell in cells:
        cid = cell['cell_id']
        row, readback = by_source[cid], by_joined[cid]
        need({k: readback.get(k) for k in row} == row and
             all(row.get(k) is False for k in FALSE_FLAGS), 'JOINED_ORIGINAL_SOURCE_IDENTITY')
        label = cell.get('requested_keyword_index') or 0
        base = {'cell_id': cid, 'role': cell['role'], 'weak_intent_class': label,
            'source_status_unchanged': row['source_status'], 'source_disposition': row['source_disposition'],
            **dict.fromkeys(FALSE_FLAGS, False)}
        reason = None
        if row.get('source_disposition') != 'VALID_UNOBSERVED_SOURCE':
            reason = 'SOURCE_UNAVAILABLE_OR_QUARANTINED'
        elif row.get('source_status') != 'GENERATED_UNADMITTED' or not hex_id(row.get('pcm_sha256')):
            reason = 'SOURCE_IDENTITY_NOT_USABLE'
        elif pcm_counts[row['pcm_sha256']] != 1:
            reason = 'WHOLE_DUPLICATE_PCM_GROUP_QUARANTINED'
        values = readback.get('observations', {})
        if reason is None:
            need(type(values) is dict and set(values) == set(PROVIDERS) and
                 all(v.get('observation_status') == 'COMPLETE' and v.get('observation_attempted') is True and
                     v.get('pcm_sha256') == row['pcm_sha256'] for v in values.values()), 'FOUR_COMPLETE_SAME_PCM_ROWS')
            transcripts = {p: values[p].get('text') for p in PROVIDERS[:3]}
            tokens = values['phones'].get('raw_ipa_tokens')
            need(all(type(v) is str for v in transcripts.values()) and type(tokens) is list and
                 all(type(t) is str and t for t in tokens), 'OBSERVER_VALUE_TYPES')
            if not tokens:
                reason = 'EMPTY_PHONE_OBSERVATION_UNKNOWN'
        if reason is not None:
            decisions.append({**base, 'evidence_status': 'UNKNOWN', 'phone_review_status': reason,
                'eligible_for_requested_intent_research': False, 'quarantine_reason': reason})
            continue
        cores = {1: old.has_core(tokens, old.SELECTION['vosen_core']),
                 2: old.has_core(tokens, old.SELECTION['mero_core'])}
        normalized_asr = {p: old.normalized(text) for p, text in transcripts.items()}
        if cell.get('intent') == 'wake-candidate':
            need(label in (1, 2), 'WAKE_INTENT_CLASS')
            names = old.SELECTION['vosen_names' if label == 1 else 'mero_names']
            votes = sum(len(words) == 2 and words[0] == 'hey' and words[1] in names for words in normalized_asr.values())
            useful = votes >= 2 and cores[label]
            status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else (
                'STRONG_CONFLICT' if cores[3 - label] and not cores[label] else 'UNKNOWN')
            phone_status = ('TARGET_CORE_MACHINE_SUPPORT_HEY_LENGTH_STRESS_UNVERIFIED' if cores[label] else
                            'TARGET_CORE_NOT_ESTABLISHED_BY_THIS_OBSERVER')
        else:
            need(cell.get('intent') == 'ordinary-negative-candidate' and label == 0, 'ORDINARY_INTENT_CLASS')
            wanted = old.normalized(cell['synthesis_text'])
            votes = sum(words == wanted for words in normalized_asr.values())
            useful = votes >= 2 and not any(cores.values())
            status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else ('STRONG_CONFLICT' if any(cores.values()) else 'UNKNOWN')
            phone_status = ('NO_TARGET_CORE_MACHINE_OBSERVED_NOT_TRUSTED_NEGATIVE' if not any(cores.values()) else
                            'TARGET_CORE_IN_ORDINARY_INTENT_CONFLICT')
        observation = {'source_pcm_sha256': row['pcm_sha256'], 'observer_artifact_sha256': manifest_sha,
            'observer_artifact_identity_kind': 'FOUR_ORIGINAL_SUMMARY_SHA256_MANIFEST',
            'observer_summaries_sha256': summaries,
            'observation_sha256': sha(canonical({'cell_id': cid, 'pcm_sha256': row['pcm_sha256'],
                'transcripts': transcripts, 'raw_ipa_tokens': tokens})),
            'evidence_status': status, 'phone_review_status': phone_status}
        observations[cid] = observation
        if useful:
            eligible.append(cid)
        decisions.append({**base, 'asr_votes': votes, 'machine_phone_core': {str(k): v for k, v in cores.items()},
            'transcripts': transcripts, 'raw_ipa_tokens': tokens, 'evidence_status': status,
            'phone_review_status': phone_status, 'eligible_for_requested_intent_research': useful})
    return {'policy': old.SELECTION, 'policy_sha256': SELECTION_SHA,
        'observer_artifact_manifest_sha256': manifest_sha, 'source_denominator': 144,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'valid_source_count': ledger['valid_count'],
        'source_dispositions': ledger['source_dispositions'],
        'source_ledger_canonical_sha256': sha(canonical(ledger)), 'observer_identity': combined['observer_identity'],
        'eligible_cell_ids': eligible, 'observations': observations, 'decisions': decisions}


def library_source_rows(ledger):
    """Create an exclusion view; retain original status separately, never promote it."""
    rows = []
    for source in ledger['rows']:
        status = source['source_status']
        if status == 'GENERATED_UNADMITTED' and source['source_disposition'] == 'INVALID_SOURCE_RECEIPT':
            status = 'INVALID_SOURCE_RECEIPT'
        rows.append({**source, 'status': status})
    return rows


def artifact_roster(pins):
    entries = [{'name': 'observer-' + p, **{k: pins['observer_artifacts'][p][k]
                for k in ('artifact_id', 'zip_bytes', 'zip_sha256')}} for p in PROVIDERS]
    entries.append({'name': 'terminal144-observations-' + str(pins['observer_run']['run_id']),
                    **{k: pins['combined_artifact'][k] for k in ('artifact_id', 'zip_bytes', 'zip_sha256')}})
    return entries


def observer_metadata(bundle, pins):
    """Verify actual API identities; a successor cannot pretend to be the source run."""
    raw = {name: regular(bundle / name, 4 * 1024**2) for name in ('run.json', 'jobs.json', 'artifacts.json')}
    run, jobs, artifacts = (loads(raw[name]) for name in ('run.json', 'jobs.json', 'artifacts.json'))
    expected = pins['observer_run']
    need(run.get('id') == expected['run_id'] and run.get('head_sha') == expected['github_sha'] and
         run.get('run_attempt') == 1 and run.get('event') == 'push' and run.get('head_branch') == 'main' and
         run.get('status') == 'completed' and run.get('conclusion') == 'success' and
         run.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         run.get('repository', {}).get('private') is False, 'ACTUAL_COMPLETED_OBSERVER_RUN_REQUIRED')
    job_rows = jobs.get('jobs')
    names = {'source-contract', 'source-evidence', 'combine'} | {'observe (' + p + ')' for p in PROVIDERS}
    need(type(job_rows) is list and jobs.get('total_count') == len(job_rows) == len(names) and
         {j['name'] for j in job_rows} == names, 'EXACT_COMPLETED_OBSERVER_JOB_ROSTER')
    terminal_rows = []
    for job in job_rows:
        need(job.get('run_id') == expected['run_id'] and job.get('head_sha') == expected['github_sha'] and
             job.get('run_attempt') == 1 and job.get('status') == 'completed' and
             job.get('conclusion') == 'success', 'OBSERVER_JOB_NOT_COMPLETE_SAME_RUN')
        terminal_rows.append({k: job[k] for k in ('id', 'name', 'run_id', 'run_attempt', 'head_sha', 'status', 'conclusion')})
    terminal = {'run_id': run['id'], 'head_sha': run['head_sha'], 'run_attempt': 1,
        'status': 'completed', 'conclusion': 'success', 'jobs': sorted(terminal_rows, key=lambda j: j['name'])}
    need(sha(canonical(terminal)) == pins['observer_terminal_evidence_canonical_sha256'],
         'FROZEN_OBSERVER_TERMINAL_EVIDENCE')
    actual = artifacts.get('artifacts')
    need(type(actual) is list and artifacts.get('total_count') == len(actual) and
         len({a['id'] for a in actual}) == len(actual), 'COMPLETE_OBSERVER_ARTIFACT_METADATA')
    for pin in artifact_roster(pins):
        matches = [a for a in actual if a['id'] == pin['artifact_id'] or a.get('name') == pin['name']]
        need(len(matches) == 1, 'UNIQUE_PINNED_OBSERVER_ARTIFACT')
        record = matches[0]
        workflow = record.get('workflow_run', {})
        need(record.get('id') == pin['artifact_id'] and record.get('name') == pin['name'] and
             record.get('size_in_bytes') == pin['zip_bytes'] and record.get('digest') == 'sha256:' + pin['zip_sha256'] and
             record.get('expired') is False and workflow.get('id') == expected['run_id'] and
             workflow.get('head_sha') == expected['github_sha'] and workflow.get('head_branch') == 'main',
             'ACTUAL_OBSERVER_ARTIFACT_IDENTITY')
    return {'observer_identity': observer_identity(pins), 'artifacts': artifact_roster(pins),
        'observer_terminal_evidence': terminal,
        'observer_terminal_evidence_canonical_sha256': sha(canonical(terminal)),
        'api_files_sha256': {name: sha(value) for name, value in raw.items()}}


def checked_zip(path, pin):
    raw = regular(path, MAX_OBSERVER_ZIP)
    need(len(raw) == pin['zip_bytes'] and sha(raw) == pin['zip_sha256'], 'EXACT_OBSERVER_ZIP_BYTES')
    return zipfile.ZipFile(io.BytesIO(raw))


def safe_members(archive, required, allowed, *, max_file=2 * 1024**2, max_total=16 * 1024**2):
    members = archive.infolist()
    names = [m.filename for m in members]
    need(len(names) == len(set(names)) and required <= set(names) <= allowed and
         sum(m.file_size for m in members) <= max_total, 'OBSERVER_ZIP_MEMBER_ROSTER')
    for member in members:
        path = PurePosixPath(member.filename)
        need(not path.is_absolute() and '..' not in path.parts and str(path) == member.filename and
             '\\' not in member.filename and not member.is_dir() and not member.flag_bits & 1 and
             stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG) and
             0 <= member.file_size <= max_file, 'UNSAFE_OBSERVER_ZIP_MEMBER')
    return members


def extract_observers(bundle, pins, ledger, output):
    need(not output.exists(), 'NEW_OBSERVER_READBACK_DIRECTORY')
    output.mkdir(mode=0o700)
    required = {'started.json', 'summary.json', 'SOURCE_LEDGER.json', 'model-lock.json'}
    for row in ledger['valid_cells']:
        required.update('cells/' + row['cell_id'] + '/' + name for name in ('started.json', 'completed.json'))
    for provider in PROVIDERS:
        pin = pins['observer_artifacts'][provider]
        folder = output / ('observer-' + provider)
        folder.mkdir()
        with checked_zip(bundle / ('observer-' + provider + '.zip'), pin) as archive:
            for member in safe_members(archive, required, required):
                target = folder / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(archive.read(member))
        need(sha(regular(folder / 'summary.json', 2 * 1024**2)) == pin['summary_sha256'] and
             sha(regular(folder / 'model-lock.json', 1024**2)) == pin['model_lock_sha256'],
             'SAVED_OBSERVER_SUMMARY_OR_MODEL_LOCK')
    pin = pins['combined_artifact']
    with checked_zip(bundle / 'combined.zip', pin) as archive:
        safe_members(archive, {pin['member']}, {pin['member']}, max_file=4 * 1024**2)
        raw = archive.read(pin['member'])
    need(sha(raw) == pin['member_sha256'], 'ACTUAL_SAVED_COMBINED_MEMBER')
    return raw


def validate_source_documents(pins, observation_pins, ledger, combined, observer_binding, old, observer):
    """Validate saved semantic bindings without models or the current environment."""
    observer.validate_pins(observation_pins, frozen=True)
    need(observation_pins.get('enabled') is True and observation_pins.get('code_sha256') == OBSERVER_SHA and
         sha(canonical(observation_pins)) == pins['observation_pins_sha256'], 'ACTUAL_FROZEN_OBSERVATION_PINS')
    need(ledger.get('source_denominator') == combined.get('source_denominator') == 144 and
         ledger.get('test_backend') is False and combined.get('test_backend') is False and
         ledger.get('original_run_conclusion') == 'failure' and
         ledger.get('original_campaign_status') == combined.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY' and
         all(ledger.get(k) == combined.get(k) == v for k, v in observer.SOURCE.items()) and
         all(doc.get(flag) is False for doc in (ledger, combined) for flag in FALSE_FLAGS),
         'ORIGINAL_FAILED_CAMPAIGN_RETAINED')
    ledger_sha = sha(canonical(ledger))
    need(ledger_sha == observation_pins['source_ledger_canonical_sha256'] == combined.get('source_ledger_canonical_sha256') and
         ledger.get('terminal_evidence_canonical_sha256') == observation_pins['terminal_evidence_canonical_sha256'] and
         ledger['terminal_evidence_canonical_sha256'] == sha(canonical(ledger.get('terminal_evidence'))) and
         ledger.get('source_dispositions') == dict(Counter(r['source_disposition'] for r in ledger['rows'])) and
         ledger.get('valid_cells') == observation_pins['valid_cells'] == observer.cohort(ledger) and
         ledger.get('valid_count') == len(ledger['valid_cells']) == observation_pins['max_calls_per_provider'],
         'FULL_SOURCE_LEDGER_AND_COHORT_BINDINGS')
    count = ledger['valid_count']
    need(combined.get('status') == 'FOUR_OBSERVERS_COMPLETE' and combined.get('valid_count') == count and
         combined.get('observer_identity') == observer_identity(pins) and
         combined.get('new_tts_calls') == combined.get('optimizer_steps') == combined.get('model_calls_by_combiner') == 0 and
         set(combined.get('providers', {})) == set(PROVIDERS) and
         all(combined['providers'][p].get('status') == 'OBSERVATIONS_COMPLETE' and
             combined['providers'][p].get('attempts') == combined['providers'][p].get('completed') == count
             for p in PROVIDERS), 'ALL_FOUR_FIRST_OBSERVERS_ACTUALLY_COMPLETE')
    need(combined.get('observer_summaries_sha256') ==
         {p: pins['observer_artifacts'][p]['summary_sha256'] for p in PROVIDERS} and
         combined.get('observer_model_locks_sha256') ==
         {p: pins['observer_artifacts'][p]['model_lock_sha256'] for p in PROVIDERS}, 'SAVED_COMPLETE_OBSERVER_HASHES')
    need(observer_binding.get('observer_identity') == observer_identity(pins) and
         observer_binding.get('artifacts') == artifact_roster(pins) and
         observer_binding.get('observer_terminal_evidence_canonical_sha256') ==
         sha(canonical(observer_binding.get('observer_terminal_evidence'))) ==
         pins['observer_terminal_evidence_canonical_sha256'] and
         set(observer_binding.get('api_files_sha256', {})) == {'run.json', 'jobs.json', 'artifacts.json'} and
         all(hex_id(v) for v in observer_binding['api_files_sha256'].values()), 'SAVED_OBSERVER_API_BINDINGS')
    for key, actual in (('source_ledger_canonical_sha256', ledger_sha),
                       ('valid_count', count), ('valid_cells_sha256', sha(canonical(ledger['valid_cells'])))):
        need(pins.get(key) is None or pins[key] == actual, 'FROZEN_ACTUAL_COHORT:' + key)
    selection = select_terminal(observer.recipe.recipe(), ledger, combined, old)
    need(pins.get('selected_rows_sha256') is None or pins['selected_rows_sha256'] == sha(canonical(selection)),
         'FROZEN_ACTUAL_SELECTION_CHANGED')
    return selection


def make_source_binding(pins, observation_pins, ledger, combined, combined_raw_sha, observer_binding):
    return {'source_identity': SOURCE_IDENTITY, 'observer_identity': observer_identity(pins),
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'source_denominator': 144,
        'valid_source_count': ledger['valid_count'], 'source_dispositions': ledger['source_dispositions'],
        'source_ledger_canonical_sha256': sha(canonical(ledger)),
        'terminal_evidence_canonical_sha256': ledger['terminal_evidence_canonical_sha256'],
        'recipe_sha256': RECIPE_SHA, 'observation_pins_sha256': sha(canonical(observation_pins)),
        'combined_observations_sha256': combined_raw_sha,
        'observer_artifact_manifest_sha256': sha(canonical(observer_binding)),
        'observer_artifact_identity_kind': 'EXACT_FOUR_ZIPS_AND_SAVED_JOIN_WITH_API_BINDINGS',
        'observer_summaries_sha256': combined['observer_summaries_sha256'], 'test_backend': False,
        **dict.fromkeys(FALSE_FLAGS, False)}


def readback(args, pins, old, observer, temporary, output):
    observation_pins = read_json(args.observation_pins)
    observer.validate_pins(observation_pins, frozen=True)
    need(sha(canonical(observation_pins)) == pins['observation_pins_sha256'], 'FROZEN_OBSERVER_PINS_READBACK')
    ledger, source = observer.read_sources(Path(args.source_bundle), observation_pins, temporary / 'source')
    bundle = Path(args.observer_bundle)
    need({p.name for p in bundle.glob('*.zip')} ==
         {'observer-' + p + '.zip' for p in PROVIDERS} | {'combined.zip'}, 'EXACT_FIVE_OBSERVER_ZIPS')
    observer_binding = observer_metadata(bundle, pins)
    combined_raw = extract_observers(bundle, pins, ledger, temporary / 'observers')
    combined = loads(combined_raw)
    recomputed = observer.combine(Path(args.source_bundle), temporary / 'observers', observation_pins)
    need(canonical(recomputed) == canonical(combined), 'ACTUAL_SAVED_JOIN_MUST_EQUAL_RECEIPT_READBACK')
    selection = validate_source_documents(pins, observation_pins, ledger, combined, observer_binding, old, observer)
    binding = make_source_binding(pins, observation_pins, ledger, combined, sha(combined_raw), observer_binding)
    for name, doc in (('SOURCE_LEDGER.json', ledger), ('OBSERVER_BINDING.json', observer_binding),
                      ('SOURCE_BINDING.json', binding), ('SELECTION.json', selection),
                      ('TRAINING_PINS.json', pins), ('OBSERVATION_PINS.json', observation_pins)):
        once(output / name, doc)
    with (output / 'COMBINED_OBSERVATIONS.json').open('xb') as stream:
        stream.write(combined_raw)
    wavs = {row['cell_id']: regular(path, 512 * 1024) for row, path in source}
    return ledger, binding, selection, wavs


def saved_source_evidence(root):
    """Packager's pure validator for the three distinct saved identities."""
    root = Path(root)
    evidence = {name: read_json(root / name) for name in SOURCE_EVIDENCE_FILES}
    pins, observation_pins = evidence['TRAINING_PINS.json'], evidence['OBSERVATION_PINS.json']
    validate_pins(pins, execution=True)
    old, observer = contracts()
    ledger, combined, observed = (evidence[name] for name in
                                ('SOURCE_LEDGER.json', 'COMBINED_OBSERVATIONS.json', 'OBSERVER_BINDING.json'))
    selected = validate_source_documents(pins, observation_pins, ledger, combined, observed, old, observer)
    raw_sha = sha(regular(root / 'COMBINED_OBSERVATIONS.json'))
    need(raw_sha == pins['combined_artifact']['member_sha256'] and
         evidence['SELECTION.json'] == selected and evidence['SOURCE_BINDING.json'] ==
         make_source_binding(pins, observation_pins, ledger, combined, raw_sha, observed),
         'SAVED_SOURCE_SELECTION_IDENTITY')
    started = evidence['STARTED.json']
    need(started.get('schema') == SCHEMA and started.get('test_backend') is False and
         started.get('source_identity') == SOURCE_IDENTITY and started.get('observer_identity') == observer_identity(pins) and
         started.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY' and started.get('source_denominator') == 144 and
         started.get('valid_source_count') == ledger['valid_count'] and started.get('source_dispositions') == ledger['source_dispositions'] and
         started.get('training_pins_sha256') == sha(canonical(pins)) and started.get('worker_code_sha256') == pins['code_sha256'] and
         started.get('selection_policy_sha256') == SELECTION_SHA and started.get('maximum_optimizer_updates') == 96 and
         started.get('no_retry') is True and all(started.get(k) is False for k in FALSE_FLAGS) and
         type(started.get('run_id')) is str and started['run_id'].isdigit() and started.get('run_attempt') == '1' and
         started['run_id'] not in (SOURCE_IDENTITY['run_id'], str(pins['observer_run']['run_id'])) and
         hex_id(started.get('github_sha'), 40) and started['github_sha'] not in
         (SOURCE_IDENTITY['github_sha'], pins['observer_run']['github_sha']), 'SAVED_THREE_DISTINCT_RUN_IDENTITIES')
    return evidence


def selected_library_rows(libs, old, recipe, ledger, selection, excluded):
    try:
        rows = libs['weak_embedding_contract'].weak_rows(recipe, library_source_rows(ledger),
            eligible_cell_ids=selection['eligible_cell_ids'], observations=selection['observations'], excluded=excluded)
    except ValueError as error:
        if str(error) not in ('EXPLICIT_INTENT_RESEARCH_SELECTION_REQUIRED', 'MISSING_WEAK_CLASS_OR_ROLE'):
            raise
        candidates = [{'role': c['role'], 'weak_class': c.get('requested_keyword_index') or 0,
                       'source_family_id': c['source_family_id']} for c in recipe['cells']
                      if c['cell_id'] in set(selection['eligible_cell_ids'])]
        minimum = old.minimum_evidence(candidates)
        minimum.update(passed=False, library_rejection=str(error))
        return [], minimum
    return rows, old.minimum_evidence(rows)


def run_worker(args, *, execute):
    pins = read_json(args.pins)
    validate_pins(pins, execution=execute)
    if execute:
        execution_guard(pins, read_json(Path(os.environ.get('GITHUB_EVENT_PATH', ''))), os.environ)
    old, observer = contracts()
    output = Path(args.output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_TERMINAL_TRAINING_OUTPUT')
    output.mkdir(mode=0o700)
    base = {'schema': SCHEMA, 'run_id': os.environ.get('GITHUB_RUN_ID') if execute else None,
        'github_sha': os.environ.get('GITHUB_SHA') if execute else None, 'run_attempt': '1' if execute else None,
        'worker_code_sha256': sha(regular(Path(__file__), 256 * 1024)),
        'training_pins_sha256': sha(canonical(pins)), 'selection_policy_sha256': SELECTION_SHA,
        'source_identity': SOURCE_IDENTITY, 'observer_identity': observer_identity(pins),
        'source_denominator': 144, 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
        'maximum_optimizer_updates': 96 if execute else 0, 'no_retry': True, 'test_backend': False,
        **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'READBACK_STARTED.json', {**base, 'operation': 'execute' if execute else 'audit-selection',
        'model_calls_at_start': 0, 'optimizer_updates_at_start': 0})
    stage, training_started, trained = 'source-and-observer-readback', False, None
    clock = time.monotonic()
    started = base
    try:
        with tempfile.TemporaryDirectory(prefix='terminal144-train-readback-') as private:
            ledger, binding, selection, wavs = readback(args, pins, old, observer, Path(private), output)
        started = {**base, 'valid_source_count': ledger['valid_count'],
                   'source_dispositions': ledger['source_dispositions']}
        if execute:
            once(output / 'STARTED.json', started)
        libs, code = old.pipeline_libraries(Path(args.pipeline_source), PIPELINE_COMMIT)
        excluded = []
        rows, minimum = selected_library_rows(libs, old, observer.recipe.recipe(), ledger, selection, excluded)
        once(output / 'MINIMUM_AFTER_DEDUPLICATION.json', minimum)
        if not execute:
            once(output / 'EXCLUSIONS.json', excluded)
            once(output / 'RESULT.json', {**started, 'status': 'SOURCE_AND_SELECTION_AUDIT_ONLY',
                'optimizer_updates': 0, 'model_calls': 0, 'execution_enabled_by_audit': False,
                'selected_rows_sha256': sha(canonical(selection)), 'source_ledger_canonical_sha256': sha(canonical(ledger)),
                'valid_cells_sha256': sha(canonical(ledger['valid_cells'])), 'minimum': minimum,
                'pipeline_source': code, 'new_tts_asr_phone_calls': 0})
            return 0 if minimum['passed'] else 2
        if not minimum['passed']:
            once(output / 'EXCLUSIONS.json', excluded)
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_TERMINAL_MINIMUM_BEFORE_TRAINING',
                'optimizer_updates': 0, 'minimum': minimum, 'new_tts_asr_phone_calls': 0})
            return 2
        stage = 'runtime-and-original-background-preflight'
        portable = libs['portable_head']
        native, arm, build = old.build_replays(portable, output)
        once(output / 'BUILD.json', build)
        reference = old.reference_records(args.reference_zip, libs['librispeech'])
        background, decoder = old.decode_background(args.corpus_archive, reference, libs['librispeech'])
        tracks = libs['weak_audio_cases'].background_tracks(background)
        once(output / 'LIBRISPEECH_SOURCES.json', {'reference': old.REFERENCE, 'corpus': old.CORPUS,
            'decoder': decoder, 'decoded_originals': [{k: v for k, v in r.items() if k != 'pcm16le'} for r in background],
            'actual_research_tracks': [{k: v for k, v in t.items() if k != 'pcm16le'} for t in tracks],
            'unique_research_track_samples': sum(t['frames'] for t in tracks), 'prior_events_reused': False,
            'source_pool_is_previously_exposed': True})
        del background
        cases = libs['weak_audio_cases'].assemble_audio_cases(rows, wavs, tracks, excluded=excluded)
        stage = 'frozen-feature-extraction'
        embedding = libs['causal_embedding'].FrozenOnnxEmbedding(args.assets)
        expected_calls = sum(max(0, 1 + (len(c['pcm16le']) // 2 - 16000) // 1280) for c in cases)
        need(expected_calls <= 40000, 'FINITE_FROZEN_EMBEDDING_BUDGET')
        once(output / 'FEATURE_PREFLIGHT.json', {'geometry': embedding.geometry,
            'maximum_embedding_calls': expected_calls, 'asset_identities': libs['weak_embedding_contract'].ASSETS,
            'pipeline_source': code})
        bags, streams, evidence = libs['weak_audio_cases'].extract_feature_bags(cases, embedding, excluded=excluded)
        need(sum(e['embedding_calls'] for e in evidence) <= expected_calls, 'FEATURE_FORWARD_ACCOUNTING')
        evidence_sha = once(output / 'FEATURE_EVIDENCE.json', evidence)
        once(output / 'EXCLUSIONS.json', excluded)
        usable = {bag['id'] for bag in bags}
        final_minimum = old.minimum_evidence([row for row in rows if row['cell_id'] in usable])
        once(output / 'MINIMUM_AFTER_FEATURE_GEOMETRY.json', final_minimum)
        if not final_minimum['passed']:
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_TERMINAL_FEATURE_ELIGIBILITY_BEFORE_TRAINING',
                'optimizer_updates': 0, 'minimum': final_minimum, 'new_tts_asr_phone_calls': 0})
            return 2
        feature_stream_sha = old.save_stream_cache(streams, cases, output)
        input_binding = {'source_ledger_canonical_sha256': binding['source_ledger_canonical_sha256'],
            'source_binding_sha256': sha(canonical(binding)),
            'observation_pins_sha256': binding['observation_pins_sha256'],
            'combined_observations_sha256': binding['combined_observations_sha256'],
            'observer_artifact_manifest_sha256': binding['observer_artifact_manifest_sha256'],
            'recipe_sha256': RECIPE_SHA, 'selection_policy_sha256': SELECTION_SHA,
            'selection_sha256': sha(canonical(selection)), 'corpus_sha256': old.CORPUS['sha256'],
            'reference_sha256': old.REFERENCE['member_sha256'], 'pipeline_source_sha256': sha(canonical(code)),
            'feature_evidence_sha256': evidence_sha, 'feature_stream_sha256': feature_stream_sha,
            'worker_code_sha256': started['worker_code_sha256'], 'legacy_worker_sha256': LEGACY_SHA,
            'terminal_observer_code_sha256': OBSERVER_SHA}
        stage = 'fixed-96-step-training'
        once(output / 'TRAINING_STARTED.json', {'maximum_optimizer_updates': 96, 'input_binding': input_binding,
            'development_tuning': False, 'no_retry': True})
        training_started = True
        trained = libs['causal_head'].train_cached(bags, output / 'train', input_binding)
        need(trained['optimizer_updates'] == 96 and trained['spec']['seed'] == 20261116, 'EXACT_FIXED_TRAINING_RECIPE')
        stage = 'saved-weight-host-and-arm-head-replay'
        reports, per_case = old.replay_saved_streams(portable, output / 'train/causal-head.npz', output, native, arm)
        once(output / 'PARITY.json', reports)
        once(output / 'PER_CASE_EVENTS.json', per_case)
        track_ids = {track['id'] for track in tracks}
        track_cases = [row for row in per_case if row['id'] in track_ids]
        background_events = sum(len(row['events']) for row in track_cases)
        seconds = sum(track['frames'] for track in tracks) / 16000
        result = {**started, 'status': SUCCESS, 'optimizer_updates': 96,
            'weights_sha256': trained['weights_sha256'], 'head_binary_sha256': sha(regular(output / 'causal-head.bin')),
            'frozen_embedding_calls': sum(e['embedding_calls'] for e in evidence),
            'selected_after_deduplication': len(rows), 'usable_source_cases': sum(r['cell_id'] in usable for r in rows),
            'excluded_rows': excluded, 'minimum': final_minimum, 'input_binding': input_binding,
            'exposed_read_speech_tracks': {'tracks': len(track_cases), 'seconds': seconds, 'events': background_events,
                'research_events_per_hour': background_events * 3600 / seconds, 'reset_per_speaker': True,
                'unique_exposure_includes_no_reused_prefixes': True, 'independent_continuous_far': False},
            'wall_seconds': time.monotonic() - clock, 'new_tts_asr_phone_calls': 0,
            'independent_frr': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'word_end_latency': 'NOT_RUN',
            'ssc305': 'NOT_RUN', 'deployment_frontend_arm_parity': 'NOT_RUN',
            'arm_scope': 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU'}
        once(output / 'RESULT.json', result)
        return 0
    except Exception as error:
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': observer.error_message(error),
            'new_tts_asr_phone_calls': 0,
            'optimizer_updates': trained['optimizer_updates'] if trained is not None else
                ('UNKNOWN_AFTER_TRAINING_STARTED' if training_started else 0)})
        return 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--template', action='store_true')
    modes.add_argument('--audit-selection', action='store_true')
    modes.add_argument('--execute', action='store_true')
    cli.add_argument('--pins', type=Path, default=HERE / 'terminal144_training_pins.json')
    cli.add_argument('--observation-pins', type=Path, default=AUTOMATION / 'terminal144_observation_pins.json')
    for name in ('source-bundle', 'observer-bundle', 'pipeline-source', 'corpus-archive', 'reference-zip', 'assets', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.execute and not args.audit_selection:
        print(json.dumps({'schema': SCHEMA, 'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
            'optimizer_updates': 0, 'execution_enabled': False, 'source_denominator': 144,
            'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'selection_policy_sha256': SELECTION_SHA,
            'pipeline_commit': PIPELINE_COMMIT, **dict.fromkeys(FALSE_FLAGS, False)}, sort_keys=True))
        return 0
    required = ['source_bundle', 'observer_bundle', 'pipeline_source', 'output']
    if args.execute:
        required += ['corpus_archive', 'reference_zip', 'assets']
    need(all(getattr(args, name) is not None for name in required), 'EXPLICIT_TERMINAL_TRAINING_INPUTS_REQUIRED')
    return run_worker(args, execute=args.execute)


if __name__ == '__main__':
    raise SystemExit(main())
