"""One separate 160-request, requested-intent weak-research snapshot and fit.

Default/import are source-only. Preparation reads saved evidence, never models.
Execution is a new public-main once request; the old 144 campaign stays failed.
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
import shutil
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
SCHEMA = 'english-snapshot160-weak-research-v1'
PINS_SCHEMA = 'english-snapshot160-training-pins-v1'
SUCCESS = 'EXECUTED_SNAPSHOT160_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'
TERMINAL_SHA = '1c5d73568042723da723403665edaa1a9569f3b056fee080346ae3677cd68bce'
OLD_TRAIN_PINS_SHA = 'b3f9958ba6a2d1b7e49119a4486f3297405cafabdebaa1a5179e659efa30a6d4'
OLD_OBSERVATION_PINS_SHA = 'b858edf92978f12e00a7666c478c2416aada56fb5d0eee4f2e4edb2647f99eec'
OLD_SELECTION_SHA = 'f51849b945cd49c4a726ce79a3d22b656408c9088532b6376d829a10c2baf6b2'
OLD_LEDGER_SHA = 'b6e157eea8eeea38f9261d8756ab43d6570c35a6ef53b892436b7c97424efb38'
MAX_SNAPSHOT_BYTES = 256 * 1024**2
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def identity(path):
    raw = regular(path, MAX_SNAPSHOT_BYTES)
    return {'bytes': len(raw), 'sha256': sha(raw)}


def regular(path, limit=16 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and path.stat().st_size <= limit, 'REGULAR_BOUNDED_FILE')
    raw = path.read_bytes()
    need(len(raw) <= limit, 'FILE_SIZE_CHANGED')
    return raw


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(path):
    return json.loads(regular(path), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha(raw)


def hex_id(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def load_file(name, filename, expected):
    path = HERE / filename
    need(sha(regular(path, 256 * 1024)) == expected, 'FROZEN_SOURCE:' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


terminal = load_file('_snapshot160_terminal', 'terminal144_train_once.py', TERMINAL_SHA)
old, old_observer = terminal.contracts()
FALSE_FLAGS = terminal.FALSE_FLAGS
PIPELINE_COMMIT = terminal.PIPELINE_COMMIT
SELECTION_SHA = terminal.SELECTION_SHA
VOICE_ROLES = {'aiden': 'train', 'dylan': 'train', 'eric': 'train', 'serena': 'train',
               'ryan': 'development', 'vivian': 'development'}
TRAIN_FIELDS = {'enabled', 'next_main_before', 'code_sha256', 'snapshot_sha256'}


def pins_template():
    return {'schema': PINS_SCHEMA, 'enabled': False, 'next_main_before': None,
        'code_sha256': None, 'snapshot_sha256': None,
        'pipeline_commit': PIPELINE_COMMIT, 'terminal_worker_sha256': TERMINAL_SHA,
        'legacy_worker_sha256': terminal.LEGACY_SHA, 'selection_policy_sha256': SELECTION_SHA,
        'old_training_pins_sha256': OLD_TRAIN_PINS_SHA,
        'old_observation_pins_sha256': OLD_OBSERVATION_PINS_SHA,
        'old_selection_sha256': OLD_SELECTION_SHA, 'old_source_ledger_sha256': OLD_LEDGER_SHA,
        'new_worker_sha256': None, 'new_recipe_code_sha256': None, 'new_recipe_sha256': None,
        'new_execution_pins_sha256': None, 'new_run': None, 'new_artifacts': None,
        'new_terminal_evidence_canonical_sha256': None,
        'source_denominator': 160, 'old_declared': 144, 'new_declared': 16,
        'source_order': ['terminal144-original-recipe', 'schwa16-original-recipe'],
        'optimizer_updates': 96, 'seed': 20261116, 'maximum_embedding_calls': 40000,
        'proposal_threshold': 0.8, 'proposal_confirm_embeddings': 3,
        'proposal_cooldown_embeddings': 13, 'development_tuning': False,
        'new_tts_asr_phone_calls': 0, 'no_retry': True,
        'source_labels_are_intents_only': True, **dict.fromkeys(FALSE_FLAGS, False)}


def validate_pins(pins, *, execution=False, ready=True):
    template = pins_template()
    dynamic = TRAIN_FIELDS | {'new_worker_sha256', 'new_recipe_code_sha256', 'new_recipe_sha256',
        'new_execution_pins_sha256', 'new_run', 'new_artifacts', 'new_terminal_evidence_canonical_sha256'}
    need(type(pins) is dict and set(pins) == set(template) and type(pins['enabled']) is bool and
         all(canonical(pins[k]) == canonical(v) for k, v in template.items() if k not in dynamic),
         'FIXED_SNAPSHOT160_TRAINING_RECIPE')
    if not ready and not execution:
        return
    for key in ('new_worker_sha256', 'new_recipe_code_sha256', 'new_recipe_sha256',
                'new_execution_pins_sha256', 'new_terminal_evidence_canonical_sha256'):
        need(hex_id(pins.get(key)), 'ACTUAL_NEW_EVIDENCE_PIN:' + key)
    run = pins.get('new_run')
    need(type(run) is dict and set(run) == {'run_id', 'github_sha', 'run_attempt', 'conclusion'} and
         type(run['run_id']) is int and run['run_id'] > 0 and run['run_id'] not in (43816869386, 43822597212)
         and hex_id(run['github_sha'], 40) and type(run['run_attempt']) is int and run['run_attempt'] == 1
         and run['conclusion'] in ('success', 'failure'), 'NEW_COMPLETED_RUN_PIN')
    artifacts = pins.get('new_artifacts')
    need(type(artifacts) is dict and set(artifacts) == {'source', *PROVIDERS}, 'FIVE_NEW_ARTIFACT_PINS')
    for key, entry in artifacts.items():
        if entry is None:
            need(key in PROVIDERS, 'SOURCE_ARTIFACT_REQUIRED')
            continue
        need(type(entry) is dict and set(entry) == {'artifact_id', 'artifact_name', 'zip_bytes', 'zip_sha256'} and
             type(entry['artifact_id']) is int and entry['artifact_id'] > 0 and
             entry['artifact_name'] == ('schwa16-source' if key == 'source' else 'schwa16-observer-' + key) and
             type(entry['zip_bytes']) is int and 0 < entry['zip_bytes'] <= 32 * 1024**2 and
             hex_id(entry['zip_sha256']), 'BOUNDED_NEW_ARTIFACT_PIN')
    present = [x for x in artifacts.values() if x is not None]
    need(len({x['artifact_id'] for x in present}) == len(present) and
         sum(x['zip_bytes'] for x in present) <= 96 * 1024**2, 'NEW_ARTIFACT_BOUNDS')
    if execution:
        need(pins['enabled'] is True and hex_id(pins['next_main_before'], 40) and
             hex_id(pins['snapshot_sha256']) and hex_id(pins['code_sha256']) and
             sha(regular(Path(__file__), 256 * 1024)) == pins['code_sha256'],
             'TRAINING_DISABLED_OR_SNAPSHOT_CODE_UNPINNED')


def execution_guard(pins, event, env):
    validate_pins(pins, execution=True)
    prior_ids = {'43816869386', '43822597212', str(pins['new_run']['run_id'])}
    need(env.get('GITHUB_ACTIONS') == 'true' and
         env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         event.get('ref') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and str(env.get('GITHUB_RUN_ID', '')).isdigit() and
         env['GITHUB_RUN_ID'] not in prior_ids and event.get('before') == pins['next_main_before'] and
         event.get('after') == env.get('GITHUB_SHA') and hex_id(event.get('after'), 40) and
         event['after'] not in (event['before'], pins['new_run']['github_sha'],
             terminal.SOURCE_IDENTITY['github_sha'], '21d807461c0afa30037362ddde7ea527d87cf3e6') and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'FIRST_DISTINCT_PUBLIC_MAIN_TRAINING_ONLY')


def source_pins(pins):
    return {k: v for k, v in pins.items() if k not in TRAIN_FIELDS}


def new_metadata(bundle, pins):
    raw = {name: regular(bundle / name, 4 * 1024**2) for name in ('run.json', 'jobs.json', 'artifacts.json')}
    run, jobs, artifacts = (json.loads(raw[name], object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON'))) for name in raw)
    expected = pins['new_run']
    need(run.get('id') == expected['run_id'] and run.get('head_sha') == expected['github_sha'] and
         run.get('run_attempt') == 1 and run.get('status') == 'completed' and
         run.get('conclusion') == expected['conclusion'] and run.get('event') == 'push' and
         run.get('head_branch') == 'main' and run.get('repository', {}).get('full_name') == 'jiying2007/kws-data'
         and run.get('repository', {}).get('private') is False, 'NEW_REAL_TERMINAL_RUN')
    names = {'source-contract', 'generate', 'combine'} | {'observe (' + p + ')' for p in PROVIDERS}
    rows = jobs.get('jobs')
    need(type(rows) is list and jobs.get('total_count') == len(rows) == 7 and
         {j['name'] for j in rows} == names, 'NEW_EXACT_SEVEN_JOBS')
    terminal_rows = []
    for job in rows:
        need(job.get('run_id') == expected['run_id'] and job.get('head_sha') == expected['github_sha'] and
             job.get('run_attempt') == 1 and job.get('status') == 'completed' and
             job.get('conclusion') in ('success', 'failure', 'skipped') and
             (job['name'] != 'source-contract' or job['conclusion'] == 'success'), 'NEW_TERMINAL_JOB_IDENTITY')
        terminal_rows.append({k: job[k] for k in ('id', 'name', 'run_id', 'run_attempt', 'head_sha', 'status', 'conclusion')})
    terminal_doc = {'run_id': run['id'], 'head_sha': run['head_sha'], 'run_attempt': 1,
        'status': 'completed', 'conclusion': run['conclusion'], 'jobs': sorted(terminal_rows, key=lambda j: j['name'])}
    need(sha(canonical(terminal_doc)) == pins['new_terminal_evidence_canonical_sha256'], 'NEW_TERMINAL_PIN')
    entries = artifacts.get('artifacts')
    need(type(entries) is list and artifacts.get('total_count') == len(entries) and
         len({e['id'] for e in entries}) == len(entries), 'NEW_ARTIFACT_METADATA_ROSTER')
    for key, entry in pins['new_artifacts'].items():
        if entry is None:
            need(not any(a.get('name') == 'schwa16-observer-' + key for a in entries) and
                 next(j for j in rows if j['name'] == 'observe (' + key + ')')['conclusion'] != 'success',
                 'MISSING_PROVIDER_MUST_BE_ACTUALLY_ABSENT_FAILED_OR_SKIPPED')
            continue
        matches = [a for a in entries if a.get('id') == entry['artifact_id']]
        need(len(matches) == 1, 'NEW_PINNED_ARTIFACT_MISSING')
        a = matches[0]
        need(a.get('name') == entry['artifact_name'] and a.get('expired') is False and
             a.get('size_in_bytes') == entry['zip_bytes'] and a.get('digest') == 'sha256:' + entry['zip_sha256'] and
             a.get('workflow_run', {}).get('id') == expected['run_id'] and
             a['workflow_run'].get('head_sha') == expected['github_sha'], 'NEW_ARTIFACT_RUN_IDENTITY')
    return {'semantic': {'terminal': terminal_doc, 'artifacts': pins['new_artifacts']},
            'api_raw_sha256': {name: sha(value) for name, value in raw.items()}}


def extract_zip(path, destination, pin, *, max_total=96 * 1024**2, max_file=8 * 1024**2):
    raw = regular(path, max_total)
    need(len(raw) == pin['zip_bytes'] and sha(raw) == pin['zip_sha256'], 'NEW_ZIP_IDENTITY')
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(1 <= len(infos) <= 2048 and len({i.filename for i in infos}) == len(infos) and
             sum(i.file_size for i in infos) <= max_total, 'NEW_ZIP_CLOSED_BOUNDS')
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            need(parts and not info.filename.startswith('/') and '\\' not in info.filename and
                 all(p not in ('', '.', '..') for p in info.filename.split('/')) and
                 not info.is_dir() and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG) and
                 0 <= info.file_size <= max_file, 'UNSAFE_ZIP_MEMBER')
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(archive.read(info))


def classify_new(cell, row, summary_hashes):
    label = cell.get('requested_keyword_index') or 0
    base = {'cell_id': cell['cell_id'], 'role': cell['role'], 'weak_intent_class': label,
        'source_status_unchanged': row['source_status'], 'source_disposition': row['source_disposition'],
        **dict.fromkeys(FALSE_FLAGS, False)}
    reason = None
    values = row.get('observations', {})
    if row['source_disposition'] != 'VALID_UNOBSERVED_SOURCE' or row['source_status'] != 'GENERATED_UNADMITTED':
        reason = 'SOURCE_UNAVAILABLE_OR_QUARANTINED'
    elif not (type(values) is dict and set(values) == set(PROVIDERS) and
              all(type(v) is dict and v.get('observation_status') == 'COMPLETE' and
                  v.get('observation_attempted') is True for v in values.values())):
        reason = 'INCOMPLETE_FOUR_OBSERVATIONS_UNKNOWN_NO_RETRY'
    if reason:
        return {**base, 'evidence_status': 'UNKNOWN', 'phone_review_status': reason,
                'eligible_for_requested_intent_research': False, 'quarantine_reason': reason}, None
    need(all(v.get('pcm_sha256') == row['pcm_sha256'] for v in values.values()), 'FOUR_SAME_PCM_OBSERVATIONS')
    transcripts = {p: values[p].get('text') for p in PROVIDERS[:3]}
    tokens = values['phones'].get('raw_ipa_tokens')
    need(all(type(t) is str for t in transcripts.values()) and type(tokens) is list and
         all(type(t) is str and t for t in tokens), 'OBSERVATION_VALUES')
    if not tokens:
        return {**base, 'evidence_status': 'UNKNOWN', 'phone_review_status': 'EMPTY_PHONE_OBSERVATION_UNKNOWN',
                'eligible_for_requested_intent_research': False}, None
    cores = {1: old.has_core(tokens, old.SELECTION['vosen_core']), 2: old.has_core(tokens, old.SELECTION['mero_core'])}
    normalized = {p: old.normalized(text) for p, text in transcripts.items()}
    if cell['intent'] == 'wake-candidate':
        names = old.SELECTION['vosen_names' if label == 1 else 'mero_names']
        votes = sum(len(words) == 2 and words[0] == 'hey' and words[1] in names for words in normalized.values())
        useful = votes >= 2 and cores[label]
        status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else (
            'STRONG_CONFLICT' if cores[3 - label] and not cores[label] else 'UNKNOWN')
        phone = ('TARGET_CORE_MACHINE_SUPPORT_HEY_LENGTH_STRESS_UNVERIFIED' if cores[label] else
                 'TARGET_CORE_NOT_ESTABLISHED_BY_THIS_OBSERVER')
    else:
        votes = sum(words == old.normalized(cell['synthesis_text']) for words in normalized.values())
        useful = votes >= 2 and not any(cores.values())
        status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else ('STRONG_CONFLICT' if any(cores.values()) else 'UNKNOWN')
        phone = 'NO_TARGET_CORE_MACHINE_OBSERVED_NOT_TRUSTED_NEGATIVE' if not any(cores.values()) else 'TARGET_CORE_IN_ORDINARY_INTENT_CONFLICT'
    observation = {'source_pcm_sha256': row['pcm_sha256'],
        'observer_artifact_sha256': sha(canonical(summary_hashes)),
        'observer_artifact_identity_kind': 'FOUR_SAVED_NEW_SUMMARY_MANIFEST',
        'observer_summaries_sha256': summary_hashes,
        'observation_sha256': sha(canonical({'cell_id': cell['cell_id'], 'pcm_sha256': row['pcm_sha256'],
            'transcripts': transcripts, 'raw_ipa_tokens': tokens})),
        'evidence_status': status, 'phone_review_status': phone}
    return {**base, 'asr_votes': votes, 'machine_phone_core': {str(k): v for k, v in cores.items()},
        'transcripts': transcripts, 'raw_ipa_tokens': tokens, 'evidence_status': status,
        'phone_review_status': phone, 'eligible_for_requested_intent_research': useful}, observation


def combine_snapshot(old_recipe, ledger, selection, reviewed):
    """New derived view only. All validated PCM, even new internal duplicates, count."""
    old_cells, new_cells = old_recipe['cells'], reviewed['recipe']['cells']
    need(len(old_cells) == len(ledger['rows']) == len(selection['decisions']) == 144 and
         len(new_cells) == len(reviewed['rows']) == 16, 'EXACT_144_PLUS_16_DISPOSITIONS')
    cells = old_cells + new_cells
    need(len({c['cell_id'] for c in cells}) == 160, 'DISTINCT_NEW_REQUEST_IDS')
    sources = ledger['rows'] + [{k: v for k, v in row.items() if k != 'observations'} for row in reviewed['rows']]
    by_source = {r['cell_id']: r for r in sources}
    need(len(by_source) == 160 and set(by_source) == {c['cell_id'] for c in cells}, 'FULL_160_SOURCE_ROSTER')
    for cell in cells:
        voice = cell.get('voice_id')
        need(voice in VOICE_ROLES and cell.get('role') == VOICE_ROLES[voice] and
             cell.get('source_family_id') == 'qwen3-preset-' + voice and
             all(cell.get(k) is False for k in FALSE_FLAGS), 'IMMUTABLE_VOICE_ROLE_FAMILY')
        label = cell.get('requested_keyword_index') or 0
        need((cell.get('intent') == 'ordinary-negative-candidate' and label == 0 and cell.get('canonical_phrase') is None) or
             (cell.get('intent') == 'wake-candidate' and label in (1, 2) and
              cell.get('canonical_phrase') == {1: 'Hey Vosen', 2: 'Hey Mero'}[label] and
              cell.get('target_ipa') == {1: '/heɪ ˈvoʊzən/', 2: '/heɪ ˈmiːroʊ/'}[label]), 'CURRENT_INTENT_IDENTITY')
        source = by_source[cell['cell_id']]
        need(all(source.get(k) is False for k in FALSE_FLAGS) and
             all(source.get(k) == cell[k] for k in ('role', 'source_family_id', 'voice_id')), 'ORIGINAL_ROW_ROLE_IDENTITY')
    need(all(c['voice_id'] in ('aiden', 'eric') and c['role'] == 'train' and
             c.get('requested_keyword_index') == 1 for c in new_cells), 'NEW_SCHWA16_SCOPE')
    verified = [r for r in sources if r.get('source_status') == 'GENERATED_UNADMITTED' and
                r.get('source_disposition') in ('VALID_UNOBSERVED_SOURCE', 'DUPLICATE_PCM_QUARANTINED')]
    need(all(hex_id(r.get('pcm_sha256')) and hex_id(r.get('wav_sha256')) and
             type(r.get('frames')) is int and 400 <= r['frames'] <= 192000 for r in verified), 'VERIFIED_SOURCE_PCM_IDENTITIES')
    counts = Counter(r['pcm_sha256'] for r in verified)
    duplicate_groups = {digest: [r['cell_id'] for r in verified if r['pcm_sha256'] == digest]
                        for digest, count in counts.items() if count > 1}
    decisions = {d['cell_id']: dict(d) for d in selection['decisions']}
    observations = dict(selection['observations'])
    new_rows = {r['cell_id']: r for r in reviewed['rows']}
    for cell in new_cells:
        decision, observation = classify_new(cell, new_rows[cell['cell_id']], reviewed['binding']['observer_summaries_sha256'])
        decisions[cell['cell_id']] = decision
        if observation is not None:
            observations[cell['cell_id']] = observation
    rows, exclusions, ordered_decisions, eligible = [], [], [], []
    for cell in cells:
        cid = cell['cell_id']; source = by_source[cid]; decision = dict(decisions[cid])
        if source.get('pcm_sha256') in duplicate_groups and source in verified:
            decision.update(eligible_for_requested_intent_research=False,
                derived_exclusion='WHOLE_CROSS_SNAPSHOT_DUPLICATE_PCM_GROUP',
                original_evidence_status=decision['evidence_status'], evidence_status='UNKNOWN')
        ordered_decisions.append(decision)
        if not decision['eligible_for_requested_intent_research']:
            exclusions.append({'id': cid, 'role': cell['role'], 'weak_class': cell.get('requested_keyword_index') or 0,
                'reason': decision.get('derived_exclusion', decision.get('quarantine_reason', decision['evidence_status']))})
            continue
        observation = observations[cid]
        need(observation['evidence_status'] == 'USEFUL_FOR_INTENT_RESEARCH' and
             observation['source_pcm_sha256'] == source['pcm_sha256'], 'PCM_BOUND_ELIGIBILITY')
        eligible.append(cid)
        rows.append({**source, 'status': source['source_status'], 'role': cell['role'],
            'weak_class': cell.get('requested_keyword_index') or 0, 'source_family_id': cell['source_family_id'],
            'target_basis': 'requested-intent-not-acoustic-truth', 'source_observation': observation,
            'phone_review_status': observation['phone_review_status'], 'independent_evaluation': False,
            **dict.fromkeys(FALSE_FLAGS, False)})
    selected = {'policy': old.SELECTION, 'policy_sha256': SELECTION_SHA, 'source_denominator': 160,
        'source_order': [c['cell_id'] for c in cells], 'eligible_cell_ids': eligible,
        'observations': observations, 'decisions': ordered_decisions, 'duplicate_groups': duplicate_groups,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'new_batch_status': reviewed.get('status', reviewed.get('combined', {}).get('status'))}
    selected_sha = sha(canonical(selected))
    for row in rows:
        row['research_selection_sha256'] = selected_sha
    return {'recipe_cells': cells, 'source_rows': sources, 'selection': selected, 'rows': rows,
            'minimum': old.minimum_evidence(rows), 'exclusions': exclusions}


def semantic_source_bindings(old_binding, old_observed, new_binding):
    """Whitelist source meaning; exclude raw API hashes and their derived hashes."""
    old_semantic = {k: old_binding[k] for k in ('source_identity', 'observer_identity',
        'original_campaign_status', 'source_denominator', 'valid_source_count', 'source_dispositions',
        'source_ledger_canonical_sha256', 'terminal_evidence_canonical_sha256', 'recipe_sha256',
        'combined_observations_sha256', 'observer_summaries_sha256')}
    old_semantic.update(observer_artifacts=old_observed['artifacts'],
        observer_terminal_evidence_canonical_sha256=old_observed['observer_terminal_evidence_canonical_sha256'])
    new_semantic = {k: new_binding[k] for k in ('source_identity', 'observer_identity', 'recipe_sha256',
        'source_ledger_canonical_sha256', 'source_result_sha256', 'source_plan_sha256',
        'source_model_lock_sha256', 'observer_summaries_sha256', 'observer_model_locks_sha256')}
    return old_semantic, new_semantic


def collect(inputs, pins, temporary):
    """Authenticate original archives twice if needed; never call an observer."""
    temporary.mkdir(parents=True, exist_ok=True)
    for filename, expected in (('old-training-pins.json', OLD_TRAIN_PINS_SHA),
                               ('old-observation-pins.json', OLD_OBSERVATION_PINS_SHA),
                               ('new-execution-pins.json', pins['new_execution_pins_sha256'])):
        need(sha(regular(inputs / filename)) == expected, 'EXACT_SAVED_INPUT_PINS:' + filename)
    old_pins = read_json(inputs / 'old-training-pins.json')
    terminal.validate_pins(old_pins)
    old_output = temporary / 'old-documents'; old_output.mkdir()
    (temporary / 'old-private').mkdir()
    ledger, old_binding, old_selection, wavs = terminal.readback(SimpleNamespace(
        source_bundle=inputs / 'old-source', observer_bundle=inputs / 'old-observer',
        observation_pins=inputs / 'old-observation-pins.json'), old_pins, old, old_observer,
        temporary / 'old-private', old_output)
    need(sha(canonical(ledger)) == OLD_LEDGER_SHA and sha(canonical(old_selection)) == OLD_SELECTION_SHA,
         'ORIGINAL_144_EVIDENCE_UNCHANGED')
    old_observed = read_json(old_output / 'OBSERVER_BINDING.json')
    new_bundle = inputs / 'new'
    metadata = new_metadata(new_bundle, pins)
    expected_zips = {'source.zip'} | {'observer-' + p + '.zip' for p in PROVIDERS if pins['new_artifacts'][p] is not None}
    need({p.name for p in new_bundle.glob('*.zip')} == expected_zips, 'EXACT_NEW_ZIP_SET')
    source_root = temporary / 'new-source'
    observers_root = temporary / 'new-observers'; observers_root.mkdir()
    extract_zip(new_bundle / 'source.zip', source_root, pins['new_artifacts']['source'])
    for provider in PROVIDERS:
        if pins['new_artifacts'][provider] is not None:
            extract_zip(new_bundle / ('observer-' + provider + '.zip'),
                        observers_root / ('observer-' + provider), pins['new_artifacts'][provider])
    need(sha(regular(HERE / 'vosen_schwa16_recipe.py', 256 * 1024)) == pins['new_recipe_code_sha256'],
         'NEW_RECIPE_CODE_PIN')
    reader = load_file('_snapshot160_new_reader', 'vosen_schwa16_once.py', pins['new_worker_sha256'])
    reviewed = reader.review_rows(source_root, observers_root, read_json(inputs / 'new-execution-pins.json'))
    binding = reviewed['binding']
    need(sha(canonical(reviewed['recipe'])) == pins['new_recipe_sha256'] == binding.get('recipe_sha256') and
         binding.get('test_backend') is False and all(binding.get(k) is False for k in FALSE_FLAGS),
         'REAL_NEW_RECIPE_BINDING')
    for name in ('source_identity', 'observer_identity'):
        actual = binding.get(name, {})
        need(actual.get('run_id') == str(pins['new_run']['run_id']) and
             actual.get('github_sha') == pins['new_run']['github_sha'] and
             actual.get('run_attempt') == '1' and actual.get('test_backend') is False, 'NEW_STAGE_IDENTITY')
    for item in reviewed['valid']:
        cid = item['cell']['cell_id']
        need(cid not in wavs, 'DISTINCT_AUDIO_ID')
        wavs[cid] = regular(item['wav'], 512 * 1024)
    combined = combine_snapshot(old_observer.recipe.recipe(), ledger, old_selection, reviewed)
    need(type(combined['selection']['new_batch_status']) is str, 'EXPLICIT_NEW_PARTIAL_OR_COMPLETE_STATUS')
    old_semantic, new_semantic = semantic_source_bindings(old_binding, old_observed, binding)
    semantic = {'schema': SCHEMA, 'source_denominator': 160, 'old_declared': 144, 'new_declared': 16,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'old_source': old_semantic,
        'new_source': new_semantic, 'new_terminal': metadata['semantic'],
        'new_batch_status': combined['selection']['new_batch_status'],
        'source_pins': source_pins(pins), 'prepared_worker_sha256': sha(regular(Path(__file__), 256 * 1024)),
        'recipe_cells': combined['recipe_cells'], 'source_rows': combined['source_rows'],
        'selection': combined['selection'], 'minimum': combined['minimum'], 'exclusions': combined['exclusions'],
        'new_valid_observation_cohort': len(reviewed['valid']),
        'new_source_dispositions': dict(Counter(r['source_disposition'] for r in reviewed['rows'])),
        'test_backend': False, **dict.fromkeys(FALSE_FLAGS, False)}
    transport = {'old_source_binding': old_binding, 'old_observer_binding': old_observed,
                 'new_binding': binding, 'new_metadata': metadata}
    return semantic, transport, combined['rows'], wavs


def closed_manifest(directory):
    files = {}
    total = 0
    for path in sorted(directory.rglob('*')):
        need(not path.is_symlink(), 'SNAPSHOT_SYMLINK')
        if path.is_file() and path != directory / 'MANIFEST.json':
            relative = path.relative_to(directory).as_posix()
            files[relative] = identity(path); total += files[relative]['bytes']
    need(1 <= len(files) <= 128 and total <= MAX_SNAPSHOT_BYTES, 'SNAPSHOT_BOUNDS')
    return {'schema': SCHEMA + '-byte-manifest', 'files': files}


def verify_snapshot_manifest(directory, pins):
    """Stdlib pre-install handoff check; snapshot pin is semantic, manifest raw."""
    directory = Path(directory)
    manifest = read_json(directory / 'MANIFEST.json')
    need(manifest == closed_manifest(directory), 'CLOSED_SNAPSHOT_BYTE_MANIFEST')
    semantic = read_json(directory / 'SNAPSHOT.json')
    digest = sha(canonical(semantic))
    need(digest == pins['snapshot_sha256'] and semantic.get('source_pins') == source_pins(pins) and
         semantic.get('prepared_worker_sha256') == pins['code_sha256'], 'EXACT_SEMANTIC_SNAPSHOT_PIN')
    return {'snapshot_sha256': digest, 'manifest_sha256': identity(directory / 'MANIFEST.json')['sha256']}


def read_snapshot(directory, pins, temporary):
    checked = verify_snapshot_manifest(directory, pins)
    semantic, transport, rows, wavs = collect(Path(directory) / 'inputs', pins, temporary)
    need(semantic == read_json(Path(directory) / 'SNAPSHOT.json') and
         transport == read_json(Path(directory) / 'TRANSPORT_BINDING.json'), 'RECOMPUTED_SNAPSHOT_AND_TRANSPORT')
    return semantic, rows, wavs, checked


def copy_input_folder(source, destination, names):
    source = Path(source); destination.mkdir(parents=True, exist_ok=False)
    need({p.name for p in source.iterdir() if p.is_file()} == set(names) and
         not any(p.is_symlink() for p in source.iterdir()), 'CLOSED_INPUT_BUNDLE_FILES')
    for name in names:
        raw = regular(source / name, 128 * 1024**2)
        with (destination / name).open('xb') as out:
            out.write(raw)


def prepare(args, pins):
    validate_pins(pins)
    output = Path(args.output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_PREPARED_SNAPSHOT')
    output.mkdir(mode=0o700)
    inputs = output / 'inputs'; inputs.mkdir()
    try:
        for filename, path in [('old-training-pins.json', HERE / 'terminal144_training_pins.json'),
            ('old-observation-pins.json', HERE / 'terminal144_observation_pins.json'),
            ('new-execution-pins.json', Path(args.new_pins))]:
            (inputs / filename).write_bytes(regular(path))
        api = ['run.json', 'jobs.json', 'artifacts.json']
        copy_input_folder(args.old_source_bundle, inputs / 'old-source', api + ['shard-%02d.zip' % i for i in range(9)])
        copy_input_folder(args.old_observer_bundle, inputs / 'old-observer', api +
                          ['observer-' + p + '.zip' for p in PROVIDERS] + ['combined.zip'])
        copy_input_folder(args.new_bundle, inputs / 'new', api + ['source.zip'] +
                          ['observer-' + p + '.zip' for p in PROVIDERS if pins['new_artifacts'][p] is not None])
        with tempfile.TemporaryDirectory(prefix='snapshot160-prepare-') as temporary:
            semantic, transport, _, _ = collect(inputs, pins, Path(temporary))
        _, code = old.pipeline_libraries(Path(args.pipeline_source), PIPELINE_COMMIT)
        once(output / 'PIPELINE_SOURCE.json', code)
        once(output / 'SNAPSHOT.json', semantic)
        once(output / 'TRANSPORT_BINDING.json', transport)
        once(output / 'MINIMUM_AFTER_DEDUPLICATION.json', semantic['minimum'])
        semantic_sha = sha(canonical(semantic))
        # This result deliberately has no wall time or local path in snapshot identity.
        once(output / 'RESULT.json', {'schema': SCHEMA, 'status': 'PREPARED_SNAPSHOT160_WEAK_RESEARCH',
            'snapshot_sha256': semantic_sha, 'minimum': semantic['minimum'],
            'source_denominator': 160, 'optimizer_updates': 0, 'model_calls': 0,
            'new_batch_status': semantic['new_batch_status'], 'execution_enabled_by_prepare': False,
            **dict.fromkeys(FALSE_FLAGS, False)})
        manifest_sha = once(output / 'MANIFEST.json', closed_manifest(output))
        print(json.dumps({'status': 'PREPARED_SNAPSHOT160_WEAK_RESEARCH', 'snapshot_sha256': semantic_sha,
            'manifest_sha256': manifest_sha, 'minimum': semantic['minimum'], 'optimizer_updates': 0, 'model_calls': 0}, sort_keys=True))
        return 0 if semantic['minimum']['passed'] else 2
    except Exception as error:
        once(output / 'FAILURE.json', {'status': 'PREPARATION_FAILED_NO_RETRY', 'error_type': type(error).__name__,
            'reason': old_observer.error_message(error), 'optimizer_updates': 0, 'model_calls': 0})
        raise


def archive_snapshot(snapshot, output):
    """Preserve the actual prepared transport bytes for the future saved validator."""
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(Path(snapshot).rglob('*')):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(snapshot).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100600 << 16
                archive.writestr(info, regular(path, MAX_SNAPSHOT_BYTES))


def training_source_binding(semantic, checked, archive_identity):
    return {**checked, 'transport_manifest_sha256': checked['manifest_sha256'],
        'snapshot_archive_identity': archive_identity,
        'old_source': semantic['old_source'], 'new_source': semantic['new_source'],
        'new_terminal': semantic['new_terminal'], 'new_batch_status': semantic['new_batch_status'],
        'source_denominator': 160, 'old_declared': 144, 'new_declared': 16,
        'old_valid_source_count': 143, 'new_valid_observation_cohort': semantic['new_valid_observation_cohort'],
        'new_source_dispositions': semantic['new_source_dispositions'],
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'test_backend': False,
        **dict.fromkeys(FALSE_FLAGS, False)}


def saved_source_evidence(root):
    """Pure saved-source readback, independent of the current Actions environment."""
    root = Path(root)
    names = ('STARTED.json', 'RESULT.json', 'TRAINING_PINS.json', 'SOURCE_BINDING.json',
             'SELECTION.json', 'MINIMUM_AFTER_DEDUPLICATION.json', 'MINIMUM_AFTER_FEATURE_GEOMETRY.json')
    evidence = {name: read_json(root / name) for name in names}
    pins = evidence['TRAINING_PINS.json']; validate_pins(pins, execution=True)
    source_zip = root / 'SOURCE_SNAPSHOT.zip'; source_identity = identity(source_zip)
    need(source_identity == evidence['SOURCE_BINDING.json']['snapshot_archive_identity'], 'SAVED_SNAPSHOT_ARCHIVE')
    with tempfile.TemporaryDirectory(prefix='snapshot160-saved-') as temporary:
        temp = Path(temporary)
        extract_zip(source_zip, temp / 'snapshot', {'zip_bytes': source_identity['bytes'], 'zip_sha256': source_identity['sha256']},
                    max_total=MAX_SNAPSHOT_BYTES, max_file=128 * 1024**2)
        semantic, _, _, checked = read_snapshot(temp / 'snapshot', pins, temp / 'readback')
    binding = evidence['SOURCE_BINDING.json']
    need(binding == training_source_binding(semantic, checked, source_identity) and
         evidence['SELECTION.json'] == semantic['selection'] and
         evidence['MINIMUM_AFTER_DEDUPLICATION.json'] == semantic['minimum'], 'SAVED_SEMANTIC_SOURCE_BINDING')
    started = evidence['STARTED.json']
    need(started['source_denominator'] == 160 and started['snapshot_sha256'] == pins['snapshot_sha256'] and
         started['training_pins_sha256'] == sha(canonical(pins)) and started['worker_code_sha256'] == pins['code_sha256'] and
         started['run_attempt'] == '1' and type(started['run_id']) is str and started['run_id'].isdigit() and
         started['run_id'] not in ('43816869386', '43822597212', str(pins['new_run']['run_id'])) and
         hex_id(started['github_sha'], 40) and started['github_sha'] not in
             (pins['next_main_before'], pins['new_run']['github_sha'],
              terminal.SOURCE_IDENTITY['github_sha'], '21d807461c0afa30037362ddde7ea527d87cf3e6') and
         all(started.get(k) is False for k in FALSE_FLAGS) and started.get('test_backend') is False,
         'ACTUAL_DISTINCT_TRAINING_IDENTITY')
    return evidence


def feature_minimum(rows, bag_ids):
    """Only original selected rows count; background tracks never fill quotas."""
    usable = set(bag_ids)
    return old.minimum_evidence([row for row in rows if row['cell_id'] in usable])


def execute(args, pins):
    execution_guard(pins, read_json(Path(os.environ.get('GITHUB_EVENT_PATH', ''))), os.environ)
    output = Path(args.output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_TRAINING_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'run_id': os.environ['GITHUB_RUN_ID'], 'github_sha': os.environ['GITHUB_SHA'],
        'run_attempt': '1', 'worker_code_sha256': sha(regular(Path(__file__), 256 * 1024)),
        'training_pins_sha256': sha(canonical(pins)), 'snapshot_sha256': pins['snapshot_sha256'],
        'source_denominator': 160, 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
        'maximum_optimizer_updates': 96, 'new_tts_asr_phone_calls': 0, 'test_backend': False,
        'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    once(output / 'TRAINING_PINS.json', pins)
    stage, trained, training_started = 'saved-snapshot-readback', None, False
    clock = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='snapshot160-execute-') as temporary:
            semantic, rows, wavs, checked = read_snapshot(Path(args.snapshot), pins, Path(temporary))
        libs, code = old.pipeline_libraries(Path(args.pipeline_source), PIPELINE_COMMIT)
        need(code == read_json(Path(args.snapshot) / 'PIPELINE_SOURCE.json'), 'FROZEN_PIPELINE_CODE_READBACK')
        archive_snapshot(Path(args.snapshot), output / 'SOURCE_SNAPSHOT.zip')
        binding = training_source_binding(semantic, checked, identity(output / 'SOURCE_SNAPSHOT.zip'))
        once(output / 'SOURCE_BINDING.json', binding)
        once(output / 'SELECTION.json', semantic['selection'])
        minimum = old.minimum_evidence(rows)
        need(minimum == semantic['minimum'], 'SAVED_FIRST_MINIMUM')
        once(output / 'MINIMUM_AFTER_DEDUPLICATION.json', minimum)
        excluded = list(semantic['exclusions'])
        if not minimum['passed']:
            once(output / 'EXCLUSIONS.json', excluded)
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_SNAPSHOT_MINIMUM_BEFORE_TRAINING',
                'optimizer_updates': 0, 'model_calls': 0, 'minimum': minimum})
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
            'unique_research_track_samples': sum(t['frames'] for t in tracks),
            'prior_events_reused': False, 'source_pool_is_previously_exposed': True})
        del background
        cases = libs['weak_audio_cases'].assemble_audio_cases(rows, wavs, tracks, excluded=excluded)
        expected_calls = sum(max(0, 1 + (len(c['pcm16le']) // 2 - 16000) // 1280) for c in cases)
        need(expected_calls <= 40000, 'FINITE_FROZEN_EMBEDDING_BUDGET')
        stage = 'frozen-feature-extraction'
        embedding = libs['causal_embedding'].FrozenOnnxEmbedding(args.assets)
        once(output / 'FEATURE_PREFLIGHT.json', {'geometry': embedding.geometry,
            'maximum_embedding_calls': expected_calls, 'asset_identities': libs['weak_embedding_contract'].ASSETS,
            'pipeline_source': code})
        bags, streams, evidence = libs['weak_audio_cases'].extract_feature_bags(cases, embedding, excluded=excluded)
        need(sum(e['embedding_calls'] for e in evidence) <= expected_calls, 'FEATURE_FORWARD_ACCOUNTING')
        evidence_sha = once(output / 'FEATURE_EVIDENCE.json', evidence)
        once(output / 'EXCLUSIONS.json', excluded)
        usable = {bag['id'] for bag in bags}
        final_minimum = feature_minimum(rows, usable)
        once(output / 'MINIMUM_AFTER_FEATURE_GEOMETRY.json', final_minimum)
        if not final_minimum['passed']:
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_SNAPSHOT_FEATURE_ELIGIBILITY_BEFORE_TRAINING',
                'optimizer_updates': 0, 'minimum': final_minimum,
                'frozen_embedding_calls': sum(e['embedding_calls'] for e in evidence)})
            return 2
        feature_stream_sha = old.save_stream_cache(streams, cases, output)
        input_binding = {'snapshot_sha256': pins['snapshot_sha256'],
            'snapshot_transport_manifest_sha256': checked['manifest_sha256'],
            'source_binding_sha256': sha(canonical(binding)), 'selection_sha256': sha(canonical(semantic['selection'])),
            'selection_policy_sha256': SELECTION_SHA, 'old_source_ledger_sha256': OLD_LEDGER_SHA,
            'old_selection_sha256': OLD_SELECTION_SHA, 'new_recipe_sha256': pins['new_recipe_sha256'],
            'new_execution_pins_sha256': pins['new_execution_pins_sha256'],
            'new_worker_sha256': pins['new_worker_sha256'], 'new_recipe_code_sha256': pins['new_recipe_code_sha256'],
            'corpus_sha256': old.CORPUS['sha256'], 'reference_sha256': old.REFERENCE['member_sha256'],
            'pipeline_source_sha256': sha(canonical(code)), 'feature_evidence_sha256': evidence_sha,
            'feature_stream_sha256': feature_stream_sha, 'worker_code_sha256': started['worker_code_sha256'],
            'legacy_worker_sha256': terminal.LEGACY_SHA, 'terminal_worker_sha256': TERMINAL_SHA}
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
        seconds = sum(track['frames'] for track in tracks) / 16000
        events = sum(len(row['events']) for row in track_cases)
        once(output / 'RESULT.json', {**started, 'status': SUCCESS, 'optimizer_updates': 96,
            'weights_sha256': trained['weights_sha256'], 'head_binary_sha256': sha(regular(output / 'causal-head.bin')),
            'frozen_embedding_calls': sum(e['embedding_calls'] for e in evidence),
            'selected_after_deduplication': len(rows), 'usable_source_cases': sum(r['cell_id'] in usable for r in rows),
            'excluded_rows': excluded, 'minimum': final_minimum, 'input_binding': input_binding,
            'source_binding': binding, 'new_batch_status': semantic['new_batch_status'],
            'exposed_read_speech_tracks': {'tracks': len(track_cases), 'seconds': seconds, 'events': events,
                'research_events_per_hour': events * 3600 / seconds, 'reset_per_speaker': True,
                'unique_exposure_includes_no_reused_prefixes': True, 'independent_continuous_far': False},
            'wall_seconds': time.monotonic() - clock, 'independent_frr': 'NOT_RUN', 'independent_far': 'NOT_RUN',
            'word_end_latency': 'NOT_RUN', 'ssc305': 'NOT_RUN', 'deployment_frontend_arm_parity': 'NOT_RUN',
            'arm_scope': 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU'})
        return 0
    except Exception as error:
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': old_observer.error_message(error),
            'optimizer_updates': trained['optimizer_updates'] if trained is not None else
                ('UNKNOWN_AFTER_TRAINING_STARTED' if training_started else 0),
            'feature_calls': 'UNKNOWN_AFTER_FEATURE_STAGE_STARTED' if stage not in
                ('saved-snapshot-readback', 'runtime-and-original-background-preflight') else 0})
        return 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--template', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    cli.add_argument('--pins', type=Path, default=HERE / 'english_snapshot160_training_pins.json')
    cli.add_argument('--new-pins', type=Path, default=HERE / 'english_vosen_schwa16_pins.json')
    for name in ('old-source-bundle', 'old-observer-bundle', 'new-bundle', 'snapshot', 'pipeline-source',
                 'corpus-archive', 'reference-zip', 'assets', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), sort_keys=True, indent=2)); return 0
    if not args.prepare and not args.execute:
        print(json.dumps({'schema': SCHEMA, 'status': 'SOURCE_CONTRACT_ONLY', 'source_denominator': 160,
            'model_calls': 0, 'optimizer_updates': 0, 'execution_enabled': False,
            **dict.fromkeys(FALSE_FLAGS, False)}, sort_keys=True)); return 0
    required = ['pipeline_source', 'output'] + (['old_source_bundle', 'old_observer_bundle', 'new_bundle']
        if args.prepare else ['snapshot', 'corpus_archive', 'reference_zip', 'assets'])
    need(all(getattr(args, key) is not None for key in required), 'EXPLICIT_SNAPSHOT_INPUTS_REQUIRED')
    pins = read_json(args.pins)
    return prepare(args, pins) if args.prepare else execute(args, pins)


if __name__ == '__main__':
    raise SystemExit(main())
