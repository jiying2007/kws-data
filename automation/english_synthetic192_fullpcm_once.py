"""Authenticate and replay a frozen synthetic research head through native and ARM C.

The default command is source-only. Execution requires an enabled plan bound to
the actual successful synthetic training artifact and a new exact public-main successor.
Five original PCM cases and nine deterministically chosen held-out synthetic cases
are used; no selection is
based on new model scores. No compilation, training, Python/ONNX forward, audio
mixing, synthesis, or speech recognition is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import struct
import subprocess
import sys
import time
import stat
import zipfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORKER = 'automation/english_synthetic192_fullpcm_once.py'
PLAN = 'automation/english_synthetic192_fullpcm_plan.json'
SCHEMA = 'english-synthetic192-fullpcm-once-v1'
REPO = 'jiying2007/kws-data'
PARSER = 'automation/english_coverage_fullpcm_once.py'
INCREMENT = 'automation/english_hardneg_increment_once.py'
HELPERS = {
    PARSER: {'bytes': 44356, 'sha256': '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'},
    INCREMENT: {'bytes': 69681, 'sha256': 'ac89f4564bb3f93661beb02f01f7766616322a39096c94a4bdac84560f5a9ec1'},
}
PREVIOUS = {'run_id': 44049954828, 'artifact_id': 14313603148,
    'name': 'english-coverage-fullpcm-44049954828',
    'head_sha': '95aec73b0ac8b58fa74e62df35c3e443107619b1', 'bytes': 13172135,
    'sha256': 'ad5494d3a7456d9c7e1fcd51fec727fa9d9a377a508459db6f88af5256170ba6'}
PREVIOUS_MANIFEST = '48879888a6690d080e0109f2ddc27a2c3d8a36208562e9b6baa8b0826e38e640'
CONSUMERS = {
    'native': {'bytes': 2452080, 'sha256': 'ae5cdef5223be6274bb61b8ed6ab4ce5fd067cdb8f1c95faeb8f38fab6bbf9d5'},
    'arm': {'bytes': 2862184, 'sha256': 'cdee7e5a2a918315abde4953894e04a8ff8136879df5088d8d81d8a7f9dc9dd3'},
}
FALSE_FLAGS = ('formal_training_admitted', 'human_gold', 'pronunciation_verified',
    'model_promotion_allowed', 'qualification_allowed', 'kws_model_delivered', 'ssc305_qualification')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_helper(path, expected, name):
    target = ROOT / path
    need(target.is_file() and not any(p.is_symlink() for p in (target, *target.parents)), 'REGULAR_HELPER')
    need(pin(target.read_bytes()) == expected, 'UNCHANGED_HELPER:' + path)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    return (load_helper(PARSER, HELPERS[PARSER], 'increment_pcm_frozen_parser'),
            load_helper(INCREMENT, HELPERS[INCREMENT], 'increment_pcm_frozen_training'))


GATES = ('legacy_all_291_case_regression', 'legacy_natural32_zero_events',
    'fresh_development_aggregate_nonregression', 'calibration_zero_false_events',
    'locked_test_per_word_nonregression', 'locked_test_strict_positive_gain',
    'weights_changed', 'normalization_byte_identical', 'native_saved_npz_bin_parity',
    'source_balanced_reviewed_ce_decreased')
CONDITIONS = ('clean', 'snr10', 'snr5')


def fixed_plan(worker_sha, candidate=None, main_before=None, head=None, npz=None):
    return {'schema': SCHEMA, 'repository': REPO, 'enabled': candidate is not None,
        'worker_sha256': worker_sha, 'main_before': main_before,
        'candidate_source': candidate, 'candidate_head': head, 'candidate_npz': npz,
        'helper_sources': HELPERS, 'previous_fullpcm': PREVIOUS,
        'previous_manifest_sha256': PREVIOUS_MANIFEST, 'consumers': CONSUMERS,
        'fixed_case_selection': 'ORIGINAL_FIVE_PLUS_LEXICOGRAPHICALLY_FIRST_LOCKED_CASE_PER_CLASS_AND_CONDITION',
        'fresh_classes': [0, 1, 2], 'fresh_conditions': list(CONDITIONS),
        'references': 'SELECTED_FROZEN_CANDIDATE_OWN_SAVED_SCORES_AND_EMBEDDINGS',
        'quality_gates': list(GATES), 'maximum_seconds': 1800,
        'maximum_fullpcm_processes': 28, 'candidate_zip_limit_bytes': 512*1024**2,
        'candidate_expanded_limit_bytes': 1024*1024**2, 'candidate_member_limit_bytes': 128*1024**2, 'new_compilations': 0, 'optimizer_updates': 0,
        'environment_setup': 'QEMU_USER_IF_MISSING_ONE_BOUNDED_APT_UPDATE_AND_INSTALL',
        'new_python_or_onnx_forwards': 0, 'new_audio_mixes': 0, 'new_tts_asr_calls': 0,
        'retry_count': 0, 'chunk_samples': 320, 'eof_padding_samples': 0,
        'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
        'ssc305': 'NOT_RUN', 'independent_real_far': 'NOT_MEASURED',
        'independent_real_frr': 'NOT_MEASURED', 'human_word_end_latency': 'NOT_MEASURED',
        'fresh_source_scope': 'HELD_OUT_SYNTHETIC_VOICES_WITH_SHARED_PIPER_PRETRAINING',
        'new_human_recording_required': False, **dict.fromkeys(FALSE_FLAGS, False)}


def source_preflight(plan, h):
    expected = fixed_plan(h.sha(h.regular(ROOT / WORKER)), plan.get('candidate_source'),
        plan.get('main_before'), plan.get('candidate_head'), plan.get('candidate_npz'))
    need(h.canonical(plan) == h.canonical(expected), 'EXACT_SOURCE_AND_PLAN')
    if not plan['enabled']:
        need(all(plan[k] is None for k in ('candidate_source', 'main_before', 'candidate_head', 'candidate_npz')),
             'DISABLED_WITHOUT_INVENTED_RESULT_PINS')
        return
    need(re.fullmatch('[0-9a-f]{40}', plan['main_before'] or '') is not None, 'ACTUAL_NEXT_BASE')
    p = plan['candidate_source']
    need(set(p) == set(PREVIOUS) and type(p['run_id']) is int and p['run_id'] > 0
         and type(p['artifact_id']) is int and p['artifact_id'] > 0 and 0 < p['bytes'] <= 512*1024**2
         and re.fullmatch('[0-9a-f]{40}', p['head_sha']) is not None
         and re.fullmatch('[0-9a-f]{64}', p['sha256']) is not None
         and p['name'] == 'english-synthetic192-training-' + str(p['run_id']), 'REAL_CANDIDATE_ARTIFACT_PIN')
    for key, size in (('candidate_head', 197908), ('candidate_npz', 199356)):
        need(set(plan[key]) == {'bytes', 'sha256'} and plan[key]['bytes'] == size
             and re.fullmatch('[0-9a-f]{64}', plan[key]['sha256']) is not None, 'CANDIDATE_WEIGHT_PIN')


def candidate_gate(doc, plan):
    p = plan['candidate_source']
    need(doc.get('schema') == 'english-synthetic192-training-once-v1'
         and doc.get('status') == 'RETAINED_RESEARCH_SYNTHETIC192_HEAD'
         and doc.get('passed') is True and doc.get('research_retention_allowed') is True
         and doc.get('numerical_execution_completed') is True,
         'RETAINED_SYNTHETIC_CANDIDATE_REQUIRED')
    need(str(doc.get('run_id')) == str(p['run_id']) and doc.get('head_sha') == p['head_sha']
         and doc.get('run_attempt') == 1 and doc.get('repository') == REPO
         and doc.get('public_repository') is True, 'ACTUAL_TRAINING_EXECUTION_BINDING')
    need(set(doc.get('gates', {})) == set(GATES)
         and all(x is True for x in doc['gates'].values()), 'ALL_FIXED_RESEARCH_QUALITY_GATES')
    need(doc.get('strong_english_fsmn_ctc') is False
         and doc.get('ssc305_qualification') is False, 'CORRECT_ARCHITECTURE_AND_DEVICE_SCOPE')
    need(doc['candidate']['bin'] == plan['candidate_head']
         and doc['candidate']['npz'] == plan['candidate_npz'], 'FROZEN_CANDIDATE_WEIGHTS')


def selected_fresh(document):
    need(type(document) is dict and type(document.get('rows')) is list
         and document.get('schema') == 'english-synthetic192-training-once-v1/fresh-cases'
         and document.get('references_bound_to_selected_weights') is True
         and document.get('locked_test_opened') is True, 'FRESH_ROSTER_DOCUMENT')
    roster = document['rows']
    need(type(roster) is list and 9 <= len(roster) <= 576, 'FRESH_CASE_ROSTER')
    seen = set()
    for row in roster:
        need(type(row) is dict and type(row.get('id')) is str
             and re.fullmatch('[A-Za-z0-9_.-]+', row['id']) is not None
             and row['id'] not in seen, 'UNIQUE_SAFE_FRESH_CASE_ID')
        seen.add(row['id'])
    selected = []
    for label in (0, 1, 2):
        for condition in CONDITIONS:
            rows = [r for r in roster if r.get('role') == 'locked-test' and r.get('usable') is True
                    and r.get('candidate_reference_available') is True
                    and r.get('weak_class') == label and r.get('condition') == condition]
            need(rows, 'LOCKED_CASE_FOR_EVERY_CLASS_AND_CONDITION')
            selected.append(min(rows, key=lambda r: r['id']))
    return selected


def fresh_reference(candidate, row, h):
    rid = row['id']; raw = candidate.read('fresh/pcm/' + rid + '.pcm')
    doc = candidate.json('fresh/references/' + rid + '.json')
    need(type(doc) is dict and set(doc) == {'case', 'scores', 'embeddings'}
         and doc['case'] == row, 'EXACT_FRESH_REFERENCE_ROSTER')
    need(row.get('kind') == 'synthetic_source_heldout' and row.get('role') == 'locked-test'
         and row.get('tier') in ('silver', 'weak')
         and row.get('context_kind') == 'generated_non_speech', 'EXPLICIT_FRESH_SYNTHETIC_SCOPE')
    need(type(row.get('samples')) is int and 35200 <= row['samples'] <= 320000
         and h.identity(raw) == {'bytes': 2 * row['samples'], 'sha256': row['pcm_sha256']}, 'EXACT_FRESH_PCM')
    ends = list(range(16000, row['samples'] + 1, 1280))
    need(row.get('records') == len(ends) and row.get('ready') == len(ends) - 15
         and len(doc['scores']) == len(doc['embeddings']) == len(ends), 'FRESH_STREAM_GEOMETRY')
    for i, (sample, score, vec) in enumerate(zip(ends, doc['scores'], doc['embeddings'])):
        need(score.get('sample') == sample and score.get('ready') is (i >= 15)
             and score.get('keyword') in (0, 1, 2) and len(vec) == 96
             and len(score.get('logits', [])) == len(score.get('probabilities', [])) == 3
             and all(math.isfinite(x) for x in vec + score['logits'] + score['probabilities']), 'FRESH_REFERENCE_CLOCK_AND_FINITE')
        need(all(0 <= p <= 1 for p in score['probabilities']), 'FRESH_PROBABILITY_RANGE')
        if i < 15:
            need(score['keyword'] == 0 and all(x == 0 for x in score['logits'] + score['probabilities']), 'FRESH_COLD_HISTORY')
    need(row['events'] == [[s['keyword'], s['sample']] for s in doc['scores'] if s['keyword']], 'FRESH_CANDIDATE_OWN_EVENTS')
    return {'case': row, 'pcm': raw, 'reference': doc['scores'], 'embeddings': doc['embeddings']}


def select_references(mapping, rows, stream, h):
    """Index actual candidate cases by identity, never reused coverage offsets."""
    need(323 <= len(mapping) <= 899 and 18982 <= len(rows) <= 110000 and stream[:8] == b'EHSTR01\0'
         and len(stream) == 12 + 396 * len(rows) and struct.unpack_from('<I', stream, 8)[0] == len(rows)
         and sum(m['embedding_rows'] for m in mapping) == len(rows),
         'COMPLETE_CANDIDATE_REFERENCE_GEOMETRY')
    result, seen, cursor = {}, set(), 0
    for index, m in enumerate(mapping[:323]):
        key = (m['condition'], m['id'])
        count = m['embedding_rows']
        need(key not in seen and m['case_index'] == index and type(count) is int and count >= 16
             and m['ready_rows'] == count - 15 and cursor + count <= len(rows), 'UNIQUE_CANDIDATE_CASE')
        seen.add(key)
        selected, embeddings = [], []
        for ordinal, row in enumerate(rows[cursor:cursor+count]):
            sample, gap = struct.unpack_from('<QI', stream, 12 + 396*(cursor+ordinal))
            vec = list(struct.unpack_from('<96f', stream, 24 + 396*(cursor+ordinal)))
            local = 16000 + ordinal*1280
            need((row['condition'], row['id']) == key and row['case_index'] == index
                 and row['local_sample'] == local and row['sample'] == sample == m['offset'] + local
                 and gap == int(ordinal == 0) and row['ready'] is (ordinal >= 15), 'CANDIDATE_STREAM_CLOCKS')
            need(row['keyword'] in (0, 1, 2) and len(row['logits']) == len(row['probabilities']) == 3
                 and all(math.isfinite(x) for x in vec + row['logits'] + row['probabilities']), 'FINITE_CANDIDATE_REFERENCE')
            need(all(0 <= x <= 1 for x in row['probabilities']), 'CANDIDATE_PROBABILITIES')
            if ordinal < 15:
                need(row['keyword'] == 0 and all(x == 0 for x in row['logits'] + row['probabilities']), 'EMPTY_HISTORY_OUTPUT')
            selected.append(dict(sample=local, ready=row['ready'], keyword=row['keyword'],
                                 logits=row['logits'], probabilities=row['probabilities']))
            embeddings.append(vec)
        need(m['observed_samples'] >= selected[-1]['sample'] and
             m['observed_samples'] < selected[-1]['sample'] + 1280, 'EXACT_CANDIDATE_EOF')
        result[key] = {'mapping': m, 'scores': selected, 'embeddings': embeddings}
        cursor += count
    need(cursor == 18982 and sum(m['ready_rows'] for m in mapping[:323]) == 14137, 'COMPLETE_CANDIDATE_DENOMINATOR')
    return result


def stream_identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512 * 1024**2, 'BOUNDED_ARCHIVE_FILE')
    digest = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            count += len(block); digest.update(block)
            need(count <= 512 * 1024**2, 'ARCHIVE_CHANGED_OR_TOO_LARGE')
    return {'bytes': count, 'sha256': digest.hexdigest()}


class CandidateArchive:
    """Larger explicit synthetic experiment budget, without changing old helpers."""
    def __init__(self, path, expected, h, inc):
        self.h = h
        self.pin = stream_identity(path)
        need(self.pin == {k: expected[k] for k in ('bytes', 'sha256')}, 'EXACT_CANDIDATE_ARCHIVE')
        self.z = zipfile.ZipFile(path)
        infos = self.z.infolist(); names = [i.filename for i in infos]
        need(len(names) == len(set(names)) <= 10000, 'CANDIDATE_UNIQUE_MEMBERS')
        need(sum(i.file_size for i in infos) <= 1024 * 1024**2, 'CANDIDATE_EXPANSION_BOUND')
        for i in infos:
            name = i.filename[:-1] if i.is_dir() else i.filename
            need(h.safe_member(name) and not i.flag_bits & 1 and i.file_size <= 128 * 1024**2
                 and stat.S_IFMT(i.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'CANDIDATE_MEMBER_BOUND')

    def read(self, name):
        need(self.h.safe_member(name), 'SAFE_CANDIDATE_READ')
        return self.z.read(name)

    def json(self, name):
        return self.h.loads(self.read(name))

    def manifest(self, name, exact=False):
        doc = self.json(name); files = doc.get('files')
        need(type(files) is dict and 1 <= len(files) <= 10000, 'CANDIDATE_MANIFEST_ROSTER')
        for member, expected in files.items():
            need(type(expected) is dict and set(expected) == {'bytes', 'sha256'}
                 and self.h.identity(self.read(member)) == expected, 'CANDIDATE_MANIFEST_BYTES:' + member)
        if exact:
            need(set(self.z.namelist()) == set(files) | {name}, 'EXACT_CANDIDATE_MANIFEST')
        return doc


def prepare_inputs(previous_path, candidate_path, plan, h, inc):
    need(plan['enabled'] is True, 'CANDIDATE_NOT_AUTHENTICATED_YET')
    previous = h.Archive(previous_path, PREVIOUS)
    candidate = CandidateArchive(candidate_path, plan['candidate_source'], h, inc)
    previous.manifest('MANIFEST.json', exact=True)
    candidate.manifest('MANIFEST.json', exact=True)
    need(h.sha(previous.read('MANIFEST.json')) == PREVIOUS_MANIFEST, 'ORIGINAL_FULLPCM_MANIFEST')
    prior = previous.json('evidence/RESULT.json')
    need(prior.get('passed') is True and prior.get('status') == 'VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM'
         and prior.get('head_sha') == PREVIOUS['head_sha'] and str(prior.get('run_id')) == str(PREVIOUS['run_id']),
         'PREVIOUS_FULLPCM_SUCCESS')
    result = candidate.json('evidence/RESULT.json')
    candidate_gate(result, plan)
    supervisor = candidate.json('evidence/SUPERVISOR_EXIT.json')
    need(supervisor.get('passed') is True and supervisor.get('returncode') == 0
         and supervisor.get('head_sha') == plan['candidate_source']['head_sha']
         and str(supervisor.get('run_id')) == str(plan['candidate_source']['run_id'])
         and supervisor.get('timed_out') is False and supervisor.get('error') is None
         and not supervisor.get('cleanup', {}).get('descendants_found'), 'CANDIDATE_SUPERVISED_SUCCESS')
    head, npz = candidate.read('models/synthetic-head.bin'), candidate.read('models/synthetic-head.npz')
    need(h.identity(head) == plan['candidate_head'] and h.identity(npz) == plan['candidate_npz'], 'NEW_HEAD_BYTES')
    inc.verify_saved_npz_bin(npz, head)
    parity = candidate.json('evidence/HEAD_NATIVE_PARITY.json')
    need(parity.get('passed') is True and parity.get('candidate_bin') == plan['candidate_head']
         and parity.get('candidate_npz') == plan['candidate_npz'] and parity.get('head_cases', 0) >= 323
         and parity.get('embedding_rows', 0) >= 18982 and parity.get('ready_rows', 0) >= 14137, 'ALL_NATIVE_CACHED_HEAD_PARITY')
    refs = select_references(candidate.json('runtime/CASE_MAP.json'),
        [h.loads(x) for x in candidate.read('runtime/reference-frames.jsonl').splitlines()],
        candidate.read('runtime/candidate-head-stream.bin'), h)
    cases = []
    for original in h.CASES:
        rid = original['id']
        pcm = previous.read('inputs/' + rid + '.pcm')
        need(h.identity(pcm) == {'bytes': 2*original['samples'], 'sha256': original['pcm_sha256']}, 'ORIGINAL_PCM_IDENTITY')
        old = previous.json('inputs/' + rid + '.reference.json')
        need(old['case'] == original, 'ORIGINAL_CASE_IDENTITY')
        current = refs[(original['condition'], rid)]
        m = current['mapping']
        need(m['observed_samples'] == original['samples'] and m['embedding_rows'] == original['records']
             and m['ready_rows'] == original['ready'], 'SAME_ORIGINAL_FIVE_PCM_CASES')
        need(len(old['embeddings']) == len(current['embeddings']) and all(
             struct.pack('<96f', *a) == struct.pack('<96f', *b)
             for a, b in zip(old['embeddings'], current['embeddings'])), 'IDENTICAL_SAVED_EMBEDDINGS')
        case = dict(original)
        if original['kind'] == 'known_vosen_miss_preserved':
            case['kind'] = 'historical_vosen_miss_regression_case'
        case['events'] = [[r['keyword'], r['sample']] for r in current['scores'] if r['keyword']]
        case['reference_origin'] = 'NEW_SYNTHETIC192_CANDIDATE_SAVED_OUTPUT'
        case.pop('first_record')
        if original['kind'] == 'unique_rescued_development_mero':
            need(any(e[0] == 2 for e in case['events']), 'RESCUED_MERO_RETAINED')
        cases.append({'case': case, 'pcm': pcm, 'reference': current['scores'], 'embeddings': current['embeddings']})
    frozen = candidate.json('evidence/CANDIDATE_FROZEN.json')
    need(frozen.get('bin') == plan['candidate_head'] and frozen.get('npz') == plan['candidate_npz']
         and frozen.get('calibration_or_test_scores_before_freeze') == 0, 'FROZEN_BEFORE_CALIBRATION_AND_TEST')
    roster = candidate.json('fresh/CASE_ROSTER.json')
    need(roster.get('candidate') == result['candidate'], 'FRESH_ROSTER_CANDIDATE_BINDING')
    cases.extend(fresh_reference(candidate, row, h) for row in selected_fresh(roster))
    need(len(cases) == 14 and len({x['case']['id'] for x in cases}) == 14, 'FIXED_FOURTEEN_CASES')
    for backend, expected in CONSUMERS.items():
        raw = previous.read('bin/trace-' + backend)
        need(h.identity(raw) == expected, 'ORIGINAL_FULLPCM_CONSUMER')
        h.validate_elf(raw, backend)
        elf = previous.json('evidence/' + backend + '-ELF.json')
        need(elf.get('identity') == expected and elf.get('installed_library') == h.SDK_LIBRARIES[backend], 'CONSUMER_INSTALLED_LIBRARY_BINDING')
    return {'previous': previous, 'candidate': candidate, 'head': head, 'npz': npz, 'cases': cases,
        'summary': {'status': 'AUTHENTICATED_SYNTHETIC192_HEAD_FIXED_FULLPCM_INPUTS', 'model_calls': 0,
            'candidate_source': plan['candidate_source'], 'candidate_head': plan['candidate_head'],
            'cases': [x['case'] for x in cases], 'outcome_based_case_selection': False, 'new_locked_synthetic_cases': 9}}


def execution_identity(plan, h):
    env = os.environ
    need(plan['enabled'] is True and env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPO
         and env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main'
         and env.get('GITHUB_RUN_ATTEMPT') == '1', 'NEW_PUBLIC_MAIN_FIRST_PUSH_ONLY')
    sha, run = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    need(re.fullmatch('[0-9a-f]{40}', sha) is not None and run.isdigit(), 'ACTUAL_RUN_IDENTITY')
    event = h.loads(h.regular(env['GITHUB_EVENT_PATH']))
    need(event.get('before') == plan['main_before'] and event.get('after') == sha and event.get('ref') == 'refs/heads/main'
         and event.get('deleted') is False and event.get('forced') is False
         and event.get('repository', {}).get('private') is False
         and event.get('repository', {}).get('full_name') == REPO, 'EXACT_NEW_SUCCESSOR_EVENT')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD', 'HEAD^1'], text=True, timeout=10).splitlines()
    need(actual == [sha, plan['main_before']], 'EXACT_HEAD_AND_PARENT')
    return {'repository': REPO, 'run_id': run, 'head_sha': sha, 'main_before': plan['main_before'], 'run_attempt': 1}


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode())


class ReplayRunner:
    """All child commands inherit the supervised worker process group."""
    def __init__(self, output, deadline, h):
        self.h = h
        self.output, self.deadline = Path(output), deadline
        self.builds, self.passes = 0, 0

    def command(self, tag, argv, timeout=180, auth=False, maximum_stdout=16 * 1024**2, model=False, build=False):
        need(re.fullmatch('[a-z0-9_.-]+', tag) is not None, 'COMMAND_TAG')
        need(not (model and build), 'COMMAND_KIND')
        need(time.monotonic() < self.deadline, 'GLOBAL_DEADLINE')
        if model:
            need(self.passes < 28, 'TWENTYEIGHT_FULLPCM_PROCESSES_MAXIMUM')
            self.passes += 1
        if build:
            need(self.builds < 2, 'TWO_CONSUMER_BUILDS_MAXIMUM')
            self.builds += 1
        argv = [str(x) for x in argv]
        evidence = self.output / 'evidence'
        started = time.monotonic()
        write_once(evidence / (tag + '.started.json'), {'argv': argv, 'attempt': 1, 'retry_count': 0,
             'model_process': model, 'consumer_build': build, 'monotonic_started': started,
             'global_deadline': self.deadline, 'new_session': False, 'cpu_threads': 1})
        stdout, stderr = evidence / (tag + '.stdout'), evidence / (tag + '.stderr')
        process, error, timed_out = None, None, False
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=self.h.command_env(auth), start_new_session=False, cwd=str(ROOT))
                end = min(self.deadline, started + timeout)
                while process.poll() is None:
                    if time.monotonic() >= end:
                        timed_out = True
                        raise TimeoutError('COMMAND_DEADLINE')
                    if stdout.stat().st_size > maximum_stdout or stderr.stat().st_size > 4 * 1024**2:
                        raise ValueError('COMMAND_OUTPUT_LIMIT')
                    time.sleep(0.05)
                need(stdout.stat().st_size <= maximum_stdout and stderr.stat().st_size <= 4 * 1024**2, 'COMMAND_OUTPUT_LIMIT')
                need(process.returncode == 0, 'COMMAND_NONZERO_EXIT')
            except BaseException as exc:
                error = type(exc).__name__ + ':' + str(exc)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                write_once(evidence / (tag + '.exit.json'), {'argv': argv, 'returncode': None if process is None else process.returncode,
                     'error': error, 'timed_out': timed_out, 'wall_seconds': time.monotonic() - started,
                     'attempt': 1, 'retry_count': 0, 'model_process': model, 'consumer_build': build,
                     'descendants_supervised_by_parent_process_group': True})
        need(error is None, error or 'COMMAND_FAILED')
        return stdout


def download(runner, name, source, h):
    run = h.loads(h.regular(runner.command(name+'-run', ['gh','api','repos/'+REPO+'/actions/runs/'+str(source['run_id'])], auth=True)))
    need(run.get('head_sha') == source['head_sha'] and run.get('id') == source['run_id']
         and run.get('status') == 'completed' and run.get('conclusion') == 'success' and run.get('run_attempt') == 1
         and run.get('repository', {}).get('private') is False and run.get('repository', {}).get('full_name') == REPO, 'PUBLIC_SOURCE_RUN')
    a = h.loads(h.regular(runner.command(name+'-artifact', ['gh','api','repos/'+REPO+'/actions/artifacts/'+str(source['artifact_id'])], auth=True)))
    need(a.get('id') == source['artifact_id'] and a.get('name') == source['name'] and a.get('expired') is False
         and a.get('size_in_bytes') == source['bytes'] and a.get('digest') == 'sha256:'+source['sha256']
         and a.get('workflow_run', {}).get('id') == source['run_id'] and a.get('workflow_run', {}).get('head_sha') == source['head_sha'], 'PUBLIC_SOURCE_ARTIFACT')
    p = runner.command(name+'-zip', ['gh','api','repos/'+REPO+'/actions/artifacts/'+str(source['artifact_id'])+'/zip'], auth=True,
                       timeout=240, maximum_stdout=source['bytes'])
    need(stream_identity(p) == {k: source[k] for k in ('bytes','sha256')}, 'EXACT_DOWNLOADED_ZIP')
    target = runner.output/'downloads'/(name+'.zip'); target.parent.mkdir(exist_ok=True); p.rename(target)
    return target


def prepare_qemu(runner):
    """Provision only the emulator, inside the same supervised deadline."""
    if shutil.which('qemu-arm') is None:
        runner.command('qemu-apt-update', ['sudo', '-n', 'env', 'DEBIAN_FRONTEND=noninteractive',
            'apt-get', '-o', 'Acquire::Retries=0', 'update'], timeout=180, maximum_stdout=4*1024**2)
        runner.command('qemu-apt-install', ['sudo', '-n', 'env', 'DEBIAN_FRONTEND=noninteractive',
            'apt-get', '-o', 'Acquire::Retries=0', 'install', '-y', '--no-install-recommends', 'qemu-user'],
            timeout=180, maximum_stdout=4*1024**2)
    need(shutil.which('qemu-arm') is not None, 'QEMU_ARM_REQUIRED_AFTER_EXPLICIT_SETUP')
    runner.command('qemu-version', ['qemu-arm', '--version'], timeout=15)


def worker(args, plan, h, inc):
    output = args.output.absolute(); start = h.loads(h.regular(output/'STARTED.json')); binding = execution_identity(plan, h)
    need(start['binding'] == binding and start['worker_sha256'] == plan['worker_sha256']
         and start['parent_pid'] == os.getppid() and os.environ.get('KWS_SYNTHETIC192_PCM_PARENT') == str(os.getppid())
         and os.getpgrp() == os.getpid(), 'SUPERVISED_NEW_WORKER')
    inc.once(output/'evidence/WORKER_STARTED.json', {'binding': binding, 'pid': os.getpid(),
        'parent_pid': os.getppid(), 'retry_count': 0})
    runner = ReplayRunner(output, start['monotonic_deadline'], h); reports = []; prepared = None
    result = {'schema': SCHEMA+'/result', **binding, 'status': 'FAILED_NO_RETRY', 'passed': False,
        'candidate_source': plan['candidate_source'], 'candidate_head': plan['candidate_head'],
        'candidate_npz': plan['candidate_npz'], 'strong_english_fsmn_ctc': False,
        'installed_original_stdin_runner_reexecuted': False,
        'new_compilations': 0, 'optimizer_updates': 0, 'new_python_or_onnx_forwards': 0,
        'new_audio_mixes': 0, 'new_tts_asr_calls': 0, 'retry_count': 0,
        'reviewed_natural_fullpcm_cases': 0, 'independent_source_count_increment': 0,
        'ssc305': 'NOT_RUN', 'independent_far': 'NOT_MEASURED', 'independent_frr': 'NOT_MEASURED',
        'word_end_latency': 'NOT_MEASURED', **dict.fromkeys(FALSE_FLAGS, False)}
    try:
        previous = download(runner, 'previous', PREVIOUS, h)
        candidate = download(runner, 'candidate', plan['candidate_source'], h)
        prepared = prepare_inputs(previous, candidate, plan, h, inc)
        inc.once(output/'evidence/INPUT_PREFLIGHT.json', prepared['summary'])
        for name in (WORKER, PLAN, *HELPERS): h.put(output/'source'/name, h.regular(ROOT/name))
        h.put(output/'models/synthetic-head.bin', prepared['head']); h.put(output/'models/synthetic-head.npz', prepared['npz'])
        for item in prepared['cases']:
            rid = item['case']['id']; h.put(output/'inputs'/(rid+'.pcm'), item['pcm'])
            inc.once(output/'inputs'/(rid+'.reference.json'), {'case': item['case'], 'scores': item['reference'], 'embeddings': item['embeddings']})
        prepare_qemu(runner)
        paired = {key: [] for key in ('native','arm')}
        for backend in ('native','arm'):
            binary = prepared['previous'].read('bin/trace-'+backend); program=output/'bin'/('trace-'+backend)
            h.put(program, binary); program.chmod(0o500)
            for index, item in enumerate(prepared['cases']):
                case = item['case']; pcm=output/'inputs'/(case['id']+'.pcm'); head=output/'models/synthetic-head.bin'
                trace=output/'traces'/(backend+'-'+case['id']+'.trace'); trace.parent.mkdir(exist_ok=True)
                need(h.identity(h.regular(program)) == CONSUMERS[backend] and h.regular(head) == prepared['head']
                     and h.regular(pcm) == item['pcm'], 'BEFORE_NEW_HEAD_REPLAY')
                argv=([program] if backend=='native' else ['qemu-arm',program])+[head,pcm,trace]
                meta=h.loads(h.regular(runner.command(backend+'-case'+str(index),argv,timeout=300,model=True)))
                need(h.identity(h.regular(program)) == CONSUMERS[backend] and h.regular(head) == prepared['head']
                     and h.regular(pcm) == item['pcm'], 'AFTER_NEW_HEAD_REPLAY')
                parsed=h.parse_trace(h.regular(trace),case); report=h.compare_trace(item,parsed,meta); report['backend']=backend
                inc.once(output/'evidence'/(backend+'-case'+str(index)+'.comparison.json'),report)
                reports.append(report);paired[backend].append(parsed);need(report['passed'],'NEW_HEAD_FULLPCM_REFERENCE_MISMATCH')
        comparisons=[]
        for i,item in enumerate(prepared['cases']):
            a,b=paired['native'][i],paired['arm'][i]
            same=a['events']==b['events'] and all((x['sample'],x['ready'],x['keyword'])==(y['sample'],y['ready'],y['keyword'])
                for x,y in zip(a['scores'],b['scores']))
            need(same,'NEW_HEAD_NATIVE_ARM_STATES')
            comparisons.append({'case_id':item['case']['id'],'exact_states_and_events':same,
                'whole_trace_bitexact':a['trace_identity']==b['trace_identity']})
        need(runner.builds==0 and runner.passes==len(reports)==28,'EXACT_TWENTYEIGHT_REPLAYS_NO_BUILD')
        result.update(passed=True,status='VERIFIED_SYNTHETIC192_HEAD_FULLPCM_NATIVE_ARM',
            full_pcm_c_frontend_chain='VERIFIED_FIVE_LEGACY_AND_NINE_LOCKED_SYNTHETIC_CASES_NATIVE_AND_ARM_QEMU',
            paired_backends=comparisons, legacy_cases=5, locked_synthetic_cases=9,
            embeddings_per_backend=sum(x['case']['records'] for x in prepared['cases']),
            ready_per_backend=sum(x['case']['ready'] for x in prepared['cases']),
            distinct_pcm_seconds=sum(x['case']['samples'] for x in prepared['cases'])/16000,
            replayed_pcm_seconds=sum(x['case']['samples'] for x in prepared['cases'])/8000,
            software_research_delivery_verified=True)
    except BaseException as exc:
        result['error']=type(exc).__name__+':'+str(exc);h.put(output/'evidence/FAILURE_TRACEBACK.txt',traceback.format_exc().encode())
    finally:
        result.update(fullpcm_process_attempts=runner.passes,completed_comparisons=len(reports),comparisons=reports)
        inc.once(output/'evidence/RESULT.json',result)
        if prepared:
            prepared['previous'].z.close();prepared['candidate'].z.close()
    return 0 if result['passed'] else 1


def supervise(args,plan,h,inc):
    binding=execution_identity(plan,h);output=args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and not any(p.is_symlink() for p in (output,*output.parents)),'NEW_OUTPUT_ONLY')
    output.mkdir(mode=0o700);(output/'evidence').mkdir();start=time.monotonic();deadline=start+1800
    inc.once(output/'STARTED.json',{'binding':binding,'parent_pid':os.getpid(),'worker_sha256':plan['worker_sha256'],
        'monotonic_started':start,'monotonic_deadline':deadline,'maximum_seconds':1800,'retry_count':0})
    process=None;timed_out=False;error=None;cleanup={}
    def cancelled(signum,_frame):raise InterruptedError('SUPERVISOR_SIGNAL_'+str(signum))
    handlers={s:signal.signal(s,cancelled) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        argv=[sys.executable,'-I','-S','-B',str(ROOT/WORKER),'--_worker','--output',str(output)]
        with (output/'evidence/SUPERVISOR.stdout').open('xb') as out,(output/'evidence/SUPERVISOR.stderr').open('xb') as err:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env=dict(os.environ,KWS_SYNTHETIC192_PCM_PARENT=str(os.getpid())),cwd=ROOT,start_new_session=True)
            inc.once(output/'evidence/WORKER_LAUNCH.json',{'pid':process.pid,'process_group':process.pid,'argv':argv,'retry_count':0})
            while process.poll() is None:
                if time.monotonic()>=deadline:timed_out=True;raise TimeoutError('1800_SECOND_BOUND')
                time.sleep(0.1)
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        if process:
            try:cleanup=inc.cleanup_group(process.pid);process.wait(timeout=5)
            except BaseException as exc:error=error or type(exc).__name__+':'+str(exc);cleanup['descendants_found']=True
        observed=None
        if (output/'evidence/RESULT.json').exists():
            try:observed=h.loads(h.regular(output/'evidence/RESULT.json'))
            except (ValueError,OSError) as exc:error=error or 'UNREADABLE_WORKER_RESULT:'+str(exc)
        receipt={'schema':SCHEMA+'/supervisor',**binding,'returncode':None if process is None else process.returncode,
            'timed_out':timed_out,'error':error,'cleanup':cleanup,'wall_seconds':time.monotonic()-start,'retry_count':0}
        receipt['passed']=process is not None and process.returncode==0 and not timed_out and error is None and not cleanup.get('descendants_found') and isinstance(observed,dict) and observed.get('passed') is True and all(observed.get(k)==v for k,v in binding.items())
        receipt['partial_receipt_errors']=inc.close_interrupted_receipts(output,receipt)
        if receipt['partial_receipt_errors']:receipt['passed']=False
        inc.once(output/'evidence/SUPERVISOR_EXIT.json',receipt)
        if observed is None and not (output/'evidence/RESULT.json').exists():
            inc.once(output/'evidence/RESULT.json',{'schema':SCHEMA+'/result',**binding,'passed':False,
                'status':'SUPERVISOR_TERMINATED_BEFORE_WORKER_RESULT','retry_count':0,**dict.fromkeys(FALSE_FLAGS,False)})
        files={p.relative_to(output).as_posix():inc.file_identity(p) for p in sorted(output.rglob('*'))
               if p.is_file() and 'downloads' not in p.relative_to(output).parts}
        inc.once(output/'MANIFEST.json',{'schema':SCHEMA+'/artifact','files':files,'supervised_success':receipt['passed'],
            'downloads_excluded':True,'authoritative_completion_receipt':'evidence/SUPERVISOR_EXIT.json',**dict.fromkeys(FALSE_FLAGS,False)})
    print(h.canonical(receipt).decode(),end='');return 0 if receipt['passed'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    mode.add_argument('--preflight',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--previous',type=Path);parser.add_argument('--candidate',type=Path);args=parser.parse_args(argv)
    h,inc=helpers();plan=h.loads(h.regular(ROOT/PLAN));source_preflight(plan,h)
    if args.execute or args._worker:
        need(args.output is not None and args.previous is args.candidate is None,'EXECUTION_INPUTS_ARE_PINNED_DOWNLOADS')
        return worker(args,plan,h,inc) if args._worker else supervise(args,plan,h,inc)
    if args.preflight:
        need(args.previous is not None and args.candidate is not None and args.output is None,'EXACT_LOCAL_SAVED_INPUTS')
        prepared=prepare_inputs(args.previous,args.candidate,plan,h,inc)
        print(h.canonical(prepared['summary']).decode(),end='');prepared['previous'].z.close();prepared['candidate'].z.close()
        return 0
    need(args.output is args.previous is args.candidate is None,'SOURCE_ONLY_NO_OUTPUT')
    print(h.canonical({'status':'SOURCE_CONTRACT_ONLY','enabled':plan['enabled'],'model_calls':0,
        'fixed_cases':14,'maximum_fullpcm_processes':28,'new_compilations':0,'worker_sha256':plan['worker_sha256']}).decode(),end='')
    return 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
