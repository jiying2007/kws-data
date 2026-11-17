"""One supervised public-Actions full-PCM landing of the retained coverage head.

Default and --preflight are standard-library-only source/evidence checks.
Only --execute may start the single child and its ten fixed C replay processes.
Neither this file nor its child trains, mixes audio, computes Python model
references, loads an ONNX runtime, or retries a consumed operation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
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
PLAN_PATH = ROOT / 'automation/english_coverage_fullpcm_plan.json'
TRACE_DIR = ROOT / 'research/english_coverage_fullpcm_20261117/src'
REPO = 'jiying2007/kws-data'
BEFORE = '4f4d3f15214613c0c3a8f92c871da0b419b3f7f7'
SCHEMA = 'english-coverage-fullpcm-once-v1'
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold',
               'trusted_negative', 'pronunciation_verified', 'model_promotion_allowed',
               'qualification_allowed', 'kws_model_delivered', 'ssc305_qualification')
SOURCES = {
    'quality': {'run_id': 44042856615, 'artifact_id': 14310138100,
        'name': 'english-coverage-quality-44042856615', 'head_sha': BEFORE,
        'bytes': 50040751, 'sha256': '56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1'},
    'sdk': {'run_id': 43984579132, 'artifact_id': 14285590605,
        'name': 'english-snapshot160-installed-sdk-43984579132',
        'head_sha': 'cf6dbed0fb95501111820d7130bfb8ccd90e162b', 'bytes': 10810227,
        'sha256': 'b6411b7b859fa59147c7cd6091113ce6d83e544b75bc55bafa5726626b2ab511'},
    'incumbent': {'run_id': 44021084918, 'artifact_id': 14300818291,
        'name': 'english-quality-snr10-continued-44021084918',
        'head_sha': '453128e605498cae2482c76b19745b2204d9fd79', 'bytes': 32878537,
        'sha256': 'acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15'},
}
HEAD = {'bytes': 197908, 'sha256': 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'}
NPZ = {'bytes': 199356, 'sha256': '75c1d13203f77fcec9b66013bd6fc8c1979cb3397207931eaf31241eeb396a35'}
AUDIT = {'bytes': 314019, 'sha256': '9c1f1769a41552ec21a07d5cc02a645c73ca689b7f901037b28a53766454057d'}
TRACE = {
    'ekws_stream_trace.c': {'bytes': 17194, 'sha256': '30e2b78629fa4a22a3c029ffd3322608a77e9d1399cadc6c9ad6976ea78413fc'},
    'ekws_stream_trace.h': {'bytes': 2290, 'sha256': '2baee6272f0d17102a3ed035a76102da87614e997dc41f6b33a2c92177971639'},
}
SDK_MANIFEST_SHA = '09ed60d3f6701837e0e376b58bccefc4c008833c4743d53bef973b5f63d70b09'
SDK_RESULT_SHA = '1647eb276f9cd9c0e67dc2e10a86bd5c2e99d621d38dec406207ad8e1705621c'
SDK_LIBRARIES = {
    'native': {'bytes': 2441224, 'sha256': '040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a'},
    'arm': {'bytes': 2431812, 'sha256': '7865cd7fcb81cdd900899a307d304862580ee3f57e51746c47cbc07e93bb9218'},
}
PUBLIC_HEADER_SHA = 'b804e564fea7afb6dc53966de0225eb770f2f73d124768f6abac4d3e22268343'
PRIVATE_HEADER_SHA = '557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9'
REFERENCE_PINS = {
    'runtime/head-stream.bin': '614bb845016bbf7e3ea87a859972659e71628acc8b1e968e69577e8c117df72a',
    'runtime/reference-frames.jsonl': 'b50484d125c0492bd43f2e01e473124e6a8104192f824fd5917aa7ae1c4a5f07',
    'runtime/CASE_MAP.json': '7d7f64e7f0c1b7950fe7e1af19a74c82319f38f0d61cf4a159e37354d7b1ac3a',
}
CASES = [
    {'id': 'encov-new8-7ec79719666821c08c694dde', 'condition': 'new8', 'kind': 'mero_event_at_eof',
     'archive': 'quality', 'member': 'audio/new/encov-new8-7ec79719666821c08c694dde.pcm',
     'pcm_sha256': '4295f2257a0ce0b1495e55685e1366bde41bc52974cb1f489b826b1fc8593643',
     'samples': 54400, 'records': 31, 'ready': 16, 'first_record': 15873, 'events': [[2, 54400]]},
    {'id': 'encov-new8-69e0f996e6049f0d58d78d65', 'condition': 'new8', 'kind': 'vosen_event',
     'archive': 'quality', 'member': 'audio/new/encov-new8-69e0f996e6049f0d58d78d65.pcm',
     'pcm_sha256': '5d764e3f68c46b3cb049ebf2e4e5b8acb56d4598c2fa54eb374b104e71c5be8d',
     'samples': 56960, 'records': 33, 'ready': 18, 'first_record': 15904, 'events': [[1, 55680]]},
    {'id': 'encov-new8-28b17376eefb7d7a62334bab', 'condition': 'new8', 'kind': 'known_vosen_miss_preserved',
     'archive': 'quality', 'member': 'audio/new/encov-new8-28b17376eefb7d7a62334bab.pcm',
     'pcm_sha256': 'ada5d3fea5aea3083880edb6b5a77abf38c52358bbe5535f8b774e4bba5c2a42',
     'samples': 58240, 'records': 34, 'ready': 19, 'first_record': 15937, 'events': []},
    {'id': 'encov-new8-1556744d0babcec74d18c76c', 'condition': 'new8', 'kind': 'background_no_event',
     'archive': 'quality', 'member': 'audio/new/encov-new8-1556744d0babcec74d18c76c.pcm',
     'pcm_sha256': '247bc5b97ad44be4ed56a6d3654debec405a191ed0ed9aae810ee642b177854f',
     'samples': 63360, 'records': 38, 'ready': 23, 'first_record': 16003, 'events': []},
    {'id': 'enfinal144-5602dd573b0afb64e36b51b2', 'condition': 'snr10', 'kind': 'unique_rescued_development_mero',
     'archive': 'incumbent', 'member': 'audio/enfinal144-5602dd573b0afb64e36b51b2.pcm',
     'pcm_sha256': 'c21fb94514ec0b9efaf8383664a1c6c9a28e65d96d090ff05ca80379d89bb95e',
     'samples': 55680, 'records': 32, 'ready': 17, 'first_record': 12614, 'events': [[2, 54400]]},
]
LIMITS = {'embedding': {'atol': 0.001, 'rtol': 0.0002},
          'logits': {'atol': 0.003, 'rtol': 0.0002},
          'probabilities': {'atol': 0.0005, 'rtol': 0.0}}
FLAGS = ['-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def unique(pairs):
    out = {}
    for k, v in pairs:
        need(k not in out, 'DUPLICATE_JSON_KEY')
        out[k] = v
    return out


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def regular(path, limit=128 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    return path.read_bytes()


def put(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'OUTPUT_SYMLINK')
    with path.open('xb') as f:
        f.write(raw)


def once(path, value):
    put(path, canonical(value))


def expected_plan(worker_sha):
    return {'schema': SCHEMA, 'main_before': BEFORE, 'repository': REPO,
        'worker_sha256': worker_sha, 'pipeline_source_commit': 'aec192dd7c5955a7fa2b709564cdacb185a590be',
        'trace_sources': TRACE, 'sources': SOURCES, 'head': HEAD, 'npz': NPZ, 'independent_audit': AUDIT,
        'sdk_manifest_sha256': SDK_MANIFEST_SHA, 'sdk_libraries': SDK_LIBRARIES,
        'reference_pins': REFERENCE_PINS, 'cases': CASES, 'limits': LIMITS,
        'tolerance_source': 'automation/verify_terminal_frontend_once.py original frozen fullchain limits; not widened after execution',
        'compiler_flags': FLAGS, 'cpu_threads': 1, 'maximum_execution_seconds': 1200,
        'maximum_builds': 2, 'maximum_c_processes': 10, 'maximum_c_processes_per_backend': 5,
        'maximum_embeddings_per_backend': 168, 'maximum_ready_per_backend': 93,
        'samples_per_backend': 288640, 'distinct_previously_exposed_pcm_seconds': 18.04,
        'new_onnx_calls': 0, 'python_head_forwards': 0, 'optimizer_updates': 0,
        'new_mixes': 0, 'new_tts_asr_phone_calls': 0, 'retry_count': 0,
        'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
        'sample_rate': 16000, 'chunk_samples': 320, 'eof_padding_samples': 0,
        'fresh_stream_per_case': True, 'resets_inside_case': 0,
        'comparison_reference': 'SAVED_COVERAGE_PYTHON_FRAMES_AND_SAVED_ORT_EMBEDDINGS',
        'full_pcm_scope': 'INSTALLED_C_LIBRARY_PUBLIC_ABI_FIVE_PREVIOUSLY_EXPOSED_CASES',
        'same_source_conditions_are_not_independent_audio': True,
        'no_new_quality_generalization_claim': True, 'performance_measured': False,
        'reset_discontinuity_retested': False, 'original_installed_stdin_runner_reexecuted': False,
        'physical_ssc305': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'independent_frr': 'NOT_RUN',
        'word_end_latency': 'NOT_MEASURED', **dict.fromkeys(FALSE_FLAGS, False)}


def validate_plan(plan):
    need(canonical(plan) == canonical(expected_plan(sha(regular(__file__)))), 'EXACT_FROZEN_PLAN')
    for name, pin in TRACE.items():
        need(identity(regular(TRACE_DIR / name)) == pin, 'FROZEN_TRACE_SOURCE:' + name)
    return plan


def safe_member(name):
    return (type(name) is str and name and not PurePosixPath(name).is_absolute()
            and '\\' not in name and '\x00' not in name
            and all(x not in ('', '.', '..') for x in name.split('/')))


class Archive:
    def __init__(self, path, pin):
        raw = regular(path)
        need(identity(raw) == {k: pin[k] for k in ('bytes', 'sha256')}, 'ARCHIVE_PIN')
        self.z = zipfile.ZipFile(io.BytesIO(raw))
        self.pin = identity(raw)
        infos = self.z.infolist()
        names = [x.filename for x in infos]
        need(len(names) == len(set(names)) <= 10000, 'UNIQUE_ZIP_MEMBERS')
        need(sum(x.file_size for x in infos) <= 384 * 1024**2, 'ZIP_EXPANSION_LIMIT')
        for x in infos:
            name = x.filename[:-1] if x.is_dir() else x.filename
            need(safe_member(name) and not x.flag_bits & 1 and x.file_size <= 128 * 1024**2,
                 'UNSAFE_ZIP_MEMBER')
            need(stat.S_IFMT(x.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'UNSAFE_ZIP_TYPE')

    def read(self, name):
        need(safe_member(name), 'UNSAFE_READ_PATH')
        return self.z.read(name)

    def json(self, name):
        return loads(self.read(name))

    def manifest(self, name, exact=False):
        doc = self.json(name)
        need(type(doc.get('files')) is dict and 1 <= len(doc['files']) <= 10000, 'MANIFEST_ROSTER')
        for member, pin in doc['files'].items():
            need(type(pin) is dict and set(pin) == {'bytes', 'sha256'} and
                 type(pin['bytes']) is int and identity(self.read(member)) == pin, 'MANIFEST_BYTE_IDENTITY:' + member)
        if exact:
            need(set(self.z.namelist()) == set(doc['files']) | {name}, 'EXACT_MANIFEST_INVENTORY')
        return doc


def check_false(doc):
    need(all(doc.get(k) is False for k in FALSE_FLAGS[:7]), 'RESEARCH_BOUNDARY')


def prepare_inputs(quality_path, sdk_path, incumbent_path, audit_path):
    """Read and authenticate saved bytes; no import/execution of archived code."""
    archives = {k: Archive(p, SOURCES[k]) for k, p in
                [('quality', quality_path), ('sdk', sdk_path), ('incumbent', incumbent_path)]}
    quality, sdk, incumbent = (archives[k] for k in ('quality', 'sdk', 'incumbent'))
    quality.manifest('evidence/WORKER_MANIFEST.json')
    incumbent.manifest('evidence/WORKER_MANIFEST.json')
    sdk_manifest = sdk.manifest('MANIFEST.json', exact=True)
    need(sha(sdk.read('MANIFEST.json')) == SDK_MANIFEST_SHA and
         sha(sdk.read('evidence/RESULT.json')) == SDK_RESULT_SHA, 'INSTALLED_SDK_ORIGINAL_PROOF')
    sr = sdk.json('evidence/RESULT.json')
    need(sr.get('installed_sdk_software_verified') is True and sr.get('passed') is True
         and str(sr.get('run_id')) == str(SOURCES['sdk']['run_id'])
         and sr.get('github_sha') == SOURCES['sdk']['head_sha'], 'INSTALLED_SDK_SUCCESS')
    check_false(sr)
    for b, pin in SDK_LIBRARIES.items():
        need(identity(sdk.read('sdk-' + b + '/lib/libekws_research_stream.a')) == pin and
             sha(sdk.read('sdk-' + b + '/include/ekws_research_stream.h')) == PUBLIC_HEADER_SHA,
             'INSTALLED_LIBRARY_HEADER')
    need(sha(sdk.read('source/research/english_final_embedding/english_causal_head.h')) == PRIVATE_HEADER_SHA,
         'EXACT_PRIVATE_LINK_HOOK_HEADER')
    result = quality.json('evidence/RESULT.json')
    need(result.get('status') == 'COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH'
         and result.get('research_retention', {}).get('passed') is True
         and str(result.get('run_id')) == str(SOURCES['quality']['run_id'])
         and result.get('head_sha') == BEFORE and str(result.get('run_attempt')) == '1', 'RETAINED_QUALITY_RESULT')
    check_false(result)
    head = quality.read('models/coverage-head.bin')
    need(identity(head) == result.get('coverage_candidate_binary') == HEAD and
         identity(quality.read('models/coverage-head.npz')) == result.get('coverage_candidate') == NPZ,
         'COVERAGE_HEAD_IDENTITY')
    audit_raw = regular(audit_path, 1024**2)
    need(identity(audit_raw) == AUDIT, 'INDEPENDENT_AUDIT_BYTES')
    audit = loads(audit_raw)
    need(audit.get('status') == 'VERIFIED' and audit.get('expected_external_binding') is True
         and str(audit.get('expected_run')) == str(SOURCES['quality']['run_id'])
         and audit.get('expected_head') == BEFORE and audit.get('quality_artifact') == quality.pin
         and audit.get('candidate_binary') == HEAD and audit.get('candidate_head') == NPZ
         and audit.get('research_retention', {}).get('passed') is True
         and all(audit.get('parity', {}).get(b, {}).get('passed') is True for b in ('host', 'arm')),
         'INDEPENDENT_RETAINED_AND_HEAD_PARITY')
    check_false(audit)
    need(all(sha(quality.read(n)) == s for n, s in REFERENCE_PINS.items()), 'FIXED_SAVED_REFERENCE_BYTES')
    mapping = quality.json('runtime/CASE_MAP.json')
    saved = [loads(line) for line in quality.read('runtime/reference-frames.jsonl').splitlines()]
    stream = quality.read('runtime/head-stream.bin')
    need(stream[:8] == b'EHSTR01\0' and len(stream) == 12 + 396 * len(saved)
         and struct.unpack_from('<I', stream, 8)[0] == len(saved), 'SAVED_EMBEDDING_STREAM_ABI')
    cases = []
    for case in CASES:
        pcm = archives[case['archive']].read(case['member'])
        need(identity(pcm) == {'bytes': case['samples'] * 2, 'sha256': case['pcm_sha256']}, 'FIXED_PCM_CASE')
        matches = [x for x in mapping if x['id'] == case['id'] and x['condition'] == case['condition']]
        need(len(matches) == 1, 'UNIQUE_REFERENCE_CASE')
        m = matches[0]
        need(m['first_record'] == case['first_record'] and m['records'] == case['records']
             and m['observed_samples'] == case['samples'] and m['head_outputs'] == case['ready'], 'REFERENCE_CASE_GEOMETRY')
        refs, embeddings, observed_events = [], [], []
        for i in range(case['records']):
            index = case['first_record'] + i
            row = saved[index]
            sample, gap = struct.unpack_from('<QI', stream, 12 + 396 * index)
            vector = list(struct.unpack_from('<96f', stream, 24 + 396 * index))
            need(row['id'] == case['id'] and row['condition'] == case['condition'] and
                 row['local_sample'] == 16000 + 1280 * i and sample == row['sample'] == m['offset'] + row['local_sample']
                 and gap == int(i == 0) and row['ready'] is (i >= 15), 'SAVED_CLOCK_RESET_READY')
            need(all(math.isfinite(x) for x in vector + row['logits'] + row['probabilities']), 'NONFINITE_REFERENCE')
            refs.append({'sample': row['local_sample'], 'ready': row['ready'], 'keyword': row['keyword'],
                         'logits': row['logits'], 'probabilities': row['probabilities']})
            embeddings.append(vector)
            if row['keyword']:
                observed_events.append([row['keyword'], row['local_sample']])
        need(observed_events == case['events'], 'FIXED_REFERENCE_EVENTS')
        cases.append({'case': case, 'pcm': pcm, 'reference': refs, 'embeddings': embeddings})
    paired = quality.json('evidence/PAIRED_COMPARISONS.json')['snr10']
    rescued = [x for x in paired if x.get('development_miss_rescued') is True]
    need(len(rescued) == 1 and rescued[0]['id'] == CASES[-1]['id']
         and rescued[0]['previous_events'] == [] and rescued[0]['coverage_events'] == [{'keyword': 2, 'sample': 54400}]
         and rescued[0]['role'] == 'development' and rescued[0]['weak_class'] == 2, 'UNIQUE_SAVED_RESCUE')
    summary = {'status': 'FULLPCM_INPUTS_VERIFIED_NO_MODEL_EXECUTION', 'sources': {k: v.pin for k, v in archives.items()},
        'candidate_head': HEAD, 'independent_audit': AUDIT, 'cases': CASES,
        'samples_per_backend': sum(c['samples'] for c in CASES),
        'embeddings_per_backend': sum(c['records'] for c in CASES),
        'ready_per_backend': sum(c['ready'] for c in CASES), 'model_calls': 0,
        'independent_acoustic_evaluation': False, 'unique_saved_rescue_verified': True,
        'sdk_original_manifest_sha256': SDK_MANIFEST_SHA, **dict.fromkeys(FALSE_FLAGS, False)}
    return {'archives': archives, 'sdk_manifest': sdk_manifest, 'head': head, 'audit': audit_raw,
            'cases': cases, 'summary': summary, 'saved_rescue': rescued[0]}


def parse_trace(raw, case):
    """Strict complete-format reader; rejects missing, reordered or extra records."""
    need(len(raw) >= 40 and raw[:8] == b'EKWSTRC1', 'TRACE_MAGIC')
    need(struct.unpack_from('<6IQ', raw, 8) == (1, 1, 16000, 320, 96, 3, case['samples']), 'TRACE_HEADER')
    pos = 40
    def record(kind, size):
        nonlocal pos
        need(pos + 5 + size <= len(raw), 'TRACE_TRUNCATED')
        got, length = struct.unpack_from('<BI', raw, pos)
        need((got, length) == (kind, size), 'TRACE_RECORD_ORDER_LENGTH')
        pos += 5
        value = raw[pos:pos + size]
        pos += size
        return value
    embeddings, scores, events = [], [], []
    for i in range(case['records']):
        row = record(1, 400)
        ordinal, sample = struct.unpack_from('<QQ', row)
        values = list(struct.unpack_from('<96f', row, 16))
        need(ordinal == i and sample == 16000 + 1280 * i and all(math.isfinite(x) for x in values), 'TRACE_EMBEDDING')
        embeddings.append(values)
        row = record(2, 48)
        ordinal, ready, keyword, clock = struct.unpack_from('<QIIQ', row)
        values = list(struct.unpack_from('<6f', row, 24))
        need(ordinal == i and ready == int(i >= 15) and keyword in (0, 1, 2) and clock == sample
             and all(math.isfinite(x) for x in values) and all(0 <= x <= 1 for x in values[3:]), 'TRACE_SCORE')
        if not ready:
            need(keyword == 0 and all(x == 0 for x in values), 'TRACE_NONREADY_ZERO')
        scores.append({'sample': clock, 'ready': bool(ready), 'keyword': keyword,
                       'logits': values[:3], 'probabilities': values[3:]})
        if keyword:
            row = record(3, 32)
            event_ordinal, embedding_ordinal, event_keyword, event_sample, confidence = struct.unpack('<QQIQf', row)
            need(event_ordinal == len(events) and embedding_ordinal == i and event_keyword == keyword
                 and event_sample == sample and math.isfinite(confidence)
                 and struct.pack('<f', confidence) == struct.pack('<f', values[3 + keyword]), 'TRACE_EVENT')
            events.append([keyword, sample])
    end = struct.unpack('<6Q', record(255, 48))
    need(end == (case['samples'], case['records'], case['records'], case['ready'], len(events),
                 (case['samples'] + 319) // 320) and pos == len(raw), 'TRACE_EXACT_END')
    return {'embeddings': embeddings, 'scores': scores, 'events': events,
            'trace_identity': identity(raw), 'end': list(end)}


def compare_numbers(reference, actual, limit):
    need(len(reference) == len(actual), 'NUMERIC_ROW_COUNT')
    count, violations, maximum = 0, 0, 0.0
    first = None
    for i, (a, b) in enumerate(zip(reference, actual)):
        need(len(a) == len(b), 'NUMERIC_WIDTH')
        for j, (x, y) in enumerate(zip(a, b)):
            need(math.isfinite(x) and math.isfinite(y), 'NONFINITE_COMPARISON')
            delta = abs(y - x)
            count += 1
            maximum = max(maximum, delta)
            if delta > limit['atol'] + limit['rtol'] * abs(x):
                violations += 1
                if first is None:
                    first = {'row': i, 'column': j, 'reference': x, 'actual': y, 'absolute_error': delta}
    return {'passed': violations == 0, 'elements': count, 'violations': violations,
            'max_abs_error': maximum, 'first_violation': first, **limit}


def compare_trace(item, parsed, metadata):
    case = item['case']
    fixed_metadata = {'format_magic': 'EKWSTRC1', 'format_version': 1, 'abi': 1,
                      'sample_rate_hz': 16000, 'encoding': 's16le_mono', 'chunk_samples': 320,
                      'trace_bytes': parsed['trace_identity']['bytes'],
                      'trace_records': 2 * case['records'] + len(parsed['events']) + 1}
    need(all(metadata.get(k) == v and type(metadata.get(k)) is type(v)
             for k, v in fixed_metadata.items()), 'TRACE_FIXED_METADATA')
    need(metadata.get('schema') == 'english-stream-numeric-trace-v1'
         and metadata.get('trace_completed') is True and metadata.get('end_record_written') is True
         and metadata.get('input_samples') == case['samples'] and metadata.get('input_bytes') == 2 * case['samples']
         and metadata.get('embedding_records') == metadata.get('real_head_calls') == metadata.get('score_records') == case['records']
         and metadata.get('ready_outputs') == case['ready'] and metadata.get('event_records') == len(parsed['events'])
         and metadata.get('feed_calls') == (case['samples'] + 319) // 320
         and metadata.get('input_repetitions') == 1 and metadata.get('resets') == metadata.get('eof_padding_samples') == 0
         and metadata.get('score_observer_enabled') is True and metadata.get('original_head_link_wrap') is True
         and metadata.get('performance_measured') is False, 'TRACE_PROCESS_METADATA')
    exact = all((a['sample'], a['ready'], a['keyword']) == (b['sample'], b['ready'], b['keyword'])
                for a, b in zip(item['reference'], parsed['scores']))
    reports = {'embedding': compare_numbers(item['embeddings'], parsed['embeddings'], LIMITS['embedding'])}
    for key in ('logits', 'probabilities'):
        reports[key] = compare_numbers([x[key] for x in item['reference']], [x[key] for x in parsed['scores']], LIMITS[key])
    events_match = parsed['events'] == case['events']
    return {'case_id': case['id'], 'kind': case['kind'], 'passed': exact and events_match and all(x['passed'] for x in reports.values()),
        'exact_clock_ready_keyword': exact, 'exact_events': events_match, 'expected_events': case['events'],
        'actual_events': parsed['events'], 'numeric': reports, 'trace': parsed['trace_identity'],
        'input_pcm_sha256': case['pcm_sha256'], 'observed_samples': case['samples'],
        'independent_acoustic_evaluation': False, 'word_end_latency': 'NOT_MEASURED'}


def execution_identity(plan, env=None, event=None, head=None):
    env = os.environ if env is None else env
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPO
         and env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main'
         and env.get('GITHUB_RUN_ATTEMPT') == '1', 'PUBLIC_MAIN_FIRST_PUSH_ONLY')
    commit, run_id = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None and run_id.isdigit(), 'ACTUAL_RUN_HEAD')
    if event is None:
        event = loads(regular(env['GITHUB_EVENT_PATH'], 8 * 1024**2))
    need(event.get('before') == plan['main_before'] and event.get('after') == commit
         and event.get('ref') == 'refs/heads/main' and event.get('repository', {}).get('private') is False
         and event.get('repository', {}).get('full_name') == REPO and not event.get('deleted'), 'EXACT_PUBLIC_SUCCESSOR_EVENT')
    if head is None:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    need(head == commit, 'WORKTREE_EQUALS_GITHUB_SHA')
    return {'repository': REPO, 'run_id': run_id, 'head_sha': commit, 'main_before': BEFORE,
            'run_attempt': 1, 'event': 'push', 'public_repository': True}


def command_env(auth=False):
    env = dict(os.environ)
    if not auth:
        for key in tuple(env):
            if key in ('GH_TOKEN', 'GITHUB_TOKEN') or key.endswith('_TOKEN') or key.startswith('ACTIONS_RUNTIME_'):
                env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return env


class Runner:
    """All child commands inherit the supervised worker process group."""
    def __init__(self, output, deadline):
        self.output, self.deadline = Path(output), deadline
        self.builds, self.passes = 0, 0

    def command(self, tag, argv, timeout=180, auth=False, maximum_stdout=16 * 1024**2, model=False, build=False):
        need(re.fullmatch('[a-z0-9_.-]+', tag) is not None, 'COMMAND_TAG')
        need(not (model and build), 'COMMAND_KIND')
        need(time.monotonic() < self.deadline, 'GLOBAL_DEADLINE')
        if model:
            need(self.passes < 10, 'TEN_FULLPCM_PROCESSES_MAXIMUM')
            self.passes += 1
        if build:
            need(self.builds < 2, 'TWO_CONSUMER_BUILDS_MAXIMUM')
            self.builds += 1
        argv = [str(x) for x in argv]
        evidence = self.output / 'evidence'
        started = time.monotonic()
        once(evidence / (tag + '.started.json'), {'argv': argv, 'attempt': 1, 'retry_count': 0,
             'model_process': model, 'consumer_build': build, 'monotonic_started': started,
             'global_deadline': self.deadline, 'new_session': False, 'cpu_threads': 1})
        stdout, stderr = evidence / (tag + '.stdout'), evidence / (tag + '.stderr')
        process, error, timed_out = None, None, False
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=command_env(auth), start_new_session=False, cwd=str(ROOT))
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
                once(evidence / (tag + '.exit.json'), {'argv': argv, 'returncode': None if process is None else process.returncode,
                     'error': error, 'timed_out': timed_out, 'wall_seconds': time.monotonic() - started,
                     'attempt': 1, 'retry_count': 0, 'model_process': model, 'consumer_build': build,
                     'descendants_supervised_by_parent_process_group': True})
        need(error is None, error or 'COMMAND_FAILED')
        return stdout

    def download(self, name):
        pin = SOURCES[name]
        run = loads(regular(self.command(name + '-run', ['gh', 'api', 'repos/' + REPO + '/actions/runs/' + str(pin['run_id'])], auth=True)))
        need(run.get('id') == pin['run_id'] and run.get('head_sha') == pin['head_sha']
             and run.get('conclusion') == 'success' and run.get('status') == 'completed'
             and run.get('run_attempt') == 1 and run.get('repository', {}).get('private') is False
             and run.get('repository', {}).get('full_name') == REPO, 'PINNED_PUBLIC_SOURCE_RUN')
        artifact = loads(regular(self.command(name + '-artifact', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id'])], auth=True)))
        need(artifact.get('id') == pin['artifact_id'] and artifact.get('name') == pin['name']
             and artifact.get('expired') is False and artifact.get('size_in_bytes') == pin['bytes']
             and artifact.get('digest') == 'sha256:' + pin['sha256']
             and artifact.get('workflow_run', {}).get('id') == pin['run_id']
             and artifact.get('workflow_run', {}).get('head_sha') == pin['head_sha'], 'PINNED_SOURCE_ARTIFACT')
        path = self.command(name + '-zip', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id']) + '/zip'],
                            auth=True, timeout=240, maximum_stdout=pin['bytes'])
        need(identity(regular(path)) == {k: pin[k] for k in ('bytes', 'sha256')}, 'DOWNLOADED_EXACT_ZIP')
        target = self.output / 'downloads' / (name + '.zip')
        target.parent.mkdir(exist_ok=True)
        path.rename(target)
        return target


def validate_elf(raw, backend):
    need(len(raw) >= 64 and raw[:4] == b'\x7fELF' and raw[5] == 1
         and raw[4] == (2 if backend == 'native' else 1)
         and struct.unpack_from('<H', raw, 18)[0] == (62 if backend == 'native' else 40), 'EXPECTED_CONSUMER_ELF')
    if backend == 'arm':
        off = struct.unpack_from('<I', raw, 28)[0]
        width, count = struct.unpack_from('<HH', raw, 42)
        need(width >= 32 and 0 < count < 128 and off + width * count <= len(raw), 'ARM_PROGRAM_HEADERS')
        need(all(struct.unpack_from('<I', raw, off + i * width)[0] != 3 for i in range(count)), 'ARM_STATIC_NO_INTERPRETER')


def child(args, plan):
    output = args.output.absolute()
    start = loads(regular(output / 'STARTED.json'))
    need(os.getppid() == start['parent_pid'] and os.environ.get('KWS_FULLPCM_PARENT_PID') == str(start['parent_pid']), 'SUPERVISED_CHILD_ONLY')
    binding = execution_identity(plan)
    need(start['binding'] == binding and start['worker_sha256'] == plan['worker_sha256'], 'CHILD_SOURCE_BINDING')
    deadline = start['monotonic_deadline']
    need(type(deadline) in (int, float) and start['monotonic_started'] < deadline
         and deadline - start['monotonic_started'] == 1200 and time.monotonic() < deadline, 'CHILD_GLOBAL_DEADLINE')
    once(output / 'evidence/WORKER_STARTED.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'binding': binding, 'retry_count': 0})
    runner = Runner(output, deadline)
    reports, completed = [], 0
    result = {'schema': SCHEMA + '/result', 'status': 'FAILED_NO_RETRY', 'passed': False,
              **binding, 'candidate_head': HEAD, 'plan_sha256': sha(canonical(plan)),
              'ssc305': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'independent_frr': 'NOT_RUN',
              'word_end_latency': 'NOT_MEASURED', 'performance_measured': False,
              'new_onnx_calls': 0, 'python_head_forwards': 0, 'optimizer_updates': 0,
              'new_tts_asr_phone_calls': 0, 'new_mixes': 0, 'retry_count': 0,
              'scope': plan['full_pcm_scope'], **dict.fromkeys(FALSE_FLAGS, False)}
    try:
        paths = {name: runner.download(name) for name in ('quality', 'sdk', 'incumbent')}
        prepared = prepare_inputs(paths['quality'], paths['sdk'], paths['incumbent'], args.audit)
        once(output / 'evidence/INPUT_PREFLIGHT.json', prepared['summary'])
        put(output / 'evidence/INDEPENDENT_QUALITY_AUDIT.json', prepared['audit'])
        once(output / 'evidence/SAVED_RESCUE.json', prepared['saved_rescue'])
        put(output / 'models/coverage-head.bin', prepared['head'])
        put(output / 'models/coverage-head.npz', prepared['archives']['quality'].read('models/coverage-head.npz'))
        sdk = prepared['archives']['sdk']
        for name in prepared['sdk_manifest']['files']:
            if name.startswith(('sdk-native/', 'sdk-arm/')) or name == 'source/research/english_final_embedding/english_causal_head.h':
                put(output / 'installed-base' / name, sdk.read(name))
        put(output / 'installed-base/ORIGINAL_MANIFEST.json', sdk.read('MANIFEST.json'))
        for name in TRACE:
            put(output / 'source' / name, regular(TRACE_DIR / name))
        put(output / 'source/english_coverage_fullpcm_once.py', regular(__file__))
        once(output / 'source/PLAN.json', plan)
        for item in prepared['cases']:
            name = item['case']['id']
            put(output / 'inputs' / (name + '.pcm'), item['pcm'])
            once(output / 'inputs' / (name + '.reference.json'), {'case': item['case'],
                 'scores': item['reference'], 'embeddings': item['embeddings']})
        for name, argv in [('native-compiler-version', ['gcc', '--version']),
                           ('arm-compiler-version', ['arm-linux-gnueabihf-gcc', '--version']),
                           ('qemu-version', ['qemu-arm', '--version'])]:
            runner.command(name, argv, timeout=15)
        parsed_by_backend = {}
        for backend, compiler in [('native', 'gcc'), ('arm', 'arm-linux-gnueabihf-gcc')]:
            install = output / 'installed-base' / ('sdk-' + backend)
            library = install / 'lib/libekws_research_stream.a'
            private = output / 'installed-base/source/research/english_final_embedding'
            program = output / 'bin' / ('trace-' + backend)
            program.parent.mkdir(exist_ok=True)
            symbols = regular(runner.command(backend + '-symbols',
                              [('nm' if backend == 'native' else 'arm-linux-gnueabihf-nm'), '--print-file-name', library]))
            need(re.search(rb'ekws_research_stream\.c\.o:.*\bU ehead_step\b', symbols) is not None
                 and re.search(rb'english_causal_head\.c\.o:.*\bT ehead_step\b', symbols) is not None, 'INSTALLED_HEAD_LINK_HOOK')
            argv = [compiler, *FLAGS, *(['-static'] if backend == 'arm' else []),
                    '-I' + str(install / 'include'), '-I' + str(private), '-I' + str(output / 'source'),
                    output / 'source/ekws_stream_trace.c', library, '-Wl,--wrap=ehead_step', '-lm', '-lrt', '-o', program]
            runner.command('build-' + backend, argv, timeout=180, build=True)
            executable = regular(program)
            validate_elf(executable, backend)
            once(output / 'evidence' / (backend + '-ELF.json'), {'backend': backend, 'identity': identity(executable),
                 'installed_library': SDK_LIBRARIES[backend], 'full_pcm_consumer': True, 'ssc305': 'NOT_RUN'})
            parsed_by_backend[backend] = []
            for index, item in enumerate(prepared['cases']):
                case = item['case']
                pcm, head = output / 'inputs' / (case['id'] + '.pcm'), output / 'models/coverage-head.bin'
                trace = output / 'traces' / (backend + '-' + case['id'] + '.trace')
                trace.parent.mkdir(exist_ok=True)
                checks = {str(p): identity(regular(p)) for p in (pcm, head, library, program)}
                need(checks[str(pcm)] == {'bytes': case['samples'] * 2, 'sha256': case['pcm_sha256']}
                     and checks[str(head)] == HEAD and checks[str(library)] == SDK_LIBRARIES[backend]
                     and checks[str(program)] == identity(executable), 'PRE_REPLAY_IDENTITIES')
                argv = ([program] if backend == 'native' else ['qemu-arm', program]) + [head, pcm, trace]
                meta = loads(regular(runner.command(backend + '-case' + str(index), argv, timeout=300, model=True)))
                need(all(identity(regular(p)) == pin for p, pin in checks.items()), 'POST_REPLAY_IDENTITIES')
                parsed = parse_trace(regular(trace, 4 * 1024**2), case)
                report = compare_trace(item, parsed, meta)
                report['backend'] = backend
                reports.append(report)
                parsed_by_backend[backend].append(parsed)
                once(output / 'evidence' / (backend + '-case' + str(index) + '.comparison.json'), report)
                completed += 1
                need(report['passed'], 'FULLPCM_REFERENCE_MISMATCH')
        paired = []
        for i, case in enumerate(CASES):
            a, b = (parsed_by_backend[k][i] for k in ('native', 'arm'))
            same_state = a['events'] == b['events'] and all(
                (x['sample'], x['ready'], x['keyword']) == (y['sample'], y['ready'], y['keyword'])
                for x, y in zip(a['scores'], b['scores']))
            paired.append({'case_id': case['id'], 'exact_native_arm_events_and_states': same_state,
                           'whole_trace_bitexact': a['trace_identity'] == b['trace_identity']})
            need(same_state, 'NATIVE_ARM_STATE_DIFFERENCE')
        need(runner.builds == 2 and runner.passes == completed == 10, 'EXACT_EXECUTION_BUDGET')
        result.update(status='VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM', passed=True,
                      full_pcm_c_frontend_chain='VERIFIED_FIVE_SAVED_CASES_NATIVE_AND_ARM_QEMU',
                      independent_source_count_increment=0, installed_original_stdin_runner_reexecuted=False,
                      reset_discontinuity_retested=False, distinct_pcm_seconds=18.04, replay_pcm_seconds=36.08,
                      embedding_rows_per_backend=168, ready_outputs_per_backend=93, events_per_backend=3,
                      unique_saved_rescue_preserved_in_native_and_arm=True, paired_backends=paired,
                      frozen_input_sources=SOURCES, frozen_installed_libraries=SDK_LIBRARIES)
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ':' + str(exc))
        put(output / 'evidence/FAILURE_TRACEBACK.txt', traceback.format_exc().encode())
    finally:
        result.update(consumer_build_attempts=runner.builds, full_pcm_process_attempts=runner.passes,
                      completed_comparisons=completed, comparisons=reports)
        once(output / 'evidence/RESULT.json', result)
    return 0 if result['passed'] else 1


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup_group(pid):
    if not group_exists(pid):
        return {'descendants_found': False, 'terminated': False, 'killed': False}
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {'descendants_found': False, 'terminated': False, 'killed': False}
    end = time.monotonic() + 2
    while time.monotonic() < end and group_exists(pid):
        time.sleep(0.05)
    killed = group_exists(pid)
    if killed:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {'descendants_found': True, 'terminated': True, 'killed': killed}


def supervise(args, plan):
    binding = execution_identity(plan)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists()
         and not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_SUPERVISED_OUTPUT_ONLY')
    need(identity(regular(args.audit)) == AUDIT, 'AUDIT_BEFORE_EXECUTION')
    output.mkdir(mode=0o700)
    (output / 'evidence').mkdir()
    started = time.monotonic()
    deadline = started + 1200
    once(output / 'STARTED.json', {'schema': SCHEMA + '/started', 'binding': binding,
         'parent_pid': os.getpid(), 'worker_sha256': plan['worker_sha256'],
         'monotonic_started': started, 'monotonic_deadline': deadline,
         'maximum_seconds': 1200, 'retry_count': 0, 'new_network_model_calls_at_start': 0})
    env = dict(os.environ, KWS_FULLPCM_PARENT_PID=str(os.getpid()))
    argv = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--_worker',
            '--audit', str(args.audit.absolute()), '--output', str(output)]
    process, timed_out, error, cleanup = None, False, None, {}
    try:
        with (output / 'evidence/SUPERVISOR.stdout').open('xb') as out, (output / 'evidence/SUPERVISOR.stderr').open('xb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=env, cwd=str(ROOT), start_new_session=True)
            once(output / 'evidence/WORKER_LAUNCH.json', {'pid': process.pid, 'process_group': process.pid,
                 'argv': argv, 'attempt': 1, 'retry_count': 0})
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise TimeoutError('SUPERVISOR_1200_SECONDS')
                time.sleep(0.1)
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        if process is not None:
            cleanup = cleanup_group(process.pid)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                error = error or 'WORKER_NOT_REAPED'
        receipt = {'schema': SCHEMA + '/supervisor', 'returncode': None if process is None else process.returncode,
                   'timed_out': timed_out, 'error': error, 'wall_seconds': time.monotonic() - started,
                   'retry_count': 0, 'cleanup': cleanup, **binding}
        receipt['passed'] = process is not None and process.returncode == 0 and not timed_out and error is None and not cleanup.get('descendants_found')
        once(output / 'evidence/SUPERVISOR_EXIT.json', receipt)
        files = {p.relative_to(output).as_posix(): identity(regular(p)) for p in sorted(output.rglob('*'))
                 if p.is_file() and 'downloads' not in p.relative_to(output).parts}
        once(output / 'MANIFEST.json', {'schema': SCHEMA + '/artifact', 'files': files,
             'downloads_excluded': True, 'supervised_success': receipt['passed'], **dict.fromkeys(FALSE_FLAGS, False)})
    print(canonical(receipt).decode(), end='')
    return 0 if receipt['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    for name in ('quality', 'sdk', 'incumbent', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    plan = validate_plan(loads(regular(PLAN_PATH, 1024**2)))
    if args.preflight:
        need(all(getattr(args, key) is not None for key in ('quality', 'sdk', 'incumbent', 'audit')), 'PREFLIGHT_SAVED_INPUTS_REQUIRED')
        result = prepare_inputs(args.quality, args.sdk, args.incumbent, args.audit)['summary']
    elif args.execute or args._worker:
        need(args.audit is not None and args.output is not None and
             all(getattr(args, key) is None for key in ('quality', 'sdk', 'incumbent')), 'EXECUTION_DOWNLOADS_OWN_PINNED_SOURCES')
        return child(args, plan) if args._worker else supervise(args, plan)
    else:
        need(all(getattr(args, key) is None for key in ('quality', 'sdk', 'incumbent', 'audit', 'output')), 'DEFAULT_SOURCE_ONLY_NO_INPUTS')
        result = {'status': 'SOURCE_CONTRACT_ONLY', 'execution_enabled': False, 'model_calls': 0,
                  'plan_sha256': sha(canonical(plan)), 'worker_sha256': plan['worker_sha256'],
                  'fixed_cases': len(CASES), 'full_pcm_process_limit': 10, 'maximum_seconds': 1200}
    print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
