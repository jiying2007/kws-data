"""One supervised native scan of 32 frozen natural training-pool utterances.

Default/--preflight inspect source contracts only. --execute is restricted to the
first public-main successor push and starts one 1200-second process group. This
worker never trains, runs ASR/TTS, changes the head, searches event thresholds,
resamples audio, or converts candidates into negative training labels.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
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
import tarfile
import time
import traceback
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'jiying2007/kws-data'
BEFORE = '61e524e89c86950ef8e284ecaf18fddf82d11ca1'
SCHEMA = 'english-hardneg-natural32-once-v1'
RESEARCH = 'research/english_hardneg_natural32_20261117'
WORKER_PATH = 'automation/english_hardneg_natural32_once.py'
PLAN_PATH = 'automation/english_hardneg_natural32_plan.json'
WORKFLOW_PATH = '.github/workflows/english-hardneg-natural32-once.yml'
FROZEN_PARSER = 'automation/english_coverage_fullpcm_once.py'
TRACE_DIR = 'research/english_coverage_fullpcm_20261117/src'
SOURCE_PINS = {
    FROZEN_PARSER: {'bytes': 44356, 'sha256': '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'},
    RESEARCH + '/prepare_natural32.py': {'bytes': 12145, 'sha256': '49d3bcc12c7507eb6d620a1344586665b8d0733f6dbac1f1356384fe35be7f5e'},
    RESEARCH + '/natural32-roster.json': {'bytes': 56281, 'sha256': '31457faf8efd0cf77629c1a6f8de67bc101190eb118bcfe9ac2f256acca41239'},
    RESEARCH + '/mine_saved_scores.py': {'bytes': 18201, 'sha256': '82325ea6eacfe418140639021eda29c4fd778764157c00cf29fca8d2ba90181a'},
    TRACE_DIR + '/ekws_stream_trace.c': {'bytes': 17194, 'sha256': '30e2b78629fa4a22a3c029ffd3322608a77e9d1399cadc6c9ad6976ea78413fc'},
    TRACE_DIR + '/ekws_stream_trace.h': {'bytes': 2290, 'sha256': '2baee6272f0d17102a3ed035a76102da87614e997dc41f6b33a2c92177971639'},
}
SOURCES = {
    'training': {'run_id': 43867764366, 'artifact_id': 14225893574,
        'name': 'snapshot160-weak-embedding-research-43867764366',
        'head_sha': 'd8be1c24b12cac0ba554e9a4a41e4b6ee0db18db', 'bytes': 61290521,
        'sha256': '2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1'},
    'quality': {'run_id': 44042856615, 'artifact_id': 14310138100,
        'name': 'english-coverage-quality-44042856615',
        'head_sha': '4f4d3f15214613c0c3a8f92c871da0b419b3f7f7', 'bytes': 50040751,
        'sha256': '56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1'},
    'sdk': {'run_id': 43984579132, 'artifact_id': 14285590605,
        'name': 'english-snapshot160-installed-sdk-43984579132',
        'head_sha': 'cf6dbed0fb95501111820d7130bfb8ccd90e162b', 'bytes': 10810227,
        'sha256': 'b6411b7b859fa59147c7cd6091113ce6d83e544b75bc55bafa5726626b2ab511'},
}
CORPUS = {'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz', 'bytes': 337926286,
          'sha256': '76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3'}
HEAD = {'bytes': 197908, 'sha256': 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'}
LIBRARY = {'bytes': 2441224, 'sha256': '040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a'}
SDK_MANIFEST_SHA = '09ed60d3f6701837e0e376b58bccefc4c008833c4743d53bef973b5f63d70b09'
SDK_RESULT_SHA = '1647eb276f9cd9c0e67dc2e10a86bd5c2e99d621d38dec406207ad8e1705621c'
PUBLIC_HEADER_SHA = 'b804e564fea7afb6dc53966de0225eb770f2f73d124768f6abac4d3e22268343'
PRIVATE_HEADER_SHA = '557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9'
FALSE_FLAGS = ('human_gold', 'trusted_negative', 'training_admitted', 'formal_training_admitted',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
FLAGS = ['-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']
CONFIG = {'model': HEAD, 'sample_rate': 16000, 'channels': 1, 'sample_width': 2,
          'chunk_samples': 320, 'first_embedding_sample': 16000, 'embedding_hop_samples': 1280,
          'history': 16, 'first_ready_sample': 35200, 'classes': ['background', 'Hey Vosen', 'Hey Mero'],
          'threshold_float32': struct.unpack('<f', struct.pack('<f', 0.8))[0],
          'confirm_embeddings': 3, 'cooldown_embeddings': 13, 'tie_policy': 'lowest_class_id',
          'fresh_stream_per_original': True, 'resets_inside_original': 0, 'eof_padding_samples': 0,
          'proposal_threshold': 0.5, 'proposal_pre_samples': 38400, 'proposal_post_samples': 16000,
          'maximum_candidates_per_source': 2, 'maximum_candidates_total': 32,
          'candidate_use': 'UNKNOWN_ACOUSTIC_REVIEW_ONLY',
          'profile': 'UNCHANGED_HISTORICAL_PROFILE_NO_IPA_RELABEL'}
MEMBER = re.compile(r'LibriSpeech/dev-clean/([0-9]+)/([0-9]+)/([0-9]+-[0-9]+-[0-9]+)\.flac')


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
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def regular(path, limit=128 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    return path.read_bytes()


def file_identity(path, limit=512 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            size += len(chunk)
            need(size <= limit, 'FILE_SIZE_LIMIT')
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def put(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'OUTPUT_SYMLINK')
    with path.open('xb') as target:
        target.write(raw)


def once(path, value):
    # Publish complete JSON create-only. A killed write leaves a visible pending
    # file, not an apparently complete receipt; link never overwrites a target.
    path = Path(path)
    # Parent sealing must not collide with an interrupted child's pending file.
    pending = path.with_name(path.name + '.pending-' + str(os.getpid()))
    put(pending, canonical(value))
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def expected_plan(worker_sha):
    return {'schema': SCHEMA, 'repository': REPO, 'main_before': BEFORE,
        'worker_sha256': worker_sha, 'source_pins': SOURCE_PINS, 'artifact_sources': SOURCES,
        'corpus': CORPUS, 'model': HEAD, 'native_installed_library': LIBRARY,
        'sdk_manifest_sha256': SDK_MANIFEST_SHA, 'configuration': CONFIG,
        'configuration_sha256': sha(canonical(CONFIG)), 'compiler_flags': FLAGS,
        'maximum_execution_seconds': 1200, 'cpu_threads': 1, 'maximum_builds': 1,
        'maximum_decoder_processes': 32, 'maximum_model_processes': 32,
        'observed_source_samples': 3270080, 'observed_source_seconds': 204.38,
        'expected_embedding_rows': 2174, 'expected_ready_rows': 1694,
        'roster_speakers': 24, 'new_independent_speaker_count': 0,
        'new_onnx_calls': 0, 'python_head_forwards': 0, 'optimizer_updates': 0,
        'new_tts_asr_phone_calls': 0, 'new_mixes': 0, 'gain_changes': 0,
        'resampling': False, 'concatenation': False, 'retry_count': 0,
        'replay_previous_fullpcm_cases': False, 'threshold_sweeps': False,
        'trigger': 'public main exact successor push; run_attempt=1; no workflow_dispatch',
        'source_commit': 'must equal GITHUB_SHA and checked-out HEAD',
        'scope': 'ONE_NATIVE_ORIGINAL_UTTERANCE_SCAN_PER_FROZEN_ROW_AND_UNREVIEWED_CANDIDATE_LEDGER',
        'independent_far': 'NOT_MEASURED', 'independent_frr': 'NOT_MEASURED',
        'word_end_latency': 'NOT_MEASURED', 'ssc305': 'NOT_RUN', 'performance_measured': False,
        'quality_improvement_claim': False, 'verified_negative_seconds': 0,
        'kws_model_delivered': False, **dict.fromkeys(FALSE_FLAGS, False)}


def source_preflight(root, plan, worker_path=None):
    root = Path(root)
    worker_path = root / WORKER_PATH if worker_path is None else Path(worker_path)
    need(canonical(plan) == canonical(expected_plan(sha(regular(worker_path)))), 'EXACT_FROZEN_PLAN')
    for path, pin in SOURCE_PINS.items():
        need(identity(regular(root / path)) == pin, 'SOURCE_PIN:' + path)
    roster = loads(regular(root / RESEARCH / 'natural32-roster.json'))
    need(roster.get('data_repository_commit') == BEFORE and roster.get('coverage_head_binary_sha256') == HEAD['sha256'],
         'FROZEN_SOURCE_ROLE_AND_HEAD')
    rows = roster.get('rows')
    need(type(rows) is list and len(rows) == 32 and len({r['recording_id'] for r in rows}) == 32,
         'EXACT_32_ORIGINALS')
    need(sum(r['frames'] for r in rows) == 3270080 and
         sum(r['expected_embedding_rows'] for r in rows) == 2174 and
         sum(r['expected_ready_rows'] for r in rows) == 1694, 'EXACT_ROSTER_BUDGET')
    for i, row in enumerate(rows):
        need(row['index'] == i and MEMBER.fullmatch(row['source_member']) is not None and
             row['source_role'] == row['old_bg48_role'] == row['current_coverage_role'] == 'train' and
             row['current_coverage_original_source_consumed'] is False and
             row['label_status'] == 'UNKNOWN_REVIEW_REQUIRED' and row['expected_events'] is None and
             all(row[k] is False for k in FALSE_FLAGS), 'UNCHANGED_UNKNOWN_TRAIN_SOURCE')
    return roster


def load_pinned_module(root, path, name):
    target = Path(root) / path
    need(identity(regular(target)) == SOURCE_PINS[path], 'IMPORT_SOURCE_PIN')
    spec = importlib.util.spec_from_file_location(name, target)
    need(spec is not None and spec.loader is not None, 'PURE_HELPER_MODULE')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_member(name):
    return (type(name) is str and name and not PurePosixPath(name).is_absolute()
            and '\\' not in name and '\x00' not in name
            and all(x not in ('', '.', '..') for x in name.split('/')))


class Archive:
    def __init__(self, path, pin):
        need(file_identity(path) == {k: pin[k] for k in ('bytes', 'sha256')}, 'ARCHIVE_PIN')
        self.z = zipfile.ZipFile(path)
        infos = self.z.infolist()
        need(len(infos) == len({i.filename for i in infos}) <= 10000, 'UNIQUE_ZIP_MEMBERS')
        need(sum(i.file_size for i in infos) <= 384 * 1024**2, 'ZIP_EXPANSION_LIMIT')
        for item in infos:
            name = item.filename[:-1] if item.is_dir() else item.filename
            need(safe_member(name) and not item.flag_bits & 1 and item.file_size <= 128 * 1024**2,
                 'UNSAFE_ZIP_MEMBER')
            need(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'UNSAFE_ZIP_TYPE')

    def read(self, name):
        need(safe_member(name), 'UNSAFE_ZIP_READ')
        return self.z.read(name)

    def json(self, name):
        return loads(self.read(name))

    def close(self):
        self.z.close()


def prepare_inputs(root, training_path, quality_path, sdk_path, roster):
    """Authenticate existing archives and regenerate the roster; no audio decode."""
    prepare = load_pinned_module(root, RESEARCH + '/prepare_natural32.py', 'natural32_prepare_pinned')
    regenerated = prepare.canonical(prepare.build_plan(training_path, quality_path))
    need(regenerated == regular(Path(root) / RESEARCH / 'natural32-roster.json'), 'REGENERATED_ROSTER_BYTES')
    need(loads(regenerated) == roster, 'ROSTER_OBJECT_IDENTITY')
    archives = {}
    try:
        for name, path in [('training', training_path), ('quality', quality_path), ('sdk', sdk_path)]:
            archives[name] = Archive(path, SOURCES[name])
        training, quality, sdk = (archives[k] for k in ('training', 'quality', 'sdk'))
        head = quality.read('models/coverage-head.bin')
        need(identity(head) == HEAD and head[:8] == b'EHKWS01\x00', 'FROZEN_COVERAGE_HEAD')
        values = struct.unpack('<49475f', head[8:])
        need(all(math.isfinite(v) for v in values) and all(v > 0 for v in values[96:192]), 'FINITE_HEAD_AND_SCALE')
        qr = quality.json('evidence/RESULT.json')
        need(qr.get('status') == 'COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH' and
             qr.get('head_sha') == SOURCES['quality']['head_sha'] and
             str(qr.get('run_id')) == str(SOURCES['quality']['run_id']) and
             str(qr.get('run_attempt')) == '1' and qr.get('research_retention', {}).get('passed') is True and
             all(qr.get(k) is False for k in FALSE_FLAGS), 'QUALITY_RETAINED_RESEARCH_RESULT')
        manifest_raw, result_raw = sdk.read('MANIFEST.json'), sdk.read('evidence/RESULT.json')
        need(sha(manifest_raw) == SDK_MANIFEST_SHA and sha(result_raw) == SDK_RESULT_SHA, 'SDK_ORIGINAL_EVIDENCE')
        manifest, sr = loads(manifest_raw), loads(result_raw)
        need(set(sdk.z.namelist()) == set(manifest['files']) | {'MANIFEST.json'}, 'SDK_EXACT_INVENTORY')
        for name, pin in manifest['files'].items():
            need(identity(sdk.read(name)) == pin, 'SDK_MEMBER_PIN:' + name)
        need(sr.get('passed') is True and sr.get('installed_sdk_software_verified') is True and
             sr.get('github_sha') == SOURCES['sdk']['head_sha'] and
             str(sr.get('run_id')) == str(SOURCES['sdk']['run_id']) and
             all(sr.get(k) is False for k in FALSE_FLAGS), 'SDK_SUCCESS_BOUNDARY')
        need(identity(sdk.read('sdk-native/lib/libekws_research_stream.a')) == LIBRARY and
             sha(sdk.read('sdk-native/include/ekws_research_stream.h')) == PUBLIC_HEADER_SHA and
             sha(sdk.read('source/research/english_final_embedding/english_causal_head.h')) == PRIVATE_HEADER_SHA,
             'SDK_LIBRARY_AND_HEADERS')
        saved = {name: training.read(name) for name in ('LIBRISPEECH_SOURCES.json', 'STREAM_ROSTER.json')}
        need(all(sha(raw) == roster['metadata_inputs'][name] for name, raw in saved.items()), 'TRAINING_METADATA_PINS')
        return {'archives': archives, 'head': head, 'sdk_manifest': manifest, 'saved_metadata': saved,
                'summary': {'status': 'STATIC_ARCHIVE_AND_ROSTER_VERIFIED', 'sources': SOURCES,
                    'roster': identity(regenerated), 'head': HEAD, 'native_library': LIBRARY, 'model_calls': 0}}
    except BaseException:
        for archive in archives.values():
            archive.close()
        raise


def flac_streaminfo(raw):
    need(len(raw) >= 42 and raw[:4] == b'fLaC' and raw[4] & 0x7f == 0 and
         int.from_bytes(raw[5:8], 'big') == 34, 'FLAC_FIRST_STREAMINFO')
    packed = int.from_bytes(raw[18:26], 'big')
    result = {'sample_rate': packed >> 44, 'channels': ((packed >> 41) & 7) + 1,
              'bits_per_sample': ((packed >> 36) & 31) + 1, 'frames': packed & ((1 << 36) - 1)}
    need((result['sample_rate'], result['channels'], result['bits_per_sample']) == (16000, 1, 16)
         and 0 < result['frames'] <= 240000, 'FLAC_LOSSLESS_GEOMETRY')
    return result


def select_corpus(archive_path, roster, output, expected_pin=CORPUS):
    """Read selected regular members only; never use extract/extractall."""
    need(file_identity(archive_path) == {k: expected_pin[k] for k in ('bytes', 'sha256')}, 'CORPUS_EXACT_TAR')
    by_member = {r['source_member']: r for r in roster['rows']}
    transcripts = {}
    for name in by_member:
        match = MEMBER.fullmatch(name)
        need(match is not None, 'CANONICAL_LIBRISPEECH_MEMBER')
        speaker, chapter, _ = match.groups()
        transcripts[f'LibriSpeech/dev-clean/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt'] = None
    selected, seen, total_bytes, count = {}, set(), 0, 0
    with tarfile.open(archive_path, mode='r|gz') as archive:
        for member in archive:
            count += 1
            name = member.name[:-1] if member.isdir() and member.name.endswith('/') else member.name
            need(count <= 20000 and name not in seen and safe_member(name), 'SAFE_UNIQUE_TAR_MEMBERS')
            seen.add(name)
            need(member.isfile() or member.isdir(), 'TAR_REGULAR_FILES_OR_DIRECTORIES_ONLY')
            need(0 <= member.size <= 32 * 1024**2, 'TAR_MEMBER_SIZE_LIMIT')
            total_bytes += member.size
            need(total_bytes <= 1024**3, 'TAR_EXPANSION_LIMIT')
            if name not in by_member and name not in transcripts:
                continue
            need(member.isfile(), 'SELECTED_TAR_REGULAR_FILE')
            stream = archive.extractfile(member)
            need(stream is not None, 'SELECTED_MEMBER_READ')
            raw = stream.read(member.size + 1)
            need(len(raw) == member.size, 'COMPLETE_SELECTED_MEMBER')
            if name in by_member:
                row = by_member[name]
                need(sha(raw) == row['flac_sha256'], 'SOURCE_FLAC_SHA256')
                info = flac_streaminfo(raw)
                need(info['frames'] == row['frames'], 'SOURCE_FLAC_SAMPLE_COUNT')
                target = Path(output) / 'selectedinputs/flac' / (row['recording_id'] + '.flac')
                put(target, raw)
                selected[name] = {'path': target, 'identity': identity(raw), 'streaminfo': info}
            else:
                transcripts[name] = raw
    need(set(selected) == set(by_member) and all(v is not None for v in transcripts.values()), 'ALL_FROZEN_MEMBERS_FOUND')
    texts = {}
    for name, raw in transcripts.items():
        need(len(raw) <= 1024**2, 'TRANSCRIPT_SIZE')
        chapter = {}
        for line in raw.decode('utf-8').splitlines():
            parts = line.strip().split(maxsplit=1)
            need(len(parts) == 2 and parts[0] not in chapter, 'UNIQUE_TRANSCRIPT_UTTERANCE')
            chapter[parts[0]] = parts[1]
        put(Path(output) / 'selectedinputs/transcript-sources' / PurePosixPath(name).name, raw)
        for member, row in by_member.items():
            if str(PurePosixPath(member).parent) != str(PurePosixPath(name).parent):
                continue
            utterance = PurePosixPath(member).stem
            need(utterance in chapter, 'SELECTED_TRANSCRIPT_PRESENT')
            text = chapter[utterance].encode('utf-8')
            need(sha(text) == row['transcript_sha256'], 'SOURCE_TRANSCRIPT_SHA256')
            put(Path(output) / 'selectedinputs/transcripts' / (row['recording_id'] + '.txt'), text)
            texts[row['recording_id']] = identity(text)
    need(set(texts) == {r['recording_id'] for r in roster['rows']}, 'ALL_TRANSCRIPT_IDENTITIES')
    return selected, {'status': 'EXACT_SELECTED_FLAC_AND_TRANSCRIPTS', 'corpus': expected_pin,
        'selected_members': {name: {'flac': x['identity'], 'streaminfo': x['streaminfo']} for name, x in selected.items()},
        'selected_transcripts': texts, 'archive_members_inspected': count,
        'archive_paths_extracted': 0, 'model_calls': 0, 'decodes_performed': 0}


def replay_events(scores, frames):
    ready = [row for row in scores if row['ready']]
    clocks = list(range(35200, frames + 1, 1280))
    need([row['sample'] for row in ready] == clocks, 'EVENT_REPLAY_READY_CLOCKS')
    candidate, confirmations, blocked_until, events = 0, 0, 0, []
    for row in ready:
        probs = row['probabilities']
        need(len(probs) == 3 and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in probs)
             and abs(sum(probs) - 1) <= 1e-5, 'EVENT_REPLAY_PROBABILITIES')
        selected = 0
        for index in (1, 2):
            if probs[index] > probs[selected]:
                selected = index
        emitted = 0
        if selected == 0 or probs[selected] < CONFIG['threshold_float32'] or row['sample'] < blocked_until:
            candidate, confirmations = 0, 0
        else:
            confirmations = confirmations + 1 if candidate == selected else 1
            candidate = selected
            if confirmations >= 3:
                emitted = selected
                events.append([emitted, row['sample']])
                blocked_until = row['sample'] + 13 * 1280
                candidate, confirmations = 0, 0
        need(row['keyword'] == emitted, 'EVENT_REPLAY_EMITTED_KEYWORD')
    return events


def validate_trace_metadata(meta, parsed, row):
    count, ready, samples = row['expected_embedding_rows'], row['expected_ready_rows'], row['frames']
    exact = {'schema': 'english-stream-numeric-trace-v1', 'format_magic': 'EKWSTRC1',
        'format_version': 1, 'abi': 1, 'sample_rate_hz': 16000, 'encoding': 's16le_mono',
        'chunk_samples': 320, 'trace_bytes': parsed['trace_identity']['bytes'],
        'trace_records': 2 * count + len(parsed['events']) + 1,
        'trace_completed': True, 'end_record_written': True, 'input_samples': samples,
        'input_bytes': 2 * samples, 'embedding_records': count, 'real_head_calls': count,
        'score_records': count, 'ready_outputs': ready, 'event_records': len(parsed['events']),
        'feed_calls': (samples + 319) // 320, 'input_repetitions': 1, 'resets': 0,
        'eof_padding_samples': 0, 'score_observer_enabled': True, 'original_head_link_wrap': True,
        'performance_measured': False}
    need(all(type(meta.get(k)) is type(v) and meta.get(k) == v for k, v in exact.items()), 'COMPLETE_TRACE_PROCESS_METADATA')
    need(replay_events(parsed['scores'], samples) == parsed['events'], 'SAVED_SCORE_EVENT_POLICY')
    return {'recording_id': row['recording_id'], 'status': 'COMPLETE_NATIVE_TRACE_AND_EVENT_POLICY_VERIFIED',
        'input_pcm_sha256': row['pcm_sha256'], 'frames': samples, 'embedding_rows': count,
        'ready_rows': ready, 'events': parsed['events'], 'trace': parsed['trace_identity'],
        'saved_score_event_replay_exact': True, 'numeric_reference_comparison': 'NOT_RUN',
        'label_status': 'UNKNOWN_REVIEW_REQUIRED', **dict.fromkeys(FALSE_FLAGS, False)}


def execution_identity(plan, env=None, event=None, head=None, parent=None):
    env = os.environ if env is None else env
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPO and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1', 'PUBLIC_MAIN_FIRST_PUSH_ONLY')
    commit, run_id = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None and run_id.isdigit(), 'ACTUAL_RUN_AND_HEAD')
    if event is None:
        event = loads(regular(env['GITHUB_EVENT_PATH'], 8 * 1024**2))
    need(event.get('before') == plan['main_before'] == BEFORE and event.get('after') == commit and
         event.get('ref') == 'refs/heads/main' and event.get('repository', {}).get('private') is False and
         event.get('repository', {}).get('full_name') == REPO and event.get('deleted') is False and
         event.get('forced') is False, 'EXACT_PUBLIC_SUCCESSOR_EVENT')
    if head is None:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    if parent is None:
        parent = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD^1'], text=True, timeout=10).strip()
    need(head == commit and parent == BEFORE, 'WORKTREE_EXACT_SUCCESSOR_OF_FROZEN_BASE')
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
    """All commands inherit the single supervised child's process group."""
    def __init__(self, output, deadline):
        self.output, self.deadline = Path(output), deadline
        self.counts = {'build': 0, 'decode': 0, 'model': 0}

    def command(self, tag, argv, timeout=120, auth=False, maximum_stdout=16 * 1024**2,
                kind=None, stdout_path=None):
        need(re.fullmatch('[a-z0-9_.-]+', tag) is not None, 'COMMAND_TAG')
        need(time.monotonic() < self.deadline, 'GLOBAL_DEADLINE')
        if kind is not None:
            need(kind in self.counts, 'COMMAND_KIND')
            need(self.counts[kind] < {'build': 1, 'decode': 32, 'model': 32}[kind], 'COMMAND_BUDGET:' + kind)
            self.counts[kind] += 1
        argv = [str(x) for x in argv]
        evidence, started = self.output / 'evidence', time.monotonic()
        stdout = evidence / (tag + '.stdout') if stdout_path is None else Path(stdout_path)
        need(stdout.is_relative_to(self.output) and not stdout.exists() and
             not any(p.is_symlink() for p in (stdout, *stdout.parents)), 'NEW_CONFINED_COMMAND_OUTPUT')
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stderr = evidence / (tag + '.stderr')
        once(evidence / (tag + '.started.json'), {'argv': argv, 'attempt': 1, 'retry_count': 0,
             'kind': kind, 'monotonic_started': started, 'global_deadline': self.deadline,
             'new_session': False, 'cpu_threads': 1, 'stdout': stdout.relative_to(self.output).as_posix()})
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
                    need(stdout.stat().st_size <= maximum_stdout and stderr.stat().st_size <= 4 * 1024**2,
                         'COMMAND_OUTPUT_LIMIT')
                    time.sleep(0.05)
                need(stdout.stat().st_size <= maximum_stdout and stderr.stat().st_size <= 4 * 1024**2,
                     'COMMAND_OUTPUT_LIMIT')
                need(process.returncode == 0, 'COMMAND_NONZERO_EXIT')
            except BaseException as exc:
                error = type(exc).__name__ + ':' + str(exc)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                once(evidence / (tag + '.exit.json'), {'argv': argv, 'returncode': None if process is None else process.returncode,
                    'error': error, 'timed_out': timed_out, 'wall_seconds': time.monotonic() - started,
                    'attempt': 1, 'retry_count': 0, 'kind': kind,
                    'descendants_supervised_by_parent_process_group': True})
        need(error is None, error or 'COMMAND_FAILED')
        return stdout

    @contextlib.contextmanager
    def stage(self, name):
        need(time.monotonic() < self.deadline and re.fullmatch('[a-z0-9-]+', name), 'STAGE_DEADLINE_AND_NAME')
        start = time.monotonic()
        once(self.output / 'evidence' / ('stage-' + name + '.started.json'), {'stage': name, 'monotonic_started': start})
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__ + ':' + str(exc)
            raise
        finally:
            once(self.output / 'evidence' / ('stage-' + name + '.exit.json'),
                 {'stage': name, 'error': error, 'wall_seconds': time.monotonic() - start, 'retry_count': 0})

    def download(self, name):
        pin = SOURCES[name]
        run = loads(regular(self.command(name + '-run', ['gh', 'api', 'repos/' + REPO + '/actions/runs/' + str(pin['run_id'])], auth=True)))
        need(run.get('id') == pin['run_id'] and run.get('head_sha') == pin['head_sha'] and
             run.get('conclusion') == 'success' and run.get('status') == 'completed' and run.get('run_attempt') == 1 and
             run.get('repository', {}).get('private') is False and run.get('repository', {}).get('full_name') == REPO,
             'PINNED_PUBLIC_SOURCE_RUN')
        artifact = loads(regular(self.command(name + '-artifact', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id'])], auth=True)))
        need(artifact.get('id') == pin['artifact_id'] and artifact.get('name') == pin['name'] and
             artifact.get('expired') is False and artifact.get('size_in_bytes') == pin['bytes'] and
             artifact.get('digest') == 'sha256:' + pin['sha256'] and
             artifact.get('workflow_run', {}).get('id') == pin['run_id'] and
             artifact.get('workflow_run', {}).get('head_sha') == pin['head_sha'], 'PINNED_SOURCE_ARTIFACT')
        target = self.output / 'downloads' / (name + '.zip')
        self.command(name + '-zip', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id']) + '/zip'],
                     auth=True, timeout=240, maximum_stdout=pin['bytes'], stdout_path=target)
        need(file_identity(target) == {k: pin[k] for k in ('bytes', 'sha256')}, 'DOWNLOADED_EXACT_ZIP')
        once(self.output / 'evidence' / (name + '-download.json'), {'source': pin, 'actual': file_identity(target), 'retry_count': 0})
        return target

    def download_corpus(self):
        target = self.output / 'downloads/dev-clean.tar.gz'
        self.command('corpus-download', ['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
            '--connect-timeout', '20', '--max-time', '300', '--retry', '0', '--silent', '--show-error', CORPUS['url']],
            timeout=305, maximum_stdout=CORPUS['bytes'], stdout_path=target)
        need(file_identity(target) == {k: CORPUS[k] for k in ('bytes', 'sha256')}, 'DOWNLOADED_CORPUS_PIN')
        once(self.output / 'evidence/CORPUS_DOWNLOAD.json', {'source': CORPUS, 'actual': file_identity(target), 'retry_count': 0})
        return target


def decode_argv(source):
    return ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(source),
            '-map', '0:a:0', '-vn', '-sn', '-dn', '-c:a', 'pcm_s16le', '-threads', '1', '-f', 's16le', 'pipe:1']


def validate_elf(raw):
    need(len(raw) >= 64 and raw[:4] == b'\x7fELF' and raw[4:6] == b'\x02\x01' and
         struct.unpack_from('<H', raw, 18)[0] == 62, 'NATIVE_X86_64_CONSUMER')


def result_boundary(binding):
    return {'schema': SCHEMA + '/result', 'status': 'FAILED_NO_RETRY', 'passed': False,
        **binding, 'candidate_head': HEAD, 'native_library': LIBRARY,
        'configuration_sha256': sha(canonical(CONFIG)), 'new_independent_speaker_count': 0,
        'verified_negative_seconds': 0, 'verified_negative_count': 0, 'optimizer_updates': 0,
        'new_onnx_calls': 0, 'python_head_forwards': 0, 'new_tts_asr_phone_calls': 0,
        'new_mixes': 0, 'retry_count': 0, 'independent_far': 'NOT_MEASURED',
        'independent_frr': 'NOT_MEASURED', 'word_end_latency': 'NOT_MEASURED', 'ssc305': 'NOT_RUN',
        'quality_improvement_claim': False, 'performance_measured': False,
        'kws_model_delivered': False, **dict.fromkeys(FALSE_FLAGS, False)}


def child(args, plan, roster):
    output = args.output.absolute()
    start = loads(regular(output / 'STARTED.json'))
    need(os.getppid() == start['parent_pid'] and os.environ.get('KWS_NATURAL32_PARENT_PID') == str(start['parent_pid']),
         'SUPERVISED_CHILD_ONLY')
    binding = execution_identity(plan)
    need(start['binding'] == binding and start['worker_sha256'] == plan['worker_sha256'], 'CHILD_SOURCE_BINDING')
    deadline = start['monotonic_deadline']
    need(deadline - start['monotonic_started'] == 1200 and time.monotonic() < deadline, 'CHILD_GLOBAL_DEADLINE')
    once(output / 'evidence/WORKER_STARTED.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'binding': binding, 'retry_count': 0})
    runner, reports, prepared = Runner(output, deadline), [], None
    result = result_boundary(binding)
    try:
        with runner.stage('source-binding'):
            current = loads(regular(runner.command('current-run', ['gh', 'api', 'repos/' + REPO + '/actions/runs/' + binding['run_id']], auth=True)))
            need(str(current.get('id')) == binding['run_id'] and current.get('run_attempt') == 1 and
                 current.get('event') == 'push' and current.get('head_branch') == 'main' and
                 current.get('head_sha') == binding['head_sha'] and current.get('path') == WORKFLOW_PATH and
                 current.get('repository', {}).get('private') is False and
                 current.get('repository', {}).get('full_name') == REPO, 'CURRENT_PUBLIC_SOURCE_RUN')
            for path in SOURCE_PINS:
                put(output / 'source' / path, regular(ROOT / path))
            put(output / 'source' / WORKER_PATH, regular(__file__))
            put(output / 'source' / PLAN_PATH, regular(ROOT / PLAN_PATH))
            put(output / 'source' / WORKFLOW_PATH, regular(ROOT / WORKFLOW_PATH))
            once(output / 'source/CONFIG.json', CONFIG)
        with runner.stage('authenticate-artifacts'):
            paths = {name: runner.download(name) for name in ('training', 'quality', 'sdk')}
            prepared = prepare_inputs(ROOT, paths['training'], paths['quality'], paths['sdk'], roster)
            once(output / 'evidence/INPUT_PREFLIGHT.json', prepared['summary'])
            put(output / 'models/coverage-head.bin', prepared['head'])
            for name, raw in prepared['saved_metadata'].items():
                put(output / 'source-metadata' / name, raw)
            sdk = prepared['archives']['sdk']
            for name in prepared['sdk_manifest']['files']:
                if name.startswith('sdk-native/') or name == 'source/research/english_final_embedding/english_causal_head.h':
                    put(output / 'installed-base' / name, sdk.read(name))
            put(output / 'installed-base/ORIGINAL_MANIFEST.json', sdk.read('MANIFEST.json'))
            put(output / 'source-metadata/ORIGINAL_SDK_RESULT.json', sdk.read('evidence/RESULT.json'))
            put(output / 'source-metadata/ORIGINAL_QUALITY_RESULT.json', prepared['archives']['quality'].read('evidence/RESULT.json'))
        with runner.stage('select-original-corpus'):
            corpus_path = runner.download_corpus()
            selected, receipt = select_corpus(corpus_path, roster, output)
            once(output / 'evidence/SELECTED_ORIGINALS.json', receipt)
        with runner.stage('decode-originals'):
            runner.command('ffmpeg-version', ['ffmpeg', '-version'], timeout=15)
            pcm_by_id = {}
            for index, row in enumerate(roster['rows']):
                rid = row['recording_id']
                path = output / 'selectedinputs/pcm' / (rid + '.pcm')
                runner.command('decode-' + str(index), decode_argv(selected[row['source_member']]['path']),
                    timeout=30, kind='decode', maximum_stdout=row['frames'] * 2, stdout_path=path)
                raw = regular(path, 480000)
                need(identity(raw) == {'bytes': row['frames'] * 2, 'sha256': row['pcm_sha256']}, 'LOSSLESS_ORIGINAL_PCM')
                pcm_by_id[rid] = raw
                once(output / 'evidence' / ('decode-' + str(index) + '.identity.json'),
                     {'recording_id': rid, 'source_member': row['source_member'], 'pcm': identity(raw),
                      'frames': row['frames'], 'resampling': False, 'gain_changes': 0, 'padding_samples': 0})
        with runner.stage('build-native-consumer'):
            runner.command('native-compiler-version', ['gcc', '--version'], timeout=15)
            install = output / 'installed-base/sdk-native'
            library = install / 'lib/libekws_research_stream.a'
            private = output / 'installed-base/source/research/english_final_embedding'
            source = output / 'source' / TRACE_DIR
            symbols = regular(runner.command('native-symbols', ['nm', '--print-file-name', library]))
            need(re.search(rb'ekws_research_stream\.c\.o:.*\bU ehead_step\b', symbols) is not None and
                 re.search(rb'english_causal_head\.c\.o:.*\bT ehead_step\b', symbols) is not None, 'INSTALLED_HEAD_LINK_HOOK')
            program = output / 'bin/trace-native'
            program.parent.mkdir(exist_ok=True)
            runner.command('build-native', ['gcc', *FLAGS, '-I' + str(install / 'include'), '-I' + str(private),
                '-I' + str(source), source / 'ekws_stream_trace.c', library, '-Wl,--wrap=ehead_step',
                '-lm', '-lrt', '-o', program], timeout=180, kind='build')
            executable = regular(program)
            validate_elf(executable)
            once(output / 'evidence/NATIVE_CONSUMER.json', {'identity': identity(executable), 'installed_library': LIBRARY,
                 'consumer_builds': runner.counts['build'], 'physical_target': 'NOT_RUN'})
        with runner.stage('scan-natural32'):
            parser = load_pinned_module(ROOT, FROZEN_PARSER, 'natural32_frozen_trace_parser')
            ready_by_id, trace_by_id = {}, {}
            head = output / 'models/coverage-head.bin'
            for index, row in enumerate(roster['rows']):
                rid = row['recording_id']
                pcm = output / 'selectedinputs/pcm' / (rid + '.pcm')
                trace = output / 'traces' / (rid + '.trace')
                trace.parent.mkdir(exist_ok=True)
                checks = {path: file_identity(path) for path in (pcm, head, library, program)}
                need(checks[pcm] == {'bytes': row['frames'] * 2, 'sha256': row['pcm_sha256']} and
                     checks[head] == HEAD and checks[library] == LIBRARY and checks[program] == identity(executable), 'PRE_SCAN_IDENTITIES')
                meta = loads(regular(runner.command('scan-' + str(index), [program, head, pcm, trace],
                                                   timeout=120, kind='model')))
                need(all(file_identity(path) == pin for path, pin in checks.items()), 'POST_SCAN_IDENTITIES')
                case = {'samples': row['frames'], 'records': row['expected_embedding_rows'], 'ready': row['expected_ready_rows']}
                parsed = parser.parse_trace(regular(trace, 1024**2), case)
                report = validate_trace_metadata(meta, parsed, row)
                ready_by_id[rid] = [x for x in parsed['scores'] if x['ready']]
                trace_by_id[rid] = parsed['trace_identity']['sha256']
                once(output / 'scores' / (rid + '.json'), {'recording_id': rid, 'source': row,
                     'model': HEAD, 'configuration_sha256': sha(canonical(CONFIG)),
                     'trace': parsed['trace_identity'], 'scores': parsed['scores'], 'events': parsed['events']})
                once(output / 'evidence' / ('scan-' + str(index) + '.verified.json'), report)
                reports.append(report)
        with runner.stage('unreviewed-candidate-ledger'):
            miner = load_pinned_module(ROOT, RESEARCH + '/mine_saved_scores.py', 'natural32_saved_score_miner')
            ledger = miner.build_ledger(roster, pcm_by_id, ready_by_id, model_sha256=HEAD['sha256'],
                config_sha256=sha(canonical(CONFIG)), source_evidence={**binding, 'trace_sha256_by_id': trace_by_id,
                    'model_processes': runner.counts['model'], 'validated_trace_count': len(reports),
                    'source_roster_sha256': SOURCE_PINS[RESEARCH + '/natural32-roster.json']['sha256'],
                    'native_consumer': identity(executable), 'installed_library': LIBRARY, 'retry_count': 0})
            put(output / 'candidates/LEDGER.json', miner.canonical(ledger))
            clips = []
            for candidate in ledger['candidates']:
                rid, start_sample, end_sample = candidate['recording_id'], candidate['start_sample'], candidate['end_sample_exclusive']
                raw = pcm_by_id[rid][2 * start_sample:2 * end_sample]
                need(len(raw) == candidate['window_samples'] * 2 and sha(raw) == candidate['window_pcm_sha256'], 'CANDIDATE_UNMODIFIED_PCM')
                cid = candidate['candidate_id']
                need(re.fullmatch(r'hardneg-review-[0-9a-f]{24}', cid) is not None, 'CANDIDATE_OUTPUT_ID')
                put(output / 'candidates/pcm' / (cid + '.pcm'), raw)
                wav = io.BytesIO()
                with wave.open(wav, 'wb') as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    writer.writeframes(raw)
                put(output / 'candidates/wav' / (cid + '.wav'), wav.getvalue())
                clips.append({'candidate_id': cid, 'pcm': identity(raw), 'wav': identity(wav.getvalue()),
                              'review': 'UNREVIEWED', **dict.fromkeys(FALSE_FLAGS, False)})
            once(output / 'candidates/CLIP_FILES.json', {'clips': clips, 'count': len(clips), 'training_tsv_emitted': False})
        need(runner.counts == {'build': 1, 'decode': 32, 'model': 32} and len(reports) == 32, 'EXACT_ONCE_EXECUTION_BUDGET')
        result.update(status='COMPLETED_NATURAL32_RESEARCH_SCAN', passed=True,
            observed_originals=32, observed_source_samples=3270080, observed_source_seconds=204.38,
            roster_speakers=24, embedding_rows=sum(r['embedding_rows'] for r in reports),
            ready_rows=sum(r['ready_rows'] for r in reports), actual_event_count=sum(len(r['events']) for r in reports),
            candidate_count=len(ledger['candidates']), candidate_ledger_status=ledger['status'],
            ledger_identity=identity(miner.canonical(ledger)), saved_score_event_replay_exact=True,
            source_pool='PREVIOUSLY_EXPOSED_TRAINING_SPEAKERS_NEW_CURRENT_HEAD_ORIGINAL_UTTERANCES',
            natural_continuous_far_denominator=False)
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ':' + str(exc)
        put(output / 'evidence/FAILURE_TRACEBACK.txt', traceback.format_exc().encode())
    finally:
        if prepared is not None:
            for archive in prepared['archives'].values():
                archive.close()
        result.update(consumer_build_attempts=runner.counts['build'], decoder_process_attempts=runner.counts['decode'],
                      model_process_attempts=runner.counts['model'], completed_validated_traces=len(reports), reports=reports)
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


def close_interrupted_receipts(output, supervisor):
    """A killed process has an unknown exit code, never an invented success."""
    errors = []
    for path in sorted((Path(output) / 'evidence').glob('*.started.json')):
        target = path.with_name(path.name.removesuffix('.started.json') + '.exit.json')
        raw = regular(path)
        try:
            started = loads(raw)
            need(type(started) is dict, 'STARTED_RECEIPT_OBJECT')
            damaged = None
        except (ValueError, TypeError) as exc:
            started = {}
            damaged = {'identity': identity(raw), 'parse_error': type(exc).__name__ + ':' + str(exc)}
            errors.append({'path': path.name, **damaged})
        if target.exists():
            continue
        once(target, {'schema': SCHEMA + '/interrupted-step', 'stage': started.get('stage'),
            'kind': started.get('kind'), 'argv': started.get('argv'), 'returncode': None,
            'error': 'INTERRUPTED_BEFORE_STEP_EXIT_RECEIPT', 'completed': False,
            'damaged_started_receipt': damaged,
            'supervisor_cleanup': supervisor['cleanup'], 'timed_out': supervisor['timed_out'],
            'retry_count': 0, 'no_step_retry': True})
    return errors


def supervise(args, plan):
    binding = execution_identity(plan)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_SUPERVISED_OUTPUT_ONLY')
    output.mkdir(mode=0o700)
    (output / 'evidence').mkdir()
    started, process, timed_out, error, cleanup = time.monotonic(), None, False, None, {}
    deadline = started + 1200
    once(output / 'STARTED.json', {'schema': SCHEMA + '/started', 'binding': binding,
        'parent_pid': os.getpid(), 'worker_sha256': plan['worker_sha256'], 'monotonic_started': started,
        'monotonic_deadline': deadline, 'maximum_seconds': 1200, 'retry_count': 0, 'network_calls_at_start': 0})
    env = dict(os.environ, KWS_NATURAL32_PARENT_PID=str(os.getpid()))
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
                    raise TimeoutError('SUPERVISOR_1200_SECONDS')
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
                            model_process_attempts=None if receipt['partial_receipt_errors'] else
                            sum(loads(regular(p)).get('kind') == 'model' for p in (output / 'evidence').glob('*.started.json')),
                            completed_validated_trace_receipts=len(list((output / 'evidence').glob('scan-*.verified.json'))))
            result_path = output / 'evidence/RESULT.json'
            if result_path.exists():
                fallback['preserved_damaged_worker_result'] = file_identity(result_path)
                result_path = output / 'evidence/RESULT_SUPERVISOR_FAILURE.json'
            once(result_path, fallback)
        files = {p.relative_to(output).as_posix(): file_identity(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and 'downloads' not in p.relative_to(output).parts}
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
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    plan = loads(regular(ROOT / PLAN_PATH, 1024**2))
    roster = source_preflight(ROOT, plan)
    if args.execute or args._worker:
        need(args.output is not None, 'EXECUTION_OUTPUT_REQUIRED')
        return child(args, plan, roster) if args._worker else supervise(args, plan)
    need(args.output is None, 'SOURCE_ONLY_HAS_NO_OUTPUT')
    print(canonical({'status': 'SOURCE_CONTRACT_ONLY', 'execution_enabled': False, 'model_calls': 0,
        'decoder_calls': 0, 'network_calls': 0, 'plan_sha256': sha(canonical(plan)),
        'worker_sha256': plan['worker_sha256'], 'fixed_originals': len(roster['rows']),
        'maximum_model_processes': 32, 'maximum_seconds': 1200}).decode(), end='')
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
