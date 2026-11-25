"""Offline, fail-only readback of the real B24R1 SOURCE execution.

This is a saved-evidence reader, never a source opening or a quality verdict.
All fourteen original archives and final APIs must be independently collected
and pinned before a report can be produced. The captured original SOURCE and
native parsers remain authoritative, including the complete nested B24 STOP.
"""
from __future__ import annotations

import argparse
import base64
import builtins
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import urllib.request
import wave
import zipfile

REPO = 'jiying2007/kws-data'
RUN = 45534020313
HEAD = 'b65bfe4974432b3d922738d7931d1c2d6d6f367c'
PARENT = 'ea9f9e3b467231efe2746cbe64f867481177d09e'
BRANCH = 'research/en-v6-b24r1-primary-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b24r1-primary-sources-once.yml'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60')
FAILED_ID = 'b24r1-primary-84-mero-fast'
ASSESSMENT = 'e783c982f66ed35f903680772842ca8068aad6fb18221e22b9512cd819efa075'
TRAIN_ASSESSMENT = '3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad'
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
OLD_READER_SHA = '2d5564039689d5fb5e31bf5cde51d49dba5bf9d0697c295cd98c96f7706585c6'
OLD_REPORT_SHA = '4063a57f1d20ffe9f534248067e1adcbe9c25ab65cb5bd3682b5db44d3ba39d8'
LOCK_SCHEMA = 'english-v6-b24r1-original-source-stop-lock-v1'
REPORT_SCHEMA = 'english-v6-b24r1-authenticated-original-source-stop-v1'
NAMES = dict(source='english-v6-b24r1-primary-source-' + str(RUN),
    source_producer='english-v6-b24r1-primary-source-producer-' + str(RUN),
    transport='english-v6-b24r1-primary-transport-' + str(RUN),
    source_consumed='english-v6-b24r1-primary-source-consumed',
    **{'native_' + p: 'b24r1-primary-dev-observer-' + p for p in PROVIDERS},
    **{'consumed_' + p: 'english-v6-b24r1-primary-observer-' + p + '-consumed' for p in PROVIDERS})
API_ROLES = ('run', 'artifacts', 'commit', 'tree', 'jobs')
COUNTERS = ('input_started', 'forward_attempted', 'native_returned', 'raw_payload_saved',
    'payload_valid', 'capture_saved', 'failed', 'not_started', 'entered_without_return', 'entered_without_raw')
FALSE_PERMISSIONS = ('training_allowed', 'primary_allowed', 'replication_allowed',
    'engineering_allowed', 'promotion_allowed', 'product_released', 'scientific_execution_authorized',
    'source_retry_allowed', 'source_continuation_allowed', 'old_source_retry_allowed',
    'old_source_continuation_allowed', 'sealed_reserve_opened', 'automatic_R2_allowed',
    'new_source_registration_performed', 'model_quality_verdict_available')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate original JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON ' + value)))


def pin(raw):
    return dict(bytes=len(raw), sha256=sha(raw))


def valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def archive_limit(role):
    return 512_000_000 if role == 'source' else 128_000_000


def valid_pin(value, maximum):
    return (type(value) is dict and set(value) == {'bytes', 'sha256'}
        and type(value['bytes']) is int and 0 < value['bytes'] <= maximum and valid_hash(value['sha256']))


def validate_counts(counts):
    """Known R1 empty-return boundary; final other-provider counts remain required."""
    need(type(counts) is dict and set(counts) == set(PROVIDERS), 'all five final provider counts required')
    for provider, row in counts.items():
        need(type(row) is dict and set(row) == set(COUNTERS) | {'status'}
             and all(type(row[k]) is int and row[k] >= 0 for k in COUNTERS), 'complete integer provider counters')
        expected = dict(input_started=120, forward_attempted=120, native_returned=120,
            raw_payload_saved=120, payload_valid=120, capture_saved=120, failed=0,
            not_started=0, entered_without_return=0, entered_without_raw=0)
        if provider == 'parakeet':
            expected.update(input_started=9, forward_attempted=9, native_returned=9,
                raw_payload_saved=9, payload_valid=8, capture_saved=8, failed=1, not_started=111)
        need({k: row[k] for k in COUNTERS} == expected
             and row['status'] == ('STOP_OBSERVER_FAILURE' if provider == 'parakeet' else 'COMPLETE'),
             'only the exact saved ninth empty return; unproven or additional failures stay unverified')
    return counts


def validate_lock(lock):
    need(type(lock) is dict and set(lock) == {'schema', 'repo', 'run_id', 'run_head', 'prepared_parent',
        'branch', 'workflow', 'artifacts', 'api', 'logs', 'provider_counts',
        'source_producer_manifest_sha256', 'source_manifest_sha256', 'history_sha256',
        'source_assessment_sha256', 'candidate_pt_sha256', 'train_result_sha256'}, 'complete saved R1 STOP lock')
    need(lock['schema'] == LOCK_SCHEMA and lock['repo'] == REPO and type(lock['run_id']) is int
         and lock['run_id'] == RUN and lock['run_head'] == HEAD and lock['prepared_parent'] == PARENT
         and lock['branch'] == BRANCH and lock['workflow'] == WORKFLOW
         and lock['source_assessment_sha256'] == ASSESSMENT and lock['candidate_pt_sha256'] == CANDIDATE_PT,
         'only the actual separate R1 SOURCE run and original fixed B24 candidate')
    need(type(lock['artifacts']) is dict and set(lock['artifacts']) == set(NAMES), 'all fourteen originals required')
    ids = set()
    for role, value in lock['artifacts'].items():
        need(type(value) is dict and set(value) == {'id', 'name', 'bytes', 'sha256'}
             and type(value['id']) is int and value['id'] > 0 and value['name'] == NAMES[role]
             and valid_pin({k: value[k] for k in ('bytes', 'sha256')}, archive_limit(role)),
             'complete original artifact pin ' + role)
        ids.add(value['id'])
    need(len(ids) == 14, 'fourteen distinct artifact IDs')
    need(type(lock['api']) is dict and set(lock['api']) == set(API_ROLES)
         and all(valid_pin(p, 32_000_000) for p in lock['api'].values()), 'exact five final API byte pins')
    need(type(lock['logs']) is dict and set(lock['logs']) == {'parakeet', 'readback'}, 'both failed original job logs')
    for role, name in [('parakeet', 'PARAKEET_JOB.log'), ('readback', 'READBACK_JOB.log')]:
        value = lock['logs'][role]
        need(type(value) is dict and set(value) == {'path', 'job_id', 'bytes', 'sha256'}
             and value['path'] == name and type(value['job_id']) is int and value['job_id'] > 0
             and valid_pin({k: value[k] for k in ('bytes', 'sha256')}, 32_000_000), 'complete original log pin ' + role)
    need(lock['logs']['parakeet']['job_id'] != lock['logs']['readback']['job_id'], 'distinct failed jobs')
    need(all(valid_hash(lock[k]) for k in ('source_producer_manifest_sha256', 'source_manifest_sha256',
        'history_sha256', 'source_assessment_sha256', 'candidate_pt_sha256', 'train_result_sha256')), 'frozen original receipt hashes')
    validate_counts(lock['provider_counts'])
    return lock


def regular(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular saved evidence file')
    return path


def read_pinned(path, expected):
    raw = regular(path).read_bytes()
    need(pin(raw) == {k: expected[k] for k in ('bytes', 'sha256')}, 'exact saved evidence bytes ' + str(path))
    return raw


def original(path, expected, role):
    raw = read_pinned(path, expected)
    need(len(raw) <= archive_limit(role), 'bounded original archive bytes ' + role)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    infos = archive.infolist()
    need(len(infos) <= 10000 and len({i.filename for i in infos}) == len(infos)
         and sum(i.file_size for i in infos) <= archive_limit(role), 'bounded unique original ZIP members ' + role)
    for info in infos:
        name = info.filename
        need(name and '\\' not in name and not PurePosixPath(name).is_absolute()
             and '..' not in PurePosixPath(name).parts and ':' not in name.split('/')[0]
             and str(PurePosixPath(name)) == name.rstrip('/') and not info.flag_bits & 1
             and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'safe original ZIP member')
    need(archive.testzip() is None, 'all original ZIP CRCs ' + role)
    return archive


@contextmanager
def offline_only():
    old_import, old_popen, old_urlopen = builtins.__import__, subprocess.Popen, urllib.request.urlopen
    old_socket, old_connection = socket.socket, socket.create_connection
    forbidden = {'torch', 'numpy', 'torchaudio', 'transformers', 'onnxruntime', 'piper', 'soundfile',
        'requests', 'httpx', 'tensorflow', 'jax', 'huggingface_hub', 'qwen_asr', 'qwen_tts', 'whisper'}
    audit = dict(original_stop_reopens=0)
    need(not forbidden.intersection(sys.modules), 'fresh stdlib-only saved-evidence process')
    def import_guard(name, *args, **kwargs):
        need(name.split('.')[0] not in forbidden, 'no model/network runtime during STOP readback')
        return old_import(name, *args, **kwargs)
    def process_guard(command, *args, **kwargs):
        need(type(command) is list and len(command) == 8 and command[:4] == [sys.executable, '-I', '-S', '-B']
             and command[5] == '--verify-worker' and not args
             and set(kwargs) <= {'stdin', 'stdout', 'stderr'}, 'only immutable nested original STOP subprocess')
        path = regular(command[4])
        need(path.is_absolute() and path == path.resolve()
             and path.name == 'english_v6_b24r1_b24_source_failure.py'
             and sha(path.read_bytes()) == OLD_READER_SHA, 'exact unchanged original B24 failure reader')
        evidence, lock = Path(command[6]), regular(command[7])
        need(evidence.is_absolute() and evidence == evidence.resolve() and evidence.is_dir()
             and evidence.name == 'ORIGINAL_B24_STOP' and lock.is_absolute() and lock == lock.resolve(),
             'actual retained old STOP directory and local complete lock')
        result = old_popen(command, **kwargs)
        audit['original_stop_reopens'] += 1
        return result
    def blocked(*args, **kwargs):
        raise ValueError('no network during saved SOURCE STOP readback')
    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs):
            blocked()
    builtins.__import__, subprocess.Popen, urllib.request.urlopen = import_guard, process_guard, blocked
    socket.socket, socket.create_connection = BlockedSocket, blocked
    try:
        yield audit
        need(not forbidden.intersection(sys.modules), 'no scientific runtime imported')
    finally:
        builtins.__import__, subprocess.Popen, urllib.request.urlopen = old_import, old_popen, old_urlopen
        socket.socket, socket.create_connection = old_socket, old_connection


def final_metadata(directory, lock):
    values = {name: document(read_pinned(directory / 'metadata' / ('source_' + name + '.json'), lock['api'][name]))
              for name in API_ROLES}
    run, inventory, commit, tree, jobs = (values[n] for n in API_ROLES)
    need(run['id'] == RUN and run['head_sha'] == HEAD and run['head_branch'] == BRANCH
         and run['status'] == 'completed' and run['conclusion'] == 'failure'
         and run['run_attempt'] == 1 and run['event'] == 'push' and run['path'] == WORKFLOW,
         'final first-attempt failed SOURCE run; generation snapshots cannot complete this proof')
    need(all(run[k]['full_name'] == REPO and run[k]['private'] is False for k in ('repository', 'head_repository')),
         'same original public repository')
    need(commit['sha'] == HEAD and [p['sha'] for p in commit['parents']] == [PARENT]
         and [f['filename'] for f in commit['files']] == [WORKFLOW]
         and tree['sha'] == commit['commit']['tree']['sha'] and tree['truncated'] is False,
         'original complete workflow-only activation Git identity')
    need(len({r['path'] for r in tree['tree']}) == len(tree['tree']), 'unique complete Git tree')
    need(inventory['total_count'] == len(inventory['artifacts']) == 14
         and {a['name'] for a in inventory['artifacts']} == set(NAMES.values()), 'fourteen originals, no SCREEN artifact')
    expected_jobs = {'contracts': 'success', 'sources': 'success', 'readback': 'failure',
        **{'observers (' + p + ')': ('failure' if p == 'parakeet' else 'success') for p in PROVIDERS}}
    need(jobs['total_count'] == len(jobs['jobs']) == 8
         and {j['name']: j['conclusion'] for j in jobs['jobs']} == expected_jobs
         and all(j['run_id'] == RUN and j['head_sha'] == HEAD and j['run_attempt'] == 1
                 and j['status'] == 'completed' for j in jobs['jobs']), 'all eight final original jobs')
    for role, name in [('parakeet', 'observers (parakeet)'), ('readback', 'readback')]:
        need(next(j['id'] for j in jobs['jobs'] if j['name'] == name) == lock['logs'][role]['job_id'],
             'original failed log belongs to final job ' + role)
    return values


def producer_facts(archive, tree, lock):
    raw = archive.read('PRODUCER.json')
    need(sha(raw) == lock['source_producer_manifest_sha256'], 'exact captured original producer manifest')
    producer = document(raw)
    need(producer['schema'] == 'english-v6-b24r1-complete-source-producer-v1' and producer['role'] == 'source'
         and producer['run_head'] == HEAD and str(producer['run_id']) == str(RUN) and producer['run_attempt'] == 1
         and producer['run_branch'] == BRANCH and producer['runtime_workflow'] == WORKFLOW
         and producer['prepared_parent'] == PARENT and producer['git_tree'] == tree['sha'], 'actual captured R1 producer identity')
    need({i.filename for i in archive.infolist() if not i.is_dir()}
         == {'PRODUCER.json'} | {'repository/' + name for name in producer['files']}, 'complete exact producer members')
    git = {r['path']: r for r in tree['tree']}
    for name, expected in producer['files'].items():
        payload = archive.read('repository/' + name)
        blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        need(pin(payload) == {k: expected[k] for k in ('bytes', 'sha256')}
             and blob == expected['git_blob_sha1'] == git[name]['sha']
             and expected['mode'] == git[name]['mode'] and git[name]['type'] == 'blob', 'original producer Git bytes/mode ' + name)
    template = archive.read('repository/automation/english_v6_b24r1_source_workflow_template.yml')
    token = b'__B24R1_REVIEWED_SOURCE_PARENT__'
    need(template.count(token) == 2 and archive.read('repository/' + WORKFLOW) == template.replace(token, PARENT.encode()),
         'only original two-parent substitutions activated the source')
    need(producer['python_files'] == sorted(n for n in producer['files'] if n.endswith('.py'))
         and producer['resource_files'] == sorted(n for n in producer['files'] if not n.endswith('.py')),
         'exact source Python and resource inventories')
    return producer


def provider_ledger(archive, rows, provider, source_sha, producer):
    names = {i.filename for i in archive.infolist() if not i.is_dir()}
    closure = document(archive.read('FILES.json'))
    need(closure == dict(schema='english-v6-b18-observer-file-closure-v1', self_excluded='FILES.json',
        files=[dict(path=n, **pin(archive.read(n))) for n in sorted(names) if n != 'FILES.json']), 'every original native FILES member ' + provider)
    allowed_rows = {r['id'] for r in rows}
    need(all(not n.startswith('rows/') or n.split('/')[1] in allowed_rows for n in names), 'no unregistered native rows')
    ledger = []
    for row in rows:
        prefix = 'rows/' + row['id'] + '/'
        item = dict(id=row['id'])
        for key, name in [('input_started', 'STARTED.json'), ('forward_attempted', 'FORWARD_STARTED.json'),
                ('native_returned', 'FORWARD_RETURNED.json'), ('payload_valid', 'PAYLOAD_VALID.json'),
                ('capture_saved', 'CAPTURE.json'), ('failed', 'FAILED.json')]:
            item[key] = prefix + name in names
            if item[key]:
                item[key + '_sha256'] = sha(archive.read(prefix + name))
        item['raw_payload_saved'] = any(prefix + n in names for n in ('RAW_RETURN.json', 'RAW_PROCESS.json', 'RAW_PHONE.json'))
        ledger.append(item)
    counts = {k: sum(int(r[k]) for r in ledger) for k in COUNTERS[:7]}
    need(document(archive.read('LEDGER.json')) == dict(schema='english-v6-b18-native-attempt-return-valid-ledger-v1',
        provider=provider, source_manifest_sha256=source_sha, rows=ledger, counts=counts,
        attempted_ids=[r['id'] for r in ledger if r['forward_attempted']],
        returned_ids=[r['id'] for r in ledger if r['native_returned']],
        payload_valid_ids=[r['id'] for r in ledger if r['payload_valid']]), 'all 120 original journal rows ' + provider)
    report = document(archive.read('RESULT.json'))
    need(report['provider'] == provider and report['source_manifest_sha256'] == source_sha
         and report['schema'] == 'english-v6-b24r1-120-original-strict-source-screen-v1'
         and report['denominator'] == 120 and report['actual_native_forward_attempts'] == report['attempted'] == counts['forward_attempted']
         and report['native_forward_returns'] == counts['native_returned'] and report['valid_native_payloads'] == counts['payload_valid']
         and report['complete'] == counts['capture_saved'] and report['missing_or_failed'] == 120 - counts['capture_saved']
         and report['retry_count'] == report['kws_forward_calls'] == report['new_tts_calls'] == 0
         and report['new_asr_calls'] == (counts['forward_attempted'] if provider in ('qwen', 'whisper', 'parakeet') else 0)
         and report['new_phone_calls'] == (counts['forward_attempted'] if provider in ('xlsr', 'lv60') else 0)
         and all(report[k] is False for k in ('target_conditioned', 'training_allowed', 'promotion_allowed', 'human_gold'))
         and report['run'] == dict(run=str(RUN), sha=HEAD) and report['initialization_error'] is None
         and report['whole_panel_stop'] is (provider == 'parakeet')
         and report['row_error'] == ('ValueError: EMPTY_NATIVE_TRANSCRIPT' if provider == 'parakeet' else None)
         and report['observer_code_sha256'] == producer['files']['automation/english_v6_b24r1_source_observe.py']['sha256']
         and report['asr_adapter_sha256'] == producer['files']['automation/english_v6_b18_asr_adapter.py']['sha256'],
         'original provider result agrees with every saved entry ' + provider)
    counts.update(status=report['status'], not_started=sum(not r['input_started'] for r in ledger),
        entered_without_return=sum(r['forward_attempted'] and not r['native_returned'] for r in ledger),
        entered_without_raw=sum(r['forward_attempted'] and not r['raw_payload_saved'] for r in ledger))
    if provider == 'parakeet':
        need([r['id'] for r in ledger if r['failed']] == [FAILED_ID]
             and [r['id'] for r in ledger if r['forward_attempted']] == [r['id'] for r in rows[:9]]
             and rows[8]['id'] == FAILED_ID, 'exact ninth original Parakeet row, no substituted failure')
        process = document(archive.read('rows/' + FAILED_ID + '/RAW_PROCESS.json'))
        need(process['native_exit_code'] == 0 and process['native_return_observed'] is True
             and process['timed_out'] is False and process['exception_type'] is None
             and archive.read('rows/' + FAILED_ID + '/RAW_STDOUT.bin') == b'\n', 'genuine saved native empty return')
    return counts


def archive_facts(directory, lock):
    validate_lock(lock)
    directory = Path(directory).resolve()
    metadata = final_metadata(directory, lock)
    with ExitStack() as stack:
        archives, pins = {}, {}
        inventory = {a['name']: a for a in metadata['artifacts']['artifacts']}
        for role, expected in lock['artifacts'].items():
            item = inventory[expected['name']]
            need(item['id'] == expected['id'] and item['size_in_bytes'] == expected['bytes']
                 and item['digest'] == 'sha256:' + expected['sha256'] and item['expired'] is False
                 and item['workflow_run']['id'] == RUN and item['workflow_run']['head_sha'] == HEAD
                 and item['workflow_run']['head_branch'] == BRANCH, 'same original artifact API provenance ' + role)
            archive = stack.enter_context(original(directory / (role + '.zip'), expected, role))
            archives[role] = archive
            pins[role] = dict(expected, run_id=RUN, run_head=HEAD, members=len(archive.infolist()),
                expanded_bytes=sum(i.file_size for i in archive.infolist()), crc_verified=True)
        producer = producer_facts(archives['source_producer'], metadata['tree'], lock)
        source = archives['source']
        manifest_raw = source.read('MANIFEST.json')
        manifest = document(manifest_raw)
        rows = manifest['rows']
        need(sha(manifest_raw) == lock['source_manifest_sha256'] and sha(source.read('HISTORY.json')) == lock['history_sha256']
             and manifest['assessment_sha256'] == ASSESSMENT and manifest['original_training_assessment_sha256'] == TRAIN_ASSESSMENT
             and manifest['schema'] == 'english-v6-b24r1-fresh-combined-source-v1' and manifest['panel'] == 'primary', 'exact separate R1 source identity')
        need(manifest['source_execution'] == dict(prepared_parent=PARENT, run_attempt=1,
            run_branch='refs/heads/' + BRANCH, run_head=HEAD, run_id=str(RUN))
             and manifest['replication_opened'] is manifest['automatic_R2_allowed'] is False, 'same stopped source and unopened reserve')
        with zipfile.ZipFile(io.BytesIO(source.read('TRAIN_models.zip'))) as models:
            train_raw = models.read('RESULT.json')
            need(sha(train_raw) == lock['train_result_sha256']
                 and document(train_raw)['model_pt_sha256']['challenger'] == CANDIDATE_PT, 'unchanged original TRAIN candidate')
        need(len(rows) == len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == 120
             and manifest['tts_calls'] == 96 and manifest['natural_original_decodes'] == 24
             and manifest['new_kws_forwards'] == manifest['new_optimizer_updates'] == 0, 'all 120 saved source originals')
        for row in rows:
            raw = source.read('wav/' + row['id'] + '.wav')
            need(sha(raw) == row['wav_sha256'], 'whole original WAV ' + row['id'])
            with wave.open(io.BytesIO(raw)) as stream:
                need(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getframerate() == 16000
                     and stream.getnframes() == row['samples'], 'whole original PCM dimensions')
                pcm = stream.readframes(stream.getnframes())
                need(not stream.readframes(1) and sha(pcm) == row['pcm_sha256'], 'all original PCM bytes')
            if row['cohort'] == 'synthetic':
                chunks = []
                for index, part in enumerate(row['native_chunks']):
                    need(part['file'] == f'{index:03d}.s16le', 'original native chunk order')
                    chunk = source.read('native/' + row['id'] + '/' + part['file'])
                    need(len(chunk) == part['samples'] * 2 and sha(chunk) == part['native_pcm_sha256'], 'all native chunk bytes')
                    chunks.append(chunk)
                need(sha(b''.join(chunks)) == row['native_pcm_sha256'], 'whole native source bytes')
        need(source.read('CONSUMPTION.json') == archives['source_consumed'].read('CONSUMPTION.json')
             and base64.b64decode(document(source.read('CONSUMPTION_UPLOAD.json'))['original_zip_base64'], validate=True)
                 == (directory / 'source_consumed.zip').read_bytes(), 'independent original source consumed marker')
        for info in archives['transport'].infolist():
            if not info.is_dir():
                need(archives['transport'].read(info.filename) == source.read(info.filename), 'original transport member ' + info.filename)
        counts = {}
        for provider in PROVIDERS:
            native = archives['native_' + provider]
            need(base64.b64decode(document(native.read('OBSERVER_BUDGET.json'))['original_zip_base64'], validate=True)
                 == (directory / ('consumed_' + provider + '.zip')).read_bytes(), 'independent original observer consumed marker ' + provider)
            counts[provider] = provider_ledger(native, rows, provider, lock['source_manifest_sha256'], producer)
        validate_counts(counts)
        need(counts == lock['provider_counts'], 'all final provider counts equal the reviewed lock')
        logs = {role: read_pinned(directory / expected['path'], expected) for role, expected in lock['logs'].items()}
        need(b'WHOLE_PANEL_STOP_NATIVE_RETURN_OR_PAYLOAD_FAILURE_NO_RETRY' in logs['parakeet'], 'original provider stopped without retry')
        need(b'ValueError: ALL_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES' in logs['readback']
             and b'No artifacts will be uploaded.' in logs['readback'] and not (directory / 'screen.zip').exists(),
             'original readback failed and no SCREEN was produced')
        return dict(original_artifacts=pins, producer_paths=len(producer['files']), provider_counts=counts,
            producer_manifest_sha256=lock['source_producer_manifest_sha256'], source_manifest_sha256=lock['source_manifest_sha256'],
            original_api=lock['api'], original_logs=lock['logs'], whole_original_pcm_verified=120,
            all_original_native_chunks_verified=True, all14_original_zip_digests_crc_verified=True,
            all_original_producer_git_blobs_verified=True, all5_original_FILES_closures_and120_journals_recomputed=True)


def reparse_empty_return(raw_asr, point, identity, capture):
    """No CAPTURE/PAYLOAD_VALID is fabricated for the one rejected native return."""
    need(identity['provider'] == 'parakeet' and identity['id'] == FAILED_ID
         and capture == document((point / 'FAILED.json').read_bytes())
         and document((point / 'STARTED.json').read_bytes()) == identity
         and all(capture.get(k) == v for k, v in identity.items())
         and capture['error'] == 'ValueError: EMPTY_NATIVE_TRANSCRIPT' and capture['retry_count'] == 0
         and capture['whole_panel_stop'] is True and not (point / 'CAPTURE.json').exists()
         and not (point / 'PAYLOAD_VALID.json').exists() and (point / 'RAW_STDOUT.bin').read_bytes() == b'\n',
         'one genuine saved empty return; no synthesized successful capture')
    pointer = dict(raw_evidence=dict(parser_contract_sha256=raw_asr.PARSER_CONTRACT_SHA256,
        payload=raw_asr.pin(point / 'RAW_PROCESS.json')))
    try:
        raw_asr.validate_capture('parakeet', point, identity, pointer)
    except ValueError as exc:
        need(str(exc) == 'EMPTY_NATIVE_TRANSCRIPT', 'original parser rejects exactly the saved empty return')
    else:
        raise ValueError('a valid return cannot establish the R1 SOURCE STOP')


def reparse_phone(original_phone, folder, provider, model, capture):
    """Repeat the captured B11 loader's pure model/vocabulary/finite/CTC checks.

    Do not call its five-provider loader or any source admission/quality decision.
    The original strict _verify_phone separately authenticates journals, full
    frame geometry and the retained RAW_PHONE/PAYLOAD_VALID receipts.
    """
    index = original_phone.policy.PHONE_MODELS.index(provider)
    name, revision = original_phone.pcm_tools.PHONE_MODELS[index]
    need(model['target_conditioned'] is False and model['model'] == name
         and model['revision'].startswith(revision)
         and model['weights_sha256'] == original_phone.phones.WEIGHT_SHA[index], 'locked original phoneme source')
    vocabulary = document((folder / 'VOCAB.json').read_bytes())
    need(vocabulary['model'] == name and vocabulary['revision'] == model['revision']
         and vocabulary['pad_token_id'] == 0, 'original native vocabulary lock')
    raw = (folder / 'logits' / (capture['id'] + '.f32')).read_bytes()
    need(len(raw) == 4 * capture['logit_frames'] * capture['vocab_size'] and capture['vocab_size'] == 392
         and sha(raw) == capture['logits_sha256'], 'all original phone logits and signed capture')
    original_phone.policy.verified_phone_input(raw, vocabulary['vocab'], pad_token_id=0)
    ids = [max(range(392), key=frame.__getitem__) for frame in struct.iter_unpack('<392f', raw)]
    need(original_phone.phones.ctc(ids, vocabulary['vocab'], 0) == capture['raw_phone_tokens'],
         'original complete greedy CTC tokens match saved capture')
    return len(ids)


def semantic_readback(directory, lock):
    """Use the captured original pure readers, preserving their rejection of the empty return."""
    with tempfile.TemporaryDirectory(prefix='b24r1-source-stop-originals-') as temporary:
        root = Path(temporary)
        source_dir, providers = root / 'source', root / 'providers'
        for role, target in [('source_producer', root / 'producer'), ('source', source_dir),
                *[('native_' + p, providers / p) for p in PROVIDERS]]:
            target.mkdir(parents=True)
            with original(Path(directory) / (role + '.zip'), lock['artifacts'][role], role) as archive:
                archive.extractall(target)
        captured = root / 'producer' / 'repository'
        sys.path.insert(0, str(captured / 'automation'))
        source = importlib.import_module('english_v6_b24r1_source')
        producer_module = importlib.import_module('english_v6_b24r1_source_producer')
        original_phone = importlib.import_module('english_v6_b11_supplement_screen')
        producer = document((root / 'producer' / 'PRODUCER.json').read_bytes())
        tree = document((Path(directory) / 'metadata/source_tree.json').read_bytes())
        tracked = {r['path']: dict(mode=r['mode'], git_blob_sha1=r['sha']) for r in tree['tree'] if r['type'] == 'blob'}
        need(set(producer['files']) == producer_module.select_paths(tracked, captured), 'complete actual producer dependency closure')
        for name, digest in producer_module.dynamic.required_paths(captured).items():
            need(sha((captured / name).read_bytes()) == digest, 'all nineteen exact dynamic helper bytes')
        # _read fully reopens retained original B24 STOP in its immutable isolated
        # worker. No replacement report, patched parser or current-checkout fallback.
        owner = source.evaluation_preflight(source_dir)
        rows = owner['rows']
        need(len(rows) == len(owner['all_original_rows']) == 120 and owner['assessment_sha256'] == ASSESSMENT,
             'captured original source preflight authenticates all 120 originals and ancestry')
        obs, raw_asr = source.observers, source.observers.raw_asr
        contract = source.code_contract()
        registration = dict(assessment_sha256=source.ASSESSMENT_SHA, source_schema=source.SCHEMA,
            batch_schema=source.BATCH, source_manifest_sha256=lock['source_manifest_sha256'],
            source_rows=120, max_calls=120, **owner['source_execution'],
            **{k: contract[k] for k in ('observer_sha256', 'asr_adapter_sha256', 'parser_contract_sha256')})
        obs.validate_rows(rows, lock['source_manifest_sha256'], registration)
        parsed, failed, phone_frames = {}, [], 0
        for provider in PROVIDERS:
            folder = providers / provider
            obs.lineage.validate_observer_budget(document((folder / 'OBSERVER_BUDGET.json').read_bytes()), provider, registration)
            started = document((folder / 'STARTED.json').read_bytes())
            need(started['registration'] == registration and started['whole_panel_stop_on_failure'] is True
                 and started['old_admission_function_sha256'] == obs.POLICY_FUNCTIONS
                 and started['provider'] == provider and started['input_manifest'] == lock['source_manifest_sha256']
                 and started['batch'] == source.BATCH and started['denominator'] == 120
                 and started['parser_contract_sha256'] == raw_asr.PARSER_CONTRACT_SHA256
                 and started['run'] == dict(run=str(RUN), sha=HEAD), 'original frozen observer registration ' + provider)
            report = document((folder / 'RESULT.json').read_bytes())
            model = document((folder / 'MODEL.json').read_bytes())
            need(report['parser_contract_sha256'] == raw_asr.PARSER_CONTRACT_SHA256, 'unchanged raw parser contract')
            by_id = {r['id']: r for r in report['rows']}
            need(len(by_id) == len(report['rows']) == 120 and set(by_id) == {r['id'] for r in rows}, 'all 120 original result slots')
            ledger = obs.row_ledger(folder, rows)
            need(obs.ledger_counts(ledger) == {k: lock['provider_counts'][provider][k] for k in COUNTERS[:7]},
                 'original captured ledger reader independently recomputes counts')
            count = 0
            for row in rows:
                point = folder / 'rows' / row['id']
                capture = by_id[row['id']]
                identity = obs.binding(provider, row, lock['source_manifest_sha256'])
                if capture['status'] == 'success':
                    need(capture == document((point / 'CAPTURE.json').read_bytes())
                         and document((point / 'STARTED.json').read_bytes()) == identity
                         and capture['input_coverage_complete'] is True and all(capture.get(k) == v for k, v in identity.items()),
                         'original complete capture identity')
                    if provider in raw_asr.TEXT_PROVIDERS:
                        need(model['adapter_sha256'] == contract['asr_adapter_sha256']
                             and model['parser_contract_sha256'] == raw_asr.PARSER_CONTRACT_SHA256
                             and model['runtime_versions'] == contract['asr_runtime_versions'][provider]
                             and model['observer'] == contract['asr_models'][provider]
                             and model['raw_before_validation'] is model['decode_config_unchanged'] is True,
                             'unchanged ASR runtime and raw parser')
                        raw_asr.validate_capture(provider, point, identity, capture)
                    else:
                        obs._verify_phone(folder, point, identity, capture, model)
                        phone_frames += reparse_phone(original_phone, folder, provider, model, capture)
                    count += 1
                elif capture['status'] == 'failed':
                    reparse_empty_return(raw_asr, point, identity, capture)
                    failed.append(row['id'])
                else:
                    need(provider == 'parakeet' and capture['status'] == 'not_started' and not point.exists(),
                         'all 111 never-entered calls remain absent without reconstruction')
            parsed[provider] = count
        need(parsed == {p: lock['provider_counts'][p]['payload_valid'] for p in PROVIDERS} and failed == [FAILED_ID],
             'every final valid payload and exactly the saved empty return reparsed')
        modules = []
        for name, value in sys.modules.items():
            path = getattr(value, '__file__', None)
            if path and name.startswith(('english_', 'b24_original_', 'b24r1_original_')):
                need(Path(path).resolve().is_relative_to(captured), 'captured original repository only ' + name)
                modules.append(name)
        old_report = (source_dir / 'B24_SOURCE_STOP_REPORT.json').read_bytes()
        need(sha(old_report) == OLD_REPORT_SHA, 'unchanged original B24 STOP report fully reopened by captured source preflight')
        return dict(valid_payloads_reparsed=parsed, total_valid_payloads_reparsed=sum(parsed.values()),
            empty_return_reparsed_by_original_parser=True, empty_return_source_id=FAILED_ID,
            full_original_phone_frames_reparsed=phone_frames, original_source_preflight_recomputed=True,
            original_phone_model_vocab_finite_logits_and_greedy_ctc_recomputed=True,
            all_original120_pcm_and_native_chunks_recomputed=True, complete_original_producer_closure_recomputed=True,
            original_B24_STOP_report_sha256=sha(old_report), nested_original_B24_STOP_fully_reauthenticated=True,
            captured_repository_modules=sorted(modules), original_captured_readers_only=True, checkout_fallback_used=False)


def verify_worker(directory, lock, audit):
    facts = archive_facts(directory, lock)
    facts.update(semantic_readback(directory, lock))
    need(audit['original_stop_reopens'] > 0, 'captured source reader actually reopened nested original B24 STOP')
    counts = facts['provider_counts']
    return dict(facts, schema=REPORT_SCHEMA, status='SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT',
        run_id=RUN, head=HEAD, prepared_parent=PARENT, run_conclusion='failure', artifact_count=14,
        original_lock_sha256=sha(canonical(lock)), reader_sha256=sha(Path(__file__).read_bytes()),
        canonical_encoding='ASCII JSON; sorted keys; compact separators; no newline; no NaN',
        original_assessment_sha256=ASSESSMENT, original_training_assessment_sha256=TRAIN_ASSESSMENT,
        original_candidate_pt_sha256=CANDIDATE_PT, original_train_result_sha256=lock['train_result_sha256'],
        original_stop_reopen_count=audit['original_stop_reopens'], whole_source_stopped=True,
        original_B24_source_stop_permanent=True, R1_source_stop_permanent=True,
        planned_native_calls=600, actual_native_attempts=sum(c['forward_attempted'] for c in counts.values()),
        actual_native_returns=sum(c['native_returned'] for c in counts.values()),
        raw_payloads_saved=sum(c['raw_payload_saved'] for c in counts.values()),
        valid_native_payloads=sum(c['payload_valid'] for c in counts.values()), empty_native_returns=1,
        never_started=sum(c['not_started'] for c in counts.values()), entered_without_return=0, entered_without_raw=0,
        screen_artifact_present=False, source_gate=None, source_admission_performed=False,
        quality_verdict=None, model_quality_claim=None, primary_kws_evaluation_performed=False,
        new_source_calls=0, new_observer_calls=0, new_model_calls=0, new_optimizer_updates=0,
        **dict.fromkeys(FALSE_PERMISSIONS, False))


def verify(directory, lock):
    validate_lock(lock)
    with tempfile.TemporaryDirectory(prefix='b24r1-source-stop-lock-') as temporary:
        path = Path(temporary) / 'lock.json'
        path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()),
            '--verify-worker', str(Path(directory).resolve()), str(path)])
    report = document(raw)
    need(raw == canonical(report) and report['schema'] == REPORT_SCHEMA
         and report['status'] == 'SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT'
         and report['original_lock_sha256'] == sha(canonical(lock)) and report['reader_sha256'] == sha(Path(__file__).read_bytes())
         and report['provider_counts'] == lock['provider_counts'] and report['original_stop_reopen_count'] > 0
         and report['nested_original_B24_STOP_fully_reauthenticated'] is True
         and report['quality_verdict'] is report['model_quality_claim'] is None
         and all(report[k] is False for k in FALSE_PERMISSIONS), 'complete fail-only isolated R1 SOURCE STOP proof')
    return report


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--verify-worker':
        lock = document(regular(sys.argv[3]).read_bytes())
        with offline_only() as audit:
            report = verify_worker(Path(sys.argv[2]), lock, audit)
        sys.stdout.buffer.write(canonical(report))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('check')
    item = modes.add_parser('verify')
    item.add_argument('originals', type=Path)
    item.add_argument('lock', type=Path)
    item.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.mode == 'check':
        print('INERT: final original evidence required; no source, observer, model, retry, R2 or downstream permission')
    else:
        need(args.report.is_absolute() and args.report.parent.is_dir() and not args.report.exists(), 'new absolute report path')
        report = verify(args.originals, document(regular(args.lock).read_bytes()))
        with args.report.open('xb') as stream:
            stream.write(canonical(report))
        print(json.dumps(dict(status=report['status'], report_sha256=sha(canonical(report)),
            actual_native_attempts=report['actual_native_attempts'], valid_native_payloads=report['valid_native_payloads'],
            never_started=report['never_started'], quality_verdict=None, automatic_R2_allowed=False), sort_keys=True))


if __name__ == '__main__':
    main()
