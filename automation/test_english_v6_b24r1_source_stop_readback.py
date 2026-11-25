"""Invented saved-evidence mutations, never execution or quality evidence."""
from __future__ import annotations
import copy
import hashlib
import importlib
import io
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24r1_source_stop_readback as reader


def counts_fixture():
    result = {}
    for provider in reader.PROVIDERS:
        row = dict(input_started=120, forward_attempted=120, native_returned=120, raw_payload_saved=120,
            payload_valid=120, capture_saved=120, failed=0, not_started=0,
            entered_without_return=0, entered_without_raw=0, status='COMPLETE')
        if provider == 'parakeet':
            row.update(input_started=9, forward_attempted=9, native_returned=9, raw_payload_saved=9,
                payload_valid=8, capture_saved=8, failed=1, not_started=111, status='STOP_OBSERVER_FAILURE')
        result[provider] = row
    return result


def lock_fixture():
    result = dict(schema=reader.LOCK_SCHEMA, repo=reader.REPO, run_id=reader.RUN, run_head=reader.HEAD,
        prepared_parent=reader.PARENT, branch=reader.BRANCH, workflow=reader.WORKFLOW,
        artifacts={role: dict(id=index + 1, name=name, bytes=100, sha256='a' * 64)
                   for index, (role, name) in enumerate(reader.NAMES.items())},
        api={role: dict(bytes=10, sha256='b' * 64) for role in reader.API_ROLES},
        logs={role: dict(path=name, job_id=index + 200, bytes=10, sha256='c' * 64)
              for index, (role, name) in enumerate([('parakeet', 'PARAKEET_JOB.log'), ('readback', 'READBACK_JOB.log')])},
        provider_counts=counts_fixture(), source_producer_manifest_sha256='d' * 64,
        source_manifest_sha256='e' * 64, history_sha256='f' * 64,
        source_assessment_sha256=reader.ASSESSMENT, candidate_pt_sha256=reader.CANDIDATE_PT,
        train_result_sha256='9' * 64)
    return result


def api_fixture(lock):
    jobs = {'contracts': 'success', 'sources': 'success', 'readback': 'failure',
        **{'observers (' + p + ')': ('failure' if p == 'parakeet' else 'success') for p in reader.PROVIDERS}}
    return dict(run=dict(id=reader.RUN, head_sha=reader.HEAD, head_branch=reader.BRANCH,
        status='completed', conclusion='failure', run_attempt=1, event='push', path=reader.WORKFLOW,
        repository=dict(full_name=reader.REPO, private=False), head_repository=dict(full_name=reader.REPO, private=False)),
        artifacts=dict(total_count=14, artifacts=[dict(name=n) for n in reader.NAMES.values()]),
        commit=dict(sha=reader.HEAD, parents=[dict(sha=reader.PARENT)], files=[dict(filename=reader.WORKFLOW)],
            commit=dict(tree=dict(sha='1' * 40))), tree=dict(sha='1' * 40, truncated=False, tree=[]),
        jobs=dict(total_count=8, jobs=[dict(name=name, conclusion=conclusion, run_id=reader.RUN,
            head_sha=reader.HEAD, run_attempt=1, status='completed', id=(
                lock['logs']['parakeet']['job_id'] if name == 'observers (parakeet)' else
                lock['logs']['readback']['job_id'] if name == 'readback' else index + 400))
            for index, (name, conclusion) in enumerate(jobs.items())]))


def write_api(root, lock, values):
    (root / 'metadata').mkdir(exist_ok=True)
    for role, value in values.items():
        raw = reader.canonical(value)
        (root / 'metadata' / ('source_' + role + '.json')).write_bytes(raw)
        lock['api'][role] = reader.pin(raw)


def zipped(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return stream.getvalue()


def native_fixture():
    rows = [dict(id=('b24r1-invented-' + str(i)) if i != 8 else reader.FAILED_ID) for i in range(120)]
    files, ledger = {}, []
    keys = [('input_started', 'STARTED.json'), ('forward_attempted', 'FORWARD_STARTED.json'),
        ('native_returned', 'FORWARD_RETURNED.json'), ('payload_valid', 'PAYLOAD_VALID.json'),
        ('capture_saved', 'CAPTURE.json'), ('failed', 'FAILED.json')]
    for index, row in enumerate(rows):
        prefix = 'rows/' + row['id'] + '/'
        entry = dict(id=row['id'])
        for key, name in keys:
            present = index < (8 if key in ('payload_valid', 'capture_saved') else 9) if key != 'failed' else index == 8
            entry[key] = present
            if present:
                raw = reader.canonical(dict(id=row['id'], journal=name))
                files[prefix + name] = raw
                entry[key + '_sha256'] = reader.sha(raw)
        entry['raw_payload_saved'] = index < 9
        if index < 9:
            files[prefix + 'RAW_PROCESS.json'] = reader.canonical(dict(native_exit_code=0,
                native_return_observed=True, timed_out=False, exception_type=None))
            files[prefix + 'RAW_STDOUT.bin'] = b'\n' if index == 8 else b'invented valid transcript\n'
        ledger.append(entry)
    source_sha = 'e' * 64
    counts = {k: counts_fixture()['parakeet'][k] for k in reader.COUNTERS[:7]}
    files['LEDGER.json'] = reader.canonical(dict(schema='english-v6-b18-native-attempt-return-valid-ledger-v1',
        provider='parakeet', source_manifest_sha256=source_sha, rows=ledger, counts=counts,
        attempted_ids=[r['id'] for r in rows[:9]], returned_ids=[r['id'] for r in rows[:9]],
        payload_valid_ids=[r['id'] for r in rows[:8]]))
    report = dict(provider='parakeet', source_manifest_sha256=source_sha,
        schema='english-v6-b24r1-120-original-strict-source-screen-v1', denominator=120,
        actual_native_forward_attempts=9, attempted=9, native_forward_returns=9, valid_native_payloads=8,
        complete=8, missing_or_failed=112, retry_count=0, kws_forward_calls=0, new_tts_calls=0,
        new_asr_calls=9, new_phone_calls=0, target_conditioned=False, training_allowed=False,
        promotion_allowed=False, human_gold=False, run=dict(run=str(reader.RUN), sha=reader.HEAD),
        initialization_error=None, whole_panel_stop=True, row_error='ValueError: EMPTY_NATIVE_TRANSCRIPT',
        observer_code_sha256='1' * 64, asr_adapter_sha256='2' * 64, status='STOP_OBSERVER_FAILURE')
    files['RESULT.json'] = reader.canonical(report)
    producer = dict(files={'automation/english_v6_b24r1_source_observe.py': dict(sha256='1' * 64),
        'automation/english_v6_b18_asr_adapter.py': dict(sha256='2' * 64)})
    return rows, files, source_sha, producer


def native_archive(files):
    files = dict(files)
    files['FILES.json'] = reader.canonical(dict(schema='english-v6-b18-observer-file-closure-v1',
        self_excluded='FILES.json', files=[dict(path=n, **reader.pin(raw)) for n, raw in sorted(files.items()) if n != 'FILES.json']))
    return zipfile.ZipFile(io.BytesIO(zipped(files)))


def producer_fixture():
    template = b'parent=__B24R1_REVIEWED_SOURCE_PARENT__\nhead=__B24R1_REVIEWED_SOURCE_PARENT__\n'
    files = {'automation/english_v6_b24r1_source_workflow_template.yml': template,
        reader.WORKFLOW: template.replace(b'__B24R1_REVIEWED_SOURCE_PARENT__', reader.PARENT.encode())}
    entries = {name: dict(mode='100644', git_blob_sha1=hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(),
        **reader.pin(raw)) for name, raw in files.items()}
    tree = dict(sha='a' * 40, tree=[dict(path=name, sha=value['git_blob_sha1'], mode=value['mode'], type='blob')
        for name, value in entries.items()])
    producer = dict(schema='english-v6-b24r1-complete-source-producer-v1', role='source', run_head=reader.HEAD,
        run_id=str(reader.RUN), run_attempt=1, run_branch=reader.BRANCH, runtime_workflow=reader.WORKFLOW,
        prepared_parent=reader.PARENT, git_tree=tree['sha'], files=entries, python_files=[], resource_files=sorted(files))
    payloads = {'repository/' + name: raw for name, raw in files.items()}
    payloads['PRODUCER.json'] = reader.canonical(producer)
    lock = lock_fixture(); lock['source_producer_manifest_sha256'] = reader.sha(payloads['PRODUCER.json'])
    return payloads, tree, lock


class Contracts(unittest.TestCase):
    def test_exact_lock_and_only_source_larger_envelope(self):
        lock = lock_fixture()
        reader.validate_lock(lock)
        for role in reader.NAMES:
            changed = copy.deepcopy(lock)
            changed['artifacts'][role]['bytes'] = 128_000_001
            if role == 'source':
                reader.validate_lock(changed)
            else:
                with self.assertRaises(ValueError): reader.validate_lock(changed)
        lock['artifacts']['source']['bytes'] = 512_000_001
        with self.assertRaises(ValueError): reader.validate_lock(lock)

    def test_no_unbound_duplicate_or_wrong_execution_lock(self):
        for key, value in [('provider_counts', None), ('run_head', 'a' * 40), ('run_id', True),
                ('source_assessment_sha256', reader.TRAIN_ASSESSMENT), ('candidate_pt_sha256', '1' * 64)]:
            lock = lock_fixture(); lock[key] = value
            with self.assertRaises(ValueError): reader.validate_lock(lock)
        lock = lock_fixture(); lock['artifacts']['source_producer']['id'] = lock['artifacts']['source']['id']
        with self.assertRaises(ValueError): reader.validate_lock(lock)
        lock = lock_fixture(); del lock['api']['jobs']
        with self.assertRaises(ValueError): reader.validate_lock(lock)

    def test_counts_cannot_promote_unentered_or_erase_empty_return(self):
        for field, value in [('payload_valid', 9), ('failed', 0), ('not_started', 110),
                ('entered_without_return', 1), ('forward_attempted', 8)]:
            counts = counts_fixture(); counts['parakeet'][field] = value
            with self.assertRaises(ValueError): reader.validate_counts(counts)
        counts = counts_fixture(); counts['qwen']['capture_saved'] = 119
        with self.assertRaises(ValueError): reader.validate_counts(counts)

    def test_final_api_requires_exact_bytes_and_final_eight_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); lock = lock_fixture(); values = api_fixture(lock)
            write_api(root, lock, values); reader.final_metadata(root, lock)
            path = root / 'metadata/source_run.json'; path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaises(ValueError): reader.final_metadata(root, lock)
            for role, key, value in [('run', 'status', 'in_progress'), ('run', 'run_attempt', 2),
                    ('tree', 'truncated', True), ('artifacts', 'total_count', 15)]:
                altered = copy.deepcopy(values); altered[role][key] = value
                write_api(root, lock, altered)
                with self.assertRaises(ValueError): reader.final_metadata(root, lock)
            altered = copy.deepcopy(values); altered['jobs']['jobs'][0]['conclusion'] = 'failure'
            write_api(root, lock, altered)
            with self.assertRaises(ValueError): reader.final_metadata(root, lock)

    def test_wrong_job_log_identity_and_extra_commit_path_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); lock = lock_fixture(); values = api_fixture(lock)
            for kind in ('job', 'commit'):
                altered = copy.deepcopy(values)
                if kind == 'job':
                    next(j for j in altered['jobs']['jobs'] if j['name'] == 'readback')['id'] += 1
                else:
                    altered['commit']['files'].append(dict(filename='automation/changed_parser.py'))
                write_api(root, lock, altered)
                with self.assertRaises(ValueError): reader.final_metadata(root, lock)

    def test_zip_crc_digest_paths_and_expanded_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.zip'
            raw = zipped({'original.bin': b'x' * 1000}); path.write_bytes(raw)
            with reader.original(path, reader.pin(raw), 'source') as archive:
                self.assertEqual(archive.read('original.bin'), b'x' * 1000)
            with self.assertRaises(ValueError): reader.original(path, dict(bytes=len(raw), sha256='0' * 64), 'source')
            with patch.object(reader, 'archive_limit', return_value=500):
                with self.assertRaises(ValueError): reader.original(path, reader.pin(raw), 'source')
            for name in ('../escape', '/absolute', 'back\\slash', 'C:/drive', './alias', 'alias//child'):
                raw = zipped({name: b'x'}); path.write_bytes(raw)
                with self.assertRaises(ValueError): reader.original(path, reader.pin(raw), 'source')

    def test_archive_symlinks_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.zip'
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                member = zipfile.ZipInfo('link'); member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, 'target')
            raw = stream.getvalue(); path.write_bytes(raw)
            with self.assertRaises(ValueError): reader.original(path, reader.pin(raw), 'source')
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('same', b'a')
                with self.assertWarns(UserWarning): archive.writestr('same', b'b')
            raw = stream.getvalue(); path.write_bytes(raw)
            with self.assertRaises(ValueError): reader.original(path, reader.pin(raw), 'source')

    def test_original_producer_git_blob_mode_and_complete_members(self):
        files, tree, lock = producer_fixture()
        with zipfile.ZipFile(io.BytesIO(zipped(files))) as archive:
            self.assertEqual(len(reader.producer_facts(archive, tree, lock)['files']), 2)
        for key, value in [('sha', '0' * 40), ('mode', '100755'), ('type', 'tree')]:
            changed = copy.deepcopy(tree); changed['tree'][0][key] = value
            with zipfile.ZipFile(io.BytesIO(zipped(files))) as archive:
                with self.assertRaises(ValueError): reader.producer_facts(archive, changed, lock)
        for mutation in ('change', 'extra', 'missing'):
            changed = dict(files)
            if mutation == 'change': changed['repository/' + reader.WORKFLOW] += b'changed\n'
            elif mutation == 'extra': changed['repository/untracked.py'] = b'import torch\n'
            else: changed.pop('repository/' + reader.WORKFLOW)
            with zipfile.ZipFile(io.BytesIO(zipped(changed))) as archive:
                with self.assertRaises(ValueError): reader.producer_facts(archive, tree, lock)

    def test_all_native_journals_recomputed(self):
        rows, files, source_sha, producer = native_fixture()
        with native_archive(files) as archive:
            self.assertEqual(reader.provider_ledger(archive, rows, 'parakeet', source_sha, producer), counts_fixture()['parakeet'])
        for name in ('FORWARD_RETURNED.json', 'RAW_PROCESS.json', 'PAYLOAD_VALID.json'):
            changed = dict(files); changed.pop('rows/' + rows[0]['id'] + '/' + name)
            with native_archive(changed) as archive:
                with self.assertRaises(ValueError): reader.provider_ledger(archive, rows, 'parakeet', source_sha, producer)

    def test_no_extra_source_row_or_reordered_failure(self):
        rows, files, source_sha, producer = native_fixture()
        changed = dict(files); changed['rows/unregistered/RAW_PROCESS.json'] = b'{}'
        with native_archive(changed) as archive:
            with self.assertRaises(ValueError): reader.provider_ledger(archive, rows, 'parakeet', source_sha, producer)
        rows[0], rows[8] = rows[8], rows[0]
        with native_archive(files) as archive:
            with self.assertRaises(ValueError): reader.provider_ledger(archive, rows, 'parakeet', source_sha, producer)

    def test_native_exit_and_saved_stdout_required(self):
        rows, files, source_sha, producer = native_fixture()
        prefix = 'rows/' + reader.FAILED_ID + '/'
        for name, raw in [('RAW_STDOUT.bin', b'Hey Mero\n'), ('RAW_STDOUT.bin', b''),
                ('RAW_PROCESS.json', reader.canonical(dict(native_exit_code=1, native_return_observed=True, timed_out=False, exception_type=None)))]:
            changed = dict(files); changed[prefix + name] = raw
            with native_archive(changed) as archive:
                with self.assertRaises(ValueError): reader.provider_ledger(archive, rows, 'parakeet', source_sha, producer)

    def test_empty_return_must_be_rejected_by_original_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = dict(id=reader.FAILED_ID, provider='parakeet')
            capture = dict(identity, error='ValueError: EMPTY_NATIVE_TRANSCRIPT', retry_count=0, whole_panel_stop=True)
            for name, raw in [('FAILED.json', reader.canonical(capture)), ('STARTED.json', reader.canonical(identity)),
                    ('RAW_STDOUT.bin', b'\n'), ('RAW_PROCESS.json', b'{}')]:
                (root / name).write_bytes(raw)
            parser = types.SimpleNamespace(PARSER_CONTRACT_SHA256='a' * 64,
                pin=lambda path: dict(path=path.name, **reader.pin(path.read_bytes())),
                validate_capture=Mock(side_effect=ValueError('EMPTY_NATIVE_TRANSCRIPT')))
            reader.reparse_empty_return(parser, root, identity, capture)
            self.assertEqual(parser.validate_capture.call_count, 1)
            for outcome in (None, ValueError('OTHER_FAILURE')):
                parser.validate_capture = Mock(return_value=None, side_effect=outcome)
                with self.assertRaises(ValueError): reader.reparse_empty_return(parser, root, identity, capture)
            (root / 'PAYLOAD_VALID.json').write_bytes(b'{}')
            with self.assertRaises(ValueError): reader.reparse_empty_return(parser, root, identity, capture)

    def test_saved_phone_model_vocabulary_all_floats_and_ctc(self):
        original_phone = importlib.import_module('english_v6_b11_supplement_screen')
        provider = original_phone.policy.PHONE_MODELS[0]
        name, revision = original_phone.pcm_tools.PHONE_MODELS[0]
        model = dict(target_conditioned=False, model=name, revision=revision,
            weights_sha256=original_phone.phones.WEIGHT_SHA[0])
        vocabulary = dict(model=name, revision=revision, pad_token_id=0, vocab={'v' + str(i): i for i in range(392)})
        ids = [1, 1, 0, 2]
        raw = b''.join(struct.pack('<392f', *(1.0 if channel == label else 0.0 for channel in range(392))) for label in ids)
        capture = dict(id='invented-phone', logit_frames=4, vocab_size=392, logits_sha256=reader.sha(raw), raw_phone_tokens=['v1', 'v2'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'logits').mkdir()
            vocab_path = root / 'VOCAB.json'; vocab_path.write_bytes(reader.canonical(vocabulary))
            logits = root / 'logits/invented-phone.f32'; logits.write_bytes(raw)
            self.assertEqual(reader.reparse_phone(original_phone, root, provider, model, capture), 4)
            for key, value in [('weights_sha256', '0' * 64), ('revision', 'wrong'), ('target_conditioned', True)]:
                changed = dict(model); changed[key] = value
                with self.assertRaises(ValueError): reader.reparse_phone(original_phone, root, provider, changed, capture)
            for key, value in [('model', 'wrong'), ('pad_token_id', 1), ('vocab', {**vocabulary['vocab'], 'v1': 0})]:
                changed = dict(vocabulary); changed[key] = value; vocab_path.write_bytes(reader.canonical(changed))
                with self.assertRaises(ValueError): reader.reparse_phone(original_phone, root, provider, model, capture)
            vocab_path.write_bytes(reader.canonical(vocabulary))
            for value in (float('nan'), float('inf'), float('-inf')):
                changed = struct.pack('<f', value) + raw[4:]; logits.write_bytes(changed)
                repinned = dict(capture, logits_sha256=reader.sha(changed))
                with self.assertRaises(ValueError): reader.reparse_phone(original_phone, root, provider, model, repinned)
            logits.write_bytes(raw)
            with self.assertRaises(ValueError):
                reader.reparse_phone(original_phone, root, provider, model, dict(capture, raw_phone_tokens=['fabricated']))
            logits.write_bytes(raw[:-4])
            with self.assertRaises(ValueError):
                reader.reparse_phone(original_phone, root, provider, model, dict(capture, logits_sha256=reader.sha(raw[:-4])))

    def test_network_runtime_and_arbitrary_processes_blocked(self):
        with reader.offline_only():
            for name in ('numpy', 'torch', 'onnxruntime', 'huggingface_hub'):
                with self.assertRaises(ValueError): __import__(name)
            with self.assertRaises(ValueError): socket.socket()
            with self.assertRaises(ValueError): subprocess.run(['true'])
            import ssl
            self.assertTrue(ssl.OPENSSL_VERSION)

    def test_nested_helper_exact_immutable_command_only(self):
        helper = HERE / 'english_v6_b24r1_b24_source_failure.py'
        self.assertEqual(reader.sha(helper.read_bytes()), reader.OLD_READER_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); evidence = root / 'ORIGINAL_B24_STOP'; evidence.mkdir()
            lock = root / 'lock.json'; lock.write_bytes(b'{}')
            command = [sys.executable, '-I', '-S', '-B', str(helper.resolve()), '--verify-worker', str(evidence), str(lock)]
            spawn = Mock(return_value=object())
            with patch('subprocess.Popen', spawn):
                with reader.offline_only() as audit:
                    subprocess.Popen(command, stdout=subprocess.PIPE)
                    self.assertEqual(audit['original_stop_reopens'], 1)
                    for kwargs in ({'shell': True}, {'executable': '/bin/sh'}, {'env': {}}, {'preexec_fn': lambda: None}):
                        with self.assertRaises(ValueError): subprocess.Popen(command, **kwargs)
                    altered = list(command); altered[2] = '-c'
                    with self.assertRaises(ValueError): subprocess.Popen(altered)
                    altered = list(command); altered[4] = str(Path(reader.__file__).resolve())
                    with self.assertRaises(ValueError): subprocess.Popen(altered)
            self.assertEqual(spawn.call_count, 1)

    def test_duplicate_nonfinite_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError): reader.document(raw)

    def test_report_requires_real_nested_reopen_and_grants_nothing(self):
        facts = dict(provider_counts=counts_fixture())
        semantic = dict(nested_original_B24_STOP_fully_reauthenticated=True)
        with patch.object(reader, 'archive_facts', return_value=facts), patch.object(reader, 'semantic_readback', return_value=semantic):
            with self.assertRaises(ValueError): reader.verify_worker(Path('/unused'), lock_fixture(), dict(original_stop_reopens=0))
            report = reader.verify_worker(Path('/unused'), lock_fixture(), dict(original_stop_reopens=1))
        self.assertEqual(report['actual_native_attempts'], 489)
        self.assertEqual(report['valid_native_payloads'], 488)
        self.assertEqual(report['never_started'], 111)
        self.assertIsNone(report['quality_verdict'])
        self.assertIsNone(report['model_quality_claim'])
        self.assertTrue(all(report[k] is False for k in reader.FALSE_PERMISSIONS))


if __name__ == '__main__':
    unittest.main()
