"""Invented stdlib raw receipts only: no native process, audio decoding or model."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24q1_quarantine as q
import english_v6_b18_asr_adapter as raw
import english_v6_b18_source_observe as native


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False) + '\n').encode())


def text_model(provider, contract):
    result = dict(observer=contract['asr_models'][provider], runtime_versions=contract['asr_runtime_versions'][provider],
        adapter_sha256=contract['asr_adapter_sha256'],
        original_adapter_sha256=contract['old_file_sha256']['english_seven_voice_asr_once.py'],
        parser_contract_sha256=raw.PARSER_CONTRACT_SHA256, initialization_model_calls=0,
        target_conditioned=False, raw_before_validation=True, decode_config_unchanged=True)
    if provider == 'parakeet':
        result.update(native_binary=copy.deepcopy(q.PARAKEET_BINARY), native_model=copy.deepcopy(q.PARAKEET_MODEL),
                      native_runtime_release=copy.deepcopy(q.PARAKEET_RUNTIME))
    return result


def original_row(source, sid='b24q1-primary-127-mero-fast', samples=32, value=100):
    (source / 'wav').mkdir(parents=True, exist_ok=True)
    pcm = struct.pack('<h', value) * samples
    wav = source / 'wav' / (sid + '.wav')
    with wave.open(str(wav), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed')); stream.writeframes(pcm)
    return dict(id=sid, samples=samples, pcm_sha256=raw.sha(pcm), wav_sha256=raw.sha(wav.read_bytes()))


def native_text_files(root, source, row, provider, value, contract, *, valid=False):
    """Save an explicitly invented native envelope; never launch a process/model."""
    directory = root / 'rows' / row['id']; directory.mkdir(parents=True)
    start = native.binding(provider, row, raw.sha((source / 'MANIFEST.json').read_bytes()))
    raw.record(directory / 'STARTED.json', start)
    if not (root / 'MODEL.json').exists(): raw.record(root / 'MODEL.json', text_model(provider, contract))
    if provider != 'parakeet':
        result = [SimpleNamespace(text=value)] if provider == 'qwen' else {'text': value}
        try:
            capture = raw.capture_python(provider, directory, start, lambda: result)
        except ValueError as exc:
            if str(exc) != 'EMPTY_NATIVE_TRANSCRIPT': raise
            capture = None
    else:
        assets = source.parent / 'b22-asr-parakeet'
        command = [str(assets / 'extract/nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech'),
            'transcribe', str(source / 'wav' / (row['id'] + '.wav')), '--model',
            str(assets / 'parakeet-tdt-0.6b-v3.q8_0.gguf'), '--device', 'cpu']
        command_pin = raw.record(directory / 'NATIVE_COMMAND.json', dict(start, schema=raw.RAW_SCHEMA,
            command=command, command_sha256=raw.sha(raw.canonical(command)), cwd=str(directory), timeout_seconds=180,
            stdout_transform=None, stderr_transform=None,
            environment={'PATH':'/usr/bin:/bin','HOME':str(directory),'LANG':'C.UTF-8',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','NEMO_SPEECH_MODEL_DIR':str(assets),'OMP_NUM_THREADS':'4'}))
        raw.record(directory / 'FORWARD_STARTED.json', dict(start, attempt=1, native_command_sha256=command_pin['sha256']))
        raw.record(directory / 'FORWARD_RETURNED.json', dict(start, native_return_observed=True, native_exit_code=0))
        stdout = raw.put(directory / 'RAW_STDOUT.bin', value.encode() if type(value) is str else value)
        stderr = raw.put(directory / 'RAW_STDERR.bin', b'invented stderr\x1b\xff\n')
        payload = raw.record(directory / 'RAW_PROCESS.json', dict(start, schema=raw.RAW_SCHEMA,
            command=command_pin, stdout=stdout, stderr=stderr, process_started=True, native_exit_code=0,
            native_return_observed=True, timed_out=False, exception_type=None))
        capture = raw._payload_valid(directory, start, payload,
                    raw.normalize_raw((directory/'RAW_STDOUT.bin').read_bytes())) if valid else None
    if valid:
        raw.record(directory / 'CAPTURE.json', dict(start, **capture, status='success', input_coverage_complete=True))
    return directory, start


class EmptyProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = native.source_check()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.source = self.base / 'source'; self.source.mkdir()
        (self.source / 'MANIFEST.json').write_bytes(b'{"invented":true}\n')
        self.row = original_row(self.source)
        self.root = self.base / 'observer'; self.root.mkdir()

    def fixture(self, provider='parakeet', text=b'\n'):
        self.provider = provider
        if provider != 'parakeet' and isinstance(text, bytes): text = text.decode()
        self.directory, self.start = native_text_files(self.root, self.source, self.row, provider, text, self.contract)
        return self.proof()

    def proof(self):
        return q.empty_receipt(self.source, self.root, self.row, self.start, self.contract)

    def rewrite(self, name, change, *, root=False):
        path = (self.root if root else self.directory) / name
        value = json.loads(path.read_bytes()); change(value); write_json(path, value)

    def command(self, change):
        self.rewrite('NATIVE_COMMAND.json', change)
        pin = raw.pin(self.directory / 'NATIVE_COMMAND.json')
        self.rewrite('RAW_PROCESS.json', lambda value: value.update(command=pin))
        self.rewrite('FORWARD_STARTED.json', lambda value: value.update(native_command_sha256=pin['sha256']))

    def test_exact_approved_contract_and_zero_whitespace_boundary(self):
        self.assertEqual(q.orchestration_contract()['experiment_id'], 'B24Q1')
        for data in (b'', b'\n', b' \t\r\n', ''.join(map(chr, sorted(raw.WHITE_SPACE))).encode()):
            with self.subTest(data=data):
                self.assertEqual(q.empty_text(data)['bytes'], len(data))
        for data in (b'x', b' \nx\n', b'\xff', b'\x1b', b'\0', b'\x1c', '\ufeff'.encode(),
                     '\u200b'.encode(), b' '*4097, ' ', None, 0):
            with self.subTest(data=str(data)[:40]), self.assertRaises((ValueError, UnicodeError)):
                q.empty_text(data)
        self.assertEqual(q.empty_text(b' '*4096)['decoded_codepoints'], 4096)

    def test_real_zero_byte_file_is_distinct_from_missing_stdout(self):
        receipt = self.fixture(text=b'')
        self.assertEqual(receipt['original_empty_text']['bytes'], 0)
        (self.directory / 'RAW_STDOUT.bin').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.proof()

    def test_process_proof_is_invalid_without_any_vote_or_capture(self):
        receipt = self.fixture()
        self.assertEqual(receipt['status'], q.EMPTY_STATE)
        self.assertIs(receipt['payload_valid'], False); self.assertIsNone(receipt['observed_class'])
        self.assertIsNone(receipt['vote']); self.assertEqual(receipt['tier'], 'unknown')
        self.assertFalse((self.directory/'CAPTURE.json').exists())
        self.assertFalse((self.directory/'PAYLOAD_VALID.json').exists())
        raw.record(self.directory/'INVALID_EMPTY.json', receipt)
        self.assertEqual(q.verify_empty(self.source,self.root,self.row,self.start,self.contract), receipt)
        self.rewrite('INVALID_EMPTY.json', lambda value: value.update(vote='ordinary'))
        with self.assertRaises(ValueError): q.verify_empty(self.source,self.root,self.row,self.start,self.contract)

    def test_success_exit_timeout_and_exception_are_independent_required_facts(self):
        self.fixture(); path = self.directory/'RAW_PROCESS.json'; saved = path.read_bytes()
        for key, value in (('native_exit_code',1),('native_exit_code',False),('timed_out',True),
                           ('exception_type','ValueError'),('process_started',False),('native_return_observed',False)):
            with self.subTest(key=key,value=value):
                self.rewrite('RAW_PROCESS.json', lambda item: item.update({key:value}))
                with self.assertRaises(ValueError): self.proof()
                path.write_bytes(saved)

    def test_absent_exception_field_is_not_proof_of_no_native_exception(self):
        self.fixture();path=self.directory/'RAW_PROCESS.json';saved=path.read_bytes()
        for change in (lambda value:value.pop('exception_type'),lambda value:value.update(unrecorded_field=False)):
            with self.subTest(change=repr(change)):
                self.rewrite('RAW_PROCESS.json',change)
                # Proof reconstructs new raw pins directly, so this cannot pass
                # merely because the byte digest has been recomputed consistently.
                with self.assertRaisesRegex(ValueError,'COMPLETE_EXACT_ORIGINAL_NATIVE_PROCESS_FIELDS'):
                    self.proof()
                path.write_bytes(saved)

    def test_complete_stderr_and_native_return_are_required(self):
        self.fixture()
        for name in ('RAW_STDERR.bin','FORWARD_RETURNED.json','FORWARD_STARTED.json','STARTED.json'):
            path=self.directory/name; saved=path.read_bytes(); path.unlink()
            with self.subTest(name=name), self.assertRaises((ValueError,FileNotFoundError)): self.proof()
            path.write_bytes(saved)

    def test_process_command_environment_and_no_filter_are_reproved_after_repinning(self):
        self.fixture()
        saved={p.name:p.read_bytes() for p in self.directory.iterdir()}
        changes=(lambda v:v.update(timeout_seconds=True), lambda v:v.update(stdout_transform='strip'),
            lambda v:v['environment'].update(OMP_NUM_THREADS='8'), lambda v:v['command'].__setitem__(2,'/tmp/cropped.wav'),
            lambda v:v['command'].__setitem__(5,'--prompt'), lambda v:v.update(cwd='/tmp/elsewhere'),
            lambda v:v['command'].__setitem__(0,'/tmp/nemo-speech'))
        for change in changes:
            with self.subTest(change=repr(change)):
                self.command(change)
                with self.assertRaises(ValueError): self.proof()
                for name,data in saved.items(): (self.directory/name).write_bytes(data)

    def test_full_input_pcm_binding_is_required_not_duration_or_rms_claim(self):
        self.fixture()
        wav=self.source/'wav'/(self.row['id']+'.wav'); saved=wav.read_bytes()
        wav.write_bytes(saved[:-2]+b'\0\0')
        with self.assertRaises(ValueError): self.proof()
        wav.write_bytes(saved)
        row=dict(self.row,samples=self.row['samples']-1)
        with self.assertRaises(ValueError): q.empty_receipt(self.source,self.root,row,self.start,self.contract)
        start=dict(self.start, source_pcm_sha256='0'*64)
        with self.assertRaises(ValueError): q.empty_receipt(self.source,self.root,self.row,start,self.contract)
        (self.source/'MANIFEST.json').write_bytes(b'{"different_registration":true}\n')
        with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_ORIGINAL_SOURCE_MANIFEST'):self.proof()

    def test_same_exact_model_runtime_and_native_binary_required(self):
        self.fixture(); path=self.root/'MODEL.json'; saved=path.read_bytes()
        changes=(lambda v:v['native_binary'].update(sha256='0'*64),lambda v:v['native_model'].update(bytes=1),
            lambda v:v['native_runtime_release'].update(sha256='0'*64), lambda v:v.update(initialization_model_calls=True),
            lambda v:v.update(target_conditioned=True),lambda v:v['observer']['decode_config'].update(prompt='hey'))
        for change in changes:
            with self.subTest(change=repr(change)):
                self.rewrite('MODEL.json',change,root=True)
                with self.assertRaises(ValueError):self.proof()
                path.write_bytes(saved)

    def test_empty_valid_and_fatal_receipts_are_mutually_exclusive(self):
        self.fixture()
        for name in ('CAPTURE.json','PAYLOAD_VALID.json','FAILED.json','unplanned.txt'):
            path=self.directory/name;path.write_bytes(b'{}')
            with self.subTest(name=name), self.assertRaises(ValueError): self.proof()
            path.unlink()

    def test_invalid_utf8_controls_oversize_and_nonempty_stay_fatal_after_raw_repin(self):
        self.fixture()
        for data in (b'\xff',b'\0',b'\x1b',b' '*4097,b'heard text',b'\xef\xbb\xbf'):
            with self.subTest(data=data[:12]):
                (self.directory/'RAW_STDOUT.bin').write_bytes(data)
                self.rewrite('RAW_PROCESS.json',lambda v:v.update(stdout=raw.pin(self.directory/'RAW_STDOUT.bin')))
                with self.assertRaises((ValueError,UnicodeError)):self.proof()

    def test_duplicate_json_unknown_numbers_and_symlinks_rejected(self):
        self.fixture();path=self.directory/'RAW_PROCESS.json';saved=path.read_bytes()
        for data in (b'{"native_exit_code":1,"native_exit_code":0}',b'{"native_exit_code":NaN}'):
            path.write_bytes(data)
            with self.assertRaises(ValueError):self.proof()
        path.write_bytes(saved)
        path=self.directory/'RAW_STDERR.bin';saved=path.read_bytes();path.unlink()
        other=self.base/'same-bytes';other.write_bytes(saved);path.symlink_to(other)
        with self.assertRaises(ValueError):self.proof()

    def test_qwen_and_whisper_actual_single_string_empty(self):
        for provider in ('qwen','whisper'):
            self.root=self.base/provider;self.root.mkdir()
            receipt=self.fixture(provider,' \n')
            self.assertEqual(receipt['status'],q.EMPTY_STATE)
            self.assertEqual(receipt['original_empty_text']['bytes'],2)
            self.rewrite('RAW_RETURN.json',lambda v:v.update(container_type='NoneType'))
            with self.assertRaises(ValueError):self.proof()

    def test_overflowing_json_numbers_in_any_raw_envelope_or_receipt_are_fatal(self):
        for provider in ('parakeet','qwen','whisper'):
            self.root=self.base/provider;self.root.mkdir()
            receipt=self.fixture(provider,'\n')
            raw.record(self.directory/'INVALID_EMPTY.json',receipt)
            payload='RAW_PROCESS.json' if provider=='parakeet' else 'RAW_RETURN.json'
            for name in (payload,'INVALID_EMPTY.json'):
                path=self.directory/name;saved=path.read_bytes()
                for numeral in (b'1e10000',b'-1e10000'):
                    with self.subTest(provider=provider,name=name,numeral=numeral):
                        path.write_bytes(saved.rstrip()[:-1]+b', "unused_nested": {"value": '+numeral+b'}}\n')
                        with self.assertRaisesRegex(ValueError,'NONFINITE_NATIVE_JSON_NUMBER'):
                            q.verify_empty(self.source,self.root,self.row,self.start,self.contract)
                        path.write_bytes(saved)

    def test_qwen_missing_multiple_wrong_type_native_fields_are_not_empty_text(self):
        self.fixture('qwen','\n');path=self.directory/'RAW_RETURN.json';saved=path.read_bytes()
        for fields in ([],[{'python_type':'str','value':''}]*2,[{'python_type':'NoneType','value':None}],
                       [{'python_type':'str','value':None}],[{'python_type':'str','value':'\n','invented':True}]):
            self.rewrite('RAW_RETURN.json',lambda v:v.update(returned_text_fields=fields))
            with self.subTest(fields=fields),self.assertRaises(ValueError):self.proof()
            path.write_bytes(saved)

    def test_whisper_missing_null_or_changed_utf8_is_fatal(self):
        self.fixture('whisper','\n');path=self.directory/'RAW_RETURN.json';saved=path.read_bytes()
        for change in (lambda v:v.update(text_field_present=False),
                       lambda v:v.update(returned_text_field={'python_type':'NoneType','value':None}),
                       lambda v:v.update(returned_text_field={'python_type':'str','value':' '})):
            self.rewrite('RAW_RETURN.json',change)
            with self.assertRaises(ValueError):self.proof()
            path.write_bytes(saved)

    def test_phone_empty_payload_never_qualifies_for_text_rule(self):
        self.fixture()
        for provider in ('xlsr','lv60'):
            with self.assertRaisesRegex(ValueError,'PHONE_EMPTY_NATIVE_PAYLOAD_IS_FATAL'):
                q.empty_receipt(self.source,self.root,self.row,dict(self.start,provider=provider),self.contract)

    def test_ledger_counts_empty_separately_and_unknown_never_completes(self):
        receipt=self.fixture();raw.record(self.directory/'INVALID_EMPTY.json',receipt)
        ledger=q.row_ledger(self.root,[self.row]);counts=q.ledger_counts(ledger)
        self.assertEqual(counts['invalid_empty'],1);self.assertEqual(counts['payload_valid'],0)
        self.assertTrue(q.complete_ledger(ledger,1));self.assertFalse(q.complete_ledger(ledger,2))
        missing=dict(self.row,id='b24q1-primary-unstarted')
        self.assertFalse(q.complete_ledger(q.row_ledger(self.root,[self.row,missing]),2))
        raw.record(self.directory/'FAILED.json',dict(self.start,status='failed'))
        self.assertFalse(q.complete_ledger(q.row_ledger(self.root,[self.row]),1))


if __name__=='__main__': unittest.main()
