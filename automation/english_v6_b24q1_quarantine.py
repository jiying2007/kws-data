"""B24Q1's explicit successful-empty proof and native-slot accounting.

This is a new orchestration policy. The original B18 valid-payload parser and
all native backends remain unchanged. A proved empty is still invalid and has
no vote; neither this module nor its saved-byte readers execute a model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import unicodedata
import wave

import english_v6_b18_asr_adapter as raw
import english_v6_b18_source_observe as native

HERE = Path(__file__).resolve().parent
PROPOSAL_SHA = 'b5149d29049d5f308efd92a966ab36b0ccb81a9e04fbc64f8ff81e52d5fe7bea'
ORCHESTRATION = HERE / 'english_v6_b24q1_orchestration_contract.json'
ORCHESTRATION_SHA = '75136b683c893e3d6457d93b2f9f74829c711e7b614b6a5b727380faf4f42f49'
EMPTY_SCHEMA = 'english-v6-b24q1-invalid-empty-native-proof-v1'
LEDGER_SCHEMA = 'english-v6-b24q1-native-valid-or-empty-ledger-v1'
EMPTY_STATE = 'INVALID_EMPTY_PROVEN'
EMPTY_REASON = 'INVALID_EMPTY_NATIVE_RETURN_QUARANTINED'
PARAKEET_BINARY = {'path': 'nemo-speech', 'bytes': 2425656,
                   'sha256': 'b38ff5adbe5faba96dea1482e1ba9f9c048d3da07f38b1ca2b092a06c968de61'}
PARAKEET_MODEL = {'bytes': 713975456,
                  'sha256': 'e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e'}
PARAKEET_RUNTIME = {'bytes': 9605101,
                    'sha256': '255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e'}
need, sha, canonical = raw.need, raw.sha, raw.canonical


def equal(left, right):
    """JSON equality which never confuses bool with int or accepts NaN."""
    return canonical(left) == canonical(right)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'DUPLICATE_NATIVE_EVIDENCE_KEY')
        value[key] = item
    return value


def _finite_float(value):
    number = float(value)
    need(math.isfinite(number), 'NONFINITE_NATIVE_JSON_NUMBER')
    return number


def document(path):
    path = Path(path)
    identity = raw.pin(path)
    need(identity['bytes'] <= 4 * 1024 * 1024, 'BOUNDED_NATIVE_JSON_RECEIPT')
    return json.loads(path.read_bytes().decode('utf-8', errors='strict'),
        object_pairs_hook=_unique, parse_float=_finite_float,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('NONFINITE_NATIVE_JSON')))


def orchestration_contract():
    need(raw.pin(ORCHESTRATION)['sha256'] == ORCHESTRATION_SHA,
         'EXACT_APPROVED_B24Q1_ORCHESTRATION_CONTRACT')
    value = document(ORCHESTRATION)
    need(value['experiment_id'] == 'B24Q1' and value['approved_proposal_sha256'] == PROPOSAL_SHA,
         'NEW_EXPERIMENT_AND_EXPLICIT_APPROVED_PROPOSAL')
    return value


def validate_text_model(provider, model, contract):
    """Authenticate the same recorded native identity for valid and empty text."""
    need(provider in raw.TEXT_PROVIDERS, 'EMPTY_RULE_ONLY_FOR_NATIVE_TEXT')
    expected = {'observer': contract['asr_models'][provider],
        'runtime_versions': contract['asr_runtime_versions'][provider],
        'adapter_sha256': contract['asr_adapter_sha256'],
        'original_adapter_sha256': contract['old_file_sha256']['english_seven_voice_asr_once.py'],
        'parser_contract_sha256': raw.PARSER_CONTRACT_SHA256,
        'initialization_model_calls': 0, 'target_conditioned': False,
        'raw_before_validation': True, 'decode_config_unchanged': True}
    if provider == 'parakeet':
        expected.update(native_binary=PARAKEET_BINARY, native_model=PARAKEET_MODEL,
                        native_runtime_release=PARAKEET_RUNTIME)
    need(equal(model, expected), 'EXACT_UNCHANGED_NATIVE_MODEL_RUNTIME_AND_DECODE')
    return model


def complete_input(source, row, binding):
    """Reopen complete original PCM; this proves input identity, not pronunciation."""
    sid = row['id']
    need(type(sid) is str and Path(sid).name == sid and sid not in ('', '.', '..')
         and type(row['samples']) is int and row['samples'] > 0,
         'ONE_COMPLETE_ORIGINAL_INPUT_IDENTITY')
    expected = native.binding(binding['provider'], row, binding['source_manifest_sha256'])
    need(equal(binding, expected), 'EXACT_ORIGINAL_ROW_BINDING')
    need(raw.pin(Path(source) / 'MANIFEST.json')['sha256'] == binding['source_manifest_sha256'],
         'EXACT_COMPLETE_ORIGINAL_SOURCE_MANIFEST')
    path = Path(source) / 'wav' / (sid + '.wav')
    need(raw.pin(path)['sha256'] == row['wav_sha256'], 'ORIGINAL_WHOLE_WAV_SHA')
    with wave.open(str(path), 'rb') as handle:
        need((handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype(),
              handle.getnframes()) == (1, 2, 16000, 'NONE', row['samples']), 'ORIGINAL_FULL_PCM16_GEOMETRY')
        pcm = handle.readframes(row['samples'])
        need(not handle.readframes(1), 'NO_EXTRA_PCM_FRAME')
    need(len(pcm) == row['samples'] * 2 and sha(pcm) == row['pcm_sha256'], 'ORIGINAL_COMPLETE_PCM_SHA')


def empty_text(raw_text):
    """The exact old type/encoding/length/control rules, without inventing a vote."""
    need(type(raw_text) is bytes, 'RAW_UTF8_BYTES_REQUIRED')
    text = raw_text.decode('utf-8', errors='strict')
    need(len(text) <= 4096, 'RAW_DECODED_TEXT_OVERSIZE_BEFORE_NORMALIZATION')
    need(all(ord(ch) in raw.WHITE_SPACE or unicodedata.category(ch) not in ('Cc', 'Cf', 'Cs')
             for ch in text), 'NON_WHITE_SPACE_CONTROL_CHARACTER')
    need(all(ord(ch) in raw.WHITE_SPACE for ch in text), 'ONLY_ACTUAL_ZERO_OR_EXPLICIT_WHITE_SPACE_TEXT')
    return {'bytes': len(raw_text), 'sha256': sha(raw_text), 'decoded_codepoints': len(text)}


def _bound(value, binding, reason):
    need(type(value) is dict and all(k in value and equal(value[k], v) for k, v in binding.items()), reason)


def _regular_text(path):
    pin = raw.pin(path)
    # Four UTF-8 bytes per allowed codepoint is an upper bound, not a new parser.
    need(pin['bytes'] <= 4 * 4096, 'RAW_UTF8_PAYLOAD_OVERSIZE')
    return path.read_bytes(), pin


def _read_pinned(directory, metadata):
    value = raw._read_pinned(directory, metadata)
    need(equal(raw.pin(Path(directory) / metadata['path']), metadata), 'EXACT_NATIVE_PIN_TYPES_AND_BYTES')
    return value


def empty_receipt(source, root, row, binding, contract):
    """Reconstruct a successful-empty proof from originals, never an error name.

    May be called before INVALID_EMPTY.json is written or while independently
    reopening that receipt. Its contents are not inputs to this reconstruction.
    """
    orchestration_contract()
    provider = binding['provider']
    need(provider in raw.TEXT_PROVIDERS, 'PHONE_EMPTY_NATIVE_PAYLOAD_IS_FATAL')
    root = Path(root)
    directory = root / 'rows' / row['id']
    complete_input(source, row, binding)
    validate_text_model(provider, document(root / 'MODEL.json'), contract)
    started = document(directory / 'FORWARD_STARTED.json')
    returned = document(directory / 'FORWARD_RETURNED.json')
    need(equal(document(directory / 'STARTED.json'), binding), 'EXACT_ORIGINAL_INPUT_START')
    _bound(started, binding, 'EMPTY_NATIVE_ATTEMPT_SOURCE_BINDING')
    _bound(returned, binding, 'EMPTY_NATIVE_RETURN_SOURCE_BINDING')
    need(type(started.get('attempt')) is int and started['attempt'] == 1
         and returned.get('native_return_observed') is True, 'ONE_OBSERVED_NATIVE_RETURN')
    forbidden = ('PAYLOAD_VALID.json', 'CAPTURE.json', 'FAILED.json')
    need(not any((directory / name).exists() for name in forbidden), 'EMPTY_VALID_FAILED_RECEIPTS_ARE_EXCLUSIVE')
    files = {'STARTED.json', 'FORWARD_STARTED.json', 'FORWARD_RETURNED.json'}
    evidence = {'model': raw.pin(root / 'MODEL.json'),
                'input_started': raw.pin(directory / 'STARTED.json'),
                'forward_started': raw.pin(directory / 'FORWARD_STARTED.json'),
                'forward_returned': raw.pin(directory / 'FORWARD_RETURNED.json')}
    if provider == 'parakeet':
        files.update(('RAW_PROCESS.json', 'NATIVE_COMMAND.json', 'RAW_STDOUT.bin', 'RAW_STDERR.bin'))
        payload = document(directory / 'RAW_PROCESS.json')
        _bound(payload, binding, 'RAW_PROCESS_ORIGINAL_SOURCE_BINDING')
        need(set(payload) == set(binding) | {'schema', 'command', 'stdout', 'stderr', 'process_started',
             'native_exit_code', 'native_return_observed', 'timed_out', 'exception_type'},
             'COMPLETE_EXACT_ORIGINAL_NATIVE_PROCESS_FIELDS')
        need(payload.get('schema') == raw.RAW_SCHEMA and payload.get('process_started') is True
             and payload.get('native_return_observed') is True
             and type(payload.get('native_exit_code')) is int and payload['native_exit_code'] == 0
             and payload.get('timed_out') is False and payload.get('exception_type') is None
             and equal(returned, dict(binding, native_return_observed=True, native_exit_code=0)),
             'SUCCESSFUL_EXIT_ZERO_NATIVE_PROCESS_WITHOUT_TIMEOUT_OR_EXCEPTION')
        command_raw = _read_pinned(directory, payload['command'])
        command = document(directory / 'NATIVE_COMMAND.json')
        need(payload['command']['path'] == 'NATIVE_COMMAND.json'
             and started['native_command_sha256'] == sha(command_raw), 'EXACT_NATIVE_COMMAND_RECEIPT')
        _bound(command, binding, 'COMMAND_ORIGINAL_SOURCE_BINDING')
        argv = command['command']
        need(type(argv) is list and len(argv) == 7 and all(type(x) is str for x in argv)
             and argv[1] == 'transcribe' and argv[3] == '--model' and argv[5:] == ['--device', 'cpu'],
             'UNCHANGED_NATIVE_COMMAND_ARGUMENTS')
        assets = Path(argv[4]).parent
        need(assets.is_absolute() and assets.name == 'b22-asr-parakeet'
             and argv[4] == str(assets / 'parakeet-tdt-0.6b-v3.q8_0.gguf')
             and argv[0] == str(assets / 'extract/nemo-speech-0.3.0-linux-x86_64-cpu/bin/nemo-speech')
             and argv[2] == str(assets.parent / 'source/wav' / (row['id'] + '.wav'))
             and all('..' not in Path(argv[i]).parts for i in (0, 2, 4))
             and Path(command['cwd']).is_absolute()
             and Path(command['cwd']).parts[-2:] == ('rows', row['id'])
             and '..' not in Path(command['cwd']).parts,
             'EXACT_FROZEN_RUNTIME_AND_WHOLE_ORIGINAL_PATH_NO_FILTER')
        need(command['schema'] == raw.RAW_SCHEMA and command['command_sha256'] == sha(canonical(argv))
             and type(command['timeout_seconds']) is int and command['timeout_seconds'] == 180
             and command['stdout_transform'] is command['stderr_transform'] is None
             and equal(command['environment'], {'PATH': '/usr/bin:/bin', 'HOME': command['cwd'], 'LANG': 'C.UTF-8',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                 'NEMO_SPEECH_MODEL_DIR': str(assets), 'OMP_NUM_THREADS': '4'}),
             'UNCHANGED_NATIVE_RUNTIME_ENVIRONMENT_AND_NO_TEXT_TRANSFORMATION')
        need(payload['stdout']['path'] == 'RAW_STDOUT.bin' and payload['stderr']['path'] == 'RAW_STDERR.bin',
             'BOTH_COMPLETE_NATIVE_OUTPUT_STREAMS')
        _read_pinned(directory, payload['stderr'])
        text, text_pin = _regular_text(directory / 'RAW_STDOUT.bin')
        need(equal(text_pin, payload['stdout']), 'ORIGINAL_NATIVE_STDOUT_BYTES')
        evidence.update(payload=raw.pin(directory / 'RAW_PROCESS.json'), command=payload['command'],
                        stdout=payload['stdout'], stderr=payload['stderr'])
    else:
        files.update(('RAW_RETURN.json', 'RAW_TEXT.utf8'))
        payload = document(directory / 'RAW_RETURN.json')
        _bound(payload, binding, 'RAW_PYTHON_ORIGINAL_SOURCE_BINDING')
        need(payload.get('schema') == raw.RAW_SCHEMA and payload.get('native_shape_valid') is True
             and equal(started, dict(binding, attempt=1))
             and equal(returned, dict(binding, native_return_observed=True)), 'COMPLETE_ORIGINAL_PYTHON_RETURN')
        if provider == 'qwen':
            need(payload.get('container_type') == 'list' and type(payload.get('returned_text_fields')) is list
                 and len(payload['returned_text_fields']) == 1, 'QWEN_ORIGINAL_EXACT_SINGLE_NATIVE_RESULT')
            field = payload['returned_text_fields'][0]
        else:
            need(provider == 'whisper' and payload.get('container_type') == 'dict'
                 and payload.get('text_field_present') is True, 'WHISPER_ORIGINAL_PRESENT_NATIVE_TEXT_FIELD')
            field = payload['returned_text_field']
        need(type(field) is dict and set(field) == {'python_type', 'value'}
             and field['python_type'] == 'str' and type(field['value']) is str, 'ORIGINAL_NATIVE_STRING_TYPE')
        text, text_pin = _regular_text(directory / 'RAW_TEXT.utf8')
        need(text == field['value'].encode('utf-8', errors='strict'), 'LOSSLESS_NATIVE_ENVELOPE_TEXT_BYTES')
        evidence.update(payload=raw.pin(directory / 'RAW_RETURN.json'), utf8_text=text_pin)
    actual_files = set()
    for item in directory.iterdir():
        need(item.is_file() and not item.is_symlink(), 'REGULAR_COMPLETE_EMPTY_ROW_FILES_ONLY')
        actual_files.add(item.name)
    need(actual_files in (files, files | {'INVALID_EMPTY.json'}), 'EXACT_EMPTY_ROW_FILE_CLOSURE')
    text_identity = empty_text(text)
    return dict(binding, schema=EMPTY_SCHEMA, status=EMPTY_STATE, payload_valid=False,
        source_admission_eligible=False, observed_class=None, tier='unknown', vote=None,
        reason=EMPTY_REASON, training_allowed=False, retry_count=0, whole_panel_stop=False,
        complete_original_input_verified=True, native_return_observed=True,
        parser_contract_sha256=raw.PARSER_CONTRACT_SHA256,
        orchestration_contract_sha256=ORCHESTRATION_SHA, raw_evidence=evidence,
        original_empty_text=text_identity)


def verify_empty(source, root, row, binding, contract):
    expected = empty_receipt(source, root, row, binding, contract)
    need(equal(document(Path(root) / 'rows' / row['id'] / 'INVALID_EMPTY.json'), expected),
         'SAVED_EMPTY_RECEIPT_INDEPENDENTLY_RECONSTRUCTED_FROM_ORIGINAL_RETURN')
    return expected


def row_ledger(root, rows):
    ledger = native.row_ledger(Path(root), rows)
    for item in ledger:
        path = Path(root) / 'rows' / item['id'] / 'INVALID_EMPTY.json'
        item['invalid_empty'] = path.is_file()
        if item['invalid_empty']:
            item['invalid_empty_sha256'] = raw.pin(path)['sha256']
        valid = item['payload_valid'] and item['capture_saved'] and not item['invalid_empty'] and not item['failed']
        empty = item['invalid_empty'] and not item['payload_valid'] and not item['capture_saved'] and not item['failed']
        item['state'] = 'VALID_NONEMPTY' if valid else EMPTY_STATE if empty else 'FATAL' if item['failed'] else 'UNKNOWN'
    return ledger


def ledger_counts(ledger):
    counts = native.ledger_counts(ledger)
    counts['invalid_empty'] = sum(item['invalid_empty'] for item in ledger)
    counts['completed_native_slots'] = sum(item['state'] in ('VALID_NONEMPTY', EMPTY_STATE) for item in ledger)
    counts['unknown'] = sum(item['state'] == 'UNKNOWN' for item in ledger)
    return counts


def complete_ledger(ledger, count):
    counts = ledger_counts(ledger)
    return len(ledger) == count and counts['failed'] == counts['unknown'] == 0 and all(
        counts[key] == count for key in ('input_started', 'forward_attempted', 'native_returned',
            'raw_payload_saved', 'completed_native_slots')) and all(
        item['state'] in ('VALID_NONEMPTY', EMPTY_STATE) for item in ledger)


def ledger_document(root, rows, provider, source_sha):
    ledger = row_ledger(root, rows)
    return dict(schema=LEDGER_SCHEMA, provider=provider, source_manifest_sha256=source_sha,
        orchestration_contract_sha256=ORCHESTRATION_SHA, denominator=len(rows), rows=ledger,
        counts=ledger_counts(ledger),
        attempted_ids=[r['id'] for r in ledger if r['forward_attempted']],
        returned_ids=[r['id'] for r in ledger if r['native_returned']],
        payload_valid_ids=[r['id'] for r in ledger if r['payload_valid']],
        invalid_empty_ids=[r['id'] for r in ledger if r['invalid_empty']])


def slot_counts(ledger):
    counts = ledger_counts(ledger)
    return dict(planned=len(ledger), attempted=counts['forward_attempted'], returned=counts['native_returned'],
        valid=counts['payload_valid'], invalid_empty=counts['invalid_empty'],
        fatal=counts['failed'], unknown=counts['unknown'])
