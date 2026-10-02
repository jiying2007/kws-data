"""Validate retained primary-run evidence before making a scoring envelope.

This is a bounded, stdlib-only consistency check, not proof that inference ran.
The caller must supply the plan SHA from its independently retained freeze, not
derive a new trusted SHA from this bundle. No models/tokenizers/audio are loaded.
"""
import hashlib
import json
import math
import os
import stat

from .adapters import (QWEN, SENSE, require_execution_declaration,
                       validate_decoder_manifest)
from .architecture import canonical_sha, qwen06_schema, sensevoice_schema
from .assets import MODELS
from .decoding import bounded_text, qwen_transcript, sense_transcript, token_ids
from .results import FLAGS, SHA, _strict_json


OUTCOME_FIELDS = {'opaque_id', 'wav_sha256', 'status', 'raw_text',
                  'completeness', 'quality_flags', 'execution_receipt_sha256'}
ROW_FIELDS = ('status', 'raw_text', 'completeness', 'quality_flags')
RECEIPT_FIELDS = (OUTCOME_FIELDS - {'execution_receipt_sha256'}) | {
    'schema', 'model', 'input_binding_sha256', 'decoder_manifest_sha256',
    'attempt_started', 'model_forward_completed', 'decoder_evidence_sha256',
    'execution_contract_sha256'}
SIDECAR_FIELDS = {'schema', 'model', 'opaque_id', 'wav_sha256',
                  'input_binding_sha256', 'outcome', 'wall_seconds', 'evidence'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields, label):
    _require(type(value) is dict and set(value) == set(fields),
             'Malformed ' + label)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _equal(actual, expected, label):
    # Ordinary Python equality would allow True in place of the integer 1.
    _require(_bytes(actual) == _bytes(expected), label + ' differs')


def _sha(value, label):
    _require(type(value) is str and SHA.fullmatch(value), 'Invalid ' + label)


def _read(directory_fd, name, inventory, limit=8 * 1024 * 1024):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                 dir_fd=directory_fd)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
                 'Nonregular or oversized bundle file: ' + name)
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    _require(len(raw) <= limit and len(raw) == before.st_size and
             (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             'Bundle file changed while reading: ' + name)
    inventory[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw, _strict_json(raw)


def _plan(plan):
    _fields(plan, {'schema', 'model_id', 'model_root', 'pcm_root', 'asset_lock',
        'asset_lock_sha256', 'source_lock', 'source_lock_sha256', 'contract',
        'contract_sha256', 'decoder_manifest_sha256', 'clip_seconds'}, 'primary plan')
    _require(plan['schema'] == 'asr-primary-run-plan-v1' and
             plan['model_id'] in (QWEN, SENSE), 'Unexpected primary plan')
    _require(type(plan['clip_seconds']) is int and 1 <= plan['clip_seconds'] <= 900,
             'Invalid per-clip deadline')
    for field in ('asset_lock_sha256', 'source_lock_sha256', 'contract_sha256',
                  'decoder_manifest_sha256'):
        _sha(plan[field], field)
    for field in ('asset_lock', 'source_lock', 'contract'):
        _require(canonical_sha(plan[field]) == plan[field + '_sha256'],
                 'Plan embedded ' + field + ' hash differs')
    schema = qwen06_schema() if plan['model_id'] == QWEN else sensevoice_schema()
    require_execution_declaration(plan['contract'], plan['contract_sha256'],
        plan['model_id'], plan['asset_lock_sha256'], plan['source_lock_sha256'],
        canonical_sha(schema))
    _require(plan['contract']['decoder_manifest_sha256'] == plan['decoder_manifest_sha256'],
             'Plan/contract decoder input identity differs')


def _outcome(row, oid, inputs):
    _fields(row, OUTCOME_FIELDS, 'model outcome')
    _require(row['opaque_id'] == oid, 'Outcome order/membership differs from plan')
    _require(row['wav_sha256'] == inputs[oid]['descriptor']['wav']['sha256'],
             'Outcome WAV differs from decoder input')
    status = row['status']
    _require(type(status) is str and status in {'success', 'error', 'timeout', 'not_run'},
             'Invalid outcome status')
    _require(type(row['completeness']) is str and
             row['completeness'] in {'complete', 'incomplete', 'unknown'},
             'Invalid outcome completeness')
    flags = row['quality_flags']
    _require(type(flags) is list and
             all(type(flag) is str and flag in FLAGS for flag in flags) and
             len(set(flags)) == len(flags), 'Invalid outcome quality flags')
    if status == 'success':
        bounded_text(row['raw_text'])
        _require(len(row['raw_text']) <= 1024, 'Oversized transcript for scoring envelope')
    else:
        _require(row['raw_text'] is None and row['completeness'] == 'unknown' and not flags,
                 'Failed/not-run outcome contains invented transcript evidence')
    if status == 'not_run':
        _require(row['execution_receipt_sha256'] is None,
                 'Not-run outcome contains invented execution evidence')
    else:
        _sha(row['execution_receipt_sha256'], 'execution receipt SHA')


def _qwen(evidence, descriptor):
    _fields(evidence, {'token_ids', 'decoded_with_special_tokens',
        'decoded_skip_special_tokens', 'metadata', 'input_binding', 'prompt_context',
        'force_language', 'processor_tensor_evidence', 'processor_input_layout',
        'transcript_repair_applied'}, 'Qwen decoder evidence')
    _equal(evidence['input_binding'], descriptor, 'Qwen input binding')
    _require(evidence['prompt_context'] == '' and evidence['force_language'] is None and
             evidence['transcript_repair_applied'] is False,
             'Qwen context/forced-language/repair declaration differs')
    bounded_text(evidence['decoded_with_special_tokens'])
    row, metadata = qwen_transcript(evidence['decoded_skip_special_tokens'], evidence['token_ids'])
    _equal(evidence['metadata'], metadata, 'Qwen extracted metadata')
    tensors, layouts = evidence['processor_tensor_evidence'], evidence['processor_input_layout']
    _require(type(tensors) is dict and tensors and type(layouts) is dict and
             set(tensors) == set(layouts) and 'input_ids' in tensors,
             'Missing/inconsistent Qwen processor evidence')
    for name, tensor in tensors.items():
        _fields(tensor, {'shape', 'dtype', 'sha256'}, 'processor tensor')
        _require(type(tensor['shape']) is list and
                 all(type(n) is int and n >= 0 for n in tensor['shape']) and
                 tensor['dtype'] in {'torch.float32', 'torch.float64', 'torch.int64',
                     'torch.int32', 'torch.int16', 'torch.int8', 'torch.uint8', 'torch.bool'},
                 'Invalid processor tensor descriptor')
        _sha(tensor['sha256'], 'processor tensor SHA')
        _fields(layouts[name], {'original_stride', 'contiguous_copy'}, 'processor input layout')
        stride = layouts[name]['original_stride']
        _require(type(stride) is list and len(stride) == len(tensor['shape']) and
                 all(type(n) is int and n >= 0 for n in stride) and
                 type(layouts[name]['contiguous_copy']) is bool,
                 'Invalid processor layout descriptor')
    return row


def _sense(evidence, descriptor, oid):
    _fields(evidence, {'raw_result', 'token_evidence', 'metadata', 'input_binding',
        'use_itn', 'rich_postprocess_applied', 'completeness_scope'}, 'SenseVoice decoder evidence')
    _equal(evidence['input_binding'], descriptor, 'SenseVoice input binding')
    _require(evidence['use_itn'] is False and evidence['rich_postprocess_applied'] is False,
             'SenseVoice normalization/postprocess declaration differs')
    _require(evidence['completeness_scope'] ==
             'Returned full-input CTC inference; not acoustic transcript completeness',
             'SenseVoice completeness scope differs')
    token = evidence['token_evidence']
    _fields(token, {'token_ids', 'decoded', 'token_scope'}, 'SenseVoice token evidence')
    token_ids(token['token_ids'], allow_empty=True)
    _require(token['token_scope'] ==
             'Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits',
             'SenseVoice token scope differs')
    raw = evidence['raw_result']
    _require(type(raw) is list and len(raw) == 2 and type(raw[0]) is list and
             len(raw[0]) == 1 and type(raw[1]) is dict, 'Malformed SenseVoice raw result')
    _equal(raw[0][0], {'key': oid, 'text': token['decoded']}, 'SenseVoice raw result key/text')
    row, metadata = sense_transcript(token['decoded'])
    _equal(evidence['metadata'], metadata, 'SenseVoice extracted metadata')
    return row


def validate_primary_bundle(directory, frozen_plan_sha256):
    """Return (outcomes, receipt_bytes_by_sha256, scoped_validation_receipt).

    All-not-run bundles are retained faithfully; output_envelope independently
    refuses to call those declared execution. Only the required plan, contract,
    input manifest, outcomes and attempted-clip sidecars are in validation scope.
    Model-load/environment/summary/progress files and current WAV/model bodies
    are not authenticated or checked here. Reuse the returned bytes for scoring.
    """
    _sha(frozen_plan_sha256, 'independently frozen plan SHA')
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    inventory, receipts, bindings = {}, {}, []
    try:
        _, plan = _read(directory_fd, 'plan.json', inventory)
        _require(inventory['plan.json']['sha256'] == frozen_plan_sha256,
                 'Plan bytes differ from independently frozen plan SHA')
        _plan(plan)
        contract_raw, contract = _read(directory_fd, 'contract.json', inventory)
        _require(contract_raw == _bytes(plan['contract']) and
                 inventory['contract.json']['sha256'] == plan['contract_sha256'],
                 'Retained contract bytes differ from frozen plan')
        decoder_raw, _ = _read(directory_fd, 'decoder-inputs.json', inventory, 1024 * 1024)
        inputs = validate_decoder_manifest(decoder_raw, plan['decoder_manifest_sha256'])
        _, outcomes = _read(directory_fd, 'outcomes.json', inventory, 1024 * 1024)
        _require(type(outcomes) is list and len(outcomes) == len(inputs),
                 'Exactly 16 explicit outcomes required')
        model = {'model_id': plan['model_id'], 'revision': MODELS[plan['model_id']][0],
                 'run_id': contract['run_id']}
        expected_sidecars, stopped = set(), False
        counts = dict.fromkeys(('success', 'error', 'timeout', 'not_run'), 0)
        for oid, outcome in zip(inputs, outcomes):
            _outcome(outcome, oid, inputs)
            status = outcome['status']
            _require(not stopped or status == 'not_run', 'Primary run continued after stopping')
            stopped = status != 'success'
            counts[status] += 1
            if status == 'not_run':
                continue
            receipt_name, evidence_name = oid + '.receipt.json', oid + '.decoder.json'
            expected_sidecars.update((receipt_name, evidence_name))
            raw, receipt = _read(directory_fd, receipt_name, inventory, 1024 * 1024)
            receipt_sha = inventory[receipt_name]['sha256']
            _require(receipt_sha == outcome['execution_receipt_sha256'], 'Receipt bytes changed')
            _fields(receipt, RECEIPT_FIELDS, 'execution receipt')
            _require(receipt['schema'] == 'asr-clip-execution-receipt-v1', 'Unexpected receipt schema')
            _equal(receipt['model'], model, 'Receipt model/revision/run')
            for field in ('opaque_id', 'wav_sha256') + ROW_FIELDS:
                _equal(receipt[field], outcome[field], 'Receipt ' + field)
            _equal(receipt['input_binding_sha256'], inputs[oid]['binding_sha256'], 'Receipt input binding')
            _equal(receipt['decoder_manifest_sha256'], plan['decoder_manifest_sha256'], 'Receipt input manifest')
            _equal(receipt['execution_contract_sha256'], plan['contract_sha256'], 'Receipt contract')
            _require(receipt['attempt_started'] is True and
                     receipt['model_forward_completed'] is (status == 'success'),
                     'Receipt forward/attempt state differs')
            _, sidecar = _read(directory_fd, evidence_name, inventory)
            _require(inventory[evidence_name]['sha256'] == receipt['decoder_evidence_sha256'],
                     'Decoder evidence bytes changed')
            _fields(sidecar, SIDECAR_FIELDS, 'decoder evidence sidecar')
            _require(sidecar['schema'] == 'asr-decoder-evidence-sidecar-v1', 'Unexpected decoder sidecar schema')
            for field in ('model', 'opaque_id', 'wav_sha256', 'input_binding_sha256'):
                _equal(sidecar[field], receipt[field], 'Decoder sidecar ' + field)
            _equal(sidecar['outcome'], {k: receipt[k] for k in ROW_FIELDS}, 'Decoder sidecar outcome')
            seconds = sidecar['wall_seconds']
            _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                     'Invalid decoder duration')
            evidence = sidecar['evidence']
            if status == 'success':
                descriptor = inputs[oid]['descriptor']
                extracted = _qwen(evidence, descriptor) if plan['model_id'] == QWEN else _sense(evidence, descriptor, oid)
                _equal(extracted, {k: outcome[k] for k in ROW_FIELDS if k != 'status'},
                       'Re-extracted transcript outcome')
            else:
                _fields(evidence, {'schema', 'exception_type', 'detail',
                    'forward_completion_confirmed'}, 'decoder failure evidence')
                _require(evidence['schema'] == 'asr-decoder-failure-evidence-v1' and
                         evidence['forward_completion_confirmed'] is False and
                         type(evidence['exception_type']) is str and evidence['exception_type'],
                         'Invalid decoder failure evidence')
                bounded_text(evidence['detail'])
                _require(len(evidence['detail']) <= 4096, 'Oversized failure detail')
                _require((evidence['exception_type'] == 'ClipDeadline') == (status == 'timeout'),
                         'Failure exception/status differs')
            receipts[receipt_sha] = raw
            bindings.append({'opaque_id': oid, 'status': status,
                'execution_receipt_sha256': receipt_sha,
                'decoder_evidence_sha256': inventory[evidence_name]['sha256']})
        actual_sidecars = {name for name in os.listdir(directory_fd)
                           if name.endswith(('.receipt.json', '.decoder.json'))}
        _require(actual_sidecars == expected_sidecars,
                 'Missing, unused or not-run execution/decoder sidecars')
    finally:
        os.close(directory_fd)
    validation = {'schema': 'asr-primary-bundle-validation-v1', 'model': model,
        'plan_sha256': frozen_plan_sha256, 'execution_contract_sha256': plan['contract_sha256'],
        'decoder_manifest_sha256': plan['decoder_manifest_sha256'],
        'outcomes_sha256': inventory['outcomes.json']['sha256'], 'counts': counts,
        'records': bindings, 'files': inventory,
        'transcript_extractions_verified': counts['success'],
        'scope': 'Exact retained-byte linkage and declaration/transcript-extraction consistency only. '
            'Not cryptographic proof that inference occurred, model authenticity, or independent '
            'tokenizer re-decoding. No model loading, downloading or forward execution. '
            'Model-load, environment, summary and progress sidecars and live WAV/model bodies '
            'are outside this validation scope.'}
    return outcomes, receipts, validation
