"""Read saved full-chain C traces using only the Python standard library.

No model, audio, compilation, or inference is run. Output files are create-only
and published only after the entire trace passes its structural/event checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

SCHEMA = 'english-fixed-c-trace-readback-v1'
MAGIC = b'EFFTR01\0'
STAGES = ((1, 3104), (2, 2432), (3, 96), (4, 8))
LABELS = {0: 'none', 1: 'Hey Vosen', 2: 'Hey Mero'}
POLICY = {'sample_rate_hz': 16000, 'first_embedding_sample': 16000,
    'embedding_hop_samples': 1280, 'history_embeddings': 16,
    'first_head_sample': 35200, 'threshold': 0.8, 'confirm_embeddings': 3,
    'cooldown_embeddings': 13, 'earliest_event_sample': 37760, 'eof_padding_samples': 0}
THRESHOLD_F32 = struct.unpack('<f', struct.pack('<f', 0.8))[0]


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def parse_stream(stream, emit):
    """Validate stages and recompute the fixed decoder from stored probabilities.

    This checks recorded decoder consistency, not whether the model's numeric
    logits/probabilities are correct for the original audio.
    """
    digest = hashlib.sha256()
    consumed = windows = ready_outputs = event_count = 0
    candidate = confirmations = blocked_until = 0

    def read(size):
        nonlocal consumed
        raw = stream.read(size)
        consumed += len(raw)
        digest.update(raw)
        return raw

    need(read(8) == MAGIC, 'TRACE_MAGIC')
    while True:
        first = read(16)
        if not first:
            break
        sample = 16000 + windows * 1280
        need(sample <= 2**64 - 1 - 13 * 1280, 'TRACE_CLOCK_RANGE')
        for stage, count in STAGES:
            header = first if stage == 1 else read(16)
            need(len(header) == 16, 'TRUNCATED_STAGE_HEADER')
            need(struct.unpack('<IIQ', header) == (stage, count, sample), 'STAGE_COUNT_OR_CLOCK')
            raw = read(4 * count)
            need(len(raw) == 4 * count, 'TRUNCATED_STAGE_VALUES')
            values = tuple(v[0] for v in struct.iter_unpack('<f', raw))
            need(all(math.isfinite(v) for v in values), 'NONFINITE_TRACE_VALUE')
        ready = int(windows >= 15)
        need(raw[:4] == struct.pack('<f', ready), 'HEAD_READY_OR_WARMUP')
        event_float = values[1]
        need(event_float in (0.0, 1.0, 2.0), 'EVENT_ID')
        event = int(event_float)
        need(raw[4:8] == struct.pack('<f', event), 'EVENT_ID_ENCODING')
        logits, probabilities = values[2:5], values[5:8]
        expected = 0
        if not ready:
            need(event == 0 and raw[8:] == b'\0' * 24, 'WARMUP_HEAD_VALUES_MUST_BE_ZERO')
        else:
            ready_outputs += 1
            need(all(0 <= p <= 1 for p in probabilities) and
                 abs(sum(probabilities) - 1.0) <= 2e-6, 'THREE_NORMALIZED_PROBABILITIES')
            selected = max(range(3), key=lambda k: probabilities[k])
            if selected == 0 or probabilities[selected] < THRESHOLD_F32 or sample < blocked_until:
                candidate = confirmations = 0
            else:
                confirmations = confirmations + 1 if selected == candidate else 1
                candidate = selected
                if confirmations == 3:
                    expected = selected
                    blocked_until = sample + 13 * 1280
                    candidate = confirmations = 0
            need(event == expected, 'FIXED_EVENT_DECODER_MISMATCH')
        event_count += int(event != 0)
        emit({'type': 'head_frame', 'embedding_index': windows + 1, 'sample_end': sample,
            'seconds': sample / 16000, 'ready': bool(ready), 'event': event, 'label': LABELS[event],
            'logits': list(logits), 'probabilities': list(probabilities)})
        windows += 1
    return {'type': 'summary', 'status': 'PARSED_SAVED_C_TRACE', 'trace_bytes': consumed,
        'trace_sha256': digest.hexdigest(), 'embedding_windows': windows, 'head_ready_outputs': ready_outputs,
        'event_count': event_count, 'last_sample_end': 16000 + (windows - 1) * 1280 if windows else None,
        'input_total_samples': None, 'no_event_is_frr': False, 'sample_clock_meaning': 'proposal_available_sample_not_word_end',
        'model_numeric_parity': 'NOT_CHECKED_BY_THIS_READER', 'new_model_calls': 0, 'new_c_processes': 0}


def convert(trace, output, format_name='jsonl'):
    trace, output = Path(trace), Path(output)
    need(trace.is_file() and not trace.is_symlink(), 'REGULAR_TRACE_REQUIRED')
    need(output.parent.is_dir() and not output.exists() and not output.is_symlink(), 'FRESH_OUTPUT_REQUIRED')
    need(format_name in ('json', 'jsonl'), 'OUTPUT_FORMAT')
    metadata = {'type': 'metadata', 'schema': SCHEMA, 'policy': POLICY, 'event_labels': LABELS,
        'class_labels': {0: 'background', 1: 'Hey Vosen', 2: 'Hey Mero'},
        'time_is_acoustic_word_end_latency': False}
    encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                        prefix='.trace-readback-', delete=False) as target:
            temporary = Path(target.name)
            if format_name == 'jsonl':
                target.write(encode(metadata) + '\n')
            else:
                target.write('{"metadata":' + encode(metadata) + ',"frames":[')
            first = True
            def emit(row):
                nonlocal first
                if format_name == 'jsonl':
                    target.write(encode(row) + '\n')
                else:
                    target.write(('' if first else ',') + encode(row))
                first = False
            with trace.open('rb') as source:
                summary = parse_stream(source, emit)
            target.write(encode(summary) + '\n' if format_name == 'jsonl'
                         else '],"summary":' + encode(summary) + '}\n')
        # Hard-link publication fails if output already exists; no replacement.
        os.link(temporary, output)
        return summary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('jsonl', 'json'), default='jsonl')
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.trace, args.output, args.format), sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
