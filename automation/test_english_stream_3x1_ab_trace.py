#!/usr/bin/env python3
"""Pure synthetic-byte tests. Never execute a kernel, model, CLI or subprocess."""
import ast
import copy
import hashlib
from pathlib import Path
import struct
import unittest

import english_stream_3x1_ab_once as target


def fixture(event_at=()):
    """Build the documented C wire layout independently of the worker parser."""
    event_at = dict(event_at)
    data = bytearray(struct.pack('<8s6IQ', b'EKWSTRC1', 1, 1, 16000, 320, 96, 3, 256000))
    positions = {'embedding': [], 'score': [], 'event': []}
    events = 0

    def append(kind, payload, group):
        positions.setdefault(group, []).append(len(data))
        data.extend(struct.pack('<BI', kind, len(payload)))
        data.extend(payload)

    for i in range(188):
        sample = 16000 + 1280 * i
        # Exact dyadic finite values, not outputs of any real model.
        embedding = [0.0] + [(i - d) / 1024 for d in range(1, 96)]
        append(1, struct.pack('<QQ96f', i, sample, *embedding), 'embedding')
        ready, keyword = int(i >= 15), event_at.get(i, 0)
        logits = [0.0, 0.0, 0.0] if not ready else [0.5, -0.25, 0.125]
        probs = [0.0, 0.0, 0.0] if not ready else [0.75, 0.125, 0.125]
        if keyword:
            if not ready or keyword not in (1, 2):
                raise ValueError('fixture event must have a ready score and valid keyword')
            probs = [0.0625, 0.0625, 0.0625]
            probs[keyword] = 0.875
        append(2, struct.pack('<QIIQ6f', i, ready, keyword, sample, *logits, *probs), 'score')
        if keyword:
            append(3, struct.pack('<QQIQf', events, i, keyword, sample, probs[keyword]), 'event')
            events += 1
    append(255, struct.pack('<6Q', 256000, 188, 188, 173, events, 800), 'end')
    return bytes(data), positions


def metadata(raw, events):
    return {
        'schema': 'english-stream-numeric-trace-v1',
        'trace_completed': True,
        'format_magic': 'EKWSTRC1',
        'format_version': 1,
        'abi': 1,
        'sample_rate_hz': 16000,
        'encoding': 's16le_mono',
        'chunk_samples': 320,
        'input_samples': 256000,
        'input_bytes': 512000,
        'input_repetitions': 1,
        'resets': 0,
        'eof_padding_samples': 0,
        'feed_calls': 800,
        'embedding_records': 188,
        'real_head_calls': 188,
        'score_records': 188,
        'ready_outputs': 173,
        'event_records': events,
        'trace_records': 377 + events,
        'trace_bytes': len(raw),
        'end_record_written': True,
        'score_observer_enabled': True,
        'original_head_link_wrap': True,
        'performance_measured': False,
    }


def changed(raw, offset, fmt, value):
    data = bytearray(raw)
    struct.pack_into(fmt, data, offset, value)
    return bytes(data)


def equality_guard():
    """Isolate only the real full-byte equality guard, never execute execute()."""
    tree = ast.parse(Path(target.__file__).read_text(encoding='utf-8'))
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == 'need' and len(node.args) == 2
                  and any(isinstance(part, ast.Constant) and isinstance(part.value, str)
                          and part.value.startswith('full numeric trace mismatch:')
                          for part in ast.walk(node.args[1]))]
    if len(candidates) != 1:
        raise AssertionError('Expected exactly one production full-numeric equality guard')
    guard = candidates[0]
    comparison = guard.args[0]
    if not (isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq) and len(comparison.comparators) == 1):
        raise AssertionError('Numeric parity must compare complete raw trace bytes')
    for expression, variant in zip([comparison.left, comparison.comparators[0]], ['baseline', 'candidate']):
        expected = ast.parse(f'files["{variant}"].read_bytes()', mode='eval').body
        if ast.dump(expression, include_attributes=False) != ast.dump(expected, include_attributes=False):
            raise AssertionError('Numeric parity must read both complete trace files')
    return compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(guard))), '<isolated-trace-equality>', 'eval')


class FixtureFile:
    def __init__(self, raw):
        self.raw, self.reads = raw, 0

    def read_bytes(self):
        self.reads += 1
        return self.raw


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zero, cls.zpos = fixture()
        cls.events, cls.epos = fixture(((17, 1), (40, 2)))
        cls.equal_code = equality_guard()

    def assert_bad(self, raw):
        with self.assertRaises(ValueError):
            target.parse_trace(raw)

    def compare(self, left, right):
        files = {'baseline': FixtureFile(left), 'candidate': FixtureFile(right)}
        # The AST was restricted to need(files[..].read_bytes() == ..., message).
        eval(self.equal_code, {'__builtins__': {}, 'need': target.need},
             {'files': files, 'case_id': 'synthetic-fixture'})
        self.assertEqual([files[k].reads for k in ('baseline', 'candidate')], [1, 1])

    def test_valid_zero_event_exact_layout(self):
        self.assertEqual(len(self.zero), 86197)
        result = target.parse_trace(self.zero)
        self.assertEqual(result, {'input_samples': 256000, 'embeddings': 188, 'scores': 188,
                                 'ready': 173, 'events': 0, 'feed_calls': 800, 'records': 377,
                                 'bytes': 86197, 'sha256': hashlib.sha256(self.zero).hexdigest()})
        target.verify_trace_metadata(metadata(self.zero, 0), result)

    def test_valid_nonzero_events_exact_layout(self):
        self.assertEqual(len(self.events), 86197 + 2 * 37)
        result = target.parse_trace(self.events)
        self.assertEqual((result['events'], result['records'], result['ready']), (2, 379, 173))
        target.verify_trace_metadata(metadata(self.events, 2), result)

    def test_reject_each_header_field(self):
        bad = [changed(self.zero, 0, '<8s', b'BADMAGIC')]
        for offset in (8, 12, 16, 20, 24, 28):
            value, = struct.unpack_from('<I', self.zero, offset)
            bad.append(changed(self.zero, offset, '<I', value + 1))
        bad.append(changed(self.zero, 32, '<Q', 256001))
        for number, raw in enumerate(bad):
            with self.subTest(field=number): self.assert_bad(raw)

    def test_reject_record_kind_length_and_order(self):
        for group, expected_type, length in [('embedding', 1, 400), ('score', 2, 48), ('end', 255, 48)]:
            start = self.zpos[group][0]
            for raw in [changed(self.zero, start, '<B', 0),
                        changed(self.zero, start + 1, '<I', length - 1),
                        changed(self.zero, start + 1, '<I', length + 1)]:
                with self.subTest(group=group): self.assert_bad(raw)
        start = self.epos['event'][0]
        self.assert_bad(changed(self.events, start, '<B', 2))
        self.assert_bad(changed(self.events, start + 1, '<I', 31))
        self.assert_bad(changed(self.zero, self.zpos['embedding'][0], '<B', 2))

    def test_reject_embedding_ordinal_and_clock(self):
        for index in (0, 1, 187):
            start = self.zpos['embedding'][index] + 5
            with self.subTest(index=index):
                self.assert_bad(changed(self.zero, start, '<Q', index + 1))
                self.assert_bad(changed(self.zero, start + 8, '<Q', 16001 + index * 1280))

    def test_reject_score_ordinal_clock_ready_and_keyword(self):
        for index in (0, 14, 15, 187):
            start = self.zpos['score'][index] + 5
            with self.subTest(index=index):
                self.assert_bad(changed(self.zero, start, '<Q', index + 1))
                self.assert_bad(changed(self.zero, start + 16, '<Q', 16001 + index * 1280))
                self.assert_bad(changed(self.zero, start + 8, '<I', 1 - int(index >= 15)))
                self.assert_bad(changed(self.zero, start + 12, '<I', 3))
        self.assert_bad(changed(self.zero, self.zpos['score'][0] + 5 + 12, '<I', 1))

    def test_reject_event_identity_and_clock(self):
        start = self.epos['event'][0] + 5
        for delta, fmt, wrong in [(0, '<Q', 1), (8, '<Q', 18), (16, '<I', 2), (20, '<Q', 0)]:
            with self.subTest(offset=delta): self.assert_bad(changed(self.events, start + delta, fmt, wrong))

    def test_reject_event_confidence_different_from_score_probability_bits(self):
        # Finite but different: the C callback rejects this too.
        start = self.epos['event'][0] + 5 + 28
        self.assert_bad(changed(self.events, start, '<f', 0.5))

    def test_reject_signed_zero_event_confidence_bit_mismatch(self):
        # Float equality would wrongly accept +0 == -0. The wire contract is bits.
        raw = changed(self.events, self.epos['score'][17] + 5 + 36 + 4, '<I', 0)
        raw = changed(raw, self.epos['event'][0] + 5 + 28, '<I', 0x80000000)
        self.assert_bad(raw)

    def test_reject_nonfinite_values_in_every_numeric_region(self):
        fields = [(self.zero, self.zpos['embedding'][3] + 5 + 16),
                  (self.zero, self.zpos['score'][17] + 5 + 24),
                  (self.zero, self.zpos['score'][17] + 5 + 36),
                  (self.events, self.epos['event'][0] + 5 + 28)]
        for number, (raw, offset) in enumerate(fields):
            for bits in (0x7fc00001, 0x7f800000, 0xff800000):
                with self.subTest(region=number, bits=bits): self.assert_bad(changed(raw, offset, '<I', bits))

    def test_reject_each_end_counter(self):
        start = self.zpos['end'][0] + 5
        for field in range(6):
            offset = start + field * 8
            value, = struct.unpack_from('<Q', self.zero, offset)
            with self.subTest(field=field): self.assert_bad(changed(self.zero, offset, '<Q', value + 1))

    def test_reject_truncation_at_header_prefix_payload_and_end(self):
        boundaries = [0, 39, 40, 41, 44, 45, 444, 445, self.zpos['score'][15] + 15,
                      self.zpos['end'][0], len(self.zero) - 1]
        for size in boundaries:
            with self.subTest(size=size): self.assert_bad(self.zero[:size])
        self.assert_bad(self.events[:self.epos['event'][0] + 36])

    def test_reject_trailing_bytes_and_duplicate_end(self):
        self.assert_bad(self.zero + b'\x00')
        self.assert_bad(self.zero + self.zero[self.zpos['end'][0]:])

    def test_reject_missing_duplicate_and_unannounced_event(self):
        start = self.epos['event'][0]
        event = self.events[start:start + 37]
        self.assert_bad(self.events[:start] + self.events[start + 37:])
        self.assert_bad(self.events[:start] + event + self.events[start:])
        before_end = self.zpos['end'][0]
        self.assert_bad(self.zero[:before_end] + event + self.zero[before_end:])

    def test_metadata_requires_success_and_diagnostic_boundaries(self):
        parsed = target.parse_trace(self.zero)
        good = metadata(self.zero, 0)
        for key in ('trace_completed', 'end_record_written', 'score_observer_enabled',
                    'original_head_link_wrap', 'performance_measured'):
            for replacement in (not good[key], None, int(good[key])):
                bad = dict(good); bad[key] = replacement
                with self.subTest(key=key, value=replacement):
                    with self.assertRaises(ValueError): target.verify_trace_metadata(bad, parsed)
        bad = dict(good); bad['schema'] = 'unrelated'
        with self.assertRaises(ValueError): target.verify_trace_metadata(bad, parsed)

    def test_metadata_requires_every_counter(self):
        parsed = target.parse_trace(self.events)
        good = metadata(self.events, 2)
        keys = ['input_samples', 'input_bytes', 'feed_calls', 'embedding_records', 'real_head_calls',
                'score_records', 'ready_outputs', 'event_records', 'trace_records', 'trace_bytes',
                'input_repetitions', 'resets', 'eof_padding_samples']
        for key in keys:
            for missing in (False, True):
                bad = dict(good)
                if missing: del bad[key]
                else: bad[key] += 1
                with self.subTest(key=key, missing=missing):
                    with self.assertRaises(ValueError): target.verify_trace_metadata(bad, parsed)

    def test_metadata_requires_header_and_encoding_fields(self):
        parsed = target.parse_trace(self.zero)
        good = metadata(self.zero, 0)
        keys = ['format_magic', 'format_version', 'abi', 'sample_rate_hz', 'encoding', 'chunk_samples']
        for key in keys:
            for missing in (False, True):
                bad = dict(good)
                if missing: del bad[key]
                else: bad[key] = 'wrong' if isinstance(bad[key], str) else bad[key] + 1
                with self.subTest(key=key, missing=missing):
                    with self.assertRaises(ValueError): target.verify_trace_metadata(bad, parsed)

    def test_metadata_rejects_noninteger_counter_types(self):
        parsed = target.parse_trace(self.zero)
        good = metadata(self.zero, 0)
        for key in ('format_version', 'abi', 'sample_rate_hz', 'chunk_samples',
                    'input_samples', 'feed_calls', 'embedding_records', 'input_repetitions',
                    'resets', 'eof_padding_samples', 'event_records'):
            replacements = [float(good[key])]
            if good[key] in (0, 1): replacements.append(bool(good[key]))
            for value in replacements:
                bad = dict(good); bad[key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError): target.verify_trace_metadata(bad, parsed)

    def test_production_full_equality_accepts_equal_complete_bytes(self):
        self.compare(self.zero, bytes(bytearray(self.zero)))
        self.compare(self.events, bytes(bytearray(self.events)))

    def test_production_full_equality_rejects_finite_embedding_difference(self):
        right = changed(self.zero, self.zpos['embedding'][10] + 5 + 16, '<I', 1)
        self.assertEqual(target.parse_trace(right)['events'], 0)
        with self.assertRaises(ValueError): self.compare(self.zero, right)

    def test_production_full_equality_rejects_signed_zero_score_difference(self):
        # Equal numerical value and identical counters/events, different f32 bits.
        right = changed(self.zero, self.zpos['score'][0] + 5 + 24, '<I', 0x80000000)
        self.assertEqual(target.parse_trace(right)['events'], 0)
        with self.assertRaises(ValueError): self.compare(self.zero, right)

    def test_production_full_equality_rejects_event_and_probability_change(self):
        right = changed(self.events, self.epos['score'][17] + 5 + 36 + 4, '<f', 0.75)
        right = changed(right, self.epos['event'][0] + 5 + 28, '<f', 0.75)
        self.assertEqual(target.parse_trace(right)['events'], 2)
        with self.assertRaises(ValueError): self.compare(self.events, right)


if __name__ == '__main__':
    unittest.main()
