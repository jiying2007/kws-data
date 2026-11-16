"""Only invented bytes: no real assets, model imports, C binaries or inference."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_c_trace as reader

FRAME_BYTES = sum(16 + 4 * count for _, count in reader.STAGES)
HEAD_OFFSET = sum(16 + 4 * count for _, count in reader.STAGES[:3]) + 16


def invented(windows=33, events=None, foreground=1):
    events = {17: 1, 32: 1} if events is None else events
    data = bytearray(reader.MAGIC)
    for index in range(windows):
        for stage, count in reader.STAGES:
            data.extend(struct.pack('<IIQ', stage, count, 16000 + index * 1280))
            if stage < 4 or index < 15:
                data.extend(b'\0' * (4 * count))
            else:
                probabilities = [0.05, 0.05, 0.05]; probabilities[foreground] = 0.9
                data.extend(struct.pack('<8f', 1, events.get(index, 0), -2, 3, -1, *probabilities))
    return bytes(data)


def replace_float(raw, window, head_index, value):
    data = bytearray(raw)
    struct.pack_into('<f', data, 8 + window * FRAME_BYTES + HEAD_OFFSET + 4 * head_index, value)
    return bytes(data)


def parse(raw):
    rows = []
    summary = reader.parse_stream(io.BytesIO(raw), rows.append)
    return rows, summary


class TraceTests(unittest.TestCase):
    def test_complete_warmup_confirm_and_cooldown(self):
        rows, summary = parse(invented())
        self.assertEqual((summary['embedding_windows'], summary['head_ready_outputs'], summary['event_count']), (33, 18, 2))
        self.assertEqual([(r['sample_end'], r['seconds'], r['label']) for r in rows if r['event']],
                         [(37760, 2.36, 'Hey Vosen'), (56960, 3.56, 'Hey Vosen')])
        self.assertTrue(all(not row['ready'] for row in rows[:15]))
        self.assertEqual(rows[15]['sample_end'], 35200)
        self.assertEqual(summary['new_model_calls'], 0)

    def test_mero_and_empty_trace(self):
        rows, _ = parse(invented(18, {17: 2}, foreground=2))
        self.assertEqual(rows[-1]['label'], 'Hey Mero')
        rows, summary = parse(reader.MAGIC)
        self.assertEqual(rows, [])
        self.assertIsNone(summary['input_total_samples'])
        self.assertIsNone(summary['last_sample_end'])
        self.assertFalse(summary['no_event_is_frr'])

    def test_magic_truncation_and_extra_bytes(self):
        raw = invented(1, {})
        for invalid in (b'', b'WRONG001', raw[:9], raw[:-1], raw + b'\0'):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                parse(invalid)

    def test_exact_stage_order_counts_and_clock(self):
        raw = invented(2, {})
        for offset, header in ((8, (2, 3104, 16000)), (8, (1, 3103, 16000)),
                               (8, (1, 3104, 16001)), (8 + FRAME_BYTES, (1, 3104, 18560)),
                               (8 + HEAD_OFFSET - 16, (4, 8, 17280))):
            invalid = bytearray(raw); struct.pack_into('<IIQ', invalid, offset, *header)
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, 'STAGE_COUNT_OR_CLOCK'):
                parse(invalid)

    def test_all_stages_must_be_finite(self):
        raw = invented(1, {})
        offsets = [24]
        position = 8
        for _, count in reader.STAGES:
            offsets.append(position + 16)
            position += 16 + 4 * count
        for offset in offsets:
            invalid = bytearray(raw); struct.pack_into('<f', invalid, offset, float('nan'))
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'NONFINITE'):
                parse(invalid)

    def test_ready_and_warmup_values(self):
        raw = invented(16, {})
        cases = ((0, 0, 1), (15, 0, 0), (0, 2, 0.1), (0, 1, -0.0), (0, 0, -0.0))
        for window, index, value in cases:
            with self.subTest(case=(window, index, value)), self.assertRaises(ValueError):
                parse(replace_float(raw, window, index, value))

    def test_probability_and_event_validation(self):
        raw = invented(18, {17: 1})
        for window, index, value in ((15, 1, 3), (15, 1, 0.5), (15, 5, -0.1),
                                      (15, 5, 0.5), (17, 1, 0), (16, 1, 1)):
            with self.subTest(case=(window, index, value)), self.assertRaises(ValueError):
                parse(replace_float(raw, window, index, value))

    def test_threshold_uses_float32_and_three_consecutive(self):
        raw = invented(18, {17: 1})
        for index in range(15, 18):
            raw = replace_float(raw, index, 5, 0.1)
            raw = replace_float(raw, index, 6, 0.8)
            raw = replace_float(raw, index, 7, 0.1)
        self.assertEqual(parse(raw)[0][-1]['event'], 1)
        below = struct.unpack('<f', struct.pack('<I', 0x3f4ccccc))[0]
        invalid = replace_float(raw, 16, 6, below)
        with self.assertRaisesRegex(ValueError, 'FIXED_EVENT_DECODER_MISMATCH'):
            parse(invalid)
        self.assertEqual(parse(replace_float(invalid, 17, 1, 0))[0][-1]['event'], 0)

    def test_cooldown_event_cannot_be_early(self):
        with self.assertRaisesRegex(ValueError, 'FIXED_EVENT_DECODER_MISMATCH'):
            parse(invented(33, {17: 1, 30: 1}))

    def test_json_jsonl_and_create_only_atomic_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); trace = root / 'input.bin'; trace.write_bytes(invented(18, {17: 1}))
            before = trace.read_bytes()
            for format_name in ('json', 'jsonl'):
                output = root / ('out.' + format_name)
                summary = reader.convert(trace, output, format_name)
                if format_name == 'json':
                    document = json.loads(output.read_text())
                    self.assertEqual(document['summary'], summary)
                    self.assertEqual(document['frames'][-1]['label'], 'Hey Vosen')
                else:
                    documents = [json.loads(line) for line in output.read_text().splitlines()]
                    self.assertEqual((documents[0]['type'], documents[-1]), ('metadata', summary))
                with self.assertRaisesRegex(ValueError, 'FRESH_OUTPUT_REQUIRED'):
                    reader.convert(trace, output, format_name)
            self.assertEqual(trace.read_bytes(), before)
            broken = root / 'broken.bin'; broken.write_bytes(before[:-1])
            output = root / 'must-not-exist.json'
            with self.assertRaises(ValueError):
                reader.convert(broken, output, 'json')
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob('.trace-readback-*')))


if __name__ == '__main__':
    unittest.main()
