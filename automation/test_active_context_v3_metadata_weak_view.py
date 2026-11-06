"""Strict dual-container fixtures and native/ASR replay; no model imports."""
from pathlib import Path
import copy
import shutil
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_context_v3_metadata_weak_view as context
import active_context_v3_weak_view as original
import test_active_context_v3_weak_view as fixture
import test_active_d1_outcomes as native_fixture
weak = context.weak


def peak_wav(pcm, timestamp=1000, *, peak=.25, position=0, extra=False):
    chunks = [(b'fmt ', struct.pack('<HHIIHH', 3, 1, 24000, 96000, 4, 32)),
              (b'fact', struct.pack('<I', len(pcm)//4)),
              (b'PEAK', struct.pack('<IIfI', 1, timestamp, peak, position)),
              *([(b'JUNK', b'unchanged')] if extra else []), (b'data', pcm)]
    payload = b'WAVE' + b''.join(kind + struct.pack('<I', len(body)) + body + (b'\0' if len(body)%2 else b'')
                               for kind, body in chunks)
    return b'RIFF' + struct.pack('<I', len(payload)) + payload


def helper_fixture(*, returned_stamp=1000, generated_stamp=1001, extra=False):
    pcm = struct.pack('<6f', .25, -.125, 0, .125, -.25, .0625)
    returned = peak_wav(pcm, returned_stamp, extra=extra)
    generated = peak_wav(pcm, generated_stamp, extra=extra)
    array = native_fixture.npy(pcm)
    native = {'wav': {'bytes': len(returned), 'sha256': weak.sha(returned)},
        'array': {'bytes': len(array), 'sha256': weak.sha(array)},
        'dtype': '<f4', 'shape': [6], 'samples_sha256': weak.sha(pcm)}
    completed = {'audio': {'native_bytes': len(generated), 'native_sha256': weak.sha(generated)}}
    return [generated, returned, array, native, weak.canonical(completed), pcm]


class ContainerTests(unittest.TestCase):
    def test_distinct_hashes_and_exact_array_pcm_survive_one_second_metadata_reconciliation(self):
        args = helper_fixture()
        before = copy.deepcopy(args)
        result = context.native_container_relation(*args)
        self.assertEqual(result['relation'], 'peak-timestamp-plus-one-second')
        self.assertNotEqual(result['returned_wav']['sha256'], result['bulk_generation_wav']['sha256'])
        self.assertEqual((result['returned_peak_timestamp'], result['bulk_peak_timestamp']), (1000, 1001))
        self.assertEqual(result['differing_byte_offsets'], [60])
        self.assertEqual(args, before)
        self.assertIs(result['signal_samples_changed'], False)
        self.assertIs(result['generation_outcome_changed'], False)
        self.assertIs(result['technical_gate_changed'], False)

    def test_one_second_carry_can_change_multiple_timestamp_bytes_only(self):
        result = context.native_container_relation(*helper_fixture(returned_stamp=255, generated_stamp=256))
        self.assertEqual(result['differing_byte_offsets'], [60, 61])

    def test_byte_identical_original_container_keeps_old_acceptance(self):
        result = context.native_container_relation(*helper_fixture(generated_stamp=1000))
        self.assertEqual(result['relation'], 'byte-identical')
        self.assertEqual(result['differing_byte_offsets'], [])
        self.assertEqual(result['returned_wav'], result['bulk_generation_wav'])

    def test_two_seconds_backward_and_large_unrelated_timestamp_changes_fail(self):
        for stamp in (999, 1002, 1700000000):
            with self.assertRaisesRegex(ValueError, 'ONLY_ONE_SECOND_FORWARD_ALLOWED'):
                context.native_container_relation(*helper_fixture(generated_stamp=stamp))

    def test_each_original_hash_remains_authoritative(self):
        for which in ('returned', 'generation', 'array'):
            args = helper_fixture()
            if which == 'returned': args[3]['wav']['sha256'] = 'a' * 64
            elif which == 'array': args[3]['array']['sha256'] = 'a' * 64
            else:
                record = weak.loads(args[4]); record['audio']['native_sha256'] = 'a' * 64
                args[4] = weak.canonical(record)
            with self.assertRaisesRegex(ValueError, 'BOTH_ORIGINAL_HASHES_REQUIRED'):
                context.native_container_relation(*args)

    def test_other_peak_fields_and_fact_are_not_metadata_exceptions(self):
        for offset in (40, 48, 56, 64, 68):
            args = helper_fixture()
            changed = bytearray(args[0]); changed[offset] ^= 1; args[0] = bytes(changed)
            completed = weak.loads(args[4]); completed['audio']['native_sha256'] = weak.sha(args[0])
            args[4] = weak.canonical(completed)
            with self.assertRaises(ValueError): context.native_container_relation(*args)

    def test_extra_ancillary_chunks_cannot_expand_the_exception(self):
        with self.assertRaisesRegex(ValueError, 'EXACT_CONTAINER_LAYOUT_REQUIRED'):
            context.native_container_relation(*helper_fixture(extra=True))

    def test_different_pcm_is_rejected_even_with_coherently_updated_wav_hash(self):
        args = helper_fixture()
        changed_pcm = struct.pack('<6f', .5, -.125, 0, .125, -.25, .0625)
        args[0] = peak_wav(changed_pcm, 1001)
        record = weak.loads(args[4]); record['audio']['native_sha256'] = weak.sha(args[0]); args[4] = weak.canonical(record)
        with self.assertRaisesRegex(ValueError, 'EXACT_NATIVE_PCM_REQUIRED'):
            context.native_container_relation(*args)

    def test_different_array_is_rejected_even_with_coherently_updated_array_hash(self):
        args = helper_fixture()
        changed_pcm = struct.pack('<6f', .5, -.125, 0, .125, -.25, .0625)
        args[2] = native_fixture.npy(changed_pcm)
        args[3]['array'] = {'bytes': len(args[2]), 'sha256': weak.sha(args[2])}
        args[3]['samples_sha256'] = weak.sha(changed_pcm)
        with self.assertRaisesRegex(ValueError, 'D1_NATIVE_ARRAY_PCM_JOIN'):
            context.native_container_relation(*args)

    def test_versioned_roster_preserves_all_original_source_files(self):
        self.assertEqual(context.PRODUCER_FILES, ('automation/active_context_v3_metadata_weak_view.py', *original.PRODUCER_FILES))
        self.assertNotEqual(context.SCHEMA, original.SCHEMA)
        self.assertNotEqual(context.SOURCE_ADAPTER, original.SOURCE_ADAPTER)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.ContextV3WeakViewTests()
        base_writer = fixture.audio_fixture.fixture_writer
        def writer(folder, values, rate, metrics):
            record = base_writer(folder, values, rate, metrics)
            path = folder / 'native-f32.wav'
            value = peak_wav(weak.float_wav_pcm(path.read_bytes()))
            path.write_bytes(value)
            record.update(native_sha256=weak.sha(value), native_bytes=len(value))
            return record
        # Explicit fictional fixture construction precedes every receipt and
        # archive. No historical or native model artifact is modified here.
        with patch.object(fixture.audio_fixture, 'fixture_writer', writer):
            self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.changed = self.f.plan['cells'][1]['cell_id']
        folder = self.f.generation / 'cells' / self.changed
        raw = peak_wav(weak.float_wav_pcm((folder / 'native-f32.wav').read_bytes()), 1001)
        (folder / 'native-f32.wav').write_bytes(raw)
        completed = weak.loads((folder / 'completed.json').read_bytes())
        completed['audio'].update(native_sha256=weak.sha(raw), native_bytes=len(raw))
        (folder / 'completed.json').write_bytes(weak.canonical(completed) + b'\n')

    def create(self):
        f = self.f
        output = f.root / 'metadata-view'
        return output, context.create_view(f.inputs / 'inputs.json', f.input_sha, f.archive, f.asr, output,
            fsmn_tokens=f.tokens, selective_source_map=f.map_path, selective_policy=f.policy_path,
            selective_audit_sha256=weak.sha((f.asr / 'audit/final-summary.json').read_bytes()), **f.options())

    def test_original_reader_still_rejects_the_two_distinct_wav_containers(self):
        f = self.f
        with self.assertRaisesRegex(ValueError, 'CONTEXT_V3_ARCHIVE_NATIVE_CHANGED'):
            original.build_source_map(f.inputs / 'inputs.json', f.input_sha, f.archive, **f.options())
        mapping = context.build_source_map(f.inputs / 'inputs.json', f.input_sha, f.archive, **f.options())
        self.assertEqual(mapping, f.mapping)

    def test_additive_package_replays_both_originals_offline_with_unchanged_labels_and_outcomes(self):
        before = {path: path.read_bytes() for root in (self.f.generation, self.f.native, self.f.inputs, self.f.asr)
                  for path in root.rglob('*') if path.is_file()}
        output, report = self.create()
        self.assertEqual(report['schema'], context.SCHEMA)
        self.assertEqual(report['source_adapter']['reconciled_peak_timestamp_cell_ids'], [self.changed])
        self.assertEqual(report['statistics']['machine_candidates'], 6)
        self.assertEqual(report['statistics']['uncertain'], 2)
        self.assertEqual(report['statistics']['confirmed_speech_negative'], 2)
        changed = report['items'][1]
        self.assertEqual(changed['source']['native_container_relation']['relation'], 'peak-timestamp-plus-one-second')
        self.assertEqual(changed['decision']['full_text'], self.f.observed_text)
        self.assertNotEqual(changed['decision']['full_text'], changed['source']['intended_text'])
        failed = report['items'][0]
        self.assertEqual(failed['source']['generation_outcome'], 'failed-native')
        self.assertIsNone(failed['source']['native_container_relation'])
        self.assertEqual(failed['decision']['ctc_weight'], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        for root in (self.f.generation, self.f.native, self.f.inputs, self.f.asr): shutil.rmtree(root)
        self.f.plan_path.unlink(); self.f.recipe_path.unlink(); self.f.stage_path.unlink()
        self.assertEqual(context.verify_view(output), report)
        rows = list(context.load_training_rows(output)) + list(context.load_development_rows(output))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['ctc_text'] == row['full_text'] and row['ctc_text'] for row in rows))
        exported = next(row for row in rows if row['source_id'] == self.changed)
        self.assertEqual(exported['native_container_relation'], changed['source']['native_container_relation'])
        self.assertEqual(exported['source_adapter_schema'], context.SOURCE_ADAPTER)

    def test_rehashed_report_cannot_remove_or_change_the_reconciliation_relation(self):
        output, _ = self.create()
        report = weak.loads((output / 'report.json').read_bytes())
        report['items'][1]['source']['native_container_relation']['returned_peak_timestamp'] = 0
        (output / 'report.json').write_bytes(weak.canonical(report) + b'\n')
        self.f.rehash_package(output)
        with self.assertRaisesRegex(ValueError, 'WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE'):
            context.verify_view(output)

    def test_new_reader_cannot_relabel_original_reader_packages_as_its_own(self):
        folder = self.f.generation / 'cells' / self.changed
        raw = (self.f.native / self.changed / 'returned-native.wav').read_bytes()
        (folder / 'native-f32.wav').write_bytes(raw)
        completed = weak.loads((folder / 'completed.json').read_bytes())
        completed['audio']['native_sha256'] = weak.sha(raw)
        (folder / 'completed.json').write_bytes(weak.canonical(completed) + b'\n')
        output, report = self.f.create()
        self.assertEqual(original.verify_view(output), report)
        with self.assertRaisesRegex(ValueError, 'CONTEXT_V3_WEAK_VIEW_SCHEMA'):
            context.verify_view(output)


if __name__ == '__main__':
    unittest.main()
