"""Pure stdlib source/data contracts, with artificial PCM only."""
import copy
import importlib.util
from pathlib import Path
import struct
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('coverage_data', HERE / 'english_coverage_quality_data.py')
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def pcm(values):
    return struct.pack('<' + 'h' * len(values), *values)


class CoverageDataContracts(unittest.TestCase):
    def tracks(self):
        return [{'id': role + '-' + str(i), 'role': role, 'frames': 256000,
                 'pcm_sha256': data.sha((role + str(i)).encode())}
                for role in ('train', 'development') for i in range(10)]

    def test_frozen_denominators_and_call_ceilings(self):
        self.assertEqual(data.PURPOSES['train']['planned_conditions'], 100)
        self.assertEqual(data.PURPOSES['development_challenge']['planned_conditions'], 27)
        self.assertEqual(sum(x['maximum_new_embedding_calls'] for x in data.PURPOSES.values()), 2635)
        self.assertEqual(sum(x['planned_rows'] for x in data.PURPOSES.values()), 4667)

    def test_new_conditions_never_change_the_old_domain(self):
        self.assertNotEqual(data.DOMAIN, data.shared.NOISE_SELECTION_DOMAIN)
        self.assertEqual(data.shared.NOISE_SELECTION_DOMAIN, 'english-quality-snr10-v1/')

    def test_new_train_pair_uses_two_distinct_same_role_tracks(self):
        tracks = self.tracks()
        old = 'train-0'
        first, start1 = data.select_noise('source-a', 'train', 'new5', 20000, tracks, {old})
        second, start2 = data.select_noise('source-a', 'train', 'new15', 20000, tracks, {old, first['id']})
        self.assertEqual(len({old, first['id'], second['id']}), 3)
        self.assertEqual((first['role'], second['role']), ('train', 'train'))
        self.assertTrue(0 <= start1 <= 236000 and 0 <= start2 <= 236000)

    def test_challenge_is_development_only(self):
        chosen, _ = data.select_noise('source-a', 'development_challenge', 'new8', 20000,
                                       self.tracks(), {'development-0'})
        self.assertEqual(chosen['role'], 'development')
        with self.assertRaises(ValueError):
            data.select_noise('a', 'train', 'new8', 20, self.tracks(), set())

    def test_selection_is_stable_and_condition_id_binds_parameters(self):
        args = ('source-a', 'train', 'new5', 20000, self.tracks(), {'train-0'})
        track, offset = data.select_noise(*args)
        self.assertEqual((track, offset), data.select_noise(*args))
        first = data.condition_id('source-a', 'train', 'new5', 'a' * 64, track, offset)
        second = data.condition_id('source-a', 'train', 'new5', 'a' * 64, track, offset + 1)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, 'source-a')

    def test_missing_same_role_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            data.select_noise('a', 'train', 'new5', 20,
                              [r for r in self.tracks() if r['role'] == 'development'], set())

    def test_three_fixed_mix_snr_values_keep_signal_gain(self):
        signal, noise = pcm([1000, -1000] * 128), pcm([500, 500] * 128)
        for snr in (5.0, 8.0, 15.0):
            mixed, receipt = data.mix_pcm16(signal, noise, snr)
            self.assertIsNotNone(mixed)
            self.assertEqual(receipt['status'], 'MIXED')
            self.assertLessEqual(abs(receipt['actual_snr_db'] - snr), 0.05)
            self.assertEqual(receipt['gain_on_signal'], 1.0)
            self.assertEqual(receipt['clipped_samples'], 0)
        with self.assertRaises(ValueError):
            data.mix_pcm16(signal, noise, 7.0)

    def test_overflow_excluded_without_repair(self):
        result, receipt = data.mix_pcm16(pcm([32000] * 32), pcm([30000] * 32), 5.0)
        self.assertIsNone(result)
        self.assertEqual(receipt['status'], 'MIX_HEADROOM_EXCLUDED')
        self.assertGreater(receipt['unbounded_peak'], 32767)
        self.assertEqual(receipt['gain_on_signal'], 1.0)

    def test_zero_energy_excluded(self):
        result, receipt = data.mix_pcm16(pcm([1000] * 32), pcm([0] * 32), 15.0)
        self.assertIsNone(result)
        self.assertEqual(receipt['status'], 'MIX_ZERO_ENERGY_EXCLUDED')

    def test_prefix_exact_input_windows_eligible_for_reuse(self):
        clean = pcm([50] * 35200 + [1000] * 8000 + [-200] * 5120)
        changed = pcm([50] * 35200 + [1100] * 8000 + [-200] * 5120)
        windows = data.prefix_reuse_windows(clean, changed)
        self.assertEqual(len(windows), 16)
        self.assertEqual(windows[0]['available_sample'], 16000)
        self.assertEqual(windows[-1]['available_sample'], 35200)
        self.assertTrue(all(x['input_pcm_bytes'] == 32000 for x in windows))

    def test_prefix_change_or_shape_change_cannot_reuse_cache(self):
        raw = pcm([40] * 50000)
        for changed in (b'\x00\x00' + raw[2:], raw[:-2]):
            with self.assertRaises(ValueError):
                data.prefix_reuse_windows(raw, changed)

    def test_train_coverage_requires_families_and_negative_templates(self):
        rows = []
        for condition in ('new5', 'new15'):
            for family in range(4):
                for text in range(8):
                    rows.append({'condition': condition, 'weak_class': 0,
                        'parent_id': f'n-{family}-{text}', 'mix': {
                            'source_family_id': str(family), 'source_synthesis_text': str(text)}})
            for label in (1, 2):
                for family in range(2):
                    rows.append({'condition': condition, 'weak_class': label,
                        'parent_id': f'p-{label}-{family}', 'mix': {
                            'source_family_id': str(family), 'source_synthesis_text': str(label)}})
        self.assertTrue(data.training_coverage(rows)['passed'])
        wrong = [r for r in rows if not (r['condition'] == 'new5' and r['weak_class'] == 1
                                        and r['mix']['source_family_id'] == '1')]
        self.assertFalse(data.training_coverage(wrong)['passed'])
        wrong = [r for r in rows if not (r['condition'] == 'new15' and r['weak_class'] == 0
                                        and r['mix']['source_synthesis_text'] == '7')]
        self.assertFalse(data.training_coverage(wrong)['passed'])


if __name__ == '__main__':
    unittest.main()
