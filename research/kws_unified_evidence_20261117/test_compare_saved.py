"""Synthetic metadata/discrete-event counterexamples; zero audio/model execution."""
import copy
import json
from pathlib import Path
import unittest

import compare_saved as c


class SavedEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.existing_helpers(Path(__file__).parent / 'sources/pipeline')

    def fixture(self):
        row = dict(id='synthetic-row', source_identity=['a' * 64, 'source'], pcm_identity='b' * 64,
                   role='exposed_development', expected_class=1, previously_exposed=True, independent_holdout=False,
                   label_authority='requested-intent-only', target_profile_identity='history-A', clock={'sample_rate_hz': 16000},
                   observation={'requested_match': False})
        contract = dict(language='en', target_profile_identity='history-A', dataset_identity='c' * 64,
                        label_semantics='weak', metric_unit='observed_context_weak_event', scoring_identity='d' * 64,
                        threshold_selection_identity='fixed-before-this-observation', condition='synthetic-fixture', class_map=c.WORDS)
        model = dict(id='fixture-left', architecture='synthetic', weights_sha256='e' * 64, frontend={'mel': 32})
        left = c.card(model, contract, [row], {})
        right = copy.deepcopy(left)
        right['model']['id'] = 'fixture-right'
        return left, right

    def test_same_rows_pair_without_formal_promotion(self):
        a, b = self.fixture()
        b['records'][0]['observation']['requested_match'] = True
        result = c.pair(a, b)
        self.assertEqual(result['status'], 'PAIRED_RESEARCH_DIAGNOSTIC')
        self.assertFalse(result['formal_ranking_allowed'])
        self.assertIsNone(result['selected_winner'])
        self.assertEqual(result['paired_changes']['exposed_development/1']['improved_requested_match'], 1)

    def test_different_frontend_and_architecture_are_system_comparable(self):
        a, b = self.fixture()
        b['model']['frontend'] = {'mel': 80}
        b['model']['architecture'] = 'different-fixture'
        result = c.pair(a, b)
        self.assertEqual(result['status'], 'PAIRED_RESEARCH_DIAGNOSTIC')
        self.assertNotEqual(result['frontends']['left'], result['frontends']['right'])

    def test_contract_mismatches_do_not_become_a_ranking(self):
        for key, changed in [('target_profile_identity', 'history-B'), ('dataset_identity', 'f' * 64),
                             ('label_semantics', 'gold'), ('metric_unit', 'whole_clip_weak_intent_class'),
                             ('scoring_identity', 'other-matcher'), ('threshold_selection_identity', 'tuned-on-test'),
                             ('condition', 'different-SNR'), ('class_map', {'Hey Mero': 1, 'Hey Vosen': 2, 'background': 0})]:
            with self.subTest(key=key):
                a, b = self.fixture()
                b['contract'][key] = changed
                result = c.pair(a, b)
                self.assertEqual(result['status'], 'BLOCKED_INCOMPATIBLE')
                self.assertNotIn('paired_changes', result)

    def test_row_split_pcm_exposure_and_clock_mismatches_reject(self):
        for key, changed in [('role', 'train'), ('pcm_identity', 'f' * 64), ('previously_exposed', False),
                             ('source_identity', ['other']), ('expected_class', 2), ('clock', {'sample_rate_hz': 8000}),
                             ('target_profile_identity', 'history-B')]:
            with self.subTest(key=key):
                a, b = self.fixture()
                b['records'][0][key] = changed
                self.assertEqual(c.pair(a, b)['status'], 'BLOCKED_INCOMPATIBLE')

    def test_exposed_cannot_be_renamed_independent_holdout(self):
        a, b = self.fixture()
        b['records'][0]['independent_holdout'] = True
        with self.assertRaisesRegex(ValueError, 'holdout'):
            c.pair(a, b)

    def test_no_silent_intersection(self):
        a, b = self.fixture()
        b['records'][0]['id'] = 'a-different-record'
        result = c.pair(a, b)
        self.assertEqual(result['status'], 'BLOCKED_INCOMPATIBLE')
        self.assertNotIn('paired_changes', result)

    def test_duplicate_ids_reject(self):
        a, b = self.fixture()
        b['records'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            c.pair(a, b)

    def test_independent_ranking_is_not_exposed_diagnostic(self):
        a, b = self.fixture()
        with self.assertRaisesRegex(ValueError, 'independent ranking'):
            c.pair(a, b, use='independent_event_ranking')

    def test_json_duplicate_and_nonfinite_reject(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                c.parse(text)

    def event_fixture(self):
        roster = dict(id='synthetic-row', role='development', weak_class=1, observed_samples=60000,
                      original_clip_eof_sample=52000, source={'clip_start_sample': 35200})
        event = {key: value for key, value in roster.items() if key != 'source'}
        event.update(clip_start_sample=35200, independent_acoustic_evaluation=False, word_end_latency='NOT_MEASURED',
                     events=[{'keyword': 1, 'sample': 55000}], matched_intent_after_clip_start=True,
                     prefix_events=[], wrong_keyword_after_clip_start=[], extra_same_keyword_events=0,
                     weak_negative_intent_events=0, no_matching_event_in_observed_context=False)
        return event, roster

    def test_event_recount_preserves_censoring_and_clock(self):
        event, roster = self.event_fixture()
        result = c.event_observation(event, roster)
        self.assertTrue(result['requested_match'])
        self.assertFalse(result['censored_no_match_not_FRR'])
        self.assertNotIn('frr', result)
        self.assertNotIn('far_per_hour', result)

    def test_saved_event_counters_cannot_override_raw_events(self):
        event, roster = self.event_fixture()
        event['matched_intent_after_clip_start'] = False
        with self.assertRaisesRegex(ValueError, 'matched count'):
            c.event_observation(event, roster)

    def test_global_clock_or_wrong_clip_join_reject(self):
        for change in ('clock', 'id'):
            event, roster = self.event_fixture()
            if change == 'clock':
                event['events'][0]['sample'] += 1000000
            else:
                event['id'] = 'wrong-id'
            with self.assertRaises(ValueError):
                c.event_observation(event, roster)

    def test_retained_capsule_reproduces_exact_report_bytes(self):
        package = Path(__file__).parent
        report = c.build_capsule(package)
        raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        self.assertEqual(raw, (package / 'READBACK.json').read_bytes())
        self.assertEqual(report['model_calls'], 0)
        self.assertFalse(report['formal_ranking_allowed'])


if __name__ == '__main__':
    unittest.main()
