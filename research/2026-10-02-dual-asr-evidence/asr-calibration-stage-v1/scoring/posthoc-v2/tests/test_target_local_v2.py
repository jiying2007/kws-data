#!/usr/bin/env python3
"""Invented fixtures only: no actual transcripts, audio, or human annotations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

v2 = module('target_local_v2', ROOT / 'calibrate.py')
v1 = module('baseline_v1', ROOT / 'baseline_v1.py')


def fixture(implementation):
    pairs = {'星': 'xīng', '河': 'hé', '开': 'kāi', '灯': 'dēng',
             '月': 'yuè', '光': 'guāng', '关': 'guān', '今': 'jīn',
             '天': 'tiān,tián', '晴': 'qíng', '朗': 'lǎng',
             '行': 'háng,xíng', '心': 'xīn', '合': 'hé', '凯': 'kǎi',
             '等': 'děng', '风': 'fēng', '雨': 'yǔ'}
    characters = {str(ord(key)): value for key, value in pairs.items()}
    phrases = {'星河': [['xīng'], ['hé']], '月光': [['yuè'], ['guāng']]}
    targets = ['星河开灯', '月光关灯']
    config = {'schema_version': implementation.SCHEMAS['rules'],
              'rule_version': implementation.RULE_VERSION,
              'target_order': targets,
              'lexicons': {name: {'sha256': hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest(),
                                  'entries': len(value)}
                          for name, value in [('characters', characters), ('phrases', phrases)]},
              'uncertainty_markers': list(implementation.REQUIRED_MARKERS)}
    return implementation.Rules(config, 'a' * 64, characters, phrases, targets)


class TargetLocalTests(unittest.TestCase):
    def setUp(self):
        self.rules = fixture(v2)
        self.baseline = fixture(v1)

    def derive(self, text, rules=None, **overrides):
        row = dict(raw_text=text, status='success', completeness='complete', quality_flags=[])
        row.update(overrides)
        return (rules or self.rules).derive(row)

    def state(self, text, **kwargs):
        return self.derive(text, **kwargs)['targets'][0]['presence']

    def candidates(self, text):
        return self.derive(text)['targets'][0]['rule_evidence']['target_local_phonetic_candidates']

    def test_irrelevant_heteronym_is_diagnostic_not_blocker(self):
        self.assertEqual(self.state('今天晴朗', rules=self.baseline), 'unknown')
        result = self.derive('今天晴朗')
        self.assertEqual([t['presence'] for t in result['targets']], ['negative', 'negative'])
        self.assertIn('lexicon_ambiguous_reading', result['lexical_diagnostics'])
        self.assertEqual(result['targets'][0]['blockers'], [])

    def test_irrelevant_oov_is_diagnostic_not_blocker(self):
        self.assertEqual(self.state('今Q晴朗', rules=self.baseline), 'unknown')
        result = self.derive('今Q晴朗')
        self.assertEqual([t['presence'] for t in result['targets']], ['negative', 'negative'])
        self.assertIn('lexicon_oov', result['lexical_diagnostics'])
        self.assertEqual(self.candidates('今Q晴朗'), [])

    def test_target_near_oov_is_unknown_not_negative(self):
        for text in ['星Q开灯', '行Q开灯', '星QQ灯', '星Q开']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'unknown')
                self.assertTrue(any(c['oov_positions'] for c in self.candidates(text)))

    def test_all_readings_preserved_even_matching_variant_is_last(self):
        self.assertEqual(self.state('行合开灯'), 'unknown')
        self.assertTrue(any(c['minimum_possible_syllable_edits'] == 0 and
                            c['ambiguous_reading_positions'] == [0]
                            for c in self.candidates('行合开灯')))
        tokens, _ = self.rules.tokenize('行')
        self.assertEqual(tokens[0]['readings'], ['háng', 'xíng'])

    def test_split_homophones_preserve_soft_and_hard_boundaries(self):
        for separator in [' ', '，', '。', '！', '；', '、', '\n', ', ', '...']:
            text = '行' + separator + '合开灯'
            with self.subTest(separator=separator):
                result = self.derive(text)
                self.assertEqual(result['targets'][0]['presence'], 'unknown')
                self.assertEqual(result['raw_text'], text)
                normalized = v2.normalize(text)
                candidates = self.candidates(text)
                self.assertTrue(any(c['boundary_positions'] for c in candidates))
                for candidate in candidates:
                    self.assertEqual(candidate['text'], normalized[candidate['start']:candidate['end']])
                    self.assertTrue(candidate['unknown_only'])
                    self.assertFalse(candidate['acoustic_tone_measured'])
                self.assertFalse(result['targets'][0]['rule_evidence']['literal_target_present'])

    def test_boundary_separated_literal_never_becomes_positive(self):
        for text in ['星，河开灯', '星。河。开。灯', '星 河 开 灯', '星，河开']:
            with self.subTest(text=text):
                evidence = self.derive(text)['targets'][0]
                self.assertEqual(evidence['presence'], 'unknown')
                self.assertFalse(evidence['rule_evidence']['literal_target_present'])

    def test_short_partial_and_inserted_syllable_are_unknown_only(self):
        for text in ['行合开', '行，合开', '星Q开', '行合雨开灯']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'unknown')
                self.assertTrue(self.candidates(text))
                self.assertTrue(all(c['minimum_possible_syllable_edits'] <= 1 for c in self.candidates(text)))

    def test_exact_literal_with_far_ambiguity_stays_positive(self):
        for text in ['星河开灯今天晴朗', '今Q晴朗，星河开灯', '星河开灯行']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'positive')

    def test_genuine_homophone_and_one_edit_candidates_preserved(self):
        for text in ['心合开灯', '星河开等', '星河凯灯', '星河开', '星河雨开灯']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text, rules=self.baseline), 'unknown')
                self.assertEqual(self.state(text), 'unknown')

    def test_no_lexical_evidence_is_unknown_even_short_oov(self):
        for text in ['Q', 'QQQQ', '9', '🌙']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'unknown')
                self.assertTrue(self.derive(text)['targets'][0]['rule_evidence']['insufficient_lexical_evidence'])

    def test_execution_quality_markers_override_literal(self):
        overrides = [{'status': status} for status in ['error', 'timeout', 'not_run']]
        overrides += [{'completeness': value} for value in ['incomplete', 'unknown']]
        overrides += [{'quality_flags': [flag]} for flag in v2.FLAGS]
        for values in overrides:
            with self.subTest(values=values):
                self.assertEqual(self.state('星河开灯', **values), 'unknown')
        for marker in v2.REQUIRED_MARKERS:
            with self.subTest(marker=marker):
                self.assertEqual(self.state('星河开灯' + marker), 'unknown')
        for text in ['<|zh|>星河开灯', 'language Chinese<asr_text>星河开灯', '星河开灯\u202e', '', ' \n\t']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'unknown')

    def test_target_lexicon_still_requires_unambiguous_readings(self):
        characters = copy.deepcopy(self.rules.characters)
        characters[str(ord('灯'))] = 'dēng,dèng'
        with self.assertRaises(v2.ContractError):
            v2.Rules(self.rules.config, self.rules.sha256, characters, self.rules.phrases, self.rules.targets)

    def test_v1_candidate_fields_retained(self):
        for text in ['星河开灯', '星，河开灯', '心合开灯', '星河开等', '星河开', '行合开灯']:
            with self.subTest(text=text):
                old = self.derive(text, rules=self.baseline)['targets'][0]['rule_evidence']
                new = self.derive(text)['targets'][0]['rule_evidence']
                for key in ['literal_target_present', 'one_edit_text_candidates',
                            'boundary_separated_target_candidate', 'dictionary_phonetic_candidates']:
                    self.assertEqual(new[key], old[key])
                if old['confusable_or_partial_target_candidate']:
                    self.assertTrue(new['confusable_or_partial_target_candidate'])

    def test_independent_target_decisions(self):
        result = self.derive('行合开灯今天晴朗')
        self.assertEqual([t['presence'] for t in result['targets']], ['unknown', 'negative'])

    def test_unambiguous_far_and_ambiguous_far_stay_negative_across_boundary(self):
        for text in ['今晴风雨', '今天，晴朗', '今天。晴朗', '今Q，晴朗']:
            with self.subTest(text=text):
                self.assertEqual(self.state(text), 'negative')

    def test_wildcard_never_becomes_dictionary_tone_measurement(self):
        evidence = self.derive('行Q开灯')['targets'][0]['rule_evidence']
        self.assertTrue(evidence['target_local_phonetic_candidates'])
        self.assertEqual(evidence['dictionary_phonetic_candidates'], [])
        self.assertFalse(evidence['pinyin_is_acoustic_measurement'])

    def test_minimum_set_distance_preserves_threshold(self):
        target = [{'xing'}, {'he'}, {'kai'}, {'deng'}]
        for text, expected in [('行合开灯', 0), ('心合开灯', 1), ('今晴开灯', 2),
                               ('行合开', 1), ('行合雨开灯', 1), ('行Q开灯', 0)]:
            with self.subTest(text=text):
                tokens, _ = self.rules.tokenize(text)
                self.assertEqual(v2.possible_phonetic_edit_distance(tokens, target), expected)

    def test_rule_version_is_explicitly_posthoc(self):
        self.assertEqual(v2.RULE_VERSION, 'posthoc-target-local-text-evidence-v2')
        self.assertNotEqual(v2.RULE_VERSION, v1.RULE_VERSION)

    def test_join_declares_posthoc_scope_and_preserves_unknown_gold(self):
        models = [{'model_id': 'fixture-only-' + slot, 'revision': slot * 40,
                   'run_id': 'synthetic-' + slot} for slot in ['a', 'b']]
        manifest = {'schema_version': v2.SCHEMAS['manifest'], 'probe_set_id': 'synthetic-local-v2',
                    'execution_kind': 'synthetic_fixture', 'target_order': self.rules.targets,
                    'models': models, 'records': [{'recording_id': 'synthetic-only',
                    'wav_sha256': 'c' * 64, 'human_target_presence': ['unknown', 'unknown']}]}
        runs = [{'schema_version': v2.SCHEMAS['input'], 'execution_kind': 'synthetic_fixture',
                 'manifest_sha256': 'b' * 64, 'rules_sha256': self.rules.sha256, 'model': model,
                 'records': [{'recording_id': 'synthetic-only', 'wav_sha256': 'c' * 64,
                              'raw_text': '今天晴朗', 'status': 'success',
                              'completeness': 'complete', 'quality_flags': []}]} for model in models]
        result = v2.join(manifest, 'b' * 64, self.rules, *runs)
        self.assertEqual(result['evaluation_scope'], 'post_hoc_diagnostic_reused_outputs')
        self.assertTrue(any('not untouched-holdout qualification' in line for line in result['limitations']))
        self.assertEqual([t['presence'] for t in result['records'][0]['weak_model_consensus']],
                         ['negative', 'negative'])
        self.assertEqual([t['presence'] for t in result['records'][0]['final_target_presence']],
                         ['unknown', 'unknown'])
        self.assertFalse(result['inference_performed_by_this_tool'])
        self.assertFalse(result['wav_bytes_verified_by_this_tool'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
