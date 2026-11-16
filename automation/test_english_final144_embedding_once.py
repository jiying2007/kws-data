"""Synthetic source/policy fixtures only; no real model inference or training."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_final144_embedding_once as worker


def fixtures():
    cells, rows = [], []
    for voice in ('aiden', 'dylan', 'eric', 'serena', 'ryan', 'vivian'):
        role = 'development' if voice in ('ryan', 'vivian') else 'train'
        for number in range(24):
            label = 1 if number < 8 else 2 if number < 16 else 0
            cid = 'invented-' + voice + '-' + str(number)
            name = {1: 'Hey Vosen', 2: 'Hey Mero', 0: None}[label]
            text = name or "Let's have dinner at seven."
            cell = {'cell_id': cid, 'voice_id': voice, 'source_family_id': 'qwen3-preset-' + voice,
                    'role': role, 'requested_keyword_index': label or None, 'canonical_phrase': name,
                    'intent': 'wake-candidate' if label else 'ordinary-negative-candidate',
                    'synthesis_text': text, **dict.fromkeys(worker.FALSE_FLAGS, False)}
            tokens = {1: ['h', 'eɪ', 'v', 'oʊ', 'z', 'ə', 'n'],
                      2: ['h', 'eɪ', 'm', 'iː', 'ɹ', 'oʊ'], 0: ['l', 'ɛ', 't', 's']}[label]
            rows.append({**cell, 'status': 'GENERATED_UNADMITTED',
                'pcm_sha256': worker.sha(cid.encode()), 'wav_sha256': worker.sha(('wav/' + cid).encode()),
                'transcripts': dict.fromkeys(worker.PROVIDERS[:3], text), 'raw_ipa_tokens': tokens[:]})
            cells.append(cell)
    binding = {'test_backend': False, 'observer_summaries_sha256': {
        key: worker.sha(('invented-summary/' + key).encode()) for key in worker.PROVIDERS}}
    return {'cells': cells}, rows, binding


def guard_fixture():
    batch = {'next_main_before': 'a' * 40, 'recipe_sha256': 'b' * 64}
    pins = {'schema': worker.PINS_SCHEMA, 'enabled': True, 'main_before': 'a' * 40,
        'pipeline_commit': 'c' * 40, 'batch_pins_sha256': worker.sha(worker.canonical(batch)),
        'selection_sha256': worker.sha(worker.canonical(worker.SELECTION)), 'recipe_sha256': 'b' * 64,
        'optimizer_updates': 96, 'max_source_cells': 144, 'source_labels_are_intents_only': True,
        'no_retry': True, **dict.fromkeys(worker.FALSE_FLAGS, False)}
    env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
           'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '123', 'GITHUB_SHA': 'd' * 40}
    event = {'before': 'a' * 40, 'after': 'd' * 40,
             'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
    return pins, batch, env, event


class WorkerBoundaryTests(unittest.TestCase):
    def test_frozen_asr_and_phone_rules_keep_every_source_row(self):
        recipe, rows, binding = fixtures()
        result = worker.select_intents(recipe, rows, binding)
        self.assertEqual((result['source_denominator'], len(result['decisions']), len(result['eligible_cell_ids'])), (144, 144, 144))
        self.assertTrue(all(not decision['pronunciation_verified'] and not decision['trusted_negative']
                            for decision in result['decisions']))
        self.assertTrue(all(observation['observer_artifact_identity_kind'] ==
            'FOUR_ORIGINAL_SUMMARY_SHA256_MANIFEST' for observation in result['observations'].values()))

    def test_normalization_is_only_declared_text_operations(self):
        self.assertEqual(worker.normalized("LET’S have dinner, at seven!"), worker.normalized("Let's have dinner at seven."))
        self.assertEqual(worker.normalized(' Hey, MIRO! '), ['hey', 'miro'])
        self.assertNotEqual(worker.normalized('Héy Miro'), ['hey', 'miro'])
        self.assertNotEqual(worker.normalized('Hey Miro please'), ['hey', 'miro'])

    def test_two_exact_word_votes_and_contiguous_core_required(self):
        recipe, rows, binding = fixtures()
        mero = rows[8]
        mero['transcripts'] = {'qwen': 'Hey, Miro!', 'whisper': 'Hey Meero.', 'parakeet': 'something else'}
        mero['raw_ipa_tokens'] = ['m', 'i', 'r', 'o']
        result = worker.select_intents(recipe, rows, binding)
        self.assertIn(mero['cell_id'], result['eligible_cell_ids'])
        self.assertIn('LENGTH_STRESS_UNVERIFIED', result['observations'][mero['cell_id']]['phone_review_status'])
        mero['transcripts']['whisper'] = 'Hey Meero please'
        self.assertNotIn(mero['cell_id'], worker.select_intents(recipe, rows, binding)['eligible_cell_ids'])
        mero['transcripts']['whisper'] = 'Hey Meero'
        mero['raw_ipa_tokens'] = ['m', 'i', 'x', 'r', 'o']
        self.assertNotIn(mero['cell_id'], worker.select_intents(recipe, rows, binding)['eligible_cell_ids'])

    def test_specific_vowel_disagreements_are_never_negative_labels(self):
        recipe, rows, binding = fixtures()
        rows[0]['raw_ipa_tokens'] = ['v', 'oʊ', 'z', 'ɪ', 'n']
        rows[8]['raw_ipa_tokens'] = ['m', 'ɪ', 'ɹ', 'oʊ']
        before = copy.deepcopy(rows)
        result = worker.select_intents(recipe, rows, binding)
        self.assertEqual(rows, before)
        self.assertEqual([result['decisions'][i]['evidence_status'] for i in (0, 8)], ['UNKNOWN', 'UNKNOWN'])
        self.assertEqual([result['decisions'][i]['weak_intent_class'] for i in (0, 8)], [1, 2])

    def test_negative_intent_requires_matching_transcript_and_no_target_core(self):
        recipe, rows, binding = fixtures()
        negative = rows[16]
        negative['raw_ipa_tokens'] = ['m', 'iː', 'ɹ', 'oʊ']
        result = worker.select_intents(recipe, rows, binding)
        self.assertNotIn(negative['cell_id'], result['eligible_cell_ids'])
        self.assertEqual(result['observations'][negative['cell_id']]['evidence_status'], 'STRONG_CONFLICT')
        negative['raw_ipa_tokens'] = ['l', 'ɛ', 't', 's']
        negative['transcripts']['qwen'] = 'I do not know'
        negative['transcripts']['whisper'] = 'I do not know'
        self.assertNotIn(negative['cell_id'], worker.select_intents(recipe, rows, binding)['eligible_cell_ids'])

    def test_missing_observation_duplicate_source_and_simulation_rejected(self):
        for change in ('missing_asr', 'duplicate', 'missing_observer_digest', 'simulation', 'source_admitted'):
            recipe, rows, binding = fixtures()
            if change == 'missing_asr':
                del rows[0]['transcripts']['qwen']
            elif change == 'duplicate':
                rows[1] = copy.deepcopy(rows[0])
            elif change == 'missing_observer_digest':
                del binding['observer_summaries_sha256']['phones']
            elif change == 'simulation':
                binding['test_backend'] = True
            else:
                rows[0]['training_admitted'] = True
            with self.assertRaises(ValueError, msg=change):
                worker.select_intents(recipe, rows, binding)

    def test_post_dedup_minimum_requires_real_split_and_source_diversity(self):
        recipe, rows, _ = fixtures()
        selected = [{**r, 'weak_class': r['requested_keyword_index'] or 0} for r in rows]
        self.assertTrue(worker.minimum_evidence(selected)['passed'])
        one_family = [r for r in selected if not (r['role'] == 'train' and r['weak_class'] == 1 and r['voice_id'] != 'aiden')]
        check = worker.minimum_evidence(one_family)
        self.assertEqual(check['counts']['train/1']['count'], 8)
        self.assertFalse(check['passed'])
        too_few = [r for r in selected if not (r['role'] == 'development' and r['weak_class'] == 2)]
        self.assertFalse(worker.minimum_evidence(too_few)['passed'])
        self.assertFalse(worker.minimum_evidence([])['passed'])

    def test_guard_requires_frozen_pins_and_exact_first_public_push(self):
        worker.execution_guard(*guard_fixture())
        for change in ('disabled', 'before', 'attempt', 'private', 'fork', 'budget', 'selection', 'promotion'):
            pins, batch, env, event = guard_fixture()
            if change == 'disabled': pins['enabled'] = False
            elif change == 'before': event['before'] = 'e' * 40
            elif change == 'attempt': env['GITHUB_RUN_ATTEMPT'] = '2'
            elif change == 'private': event['repository']['private'] = True
            elif change == 'fork': env['GITHUB_REPOSITORY'] = 'other/kws-data'
            elif change == 'budget': pins['optimizer_updates'] = 97
            elif change == 'selection': pins['selection_sha256'] = 'e' * 64
            else: pins['model_promotion_allowed'] = True
            with self.assertRaises(ValueError, msg=change):
                worker.execution_guard(pins, batch, env, event)

    def test_source_checks_never_import_ml_or_trigger_training(self):
        self.assertFalse({'numpy', 'torch', 'onnxruntime', 'soundfile'} & set(sys.modules))
        self.assertEqual(worker.SELECTION['optimizer_updates'], 96)
        self.assertFalse(worker.SELECTION['development_tuning'])


if __name__ == '__main__':
    unittest.main()
