"""Focused stdlib fixtures for the new intake boundaries; no numeric runtime."""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


fixtures = module(HERE/'test_english_source_identity.py', 'typed_intake_existing_fixtures')
core, identity = fixtures.c, fixtures.i
intake = core.typed_intake


def packet(role='train', tier='silver', unknown=False):
    row = fixtures.source(role=role)
    row.update(generation_status='success', request_sha256='1'*64,
               pcm_sha256='2'*64, parent_pcm_sha256='3'*64)
    corpus, admission, contract = fixtures.packet([row])
    run = {'repository': 'jiying2007/kws-data', 'run_id': 1001,
           'commit': 'f'*40, 'run_attempt': 1, 'event': 'push'}
    corpus.update(source_design_id=corpus['campaign_id'], run=run)
    allowed = admission['rows'][0]
    allowed.update(training_allowed=role == 'train', tier=tier,
                   training_weight=intake.TIERS[tier] if role == 'train' else 0.0)
    if tier == 'context_supported_weak':
        allowed.update(phonetic_evidence_scope='PARENT_FULL_INPUT_FIXED_CHILD_REGION',
                       child_independent_phone_silver=False)
    if unknown:
        allowed.update(supervision_allowed=False, training_allowed=False, tier='unknown',
                       research_class=None, training_class=None, training_weight=0.0)
    evidence = {'rows': [{'id': row['id'], 'fixture_only': True}]}
    binding = {
        'schema': intake.SCHEMA, 'campaign_id': corpus['campaign_id'],
        'producer_run': run, 'source_artifact': {'id': 2001, 'name': 'fixture-only-not-an-actual-corpus',
            'bytes': 1, 'sha256': 'a'*64},
        'planned_row_ids': [row['id']], 'source_identity_contract_sha256': intake.digest(contract),
        'corpus_sha256': intake.digest(corpus), 'admission_sha256': intake.digest(admission),
        'machine_evidence_sha256': intake.digest(evidence), 'admission_policy_sha256': 'b'*64,
        'legacy_libri_ids': ['239'], 'exclusion_evidence_sha256': 'c'*64,
        'excluded_identities': {'row_ids': ['fixture-consumed-probe-row'],
            'request_sha256': ['4'*64], 'pcm_sha256': ['5'*64], 'parent_pcm_sha256': ['6'*64]},
        'parent_context_weak_policy': tier == 'context_supported_weak'}
    return [corpus, admission, contract, evidence, copy.deepcopy(admission), binding]


def freeze(parts):
    # Fixture-only reviewed input: never constructs an actual campaign binding.
    binding = parts[-1]
    for name, value in (('corpus', parts[0]), ('admission', parts[1]),
                        ('source_identity_contract', parts[2]), ('machine_evidence', parts[3])):
        binding[name+'_sha256'] = intake.digest(value)
    parts[4] = copy.deepcopy(parts[1])
    return intake.digest(binding)


def verify(parts, expected=None):
    return intake.validate_packet(*parts, expected_binding_sha256=expected or intake.digest(parts[-1]),
                                  normalize_libri=core.libri_identity)


class IntakeBoundaries(unittest.TestCase):
    def test_valid_fixture_preserves_typed_identity_and_unknown_mask(self):
        for role in ('train', 'development', 'calibration', 'locked-test'):
            for unknown in (False, True):
                proof = verify(packet(role, unknown=unknown))
                self.assertEqual(proof['planned_row_count'], 1)
                self.assertFalse(proof['independent_human_speakers_claimed'])
                self.assertEqual(proof['actual_model_calls'], 0)

    def test_external_binding_cannot_be_self_replaced(self):
        parts = packet(); expected = intake.digest(parts[-1])
        parts[2]['train_side_exposed'] = ['kokoro-v1_0-preset:af_fixtureone']
        parts[0]['source_identity_contract_sha256'] = intake.digest(parts[2])
        parts[1]['source_identity_contract_sha256'] = intake.digest(parts[2])
        freeze(parts)
        with self.assertRaisesRegex(ValueError, 'EXTERNAL_FROZEN_INTAKE'):
            verify(parts, expected)
        with self.assertRaisesRegex(ValueError, 'EXTERNAL_FROZEN_INTAKE'):
            intake.validate_binding(parts[-1], None)

    def test_context32_context64_and_permanent_exclusions_fail_even_after_flag_rewrite(self):
        changes = [{'campaign_id': name} for name in identity.PROBE_CAMPAIGNS]
        changes += [{'schema': prefix+'fixture'} for prefix in identity.PROBE_SCHEMA_PREFIXES]
        changes += [{field: True} for field in identity.PERMANENT_EXCLUSION_FIELDS]
        for change in changes:
            parts = packet(); parts[0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'DIAGNOSTIC_PROBE'):
                identity.require_training_corpus(parts[0], parts[1])
        for location in (0, 1):
            parts = packet(); parts[location]['rows'][0]['permanently_excluded_from_training'] = True
            freeze(parts)
            with self.assertRaisesRegex(ValueError, 'PERMANENT_DIAGNOSTIC_ROW'):
                verify(parts)

    def test_renamed_probe_request_pcm_or_parent_is_still_excluded(self):
        for field, blocked in (('request_sha256', '4'*64), ('pcm_sha256', '5'*64), ('parent_pcm_sha256', '6'*64),
                               ('pcm_sha256', '6'*64), ('parent_pcm_sha256', '5'*64)):
            parts = packet(); parts[0]['rows'][0][field] = blocked; freeze(parts)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'CONSUMED_PROBE_BYTES'):
                verify(parts)

    def test_heldout_cannot_gain_optimizer_weight_or_training_class(self):
        for role in ('development', 'calibration', 'locked-test'):
            for field, value in (('training_weight', 1.0), ('training_class', 2), ('training_allowed', True)):
                parts = packet(role); parts[1]['rows'][0][field] = value; freeze(parts)
                with self.subTest(role=role, field=field), self.assertRaisesRegex(ValueError, 'HELDOUT_NEVER'):
                    verify(parts)

    def test_unknown_cannot_be_background_or_weak_training(self):
        for field, value in (('research_class', 0), ('training_class', 0), ('training_weight', 0.5),
                             ('training_allowed', True), ('tier', 'weak')):
            parts = packet(unknown=True); parts[1]['rows'][0][field] = value; freeze(parts)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'UNKNOWN_IS_MASKED'):
                verify(parts)

    def test_context_tier_is_explicit_and_never_silver(self):
        verify(packet(tier='context_supported_weak'))
        for mutation in ('tier', 'scope', 'independence', 'policy'):
            parts = packet(tier='context_supported_weak')
            if mutation == 'tier': parts[1]['rows'][0]['tier'] = 'silver'
            elif mutation == 'scope': parts[1]['rows'][0]['phonetic_evidence_scope'] = 'CHILD_INDEPENDENT'
            elif mutation == 'independence': parts[1]['rows'][0]['child_independent_phone_silver'] = True
            else: parts[-1]['parent_context_weak_policy'] = False
            freeze(parts)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'CONTEXT_WEAK|PARENT_PHONE'):
                verify(parts)

    def test_failed_unknown_row_kept_in_full_denominator(self):
        parts = packet(unknown=True); parts[0]['rows'][0]['generation_status'] = 'failed'; freeze(parts)
        verify(parts)
        parts[3]['rows'] = []; freeze(parts)
        with self.assertRaisesRegex(ValueError, 'COMPLETE_MACHINE_EVIDENCE_DENOMINATOR'):
            verify(parts)

    def test_missing_row_or_policy_recompute_disagreement_rejects(self):
        parts = packet(); parts[4]['rows'][0]['research_class'] = 1
        with self.assertRaisesRegex(ValueError, 'POLICY_RECOMPUTATION'):
            verify(parts)
        parts = packet(); parts[-1]['planned_row_ids'].append('fixture-missing')
        with self.assertRaisesRegex(ValueError, 'ALL_PLANNED_ROWS'):
            verify(parts)

    def test_source_run_cannot_be_rebound_to_different_execution(self):
        parts = packet(); parts[0]['run'] = dict(parts[0]['run'], run_id=9999); freeze(parts)
        with self.assertRaisesRegex(ValueError, 'BOUND_SOURCE_RUN'):
            verify(parts)

    def test_unbound_numeric_core_stops_before_any_numeric_import(self):
        parts = packet(); paths = {'CORPUS_MANIFEST.json': parts[0], 'ADMISSION.json': parts[1],
                                  'SOURCE_IDENTITY_CONTRACT.json': parts[2]}
        frozen = SimpleNamespace(loads=lambda value: value, regular=lambda path: paths[path.name])
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'torch', 'onnx', 'onnxruntime', 'causal_embedding'):
                raise AssertionError('NUMERICAL_IMPORT_IN_SOURCE_TEST')
            return original(name, *args, **kwargs)
        with patch.object(core, 'SOURCE_IDENTITY_CANDIDATE_ENABLED', True), \
             patch('builtins.__import__', side_effect=guarded_import), \
             self.assertRaisesRegex(ValueError, 'EXTERNAL_FROZEN_INTAKE'):
            core.train_and_evaluate(Path('/fixture-only-no-files'), frozen, None, None,
                intake_binding=parts[-1], machine_evidence=parts[3], recomputed_admission=parts[4])

    def test_tampered_helper_rejected_before_import(self):
        with patch.object(core.Path, 'is_file', return_value=True), \
             patch.object(core.Path, 'is_symlink', return_value=False), \
             patch.object(core.Path, 'stat', return_value=SimpleNamespace(st_size=20)), \
             patch.object(core.Path, 'read_bytes', return_value=b'raise RuntimeError("executed")'), \
             self.assertRaisesRegex(ValueError, 'PINNED_INTAKE_HELPER_BEFORE_IMPORT'):
            core._pinned_source_helper('english_source_identity.py', '0'*64, 'never_imported')


if __name__ == '__main__':
    unittest.main()
