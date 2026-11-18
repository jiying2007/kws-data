"""Pure stdlib/mock candidate checks; no future source selection or model run.

Names containing 'fixture' and their declared edges are invented test data,
never assertions about real presets or a proposed future corpus.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASELINE = HERE / 'english_synthetic192_training_core.py'
BASELINE_SHA = '0896d375f47318adbc42f63835cb85fc77441ae2472d9df8212dfca4dd7e62c0'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


c = module(HERE/'english_typed_synthetic_training_core.py', 'fixture_identity_core')
i = c.source_identity
old = module(BASELINE, 'fixture_immutable_old_core')


def source(ident='af_fixtureone', namespace='kokoro-v1_0-preset', role='train',
           group=None, rid='fixture-row-one'):
    if group is None:
        group = i.source_key(namespace, ident, c.libri_identity)
    return {'id': rid, 'source_namespace': namespace, 'source_id': ident,
        'source_group_id': group, 'role': role, 'diagnostic_only': False,
        'human_gold': False, 'source_speaker_id': None if namespace == 'kokoro-v1_0-preset' else ident,
        'preset_id': ident if namespace == 'kokoro-v1_0-preset' else None,
        'human_speaker_identity': None, 'human_speaker_identity_verified': False,
        'fixture_only': True}


def relationship(left, right, relation='mix_parent'):
    evidence = ('Fictional fixture relationship only: '+left+' -> '+right).encode()
    return {'left': left, 'right': right, 'relation': relation,
        'evidence_source': 'fixture-only/relationship.txt',
        'evidence_sha256': hashlib.sha256(evidence).hexdigest()}


def packet(rows, edges=(), exposures=()):
    contract = {'schema': i.SCHEMA, 'purpose': 'synthetic_training_source_identity',
        'diagnostic_only': False, 'training_allowed': True,
        'human_speaker_identity_claimed': False,
        'source_namespaces': sorted({r['source_namespace'] for r in rows}),
        'known_lineage_edges': copy.deepcopy(list(edges)), 'train_side_exposed': list(exposures)}
    corpus = {'schema': 'fixture-only-synthetic-corpus', 'campaign_id': 'fixture-only-campaign',
        'diagnostic_only': False, 'training_allowed': True,
        'source_identity_contract_sha256': i.digest(contract), 'rows': copy.deepcopy(rows)}
    admission = {'training_snapshot_allowed': True,
        'source_identity_contract_sha256': i.digest(contract),
        'rows': [dict(r, supervision_allowed=True, research_class=2,
            training_class=2 if r['role'] == 'train' else None,
            training_weight=1.0 if r['role'] == 'train' else 0.0,
            tier='silver', pronunciation_hypothesis='fixture-only-hypothesis',
            matched_pronunciation_hypothesis='fixture-only-hypothesis') for r in rows]}
    return corpus, admission, contract


def verify(parts, legacy=()):
    return i.validate_source_identities(*parts, legacy, normalize_libri=c.libri_identity)


class NamespaceIsolation(unittest.TestCase):
    def test_original_libri_normalization_kept(self):
        for value in ('3615', 'p3615', 'libritts-speaker-3615', 'librispeech:3615',
                      'piper-libritts-3615', 'librispeech-dev-clean-speaker-3615', 'libri-speaker-03615'):
            self.assertEqual(c.libri_identity(value), '3615')
            with self.assertRaisesRegex(ValueError, 'CROSS_LIBRITTS_LIBRISPEECH'):
                verify(packet([source(value, 'libri-speaker')]), ('3615',))

    def test_vctk_p239_does_not_become_libri239(self):
        result = verify(packet([source('p239', 'vctk-speaker')]), ('239',))
        self.assertEqual(result['rows']['fixture-row-one']['source_identity_key'], 'vctk-speaker:239')
        with self.assertRaisesRegex(ValueError, 'CROSS_LIBRITTS_LIBRISPEECH'):
            verify(packet([source('p239', 'libri-speaker')]), ('239',))

    def test_probe_preset_can_be_typed_train_source_without_fake_number(self):
        result = verify(packet([source('af_bella')]), ('239',))
        meta = result['rows']['fixture-row-one']
        self.assertEqual(meta['source_identity_key'], 'kokoro-v1_0-preset:af_bella')
        self.assertEqual(meta['preset_id'], 'af_bella')
        self.assertIsNone(meta['human_speaker_identity'])
        self.assertFalse(meta['human_speaker_identity_verified'])

    def test_missing_unknown_or_mislabeled_namespace_fails(self):
        for namespace, ident in ((None, 'af_bella'), ('unknown', 'p239'),
                                 ('libri-speaker', 'af_bella'), ('kokoro-v1_0-preset', '239')):
            with self.subTest(namespace=namespace, ident=ident), self.assertRaises(ValueError):
                i.source_key(namespace, ident, c.libri_identity)

    def test_preset_cannot_claim_numeric_or_authenticated_human(self):
        for field, value in (('source_speaker_id', '239'), ('human_speaker_identity', 'person239'),
                              ('human_speaker_identity_verified', True), ('human_source_identity_verified', True)):
            row = source('af_bella'); row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'NOT_AUTHENTICATED_HUMAN'):
                verify(packet([row]))


class LineageAndExposure(unittest.TestCase):
    def test_all_four_probe_presets_are_train_side_even_when_contract_omits_exposure(self):
        for preset in ('af_bella', 'af_sarah', 'am_michael', 'am_puck'):
            for role in ('development', 'calibration', 'locked-test'):
                with self.subTest(preset=preset, role=role), self.assertRaisesRegex(ValueError, 'TRAIN_SIDE_PRESET_LINEAGE'):
                    verify(packet([source(preset, role=role)]))

    def test_exposed_preset_ancestor_and_descendant_components_stay_train_side(self):
        edges = [relationship('kokoro-v1_0-preset:af_fixtureblend', 'kokoro-v1_0-preset:af_bella'),
                 relationship('kokoro-v1_0-preset:af_bella', 'kokoro-v1_0-preset:af_fixtureparent')]
        for preset in ('af_fixtureblend', 'af_fixtureparent'):
            with self.subTest(preset=preset), self.assertRaisesRegex(ValueError, 'TRAIN_SIDE_PRESET_LINEAGE'):
                verify(packet([source(preset, role='locked-test')], edges))

    def test_shared_parent_mixtures_cannot_cross_roles(self):
        edges = [relationship('kokoro-v1_0-preset:af_fixtureone', 'kokoro-v1_0-preset:af_fixtureparent'),
                 relationship('kokoro-v1_0-preset:af_fixturetwo', 'kokoro-v1_0-preset:af_fixtureparent')]
        rows = [source(), source('af_fixturetwo', role='development', rid='fixture-two')]
        with self.assertRaisesRegex(ValueError, 'LINEAGE_CANNOT_CROSS'):
            verify(packet(rows, edges))

    def test_one_component_must_use_one_actual_group_for_weights_and_minima(self):
        edges = [relationship('kokoro-v1_0-preset:af_fixtureone', 'kokoro-v1_0-preset:af_fixturetwo')]
        rows = [source(), source('af_fixturetwo', rid='fixture-two', group='fixture-group-two')]
        with self.assertRaisesRegex(ValueError, 'ONE_LINEAGE_ONE_CONSUMED_SOURCE_GROUP'):
            verify(packet(rows, edges))
        rows[1]['source_group_id'] = rows[0]['source_group_id']
        self.assertEqual(verify(packet(rows, edges))['known_component_count'], 1)

    def test_same_preset_revision_text_speed_or_crop_cannot_create_new_group(self):
        rows = [source(), source(rid='fixture-two', group='fixture-group-two')]
        rows[0].update(asset_revision='fixture-revision-a', mode='short', speed=1.0)
        rows[1].update(asset_revision='fixture-revision-b', mode='carrier', speed=1.2)
        with self.assertRaisesRegex(ValueError, 'ONE_LINEAGE_ONE_CONSUMED_SOURCE_GROUP'):
            verify(packet(rows))

    def test_distinct_components_cannot_collide_on_one_group_label(self):
        with self.assertRaisesRegex(ValueError, 'ONE_GROUP_ONE_KNOWN_LINEAGE'):
            verify(packet([source(), source('af_fixturetwo', rid='fixture-two',
                group='kokoro-v1_0-preset:af_fixtureone')]))

    def test_fresh_source_cannot_impersonate_legacy_replay_group(self):
        # This label is actually consumed by bag_weights. Source namespaces
        # alone must not allow a fresh VCTK source to merge into old Libri bags.
        for group in ('librispeech:124', 'libri-speaker:124', 'fixture-arbitrary'):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'CANONICAL_TYPED_COMPONENT_SOURCE_GROUP'):
                verify(packet([source('p225', 'vctk-speaker', group=group)]), ('124',))
        row = source('p225', 'vctk-speaker')
        verify(packet([row]), ('124',))
        self.assertEqual(row['source_group_id'], 'vctk-speaker:225')

    def test_lineage_group_canonicalization_is_deterministic_and_never_silent(self):
        left, right = 'kokoro-v1_0-preset:af_fixtureone', 'kokoro-v1_0-preset:af_fixturetwo'
        rows = [source(), source('af_fixturetwo', rid='fixture-two', group=left)]
        parts = packet(rows, [relationship(right, left)])
        before = copy.deepcopy(parts)
        self.assertEqual(verify(parts)['known_component_count'], 1)
        self.assertEqual(parts, before)
        for row in rows:
            row['source_group_id'] = right
        with self.assertRaisesRegex(ValueError, 'CANONICAL_TYPED_COMPONENT_SOURCE_GROUP'):
            verify(packet(rows, [relationship(left, right)]))

    def test_unknown_masked_failed_row_still_participates_in_split_barrier(self):
        rows = [source(), source(role='locked-test', rid='fixture-two')]
        corpus, admission, contract = packet(rows)
        corpus['rows'][1]['generation_status'] = 'failed'
        admission['rows'][1].update(supervision_allowed=False, tier='unknown', research_class=None)
        with self.assertRaisesRegex(ValueError, 'LINEAGE_CANNOT_CROSS'):
            verify((corpus, admission, contract))

    def test_declared_relation_to_legacy_libri_cannot_hide_in_another_namespace(self):
        edges = [relationship('kokoro-v1_0-preset:af_fixtureone', 'libri-speaker:239', 'shared_lineage')]
        with self.assertRaisesRegex(ValueError, 'LEGACY_LIBRI_KNOWN_LINEAGE'):
            verify(packet([source()], edges), ('p239',))

    def test_known_but_unresolved_mixture_is_not_an_independent_group(self):
        with self.assertRaisesRegex(ValueError, 'UNRESOLVED_KNOWN_MIXTURE'):
            verify(packet([source('af_heart')]))


class CorpusAndContractBoundary(unittest.TestCase):
    def test_diagnostic_campaign_cannot_be_rewrapped_by_only_changing_flags(self):
        for changes in ({'campaign_id': i.PROBE_CAMPAIGN}, {'schema': 'english-kokoro48-corpus-v1'},
                        {'diagnostic_only': True}, {'training_allowed': False}, {'auto_admission_allowed': False}):
            parts = packet([source()]); parts[0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'NO_DIAGNOSTIC_PROBE_TRAINING'):
                verify(parts)

    def test_diagnostic_row_cannot_be_hidden_in_a_training_labeled_corpus(self):
        row = source(); row['diagnostic_only'] = True
        with self.assertRaisesRegex(ValueError, 'NO_DIAGNOSTIC_OR_HUMAN_GOLD'):
            verify(packet([row]))

    def test_contract_digest_and_admission_identity_are_bound(self):
        for mutation in ('contract', 'admission-group', 'admission-namespace', 'admission-id'):
            parts = packet([source()])
            if mutation == 'contract': parts[2]['train_side_exposed'].append('kokoro-v1_0-preset:af_fixtureone')
            elif mutation == 'admission-group': parts[1]['rows'][0]['source_group_id'] = 'different'
            elif mutation == 'admission-namespace': parts[1]['rows'][0]['source_namespace'] = 'libri-speaker'
            else: parts[1]['rows'][0]['source_id'] = 'af_fixturetwo'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(parts)

    def test_unknown_lineage_is_scoped_as_preset_not_human(self):
        result = verify(packet([source(role='development')]))
        meta = result['rows']['fixture-row-one']
        self.assertEqual(meta['evaluation_scope'], 'HELDOUT_SYNTHETIC_PRESET_WITH_KNOWN_LINEAGE_EXCLUSIONS')
        self.assertFalse(meta['known_lineage']['human_ancestry_complete'])
        self.assertEqual(meta['known_lineage']['declared_relations'], [])
        self.assertFalse(result['independent_human_speakers_claimed'])


class CoreIntegration(unittest.TestCase):
    def test_both_numeric_entrypoints_disabled_before_runtime_import_or_data_access(self):
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_DISABLED'):
            c.train_and_evaluate(None, None, None, None)
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_DISABLED'):
            c.feature_packet(None, None, None, None, None, None)

    def test_diagnostic_input_rejected_before_numeric_runtime_import_even_if_candidate_flag_patched(self):
        corpus, admission, _ = packet([source()]); corpus['campaign_id'] = i.PROBE_CAMPAIGN
        frozen = SimpleNamespace(regular=lambda p: corpus if p.name == 'CORPUS_MANIFEST.json' else admission,
                                 loads=lambda obj: obj)
        with patch.object(c, 'SOURCE_IDENTITY_CANDIDATE_ENABLED', True), \
             self.assertRaisesRegex(ValueError, 'NO_DIAGNOSTIC_PROBE_TRAINING'):
            c.train_and_evaluate(Path('/fixture-only-no-files'), frozen, None, None)

    def test_event_matching_unchanged_and_identity_retained(self):
        result = verify(packet([source(role='development')]))
        row = dict(id='fresh', parent_id='source', role='development', source_group_id='fixture-group-one',
            weak_class=1, tier='silver', condition='clean', pronunciation_hypothesis='fixture',
            matched_pronunciation_hypothesis='fixture', clip_start_sample=38400,
            original_clip_eof_sample=54400, samples=70400)
        row.update(result['rows']['fixture-row-one'])
        for event_sample in (54401, 54527, 54528, 70400, 70401):
            scores = [{'sample': event_sample, 'keyword': 1}]
            observed = c.fresh_summary(row, scores)
            self.assertEqual({k: v for k, v in observed.items() if k not in i.METADATA_FIELDS}, old.fresh_summary(row, scores))
            self.assertEqual(i.metadata(observed), i.metadata(row))

    def test_mock_feature_packet_carries_identity_without_model_or_numpy(self):
        row = source(); row.update(pcm_path='fixture.pcm', pcm_samples=6400)
        raw = bytes(12800); row['pcm_sha256'] = hashlib.sha256(raw).hexdigest()
        corpus, admission, contract = packet([row]); proof = verify((corpus, admission, contract))
        proof['intake_binding_sha256'] = 'f' * 64
        saved = {}; calls = []
        class NoModelSession:
            def run(self, *args, **kwargs):
                raise AssertionError('Fixture must not call any model session')
        class FakeEmbedding:
            def __init__(self, *args):
                self.geometry = {'head_support_start_before_available': 31840, 'support_end_before_available': 128}
                self.mel = self.embedding = NoModelSession()
                calls.append('mock_constructor_only')
        def frozen_postgeometry(rows):
            # A deliberate mock of qualification, never an acoustic result.
            for case in rows: case['usable'] = False
            return {'passed': True, 'fixture_only': True}
        with tempfile.TemporaryDirectory(prefix='source-identity-fixture-') as tmp:
            output = Path(tmp)
            def read(path):
                if path.name == 'CORPUS_MANIFEST.json': return c.canonical(corpus)
                if path.name == 'ADMISSION.json': return c.canonical(admission)
                if path.name == 'fixture.pcm': return raw
                return path.read_bytes()
            def put(path, data):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            def save_npz(path, **arrays):
                self.assertEqual(arrays, {}); put(path, b'fixture-only-empty-cache')
            frozen = SimpleNamespace(loads=json.loads, regular=read, once=lambda path, obj: saved.update({path.name: copy.deepcopy(obj)}),
                put=put, file_identity=lambda path: c.identity(path.read_bytes()))
            progress = {'new_embedding_calls_completed': 0}
            with patch.object(c, 'SOURCE_IDENTITY_CANDIDATE_ENABLED', True), \
                 patch.object(c, 'BOUND_INTAKE_BINDING_SHA256', 'f' * 64), \
                 patch.object(c, 'postgeometry_admission', side_effect=frozen_postgeometry), \
                 patch.dict(sys.modules, {'causal_embedding': SimpleNamespace(FrozenOnnxEmbedding=FakeEmbedding)}), \
                 patch.object(sys, 'path', sys.path.copy()):
                features, streams = c.feature_packet(output, frozen, progress, lambda: None,
                    SimpleNamespace(savez=save_npz), proof)
            self.assertEqual(len(features), 3); self.assertEqual(streams, {})
            self.assertEqual(calls, ['mock_constructor_only'])
            for feature in features:
                self.assertEqual(i.metadata(feature), proof['rows'][row['id']])
                # The unchanged production CASE_ROSTER/reference construction
                # uses this same **row propagation; no head is called here.
                final_case = {**feature, 'events': [], 'candidate_reference_available': True}
                self.assertEqual(i.metadata(final_case), proof['rows'][row['id']])
            self.assertEqual(saved['SYNTHETIC_FEATURE_ROSTER.json']['rows'], features)

    def test_original_core_and_numeric_function_ast_remain_unchanged(self):
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(), BASELINE_SHA)
        before = ast.parse(BASELINE.read_text()); after = ast.parse((HERE/'english_typed_synthetic_training_core.py').read_text())
        old_functions = {x.name: x for x in before.body if isinstance(x, ast.FunctionDef)}
        new_functions = {x.name: x for x in after.body if isinstance(x, ast.FunctionDef)}
        self.assertEqual(set(old_functions), set(new_functions) - {'_pinned_source_helper'})
        changed = {'feature_packet', 'fresh_summary', 'train_and_evaluate'}
        for name in set(old_functions)-changed:
            self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]), name)
        for name in ('RECIPE', 'CONTEXT', 'ROLES', 'CONDITIONS', 'SCHEMA'):
            self.assertEqual(getattr(c, name), getattr(old, name))
        def gates(tree):
            return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AugAssign))
                and any(isinstance(x, ast.Name) and x.id == 'gates' for x in ast.walk(node))]
        self.assertEqual(gates(before), gates(after))


if __name__ == '__main__':
    unittest.main()
