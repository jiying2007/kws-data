"""Approved Q1 metadata freeze: exact approval, complete exclusions, zero calls."""
import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

PATH = Path(__file__).with_name('english_v6_b24q1_source_metadata.py')
SPEC = importlib.util.spec_from_file_location('b24q1_metadata_test_target', PATH)
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.hashes = metadata.build_documents()
        cls.base, cls.old, cls.history, cls.exposure, _, _ = metadata.inputs()

    def test_all_2943_pcm_and_every_old_exposure_and_reserve_are_preserved(self):
        ancestry = self.documents['ancestry.json']
        self.assertEqual(len(self.history['all_pcm_sha256']), 2823)
        self.assertEqual(len(ancestry['all_pcm_sha256']), 2943)
        for key in ('all_pcm_sha256', 'all_source_groups', 'canonical_people',
                    'synthetic_excluded_people', 'natural_excluded_people'):
            self.assertLessEqual(set(self.history[key]), set(ancestry[key]), key)
        self.assertLessEqual({r['pcm_sha256'] for r in self.exposure['rows']}, set(ancestry['all_pcm_sha256']))
        actual_old_people = set(self.history['canonical_people']) | set(self.exposure['all_excluded_people'])
        actual_old_people |= set(self.history['prior_explicit_and_conservative_roster_people'])
        for generation in ('B22', 'B23', 'B24', 'B24R1'):
            for scope in ('exposed_primary_people', 'sealed_replication_people'):
                actual_old_people |= set(ancestry[generation + '_' + scope])
        for key in ('synthetic_excluded_people', 'natural_excluded_people'):
            self.assertLessEqual(actual_old_people, set(ancestry[key]))
        for key in ('prior_never_reclaimed_reserves', 'prior_explicit_and_conservative_roster_people',
                    'B22_exposed_primary_people', 'B22_sealed_replication_people',
                    'B23_exposed_primary_people', 'B23_sealed_replication_people',
                    'B24_exposed_primary_people', 'B24_sealed_replication_people'):
            self.assertEqual(ancestry[key], self.history[key])

    def test_new72_are_distinct_across_both_panels_and_source_types(self):
        assessment, ancestry = self.documents['assessment.json'], self.documents['ancestry.json']
        all_people = []
        for role in ('primary', 'replication'):
            synthetic = [self.base.canonical(p) for p in assessment[role]['synthetic']['speaker_id_map'].values()]
            natural = assessment[role]['natural']['speakers']
            self.assertEqual((len(synthetic), len(natural)), (24, 12))
            self.assertFalse(set(synthetic) & set(ancestry['synthetic_excluded_people']))
            self.assertFalse(set(natural) & set(ancestry['natural_excluded_people']))
            all_people.extend(synthetic + natural)
        self.assertEqual(len(all_people), 72)
        self.assertEqual(len(set(all_people)), 72)
        self.assertFalse(set(all_people) & set(self.exposure['all_excluded_people']))

    def test_strict_source_format_b14_gains_and_budgets_are_unchanged(self):
        assessment = self.documents['assessment.json']
        for key in ('source_gate', 'gains', 'quality_criteria', 'quality_assessment_sha256',
                    'observer_scientific_contract', 'maximum_native_observer_calls',
                    'maximum_primary_KWS_forwards', 'maximum_public_Piper_synthesize_calls',
                    'maximum_natural_original_decodes', 'source_format', 'target'):
            self.assertEqual(assessment[key], self.old[key], key)
        self.assertEqual(assessment['maximum_native_observer_calls'], 600)
        self.assertEqual(assessment['maximum_primary_KWS_forwards'], 1440)
        self.assertEqual(assessment['metadata_selection']['algorithm'], self.old['metadata_selection']['algorithm'])
        self.assertNotEqual(assessment['metadata_selection']['salts'], self.old['metadata_selection']['salts'])

    def test_whole_source_cells_are_fixed_and_natural_has_two_originals_per_person(self):
        for role in ('primary', 'replication'):
            rows = self.documents[role + '.json']['rows']
            self.assertEqual(len(rows), 96)
            self.assertEqual(sum(r['requested_class'] == 2 for r in rows), 48)
            self.assertEqual(sum(r['requested_class'] == 0 for r in rows), 48)
            self.assertEqual({(r['cell_id'], r['length_scale']) for r in rows},
                             {('mero-fast', 1.4), ('mero-slow', 1.8), ('question', 1.0), ('request', 1.08)})
            self.assertTrue(all(r['observed_class'] is None and r['training_allowed'] is False for r in rows))
            natural = self.documents['natural-' + role + '.json']
            self.assertFalse(natural['opened'])
            self.assertEqual(len(natural['roster']), 24)
            self.assertEqual(len(set(natural['selected_utterance_ids'])), 24)
            self.assertEqual(natural['utterance_selection'], self.old[role]['natural']['utterance_selection'])
            self.assertTrue(all(sum(r['speaker'] == speaker for r in natural['roster']) == 2
                                for speaker in natural['speakers']))

    def test_post_training_evaluation_never_relabels_original_training(self):
        a = self.documents['assessment.json']
        self.assertEqual(a['training_assessment_sha256'], metadata.TRAINING_ASSESSMENT_SHA)
        self.assertEqual(a['training_design_sha256'], metadata.DESIGN_SHA)
        self.assertEqual(a['frozen_model_pt_sha256'], metadata.FROZEN_MODELS)
        self.assertEqual(a['training']['challenger_pt_sha256'], metadata.CANDIDATE_PT)
        self.assertEqual(a['training']['control_pt_sha256'], metadata.CONTROL_PT)
        self.assertEqual(a['training']['original_completed_challenger_updates'], 192)
        self.assertEqual(a['training']['new_challenger_updates'], 0)
        self.assertTrue(a['training']['original_training_assessment_not_rebound'])
        self.assertFalse(a['freeze_before_optimizer_and_new_source'])
        self.assertTrue(a['freeze_after_training_before_new_source'])
        self.assertFalse(a['metadata_selection']['selection_before_optimizer'])
        self.assertTrue(a['metadata_selection']['selection_after_training'])
        self.assertTrue(a['metadata_selection']['selection_before_first_independent_KWS'])
        self.assertNotEqual(metadata.digest(metadata.encoded(a)), metadata.TRAINING_ASSESSMENT_SHA)

    def test_one_bounded_new_evaluation_has_no_automatic_execution_or_next_trial(self):
        a = self.documents['assessment.json']
        self.assertTrue(a['evaluation_only'])
        self.assertTrue(a['B24_original_source_stop_preserved'])
        self.assertTrue(a['original_B24R1_SOURCE_STOP_proof_required_before_any_new_budget'])
        self.assertTrue(a['Q1_source_failure_terminal'])
        self.assertTrue(a['old_B25_opening_condition_unchanged'])
        self.assertEqual(a['new_optimizer_updates'], 0)
        self.assertNotIn('original_B24_NO_PROMOTION_required_before_any_new_budget', a)
        for key in ('execution_authorized', 'source_execution_authorized', 'primary_scoring_authorized',
                    'replica_opening_authorized', 'engineering_authorized', 'optimizer_allowed',
                    'promotion_allowed', 'replication_opened', 'automatic_R2_allowed',
                    'old_B24_sources_used_for_scoring', 'old_B24_replication_opened',
                    'threshold_search', 'training_allowed_for_sources', 'B24_quality_results_read'):
            self.assertIs(a[key], False, key)

    def test_frozen_evaluation_binding_is_complete_and_explicitly_not_a_train_binding(self):
        report = metadata.validate()
        b = json.loads((metadata.HERE / 'english_v6_b24q1_evaluation_binding.json').read_bytes())
        self.assertEqual(b['files'], report['hashes'])
        self.assertEqual(b['assessment']['sha256'], report['hashes'][b['assessment']['path']])
        self.assertEqual(b['original_training_design']['sha256'], metadata.DESIGN_SHA)
        self.assertEqual(b['original_training_assessment']['sha256'], metadata.TRAINING_ASSESSMENT_SHA)
        self.assertEqual(b['evaluation_design']['sha256'], metadata.EVALUATION_DESIGN_SHA)
        self.assertTrue(b['frozen'])
        self.assertTrue(b['evaluation_only'])
        self.assertFalse(b['original_B24R1_SOURCE_STOP_proof_bound'])
        self.assertFalse(b['optimizer_allowed'])
        self.assertEqual(len(report['hashes']), 60)
        self.assertEqual(report['status'], 'PASS_METADATA_ONLY')

    def test_only_eight_public_panel_documents_and_five_row_fields(self):
        expected = set(metadata.OUTPUT_NAMES) | {'b24r1-source-exposure.json', 'b24r1-source-origin.json'}
        self.assertEqual({p.name for p in metadata.PANELS.iterdir()}, expected)
        self.assertTrue(all(set(r) == {'id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person'}
                            for r in self.exposure['rows']))
        self.assertEqual(len(self.exposure['primary_people']), 36)
        self.assertEqual(len(self.exposure['sealed_replication_people']), 36)
        self.assertEqual(len(self.exposure['all_excluded_people']), 72)
        origin = json.loads((metadata.ROOT / metadata.ORIGIN).read_bytes())
        self.assertEqual(origin['source_producer_artifact'], metadata.SOURCE_PRODUCER_PIN)
        self.assertEqual(origin['original_assessment'], metadata.ORIGINAL_ASSESSMENT)
        self.assertTrue(origin['full_original_SOURCE_STOP_proof_is_separate'])
        self.assertFalse(origin['original_source_gate_pass'])
        noisy = copy.deepcopy(origin['source_api_identity'])
        noisy['run']['actor'] = {'email': 'must-not-publish@example.invalid'}
        noisy['commit']['author'] = {'name': 'must-not-publish'}
        self.assertEqual(metadata._origin_document(self.exposure, noisy), origin)
        self.assertEqual(set(origin['source_api_identity']['run']),
                         {'id', 'run_attempt', 'head_sha', 'head_branch', 'event', 'path',
                          'status', 'conclusion', 'repository'})

    def test_modified_old_train_design_is_rejected_before_selection(self):
        read = Path.read_bytes
        def altered(path):
            raw = read(path)
            if path == metadata.ROOT / metadata.DESIGN_PATH:
                obj = json.loads(raw); obj['updates'] = 193
                return metadata.encoded(obj)
            return raw
        with mock.patch.object(Path, 'read_bytes', altered), self.assertRaises(ValueError):
            metadata.build_documents()

    def test_changed_new_roster_or_permission_is_rejected_without_redraw(self):
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temporary:
            folder = Path(temporary)
            for name, value in self.documents.items(): (folder / name).write_bytes(metadata.encoded(value))
            for change in ('roster', 'permission'):
                modified = copy.deepcopy(self.documents['assessment.json'])
                if change == 'roster': modified['primary']['natural']['speakers'][0] = '1'
                else: modified['execution_authorized'] = True
                (folder / 'assessment.json').write_bytes(metadata.encoded(modified))
                with mock.patch.object(metadata, 'PANELS', folder), self.assertRaisesRegex(ValueError, 'complete deterministic frozen B24Q1 metadata'):
                    metadata.validate()

    def test_missing_old_original_or_sealed_person_is_rejected(self):
        for mutation in ('row', 'person'):
            arguments = list(metadata.inputs())
            arguments[3] = copy.deepcopy(arguments[3])
            if mutation == 'row': arguments[3]['rows'].pop()
            else: arguments[3]['all_excluded_people'].pop()
            with mock.patch.object(metadata, 'inputs', return_value=tuple(arguments)), self.assertRaises(ValueError):
                metadata.build_documents()

    def test_original_source_reexport_reads_only_the_two_metadata_members(self):
        # Deliberately invented ZIP fixture. It cannot certify real SOURCE STOP.
        manifest = copy.deepcopy(self.exposure)
        manifest['schema'] = 'english-v6-b24r1-fresh-combined-source-v1'
        manifest['original_training_assessment_sha256'] = metadata.TRAINING_ASSESSMENT_SHA
        history = b'{"all_prior_replication_sealed":true}'
        manifest['source_ancestry_sha256'] = manifest['history_receipt_sha256'] = metadata.digest(history)
        members = {'MANIFEST.json': metadata.encoded(manifest), 'HISTORY.json': history}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items(): archive.writestr(name, raw)
            archive.writestr('wav/must-not-be-read.wav', b'opaque fixture')
        pin = dict(metadata.SOURCE_PIN, bytes=len(stream.getvalue()), sha256=metadata.digest(stream.getvalue()))
        api = copy.deepcopy(json.loads((metadata.ROOT / metadata.ORIGIN).read_bytes())['source_api_identity'])
        api['artifact'].update(size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'])
        hashes = {name: metadata.digest(raw) for name, raw in members.items()}
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temporary, \
             mock.patch.object(metadata, 'SOURCE_PIN', pin), mock.patch.object(metadata, 'SOURCE_MEMBERS', hashes):
            folder = Path(temporary); original = folder / 'source.zip'; original.write_bytes(stream.getvalue())
            exported = metadata.derive_original_metadata(members, api, pin)
            expected = {}
            for name, obj in exported.items():
                path = folder / name; path.write_bytes(metadata.encoded(obj))
                expected[path.relative_to(metadata.ROOT).as_posix()] = metadata.digest(path.read_bytes())
            read = zipfile.ZipFile.read; observed = []
            def selected(archive, name, *args, **kwargs):
                observed.append(name); self.assertIn(name, members)
                return read(archive, name, *args, **kwargs)
            with mock.patch.object(metadata, 'PANELS', folder), mock.patch.object(metadata, 'EXTRA_INPUTS', expected), \
                 mock.patch.object(metadata, 'validate', return_value={'status': 'PASS_METADATA_ONLY'}), \
                 mock.patch.object(zipfile.ZipFile, 'read', selected):
                report = metadata.verify_original_source(original, api, pin)
                self.assertEqual(observed, ['MANIFEST.json', 'HISTORY.json'])
                self.assertEqual(report['original_rows'], 120)
                self.assertFalse(report['scientific_execution_authorized'])
                self.assertTrue(report['full_original_SOURCE_STOP_proof_is_separate'])
                self.assertFalse(report['captured_original_assessment_verified'])
                broken = copy.deepcopy(api); broken['run']['conclusion'] = 'success'
                with self.assertRaises(ValueError): metadata.verify_original_source(original, broken, pin)
                broken = copy.deepcopy(api); broken['commit']['parents'][0]['sha'] = 'f' * 40
                with self.assertRaises(ValueError): metadata.verify_original_source(original, broken, pin)
                (folder / 'b24r1-source-exposure.json').write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError, 'reexport equals frozen public whitelist'):
                    metadata.verify_original_source(original, api, pin)

    def test_wrong_pin_rejects_before_any_file_read(self):
        with mock.patch.object(Path, 'open', side_effect=AssertionError('no file read before pin validation')):
            with self.assertRaises(ValueError):
                metadata.verify_original_source(Path('/must-not-open'), {}, dict(metadata.SOURCE_PIN, id=1))

    def test_validation_is_pure_stdlib_registered_metadata_only(self):
        original_import, original_read = builtins.__import__, Path.read_bytes
        denied = {'torch', 'numpy', 'torchaudio', 'transformers', 'piper', 'soundfile', 'onnxruntime', 'requests', 'httpx'}
        reads = []
        def import_guard(name, *args, **kwargs):
            self.assertNotIn(name.split('.')[0], denied)
            return original_import(name, *args, **kwargs)
        def read_guard(path):
            self.assertIn(path.suffix, {'.py', '.json'}); reads.append(path)
            return original_read(path)
        with mock.patch('builtins.__import__', side_effect=import_guard), \
             mock.patch.object(Path, 'read_bytes', read_guard), \
             mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('no subprocess')), \
             mock.patch.object(socket, 'socket', side_effect=AssertionError('no socket')):
            report = metadata.validate()
        self.assertTrue(reads)
        self.assertFalse(denied & set(sys.modules))
        for key in ('source_calls', 'observer_calls', 'kws_forwards', 'optimizer_updates'):
            self.assertEqual(report[key], 0)
        self.assertFalse(report['source_execution_authorized'])


    def test_entire_piper_population_is_blocked_from_natural_pool(self):
        mapping = self.base.read_inputs()['piper-config.json']['speaker_id_map']
        whole_piper = {self.base.canonical(p) for p in mapping}
        self.assertLessEqual(whole_piper,set(self.documents['ancestry.json']['natural_excluded_people']))
        for role in ('primary','replication'):
            self.assertFalse(whole_piper & set(self.documents['assessment.json'][role]['natural']['speakers']))
        selection = self.documents['assessment.json']['metadata_selection']
        self.assertEqual(selection['eligible_synthetic_voices'],160)
        self.assertEqual(selection['eligible_natural_readers'],66)
        self.assertEqual(selection['available_natural_census'],86)

    def test_exact_original_proposal_is_preserved_with_separate_explicit_approval(self):
        approval=metadata.approval()
        proposal=json.loads((metadata.ROOT/metadata.PROPOSAL_PATH).read_bytes())
        self.assertEqual(metadata.digest((metadata.ROOT/metadata.PROPOSAL_PATH).read_bytes()),metadata.PROPOSAL_SHA)
        self.assertEqual(approval['user_statement'],'批准提案，继续一次全新评估（推荐）')
        self.assertIsNone(approval['user_message_sent_at_utc'])
        self.assertTrue(approval['approved'])
        self.assertTrue(approval['scope']['metadata_selection_authorized'])
        self.assertEqual(approval['scope']['maximum_new_assessments'],1)
        self.assertFalse(approval['grants_immediate_scientific_execution'])
        self.assertFalse(approval['source_workflow_activated'])
        self.assertEqual(proposal['protected_existing_state']['B26_opening_unchanged']['condition'],
                         'only_after_actual_complete_B24R1_primary_NO_PROMOTION')
        self.assertFalse(proposal['protected_existing_state']['B26_opening_unchanged']['enabled'])

    def test_amendment_is_explicit_and_does_not_relabel_full_observer_contract(self):
        a=self.documents['assessment.json']
        contract=metadata.orchestration_contract()
        self.assertTrue(a['original_whole_panel_empty_return_policy_changed'])
        self.assertTrue(a['original_native_validity_and_numerical_admission_unchanged'])
        self.assertIn('full observer orchestration policy has changed',a['observer_scientific_contract_scope'])
        self.assertEqual(a['observer_scientific_contract'],self.old['observer_scientific_contract'])
        self.assertEqual(a['source_orchestration_contract']['sha256'],metadata.ORCHESTRATION_SHA)
        self.assertTrue(a['proved_successful_empty_original_quarantined_without_vote'])
        self.assertTrue(a['all_other_native_faults_whole_panel_stop'])
        self.assertTrue(a['pre_observer_rejects_reported_separately'])
        self.assertTrue(a['all_five_valid_nonempty_payloads_required_for_admission'])
        self.assertIn('N=96+',a['source_observer_denominator'])
        self.assertIn('exact 5*N',a['source_observer_denominator'])
        self.assertTrue(contract['old_whole_panel_empty_return_policy_changed'])
        self.assertFalse(contract['automatic_next_assessment_allowed'])
        self.assertTrue(a['B24R1_original_source_stop_preserved'])
        self.assertTrue(a['old_B26_opening_condition_unchanged'])
        self.assertEqual(a['B26_opener'],'only_after_actual_complete_B24R1_primary_NO_PROMOTION')

    def test_changed_approval_proposal_or_orchestration_rejects_before_selection(self):
        read=Path.read_bytes
        for name in (metadata.APPROVAL_PATH,metadata.PROPOSAL_PATH,metadata.ORCHESTRATION_PATH):
            def altered(path):
                raw=read(path)
                return raw+b' ' if path==metadata.ROOT/name else raw
            with self.subTest(name=name),mock.patch.object(Path,'read_bytes',altered),self.assertRaises(ValueError):
                metadata.build_documents()

    def test_second_freeze_is_rejected_without_rewriting_or_new_selection(self):
        with mock.patch.object(sys,'argv',['metadata','--freeze']), \
             mock.patch.object(metadata,'build_documents',side_effect=AssertionError('no reselection')), \
             self.assertRaisesRegex(ValueError,'freeze only once'):
            metadata.main()


if __name__ == '__main__':
    unittest.main()
