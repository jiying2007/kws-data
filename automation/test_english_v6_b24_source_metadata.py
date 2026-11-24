"""Metadata-only exclusion, publication and original reexport contracts."""
import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

PATH = Path(__file__).with_name('english_v6_b24_source_metadata.py')
SPEC = importlib.util.spec_from_file_location('b24_metadata_test_target', PATH)
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.hashes = metadata.build_documents()
        cls.base, cls.old, cls.history, cls.exposure, _, _ = metadata.inputs()

    def test_all_history_original120_and_all72_reserves_remain_excluded(self):
        ancestry = self.documents['ancestry.json']
        for key in ('all_pcm_sha256', 'all_source_groups', 'canonical_people',
                    'synthetic_excluded_people', 'natural_excluded_people'):
            self.assertLessEqual(set(self.history[key]), set(ancestry[key]), key)
        self.assertEqual(len(ancestry['all_pcm_sha256']), len(self.history['all_pcm_sha256']) + 120)
        self.assertLessEqual({row['pcm_sha256'] for row in self.exposure['rows']}, set(ancestry['all_pcm_sha256']))
        all_b23 = set(ancestry['B23_exposed_primary_people']) | set(ancestry['B23_sealed_replication_people'])
        self.assertEqual(len(all_b23), 72)
        for key in ('synthetic_excluded_people', 'natural_excluded_people'):
            self.assertLessEqual(all_b23, set(ancestry[key]))
        for key in ('B22_exposed_primary_people', 'B22_sealed_replication_people', 'prior_never_reclaimed_reserves',
                    'prior_explicit_and_conservative_roster_people'):
            self.assertEqual(ancestry[key], self.history[key])

    def test_new72_are_disjoint_across_both_panels_and_source_types(self):
        all_people = []
        assessment, ancestry = self.documents['assessment.json'], self.documents['ancestry.json']
        for panel in ('primary', 'replication'):
            synthetic = [self.base.canonical(p) for p in assessment[panel]['synthetic']['speaker_id_map'].values()]
            natural = assessment[panel]['natural']['speakers']
            self.assertEqual((len(synthetic), len(natural)), (24, 12))
            self.assertFalse(set(synthetic) & set(ancestry['synthetic_excluded_people']))
            self.assertFalse(set(natural) & set(ancestry['natural_excluded_people']))
            all_people.extend(synthetic + natural)
        self.assertEqual(len(all_people), len(set(all_people)))
        self.assertEqual(len(all_people), 72)

    def test_original_b14_sources_strata_and_five_observers_are_unchanged(self):
        assessment = self.documents['assessment.json']
        for key in ('source_gate', 'gains', 'quality_criteria', 'quality_assessment_sha256',
                    'observer_scientific_contract', 'maximum_native_observer_calls',
                    'maximum_primary_KWS_forwards', 'maximum_public_Piper_synthesize_calls',
                    'maximum_natural_original_decodes', 'source_format'):
            self.assertEqual(assessment[key], self.old[key], key)
        self.assertEqual(assessment['metadata_selection']['algorithm'], self.old['metadata_selection']['algorithm'])
        self.assertNotEqual(assessment['metadata_selection']['salts'], self.old['metadata_selection']['salts'])
        train = assessment['training']
        self.assertEqual(train['shared_parent'], 'actual_original_B23_challenger')
        self.assertEqual(train['shared_parent_pt_sha256'], train['control_pt_sha256'])
        self.assertEqual(train['control_pt_sha256'], metadata.CONTROL_PT)
        self.assertEqual(train['design_sha256'], metadata.DESIGN_SHA)
        self.assertEqual(train['optimizer'], 'fresh_AdamW')
        self.assertFalse(train['restore_optimizer_state'])
        self.assertFalse(train['exact_optimizer_continuation'])
        self.assertTrue(assessment['original_B23_NO_PROMOTION_required_before_any_new_budget'])
        self.assertTrue(assessment['freeze_before_optimizer_and_new_source'])
        for key in ('execution_authorized', 'promotion_allowed', 'replication_opened', 'threshold_search',
                    'training_allowed_for_sources', 'B23_quality_results_read'):
            self.assertIs(assessment[key], False, key)

    def test_whole_fixed_source_cells_and_two_natural_originals_per_reader(self):
        for panel in ('primary', 'replication'):
            rows = self.documents[panel + '.json']['rows']
            self.assertEqual(len(rows), 96)
            self.assertEqual(sum(row['requested_class'] == 2 for row in rows), 48)
            self.assertEqual(sum(row['requested_class'] == 0 for row in rows), 48)
            self.assertEqual({(r['cell_id'], r['length_scale']) for r in rows},
                             {('mero-fast', 1.4), ('mero-slow', 1.8), ('question', 1.0), ('request', 1.08)})
            self.assertTrue(all(row['observed_class'] is None and row['training_allowed'] is False for row in rows))
            natural = self.documents['natural-' + panel + '.json']
            self.assertFalse(natural['opened'])
            self.assertEqual(len(natural['roster']), 24)
            self.assertEqual(len(set(natural['selected_utterance_ids'])), 24)
            self.assertEqual(natural['utterance_selection'], self.old[panel]['natural']['utterance_selection'])
            self.assertTrue(all(sum(row['speaker'] == speaker for row in natural['roster']) == 2 for speaker in natural['speakers']))

    def test_frozen_binding_keeps_every_dependency_and_does_not_authorize(self):
        report = metadata.validate()
        binding = json.loads((metadata.HERE / 'english_v6_b24_assessment_binding.json').read_bytes())
        self.assertEqual(binding['files'], report['hashes'])
        self.assertEqual(binding['recipe_sha256'], metadata.DESIGN_SHA)
        self.assertEqual(binding['assessment']['sha256'], report['hashes'][binding['assessment']['path']])
        self.assertTrue(binding['frozen'])
        self.assertFalse(binding['optimizer_allowed'])
        self.assertEqual(report['status'], 'PASS_METADATA_ONLY')
        rebuilt, _ = metadata.build_documents()
        self.assertEqual(metadata.encoded(rebuilt['assessment.json']), metadata.encoded(self.documents['assessment.json']))

    def test_public_origin_and_exposure_contain_only_necessary_whitelist_fields(self):
        origin = json.loads((metadata.ROOT / metadata.ORIGIN).read_bytes())
        expected_rows = {'id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person'}
        self.assertTrue(all(set(row) == expected_rows for row in self.exposure['rows']))
        identity = origin['source_api_identity']
        self.assertEqual(set(identity['run']), {'id', 'run_attempt', 'head_sha', 'head_branch', 'event', 'path', 'repository'})
        self.assertEqual(set(identity['commit']), {'sha', 'parents', 'files'})
        self.assertEqual(set(identity['artifact']), {'id', 'name', 'size_in_bytes', 'digest', 'expired', 'workflow_run'})
        self.assertEqual(set(identity['run']['repository']), {'full_name', 'private'})
        noisy = copy.deepcopy(identity)
        noisy['run']['actor'] = {'login': 'invented-must-not-publish', 'email': 'invented@example.invalid'}
        noisy['commit']['author'] = {'name': 'invented-must-not-publish'}
        self.assertEqual(metadata._source_api_identity(noisy, metadata.SOURCE_PIN), identity)
        self.assertEqual(origin, metadata._origin_document(self.exposure, noisy))
        for name in ('b23-source-run.json', 'b23-source-artifacts.json', 'b23-source-commit.json', 'b23-source-manifest.json'):
            self.assertFalse((metadata.PANELS / name).exists())

    def test_mutated_frozen_roster_is_rejected_without_redraw(self):
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temporary:
            folder = Path(temporary)
            for name, value in self.documents.items():
                (folder / name).write_bytes(metadata.encoded(value))
            changed = copy.deepcopy(self.documents['assessment.json'])
            changed['primary']['natural']['speakers'][0] = '1'
            (folder / 'assessment.json').write_bytes(metadata.encoded(changed))
            with mock.patch.object(metadata, 'PANELS', folder), self.assertRaisesRegex(ValueError, 'complete deterministic frozen B24 metadata: assessment.json'):
                metadata.validate()

    def test_omitted_original_is_rejected(self):
        arguments = list(metadata.inputs())
        arguments[3] = copy.deepcopy(arguments[3]); arguments[3]['rows'].pop()
        with mock.patch.object(metadata, 'inputs', return_value=tuple(arguments)), self.assertRaisesRegex(ValueError, 'complete original120'):
            metadata.build_documents()

    def test_original_source_reexport_reads_only_two_metadata_members_and_never_authorizes(self):
        # Invented ZIP fixture, not a real source artifact or quality decision.
        manifest = copy.deepcopy(self.exposure)
        manifest['schema'] = 'english-v6-b23-fresh-combined-source-v1'
        history = b'{"all_prior_replication_sealed":true}'
        manifest['source_ancestry_sha256'] = manifest['history_receipt_sha256'] = metadata.digest(history)
        members = {'MANIFEST.json': metadata.encoded(manifest), 'HISTORY.json': history}
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items(): archive.writestr(name, value)
            archive.writestr('wav/must-not-be-interpreted.wav', b'invented opaque payload')
        pin = dict(metadata.SOURCE_PIN, bytes=len(raw.getvalue()), sha256=metadata.digest(raw.getvalue()))
        api = copy.deepcopy(json.loads((metadata.ROOT / metadata.ORIGIN).read_bytes())['source_api_identity'])
        api['artifact'].update(size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'])
        member_hashes = {name: metadata.digest(value) for name, value in members.items()}
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temporary, \
             mock.patch.object(metadata, 'SOURCE_PIN', pin), mock.patch.object(metadata, 'SOURCE_MEMBERS', member_hashes):
            folder = Path(temporary)
            archive_path = folder / 'source.zip'; archive_path.write_bytes(raw.getvalue())
            docs = metadata.derive_original_metadata(members, api, pin)
            hashes = {}
            for name, value in docs.items():
                path = folder / name; path.write_bytes(metadata.encoded(value))
                hashes[path.relative_to(metadata.ROOT).as_posix()] = metadata.digest(path.read_bytes())
            read = zipfile.ZipFile.read
            observed = []
            def selected(archive, name, *args, **kwargs):
                observed.append(name)
                self.assertIn(name, members)
                return read(archive, name, *args, **kwargs)
            with mock.patch.object(metadata, 'PANELS', folder), mock.patch.object(metadata, 'EXTRA_INPUTS', hashes), \
                 mock.patch.object(metadata, 'validate', return_value={'status': 'PASS_METADATA_ONLY'}), \
                 mock.patch.object(zipfile.ZipFile, 'read', selected):
                report = metadata.verify_original_source(archive_path, api, pin)
                self.assertEqual(observed, ['MANIFEST.json', 'HISTORY.json'])
                self.assertEqual(report['original_rows'], 120)
                self.assertEqual(report['prior_panel_people_excluded'], 72)
                self.assertFalse(report['scientific_execution_authorized'])
                self.assertEqual(report['audio_members_interpreted'], 0)
                broken = copy.deepcopy(api); broken['commit']['parents'][0]['sha'] = 'f' * 40
                with self.assertRaisesRegex(ValueError, 'complete pre-scoring original SOURCE'):
                    metadata.verify_original_source(archive_path, broken, pin)
                (folder / 'b23-source-exposure.json').write_bytes(b'{}')
                with self.assertRaisesRegex(ValueError, 'reexport equals frozen public whitelist'):
                    metadata.verify_original_source(archive_path, api, pin)

    def test_wrong_source_pin_blocks_before_any_original_file_open(self):
        with mock.patch.object(Path, 'open', side_effect=AssertionError('must not read before pin matches')):
            with self.assertRaisesRegex(ValueError, 'same complete original source ZIP'):
                metadata.verify_original_source(Path('/not-opened'), {}, dict(metadata.SOURCE_PIN, id=1))

    def test_metadata_validation_uses_only_registered_python_and_json(self):
        original_import, original_read = builtins.__import__, Path.read_bytes
        denied = {'torch', 'numpy', 'torchaudio', 'transformers', 'piper', 'soundfile', 'onnxruntime', 'requests', 'httpx'}
        reads = []
        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split('.')[0], denied)
            return original_import(name, *args, **kwargs)
        def guarded_read(path):
            self.assertIn(path.suffix, {'.py', '.json'})
            reads.append(path)
            return original_read(path)
        with mock.patch('builtins.__import__', side_effect=guarded_import), mock.patch.object(Path, 'read_bytes', guarded_read):
            report = metadata.validate()
        self.assertTrue(reads)
        for key in ('source_calls', 'observer_calls', 'kws_forwards', 'optimizer_updates'):
            self.assertEqual(report[key], 0)
        self.assertFalse(report['source_execution_authorized'])
        self.assertFalse(denied & set(sys.modules))


if __name__ == '__main__':
    unittest.main()
