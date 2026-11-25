"""Independent SOURCE-only contracts; no quality evaluator, audio or model runtime."""
from pathlib import Path
import base64
import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
import english_v6_b27_evidence_scope as scope
import english_v6_b24q1_evidence_scope as old_scope
import english_v6_b27_source as source
import english_v6_b27_replication_source as replica_source
import english_v6_b27_natural_sources as natural
import english_v6_b27_replication_natural_sources as replica_natural
import english_v6_b27_capacity as capacity
import english_v6_b27_original_proof as transport
import english_v6_b27_source_producer as producer
HERE = Path(__file__).resolve().parent

class SourceContracts(unittest.TestCase):
    def test_original_native_quarantine_and_admission_code_objects_are_unchanged(self):
        frozen = profile.preserved('english_v6_b24q1_source_observe')
        for owner in (source, replica_source):
            for name in ('observe', 'read_valid', 'load_observations', 'started_document', 'report_document'):
                self.assertIs(getattr(owner.observers, name).__code__, getattr(frozen, name).__code__)
            self.assertIs(owner.observers.owner(), owner)
            self.assertIs(owner.observe.__globals__['lineage'], owner.lineage)
            self.assertIs(owner.screen_document.__code__, profile.preserved('english_v6_b24q1_source').screen_document.__code__)
        self.assertEqual(source.lineage.ASSESSMENT_SHA, replica_source.lineage.ASSESSMENT_SHA)
        self.assertEqual(source.FILES, profile.profile(False)['panel_sha256'])

    def test_missing_proof_or_wrong_panel_never_calls_science(self):
        for owner in (source, replica_source):
            with patch.object(owner.B13, 'render_panel') as render, patch.object(owner.observers, 'observe') as native:
                with self.assertRaises(ValueError): profile.require_proof(owner.PANEL)
                render.assert_not_called(); native.assert_not_called()
        with self.assertRaises(ValueError): replica_source.source_plan('primary')

    def test_primary_runtime_excludes_future_quality_and_preserves_full_training(self):
        paths = producer.runtime_paths(HERE.parent, 'primary')
        for name in ('english_v6_b27_quality_readback.py','english_v6_b27_primary_eval.py','english_v6_b27_replication_eval.py'):
            self.assertNotIn('automation/' + name, paths)
        self.assertIn('automation/english_v6_b27_training_readback.py', paths)
        self.assertIn('automation/english_v6_b27_evidence_scope.py', paths)
        self.assertTrue({'automation/' + name for name in profile.profile(False)['fixed_base_sha256']} <= paths)

    def test_primary_does_not_traverse_any_future_quality_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'automation'; directory.mkdir()
            payloads = {'english_v6_b27_source': 'import english_v6_b27_profile\n',
                'english_v6_b27_replication_source': 'import english_v6_b27_source\n',
                'english_v6_b27_source_producer': '', 'english_v6_b27_original_proof': '',
                'english_v6_b27_profile': 'import english_v6_b27_quality_readback\nimport source_helper\n',
                'english_v6_b27_quality_readback': '', 'english_v6_b27_primary_eval': '',
                'source_helper': '', 'future_only_helper': ''}
            for name, raw in payloads.items(): (directory / (name + '.py')).write_text(raw)
            with patch.object(profile, 'profile', return_value={'fixed_base_sha256': {}}):
                before = producer.runtime_paths(root, 'primary')
                (directory / 'english_v6_b27_quality_readback.py').write_text('import future_only_helper\n')
                self.assertEqual(producer.runtime_paths(root, 'primary'), before)
                self.assertNotIn('automation/future_only_helper.py', before)
                self.assertIn('automation/source_helper.py', before)
                self.assertIn('automation/future_only_helper.py', producer.runtime_paths(root, 'replication'))

    def test_source_workflow_is_an_exact_inert_template(self):
        for panel in ('primary','replication'):
            template = HERE / ('english_v6_b27_' + ('' if panel == 'primary' else 'replication_') + 'source_workflow_template.yml')
            raw = template.read_text(); token = '__B27_REVIEWED_' + panel.upper() + '_SOURCE_PARENT__'
            self.assertEqual(raw.count(token),2)
            runtime = (HERE.parent / '.github/workflows' / ('english-v6-b27-' + panel + '-sources-once.yml')).read_text()
            if token in runtime: self.assertEqual(runtime,raw)
            self.assertIn("github.run_attempt == 1",raw)
            self.assertIn("github.ref == 'refs/heads/research/en-v6-b27-" + panel,raw)
            self.assertIn('cancel-in-progress: false',raw)
            self.assertIn('include-hidden-files: true',raw)
            self.assertNotIn('workflow_dispatch',raw)
            self.assertNotIn('test_english_v6_b27_quality_execution.py',raw)
            self.assertNotIn('TRAIN_*.zip',raw)


    def test_replica_natural_roster_never_resolves_primary_helper(self):
        self.assertIs(replica_natural.original, natural.original)
        with patch.object(natural, 'plan', side_effect=AssertionError('primary roster escaped')), \
             patch.object(replica_natural, 'plan', return_value={'roster': ['replica']}), \
             patch.object(natural.original, 'rank_metadata', return_value=['replica']), \
             patch.object(natural.original, 'pick', return_value=(['replica'], {'metadata': True})):
            self.assertEqual(replica_natural.rank_metadata({}, {}), ['replica'])
            self.assertEqual(replica_natural.pick(None, {}), (['replica'], {'metadata': True}))

    def test_private_scope_does_not_touch_old_q1_scope(self):
        self.assertEqual(old_scope.KINDS, frozenset(('r1_source_stop', 'q1_primary_quality')))
        with scope.scope():
            active = scope._ACTIVE.get(); self.assertIsInstance(active, dict); self.assertIsNone(old_scope._ACTIVE.get())
            with scope.scope(): self.assertIs(scope._ACTIVE.get(), active)
        self.assertIsNone(scope._ACTIVE.get()); self.assertIsNone(old_scope._ACTIVE.get())

    def test_complete_scope_rehashes_mutation_and_destroys_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); evidence = root / 'proof'; evidence.mkdir()
            (evidence / 'original.bin').write_bytes(b'abcd'); reader = root / 'reader.py'; reader.write_bytes(b'fixed')
            files = profile.inventory(evidence); report = profile.canonical(dict(original=True, calls=0)); executions = []
            def execute(): executions.append(True); return report
            def read(): return scope.verify_once('b27_complete_training', evidence, files, profile.canonical(dict(lock=True)),
                reader, profile.sha(reader.read_bytes()), profile.sha(report), execute)
            with scope.scope():
                self.assertEqual(read(), report); self.assertEqual(read(), report); self.assertEqual(len(executions), 1)
                (evidence / 'original.bin').write_bytes(b'abce')
                with self.assertRaises(ValueError): read()
                self.assertEqual(len(executions), 1)
                (evidence / 'original.bin').write_bytes(b'abcd')
            with scope.scope(): self.assertEqual(read(), report)
            self.assertEqual(len(executions), 2)
            with self.assertRaises(ValueError):
                scope.verify_once('q1_primary_quality', evidence, files, profile.canonical(dict(lock=True)),
                    reader, profile.sha(reader.read_bytes()), profile.sha(report), execute)

    def test_explicit_proof_layout_rejects_unknown_sibling_symlink_or_extra_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); origin = root / 'training-origin'; origin.mkdir()
            (origin / 'receipt.json').write_bytes(b'{}')
            bound = dict(files=profile.inventory(origin))
            with patch.object(profile, 'binding', return_value=bound):
                with profile.proof_scope(root, 'primary'): pass
                sibling = root / 'primary-quality'; sibling.mkdir()
                with profile.proof_scope(root, 'primary'): pass
                unknown = root / 'replica-quality'; unknown.mkdir()
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root, 'primary'): pass
                unknown.rmdir(); (origin / 'extra.json').write_bytes(b'{}')
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root, 'primary'): pass
                (origin / 'extra.json').unlink(); (origin / 'link').symlink_to(origin / 'receipt.json')
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root, 'primary'): pass

    def test_transport_preserves_exact_raw_api_bytes_and_only_specific_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); original = root / 'original'; original.mkdir()
            (original / 'run.json').write_bytes(b'{"original":true}\n'); (original / 'source.zip').write_bytes(b'PK\x03\x04\xff\x00')
            files = profile.inventory(original)
            locations = dict(run_json=None)
            locations = {'run.json': dict(original_api_base64=base64.b64encode((original / 'run.json').read_bytes()).decode()),
                         'source.zip': dict(artifact_id=123)}
            with patch.object(transport.subprocess, 'run') as network:
                transport.fetch_files(root / 'reuse', files, locations, original); network.assert_not_called()
            self.assertEqual(profile.inventory(root / 'reuse'), files)
            self.assertEqual((root / 'reuse/run.json').read_bytes(), b'{"original":true}\n')
            bad = copy.deepcopy(locations); bad['source.zip'] = dict(latest='main')
            with self.assertRaises(ValueError): profile.validate_transport(bad, files)
            (original / 'source.zip').write_bytes(b'changed')
            with self.assertRaises(ValueError): transport.fetch_files(root / 'rejected', files, locations, original)

    def test_whole_source_inventory_never_drops_rejects_or_native_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ('AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json', 'CORPUS_TRANSPORT.json'):
                (root / name).write_bytes(b'{}')
            (root / 'TRANSPORT').mkdir(); (root / 'TRANSPORT/receipt.json').write_bytes(b'{}')
            known = profile.inventory(root)
            for name in ('wav', 'flac', 'native'): (root / name).mkdir()
            naturals = {}
            for index in range(24):
                ident = str(index); stem = 'b27-primary-natural-' + ident
                (root / 'wav' / (stem + '.wav')).write_bytes(b'0' * 2044)
                (root / 'flac' / (stem + '.flac')).write_bytes(b'0' * 42)
                naturals[ident] = dict(source_id=ident, samples=1000, flac_bytes=42, wav_bytes=2044)
            for index in range(96):
                stem = 'synthetic-' + str(index)
                (root / 'wav' / (stem + '.wav')).write_bytes(b'0' * 2044)
                directory = root / 'native' / stem; directory.mkdir(); (directory / '000.s16le').write_bytes(b'0' * 2000)
            natural_bytes = sum(row['wav_bytes'] + row['flac_bytes'] for row in naturals.values())
            expanded = sum(pin['bytes'] for pin in known.values()) + capacity.base.TTS_AUDIO_BOUND + natural_bytes + capacity.base.NEW_METADATA_LIMIT
            value = dict(schema=capacity.SCHEMA, panel='primary', limit_bytes=capacity.SOURCE_LIMIT,
                original_artifact_limit_bytes=capacity.CORE_LIMIT, tts_audio_bytes_max=capacity.base.TTS_AUDIO_BOUND,
                new_metadata_bytes_max=capacity.base.NEW_METADATA_LIMIT, known_files=known,
                proof_binding_sha256=profile.sha(b'{}'), natural=dict(members=naturals, bytes=natural_bytes),
                maximum_expanded_bytes=expanded,
                maximum_zip_bytes=capacity.base.zip_bound(expanded, capacity.base.MAX_MEMBERS, capacity.base.MAX_MEMBERS * capacity.base.MAX_PATH_BYTES))
            (root / 'CAPACITY.json').write_bytes(profile.canonical(value))
            self.assertEqual(capacity.source_members(root, 'primary')['original_wavs'], 120)
            target = root / 'wav/b27-primary-natural-0.wav'; target.write_bytes(b'partial')
            with self.assertRaises(ValueError): capacity.source_members(root, 'primary')
            target.write_bytes(b'0' * 2044); extra = root / 'unretained.zip'; extra.write_bytes(b'PK')
            with self.assertRaises(ValueError): capacity.source_members(root, 'primary')

if __name__ == '__main__':
    unittest.main()
