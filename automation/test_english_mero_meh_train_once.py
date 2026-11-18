"""Offline source/binding boundaries; no model, decoder, subprocess or network."""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT/'automation'/(name+'.py')
    spec = importlib.util.spec_from_file_location(name+'_test_import', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


w = module('english_mero_meh_train_once')
b = module('bind_english_mero_meh_training')
c = module('english_mero_meh_training_core')


def digest(value):
    return c.typed_intake.digest(value)


def metadata(pin):
    head, before = '1'*40, '2'*40
    run = {'id': 91, 'head_sha': head, 'repository': {'full_name': w.REPO, 'private': False},
        'event': 'push', 'head_branch': 'main', 'status': 'completed', 'conclusion': 'success',
        'run_attempt': 1, 'path': w.FRESH_SOURCE['workflow_path']}
    artifact = {'id': 92, 'name': w.FRESH_SOURCE['artifact_prefix']+'91', 'expired': False,
        'workflow_run': {'id': 91, 'head_sha': head}, 'size_in_bytes': pin['bytes'],
        'digest': 'sha256:'+pin['sha256']}
    job = {'id': 93, 'name': 'combine', 'run_id': 91, 'head_sha': head,
        'status': 'completed', 'conclusion': 'success'}
    branch = {'name': 'main', 'commit': {'sha': before,
        'url': 'https://api.github.com/repos/'+w.REPO+'/commits/'+before}}
    return run, artifact, job, branch


def packet_fixture(root):
    """Explicit mock admission; production policy behavior has separate tests."""
    run = {'repository': w.REPO, 'run_id': 91, 'commit': '1'*40, 'run_attempt': 1, 'event': 'push'}
    contract = {'schema': 'english-source-identity-contract-v1',
        'purpose': 'synthetic_training_source_identity', 'diagnostic_only': False,
        'training_allowed': True, 'human_speaker_identity_claimed': False,
        'source_namespaces': ['kokoro-v1_0-preset'], 'known_lineage_edges': [], 'train_side_exposed': []}
    rows, admitted, evidence = [], [], []
    for group, role in enumerate(['train']*6+['development']*2+['calibration']*2+['locked-test']*2):
        preset = 'af_fixture'+chr(97+group)
        for label in (0, 1, 2):
            ident = 'mock-'+str(group)+'-'+str(label)
            pcm = bytes((group+1, label+1))*16000
            row = {'id': ident, 'role': role, 'source_namespace': 'kokoro-v1_0-preset',
                'source_id': preset, 'preset_id': preset, 'source_speaker_id': None,
                'source_group_id': 'kokoro-v1_0-preset:'+preset, 'diagnostic_only': False,
                'human_gold': False, 'generation_status': 'success', 'pcm_path': 'pcm/'+ident+'.pcm',
                'pcm_samples': 16000, 'pcm_sha256': w.sha(pcm),
                'parent_pcm_sha256': w.sha(('native/'+ident).encode()),
                'request_sha256': w.sha(('request/'+ident).encode())}
            rows.append(row)
            admitted.append({k: row[k] for k in ('id', 'role', 'source_namespace', 'source_id',
                'preset_id', 'source_group_id')} | {'supervision_allowed': True,
                'training_allowed': role == 'train', 'research_class': label,
                'training_class': label if role == 'train' else None, 'tier': 'silver',
                'training_weight': 1.0 if role == 'train' else 0.0,
                'pronunciation_hypothesis': 'mock-'+str(label),
                'matched_pronunciation_hypothesis': 'mock-'+str(label)})
            evidence.append({'id': ident})
            w.put(root/row['pcm_path'], pcm)
    corpus = {'schema': 'english-mero-meh-corpus-v1', 'campaign_id': w.FRESH_SOURCE['campaign_id'],
        'source_design_id': w.FRESH_SOURCE['campaign_id'], 'run': run, 'diagnostic_only': False,
        'training_allowed': True, 'source_identity_contract_sha256': digest(contract), 'rows': rows}
    admission = {'source_identity_contract_sha256': digest(contract),
        'training_snapshot_allowed': True, 'rows': admitted}
    exclusion = {'source_runs': [44202856867, 44208480336], 'excluded_identities': {
        'row_ids': ['consumed-fixture'], 'request_sha256': [w.sha(b'old-request')],
        'pcm_sha256': [w.sha(b'old-pcm')], 'parent_pcm_sha256': [w.sha(b'old-parent')]}}
    objects = {'CORPUS_MANIFEST.json': corpus, 'ADMISSION.json': admission,
        'MACHINE_EVIDENCE.json': {'rows': evidence}, 'SOURCE_IDENTITY_CONTRACT.json': contract,
        'EXCLUSION_EVIDENCE.json': exclusion,
        'EXECUTION_PLAN.json': {'campaign_id': w.FRESH_SOURCE['campaign_id']},
        'RESULT.json': {'stage': 'combine', 'status': 'MACHINE_EVIDENCE_PACKET_RETAINED', 'run': run}}
    for name, value in objects.items():
        w.put(root/name, w.canonical(value))
    policy = ('def build_admission(corpus, machine_evidence, identity_contract):\n    return '
              + repr(admission)+'\n').encode()
    w.put(root/w.FRESH_SOURCE['policy_path'], policy)
    pins = {p.relative_to(root).as_posix(): w.file_identity(p) for p in root.rglob('*') if p.is_file()}
    w.put(root/'FILES.json', w.canonical({'schema': w.FRESH_SOURCE['files_schema'],
        'files': [{'path': name, **pin} for name, pin in sorted(pins.items())]}))
    return objects, pins


def fixture_plan(root, source, pins):
    intake = b.intake_for_packet(root, source, w)
    return {'corpus_source': source, 'intake_binding': intake,
        'fresh_execution_plan': w.file_identity(root/'EXECUTION_PLAN.json'),
        'fresh_source_pins': {name: pin for name, pin in pins.items() if name.startswith('frozen_source/')}}


class CrossModuleWriterTests(unittest.TestCase):
    def test_actual_source_observer_writers_reach_typed_intake(self):
        if not (ROOT/'automation/test_english_mero_meh_once.py').is_file():
            self.skipTest('Full source campaign fixtures are absent from this minimal trainer staging tree.')
        original_path = list(sys.path)
        sys.path.insert(0, str(ROOT/'automation'))
        try:
            from test_english_mero_meh_once import mock_pipeline
            from test_english_mero_meh_generate import MOCK_RUN, MOCK_VOCAB
            with tempfile.TemporaryDirectory() as tmp:
                packet = mock_pipeline(Path(tmp))
                root = packet['output']
                source = {'run_id': int(MOCK_RUN['run_id']), 'artifact_id': 999991,
                    'name': w.FRESH_SOURCE['artifact_prefix']+str(MOCK_RUN['run_id']),
                    'head_sha': MOCK_RUN['commit'], 'bytes': 1234, 'sha256': 'f'*64}
                manifest = w.loads(w.regular(root/'FILES.json'))
                pins = {r['path']: {k: r[k] for k in ('bytes', 'sha256')} for r in manifest['files']}
                plan = fixture_plan(root, source, pins)
                real_loader = w._load_source_component
                def mock_native_initialization_only(name):
                    loaded = real_loader(name)
                    if name == 'generate':
                        # Artificial audio cannot claim a real native model lock.
                        # All source/observer byte readers and policies stay real.
                        loaded.verify_native_lock = lambda unused_root: MOCK_VOCAB
                    return loaded
                with mock.patch.object(w, '_load_source_component', side_effect=mock_native_initialization_only):
                    accepted = w.source_packet(root, plan,
                        expected_binding_sha256=digest(plan['intake_binding']))
                    self.assertTrue(accepted['byte_readback']['all_machine_observations_reconstructed'])
                    self.assertEqual(accepted['identity_proof']['groups_by_role'],
                        {'train': 6, 'development': 2, 'calibration': 2, 'locked-test': 2})
                    self.assertEqual(len(accepted['admission']['rows']), 144)
                    machine = w.loads(w.regular(root/'MACHINE_EVIDENCE.json'))
                    machine['rows'][0]['observations']['qwen']['raw_transcript'] = 'crafted evidence'
                    (root/'MACHINE_EVIDENCE.json').write_bytes(w.canonical(machine))
                    # Even a newly frozen hash and refreshed container manifest
                    # cannot replace the retained observer output bytes.
                    for record in manifest['files']:
                        if record['path'] == 'MACHINE_EVIDENCE.json':
                            record.update(w.file_identity(root/'MACHINE_EVIDENCE.json'))
                    (root/'FILES.json').write_bytes(w.canonical(manifest))
                    plan['intake_binding'] = b.intake_for_packet(root, source, w)
                    with self.assertRaisesRegex(ValueError, 'BYTE_RECOMPUTED_MACHINE_EVIDENCE:rows'):
                        w.source_packet(root, plan,
                            expected_binding_sha256=digest(plan['intake_binding']))
        finally:
            sys.path[:] = original_path


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        # These small fixtures isolate binding/typed-intake behavior. The source
        # orchestration suite exercises the real byte readers on actual writer
        # output. Production has no byte-readback bypass or environment switch.
        patch = mock.patch.object(w, 'byte_evidence_readback', return_value={
            'new_model_calls': 0, 'all_machine_observations_reconstructed': True})
        self.byte_reader = patch.start()
        self.addCleanup(patch.stop)

    def test_no_numeric_runtime_imported(self):
        self.assertFalse({'torch', 'numpy', 'onnx', 'onnxruntime'}.intersection(sys.modules))

    def test_current_source_closure(self):
        plan = w.loads(w.regular(ROOT/w.PLAN_PATH))
        w.source_preflight(plan)
        self.assertEqual(plan['enabled'], c.SOURCE_IDENTITY_CANDIDATE_ENABLED)

    def test_numeric_ast_unchanged(self):
        tree = ast.parse((ROOT/w.CORE_PATH).read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in
                {'SOURCE_IDENTITY_CANDIDATE_ENABLED', 'BOUND_INTAKE_BINDING_SHA256'} for t in node.targets))]
        self.assertEqual(hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            '3314f05bf5d6927265b446c430374679a8264c54ebcd833e5eab1ea4b319ecd6')

    def test_actual_metadata_binding(self):
        pin = {'bytes': 15, 'sha256': 'a'*64}
        source, before = b.validate_metadata(*metadata(pin), pin, w.FRESH_SOURCE)
        self.assertEqual(source['run_id'], 91)
        self.assertEqual(before, '2'*40)

    def test_metadata_cannot_relabel_private_retry_wrong_workflow(self):
        pin = {'bytes': 15, 'sha256': 'a'*64}
        for field, bad in (('run_attempt', 2), ('event', 'workflow_dispatch'),
                ('head_branch', 'other'), ('path', '.github/workflows/english-pocket12-once.yml')):
            with self.subTest(field=field):
                values = metadata(pin); values[0][field] = bad
                with self.assertRaises(ValueError):
                    b.validate_metadata(*values, pin, w.FRESH_SOURCE)
        values = metadata(pin); values[0]['repository']['private'] = True
        with self.assertRaises(ValueError):
            b.validate_metadata(*values, pin, w.FRESH_SOURCE)

    def test_metadata_artifact_and_job_fail_closed(self):
        pin = {'bytes': 15, 'sha256': 'a'*64}
        for index, field, value in ((1, 'expired', True), (1, 'digest', 'sha256:'+'b'*64),
                (1, 'size_in_bytes', 16), (2, 'conclusion', 'failure'), (2, 'run_id', 2)):
            with self.subTest(field=field):
                values = metadata(pin); values[index][field] = value
                with self.assertRaises(ValueError):
                    b.validate_metadata(*values, pin, w.FRESH_SOURCE)

    def test_enable_only_disabled_copy(self):
        raw = b'SOURCE_IDENTITY_CANDIDATE_ENABLED = False\nBOUND_INTAKE_BINDING_SHA256 = None\n'
        enabled = b.enabled_core(raw, 'c'*64)
        self.assertIn(b"= '"+b'c'*64+b"'", enabled)
        with self.assertRaises(ValueError):
            b.enabled_core(enabled, 'd'*64)
        wf = b.enabled_workflow(b'    if: ${{ false }}\n', '2'*40)
        self.assertIn(b'github.run_attempt == 1', wf)

    def test_complete_typed_packet_with_external_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, pins = packet_fixture(root)
            source, _ = b.validate_metadata(*metadata({'bytes': 15, 'sha256': 'a'*64}),
                {'bytes': 15, 'sha256': 'a'*64}, w.FRESH_SOURCE)
            plan = fixture_plan(root, source, pins)
            packet = w.source_packet(root, plan, expected_binding_sha256=digest(plan['intake_binding']))
            self.byte_reader.assert_called_once()
            self.assertEqual(packet['identity_proof']['planned_row_count'], 36)
            self.assertFalse(packet['identity_proof']['probe_reuse_allowed'])
            with self.assertRaises(ValueError):
                w.source_packet(root, plan, expected_binding_sha256='f'*64)

    def test_saved_admission_is_not_policy_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); objects, pins = packet_fixture(root)
            source, _ = b.validate_metadata(*metadata({'bytes': 15, 'sha256': 'a'*64}),
                {'bytes': 15, 'sha256': 'a'*64}, w.FRESH_SOURCE)
            changed = copy.deepcopy(objects['ADMISSION.json']); changed['rows'][0]['research_class'] = 1
            (root/'ADMISSION.json').write_bytes(w.canonical(changed))
            plan = fixture_plan(root, source, pins)
            with self.assertRaisesRegex(ValueError, 'RECOMPUTED_MACHINE_ADMISSION'):
                w.source_packet(root, plan, expected_binding_sha256=digest(plan['intake_binding']))

    def test_actual_pcm_tampering_rejected_before_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); objects, pins = packet_fixture(root)
            source, _ = b.validate_metadata(*metadata({'bytes': 15, 'sha256': 'a'*64}),
                {'bytes': 15, 'sha256': 'a'*64}, w.FRESH_SOURCE)
            plan = fixture_plan(root, source, pins)
            (root/objects['CORPUS_MANIFEST.json']['rows'][0]['pcm_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'ACTUAL_SOURCE_PCM_BYTES'):
                w.source_packet(root, plan, expected_binding_sha256=digest(plan['intake_binding']))

    def test_full_old_probe_exclusions_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); objects, pins = packet_fixture(root)
            source, _ = b.validate_metadata(*metadata({'bytes': 15, 'sha256': 'a'*64}),
                {'bytes': 15, 'sha256': 'a'*64}, w.FRESH_SOURCE)
            changed = copy.deepcopy(objects['EXCLUSION_EVIDENCE.json']); changed['source_runs'].remove(44208480336)
            (root/'EXCLUSION_EVIDENCE.json').write_bytes(w.canonical(changed))
            plan = fixture_plan(root, source, pins)
            with self.assertRaisesRegex(ValueError, 'COMPLETE_CONTEXT64_AND_POCKET12'):
                w.source_packet(root, plan, expected_binding_sha256=digest(plan['intake_binding']))

    def test_consumed_request_cannot_be_renamed_into_new_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); objects, pins = packet_fixture(root)
            source, _ = b.validate_metadata(*metadata({'bytes': 15, 'sha256': 'a'*64}),
                {'bytes': 15, 'sha256': 'a'*64}, w.FRESH_SOURCE)
            changed = copy.deepcopy(objects['EXCLUSION_EVIDENCE.json'])
            changed['excluded_identities']['request_sha256'].append(
                objects['CORPUS_MANIFEST.json']['rows'][0]['request_sha256'])
            (root/'EXCLUSION_EVIDENCE.json').write_bytes(w.canonical(changed))
            plan = fixture_plan(root, source, pins)
            with self.assertRaisesRegex(ValueError, 'CONSUMED_PROBE_BYTES:request_sha256'):
                w.source_packet(root, plan, expected_binding_sha256=digest(plan['intake_binding']))

    def test_archive_exact_members_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source_root = root/'packet'; source_root.mkdir()
            packet_fixture(source_root)
            path = root/'source.zip'
            with zipfile.ZipFile(path, 'w') as z:
                for p in source_root.rglob('*'):
                    if p.is_file(): z.writestr(p.relative_to(source_root).as_posix(), p.read_bytes())
            pin = w.file_identity(path); w.extract_archive(path, root/'extracted', pin)
            with zipfile.ZipFile(path, 'a') as z:
                z.writestr('unexpected.txt', b'unlisted')
            with self.assertRaisesRegex(ValueError, 'FRESH_EXACT_ARCHIVE_CLOSURE'):
                w.extract_archive(path, root/'other', w.file_identity(path))

    def test_complete_offline_binding_writes_no_training_result(self):
        if w.loads(w.regular(ROOT/w.PLAN_PATH))['enabled']:
            self.skipTest('One-time binder fixture applies to the disabled source stage.')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source_root = root/'packet'; source_root.mkdir()
            _, _pins = packet_fixture(source_root)
            trainer = root/'trainer'; disabled = w.loads(w.regular(ROOT/w.PLAN_PATH))
            for name in set(disabled['source_pins']) | {w.WORKER_PATH, w.PLAN_PATH}:
                w.put(trainer/name, w.regular(ROOT/name))
            policy_relative = w.FRESH_SOURCE['policy_path'].removeprefix('frozen_source/')
            w.put(trainer/policy_relative, w.regular(source_root/w.FRESH_SOURCE['policy_path']))
            path = root/'source.zip'
            with zipfile.ZipFile(path, 'w') as z:
                for p in source_root.rglob('*'):
                    if p.is_file(): z.writestr(p.relative_to(source_root).as_posix(), p.read_bytes())
            args = {'trainer_root': trainer, 'corpus_zip': path, 'output': root/'bound'}
            for name, value in zip(('run', 'artifact', 'combine_job', 'main'), metadata(w.file_identity(path))):
                location = root/(name+'.json'); location.write_bytes(w.canonical(value))
                args[name+'_metadata'] = location
            real_loader = b.load_module
            def fixture_loader(path, name):
                loaded = real_loader(path, name)
                if name == 'mero_meh_offline_worker':
                    loaded.byte_evidence_readback = self.byte_reader
                return loaded
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(b, 'load_module', side_effect=fixture_loader):
                self.assertEqual(b.bind(argparse.Namespace(**args)), 0)
            result = w.loads(w.regular(root/'bound/BINDING_RECEIPT.json'))
            self.assertFalse(result['training_executed'])
            self.assertEqual(result['new_model_calls'], 0)
            self.assertTrue((root/'bound'/w.CORE_PATH).is_file())
            self.assertFalse((root/'bound/models').exists())


if __name__ == '__main__':
    unittest.main()
