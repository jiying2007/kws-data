"""Pure stdlib fixtures; invented outputs never authorize any real execution."""
from __future__ import annotations
import ast
import base64
import copy
import io
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile
import english_v6_b27_evidence_scope as scope
import english_v6_b24q1_evidence_scope as old_scope
import english_v6_b27_primary_eval as primary
import english_v6_b27_replication_eval as replica
import english_v6_b27_quality_readback as quality
import english_v6_b27_source as source
import english_v6_b27_replication_source as replica_source
import english_v6_b27_natural_sources as natural
import english_v6_b27_replication_natural_sources as replica_natural
import english_v6_b27_capacity as capacity
import english_v6_b27_original_proof as transport
import english_v6_b27_source_producer as source_producer
import bind_b27_quality as binder
import test_english_v6_b24q1_quality_readback as fixtures

HERE = Path(__file__).resolve().parent
CANDIDATE = '9' * 64


def syntax_function(module, name):
    return next(node for node in ast.parse(Path(module.__file__).read_bytes()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def fixture(panel='primary', failed=False):
    driver = primary if panel == 'primary' else replica
    with patch.object(fixtures, 'primary', driver), patch.object(fixtures, 'reader', quality), patch.object(driver, 'CANDIDATE_PT', CANDIDATE):
        members, spec, admission, rows = fixtures.fixture(panel)
    policy = dict(spec['source_policy'], schema='english-v6-b27-source-policy-v1', experiment_id='B27',
                  b27_execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()))
    spec['source_policy'] = policy
    admission['source_audit'].update(schema='english-v6-b27-full-source-and-quarantine-audit-v1', source_policy=policy)
    admission['output_capacity'] = capacity.score_budget(rows)
    result = quality.document(members['RESULT.json'])
    started = quality.document(members['STARTED.json'])
    for item in (result, started):
        item.update(source_audit=admission['source_audit'], B24Q1_failed_primary_preserved=True,
                    all_old_stops_and_reserves_preserved=True)
    if failed:
        with zipfile.ZipFile(io.BytesIO(members['ALL_FRAME_LOGITS.npz'])) as archive:
            arrays = {name: archive.read(name) for name in archive.namelist()}
        record = result['records'][0]; key = record['id'] + '__1.0__challenger'
        labels = [0] * 17; raw, body = fixtures.npy(labels)
        arrays[key + '.npy'] = raw; members['raw_outputs/' + key + '.npy'] = fixtures.npy(labels, True)[0]
        record['gains']['1.0']['challenger'] = dict(array=key, frames=17, sha256=quality.sha(body), **driver.ctc_events(labels))
        members['ALL_FRAME_LOGITS.npz'] = fixtures.zip_bytes(arrays)
        result.update(fixtures.decision.decide(result['records']), logits_sha256=quality.sha(members['ALL_FRAME_LOGITS.npz']),
                      replication_open_allowed=False, candidate_engineering_validation_allowed=False)
        if panel == 'replication': result['research_replication_passed'] = False
    members.update({'RESULT.json': quality.canonical(result), 'STARTED.json': quality.canonical(started),
                    'ADMISSION.json': quality.canonical(admission)})
    return members, spec, admission, rows, driver


def copied(value):
    return (*copy.deepcopy(value[:4]), value[4])


def verify(value):
    members, spec, admission, rows, driver = value
    panel = 'replication' if driver is replica else 'primary'
    with patch.object(driver, 'CANDIDATE_PT', CANDIDATE):
        return quality.saved_arrays(fixtures.zip_bytes(members), spec, panel, admission, rows, driver, fixtures.decision)


def pair_fixture():
    hashes = dict(B0=primary.B0_PT, B9=primary.B9_PT, control=profile.CONTROL_PT, challenger=CANDIDATE)
    result = dict(schema=primary.PAIR_SCHEMA, status='TRAINED_NATURAL_WEIGHT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY',
        actual_optimizer_updates=192, actual_new_model_forwards=1847, natural_weight=0.16,
        model_pt_sha256=hashes, assessment_sha256=primary.TRAINING_ASSESSMENT,
        design_sha256=profile.DESIGN_SHA, recipe_sha256=profile.DESIGN_SHA,
        phase_initialization_pt_sha256=profile.INITIAL_PT, optimizer_initialization='fresh_AdamW',
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        original_B24Q1_NO_PROMOTION_reauthenticated=True,
        **dict.fromkeys(('restored_optimizer_state', 'exact_optimizer_continuation', 'historical_numerical_replay_claimed',
            'same_job_matched_control', 'promotion_allowed', 'Q1_metrics_used_to_choose_parameters'), False),
        **dict.fromkeys(('old_B23_optimizer_updates_reexecuted', 'old_B24_optimizer_updates_reexecuted',
            'new_control_optimizer_updates', 'new_control_forwards', 'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls'), 0))
    for role, stem in primary.STEMS.items():
        result[role] = dict(files={Path(stem).name + '.' + ext: dict(bytes=20_000,
            sha256=hashes[role] if ext == 'pt' else 'a' * 64) for ext in ('pt', 'npz', 'onnx')})
    result['control']['output_stem'] = primary.STEMS['control']
    result['challenger'].update(stem=primary.STEMS['challenger'], parent_pt_sha256=profile.INITIAL_PT,
        historical_control_pt_sha256=profile.CONTROL_PT, optimization_forwards=1844, checkpoint_reload_forwards=2, onnx_export_forwards=1)
    spec = dict(model_pt_sha256=hashes, training_assessment_sha256=primary.TRAINING_ASSESSMENT, training_design_sha256=profile.DESIGN_SHA)
    return result, spec


class ProfileContracts(unittest.TestCase):
    def test_binder_producer_contract_reopens_complete_captured_resources(self):
        repository = HERE.parent
        for owner in (primary, replica):
            with self.subTest(panel=owner.PANEL), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source_names = source_producer.runtime_paths(repository, owner.PANEL)
                # Preserve a template if an actual upstream captured it; this
                # regression also covers the former broad SOURCE closure.
                source_names |= {owner.SOURCE_WORKFLOW, 'automation/' + owner.WORKFLOW_TEMPLATE.name,
                    'automation/english_v6_b27_execution_profile.json',
                    'automation/english_v6_b27_' + owner.PANEL + '_proof_binding.json'}
                if owner.PANEL == 'replication':
                    source_names |= {'automation/english_v6_b27_primary_inputs.json',
                                     'automation/english_v6_b27_primary_producer.json'}
                def pins(names):
                    return {name: dict(bytes=(repository / name).stat().st_size,
                        sha256=profile.sha((repository / name).read_bytes())) for name in names}
                snapshots = dict(source=dict(files=pins(source_names), runtime_workflow=owner.SOURCE_WORKFLOW),
                    train=dict(files=pins({'automation/english_v6_b14_decision.py', owner.TRAIN_WORKFLOW}),
                               runtime_workflow=owner.TRAIN_WORKFLOW))
                spec = {}
                if owner.PANEL == 'replication':
                    spec['primary_quality_lock'] = dict(inputs_sha256=profile.sha(primary.SPEC.read_bytes()),
                        producer_manifest_sha256=profile.sha(primary.PRODUCER.read_bytes()))
                doc, code, resources = binder.producer_document(owner, owner.PANEL, snapshots, spec, repository)
                template_name = 'automation/' + owner.WORKFLOW_TEMPLATE.name
                self.assertIn(template_name, resources)
                names = code | resources | {template_name, 'automation/' + owner.SPEC.name}
                for name in names:
                    target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(repository / name, target)
                parent = 'a' * 40
                workflow = root / owner.WORKFLOW; workflow.parent.mkdir(parents=True, exist_ok=True)
                workflow.write_bytes((root / template_name).read_bytes().replace(owner.PARENT_TOKEN.encode(), parent.encode()))
                producer_path = root / 'automation' / owner.PRODUCER.name
                def write(value):
                    raw = profile.canonical(value); producer_path.write_bytes(raw)
                    spec['producer_manifest_sha256'] = profile.sha(raw)
                write(doc)
                actual, activated = owner.producer_contract(spec, parent, root)
                self.assertEqual(actual, doc); self.assertEqual(activated, workflow.read_bytes())
                missing = copy.deepcopy(doc); del missing['resources'][template_name]; write(missing)
                with self.assertRaises(ValueError): owner.producer_contract(spec, parent, root)
                write(doc)
                member = next(iter(doc['source']['files']))
                (root / member).write_bytes((root / member).read_bytes() + b'\n# altered original\n')
                with self.assertRaises(ValueError): owner.producer_contract(spec, parent, root)

    def test_original_b14_and_numerical_scoring_loop_are_exact(self):
        self.assertEqual(profile.profile(False)['fixed_base_sha256']['english_v6_b14_decision.py'], primary.QUALITY_CODE)
        self.assertEqual(profile.sha(Path(fixtures.decision.__file__).read_bytes()), primary.QUALITY_CODE)
        for name in ('snapshots', 'Ledger'):
            self.assertEqual(ast.dump(syntax_function(primary, name), include_attributes=False),
                             ast.dump(syntax_function(primary.original, name), include_attributes=False))
        before = next(node for node in syntax_function(primary.original, 'score').body if isinstance(node, ast.Try))
        after = next(node for node in syntax_function(primary, 'score').body if isinstance(node, ast.Try))
        self.assertEqual([ast.dump(n, include_attributes=False) for n in before.body[:-1]],
                         [ast.dump(n, include_attributes=False) for n in after.body[:-1]])

    def test_active_inherited_globals_use_only_the_b27_profile(self):
        for owner, panel in ((source, 'primary'), (replica_source, 'replication')):
            for name in ('observe', 'screen_document', 'registration', 'evaluation_preflight'):
                namespace = getattr(owner, name).__globals__
                self.assertIs(namespace['lineage'], owner.lineage)
                self.assertEqual(namespace['PANEL'], panel)
            for name in ('expected_registration', 'require_original_rows', 'validate_rows'):
                self.assertIs(getattr(owner.observers, name).__globals__['lineage'], owner.lineage)
            self.assertIs(owner.observers.owner(), owner)
        self.assertEqual(primary.CONTROL_PT, profile.CONTROL_PT)
        self.assertNotEqual(primary.CONTROL_PT, primary.INITIAL_PT)
        self.assertEqual(primary.TRAINING_ASSESSMENT, primary.EVALUATION_ASSESSMENT)
        self.assertNotEqual(primary.TRAINING_ASSESSMENT, primary.ORIGINAL_B24_TRAINING_ASSESSMENT)
        for owner in (primary, replica):
            for name in ('guard', 'validate_execution_metadata', 'original_zip', 'validate_pair_receipt'):
                self.assertEqual(getattr(owner, name).__globals__['MODEL_COMPARISON'], primary.MODEL_COMPARISON)
                self.assertIs(getattr(owner, name).__globals__['SPEC'], owner.SPEC)

    def test_replica_natural_roster_never_resolves_primary_helper(self):
        self.assertIs(replica_natural.original, natural.original)
        with patch.object(natural, 'plan', side_effect=AssertionError('primary roster escaped')), \
             patch.object(replica_natural, 'plan', return_value={'roster': ['replica']}), \
             patch.object(natural.original, 'rank_metadata', return_value=['replica']), \
             patch.object(natural.original, 'pick', return_value=(['replica'], {'metadata': True})):
            self.assertEqual(replica_natural.rank_metadata({}, {}), ['replica'])
            self.assertEqual(replica_natural.pick(None, {}), (['replica'], {'metadata': True}))

    def test_inert_inputs_and_missing_proof_fail_before_any_call(self):
        for owner in (primary, replica):
            inert = profile.document(owner.SPEC.read_bytes())
            with patch.object(owner, 'gh') as network, patch.object(owner, 'snapshots') as model:
                if inert['scoring_allowed'] is False:
                    with self.assertRaisesRegex(ValueError, 'INERT'): owner.protocol()
                with self.assertRaises(ValueError): profile.require_proof(owner.PANEL)
                network.assert_not_called(); model.assert_not_called()
        for panel in ('primary', 'replication'):
            value = profile.binding(panel, False)
            if value['status'].startswith('INERT'):
                with self.assertRaises(ValueError): profile.binding(panel)

    def test_pair_cannot_be_q1_old_b24_or_b23_quality_control(self):
        receipt, spec = pair_fixture(); primary.validate_pair_receipt(receipt, spec)
        for field, bad in (('schema', primary.original.PAIR_SCHEMA), ('natural_weight', 0.08),
                ('assessment_sha256', primary.ORIGINAL_B24_TRAINING_ASSESSMENT),
                ('status', 'STOP'), ('actual_new_model_forwards', 1846),
                ('original_B24Q1_NO_PROMOTION_reauthenticated', False), ('Q1_metrics_used_to_choose_parameters', True)):
            value = copy.deepcopy(receipt); value[field] = bad
            with self.assertRaises(ValueError, msg=field): primary.validate_pair_receipt(value, spec)
        for role, wrong in (('control', profile.INITIAL_PT), ('challenger', profile.CONTROL_PT), ('challenger', profile.INITIAL_PT)):
            changed = copy.deepcopy(spec); changed['model_pt_sha256'][role] = wrong
            with self.assertRaises(ValueError): primary.validate_pair_receipt(receipt, changed)

    def test_both_panel_guards_reject_q1_rerun_forced_or_different_parent(self):
        spec = dict(prepared_parent=None, parent_authority='reviewed_workflow')
        for owner in (primary, replica):
            env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=owner.REPO, GITHUB_EVENT_NAME='push',
                GITHUB_REF=owner.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA='a' * 40, GITHUB_RUN_ID='123')
            event = dict(before='b' * 40, after='a' * 40, ref=owner.BRANCH, forced=False, deleted=False,
                         repository=dict(full_name=owner.REPO, private=False))
            self.assertEqual(owner.guard(spec, 'b' * 40, env, event, 'a' * 40, 'b' * 40)['run_id'], '123')
            for bad_env, bad_event, parent in ((dict(env, GITHUB_REF=primary.original.BRANCH), event, 'b' * 40),
                    (dict(env, GITHUB_RUN_ATTEMPT='2'), event, 'b' * 40), (env, dict(event, forced=True), 'b' * 40),
                    (env, event, 'c' * 40)):
                with self.assertRaises(ValueError): owner.guard(spec, 'b' * 40, bad_env, bad_event, 'a' * 40, parent)

    def test_workflows_are_inert_exact_templates_with_closed_child_gates(self):
        for panel in ('primary', 'replication'):
            for stage in ('source', 'quality'):
                if stage == 'source':
                    template = HERE / ('english_v6_b27_' + ('' if panel == 'primary' else 'replication_') + 'source_workflow_template.yml')
                    path = HERE.parent / '.github/workflows' / ('english-v6-b27-' + panel + '-sources-once.yml')
                    token = '__B27_REVIEWED_' + panel.upper() + '_SOURCE_PARENT__'
                else:
                    template = HERE / ('english_v6_b27_' + panel + '_workflow_template.yml')
                    path = HERE.parent / '.github/workflows' / ('english-v6-b27-' + panel + '-model-eval-once.yml')
                    token = '__B27_REVIEWED_' + panel.upper() + '_SCORING_PARENT__'
                raw = template.read_text()
                self.assertEqual(raw.count(token), 2)
                # A separately reviewed two-token activation may coexist with this inert test.
                runtime = path.read_text()
                if token in runtime: self.assertEqual(runtime, raw)
                self.assertIn("github.event_name == 'push'", raw)
                self.assertIn("github.run_attempt == 1", raw)
                self.assertIn("github.ref == 'refs/heads/research/en-v6-b27-" + panel, raw)
                self.assertIn('cancel-in-progress: false', raw)
                self.assertIn('include-hidden-files: true', raw)
                self.assertIn('english_v6_b27_original_proof.py" ' + panel if stage == 'source'
                              else 'english_v6_b27_original_proof.py ' + panel, raw)
                self.assertNotIn('workflow_dispatch', raw)
                self.assertNotIn('originals/*.zip', raw)


class CompleteArrayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = {(p, failed): fixture(p, failed) for p in ('primary', 'replication') for failed in (False, True)}

    def test_complete_pass_and_complete_failure_recompute_for_both_panels(self):
        for (panel, failed), value in self.values.items():
            checked = verify(value)['result']
            self.assertEqual(checked['panel_gate'], 'NO_PROMOTION' if failed else 'PASS')
            self.assertEqual(checked['actual_forwards'], 576)
            self.assertEqual(checked['replication_open_allowed'], panel == 'primary' and not failed)
            self.assertEqual(checked['candidate_engineering_validation_allowed'], panel == 'replication' and not failed)

    def test_stop_unknown_partial_and_forged_outcome_never_open(self):
        for field, wrong in (('panel_gate', 'STOP'), ('panel_gate', 'NO_PROMOTION'), ('panel_gate', 'UNKNOWN'),
                ('consumed_unknown', ['one']), ('actual_forwards', 575), ('raw_output_count', 575), ('error_type', 'HiddenFailure')):
            value = copied(self.values['primary', False]); report = quality.document(value[0]['RESULT.json'])
            report[field] = wrong; value[0]['RESULT.json'] = quality.canonical(report)
            with self.assertRaises(ValueError, msg=field): verify(value)

    def test_raw_tensor_journal_clock_and_aggregate_tampering_reject(self):
        for kind in ('raw', 'journal', 'clock', 'orphan', 'aggregate'):
            value = copied(self.values['primary', False]); members = value[0]
            report = quality.document(members['RESULT.json'])
            if kind == 'raw': members['raw_outputs/fixture-1__1.0__B0.npy'] = fixtures.npy([0] * 17, True)[0]
            if kind == 'journal': del members['attempts/fixture-1__1.0__B0.json']
            if kind == 'clock': report['records'][1]['gains']['1.0']['B0']['event_samples'] = [999]
            if kind == 'orphan': members['raw_outputs/unregistered.npy'] = fixtures.npy([0] * 17, True)[0]
            if kind == 'aggregate': report['restored_positives'] = []
            members['RESULT.json'] = quality.canonical(report)
            with self.assertRaises(ValueError, msg=kind): verify(value)

    def test_failure_cannot_claim_downstream_permission_or_erase_old_stop(self):
        for panel in ('primary', 'replication'):
            for field, wrong in (('replication_open_allowed', True), ('candidate_engineering_validation_allowed', True),
                    ('promotion_allowed', True), ('B24Q1_failed_primary_preserved', False), ('all_old_stops_and_reserves_preserved', False)):
                value = copied(self.values[panel, True]); report = quality.document(value[0]['RESULT.json'])
                report[field] = wrong; value[0]['RESULT.json'] = quality.canonical(report)
                with self.assertRaises(ValueError, msg=field): verify(value)

    def test_old_q1_assessment_or_control_cannot_enter_new_array_readback(self):
        for field, wrong in (('assessment_sha256', primary.original.EVALUATION_ASSESSMENT),
                             ('training_assessment_sha256', primary.ORIGINAL_B24_TRAINING_ASSESSMENT)):
            value = copied(self.values['primary', False]); value[1][field] = wrong
            with self.assertRaises(ValueError): verify(value)
        value = copied(self.values['primary', False]); value[1]['model_pt_sha256']['control'] = profile.INITIAL_PT
        with self.assertRaises(ValueError): verify(value)

    def test_source_population_quarantine_or_capacity_cannot_be_edited(self):
        for field, wrong in (('qualified_rows', 47), ('quarantined_kws_forwards', 1), ('observer_eligible_rows', 116)):
            value = copied(self.values['primary', False]); value[2]['source_audit'][field] = wrong
            with self.assertRaises(ValueError): verify(value)
        value = copied(self.values['primary', False]); value[2]['source_audit']['quarantined_ids'].append('fixture-0')
        with self.assertRaises(ValueError): verify(value)
        value = copied(self.values['primary', False]); value[2]['output_capacity']['limit_bytes'] += 1
        with self.assertRaises(ValueError): verify(value)


class ExternalProofContracts(unittest.TestCase):
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

    def test_unknown_quality_lock_rejects_before_network(self):
        for lock in ({}, dict(schema='english-v6-b24q1-original-quality-lock-v1'), dict(schema=quality.LOCK_SCHEMA, panel='unknown')):
            with patch.object(quality.subprocess, 'run') as network, patch.object(quality.subprocess, 'check_output') as child:
                with self.assertRaises((ValueError, KeyError)):
                    quality.fetch_quality_original(Path('/never-created-b27-fixture'), lock, Path('/missing-proof'))
                network.assert_not_called(); child.assert_not_called()

    def test_offline_guard_only_allows_exact_scoped_training_child_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); origin = root / 'training-origin'; directory = origin / 'trained'; inputs = origin / 'qualification'
            directory.mkdir(parents=True); inputs.mkdir(); (directory / 'TRAIN_LIVE.json').write_bytes(b'{}')
            (inputs / 'receipt.json').write_bytes(b'{}')
            reader = HERE / 'english_v6_b27_training_readback.py'; bound_lock = dict(fixture=True)
            lock = root / 'lock.json'; lock.write_bytes(profile.canonical(bound_lock))
            bound = dict(training_reader_sha256=profile.sha(reader.read_bytes()), training_lock=bound_lock, files=profile.inventory(origin))
            command = [sys.executable, '-I', '-S', '-B', str(reader), '--verify-worker', str(directory), str(lock), str(inputs)]
            with patch.object(profile, 'binding', return_value=bound), patch.object(subprocess, 'Popen') as launch:
                with quality.offline_only(root, 'primary'):
                    subprocess.Popen(command, stdout=subprocess.PIPE)
                    with self.assertRaises(ValueError): subprocess.Popen(command, stdout=subprocess.PIPE)
                    for index, bad in ((1, '-E'), (5, '--fetch'), (6, str(root)), (8, str(root))):
                        changed = list(command); changed[index] = bad
                        with self.assertRaises(ValueError): subprocess.Popen(changed)
                    with self.assertRaises(ValueError): subprocess.Popen(command, env={})
                    with self.assertRaises(ValueError): subprocess.Popen(['gh', 'api', 'anything'])
                    with self.assertRaises(ValueError): socket.socket()
                    with self.assertRaises(ValueError): __import__('numpy')
                self.assertEqual(launch.call_count, 1)

    def test_replica_offline_guard_only_allows_exact_original_primary_child_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'primary-quality'; directory.mkdir()
            names = {'api/' + name + '.json' for name in ('run', 'commit', 'artifacts')}
            names |= {'originals/' + name + '.zip' for name in ('result', 'producer', 'consumed')}
            names |= {'inputs/' + name + '.zip' for name in binder.ROLES}
            self.assertEqual(len(names), 18)
            for name in names:
                target = directory / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'fixture')
            reader = Path(quality.__file__).resolve()
            bound_lock = dict(schema=quality.LOCK_SCHEMA, panel='primary', fixture=True)
            lock = root / 'lock.json'; lock.write_bytes(profile.canonical(bound_lock))
            bound = dict(primary_quality_reader_sha256=profile.sha(reader.read_bytes()),
                         primary_quality_lock=bound_lock, primary_quality_files=profile.inventory(directory))
            command = [sys.executable, '-I', '-S', '-B', str(reader), '--verify-worker', str(directory), str(lock), str(root)]
            with patch.object(profile, 'binding', return_value=bound), patch.object(subprocess, 'Popen') as launch:
                with quality.offline_only(root, 'replication'):
                    subprocess.Popen(command, stdout=subprocess.PIPE)
                    with self.assertRaises(ValueError): subprocess.Popen(command, stdout=subprocess.PIPE)
                self.assertEqual(launch.call_count, 1)
                def rejected(value=command, panel='replication'):
                    with quality.offline_only(root, panel):
                        with self.assertRaises(ValueError): subprocess.Popen(value, stdout=subprocess.PIPE)
                rejected(panel='primary')
                for index, wrong in ((4, str(HERE / 'english_v6_b24q1_quality_readback.py')),
                        (5, '--fetch'), (6, str(root)), (8, str(directory))):
                    changed = list(command); changed[index] = wrong; rejected(changed)
                lock.write_bytes(profile.canonical(dict(bound_lock, fixture=False))); rejected()
                lock.write_bytes(profile.canonical(bound_lock) + b'\n'); rejected()
                lock.write_bytes(profile.canonical(bound_lock))
                digest = bound['primary_quality_reader_sha256']; bound['primary_quality_reader_sha256'] = '0' * 64
                rejected(); bound['primary_quality_reader_sha256'] = digest
                extra = directory / 'unbound.json'; extra.write_bytes(b'{}'); rejected(); extra.unlink()
                self.assertEqual(launch.call_count, 1)


class CapacityContracts(unittest.TestCase):
    def test_frozen_role_limits_and_maximum_complete_output_fit(self):
        value = capacity.score_budget([dict(samples=240000)] * 120)
        self.assertEqual(value['arrays'], 1440)
        self.assertLessEqual(value['maximum_zip_bytes'], 128_000_000)
        for panel in ('primary', 'replication'):
            self.assertEqual(quality.archive_limit(panel, 'source'), 256_000_000)
            for role in ('pair', 'producer', 'result', 'train_producer', 'native_whisper'):
                self.assertEqual(quality.archive_limit(panel, role), 128_000_000)
        self.assertEqual(capacity.base.PRIMARY_SOURCE_LIMIT, 768_000_000)
        with self.assertRaises(ValueError): capacity.score_budget([dict(samples=240001)])
        with self.assertRaises(ValueError): capacity.score_budget([dict(samples=3200)] * 121)

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
