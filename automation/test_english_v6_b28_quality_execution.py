"""Invented stdlib arrays and byte fixtures; no real execution or qualification."""
from __future__ import annotations

import base64
import copy
from contextlib import ExitStack
import io
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile
import english_v6_b28_primary_eval as primary
import english_v6_b28_replication_eval as replica
import english_v6_b28_quality_readback as quality
import english_v6_b28_training_readback as training
import test_english_v6_b24q1_quality_readback as fixtures
import test_english_v6_b28_training_readback as train_fixtures

CANDIDATE = '9' * 64
ASSESSMENT = 'a' * 64
DESIGN = 'b' * 64
TRAIN_REPORT = 'c' * 64
PROOF_BINDING = 'd' * 64


def synthetic_binding(owner):
    stack = ExitStack()
    for module, name, value in ((quality, 'ASSESSMENT_SHA', ASSESSMENT),
        (quality, 'TRAINING_ASSESSMENT_SHA', ASSESSMENT), (quality, 'DESIGN_SHA', DESIGN),
        (owner, 'TRAINING_ASSESSMENT', ASSESSMENT), (owner, 'EVALUATION_ASSESSMENT', ASSESSMENT),
        (owner, 'TRAINING_DESIGN', DESIGN), (owner, 'CANDIDATE_PT', CANDIDATE)):
        stack.enter_context(patch.object(module, name, value))
    return stack


def quality_fixture(panel='primary', failed=False):
    owner = primary if panel == 'primary' else replica
    with synthetic_binding(owner), patch.object(fixtures, 'primary', owner), patch.object(fixtures, 'reader', quality):
        members, spec, admission, rows = fixtures.fixture(panel)
    contract = owner.objective_contract()
    spec.update(contract, complete_original_B27_failure_required=True,
        original_training_report_sha256=TRAIN_REPORT, external_proof_binding_sha256=PROOF_BINDING)
    policy = dict(spec['source_policy'], schema='english-v6-b28-source-policy-v1', experiment_id='B28',
                  b28_execution_profile_sha256=profile.sha(profile.PROFILE.read_bytes()))
    spec['source_policy'] = policy
    admission['source_audit'].update(schema='english-v6-b27-full-source-and-quarantine-audit-v1', source_policy=policy)
    fields = dict(complete_original_B27_failure_verified=True, original_training_report_sha256=TRAIN_REPORT,
        external_proof_binding_sha256=PROOF_BINDING, natural_objective_schema=contract['objective_schema'],
        natural_objective_returns_verified=384, derived_B24_originals=5, derived_B24_is_new_qualification_source=False)
    admission.update(fields, output_capacity=quality.capacity.score_budget(rows), primary_independently_verified=panel == 'replication')
    result, started = (quality.document(members[name]) for name in ('RESULT.json', 'STARTED.json'))
    for value in (result, started):
        value.update(fields, source_audit=admission['source_audit'], B24Q1_failed_primary_preserved=True,
            B27_failed_primary_preserved=True, all_old_stops_and_reserves_preserved=True)
    if failed:
        with zipfile.ZipFile(io.BytesIO(members['ALL_FRAME_LOGITS.npz'])) as archive:
            arrays = {name: archive.read(name) for name in archive.namelist()}
        record = result['records'][0]
        key = record['id'] + '__1.0__challenger'
        labels = [0] * 17
        array, body = fixtures.npy(labels)
        arrays[key + '.npy'] = array
        members['raw_outputs/' + key + '.npy'] = fixtures.npy(labels, True)[0]
        record['gains']['1.0']['challenger'] = dict(array=key, frames=17, sha256=quality.sha(body), **owner.ctc_events(labels))
        members['ALL_FRAME_LOGITS.npz'] = fixtures.zip_bytes(arrays)
        result.update(fixtures.decision.decide(result['records']), logits_sha256=quality.sha(members['ALL_FRAME_LOGITS.npz']),
            replication_open_allowed=False, candidate_engineering_validation_allowed=False)
        if panel == 'replication':
            result['research_replication_passed'] = False
    members.update({'RESULT.json': quality.canonical(result), 'STARTED.json': quality.canonical(started),
                    'ADMISSION.json': quality.canonical(admission)})
    return members, spec, admission, rows, owner


def copied(value):
    return (*copy.deepcopy(value[:4]), value[4])


def verify(value):
    members, spec, admission, rows, owner = value
    with synthetic_binding(owner):
        return quality.saved_arrays(fixtures.zip_bytes(members), spec, owner.PANEL, admission, rows, owner, fixtures.decision)


def change_report(value, name, field, replacement):
    copied_value = copied(value)
    report = quality.document(copied_value[0][name])
    report[field] = replacement
    copied_value[0][name] = quality.canonical(report)
    return copied_value


def pair_fixture(owner=primary):
    models = dict(B0=owner.B0_PT, B9=owner.B9_PT, control=profile.CONTROL_PT, challenger=CANDIDATE)
    result = dict(schema=owner.PAIR_SCHEMA, status=owner.PAIR_STATUS, actual_optimizer_updates=192,
        actual_new_model_forwards=1847, natural_weight=0.16, model_pt_sha256=models,
        assessment_sha256=ASSESSMENT, design_sha256=DESIGN, recipe_sha256=DESIGN,
        phase_initialization_pt_sha256=profile.INITIAL_PT, optimizer_initialization='fresh_AdamW',
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        original_B27_NO_PROMOTION_reauthenticated=True, natural_objective_returns=384, prefix_blank_reduction='amax',
        natural_objective_schema=owner.objective_contract()['objective_schema'], **owner.objective_contract())
    result.update(dict.fromkeys(('restored_optimizer_state', 'exact_optimizer_continuation', 'historical_numerical_replay_claimed',
        'same_job_matched_control', 'promotion_allowed', 'B27_metrics_used_to_choose_parameters',
        'source_execution_allowed', 'new_primary_allowed', 'replication_allowed', 'candidate_engineering_allowed', 'product_released'), False))
    result.update(dict.fromkeys(('old_B23_optimizer_updates_reexecuted', 'old_B24_optimizer_updates_reexecuted',
        'new_control_optimizer_updates', 'new_control_forwards', 'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls'), 0))
    for role, stem in owner.STEMS.items():
        result[role] = dict(files={Path(stem).name + '.' + ext: dict(bytes=20_000,
            sha256=models[role] if ext == 'pt' else 'e' * 64) for ext in ('pt', 'npz', 'onnx')})
    result['control']['output_stem'] = owner.STEMS['control']
    result['challenger'].update(stem=owner.STEMS['challenger'], parent_pt_sha256=profile.INITIAL_PT,
        historical_control_pt_sha256=profile.CONTROL_PT, optimization_forwards=1844, checkpoint_reload_forwards=2, onnx_export_forwards=1)
    return result, dict(model_pt_sha256=models, training_assessment_sha256=ASSESSMENT, training_design_sha256=DESIGN)


def pins_and_transport(names):
    json_body, zip_body = b'{"fictional":true}\n', b'PK\x03\x04fictional'
    files, locations = {}, {}
    for index, name in enumerate(sorted(names)):
        raw = json_body if name.endswith('.json') else zip_body
        files[name] = dict(bytes=len(raw), sha256=profile.sha(raw))
        locations[name] = (dict(original_api_base64=base64.b64encode(raw).decode())
                           if name.endswith('.json') else dict(artifact_id=index + 1))
    return files, locations


def bound_fixture(panel='primary'):
    # The following saved files are invented; production separately checks
    # their exact equality to the old41+18 registrations and catalog7.
    inputs = {'inputs/proof/training-origin/old-' + str(i) + '.zip' for i in range(41)}
    inputs |= {'inputs/proof/primary-quality/' + name for name in profile.QUALITY_FILES}
    inputs |= {'inputs/metadata/train360/' + name for name in profile.CATALOG_FILES}
    required, _locations = pins_and_transport(inputs)
    files, locations = pins_and_transport(inputs | {'trained/' + name for name in profile.TRAIN_FILES})
    lock = train_fixtures.lock()
    for role, pin in lock['training']['artifacts'].items():
        files['trained/TRAIN_' + role + '.zip'] = dict(bytes=pin['bytes'], sha256=pin['sha256'])
    value = profile.inert_binding(panel)
    value.update(status='BOUND_COMPLETE_ORIGINAL_TRAIN', assessment_sha256=ASSESSMENT,
        training_lock=lock, training_reader_sha256=profile.sha(Path(training.__file__).read_bytes()),
        training_report_sha256=TRAIN_REPORT, files=files, transport=locations)
    if panel == 'replication':
        original, transport = pins_and_transport(profile.QUALITY_FILES)
        value.update(primary_quality_lock=quality_lock(), primary_quality_reader_sha256=profile.sha(Path(quality.__file__).read_bytes()),
            primary_quality_report_sha256='e' * 64, primary_quality_files=original, primary_quality_transport=transport)
    return value, required


def quality_lock(panel='primary'):
    def pin(index):
        return dict(id=index, run_id=123, run_head='a' * 40, bytes=100, sha256='b' * 64)
    return dict(schema=quality.LOCK_SCHEMA, panel=panel, result=pin(1), producer=pin(2), consumed=pin(3),
        evaluator_sha256='c' * 64, inputs_sha256='d' * 64, producer_manifest_sha256='e' * 64,
        original_result_sha256='f' * 64, external_proof_binding_sha256=PROOF_BINDING)


def quality_report(panel='primary', gate='PASS'):
    lock = quality_lock(panel)
    result = dict(schema=quality.REPORT_SCHEMA, panel=panel, panel_gate=gate,
        original_result_sha256=lock['original_result_sha256'], original_lock_sha256=quality.sha(quality.canonical(lock)),
        external_proof_binding_sha256=lock['external_proof_binding_sha256'],
        natural_objective_returns_verified=384, external_training_originals=71,
        external_total_originals=71 if panel == 'primary' else 89, derived_B24_originals=5,
        derived_B24_is_new_qualification_source=False, derived_B24_origin='same-original-B27-TRAIN_receipts-members',
        primary_independently_verified=panel == 'replication',
        replication_open_allowed=gate == 'PASS' and panel == 'primary',
        candidate_engineering_validation_allowed=gate == 'PASS' and panel == 'replication',
        model_pt_sha256=dict(B0=primary.B0_PT, B9=primary.B9_PT, control=profile.CONTROL_PT, challenger=CANDIDATE),
        assessment_sha256=ASSESSMENT,
        original_B27_failure_certification=train_fixtures.fictional_failure_certification(),
        original_B27_failure_report_sha256=train_fixtures.fictional_failure_certification()['original_report_sha256'])
    result.update(dict.fromkeys(('all_original_arrays_verified', 'original_source_admission_recomputed',
        'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified',
        'complete_original_B27_failure_verified', 'natural_no_event_terms_recomputed',
        'captured_code_matches_pinned_local_reader_root', 'old_stops_and_reserves_preserved'), True))
    result.update(dict.fromkeys(('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates'), 0))
    result.update(dict.fromkeys(('promotion_allowed', 'product_released', 'automatic_next_assessment_allowed'), False))
    return result, lock


class ProspectiveProfileContracts(unittest.TestCase):
    def test_prospective_profile_keeps_original_science_fixed_in_either_registration_state(self):
        value = profile.profile(False)
        self.assertEqual(value['limits'], dict(core=128000000, source=256000000, source_metadata=32000000,
            primary_external_proof=1920000000, replication_external_proof=2560000000))
        hashes = [value['training_design_sha256'], value['assessment_sha256'], value['ancestry_sha256'],
                  *value['panel_sha256'].values()]
        self.assertTrue(all(item is None for item in hashes) or all(profile.valid_hash(item) for item in hashes))
        self.assertFalse(value['scientific_execution_authorized'])
        self.assertEqual(value['fixed_base_sha256']['english_v6_b14_decision.py'], primary.QUALITY_CODE)
        for owner in (primary, replica):
            registered = profile.document(owner.SPEC.read_bytes()) if owner.SPEC.is_file() else {'scoring_allowed': False}
            checked = owner.source_check()
            self.assertEqual(checked['status'], 'BOUND_NOT_EXECUTED' if registered.get('scoring_allowed') is True else 'INERT_SCAFFOLD')
            self.assertFalse(checked['scientific_execution_authorized'])
            self.assertFalse(checked['promotion_allowed'])
            self.assertFalse(checked['product_released'])
            self.assertEqual((checked['model_calls'], checked['source_calls'], checked['pcm_reads']), (0, 0, 0))
            self.assertIs(owner.score.__code__, primary.original.score.__code__)
            self.assertIs(owner.snapshots.__code__, primary.original.snapshots.__code__)
            self.assertIs(owner.source_audit.__code__, primary.original.source_audit.__code__)
            self.assertIs(owner.Ledger, primary.original.Ledger)
            self.assertEqual(owner.guard.__globals__['PANEL'], owner.PANEL)
            self.assertIs(owner.score.__globals__['profile'], profile)
        self.assertIs(quality._raw_saved_arrays.__code__, quality._original_reader.saved_arrays.__code__)
        self.assertFalse({'numpy', 'torch', 'onnxruntime'} & set(sys.modules))

    def test_inert_guards_reject_before_network_model_process_or_directory(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            # Keep the negative boundary explicit when the real repository
            # later contains a legitimately frozen design and original inputs.
            inert = profile.profile(False)
            inert.update(training_design_sha256=None, assessment_sha256=None, ancestry_sha256=None,
                         panel_sha256=dict(primary=None, replication=None))
            inert_path = Path(temporary) / 'inert-profile.json'
            inert_path.write_bytes(profile.canonical(inert))
            stack.enter_context(patch.object(profile, 'PROFILE', inert_path))
            stack.enter_context(patch.object(profile, 'DESIGN_SHA', None))
            target = Path(temporary) / 'never-created'
            for owner in (primary, replica):
                with patch.object(owner, 'gh') as network, patch.object(owner, 'snapshots') as models:
                    for function, args in ((owner.protocol, ()), (owner.fetch, (target, None)),
                            (owner.score, (target, target, None))):
                        with self.assertRaisesRegex(ValueError, 'INERT'):
                            function(*args)
                    network.assert_not_called(); models.assert_not_called()
                self.assertFalse(target.exists())
                with self.assertRaises(ValueError): profile.require_proof(owner.PANEL)
            with patch.object(quality.subprocess, 'run') as network, patch.object(quality.subprocess, 'check_output') as process:
                with self.assertRaises(ValueError): quality.fetch_quality_original(target, quality_lock(), Path(temporary))
                with self.assertRaises(ValueError): quality.verify_quality_original(target, quality_lock(), Path(temporary))
                network.assert_not_called(); process.assert_not_called()
                self.assertFalse(target.exists())

    def test_partial_binding_cannot_fill_one_null_lock_or_claim_authority(self):
        for panel in ('primary', 'replication'):
            value = profile.inert_binding(panel)
            self.assertEqual(profile.validate_binding(value, panel, False), value)
            with self.assertRaisesRegex(ValueError, 'INERT'): profile.validate_binding(value, panel)
            for field, wrong in (('training_report_sha256', TRAIN_REPORT), ('assessment_sha256', ASSESSMENT),
                    ('scientific_execution_authorized', True), ('original_B27_failure_required', False)):
                changed = dict(value, **{field: wrong})
                with self.assertRaises(ValueError): profile.validate_binding(changed, panel, False)

    def test_complete71_and_replica89_layout_not_old41_or_summary(self):
        for panel in ('primary', 'replication'):
            value, required = bound_fixture(panel)
            with patch.object(profile, 'profile', return_value=dict(assessment_sha256=ASSESSMENT)), \
                 patch.object(profile, 'expected_training_inputs', return_value=required):
                self.assertEqual(profile.validate_binding(value, panel), value)
                for name in ('trained/TRAIN_models.zip', 'inputs/metadata/train360/api/tree.json', next(iter(required))):
                    changed = copy.deepcopy(value); del changed['files'][name]
                    with self.assertRaises(ValueError): profile.validate_binding(changed, panel)
                changed = copy.deepcopy(value); changed['files']['summary.json'] = dict(bytes=2, sha256='0' * 64)
                with self.assertRaises(ValueError): profile.validate_binding(changed, panel)
                changed = copy.deepcopy(value); name = next(iter(required)); changed['files'][name]['sha256'] = '0' * 64
                with self.assertRaises(ValueError): profile.validate_binding(changed, panel)
                if panel == 'replication':
                    changed = copy.deepcopy(value); del changed['primary_quality_files']['inputs/source.zip']
                    with self.assertRaises(ValueError): profile.validate_binding(changed, panel)

    def test_individual_and_total_prospective_caps_are_closed(self):
        for name, limit in (('inputs/source.zip', 256000000), ('inputs/pair.zip', 128000000), ('api/run.json', 32000000)):
            profile.proof_file_pins({name: dict(bytes=limit, sha256='1' * 64)}, 1920000000)
            with self.assertRaises(ValueError):
                profile.proof_file_pins({name: dict(bytes=limit + 1, sha256='1' * 64)}, 1920000000)
        files = {'old/' + str(i) + '.zip': dict(bytes=128000000, sha256='1' * 64) for i in range(16)}
        with self.assertRaises(ValueError): profile.proof_file_pins(files, 1920000000)
        ceiling = quality.capacity.score_budget([dict(samples=240000)] * 120)
        self.assertEqual(ceiling['arrays'], 1440)
        self.assertLessEqual(ceiling['maximum_zip_bytes'], 128000000)
        for rowset in ([dict(samples=240001)], [dict(samples=3200)] * 121):
            with self.assertRaises(ValueError): quality.capacity.score_budget(rowset)

    def test_catalog_seven_have_only_their_original_metadata_transport_capacity(self):
        name = 'inputs/metadata/train360/originals/metadata.zip'
        profile.proof_file_pins({name: dict(bytes=201326592, sha256='1' * 64)}, 1920000000)
        for member, size in ((name, 201326593), ('inputs/metadata/train360/api/run.json', 8000001),
                ('inputs/metadata/train360/originals/unknown.zip', 1), ('inputs/pair.zip', 128000001)):
            with self.assertRaises(ValueError):
                profile.proof_file_pins({member: dict(bytes=size, sha256='1' * 64)}, 1920000000)
        files = {name: dict(bytes=201326592, sha256='1' * 64),
            'inputs/metadata/train360/originals/producer.zip': dict(bytes=33554433, sha256='2' * 64)}
        with self.assertRaises(ValueError): profile.proof_file_pins(files, 1920000000)
        self.assertEqual(profile.original_file_limit('inputs/source.zip'), 256000000)
        self.assertEqual(profile.original_file_limit('inputs/pair.zip'), 128000000)

    def test_exact_historical_q1_source_keeps_its_original_bound_without_expanding_new_source(self):
        name = 'inputs/proof/training-origin/qualification/no-promotion/inputs/source.zip'
        self.assertEqual(profile.original_file_limit(name), 768000000)
        profile.proof_file_pins({name: dict(bytes=367799776, sha256='1' * 64)}, 1920000000)
        with self.assertRaises(ValueError):
            profile.proof_file_pins({name: dict(bytes=768000001, sha256='1' * 64)}, 1920000000)
        for new_name in ('inputs/source.zip', 'inputs/proof/primary-quality/inputs/source.zip'):
            self.assertEqual(profile.original_file_limit(new_name), 256000000)
            with self.assertRaises(ValueError):
                profile.proof_file_pins({new_name: dict(bytes=367799776, sha256='1' * 64)}, 1920000000)

        # Preserve the actual fixed historical metadata row, without reading
        # any source samples. The larger role cannot admit a different package.
        old = profile._old_profile.binding('primary')
        old_name = 'qualification/no-promotion/inputs/source.zip'
        immutable = dict(old['files'][old_name])
        bound, required = bound_fixture()
        replaced = 'inputs/proof/training-origin/old-0.zip'
        for mapping in (bound['files'], required):
            del mapping[replaced]
            mapping[name] = dict(immutable)
        del bound['transport'][replaced]
        bound['transport'][name] = dict(old['transport'][old_name])
        with patch.object(profile, 'profile', return_value=dict(assessment_sha256=ASSESSMENT,
                fixed_base_sha256=profile.profile(False)['fixed_base_sha256'])), \
             patch.object(profile, 'expected_training_inputs', return_value=required):
            self.assertEqual(profile.validate_binding(bound, 'primary'), bound)
            for field, wrong in (('sha256', '0' * 64), ('bytes', immutable['bytes'] + 1)):
                changed = copy.deepcopy(bound)
                changed['files'][name][field] = wrong
                with self.assertRaises(ValueError): profile.validate_binding(changed, 'primary')
            changed = copy.deepcopy(bound)
            changed['files'][name.replace('/no-promotion/', '/other/')]=changed['files'].pop(name)
            with self.assertRaises(ValueError): profile.validate_binding(changed, 'primary')

    def test_original_api_transport_cannot_follow_latest_or_change_raw_bytes(self):
        files, transport = pins_and_transport({'api/run.json', 'inputs/source.zip'})
        self.assertEqual(profile.validate_transport(transport, files), transport)
        for name, wrong in (('inputs/source.zip', dict(latest=True)),
                ('api/run.json', dict(original_api_base64=base64.b64encode(b'{}').decode()))):
            changed = copy.deepcopy(transport); changed[name] = wrong
            with self.assertRaises(ValueError): profile.validate_transport(changed, files)

    def test_proof_scope_rejects_extras_mutation_symlink_and_wrong_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); origin = root / 'training-origin'; origin.mkdir()
            path = origin / 'original.zip'; path.write_bytes(b'fictional')
            value = dict(files=profile.inventory(origin))
            with patch.object(profile, 'binding', return_value=value):
                with profile.proof_scope(root):
                    self.assertEqual(profile.proof_root('primary'), root)
                    with self.assertRaises(ValueError): profile.proof_root('replication')
                with self.assertRaises(ValueError): profile.proof_root()
                (root / 'unknown').mkdir()
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root): pass
                (root / 'unknown').rmdir()
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root): pass
                path.write_bytes(b'fictional')
                (origin / 'link.zip').symlink_to(path)
                with self.assertRaises(ValueError):
                    with profile.proof_scope(root): pass

    def test_private_full_byte_cache_rehashes_and_destroys_results(self):
        old = profile.preserved('english_v6_b24q1_evidence_scope')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); originals = root / 'proof'; originals.mkdir()
            body = originals / 'entire.zip'; body.write_bytes(b'fictional original bytes')
            reader = root / 'reader.py'; reader.write_bytes(b'fictional fixed reader')
            files = profile.inventory(originals)
            raw = profile.canonical(dict(complete=True, model_calls=0)); calls = []
            def execute(): calls.append(True); return raw
            def verify_once():
                return profile.evidence.verify_once('b28_complete_training', originals, files, b'{}', reader,
                    profile.sha(reader.read_bytes()), profile.sha(raw), execute)
            with profile.evidence.scope():
                self.assertIsNone(old._ACTIVE.get())
                self.assertEqual(verify_once(), raw); self.assertEqual(verify_once(), raw)
                self.assertEqual(len(calls), 1)
                body.write_bytes(b'altered original bytes')
                with self.assertRaises(ValueError): verify_once()
                self.assertEqual(len(calls), 1)
            self.assertIsNone(profile.evidence._ACTIVE.get())

    def test_temporary_b24_five_are_exact_derivatives_of_same_original66(self):
        train, _producer = training.modules()
        old_files = {name: (b'{"fictional_original":true}\n' if name.endswith('.json') else
            b'PK\x03\x04fictional complete original ' + name.encode()) for name in profile.TRAIN_FILES}
        receipt = fixtures.zip_bytes({'original_b24/' + name: raw for name, raw in old_files.items()})
        receipt_pin = dict(bytes=len(receipt), sha256=profile.sha(receipt))
        inputs = {'proof/training-origin/trained/TRAIN_receipts.zip': receipt}
        inputs.update({'proof/training-origin/old-' + str(i) + '.zip': b'fictional old original ' + str(i).encode() for i in range(40)})
        inputs.update({'proof/primary-quality/' + name: b'fictional original ' + name.encode() for name in profile.QUALITY_FILES})
        inputs.update({'metadata/train360/' + name: b'fictional catalog original ' + name.encode() for name in profile.CATALOG_FILES})
        self.assertEqual(len(inputs), 66)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); original = root / 'training-origin'; trained = original / 'trained'
            trained.mkdir(parents=True)
            for name, raw in inputs.items():
                path = original / 'inputs' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            original_pins = profile.inventory(original)
            bound = dict(files=original_pins, training_lock={'fictional': True})
            seen = []
            def verify(directory, lock, staged):
                seen.append(staged)
                self.assertEqual(directory, trained)
                self.assertEqual(lock, bound['training_lock'])
                self.assertNotEqual(staged, original / 'inputs')
                self.assertEqual(len(profile.inventory(staged)), 71)
                for name, raw in old_files.items(): self.assertEqual((staged / 'original_b24' / name).read_bytes(), raw)
                self.assertEqual(profile.training_child()['files'], profile.inventory(staged))
                return dict(fictional_staging_test=True, model_calls=0)
            fake_reader = SimpleNamespace(verify_training_original=verify)
            with patch.object(train.profile, 'binding', return_value=dict(files={'trained/TRAIN_receipts.zip': receipt_pin})):
                returned = profile._training_readback(root, bound, fake_reader)
                self.assertEqual(profile.document(returned), dict(fictional_staging_test=True, model_calls=0))
                self.assertEqual(profile.inventory(original), original_pins)
                self.assertFalse((original / 'inputs/original_b24').exists())
                self.assertTrue(seen and all(not path.exists() for path in seen))
                with self.assertRaises(ValueError): profile.training_child()
                def altered(directory, lock, staged):
                    result = verify(directory, lock, staged)
                    (staged / 'original_b24/TRAIN_models.zip').write_bytes(b'changed derivative')
                    return result
                fake_reader.verify_training_original = altered
                with self.assertRaisesRegex(ValueError, 'unaltered original receipt member'):
                    profile._training_readback(root, bound, fake_reader)
                self.assertEqual(profile.inventory(original), original_pins)

    def test_require_proof_opens_replica_only_after_complete_new_primary_pass(self):
        bound, _required = bound_fixture('replication')
        bound['training_lock']['training']['model_pt_sha256']['challenger'] = CANDIDATE
        train_result = dict(schema='english-v6-b28-train-natural-event-v1',
            status='TRAINED_NATURAL_EVENT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY',
            assessment_sha256=ASSESSMENT, model_pt_sha256=bound['training_lock']['training']['model_pt_sha256'])
        report = dict(schema=training.REPORT_SCHEMA, result=train_result,
            original_lock_sha256=profile.sha(profile.canonical(bound['training_lock'])),
            result_sha256=bound['training_lock']['training']['result_sha256'],
            saved_training_observations=dict(natural_objective_returns_verified=384))
        report.update(dict.fromkeys(('all_original_producer_git_verified', 'all_original_inputs_verified',
            'all_saved_training_arrays_verified', 'natural_no_event_terms_recomputed',
            'complete_original_B27_NO_PROMOTION_verified', 'original_B24_control_unchanged'), True))
        report.update(dict.fromkeys(('new_model_calls', 'new_source_calls', 'new_optimizer_updates'), 0))
        report.update(dict.fromkeys(('promotion_allowed', 'product_released', 'gradients_recomputed', 'optimizer_updates_replayed'), False))
        prior, _lock = quality_report('primary')
        with tempfile.TemporaryDirectory() as temporary, patch.object(profile, 'binding', return_value=bound), \
             patch.object(profile, 'profile', return_value=dict(assessment_sha256=ASSESSMENT)), \
             patch.object(quality, 'failure_certification', side_effect=train_fixtures.fictional_failure_certification):
            token = profile._PROOF.set((Path(temporary).resolve(), 'replication', bound))
            try:
                def original_read(kind, *args):
                    return profile.canonical(report if kind == 'b28_complete_training' else prior)
                with patch.object(profile.evidence, 'verify_once', side_effect=original_read) as original:
                    actual = profile.require_proof('replication')
                    self.assertEqual(actual['primary_quality']['panel_gate'], 'PASS')
                    self.assertEqual([call.args[0] for call in original.call_args_list], ['b28_complete_training', 'b28_primary_quality'])
                    failed, _lock = quality_report('primary', 'NO_PROMOTION')
                    prior.clear(); prior.update(failed)
                    with self.assertRaisesRegex(ValueError, 'only complete independently'): profile.require_proof('replication')
                    passed, _lock = quality_report('primary')
                    prior.clear(); prior.update(passed)
                    prior['model_pt_sha256']['challenger'] = '8' * 64
                    with self.assertRaisesRegex(ValueError, 'only complete independently'): profile.require_proof('replication')
                    prior.clear(); prior.update(quality_report('primary')[0])
                    report['complete_original_B27_NO_PROMOTION_verified'] = False
                    with self.assertRaises(ValueError): profile.require_proof('replication')
            finally:
                profile._PROOF.reset(token)


class EvaluationContracts(unittest.TestCase):
    def test_source_resource_preflight_requires_complete_captured_files_and_exact_zero_call_semantics(self):
        policy, scientific = dict(fictional_policy=True), dict(fictional_scientific_contract=True)
        metadata = {'automation/frozen.json': 'a' * 64}
        collector = {'automation/collector.py': 'b' * 64}
        paths = set(metadata) | set(collector) | {'automation/extra.py'}
        receipt = dict(schema='english-v6-b28-captured-source-resource-preflight-v1', status='PASS_FROZEN_REGISTRATION_ONLY',
            panel='primary', metadata_hashes=metadata, collector_sha256='c' * 64, source_policy=policy,
            scientific_contract_sha256=profile.sha(profile.canonical(scientific)), opened_resources=sorted(paths),
            complete_external_originals_still_required=True, new_model_calls=0, new_source_calls=0,
            new_network_calls=0, new_audio_reads=0, optimizer_updates=0)
        def verify(value):
            return primary.source_resource_contract(value, policy, scientific, metadata, 'c' * 64, collector, paths)
        self.assertEqual(verify(receipt), receipt)
        for key, wrong in (('schema', 'old-source'), ('status', 'FULLY_QUALIFIED'), ('panel', 'replication'),
            ('collector_sha256', '0' * 64), ('source_policy', {}), ('scientific_contract_sha256', '0' * 64),
            ('opened_resources', list(metadata)), ('opened_resources', sorted(paths | {'unregistered.py'})),
            ('new_network_calls', 1), ('new_audio_reads', 1), ('complete_external_originals_still_required', False)):
            with self.assertRaises(ValueError, msg=key): verify(dict(receipt, **{key: wrong}))

    def test_pair_requires_new384_objective_and_all_old_terminal_evidence(self):
        result, spec = pair_fixture()
        with synthetic_binding(primary):
            primary.validate_pair_receipt(result, spec)
            for field, wrong in (('schema', 'english-v6-b27-train-natural-weight-v1'), ('natural_objective_returns', 383),
                ('natural_objective_schema', 'empty-ctc'), ('reference_sha256', '0' * 64), ('prefix_blank_reduction', 'mean'),
                ('natural_weight', 0.08), ('actual_optimizer_updates', 191), ('actual_new_model_forwards', 1848),
                ('original_B27_NO_PROMOTION_reauthenticated', False), ('B27_metrics_used_to_choose_parameters', True),
                ('new_primary_allowed', True), ('phase_initialization_pt_sha256', profile.CONTROL_PT)):
                changed = copy.deepcopy(result); changed[field] = wrong
                with self.assertRaises(ValueError, msg=field): primary.validate_pair_receipt(changed, spec)
            for bad in (profile.CONTROL_PT, profile.INITIAL_PT, profile.B27_PT):
                changed = copy.deepcopy(spec); changed['model_pt_sha256']['challenger'] = bad
                with self.assertRaises(ValueError): primary.validate_pair_receipt(result, changed)

    def test_once_workflow_identity_rejects_old_branch_retry_force_or_wrong_parent(self):
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

    def test_scored_write_can_only_add_ancestry_after_complete_real_readback(self):
        proof = dict(complete_original_B27_NO_PROMOTION_verified=True, natural_no_event_terms_recomputed=True)
        with tempfile.TemporaryDirectory() as temporary, patch.object(profile, 'require_proof', return_value=proof), \
             patch.object(profile, 'binding', return_value=dict(training_report_sha256=TRAIN_REPORT)), \
             patch.object(primary, 'spec_binding_sha', return_value=PROOF_BINDING):
            path = Path(temporary) / 'STARTED.json'
            primary.write(path, dict(schema=primary.SCHEMA))
            actual = profile.document(path.read_bytes())
            self.assertTrue(actual['B27_failed_primary_preserved'])
            self.assertEqual(actual['natural_objective_returns_verified'], 384)
            self.assertFalse(actual['derived_B24_is_new_qualification_source'])
            proof['natural_no_event_terms_recomputed'] = False
            with self.assertRaises(ValueError): primary.write(Path(temporary) / 'RESULT.json', dict(schema=primary.SCHEMA))
            self.assertFalse((Path(temporary) / 'RESULT.json').exists())


class CompleteRawArrayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = {(panel, failed): quality_fixture(panel, failed)
            for panel in ('primary', 'replication') for failed in (False, True)}

    def test_exact_original_b14_recomputes_complete_pass_or_no_promotion_for_both(self):
        for (panel, failed), value in self.values.items():
            report = verify(value)['result']
            self.assertEqual(report['panel_gate'], 'NO_PROMOTION' if failed else 'PASS')
            self.assertEqual(report['actual_forwards'], 576)
            self.assertEqual(report['replication_open_allowed'], not failed and panel == 'primary')
            self.assertEqual(report['candidate_engineering_validation_allowed'], not failed and panel == 'replication')

    def test_complete_failure_cannot_open_next_stage_or_erase_any_terminal(self):
        for panel in ('primary', 'replication'):
            for field, wrong in (('replication_open_allowed', True), ('candidate_engineering_validation_allowed', True),
                ('promotion_allowed', True), ('B27_failed_primary_preserved', False), ('B24Q1_failed_primary_preserved', False),
                ('all_old_stops_and_reserves_preserved', False), ('B24R1_original_source_stop_preserved', False)):
                with self.assertRaises(ValueError, msg=(panel, field)):
                    verify(change_report(self.values[panel, True], 'RESULT.json', field, wrong))

    def test_partial_unknown_stop_missing_journal_and_extra_arrays_stay_closed(self):
        for field, wrong in (('panel_gate', 'STOP'), ('panel_gate', 'UNKNOWN'), ('panel_gate', 'NO_PROMOTION'),
            ('consumed_unknown', ['one']), ('actual_forwards', 575), ('raw_output_count', 575), ('error_type', 'HiddenFailure')):
            with self.assertRaises(ValueError, msg=field):
                verify(change_report(self.values['primary', False], 'RESULT.json', field, wrong))
        for kind in ('journal', 'raw', 'extra'):
            value = copied(self.values['primary', False])
            if kind == 'journal': del value[0]['attempts/fixture-0__1.0__B0.json']
            if kind == 'raw': del value[0]['raw_outputs/fixture-0__1.0__B0.npy']
            if kind == 'extra': value[0]['raw_outputs/unregistered.npy'] = fixtures.npy([0] * 17, True)[0]
            with self.assertRaises(ValueError, msg=kind): verify(value)

    def test_wrong_raw_bytes_clock_nan_gain_or_forged_summary_rejected(self):
        for kind in ('raw', 'nan', 'clock', 'gain', 'summary'):
            value = copied(self.values['primary', False]); members = value[0]
            result = quality.document(members['RESULT.json'])
            if kind == 'raw': members['raw_outputs/fixture-0__1.0__B0.npy'] = fixtures.npy([0] * 17, True)[0]
            if kind == 'nan':
                name = 'raw_outputs/fixture-0__1.0__B0.npy'
                members[name] = members[name][:-4] + struct.pack('<f', float('nan'))
            if kind == 'clock': result['records'][0]['gains']['1.0']['B0']['event_samples'] = [999]
            if kind == 'gain': result['records'][0]['gains']['0.8'] = result['records'][0]['gains'].pop('0.75')
            if kind == 'summary': result['restored_positives'] = []
            members['RESULT.json'] = quality.canonical(result)
            with self.assertRaises(ValueError, msg=kind): verify(value)

    def test_source_gate_full120_quarantine_and_full_budget_cannot_shrink(self):
        for field, wrong in (('original_source_rows', 119), ('qualified_rows', 47), ('observer_eligible_rows', 117),
                            ('quarantined_kws_forwards', 1)):
            value = copied(self.values['primary', False]); value[2]['source_audit'][field] = wrong
            with self.assertRaises(ValueError, msg=field): verify(value)
        value = copied(self.values['primary', False]); value[2]['source_audit']['quarantined_ids'].append('fixture-0')
        with self.assertRaises(ValueError): verify(value)
        value = copied(self.values['primary', False]); value[2]['output_capacity']['limit_bytes'] += 1
        with self.assertRaises(ValueError): verify(value)

    def test_new_train_objective_and_primary_certification_cannot_be_forged(self):
        for panel in ('primary', 'replication'):
            for field, wrong in (('natural_objective_returns_verified', 383), ('complete_original_B27_failure_verified', False),
                ('derived_B24_is_new_qualification_source', True), ('original_training_report_sha256', '0' * 64)):
                with self.assertRaises(ValueError, msg=(panel, field)):
                    verify(change_report(self.values[panel, False], 'STARTED.json', field, wrong))
        value = copied(self.values['replication', False]); value[2]['primary_independently_verified'] = False
        with self.assertRaises(ValueError): verify(value)
        for field, wrong in (('reference_sha256', '0' * 64), ('assessment_sha256', primary.original.EVALUATION_ASSESSMENT),
                            ('maximum_model_forwards', 1441)):
            value = copied(self.values['primary', False]); value[1][field] = wrong
            with self.assertRaises(ValueError, msg=field): verify(value)


class ReaderIsolationContracts(unittest.TestCase):
    def setUp(self):
        bound = patch.object(quality, 'failure_certification', side_effect=train_fixtures.fictional_failure_certification)
        bound.start()
        self.addCleanup(bound.stop)

    def test_capture_must_match_one_local_root_before_any_captured_import(self):
        names = {'automation/english_v6_b28_quality_readback.py', 'automation/english_v6_b28_profile.py',
            'automation/english_v6_b14_decision.py', 'automation/frozen.json', '.github/workflows/one.yml'}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve(); local = directory / 'local'; captured = directory / 'captured'
            contents = {name: ('fictional ' + name).encode() for name in names}
            for base in (local, captured):
                for name, raw in contents.items():
                    path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            expected = {name: profile.sha(raw) for name, raw in contents.items()}
            owner = SimpleNamespace(ROOT=local, PANEL='primary', WORKFLOW='.github/workflows/one.yml',
                producer_files=lambda *args: expected)
            workflow = contents[owner.WORKFLOW]
            self.assertEqual(quality.authenticate_local_capture(owner, captured, {}, {}, workflow), expected)
            target = local / 'automation/frozen.json'; target.write_bytes(b'changed current-root resource')
            with self.assertRaisesRegex(ValueError, 'one unchanged local'): quality.authenticate_local_capture(owner, captured, {}, {}, workflow)
            target.write_bytes(contents['automation/frozen.json'])
            target = captured / 'automation/english_v6_b28_profile.py'; target.write_bytes(b'changed captured code')
            with self.assertRaisesRegex(ValueError, 'captured member'): quality.authenticate_local_capture(owner, captured, {}, {}, workflow)

    def test_offline_guard_only_exact_staged_complete_training_child_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); originals = root / 'training-origin'; trained = originals / 'trained'
            trained.mkdir(parents=True); (trained / 'TRAIN_LIVE.json').write_bytes(b'{}')
            staged = root / 'temporary-inputs'; staged.mkdir(); (staged / 'saved.json').write_bytes(b'{}')
            lock_path = root / 'lock.json'; lock_path.write_bytes(b'{}')
            reader = Path(training.__file__).resolve()
            bound = dict(training_reader_sha256=profile.sha(reader.read_bytes()), training_lock={}, files=profile.inventory(originals))
            child = dict(directory=trained, inputs=staged, files=profile.inventory(staged))
            command = [sys.executable, '-I', '-S', '-B', str(reader), '--verify-worker', str(trained), str(lock_path), str(staged)]
            with patch.object(profile, 'binding', return_value=bound), patch.object(profile, 'training_child', return_value=child), \
                 patch.object(subprocess, 'Popen') as launch:
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
                (staged / 'saved.json').write_bytes(b'changed derived bytes')
                with quality.offline_only(root, 'primary'), self.assertRaises(ValueError):
                    subprocess.Popen(command, stdout=subprocess.PIPE)
                self.assertEqual(launch.call_count, 1)

    def test_replication_allows_only_its_exact_complete_original_primary_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'primary-quality'; directory.mkdir()
            for name in profile.QUALITY_FILES:
                path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fictional')
            lock = quality_lock(); lock_path = root / 'lock.json'; lock_path.write_bytes(profile.canonical(lock))
            reader = Path(quality.__file__).resolve()
            bound = dict(primary_quality_reader_sha256=profile.sha(reader.read_bytes()), primary_quality_lock=lock,
                primary_quality_files=profile.inventory(directory))
            command = [sys.executable, '-I', '-S', '-B', str(reader), '--verify-worker', str(directory), str(lock_path), str(root)]
            with patch.object(profile, 'binding', return_value=bound), patch.object(subprocess, 'Popen') as launch:
                with quality.offline_only(root, 'replication'):
                    subprocess.Popen(command, stdout=subprocess.PIPE)
                    with self.assertRaises(ValueError): subprocess.Popen(command, stdout=subprocess.PIPE)
                with quality.offline_only(root, 'primary'), self.assertRaises(ValueError): subprocess.Popen(command)
                lock_path.write_bytes(profile.canonical(lock) + b'\n')
                with quality.offline_only(root, 'replication'), self.assertRaises(ValueError): subprocess.Popen(command)
                self.assertEqual(launch.call_count, 1)

    def test_reports_require_both_passes_before_engineering_and_never_release(self):
        for panel in ('primary', 'replication'):
            for gate in ('PASS', 'NO_PROMOTION'):
                report, lock = quality_report(panel, gate)
                self.assertEqual(quality.validate_report(report, lock), report)
                for field, wrong in (('natural_no_event_terms_recomputed', False), ('external_training_originals', 41),
                    ('complete_original_B27_failure_verified', False), ('captured_code_matches_pinned_local_reader_root', False),
                    ('derived_B24_is_new_qualification_source', True), ('new_optimizer_updates', 1), ('product_released', True),
                    ('original_B27_failure_certification', dict(untrusted_summary=True)),
                    ('original_B27_failure_report_sha256', '0' * 64)):
                    with self.assertRaises(ValueError, msg=(panel, gate, field)):
                        quality.validate_report(dict(report, **{field: wrong}), lock)
            report, lock = quality_report(panel, 'NO_PROMOTION')
            for field in ('replication_open_allowed', 'candidate_engineering_validation_allowed'):
                with self.assertRaises(ValueError): quality.validate_report(dict(report, **{field: True}), lock)
        report, lock = quality_report('replication')
        with self.assertRaises(ValueError): quality.validate_report(dict(report, primary_independently_verified=False), lock)


if __name__ == '__main__':
    unittest.main()
