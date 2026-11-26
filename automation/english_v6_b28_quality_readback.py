"""Independent complete-original B28 quality readback, without scientific calls.

The frozen B27 raw-array reader recomputes every return and unchanged B14 gate.
All new original TRAIN71 evidence, including384 independently rebuilt natural
losses, precedes source and quality admission. No captured module is imported
from a second root: its entire Git-bound closure must equal the pinned local
verification closure before those local implementations examine original data.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
import importlib
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile

_original_reader = profile.preserved('english_v6_b27_quality_readback')
profile.inherit(_original_reader, globals())
capacity = profile.preserved('english_v6_b27_capacity')
EVALUATOR_STEMS = {panel: 'english_v6_b28_' + panel for panel in ('primary', 'replication')}
SCHEMAS = {panel: 'english-v6-b28-fresh-' + panel + '-four-frozen-models-v1' for panel in EVALUATOR_STEMS}
ASSESSMENT_SHA = TRAINING_ASSESSMENT_SHA = profile.profile(False)['assessment_sha256']
DESIGN_SHA = profile.DESIGN_SHA
LOCK_SCHEMA = 'english-v6-b28-original-quality-lock-v1'
REPORT_SCHEMA = 'english-v6-b28-verified-original-quality-v1'
LOCK_FIELDS = _original_reader.LOCK_FIELDS
_raw_saved_arrays = saved_arrays
_original_source_admission = source_admission


def validate_lock(lock):
    # The same closed preparation boundary applies to read-only acquisition and
    # subprocess verification; a fabricated future lock cannot open either.
    profile.profile()
    need(type(lock) is dict and set(lock) == LOCK_FIELDS and lock['schema'] == LOCK_SCHEMA
         and lock['panel'] in SCHEMAS, 'exact complete original B28 quality lock')
    panel = lock['panel']
    bound = profile.binding(panel)
    for role in ('result', 'producer', 'consumed'):
        valid_artifact(lock[role], panel=panel, role=role)
    need(len({lock[key]['id'] for key in ('result', 'producer', 'consumed')}) == 3
         and all(lock[key][field] == lock['result'][field]
                 for key in ('producer', 'consumed') for field in ('run_id', 'run_head')),
         'three distinct entire originals from the same first B28 quality run')
    for field in ('evaluator_sha256', 'inputs_sha256', 'producer_manifest_sha256',
                  'original_result_sha256', 'external_proof_binding_sha256'):
        need(valid_hash(lock[field]), 'exact immutable original digest ' + field)
    stem = EVALUATOR_STEMS[panel]
    evaluator = profile.HERE / (stem + '_eval.py')
    need(sha(evaluator.read_bytes()) == lock['evaluator_sha256']
         and sha(profile.proof_binding_path(panel).read_bytes()) == lock['external_proof_binding_sha256']
         and sha((profile.HERE / (stem + '_inputs.json')).read_bytes()) == lock['inputs_sha256']
         and sha((profile.HERE / (stem + '_producer.json')).read_bytes()) == lock['producer_manifest_sha256'],
         'entire original evaluator, input/proof registrations and full producer manifest')
    owner = importlib.import_module(stem + '_eval')
    need(Path(owner.__file__).resolve() == evaluator, 'one pinned local evaluator root')
    spec = owner.protocol()
    need(spec['original_training_report_sha256'] == bound['training_report_sha256'], 'same fully certified original TRAIN')
    # protocol authenticates all twelve prospective input identities and their
    # unchanged per-role caps before any original output is fetched.
    return lock


def artifact_names(panel, run):
    need(panel in SCHEMAS, 'explicit original B28 panel')
    return dict(result='english-v6-b28-historical-control-' + panel + '-logits-' + str(run),
                producer='english-v6-b28-' + panel + '-producer-evidence-' + str(run),
                consumed='english-v6-b28-' + panel + '-consumed')


def failure_certification():
    return importlib.import_module('english_v6_b28_train').failure_certification()


def training_context(proof, bound):
    need(proof['complete_original_B27_NO_PROMOTION_verified'] is True
         and proof['natural_no_event_terms_recomputed'] is True
         and proof['saved_training_observations']['natural_objective_returns_verified'] == 384,
         'full original failure and384 recomputed no-event observations')
    return dict(complete_original_B27_failure_verified=True,
        original_training_report_sha256=bound['training_report_sha256'],
        natural_objective_schema=proof['result']['natural_objective_schema'],
        natural_objective_returns_verified=384, derived_B24_originals=5,
        derived_B24_is_new_qualification_source=False)


def source_admission(primary, inputs, spec, identity, workflow, fetched, panel, primary_quality=None):
    proof = profile.require_proof(panel)
    if panel == 'replication':
        need(primary_quality is not None and primary_quality == proof['primary_quality']
             and primary_quality['panel_gate'] == 'PASS'
             and primary_quality['model_pt_sha256'] == spec['model_pt_sha256'],
             'source and scoring require the same fully recomputed original PRIMARY PASS')
    value = _original_source_admission(primary, inputs, spec, identity, workflow, fetched, panel, primary_quality)
    value['admission'].update(training_context(proof, profile.binding(panel)),
        primary_independently_verified=panel == 'replication')
    return value


def saved_arrays(raw_zip, spec, panel, admission, selected_rows, primary, decision):
    """Add B28 ancestry checks around the unchanged complete raw/B14 algorithm."""
    contract = primary.objective_contract()
    fields = dict(complete_original_B27_failure_verified=True,
        original_training_report_sha256=spec['original_training_report_sha256'],
        external_proof_binding_sha256=spec['external_proof_binding_sha256'],
        natural_objective_schema=contract['objective_schema'], natural_objective_returns_verified=384,
        derived_B24_originals=5, derived_B24_is_new_qualification_source=False)
    need(valid_hash(fields['original_training_report_sha256']) and valid_hash(fields['external_proof_binding_sha256'])
         and all(spec.get(key) == value for key, value in contract.items())
         and spec['complete_original_B27_failure_required'] is True
         and all(admission.get(key) == value for key, value in fields.items())
         and admission['primary_independently_verified'] is (panel == 'replication')
         and primary.CANDIDATE_PT not in (profile.CONTROL_PT, profile.INITIAL_PT, profile.B27_PT),
         'new B28 objective, original terminal and full TRAIN identities before raw quality admission')
    with opened(raw_zip, archive_limit(panel, 'result')) as archive:
        for name in ('STARTED.json', 'RESULT.json'):
            report = document(archive.read(name))
            need(all(report.get(key) == value for key, value in fields.items())
                 and report.get('B27_failed_primary_preserved') is True,
                 'unchanged complete B27 terminal and exact TRAIN evidence in every original scoring record')
    return _raw_saved_arrays(raw_zip, spec, panel, admission, selected_rows, primary, decision)


def offline_only(proof_root, panel):
    """Allow only exact scoped TRAIN/PRIMARY verifier children; block all science."""
    root = Path(proof_root)
    need(root.is_absolute() and root == root.resolve() and root.is_dir() and not root.is_symlink(),
         'regular explicit original proof root')
    bound = profile.binding(panel)
    stack = ExitStack()
    old_import, old_popen, old_socket = builtins.__import__, subprocess.Popen, socket.socket
    forbidden = {'numpy', 'torch', 'torchaudio', 'onnxruntime', 'soundfile', 'transformers', 'piper', 'requests', 'httpx'}
    used = set()
    def import_guard(name, *args, **kwargs):
        need(name.split('.')[0] not in forbidden, 'no scientific or network runtime during independent B28 readback')
        return old_import(name, *args, **kwargs)
    def popen_guard(command, *args, **kwargs):
        need(type(command) is list and len(command) == 9
             and command[:4] == [sys.executable, '-I', '-S', '-B'] and command[5] == '--verify-worker'
             and not args and set(kwargs) <= {'stdin', 'stdout', 'stderr'}, 'only exact isolated complete-proof children')
        path, directory, lock_path, inputs = map(Path, (command[4], command[6], command[7], command[8]))
        for entry in (path, directory, lock_path, inputs):
            need(entry.is_absolute() and entry == entry.resolve() and not entry.is_symlink(), 'regular resolved verifier paths')
        need(path.is_file() and lock_path.is_file() and directory.is_dir() and inputs.is_dir(), 'complete original child files')
        raw = lock_path.read_bytes(); lock = document(raw)
        need(raw == canonical(lock), 'exact canonical original verifier lock')
        if path == profile.HERE / 'english_v6_b28_training_readback.py':
            kind = 'training'
            staged = profile.training_child()
            need(sha(path.read_bytes()) == bound['training_reader_sha256']
                 and lock == bound['training_lock']
                 and directory == staged['directory'] == root / 'training-origin/trained'
                 and inputs == staged['inputs'] and profile.inventory(inputs) == staged['files']
                 and profile.inventory(root / 'training-origin') == bound['files'],
                 'one exact TRAIN reader with all71 originals and byte-identical temporary derived B24 members')
        else:
            kind = 'primary'
            need(panel == 'replication' and path == Path(__file__).resolve()
                 and sha(path.read_bytes()) == bound['primary_quality_reader_sha256']
                 and lock == bound['primary_quality_lock'] and lock['panel'] == 'primary'
                 and directory == root / 'primary-quality' and inputs == root
                 and profile.inventory(directory) == bound['primary_quality_files'],
                 'only the complete explicitly pinned same-B28 original PRIMARY child')
        need(kind not in used, 'one complete proof reader per audit; no arbitrary additional process')
        used.add(kind)
        return old_popen(command, **kwargs)
    def blocked(*args, **kwargs):
        raise ValueError('no network during B28 original quality readback')
    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs): blocked()
    stack.enter_context(mock.patch('builtins.__import__', side_effect=import_guard))
    stack.enter_context(mock.patch('subprocess.Popen', side_effect=popen_guard))
    stack.enter_context(mock.patch('socket.socket', BlockedSocket))
    stack.enter_context(mock.patch('socket.create_connection', side_effect=blocked))
    return stack


def authenticate_local_capture(primary, root, spec, producer, workflow):
    """Keep path guards: never import a second captured root into this process."""
    expected = primary.producer_files(spec, producer, workflow, primary.PANEL)
    for name, digest in expected.items():
        captured = Path(root) / name
        need(captured.is_file() and not captured.is_symlink() and sha(captured.read_bytes()) == digest,
             'complete original captured member ' + name)
        if name == primary.WORKFLOW:
            need(captured.read_bytes() == workflow, 'original reviewed two-token runtime workflow')
            continue
        local = primary.ROOT / name
        need(local.is_file() and not local.is_symlink() and local.read_bytes() == captured.read_bytes(),
             'one unchanged local verification root matches original captured bytes ' + name)
    need('automation/english_v6_b28_quality_readback.py' in expected
         and 'automation/english_v6_b28_profile.py' in expected
         and 'automation/english_v6_b14_decision.py' in expected,
         'the reader, explicit proof profile and B14 implementation are inside the original Git closure')
    return expected


def verify_worker(directory, lock, proof_root):
    validate_lock(lock)
    directory, proof_root = Path(directory), Path(proof_root)
    panel = lock['panel']
    need({path.name for path in directory.iterdir()} == {'api', 'originals', 'inputs'}, 'complete original quality roots only')
    run = document((directory / 'api/run.json').read_bytes())
    commit = document((directory / 'api/commit.json').read_bytes())
    artifacts = document((directory / 'api/artifacts.json').read_bytes())
    need(artifacts['total_count'] == len(artifacts['artifacts']) == 3, 'all three actual original quality artifacts')
    names = artifact_names(panel, run['id'])
    with profile.proof_scope(proof_root, panel), ExitStack() as stack, tempfile.TemporaryDirectory(prefix='b28-quality-originals-') as temporary:
        proof = profile.require_proof(panel)
        archives = {key: stack.enter_context(original_archive(directory / 'originals' / (key + '.zip'), lock[key], panel, key))
                    for key in ('result', 'producer', 'consumed')}
        captured = Path(temporary) / 'producer'; captured.mkdir()
        archives['producer'].extractall(captured)
        root = captured / (panel + '_producer')
        stem = EVALUATOR_STEMS[panel]
        driver = root / 'automation' / (stem + '_eval.py')
        spec_path, producer_path = (root / 'automation' / (stem + suffix) for suffix in ('_inputs.json', '_producer.json'))
        need(sha(driver.read_bytes()) == lock['evaluator_sha256'] and sha(spec_path.read_bytes()) == lock['inputs_sha256']
             and sha(producer_path.read_bytes()) == lock['producer_manifest_sha256'], 'original code and complete registrations')
        primary = importlib.import_module(stem + '_eval')
        need(Path(primary.__file__).resolve() == profile.HERE / driver.name
             and Path(primary.__file__).read_bytes() == driver.read_bytes()
             and primary.SPEC.read_bytes() == spec_path.read_bytes()
             and primary.PRODUCER.read_bytes() == producer_path.read_bytes(),
             'same exact original evaluator and registrations in one clean local interpreter root')
        for key in ('result', 'producer', 'consumed'):
            pin = lock[key]
            selected = [item for item in artifacts['artifacts'] if item['id'] == pin['id']]
            need(len(selected) == 1 and selected[0]['name'] == names[key], 'exact named original output ' + key)
            primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=selected[0]), role=key)
        spec = primary.protocol()
        need(spec['schema'] == SCHEMAS[panel] and spec['external_proof_binding_sha256'] == lock['external_proof_binding_sha256']
             and spec['producer_manifest_sha256'] == lock['producer_manifest_sha256'], 'same complete original B28 registration')
        fetched = document((captured / 'FETCH.json').read_bytes())
        original_identity = fetched['identity']; identity(original_identity)
        need(original_identity['run_id'] == str(lock['result']['run_id'])
             and original_identity['run_head'] == lock['result']['run_head'], 'actual first original quality execution')
        producer, workflow = primary.producer_contract(spec, original_identity['prepared_parent'], root)
        authenticate_local_capture(primary, root, spec, producer, workflow)
        primary.verify_fetched(captured, spec, original_identity, producer, workflow, fetched)
        primary.validate_execution_metadata(original_identity, run, commit)
        need(equal(commit, fetched[panel + '_commit']), 'independently saved complete commit equals original acquisition')
        expected = {'FETCH.json', 'CONSUMPTION.json'}
        for role in (panel, 'source'):
            expected |= {role + '_producer/' + name for name in primary.producer_files(spec, producer, workflow, role)}
        need({item.filename for item in archives['producer'].infolist() if not item.is_dir()} == expected,
             'entire original producer output; no supplementary missing-byte repair')
        need(profile.inventory(directory).keys() == profile.QUALITY_FILES,
             'all and only18 original APIs, outputs and twelve complete input ZIPs')
        value = source_admission(primary, directory / 'inputs', spec, original_identity, workflow, fetched, panel,
                                 proof.get('primary_quality'))
        admission = value['admission']
        consumed_raw = (captured / 'CONSUMPTION.json').read_bytes()
        need({item.filename for item in archives['consumed'].infolist() if not item.is_dir()} == {'CONSUMPTION.json'}
             and archives['consumed'].read('CONSUMPTION.json') == consumed_raw, 'same exact original durable no-replay marker')
        need(equal(document(consumed_raw), dict(identity=original_identity, marker_name=primary.MARKER,
            admission_sha256=sha(canonical(admission)))), 'original marker binds complete independently rebuilt source admission')
        checked = saved_arrays((directory / 'originals/result.zip').read_bytes(), spec, panel, admission,
                               value['selected_rows'], primary, value['decision'])
        result = checked['result']
        need(checked['original_result_sha256'] == lock['original_result_sha256'], 'exact original complete RESULT member')
        passed = result['panel_gate'] == 'PASS'
        bound = profile.binding(panel)
        certification = proof['result']['original_B27_failure_certification']
        return dict(schema=REPORT_SCHEMA, panel=panel, panel_gate=result['panel_gate'],
            model_pt_sha256=spec['model_pt_sha256'], assessment_sha256=spec['assessment_sha256'],
            training_design_sha256=spec['training_design_sha256'], training_assessment_sha256=spec['training_assessment_sha256'],
            original_B24_training_assessment_sha256=spec['original_B24_training_assessment_sha256'],
            pair_pin=spec['inputs']['pair'], original_result_sha256=checked['original_result_sha256'],
            original_identity=original_identity, original_lock_sha256=sha(canonical(lock)),
            external_proof_binding_sha256=lock['external_proof_binding_sha256'],
            original_training_report_sha256=bound['training_report_sha256'],
            original_B27_failure_certification=certification,
            original_B27_failure_report_sha256=certification['original_report_sha256'],
            logits_sha256=checked['logits_sha256'], actual_forwards=result['actual_forwards'],
            model_forward_calls=result['model_forward_calls'],
            verified_frame_count=sum(state['frames'] for row in result['records'] for roles in row['gains'].values() for state in roles.values()),
            admission=admission, source_inventory=value['source_inventory'], decision=value['decision'].decide(result['records']),
            all_original_arrays_verified=True, original_source_admission_recomputed=True, original_producer_git_verified=True,
            all_original_training_proof_verified=True, complete_original_Q1_failure_verified=True,
            complete_original_B27_failure_verified=True, natural_objective_returns_verified=384,
            natural_no_event_terms_recomputed=True, captured_code_matches_pinned_local_reader_root=True,
            external_training_originals=71, external_total_originals=71 if panel == 'primary' else 89,
            derived_B24_originals=5, derived_B24_is_new_qualification_source=False,
            derived_B24_origin='same-original-B27-TRAIN_receipts-members',
            primary_independently_verified=panel == 'replication',
            replication_open_allowed=passed and panel == 'primary', candidate_engineering_validation_allowed=passed and panel == 'replication',
            automatic_next_assessment_allowed=False, old_stops_and_reserves_preserved=True,
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0,
            promotion_allowed=False, product_released=False)


def validate_report(report, lock):
    certificate = failure_certification()
    need(report['schema'] == REPORT_SCHEMA and report['panel'] == lock['panel']
         and report['panel_gate'] in ('PASS', 'NO_PROMOTION')
         and report['original_result_sha256'] == lock['original_result_sha256']
         and report['original_lock_sha256'] == sha(canonical(lock))
         and report['external_proof_binding_sha256'] == lock['external_proof_binding_sha256']
         and report['original_B27_failure_certification'] == certificate
         and report['original_B27_failure_report_sha256'] == certificate['original_report_sha256']
         and all(report[key] is True for key in ('all_original_arrays_verified', 'original_source_admission_recomputed',
             'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified',
             'complete_original_B27_failure_verified', 'natural_no_event_terms_recomputed',
             'captured_code_matches_pinned_local_reader_root', 'old_stops_and_reserves_preserved'))
         and report['natural_objective_returns_verified'] == 384
         and report['external_training_originals'] == 71
         and report['external_total_originals'] == (71 if lock['panel'] == 'primary' else 89)
         and report['derived_B24_originals'] == 5 and report['derived_B24_is_new_qualification_source'] is False
         and report['derived_B24_origin'] == 'same-original-B27-TRAIN_receipts-members'
         and report['primary_independently_verified'] is (lock['panel'] == 'replication')
         and all(type(report[key]) is int and report[key] == 0 for key in
             ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates'))
         and report['replication_open_allowed'] is (report['panel_gate'] == 'PASS' and lock['panel'] == 'primary')
         and report['candidate_engineering_validation_allowed'] is (report['panel_gate'] == 'PASS' and lock['panel'] == 'replication')
         and report['promotion_allowed'] is report['product_released'] is report['automatic_next_assessment_allowed'] is False,
         'complete independently recomputed B28 result and exactly matching closed/open next-stage gates')
    return report


def verify_quality_original(directory, lock, proof_root):
    validate_lock(lock)
    need(not {'numpy', 'torch', 'onnxruntime'} & set(sys.modules), 'independent readback before scientific runtime imports')
    with tempfile.TemporaryDirectory(prefix='b28-quality-lock-') as temporary:
        path = Path(temporary) / 'lock.json'; path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--verify-worker',
                                       str(Path(directory).resolve()), str(path), str(Path(proof_root).resolve())])
    return validate_report(document(raw), lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    parser.add_argument('proof_root', type=Path)
    args = parser.parse_args()
    lock = document(args.lock.read_bytes())
    if args.verify_worker:
        validate_lock(lock)
        with offline_only(args.proof_root, lock['panel']):
            report = verify_worker(args.directory, lock, args.proof_root)
    else:
        report = verify_quality_original(args.directory, lock, args.proof_root)
    sys.stdout.buffer.write(canonical(report))


if __name__ == '__main__':
    main()
