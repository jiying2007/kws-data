"""Independent B27 complete-original quality readback, using stdlib only.

Every original source/provider payload and actual model return is reopened. An
exact complete PASS may open the next declared stage; complete NO_PROMOTION is
reported with all permissions false. STOP, missing output and unknown calls are
not complete results and can never pass this reader. No model is loaded or run.
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
import english_v6_b27_profile as profile
_base_reader = profile.preserved('english_v6_b24q1_quality_readback')
profile.inherit(_base_reader, globals())
import english_v6_b27_capacity as capacity
EVALUATOR_STEMS = {panel: 'english_v6_b27_' + panel for panel in ('primary', 'replication')}
SCHEMAS = {panel: 'english-v6-b27-fresh-' + panel + '-four-frozen-models-v1' for panel in EVALUATOR_STEMS}
ASSESSMENT_SHA = TRAINING_ASSESSMENT_SHA = profile.profile(False)['assessment_sha256']
DESIGN_SHA = profile.DESIGN_SHA
LOCK_SCHEMA = 'english-v6-b27-original-quality-lock-v1'
REPORT_SCHEMA = 'english-v6-b27-verified-original-quality-v1'
LOCK_FIELDS = _base_reader.LOCK_FIELDS | {'external_proof_binding_sha256'}
_base_source_admission = source_admission


def archive_limit(panel='primary', role=None):
    need(panel in SCHEMAS and role in {None, 'result', 'producer', 'consumed', 'B0', 'B9', 'pair', 'source', 'screen',
         'train_producer', 'source_producer', *('native_' + p for p in ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60'))},
         'explicit B27 panel and original role')
    return profile.LIMITS['source'] if role == 'source' else profile.LIMITS['core']


def validate_lock(lock):
    need(type(lock) is dict and set(lock) == LOCK_FIELDS and lock['schema'] == LOCK_SCHEMA
         and lock['panel'] in SCHEMAS, 'complete exact B27 original quality lock')
    for role in ('result', 'producer', 'consumed'):
        valid_artifact(lock[role], panel=lock['panel'], role=role)
    need(len({lock[key]['id'] for key in ('result', 'producer', 'consumed')}) == 3
         and all(lock[key][field] == lock['result'][field]
                 for key in ('producer', 'consumed') for field in ('run_id', 'run_head')),
         'three distinct complete originals from the same first quality run')
    for field in ('evaluator_sha256', 'inputs_sha256', 'producer_manifest_sha256',
                  'original_result_sha256', 'external_proof_binding_sha256'):
        need(valid_hash(lock[field]), 'exact original B27 digest ' + field)
    evaluator = profile.HERE / (EVALUATOR_STEMS[lock['panel']] + '_eval.py')
    need(sha(evaluator.read_bytes()) == lock['evaluator_sha256']
         and sha((profile.HERE / ('english_v6_b27_' + lock['panel'] + '_proof_binding.json')).read_bytes())
             == lock['external_proof_binding_sha256'], 'original immutable evaluator and complete proof registration')


def artifact_names(panel, run):
    need(panel in SCHEMAS, 'explicit original B27 panel')
    return dict(result='english-v6-b27-historical-control-' + panel + '-logits-' + str(run),
                producer='english-v6-b27-' + panel + '-producer-evidence-' + str(run),
                consumed='english-v6-b27-' + panel + '-consumed')


def source_admission(primary, inputs, spec, identity, workflow, fetched, panel, primary_quality=None):
    value = _base_source_admission(primary, inputs, spec, identity, workflow, fetched, panel, primary_quality)
    value['admission']['output_capacity'] = capacity.score_budget(value['selected_rows'])
    value['admission']['external_proof_binding_sha256'] = primary.spec_binding_sha()
    return value


def offline_only(proof_root, panel):
    """Only exact, scoped B27 TRAIN and recursive PRIMARY reader children."""
    root = Path(proof_root)
    need(root.is_absolute() and root == root.resolve() and not root.is_symlink(), 'regular exact original proof root')
    bound = profile.binding(panel)
    stack = ExitStack()
    old_import, old_popen, old_socket = builtins.__import__, subprocess.Popen, socket.socket
    forbidden = {'numpy', 'torch', 'torchaudio', 'onnxruntime', 'soundfile', 'transformers', 'piper', 'requests', 'httpx'}
    used = set()
    def import_guard(name, *args, **kwargs):
        need(name.split('.')[0] not in forbidden, 'no scientific or network runtime during independent B27 readback')
        return old_import(name, *args, **kwargs)
    def popen_guard(command, *args, **kwargs):
        need(type(command) is list and len(command) == 9
             and command[:4] == [sys.executable, '-I', '-S', '-B'] and command[5] == '--verify-worker'
             and not args and set(kwargs) <= {'stdin', 'stdout', 'stderr'}, 'only exact isolated complete-proof children')
        path, directory, lock_path, inputs = map(Path, (command[4], command[6], command[7], command[8]))
        for entry in (path, directory, lock_path, inputs):
            need(entry.is_absolute() and entry == entry.resolve() and not entry.is_symlink(), 'regular resolved child proof paths')
        need(path.is_file() and lock_path.is_file() and directory.is_dir() and inputs.is_dir(), 'complete original child files/directories')
        lock_raw = lock_path.read_bytes(); lock = document(lock_raw)
        need(lock_raw == canonical(lock), 'exact canonical child lock bytes')
        if path == profile.HERE / 'english_v6_b27_training_readback.py':
            kind = 'training'
            need(sha(path.read_bytes()) == bound['training_reader_sha256']
                 and lock == bound['training_lock'] and directory == root / 'training-origin/trained'
                 and inputs == root / 'training-origin/qualification'
                 and profile.inventory(root / 'training-origin') == bound['files'],
                 'one exact pinned TRAIN reader, complete41 originals and actual original lock')
        else:
            kind = 'primary'
            need(panel == 'replication' and path == Path(__file__).resolve()
                 and sha(path.read_bytes()) == bound['primary_quality_reader_sha256']
                 and lock == bound['primary_quality_lock'] and lock['panel'] == 'primary'
                 and directory == root / 'primary-quality' and inputs == root
                 and profile.inventory(directory) == bound['primary_quality_files'],
                 'only the complete explicitly pinned same-B27 original PRIMARY child')
        need(kind not in used, 'no verifier replay or arbitrary additional subprocess')
        used.add(kind)
        return old_popen(command, **kwargs)
    def blocked(*args, **kwargs):
        raise ValueError('no network during B27 original quality readback')
    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs): blocked()
    stack.enter_context(mock.patch('builtins.__import__', side_effect=import_guard))
    stack.enter_context(mock.patch('subprocess.Popen', side_effect=popen_guard))
    stack.enter_context(mock.patch('socket.socket', BlockedSocket))
    stack.enter_context(mock.patch('socket.create_connection', side_effect=blocked))
    return stack


def verify_worker(directory, lock, proof_root):
    validate_lock(lock)
    directory, proof_root = Path(directory), Path(proof_root)
    panel = lock['panel']
    need({p.name for p in directory.iterdir()} == {'api', 'originals', 'inputs'}, 'complete original quality proof roots only')
    run = document((directory / 'api/run.json').read_bytes())
    commit = document((directory / 'api/commit.json').read_bytes())
    artifacts = document((directory / 'api/artifacts.json').read_bytes())
    need(artifacts['total_count'] == len(artifacts['artifacts']) == 3, 'all three actual original quality artifacts')
    names = artifact_names(panel, run['id'])
    with profile.proof_scope(proof_root, panel), ExitStack() as stack, tempfile.TemporaryDirectory(prefix='b27-quality-originals-') as temporary:
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
             and sha(producer_path.read_bytes()) == lock['producer_manifest_sha256'], 'actual original code and full registrations')
        primary = load(driver, stem + '_eval')
        for key in ('result', 'producer', 'consumed'):
            pin = lock[key]
            selected = [item for item in artifacts['artifacts'] if item['id'] == pin['id']]
            need(len(selected) == 1 and selected[0]['name'] == names[key], 'exact named original output ' + key)
            primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=selected[0]), role=key)
        spec = primary.protocol()
        need(spec['schema'] == SCHEMAS[panel] and spec['external_proof_binding_sha256'] == lock['external_proof_binding_sha256']
             and spec['producer_manifest_sha256'] == lock['producer_manifest_sha256'], 'same complete original B27 registration')
        fetched = document((captured / 'FETCH.json').read_bytes())
        original_identity = fetched['identity']; identity(original_identity)
        need(original_identity['run_id'] == str(lock['result']['run_id'])
             and original_identity['run_head'] == lock['result']['run_head'], 'actual first original quality execution')
        producer, workflow = primary.producer_contract(spec, original_identity['prepared_parent'], root)
        primary.verify_fetched(captured, spec, original_identity, producer, workflow, fetched)
        primary.validate_execution_metadata(original_identity, run, commit)
        need(equal(commit, fetched[panel + '_commit']), 'complete independently saved original commit equals acquisition')
        expected = {'FETCH.json', 'CONSUMPTION.json'}
        for role in (panel, 'source'):
            expected |= {role + '_producer/' + name for name in primary.producer_files(spec, producer, workflow, role)}
        need({item.filename for item in archives['producer'].infolist() if not item.is_dir()} == expected,
             'complete original producer output; no supplement and no nested historical archive')
        need({p.name for p in (directory / 'api').iterdir()} == {'run.json', 'commit.json', 'artifacts.json'}
             and {p.name for p in (directory / 'originals').iterdir()} == {'result.zip', 'producer.zip', 'consumed.zip'}
             and {p.name for p in (directory / 'inputs').iterdir()} == {name + '.zip' for name in primary.archive_pins(spec)},
             'exact full original API, output and input directory closure')
        evidence = source_admission(primary, directory / 'inputs', spec, original_identity, workflow, fetched, panel,
                                    proof.get('primary_quality'))
        admission = evidence['admission']
        consumed_raw = (captured / 'CONSUMPTION.json').read_bytes()
        need({item.filename for item in archives['consumed'].infolist() if not item.is_dir()} == {'CONSUMPTION.json'}
             and archives['consumed'].read('CONSUMPTION.json') == consumed_raw, 'same exact original durable no-replay marker')
        need(equal(document(consumed_raw), dict(identity=original_identity, marker_name=primary.MARKER,
                    admission_sha256=sha(canonical(admission)))), 'original consumption matches full independent raw-source admission')
        checked = saved_arrays((directory / 'originals/result.zip').read_bytes(), spec, panel, admission,
                              evidence['selected_rows'], primary, evidence['decision'])
        result = checked['result']
        need(checked['original_result_sha256'] == lock['original_result_sha256'], 'exact original RESULT member')
        passed = result['panel_gate'] == 'PASS'
        return dict(schema=REPORT_SCHEMA, panel=panel, panel_gate=result['panel_gate'],
            model_pt_sha256=spec['model_pt_sha256'], assessment_sha256=spec['assessment_sha256'],
            training_design_sha256=spec['training_design_sha256'], training_assessment_sha256=spec['training_assessment_sha256'],
            original_B24_training_assessment_sha256=spec['original_B24_training_assessment_sha256'],
            pair_pin=spec['inputs']['pair'], original_result_sha256=checked['original_result_sha256'],
            original_identity=original_identity, original_lock_sha256=sha(canonical(lock)),
            external_proof_binding_sha256=lock['external_proof_binding_sha256'],
            original_training_report_sha256=profile.binding(panel)['training_report_sha256'],
            original_B24Q1_failure_report_sha256=profile.Q1_REPORT_SHA,
            logits_sha256=checked['logits_sha256'], actual_forwards=result['actual_forwards'],
            model_forward_calls=result['model_forward_calls'],
            verified_frame_count=sum(state['frames'] for row in result['records'] for roles in row['gains'].values() for state in roles.values()),
            admission=admission, source_inventory=evidence['source_inventory'], decision=evidence['decision'].decide(result['records']),
            all_original_arrays_verified=True, original_source_admission_recomputed=True, original_producer_git_verified=True,
            all_original_training_proof_verified=True, complete_original_Q1_failure_verified=True,
            replication_open_allowed=passed and panel == 'primary', candidate_engineering_validation_allowed=passed and panel == 'replication',
            automatic_next_assessment_allowed=False, old_stops_and_reserves_preserved=True,
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, promotion_allowed=False, product_released=False)


def original_archive(path, pin, panel, role):
    valid_artifact(pin, panel=panel, role=role)
    raw = Path(path).read_bytes()
    need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'complete exact original artifact bytes')
    return opened(raw, archive_limit(panel, role))


def verify_quality_original(directory, lock, proof_root):
    validate_lock(lock)
    need(not {'numpy', 'torch', 'onnxruntime'} & set(sys.modules), 'independent readback before scientific runtime imports')
    with tempfile.TemporaryDirectory(prefix='b27-quality-lock-') as temporary:
        path = Path(temporary) / 'lock.json'; path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--verify-worker',
                                       str(Path(directory).resolve()), str(path), str(Path(proof_root).resolve())])
    report = document(raw)
    need(report['schema'] == REPORT_SCHEMA and report['panel'] == lock['panel']
         and report['panel_gate'] in ('PASS', 'NO_PROMOTION')
         and report['original_result_sha256'] == lock['original_result_sha256']
         and report['original_lock_sha256'] == sha(canonical(lock))
         and all(report[key] is True for key in ('all_original_arrays_verified', 'original_source_admission_recomputed',
             'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified'))
         and all(type(report[key]) is int and report[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls'))
         and report['replication_open_allowed'] is (report['panel_gate'] == 'PASS' and lock['panel'] == 'primary')
         and report['candidate_engineering_validation_allowed'] is (report['panel_gate'] == 'PASS' and lock['panel'] == 'replication')
         and report['promotion_allowed'] is report['product_released'] is report['automatic_next_assessment_allowed'] is False,
         'complete independently recomputed original result and exactly matching downstream flags')
    return report


def saved_arrays(raw_zip, spec, panel, admission, selected_rows, primary, decision):
    """Require a complete actual outcome from every original return and unchanged B14."""
    need(spec["schema"] == SCHEMAS[panel] and spec["scoring_allowed"] is True
         and spec["assessment_sha256"] == ASSESSMENT_SHA
         and spec["training_assessment_sha256"] == TRAINING_ASSESSMENT_SHA
         and spec["training_design_sha256"] == DESIGN_SHA and spec["gains"] == list(GAINS)
         and type(spec["maximum_model_forwards"]) is int and spec["maximum_model_forwards"] == 1440
         and spec["same_job_matched_control"] is spec["historical_control_retrained"] is False
         and spec["model_comparison"] == primary.MODEL_COMPARISON,
         "same preregistered assessment, models, gains and full forward ceiling")
    fixed_models = spec["model_pt_sha256"]
    need(set(fixed_models) == set(ARMS) and all(valid_hash(value) for value in fixed_models.values())
         and len(set(fixed_models.values())) == 4
         and fixed_models == dict(B0=primary.B0_PT, B9=primary.B9_PT, control=primary.CONTROL_PT, challenger=primary.CANDIDATE_PT),
         "four exact original fixed model roles")
    need(type(selected_rows) is list and bool(selected_rows), "complete exact source-admitted originals")
    for row in selected_rows:
        need(type(row["id"]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", row["id"])
             and valid_hash(row["pcm_sha256"]) and valid_hash(row["wav_sha256"])
             and type(row["samples"]) is int and 3200 <= row["samples"] <= 240000
             and row["cohort"] in ("synthetic", "natural") and row["tier"] in ("silver", "weak")
             and type(row["observed_class"]) is int and row["observed_class"] in (0, 2)
             and (row["cohort"] != "natural" or row["observed_class"] == 0),
             "whole exact admitted original with its frozen source label")
    need(len({row["id"] for row in selected_rows}) == len({row["pcm_sha256"] for row in selected_rows}) == len(selected_rows),
         "every admitted original is present once")
    expected_rows = [primary.score_identity(row, row["cohort"]) for row in selected_rows]
    audit = admission["source_audit"]
    need(audit["schema"] == "english-v6-b27-full-source-and-quarantine-audit-v1"
         and audit["original_source_rows"] == 120 and audit["observer_eligible_rows"] == admission["observer_eligible_rows"]
         and audit["qualified_rows"] == len(selected_rows) and audit["quarantined_kws_forwards"] == 0
         and not set(audit["quarantined_ids"]) & {r["id"] for r in selected_rows}
         and equal(audit["source_policy"], spec["source_policy"]),
         "all admitted originals only; quarantine and all120/N evidence survive independent readback")
    measured = primary.source_counts(selected_rows)
    need(admission["schema"] == SCHEMAS[panel] and admission["assessment_sha256"] == ASSESSMENT_SHA
         and admission["training_design_sha256"] == DESIGN_SHA
         and admission["training_assessment_sha256"] == TRAINING_ASSESSMENT_SHA
         and equal(admission["admitted_rows"], expected_rows) and equal(admission["source_coverage"], measured)
         and equal(spec["source_gate"], primary.GATE) and all(measured[name] >= minimum for name, minimum in primary.GATE.items())
         and admission["source_gate"] == "SOURCE_GATE_PASS" and admission["source_shortfall"] == {}
         and type(admission["model_calls"]) is int and admission["model_calls"] == 0
         and type(admission["original_source_rows"]) is int and admission["original_source_rows"] == 120
         and admission["old_development_reused"] is admission["promotion_allowed"] is False
         and admission["replication_opened"] is (panel == "replication")
         and admission["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and admission["same_job_matched_control"] is admission["historical_control_retrained"] is False
         and admission["model_comparison"] == primary.MODEL_COMPARISON,
         "exact independently reauthenticated whole-source admission")
    identity(admission["identity"])
    need(admission['output_capacity'] == capacity.score_budget(selected_rows), 'complete output capacity frozen before model calls')
    calls = len(selected_rows) * 12
    need(calls <= 1440, "unchanged maximum 1440 actual forwards")
    keys = {row["id"] + "__" + gain + "__" + role for row in selected_rows for gain in GAINS for role in ARMS}
    required = {"RESULT.json", "STARTED.json", "ADMISSION.json", "ALL_FRAME_LOGITS.npz"}
    required |= {"attempts/" + key + ".json" for key in keys} | {"raw_outputs/" + key + ".npy" for key in keys}
    need(len(raw_zip) <= admission['output_capacity']['maximum_zip_bytes'], 'original output fits pre-call complete envelope')
    with opened(raw_zip) as archive:
        files = {item.filename for item in archive.infolist() if not item.is_dir()}
        dirs = {item.filename for item in archive.infolist() if item.is_dir()}
        need(files == required and dirs <= {"attempts/", "raw_outputs/"}, "every original output, journal and metadata; no missing or orphan member")
        stored_admission = document(archive.read("ADMISSION.json"))
        result = document(archive.read("RESULT.json"))
        started = document(archive.read("STARTED.json"))
        need(equal(stored_admission, admission), "original ADMISSION exactly equals independent raw-source recomputation")
        for report in (started, result):
            need(report["schema"] == SCHEMAS[panel] and equal({key: report[key] for key in IDENTITY_FIELDS}, admission["identity"])
                 and report["assessment_sha256"] == ASSESSMENT_SHA
                 and report["training_assessment_sha256"] == TRAINING_ASSESSMENT_SHA
                 and report["training_design_sha256"] == DESIGN_SHA
                 and report["registered_inputs_sha256"] == admission["registered_inputs_sha256"]
                 and report["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
                 and report["workflow_sha256"] == admission["workflow_sha256"]
                 and report["implementation_sha256"] == sha(Path(primary.__file__).read_bytes())
                 and report["B24Q1_failed_primary_preserved"] is report["all_old_stops_and_reserves_preserved"] is True
                 and report["source_gate"] == "SOURCE_GATE_PASS" and report["source_shortfall"] == {}
                 and equal(report["source_coverage"], measured) and equal(report["model_pt_sha256"], fixed_models)
                 and equal(report["source_audit"], audit) and report["B24R1_original_source_stop_preserved"] is True
                 and report["original_whole_panel_failure_contract_unchanged"] is False
                 and report["installed_software_baseline"] == "B0" and report["B20_failed_primary_preserved"] is report["B21_failed_primary_preserved"] is report["B22_failed_primary_preserved"] is report["B23_failed_primary_preserved"] is report["B24_original_source_stop_preserved"] is True
                 and report["same_job_matched_control"] is report["historical_control_retrained"] is False
                 and report["model_comparison"] == primary.MODEL_COMPARISON
                 and report["fresh_source_originals"] is True and report["original_natural_reused_without_reobservation"] is False
                 and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
                 and all(report[key] is False for key in FALSE_FIELDS)
                 and report["replication_opened"] is (panel == "replication") and report["ssc305_physical"] is False
                 and report["natural_field_far_per_hour"] is report["field_3_5m_frr"] is None,
                 "same actual producer, exact original models and honest unreleased scope")
            if panel == "replication":
                need(report["panel"] == "replication"
                     and report["primary_result_sha256"] == admission["primary_result_sha256"]
                     and report["primary_quality_lock_sha256"] == admission["primary_quality_lock_sha256"],
                     "replica opening binds the exact original primary PASS")
        need(started["panel_gate"] == "STARTED" and started["records"] == []
             and all(type(started[key]) is int and started[key] == 0 for key in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and started["candidate_engineering_validation_allowed"] is started["replication_open_allowed"] is False,
             "honest before-forward STARTED record")
        need(result["panel_gate"] in ("PASS", "NO_PROMOTION") and result["replay_allowed"] is False and result["consumed_unknown"] == []
             and "error" not in result and "error_type" not in result
             and all(type(result[key]) is int and result[key] == calls for key in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and equal(result["model_forward_calls"], dict.fromkeys(ARMS, len(selected_rows) * 3)),
             "only complete actual PASS or NO_PROMOTION; STOP, partial return and unknown consumption stay closed")
        need(result["replication_open_allowed"] is (panel == "primary" and result["panel_gate"] == "PASS")
             and result["candidate_engineering_validation_allowed"] is (panel == "replication" and result["panel_gate"] == "PASS"), "next-stage flags exactly match this original PASS")
        if panel == "replication":
            need(result["research_replication_passed"] is (result["panel_gate"] == "PASS") and started["research_replication_passed"] is False,
                 "only completed original replication admits engineering")
        records = result["records"]
        primary.validate_scored_join(records, expected_rows)
        npz = archive.read("ALL_FRAME_LOGITS.npz")
        need(sha(npz) == result["logits_sha256"], "exact original full-frame NPZ hash")
        with opened(npz) as arrays:
            need(set(arrays.namelist()) == {key + ".npy" for key in keys}, "every exact original/gain/model array")
            for source, record in zip(selected_rows, records, strict=True):
                need(set(record) == set(primary.IDENTITY_FIELDS) | {"gains"} and set(record["gains"]) == set(GAINS),
                     "only exact admitted row identity and three fixed gains")
                frames = (source["samples"] - 512) // 160 + 1
                for gain in GAINS:
                    need(set(record["gains"][gain]) == set(ARMS), "all four frozen model roles")
                    for role in ARMS:
                        key = source["id"] + "__" + gain + "__" + role
                        state = record["gains"][gain][role]
                        need(set(state) == STATE_FIELDS and state["array"] == key
                             and type(state["frames"]) is int and state["frames"] == frames
                             and type(state["events"]) is int and state["events"] >= 0, "whole uncropped original frame count")
                        journal = document(archive.read("attempts/" + key + ".json"))
                        need(equal(journal, dict(key=key, role=role, source_id=source["id"], gain=gain, status="consumed_unknown")),
                             "one durable actual-forward-entry journal; no replay")
                        body, labels = read_npy(arrays.read(key + ".npy"), (frames, 4))
                        returned, returned_labels = read_npy(archive.read("raw_outputs/" + key + ".npy"), (1, frames, 4))
                        need(body == returned and labels == returned_labels and sha(body) == state["sha256"],
                             "every retained tensor byte matches the complete actual returned batch")
                        events = primary.ctc_events(labels)
                        need(equal({name: state[name] for name in events}, events), "tokens, events and sample clocks recompute from raw frames")
        replay = decision.decide(records)
        need(replay["panel_gate"] == result["panel_gate"] and all(equal(result[key], value) for key, value in replay.items()),
             "every unchanged original B14 condition independently recomputes the exact actual verdict")
        return dict(result=result, original_result_sha256=sha(archive.read("RESULT.json")), logits_sha256=sha(npz))


def fetch_quality_original(directory, lock, proof_root, reuse_inputs=None):
    """Fetch exact output/input artifact IDs, then independently reopen everything."""
    validate_lock(lock)
    profile.binding(lock['panel'])
    directory = Path(directory)
    need(directory.is_absolute() and directory == directory.resolve() and directory.parent.is_dir()
         and not directory.exists(), 'fresh complete immutable quality proof directory')
    def gh(endpoint):
        return subprocess.run(['gh', 'api', 'repos/' + REPO + '/' + endpoint, '--allow-escape-sequences'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    directory.mkdir()
    for name in ('api', 'originals', 'inputs'): (directory / name).mkdir()
    for name, endpoint in (('run', 'actions/runs/' + str(lock['result']['run_id'])),
            ('commit', 'commits/' + lock['result']['run_head']),
            ('artifacts', 'actions/runs/' + str(lock['result']['run_id']) + '/artifacts?per_page=100')):
        (directory / 'api' / (name + '.json')).write_bytes(gh(endpoint))
    for role in ('result', 'producer', 'consumed'):
        pin = lock[role]; raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'entire exact fetched original ' + role)
        (directory / 'originals' / (role + '.zip')).write_bytes(raw)
    with original_archive(directory / 'originals/producer.zip', lock['producer'], lock['panel'], 'producer') as archive:
        raw = archive.read(lock['panel'] + '_producer/automation/' + EVALUATOR_STEMS[lock['panel']] + '_inputs.json')
        need(sha(raw) == lock['inputs_sha256'], 'exact original registration before following any input pin')
        spec = document(raw)
    need(set(spec['inputs']) == {'B0', 'B9', 'pair', 'source', 'screen'}
         and set(spec['native_inputs']) == {'qwen', 'whisper', 'parakeet', 'xlsr', 'lv60'}
         and set(spec['upstream_producers']) == {'train', 'source'}, 'complete twelve exact original inputs')
    pins = {**spec['inputs'], **{'native_' + key: pin for key, pin in spec['native_inputs'].items()},
            **{key + '_producer': pin for key, pin in spec['upstream_producers'].items()}}
    files = {key + '.zip': dict(bytes=pin['bytes'], sha256=pin['sha256']) for key, pin in pins.items()}
    if reuse_inputs is not None:
        reuse_inputs = Path(reuse_inputs)
        need(profile.inventory(reuse_inputs) == files, 'all and only explicitly reused complete original12 inputs')
    for key, pin in pins.items():
        valid_artifact(pin, panel=lock['panel'], role=key)
        raw = (reuse_inputs / (key + '.zip')).read_bytes() if reuse_inputs is not None else gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'whole original scientific input ' + key)
        (directory / 'inputs' / (key + '.zip')).write_bytes(raw)
    need(profile.inventory(directory / 'inputs') == files, 'every full original input retained')
    return verify_quality_original(directory, lock, proof_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    parser.add_argument('proof_root', type=Path)
    args = parser.parse_args()
    lock = document(args.lock.read_bytes())
    if args.verify_worker:
        with offline_only(args.proof_root, lock['panel']):
            report = verify_worker(args.directory, lock, args.proof_root)
    else:
        report = verify_quality_original(args.directory, lock, args.proof_root)
    sys.stdout.buffer.write(canonical(report))
