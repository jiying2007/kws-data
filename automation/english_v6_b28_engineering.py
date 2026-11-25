"""Inert B28 engineering over the unchanged frozen numerical implementation.

The historical source file and every numeric helper remain unchanged.
B28 requires its own complete TRAIN and two independently authenticated quality
PASSes, including the original B27 failure and 384 no-event terms. The old B27
failure grants no permission here. Future weights, quality and proof bindings
remain null; only a separately reviewed binding can admit an engineering run.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "english_v6_b24r1_engineering.py"
ORIGINAL_SHA = "6e6c552804504aa4a027a71312b59bde6fa97237eec7cb138463803fbe9bc1e1"
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError("unchanged original 140-call engineering implementation")
_definition = importlib.util.spec_from_file_location("b28_frozen_engineering", ORIGINAL)
original = importlib.util.module_from_spec(_definition)
_definition.loader.exec_module(original)
# This private module's functions retain their original implementations. A
# small, explicit set of identity adapters is installed below.
for _name, _value in vars(original).copy().items():
    if not _name.startswith("_"):
        globals()[_name] = _value

SPEC = HERE / "english_v6_b28_engineering_inputs.json"
TEMPLATE = HERE / "english_v6_b28_engineering_workflow_template.yml"
SCHEMA = "english-v6-b28-dual-pass-software-sdk-v1"
QUALITY_READER = "english_v6_b28_quality_readback"
WORKFLOW = ".github/workflows/english-v6-b28-engineering-once.yml"
BRANCH = "research/en-v6-b28-engineering-once-20261125"
PARENT_TOKEN = "__B28_REVIEWED_ENGINEERING_PARENT__"
MARKER = "english-v6-b28-engineering-consumed"
ASSESSMENT_SHA = None
TRAINING_ASSESSMENT_SHA = ASSESSMENT_SHA
ORIGINAL_B24_TRAINING_ASSESSMENT_SHA = "3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad"
DESIGN_SHA = None
CONTROL_PT = "f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4"
INITIAL_PT = "784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f"
B27_PT = "dd0d4bc08be1162886ab462dfb3ad7f4503150a57ee13272d01f41e80ca1069f"
CANDIDATE_PT = None  # Only the actual B28 input spec and full TRAIN proof bind it.
MODEL_COMPARISON = "B28_new_challenger_against_immutable_historical_B24_challenger"
STEM = "mero43-b28-challenger192"
STEMS = {"control": "control/models/mero43-b24-challenger192", "challenger": "challenger/models/" + STEM}
Q1_FAILURE_REPORT_SHA = "9cd5df45c5b9ca22b0eb56f3fdda4788b2420e906a35bb754af9c23eb38be62f"
for _name in ("SPEC", "TEMPLATE", "SCHEMA", "QUALITY_READER", "WORKFLOW", "BRANCH",
              "PARENT_TOKEN", "MARKER", "ASSESSMENT_SHA", "TRAINING_ASSESSMENT_SHA", "DESIGN_SHA",
              "CONTROL_PT", "CANDIDATE_PT", "MODEL_COMPARISON", "STEM", "STEMS"):
    setattr(original, _name, globals()[_name])

_frozen_producer = original.producer
_frozen_verify_producer = original.verify_producer
_frozen_source_check = original.source_check


def evidence_capacity(directory):
    """Bound the complete actual code archive before any engineering budget.

    Full historical qualification and TRAIN originals remain externally pinned.
    Neither those ZIPs nor the two quality archives enter this code producer.
    All four new engineering archive roles retain the original 128MB ceiling.
    """
    capacity = module("english_v6_b24q1_capacity")
    value = capacity.inventory(directory)
    names = list(value["files"]) + value["directories"]
    expanded = sum(pin["bytes"] for pin in value["files"].values())
    maximum = capacity.archive_limit("core")
    envelope = capacity.zip_bound(expanded, len(names), sum(len(name.encode()) for name in names))
    need(maximum == 128_000_000 and expanded <= maximum and envelope <= maximum,
         "complete actual B28 engineering producer fits its unchanged 128MB archive boundary")
    return dict(expanded_bytes=expanded, members=len(names), conservative_zip_bytes=envelope,
                maximum_bytes=maximum, files_verified=len(value["files"]))


def producer(inputs, pipeline, identity):
    _frozen_producer(inputs, pipeline, identity)
    evidence_capacity(inputs / "producer")


def verify_producer(inputs, pipeline, identity):
    _frozen_verify_producer(inputs, pipeline, identity)
    evidence_capacity(inputs / "producer")


def source_check():
    doc = json.loads(SPEC.read_bytes())
    registration(doc)
    if doc['enabled'] is True:
        protocol(doc)
        status = 'BOUND_REQUIRES_ORIGINAL_DUAL_PASS_READBACK'
    else:
        need(doc['enabled'] is False and PARENT_TOKEN in TEMPLATE.read_text()
             and all(doc[name] is None for name in ('assessment_sha256', 'training_assessment_sha256',
                 'training_design_sha256', 'quality_reader_sha256', 'original_training_report_sha256',
                 'candidate_files', 'original_B27_failure_certification', 'original_B27_failure_report_sha256'))
             and doc['quality_locks'] == dict(primary=None, replication=None)
             and doc['inputs']['pair'] is None and doc['model_pt_sha256']['challenger'] is None,
             'prepared B28 engineering is inert; no invented future proof or candidate identity')
        status = 'INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS'
    return dict(schema=SCHEMA, status=status, candidate_id='B28', pack_bytes=PACK_BYTES,
        original_count=ORIGINALS, maximum_backend_calls=MAX_CALLS, model_forwards=0,
        compiled_programs=0, packs_created=0, B27_terminal_NO_PROMOTION_preserved=True,
        physical_ssc305=False, product_promotion_allowed=False)


def registration(doc):
    need(doc['schema'] == SCHEMA and doc['candidate_role'] == 'challenger'
         and doc['pipeline_commit'] == PIPELINE_COMMIT and doc['quality_reader'] == QUALITY_READER
         and doc['prepared_parent'] is None and doc['parent_authority'] == 'reviewed_workflow'
         and type(doc['original_count']) is int and doc['original_count'] == ORIGINALS == 14
         and doc['conditions'] == list(CONDITIONS)
         and type(doc['maximum_backend_calls']) is int and doc['maximum_backend_calls'] == MAX_CALLS == 140
         and doc['onnx_max_abs'] == .001 and doc['onnx_mean_abs'] == .0001
         and doc['c_max_abs'] == .005 and doc['c_mean_abs'] == .0005
         and doc['model_comparison'] == MODEL_COMPARISON
         and doc['same_job_matched_control'] is doc['historical_control_retrained'] is False,
         'same fixed14 originals, two conditions, five backends and original numerical limits')
    need(all(doc[name] is True for name in ('evaluation_only', 'B24_original_source_stop_preserved',
             'B24R1_original_source_stop_preserved', 'B24Q1_terminal_NO_PROMOTION_preserved',
             'B27_terminal_NO_PROMOTION_preserved'))
         and all(doc[name] is False for name in ('old_B24_sources_used_for_scoring', 'old_B24_replication_opened',
             'old_B24R1_sources_used_for_scoring', 'old_B24R1_replication_opened', 'old_B24Q1_replication_opened',
             'old_B27_sources_used_for_scoring', 'old_B27_replication_opened', 'old_B27_candidate_engineering_opened',
             'automatic_R2_allowed', 'training_allowed', 'threshold_search', 'new_source_allowed',
             'cal_locked_access', 'physical_ssc305', 'product_promotion_allowed', 'product_decoder_settings_defined',
             'derived_B24_is_new_qualification_source'))
         and all(type(doc[name]) is int and doc[name] == 0 for name in
             ('new_optimizer_updates', 'old_B27_optimizer_updates_reexecuted'))
         and doc['original_B24Q1_failure_report_sha256'] == Q1_FAILURE_REPORT_SHA
         and doc['original_B24_training_assessment_sha256'] == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA,
         'every old stop remains terminal; no old panel, optimizer or product permission is reopened')
    need(all(type(doc[name]) is int and doc[name] == expected for name, expected in
             (('natural_objective_returns_verified', 384), ('external_training_originals', 71),
              ('external_replication_originals', 89), ('derived_B24_originals', 5)))
         and set(doc['inputs']) == set(FROZEN_INPUTS) | {'pair'}
         and all(doc['inputs'][name] == pin for name, pin in FROZEN_INPUTS.items())
         and set(doc['model_pt_sha256']) == {'B0', 'B9', 'control', 'challenger'}
         and all(doc['model_pt_sha256'][name] == digest for name, digest in
             (('B0', B0_PT), ('B9', B9_PT), ('control', CONTROL_PT))),
         'unchanged engineering source/B0 archives and complete distinct B28 proof counts')
    return doc


def failure_certification(doc):
    value = doc['original_B27_failure_certification']
    need(type(value) is dict and value.get('schema') == 'english-v6-b28-original-b27-failure-certification-binding-v1'
         and value.get('original_reader_path') == 'automation/english_v6_b27_quality_readback.py'
         and value.get('original_reader_sha256') == '87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa'
         and value.get('adapter_path') == 'automation/english_v6_b27_primary_no_promotion_readback.py'
         and value.get('adapter_report_schema') == 'english-v6-b27-primary-rooted-no-promotion-readback-v1'
         and value.get('original_report_sha256') == doc['original_B27_failure_report_sha256']
         and all(valid_hash(value.get(name)) for name in
             ('original_report_sha256', 'adapter_sha256', 'adapter_report_sha256'))
         and value.get('original_standalone_continuation_succeeded') is False
         and all(type(value.get(name)) is int and value[name] == 0 for name in
             ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates')),
         'separate original B27 reader and actual import-scope certificate; never an invented old-reader success')
    return value


def proof_bindings(doc):
    """Bind both quality locks to the same complete actual B28 TRAIN proof."""
    profile = module("english_v6_b28_profile")
    profile.profile()
    bound = {panel: profile.binding(panel) for panel in ("primary", "replication")}
    for panel, value in bound.items():
        path = HERE / ("english_v6_b28_" + panel + "_proof_binding.json")
        need(sha(path.read_bytes()) == doc["quality_locks"][panel]["external_proof_binding_sha256"]
             and value["training_report_sha256"] == doc["original_training_report_sha256"],
             "same complete externally retained original proof binding: " + panel)
    first, second = bound["primary"], bound["replication"]
    need(all(first[key] == second[key] for key in
             ("files", "training_lock", "training_reader_sha256", "training_report_sha256"))
         and second["primary_quality_lock"] == doc["quality_locks"]["primary"]
         and second["primary_quality_reader_sha256"] == doc["quality_reader_sha256"],
         "both panels reauthenticate the same full TRAIN and replication requires this exact PRIMARY")
    need(len(first['files']) == len(second['files']) == 71
         and len(second['primary_quality_files']) == 18
         and len(second['files']) + len(second['primary_quality_files']) == 89,
         'all71 original TRAIN files and all89 replica originals, never the88 training audio identities')
    need(failure_certification(doc) == module('english_v6_b28_train').failure_certification(),
         'actual original B27 failure binding is identical to the reviewed B28 TRAIN prerequisite')
    training = first["training_lock"]["training"]
    pin = training["artifacts"]["models"]
    need(training["model_pt_sha256"] == doc["model_pt_sha256"]
         and doc["inputs"]["pair"] == dict(id=pin["id"], bytes=pin["bytes"], sha256=pin["sha256"],
             run_id=training["run_id"], run_head=training["run_head"]),
         "candidate hashes and pair archive derive only from the original completed B28 TRAIN")
    return bound


def protocol(value=None):
    # No code, proof, network, source or candidate work before this closed gate.
    doc = json.loads(SPEC.read_bytes()) if value is None else value
    need(doc.get("enabled") is True, "INERT: actual B28 primary and replication PASSes are not bound")
    registration(doc)
    need(all(valid_hash(doc[name]) for name in ('assessment_sha256', 'training_assessment_sha256',
             'training_design_sha256', 'original_training_report_sha256', 'original_B27_failure_report_sha256')),
         'all prospective and completed B28 identities must be explicit before opening engineering')
    profile = module('english_v6_b28_profile')
    p = profile.profile()
    need(doc['assessment_sha256'] == doc['training_assessment_sha256'] == p['assessment_sha256']
         and doc['training_design_sha256'] == p['training_design_sha256']
         and sha((HERE / 'english_v6_b28_panels/assessment.json').read_bytes()) == p['assessment_sha256']
         and sha((HERE / 'english_v6_b28_natural_event_design.json').read_bytes()) == p['training_design_sha256']
         and sha((HERE / 'english_v6_b24_panels/assessment.json').read_bytes()) == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA,
         'the actual prospective B28 design and independent assessment, with old B24 ancestry kept separate')
    failure_certification(doc)
    need(valid_hash(doc["quality_reader_sha256"])
         and sha((HERE / (QUALITY_READER + ".py")).read_bytes()) == doc["quality_reader_sha256"],
         "reviewed original-array and producer/source-admission reader")
    need(set(doc["inputs"]) == set(FROZEN_INPUTS) | {"pair"}
         and all(doc["inputs"][key] == pin for key, pin in FROZEN_INPUTS.items()), "only existing exposed engineering PCM")
    for pin in doc["inputs"].values():
        valid_pin(pin)
    need(set(doc["quality_locks"]) == {"primary", "replication"}, "two actual acoustic results")
    for panel, lock in doc["quality_locks"].items():
        need(type(lock) is dict and set(lock) == {"schema", "panel", "result", "producer", "consumed",
             "evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256",
             "external_proof_binding_sha256"}
             and lock["schema"] == "english-v6-b28-original-quality-lock-v1" and lock["panel"] == panel,
             "panel-specific full quality readback lock")
        for key in ("result", "producer", "consumed"):
            valid_pin(lock[key])
            need(lock[key]["bytes"] <= 128_000_000, "B28 quality originals retain the registered 128MB core cap")
        need(all(valid_hash(lock[key]) for key in
             ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256",
              "external_proof_binding_sha256"))
             and len({lock[key]["id"] for key in ("result", "producer", "consumed")}) == 3
             and len({(lock[key]["run_id"], lock[key]["run_head"]) for key in ("result", "producer", "consumed")}) == 1,
             "three distinct same-run originals and frozen actual quality producer")
    models = doc["model_pt_sha256"]
    need(set(models) == {"B0", "B9", "control", "challenger"}
         and all(valid_hash(v) for v in models.values()) and len(set(models.values())) == 4
         and models["B0"] == B0_PT and models["B9"] == B9_PT and models["control"] == CONTROL_PT
         and models["challenger"] not in (CONTROL_PT, INITIAL_PT, B27_PT),
         "actual B28 challenger against original historical B24 control; no B23 substitution")
    need(set(doc["candidate_files"]) == {STEM + "." + ext for ext in ("pt", "npz", "onnx")}, "three original challenger exports")
    for pin in doc["candidate_files"].values():
        need(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and pin["bytes"] > 10000
             and valid_hash(pin["sha256"]), "full original candidate member bytes")
    need(doc["candidate_files"][STEM + ".pt"]["sha256"] == models["challenger"], "exact selected challenger")
    proof_bindings(doc)
    return doc

def validate_dual_quality(proofs, doc):
    certificate = failure_certification(doc)
    need(valid_hash(doc['model_pt_sha256']['challenger'])
         and doc['model_pt_sha256']['challenger'] not in (CONTROL_PT, INITIAL_PT, B27_PT)
         and doc['model_pt_sha256']['control'] == CONTROL_PT,
         'dual quality belongs to the actual frozen B28 candidate and original B24 control')
    need(set(proofs) == {"primary", "replication"}, "two original panel readbacks")
    for panel, proof in proofs.items():
        need(proof["schema"] == "english-v6-b28-verified-original-quality-v1" and proof["panel"] == panel
             and proof["panel_gate"] == "PASS" and proof["model_pt_sha256"] == doc["model_pt_sha256"]
             and proof["assessment_sha256"] == doc['assessment_sha256']
             and proof["training_design_sha256"] == doc['training_design_sha256']
             and proof["training_assessment_sha256"] == doc['training_assessment_sha256']
             and proof["original_B24_training_assessment_sha256"] == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA
             and proof["pair_pin"] == doc["inputs"]["pair"]
             and proof["original_result_sha256"] == doc["quality_locks"][panel]["original_result_sha256"]
             and proof["external_proof_binding_sha256"] == doc["quality_locks"][panel]["external_proof_binding_sha256"]
             and proof["original_training_report_sha256"] == doc["original_training_report_sha256"]
             and proof['original_lock_sha256'] == sha(json.dumps(doc['quality_locks'][panel],
                 sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
             and proof['original_B27_failure_certification'] == certificate
             and proof['original_B27_failure_report_sha256'] == doc['original_B27_failure_report_sha256']
             and proof["all_original_arrays_verified"] is proof["original_source_admission_recomputed"]
                 is proof["original_producer_git_verified"] is proof["all_original_training_proof_verified"]
                 is proof["complete_original_Q1_failure_verified"] is True
             and all(proof[name] is True for name in ('complete_original_B27_failure_verified',
                  'natural_no_event_terms_recomputed', 'captured_code_matches_pinned_local_reader_root',
                  'old_stops_and_reserves_preserved'))
             and all(type(proof[name]) is int and proof[name] == expected for name, expected in
                  (('natural_objective_returns_verified', 384), ('external_training_originals', 71),
                   ('external_total_originals', 71 if panel == 'primary' else 89), ('derived_B24_originals', 5)))
             and proof['derived_B24_is_new_qualification_source'] is False
             and proof['derived_B24_origin'] == 'same-original-B27-TRAIN_receipts-members'
             and proof['primary_independently_verified'] is (panel == 'replication')
             and proof["replication_open_allowed"] is (panel == "primary")
             and proof["candidate_engineering_validation_allowed"] is (panel == "replication")
             and proof["automatic_next_assessment_allowed"] is proof["promotion_allowed"] is proof["product_released"] is False
             and all(type(proof[name]) is int and proof[name] == 0 for name in
                  ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates'))
             and proof["admission"]["same_job_matched_control"] is proof["admission"]["historical_control_retrained"] is False
             and proof["admission"]["model_comparison"] == MODEL_COMPARISON,
             "genuine original dual PASS of exactly one frozen B28 challenger")
    a, b = (proofs[p]["source_inventory"] for p in ("primary", "replication"))
    for key in ("pcm_sha256", "source_group_id", "canonical_people"):
        need(type(a[key]) is list and type(b[key]) is list and bool(a[key]) and bool(b[key])
             and len(a[key]) == len(set(a[key])) and len(b[key]) == len(set(b[key]))
             and not set(a[key]) & set(b[key]), "all original primary/replication sources independent: " + key)
    need(len(a["pcm_sha256"]) == len(b["pcm_sha256"]) == 120
         and doc["quality_locks"]["primary"]["result"]["run_id"] != doc["quality_locks"]["replication"]["result"]["run_id"],
         "two distinct complete original120 panels")
    return proofs


def quality(inputs, doc, fetch=False):
    """Reopen both complete originals and their explicitly scoped full proof."""
    reader = module(QUALITY_READER)
    need(sha(Path(reader.__file__).read_bytes()) == doc["quality_reader_sha256"],
         "the executing B28 quality reader matches the reviewed digest")
    bindings = proof_bindings(doc)
    proofs = {}
    if fetch:
        (inputs / "proof_roots").mkdir()
    for panel in ("primary", "replication"):
        directory = inputs / "quality" / panel
        proof_root = inputs / "proof_roots" / panel
        if fetch:
            acquire = module("english_v6_b28_original_proof")
            acquire.fetch(proof_root, panel,
                training_reuse=(inputs / "proof_roots/primary/training-origin") if panel == "replication" else None,
                primary_reuse=(inputs / "quality/primary") if panel == "replication" else None)
            if panel == "primary":
                # The replica froze exact PRIMARY API bodies, whose live API
                # representation may later change. Retain those original bytes
                # so the replica can reopen the identical complete18 closure.
                bound = bindings["replication"]
                acquire.fetch_files(directory, bound["primary_quality_files"], bound["primary_quality_transport"])
                proofs[panel] = reader.verify_quality_original(directory, doc["quality_locks"][panel], proof_root)
            else:
                proofs[panel] = reader.fetch_quality_original(directory, doc["quality_locks"][panel], proof_root)
        else:
            proofs[panel] = reader.verify_quality_original(directory, doc["quality_locks"][panel], proof_root)
        need(proofs[panel]["panel_gate"] == "PASS", "complete actual panel PASS before the next panel is opened")
    return validate_dual_quality(proofs, doc)

def producer_paths(tracked, root=ROOT):
    """Retain the actual library and all upstream resources, including dynamic reads."""
    root = Path(root)
    paths = {name for name in tracked if name.startswith(("automation/", ".github/workflows/"))}
    original = json.loads((root / "automation/english_v6_b24_source_training_producer.json").read_bytes())
    need(original["schema"] == "english-v6-b24-complete-git-producer-v1" and len(original["files"]) > 306
         and original["original_b23_dependency_files"] == 306,
         "complete original B23-warm-start TRAIN producer and every inherited historical resource")
    paths |= set(original["files"])
    for panel in ("primary", "replication"):
        name = "automation/english_v6_b28_" + panel + "_producer.json"
        value = json.loads((root / name).read_bytes())
        need(value["frozen"] is True and value["schema"] == "english-v6-b28-" + panel + "-producer-v1"
             and value["upstream_manifests"]["train"]["schema"] == "english-v6-b28-complete-git-producer-v1",
             "original dual-PASS producer and actual new B28 TRAIN must be preserved: " + panel)
        paths.add(name)
        for role in (panel, "source"):
            paths |= set(value[role]["files"])
            paths |= set(value[role].get("resources", {}))
        paths |= set(value["resources"])
        for upstream in value["upstream_manifests"].values():
            paths |= set(upstream["files"])
    required = {WORKFLOW, "automation/english_v6_b28_engineering.py",
                "automation/test_english_v6_b28_engineering.py", "automation/" + SPEC.name,
                "automation/" + TEMPLATE.name, "automation/" + PIPELINE_FILES.name,
                "automation/" + QUALITY_READER + ".py",
                "automation/english_v6_b24r1_engineering.py"}
    need(required <= paths and paths <= set(tracked), "all engineering and original dynamic dependencies are tracked")
    return paths

def original_pair(inputs, doc, proof_root=None):
    profile = module("english_v6_b28_profile")
    proof_bindings(doc)
    proof_root = inputs / "proof_roots/primary" if proof_root is None else Path(proof_root)
    with profile.proof_scope(proof_root, "primary"), original_zip(inputs / "pair.zip", doc["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        primary = module("english_v6_b28_primary_eval")
        need(primary.STEMS == STEMS and primary.MODEL_COMPARISON == MODEL_COMPARISON
             and primary.CONTROL_PT == CONTROL_PT, "same new candidate and immutable historical control contract")
        primary.validate_training_archive(archive, doc, None)
        need(receipt["design_sha256"] == doc['training_design_sha256']
             and receipt["assessment_sha256"] == doc['training_assessment_sha256']
             and receipt['original_B27_failure_certification'] == failure_certification(doc)
             and receipt["challenger"]["files"] == doc["candidate_files"]
             and receipt["same_job_matched_control"] is False
             and receipt["new_control_optimizer_updates"] == 0, "original selected single-challenger TRAIN artifact")
        for arm in ("control", "challenger"):
            stem = STEMS[arm]
            for suffix in ("pt", "npz", "onnx"):
                raw = archive.read(stem + "." + suffix)
                need(receipt[arm]["files"][Path(stem).name + "." + suffix] == dict(bytes=len(raw), sha256=sha(raw)),
                     "exact original B28 challenger or historical B24 control export")
                if suffix == "pt":
                    need(sha(raw) == doc["model_pt_sha256"][arm], "same original acoustic-gate candidate")
        return {suffix: archive.read("challenger/models/" + STEM + "." + suffix) for suffix in ("pt", "npz", "onnx")}

def _execute(inputs, pipeline, output, prepared):
    doc, identity, rows, waves, admission = preflight(inputs, pipeline, prepared)
    permit = json.loads((inputs / "CONSUMPTION.json").read_bytes())
    need(permit == dict(identity=identity, marker_name=MARKER,
         admission_sha256=sha(json.dumps(admission, sort_keys=True, separators=(",", ":")).encode())), "exact uploaded preflight consumption")
    one_execution(identity, True)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new immutable SDK evidence output")
    output.mkdir()
    for name in ("logs", "attempts", "raw_outputs", "models"):
        (output / name).mkdir()
    ledger = Ledger(output / "attempts"); report = dict(schema=SCHEMA, identity=identity, status="STARTED",
        candidate_files=doc["candidate_files"], pipeline_commit=PIPELINE_COMMIT, input_spec_sha256=sha(SPEC.read_bytes()),
        quality=admission["quality"], original_count=ORIGINALS, conditions=list(CONDITIONS), backends=list(BACKENDS),
        records=[], optimizer_updates=0, new_tts_asr_calls=0, independent_quality_claimed=False,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=MODEL_COMPARISON,
        physical_ssc305=False, field_frr=None, continuous_far_per_hour=None, word_end_p95_ms=None,
        product_promotion_allowed=False, product_decoder_settings=None, replay_allowed=False)
    write(output / "STARTED.json", report); write(output / "ADMISSION.json", admission)
    try:
        import numpy as np
        import torch
        import onnxruntime as ort
        need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu" and ort.__version__ == "1.22.1", "pinned numeric runtime")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled = False
        native = pipeline / "research/english_v5_mero"
        net, frontend, packed, values, old = load_candidate(inputs, doc, native, torch, np)
        pack = output / "models/b28-challenger.pack"; pack.write_bytes(packed)
        for suffix, raw in values.items():
            (output / "models" / (STEM + "." + suffix)).write_bytes(raw)
        write(output / "PACK.json", dict(schema=SCHEMA, bytes=len(packed), sha256=sha(packed),
            candidate_files=doc["candidate_files"], pair_pin=doc["inputs"]["pair"], layout="MERO43V1", quality_passes=2))
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(output / "models" / (STEM + ".onnx")), sess_options=options, providers=["CPUExecutionProvider"])
        need([x.name for x in session.get_inputs()] == ["fbank40"]
             and [x.name for x in session.get_outputs()] == ["frame_ctc_logits"], "original ONNX input/output ABI")
        with tempfile.TemporaryDirectory(prefix="b28-sdk-build-") as temp:
            scratch = Path(temp); binaries = build_sdk(pipeline, output, scratch)
            mel, mean, scale = frontend
            for row in rows:
                for condition in CONDITIONS:
                    signal = (b"\0" * 32000 if condition == CONDITIONS[1] else b"") + waves[row["id"]]
                    need(512 <= len(signal) // 2 < 1000000, "bounded engineering clips below frozen offline decoder gap")
                    case = row["id"] + "__" + condition
                    pcm = scratch / (case + ".pcm"); pcm.write_bytes(signal)
                    feature = old.v5.features(signal, {**row, "samples": len(signal) // 2}, torch, np, mel)
                    x = ((feature - mean) / scale).astype(np.float32)[None, :, :]
                    frames = 1 + (len(signal) // 2 - 512) // 160
                    need(x.shape == (1, frames, 40), "same native center-free frame grid")
                    key = case + "__pytorch"; ledger.begin(key, "pytorch")
                    with torch.inference_mode():
                        y = net(torch.from_numpy(x)).detach().cpu().numpy()
                    need(y.shape == (1, frames, 4) and y.dtype == np.dtype("float32") and np.isfinite(y).all(), "full original PyTorch output")
                    reference = np.ascontiguousarray(y[0], dtype="<f4").tobytes()
                    (output / "raw_outputs" / (key + ".f32")).write_bytes(reference); ledger.complete(key)
                    expected = ctc(y[0].argmax(-1).tolist())
                    record = dict(id=row["id"], condition=condition, source_pcm_sha256=sha(waves[row["id"]]),
                        derived_pcm_sha256=sha(signal), frames=frames, pytorch_sha256=sha(reference), **expected, comparisons={})
                    key = case + "__onnxruntime"; ledger.begin(key, "onnxruntime")
                    y = session.run(["frame_ctc_logits"], {"fbank40": x})[0]
                    need(y.shape == (1, frames, 4) and y.dtype == np.dtype("float32"), "full original ONNX output")
                    raw = np.ascontiguousarray(y[0], dtype="<f4").tobytes()
                    (output / "raw_outputs" / (key + ".f32")).write_bytes(raw); ledger.complete(key)
                    record["comparisons"]["onnxruntime"] = dict(sha256=sha(raw), **compare_logits(reference, raw, .001, .0001))
                    for backend, binary in binaries.items():
                        key = case + "__" + backend; ledger.begin(key, backend)
                        logits = output / "raw_outputs" / (key + ".f32")
                        events = output / "raw_outputs" / (key + ".csv")
                        command(binary + [pack, pcm, logits, events], output / "logs" / (key + ".txt"), timeout=130)
                        raw = logits.read_bytes(); event_raw = events.read_bytes(); ledger.complete(key)
                        comparison = compare_logits(reference, raw, .005, .0005)
                        validate_events(event_raw, frames, expected)
                        need("MERO43_C_PCM_STREAM_OK" in (output / "logs" / (key + ".txt")).read_text(), "actual installed C stream returned")
                        record["comparisons"][backend] = dict(sha256=sha(raw), event_csv_sha256=sha(event_raw), **comparison)
                    report["records"].append(record)
        need(len(report["records"]) == 28 and len(ledger.entered) == len(ledger.saved) == MAX_CALLS, "all 28 complete five-backend engineering cases")
        (output / "LICENSE").write_bytes((pipeline / "LICENSE").read_bytes())
        (output / "README.md").write_text("# B28 Hey Mero software SDK candidate\n\nThis immutable candidate passed original primary and replication acoustic gates and saved PT/ONNX/GCC/Clang/ARM-QEMU numerical replay. See RESULT.json, PACK.json and ADMISSION.json for exact provenance.\n\nUse the installed HeyMero43::stream target and authenticate models/b28-challenger.pack before load. The live adapter requires explicit max-gap/cooldown values; this package defines no product decoder settings. The engineering host CLI uses a long gap and zero cooldown only for offline numerical parity.\n\nSSC305 BSP/ABI, physical AFE/callback, board resources, real-human3–5m FRR, continuous natural FAR/h and physical latency remain NOT_RUN. Ubuntu hard-float/QEMU builds are software evidence. Do not identify this artifact as a physically qualified board model.\n")
        report.update(status="PASS_B28_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", pack_sha256=sha(packed), pack_bytes=len(packed),
                      full_frame_tokens_and_event_sample_clocks_equal=True, sdk_installed_and_consumed=True)
    except BaseException as exc:
        report.update(status="STOP_ENGINEERING_NO_RETRY", error_type=type(exc).__name__, error=str(exc)[:2000])
        raise
    finally:
        report.update(entered_backend_calls=len(ledger.entered), saved_backend_outputs=len(ledger.saved),
            consumed_unknown=sorted(set(ledger.entered) - ledger.saved),
            backend_calls={backend: list(ledger.entered.values()).count(backend) for backend in BACKENDS})
        write(output / "RESULT.json", report)
        files = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
                 for p in sorted(output.rglob("*")) if p.is_file()}
        write(output / "MANIFEST.json", dict(schema=SCHEMA, status=report["status"], files=files,
              physical_ssc305=False, product_promotion_allowed=False))

def execute(inputs, pipeline, output, prepared):
    # Closed registrations stop before loading even the proof machinery. The
    # full original TRAIN is verified during the unchanged preflight before
    # scientific imports; every later reuse rehashes its complete byte closure.
    protocol()
    profile = module("english_v6_b28_profile")
    with profile.proof_scope(inputs / "proof_roots/primary", "primary"):
        return _execute(inputs, pipeline, output, prepared)


# Route existing fetch/preflight/producer functions through the B28 authority.
for _name in ("protocol", "source_check", "validate_dual_quality", "quality", "producer_paths", "original_pair",
              "execute", "producer", "verify_producer"):
    setattr(original, _name, globals()[_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fetch", "preflight", "execute"))
    parser.add_argument("--inputs", type=Path); parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--prepared-parent")
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(source_check()))
    else:
        protocol()
        need(args.inputs is not None and args.pipeline is not None, "original inputs and immutable pipeline checkout")
        if args.mode == "fetch":
            fetch(args.inputs, args.pipeline, args.prepared_parent)
        elif args.mode == "preflight":
            preflight(args.inputs, args.pipeline, args.prepared_parent, write_permit=True)
        else:
            need(args.output is not None, "new output required")
            execute(args.inputs, args.pipeline, args.output, args.prepared_parent)
