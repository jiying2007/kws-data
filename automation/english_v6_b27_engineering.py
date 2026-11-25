"""B27 provenance bridge over the exact frozen R1 engineering implementation.

The historical source file and every numeric helper remain unchanged. Only
B27 provenance, report identities and artifact names differ. This module is
inert until both independently authenticated B27 quality PASSes are bound.
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
_definition = importlib.util.spec_from_file_location("b27_frozen_engineering", ORIGINAL)
original = importlib.util.module_from_spec(_definition)
_definition.loader.exec_module(original)
# This private module's functions retain their original implementations. A
# small, explicit set of identity adapters is installed below.
for _name, _value in vars(original).copy().items():
    if not _name.startswith("_"):
        globals()[_name] = _value

SPEC = HERE / "english_v6_b27_engineering_inputs.json"
TEMPLATE = HERE / "english_v6_b27_engineering_workflow_template.yml"
SCHEMA = "english-v6-b27-dual-pass-software-sdk-v1"
QUALITY_READER = "english_v6_b27_quality_readback"
WORKFLOW = ".github/workflows/english-v6-b27-engineering-once.yml"
BRANCH = "research/en-v6-b27-engineering-once-20261125"
PARENT_TOKEN = "__B27_REVIEWED_ENGINEERING_PARENT__"
MARKER = "english-v6-b27-engineering-consumed"
ASSESSMENT_SHA = "8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee"
TRAINING_ASSESSMENT_SHA = ASSESSMENT_SHA
ORIGINAL_B24_TRAINING_ASSESSMENT_SHA = "3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad"
DESIGN_SHA = "956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf"
CONTROL_PT = "f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4"
INITIAL_PT = "784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f"
CANDIDATE_PT = None  # Only the actual B27 input spec and full TRAIN proof bind it.
MODEL_COMPARISON = "B27_new_challenger_against_immutable_historical_B24_challenger"
STEM = "mero43-b27-challenger192"
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
         "complete actual B27 engineering producer fits its unchanged 128MB archive boundary")
    return dict(expanded_bytes=expanded, members=len(names), conservative_zip_bytes=envelope,
                maximum_bytes=maximum, files_verified=len(value["files"]))


def producer(inputs, pipeline, identity):
    _frozen_producer(inputs, pipeline, identity)
    evidence_capacity(inputs / "producer")


def verify_producer(inputs, pipeline, identity):
    _frozen_verify_producer(inputs, pipeline, identity)
    evidence_capacity(inputs / "producer")


def source_check():
    result = _frozen_source_check()
    doc = json.loads(SPEC.read_bytes())
    need(doc["B24Q1_terminal_NO_PROMOTION_preserved"] is True
         and doc["old_B24Q1_replication_opened"] is False
         and doc["original_B24Q1_failure_report_sha256"] == Q1_FAILURE_REPORT_SHA,
         "the completed original Q1 failure and its closed follow-up stages remain terminal")
    if doc["enabled"] is False:
        need(doc["original_training_report_sha256"] is None,
             "no invented completed B27 TRAIN identity")
    return result


def proof_bindings(doc):
    """Bind both quality locks to the same complete actual B27 TRAIN proof."""
    profile = module("english_v6_b27_profile")
    profile.profile()
    bound = {panel: profile.binding(panel) for panel in ("primary", "replication")}
    for panel, value in bound.items():
        path = HERE / ("english_v6_b27_" + panel + "_proof_binding.json")
        need(sha(path.read_bytes()) == doc["quality_locks"][panel]["external_proof_binding_sha256"]
             and value["training_report_sha256"] == doc["original_training_report_sha256"],
             "same complete externally retained original proof binding: " + panel)
    first, second = bound["primary"], bound["replication"]
    need(all(first[key] == second[key] for key in
             ("files", "training_lock", "training_reader_sha256", "training_report_sha256"))
         and second["primary_quality_lock"] == doc["quality_locks"]["primary"]
         and second["primary_quality_reader_sha256"] == doc["quality_reader_sha256"],
         "both panels reauthenticate the same full TRAIN and replication requires this exact PRIMARY")
    training = first["training_lock"]["training"]
    pin = training["artifacts"]["models"]
    need(training["model_pt_sha256"] == doc["model_pt_sha256"]
         and doc["inputs"]["pair"] == dict(id=pin["id"], bytes=pin["bytes"], sha256=pin["sha256"],
             run_id=training["run_id"], run_head=training["run_head"]),
         "candidate hashes and pair archive derive only from the original completed B27 TRAIN")
    return bound


def protocol(value=None):
    # A binder may validate a prospective document without writing SPEC. Every
    # actual fetch/preflight/execute call reads the committed default SPEC.
    doc = json.loads(SPEC.read_bytes()) if value is None else value
    need(doc.get("enabled") is True, "INERT: original B27 primary and replication PASSes are not bound")
    need(doc["schema"] == SCHEMA and doc["candidate_role"] == "challenger"
         and doc["assessment_sha256"] == ASSESSMENT_SHA and doc["training_design_sha256"] == DESIGN_SHA
         and doc["training_assessment_sha256"] == TRAINING_ASSESSMENT_SHA
         and doc["original_B24_training_assessment_sha256"] == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA
         and doc["evaluation_only"] is doc["B24_original_source_stop_preserved"] is doc["B24R1_original_source_stop_preserved"] is True
         and doc["B24Q1_terminal_NO_PROMOTION_preserved"] is True
         and doc["old_B24Q1_replication_opened"] is False
         and doc["original_B24Q1_failure_report_sha256"] == Q1_FAILURE_REPORT_SHA
         and valid_hash(doc["original_training_report_sha256"])
         and doc["old_B24R1_sources_used_for_scoring"] is doc["old_B24R1_replication_opened"] is False
         and type(doc["new_optimizer_updates"]) is int and doc["new_optimizer_updates"] == 0
         and doc["old_B24_sources_used_for_scoring"] is doc["old_B24_replication_opened"] is doc["automatic_R2_allowed"] is False
         and doc["pipeline_commit"] == PIPELINE_COMMIT and doc["quality_reader"] == QUALITY_READER
         and doc["prepared_parent"] is None and doc["parent_authority"] == "reviewed_workflow"
         and doc["original_count"] == ORIGINALS and doc["conditions"] == list(CONDITIONS)
         and doc["maximum_backend_calls"] == MAX_CALLS
         and doc["onnx_max_abs"] == .001 and doc["onnx_mean_abs"] == .0001
         and doc["c_max_abs"] == .005 and doc["c_mean_abs"] == .0005
         and doc["same_job_matched_control"] is doc["historical_control_retrained"] is False
         and doc["model_comparison"] == MODEL_COMPARISON
         and all(doc[key] is False for key in ("training_allowed", "threshold_search", "new_source_allowed",
             "cal_locked_access", "physical_ssc305", "product_promotion_allowed", "product_decoder_settings_defined")),
         "same frozen B27 weights and numerical thresholds; software-only output")
    need(sha((HERE / "english_v6_b27_panels/assessment.json").read_bytes()) == ASSESSMENT_SHA
         and sha((HERE / "english_v6_b24_panels/assessment.json").read_bytes()) == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA
         and sha((HERE / "english_v6_b27_natural_weight_design.json").read_bytes()) == DESIGN_SHA,
         "preoptimizer B27 TRAIN and fresh assessment, with original B24 ancestry separate")
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
             and lock["schema"] == "english-v6-b27-original-quality-lock-v1" and lock["panel"] == panel,
             "panel-specific full quality readback lock")
        for key in ("result", "producer", "consumed"):
            valid_pin(lock[key])
            need(lock[key]["bytes"] <= 128_000_000, "B27 quality originals retain the registered 128MB core cap")
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
         and models["challenger"] not in (CONTROL_PT, INITIAL_PT),
         "actual B27 challenger against original historical B24 control; no B23 substitution")
    need(set(doc["candidate_files"]) == {STEM + "." + ext for ext in ("pt", "npz", "onnx")}, "three original challenger exports")
    for pin in doc["candidate_files"].values():
        need(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and pin["bytes"] > 10000
             and valid_hash(pin["sha256"]), "full original candidate member bytes")
    need(doc["candidate_files"][STEM + ".pt"]["sha256"] == models["challenger"], "exact selected challenger")
    proof_bindings(doc)
    return doc

def validate_dual_quality(proofs, doc):
    need(valid_hash(doc['model_pt_sha256']['challenger'])
         and doc['model_pt_sha256']['challenger'] not in (CONTROL_PT, INITIAL_PT)
         and doc['model_pt_sha256']['control'] == CONTROL_PT,
         'dual quality belongs to the actual frozen B27 candidate and original B24 control')
    need(set(proofs) == {"primary", "replication"}, "two original panel readbacks")
    for panel, proof in proofs.items():
        need(proof["schema"] == "english-v6-b27-verified-original-quality-v1" and proof["panel"] == panel
             and proof["panel_gate"] == "PASS" and proof["model_pt_sha256"] == doc["model_pt_sha256"]
             and proof["assessment_sha256"] == ASSESSMENT_SHA and proof["training_design_sha256"] == DESIGN_SHA
             and proof["training_assessment_sha256"] == TRAINING_ASSESSMENT_SHA
             and proof["original_B24_training_assessment_sha256"] == ORIGINAL_B24_TRAINING_ASSESSMENT_SHA
             and proof["pair_pin"] == doc["inputs"]["pair"]
             and proof["original_result_sha256"] == doc["quality_locks"][panel]["original_result_sha256"]
             and proof["external_proof_binding_sha256"] == doc["quality_locks"][panel]["external_proof_binding_sha256"]
             and proof["original_training_report_sha256"] == doc["original_training_report_sha256"]
             and proof["original_B24Q1_failure_report_sha256"] == Q1_FAILURE_REPORT_SHA
             and proof["all_original_arrays_verified"] is proof["original_source_admission_recomputed"]
                 is proof["original_producer_git_verified"] is proof["all_original_training_proof_verified"]
                 is proof["complete_original_Q1_failure_verified"] is True
             and proof["replication_open_allowed"] is (panel == "primary")
             and proof["candidate_engineering_validation_allowed"] is (panel == "replication")
             and proof["automatic_next_assessment_allowed"] is proof["promotion_allowed"] is proof["product_released"] is False
             and proof["new_model_calls"] == proof["new_source_calls"] == proof["new_observer_calls"] == 0
             and proof["admission"]["same_job_matched_control"] is proof["admission"]["historical_control_retrained"] is False
             and proof["admission"]["model_comparison"] == MODEL_COMPARISON,
             "genuine original dual PASS of exactly one frozen B27 challenger")
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
         "the executing B27 quality reader matches the reviewed digest")
    bindings = proof_bindings(doc)
    proofs = {}
    if fetch:
        (inputs / "proof_roots").mkdir()
    for panel in ("primary", "replication"):
        directory = inputs / "quality" / panel
        proof_root = inputs / "proof_roots" / panel
        if fetch:
            acquire = module("english_v6_b27_original_proof")
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
        name = "automation/english_v6_b27_" + panel + "_producer.json"
        value = json.loads((root / name).read_bytes())
        need(value["frozen"] is True and value["schema"] == "english-v6-b27-" + panel + "-producer-v1"
             and value["upstream_manifests"]["train"]["schema"] == "english-v6-b27-complete-git-producer-v1",
             "original dual-PASS producer and actual new B27 TRAIN must be preserved: " + panel)
        paths.add(name)
        for role in (panel, "source"):
            paths |= set(value[role]["files"])
            paths |= set(value[role].get("resources", {}))
        paths |= set(value["resources"])
        for upstream in value["upstream_manifests"].values():
            paths |= set(upstream["files"])
    required = {WORKFLOW, "automation/english_v6_b27_engineering.py",
                "automation/test_english_v6_b27_engineering.py", "automation/" + SPEC.name,
                "automation/" + TEMPLATE.name, "automation/" + PIPELINE_FILES.name,
                "automation/" + QUALITY_READER + ".py",
                "automation/english_v6_b24r1_engineering.py"}
    need(required <= paths and paths <= set(tracked), "all engineering and original dynamic dependencies are tracked")
    return paths

def original_pair(inputs, doc, proof_root=None):
    profile = module("english_v6_b27_profile")
    proof_bindings(doc)
    proof_root = inputs / "proof_roots/primary" if proof_root is None else Path(proof_root)
    with profile.proof_scope(proof_root, "primary"), original_zip(inputs / "pair.zip", doc["inputs"]["pair"]) as archive:
        receipt = json.loads(archive.read("RESULT.json"))
        primary = module("english_v6_b27_primary_eval")
        need(primary.STEMS == STEMS and primary.MODEL_COMPARISON == MODEL_COMPARISON
             and primary.CONTROL_PT == CONTROL_PT, "same new candidate and immutable historical control contract")
        primary.validate_training_archive(archive, doc, None)
        need(receipt["design_sha256"] == DESIGN_SHA and receipt["assessment_sha256"] == TRAINING_ASSESSMENT_SHA
             and receipt["challenger"]["files"] == doc["candidate_files"]
             and receipt["same_job_matched_control"] is False
             and receipt["new_control_optimizer_updates"] == 0, "original selected single-challenger TRAIN artifact")
        for arm in ("control", "challenger"):
            stem = STEMS[arm]
            for suffix in ("pt", "npz", "onnx"):
                raw = archive.read(stem + "." + suffix)
                need(receipt[arm]["files"][Path(stem).name + "." + suffix] == dict(bytes=len(raw), sha256=sha(raw)),
                     "exact original B27 challenger or historical B24 control export")
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
        pack = output / "models/b27-challenger.pack"; pack.write_bytes(packed)
        for suffix, raw in values.items():
            (output / "models" / (STEM + "." + suffix)).write_bytes(raw)
        write(output / "PACK.json", dict(schema=SCHEMA, bytes=len(packed), sha256=sha(packed),
            candidate_files=doc["candidate_files"], pair_pin=doc["inputs"]["pair"], layout="MERO43V1", quality_passes=2))
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(output / "models" / (STEM + ".onnx")), sess_options=options, providers=["CPUExecutionProvider"])
        need([x.name for x in session.get_inputs()] == ["fbank40"]
             and [x.name for x in session.get_outputs()] == ["frame_ctc_logits"], "original ONNX input/output ABI")
        with tempfile.TemporaryDirectory(prefix="b27-sdk-build-") as temp:
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
        (output / "README.md").write_text("# B27 Hey Mero software SDK candidate\n\nThis immutable candidate passed original primary and replication acoustic gates and saved PT/ONNX/GCC/Clang/ARM-QEMU numerical replay. See RESULT.json, PACK.json and ADMISSION.json for exact provenance.\n\nUse the installed HeyMero43::stream target and authenticate models/b27-challenger.pack before load. The live adapter requires explicit max-gap/cooldown values; this package defines no product decoder settings. The engineering host CLI uses a long gap and zero cooldown only for offline numerical parity.\n\nSSC305 BSP/ABI, physical AFE/callback, board resources, real-human3–5m FRR, continuous natural FAR/h and physical latency remain NOT_RUN. Ubuntu hard-float/QEMU builds are software evidence. Do not identify this artifact as a physically qualified board model.\n")
        report.update(status="PASS_B27_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", pack_sha256=sha(packed), pack_bytes=len(packed),
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
    profile = module("english_v6_b27_profile")
    with profile.proof_scope(inputs / "proof_roots/primary", "primary"):
        return _execute(inputs, pipeline, output, prepared)


# Route existing fetch/preflight/producer functions through the B27 authority.
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
