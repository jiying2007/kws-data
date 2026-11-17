#!/usr/bin/env python3
"""Independent stdlib audit of the final fixed, cache-only dense-CE arm.

The authenticated temporal auditor supplies binary readers and saved-logit
event replay. This file verifies the new source, three immutable inputs,
unchanged caches/draws, zero frontend execution and the unchanged research gate.
No model, gradient, matrix product, tensor framework or network is executed.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

FROZEN_AUDITOR_SHA = "c0b91dc1b8f22b3daf600a2acb7bb1958ceb829aed3a1893b622cab72e405acd"
FROZEN_AUDITOR_BYTES = 68982
TEMPORAL_PIN = {"bytes": 17980402, "sha256": "6a1e38febf22b76a7cf2227362e51f783d5ad524b2d81b5677c22edc6dccc631"}
TEMPORAL_RUN = "44031666420"
TEMPORAL_COMMIT = "b4982d7a2f39c4b2647a86cbf96d3701a1123b6a"
EXPECTED_COUNTS = {"new_embedding_calls_started": 0, "new_embedding_calls_completed": 0,
    "onnx_forward_started": 0, "onnx_forward_completed": 0,
    "head_case_passes_started": 264, "head_case_passes_completed": 264, "head_ready_outputs": 11847,
    "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
    "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576,
    "new_training_frontend_calls": 0, "new_tts_asr_phone_calls": 0, "retry_count": 0}


def load_frozen_auditor():
    candidates = []
    for parent in Path(__file__).resolve().parents:
        for relative in ("english_temporal/audit/audit_temporal.py",
                         "english_temporal_quality_20261117/audit/audit_temporal.py"):
            candidates.append(parent / relative)
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise ValueError("required frozen temporal auditor is absent from the sibling research directory")
    raw = source.read_bytes()
    if len(raw) != FROZEN_AUDITOR_BYTES or hashlib.sha256(raw).hexdigest() != FROZEN_AUDITOR_SHA:
        raise ValueError("frozen temporal auditor source SHA/bytes changed")
    spec = importlib.util.spec_from_file_location("_frozen_temporal_artifact_auditor_v1", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_frozen_auditor()
require, ident, digest = legacy.require, legacy.ident, legacy.digest
FALSE_FLAGS, SHAPES = legacy.FALSE_FLAGS, legacy.SHAPES


def source_only_dense_contract(raw):
    """Inspect fixed AST expressions without importing the producer module."""
    tree = ast.parse(raw.decode("utf-8"))
    worker = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"), None)
    require(worker is not None, "dense worker source is present")
    assignments = {}
    for node in ast.walk(worker):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
    expected = {
        "windows": 'np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)',
        "normalized": '((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)',
        "model": 'torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))',
        "optimizer": 'torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)',
        "logp": 'torch.log_softmax(model(prepared[key]), dim=-1)',
        "loss": 'torch.stack(terms).mean()',
    }
    for name, expression in expected.items():
        nodes = assignments.get(name, [])
        wanted = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
        require(len(nodes) == 1 and ast.dump(nodes[0], include_attributes=False) == wanted,
                "unchanged training expression: " + name)
    branches = [node for node in ast.walk(worker) if isinstance(node, ast.If)
                and ast.dump(node.test, include_attributes=False) ==
                ast.dump(ast.parse("label == 0", mode="eval").body, include_attributes=False)]
    require(len(branches) == 1, "one fixed negative/positive loss branch")
    branch = branches[0]
    for nodes, text in ((branch.body, "terms.append(-logp[:, 0].mean())"),
                        (branch.orelse, "terms.append(-logp[:, label].mean())")):
        require([ast.dump(n, include_attributes=False) for n in nodes] ==
                [ast.dump(n, include_attributes=False) for n in ast.parse(text).body],
                "exact per-bag mean CE; no sum, loss blend or temporal pooling")
    calls = [ast.unparse(node) for node in ast.walk(worker) if isinstance(node, ast.Call)]
    for call in ("torch.set_num_threads(1)", "torch.manual_seed(20261117)",
                 "torch.use_deterministic_algorithms(True)", "optimizer.zero_grad(set_to_none=True)",
                 "loss.backward()", "optimizer.step()",
                 "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)"):
        require(calls.count(call) == 1, "exact original one-time training operation: " + call)
    loops = [n for n in ast.walk(worker) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "step"]
    require(len(loops) == 1 and ast.unparse(loops[0].iter) == "range(96)", "one fixed 96-update training loop")
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    require(not set(imports) & {"onnx", "onnxruntime", "causal_embedding", "soundfile"},
            "cache-only worker cannot import a frontend or audio decoder")
    require(not any("FrozenOnnxEmbedding(" in call or "InferenceSession(" in call for call in calls),
            "no frontend construction in cache-only worker")
    require(not any(call.startswith("common.worker(") or call.startswith("common.execute(") for call in calls),
            "consumed temporal numerical worker cannot be re-executed")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "english_temporal_quality_once":
            require(not {item.name for item in node.names} & {"worker", "execute"},
                    "cannot import consumed numerical entrypoints as aliases")
    return {"source_sha256": digest(raw), "dense_positive_mean_ce_checked": True,
            "same_background_loss_data_geometry_optimizer_checked": True,
            "no_frontend_or_decoder_imports": True, "static_inspection_only": True,
            "local_numerical_loss_or_gradient_execution": False}


def verify_dense_phases(phases, frozen, final, candidate_sha):
    require(final == EXPECTED_COUNTS and all(type(v) is int for v in final.values()),
            "exact final one-head/96-step/zero-frontend counters")
    names = ["WORKER_STARTED", "TRAINING_STARTED", "CANDIDATE_FROZEN",
             "EVALUATION_STARTED", "EVALUATION_COMPLETED", "COMPLETED"]
    require([p["phase"] for p in phases] == names
            and [p["phase_number"] for p in phases] == list(range(1, 7)), "closed six-phase cache-only ledger")
    zero = dict.fromkeys(final, 0)
    trained = {**zero, "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
               "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576}
    for index, item in enumerate(phases):
        expected = zero if index < 2 else trained if index < 4 else final
        require(item["progress"] == expected and all(type(v) is int for v in item["progress"].values()),
                "exact phase counters; no hidden frontend or extra head passes")
        require(item["candidate_sha256"] == (None if index < 2 else candidate_sha), "one frozen candidate through evaluation")
    for key in ("new_embedding_calls_started", "new_embedding_calls_completed", "onnx_forward_started",
                "onnx_forward_completed", "new_training_frontend_calls", "new_tts_asr_phone_calls", "retry_count"):
        require(final[key] == 0, "zero new frontend/decoder/retry execution: " + key)
    require(frozen["phase_number"] == 3 and frozen["progress"] == trained
            and frozen["candidate"]["sha256"] == candidate_sha
            and frozen["snr5_model_calls_before_freeze"] == 0
            and frozen["all_future_evaluation_uses_this_frozen_head"] is True,
            "candidate frozen before every cache-only evaluation")
    return {"phases": 6, "candidate_frozen_phase": 3, "evaluation_start_phase": 4,
            "all_frontend_counters_zero": True, "source_and_receipt_order_not_external_runtime_attestation": True}


def verify_copied_members(quality, copied, sources):
    for destination, info in copied.items():
        require(info["source"] in sources, "known copied input source")
        raw = sources[info["source"]].read(info["member"])
        require(quality.read(destination) == raw and ident(raw) == {k: info[k] for k in ("bytes", "sha256")},
                "authenticated member copied byte-for-byte: " + destination)
    return len(copied)


def authenticated_inputs(training, incumbent, temporal):
    legacy.authenticate_sources(training, incumbent)
    require(ident(temporal.raw) == TEMPORAL_PIN, "third immutable cache-source ZIP SHA/bytes")
    result = temporal.json("evidence/RESULT.json")
    require(result["status"] == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH"
            and str(result["run_id"]) == TEMPORAL_RUN and result["head_sha"] == TEMPORAL_COMMIT,
            "completed cache-source run identity")
    require(result["research_retention"]["passed"] is False, "failed temporal head remains rejected")
    require(temporal.read("cache/clean-original.npz") == training.read("actual-embedding-streams.npz")
            and temporal.read("cache/mixed-snr10.npz") == incumbent.read("cache/mixed-embeddings.npz"),
            "all three sources agree on reused old feature bytes")
    roster, clean_clocks, bags = legacy.audit_clean_cache(training, training.read("actual-embedding-streams.npz"))
    snr10_rows = incumbent.json("cache/mixed-roster.json")
    snr10_clocks, snr10_count = legacy.audit_condition_cache(incumbent.read("cache/mixed-embeddings.npz"),
                                                          snr10_rows, clean_clocks, "snr10")
    snr5_rows = temporal.json("cache/snr5-roster.json")
    snr5_clocks, snr5_count = legacy.audit_condition_cache(temporal.read("cache/snr5-embeddings.npz"),
                                                        snr5_rows, clean_clocks, "snr5")
    receipts = temporal.json("evidence/SNR5_RECEIPTS.json")
    receipt_map = legacy.unique_rows(receipts, "77 cached 5 dB receipts")
    source_map = legacy.unique_rows(roster, "original clean roster")
    require(len(snr10_rows) == 76 and snr10_count == 2804 and len(snr5_rows) == 71 and snr5_count == 2649
            and len(receipts) == 77 and {r["id"] for r in snr5_rows} ==
            {r["id"] for r in receipts if r["status"] == "MIXED"}, "exact already-consumed eligible conditions")
    for row in snr5_rows:
        require(all(row[k] == source_map[row["id"]][k] for k in source_map[row["id"]])
                and row["mix"] == receipt_map[row["id"]]
                and row["mix"]["condition_training_allowed"] is False,
                "same original clip/class/role and evaluation-only cached condition")
    require(set(receipt_map) == set(training.json("SELECTION.json")["eligible_cell_ids"]),
            "all planned original IDs retained, including six fixed exclusions")
    return {"clean": roster, "snr10": snr10_rows, "snr5": snr5_rows}, \
           {"clean": clean_clocks, "snr10": snr10_clocks, "snr5": snr5_clocks}, bags


def input_preflight(training, incumbent, temporal):
    rows, clocks, bags = authenticated_inputs(training, incumbent, temporal)
    old_training = incumbent.json("evidence/TRAINING.json")
    draws = legacy.verify_draws(old_training["draws"], old_training["draws"], rows["clean"], rows["snr10"])
    return {"schema": "independent-dense-cache-input-preflight-v1", "status": "CACHE_ONLY_INPUTS_VERIFIED",
            "source_training_zip": ident(training.raw), "source_incumbent_zip": ident(incumbent.raw),
            "source_temporal_cache_zip": ident(temporal.raw), "frozen_auditor_sha256": FROZEN_AUDITOR_SHA,
            "cached_streams": {k: len(v) for k, v in rows.items()},
            "cached_embedding_rows": {k: sum(len(t) for t in group.values()) for k, group in clocks.items()},
            "new_head_case_passes": sum(len(v) for v in rows.values()),
            "new_head_ready_outputs": sum(len(t) - 15 for group in clocks.values() for t in group.values()),
            "training_draws": draws, "positive_supported_bags": bags,
            "new_model_calls_by_this_auditor": 0, "new_audio_mixes_by_this_auditor": 0,
            "planned_new_frontend_calls": 0, "same_source_condition_is_not_independent_audio": True,
            "development_conditions_already_exposed": True, "word_end_latency": "NOT_MEASURED",
            **dict.fromkeys(FALSE_FLAGS, False)}


def verify_dense_plan(plan):
    fixed = {"schema": "english-dense-quality-once-v1", "main_before": TEMPORAL_COMMIT,
        "snr_db": 5.0, "planned_new_conditions": 0, "planned_maximum_embedding_calls": 0,
        "maximum_embedding_calls": 0, "maximum_onnx_forwards": 0,
        "maximum_head_case_passes": 264, "expected_head_ready_outputs": 11847,
        "maximum_optimizer_updates": 96, "maximum_training_bag_forwards": 576,
        "maximum_execution_seconds": 900, "retry_count": 0, "new_tts_asr_phone_calls": 0,
        "new_training_frontend_calls": 0, "seed": 20261117, "cpu_threads": 1,
        "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
        "threshold": 0.8, "confirm_embeddings": 3, "cooldown_embeddings": 13,
        "new_training_arms": 1, "optimizer": "FRESH_ADAMW", "warmstart": "ORIGINAL_BASELINE",
        "normalization": "FROZEN_ORIGINAL_TRAIN_ONLY", "positive_loss": "MEAN_WINDOW_TARGET_CROSS_ENTROPY",
        "background_loss": "MEAN_WINDOW_BACKGROUND_CROSS_ENTROPY", "bag_aggregation": "SIX_EQUAL_CONDITION_CLASS_TERMS",
        "training_conditions": ["clean", "snr10"], "evaluation_conditions": ["clean", "snr10", "snr5"],
        "draws": "EXACT_PREVIOUS_96_BY_6_DRAW_SEQUENCE",
        "positive_geometry": "UNCHANGED_ORIGINAL_MEL_SUPPORTED_HEAD_INSTANCE_ENDS",
        "development_tuning": False, "checkpoint_selection": False, "incumbent_retraining": False,
        "old_frontend_recomputation": False, "snr5_training_allowed": False,
        "snr5_evaluation_after_candidate_frozen": True, "planned_reused_snr5_conditions": 77,
        "eligible_reused_snr5_conditions": 71, "reused_snr5_embedding_rows": 2649,
        "evaluation_frontend": "REUSE_FROZEN_CACHES_FOR_ALL_THREE_CONDITIONS",
        "snr5_previous_evaluation_exposed": True, "failed_temporal_head_used_as_warmstart": False,
        "last_additional_objective_arm": True, "subsequent_same_pool_loss_seed_threshold_search_allowed": False}
    for key, value in fixed.items():
        require(type(plan.get(key)) is type(value) and plan[key] == value, "fixed final dense plan: " + key)
    require("positive_min_operator" not in plan, "no temporal minimum operator in dense objective")
    expected_gate = {"clean_no_previously_correct_case_regression": True,
        "snr10_no_previously_correct_case_regression": True, "snr10_minimum_rescued_development_misses": 1,
        "all_conditions_no_per_case_prefix_wrong_extra_or_negative_increase": True,
        "snr5_no_previously_correct_case_regression": True, "snr5_minimum_development_positive_cases_per_keyword": 1,
        "reference": "PREVIOUS_AUGMENTED_96_UPDATE_HEAD", "formal_promotion_allowed": False}
    require(plan["research_retention"] == expected_gate and all(plan.get(k) is False for k in FALSE_FLAGS),
            "same strict gate and no formal admission")
    for label, pin in (("training", legacy.TRAINING_PIN), ("incumbent", legacy.INCUMBENT_PIN), ("temporal", TEMPORAL_PIN)):
        require({k: plan[label + "_source"][k] for k in pin} == pin, "fixed source plan binding: " + label)
    require(plan["common_worker_sha256"] == "15389e1101871352e6e7477c893f1dadf019fd57b572fa1388ec53d1dd8b0988",
            "unchanged common evidence/guard utilities")
    legacy.reject_formal_claims(plan)


def artifact_boundary(quality, plan, result, expected_run, expected_head):
    verify_dense_plan(plan)
    require(result["status"] == "COMPLETED_WEAK_DENSE_QUALITY_RESEARCH" and result["schema"] == plan["schema"]
            and result["run_attempt"] == 1, "completed one-attempt final dense research")
    require(expected_run is None or str(result["run_id"]) == str(expected_run), "expected public run")
    require(expected_head is None or result["head_sha"] == expected_head, "expected public source commit")
    require(all(result.get(k) is False for k in FALSE_FLAGS), "result does not grant product admission")
    require("evidence/FAILED_NO_RETRY.json" not in quality.names, "completed evidence has no failure receipt")
    for name in quality.names:
        if name.endswith(".json"):
            legacy.reject_formal_claims(quality.json(name), name)
    manifest = quality.json("evidence/WORKER_MANIFEST.json")["files"]
    for name, pin in manifest.items():
        require(name in quality.names and ident(quality.read(name)) == pin, "manifest identity: " + name)
    later = {"evidence/WORKER_MANIFEST.json", "evidence/model-worker.exit.json",
             "evidence/PARENT_RESULT.json", "evidence/SUPERVISOR_EXIT.json"}
    require(quality.names - set(manifest) == later, "closed saved artifact and known post-worker receipts")
    supervisor = quality.json("evidence/SUPERVISOR_EXIT.json")
    require(supervisor["retry_count"] == 0 and supervisor["numeric_attempt"] == 1
            and supervisor["maximum_seconds"] == 900 and 0 <= supervisor["wall_seconds"] <= 900,
            "single supervised finite execution")
    exit_record = quality.json("evidence/model-worker.exit.json")
    require(exit_record["returncode"] == 0 and exit_record["error"] is None
            and exit_record["group_cleanup_checked_after_leader_exit"] is True, "clean completed worker process")
    parent = quality.json("evidence/PARENT_RESULT.json")
    require(parent["status"] == "COMPLETED_ONE_SUPERVISED_WORKER" and parent["retry_count"] == 0
            and parent["numeric_attempt"] == 1
            and parent["worker_result"] == ident(quality.read("evidence/RESULT.json")), "owning parent binds result")
    return len(manifest), supervisor["wall_seconds"]


def audit(quality, training, incumbent, temporal, expected_run=None, expected_head=None):
    rows, clocks, positive_bags = authenticated_inputs(training, incumbent, temporal)
    plan, result = quality.json("evidence/PLAN.json"), quality.json("evidence/RESULT.json")
    manifest_count, wall_seconds = artifact_boundary(quality, plan, result, expected_run, expected_head)
    inputs = quality.json("evidence/INPUTS.json")
    require(inputs["old_embedding_rows_reused"] == {"clean": 10354, "snr10": 2804, "snr5": 2649}
            and inputs["old_frontend_recomputation"] is False
            and inputs["new_training_frontend_calls"] == inputs["new_evaluation_frontend_calls"] == 0
            and inputs["failed_temporal_head_used_as_warmstart"] is False, "cache-only reuse and original-baseline starting point")
    sources = {"training": training, "incumbent": incumbent, "temporal": temporal}
    for label, archive in sources.items():
        require(inputs["actual_" + label] == ident(archive.raw)
                and inputs[label + "_source"] == plan[label + "_source"] == result[label + "_source"],
                "new run binds immutable source: " + label)
    copied = inputs["copied_members"]
    verify_copied_members(quality, copied, sources)
    required_copies = {"sources/" + n: (training, n) for n in (
        "STREAM_ROSTER.json", "FEATURE_EVIDENCE.json", "FEATURE_PREFLIGHT.json", "PER_CASE_EVENTS.json",
        "LIBRISPEECH_SOURCES.json", "SELECTION.json")}
    required_copies.update({"cache/clean-original.npz": (training, "actual-embedding-streams.npz"),
        "models/baseline-head.npz": (training, "train/causal-head.npz"),
        "models/aug96-control-head.npz": (incumbent, "models/candidate-head.npz"),
        "cache/mixed-snr10.npz": (incumbent, "cache/mixed-embeddings.npz"),
        "cache/snr10-roster.json": (incumbent, "cache/mixed-roster.json"),
        "sources/PREVIOUS_TRAINING.json": (incumbent, "evidence/TRAINING.json"),
        "cache/snr5-embeddings.npz": (temporal, "cache/snr5-embeddings.npz"),
        "cache/snr5-roster.json": (temporal, "cache/snr5-roster.json"),
        "evidence/SNR5_SUMMARY.json": (temporal, "evidence/SNR5_SUMMARY.json"),
        "evidence/SNR5_RECEIPTS.json": (temporal, "evidence/SNR5_RECEIPTS.json"),
        "sources/PREVIOUS_TEMPORAL_RESULT.json": (temporal, "evidence/RESULT.json")})
    bundle = "english-snapshot160-embedding-research-bundle/"
    for name in ("portable_head.py", "weak_embedding_contract.py"):
        required_copies["sources/" + name] = (training, bundle + "runtime/" + name)
    for name in incumbent.names:
        if name.startswith("predictions/"):
            required_copies["predictions/reused/" + Path(name).name] = (incumbent, name)
    for prefix in ("baseline-snr5", "aug96-snr5"):
        for suffix in ("-frames.jsonl", "-events.json"):
            name = "predictions/" + prefix + suffix
            required_copies["predictions/reused/" + prefix + suffix] = (temporal, name)
    for destination, (source, member) in required_copies.items():
        require(destination in copied and quality.read(destination) == source.read(member),
                "mandatory pinned reuse: " + destination)
    require(not any(name.startswith("audio/") or name.endswith(".onnx") for name in quality.names),
            "cache-only artifact has no new audio or frontend model asset")
    for key, name in (("worker_sha256", "english_dense_quality_once.py"),
                      ("common_worker_sha256", "english_temporal_quality_once.py"),
                      ("workflow_sha256", "english-dense-quality-once.yml")):
        raw = quality.read("sources/" + name)
        require(digest(raw) == plan[key] and inputs["source_contract"][key]["sha256"] == plan[key]
                and inputs["source_contract"][key]["bytes"] == len(raw), "executed exact source contract")
    require(quality.read("sources/english_temporal_quality_once.py") == temporal.read("sources/english_temporal_quality_once.py"),
            "common module is the frozen previously executed source; old worker is not re-executed")
    source_review = source_only_dense_contract(quality.read("sources/english_dense_quality_once.py"))
    runtime, old_runtime = quality.json("evidence/RUNTIME.json"), incumbent.json("evidence/RUNTIME.json")
    require(all(runtime[k] == old_runtime[k] for k in ("numpy", "torch", "numeric_threads"))
            and runtime["embedding_runtime_loaded"] is False and runtime["onnx_forward_execution"] == "NOT_RUN"
            and runtime["soundfile_or_corpus_decoder_used"] is False, "same head dependencies and no frontend/decoder runtime")
    training_receipt, old_training = quality.json("evidence/TRAINING.json"), incumbent.json("evidence/TRAINING.json")
    draw_check = legacy.verify_draws(training_receipt["draws"], old_training["draws"], rows["clean"], rows["snr10"])
    losses = training_receipt["losses"]
    require(training_receipt["optimizer_updates"] == len(losses) == 96 and training_receipt["training_bag_forwards"] == 576
            and all(type(v) is float and math.isfinite(v) and v >= 0 for v in losses), "fixed 96 finite training losses")
    require(training_receipt["draws_exact_match"] is True
            and training_receipt["previous_training"] == ident(incumbent.read("evidence/TRAINING.json"))
            and training_receipt["development_checkpoint_or_threshold_selection"] is False
            and training_receipt["snr5_used_for_training"] is False and training_receipt["new_training_frontend_calls"] == 0
            and training_receipt["comparison"] == "ONLY_POSITIVE_DENSE_CE_CHANGED_EQUAL_DATA_DRAWS_AND_UPDATE_BUDGET",
            "unchanged data/draw/budget counterfactual and evaluation-only 5 dB")
    for key in ("positive_loss", "background_loss", "bag_aggregation"):
        require(training_receipt[key] == plan[key], "exact recorded dense loss contract")
    features = legacy.unique_rows(training.json("FEATURE_EVIDENCE.json"), "original features")
    geometry = [{"condition": condition, "id": row["id"], "role": "train", "weak_class": row["weak_class"],
                 "head_instance_ends": features[row["id"]]["head_instance_ends"]}
                for condition in ("clean", "snr10") for row in rows[condition] if row["role"] == "train"]
    reported_geometry = quality.json("evidence/TRAINING_GEOMETRY.json")
    require(len(reported_geometry) == len(geometry) and all(all(actual[k] == expected[k] for k in expected)
            for actual, expected in zip(reported_geometry, geometry)), "all dense bags use only original train-supported windows")
    weights, pins = {}, {}
    for label, filename in (("baseline", "baseline-head.npz"), ("control", "aug96-control-head.npz"), ("candidate", "dense-head.npz")):
        raw = quality.read("models/" + filename)
        weights[label], pins[label] = legacy.npz(raw), ident(raw)
        require(set(weights[label]) == set(SHAPES) and all(weights[label][k]["dtype"] == "<f4"
                and weights[label][k]["shape"] == shape for k, shape in SHAPES.items()), "same finite float32 head geometry")
        require(all(v[0] > 0 for v in legacy.struct.iter_unpack("<f", weights[label]["scale96"]["payload"])), "positive frozen scales")
    normalization = {}
    for key in ("mean96", "scale96"):
        raw = weights["baseline"][key]["payload"]
        require(raw == weights["control"][key]["payload"] == weights["candidate"][key]["payload"], "frozen normalization bytes")
        normalization[key] = {"bytes": len(raw), "sha256": digest(raw), "candidate_sha256": digest(raw), "byte_identical": True}
    require(training_receipt["warmstart"] == pins["baseline"] and training_receipt["control"] == pins["control"]
            and training_receipt["candidate"] == pins["candidate"] and training_receipt["normalization"] == normalization,
            "original baseline warmstart, saved candidate and norm byte binding")
    require(any(weights["candidate"][k]["payload"] != weights["baseline"][k]["payload"] for k in ("w1", "b1", "w2", "b2")),
            "new candidate trainable arrays changed")
    binary = b"EHKWS01\0" + b"".join(weights["candidate"][k]["payload"] for k in SHAPES)
    require(quality.read("models/dense-head.bin") == binary, "portable binary contains exact dense arrays")
    frozen = quality.json("evidence/CANDIDATE_FROZEN.json")
    require(frozen["candidate"] == pins["candidate"] and frozen["candidate_bin"] == ident(binary)
            and frozen["normalization"] == normalization, "candidate frozen byte identities")
    metrics, new_traces, old_traces = {}, [], []
    for condition in ("clean", "snr10", "snr5"):
        name = "dense-" + condition
        metrics["dense_" + condition], trace = legacy.read_trace(quality, name, rows[condition], clocks[condition])
        new_traces.append({"name": name, **trace})
        control_name = "candidate-" + condition if condition != "snr5" else "aug96-snr5"
        metrics["aug96_" + condition], trace = legacy.read_trace(quality, "reused/" + control_name, rows[condition], clocks[condition])
        old_traces.append({"name": control_name, **trace, "new_model_calls": 0})
    metrics["baseline_snr5"], trace = legacy.read_trace(quality, "reused/baseline-snr5", rows["snr5"], clocks["snr5"])
    old_traces.append({"name": "baseline-snr5", **trace, "new_model_calls": 0})
    counts = quality.json("evidence/PROGRESS.json")
    require(counts == EXPECTED_COUNTS == result["new_counts"]
            and sum(t["case_passes"] for t in new_traces) == 264 and sum(t["ready_outputs"] for t in new_traces) == 11847,
            "zero frontend and exact one-head/update budgets agree with saved traces")
    phases = verify_dense_phases(quality.json("evidence/PHASES.json"), frozen, counts, pins["candidate"]["sha256"])
    paired = [item for condition in ("clean", "snr10", "snr5") for item in legacy.compare_cases(
        metrics["aug96_" + condition], metrics["dense_" + condition], condition)]
    gate = legacy.strict_gate(paired, rows["snr5"])
    retention = legacy.expected_retention_record(gate, paired)
    require(quality.json("evidence/RESEARCH_RETENTION.json") == result["research_retention"] == retention,
            "identical strict incumbent-relative gate, never comparison to failed temporal")
    reported_pairs = quality.json("evidence/PAIRED_COMPARISONS.json")
    require(set(reported_pairs) == {"clean", "snr10", "snr5"}, "all three paired conditions")
    safety = {"prefix_count": "prefix_events", "wrong_count": "wrong_keyword_after_clip_start",
              "extra_count": "extra_same_keyword_events", "weak_negative_disagreement_count": "weak_negative_intent_events"}
    for condition in reported_pairs:
        expected = [r for r in paired if r["condition"] == condition]
        old = legacy.unique_rows(metrics["aug96_" + condition], "paired control")
        new = legacy.unique_rows(metrics["dense_" + condition], "paired dense")
        require(len(reported_pairs[condition]) == len(expected), "complete paired report")
        for report, row in zip(reported_pairs[condition], expected):
            wanted = {"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
                "previous_research_error": row["control_error"], "dense_research_error": row["candidate_error"],
                "previously_correct_case_regressed": row["regressed"],
                "event_error_categories_increased": [safety[k] for k in safety if k in row["introduced_event_errors"]],
                "development_miss_rescued": row["role"] == "development" and row["weak_class"] > 0
                    and row["control_missing_in_observation"] and not row["candidate_error"],
                "previous_events": old[row["id"]]["events"], "dense_events": new[row["id"]]["events"]}
            require(report == wanted, "independent complete dense paired record")
    for key, values in metrics.items():
        require(result[key] == legacy.grouped(values), "independent dense aggregate: " + key)
    require(result["baseline"] == pins["baseline"] and result["aug96_control"] == pins["control"]
            and result["dense_candidate"] == pins["candidate"], "result binds original, incumbent and dense weights")
    require(result["planned_new_conditions"] == 0 and result["planned_reused_snr5_conditions"] == 77
            and result["eligible_reused_snr5_conditions"] == 71
            and result["snr5_dispositions"] == {"MIXED": 71, "MIX_HEADROOM_EXCLUDED": 6}
            and result["frozen_embedding_rows_reused"] == {"clean": 10354, "snr10": 2804, "snr5": 2649}
            and result["failed_temporal_head_used_as_warmstart"] is False, "exact already-exposed condition counts and cache reuse")
    require(result["word_end_latency"] == "NOT_MEASURED"
            and all(result[key] == "NOT_RUN" for key in ("independent_far", "independent_frr", "new_head_c_numeric_parity", "ssc305"))
            and result["model_delivered_as_product"] is False, "no unsupported product, acoustic or latency claims")
    require(result["last_additional_objective_arm"] is True
            and result["subsequent_same_pool_loss_seed_threshold_search_allowed"] is False
            and result["same_source_condition_is_not_independent_audio"] is True
            and result["snr5_previous_evaluation_exposed"] is True, "stop rule and exposed evaluation status preserved")
    paired_summary = {}
    for row in paired:
        key = row["condition"] + "/" + row["role"] + "/" + str(row["weak_class"])
        item = paired_summary.setdefault(key, {"cases": 0, "rescued": 0, "regressed": 0, "new_event_error_cases": 0})
        item["cases"] += 1; item["rescued"] += int(row["rescued"]); item["regressed"] += int(row["regressed"])
        item["new_event_error_cases"] += bool(row["introduced_event_errors"])
    return {"schema": "independent-dense-quality-audit-v1", "status": "SAVED_DENSE_EVIDENCE_VERIFIED",
        "class_mapping": {"background": 0, "Hey Vosen": 1, "Hey Mero": 2},
        "quality_artifact": ident(quality.raw), "source_training_zip": ident(training.raw),
        "source_incumbent_zip": ident(incumbent.raw), "source_temporal_cache_zip": ident(temporal.raw),
        "frozen_auditor_sha256": FROZEN_AUDITOR_SHA, "run_id": result["run_id"], "head_sha": result["head_sha"],
        "supervisor_wall_seconds": wall_seconds, "worker_manifest_files": manifest_count, "source_copy_bindings": len(copied),
        "source_contract": source_review, "cached_streams": {k: len(v) for k, v in rows.items()},
        "cached_embedding_rows": {k: sum(len(t) for t in group.values()) for k, group in clocks.items()},
        "positive_supported_bags": positive_bags, "training_draws": draw_check, "weights": pins,
        "normalization": normalization, "portable_export_bytes_verified": True, "phases": phases, "counts": counts,
        "new_trace_checks": new_traces, "reused_trace_checks": old_traces,
        "metrics": {k: legacy.grouped(v, source_kind=True) for k, v in metrics.items()},
        "paired_summary": paired_summary, "paired_cases": paired, "research_gate": gate,
        "new_model_calls_by_this_auditor": 0, "new_audio_mixes_by_this_auditor": 0,
        "model_forward_numerics_recomputed": False, "head_logit_weight_correspondence_independently_recomputed": False,
        "independent_source_count_increment": 0, "same_source_condition_is_not_independent_audio": True,
        "development_conditions_already_exposed": True, "high_training_window_scores_prove_development_cause": False,
        "gradient_density_is_hypothesis_not_established_failure_cause": True, "quality_gain_guaranteed": False,
        "last_additional_objective_arm": True, "subsequent_same_pool_loss_seed_threshold_search_allowed": False,
        "fallback_if_gate_fails": "KEEP_PREVIOUS_AUG96_HEAD_AND_STOP_SAME_POOL_SEARCH",
        "original_eof_views": [{"condition": k, "id": r["id"], "role": r["role"], "weak_class": r["weak_class"],
            "no_matching_event_by_original_eof": r["original_eof_no_matching_event_censored"],
            "interpretation": "CENSORED_NOT_FRR", "word_end_latency": "NOT_MEASURED"}
            for k, values in metrics.items() for r in values if r["weak_class"]],
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED",
        "new_head_c_numeric_parity": "NOT_RUN", "ssc305": "NOT_RUN", **dict.fromkeys(FALSE_FLAGS, False)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True, help="pinned completed temporal ZIP used only for cached evidence")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--expected-run")
    parser.add_argument("--expected-head")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(args.preflight != bool(args.quality), "choose exactly --preflight or --quality")
    sources = [legacy.Archive(path.read_bytes()) for path in (args.training, args.incumbent, args.temporal)]
    report = input_preflight(*sources) if args.preflight else audit(legacy.Archive(args.quality.read_bytes()), *sources,
                                                                 args.expected_run, args.expected_head)
    payload = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                          "bytes": len(payload.encode()), "sha256": digest(payload.encode()),
                          "new_model_calls_by_this_auditor": 0}, sort_keys=True))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
