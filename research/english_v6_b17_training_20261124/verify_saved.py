"""Audit authenticated B17 saved evidence only. No Torch import or model forward."""
import argparse
import ast
from collections import Counter, OrderedDict, defaultdict
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RECEIPT_SHA256 = "c2d1af6512945bcfdee1493847c82a3a39a6c8dae48170ae4fa641540274bf1f"
RUN, HEAD = 45319580193, "9654cfe1c28ed84fe7d1d2a91dab450d8efb1abf"
PARENT = "e46a54fb3e39b3a7a7abd8e8b1afdc00d1ec8cde"
B0 = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
B9 = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
DESIGN = "f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831"
ASSESSMENT = "7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948"


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def readj(path):
    return json.loads(path.read_bytes())


def decode(matrix):
    frame = matrix.argmax(axis=1).tolist()
    starts = [(i, t) for i, t in enumerate(frame) if t and (not i or t != frame[i - 1])]
    tokens = [t for _, t in starts]
    events = [starts[i][0] for i in range(2, len(tokens)) if tokens[i - 2:i + 1] == [1, 2, 3]]
    return tokens, events


def completed_spans(matrix):
    frame = matrix.argmax(axis=1).tolist()
    runs = []
    begin = 0
    while begin < len(frame):
        end = begin + 1
        while end < len(frame) and frame[end] == frame[begin]:
            end += 1
        if frame[begin]:
            runs.append([frame[begin], begin, end])
        begin = end
    for i in range(len(runs) - 2):
        if [r[0] for r in runs[i:i + 3]] == [1, 2, 3]:
            return runs[i:i + 3]
    return []


def checkpoint_tensors(raw):
    """Restricted data-only decoding; permit two known Torch state-dict constructors."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(archive.testzip() is None, "checkpoint ZIP CRC")
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        require(len(pkl) == 1, "one tensor pickle")
        base = pkl[0].rsplit("/", 1)[0]
        require(archive.read(base + "/byteorder") == b"little", "checkpoint byte order")

        def tensor(storage, offset, shape, stride, requires_grad, hooks, *extra):
            require(not extra and isinstance(storage, tuple) and len(storage) == 3,
                    "only expected state tensor representation")
            key, dtype, count = storage
            raw_values = archive.read(base + "/data/" + key)
            require(len(raw_values) == count * np.dtype(dtype).itemsize,
                    "storage byte size")
            return np.ndarray(tuple(shape), dtype=dtype, buffer=raw_values,
                              offset=offset * np.dtype(dtype).itemsize,
                              strides=tuple(s * np.dtype(dtype).itemsize for s in stride)).copy()

        class DataOnly(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("collections", "OrderedDict"):
                    return OrderedDict
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return tensor
                if (module, name) == ("torch", "FloatStorage"):
                    return "<f4"
                raise ValueError("unexpected pickle class " + module + "." + name)

            def persistent_load(self, pid):
                require(isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"
                        and pid[1] == "<f4" and pid[3] == "cpu"
                        and isinstance(pid[4], int) and pid[4] >= 0
                        and re.fullmatch(r"[0-9]+", pid[2]) is not None,
                        "bounded float CPU storage reference")
                return pid[2], pid[1], pid[4]

        result = DataOnly(io.BytesIO(archive.read(pkl[0]))).load()
        require(isinstance(result, OrderedDict) and all(isinstance(k, str) and isinstance(v, np.ndarray)
                                                      for k, v in result.items()), "plain tensor dictionary")
        return result


def main():
    global REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="Unmodified original GitHub artifact ZIP")
    parser.add_argument("--repository", type=Path, default=REPO, help="Checkout root; defaults to this verifier repository")
    parser.add_argument("--output", type=Path, help="Optional recomputed JSON audit path; otherwise only print")
    args = parser.parse_args()
    REPO = args.repository.resolve()
    receipt_raw = (ROOT / "RECEIPT.json").read_bytes()
    require(sha(receipt_raw) == RECEIPT_SHA256, "immutable original run and artifact receipt")
    receipt = json.loads(receipt_raw)
    metadata, run, commit = (receipt[k] for k in ("artifact_metadata", "run_metadata", "commit_metadata"))
    require(commit["sha"] == HEAD and [p["sha"] for p in commit["parents"]] == [PARENT],
            "server-verified actual single predecessor")
    raw = args.zip.read_bytes()
    require(metadata["id"] == 14887048023 and metadata["workflow_run"]["id"] == RUN
            and metadata["workflow_run"]["head_sha"] == HEAD
            and metadata["size_in_bytes"] == len(raw) == 2506252
            and metadata["digest"] == "sha256:" + sha(raw)
            and sha(raw) == "1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d",
            "exact server-authenticated training ZIP")
    require(run["id"] == RUN and run["head_sha"] == HEAD and run["run_attempt"] == 1
            and run["conclusion"] == "success" and run["status"] == "completed"
            and run["repository"]["private"] is False, "actual completed public attempt one")
    design_raw = (REPO / "automation/english_v6_b17_b0_margin_design.json").read_bytes()
    require(sha(design_raw) == DESIGN and sha((REPO / "automation/english_v6_b17_panels/assessment.json").read_bytes()) == ASSESSMENT,
            "local review protocol identical to saved frozen hashes")
    design = json.loads(design_raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 18,
                "18 unique complete artifact members")
        members = {n: {"bytes": len(z.read(n)), "sha256": sha(z.read(n))} for n in z.namelist()}
        require(members == receipt["original_member_sha256"], "every original member byte count and hash")
        doc = lambda name: json.loads(z.read(name))
        result, started = doc("RESULT.json"), doc("STARTED.json")
        source, diagnostic = doc("SOURCE.json"), doc("TRAIN_DIAGNOSTIC.json")
        teacher, draw = doc("B0_ORDINARY_TEACHER.json"), doc("COMMON_DRAW.json")
        for d in (result, started):
            require(d["run_id"] == str(RUN) and d["head"] == HEAD and d["parent"] == PARENT
                    and d["design_sha256"] == DESIGN and d["assessment_sha256"] == ASSESSMENT
                    and d["dev_cal_locked_forwards"] == d["new_tts_asr_calls"] == 0
                    and d["promotion_allowed"] is False, "truthful saved run and boundary")
        require(result["status"] == "TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV"
                and result["parent_pt_sha256"] == source["parent_pt_sha256"] == teacher["parent_pt_sha256"] == B0
                and source["positive_teacher_pt_sha256"] == B9
                and result["actual_optimizer_updates"] == started["maximum_optimizer_updates"] == 384
                and result["trained_model_parameters"] == 43876
                and result["seed"] == 20261120 and result["common_learning_rate"] == 6e-5
                and source["design_sha256"] == DESIGN and source["independent_assessment_sha256"] == ASSESSMENT
                and source["forbidden_sources_opened"] is False, "fixed pair and no promotion")
        # Compare every recorded original input archive against immutable source pins.
        tree = ast.parse((REPO / "automation/english_v6_b11_train_hardness.py").read_text())
        archives = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "ARCHIVES" for t in n.targets))
        expected_inputs = {name + ".zip": pin[1] for name, pin in archives.items()}
        old = readj(REPO / "automation/english_v6_b15_nonblank_contrast_design.json")
        natural_pins = readj(REPO / "automation/english_v6_b13_natural_pair_inputs.json")
        expected_inputs.update({"B0.zip": design["baseline"]["sha256"],
                                "diagnostic.zip": old["diagnostic"]["sha256"],
                                "pair.zip": old["historical_b13"]["sha256"],
                                **{name + ".zip": natural_pins[name]["sha256"] for name in ("natural", "replay")}})
        require(source["input_zip_sha256"] == expected_inputs, "all 15 input archive hash receipts")
        original = {r["id"]: r for r in source["base_train"]}
        natural = {r["id"]: r for r in source["natural_train"]}
        require(len(original) == 79 and len(natural) == 9 and not set(original) & set(natural)
                and len({r["source_group_id"] for r in natural.values()}) == 8,
                "exact 79 synthetic and nine natural originals / eight readers")
        review_raw = (REPO / "research/english_hardneg_increment_20261117/OWNER_REVIEW.json").read_bytes()
        require(sha(review_raw) == natural_pins["review_sha256"], "original owner review SHA")
        review = {r["candidate_id"]: r for r in json.loads(review_raw)["candidates"]}
        require(set(natural) == set(review), "exact owner-reviewed window roster")
        for uid, r in natural.items():
            approved = review[uid]
            require(r["source_pcm_sha256"] == approved["window_pcm_sha256"]
                    and r["samples"] == approved["window_samples"]
                    and r["start_sample"] == approved["start_sample"]
                    and r["end_sample_exclusive"] == approved["end_sample_exclusive"]
                    and r["speaker_id"] == approved["speaker_id"]
                    and r["recording_id"] == approved["recording_id"]
                    and r["role"] == "train" and r["mono_class"] == 0,
                    "same owner-reviewed natural window and role")
        require(len(draw) == 192 and sha(z.read("COMMON_DRAW.json")) == result["common_draw_sha256"],
                "shared complete 192-step draw")
        rows = {}
        for step, slot in enumerate(draw):
            require(slot["step"] == step + 1 and slot["prefix"] == (6400, 12800, 19200)[(step // 35 + step % 3) % 3],
                    "fixed step and prefix")
            for kind, klass in (("ordinary", 0), ("positive", 1), ("natural", 0)):
                r = slot[kind]
                require(r["role"] == "train" and r["mono_class"] == klass, "only appropriate TRAIN rows")
                require(rows.get(r["id"], r) == r, "consistent source identity across steps")
                rows[r["id"]] = r
                if kind == "natural":
                    require(r == natural[r["id"]], "exact natural draw identity")
                else:
                    require(original[r["id"]] == {"id": r["id"], "group": r["source_group_id"], "pcm_sha256": r["source_pcm_sha256"]},
                            "exact base draw identity")
        require(set(rows) == set(original) | set(natural), "every approved original exposed by schedule")
        cohorts = {k: defaultdict(list) for k in ("ordinary", "positive", "natural")}
        for r in rows.values():
            kind = "natural" if r["id"] in natural else "positive" if r["mono_class"] else "ordinary"
            cohorts[kind][r["source_group_id"]].append(r)
        for step, slot in enumerate(draw):
            for kind, groups in cohorts.items():
                order = sorted(groups)
                bucket = sorted(groups[order[step % len(order)]], key=lambda r: r["id"])
                require(slot[kind] == bucket[(step // len(order)) % len(bucket)], "shared deterministic group-balanced draw")
        require(len(cohorts["ordinary"]) == len(cohorts["positive"]) == 35, "35 synthetic TRAIN families")
        require(diagnostic["logits_sha256"] == sha(z.read("TRAIN_DIAGNOSTIC_LOGITS.npz"))
                and diagnostic["optimizer_updates"] == diagnostic["dev_cal_locked_forwards"] == diagnostic["new_audio_asr_calls"] == 0,
                "prerequisite diagnostic archive and no updates")
        elevated, recovered, origins = set(), set(), Counter()
        diagnostic_rows = {r["id"]: r for r in diagnostic["records"]}
        require(len(diagnostic_rows) == 88 and set(diagnostic_rows) == set(rows), "complete diagnostic source roster")
        gain_counts = {g: {name: {kind: 0 for kind in ("positive", "ordinary")} for name in ("B0", "B9")}
                       for g in ("1.0", "0.75", "0.5")}
        with np.load(io.BytesIO(z.read("TRAIN_DIAGNOSTIC_LOGITS.npz")), allow_pickle=False) as arrays:
            used = set()
            for uid, r in diagnostic_rows.items():
                src = rows[uid]
                require(r["role"] == "train" and r["positive"] == bool(src["mono_class"])
                        and r["pcm_sha256"] == src["source_pcm_sha256"] and r["group"] == src["source_group_id"]
                        and set(r["gains"]) == {"1.0", "0.75", "0.5"}, "diagnostic true TRAIN identity")
                for gain, state in r["gains"].items():
                    matrices = {}
                    for name in ("B0", "B9"):
                        s = state[name]
                        mat = arrays[s["array"]]
                        require(s["array"] not in used and mat.shape == ((src["samples"] - 512) // 160 + 1, 4)
                                and np.isfinite(mat).all() and sha(mat.tobytes()) == s["sha256"], "native diagnostic tensor SHA/grid")
                        used.add(s["array"])
                        tokens, events = decode(mat)
                        require(s["tokens"] == tokens and s["event_frames"] == events and s["events"] == len(events), "independent saved-logit CTC replay")
                        expected_origin = "saved_original_B13_TRAIN_diagnostic" if name == "B9" and r["positive"] else "new_B17_TRAIN_forward"
                        require(s["origin"] == expected_origin, "actual forward versus saved-array provenance")
                        origins[expected_origin] += 1
                        matrices[name] = mat
                        gain_counts[gain][name]["positive" if r["positive"] else "ordinary"] += int(bool(events))
                    b0, b9 = matrices["B0"], matrices["B9"]
                    margin = float(np.maximum((b9[:, 1:] - b9[:, :1]) - (b0[:, 1:] - b0[:, :1]), 0).mean())
                    require(margin == state["mean_positive_margin_increase"], "independent saved margin recomputation")
                    if not r["positive"] and margin > 0:
                        elevated.add(r["group"])
                    if r["positive"] and state["B9"]["events"] and not state["B0"]["events"]:
                        recovered.add(r["group"])
            require(used == set(arrays.files) and len(used) == diagnostic["saved_arrays"] == 528, "all 528 diagnostic arrays audited")
            for r in source["positive_retention"]:
                s = diagnostic_rows[r["id"]]["gains"][r["gain"]]["B9"]
                require(r["b9_array_sha256"] == s["sha256"] and r["b9_registered_spans"] == completed_spans(arrays[s["array"]])
                        and r["pcm_sha256"] == rows[r["id"]]["source_pcm_sha256"] and rows[r["id"]]["mono_class"] == 1,
                        "positive retention exactly bound to complete B9 word frames")
        require(sorted(elevated) == diagnostic["ordinary_margin_elevation_groups"] and len(elevated) >= 2
                and sorted(recovered) == diagnostic["b9_recovered_b0_positive_groups"] and len(recovered) >= 1
                and diagnostic["status"] == "TRAIN_SIGNAL" and diagnostic["optimizer_start_allowed"] is True,
                "unchanged predeclared TRAIN mechanism gate")
        require(origins["new_B17_TRAIN_forward"] == diagnostic["actual_new_model_forwards"] == result["actual_new_diagnostic_forwards"] == 396
                and origins["saved_original_B13_TRAIN_diagnostic"] == diagnostic["reused_saved_b9_positive_arrays"] == result["reused_saved_diagnostic_arrays"] == 132,
                "truthful new/reused diagnostic counters")
        positive = source["positive_retention"]
        require(len(positive) == len({(r["id"], r["gain"]) for r in positive}) == 120
                and len({r["id"] for r in positive}) == 43 and len({r["group"] for r in positive}) == 34
                and Counter(r["gain"] for r in positive) == {"1.0": 42, "0.75": 42, "0.5": 36}, "all frozen eligible positive conditions")
        expected_teacher = {(slot[kind]["id"], gain, slot["prefix"]) for slot in draw
                            for kind in ("ordinary", "natural") for gain in (1.0, 0.5)}
        require(teacher["logits_sha256"] == sha(z.read("B0_ORDINARY_TEACHER_LOGITS.npz"))
                and teacher["positive_rows"] == teacher["dev_cal_locked_forwards"] == 0
                and teacher["reused_for_both_arms"] is True, "shared B0 teacher boundary")
        with np.load(io.BytesIO(z.read("B0_ORDINARY_TEACHER_LOGITS.npz")), allow_pickle=False) as arrays:
            observed, keys = set(), set()
            for r in teacher["rows"]:
                key = r["id"], r["gain"], r["prefix_samples"]
                src = rows[r["id"]]
                mat = arrays[r["array"]]
                require(key not in observed and r["array"] not in keys and src["mono_class"] == r["mono_class"] == 0
                        and r["role"] == "train" and r["pcm_sha256"] == src["source_pcm_sha256"]
                        and r["source_group_id"] == src["source_group_id"] and np.isfinite(mat).all()
                        and mat.shape == ((src["samples"] + r["prefix_samples"] - 512) // 160 + 1, 4)
                        and r["frames"] == len(mat) and r["sha256"] == sha(mat.tobytes()), "same ordinary TRAIN gain/prefix B0 array")
                observed.add(key)
                keys.add(r["array"])
            require(observed == expected_teacher and keys == set(arrays.files)
                    and len(observed) == teacher["actual_model_forwards"] == result["actual_ordinary_teacher_forwards"] == 124,
                    "all 124 unique required ordinary teacher forwards")
        grouped = defaultdict(list)
        for r in positive:
            grouped[r["group"]].append(r)
        group_order = sorted(grouped)
        expected_optimization = 192 * 6 + sum(len(grouped[group_order[s % 34]]) for s in range(192))
        tensors, frontends, model_pins, loss_summary = {}, {}, {}, {}
        for arm, coefficient in (("control", 0.0), ("challenger", 0.08)):
            receipt = result[arm]
            require(receipt["optimizer_updates"] == 192 and receipt["parent_pt_sha256"] == B0
                    and receipt["natural_loss_weight"] == receipt["positive_retention_weight"] == 0.08
                    and receipt["ordinary_ceiling_weight"] == coefficient
                    and receipt["forward_counts"] == {"optimization": expected_optimization, "checkpoint_reload_parity": 2, "onnx_export_trace": 1},
                    "shared fit counters and sole ceiling factor")
            trace, retention = doc(arm + "/OPTIMIZER_TRACE.json"), doc(arm + "/POSITIVE_RETENTION_TRACE.json")
            require(len(trace) == len(retention) == 192, "actual complete per-update traces")
            for step, (tr, pr, slot) in enumerate(zip(trace, retention, draw)):
                group = group_order[step % 34]
                require(tr["step"] == pr["step"] == step + 1 and tr["prefix"] == slot["prefix"]
                        and all(tr[k + "_id"] == slot[k]["id"] for k in ("ordinary", "positive", "natural"))
                        and tr["ceiling_coefficient"] == coefficient and pr["group"] == group
                        and pr["eligible_originals"] == len({r["id"] for r in grouped[group]})
                        and all(np.isfinite(tr[k]) for k in ("loss", "positive_retention", "ordinary_ceiling"))
                        and np.isclose(tr["positive_retention"], pr["unweighted_loss"] * 0.08, atol=1e-7, rtol=1e-6),
                        "actual saved update/source/positive-loss correspondence")
            stem = arm + "/models/mero43-b17-" + arm + "192"
            require(receipt["stem"] == stem, "expected model stem")
            for ext in ("pt", "npz", "onnx"):
                name = stem + "." + ext
                require(receipt["files"][Path(name).name] == members[name], "actual model bytes and digest")
            state = checkpoint_tensors(z.read(stem + ".pt"))
            require(len(state) == 20 and sum(v.size for v in state.values()) == 43876
                    and all(np.isfinite(v).all() for v in state.values()), "complete finite checkpoint state")
            with np.load(io.BytesIO(z.read(stem + ".npz")), allow_pickle=False) as encoded:
                require(set(encoded.files) == set(state) | {"mel_filters", "train_mean", "train_scale"}
                        and all(np.array_equal(encoded[k], v) for k, v in state.items()), "PT/NPZ exact numerical tensor identity without inference")
                frontends[arm] = {k: encoded[k].copy() for k in ("mel_filters", "train_mean", "train_scale")}
            tensors[arm] = state
            model_pins[arm] = {"pt_sha256": sha(z.read(stem + ".pt")), "files": receipt["files"]}
            loss_summary[arm] = {"first": trace[0]["loss"], "last": trace[-1]["loss"],
                                 "maximum_ordinary_ceiling": max(t["ordinary_ceiling"] for t in trace)}
        require(all(np.array_equal(frontends["control"][k], v) for k, v in frontends["challenger"].items()), "same frozen frontend across arms")
        differing = sum(np.count_nonzero(tensors["control"][k] != v) for k, v in tensors["challenger"].items())
        require(int(differing) == result["different_trained_float_values"] == 43870
                and result["actual_student_forwards"] == 2 * (expected_optimization + 3) == 3670,
                "actual differing weights and full forward counts")
        summary = dict(schema="english-v6-b17-saved-training-evidence-audit-v1", status="PASS_TRAINING_EVIDENCE_ONLY",
                       run_id=RUN, run_head=HEAD, parent=PARENT, artifact_id=metadata["id"],
                       artifact_bytes=len(raw), artifact_sha256=sha(raw), design_sha256=DESIGN,
                       assessment_sha256=ASSESSMENT, parent_pt_sha256=B0, positive_teacher_pt_sha256=B9,
                       optimizer_updates=384, diagnostic_arrays=528,
                       new_diagnostic_forwards=396, reused_diagnostic_arrays=132,
                       ordinary_teacher_arrays=124, student_forwards=3670,
                       ordinary_margin_elevation_groups=len(elevated), b9_recovered_b0_positive_groups=len(recovered),
                       train_conditions_with_event=gain_counts, diagnostic_ctc_and_margin_recomputed=True,
                       original_source_zip_hash_receipts=len(expected_inputs), owner_reviewed_natural_windows=9,
                       positive_retention_combinations=120, positive_retention_originals=43, positive_retention_groups=34,
                       finite_model_parameters_per_arm=43876, differing_trained_float_values=int(differing),
                       checkpoint_npz_tensor_identity=True, model_pins=model_pins, saved_trace_loss_summary=loss_summary,
                       actual_new_local_model_forwards=0, actual_new_optimizer_updates=0,
                       dev_cal_locked_forwards=0, promotion_allowed=False,
                       limits="Audits authenticated saved evidence, exact input hash receipts and source identities; no original audio re-download, model inference, quality scoring, deployment, C11/ARM or board claim.",
                       member_sha256=members)
        if args.output is not None:
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
