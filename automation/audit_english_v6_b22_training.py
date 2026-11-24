"""Read original B22 training artifacts without loading or running a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(archive, name):
    return json.loads(archive.read(name))


def audit(models_path, producer_path, expected_models_sha, expected_producer_sha):
    import numpy as np

    model_raw, producer_raw = Path(models_path).read_bytes(), Path(producer_path).read_bytes()
    require(digest(model_raw) == expected_models_sha, "original models archive digest")
    require(digest(producer_raw) == expected_producer_sha, "original producer archive digest")
    with zipfile.ZipFile(io.BytesIO(producer_raw)) as producer, zipfile.ZipFile(io.BytesIO(model_raw)) as models:
        require(producer.testzip() is models.testzip() is None, "complete original ZIP CRCs")
        manifest = read_json(producer, "PRODUCER.json")
        initial = read_json(producer, "repository/research/english_v6_b22_training_20261124/initial_failure/PRODUCER.json")
        require(len(initial["files"]) == 243 and set(initial["files"]) <= set(manifest["files"])
                and len(manifest["files"]) > 243, "all original243 files plus complete first-fit continuation")
        for name, pin in manifest["files"].items():
            raw = producer.read("repository/" + name)
            require(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "producer bytes: " + name)
            require(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                    == pin["git_blob_sha1"], "producer Git blob: " + name)

        result = read_json(models, "RESULT.json")
        design_raw = models.read("DESIGN.json")
        design = json.loads(design_raw)
        source_raw = models.read("SOURCE.json")
        source = json.loads(source_raw)
        require(result["status"] == "TRAINED_UNION_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY",
                "completed original TRAIN result")
        require(result["identity"] == manifest["identity"] == read_json(models, "STARTED.json")["identity"],
                "one exact original execution identity")
        require(design_raw == producer.read("repository/automation/english_v6_b22_union_retention_design.json")
                and digest(design_raw) == result["recipe_sha256"] == manifest["recipe_sha256"],
                "exact prospectively frozen training design")
        require(models.read("ASSESSMENT_BINDING.json")
                == producer.read("repository/automation/english_v6_b22_assessment_binding.json"),
                "assessment frozen before optimizer")
        require(digest(source_raw) == result["source_sha256"] and source["originals"] == 88
                and source["ordinary_teachers_reused"] == 264, "original TRAIN source inventory")
        require(result["actual_optimizer_updates"] == 192 and result["actual_new_model_forwards"] == 1847,
                "registered complete optimizer and forward budget")
        require(all(result[key] == 0 for key in ("new_control_optimizer_updates", "new_control_forwards",
                "new_teacher_forwards", "dev_cal_locked_forwards", "new_tts_asr_calls")),
                "no historical control/teacher or evaluation replay")
        require(result["promotion_allowed"] is result["physical_ssc305"] is result["same_job_matched_control"] is False,
                "accurate research-only historical-control result")
        require(result["original_retention_conditions"] == 120 and result["union_retention_conditions"] == 122,
                "exact positive coverage intervention")

        for member, field in (("COMMON_DRAW.json", "common_draw_sha256"),
                              ("UNION_EVIDENCE.json", "union_evidence_sha256"),
                              ("OLD120_POSITIVE_REFERENCES.json", "old120_references_sha256")):
            require(digest(models.read(member)) == result[field], "original receipt: " + member)
        require(read_json(models, "UNION_EVIDENCE.json")["additions"] == design["retention_additions"],
                "two and only two frozen teacher additions")

        original_models = {}
        for arm in ("control", "challenger"):
            for name, pin in result[arm]["files"].items():
                raw = models.read(arm + "/models/" + name)
                require(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "original model export: " + name)
                original_models[arm + "/" + name] = pin
                if name.endswith(".pt"):
                    require(pin["sha256"] == result["model_pt_sha256"][arm], "model identity: " + arm)
        require(result["model_pt_sha256"]["control"] == design["historical_control"]["pt_sha256"],
                "unchanged original B21 historical control")
        with np.load(io.BytesIO(models.read("challenger/models/mero43-b22-challenger192.npz")), allow_pickle=False) as state:
            weights = [state[key] for key in state.files if key not in {"mel_filters", "train_mean", "train_scale"}]
            require(sum(value.size for value in weights) == 43876
                    and all(np.isfinite(value).all() for value in weights), "complete finite 43,876 weights")

        forward_index = read_json(models, "challenger/TRAIN_FORWARD_INDEX.json")
        arrays_raw = models.read("challenger/TRAIN_FORWARD_LOGITS.npz")
        records = forward_index["records"]
        require(forward_index["count"] == len(records) == 1844
                and forward_index["pending_unreturned_forward"] is None
                and forward_index["logits_are_raw"] is True
                and digest(arrays_raw) == forward_index["arrays_sha256"], "all original optimization returns")
        frames = 0
        with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as arrays:
            require(len(arrays.files) == len(records)
                    and set(arrays.files) == {row["array"] for row in records}, "one array per original forward")
            for row in records:
                value = arrays[row["array"]]
                require(value.shape == (row["frames"], 4) and np.isfinite(value).all()
                        and digest(value.tobytes()) == row["sha256"], "finite authentic full-grid forward")
                frames += len(value)
        counts = Counter(row["kind"] for row in records)
        require(counts == {"synthetic_ordinary": 384, "synthetic_positive": 384,
                           "natural_ordinary": 384, "positive_retention": 692}, "independent forward denominator")
        trace = read_json(models, "challenger/OPTIMIZER_TRACE.json")
        positive = read_json(models, "challenger/POSITIVE_RETENTION_TRACE.json")
        ordinary = read_json(models, "challenger/ORDINARY_MARGIN_TRACE.json")
        require([row["step"] for row in trace] == list(range(1, 193))
                and len(positive) == 692 and len(ordinary) == 768, "complete original optimizer/statistic traces")
        require(all(np.isfinite(row["loss"]) and np.isfinite(row["preclip_gradient_norm"]) for row in trace),
                "finite actual training loss and gradient trace")
        with np.load(io.BytesIO(models.read("challenger/CHECKPOINT_RELOAD_LOGITS.npz")), allow_pickle=False) as arrays:
            require(set(arrays.files) == {"saved", "reloaded"}
                    and np.array_equal(arrays["saved"], arrays["reloaded"])
                    and np.isfinite(arrays["saved"]).all(), "two saved original checkpoint checks agree")
        return dict(schema="english-v6-b22-original-training-readback-v1", status="PASS",
                    identity=result["identity"], models_zip_sha256=expected_models_sha,
                    producer_zip_sha256=expected_producer_sha, producer_files=len(manifest["files"]),
                    optimizer_updates=192, saved_optimization_arrays=len(records), saved_optimization_frames=frames,
                    original_model_forwards=1847, forward_counts=dict(counts), parameters=43876,
                    model_pt_sha256=result["model_pt_sha256"], model_files=original_models,
                    newly_protected_training_visits=sum(row["newly_protected"] for row in positive),
                    checkpoint_original_returns_equal=True, no_new_model_calls=True,
                    promotion_allowed=False, replication_allowed=False, engineering_allowed=False,
                    actual_quality_status="UNASSESSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("producer", type=Path)
    parser.add_argument("--models-sha256", required=True)
    parser.add_argument("--producer-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.models, args.producer, args.models_sha256, args.producer_sha256),
                     indent=2, sort_keys=True, allow_nan=False))
