"""Bind the six immutable, already-screened source archives into two candidate-only V7 snapshots.

No TTS, ASR, phoneme model, KWS inference, optimizer, relabeling or retry occurs here.
TRAIN/DEV candidate snapshots are physically separate from the sealed CAL/LOCKED evaluation packet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import wave
import zipfile
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_canonical_multisource_bind_once as source_binder


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_once(path: Path, raw: bytes):
    path = Path(path)
    need(isinstance(raw, bytes) and not path.exists() and not path.is_symlink(),
         "NEW_IMMUTABLE_OUTPUT_FILE")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    return {"path": path.relative_to(path.parents[1]).as_posix(),
            "bytes": len(raw), "sha256": sha(raw)}


def write_json(path: Path, value):
    raw = canonical(value) + b"\n"
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), "NEW_IMMUTABLE_JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    return raw


def bundle_from_config(cfg):
    artifacts = []
    for corpus in cfg["corpus"]:
        key = corpus["key"]
        for kind in ("source", "screen"):
            part = corpus[kind]
            item = {
                "key": key, "kind": kind, "run_id": part["run"],
                "commit": part["sha"], "artifact_id": part["id"],
                "name": part["name"], "bytes": part["bytes"],
                "sha256": part["sha256"],
            }
            if kind == "source":
                item.update({"denominator": part["denominator"],
                             "source_schema": part["source_schema"]})
            else:
                item.update({"status": part["status"],
                             "qualified_by_role": corpus["qualified_by_role"]})
            artifacts.append(item)
    return {
        "schema": "english-canonical-multisource-reviewed-pcm-v1",
        "artifacts": artifacts,
        "source_policy_git_blob_sha1": cfg["frozen_phones_rule_git_blob_sha1"],
        "source_plan_git_blobs": {
            "fresh24": "dbe911fbcf8a697492ad29c7ca11858e707a72ff",
            "fresh18": "c2ee2f7bf2fbdeb8821920d6cb5e6890d0df1f76",
            "piper24": "d477ecec2a07910fc1a91063752aff1acc9006ef",
        },
        "qualified_groups_by_role": cfg["exact_expected_qualified"],
        "minimum_groups_by_role": cfg["full_qualified_role_minimum"],
        "selected_machine_reviewed_rows": 63,
        "original_rows_authenticated": sum(x["source"]["denominator"] for x in cfg["corpus"]),
        "new_tts_calls": cfg["new_tts_calls"],
        "new_asr_calls": cfg["new_asr_calls"],
        "new_phone_calls": cfg["new_phone_calls"],
        "audio_regenerated": False,
        "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "original_screen_training_snapshot_allowed": False,
        "fresh_candidate_snapshot_required": True,
    }


def source_check(pipeline_root: Path):
    pipeline_root = Path(pipeline_root)
    sys.path.insert(0, (pipeline_root / "training").as_posix())
    import english_kws_dual_track_v7_train as trainer
    cfg = source_binder.config()
    plan = trainer.load_plan()
    need(bundle_from_config(cfg) == plan["source_bundle"],
         "DATA_AND_PIPELINE_SOURCE_BUNDLE_MUST_MATCH_EXACTLY")
    need(plan["source_roles"] == cfg["exact_expected_qualified"],
         "SOURCE_ROLE_COVERAGE_MATCH")
    return {
        "status": "MULTISOURCE_V7_BINDER_READY_NO_MODEL_EXECUTION",
        "source_artifacts": 6,
        "original_source_rows": 462,
        "selected_machine_reviewed_rows": 63,
        "qualified_groups": cfg["exact_expected_qualified"],
        "new_tts_calls": 0, "new_asr_calls": 0, "new_phone_calls": 0,
        "optimizer_updates": 0, "human_gold": False,
        "source_bundle_sha256": plan["source_plan_sha256"],
    }


def read_source_audio(zips, original_rows, row):
    corpus = row["corpus"]
    original = original_rows[corpus][row["id"]]
    raw_wav = zips[corpus].read("wav/" + row["id"] + ".wav")
    pcm = source_binder.read_wav(raw_wav, original)
    need(sha(raw_wav) == row["source_wav_sha256"]
         and sha(pcm) == row["source_pcm_sha256"]
         and len(pcm) == row["samples"] * 2,
         "ORIGINAL_WHOLE_PCM_AND_WAV_IDENTITY")
    return pcm, raw_wav


def group_rows(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["source_group_id"]].append(row)
    return result


def write_candidate_snapshot(output, candidate, selected, groups, zips,
                             original_rows, plan, run_identity):
    candidate_plan = plan["candidates"][candidate]
    positive_class = 1 if candidate == "VOSEN" else 2
    role_rows = [row for row in selected if row["role"] in ("train", "development")]
    role_groups = [{**group, "source_speaker_id": group["source_group_id"]}
                    for group in groups if group["role"] in ("train", "development")]
    need(Counter(group["role"] for group in role_groups) == Counter({"train": 8, "development": 4}),
         "CANDIDATE_TRAIN_DEV_GROUPS_8_4")
    by_group = group_rows(role_rows)
    for group in role_groups:
        rows = by_group[group["source_group_id"]]
        positives = sum(row["weak_class"] == positive_class for row in rows)
        negatives = sum(row["weak_class"] != positive_class for row in rows)
        need(positives >= 1 and negatives >= 2, "CANDIDATE_GROUP_POSITIVE_NEGATIVE_BALANCE")

    need(not output.exists() and not output.is_symlink(), "NEW_CANDIDATE_SNAPSHOT")
    output.mkdir(parents=True)
    snapshot_rows = []
    file_entries = []
    total_bytes = 0
    for source_row in sorted(role_rows, key=lambda row: row["id"]):
        pcm, wav = read_source_audio(zips, original_rows, source_row)
        row_id = source_row["id"]
        pcm_name = "audio/" + row_id + ".pcm"
        wav_name = "audio/" + row_id + ".wav"
        for name, raw in ((pcm_name, pcm), (wav_name, wav)):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            need(not target.exists() and not target.is_symlink(), "NEW_CANDIDATE_AUDIO")
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
            file_entries.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
            total_bytes += len(raw)
        research_class = 1 if source_row["weak_class"] == positive_class else 0
        train = source_row["role"] == "train"
        weight = 1.0 if source_row["tier"] == "silver" else 0.5
        snapshot_rows.append({
            "id": row_id, "role": source_row["role"],
            "source_group_id": source_row["source_group_id"],
            "source_speaker_id": source_row["source_group_id"],
            "source_corpus": source_row["corpus"],
            "candidate_id": candidate_plan["candidate_id"],
            "research_class": research_class,
            "original_three_class_label": source_row["weak_class"],
            "requested_class": source_row["requested_class"],
            "tier": source_row["tier"], "machine_reason": source_row["machine_reason"],
            "training_eligible": train,
            "training_class": research_class if train else None,
            "training_weight": weight if train else 0.0,
            "optimizer_allowed": False, "locked_test_sealed": False,
            "human_gold": False, "training_allowed_in_original": False,
            "pcm_path": pcm_name, "wav_path": wav_name,
            "pcm_sha256": sha(pcm), "wav_sha256": sha(wav),
            "source_pcm_sha256": source_row["source_pcm_sha256"],
            "source_wav_sha256": source_row["source_wav_sha256"],
            "pcm_samples": source_row["samples"],
        })
    snapshot_rows.sort(key=lambda row: row["id"])
    manifest = {
        "schema": "english-kws-dual-track-v7-training-snapshot-manifest-v1",
        "candidate": candidate, "candidate_id": candidate_plan["candidate_id"],
        "phrase": candidate_plan["phrase"], "locale": "en-US",
        "ipa": candidate_plan["ipa"], "ctc_tokens": candidate_plan["token_labels"],
        "ctc_token_ids": candidate_plan["token_ids"],
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_run": run_identity,
        "source_groups": role_groups,
        "role_counts": {"train": 8, "development": 4},
        "rows": snapshot_rows,
        "training_rows": sum(row["role"] == "train" for row in snapshot_rows),
        "development_rows": sum(row["role"] == "development" for row in snapshot_rows),
        "training_snapshot_allowed": True,
        "calibration_rows_included": False, "locked_test_rows_included": False,
        "calibration_labels_included": False, "locked_test_labels_included": False,
        "calibration_audio_included": False, "locked_test_audio_included": False,
        "optimizer_updates": 0, "kws_model_forwards": 0,
        "product_promotion_allowed": False,
    }
    manifest_raw = write_json(output / "MANIFEST.json", manifest)
    file_entries.append({"path": "MANIFEST.json", "bytes": len(manifest_raw),
                         "sha256": sha(manifest_raw)})
    total_bytes += len(manifest_raw)
    file_entries.sort(key=lambda item: item["path"])
    inventory = {
        "schema": "english-kws-dual-track-v7-training-snapshot-files-v1",
        "candidate": candidate, "files": file_entries,
    }
    inventory_raw = write_json(output / "FILES.json", inventory)
    snapshot = {
        "schema": "english-kws-dual-track-v7-training-snapshot-v1",
        "candidate": candidate, "candidate_id": candidate_plan["candidate_id"],
        "source_run": run_identity, "source_plan_sha256": plan["source_plan_sha256"],
        "source_candidate_manifest_sha256": sha(canonical(manifest)),
        "source_packet_files_manifest_sha256": plan["source_plan_sha256"],
        "manifest_sha256": sha(manifest_raw),
        "files_manifest_sha256": sha(inventory_raw),
        "file_count": len(file_entries), "verified_bytes": total_bytes,
        "training_rows": manifest["training_rows"],
        "development_rows": manifest["development_rows"],
        "role_counts": {"train": 8, "development": 4},
        "calibration_rows_included": False, "locked_test_rows_included": False,
        "calibration_labels_included": False, "locked_test_labels_included": False,
        "calibration_audio_included": False, "locked_test_audio_included": False,
        "locked_test_access_in_training_job": False,
        "optimizer_updates": 0, "kws_model_forwards": 0,
        "product_promotion_allowed": False,
    }
    write_json(output / "SNAPSHOT.json", snapshot)
    return {"candidate": candidate, "rows": len(snapshot_rows),
            "training_rows": manifest["training_rows"],
            "development_rows": manifest["development_rows"],
            "snapshot_sha256": sha(canonical(snapshot))}


def write_evaluation_packet(output, selected, groups, zips, original_rows, plan, run_identity):
    roles = ("calibration", "locked-test")
    eval_rows = [row for row in selected if row["role"] in roles]
    eval_groups = [group for group in groups if group["role"] in roles]
    need(Counter(group["role"] for group in eval_groups)
         == Counter({"calibration": 3, "locked-test": 3}),
         "SEALED_EVALUATION_GROUPS_3_3")
    need(not output.exists() and not output.is_symlink(), "NEW_SEALED_EVALUATION_PACKET")
    output.mkdir(parents=True)
    entries = []
    rows = []
    total = 0
    for row in sorted(eval_rows, key=lambda item: item["id"]):
        pcm, wav = read_source_audio(zips, original_rows, row)
        for suffix, raw in (("pcm", pcm), ("wav", wav)):
            name = "audio/" + row["id"] + "." + suffix
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
            entries.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
            total += len(raw)
        rows.append({
            "id": row["id"], "role": row["role"], "source_group_id": row["source_group_id"],
            "source_speaker_id": row["source_group_id"], "source_corpus": row["corpus"],
            "observed_class": row["weak_class"], "tier": row["tier"],
            "machine_reason": row["machine_reason"], "human_gold": False,
            "pcm_path": "audio/" + row["id"] + ".pcm",
            "wav_path": "audio/" + row["id"] + ".wav",
            "pcm_sha256": sha(pcm), "wav_sha256": sha(wav),
            "pcm_samples": row["samples"],
        })
    manifest = {
        "schema": "english-kws-dual-track-v7-sealed-evaluation-manifest-v1",
        "source_plan_sha256": plan["source_plan_sha256"], "source_run": run_identity,
        "source_groups": eval_groups, "role_counts": {"calibration": 3, "locked-test": 3},
        "rows": rows, "training_allowed": False, "human_gold": False,
        "threshold_tuning_before_weight_freeze": False,
        "locked_test_single_pass_after_calibration_and_weight_freeze": True,
    }
    raw = write_json(output / "EVALUATION.json", manifest)
    entries.append({"path": "EVALUATION.json", "bytes": len(raw), "sha256": sha(raw)})
    total += len(raw)
    inventory = {"schema": "english-kws-dual-track-v7-sealed-evaluation-files-v1",
                 "files": sorted(entries, key=lambda item: item["path"])}
    inv_raw = write_json(output / "FILES.json", inventory)
    return {"rows": len(rows), "groups": len(eval_groups),
            "verified_bytes": total, "files_manifest_sha256": sha(inv_raw)}


def bind(sources: Path, pipeline_root: Path, output: Path, evaluation_output: Path):
    pipeline_root = Path(pipeline_root)
    sys.path.insert(0, (pipeline_root / "training").as_posix())
    import english_kws_dual_track_v7_train as trainer
    plan = trainer.load_plan()
    cfg = source_binder.config()
    need(bundle_from_config(cfg) == plan["source_bundle"],
         "EXACT_PIPELINE_AND_DATA_SOURCE_BUNDLE")
    run_identity = trainer.guard(plan)
    admission = source_binder.admission(Path(sources), verify_audio=True)
    need(admission["source_group_counts"] == plan["source_roles"]
         and admission["selected_machine_reviewed_rows"] == 63
         and admission["original_candidate_denominator"] == 462,
         "REAL_CUMULATIVE_8_4_3_3_AND_63_ROW_ADMISSION")
    source_zips = {}
    original_rows = {}
    try:
        for corpus in cfg["corpus"]:
            key = corpus["key"]
            z = zipfile.ZipFile(Path(sources) / (key + "-source.zip"))
            source_zips[key] = z
            source_manifest = json.loads(z.read("MANIFEST.json"))
            original_rows[key] = {row["id"]: row for row in source_manifest["rows"]}
        groups = admission["qualified_source_groups"]
        summaries = []
        for candidate in ("VOSEN", "MERO"):
            summaries.append(write_candidate_snapshot(
                Path(output) / candidate, candidate, admission["rows"], groups,
                source_zips, original_rows, plan, run_identity))
        evaluation = write_evaluation_packet(
            Path(evaluation_output), admission["rows"], groups, source_zips,
            original_rows, plan, run_identity)
        result = {
            "schema": "english-kws-dual-track-v7-multisource-bind-result-v1",
            "status": "TWO_CANDIDATE_SNAPSHOTS_AND_SEALED_EVALUATION_READY",
            "run": run_identity, "source_bundle_sha256": plan["source_plan_sha256"],
            "source_group_counts": admission["source_group_counts"],
            "source_rows": admission["original_candidate_denominator"],
            "selected_rows": admission["selected_machine_reviewed_rows"],
            "candidate_snapshots": summaries, "sealed_evaluation": evaluation,
            "new_tts_calls": 0, "new_asr_calls": 0, "new_phone_calls": 0,
            "optimizer_updates": 0, "kws_model_forwards": 0,
            "calibration_or_locked_data_in_candidate_snapshots": False,
            "human_gold": False, "product_promotion_allowed": False,
        }
        write_json(Path(output) / "BIND_RESULT.json", result)
        return result
    finally:
        for z in source_zips.values():
            z.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    need(args.source_check != args.bind, "ONE_BINDER_STAGE")
    if args.source_check:
        need(args.pipeline_root, "PIPELINE_ROOT_REQUIRED")
        print(json.dumps(source_check(args.pipeline_root), sort_keys=True))
        return
    need(args.sources and args.pipeline_root and args.output and args.evaluation_output,
         "SOURCES_PIPELINE_AND_TWO_OUTPUTS_REQUIRED")
    result = bind(args.sources, args.pipeline_root, args.output, args.evaluation_output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
