"""Read-only, exact-artifact audit of the 63 real English TTS/three-ASR receipts.

This verifies original WAV/PCM and every independent native result BEFORE
comparing orthography. An ASR spelling is not acoustic ground truth:
no row is admitted to training, and near-misses cannot silently be negatives.
No TTS, ASR or optimizer calls; no network calls or artifact downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
import zipfile

import english_blind_asr as blind
import english_seven_voice_asr_once as asr
import multilingual_campaign as market

POLICY = "english-63-three-asr-source-linked-review-v1"
ASR_RUN_ID = 43246104418
ASR_HEAD_SHA = "25a328f41aa7b8a2473a1a30024d3e7fcb8dd2c0"
# Digests are of the original GitHub Actions artifact ZIP bytes, not extracted trees.
ARCHIVES = {
    "tts": (13954744587, "f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc"),
    "qwen": (13956812495, "aa483b6c7f360356c4851f1f7d65e8b0db20414129ded0de95b4e2b7575ac601"),
    "whisper": (13957321094, "ea7041d3eaa6bbc9e44bc48ba0a90ce521c532a0e0471071fcab52fa38f7f489"),
    "parakeet": (13958425005, "cff9be11d89b4ae9f74fa7150b0ab957bc5ccb9dd50be5c1f55bdec390607113"),
}
MAX_ARCHIVE_MEMBERS = 350
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024


def need(ok: bool, error: str) -> None:
    if not ok:
        raise ValueError(error)


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unpack_pinned(path: Path, expected_sha: str, directory: Path) -> None:
    """Bounded, alias-free extraction after exact ZIP identity verification."""
    need(path.is_file() and not path.is_symlink() and
         digest_file(path) == expected_sha, "ASR_REVIEW_ARTIFACT_DIGEST_DRIFT")
    need(directory.is_dir() and not directory.is_symlink(),
         "ASR_REVIEW_DESTINATION_INVALID")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        need(0 < len(infos) <= MAX_ARCHIVE_MEMBERS, "ASR_REVIEW_ZIP_MEMBER_COUNT")
        names, total = set(), 0
        for info in infos:
            name = info.filename
            member = PurePosixPath(name)
            mode = info.external_attr >> 16
            need(name and not name.startswith("/") and "\\" not in name and
                 "\x00" not in name and
                 ".." not in member.parts and "." not in member.parts and
                 all(part not in ("", ".") for part in member.parts) and
                 name not in names and not stat.S_ISLNK(mode),
                 "ASR_REVIEW_ZIP_PATH_OR_ALIAS")
            names.add(name)
            need(0 <= info.file_size <= MAX_MEMBER_BYTES,
                 "ASR_REVIEW_ZIP_MEMBER_LIMIT")
            total += info.file_size
            need(total <= MAX_ARCHIVE_BYTES, "ASR_REVIEW_ZIP_TOTAL_LIMIT")
        for info in infos:
            dest = directory.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            dest.chmod(0o600)
            need(dest.stat().st_size == info.file_size, "ASR_REVIEW_ZIP_SIZE_DRIFT")


def normalize_observation(text: str) -> str:
    """Orthographic observation only; never phoneme/accent authentication."""
    clean = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)?", clean))


def read_provider(root: Path, provider: str, plan: dict,
                  cells: list[tuple[dict, Path, dict]]) -> dict[str, dict]:
    need(provider in asr.PROVIDERS, "ASR_REVIEW_PROVIDER_INVALID")
    provider_id, model_id, revision = asr.PROVENANCE[provider]
    run_root = root / "en-seven-asr-result"
    need(run_root.is_dir() and not run_root.is_symlink(), "ASR_REVIEW_ASR_ROOT")
    plan_sha = asr.sha(market.canonical(plan))
    summary = market.read(run_root / "summary.json")
    need(summary.get("schema") == asr.POLICY and
         summary.get("provider") == provider and
         summary.get("provider_id") == provider_id and
         summary.get("model_id") == model_id and
         summary.get("model_revision_sha") == revision and
         summary.get("source_tts_run_id") == asr.SOURCE_RUN_ID and
         summary.get("source_plan_sha256") == plan_sha and
         summary.get("source_audio_count") == asr.MAX_CALLS and
         summary.get("asr_attempts") == asr.MAX_CALLS and
         summary.get("completed_asr") == asr.MAX_CALLS and
         summary.get("error_type") is None and
         summary.get("status") == "COMPLETE" and
         summary.get("training_admitted") is False and
         summary.get("human_gold") is False and
         summary.get("three_asr_completed") is False,
         "ASR_REVIEW_PROVIDER_SUMMARY_DRIFT")
    started = market.read(run_root / "started.json")
    need(started.get("schema") == asr.POLICY and
         started.get("provider") == provider and
         started.get("provider_id") == provider_id and
         started.get("model_id") == model_id and
         started.get("model_revision_sha") == revision and
         started.get("source_tts_run_id") == asr.SOURCE_RUN_ID and
         started.get("source_tts_head_sha") == asr.SOURCE_HEAD_SHA and
         started.get("source_plan_sha256") == plan_sha and
         started.get("training_admitted") is False and
         started.get("no_retry") is True,
         "ASR_REVIEW_PROVIDER_STARTED_DRIFT")
    rows = summary.get("rows")
    need(type(rows) is list and len(rows) == asr.MAX_CALLS,
         "ASR_REVIEW_PROVIDER_ROW_COUNT")
    by_id = {r.get("cell_id"): r for r in rows if type(r) is dict}
    need(len(by_id) == asr.MAX_CALLS and
         set(by_id) == {cell["cell_id"] for cell, _, _ in cells},
         "ASR_REVIEW_PROVIDER_ROW_ROSTER")
    results = {}
    for cell, _, pcm in cells:
        cell_id = cell["cell_id"]
        row = by_id[cell_id]
        need(row.get("wav_sha256") == pcm["wav_sha256"] and
             row.get("role") == cell["role"] and
             row.get("source_family_id") == cell["source_family_id"] and
             row.get("status") == "BLIND_ASR_UNADMITTED",
             "ASR_REVIEW_PROVIDER_ROW_DRIFT")
        folder = run_root / "cells" / cell_id
        begun = market.read(folder / "started.json")
        need(begun.get("schema") == asr.POLICY and
             begun.get("cell_id") == cell_id and
             begun.get("provider_id") == provider_id and
             begun.get("wav_sha256") == pcm["wav_sha256"] and
             begun.get("model_revision_sha") == revision and
             begun.get("attempt") == 1 and
             begun.get("training_admitted") is False,
             "ASR_REVIEW_CELL_STARTED_DRIFT")
        complete = market.read(folder / "completed.json")
        native_path = folder / "native.json"
        need(complete.get("schema") == asr.POLICY and
             complete.get("cell_id") == cell_id and
             complete.get("provider_id") == provider_id and
             complete.get("wav_sha256") == pcm["wav_sha256"] and
             complete.get("native_sha256") == digest_file(native_path) and
             complete.get("status") == "BLIND_ASR_UNADMITTED" and
             complete.get("training_admitted") is False and
             complete.get("human_gold") is False and
             not (folder / "failed.json").exists(),
             "ASR_REVIEW_CELL_COMPLETION_DRIFT")
        native = blind.native_result(native_path, {
            "provider_id": provider_id, "model_id": model_id,
            "model_revision_sha": revision}, cell, pcm)
        results[cell_id] = {
            "text": native["actual_text"],
            "normalized": normalize_observation(native["actual_text"]),
            "native_sha256": complete["native_sha256"],
            "model_id": model_id, "model_revision_sha": revision}
    need(len(results) == asr.MAX_CALLS, "ASR_REVIEW_PROVIDER_INCOMPLETE")
    return results


def build_review(source_parent: Path, asr_parents: dict[str, Path]) -> dict:
    """Every source WAV/PCM and 189 native receipts must pass before reporting."""
    plan, cells = asr.source_from_artifact(source_parent)
    need(set(asr_parents) == set(asr.PROVIDERS), "ASR_REVIEW_THREE_FAMILIES_REQUIRED")
    provider_rows = {
        name: read_provider(asr_parents[name], name, plan, cells)
        for name in asr.PROVIDERS
    }
    rows = []
    counts = Counter()
    by_intent = {"wake_vosen": Counter(), "wake_mero": Counter(),
                 "near_miss": Counter(), "ordinary_negative": Counter()}
    by_voice: dict[str, Counter] = {}
    roles = {}
    for cell, _, pcm in cells:
        cid = cell["cell_id"]
        requested = normalize_observation(cell["text_requested"])
        observations = {p: provider_rows[p][cid] for p in asr.PROVIDERS}
        texts = [r["normalized"] for r in observations.values()]
        vote = Counter(texts)
        unanimous = len(vote) == 1
        target = all(t == requested for t in texts)
        status = ("ALL_THREE_MATCH_REQUEST" if target else
                  "ALL_THREE_AGREE_OTHER" if unanimous else
                  "TWO_AGREE_OTHER_OR_REQUEST" if max(vote.values()) == 2
                  else "THREE_DISAGREE")
        lane = ("WAKE_PRONUNCIATION_REVIEW" if cell["intent"] == "wake" else
                "NEAR_MISS_AMBIGUOUS_QUARANTINE" if cell["intent"] == "near_miss"
                else "ORDINARY_NEGATIVE_REVIEW")
        family = cell["source_family_id"]
        need(family not in roles or roles[family] == cell["role"],
             "ASR_REVIEW_SOURCE_FAMILY_SPLIT_LEAK")
        roles[family] = cell["role"]
        subtype = ("wake_vosen" if cell["requested_keyword_index"] == 1 else
                   "wake_mero") if cell["intent"] == "wake" else cell["intent"]
        counts["total"] += 1
        counts["three_way_unanimous"] += int(unanimous)
        counts["three_way_request_exact"] += int(target)
        counts["two_or_more_agree"] += int(max(vote.values()) >= 2)
        counts["three_way_completed_calls"] += 3
        for bucket in (by_intent[subtype], by_voice.setdefault(cell["voice_id"], Counter())):
            bucket["total"] += 1
            bucket["three_way_unanimous"] += int(unanimous)
            bucket["three_way_request_exact"] += int(target)
        rows.append({
            "cell_id": cid, "wav_sha256": pcm["wav_sha256"],
            "source_family_id": family, "voice_id": cell["voice_id"],
            "role": cell["role"], "requested_text": cell["text_requested"],
            "requested_intent": cell["intent"], "requested_keyword_index": cell["requested_keyword_index"],
            "challenge": cell["challenge"], "asr": observations,
            "machine_agreement": status, "review_lane": lane,
            "acoustic_pronunciation_verified": False,
            "human_gold": False, "training_admitted": False,
            "trusted_negative": False, "independent_test_qualified": False,
        })
    need(len(roles) == 7 and
         {k for k,v in roles.items() if v == "development-only"} ==
         {"qwen3-preset-serena", "qwen3-preset-uncle_fu"} and
         len({r["wav_sha256"] for r in rows}) == len(rows),
         "ASR_REVIEW_SOURCE_SPLIT_OR_DUPLICATE_WAV")
    source_rows = market.read(source_parent / asr.ARTIFACT_ROOT / "summary.json")["source_rows"]
    need(len({r["pcm_sha256"] for r in source_rows}) == len(rows),
         "ASR_REVIEW_DUPLICATE_SOURCE_PCM")
    need(counts["total"] == 63 and counts["three_way_completed_calls"] == 189,
         "ASR_REVIEW_COMPLETION_COUNT")
    return {
        "schema": POLICY, "status": "MACHINE_REVIEW_COMPLETE_NOT_ADMITTED",
        "source_tts_run_id": asr.SOURCE_RUN_ID,
        "source_tts_artifact_id": ARCHIVES["tts"][0],
        "three_asr_run_id": ASR_RUN_ID, "three_asr_head_sha": ASR_HEAD_SHA,
        "artifact_pins": {k: {"id": v[0], "zip_sha256": v[1]} for k,v in ARCHIVES.items()},
        "model_calls_performed_by_this_audit": 0,
        "counts": dict(counts),
        "by_intent": {k: dict(v) for k,v in by_intent.items()},
        "by_voice": {k: dict(v) for k,v in by_voice.items()},
        "rows": rows, "human_gold": False,
        "training_admitted": False, "acoustic_gold_admitted": 0,
        "independent_real_human_far_frr": "NOT_RUN",
        "physical_ssc305": "NOT_RUN",
    }


def audit_zip_files(paths: dict[str, Path]) -> dict:
    need(set(paths) == set(ARCHIVES), "ASR_REVIEW_FOUR_ARCHIVES_REQUIRED")
    with tempfile.TemporaryDirectory(prefix="en-asr63-") as tmp:
        tmp_root = Path(tmp)
        for name, (_, sha) in ARCHIVES.items():
            (tmp_root / name).mkdir(mode=0o700)
            unpack_pinned(paths[name], sha, tmp_root / name)
        return build_review(
            tmp_root / "tts",
            {p: tmp_root / p for p in asr.PROVIDERS})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARCHIVES:
        parser.add_argument("--" + name + "-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = {n: getattr(args, n + "_zip") for n in ARCHIVES}
        report = audit_zip_files(paths)
        dest = args.output
        blind.private_path(dest, "ASR_REVIEW_OUTPUT_NOT_PRIVATE")
        need(dest.is_absolute() and dest.parent.is_dir() and not dest.exists(),
             "ASR_REVIEW_OUTPUT_CREATE_ONLY")
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
        dest.chmod(0o600)
        print(json.dumps({"status":report["status"],"counts":report["counts"],
                          "by_intent":report["by_intent"],
                          "training_admitted":False},sort_keys=True))
        return 0
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print("ENGLISH_63_ASR_REVIEW_STOP: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
