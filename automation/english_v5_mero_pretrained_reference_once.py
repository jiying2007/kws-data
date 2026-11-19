"""One bounded real English-pretrained reference for the V5 Hey Mero research model.

Reuse only the already authenticated, source-disjoint V5 train/development WAVs.
No historical second-target labels are reclassified, no audio is regenerated,
and an exposed development score cannot authorize model/threshold promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import english_v5_mero_fsmn_ctc_once as v5
import english63_zipformer_events_once as pretrained

SCHEMA = "english-v5-mero-pretrained-source-disjoint-baseline-v1"
REPO = "jiying2007/kws-data"
BRANCH = "research/english-v5-mero-pretrained-reference-20261119"
MODEL_SHA256 = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
MODEL_BYTES = 32885699
KEYWORD = "HEY_MERO"
FEED_SAMPLES = 1600
TAIL_SAMPLES = 10560


def need(ok, code):
    if not ok:
        raise ValueError(code)


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_contract():
    state = v5.source_check()
    spec = pretrained.MODEL_SPECS["zh-en"]
    need(state["active_target_ids"] == ["hey_mero"] and
         state["positive_ctc"] == [1, 2, 3] and
         state["old_locked_reopened"] is False and
         v5.config()["target"]["text"] == "Hey Mero" and
         len(v5.ARCHIVES) == 8 and
         spec["archive_bytes"] == MODEL_BYTES and
         spec["tokens_type"] == "phone+ppinyin" and
         spec["archive_sha256"] == MODEL_SHA256 and
         pretrained.SAMPLE_RATE == 16000 and
         pretrained.FEED_SAMPLES == FEED_SAMPLES and
         pretrained.TAIL_SAMPLES == TAIL_SAMPLES,
         "V5_SINGLE_MERO_PRETRAINED_SOURCE_CONTRACT")
    return {"schema": SCHEMA, "status": "SOURCE_ONLY_READY_NOT_EXECUTED",
            "keyword_ids": ["hey_mero"], "reference_model": "zh-en",
            "model_sha256": MODEL_SHA256,
            "original_source_artifact_ids": v5.ARCHIVES,
            "source_splits": ["train", "development"],
            "new_audio_generation": 0, "optimizer_updates": 0,
            "calibration_or_locked_access": False,
            "model_promotion_allowed": False, "ssc305": "NOT_RUN"}


def run_guard(env=None, event=None):
    env = os.environ if env is None else env
    need(all(env.get(k) == expected for k, expected in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "pull_request", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_HEAD_REF": BRANCH}.items()), "PR_ONLY_ONE_ATTEMPT")
    if event is None:
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request", {})
    need(event.get("repository", {}).get("full_name") == REPO and
         event.get("repository", {}).get("private") is False and
         pr.get("head", {}).get("repo", {}).get("full_name") == REPO and
         pr.get("head", {}).get("ref") == BRANCH and
         pr.get("base", {}).get("ref") == "main" and
         event.get("action") in ("opened", "synchronize", "reopened"),
         "EXACT_PUBLIC_SAME_REPO_PR_REQUIRED")
    return {"run_id": env.get("GITHUB_RUN_ID"),
            "head_sha": pr["head"]["sha"], "branch": BRANCH}


def make_mero_keyword(parts, folder, tokenizer=None):
    """One verified official phone-lexicon HEY MERO; no ungrounded G2P."""
    lexicon = parts["lexicon"]
    need(lexicon.is_file() and not lexicon.is_symlink()
         and 0 < lexicon.stat().st_size <= 6*1024*1024,
         "OFFICIAL_PHONE_LEXICON_FILE")
    words = {v[0] for line in lexicon.read_text(encoding="utf-8").splitlines()
             if len(v := line.strip().split()) >= 2}
    need({"HEY", "MERO"} <= words, "OFFICIAL_MERO_LEXICON_WORDS")
    if tokenizer is None:
        from sherpa_onnx import text2token as tokenizer
    encoded = tokenizer(["HEY MERO"], tokens=str(parts["tokens"]),
                        tokens_type="phone+ppinyin", bpe_model=None,
                        lexicon=str(lexicon))
    need(type(encoded) is list and len(encoded) == 1
         and type(encoded[0]) is list and len(encoded[0]) >= 2
         and all(type(p) is str and p and not p.startswith(("@", "#", ":"))
                 for p in encoded[0]), "OFFICIAL_PHONE_ENCODING")
    vocab = {v[0] for line in parts["tokens"].read_text(encoding="utf-8").splitlines()
             if (v := line.strip().split())}
    need(vocab and all(p in vocab for p in encoded[0]),
         "OFFICIAL_PHONE_VOCAB_MEMBERSHIP")
    path = folder / "mero.keyword.txt"
    path.write_text(" ".join(encoded[0]) + " @" + KEYWORD + "\n", encoding="utf-8")
    return path


def summarize(rows):
    result = {}
    for role in ("train", "development"):
        part = [x for x in rows if x["role"] == role]
        pos = [x for x in part if x["mono_class"] == 1]
        neg = [x for x in part if x["mono_class"] == 0]
        need(pos and neg and
             len({x["source_group_id"] for x in part}) >=
             (6 if role == "train" else 2), "SOURCE_DISJOINT_ROLE_COVERAGE")
        result[role] = {
            "source_groups": len({x["source_group_id"] for x in part}),
            "mero_positive_original_clips": len(pos),
            "mero_positive_detected_clips": sum(bool(x["events"]) for x in pos),
            "ordinary_original_clips": len(neg),
            "ordinary_triggered_clips": sum(bool(x["events"]) for x in neg),
            "ordinary_total_events": sum(len(x["events"]) for x in neg),
            "original_clips_only": True, "fixed_threshold": 0.25}
    need(not ({x["source_group_id"] for x in rows if x["role"] == "train"} &
              {x["source_group_id"] for x in rows if x["role"] == "development"}),
         "TRAIN_DEVELOPMENT_SOURCE_LEAKAGE")
    need(len({x["id"] for x in rows}) == len(rows),
         "DUPLICATE_REVIEWED_AUDIO_ID")
    return result


def write_once(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2,
                  ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def execute(sources, model_archive, output):
    import numpy as np
    import sherpa_onnx
    identity = run_guard()
    source_contract()
    need(output.is_absolute() and output.parent.is_dir() and
         not output.exists() and not output.parent.is_symlink(),
         "CREATE_ONCE_EVIDENCE_DIR")
    output.mkdir()
    write_once(output / "STARTED.json", {
        "schema": SCHEMA, "identity": identity,
        "model_reference_sha256": MODEL_SHA256, "retry": False})
    opened = {}
    try:
        rows, audit = v5.admission(sources)
        need(rows and all(r["role"] in ("train", "development") and
             r["mono_class"] in (0, 1) for r in rows),
             "ONLY_AUTHENTICATED_V5_MERO_AND_ORDINARY")
        write_once(output / "ADMISSION.json",
                   {"schema": SCHEMA, "summary": audit,
                    "selected_audio": [{"id": r["id"], "role": r["role"],
                        "source_group_id": r["source_group_id"],
                        "pcm_sha256": r["source_pcm_sha256"],
                        "mono_class": r["mono_class"]} for r in rows]})
        need(model_archive.is_absolute() and
             model_archive.stat().st_size == MODEL_BYTES and
             hash_file(model_archive) == MODEL_SHA256,
             "EXACT_OFFICIAL_GIGASPEECH_MODEL_BYTES")
        with tempfile.TemporaryDirectory(prefix="v5-mero-pretrained-") as temp:
            model_root = Path(temp) / "model"
            model_root.mkdir()
            parts, digest = pretrained.extract_locked_model(
                model_archive, "zh-en", model_root)
            need(digest == MODEL_SHA256, "PRETRAINED_ARCHIVE_HASH")
            keyword = make_mero_keyword(parts, Path(temp))
            spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(parts["tokens"]), encoder=str(parts["encoder"]),
                decoder=str(parts["decoder"]), joiner=str(parts["joiner"]),
                keywords_file=str(keyword), num_threads=1, sample_rate=16000,
                feature_dim=80, max_active_paths=4, keywords_score=1.0,
                keywords_threshold=0.25, num_trailing_blanks=1,
                provider="cpu")
            for spec in v5.binding.config()["corpus"]:
                opened[spec["key"]] = v5.binding.archive(
                    sources / (spec["key"] + "-source.zip"), spec["source"])
            opened["newtrain24"] = v5.binding.archive(
                sources / "newtrain24-source.zip", v5.prior.SOURCE)
            observations = []
            total_samples = 0
            began = time.perf_counter()
            for r in sorted(rows, key=lambda row: (row["role"], row["id"])):
                pcm = v5.base.load_source_pcm(r, opened)
                need(type(pcm) is bytes and len(pcm) == 2 * r["samples"]
                     and hashlib.sha256(pcm).hexdigest() ==
                     r["source_pcm_sha256"], "ORIGINAL_PCM_HASH_AND_GEOMETRY")
                samples = (np.frombuffer(pcm, dtype="<i2")
                           .astype(np.float32) / 32768.0)
                events, steps = pretrained.stream_events(spotter, samples)
                need(all(x["mapped_keyword"] == KEYWORD for x in events),
                     "ONLY_MERO_EVENT_ALLOWED")
                observations.append({
                    "id": r["id"], "source_group_id": r["source_group_id"],
                    "pcm_sha256": r["source_pcm_sha256"],
                    "role": r["role"], "mono_class": r["mono_class"],
                    "samples": r["samples"], "decoder_steps": steps,
                    "events": events})
                total_samples += len(samples)
            elapsed = time.perf_counter() - began
            report = {"schema": SCHEMA,
                "status": "REAL_MERO_PRETRAINED_REFERENCE_REPLAY_COMPLETE",
                "pretrained_model": "zh-en",
                "keyword_ids": ["hey_mero"], "keyword_file_sha256":
                    hash_file(keyword), "model_archive_sha256": digest,
                "sherpa_onnx_version": version("sherpa-onnx"),
                "fixed_decoding": {"max_active_paths": 4, "score": 1.0,
                    "threshold": 0.25, "trailing_blanks": 1,
                    "feed_samples": FEED_SAMPLES, "tail_samples": TAIL_SAMPLES},
                "reviewed_source": audit, "summary": summarize(observations),
                "clips": observations, "elapsed_host_seconds": round(elapsed, 4),
                "pcm_seconds": round(total_samples / 16000, 4),
                "host_python_rtf": round(elapsed * 16000 / total_samples, 5),
                "optimizer_updates": 0, "new_tts_asr_calls": 0,
                "model_weights_copied": False,
                "model_promotion_allowed": False,
                "no_calibration_or_locked_test": True,
                "development_is_exposed_and_synthetic": True,
                "independent_far_per_hour": "NOT_MEASURED",
                "independent_human_frr": "NOT_MEASURED",
                "ssc305_physical": "NOT_RUN"}
            write_once(output / "RESULT.json", report)
            print(json.dumps({"status": report["status"],
                 "summary": report["summary"], "research_only": True},
                 sort_keys=True))
    except BaseException as exc:
        write_once(output / "STOP.json", {"schema": SCHEMA,
                   "status": "STOP_NO_MERO_PRETRAINED_QUALITY_PROOF",
                   "error_type": type(exc).__name__,
                   "error": str(exc)[:450], "retry": False})
        raise
    finally:
        for item in opened.values():
            item.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--model-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(args.source_check != args.execute, "EXACT_ONE_MODE")
    if args.source_check:
        print(json.dumps(source_contract(), sort_keys=True))
    else:
        need(args.sources and args.model_archive and args.output,
             "EXPLICIT_SAVED_INPUTS_AND_OUTPUT")
        execute(args.sources, args.model_archive, args.output)


if __name__ == "__main__":
    main()
