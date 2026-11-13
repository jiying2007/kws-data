"""One-shot, real Sherpa-ONNX pretrained English-63 event probe.

Uses already-verified original 63 synthetic PCM16 WAVs; no TTS, ASR,
new training, promotion, or true FAR/FRR calculation. The corpus is not
independent acoustic gold. A separate model is evaluated per invocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_seven_voice_asr_once as source
import english_seven_voice_review as review
import english_seven_voice_tts_once as seven
import multilingual_campaign as market

POLICY = "english63-pretrained-zipformer-event-probe-v1"
TRIGGER = (Path(__file__).resolve().parents[1] /
           ".github/triggers/english63-zipformer-event-probe-20261113.json")
ALIASES = ("HEY_VOSEN", "HEY_MERO")
MODEL_SPECS = {
    "gigaspeech": {
        "directory": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
        "asset_id": 166405338, "archive_bytes": 17626723,
        "archive_sha256": None,  # Upstream release metadata publishes no digest.
        "epoch": 12, "tokens_type": "bpe",
        "source_integrity": "EXACT_RELEASE_ASSET_ID_AND_SIZE_ONLY_NO_UPSTREAM_DIGEST",
    },
    "zh-en": {
        "directory": "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
        "asset_id": 331699066, "archive_bytes": 32885699,
        "archive_sha256": "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6",
        "epoch": 13, "tokens_type": "phone+ppinyin",
        "source_integrity": "RELEASE_SHA256_PINNED",
    },
}
WAVS = 63
SAMPLE_RATE = 16000
FEED_SAMPLES = 1600
TAIL_SAMPLES = 10560  # Fixed 660 ms end-of-clip flush; never used as negative test.
EVENT_LIMIT = 16


def need(value, error):
    if not value:
        raise ValueError(error)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_claim():
    expected = {
        "schema": POLICY, "source_run_id": source.SOURCE_RUN_ID,
        "source_artifact_id": review.ARCHIVES["tts"][0],
        "source_zip_sha256": review.ARCHIVES["tts"][1],
        "source_clips": WAVS, "keywords": ["Hey Vosen", "Hey Mero"],
        "models": {name: {"asset_id": spec["asset_id"],
                           "archive_bytes": spec["archive_bytes"],
                           "sha256": spec["archive_sha256"]}
                   for name, spec in MODEL_SPECS.items()},
        "fixed_keywords_score": 1.0, "fixed_keywords_threshold": 0.25,
        "fixed_trailing_blanks": 1, "fixed_max_paths": 4,
        "new_tts_calls": 0, "new_asr_calls": 0, "optimizer_updates": 0,
        "formal_training_admitted": False, "independent_human_gold": False,
        "model_promotion_allowed": False, "scheduled_patrol": False,
    }
    need(market.read(TRIGGER) == expected, "EN_ZIPFORMER_FROZEN_TRIGGER_DRIFT")
    return expected


def gate():
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_EVENT_NAME") == "push" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and
         os.environ.get("GITHUB_REPOSITORY_OWNER") == "jiying2007",
         "EN_ZIPFORMER_PROTECTED_MAIN_ONCE_ONLY")


def extract_locked_model(archive: Path, model_name: str, root: Path):
    """Reject archive replacement/traversal/links before extracting any weight."""
    need(model_name in MODEL_SPECS, "EN_ZIPFORMER_UNKNOWN_MODEL")
    spec = MODEL_SPECS[model_name]
    need(archive.is_absolute() and archive.is_file() and
         not archive.is_symlink() and
         archive.stat().st_size == spec["archive_bytes"],
         "EN_ZIPFORMER_ARCHIVE_BYTES_DRIFT")
    digest = sha_file(archive)
    if spec["archive_sha256"]:
        need(digest == spec["archive_sha256"], "EN_ZIPFORMER_ARCHIVE_SHA_DRIFT")
    need(root.is_dir() and not root.is_symlink(), "EN_ZIPFORMER_OUTPUT_ROOT")
    with tarfile.open(archive, "r:bz2") as tar:
        members = tar.getmembers()
        need(0 < len(members) <= 600 and
             sum(m.size for m in members) <= 180 * 1024 * 1024,
             "EN_ZIPFORMER_TAR_BUDGET")
        names = set()
        for member in members:
            p = PurePosixPath(member.name)
            need(not p.is_absolute() and p.parts and
                 p.parts[0] == spec["directory"] and
                 all(part not in ("", ".", "..") for part in p.parts) and
                 member.name not in names and
                 (member.isfile() or member.isdir()) and
                 0 <= member.size <= 40 * 1024 * 1024,
                 "EN_ZIPFORMER_TAR_UNSAFE")
            names.add(member.name)
        for member in members:
            dest = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(member)
                need(f is not None, "EN_ZIPFORMER_TAR_MEMBER")
                with f, dest.open("xb") as target:
                    shutil.copyfileobj(f, target, 1024 * 1024)
                need(dest.stat().st_size == member.size,
                     "EN_ZIPFORMER_TAR_SHORT_WRITE")
    model_dir = root / spec["directory"]
    stem = "epoch-" + str(spec["epoch"]) + "-avg-2-chunk-16-left-64"
    parts = {
        "encoder": model_dir / ("encoder-" + stem + ".onnx"),
        "decoder": model_dir / ("decoder-" + stem + ".onnx"),
        "joiner": model_dir / ("joiner-" + stem + ".onnx"),
        "tokens": model_dir / "tokens.txt",
    }
    if model_name == "gigaspeech":
        parts["bpe"] = model_dir / "bpe.model"
    else:
        parts["lexicon"] = model_dir / "en.phone"
    need(all(p.is_file() and not p.is_symlink() and 0 < p.stat().st_size
             for p in parts.values()), "EN_ZIPFORMER_MODEL_COMPONENT_MISSING")
    return parts, digest


def generate_keywords(model_name: str, model_parts: dict, parent: Path):
    """Official text2token, no manually guessed proper-name pronunciation."""
    need(model_name in MODEL_SPECS, "EN_ZIPFORMER_KEYWORDS_MODEL")
    raw = parent / "keywords.raw.txt"
    out = parent / "keywords.txt"
    if model_name == "gigaspeech":
        raw.write_text("HEY VOSEN\nHEY MERO\n", encoding="utf-8")
    else:
        raw.write_text("HEY VOSEN @HEY_VOSEN\nHEY MERO @HEY_MERO\n",
                       encoding="utf-8")
    command = [
        "sherpa-onnx-cli", "text2token", "--tokens", str(model_parts["tokens"]),
        "--tokens-type", MODEL_SPECS[model_name]["tokens_type"],
    ]
    if model_name == "gigaspeech":
        command += ["--bpe-model", str(model_parts["bpe"])]
    else:
        command += ["--lexicon", str(model_parts["lexicon"])]
    command += [str(raw), str(out)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=40)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("EN_ZIPFORMER_TEXT2TOKEN_UNAVAILABLE") from exc
    need(result.returncode == 0 and out.is_file(),
         "EN_ZIPFORMER_TEXT2TOKEN_FAILED_PRONUNCIATION_UNVERIFIED: " +
         result.stderr[-600:])
    lines = [x.strip() for x in out.read_text(encoding="utf-8").splitlines()
             if x.strip()]
    need(len(lines) == 2, "EN_ZIPFORMER_KEYWORD_ROSTER")
    if model_name == "gigaspeech":
        need(all("@" not in line for line in lines),
             "EN_ZIPFORMER_BPE_UNEXPECTED_ALIAS")
        out.write_text("\n".join(line + " @" + ALIASES[i]
                                 for i, line in enumerate(lines)) + "\n",
                       encoding="utf-8")
    else:
        need(all(line.endswith(" @" + ALIASES[i])
                 for i, line in enumerate(lines)),
             "EN_ZIPFORMER_PHONEME_ALIAS_DRIFT")
    return out


def preflight_model(archive: Path, model_name: str):
    """Bounded real pretrained tokenizer probe; never touches original audio."""
    frozen_claim()
    need(model_name in MODEL_SPECS, "EN_ZIPFORMER_UNKNOWN_MODEL")
    with tempfile.TemporaryDirectory(prefix="en63-pronunciation-preflight-") as tmp:
        root = Path(tmp)
        parts, archive_sha = extract_locked_model(archive, model_name, root)
        # The official zh-en model's English names require its own en.phone
        # lexicon entries. A missing proper name is an explicit blocked
        # research candidate, never a guessed phonetic transcription.
        if model_name == "zh-en":
            with parts["lexicon"].open(encoding="utf-8") as handle:
                words = {line.split()[0] for line in handle if line.strip()}
            missing = [word for word in ("HEY", "VOSEN", "MERO")
                       if word not in words]
            if missing:
                return {
                    "schema": "english63-pretrained-tokenization-preflight-v1",
                    "model": model_name,
                    "status": "BLOCKED_OFFICIAL_LEXICON_PROPER_NAME_OOV",
                    "model_archive_sha256": archive_sha,
                    "model_source_integrity": MODEL_SPECS[model_name]["source_integrity"],
                    "missing_official_lexicon_words": missing,
                    "keyword_lines": [],
                    "keyword_sha256": None, "tokenization_complete": False,
                    "source_audio_read": False, "new_tts_calls": 0,
                    "new_asr_calls": 0, "kws_inference_calls": 0,
                    "training_admitted": False, "model_promotion_allowed": False,
                }
        keywords = generate_keywords(model_name, parts, root)
        lines = keywords.read_text(encoding="utf-8").splitlines()
        need(len(lines) == len(ALIASES) and
             all(lines[i].endswith(" @" + ALIASES[i]) for i in range(len(ALIASES))),
             "EN_ZIPFORMER_PREFLIGHT_TOKEN_COUNT_OR_ALIAS")
        return {
            "schema": "english63-pretrained-tokenization-preflight-v1",
            "model": model_name, "status": "TWO_KEYWORDS_ENCODED_NO_AUDIO",
            "model_archive_sha256": archive_sha,
            "model_source_integrity": MODEL_SPECS[model_name]["source_integrity"],
            "keyword_lines": lines,
            "keyword_sha256": sha_file(keywords),
            "tokenization_complete": True,
            "source_audio_read": False, "new_tts_calls": 0,
            "new_asr_calls": 0, "kws_inference_calls": 0,
            "training_admitted": False, "model_promotion_allowed": False,
        }


def read_pcm(wav: Path):
    import numpy as np
    with wave.open(str(wav), "rb") as f:
        count = f.getnframes()
        need(f.getnchannels() == 1 and f.getsampwidth() == 2 and
             f.getframerate() == SAMPLE_RATE and f.getcomptype() == "NONE" and
             400 <= count <= SAMPLE_RATE * 120,
             "EN_ZIPFORMER_PCM_GEOMETRY")
        data = f.readframes(count)
        need(len(data) == count * 2 and not f.readframes(1),
             "EN_ZIPFORMER_PCM_SHORT_READ")
    values = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return values, hashlib.sha256(data).hexdigest()


def normalize_event_keyword(keyword: str) -> str:
    need(isinstance(keyword, str) and 0 < len(keyword) < 256,
         "EN_ZIPFORMER_KEYWORD_RESULT_INVALID")
    simplified = keyword.upper().strip().replace(" ", "_")
    return simplified if simplified in ALIASES else "UNEXPECTED_KEYWORD"


def stream_events(spotter, samples):
    import numpy as np
    stream = spotter.create_stream()
    events = []
    decoded = 0
    supplied = 0

    def drain():
        nonlocal decoded
        while spotter.is_ready(stream):
            decoded += 1
            need(decoded < 20000, "EN_ZIPFORMER_UNBOUNDED_DECODER")
            spotter.decode_stream(stream)
            keyword = spotter.get_result(stream)
            if keyword:
                tokens = spotter.tokens(stream)
                timestamps = [float(t) for t in spotter.timestamps(stream)]
                need(all(math.isfinite(t) and t >= 0.0 for t in timestamps) and
                     len(tokens) == len(timestamps),
                     "EN_ZIPFORMER_EVENT_TIMESTAMPS")
                need(len(events) < EVENT_LIMIT, "EN_ZIPFORMER_TOO_MANY_EVENTS")
                events.append({
                    "keyword": keyword,
                    "mapped_keyword": normalize_event_keyword(keyword),
                    "token_timestamps_sec": timestamps,
                    "tokens": list(tokens),
                    "available_after_input_sec": round(supplied / SAMPLE_RATE, 4),
                    "is_streaming_event": True,
                })
                spotter.reset_stream(stream)

    for i in range(0, len(samples), FEED_SAMPLES):
        chunk = samples[i:i + FEED_SAMPLES]
        stream.accept_waveform(SAMPLE_RATE, chunk)
        supplied += len(chunk)
        drain()
    # Deterministic per-clip flush, NOT a continuous-background experiment.
    tail = np.zeros(TAIL_SAMPLES, dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, tail)
    supplied += TAIL_SAMPLES
    drain()
    stream.input_finished()
    drain()
    return events, decoded


def validate_result(result):
    need(type(result) is dict and result.get("schema") == POLICY and
         result.get("status") == "EXECUTED_REAL_PRETRAINED_SYNTHETIC_CLIP_EVENTS" and
         result.get("model_name") in MODEL_SPECS and
         result.get("source_wav_count") == WAVS and
         type(result.get("clip_results")) is list and
         len(result["clip_results"]) == WAVS and
         len({r.get("cell_id") for r in result["clip_results"]}) == WAVS and
         result.get("new_tts_calls") == result.get("new_asr_calls") ==
         result.get("optimizer_updates") == 0 and
         result.get("model_promotion_allowed") is False and
         result.get("formal_training_admitted") is False and
         result.get("independent_human_gold") is False and
         result.get("continuous_negative_far_per_hour") == "NOT_RUN" and
         result.get("true_frr") == "NOT_RUN" and
         result.get("ssc305_profile") == "NOT_RUN",
         "EN_ZIPFORMER_FALSE_PROMOTION_OR_RECEIPT")
    for row in result["clip_results"]:
        need(row.get("training_admitted") is False and
             row.get("human_gold") is False and
             row.get("trusted_negative") is False and
             isinstance(row.get("events"), list),
             "EN_ZIPFORMER_CELL_FALSE_LABEL")
    return True


def execute(tts_zip: Path, archive: Path, model_name: str):
    frozen_claim()
    gate()
    need(model_name in MODEL_SPECS, "EN_ZIPFORMER_UNKNOWN_MODEL")
    import numpy as np
    import sherpa_onnx
    with tempfile.TemporaryDirectory(prefix="en63-zipformer-real-") as tmp:
        root = Path(tmp)
        source_parent = root / "tts-source"
        source_parent.mkdir(mode=0o700)
        review.unpack_pinned(tts_zip, review.ARCHIVES["tts"][1], source_parent)
        plan, cells = source.source_from_artifact(source_parent)
        need(len(cells) == WAVS and
             plan.get("wake_phrases_requested") == ["Hey Vosen", "Hey Mero"] and
             len({c["cell_id"] for c, _, _ in cells}) == WAVS,
             "EN_ZIPFORMER_SOURCE_ROSTER_DRIFT")
        model_root = root / "model"
        model_root.mkdir(mode=0o700)
        model_parts, archive_sha = extract_locked_model(archive, model_name,
                                                         model_root)
        keywords_path = generate_keywords(model_name, model_parts, root)
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(model_parts["tokens"]),
            encoder=str(model_parts["encoder"]),
            decoder=str(model_parts["decoder"]),
            joiner=str(model_parts["joiner"]),
            keywords_file=str(keywords_path),
            num_threads=1, sample_rate=SAMPLE_RATE, feature_dim=80,
            max_active_paths=4, keywords_score=1.0, keywords_threshold=0.25,
            num_trailing_blanks=1, provider="cpu")
        rows, groups = [], Counter()
        total_seconds = 0.0
        start = time.perf_counter()
        for cell, wav, pcm in sorted(cells, key=lambda e: e[0]["cell_id"]):
            samples, checked_pcm = read_pcm(wav)
            need(checked_pcm == pcm["pcm_sha256"],
                 "EN_ZIPFORMER_PCM_SHA_DRIFT")
            events, steps = stream_events(spotter, samples)
            total_seconds += len(samples) / SAMPLE_RATE
            groups[cell["intent"]] += len(events)
            rows.append({
                "cell_id": cell["cell_id"],
                "wav_sha256": pcm["wav_sha256"],
                "pcm_sha256": checked_pcm,
                "voice_source_family": cell["source_family_id"],
                "voice_role": cell["role"],
                "requested_text": cell["text_requested"],
                "requested_intent": cell["intent"],
                "requested_keyword_index": cell["requested_keyword_index"],
                "audio_samples": len(samples),
                "decoder_steps": steps, "events": events,
                "training_admitted": False, "human_gold": False,
                "trusted_negative": False,
            })
        elapsed = time.perf_counter() - start
        result = {
            "schema": POLICY,
            "status": "EXECUTED_REAL_PRETRAINED_SYNTHETIC_CLIP_EVENTS",
            "model_name": model_name,
            "model_archive_sha256": archive_sha,
            "model_source_integrity": MODEL_SPECS[model_name]["source_integrity"],
            "release_tag": "kws-models",
            "release_asset_id": MODEL_SPECS[model_name]["asset_id"],
            "original_tts_run_id": source.SOURCE_RUN_ID,
            "original_tts_artifact_id": review.ARCHIVES["tts"][0],
            "original_tts_zip_sha256": review.ARCHIVES["tts"][1],
            "source_wav_count": WAVS,
            "keyword_file_sha256": sha_file(keywords_path),
            "actual_keywords_lines": keywords_path.read_text(encoding="utf-8").splitlines(),
            "sherpa_onnx_version": version("sherpa-onnx"),
            "fixed_decode": {"num_threads": 1, "features": 80, "max_active_paths": 4,
                             "keywords_score": 1.0, "keywords_threshold": 0.25,
                             "num_trailing_blanks": 1, "pcm_feed_samples": FEED_SAMPLES,
                             "end_padding_samples": TAIL_SAMPLES},
            "host_observed_runtime_sec": round(elapsed, 3),
            "synthetic_audio_duration_sec": round(total_seconds, 3),
            "host_observed_rtf_including_python_io": round(elapsed / total_seconds, 5),
            "event_count_by_requested_intent_not_truth": dict(groups),
            "clip_results": rows,
            "new_tts_calls": 0, "new_asr_calls": 0, "optimizer_updates": 0,
            "formal_training_admitted": False, "independent_human_gold": False,
            "model_promotion_allowed": False,
            "continuous_negative_far_per_hour": "NOT_RUN",
            "true_frr": "NOT_RUN", "ssc305_profile": "NOT_RUN",
            "independent_accent_eval": "NOT_RUN",
            "observation": "Per-clip real streaming keyword events on synthetic source prompts only; not continuous FAR/FRR.",
        }
        validate_result(result)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight-model", choices=sorted(MODEL_SPECS))
    p.add_argument("--tts-zip", type=Path)
    p.add_argument("--model-archive", type=Path)
    p.add_argument("--model", choices=sorted(MODEL_SPECS))
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    try:
        frozen_claim()
        need(not (a.execute and a.preflight_model),
             "EN_ZIPFORMER_MODE_CONFLICT")
        if a.preflight_model:
            need(a.model_archive is not None and a.model is None and
                 a.tts_zip is None and a.output is None,
                 "EN_ZIPFORMER_PREFLIGHT_MODEL_ONLY")
            print(json.dumps(preflight_model(a.model_archive, a.preflight_model),
                             sort_keys=True))
            return 0
        if not a.execute:
            print(json.dumps({"schema": POLICY, "models": sorted(MODEL_SPECS),
                              "original_audio_reused": WAVS,
                              "new_tts_calls": 0, "model_promotion_allowed": False}))
            return 0
        need(all(x is not None for x in (a.tts_zip, a.model_archive,
                                         a.model, a.output)),
             "EN_ZIPFORMER_EXPLICIT_INPUTS_REQUIRED")
        need(a.output.is_absolute() and a.output.parent.is_dir() and
             not a.output.parent.is_symlink() and not a.output.exists(),
             "EN_ZIPFORMER_CREATE_ONLY_OUTPUT")
        result = execute(a.tts_zip, a.model_archive, a.model)
        with a.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
        a.output.chmod(0o600)
        print(json.dumps({"status": result["status"],
                          "model": result["model_name"],
                          "clips": result["source_wav_count"],
                          "event_count_by_requested_intent_not_truth":
                          result["event_count_by_requested_intent_not_truth"],
                          "promotion": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, tarfile.TarError, wave.Error) as exc:
        print("EN_ZIPFORMER_REAL_STOP: " + type(exc).__name__ +
              ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
