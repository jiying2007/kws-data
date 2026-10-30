"""Fixed, not-yet-executed corpus campaign. Standard library only.

This module describes new *unadmitted* audio, never labels or qualification.
It does not import a model, perform downloads, or alter historical evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
import wave
from typing import Any

CAMPAIGN = "qwen16-intake-20261030-v1"
REPOSITORY = "jiying2007/kws-data"
DONOR_REPOSITORY = "jiying2007/kws-pipeline"
DONOR_COMMIT = "983e8082f712bd5099feb32bfb099e9f6df22d3a"
EXECUTION_BRANCH = "research/run-qwen16-intake-20261030-v1"
DATA_BASE_COMMIT = "dbf8aa7c1d04f785874c1d089b1646348308af3b"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
VOICES = ("Vivian", "Serena", "Uncle_Fu", "Dylan")
SPLIT_BY_VOICE = {"Vivian": "train", "Serena": "development_a",
                  "Uncle_Fu": "train", "Dylan": "train"}
TEXTS = ((1, "你好小窝"), (2, "小窝小窝"))
RECIPE = {
    "language": "Chinese", "instruct": "", "do_sample": True,
    "top_k": 50, "top_p": 1.0, "temperature": 0.9,
    "repetition_penalty": 1.05, "subtalker_dosample": True,
    "subtalker_top_k": 50, "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9, "max_new_tokens": 120,
    "non_streaming_mode": True,
}
BUDGET = {"model_loads": 1, "max_tts_calls": 16, "max_asr_calls": 0,
          "max_training_runs": 0, "retries": 0, "cpu": 4,
          "memory_bytes": 12 * 1024**3, "pids": 256,
          "setup_seconds": 1200, "generation_stage_seconds": 4800,
          "per_cell_seconds": 300, "artifact_bytes": 32 * 1024**2,
          "monetary_cap": 0}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, limit: int = 1024 * 1024) -> Any:
    require(not path.is_symlink() and path.is_file(), "regular JSON required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "JSON exceeds limit")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def nonfinite(_: str) -> None:
        raise ValueError("nonfinite JSON")
    def finite_float(value: str) -> float:
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON float")
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite,
                      parse_float=finite_float)


def save_json(path: Path, value: Any) -> None:
    """Atomic replacement of one owned JSON file, not a multi-file transaction."""
    raw = canonical(value) + b"\n"
    # A prior interrupted write must not block final failure evidence. Never
    # remove someone else's fixed-name temporary or follow a preexisting alias.
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for voice in VOICES:
        for keyword_id, text in TEXTS:
            for repeat in range(2):
                rows.append({
                    "cell_id": f"qwen16-20261030-{len(rows):03d}",
                    "speaker": voice, "seed": 20261030 + repeat,
                    "repeat": repeat, "intended_text": text,
                    "intended_keyword_id": keyword_id,
                    "proposed_split": SPLIT_BY_VOICE[voice],
                    # No seed/batch-local family to conceal cross-batch reuse.
                    "source_family_id": f"qwen3-customvoice:preset:{voice.lower()}",
                    "synthetic": True, "source_type": "tts",
                    "actual_text": None, "actual_tokens": None,
                    "label_status": "PENDING_REVIEW",
                    "ctc_training_allowed": False,
                    "evaluation_gold_allowed": False,
                    "qualification_allowed": False,
                })
    return rows


def validate_plan(value: Any) -> list[dict[str, Any]]:
    # Canonical bytes also distinguish False/0 and integer/float JSON values.
    require(canonical(value) == canonical(expected_plan()), "fixed plan changed")
    return value


def validate_launch(env: dict[str, str], event: Any, expected_sha: str) -> None:
    require(re.fullmatch(r"[0-9a-f]{40}", expected_sha) is not None,
            "full approved source commit required")
    fixed = {"GITHUB_REPOSITORY": REPOSITORY, "GITHUB_RUN_ATTEMPT": "1",
             "GITHUB_RUN_NUMBER": "1", "GITHUB_SHA": expected_sha}
    require(all(env.get(key) == value for key, value in fixed.items()),
            "first reviewed execution only; no rerun")
    require(re.fullmatch(r"[1-9][0-9]{0,19}", env.get("GITHUB_RUN_ID", "")) is not None,
            "run identity required")
    require(type(event) is dict and type(event.get("repository")) is dict,
            "repository event missing")
    repository = event["repository"]
    require(repository.get("full_name") == REPOSITORY and
            repository.get("private") is False and
            repository.get("default_branch") == "main", "public default-main only")
    if env.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        require(env.get("GITHUB_REF") == "refs/heads/main", "dispatch must use main")
        inputs = event.get("inputs")
        require(type(inputs) is dict and inputs.get("expected_source_sha") == expected_sha,
                "dispatch/source identity mismatch")
    else:
        require(env.get("GITHUB_EVENT_NAME") == "push" and
                env.get("GITHUB_REF") == "refs/heads/" + EXECUTION_BRANCH and
                event.get("ref") == env["GITHUB_REF"] and
                event.get("created") is True and event.get("deleted") is False and
                event.get("forced") is False and event.get("after") == expected_sha and
                event.get("before") == "0" * 40,
                "only first creation of the fixed execution branch is permitted")


def signal_metrics(values: Any, rate: int, *, eos: bool | None = None) -> dict[str, Any]:
    require(type(rate) is int and rate in (16000, 24000), "fixed audio rate")
    require(0 < len(values) <= 12 * rate, "bounded mono audio required")
    peak, energy, saturated = 0.0, 0.0, 0
    for sample in values:
        sample = float(sample)
        require(math.isfinite(sample), "nonfinite audio")
        peak = max(peak, abs(sample))
        energy += sample * sample
        saturated += abs(sample) >= 1.0
    rms = math.sqrt(energy / len(values))
    flags = []
    if peak >= 1.0:
        flags.append("peak_ge_one")
    if rms < 1e-4:
        flags.append("near_silent")
    if rate == 24000 and eos is not True:
        flags.append("native_termination_unverified")
    return {"rate_hz": rate, "frames": len(values), "peak": peak, "rms": rms,
            "duration_seconds": len(values) / rate,
            "clipped_fraction": saturated / len(values), "flags": flags,
            "technically_valid": not flags,
            "human_phonetic_review": "PENDING", "training_admitted": False}


def terminal_ledger(output: Path, receipt: Any) -> list[dict[str, Any]]:
    """A claim consumes a slot even if its outcome write was interrupted."""
    outcomes = receipt.get("outcomes", []) if type(receipt) is dict else []
    require(type(outcomes) is list and len(outcomes) <= 16, "outcome inventory")
    by_id: dict[str, dict[str, Any]] = {}
    allowed_ids = {row["cell_id"] for row in expected_plan()}
    for row in outcomes:
        require(type(row) is dict and row.get("cell_id") in allowed_ids and
                row["cell_id"] not in by_id, "invalid/duplicate outcome")
        by_id[row["cell_id"]] = row
    require({p.name for p in output.glob("*.claim")} <=
            {cell + ".claim" for cell in allowed_ids}, "unknown claim")
    ledger = []
    for cell in expected_plan():
        cell_id = cell["cell_id"]
        claim = output / (cell_id + ".claim")
        row = by_id.get(cell_id)
        if claim.exists() or claim.is_symlink():
            require(not claim.is_symlink() and claim.is_file() and
                    claim.stat().st_nlink == 1 and claim.stat().st_size <= 128,
                    "invalid claim file")
            completed = (claim.read_bytes() == (cell_id + "\n").encode() and
                         row is not None and row.get("status") == "generated_pending_review")
            state = "GENERATED_PENDING_REVIEW" if completed else "FAILED_NO_RETRY"
            attempts = 1
        else:
            require(row is None, "outcome without durable claim")
            state, attempts = "NOT_RUN", 0
        ledger.append({"cell_id": cell_id, "status": state, "attempts": attempts,
                       "ctc_training_allowed": False})
    consumed = [i for i, row in enumerate(ledger) if row["attempts"]]
    require(consumed == list(range(len(consumed))), "claims must form an attempted prefix")
    return ledger


def audio_payload(path: Path, *, rate: int, format_tag: int, bits: int) -> bytes:
    """Read a bounded complete RIFF file, including the original float payload."""
    require(not path.is_symlink() and path.is_file(), "regular original WAV required")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
            44 <= before.st_size <= 2 * 1024**2, "bounded unaliased WAV required")
    raw = path.read_bytes()
    require(len(raw) == before.st_size and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and
            struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw), "complete RIFF extent required")
    offset, fmt, data = 12, None, None
    while offset < len(raw):
        require(offset + 8 <= len(raw), "truncated chunk header")
        tag, count = struct.unpack_from("<4sI", raw, offset)
        start, end = offset + 8, offset + 8 + count
        require(end + (count & 1) <= len(raw), "truncated chunk/padding")
        if tag == b"fmt ":
            require(fmt is None and count in (16, 18), "duplicate/unsupported fmt")
            fmt = struct.unpack_from("<HHIIHH", raw, start)
            if count == 18:
                require(raw[start + 16:end] == b"\0\0", "unsupported format extension")
        elif tag == b"data":
            require(data is None, "duplicate data chunk")
            data = raw[start:end]
        offset = end + (count & 1)
    width = bits // 8
    require(fmt == (format_tag, 1, rate, rate * width, width, bits), "WAV format identity")
    require(data is not None and 0 < len(data) <= 12 * rate * width and
            len(data) % width == 0, "whole bounded WAV samples required")
    return data


def verify_completed_outputs(output: Path, receipt: Any) -> None:
    """Revalidate raw float/PCM identities and technical state before publication."""
    require(type(receipt) is dict and type(receipt.get("outcomes")) is list,
            "generation receipt shape")
    planned = {r["cell_id"]: r for r in expected_plan()}
    seen = set()
    for row in receipt["outcomes"]:
        require(type(row) is dict and row.get("cell_id") in planned and
                row["cell_id"] not in seen, "outcome identity")
        seen.add(row["cell_id"])
        if row.get("status") != "generated_pending_review":
            continue
        cell_id = row["cell_id"]
        require(canonical({k: row.get(k) for k in planned[cell_id]}) ==
                canonical(planned[cell_id]), "completed-row plan/label drift")
        require(type(row.get("attempts")) is int and row["attempts"] == 1,
                "one actual cell attempt required")
        for suffix, key in ((".raw-float.wav", "raw_wav_sha256"), (".wav", "wav_sha256")):
            path = output / (cell_id + suffix)
            require(not path.is_symlink() and path.is_file() and
                    path.stat().st_nlink == 1 and path.stat().st_size <= 2 * 1024**2,
                    "bounded original audio required")
            require(file_sha256(path) == row.get(key), "audio drift before publication")
        native = audio_payload(output / (cell_id + ".raw-float.wav"), rate=24000, format_tag=3, bits=32)
        require(hashlib.sha256(native).hexdigest() == row.get("native_float_values_sha256"),
                "original float values identity")
        termination = row.get("native_termination")
        require(type(termination) is dict and termination.get("ended_with_eos") is True,
                "observed native termination required")
        eos, last = termination.get("eos_token_id"), termination.get("last_token_ids")
        require(type(eos) is int and type(last) is list and len(last) == 1 and
                type(last[0]) is int and last[0] == eos, "native EOS observation identity")
        metrics = signal_metrics([v[0] for v in struct.iter_unpack("<f", native)], 24000, eos=True)
        require(metrics["technically_valid"] and canonical(metrics) == canonical(row.get("native_metrics")),
                "native signal metrics drift")
        pcm = audio_payload(output / (cell_id + ".wav"), rate=16000, format_tag=1, bits=16)
        frames = len(pcm) // 2
        require(type(row.get("frames")) is int and frames == row["frames"] and
                frames == (len(native) // 4 * 2 + 2) // 3,
                "complete derived frame geometry")
        require(hashlib.sha256(pcm).hexdigest() == row.get("pcm_sha256"), "completed PCM16 identity")
        require(row.get("resample_policy") == "scipy-resample-poly-f64-2-over-3-no-gain-no-trim-v1",
                "fixed derivative recipe")
        derived = row.get("derived_metrics")
        require(type(derived) is dict and derived.get("technically_valid") is True and
                derived.get("flags") == [] and type(derived.get("frames")) is int and
                derived["frames"] == frames and derived.get("rate_hz") == 16000 and
                derived.get("training_admitted") is False, "derived technical evidence")
