#!/usr/bin/env python3
"""Private research views of saved active ASR outputs; no model execution.

Keeps full transcripts and original receipts. A separate, explicit machine-weak
decision never changes historical admission fields or creates human gold.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kws_data_contract.identity import SourceRoles  # noqa: E402
from kws_data_contract.reference_text import normalize  # noqa: E402
from kws_data_contract.source_registry import canonical, read_bytes, require, _pairs  # noqa: E402

SCHEMA = "kws-active-asr-research-weak-view-v1"
SCHEMA_V2 = "kws-active-asr-research-weak-view-v2"
POLICY = "full-transcript-two-of-three-no-intended-positive-to-negative-v1"
POLICY_V2 = "full-transcript-agreement-with-explicit-not-requested-v2"
SELECTIVE_ADAPTER = "kws-active-asr-selective-consumer-v2"
SELECTIVE_POLICY_SHA = "5cba6043b6c7e39aaa39482b31e0e14ae478cd5485b386e7d87f5bec78a3c501"
MODELS = ("qwen06", "sensevoice", "whisper")
KEYWORDS = ("你好小窝", "小窝小窝")
FOUR_CHAR_IDS = {"你": 1, "好": 2, "小": 3, "窝": 4}
FOUR_TOKEN_NAMES = {1: "ni3", 2: "hao3", 3: "xiao3", 4: "wo1"}
FSMN_TOKENS_SHA = "5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa"
SHA = re.compile(r"[0-9a-f]{64}\Z")
D1_ADAPTER = "kws-active-d1-frozen-plan-source-v1"
D1_ADAPTER_V2 = "kws-active-d1-frozen-plan-source-v2"
D1_ADAPTER_V3 = "kws-active-d1-partial-observation-source-v3"
BULK_POOL_PATH = "research/auto-tts-voice-clone-20261102/bulk_pool.py"
ACCOUNTED_SOURCE_PATHS = (
    "automation/active_asr_weak_view.py", "automation/research_weak_sampling.py",
    "kws_data_contract/__init__.py", "kws_data_contract/identity.py",
    "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py", BULK_POOL_PATH,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def loads(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (
        _ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def relative(name):
    require(type(name) is str and 0 < len(name) <= 512 and "\\" not in name,
            "PRIVATE_RELATIVE_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and
            all(part not in ("", ".", "..") for part in path.parts),
            "PRIVATE_RELATIVE_PATH")
    return name


def save(root, name, raw):
    path = root / relative(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def save_json(root, name, value):
    save(root, name, canonical(value) + b"\n")


def save_rows(root, name, rows):
    save(root, name, b"".join(canonical(row) + b"\n" for row in rows))


def empty_evidence(path):
    """An inventory can legitimately include a quiet, empty process log."""
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "EMPTY_EVIDENCE_ALIAS")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == 0,
                "EMPTY_EVIDENCE_CHANGED")
        raw = stream.read(1)
        after = os.fstat(stream.fileno())
        require(not raw and all(getattr(before, key) == getattr(after, key) for key in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")), "EMPTY_EVIDENCE_CHANGED")
        return raw


def float_wav_pcm(raw):
    """Read original IEEE FLOAT RIFF without silently quantizing it."""
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and
            len(raw) == struct.unpack_from("<I", raw, 4)[0] + 8, "ORIGINAL_RIFF")
    chunks, offset = {}, 12
    while offset + 8 <= len(raw):
        kind, size = struct.unpack_from("<4sI", raw, offset)
        require(kind not in chunks and offset + 8 + size <= len(raw), "ORIGINAL_RIFF_CHUNK")
        chunks[kind] = raw[offset + 8:offset + 8 + size]
        offset += 8 + size + size % 2
    require(offset == len(raw) and b"fmt " in chunks and b"data" in chunks,
            "ORIGINAL_RIFF_CHUNK")
    fmt = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    require(fmt == (3, 1, 24000, 96000, 4, 32), "ORIGINAL_FLOAT24K_REQUIRED")
    pcm = chunks[b"data"]
    require(0 < len(pcm) <= 24000 * 30 * 4 and len(pcm) % 4 == 0,
            "ORIGINAL_AUDIO_EXTENT")
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", pcm)),
            "NONFINITE_ORIGINAL_AUDIO")
    return pcm


def quantize_pcm16(raw, frames, *, minimum_frames=400):
    """Explicit nearest-even 32768-scale saturation; no gain or time edits."""
    require(type(frames) is int and minimum_frames <= frames <= 16000 * 30 and
            len(raw) == frames * 4, "DERIVED_PCM_EXTENT")
    values = [x[0] for x in struct.iter_unpack("<f", raw)]
    require(all(math.isfinite(x) for x in values), "NONFINITE_DERIVED_AUDIO")
    unbounded = [round(x * 32768) for x in values]
    clipped = sum(x < -32768 or x > 32767 for x in unbounded)
    samples = [max(-32768, min(32767, x)) for x in unbounded]
    pcm = struct.pack("<" + "h" * frames, *samples)
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm)
    wav = target.getvalue()
    flags = ([] if any(samples) else ["no_nonzero_pcm16_samples"])
    if clipped:
        flags.append("quantization_saturation")
    return wav, {
        "recipe": "float32-to-pcm16-round-nearest-even-scale32768-saturate-v1",
        "source_pcm_sha256": sha(raw), "file_sha256": sha(wav),
        "pcm_sha256": sha(pcm), "sample_rate_hz": 16000, "channels": 1,
        "frames": frames, "clipped_samples": clipped,
        "peak_float": max(abs(x) for x in values),
        "rms_float": math.sqrt(sum(x * x for x in values) / frames),
        "gain_normalization": False, "trimming": False, "padding": False,
        "quality_flags": flags,
    }


def source_rows(archive_raw, inputs):
    """Join private original generation receipts, never infer a voice from a lane."""
    require(sha(archive_raw) == inputs["source_archive_sha256"], "SOURCE_ARCHIVE_CHANGED")
    sources, evidence = {}, {}
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        roster = archive.infolist()
        names = [row.filename for row in roster]
        require(len(names) == len(set(names)) and len(names) <= 2048 and
                sum(row.file_size for row in roster) <= 256 * 1024 * 1024,
                "SOURCE_ARCHIVE_ROSTER")
        for row in roster:
            relative(row.filename)
            require(not row.is_dir() and row.file_size <= 16 * 1024 * 1024,
                    "SOURCE_ARCHIVE_MEMBER")
        wav_names = {name for name in names if name.endswith(".wav")}
        require(wav_names == {clip["original_member"] for clip in inputs["clips"]},
                "SOURCE_AUDIO_ROSTER_CHANGED")
        for clip in inputs["clips"]:
            member = relative(clip["original_member"])
            parent = PurePosixPath(member).parent
            generation_name = str(parent / "generation.json")
            runtime_name = str(parent / "runtime-verification.json")
            if generation_name not in evidence:
                evidence[generation_name] = archive.read(generation_name)
                evidence[runtime_name] = archive.read(runtime_name)
            generation = loads(evidence[generation_name])
            runtime = loads(evidence[runtime_name])
            require(generation["source_kind"] == "synthetic-public-preset" and
                    generation["human_gold"] is False and generation["training_admitted"] is False and
                    runtime["campaign"] == generation["campaign"] and
                    runtime["training_admitted"] is False and
                    re.fullmatch(r"[0-9a-f]{40}", runtime["model_revision"]) and
                    runtime["model_id"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "SOURCE_PRESET_IDENTITY")
            outcomes = {row["cell_id"]: row for row in generation["outcomes"]}
            planned = {row["cell_id"]: row for row in generation["planned"]}
            require(len(outcomes) == len(generation["outcomes"]) == len(planned) and
                    set(outcomes) == set(planned), "SOURCE_CELL_ROSTER")
            matching = [row for row in outcomes.values()
                        if str(parent / (row["cell_id"] + ".raw-float.wav")) == member]
            require(len(matching) == 1, "SOURCE_MEMBER_JOIN")
            row = matching[0]
            require(all(row.get(key) == value for key, value in planned[row["cell_id"]].items()) and
                    row["status"] == "generated" and row["training_admitted"] is False and
                    row["intended_split"] == "train_only" and
                    row["label_authority"] == "requested_text_only" and
                    row["intended_label_tier"] == "machine_weak_label",
                    "SOURCE_GENERATION_RECORD")
            raw = archive.read(member)
            pcm = float_wav_pcm(raw)
            require(row["raw_wav_sha256"] == clip["original_wav"]["sha256"] == sha(raw) and
                    row["native"]["source_values_sha256"] == clip["original_pcm"]["sha256"] == sha(pcm) and
                    row["native"]["frames"] == clip["original_pcm"]["frames"] == len(pcm) // 4,
                    "SOURCE_WAV_PCM_JOIN")
            speaker = row["speaker"]
            require(type(speaker) is str and re.fullmatch(r"[A-Za-z][A-Za-z_]{0,63}", speaker),
                    "SOURCE_PRESET_SPEAKER")
            # Same public preset stays one lineage across releases and campaigns.
            family = "qwen3-preset-" + speaker.lower()
            sources[clip["id"]] = {
                "source_id": row["cell_id"], "source_family_id": family,
                "speaker_id": family, "role": "train-candidate", "split": "train",
                "original_intended_split": row["intended_split"],
                "source_kind": generation["source_kind"], "preset": speaker,
                "tts_model_id": runtime["model_id"], "tts_model_revision": runtime["model_revision"],
                "intended_text": row["text"], "intended_kind": row["intended_kind"],
                "intended_keyword_id": row["intended_keyword_id"],
                "generation_receipt_path": "evidence/source/" + generation_name,
                "generation_receipt_sha256": sha(evidence[generation_name]),
                "runtime_receipt_path": "evidence/source/" + runtime_name,
                "runtime_receipt_sha256": sha(evidence[runtime_name]),
                "original_generation_row": row,
            }
    return sources, evidence


def _d1_plan(plan_raw, expected_plan_sha):
    spec = importlib.util.spec_from_file_location("_active_d1_pool", ROOT / BULK_POOL_PATH)
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    plan = loads(plan_raw)
    require(type(expected_plan_sha) is str and SHA.fullmatch(expected_plan_sha) and
            sha(canonical(plan)) == expected_plan_sha, "D1_FROZEN_PLAN_CHANGED")
    pool.verify_plan(plan)
    require(plan.get("cell_layout") == pool.D1_LAYOUT, "D1_LAYOUT_REQUIRED")
    source_manifest = {"schema": pool.SCHEMA, "references": plan["references"],
                       "training_admitted": False, "qualification_allowed": False}
    require(sha(canonical(source_manifest)) == plan["source_manifest_sha256"],
            "D1_REFERENCE_MANIFEST_CHANGED")
    refs = {row["reference_id"]: row for row in plan["references"]}
    for ref in refs.values():
        for key in ("reference_id", "source_family_id", "speaker_group_id"):
            pool._id(ref[key])
        for key in ("wav_sha256", "pcm_sha256", "source_terms_sha256", "rights_record_sha256"):
            pool._sha(ref[key])
        if ref["parent_source_sha256"] is not None:
            pool._sha(ref["parent_source_sha256"])
    return plan, refs, {row["cell_id"]: row for row in plan["cells"]}


def _d1_lock(raw, plan, plan_sha):
    require(loads(raw) == {"schema": "kws-bulk-clone-execution-lock-v1",
                          "plan_sha256": plan_sha, "training_admitted": False,
                          "model_id": plan["model_id"]}, "D1_EXECUTION_LOCK_CHANGED")


def _d1_audio_rows(archive_raw, inputs, cells):
    require(sha(archive_raw) == inputs["source_archive_sha256"], "SOURCE_ARCHIVE_CHANGED")
    seen_cells, seen_pcm = set(), set()
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        roster = archive.infolist()
        names = [row.filename for row in roster]
        require(len(names) == len(set(names)) == len(inputs["clips"]) and
                len({clip["id"] for clip in inputs["clips"]}) == len(inputs["clips"]) and
                sum(row.file_size for row in roster) <= 256 * 1024 * 1024 and
                set(names) == {clip["original_member"] for clip in inputs["clips"]},
                "D1_SOURCE_AUDIO_ROSTER")
        for member in roster:
            relative(member.filename)
            require(not member.is_dir() and 0 < member.file_size <= 16 * 1024 * 1024,
                    "D1_SOURCE_AUDIO_MEMBER")
        for clip in inputs["clips"]:
            match = re.fullmatch(r"cells/(bulk-[0-9a-f]{24})/native-f32\.wav", clip["original_member"])
            require(match is not None and match[1] in cells and match[1] not in seen_cells,
                    "D1_SOURCE_CELL_JOIN")
            cell = cells[match[1]]
            raw = archive.read(clip["original_member"])
            pcm = float_wav_pcm(raw)
            require(clip["original_wav"]["sha256"] == sha(raw) and
                    clip["original_wav"]["bytes"] == len(raw) and
                    clip["original_pcm"]["sha256"] == sha(pcm) and sha(pcm) not in seen_pcm and
                    clip["original_pcm"]["frames"] == len(pcm) // 4, "D1_NATIVE_WAV_PCM_JOIN")
            seen_cells.add(cell["cell_id"])
            seen_pcm.add(sha(pcm))
            yield clip, cell, raw, pcm


def _d1_started(raw, cell, ref, plan_sha):
    require(loads(raw) == {"schema": "kws-bulk-clone-start-v1", "cell_id": cell["cell_id"],
                          "reference_id": ref["reference_id"], "reference_wav_sha256": ref["wav_sha256"],
                          "plan_sha256": plan_sha, "source_family_id": cell["source_family_id"],
                          "seed": cell["seed"], "attempts": 1, "training_admitted": False},
            "D1_STARTED_PLAN_JOIN")


def _d1_completed(completed_raw, cell, clip, raw, pcm, old_wav):
    completed = loads(completed_raw)
    require(set(completed) == {"schema", "cell_id", "status", "technical", "audio", "source_family_id",
                               "intended_kind", "actual_text", "training_admitted", "qualification_allowed"} and
            completed["schema"] == "kws-bulk-clone-complete-v1" and
            completed["cell_id"] == cell["cell_id"] and completed["status"] == "GENERATED_PENDING_ASR" and
            completed["source_family_id"] == cell["source_family_id"] and
            completed["intended_kind"] == cell["intended_kind"] and completed["actual_text"] is None and
            completed["training_admitted"] is False and completed["qualification_allowed"] is False,
            "D1_COMPLETED_PLAN_JOIN")
    audio, technical = completed["audio"], completed["technical"]
    require(audio["native_sha256"] == sha(raw) and audio["native_bytes"] == len(raw) and
            technical["native_frames"] == len(pcm) // 4 and technical["native_rate"] == 24000 and
            technical["duration_s"] == len(pcm) / (4 * 24000) and
            type(technical["technical_audio_pass"]) is bool, "D1_NATIVE_WAV_PCM_JOIN")
    with wave.open(io.BytesIO(old_wav), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                 reader.getnframes(), reader.getcomptype()) ==
                (1, 2, 16000, audio["audio_frames"], "NONE"), "D1_GENERATED_PCM_FORMAT")
        old_pcm = reader.readframes(reader.getnframes())
    require(audio["sample_rate_hz"] == 16000 and sha(old_wav) == audio["wav_sha256"] and
            len(old_wav) == audio["wav_bytes"] == len(old_pcm) + 44 and
            sha(old_pcm) == audio["pcm_sha256"] and len(old_pcm) == audio["audio_frames"] * 2 and
            audio["audio_frames"] == clip["derived_pcm"]["frames"], "D1_GENERATED_WAV_PCM_JOIN")
    return completed


def _d1_source(cell, ref, plan, plan_sha, started_raw, outcome_raw, prefix, outcome_name):
    return {"source_id": cell["cell_id"], "source_family_id": ref["source_family_id"],
            "speaker_id": ref["speaker_group_id"], "role": ref["role"],
            "split": "train" if ref["role"] == "train-candidate" else "development",
            "source_kind": "synthetic-reference-clone", "reference_id": ref["reference_id"],
            "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
            "parent_source_sha256": ref["parent_source_sha256"],
            "tts_model_id": plan["model_id"], "tts_model_revision": plan["model_revision"],
            "intended_text": cell["prompt"], "intended_kind": cell["intended_kind"],
            "intended_keyword_id": cell["intended_keyword_id"],
            "generation_category": cell["generation_category"], "source_plan_sha256": plan_sha,
            "generation_receipt_path": "evidence/source/" + prefix + outcome_name,
            "generation_receipt_sha256": sha(outcome_raw),
            "generation_start_receipt_path": "evidence/source/" + prefix + "started.json",
            "generation_start_receipt_sha256": sha(started_raw), "original_generation_row": loads(outcome_raw)}


def d1_source_rows(archive_raw, inputs, plan_raw, expected_plan_sha, generation_root):
    """Replay v1 successful bulk receipts against the externally pinned full plan."""
    plan, refs, cells = _d1_plan(plan_raw, expected_plan_sha)
    root = Path(generation_root)
    evidence = {"d1/plan.json": plan_raw}

    def retained(name, limit=1024 * 1024):
        raw = read_bytes(root / relative(name), limit)
        evidence["d1/" + name] = raw
        return raw

    _d1_lock(retained("plan-lock.json"), plan, expected_plan_sha)
    sources = {}
    for clip, cell, raw, pcm in _d1_audio_rows(archive_raw, inputs, cells):
        ref = refs[cell["reference_id"]]
        folder = "cells/" + cell["cell_id"] + "/"
        require(not (root / folder / "failed.json").exists(), "D1_SOURCE_FAILED_CELL")
        started_raw = retained(folder + "started.json")
        _d1_started(started_raw, cell, ref, expected_plan_sha)
        completed_raw = retained(folder + "completed.json")
        _d1_completed(completed_raw, cell, clip, raw, pcm,
                      retained(folder + "audio-16k.wav", 2 * 1024 * 1024))
        sources[clip["id"]] = _d1_source(cell, ref, plan, expected_plan_sha, started_raw,
                                         completed_raw, "d1/" + folder, "completed.json")
    return sources, evidence, {"schema": D1_ADAPTER, "plan_sha256": expected_plan_sha,
                                "plan_file_sha256": sha(plan_raw)}


def _d1_native_array(raw, native, pcm):
    """Read bounded numeric NPY data without NumPy, pickle, or model imports."""
    require(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00", b"\x03\x00"),
            "D1_NATIVE_ARRAY_FORMAT")
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    start = 8 + width + size
    require(1 <= size <= 4096 and start <= len(raw), "D1_NATIVE_ARRAY_HEADER")
    header = ast.literal_eval(raw[8 + width:start].decode("utf-8" if raw[6] == 3 else "latin1"))
    require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"} and
            header["fortran_order"] is False and header["descr"] in ("<f4", "<f8", ">f4", ">f8") and
            type(header["shape"]) is tuple and len(header["shape"]) == 1 and
            type(header["shape"][0]) is int and 1 <= header["shape"][0] <= 720000 and
            native["dtype"] == header["descr"] and native["shape"] == list(header["shape"]),
            "D1_NATIVE_ARRAY_GEOMETRY")
    body = raw[start:]
    require(len(body) == header["shape"][0] * int(header["descr"][-1]) and
            sha(body) == native["samples_sha256"], "D1_NATIVE_ARRAY_VALUES")
    fmt = header["descr"][0] + ("f" if header["descr"][-1] == "4" else "d")
    values = [item[0] for item in struct.iter_unpack(fmt, body)]
    require(all(math.isfinite(value) for value in values) and
            struct.pack("<" + "f" * len(values), *values) == pcm, "D1_NATIVE_ARRAY_PCM_JOIN")


def _d1_observed_source(clip, cell, raw, pcm, ref, plan, expected_plan_sha, row, binding, evidence,
                        *, artifact_roots=None, retained_root=None, prefix=None,
                        checkpoint_reader=None, owner_files=None):
    """Replay one actual native return with unchanged audio, text and mode checks."""
    name = cell["cell_id"]
    record = row["record"]
    prefix = prefix if prefix is not None else "d1/artifacts/" + name + "/"
    if checkpoint_reader is not None:
        require(artifact_roots is None and retained_root is None and type(owner_files) is dict,
                "D1_SINGLE_ARTIFACT_LOCATION")
        root = PurePosixPath(record["output_root"])
    elif retained_root is not None:
        require(artifact_roots is None, "D1_SINGLE_ARTIFACT_LOCATION")
        root = Path(retained_root) / prefix
    else:
        roots = artifact_roots if artifact_roots is not None else {record["output_root"]: record["output_root"]}
        require(type(roots) is dict and record["output_root"] in roots, "D1_ARTIFACT_ROOT_MAPPING")
        root = Path(roots[record["output_root"]])
    files = record["files"]
    generated, native_folder = "generation/cells/" + name + "/", "native/" + name + "/"
    success = record["status"] == "GENERATED_PENDING_ASR"
    outcome_name = "completed.json" if success else "failed.json"
    required = {generated + "started.json", generated + outcome_name,
                *(native_folder + suffix for suffix in
                  ("call.json", "native.json", "returned.json", "returned-array.npy", "returned-native.wav"))}
    if success:
        required |= {generated + "native-f32.wav", generated + "audio-16k.wav"}
    require(type(files) is dict and set(files) == required and
            record["native_wav"] == (generated + "native-f32.wav" if success else native_folder + "returned-native.wav"),
            "D1_NATIVE_FILE_ROSTER")
    if checkpoint_reader is not None:
        actual = {path.removeprefix("outputs/") for path in owner_files
                  if path.startswith(("outputs/" + generated, "outputs/" + native_folder))}
    else:
        actual = {path.relative_to(root).as_posix() for location in (root / generated, root / native_folder)
                  for path in location.rglob("*") if path.is_file()}
    require(actual == required, "D1_NATIVE_FILE_ROSTER")
    bodies = {}
    for path, expected in files.items():
        body = (checkpoint_reader.file(str(root), path, expected, 16 * 1024 * 1024)
                if checkpoint_reader is not None else read_bytes(root / relative(path), 16 * 1024 * 1024))
        require(expected == {"bytes": len(body), "sha256": sha(body)}, "D1_NATIVE_FILE_CHANGED")
        bodies[path] = body
        if checkpoint_reader is None:
            evidence[prefix + path] = body
    lock = (checkpoint_reader.file(str(root), "generation/plan-lock.json",
            owner_files["outputs/generation/plan-lock.json"]) if checkpoint_reader is not None
            else read_bytes(root / "generation/plan-lock.json"))
    _d1_lock(lock, plan, expected_plan_sha)
    if checkpoint_reader is None:
        evidence[prefix + "generation/plan-lock.json"] = lock
    require(bodies[record["native_wav"]] == raw, "D1_ARCHIVE_NATIVE_CHANGED")
    started_raw = bodies[generated + "started.json"]
    _d1_started(started_raw, cell, ref, expected_plan_sha)
    call = loads(bodies[native_folder + "call.json"])
    checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
    expected_call = {"schema": "active-tts-generation-call-v1", "cell_id": name,
                     "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
                     "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                     "reference_text_status": ref["reference_text_status"],
                     "reference_mode": "ICL" if checked else "x-vector-only", "reference_text_passed": checked,
                     "seed": cell["seed"], "intended_text": cell["prompt"], "generation": binding["generation"], "attempt": 1}
    require(set(call) == set(expected_call) | {"reference_prompt_seconds"} and
            all(call[key] == value for key, value in expected_call.items()) and
            type(call["reference_prompt_seconds"]) in (int, float) and call["reference_prompt_seconds"] >= 0,
            "D1_NATIVE_CALL_REFERENCE_MODE_JOIN")
    native = loads(bodies[native_folder + "native.json"])
    require(native["schema"] == "active-tts-native-return-v1" and
            native["array"] == files[native_folder + "returned-array.npy"] and
            native["wav"] == files[native_folder + "returned-native.wav"] and
            native["technical_gate_applied"] is False and native["all_finite"] is True and
            native["sample_rate_hz"] == 24000 and native.get("wav_unavailable_reason") is None and
            float_wav_pcm(bodies[native_folder + "returned-native.wav"]) == pcm,
            "D1_NATIVE_RETURN_JOIN")
    _d1_native_array(bodies[native_folder + "returned-array.npy"], native, pcm)
    returned = loads(bodies[native_folder + "returned.json"])
    require(set(returned) == {"generation_seconds", "total_seconds", "native_stop_reason"} and
            returned["native_stop_reason"] == "not_exposed_by_voice_clone_API" and
            all(type(returned[key]) in (int, float) and returned[key] >= 0
                for key in ("generation_seconds", "total_seconds")), "D1_NATIVE_RETURN_RECEIPT")
    terminal_raw = bodies[generated + outcome_name]
    if success:
        require(record["technical_failure_reason"] is None, "D1_COMPLETION_FAILURE_CONFLICT")
        _d1_completed(terminal_raw, cell, clip, raw, pcm, bodies[generated + "audio-16k.wav"])
    else:
        failure = loads(terminal_raw)
        require(set(failure) == {"schema", "cell_id", "status", "error_type", "training_admitted"} and
                failure["schema"] == "kws-bulk-clone-failure-v1" and failure["cell_id"] == name and
                failure["status"] == "FAILED_NO_RETRY" and failure["training_admitted"] is False and
                type(failure["error_type"]) is str and 1 <= len(failure["error_type"]) <= 128 and
                type(record["technical_failure_reason"]) is str and 1 <= len(record["technical_failure_reason"]) <= 4096,
                "D1_FAILED_NATIVE_RECEIPT")
    source = _d1_source(cell, ref, plan, expected_plan_sha, started_raw, terminal_raw,
                        prefix + generated, outcome_name)
    source.update(generation_outcome="completed" if success else "failed-native",
                  technical_failure_reason=record["technical_failure_reason"],
                  outcome_record_sha256=row["ledger_record_sha256"],
                  generation_call_receipt_sha256=sha(bodies[native_folder + "call.json"]))
    if checkpoint_reader is not None:
        source["generation_receipt_path"] = "evidence/source/" + checkpoint_reader.location(
            str(root / generated / outcome_name))
        source["generation_start_receipt_path"] = "evidence/source/" + checkpoint_reader.location(
            str(root / generated / "started.json"))
    return source


def d1_source_rows_v2(archive_raw, inputs, plan_raw, expected_plan_sha, outcomes_raw,
                      expected_outcomes_sha, stage_raw, job_raw, *, artifact_roots=None,
                      retained_root=None):
    """Replay SHA-pinned immutable outcomes, including actual failed native returns.

    Only the archive roster is consumed. Reused and non-decodable outcomes stay
    in the retained outcomes document, never become a new recognition request.
    At intake, artifact_roots may relocate historical output roots. Package
    replay exclusively uses retained_root and never consults the live ledger.
    """
    plan, refs, cells = _d1_plan(plan_raw, expected_plan_sha)
    require(type(expected_outcomes_sha) is str and SHA.fullmatch(expected_outcomes_sha) and
            sha(outcomes_raw) == expected_outcomes_sha, "D1_OUTCOMES_PIN_CHANGED")
    outcome, stage, job = map(loads, (outcomes_raw, stage_raw, job_raw))
    require(all(raw == canonical(value) + b"\n" for raw, value in
                ((outcomes_raw, outcome), (stage_raw, stage), (job_raw, job))), "D1_CANONICAL_RECEIPTS_REQUIRED")
    require(set(outcome) == {"schema", "stage", "stage_sha256", "job_id", "job_recipe_sha256",
                            "plan_canonical_sha256", "binding", "selected_cells", "new_call_ids",
                            "reused_cell_ids", "rows"} and
            outcome["schema"] == "active-tts-stage-outcomes-v1" and
            outcome["stage_sha256"] == sha(stage_raw) and outcome["job_recipe_sha256"] == sha(job_raw) and
            outcome["plan_canonical_sha256"] == expected_plan_sha, "D1_OUTCOMES_STAGE_JOB_JOIN")
    binding = stage["binding"]
    require(type(binding) is dict and set(binding) == {
        "plan_canonical_sha256", "plan_file_sha256", "model_assets_sha256", "source_manifest_sha256",
        "producer_source_sha256", "runtime_setup_sha256", "generation"} and
        outcome["binding"] == binding and binding["plan_canonical_sha256"] == expected_plan_sha and
        binding["plan_file_sha256"] == sha(plan_raw) and type(binding["generation"]) is dict and
        all(type(value) is str and SHA.fullmatch(value) for key, value in binding.items() if key != "generation"),
        "D1_STAGE_PLAN_BINDING")
    require(stage["schema"] == "active-tts-extension-stage-v1" and
            stage["training_admitted"] is False and stage["paired_cases_included"] is False and
            stage["max_retries"] == 0 and stage["maximum_job_seconds"] == 600 and
            stage["stage_budget_minutes"] == 20 and len(plan["cells"]) == 256 and
            type(stage["target_count"]) is int and 8 <= stage["target_count"] <= 256 and
            stage["target_cells"] == [cell["cell_id"] for cell in plan["cells"][:stage["target_count"]]] and
            type(stage["chunks"]) is list and 1 <= len(stage["chunks"]) <= 2 and
            all(type(chunk) is list and 1 <= len(chunk) <= 28 for chunk in stage["chunks"]),
            "D1_STAGE_SCOPE")
    selected = [name for chunk in stage["chunks"] for name in chunk]
    reused = stage["initial_reused_cells"]
    require(len(selected) == len(set(selected)) == stage["maximum_new_calls"] <= 56 and
            type(reused) is list and len(reused) == len(set(reused)) and
            not set(selected) & set(reused) and set(selected) | set(reused) == set(stage["target_cells"]) and
            {cell["cell_id"] for cell in plan["cells"][:8]} <= set(reused),
            "D1_STAGE_PARTITION")
    index = job["chunk_index"]
    require(type(index) is int and 0 <= index < len(stage["chunks"]) and
            job == {"schema": "active-tts-extension-job-v1", "stage": outcome["stage"],
                    "stage_sha256": sha(stage_raw), "job_id": "chunk-" + str(index + 1).zfill(3),
                    "chunk_index": index, "cells": stage["chunks"][index],
                    "maximum_new_calls": len(stage["chunks"][index]),
                    "timeout_seconds": stage["maximum_job_seconds"]} and
            outcome["job_id"] == job["job_id"] and outcome["selected_cells"] == job["cells"] and
            outcome["stage"] == str(PurePosixPath(stage["prepared"]) / "stage.json"),
            "D1_JOB_CHUNK_JOIN")
    new, reused = outcome["new_call_ids"], outcome["reused_cell_ids"]
    require(type(new) is list and type(reused) is list and len(new) == len(set(new)) and
            len(reused) == len(set(reused)) and not set(new) & set(reused) and
            set(new) | set(reused) == set(job["cells"]) and
            [row["cell_id"] for row in outcome["rows"]] == job["cells"], "D1_OUTCOME_ROSTER")
    entries = {}
    for row in outcome["rows"]:
        require(set(row) == {"cell_id", "ledger_record_sha256", "record", "claim"}, "D1_OUTCOME_ROW")
        name, record, claim = row["cell_id"], row["record"], row["claim"]
        cell = cells[name]
        identity = sha(canonical({"binding": binding, "cell": cell, "reference": refs[cell["reference_id"]]}))
        require(record["schema"] == "active-tts-plan-cell-v1" and record["cell_id"] == name and
                record["identity_sha256"] == identity and sha(canonical(record) + b"\n") == row["ledger_record_sha256"] and
                record["training_admitted"] is False and record["status"] in ("GENERATED_PENDING_ASR", "FAILED_NO_RETRY") and
                type(record["output_root"]) is str and PurePosixPath(record["output_root"]).is_absolute() and
                record["generation_root"] == str(PurePosixPath(record["output_root"]) / "generation"),
                "D1_TERMINAL_RECORD_JOIN")
        require(record["origin"] == "extension-call" and record["new_generation_calls"] == 1 and
                type(claim) is dict and set(claim) == {"schema", "cell_id", "identity_sha256", "attempt", "owner", "started_at"} and
                claim["schema"] == "active-tts-plan-call-claim-v1" and claim["cell_id"] == name and
                claim["identity_sha256"] == identity and claim["attempt"] == 1 and
                sha(canonical(claim) + b"\n") == record["claim_sha256"], "D1_CALL_CLAIM_JOIN")
        if name in new:
            require(claim["owner"] == {"stage_sha256": sha(stage_raw), "job_id": job["job_id"],
                                       "job_recipe_sha256": sha(job_raw)}, "D1_NEW_CALL_OWNER")
        entries[name] = row
    evidence = {"d1/plan.json": plan_raw, "d1/stage.json": stage_raw,
                "d1/job-recipe.json": job_raw, "d1/outcomes.json": outcomes_raw}
    sources = {}
    for clip, cell, raw, pcm in _d1_audio_rows(archive_raw, inputs, cells):
        name = cell["cell_id"]
        require(name in new, "D1_ARCHIVE_MUST_BE_NEW_CALL_SUBSET")
        sources[clip["id"]] = _d1_observed_source(
            clip, cell, raw, pcm, refs[cell["reference_id"]], plan, expected_plan_sha, entries[name],
            binding, evidence, artifact_roots=artifact_roots, retained_root=retained_root)
    return sources, evidence, {"schema": D1_ADAPTER_V2, "plan_sha256": expected_plan_sha,
        "plan_file_sha256": sha(plan_raw), "outcomes_sha256": expected_outcomes_sha,
        "stage_sha256": sha(stage_raw), "job_recipe_sha256": sha(job_raw)}


class _D1CheckpointEvidence:
    """Pin every read; retained replay never consults the original ledger."""

    def __init__(self, *, roots=None, retained_root=None):
        require(roots is None or retained_root is None, "D1_SINGLE_ARTIFACT_LOCATION")
        self.roots, self.retained_root = roots or {}, retained_root
        self.evidence, self.facts = {}, {}

    @staticmethod
    def absolute(value):
        require(type(value) is str and PurePosixPath(value).is_absolute() and
                PurePosixPath(value).as_posix() == value and "\\" not in value and
                ".." not in PurePosixPath(value).parts, "D1_ABSOLUTE_EVIDENCE_PATH")
        return value

    @staticmethod
    def location(path):
        return "d1/accounted/pinned/" + sha(path.encode()) + ".bin"

    def effective(self, path):
        self.absolute(path)
        if self.retained_root is not None:
            return Path(self.retained_root) / self.location(path)
        matches = [root for root in self.roots if Path(path).is_relative_to(root)]
        if matches:
            root = max(matches, key=len)
            return Path(self.roots[root]) / Path(path).relative_to(root)
        return Path(path)

    def pin(self, value, limit=32 * 1024 * 1024):
        require(type(value) is dict and set(value) == {"path", "bytes", "sha256"} and
                type(value["bytes"]) is int and 0 <= value["bytes"] <= limit and
                type(value["sha256"]) is str and SHA.fullmatch(value["sha256"]), "D1_EVIDENCE_PIN")
        path = self.effective(value["path"])
        body = empty_evidence(path) if value["bytes"] == 0 else read_bytes(path, limit)
        require(len(body) == value["bytes"] and sha(body) == value["sha256"], "D1_PINNED_EVIDENCE_CHANGED")
        name = self.location(value["path"])
        require(name not in self.evidence or self.evidence[name] == body, "D1_EVIDENCE_CHANGED_BETWEEN_READS")
        self.evidence[name] = body
        self.facts[value["path"]] = {"path": str(path.absolute()), "bytes": len(body), "sha256": sha(body)}
        return body

    def file(self, root, name, record, limit=32 * 1024 * 1024):
        relative(name)
        require(type(record) is dict and set(record) == {"bytes", "sha256"}, "D1_FILE_RECORD")
        return self.pin({"path": str(PurePosixPath(self.absolute(root)) / name), **record}, limit)

    def json(self, value):
        return loads(self.pin(value))

    def roster(self, root, records):
        require(type(records) is dict and len(records) <= 10000, "D1_EVIDENCE_ROSTER")
        for name, value in records.items():
            self.file(root, name, value)
        if self.retained_root is None:
            target = self.effective(root)
            paths = list(target.rglob("*"))
            require(not any(path.is_symlink() for path in paths) and
                    {p.relative_to(target).as_posix() for p in paths if p.is_file()} == set(records),
                    "D1_EXACT_OWNER_FILE_ROSTER")

    def absent(self, paths):
        require(type(paths) is list and len(paths) == len(set(paths)), "D1_ABSENT_PATH_ROSTER")
        for name in paths:
            path = self.effective(name)
            require(not path.exists() and not path.is_symlink(), "D1_UNOBSERVABLE_HAS_RETURN_OR_TERMINAL")


def _d1_accounted_owner(owner, reader):
    require(type(owner) is dict and set(owner) == {"origin", "job_id", "stage", "job_recipe",
        "queue_root", "queue_files", "job_root", "job_files", "result", "outputs_root",
        "source_root", "source_files"} and owner["origin"] in ("original-timed-out", "continuation-completed", "asr-completed"),
        "D1_ACCOUNTED_OWNER_SCHEMA")
    queue_root, job_root = owner["queue_root"], owner["job_root"]
    require(job_root == str(PurePosixPath(queue_root) / "jobs" / owner["job_id"]) and
            owner["outputs_root"] == str(PurePosixPath(job_root) / "outputs") and
            set(owner["queue_files"]) == {"queue.json", "request.json", "active-plan.json"},
            "D1_ACCOUNTED_OWNER_PATHS")
    bodies = {}
    for name, pin in owner["queue_files"].items():
        require(pin["path"] == str(PurePosixPath(queue_root) / name), "D1_QUEUE_FILE_LOCATION")
        bodies[name] = reader.pin(pin)
    queue, request = loads(bodies["queue.json"]), loads(bodies["request.json"])
    require(queue["schema"] == request["schema"] == "kws-local-research-queue-v1" and
            queue["request_sha256"] == sha(bodies["request.json"]) and
            queue["active_plan_sha256"] == sha(bodies["active-plan.json"]) and
            queue["automatic_retries"] == queue["historical_jobs_imported"] == 0,
            "D1_QUEUE_ROOT_BINDING")
    for name, expected in owner["source_files"].items():
        reader.file(owner["source_root"], name, expected)
    require(set(queue["source_sha256"]) == {"research_queue.py", "research_job_lock.py"} and
            all(owner["source_files"]["tools/" + name]["sha256"] == expected
                for name, expected in queue["source_sha256"].items()), "D1_QUEUE_CONTROLLER_SOURCE")
    jobs = {row["job_id"]: row for row in request["jobs"]}
    require(len(jobs) == len(request["jobs"]) and owner["job_id"] in jobs, "D1_QUEUE_JOB_ROSTER")
    job = jobs[owner["job_id"]]
    recipe_raw = reader.pin(owner["job_recipe"])
    require(owner["job_recipe"]["path"] == str(PurePosixPath(queue_root) / "recipes" / (job["job_id"] + ".json")) and
            job["recipe_sha256"] == sha(recipe_raw), "D1_QUEUE_FROZEN_RECIPE")
    reader.roster(job_root, owner["job_files"])
    require({"claim.json", "result.json", "dependencies.json", "stdout.log", "stderr.log",
             "worker-started.json", "command-started.json"} <= set(owner["job_files"]), "D1_QUEUE_RECEIPTS_REQUIRED")
    leaf = lambda name: reader.file(job_root, name, owner["job_files"][name])
    claim_raw, terminal_raw, dependency_raw = leaf("claim.json"), leaf("result.json"), leaf("dependencies.json")
    claim, terminal, dependencies = map(loads, (claim_raw, terminal_raw, dependency_raw))
    binding = {"job_id": job["job_id"], "queue_sha256": sha(bodies["queue.json"]),
               "job_definition_sha256": sha(canonical(job) + b"\n"), "recipe_sha256": sha(recipe_raw)}
    require(claim["schema"] == "kws-local-research-job-claim-v1" and claim["attempt"] == 1 and
            claim["automatic_retry"] is False and all(claim[key] == value for key, value in binding.items()) and
            claim["dependencies_sha256"] == sha(dependency_raw) and
            type(claim["allowed_runner_nanoseconds"]) is int and 0 < claim["allowed_runner_nanoseconds"] <= 600_000_000_000 and
            claim["deadline_monotonic_ns"] == claim["started_monotonic_ns"] + claim["allowed_runner_nanoseconds"],
            "D1_QUEUE_CALL_CLAIM")
    require(owner["result"]["path"] == str(PurePosixPath(job_root) / "result.json") and
            reader.pin(owner["result"]) == terminal_raw and
            terminal["schema"] == "kws-local-research-job-result-v1" and terminal["claim_sha256"] == sha(claim_raw) and
            all(terminal[key] == value for key, value in binding.items()) and
            terminal["automatic_retry"] is False and terminal["actual_duration_known"] is True and
            type(terminal["actual_elapsed_ns"]) is int and terminal["actual_elapsed_ns"] > 0 and
            terminal["allowed_runner_nanoseconds"] == claim["allowed_runner_nanoseconds"] and
            terminal["cwd"] == job["cwd"] == owner["source_root"], "D1_QUEUE_TERMINAL_BINDING")
    expanded = [part.replace("{recipe}", owner["job_recipe"]["path"]).replace("{output_dir}", owner["outputs_root"])
                for part in job["argv"]]
    require(terminal["argv"] == expanded and set(terminal["logs"]) == {"stdout.log", "stderr.log"},
            "D1_QUEUE_ACTUAL_INVOCATION")
    for name, expected in terminal["logs"].items():
        require(owner["job_files"][name] == expected, "D1_QUEUE_LOG_HASH")
    if owner["origin"] == "original-timed-out":
        require(terminal["status"] == "timed_out" and terminal["exit_code"] == -9 and
                terminal["reason"] == "actual_runner_deadline" and terminal["outputs"] == {} and
                terminal["actual_elapsed_ns"] == 600_186_647_856 and
                claim["allowed_runner_nanoseconds"] == 600_000_000_000 and dependencies == {},
                "D1_REAL_ORIGINAL_TIMEOUT_REQUIRED")
    else:
        require(terminal["status"] == "completed" and terminal["exit_code"] == 0 and
                set(terminal["outputs"]) == set(job["outputs"]), "D1_REAL_CONTINUATION_COMPLETION_REQUIRED")
        for name, expected in terminal["outputs"].items():
            require(owner["job_files"]["outputs/" + relative(name)] == expected, "D1_QUEUE_COMPLETED_OUTPUT_HASH")
    for name in ("worker-started.json", "command-started.json"):
        require(loads(leaf(name))["claim_sha256"] == sha(claim_raw), "D1_QUEUE_PROCESS_CLAIM_BINDING")
    return {"owner": owner, "stage": reader.json(owner["stage"]), "recipe": loads(recipe_raw),
            "queue": queue, "request": request, "job": job, "claim": claim,
            "terminal": terminal, "dependencies": dependencies}


def _d1_accounted_call(raw, cell, ref, binding):
    call = loads(raw)
    checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
    expected = {"schema": "active-tts-generation-call-v1", "cell_id": cell["cell_id"],
        "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
        "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
        "reference_text_status": ref["reference_text_status"],
        "reference_mode": "ICL" if checked else "x-vector-only", "reference_text_passed": checked,
        "seed": cell["seed"], "intended_text": cell["prompt"], "generation": binding["generation"], "attempt": 1}
    require(set(call) == set(expected) | {"reference_prompt_seconds"} and
            all(call[key] == value for key, value in expected.items()) and
            type(call["reference_prompt_seconds"]) in (int, float) and call["reference_prompt_seconds"] >= 0,
            "D1_ACCOUNTED_CALL_REFERENCE_BINDING")


def _d1_accounted_claim(pin, claim, cell, ref, binding, owner, ledger, reader):
    identity = sha(canonical({"binding": binding, "cell": cell, "reference": ref}))
    require(set(claim) == {"schema", "cell_id", "identity_sha256", "attempt", "owner", "started_at"} and
            claim["schema"] == "active-tts-plan-call-claim-v1" and claim["cell_id"] == cell["cell_id"] and
            claim["identity_sha256"] == identity and claim["attempt"] == 1 and
            claim["owner"] == {"stage_sha256": owner["stage"]["sha256"], "job_id": owner["job_id"],
                              "job_recipe_sha256": owner["job_recipe"]["sha256"]} and
            pin["path"] == str(PurePosixPath(ledger) / "claims" / (cell["cell_id"] + ".json")) and
            reader.pin(pin) == canonical(claim) + b"\n", "D1_ACCOUNTED_ORIGINAL_CLAIM")
    return identity


def _d1_accounted_eligibility(record, reader):
    """Reproduce the frozen export geometry, including its explicit rejection reason."""
    if record["native_wav"] is None:
        return {"eligible": False, "reason": "NO_DECODABLE_NATIVE_WAV"}
    body = reader.file(record["output_root"], record["native_wav"], record["files"][record["native_wav"]])
    try:
        require(44 <= len(body) <= 16_000_000, "ASR_WAV_FILE_BOUND")
        require(body[:4] == b"RIFF" and body[8:12] == b"WAVE" and
                struct.unpack_from("<I", body, 4)[0] + 8 == len(body), "ASR_RIFF_GEOMETRY")
        offset, fmt, data = 12, None, None
        while offset < len(body):
            require(offset + 8 <= len(body), "ASR_RIFF_CHUNK_BOUND")
            name, size = body[offset:offset + 4], struct.unpack_from("<I", body, offset + 4)[0]
            offset += 8
            require(offset + size <= len(body), "ASR_RIFF_CHUNK_BOUND")
            chunk = body[offset:offset + size]
            if name == b"fmt ":
                require(fmt is None and len(chunk) >= 16, "ASR_FORMAT_CHUNK")
                fmt = struct.unpack_from("<HHIIHH", chunk)
            elif name == b"data":
                require(data is None, "ASR_DUPLICATE_DATA_CHUNK")
                data = chunk
            offset += size + size % 2
        require(offset == len(body) and fmt == (3, 1, 24000, 96000, 4, 32) and data is not None,
                "ASR_MONO_FLOAT_24000_REQUIRED")
        require(len(data) % 4 == 0 and 1 <= len(data) // 4 <= 720000, "ASR_FRAME_BOUND")
        maximum = 0.0
        for (sample,) in struct.iter_unpack("<f", data):
            require(math.isfinite(sample), "ASR_NONFINITE_SAMPLE")
            maximum = max(maximum, abs(sample))
        require(maximum <= 1_000_000, "ASR_UNSAFE_NUMERIC_RANGE")
        return {"eligible": True, "reason": None, "frames": len(data) // 4,
                "native_wav_sha256": record["files"][record["native_wav"]]["sha256"]}
    except (ValueError, struct.error) as error:
        return {"eligible": False, "reason": str(error)}


def _d1_accounted_return(row, owner, cell, ref, plan, binding, ledger, reader):
    require(type(row) is dict and set(row) == {"cell_id", "origin", "owner_id", "ledger_record_sha256",
        "record_pin", "claim_pin", "record", "claim", "eligibility"} and row["cell_id"] == cell["cell_id"] and
        row["origin"] == owner["origin"], "D1_ACCOUNTED_RETURN_SCHEMA")
    name, record = cell["cell_id"], row["record"]
    identity = _d1_accounted_claim(row["claim_pin"], row["claim"], cell, ref, binding, owner, ledger, reader)
    require(record["schema"] == "active-tts-plan-cell-v1" and record["cell_id"] == name and
            record["identity_sha256"] == identity and record["origin"] == "extension-call" and
            record["new_generation_calls"] == 1 and record["training_admitted"] is False and
            record["status"] in ("GENERATED_PENDING_ASR", "FAILED_NO_RETRY") and
            record["claim_sha256"] == row["claim_pin"]["sha256"] and
            record["output_root"] == owner["outputs_root"] and
            record["generation_root"] == str(PurePosixPath(record["output_root"]) / "generation") and
            row["record_pin"]["path"] == str(PurePosixPath(ledger) / "cells" / (name + ".json")) and
            row["ledger_record_sha256"] == row["record_pin"]["sha256"] and
            reader.pin(row["record_pin"]) == canonical(record) + b"\n", "D1_ACCOUNTED_RETURN_IDENTITY")
    native, generated = "native/" + name + "/", "generation/cells/" + name + "/"
    files = record["files"]
    actual = {path.removeprefix("outputs/") for path in owner["job_files"]
              if path.startswith(("outputs/" + native, "outputs/" + generated))}
    require(type(files) is dict and actual == set(files) and
            all(owner["job_files"]["outputs/" + path] == value for path, value in files.items()),
            "D1_ACCOUNTED_OWNER_NATIVE_ROSTER")
    required = {native + leaf for leaf in ("call.json", "native.json", "returned.json", "returned-array.npy")}
    require(required | {generated + "started.json"} <= set(files), "D1_ACCOUNTED_REAL_RETURN_REQUIRED")
    bodies = {path: reader.file(record["output_root"], path, value) for path, value in files.items()}
    _d1_accounted_call(bodies[native + "call.json"], cell, ref, binding)
    _d1_started(bodies[generated + "started.json"], cell, ref, binding["plan_canonical_sha256"])
    rawmeta = loads(bodies[native + "native.json"])
    require(rawmeta["schema"] == "active-tts-native-return-v1" and rawmeta["technical_gate_applied"] is False and
            rawmeta["array"] == files[native + "returned-array.npy"], "D1_ACCOUNTED_NATIVE_RETURN")
    eligibility = _d1_accounted_eligibility(record, reader)
    require(row["eligibility"] == eligibility, "D1_ACCOUNTED_ELIGIBILITY_DIFFERS_FROM_NATIVE")
    if eligibility["eligible"]:
        wav = bodies[record["native_wav"]]
        clip = ({"derived_pcm": {"frames": loads(bodies[generated + "completed.json"])["audio"]["audio_frames"]}}
                if record["status"] == "GENERATED_PENDING_ASR" else None)
        return _d1_observed_source(clip, cell, wav, float_wav_pcm(wav), ref, plan,
            binding["plan_canonical_sha256"], row, binding, {}, checkpoint_reader=reader,
            owner_files=owner["job_files"])
    # A return rejected by the input contract is still a real failed return.
    # Its original array/metadata/failure are retained; no substitute PCM is made.
    require(record["status"] == "FAILED_NO_RETRY" and generated + "completed.json" not in files and
            generated + "failed.json" in files and record["technical_failure_reason"],
            "D1_UNRECOGNIZABLE_REQUIRES_REAL_FAILURE")
    failure = loads(bodies[generated + "failed.json"])
    require(failure["schema"] == "kws-bulk-clone-failure-v1" and failure["cell_id"] == name and
            failure["status"] == "FAILED_NO_RETRY" and failure["training_admitted"] is False,
            "D1_UNRECOGNIZABLE_FAILURE_RECEIPT")
    if rawmeta["wav"] is None:
        require(record["native_wav"] is None and rawmeta.get("wav_unavailable_reason") and
                native + "returned-native.wav" not in files and
                set(files) == required | {generated + "started.json", generated + "failed.json"},
                "D1_UNRECOGNIZABLE_NO_WAV")
    else:
        require(record["native_wav"] == native + "returned-native.wav" and
                rawmeta["wav"] == files[native + "returned-native.wav"] and
                set(files) == required | {generated + "started.json", generated + "failed.json",
                                         native + "returned-native.wav"}, "D1_UNRECOGNIZABLE_WAV_BINDING")
    return None


def _d1_accounted_unknown(row, owner, cell, ref, binding, ledger, reader):
    require(set(row) == {"cell_id", "origin", "status", "owner_id", "identity_sha256", "claim_pin", "claim",
        "output_root", "artifacts", "absent_paths", "model_return_known", "may_retry", "asr_calls",
        "native_system_calls", "observed_text", "training_admitted"} and
        row["cell_id"] == cell["cell_id"] and row["origin"] == owner["origin"] == "original-timed-out" and
        row["owner_id"] == "original-232-chunk-001" and row["status"] == "unobservable-consumed" and
        row["output_root"] == owner["outputs_root"] and row["model_return_known"] is False and
        row["may_retry"] is False and row["asr_calls"] == row["native_system_calls"] == 0 and
        row["observed_text"] is None and row["training_admitted"] is False, "D1_CONSUMED_UNKNOWN_SCHEMA")
    require(row["identity_sha256"] == _d1_accounted_claim(
        row["claim_pin"], row["claim"], cell, ref, binding, owner, ledger, reader), "D1_CONSUMED_UNKNOWN_IDENTITY")
    name, root = cell["cell_id"], row["output_root"]
    native, generated = "native/" + name + "/", "generation/cells/" + name + "/"
    require(set(row["artifacts"]) == {native + "call.json", generated + "started.json"} and
            {path.removeprefix("outputs/"): value for path, value in owner["job_files"].items()
             if path.startswith(("outputs/" + native, "outputs/" + generated))} == row["artifacts"],
            "D1_CONSUMED_UNKNOWN_HAS_RETURN")
    for path, expected in row["artifacts"].items():
        reader.file(root, path, expected)
    _d1_accounted_call(reader.file(root, native + "call.json", row["artifacts"][native + "call.json"]),
                       cell, ref, binding)
    _d1_started(reader.file(root, generated + "started.json", row["artifacts"][generated + "started.json"]),
                cell, ref, binding["plan_canonical_sha256"])
    absent = [str(PurePosixPath(ledger) / "cells" / (name + ".json"))] + [
        str(PurePosixPath(root) / native / leaf) for leaf in
        ("returned-array.npy", "returned-native.wav", "native.json", "returned.json")] + [
        str(PurePosixPath(root) / generated / leaf) for leaf in
        ("completed.json", "failed.json", "native-f32.wav", "audio-16k.wav")]
    require(row["absent_paths"] == absent, "D1_CONSUMED_UNKNOWN_ABSENCE_ROSTER")
    reader.absent(absent)


def _d1_prior176(pin, plan, reader):
    prior = reader.json(pin)
    plan_sha = sha(canonical(plan))
    require(prior["schema"] == "d1-sealed-prior-evidence-v1" and prior["prefix_count"] == 176 and
            prior["plan_sha256"] == plan_sha and prior["excluded_exports"] == [], "D1_ACCOUNTED_PRIOR176")
    seen = []
    for entry in prior["views"]:
        require(set(entry) == {"path", "manifest_sha256"}, "D1_ACCOUNTED_PRIOR_VIEW")
        path = str(PurePosixPath(entry["path"]) / "MANIFEST.json")
        # The prior's manifest is the external pin. Its report is enough for
        # checkpoint source accounting; verify_accounted_prior replays all audio.
        effective = reader.effective(path)
        body = read_bytes(effective, 16 * 1024 * 1024)
        require(sha(body) == entry["manifest_sha256"], "D1_ACCOUNTED_PRIOR_MANIFEST")
        manifest = loads(reader.pin({"path": path, "bytes": len(body), "sha256": entry["manifest_sha256"]}))
        report_path = str(PurePosixPath(entry["path"]) / "report.json")
        report_raw = read_bytes(reader.effective(report_path), 32 * 1024 * 1024)
        report = loads(reader.pin({"path": report_path, "bytes": len(report_raw), "sha256": manifest["report.json"]}))
        require(report["source_adapter"]["plan_sha256"] == plan_sha, "D1_ACCOUNTED_PRIOR_PLAN")
        for row in report["items"]:
            source = row["source"]
            require(source["source_plan_sha256"] == plan_sha, "D1_ACCOUNTED_PRIOR_PLAN")
            seen.append(source["source_id"])
    expected = [row["cell_id"] for row in plan["cells"][:176]]
    require(len(seen) == len(set(seen)) == 176 and set(seen) == set(expected), "D1_ACCOUNTED_PRIOR_EXACT176")
    return prior


def _d1_accounted_original(proof, protocol_pin, plan, refs, cells, reader):
    require(set(proof) == {"schema", "protocol_declaration", "old_stage", "old_terminal", "prior_returned",
        "returned_rows", "unobservable_consumed", "owner", "never_claimed_ids_at_registration", "binding",
        "plan", "ledger", "old_source_root", "old_source_files"} and
        proof["schema"] == "active-tts-original-partial-proof-v2" and
        proof["protocol_declaration"] == protocol_pin, "D1_ORIGINAL_PARTIAL_PROOF_SCHEMA")
    protocol = reader.json(protocol_pin)
    ids = [row["cell_id"] for row in plan["cells"]]
    require(protocol["schema"] == "kws-d1-post-timeout-continuation-predeclaration-v1" and
            protocol["new_generation_maximum_calls"] == 58 and protocol["pause_after_jobs"] == 3 and
            protocol["known_returned_cells"] == 197 and protocol["old_stage_returned_cells"] == 21 and
            protocol["old_stage_consumed_calls"] == 22 and
            protocol["first_pause_requires_ASR_accounted232_before_last_two_jobs"] is True and
            all(protocol[key] is False for key in ("unknown_receives_asr_or_native_call",
                "unknown_becomes_failed_native", "unknown_has_observed_text_or_pcm", "unknown_is_negative")) and
            protocol["never_claimed_ids"] == proof["never_claimed_ids_at_registration"] == ids[198:] and
            protocol["consumed_unknown_ids"] == [ids[197]], "D1_ORIGINAL_PROTOCOL_SCOPE")
    budget = protocol["budget"]
    require(budget["prior_actual_elapsed_ns"] == 600_186_647_856 and
            budget["new_queue_global_minutes"] == 29.9 and budget["new_queue_global_ns"] == 1_794_000_000_000 and
            budget["combined_allocation_ns"] == 2_400_000_000_000 and budget["per_job_maximum_ns"] == 600_000_000_000 and
            budget["combined_declared_actual_ceiling_ns"] == 2_394_186_647_856 and
            budget["cleanup_reserve_ns"] == 5_813_352_144, "D1_ORIGINAL_CONTINUATION_BUDGET")
    for key in ("old_tts_stage", "old_queue_terminal", "partial_audit", "recovery_contract_review",
                "original_training_protocol"):
        reader.pin(protocol[key])
    require(proof["old_stage"] == protocol["old_tts_stage"] and
            proof["old_terminal"] == protocol["old_queue_terminal"], "D1_ORIGINAL_PROTOCOL_PINS")
    owner = proof["owner"]
    require(owner["origin"] == "original-timed-out" and owner["job_id"] == "chunk-001" and
            owner["stage"] == proof["old_stage"] and owner["result"] == proof["old_terminal"] and
            owner["source_root"] == proof["old_source_root"] and owner["source_files"] == proof["old_source_files"],
            "D1_ORIGINAL_OWNER_BINDING")
    verified = _d1_accounted_owner(owner, reader)
    stage, definition = verified["stage"], verified["request"]
    old_sources = {"tools/" + name for name in ("active_tts_expand.py", "active_tts_smoke.py",
        "research_job_lock.py", "research_no_network.py", "research_runtime.py", "research_queue.py")}
    require(set(owner["source_files"]) == old_sources and
            stage["schema"] == "active-tts-extension-stage-v1" and
            stage["source_revision"] == "43ffea59304121691a84424236543c00a56de588" and
            stage["target_count"] == 232 and stage["target_cells"] == ids[:232] and
            len(stage["initial_reused_cells"]) == 176 and set(stage["initial_reused_cells"]) == set(ids[:176]) and
            stage["chunks"] == [ids[176:204], ids[204:232]] and stage["maximum_new_calls"] == 56 and
            stage["stage_budget_minutes"] == 20 and stage["maximum_job_seconds"] == 600 and stage["max_retries"] == 0 and
            stage["training_admitted"] is False and stage["paired_cases_included"] is False and
            stage["execution_root"] == owner["source_root"] and
            stage["execution_source"] == {key: owner["source_files"][key] for key in old_sources - {"tools/research_queue.py"}} and
            stage["binding"] == proof["binding"] and stage["ledger"] == proof["ledger"] and
            stage["queue_root"] == owner["queue_root"], "D1_ORIGINAL_STAGE_SCOPE")
    require([row["job_id"] for row in definition["jobs"]] == ["chunk-001", "chunk-002"] and
            definition["global_budget_minutes"] == 20 and verified["job"]["timeout_minutes"] == 10 and
            verified["claim"]["global_actual_before_nanoseconds"] == 0, "D1_ORIGINAL_QUEUE_SCOPE")
    expected_job = {"schema": "active-tts-extension-job-v1", "stage": proof["old_stage"]["path"],
        "stage_sha256": proof["old_stage"]["sha256"], "job_id": "chunk-001", "chunk_index": 0,
        "cells": ids[176:204], "maximum_new_calls": 28, "timeout_seconds": 600}
    require(verified["recipe"] == expected_job, "D1_ORIGINAL_RECIPE_SCOPE")
    audit = reader.json(protocol["partial_audit"])
    require(audit["schema"] == "active-tts-timed-out-partial-independent-readback-v1" and
            audit["stage"] == proof["old_stage"] and audit["new_terminal_cells"] == 21 and
            audit["unknown_cell_ids"] == [ids[197]] and audit["native_arrays_read_back"] == 21 and
            audit["derived_pcm_read_back"] == 18 and audit["model_calls_during_audit"] == 0,
            "D1_ORIGINAL_PARTIAL_READBACK")
    reader.roster(stage["prepared"], audit["original_stage_files"])
    require(not any(name.startswith("queue/jobs/chunk-002/") for name in audit["original_stage_files"]),
            "D1_ORIGINAL_SECOND_JOB_WAS_STARTED")
    reader.absent([str(PurePosixPath(owner["queue_root"]) / "jobs/chunk-002" / name)
                   for name in ("claim.json", "result.json")])
    require([row["cell_id"] for row in proof["prior_returned"]] == ids[:176] and
            [row["cell_id"] for row in proof["returned_rows"]] == ids[176:197] and
            len(proof["unobservable_consumed"]) == 1, "D1_ORIGINAL_PARTIAL_ROSTERS")
    for row in proof["prior_returned"]:
        require(set(row) == {"cell_id", "record_pin"} and row["record_pin"]["path"] ==
                str(PurePosixPath(proof["ledger"]) / "cells" / (row["cell_id"] + ".json")),
                "D1_ORIGINAL_PRIOR_RECORD_PIN")
        record = reader.json(row["record_pin"])
        cell = cells[row["cell_id"]]
        require(record["cell_id"] == cell["cell_id"] and
                record["identity_sha256"] == sha(canonical({"binding": proof["binding"], "cell": cell,
                    "reference": refs[cell["reference_id"]]})) and record["training_admitted"] is False and
                record["status"] in ("GENERATED_PENDING_ASR", "FAILED_NO_RETRY"), "D1_ORIGINAL_PRIOR_RECORD")
    _d1_accounted_unknown(proof["unobservable_consumed"][0], owner, cells[ids[197]],
        refs[cells[ids[197]]["reference_id"]], proof["binding"], proof["ledger"], reader)
    return protocol, verified


def _d1_accounted_checkpoint(raw, expected_sha, reader, *, previous=False):
    require(type(expected_sha) is str and SHA.fullmatch(expected_sha) and sha(raw) == expected_sha,
            "D1_ACCOUNTED_CHECKPOINT_PIN")
    value = loads(raw)
    require(raw == canonical(value) + b"\n" and set(value) == {"schema", "declaration", "registration",
        "protocol_declaration", "prior_evidence", "old_partial_proof", "plan", "plan_canonical_sha256", "binding",
        "prefix_count", "prior_count", "planned_cell_ids", "prior_cell_ids", "owners", "returned_rows",
        "unobservable_consumed", "eligible_cell_ids", "excluded_native_rows", "accounted_cell_ids", "budget",
        "previous_checkpoint", "new_model_calls_during_checkpoint", "training_admitted"} and
        value["schema"] == "active-tts-accounted-checkpoint-v2" and value["prefix_count"] in (232, 256) and
        value["prior_count"] == 176 and value["new_model_calls_during_checkpoint"] == 0 and
        value["training_admitted"] is False, "D1_ACCOUNTED_CHECKPOINT_SCHEMA")
    plan_raw = reader.pin(value["plan"])
    plan, refs, cells = _d1_plan(plan_raw, value["plan_canonical_sha256"])
    ids = [row["cell_id"] for row in plan["cells"]]
    prefix = value["prefix_count"]
    require(len(ids) == 256 and value["planned_cell_ids"] == value["accounted_cell_ids"] == ids[:prefix] and
            value["prior_cell_ids"] == ids[:176], "D1_ACCOUNTED_CHECKPOINT_PREFIX")
    binding = value["binding"]
    require(set(binding) == {"plan_canonical_sha256", "plan_file_sha256", "model_assets_sha256",
        "source_manifest_sha256", "producer_source_sha256", "runtime_setup_sha256", "generation"} and
        binding["plan_canonical_sha256"] == value["plan_canonical_sha256"] and
        binding["plan_file_sha256"] == sha(plan_raw) and type(binding["generation"]) is dict and
        all(type(item) is str and SHA.fullmatch(item) for key, item in binding.items() if key != "generation"),
        "D1_ACCOUNTED_PLAN_BINDING")
    d = reader.json(value["declaration"])
    proof = reader.json(value["old_partial_proof"])
    protocol, old_verified = _d1_accounted_original(proof, value["protocol_declaration"], plan, refs, cells, reader)
    require(proof["plan"] == value["plan"] and proof["binding"] == binding and
            value["owners"].get("original-232-chunk-001") == proof["owner"] and
            value["returned_rows"][:21] == proof["returned_rows"] and
            value["unobservable_consumed"] == proof["unobservable_consumed"], "D1_ACCOUNTED_OLD_ROWS_UNCHANGED")
    chunks = [ids[198:210], ids[210:222], ids[222:232], ids[232:244], ids[244:256]]
    require(d["schema"] == "active-tts-unclaimed-continuation-v2" and
            value["declaration"]["path"] == str(PurePosixPath(d["prepared"]) / "declaration.json") and
            d["queue_root"] == str(PurePosixPath(d["prepared"]) / "queue") and
            d["protocol_declaration"] == value["protocol_declaration"] == proof["protocol_declaration"] and
            d["old_partial_proof"] == value["old_partial_proof"] and
            d["old_stage"] == proof["old_stage"] and d["old_terminal"] == proof["old_terminal"] and
            d["prior_evidence"] == value["prior_evidence"] and d["binding"] == binding and
            d["ledger"] == proof["ledger"] and d["heavy_lock"] == old_verified["stage"]["heavy_lock"] and
            d["chunks"] == protocol["new_chunks"] == chunks and d["consumed_unknown_ids"] == [ids[197]] and
            d["maximum_new_calls"] == 58 and d["new_queue_global_minutes"] == 29.9 and
            d["per_job_maximum_seconds"] == 600 and d["combined_allocation_ns"] == 2_400_000_000_000 and
            d["prior_actual_elapsed_ns"] == 600_186_647_856 and d["pause_after_jobs"] == 3 and
            d["max_retries"] == 0 and d["further_continuation_allowed"] is False and
            d["paired_cases_included"] is False and d["training_admitted"] is False and
            d["registration_path"] == str(PurePosixPath(d["ledger"]) / "continuations" / (d["old_stage"]["sha256"] + ".json")),
            "D1_ACCOUNTED_CONTINUATION_SCOPE")
    required_sources = set(proof["old_source_files"]) | {"tools/research_tts_partial.py",
        "tools/active_tts_continuation.py", "configs/research/active-plan-20261105.json"}
    require(required_sources <= set(d["execution_source"]), "D1_ACCOUNTED_VERIFIER_SOURCE_ROSTER")
    for name, record in d["execution_source"].items():
        reader.file(d["execution_root"], name, record)
        reader.file(str(PurePosixPath(d["prepared"]) / "execution-source"), name, record)
    prior = _d1_prior176(value["prior_evidence"], plan, reader)
    registration = reader.json(value["registration"])
    queue_request_raw = reader.pin(registration["queue_request"])
    request = loads(queue_request_raw)
    expected_registration = {"schema": "active-tts-single-continuation-registration-v2",
        "declaration": value["declaration"], "queue_request": registration["queue_request"],
        "protocol_declaration": value["protocol_declaration"], "original_stage_sha256": d["old_stage"]["sha256"],
        "original_terminal_sha256": d["old_terminal"]["sha256"], "never_claimed_cell_ids": ids[198:],
        "permanently_consumed_unknown_ids": [ids[197]], "maximum_new_calls": 58,
        "automatic_retry": False, "further_continuation_allowed": False, "new_queue_budget_ns": 1_794_000_000_000,
        "old_actual_ns": 600_186_647_856, "combined_allocation_ns": 2_400_000_000_000}
    require(registration == expected_registration and value["registration"]["path"] == d["registration_path"] and
            registration["queue_request"]["path"] == str(PurePosixPath(d["prepared"]) / "queue-request.json") and
            reader.pin({"path": str(PurePosixPath(d["prepared"]) / "registration.json"),
                        "bytes": value["registration"]["bytes"], "sha256": value["registration"]["sha256"]}) ==
            canonical(registration) + b"\n", "D1_ACCOUNTED_SINGLE_REGISTRATION")
    count = 3 if prefix == 232 else 5
    owner_ids = ["continuation-" + str(i + 1).zfill(3) for i in range(5)]
    require(set(value["owners"]) == {"original-232-chunk-001", *owner_ids[:count]} and
            [row["cell_id"] for row in value["returned_rows"]] == ids[176:197] + ids[198:prefix],
            "D1_ACCOUNTED_OWNER_RETURN_ROSTER")
    require(request["schema"] == "kws-local-research-queue-v1" and request["queue_id"] == d["continuation_id"] and
            request["global_budget_minutes"] == 29.9 and
            [row["job_id"] for row in request["jobs"]] == owner_ids, "D1_ACCOUNTED_QUEUE_REQUEST")
    expected_rows, actual_before, verified_owners = list(proof["returned_rows"]), 0, {}
    for index, name in enumerate(owner_ids):
        requested = request["jobs"][index]
        # Read the declared recipe using its actual request pin; its byte length
        # is measured, never inferred from an unrelated checkpoint or source.
        recipe_path = requested["recipe"]
        recipe_body = read_bytes(reader.effective(recipe_path))
        recipe = loads(reader.pin({"path": recipe_path, "bytes": len(recipe_body),
                                  "sha256": requested["recipe_sha256"]}))
        expected_recipe = {"schema": "active-tts-unclaimed-job-v2",
            "declaration": value["declaration"]["path"], "declaration_sha256": value["declaration"]["sha256"],
            "job_id": name, "chunk_index": index, "cells": chunks[index], "maximum_new_calls": len(chunks[index]),
            "timeout_seconds": 600, "requires_asr232_handoff": index >= 3}
        require(recipe_path == str(PurePosixPath(d["prepared"]) / "jobs" / (name + ".json")) and
                recipe == expected_recipe and requested["timeout_minutes"] == 10 and
                requested["depends_on"] == ([] if index == 0 else [owner_ids[index - 1]]) and
                requested["cwd"] == d["execution_root"] and
                requested["argv"] == [d["runtime_python"], "-I", "-B",
                    str(PurePosixPath(d["execution_root"]) / "tools/active_tts_continuation.py"), "run",
                    "--recipe", "{recipe}", "--output", "{output_dir}", "--heavy-lock", d["heavy_lock"]] and
                requested["outputs"] == ["summary.json", "inventory.json", "invocation.json", "outcomes.json",
                    "network-guard.json", "model-loaded.json", "generation/plan-lock.json"],
                "D1_ACCOUNTED_JOB_DEFINITION")
        if index >= count:
            continue
        owner = value["owners"][name]
        require(owner["origin"] == "continuation-completed" and owner["job_id"] == name and
                owner["stage"] == value["declaration"] and owner["queue_root"] == d["queue_root"] and
                owner["source_root"] == d["execution_root"] and owner["source_files"] == d["execution_source"],
                "D1_ACCOUNTED_NEW_OWNER_BINDING")
        current = _d1_accounted_owner(owner, reader)
        require(current["request"] == request and reader.pin(owner["queue_files"]["request.json"]) == queue_request_raw and
                current["recipe"] == recipe and current["queue"]["heavy_lock"] == d["heavy_lock"] and
                current["claim"]["global_actual_before_nanoseconds"] == actual_before and
                current["claim"]["allowed_runner_nanoseconds"] == min(600_000_000_000, 1_794_000_000_000 - actual_before),
                "D1_ACCOUNTED_CUMULATIVE_QUEUE_BUDGET")
        dependencies = {}
        if index:
            previous_owner, previous_verified = value["owners"][owner_ids[index - 1]], verified_owners[owner_ids[index - 1]]
            dependencies[owner_ids[index - 1]] = {"receipt": previous_owner["result"]["path"],
                "receipt_sha256": previous_owner["result"]["sha256"], "outputs_root": previous_owner["outputs_root"],
                "outputs": previous_verified["terminal"]["outputs"]}
        require(current["dependencies"] == dependencies, "D1_ACCOUNTED_QUEUE_DEPENDENCIES")
        output_files = {path.removeprefix("outputs/"): record for path, record in owner["job_files"].items()
                        if path.startswith("outputs/")}
        output = lambda leaf: reader.file(owner["outputs_root"], leaf, output_files[leaf])
        inventory, summary, outcomes = (loads(output(leaf)) for leaf in ("inventory.json", "summary.json", "outcomes.json"))
        require(set(output_files) == set(inventory["files"]) | {"inventory.json", "summary.json"} and
                all(output_files[path] == record for path, record in inventory["files"].items()),
                "D1_ACCOUNTED_COMPLETED_OUTPUT_ROSTER")
        require(outcomes == {"schema": "active-tts-continuation-outcomes-v2", "declaration": value["declaration"],
            "registration": value["registration"], "job_id": name,
            "job_recipe": {"path": recipe_path, "bytes": len(recipe_body), "sha256": requested["recipe_sha256"]},
            "plan_canonical_sha256": value["plan_canonical_sha256"], "binding": binding,
            "selected_cells": chunks[index], "new_call_ids": chunks[index], "reused_cell_ids": [], "rows": outcomes["rows"]} and
            [row["cell_id"] for row in outcomes["rows"]] == chunks[index] and
            all(row["owner_id"] == name and row["origin"] == owner["origin"] for row in outcomes["rows"]),
            "D1_ACCOUNTED_REAL_NEW_OUTCOMES")
        require(summary["schema"] == "active-tts-continuation-result-v2" and summary["status"] == "completed" and
                summary["job_recipe_sha256"] == requested["recipe_sha256"] and
                summary["declaration_sha256"] == value["declaration"]["sha256"] and
                summary["new_generation_calls"] == len(chunks[index]) and summary["new_call_ids"] == chunks[index] and
                summary["reused_cell_ids"] == [] and summary["inventory"] == output_files["inventory.json"] and
                summary["outcomes"] == output_files["outcomes.json"], "D1_ACCOUNTED_REAL_NEW_SUMMARY")
        expected_rows.extend(outcomes["rows"])
        actual_before += current["terminal"]["actual_elapsed_ns"]
        verified_owners[name] = current
    require(value["returned_rows"] == expected_rows, "D1_ACCOUNTED_ORIGINAL_AND_NEW_ROWS")
    expected_budget = {"old_actual_elapsed_ns": 600_186_647_856, "new_actual_known_ns": actual_before,
        "combined_actual_known_ns": 600_186_647_856 + actual_before, "new_actual_total_complete": True,
        "new_queue_allocation_ns": 1_794_000_000_000, "combined_allocation_ns": 2_400_000_000_000,
        "combined_declared_ceiling_ns": 2_394_186_647_856, "cleanup_reserve_ns": 5_813_352_144,
        "new_queue_overrun_ns": max(0, actual_before - 1_794_000_000_000),
        "observed_combined_overrun_ns": max(0, 600_186_647_856 + actual_before - 2_400_000_000_000),
        "selected_job_count": count, "completed_job_count": count,
        "within_new_queue_allocation": actual_before <= 1_794_000_000_000,
        "within_combined_allocation": 600_186_647_856 + actual_before <= 2_400_000_000_000,
        "historical_original_smoke_actual_total_ns": None, "historical_original_smoke_actual_total_complete": False}
    require(value["budget"] == expected_budget, "D1_ACCOUNTED_ACTUAL_BUDGET")
    sources = {}
    for row in value["returned_rows"]:
        cell = cells[row["cell_id"]]
        source = _d1_accounted_return(row, value["owners"][row["owner_id"]], cell, refs[cell["reference_id"]],
                                     plan, binding, proof["ledger"], reader)
        if source is not None:
            sources[row["cell_id"]] = source
    require(value["eligible_cell_ids"] == list(sources) and value["excluded_native_rows"] == [
        {"cell_id": row["cell_id"], "reason": row["eligibility"]["reason"], "ledger_record_sha256": row["ledger_record_sha256"]}
        for row in value["returned_rows"] if row["cell_id"] not in sources], "D1_ACCOUNTED_ELIGIBLE_EXCLUDED_PARTITION")
    if prefix == 232:
        require(value["previous_checkpoint"] is None, "D1_ACCOUNTED_FIRST_CHECKPOINT")
    else:
        require(not previous and type(value["previous_checkpoint"]) is dict, "D1_ACCOUNTED_PREVIOUS_CHECKPOINT")
        earlier, _, _, _, _ = _d1_accounted_checkpoint(reader.pin(value["previous_checkpoint"]),
            value["previous_checkpoint"]["sha256"], reader, previous=True)
        require(earlier["prefix_count"] == 232 and
                all(earlier[key] == value[key] for key in ("declaration", "registration", "protocol_declaration",
                    "prior_evidence", "old_partial_proof", "plan", "plan_canonical_sha256", "binding",
                    "unobservable_consumed")) and
                earlier["returned_rows"] == value["returned_rows"][:55] and
                all(value["owners"][name] == owner for name, owner in earlier["owners"].items()),
                "D1_ACCOUNTED_PREVIOUS_CHECKPOINT_CHANGED")
    return value, plan, refs, sources, prior


def d1_source_rows_v3(archive_raw, inputs, checkpoint_raw, expected_checkpoint_sha, *,
                      artifact_roots=None, retained_root=None):
    """Consume only authenticated real returns from the separately typed continuation."""
    reader = _D1CheckpointEvidence(roots=artifact_roots, retained_root=retained_root)
    value, plan, _, by_cell, _ = _d1_accounted_checkpoint(checkpoint_raw, expected_checkpoint_sha, reader)
    cells = {row["cell_id"]: row for row in plan["cells"]}
    new_ids = set(value["eligible_cell_ids"])
    if value["previous_checkpoint"] is not None:
        previous = reader.json(value["previous_checkpoint"])
        new_ids -= {row["cell_id"] for row in previous["returned_rows"]}
    sources = {}
    for clip, cell, wav, pcm in _d1_audio_rows(archive_raw, inputs, cells):
        name = cell["cell_id"]
        require(name in new_ids and name in by_cell, "D1_ACCOUNTED_ARCHIVE_NEW_OBSERVABLE_ONLY")
        row = next(row for row in value["returned_rows"] if row["cell_id"] == name)
        record = row["record"]
        require(reader.file(record["output_root"], record["native_wav"], record["files"][record["native_wav"]]) == wav,
                "D1_ACCOUNTED_ARCHIVE_NATIVE_JOIN")
        if record["status"] == "GENERATED_PENDING_ASR":
            require(by_cell[name]["original_generation_row"]["audio"]["audio_frames"] == clip["derived_pcm"]["frames"],
                    "D1_GENERATED_WAV_PCM_JOIN")
        sources[clip["id"]] = by_cell[name]
    reader.evidence["d1/plan.json"] = reader.pin(value["plan"])
    reader.evidence["d1/accounted/checkpoint.json"] = checkpoint_raw
    return sources, reader.evidence, {"schema": D1_ADAPTER_V3, "plan_sha256": value["plan_canonical_sha256"],
        "plan_file_sha256": value["plan"]["sha256"], "checkpoint_sha256": expected_checkpoint_sha,
        "accounted_prefix_count": value["prefix_count"], "declaration_sha256": value["declaration"]["sha256"],
        "original_timeout_sha256": value["owners"]["original-232-chunk-001"]["result"]["sha256"],
        "unknown_is_not_an_observation": True}


def add_source_role(roles, source, clip, measured=None):
    roles.add(source["role"], family=source["source_family_id"], speaker=source["speaker_id"],
              wav_sha256=clip["original_wav"]["sha256"], pcm_sha256=clip["original_pcm"]["sha256"])
    if "reference_id" in source:
        roles.add(source["role"], wav_sha256=source["reference_wav_sha256"],
                  pcm_sha256=source["reference_pcm_sha256"], parent_source_sha256=source["parent_source_sha256"])
        if source.get("generation_outcome") != "failed-native":
            original_derived = source["original_generation_row"]["audio"]
            roles.add(source["role"], wav_sha256=original_derived["wav_sha256"],
                      pcm_sha256=original_derived["pcm_sha256"])
    if measured is not None:
        roles.add(source["role"], pcm_sha256=measured["pcm_sha256"], wav_sha256=measured["file_sha256"])


def observation(asr_root, model, clip, inputs_sha, retained):
    """Validate one saved result/native/claim binding, retaining rejected evidence."""
    paths = {kind: f"{kind}/{model}/{clip['id']}.json" for kind in ("results", "native", "claims")}
    result = None
    try:
        bodies = {}
        for kind, name in paths.items():
            try:
                body = read_bytes(asr_root / name, 2 * 1024 * 1024)
            except FileNotFoundError:
                continue
            retained["evidence/asr/" + name] = body
            bodies[kind] = body
        result = loads(bodies["results"]) if "results" in bodies else None
        if set(bodies) != set(paths):
            raise FileNotFoundError("incomplete saved ASR evidence")
        native, claim = (loads(bodies[key]) for key in ("native", "claims"))
        identity = result["model_identity"]
        expected = sha(canonical({"model_identity": identity, "input": clip}))
        for value, schema in ((result, "active-asr-result-v1"),
                              (native, "active-asr-native-output-v1"),
                              (claim, "active-asr-call-claim-v1")):
            require(value["schema"] == schema and value["identity_sha256"] == expected and
                    value["model"] == model and value["clip_id"] == clip["id"], "ASR_IDENTITY")
        require(result["status"] == "complete" and result["input"] == claim["input"] == clip and
                identity == claim["model_identity"] and identity["model"] == model and
                identity["input_manifest_sha256"] == inputs_sha and
                result["attempt"] == claim["attempt"] and
                result["native_output_sha256"] == sha(bodies["native"]), "ASR_NATIVE_BINDING")
        require(identity["recipe"] == "active-asr-native-v1" and
                type(identity["model_assets"]) is dict and type(identity["runtime_lock_sha256"]) is dict and
                SHA.fullmatch(identity["runtime_packages_sha256"]) and
                SHA.fullmatch(result["decoder_source_sha256"]), "ASR_RUNTIME_IDENTITY")
        require(all(result["boundary"][key] is False for key in
                    ("human_gold", "training_admitted", "shipping_approved",
                     "target_text_received", "peer_transcript_received")) and
                result["boundary"]["acoustic_verbatim_completeness"] == "unknown",
                "ASR_OBSERVATION_BOUNDARY")
        text, payload, config = result["text"], native["native"], identity["decoder_config"]
        require(type(text) is str and len(text) <= 32768 and
                type(result["quality_flags"]) is list and
                all(type(flag) is str for flag in result["quality_flags"]) and
                type(result["decoder_terminated"]) is bool, "ASR_TEXT_OR_QUALITY")
        if model == "qwen06":
            full = payload["decoded_skip_special_tokens"]
            prefix = re.match(r"^language\s+([^<]+)<asr_text>", full)
            projected = full[prefix.end():] if prefix else full
            require(text == projected and config["context"] == "" and
                    config["force_language"] is None and
                    result["decoder_terminated"] == payload["metadata"]["eos_terminated"] and
                    result["decoder_terminated"] == (payload["token_ids"][-1] in (151645, 151643)),
                    "QWEN_TEXT_OR_CONTEXT")
        elif model == "sensevoice":
            require(len(payload["result"]) == 1 and payload["result"][0]["key"] == clip["id"] and
                    text == re.sub(r"<\|[^<>]*\|>", "", payload["result"][0]["text"]) and
                    config["language"] == "auto" and config["use_itn"] is False,
                    "SENSE_TEXT_OR_CONTEXT")
        else:
            require(text == payload["text"] and config["prompt"] is None and
                    config["prefix"] is None and config["language"] == "zh" and
                    result["decoder_terminated"] == payload["decoder_terminated"] ==
                    payload["pre_finalize_greedy_trace"][-1]["completed"],
                    "WHISPER_TEXT_OR_CONTEXT")
        valid = result["decoder_terminated"] and not result["quality_flags"] and bool(normalize(text))
        return {"model": model, "status": "complete", "usable": valid,
                "text": text, "normalized_text": normalize(text),
                "quality_flags": result["quality_flags"], "decoder_terminated": result["decoder_terminated"],
                "result_path": "evidence/asr/" + paths["results"],
                "result_sha256": sha(bodies["results"]), "native_output_sha256": sha(bodies["native"]),
                "model_identity": identity, "boundary": result["boundary"]}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {"model": model, "status": "missing" if isinstance(error, FileNotFoundError) else "rejected",
                "usable": False, "text": result.get("text") if type(result) is dict else None,
                "reason": type(error).__name__ + (":" + str(error) if not isinstance(error, OSError) else "")}


def selective_route(source, plan_sha, primary):
    """Independently replay the predeclared primary/quality/hash routing rule."""
    reasons, texts, hashes = [], {}, {}
    for model in MODELS[:2]:
        row = primary[model]
        require(row["status"] == "complete", "SELECTIVE_PRIMARY_MUST_BE_COMPLETE_RECORD")
        texts[model] = normalize(row["text"])
        hashes[model] = row["result_sha256"]
        if not row["decoder_terminated"] or not texts[model]:
            reasons.append("primary_incomplete")
        if row["quality_flags"]:
            reasons.append("primary_quality_uncertain")
    if texts["qwen06"] != texts["sensevoice"]:
        reasons.append("primary_disagreement")
    if source["quality_uncertain"]:
        reasons.append("source_quality_uncertain")
    sample_hash = sha(("kws-d1-whisper-agreement-audit-v2|20261105|" + plan_sha + "|" +
                       source["source_id"]).encode("utf-8"))
    reasons = [reason for reason in ("primary_incomplete", "primary_quality_uncertain",
                                     "primary_disagreement", "source_quality_uncertain") if reason in reasons]
    if not reasons:
        reasons = ["deterministic_20pct" if int(sample_hash, 16) * 5 < 2**256
                   else "two_primary_agree_not_sampled"]
    return {"clip_id": source["clip_id"], "source_id": source["source_id"],
            "primary_result_sha256": hashes, "sample_hash": sample_hash,
            "whisper_state": "not_requested" if reasons == ["two_primary_agree_not_sampled"] else "requested",
            "reasons": reasons}


def selective_evidence(asr_root, inputs_raw, input_root, sources, plan_sha, source_map_raw,
                       policy_raw, expected_audit_sha, retained):
    """Replay a sealed sparse audit and retain every byte in its inventory.

    The model-application audit remains an externally pinned upstream receipt;
    this consumer independently replays native text, source quality, routing,
    the exact requested matrix, absence of unrequested calls, and loss inputs.
    No pipeline module, model, current source checkout or mutable ledger loads.
    """
    require(loads(source_map_raw) == _source_map(inputs_raw, input_root, sources, plan_sha),
            "SELECTIVE_SOURCE_MAP_CHANGED")
    return replay_selective_evidence(asr_root, inputs_raw, input_root, plan_sha, source_map_raw,
                                     policy_raw, expected_audit_sha, retained)


def replay_selective_evidence(asr_root, inputs_raw, input_root, plan_sha, source_map_raw,
                              policy_raw, expected_audit_sha, retained):
    """Read an already sealed native ASR run after the caller verifies its source map.

    Source adapters own input PCM/ancestry verification. This shared reader owns
    native text, sealed inventory, exact requested calls, routing and deadlines.
    It never creates a seal, loads a model or changes the saved evidence.
    """
    asr_root, input_root = Path(asr_root), Path(input_root)
    require(type(expected_audit_sha) is str and SHA.fullmatch(expected_audit_sha), "SELECTIVE_AUDIT_PIN_REQUIRED")
    require(sha(policy_raw) == SELECTIVE_POLICY_SHA, "SELECTIVE_FROZEN_POLICY_CHANGED")
    mapping = loads(source_map_raw)
    require(mapping["schema"] == "active-asr-source-map-v2" and
            mapping["plan_sha256"] == plan_sha and mapping["input_manifest_sha256"] == sha(inputs_raw),
            "SELECTIVE_VERIFIED_SOURCE_MAP_BINDING")
    inputs = loads(inputs_raw)

    def take(name, limit=32 * 1024 * 1024, *, empty=False):
        path = asr_root / relative(name)
        body = empty_evidence(path) if empty else read_bytes(path, limit)
        retained["evidence/asr/" + name] = body
        return body

    audit_raw = take("audit/final-summary.json")
    require(sha(audit_raw) == expected_audit_sha, "SELECTIVE_FINAL_AUDIT_CHANGED")
    audit, request_raw, routing_raw, inventory_raw = loads(audit_raw), take("request.json"), take("routing.json"), take("audit/file-hashes.json")
    request, routing, inventory = map(loads, (request_raw, routing_raw, inventory_raw))
    require(set(audit) == {"schema", "created_at", "request_sha256", "inventory_sha256", "state"} and
            audit["schema"] == "active-asr-selective-final-audit-v2" and
            audit["request_sha256"] == sha(request_raw) and audit["inventory_sha256"] == sha(inventory_raw),
            "SELECTIVE_AUDIT_REQUEST_INVENTORY_JOIN")
    require(request["schema"] == "active-asr-selective-request-v2" and
            request["inputs"]["sha256"] == sha(inputs_raw) and
            request["source_map"]["sha256"] == sha(source_map_raw) and
            request["policy"]["sha256"] == sha(policy_raw) and
            request["max_calls"] == 3 * len(inputs["clips"]) and
            type(request["budget_seconds"]) is int and 1 <= request["budget_seconds"] <= 600 and
            type(request["threads"]) is int and 1 <= request["threads"] <= 4 and
            type(request["worker_sha256"]) is str and SHA.fullmatch(request["worker_sha256"]) and
            request["source_files"]["tools/active_asr_worker.py"] == request["worker_sha256"] and
            set(request["model_identities"]) == set(MODELS), "SELECTIVE_REQUEST_BINDING")
    require(all(request["boundary"][key] is False for key in
                ("human_gold", "training_admitted", "shipping_approved", "target_text_received", "peer_transcript_received")) and
            request["boundary"]["acoustic_verbatim_completeness"] == "unknown", "SELECTIVE_NATIVE_BOUNDARY")
    require(set(inventory) == {"schema", "files_sha256", "files", "file_count"} and
            inventory["schema"] == "active-asr-selective-inventory-v2" and
            inventory["file_count"] == len(inventory["files"]) <= 4096 and
            inventory["files_sha256"] == sha(canonical(inventory["files"])), "SELECTIVE_INVENTORY_BINDING")
    output_files, input_files = set(), set()
    for name, expected in inventory["files"].items():
        prefix, path = relative(name).split("/", 1)
        empty = expected == {"bytes": 0, "sha256": sha(b"")}
        if prefix == "output":
            require(not path.startswith("audit/"), "SELECTIVE_INVENTORY_AUDIT_RECURSION")
            output_files.add(path)
            body = take(path, empty=empty)
        elif prefix == "inputs":
            input_files.add(path)
            body = (empty_evidence(input_root / relative(path)) if empty else
                    read_bytes(input_root / relative(path), 32 * 1024 * 1024))
            retained["evidence/inputs/" + path] = body
        else:
            require(prefix == "metadata" and path in ("source_map.json", "policy.json"), "SELECTIVE_INVENTORY_NAMESPACE")
            body = source_map_raw if path == "source_map.json" else policy_raw
        require(expected == {"bytes": len(body), "sha256": sha(body)}, "SELECTIVE_INVENTORY_FILE_CHANGED")
    require({p.relative_to(asr_root).as_posix() for p in asr_root.rglob("*")
             if p.is_file() and not p.relative_to(asr_root).as_posix().startswith("audit/")} == output_files and
            {p.relative_to(input_root).as_posix() for p in input_root.rglob("*") if p.is_file()} == input_files and
            {"metadata/source_map.json", "metadata/policy.json", "inputs/inputs.json",
             "output/request.json", "output/routing.json", "output/job.json", "output/network-guard.json",
             "output/attempts/selective-001/summary.json"} <= set(inventory["files"]), "SELECTIVE_EVIDENCE_ROSTER")
    retained["evidence/selective/source-map.json"] = source_map_raw
    retained["evidence/selective/policy.json"] = policy_raw
    observed, expected_routes = {}, []
    expected, skipped = set(), set()
    for clip, source in zip(inputs["clips"], mapping["rows"]):
        primary = {model: observation(asr_root, model, clip, sha(inputs_raw), retained) for model in MODELS[:2]}
        route = selective_route(source, plan_sha, primary)
        expected_routes.append(route)
        rows = list(primary.values())
        expected.update((model, clip["id"]) for model in MODELS[:2])
        if route["whisper_state"] == "requested":
            rows.append(observation(asr_root, "whisper", clip, sha(inputs_raw), retained))
            expected.add(("whisper", clip["id"]))
        else:
            rows.append({"model": "whisper", "status": "not_requested", "usable": False, "text": None,
                         "reason": "two_primary_agree_not_sampled", "routing_sha256": sha(routing_raw)})
            skipped.add(("whisper", clip["id"]))
        for row in rows:
            if row["status"] == "not_requested":
                continue
            require(row["status"] == "complete" and row["model_identity"] == request["model_identities"][row["model"]] and
                    row["boundary"] == request["boundary"] and
                    loads(retained[row["result_path"]])["decoder_source_sha256"] == request["worker_sha256"],
                    "SELECTIVE_REQUESTED_NATIVE_RESULT")
        observed[clip["id"]] = rows
    require(routing == {"schema": "active-asr-selective-routing-v2", "input_manifest_sha256": sha(inputs_raw),
                        "source_map_sha256": sha(source_map_raw), "policy_sha256": sha(policy_raw),
                        "plan_sha256": plan_sha, "decoder_source_sha256": request["worker_sha256"],
                        "items": expected_routes}, "SELECTIVE_ROUTING_DIFFERS_FROM_EVIDENCE")
    for kind in ("claims", "native", "results"):
        actual = {(p.parent.name, p.stem) for p in (asr_root / kind).glob("*/*.json")}
        require(actual == expected, "SELECTIVE_REQUESTED_MATRIX_CHANGED")
    job = loads(retained["evidence/asr/job.json"])
    require(job["schema"] == "active-asr-selective-job-v2" and
            job["identity"] == {"request_sha256": sha(request_raw), "max_calls": request["max_calls"],
                                "heavy_model_concurrency": 1, "threads": request["threads"]}, "SELECTIVE_JOB_BINDING")
    by_model = {}
    for model in MODELS:
        ids = {clip_id for engine, clip_id in expected if engine == model}
        receipt_paths = [name for name in output_files if re.fullmatch(r"attempts/[^/]+/" + model + r"\.json", name)]
        require(len(receipt_paths) == (1 if ids else 0), "SELECTIVE_WORKER_RECEIPT_ROSTER")
        if ids:
            require(receipt_paths == ["attempts/selective-001/" + model + ".json"] and
                    "attempts/selective-001/" + model + "-load.json" in output_files, "SELECTIVE_ACTUAL_MODEL_LOAD_REQUIRED")
            receipt = loads(retained["evidence/asr/" + receipt_paths[0]])
            require(receipt["schema"] == "active-asr-worker-v1" and receipt["status"] == "complete" and
                    receipt["model"] == model and receipt["attempt"] == "selective-001" and
                    receipt["identity"] == request["model_identities"][model] and
                    receipt["source_sha256"] == request["worker_sha256"] and receipt["boundary"] == request["boundary"] and
                    len(receipt["selected"]) == len(ids) and set(receipt["selected"]) == ids and
                    receipt["new_calls"] == len(ids) and receipt["reused"] == 0 and receipt["unresolved_prior_calls"] == [] and
                    receipt["heavy_lock"]["inherited"] is True and receipt["heavy_lock"]["supervised"] is True and
                    receipt["heavy_lock"]["process_group"] == job["supervision"]["process_group"], "SELECTIVE_ACTUAL_CALL_RECEIPT")
        else:
            require("attempts/selective-001/" + model + "-load.json" not in output_files, "SELECTIVE_UNREQUESTED_MODEL_LOAD")
        by_model[model] = {"requested": len(ids), "not_requested": sum(m == model for m, _ in skipped),
                           "verified": len(ids), "claims": len(ids), "worker_new_calls": len(ids)}
    attempt = loads(retained["evidence/asr/attempts/selective-001/summary.json"])
    require(attempt["status"] == "complete" and
            datetime.fromisoformat(attempt["finished_at"]).timestamp() <= job["deadline_unix"], "SELECTIVE_DEADLINE_OR_COMPLETION")
    guard = loads(retained["evidence/asr/network-guard.json"])
    require(guard["seccomp_installed"] is True and guard["no_new_privileges"] is True, "SELECTIVE_NATIVE_NETWORK_GUARD")
    require(audit["state"] == {"status": "complete", "input_manifest_sha256": sha(inputs_raw),
        "source_map_sha256": sha(source_map_raw), "policy_sha256": sha(policy_raw), "routing_sha256": sha(routing_raw),
        "decoder_source_sha256": request["worker_sha256"], "inputs": len(inputs["clips"]),
        "requested_matrix": [list(pair) for pair in sorted(expected)],
        "not_requested_matrix": [list(pair) for pair in sorted(skipped)],
        "expected_matrix_cells": len(expected), "verified_matrix_cells": len(expected),
        "original_recognition_calls": len(expected), "call_budget": request["max_calls"],
        "duplicate_model_clip_pairs": 0, "models": by_model, "errors": [], "finished_before_deadline": True,
        "boundary": request["boundary"], "unrequested_is_not_a_negative_or_agreement": True},
        "SELECTIVE_AUDIT_DIFFERS_FROM_REPLAY")
    return observed, {"schema": SELECTIVE_ADAPTER, "final_audit_sha256": expected_audit_sha,
        "request_sha256": sha(request_raw), "inventory_sha256": sha(inventory_raw),
        "source_map_sha256": sha(source_map_raw), "policy_sha256": sha(policy_raw),
        "routing_sha256": sha(routing_raw), "decoder_source_sha256": request["worker_sha256"]}


def contains_ordered(text, keyword):
    stream = iter(text)
    return all(any(char == expected for char in stream) for expected in keyword)


def decide(source, observations, audio_flags=()):
    """Whole-sentence agreement; intended positives never become automatic negatives."""
    names = [row["model"] for row in observations]
    require(len(names) == len(set(names)) and set(names) <= set(MODELS), "ASR_VOTE_MODEL_ROSTER")
    usable = [row for row in observations if row.get("usable") is True and row.get("normalized_text")]
    observed = [row for row in observations if (row.get("usable") is True or row.get("status") == "complete")
                and type(row.get("text")) is str and normalize(row["text"])]
    votes = Counter(row["normalized_text"] for row in usable)
    agreed = [text for text, count in votes.items() if count >= 2]
    decision = {"status": "uncertain", "reason": "no_two_complete_full_transcript_votes",
                "human_gold": False, "label_tier": "machine-weak", "full_text": None,
                "selected_raw_text": None, "agreeing_models": [], "ctc_mask": False,
                "ctc_weight": 0.0, "keyword_targets": [None, None],
                "keyword_mask": [False, False], "keyword_weight": [0.0, 0.0],
                "disagreement": len({normalize(row["text"]) for row in observed}) > 1,
                "acoustic_verbatim_completeness": "unknown", "event_times": None}
    if source.get("generation_outcome") == "failed-native":
        decision["reason"] = "failed_native_generation_never_auto_label"
        return decision
    if not agreed:
        return decision
    text = agreed[0]
    supporters = [row for row in usable if row["normalized_text"] == text]
    decision.update(full_text=text, selected_raw_text=supporters[0]["text"],
                    agreeing_models=[row["model"] for row in supporters])
    if audio_flags:
        decision["reason"] = "signal_quality_requires_review"
        return decision
    labels = [int(keyword in text) for keyword in KEYWORDS]
    intended = source["intended_kind"]
    positive_intent = intended == "positive-intent"
    if positive_intent and (source["intended_keyword_id"] not in (1, 2) or
                            not labels[source["intended_keyword_id"] - 1]):
        decision["reason"] = "intended_full_wake_not_confirmed_never_auto_negative"
        return decision
    if not positive_intent:
        # Byte-bound text can contradict a negative even when decoder quality
        # prevents that observation from casting a positive training vote.
        if any(contains_ordered(normalize(row["text"]), keyword)
               for row in observed for keyword in KEYWORDS):
            decision["reason"] = "negative_intent_has_possible_full_wake"
            return decision
        if intended not in ("near-miss-intent", "hard-negative-intent", "partial-intent",
                            "speech-negative-intent"):
            decision["reason"] = "unsupported_generation_intent"
            return decision
    weight = 1.0 if len(supporters) == 3 else 0.5
    decision.update(status="machine_candidate", reason="full_transcript_machine_agreement",
                    ctc_mask=True, ctc_weight=weight, keyword_targets=labels,
                    keyword_mask=[True, True], keyword_weight=[weight, weight],
                    label_tier="machine-weak-three" if len(supporters) == 3 else "machine-weak-two")
    return decision


def fsmn_vocabulary(raw):
    require(sha(raw) == FSMN_TOKENS_SHA, "FSMN_VOCABULARY_CHANGED")
    vocab = {word: int(number) - 1 for word, number in
             (line.rsplit(maxsplit=1) for line in raw.decode().splitlines())
             if 0 <= int(number) - 1 < 2599}
    require(len(set(vocab.values())) == 2599, "FSMN_VOCABULARY_CLASS_COUNT")
    return vocab


def consumer_targets(text, vocab, frames, *, frame_stride):
    if text is None:
        return {"supported": False, "reason": "no_full_transcript", "target_ids": None}
    if vocab is None:
        return {"supported": False, "reason": "vocabulary_not_supplied", "target_ids": None}
    if any(char not in vocab or vocab[char] <= 0 for char in text):
        return {"supported": False, "reason": "full_transcript_out_of_vocabulary", "target_ids": None}
    ids = [vocab[char] for char in text]
    steps = (max(0, 1 + (frames - 400) // 320) if frame_stride == 2 else
             max(0, 1 + (frames - 400) // 160) // frame_stride)
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
    if not ids or steps < minimum:
        return {"supported": False, "reason": "insufficient_ctc_frames", "target_ids": None}
    return {"supported": True, "reason": "complete_nonblank_full_transcript",
            "target_ids": ids, "expected_output_frames": steps,
            "padding_samples": 0, "frame_stride": frame_stride}


def source_quality_flags(source, measured):
    generation = source["original_generation_row"]
    if source["source_kind"] == "synthetic-reference-clone":
        flags = list(measured["quality_flags"])
        technical_pass = (False if source.get("generation_outcome") == "failed-native"
                          else generation["technical"]["technical_audio_pass"])
    else:
        flags = measured["quality_flags"] + generation["metrics"]["quality_flags"]
        technical_pass = generation["metrics"]["technically_valid"]
    if technical_pass is not True:
        flags = flags + ["original_generator_signal_invalid"]
    return flags


def make_row(clip, source, input_root, asr_root, inputs_sha, vocab, retained, *, observations=None):
    derived = read_bytes(input_root / relative(clip["derived_pcm"]["path"]), 2 * 1024 * 1024)
    original = read_bytes(input_root / relative(clip["original_wav"]["path"]), 16 * 1024 * 1024)
    minimum = 1 if source.get("generation_outcome") == "failed-native" else 400
    wav, measured = quantize_pcm16(derived, clip["derived_pcm"]["frames"], minimum_frames=minimum)
    flags = source_quality_flags(source, measured)
    observed = (observations if observations is not None else
                [observation(asr_root, model, clip, inputs_sha, retained) for model in MODELS])
    label = decide(source, observed, flags)
    row = {"item_id": clip["id"], "source": source, "input": clip, "asr": observed,
           "audio": "audio/" + clip["id"] + ".wav", "audio_derivation": measured, "decision": label,
           "consumers": {
               "rnn-four-token": consumer_targets(label["full_text"], FOUR_CHAR_IDS,
                                                   measured["frames"], frame_stride=2),
               "fsmn-fulltext": consumer_targets(label["full_text"], vocab,
                                                  measured["frames"], frame_stride=3)},
           "human_gold": False, "independent_test": False}
    return row, wav, original, derived


def verify_inputs(inputs, input_root):
    require(inputs["schema"] == "active-asr-inputs-v1" and
            type(inputs["clips"]) is list and 1 <= len(inputs["clips"]) <= 64,
            "ACTIVE_INPUTS_SCHEMA")
    ids, wavs, members = set(), set(), set()
    for clip in inputs["clips"]:
        require(set(clip) == {"id", "original_member", "original_wav", "original_pcm", "derived_pcm"} and
                re.fullmatch(r"clip-[0-9]{3}", clip["id"]) and clip["id"] not in ids and
                clip["original_wav"]["sha256"] not in wavs and clip["original_member"] not in members,
                "ACTIVE_INPUT_ROSTER")
        ids.add(clip["id"]); wavs.add(clip["original_wav"]["sha256"]); members.add(clip["original_member"])
        original = clip["original_pcm"]
        derived = clip["derived_pcm"]
        require(original["sample_rate"] == 24000 and original["channels"] == 1 and original["dtype"] == "<f4" and
                derived["sample_rate"] == 16000 and derived["channels"] == 1 and derived["dtype"] == "<f4",
                "ACTIVE_PCM_FORMAT")
        for kind, maximum in (("original_wav", 16 * 1024 * 1024), ("derived_pcm", 2 * 1024 * 1024)):
            record = clip[kind]
            raw = read_bytes(input_root / relative(record["path"]), maximum)
            require(SHA.fullmatch(record["sha256"]) and sha(raw) == record["sha256"], "ACTIVE_AUDIO_CHANGED")
            if kind == "original_wav":
                pcm = float_wav_pcm(raw)
                require(len(raw) == record["bytes"] and sha(pcm) == original["sha256"] and
                        len(pcm) == original["frames"] * 4, "ACTIVE_ORIGINAL_PCM_CHANGED")
            else:
                require(len(raw) == derived["frames"] * 4, "ACTIVE_DERIVED_PCM_CHANGED")


def _join_source_files(archive_raw, inputs, *, d1_plan=None, d1_plan_sha256=None, d1_generation_root=None,
                       d1_outcomes=None, d1_outcomes_sha256=None, d1_stage=None, d1_job_recipe=None,
                       d1_artifact_roots=None, d1_checkpoint=None, d1_checkpoint_sha256=None):
    if d1_checkpoint is not None or d1_checkpoint_sha256 is not None:
        require(d1_checkpoint is not None and d1_checkpoint_sha256 is not None and
                all(value is None for value in (d1_plan, d1_plan_sha256, d1_generation_root,
                    d1_outcomes, d1_outcomes_sha256, d1_stage, d1_job_recipe)), "D1_V3_DISTINCT_CHECKPOINT_REQUIRED")
        return d1_source_rows_v3(archive_raw, inputs, read_bytes(d1_checkpoint, 32 * 1024 * 1024),
                                d1_checkpoint_sha256, artifact_roots=d1_artifact_roots)
    v2_args = (d1_outcomes, d1_outcomes_sha256, d1_stage, d1_job_recipe)
    if any(value is not None for value in v2_args):
        require(all(value is not None for value in (*v2_args, d1_plan, d1_plan_sha256)) and
                d1_generation_root is None, "D1_V2_PINS_AND_ARTIFACTS_REQUIRED")
        return d1_source_rows_v2(archive_raw, inputs, read_bytes(d1_plan), d1_plan_sha256,
            read_bytes(d1_outcomes, 16 * 1024 * 1024), d1_outcomes_sha256,
            read_bytes(d1_stage), read_bytes(d1_job_recipe), artifact_roots=d1_artifact_roots)
    require(d1_artifact_roots is None, "D1_ARTIFACT_ROOTS_REQUIRE_V2")
    v1_args = (d1_plan, d1_plan_sha256, d1_generation_root)
    require(all(value is None for value in v1_args) or all(value is not None for value in v1_args),
            "D1_PLAN_PIN_AND_GENERATION_REQUIRED")
    if d1_plan is not None:
        return d1_source_rows(archive_raw, inputs, read_bytes(d1_plan), d1_plan_sha256, d1_generation_root)
    sources, evidence = source_rows(archive_raw, inputs)
    return sources, evidence, None


def _source_map(inputs_raw, input_root, sources, plan_sha):
    inputs = loads(inputs_raw)
    rows, roles = [], SourceRoles()
    for clip in inputs["clips"]:
        source = sources[clip["id"]]
        raw = read_bytes(Path(input_root) / relative(clip["derived_pcm"]["path"]), 2 * 1024 * 1024)
        require(sha(raw) == clip["derived_pcm"]["sha256"], "ACTIVE_AUDIO_CHANGED")
        minimum = 1 if source.get("generation_outcome") == "failed-native" else 400
        _, measured = quantize_pcm16(raw, clip["derived_pcm"]["frames"], minimum_frames=minimum)
        add_source_role(roles, source, clip, measured)
        flags = source_quality_flags(source, measured)
        rows.append({"clip_id": clip["id"], "source_id": source["source_id"],
                     **{key + "_sha256": clip[key]["sha256"] for key in
                        ("original_wav", "original_pcm", "derived_pcm")},
                     "quality_uncertain": bool(flags), "quality_reasons": flags})
    return {"schema": "active-asr-source-map-v2", "input_manifest_sha256": sha(inputs_raw),
            "source_archive_sha256": inputs["source_archive_sha256"], "plan_sha256": plan_sha, "rows": rows}


def build_d1_source_map(inputs_path, expected_inputs_sha, archive_path, **source_options):
    """Create audio/source-only routing metadata before any ASR call or result."""
    inputs_path = Path(inputs_path)
    inputs_raw = read_bytes(inputs_path)
    require(sha(inputs_raw) == expected_inputs_sha, "ACTIVE_INPUT_MANIFEST_CHANGED")
    inputs = loads(inputs_raw)
    verify_inputs(inputs, inputs_path.parent)
    sources, _, adapter = _join_source_files(read_bytes(archive_path, 256 * 1024 * 1024), inputs, **source_options)
    require(adapter is not None, "D1_SOURCE_MAP_REQUIRES_FROZEN_PLAN")
    return _source_map(inputs_raw, inputs_path.parent, sources, adapter["plan_sha256"])


def create_view(inputs_path, expected_inputs_sha, archive_path, asr_root, output, *, fsmn_tokens=None,
                d1_plan=None, d1_plan_sha256=None, d1_generation_root=None, d1_outcomes=None,
                d1_outcomes_sha256=None, d1_stage=None, d1_job_recipe=None, d1_artifact_roots=None,
                selective_source_map=None, selective_policy=None, selective_audit_sha256=None,
                d1_checkpoint=None, d1_checkpoint_sha256=None):
    inputs_path, asr_root, output = Path(inputs_path), Path(asr_root), Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and ROOT not in output.parents and
            not any(path.is_symlink() for path in (output, *output.parents)), "NEW_PRIVATE_VIEW_REQUIRED")
    inputs_raw = read_bytes(inputs_path)
    require(sha(inputs_raw) == expected_inputs_sha, "ACTIVE_INPUT_MANIFEST_CHANGED")
    inputs = loads(inputs_raw)
    verify_inputs(inputs, inputs_path.parent)
    archive_raw = read_bytes(archive_path, 256 * 1024 * 1024)
    sources, evidence, source_adapter = _join_source_files(archive_raw, inputs, d1_plan=d1_plan,
        d1_plan_sha256=d1_plan_sha256, d1_generation_root=d1_generation_root, d1_outcomes=d1_outcomes,
        d1_outcomes_sha256=d1_outcomes_sha256, d1_stage=d1_stage, d1_job_recipe=d1_job_recipe,
        d1_artifact_roots=d1_artifact_roots, d1_checkpoint=d1_checkpoint, d1_checkpoint_sha256=d1_checkpoint_sha256)
    token_raw = read_bytes(fsmn_tokens, 65536) if fsmn_tokens is not None else None
    vocab = fsmn_vocabulary(token_raw) if token_raw is not None else None
    retained, selective_rows, asr_adapter = {}, None, None
    selective_args = (selective_source_map, selective_policy, selective_audit_sha256)
    require(all(value is None for value in selective_args) or all(value is not None for value in selective_args),
            "SELECTIVE_POLICY_MAP_AND_AUDIT_PIN_REQUIRED")
    if selective_source_map is not None:
        require(source_adapter is not None, "SELECTIVE_D1_SOURCE_ADAPTER_REQUIRED")
        selective_rows, asr_adapter = selective_evidence(asr_root, inputs_raw, inputs_path.parent, sources,
            source_adapter["plan_sha256"], read_bytes(selective_source_map), read_bytes(selective_policy),
            selective_audit_sha256, retained)
    output.mkdir(mode=0o700)
    producer_files = {}
    producer_names = ["automation/active_asr_weak_view.py", "kws_data_contract/__init__.py",
                      "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py",
                      "kws_data_contract/identity.py"]
    if source_adapter is not None:
        producer_names.append(BULK_POOL_PATH)
    if source_adapter is not None and source_adapter["schema"] == D1_ADAPTER_V3:
        producer_names = list(ACCOUNTED_SOURCE_PATHS)
    for name in producer_names:
        raw = (ROOT / name).read_bytes()
        producer_files[name] = sha(raw)
        save(output, "evidence/view-code/" + name, raw)
    save(output, "evidence/inputs.json", inputs_raw)
    save(output, "evidence/source-original.zip", archive_raw)
    for name, raw in evidence.items():
        save(output, "evidence/source/" + name, raw)
    if token_raw is not None:
        save(output, "evidence/fsmn-tokens.txt", token_raw)
    rows, roles = [], SourceRoles()
    for clip in inputs["clips"]:
        source = sources[clip["id"]]
        row, wav, original, derived = make_row(clip, source, inputs_path.parent, asr_root,
            expected_inputs_sha, vocab, retained,
            observations=selective_rows[clip["id"]] if selective_rows is not None else None)
        add_source_role(roles, source, clip, row["audio_derivation"])
        save(output, "evidence/inputs/" + clip["derived_pcm"]["path"], derived)
        save(output, "evidence/inputs/" + clip["original_wav"]["path"], original)
        save(output, row["audio"], wav)
        rows.append(row)
    for name, raw in retained.items():
        if (output / name).exists():
            require(read_bytes(output / name, 32 * 1024 * 1024) == raw, "RETAINED_EVIDENCE_CONFLICT")
        else:
            save(output, name, raw)
    report = {"schema": SCHEMA_V2 if asr_adapter is not None else SCHEMA,
              "policy": POLICY_V2 if asr_adapter is not None else POLICY, "inputs_sha256": expected_inputs_sha,
              "producer": {"python": sys.version.split()[0], "source_files_sha256": producer_files},
              "source_archive_sha256": inputs["source_archive_sha256"], "items": rows,
              **({"source_adapter": source_adapter} if source_adapter is not None else {}),
              **({"asr_adapter": asr_adapter} if asr_adapter is not None else {}),
              "split_scope": ("source-group-train-and-development-machine-weak-no-independent-test"
                              if source_adapter is not None else "original-train-only-no-independent-development"),
              "new_model_calls": 0, "training_runs": 0, "human_gold": False,
              "statistics": {"source_items": len(rows), "source_families": len({r['source']['source_family_id'] for r in rows}),
                             "completed_asr_results": sum(o["status"] == "complete" for r in rows for o in r["asr"]),
                             "machine_candidates": sum(r["decision"]["status"] == "machine_candidate" for r in rows),
                             "uncertain": sum(r["decision"]["status"] == "uncertain" for r in rows),
                             "disagreement": sum(r["decision"]["disagreement"] for r in rows),
                             "train_only": sum(r["source"]["split"] == "train" for r in rows),
                             "development": sum(r["source"]["split"] == "development" for r in rows),
                             **({"not_requested_asr_results": sum(o["status"] == "not_requested" for r in rows for o in r["asr"])}
                                if asr_adapter is not None else {})}}
    save_json(output, "report.json", report)
    save_rows(output, "weak-fulltext.jsonl", [row for row in rows if row["decision"]["status"] == "machine_candidate"])
    save_rows(output, "uncertain.jsonl", [row for row in rows if row["decision"]["status"] != "machine_candidate"])
    save_rows(output, "disagreement.jsonl", [row for row in rows if row["decision"]["disagreement"]])
    for consumer in ("rnn-four-token", "fsmn-fulltext"):
        for split in (("train", "development") if source_adapter is not None else ("train",)):
            save_rows(output, consumer + "-" + split + ".jsonl", [training_row(row, consumer) for row in rows
                      if row["source"]["split"] == split and row["decision"]["ctc_mask"] and
                      row["consumers"][consumer]["supported"]])
    files = {path.relative_to(output).as_posix(): sha(path.read_bytes())
             for path in sorted(output.rglob("*")) if path.is_file()}
    save_json(output, "MANIFEST.json", files)
    return report


def training_row(row, consumer):
    source, decision, measured = row["source"], row["decision"], row["audio_derivation"]
    return {"source_id": source["source_id"], "item_id": row["item_id"], "audio": row["audio"],
            "target_ids": row["consumers"][consumer]["target_ids"], "full_text": decision["full_text"],
            "selected_raw_text": decision["selected_raw_text"], "split": source["split"],
            "intended_kind": source["intended_kind"],
            **({key: source[key] for key in ("generation_category", "source_plan_sha256", "reference_id")}
               if "generation_category" in source else {}),
            **({key: source[key] for key in ("generation_outcome", "technical_failure_reason", "outcome_record_sha256")}
               if "generation_outcome" in source else {}),
            "source_family_id": source["source_family_id"], "speaker_id": source["speaker_id"],
            "original_wav_sha256": row["input"]["original_wav"]["sha256"],
            "original_pcm_sha256": row["input"]["original_pcm"]["sha256"],
            "file_sha256": measured["file_sha256"], "pcm_sha256": measured["pcm_sha256"],
            "samples": measured["frames"], "ctc_mask": decision["ctc_mask"], "ctc_weight": decision["ctc_weight"],
            "keyword_targets": decision["keyword_targets"], "keyword_mask": decision["keyword_mask"],
            "keyword_weight": decision["keyword_weight"], "label_tier": decision["label_tier"],
            "label_source": "weak-machine-agreement", "purpose": "technical-research-only",
            "human_gold": False, "independent_test": False, "consumer": consumer}


def load_training_rows(package, consumer="fsmn-fulltext", *, other_split_sources=()):
    """Verified full-text numeric targets, PCM paths and loss weights for a trainer.

    Returns the same full-text/PCM information as the maintained FSMN candidate
    adapter, plus explicit CTC and keyword weights/masks. No optimizer is run.
    Legacy Manifest alone ignores these weights; callers must apply them to loss.
    """
    yield from _load_rows(package, consumer, "train", other_split_sources)


def load_development_rows(package, consumer="fsmn-fulltext", *, other_split_sources=()):
    """Explicitly separate source-group development rows; still machine weak."""
    yield from _load_rows(package, consumer, "development", other_split_sources)


def _load_rows(package, consumer, split, other_split_sources):
    require(consumer in ("rnn-four-token", "fsmn-fulltext"), "UNKNOWN_WEAK_CONSUMER")
    package = Path(package).absolute()
    manifest = loads(read_bytes(package / "MANIFEST.json"))
    actual_files = {path.relative_to(package).as_posix() for path in package.rglob("*")
                    if path.is_file() and path != package / "MANIFEST.json"}
    require(actual_files == set(manifest), "WEAK_PACKAGE_ROSTER_CHANGED")
    for name, expected in manifest.items():
        path = package / relative(name)
        if expected == sha(b""):
            require(not any(p.is_symlink() for p in (path, *path.parents)), "WEAK_PACKAGE_ALIAS")
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 0,
                        "WEAK_EMPTY_FILE_CHANGED")
                raw = stream.read(1)
        else:
            raw = read_bytes(path, 256 * 1024 * 1024)
        require(sha(raw) == expected, "WEAK_PACKAGE_CHANGED")
    report = loads(read_bytes(package / "report.json", 32 * 1024 * 1024))
    require((report["schema"], report["policy"]) in ((SCHEMA, POLICY), (SCHEMA_V2, POLICY_V2)) and
            report["human_gold"] is False and
            ((report["schema"] == SCHEMA_V2) == ("asr_adapter" in report)),
            "WEAK_VIEW_SCHEMA")
    inputs_raw = read_bytes(package / "evidence/inputs.json")
    require(sha(inputs_raw) == report["inputs_sha256"], "WEAK_INPUTS_CHANGED")
    inputs = loads(inputs_raw)
    verify_inputs(inputs, package / "evidence/inputs")
    archive_raw = read_bytes(package / "evidence/source-original.zip", 256 * 1024 * 1024)
    source_adapter = report.get("source_adapter")
    if source_adapter is not None and source_adapter["schema"] == D1_ADAPTER_V3:
        evidence = package / "evidence/source"
        sources, _, rebuilt_adapter = d1_source_rows_v3(archive_raw, inputs,
            read_bytes(evidence / "d1/accounted/checkpoint.json", 32 * 1024 * 1024),
            source_adapter["checkpoint_sha256"], retained_root=evidence)
        require(source_adapter == rebuilt_adapter and
                sha(read_bytes(evidence / "d1/plan.json")) == source_adapter["plan_file_sha256"],
                "D1_SOURCE_ADAPTER_CHANGED")
    elif source_adapter is not None and source_adapter["schema"] == D1_ADAPTER_V2:
        evidence = package / "evidence/source"
        sources, _, rebuilt_adapter = d1_source_rows_v2(
            archive_raw, inputs, read_bytes(evidence / "d1/plan.json"), source_adapter["plan_sha256"],
            read_bytes(evidence / "d1/outcomes.json", 16 * 1024 * 1024), source_adapter["outcomes_sha256"],
            read_bytes(evidence / "d1/stage.json"), read_bytes(evidence / "d1/job-recipe.json"),
            retained_root=evidence)
        require(source_adapter == rebuilt_adapter, "D1_SOURCE_ADAPTER_CHANGED")
    elif source_adapter is not None:
        require(source_adapter["schema"] == D1_ADAPTER, "D1_SOURCE_ADAPTER_SCHEMA")
        sources, _, rebuilt_adapter = d1_source_rows(
            archive_raw, inputs, read_bytes(package / "evidence/source/d1/plan.json"),
            source_adapter["plan_sha256"], package / "evidence/source/d1")
        require(source_adapter == rebuilt_adapter, "D1_SOURCE_ADAPTER_CHANGED")
    else:
        sources, _ = source_rows(archive_raw, inputs)
    selective_rows = None
    if report["schema"] == SCHEMA_V2:
        require(source_adapter is not None and report["asr_adapter"]["schema"] == SELECTIVE_ADAPTER,
                "SELECTIVE_D1_SOURCE_ADAPTER_REQUIRED")
        selective_rows, rebuilt_asr_adapter = selective_evidence(package / "evidence/asr", inputs_raw,
            package / "evidence/inputs", sources, source_adapter["plan_sha256"],
            read_bytes(package / "evidence/selective/source-map.json"),
            read_bytes(package / "evidence/selective/policy.json"),
            report["asr_adapter"]["final_audit_sha256"], {})
        require(report["asr_adapter"] == rebuilt_asr_adapter, "SELECTIVE_ASR_ADAPTER_CHANGED")
    vocab_path = package / "evidence/fsmn-tokens.txt"
    vocab = fsmn_vocabulary(read_bytes(vocab_path, 65536)) if vocab_path.exists() else None
    require(vocab is not None or consumer != "fsmn-fulltext", "FSMN_VOCABULARY_NOT_SUPPLIED")
    require([row["item_id"] for row in report["items"]] == [clip["id"] for clip in inputs["clips"]],
            "WEAK_INPUT_ROSTER_CHANGED")
    roles = SourceRoles()
    for source in other_split_sources:
        roles.add(source["role"], family=source.get("source_family_id"), speaker=source.get("speaker_id"),
                  pcm_sha256=source.get("pcm_sha256"), wav_sha256=source.get("wav_sha256"))
    by_split = {"train": [], "development": []}
    for row, clip in zip(report["items"], inputs["clips"]):
        rebuilt, wav, _, _ = make_row(clip, sources[clip["id"]], package / "evidence/inputs",
            package / "evidence/asr", report["inputs_sha256"], vocab, {},
            observations=selective_rows[clip["id"]] if selective_rows is not None else None)
        require(row == rebuilt, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE")
        require(read_bytes(package / relative(row["audio"]), 2 * 1024 * 1024) == wav,
                "WEAK_QUANTIZED_AUDIO_CHANGED")
        source, measured = row["source"], row["audio_derivation"]
        add_source_role(roles, source, clip, measured)
        expected = row["consumers"][consumer]
        if row["decision"]["ctc_mask"] and expected["supported"]:
            require((source["split"], source["role"]) in (("train", "train-candidate"),
                                                        ("development", "development-only")) and
                    row["human_gold"] is False and row["independent_test"] is False,
                    "WEAK_SOURCE_SPLIT_SCOPE")
            by_split[source["split"]].append(training_row(row, consumer))
    for subset in (("train", "development") if source_adapter is not None else ("train",)):
        encoded = b"".join(canonical(row) + b"\n" for row in by_split[subset])
        require((package / (consumer + "-" + subset + ".jsonl")).read_bytes() == encoded,
                "WEAK_TRAIN_ROWS_CHANGED")
    for row in by_split[split]:
        row = dict(row)
        row["audio"] = row["audio_path"] = str(package / row["audio"])
        if consumer == "fsmn-fulltext":
            # Match the maintained full-text adapter's trainer-facing names.
            row.update(recording=row["item_id"], ctc_ids=list(row["target_ids"]),
                       ctc_text=row["full_text"], audio_sha256=row["file_sha256"])
        yield row


def _d1_local_pin(path, expected_sha=None, *, limit=32 * 1024 * 1024):
    path = Path(path).absolute()
    body = empty_evidence(path) if path.stat().st_size == 0 else read_bytes(path, limit)
    require(expected_sha is None or sha(body) == expected_sha, "D1_ACCOUNTED_EXTERNAL_PIN_CHANGED")
    return {"path": str(path), "bytes": len(body), "sha256": sha(body)}


def _d1_replay_view(entry, reader):
    """One complete source/audio/ASR replay, then verify all four saved projections."""
    require(set(entry) == {"path", "manifest_sha256"}, "D1_ACCOUNTED_VIEW_PIN")
    package = Path(entry["path"]).absolute()
    manifest_pin = _d1_local_pin(package / "MANIFEST.json", entry["manifest_sha256"])
    manifest = reader.json(manifest_pin)
    # This iterator validates every report row, not only the admitted subset.
    list(load_training_rows(package, "rnn-four-token"))
    report = loads(read_bytes(package / "report.json", 32 * 1024 * 1024))
    counts = {}
    for consumer in ("rnn-four-token", "fsmn-fulltext"):
        counts[consumer] = {}
        for split in ("train", "development"):
            rows = [training_row(row, consumer) for row in report["items"]
                    if row["source"]["split"] == split and row["decision"]["ctc_mask"] and
                    row["consumers"][consumer]["supported"]]
            encoded = b"".join(canonical(row) + b"\n" for row in rows)
            require((package / (consumer + "-" + split + ".jsonl")).read_bytes() == encoded,
                    "D1_ACCOUNTED_FOUR_CONSUMER_REPLAY")
            counts[consumer][split] = len(rows)
    for name, expected in manifest.items():
        reader.pin(_d1_local_pin(package / relative(name), expected, limit=256 * 1024 * 1024),
                   limit=256 * 1024 * 1024)
    require(reader.pin(manifest_pin) == canonical(manifest) + b"\n", "D1_ACCOUNTED_VIEW_CHANGED_DURING_REPLAY")
    return report, counts


def _verify_accounted_prior(path, expected_sha256, *, expected_count):
    require(type(expected_count) is int and expected_count in (232, 256), "D1_ACCOUNTED_EXPECTED_COUNT")
    reader = _D1CheckpointEvidence()
    pin = _d1_local_pin(path, expected_sha256)
    value = reader.json(pin)
    require(set(value) == {"schema", "plan_sha256", "plan_file_sha256", "accounted_prefix_count",
        "observed_source_count", "views", "parent_evidence", "continuation_checkpoint",
        "unobservable_consumed", "excluded_returned"} and value["schema"] == "d1-accounted-prior-evidence-v2",
        "D1_ACCOUNTED_PRIOR_SCHEMA")
    checkpoint, plan, refs, sources, parent = _d1_accounted_checkpoint(
        reader.pin(value["continuation_checkpoint"]), value["continuation_checkpoint"]["sha256"], reader)
    prefix = checkpoint["prefix_count"]
    require(prefix <= expected_count and value["accounted_prefix_count"] == prefix and
            value["plan_sha256"] == checkpoint["plan_canonical_sha256"] and
            value["plan_file_sha256"] == checkpoint["plan"]["sha256"] and
            value["parent_evidence"] == checkpoint["prior_evidence"] and
            value["unobservable_consumed"] == checkpoint["unobservable_consumed"] and
            value["excluded_returned"] == checkpoint["excluded_native_rows"] and
            value["views"][:len(parent["views"])] == parent["views"], "D1_ACCOUNTED_PRIOR_CHECKPOINT_JOIN")
    ids = [row["cell_id"] for row in plan["cells"]]
    cells = {row["cell_id"]: row for row in plan["cells"]}
    expected_observed = set(ids[:176]) | set(sources)
    require(len(value["views"]) == len({row["path"] for row in value["views"]}) and
            value["observed_source_count"] == len(expected_observed), "D1_ACCOUNTED_OBSERVED_COUNT")
    reports, observed, roles = {}, [], SourceRoles()
    matrices = set()
    for entry in value["views"]:
        report, counts = _d1_replay_view(entry, reader)
        require(report["source_adapter"]["plan_sha256"] == value["plan_sha256"] and
                report["source_adapter"]["plan_file_sha256"] == value["plan_file_sha256"],
                "D1_ACCOUNTED_VIEW_PLAN")
        if entry not in parent["views"]:
            adapter = report["source_adapter"]
            allowed = {value["continuation_checkpoint"]["sha256"]}
            if checkpoint["previous_checkpoint"] is not None:
                allowed.add(checkpoint["previous_checkpoint"]["sha256"])
            require(adapter["schema"] == D1_ADAPTER_V3 and adapter["checkpoint_sha256"] in allowed and
                    set(report["producer"]["source_files_sha256"]) == set(ACCOUNTED_SOURCE_PATHS),
                    "D1_ACCOUNTED_NEW_VIEW_SOURCE")
        for row in report["items"]:
            source = row["source"]
            name = source["source_id"]
            require(name in expected_observed and name not in observed, "D1_ACCOUNTED_OBSERVED_EXACT_UNIQUE")
            cell = cells[name]
            ref = refs[cell["reference_id"]]
            require(source["source_family_id"] == cell["source_family_id"] == ref["source_family_id"] and
                    source["speaker_id"] == cell["speaker_group_id"] == ref["speaker_group_id"] and
                    source["role"] == cell["role"] == ref["role"] and
                    source["split"] == ("train" if cell["role"] == "train-candidate" else "development"),
                    "D1_ACCOUNTED_SOURCE_ROLE")
            add_source_role(roles, source, row["input"], row["audio_derivation"])
            for observation_row in row["asr"]:
                if observation_row["status"] != "not_requested":
                    pair = (name, observation_row["model"])
                    require(pair not in matrices, "D1_ACCOUNTED_DUPLICATE_RECOGNITION")
                    matrices.add(pair)
            observed.append(name)
        reports[entry["path"]] = (report, counts)
    require(set(observed) == expected_observed, "D1_ACCOUNTED_OBSERVED_UNION")

    def unobserved(row, *, consumed):
        name = row["cell_id"]
        cell = cells[name]
        proof = row
        if not consumed:
            returned = next(item for item in checkpoint["returned_rows"] if item["cell_id"] == name)
            proof = {"exclusion": row, "returned_row": returned,
                     "owner": checkpoint["owners"][returned["owner_id"]],
                     "continuation_checkpoint": value["continuation_checkpoint"]}
        return {"source_id": name, "source_family_id": cell["source_family_id"], "source_role": cell["role"],
            "status": "unobservable-consumed" if consumed else "returned-unrecognizable",
            "reason": "consumed_call_without_native_return" if consumed else row["reason"],
            "asr_calls": 0, "observed_text": None, "training_admitted": False, "original_proof": proof}

    excluded = [unobserved(row, consumed=False) for row in value["excluded_returned"]]
    unknown = [unobserved(row, consumed=True) for row in value["unobservable_consumed"]]
    unavailable = {row["source_id"] for row in excluded + unknown}
    require(not unavailable & expected_observed and
            len(unavailable) == len(excluded) + len(unknown) and
            expected_observed | unavailable == set(ids[:prefix]), "D1_ACCOUNTED_DISJOINT_UNION")
    for name in ACCOUNTED_SOURCE_PATHS:
        reader.pin(_d1_local_pin(ROOT / name))
    result = {"schema": "kws-d1-accounted-prior-verification-v2", "plan": plan,
        "plan_sha256": value["plan_sha256"], "accounted_prefix_count": prefix,
        "observed_ids": [name for name in ids if name in expected_observed],
        "excluded_returned": excluded, "unobservable_consumed": unknown,
        "pending_ids": ids[prefix:expected_count], "views": value["views"],
        "evidence_files": sorted(reader.facts.values(), key=lambda row: row["path"]),
        "counts": {"observed": len(observed), "excluded_returned": len(excluded),
                   "unobservable_consumed": len(unknown), "pending": expected_count - prefix},
        "accounted_complete": prefix == expected_count,
        "complete_terminal_coverage": prefix == expected_count and not unknown,
        "ml_imports": 0, "new_model_calls": 0}
    return result, reports, reader, value


def verify_accounted_prior(path, expected_sha256, *, expected_count=256):
    """Replay typed generation accounting and every immutable weak observation."""
    return _verify_accounted_prior(path, expected_sha256, expected_count=expected_count)[0]


def _d1_asr_sealed_queue(queue_root, entries, reports, checkpoint_pin, reader):
    root = Path(queue_root).absolute()
    queue_pins = {name: _d1_local_pin(root / name) for name in ("queue.json", "request.json", "active-plan.json")}
    queue, request = reader.json(queue_pins["queue.json"]), reader.json(queue_pins["request.json"])
    require(request["schema"] == "kws-local-research-queue-v1" and request["global_budget_minutes"] == 20 and
            len(request["jobs"]) == len(entries) and 1 <= len(entries) <= 2, "D1_ACCOUNTED_ASR_QUEUE_SCOPE")
    states, total, seen_ids = {}, 0, set()
    for index, (job, entry) in enumerate(zip(request["jobs"], entries)):
        name = job["job_id"]
        folder = root / "jobs" / name
        report, consumers = reports[entry["path"]]
        require(entry["job_id"] == name and entry["consumers"] == consumers and
                entry["statistics"] == report["statistics"] and job["timeout_minutes"] == 10 and
                job["depends_on"] == ([] if index == 0 else [request["jobs"][index - 1]["job_id"]]),
                "D1_ACCOUNTED_ASR_JOB_VIEW_ORDER")
        recipe_pin = _d1_local_pin(root / "recipes" / (name + ".json"), job["recipe_sha256"])
        recipe = reader.json(recipe_pin)
        require(recipe["schema"] == "active-asr-selective-request-v2" and
                recipe["output"] == str(folder / "outputs/recognition") and recipe["budget_seconds"] == 570 and
                recipe["policy"]["sha256"] == SELECTIVE_POLICY_SHA and
                recipe["max_calls"] == 3 * len(report["items"]) and
                job["argv"] == [str(PurePosixPath(recipe["runtime"]) / "venv/bin/python"), "-I", "-B",
                    str(PurePosixPath(job["cwd"]) / "tools/active_asr_selective.py"), "run", "--request", "{recipe}"],
                "D1_ACCOUNTED_ASR_REQUEST_SCOPE")
        source_files = {}
        for source_name, expected in {**recipe["source_files"], **{
            "tools/" + key: value for key, value in queue["source_sha256"].items()}}.items():
            current = _d1_local_pin(Path(job["cwd"]) / relative(source_name), expected)
            source_files[source_name] = {key: current[key] for key in ("bytes", "sha256")}
        job_files = {p.relative_to(folder).as_posix(): {key: current[key] for key in ("bytes", "sha256")}
                     for p in sorted(folder.rglob("*")) if p.is_file()
                     for current in [_d1_local_pin(p)]}
        owner = {"origin": "asr-completed", "job_id": name, "stage": checkpoint_pin, "job_recipe": recipe_pin,
            "queue_root": str(root), "queue_files": queue_pins, "job_root": str(folder), "job_files": job_files,
            "result": _d1_local_pin(folder / "result.json"), "outputs_root": str(folder / "outputs"),
            "source_root": job["cwd"], "source_files": source_files}
        actual = _d1_accounted_owner(owner, reader)
        require(actual["claim"]["global_actual_before_nanoseconds"] == total and
                actual["claim"]["allowed_runner_nanoseconds"] == min(600_000_000_000, 1_200_000_000_000 - total) and
                actual["terminal"]["deadline_overrun_nanoseconds"] == 0, "D1_ACCOUNTED_ASR_ACTUAL_BUDGET")
        dependencies = {dep: {"receipt": str(root / "jobs" / dep / "result.json"),
            "receipt_sha256": states[dep]["receipt_sha256"], "outputs_root": str(root / "jobs" / dep / "outputs"),
            "outputs": states[dep]["outputs"]} for dep in job["depends_on"]}
        require(actual["dependencies"] == dependencies, "D1_ACCOUNTED_ASR_DEPENDENCIES")
        package = Path(entry["path"])
        manifest = loads(read_bytes(package / "MANIFEST.json"))
        actual_asr = folder / "outputs/recognition"
        actual_inputs = Path(recipe["inputs"]["path"]).parent
        require({p.relative_to(actual_asr).as_posix() for p in actual_asr.rglob("*") if p.is_file()} ==
                {name.removeprefix("evidence/asr/") for name in manifest if name.startswith("evidence/asr/")} and
                {p.relative_to(actual_inputs).as_posix() for p in actual_inputs.rglob("*") if p.is_file()} ==
                {name.removeprefix("evidence/inputs/") for name in manifest if name.startswith("evidence/inputs/")},
                "D1_ACCOUNTED_ASR_EXTRA_OR_MISSING_EVIDENCE")
        # Each raw request, call, native result, model load, routing decision and
        # input waveform must be the same bytes that the full view replay used.
        for saved, expected in manifest.items():
            if saved.startswith("evidence/asr/"):
                source_path = folder / "outputs/recognition" / saved.removeprefix("evidence/asr/")
            elif saved.startswith("evidence/inputs/"):
                source_path = Path(recipe["inputs"]["path"]).parent / saved.removeprefix("evidence/inputs/")
            elif saved == "evidence/selective/source-map.json":
                source_path = Path(recipe["source_map"]["path"])
            elif saved == "evidence/selective/policy.json":
                source_path = Path(recipe["policy"]["path"])
            else:
                continue
            reader.pin(_d1_local_pin(source_path, expected))
        require(report["asr_adapter"]["request_sha256"] == recipe_pin["sha256"] and
                report["asr_adapter"]["source_map_sha256"] == recipe["source_map"]["sha256"] and
                report["inputs_sha256"] == recipe["inputs"]["sha256"] and
                report["asr_adapter"]["final_audit_sha256"] ==
                    actual["terminal"]["outputs"]["recognition/audit/final-summary.json"]["sha256"],
                "D1_ACCOUNTED_ASR_TERMINAL_VIEW_JOIN")
        current_ids = {row["source"]["source_id"] for row in report["items"]}
        require(not current_ids & seen_ids, "D1_ACCOUNTED_ASR_DUPLICATE_SOURCES")
        seen_ids |= current_ids
        total += actual["terminal"]["actual_elapsed_ns"]
        states[name] = {**actual["terminal"], "receipt_sha256": owner["result"]["sha256"]}
    require(total <= 1_200_000_000_000, "D1_ACCOUNTED_ASR_GLOBAL_BUDGET")
    return {"schema": "kws-local-research-queue-status-v1", "queue_id": request["queue_id"], "status": "completed",
        "global_budget_minutes": 20, "actual_known_job_minutes": total / 60_000_000_000,
        "actual_known_job_nanoseconds": total, "actual_total_is_complete": True, "unresolved_jobs": [],
        "remaining_budget_nanoseconds": 1_200_000_000_000 - total, "budget_overrun_nanoseconds": 0,
        "budget_scope": "all_jobs_in_this_frozen_queue", "jobs": states, "automatic_retries": 0,
        "historical_jobs_imported": 0, "heavy_lock": queue["heavy_lock"], "remote_watchdog_deployed": False}


def verify_accounted_seal(path, expected_sha256, *, expected_count=232):
    """Verify the actual completed ASR queue→native/source-map→retained-view handoff."""
    require(expected_count in (232, 256), "D1_ACCOUNTED_SEAL_EXPECTED_PREFIX")
    pin = _d1_local_pin(path, expected_sha256)
    sealed = loads(read_bytes(pin["path"]))
    require(set(sealed) == {"schema", "accounted_prefix_count", "observed_prefix_count", "new_observed_audio",
        "continuation_checkpoint", "accounted_prior_evidence", "queue_root", "queue_state", "views",
        "full_replay_completed", "new_model_calls_in_sealing", "unknown_asr_requests",
        "accounted_complete", "complete_terminal_coverage"} and
        sealed["schema"] == "d1-accounted-selective-stage-closed-v3" and
        sealed["accounted_prefix_count"] == expected_count and sealed["full_replay_completed"] is True and
        sealed["new_model_calls_in_sealing"] == sealed["unknown_asr_requests"] == 0 and
        sealed["accounted_complete"] is True and sealed["complete_terminal_coverage"] is False,
        "D1_ACCOUNTED_SEAL_SCHEMA")
    prior_pin = sealed["accounted_prior_evidence"]
    result, reports, reader, prior = _verify_accounted_prior(
        prior_pin["path"], prior_pin["sha256"], expected_count=expected_count)
    reader.pin(prior_pin)
    reader.pin(pin)
    require(result["accounted_complete"] and not result["pending_ids"] and
            prior["continuation_checkpoint"] == sealed["continuation_checkpoint"] and
            sealed["observed_prefix_count"] == len(result["observed_ids"]) and
            sealed["queue_root"] == str(Path(path).absolute().parent / "queue"), "D1_ACCOUNTED_SEAL_PRIOR_JOIN")
    entries = sealed["views"]
    require(entries and prior["views"][-len(entries):] == [
        {"path": entry["path"], "manifest_sha256": entry["manifest_sha256"]} for entry in entries],
        "D1_ACCOUNTED_SEAL_VIEW_ROSTER")
    checkpoint = reader.json(sealed["continuation_checkpoint"])
    new_ids = set(checkpoint["eligible_cell_ids"])
    if checkpoint["previous_checkpoint"] is not None:
        earlier = reader.json(checkpoint["previous_checkpoint"])
        new_ids -= {row["cell_id"] for row in earlier["returned_rows"]}
    source_snapshot = Path(path).absolute().parent / "source-snapshot/data"
    source_hashes = {}
    for source in ACCOUNTED_SOURCE_PATHS:
        record = _d1_local_pin(source_snapshot / source)
        require(record["sha256"] == sha(read_bytes(ROOT / source)), "D1_ACCOUNTED_HANDOFF_VERIFIER_SOURCE_CHANGED")
        reader.pin(record)
        source_hashes[source] = record["sha256"]
    observed_new = []
    for entry in entries:
        report = reports[entry["path"]][0]
        require(report["source_adapter"]["schema"] == D1_ADAPTER_V3 and
                report["source_adapter"]["checkpoint_sha256"] == sealed["continuation_checkpoint"]["sha256"] and
                report["producer"]["source_files_sha256"] == source_hashes,
                "D1_ACCOUNTED_HANDOFF_PRODUCER_SOURCE")
        observed_new.extend(row["source"]["source_id"] for row in report["items"])
    require(len(observed_new) == len(set(observed_new)) == sealed["new_observed_audio"] and
            set(observed_new) == new_ids, "D1_ACCOUNTED_SEAL_EXACT_NEW_OBSERVATIONS")
    queue = _d1_asr_sealed_queue(sealed["queue_root"], entries, reports, sealed["continuation_checkpoint"], reader)
    require(sealed["queue_state"] == queue, "D1_ACCOUNTED_SEAL_ACTUAL_QUEUE_SNAPSHOT")
    result["evidence_files"] = sorted(reader.facts.values(), key=lambda row: row["path"])
    return {"schema": "d1-accounted-selective-seal-verification-v3", "status": "verified", "verified": True,
        "accounted_prefix_count": expected_count, "prior": result, "queue_state": queue,
        "new_observed_ids": observed_new, "evidence_files": result["evidence_files"],
        "full_replay_completed": True, "ml_imports": 0, "new_model_calls": 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for name in ("inputs", "source-archive", "asr-results", "output"):
        create.add_argument("--" + name, required=True, type=Path)
    create.add_argument("--inputs-sha256", required=True)
    create.add_argument("--fsmn-tokens", type=Path)
    create.add_argument("--selective-source-map", type=Path)
    create.add_argument("--selective-policy", type=Path)
    create.add_argument("--selective-audit-sha256")
    mapping = sub.add_parser("source-map", help="verify D1 sources and freeze text-free sparse-ASR routing metadata")
    for name in ("inputs", "source-archive", "output"):
        mapping.add_argument("--" + name, required=True, type=Path)
    mapping.add_argument("--inputs-sha256", required=True)
    for parser in (create, mapping):
        parser.add_argument("--d1-plan", type=Path)
        parser.add_argument("--d1-plan-sha256")
        parser.add_argument("--d1-generation-root", type=Path)
        parser.add_argument("--d1-outcomes", type=Path)
        parser.add_argument("--d1-outcomes-sha256")
        parser.add_argument("--d1-stage", type=Path)
        parser.add_argument("--d1-job-recipe", type=Path)
        parser.add_argument("--d1-artifact-roots-json", type=Path)
        parser.add_argument("--d1-checkpoint", type=Path)
        parser.add_argument("--d1-checkpoint-sha256")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("package", type=Path)
    inspect.add_argument("--consumer", choices=("rnn-four-token", "fsmn-fulltext"), default="fsmn-fulltext")
    inspect.add_argument("--split", choices=("train", "development"), default="train")
    args = cli.parse_args(argv)
    if args.command in ("create", "source-map"):
        source_options = {name: getattr(args, name) for name in (
            "d1_plan", "d1_plan_sha256", "d1_generation_root", "d1_outcomes", "d1_outcomes_sha256",
            "d1_stage", "d1_job_recipe", "d1_checkpoint", "d1_checkpoint_sha256")}
        source_options["d1_artifact_roots"] = (loads(read_bytes(args.d1_artifact_roots_json))
                                               if args.d1_artifact_roots_json else None)
    if args.command == "create":
        report = create_view(args.inputs, args.inputs_sha256, args.source_archive, args.asr_results,
            args.output, fsmn_tokens=args.fsmn_tokens, **source_options,
            selective_source_map=args.selective_source_map, selective_policy=args.selective_policy,
            selective_audit_sha256=args.selective_audit_sha256)
        print(json.dumps(report["statistics"], sort_keys=True))
    elif args.command == "source-map":
        result = build_d1_source_map(args.inputs, args.inputs_sha256, args.source_archive, **source_options)
        require(args.output.parent.is_dir() and ROOT not in args.output.absolute().parents,
                "PRIVATE_SOURCE_MAP_REQUIRED")
        save_json(args.output.parent, args.output.name, result)
        print(json.dumps({"schema": result["schema"], "rows": len(result["rows"]),
                          "sha256": sha(canonical(result) + b"\n"), "new_model_calls": 0}, sort_keys=True))
    else:
        loader = load_training_rows if args.split == "train" else load_development_rows
        rows = list(loader(args.package, args.consumer))
        print(json.dumps({"consumer": args.consumer, "train_examples": len(rows) if args.split == "train" else 0,
                          "source_families": len({row['source_family_id'] for row in rows}),
                          "development_examples": len(rows) if args.split == "development" else 0, "human_gold": False,
                          "new_model_calls": 0, "training_runs": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
