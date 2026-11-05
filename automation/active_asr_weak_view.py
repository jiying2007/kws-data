#!/usr/bin/env python3
"""Private research views of saved active ASR outputs; no model execution.

Keeps full transcripts and original receipts. A separate, explicit machine-weak
decision never changes historical admission fields or creates human gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
POLICY = "full-transcript-two-of-three-no-intended-positive-to-negative-v1"
MODELS = ("qwen06", "sensevoice", "whisper")
KEYWORDS = ("你好小窝", "小窝小窝")
FOUR_CHAR_IDS = {"你": 1, "好": 2, "小": 3, "窝": 4}
FOUR_TOKEN_NAMES = {1: "ni3", 2: "hao3", 3: "xiao3", 4: "wo1"}
FSMN_TOKENS_SHA = "5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa"
SHA = re.compile(r"[0-9a-f]{64}\Z")
D1_ADAPTER = "kws-active-d1-frozen-plan-source-v1"
BULK_POOL_PATH = "research/auto-tts-voice-clone-20261102/bulk_pool.py"


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


def quantize_pcm16(raw, frames):
    """Explicit nearest-even 32768-scale saturation; no gain or time edits."""
    require(type(frames) is int and 400 <= frames <= 16000 * 30 and
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


def d1_source_rows(archive_raw, inputs, plan_raw, expected_plan_sha, generation_root):
    """Bind native-only D1 archive members to the unchanged bulk runner receipts.

    The external plan pin is the canonical SHA used by bulk_runner's plan lock.
    Source lineage and generation category come only from that frozen plan.
    This verifies retained evidence; it does not authenticate source permissions.
    """
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
    cells = {row["cell_id"]: row for row in plan["cells"]}
    for ref in refs.values():
        for key in ("reference_id", "source_family_id", "speaker_group_id"):
            pool._id(ref[key])
        for key in ("wav_sha256", "pcm_sha256", "source_terms_sha256", "rights_record_sha256"):
            pool._sha(ref[key])
        if ref["parent_source_sha256"] is not None:
            pool._sha(ref["parent_source_sha256"])
    root = Path(generation_root)
    evidence = {"d1/plan.json": plan_raw}

    def retained(name, limit=1024 * 1024):
        raw = read_bytes(root / relative(name), limit)
        evidence["d1/" + name] = raw
        return raw

    lock = loads(retained("plan-lock.json"))
    require(lock == {"schema": "kws-bulk-clone-execution-lock-v1",
                     "plan_sha256": expected_plan_sha, "training_admitted": False,
                     "model_id": pool.MODEL}, "D1_EXECUTION_LOCK_CHANGED")
    require(sha(archive_raw) == inputs["source_archive_sha256"], "SOURCE_ARCHIVE_CHANGED")
    sources, seen_cells, seen_pcm = {}, set(), set()
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
            seen_cells.add(cell["cell_id"])
            ref = refs[cell["reference_id"]]
            folder = "cells/" + cell["cell_id"] + "/"
            require(not (root / folder / "failed.json").exists(), "D1_SOURCE_FAILED_CELL")
            started_raw = retained(folder + "started.json")
            started = loads(started_raw)
            require(started == {"schema": "kws-bulk-clone-start-v1", "cell_id": cell["cell_id"],
                                "reference_id": ref["reference_id"], "reference_wav_sha256": ref["wav_sha256"],
                                "plan_sha256": expected_plan_sha, "source_family_id": cell["source_family_id"],
                                "seed": cell["seed"], "attempts": 1, "training_admitted": False},
                    "D1_STARTED_PLAN_JOIN")
            completed_raw = retained(folder + "completed.json")
            completed = loads(completed_raw)
            require(set(completed) == {"schema", "cell_id", "status", "technical", "audio", "source_family_id",
                                       "intended_kind", "actual_text", "training_admitted", "qualification_allowed"} and
                    completed["schema"] == "kws-bulk-clone-complete-v1" and
                    completed["cell_id"] == cell["cell_id"] and completed["status"] == "GENERATED_PENDING_ASR" and
                    completed["source_family_id"] == cell["source_family_id"] and
                    completed["intended_kind"] == cell["intended_kind"] and completed["actual_text"] is None and
                    completed["training_admitted"] is False and completed["qualification_allowed"] is False,
                    "D1_COMPLETED_PLAN_JOIN")
            raw = archive.read(clip["original_member"])
            pcm = float_wav_pcm(raw)
            audio, technical = completed["audio"], completed["technical"]
            require(audio["native_sha256"] == clip["original_wav"]["sha256"] == sha(raw) and
                    audio["native_bytes"] == clip["original_wav"]["bytes"] == len(raw) and
                    sha(pcm) == clip["original_pcm"]["sha256"] and sha(pcm) not in seen_pcm and
                    technical["native_frames"] == clip["original_pcm"]["frames"] == len(pcm) // 4 and
                    technical["native_rate"] == 24000 and
                    technical["duration_s"] == len(pcm) / (4 * 24000) and
                    type(technical["technical_audio_pass"]) is bool,
                    "D1_NATIVE_WAV_PCM_JOIN")
            seen_pcm.add(sha(pcm))
            old_wav = retained(folder + "audio-16k.wav", 2 * 1024 * 1024)
            with wave.open(io.BytesIO(old_wav), "rb") as reader:
                require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                         reader.getnframes(), reader.getcomptype()) ==
                        (1, 2, 16000, audio["audio_frames"], "NONE"), "D1_GENERATED_PCM_FORMAT")
                old_pcm = reader.readframes(reader.getnframes())
            require(audio["sample_rate_hz"] == 16000 and sha(old_wav) == audio["wav_sha256"] and
                    len(old_wav) == audio["wav_bytes"] == len(old_pcm) + 44 and
                    sha(old_pcm) == audio["pcm_sha256"] and len(old_pcm) == audio["audio_frames"] * 2 and
                    audio["audio_frames"] == clip["derived_pcm"]["frames"],
                    "D1_GENERATED_WAV_PCM_JOIN")
            sources[clip["id"]] = {
                "source_id": cell["cell_id"], "source_family_id": ref["source_family_id"],
                "speaker_id": ref["speaker_group_id"], "role": ref["role"],
                "split": "train" if ref["role"] == "train-candidate" else "development",
                "source_kind": "synthetic-reference-clone", "reference_id": ref["reference_id"],
                "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                "parent_source_sha256": ref["parent_source_sha256"],
                "tts_model_id": plan["model_id"], "tts_model_revision": plan["model_revision"],
                "intended_text": cell["prompt"], "intended_kind": cell["intended_kind"],
                "intended_keyword_id": cell["intended_keyword_id"],
                "generation_category": cell["generation_category"], "source_plan_sha256": expected_plan_sha,
                "generation_receipt_path": "evidence/source/d1/" + folder + "completed.json",
                "generation_receipt_sha256": sha(completed_raw),
                "generation_start_receipt_path": "evidence/source/d1/" + folder + "started.json",
                "generation_start_receipt_sha256": sha(started_raw),
                "original_generation_row": completed,
            }
    return sources, evidence, {"schema": D1_ADAPTER, "plan_sha256": expected_plan_sha,
                                "plan_file_sha256": sha(plan_raw)}


def add_source_role(roles, source, clip, measured=None):
    roles.add(source["role"], family=source["source_family_id"], speaker=source["speaker_id"],
              wav_sha256=clip["original_wav"]["sha256"], pcm_sha256=clip["original_pcm"]["sha256"])
    if "reference_id" in source:
        roles.add(source["role"], wav_sha256=source["reference_wav_sha256"],
                  pcm_sha256=source["reference_pcm_sha256"], parent_source_sha256=source["parent_source_sha256"])
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


def make_row(clip, source, input_root, asr_root, inputs_sha, vocab, retained):
    derived = read_bytes(input_root / relative(clip["derived_pcm"]["path"]), 2 * 1024 * 1024)
    original = read_bytes(input_root / relative(clip["original_wav"]["path"]), 16 * 1024 * 1024)
    wav, measured = quantize_pcm16(derived, clip["derived_pcm"]["frames"])
    generation = source["original_generation_row"]
    if source["source_kind"] == "synthetic-reference-clone":
        flags = measured["quality_flags"]
        technical_pass = generation["technical"]["technical_audio_pass"]
    else:
        flags = measured["quality_flags"] + generation["metrics"]["quality_flags"]
        technical_pass = generation["metrics"]["technically_valid"]
    if technical_pass is not True:
        flags = flags + ["original_generator_signal_invalid"]
    observed = [observation(asr_root, model, clip, inputs_sha, retained) for model in MODELS]
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


def create_view(inputs_path, expected_inputs_sha, archive_path, asr_root, output, *, fsmn_tokens=None,
                d1_plan=None, d1_plan_sha256=None, d1_generation_root=None):
    inputs_path, asr_root, output = Path(inputs_path), Path(asr_root), Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and ROOT not in output.parents and
            not any(path.is_symlink() for path in (output, *output.parents)), "NEW_PRIVATE_VIEW_REQUIRED")
    inputs_raw = read_bytes(inputs_path)
    require(sha(inputs_raw) == expected_inputs_sha, "ACTIVE_INPUT_MANIFEST_CHANGED")
    inputs = loads(inputs_raw)
    verify_inputs(inputs, inputs_path.parent)
    archive_raw = read_bytes(archive_path, 256 * 1024 * 1024)
    d1_args = (d1_plan, d1_plan_sha256, d1_generation_root)
    require(all(value is None for value in d1_args) or all(value is not None for value in d1_args),
            "D1_PLAN_PIN_AND_GENERATION_REQUIRED")
    source_adapter = None
    if d1_plan is not None:
        sources, evidence, source_adapter = d1_source_rows(
            archive_raw, inputs, read_bytes(d1_plan), d1_plan_sha256, d1_generation_root)
    else:
        sources, evidence = source_rows(archive_raw, inputs)
    token_raw = read_bytes(fsmn_tokens, 65536) if fsmn_tokens is not None else None
    vocab = fsmn_vocabulary(token_raw) if token_raw is not None else None
    output.mkdir(mode=0o700)
    producer_files = {}
    producer_names = ["automation/active_asr_weak_view.py", "kws_data_contract/__init__.py",
                      "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py",
                      "kws_data_contract/identity.py"]
    if source_adapter is not None:
        producer_names.append(BULK_POOL_PATH)
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
    retained, rows, roles = {}, [], SourceRoles()
    for clip in inputs["clips"]:
        source = sources[clip["id"]]
        row, wav, original, derived = make_row(clip, source, inputs_path.parent, asr_root,
                                               expected_inputs_sha, vocab, retained)
        add_source_role(roles, source, clip, row["audio_derivation"])
        save(output, "evidence/inputs/" + clip["derived_pcm"]["path"], derived)
        save(output, "evidence/inputs/" + clip["original_wav"]["path"], original)
        save(output, row["audio"], wav)
        rows.append(row)
    for name, raw in retained.items():
        save(output, name, raw)
    report = {"schema": SCHEMA, "policy": POLICY, "inputs_sha256": expected_inputs_sha,
              "producer": {"python": sys.version.split()[0], "source_files_sha256": producer_files},
              "source_archive_sha256": inputs["source_archive_sha256"], "items": rows,
              **({"source_adapter": source_adapter} if source_adapter is not None else {}),
              "split_scope": ("source-group-train-and-development-machine-weak-no-independent-test"
                              if source_adapter is not None else "original-train-only-no-independent-development"),
              "new_model_calls": 0, "training_runs": 0, "human_gold": False,
              "statistics": {"source_items": len(rows), "source_families": len({r['source']['source_family_id'] for r in rows}),
                             "completed_asr_results": sum(o["status"] == "complete" for r in rows for o in r["asr"]),
                             "machine_candidates": sum(r["decision"]["status"] == "machine_candidate" for r in rows),
                             "uncertain": sum(r["decision"]["status"] == "uncertain" for r in rows),
                             "disagreement": sum(r["decision"]["disagreement"] for r in rows),
                             "train_only": sum(r["source"]["split"] == "train" for r in rows),
                             "development": sum(r["source"]["split"] == "development" for r in rows)}}
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
    require(report["schema"] == SCHEMA and report["policy"] == POLICY and report["human_gold"] is False,
            "WEAK_VIEW_SCHEMA")
    inputs_raw = read_bytes(package / "evidence/inputs.json")
    require(sha(inputs_raw) == report["inputs_sha256"], "WEAK_INPUTS_CHANGED")
    inputs = loads(inputs_raw)
    verify_inputs(inputs, package / "evidence/inputs")
    archive_raw = read_bytes(package / "evidence/source-original.zip", 256 * 1024 * 1024)
    source_adapter = report.get("source_adapter")
    if source_adapter is not None:
        sources, _, rebuilt_adapter = d1_source_rows(
            archive_raw, inputs, read_bytes(package / "evidence/source/d1/plan.json"),
            source_adapter["plan_sha256"], package / "evidence/source/d1")
        require(source_adapter == rebuilt_adapter, "D1_SOURCE_ADAPTER_CHANGED")
    else:
        sources, _ = source_rows(archive_raw, inputs)
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
                                      package / "evidence/asr", report["inputs_sha256"], vocab, {})
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


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    for name in ("inputs", "source-archive", "asr-results", "output"):
        create.add_argument("--" + name, required=True, type=Path)
    create.add_argument("--inputs-sha256", required=True)
    create.add_argument("--fsmn-tokens", type=Path)
    create.add_argument("--d1-plan", type=Path)
    create.add_argument("--d1-plan-sha256")
    create.add_argument("--d1-generation-root", type=Path)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("package", type=Path)
    inspect.add_argument("--consumer", choices=("rnn-four-token", "fsmn-fulltext"), default="fsmn-fulltext")
    inspect.add_argument("--split", choices=("train", "development"), default="train")
    args = cli.parse_args(argv)
    if args.command == "create":
        report = create_view(args.inputs, args.inputs_sha256, args.source_archive, args.asr_results,
                             args.output, fsmn_tokens=args.fsmn_tokens, d1_plan=args.d1_plan,
                             d1_plan_sha256=args.d1_plan_sha256, d1_generation_root=args.d1_generation_root)
        print(json.dumps(report["statistics"], sort_keys=True))
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
