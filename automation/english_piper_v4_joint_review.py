"""Join saved Piper v4 evidence without generating audio or calling a model.

The original six archives are byte-pinned. Every transcript is checked against
its native receipt and the actual source WAV. Phone agreement is only a review
aid; this tool cannot confer acoustic truth, a dataset split, or admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import wave
import zipfile
from pathlib import Path

SCHEMA = "english-piper-v4-joint-review-v1"
SOURCE_RUN = 43898728142
SOURCE_ARTIFACT = 14240252131
PROVIDERS = ("qwen", "whisper", "parakeet")
SPEAKERS = {"p4535": 2, "p6701": 3, "p3638": 4, "p922": 5,
            "p2531": 6, "p1638": 7, "p8848": 8, "p6544": 9}
TARGETS = {
    "Hey Mero": {"ipa": "/heɪ ˈmiːroʊ/", "raw": "heɪ mˈiːɹoʊ",
                 "phones": ["h", "eɪ", "m", "i", "ɹ", "oʊ"]},
    "Hey Vosen": {"ipa": "/heɪ ˈvoʊzən/", "raw": "heɪ vˈoʊzən",
                  "phones": ["h", "eɪ", "v", "oʊ", "z", "ə", "n"]},
}
# Bytes and SHA256 values were independently read from GitHub artifact metadata.
ARCHIVES = {
    "source": ("piper_v4_source.zip", 655892,
        "9a611676829195c8ac07c07e50705cd8e98822703c60ab66e21ee5be03567193", 43898728142, 14240252131),
    "asr": ("piper_v4_asr_summary.zip", 1449,
        "00f8c92482303f236c7c1f9cfa0a84fecad5645b8b465fb4ce53e5c5f64eb3e0", 43900153196, 14241685973),
    "phones": ("piper_v4_phones.zip", 4641,
        "a0da3a19b1c301111d3c03ced029f3b133073538d72c3688ec554293a70df6ab", 43985775171, 14284288812),
    "qwen": ("piper_v4_asr_qwen.zip", 22882,
        "60e4a79d11b6974dd83a6aa387305f02aeb0c119d3a6406bfb36612dc6f02c41", 43900153196, 14241431616),
    "whisper": ("piper_v4_asr_whisper.zip", 23226,
        "58a3ee6d8a771ba305717df7a8e07d8128d47b1ae070d1c6f8e410d83f804435", 43900153196, 14241077857),
    "parakeet": ("piper_v4_asr_parakeet.zip", 23379,
        "8d629df169b4057f155352a0c20028cb30ed5d9660bb4d5ab03118b8e1510c19", 43900153196, 14240557844),
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs)


def archive_bytes(path, key):
    path = Path(path)
    _, size, digest, _, _ = ARCHIVES[key]
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == size,
         "ARCHIVE_SIZE_OR_TYPE: " + key)
    raw = path.read_bytes()
    need(sha(raw) == digest, "ARCHIVE_HASH: " + key)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        need(len({x.filename for x in entries}) == len(entries), "DUPLICATE_ZIP_MEMBER")
        need(sum(x.file_size for x in entries) <= 4 * 1024**2, "ARCHIVE_EXPANDED_LIMIT")
        # Only read bytes; no archive extraction or path traversal is performed.
        return {x.filename: archive.read(x) for x in entries if not x.is_dir()}


def by_id(rows, name):
    need(type(rows) is list and len(rows) == 16, "ROSTER_SIZE: " + name)
    result = {}
    for row in rows:
        cid = row.get("cell_id")
        need(type(cid) is str and cid not in result, "DUPLICATE_CELL: " + name)
        result[cid] = row
    return result


def unadmitted(row):
    need(row.get("human_gold") is False and row.get("training_admitted") is False,
         "UNEXPECTED_ADMISSION_OR_GOLD")


def source_rows(bundle):
    root = "piper-probe-parent/probe/"
    result = read_json(bundle[root + "result.json"])
    plan = read_json(bundle[root + "source-plan.json"])
    lock = read_json(bundle[root + "model-lock.json"])
    need(result.get("schema") == "english-ipa-piper-phoneme-probe-v4" and
         result.get("attempts") == result.get("completed") == 16 and
         result.get("status") == "INCOMPLETE_NO_RETRY", "SOURCE_TERMINAL_IDENTITY")
    need(sha(bundle[root + "source-plan.json"]) == result["source_plan_sha256"] and
         sha(bundle[root + "model-lock.json"]) == result["model_lock_sha256"],
         "SOURCE_PLAN_OR_MODEL_BINDING")
    need(lock.get("speaker_ids") == SPEAKERS, "SOURCE_VOICE_MAP")
    expected = {(word, voice) for word in TARGETS for voice in SPEAKERS}
    need(len(plan["cells"]) == 16 and
         {(c["phrase"], c["source_speaker_id"]) for c in plan["cells"]} == expected,
         "SOURCE_PLAN_ROSTER")
    for c in plan["cells"]:
        need(c["ipa"] == TARGETS[c["phrase"]]["ipa"] and
             c["raw_espeak_ipa"] == TARGETS[c["phrase"]]["raw"] and
             c["speaker_id"] == SPEAKERS[c["source_speaker_id"]], "SOURCE_FINAL_IPA")
    unadmitted(result)
    unadmitted(plan)
    rows, seen = {}, set()
    for name in sorted(bundle):
        if not name.startswith(root + "cells/") or not name.endswith("/completed.json"):
            continue
        record = read_json(bundle[name])
        cid = record["cell_id"]
        pair = (record["phrase"], record["source_speaker_id"])
        need(cid == name.split("/")[-2] and cid not in rows and pair in expected and
             pair not in seen, "SOURCE_RECEIPT_ROSTER")
        seen.add(pair)
        need(record["ipa_target"] == TARGETS[pair[0]]["ipa"] and
             record["raw_espeak_ipa"] == TARGETS[pair[0]]["raw"] and
             record["speaker_id"] == SPEAKERS[pair[1]], "SOURCE_RECEIPT_FINAL_IPA")
        unadmitted(record)
        need(record.get("pronunciation_verified") is False, "SOURCE_PRONUNCIATION_STATE")
        wav_member = name.rsplit("/", 1)[0] + "/audio-16k.wav"
        raw = bundle[wav_member]
        audio = record["audio"]
        need(len(raw) == audio["wav_bytes"] and sha(raw) == audio["wav_sha256"], "WAV_HASH")
        with wave.open(io.BytesIO(raw), "rb") as wav:
            need((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) ==
                 (16000, 1, 2, "NONE") and wav.getnframes() == audio["audio_frames"], "WAV_GEOMETRY")
            pcm = wav.readframes(wav.getnframes())
        need(sha(pcm) == audio["pcm_sha256"], "PCM_HASH")
        rows[cid] = {"cell_id": cid, "phrase": pair[0], "speaker": pair[1],
            "source_family_id": "piper-libritts-" + pair[1], "ipa": record["ipa_target"],
            "raw_espeak_ipa": record["raw_espeak_ipa"], "wav_member": wav_member,
            "wav_sha256": audio["wav_sha256"], "pcm_sha256": audio["pcm_sha256"],
            "frames": audio["audio_frames"], "duration_seconds": audio["audio_frames"] / 16000}
    need(seen == expected and len(rows) == 16 and
         len({r["pcm_sha256"] for r in rows.values()}) == 16, "SOURCE_COMPLETE_UNIQUE_PCM")
    return rows, result


def normalized_text(text):
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def comparison_phones(tokens):
    # This deliberately cannot verify lexical stress or phonetic duration.
    return [t.replace("ˈ", "").replace("ˌ", "").replace("ː", "") for t in tokens]


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        out = [i]
        for j, b in enumerate(right, 1):
            out.append(min(out[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = out
    return row[-1]


def join(source, source_result, asr, phones, natives):
    need(asr.get("schema") == "english-ipa-piper-v4-three-blind-asr-v1" and
         asr.get("asr_calls") == 48 and set(asr.get("providers", [])) == set(PROVIDERS) and
         asr.get("source_cell_closure_complete") is True, "ASR_COMPLETE")
    need(phones.get("schema") == "english-ipa-piper-v4-independent-ctc-observation-v1" and
         phones.get("status") == "COMPLETE" and
         phones.get("attempted_forwards") == phones.get("completed_forwards") == 16,
         "PHONES_COMPLETE")
    for observation in (asr, phones):
        unadmitted(observation)
        need(observation.get("source_run_id") == SOURCE_RUN and
             observation.get("source_artifact_id") == SOURCE_ARTIFACT, "OBSERVATION_SOURCE_ID")
    need(phones.get("source_artifact_sha256") == ARCHIVES["source"][2], "PHONE_SOURCE_HASH")
    by_asr, by_phone = by_id(asr["cells"], "asr"), by_id(phones["observations"], "phones")
    need(set(source) == set(by_asr) == set(by_phone), "SOURCE_ASR_PHONE_ROSTER")
    rows = []
    for cid, raw in sorted(source.items()):
        a, p = by_asr[cid], by_phone[cid]
        for x in (a, p):
            unadmitted(x)
        need(a.get("phrase") == p.get("target_phrase") == raw["phrase"] and
             a.get("ipa") == p.get("target_ipa") == raw["ipa"] and
             a.get("speaker") == raw["speaker"] and
             p.get("source_wav_sha256") == raw["wav_sha256"] and
             p.get("source_frames") == raw["frames"], "JOIN_IDENTITY_MISMATCH")
        need(p.get("pronunciation_verified") is False and
             p.get("machine_observation_only") is True and
             p.get("target_is_not_passed_to_model") is True, "PHONE_OBSERVATION_BOUNDARY")
        transcripts, native_hashes = {}, {}
        for provider in PROVIDERS:
            member = provider + "/cells/" + cid + "/native.json"
            receipt_bytes = natives[provider][member]
            receipt = read_json(receipt_bytes)
            need(receipt.get("decoder_completed") is True and
                 receipt.get("audio_wav_sha256") == raw["wav_sha256"] and
                 receipt.get("processed_start_frame") == 0 and
                 receipt.get("processed_end_frame") == raw["frames"] and
                 receipt.get("actual_text") == a["transcripts"][provider], "NATIVE_ASR_BINDING")
            transcripts[provider] = receipt["actual_text"]
            native_hashes[provider] = sha(receipt_bytes)
        folded = {k: normalized_text(v) for k, v in transcripts.items()}
        tokens = p["predicted_stressless_phone_tokens"]
        reference = TARGETS[raw["phrase"]]["phones"]
        edits = edit_distance(comparison_phones(tokens), reference)
        rows.append({**raw, "asr_transcripts": transcripts, "asr_native_sha256": native_hashes,
            "asr_normalized": folded, "asr_exact_spelling_providers":
                [k for k in PROVIDERS if folded[k] == normalized_text(raw["phrase"])],
            "asr_three_way_spelling_agreement": len(set(folded.values())) == 1,
            "phone_raw_tokens": p["predicted_raw_ipa_tokens"], "phone_tokens": tokens,
            "reference_segment_tokens": reference, "phone_segment_edit_distance": edits,
            "phone_segments_match_ignoring_stress_length": edits == 0,
            "hey_segment_prefix_observed": comparison_phones(tokens)[:2] == ["h", "eɪ"],
            "review_focus": "SPELLING_VERSUS_PHONETICS" if edits == 0 else "PHRASE_AND_KEY_PHONEMES",
            "label_status": "UNCERTAIN_REVIEW_REQUIRED", "source_split": None,
            "exposed_diagnostic": True, "allowed_use": ["pronunciation_review", "engineering_pcm_replay"],
            "formal_usable_train": False, "formal_usable_dev": False, "trusted_negative": False,
            "human_gold": False, "pronunciation_verified": False, "formal_training_admitted": False,
            "training_admitted": False, "independent_evaluation": False,
            "word_end_annotation": None, "word_end_latency_evaluable": False})
    counts = {}
    for phrase in TARGETS:
        selected = [r for r in rows if r["phrase"] == phrase]
        counts[phrase] = {"source_rows": len(selected), "source_voice_families": len({r["speaker"] for r in selected}),
            "saved_asr_observations": 3 * len(selected), "saved_phone_observations": len(selected),
            "phone_segment_matches_ignoring_stress_length": sum(r["phone_segments_match_ignoring_stress_length"] for r in selected),
            "uncertain_review_required": len(selected), "formal_usable_train": 0,
            "formal_usable_dev": 0, "trusted_negative": 0, "independent_evaluation": 0}
    return {"schema": SCHEMA, "status": "JOIN_COMPLETE_REVIEW_REQUIRED", "source_run_id": SOURCE_RUN,
        "source_artifact_id": SOURCE_ARTIFACT, "source_original_status": source_result["status"],
        "source_actual_completed_cells": len(rows), "source_aggregate_status_mismatch_preserved": True,
        "input_archives": {k: {"file": v[0], "bytes": v[1], "sha256": v[2],
            "run_id": v[3], "artifact_id": v[4]} for k, v in ARCHIVES.items()},
        "counts_by_phrase": counts, "total_audio_seconds": sum(r["frames"] for r in rows) / 16000,
        "new_tts_calls": 0, "new_asr_calls": 0, "new_phone_forwards": 0,
        "new_kws_forwards": 0, "optimizer_updates": 0, "training_admitted": False,
        "human_gold": False, "pronunciation_verified": False,
        "limitations": ["Machine segment agreement does not verify stress, duration or acoustic truth.",
            "Spelling disagreement does not establish a negative label.",
            "Source plan assigns no train/development split; rows are exposed diagnostic material.",
            "No word-end annotation or verified one-second real post-word context exists.",
            "No formal training can start from this batch until evidence-backed labels and grouped roles exist."],
        "rows": rows}


def build(input_dir):
    archives = {key: archive_bytes(Path(input_dir) / value[0], key) for key, value in ARCHIVES.items()}
    source, terminal = source_rows(archives["source"])
    asr = read_json(archives["asr"]["three-asr-summary.json"])
    phones = read_json(archives["phones"]["piper-ipa-result/result.json"])
    phone_lock = read_json(archives["phones"]["piper-ipa-result/model-lock.json"])
    lock_hash = sha(json.dumps(phone_lock["meta"], ensure_ascii=False,
                               sort_keys=True, separators=(",", ":")).encode())
    need(lock_hash == phone_lock["manifest_sha256"] == phones["model_manifest_sha256"] and
         phone_lock["meta"]["model_revision"] == "7be1e6f27c8163ce790688d62fd9ce1731177c2e" and
         phone_lock["meta"]["files"]["vocab.json"]["sha256"] ==
         "f8fb8fa1768ddf44c3ba04248be4dce226a7d9779bef329a825ba572720eb388",
         "PHONE_MODEL_VOCABULARY_BINDING")
    for provider in PROVIDERS:
        summary = read_json(archives[provider][provider + "/summary.json"])
        need(summary.get("status") == "COMPLETE" and summary.get("provider") == provider and
             summary.get("asr_attempts") == summary.get("completed_asr") == 16 and
             summary.get("source_artifact_sha256") == ARCHIVES["source"][2] and
             summary.get("source_run_id") == SOURCE_RUN, "NATIVE_PROVIDER_COMPLETE")
        need(set(by_id(summary["rows"], provider)) == set(source), "NATIVE_PROVIDER_ROSTER")
        unadmitted(summary)
    result = join(source, terminal, asr, phones, {p: archives[p] for p in PROVIDERS})
    result["source_plan_sha256"] = terminal["source_plan_sha256"]
    result["source_model_lock_sha256"] = terminal["model_lock_sha256"]
    result["input_manifest_sha256"] = sha(json.dumps(result["input_archives"], ensure_ascii=False,
                                                     sort_keys=True, separators=(",", ":")).encode())
    result["observer_representation"] = {
        "model_id": phone_lock["meta"]["model_id"], "model_revision": phone_lock["meta"]["model_revision"],
        "model_manifest_sha256": lock_hash, "vocab_sha256": phone_lock["meta"]["files"]["vocab.json"]["sha256"],
        "vocab_bytes": 509, "vocab_size_without_added_specials": 43,
        "vocabulary_source": "https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut/blob/main/vocab.json",
        "vocabulary_verification": "Official displayed JSON reconstructed to exact original asset size/SHA256.",
        "i_and_small_capital_i_are_distinct": True, "rhotic_symbol": "ɹ",
        "can_emit_length_mark": False, "can_emit_lexical_stress": False,
        "target_projection": "iː -> i; target r notation -> ɹ; stress excluded from segment comparison.",
        "cannot_verify": ["vowel duration", "lexical stress", "naturalness", "acoustic truth"]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build(args.input_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "joint-review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["cell_id", "phrase", "speaker", "wav_sha256", "duration_seconds", "qwen", "whisper", "parakeet",
              "phone_tokens", "phone_segment_edit_distance", "review_focus", "label_status", "formal_usable_train", "formal_usable_dev"]
    with (args.output / "joint-review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(result["rows"], key=lambda r: (r["phrase"], r["phone_segment_edit_distance"], r["speaker"])):
            record = {k: row[k] for k in fields if k in row}
            record.update(row["asr_transcripts"])
            record["phone_tokens"] = " ".join(row["phone_tokens"])
            writer.writerow(record)
    print(json.dumps({"status": result["status"], "counts_by_phrase": result["counts_by_phrase"],
                      "new_model_calls": 0, "training_admitted": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
