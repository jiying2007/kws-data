"""Frozen Kokoro V7 train/dev source recipe and independent observer identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_PLAN_PATH = HERE / "english_kws_dual_track_v7_source_plan.json"
REGISTRATION_PATH = HERE / "english_kws_dual_track_v7_registration.json"
SOURCE_PLAN_BLOB_SHA1 = "9eb148ed2dd39df67ac28573e2f3a752120be366"
REGISTRATION_BLOB_SHA1 = "9f9542b36d96d3621ed9935d5a9627e80c9d1e37"

SCHEMA = "english-kws-dual-track-v7-kokoro-source-recipe-v1"
CORPUS_SCHEMA = "english-kws-dual-track-v7-kokoro-corpus-v1"
EVIDENCE_SCHEMA = "english-kws-dual-track-v7-kokoro-machine-evidence-v1"
CAPTURE_SCHEMA = "english-kws-dual-track-v7-kokoro-observer-capture-v1"
SOURCE_COUNT = 96
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
ROLES = ("train", "development")

MODEL = {
    "repository": "hexgrad/Kokoro-82M",
    "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "checkpoint_path": "kokoro-v1_0.pth",\n    "model_path": "kokoro-v1_0.pth",
    "checkpoint_sha256": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",\n    "model_sha256": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "config_path": "config.json",
    "voices_directory": "voices",
    "package": "kokoro==0.9.4",
    "package_license": "Apache-2.0",
    "source_commit": "dfb907a02bba8152ca444717ca5d78747ccb4bec",
    "model_license": "Apache-2.0",
    "code_license": "Apache-2.0",
    "native_sample_rate": 24000,
    "output_sample_rate": 16000,
    "phoneme_path": "direct KModel raw IPA; no G2P, no eSpeak, no orthographic fallback",
    "voice_pack_format": "torch weights_only tensor [1,256]",
}

OBSERVERS = {
    "qwen": {"recognizer_id": "qwen", "model_family": "qwen3-asr",
        "model_id": "Qwen/Qwen3-ASR-0.6B", "model_revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "model_sha256": "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
        "decode_config": {"language": "English", "context": "", "max_new_tokens": 128,
            "batch_size": 1, "dtype": "float32", "device": "cpu", "attention": "sdpa"}},
    "whisper": {"recognizer_id": "whisper", "model_family": "whisper",
        "model_id": "openai/whisper-small", "model_revision": "31243bad24cc746f07d4c8bfdd2d974872cb1803",
        "model_sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
        "decode_config": {"language": "en", "fp16": False, "initial_prompt": None,
            "condition_on_previous_text": False, "temperature": 0.0, "device": "cpu"}},
    "parakeet": {"recognizer_id": "parakeet", "model_family": "parakeet-tdt",
        "model_id": "nvidia/parakeet-tdt-0.6b-v3", "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "model_sha256": "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
        "decode_config": {"device": "cpu", "prompt": None, "threads": 4, "timeout_seconds": 180}},
    "phoneme": {"recognizer_id": "phoneme", "model_family": "wav2vec2-xlsr",
        "model_id": "facebook/wav2vec2-xlsr-53-espeak-cv-ft", "model_revision": "2c733782da5604684829819a5eb744c193fe9398",
        "model_sha256": "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551",
        "decode_config": {"device": "cpu", "decoder": "ctc-greedy", "target_input": None,
            "special_tokens_removed": True, "phonemizer_used": False, "weights_only": True}},
}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_pairs)


def source_documents():
    source_plan = read_json(SOURCE_PLAN_PATH)
    registration = read_json(REGISTRATION_PATH)
    plan_sha = hashlib.sha256(SOURCE_PLAN_PATH.read_bytes()).hexdigest()
    if plan_sha != registration["source_map_evidence"]["source_plan_sha256"]:
        raise ValueError("SOURCE_PLAN_REGISTRATION_SHA256")
    if source_plan["source_model"]["revision"] != MODEL["revision"]:
        raise ValueError("KOKORO_MODEL_REVISION_PIN")
    if source_plan["source_model"]["checkpoint_sha256"] != MODEL["checkpoint_sha256"]:
        raise ValueError("KOKORO_MODEL_CHECKPOINT_PIN")
    return source_plan, registration


def source_groups():
    source_plan, registration = source_documents()
    roles = source_plan["source_selection"]["roles"]
    selected = registration["source_map_evidence"]["selected_voice_ids"]
    result = []
    seen = set()
    for item in roles:
        index = item["speaker_index"]
        voice_id = item["voice_id"]
        if selected.get(str(index)) != voice_id or voice_id in seen:
            raise ValueError("KOKORO_VOICE_SOURCE_MAP_IDENTITY")
        seen.add(voice_id)
        result.append({
            "source_group_id": item["source_group_id"],
            "source_speaker_id": voice_id,
            "source_speaker_index": index,
            "voice_family": "kokoro-82m-" + voice_id,
            "role": item["role"],
        })
    if len(result) != 8 or len(seen) != 8:
        raise ValueError("EXACT_EIGHT_DISTINCT_KOKORO_VOICES")
    return result


def observer_identity(provider):
    row = OBSERVERS[provider]
    return {key: row[key] for key in ("recognizer_id", "model_family", "model_sha256")} | {
        "decode_config_sha256": digest(row["decode_config"])
    }


def recipe():
    source_plan, registration = source_documents()
    groups = source_groups()
    rows = []
    for group in groups:
        for cell in source_plan["cells"]:
            labels = cell["class_by_candidate"]
            requested_class = 1 if labels["VOSEN"] == 1 else 2 if labels["MERO"] == 1 else 0
            hypothesis = ("vosen-us-s-v7" if requested_class == 1 else
                          "mero-us-eh-v7" if requested_class == 2 else
                          "contrast-negative-v7" if cell["kind"] == "contrast-negative" else
                          "ordinary-negative-v7")
            speed = round(1.0 / float(cell["length_scale"]), 8)
            request = {
                "schema": "english-kws-dual-track-v7-kokoro-tts-request-v1",
                "model": MODEL,
                "voice_id": group["source_speaker_id"],
                "source_group_id": group["source_group_id"],
                "role": group["role"],
                "cell_id": cell["id"],
                "text": cell["text"],
                "raw_ipa": cell["raw_ipa"],
                "speed": speed,
                "length_scale": cell["length_scale"],
                "sample_rate": 16000,
            }
            request_sha = hashlib.sha256(canonical(request)).hexdigest()
            row_id = "v7-kokoro-" + request_sha[:24]
            rows.append({
                **{key: group[key] for key in (
                    "source_group_id", "source_speaker_id", "source_speaker_index",
                    "voice_family", "role")},
                "id": row_id,
                "request_sha256": request_sha,
                "cell_id": cell["id"],
                "requested_text": cell["text"],
                "requested_class": requested_class,
                "pronunciation_hypothesis": hypothesis,
                "kind": cell["kind"],
                "length_scale": cell["length_scale"],
                "speed": speed,
                "raw_phonemes": cell["raw_ipa"],
                "candidate_labels": labels,
                "voice_id": group["source_speaker_id"],
                "generation_status": "pending",
            })
    if len(rows) != SOURCE_COUNT or len({row["id"] for row in rows}) != SOURCE_COUNT:
        raise ValueError("EXACT_96_UNIQUE_KOKORO_SOURCE_ROWS")
    return {
        "schema": SCHEMA,
        "source_plan_sha256": hashlib.sha256(SOURCE_PLAN_PATH.read_bytes()).hexdigest(),
        "registration_sha256": hashlib.sha256(REGISTRATION_PATH.read_bytes()).hexdigest(),
        "source_count": SOURCE_COUNT,
        "source_groups": groups,
        "source_role_counts": {"train": 6, "development": 2},
        "candidate_ids": ["hey-vosen-v7", "hey-mero-v7"],
        "observer_providers": list(PROVIDERS),
        "rows": rows,
        "human_gold": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
