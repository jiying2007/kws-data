"""Frozen V7 dual-track source recipe; no audio/model execution at import time."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_PLAN_PATH = HERE / "english_kws_dual_track_v7_source_plan.json"
REGISTRATION_PATH = HERE / "english_kws_dual_track_v7_registration.json"

SCHEMA = "english-kws-dual-track-v7-source-recipe-v1"
SOURCE_COUNT = 144
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
ROLES = ("train", "development", "calibration", "locked-test")

MODEL = {
    "repository": "rhasspy/piper-voices",
    "revision": "de3dcfdf1912bb49726dc4aa11c26017ce2ac62a",
    "model_path": "en/en_US/libritts/high/en_US-libritts-high.onnx",
    "model_bytes": 136673811,
    "model_sha256": "9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f",
    "config_path": "en/en_US/libritts/high/en_US-libritts-high.onnx.json",
    "config_bytes": 20163,
    "config_md5": "cb1564bb779f3b1f87e2de778adeb669",
    "piper_version": "1.8.0",
    "native_sample_rate": 22050,
    "phonemizer_voice": "en-us",
    "num_speakers": 904,
    "dataset": "LibriTTS train-clean-360",
    "dataset_license": "CC-BY-4.0",
    "engine_source": "https://github.com/OHF-Voice/piper1-gpl/tree/v1.8.0",
    "engine_license": "GPL-3.0-or-later",
    "voice_repository_license": "MIT (repository metadata only; model card governs voice data provenance)",
}
MODEL_CARD = {
    "path": "en/en_US/libritts/high/MODEL_CARD",
    "bytes": 255,
    "md5": "cdeac934f1154489924a071470b22365",
    "dataset_url": "http://www.openslr.org/60/",
    "dataset_license": "CC BY 4.0",
    "training": "Trained from scratch on train-clean-360.",
}
ENGINE = {
    "repository": "OHF-Voice/piper1-gpl",
    "commit": "639388b6317fc4731e91d53da42aea68fd4166ff",
    "version": "1.8.0",
    "license": "GPL-3.0-or-later",
}

NEW_VOICES = (
    (496, 4290),
    (497, 2999),
    (498, 2427),
    (499, 637),
    (500, 2229),
    (501, 1874),
    (502, 3446),
    (503, 9023),
    (504, 3114),
    (505, 6235),
    (506, 4860),
    (507, 4519),
)

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


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_pairs)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_documents():
    source_plan = read_json(SOURCE_PLAN_PATH)
    registration = read_json(REGISTRATION_PATH)
    return source_plan, registration


def source_groups():
    source_plan, registration = source_documents()
    speaker_ids = registration["source_map_evidence"]["selected_speaker_ids"]
    result = []
    for item in source_plan["source_selection"]["roles"]:
        index = item["speaker_index"]
        speaker_id = speaker_ids[str(index)]
        result.append({
            "source_group_id": item["source_group_id"],
            "source_speaker_id": speaker_id,
            "source_speaker_index": index,
            "voice_family": "piper-libritts-" + speaker_id,
            "role": item["role"],
        })
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
            candidate_labels = cell["class_by_candidate"]
            requested_class = 1 if candidate_labels["VOSEN"] == 1 else 2 if candidate_labels["MERO"] == 1 else 0
            if requested_class == 1:
                hypothesis = "vosen-us-s-v7"
            elif requested_class == 2:
                hypothesis = "mero-us-eh-v7"
            elif cell["kind"] == "contrast-negative":
                hypothesis = "contrast-negative-v7"
            else:
                hypothesis = "ordinary-negative-v7"
            frozen_cell = {
                "cell_id": cell["id"],
                "requested_text": cell["text"],
                "requested_class": requested_class,
                "pronunciation_hypothesis": hypothesis,
                "generation_mode": "direct-phoneme-ids",
                "raw_phonemes": cell["raw_ipa"],
                "length_scale": cell["length_scale"],
                "kind": cell["kind"],
                "candidate_labels": candidate_labels,
            }
            synthesis = {
                "length_scale": cell["length_scale"],
                "noise_scale": 0.333,
                "noise_w_scale": 0.333,
                "normalize_audio": False,
                "volume": 1.0,
            }
            request = {"model": MODEL, "model_card": MODEL_CARD, "source": group,
                       "cell": frozen_cell, "synthesis_config": synthesis}
            rows.append(group | frozen_cell | {
                "id": "kws-v7-" + digest(request)[:24],
                "request_sha256": digest(request),
                "human_gold": False,
                "label_basis": "PREREGISTERED_SYNTHETIC_RESEARCH_HYPOTHESIS",
            })
    return {
        "schema": SCHEMA,
        "source_count": SOURCE_COUNT,
        "model": MODEL,
        "model_card": MODEL_CARD,
        "engine": ENGINE,
        "source_plan_sha256": registration["source_map_evidence"]["source_plan_sha256"],
        "source_groups": groups,
        "rows": rows,
        "observer_identities": {p: observer_identity(p) for p in PROVIDERS},
        "same_engine_shared_pretraining": True,
        "independent_human_speakers_claimed": False,
        "new_tts_calls_max": SOURCE_COUNT,
        "new_calls_per_observer_max": SOURCE_COUNT,
        "retry_count": 0,
        "regenerate_old_audio": False,
        "final_owner_ipa_declared": True,
        "kws_forward_calls": 0,
        "optimizer_updates": 0,
        "candidate_ids": ["hey-vosen-v7", "hey-mero-v7"],
    }
