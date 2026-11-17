"""Frozen new-source recipe. Standard library only; no audio/model execution."""
from __future__ import annotations

import hashlib
import json

SCHEMA = "english-synthetic192-recipe-v1"
SOURCE_COUNT = 192
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
    "piper_version": "1.8.0", "native_sample_rate": 22050,
    "phonemizer_voice": "en-us", "num_speakers": 904,
    "dataset": "LibriTTS train-clean-360", "dataset_license": "CC-BY-4.0",
    "engine_source": "https://github.com/OHF-Voice/piper1-gpl/tree/v1.8.0",
    "engine_license": "GPL-3.0", "voice_repository_license": "MIT",
}
LEXICON = {
    "repository": "cmusphinx/cmudict",
    "commit": "74790861f652b15e4ac49015a90074ad62a27690",
    "path": "cmudict.dict", "git_blob_sha1": "2c0411740cce3e2026a80b90b650d5f6a7258164",
    "entries": ["hey HH EY1", "mero M EH1 R OW0", "frozen F R OW1 Z AH0 N"],
    "interpretation": "US lexicon research hypotheses; not owner pronunciation truth",
}
# Numeric IDs are shared across LibriSpeech/LibriTTS and all descendants.
OLD_LIBRI_IDS = (84, 174, 251, 422, 652, 777, 1272, 1462, 1673, 1919,
    1988, 1993, 2035, 2078, 2086, 2277, 2412, 2428, 2803, 2902,
    3000, 3081, 3170, 3536, 3576, 3752, 3853, 5338, 5536, 5694,
    5895, 6241, 6295, 6313, 6319, 6345, 7850, 7976, 8297, 8842)
OLD_PIPER_IDS = (3922, 8699, 4535, 6701, 3638, 922, 2531, 1638, 8848, 6544)
NEW_VOICES = (
    (10, 3615), (11, 318), (12, 6104), (13, 1382), (14, 5400), (15, 5712),
    (16, 2769), (17, 2573), (18, 1463), (19, 6458), (20, 3274), (21, 4356),
    (22, 8498), (23, 5570), (24, 176), (25, 339),
    (26, 28), (27, 5909), (28, 3869), (29, 4899),
    (30, 64), (31, 3368), (32, 3307), (33, 5618),
)
CELLS = (
    {"cell_id": "mero-natural", "requested_text": "Hey Mero.", "requested_class": 2,
     "pronunciation_hypothesis": "mero-us-eh-v1", "generation_mode": "en-us-text"},
    {"cell_id": "mero-historical-i", "requested_text": "Hey Meero.", "requested_class": 2,
     "pronunciation_hypothesis": "mero-us-i-v1", "generation_mode": "en-us-text"},
    {"cell_id": "vosen-natural", "requested_text": "Hey Vosen.", "requested_class": 1,
     "pronunciation_hypothesis": "vosen-us-s-v1", "generation_mode": "en-us-text"},
    {"cell_id": "vosen-voiced-z", "requested_text": "Hey Vosen", "requested_class": 1,
     "pronunciation_hypothesis": "vosen-us-z-v1", "generation_mode": "direct-phoneme-ids",
     "raw_phonemes": "heɪ vˈoʊzən"},
    {"cell_id": "ordinary-time", "requested_text": "What time is it?", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "generation_mode": "en-us-text"},
    {"cell_id": "ordinary-light", "requested_text": "Please turn on the kitchen light.", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "generation_mode": "en-us-text"},
    {"cell_id": "ordinary-package", "requested_text": "The package is by the front door.", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "generation_mode": "en-us-text"},
    {"cell_id": "ordinary-window", "requested_text": "Could you open the window?", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "generation_mode": "en-us-text"},
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

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def source_groups():
    result = []
    for index, speaker in NEW_VOICES:
        role = "train" if index < 22 else "development" if index < 26 else "calibration" if index < 30 else "locked-test"
        result.append({"source_group_id": f"libritts-speaker-{speaker}",
            "source_speaker_id": f"p{speaker}", "source_speaker_index": index,
            "voice_family": f"piper-libritts-{speaker}", "role": role})
    return result

def observer_identity(provider):
    row = OBSERVERS[provider]
    return {key: row[key] for key in ("recognizer_id", "model_family", "model_sha256")} | {
        "decode_config_sha256": digest(row["decode_config"])}

def recipe():
    rows = []
    for group in source_groups():
        for cell in CELLS:
            request = {"model": MODEL, "source": group, "cell": cell,
                "synthesis_config": {"length_scale": 1.0, "noise_scale": 0.333,
                    "noise_w_scale": 0.333, "normalize_audio": False, "volume": 1.0}}
            rows.append(group | dict(cell) | {"id": "ens192-" + digest(request)[:24],
                "request_sha256": digest(request), "human_gold": False,
                "label_basis": "PREREGISTERED_SYNTHETIC_RESEARCH_HYPOTHESIS"})
    return {"schema": SCHEMA, "source_count": SOURCE_COUNT,
        "model": MODEL, "lexicon_evidence": LEXICON, "rows": rows,
        "exclusion_speakers": [f"p{x}" for x in sorted(set(OLD_LIBRI_IDS + OLD_PIPER_IDS))],
        "same_engine_shared_pretraining": True, "independent_human_speakers_claimed": False,
        "new_tts_calls_max": 192, "new_calls_per_observer_max": 192,
        "retry_count": 0, "regenerate_old_audio": False,
        "final_owner_ipa_declared": False, "kws_forward_calls": 0}
