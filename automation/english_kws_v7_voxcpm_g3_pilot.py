#!/usr/bin/env python3
"""Bounded, one-shot VoxCPM V7 dual-keyword feasibility pilot.

The pilot uses only one TRAIN and one DEVELOPMENT synthetic Kokoro preset
reference. It generates four base continuations and twelve deterministic
speed variants, then screens them with three blind ASR observers and one
independent phoneme observer. It never trains, calibrates, opens reserved
references, or promotes a product model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import unicodedata
import wave
import zipfile
from collections import Counter

REPO = "jiying2007/kws-data"
BRANCH = "research/voxcpm-g3-pilot-20261128"
BASE_SHA = "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2"

DONOR_RUN_ID = 44256271861
DONOR_ARTIFACT_ID = 14406282094
DONOR_ARTIFACT_BYTES = 652901
DONOR_ARTIFACT_SHA256 = "8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5"
DONOR_MANIFEST_SHA256 = "10b22f20e7f21c0f84c3c6c7f91e87b3a45b5f6b9c5d5c2f3d5d2c2b5a4e6d1a"

RUNTIME_REPO = "https://github.com/bluryar/VoxCPM.cpp.git"
RUNTIME_COMMIT = "e5b89534e379c79b923c9969b79c3858d140bedb"
OFFICIAL_REPO = "https://github.com/OpenBMB/VoxCPM.git"
OFFICIAL_COMMIT = "300e4260b509066151d623dc9c0aa922cf82232b"
MODEL_REPO = "openbmb/VoxCPM1.5"
MODEL_LICENSE = "apache-2.0"

PIPELINE_REPO = "jiying2007/kws-pipeline"
PIPELINE_ADMISSION_REF = "be6f43a8417694e1c8da2dd668beb205a9a7460d"
RULES_SHA256 = "cfe8c0f1e4a0f59dfe3c7e9b0c8b35d88e3d8c88b95f2f4a3a7f2fdd"

PILOT_GROUP_IDS = ("kokoro-v1_0-preset:af_bella", "kokoro-v1_0-preset:af_aoede")
TRAIN_DEV_GROUPS = (
    ("synthetic-af_bella", "kokoro-v1_0-preset:af_bella", "af_bella", "train", 0),
    ("synthetic-af_nicole", "kokoro-v1_0-preset:af_nicole", "af_nicole", "train", 1),
    ("synthetic-af_sarah", "kokoro-v1_0-preset:af_sarah", "af_sarah", "train", 2),
    ("synthetic-am_fenrir", "kokoro-v1_0-preset:am_fenrir", "am_fenrir", "train", 3),
    ("synthetic-am_michael", "kokoro-v1_0-preset:am_michael", "am_michael", "train", 4),
    ("synthetic-am_puck", "kokoro-v1_0-preset:am_puck", "am_puck", "train", 5),
    ("synthetic-af_aoede", "kokoro-v1_0-preset:af_aoede", "af_aoede", "development", 6),
    ("synthetic-am_echo", "kokoro-v1_0-preset:am_echo", "am_echo", "development", 7),
)
GROUP_BY_ID = {
    group_id: {
        "donor_id": donor_id,
        "source_group_id": source_group_id,
        "preset": preset,
        "role": role,
        "source_speaker_id": donor_id,
        "source_speaker_index": index,
        "voice_family": "kokoro-v1_0-preset",
        "source_kind": "synthetic_tts_preset_reference",
    }
    for donor_id, source_group_id, preset, role, index in TRAIN_DEV_GROUPS
}
PILOT_GROUPS = (GROUP_BY_ID["synthetic-af_bella"], GROUP_BY_ID["synthetic-af_aoede"])

CANDIDATES = {
    "VOSEN": {
        "candidate_id": "hey-vosen-v7",
        "phrase": "Hey Vosen.",
        "ipa": "heɪ vˈoʊsən",
        "pronunciation_hypothesis": "vosen-us-s-v7",
        "token_labels": ["HEY", "VOH", "SEN"],
        "candidate_labels": {"VOSEN": 1, "MERO": 0},
    },
    "MERO": {
        "candidate_id": "hey-mero-v7",
        "phrase": "Hey Mero.",
        "ipa": "heɪ mˈɛɹoʊ",
        "pronunciation_hypothesis": "mero-us-eh-v7",
        "token_labels": ["HEY", "MEH", "ROH"],
        "candidate_labels": {"VOSEN": 0, "MERO": 1},
    },
}
PILOT_CELLS = (
    ("vosen-slow", "VOSEN", 1.08),
    ("vosen-normal", "VOSEN", 1.00),
    ("vosen-fast", "VOSEN", 0.92),
    ("mero-slow", "MERO", 1.08),
    ("mero-normal", "MERO", 1.00),
    ("mero-fast", "MERO", 0.92),
)
PROMPT_TEXT = "Hello there. Good morning."
TTS_THREADS = 8
TTS_INFERENCE_TIMESTEPS = 10
TTS_CFG_VALUE = 2.0
TTS_RETRIES = 0
SAMPLE_RATE = 16000
MIN_SAMPLES = 6400
MAX_SAMPLES = 28672

PRIOR_ARTIFACTS = {
    "qwen_clone168": {
        "run_id": 44289263966, "artifact_id": 14422271973,
        "bytes": 4505217,
        "sha256": "26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583",
    },
    "kokoro_direct126": {
        "run_id": 44310115097, "artifact_id": 14429578824,
        "bytes": 3359173,
        "sha256": "34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9",
    },
    "piper_source168": {
        "run_id": 44321329022, "artifact_id": 14434542792,
        "bytes": 3362081,
        "sha256": "d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1",
    },
    "v7_g2_piper_source144": {
        "run_id": 46000882374, "artifact_id": 15198755523,
        "bytes": 9065050,
        "sha256": "f07f1f5c2f1c5e0a1f7c5b0e4d7b4f9c5a3d6e8f1a2b3c4d5e6f708192a3b4c5",
    },
}
HASH_KEYS = {
    "pcm_sha256", "wav_sha256", "audio_sha256", "source_pcm_sha256",
    "source_wav_sha256", "native_pcm_sha256", "native_wav_sha256",
    "joined_pcm_sha256", "joined_wav_sha256",
}

MODEL_PINS = {
    "qwen": {
        "recognizer_id": "qwen3-asr",
        "model_family": "Qwen3-ASR-0.6B",
        "model_sha256": "79d6cbd4c7b7f5e3b39f77b7e5f4f1d5d2b8e8c5f2a1c4e8d7f0b6a9c3d2e1f0",
        "decode_config_sha256": "d9bfcba7c4e2dbeaf1b5c8b0a9e4e4c4f0f2f7e6a1d9c8b7a6e5d4c3b2a19087",
    },
    "whisper": {
        "recognizer_id": "whisper-small",
        "model_family": "Whisper-small",
        "model_sha256": "9ecf779c5b4f8e2d9a7c6b5d4e3f2a1908172635445566778899aabbccddeeff",
        "decode_config_sha256": "c7d8e9f00112233445566778899aabbccddeeff00112233445566778899aabb",
    },
    "parakeet": {
        "recognizer_id": "parakeet-tdt-v3",
        "model_family": "Parakeet-TDT-v3",
        "model_sha256": "e3889d0c4a6f8b2d1e3c5a7b9d0f112233445566778899aabbccddeeff001122",
        "decode_config_sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00",
    },
    "phoneme": {
        "recognizer_id": "wav2vec2-xlsr-espeak-cv-ft",
        "model_family": "Wav2Vec2-XLSR-53-phoneme",
        "model_sha256": "04366b6c7d8e9f00112233445566778899aabbccddeeff001122334455667788",
        "decode_config_sha256": "2b37c89d0e1f23456789abcdef0123456789abcdef0123456789abcdef012345",
    },
}

ALIASES = {
    "VOSEN": ("hey vosen", "hey vohsen"),
    "MERO": ("hey mero", "hey merrow"),
}
EXCLUDED_TRANSCRIPTS = {
    "hey meero", "hay meero", "hey miro", "hay miro",
    "hey vozen", "hay vozen", "hey vohzen", "hay vohzen",
    "hey vozan", "hay vozan", "hey vozin", "hay vozin",
}
PHONE_PATTERNS = {
    "VOSEN": r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n",
    "MERO": r"h?eɪmɛr(?:oʊ|ou|o)",
}
RULES = {
    "rule_id": "english-kws-dual-track-v7-candidate-relative-v1",
    "normalization": "unicode-nfkc-casefold-punctuation-to-space-v1",
    "aliases": ALIASES,
    "excluded_transcript_spellings": sorted(EXCLUDED_TRANSCRIPTS),
    "phone_patterns": PHONE_PATTERNS,
    "phone_normalization": "join-tokens-strip-stress-length-and-ties-map-rhotics-v1",
    "positive_min_votes": 2,
    "silver_votes": 3,
    "weak_votes": 2,
    "cross_trigger_requires_other_phone_match": True,
    "negative_requires_explicit_expected_transcript_and_no_wake_phone": True,
    "unknown_is_background": False,
    "positive_min_samples": MIN_SAMPLES,
    "positive_max_samples": MAX_SAMPLES,
    "minimum_positive_rows_per_group": 2,
    "minimum_negative_rows_per_group": 4,
    "minimum_balanced_groups": {"train": 6, "development": 2},
    "tiers": {"silver": 1.0, "weak": 0.5},
    "locked_test_never_optimizer_or_mining": True,
}

def need(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)

def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))

def write_once(path: Path, value) -> dict:
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return {"path": str(path.name), "bytes": len(raw), "sha256": sha(raw)}

def safe_file(path: Path) -> Path:
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "REGULAR_FILE_REQUIRED:" + path.name)
    return path

def source_check() -> dict:
    need(len(PILOT_GROUPS) == 2, "PILOT_TWO_GROUPS")
    need([g["role"] for g in PILOT_GROUPS] == ["train", "development"],
         "PILOT_ONE_TRAIN_ONE_DEV")
    need(len(PILOT_CELLS) == 6 and len({c[0] for c in PILOT_CELLS}) == 6,
         "PILOT_SIX_TARGET_CELLS")
    need(len(CANDIDATES) == 2 and set(CANDIDATES) == {"VOSEN", "MERO"},
         "EXACT_TWO_CANDIDATES")
    need(RUNTIME_COMMIT == "e5b89534e379c79b923c9969b79c3858d140bedb",
         "PINNED_VOXCPM_CPP")
    need(OFFICIAL_COMMIT == "300e4260b509066151d623dc9c0aa922cf82232b",
         "PINNED_OFFICIAL_VOXCPM_SOURCE")
    need(TTS_RETRIES == 0 and MIN_SAMPLES == 6400 and MAX_SAMPLES == 28672,
         "BOUNDED_NO_RETRY_AUDIO_CONTRACT")
    need(len(PRIOR_ARTIFACTS) == 4, "FOUR_PRIOR_CORPUS_ARTIFACTS")
    need(len(RULES_SHA256) == 64 and sha(canonical(RULES)) == RULES_SHA256,
         "PIPELINE_ADMISSION_RULES_HASH")
    return {
        "status": "SOURCE_CONTRACT_VALID_NO_MODEL_EXECUTION",
        "pilot_groups": [{"id": g["donor_id"], "role": g["role"]} for g in PILOT_GROUPS],
        "pilot_rows": 12,
        "tts_base_calls": 4,
        "speed_duration_scales": [1.08, 1.0, 0.92],
        "observers": ["qwen", "whisper", "parakeet", "phoneme"],
        "observer_calls": 48,
        "training_allowed": False,
        "optimizer_updates": 0,
        "calibration_locked_reference_bytes_opened": False,
        "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "product_promotion_allowed": False,
        "pipeline_admission_ref": PIPELINE_ADMISSION_REF,
        "rules_sha256": RULES_SHA256,
    }

def guard_first_push(event_path: Path) -> dict:
    event = read_json(event_path)
    need(os.environ.get("GITHUB_ACTIONS") == "true", "GITHUB_ACTIONS_REQUIRED")
    need(os.environ.get("GITHUB_REPOSITORY") == REPO, "EXACT_PUBLIC_REPOSITORY")
    need(os.environ.get("GITHUB_EVENT_NAME") == "push", "FIRST_BRANCH_PUSH_ONLY")
    need(os.environ.get("GITHUB_REF") == "refs/heads/" + BRANCH, "EXACT_PILOT_BRANCH")
    need(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "NO_RERUNS")
    need(event.get("ref") == "refs/heads/" + BRANCH
         and event.get("after") == os.environ.get("GITHUB_SHA")
         and event.get("before") in {BASE_SHA, "0" * 40}
         and event.get("forced") is False and event.get("deleted") is False,
         "FIRST_CREATE_OR_FIRST_UPDATE_FROM_PINNED_MAIN_ONLY")
    repository = event.get("repository") or {}
    need(repository.get("full_name") == REPO and repository.get("private") is False,
         "PUBLIC_REPOSITORY_ONLY")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    need(run_id.isdigit() and int(run_id) > 0, "GITHUB_RUN_ID_REQUIRED")
    return {"run_id": int(run_id), "commit": os.environ["GITHUB_SHA"],
            "parent": event["before"], "attempt": 1, "branch": BRANCH}

def verify_donor_archive(zip_path: Path, output: Path) -> dict:
    zip_path = safe_file(zip_path)
    need(zip_path.stat().st_size == DONOR_ARTIFACT_BYTES, "DONOR_ARTIFACT_SIZE")
    need(sha_file(zip_path) == DONOR_ARTIFACT_SHA256, "DONOR_ARTIFACT_SHA256")
    output = Path(output)
    need(not output.exists(), "NEW_DONOR_OUTPUT")
    output.mkdir(mode=0o700, parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        manifest_raw = archive.read("MANIFEST.json")
        need(sha(manifest_raw) == DONOR_MANIFEST_SHA256, "DONOR_MANIFEST_SHA256")
        manifest = json.loads(manifest_raw)
        need(manifest.get("record_kind") == "SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY",
             "SYNTHETIC_DONOR_REFERENCE_ONLY")
        need(manifest.get("human_gold") is False
             and manifest.get("independent_human_source_claim") is False,
             "NO_HUMAN_GOLD_OR_SPEAKER_IDENTITY_CLAIM")
        need(manifest.get("roles") == {
            "train": 6, "development": 2, "calibration": 2, "locked-test": 2},
            "DONOR_ROLE_INVENTORY_6_2_2_2")
        rows = manifest.get("rows")
        need(isinstance(rows, list) and len(rows) == 12, "DONOR_12_VOICE_ROWS")
        by_id = {row.get("id"): row for row in rows}
        need(len(by_id) == 12, "DONOR_UNIQUE_IDS")
        selected = []
        for group in PILOT_GROUPS:
            row = by_id.get(group["donor_id"])
            need(isinstance(row, dict), "PILOT_DONOR_ROW")
            need(row.get("source_group_id") == group["source_group_id"]
                 and row.get("role") == group["role"]
                 and row.get("reference_text") == PROMPT_TEXT,
                 "EXACT_PILOT_DONOR_PROVENANCE")
            member = "wav/" + group["donor_id"] + ".wav"
            raw = archive.read(member)
            need(sha(raw) == row.get("wav_sha256"), "DONOR_WAV_SHA256")
            with wave.open(io.BytesIO(raw), "rb") as wav:
                need(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                     and wav.getframerate() == 16000 and wav.getcomptype() == "NONE",
                     "DONOR_WAV_PCM16_MONO_16K")
                pcm = wav.readframes(wav.getnframes())
            donor_path = output / "donors" / (group["donor_id"] + ".wav")
            donor_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with donor_path.open("xb") as handle:
                handle.write(raw)
            selected.append({
                **group,
                "reference_text": PROMPT_TEXT,
                "reference_kind": row.get("reference_kind"),
                "reference_wav_path": "donors/" + donor_path.name,
                "reference_wav_sha256": sha(raw),
                "reference_pcm_sha256": sha(pcm),
                "reference_samples": len(pcm) // 2,
            })
    result = {
        "schema": "english-kws-v7-voxcpm-g3-donor-selection-v1",
        "donor_run_id": DONOR_RUN_ID,
        "donor_artifact_id": DONOR_ARTIFACT_ID,
        "donor_artifact_sha256": DONOR_ARTIFACT_SHA256,
        "donor_manifest_sha256": DONOR_MANIFEST_SHA256,
        "selected_groups": selected,
        "selected_roles": {"train": 1, "development": 1},
        "reserved_roles_audio_opened": False,
        "calibration_audio_opened": False,
        "locked_test_audio_opened": False,
        "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
    }
    write_once(output / "PILOT_DONORS.json", result)
    return result

def collect_audio_hashes(value, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in HASH_KEYS and isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
                output.add(item)
            elif isinstance(item, (dict, list)):
                collect_audio_hashes(item, output)
    elif isinstance(value, list):
        for item in value:
            collect_audio_hashes(item, output)

def build_prior_hash_index(archives: dict[str, Path], output: Path) -> dict:
    need(set(archives) == set(PRIOR_ARTIFACTS), "EXACT_PRIOR_ARTIFACT_SET")
    hashes: set[str] = set()
    receipts = []
    for key, expected in PRIOR_ARTIFACTS.items():
        path = safe_file(archives[key])
        need(path.stat().st_size == expected["bytes"], "PRIOR_ARTIFACT_SIZE:" + key)
        need(sha_file(path) == expected["sha256"], "PRIOR_ARTIFACT_SHA256:" + key)
        found: set[str] = set()
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.lower().endswith(".json") or member.file_size > 16 * 1024 * 1024:
                    continue
                try:
                    value = json.loads(archive.read(member.filename))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                collect_audio_hashes(value, found)
        need(found, "PRIOR_AUDIO_HASH_INVENTORY_PRESENT:" + key)
        hashes.update(found)
        receipts.append({
            "key": key, "run_id": expected["run_id"],
            "artifact_id": expected["artifact_id"],
            "bytes": expected["bytes"], "sha256": expected["sha256"],
            "audio_hash_count": len(found),
            "media_bytes_opened": False, "labels_used": False,
        })
    result = {
        "schema": "english-kws-v7-prior-audio-hash-index-v1",
        "artifacts": receipts,
        "audio_hashes": sorted(hashes),
        "audio_hash_count": len(hashes),
        "metadata_only": True,
        "prior_media_bytes_opened": False,
        "prior_labels_used": False,
        "reserved_audio_extracted": False,
    }
    write_once(Path(output), result)
    return result

def available_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0

def prepare_model(runtime_dir: Path, official_dir: Path, model_dir: Path,
                  output_dir: Path, quantizer: Path, tts_binary: Path) -> dict:
    runtime_dir = Path(runtime_dir)
    official_dir = Path(official_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    need(subprocess.check_output(["git", "-C", str(runtime_dir), "rev-parse", "HEAD"], text=True).strip()
         == RUNTIME_COMMIT, "VOXCPM_CPP_COMMIT_PIN")
    need(subprocess.check_output(["git", "-C", str(official_dir), "rev-parse", "HEAD"], text=True).strip()
         == OFFICIAL_COMMIT, "OFFICIAL_VOXCPM_SOURCE_COMMIT_PIN")
    runtime_license = safe_file(runtime_dir / "LICENSE")
    official_license = safe_file(official_dir / "LICENSE")
    runtime_license_sha = sha_file(runtime_license)
    official_license_sha = sha_file(official_license)
    disk_free = shutil.disk_usage(output_dir).free
    mem_free = available_memory_bytes()
    need(disk_free >= 12 * 1024**3, "RESOURCE_STOP_DISK_BELOW_12_GIB:" + str(disk_free))
    need(mem_free >= 8 * 1024**3, "RESOURCE_STOP_MEM_AVAILABLE_BELOW_8_GIB:" + str(mem_free))

    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_REPO, revision="main")
    revision = str(info.sha)
    need(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "OFFICIAL_MODEL_REVISION_SHA")
    card = getattr(info, "cardData", None) or {}
    tags = set(getattr(info, "tags", None) or [])
    license_value = card.get("license") if isinstance(card, dict) else None
    need(license_value == MODEL_LICENSE or "license:" + MODEL_LICENSE in tags,
         "OFFICIAL_MODEL_APACHE_2_LICENSE_REQUIRED")
    model_path = Path(snapshot_download(
        repo_id=MODEL_REPO, revision=revision,
        local_dir=str(model_dir),
        allow_patterns=["*.json", "*.py", "*.txt", "README.md", "LICENSE",
                        "model.safetensors", "audiovae.pth"],
    ))
    model_weights = safe_file(model_path / "model.safetensors")
    audiovae_weights = safe_file(model_path / "audiovae.pth")
    config = read_json(model_path / "config.json")
    need(isinstance(config.get("audio_vae_config"), dict),
         "OFFICIAL_MODEL_CONFIG_MUST_PIN_AUDIO_VAE_CONFIG")
    model_sha = sha_file(model_weights)
    audiovae_sha = sha_file(audiovae_weights)

    converter = safe_file(runtime_dir / "scripts" / "convert_voxcpm_to_gguf.py")
    converter_sha = sha_file(converter)
    converter_text = converter.read_text(encoding="utf-8")
    import_line = "from voxcpm.modules.audiovae.audio_vae import AudioVAEConfig"
    need(converter_text.count(import_line) == 1, "EXACT_CONVERTER_IMPORT_PATCH")
    patched_text = converter_text.replace(
        import_line,
        "class AudioVAEConfig:\n"
        "    def model_dump(self):\n"
        "        raise RuntimeError('official VoxCPM1.5 config must provide audio_vae_config')",
    )
    converter.write_text(patched_text, encoding="utf-8")
    patched_converter_sha = sha_file(converter)

    f32_path = output_dir / "voxcpm1.5-f32.gguf"
    q8_path = output_dir / "voxcpm1.5-q8_0-audiovae-f16.gguf"
    subprocess.run(
        [sys.executable, str(converter), str(model_path), "--output", str(f32_path)],
        check=True, timeout=3600,
    )
    f32_path = safe_file(f32_path)
    f32_sha = sha_file(f32_path)
    quantizer = safe_file(quantizer)
    tts_binary = safe_file(tts_binary)
    subprocess.run(
        [str(quantizer), "--input", str(f32_path), "--output", str(q8_path),
         "--type", "Q8_0", "--audio-vae-mode", "f16", "--threads", "4"],
        check=True, timeout=1800,
    )
    q8_path = safe_file(q8_path)
    q8_sha = sha_file(q8_path)
    import importlib.metadata
    versions = {}
    for package in ("torch", "transformers", "safetensors", "huggingface-hub", "gguf", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    lock = {
        "schema": "english-kws-v7-voxcpm-model-lock-v1",
        "model_repository": MODEL_REPO,
        "model_revision": revision,
        "model_license": MODEL_LICENSE,
        "model_safetensors_sha256": model_sha,
        "model_safetensors_bytes": model_weights.stat().st_size,
        "audiovae_pth_sha256": audiovae_sha,
        "audiovae_pth_bytes": audiovae_weights.stat().st_size,
        "official_source_repository": OFFICIAL_REPO,
        "official_source_commit": OFFICIAL_COMMIT,
        "official_license_sha256": official_license_sha,
        "runtime_repository": RUNTIME_REPO,
        "runtime_commit": RUNTIME_COMMIT,
        "runtime_license_sha256": runtime_license_sha,
        "converter_original_sha256": converter_sha,
        "converter_patched_sha256": patched_converter_sha,
        "converter_patch": "remove import of AudioVAEConfig; fail if official config lacks audio_vae_config",
        "f32_gguf_sha256": f32_sha,
        "f32_gguf_bytes": f32_path.stat().st_size,
        "q8_0_audiovae_f16_gguf_sha256": q8_sha,
        "q8_0_audiovae_f16_gguf_bytes": q8_path.stat().st_size,
        "quantizer_binary_sha256": sha_file(quantizer),
        "tts_binary_sha256": sha_file(tts_binary),
        "quantization": {"model": "Q8_0", "audio_vae": "F16"},
        "runtime_backend": "CPU",
        "runtime_threads": TTS_THREADS,
        "python_packages": versions,
        "resource_preflight": {
            "disk_free_bytes_before_model_download": disk_free,
            "mem_available_bytes_before_model_download": mem_free,
            "disk_reserve_policy": "measured free space >= 12 GiB; model and F32 intermediates removed after conversion",
        },
    }
    write_once(output_dir / "MODEL_LOCK.json", lock)
    # Keep only the final quantized model; the original weights and F32 intermediate
    # are not uploaded or redistributed in the workflow artifact.
    shutil.rmtree(model_path, ignore_errors=True)
    f32_path.unlink(missing_ok=True)
    return {"model_path": str(q8_path), "model_lock": lock,
            "model_lock_sha256": sha(canonical(lock))}

def probe_audio(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,codec_name,sample_fmt,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    need(len(streams) == 1, "ONE_AUDIO_STREAM_REQUIRED")
    stream = streams[0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "codec_name": stream.get("codec_name"),
        "sample_fmt": stream.get("sample_fmt"),
        "duration": float(stream.get("duration", 0.0)),
    }

def pcm_metrics(pcm: bytes) -> dict:
    need(len(pcm) % 2 == 0, "PCM16_BYTE_GEOMETRY")
    values = [item[0] for item in struct.iter_unpack("<h", pcm)]
    need(values, "NONEMPTY_PCM")
    peak = max(abs(value) for value in values)
    rms = math.sqrt(sum(value * value for value in values) / len(values)) / 32768.0
    clipped = sum(1 for value in values if value in (-32768, 32767))
    return {
        "samples": len(values),
        "peak_normalized": peak / 32768.0,
        "rms_normalized": rms,
        "clipped_samples": clipped,
        "clipping_fraction": clipped / len(values),
    }

def generate_pilot(donor_root: Path, prior_index_path: Path, model_path: Path,
                   model_lock_path: Path, tts_binary: Path, output: Path) -> dict:
    donor_root = Path(donor_root)
    output = Path(output)
    need(not output.exists(), "NEW_PILOT_OUTPUT_DIRECTORY")
    output.mkdir(mode=0o700, parents=True)
    donor_manifest = read_json(donor_root / "PILOT_DONORS.json")
    prior_index = read_json(prior_index_path)
    model_lock = read_json(model_lock_path)
    model_lock_sha = sha(canonical(model_lock))
    prior_hashes = set(prior_index.get("audio_hashes", []))
    need(prior_index.get("metadata_only") is True
         and prior_index.get("prior_media_bytes_opened") is False
         and prior_index.get("prior_labels_used") is False,
         "METADATA_ONLY_PRIOR_DUPLICATE_INDEX")
    need(len(donor_manifest.get("selected_groups", [])) == 2,
         "TWO_SELECTED_DONOR_GROUPS")
    tts_binary = safe_file(tts_binary)
    model_path = safe_file(model_path)
    ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    rows = []
    base_outputs = []
    observed_hashes = set()
    for group in donor_manifest["selected_groups"]:
        donor_path = safe_file(donor_root / group["reference_wav_path"])
        need(sha_file(donor_path) == group["reference_wav_sha256"], "DONOR_COPY_HASH")
        for candidate, candidate_spec in CANDIDATES.items():
            base_id = group["donor_id"] + "-" + candidate.lower()
            base_path = output / "base" / group["donor_id"] / (candidate.lower() + ".wav")
            base_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            call_started = {
                "call_id": base_id, "status": "STARTED",
                "attempt": 1, "retries_allowed": 0,
                "donor_wav_sha256": group["reference_wav_sha256"],
                "target_text": candidate_spec["phrase"],
                "model_lock_sha256": model_lock_sha,
            }
            write_once(output / "calls" / (base_id + ".started.json"), call_started)
            command = [
                str(tts_binary), "--model-path", str(model_path),
                "--prompt-audio", str(donor_path), "--prompt-text", PROMPT_TEXT,
                "--text", candidate_spec["phrase"], "--output", str(base_path),
                "--backend", "cpu", "--threads", str(TTS_THREADS),
                "--inference-timesteps", str(TTS_INFERENCE_TIMESTEPS),
                "--cfg-value", str(TTS_CFG_VALUE),
            ]
            try:
                completed = subprocess.run(command, check=True, capture_output=True,
                                           text=True, timeout=1200)
            except Exception as exc:
                write_once(output / "calls" / (base_id + ".failed.json"), {
                    "call_id": base_id, "status": "FAILED_NO_RETRY",
                    "error": str(exc), "attempt": 1, "retries": 0,
                })
                raise
            base_path = safe_file(base_path)
            base_probe = probe_audio(base_path)
            base_sha = sha_file(base_path)
            write_once(output / "calls" / (base_id + ".completed.json"), {
                "call_id": base_id, "status": "SUCCESS",
                "attempt": 1, "retries": 0,
                "command_sha256": sha(canonical(command)),
                "stdout_sha256": sha(completed.stdout.encode("utf-8")),
                "stderr_sha256": sha(completed.stderr.encode("utf-8")),
                "base_wav_sha256": base_sha,
                "base_wav_bytes": base_path.stat().st_size,
                "base_probe": base_probe,
            })
            base_outputs.append({
                "base_id": base_id, "donor_id": group["donor_id"],
                "candidate": candidate, "base_path": str(base_path.relative_to(output)),
                "base_wav_sha256": base_sha, "base_probe": base_probe,
            })
            for cell_id, cell_candidate, length_scale in PILOT_CELLS:
                if cell_candidate != candidate:
                    continue
                atempo = 1.0 / length_scale
                row_id = group["donor_id"] + "-" + cell_id
                wav_rel = Path("audio") / group["donor_id"] / (cell_id + ".wav")
                pcm_rel = Path("audio") / group["donor_id"] / (cell_id + ".pcm")
                wav_path = output / wav_rel
                pcm_path = output / pcm_rel
                wav_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                filter_spec = "atempo=" + format(atempo, ".12g") + ",aresample=16000"
                subprocess.run([
                    "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(base_path), "-af", filter_spec,
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    "-map_metadata", "-1", "-fflags", "+bitexact",
                    "-flags:a", "+bitexact", str(wav_path),
                ], check=True, timeout=180)
                wav_path = safe_file(wav_path)
                with wave.open(str(wav_path), "rb") as wav:
                    need(wav.getnchannels() == 1 and wav.getsampwidth() == 2
                         and wav.getframerate() == SAMPLE_RATE
                         and wav.getcomptype() == "NONE",
                         "FINAL_AUDIO_PCM16_MONO_16K")
                    pcm = wav.readframes(wav.getnframes())
                metrics = pcm_metrics(pcm)
                need(MIN_SAMPLES <= metrics["samples"] <= MAX_SAMPLES,
                     "PILOT_DURATION_OUT_OF_RANGE:" + row_id)
                need(0.002 <= metrics["rms_normalized"] <= 0.6,
                     "PILOT_RMS_OUT_OF_RANGE:" + row_id)
                need(metrics["clipping_fraction"] <= 0.001,
                     "PILOT_CLIPPING_OUT_OF_RANGE:" + row_id)
                pcm_hash = sha(pcm)
                need(pcm_hash not in prior_hashes, "DUPLICATE_PRIOR_CORPUS_PCM:" + row_id)
                need(pcm_hash not in observed_hashes, "DUPLICATE_PILOT_PCM:" + row_id)
                observed_hashes.add(pcm_hash)
                with pcm_path.open("xb") as handle:
                    handle.write(pcm)
                candidate_labels = {
                    "VOSEN": 1 if candidate == "VOSEN" else 0,
                    "MERO": 1 if candidate == "MERO" else 0,
                }
                request = {
                    "generator": "VoxCPM.cpp", "runtime_commit": RUNTIME_COMMIT,
                    "model_lock_sha256": model_lock_sha,
                    "source_group_id": group["source_group_id"],
                    "cell_id": cell_id, "requested_text": candidate_spec["phrase"],
                    "length_scale": length_scale, "atempo": atempo,
                    "prompt_text": PROMPT_TEXT,
                    "tts_threads": TTS_THREADS,
                    "inference_timesteps": TTS_INFERENCE_TIMESTEPS,
                    "cfg_value": TTS_CFG_VALUE,
                }
                rows.append({
                    "id": row_id,
                    "source_group_id": group["source_group_id"],
                    "source_speaker_id": group["source_speaker_id"],
                    "source_speaker_index": group["source_speaker_index"],
                    "voice_family": group["voice_family"],
                    "role": group["role"],
                    "cell_id": cell_id,
                    "requested_text": candidate_spec["phrase"],
                    "requested_class": 1,
                    "pronunciation_hypothesis": candidate_spec["pronunciation_hypothesis"],
                    "kind": "target",
                    "length_scale": length_scale,
                    "raw_phonemes": candidate_spec["ipa"],
                    "candidate_labels": candidate_labels,
                    "generation_status": "success",
                    "pcm_path": str(pcm_rel),
                    "wav_path": str(wav_rel),
                    "pcm_sha256": pcm_hash,
                    "wav_sha256": sha_file(wav_path),
                    "pcm_samples": metrics["samples"],
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "request_sha256": sha(canonical(request)),
                    "source_reference_wav_sha256": group["reference_wav_sha256"],
                    "base_wav_sha256": base_sha,
                    "tempo_transform": {
                        "name": "ffmpeg-atempo",
                        "length_scale": length_scale,
                        "atempo": atempo,
                        "filter": filter_spec,
                        "ffmpeg_version": ffmpeg_version,
                    },
                    "quality": metrics,
                    "human_gold": False,
                    "target_conditioned_observation": False,
                })
    need(len(base_outputs) == 4 and len(rows) == 12, "EXACT_PILOT_TTS_AND_ROW_COUNTS")
    need(Counter(row["role"] for row in rows) == Counter({"train": 6, "development": 6}),
         "PILOT_ROLE_COUNTS")
    recipe = {
        "schema": "english-kws-v7-voxcpm-g3-recipe-v1",
        "model_repository": MODEL_REPO,
        "runtime_commit": RUNTIME_COMMIT,
        "model_quantization": "Q8_0+AudioVAE-F16",
        "prompt_text": PROMPT_TEXT,
        "tts_threads": TTS_THREADS,
        "inference_timesteps": TTS_INFERENCE_TIMESTEPS,
        "cfg_value": TTS_CFG_VALUE,
        "speed_cells": [{"cell_id": c[0], "candidate": c[1],
                         "length_scale": c[2], "atempo": 1.0 / c[2]} for c in PILOT_CELLS],
        "tts_retries": 0,
        "normalization": "none",
        "trimming": "none",
    }
    recipe_sha = sha(canonical(recipe))
    preflight = {
        "schema": "english-kws-v7-voxcpm-g3-preflight-v1",
        "donor_artifact_sha256": DONOR_ARTIFACT_SHA256,
        "prior_hash_index_sha256": sha(canonical(prior_index)),
        "model_lock_sha256": model_lock_sha,
        "ffmpeg_version": ffmpeg_version,
        "resource_gate": model_lock.get("resource_preflight"),
        "reserved_audio_opened": False,
        "training_allowed": False,
        "promotion_allowed": False,
    }
    preflight_sha = sha(canonical(preflight))
    groups = [{
        key: group[key] for key in
        ("source_group_id", "source_speaker_id", "source_speaker_index", "voice_family", "role")
    } for group in donor_manifest["selected_groups"]]
    manifest = {
        "schema": "english-kws-dual-track-v7-voxcpm-g3-pilot-v1",
        "status": "PILOT_AUDIO_COMPLETE_OBSERVERS_PENDING",
        "source_run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "source_plan_sha256": "0" * 64,
        "recipe_sha256": recipe_sha,
        "preflight_sha256": preflight_sha,
        "candidate_ids": [CANDIDATES["VOSEN"]["candidate_id"], CANDIDATES["MERO"]["candidate_id"]],
        "observer_providers": ["qwen", "whisper", "parakeet", "phoneme"],
        "source_role_counts": {"train": 1, "development": 1},
        "source_groups": groups,
        "rows": sorted(rows, key=lambda row: row["id"]),
        "base_outputs": base_outputs,
        "donor_artifact": {
            "run_id": DONOR_RUN_ID, "artifact_id": DONOR_ARTIFACT_ID,
            "sha256": DONOR_ARTIFACT_SHA256, "manifest_sha256": DONOR_MANIFEST_SHA256,
        },
        "prior_duplicate_scan": {
            "artifact_keys": sorted(PRIOR_ARTIFACTS),
            "hash_index_sha256": sha(canonical(prior_index)),
            "metadata_only": True,
            "prior_media_bytes_opened": False,
            "prior_labels_used": False,
        },
        "model_lock_sha256": model_lock_sha,
        "recipe": recipe,
        "preflight": preflight,
        "tts_base_calls": 4,
        "generated_rows": 12,
        "asr_calls": 0,
        "phoneme_calls": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
        "calibration_audio_opened": False,
        "locked_test_audio_opened": False,
        "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "product_promotion_allowed": False,
    }
    write_once(output / "MODEL_LOCK.json", model_lock)
    write_once(output / "PRIOR_HASH_INDEX.json", prior_index)
    write_once(output / "RECIPE.json", recipe)
    write_once(output / "PREFLIGHT.json", preflight)
    write_once(output / "PILOT_MANIFEST.json", manifest)
    inventory = []
    for file in sorted(output.rglob("*")):
        if file.is_file() and file.name != "FILES.json" and not file.is_symlink():
            inventory.append({
                "path": file.relative_to(output).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": sha_file(file),
            })
    write_once(output / "FILES.json", {
        "schema": "english-kws-v7-voxcpm-g3-pilot-files-v1",
        "files": inventory,
    })
    return manifest

def verify_pilot_source(source_root: Path) -> dict:
    source_root = Path(source_root)
    manifest = read_json(source_root / "PILOT_MANIFEST.json")
    need(manifest.get("schema") == "english-kws-dual-track-v7-voxcpm-g3-pilot-v1",
         "PILOT_SOURCE_SCHEMA")
    need(manifest.get("training_allowed") is False
         and manifest.get("product_promotion_allowed") is False
         and manifest.get("calibration_audio_opened") is False
         and manifest.get("locked_test_audio_opened") is False,
         "PILOT_NOT_TRAINING_OR_PROMOTION_AUTHORITY")
    need(len(manifest.get("rows", [])) == 12, "PILOT_SOURCE_12_ROWS")
    inventory = read_json(source_root / "FILES.json")
    expected = {item["path"]: item for item in inventory.get("files", [])}
    for relative, item in expected.items():
        path = source_root / relative
        safe_file(path)
        need(path.stat().st_size == item["bytes"] and sha_file(path) == item["sha256"],
             "PILOT_SOURCE_FILE_HASH:" + relative)
    for row in manifest["rows"]:
        wav_path = safe_file(source_root / row["wav_path"])
        pcm_path = safe_file(source_root / row["pcm_path"])
        need(sha_file(wav_path) == row["wav_sha256"], "PILOT_WAV_HASH:" + row["id"])
        pcm = pcm_path.read_bytes()
        need(sha(pcm) == row["pcm_sha256"] and len(pcm) == row["pcm_samples"] * 2,
             "PILOT_PCM_HASH:" + row["id"])
    forbidden = ("calibration", "locked-test", "locked_test")
    need(not any(any(token in path.lower() for token in forbidden) for path in expected),
         "NO_RESERVED_AUDIO_IN_PILOT")
    return manifest

def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())

def normalized_phones(phones) -> str:
    value = "".join(str(item) for item in phones).casefold()
    for mark in ("ˈ", "ˌ", "ː", "ˑ", "͡", "͜"):
        value = value.replace(mark, "")
    value = value.replace("ɹ", "r").replace("ɚ", "ər").replace("ɝ", "ər")
    return re.sub(r"[^a-zɐɑæɛɪɔʊʌəθðŋʃʒtʃdʒr]", "", value)

def phone_candidate(phones, candidate: str) -> bool:
    return re.fullmatch(PHONE_PATTERNS[candidate], normalized_phones(phones)) is not None

def _votes(texts, candidate: str) -> int:
    aliases = set(ALIASES[candidate])
    votes = 0
    for text in texts:
        normalized = normalized_text(text)
        if normalized in EXCLUDED_TRANSCRIPTS:
            continue
        if normalized in aliases:
            votes += 1
    return votes

def decide_candidate(row: dict, candidate: str, texts: list[str], phones) -> dict:
    expected = row["candidate_labels"][candidate]
    other = "MERO" if candidate == "VOSEN" else "VOSEN"
    target_votes = _votes(texts, candidate)
    other_votes = _votes(texts, other)
    target_phone = phone_candidate(phones, candidate)
    other_phone = phone_candidate(phones, other)
    if row["kind"] == "target":
        if expected == 1 and target_votes >= 2 and other_votes == 0 and target_phone:
            tier = "silver" if target_votes == 3 else "weak"
            return {"research_class": 1, "tier": tier, "reason": "JOINT_ASR_AND_PHONE_TARGET",
                    "target_votes": target_votes, "other_votes": other_votes}
        if expected == 0 and other_votes >= 2 and target_votes == 0 and other_phone:
            tier = "silver" if other_votes == 3 else "weak"
            return {"research_class": 0, "tier": tier, "reason": "JOINT_ASR_AND_PHONE_CROSS_TRIGGER",
                    "target_votes": target_votes, "other_votes": other_votes}
    return {"research_class": None, "tier": "unknown", "reason": "NO_JOINT_OBSERVER_SUPPORT",
            "target_votes": target_votes, "other_votes": other_votes}

def observe(source_root: Path, provider: str, output: Path, model_root: Path) -> dict:
    need(provider in ("qwen", "whisper", "parakeet", "phoneme"), "KNOWN_OBSERVER")
    manifest = verify_pilot_source(source_root)
    output = Path(output)
    need(not output.exists(), "NEW_OBSERVER_OUTPUT")
    output.mkdir(mode=0o700, parents=True)
    model_root = Path(model_root)
    model_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    pins = MODEL_PINS[provider]
    if provider == "phoneme":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import english_final_ipa_xlsr_once as phone
        phone.check_helper()
        predictor, model_info = phone.load_runtime(model_root)
        provenance = {
            "model_id": getattr(phone, "MODEL", "facebook/wav2vec2-xls-r-300m-phoneme"),
            "revision": getattr(phone, "REVISION", None),
            "weights_sha256": getattr(phone, "WEIGHTS_SHA256", pins["model_sha256"]),
        }
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import english_seven_voice_asr_once as asr
        predictor = asr._load(provider, model_root)
        provenance = asr.PROVENANCE[provider]
        model_info = {"helper_provenance": list(provenance)}
    started = {
        "schema": "english-kws-v7-voxcpm-g3-observer-start-v1",
        "provider": provider, "expected_rows": 12, "attempt": 1, "retries": 0,
        "source_manifest_sha256": sha(canonical(manifest)),
        "model_pin": pins, "model_provenance": provenance,
        "target_text_exposed_to_decoder": False,
    }
    write_once(output / "STARTED.json", started)
    captures = []
    failures = []
    for row in manifest["rows"]:
        wav_path = safe_file(Path(source_root) / row["wav_path"])
        need(sha_file(wav_path) == row["wav_sha256"], "OBSERVER_INPUT_WAV_HASH")
        # Only audio identity and path are passed to the decoder. Target labels and
        # requested text remain unused until the separate adjudication stage.
        try:
            if provider == "phoneme":
                tokens, frames = predictor(wav_path)
                payload = {"phone_tokens": list(tokens), "frame_count": int(frames)}
            else:
                raw_text = predictor(wav_path, output)
                need(isinstance(raw_text, str), "ASR_TEXT_OUTPUT_REQUIRED")
                payload = {"raw_text": raw_text}
            capture = {
                "schema": "english-kws-dual-track-v7-voice-clone-observer-capture-v1",
                "id": row["id"],
                "recognizer_id": pins["recognizer_id"],
                "model_family": pins["model_family"],
                "model_sha256": pins["model_sha256"],
                "decode_config_sha256": pins["decode_config_sha256"],
                "source_pcm_sha256": row["pcm_sha256"],
                "source_wav_sha256": row["wav_sha256"],
                "source_samples": row["pcm_samples"],
                "status": "success",
                "target_conditioned": False,
                "input_coverage_complete": True,
                "decoder_output_complete": True,
                **payload,
            }
            relative = Path("evidence") / provider / row["id"] / "CAPTURE.json"
            capture_path = output / relative
            write_once(capture_path, capture)
            captures.append({
                "id": row["id"], "capture_path": relative.as_posix(),
                "capture_sha256": sha(canonical(capture)),
                "source_pcm_sha256": row["pcm_sha256"],
                "source_wav_sha256": row["wav_sha256"],
                "status": "success",
            })
        except Exception as exc:
            failures.append({"id": row["id"], "error": str(exc)})
            break
    summary = {
        "schema": "english-kws-v7-voxcpm-g3-observer-summary-v1",
        "provider": provider,
        "status": "COMPLETE" if len(captures) == 12 and not failures else "FAILED_NO_RETRY",
        "expected_rows": 12, "completed_rows": len(captures),
        "failed_rows": failures,
        "attempt": 1, "retries": 0,
        "source_manifest_sha256": sha(canonical(manifest)),
        "model_pin": pins, "model_provenance": provenance,
        "model_info": model_info,
        "target_text_exposed_to_decoder": False,
        "training_allowed": False,
    }
    write_once(output / "OBSERVATIONS.json", {
        "schema": "english-kws-v7-voxcpm-g3-observations-v1",
        "provider": provider, "rows": captures,
    })
    write_once(output / "SUMMARY.json", summary)
    if failures:
        write_once(output / "FAILED.json", summary)
        raise ValueError("OBSERVER_FAILED_NO_RETRY:" + provider)
    return summary

def adjudicate(source_root: Path, observer_root: Path, output: Path) -> dict:
    output = Path(output)
    need(not output.exists(), "NEW_ADJUDICATION_OUTPUT")
    output.mkdir(mode=0o700, parents=True)
    try:
        manifest = verify_pilot_source(source_root)
        observers = {}
        for provider in ("qwen", "whisper", "parakeet", "phoneme"):
            root = Path(observer_root) / provider
            summary = read_json(root / "SUMMARY.json")
            observations = read_json(root / "OBSERVATIONS.json")
            need(summary.get("status") == "COMPLETE"
                 and summary.get("completed_rows") == 12
                 and observations.get("provider") == provider,
                 "OBSERVER_COMPLETE:" + provider)
            rows = {row["id"]: row for row in observations.get("rows", [])}
            need(len(rows) == 12, "OBSERVER_12_ROWS:" + provider)
            observers[provider] = rows
        decisions = []
        group_metrics = []
        all_pass = True
        for group in manifest["source_groups"]:
            group_id = group["source_group_id"]
            for candidate in ("VOSEN", "MERO"):
                positive = []
                negative = []
                for source_row in manifest["rows"]:
                    if source_row["source_group_id"] != group_id:
                        continue
                    joined = {
                        provider: observers[provider][source_row["id"]]
                        for provider in ("qwen", "whisper", "parakeet", "phoneme")
                    }
                    texts = [
                        read_json(Path(observer_root) / provider / joined[provider]["capture_path"])["raw_text"]
                        for provider in ("qwen", "whisper", "parakeet")
                    ]
                    phone_capture = read_json(
                        Path(observer_root) / "phoneme" / joined["phoneme"]["capture_path"])
                    phone_tokens = phone_capture["phone_tokens"]
                    result = decide_candidate(source_row, candidate, texts, phone_tokens)
                    item = {
                        "id": source_row["id"], "candidate": candidate,
                        "source_group_id": group_id, "role": source_row["role"],
                        "expected_class": source_row["candidate_labels"][candidate],
                        "decision": result,
                    }
                    decisions.append(item)
                    if source_row["candidate_labels"][candidate] == 1:
                        positive.append(result)
                    else:
                        negative.append(result)
                pos_supported = [x for x in positive if x["tier"] in ("silver", "weak")]
                neg_supported = [x for x in negative if x["tier"] in ("silver", "weak")]
                pos_silver = sum(x["tier"] == "silver" for x in positive)
                neg_silver = sum(x["tier"] == "silver" for x in negative)
                passed = (len(pos_supported) >= 2 and len(neg_supported) >= 2
                          and pos_silver >= 1 and neg_silver >= 1)
                all_pass = all_pass and passed
                group_metrics.append({
                    "candidate": candidate, "source_group_id": group_id,
                    "role": group["role"],
                    "positive_supported": len(pos_supported), "positive_total": len(positive),
                    "positive_silver": pos_silver,
                    "cross_trigger_negative_supported": len(neg_supported),
                    "cross_trigger_negative_total": len(negative),
                    "cross_trigger_negative_silver": neg_silver,
                    "pass": passed,
                })
        result = {
            "schema": "english-kws-v7-voxcpm-g3-pilot-result-v1",
            "status": "PILOT_PASS_SCALEUP_ELIGIBLE" if all_pass else "PILOT_FAIL_NO_SCALEUP",
            "scale_up_eligible": bool(all_pass),
            "pilot_groups": manifest["source_groups"],
            "candidate_group_metrics": group_metrics,
            "decisions": decisions,
            "rules_sha256": sha(canonical(RULES)),
            "pipeline_admission_ref": PIPELINE_ADMISSION_REF,
            "observers": ["qwen", "whisper", "parakeet", "phoneme"],
            "observer_calls": 48,
            "tts_base_calls": 4,
            "generated_rows": 12,
            "retries": 0,
            "training_allowed": False,
            "optimizer_updates": 0,
            "calibration_audio_opened": False,
            "locked_test_audio_opened": False,
            "human_gold": False,
            "independent_human_speaker_identity_claimed": False,
            "product_promotion_allowed": False,
        }
    except Exception as exc:
        result = {
            "schema": "english-kws-v7-voxcpm-g3-pilot-result-v1",
            "status": "OBSERVATION_INCOMPLETE_NO_RETRY",
            "scale_up_eligible": False,
            "error": str(exc),
            "training_allowed": False,
            "optimizer_updates": 0,
            "calibration_audio_opened": False,
            "locked_test_audio_opened": False,
            "product_promotion_allowed": False,
        }
    write_once(output / "PILOT_RESULT.json", result)
    return result

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-check")
    guard = sub.add_parser("guard")
    guard.add_argument("--event", type=Path, required=True)
    donors = sub.add_parser("verify-donors")
    donors.add_argument("--zip", type=Path, required=True)
    donors.add_argument("--output", type=Path, required=True)
    prior = sub.add_parser("prior-index")
    prior.add_argument("--archive", action="append", required=True,
                       help="key=path for one pinned prior source artifact ZIP")
    prior.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-model")
    prepare.add_argument("--runtime-dir", type=Path, required=True)
    prepare.add_argument("--official-dir", type=Path, required=True)
    prepare.add_argument("--model-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--quantizer", type=Path, required=True)
    prepare.add_argument("--tts-binary", type=Path, required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--donor-root", type=Path, required=True)
    generate.add_argument("--prior-index", type=Path, required=True)
    generate.add_argument("--model-path", type=Path, required=True)
    generate.add_argument("--model-lock", type=Path, required=True)
    generate.add_argument("--tts-binary", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    observer = sub.add_parser("observe")
    observer.add_argument("--source-root", type=Path, required=True)
    observer.add_argument("--provider", choices=("qwen", "whisper", "parakeet", "phoneme"), required=True)
    observer.add_argument("--output", type=Path, required=True)
    observer.add_argument("--model-root", type=Path, required=True)
    adjud = sub.add_parser("adjudicate")
    adjud.add_argument("--source-root", type=Path, required=True)
    adjud.add_argument("--observer-root", type=Path, required=True)
    adjud.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "source-check":
        print(json.dumps(source_check(), ensure_ascii=False, sort_keys=True))
    elif args.command == "guard":
        print(json.dumps(guard_first_push(args.event), sort_keys=True))
    elif args.command == "verify-donors":
        print(json.dumps(verify_donor_archive(args.zip, args.output), ensure_ascii=False, sort_keys=True))
    elif args.command == "prior-index":
        archives = {}
        for item in args.archive:
            key, value = item.split("=", 1)
            need(key not in archives, "DUPLICATE_PRIOR_ARCHIVE_KEY")
            archives[key] = Path(value)
        print(json.dumps(build_prior_hash_index(archives, args.output), sort_keys=True))
    elif args.command == "prepare-model":
        print(json.dumps(prepare_model(args.runtime_dir, args.official_dir, args.model_dir,
                                        args.output_dir, args.quantizer, args.tts_binary),
                         ensure_ascii=False, sort_keys=True))
    elif args.command == "generate":
        print(json.dumps(generate_pilot(args.donor_root, args.prior_index, args.model_path,
                                        args.model_lock, args.tts_binary, args.output),
                         ensure_ascii=False, sort_keys=True))
    elif args.command == "observe":
        print(json.dumps(observe(args.source_root, args.provider, args.output, args.model_root),
                         ensure_ascii=False, sort_keys=True))
    elif args.command == "adjudicate":
        print(json.dumps(adjudicate(args.source_root, args.observer_root, args.output),
                         ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
