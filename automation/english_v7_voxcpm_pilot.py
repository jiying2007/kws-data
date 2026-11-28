#!/usr/bin/env python3
"""Bounded VoxCPM1.5 V7 feasibility pilot; never admits training data.

The source stage creates four base utterances (two exact keywords x one TRAIN
and one DEVELOPMENT synthetic donor), then derives three deterministic tempo
variants per base. Four blind observers reuse the frozen Pocket12 runtimes.
The pilot is diagnostic only: it cannot enable training or promotion.
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
import shutil
import signal
import struct
import subprocess
import sys
import time
import wave
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = "jiying2007/kws-data"
MAIN_BEFORE = "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2"
PIPELINE_REPO = "jiying2007/kws-pipeline"
PIPELINE_COMMIT = "b2f342c6204c2dbdd2c2a29f91868642a021b228"

VOXCPM_CPP_REPO = "bluryar/VoxCPM.cpp"
VOXCPM_CPP_COMMIT = "e5b89534e379c79b923c9969b79c3858d140bedb"
HF_MODEL_REPO = "openbmb/VoxCPM1.5"
HF_MODEL_REVISION = "09c7a389438c27100dfa4e6cda1f8f0dd0d2fbfa"
VOXCPM_PYPI_VERSION = "1.5.0"
VOXCPM_PYPI_SDIST_SHA256 = "6b623812006d35f16d9dc93c3d1bd69a8440ae3970d0d2b8ff27954aae607f40"
GGUF_PYPI_VERSION = "0.17.1"
GGUF_PYPI_WHEEL_SHA256 = "7bc5aa7eeb1931f7d39b48fdc5b38fda6b294b9dca75cf607ac69557840a3943"

DONOR_ARTIFACT = {
    "id": 14406282094,
    "run_id": 44256271861,
    "name": "english-neutral-clone-reference12-44256271861",
    "bytes": 652901,
    "zip_sha256": "8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5",
    "manifest_sha256": "10b22f20cf8e012d0c5ec4c494674bd53fd9ee874cfa69fe68955482d64d7cf5",
    "schema": "english-synthetic-clone-reference-12-v1",
    "record_kind": "SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY",
    "internal_source": {
        "run": 44218734553,
        "artifact_id": 14389757284,
        "archive_sha256": "712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da",
        "head_sha": "7e9797789cba7d1476d02f14e21af91dc3cd0712",
    },
}
DONORS = {
    "af_bella": {
        "role": "train",
        "source_group_id": "kokoro-v1_0-preset:af_bella",
        "wav_path": "wav/synthetic-af_bella.wav",
        "wav_sha256": "cb8103bb68e8888b68c85780e7b3b4dfe2eac0d7f48022e102e0268abb9dea59",
        "pcm_sha256": "e1b55a910c28025545d7ebcac4e0f2b8c95e243ec9376a9d989997ed57ad7985",
        "pcm_samples": 52400,
        "id": "synthetic-af_bella",
    },
    "af_aoede": {
        "role": "development",
        "source_group_id": "kokoro-v1_0-preset:af_aoede",
        "wav_path": "wav/synthetic-af_aoede.wav",
        "wav_sha256": "9cbc06a7b4829ac90c68ae1bb609fd2a013e8537ffa4ec687c11c7ee38c3742a",
        "pcm_sha256": "212a57c684d708bfba3398f58269755e205ccc2ee330550036a0b523ae24ee13",
        "pcm_samples": 48800,
        "id": "synthetic-af_aoede",
    },
}
REFERENCE_TEXT = "Hello there. Good morning."
TARGETS = {
    "VOSEN": {"text": "Hey Vosen", "ipa": "/heɪ ˈvoʊ.sən/", "token_labels": ["HEY", "VOH", "SEN"]},
    "MERO": {"text": "Hey Mero", "ipa": "/heɪ ˈmɛroʊ/", "token_labels": ["HEY", "MEH", "ROH"]},
}
SPEEDS = (0.92, 1.00, 1.08)
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
FINAL_ROWS = 12
BASE_CALLS = 4
MAX_BASE_SECONDS = 1200
MAX_DERIVE_SECONDS = 300
MIN_DISK_BYTES = 25 * 1024**3
MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
POSITIVE_MIN_SAMPLES = 6400
POSITIVE_MAX_SAMPLES = 28672
SOURCE_SCHEMA = "english-kws-v7-voxcpm-pilot-source-v1"
RESULT_SCHEMA = "english-kws-v7-voxcpm-pilot-result-v1"
OBSERVER_SCHEMA = "english-kws-v7-voxcpm-pilot-observer-v1"
HISTORY = (
    {
        "id": 14422271973,
        "zip_sha256": "26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583",
        "metadata": ["MANIFEST.json"],
        "manifest_sha256": "d23a13424de1da07f369ea355451ae1619cda0526ffebdc0471fdb640b3f47be",
        "rows": 168,
    },
    {
        "id": 14429578824,
        "zip_sha256": "34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9",
        "metadata": ["MANIFEST.json"],
        "manifest_sha256": "c3264e8c621f119dc8d84f8cf32e6abd661d30eae27ddfc90ea9d8830b8a9c92",
        "rows": 126,
    },
    {
        "id": 14434542792,
        "zip_sha256": "d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1",
        "metadata": ["MANIFEST.json"],
        "manifest_sha256": "1d881ec61a436f1f8b07c03afc0ce2f06c5f93c21ca6993bb2d994a81553bc2a",
        "rows": 168,
    },
    {
        "id": 15198755523,
        "zip_sha256": "f07f1f1db965838aa470a2c0cae25a2110def8296a6e990bac602c53a854dcdf",
        "metadata": ["RECIPE.json", "RESULT.json", "audio/*/COMPLETED.json"],
        "recipe_sha256": "0312dbe7125d9a5ba487b90acb7cdcbdc3adade40aa8b5ec5b88040487f22a92",
        "result_sha256": "117afbc49feb012ba63fd22a31ced91f5d3b50b58207bf5c9aedfaa553d25d8a",
        "rows": 144,
    },
)

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result

def read_json(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "REGULAR_JSON_INPUT")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    need(not temp.exists() and not temp.is_symlink(), "NO_STALE_JSON_TEMP")
    raw = canonical(value) + b"\n"
    with temp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return {"path": path.name, "bytes": len(raw), "sha256": sha(raw)}

def write_bytes_once(path, raw):
    path = Path(path)
    need(isinstance(raw, bytes), "BYTE_INPUT_REQUIRED")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path.name, "bytes": len(raw), "sha256": sha(raw)}

def safe_relative(name):
    need(type(name) is str and name and "\\" not in name, "SAFE_RELATIVE_PATH")
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.as_posix() == name
         and not any(part in ("", ".", "..", ".git") for part in path.parts),
         "SAFE_RELATIVE_PATH")
    return name

def read_member(root, relative, expected_sha=None, limit=64 * 1024 * 1024):
    name = safe_relative(relative)
    path = Path(root)
    for part in PurePosixPath(name).parts:
        path = path / part
        need(not path.is_symlink(), "NO_SYMLINK_MEMBER")
    need(path.is_file() and path.stat().st_size <= limit, "REGULAR_BOUNDED_MEMBER")
    raw = path.read_bytes()
    if expected_sha is not None:
        need(sha(raw) == expected_sha, "MEMBER_SHA256_MISMATCH")
    return raw

def source_check():
    need(REPO == "jiying2007/kws-data" and len(MAIN_BEFORE) == 40, "EXACT_PUBLIC_SOURCE")
    need(len(DONORS) == 2 and Counter(v["role"] for v in DONORS.values())
         == Counter({"train": 1, "development": 1}), "ONE_TRAIN_ONE_DEV_DONOR")
    need(len(TARGETS) == 2 and set(TARGETS) == {"VOSEN", "MERO"}, "TWO_EXACT_BRAND_CANDIDATES")
    need(TARGETS["VOSEN"]["text"] == "Hey Vosen" and TARGETS["VOSEN"]["ipa"] == "/heɪ ˈvoʊ.sən/"
         and TARGETS["MERO"]["text"] == "Hey Mero" and TARGETS["MERO"]["ipa"] == "/heɪ ˈmɛroʊ/",
         "EXACT_BRAND_PHRASES_AND_IPA")
    need(len(SPEEDS) == 3 and SPEEDS == (0.92, 1.0, 1.08), "FIXED_TEMPO_VARIANTS")
    need(BASE_CALLS == len(DONORS) * len(TARGETS) and FINAL_ROWS == BASE_CALLS * len(SPEEDS),
         "FOUR_BASE_CALLS_TWELVE_FINAL_CLIPS")
    need(VOXCPM_CPP_COMMIT == "e5b89534e379c79b923c9969b79c3858d140bedb"
         and HF_MODEL_REVISION == "09c7a389438c27100dfa4e6cda1f8f0dd0d2fbfa",
         "EXACT_VOXCPM_SOURCE_PINS")
    need(DONOR_ARTIFACT["id"] == 14406282094 and DONOR_ARTIFACT["run_id"] == 44256271861
         and len(HISTORY) == 4 and len(PROVIDERS) == 4,
         "EXACT_DONOR_HISTORY_AND_OBSERVERS")
    need(all(p in PROVIDERS for p in ("qwen", "whisper", "parakeet", "phoneme")),
         "THREE_ASR_PLUS_PINNED_PHONEME")
    return {
        "source_check": "PASS",
        "schema": SOURCE_SCHEMA,
        "donor_groups": ["af_bella:train", "af_aoede:development"],
        "base_tts_calls": BASE_CALLS,
        "final_audio_rows": FINAL_ROWS,
        "speed_ratios": list(SPEEDS),
        "providers": list(PROVIDERS),
        "training_allowed": False,
        "promotion_allowed": False,
        "human_gold": False,
        "new_model_calls": 0,
    }

def run_identity(environ=None, event=None):
    env = os.environ if environ is None else environ
    expected = {
        "repository": env.get("GITHUB_REPOSITORY"),
        "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"),
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME"),
        "ref": env.get("GITHUB_REF"),
    }
    need(expected["repository"] == REPO and expected["event"] == "push"
         and expected["ref"] == "refs/heads/main" and expected["run_attempt"] == "1",
         "PUBLIC_FIRST_MAIN_PUSH_ONLY")
    need(re.fullmatch(r"[0-9a-f]{40}", expected["commit"] or "") is not None
         and re.fullmatch(r"[1-9][0-9]*", expected["run_id"] or "") is not None,
         "EXACT_GITHUB_COMMIT_AND_RUN")
    if event is None:
        event_path = env.get("GITHUB_EVENT_PATH")
        need(event_path and Path(event_path).is_file(), "GITHUB_EVENT_REQUIRED")
        event = read_json(event_path)
    repo = event.get("repository", {})
    need(event.get("before") == MAIN_BEFORE and event.get("after") == expected["commit"]
         and event.get("ref") == "refs/heads/main" and event.get("forced") is False
         and event.get("deleted") is False and repo.get("full_name") == REPO
         and repo.get("private") is False,
         "EXACT_FIRST_PUBLIC_MAIN_SUCCESSOR")
    return expected | {"before": event["before"]}

def capacity_snapshot(stage):
    disk = shutil.disk_usage("/")
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        mem = {}
    return {
        "stage": stage,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
        "cpu_count": os.cpu_count(),
    }

def capacity_gate(snapshot):
    need(snapshot["disk_free_bytes"] >= MIN_DISK_BYTES, "DISK_BELOW_25_GIB_STOP_NO_RETRY")
    available = snapshot.get("memory_available_bytes")
    need(isinstance(available, int) and available >= MIN_AVAILABLE_MEMORY_BYTES,
         "MEMORY_BELOW_4_GIB_STOP_NO_RETRY")

def wav_pcm(raw, sample_rate, *, min_samples=1, max_samples=10000000):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        count = stream.getnframes()
        comp = stream.getcomptype()
        pcm = stream.readframes(count)
        extra = stream.readframes(1)
    need((channels, width, rate, comp) == (1, 2, sample_rate, "NONE"),
         "MONO_PCM16_EXPECTED_SAMPLE_RATE")
    need(min_samples <= count <= max_samples and len(pcm) == count * 2 and not extra,
         "WAV_FRAME_GEOMETRY")
    return pcm, count

def pcm_metrics(pcm):
    need(len(pcm) % 2 == 0 and pcm, "PCM16_NONEMPTY")
    peak = 0
    clipped = 0
    square_sum = 0
    count = 0
    for (sample,) in struct.iter_unpack("<h", pcm):
        value = int(sample)
        peak = max(peak, abs(value))
        clipped += int(value in (-32768, 32767))
        square_sum += value * value
        count += 1
    rms = math.sqrt(square_sum / count) / 32768.0
    return {"samples": count, "peak": peak / 32768.0, "rms": rms,
            "clipped_samples": clipped, "clipping_pass": clipped == 0}

def validate_donor_manifest(manifest):
    need(manifest.get("schema") == DONOR_ARTIFACT["schema"]
         and manifest.get("record_kind") == DONOR_ARTIFACT["record_kind"]
         and manifest.get("reference_audio_count") == 12
         and manifest.get("roles") == {"train": 6, "development": 2, "calibration": 2, "locked-test": 2}
         and manifest.get("training_allowed") is False
         and manifest.get("human_gold") is False,
         "EXACT_SYNTHETIC_DONOR_MANIFEST")
    need(manifest.get("source") == DONOR_ARTIFACT["internal_source"],
         "DONOR_INTERNAL_LINEAGE")
    rows = manifest.get("rows")
    need(isinstance(rows, list) and len(rows) == 12, "TWELVE_DONOR_METADATA_ROWS")
    by_preset = {row.get("preset_id"): row for row in rows}
    need(len(by_preset) == 12 and set(by_preset) >= set(DONORS), "UNIQUE_DONOR_PRESETS")
    for preset, expected in DONORS.items():
        row = by_preset[preset]
        need(row.get("id") == expected["id"] and row.get("role") == expected["role"]
             and row.get("source_group_id") == expected["source_group_id"]
             and row.get("reference_text") == REFERENCE_TEXT
             and row.get("reference_kind") == "SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY"
             and row.get("training_allowed") is False
             and row.get("no_human_identity_gold") is True
             and row.get("independent_human_source_claim") is False
             and row.get("wav_sha256") == expected["wav_sha256"]
             and row.get("joined_pcm_sha256") == expected["pcm_sha256"]
             and row.get("pcm_samples") == expected["pcm_samples"],
             "EXACT_SELECTED_DONOR_METADATA:" + preset)
    return by_preset

def extract_donors(donor_zip, private_dir):
    donor_zip, private_dir = Path(donor_zip), Path(private_dir)
    need(donor_zip.is_file() and not donor_zip.is_symlink(), "DONOR_ARCHIVE_PRESENT")
    need(donor_zip.stat().st_size == DONOR_ARTIFACT["bytes"]
         and sha_file(donor_zip) == DONOR_ARTIFACT["zip_sha256"], "DONOR_ARCHIVE_IDENTITY")
    private_dir.mkdir(parents=True, exist_ok=False)
    opened = []
    with zipfile.ZipFile(donor_zip) as archive:
        names = set(archive.namelist())
        need("MANIFEST.json" in names, "DONOR_MANIFEST_PRESENT")
        info = archive.getinfo("MANIFEST.json")
        need(not (info.external_attr >> 16) & 0o170000 == 0o120000, "DONOR_MANIFEST_NOT_SYMLINK")
        manifest_raw = archive.read("MANIFEST.json")
        opened.append("MANIFEST.json")
        need(sha(manifest_raw) == DONOR_ARTIFACT["manifest_sha256"], "DONOR_MANIFEST_SHA256")
        manifest = json.loads(manifest_raw, object_pairs_hook=unique_pairs)
        rows = validate_donor_manifest(manifest)
        selected = {}
        for preset, expected in DONORS.items():
            name = expected["wav_path"]
            need(name in names, "SELECTED_DONOR_WAV_PRESENT:" + preset)
            info = archive.getinfo(name)
            need(not (info.external_attr >> 16) & 0o170000 == 0o120000, "SELECTED_DONOR_NOT_SYMLINK")
            raw = archive.read(name)
            opened.append(name)
            need(sha(raw) == expected["wav_sha256"], "SELECTED_DONOR_WAV_SHA256:" + preset)
            pcm, samples = wav_pcm(raw, 16000, min_samples=40000, max_samples=64000)
            need(samples == expected["pcm_samples"] and sha(pcm) == expected["pcm_sha256"],
                 "SELECTED_DONOR_PCM_IDENTITY:" + preset)
            path = private_dir / (preset + ".wav")
            write_bytes_once(path, raw)
            selected[preset] = {
                "private_file": preset + ".wav",
                "role": expected["role"],
                "source_group_id": expected["source_group_id"],
                "wav_sha256": expected["wav_sha256"],
                "pcm_sha256": expected["pcm_sha256"],
                "pcm_samples": samples,
                "reference_text": REFERENCE_TEXT,
                "synthetic_donor": True,
            }
    need(set(opened) == {"MANIFEST.json", DONORS["af_bella"]["wav_path"], DONORS["af_aoede"]["wav_path"]},
         "ONLY_TRAIN_AND_DEVELOPMENT_AUDIO_OPENED")
    return {
        "artifact": DONOR_ARTIFACT,
        "manifest_sha256": DONOR_ARTIFACT["manifest_sha256"],
        "opened_archive_members": opened,
        "opened_roles": ["train", "development"],
        "calibration_audio_opened": False,
        "locked_test_audio_opened": False,
        "human_gold": False,
        "independent_human_speaker_claimed": False,
        "selected": selected,
    }

def collect_audio_hashes(row):
    found = {"pcm": [], "wav": [], "native": []}
    if not isinstance(row, dict):
        return found
    for key in ("pcm_sha256", "native_pcm_sha256", "joined_pcm_sha256", "clone_reference_pcm_sha256"):
        value = row.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            found["pcm"].append(value)
    for key in ("wav_sha256", "clone_reference_wav_sha256"):
        value = row.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            found["wav"].append(value)
    value = row.get("native_wav_sha256")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        found["native"].append(value)
    return found

def read_history_metadata(history_dir, donor_manifest):
    history_dir = Path(history_dir)
    pcm_hashes, wav_hashes, native_hashes = set(), set(), set()
    receipts = []
    for spec in HISTORY:
        archive_path = history_dir / (str(spec["id"]) + ".zip")
        need(archive_path.is_file() and not archive_path.is_symlink(), "HISTORICAL_ARCHIVE_PRESENT:" + str(spec["id"]))
        need(sha_file(archive_path) == spec["zip_sha256"], "HISTORICAL_ARCHIVE_SHA256:" + str(spec["id"]))
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            if spec["id"] != 15198755523:
                need("MANIFEST.json" in names, "HISTORICAL_MANIFEST_PRESENT")
                raw = archive.read("MANIFEST.json")
                need(sha(raw) == spec["manifest_sha256"], "HISTORICAL_MANIFEST_SHA256")
                manifest = json.loads(raw, object_pairs_hook=unique_pairs)
                need(len(manifest.get("rows", [])) == spec["rows"]
                     and manifest.get("completed") == spec["rows"]
                     and manifest.get("denominator") == spec["rows"]
                     and manifest.get("training_allowed") is False
                     and manifest.get("human_gold") is False,
                     "HISTORICAL_COMPLETED_DIAGNOSTIC_MANIFEST")
                for row in manifest["rows"]:
                    hashes = collect_audio_hashes(row)
                    pcm_hashes.update(hashes["pcm"])
                    wav_hashes.update(hashes["wav"])
                    native_hashes.update(hashes["native"])
                receipts.append({"artifact_id": spec["id"], "zip_sha256": spec["zip_sha256"],
                                 "manifest_sha256": spec["manifest_sha256"], "rows": spec["rows"],
                                 "metadata_only": True})
            else:
                for name, expected in (("RECIPE.json", spec["recipe_sha256"]),
                                       ("RESULT.json", spec["result_sha256"])):
                    need(name in names, "V7_G2_METADATA_PRESENT:" + name)
                    raw = archive.read(name)
                    need(sha(raw) == expected, "V7_G2_METADATA_SHA256:" + name)
                    if name == "RECIPE.json":
                        recipe = json.loads(raw, object_pairs_hook=unique_pairs)
                        need(recipe.get("schema") == "english-kws-dual-track-v7-source-recipe-v1"
                             and len(recipe.get("rows", [])) == 144, "V7_G2_RECIPE_ROSTER")
                    else:
                        result = json.loads(raw, object_pairs_hook=unique_pairs)
                        need(result.get("status") == "COMPLETE" and result.get("completed") == 144
                             and result.get("retry_count") == 0 and result.get("kws_forward_calls") == 0,
                             "V7_G2_RESULT_TERMINAL")
                completed_names = sorted(n for n in names if re.fullmatch(r"audio/[^/]+/COMPLETED\.json", n))
                need(len(completed_names) == 144, "V7_G2_COMPLETION_METADATA_DENOMINATOR")
                ids = set()
                for name in completed_names:
                    raw = archive.read(name)
                    row = json.loads(raw, object_pairs_hook=unique_pairs)
                    need(row.get("schema") == "english-kws-dual-track-v7-execution-v1"
                         and row.get("status") == "COMPLETE" and row.get("sample_rate") == 16000
                         and row.get("sample_width") == 2 and row.get("human_gold") is False,
                         "V7_G2_COMPLETION_METADATA_ONLY")
                    ids.add(row.get("id"))
                    hashes = collect_audio_hashes(row)
                    pcm_hashes.update(hashes["pcm"])
                    wav_hashes.update(hashes["wav"])
                    native_hashes.update(hashes["native"])
                need(len(ids) == 144, "V7_G2_UNIQUE_COMPLETION_IDS")
                receipts.append({"artifact_id": spec["id"], "zip_sha256": spec["zip_sha256"],
                                 "recipe_sha256": spec["recipe_sha256"], "result_sha256": spec["result_sha256"],
                                 "rows": 144, "metadata_only": True, "opened_completion_json": 144})
    for row in donor_manifest["rows"]:
        hashes = collect_audio_hashes(row)
        pcm_hashes.update(hashes["pcm"])
        wav_hashes.update(hashes["wav"])
    index = {
        "schema": "english-kws-v7-voxcpm-prior-audio-metadata-index-v1",
        "artifacts": receipts,
        "donor_artifact_id": DONOR_ARTIFACT["id"],
        "pcm_sha256s": sorted(pcm_hashes),
        "wav_sha256s": sorted(wav_hashes),
        "native_sha256s": sorted(native_hashes),
        "audio_hash_count": len(pcm_hashes | wav_hashes | native_hashes),
        "prior_audio_bytes_opened": False,
        "historical_audio_members_extracted": False,
        "sealed_audio_members_opened": False,
    }
    return index

def validate_model_lock(lock, model_path):
    need(lock.get("schema") == "english-v7-voxcpm-model-lock-v1"
         and lock.get("model_repository") == HF_MODEL_REPO
         and lock.get("model_revision") == HF_MODEL_REVISION
         and lock.get("model_license") == "Apache-2.0"
         and lock.get("cpp_repository") == VOXCPM_CPP_REPO
         and lock.get("cpp_commit") == VOXCPM_CPP_COMMIT
         and lock.get("cpp_license") == "Apache-2.0"
         and lock.get("voxcpm_pypi_version") == VOXCPM_PYPI_VERSION
         and lock.get("voxcpm_pypi_sdist_sha256") == VOXCPM_PYPI_SDIST_SHA256
         and lock.get("gguf_package_version") == GGUF_PYPI_VERSION
         and lock.get("gguf_package_wheel_sha256") == GGUF_PYPI_WHEEL_SHA256,
         "EXACT_VOXCPM_MODEL_AND_RUNTIME_LOCK")
    files = lock.get("model_files")
    need(isinstance(files, dict) and set(files) >= {"model.safetensors", "audiovae.pth"},
         "OFFICIAL_MODEL_WEIGHT_FILES")
    for name in ("model.safetensors", "audiovae.pth"):
        item = files[name]
        need(type(item.get("bytes")) is int and item["bytes"] > 0
             and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) is not None,
             "OFFICIAL_MODEL_WEIGHT_SHA256:" + name)
    need(lock.get("backend") == "cpu" and lock.get("threads") == 8
         and lock.get("cfg_value") == 2.0 and lock.get("inference_timesteps") == 10
         and lock.get("retry_badcase") is False and lock.get("gpu_used") is False,
         "FIXED_CPU_INFERENCE_CONFIGURATION")
    need(lock.get("q8_0_audiovae_f16_gguf_sha256")
         and re.fullmatch(r"[0-9a-f]{64}", lock["q8_0_audiovae_f16_gguf_sha256"]),
         "QUANTIZED_GGUF_SHA256")
    need(type(lock.get("q8_0_audiovae_f16_gguf_bytes")) is int
         and lock["q8_0_audiovae_f16_gguf_bytes"] > 0
         and lock.get("ffmpeg_version") and lock.get("ffmpeg_binary_sha256")
         and lock.get("tts_binary_sha256") and lock.get("quantizer_binary_sha256"),
         "RUNTIME_BINARY_AND_QUANTIZED_MODEL_PROVENANCE")
    if model_path is not None:
        model_path = Path(model_path)
        need(model_path.is_file() and not model_path.is_symlink()
             and sha_file(model_path) == lock["q8_0_audiovae_f16_gguf_sha256"]
             and model_path.stat().st_size == lock["q8_0_audiovae_f16_gguf_bytes"],
             "QUANTIZED_MODEL_BYTES_MATCH_LOCK")
    return lock

def planned_base_rows():
    rows = []
    for preset, donor in DONORS.items():
        for candidate, target in TARGETS.items():
            rows.append({
                "id": "base-" + preset + "-" + candidate.lower(),
                "source_preset": preset,
                "role": donor["role"],
                "source_group_id": donor["source_group_id"],
                "candidate": candidate,
                "requested_text": target["text"],
                "requested_ipa": target["ipa"],
                "prompt_text": REFERENCE_TEXT,
                "prompt_wav_sha256": donor["wav_sha256"],
                "status": "not_started",
                "attempts": 0,
            })
    return rows

def planned_final_rows(base_rows):
    rows = []
    base_by_key = {(r["source_preset"], r["candidate"]): r for r in base_rows}
    for preset, donor in DONORS.items():
        for candidate, target in TARGETS.items():
            base = base_by_key[(preset, candidate)]
            for speed in SPEEDS:
                suffix = int(round(speed * 100))
                ident = f"v7-vcp-{preset}-{candidate.lower()}-{suffix:03d}"
                labels = {"VOSEN": 1 if candidate == "VOSEN" else 0,
                          "MERO": 1 if candidate == "MERO" else 0}
                rows.append({
                    "id": ident,
                    "kind": "target",
                    "role": donor["role"],
                    "source_group_id": donor["source_group_id"],
                    "source_preset": preset,
                    "candidate": candidate,
                    "requested_text": target["text"],
                    "requested_ipa": target["ipa"],
                    "candidate_labels": labels,
                    "speed_ratio": speed,
                    "parent_base_id": base["id"],
                    "tts_calls": 0,
                    "derivation_attempts": 0,
                    "generation_status": "NOT_STARTED",
                    "sample_rate": 16000,
                    "channels": 1,
                    "wav_path": "wav/" + ident + ".wav",
                    "pcm_path": "pcm/" + ident + ".pcm",
                })
    need(len(rows) == FINAL_ROWS and len({r["id"] for r in rows}) == FINAL_ROWS,
         "EXACT_TWELVE_FINAL_ROW_ROSTER")
    return rows

def source_inventory(root):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "NO_SOURCE_SYMLINK")
        if path.is_file() and path.name != "FILES.json":
            raw = path.read_bytes()
            records.append({"path": path.relative_to(root).as_posix(),
                            "bytes": len(raw), "sha256": sha(raw)})
    return records

def update_source_state(root, source, result):
    source["updated_at_epoch_seconds"] = int(time.time())
    write_json(Path(root) / "SOURCE.json", source)
    write_json(Path(root) / "RESULT.json", result)

def generate(*, donor_zip, history_dir, output, private_dir, tts_bin, model_path, model_lock_path, runtime_dir):
    identity = run_identity()
    output, private_dir, runtime_dir = Path(output), Path(private_dir), Path(runtime_dir)
    need(not output.exists() and not private_dir.exists() and not runtime_dir.exists(),
         "FRESH_ONE_SHOT_OUTPUT_DIRECTORIES")
    output.mkdir(parents=True)
    private_dir.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True)
    model_lock = validate_model_lock(read_json(model_lock_path), model_path)
    capacity = [capacity_snapshot("before_generation")]
    capacity_gate(capacity[-1])
    model_lock = dict(model_lock)
    model_lock["run"] = identity
    model_lock["capacity_before_generation"] = capacity[-1]
    write_json(output / "MODEL_LOCK.json", model_lock)

    donor_info = extract_donors(donor_zip, private_dir)
    donor_manifest = None
    with zipfile.ZipFile(donor_zip) as archive:
        donor_manifest = json.loads(archive.read("MANIFEST.json"), object_pairs_hook=unique_pairs)
    history_index = read_history_metadata(history_dir, donor_manifest)
    write_json(output / "HISTORY_INDEX.json", history_index)
    base_rows = planned_base_rows()
    final_rows = planned_final_rows(base_rows)
    source = {
        "schema": SOURCE_SCHEMA,
        "run": identity,
        "pipeline_repository": PIPELINE_REPO,
        "pipeline_commit": PIPELINE_COMMIT,
        "model_lock_sha256": sha((output / "MODEL_LOCK.json").read_bytes()),
        "history_index_sha256": sha((output / "HISTORY_INDEX.json").read_bytes()),
        "donor": donor_info,
        "opened_roles": ["train", "development"],
        "calibration_audio_opened": False,
        "locked_test_audio_opened": False,
        "base_tts_call_budget": BASE_CALLS,
        "base_tts_calls_actual": 0,
        "final_row_denominator": FINAL_ROWS,
        "base_outputs": base_rows,
        "rows": final_rows,
        "source_ready": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "human_gold": False,
        "independent_human_speaker_claimed": False,
        "status": "STARTED",
    }
    result = {
        "schema": RESULT_SCHEMA,
        "run": identity,
        "status": "STARTED",
        "source_ready": False,
        "base_tts_calls_expected": BASE_CALLS,
        "base_tts_calls_started": 0,
        "base_tts_calls_returned": 0,
        "derived_rows_expected": FINAL_ROWS,
        "derived_rows_returned": 0,
        "retry_count": 0,
        "duplicate_count": 0,
        "duration_pass_count": 0,
        "clipping_pass_count": 0,
        "training_allowed": False,
        "promotion_allowed": False,
        "human_gold": False,
    }
    update_source_state(output, source, result)
    prior_pcm = set(history_index["pcm_sha256s"])
    prior_wav = set(history_index["wav_sha256s"]) | set(history_index["native_sha256s"])
    seen_pcm, seen_wav = set(), set()
    tts_bin, model_path = Path(tts_bin), Path(model_path)
    need(tts_bin.is_file() and model_path.is_file(), "PINNED_TTS_BINARY_AND_MODEL_PRESENT")
    error = None
    try:
        base_lookup = {(r["source_preset"], r["candidate"]): r for r in base_rows}
        for base in base_rows:
            donor = donor_info["selected"][base["source_preset"]]
            base_path = output / "base" / (base["id"] + ".wav")
            base_path.parent.mkdir(parents=True, exist_ok=True)
            started = {"id": base["id"], "status": "STARTED", "attempt": 1,
                       "run": identity, "prompt_wav_sha256": donor["wav_sha256"],
                       "requested_text": base["requested_text"]}
            write_json(output / "base" / (base["id"] + ".STARTED.json"), started)
            base["status"] = "started"
            base["attempts"] = 1
            result["base_tts_calls_started"] += 1
            update_source_state(output, source, result)
            stdout = runtime_dir / (base["id"] + ".stdout.txt")
            stderr = runtime_dir / (base["id"] + ".stderr.txt")
            cmd = [
                str(tts_bin), "--text", base["requested_text"],
                "--output", str(base_path),
                "--prompt-audio", str(private_dir / donor["private_file"]),
                "--prompt-text", REFERENCE_TEXT,
                "--model-path", str(model_path),
                "--backend", "cpu",
                "--threads", "8",
                "--cfg-value", "2.0",
                "--inference-timesteps", "10",
            ]
            with stdout.open("xb") as out, stderr.open("xb") as err:
                proc = subprocess.run(cmd, stdout=out, stderr=err, check=False,
                                      timeout=MAX_BASE_SECONDS, env=os.environ.copy())
            need(proc.returncode == 0 and base_path.is_file(), "VOXCPM_BASE_SYNTHESIS_FAILED_NO_RETRY:" + base["id"])
            raw = base_path.read_bytes()
            pcm, samples = wav_pcm(raw, 44100, min_samples=1000, max_samples=441000)
            metrics = pcm_metrics(pcm)
            base_sha = sha(raw)
            duplicate = base_sha in prior_wav or base_sha in seen_wav
            seen_wav.add(base_sha)
            base.update({
                "status": "success",
                "attempts": 1,
                "wav_path": "base/" + base_path.name,
                "wav_bytes": len(raw),
                "wav_sha256": base_sha,
                "sample_rate": 44100,
                "pcm_samples": samples,
                "pcm_sha256": sha(pcm),
                "peak": metrics["peak"],
                "rms": metrics["rms"],
                "clipped_samples": metrics["clipped_samples"],
                "duplicate_quarantined": duplicate,
                "duration_seconds": samples / 44100.0,
            })
            write_json(output / "base" / (base["id"] + ".COMPLETED.json"),
                       {"schema": SOURCE_SCHEMA + "/base", **base, "run": identity})
            result["base_tts_calls_returned"] += 1
            result["duplicate_count"] += int(duplicate)
            update_source_state(output, source, result)

        base_by_id = {r["id"]: r for r in base_rows}
        for row in final_rows:
            base = base_by_id[row["parent_base_id"]]
            base_path = output / base["wav_path"]
            wav_path = output / row["wav_path"]
            pcm_path = output / row["pcm_path"]
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            pcm_path.parent.mkdir(parents=True, exist_ok=True)
            row["derivation_attempts"] = 1
            transform = f"atempo={row['speed_ratio']:.2f}"
            cmd = [
                "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
                "-i", str(base_path), "-af", transform,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
                "-threads", "1", "-y", str(wav_path),
            ]
            stdout = runtime_dir / (row["id"] + ".ffmpeg.stdout.txt")
            stderr = runtime_dir / (row["id"] + ".ffmpeg.stderr.txt")
            with stdout.open("xb") as out, stderr.open("xb") as err:
                proc = subprocess.run(cmd, stdout=out, stderr=err, check=False,
                                      timeout=MAX_DERIVE_SECONDS, env=os.environ.copy())
            need(proc.returncode == 0 and wav_path.is_file(), "TEMPO_DERIVATION_FAILED_NO_RETRY:" + row["id"])
            wav_raw = wav_path.read_bytes()
            pcm, samples = wav_pcm(wav_raw, 16000, min_samples=1, max_samples=160000)
            metrics = pcm_metrics(pcm)
            pcm_sha, wav_sha = sha(pcm), sha(wav_raw)
            duplicate = pcm_sha in prior_pcm or pcm_sha in seen_pcm or wav_sha in prior_wav or wav_sha in seen_wav
            seen_pcm.add(pcm_sha)
            seen_wav.add(wav_sha)
            write_bytes_once(pcm_path, pcm)
            duration_pass = POSITIVE_MIN_SAMPLES <= samples <= POSITIVE_MAX_SAMPLES
            clipping_pass = metrics["clipping_pass"]
            row.update({
                "generation_status": "SUCCESS",
                "tts_calls": 0,
                "parent_base_sha256": base["wav_sha256"],
                "transform": transform,
                "transform_tool": "ffmpeg-atempo",
                "wav_bytes": len(wav_raw),
                "wav_sha256": wav_sha,
                "pcm_bytes": len(pcm),
                "pcm_sha256": pcm_sha,
                "pcm_samples": samples,
                "duration_seconds": samples / 16000.0,
                "duration_pass": duration_pass,
                "peak": metrics["peak"],
                "rms": metrics["rms"],
                "clipped_samples": metrics["clipped_samples"],
                "clipping_pass": clipping_pass,
                "duplicate_quarantined": duplicate,
            })
            result["derived_rows_returned"] += 1
            result["duplicate_count"] += int(duplicate)
            result["duration_pass_count"] += int(duration_pass)
            result["clipping_pass_count"] += int(clipping_pass)
            update_source_state(output, source, result)

        source["status"] = "COMPLETE"
        source["source_ready"] = (
            result["base_tts_calls_returned"] == BASE_CALLS
            and result["derived_rows_returned"] == FINAL_ROWS
            and result["duplicate_count"] == 0
        )
        result["status"] = "COMPLETE"
        result["source_ready"] = source["source_ready"]
        result["duration_gate_pass"] = result["duration_pass_count"] == FINAL_ROWS
        result["clipping_gate_pass"] = result["clipping_pass_count"] == FINAL_ROWS
        result["source_ready_reason"] = (
            "ALL_TWELVE_UNIQUE_AUDIO_ROWS" if source["source_ready"]
            else "INCOMPLETE_OR_DUPLICATE_SOURCE_STOP_NO_RETRY"
        )
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)[:2000]
        source["status"] = "FAILED_NO_RETRY"
        result["status"] = "FAILED_NO_RETRY"
        result["error"] = error
        source["source_ready"] = False
        result["source_ready"] = False
        raise
    finally:
        capacity.append(capacity_snapshot("after_generation"))
        source["capacity_snapshots"] = capacity
        source["source_ready"] = bool(source.get("source_ready")) and error is None
        result["source_ready"] = source["source_ready"]
        result["capacity_snapshots"] = capacity
        result["retry_count"] = 0
        source["status"] = "COMPLETE" if error is None else "FAILED_NO_RETRY"
        result["status"] = "COMPLETE" if error is None else "FAILED_NO_RETRY"
        update_source_state(output, source, result)
        write_json(output / "FILES.json", {"schema": SOURCE_SCHEMA + "/files",
                                           "files": source_inventory(output),
                                           "self_excluded": "FILES.json"})
    return result

def verify_source(source_root):
    root = Path(source_root)
    need(root.is_dir() and not root.is_symlink(), "REGULAR_SOURCE_DIRECTORY")
    source = read_json(root / "SOURCE.json")
    result = read_json(root / "RESULT.json")
    history = read_json(root / "HISTORY_INDEX.json")
    lock = read_json(root / "MODEL_LOCK.json")
    inventory = read_json(root / "FILES.json")
    need(source.get("schema") == SOURCE_SCHEMA and source.get("training_allowed") is False
         and source.get("promotion_allowed") is False and source.get("human_gold") is False
         and source.get("independent_human_speaker_claimed") is False,
         "DIAGNOSTIC_SOURCE_ONLY")
    need(source.get("opened_roles") == ["train", "development"]
         and source.get("calibration_audio_opened") is False
         and source.get("locked_test_audio_opened") is False,
         "SEALED_DONOR_ROLES_UNOPENED")
    need(sha((root / "HISTORY_INDEX.json").read_bytes()) == source.get("history_index_sha256")
         and sha((root / "MODEL_LOCK.json").read_bytes()) == source.get("model_lock_sha256"),
         "SOURCE_METADATA_BYTE_BINDING")
    need(inventory == {"schema": SOURCE_SCHEMA + "/files", "files": source_inventory(root),
                       "self_excluded": "FILES.json"}, "SOURCE_FILE_CLOSURE")
    need(result.get("schema") == RESULT_SCHEMA and result.get("run") == source.get("run")
         and result.get("training_allowed") is False and result.get("promotion_allowed") is False
         and result.get("retry_count") == 0, "SOURCE_RESULT_IDENTITY")
    need(source.get("pipeline_commit") == PIPELINE_COMMIT, "EXACT_PIPELINE_COMMIT")
    validate_model_lock(lock, None)
    base_rows, rows = source.get("base_outputs"), source.get("rows")
    need(isinstance(base_rows, list) and len(base_rows) == BASE_CALLS
         and isinstance(rows, list) and len(rows) == FINAL_ROWS, "SOURCE_DENOMINATORS")
    need(Counter(r.get("role") for r in rows) == Counter({"train": 6, "development": 6}),
         "PILOT_ROLE_DENOMINATOR")
    history_pcm = set(history.get("pcm_sha256s", []))
    history_wav = set(history.get("wav_sha256s", [])) | set(history.get("native_sha256s", []))
    seen_pcm, seen_wav = set(), set()
    for base in base_rows:
        need(base.get("status") == "success" and base.get("attempts") == 1
             and base.get("role") in ("train", "development"), "BASE_OUTPUT_TERMINAL")
        raw = read_member(root, base["wav_path"], base["wav_sha256"])
        pcm, samples = wav_pcm(raw, 44100, min_samples=1000, max_samples=441000)
        need(samples == base["pcm_samples"] and sha(pcm) == base["pcm_sha256"]
             and base["duplicate_quarantined"] is False, "BASE_AUDIO_READBACK")
        need(base["wav_sha256"] not in history_wav and base["wav_sha256"] not in seen_wav,
             "NO_BASE_AUDIO_DUPLICATES")
        seen_wav.add(base["wav_sha256"])
    for row in rows:
        need(row.get("generation_status") == "SUCCESS" and row.get("role") in ("train", "development")
             and row.get("kind") == "target" and row.get("tts_calls") == 0
             and row.get("candidate_labels") == {
                 "VOSEN": 1 if row["candidate"] == "VOSEN" else 0,
                 "MERO": 1 if row["candidate"] == "MERO" else 0,
             }, "FINAL_ROW_PROVENANCE_AND_CANDIDATE_LABELS")
        pcm = read_member(root, row["pcm_path"], row["pcm_sha256"])
        wav = read_member(root, row["wav_path"], row["wav_sha256"])
        decoded, samples = wav_pcm(wav, 16000, min_samples=1, max_samples=160000)
        metrics = pcm_metrics(pcm)
        need(decoded == pcm and samples == row["pcm_samples"]
             and sha(pcm) == row["pcm_sha256"]
             and metrics["clipped_samples"] == row["clipped_samples"],
             "FINAL_PCM_WAV_IDENTITY")
        need(row["pcm_sha256"] not in history_pcm and row["wav_sha256"] not in history_wav
             and row["pcm_sha256"] not in seen_pcm and row["wav_sha256"] not in seen_wav
             and row["duplicate_quarantined"] is False, "NO_HISTORICAL_OR_INTERNAL_DUPLICATES")
        seen_pcm.add(row["pcm_sha256"])
        seen_wav.add(row["wav_sha256"])
    source_ready = (
        result.get("source_ready") is True
        and len(base_rows) == BASE_CALLS and len(rows) == FINAL_ROWS
        and all(row.get("generation_status") == "SUCCESS" for row in rows)
    )
    return {"source": source, "result": result, "history": history,
            "source_ready": source_ready, "rows": rows, "base_rows": base_rows}

def read_source_cli(source_root, output_path):
    try:
        verified = verify_source(source_root)
        value = {
            "schema": SOURCE_SCHEMA + "/readback",
            "source_ready": verified["source_ready"],
            "base_tts_calls": len(verified["base_rows"]),
            "final_rows": len(verified["rows"]),
            "opened_roles": verified["source"]["opened_roles"],
            "calibration_audio_opened": False,
            "locked_test_audio_opened": False,
            "training_allowed": False,
            "promotion_allowed": False,
        }
    except Exception as exc:
        root = Path(source_root)
        source = read_json(root / "SOURCE.json") if (root / "SOURCE.json").is_file() else {}
        result = read_json(root / "RESULT.json") if (root / "RESULT.json").is_file() else {}
        value = {
            "schema": SOURCE_SCHEMA + "/readback", "source_ready": False,
            "status": result.get("status", "UNKNOWN"),
            "error": type(exc).__name__ + ": " + str(exc)[:1000],
            "base_tts_calls": result.get("base_tts_calls_returned"),
            "final_rows": result.get("derived_rows_returned"),
            "opened_roles": source.get("opened_roles", ["train", "development"]),
            "calibration_audio_opened": False, "locked_test_audio_opened": False,
            "training_allowed": False, "promotion_allowed": False,
        }
    write_json(output_path, value)
    return value

def clean_environment():
    env = os.environ.copy()
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"):
        env.pop(key, None)
    env.update({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"})
    return env

def install_observer_runtime(provider, venv, runtime):
    need(provider in PROVIDERS, "KNOWN_OBSERVER_PROVIDER")
    venv, runtime = Path(venv), Path(runtime)
    need(not venv.exists() and not runtime.exists(), "FRESH_OBSERVER_RUNTIME")
    runtime.mkdir(parents=True)
    commands = []
    deadline = time.monotonic() + 840

    def execute(args, name):
        commands.append({"step": name, "status": "STARTED"})
        write_json(runtime / "INSTALLATION.json", {"provider": provider, "status": "STARTED",
                                                    "retry_count": 0, "commands": commands})
        with (runtime / (name + ".stdout.txt")).open("xb") as out, \
             (runtime / (name + ".stderr.txt")).open("xb") as err:
            proc = subprocess.run([str(x) for x in args], stdout=out, stderr=err,
                                  env=clean_environment(), timeout=max(1, deadline - time.monotonic()),
                                  check=False)
        commands[-1].update(status="RETURNED", returncode=proc.returncode)
        need(proc.returncode == 0, "OBSERVER_INSTALL_FAILED:" + name)

    py = venv / "bin/python"
    pip = [py, "-I", "-m", "pip", "install", "--disable-pip-version-check",
           "--no-cache-dir", "--retries", "0", "--timeout", "60"]
    cpu = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    try:
        execute([sys.executable, "-m", "venv", venv], "venv")
        if provider == "parakeet":
            execute(pip + ["huggingface-hub==0.36.2", "hf-xet==1.2.0"], "observer_dependencies")
        else:
            torch_packages = ["torch==2.9.1+cpu"] + ([] if provider == "phoneme" else ["torchaudio==2.9.1+cpu"])
            execute(pip + cpu + torch_packages, "torch_cpu")
            packages = {
                "qwen": ["qwen-asr==0.0.6", "huggingface-hub==0.36.2"],
                "whisper": ["openai-whisper==20250625"],
                "phoneme": ["transformers==4.46.3", "huggingface-hub==0.36.2", "numpy==1.26.4"],
            }
            execute(pip + packages[provider], "observer_dependencies")
        execute([py, "-I", "-m", "pip", "check"], "pip_check")
        with (runtime / "PIP_FREEZE.txt").open("xb") as stream:
            subprocess.run([str(py), "-I", "-m", "pip", "freeze"], stdout=stream,
                           stderr=subprocess.STDOUT, env=clean_environment(), timeout=30, check=False)
        result = {"provider": provider, "status": "COMPLETE", "retry_count": 0,
                  "commands": commands, "new_model_calls": 0}
    except BaseException as exc:
        result = {"provider": provider, "status": "FAILED_NO_RETRY",
                  "error": type(exc).__name__ + ": " + str(exc), "retry_count": 0,
                  "commands": commands, "new_model_calls": 0}
        raise
    finally:
        write_json(runtime / "INSTALLATION.json", result)
    return result

def observer_run(provider, source_root, output_root, asset_root):
    import english_pocket12_observe as observe
    identity = run_identity()
    need(provider in PROVIDERS, "KNOWN_OBSERVER_PROVIDER")
    observe.source_check()
    source_info = verify_source(source_root)
    need(source_info["source_ready"], "COMPLETE_SOURCE_REQUIRED_FOR_OBSERVATION")
    source_root, output_root, asset_root = map(lambda p: Path(p).absolute(),
                                               (source_root, output_root, asset_root))
    need(not output_root.exists() and not asset_root.exists(), "FRESH_OBSERVER_OUTPUT")
    need(output_root != asset_root and output_root not in asset_root.parents
         and asset_root not in output_root.parents and source_root not in output_root.parents,
         "PRIVATE_ASSETS_SEPARATE")
    output_root.mkdir(parents=True)
    rows = source_info["rows"]
    source_sha = sha((source_root / "SOURCE.json").read_bytes())
    ledger = observe.initial_rows(rows)
    observe.write(output_root / "STARTED.json", {
        "schema": observe.SCHEMA, "provider": provider, "run": identity,
        "source_sha256": source_sha, "denominator": 12, "maximum_calls": 12, **observe.FLAGS
    })
    failure = None
    def expired(signum, frame):
        raise TimeoutError("MODEL_INITIALIZATION_DEADLINE_840S")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        observe.write(output_root / "MODEL_LOAD_STARTED.json",
                      {"provider": provider, "actual_forward_attempts": 0})
        signal.alarm(840)
        loaded = observe._load(provider, asset_root, output_root)
        signal.alarm(0)
        backend, lock = (loaded[:3], loaded[3]) if provider == "phoneme" else loaded
        observe.write(output_root / "MODEL_LOCK.json", lock)
        observe._execute_rows(provider, source_root, rows, output_root, identity,
                              source_sha, backend, ledger)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if asset_root.exists():
            shutil.rmtree(asset_root)
    result = observe._finish(output_root, provider, ledger, identity, source_sha, failure)
    return result

def load_pipeline_admission(pipeline_root):
    path = Path(pipeline_root) / "training" / "english_kws_dual_track_v7_silver.py"
    need(path.is_file() and not path.is_symlink(), "PINNED_PIPELINE_ADMISSION_SOURCE")
    spec = importlib.util.spec_from_file_location("v7_pilot_admission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def find_observer_root(observers_root, provider):
    root = Path(observers_root)
    options = [root / provider, root / ("v7-voxcpm-observer-" + provider)]
    for candidate in options:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    return None

def verify_observer(provider, observer_root, source_info, output_root, pipeline):
    import english_pocket12_observe as observe
    if observer_root is None:
        return {"status": "MISSING", "rows": {}, "lock": None, "result": None}
    root = Path(observer_root)
    try:
        result = read_json(root / "RESULT.json")
        files = read_json(root / "FILES.json")
        need(files == {"schema": observe.SCHEMA + "/files", "self_excluded": "FILES.json",
                       "files": observe.file_records(root)}, "OBSERVER_FILE_CLOSURE")
        need(result.get("schema") == observe.SCHEMA and result.get("provider") == provider
             and result.get("run") == source_info["source"]["run"]
             and result.get("source_sha256") == sha((Path(source_info["root"]) / "SOURCE.json").read_bytes())
             and result.get("denominator") == FINAL_ROWS and result.get("retry_count") == 0
             and all(result.get(k) == v for k, v in observe.FLAGS.items()),
             "OBSERVER_RESULT_BINDING")
        need(result.get("status") == "COMPLETE" and result.get("completed_count") == FINAL_ROWS
             and result.get("actual_forward_attempts") == FINAL_ROWS
             and result.get("native_forward_returns") == FINAL_ROWS
             and result.get("failed_count") == 0 and result.get("not_started_count") == 0,
             "ALL_TWELVE_OBSERVER_CALLS_COMPLETE")
        lock = read_json(root / "MODEL_LOCK.json")
        if provider == "phoneme":
            native = observe.native_module()
            need(lock.get("model") == native.OBSERVERS[provider]
                 and lock.get("files") == native.OBSERVER_FILES[provider]
                 and lock.get("runtime_versions") == native.RUNTIME_VERSIONS
                 and lock.get("native_forward_attempts_at_initialization") == 0
                 and lock.get("model_parameters_modified") is False
                 and lock.get("target_conditioned") is False,
                 "FROZEN_PHONEME_MODEL_LOCK")
            for name, pin in native.OBSERVER_FILES[provider].items():
                if name != "pytorch_model.bin":
                    raw = read_member(root, "model-metadata/" + name, pin["sha256"])
                    need(len(raw) == pin["bytes"], "PHONEME_METADATA_PIN")
            vocab = read_json(root / "model-metadata" / "vocab.json")
        else:
            need(lock == {"observer": observe.identity(provider),
                          "runtime_versions": observe.TEXT_RUNTIME[provider],
                          "adapter_sha256": observe.HELPER_PINS["automation/english_seven_voice_asr_once.py"],
                          "initialization_model_calls": 0, "target_conditioned": False},
                 "FROZEN_ASR_MODEL_LOCK")
            vocab = None
        source_root = Path(source_info["root"])
        rows = {}
        for row in source_info["rows"]:
            directory = root / "rows" / row["id"]
            capture = read_json(directory / "CAPTURE.json")
            need(capture.get("status") == "success" and capture.get("id") == row["id"]
                 and capture.get("input_pcm_sha256") == row["pcm_sha256"]
                 and capture.get("input_wav_sha256") == row["wav_sha256"]
                 and capture.get("input_samples") == row["pcm_samples"]
                 and capture.get("target_conditioned") is False
                 and capture.get("retry_count") == 0,
                 "BLIND_OBSERVER_SOURCE_BINDING")
            if provider == "phoneme":
                pcm = read_member(source_root, row["pcm_path"], row["pcm_sha256"])
                tokens = observe._verify_phone(directory, capture, row, pcm, vocab)
                rows[row["id"]] = {"phone_tokens": tokens}
            else:
                raw_text = capture.get("raw_transcript")
                need(type(raw_text) is str and 0 < len(raw_text.strip()) <= 512
                     and not any(ord(ch) < 32 for ch in raw_text.strip()),
                     "NONEMPTY_BLIND_ASR_TRANSCRIPT")
                rows[row["id"]] = {"raw_text": raw_text}
        return {"status": "COMPLETE", "rows": rows, "lock": lock, "result": result}
    except Exception as exc:
        return {"status": "INVALID", "error": type(exc).__name__ + ": " + str(exc),
                "rows": {}, "lock": None, "result": None}

def combine(*, source_root, observers_root, output_root, pipeline_root):
    import english_pocket12_observe as observe
    source_info = verify_source(source_root)
    pipeline = load_pipeline_admission(pipeline_root)
    observe.source_check()
    output = Path(output_root)
    need(not output.exists(), "FRESH_COMBINED_OUTPUT")
    output.mkdir(parents=True)
    source_sha = sha((Path(source_root) / "SOURCE.json").read_bytes())
    provider_data = {}
    for provider in PROVIDERS:
        root = find_observer_root(observers_root, provider)
        provider_data[provider] = verify_observer(provider, root, source_info | {"root": str(source_root)},
                                                   output, pipeline)
    rows_out = []
    candidate_counts = {
        c: {"positive": {"train": 0, "development": 0},
            "negative": {"train": 0, "development": 0},
            "unknown": 0, "silver": 0, "weak": 0}
        for c in TARGETS
    }
    observers_complete = all(provider_data[p]["status"] == "COMPLETE" for p in PROVIDERS)
    for row in source_info["rows"]:
        texts = []
        phones = None
        if observers_complete:
            try:
                for provider in ("qwen", "whisper", "parakeet"):
                    capture_value = provider_data[provider]["rows"][row["id"]]["raw_text"]
                    texts.append(capture_value)
                phones = provider_data["phoneme"]["rows"][row["id"]]["phone_tokens"]
            except Exception:
                texts, phones = [], None
        candidate_results = {}
        for candidate in TARGETS:
            if len(texts) == 3 and phones:
                tier, label, reason, observed = pipeline.decide_candidate(row, candidate, texts, phones)
            else:
                tier, label, reason, observed = "unknown", None, "MISSING_OR_INCOMPLETE_OBSERVERS", None
            candidate_results[candidate] = {
                "tier": tier, "label": label, "reason": reason, "observed_phrase": observed,
            }
            counts = candidate_counts[candidate]
            if label is None:
                counts["unknown"] += 1
            else:
                counts[tier] += 1
                key = "positive" if label == 1 else "negative"
                counts[key][row["role"]] += 1
        if row.get("duration_pass") is True:
            pipeline.verify_pcm(row, source_root)
        row_out = {
            "id": row["id"], "role": row["role"], "source_group_id": row["source_group_id"],
            "candidate": row["candidate"], "requested_text": row["requested_text"],
            "requested_ipa": row["requested_ipa"], "speed_ratio": row["speed_ratio"],
            "pcm_sha256": row["pcm_sha256"], "wav_sha256": row["wav_sha256"],
            "pcm_samples": row["pcm_samples"], "duration_pass": row["duration_pass"],
            "clipping_pass": row["clipping_pass"], "duplicate_quarantined": row["duplicate_quarantined"],
            "candidate_results": candidate_results,
        }
        rows_out.append(row_out)

        # Adapt the blind provider captures into the exact V7 admission capture schema.
        if observers_complete and len(texts) == 3 and phones:
            for provider in PROVIDERS:
                value = ({"raw_text": provider_data[provider]["rows"][row["id"]]["raw_text"]}
                         if provider != "phoneme"
                         else {"phone_tokens": provider_data[provider]["rows"][row["id"]]["phone_tokens"]})
                identity = pipeline.observer_identity(provider)
                capture = {
                    "recognizer_id": identity["recognizer_id"],
                    "model_family": identity["model_family"],
                    "model_sha256": identity["model_sha256"],
                    "decode_config_sha256": identity["decode_config_sha256"],
                    "id": row["id"],
                    "schema": pipeline.CAPTURE_SCHEMA,
                    "status": "success",
                    "source_pcm_sha256": row["pcm_sha256"],
                    "source_wav_sha256": row["wav_sha256"],
                    "source_samples": row["pcm_samples"],
                    "target_conditioned": False,
                    "input_coverage_complete": True,
                    "decoder_output_complete": True,
                } | value
                capture_path = "observations/" + provider + "/" + row["id"] + ".json"
                capture_sha = write_json(output / capture_path, capture)["sha256"]
                observation = capture | {"capture_path": capture_path, "capture_sha256": capture_sha}
                # Recompute from the actual bytes using the same V7 verifier as training admission.
                if provider == "phoneme":
                    verified_phone = pipeline.verify_observation(observation, provider, row, output, phones=True)
                    need(verified_phone == phones, "V7_PHONE_CAPTURE_RECONSTRUCTION")
                else:
                    verified_text = pipeline.verify_observation(observation, provider, row, output, phones=False)
                    need(verified_text == pipeline.normalized_text(value["raw_text"]),
                         "V7_ASR_CAPTURE_RECONSTRUCTION")

    source_result = source_info["result"]
    duration_pass = all(r.get("duration_pass") is True for r in source_info["rows"])
    clipping_pass = all(r.get("clipping_pass") is True for r in source_info["rows"])
    duplicate_pass = all(r.get("duplicate_quarantined") is False for r in source_info["rows"])
    candidate_pass = all(
        counts["unknown"] == 0
        and counts["positive"]["train"] >= 2
        and counts["positive"]["development"] >= 2
        and counts["negative"]["train"] >= 2
        and counts["negative"]["development"] >= 2
        for counts in candidate_counts.values()
    )
    observer_pass = observers_complete and all(
        provider_data[p]["result"]["actual_forward_attempts"] == FINAL_ROWS for p in PROVIDERS
    )
    machine_pass = bool(source_info["source_ready"] and observer_pass and candidate_pass
                        and duration_pass and clipping_pass and duplicate_pass)
    result = {
        "schema": RESULT_SCHEMA,
        "run": source_info["source"]["run"],
        "source_sha256": source_sha,
        "pipeline_repository": PIPELINE_REPO,
        "pipeline_commit": PIPELINE_COMMIT,
        "status": "MACHINE_PILOT_PASS_REVIEW_PENDING" if machine_pass else "STOP_OR_INCOMPLETE",
        "source_ready": source_info["source_ready"],
        "observers_complete": observer_pass,
        "candidate_joint_asr_phone_pass": candidate_pass,
        "duration_pass": duration_pass,
        "clipping_pass": clipping_pass,
        "duplicate_pass": duplicate_pass,
        "candidate_counts": candidate_counts,
        "observer_status": {p: provider_data[p]["status"] for p in PROVIDERS},
        "rows": rows_out,
        "naturalness_manual_review": "NOT_REVIEWED",
        "human_gold": False,
        "independent_human_speaker_claimed": False,
        "synthetic_to_synthetic_only": True,
        "scale_up_allowed": False,
        "training_allowed": False,
        "promotion_allowed": False,
        "required_next_gate": "BLIND_HUMAN_NATURALNESS_REVIEW_AND_SOURCE_RIGHTS_REVIEW",
        "base_tts_calls": BASE_CALLS,
        "final_audio_rows": FINAL_ROWS,
        "observer_calls_per_provider": FINAL_ROWS,
        "retry_count": 0,
    }
    write_json(output / "PILOT_RESULT.json", result)
    write_json(output / "FILES.json", {"schema": RESULT_SCHEMA + "/files",
                                       "files": source_inventory(output),
                                       "self_excluded": "FILES.json"})
    return result

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--read-source", action="store_true")
    parser.add_argument("--install-observer")
    parser.add_argument("--observe")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--donor-zip")
    parser.add_argument("--history-dir")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--private-dir")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--tts-bin")
    parser.add_argument("--model")
    parser.add_argument("--model-lock")
    parser.add_argument("--venv")
    parser.add_argument("--observers-root")
    parser.add_argument("--assets")
    parser.add_argument("--pipeline-root")
    args = parser.parse_args(argv)

    if args.source_check:
        value = source_check()
        import english_pocket12_observe as observe
        value["observer_helper_source_check"] = observe.source_check()
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.guard:
        print(json.dumps(run_identity(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.generate:
        value = generate(donor_zip=args.donor_zip, history_dir=args.history_dir,
                         output=args.output, private_dir=args.private_dir,
                         tts_bin=args.tts_bin, model_path=args.model,
                         model_lock_path=args.model_lock, runtime_dir=args.runtime_dir)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.read_source:
        value = read_source_cli(args.source, args.output)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.install_observer:
        value = install_observer_runtime(args.install_observer, args.venv, args.runtime_dir)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.observe:
        value = observer_run(args.observe, args.source, args.output, args.assets)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.combine:
        value = combine(source_root=args.source, observers_root=args.observers_root,
                        output_root=args.output, pipeline_root=args.pipeline_root)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    parser.error("choose one explicit stage")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
