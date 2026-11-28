"""One-shot VoxCPM1.5 CPU pilot source preparation and synthesis.

Only two synthetic TRAIN/DEVELOPMENT donor references are opened. The other
ten reference WAVs, including CALIBRATION and LOCKED-TEST, are never read.
The pilot is diagnostic only: no KWS forwards, optimizer updates or promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import wave
import zipfile

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "english_kws_v7_voxcpm_pilot_plan.json"
SCHEMA = "english-kws-v7-voxcpm-pilot-source-v1"
HEX40 = re.compile(r"[0-9a-f]{40}$")
HEX64 = re.compile(r"[0-9a-f]{64}$")
MAX_WAV_BYTES = 4 * 1024 * 1024
MAX_PCM_BYTES = 1024 * 1024
PROMPT_TEXT = "Hello there. Good morning."


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + chr(10)).encode()


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


def write_once(path, value):
    path = Path(path)
    need(path.parent.is_dir() and not path.exists() and not path.is_symlink(),
         "CREATE_ONCE_FILE")
    raw = canonical(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "sha256": sha(raw)}


def regular(path, limit):
    path = Path(path)
    need(path.is_file() and not path.is_symlink()
         and not any(parent.is_symlink() for parent in path.parents),
         "REGULAR_NON_SYMLINK_FILE")
    need(0 < path.stat().st_size <= limit, "BOUNDED_FILE_SIZE")
    return path.read_bytes()


def sha_file(path, limit):
    path = Path(path)
    need(path.is_file() and not path.is_symlink()
         and not any(parent.is_symlink() for parent in path.parents)
         and 0 < path.stat().st_size <= limit, "BOUNDED_REGULAR_HASH_FILE")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name):
    need(type(name) is str and name and chr(92) not in name, "SAFE_RELATIVE_PATH")
    rel = PurePosixPath(name)
    need(not rel.is_absolute() and rel.as_posix() == name
         and all(part not in ("", ".", "..", ".git") for part in rel.parts),
         "SAFE_RELATIVE_PATH")
    return name


def within(root, name):
    rel = PurePosixPath(safe_relative(name))
    path = Path(root)
    for part in rel.parts:
        path = path / part
        need(not path.is_symlink(), "NO_SYMLINK_MEMBER")
    return path


def load_plan():
    raw = regular(PLAN_PATH, 64 * 1024)
    plan = json.loads(raw, object_pairs_hook=unique_pairs)
    need(plan.get("schema") == "english-kws-v7-voxcpm-cpu-pilot-v1"
         and plan.get("campaign_id") == "v7-voxcpm-two-donor-12-clip-pilot"
         and plan.get("repository") == "jiying2007/kws-data"
         and plan.get("enabled") is True
         and plan.get("status") == "PILOT_ONLY_NO_TRAINING",
         "FROZEN_VOXCPM_PILOT_PLAN")
    need(plan.get("main_before") == "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2"
         and plan.get("execution") == {
             "event": "push", "ref": "refs/heads/main", "run_attempt": 1,
             "public_repository_only": True, "retries": 0, "automatic_expansion": False,
         }, "EXACT_PUBLIC_MAIN_ONE_SHOT")
    donors = plan.get("donors")
    need(isinstance(donors, list) and len(donors) == 2
         and [(x.get("donor_id"), x.get("role")) for x in donors] ==
             [("synthetic-af_bella", "train"), ("synthetic-af_aoede", "development")],
         "ONLY_ONE_TRAIN_AND_ONE_DEVELOPMENT_DONOR")
    for donor in donors:
        need(donor.get("reference_kind") == "SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY"
             and donor.get("human_gold") is False and donor.get("training_allowed") is False
             and donor.get("sample_rate") == 16000 and donor.get("channels") == 1
             and donor.get("reference_text") == PROMPT_TEXT
             and HEX64.fullmatch(donor.get("reference_wav_sha256", "")) is not None
             and HEX64.fullmatch(donor.get("joined_pcm_sha256", "")) is not None,
             "SYNTHETIC_DONOR_REFERENCE_ONLY")
    ref = plan["donor_reference"]
    need(ref.get("artifact_id") == 14406282094
         and ref.get("run_id") == 44256271861
         and ref.get("artifact_sha256") == "8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5"
         and ref.get("record_kind") == "SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY"
         and ref.get("human_gold") is False and ref.get("training_allowed") is False
         and ref.get("calibration_and_locked_test_audio_opened") is False,
         "IMMUTABLE_SYNTHETIC_REFERENCE_ARTIFACT")
    need(plan["generator"] == {
        "repository": "bluryar/VoxCPM.cpp",
        "commit": "e5b89534e379c79b923c9969b79c3858d140bedb",
        "license": "Apache-2.0", "backend": "cpu", "threads": 8,
        "retry_badcase": False,
    }, "PINNED_VOXCPM_CPP_ENGINE")
    need(plan["converter"] == {
        "repository": "OpenBMB/VoxCPM",
        "commit": "300e4260b509066151d623dc9c0aa922cf82232b",
        "license": "Apache-2.0", "script": "scripts/convert_voxcpm_to_gguf.py",
    }, "PINNED_OFFICIAL_CONVERTER")
    need(plan["model"] == {
        "repository": "openbmb/VoxCPM1.5",
        "revision": "09c7a389438c27100dfa4e6cda1f8f0dd0d2fbfa",
        "license": "Apache-2.0",
        "files": {
            "model.safetensors": "7e97ee2cd2ca891662f7da368921df08368a616410afe67461253e4dcd3823cc",
            "audiovae.pth": "8160987a7f10acd4c49b72a9a3e974c0b5c1c786538cd4bcaac5f2df388c0f14",
        },
        "audio_vae_sample_rate": 44100,
        "quantization": "Q8_0+AudioVAE-F16",
    }, "PINNED_OFFICIAL_VOXCPM15_WEIGHTS")
    need(plan["candidates"] == {
        "VOSEN": {"candidate_id": "hey-vosen-v7", "phrase": "Hey Vosen.",
                  "ipa": "/heɪ ˈvoʊ.sən/", "phone_core": "voiceless /s/"},
        "MERO": {"candidate_id": "hey-mero-v7", "phrase": "Hey Mero.",
                 "ipa": "/heɪ ˈmɛroʊ/", "phone_core": "DRESS /ɛ/ + rhotic /ɹ/"},
    }, "EXACT_BRAND_PRONUNCIATIONS")
    need(plan["synthesis"]["base_tts_calls"] == 4
         and plan["synthesis"]["final_clip_count"] == 12
         and plan["synthesis"]["deterministic_tempo_variants"] == [0.92, 1.0, 1.08]
         and plan["synthesis"]["retry_badcase"] is False
         and plan["synthesis"]["no_regenerate_old_audio"] is True
         and plan["synthesis"]["output_sample_rate"] == 16000
         and plan["synthesis"]["output_channels"] == 1
         and plan["synthesis"]["output_sample_width_bytes"] == 2
         and plan["synthesis"]["min_samples"] == 6400
         and plan["synthesis"]["max_samples"] == 28672,
         "FINITE_SYNTHESIS_AND_PCM_GEOMETRY")
    need(plan["budgets"] == {
        "tts_model_calls": 4, "asr_model_calls": 36, "phoneme_model_calls": 12,
        "total_model_calls": 52, "tempo_transforms": 12,
        "kws_model_forwards": 0, "optimizer_updates": 0,
    }, "FIXED_52_CALL_PILOT_BUDGET")
    need(plan["observers"]["max_calls_per_provider"] == 12
         and plan["observers"]["total_observer_calls"] == 48
         and plan["observers"]["target_text_supplied_to_observers"] is False
         and plan["observers"]["retries"] == 0,
         "FOUR_BLIND_OBSERVERS_ONLY")
    need(plan["flags"] == {
        "synthetic_only": True, "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "real_human_far_field_quality_claimed": False,
        "training_allowed": False, "promotion_allowed": False,
        "calibration_locked_test_opened": False,
    }, "NO_HUMAN_GOLD_TRAINING_OR_PROMOTION")
    return plan, raw


def source_check():
    plan, raw = load_plan()
    return {
        "source_check": "PASS",
        "plan_sha256": sha(raw),
        "donor_ids": [d["donor_id"] for d in plan["donors"]],
        "donor_roles": {"train": 1, "development": 1},
        "reserved_donor_audio_opened": 0,
        "planned_tts_calls": 4,
        "planned_observer_calls": 48,
        "planned_total_model_calls": 52,
        "new_model_calls": 0,
        "new_network_calls": 0,
        "kws_forwards": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    need(env.get("GITHUB_REPOSITORY") == plan["repository"]
         and env.get("GITHUB_EVENT_NAME") == "push"
         and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1",
         "PUBLIC_MAIN_FIRST_PUSH_ONLY")
    need(env.get("GITHUB_SHA") and HEX40.fullmatch(env["GITHUB_SHA"])
         and env.get("GITHUB_RUN_ID", "").isdigit(),
         "GITHUB_RUN_IDENTITY")
    if event is None:
        event = read_json(env["GITHUB_EVENT_PATH"])
    need(event.get("before") == plan["main_before"]
         and event.get("after") == env["GITHUB_SHA"]
         and event.get("forced") is False
         and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == plan["repository"]
         and event.get("repository", {}).get("private") is False,
         "EXACT_PUBLIC_MAIN_SUCCESSOR")
    return {
        "repository": env["GITHUB_REPOSITORY"], "event": env["GITHUB_EVENT_NAME"],
        "ref": env["GITHUB_REF"], "commit": env["GITHUB_SHA"],
        "run_id": env["GITHUB_RUN_ID"], "run_attempt": 1,
        "parent": event["before"],
    }


def audio_info(raw, *, expected_rate, expected_channels=1, expected_width=2):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        channels, width, rate, samples, comp = (
            stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
            stream.getnframes(), stream.getcomptype())
        pcm = stream.readframes(samples)
        need(not stream.readframes(1), "NO_EXTRA_WAV_FRAMES")
    need((channels, width, rate, comp) ==
         (expected_channels, expected_width, expected_rate, "NONE")
         and len(pcm) == samples * width, "EXACT_PCM_WAV_GEOMETRY")
    return pcm, samples


def verify_reference_archive(archive, output, plan=None):
    plan = load_plan()[0] if plan is None else plan
    archive = Path(archive)
    raw_archive = regular(archive, 8 * 1024 * 1024)
    need(sha(raw_archive) == plan["donor_reference"]["artifact_sha256"],
         "EXACT_DONOR_ARTIFACT_SHA256")
    root = Path(output)
    need(root.is_absolute() and root.parent.is_dir() and not root.exists(),
         "FRESH_DONOR_OUTPUT_ROOT")
    root.mkdir(mode=0o700)
    extracted = root / "reference"
    extracted.mkdir(mode=0o700)
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as zf:
        names = set(zf.namelist())
        need("MANIFEST.json" in names, "DONOR_MANIFEST_MISSING")
        manifest_raw = zf.read("MANIFEST.json")
        manifest = json.loads(manifest_raw, object_pairs_hook=unique_pairs)
        need(manifest.get("schema") == "english-synthetic-clone-reference-12-v1"
             and manifest.get("record_kind") == "SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY"
             and manifest.get("reference_audio_count") == 12
             and manifest.get("roles") == {
                 "train": 6, "development": 2, "calibration": 2, "locked-test": 2}
             and manifest.get("human_gold") is False
             and manifest.get("training_allowed") is False,
             "SYNTHETIC_REFERENCE_MANIFEST_CONTRACT")
        need(manifest.get("source", {}).get("run") ==
             plan["donor_reference"]["original_source_run_id"]
             and manifest.get("source", {}).get("artifact_id") ==
             plan["donor_reference"]["original_source_artifact_id"],
             "DONOR_ORIGINAL_SOURCE_LINEAGE")
        rows = {row.get("id"): row for row in manifest.get("rows", [])}
        need(len(rows) == 12, "TWELVE_DONOR_METADATA_ROWS")
        selected = []
        opened = []
        for donor in plan["donors"]:
            row = rows.get(donor["donor_id"])
            need(isinstance(row, dict)
                 and row.get("role") == donor["role"]
                 and row.get("source_group_id") == donor["source_group_id"]
                 and row.get("reference_text") == donor["reference_text"]
                 and row.get("reference_kind") == donor["reference_kind"]
                 and row.get("sample_rate") == donor["sample_rate"]
                 and row.get("training_allowed") is False
                 and row.get("human_gold") is False
                 and row.get("independent_human_source_claim") is False
                 and row.get("wav_sha256") == donor["reference_wav_sha256"]
                 and row.get("joined_pcm_sha256") == donor["joined_pcm_sha256"],
                 "DONOR_ROLE_OR_LINEAGE_MISMATCH:" + donor["donor_id"])
            name = safe_relative(donor["reference_path"])
            need(name in names, "SELECTED_DONOR_WAV_MISSING")
            wav = zf.read(name)
            need(sha(wav) == donor["reference_wav_sha256"], "DONOR_WAV_SHA256")
            pcm, samples = audio_info(wav, expected_rate=16000)
            need(sha(pcm) == donor["joined_pcm_sha256"]
                 and samples == row["pcm_samples"], "DONOR_PCM_LINEAGE")
            destination = extracted / (donor["donor_id"] + ".wav")
            with destination.open("xb") as stream:
                stream.write(wav)
                stream.flush()
                os.fsync(stream.fileno())
            selected.append({
                "donor_id": donor["donor_id"], "role": donor["role"],
                "source_group_id": donor["source_group_id"],
                "reference_path": name,
                "reference_wav_sha256": sha(wav),
                "reference_pcm_sha256": sha(pcm),
                "reference_samples": samples,
                "reference_text": donor["reference_text"],
                "reference_kind": donor["reference_kind"],
                "human_gold": False, "training_allowed": False,
            })
            opened.append(name)
        allowed = {d["reference_path"] for d in plan["donors"]}
        need(set(opened) == allowed
             and not any("calibration" in name.lower() or "locked" in name.lower()
                         for name in opened),
             "NO_RESERVED_REFERENCE_AUDIO_OPENED")
        result = {
            "schema": "english-kws-v7-voxcpm-donor-readback-v1",
            "status": "PASS_TWO_SELECTED_SYNTHETIC_DONORS",
            "artifact_id": plan["donor_reference"]["artifact_id"],
            "artifact_sha256": sha(raw_archive),
            "manifest_sha256": sha(manifest_raw),
            "selected_donors": selected,
            "selected_audio_members_opened": opened,
            "calibration_audio_opened": False,
            "locked_test_audio_opened": False,
            "human_gold": False,
            "training_allowed": False,
        }
    write_once(root / "DONORS.json", result)
    return result


def create_output_root(path):
    path = Path(path).absolute()
    need(path.parent.is_dir() and not path.exists()
         and not any(parent.is_symlink() for parent in path.parents),
         "NEW_OUTPUT_ROOT_ONLY")
    path.mkdir(mode=0o700)
    return path


def file_inventory(root):
    result = []
    for path in sorted(Path(root).rglob("*")):
        need(not path.is_symlink(), "NO_SYMLINK_OUTPUT")
        if path.is_file() and path.name != "FILES.json":
            raw = regular(path, 32 * 1024 * 1024)
            result.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw), "sha256": sha(raw),
            })
    return result


def generate(donors_root, output, voxcpm_tts, model_path, ffmpeg, model_lock):
    plan, plan_raw = load_plan()
    run = guard(plan)
    donors_root = Path(donors_root)
    donor_readback = read_json(donors_root / "DONORS.json")
    need(donor_readback.get("status") == "PASS_TWO_SELECTED_SYNTHETIC_DONORS"
         and donor_readback.get("calibration_audio_opened") is False
         and donor_readback.get("locked_test_audio_opened") is False,
         "VERIFIED_SELECTED_DONOR_PACKET_REQUIRED")
    binary = Path(voxcpm_tts)
    model = Path(model_path)
    ffmpeg_bin = shutil.which(str(ffmpeg)) if "/" not in str(ffmpeg) else str(ffmpeg)
    need(binary.is_file() and not binary.is_symlink()
         and model.is_file() and not model.is_symlink()
         and ffmpeg_bin is not None and Path(ffmpeg_bin).is_file(),
         "VERIFIED_VOXCPM_RUNTIME_MODEL_AND_FFMPEG_REQUIRED")
    model_lock = read_json(model_lock)
    need(model_lock.get("model_repository") == plan["model"]["repository"]
         and model_lock.get("model_revision") == plan["model"]["revision"]
         and model_lock.get("model_license") == "Apache-2.0"
         and model_lock.get("engine_repository") == plan["generator"]["repository"]
         and model_lock.get("engine_commit") == plan["generator"]["commit"]
         and model_lock.get("quantization") == "Q8_0+AudioVAE-F16"
         and model_lock.get("model_gguf_sha256") == sha_file(model, 8 * 1024**3),
         "MODEL_LOCK_AND_GGUF_SHA256")
    root = create_output_root(output)
    (root / "audio").mkdir(mode=0o700)
    (root / "base").mkdir(mode=0o700)
    (root / "calls").mkdir(mode=0o700)
    donor_rows = {row["donor_id"]: row for row in donor_readback["selected_donors"]}
    rows = []
    calls = []
    cell_specs = []
    for donor in plan["donors"]:
        for candidate, candidate_id in (("VOSEN", "hey-vosen-v7"), ("MERO", "hey-mero-v7")):
            phrase = plan["candidates"][candidate]
            base_id = donor["donor_id"] + "-" + candidate.lower()
            cell_specs.append((donor, candidate, candidate_id, phrase, base_id))
    need(len(cell_specs) == 4, "EXACT_FOUR_BASE_TTS_REQUESTS")
    for donor, candidate, candidate_id, phrase, base_id in cell_specs:
        donor_info = donor_rows[donor["donor_id"]]
        prompt_audio = donors_root / "reference" / (donor["donor_id"] + ".wav")
        prompt_raw = regular(prompt_audio, MAX_WAV_BYTES)
        need(sha(prompt_raw) == donor["reference_wav_sha256"], "DONOR_BYTES_CHANGED_BEFORE_TTS")
        folder = root / "calls" / base_id
        folder.mkdir(mode=0o700)
        base_wav = root / "base" / (base_id + ".wav")
        command = [
            str(binary), "--model-path", str(model),
            "--prompt-audio", str(prompt_audio),
            "--prompt-text", PROMPT_TEXT,
            "--text", phrase["phrase"],
            "--output", str(base_wav),
            "--backend", "cpu", "--threads", "8",
            "--cfg-value", "2.0", "--inference-timesteps", "10",
        ]
        started = {
            "id": base_id, "donor_id": donor["donor_id"], "role": donor["role"],
            "candidate": candidate, "candidate_id": candidate_id,
            "phrase": phrase["phrase"], "ipa": phrase["ipa"],
            "prompt_audio_sha256": sha(prompt_raw),
            "model_gguf_sha256": model_lock["model_gguf_sha256"],
            "command": command, "attempt": 1, "retry_count": 0,
            "target_text_supplied_to_observers": False,
            "training_allowed": False,
        }
        write_once(folder / "STARTED.json", started)
        call = {"id": base_id, "attempts": 1, "returned": False, "status": "FAILED"}
        stdout_path, stderr_path = folder / "stdout.txt", folder / "stderr.txt"
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                proc = subprocess.run(command, stdout=stdout, stderr=stderr,
                                      timeout=900, check=False, env={
                                          "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                                          "HOME": os.environ.get("HOME", "/tmp"),
                                          "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                                          "OMP_NUM_THREADS": "8",
                                          "OPENBLAS_NUM_THREADS": "1",
                                          "MKL_NUM_THREADS": "1",
                                      })
            if proc.returncode != 0:
                raise RuntimeError("VOXCPM_TTS_EXIT:" + str(proc.returncode))
            base_raw = regular(base_wav, MAX_WAV_BYTES)
            base_pcm, base_samples = audio_info(base_raw, expected_rate=44100)
            need(base_samples > 0, "EMPTY_VOXCPM_BASE_AUDIO")
            call.update(returned=True, status="BASE_AUDIO_SAVED",
                        returncode=proc.returncode, base_wav_sha256=sha(base_raw),
                        base_pcm_sha256=sha(base_pcm), base_samples=base_samples,
                        sample_rate=44100)
            write_once(folder / "RETURNED.json", call)
        except Exception as exc:
            call.update(status="FAILED_NO_RETRY",
                        error=type(exc).__name__ + ":" + str(exc)[:500])
            write_once(folder / "FAILED.json", call)
            calls.append(call)
            continue
        calls.append(call)
        for speed in plan["synthesis"]["deterministic_tempo_variants"]:
            speed_label = {0.92: "slow", 1.0: "normal", 1.08: "fast"}[speed]
            cell_id = candidate.lower() + "-" + speed_label
            row_id = donor["donor_id"] + "-" + cell_id
            final_wav = root / "audio" / (row_id + ".wav")
            transform_dir = folder / speed_label
            transform_dir.mkdir(mode=0o700)
            transform = [
                ffmpeg_bin, "-hide_banner", "-nostdin", "-v", "error", "-n",
                "-i", str(base_wav),
                "-filter:a", "atempo=" + str(speed) + ",aresample=16000",
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
                str(final_wav),
            ]
            write_once(transform_dir / "STARTED.json", {
                "id": row_id, "parent_base_id": base_id,
                "parent_base_wav_sha256": call["base_wav_sha256"],
                "tempo": speed, "command": transform,
                "retry_count": 0, "no_trim": True, "no_crop": True,
                "no_normalize": True, "training_allowed": False,
            })
            row = {
                "id": row_id, "donor_id": donor["donor_id"],
                "source_group_id": donor["source_group_id"],
                "source_speaker_id": donor["donor_id"], "role": donor["role"],
                "candidate": candidate, "candidate_id": candidate_id,
                "cell_id": cell_id, "requested_text": phrase["phrase"],
                "requested_class": 1 if candidate == "VOSEN" else 2,
                "kind": "target", "candidate_labels": {
                    "VOSEN": 1 if candidate == "VOSEN" else 0,
                    "MERO": 1 if candidate == "MERO" else 0,
                },
                "pronunciation_hypothesis": (
                    "heɪ vˈoʊsən" if candidate == "VOSEN" else "heɪ mˈɛɹoʊ"
                ),
                "ipa": phrase["ipa"], "tempo": speed,
                "parent_base_id": base_id,
                "parent_base_wav_sha256": call["base_wav_sha256"],
                "generation_status": "FAILED_NO_RETRY",
                "training_allowed": False, "human_gold": False,
            }
            try:
                with (transform_dir / "stdout.txt").open("xb") as stdout, (transform_dir / "stderr.txt").open("xb") as stderr:
                    proc = subprocess.run(transform, stdout=stdout, stderr=stderr,
                                          timeout=120, check=False)
                need(proc.returncode == 0, "TEMPO_TRANSFORM_EXIT:" + str(proc.returncode))
                wav_raw = regular(final_wav, MAX_WAV_BYTES)
                pcm, samples = audio_info(wav_raw, expected_rate=16000)
                values = [sample[0] for sample in __import__("struct").iter_unpack("<h", pcm)]
                need(samples > 0, "EMPTY_FINAL_PCM")
                clipped = sum(value in (-32768, 32767) for value in values)
                clip_ratio = clipped / samples
                square_sum = math.fsum((value / 32768.0) ** 2 for value in values)
                rms = math.sqrt(square_sum / samples)
                rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
                row.update(
                    generation_status="success" if (
                        6400 <= samples <= 28672 and clip_ratio <= 0.001 and rms_dbfs > -55.0
                    ) else "MODEL_INPUT_GEOMETRY_OR_CLIPPING_QUARANTINED",
                    wav_path="audio/" + row_id + ".wav",
                    pcm_path="audio/" + row_id + ".pcm",
                    wav_sha256=sha(wav_raw), pcm_sha256=sha(pcm),
                    pcm_samples=samples, sample_rate=16000, channels=1,
                    sample_width_bytes=2, clipped_samples=clipped,
                    clip_ratio=clip_ratio, rms_dbfs=rms_dbfs,
                    max_abs_pcm=max(abs(value) for value in values),
                    transform_returncode=proc.returncode,
                )
                with (root / row["pcm_path"]).open("xb") as stream:
                    stream.write(pcm)
                    stream.flush()
                    os.fsync(stream.fileno())
                write_once(transform_dir / "RETURNED.json", {
                    "id": row_id, "status": row["generation_status"],
                    "wav_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"],
                    "pcm_samples": samples, "clip_ratio": clip_ratio,
                    "rms_dbfs": rms_dbfs, "retry_count": 0,
                })
            except Exception as exc:
                row.update(generation_status="FAILED_NO_RETRY",
                           reason=type(exc).__name__ + ":" + str(exc)[:500])
                write_once(transform_dir / "FAILED.json", {
                    "id": row_id, "status": "FAILED_NO_RETRY",
                    "reason": row["reason"], "retry_count": 0,
                })
            rows.append(row)
    expected_ids = {
        donor["donor_id"] + "-" + candidate.lower() + "-" + speed
        for donor in plan["donors"] for candidate in ("VOSEN", "MERO")
        for speed in ("slow", "normal", "fast")
    }
    present = {row["id"] for row in rows}
    for missing in sorted(expected_ids - present):
        donor = next(item for item in plan["donors"]
                     if missing.startswith(item["donor_id"] + "-"))
        donor_id = donor["donor_id"]
        cell_id = missing[len(donor_id) + 1:]
        candidate = "VOSEN" if cell_id.startswith("vosen-") else "MERO"
        phrase = plan["candidates"][candidate]
        rows.append({
            "id": missing, "donor_id": donor_id,
            "source_group_id": donor["source_group_id"],
            "source_speaker_id": donor_id, "role": donor["role"],
            "candidate": candidate,
            "candidate_id": phrase["candidate_id"],
            "cell_id": cell_id, "requested_text": phrase["phrase"],
            "requested_class": 1 if candidate == "VOSEN" else 2,
            "kind": "target",
            "candidate_labels": {"VOSEN": int(candidate == "VOSEN"),
                                 "MERO": int(candidate == "MERO")},
            "pronunciation_hypothesis": (
                "heɪ vˈoʊsən" if candidate == "VOSEN" else "heɪ mˈɛɹoʊ"
            ),
            "ipa": phrase["ipa"], "generation_status": "NOT_STARTED",
            "training_allowed": False, "human_gold": False,
        })
    rows.sort(key=lambda row: row["id"])
    hashes = [row.get("pcm_sha256") for row in rows if row.get("pcm_sha256")]
    duplicate_hashes = {value for value, count in Counter(hashes).items() if count > 1}
    for row in rows:
        if row.get("pcm_sha256") in duplicate_hashes:
            row["generation_status"] = "DUPLICATE_PCM_QUARANTINED"
    status = "COMPLETE" if len(rows) == 12 and all(
        row["generation_status"] == "success" for row in rows) else "INCOMPLETE"
    source = {
        "schema": SCHEMA, "status": status,
        "run": run, "plan_sha256": sha(plan_raw),
        "donor_artifact": plan["donor_reference"],
        "donor_manifest_sha256": donor_readback["manifest_sha256"],
        "donors": donor_readback["selected_donors"],
        "generator": plan["generator"], "converter": plan["converter"],
        "model": plan["model"], "model_lock": model_lock,
        "rows": rows, "base_tts_calls": calls,
        "denominator": 12, "actual_base_tts_attempts": sum(c["attempts"] for c in calls),
        "base_tts_returns": sum(c["returned"] for c in calls),
        "successful_final_clips": sum(r["generation_status"] == "success" for r in rows),
        "failed_or_quarantined_final_clips": sum(r["generation_status"] not in ("success","NOT_STARTED") for r in rows),
        "not_started_final_clips": sum(r["generation_status"] == "NOT_STARTED" for r in rows),
        "reserved_donor_audio_opened": 0, "calibration_audio_opened": False,
        "locked_test_audio_opened": False, "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "training_allowed": False, "promotion_allowed": False,
        "new_kws_forwards": 0, "optimizer_updates": 0, "retry_count": 0,
        "naturalness_review": "PENDING_BLIND_HUMAN_LISTENING",
    }
    write_once(root / "SOURCE.json", source)
    inventory = {"schema": SCHEMA + "/files", "self_excluded": "FILES.json",
                 "files": file_inventory(root)}
    write_once(root / "FILES.json", inventory)
    return source


def readback(root):
    root = Path(root)
    source = read_json(root / "SOURCE.json")
    inventory = read_json(root / "FILES.json")
    need(inventory.get("schema") == SCHEMA + "/files"
         and inventory.get("self_excluded") == "FILES.json"
         and inventory.get("files") == file_inventory(root),
         "SOURCE_FILE_INVENTORY_CLOSURE")
    need(source.get("schema") == SCHEMA and source.get("denominator") == 12
         and len(source.get("rows", [])) == 12
         and source.get("retry_count") == 0
         and source.get("training_allowed") is False
         and source.get("promotion_allowed") is False
         and source.get("calibration_audio_opened") is False
         and source.get("locked_test_audio_opened") is False,
         "SOURCE_TERMINAL_FLAGS_AND_DENOMINATOR")
    for row in source["rows"]:
        if row.get("generation_status") != "success":
            continue
        wav = regular(within(root, row["wav_path"]), MAX_WAV_BYTES)
        pcm = regular(within(root, row["pcm_path"]), MAX_PCM_BYTES)
        decoded, samples = audio_info(wav, expected_rate=16000)
        need(decoded == pcm and sha(wav) == row["wav_sha256"]
             and sha(pcm) == row["pcm_sha256"] and samples == row["pcm_samples"],
             "FINAL_WAV_PCM_HASH_READBACK")
    return {
        "status": source["status"],
        "denominator": 12,
        "successful_final_clips": sum(r["generation_status"] == "success" for r in source["rows"]),
        "actual_base_tts_attempts": source["actual_base_tts_attempts"],
        "base_tts_returns": source["base_tts_returns"],
        "reserved_donor_audio_opened": source["reserved_donor_audio_opened"],
        "training_allowed": False, "optimizer_updates": 0,
        "naturalness_review": "PENDING_BLIND_HUMAN_LISTENING",
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("source-check")
    prep = sub.add_parser("prepare-donors")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--donors", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--voxcpm-tts", required=True)
    gen.add_argument("--model-path", required=True)
    gen.add_argument("--ffmpeg", default="ffmpeg")
    gen.add_argument("--model-lock", type=Path, required=True)
    check = sub.add_parser("readback")
    check.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source-check":
        print(json.dumps(source_check(), sort_keys=True))
    elif args.action == "prepare-donors":
        print(json.dumps(verify_reference_archive(args.archive, args.output), sort_keys=True))
    elif args.action == "generate":
        result = generate(args.donors, args.output, args.voxcpm_tts,
                          args.model_path, args.ffmpeg, args.model_lock)
        print(json.dumps({
            "status": result["status"], "base_tts_calls": result["actual_base_tts_attempts"],
            "base_tts_returns": result["base_tts_returns"],
            "successful_final_clips": result["successful_final_clips"],
            "training_allowed": False, "optimizer_updates": 0,
        }, sort_keys=True))
    else:
        print(json.dumps(readback(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
