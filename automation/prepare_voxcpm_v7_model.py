"""Prepare the exact pinned VoxCPM 1.5 CPU model for the V7 pilot.

This stage downloads only the Apache-2.0 upstream revision, verifies source
weights, builds F32 GGUF, quantizes Q8_0 + AudioVAE-F16, and emits a complete
runtime lock. It never synthesizes audio or calls an observer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "automation/english_kws_v7_voxcpm_pilot_plan.json"
RUNTIME_REPO = "bluryar/VoxCPM.cpp"
RUNTIME_COMMIT = "e5b89534e379c79b923c9969b79c3858d140bedb"
MODEL_REPO = "openbmb/VoxCPM1.5"
MODEL_REVISION = "09c7a389438c27100dfa4e6cda1f8f0dd0d2fbfa"
MODEL_LICENSE = "Apache-2.0"
EXPECTED_MODEL_FILES = {
    "model.safetensors": "7e97ee2cd2ca891662f7da368921df08368a616410afe67461253e4dcd3823cc",
    "audiovae.pth": "8160987a7f10acd4c49b72a9a3e974c0b5c1c786538cd4bcaac5f2df388c0f14",
}
GGUF_VERSION = "0.17.1"
GGUF_WHEEL_SHA256 = "7bc5aa7eeb1931f7d39b48fdc5b38fda6b294b9dca75cf607ac69557840a3943"
VOXCPM_VERSION = "1.5.0"
VOXCPM_SDIST_SHA256 = "6b623812006d35f16d9dc93c3d1bd69a8440ae3970d0d2b8ff27954aae607f40"
IMPORT_LINE = "from voxcpm.modules.audiovae.audio_vae import AudioVAEConfig"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value) -> None:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def source_check():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    need(plan["model"]["repository"] == MODEL_REPO
         and plan["model"]["revision"] == MODEL_REVISION
         and plan["model"]["license"] == MODEL_LICENSE
         and plan["model"]["files"] == EXPECTED_MODEL_FILES,
         "EXACT_OFFICIAL_MODEL_REVISION_AND_WEIGHT_HASHES")
    need(plan["generator"]["repository"] == RUNTIME_REPO
         and plan["generator"]["commit"] == RUNTIME_COMMIT
         and plan["generator"]["license"] == MODEL_LICENSE,
         "EXACT_VOXCPM_CPP_SOURCE_AND_LICENSE")
    need(plan["model"]["quantization"] == "Q8_0+AudioVAE-F16"
         and plan["synthesis"]["backend"] == "cpu"
         and plan["synthesis"]["threads"] == 8
         and plan["synthesis"]["retry_badcase"] is False,
         "FIXED_CPU_Q8_AUDIOVAE_CONFIGURATION")
    return {
        "source_check": "PASS",
        "model_repository": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "generator_repository": RUNTIME_REPO,
        "generator_commit": RUNTIME_COMMIT,
        "quantization": "Q8_0+AudioVAE-F16",
        "network_calls": 0,
        "model_downloads": 0,
        "model_forwards": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def prepare(runtime_dir: Path, model_dir: Path, output_dir: Path,
            tts_binary: Path, quantizer: Path) -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    source_check()
    runtime_dir, model_dir, output_dir = map(Path, (runtime_dir, model_dir, output_dir))
    tts_binary, quantizer = map(Path, (tts_binary, quantizer))
    need(not output_dir.exists() and not model_dir.exists(), "FRESH_MODEL_PREPARATION_DIRECTORIES")
    need(subprocess.check_output(["git", "-C", str(runtime_dir), "rev-parse", "HEAD"],
                                 text=True).strip() == RUNTIME_COMMIT,
         "EXACT_VOXCPM_CPP_COMMIT")
    for path in (runtime_dir / "LICENSE", tts_binary, quantizer):
        need(path.is_file() and not path.is_symlink(), "REGULAR_RUNTIME_INPUT:" + str(path))
    output_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    runtime_license_sha = sha_file(runtime_dir / "LICENSE")
    converter = runtime_dir / "scripts" / "convert_voxcpm_to_gguf.py"
    need(converter.is_file() and not converter.is_symlink(), "PINNED_GGUF_CONVERTER_PRESENT")
    original_converter_sha = sha_file(converter)
    converter_text = converter.read_text(encoding="utf-8")
    need(converter_text.count(IMPORT_LINE) == 1, "EXACT_AUDIOVAE_IMPORT_PATCH_POINT")
    # The official checkpoint carries audio_vae_config. Keep conversion independent
    # from importing the entire upstream training package; fail closed if it is absent.
    patched = converter_text.replace(
        IMPORT_LINE,
        "class AudioVAEConfig:\n"
        "    def model_dump(self):\n"
        "        raise RuntimeError('pinned VoxCPM1.5 config must contain audio_vae_config')",
    )
    converter.write_text(patched, encoding="utf-8")
    patched_converter_sha = sha_file(converter)

    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(MODEL_REPO, revision=MODEL_REVISION)
    need(str(info.sha) == MODEL_REVISION, "IMMUTABLE_HF_MODEL_REVISION")
    card = getattr(info, "cardData", None) or {}
    tags = set(getattr(info, "tags", None) or [])
    license_value = card.get("license") if isinstance(card, dict) else None
    need(license_value in ("apache-2.0", "Apache-2.0")
         or "license:apache-2.0" in tags,
         "OFFICIAL_MODEL_APACHE_2_LICENSE")
    downloaded = Path(snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=str(model_dir),
        allow_patterns=["config.json", "*.json", "*.py", "*.txt", "README.md",
                        "LICENSE", "model.safetensors", "audiovae.pth"],
        max_workers=2, token=False,
    ))
    config = json.loads((downloaded / "config.json").read_text(encoding="utf-8"))
    need(isinstance(config.get("audio_vae_config"), dict),
         "OFFICIAL_CONFIG_HAS_AUDIO_VAE_CONFIG")
    model_files = {}
    for name, expected_sha in EXPECTED_MODEL_FILES.items():
        path = downloaded / name
        need(path.is_file() and not path.is_symlink(), "OFFICIAL_WEIGHT_PRESENT:" + name)
        actual_sha = sha_file(path)
        need(actual_sha == expected_sha, "OFFICIAL_WEIGHT_SHA256:" + name)
        model_files[name] = {"bytes": path.stat().st_size, "sha256": actual_sha}

    f32_path = output_dir / "voxcpm1.5-f32.gguf"
    q8_path = output_dir / "voxcpm1.5-q8_0-audiovae-f16.gguf"
    subprocess.run([sys.executable, str(converter), str(downloaded),
                    "--output", str(f32_path)], check=True, timeout=3600)
    need(f32_path.is_file() and not f32_path.is_symlink(), "F32_GGUF_CREATED")
    f32_sha = sha_file(f32_path)
    subprocess.run([str(quantizer), "--input", str(f32_path), "--output", str(q8_path),
                    "--type", "Q8_0", "--audio-vae-mode", "f16", "--threads", "4"],
                   check=True, timeout=1800)
    need(q8_path.is_file() and not q8_path.is_symlink(), "Q8_AUDIOVAE_GGUF_CREATED")
    q8_sha = sha_file(q8_path)
    importlib.metadata.version("gguf")
    need(importlib.metadata.version("gguf") == GGUF_VERSION, "PINNED_GGUF_PACKAGE_VERSION")
    ffmpeg = shutil.which("ffmpeg")
    need(ffmpeg is not None, "FFMPEG_PRESENT")
    ffmpeg_path = Path(ffmpeg).resolve()
    lock = {
        "schema": "english-v7-voxcpm-model-lock-v1",
        "model_repository": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "model_files": model_files,
        "cpp_repository": RUNTIME_REPO,
        "cpp_commit": RUNTIME_COMMIT,
        "cpp_license": MODEL_LICENSE,
        "runtime_license_sha256": runtime_license_sha,
        "converter_original_sha256": original_converter_sha,
        "converter_patched_sha256": patched_converter_sha,
        "converter_patch": "stub AudioVAEConfig import; require checkpoint audio_vae_config",
        "voxcpm_pypi_version": VOXCPM_VERSION,
        "voxcpm_pypi_sdist_sha256": VOXCPM_SDIST_SHA256,
        "gguf_package_version": GGUF_VERSION,
        "gguf_package_wheel_sha256": GGUF_WHEEL_SHA256,
        "backend": "cpu",
        "threads": 8,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "retry_badcase": False,
        "gpu_used": False,
        "f32_gguf_sha256": f32_sha,
        "f32_gguf_bytes": f32_path.stat().st_size,
        "q8_0_audiovae_f16_gguf_sha256": q8_sha,
        "q8_0_audiovae_f16_gguf_bytes": q8_path.stat().st_size,
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "ffmpeg_binary_sha256": sha_file(ffmpeg_path),
        "tts_binary_sha256": sha_file(tts_binary),
        "quantizer_binary_sha256": sha_file(quantizer),
    }
    validate = __import__("english_v7_voxcpm_pilot")
    validate.validate_model_lock(lock, q8_path)
    write_once(output_dir / "MODEL_LOCK.json", lock)
    # Retain only the quantized model and its lock; do not publish source weights
    # or the F32 intermediate.
    f32_path.unlink()
    shutil.rmtree(downloaded)
    return {"model_path": str(q8_path), "model_lock": str(output_dir / "MODEL_LOCK.json"),
            "model_lock_sha256": sha_file(output_dir / "MODEL_LOCK.json"),
            "q8_0_audiovae_f16_gguf_sha256": q8_sha,
            "q8_0_audiovae_f16_gguf_bytes": q8_path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tts-binary", type=Path)
    parser.add_argument("--quantizer", type=Path)
    args = parser.parse_args()
    if args.source_check:
        print(json.dumps(source_check(), sort_keys=True))
        return
    need(all((args.runtime_dir, args.model_dir, args.output_dir, args.tts_binary, args.quantizer)),
         "ALL_MODEL_PREPARATION_PATHS_REQUIRED")
    print(json.dumps(prepare(args.runtime_dir, args.model_dir, args.output_dir,
                             args.tts_binary, args.quantizer), sort_keys=True))


if __name__ == "__main__":
    main()
