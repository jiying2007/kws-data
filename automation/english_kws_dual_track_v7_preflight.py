"""Metadata-only Kokoro model, voice and raw-IPA preflight; no weights or audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

from huggingface_hub import hf_hub_download
import english_kws_dual_track_v7_recipe as recipe


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_identity():
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "ref": os.environ.get("GITHUB_REF"),
        "sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def fetch_github_license():
    url = ("https://raw.githubusercontent.com/hexgrad/kokoro/" +
           recipe.MODEL["source_commit"] + "/LICENSE")
    request = urllib.request.Request(url, headers={"User-Agent": "kws-kokoro-source-preflight"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("KOKORO_LICENSE_SIZE_LIMIT")
    return raw


def run(output_dir: Path):
    source_plan, registration = recipe.source_documents()
    model = recipe.MODEL
    repo = model["repository"]
    revision = model["revision"]

    config_path = Path(hf_hub_download(
        repo_id=repo, filename=model["config_path"], revision=revision, token=False))
    voices_path = Path(hf_hub_download(
        repo_id=repo, filename="VOICES.md", revision=revision, token=False))
    readme_path = Path(hf_hub_download(
        repo_id=repo, filename="README.md", revision=revision, token=False))

    config_raw = config_path.read_bytes()
    voices_raw = voices_path.read_bytes()
    readme_raw = readme_path.read_bytes()
    license_raw = fetch_github_license()
    config = json.loads(config_raw)
    raw_vocab = config.get("vocab")
    if not isinstance(raw_vocab, dict) or not raw_vocab:
        raise ValueError("KOKORO_CONFIG_VOCAB_REQUIRED")
    # KModel reverses the config's id->phoneme mapping into phoneme->id.
    vocab = {phoneme: int(token_id) for token_id, phoneme in raw_vocab.items()}

    required_chars = sorted({char for cell in source_plan["cells"] for char in cell["raw_ipa"]})
    missing_chars = [char for char in required_chars if char not in vocab]
    if missing_chars:
        raise ValueError("KOKORO_RAW_IPA_VOCAB_MISSING:" + repr(missing_chars))

    voices = [item["voice_id"] for item in source_plan["source_selection"]["roles"]]
    if len(voices) != 8 or len(set(voices)) != 8:
        raise ValueError("EXACT_EIGHT_UNIQUE_TRAIN_DEV_VOICES")
    voice_text = voices_raw.decode("utf-8")
    missing_voices = [voice for voice in voices if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(voice) + r"(?![A-Za-z0-9_])", voice_text)]
    if missing_voices:
        raise ValueError("KOKORO_VOICE_NOT_IN_PINNED_VOICES_MD:" + repr(missing_voices))

    readme_text = readme_raw.decode("utf-8", errors="replace")
    license_match = re.search(r"(?im)^license:\s*apache-2\.0\s*$", readme_text)
    if license_match is None:
        raise ValueError("KOKORO_MODEL_CARD_APACHE_2_LICENSE_METADATA")
    license_text = license_raw.decode("utf-8", errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("KOKORO_CODE_APACHE_2_LICENSE_TEXT")

    selected = {str(item["speaker_index"]): item["voice_id"]
                for item in source_plan["source_selection"]["roles"]}
    registration_selected = registration["source_map_evidence"]["selected_voice_ids"]
    if selected != registration_selected:
        raise ValueError("KOKORO_SOURCE_MAP_REGISTRATION_MISMATCH")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in (("config.json", config_raw), ("VOICES.md", voices_raw),
                      ("README.md", readme_raw), ("KOKORO_LICENSE", license_raw)):
        (output_dir / name).write_bytes(raw)

    report = {
        "schema": "english-kws-dual-track-v7-kokoro-source-preflight-v1",
        "status": "PASS",
        "run": run_identity(),
        "source_plan_sha256": hashlib.sha256(recipe.SOURCE_PLAN_PATH.read_bytes()).hexdigest(),
        "source_plan_blob_sha1": recipe.SOURCE_PLAN_BLOB_SHA1,
        "registration_blob_sha1": recipe.REGISTRATION_BLOB_SHA1,
        "model": {
            "repository": repo, "revision": revision,
            "checkpoint_path": model["checkpoint_path"],
            "expected_checkpoint_sha256": model["checkpoint_sha256"],
            "config_sha256": sha(config_raw),
            "voices_index_sha256": sha(voices_raw),
            "model_card_sha256": sha(readme_raw),
            "native_sample_rate": model["native_sample_rate"],
            "output_sample_rate": model["output_sample_rate"],
            "license": "Apache-2.0",
        },
        "source_map": {
            "selected_voice_ids": selected,
            "selected_count": len(selected),
            "role_counts": {"train": 6, "development": 2},
            "voice_id_unique": True,
            "human_speaker_identity_claimed": False,
        },
        "phoneme_map": {
            "required_characters": required_chars,
            "missing_characters": missing_chars,
            "covered_count": len(required_chars),
        },
        "license": {
            "model_repository": "hexgrad/Kokoro-82M",
            "model_revision": revision,
            "model_license": "Apache-2.0",
            "inference_code_repository": "hexgrad/kokoro",
            "inference_code_commit": model["source_commit"],
            "inference_code_license": "Apache-2.0",
            "inference_code_license_sha256": sha(license_raw),
            "voice_asset_license_scope": "Pinned model repository metadata; no human speaker consent or commercial derivative-rights claim is made",
            "generated_audio_derivative_clearance": "NOT_ASSERTED_LEGAL_REVIEW_REQUIRED_BEFORE_COMMERCIAL_PRODUCT_CLAIM",
        },
        "no_execution": {
            "model_weights_downloaded": False,
            "voice_weights_downloaded": False,
            "audio_downloaded": False,
            "tts_calls": 0,
            "asr_calls": 0,
            "phoneme_observer_calls": 0,
            "kws_forward_calls": 0,
            "optimizer_updates": 0,
            "training_allowed": False,
        },
    }
    (output_dir / "source-license-preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\\n",
        encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_voice_count": len(voices),
                      "raw_ipa_characters": len(required_chars),
                      "model_weights_downloaded": False, "audio_generated": False}, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
