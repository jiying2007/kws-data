"""Metadata-only V7 source, phoneme-map and license preflight.

This gate downloads only the pinned Piper config and license/provenance text.
It does not download model weights, synthesize audio, run ASR/phoneme observers,
run KWS, or train/update weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import english_kws_dual_track_v7_recipe as recipe
import english_synthetic192_recipe as history

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_URL = (
    "https://huggingface.co/" + recipe.MODEL["repository"] + "/resolve/"
    + recipe.MODEL["revision"] + "/" + recipe.MODEL["config_path"]
)
MODEL_CARD_URL = (
    "https://huggingface.co/" + recipe.MODEL["repository"] + "/resolve/"
    + recipe.MODEL["revision"] + "/" + recipe.MODEL_CARD["path"]
)
ENGINE_SETUP_URL = (
    "https://raw.githubusercontent.com/" + recipe.ENGINE["repository"] + "/"
    + recipe.ENGINE["commit"] + "/setup.py"
)
ENGINE_COPYING_URL = (
    "https://raw.githubusercontent.com/" + recipe.ENGINE["repository"] + "/"
    + recipe.ENGINE["commit"] + "/COPYING"
)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw):
    header = b"blob " + str(len(raw)).encode("ascii") + b"\\0"
    return hashlib.sha1(header + raw).hexdigest()


def fetch(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "kws-v7-provenance-preflight/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        need(response.status == 200, "HTTP_STATUS")
        raw = response.read(limit + 1)
    need(0 < len(raw) <= limit, "REMOTE_FILE_SIZE")
    return raw


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write_once(path, raw):
    path = Path(path)
    need(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "CREATE_ONCE_FILE")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def identity():
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "ref": os.environ.get("GITHUB_REF"),
        "sha": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output_dir.absolute()
    need(not output.exists() and output.parent.is_dir(), "NEW_PREFLIGHT_DIRECTORY")
    output.mkdir(mode=0o700)

    source_plan_path = HERE / "english_kws_dual_track_v7_source_plan.json"
    registration_path = HERE / "english_kws_dual_track_v7_registration.json"
    source_plan_raw = source_plan_path.read_bytes()
    registration_raw = registration_path.read_bytes()
    source_plan = json.loads(source_plan_raw, object_pairs_hook=unique)
    registration = json.loads(registration_raw, object_pairs_hook=unique)

    need(git_blob_sha1(source_plan_raw) == recipe.SOURCE_PLAN_BLOB_SHA1, "SOURCE_PLAN_BLOB_PIN")
    need(git_blob_sha1(registration_raw) == recipe.REGISTRATION_BLOB_SHA1, "REGISTRATION_BLOB_PIN")
    source_plan_sha256 = sha(source_plan_raw)
    need(source_plan_sha256 == registration["source_map_evidence"]["source_plan_sha256"],
         "SOURCE_PLAN_CONTENT_PIN")

    config_raw = fetch(CONFIG_URL, 1024 * 1024)
    need(len(config_raw) == recipe.MODEL["config_bytes"], "PIPER_CONFIG_BYTES")
    need(hashlib.md5(config_raw).hexdigest() == recipe.MODEL["config_md5"], "PIPER_CONFIG_MD5")
    config = json.loads(config_raw, object_pairs_hook=unique)
    need(config.get("num_speakers") == recipe.MODEL["num_speakers"] == 904, "PIPER_SPEAKER_COUNT")\n    need(config.get("audio", {}).get("sample_rate") == recipe.MODEL["native_sample_rate"] and\n         config.get("espeak", {}).get("voice") == recipe.MODEL["phonemizer_voice"], "PIPER_AUDIO_PHONEMIZER_GEOMETRY")\n    need(len(source_plan["cells"]) == 12 and len(recipe.recipe()["rows"]) == recipe.SOURCE_COUNT, "V7_RECIPE_GEOMETRY")
    speaker_map = config.get("speaker_id_map")
    need(type(speaker_map) is dict and len(speaker_map) == 904, "PIPER_SPEAKER_MAP")
    inverse = {index: speaker for speaker, index in speaker_map.items()}
    need(len(inverse) == 904 and set(inverse) == set(range(904)), "PIPER_SPEAKER_MAP_COVERAGE")

    used_indices = (
        set(range(58)) | set(range(400, 496))
        | {index for index, _speaker in history.NEW_VOICES}
    )
    used_speaker_ids = set(history.OLD_LIBRI_IDS) | set(history.OLD_PIPER_IDS)
    used_speaker_ids.update(int(inverse[index][1:]) for index in used_indices if index in inverse)
    selected = registration["source_map_evidence"]["selected_speaker_ids"]
    expected_indices = [item["speaker_index"] for item in source_plan["source_selection"]["roles"]]
    need(len(expected_indices) == 12 and len(set(expected_indices)) == 12, "V7_SOURCE_INDEX_COUNT")
    need(set(selected) == {str(index) for index in expected_indices}, "V7_SOURCE_INDEX_KEYS")
    selected_ids = {}
    for index in expected_indices:
        speaker_id = selected[str(index)]
        need(type(speaker_id) is str and speaker_id.startswith("p") and speaker_id[1:].isdigit(),
             "V7_SOURCE_SPEAKER_ID")
        need(speaker_map.get(speaker_id) == index, "V7_SOURCE_MAP_BINDING")
        need(int(speaker_id[1:]) not in used_speaker_ids, "HISTORICAL_SPEAKER_EXCLUSION")
        selected_ids[str(index)] = speaker_id
    need(len(set(selected_ids.values())) == 12, "V7_SPEAKER_ID_UNIQUENESS")

    phoneme_map = config.get("phoneme_id_map")
    need(type(phoneme_map) is dict and phoneme_map, "PIPER_PHONEME_MAP")
    required_chars = sorted({char for cell in source_plan["cells"] for char in cell["raw_ipa"]})
    missing_chars = [char for char in required_chars if char not in phoneme_map]
    need(not missing_chars, "PIPER_RAW_IPA_COVERAGE:" + repr(missing_chars))

    model_card_raw = fetch(MODEL_CARD_URL, 4096)
    need(len(model_card_raw) == recipe.MODEL_CARD["bytes"], "MODEL_CARD_BYTES")
    need(hashlib.md5(model_card_raw).hexdigest() == recipe.MODEL_CARD["md5"], "MODEL_CARD_MD5")
    model_card_text = model_card_raw.decode("utf-8")
    need(recipe.MODEL_CARD["dataset_url"] in model_card_text and
         "License: CC BY 4.0" in model_card_text and
         recipe.MODEL_CARD["training"] in model_card_text, "MODEL_CARD_PROVENANCE")
    need("lessac" not in model_card_text.lower(), "UNDECLARED_BASE_VOICE_LINEAGE")

    setup_raw = fetch(ENGINE_SETUP_URL, 128 * 1024)
    setup_text = setup_raw.decode("utf-8")
    need('version="1.8.0"' in setup_text and
         'license="GPL-3.0-or-later"' in setup_text and
         'license_files=["COPYING"' in setup_text, "PIPER_ENGINE_LICENSE_METADATA")
    copying_raw = fetch(ENGINE_COPYING_URL, 128 * 1024)
    copying_text = copying_raw.decode("utf-8")
    need("GNU GENERAL PUBLIC LICENSE" in copying_text and
         "Version 3, 29 June 2007" in copying_text, "PIPER_ENGINE_COPYING")

    selected_count = len(selected_ids)
    historical_count = len(used_speaker_ids)
    report = {
        "schema": "english-kws-dual-track-v7-source-license-preflight-v1",
        "status": "PASS",
        "run": identity(),
        "source_plan_sha256": source_plan_sha256,
        "source_plan_blob_sha1": recipe.SOURCE_PLAN_BLOB_SHA1,
        "registration_blob_sha1": recipe.REGISTRATION_BLOB_SHA1,
        "source_map": {
            "num_speakers": 904,
            "selected_count": selected_count,
            "selected_speaker_ids": selected_ids,
            "historical_exclusion_count": historical_count,
            "historical_speaker_overlap": False,
            "speaker_ids_unique": True,
        },
        "phoneme_map": {
            "required_characters": required_chars,
            "covered_count": len(required_chars),
            "missing_characters": missing_chars,
            "phoneme_id_map_sha256": sha(json.dumps(phoneme_map, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()),
        },
        "license": {
            "voice_model_card": {
                "repository": recipe.MODEL["repository"],
                "revision": recipe.MODEL["revision"],
                "path": recipe.MODEL_CARD["path"],
                "url": MODEL_CARD_URL,
                "bytes": len(model_card_raw),
                "md5": hashlib.md5(model_card_raw).hexdigest(),
                "sha256": sha(model_card_raw),
                "dataset": recipe.MODEL["dataset"],
                "dataset_url": recipe.MODEL_CARD["dataset_url"],
                "dataset_license": recipe.MODEL_CARD["dataset_license"],
                "training": recipe.MODEL_CARD["training"],
                "lineage": "trained-from-scratch-no-lessac-base-declared",
            },
            "piper_engine": {
                "repository": recipe.ENGINE["repository"],
                "commit": recipe.ENGINE["commit"],
                "version": recipe.ENGINE["version"],
                "license": recipe.ENGINE["license"],
                "setup_url": ENGINE_SETUP_URL,
                "setup_sha256": sha(setup_raw),
                "copying_url": ENGINE_COPYING_URL,
                "copying_sha256": sha(copying_raw),
                "scope": "source-generation-only-not-in-device-runtime",
            },
            "voice_repository_license": "MIT-repository-metadata-only",
            "generated_audio_derivative_clearance": "NOT_ASSERTED_LEGAL_REVIEW_REQUIRED_BEFORE_COMMERCIAL_PRODUCT_CLAIM",
        },
        "config": {
            "url": CONFIG_URL,
            "bytes": len(config_raw),
            "md5": hashlib.md5(config_raw).hexdigest(),
            "sha256": sha(config_raw),
        },
        "no_execution": {
            "model_weights_downloaded": False,
            "audio_downloaded": False,
            "tts_calls": 0,
            "asr_calls": 0,
            "phoneme_observer_calls": 0,
            "kws_forward_calls": 0,
            "optimizer_updates": 0,
            "training_allowed": False,
        },
    }

    write_once(output / "piper-config.json", config_raw)
    write_once(output / "MODEL_CARD", model_card_raw)
    write_once(output / "PIPER_SETUP.py", setup_raw)
    write_once(output / "COPYING", copying_raw)
    write_once(output / "source-license-preflight.json",
        (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"status": report["status"], "source_groups": selected_count,
        "historical_exclusions": historical_count, "phoneme_characters": len(required_chars),
        "license_model_card": report["license"]["voice_model_card"]["sha256"],
        "license_engine_setup": report["license"]["piper_engine"]["sha256"] if "sha256" in report["license"]["piper_engine"] else report["license"]["piper_engine"]["setup_sha256"],
        "no_execution": report["no_execution"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
