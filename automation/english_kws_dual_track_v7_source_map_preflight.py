"""Metadata-only audit of 12 new Piper/LibriTTS speaker identities; no audio or model weights."""
from __future__ import annotations
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = HERE / "english_kws_dual_track_v7_source_plan.json"
HISTORY = HERE / "english_synthetic192_recipe.py"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_history():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_v7_historical_speaker_ids", HISTORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit():
    plan = json.loads(PLAN.read_bytes())
    model = plan["source_model"]
    indices = plan["source_selection"]["indices"]
    need(indices == list(range(496, 508)), "EXACT_FROZEN_V7_SOURCE_INDICES")
    url = plan["source_map_preflight"]["config_url"]
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=45) as response:
        need(response.status == 200, "PINNED_PUBLIC_CONFIG_HTTP_200")
        raw = response.read(model["config_bytes"] + 1)
    need(len(raw) == model["config_bytes"]
         and hashlib.md5(raw, usedforsecurity=False).hexdigest() == model["config_md5"],
         "EXACT_PINNED_904_SPEAKER_CONFIG_BYTES_AND_MD5")
    config = json.loads(raw)
    mapping = config.get("speaker_id_map", {})
    need(config.get("num_speakers") == 904 and type(mapping) is dict
         and len(mapping) == 904 and set(mapping.values()) == set(range(904)),
         "EXACT_904_SPEAKER_MAP")
    inverse = {index: label for label, index in mapping.items()}
    history = load_history()
    used_indices = set(range(58)) | set(range(400, 496)) | {
        index for index, _speaker in history.NEW_VOICES
    }
    used_speaker_ids = set(history.OLD_LIBRI_IDS) | set(history.OLD_PIPER_IDS) | {
        speaker for _index, speaker in history.NEW_VOICES
    }
    used_speaker_ids |= {
        int(inverse[index][1:]) for index in used_indices
        if index in inverse and re.fullmatch(r"p[0-9]+", inverse[index])
    }
    need(not set(indices).intersection(used_indices), "NO_PRIOR_PIPER_INDEX_REUSE")
    selected = {str(index): inverse[index] for index in indices}
    need(len(set(selected.values())) == len(indices)
         and all(re.fullmatch(r"p[0-9]+", speaker) for speaker in selected.values())
         and all(int(speaker[1:]) not in used_speaker_ids for speaker in selected.values()),
         "NO_PRIOR_LIBRITTS_OR_PIPER_SPEAKER_ID_REUSE")
    return {
        "schema": "english-kws-dual-track-v7-source-map-preflight-v1",
        "status": "PASS_NEW_SOURCE_IDENTITIES_AUTHENTICATED",
        "source_plan_sha256": sha(PLAN.read_bytes()),
        "config_url": url,
        "config_bytes": len(raw),
        "config_md5": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
        "config_sha256": sha(raw),
        "speaker_count": 904,
        "selected_speaker_ids": selected,
        "source_groups": plan["source_selection"]["roles"],
        "historical_index_exclusions": sorted(used_indices),
        "historical_speaker_id_exclusion_count": len(used_speaker_ids),
        "new_tts_calls": 0,
        "new_asr_calls": 0,
        "new_phone_calls": 0,
        "new_kws_forwards": 0,
        "audio_downloaded": False,
        "model_weights_downloaded": False,
        "training_allowed": False,
        "locked_test_opened": False,
        "retry_count": 0,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: english_kws_dual_track_v7_source_map_preflight.py OUTPUT.json")
    output = Path(sys.argv[1])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "FRESH_SOURCE_MAP_PREFLIGHT_OUTPUT")
    value = audit()
    output.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
