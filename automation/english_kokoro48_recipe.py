"""Frozen train-only Kokoro feasibility probe; importing this never loads a model.

The 48 clips are diagnostic forever: a successful receipt is not training
admission. Preset packs identify synthesis styles, not independent human people.
All strings are preregistered requests, not observations of the resulting audio.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA = "english-kokoro48-recipe-v1"
CAMPAIGN_ID = "kokoro-v1_0-four-presets-short-carrier48-v1"
SOURCE_COUNT = 48
PRESETS = ("af_bella", "af_sarah", "am_michael", "am_puck")
HF_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CODE_COMMIT = "dfb907a02bba8152ca444717ca5d78747ccb4bec"

MODEL = {
    "repository": "hexgrad/Kokoro-82M", "revision": HF_REVISION,
    "model_path": "kokoro-v1_0.pth", "model_bytes": 327212226,
    "model_sha256": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "config_path": "config.json", "config_bytes": 2351,
    "config_sha256": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "model_card_path": "README.md", "model_card_bytes": 6348,
    "model_card_sha256": "91dcabced89db6f109b8786642f50402d3ee87450e8189589b6f85520e7f4d78",
    "native_sample_rate": 24000, "model_license": "Apache-2.0",
    "code_repository": "hexgrad/kokoro", "code_commit": CODE_COMMIT,
    "code_license": "Apache-2.0", "kokoro_version": "0.9.4",
    "parameter_count_description": "82M (upstream description)",
    "device": "cpu", "dtype": "float32", "quantized": False,
}

VOICE_ASSETS = {
    "af_bella": {"path": "voices/af_bella.pt", "bytes": 523425,
        "sha256": "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6"},
    "af_sarah": {"path": "voices/af_sarah.pt", "bytes": 523425,
        "sha256": "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9"},
    "am_michael": {"path": "voices/am_michael.pt", "bytes": 523435,
        "sha256": "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37"},
    "am_puck": {"path": "voices/am_puck.pt", "bytes": 523420,
        "sha256": "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223"},
}

PACKAGE = {
    "name": "kokoro", "version": "0.9.4", "filename": "kokoro-0.9.4-py3-none-any.whl",
    "bytes": 32592,
    "sha256": "a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd",
    "url": "https://files.pythonhosted.org/packages/ea/cc/75f41633c75224ba820a4533163bc8b070b6bf25416014074c63284c2d4e/kokoro-0.9.4-py3-none-any.whl",
    "metadata_url": "https://pypi.org/pypi/kokoro/0.9.4/json",
    "code_commit_model_blob_sha1": "a7622ba923d0b8b9fc559057cfc5e90b90846231",
    "wheel_and_code_commit_model_identical": True,
    "pipeline_is_pinned_wheel_version": True,
}

# These are hashes of actual bytes in the verified 32592-byte PyPI wheel.
# model.py equals the inspected Git commit. pipeline.py is the RELEASE wheel
# version, whose duration rule was checked separately; it is not claimed to
# equal the later default-branch pipeline.py.
PACKAGE_FILES = {
    "kokoro/__init__.py": [655, "9ef0313aea8ce55a6949240d71c5fd4371ad3245e4df4473fcf61ea24bb10a96"],
    "kokoro/__main__.py": [4089, "cc7ddae46048b6480f5d65cfca58349b7bc7f393dcb727aa71d75e7c524e455f"],
    "kokoro/custom_stft.py": [7746, "94fed3f9ac22bba31135bf8c4ea795762c7ab3ff44f1984d11e7babf27462d69"],
    "kokoro/istftnet.py": [19488, "eb45e18d73d09e5a42801dedf16b6e3fb07d9741a142cd48fb8ebc1f68c11f26"],
    "kokoro/model.py": [6605, "301c55350a068c4ed3b4cbd1209d2afb848bebca72db5657405e47e9dfb604db"],
    "kokoro/modules.py": [7784, "569e3caed16147fa4f66f7fff33ff632954d0a598094c947a5b0fd095c17d5d0"],
    "kokoro/pipeline.py": [17578, "09da32aab781f7a163cbf9f6e379d53db40957cbbab50bc26a21c8440c0eabd6"],
    "kokoro-0.9.4.dist-info/licenses/LICENSE": [11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"],
}

RUNTIME_VERSIONS = {
    "kokoro": "0.9.4", "torch": "2.9.1+cpu", "numpy": "1.26.4",
    "scipy": "1.15.3", "transformers": "4.46.3", "huggingface-hub": "0.36.2",
    "misaki": "0.9.4", "loguru": "0.7.3", "spacy": "3.7.5",
    "spacy-curated-transformers": "0.2.2", "attrs": "25.3.0",
}

SYNTHESIS = {
    "speed": 1.0, "device": "cpu", "dtype": "float32", "torch_threads": 4,
    "base_seed": 20261117, "one_seed_per_preset_and_cell_shared_between_modes": True,
    "deterministic_algorithms": True, "runtime_g2p_calls": 0,
    "voice_selection": "one_original_pack[len(raw_phonemes)-1]",
    "voice_blending": False, "normalize_audio": False,
    "resampling": "scipy.signal.resample_poly(float32_native_crop,2,3)",
    "pcm16_conversion": "rint(clip(float,-1,32767/32768)*32768).astype('<i2')",
    "silence_added": False,
}

CROP = {
    "native_sample_rate": 24000, "samples_per_duration_frame": 600,
    "bos_offset_frames": 3, "left_margin_samples": 3600,
    "left_margin_milliseconds": 150, "right_endpoint": "native_eof",
    "formula": "600*(max(d[0]-3,0)+sum(d[1:a+1]))",
    "a_definition": "zero-based character index of target first phoneme; d includes BOS/EOS",
    "boundary_source": "kokoro_predicted_duration", "boundary_is_acoustic_gold": False,
    "upstream_bos_offset_has_open_todo": True, "adaptive_boundary_search": False,
    "observer_feedback_used": False, "preceding_space_duration_counted_in_full": True,
    "not_claimed_equal_to_pipeline_word_half_space_timestamps": True,
}

# The full prefix is 101 character tokens, then a complete target of 10/11.
# These neutral words and their direct phonemes were frozen before synthesis.
CARRIER_TEXT = ("The room is quiet and the speaker is taking time to read this sentence clearly. "
                "Please listen carefully. ")
CARRIER_PHONEMES = ("ðə ɹˈum ɪz kwˈIət ænd ðə spˈikəɹ ɪz tˈAkɪŋ tˈIm tə ɹˈid ðɪs "
                    "sˈɛntəns klˈɪɹli. plˈiz lˈɪsən kˈɛɹfəli. ")

WAKE_CELLS = (
    {"cell_id": "mero-direct-eh", "requested_text": "Hey Mero.", "requested_class": 2,
     "pronunciation_hypothesis": "mero-us-eh-v1", "target_phonemes": "hˈA mˈɛɹO."},
    {"cell_id": "mero-direct-i", "requested_text": "Hey Mero.", "requested_class": 2,
     "pronunciation_hypothesis": "mero-us-i-v1", "target_phonemes": "hˈA mˈiɹO."},
    {"cell_id": "vosen-direct-s", "requested_text": "Hey Vosen.", "requested_class": 1,
     "pronunciation_hypothesis": "vosen-us-s-v1", "target_phonemes": "hˈA vˈOsən."},
    {"cell_id": "vosen-direct-z", "requested_text": "Hey Vosen.", "requested_class": 1,
     "pronunciation_hypothesis": "vosen-us-z-v1", "target_phonemes": "hˈA vˈOzən."},
)
ORDINARY_CELLS = (
    {"cell_id": "ordinary-time", "requested_text": "What time is it?", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "target_phonemes": "wˈʌt tˈIm ɪz ɪt?"},
    {"cell_id": "ordinary-light", "requested_text": "Please turn on the kitchen light.", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "target_phonemes": "plˈiz tˈɜɹn ɑn ðə kˈɪʧən lˈIt."},
    {"cell_id": "ordinary-package", "requested_text": "The package is by the front door.", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "target_phonemes": "ðə pˈækɪʤ ɪz bˈI ðə fɹˈʌnt dˈɔɹ."},
    {"cell_id": "ordinary-window", "requested_text": "Could you open the window?", "requested_class": 0,
     "pronunciation_hypothesis": "ordinary-negative-v1", "target_phonemes": "kʊd ju ˈOpən ðə wˈɪndO?"},
)

OFFICIAL_EVIDENCE = {
    "phoneme_mapping_url": "https://github.com/hexgrad/misaki/blob/main/misaki/espeak.py",
    "phoneme_mapping_git_blob_sha1": "e5d880c617752cfac03031d20f8f7bd6b3c5a91a",
    "mapping": {"eɪ": "A", "oʊ": "O", "aɪ": "I", "dʒ": "ʤ", "tʃ": "ʧ"},
    "us_length_mark_removed_by_upstream_frontend": True,
    "native_duration_rule_url": "https://github.com/hexgrad/kokoro/blob/" + CODE_COMMIT + "/kokoro/pipeline.py",
    "direct_model_url": "https://github.com/hexgrad/kokoro/blob/" + CODE_COMMIT + "/kokoro/model.py",
    "voice_documentation_url": "https://huggingface.co/hexgrad/Kokoro-82M/blob/" + HF_REVISION + "/VOICES.md",
    "upstream_short_utterance_warning": "less than 10-20 tokens may be weak; best range described as 100-200",
    "model_card_url": "https://huggingface.co/hexgrad/Kokoro-82M/blob/" + HF_REVISION + "/README.md",
    "tts_accuracy_measured_by_this_recipe": False,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seed_for(preset_id, cell_id):
    # The short/carrier pair receives the same seed, not a selected random draw.
    return int(digest([CAMPAIGN_ID, SYNTHESIS["base_seed"], preset_id, cell_id])[:8], 16)


def assets():
    rows = [
        {"path": MODEL["model_path"], "bytes": MODEL["model_bytes"], "sha256": MODEL["model_sha256"]},
        {"path": MODEL["config_path"], "bytes": MODEL["config_bytes"], "sha256": MODEL["config_sha256"]},
        {"path": MODEL["model_card_path"], "bytes": MODEL["model_card_bytes"], "sha256": MODEL["model_card_sha256"]},
        *[dict(VOICE_ASSETS[name]) for name in PRESETS],
    ]
    return [row | {"url": "https://huggingface.co/" + MODEL["repository"] + "/resolve/" +
                         HF_REVISION + "/" + row["path"]} for row in rows]


def recipe():
    rows = []
    for preset in PRESETS:
        group = {"preset_id": preset, "source_group_id": "kokoro-v1_0-preset:" + preset,
                 "source_identity_kind": "model_voice_preset", "human_speaker_identity": None,
                 "voice_family": "kokoro-v1_0-preset:" + preset, "role": "train"}
        cells = [(cell, mode) for cell in WAKE_CELLS for mode in ("short", "carrier")]
        cells += [(cell, "ordinary") for cell in ORDINARY_CELLS]
        for cell, mode in cells:
            raw = (CARRIER_PHONEMES if mode == "carrier" else "") + cell["target_phonemes"]
            request = {"campaign_id": CAMPAIGN_ID, "model": MODEL, "preset": group,
                       "voice_asset": VOICE_ASSETS[preset], "cell": cell, "mode": mode,
                       "raw_phonemes": raw, "synthesis_config": SYNTHESIS, "crop_config": CROP,
                       "seed": seed_for(preset, cell["cell_id"])}
            rows.append(group | dict(cell) | {
                "id": "enk48-" + digest(request)[:24], "mode": mode,
                "request_sha256": digest(request), "raw_phonemes": raw,
                "target_start_char": len(CARRIER_PHONEMES) if mode == "carrier" else 0,
                "full_requested_text": (CARRIER_TEXT if mode == "carrier" else "") + cell["requested_text"],
                "seed": request["seed"], "generation_mode": "kokoro-direct-phonemes",
                "human_gold": False, "diagnostic_only": True, "training_allowed": False,
                "label_basis": "PREREGISTERED_REQUEST_NOT_OBSERVED_PRONUNCIATION",
            })
    return copy.deepcopy({
        "schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "source_count": SOURCE_COUNT,
        "rows": rows, "model": MODEL, "voice_assets": VOICE_ASSETS, "package": PACKAGE,
        "package_files": PACKAGE_FILES, "runtime_versions": RUNTIME_VERSIONS,
        "synthesis_config": SYNTHESIS, "crop_config": CROP,
        "carrier": {"text": CARRIER_TEXT, "phonemes": CARRIER_PHONEMES,
                    "character_tokens": len(CARRIER_PHONEMES)},
        "official_evidence": OFFICIAL_EVIDENCE,
        "new_tts_calls_max": 48, "new_calls_per_observer_max": 48, "observer_count": 5,
        "all_model_calls_max": 288, "retry_count": 0, "kws_forward_calls": 0,
        "training_calls": 0, "all_roles": ["train"], "diagnostic_only": True,
        "training_allowed": False, "auto_admission_allowed": False,
        "existing_admission_rules_modified": False,
        "independent_human_speakers_claimed": False, "same_engine_shared_pretraining": True,
        "base_presets_without_adapter_mixing": True, "human_source_identity_verified": False,
    })
