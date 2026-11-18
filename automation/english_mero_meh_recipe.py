"""Prospective 144-row whole-short Kokoro corpus for the selected MEH-roh target.

Pure standard-library declarations. Importing this file loads no numerical
runtime, model, audio or network resource. Historical probes remain excluded.
Preset identity denotes an upstream synthesis style, never a verified person.
"""
from __future__ import annotations
import copy
import hashlib
import json

SCHEMA = 'english-mero-meh-recipe-v1'

CAMPAIGN_ID = 'english-mero-meh-corpus-v1'

SOURCE_COUNT = 144

HF_REVISION = 'f3ff3571791e39611d31c381e3a41a3af07b4987'

PRESETS = ('af_bella',
 'af_sarah',
 'am_michael',
 'am_puck',
 'af_nicole',
 'am_fenrir',
 'af_aoede',
 'am_echo',
 'af_kore',
 'af_alloy',
 'af_nova',
 'am_onyx')

PRESET_ROLES = {'af_bella': 'train',
 'af_sarah': 'train',
 'am_michael': 'train',
 'am_puck': 'train',
 'af_nicole': 'train',
 'am_fenrir': 'train',
 'af_aoede': 'development',
 'am_echo': 'development',
 'af_kore': 'calibration',
 'af_alloy': 'calibration',
 'af_nova': 'locked-test',
 'am_onyx': 'locked-test'}

VOICE_ASSETS = {'af_bella': {'path': 'voices/af_bella.pt',
              'bytes': 523425,
              'sha256': '8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6'},
 'af_sarah': {'path': 'voices/af_sarah.pt',
              'bytes': 523425,
              'sha256': '49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9'},
 'am_michael': {'path': 'voices/am_michael.pt',
                'bytes': 523435,
                'sha256': '9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37'},
 'am_puck': {'path': 'voices/am_puck.pt',
             'bytes': 523420,
             'sha256': 'dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223'},
 'af_nicole': {'path': 'voices/af_nicole.pt',
               'bytes': 523430,
               'sha256': 'c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da'},
 'am_fenrir': {'path': 'voices/am_fenrir.pt',
               'bytes': 523430,
               'sha256': '98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a'},
 'af_aoede': {'path': 'voices/af_aoede.pt',
              'bytes': 523425,
              'sha256': 'c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4'},
 'am_echo': {'path': 'voices/am_echo.pt',
             'bytes': 523420,
             'sha256': '8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8'},
 'af_kore': {'path': 'voices/af_kore.pt',
             'bytes': 523420,
             'sha256': '8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d'},
 'af_alloy': {'path': 'voices/af_alloy.pt',
              'bytes': 523425,
              'sha256': '6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3'},
 'af_nova': {'path': 'voices/af_nova.pt',
             'bytes': 523420,
             'sha256': 'e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2'},
 'am_onyx': {'path': 'voices/am_onyx.pt',
             'bytes': 523420,
             'sha256': 'e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584'}}

PROFILE = {'path': 'automation/english_mero_meh_profile_v1.json',
 'profile_id': 'english_mero_meh_v1',
 'sha256': '1ef68a0aac3883304719023efee2a33d4510efaf2a7364328f98470920cc6ca6'}

MODEL = {'repository': 'hexgrad/Kokoro-82M',
 'revision': 'f3ff3571791e39611d31c381e3a41a3af07b4987',
 'model_path': 'kokoro-v1_0.pth',
 'model_bytes': 327212226,
 'model_sha256': '496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4',
 'config_path': 'config.json',
 'config_bytes': 2351,
 'config_sha256': '5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f',
 'model_card_path': 'README.md',
 'model_card_bytes': 6348,
 'model_card_sha256': '91dcabced89db6f109b8786642f50402d3ee87450e8189589b6f85520e7f4d78',
 'native_sample_rate': 24000,
 'model_license': 'Apache-2.0',
 'code_repository': 'hexgrad/kokoro',
 'code_commit': 'dfb907a02bba8152ca444717ca5d78747ccb4bec',
 'code_license': 'Apache-2.0',
 'kokoro_version': '0.9.4',
 'parameter_count_description': '82M (upstream description)',
 'device': 'cpu',
 'dtype': 'float32',
 'quantized': False}

PACKAGE = {'name': 'kokoro',
 'version': '0.9.4',
 'filename': 'kokoro-0.9.4-py3-none-any.whl',
 'bytes': 32592,
 'sha256': 'a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd',
 'url': 'https://files.pythonhosted.org/packages/ea/cc/75f41633c75224ba820a4533163bc8b070b6bf25416014074c63284c2d4e/kokoro-0.9.4-py3-none-any.whl',
 'metadata_url': 'https://pypi.org/pypi/kokoro/0.9.4/json',
 'code_commit_model_blob_sha1': 'a7622ba923d0b8b9fc559057cfc5e90b90846231',
 'wheel_and_code_commit_model_identical': True,
 'pipeline_is_pinned_wheel_version': True}

PACKAGE_FILES = {'kokoro/__init__.py': [655, '9ef0313aea8ce55a6949240d71c5fd4371ad3245e4df4473fcf61ea24bb10a96'],
 'kokoro/__main__.py': [4089, 'cc7ddae46048b6480f5d65cfca58349b7bc7f393dcb727aa71d75e7c524e455f'],
 'kokoro/custom_stft.py': [7746, '94fed3f9ac22bba31135bf8c4ea795762c7ab3ff44f1984d11e7babf27462d69'],
 'kokoro/istftnet.py': [19488, 'eb45e18d73d09e5a42801dedf16b6e3fb07d9741a142cd48fb8ebc1f68c11f26'],
 'kokoro/model.py': [6605, '301c55350a068c4ed3b4cbd1209d2afb848bebca72db5657405e47e9dfb604db'],
 'kokoro/modules.py': [7784, '569e3caed16147fa4f66f7fff33ff632954d0a598094c947a5b0fd095c17d5d0'],
 'kokoro/pipeline.py': [17578, '09da32aab781f7a163cbf9f6e379d53db40957cbbab50bc26a21c8440c0eabd6'],
 'kokoro-0.9.4.dist-info/licenses/LICENSE': [11357,
                                             'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4']}

RUNTIME_VERSIONS = {'kokoro': '0.9.4',
 'torch': '2.9.1+cpu',
 'numpy': '1.26.4',
 'scipy': '1.15.3',
 'transformers': '4.46.3',
 'huggingface-hub': '0.36.2',
 'misaki': '0.9.4',
 'loguru': '0.7.3',
 'spacy': '3.7.5',
 'spacy-curated-transformers': '0.2.2',
 'attrs': '25.3.0',
 'click': '8.3.1',
 'typer': '0.29.1',
 'typer-slim': '0.24.0'}

SYNTHESIS = {'base_seed': 2026111801,
 'device': 'cpu',
 'dtype': 'float32',
 'torch_threads': 4,
 'deterministic_algorithms': True,
 'speeds': [0.9, 1.0, 1.1],
 'voice_selection': 'one_original_pack[len(raw_phonemes)-1]',
 'voice_blending': False,
 'runtime_g2p_calls': 0,
 'normalize_audio': False,
 'silence_added': False,
 'crop_calls': 0,
 'carrier_calls': 0,
 'native_sample_rate': 24000,
 'sample_rate': 16000,
 'resampling': 'scipy.signal.resample_poly(float32_native_full,2,3)',
 'pcm16_conversion': "rint(clip(float,-1,32767/32768)*32768).astype('<i2')",
 'retry_count': 0}

VOICE_METADATA_EVIDENCE = {'tree_url': 'https://huggingface.co/api/models/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987/voices?recursive=false&expand=false',
 'tree_sha256': '661ac8dfdcfaac54aeccedd34308db33c19e85febdf3a58f086dd1ae5e70255e',
 'voices_url': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/f3ff3571791e39611d31c381e3a41a3af07b4987/VOICES.md',
 'voices_sha256': 'ec7e4941ad7e194af61e3455928528a9ff5360c7c505e412efab27d6a69ea106',
 'voice_asset_hash_basis': 'Official fixed-revision LFS oid and size; producer verifies downloaded bytes '
                           'before model construction.',
 'source_identity_scope': 'Upstream US preset assets; no runtime blends; no independent-human or '
                          'complete-human-ancestry claim.',
 'known_unresolved_mixture_excluded': 'af_heart',
 'roster_choice_basis': 'Before synthesis, upstream listed US presets with stronger published grades where '
                        'available; no observer feedback or replacement.'}

MINIMUM_BALANCED_GROUPS = {'train': 6, 'development': 2, 'calibration': 2, 'locked-test': 2}

BUDGET = {'maximum_tts_calls': 144,
 'maximum_calls_per_observer': 144,
 'maximum_total_model_calls': 720,
 'maximum_raw_rows': 144,
 'maximum_derived_bags': 432,
 'retry_count': 0,
 'automatic_expansion': False}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity_contract():
    return {"schema": "english-source-identity-contract-v1",
            "purpose": "synthetic_training_source_identity", "diagnostic_only": False,
            "training_allowed": True, "human_speaker_identity_claimed": False,
            "source_namespaces": ["kokoro-v1_0-preset"], "known_lineage_edges": [],
            "train_side_exposed": ["kokoro-v1_0-preset:" + x for x in
                ("af_bella", "af_sarah", "am_michael", "am_puck")]}


def cells():
    """Requested labels are intent. Admission separately classifies observations."""
    result = []
    for key, text, phones, label, hypothesis in (
        ("mero", "Hey Mero.", "hˈA mˈɛɹO.", 2, "mero-meh-us-v1"),
        ("vosen", "Hey Vosen.", "hˈA vˈOsən.", 1, "vosen-us-s-v1"),
    ):
        for tag, speed in (("slow", 0.9), ("fast", 1.1)):
            result.append({"cell_id": key + "-" + tag, "requested_text": text,
                "raw_phonemes": phones, "requested_class": label, "kind": "positive",
                "pronunciation_hypothesis": hypothesis, "speed": speed})
    for key, text, phones, hypothesis in (
        ("nero", "Hey Nero.", "hˈA nˈɛɹO.", "nero-eh-us-v1"),
        ("memo", "Hey Memo.", "hˈA mˈɛmO.", "memo-us-v1"),
        ("bosen-s", "Hey Bosen.", "hˈA bˈOsən.", "bosen-us-s-v1"),
        ("bosen-z", "Hey Bosen.", "hˈA bˈOzən.", "bosen-us-z-v1"),
    ):
        result.append({"cell_id": key, "requested_text": text, "raw_phonemes": phones,
            "requested_class": 0, "kind": "contrast", "pronunciation_hypothesis": hypothesis, "speed": 1.0})
    for key, text, phones in (
        ("hello", "Hello there.", "həlˈO ðˈɛɹ."),
        ("morning", "Good morning.", "ɡˈʊd mˈɔɹnɪŋ."),
        ("turn-off", "Turn it off.", "tˈɜɹn ɪt ˈɔf."),
        ("open-door", "Open the door.", "ˈOpən ðə dˈɔɹ."),
    ):
        result.append({"cell_id": key, "requested_text": text, "raw_phonemes": phones,
            "requested_class": 0, "kind": "ordinary", "pronunciation_hypothesis": "ordinary-negative-v1",
            "speed": 1.0})
    need(len(result) == 12 and len({x["cell_id"] for x in result}) == 12, "TWELVE_FIXED_CELLS")
    return result


def rows():
    result = []
    for preset in PRESETS:
        role = PRESET_ROLES[preset]
        key = "kokoro-v1_0-preset:" + preset
        identity = {"source_namespace": "kokoro-v1_0-preset", "source_id": preset,
            "source_identity_key": key, "source_group_id": key, "preset_id": preset,
            "source_speaker_id": None, "human_speaker_identity": None,
            "human_speaker_identity_verified": False, "human_source_identity_verified": False,
            "voice_family": key, "role": role,
            "known_lineage": {"component_keys": [key], "declared_relations": [], "human_ancestry_complete": False},
            "evaluation_scope": "SYNTHETIC_TRAIN_PRESET" if role == "train" else
                "HELDOUT_SYNTHETIC_PRESET_WITH_KNOWN_LINEAGE_EXCLUSIONS"}
        for cell in cells():
            seed = int(digest([CAMPAIGN_ID, SYNTHESIS["base_seed"], preset, cell["cell_id"]])[:8], 16)
            request = {"campaign_id": CAMPAIGN_ID, "identity": identity, "cell": cell,
                "seed": seed, "synthesis": SYNTHESIS, "model_sha256": MODEL["model_sha256"],
                "voice_asset": VOICE_ASSETS[preset], "profile": PROFILE}
            request_sha = digest(request)
            result.append({**identity, **cell, "id": "enmeh-" + request_sha[:24],
                "campaign_id": CAMPAIGN_ID, "source_campaign_id": CAMPAIGN_ID,
                "seed": seed, "request_sha256": request_sha, "profile_id": PROFILE["profile_id"],
                "generation_mode": "kokoro-direct-phonemes", "mode": "whole-short",
                "target_phonemes": cell["raw_phonemes"], "target_start_char": 0,
                "diagnostic_only": False, "auto_admission_allowed": True,
                "training_allowed": False, "human_gold": False,
                "permanently_excluded_from_training": False,
                "label_basis": "PREREGISTERED_REQUEST_NOT_OBSERVED_PRONUNCIATION"})
    need(len(result) == SOURCE_COUNT and len({x["id"] for x in result}) == SOURCE_COUNT,
         "EXACT_144_UNIQUE_REQUESTS")
    return copy.deepcopy(result)


def assets():
    result = [
        {"path": MODEL["model_path"], "bytes": MODEL["model_bytes"], "sha256": MODEL["model_sha256"]},
        {"path": MODEL["config_path"], "bytes": MODEL["config_bytes"], "sha256": MODEL["config_sha256"]},
        {"path": MODEL["model_card_path"], "bytes": MODEL["model_card_bytes"], "sha256": MODEL["model_card_sha256"]},
        *[copy.deepcopy(VOICE_ASSETS[name]) for name in PRESETS],
    ]
    return [row | {"url": "https://huggingface.co/" + MODEL["repository"] + "/resolve/" +
                         HF_REVISION + "/" + row["path"]} for row in result]


def recipe():
    return {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "source_design_id": CAMPAIGN_ID,
        "rows": rows(), "row_count": SOURCE_COUNT, "source_count": len(PRESETS),
        "preset_roles": copy.deepcopy(PRESET_ROLES), "cells": cells(), "model": copy.deepcopy(MODEL),
        "voice_assets": copy.deepcopy(VOICE_ASSETS), "package": copy.deepcopy(PACKAGE),
        "package_files": copy.deepcopy(PACKAGE_FILES), "runtime_versions": copy.deepcopy(RUNTIME_VERSIONS),
        "synthesis": copy.deepcopy(SYNTHESIS), "profile": copy.deepcopy(PROFILE),
        "voice_metadata_evidence": copy.deepcopy(VOICE_METADATA_EVIDENCE), "budget": copy.deepcopy(BUDGET),
        "source_identity_contract_sha256": digest(identity_contract()),
        "minimum_balanced_groups": copy.deepcopy(MINIMUM_BALANCED_GROUPS),
        "diagnostic_only": False, "training_allowed": True, "auto_admission_allowed": True,
        "permanently_excluded_from_training": False, "human_gold": False,
        "parent_context_weak_policy": False, "automatic_expansion": False, "retry_count": 0}
