"""Fixed prospective matched controls, diagnostic forever; no runtime imports.

48 independent native parents yield 64 mechanically sliced children.  Preset
names identify exposed synthesis styles, not independently identified people.
No requested pronunciation or intended negative is an observed label.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

SCHEMA = "english-kokoro-context64-recipe-v1"
CAMPAIGN_ID = "kokoro-exact-child-context-controls64-v1"
PARENT_COUNT, CHILD_COUNT = 48, 64
PRESETS = ("af_bella", "af_sarah", "am_michael", "am_puck")
LEGACY_RECIPE_SHA256 = "d010fff7f2144a672265d47367f07ab57732e5549a8c4ca7e9934a084151b369"
LEGACY_GENERATOR_SHA256 = "77def484c22f26fe376fc3b3ae7e93b608cb748c913c46c93beeba46246b5654"
GEOMETRY_HELPER_SHA256 = "0896d375f47318adbc42f63835cb85fc77441ae2472d9df8212dfca4dd7e62c0"
REFERENCE_POLICY_SHA256 = "fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da"
DESIGN_DOCUMENT_SHA256 = "50b7d28ab3cb68a633f5bea60f0506fd112c1f9385409c2991031696e761d29e"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


_old_path = Path(__file__).with_name("english_kokoro48_recipe.py")
need(hashlib.sha256(_old_path.read_bytes()).hexdigest() == LEGACY_RECIPE_SHA256, "IMMUTABLE_RUNTIME_RECIPE")
_spec = importlib.util.spec_from_file_location("context64_immutable_native_recipe", _old_path)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
MODEL = copy.deepcopy(_old.MODEL)
VOICE_ASSETS = copy.deepcopy(_old.VOICE_ASSETS)
PACKAGE = copy.deepcopy(_old.PACKAGE)
PACKAGE_FILES = copy.deepcopy(_old.PACKAGE_FILES)
RUNTIME_VERSIONS = copy.deepcopy(_old.RUNTIME_VERSIONS)
OFFICIAL_EVIDENCE = copy.deepcopy(_old.OFFICIAL_EVIDENCE)

CARRIER = {'boundary_control': {'phonemes': 'kʊd ju ˈOpən ðə wˈɪndO?', 'text': 'Could you open the window?'},
 'boundary_end_char_exclusive': 101,
 'boundary_start_char': 78,
 'phonemes': 'ðə ɹˈum ɪz kwˈIət ænd ðə spˈikəɹ ɪz tˈAkɪŋ tˈIm tə ɹˈid ðɪs sˈɛntəns klˈɪɹli. kʊd ju '
             'ˈOpən ðə wˈɪndO? plˈiz lˈɪsən kˈɛɹfəli. ',
 'text': 'The room is quiet and the speaker is taking time to read this sentence clearly. Could '
         'you open the window? Please listen carefully. '}
CARRIER_TEXT = CARRIER["text"]
CARRIER_PHONEMES = CARRIER["phonemes"]
POSITIVE_CELLS = ({'cell_id': 'mero-direct-eh',
  'pronunciation_hypothesis': 'mero-us-eh-v1',
  'requested_class': 2,
  'requested_text': 'Hey Mero.',
  'target_phonemes': 'hˈA mˈɛɹO.'},
 {'cell_id': 'mero-direct-i',
  'pronunciation_hypothesis': 'mero-us-i-v1',
  'requested_class': 2,
  'requested_text': 'Hey Mero.',
  'target_phonemes': 'hˈA mˈiɹO.'},
 {'cell_id': 'vosen-direct-s',
  'pronunciation_hypothesis': 'vosen-us-s-v1',
  'requested_class': 1,
  'requested_text': 'Hey Vosen.',
  'target_phonemes': 'hˈA vˈOsən.'},
 {'cell_id': 'vosen-direct-z',
  'pronunciation_hypothesis': 'vosen-us-z-v1',
  'requested_class': 1,
  'requested_text': 'Hey Vosen.',
  'target_phonemes': 'hˈA vˈOzən.'})
CONTRAST_CELLS = ({'cell_id': 'contrast-mero-ae',
  'contrast_pattern': 'h?eɪmær(?:oʊ|ou|o)',
  'requested_text': 'Hey Marrow.',
  'target_phonemes': 'hˈA mˈæɹO.'},
 {'cell_id': 'contrast-mero-ih',
  'contrast_pattern': 'h?eɪmɪr(?:oʊ|ou|o)',
  'requested_text': 'Hey Miro.',
  'target_phonemes': 'hˈA mˈɪɹO.'},
 {'cell_id': 'contrast-vosen-b',
  'contrast_pattern': 'h?eɪb(?:oʊ|ou|o)s[əɐʌ]n',
  'requested_text': 'Hey Bosen.',
  'target_phonemes': 'hˈA bˈOsən.'},
 {'cell_id': 'contrast-vosen-ih',
  'contrast_pattern': 'h?eɪv(?:oʊ|ou|o)zɪn',
  'requested_text': 'Hey Vozin.',
  'target_phonemes': 'hˈA vˈOzɪn.'})
ORDINARY_CELLS = ({'cell_id': 'ordinary-time',
  'pronunciation_hypothesis': 'ordinary-negative-v1',
  'requested_class': 0,
  'requested_text': 'What time is it?',
  'target_phonemes': 'wˈʌt tˈIm ɪz ɪt?'},
 {'cell_id': 'ordinary-light',
  'pronunciation_hypothesis': 'ordinary-negative-v1',
  'requested_class': 0,
  'requested_text': 'Please turn on the kitchen light.',
  'target_phonemes': 'plˈiz tˈɜɹn ɑn ðə kˈɪʧən lˈIt.'},
 {'cell_id': 'ordinary-package',
  'pronunciation_hypothesis': 'ordinary-negative-v1',
  'requested_class': 0,
  'requested_text': 'The package is by the front door.',
  'target_phonemes': 'ðə pˈækɪʤ ɪz bˈI ðə fɹˈʌnt dˈɔɹ.'},
 {'cell_id': 'ordinary-window',
  'pronunciation_hypothesis': 'ordinary-negative-v1',
  'requested_class': 0,
  'requested_text': 'Could you open the window?',
  'target_phonemes': 'kʊd ju ˈOpən ðə wˈɪndO?'})
GEOMETRY = {'boundary_child': 'boundary_start_prediction minus3600 to boundary_end_prediction plus3600, '
                   'clamped to native length',
 'child_is_exact_parent_int16_byte_slice': True,
 'duration_boundary_is_acoustic_gold': False,
 'duration_sample_formula': '600*(max(d[0]-3,0)+sum(d[1:a+1]))',
 'full_resample_calls_per_parent': 1,
 'main_child': 'target_start_prediction minus3600 native samples to native EOF',
 'minimum_whole_support_windows_positive': 3,
 'native_boundaries_must_be_divisible_by3': True,
 'parent_frame_limit_samples': 192000,
 'quantize_parent_once': True,
 'resampler': 'scipy.signal.resample_poly(native_float32_full,2,3)',
 'selected_phone_frame_rule': '320*t >= child_start AND320*t+400 <= child_end'}
VALIDATION_GATES = {'all16_contrast_controls_reference_pattern_corroborated': True,
 'all16_ordinary_controls_all3_exact_text_and_no_target_in_fixed_region': True,
 'all16_outside_controls_all3_exact_text_and_no_target_in_fixed_region': True,
 'all192_child_text_captures_complete': True,
 'all48_controls_candidate_positive_count': 0,
 'all48_generation_complete': True,
 'all96_parent_phone_captures_complete': True,
 'candidate_positive_requires_all3_child_text_alias_votes': True,
 'candidate_positive_requires_unique_xlsr_fixed_region_whole_match_and_same_full_terminal_suffix': True,
 'minimum_distinct_preset_components_per_positive_hypothesis': 2,
 'outside_control_parent_target_support_per_hypothesis_minimum': 2,
 'result_is_contract_validation_not_model_quality': True,
 'unknown_control_is_successful_negative': False}

SYNTHESIS = copy.deepcopy(_old.SYNTHESIS) | {
    "one_seed_per_preset_and_cell_shared_between_modes": False,
    "one_seed_per_parent_request": True,
    "resampling": "scipy.signal.resample_poly(float32_native_full,2,3)",
    "children_resampled": False,
    "parent_16k_quantized_once": True,
    "child_derivation": "exact_parent_pcm16_byte_slice",
}
BOUNDARIES = {
    "formula": "600*(max(d[0]-3,0)+sum(d[1:a+1]))",
    "a_definition": "zero-based character boundary; d includes BOS/EOS",
    "native_sample_rate": 24000, "sample_rate": 16000,
    "samples_per_duration_frame": 600, "bos_offset_frames": 3,
    "left_margin_native_samples": 3600, "boundary_right_margin_native_samples": 3600,
    "main_right_endpoint": "native_eof", "outside_right_endpoint": "predicted_boundary_end_plus150ms",
    "boundary_source": "kokoro_predicted_duration", "boundary_is_acoustic_gold": False,
    "preceding_space_duration_counted_in_full": True,
    "not_claimed_equal_to_pipeline_word_half_space_timestamps": True,
    "adaptive_boundary_search": False, "observer_feedback_used": False,
    "recrop_count": 0, "new_silence_samples": 0,
}


def assets():
    return copy.deepcopy(_old.assets())


def seed_for(preset_id, cell_id):
    return int(digest([CAMPAIGN_ID, SYNTHESIS["base_seed"], preset_id, cell_id])[:8], 16)


def recipe():
    need(len(CARRIER_PHONEMES) == 125 and
         CARRIER_PHONEMES[78:101] == CARRIER["boundary_control"]["phonemes"],
         "FIXED_CARRIER_AND_BOUNDARY_CHARACTERS")
    parents, children = [], []
    for preset in PRESETS:
        identity = {"preset_id": preset, "preset_component": "kokoro-v1_0-preset:" + preset,
            "source_group_id": "kokoro-v1_0-preset:" + preset,
            "source_identity_key": "kokoro-v1_0-preset:" + preset,
            "source_namespace": "kokoro-v1_0-preset", "source_identity_kind": "model_voice_preset",
            "human_speaker_identity": None, "human_speaker_identity_verified": False, "role": "train"}
        for kind, cells in (("positive", POSITIVE_CELLS), ("contrast", CONTRAST_CELLS), ("ordinary", ORDINARY_CELLS)):
            for raw_cell in cells:
                cell = copy.deepcopy(raw_cell)
                cell.setdefault("requested_class", 0)
                cell.setdefault("pronunciation_hypothesis", cell["cell_id"] + "-v1")
                raw_phonemes = CARRIER_PHONEMES + cell["target_phonemes"]
                request = {"campaign_id": CAMPAIGN_ID, "preset": identity, "model": MODEL,
                    "voice_asset": VOICE_ASSETS[preset], "kind": kind, "cell": cell,
                    "raw_phonemes": raw_phonemes, "full_requested_text": CARRIER_TEXT + cell["requested_text"],
                    "synthesis": SYNTHESIS, "boundaries": BOUNDARIES,
                    "seed": seed_for(preset, cell["cell_id"])}
                request_sha = digest(request)
                parent = identity | cell | {"id": "enkc64p-" + request_sha[:24],
                    "kind": kind, "mode": "carrier", "hypothesis": cell["pronunciation_hypothesis"],
                    "raw_phonemes": raw_phonemes, "full_requested_text": request["full_requested_text"],
                    "target_start_char": 125, "target_end_char_exclusive": len(raw_phonemes),
                    "seed": request["seed"], "request_sha256": request_sha,
                    "generation_mode": "kokoro-direct-phonemes", "human_gold": False,
                    "diagnostic_only": True, "training_allowed": False,
                    "label_basis": "PREREGISTERED_REQUEST_NOT_OBSERVED_PRONUNCIATION"}
                parents.append(parent)
                for child_kind in (kind, "outside") if kind == "positive" else (kind,):
                    outside = child_kind == "outside"
                    child_spec = {"campaign_id": CAMPAIGN_ID, "parent_id": parent["id"],
                        "parent_request_sha256": request_sha, "kind": child_kind,
                        "character_range": [78, 101] if outside else [125, len(raw_phonemes)],
                        "end_rule": "predicted_end_plus150ms" if outside else "native_eof",
                        "boundaries": BOUNDARIES}
                    child_sha = digest(child_spec)
                    child = identity | {"id": "enkc64c-" + child_sha[:24], "parent_id": parent["id"],
                        "parent_request_sha256": request_sha, "request_sha256": child_sha,
                        "kind": child_kind, "cell_id": ("outside-" if outside else "") + cell["cell_id"],
                        "parent_cell_id": cell["cell_id"], "parent_hypothesis": cell["pronunciation_hypothesis"],
                        "hypothesis": "ordinary-negative-v1" if outside else cell["pronunciation_hypothesis"],
                        "pronunciation_hypothesis": "ordinary-negative-v1" if outside else cell["pronunciation_hypothesis"],
                        "requested_text": CARRIER["boundary_control"]["text"] if outside else cell["requested_text"],
                        "requested_class": 0 if outside else cell["requested_class"],
                        "target_phonemes": CARRIER["boundary_control"]["phonemes"] if outside else cell["target_phonemes"],
                        "child_character_range": child_spec["character_range"],
                        "end_rule": child_spec["end_rule"],
                        "human_gold": False, "diagnostic_only": True, "training_allowed": False,
                        "label_basis": "PREREGISTERED_REQUEST_NOT_OBSERVED_PRONUNCIATION"}
                    if "contrast_pattern" in cell and not outside:
                        child["contrast_pattern"] = cell["contrast_pattern"]
                    children.append(child)
    need(len(parents) == 48 and len(children) == 64 and
         len({r["id"] for r in parents + children}) == 112, "FIXED_PARENT_CHILD_DENOMINATORS")
    return copy.deepcopy({"schema": SCHEMA, "campaign_id": CAMPAIGN_ID,
        "design_document_sha256": DESIGN_DOCUMENT_SHA256,
        "parents": parents, "children": children, "parent_count": 48, "child_count": 64,
        "model": MODEL, "voice_assets": VOICE_ASSETS, "package": PACKAGE,
        "package_files": PACKAGE_FILES, "runtime_versions": RUNTIME_VERSIONS,
        "carrier": CARRIER, "synthesis_config": SYNTHESIS, "boundary_config": BOUNDARIES,
        "geometry": GEOMETRY, "validation_gates": VALIDATION_GATES,
        "official_evidence": OFFICIAL_EVIDENCE, "reference_policy_sha256": REFERENCE_POLICY_SHA256,
        "legacy_recipe_sha256": LEGACY_RECIPE_SHA256, "legacy_generator_sha256": LEGACY_GENERATOR_SHA256,
        "integer_geometry_helper_sha256": GEOMETRY_HELPER_SHA256,
        "new_tts_calls_max": 48, "parent_phone_calls_per_provider": 48,
        "child_text_calls_per_provider": 64, "parent_phone_providers": 2, "child_text_providers": 3,
        "all_model_calls_max": 336, "full_resample_calls_max": 48,
        "retry_count": 0, "kws_forward_calls": 0, "training_calls": 0,
        "all_roles": ["train"], "diagnostic_only": True, "training_allowed": False,
        "permanently_excluded_from_training": True, "automatic_expansion": False,
        "human_source_identity_verified": False, "same_engine_shared_pretraining": True})
