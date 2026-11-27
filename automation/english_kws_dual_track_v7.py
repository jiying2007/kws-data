"""Validate the inert V7 dual-track registration and paired source plan."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REGISTRATION = ROOT / "english_kws_dual_track_v7_registration.json"
SOURCE_PLAN = ROOT / "english_kws_dual_track_v7_source_plan.json"

EXPECTED = {
    "VOSEN": ("hey-vosen-v7", "Hey Vosen", "en-US", "/heɪ ˈvoʊ.sən/", "Hey Mero"),
    "MERO": ("hey-mero-v7", "Hey Mero", "en-US", "/heɪ ˈmɛroʊ/", "Hey Vosen"),
}
EXPECTED_CELLS = (
    "vosen-slow", "vosen-normal", "vosen-fast",
    "mero-slow", "mero-normal", "mero-fast",
    "bosen-s", "bosen-z", "memo", "nero", "hello", "morning",
)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def validate_source_plan(plan: dict) -> dict:
    need(type(plan) is dict and plan.get("schema") == "english-kws-dual-track-v7-paired-source-plan-v1",
         "source plan schema")
    need(plan.get("status") == "INERT_AWAITING_PINNED_904_SPEAKER_MAP_AUDIT"
         and plan.get("execution_enabled") is False, "source generation must remain closed")
    model = plan["source_model"]
    need(model["repository"] == "rhasspy/piper-voices"
         and model["revision"] == "de3dcfdf1912bb49726dc4aa11c26017ce2ac62a"
         and model["model_sha256"] == "9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f"
         and model["config_bytes"] == 20163
         and model["config_md5"] == "cb1564bb779f3b1f87e2de778adeb669"
         and model["num_speakers"] == 904, "exact pinned Piper/LibriTTS source model")
    groups = plan["source_selection"]["roles"]
    need([g["speaker_index"] for g in groups] == list(range(496, 508)),
         "exact prospectively selected new source indices")
    from collections import Counter
    need(Counter(g["role"] for g in groups) ==
         {"train": 6, "development": 2, "calibration": 2, "locked-test": 2},
         "exact 6/2/2/2 source-group split")
    need(len({g["source_group_id"] for g in groups}) == 12
         and all(g["source_group_id"] == "piper-libritts-index-" + str(g["speaker_index"]) for g in groups),
         "unique source group identity")
    cells = plan["cells"]
    need(tuple(x["id"] for x in cells) == EXPECTED_CELLS, "exact fixed twelve-cell roster")
    need(all(set(x["class_by_candidate"]) == {"VOSEN", "MERO"} for x in cells),
         "candidate-relative label projection")
    for cell in cells[:3]:
        need(cell["text"] == "Hey Vosen." and cell["raw_ipa"] == "heɪ vˈoʊsən"
             and cell["class_by_candidate"] == {"VOSEN": 1, "MERO": 0},
             "Vosen positive is strictly voiceless /s/ and Mero cross-trigger negative")
    for cell in cells[3:6]:
        need(cell["text"] == "Hey Mero." and cell["raw_ipa"] == "heɪ mˈɛɹoʊ"
             and cell["class_by_candidate"] == {"VOSEN": 0, "MERO": 1},
             "Mero positive and Vosen cross-trigger negative")
    need(all(x["class_by_candidate"] == {"VOSEN": 0, "MERO": 0} for x in cells[6:]),
         "registered contrast and ordinary negatives")
    need(not any(x["id"].startswith("vosen") and "z" in x["raw_ipa"] for x in cells),
         "no voiced-z Vosen positive or variant")
    counts = plan["budget"]
    need(counts == {
        "tts_requests": 144, "qwen_asr_requests": 144, "whisper_requests": 144,
        "parakeet_requests": 144, "phoneme_observer_requests": 144,
        "total_high_level_model_requests": 720, "new_kws_model_forwards": 0,
        "optimizer_updates": 0, "retries": 0, "automatic_expansion": False
    }, "fixed source-stage budget with no retry")
    preflight = plan["source_map_preflight"]
    need(preflight["verify_exact_candidate_indices"] is True
         and preflight["verify_unique_source_speaker_ids"] is True
         and preflight["verify_no_historical_original_speaker_id_collision"] is True
         and preflight["downloads_model_weights"] is False
         and preflight["downloads_or_generates_audio"] is False
         and preflight["observer_calls"] == 0 and preflight["kws_model_calls"] == 0
         and preflight["config_bytes"] == model["config_bytes"]
         and preflight["config_md5"] == model["config_md5"]
         and preflight["expected_num_speakers"] == model["num_speakers"],
         "exact source-map and no-model/no-audio preflight")
    need(plan["source_selection"]["no_fallback_or_replacement"] is True
         and plan["source_selection"]["requires_exact_source_map_preflight"] is True,
         "source map must pass before any TTS; no replacements")
    need(plan["audio_policy"]["tts_only_not_human_recording"] is True
         and plan["audio_policy"]["no_crop_or_trimming"] is True
         and plan["audio_policy"]["no_retry_or_replacement"] is True
         and plan["audio_policy"]["empty_result_quarantines_whole_item"] is True,
         "source integrity and synthetic-vs-human provenance")
    need(plan["status_flags"]["source_map_authenticated"] is False
         and plan["status_flags"]["audio_generated"] is False
         and plan["status_flags"]["observers_completed"] is False
         and plan["status_flags"]["data_admitted"] is False
         and plan["status_flags"]["training_allowed"] is False
         and plan["status_flags"]["promotion_allowed"] is False,
         "all scientific execution and promotion gates remain closed")
    return plan


def load_source_plan() -> dict:
    return json.loads(SOURCE_PLAN.read_bytes(), object_pairs_hook=_unique_pairs)


def validate_source_map_evidence(value: dict, plan: dict) -> dict:
    import hashlib
    evidence = value["source_map_evidence"]
    expected_ids = {
        "496": "p4290", "497": "p2999", "498": "p2427", "499": "p637",
        "500": "p2229", "501": "p1874", "502": "p3446", "503": "p9023",
        "504": "p3114", "505": "p6235", "506": "p4860", "507": "p4519",
    }
    expected_exclusions = list(range(58)) + list(range(400, 496))
    need(evidence["schema"] == "english-kws-dual-track-v7-source-map-evidence-lock-v1"
         and evidence["status"] == "PASS_NEW_SOURCE_IDENTITIES_AUTHENTICATED",
         "source-map evidence schema/status")
    run = evidence["workflow"]
    need(run == {
        "run_id": 45840830422, "job_id": 139983043103,
        "branch": "research/english-kws-dual-track-v7-20261126",
        "head_sha": "ca3dfe8c7fed8803d2650f416d63a6146ca82a9c", "attempt": 1,
    }, "exact original one-shot source-map workflow identity")
    artifact = evidence["artifact"]
    need(artifact == {
        "artifact_id": 15127752986,
        "name": "english-kws-v7-source-map-preflight-45840830422",
        "zip_bytes": 1325,
        "zip_sha256": "ee87db9d7469663d6d1a9820900dcadcc65d1cbee8faa4a704815eaeb625bbf2",
        "member_path": "v7-source-map-preflight.json",
        "member_bytes": 3937,
        "member_sha256": "4e0e4d5cc1ab2873fb9ddcacd0f3b55935a856851588b94a4445abc57ff2f93c",
    }, "exact original preflight artifact identity")
    source_plan_raw = SOURCE_PLAN.read_bytes()
    need(evidence["source_plan_sha256"] == hashlib.sha256(source_plan_raw).hexdigest()
         == "baad365440a789db351aa6277214a631feae0383ef5550664bcddf118fd94c31",
         "same frozen source-plan bytes as authenticated preflight")
    config = evidence["config"]
    need(config == {
        "url": plan["source_map_preflight"]["config_url"],
        "bytes": 20163,
        "md5": "cb1564bb779f3b1f87e2de778adeb669",
        "sha256": "2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d",
        "speaker_count": 904,
    }, "exact immutable 904-speaker config identity")
    need(evidence["selected_speaker_ids"] == expected_ids
         and len(set(expected_ids.values())) == 12
         and [g["speaker_index"] for g in evidence["source_groups"]] == list(range(496, 508))
         and evidence["source_groups"] == plan["source_selection"]["roles"],
         "all 12 selected speaker IDs and frozen source roles")
    need(evidence["historical_index_exclusions"] == expected_exclusions
         and evidence["historical_speaker_id_exclusion_count"] == 194,
         "exact historical index and speaker-ID exclusion audit")
    need(evidence["no_audio_or_model_execution"] == {
        "audio_downloaded": False, "model_weights_downloaded": False, "locked_test_opened": False,
        "new_tts_calls": 0, "new_asr_calls": 0, "new_phone_calls": 0, "new_kws_forwards": 0,
        "training_allowed": False, "retry_count": 0,
    }, "metadata-only preflight; no source or scientific execution")
    return evidence


def validate(value: dict) -> dict:
    need(type(value) is dict and value.get("schema") == "english-kws-dual-track-v7-registration-v1", "schema")
    need(value.get("version") == "V7.0" and value.get("status") == "INERT_DESIGN_ONLY", "version/status")
    need(value.get("execution_enabled") is False, "execution must remain disabled")
    shared = value["shared_runtime"]
    need(shared == {
        "frontend": "PCM16 -> FFT512/Hann400 -> 40 Mel",
        "model_family": "streaming FSMN-CTC",
        "runtime": "shared C11 runtime",
        "model_parameters_shared": False,
        "optimizer_state_shared": False,
        "checkpoints_shared": False,
        "thresholds_shared": False,
        "evaluation_results_shared": False,
    }, "shared runtime contract")
    need(set(value["candidates"]) == set(EXPECTED), "exact two independent candidates")
    for key, (candidate_id, phrase, locale, ipa, cross) in EXPECTED.items():
        item = value["candidates"][key]
        need(item["candidate_id"] == candidate_id and item["phrase"] == phrase
             and item["locale"] == locale and item["ipa"] == ipa, key + " pronunciation identity")
        need(item["role"] == "single-keyword detector" and item["cross_trigger_negative"] == cross,
             key + " target/cross-trigger")
        need(item["partition"] == {"train": 6, "development": 2, "calibration": 2, "locked_test": 2},
             key + " 6/2/2/2")
        need(item["initializer_sha256"] is None and item["dataset_manifest_sha256"] is None
             and item["training_design_sha256"] is None and item["model_sha256"] is None
             and item["primary_report_sha256"] is None and item["replication_report_sha256"] is None
             and item["engineering_report_sha256"] is None and item["promotion_allowed"] is False,
             key + " inert evidence pins")
        need(item["budget"] == {"optimizer_updates": None, "model_forwards": None, "training_runs": 1},
             key + " budget cannot be silently copied or doubled")
    v = value["candidates"]["VOSEN"]["pronunciation_contract"]
    need(v["medial_consonant"] == "voiceless /s/" and v["forbidden_variant"] == "voiced /z/"
         and v["source_ipa"] == "heɪ vˈoʊsən", "Vosen /s/ phoneme lock")
    need(value["shared_negative_pool_policy"]["cross_trigger_phrases_are_mandatory_negatives"] is True
         and value["shared_negative_pool_policy"]["no_positive_relabeling_within_a_candidate"] is True
         and value["shared_negative_pool_policy"]["locked_test_never_used_for_training_or_mining"] is True,
         "cross-trigger and locked-test policy")
    need(value["paired_source_panel"]["source_indices"] == list(range(496, 508))
         and value["paired_source_panel"]["source_map_preflight_required"] is True
         and value["paired_source_panel"]["source_audio_generated"] is False,
         "paired source panel remains inert pending map audit")
    history = value["historical_campaigns"]
    need(history["vosen_schwa16"]["requested_ipa"] == "/heɪ ˈvoʊzən/"
         and history["vosen_schwa16"]["v7_positive_eligible"] is False,
         "exclude voiced-z Vosen historical recipe")
    need(history["mero_meh_44218734553"]["terminal_status"] == "STOP_FRESH_CORPUS_COVERAGE"
         and history["mero_meh_44218734553"]["final_training_rows"] == 0
         and history["mero_meh_44218734553"]["v7_training_eligible"] is False,
         "do not inherit failed historical Mero admission")
    need(value["migration"]["B28_immutable"] is True
         and value["migration"]["B28_one_shot_retry_allowed"] is False
         and value["migration"]["B28_train360_run_id"] == 45804250909,
         "B28 historical one-shot boundary")
    need(value["promotion"]["allowed"] is False
         and value["promotion"]["hard_gates_before_comparison"] is True,
         "promotion remains closed")
    plan = validate_source_plan(load_source_plan())
    validate_source_map_evidence(value, plan)
    return value


def load() -> dict:
    value = json.loads(REGISTRATION.read_bytes(), object_pairs_hook=_unique_pairs)
    return validate(value)


def validate_g2_source_contract():
    sys.path.insert(0, str(ROOT))
    import english_kws_dual_track_v7_once as g2
    plan = json.loads((ROOT / "english_kws_dual_track_v7_execution_plan.json").read_bytes(),
                      object_pairs_hook=_unique_pairs)
    g2.validate_plan(plan, allow_autofill=True)
    return {
        "status": "SOURCE_ONLY_NO_MODEL",
        "worker_sha256": g2.sha(g2.regular(ROOT / "english_kws_dual_track_v7_once.py", 256 * 1024)),
        "recipe_code_sha256": g2.sha(g2.regular(ROOT / "english_kws_dual_track_v7_recipe.py", 128 * 1024)),
        "preflight_script_sha256": g2.sha(g2.regular(ROOT / "english_kws_dual_track_v7_preflight.py", 128 * 1024)),
        "recipe_sha256": g2.recipe.digest(g2.recipe.recipe()),
        "source_plan_sha256": g2.recipe.recipe()["source_plan_sha256"],
        "source_plan_blob_sha1": g2.recipe.SOURCE_PLAN_BLOB_SHA1,
        "registration_blob_sha1": g2.recipe.REGISTRATION_BLOB_SHA1,
        "maximum_tts_calls": 144,
        "maximum_calls_per_observer": 144,
        "maximum_total_model_calls": 720,
        "retry_count": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps({"registration": load(), "g2_source_contract": validate_g2_source_contract()},
                     ensure_ascii=False, sort_keys=True))
