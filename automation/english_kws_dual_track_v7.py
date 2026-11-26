"""Validate the inert V7 dual-track registration; never run models or read audio."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRATION = ROOT / "english_kws_dual_track_v7_registration.json"

EXPECTED = {
    "VOSEN": ("hey-vosen-v7", "Hey Vosen", "en-US", "/heɪ ˈvoʊ.sən/", "Hey Mero"),
    "MERO": ("hey-mero-v7", "Hey Mero", "en-US", "/heɪ ˈmɛroʊ/", "Hey Vosen"),
}


def validate(value: dict) -> dict:
    need = lambda condition, reason: (_ for _ in ()).throw(ValueError(reason)) if not condition else None
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
    need(value["shared_negative_pool_policy"]["cross_trigger_phrases_are_mandatory_negatives"] is True,
         "cross-trigger negative policy")
    need(value["shared_negative_pool_policy"]["no_positive_relabeling"] is True
         and value["shared_negative_pool_policy"]["locked_test_never_used_for_training_or_mining"] is True,
         "positive/locked-test isolation")
    need(value["admission"]["empty_result_quarantines_whole_item"] is True
         and value["admission"]["duplicate_audio_reused_not_regenerated"] is True,
         "empty/duplicate audio policy")
    need(value["migration"]["B28_immutable"] is True
         and value["migration"]["B28_one_shot_retry_allowed"] is False
         and value["migration"]["B28_train360_run_id"] == 45804250909,
         "B28 historical one-shot boundary")
    need(value["promotion"]["allowed"] is False
         and value["promotion"]["hard_gates_before_comparison"] is True,
         "promotion remains closed")
    return value


def load() -> dict:
    raw = REGISTRATION.read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_pairs)
    return validate(value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


if __name__ == "__main__":
    print(json.dumps(load(), ensure_ascii=False, sort_keys=True))
