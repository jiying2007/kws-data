"""Fixed diagnostic parents and exact PCM children; no local numerical entry point.

The public worker owns the artifact and run. This module owns source/, one
initially empty asset cache, and at most 48 native calls and 48 full resamples.
A child never calls a model or resampler. Requested classes are experiment
intent, not observed labels. Every row is permanently excluded from training.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("context64_fixed_recipe", HERE / "english_kokoro_context64_recipe.py")
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)
recipe.need(hashlib.sha256((HERE / "english_kokoro48_generate.py").read_bytes()).hexdigest() ==
            recipe.LEGACY_GENERATOR_SHA256, "IMMUTABLE_NATIVE_GENERATOR")
_spec = importlib.util.spec_from_file_location("context64_immutable_native_components", HERE / "english_kokoro48_generate.py")
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)

SCHEMA = "english-kokoro-context64-generation-v1"
CAMPAIGN_ID = recipe.CAMPAIGN_ID
MAIN_BEFORE = "2ca611210126e2f78939e6e50f7cd0157c76d5dd"
WORKFLOW = "jiying2007/kws-data/.github/workflows/english-kokoro-context64-once.yml@refs/heads/main"
EXPOSURE_SHA256 = "f9e0644e7db12b411c131890df90eebad7037a5cb285735079ff0c5e1cb443e0"
need, sha, within, regular, put, once = native.need, native.sha, native.within, native.regular, native.put, native.once
phoneme_ids, materialize_native_result = native.phoneme_ids, native.materialize_native_result


class GenerationInterrupted(BaseException):
    """A controlled external stop retains consumed calls and every denominator."""


class SourceIntegrityError(ValueError):
    pass


def source_preflight(plan=None):
    for name, pin in (("english_kokoro48_generate.py", recipe.LEGACY_GENERATOR_SHA256),
                      ("english_kokoro48_recipe.py", recipe.LEGACY_RECIPE_SHA256),
                      ("english_synthetic192_training_core.py", recipe.GEOMETRY_HELPER_SHA256)):
        need(sha(regular(HERE / name, 1024 * 1024)) == pin, "IMMUTABLE_COMPONENT_" + name)
        if plan is not None:
            need(plan.get("helper_sha256", {}).get("automation/" + name) == pin, "PLAN_COMPONENT_" + name)
    need(recipe.MODEL == native.recipe.MODEL and recipe.VOICE_ASSETS == native.recipe.VOICE_ASSETS and
         recipe.PACKAGE == native.recipe.PACKAGE and recipe.PACKAGE_FILES == native.recipe.PACKAGE_FILES and
         recipe.RUNTIME_VERSIONS == native.recipe.RUNTIME_VERSIONS and recipe.assets() == native.recipe.assets(),
         "UNCHANGED_NATIVE_RUNTIME_ASSETS")
    if plan is not None:
        need(plan.get("generator_code_sha256") == sha(regular(Path(__file__), 1024 * 1024)) and
             plan.get("recipe_code_sha256") == sha(regular(HERE / "english_kokoro_context64_recipe.py", 1024 * 1024)),
             "NEW_SOURCE_CODE_PINS")
    return {"schema": SCHEMA, "source_check": "PASS", "model_calls": 0,
            "processor_calls": 0, "resample_calls": 0, "network_calls": 0}


def exclusion_sets(plan):
    value = plan.get("source_exposure_exclusion")
    need(type(value) is dict and recipe.digest(value) == EXPOSURE_SHA256 and
         plan.get("source_exposure_exclusion_sha256") == EXPOSURE_SHA256,
         "EXACT_RECORDED_OLD_EXPOSURE")
    result = {}
    for key, count in (("old_pcm_sha256", 64), ("old_native_full_f32_sha256", 48),
                       ("old_request_ids", 48), ("old_request_sha256", 48)):
        items = value.get(key)
        need(type(items) is list and len(items) == count and len(set(items)) == count and
             all(type(x) is str for x in items), "EXPOSURE_DENOMINATOR_" + key)
        if key != "old_request_ids":
            need(all(re.fullmatch(r"[0-9a-f]{64}", x) for x in items), "EXPOSURE_SHA256_" + key)
        result[key] = set(items)
    return result


def validate_plan(plan, *, executing=False):
    need(type(plan) is dict, "PLAN_OBJECT")
    frozen = recipe.recipe()
    exact = {"campaign_id": CAMPAIGN_ID, "recipe_sha256": recipe.digest(frozen),
        "maximum_tts_calls": 48, "maximum_parent_count": 48, "maximum_child_count": 64,
        "maximum_resample_calls": 48, "generation_deadline_seconds": 1800,
        "retry_count": 0, "diagnostic_only": True, "training_allowed": False,
        "automatic_expansion": False, "total_high_level_model_calls_max": 336}
    for key, expected in exact.items():
        need(type(plan.get(key)) is type(expected) and plan[key] == expected, "PLAN_" + key.upper())
    need(type(plan.get("enabled")) is bool and (not executing or plan["enabled"]), "PLAN_ENABLED")
    old = exclusion_sets(plan)
    need(len(frozen["parents"]) == 48 and len(frozen["children"]) == 64 and
         all(row["role"] == "train" and row["diagnostic_only"] is True and
             row["training_allowed"] is False for row in frozen["parents"] + frozen["children"]),
         "FIXED_DIAGNOSTIC_DENOMINATORS")
    need(not ({r["id"] for r in frozen["parents"] + frozen["children"]} & old["old_request_ids"]) and
         not ({r["request_sha256"] for r in frozen["parents"] + frozen["children"]} & old["old_request_sha256"]),
         "NO_PREVIOUS_REQUEST_IDENTITY")
    return frozen


def public_guard(plan, run, env=None):
    env = os.environ if env is None else env
    source_preflight(plan)
    need(plan.get("enabled") is True and plan.get("main_before") == MAIN_BEFORE, "EXACT_ENABLED_NEW_SUCCESSOR")
    actual = {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME")}
    need(type(run) is dict and actual == run and env.get("GITHUB_ACTIONS") == "true" and
         env.get("GITHUB_WORKFLOW_REF") == WORKFLOW and env.get("GITHUB_REF") == "refs/heads/main" and
         actual["repository"] == "jiying2007/kws-data" and actual["run_attempt"] == "1" and
         actual["event"] == "push" and type(actual["commit"]) is str and
         re.fullmatch(r"[0-9a-f]{40}", actual["commit"]) and type(actual["run_id"]) is str and
         re.fullmatch(r"[1-9][0-9]*", actual["run_id"]), "FIRST_PUBLIC_MAIN_WORKFLOW_ONLY")
    event = json.loads(regular(Path(env["GITHUB_EVENT_PATH"]), 16 * 1024 * 1024))
    need(event.get("before") == MAIN_BEFORE and event.get("after") == actual["commit"] and
         event["before"] != event["after"] and event.get("repository", {}).get("full_name") == actual["repository"] and
         event["repository"].get("private") is False, "EXACT_PUBLIC_MAIN_EVENT")
    return actual


def duration_geometry(parent, duration, native_samples):
    """Integer model-duration coordinates, never acoustic word-end annotations."""
    ps = parent["raw_phonemes"]
    need(ps == recipe.CARRIER_PHONEMES + parent["target_phonemes"] and
         parent["target_start_char"] == 125 and parent["target_end_char_exclusive"] == len(ps),
         "FIXED_PARENT_PHONEME_GEOMETRY")
    need(type(duration) is list and len(duration) == len(ps) + 2 and
         all(type(x) is int and 1 <= x <= 50 for x in duration), "NATIVE_DURATION_BOS_EOS")
    need(type(native_samples) is int and native_samples == 600 * sum(duration), "NATIVE_DURATION_TOTAL_SAMPLES")
    def boundary(a):
        need(type(a) is int and 0 <= a <= len(ps), "DURATION_CHARACTER_BOUNDARY")
        return 600 * (max(duration[0] - 3, 0) + sum(duration[1:a + 1]))
    target, outside_start, outside_end = boundary(125), boundary(78), boundary(101)
    need(0 <= outside_start < outside_end < target < native_samples and
         all(x % 3 == 0 for x in (target, outside_start, outside_end, native_samples)),
         "ORDERED_INTEGRAL_DURATION_COORDINATES")
    return {"native_samples": native_samples, "target_start_prediction_native": target,
        "target_start_parent16k": 2 * target // 3,
        "boundary_start_prediction_native": outside_start, "boundary_end_prediction_native": outside_end,
        "boundary_start_parent16k": 2 * outside_start // 3,
        "boundary_end_parent16k": 2 * outside_end // 3,
        "duration_boundary_is_acoustic_gold": False, "timestamp_kind": "model_prediction_only"}


def child_bounds(parent, child, duration, native_samples):
    need(child["parent_id"] == parent["id"] and child["parent_request_sha256"] == parent["request_sha256"],
         "EXACT_CHILD_PARENT_IDENTITY")
    geo = duration_geometry(parent, duration, native_samples)
    outside = child["kind"] == "outside"
    need(child["child_character_range"] == ([78, 101] if outside else [125, len(parent["raw_phonemes"])]) and
         child["end_rule"] == ("predicted_end_plus150ms" if outside else "native_eof") and
         (not outside or parent["kind"] == "positive"), "FIXED_CHILD_CHARACTER_RULE")
    start = max((geo["boundary_start_prediction_native"] if outside else geo["target_start_prediction_native"]) - 3600, 0)
    end = min(geo["boundary_end_prediction_native"] + 3600, native_samples) if outside else native_samples
    need(0 <= start < end <= native_samples and start % 3 == 0 and end % 3 == 0,
         "EXACT_CHILD_NATIVE_BOUNDARIES")
    start16, end16 = 2 * start // 3, 2 * end // 3
    need(not outside or end16 <= geo["target_start_parent16k"], "OUTSIDE_END_BEFORE_PARENT_TARGET")
    return {"child_start_native_sample": start, "child_end_native_sample": end,
        "child_start_parent16k": start16, "child_end_parent16k": end16,
        "target_start_prediction_native": geo["target_start_prediction_native"],
        "target_start_parent16k": geo["target_start_parent16k"],
        "left_margin_native_samples": 3600, "right_margin_native_samples": 3600 if outside else 0,
        "right_endpoint": "predicted_boundary_end_plus150ms" if outside else "native_eof",
        "exact_parent_pcm_slice": True, "boundaries_model_predicted": True,
        "boundary_is_acoustic_gold": False, "parent_resampled_once": True,
        "child_resample_calls": 0, "recrop_count": 0, "added_silence_samples": 0}


def whole_support_geometry(samples):
    need(type(samples) is int and 0 < samples <= 192000, "CHILD_PCM_GEOMETRY_LIMIT")
    start, observed = 38400, 38400 + samples + 16000
    ends = [end for end in range(16000, observed + 1, 1280)
            if end >= 35200 and end - 31840 <= start and end - 128 >= start + samples]
    return {"whole_support_windows": len(ends), "geometry_usable_positive": len(ends) >= 3,
        "whole_support_embedding_end_samples": ends, "geometry_helper_sha256": recipe.GEOMETRY_HELPER_SHA256,
        "geometry_prefix_samples": 38400, "geometry_suffix_samples": 16000,
        "geometry_is_integer_check_only": True, "geometry_audio_generated": False,
        "human_word_end_annotation": False}


def _load_native(output, asset_root):
    """Same verified native graph and proof, with the actual new campaign origin."""
    versions, package_files = native._installed_runtime(output)
    receipts = [native._download(a, within(asset_root, a["path"])) for a in recipe.assets()]
    wheel = {"path": recipe.PACKAGE["filename"], **{k: recipe.PACKAGE[k] for k in ("url", "bytes", "sha256")}}
    wheel_receipt = native._download(wheel, asset_root / wheel["path"])
    put(output / "source/runtime" / wheel["path"], regular(asset_root / wheel["path"], wheel["bytes"]))
    for name in ("config.json", "README.md", *[recipe.VOICE_ASSETS[p]["path"] for p in recipe.PRESETS]):
        put(within(output, "source/runtime/" + name), regular(within(asset_root, name), 1024 * 1024))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    import torch
    from kokoro import KModel
    from kokoro.istftnet import AdaIN1d
    torch.set_num_threads(recipe.SYNTHESIS["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(recipe.SYNTHESIS["base_seed"])
    origin = {"campaign_id": CAMPAIGN_ID, "legacy_helper_source_sha": recipe.LEGACY_GENERATOR_SHA256,
              "native_graph_or_weights_modified": False}
    once(output / "source/MODEL_LOAD_STARTED.json", {"schema": SCHEMA, **origin,
         "assets_verified": receipts, "forward_attempts": 0, "local_files_only": True})
    model = KModel(repo_id=recipe.MODEL["repository"], config=str(asset_root / "config.json"),
                   model=str(asset_root / recipe.MODEL["model_path"]), disable_complex=False).to("cpu").float().eval()
    once(output / "source/MODEL_CONSTRUCTED.json", {"schema": SCHEMA + "/constructed", **origin,
        "constructor_and_runtime_transforms_returned": True, "model_instances": 1,
        "model_forward_calls": 0, "checkpoint_proof_pending": True})
    checkpoint = torch.load(asset_root / recipe.MODEL["model_path"], map_location="cpu", weights_only=True)
    component_proof = native._strict_checkpoint_readback(model, checkpoint, torch, output=output,
        native_adain_type=AdaIN1d)
    del checkpoint
    config = json.loads(regular(asset_root / "config.json", recipe.MODEL["config_bytes"]))
    need(model.vocab == config["vocab"] and model.context_length == 512, "EXACT_NATIVE_VOCAB_CONTEXT")
    packs = {}
    for preset in recipe.PRESETS:
        pack = torch.load(within(asset_root, recipe.VOICE_ASSETS[preset]["path"]), map_location="cpu", weights_only=True)
        need(isinstance(pack, torch.Tensor) and pack.dtype == torch.float32 and pack.device.type == "cpu" and
             pack.ndim == 3 and tuple(pack.shape[1:]) == (1, 256) and torch.isfinite(pack).all().item(), "NATIVE_VOICE_PACK")
        packs[preset] = pack
    for row in recipe.recipe()["parents"]:
        phoneme_ids(row["raw_phonemes"], model.vocab, model.context_length)
        need(0 <= len(row["raw_phonemes"]) - 1 < len(packs[row["preset_id"]]), "NATIVE_VOICE_ROW_INDEX")
    lock = {"schema": SCHEMA + "/model-lock", **origin, **dict(recipe.MODEL), "assets": receipts,
        "package": recipe.PACKAGE, "package_download": wheel_receipt,
        "installed_package_files": package_files, "runtime_versions": versions,
        "weight_loading": component_proof, "all_component_weights_verified": True,
        "weight_contract": native.native_affine_contract()["schema"],
        "native_affine_contract": native.native_affine_contract(),
        "native_affine_contract_sha256": native.native_affine_contract_sha256(),
        "weight_diagnostics": "source/weight-diagnostics/SUMMARY.json",
        "weight_diagnostics_sha256": sha(regular(output / "source/weight-diagnostics/SUMMARY.json", 1024 * 1024)),
        "weight_diagnostics_schema_origin": "immutable_english_kokoro48_native_component_proof",
        "voice_pack_shapes": {k: list(v.shape) for k, v in packs.items()},
        "preset_mixing": False, "runtime_g2p_calls": 0, "device": "cpu", "dtype": "float32",
        "human_speaker_identity_claimed": False, "forward_attempts_at_lock": 0,
        "diagnostic_only": True, "training_allowed": False}
    once(output / "source/MODEL_LOCK.json", lock)

    def forward(row, ids):
        need(ids == phoneme_ids(row["raw_phonemes"], model.vocab, model.context_length), "NATIVE_FORWARD_TOKEN_IDENTITY")
        torch.manual_seed(row["seed"])
        with torch.inference_mode():
            result = model(row["raw_phonemes"], packs[row["preset_id"]][len(row["raw_phonemes"]) - 1],
                           speed=1.0, return_output=True)
        return result
    return forward, config["vocab"]


def _identity_marker(row, entity):
    return {"schema": SCHEMA + "/" + entity, "campaign_id": CAMPAIGN_ID,
        "id": row["id"], "request_sha256": row["request_sha256"],
        "preset_id": row["preset_id"], "kind": row["kind"], "role": "train",
        "diagnostic_only": True, "training_allowed": False, "retry_count": 0}


def _check_new_hash(digest, old, seen, code):
    if digest in old or digest in seen:
        raise SourceIntegrityError(code)
    seen.add(digest)


def write_parent_audio(output, row, audio, duration, ids, counters, exposure, seen):
    """Production only: one full float resample and one parent16k quantization."""
    import numpy as np
    from scipy.signal import resample_poly
    need(isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1 and
         0 < len(audio) <= 600 * 50 * (len(row["raw_phonemes"]) + 2) and
         bool(np.isfinite(audio).all()), "BOUNDED_FINITE_NATIVE_FLOAT32")
    folder = "source/parents/" + row["id"] + "/"
    raw = audio.astype("<f4", copy=False).tobytes()
    row.update(native_samples=len(audio), native_full_f32_path=folder + "native-full.f32",
               native_full_wav_path=folder + "native-full.wav")
    row["native_full_f32_sha256"] = put(within(output, row["native_full_f32_path"]), raw)
    row["native_full_wav_sha256"] = put(within(output, row["native_full_wav_path"]), native._wav(native._pcm16(audio, np), 24000))
    doc = {"schema": SCHEMA + "/native-duration", "campaign_id": CAMPAIGN_ID, "id": row["id"],
        "phonemes": row["raw_phonemes"], "input_ids": ids, "pred_dur": duration,
        "includes_bos_eos": True, "samples_per_frame": 600, "sample_rate": 24000,
        "native_samples": len(audio), "boundary_is_acoustic_gold": False,
        "target_start_char": row["target_start_char"], "target_end_char_exclusive": row["target_end_char_exclusive"],
        "duration_source": "native_KModel_output", "boundary_config": recipe.BOUNDARIES}
    row["duration_path"] = folder + "pred_dur.json"
    row["duration_sha256"] = once(within(output, row["duration_path"]), doc)
    _check_new_hash(row["native_full_f32_sha256"], exposure["old_native_full_f32_sha256"], seen["native"],
                    "SOURCE_INTEGRITY_NATIVE_REUSE_OR_DUPLICATE")
    row.update(duration_geometry(row, duration, len(audio)))
    expected_samples = (len(audio) * 2 + 2) // 3
    need(400 <= expected_samples <= 192000, "FULL_PARENT_16K_SAMPLE_LIMIT")
    need(counters["actual_resample_attempts"] < 48, "FULL_RESAMPLE_CEILING")
    marker = _identity_marker(row, "resample") | {"resampler": recipe.GEOMETRY["resampler"],
        "native_full_f32_sha256": row["native_full_f32_sha256"], "full_parent_input": True,
        "child_resample_calls": 0, "resample_attempts": 1, "resample_returned": False}
    once(within(output, folder + "RESAMPLE_STARTED.json"), marker)
    row["resample_attempts"] = 1
    counters["actual_resample_attempts"] += 1
    derived = resample_poly(audio, 2, 3)
    counters["actual_resample_returns"] += 1
    row["resample_returned"] = True
    once(within(output, folder + "RESAMPLE_RETURNED.json"), marker | {"resample_returned": True})
    need(isinstance(derived, np.ndarray) and derived.dtype == np.float32 and derived.ndim == 1 and
         len(derived) == expected_samples and bool(np.isfinite(derived).all()), "DERIVED_FULL_PARENT_FLOAT32_GEOMETRY")
    row["resampled_f32_path"] = folder + "resampled-16k.f32"
    row["resampled_f32_sha256"] = put(within(output, row["resampled_f32_path"]), derived.astype("<f4", copy=False).tobytes())
    pcm = native._pcm16(derived, np)
    row.update(pcm_path=folder + "parent-16k.pcm", wav_path=folder + "parent-16k.wav",
               pcm_samples=len(pcm) // 2, sample_rate=16000, channels=1, sample_width=2)
    row["pcm_sha256"] = put(within(output, row["pcm_path"]), pcm)
    row["wav_sha256"] = put(within(output, row["wav_path"]), native._wav(pcm, 16000))
    _check_new_hash(row["pcm_sha256"], exposure["old_pcm_sha256"], seen["pcm"],
                    "SOURCE_INTEGRITY_PARENT_PCM_REUSE_OR_DUPLICATE")
    row.update(native_sample_rate=24000, native_wav_sample_width=2,
        native_float_format="float32-little-endian", resampled_float_format="float32-little-endian",
        native_peak=float(np.max(np.abs(audio))), native_rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        native_pcm16_clipped_samples=int(np.count_nonzero((audio < -1.0) | (audio > 32767 / 32768))),
        parent_pcm16_clipped_samples=int(np.count_nonzero((derived < -1.0) | (derived > 32767 / 32768))),
        resampler=recipe.GEOMETRY["resampler"], normalization_applied=False,
        parent_16k_quantized_once=True, synthetic_silence_added=False,
        observed_target_pronunciation=False, human_gold=False, diagnostic_only=True, training_allowed=False)
    return pcm


def write_child_audio(output, parent, child, duration, parent_pcm, exposure, seen):
    """Only standard-library byte slicing. No float operation or model here."""
    need(type(parent_pcm) is bytes and len(parent_pcm) == 2 * parent["pcm_samples"] and
         sha(parent_pcm) == parent["pcm_sha256"], "EXACT_PARENT_PCM_BEFORE_SLICE")
    bounds = child_bounds(parent, child, duration, parent["native_samples"])
    a, b = bounds["child_start_parent16k"], bounds["child_end_parent16k"]
    need(0 <= a < b <= parent["pcm_samples"], "CHILD_WITHIN_SAVED_PARENT")
    pcm = parent_pcm[a * 2:b * 2]
    folder = "source/children/" + child["id"] + "/"
    child.update(bounds, parent_pcm_sha256=parent["pcm_sha256"],
        pcm_path=folder + "child-16k.pcm", wav_path=folder + "child-16k.wav",
        pcm_samples=b - a, sample_rate=16000, channels=1, sample_width=2)
    child["pcm_sha256"] = put(within(output, child["pcm_path"]), pcm)
    child["wav_sha256"] = put(within(output, child["wav_path"]), native._wav(pcm, 16000))
    child.update(whole_support_geometry(len(pcm) // 2))
    derivation = _identity_marker(child, "child-derivation") | {"parent_id": parent["id"],
        "parent_request_sha256": parent["request_sha256"], "parent_pcm_path": parent["pcm_path"],
        "parent_pcm_sha256": parent["pcm_sha256"], "parent_pcm_samples": parent["pcm_samples"],
        "duration_path": parent["duration_path"], "duration_sha256": parent["duration_sha256"],
        "child_character_range": child["child_character_range"], "boundary_config": recipe.BOUNDARIES,
        **bounds, "pcm_path": child["pcm_path"], "wav_path": child["wav_path"],
        "pcm_sha256": child["pcm_sha256"], "wav_sha256": child["wav_sha256"], "pcm_samples": child["pcm_samples"],
        "sample_rate": 16000, "sample_width": 2, "channels": 1,
        "whole_support_windows": child["whole_support_windows"],
        "geometry_usable_positive": child["geometry_usable_positive"],
        "child_model_calls": 0, "child_resample_calls": 0, "normalization_applied": False}
    child["derivation_path"] = folder + "DERIVATION.json"
    child["derivation_sha256"] = once(within(output, child["derivation_path"]), derivation)
    _check_new_hash(child["pcm_sha256"], exposure["old_pcm_sha256"], seen["pcm"],
                    "SOURCE_INTEGRITY_CHILD_PCM_REUSE_OR_DUPLICATE")
    return bounds


def _unmeasured(row, entity):
    value = copy.deepcopy(row) | {"generation_status": "not_started", "pcm_path": None,
        "wav_path": None, "pcm_sha256": None, "wav_sha256": None, "pcm_samples": None,
        "sample_rate": 16000, "channels": 1, "sample_width": 2}
    if entity == "parent":
        value.update(forward_attempts=0, forward_returned=False, resample_attempts=0, resample_returned=False)
    else:
        value.update(derivation_attempts=0, derivation_returned=False, child_model_calls=0, child_resample_calls=0)
    return value


def _failure(output, row, entity, exc, counters):
    row["error_type"], row["error_code"] = type(exc).__name__, str(exc)[:200]
    if isinstance(exc, SourceIntegrityError):
        counters["source_integrity_failures"].append({"id": row["id"], "code": str(exc)})
    attempted = row["forward_attempts"] if entity == "parent" else row["derivation_attempts"]
    # A begin/validation failure is explicit but is not counted as a model call.
    status = "FAILED_NO_RETRY" if attempted else "NOT_STARTED_NO_RETRY"
    receipt = _identity_marker(row, entity) | {"status": status, "row": copy.deepcopy(row),
        "error_type": type(exc).__name__, "error_code": str(exc)[:200]}
    for key in ("forward_attempts", "forward_returned", "resample_attempts", "resample_returned",
                "derivation_attempts", "derivation_returned"):
        if key in row:
            receipt[key] = row[key]
    path = within(output, "source/" + ("parents" if entity == "parent" else "children") + "/" + row["id"] + "/FAILED.json")
    if not path.exists():
        try:
            once(path, receipt)
        except BaseException as write_exc:
            counters["receipt_write_failures"].append({"id": row["id"], "type": type(write_exc).__name__,
                "code": str(write_exc)[:160]})
            if not isinstance(write_exc, Exception):
                raise


def _run_parents(output, parents, children, forward, vocab, deadline, counters, exposure):
    seen = {"native": set(), "pcm": set()}
    child_map = {row["id"]: [] for row in parents}
    for child in children:
        need(child["parent_id"] in child_map, "CHILD_PARENT_IN_FIXED_ROSTER")
        child_map[child["parent_id"]].append(child)
    for parent in parents:
        folder = within(output, "source/parents/" + parent["id"])
        try:
            need(time.monotonic() < deadline, "GENERATION_DEADLINE")
            ids = phoneme_ids(parent["raw_phonemes"], vocab)
            need(counters["actual_forward_attempts"] < 48, "TTS_FORWARD_CEILING")
            started = _identity_marker(parent, "parent") | {"forward_attempts": 0, "forward_returned": False,
                "phonemes": parent["raw_phonemes"], "phoneme_ids": ids, "target_start_char": 125,
                "seed": parent["seed"], "voice_pack_index": len(parent["raw_phonemes"]) - 1,
                "synthesis_config": recipe.SYNTHESIS, "direct_api": "KModel.forward", "runtime_g2p_calls": 0}
            once(folder / "STARTED.json", started)
            parent["generation_status"] = "failed"
            once(folder / "FORWARD_STARTED.json", started | {"forward_attempts": 1})
            parent["forward_attempts"] = 1
            counters["actual_forward_attempts"] += 1
            native_result = forward(parent, ids)
            counters["actual_forward_returns"] += 1
            parent["forward_returned"] = True
            once(folder / "FORWARD_RETURNED.json", started | {"forward_attempts": 1, "forward_returned": True})
            audio, duration = materialize_native_result(native_result)
            parent_pcm = write_parent_audio(output, parent, audio, duration, ids, counters, exposure, seen)
            completed = _identity_marker(parent, "parent") | {"status": "COMPLETE",
                "generation_status": "success", "forward_attempts": 1, "forward_returned": True,
                "resample_attempts": 1, "resample_returned": True, "phoneme_ids": ids,
                "generation_model_sha256": recipe.MODEL["model_sha256"],
                "voice_pack_sha256": recipe.VOICE_ASSETS[parent["preset_id"]]["sha256"],
                "row": copy.deepcopy(parent) | {"generation_status": "success"}}
            once(folder / "COMPLETED.json", completed)
            parent["generation_status"] = "success"
            counters["completed_parents"] += 1
        except BaseException as exc:
            _failure(output, parent, "parent", exc, counters)
            if not isinstance(exc, Exception) or str(exc) == "GENERATION_DEADLINE":
                raise
            continue
        for child in child_map[parent["id"]]:
            child_folder = within(output, "source/children/" + child["id"])
            try:
                need(time.monotonic() < deadline, "GENERATION_DEADLINE")
                marker = _identity_marker(child, "child") | {"parent_id": parent["id"],
                    "parent_pcm_sha256": parent["pcm_sha256"], "derivation_attempts": 0,
                    "derivation_returned": False, "child_model_calls": 0, "child_resample_calls": 0}
                once(child_folder / "STARTED.json", marker)
                child["generation_status"] = "failed"
                once(child_folder / "DERIVATION_STARTED.json", marker | {"derivation_attempts": 1})
                child["derivation_attempts"] = 1
                write_child_audio(output, parent, child, duration, parent_pcm, exposure, seen)
                child["derivation_returned"] = True
                once(child_folder / "DERIVATION_RETURNED.json", marker | {"derivation_attempts": 1,
                    "derivation_returned": True})
                receipt = _identity_marker(child, "child") | {"status": "COMPLETE",
                    "generation_status": "success", "derivation_attempts": 1, "derivation_returned": True,
                    "child_model_calls": 0, "child_resample_calls": 0,
                    "row": copy.deepcopy(child) | {"generation_status": "success"}}
                once(child_folder / "COMPLETED.json", receipt)
                child["generation_status"] = "success"
                counters["completed_children"] += 1
            except BaseException as exc:
                _failure(output, child, "child", exc, counters)
                if not isinstance(exc, Exception) or str(exc) == "GENERATION_DEADLINE":
                    raise


def _counters():
    return {"actual_forward_attempts": 0, "actual_forward_returns": 0, "completed_parents": 0,
        "actual_resample_attempts": 0, "actual_resample_returns": 0, "completed_children": 0,
        "source_integrity_failures": [], "receipt_write_failures": []}


def generate(plan, output_root, *, run, asset_root=None):
    """Only enabled first public main push; no backend injection or numeric CLI."""
    frozen = validate_plan(plan, executing=True)
    public_guard(plan, run)
    output = Path(output_root).absolute()
    need(output.is_dir() and not any(p.is_symlink() for p in (output, *output.parents)), "ARTIFACT_ROOT")
    source = native.empty_directory(output / "source")
    parents = [_unmeasured(r, "parent") for r in frozen["parents"]]
    children = [_unmeasured(r, "child") for r in frozen["children"]]
    once(source / "RECIPE.json", frozen)
    once(source / "EXECUTION_PLAN.json", plan)
    once(source / "REQUESTS.json", {"schema": SCHEMA + "/requests", "campaign_id": CAMPAIGN_ID,
        "run": run, "parents": frozen["parents"], "children": frozen["children"],
        "declared_parent_count": 48, "declared_child_count": 64,
        "forward_attempts_at_write": 0, "resample_attempts_at_write": 0,
        "diagnostic_only": True, "training_allowed": False, "permanently_excluded_from_training": True})
    once(source / "STARTED.json", {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run,
        "declared_parent_count": 48, "declared_child_count": 64, "maximum_tts_calls": 48,
        "maximum_resample_calls": 48, "retry_count": 0, "forward_attempts_at_start": 0,
        "diagnostic_only": True, "training_allowed": False})
    counters, start, error, private = _counters(), time.monotonic(), None, None
    old_handlers = {}
    def interrupt(signum, frame):
        raise GenerationInterrupted("GENERATION_DEADLINE" if signum == signal.SIGALRM else "GENERATION_TERMINATED")
    try:
        for sig in (signal.SIGALRM, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        signal.alarm(plan["generation_deadline_seconds"])
        candidate = Path(asset_root).absolute() if asset_root is not None else source / "_private_model"
        need(candidate not in (output, source) and candidate not in output.parents,
             "DEDICATED_PRIVATE_MODEL_DIRECTORY")
        private = native.empty_directory(candidate)
        forward, vocab = _load_native(output, private)
        _run_parents(output, parents, children, forward, vocab,
            start + plan["generation_deadline_seconds"], counters, exclusion_sets(plan))
    except BaseException as exc:
        error = {"type": type(exc).__name__, "code": str(exc)[:200]}
    finally:
        signal.alarm(0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for entity, rows in (("parent", parents), ("child", children)):
            for row in rows:
                folder = within(output, "source/" + ("parents" if entity == "parent" else "children") + "/" + row["id"])
                if not (folder / "COMPLETED.json").exists() and not (folder / "FAILED.json").exists():
                    _failure(output, row, entity, ValueError("NOT_COMPLETED_NO_RETRY"), counters)
        if private is not None:
            try:
                shutil.rmtree(private)
            except Exception as exc:
                error = error or {"type": type(exc).__name__, "code": "PRIVATE_ASSET_CLEANUP_FAILED"}
        complete = (counters["completed_parents"] == 48 and counters["completed_children"] == 64 and
            counters["actual_forward_attempts"] == counters["actual_forward_returns"] == 48 and
            counters["actual_resample_attempts"] == counters["actual_resample_returns"] == 48 and
            not counters["source_integrity_failures"] and not counters["receipt_write_failures"] and error is None)
        result = {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "stage": "generate", "run": run,
            "status": "COMPLETE" if complete else "INCOMPLETE_NO_RETRY", "complete": complete,
            "declared_parent_count": 48, "declared_child_count": 64, **counters,
            "parents": parents, "children": children, "retry_count": 0,
            "elapsed_seconds": time.monotonic() - start, "error": error,
            "source_integrity_passed": complete and not counters["source_integrity_failures"],
            "source_exposure_exclusion_sha256": EXPOSURE_SHA256,
            "old_audio_model_calls": 0, "child_tts_calls": 0, "child_resample_calls": 0,
            "parent_phone_calls": 0, "child_text_calls": 0, "kws_forward_calls": 0, "training_calls": 0,
            "diagnostic_only": True, "training_allowed": False, "auto_admission_allowed": False,
            "permanently_excluded_from_training": True, "automatic_expansion": False,
            "human_gold": False, "reference_phone_scope": "parent_full_pcm_not_independent_child_phone"}
        once(source / "PARENTS.json", {"schema": SCHEMA + "/parents", "run": run, "rows": parents})
        once(source / "CHILDREN.json", {"schema": SCHEMA + "/children", "run": run, "rows": children})
        once(source / "RESULT.json", result)
    return result
