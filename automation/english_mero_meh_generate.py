"""One native synthesis per preregistered whole-short request, with byte readback.

Only generate() with the default backend loads a numerical runtime. Historical
native component proofs are reused verbatim; no old campaign is restarted.
"""
from __future__ import annotations

from array import array
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import sys
import time
from types import SimpleNamespace
import wave

import english_kokoro48_generate as native
import english_mero_meh_recipe as recipe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "english-mero-meh-generation-v1"
FILES_SCHEMA = "english-mero-meh-corpus-files-v1"
need, sha, within, regular, put, once = native.need, native.sha, native.within, native.regular, native.put, native.once
LEGACY_PINS = {
    "automation/english_kokoro48_generate.py": "77def484c22f26fe376fc3b3ae7e93b608cb748c913c46c93beeba46246b5654",
    "automation/english_kokoro48_recipe.py": "d010fff7f2144a672265d47367f07ab57732e5549a8c4ca7e9934a084151b369",
    "automation/english_kokoro48_once.py": "8edd94fb7142fd0fe16ee9473f110b2fa23b9f973c74c5484eb6a6ff68fa2f29",
}


class GenerationInterrupted(BaseException):
    pass


def read(path):
    return json.loads(regular(path, 64 * 1024**2))


def file_records(root):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "NO_SYMLINK_IN_PACKET")
        if not path.is_file() or path == root / "FILES.json":
            continue
        raw = regular(path, 256 * 1024**2)
        records.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    return records


def exclusion_sets():
    doc = read(ROOT / "EXCLUSION_EVIDENCE.json")
    need(doc.get("schema") == "english-mero-meh-exclusion-evidence-v1", "EXCLUSION_SCHEMA")
    result = {}
    for key in ("row_ids", "request_sha256", "pcm_sha256", "parent_pcm_sha256"):
        values = doc.get("excluded_identities", {}).get(key)
        need(type(values) is list and values and values == sorted(set(values)), "EXCLUSION_LIST_" + key)
        if key != "row_ids":
            need(all(type(x) is str and re.fullmatch(r"[0-9a-f]{64}", x) for x in values), "EXCLUSION_HASHES")
        result[key] = set(values)
    return result


def source_preflight(plan=None):
    for name, digest in LEGACY_PINS.items():
        need(sha(regular(ROOT / name, 1024**2)) == digest, "IMMUTABLE_NATIVE_COMPONENT")
    need(recipe.MODEL == native.recipe.MODEL and recipe.PACKAGE == native.recipe.PACKAGE
         and recipe.PACKAGE_FILES == native.recipe.PACKAGE_FILES
         and recipe.RUNTIME_VERSIONS == native.recipe.RUNTIME_VERSIONS, "UNCHANGED_NATIVE_RUNTIME")
    need(sha(regular(ROOT / recipe.PROFILE["path"], 1024**2)) == recipe.PROFILE["sha256"], "FIXED_PROFILE")
    rows = recipe.rows()
    old = exclusion_sets()
    need(len(rows) == 144 and not ({r["id"] for r in rows} & old["row_ids"])
         and not ({r["request_sha256"] for r in rows} & old["request_sha256"]), "FRESH_PLANNED_IDENTITIES")
    need(set(recipe.PRESETS) == set(recipe.VOICE_ASSETS) and len(recipe.PRESETS) == 12
         and "af_heart" not in recipe.PRESETS, "FIXED_NONBLENDED_PRESETS")
    if plan is not None:
        need(plan.get("campaign_id") == recipe.CAMPAIGN_ID
             and plan.get("native_tts_request_budget") == 144
             and plan.get("maximum_total_model_calls") == 720
             and plan.get("retry_count") == 0 and plan.get("warmup_count") == 0,
             "FIXED_SOURCE_BUDGET")
        pins = plan.get("source_files_sha256", {})
        for name in (*LEGACY_PINS, "automation/english_mero_meh_generate.py",
                     "automation/english_mero_meh_recipe.py", recipe.PROFILE["path"], "EXCLUSION_EVIDENCE.json"):
            need(pins.get(name) == sha(regular(ROOT / name, 64 * 1024**2)), "SOURCE_PLAN_PIN_" + name)
        if "recipe_sha256" in plan:
            need(plan["recipe_sha256"] == recipe.digest(recipe.recipe()), "RECIPE_CONTENT_PIN")
    return {"source_check": "PASS", "planned_rows": 144, "model_calls": 0,
            "resample_calls": 0, "network_calls": 0}


def whole_support_geometry(samples):
    need(type(samples) is int and 0 < samples <= 192000, "BOUNDED_WHOLE_PCM")
    ends = [end for end in range(16000, 38400 + samples + 16000 + 1, 1280)
            if end >= 35200 and end - 31840 <= 38400 and end - 128 >= 38400 + samples]
    return {"whole_support_windows": len(ends), "whole_support_embedding_end_samples": ends,
            "geometry_usable_positive": 6400 <= samples <= 28672 and len(ends) >= 3,
            "geometry_audio_generated": False, "human_word_end_annotation": False}


def _float_values(raw):
    need(type(raw) is bytes and 0 < len(raw) <= 4 * 288000 and len(raw) % 4 == 0, "BOUNDED_FLOAT32_BYTES")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    need(all(math.isfinite(x) for x in values), "FINITE_FLOAT32")
    return values


def quantize(raw):
    values = _float_values(raw)
    pcm = array("h", (round(max(-1.0, min(32767 / 32768, value)) * 32768) for value in values))
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


def pcm_statistics(pcm):
    need(type(pcm) is bytes and len(pcm) > 0 and len(pcm) % 2 == 0, "PCM_BYTES")
    values = array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    rms = math.sqrt(sum(x * x for x in values) / len(values)) / 32768
    rail = sum(x in (-32768, 32767) for x in values) / len(values)
    return {"pcm_rms": rms, "pcm_rail_fraction": rail,
            "pcm_quality_passed": 6400 <= len(values) <= 160000 and rms >= 1e-4 and rail <= 0.01}


def _load_native(output, assets):
    versions, package_files = native._installed_runtime(output)
    receipts = [native._download(pin, within(assets, pin["path"])) for pin in recipe.assets()]
    wheel = {"path": recipe.PACKAGE["filename"], **{k: recipe.PACKAGE[k] for k in ("url", "bytes", "sha256")}}
    wheel_receipt = native._download(wheel, within(assets, wheel["path"]))
    for name in (wheel["path"], "config.json", "README.md", *[recipe.VOICE_ASSETS[p]["path"] for p in recipe.PRESETS]):
        put(within(output, "source/runtime/" + name), regular(within(assets, name), 1024**2))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    import numpy as np
    from scipy.signal import resample_poly
    import torch
    from kokoro import KModel
    from kokoro.istftnet import AdaIN1d
    torch.set_num_threads(recipe.SYNTHESIS["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(recipe.SYNTHESIS["base_seed"])
    once(output / "source/MODEL_LOAD_STARTED.json", {"schema": SCHEMA, "assets_verified": receipts,
         "campaign_id": recipe.CAMPAIGN_ID, "forward_attempts": 0, "local_files_only": True})
    model = KModel(repo_id=recipe.MODEL["repository"], config=str(assets / "config.json"),
                   model=str(assets / recipe.MODEL["model_path"]), disable_complex=False).to("cpu").float().eval()
    once(output / "source/MODEL_CONSTRUCTED.json", {"schema": SCHEMA,
         "constructor_and_runtime_transforms_returned": True, "model_forward_calls": 0})
    checkpoint = torch.load(assets / recipe.MODEL["model_path"], map_location="cpu", weights_only=True)
    proof = native._strict_checkpoint_readback(model, checkpoint, torch, output=output, native_adain_type=AdaIN1d)
    del checkpoint
    config = read(assets / "config.json")
    need(model.vocab == config["vocab"] and model.context_length == 512, "EXACT_NATIVE_VOCAB_CONTEXT")
    packs = {}
    for preset in recipe.PRESETS:
        pack = torch.load(within(assets, recipe.VOICE_ASSETS[preset]["path"]), map_location="cpu", weights_only=True)
        need(isinstance(pack, torch.Tensor) and pack.dtype == torch.float32 and pack.device.type == "cpu"
             and pack.ndim == 3 and tuple(pack.shape[1:]) == (1, 256)
             and torch.isfinite(pack).all().item(), "NATIVE_VOICE_PACK")
        packs[preset] = pack
    for row in recipe.rows():
        native.phoneme_ids(row["raw_phonemes"], model.vocab, 512)
        need(0 <= len(row["raw_phonemes"]) - 1 < len(packs[row["preset_id"]]), "NATIVE_VOICE_INDEX")
    once(output / "source/MODEL_LOCK.json", {"schema": SCHEMA + "/native-lock", "campaign_id": recipe.CAMPAIGN_ID,
         "model": recipe.MODEL, "assets": receipts, "runtime_versions": versions,
         "installed_package_files": package_files, "package_download": wheel_receipt,
         "weight_loading": proof, "all_component_weights_verified": True,
         "weight_contract": native.native_affine_contract()["schema"],
         "native_affine_contract": native.native_affine_contract(),
         "native_affine_contract_sha256": native.native_affine_contract_sha256(),
         "weight_diagnostics": "source/weight-diagnostics/SUMMARY.json",
         "weight_diagnostics_sha256": sha(regular(output / "source/weight-diagnostics/SUMMARY.json", 1024**2)),
         "native_graph_or_weights_modified": False, "preset_mixing": False,
         "voice_pack_shapes": {k: list(v.shape) for k, v in packs.items()},
         "runtime_g2p_calls": 0, "warmup_calls": 0, "forward_attempts_at_lock": 0})

    def forward(row, ids):
        need(ids == native.phoneme_ids(row["raw_phonemes"], model.vocab, 512), "NATIVE_TOKEN_IDENTITY")
        torch.manual_seed(row["seed"])
        with torch.inference_mode():
            return model(row["raw_phonemes"], packs[row["preset_id"]][len(row["raw_phonemes"]) - 1],
                         speed=row["speed"], return_output=True)

    def materialize(result):
        audio, duration = native.materialize_native_result(result)
        need(isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1,
             "NATIVE_FLOAT32_MONO")
        return audio.astype("<f4", copy=False).tobytes(), duration

    def resample(raw):
        output_values = resample_poly(np.frombuffer(raw, dtype="<f4"), 2, 3)
        need(output_values.dtype == np.float32 and output_values.ndim == 1, "RESAMPLED_FLOAT32_MONO")
        return output_values.astype("<f4", copy=False).tobytes()

    return SimpleNamespace(vocab=config["vocab"], forward=forward, materialize=materialize, resample=resample)


def _write_audio(output, row, raw, duration, ids, backend, old, seen):
    values = _float_values(raw)
    need(type(duration) is list and len(duration) == len(row["raw_phonemes"]) + 2
         and all(type(x) is int and 1 <= x <= 200 for x in duration)
         and len(values) == 600 * sum(duration), "NATIVE_DURATION_GEOMETRY")
    row.update(native_samples=len(values), native_f32_path="native/" + row["id"] + ".f32",
               native_sample_rate=24000, parent_sample_rate=24000,
               parent_pcm_format="float32-little-endian", parent_scope="ENTIRE_NATIVE_OUTPUT")
    row["parent_pcm_sha256"] = put(within(output, row["native_f32_path"]), raw)
    need(row["parent_pcm_sha256"] not in old["parent_pcm_sha256"] | old["pcm_sha256"]
         and row["parent_pcm_sha256"] not in seen["parent_pcm_sha256"] | seen["pcm_sha256"], "NO_REUSED_NATIVE_AUDIO")
    seen["parent_pcm_sha256"].add(row["parent_pcm_sha256"])
    row["duration_path"] = "rows/" + row["id"] + "/DURATION.json"
    row["duration_sha256"] = once(within(output, row["duration_path"]), {
        "id": row["id"], "input_ids": ids, "raw_phonemes": row["raw_phonemes"], "pred_dur": duration,
        "native_samples": len(values), "sample_rate": 24000, "samples_per_frame": 600,
        "includes_bos_eos": True, "timestamp_kind": "model_prediction_only", "acoustic_gold": False})
    row["resample_calls"] = 1
    derived = backend.resample(raw)
    derived_values = _float_values(derived)
    need(len(derived_values) == (len(values) * 2 + 2) // 3, "ONE_WHOLE_RESAMPLE_GEOMETRY")
    row["resample_returned"] = True
    row["resampled_f32_path"] = "rows/" + row["id"] + "/whole_16k.f32"
    row["resampled_f32_sha256"] = put(within(output, row["resampled_f32_path"]), derived)
    pcm = quantize(derived)
    row.update(pcm_path="pcm/" + row["id"] + ".pcm", wav_path="wav/" + row["id"] + ".wav")
    row["pcm_sha256"] = put(within(output, row["pcm_path"]), pcm)
    need(row["pcm_sha256"] not in old["pcm_sha256"] | old["parent_pcm_sha256"]
         and row["pcm_sha256"] not in seen["pcm_sha256"] | seen["parent_pcm_sha256"],
         "NO_REUSED_FINAL_PCM")
    seen["pcm_sha256"].add(row["pcm_sha256"])
    row["wav_sha256"] = put(within(output, row["wav_path"]), native._wav(pcm, 16000))
    row.update(pcm_samples=len(pcm) // 2, sample_rate=16000, sample_width=2, channels=1,
               crop_calls=0, padding_samples=0, normalization_applied=False, quantization_calls=1,
               parent_context_weak=False, label_transfer_from_parent=False,
               resampler=recipe.SYNTHESIS["resampling"], **pcm_statistics(pcm),
               **whole_support_geometry(len(pcm) // 2))


def _empty_row(row):
    return copy.deepcopy(row) | {"generation_status": "not_started", "request_started": False,
        "tts_attempts": 0, "tts_returned": False, "capture_saved": False,
        "resample_calls": 0, "resample_returned": False,
        "pcm_path": None, "pcm_sha256": None, "wav_path": None, "wav_sha256": None,
        "parent_pcm_sha256": None, "pcm_samples": None, "sample_rate": 16000,
        "channels": 1, "sample_width": 2, "error": None}


def generate(plan, output, assets, run_identity, backend_factory=None):
    """Default path is guarded public execution; backend_factory is a mock seam."""
    source_preflight(plan)
    if backend_factory is None:
        import english_mero_meh_once as worker
        need(worker.execution_identity(plan) == run_identity, "FIRST_PUBLIC_MAIN_SOURCE_ONLY")
    output, assets = Path(output).absolute(), Path(assets).absolute()
    need(output != assets and output not in assets.parents and assets not in output.parents, "SEPARATE_SOURCE_ASSETS")
    output = native.empty_directory(output)
    need(not assets.exists() and not any(p.is_symlink() for p in (assets, *assets.parents)), "FRESH_ASSET_CACHE")
    rows = [_empty_row(r) for r in recipe.rows()]
    once(output / "RECIPE.json", recipe.recipe())
    once(output / "SOURCE_IDENTITY_CONTRACT.json", recipe.identity_contract())
    once(output / "EXECUTION_PLAN.json", plan)
    put(output / "EXCLUSION_EVIDENCE.json", regular(ROOT / "EXCLUSION_EVIDENCE.json", 16 * 1024**2))
    once(output / "STARTED.json", {"schema": SCHEMA, "run": run_identity, "campaign_id": recipe.CAMPAIGN_ID,
         "denominator": 144, "tts_attempts": 0, "recipe_sha256": recipe.digest(recipe.recipe())})
    old, seen = exclusion_sets(), {"pcm_sha256": set(), "parent_pcm_sha256": set()}
    started, failure = time.monotonic(), None
    def expired(signum, frame):
        raise GenerationInterrupted("SOURCE_DEADLINE_OR_TERMINATION")
    previous = {sig: signal.signal(sig, expired) for sig in (signal.SIGALRM, signal.SIGTERM)}
    try:
        signal.alarm(plan["time_budget_seconds"]["initialization"])
        backend = (backend_factory or _load_native)(output, assets)
        signal.alarm(plan["time_budget_seconds"]["generation"])
        for row in rows:
            folder = within(output, "rows/" + row["id"])
            try:
                ids = native.phoneme_ids(row["raw_phonemes"], backend.vocab, 512)
                request = {"id": row["id"], "request_sha256": row["request_sha256"],
                    "raw_phonemes": row["raw_phonemes"], "input_ids": ids, "preset_id": row["preset_id"],
                    "voice_pack_sha256": recipe.VOICE_ASSETS[row["preset_id"]]["sha256"],
                    "voice_pack_index": len(row["raw_phonemes"]) - 1, "seed": row["seed"],
                    "speed": row["speed"], "direct_api": "KModel.forward", "runtime_g2p_calls": 0,
                    "model_sha256": recipe.MODEL["model_sha256"], "retry_count": 0}
                once(folder / "STARTED.json", request)
                row.update(request_started=True, generation_status="failed", tts_attempts=1)
                result = backend.forward(row, ids)
                row["tts_returned"] = True
                once(folder / "FORWARD_RETURNED.json", {"id": row["id"], "request_sha256": row["request_sha256"]})
                raw, duration = backend.materialize(result)
                _write_audio(output, row, raw, duration, ids, backend, old, seen)
                row.update(generation_status="success", capture_saved=True)
                once(folder / "COMPLETED.json", copy.deepcopy(row))
            except BaseException as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:2000]
                once(folder / "FAILED.json", copy.deepcopy(row))
                if not isinstance(exc, Exception):
                    raise
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    successful = sum(r["generation_status"] == "success" for r in rows)
    result = {"schema": SCHEMA, "stage": "source", "campaign_id": recipe.CAMPAIGN_ID,
        "run": run_identity, "status": "COMPLETE" if successful == 144 and failure is None else "INCOMPLETE",
        "denominator": 144, "tts_attempts": sum(r["tts_attempts"] for r in rows),
        "tts_returned": sum(r["tts_returned"] for r in rows), "successful_count": successful,
        "failed_count": sum(r["generation_status"] == "failed" for r in rows),
        "not_started_count": sum(r["generation_status"] == "not_started" for r in rows),
        "resample_attempts": sum(r["resample_calls"] for r in rows),
        "resample_returns": sum(r["resample_returned"] for r in rows),
        "initialization_or_execution_error": failure, "retry_count": 0, "warmup_count": 0,
        "elapsed_seconds": round(time.monotonic() - started, 6)}
    corpus = {"schema": "english-mero-meh-corpus-v1", "campaign_id": recipe.CAMPAIGN_ID,
        "source_design_id": recipe.CAMPAIGN_ID, "run": run_identity, "rows": rows,
        "recipe_sha256": recipe.digest(recipe.recipe()),
        "source_identity_contract_sha256": recipe.digest(recipe.identity_contract()),
        "requested_text_is_observation": False, "diagnostic_only": False, "training_allowed": True,
        "human_gold": False, "parent_context_weak_policy": False}
    once(output / "CORPUS_MANIFEST.json", corpus)
    once(output / "SOURCE_RESULT.json", result)
    once(output / "RESULT.json", result)
    once(output / "FILES.json", {"schema": FILES_SCHEMA, "files": file_records(output)})
    return result


def verify_native_lock(root):
    """Byte-only reuse of the already verified 548 + 140 native weight proof."""
    import english_kokoro48_once as retained
    lock = read(root / "source/MODEL_LOCK.json")
    expected = {"schema": SCHEMA + "/native-lock", "campaign_id": recipe.CAMPAIGN_ID,
        "model": recipe.MODEL, "runtime_versions": recipe.RUNTIME_VERSIONS,
        "all_component_weights_verified": True, "native_graph_or_weights_modified": False,
        "preset_mixing": False, "runtime_g2p_calls": 0, "warmup_calls": 0, "forward_attempts_at_lock": 0}
    need(all(lock.get(k) == v for k, v in expected.items()), "NATIVE_MODEL_RUNTIME_LOCK")
    assets = lock.get("assets")
    need(type(assets) is list and len(assets) == 15
         and len({a["path"] for a in assets}) == 15, "FIFTEEN_EXACT_ASSETS")
    by_path = {a["path"]: a for a in assets}
    for pin in recipe.assets():
        saved = by_path.get(pin["path"], {})
        need(all(saved.get(k) == v for k, v in pin.items()) and saved.get("verified") is True
             and saved.get("download_attempts") == 1, "VERIFIED_PINNED_ASSET")
        if pin["path"] != recipe.MODEL["model_path"]:
            raw = regular(within(root, "source/runtime/" + pin["path"]), 1024**2)
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "RETAINED_RUNTIME_ASSET")
    package = lock.get("installed_package_files")
    need(type(package) is list and len(package) == len(recipe.PACKAGE_FILES)
         and {p["path"]: [p["bytes"], p["sha256"]] for p in package} == recipe.PACKAGE_FILES,
         "EXACT_NATIVE_INSTALLED_CODE")
    for name, (size, digest) in recipe.PACKAGE_FILES.items():
        raw = regular(within(root, "source/runtime/installed/" + name), 1024**2)
        need(len(raw) == size and sha(raw) == digest, "NATIVE_CODE_BYTES")
    raw = regular(root / "source/runtime" / recipe.PACKAGE["filename"], 1024**2)
    need(len(raw) == recipe.PACKAGE["bytes"] and sha(raw) == recipe.PACKAGE["sha256"], "NATIVE_WHEEL_BYTES")
    shapes = lock.get("voice_pack_shapes")
    need(type(shapes) is dict and set(shapes) == set(recipe.PRESETS)
         and all(type(s) is list and len(s) == 3 and type(s[0]) is int and s[0] > 100
                 and s[1:] == [1, 256] for s in shapes.values()), "TWELVE_ORIGINAL_VOICE_PACKS")
    retained.verify_weight_diagnostics(root, lock)
    loading, constructed = read(root / "source/MODEL_LOAD_STARTED.json"), read(root / "source/MODEL_CONSTRUCTED.json")
    need(loading.get("assets_verified") == assets and loading.get("forward_attempts") == 0
         and loading.get("local_files_only") is True and loading.get("campaign_id") == recipe.CAMPAIGN_ID
         and constructed.get("constructor_and_runtime_transforms_returned") is True
         and constructed.get("model_forward_calls") == 0, "INITIALIZATION_BEFORE_ANY_FORWARD")
    return read(root / "source/runtime/config.json")["vocab"]


def _verify_row(root, row, old, seen, vocab):
    folder = within(root, "rows/" + row["id"])
    started, returned = (folder / "STARTED.json").is_file(), (folder / "FORWARD_RETURNED.json").is_file()
    need(row["request_started"] == started and row["tts_attempts"] == int(started)
         and row["tts_returned"] == returned, "SOURCE_CALL_MARKERS")
    if started:
        request = read(folder / "STARTED.json")
        for key in ("id", "request_sha256", "raw_phonemes", "preset_id", "seed", "speed"):
            need(request.get(key) == row[key], "EXACT_NATIVE_REQUEST_" + key)
        need(request.get("runtime_g2p_calls") == 0 and request.get("retry_count") == 0
             and request.get("voice_pack_sha256") == recipe.VOICE_ASSETS[row["preset_id"]]["sha256"]
             and request.get("voice_pack_index") == len(row["raw_phonemes"]) - 1
             and request.get("model_sha256") == recipe.MODEL["model_sha256"], "FIXED_NATIVE_INPUT")
        need(request.get("input_ids") == native.phoneme_ids(row["raw_phonemes"], vocab, 512), "SAVED_NATIVE_TOKEN_IDS")
    if row["generation_status"] != "success":
        need(not row["capture_saved"] and not (folder / "COMPLETED.json").exists(), "NO_UNRECORDED_SUCCESS")
        return False
    need(started and returned and row["capture_saved"] and not (folder / "FAILED.json").exists()
         and read(folder / "COMPLETED.json") == row, "EXACT_COMPLETED_ROW")
    raw = regular(within(root, row["native_f32_path"]), 4 * 288000)
    values = _float_values(raw)
    need(sha(raw) == row["parent_pcm_sha256"] and len(values) == row["native_samples"]
         and row["parent_sample_rate"] == row["native_sample_rate"] == 24000
         and row["parent_pcm_format"] == "float32-little-endian"
         and row["parent_scope"] == "ENTIRE_NATIVE_OUTPUT", "NATIVE_PARENT_BYTES")
    duration_raw = regular(within(root, row["duration_path"]), 1024**2)
    duration = json.loads(duration_raw)
    need(sha(duration_raw) == row["duration_sha256"] and duration["raw_phonemes"] == row["raw_phonemes"]
         and duration["input_ids"] == request["input_ids"] and len(duration["pred_dur"]) == len(row["raw_phonemes"]) + 2
         and all(type(x) is int and 1 <= x <= 200 for x in duration["pred_dur"])
         and sum(duration["pred_dur"]) * 600 == len(values), "NATIVE_DURATION_BYTES")
    derived = regular(within(root, row["resampled_f32_path"]), 4 * 192000)
    need(sha(derived) == row["resampled_f32_sha256"]
         and len(derived) // 4 == (len(values) * 2 + 2) // 3, "WHOLE_RESAMPLED_BYTES")
    pcm = regular(within(root, row["pcm_path"]), 2 * 192000)
    need(sha(pcm) == row["pcm_sha256"] and len(pcm) == row["pcm_samples"] * 2
         and pcm == quantize(derived), "EXACT_FINAL_PCM")
    wav_raw = regular(within(root, row["wav_path"]), 2 * 192000 + 1024)
    with wave.open(io.BytesIO(wav_raw), "rb") as wav:
        need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 16000
             and wav.getnframes() == row["pcm_samples"] and wav.readframes(wav.getnframes()) == pcm
             and wav.getcomptype() == "NONE", "SAME_COMPLETE_WAV_PCM")
    need(sha(wav_raw) == row["wav_sha256"] and row["sample_rate"] == 16000
         and row["sample_width"] == 2 and row["channels"] == 1, "WAV_HASH_FORMAT")
    need(all(row.get(k) == v for k, v in (pcm_statistics(pcm) | whole_support_geometry(row["pcm_samples"])).items()),
         "BYTE_DERIVED_QUALITY_GEOMETRY")
    need(row["resample_calls"] == 1 and row["resample_returned"] is True and row["crop_calls"] == 0
         and row["padding_samples"] == 0 and row["normalization_applied"] is False
         and row["label_transfer_from_parent"] is False, "WHOLE_SOURCE_TRANSFORM_CONTRACT")
    for key in ("pcm_sha256", "parent_pcm_sha256"):
        need(row[key] not in old["pcm_sha256"] | old["parent_pcm_sha256"]
             and row[key] not in seen["pcm_sha256"] | seen["parent_pcm_sha256"], "NO_OLD_OR_DUPLICATE_AUDIO")
        seen[key].add(row[key])
    return True


def validate_source(root, expected_run=None):
    """Validate saved source bytes without loading models or repeating resampling."""
    root = Path(root)
    source_preflight()
    need(read(root / "FILES.json") == {"schema": FILES_SCHEMA, "files": file_records(root)}, "EXACT_SOURCE_FILE_CLOSURE")
    corpus, terminal = read(root / "CORPUS_MANIFEST.json"), read(root / "SOURCE_RESULT.json")
    need(corpus.get("campaign_id") == corpus.get("source_design_id") == recipe.CAMPAIGN_ID
         and corpus.get("recipe_sha256") == recipe.digest(recipe.recipe())
         and corpus.get("source_identity_contract_sha256") == recipe.digest(recipe.identity_contract())
         and read(root / "RECIPE.json") == recipe.recipe()
         and read(root / "SOURCE_IDENTITY_CONTRACT.json") == recipe.identity_contract(), "FROZEN_SOURCE_CONTRACT")
    need(regular(root / "EXCLUSION_EVIDENCE.json", 16 * 1024**2)
         == regular(ROOT / "EXCLUSION_EVIDENCE.json", 16 * 1024**2), "EXACT_EXCLUSION_EVIDENCE")
    need(corpus["run"] == terminal["run"] and (expected_run is None or corpus["run"] == expected_run), "SOURCE_RUN_BINDING")
    planned, rows = recipe.rows(), corpus["rows"]
    need(type(rows) is list and len(rows) == 144
         and [r["id"] for r in rows] == [r["id"] for r in planned], "ALL_144_PLANNED_ROWS")
    old, seen, successful = exclusion_sets(), {"pcm_sha256": set(), "parent_pcm_sha256": set()}, []
    vocab = verify_native_lock(root) if any(row.get("tts_attempts") for row in rows) else None
    for expected, row in zip(planned, rows):
        need(all(row.get(k) == v for k, v in expected.items()), "UNCHANGED_PREREGISTERED_REQUEST")
        need(row.get("generation_status") in ("success", "failed", "not_started"), "FIXED_SOURCE_STATUS")
        if _verify_row(root, row, old, seen, vocab):
            successful.append(row)
    counts = {"denominator": 144, "tts_attempts": sum(r["tts_attempts"] for r in rows),
        "tts_returned": sum(r["tts_returned"] for r in rows), "successful_count": len(successful),
        "failed_count": sum(r["generation_status"] == "failed" for r in rows),
        "not_started_count": sum(r["generation_status"] == "not_started" for r in rows),
        "resample_attempts": sum(r["resample_calls"] for r in rows),
        "resample_returns": sum(r["resample_returned"] for r in rows)}
    need(all(terminal.get(k) == v for k, v in counts.items()), "BYTE_DERIVED_SOURCE_COUNTS")
    need(0 <= counts["successful_count"] <= counts["tts_returned"] <= counts["tts_attempts"] <= 144
         and counts["successful_count"] + counts["failed_count"] + counts["not_started_count"] == 144,
         "COMPLETE_SOURCE_DENOMINATOR")
    return corpus, successful


def source_readback(root, plan=None, run_identity=None):
    source_preflight(plan)
    corpus, successful = validate_source(root, run_identity)
    result = read(Path(root) / "SOURCE_RESULT.json")
    return result | {"source_ready": len(successful) == 144 and result["status"] == "COMPLETE",
        "new_model_calls": 0, "resampling_reexecuted": False,
        "source_manifest_sha256": sha(regular(Path(root) / "CORPUS_MANIFEST.json", 16 * 1024**2)),
        "rows": corpus["rows"]}
