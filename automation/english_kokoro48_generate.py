"""One bounded native Kokoro probe. Heavy imports occur only in generate().

The caller owns GitHub/event/parent supervision and the root artifact receipt.
This module owns source/ and one initially empty private asset directory. Its
audio is diagnostic and cannot become a training admission. No result-dependent
recrop, regeneration, preset substitution, G2P or additional forward is allowed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import urllib.request
import wave

_spec = importlib.util.spec_from_file_location("english_kokoro48_recipe",
    Path(__file__).with_name("english_kokoro48_recipe.py"))
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)
SCHEMA = "english-kokoro48-generation-v1"

# Exact native-only AdaIN affine sites observed in public run 44122224557.
# Every admitted value is read back against its neutral value; none is written.
_NATIVE_AFFINE_CONTRACT = {'actual_diagnostic_artifact_sha256': '88e22b8cb7f2d9b410030063b385ba3894a7ca24e05d27f32a07dec42f352b1e',
 'actual_diagnostic_run': '44122224557',
 'checkpoint_key_counts': {'bert': 25,
                           'bert_encoder': 2,
                           'decoder': 375,
                           'predictor': 122,
                           'text_encoder': 24},
 'generic_missing_key_allowance': False,
 'kokoro_version': '0.9.4',
 'model_sha256': '496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4',
 'native_extra_scalar_count': 46612,
 'native_extra_tensor_count': 140,
 'native_type': 'torch.nn.modules.instancenorm.InstanceNorm1d',
 'norm_count': 70,
 'norms': {'decoder': {'decode.0.norm1.norm': 1090,
                       'decode.0.norm2.norm': 1024,
                       'decode.1.norm1.norm': 1090,
                       'decode.1.norm2.norm': 1024,
                       'decode.2.norm1.norm': 1090,
                       'decode.2.norm2.norm': 1024,
                       'decode.3.norm1.norm': 1090,
                       'decode.3.norm2.norm': 512,
                       'encode.norm1.norm': 514,
                       'encode.norm2.norm': 1024,
                       'generator.noise_res.0.adain1.0.norm': 256,
                       'generator.noise_res.0.adain1.1.norm': 256,
                       'generator.noise_res.0.adain1.2.norm': 256,
                       'generator.noise_res.0.adain2.0.norm': 256,
                       'generator.noise_res.0.adain2.1.norm': 256,
                       'generator.noise_res.0.adain2.2.norm': 256,
                       'generator.noise_res.1.adain1.0.norm': 128,
                       'generator.noise_res.1.adain1.1.norm': 128,
                       'generator.noise_res.1.adain1.2.norm': 128,
                       'generator.noise_res.1.adain2.0.norm': 128,
                       'generator.noise_res.1.adain2.1.norm': 128,
                       'generator.noise_res.1.adain2.2.norm': 128,
                       'generator.resblocks.0.adain1.0.norm': 256,
                       'generator.resblocks.0.adain1.1.norm': 256,
                       'generator.resblocks.0.adain1.2.norm': 256,
                       'generator.resblocks.0.adain2.0.norm': 256,
                       'generator.resblocks.0.adain2.1.norm': 256,
                       'generator.resblocks.0.adain2.2.norm': 256,
                       'generator.resblocks.1.adain1.0.norm': 256,
                       'generator.resblocks.1.adain1.1.norm': 256,
                       'generator.resblocks.1.adain1.2.norm': 256,
                       'generator.resblocks.1.adain2.0.norm': 256,
                       'generator.resblocks.1.adain2.1.norm': 256,
                       'generator.resblocks.1.adain2.2.norm': 256,
                       'generator.resblocks.2.adain1.0.norm': 256,
                       'generator.resblocks.2.adain1.1.norm': 256,
                       'generator.resblocks.2.adain1.2.norm': 256,
                       'generator.resblocks.2.adain2.0.norm': 256,
                       'generator.resblocks.2.adain2.1.norm': 256,
                       'generator.resblocks.2.adain2.2.norm': 256,
                       'generator.resblocks.3.adain1.0.norm': 128,
                       'generator.resblocks.3.adain1.1.norm': 128,
                       'generator.resblocks.3.adain1.2.norm': 128,
                       'generator.resblocks.3.adain2.0.norm': 128,
                       'generator.resblocks.3.adain2.1.norm': 128,
                       'generator.resblocks.3.adain2.2.norm': 128,
                       'generator.resblocks.4.adain1.0.norm': 128,
                       'generator.resblocks.4.adain1.1.norm': 128,
                       'generator.resblocks.4.adain1.2.norm': 128,
                       'generator.resblocks.4.adain2.0.norm': 128,
                       'generator.resblocks.4.adain2.1.norm': 128,
                       'generator.resblocks.4.adain2.2.norm': 128,
                       'generator.resblocks.5.adain1.0.norm': 128,
                       'generator.resblocks.5.adain1.1.norm': 128,
                       'generator.resblocks.5.adain1.2.norm': 128,
                       'generator.resblocks.5.adain2.0.norm': 128,
                       'generator.resblocks.5.adain2.1.norm': 128,
                       'generator.resblocks.5.adain2.2.norm': 128},
           'predictor': {'F0.0.norm1.norm': 512,
                         'F0.0.norm2.norm': 512,
                         'F0.1.norm1.norm': 512,
                         'F0.1.norm2.norm': 256,
                         'F0.2.norm1.norm': 256,
                         'F0.2.norm2.norm': 256,
                         'N.0.norm1.norm': 512,
                         'N.0.norm2.norm': 512,
                         'N.1.norm1.norm': 512,
                         'N.1.norm2.norm': 256,
                         'N.2.norm1.norm': 256,
                         'N.2.norm2.norm': 256}},
 'official_istftnet_sha256': 'eb45e18d73d09e5a42801dedf16b6e3fb07d9741a142cd48fb8ebc1f68c11f26',
 'parameter_mutation_allowed': False,
 'parent_type': 'kokoro.istftnet.AdaIN1d',
 'required_affine': True,
 'required_bias_value': 0.0,
 'required_device': 'cpu',
 'required_dtype': 'torch.float32',
 'required_eps': 1e-05,
 'required_momentum': 0.1,
 'required_track_running_stats': False,
 'required_weight_value': 1.0,
 'schema': 'kokoro-v1_0-native-neutral-adain-v1',
 'torch_version': '2.9.1+cpu'}


def native_affine_contract():
    return json.loads(json.dumps(_NATIVE_AFFINE_CONTRACT))


def expected_native_extra_keys(component=None):
    result = []
    for name, paths in sorted(_NATIVE_AFFINE_CONTRACT["norms"].items()):
        if component is not None and name != component:
            continue
        for path in sorted(paths):
            for suffix in ("bias", "weight"):
                result.append((name + "." if component is None else "") + path + "." + suffix)
    return result


def native_affine_contract_sha256():
    return recipe.digest(native_affine_contract())


def need(condition, code):
    if not condition:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def within(root, relative):
    need(type(relative) is str, "RELATIVE_PATH_STRING")
    part = PurePosixPath(relative)
    need(relative == part.as_posix() and not part.is_absolute() and part.parts and
         all(x not in (".", "..", "") for x in part.parts), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*part.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK")
    return path


def regular(path, limit):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "REGULAR_FILE")
    need(0 < path.stat().st_size <= limit, "BOUNDED_FILE")
    return path.read_bytes()


def put(path, raw):
    path = Path(path)
    need(type(raw) is bytes and not path.exists() and not path.is_symlink(), "CREATE_ONCE_BYTES")
    need(not any(p.is_symlink() for p in path.parents), "NO_SYMLINK_PARENT")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def once(path, value):
    return put(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def empty_directory(path):
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK_DIRECTORY")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(path.is_dir() and not any(path.iterdir()), "EMPTY_CREATE_ONCE_DIRECTORY")
    return path


def validate_plan(plan, *, executing=False):
    frozen = recipe.recipe()
    need(type(plan) is dict, "PLAN_OBJECT")
    expected = {"campaign_id": recipe.CAMPAIGN_ID, "recipe_sha256": recipe.digest(frozen),
        "maximum_tts_calls": 48, "retry_count": 0, "diagnostic_only": True,
        "training_allowed": False, "generation_deadline_seconds": 900}
    for key, value in expected.items():
        need(type(plan.get(key)) is type(value) and plan[key] == value, "PLAN_" + key.upper())
    need(type(plan.get("enabled")) is bool and (not executing or plan["enabled"]), "PLAN_ENABLED")
    need(len(frozen["rows"]) == 48 and len({r["id"] for r in frozen["rows"]}) == 48, "FIXED_48_UNIQUE_REQUESTS")
    need(all(r["role"] == "train" and r["diagnostic_only"] and not r["training_allowed"]
             for r in frozen["rows"]), "TRAIN_ONLY_DIAGNOSTIC")
    return frozen


def phoneme_ids(phonemes, vocab, context_length=512):
    """Reject unknowns before upstream's silent vocab filtering can occur."""
    need(type(phonemes) is str and 1 <= len(phonemes) <= context_length - 2, "PHONEME_CONTEXT_LENGTH")
    need(type(vocab) is dict and all(p in vocab for p in phonemes), "UNKNOWN_PHONEME_TOKEN")
    ids = [vocab[p] for p in phonemes]
    need(all(type(i) is int and 0 < i < 178 for i in ids), "PHONEME_TOKEN_ID_RANGE")
    return [0, *ids, 0]


def crop_bounds(row, duration, native_samples):
    """Frozen raw-character duration estimate, deliberately not acoustic gold."""
    ps = row["raw_phonemes"]
    need(type(duration) is list and len(duration) == len(ps) + 2 and
         all(type(x) is int and 1 <= x <= 50 for x in duration), "NATIVE_DURATION_VECTOR")
    need(type(native_samples) is int and native_samples == 600 * sum(duration), "NATIVE_DURATION_SAMPLE_GEOMETRY")
    mode = row["mode"]
    need(mode in ("short", "carrier", "ordinary"), "FIXED_MODE")
    a = row["target_start_char"]
    need(type(a) is int and 0 <= a < len(ps) and ps[a:] == row["target_phonemes"], "EXACT_TARGET_CHARACTER_RANGE")
    if mode == "carrier":
        need(a == len(recipe.CARRIER_PHONEMES) and ps[:a] == recipe.CARRIER_PHONEMES,
             "EXACT_PREREGISTERED_CARRIER")
        predicted = 600 * (max(duration[0] - 3, 0) + sum(duration[1:a+1]))
        start = max(0, predicted - 3600)
    else:
        need(a == 0 and ps == row["target_phonemes"], "WHOLE_SHORT_OR_ORDINARY")
        predicted = 600 * max(duration[0] - 3, 0)
        start = 0
    need(0 <= start < native_samples and 0 <= predicted < native_samples, "CROP_INSIDE_NATIVE_PCM")
    return {
        "schema": SCHEMA + "/crop", "id": row["id"], "mode": mode,
        "target_start_char": a, "target_end_char_exclusive": len(ps),
        "prefix_character_range": [0, a], "target_character_range": [a, len(ps)],
        "predicted_target_start_native_sample": predicted,
        "crop_start_native_sample": start, "crop_end_native_sample": native_samples,
        "crop_native_samples": native_samples - start,
        "padding_before_samples": 3600 if mode == "carrier" else 0,
        "padding_is_added_silence": False, "actual_added_silence_samples": 0,
        "boundary_source": "kokoro_predicted_duration" if mode == "carrier" else "whole_native_utterance",
        "right_endpoint": "native_eof", "human_word_end": False,
        "boundary_is_acoustic_gold": False, "recrop_count": 0,
        "observer_feedback_used": False, "prediction_rule": dict(recipe.CROP),
    }


def _download(asset, destination):
    """A single bounded request; no automatic download or synthesis retry."""
    destination = Path(destination)
    need(not destination.exists() and not destination.is_symlink(), "NEW_ASSET_PATH")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "kws-kokoro48-source-audit/1"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            need(total <= asset["bytes"], "ASSET_SIZE_CEILING")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    need(total == asset["bytes"] and digest.hexdigest() == asset["sha256"], "EXACT_OFFICIAL_ASSET")
    return {"path": asset["path"], "url": asset["url"], "bytes": total,
            "sha256": digest.hexdigest(), "verified": True, "download_attempts": 1}


def _installed_runtime(output):
    versions = {name: importlib.metadata.version(name) for name in recipe.RUNTIME_VERSIONS}
    need(versions == recipe.RUNTIME_VERSIONS, "EXACT_RUNTIME_PACKAGE_VERSIONS")
    need(sys.version_info[:2] == (3, 12), "PYTHON_3_12_RUNTIME")
    distribution = importlib.metadata.distribution("kokoro")
    package_files = []
    for name, (size, expected_sha) in recipe.PACKAGE_FILES.items():
        raw = regular(distribution.locate_file(name), size)
        need(len(raw) == size and sha(raw) == expected_sha, "EXACT_INSTALLED_KOKORO_SOURCE")
        put(within(output, "source/runtime/installed/" + name), raw)
        package_files.append({"path": name, "bytes": size, "sha256": sha(raw)})
    all_versions = sorted({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}.items())
    once(output / "source/runtime/DEPENDENCIES.json", {"required": versions,
         "actual_distributions": dict(all_versions), "python": sys.version,
         "runtime_g2p_calls": 0, "implicit_hub_model_loading_allowed": False})
    return versions, package_files


def _tensor_description(value, torch):
    if not isinstance(value, torch.Tensor):
        return {"is_tensor": False, "type": type(value).__name__}
    return {"is_tensor": True, "shape": list(value.shape), "dtype": str(value.dtype),
            "device_type": value.device.type}


def _module_inventory(module, torch):
    parameters = [{"key": key, **_tensor_description(value, torch)}
        for key, value in module.named_parameters(recurse=True, remove_duplicate=False)]
    buffers = []
    for prefix, child in module.named_modules(remove_duplicate=False):
        for key, value in child.named_buffers(recurse=False, remove_duplicate=False):
            buffers.append({"key": (prefix + "." if prefix else "") + key,
                "persistent": key not in child._non_persistent_buffers_set,
                **_tensor_description(value, torch)})
    return {"named_parameters": parameters, "named_buffers": buffers}


def _native_affine_readback(module, component, loaded, torch, native_adain_type):
    """Read only the fixed native AdaIN sites; never repair or reload a value."""
    checks = []
    for path, channels in sorted(_NATIVE_AFFINE_CONTRACT["norms"].get(component, {}).items()):
        check = {"path": path, "num_features": channels, "passed": False,
            "parameters_mutated": False, "error": None}
        try:
            parent_path = path.rsplit('.', 1)[0]
            parent = module.get_submodule(parent_path)
            norm = module.get_submodule(path)
            check.update(parent_path=parent_path,
                parent_type=type(parent).__module__ + '.' + type(parent).__name__,
                norm_type=type(norm).__module__ + '.' + type(norm).__name__)
            need(type(parent) is native_adain_type and parent.norm is norm
                and type(norm) is torch.nn.InstanceNorm1d, "EXACT_NATIVE_ADAIN_PARENT_AND_NORM_TYPE")
            attributes = {name: getattr(norm, name) for name in
                ("num_features", "affine", "track_running_stats", "eps", "momentum", "training")}
            check.update(attributes=attributes, parent_training=parent.training)
            need(type(norm.num_features) is int and norm.num_features == channels
                and norm.affine is True and norm.track_running_stats is False
                and type(norm.eps) is float and norm.eps == 1e-5
                and type(norm.momentum) is float and norm.momentum == 0.1
                and norm.training is False and parent.training is False,
                "EXACT_NATIVE_ADAIN_NEUTRAL_CONFIGURATION")
            check["running_buffers_are_none"] = {name: getattr(norm, name) is None
                for name in ("running_mean", "running_var", "num_batches_tracked")}
            need(set(norm._buffers) == {"running_mean", "running_var", "num_batches_tracked"}
                and all(check["running_buffers_are_none"].values()), "NATIVE_ADAIN_NO_RUNNING_STATISTICS")
            need(set(norm._parameters) == {"weight", "bias"}, "NATIVE_ADAIN_EXACT_AFFINE_PARAMETERS")
            values = []
            for suffix, neutral_value in (("weight", 1), ("bias", 0)):
                value, key = getattr(norm, suffix), path + "." + suffix
                stored = loaded[key]
                description = _tensor_description(value, torch)
                stored_description = _tensor_description(stored, torch)
                record = {"key": key, "required_neutral_value": neutral_value,
                    "parameter": description, "state_dict": stored_description,
                    "exact_neutral_value": False, "state_dict_value_equal": False}
                values.append(record)
                check["parameters"] = values
                need(type(value) is torch.nn.Parameter and description["is_tensor"]
                    and stored_description["is_tensor"] and tuple(value.shape) == (channels,)
                    and tuple(stored.shape) == (channels,)
                    and value.dtype == stored.dtype == torch.float32
                    and value.device.type == stored.device.type == "cpu", "NATIVE_ADAIN_EXACT_PARAMETER_METADATA")
                expected = torch.ones_like(value) if neutral_value else torch.zeros_like(value)
                record["exact_neutral_value"] = bool(torch.equal(value, expected))
                record["state_dict_value_equal"] = bool(torch.equal(stored, value))
                need(record["exact_neutral_value"] and record["state_dict_value_equal"],
                    "NATIVE_ADAIN_PARAMETER_NOT_NEUTRAL")
            check["passed"] = True
        except Exception as exc:
            check["error"] = {"type": type(exc).__name__, "code": str(exc)[:300]}
        checks.append(check)
    return checks


def _strict_checkpoint_readback(model, checkpoint, torch, *, output=None, native_adain_type=None):
    """Upstream has a strict=False fallback; never accept it as load proof."""
    expected_components = {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
    need(type(checkpoint) is dict and set(checkpoint) == expected_components, "CHECKPOINT_COMPONENT_CLOSURE")
    native_contract = native_adain_type is not None
    if native_contract:
        need(native_adain_type.__module__ + '.' + native_adain_type.__name__ ==
            _NATIVE_AFFINE_CONTRACT["parent_type"], "PINNED_NATIVE_ADAIN_CLASS")
    components, diagnostics, component_records = [], [], []
    for name in sorted(expected_components):
        diagnostic = {"schema": SCHEMA + "/checkpoint-component", "component": name,
            "passed": False, "error": None, "tensor_values_reloaded": False,
            "model_forward_calls": 0, "allowed_key_exceptions": [], "neutral_affine_checks": []}
        try:
            module = getattr(model, name)
            loaded = module.state_dict()
            diagnostic["loaded_state_keys"] = sorted(loaded)
            diagnostic.update(_module_inventory(module, torch))
            source = checkpoint[name]
            need(hasattr(source, "items") and source, "CHECKPOINT_COMPONENT_MAPPING")
            keys = list(source)
            need(all(type(k) is str for k in keys), "CHECKPOINT_KEY_STRINGS")
            diagnostic["source_keys"] = sorted(keys)
            if all(k.startswith("module.") for k in keys):
                state = {k[7:]: value for k, value in source.items()}
                mapping = "strip_exact_module_dot_prefix"
            else:
                need(not any(k.startswith("module.") for k in keys), "NO_MIXED_CHECKPOINT_PREFIXES")
                state = dict(source)
                mapping = "identity"
            diagnostic.update(mapping=mapping, normalized_keys=sorted(state),
                missing_in_checkpoint=sorted(set(loaded)-set(state)),
                unexpected_in_checkpoint=sorted(set(state)-set(loaded)))
            diagnostic["checkpoint_tensors"] = [{"key": key, **_tensor_description(value, torch)}
                for key, value in sorted(state.items())]
            comparisons = []
            for key in sorted(set(state).intersection(loaded)):
                left, right = state[key], loaded[key]
                a, b = _tensor_description(left, torch), _tensor_description(right, torch)
                comparable = (a["is_tensor"] and b["is_tensor"] and
                    a["device_type"] == b["device_type"] == "cpu" and
                    left.shape == right.shape and left.dtype == right.dtype)
                comparisons.append({"key": key, "checkpoint": a, "loaded": b,
                    "exact_value_equal": bool(torch.equal(left, right)) if comparable else False})
            diagnostic["common_key_comparisons"] = comparisons
            diagnostic["mismatched_common_keys"] = [r["key"] for r in comparisons if not r["exact_value_equal"]]
            extra_keys = set(expected_native_extra_keys(name)) if native_contract else set()
            diagnostic["allowed_key_exceptions"] = sorted(extra_keys)
            need(not set(state).intersection(extra_keys) and set(state) | extra_keys == set(loaded),
                "CHECKPOINT_EXACT_KEYS")
            need(not diagnostic["mismatched_common_keys"], "CHECKPOINT_EXACT_TENSOR_READBACK")
            if native_contract:
                need(len(state) == _NATIVE_AFFINE_CONTRACT["checkpoint_key_counts"][name],
                    "EXACT_PINNED_CHECKPOINT_COMPONENT_KEY_COUNT")
                checks = _native_affine_readback(module, name, loaded, torch, native_adain_type)
                diagnostic["neutral_affine_checks"] = checks
                failed_checks = [r for r in checks if not r["passed"]]
                need(not failed_checks, failed_checks[0]["error"]["code"] if failed_checks else "NATIVE_ADAIN_NEUTRAL_PROOF")
            diagnostic["passed"] = True
            components.append({"component": name, "checkpoint_key_count": len(state),
                "native_key_count": len(loaded), "mapping": mapping,
                "all_checkpoint_keys_shapes_dtypes_values_equal": True,
                "verified_native_neutral_affine_key_count": len(extra_keys)})
        except Exception as exc:
            diagnostic["error"] = {"type": type(exc).__name__, "code": str(exc)[:300]}
        diagnostics.append(diagnostic)
        # Save every component before rejecting the whole model. No generic
        # key allowance, strict=False reload, tensor mutation or inference is used.
        if output is not None:
            relative = "source/weight-diagnostics/" + name + ".json"
            path = Path(output) / relative
            digest = once(path, diagnostic)
            component_records.append({"component": name, "path": relative,
                "bytes": path.stat().st_size, "sha256": digest, "passed": diagnostic["passed"]})
    passed = len(components) == len(expected_components)
    if output is not None:
        once(Path(output) / "source/weight-diagnostics/SUMMARY.json", {
            "schema": SCHEMA + "/checkpoint-readback", "passed": passed,
            "component_count": len(diagnostics), "passed_components": len(components),
            "failed_components": [r["component"] for r in diagnostics if not r["passed"]],
            "component_files": ["source/weight-diagnostics/" + r["component"] + ".json" for r in diagnostics],
            "component_records": component_records,
            "weight_contract": _NATIVE_AFFINE_CONTRACT["schema"] if native_contract else "checkpoint-exact-only",
            "native_affine_contract_sha256": native_affine_contract_sha256() if native_contract else None,
            "checkpoint_tensor_count": sum(len(r.get("normalized_keys", [])) for r in diagnostics),
            "exact_checkpoint_tensor_count": sum(sum(c["exact_value_equal"] for c in r.get("common_key_comparisons", [])) for r in diagnostics),
            "native_tensor_count": sum(len(r.get("loaded_state_keys", [])) for r in diagnostics),
            "verified_native_neutral_affine_tensor_count": 2 * sum(sum(c["passed"] for c in r["neutral_affine_checks"]) for r in diagnostics),
            "model_forward_calls": 0, "tensor_values_reloaded": False,
            "allowed_key_exceptions": expected_native_extra_keys() if native_contract else []})
    if not passed:
        failed = next(r for r in diagnostics if not r["passed"])
        raise ValueError(failed["error"]["code"])
    return components


def _load_native(output, asset_root):
    versions, package_files = _installed_runtime(output)
    receipts = [_download(a, within(asset_root, a["path"])) for a in recipe.assets()]
    wheel = {"path": recipe.PACKAGE["filename"], **{k: recipe.PACKAGE[k] for k in ("url", "bytes", "sha256")}}
    wheel_receipt = _download(wheel, asset_root / wheel["path"])
    put(output / "source/runtime" / wheel["path"], regular(asset_root / wheel["path"], wheel["bytes"]))
    for name in ("config.json", "README.md", *[recipe.VOICE_ASSETS[p]["path"] for p in recipe.PRESETS]):
        raw = regular(within(asset_root, name), 1024 * 1024)
        put(within(output, "source/runtime/" + name), raw)
    # No library is allowed to fetch an unpinned alternative during construction.
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
    once(output / "source/MODEL_LOAD_STARTED.json", {"schema": SCHEMA, "assets_verified": receipts,
         "forward_attempts": 0, "local_files_only": True})
    model = KModel(repo_id=recipe.MODEL["repository"], config=str(asset_root / "config.json"),
                   model=str(asset_root / recipe.MODEL["model_path"]), disable_complex=False).to("cpu").float().eval()
    once(output / "source/MODEL_CONSTRUCTED.json", {"schema": SCHEMA + "/constructed",
        "constructor_and_runtime_transforms_returned": True, "model_instances": 1,
        "model_forward_calls": 0, "checkpoint_proof_pending": True})
    checkpoint = torch.load(asset_root / recipe.MODEL["model_path"], map_location="cpu", weights_only=True)
    component_proof = _strict_checkpoint_readback(model, checkpoint, torch, output=output,
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
    for row in recipe.recipe()["rows"]:
        phoneme_ids(row["raw_phonemes"], model.vocab, model.context_length)
        need(0 <= len(row["raw_phonemes"]) - 1 < len(packs[row["preset_id"]]), "NATIVE_VOICE_ROW_INDEX")
    lock = {"schema": SCHEMA + "/model-lock", "campaign_id": recipe.CAMPAIGN_ID,
        **dict(recipe.MODEL), "assets": receipts,
        "package": recipe.PACKAGE, "package_download": wheel_receipt,
        "installed_package_files": package_files, "runtime_versions": versions,
        "weight_loading": component_proof, "all_component_weights_verified": True,
        "weight_contract": _NATIVE_AFFINE_CONTRACT["schema"],
        "native_affine_contract": native_affine_contract(),
        "native_affine_contract_sha256": native_affine_contract_sha256(),
        "weight_diagnostics": "source/weight-diagnostics/SUMMARY.json",
        "weight_diagnostics_sha256": sha(regular(output / "source/weight-diagnostics/SUMMARY.json", 1024 * 1024)),
        "voice_pack_shapes": {k: list(v.shape) for k, v in packs.items()},
        "preset_mixing": False, "runtime_g2p_calls": 0, "device": "cpu", "dtype": "float32",
        "human_speaker_identity_claimed": False, "forward_attempts_at_lock": 0}
    once(output / "source/MODEL_LOCK.json", lock)

    def forward(row, ids):
        need(ids == phoneme_ids(row["raw_phonemes"], model.vocab, model.context_length), "NATIVE_FORWARD_TOKEN_IDENTITY")
        torch.manual_seed(row["seed"])
        with torch.inference_mode():
            result = model(row["raw_phonemes"], packs[row["preset_id"]][len(row["raw_phonemes"]) - 1],
                           speed=1.0, return_output=True)
        return result

    return forward, config["vocab"]


def materialize_native_result(result):
    # Called only after recording that the native forward returned. A malformed
    # output or conversion failure must not be miscounted as a failed forward.
    need(getattr(result, "pred_dur", None) is not None, "NATIVE_DURATION_RETURNED")
    return result.audio.detach().cpu().numpy(), result.pred_dur.detach().cpu().tolist()


def _pcm16(values, np):
    return np.rint(np.clip(values, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()


def _wav(pcm, rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return buffer.getvalue()


def write_audio(output, row, audio, duration, ids):
    """Persist native output before checking predicted crop geometry."""
    import numpy as np
    from scipy.signal import resample_poly
    need(isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1 and
         0 < len(audio) <= 600 * 50 * (len(row["raw_phonemes"]) + 2) and
         bool(np.isfinite(audio).all()), "BOUNDED_FINITE_NATIVE_FLOAT32")
    folder = "source/audio/" + row["id"] + "/"
    full_raw = audio.astype("<f4", copy=False).tobytes()
    full_pcm = _pcm16(audio, np)
    measured = {"native_samples": len(audio), "native_full_f32_path": folder + "native-full.f32",
                "native_full_wav_path": folder + "native-full.wav"}
    measured["native_full_f32_sha256"] = put(within(output, measured["native_full_f32_path"]), full_raw)
    measured["native_full_wav_sha256"] = put(within(output, measured["native_full_wav_path"]), _wav(full_pcm, 24000))
    duration_doc = {"schema": SCHEMA + "/native-duration", "id": row["id"],
        "phonemes": row["raw_phonemes"], "input_ids": ids, "pred_dur": duration,
        "includes_bos_eos": True, "samples_per_frame": 600, "sample_rate": 24000,
        "native_samples": len(audio), "boundary_is_acoustic_gold": False,
        "target_start_char": row["target_start_char"], "duration_source": "native_KModel_output"}
    measured["duration_path"] = folder + "pred_dur.json"
    measured["duration_sha256"] = once(within(output, measured["duration_path"]), duration_doc)
    crop = crop_bounds(row, duration, len(audio))
    crop.update(native_full_f32_sha256=measured["native_full_f32_sha256"],
                duration_sha256=measured["duration_sha256"])
    measured["crop_path"] = folder + "CROP.json"
    measured["crop_sha256"] = once(within(output, measured["crop_path"]), crop)
    start, end = crop["crop_start_native_sample"], crop["crop_end_native_sample"]
    native_crop = full_raw[start * 4:end * 4]
    measured["native_crop_f32_path"] = folder + "native-crop.f32"
    measured["native_crop_f32_sha256"] = put(within(output, measured["native_crop_f32_path"]), native_crop)
    values = np.frombuffer(native_crop, dtype="<f4")
    derived = resample_poly(values, 2, 3)
    need(len(derived) == (len(values) * 2 + 2) // 3 and bool(np.isfinite(derived).all()), "DERIVED_16K_GEOMETRY")
    pcm = _pcm16(derived, np)
    measured.update(pcm_path=folder + "audio-16k.pcm", wav_path=folder + "audio-16k.wav")
    measured["pcm_sha256"] = put(within(output, measured["pcm_path"]), pcm)
    measured["wav_sha256"] = put(within(output, measured["wav_path"]), _wav(pcm, 16000))
    measured.update({key: crop[key] for key in ("crop_start_native_sample", "crop_end_native_sample", "crop_native_samples")})
    measured.update(pcm_samples=len(pcm) // 2, sample_rate=16000, channels=1, sample_width=2,
        native_sample_rate=24000, native_wav_sample_width=2, native_float_format="float32-little-endian",
        native_peak=float(np.max(np.abs(audio))),
        native_rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        native_pcm16_clipped_samples=int(np.count_nonzero((audio < -1.0) | (audio > 32767 / 32768))),
        derived_pcm16_clipped_samples=int(np.count_nonzero((derived < -1.0) | (derived > 32767 / 32768))),
        resampler=recipe.SYNTHESIS["resampling"], normalization_applied=False,
        observed_target_pronunciation=False, synthetic_silence_added=False,
        human_gold=False, diagnostic_only=True, training_allowed=False)
    return measured


def _unmeasured(row):
    return dict(row) | {"generation_status": "not_started", "forward_attempts": 0,
        "forward_returned": False, "pcm_path": None, "wav_path": None,
        "pcm_sha256": None, "wav_sha256": None, "pcm_samples": None,
        "sample_rate": 16000, "channels": 1, "sample_width": 2}


def _run_rows(output, rows, forward, vocab, deadline, counters):
    for row in rows:
        folder = within(output, "source/audio/" + row["id"])
        try:
            need(time.monotonic() < deadline, "GENERATION_DEADLINE")
            ids = phoneme_ids(row["raw_phonemes"], vocab)
            need(counters["actual_forward_attempts"] < 48, "TTS_FORWARD_CEILING")
            once(folder / "STARTED.json", {"schema": SCHEMA, "id": row["id"], "attempt": 1,
                "request_sha256": row["request_sha256"], "preset_id": row["preset_id"],
                "mode": row["mode"], "role": "train", "phonemes": row["raw_phonemes"],
                "phoneme_ids": ids, "target_start_char": row["target_start_char"],
                "voice_pack_index": len(row["raw_phonemes"]) - 1, "seed": row["seed"],
                "synthesis_config": recipe.SYNTHESIS, "direct_api": "KModel.forward",
                "runtime_g2p_calls": 0, "retry_count": 0, "diagnostic_only": True,
                "training_allowed": False, "timestamp_kind": "model_prediction_only"})
            row["generation_status"] = "failed"
            row["forward_attempts"] = 1
            counters["actual_forward_attempts"] += 1
            native_result = forward(row, ids)
            counters["actual_forward_returns"] += 1
            row["forward_returned"] = True
            audio, duration = materialize_native_result(native_result)
            measured = write_audio(output, row, audio, duration, ids)
            receipt = {"schema": SCHEMA, "id": row["id"], "status": "COMPLETE",
                "generation_status": "success", "request_sha256": row["request_sha256"],
                "preset_id": row["preset_id"], "mode": row["mode"], "role": "train",
                "phonemes": row["raw_phonemes"], "phoneme_ids": ids,
                "target_start_char": row["target_start_char"], "seed": row["seed"],
                "voice_pack_index": len(row["raw_phonemes"]) - 1,
                "generation_model_sha256": recipe.MODEL["model_sha256"],
                "voice_pack_sha256": recipe.VOICE_ASSETS[row["preset_id"]]["sha256"],
                "forward_attempts": 1, "forward_returned": True, "retry_count": 0, **measured}
            once(folder / "COMPLETED.json", receipt)
            row.update(measured, generation_status="success")
            counters["completed"] += 1
        except BaseException as exc:
            once(folder / "FAILED.json", {"schema": SCHEMA, "id": row["id"],
                "status": "FAILED_NO_RETRY" if row["forward_attempts"] else "NOT_STARTED_NO_RETRY",
                "request_sha256": row["request_sha256"], "error_type": type(exc).__name__,
                "error_code": str(exc)[:160], "forward_attempts": row["forward_attempts"],
                "forward_returned": row["forward_returned"], "retry_count": 0,
                "diagnostic_only": True, "training_allowed": False})
            row["error_type"] = type(exc).__name__
            row["error_code"] = str(exc)[:160]
            if not isinstance(exc, Exception) or str(exc) == "GENERATION_DEADLINE":
                raise


def generate(plan, output_root, *, run, asset_root=None):
    """Called only after the worker's exact event/run/parent execution guard.

    This function has no injectable model/backend and no CLI execution shortcut.
    Fixtures exercise pure helpers and _run_rows with explicitly fake outputs;
    the production path always verifies and constructs the pinned native model.
    """
    frozen = validate_plan(plan, executing=True)
    need(type(run) is dict and run.get("run_attempt") in (1, "1"), "ONE_RUN_ATTEMPT")
    output = Path(output_root).absolute()
    need(output.is_dir() and not any(p.is_symlink() for p in (output, *output.parents)), "ARTIFACT_ROOT")
    source = empty_directory(output / "source")
    once(source / "RECIPE.json", frozen)
    once(source / "EXECUTION_PLAN.json", plan)
    once(source / "STARTED.json", {"schema": SCHEMA, "run": run, "declared_count": 48,
         "maximum_calls": 48, "retry_count": 0, "forward_attempts_at_start": 0,
         "diagnostic_only": True, "training_allowed": False})
    rows = [_unmeasured(r) for r in frozen["rows"]]
    counters = {"actual_forward_attempts": 0, "actual_forward_returns": 0, "completed": 0}
    start = time.monotonic()
    error = None
    private = None
    try:
        candidate_private = Path(asset_root) if asset_root is not None else source / "_private_model"
        need(candidate_private.absolute() not in (output, source) and
             candidate_private.absolute() not in output.parents, "DEDICATED_PRIVATE_MODEL_DIRECTORY")
        private = empty_directory(candidate_private)
        forward, vocab = _load_native(output, private)
        _run_rows(output, rows, forward, vocab, start + plan["generation_deadline_seconds"], counters)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "code": str(exc)[:160]}
    finally:
        # A load failure or interrupted loop retains all 48 denominators. Never
        # claim a forward for rows that did not reach the native call boundary.
        for row in rows:
            folder = within(output, "source/audio/" + row["id"])
            if not (folder / "COMPLETED.json").exists() and not (folder / "FAILED.json").exists():
                once(folder / "FAILED.json", {"schema": SCHEMA, "id": row["id"],
                    "status": "NOT_STARTED_NO_RETRY" if not row["forward_attempts"] else "INTERRUPTED_NO_RETRY",
                    "request_sha256": row["request_sha256"], "forward_attempts": row["forward_attempts"],
                    "forward_returned": row["forward_returned"], "retry_count": 0,
                    "diagnostic_only": True, "training_allowed": False, "outer_error": error})
        if private is not None:
            try:
                shutil.rmtree(private)
            except Exception as exc:
                # Failure to clean an owned cache must not erase the actual
                # attempt ledger. It makes this generation result incomplete.
                error = {"type": type(exc).__name__, "code": "PRIVATE_ASSET_CLEANUP_FAILED"}
        result = {"schema": SCHEMA, "stage": "generate", "run": run,
            "status": "COMPLETE" if counters["completed"] == 48 and error is None else "INCOMPLETE_NO_RETRY",
            "declared_count": 48, **counters, "retry_count": 0, "old_audio_model_calls": 0,
            "elapsed_seconds": time.monotonic() - start, "error": error, "rows": rows,
            "diagnostic_only": True, "training_allowed": False, "auto_admission_allowed": False,
            "runtime_g2p_calls": 0, "kws_forward_calls": 0, "training_calls": 0}
        once(source / "RESULT.json", result)
    return result
