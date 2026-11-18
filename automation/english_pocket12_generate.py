"""Twelve fixed Pocket short requests; imports remain stdlib until initialization.

No downloads, warmup, audio prompting, parent crop, retry or training admission.
The public once worker owns the event guard and runtime/asset installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import struct
import sys
import time
import wave
import zipfile

HERE = Path(__file__).resolve().parent
RECIPE_PATH = HERE / "english_pocket12_recipe.json"
RECIPE_SHA256 = "229e87f164d8fc366e9fdc2c2c085e90881a68d61858e77e357947969a1cfc49"
SCHEMA = "english-pocket12-source-v1"
FLAGS = {"diagnostic_only": True, "training_allowed": False}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(canonical(value))
        temporary.replace(path)
    else:
        with path.open("xb") as stream:
            stream.write(canonical(value))


def within(root, name):
    need(type(name) is str and "\\" not in name, "RELATIVE_ARTIFACT_PATH")
    rel = PurePosixPath(name)
    need(not rel.is_absolute() and rel.parts and name == rel.as_posix()
         and all(p not in ("", ".", "..") for p in rel.parts), "RELATIVE_ARTIFACT_PATH")
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK_ARTIFACT")
    return path


def recipe():
    raw = RECIPE_PATH.read_bytes()
    need(sha(raw) == RECIPE_SHA256, "EXACT_FIXED_RECIPE")
    return json.loads(raw)


def source_preflight(plan=None):
    r = recipe()
    need(len(r["rows"]) == 12 and len({x["id"] for x in r["rows"]}) == 12,
         "TWELVE_FIXED_REQUESTS")
    if plan is not None:
        need(plan.get("training_allowed") is False, "NO_TRAINING")
        for key, expected in (("campaign_id", r["campaign_id"]),
                              ("recipe_sha256", RECIPE_SHA256),
                              ("source_code_sha256", digest(Path(__file__)))):
            if key in plan:
                need(plan[key] == expected, "PLAN_SOURCE_PIN: " + key)
        for key in ("native_tts_request_budget", "retry_count"):
            if key in plan:
                need(plan[key] == r[key], "PLAN_BUDGET: " + key)
    return {"source_check": "PASS", "recipe_sha256": RECIPE_SHA256,
            "new_model_calls": 0, "new_network_calls": 0}


def verify_assets(asset_root):
    records = []
    for pin in recipe()["assets"]:
        path = within(asset_root, pin["path"])
        need(path.is_file() and path.stat().st_size == pin["bytes"]
             and digest(path) == pin["sha256"], "ASSET_BYTES: " + pin["path"])
        records.append(dict(pin))
    return records


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        need(len(prefix) == 8, "SAFETENSORS_LENGTH")
        size = struct.unpack("<Q", prefix)[0]
        need(0 < size <= 16 * 1024**2, "SAFETENSORS_HEADER_BOUND")
        raw = stream.read(size)
    need(len(raw) == size, "SAFETENSORS_HEADER_COMPLETE")
    header = json.loads(raw)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    need(bool(tensors), "SAFETENSORS_NOT_EMPTY")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
              "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
              "I64": 8, "U64": 8, "F64": 8}
    cursor = 0
    for key, item in sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"]):
        need(type(key) is str and item["dtype"] in widths, "TENSOR_DTYPE")
        count = 1
        for dim in item["shape"]:
            need(type(dim) is int and dim >= 0, "TENSOR_SHAPE")
            count *= dim
        start, end = item["data_offsets"]
        need(type(start) is int and type(end) is int and start == cursor
             and end - start == count * widths[item["dtype"]], "TENSOR_STORAGE")
        cursor = end
    need(8 + size + cursor == Path(path).stat().st_size, "EXACT_TENSOR_STORAGE")
    return tensors


def installed_wheel_receipt(asset_root):
    r = recipe()
    versions = {name: importlib.metadata.version(name) for name in r["runtime_versions"]}
    need(versions == r["runtime_versions"], "EXACT_POCKET_RUNTIME_VERSIONS")
    distribution = importlib.metadata.distribution("pocket-tts")
    receipts = []
    with zipfile.ZipFile(Path(asset_root) / r["assets"][0]["path"]) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("pocket_tts/") and name.endswith((".py", ".yaml")):
                expected = archive.read(name)
                installed = Path(distribution.locate_file(name))
                need(installed.is_file() and installed.read_bytes() == expected,
                     "INSTALLED_WHEEL_SOURCE: " + name)
                receipts.append({"path": name, "bytes": len(expected), "sha256": sha(expected)})
    return {"required_versions": versions, "installed_source_files": receipts,
            "all_distributions": sorted({(d.metadata["Name"], d.version)
                                         for d in importlib.metadata.distributions()})}


class NativePocket:
    """Only instantiated by generate after the guarded public worker calls it."""
    def __init__(self, asset_root, output, lock):
        self.output, self.assets, self.lock = Path(output), Path(asset_root).resolve(), lock
        self.r = recipe()
        lock["python_version"] = sys.version
        lock["verified_assets"] = verify_assets(self.assets)
        lock["runtime"] = installed_wheel_receipt(self.assets)
        import numpy as np
        import scipy.signal
        import torch
        import yaml
        from pocket_tts import TTSModel
        from pocket_tts.modules.stateful_module import StatefulModule
        self.np, self.signal, self.torch = np, scipy.signal, torch
        need(torch.version.cuda is None, "CPU_TORCH_BUILD_ONLY")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.forward_counts = Counter()
        self.phase = "initialization"
        self.active = None
        lock["native_forward_calls_during_initialization"] = 0
        def count_forward(module, inputs):
            self.forward_counts.update([self.phase + ":" + type(module).__module__ + "." + type(module).__name__])
            if self.phase == "initialization":
                lock["native_forward_calls_during_initialization"] += 1
        self.hook = torch.nn.modules.module.register_module_forward_pre_hook(count_forward)
        config_raw = Path(importlib.metadata.distribution("pocket-tts").locate_file(
            "pocket_tts/config/english.yaml")).read_bytes()
        need(sha(config_raw) == self.r["config_sha256"], "EXACT_DEFAULT_ENGLISH_CONFIG")
        config = yaml.safe_load(config_raw)
        config["weights_path"] = str(self.assets / "model.safetensors")
        config["weights_path_without_voice_cloning"] = None
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(self.assets / "tokenizer.json")
        need(config["flow_lm"].get("weights_path") is None
             and config["mimi"].get("weights_path") is None, "ONE_COMBINED_CHECKPOINT")
        local_config = self.output / "LOCAL_CONFIG.yaml"
        with local_config.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=True)
        lock["local_config_path"] = "LOCAL_CONFIG.yaml"
        lock["local_config_sha256"] = digest(local_config)
        header = safetensors_header(self.assets / "model.safetensors")
        write_json(self.output / "CHECKPOINT_TENSORS.json", header)
        lock["checkpoint_header_path"] = "CHECKPOINT_TENSORS.json"
        dtype_names = {"torch.float32": "F32", "torch.bfloat16": "BF16", "torch.float16": "F16",
                       "torch.float64": "F64", "torch.int64": "I64", "torch.int32": "I32",
                       "torch.int16": "I16", "torch.int8": "I8", "torch.uint8": "U8", "torch.bool": "BOOL"}
        original_load = TTSModel.load_state_dict
        own_load = "load_state_dict" in TTSModel.__dict__
        strict_calls = []

        def checked_load(model, state_dict, strict=True, assign=False):
            need(not strict_calls and strict is True and assign is False, "ONE_NATIVE_STRICT_LOAD")
            need(set(state_dict) == set(header), "CHECKPOINT_KEYS_EXACT")
            expected = model.state_dict()
            need(set(expected) == set(state_dict), "ARCHITECTURE_KEYS_EXACT")
            for name, tensor in state_dict.items():
                need(list(tensor.shape) == header[name]["shape"] == list(expected[name].shape)
                     and dtype_names.get(str(tensor.dtype)) == header[name]["dtype"]
                     and tensor.device.type == "cpu", "CHECKPOINT_TENSOR: " + name)
                if tensor.is_floating_point():
                    need(bool(torch.isfinite(tensor).all()), "CHECKPOINT_FINITE: " + name)
            result = original_load(model, state_dict, strict=strict, assign=assign)
            need(not result.missing_keys and not result.unexpected_keys, "STRICT_LOAD_RESULT")
            for name, tensor in model.state_dict().items():
                need(torch.equal(tensor, state_dict[name].to(dtype=tensor.dtype, device="cpu")),
                     "LOADED_TENSOR_EQUALS_CHECKPOINT: " + name)
            strict_calls.append({"strict": True, "assign": False, "tensor_count": len(header),
                                 "missing_keys": [], "unexpected_keys": [],
                                 "keys_shapes_storage_dtypes_finite_and_loaded_values_verified": True})
            return result

        TTSModel.load_state_dict = checked_load
        lock["native_model_construction_attempts"] = 1
        self.persist_lock()
        try:
            self.model = TTSModel.load_model(config=local_config, language=None,
                                            checkpoint=None, quantize=False)
        finally:
            if own_load:
                TTSModel.load_state_dict = original_load
            else:
                delattr(TTSModel, "load_state_dict")
        lock["native_model_constructions_returned"] = 1
        need(len(strict_calls) == 1, "NATIVE_STRICT_LOAD_OBSERVED")
        need(self.model.device.type == "cpu" and self.model.sample_rate == 24000,
             "NATIVE_CPU_MONO_24K_MODEL")
        self.model.eval()
        self.parameters = {key: getattr(self.model, key) for key in self.r["native_parameters"]}
        need(self.parameters == self.r["native_parameters"], "UNCHANGED_NATIVE_DEFAULT_PARAMETERS")
        lock["native_parameters"] = self.parameters
        lock["strict_checkpoint_loads"] = strict_calls
        modules = {name: m for name, m in self.model.flow_lm.named_modules()
                   if isinstance(m, StatefulModule)}
        need(len(modules) == 6, "SIX_FLOW_LM_STATES")
        self.states, self.state_hashes, voice_receipts = {}, {}, []
        for voice in self.r["voices"]:
            path = self.assets / voice["state_path"]
            state_header = safetensors_header(path)
            state = self.model.get_state_for_audio_prompt(path)
            need(set(state) == set(modules), "VOICE_STATE_MODULE_KEYS")
            raw_modules = {name.split("/")[0] for name in state_header}
            need(raw_modules == set(modules), "VOICE_HEADER_MODULE_KEYS")
            entries, offsets = [], []
            for name, values in state.items():
                need(set(values) in ({"cache", "offset"}, {"cache", "offset", "pad"}),
                     "VOICE_STATE_TENSOR_KEYS")
                cache, offset = values["cache"], values["offset"]
                need(list(cache.shape[:2]) == [2, 1] and list(cache.shape[3:]) == [16, 64]
                     and cache.ndim == 5 and cache.dtype == self.model.flow_lm.dtype
                     and cache.device.type == "cpu", "VOICE_CACHE_GEOMETRY")
                need(list(offset.shape) == [1] and offset.dtype == torch.int64, "VOICE_OFFSET_GEOMETRY")
                populated = int(offset.item())
                need(0 < populated <= cache.shape[2], "VOICE_POPULATED_PREFIX")
                need(bool(torch.isfinite(cache[:, :, :populated]).all()), "VOICE_ACTIVE_PREFIX_FINITE")
                if "pad" in values:
                    pad = values["pad"]
                    need(list(pad.shape) == [1] and pad.dtype == torch.int64 and int(pad.item()) == 0,
                         "VOICE_NO_BATCH_PADDING")
                module = modules[name]
                need(module.num_kv_heads == 16 and module.dim_per_head == 64, "VOICE_MODEL_HEAD_MATCH")
                offsets.append(populated)
                entries.append({"module": name, "populated": populated,
                                "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                            for k, v in values.items()},
                                "active_prefix_finite": True})
            need(len(set(offsets)) == 1, "VOICE_LAYER_OFFSETS_AGREE")
            self.states[voice["voice_id"]] = state
            self.state_hashes[voice["voice_id"]] = self.state_hash(state)
            voice_receipts.append({**voice, "raw_header": state_header, "loaded_state": entries,
                                   "state_sha256": self.state_hashes[voice["voice_id"]]})
        lock["voice_state_receipts"] = voice_receipts
        lock["native_forward_calls_during_initialization"] = sum(self.forward_counts.values())
        need(lock["native_forward_calls_during_initialization"] == 0, "NO_INITIALIZATION_FORWARD_OR_WARMUP")
        original_prepare = self.model.flow_lm.conditioner.prepare

        def captured_prepare(text):
            tokens = original_prepare(text)
            need(self.active is not None, "REQUEST_RECORD_REQUIRED")
            self.active["actual_inputs"].append({"formatted_text": text,
                "token_ids": tokens.detach().cpu().tolist(), "shape": list(tokens.shape),
                "dtype": str(tokens.dtype), "max_native_latent_steps": self.model._estimate_max_gen_len(tokens.shape[1])})
            self.persist_request()
            return tokens

        self.model.flow_lm.conditioner.prepare = captured_prepare
        original_short = self.model._generate_audio_stream_short_text

        def captured_short(*args, **kwargs):
            need(self.active is not None, "REQUEST_RECORD_REQUIRED")
            self.active["actual_short_calls"].append({key: kwargs[key] for key in
                ("text_to_generate", "frames_after_eos", "copy_state")})
            self.persist_request()
            return original_short(*args, **kwargs)

        self.model._generate_audio_stream_short_text = captured_short
        lock.update({"status": "INITIALIZED_VERIFIED", "voice_audio_encoding_calls": 0,
                     "warmup_calls": 0, "quantize": False, "checkpoint_argument": None,
                     "has_voice_cloning_native_flag": self.model.has_voice_cloning,
                     "voice_route": "explicit_pinned_local_precomputed_safetensors_only"})
        self.persist_lock()

    def persist_lock(self):
        write_json(self.output / "MODEL_LOCK.json", self.lock, replace=True)

    def persist_request(self):
        write_json(self.output / ("rows/" + self.active["id"] + "/NATIVE_REQUEST.json"),
                   self.active, replace=True)

    def state_hash(self, state):
        h = hashlib.sha256()
        for name, tensors in sorted(state.items()):
            for key, tensor in sorted(tensors.items()):
                h.update((name + "/" + key).encode())
                h.update(tensor.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def generate(self, row):
        voice = row["voice_id"]
        need(self.state_hash(self.states[voice]) == self.state_hashes[voice], "UNMODIFIED_PRECOMPUTED_STATE")
        self.active = {"id": row["id"], "requested_text": row["requested_text"],
            "requested_text_is_observation": False, "seed": row["seed"],
            "model_id": self.r["model_id"], "model_revision": self.r["model_revision"],
            "native_parameters": self.parameters, "request_parameters": self.r["request_parameters"],
            "actual_inputs": [], "actual_short_calls": [], "module_forward_calls": {},
            "native_tts_attempts": 0, "native_returned": False}
        self.persist_request()
        self.phase = row["id"]
        before = self.forward_counts.copy()
        self.torch.manual_seed(row["seed"])
        try:
            mark_native_started(self.output, row)
            self.active["native_tts_attempts"] = 1
            self.persist_request()
            audio = self.model.generate_audio(self.states[voice], row["requested_text"],
                                             **self.r["request_parameters"])
            self.active["native_returned"] = True
            mark_native_returned(self.output, row)
            return audio
        finally:
            self.active["module_forward_calls"] = dict(self.forward_counts - before)
            self.active["precomputed_state_unchanged"] = self.state_hash(self.states[voice]) == self.state_hashes[voice]
            self.persist_request()
            need(self.active["precomputed_state_unchanged"], "COPY_STATE_PRESERVES_VOICE")

    def materialize(self, audio, row, directory):
        np = self.np
        need(audio.device.type == "cpu" and audio.ndim == 1 and audio.dtype == self.torch.float32,
             "NATIVE_1D_FLOAT32_CPU_AUDIO")
        values = audio.detach().cpu().contiguous().numpy()
        raw = values.astype("<f4", copy=False).tobytes()
        with (directory / "native_24k.f32").open("xb") as stream:
            stream.write(raw)
        native_fields = {"native_f32_path": f"rows/{row['id']}/native_24k.f32",
            "native_f32_sha256": sha(raw), "native_samples": int(values.size), "native_sample_rate": 24000}
        row.update(native_fields)
        need(len(self.active["actual_inputs"]) == len(self.active["actual_short_calls"]) == 1,
             "ONE_NATIVE_SHORT_TEXT_CHUNK")
        need(0 < values.size <= self.r["max_native_samples"] and bool(np.isfinite(values).all()),
             "NATIVE_COMPLETE_FINITE_BOUNDED_AUDIO")
        # Exactly one resample of the entire native waveform; no trimming/padding/normalization.
        resampled = self.signal.resample_poly(values.astype(np.float64), 2, 3,
            window=("kaiser", 5.0), padtype="constant", cval=0.0)
        need(resampled.size == (values.size * 2 + 2) // 3 and bool(np.isfinite(resampled).all()),
             "WHOLE_CLIP_2_OVER_3_GEOMETRY")
        scaled = resampled * 32768.0
        pcm = np.rint(np.clip(scaled, -32768, 32767)).astype("<i2").tobytes()
        with (directory / "audio.pcm").open("xb") as stream:
            stream.write(pcm)
        with wave.open(str(directory / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, len(pcm) // 2, "NONE", "not compressed"))
            stream.writeframes(pcm)
        return {**native_fields, "pcm_path": f"rows/{row['id']}/audio.pcm", "pcm_sha256": sha(pcm),
            "pcm_samples": len(pcm) // 2, "wav_path": f"rows/{row['id']}/audio.wav",
            "wav_sha256": digest(directory / "audio.wav"),
            "quantization_clipped_samples": int(np.count_nonzero((scaled < -32768) | (scaled > 32767))),
            "resample_calls": 1, "quantization_calls": 1,
            "actual_input_record_path": f"rows/{row['id']}/NATIVE_REQUEST.json"}

    def close(self):
        self.hook.remove()


def initial_rows():
    return [{**r, **FLAGS, "generation_status": "not_started", "sample_rate": 16000,
             "channels": 1, "sample_width": 2, "pcm_path": None, "pcm_sha256": None,
             "pcm_samples": None, "wav_path": None, "wav_sha256": None,
             "tts_attempts": 0, "tts_returned": False, "capture_saved": False,
             "request_started": False, "error": None} for r in recipe()["rows"]]


def mark_native_started(output, row):
    need(row["tts_attempts"] == 0 and row["tts_returned"] is False, "ONE_TTS_ATTEMPT")
    row["tts_attempts"] = 1
    write_json(Path(output) / "rows" / row["id"] / "STARTED.json",
               {"id": row["id"], "tts_attempts": 1})


def mark_native_returned(output, row):
    need(row["tts_attempts"] == 1 and row["tts_returned"] is False, "ONE_NATIVE_RETURN")
    row["tts_returned"] = True
    write_json(Path(output) / "rows" / row["id"] / "RETURNED.json",
               {"id": row["id"], "tts_returned": True})


def generate(plan, output, assets, run_identity, *, backend_factory=None):
    """backend_factory exists only for stdlib tests; production worker never supplies it."""
    source_preflight(plan)
    output = Path(output)
    need(not output.exists() or not any(output.iterdir()), "FRESH_OUTPUT_NO_RETRY")
    output.mkdir(parents=True, exist_ok=True)
    rows = initial_rows()
    source = {"schema": SCHEMA, "recipe_sha256": RECIPE_SHA256, "run": run_identity,
              **FLAGS, "requested_text_is_observation": False, "rows": rows}
    lock = {"schema": "english-pocket12-model-lock-v1", "status": "NOT_INITIALIZED",
            "native_model_construction_attempts": 0, "native_model_constructions_returned": 0,
            "native_forward_calls_during_initialization": 0, "model_revision": recipe()["model_revision"]}
    write_json(output / "MODEL_LOCK.json", lock)
    write_json(output / "CHECKPOINT.json", source)
    backend, init_error = None, None
    started = time.monotonic()
    try:
        if backend_factory is None:
            import signal
            def expired(signum, frame):
                raise TimeoutError("POCKET_INITIALIZATION_840_SECONDS")
            need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "NO_EXISTING_TIMER_TO_OVERRIDE")
            previous = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, 840)
            try:
                backend = NativePocket(assets, output, lock)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        else:
            backend = backend_factory(assets, output, lock)
    except Exception as exc:
        init_error = type(exc).__name__ + ": " + str(exc)
        lock.update({"status": "INITIALIZATION_FAILED", "error": init_error})
        write_json(output / "MODEL_LOCK.json", lock, replace=True)
    if backend is not None:
        for row in rows:
            directory = output / "rows" / row["id"]
            directory.mkdir(parents=True)
            write_json(directory / "REQUEST.json", {**row, "recipe_sha256": RECIPE_SHA256})
            row["request_started"] = True
            write_json(directory / "REQUEST_STARTED.json", {"id": row["id"], "request_started": True})
            write_json(output / "CHECKPOINT.json", source, replace=True)
            t0 = time.monotonic()
            try:
                audio = backend.generate(row)
                need(row["tts_attempts"] == 1 and row["tts_returned"] is True, "ACTUAL_NATIVE_RETURN_RECEIPT")
                row.update(backend.materialize(audio, row, directory))
                row.update({"generation_status": "success", "capture_saved": True})
            except Exception as exc:
                row.update({"generation_status": "failed", "error": type(exc).__name__ + ": " + str(exc)})
                write_json(directory / "FAILURE.json", {"id": row["id"], "error": row["error"],
                                                        "tts_returned": row["tts_returned"]})
            row["elapsed_seconds"] = round(time.monotonic() - t0, 6)
            write_json(directory / "RESULT.json", row)
            write_json(output / "CHECKPOINT.json", source, replace=True)
        backend.close()
    counts = Counter(r["generation_status"] for r in rows)
    result = {"schema": "english-pocket12-generation-result-v1", "run": run_identity, **FLAGS,
        "recipe_sha256": RECIPE_SHA256, "denominator": 12,
        "status": "COMPLETE" if counts["success"] == 12 else "INCOMPLETE",
        "tts_attempts": sum(r["tts_attempts"] for r in rows),
        "tts_returned": sum(r["tts_returned"] for r in rows),
        "requests_started": sum(r["request_started"] for r in rows),
        "successful_count": counts["success"], "failed_count": counts["failed"],
        "not_started_count": counts["not_started"], "retry_count": 0, "warmup_count": 0,
        "initialization_error": init_error, "elapsed_seconds": round(time.monotonic() - started, 6)}
    write_json(output / "SOURCE.json", source)
    write_json(output / "RESULT.json", result)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "FILES.json", {"schema": "english-pocket12-files-v1", "files": files,
                                      "self_excluded": "FILES.json"})
    return result


def validate_source(output, expected_run=None):
    root = Path(output)
    inventory = json.loads((root / "FILES.json").read_bytes())
    need(inventory.get("schema") == "english-pocket12-files-v1"
         and inventory.get("self_excluded") == "FILES.json", "FILES_SCHEMA")
    entries = inventory["files"]
    need(type(entries) is list and len({r["path"] for r in entries}) == len(entries), "UNIQUE_FILES_INVENTORY")
    actual_paths = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "NO_SYMLINK_ARTIFACT")
        if path.is_file() and path != root / "FILES.json":
            actual_paths.add(path.relative_to(root).as_posix())
    need(actual_paths == {e["path"] for e in entries}, "CLOSED_FILES_INVENTORY")
    for entry in entries:
        path = within(root, entry["path"])
        need(path.is_file() and path.stat().st_size == entry["bytes"]
             and digest(path) == entry["sha256"], "FILES_BYTES: " + entry["path"])
    source = json.loads((root / "SOURCE.json").read_bytes())
    need(source.get("schema") == SCHEMA and source.get("recipe_sha256") == RECIPE_SHA256
         and source.get("training_allowed") is False and source.get("diagnostic_only") is True
         and source.get("requested_text_is_observation") is False, "SOURCE_FIXED_CONTRACT")
    if expected_run is not None:
        need(source.get("run") == expected_run, "SOURCE_RUN_IDENTITY")
    rows = source.get("rows")
    need(type(rows) is list and len(rows) == 12, "COMPLETE_TWELVE_SOURCE_DENOMINATOR")
    successful = []
    for row, fixed in zip(rows, recipe()["rows"]):
        need(all(type(row.get(k)) is type(v) and row.get(k) == v for k, v in fixed.items()), "FIXED_ROSTER")
        need(row.get("diagnostic_only") is True and row.get("training_allowed") is False,
             "PERMANENT_TRAINING_EXCLUSION")
        need((row.get("sample_rate"), row.get("channels"), row.get("sample_width")) == (16000, 1, 2),
             "SOURCE_PCM_FORMAT")
        need(row.get("generation_status") in ("success", "failed", "not_started")
             and type(row.get("tts_attempts")) is int and row["tts_attempts"] in (0, 1)
             and type(row.get("tts_returned")) is bool, "SOURCE_CALL_LEDGER")
        if row["generation_status"] == "not_started":
            need(row["tts_attempts"] == 0 and row["tts_returned"] is False
                 and row.get("request_started") is False, "UNSTARTED_LEDGER")
        elif row["generation_status"] == "failed":
            need(row.get("request_started") is True and row.get("capture_saved") is False
                 and (row["tts_attempts"] == 1 or row["tts_returned"] is False), "FAILED_LEDGER")
        else:
            need(row["tts_attempts"] == 1 and row["tts_returned"] is True
                 and row.get("capture_saved") is True, "SUCCESS_LEDGER")
            pcm, wav = within(root, row["pcm_path"]).read_bytes(), within(root, row["wav_path"]).read_bytes()
            samples = row["pcm_samples"]
            need(type(samples) is int and 0 < samples <= 480000 and len(pcm) == 2 * samples
                 and sha(pcm) == row["pcm_sha256"] and sha(wav) == row["wav_sha256"], "EXACT_PCM_WAV_BYTES")
            with wave.open(io.BytesIO(wav), "rb") as stream:
                need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                      stream.getnframes(), stream.getcomptype()) == (1, 2, 16000, samples, "NONE")
                     and stream.readframes(samples) == pcm, "WAV_IS_SAME_OBSERVER_PCM")
            native = within(root, row["native_f32_path"]).read_bytes()
            nsamples = row["native_samples"]
            need(type(nsamples) is int and 0 < nsamples <= recipe()["max_native_samples"]
                 and row["native_sample_rate"] == 24000 and len(native) == nsamples * 4
                 and sha(native) == row["native_f32_sha256"]
                 and samples == (nsamples * 2 + 2) // 3
                 and row.get("resample_calls") == row.get("quantization_calls") == 1,
                 "NATIVE_FULL_AUDIO_TO_OBSERVER_GEOMETRY")
            import math
            need(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", native)), "NATIVE_FLOAT_FINITE")
            request = json.loads(within(root, row["actual_input_record_path"]).read_bytes())
            need(request.get("id") == row["id"] and request.get("requested_text") == row["requested_text"]
                 and request.get("requested_text_is_observation") is False
                 and request.get("model_revision") == recipe()["model_revision"]
                 and request.get("native_parameters") == recipe()["native_parameters"]
                 and request.get("request_parameters") == recipe()["request_parameters"]
                 and request.get("native_tts_attempts") == 1 and request.get("native_returned") is True
                 and request.get("precomputed_state_unchanged") is True,
                 "NATIVE_REQUEST_RECEIPT")
            inputs, shorts = request.get("actual_inputs"), request.get("actual_short_calls")
            need(type(inputs) is list and type(shorts) is list and len(inputs) == len(shorts) == 1,
                 "ONE_ACTUAL_NATIVE_INPUT")
            tokens = inputs[0]["token_ids"]
            need(type(tokens) is list and len(tokens) == 1 and type(tokens[0]) is list and bool(tokens[0])
                 and all(type(t) is int and 0 <= t < 4000 for t in tokens[0])
                 and inputs[0]["shape"] == [1, len(tokens[0])]
                 and inputs[0]["dtype"] == "torch.int64"
                 and inputs[0]["formatted_text"] == shorts[0]["text_to_generate"]
                 and shorts[0]["copy_state"] is True and shorts[0]["frames_after_eos"] == 5,
                 "ACTUAL_TOKENS_AND_SHORT_PARAMETERS")
            successful.append(row)
        directory = root / "rows" / row["id"]
        need((directory / "STARTED.json").exists() == bool(row["tts_attempts"])
             and (directory / "RETURNED.json").exists() == row["tts_returned"], "CALL_MARKERS_MATCH_LEDGER")
        if row["tts_attempts"]:
            need(json.loads((directory / "STARTED.json").read_bytes()) ==
                 {"id": row["id"], "tts_attempts": 1}, "EXACT_START_MARKER")
        if row["tts_returned"]:
            need(json.loads((directory / "RETURNED.json").read_bytes()) ==
                 {"id": row["id"], "tts_returned": True}, "EXACT_RETURN_MARKER")
    result = json.loads((root / "RESULT.json").read_bytes())
    counts = Counter(r["generation_status"] for r in rows)
    expected = {"run": source["run"], "denominator": 12,
        "tts_attempts": sum(r["tts_attempts"] for r in rows), "tts_returned": sum(r["tts_returned"] for r in rows),
        "requests_started": sum(r["request_started"] for r in rows),
        "successful_count": counts["success"], "failed_count": counts["failed"],
        "not_started_count": counts["not_started"], "retry_count": 0, "warmup_count": 0,
        "status": "COMPLETE" if len(successful) == 12 else "INCOMPLETE", **FLAGS}
    need(all(type(result.get(k)) is type(v) and result.get(k) == v for k, v in expected.items()),
         "RESULT_MATCHES_COMPLETE_LEDGER")
    return source, successful


def source_readback(output, plan=None, run_identity=None):
    source_preflight(plan)
    source, successful = validate_source(output, run_identity)
    counts = Counter(r["generation_status"] for r in source["rows"])
    return {"rows": source["rows"], "source_ready": len(successful) == 12,
            "successful_count": counts["success"], "failed_count": counts["failed"],
            "not_started_count": counts["not_started"], "new_model_calls": 0,
            "tts_attempts": sum(r["tts_attempts"] for r in source["rows"]),
            "tts_returned": sum(r["tts_returned"] for r in source["rows"])}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source-check", "readback"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(source_preflight() if args.action == "source-check"
                     else source_readback(args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
