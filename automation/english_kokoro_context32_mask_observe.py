"""One new full-carrier input per fixed observer; diagnostic capture only.

Imports are standard-library only until the public execution guard passes.
The old observed crops, old labels, model weights and old CTC helper are unchanged.
"""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import importlib.metadata
import io
from itertools import groupby
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import wave

HERE = Path(__file__).resolve().parent
SCHEMA = "english-kokoro-context32-observer-v2"
CAMPAIGN_ID = "kokoro-full-context32-native-int32-first-forward-v1"
INPUT_SCHEMA = "english-kokoro-context32-v1/input-manifest"
PROVIDERS = ("phoneme", "phoneme_lv60")
PRESETS = ("af_bella", "af_sarah", "am_michael", "am_puck")
CELLS = ("mero-direct-eh", "mero-direct-i", "vosen-direct-s", "vosen-direct-z")
CTC_HELPER_SHA256 = "716928f7707fc662f62e59a988c0c264123be301abac4c23fe835bc09c052b3a"
RUNTIME_VERSIONS = {"torch": "2.9.1+cpu", "transformers": "4.46.3", "numpy": "1.26.4", "huggingface-hub": "0.36.2"}
# The official 4.46.3 PyPI wheel and release-tag sources are byte-identical.
PROCESSOR_SOURCE_FILES = {
    "models/wav2vec2/feature_extraction_wav2vec2.py": {"bytes": 11559, "sha256": "72a03611b72530ff9a7d595ba1057034d01b98576baeee641af3db1a377c11a6"},
    "feature_extraction_utils.py": {"bytes": 30307, "sha256": "a02883a2bed98b55ad317ffc9a607e92d5087dab04933e8cd64fe4f5e6855262"},
    "feature_extraction_sequence_utils.py": {"bytes": 18307, "sha256": "c44e5fd1c4965a87448c2c0d0ec1b40e5f642f707d28fb3e4ac17861358d8743"},
}
CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
DECODE_CONFIG = {"device": "cpu", "decoder": "ctc-greedy", "target_input": None,
                 "special_tokens_removed": True, "phonemizer_used": False, "weights_only": True}
OBSERVERS = {
    "phoneme": {"recognizer_id": "phoneme", "model_family": "wav2vec2-xlsr",
        "model_id": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        "model_revision": "2c733782da5604684829819a5eb744c193fe9398",
        "model_sha256": "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551",
        "decode_config": DECODE_CONFIG},
    "phoneme_lv60": {"recognizer_id": "phoneme_lv60", "model_family": "wav2vec2-lv60",
        "model_id": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "model_revision": "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
        "model_sha256": "3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e",
        "decode_config": DECODE_CONFIG},
}
COMMON_FILES = {
    "config.json": {"bytes": 1856, "sha256": "4609fb49b7e1d28aecb2840da1926c40bd915bc6f1120a940afacf7159bbfb13"},
    "preprocessor_config.json": {"bytes": 212, "sha256": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"},
    "special_tokens_map.json": {"bytes": 85, "sha256": "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"},
    "vocab.json": {"bytes": 4637, "sha256": "d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0"},
}
OBSERVER_FILES = {
    "phoneme": COMMON_FILES | {
        "pytorch_model.bin": {"bytes": 1263535127, "sha256": OBSERVERS["phoneme"]["model_sha256"]},
        "tokenizer_config.json": {"bytes": 412, "sha256": "d663833dacef7d29f563e23029d448fe41415dbe5e8e6d5a98b598a5258c18d8"}},
    "phoneme_lv60": COMMON_FILES | {
        "pytorch_model.bin": {"bytes": 1263535127, "sha256": OBSERVERS["phoneme_lv60"]["model_sha256"]},
        "tokenizer_config.json": {"bytes": 321, "sha256": "1273ec458332cc55226ec596904bae811420755ace7e5d69462b61c4f480b457"}},
}
FROZEN_INPUT_FIELDS = ("id", "source_id", "preset_id", "mode", "cell_id", "requested_class",
    "pronunciation_hypothesis", "native_full_f32_sha256", "native_samples", "crop_start_native_sample",
    "crop_end_native_sample", "old_pcm_sha256", "old_pcm_samples", "full_to_crop_offset_16k",
    "conv_grid_offset_mod_320", "wav_path", "pcm_path")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def within(root, name):
    need(type(name) is str and "\\" not in name, "SAFE_RELATIVE_PATH")
    rel = PurePosixPath(name)
    need(name == rel.as_posix() and rel.parts and not rel.is_absolute()
         and all(p not in ("", ".", "..") for p in rel.parts), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK_PATH")
    return path


def regular(path, limit=8 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and 0 < path.stat().st_size <= limit, "REGULAR_BOUNDED_FILE")
    return path.read_bytes()


def once(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return {"bytes": len(raw), "sha256": sha(raw)}


def put(path, raw):
    need(type(raw) is bytes and 0 < len(raw) <= 32 * 1024**2, "BOUNDED_BINARY_CAPTURE")
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return {"path": Path(path).name, "bytes": len(raw), "sha256": sha(raw)}


def observer_identity(provider):
    need(provider in PROVIDERS, "EXACT_TWO_PROVIDERS")
    item = OBSERVERS[provider]
    return {k: item[k] for k in ("recognizer_id", "model_family", "model_sha256")} | {"decode_config_sha256": sha(canonical(DECODE_CONFIG))}


def verify_processor_sources(package_root):
    measured = {}
    for name, expected in PROCESSOR_SOURCE_FILES.items():
        raw = regular(within(package_root, name))
        measured[name] = {"bytes": len(raw), "sha256": sha(raw)}
        need(measured[name] == expected, "EXACT_NATIVE_PROCESSOR_SOURCE: " + name)
    return measured


def prepare_native_pcm(np, torch, extractor, pcm, row):
    """Production preparation, also exercised by the public processor-only check.

    The extractor's batch is returned unchanged. Only saved CPU arrays receive
    explicit little-endian storage dtypes; the native mask remains torch.int32.
    """
    values = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    need(values.shape == (row["pcm_samples"],) and bool(np.isfinite(values).all()), "PCM_FLOAT_GEOMETRY")
    batch = extractor(values, sampling_rate=16000, return_tensors="pt")
    need(set(batch) == {"input_values", "attention_mask"}, "PROCESSOR_EXACT_FIELDS")
    normalized, mask = batch["input_values"], batch["attention_mask"]
    need(normalized.dtype == torch.float32 and normalized.device.type == "cpu"
         and list(normalized.shape) == [1, row["pcm_samples"]] and bool(torch.isfinite(normalized).all()), "ACTUAL_NORMALIZED_INPUT")
    need(mask.dtype == torch.int32 and mask.device.type == "cpu" and list(mask.shape) == list(normalized.shape)
         and bool((mask == 1).all()), "ACTUAL_UNPADDED_ATTENTION_MASK")
    norm_array = normalized.detach().cpu().numpy()
    mask_array = mask.detach().cpu().numpy()
    facts = {"sample_rate": 16000, "pcm_samples": row["pcm_samples"], "pcm_divisor": 32768.0,
        "do_normalize": True, "normalization_scope": "Entire new full-carrier input, independently of the old crop.",
        "pcm_float_mean": float(values.mean(dtype=np.float64)), "pcm_float_std_population": float(values.std(dtype=np.float64)),
        "normalized_mean": float(norm_array.mean(dtype=np.float64)), "normalized_std_population": float(norm_array.std(dtype=np.float64)),
        "normalized_shape": list(normalized.shape), "normalized_dtype": str(normalized.dtype),
        "attention_mask_shape": list(mask.shape), "attention_mask_dtype": str(mask.dtype),
        "attention_mask_sum": int(mask.sum().item()), "attention_mask_unique_values": [1],
        "valid_input_samples": row["pcm_samples"], "padded_input_samples": row["pcm_samples"], "right_padding_samples": 0}
    return batch, facts, norm_array.astype("<f4", copy=False).tobytes(order="C"), mask_array.astype("<i4", copy=False).tobytes(order="C")


def ctc_function(raw):
    """Load only the immutable pure function, never an old execution entry point."""
    need(sha(raw) == CTC_HELPER_SHA256, "FROZEN_CTC_HELPER_BYTES")
    tree = ast.parse(raw.decode())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ctc_tokens")
    namespace = {"groupby": groupby, "need": need}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "immutable_ctc_tokens", "exec"), namespace)
    return namespace["ctc_tokens"], sha(ast.get_source_segment(raw.decode(), function).encode())


def frame_geometry(samples, crop_offset):
    need(type(samples) is int and 400 <= samples <= 192000 and type(crop_offset) is int
         and 0 <= crop_offset < samples, "FRAME_INPUT_GEOMETRY")
    length, stride, support = samples, 1, 1
    for kernel, step in zip(CONV_KERNEL, CONV_STRIDE):
        length = (length - kernel) // step + 1
        support += (kernel - 1) * stride
        stride *= step
    need(0 < length <= 2000 and (support, stride) == (400, 320), "FIXED_CONV_GEOMETRY")
    return {"frames": length, "conv_kernel": CONV_KERNEL, "conv_stride": CONV_STRIDE,
        "frame_stride_samples": stride, "nominal_conv_support_samples": support,
        "nominal_support_interval": "[320*t,320*t+400)",
        "support_scope": "Convolution input support only; each final transformer logit also uses global input context.",
        "valid_input_samples": samples, "padded_input_samples": samples, "right_padding_samples": 0,
        "full_to_crop_offset_16k": crop_offset, "crop_offset_mod_stride": crop_offset % stride,
        "old_crop_frame_index_equivalence_claimed": False,
        "duration_boundary_is_acoustic_gold": False}


def public_guard(plan, provider, run_identity, env=None):
    env = os.environ if env is None else env
    need(provider in PROVIDERS and plan.get("enabled") is True, "ENABLED_PUBLIC_OBSERVER_ONLY")
    need(plan.get("campaign_id") == CAMPAIGN_ID and type(plan.get("main_before")) is str
         and re.fullmatch(r"[0-9a-f]{40}", plan["main_before"]), "FROZEN_CONTEXT_CAMPAIGN")
    need(plan.get("observers") == OBSERVERS and plan.get("observer_files") == OBSERVER_FILES,
         "EXACT_MODEL_PROCESSOR_DECODE_PINS")
    need(plan.get("maximum_calls_per_observer") == 16 and plan.get("maximum_total_model_calls") == 32
         and plan.get("retry_count") == 0 and plan.get("diagnostic_only") is True,
         "FIXED_DIAGNOSTIC_BUDGET")
    need(plan.get("observer_code_sha256") == sha(regular(Path(__file__))), "OBSERVER_SOURCE_PIN")
    actual = {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"), "event": env.get("GITHUB_EVENT_NAME")}
    need(run_identity == actual and actual["repository"] == "jiying2007/kws-data"
         and actual["event"] == "push" and actual["run_attempt"] == "1"
         and env.get("GITHUB_REF") == "refs/heads/main"
         and type(actual["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", actual["run_id"])
         and type(actual["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", actual["commit"]),
         "FIRST_PUBLIC_MAIN_PUSH_ONLY")
    event = json.loads(regular(Path(env["GITHUB_EVENT_PATH"])))
    need(event.get("before") == plan.get("main_before") and event.get("after") == actual["commit"]
         and event.get("after") != event.get("before")
         and event.get("repository", {}).get("full_name") == actual["repository"]
         and event.get("repository", {}).get("private") is False, "EXACT_FIRST_SUCCESSOR_BINDING")


def validate_inputs(source_root, input_rows, plan):
    need(type(input_rows) is list and len(input_rows) == 16 and type(plan.get("input_roster")) is list
         and len(plan["input_roster"]) == 16, "EXACT_SIXTEEN_FULL_INPUTS")
    manifest = json.loads(regular(Path(source_root) / "INPUT_MANIFEST.json"))
    need(manifest.get("schema") == INPUT_SCHEMA and manifest.get("rows") == input_rows, "VERIFIED_INPUT_MANIFEST_ROWS")
    old = plan.get("old_observed_pcm_sha256")
    need(type(old) is list and len(old) == len(set(old)) == 48 and
         all(type(x) is str and re.fullmatch(r"[0-9a-f]{64}", x) for x in old), "EXACT_OLD_OBSERVED_DENOMINATOR")
    ids, pcms, pairs, checked = set(), set(), set(), []
    for row, fixed in zip(input_rows, plan["input_roster"]):
        need(all(row.get(k) == fixed.get(k) for k in FROZEN_INPUT_FIELDS)
             and all(k in row and k in fixed for k in FROZEN_INPUT_FIELDS), "FROZEN_NATIVE_SOURCE_ROSTER")
        need(type(row["id"]) is str and re.fullmatch(r"enkctx32-[0-9a-f]{24}", row["id"])
             and type(row["source_id"]) is str and re.fullmatch(r"enk48-[0-9a-f]{24}", row["source_id"])
             and row["id"] not in ids and row["mode"] == "full-carrier"
             and row["preset_id"] in PRESETS and row["cell_id"] in CELLS,
             "UNIQUE_ALL_CARRIER_IDENTITY")
        ids.add(row["id"])
        pairs.add((row["preset_id"], row["cell_id"]))
        expected_class = 2 if row["cell_id"].startswith("mero-") else 1
        need(row["requested_class"] == expected_class and row.get("role", "train") == "train", "TRAIN_SIDE_DIAGNOSTIC_IDENTITY")
        native, start, end, offset = (row[k] for k in ("native_samples", "crop_start_native_sample", "crop_end_native_sample", "full_to_crop_offset_16k"))
        need(all(type(x) is int for x in (native, start, end, offset)) and 0 < start < end == native
             and start * 2 == offset * 3 and row["conv_grid_offset_mod_320"] == offset % 320,
             "EXACT_NATIVE_FULL_CROP_COORDINATES")
        samples = row.get("pcm_samples")
        need(type(samples) is int and samples == (native * 2 + 2) // 3 and
             (row.get("sample_rate"), row.get("channels"), row.get("sample_width")) == (16000, 1, 2), "FULL_PCM_GEOMETRY")
        frame_geometry(samples, offset)
        raw = regular(within(source_root, row["pcm_path"]))
        wav = regular(within(source_root, row["wav_path"]))
        need(len(raw) == samples * 2 and sha(raw) == row.get("pcm_sha256") and sha(wav) == row.get("wav_sha256"), "ACTUAL_INPUT_PCM_WAV_BYTES")
        with wave.open(io.BytesIO(wav), "rb") as stream:
            need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes(), stream.getcomptype())
                 == (1, 2, 16000, samples, "NONE") and stream.readframes(samples) == raw, "EXACT_WAV_PCM_JOIN")
        need(row["old_pcm_sha256"] in old and row["pcm_sha256"] not in old and row["pcm_sha256"] not in pcms,
             "NEW_UNOBSERVED_FULL_PCM_REQUIRED")
        pcms.add(row["pcm_sha256"])
        checked.append((row, raw))
    need(pairs == {(p, c) for p in PRESETS for c in CELLS}, "ALL_FOUR_BY_FOUR_CARRIERS_NO_SELECTION")
    manifest_sha = sha(regular(Path(source_root) / "INPUT_MANIFEST.json"))
    need(plan.get("input_manifest_sha256") == manifest_sha, "EXACT_PRIOR_INPUT_MANIFEST_BYTES")
    return checked, manifest_sha


def _load_native(provider, asset_root, output_root):
    versions = {name: importlib.metadata.version(name) for name in RUNTIME_VERSIONS}
    need(versions == RUNTIME_VERSIONS, "EXACT_OBSERVER_RUNTIME")
    from huggingface_hub import snapshot_download
    model_info = OBSERVERS[provider]
    folder = Path(asset_root)
    need(folder.is_absolute() and not folder.exists() and folder.parent.is_dir()
         and not any(p.is_symlink() for p in folder.parents), "NEW_PRIVATE_MODEL_DIRECTORY")
    folder.mkdir(mode=0o700)
    snapshot_download(repo_id=model_info["model_id"], revision=model_info["model_revision"],
        allow_patterns=list(OBSERVER_FILES[provider]), local_dir=str(folder), max_workers=2, token=False)
    files = {}
    for name, expected in OBSERVER_FILES[provider].items():
        path = within(folder, name)
        need(path.is_file() and path.stat().st_size == expected["bytes"], "EXACT_ASSET_SIZE: " + name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        files[name] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
        need(files[name] == expected, "EXACT_ASSET_SHA: " + name)
    small = Path(output_root) / "model-metadata"
    small.mkdir(mode=0o700)
    for name in OBSERVER_FILES[provider]:
        if name != "pytorch_model.bin":
            put(small / name, regular(folder / name))
    vocab = json.loads(regular(folder / "vocab.json"))
    config_json = json.loads(regular(folder / "config.json"))
    preprocessor = json.loads(regular(folder / "preprocessor_config.json"))
    need(len(vocab) == 392 and set(vocab.values()) == set(range(392)) and vocab.get("iː") == 30
         and vocab.get("z") == 21 and config_json.get("pad_token_id") == 0
         and config_json.get("conv_kernel") == CONV_KERNEL and config_json.get("conv_stride") == CONV_STRIDE,
         "FIXED_NATIVE_CTC_AND_CONV_CONFIG")
    ctc, function_sha = ctc_function(regular(HERE / "english_final_ipa_xlsr_once.py"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    import transformers
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    processor_sources = verify_processor_sources(Path(transformers.__file__).resolve().parent)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    need(extractor.sampling_rate == 16000 and extractor.do_normalize is True
         and extractor.return_attention_mask is True, "UNCHANGED_FEATURE_EXTRACTOR")
    config = Wav2Vec2Config.from_json_file(str(folder / "config.json"))
    model = Wav2Vec2ForCTC(config)
    state = torch.load(str(folder / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    need(isinstance(state, dict) and all(type(k) is str and isinstance(v, torch.Tensor) for k, v in state.items()), "TENSOR_STATE_DICT_ONLY")
    model.load_state_dict(state, strict=True)
    del state
    model.to("cpu").eval()

    def prepare(pcm, row):
        return prepare_native_pcm(np, torch, extractor, pcm, row)

    def native_forward(batch):
        with torch.inference_mode():
            return model(**batch)

    def materialize(native_output, row):
        logits = native_output.logits
        need(logits.dtype == torch.float32 and logits.device.type == "cpu" and logits.ndim == 3
             and list(logits.shape) == [1, frame_geometry(row["pcm_samples"], row["full_to_crop_offset_16k"])["frames"], 392]
             and bool(torch.isfinite(logits).all()), "ACTUAL_FULL_LOGITS_GEOMETRY")
        ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        tokens = ctc(ids, vocab, {0, 1, 2, 3})
        raw = logits.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
        return {"shape": list(logits.shape), "dtype": "float32-little-endian", "order": "C",
                "frame_ids": ids, "phone_tokens": tokens,
                "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}, raw

    lock = {"schema": SCHEMA, "model": model_info, "files": files, "runtime_versions": versions,
        "native_processor_source_files": processor_sources,
        "preprocessor_config": preprocessor, "ctc_helper_sha256": CTC_HELPER_SHA256, "ctc_function_sha256": function_sha,
        "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
        "weights_only": True, "strict_state_dict_loading": True, "eval": True, "model_parameters_modified": False,
        "trust_remote_code": False, "phonemizer_used": False, "target_conditioned": False,
        "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0}
    return prepare, native_forward, materialize, lock


def initial_rows(checked):
    return [{"id": row["id"], "source_id": row["source_id"], "status": "NOT_STARTED_NO_RETRY",
             "input_started": False, "forward_attempts": 0, "forward_returned": False,
             "capture_saved": False, "retry_count": 0} for row, _ in checked]


def _execute_rows(checked, output_root, provider, run_identity, native, ledger):
    """The only row loop. Test fixtures may supply fake functions; public run cannot."""
    prepare, forward, materialize = native
    for (row, pcm), progress in zip(checked, ledger):
        directory = Path(output_root) / row["id"]
        directory.mkdir(mode=0o700)
        base = {"schema": SCHEMA, "run": run_identity, "id": row["id"], "source_id": row["source_id"],
            "provider": provider, "observer_identity": observer_identity(provider), "input_pcm_sha256": row["pcm_sha256"],
            "input_wav_sha256": row["wav_sha256"], "input_samples": row["pcm_samples"],
            "diagnostic_only": True, "target_conditioned": False, "training_allowed": False, "retry_count": 0,
            "observer_code_sha256": sha(regular(Path(__file__))),
            "native_processor_source_files": PROCESSOR_SOURCE_FILES}
        try:
            once(directory / "STARTED.json", base | {"forward_attempts": 0, "forward_returned": False})
            progress["input_started"] = True
            progress["status"] = "STARTED"
            batch, facts, normalized_raw, mask_raw = prepare(pcm, row)
            normalized = put(directory / "normalized-input.f32", normalized_raw)
            mask = put(directory / "attention-mask.i32", mask_raw) | {"dtype": "int32-little-endian", "order": "C", "shape": [1, row["pcm_samples"]]}
            need(normalized["bytes"] == row["pcm_samples"] * 4 and mask["bytes"] == row["pcm_samples"] * 4,
                 "SAVED_PROCESSOR_INPUT_SIZE")
            once(directory / "INPUT.json", base | {"processor": facts, "normalized_input": normalized, "attention_mask": mask})
            once(directory / "FORWARD_STARTED.json", base | {"forward_attempts": 1, "forward_returned": False})
            progress["forward_attempts"] = 1
            logits = forward(batch)
            progress["forward_returned"] = True
            once(directory / "FORWARD_RETURNED.json", base | {"forward_attempts": 1, "forward_returned": True})
            measurements, logits_raw = materialize(logits, row)
            geometry = frame_geometry(row["pcm_samples"], row["full_to_crop_offset_16k"])
            need(measurements["shape"] == [1, geometry["frames"], 392]
                 and len(measurements["frame_ids"]) == geometry["frames"]
                 and all(type(i) is int and 0 <= i < 392 for i in measurements["frame_ids"])
                 and len(logits_raw) == geometry["frames"] * 392 * 4, "COMPLETE_FRAME_CAPTURE")
            logits_file = put(directory / "logits.f32", logits_raw)
            ids_file = once(directory / "frame-ids.json", {"schema": SCHEMA, "id": row["id"], "frame_ids": measurements["frame_ids"]})
            capture = base | {"status": "success", "forward_attempts": 1, "forward_returned": True,
                "input_coverage_complete": True, "decoder_output_complete": True,
                "processor": facts, "normalized_input": normalized, "attention_mask": mask,
                "logits": logits_file | {k: measurements[k] for k in ("shape", "dtype", "order")},
                "frame_ids": {"path": "frame-ids.json"} | ids_file,
                "phone_tokens": measurements["phone_tokens"], "argmax_special_frame_counts": measurements["argmax_special_frame_counts"],
                "frame_geometry": geometry, "ctc_helper_sha256": CTC_HELPER_SHA256,
                "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
                "old_crop_labels_modified": False, "whole_target_pronunciation_gold": False}
            capture_pin = once(directory / "CAPTURE.json", capture)
            progress.update(status="success", capture_saved=True, capture_path=row["id"] + "/CAPTURE.json", capture_sha256=capture_pin["sha256"])
        except BaseException as exc:
            progress.update(status="failed", failure=type(exc).__name__ + ": " + str(exc)[:2000])
            try:
                once(directory / "FAILED.json", base | progress)
            except Exception as write_error:
                progress["failure_receipt_write_error"] = type(write_error).__name__ + ": " + str(write_error)[:500]
            if not isinstance(exc, Exception):
                raise


def finish(output_root, provider, run_identity, ledger, failure, manifest_sha):
    attempts = sum(r["forward_attempts"] for r in ledger)
    returned = sum(r["forward_returned"] for r in ledger)
    completed = sum(r["capture_saved"] for r in ledger)
    need(len(ledger) == 16 and 0 <= completed <= returned <= attempts <= 16
         and all(r["forward_attempts"] in (0, 1) and (not r["forward_returned"] or r["forward_attempts"] == 1) for r in ledger), "ACTUAL_COUNTER_INVARIANTS")
    good = failure is None and completed == returned == attempts == 16 and all(r["status"] == "success" for r in ledger)
    result = {"schema": SCHEMA, "provider": provider, "run": run_identity,
        "status": "COMPLETE_DIAGNOSTIC" if good else "INCOMPLETE_DIAGNOSTIC_NO_RETRY",
        "declared_count": 16, "input_started": sum(r["input_started"] for r in ledger),
        "actual_forward_attempts": attempts, "native_forward_returns": returned, "completed": completed,
        "failed": sum(r["status"] == "failed" for r in ledger), "not_started": sum(r["status"] == "NOT_STARTED_NO_RETRY" for r in ledger),
        "retry_count": 0, "new_tts_calls": 0, "new_text_asr_calls": 0, "old_cropped_input_forwards": 0,
        "training_allowed": False, "old_labels_modified": False, "diagnostic_only": True,
        "input_manifest_sha256": manifest_sha, "failure": failure, "rows": ledger}
    once(Path(output_root) / "RESULT.json", result)
    return result


def run(*, provider, source_root, output_root, asset_root, plan, run_identity, input_rows):
    public_guard(plan, provider, run_identity)
    checked, manifest_sha = validate_inputs(source_root, input_rows, plan)
    output = Path(output_root)
    need(output.is_absolute() and not output.exists() and output.parent.is_dir()
         and not any(p.is_symlink() for p in output.parents), "CREATE_ONLY_PROVIDER_OUTPUT")
    output.mkdir(mode=0o700)
    ledger = initial_rows(checked)
    once(output / "STARTED.json", {"schema": SCHEMA, "provider": provider, "run": run_identity,
        "input_manifest_sha256": manifest_sha, "maximum_calls": 16, "retry_count": 0,
        "observer_identity": observer_identity(provider), "diagnostic_only": True})
    failure = None
    try:
        once(output / "MODEL_LOAD_STARTED.json", {"schema": SCHEMA, "provider": provider, "actual_forward_attempts": 0})
        prepare, forward, materialize, lock = _load_native(provider, asset_root, output)
        once(output / "MODEL_LOCK.json", lock | {"observer_identity": observer_identity(provider),
            "observer_code_sha256": sha(regular(Path(__file__))), "input_manifest_sha256": manifest_sha})
        _execute_rows(checked, output, provider, run_identity, (prepare, forward, materialize), ledger)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    return finish(output, provider, run_identity, ledger, failure, manifest_sha)


def source_preflight():
    ctc, function_sha = ctc_function(regular(HERE / "english_final_ipa_xlsr_once.py"))
    need(ctc([4, 4, 0, 4], {"<pad>": 0, "x": 4}, {0}) == ["x", "x"], "CTC_COLLAPSE_BEFORE_BLANK")
    return {"schema": SCHEMA, "providers": list(PROVIDERS), "declared_calls_per_provider": 16,
        "maximum_total_model_calls": 32, "ctc_helper_sha256": CTC_HELPER_SHA256,
        "ctc_function_sha256": function_sha, "native_processor_source_files": PROCESSOR_SOURCE_FILES,
        "native_mask_dtype": "torch.int32", "new_model_calls": 0, "new_network_calls": 0}


if __name__ == "__main__":
    need(sys.argv[1:] == ["--source-check"], "NO_DIRECT_NUMERICAL_CLI")
    print(json.dumps(source_preflight(), ensure_ascii=False, sort_keys=True))
