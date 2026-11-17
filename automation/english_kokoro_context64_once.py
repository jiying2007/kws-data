"""Finite parent-context / exact-child diagnostic; never training admission.

Source readback, text-capture readback and combine are standard-library only.
The three text observers consume child PCM. The two phone observers consume
parent PCM. Neither evidence scope is silently substituted for the other.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import struct
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
SCHEMA = "english-kokoro-context64-once-v1"
SOURCE_SCHEMA = "english-kokoro-context64-source-v1"
TEXT_SCHEMA = "english-kokoro-context64-child-text-v1"
CAMPAIGN_ID = "kokoro-exact-child-context-controls64-v1"
MAIN_BEFORE = "2ca611210126e2f78939e6e50f7cd0157c76d5dd"
DESIGN_SHA256 = "d0460da69b64aed66aec8526d253ed6b8a4762eebff234a862331207dd5b99f6"
POLICY_SHA256 = "fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da"
EXCLUSION_SHA256 = "f9e0644e7db12b411c131890df90eebad7037a5cb285735079ff0c5e1cb443e0"
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
PHONE_PROVIDERS = ("phoneme", "phoneme_lv60")
PROVIDERS = TEXT_PROVIDERS + PHONE_PROVIDERS
MAX_FILE = 256 * 1024 * 1024
MAX_EXPANDED = 1024 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CODE_PATHS = {
    "worker_sha256": "automation/english_kokoro_context64_once.py",
    "recipe_code_sha256": "automation/english_kokoro_context64_recipe.py",
    "generator_code_sha256": "automation/english_kokoro_context64_generate.py",
    "phone_code_sha256": "automation/english_kokoro_context64_phone.py",
    "evaluate_code_sha256": "automation/english_kokoro_context64_evaluate.py",
    "reference_policy_sha256": "automation/english_context64_policy_v2_reference.py",
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, "DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def regular(path, maximum=MAX_FILE, *, allow_empty=False):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "REGULAR_NO_SYMLINK")
    need((0 if allow_empty else 1) <= path.stat().st_size <= maximum, "FILE_BYTE_BOUND")
    return path.read_bytes()


def read(path):
    return loads(regular(path))


def within(root, rel):
    need(type(rel) is str and rel and "\\" not in rel and "\0" not in rel, "RELATIVE_PATH")
    part = PurePosixPath(rel)
    need(not part.is_absolute() and part.as_posix() == rel
         and all(x not in ("", ".", "..") for x in rel.split("/")), "SAFE_RELATIVE_PATH")
    root = Path(root).absolute()
    target = root.joinpath(*part.parts)
    need(not any(p.is_symlink() for p in (target, *target.parents)), "NO_PATH_SYMLINK")
    return target


def put(path, raw):
    path = Path(path)
    need(type(raw) is bytes and not any(p.is_symlink() for p in (path, *path.parents)), "CREATE_REGULAR_BYTES")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "sha256": sha(raw)}


def once(path, value):
    return put(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n").encode())


def new_directory(path):
    path = Path(path).absolute()
    need(path.parent.is_dir() and not path.exists()
         and not any(p.is_symlink() for p in (path, *path.parents)), "NEW_OUTPUT_DIRECTORY")
    path.mkdir(mode=0o700)
    return path


def legacy():
    return importlib.import_module("english_kokoro48_once")


def recipe_module():
    return importlib.import_module("english_kokoro_context64_recipe")


def validate_plan(plan, *, check_sources=True):
    need(type(plan) is dict and plan.get("schema") == SCHEMA
         and type(plan.get("enabled")) is bool and plan.get("campaign_id") == CAMPAIGN_ID, "PLAN_IDENTITY")
    need(plan.get("main_before") == (MAIN_BEFORE if plan["enabled"] else None), "UNIQUE_PREDECESSOR")
    design = plan.get("design")
    need(type(design) is dict and digest(design) == plan.get("design_sha256") == DESIGN_SHA256
         and design.get("campaign_id") == CAMPAIGN_ID, "FROZEN_PROSPECTIVE_DESIGN")
    budgets = {"native_tts_request_budget": 48, "parent_phone_calls_per_provider": 48,
        "child_text_calls_per_provider": 64, "total_high_level_model_calls_max": 336,
        "maximum_tts_calls": 48, "maximum_parent_count": 48, "maximum_child_count": 64,
        "maximum_resample_calls": 48, "generation_deadline_seconds": 1800,
        "phone_deadline_seconds": 1800, "text_deadline_seconds": 2400,
        "combine_deadline_seconds": 900, "retry_count": 0,
        "maximum_file_bytes": MAX_FILE, "maximum_expanded_bytes": MAX_EXPANDED}
    need(all(type(plan.get(k)) is int and plan[k] == value for k, value in budgets.items()), "FIXED_FINITE_BUDGETS")
    need(plan.get("diagnostic_only") is True and plan.get("training_allowed") is False
         and plan.get("training_snapshot_allowed") is False and plan.get("automatic_expansion") is False,
         "PERMANENT_PROBE_EXCLUSION")
    old = legacy()
    phone = importlib.import_module("english_kokoro_context32_mask_observe")
    need(plan.get("text_observers") == {p: old.OBSERVERS[p] for p in TEXT_PROVIDERS}
         and plan.get("observers") == phone.OBSERVERS and plan.get("observer_files") == phone.OBSERVER_FILES,
         "FIXED_OBSERVER_IDENTITIES")
    need(plan.get("reference_policy_sha256") == POLICY_SHA256, "UNCHANGED_V2_REFERENCE")
    exclusion = plan.get("source_exposure_exclusion")
    need(type(exclusion) is dict and digest(exclusion) == plan.get("source_exposure_exclusion_sha256") == EXCLUSION_SHA256
         and len(exclusion.get("old_pcm_sha256", [])) == 64
         and len(exclusion.get("old_native_full_f32_sha256", [])) == 48
         and len(exclusion.get("old_request_ids", [])) == 48, "EXACT_KNOWN_PREVIOUS_EXPOSURE")
    frozen = recipe_module().recipe()
    need(digest(frozen) == plan.get("recipe_sha256") and frozen.get("campaign_id") == CAMPAIGN_ID,
         "ACTUAL_FROZEN_RECIPE")
    parents, children = frozen.get("parents"), frozen.get("children")
    need(type(parents) is list and len(parents) == len({r["id"] for r in parents}) == 48
         and type(children) is list and len(children) == len({r["id"] for r in children}) == 64,
         "EXACT_48_PARENTS_64_CHILDREN")
    need(all(r.get("role") == "train" for r in parents + children)
         and Counter(r["kind"] for r in parents) == {"positive": 16, "contrast": 16, "ordinary": 16}
         and Counter(r["kind"] for r in children) == {"positive": 16, "contrast": 16, "ordinary": 16, "outside": 16},
         "FIXED_SOURCE_AND_CONTROL_DENOMINATORS")
    need(not {r["id"] for r in parents + children}.intersection(exclusion["old_request_ids"])
         and not {r["request_sha256"] for r in parents}.intersection(exclusion["old_request_sha256"]),
         "NEW_REQUESTS_NOT_CONSUMED_OLD_ROSTER")
    helpers = plan.get("helper_sha256")
    need(type(helpers) is dict and helpers == design.get("helper_sha256") and len(helpers) == 30,
         "EXACT_EXISTING_30_HELPERS")
    if check_sources:
        for key, path in CODE_PATHS.items():
            need(plan.get(key) == sha(regular(ROOT / path, 256 * 1024)), "EXECUTED_SOURCE_PIN: " + key)
        for path, expected in helpers.items():
            need(type(expected) is str and HEX64.fullmatch(expected)
                 and sha(regular(within(ROOT, path), 256 * 1024)) == expected, "UNCHANGED_HELPER_PIN: " + path)
    return frozen


def identity(env=None):
    env = os.environ if env is None else env
    return {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME")}


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    validate_plan(plan)
    event = read(Path(env["GITHUB_EVENT_PATH"])) if event is None else event
    run = identity(env)
    need(plan["enabled"] is True and env.get("GITHUB_ACTIONS") == "true"
         and run["repository"] == "jiying2007/kws-data" and run["event"] == "push"
         and run["run_attempt"] == "1" and env.get("GITHUB_REF") == "refs/heads/main"
         and type(run["commit"]) is str and HEX40.fullmatch(run["commit"])
         and type(run["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", run["run_id"]), "FIRST_PUBLIC_MAIN_ONLY")
    need(event.get("before") == MAIN_BEFORE == plan["main_before"]
         and event.get("after") == run["commit"] and event["after"] != event["before"]
         and event.get("repository", {}).get("private") is False
         and event["repository"].get("full_name") == run["repository"], "EXACT_NEW_MAIN_SUCCESSOR")
    need(env.get("GITHUB_WORKFLOW_REF") == "jiying2007/kws-data/.github/workflows/english-kokoro-context64-once.yml@refs/heads/main",
         "EXACT_NEW_WORKFLOW")
    return run


class Deadline(BaseException):
    pass


def set_deadline(seconds):
    def stop(signum, _frame):
        raise Deadline("FINITE_STAGE_DEADLINE_OR_TERM: " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)


def file_manifest(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "NO_OUTPUT_SYMLINK")
        if path.is_file() and path.relative_to(root).as_posix() != "FILES.json":
            name = path.relative_to(root).as_posix()
            need("private-model" not in name.split("/"), "PRIVATE_WEIGHTS_NOT_DELIVERED")
            raw = regular(path, allow_empty=True)
            rows.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
    need(len(rows) <= 20000 and sum(r["bytes"] for r in rows) <= MAX_EXPANDED, "FINITE_PACKET_SIZE")
    return {"schema": SCHEMA + "/files", "files": rows}


def closed_readback(root):
    need(read(Path(root) / "FILES.json") == file_manifest(root), "EXACT_CLOSED_PACKET_FILES")


def retain_sources(output, plan):
    for name in sorted(set(CODE_PATHS.values()) | set(plan["helper_sha256"])):
        put(within(output, "code/" + name), regular(within(ROOT, name), 256 * 1024))


def packet_readback(root, plan):
    closed_readback(root)
    need(read(Path(root) / "EXECUTION_PLAN.json") == plan, "PACKET_PLAN_IDENTITY")
    paths = set(CODE_PATHS.values()) | set(plan["helper_sha256"])
    actual = {p.relative_to(Path(root) / "code").as_posix() for p in (Path(root) / "code").rglob("*") if p.is_file()}
    need(actual == paths, "EXACT_EXECUTED_CODE_CLOSURE")
    for name in paths:
        need(regular(within(root, "code/" + name)) == regular(within(ROOT, name)), "EXECUTED_SOURCE_BYTES: " + name)


def verify_wav(raw, pcm, samples, rate=16000):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
              stream.getnframes(), stream.getcomptype()) == (1, 2, rate, samples, "NONE")
             and stream.readframes(samples) == pcm and not stream.readframes(1), "EXACT_MONO_WAV_PCM")


def float_pcm(raw):
    need(len(raw) > 0 and len(raw) % 4 == 0, "FLOAT32_BYTES")
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    need(all(math.isfinite(x) for x in values), "FINITE_FLOAT32")
    return struct.pack("<" + "h" * len(values),
        *(round(max(-1.0, min(32767 / 32768, x)) * 32768) for x in values))


def geometry(samples):
    need(type(samples) is int and samples > 0, "POSITIVE_CHILD_SAMPLES")
    ends = [end for end in range(16000, 38400 + samples + 16000 + 1, 1280)
        if end >= 35200 and end - 31840 <= 38400 and end - 128 >= 38400 + samples]
    return {"positive_support_ends": ends, "positive_support_count": len(ends),
        "minimum": 3, "eligible": len(ends) >= 3, "new_embedding_calls": 0}


def verify_native_lock(root, plan):
    """New campaign identity; unchanged exact 548+140 native weight proof."""
    r = recipe_module()
    lock = read(root / "source/MODEL_LOCK.json")
    need(lock.get("schema") == "english-kokoro-context64-generation-v1/model-lock"
         and lock.get("campaign_id") == CAMPAIGN_ID
         and lock.get("legacy_helper_source_sha") == plan["helper_sha256"]["automation/english_kokoro48_generate.py"]
         and lock.get("native_graph_or_weights_modified") is False
         and all(lock.get(k) == v for k, v in r.MODEL.items())
         and lock.get("package") == r.PACKAGE and lock.get("runtime_versions") == r.RUNTIME_VERSIONS
         and lock.get("all_component_weights_verified") is True
         and lock.get("preset_mixing") is False and lock.get("runtime_g2p_calls") == 0
         and lock.get("forward_attempts_at_lock") == 0 and lock.get("human_speaker_identity_claimed") is False
         and lock.get("diagnostic_only") is True and lock.get("training_allowed") is False, "NEW_CAMPAIGN_NATIVE_LOCK")
    assets = lock.get("assets")
    need(type(assets) is list and len(assets) == len(r.assets()) and len({a["path"] for a in assets}) == len(assets),
         "EXACT_NATIVE_ASSET_CLOSURE")
    by_path = {a["path"]: a for a in assets}
    for expected in r.assets():
        actual = by_path.get(expected["path"], {})
        need(all(actual.get(k) == value for k, value in expected.items())
             and actual.get("verified") is True and actual.get("download_attempts") == 1, "NATIVE_ASSET_PIN")
        if expected["path"] != r.MODEL["model_path"]:
            raw = regular(within(root, "source/runtime/" + expected["path"]))
            need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "RETAINED_NATIVE_ASSET")
    installed = lock.get("installed_package_files")
    need(type(installed) is list and len(installed) == len(r.PACKAGE_FILES)
         and {x["path"]: [x["bytes"], x["sha256"]] for x in installed} == r.PACKAGE_FILES,
         "NATIVE_PACKAGE_IMPLEMENTATION_LOCK")
    for name, (count, expected) in r.PACKAGE_FILES.items():
        raw = regular(within(root, "source/runtime/installed/" + name))
        need(len(raw) == count and sha(raw) == expected, "ACTUAL_INSTALLED_NATIVE_SOURCE")
    wheel = regular(root / "source/runtime" / r.PACKAGE["filename"])
    need(len(wheel) == r.PACKAGE["bytes"] and sha(wheel) == r.PACKAGE["sha256"], "ORIGINAL_NATIVE_WHEEL")
    legacy().verify_weight_diagnostics(root, lock)
    constructed = read(root / "source/MODEL_CONSTRUCTED.json")
    need(constructed.get("campaign_id") == CAMPAIGN_ID
         and constructed.get("legacy_helper_source_sha") == r.LEGACY_GENERATOR_SHA256
         and constructed.get("constructor_and_runtime_transforms_returned") is True
         and constructed.get("model_instances") == 1 and constructed.get("model_forward_calls") == 0,
         "ONE_NATIVE_CONSTRUCTION_BEFORE_FORWARD")
    loading = read(root / "source/MODEL_LOAD_STARTED.json")
    need(loading.get("campaign_id") == CAMPAIGN_ID and loading.get("forward_attempts") == 0
         and loading.get("local_files_only") is True and loading.get("assets_verified") == assets,
         "VERIFIED_ASSETS_BEFORE_NATIVE_CONSTRUCTION")
    vocab = read(root / "source/runtime/config.json").get("vocab")
    need(type(vocab) is dict and all(all(c in vocab for c in p["raw_phonemes"])
        for p in r.recipe()["parents"]), "EXACT_NATIVE_PARENT_VOCABULARY")
    return vocab


def duration_coordinates(parent, duration, native_samples):
    need(type(duration) is list and len(duration) == len(parent["raw_phonemes"]) + 2
         and all(type(x) is int and 1 <= x <= 50 for x in duration)
         and type(native_samples) is int and native_samples == 600 * sum(duration), "ACTUAL_DURATION_GEOMETRY")
    def point(char):
        return 600 * (max(duration[0] - 3, 0) + sum(duration[1:char + 1]))
    target, start, end = point(125), point(78), point(101)
    need(0 <= start < end < target < native_samples and all(x % 3 == 0 for x in (target, start, end, native_samples)),
         "ORDERED_PREDICTED_CHAR_BOUNDARIES")
    return {"native_samples": native_samples, "target_start_prediction_native": target,
        "target_start_parent16k": 2 * target // 3, "boundary_start_prediction_native": start,
        "boundary_end_prediction_native": end, "boundary_start_parent16k": 2 * start // 3,
        "boundary_end_parent16k": 2 * end // 3, "duration_boundary_is_acoustic_gold": False,
        "timestamp_kind": "model_prediction_only"}


def expected_child_bounds(parent, child, duration):
    geo = duration_coordinates(parent, duration, parent["native_samples"])
    outside = child["kind"] == "outside"
    start = max((geo["boundary_start_prediction_native"] if outside else geo["target_start_prediction_native"]) - 3600, 0)
    end = min(geo["boundary_end_prediction_native"] + 3600, parent["native_samples"]) if outside else parent["native_samples"]
    need(start % 3 == end % 3 == 0 and 0 <= start < end <= parent["native_samples"], "INTEGER_EXACT_CHILD_BOUNDARY")
    a, b = 2 * start // 3, 2 * end // 3
    need(not outside or b <= geo["target_start_parent16k"], "TARGET_OUTSIDE_CONTROL_CHILD")
    return {"child_start_native_sample": start, "child_end_native_sample": end,
        "child_start_parent16k": a, "child_end_parent16k": b,
        "target_start_prediction_native": geo["target_start_prediction_native"],
        "target_start_parent16k": geo["target_start_parent16k"],
        "left_margin_native_samples": 3600, "right_margin_native_samples": 3600 if outside else 0,
        "right_endpoint": "predicted_boundary_end_plus150ms" if outside else "native_eof",
        "exact_parent_pcm_slice": True, "boundaries_model_predicted": True,
        "boundary_is_acoustic_gold": False, "parent_resampled_once": True,
        "child_resample_calls": 0, "recrop_count": 0, "added_silence_samples": 0}


def media(root, row, key, expected_path):
    need(row.get(key + "_path") == expected_path, "REGISTERED_MEDIA_PATH: " + key)
    raw = regular(within(root, expected_path))
    need(sha(raw) == row.get(key + "_sha256"), "REGISTERED_MEDIA_SHA: " + key)
    return raw


def parent_audio_readback(root, row, vocab):
    r = recipe_module()
    prefix = "source/parents/" + row["id"] + "/"
    native = media(root, row, "native_full_f32", prefix + "native-full.f32")
    native_wav = media(root, row, "native_full_wav", prefix + "native-full.wav")
    derived = media(root, row, "resampled_f32", prefix + "resampled-16k.f32")
    pcm = media(root, row, "pcm", prefix + "parent-16k.pcm")
    wav = media(root, row, "wav", prefix + "parent-16k.wav")
    duration_raw = media(root, row, "duration", prefix + "pred_dur.json")
    duration = loads(duration_raw)
    need(type(row.get("native_samples")) is int and len(native) == 4 * row["native_samples"]
         and type(row.get("pcm_samples")) is int and 400 <= row["pcm_samples"] <= 192000
         and row["pcm_samples"] == (row["native_samples"] * 2 + 2) // 3
         and len(derived) == row["pcm_samples"] * 4 and len(pcm) == row["pcm_samples"] * 2,
         "FULL_PARENT_FLOAT_AND_PCM_GEOMETRY")
    verify_wav(native_wav, float_pcm(native), row["native_samples"], 24000)
    need(pcm == float_pcm(derived), "PARENT_INT16_SINGLE_QUANTIZATION_READBACK")
    verify_wav(wav, pcm, row["pcm_samples"])
    ids = [0, *[vocab[c] for c in row["raw_phonemes"]], 0]
    need(duration.get("schema") == "english-kokoro-context64-generation-v1/native-duration"
         and duration.get("campaign_id") == CAMPAIGN_ID and duration.get("id") == row["id"]
         and duration.get("phonemes") == row["raw_phonemes"] and duration.get("input_ids") == ids
         and duration.get("includes_bos_eos") is True and duration.get("samples_per_frame") == 600
         and duration.get("sample_rate") == 24000 and duration.get("native_samples") == row["native_samples"]
         and duration.get("boundary_is_acoustic_gold") is False and duration.get("boundary_config") == r.BOUNDARIES
         and duration.get("target_start_char") == row["target_start_char"] == 125
         and duration.get("target_end_char_exclusive") == row["target_end_char_exclusive"] == len(row["raw_phonemes"]),
         "ACTUAL_NATIVE_DURATION_AND_DIRECT_IDS")
    geo = duration_coordinates(row, duration.get("pred_dur"), row["native_samples"])
    need(all(row.get(k) == value for k, value in geo.items())
         and row.get("sample_rate") == 16000 and row.get("channels") == 1 and row.get("sample_width") == 2
         and row.get("native_sample_rate") == 24000 and row.get("native_float_format") == "float32-little-endian"
         and row.get("resampled_float_format") == "float32-little-endian"
         and row.get("resampler") == r.GEOMETRY["resampler"] and row.get("normalization_applied") is False
         and row.get("parent_16k_quantized_once") is True and row.get("synthetic_silence_added") is False
         and row.get("observed_target_pronunciation") is False and row.get("human_gold") is False,
         "FULL_PARENT_DERIVATION_SCOPE")
    return {"pcm": pcm, "duration": duration["pred_dur"], "ids": ids,
        "native_sha256": sha(native), "pcm_sha256": sha(pcm)}


def child_audio_readback(root, row, parent, parent_media):
    r = recipe_module()
    prefix = "source/children/" + row["id"] + "/"
    pcm = media(root, row, "pcm", prefix + "child-16k.pcm")
    wav = media(root, row, "wav", prefix + "child-16k.wav")
    derivation = loads(media(root, row, "derivation", prefix + "DERIVATION.json"))
    bounds = expected_child_bounds(parent, row, parent_media["duration"])
    need(all(row.get(k) == value for k, value in bounds.items()), "UNMODIFIED_CHILD_BOUNDARY_RULE")
    a, b = bounds["child_start_parent16k"], bounds["child_end_parent16k"]
    need(0 <= a < b <= parent["pcm_samples"] and row.get("pcm_samples") == b - a
         and len(pcm) == 2 * (b - a) and pcm == parent_media["pcm"][2 * a:2 * b]
         and row.get("parent_pcm_sha256") == parent["pcm_sha256"]
         and (row.get("sample_rate"), row.get("channels"), row.get("sample_width")) == (16000, 1, 2),
         "CHILD_EXACT_INT16_PARENT_SUBSTRING")
    verify_wav(wav, pcm, row["pcm_samples"])
    geom = geometry(row["pcm_samples"])
    need(row.get("whole_support_windows") == geom["positive_support_count"]
         and row.get("geometry_usable_positive") == geom["eligible"]
         and row.get("whole_support_embedding_end_samples") == geom["positive_support_ends"]
         and row.get("geometry_helper_sha256") == r.GEOMETRY_HELPER_SHA256
         and row.get("geometry_prefix_samples") == 38400 and row.get("geometry_suffix_samples") == 16000
         and row.get("geometry_is_integer_check_only") is True and row.get("geometry_audio_generated") is False
         and row.get("human_word_end_annotation") is False, "ORIGINAL_INTEGER_KWS_SUPPORT_GEOMETRY")
    required = {"campaign_id": CAMPAIGN_ID, "id": row["id"], "request_sha256": row["request_sha256"],
        "parent_id": parent["id"], "parent_request_sha256": parent["request_sha256"],
        "parent_pcm_path": parent["pcm_path"], "parent_pcm_sha256": parent["pcm_sha256"],
        "parent_pcm_samples": parent["pcm_samples"], "duration_path": parent["duration_path"],
        "duration_sha256": parent["duration_sha256"], "boundary_config": r.BOUNDARIES,
        "child_character_range": row["child_character_range"], "pcm_path": row["pcm_path"],
        "pcm_sha256": row["pcm_sha256"], "wav_path": row["wav_path"], "wav_sha256": row["wav_sha256"],
        "pcm_samples": row["pcm_samples"], "sample_rate": 16000, "channels": 1, "sample_width": 2,
        "whole_support_windows": geom["positive_support_count"], "geometry_usable_positive": geom["eligible"],
        "child_model_calls": 0, "child_resample_calls": 0, "normalization_applied": False, **bounds}
    need(all(derivation.get(k) == value for k, value in required.items())
         and derivation.get("training_allowed") is False and derivation.get("diagnostic_only") is True,
         "SAVED_EXACT_SLICE_DERIVATION")
    return {"pcm_sha256": sha(pcm), "geometry": geom}


def generation_marker_identity(row, entity):
    return {"schema": "english-kokoro-context64-generation-v1/" + entity, "campaign_id": CAMPAIGN_ID,
        "id": row["id"], "request_sha256": row["request_sha256"], "preset_id": row["preset_id"],
        "kind": row["kind"], "role": "train", "diagnostic_only": True, "training_allowed": False, "retry_count": 0}


def generation_ledger_readback(root, row, entity, *, vocab=None, parent=None):
    folder = root / "source" / ("parents" if entity == "parent" else "children") / row["id"]
    identity_doc = generation_marker_identity(row, entity)
    need(row.get("generation_status") in ("success", "failed", "not_started"), "GENERATION_ROW_STATUS")
    if entity == "parent":
        pairs = (("FORWARD", "forward_attempts", "forward_returned", "parent"),
                 ("RESAMPLE", "resample_attempts", "resample_returned", "resample"))
    else:
        pairs = (("DERIVATION", "derivation_attempts", "derivation_returned", "child"),)
    started = read(folder / "STARTED.json") if (folder / "STARTED.json").is_file() else None
    if started is not None:
        need(all(started.get(k) == v for k, v in identity_doc.items()), "GENERATION_START_IDENTITY")
        if entity == "parent":
            need(vocab is not None and started.get("phonemes") == row["raw_phonemes"]
                 and started.get("phoneme_ids") == [0, *[vocab[c] for c in row["raw_phonemes"]], 0]
                 and started.get("seed") == row["seed"] and started.get("target_start_char") == 125
                 and started.get("voice_pack_index") == len(row["raw_phonemes"]) - 1
                 and started.get("synthesis_config") == recipe_module().SYNTHESIS
                 and started.get("forward_attempts") == 0 and started.get("forward_returned") is False
                 and started.get("runtime_g2p_calls") == 0, "PREREGISTERED_DIRECT_NATIVE_REQUEST")
        else:
            need(parent is not None and started.get("parent_id") == parent["id"]
                 and started.get("parent_pcm_sha256") == parent["pcm_sha256"]
                 and started.get("derivation_attempts") == 0 and started.get("derivation_returned") is False
                 and started.get("child_model_calls") == started.get("child_resample_calls") == 0, "CHILD_STARTED_ON_SAVED_PARENT")
    completed = read(folder / "COMPLETED.json") if (folder / "COMPLETED.json").is_file() else None
    failed = read(folder / "FAILED.json") if (folder / "FAILED.json").is_file() else None
    need(not (completed is not None and failed is not None), "ONE_TERMINAL_GENERATION_RECEIPT")
    terminal = completed if completed is not None else failed
    if terminal is not None:
        need(all(terminal.get(k) == v for k, v in identity_doc.items()) and terminal.get("row") == row,
             "FINAL_GENERATION_ROW_EXACT_RECEIPT")
    for prefix, attempts_key, returns_key, marker_entity in pairs:
        attempts, returned = row.get(attempts_key), row.get(returns_key)
        need(type(attempts) is int and attempts in (0, 1) and type(returned) is bool
             and (not returned or attempts == 1), "BOUNDED_GENERATION_LEDGER")
        before_path, after_path = folder / (prefix + "_STARTED.json"), folder / (prefix + "_RETURNED.json")
        before = read(before_path) if before_path.is_file() else None
        after = read(after_path) if after_path.is_file() else None
        need(attempts == int(before is not None) and (before is None or started is not None)
             and (after is None or returned and before is not None), "GENERATION_COUNTER_MARKERS")
        expected = generation_marker_identity(row, marker_entity)
        for marker, is_return in ((before, False), (after, True)):
            if marker is not None:
                need(all(marker.get(k) == v for k, v in expected.items())
                     and marker.get(attempts_key) == 1 and marker.get(returns_key) is is_return, "GENERATION_STAGE_MARKER_IDENTITY")
                if prefix == "RESAMPLE":
                    need(marker.get("native_full_f32_sha256") == row.get("native_full_f32_sha256")
                         and marker.get("full_parent_input") is True and marker.get("child_resample_calls") == 0
                         and marker.get("resampler") == recipe_module().GEOMETRY["resampler"], "ONLY_ONE_FULL_PARENT_RESAMPLE")
        if returned and after is None:
            need(failed is not None and failed.get(returns_key) is True, "RETURN_WITHOUT_MARKER_HAS_FAILED_RECEIPT")
        if terminal is not None:
            need(terminal.get(attempts_key) == attempts and terminal.get(returns_key) is returned, "GENERATION_TERMINAL_COUNTERS")
    if row["generation_status"] == "success":
        need(completed is not None and completed.get("status") == "COMPLETE"
             and completed.get("generation_status") == "success"
             and all(row[a] == 1 and row[b] is True for _, a, b, _ in pairs), "GENERATION_SUCCESS_REQUIRES_COMPLETE_EXECUTION")
        if entity == "parent":
            r = recipe_module()
            need(completed.get("phoneme_ids") == [0, *[vocab[c] for c in row["raw_phonemes"]], 0]
                 and completed.get("generation_model_sha256") == r.MODEL["model_sha256"]
                 and completed.get("voice_pack_sha256") == r.VOICE_ASSETS[row["preset_id"]]["sha256"], "ACTUAL_NATIVE_AND_PRESET_IDENTITY")
    else:
        need(completed is None and type(row.get("error_type")) is str and type(row.get("error_code")) is str,
             "EXPLICIT_FAILED_OR_UNSTARTED_SOURCE")
        if failed is not None:
            first_count = row[pairs[0][1]]
            need(failed.get("status") == ("FAILED_NO_RETRY" if first_count else "NOT_STARTED_NO_RETRY")
                 and failed.get("error_type") == row["error_type"] and failed.get("error_code") == row["error_code"], "FAILURE_REASON_NOT_REWRITTEN")
    if entity == "parent":
        need(not row["resample_attempts"] or row["forward_returned"], "RESAMPLE_ONLY_AFTER_NATIVE_RETURN")
    else:
        need(row.get("child_model_calls") == row.get("child_resample_calls") == 0,
             "CHILD_NEVER_MODEL_OR_RESAMPLE")
    return {"completed": int(row["generation_status"] == "success"), "terminal_retained": terminal is not None}


def read_generation(root, plan, *, allow_partial):
    root = Path(root)
    frozen = recipe_module().recipe()
    result = read(root / "source/RESULT.json")
    run = result.get("run")
    need(result.get("schema") == "english-kokoro-context64-generation-v1" and result.get("campaign_id") == CAMPAIGN_ID
         and type(run) is dict and run.get("repository") == "jiying2007/kws-data"
         and run.get("event") == "push" and run.get("run_attempt") == "1"
         and type(run.get("commit")) is str and HEX40.fullmatch(run["commit"])
         and type(run.get("run_id")) is str and re.fullmatch(r"[1-9][0-9]*", run["run_id"])
         and result.get("declared_parent_count") == 48 and result.get("declared_child_count") == 64
         and result.get("retry_count") == 0 and result.get("training_allowed") is False
         and result.get("permanently_excluded_from_training") is True and result.get("automatic_expansion") is False
         and result.get("source_exposure_exclusion_sha256") == EXCLUSION_SHA256,
         "SOURCE_RESULT_SCOPE_AND_RUN")
    need(read(root / "source/RECIPE.json") == frozen and read(root / "source/EXECUTION_PLAN.json") == plan,
         "SOURCE_FROZEN_RECIPE_AND_PLAN")
    request = read(root / "source/REQUESTS.json")
    need(request.get("run") == run and request.get("parents") == frozen["parents"]
         and request.get("children") == frozen["children"]
         and request.get("forward_attempts_at_write") == request.get("resample_attempts_at_write") == 0,
         "ALL_REQUESTS_SAVED_BEFORE_CALLS")
    source_start = read(root / "source/STARTED.json")
    need(source_start.get("run") == run and source_start.get("campaign_id") == CAMPAIGN_ID
         and source_start.get("declared_parent_count") == source_start.get("maximum_tts_calls") == 48
         and source_start.get("declared_child_count") == 64 and source_start.get("retry_count") == 0
         and source_start.get("forward_attempts_at_start") == 0, "SOURCE_STARTED_COMPLETE_DENOMINATOR")
    parents, children = result.get("parents"), result.get("children")
    need(type(parents) is list and len(parents) == 48 and type(children) is list and len(children) == 64,
         "PRESERVED_PARENT_CHILD_DENOMINATORS")
    for name, rows in (("parents", parents), ("children", children)):
        doc = read(root / "source" / (name.upper() + ".json"))
        need(doc.get("run") == run and doc.get("rows") == rows, "DUPLICATE_ROSTER_BYTES_BINDING")
        need([r["id"] for r in rows] == [r["id"] for r in frozen[name]], "NO_ROW_SELECTION_OR_REORDERING")
        for row, expected in zip(rows, frozen[name]):
            need(all(row.get(k) == value for k, value in expected.items()), "PREREGISTERED_ROW_UNCHANGED")
    vocab = verify_native_lock(root, plan) if (root / "source/MODEL_LOCK.json").is_file() else None
    need(vocab is not None or not any(r.get("forward_attempts") for r in parents), "NO_FORWARD_BEFORE_VALID_NATIVE_LOCK")
    parent_map = {r["id"]: r for r in parents}
    parent_media = {}
    pcm_seen, native_seen = set(), set()
    exposure = plan["source_exposure_exclusion"]
    missing_terminal = []
    for parent in parents:
        ledger = generation_ledger_readback(root, parent, "parent", vocab=vocab)
        if not ledger["terminal_retained"]:
            missing_terminal.append(parent["id"])
        if parent["generation_status"] == "success":
            values = parent_audio_readback(root, parent, vocab)
            need(values["native_sha256"] not in native_seen and values["native_sha256"] not in exposure["old_native_full_f32_sha256"]
                 and values["pcm_sha256"] not in pcm_seen and values["pcm_sha256"] not in exposure["old_pcm_sha256"],
                 "SUCCESSFUL_PARENT_IS_NEW_UNIQUE_BYTES")
            native_seen.add(values["native_sha256"])
            pcm_seen.add(values["pcm_sha256"])
            parent_media[parent["id"]] = values
    for child in children:
        parent = parent_map[child["parent_id"]]
        ledger = generation_ledger_readback(root, child, "child", parent=parent)
        if not ledger["terminal_retained"]:
            missing_terminal.append(child["id"])
        if child["generation_status"] == "success":
            need(parent["generation_status"] == "success", "SUCCESSFUL_CHILD_HAS_COMPLETE_PARENT")
            values = child_audio_readback(root, child, parent, parent_media[parent["id"]])
            need(values["pcm_sha256"] not in pcm_seen and values["pcm_sha256"] not in exposure["old_pcm_sha256"],
                 "SUCCESSFUL_CHILD_IS_NEW_UNIQUE_PCM")
            pcm_seen.add(values["pcm_sha256"])
    counts = {"actual_forward_attempts": sum(r["forward_attempts"] for r in parents),
        "actual_forward_returns": sum(r["forward_returned"] for r in parents),
        "actual_resample_attempts": sum(r["resample_attempts"] for r in parents),
        "actual_resample_returns": sum(r["resample_returned"] for r in parents),
        "completed_parents": sum(r["generation_status"] == "success" for r in parents),
        "completed_children": sum(r["generation_status"] == "success" for r in children)}
    need(all(type(result.get(k)) is int and result[k] == value for k, value in counts.items()), "RECOUNTED_ALL_GENERATION_CALLS")
    failures, writes = result.get("source_integrity_failures"), result.get("receipt_write_failures")
    need(type(failures) is list and type(writes) is list and all(r.get("id") in parent_map or r.get("id") in {c["id"] for c in children}
         for r in failures + writes), "EXPLICIT_SOURCE_FAILURE_IDENTITIES")
    need(set(missing_terminal) <= {r["id"] for r in writes}, "MISSING_TERMINAL_HAS_EXPLICIT_WRITE_FAILURE")
    for failure in failures:
        row = next(r for r in parents + children if r["id"] == failure["id"])
        need(row.get("generation_status") != "success" and row.get("error_type") == "SourceIntegrityError"
             and row.get("error_code") == failure.get("code"), "SOURCE_INTEGRITY_FAILURE_NOT_HIDDEN")
    complete = counts["completed_parents"] == 48 and counts["completed_children"] == 64
    complete = complete and all(counts[k] == 48 for k in ("actual_forward_attempts", "actual_forward_returns", "actual_resample_attempts", "actual_resample_returns"))
    complete = complete and not failures and not writes and result.get("error") is None
    need(result.get("complete") is complete and result.get("source_integrity_passed") is complete
         and result.get("status") == ("COMPLETE" if complete else "INCOMPLETE_NO_RETRY"), "SOURCE_COMPLETENESS_FOLLOWS_ACTUAL_ROWS")
    need(all(result.get(k) == 0 for k in ("old_audio_model_calls", "child_tts_calls", "child_resample_calls",
        "parent_phone_calls", "child_text_calls", "kws_forward_calls", "training_calls")), "SOURCE_ONLY_TTS_AND_EXACT_CHILD_BYTES")
    need(allow_partial or complete, "SOURCE_NOT_COMPLETE_NO_OBSERVATION")
    return {"schema": SOURCE_SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run,
        "source_result_sha256": sha(regular(root / "source/RESULT.json")),
        "parents": parents, "children": children, "source_integrity_passed": complete,
        "generation_summary": {"declared_count": 48, "actual_forward_attempts": counts["actual_forward_attempts"],
            "native_forward_returns": counts["actual_forward_returns"], "completed": counts["completed_parents"],
            "forward_attempts_lower_bound": counts["actual_forward_attempts"],
            "forward_attempts_upper_bound": counts["actual_forward_attempts"], "complete": complete,
            "completed_children": counts["completed_children"], "resample_attempts": counts["actual_resample_attempts"],
            "resample_returns": counts["actual_resample_returns"]},
        "source_exposure_exclusion_sha256": EXCLUSION_SHA256,
        "training_allowed": False, "permanently_excluded_from_training": True,
        "resampling_reexecuted_during_readback": False, "model_calls_during_readback": 0}


def source_readback(root, plan, allow_partial=False):
    validate_plan(plan)
    packet_readback(root, plan)
    manifest = read_generation(root, plan, allow_partial=allow_partial)
    need(read(Path(root) / "INPUT_MANIFEST.json") == manifest, "ROOT_INPUT_MANIFEST_ACTUAL_SOURCE_BINDING")
    outer = read(Path(root) / "RESULT.json")
    need(outer.get("run") == manifest["run"] and outer.get("generation_summary") == manifest["generation_summary"]
         and outer.get("source_integrity_passed") == manifest["source_integrity_passed"]
         and outer.get("training_allowed") is False, "ROOT_SOURCE_RESULT_ACTUAL_COUNTS")
    return manifest


def text_identity(provider, plan):
    need(provider in TEXT_PROVIDERS, "THREE_TEXT_PROVIDERS_ONLY")
    row = plan["text_observers"][provider]
    return {k: row[k] for k in ("recognizer_id", "model_family", "model_sha256")} | {
        "decode_config_sha256": digest(row["decode_config"])}


def child_identity(row, provider, plan, run):
    return {"schema": TEXT_SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run,
        "provider": provider, "id": row["id"], "child_id": row["id"], "parent_id": row["parent_id"],
        "input_scope": "exact_child_16k_pcm", "source_pcm_sha256": row.get("pcm_sha256"),
        "source_wav_sha256": row.get("wav_sha256"), "source_samples": row.get("pcm_samples"),
        "observer_identity": text_identity(provider, plan), "retry_count": 0,
        "target_conditioned": False, "target_input": None, "requested_text_passed": False,
        "role_passed": False, "training_allowed": False}


def initial_text_ledger(children):
    return [{"id": r["id"], "parent_id": r["parent_id"], "status": "not_started",
        "input_started": False, "forward_attempts": 0, "forward_returned": False,
        "capture_saved": False, "failure": None, "retry_count": 0} for r in children]


def text_loop(provider, root, children, plan, run, output, decode, ledger):
    """One high-level native decode per successful child; never retries."""
    for row, state in zip(children, ledger):
        if row.get("generation_status") != "success":
            state["failure"] = "SOURCE_CHILD_NOT_SUCCESSFUL"
            continue
        folder = output / row["id"]
        base = child_identity(row, provider, plan, run)
        try:
            folder.mkdir(mode=0o700)
            pcm = regular(within(root, row["pcm_path"]))
            wav = regular(within(root, row["wav_path"]))
            need(sha(pcm) == row["pcm_sha256"] and sha(wav) == row["wav_sha256"], "CHILD_INPUT_RECHECK")
            verify_wav(wav, pcm, row["pcm_samples"])
            once(folder / "STARTED.json", base | {"forward_attempts": 0, "forward_returned": False})
            state["input_started"] = True
            once(folder / "FORWARD_STARTED.json", base | {"forward_attempts": 1, "forward_returned": False})
            state["forward_attempts"] = 1
            value = decode(within(root, row["wav_path"]), folder)
            state["forward_returned"] = True
            once(folder / "FORWARD_RETURNED.json", base | {"forward_attempts": 1, "forward_returned": True})
            need(type(value) is str and 0 < len(value.strip()) <= 1024
                 and not any(ord(c) < 32 for c in value.strip()), "COMPLETE_NATIVE_TEXT_STRING")
            capture = base | text_identity(provider, plan) | {"status": "success", "raw_text": value,
                "input_coverage_complete": True, "decoder_output_complete": True,
                "counter_scope": "high_level_one_input_decode", "diagnostic_only": True}
            saved = once(folder / "CAPTURE.json", capture)
            state.update(status="complete", capture_saved=True, capture_sha256=saved["sha256"])
        except BaseException as exc:
            state.update(status="failed", failure=type(exc).__name__ + ": " + str(exc))
            try:
                once(folder / "FAILED.json", base | state)
            except BaseException as write_exc:
                state["failure_receipt_write_error"] = type(write_exc).__name__ + ": " + str(write_exc)
            if not isinstance(exc, Exception):
                raise


def observe_text(provider, source_root, output_root, asset_root, plan, run, manifest):
    need(provider in TEXT_PROVIDERS and manifest["run"] == run, "CHILD_TEXT_SCOPE_AND_RUN")
    children = manifest["children"]
    output = new_directory(output_root)
    assets = new_directory(asset_root)
    need(output != assets and output not in assets.parents and assets not in output.parents, "DISTINCT_PRIVATE_ASSETS")
    ledger = initial_text_ledger(children)
    failure = None
    loader_attempts = loader_returns = 0
    once(output / "STARTED.json", {"schema": TEXT_SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run,
        "provider": provider, "declared_count": 64, "maximum_calls": 64,
        "source_result_sha256": sha(regular(Path(source_root) / "source/RESULT.json")),
        "child_roster_sha256": digest(children),
        "input_scope": "exact_child_16k_pcm", "observer_identity": text_identity(provider, plan),
        "counter_scope": "high_level_one_input_decode", "retry_count": 0, "training_allowed": False})
    start = time.monotonic()
    try:
        if any(r.get("generation_status") == "success" for r in children):
            once(output / "MODEL_LOAD_STARTED.json", {"schema": TEXT_SCHEMA, "run": run,
                "provider": provider, "loader_attempts": 1, "high_level_decode_attempts": 0})
            loader_attempts = 1
            decode, native_lock = legacy().load_observer(provider, assets, plan)
            loader_returns = 1
            once(output / "MODEL_LOCK.json", {"schema": TEXT_SCHEMA, "run": run, "provider": provider,
                "model": plan["text_observers"][provider], "identity": text_identity(provider, plan),
                "actual_runtime_lock": native_lock, "loader_attempts": 1, "loader_returns": 1,
                "native_model_construction_count_claimed": False,
                "worker_sha256": plan["worker_sha256"], "input_scope": "exact_child_16k_pcm"})
            text_loop(provider, Path(source_root), children, plan, run, output, decode, ledger)
        else:
            for state in ledger:
                state["failure"] = "SOURCE_CHILD_NOT_SUCCESSFUL"
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        if assets.exists():
            try:
                need(assets.is_dir() and not assets.is_symlink(), "PRIVATE_ASSET_CLEANUP")
                shutil.rmtree(assets)
            except BaseException as exc:
                failure = failure or ("PRIVATE_ASSET_CLEANUP: " + type(exc).__name__ + ": " + str(exc))
        calls = sum(r["forward_attempts"] for r in ledger)
        returns = sum(r["forward_returned"] for r in ledger)
        completed = sum(r["capture_saved"] for r in ledger)
        result = {"schema": TEXT_SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run, "provider": provider,
            "status": "COMPLETE_DIAGNOSTIC" if completed == 64 and failure is None else "INCOMPLETE_DIAGNOSTIC_NO_RETRY",
            "declared_count": 64, "rows": ledger, "input_started": sum(r["input_started"] for r in ledger),
            "actual_forward_attempts": calls, "native_forward_returns": returns, "completed": completed,
            "failed": sum(r["status"] == "failed" for r in ledger),
            "not_started": sum(r["status"] == "not_started" for r in ledger),
            "loader_attempts": loader_attempts, "loader_returns": loader_returns, "failure": failure,
            "counter_scope": "high_level_one_input_decode", "input_scope": "exact_child_16k_pcm",
            "retry_count": 0, "training_allowed": False, "old_labels_modified": False,
            "elapsed_seconds": time.monotonic() - start}
        once(output / "RESULT.json", result)
    return result


def read_text_provider(root, children, plan, run_identity, *, provider):
    """Read terminal ledgers honestly; missing final means bounded unknown calls."""
    need(provider in TEXT_PROVIDERS and len(children) == 64, "EXPECTED_CHILD_PROVIDER_AND_DENOMINATOR")
    root = Path(root)
    if root.exists():
        need(root.is_dir() and not root.is_symlink(), "TEXT_PROVIDER_DIRECTORY")
        allowed_roots = {"STARTED.json", "MODEL_LOAD_STARTED.json", "MODEL_LOCK.json", "RESULT.json", "FILES.json"}
        allowed_children = {r["id"] for r in children}
        allowed_leaf = {"STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "CAPTURE.json", "FAILED.json"}
        for path in root.rglob("*"):
            need(not path.is_symlink(), "TEXT_EVIDENCE_NO_SYMLINK")
            if path.is_file():
                parts = path.relative_to(root).parts
                need((len(parts) == 1 and parts[0] in allowed_roots) or
                     (len(parts) == 2 and parts[0] in allowed_children and parts[1] in allowed_leaf),
                     "ONLY_REGISTERED_CHILD_TEXT_EVIDENCE")
    terminal = read(root / "RESULT.json") if (root / "RESULT.json").is_file() else None
    if (root / "FILES.json").is_file():
        closed_readback(root)
    if terminal is not None:
        need((root / "FILES.json").is_file(), "TERMINAL_REQUIRES_CLOSED_FILES")
        need(terminal.get("schema") == TEXT_SCHEMA and terminal.get("campaign_id") == CAMPAIGN_ID
             and terminal.get("run") == run_identity and terminal.get("provider") == provider
             and terminal.get("declared_count") == 64 and terminal.get("retry_count") == 0
             and terminal.get("input_scope") == "exact_child_16k_pcm"
             and terminal.get("counter_scope") == "high_level_one_input_decode"
             and terminal.get("training_allowed") is False, "TERMINAL_TEXT_IDENTITY")
        ledger = terminal.get("rows")
        need(type(ledger) is list and len(ledger) == 64 and [r["id"] for r in ledger] == [r["id"] for r in children], "ALL64_TEXT_LEDGER")
        started = read(root / "STARTED.json")
        need(started.get("schema") == TEXT_SCHEMA and started.get("campaign_id") == CAMPAIGN_ID
             and started.get("run") == run_identity and started.get("provider") == provider
             and started.get("declared_count") == started.get("maximum_calls") == 64
             and started.get("child_roster_sha256") == digest(children)
             and type(started.get("source_result_sha256")) is str and HEX64.fullmatch(started["source_result_sha256"])
             and started.get("observer_identity") == text_identity(provider, plan)
             and started.get("input_scope") == "exact_child_16k_pcm" and started.get("retry_count") == 0,
             "EXPECTED_ROOT_TEXT_START")
    else:
        ledger = [None] * 64
    rows = []
    retained_calls = retained_returns = retained_captures = 0
    for child, state in zip(children, ledger):
        base = child_identity(child, provider, plan, run_identity)
        folder = root / child["id"]
        markers = {}
        for name, count, returned in (("STARTED", 0, False), ("FORWARD_STARTED", 1, False), ("FORWARD_RETURNED", 1, True)):
            path = folder / (name + ".json")
            if path.is_file():
                doc = read(path)
                need(all(doc.get(k) == v for k, v in base.items()) and doc.get("forward_attempts") == count
                     and doc.get("forward_returned") is returned, "EXACT_TEXT_MARKER: " + name)
                markers[name] = doc
        need("FORWARD_STARTED" not in markers or "STARTED" in markers, "TEXT_FORWARD_HAS_STARTED")
        need("FORWARD_RETURNED" not in markers or "FORWARD_STARTED" in markers, "TEXT_RETURN_HAS_ATTEMPT")
        retained_calls += int("FORWARD_STARTED" in markers)
        retained_returns += int("FORWARD_RETURNED" in markers)
        capture = read(folder / "CAPTURE.json") if (folder / "CAPTURE.json").is_file() else None
        if capture is not None:
            need(all(capture.get(k) == v for k, v in base.items())
                 and all(capture.get(k) == v for k, v in text_identity(provider, plan).items())
                 and capture.get("status") == "success" and capture.get("input_coverage_complete") is True
                 and capture.get("decoder_output_complete") is True and capture.get("diagnostic_only") is True
                 and type(capture.get("raw_text")) is str and 0 < len(capture["raw_text"].strip()) <= 1024
                 and not any(ord(c) < 32 for c in capture["raw_text"].strip())
                 and set(markers) == {"STARTED", "FORWARD_STARTED", "FORWARD_RETURNED"}, "COMPLETE_CHILD_TEXT_CAPTURE")
            retained_captures += 1
        failure = read(folder / "FAILED.json") if (folder / "FAILED.json").is_file() else None
        if state is not None:
            need(state.get("parent_id") == child["parent_id"] and state.get("retry_count") == 0
                 and state.get("status") in ("not_started", "failed", "complete")
                 and type(state.get("input_started")) is bool and type(state.get("forward_attempts")) is int
                 and state["forward_attempts"] in (0, 1) and type(state.get("forward_returned")) is bool
                 and type(state.get("capture_saved")) is bool, "VALID_TEXT_LEDGER_STATE")
            need(state["input_started"] == ("STARTED" in markers)
                 and state["forward_attempts"] == int("FORWARD_STARTED" in markers)
                 and (not state["forward_returned"] or state["forward_attempts"] == 1)
                 and ("FORWARD_RETURNED" not in markers or state["forward_returned"])
                 and state["capture_saved"] == (capture is not None), "TEXT_LEDGER_MARKER_CLOSURE")
            if failure is not None:
                need(all(failure.get(k) == v for k, v in base.items())
                     and all(failure.get(k) == state.get(k) for k in ("status", "input_started", "forward_attempts", "forward_returned", "capture_saved", "failure")),
                     "FAILED_TEXT_RECEIPT_CLOSURE")
            if state["status"] == "complete":
                need(capture is not None and failure is None and state["forward_returned"] is True
                     and state.get("failure") is None and state.get("capture_sha256") == sha(regular(folder / "CAPTURE.json")), "SUCCESS_TEXT_LEDGER")
            elif state["status"] == "failed":
                need(capture is None and type(state.get("failure")) is str
                     and (failure is not None or type(state.get("failure_receipt_write_error")) is str), "EXPLICIT_FAILED_TEXT_LEDGER")
            else:
                need(not markers and capture is None and failure is None and state["forward_attempts"] == 0
                     and state["forward_returned"] is False and state["input_started"] is False, "EXPLICIT_NOT_STARTED_TEXT_LEDGER")
            if state["forward_returned"] and "FORWARD_RETURNED" not in markers:
                need(state["status"] == "failed" and failure is not None, "RETURN_WITHOUT_RECEIPT_REQUIRES_FAILED_TERMINAL_PROOF")
        need(child.get("generation_status") == "success" or not markers and capture is None, "FAILED_SOURCE_NOT_OBSERVED")
        rows.append({"id": child["id"], "parent_id": child["parent_id"],
            "status": state["status"] if state is not None else "missing_terminal",
            "raw_text": capture["raw_text"] if capture is not None and state is not None and state["status"] == "complete" else None,
            "capture_sha256": sha(regular(folder / "CAPTURE.json")) if capture is not None else None,
            "reason": state.get("failure") if state is not None else "MISSING_PROVIDER_TERMINAL",
            "retained_forward_attempts": int("FORWARD_STARTED" in markers),
            "retained_forward_returns": int("FORWARD_RETURNED" in markers), "capture_retained": capture is not None})
    if terminal is not None:
        calls = sum(r["forward_attempts"] for r in ledger)
        returns = sum(r["forward_returned"] for r in ledger)
        complete_count = sum(r["capture_saved"] for r in ledger)
        need(terminal.get("actual_forward_attempts") == calls and terminal.get("native_forward_returns") == returns
             and terminal.get("completed") == complete_count and calls <= 64
             and terminal.get("input_started") == sum(r["input_started"] for r in ledger)
             and terminal.get("failed") == sum(r["status"] == "failed" for r in ledger)
             and terminal.get("not_started") == sum(r["status"] == "not_started" for r in ledger), "RECOUNTED_TEXT_TOTALS")
        if calls:
            lock = read(root / "MODEL_LOCK.json")
            need(lock.get("run") == run_identity and lock.get("provider") == provider
                 and lock.get("model") == plan["text_observers"][provider]
                 and lock.get("identity") == text_identity(provider, plan)
                 and lock.get("worker_sha256") == plan["worker_sha256"]
                 and lock.get("input_scope") == "exact_child_16k_pcm"
                 and lock.get("loader_attempts") == lock.get("loader_returns") == 1
                 and lock.get("native_model_construction_count_claimed") is False
                 and lock.get("actual_runtime_lock") == {"verified_pinned_loader": "english_seven_voice_asr_once._load",
                     "provider": provider, "source_sha256": plan["helper_sha256"]["automation/english_seven_voice_asr_once.py"]},
                 "ACTUAL_TEXT_LOADER_LOCK")
            loading = read(root / "MODEL_LOAD_STARTED.json")
            need(loading.get("run") == run_identity and loading.get("provider") == provider
                 and loading.get("loader_attempts") == 1 and loading.get("high_level_decode_attempts") == 0,
                 "LOADER_STARTED_BEFORE_CHILD_DECODE")
        complete = complete_count == 64 and terminal.get("failure") is None
        need(terminal.get("status") == ("COMPLETE_DIAGNOSTIC" if complete else "INCOMPLETE_DIAGNOSTIC_NO_RETRY"), "TEXT_STATUS_FOLLOWS_COUNTS")
        lower = upper = calls
    else:
        calls = returns = complete_count = None
        complete = False
        lower, upper = retained_calls, 64
    return {"schema": TEXT_SCHEMA + "/readback", "provider": provider, "complete": complete,
        "actual_forward_attempts": calls, "native_forward_returns": returns, "completed": complete_count,
        "forward_attempts_lower_bound": lower, "forward_attempts_upper_bound": upper,
        "retained_native_forward_returns_lower_bound": retained_returns,
        "retained_captures_lower_bound": retained_captures, "rows": rows,
        "input_scope": "exact_child_16k_pcm", "counter_scope": "high_level_one_input_decode"}


def generate_stage(plan, output_root, asset_root):
    run = execution_guard(plan)
    output = new_directory(output_root)
    once(output / "EXECUTION_PLAN.json", plan)
    retain_sources(output, plan)
    generator = importlib.import_module("english_kokoro_context64_generate")
    # The generator owns its one 1800-second alarm and its complete denominator.
    try:
        generated = generator.generate(plan, output, run=run, asset_root=Path(asset_root).absolute())
        manifest = read_generation(output, plan, allow_partial=True)
        need(generated == read(output / "source/RESULT.json"), "GENERATOR_RETURN_IS_RETAINED_RESULT")
        once(output / "INPUT_MANIFEST.json", manifest)
        result = {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run, "stage": "generate",
            "status": "PARENT_CHILD_SOURCE_PACKET_RETAINED", "generation_summary": manifest["generation_summary"],
            "source_integrity_passed": manifest["source_integrity_passed"],
            "source_result_sha256": manifest["source_result_sha256"],
            "declared_parents": 48, "declared_children": 64, "training_allowed": False,
            "training_snapshot_allowed": False, "permanently_excluded_from_training": True,
            "automatic_expansion": False, "retry_count": 0}
        once(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        once(output / "SOURCE_PACKET_INCOMPLETE.json", {"schema": SCHEMA, "run": run,
            "status": "UNVERIFIED_PARTIAL_SOURCE", "source_ready": False, "observer_ready": False,
            "error": type(exc).__name__ + ": " + str(exc), "actual_forward_attempts": None,
            "declared_parents": 48, "declared_children": 64, "maximum_high_level_calls": 336,
            "training_allowed": False, "automatic_expansion": False})
        raise
    finally:
        once(output / "FILES.json", file_manifest(output))


def observe_stage(plan, provider, source_root, output_root, asset_root):
    run = execution_guard(plan)
    need(provider in PROVIDERS, "FIVE_FIXED_OBSERVERS")
    manifest = source_readback(source_root, plan, allow_partial=True)
    need(manifest["run"] == run and manifest["source_integrity_passed"] is True, "COMPLETE_NEW_SOURCE_BEFORE_OBSERVATIONS")
    output, assets = Path(output_root).absolute(), Path(asset_root).absolute()
    need(output.parent.is_dir() and assets.parent.is_dir() and not output.exists() and not assets.exists()
         and output != assets and output not in assets.parents and assets not in output.parents,
         "NEW_SEPARATE_OBSERVER_AND_PRIVATE_ASSET_DIRECTORIES")
    set_deadline(plan["text_deadline_seconds"] if provider in TEXT_PROVIDERS else plan["phone_deadline_seconds"])
    try:
        if provider in TEXT_PROVIDERS:
            result = observe_text(provider, Path(source_root), output, assets, plan, run, manifest)
        else:
            phone = importlib.import_module("english_kokoro_context64_phone")
            result = phone.run(provider=provider, source_root=Path(source_root), output_root=output,
                asset_root=assets, plan=plan, run_identity=run, parent_rows=manifest["parents"])
    finally:
        signal.alarm(0)
        try:
            if assets.exists():
                need(assets.is_dir() and not assets.is_symlink(), "PRIVATE_OBSERVER_ASSET_CLEANUP")
                shutil.rmtree(assets)
        finally:
            if output.is_dir() and not (output / "FILES.json").exists():
                once(output / "FILES.json", file_manifest(output))
    return result


def copy_tree(source, target):
    source, target = Path(source), Path(target)
    need(source.is_dir() and not source.is_symlink(), "COPY_ONLY_SOURCE_DIRECTORY")
    for path in sorted(source.rglob("*")):
        need(not path.is_symlink(), "NO_COPIED_SYMLINK")
        if path.is_file():
            put(within(target, path.relative_to(source).as_posix()), regular(path, allow_empty=True))


def combine(plan, source_root, observers_root, output_root):
    run = execution_guard(plan)
    manifest = source_readback(source_root, plan, allow_partial=True)
    need(manifest["run"] == run, "SAME_NEW_RUN_SOURCE_FOR_COMBINE")
    output = new_directory(output_root)
    copy_tree(source_root, output / "source-packet")
    provider_roots = {}
    for provider in PROVIDERS:
        source = Path(observers_root) / ("kokoro-context64-observer-" + provider)
        target = output / "observers" / provider
        provider_roots[provider] = target
        if source.exists():
            copy_tree(source, target)
    set_deadline(plan["combine_deadline_seconds"])
    try:
        evaluator = importlib.import_module("english_kokoro_context64_evaluate")
        evaluation = evaluator.evaluate(output / "source-packet", provider_roots, plan, run)
        once(output / "EVALUATION.json", evaluation)
        result = {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "run": run, "stage": "combine",
            "status": "DIAGNOSTIC_PACKET_RETAINED", "evaluation_status": evaluation["status"],
            "counts": evaluation["counts"], "gates": evaluation["gates"],
            "context_contract_control_test_passed": evaluation["context_contract_control_test_passed"],
            "evaluation_sha256": sha(regular(output / "EVALUATION.json")),
            "declared_parents": 48, "declared_children": 64, "maximum_high_level_calls": 336,
            "retry_count": 0, "training_allowed": False, "training_snapshot_allowed": False,
            "permanently_excluded_from_training": True, "automatic_expansion": False,
            "old_labels_modified": False, "old_data_re_admitted": False, "high_quality_model_passed": False,
            "child_independent_phone_silver": False, "new_model_calls_during_combine": 0}
        once(output / "RESULT.json", result)
        once(output / "EXECUTION_PLAN.json", plan)
        return result
    except BaseException as exc:
        once(output / "EVIDENCE_INTEGRITY_FAILED.json", {"schema": SCHEMA, "run": run,
            "error": type(exc).__name__ + ": " + str(exc), "training_allowed": False,
            "automatic_expansion": False, "new_model_calls_during_combine": 0})
        raise
    finally:
        signal.alarm(0)
        once(output / "FILES.json", file_manifest(output))


def source_check(plan):
    frozen = validate_plan(plan)
    generator = importlib.import_module("english_kokoro_context64_generate")
    phone = importlib.import_module("english_kokoro_context64_phone")
    evaluator = importlib.import_module("english_kokoro_context64_evaluate")
    generator.source_preflight(plan)
    generator.validate_plan(plan)
    phone.source_preflight(plan)
    evaluator.source_preflight()
    need(not any(name in sys.modules for name in ("torch", "transformers", "numpy", "scipy", "kokoro", "onnxruntime")),
         "SOURCE_ONLY_NO_NUMERIC_OR_MODEL_IMPORT")
    return {"schema": SCHEMA, "source_check": "PASS", "enabled": plan["enabled"],
        "declared_parents": len(frozen["parents"]), "declared_children": len(frozen["children"]),
        "maximum_high_level_calls": 336, "helper_files": 30, "new_model_calls": 0,
        "new_processor_calls": 0, "new_resampling_calls": 0, "new_network_calls": 0,
        "training_allowed": False}


def source_status(root, plan):
    """Preserve unknown counts when an interrupted source cannot be verified."""
    try:
        manifest = source_readback(root, plan, allow_partial=True)
        return {"schema": SCHEMA, "status": "VERIFIED_SOURCE_PACKET", "source_ready": True,
            "observer_ready": manifest["source_integrity_passed"], "run": manifest["run"],
            "generation_summary": manifest["generation_summary"],
            "valid_parent_count": sum(r["generation_status"] == "success" for r in manifest["parents"]),
            "valid_child_count": sum(r["generation_status"] == "success" for r in manifest["children"]),
            "training_allowed": False, "new_model_calls": 0, "new_resampling_calls": 0}, True
    except Exception as exc:
        return {"schema": SCHEMA, "status": "SOURCE_READBACK_FAILED_COUNTS_UNKNOWN",
            "source_ready": False, "observer_ready": False,
            "error": type(exc).__name__ + ": " + str(exc),
            "declared_parents": 48, "declared_children": 64, "maximum_high_level_calls": 336,
            "generation_summary": {"declared_count": 48, "actual_forward_attempts": None,
                "native_forward_returns": None, "completed": None, "complete": False,
                "forward_attempts_lower_bound": 0, "forward_attempts_upper_bound": 48,
                "bounds_basis": "unknown_unverified_source_not_an_actual_zero_claim"},
            "valid_parent_count": None, "valid_child_count": None,
            "training_allowed": False, "automatic_expansion": False,
            "new_model_calls_during_readback": 0, "new_resampling_calls_during_readback": 0}, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source-check", action="store_true")
    action.add_argument("--guard", action="store_true")
    action.add_argument("--generate", action="store_true")
    action.add_argument("--read-source", action="store_true")
    action.add_argument("--observe", choices=PROVIDERS)
    action.add_argument("--combine", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--observers")
    parser.add_argument("--output")
    parser.add_argument("--assets")
    args = parser.parse_args()
    plan = read(Path(args.plan))
    if args.source_check:
        result = source_check(plan)
    elif args.guard:
        result = {"schema": SCHEMA, "guard": "PASS", "run": execution_guard(plan), "new_model_calls": 0}
    elif args.generate:
        need(args.output is not None and args.assets is not None, "GENERATION_PATHS_REQUIRED")
        result = generate_stage(plan, args.output, args.assets)
    elif args.read_source:
        need(args.source is not None, "SOURCE_PATH_REQUIRED")
        result, verified = source_status(args.source, plan)
        if args.output:
            once(Path(args.output), result)
    elif args.observe:
        need(all((args.source, args.output, args.assets)), "OBSERVER_PATHS_REQUIRED")
        result = observe_stage(plan, args.observe, args.source, args.output, args.assets)
    else:
        need(all((args.source, args.observers, args.output)), "COMBINE_PATHS_REQUIRED")
        result = combine(plan, args.source, args.observers, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if args.read_source and not verified:
        raise SystemExit(1)
    if args.generate and result["source_integrity_passed"] is not True:
        raise SystemExit(1)
    if args.observe and result.get("status") != "COMPLETE_DIAGNOSTIC":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

