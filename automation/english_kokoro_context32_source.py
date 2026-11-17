"""Derive all sixteen new full-carrier views from two immutable actual ZIPs.

This module never loads a speech model. Source inspection is standard-library
only; the sole numerical operation is the explicitly guarded public derivation.
The old cropped inputs and their decisions remain byte-for-byte retained.
"""
from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import sys
import tempfile
import wave
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "english-kokoro-context32-v1"
CAMPAIGN_ID = "kokoro-v1_0-original-carrier-full-context32-v1"
MAIN_BEFORE = "573e807c330f7b55b239220e9380e761bb95f42e"
DERIVATION = {"id": "scipy-resample-poly-2-3-pcm16-v1", "input_sample_rate": 24000,
    "input_dtype": "float32-little-endian", "output_sample_rate": 16000,
    "resampler": "scipy.signal.resample_poly", "up": 2, "down": 3,
    "window": ["kaiser", 5.0], "padtype": "constant", "cval": 0.0,
    "quantizer": "np.rint(np.clip(values,-1.0,32767/32768)*32768).astype('<i2')",
    "runtime_versions": {"numpy": "1.26.4", "scipy": "1.15.3"},
    "entire_native_input": True, "new_crop": False, "right_endpoint": "native_eof"}
PREVIOUS = {"repository": "jiying2007/kws-data", "run_id": 44125960085,
    "head_sha": MAIN_BEFORE, "workflow": ".github/workflows/english-kokoro48-once.yml",
    "recipe_sha256": "e104a64717ea195b5e7544285e07fa66441be24f3d7cdc9ea8493050cd715a83",
    "artifacts": {
        "source": {"id": 14347901645, "name": "kokoro48-source", "bytes": 59802210,
            "sha256": "eb6c5f6da584e2517915c517ed316cb45d4a207e9f863f949f5a110f730022e9"},
        "probe": {"id": 14348910717, "name": "english-kokoro48-probe-44125960085", "bytes": 60974804,
            "sha256": "3a0cf8fb1aeefa658a1418aa5eaa43e9b867caedfc3fbb09004df893b3ed7a66"}}}
LEGACY_PINS = {
    "automation/english_kokoro48_once.py": "8edd94fb7142fd0fe16ee9473f110b2fa23b9f973c74c5484eb6a6ff68fa2f29",
    "automation/english_kokoro48_recipe.py": "d010fff7f2144a672265d47367f07ab57732e5549a8c4ca7e9934a084151b369",
    "automation/english_kokoro48_generate.py": "77def484c22f26fe376fc3b3ae7e93b608cb748c913c46c93beeba46246b5654",
    "automation/english_kokoro48_runtime.py": "fb8da71bdd2460700a84224003ea9c15ae0fa2b89cf27d78d5b4008db1ee3b43",
    "automation/english_kokoro48_initialize.py": "9f4b65c505cba5afdea647d9df42311c1ada92318a43be9f9a83c2b27674f6c3"}
MAX_FILE = 128 * 1024**2
MAX_EXPANDED = 1024 * 1024**2
INPUT_FILES = ("source.zip", "source.artifact.json", "probe.zip", "probe.artifact.json", "run.json")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def regular(path, limit=MAX_FILE, allow_empty=False):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and (0 <= path.stat().st_size if allow_empty else 0 < path.stat().st_size)
         and path.stat().st_size <= limit, "REGULAR_BOUNDED_FILE")
    return path.read_bytes()


def within(root, name):
    need(type(name) is str and "\\" not in name and "\0" not in name, "SAFE_RELATIVE_PATH")
    rel = PurePosixPath(name)
    need(name == rel.as_posix() and rel.parts and not rel.is_absolute()
         and all(p not in ("", ".", "..") for p in name.split("/")), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK")
    return path


def once(path, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    return put(path, raw)


def put(path, raw):
    path = Path(path)
    need(type(raw) is bytes and 0 < len(raw) <= MAX_FILE and path.parent.is_dir()
         and not any(p.is_symlink() for p in (path, *path.parents)), "CREATE_BOUNDED_FILE")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "sha256": sha(raw)}


def new_directory(path):
    path = Path(path).absolute()
    need(path.parent.is_dir() and not path.exists()
         and not any(p.is_symlink() for p in (path, *path.parents)), "CREATE_ONLY_DIRECTORY")
    path.mkdir(mode=0o700)
    return path


def archive_bytes(raw, kind):
    expected = PREVIOUS["artifacts"][kind]
    need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "EXACT_OLD_ARCHIVE: " + kind)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        need(len(names) == len(set(names)) <= 5000 and sum(i.file_size for i in infos) <= MAX_EXPANDED,
             "ARCHIVE_BOUNDS")
        for info in infos:
            within(Path(tempfile.gettempdir()), info.filename)
            need(not info.is_dir() and not info.flag_bits & 1 and info.file_size <= 16 * 1024**2
                 and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)
                 and "private-model" not in info.filename.split("/"), "SAFE_ARCHIVE_MEMBER")
        files = {name: archive.read(name) for name in names}
    if kind == "probe":
        manifest = loads(files["FILES.json"])
        need(manifest.get("schema") == "english-kokoro48-files-v1", "OLD_PROBE_MANIFEST_SCHEMA")
        rows = manifest.get("files")
        need(type(rows) is list and len({r["path"] for r in rows}) == len(rows)
             and {r["path"] for r in rows} | {"FILES.json"} == set(files), "OLD_PROBE_CLOSED_SET")
        for row in rows:
            member = files[row["path"]]
            need(len(member) == row["bytes"] and sha(member) == row["sha256"], "OLD_PROBE_CLOSED_BYTES")
    return files


def legacy():
    for name, expected in LEGACY_PINS.items():
        need(sha(regular(ROOT / name, 256 * 1024)) == expected, "UNCHANGED_LEGACY_SOURCE: " + name)
    spec = importlib.util.spec_from_file_location("context32_immutable_source_reader", HERE / "english_kokoro48_once.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(inputs):
    run = loads(regular(inputs / "run.json"))
    need(run.get("id") == PREVIOUS["run_id"] and run.get("head_sha") == PREVIOUS["head_sha"]
         and run.get("path") == PREVIOUS["workflow"] and run.get("event") == "push"
         and run.get("run_attempt") == 1 and run.get("head_branch") == "main"
         and run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == PREVIOUS["repository"]
         and run["repository"].get("private") is False, "ACTUAL_OLD_PUBLIC_COMPLETED_RUN")
    for kind, expected in PREVIOUS["artifacts"].items():
        item = loads(regular(inputs / (kind + ".artifact.json")))
        need(item.get("id") == expected["id"] and item.get("name") == expected["name"]
             and item.get("size_in_bytes") == expected["bytes"]
             and item.get("digest") == "sha256:" + expected["sha256"] and item.get("expired") is False
             and item.get("workflow_run", {}).get("id") == PREVIOUS["run_id"]
             and item["workflow_run"].get("head_sha") == PREVIOUS["head_sha"]
             and item["workflow_run"].get("head_branch") == "main", "EXACT_OLD_ARTIFACT_API: " + kind)
    return {name: {"bytes": len(raw), "sha256": sha(raw)}
            for name in INPUT_FILES if not name.endswith(".zip") for raw in [regular(inputs / name)]}


def static_row(old):
    new_id = "enkctx32-" + sha(canonical({"source_id": old["id"],
        "native_full_f32_sha256": old["native_full_f32_sha256"], "derivation_id": DERIVATION["id"]}))[:24]
    start, end = old["crop_start_native_sample"], old["crop_end_native_sample"]
    need(start % 3 == 0 and 0 < start < end == old["native_samples"], "SAVED_CARRIER_INTEGER_PHASE")
    offset = start * 2 // 3
    return {"id": new_id, "source_id": old["id"], "source_request_sha256": old["request_sha256"],
        "preset_id": old["preset_id"], "mode": "full-carrier", "role": "train",
        "cell_id": old["cell_id"], "requested_class": old["requested_class"],
        "pronunciation_hypothesis": old["pronunciation_hypothesis"],
        "native_full_f32_path": old["native_full_f32_path"],
        "native_full_f32_sha256": old["native_full_f32_sha256"], "native_samples": old["native_samples"],
        "crop_start_native_sample": start, "crop_end_native_sample": end,
        "old_pcm_path": old["pcm_path"], "old_pcm_sha256": old["pcm_sha256"], "old_pcm_samples": old["pcm_samples"],
        "old_duration_sha256": old["duration_sha256"], "old_crop_sha256": old["crop_sha256"],
        "full_to_crop_offset_16k": offset, "conv_grid_offset_mod_320": offset % 320,
        "wav_path": "source/audio/" + new_id + "/full-16k.wav",
        "pcm_path": "source/audio/" + new_id + "/full-16k.pcm"}


def verified_original(inputs):
    """Read actual old bytes, with the old reader; never run any old stage."""
    inputs = Path(inputs)
    api = metadata(inputs)
    source = archive_bytes(regular(inputs / "source.zip"), "source")
    probe = archive_bytes(regular(inputs / "probe.zip"), "probe")
    old = legacy()
    plan = loads(source["EXECUTION_PLAN.json"])
    old.validate_plan(plan)
    with tempfile.TemporaryDirectory(prefix="context32-old-readback-") as folder:
        root = Path(folder)
        for name, raw in source.items():
            target = within(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            put(target, raw)
        corpus = old.source_readback(root, plan)
    expected_run = {"repository": PREVIOUS["repository"], "commit": MAIN_BEFORE,
        "run_id": str(PREVIOUS["run_id"]), "run_attempt": "1", "event": "push"}
    need(corpus.get("run") == expected_run and corpus.get("valid_count") == 48
         and corpus.get("declared_count") == 48 and corpus.get("recipe_sha256") == PREVIOUS["recipe_sha256"]
         and corpus == loads(probe["CORPUS_MANIFEST.json"]), "ACTUAL_OLD_48_SOURCE_PROBE_JOIN")
    for name, raw in source.items():
        dest = "generation/" + name if "/" not in name else name
        need(probe.get(dest) == raw, "OLD_SOURCE_RETAINED_IN_PROBE: " + name)
    result = loads(probe["RESULT.json"])
    analysis = loads(probe["PROBE_ANALYSIS.json"])
    need(result.get("run") == expected_run and result.get("total_actual_model_attempts") == 288
         and result.get("probe_decision") == "STOP_DIAGNOSTIC_NO_EXPANSION"
         and result.get("training_snapshot_allowed") is False
         and result.get("training_admitted") is False and result.get("automatic_expansion") is False
         and result.get("analysis_sha256") == sha(probe["PROBE_ANALYSIS.json"])
         and analysis.get("continuation", {}).get("status") == "STOP_DIAGNOSTIC_NO_EXPANSION", "OLD_STOP_RETAINED")
    old_hashes = sorted(r["pcm_sha256"] for r in corpus["rows"])
    need(len(old_hashes) == len(set(old_hashes)) == 48, "OLD48_EXCLUSION_FROM_ACTUAL_AUDIO")
    carriers = [r for r in corpus["rows"] if r["mode"] == "carrier"]
    need(len(carriers) == 16 and Counter(r["requested_class"] for r in carriers) == {1: 8, 2: 8}
         and Counter(r["preset_id"] for r in carriers) == {p: 4 for p in old.PRESETS}, "ALL16_CARRIER_ROSTER")
    return {"source": source, "probe": probe, "corpus": corpus, "carriers": carriers,
        "input_roster": [static_row(r) for r in carriers], "old_observed_pcm_sha256": old_hashes,
        "api_files": api, "old_plan_sha256": sha(source["EXECUTION_PLAN.json"]),
        "old_source_corpus_sha256": sha(canonical(corpus)), "old_stop_analysis_sha256": sha(probe["PROBE_ANALYSIS.json"])}


def pcm_values(raw):
    need(type(raw) is bytes and len(raw) % 2 == 0, "PCM16_BYTES")
    values = array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def difference(full_pcm, old_pcm, offset):
    need(type(offset) is int and offset >= 0 and len(full_pcm) == offset * 2 + len(old_pcm), "ALIGNED_SUFFIX_LENGTH")
    segment = full_pcm[offset * 2:]
    a, b = pcm_values(segment), pcm_values(old_pcm)
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    ranges = []
    for index in changed:
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    first, last = (changed[0], changed[-1]) if changed else (None, None)
    suffix_start = 0 if last is None else last + 1
    prefix_end = len(a) if first is None else first
    return {"old_crop_samples": len(b), "full_suffix_samples": len(a), "full_crop_offset_16k": offset,
        "integer_resampling_phase": True, "offset_mod_320": offset % 320,
        "differing_samples": len(changed), "first_differing_crop_index": first, "last_differing_crop_index": last,
        "differing_ranges_crop": ranges, "difference_ranges_are_half_open": True,
        "maximum_absolute_sample_difference": max((abs(x-y) for x, y in zip(a, b)), default=0),
        "sum_absolute_sample_difference": sum(abs(x-y) for x, y in zip(a, b)),
        "matching_prefix_crop_range": [0, prefix_end], "matching_suffix_crop_range": [suffix_start, len(b)],
        "matching_suffix_full_range": [offset + suffix_start, offset + len(b)],
        "matching_suffix_bytes_sha256": sha(old_pcm[suffix_start * 2:]),
        "entire_pcm_suffix_equal": segment == old_pcm,
        "frame_equivalence_claimed": False, "model_normalization_equivalence_claimed": False,
        "duration_boundary_is_acoustic_gold": False,
        "interpretation": "Sample comparison only. FIR left-boundary effects, convolution phase, and full-input normalization remain separate."}


def wav_bytes(pcm):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(pcm)
    return stream.getvalue()


def numerical_derivative(raw, np, resample_poly):
    values = np.frombuffer(raw, dtype="<f4")
    need(values.ndim == 1 and bool(np.isfinite(values).all()) and len(values) > 0, "NATIVE_FULL_FINITE_FLOAT32")
    derived = resample_poly(values, 2, 3, window=("kaiser", 5.0), padtype="constant", cval=0.0)
    need(derived.dtype == np.float32 and len(derived) == (len(values) * 2 + 2) // 3
         and bool(np.isfinite(derived).all()), "FIXED_FULL_RESAMPLING_GEOMETRY")
    return np.rint(np.clip(derived, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()


def prepare(plan, inputs, output, run):
    """Called only after the new worker's exact-public-successor guard."""
    need(plan.get("enabled") is True and plan.get("campaign_id") == CAMPAIGN_ID, "ENABLED_NEW_DERIVATION")
    original = verified_original(inputs)
    need(plan.get("input_roster") == original["input_roster"]
         and plan.get("old_observed_pcm_sha256") == original["old_observed_pcm_sha256"], "FROZEN_ACTUAL_ROSTER")
    output = new_directory(output)
    (output / "inputs").mkdir(mode=0o700)
    for name in INPUT_FILES:
        put(output / "inputs" / name, regular(Path(inputs) / name))
    once(output / "EXECUTION_PLAN.json", plan)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "prepare", "run": run, "declared_count": 16,
        "new_tts_calls": 0, "new_model_calls": 0, "source_material_reused": True, "retry_count": 0})
    rows = [dict(r, preparation_status="NOT_STARTED_NO_RETRY", pcm_sha256=None, wav_sha256=None,
        pcm_samples=None, sample_rate=16000, channels=1, sample_width=2) for r in original["input_roster"]]
    versions, failure, attempted = {}, None, 0
    try:
        versions = {name: importlib.metadata.version(name) for name in DERIVATION["runtime_versions"]}
        need(versions == DERIVATION["runtime_versions"], "EXACT_NUMPY_SCIPY_DERIVATION_RUNTIME")
        import numpy as np
        from scipy.signal import resample_poly
        observed = set(original["old_observed_pcm_sha256"])
        new_hashes = set()
        for row, old in zip(rows, original["carriers"]):
            folder = within(output, "source/audio/" + row["id"])
            folder.mkdir(mode=0o700, parents=True)
            once(folder / "STARTED.json", {"schema": SCHEMA, "stage": "derive-full-input", "run": run,
                "id": row["id"], "source_id": row["source_id"], "derivation": DERIVATION, "attempt": 1,
                "new_model_calls": 0, "retry_count": 0, "native_full_f32_sha256": row["native_full_f32_sha256"]})
            attempted += 1
            row["preparation_status"] = "STARTED"
            raw = original["source"][old["native_full_f32_path"]]
            need(len(raw) == row["native_samples"] * 4 and sha(raw) == row["native_full_f32_sha256"], "ORIGINAL_NATIVE_FULL_BYTES")
            pcm = numerical_derivative(raw, np, resample_poly)
            value = sha(pcm)
            need(value not in observed and value not in new_hashes, "FULL_INPUT_NOT_PREVIOUSLY_OBSERVED_OR_DUPLICATE")
            new_hashes.add(value)
            old_pcm = original["source"][old["pcm_path"]]
            diff = difference(pcm, old_pcm, row["full_to_crop_offset_16k"])
            pcm_pin = put(within(output, row["pcm_path"]), pcm)
            wav_pin = put(within(output, row["wav_path"]), wav_bytes(pcm))
            candidate_row = dict(row, preparation_status="COMPLETE", pcm_sha256=pcm_pin["sha256"],
                wav_sha256=wav_pin["sha256"], pcm_samples=len(pcm)//2)
            proof = {"schema": SCHEMA + "/derivation", "run": run, "row": candidate_row, "derivation": DERIVATION,
                "actual_runtime_versions": versions, "source_code_sha256": sha(regular(Path(__file__))),
                "original_source_artifact": PREVIOUS["artifacts"]["source"], "native_full_reused": True,
                "native_source_retained_in": "inputs/source.zip", "native_source_member": old["native_full_f32_path"],
                "all_native_samples_consumed": True, "old_crop_unchanged": True,
                "new_model_calls": 0, "retry_count": 0, "old_crop_comparison": diff}
            pin = once(folder / "DERIVATION.json", proof)
            row.update(candidate_row)
            row.update(derivation_path="source/audio/" + row["id"] + "/DERIVATION.json", derivation_sha256=pin["sha256"])
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
        for row in rows:
            if row["preparation_status"] == "STARTED":
                row["preparation_status"] = "FAILED_NO_RETRY"
                folder = within(output, "source/audio/" + row["id"])
                try:
                    once(folder / "FAILED.json", {"schema": SCHEMA, "run": run, "id": row["id"],
                        "failure": failure, "attempt": 1, "retry_count": 0, "new_model_calls": 0})
                except Exception as write_error:
                    row["failure_receipt_write_error"] = type(write_error).__name__ + ": " + str(write_error)[:500]
    manifest = {"schema": SCHEMA + "/input-manifest", "campaign_id": CAMPAIGN_ID, "run": run,
        "declared_count": 16, "rows": rows, "previous": PREVIOUS, "derivation": DERIVATION,
        "old_observed_pcm_sha256": original["old_observed_pcm_sha256"], "original_api_files": original["api_files"],
        "original_plan_sha256": original["old_plan_sha256"], "old_stop_analysis_sha256": original["old_stop_analysis_sha256"],
        "source_material_reused": True, "old_observed_crop_forwards": 0, "new_tts_calls": 0,
        "new_text_asr_calls": 0, "new_embedding_calls": 0, "new_training_calls": 0,
        "old_labels_modified": False, "training_allowed": False, "diagnostic_only": True, "retry_count": 0}
    pin = once(output / "INPUT_MANIFEST.json", manifest)
    completed = sum(r["preparation_status"] == "COMPLETE" for r in rows)
    result = {"schema": SCHEMA, "stage": "prepare", "run": run, "declared_count": 16,
        "status": "COMPLETE_NEW_INPUTS" if completed == 16 and failure is None else "INCOMPLETE_NO_RETRY",
        "actual_derivation_attempts": attempted, "new_derivative_inputs": completed, "failure": failure,
        "new_model_calls": 0, "new_tts_calls": 0, "retry_count": 0, "source_material_reused": True,
        "old_observed_crop_forwards": 0, "input_manifest_sha256": pin["sha256"], "training_allowed": False}
    once(output / "RESULT.json", result)
    return result


def readback(root, plan, *, allow_partial=False):
    """Recheck source/joins/geometry/saved comparisons without re-resampling."""
    root = Path(root)
    original = verified_original(root / "inputs")
    need(loads(regular(root / "EXECUTION_PLAN.json")) == plan, "NEW_PLAN_READBACK")
    manifest = loads(regular(root / "INPUT_MANIFEST.json"))
    result = loads(regular(root / "RESULT.json"))
    need(manifest.get("schema") == SCHEMA + "/input-manifest" and manifest.get("campaign_id") == CAMPAIGN_ID
         and manifest.get("derivation") == DERIVATION and manifest.get("previous") == PREVIOUS
         and manifest.get("old_observed_pcm_sha256") == original["old_observed_pcm_sha256"] == plan["old_observed_pcm_sha256"]
         and plan["input_roster"] == original["input_roster"] and len(manifest.get("rows", [])) == 16
         and manifest.get("original_api_files") == original["api_files"]
         and manifest.get("original_plan_sha256") == original["old_plan_sha256"]
         and manifest.get("old_stop_analysis_sha256") == original["old_stop_analysis_sha256"], "EXACT_MANIFEST_PROVENANCE")
    need(result.get("run") == manifest.get("run") == loads(regular(root / "STARTED.json")).get("run")
         and result.get("input_manifest_sha256") == sha(regular(root / "INPUT_MANIFEST.json"))
         and result.get("new_model_calls") == 0 and result.get("retry_count") == 0, "PREP_RESULT_BINDING")
    need(manifest.get("diagnostic_only") is True and manifest.get("source_material_reused") is True
         and manifest.get("training_allowed") is False and manifest.get("old_labels_modified") is False
         and all(type(manifest.get(k)) is int and manifest[k] == 0 for k in
            ("old_observed_crop_forwards", "new_tts_calls", "new_text_asr_calls", "new_embedding_calls", "new_training_calls", "retry_count"))
         and result.get("source_material_reused") is True and result.get("training_allowed") is False
         and result.get("old_observed_crop_forwards") == result.get("new_tts_calls") == 0,
         "UNCHANGED_DIAGNOSTIC_SCOPE")
    seen, completed, attempted = set(), 0, 0
    for row, fixed, old in zip(manifest["rows"], original["input_roster"], original["carriers"]):
        need(all(row.get(k) == v for k, v in fixed.items()), "ALL16_ORIGINAL_STATIC_IDENTITIES")
        folder = within(root, "source/audio/" + row["id"])
        start = folder / "STARTED.json"
        attempted += int(start.is_file())
        if row.get("preparation_status") != "COMPLETE":
            need(allow_partial and row.get("preparation_status") in ("NOT_STARTED_NO_RETRY", "FAILED_NO_RETRY"), "NO_FILTERED_INCOMPLETE_INPUT")
            continue
        need(start.is_file(), "DERIVATION_STARTED_BEFORE_SAVED_OUTPUT")
        begun = loads(regular(start))
        need(begun.get("id") == row["id"] and begun.get("run") == manifest["run"]
             and begun.get("derivation") == DERIVATION and begun.get("attempt") == 1
             and begun.get("retry_count") == begun.get("new_model_calls") == 0, "DERIVATION_START_BINDING")
        pcm = regular(within(root, row["pcm_path"]))
        wav = regular(within(root, row["wav_path"]))
        need(sha(pcm) == row["pcm_sha256"] and sha(wav) == row["wav_sha256"]
             and len(pcm) == row["pcm_samples"] * 2 and row["pcm_samples"] == (row["native_samples"] * 2 + 2)//3
             and (row.get("sample_rate"), row.get("channels"), row.get("sample_width")) == (16000, 1, 2), "DERIVED_PCM_HASH_GEOMETRY")
        need(wav == wav_bytes(pcm), "DERIVED_WAV_EXACT_PCM")
        need(row["pcm_sha256"] not in seen and row["pcm_sha256"] not in original["old_observed_pcm_sha256"], "NO_OLD_OR_DUPLICATE_PCM")
        seen.add(row["pcm_sha256"])
        proof_raw = regular(within(root, row["derivation_path"]))
        proof = loads(proof_raw)
        need(sha(proof_raw) == row["derivation_sha256"] and proof.get("derivation") == DERIVATION
             and proof.get("actual_runtime_versions") == DERIVATION["runtime_versions"]
             and proof.get("source_code_sha256") == plan["source_code_sha256"]
             and proof.get("run") == manifest["run"] and proof.get("new_model_calls") == proof.get("retry_count") == 0
             and proof.get("row") == {k:v for k,v in row.items() if k not in ("derivation_path", "derivation_sha256")}
             and proof.get("old_crop_comparison") == difference(pcm, original["source"][old["pcm_path"]], row["full_to_crop_offset_16k"]),
             "DERIVATION_RECEIPT_AND_SAMPLE_READBACK")
        completed += 1
    need(result.get("declared_count") == 16 and result.get("actual_derivation_attempts") == attempted
         and result.get("new_derivative_inputs") == completed, "FULL_DENOMINATOR_PREP_COUNTS")
    if not allow_partial:
        need(completed == attempted == 16 and result.get("status") == "COMPLETE_NEW_INPUTS" and result.get("failure") is None, "ALL16_NEW_INPUTS_REQUIRED")
    return manifest
