"""Reopen complete original B28 engineering evidence without executing models.

The reviewed engineering lock is required before any input is opened. This
stdlib reader authenticates the original four ZIPs and Git/API provenance,
repeats both complete quality readers, then checks every saved engineering
array, event row, original export and installed SDK byte. It does not fetch,
compile, infer, retry, select a candidate, or write a model pack.
"""
from __future__ import annotations
import argparse
import builtins
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from unittest import mock

REPO = "jiying2007/kws-data"
PIPELINE_REPO = "jiying2007/kws-pipeline"
PIPELINE_COMMIT = "88a043c6c00bb3a1a62f45098a023c7601042ded"
ENGINEERING_SHA = "453b141952066b148d68ee8ca3605118dc28c44afd4633a850be8509cfe84999"
QUALITY_READER_SHA = None  # Future reviewed B28 reader, never the old B27 allowance.
PIPELINE_DESCRIPTOR_SHA = "0db8c6f8807d62b6aabf1e0c4ce32c043e895d3ae08976e87a8079ed978b2051"
SCHEMA = "english-v6-b28-verified-original-engineering-v1"
LOCK_SCHEMA = "english-v6-b28-original-engineering-lock-v1"
STATUS = "VERIFIED_ORIGINAL_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE"
LIMIT = 128_000_000
REPORT_LIMIT = 4_000_000
ROLES = {"result", "producer", "consumed", "input_proof"}
API_ROLES = {"run", "commit", "tree", "artifacts", "runs", "markers", "pipeline_commit", "pipeline_tree"}
HASH_FIELDS = {"evaluator_sha256", "inputs_sha256", "producer_manifest_sha256",
               "original_result_sha256", "result_manifest_sha256", "quality_reader_sha256"}
WORKFLOW = ".github/workflows/english-v6-b28-engineering-once.yml"
BRANCH = "research/en-v6-b28-engineering-once-20261125"
MARKER = "english-v6-b28-engineering-consumed"
BACKENDS = ("pytorch", "onnxruntime", "gcc", "clang", "cortex-a32-qemu")
CONDITIONS = ("original", "one_second_zero_prefix")
PACK_MEMBER = "models/b28-challenger.pack"
B0_PACK_SHA = "ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b"
SCOPE = dict(use="research_software_integration", physical_ssc305=False,
    product_promotion_allowed=False, cal_locked_access=False, field_frr=None,
    continuous_far_per_hour=None, word_end_p95_ms=None, product_decoder_settings=None)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def equal(left, right):
    return canonical(left) == canonical(right)


def document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "no duplicate JSON fields")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite original JSON " + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    need(type(value) is dict, "complete original JSON object")
    return value


def valid_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def byte_pin(pin, empty=False):
    need(type(pin) is dict and set(pin) == {"bytes", "sha256"}
         and type(pin["bytes"]) is int and (0 if empty else 1) <= pin["bytes"] <= LIMIT
         and valid_hash(pin["sha256"]), "exact bounded original bytes and digest")


def artifact_pin(pin):
    need(type(pin) is dict and set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and type(pin["id"]) is type(pin["run_id"]) is int and pin["id"] > 0 and pin["run_id"] > 0
         and valid_hash(pin["run_head"], 40), "actual original artifact identity")
    byte_pin({key: pin[key] for key in ("bytes", "sha256")})


def validate_lock(lock):
    need(valid_hash(QUALITY_READER_SHA),
         'INERT: actual reviewed B28 quality reader has not been bound for engineering readback')
    need(type(lock) is dict and set(lock) == {"schema", "repository", "artifacts", "api"} | HASH_FIELDS
         and lock["schema"] == LOCK_SCHEMA and lock["repository"] == REPO,
         "INERT: complete reviewed original engineering lock required")
    need(type(lock["artifacts"]) is dict and set(lock["artifacts"]) == ROLES
         and type(lock["api"]) is dict and set(lock["api"]) == API_ROLES,
         "all four original ZIPs and eight exact saved API responses")
    for pin in lock["artifacts"].values():
        artifact_pin(pin)
    for pin in lock["api"].values():
        byte_pin(pin)
    pins = lock["artifacts"].values()
    need(len({p["id"] for p in pins}) == 4 and len({(p["run_id"], p["run_head"]) for p in pins}) == 1,
         "four distinct original artifacts from one engineering execution")
    need(all(valid_hash(lock[key]) for key in HASH_FIELDS)
         and lock["evaluator_sha256"] == ENGINEERING_SHA
         and lock["quality_reader_sha256"] == QUALITY_READER_SHA,
         "unchanged reviewed engineering and complete quality readers")


def read(path, maximum=LIMIT):
    path = Path(path).absolute()
    need(all(not p.is_symlink() for p in (path, *path.parents)) and path.is_file()
         and stat.S_ISREG(path.stat().st_mode) and path.stat().st_size <= maximum,
         "bounded regular saved evidence without symlink ancestors")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    need(len(raw) <= maximum, "saved evidence grew past its byte limit")
    return raw


def matches(raw, pin):
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact original byte identity")


def safe_name(name):
    need(type(name) is str and 0 < len(name.encode("utf-8")) <= 1024
         and "\\" not in name and "\0" not in name, "bounded original archive path")
    path = PurePosixPath(name)
    need(path.parts and not path.is_absolute() and path.as_posix() == name
         and all(part not in ("", ".", "..") for part in name.split("/")), "canonical relative original path")
    return name


def opened(raw):
    need(type(raw) is bytes and 0 < len(raw) <= LIMIT, "complete bounded original ZIP")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        members = archive.infolist()
        need(len(members) <= 8192 and len(members) == len({item.filename for item in members})
             and sum(item.file_size for item in members) <= LIMIT, "unique complete bounded ZIP inventory")
        for item in members:
            safe_name(item.filename)
            need(not item.is_dir() and not (item.flag_bits & 1)
                 and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
                 and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                 "regular unencrypted original ZIP members only")
        need(archive.testzip() is None, "all original ZIP member CRCs")
        return archive
    except BaseException:
        archive.close()
        raise


def original(path, pin):
    raw = read(path)
    matches(raw, pin)
    return opened(raw)


def tree_members(tree, expected):
    need(tree["sha"] == expected and tree["truncated"] is False and type(tree["tree"]) is list,
         "complete original recursive Git tree")
    need(len(tree["tree"]) == len({v["path"] for v in tree["tree"]}), "unique original Git tree paths")
    result = {}
    for item in tree["tree"]:
        safe_name(item["path"])
        if item["type"] == "blob":
            result[item["path"]] = item
    return result


def validate_artifact(pin, run, artifact, name=None):
    artifact_pin(pin)
    need(run["id"] == pin["run_id"] and run["head_sha"] == pin["run_head"]
         and run["status"] == "completed" and run["conclusion"] == "success"
         and type(run["run_attempt"]) is int and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and artifact["id"] == pin["id"] and artifact["expired"] is False
         and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
         and artifact["workflow_run"]["id"] == pin["run_id"]
         and artifact["workflow_run"]["head_sha"] == pin["run_head"]
         and (name is None or artifact["name"] == name), "original successful public first-attempt artifact")


def api_metadata(directory, lock):
    docs = {}
    for role, pin in lock["api"].items():
        raw = read(directory / "api" / (role + ".json")); matches(raw, pin)
        docs[role] = document(raw)
    run, commit = docs["run"], docs["commit"]
    pin = lock["artifacts"]["result"]
    need(commit["sha"] == pin["run_head"] and len(commit["parents"]) == 1
         and valid_hash(commit["parents"][0]["sha"], 40)
         and [f["filename"] for f in commit["files"]] == [WORKFLOW]
         and run["head_branch"] == BRANCH and run["path"] == WORKFLOW,
         "one reviewed parent and exact workflow-only original activation")
    identity = dict(run_id=pin["run_id"], run_head=pin["run_head"], run_attempt=1,
                    prepared_parent=commit["parents"][0]["sha"])
    need(identity["prepared_parent"] != identity["run_head"], "distinct original activation and prepared parent")
    names = dict(result="english-v6-b28-software-sdk-evidence-" + str(pin["run_id"]),
        producer="english-v6-b28-engineering-producer-" + str(pin["run_id"]),
        consumed=MARKER, input_proof="english-v6-b28-engineering-input-proof-" + str(pin["run_id"]))
    artifacts = docs["artifacts"]
    need(artifacts["total_count"] == len(artifacts["artifacts"]) == 4
         and {v["id"] for v in artifacts["artifacts"]} == {p["id"] for p in lock["artifacts"].values()},
         "all four and only original engineering artifacts")
    for role, item in lock["artifacts"].items():
        actual = next(a for a in artifacts["artifacts"] if a["id"] == item["id"])
        validate_artifact(item, run, actual, names[role])
    runs, markers = docs["runs"], docs["markers"]
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1
         and runs["workflow_runs"][0]["id"] == pin["run_id"]
         and runs["workflow_runs"][0]["head_sha"] == pin["run_head"]
         and markers["total_count"] == len(markers["artifacts"]) == 1
         and equal(markers["artifacts"][0], next(a for a in artifacts["artifacts"] if a["id"] == lock["artifacts"]["consumed"]["id"])),
         "one engineering execution and its one consumed marker, never a rerun")
    tracked = tree_members(docs["tree"], commit["commit"]["tree"]["sha"])
    need(docs["pipeline_commit"]["sha"] == PIPELINE_COMMIT, "immutable frozen pipeline commit")
    pipeline = tree_members(docs["pipeline_tree"], docs["pipeline_commit"]["commit"]["tree"]["sha"])
    return docs, identity, tracked, pipeline


def retained_api_metadata(directory, lock, input_proof):
    need(set(input_proof.namelist()) == {"FETCH.json", "CONSUMPTION.json", "api/runs.json", "api/markers.json"},
         "complete original input proof with both raw execution API responses")
    for role in ("runs", "markers"):
        raw = input_proof.read("api/" + role + ".json")
        matches(raw, lock["api"][role])
        need(raw == read(directory / "api" / (role + ".json")),
             "saved API must be copied byte-for-byte from the original engineering input proof: " + role)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def producer_files(archive, lock, identity, metadata, tracked, pipeline_tree, output):
    raw = archive.read("PRODUCER.json")
    need(sha(raw) == lock["producer_manifest_sha256"], "exact original engineering producer manifest")
    value = document(raw)
    need(value["schema"] == "english-v6-b28-dual-pass-software-sdk-v1-producer"
         and all(type(value[k]) is type(v) and value[k] == v for k, v in identity.items())
         and value["git_tree"] == metadata["tree"]["sha"] and value["pipeline_commit"] == PIPELINE_COMMIT
         and value["include_hidden_files"] is True and value["new_model_forwards"] == value["packs_created"] == 0
         and value["closure_rule"] == "complete_tracked_automation_and_workflows_plus_both_original_quality_producers",
         "original complete producer captured before engineering")
    expected = {"PRODUCER.json"}
    for label, files, tree in (("repository", value["files"], tracked), ("pipeline", value["pipeline_files"], pipeline_tree)):
        need(type(files) is dict and bool(files), "complete original producer file map")
        for name, pin in files.items():
            safe_name(name)
            need(set(pin) == {"mode", "git_blob_sha1", "bytes", "sha256"} if label == "repository"
                 else set(pin) == {"git_blob_sha1", "bytes", "sha256"}, "exact original producer byte pin")
            byte_pin({key: pin[key] for key in ("bytes", "sha256")}, empty=True)
            member = label + "/" + name; raw = archive.read(member); matches(raw, pin)
            entry = tree.get(name, {})
            need(entry.get("type") == "blob" and entry.get("mode") in ("100644", "100755")
                 and entry.get("sha") == pin["git_blob_sha1"] == blob(raw)
                 and entry.get("size") == len(raw)
                 and (label != "repository" or pin["mode"] == entry["mode"]),
                 "original producer byte matches actual Git blob: " + member)
            expected.add(member)
    need(set(archive.namelist()) == expected, "all and only complete original producer files")
    archive.extractall(output)
    root = output / "repository"; automation = root / "automation"
    driver = automation / "english_v6_b28_engineering.py"
    reader = automation / "english_v6_b28_quality_readback.py"
    descriptor = automation / "english_v6_b24_engineering_pipeline.json"
    need(sha(driver.read_bytes()) == ENGINEERING_SHA and sha(reader.read_bytes()) == QUALITY_READER_SHA
         and sha(descriptor.read_bytes()) == PIPELINE_DESCRIPTOR_SHA
         and sha((automation / "english_v6_b28_engineering_inputs.json").read_bytes()) == lock["inputs_sha256"],
         "reviewed original engineering, quality reader, input registration and thirteen-file descriptor")
    registered = document(descriptor.read_bytes())
    need(registered["commit"] == PIPELINE_COMMIT and registered["repository"] == PIPELINE_REPO
         and len(registered["files"]) == 13 and set(registered["files"]) == set(value["pipeline_files"]),
         "all thirteen unchanged frozen pipeline source files")
    for name, pin in registered["files"].items():
        need(all(value["pipeline_files"][name][key] == item for key, item in pin.items()), "exact frozen pipeline file " + name)
    engineering = load(driver, "b28_original_engineering_readback_authority")
    quality = load(reader, "english_v6_b28_quality_readback")
    return value, engineering, quality


def npy_body(raw, shape):
    need(raw[:6] == b"\x93NUMPY" and tuple(raw[6:8]) in ((1, 0), (2, 0)), "complete original FP32 NPY")
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    need(0 < size <= 8192 and len(raw) >= 8 + width + size, "bounded complete original NPY header")
    header = ast.literal_eval(raw[8 + width:8 + width + size].decode("latin1").strip())
    need(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
         and header["descr"] == "<f4" and header["fortran_order"] is False
         and type(header["shape"]) is tuple and all(type(v) is int for v in header["shape"])
         and header["shape"] == shape, "exact original contiguous FP32 tensor geometry")
    body = raw[8 + width + size:]
    need(len(body) == 4 * math.prod(shape) and all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", body)),
         "every finite original tensor byte")
    return body


def pack_bytes(pack, npz, baseline_npz, geometry):
    need(len(pack) == 216960 and pack[:16] == struct.pack("<8sII", b"MERO43V1", 1, 54236),
         "exact original 216960-byte candidate pack header")
    with opened(npz) as arrays, opened(baseline_npz) as baseline:
        need(set(arrays.namelist()) == {name + ".npy" for name, shape in geometry}, "all and only original candidate tensors")
        offset = 16
        for name, shape in geometry:
            body = npy_body(arrays.read(name + ".npy"), shape)
            need(pack[offset:offset + len(body)] == body, "every candidate pack byte equals the original NPZ tensor: " + name)
            offset += len(body)
            if name in ("mel_filters", "train_mean", "train_scale"):
                need(body == npy_body(baseline.read(name + ".npy"), shape), "unchanged original B0 frontend: " + name)
            if name == "train_scale":
                need(all(v[0] > 0 for v in struct.iter_unpack("<f", body)), "unchanged positive train scale")
            if name == "mel_filters":
                values = [v[0] for v in struct.iter_unpack("<f", body)]
                need(all(0 <= v <= 1 for v in values) and all(sum(values[c::40]) > 0 for c in range(40)),
                     "unchanged complete original Mel bank")
        need(offset == len(pack), "no omitted or appended candidate pack tensor bytes")


def saved_arrays(archive, report, rows, waves, engineering):
    need(len(rows) == len({row["id"] for row in rows}) == 14 and len(report["records"]) == 28,
         "all fourteen original engineering inputs and twenty-eight conditions")
    expected_records, expected_raw, expected_attempts = [], set(), set()
    for row in rows:
        for condition in CONDITIONS:
            original_pcm = waves[row["id"]]
            signal = (b"\0" * 32000 if condition == CONDITIONS[1] else b"") + original_pcm
            need(len(signal) % 2 == 0 and 512 <= len(signal) // 2 < 1000000, "same bounded original PCM condition")
            frames = 1 + (len(signal) // 2 - 512) // 160
            case = row["id"] + "__" + condition
            reference = archive.read("raw_outputs/" + case + "__pytorch.f32")
            need(len(reference) == frames * 16, "complete original PyTorch frame count")
            floats = list(struct.iter_unpack("<4f", reference))
            need(all(math.isfinite(value) for frame in floats for value in frame), "finite complete PyTorch reference")
            expected = engineering.ctc([max(range(4), key=frame.__getitem__) for frame in floats])
            record = dict(id=row["id"], condition=condition, source_pcm_sha256=sha(original_pcm),
                derived_pcm_sha256=sha(signal), frames=frames, pytorch_sha256=sha(reference), **expected, comparisons={})
            for backend in BACKENDS:
                key = case + "__" + backend
                attempt = "attempts/" + key + ".json"; expected_attempts.add(attempt)
                need(equal(document(archive.read(attempt)), dict(key=key, backend=backend, status="consumed_unknown")),
                     "one original pre-call ledger record per saved return")
                member = "raw_outputs/" + key + ".f32"; expected_raw.add(member)
                raw = archive.read(member)
                need(len(raw) == frames * 16, "every original backend returned all frames")
                if backend == "pytorch":
                    continue
                maximum, mean = (.001, .0001) if backend == "onnxruntime" else (.005, .0005)
                comparison = engineering.compare_logits(reference, raw, maximum, mean)
                value = dict(sha256=sha(raw), **comparison)
                if backend in BACKENDS[2:]:
                    member = "raw_outputs/" + key + ".csv"; expected_raw.add(member)
                    events = archive.read(member)
                    engineering.validate_events(events, frames, expected)
                    value["event_csv_sha256"] = sha(events)
                    log = archive.read("logs/" + key + ".txt").decode("utf-8")
                    calls = re.findall(r"MERO43_C_PCM_STREAM_OK samples=([0-9]+) frames=([0-9]+) events=([0-9]+)", log)
                    need(calls == [(str(len(signal) // 2), str(frames), str(len(expected["event_frames"])))],
                         "same original complete C stream return and sample clock")
                record["comparisons"][backend] = value
            expected_records.append(record)
    need(equal(report["records"], expected_records), "recompute every saved numeric, token and event comparison")
    names = set(archive.namelist())
    need({n for n in names if n.startswith("raw_outputs/")} == expected_raw
         and {n for n in names if n.startswith("attempts/")} == expected_attempts
         and len(expected_attempts) == 140 and sum(n.endswith(".csv") for n in expected_raw) == 84,
         "exactly 140 raw arrays and 84 event CSVs; no omitted or extra attempts")


def installed_sdk(archive, pipeline):
    required = {"bin/heym43-live", "bin/heym43_host_cli", "lib/libheym43_stream.a",
        "include/heym43/mero43_stream.h", "include/heym43/mero_ctc.h",
        *("lib/cmake/HeyMero43/" + name for name in ("HeyMero43Config.cmake", "HeyMero43ConfigVersion.cmake",
                                                   "HeyMero43Targets.cmake", "HeyMero43Targets-release.cmake"))}
    for backend in BACKENDS[2:]:
        prefix = "sdk-" + backend + "/"
        need({name.removeprefix(prefix) for name in archive.namelist() if name.startswith(prefix)} == required,
             "complete exact installed SDK inventory for " + backend)
        for header in ("mero43_stream.h", "mero_ctc.h"):
            need(archive.read(prefix + "include/heym43/" + header) == read(pipeline / "research/english_v5_mero" / header),
                 "installed SDK header equals frozen pipeline source")
        need(archive.read(prefix + "lib/libheym43_stream.a").startswith(b"!<arch>\n"), "actual original static library archive")
        for program in ("heym43-live", "heym43_host_cli"):
            raw = archive.read(prefix + "bin/" + program)
            arm = backend == "cortex-a32-qemu"
            need(len(raw) >= 64 and raw[:4] == b"\x7fELF" and raw[4:7] == bytes((1 if arm else 2, 1, 1))
                 and int.from_bytes(raw[18:20], "little") == (40 if arm else 62), "actual original native/ARM ELF SDK executable")
            if arm:
                need(int.from_bytes(raw[36:40], "little") & 0x600 == 0x400, "original ARM hard-float executable")
        need(b"HeyMero43::stream" in archive.read(prefix + "lib/cmake/HeyMero43/HeyMero43Targets.cmake")
             and b"libheym43_stream.a" in archive.read(prefix + "lib/cmake/HeyMero43/HeyMero43Targets-release.cmake"),
             "installed CMake package exports its original static stream library")
        for label in ("configure", "build", "ctest", "install", "ctest-inventory", "consumer-configure", "consumer-build", "consumer-run"):
            archive.read("logs/" + backend + "-" + label + ".txt")
        need(b"Total Tests: 2" in archive.read("logs/" + backend + "-ctest-inventory.txt")
             and b"100% tests passed, 0 tests failed out of 2" in archive.read("logs/" + backend + "-ctest.txt"),
             "both frozen synthetic CTests completed in the original engineering run")
        if backend != "cortex-a32-qemu":
            archive.read("logs/" + backend + "-live-input.txt")
    need(all(word in archive.read("logs/arm-elf.txt").decode("utf-8")
             for word in ("ELF32", "ARM", "little endian", "hard-float ABI")), "saved original ARM ELF readback")
    need(archive.read("LICENSE") == read(pipeline / "LICENSE"), "exact frozen software license")


def result_manifest(archive, lock):
    raw = archive.read("MANIFEST.json")
    need(sha(raw) == lock["result_manifest_sha256"], "exact original engineering result manifest")
    manifest = document(raw)
    need(manifest["schema"] == "english-v6-b28-dual-pass-software-sdk-v1"
         and manifest["status"] == "PASS_B28_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE"
         and manifest["physical_ssc305"] is manifest["product_promotion_allowed"] is False
         and set(manifest["files"]) == set(archive.namelist()) - {"MANIFEST.json"}, "entire original engineering result inventory")
    for name, pin in manifest["files"].items():
        byte_pin(pin, empty=True); matches(archive.read(name), pin)
    raw = archive.read("RESULT.json")
    need(sha(raw) == lock["original_result_sha256"], "exact original engineering RESULT")
    return manifest, document(raw)


def result_identity(archive, report, engineering, doc, identity, quality, admission):
    started = dict(schema=engineering.SCHEMA, identity=identity, status="STARTED", candidate_files=doc["candidate_files"],
        pipeline_commit=PIPELINE_COMMIT, input_spec_sha256=sha(engineering.SPEC.read_bytes()), quality=quality,
        original_count=14, conditions=list(CONDITIONS), backends=list(BACKENDS), records=[], optimizer_updates=0,
        new_tts_asr_calls=0, independent_quality_claimed=False, same_job_matched_control=False,
        historical_control_retrained=False, model_comparison=engineering.MODEL_COMPARISON, physical_ssc305=False,
        field_frr=None, continuous_far_per_hour=None, word_end_p95_ms=None, product_promotion_allowed=False,
        product_decoder_settings=None, replay_allowed=False)
    need(equal(document(archive.read("STARTED.json")), started)
         and equal(document(archive.read("ADMISSION.json")), admission), "same original start and fully recomputed admission")
    expected = {**started, "status": "PASS_B28_DUAL_QUALITY_SOFTWARE_SDK_CANDIDATE", "records": report["records"],
        "pack_sha256": sha(archive.read(PACK_MEMBER)), "pack_bytes": 216960,
        "full_frame_tokens_and_event_sample_clocks_equal": True, "sdk_installed_and_consumed": True,
        "entered_backend_calls": 140, "saved_backend_outputs": 140, "consumed_unknown": [],
        "backend_calls": {backend: 28 for backend in BACKENDS}}
    need(equal(report, expected), "complete original successful engineering identity, scope and exact 140-call accounting")
    pack = dict(schema=engineering.SCHEMA, bytes=216960, sha256=expected["pack_sha256"], candidate_files=doc["candidate_files"],
                pair_pin=doc["inputs"]["pair"], layout="MERO43V1", quality_passes=2)
    need(equal(document(archive.read("PACK.json")), pack), "same original pack and original dual-quality candidate exports")


def offline_engineering(directory, locks, reader, training):
    """Allow two exact quality workers and one exact full TRAIN worker only.

    Each child keeps its own complete nested-original proof guard. These
    explicit roots confer no authority to models, compilers or other commands.
    """
    stack = ExitStack()
    directory = Path(directory).resolve()
    reader_path = Path(reader.__file__).resolve()
    train_path = reader_path.parent / "english_v6_b28_training_readback.py"
    need(sha(read(reader_path, REPORT_LIMIT)) == QUALITY_READER_SHA
         and sha(read(train_path, REPORT_LIMIT)) == training["training_reader_sha256"],
         "exact independent B28 quality and full TRAIN reader authorities")
    need(set(locks) == {"primary", "replication"}, "two exact locked top-level panels")
    for panel in locks:
        reader.validate_lock(locks[panel])
        need(locks[panel]["panel"] == panel, "panel-specific original lock")
    old_import, old_popen, old_socket = builtins.__import__, subprocess.Popen, socket.socket
    forbidden = {"numpy", "torch", "torchaudio", "onnxruntime", "soundfile", "transformers", "piper", "requests", "httpx"}
    entered = set()

    def import_guard(name, *args, **kwargs):
        need(name.split(".")[0] not in forbidden, "no acoustic/network runtime during engineering readback")
        return old_import(name, *args, **kwargs)

    def popen_guard(command, *args, **kwargs):
        need(type(command) is list and len(command) == 9
             and command[:4] == [sys.executable, "-I", "-S", "-B"]
             and command[5] == "--verify-worker" and not args
             and set(kwargs) <= {"stdin", "stdout", "stderr"},
             "only exact isolated original quality or TRAIN verification is permitted")
        path, evidence, lock_path, proof_root = (Path(command[index]) for index in (4, 6, 7, 8))
        need(all(value.is_absolute() and value == value.resolve() and not value.is_symlink()
                 for value in (path, evidence, lock_path, proof_root)), "only exact regular absolute worker inputs")
        raw = read(lock_path, REPORT_LIMIT)
        if path == reader_path:
            panel = evidence.name
            need(panel in locks and panel not in entered
                 and sha(read(path, REPORT_LIMIT)) == QUALITY_READER_SHA
                 and evidence == directory / "quality" / panel
                 and {p.name for p in evidence.iterdir()} == {"api", "originals", "inputs"}
                 and all((evidence / role).is_dir() and not (evidence / role).is_symlink()
                         for role in ("api", "originals", "inputs"))
                 and proof_root == directory / "proof_roots" / panel
                 and raw == canonical(locks[panel]), "one complete exact original quality panel and scoped proof")
            key = panel
        else:
            staged = reader.profile.training_child()
            need(path == train_path and "training" not in entered
                 and sha(read(path, REPORT_LIMIT)) == training["training_reader_sha256"]
                 and evidence == staged['directory'] == directory / "proof_roots/primary/training-origin/trained"
                 and proof_root == staged['inputs'] and reader.profile.inventory(proof_root) == staged['files']
                 and reader.profile.inventory(directory / 'proof_roots/primary/training-origin') == training['files']
                 and raw == canonical(training["training_lock"]),
                 "one exact complete71-original B28 TRAIN readback with byte-identical66+5 temporary inputs")
            key = "training"
        entered.add(key)
        return old_popen(command, **kwargs)

    def blocked_network(*args, **kwargs):
        raise ValueError("no network during original engineering verification")

    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs):
            blocked_network()

    stack.enter_context(mock.patch("builtins.__import__", side_effect=import_guard))
    stack.enter_context(mock.patch("subprocess.Popen", side_effect=popen_guard))
    stack.enter_context(mock.patch("socket.socket", BlockedSocket))
    stack.enter_context(mock.patch("socket.create_connection", side_effect=blocked_network))
    return stack


def verify_worker(directory, lock):
    validate_lock(lock)
    directory = Path(directory).absolute()
    metadata, identity, tracked, pipeline_tree = api_metadata(directory, lock)
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="b28-original-engineering-") as temporary:
        archives = {role: stack.enter_context(original(directory / "originals" / (role + ".zip"), pin))
                    for role, pin in lock["artifacts"].items()}
        retained_api_metadata(directory, lock, archives["input_proof"])
        producer_root = Path(temporary)
        producer, engineering, reader = producer_files(archives["producer"], lock, identity, metadata,
                                                       tracked, pipeline_tree, producer_root)
        # Read only the authenticated registration to bind the outer process guard.
        guard_doc = document(engineering.SPEC.read_bytes())
        with offline_engineering(directory, guard_doc["quality_locks"], reader, reader.profile.binding("primary")):
            engineering.evidence_capacity(producer_root)
            need(set(producer["files"]) == engineering.producer_paths(set(tracked)), "complete original dynamic engineering producer closure")
            doc = engineering.protocol()
            workflow = (engineering.ROOT / WORKFLOW).read_bytes()
            need(workflow == engineering.TEMPLATE.read_bytes().replace(engineering.PARENT_TOKEN.encode(), identity["prepared_parent"].encode()),
                 "original workflow differs only by its reviewed parent token")
            quality = {panel: reader.verify_quality_original(directory / "quality" / panel, doc["quality_locks"][panel],
                        directory / "proof_roots" / panel)
                       for panel in ("primary", "replication")}
            engineering.validate_dual_quality(quality, doc)
            need(set(archives["consumed"].namelist()) == {"CONSUMPTION.json"}
                 and archives["consumed"].read("CONSUMPTION.json") == archives["input_proof"].read("CONSUMPTION.json"),
                 "exact original pre-execution consumed marker retained twice")
            fetched = document(archives["input_proof"].read("FETCH.json"))
            need(fetched["schema"] == engineering.SCHEMA and equal(fetched["identity"], identity)
                 and fetched["input_spec_sha256"] == lock["inputs_sha256"]
                 and fetched["producer_manifest_sha256"] == lock["producer_manifest_sha256"]
                 and equal(fetched["quality"], quality) and set(fetched["provenance"]) == set(doc["inputs"]),
                 "original acquisition binds this producer and both complete quality readbacks")
            for role, pin in doc["inputs"].items():
                recorded = fetched["provenance"][role]
                validate_artifact(pin, recorded["run"], recorded["artifact"])
                with original(directory / "inputs" / (role + ".zip"), pin):
                    pass
            exports = engineering.original_pair(directory / "inputs", doc, directory / "proof_roots/primary")
            rows, waves, source_admission = engineering.engineering_sources(directory / "inputs", doc)
            admission = dict(schema=engineering.SCHEMA, identity=identity, input_spec_sha256=lock["inputs_sha256"],
                candidate_files=doc["candidate_files"], quality=quality, source_admission=source_admission,
                pcm_sha256={row["id"]: sha(waves[row["id"]]) for row in rows}, pipeline_commit=PIPELINE_COMMIT,
                original_count=14, maximum_backend_calls=140, optimizer_updates=0,
                same_job_matched_control=False, historical_control_retrained=False, model_comparison=engineering.MODEL_COMPARISON)
            need(equal(document(archives["consumed"].read("CONSUMPTION.json")),
                 dict(identity=identity, marker_name=MARKER, admission_sha256=sha(canonical(admission)))),
                 "one original consumed permit matches recomputed complete admission")
            result = archives["result"]
            manifest, report = result_manifest(result, lock)
            result_identity(result, report, engineering, doc, identity, quality, admission)
            saved_arrays(result, report, rows, waves, engineering)
            for suffix, raw in exports.items():
                need(result.read("models/" + engineering.STEM + "." + suffix) == raw, "unchanged original candidate export " + suffix)
            pipeline = producer_root / "pipeline"
            packer = load(pipeline / "research/english_v5_mero/pack_mero43.py", "b28_original_geometry_readback")
            with original(directory / "inputs/B0.zip", doc["inputs"]["B0"]) as baseline:
                baseline_npz = baseline.read("models/mero-fsmn-ctc-expanded256.npz")
                need(sha(baseline_npz) == engineering.B0_NPZ, "original immutable B0 frontend NPZ")
            pack_bytes(result.read(PACK_MEMBER), exports["npz"], baseline_npz, packer.geometry())
            installed_sdk(result, pipeline)
            models = {PACK_MEMBER, *("models/" + engineering.STEM + "." + suffix for suffix in ("pt", "npz", "onnx"))}
            need({name for name in result.namelist() if name.startswith("models/")} == models,
                 "one original candidate pack and exactly its three original exports")
            payload = {name for name in result.namelist() if name.startswith(("models/", "sdk-gcc/", "sdk-clang/", "sdk-cortex-a32-qemu/"))}
            payload |= {"LICENSE", "README.md"}
            files = {name: {**manifest["files"][name], "mode": 0o755 if re.fullmatch(r"sdk-(gcc|clang|cortex-a32-qemu)/bin/(heym43-live|heym43_host_cli)", name) else 0o644}
                     for name in sorted(payload)}
            files["models/B0.pack"] = dict(bytes=216960, sha256=B0_PACK_SHA, mode=0o644)
            need(sum(pin["bytes"] for pin in files.values()) <= LIMIT, "complete bounded software delivery payload")
            script = read(Path(__file__).resolve(), REPORT_LIMIT)
            return dict(schema=SCHEMA, status=STATUS, candidate_id="B28", candidate_pack=PACK_MEMBER,
                pipeline_commit=PIPELINE_COMMIT, files=files, quality_locks=doc["quality_locks"], quality=quality,
                engineering=dict(repository=REPO, artifacts=lock["artifacts"], result_sha256=lock["original_result_sha256"],
                                 manifest_sha256=lock["result_manifest_sha256"]),
                original_training_report_sha256=doc["original_training_report_sha256"],
                original_B24Q1_failure_report_sha256=engineering.Q1_FAILURE_REPORT_SHA,
                original_B27_failure_certification=doc['original_B27_failure_certification'],
                original_B27_failure_report_sha256=doc['original_B27_failure_report_sha256'],
                evidence_lock=lock, original_lock_sha256=sha(canonical(lock)), original_identity=identity,
                readback_verifier=dict(bytes=len(script), sha256=sha(script)),
                checks=dict(original_engineering_artifacts=4, raw_arrays=140, event_csvs=84, original_conditions=28,
                    complete_original_quality_passes=2, complete_original_TRAIN_verified=True,
                    complete_original_B27_failure_verified=True, natural_no_event_terms_recomputed=True,
                    natural_objective_returns_verified=384, external_training_originals=71,
                    replication_external_total_originals=89,
                    complete_original_Q1_failure_verified=True, external_originals_retained=True, original_producer_files=len(producer["files"]), frozen_pipeline_files=13,
                    original_pack_tensor_bytes_verified=True, unchanged_original_exports=True, installed_sdk_bytes_verified=True,
                    every_saved_array_and_event_recomputed=True, original_source_admission_recomputed=True,
                    original_api_and_git_provenance_verified=True),
                baseline_pack_requires_separate_byte_authentication=True,
                new_model_calls=0, new_source_calls=0, new_observer_calls=0, compiler_calls=0, packs_created=0,
                candidate_selection_changed=False, scope=SCOPE)


def verify(directory, lock_path, output):
    lock = document(read(lock_path)); validate_lock(lock)
    output = Path(output).absolute()
    need(not output.exists() and output.parent.is_dir() and all(not p.is_symlink() for p in (output, *output.parents)),
         "fresh regular report directory; original evidence remains unchanged")
    script = read(Path(__file__).resolve(), REPORT_LIMIT)
    with tempfile.TemporaryDirectory(prefix="b28-engineering-lock-") as temporary:
        saved_lock = Path(temporary) / "lock.json"; saved_lock.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                                       "--verify-worker", str(Path(directory).absolute()), str(saved_lock)])
    report = document(raw)
    need(len(raw) <= REPORT_LIMIT and report["schema"] == SCHEMA and report["status"] == STATUS
         and report["original_lock_sha256"] == sha(canonical(lock)) and equal(report["evidence_lock"], lock)
         and report["readback_verifier"] == dict(bytes=len(script), sha256=sha(script))
         and read(Path(__file__).resolve(), REPORT_LIMIT) == script, "actual isolated complete engineering readback")
    output.mkdir()
    (output / "completion.json").write_bytes(raw)
    (output / "readback.py").write_bytes(script)
    return report


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--verify-worker":
        lock = document(read(Path(sys.argv[3])))
        print(json.dumps(verify_worker(Path(sys.argv[2]), lock), sort_keys=True, allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    item = modes.add_parser("verify")
    item.add_argument("directory", type=Path); item.add_argument("lock", type=Path); item.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(dict(schema=SCHEMA, status="INERT_REQUIRES_LOCKED_ORIGINAL_ENGINEERING_EVIDENCE",
                              new_model_calls=0, compiler_calls=0, packs_created=0, candidate_selection_changed=False)))
    else:
        report = verify(args.directory, args.lock, args.output)
        print(json.dumps(dict(schema=SCHEMA, status=report["status"], original_lock_sha256=report["original_lock_sha256"],
                              completion_sha256=sha(read(args.output / "completion.json", REPORT_LIMIT)), new_model_calls=0)))


if __name__ == "__main__":
    main()
