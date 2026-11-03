#!/usr/bin/env python3
"""Read-only private small-pilot preflight; never invokes TTS, ASR or training.

Checks real local source/reference bytes, declared rights records, model files,
three ASR adapter executables/weights, and exact FIRST-N scheduling coverage.
A successful preflight is technical source consistency, NOT permission to clone,
label authenticity, independent human qualification, or model deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import bulk_pool as pool  # noqa: E402
import bulk_runner as tts  # noqa: E402
import receipt_contract as receipt  # noqa: E402
import runner  # noqa: E402

ASR_ASSETS_SCHEMA = "kws-private-native-asr-weight-inventory-v1"
ASR_DIGEST_SCHEMA = "kws-native-asr-model-asset-lock-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_INVENTORY_BYTES = 1024 * 1024
MAX_MODEL_FILES = 256
MAX_SINGLE_FILE = 8 * 1024**3
MAX_PER_MODEL = 12 * 1024**3
MAX_SCAN_ENTRIES = 768
MAX_PILOT_TTS = 64


def need(value, why):
    if not value:
        raise ValueError(why)


def sha_file(path, limit):
    st = path.stat()
    need(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
         and 0 < st.st_size <= limit, "MODEL_FILE_TYPE_OR_SIZE")
    h = hashlib.sha256()
    length = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            length += len(chunk)
            need(length <= limit, "MODEL_STREAM_BUDGET")
            h.update(chunk)
        end = os.fstat(f.fileno())
    need(length == st.st_size and
         (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) ==
         (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns, end.st_ctime_ns),
         "MODEL_CHANGED_DURING_READ")
    return length, h.hexdigest()


def rooted_regular(folder, relative):
    need(type(relative) is str and 1 <= len(relative) <= 256
         and "\\" not in relative and "\0" not in relative, "ASSET_RELATIVE_PATH")
    path = PurePosixPath(relative)
    need(not path.is_absolute() and path.as_posix() == relative
         and all(x not in (".", "..", ".git") for x in path.parts),
         "ASSET_PATH_TRAVERSAL")
    result = folder
    for part in path.parts:
        result = result / part
        need(not result.is_symlink(), "ASSET_SYMLINK")
    need(result.is_file(), "ASSET_FILE_MISSING")
    return result


def verify_native_assets(adapters, inventory):
    engines = receipt.check_adapters(adapters)
    need(type(inventory) is dict and
         set(inventory) == {"schema", "models"} and
         inventory["schema"] == ASR_ASSETS_SCHEMA, "ASR_ASSET_INVENTORY_SCHEMA")
    rows = inventory["models"]
    need(type(rows) is list and len(rows) == 3, "THREE_PHYSICAL_ASR_MODEL_ROOTS")
    found, roots, total_bytes = set(), set(), 0
    for item in rows:
        need(type(item) is dict and
             set(item) == {"engine", "model_id", "model_revision", "root", "files"},
             "ASR_ASSET_ENTRY_FIELDS")
        engine = item["engine"]
        need(type(engine) is str and engine in engines and engine not in found,
             "ASR_ASSET_ENGINE_IDENTITY")
        found.add(engine)
        adapter = engines[engine]
        need(item["model_id"] == adapter["model_id"] and
             item["model_revision"] == adapter["model_revision"],
             "ASR_ASSET_MODEL_IDENTITY")
        root_name = item["root"]
        need(type(root_name) is str and root_name and Path(root_name).is_absolute(),
             "ASR_ROOT_ABSOLUTE")
        root = Path(root_name)
        need(root.is_dir() and
             not any(parent.is_symlink() for parent in (root, *root.parents)),
             "ASR_ROOT_SYMLINK_OR_MISSING")
        root_identity = str(root.resolve(strict=True))
        need(root_identity not in roots, "ASR_SAME_MODEL_DIRECTORY")
        roots.add(root_identity)
        entries = item["files"]
        need(type(entries) is list and 1 <= len(entries) <= MAX_MODEL_FILES,
             "ASR_MODEL_FILE_COUNT")
        filenames, saved, model_bytes = set(), [], 0
        for row in entries:
            need(type(row) is dict and set(row) == {"path", "sha256", "bytes"},
                 "ASR_MODEL_FILE_FIELDS")
            rel = row["path"]
            need(rel not in filenames, "ASR_DUPLICATE_FILE")
            filenames.add(rel)
            need(type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]) and
                 type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_SINGLE_FILE,
                 "ASR_MODEL_FILE_LOCK_INVALID")
            size, digest = sha_file(rooted_regular(root, rel), MAX_SINGLE_FILE)
            need(size == row["bytes"] and digest == row["sha256"],
                 "ASR_MODEL_FILE_CHANGED")
            model_bytes += size
            need(model_bytes <= MAX_PER_MODEL, "ASR_MODEL_TOTAL_BYTES")
            saved.append({"path": rel, "bytes": size, "sha256": digest})
        seen_files, scanned = set(), 0
        for entry in root.rglob("*"):
            scanned += 1
            need(scanned <= MAX_SCAN_ENTRIES and not entry.is_symlink(),
                 "ASR_UNBOUNDED_OR_ALIAS_TREE")
            if entry.is_file():
                need(stat.S_ISREG(entry.stat().st_mode), "ASR_NONREGULAR_MODEL_ENTRY")
                seen_files.add(entry.relative_to(root).as_posix())
        need(seen_files == filenames, "ASR_UNLISTED_OR_MISSING_MODEL_FILE")
        declaration = {"schema": ASR_DIGEST_SCHEMA, "engine": engine,
                       "model_id": item["model_id"],
                       "model_revision": item["model_revision"],
                       "files": sorted(saved, key=lambda x: x["path"])}
        actual_lock = hashlib.sha256(pool.canonical(declaration)).hexdigest()
        need(actual_lock == adapter["model_assets_sha256"],
             "ASR_ASSET_DECLARATION_SHA_MISMATCH")
        # The adapter executable is also checked against the actual bytes.
        exe = Path(adapter["executable"])
        need(not any(p.is_symlink() for p in (exe, *exe.parents)),
             "ASR_EXECUTABLE_ALIAS")
        size, actual_exe = sha_file(exe, 8 * 1024 * 1024)
        need(actual_exe == adapter["executable_sha256"],
             "ASR_EXECUTABLE_CHANGED")
        total_bytes += model_bytes
    need(found == set(engines), "INCOMPLETE_ASR_ASSET_ROSTER")
    return {"models_verified": len(found), "total_model_bytes_verified": total_bytes}


def _coverage(cells):
    kinds = Counter()
    families = set()
    roles = set()
    for row in cells:
        families.add(row["source_family_id"])
        roles.add(row["role"])
        if row["intended_kind"] == "positive-intent":
            kinds["kw" + str(row["intended_keyword_id"])] += 1
        else:
            kinds[row["intended_kind"]] += 1
    return {"kw1": kinds["kw1"], "kw2": kinds["kw2"],
            "hard_negatives": kinds["hard-negative-intent"],
            "ordinary_negatives": kinds["speech-negative-intent"],
            "source_families": len(families),
            "train_source_present": "train-candidate" in roles}


def missing_coverage(score, min_families):
    return sorted([
        code for code, failed in (
            ("MISSING_KW1_POSITIVE", score["kw1"] == 0),
            ("MISSING_KW2_POSITIVE", score["kw2"] == 0),
            ("MISSING_HARD_NEGATIVE", score["hard_negatives"] == 0),
            ("MISSING_ORDINARY_NEGATIVE", score["ordinary_negatives"] == 0),
            ("INSUFFICIENT_SOURCE_FAMILIES", score["source_families"] < min_families),
            ("NO_TRAIN_SOURCE_IN_PILOT", not score["train_source_present"]),
        ) if failed
    ])


def audit(plan, sources, audio_root, tts_budget, asr_budget, min_families):
    need(type(tts_budget) is int and
         1 <= tts_budget <= MAX_PILOT_TTS, "PILOT_TTS_BUDGET")
    need(type(asr_budget) is int and asr_budget == 3 * tts_budget,
         "PILOT_THREE_ASR_PER_WAV_REQUIRED")
    need(type(min_families) is int and 1 <= min_families <= 4,
         "PILOT_FAMILY_REQUIREMENT")
    pool.verify_plan(plan)
    expected = pool.plan(sources, audio_root,
                         repetitions=plan["repetitions"],
                         batch_id=plan["batch_id"],
                         revision=plan["model_revision"],
                         max_calls=plan["budgets"]["max_total_calls"])
    need(pool.canonical(expected) == pool.canonical(plan),
         "PILOT_SOURCE_OR_PLAN_DRIFT")
    need(tts_budget <= plan["budgets"]["max_total_calls"] and
         tts_budget <= len(plan["cells"]), "PILOT_EXCEEDS_FROZEN_BUDGET")
    # The original runner chooses the exact FIRST eligible entries of this
    # canonical cell list, not random sampling or round-robin families.
    selected = plan["cells"][:tts_budget]
    score = _coverage(selected)
    missing = missing_coverage(score, min_families)
    minimal = None
    for index in range(1, min(len(plan["cells"]),
                              plan["budgets"]["max_total_calls"],
                              MAX_PILOT_TTS) + 1):
        if not missing_coverage(_coverage(plan["cells"][:index]), min_families):
            minimal = index
            break
    return {"policy": "source-pinned-kws-bounded-pilot-v1",
            "status": "SOURCE_CONSISTENT_PILOT_PLANNED" if not missing else "PILOT_COVERAGE_HOLD",
            "planned_source_sha256": hashlib.sha256(pool.canonical(sources)).hexdigest(),
            "plan_sha256": hashlib.sha256(pool.canonical(plan)).hexdigest(),
            "tts_calls_budgeted": tts_budget, "asr_calls_budgeted": asr_budget,
            "selected_coverage": score,
            "coverage_gaps": missing, "min_tts_calls_for_coverage": minimal,
            "requested_source_families": min_families,
            "rights_externally_authenticated": False,
            "asr_runtime_execution_authenticated": False,
            "spoken_labels_authenticated": False,
            "training_admitted": False, "qualification_allowed": False,
            "new_model_calls": 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sources", "audio-root", "model-root",
                 "model-lock", "adapter-lock", "asr-assets"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--max-tts", type=int, required=True)
    p.add_argument("--max-asr", type=int, required=True)
    p.add_argument("--min-source-families", type=int, default=2)
    args = p.parse_args(argv)
    plan = pool.read_json(args.plan, MAX_INVENTORY_BYTES)
    sources = pool.read_json(args.sources, MAX_INVENTORY_BYTES)
    result = audit(plan, sources, args.audio_root, args.max_tts,
                   args.max_asr, args.min_source_families)
    model_lock = pool.read_json(args.model_lock, MAX_INVENTORY_BYTES)
    tts.asset_lock(args.model_root, model_lock, plan["model_revision"])
    adapters = pool.read_json(args.adapter_lock, MAX_INVENTORY_BYTES)
    asr_assets = pool.read_json(args.asr_assets, MAX_INVENTORY_BYTES)
    verified = verify_native_assets(adapters, asr_assets)
    result.update(tts_model_bytes_verified=sum(r["bytes"] for r in model_lock["files"]),
                  **verified)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not result["coverage_gaps"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print("PILOT_PREFLIGHT_BLOCKED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
