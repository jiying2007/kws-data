"""Independent 8/16-cell contextual wake supplement; D1 and D2 stay frozen.

Planning and validation never execute a model. The run API accepts the existing
bounded native backend protocol; no model is constructed by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path

import bulk_pool as pool
import bulk_runner
import d2_pool
import runner as media

# Re-export the existing offline backend/asset protocol for a thin owner driver.
# These aliases do not load a model or change the frozen bulk implementation.
p = pool
LOCK_SCHEMA = bulk_runner.LOCK_SCHEMA
asset_lock = bulk_runner.asset_lock
QwenBulkClone = bulk_runner.QwenBulkClone

SCHEMA = "kws-context-supplement-plan-v1"
STAGE = "context-supplement"
MAX_PLAN_BYTES = 8 * 1024 * 1024
CLASSES = ("prefix", "middle", "suffix", "repeated-call")
KEYWORDS = ("你好小窝", "小窝小窝")
GENERATION = dict(language="Chinese", non_streaming_mode=True, do_sample=True,
                  top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05,
                  subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
                  subtalker_temperature=0.9, max_new_tokens=256)


def digest(value):
    return hashlib.sha256(pool.canonical(value)).hexdigest()


def templates():
    rows = []
    for kind in CLASSES:
        for keyword_id, keyword in enumerate(KEYWORDS, 1):
            text = {"prefix": keyword + "，请把客厅的灯打开。",
                    "middle": "客厅有点暗，" + keyword + "，请把灯打开。",
                    "suffix": "如果你准备好了就回应我，" + keyword + "。",
                    "repeated-call": keyword + "，" + keyword + "。"}[kind]
            rows.append({"template_id": kind + "-kw" + str(keyword_id),
                         "context_class": kind, "prompt": text, "intended_keyword_id": keyword_id,
                         "requested_keyword_occurrences": 2 if kind == "repeated-call" else 1})
    return rows


def _parent(value, expected_sha):
    pool._sha(expected_sha)
    d2_pool.verify_plan(value)
    pool.require(value["stage"] == d2_pool.EXISTING and digest(value) == expected_sha,
                 "CONTEXT_REQUIRES_FROZEN_EXISTING_EIGHT_D2")


def _assemble(parent, parent_sha, *, count, batch_id):
    _parent(parent, parent_sha)
    pool.require(type(count) is int and count in (8, 16), "CONTEXT_BOUNDED_REQUEST_COUNT")
    pool._id(batch_id, "CONTEXT_BATCH_ID")
    parent = copy.deepcopy(parent)
    refs = parent["references"]
    ordered = sorted(refs, key=lambda row: row["reference_id"])
    recipe = templates()
    cells = []
    for repetition in range(count // 8):
        for index, ref in enumerate(ordered):
            # The odd offset gives each original identity both keywords and
            # two different contexts in the 16-cell plan. All eight templates
            # appear twice, while the original 6/2 source split stays fixed.
            template = recipe[(index + 3 * repetition) % 8]
            identity = digest({"namespace": SCHEMA, "parent_plan_sha256": parent_sha,
                "batch_id": batch_id, "requested_cells": count, "reference_id": ref["reference_id"],
                "template_id": template["template_id"], "prompt": template["prompt"]})
            cells.append({"cell_id": "bulk-" + identity[:24],
                **{key: ref[key] for key in ("reference_id", "speaker_group_id", "source_family_id", "role")},
                **template, "intended_kind": "positive-intent", "generation_category": "full-positive",
                "seed": 1000000 + int(identity[:8], 16) % 1000000000, "repetition": repetition,
                "status": "NOT_RUN", "label_authority": "intended_only", "synthetic": True,
                "actual_text": None, "training_admitted": False, "qualification_allowed": False})
    original_ids = {c["cell_id"] for c in parent["cells"] + parent["d1_parent"]["plan"]["cells"]}
    pool.require(len({c["cell_id"] for c in cells}) == count and
                 not original_ids & {c["cell_id"] for c in cells}, "CONTEXT_MUST_NOT_REISSUE_D1_D2")
    return {"schema": SCHEMA, "stage": STAGE, "batch_id": batch_id, "requested_cells": count,
            "model_id": parent["model_id"], "model_revision": parent["model_revision"],
            "source_manifest_sha256": parent["source_manifest_sha256"], "references": refs,
            "d2_parent": {"plan_sha256": parent_sha, "plan": parent},
            "speaker_registry": copy.deepcopy(parent["speaker_registry"]), "cells": cells,
            "generation": copy.deepcopy(GENERATION),
            "budgets": {"max_total_calls": count, "max_calls_per_invocation": 16,
                        "max_retries": 0, "max_paid_usd": 0},
            "d1_cells_generated_again": 0, "d2_cells_generated_again": 0,
            "model_calls": 0, "actual_spoken_labels_verified": False,
            "human_gold": False, "training_admitted": False, "qualification_allowed": False}


def plan(parent, parent_sha, source_manifest, audio_root, *, count=16, batch_id):
    value = _assemble(parent, parent_sha, count=count, batch_id=batch_id)
    return rebuild(value, source_manifest, audio_root)


def verify_plan(value):
    pool.require(type(value) is dict and value.get("schema") == SCHEMA and
                 type(value.get("d2_parent")) is dict and set(value["d2_parent"]) == {"plan_sha256", "plan"},
                 "CONTEXT_PLAN_SCHEMA")
    expected = _assemble(value["d2_parent"]["plan"], value["d2_parent"]["plan_sha256"],
                         count=value["requested_cells"], batch_id=value["batch_id"])
    pool.require(pool.canonical(value) == pool.canonical(expected), "CONTEXT_PLAN_CHANGED")
    return value


def rebuild(value, source_manifest, audio_root):
    verify_plan(value)
    pool.validate_sources(source_manifest, audio_root)
    parent = value["d2_parent"]["plan"]
    pool.require(digest(source_manifest) == parent["source_manifest_sha256"] and
                 source_manifest["references"] == parent["references"], "CONTEXT_SOURCE_MANIFEST_CHANGED")
    return _assemble(parent, value["d2_parent"]["plan_sha256"], count=value["requested_cells"], batch_id=value["batch_id"])


def runner_recipe(value, selected_cells=None):
    """Exact fields consumed by active_tts_smoke.BoundedBackend.

    Owner stage/job metadata is supplied by the independent pipeline wrapper.
    This declaration is not a queue claim, native return or model invocation.
    """
    verify_plan(value)
    ids = [row["cell_id"] for row in value["cells"]]
    selected = ids if selected_cells is None else selected_cells
    pool.require(type(selected) is list and bool(selected) and len(selected) <= 16 and
                 selected[0] in ids and selected == ids[ids.index(selected[0]):ids.index(selected[0]) + len(selected)],
                 "CONTEXT_CONTIGUOUS_SELECTION")
    return {"selected_cells": list(selected), "max_calls": len(selected), "generation": copy.deepcopy(GENERATION)}


def run(value, source_manifest, audio_root, output, backend, *, max_new_calls, writer=media._write_waves):
    """Execute only through a supplied BoundedBackend; never retry a claim.

    The backend retains call/returned/native JSON and returned NPY/WAV before
    this shared signal gate. A failed gate does not remove the native return.
    The independent owner must supply process/time/model-asset supervision.
    """
    expected = rebuild(value, source_manifest, audio_root)
    pool.require(pool.canonical(expected) == pool.canonical(value), "CONTEXT_SOURCE_OR_PLAN_CHANGED")
    pool.require(type(max_new_calls) is int and 0 <= max_new_calls <= value["budgets"]["max_calls_per_invocation"],
                 "CONTEXT_RUN_INVOCATION_BUDGET")
    root = Path(output)
    plan_sha = digest(value)
    lock = {"schema": "kws-bulk-clone-execution-lock-v1", "plan_sha256": plan_sha,
            "training_admitted": False, "model_id": value["model_id"]}
    if root.exists():
        pool.require(pool.read_json(root / "plan-lock.json") == lock, "CONTEXT_EXECUTION_LOCK_DRIFT")
    pending = [cell for cell in value["cells"] if not (root / "cells" / cell["cell_id"] / "started.json").exists()]
    started = len(value["cells"]) - len(pending)
    pool.require(started <= value["budgets"]["max_total_calls"], "CONTEXT_BUDGET_ALREADY_EXCEEDED")
    selected = pending[:min(max_new_calls, value["budgets"]["max_total_calls"] - started)]
    if selected:
        recipe = runner_recipe(value, [row["cell_id"] for row in selected])
        supplied = getattr(backend, "recipe", None)
        pool.require(type(supplied) is dict and all(supplied.get(k) == v for k, v in recipe.items()),
                     "CONTEXT_BOUNDED_NATIVE_BACKEND_REQUIRED")
    if not root.exists():
        root.mkdir(parents=True)
        media._once(root / "plan-lock.json", lock)
    refs = {r["reference_id"]: r for r in value["references"]}
    invoked = 0
    for cell in selected:
        folder = root / "cells" / cell["cell_id"]
        pool.require(not folder.exists(), "CONTEXT_UNKNOWN_CELL_DIRECTORY")
        folder.mkdir(parents=True)
        ref = refs[cell["reference_id"]]
        media._once(folder / "started.json", {"schema": "kws-bulk-clone-start-v1", "cell_id": cell["cell_id"],
            "reference_id": cell["reference_id"], "reference_wav_sha256": ref["wav_sha256"],
            "plan_sha256": plan_sha, "source_family_id": cell["source_family_id"],
            "seed": cell["seed"], "attempts": 1, "training_admitted": False})
        invoked += 1
        try:
            audio, rate = backend.synthesize(cell, ref, audio_root)
            metrics = media.quality(audio, rate)
            rendered = writer(folder, audio, rate, metrics)
            required = {"native_sha256", "native_bytes", "wav_sha256", "wav_bytes", "pcm_sha256", "audio_frames", "sample_rate_hz"}
            pool.require(type(rendered) is dict and set(rendered) == required and
                         rendered["sample_rate_hz"] == 16000 and rendered["audio_frames"] > 0,
                         "CONTEXT_WRITER_CONTRACT")
            media._once(folder / "completed.json", {"schema": "kws-bulk-clone-complete-v1", "cell_id": cell["cell_id"],
                "status": "GENERATED_PENDING_ASR", "technical": metrics, "audio": rendered,
                "source_family_id": cell["source_family_id"], "intended_kind": cell["intended_kind"],
                "actual_text": None, "training_admitted": False, "qualification_allowed": False})
        except Exception as error:
            media._once(folder / "failed.json", {"schema": "kws-bulk-clone-failure-v1", "cell_id": cell["cell_id"],
                "status": "FAILED_NO_RETRY", "error_type": type(error).__name__, "training_admitted": False})
    return {"attempts_this_call": invoked,
            "total_started": sum((root / "cells" / c["cell_id"] / "started.json").exists() for c in value["cells"]),
            "total_completed": sum((root / "cells" / c["cell_id"] / "completed.json").exists() for c in value["cells"]),
            "model_calls_requested": invoked, "qualification_allowed": False, "training_admitted": False,
            "status": "BULK_SYNTHETIC_UNADMITTED"}


def summary(value):
    verify_plan(value)
    return {"schema": "kws-context-supplement-summary-v1", "status": "PLANNED_ONLY", "plan_sha256": digest(value),
            "d2_parent_plan_sha256": value["d2_parent"]["plan_sha256"], "new_requested_cells": len(value["cells"]),
            "requested_context_counts": dict(Counter(c["context_class"] for c in value["cells"])),
            "requested_keyword_counts": dict(Counter(str(c["intended_keyword_id"]) for c in value["cells"])),
            "requested_split_counts": dict(Counter(c["role"] for c in value["cells"])),
            "new_model_calls": 0, "observed_usable_cells": None, "training_admitted": False}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-plan", "parent-plan-sha256", "sources", "audio-root", "batch-id"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--count", type=int, choices=(8, 16), default=16)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args(argv)
    value = plan(pool.read_json(args.parent_plan, MAX_PLAN_BYTES), args.parent_plan_sha256,
        pool.read_json(args.sources), args.audio_root, count=args.count, batch_id=args.batch_id)
    if args.output is not None:
        output = args.output.absolute()
        pool.require(output.parent.is_dir() and not output.exists() and
                     not any(p.is_symlink() for p in (output, *output.parents)), "CONTEXT_NEW_PLAN_REQUIRED")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(pool.canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
