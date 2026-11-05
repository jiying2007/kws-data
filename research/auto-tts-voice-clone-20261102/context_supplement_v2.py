"""Explicit additive context v2; fixed-vocabulary requests, never old retries.

The complete context v1 plan is an immutable ancestor. This new schema, file
and cell namespace leave every running v1 producer and receipt unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path

import context_supplement as v1

pool = p = v1.pool
media = v1.media
LOCK_SCHEMA = v1.LOCK_SCHEMA
asset_lock = v1.asset_lock
QwenBulkClone = v1.QwenBulkClone
GENERATION = copy.deepcopy(v1.GENERATION)
SCHEMA = "kws-context-supplement-plan-v2"
STAGE = "context-supplement-v2"
MAX_PLAN_BYTES = 8 * 1024 * 1024
KEYWORDS = v1.KEYWORDS
CLASSES = v1.CLASSES
VOCABULARY_SCHEMA = "kws-d1-fulltext-vocabulary-v2"
VOCABULARY_SHA256 = "104575dab3a9f7ed622d8fa29aa8a897e0590cca8aed9214b26ef4cd1c8b4e4b"
VOCABULARY_FILE_SHA256 = "7cfd3e1f62ef1700be65123257a3073083fafaf50e4615533d82ad002e19203e"
BASE_TOKENS_SHA256 = "5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa"
FIXTURE = Path(__file__).resolve().parent / "fixtures/context-full2601-vocabulary.json"


def digest(value):
    return hashlib.sha256(pool.canonical(value)).hexdigest()


def vocabulary_binding(vocabulary):
    if type(vocabulary) is bytes:
        pool.require(hashlib.sha256(vocabulary).hexdigest() == VOCABULARY_FILE_SHA256,
                     "CONTEXT_V2_FIXED_VOCABULARY_FILE_CHANGED")
        value = json.loads(vocabulary)
    else:
        pool.require(type(vocabulary) is dict, "CONTEXT_V2_VOCABULARY_REQUIRED")
        value = copy.deepcopy(vocabulary)
    pool.require(digest(value) == VOCABULARY_SHA256 and
                 hashlib.sha256(pool.canonical(value) + b"\n").hexdigest() == VOCABULARY_FILE_SHA256 and
                 value["schema"] == VOCABULARY_SCHEMA and value["base_tokens_sha256"] == BASE_TOKENS_SHA256 and
                 value["output_count"] == 2601 and value["blank_id"] == 0,
                 "CONTEXT_V2_FIXED_VOCABULARY_CHANGED")
    return {"value": value, "canonical_sha256": VOCABULARY_SHA256,
            "file_sha256": VOCABULARY_FILE_SHA256, "base_tokens_sha256": BASE_TOKENS_SHA256}


def requested_text_preflight(text, binding):
    normalized = pool.reference_text.normalize(text)
    mapping = binding["value"]["token_to_id"]
    missing = [char for char in normalized if char not in mapping or
               type(mapping[char]) is not int or not 0 < mapping[char] < 2601]
    pool.require(bool(normalized) and not missing, "CONTEXT_V2_REQUESTED_FULLTEXT_OOV")
    return {"vocabulary_sha256": VOCABULARY_SHA256,
            "normalized_text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "all_characters_representable": True, "scope": "requested_text_only"}


def templates():
    result = []
    for kind in CLASSES:
        for number, keyword in enumerate(KEYWORDS, 1):
            prompt = {"prefix": keyword + "，请把灯打开。",
                      "middle": "现在，" + keyword + "，请把灯打开。",
                      "suffix": "如果你准备好了就回应我，" + keyword + "。",
                      "repeated-call": keyword + "，" + keyword + "。"}[kind]
            result.append({"template_id": kind + "-kw" + str(number), "prompt": prompt,
                           "context_class": kind, "intended_keyword_id": number,
                           "requested_keyword_occurrences": 2 if kind == "repeated-call" else 1})
    return result


def _assemble(parent, parent_sha, binding, *, count, batch_id):
    v1.verify_plan(parent)
    pool._sha(parent_sha)
    pool.require(digest(parent) == parent_sha and parent["requested_cells"] == 16,
                 "CONTEXT_V2_REQUIRES_FROZEN_CONTEXT16_PARENT")
    pool.require(type(count) is int and count in (8, 16), "CONTEXT_V2_REQUEST_COUNT")
    pool._id(batch_id, "CONTEXT_V2_BATCH_ID")
    pool.require(type(binding) is dict and binding == vocabulary_binding(binding["value"]),
                 "CONTEXT_V2_VOCABULARY_BINDING_CHANGED")
    parent, binding = copy.deepcopy(parent), copy.deepcopy(binding)
    refs = parent["references"]
    ordered = sorted(refs, key=lambda row: row["reference_id"])
    recipe = templates()
    old_by_reference = {}
    for row in parent["cells"]:
        old_by_reference.setdefault(row["reference_id"], set()).add(row["prompt"])
    cells = []
    for repetition in range(count // 8):
        for index, ref in enumerate(ordered):
            template = recipe[index % 4] if count == 8 else recipe[(index + 3 * repetition) % 8]
            identity = digest({"namespace": SCHEMA, "parent_context_plan_sha256": parent_sha,
                "batch_id": batch_id, "requested_cells": count,
                "reference_id": ref["reference_id"], "template": template})
            cells.append({"cell_id": "bulk-" + identity[:24],
                **{key: ref[key] for key in ("reference_id", "speaker_group_id", "source_family_id", "role")},
                **template, "intended_kind": "positive-intent", "generation_category": "full-positive",
                "seed": 1000000 + int(identity[:8], 16) % 1000000000, "repetition": repetition,
                "label_authority": "intended_only", "status": "NOT_RUN", "actual_text": None,
                "synthetic": True, "training_admitted": False, "qualification_allowed": False,
                "explicit_additive_revision": "fixed-vocabulary-v2",
                "same_requested_text_as_parent": template["prompt"] in old_by_reference[ref["reference_id"]],
                "requested_text_preflight": requested_text_preflight(template["prompt"], binding)})
    d2 = parent["d2_parent"]["plan"]
    old_ids = {row["cell_id"] for row in parent["cells"] + d2["cells"] + d2["d1_parent"]["plan"]["cells"]}
    pool.require(len({row["cell_id"] for row in cells}) == count and
                 not old_ids & {row["cell_id"] for row in cells}, "CONTEXT_V2_HISTORICAL_CELL_REISSUE")
    return {"schema": SCHEMA, "stage": STAGE, "batch_id": batch_id, "requested_cells": count,
            "model_id": parent["model_id"], "model_revision": parent["model_revision"],
            "source_manifest_sha256": parent["source_manifest_sha256"], "references": refs,
            "context_parent": {"plan_sha256": parent_sha, "plan": parent},
            "d2_parent": copy.deepcopy(parent["d2_parent"]),
            "speaker_registry": copy.deepcopy(parent["speaker_registry"]), "vocabulary": binding,
            "cells": cells, "generation": copy.deepcopy(GENERATION),
            "budgets": {"max_total_calls": count, "max_calls_per_invocation": 16,
                        "max_retries": 0, "max_paid_usd": 0},
            "d1_cells_generated_again": 0, "d2_cells_generated_again": 0,
            "context_v1_cells_generated_again": 0,
            "model_calls": 0, "actual_spoken_labels_verified": False,
            "requested_text_preflight_is_observed_label": False,
            "human_gold": False, "training_admitted": False, "qualification_allowed": False}


def plan(parent_context, parent_sha, source_manifest, audio_root, vocabulary, *, count=8, batch_id):
    value = _assemble(parent_context, parent_sha, vocabulary_binding(vocabulary), count=count, batch_id=batch_id)
    return rebuild(value, source_manifest, audio_root)


def verify_plan(value):
    pool.require(type(value) is dict and value.get("schema") == SCHEMA and
                 type(value.get("context_parent")) is dict and
                 set(value["context_parent"]) == {"plan_sha256", "plan"}, "CONTEXT_V2_PLAN_SCHEMA")
    expected = _assemble(value["context_parent"]["plan"], value["context_parent"]["plan_sha256"],
        value["vocabulary"], count=value["requested_cells"], batch_id=value["batch_id"])
    pool.require(pool.canonical(value) == pool.canonical(expected), "CONTEXT_V2_PLAN_CHANGED")
    return value


def rebuild(value, source_manifest, audio_root):
    verify_plan(value)
    pool.validate_sources(source_manifest, audio_root)
    parent = value["context_parent"]["plan"]
    pool.require(digest(source_manifest) == parent["source_manifest_sha256"] and
                 source_manifest["references"] == parent["references"], "CONTEXT_V2_SOURCE_MANIFEST_CHANGED")
    return _assemble(parent, value["context_parent"]["plan_sha256"], value["vocabulary"],
                     count=value["requested_cells"], batch_id=value["batch_id"])


def runner_recipe(value, selected_cells=None):
    verify_plan(value)
    ids = [row["cell_id"] for row in value["cells"]]
    selected = ids if selected_cells is None else selected_cells
    pool.require(type(selected) is list and bool(selected) and len(selected) <= 16 and
                 selected[0] in ids and selected == ids[ids.index(selected[0]):ids.index(selected[0]) + len(selected)],
                 "CONTEXT_V2_CONTIGUOUS_SELECTION")
    return {"selected_cells": list(selected), "max_calls": len(selected), "generation": copy.deepcopy(GENERATION)}


def run(value, source_manifest, audio_root, output, backend, *, max_new_calls, writer=media._write_waves):
    """Execute only through a supplied BoundedBackend; never retry a claim.

    The backend retains call/returned/native JSON and returned NPY/WAV before
    this shared signal gate. A failed gate does not remove the native return.
    The independent owner must supply process/time/model-asset supervision.
    """
    expected = rebuild(value, source_manifest, audio_root)
    pool.require(pool.canonical(expected) == pool.canonical(value), "CONTEXT_V2_SOURCE_OR_PLAN_CHANGED")
    pool.require(type(max_new_calls) is int and 0 <= max_new_calls <= value["budgets"]["max_calls_per_invocation"],
                 "CONTEXT_V2_RUN_INVOCATION_BUDGET")
    root = Path(output)
    plan_sha = digest(value)
    lock = {"schema": "kws-bulk-clone-execution-lock-v1", "plan_sha256": plan_sha,
            "training_admitted": False, "model_id": value["model_id"]}
    if root.exists():
        pool.require(pool.read_json(root / "plan-lock.json") == lock, "CONTEXT_V2_EXECUTION_LOCK_DRIFT")
    pending = [cell for cell in value["cells"] if not (root / "cells" / cell["cell_id"] / "started.json").exists()]
    started = len(value["cells"]) - len(pending)
    pool.require(started <= value["budgets"]["max_total_calls"], "CONTEXT_V2_BUDGET_ALREADY_EXCEEDED")
    selected = pending[:min(max_new_calls, value["budgets"]["max_total_calls"] - started)]
    if selected:
        recipe = runner_recipe(value, [row["cell_id"] for row in selected])
        supplied = getattr(backend, "recipe", None)
        pool.require(type(supplied) is dict and all(supplied.get(k) == v for k, v in recipe.items()),
                     "CONTEXT_V2_BOUNDED_NATIVE_BACKEND_REQUIRED")
    if not root.exists():
        root.mkdir(parents=True)
        media._once(root / "plan-lock.json", lock)
    refs = {r["reference_id"]: r for r in value["references"]}
    invoked = 0
    for cell in selected:
        folder = root / "cells" / cell["cell_id"]
        pool.require(not folder.exists(), "CONTEXT_V2_UNKNOWN_CELL_DIRECTORY")
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
                         "CONTEXT_V2_WRITER_CONTRACT")
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
    return {"schema": "kws-context-supplement-v2-summary-v1", "status": "PLANNED_ONLY",
            "plan_sha256": digest(value), "parent_context_plan_sha256": value["context_parent"]["plan_sha256"],
            "vocabulary_sha256": value["vocabulary"]["canonical_sha256"],
            "new_requested_cells": len(value["cells"]),
            "requested_context_counts": dict(Counter(c["context_class"] for c in value["cells"])),
            "requested_keyword_counts": dict(Counter(str(c["intended_keyword_id"]) for c in value["cells"])),
            "requested_split_counts": dict(Counter(c["role"] for c in value["cells"])),
            "all_requested_fulltext_in_fixed_vocabulary": True,
            "new_model_calls": 0, "observed_usable_cells": None, "training_admitted": False}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-context-plan", "parent-context-plan-sha256", "sources", "audio-root", "vocabulary", "batch-id"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--count", type=int, choices=(8, 16), default=8)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args(argv)
    value = plan(pool.read_json(args.parent_context_plan, MAX_PLAN_BYTES), args.parent_context_plan_sha256,
        pool.read_json(args.sources), args.audio_root, Path(args.vocabulary).read_bytes(), count=args.count,
        batch_id=args.batch_id)
    if args.output is not None:
        output = args.output.absolute()
        pool.require(output.parent.is_dir() and not output.exists() and
                     not any(p.is_symlink() for p in (output, *output.parents)), "CONTEXT_V2_NEW_PLAN_REQUIRED")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(pool.canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
