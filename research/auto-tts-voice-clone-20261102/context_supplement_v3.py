"""New full-text coverage for the 24 added identities; no historical retry.

Every identity receives K1, K2 and ordinary speech. The frozen D1, both D2
increments and context v1/v2 remain complete ancestors. Both normalized prompt
hashes and cell identities are excluded before any new request can be planned.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path

import context_supplement_v2 as v2
import d2_pool

pool = p = v2.pool
media = v2.media
LOCK_SCHEMA = v2.LOCK_SCHEMA
asset_lock = v2.asset_lock
QwenBulkClone = v2.QwenBulkClone
GENERATION = copy.deepcopy(v2.GENERATION)
SCHEMA = "kws-context-supplement-plan-v3"
STAGE = "context-supplement-v3"
MAX_PLAN_BYTES = 8 * 1024 * 1024
KEYWORDS = v2.KEYWORDS
CLASSES = v2.CLASSES
ORDINARY_CLASS = "ordinary-speech"
VOCABULARY_SCHEMA = v2.VOCABULARY_SCHEMA
VOCABULARY_SHA256 = v2.VOCABULARY_SHA256
VOCABULARY_FILE_SHA256 = v2.VOCABULARY_FILE_SHA256
BASE_TOKENS_SHA256 = v2.BASE_TOKENS_SHA256
FIXTURE = v2.FIXTURE
vocabulary_binding = v2.vocabulary_binding
requested_text_preflight = v2.requested_text_preflight

ORDINARY = (
    "请先把房间的灯关上。", "我们现在把音乐声音调小。",
    "等一会儿再给我打电话。", "今天下午一起去公园走走。",
    "你能听清我说的话吗。", "门外有人在说话。",
    "这个房间现在很安静。", "请告诉我明天的天气。",
)


def digest(value):
    return hashlib.sha256(pool.canonical(value)).hexdigest()


def prompt_sha(text):
    return hashlib.sha256(pool.reference_text.normalize(text).encode()).hexdigest()


def templates():
    rows = []
    for kind in CLASSES:
        for number, keyword in enumerate(KEYWORDS, 1):
            text = {"prefix": keyword + "，请把声音调小一点。",
                    "middle": "听到了吗，" + keyword + "，把声音调小一点。",
                    "suffix": "请把声音调小一点，谢谢，" + keyword + "。",
                    "repeated-call": keyword + "，声音太大了，" + keyword + "。"}[kind]
            rows.append({"template_id": "new24-" + kind + "-kw" + str(number),
                         "prompt": text, "context_class": kind,
                         "intended_kind": "positive-intent", "generation_category": "full-positive",
                         "intended_keyword_id": number,
                         "requested_keyword_occurrences": 2 if kind == "repeated-call" else 1})
    rows.extend({"template_id": "new24-ordinary-" + str(index + 1), "prompt": text,
                 "context_class": ORDINARY_CLASS, "intended_kind": "speech-negative-intent",
                 "generation_category": "ordinary-negative", "intended_keyword_id": None,
                 "requested_keyword_occurrences": 0} for index, text in enumerate(ORDINARY))
    return rows


def historical_plans(parent_d2, parent_context):
    """Both validated lineages bind every historical requested payload."""
    return {"d1": parent_d2["d1_parent"]["plan"],
            "d2-existing-eight": parent_context["d2_parent"]["plan"],
            "d2-new-twenty-four": parent_d2,
            "context-v1": parent_context["context_parent"]["plan"],
            "context-v2": parent_context}


def _history(parent_d2, parent_context):
    plans = historical_plans(parent_d2, parent_context)
    rows = [row for value in plans.values() for row in value["cells"]]
    ids = {row["cell_id"] for row in rows}
    prompts = {prompt_sha(row["prompt"]) for row in rows}
    return {"plan_sha256": {key: digest(value) for key, value in plans.items()},
            "unique_historical_cell_count": len(ids),
            "historical_cell_ids_sha256": digest(sorted(ids)),
            "normalized_prompt_sha256": sorted(prompts),
            "normalization": "kws_data_contract.reference_text.normalize"}, ids, prompts


def _assemble(parent_d2, parent_d2_sha, parent_context, parent_context_sha, binding, *, batch_id):
    d2_pool.verify_plan(parent_d2)
    v2.verify_plan(parent_context)
    pool._sha(parent_d2_sha); pool._sha(parent_context_sha)
    pool.require(parent_d2["stage"] == d2_pool.NEW and digest(parent_d2) == parent_d2_sha,
                 "CONTEXT_V3_FROZEN_NEW24_D2_REQUIRED")
    pool.require(parent_context["requested_cells"] == 16 and digest(parent_context) == parent_context_sha,
                 "CONTEXT_V3_FROZEN_CONTEXT_V2_16_REQUIRED")
    pool.require(parent_d2["d1_parent"]["plan_sha256"] ==
                 parent_context["d2_parent"]["plan"]["d1_parent"]["plan_sha256"] and
                 parent_d2["model_id"] == parent_context["model_id"] and
                 parent_d2["model_revision"] == parent_context["model_revision"],
                 "CONTEXT_V3_COMMON_ANCESTRY_REQUIRED")
    pool._id(batch_id, "CONTEXT_V3_BATCH_ID")
    pool.require(type(binding) is dict and binding == vocabulary_binding(binding["value"]),
                 "CONTEXT_V3_VOCABULARY_BINDING_CHANGED")
    parent_d2, parent_context, binding = copy.deepcopy((parent_d2, parent_context, binding))
    refs = parent_d2["references"]
    ordered = sorted(refs, key=lambda row: row["reference_id"])
    pool.require(len(ordered) == 24 and Counter(row["role"] for row in ordered) ==
                 {"train-candidate": 18, "development-only": 6}, "CONTEXT_V3_NEW24_SOURCE_ROSTER")
    exclusion, old_ids, old_prompts = _history(parent_d2, parent_context)
    recipe = templates()
    for template in recipe:
        requested_text_preflight(template["prompt"], binding)
        pool.require(prompt_sha(template["prompt"]) not in old_prompts,
                     "CONTEXT_V3_HISTORICAL_PROMPT_REISSUE")
        text = pool.reference_text.normalize(template["prompt"])
        if template["intended_keyword_id"] is None:
            pool.require("小窝" not in text and all(keyword not in text for keyword in KEYWORDS),
                         "CONTEXT_V3_ORDINARY_REQUEST_CONTAINS_WAKE")
    cells = []
    # Interleave the three categories so the first bounded chunk is useful;
    # every fixed identity still receives exactly one request of each kind.
    for turn in range(3):
        for index, ref in enumerate(ordered):
            category = (turn + index) % 3
            if category < 2:
                kind = CLASSES[(index + category) % 4]
                template = next(row for row in recipe if row["context_class"] == kind and
                                row["intended_keyword_id"] == category + 1)
            else:
                template = recipe[8 + index % len(ORDINARY)]
            identity = digest({"namespace": SCHEMA, "d2_parent_sha256": parent_d2_sha,
                "context_parent_sha256": parent_context_sha, "batch_id": batch_id,
                "reference_id": ref["reference_id"], "template": template})
            seed = 1000000 + int(identity[:8], 16) % 1000000000
            payload = {"model_id": parent_d2["model_id"], "model_revision": parent_d2["model_revision"],
                       "reference_id": ref["reference_id"], "reference_wav_sha256": ref["wav_sha256"],
                       "reference_pcm_sha256": ref["pcm_sha256"], "prompt": template["prompt"],
                       "generation": GENERATION, "seed": seed}
            cells.append({"cell_id": "bulk-" + identity[:24],
                **{key: ref[key] for key in ("reference_id", "speaker_group_id", "source_family_id", "role")},
                **template, "seed": seed, "repetition": 0,
                "label_authority": "intended_only", "status": "NOT_RUN", "actual_text": None,
                "synthetic": True, "training_admitted": False, "qualification_allowed": False,
                "explicit_additive_revision": "new24-fulltext-v3",
                "same_requested_text_as_parent": False,
                "normalized_prompt_sha256": prompt_sha(template["prompt"]),
                "request_payload_sha256": digest(payload), "historical_prompt_reuse": False,
                "requested_text_preflight": requested_text_preflight(template["prompt"], binding)})
    pool.require(len(cells) == len({row["cell_id"] for row in cells}) ==
                 len({row["request_payload_sha256"] for row in cells}) == 72 and
                 not old_ids & {row["cell_id"] for row in cells}, "CONTEXT_V3_HISTORICAL_CELL_REISSUE")
    return {"schema": SCHEMA, "stage": STAGE, "batch_id": batch_id, "requested_cells": 72,
            "model_id": parent_d2["model_id"], "model_revision": parent_d2["model_revision"],
            "source_manifest_sha256": parent_d2["source_manifest_sha256"], "references": refs,
            "d2_parent": {"plan_sha256": parent_d2_sha, "plan": parent_d2},
            "context_parent": {"plan_sha256": parent_context_sha, "plan": parent_context},
            "speaker_registry": copy.deepcopy(parent_d2["speaker_registry"]), "vocabulary": binding,
            "history_exclusion": exclusion, "cells": cells, "generation": copy.deepcopy(GENERATION),
            "budgets": {"max_total_calls": 72, "max_calls_per_invocation": 8,
                        "max_retries": 0, "max_paid_usd": 0},
            "d1_cells_generated_again": 0, "d2_cells_generated_again": 0,
            "context_v1_cells_generated_again": 0, "context_v2_cells_generated_again": 0,
            "historical_prompts_generated_again": 0, "model_calls": 0,
            "actual_spoken_labels_verified": False, "requested_text_preflight_is_observed_label": False,
            "human_gold": False, "training_admitted": False, "qualification_allowed": False}


def plan(parent_d2, parent_d2_sha, parent_context, parent_context_sha, source_manifest,
         audio_root, vocabulary, *, batch_id):
    value = _assemble(parent_d2, parent_d2_sha, parent_context, parent_context_sha,
                      vocabulary_binding(vocabulary), batch_id=batch_id)
    return rebuild(value, source_manifest, audio_root)


def verify_plan(value):
    pool.require(type(value) is dict and value.get("schema") == SCHEMA and
                 type(value.get("context_parent")) is dict and type(value.get("d2_parent")) is dict and
                 set(value["context_parent"]) == set(value["d2_parent"]) == {"plan_sha256", "plan"},
                 "CONTEXT_V3_PLAN_SCHEMA")
    expected = _assemble(value["d2_parent"]["plan"], value["d2_parent"]["plan_sha256"],
                         value["context_parent"]["plan"], value["context_parent"]["plan_sha256"],
                         value["vocabulary"], batch_id=value["batch_id"])
    pool.require(pool.canonical(value) == pool.canonical(expected), "CONTEXT_V3_PLAN_CHANGED")
    return value


def rebuild(value, source_manifest, audio_root):
    verify_plan(value)
    pool.validate_sources(source_manifest, audio_root)
    pool.require(digest(source_manifest) == value["source_manifest_sha256"] and
                 source_manifest["references"] == value["references"], "CONTEXT_V3_SOURCE_MANIFEST_CHANGED")
    return _assemble(value["d2_parent"]["plan"], value["d2_parent"]["plan_sha256"],
                     value["context_parent"]["plan"], value["context_parent"]["plan_sha256"],
                     value["vocabulary"], batch_id=value["batch_id"])


def runner_recipe(value, selected_cells=None):
    verify_plan(value)
    ids = [row["cell_id"] for row in value["cells"]]
    selected = ids if selected_cells is None else selected_cells
    pool.require(type(selected) is list and bool(selected) and len(selected) <= 8 and
                 selected[0] in ids and selected == ids[ids.index(selected[0]):ids.index(selected[0]) + len(selected)],
                 "CONTEXT_V3_CONTIGUOUS_SELECTION")
    return {"selected_cells": list(selected), "max_calls": len(selected), "generation": copy.deepcopy(value["generation"])}


def run(value, source_manifest, audio_root, output, backend, *, max_new_calls, writer=media._write_waves):
    """Retain the native return before quality screening; never retry a claim."""
    expected = rebuild(value, source_manifest, audio_root)
    pool.require(pool.canonical(expected) == pool.canonical(value), "CONTEXT_V3_SOURCE_OR_PLAN_CHANGED")
    pool.require(type(max_new_calls) is int and 0 <= max_new_calls <= 8, "CONTEXT_V3_RUN_INVOCATION_BUDGET")
    root = Path(output)
    plan_sha = digest(value)
    lock = {"schema": "kws-bulk-clone-execution-lock-v1", "plan_sha256": plan_sha,
            "training_admitted": False, "model_id": value["model_id"]}
    if root.exists():
        pool.require(pool.read_json(root / "plan-lock.json") == lock, "CONTEXT_V3_EXECUTION_LOCK_DRIFT")
    pending = [cell for cell in value["cells"] if not (root / "cells" / cell["cell_id"] / "started.json").exists()]
    started = len(value["cells"]) - len(pending)
    pool.require(started <= value["budgets"]["max_total_calls"], "CONTEXT_V3_BUDGET_ALREADY_EXCEEDED")
    selected = pending[:min(max_new_calls, value["budgets"]["max_total_calls"] - started)]
    if selected:
        recipe = runner_recipe(value, [row["cell_id"] for row in selected])
        supplied = getattr(backend, "recipe", None)
        pool.require(type(supplied) is dict and all(supplied.get(k) == v for k, v in recipe.items()),
                     "CONTEXT_V3_BOUNDED_NATIVE_BACKEND_REQUIRED")
    if not root.exists():
        root.mkdir(parents=True)
        media._once(root / "plan-lock.json", lock)
    refs = {r["reference_id"]: r for r in value["references"]}
    invoked = 0
    for cell in selected:
        folder = root / "cells" / cell["cell_id"]
        pool.require(not folder.exists(), "CONTEXT_V3_UNKNOWN_CELL_DIRECTORY")
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
                         "CONTEXT_V3_WRITER_CONTRACT")
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
    return {"schema": "kws-context-supplement-v3-summary-v1", "status": "PLANNED_ONLY",
            "plan_sha256": digest(value), "parent_d2_plan_sha256": value["d2_parent"]["plan_sha256"],
            "parent_context_plan_sha256": value["context_parent"]["plan_sha256"],
            "vocabulary_sha256": value["vocabulary"]["canonical_sha256"],
            "new_requested_cells": len(value["cells"]), "source_families": len(value["references"]),
            "requested_context_counts": dict(Counter(c["context_class"] for c in value["cells"])),
            "requested_keyword_counts": dict(Counter(str(c["intended_keyword_id"]) for c in value["cells"])),
            "requested_split_counts": dict(Counter(c["role"] for c in value["cells"])),
            "all_requested_fulltext_in_fixed_vocabulary": True, "historical_prompt_reuse": 0,
            "new_model_calls": 0, "observed_usable_cells": None, "training_admitted": False}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-d2-plan", "parent-d2-plan-sha256", "parent-context-plan",
                 "parent-context-plan-sha256", "sources", "audio-root", "vocabulary", "batch-id"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args(argv)
    value = plan(pool.read_json(args.parent_d2_plan, MAX_PLAN_BYTES), args.parent_d2_plan_sha256,
        pool.read_json(args.parent_context_plan, MAX_PLAN_BYTES), args.parent_context_plan_sha256,
        pool.read_json(args.sources), args.audio_root, Path(args.vocabulary).read_bytes(), batch_id=args.batch_id)
    if args.output is not None:
        output = args.output.absolute()
        pool.require(output.parent.is_dir() and not output.exists() and
                     not any(p.is_symlink() for p in (output, *output.parents)), "CONTEXT_V3_NEW_PLAN_REQUIRED")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(pool.canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(summary(value), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
