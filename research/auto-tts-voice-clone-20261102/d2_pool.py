"""Append-only D2 clone requests, retaining the complete frozen D1 plan.

The two increments are executable independently: the original eight sources
receive 32 additional requests each; 24 new sources receive 64 each. D1 cells
are inherited as evidence, never submitted to the generator again. Requested
text and categories do not establish observed labels or usable audio counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import bulk_pool as pool
from kws_data_contract import reference_text, source_registry

SCHEMA = "kws-d2-increment-plan-v1"
MAX_PLAN_BYTES = 8 * 1024 * 1024
EXISTING = "existing-eight"
NEW = "new-twenty-four"
STAGES = (EXISTING, NEW)
COUNTS = {
    EXISTING: {"kw1": 8, "kw2": 8, "ordinary-negative": 12,
               "partial-negative": 4, "near-negative": 0},
    NEW: {"kw1": 16, "kw2": 16, "ordinary-negative": 20,
          "partial-negative": 8, "near-negative": 4},
}
TARGET_COUNTS = {"kw1": 16, "kw2": 16, "ordinary-negative": 20,
                 "partial-negative": 8, "near-negative": 4}
ORDINARY = (
    "今天早上天气很好", "把窗帘拉开一点", "桌子上放着一本书", "厨房里的水烧开了",
    "我们准备出门散步", "等一下再播放音乐", "门口的包裹已经到了", "记得给花浇水",
    "客厅的灯还开着", "沙发旁边有一只猫", "晚饭以后收拾房间", "明天上午有个会议",
    "今天晚上吃什么", "请把灯打开", "我想听一首音乐", "现在外面下雨了",
    "电视声音调低一点", "电池还剩多少电量", "请慢一点向前走", "现在可以休息了",
)
PARTIAL = ("小窝", "你好", "你好小", "小窝你好")
NEAR = ("你好小屋", "小沃小沃", "你好小王", "小屋小屋")
# Reference content observations may improve after D1. The identity, actual
# reference bytes, original ancestry, split and publisher text cannot change.
REFERENCE_PINS = (
    "reference_id", "speaker_group_id", "source_family_id", "dataset_id",
    "source_type", "role", "wav_sha256", "pcm_sha256", "parent_source_sha256",
    "sample_rate_hz", "frames", "reference_text", "source_license",
    "source_terms_sha256", "rights_record_sha256",
)
ASSIGNMENT_KEYS = ("dataset_id", "speaker_group_id", "source_family_id", "role")


def digest(value):
    return hashlib.sha256(pool.canonical(value)).hexdigest()


def parent_plan(value, expected_sha):
    pool._sha(expected_sha)
    pool.require(type(value) is dict and value.get("cell_layout") == pool.D1_LAYOUT,
                 "D2_REQUIRES_FROZEN_D1_PARENT")
    pool.verify_plan(value)
    pool.require(digest(value) == expected_sha, "D2_PARENT_PLAN_CHANGED")
    source = {"schema": pool.SCHEMA, "references": value["references"],
              "training_admitted": False, "qualification_allowed": False}
    pool.require(digest(source) == value["source_manifest_sha256"],
                 "D2_PARENT_SOURCE_MANIFEST_CHANGED")
    return value


def _all_references(refs, parent, stage):
    pool.require(stage in STAGES and type(refs) is list,
                 "D2_STAGE_OR_REFERENCES")
    anchors = {ref["reference_id"]: ref for ref in parent["references"]}
    expected_count = 8 if stage == EXISTING else 24
    pool.require(len(refs) == expected_count and
                 len({ref["reference_id"] for ref in refs}) == expected_count,
                 "D2_STAGE_REFERENCE_COUNT")
    if stage == EXISTING:
        pool.require({ref["reference_id"] for ref in refs} == set(anchors),
                     "D2_ORIGINAL_EIGHT_REQUIRED")
        for ref in refs:
            pool.require(all(ref.get(key) == anchors[ref["reference_id"]].get(key)
                             for key in REFERENCE_PINS), "D2_ORIGINAL_SOURCE_CHANGED")
        combined = refs
    else:
        combined = list(parent["references"]) + refs
        pool.require(Counter(ref["role"] for ref in refs) ==
                     {"train-candidate": 18, "development-only": 6},
                     "D2_NEW_SOURCES_REQUIRE_EIGHTEEN_TRAIN_SIX_DEVELOPMENT")
    count = 8 if stage == EXISTING else 32
    for key in ("reference_id", "source_family_id", "speaker_group_id"):
        pool.require(len({ref[key] for ref in combined}) == count,
                     "D2_DUPLICATE_OR_OLD_SOURCE:" + key)
    roles = pool.SourceRoles()
    owners = {}
    for ref in combined:
        for key in ("reference_id", "speaker_group_id", "source_family_id", "dataset_id"):
            pool._id(ref[key])
        reference_text.validate_reference(ref)
        for key in ("wav_sha256", "pcm_sha256", "source_terms_sha256", "rights_record_sha256"):
            pool._sha(ref[key])
        roles.add(ref["role"], family=ref["source_family_id"],
                  speaker=ref["speaker_group_id"], pcm_sha256=ref["pcm_sha256"],
                  wav_sha256=ref["wav_sha256"], parent_source_sha256=ref["parent_source_sha256"])
        for key in ("wav_sha256", "pcm_sha256", "parent_source_sha256"):
            if ref[key] is None:
                continue
            pool._sha(ref[key])
            owner = owners.setdefault((key, ref[key]), ref["source_family_id"])
            pool.require(owner == ref["source_family_id"],
                         "D2_REFERENCE_ANCESTRY_ALIAS:" + key)
    return combined


def registry_binding(registry, refs, parent, stage):
    """Read the persistent registry; never assign identities or splits here."""
    source_registry.validate(registry)
    combined = _all_references(refs, parent, stage)
    actual = {tuple(row[key] for key in ASSIGNMENT_KEYS) for row in registry["entries"]}
    selected = [{key: ref[key] for key in ASSIGNMENT_KEYS} for ref in combined]
    pool.require(all(tuple(row[key] for key in ASSIGNMENT_KEYS) in actual for row in selected),
                 "D2_SOURCE_NOT_IN_PERSISTENT_REGISTRY")
    return {"registry_id": registry["registry_id"], "sha256": digest(registry),
            "assignments": sorted(selected, key=lambda row: row["source_family_id"])}


def _check_registry_binding(binding, combined):
    pool.require(type(binding) is dict and set(binding) ==
                 {"registry_id", "sha256", "assignments"}, "D2_REGISTRY_BINDING")
    pool._id(binding["registry_id"])
    pool._sha(binding["sha256"])
    expected = sorted(({key: ref[key] for key in ASSIGNMENT_KEYS} for ref in combined),
                      key=lambda row: row["source_family_id"])
    pool.require(binding["assignments"] == expected, "D2_REGISTRY_ASSIGNMENTS_CHANGED")


def prompts(stage):
    """Deficit scheduling balances the prefix without inventing extra labels."""
    pool.require(stage in STAGES, "D2_STAGE")
    counts = COUNTS[stage]
    used = dict.fromkeys(counts, 0)
    order = ("kw1", "ordinary-negative", "kw2", "partial-negative", "near-negative")
    text_by_class = {"kw1": ("你好小窝", "你好，小窝"),
                     "kw2": ("小窝小窝", "小窝，小窝"),
                     "ordinary-negative": ORDINARY,
                     "partial-negative": PARTIAL, "near-negative": NEAR}
    total = sum(counts.values())
    result = []
    for position in range(total):
        available = [key for key in order if used[key] < counts[key]]
        kind = max(available, key=lambda key: counts[key] * (position + 1) - used[key] * total)
        text = text_by_class[kind][used[kind] % len(text_by_class[kind])]
        positive = kind in ("kw1", "kw2")
        result.append((text, "positive-intent" if positive else
                       "speech-negative-intent" if kind == "ordinary-negative" else
                       "hard-negative-intent", int(kind[-1]) if positive else None,
                       "full-positive" if positive else kind))
        used[kind] += 1
    return result


def _cells(refs, parent_sha, stage, batch_id):
    recipe = prompts(stage)
    ordered = sorted(refs, key=lambda ref: ref["reference_id"])
    cells = []
    for turn in range(len(recipe)):
        for offset in range(len(ordered)):
            index = (turn + offset) % len(ordered)
            ref = ordered[index]
            prompt_index = (turn + index) % len(recipe)
            text, kind, keyword, category = recipe[prompt_index]
            identity = digest({"batch_id": batch_id, "stage": stage,
                               "parent_plan_sha256": parent_sha,
                               "reference_id": ref["reference_id"],
                               "prompt_index": prompt_index, "text": text})
            cells.append({
                "cell_id": "bulk-" + identity[:24], "reference_id": ref["reference_id"],
                "speaker_group_id": ref["speaker_group_id"],
                "source_family_id": ref["source_family_id"], "role": ref["role"],
                "prompt": text, "intended_kind": kind, "intended_keyword_id": keyword,
                "generation_category": category, "prompt_index": prompt_index,
                "seed": 1000000 + int(identity[:8], 16) % 1000000000,
                "repetition": 0, "status": "NOT_RUN", "label_authority": "intended_only",
                "synthetic": True, "training_admitted": False, "qualification_allowed": False,
            })
    return cells


def _assemble(manifest, parent, parent_sha, binding, *, stage, batch_id, revision, max_calls):
    parent_plan(parent, parent_sha)
    refs = manifest["references"]
    combined = _all_references(refs, parent, stage)
    _check_registry_binding(binding, combined)
    pool._id(batch_id, "BATCH_ID")
    pool.require(type(revision) is str and pool._HEX40.fullmatch(revision), "PINNED_BASE_REVISION")
    cells = _cells(refs, parent_sha, stage, batch_id)
    pool.require(not ({cell["cell_id"] for cell in cells} &
                     {cell["cell_id"] for cell in parent["cells"]}), "D2_MUST_NOT_REISSUE_D1_CELL")
    ceiling = len(cells) if max_calls is None else max_calls
    pool.require(type(ceiling) is int and 1 <= ceiling <= len(cells), "D2_CALL_BUDGET")
    return {
        "schema": SCHEMA, "stage": stage, "batch_id": batch_id,
        "model_id": pool.MODEL, "model_revision": revision,
        "source_manifest_sha256": digest(manifest), "references": refs,
        "d1_parent": {"plan_sha256": parent_sha, "plan": parent},
        "speaker_registry": binding, "cells": cells,
        "budgets": {"max_total_calls": ceiling, "max_calls_per_invocation": 64,
                    "max_retries": 0, "max_paid_usd": 0},
        "model_calls": 0, "source_rights_authenticated": False,
        "actual_spoken_labels_verified": False, "training_admitted": False,
        "qualification_allowed": False,
    }


def plan(manifest, audio_root, *, parent, parent_sha, registry, stage, batch_id,
         revision=None, max_calls=None):
    parent_plan(parent, parent_sha)
    refs = pool.validate_sources(manifest, audio_root)
    binding = registry_binding(registry, refs, parent, stage)
    return _assemble(manifest, parent, parent_sha, binding, stage=stage,
                     batch_id=batch_id, revision=revision or parent["model_revision"],
                     max_calls=max_calls)


def verify_plan(value):
    pool.require(type(value) is dict and set(value) == {
                 "schema", "stage", "batch_id", "model_id", "model_revision", "source_manifest_sha256",
                 "references", "d1_parent", "speaker_registry", "cells", "budgets", "model_calls",
                 "source_rights_authenticated", "actual_spoken_labels_verified", "training_admitted",
                 "qualification_allowed"} and value.get("schema") == SCHEMA and
                 type(value.get("d1_parent")) is dict and set(value["d1_parent"]) ==
                 {"plan_sha256", "plan"} and type(value.get("budgets")) is dict and
                 "max_total_calls" in value["budgets"], "D2_PLAN_SCHEMA")
    manifest = {"schema": pool.SCHEMA, "references": value["references"],
                "training_admitted": False, "qualification_allowed": False}
    expected = _assemble(manifest, value["d1_parent"]["plan"], value["d1_parent"]["plan_sha256"],
                         value["speaker_registry"], stage=value["stage"], batch_id=value["batch_id"],
                         revision=value["model_revision"], max_calls=value["budgets"]["max_total_calls"])
    pool.require(pool.canonical(value) == pool.canonical(expected), "D2_PLAN_CHANGED")
    return value


def rebuild(value, manifest, audio_root):
    verify_plan(value)
    pool.validate_sources(manifest, audio_root)
    return _assemble(manifest, value["d1_parent"]["plan"], value["d1_parent"]["plan_sha256"],
                     value["speaker_registry"], stage=value["stage"], batch_id=value["batch_id"],
                     revision=value["model_revision"], max_calls=value["budgets"]["max_total_calls"])


def requested_counts(cells):
    return dict(Counter("kw" + str(cell["intended_keyword_id"])
                        if cell["generation_category"] == "full-positive" else
                        cell["generation_category"] for cell in cells))


def summary(value):
    verify_plan(value)
    return {"schema": "kws-d2-plan-summary-v1", "status": "PLANNED_ONLY",
            "stage": value["stage"], "plan_sha256": digest(value),
            "parent_plan_sha256": value["d1_parent"]["plan_sha256"],
            "inherited_d1_cells": 256, "new_requested_cells": len(value["cells"]),
            "new_requested_mix": requested_counts(value["cells"]),
            "target_cumulative_cells": 2048, "target_reference_identities": 32,
            "other_increment_requested_cells": 1536 if value["stage"] == EXISTING else 256,
            "observed_usable_cells": None, "new_model_calls": 0,
            "training_admitted": False, "qualification_allowed": False}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage", required=True, choices=STAGES)
    for name in ("sources", "audio-root", "parent-plan", "speaker-registry"):
        cli.add_argument("--" + name, required=True, type=Path)
    cli.add_argument("--parent-plan-sha256", required=True)
    cli.add_argument("--batch-id", required=True)
    cli.add_argument("--model-revision")
    cli.add_argument("--max-total-calls", type=int)
    cli.add_argument("--output", type=Path, help="Create a new private plan; omission is a read-only preview.")
    args = cli.parse_args(argv)
    value = plan(pool.read_json(args.sources), args.audio_root,
                 parent=pool.read_json(args.parent_plan, MAX_PLAN_BYTES),
                 parent_sha=args.parent_plan_sha256,
                 registry=source_registry.read_json(args.speaker_registry),
                 stage=args.stage, batch_id=args.batch_id, revision=args.model_revision,
                 max_calls=args.max_total_calls)
    if args.output is not None:
        path = args.output.absolute()
        pool.require(path.parent.is_dir() and not path.exists() and
                     not any(p.is_symlink() for p in (path, *path.parents)), "D2_NEW_OUTPUT_REQUIRED")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(pool.canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(summary(value), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("D2_EXPANSION_HOLD: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
