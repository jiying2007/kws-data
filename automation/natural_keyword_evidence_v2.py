#!/usr/bin/env python3
"""Read-only keyword evidence layered on the existing natural-ASR verifier.

No recognizers, training, event-time inference, or admission are implemented.
The CLI always replays the upstream native evidence; report.json alone is not
accepted. This is a proposed research protocol requiring acoustic validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any, Callable

SCHEMA = "kws-natural-keyword-evidence-v2"
POLICY = "three-clean-positive-or-inherited-fulltext-v2"
MANIFEST_SCHEMA = "kws-natural-keyword-evidence-manifest-v2"
UPSTREAM_SCHEMA = "kws-natural-asr-evidence-view-v1"
MODELS = ("qwen06", "sensevoice", "whisper")
KEYWORDS = ("你好小窝", "小窝小窝")
MAX_ROWS = 1024
MAX_JSON = 32 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
BOUNDARY = {
    "human_gold": False,
    "training_admitted": False,
    "independent_test": False,
    "natural_far_available": False,
    "accepted_negative_seconds": 0,
    "event_times": None,
    "new_model_calls": 0,
    "new_optimizer_steps": 0,
    "acoustic_target_presence": "unknown",
    "protocol_acoustic_validation": "NOT_RUN",
}
FILES = ("report.json", "audit-blind.json", "audit-owner.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("KEYWORD_V2_" + message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(value: Any) -> bool:
    return type(value) is str and SHA.fullmatch(value) is not None


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def loads(raw: bytes) -> Any:
    def bad_constant(_: str) -> None:
        raise ValueError("KEYWORD_V2_NONFINITE_JSON")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=bad_constant)


def real_path(path: Path) -> Path:
    path = path.absolute()
    require(".." not in path.parts, "NONCANONICAL_PATH")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK")
    return path.resolve()


def read_bytes(path: Path, limit: int = MAX_JSON) -> bytes:
    path = real_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= limit, "REGULAR_SINGLE_LINK_BOUNDED_FILE")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        require(len(data) == before.st_size and
                all(getattr(before, k) == getattr(after, k) for k in fields), "FILE_CHANGED")
        return data


def record(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": digest(raw)}


def save_new(path: Path, raw: bytes) -> None:
    path = real_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_observations(observations: Any) -> list[dict[str, Any]]:
    require(type(observations) is list and len(observations) == 3, "THREE_MODELS_REQUIRED")
    require(all(type(o) is dict for o in observations), "OBSERVATION_OBJECT")
    require([o.get("model") for o in observations] == list(MODELS), "EXACT_MODEL_ROSTER")
    for observed in observations:
        require(observed.get("requested") is True and observed.get("status") == "complete",
                "COMPLETE_REQUESTED_OBSERVATION")
        require(type(observed.get("usable")) is bool and
                type(observed.get("decoder_terminated")) is bool, "QUALITY_BOOLEAN")
        flags = observed.get("quality_flags")
        require(type(flags) is list and all(type(x) is str for x in flags), "QUALITY_FLAGS")
        require(type(observed.get("normalized_text")) is str and
                len(observed["normalized_text"]) <= 65536, "NORMALIZED_TEXT")
    return observations


def project_keywords(observations: Any, old_decision: Any) -> list[dict[str, Any]]:
    """Pure projection, NOT a native-evidence authentication entry point.

    Inputs must come from the existing verifier. Synthetic inputs are suitable
    for software unit tests only. Absence without old full-text quorum stays
    unknown; no event timestamp or acoustic truth is inferred.
    """
    observed = validate_observations(observations)
    require(type(old_decision) is dict, "UPSTREAM_DECISION")
    require(type(old_decision.get("full_text_quorum")) is bool and
            type(old_decision.get("ctc_mask")) is bool, "UPSTREAM_MASK_BOOLEAN")
    require(old_decision["ctc_mask"] is old_decision["full_text_quorum"], "UPSTREAM_CTC_QUORUM")
    masks, targets = old_decision.get("keyword_mask"), old_decision.get("keyword_targets")
    require(type(masks) is list and len(masks) == 2 and all(type(x) is bool for x in masks),
            "UPSTREAM_KEYWORD_MASK")
    require(type(targets) is list and len(targets) == 2, "UPSTREAM_KEYWORD_TARGETS")
    for mask, target in zip(masks, targets):
        require((mask and type(target) is int and target in (0, 1)) or
                (not mask and target is None), "UNKNOWN_IS_NOT_NEGATIVE")
    require(all(x is old_decision["full_text_quorum"] for x in masks), "UPSTREAM_QUORUM_MASK")
    clean = all(o["usable"] and o["decoder_terminated"] and not o["quality_flags"] and
                bool(o["normalized_text"]) for o in observed)
    require(not old_decision["full_text_quorum"] or clean, "DIRTY_INHERITED_QUORUM")
    rows = []
    for index, keyword in enumerate(KEYWORDS):
        votes = [keyword in o["normalized_text"] for o in observed]
        require(not masks[index] or all(int(v) == targets[index] for v in votes),
                "INHERITED_TARGET_DISAGREES")
        state, reason, target, candidate_mask = "unknown", "uncertain_keyword_presence", None, False
        if masks[index]:
            target = targets[index]
            state = "present" if target else "absent"
            reason, candidate_mask = "inherited_frozen_fulltext_decision", True
        elif not clean:
            reason = "three_clean_nonempty_observations_required"
        elif all(votes):
            state, reason, target, candidate_mask = (
                "present", "three_clean_literal_positive_votes", 1, True)
        elif not any(votes):
            reason = "three_non_detections_do_not_prove_absence"
        else:
            reason = "recognizers_disagree_on_keyword_presence"
        rows.append({
            "keyword_id": index + 1, "keyword": keyword,
            "machine_candidate_state": state, "candidate_target": target,
            "candidate_mask": candidate_mask, "reason": reason,
            "literal_votes": dict(zip(MODELS, votes)),
            "source": "inherited" if masks[index] else "keyword-v2-proposal",
            "loss_weight": None, "training_mask": False,
            "acoustic_state": "unknown", "event_times": None,
        })
    return rows


def project_report(upstream: Any, manifest_sha256: str, *,
                   normalize: Callable[[str], str]) -> dict[str, Any]:
    """Project an already native-verified report, preserving its label decisions."""
    require(sha(manifest_sha256), "UPSTREAM_MANIFEST_PIN")
    require(type(upstream) is dict and upstream.get("schema") == UPSTREAM_SCHEMA, "UPSTREAM_SCHEMA")
    boundary = upstream.get("boundary")
    require(type(boundary) is dict and boundary.get("training_admitted") is False and
            boundary.get("independent_test") is False and boundary.get("natural_far_available") is False,
            "UPSTREAM_SCOPE")
    inputs = upstream.get("rows")
    require(type(inputs) is list and 1 <= len(inputs) <= MAX_ROWS, "ROW_BOUND")
    result, seen, roles, intervals, parent_metadata = [], set(), {}, set(), {}
    counts = Counter()
    for row in inputs:
        require(type(row) is dict, "ROW_OBJECT")
        source = row.get("source_id")
        require(type(source) is str and 0 < len(source) <= 256 and source not in seen, "SOURCE_IDENTITY")
        seen.add(source)
        split = row.get("split")
        require(split in ("train", "development"), "NO_TEST_RELABEL")
        for key in ("parent_pcm_sha256", "window_pcm_sha256", "window_wav_sha256"):
            require(sha(row.get(key)), "AUDIO_BINDING")
        family = row.get("source_family_id")
        require(type(family) is str and 0 < len(family) <= 256, "FAMILY_IDENTITY")
        for key in (("family", family), ("pcm", row["parent_pcm_sha256"]),
                    ("pcm", row["window_pcm_sha256"]), ("wav", row["window_wav_sha256"])):
            require(roles.setdefault(key, split) == split, "CROSS_ROLE_LEAK")
        for key in ("dataset_id", "session_id"):
            require(type(row.get(key)) is str and row[key], "SESSION_IDENTITY")
        identity = (row["dataset_id"], row["session_id"], family, split)
        require(parent_metadata.setdefault(row["parent_pcm_sha256"], identity) == identity,
                "PARENT_METADATA_CONFLICT")
        start, end, rate = row.get("start_sample"), row.get("end_sample"), row.get("sample_rate_hz")
        require(type(start) is int and type(end) is int and type(rate) is int and
                rate == 16000 and 0 <= start < end and end - start <= 30 * rate, "WINDOW_INTERVAL")
        interval = (row["parent_pcm_sha256"], start, end)
        require(interval not in intervals, "DUPLICATE_PARENT_INTERVAL")
        intervals.add(interval)
        originals = row.get("observations")
        require(type(originals) is list and all(type(o) is dict and type(o.get("text")) is str
                                               for o in originals), "NATIVE_TEXT")
        normalized = [{**o, "normalized_text": normalize(o["text"])} for o in originals]
        keywords = project_keywords(normalized, row.get("decision"))
        for keyword in keywords:
            counts[str(keyword["keyword_id"]) + ":" + keyword["machine_candidate_state"]] += 1
        result.append({
            "source_id": source, "split": split, "source_family_id": family,
            "dataset_id": row["dataset_id"], "session_id": row["session_id"],
            "parent_pcm_sha256": row["parent_pcm_sha256"],
            "window_pcm_sha256": row["window_pcm_sha256"], "window_wav_sha256": row["window_wav_sha256"],
            "start_sample": start, "end_sample": end, "sample_rate_hz": rate,
            "upstream_row_sha256": digest(canonical(row)),
            "original_fulltext_decision": loads(canonical(row["decision"])),
            "keyword_candidates": keywords,
        })
    return {"schema": SCHEMA, "policy": POLICY,
            "upstream_manifest_sha256": manifest_sha256,
            "upstream_report_sha256": digest(canonical(upstream)),
            "rows": result, "statistics": {"rows": len(result), "by_keyword_state": dict(sorted(counts.items()))},
            "boundary": dict(BOUNDARY)}


def make_audit_plan(report: dict[str, Any], *, salt: str, limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministic stratified selection; reviewer payload has no transcripts.

    This is a sampling proposal, not a completed blind annotation or a fresh
    independent test. The owner mapping must not be supplied to the reviewer.
    """
    require(type(salt) is str and 16 <= len(salt) <= 256, "AUDIT_SALT")
    require(type(limit) is int and 1 <= limit <= 128, "AUDIT_LIMIT")
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        state = "|".join(k["machine_candidate_state"] for k in row["keyword_candidates"])
        strata.setdefault(state, []).append(row)
    def rank(row: dict[str, Any]) -> str:
        return digest(canonical([salt, row["source_id"], row["window_pcm_sha256"]]))
    for pool in strata.values():
        pool.sort(key=rank)
    selected = []
    # Equal-stratum round robin is intentional, not a population prevalence estimator.
    while len(selected) < limit and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < limit:
                selected.append((key, strata[key].pop(0)))
    blind, owner = [], []
    for stratum, row in selected:
        code = digest(canonical(["blind", salt, row["source_id"], row["window_pcm_sha256"]]))
        blind.append({"blind_id": code, "audio_wav_sha256": row["window_wav_sha256"],
                      "sample_rate_hz": row["sample_rate_hz"],
                      "sample_count": row["end_sample"] - row["start_sample"]})
        owner.append({"blind_id": code, "source_id": row["source_id"], "stratum": stratum,
                      "source_family_id": row["source_family_id"], "split": row["split"]})
    blind.sort(key=lambda x: x["blind_id"])
    owner.sort(key=lambda x: x["blind_id"])
    common = {"status": "planned_not_annotated", "human_gold": False,
              "independent_test": False, "selected_rows": len(blind)}
    return ({"schema": "kws-keyword-v2-blind-review-v1", **common, "rows": blind},
            {"schema": "kws-keyword-v2-review-owner-v1", **common, "salt": salt,
             "limit": limit, "selection": "deterministic-equal-stratum-round-robin",
             "prevalence_estimate_allowed": False, "rows": owner})


def _upstream_api():
    # Import only the repository's existing verifier. No arbitrary module path,
    # callback, executable, external code or URL is accepted from a JSON recipe.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from automation import active_natural_asr_weak_view as natural
    require(Path(natural.__file__).resolve() == root / "automation/active_natural_asr_weak_view.py",
            "MIXED_UPSTREAM_IMPORT")
    natural.verify_loaded_sources()
    return natural


def _replay(upstream: Path, upstream_manifest: str) -> dict[str, Any]:
    natural = _upstream_api()
    verified = natural.verify_view(upstream, expected_manifest_sha256=upstream_manifest)
    return project_report(verified, upstream_manifest, normalize=natural.weak.normalize)


def _payloads(report: dict[str, Any], salt: str, limit: int) -> dict[str, bytes]:
    blind, owner = make_audit_plan(report, salt=salt, limit=limit)
    return {"report.json": canonical(report) + b"\n", "audit-blind.json": canonical(blind) + b"\n",
            "audit-owner.json": canonical(owner) + b"\n"}


def create(upstream: Path, upstream_manifest: str, output: Path, *, salt: str, limit: int) -> dict[str, Any]:
    require(sha(upstream_manifest), "UPSTREAM_MANIFEST_PIN")
    upstream, output = real_path(upstream), real_path(output)
    root = Path(__file__).resolve().parents[1]
    require(not output.exists() and output.parent.is_dir() and output != root and
            root not in output.parents and output != upstream and upstream not in output.parents,
            "NEW_OUTPUT_OUTSIDE_INPUT_AND_CHECKOUT")
    report = _replay(upstream, upstream_manifest)  # Authenticate native evidence before any output.
    payloads = _payloads(report, salt, limit)
    source = record(read_bytes(Path(__file__)))
    output.mkdir(mode=0o700)
    for name, raw in payloads.items():
        save_new(output / name, raw)
    manifest = {"schema": MANIFEST_SCHEMA, "policy": POLICY,
                "upstream_manifest_sha256": upstream_manifest, "producer": source,
                "files": {name: record(raw) for name, raw in payloads.items()},
                "audit_salt": salt, "audit_limit": limit}
    save_new(output / "MANIFEST.json", canonical(manifest) + b"\n")
    return {"status": "created", "statistics": report["statistics"], "boundary": dict(BOUNDARY)}


def verify(upstream: Path, upstream_manifest: str, package: Path, expected_manifest: str) -> dict[str, Any]:
    require(sha(upstream_manifest) and sha(expected_manifest), "MANIFEST_PIN")
    package = real_path(package)
    raw = read_bytes(package / "MANIFEST.json", 65536)
    require(digest(raw) == expected_manifest, "MANIFEST_CHANGED")
    manifest = loads(raw)
    require(type(manifest) is dict and set(manifest) == {
        "schema", "policy", "upstream_manifest_sha256", "producer", "files", "audit_salt", "audit_limit"},
        "MANIFEST_SCHEMA")
    require(manifest["schema"] == MANIFEST_SCHEMA and manifest["policy"] == POLICY and
            manifest["upstream_manifest_sha256"] == upstream_manifest and
            manifest["producer"] == record(read_bytes(Path(__file__))), "MANIFEST_IDENTITY")
    require(type(manifest["files"]) is dict and set(manifest["files"]) == set(FILES) and
            {p.name for p in package.iterdir()} == set(FILES) | {"MANIFEST.json"}, "FILE_ROSTER")
    for name, pin in manifest["files"].items():
        require(pin == record(read_bytes(package / name)), "PAYLOAD_CHANGED")
    report = _replay(upstream, upstream_manifest)  # Rehashing a fake report cannot pass this replay.
    for name, actual in _payloads(report, manifest["audit_salt"], manifest["audit_limit"]).items():
        require(read_bytes(package / name) == actual, "NOT_DERIVED_FROM_NATIVE_EVIDENCE")
    return {"status": "verified", "statistics": report["statistics"], "boundary": dict(BOUNDARY)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--upstream-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-salt")
    parser.add_argument("--audit-limit", type=int, default=32)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    try:
        require(platform.python_implementation() == "CPython" and
                sys.version_info[:3] == (3, 12, 16), "PINNED_CPYTHON_3_12_16_REQUIRED")
        if args.command == "create":
            require(args.output is not None and args.audit_salt is not None and
                    args.package is None and args.manifest_sha256 is None, "CREATE_ARGUMENTS")
            result = create(args.upstream, args.upstream_manifest_sha256, args.output,
                            salt=args.audit_salt, limit=args.audit_limit)
        else:
            require(args.package is not None and args.manifest_sha256 is not None and
                    args.output is None and args.audit_salt is None, "VERIFY_ARGUMENTS")
            result = verify(args.upstream, args.upstream_manifest_sha256, args.package, args.manifest_sha256)
        print(canonical(result).decode())  # No transcript/subject identity on stdout.
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print("NATURAL_KEYWORD_V2_REJECTED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
