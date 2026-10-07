#!/usr/bin/env python3
"""Offline archive verification and optional safe restoration; no inference."""
import argparse
import collections
import hashlib
import json
import pathlib
import stat
import unicodedata
import zipfile


ROOT = pathlib.Path(__file__).resolve().parent
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 20 * 1024 * 1024


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def git_blob_sha1(body):
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def normalize(text):
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError("Text label must be a string or null")
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def safe_relative(name):
    p = pathlib.PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("Unsafe archive path")
    return p


def read_raw(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member")
        if sum(e.file_size for e in entries) > MAX_TOTAL:
            raise ValueError("Expanded artifact exceeds reviewed bound")
        raw = {}
        for entry in entries:
            safe_relative(entry.filename)
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or not entry.filename.endswith(".json"):
                raise ValueError("Unexpected member type")
            if entry.file_size > MAX_FILE:
                raise ValueError("Artifact member exceeds reviewed bound")
            body = archive.read(entry)
            json.loads(body)
            raw[entry.filename] = body
    freeze = json.loads(raw["artifact-freeze.json"])
    if set(raw) != set(freeze["files"]) | {"artifact-freeze.json"}:
        raise ValueError("Artifact freeze membership mismatch")
    for name, expected in freeze["files"].items():
        if sha256(raw[name]) != expected:
            raise ValueError("Artifact member hash mismatch")
    return raw


def verify_summary(summary, failure, recovery):
    if summary["scope"] != "exposed_calibration_regression_only":
        raise ValueError("Unexpected scientific scope")
    rows = summary["rows"]
    if len(rows) != 30 or len({r["opaque_id"] for r in rows}) != 30:
        raise ValueError("Fixed30 row identity mismatch")
    old_status = json.loads(failure["final-status.json"])["models"]
    new_status = json.loads(recovery["final-status.json"])["models"]
    if any(len(v) != 30 or any(r["status"] != "not_run" for r in v) for v in old_status.values()):
        raise ValueError("Predecessor must remain all not_run")
    models = ("sensevoice", "qwen06")
    outputs = {}
    for model in models:
        status = new_status[model]
        if len(status) != 30 or any(r["status"] != "success" or r["completeness"] != "complete" for r in status):
            raise ValueError("Recovery execution status mismatch")
        outcome = json.loads(recovery["raw/" + model + "/outcomes.json"])
        outputs[model] = {r["opaque_id"]: r for r in outcome}
        started = [json.loads(body) for name, body in recovery.items()
                   if name.startswith("raw/" + model + "/") and name.endswith(".started.json")]
        if len(started) != 30 or len({r["opaque_id"] for r in started}) != 30:
            raise ValueError("Unique model attempt count mismatch")
    counts = collections.Counter()
    agreement = collections.Counter()
    for row in rows:
        actual = normalize(row["human_text_label"])
        plan = normalize(row["original_plan_text"])
        if not plan:
            raise ValueError("Original plan must be sourced, not inferred or null")
        observed = {}
        for model in models:
            raw = outputs[model][row["opaque_id"]]
            if raw["wav_sha256"] != row["wav_sha256"]:
                raise ValueError("Projection WAV binding mismatch")
            observed[model] = normalize(raw["raw_text"])
            if observed[model] != row["asr_normalized_text"][model]:
                raise ValueError("Normalized ASR projection mismatch")
        a, b = (observed[m] for m in models)
        plan_pass = bool(a and b and a == b == plan)
        actual_support = bool(actual and a and b and a == b == actual)
        if plan_pass != row["both_equal_original_plan"] or actual_support != row["both_support_human_words"]:
            raise ValueError("Derived lexical-rule mismatch")
        counts["original_plan_lexical_pass"] += plan_pass
        counts["original_plan_lexical_reject_or_dispute"] += not plan_pass
        counts["human_actual_lexical_supported"] += actual_support
        counts["actual_only_supported_but_plan_rejected"] += bool(actual_support and not plan_pass)
        counts["plan_pass_human_word_mismatch"] += bool(plan_pass and actual is not None and actual != plan)
        counts["plan_pass_human_inaudible"] += bool(plan_pass and actual is None)
        category = ("empty_output_reject_or_dispute" if not a or not b else
                    "asr_disagreement_reject_or_dispute" if a != b else
                    "both_equal_plan_lexical_pass" if plan_pass else
                    "agree_but_not_plan_reject_or_dispute")
        if row["plan_rule_category"] != category or row["training_or_kws_admission"] is not False:
            raise ValueError("Projection category or admission boundary changed")
        if row["human_audibility"] == "words_transcribed":
            counts["human_transcribed"] += 1
            if not a or not b:
                agreement["empty_or_unavailable"] += 1
            elif a != b:
                agreement["disagree"] += 1
            elif a == actual:
                agreement["both_agree_match"] += 1
            else:
                agreement["both_agree_wrong"] += 1
                if plan_pass:
                    raise ValueError("Both-agree-wrong cannot pass the retained intended-plan rule")
        else:
            if actual is not None or row["human_audibility"] != "human_inaudible":
                raise ValueError("Human audibility label mismatch")
            counts["human_inaudible"] += 1
        if row["acoustic_completeness"] == "tail_unknown":
            counts["plan_pass_with_tail_uncertainty"] += plan_pass
    expected = summary["calibration_counts"]
    for key, value in counts.items():
        if expected[key] != value:
            raise ValueError("Aggregate projection mismatch: " + key)
    for key in ("both_agree_match", "both_agree_wrong", "disagree", "empty_or_unavailable"):
        if summary["agreement_fixed28"][key] != agreement[key]:
            raise ValueError("Agreement aggregate mismatch")
    resources = json.loads(recovery["resources.json"])
    for model, stage in summary["recovery_resources"].items():
        original = resources[model]
        if (stage["wall_seconds"] != original["wall_seconds"] or
                stage["sample_count"] != original["sample_count"] or
                stage["sampled_rss_peak_bytes"] != original["max_sampled"]["rss_bytes"] or
                stage["actual_maximum_sampling_gap"] != "UNKNOWN"):
            raise ValueError("Resource projection mismatch")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--restore", type=pathlib.Path, help="Restore verified JSON members into a new directory")
    args = parser.parse_args()
    manifest_body = (ROOT / "archive-manifest.json").read_bytes()
    if args.expected_manifest_sha256 and sha256(manifest_body) != args.expected_manifest_sha256:
        raise ValueError("Manifest does not match the supplied immutable pin")
    manifest = json.loads(manifest_body)
    expected = {x["path"]: x for x in manifest["files"]}
    paths = list(ROOT.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Archive symlink rejected")
    observed = {p.relative_to(ROOT).as_posix() for p in paths if p.is_file()}
    if observed != set(expected) | {"archive-manifest.json"}:
        raise ValueError("Archive file membership mismatch")
    for name, row in expected.items():
        safe_relative(name)
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("Archive symlink rejected")
        body = path.read_bytes()
        if (len(body), sha256(body), git_blob_sha1(body)) != (row["bytes"], row["sha256"], row["git_blob_sha1"]):
            raise ValueError("Public file identity mismatch: " + name)
    failure = read_raw(ROOT / "raw/setup-attempt1-failed.zip")
    recovery = read_raw(ROOT / "raw/setup-attempt2-asr60.zip")
    counts = verify_summary(json.loads((ROOT / "summary.json").read_bytes()), failure, recovery)
    if args.restore:
        destination = args.restore.resolve()
        if destination.exists() or destination == ROOT or ROOT in destination.parents:
            raise ValueError("Restoration requires a new directory outside this archive")
        destination.mkdir(parents=True)
        for group, raw in (("setup-attempt1-failed", failure), ("setup-attempt2-asr60", recovery)):
            for name, body in sorted(raw.items()):
                path = destination / group / safe_relative(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                if path.read_bytes() != body:
                    raise ValueError("Restored member differs")
    print(json.dumps({"status": "PASS", "public_files": len(expected) + 1,
                      "failure_members": len(failure), "recovery_members": len(recovery),
                      "calibration_counts": counts, "model_calls_performed": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
