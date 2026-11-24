"""Original B20 result audit and descriptive saved-logit failure diagnostics.

No model runtime, source archive, waveform, network or mutable gate is opened.
The separate source auditor supplies an externally reviewed SOURCE_AUTHORITY
hash. This program does not create that source authority from model results.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

BASE = Path(__file__).resolve().parents[1]
WORK = BASE.parent
SAVED = WORK / "b20_replication_eval_work/automation/english_v6_b20_saved_scores.py"
SAVED_SHA = "96803fe5940c90bc9a5cc3a022737b9982ca413dac3c4bbb8600627651c88e3d"
PRIMARY_SHA = "87cb205d5bbd5e5ba17bd0a5929b7e123c5b5688894e753d34406535273ca802"
SPEC_SHA = "78b0096f2712b92fa4c2988ad5888e46ecd68df69c8d836b1fc53f843bf7b3a7"
RESULT_ZIP_SHA = "9972b33b74facbf1e9c055eec3820b981b8d0f9c829791e52c17468bb4891eb5"
RESULT_SHA = "86d4cc6a84494f18b67479db9d986f722103566b81ba7c1f6772a5c56d719ff7"
NPZ_SHA = "272e69d0f9bfd72aa8a9da606d67db3677991a214dd950df5cd04aa5bcd315cb"
TOKENS = ("blank", "HEY", "MEH", "ROH")
ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def describe(labels, values, primary):
    runs = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end < len(labels) and labels[end] == labels[start]:
            continue
        token = labels[start]
        if token:
            margins = [values[i][token] - values[i][0] for i in range(start, end)]
            runs.append(dict(token=token, name=TOKENS[token], first_frame=start, last_frame=end - 1,
                samples_available_first=512 + 160 * start, samples_available_last=512 + 160 * (end - 1),
                margin_over_blank_min=min(margins), margin_over_blank_max=max(margins)))
        start = end
    return dict(**primary.ctc_events(labels), frame_count=len(labels),
        argmax_frame_counts={name: labels.count(i) for i, name in enumerate(TOKENS)}, nonblank_runs=runs)


def mechanism(result, arrays, saved, primary):
    indexed = {row["id"]: row for row in result["records"]}
    cases = [(ident, gain, "lost_positive", reference) for ident, gain, reference in result["lost_positives"]]
    cases += [(ident, gain, "added_ordinary_event", reference) for ident, gain, reference in result["added_ordinary_events"]]
    explanations = []
    for ident, gain, failure, reference in cases:
        row = indexed[ident]
        data = {}
        for arm in ARMS:
            state = row["gains"][gain][arm]
            body, labels = saved.read_npy(arrays.read(state["array"] + ".npy"), (state["frames"], 4))
            data[arm] = (labels, list(struct.iter_unpack("<4f", body)))
        comparisons = {}
        for arm in ("B0", "B9", "challenger"):
            transitions = Counter((left, right) for left, right in zip(data["control"][0], data[arm][0], strict=True) if left != right)
            comparisons[arm] = {TOKENS[a] + "->" + TOKENS[b]: count for (a, b), count in sorted(transitions.items())}
        changes = []
        for frame, (left, right) in enumerate(zip(data["control"][0], data["challenger"][0], strict=True)):
            if left == right:
                continue
            changes.append(dict(frame=frame, available_sample=512 + 160 * frame,
                argmax={arm: data[arm][0][frame] for arm in ARMS},
                logits={arm: list(data[arm][1][frame]) for arm in ARMS}))
        explanations.append(dict(id=ident, group=row["group"], cohort=row["cohort"], gain=gain,
            failure=failure, reference=reference,
            models={arm: describe(*data[arm], primary) for arm in ARMS},
            changed_argmax_counts_vs_control=comparisons,
            every_control_challenger_changed_frame=changes))
    opportunities = []
    for row in result["records"]:
        if not row["positive"]:
            continue
        for gain in GAINS:
            state = row["gains"][gain]
            if state["B9"]["events"] and not state["control"]["events"]:
                opportunities.append(dict(id=row["id"], group=row["group"], gain=gain,
                    events={arm: state[arm]["events"] for arm in ARMS}, recovered=bool(state["challenger"]["events"])))
    by_gain = {gain: dict(originals=len({r["id"] for r in opportunities if r["gain"] == gain}),
        source_groups=len({r["group"] for r in opportunities if r["gain"] == gain}),
        original_gain_cases=sum(r["gain"] == gain for r in opportunities),
        recovered_original_gain_cases=sum(r["gain"] == gain and r["recovered"] for r in opportunities)) for gain in GAINS}
    return dict(schema="english-v6-b20-original-logit-failure-description-v1",
        original_result_sha256=RESULT_SHA, original_frame_archive_sha256=NPZ_SHA,
        purpose="Descriptive failure explanation only; no threshold selection, training or new candidate choice",
        new_model_forwards=0, audio_reads=0, source_calls=0, threshold_search=False,
        training_allowed=False, candidate_selection_allowed=False, promotion_allowed=False,
        token_ids=dict(enumerate(TOKENS)), failure_cases=explanations,
        positive_denominator=dict(originals=sum(r["positive"] for r in result["records"]),
            source_groups=len({r["group"] for r in result["records"] if r["positive"]}),
            original_gain_cases=3 * sum(r["positive"] for r in result["records"])),
        recovery_opportunities=dict(definition="B9 has at least one complete event and paired control has zero on the same original/gain",
            original_gain_cases=len(opportunities), unique_originals=len({r["id"] for r in opportunities}),
            unique_source_groups=len({r["group"] for r in opportunities}), by_gain=by_gain, cases=opportunities))


def main(expected_authority_sha):
    authority_path = BASE / "independent/SOURCE_AUTHORITY.json"
    authority_raw = authority_path.read_bytes()
    need(sha(authority_raw) == expected_authority_sha, "separately reviewed exact source authority")
    authority = json.loads(authority_raw)
    need(authority["schema"] == "english-v6-b20-independent-primary-source-authority-v1"
         and authority["status"] == "AUTHENTICATED_ORIGINAL_SOURCES_ONLY"
         and authority["implementation_sha256"] == PRIMARY_SHA
         and authority["registered_inputs_sha256"] == SPEC_SHA
         and authority["actual_admitted_originals"] == len(authority["selected_rows"]) == 84
         and authority["source_originals"] == len(authority["all_original_rows"]) == 120
         and authority["model_calls"] == authority["source_calls"] == authority["observer_calls"] == 0,
         "complete prior source/producer authentication, never RESULT-derived admission")
    need(sha(SAVED.read_bytes()) == SAVED_SHA, "reviewed audit-only saved-array verifier")
    primary_root = BASE / "producer/primary_producer/automation"
    sys.path[:0] = [str(primary_root), str(primary_root.parent)]
    saved = load(SAVED, "independent_b20_saved_scores")
    primary, _ = saved.dependencies()
    need(sha(primary.SPEC.read_bytes()) == SPEC_SHA, "actual activated registration, unchanged driver")
    spec = primary.protocol()
    original = (BASE / "artifacts/result-original.zip").read_bytes()
    need(len(original) == 5916607 and sha(original) == RESULT_ZIP_SHA,
         "one exact original completed score artifact")
    with saved.opened(original) as archive:
        need(sha(archive.read("RESULT.json")) == RESULT_SHA and sha(archive.read("ALL_FRAME_LOGITS.npz")) == NPZ_SHA,
             "exact original result and complete saved frame archive")
    checked = saved.audit_saved_result(original, spec, "primary", authority["admission"], authority["selected_rows"])
    need(checked["audit_only"] is True and checked["authorization_granted"] is False
         and checked["panel_gate"] == "NO_PROMOTION", "honest completed failure audit without a permit")
    result = checked["original_result"]
    rejected = False
    try:
        saved.verify_saved_result(original, spec, "primary", authority["admission"], authority["selected_rows"])
    except ValueError:
        rejected = True
    need(rejected, "the actual failed original result cannot pass the unchanged opening reader")
    with saved.opened(checked["npz_bytes"]) as arrays:
        diagnostic = mechanism(result, arrays, saved, primary)
    need("numpy" not in sys.modules and "torch" not in sys.modules, "pure standard-library original-array readback only")
    report = {name: value for name, value in checked.items() if name not in ("npz_bytes", "records", "original_result", "original_admission")}
    report.update(source_authority_sha256=expected_authority_sha, source_authority_path="independent/SOURCE_AUTHORITY.json",
        source_authority_authenticated_by="separate original producer/source auditor",
        original_identity=authority["actual_identity"], original_artifact=authority["original_artifacts"]["result"],
        result_sha256=RESULT_SHA, logits_sha256=NPZ_SHA, saved_verifier_sha256=SAVED_SHA,
        primary_implementation_sha256=PRIMARY_SHA, registered_inputs_sha256=SPEC_SHA,
        exact_source_coverage=result["source_coverage"], originals=84, raw_source_originals=120,
        actual_forward_attempts=result["actual_forwards"], raw_complete_returns=result["raw_output_count"],
        saved_array_count=result["saved_array_count"], model_forward_calls=result["model_forward_calls"],
        consumed_unknown=result["consumed_unknown"], all_original_arrays_recomputed=True,
        original_next_gate_reader_rejected=True,
        failures={name: result[name] for name in ("restored_positives", "lost_positives", "extra_positive_events",
            "added_ordinary_events", "natural_readers_improved_vs_B9", "natural_above_B0_gains")},
        summary=result["summary"], replay_allowed=False)
    output = Path(__file__).resolve().parent
    for name, document in (("ARRAY_AUDIT.json", report), ("SAVED_LOGIT_DIAGNOSTIC.json", diagnostic)):
        path = output / name
        path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(dict(file=str(path), sha256=sha(path.read_bytes()))))
    print(json.dumps(dict(status=checked["audit_status"], actual_gate=result["panel_gate"],
        raw_arrays=result["raw_output_count"], actual_forwards=result["actual_forwards"],
        new_model_forwards=0, authorization_granted=False,
        recovery_opportunities=diagnostic["recovery_opportunities"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-authority-sha", required=True)
    main(parser.parse_args().source_authority_sha)
