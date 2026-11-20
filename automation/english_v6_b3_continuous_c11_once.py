#!/usr/bin/env python3
"""V6 B3 continuous (no resets) C11 acoustic-mechanism investigation.

Reuse only the 14 already EXPOSED machine-reviewed synthetic DEV originals:
6 positives, 8 ordinary. No new data, training, threshold search, CAL/LOCKED,
human FAR/FRR, or promotion. Preserve one C11 state across joined clips.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location(
    "v6_b3_verified_parity", HERE / "english_v6_b3_c11_arm_parity_once.py")
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)
SCHEMA = "english-v6-b3-exposed-dev-continuous-c11-mechanism-v1"
BEFORE = "acbd6b2bab42374d2125e93537e040bc32b9c1f7"
PRIOR_PARITY_RUN = 44653612830
PRIOR_PARITY_ARTIFACT = 14593110591
PRIOR_PARITY_SHA = "f288798c50a85cce0a2fef22b7006d71237f53afec7dadbdbbe0e610fd424036"
SILENCE_SAMPLES = 16000
GRACE_SAMPLES = 4000
COMPILERS = ("gcc", "clang", "cortex-a32-qemu")
CASES = ("mixed", "reverse", "ordinary_only", "silence_5s")

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def store(path, record):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")

def source_check():
    parent = b3.source_check()
    need(parent["status"] == "SOURCE_ONLY_NO_ACOUSTIC_FORWARDS"
         and parent["model_artifact"] == 14590093798
         and parent["original_reviewed_dev_positive"] == 6
         and parent["original_reviewed_dev_ordinary"] == 8
         and parent["unknown_excluded"] == 18
         and b3.B3["sha"] == "59373f0e7e5f9d4ed35d6b116cdd6f8ac8eb5abcfa1acb5311ef658c59a20e04"
         and len(PRIOR_PARITY_SHA) == 64 and SILENCE_SAMPLES == 16000
         and GRACE_SAMPLES == 4000 and len(COMPILERS) == 3
         and len(CASES) == 4,
         "EXACT_B3_AND_EXPOSED_DEV_SOURCE_ONLY")
    return {"schema": SCHEMA, "status": "SOURCE_ONLY_NO_AUDIO_FORWARD",
            "keyword": "Hey Mero", "model": "B3 suffix-only CTC 43876",
            "B3_artifact": b3.B3["id"],
            "prior_software_parity_run": PRIOR_PARITY_RUN,
            "source_positive": 6, "source_ordinary": 8,
            "unknown_excluded": 18, "cases": list(CASES),
            "one_stream_without_reset": True,
            "independent_development_quality": False,
            "far_per_hour": "NOT_MEASURED", "word_end_p95": "NOT_MEASURED",
            "ssc305_board": "NOT_RUN", "no_training": True,
            "no_tts_asr": True, "model_selection_allowed": False}

def verify_prior(token):
    need(isinstance(token, str) and len(token) > 10, "READONLY_GITHUB_TOKEN_REQUIRED")
    b3_receipt = b3.provenance(token)
    def get(suffix):
        req = urllib.request.Request(
            "https://api.github.com/repos/jiying2007/kws-data/actions/" + suffix,
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "kws-b3-continuous-once"})
        with urllib.request.urlopen(req, timeout=35) as response:
            raw = response.read(1048577)
        need(len(raw) <= 1048576, "BOUNDED_PARITY_METADATA")
        return json.loads(raw)
    run = get("runs/" + str(PRIOR_PARITY_RUN))
    need(run.get("id") == PRIOR_PARITY_RUN
         and run.get("head_sha") == BEFORE
         and run.get("event") == "push" and run.get("run_attempt") == 1
         and run.get("status") == "completed" and run.get("conclusion") == "success",
         "EXACT_SUCCESSFUL_B3_NUMERIC_PARITY_FIRST_MAIN")
    art = get("artifacts/" + str(PRIOR_PARITY_ARTIFACT))
    need(art.get("id") == PRIOR_PARITY_ARTIFACT
         and art.get("name") == "english-v6-b3-c11-arm32-parity-44653612830"
         and art.get("digest") == "sha256:" + PRIOR_PARITY_SHA
         and art.get("size_in_bytes") == 5887 and art.get("expired") is False
         and art.get("workflow_run", {}).get("id") == PRIOR_PARITY_RUN,
         "EXACT_ORIGINAL_B3_SOFTWARE_PARITY_ARTIFACT")
    return {"B3_provenance": b3_receipt, "prior_parity_run": PRIOR_PARITY_RUN,
            "prior_parity_artifact": PRIOR_PARITY_ARTIFACT,
            "prior_parity_sha256": PRIOR_PARITY_SHA}

def guard():
    source_check()
    env = os.environ
    need(all(env.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1"}.items()), "FIRST_PROTECTED_PUBLIC_MAIN_ONLY")
    head = env.get("GITHUB_SHA", "")
    need(re.fullmatch("[0-9a-f]{40}", head) is not None
         and re.fullmatch("[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")),
         "EXACT_RUN_AND_COMMIT")
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before") == BEFORE and event.get("after") == head
         and event.get("ref") == "refs/heads/main"
         and event.get("repository", {}).get("full_name") == "jiying2007/kws-data"
         and event.get("repository", {}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False
         and subprocess.check_output(["git", "rev-parse", "HEAD^1"],
                                     cwd=ROOT, text=True).strip() == BEFORE,
         "NO_REPLAY_OR_MOVED_MAIN")
    return {"run": int(env["GITHUB_RUN_ID"]), "commit": head,
            "previous": verify_prior(env.get("GH_TOKEN"))}

def select_rows(rows):
    need(len(rows) == 14
         and len({r["id"] for r in rows}) == 14
         and all(r["role"] == "development" and
                 r["mono_class"] in (0, 1) for r in rows)
         and len({r["source_group_id"] for r in rows}) == 8,
         "ONLY_14_SOURCE_DISJOINT_EXPOSED_DEV_ORIGINALS")
    pos = sorted((r for r in rows if r["mono_class"] == 1), key=lambda r: r["id"])
    neg = sorted((r for r in rows if r["mono_class"] == 0), key=lambda r: r["id"])
    need(len(pos) == 6 and len(neg) == 8, "SIX_POSITIVE_EIGHT_ORDINARY")
    mixed = []
    for i, item in enumerate(neg):
        mixed.append(item)
        if i < len(pos):
            mixed.append(pos[i])
    return {"mixed": mixed, "reverse": list(reversed(mixed)),
            "ordinary_only": neg, "silence_5s": []}

def assemble(rows, read_pcm, case):
    order = select_rows(rows)
    need(case in order, "FROZEN_CONTINUOUS_CASE")
    output = bytearray(b"\0" * (2 * SILENCE_SAMPLES))
    segments = []
    for row in order[case]:
        original = read_pcm(row)
        need(isinstance(original, bytes) and len(original) % 2 == 0
             and 512 <= len(original) <= 2 * 16000 * 6,
             "COMPLETE_BOUND_ORIGINAL_PCM16")
        need(b3.sha(original) == row["source_pcm_sha256"], "ORIGINAL_PCM_SHA")
        start = len(output) // 2
        output.extend(original)
        end = len(output) // 2
        output.extend(b"\0" * 2 * SILENCE_SAMPLES)
        segments.append({"id": row["id"], "source_group": row["source_group_id"],
                         "class": row["mono_class"], "original_sha256": b3.sha(original),
                         "start": start, "end": end,
                         "event_window_end": end + GRACE_SAMPLES})
    if case == "silence_5s":
        output = bytearray(b"\0" * 2 * 16000 * 5)
    need(0 < len(output) <= 2 * 16000 * 120, "BOUNDED_CONSTRUCTED_SEQUENCE")
    return bytes(output), segments

def classify(clocks, segments):
    assignment = []
    for clock in clocks:
        candidates = [r for r in segments
                      if r["start"] <= clock < r["event_window_end"]]
        need(len(candidates) <= 1, "NON_OVERLAPPING_EVENT_WINDOWS")
        seg = candidates[0] if candidates else None
        assignment.append({"available_after_sample": clock,
                           "segment_id": seg["id"] if seg else None,
                           "segment_class": seg["class"] if seg else None,
                           "offset_from_segment_start_samples":
                               clock - seg["start"] if seg else None})
    return assignment

def execute(source, screen, archive, pipeline, output):
    import numpy as np
    import torch
    receipt = guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "NEW_OUTPUT_DIRECTORY_ONLY")
    output.mkdir()
    store(output / "STARTED.json", {"schema": SCHEMA, "receipt": receipt,
          "training": False, "source": "already_exposed_synthetic_dev",
          "no_cal_or_locked_read": True, "threshold_search": False})
    try:
        rows, source_receipt = b3.audit(source, screen, archive)
        native = b3.pipeline_files(pipeline)
        packer = b3.imported(native / "pack_mero43.py", "continuous_b3_geometry")
        _, _, _, _, packed = b3.load_b3(torch, np, archive, packer)
        need(len(packed) == 216960, "FROZEN_B3_PACK_ABI")
        torch.set_num_threads(1)
        binaries = {v: b3.compile_c(native, v, output / ("bin-" + v))
                    for v in COMPILERS}
        pack = output / "temp.pack"
        pack.write_bytes(packed)
        reports = []
        for case in CASES:
            pcm, segments = assemble(rows, lambda r: b3.v6.pcm(r, source), case)
            scratch = output / ("scratch-" + case)
            scratch.mkdir()
            path = scratch / "input.pcm"
            path.write_bytes(pcm)
            by_compiler = {}
            for kind, binary in binaries.items():
                logits = scratch / (kind + ".f32")
                events = scratch / (kind + ".csv")
                cmd = (["qemu-arm", str(binary)] if kind == "cortex-a32-qemu"
                       else [str(binary)])
                completed = subprocess.run(
                    cmd + [str(pack), str(path), str(logits), str(events)],
                    text=True, capture_output=True, timeout=150, check=True)
                need("MERO43_C_PCM_STREAM_OK" in completed.stderr,
                     "REAL_CONTINUOUS_C11_FEED_" + kind)
                need(logits.stat().st_size % 16 == 0, "FINITE_FRAME_BYTE_LAYOUT")
                frames = logits.stat().st_size // 16
                clocks = b3.c_events(events, frames)
                by_compiler[kind] = {
                    "frames": frames, "ctc_event_clocks": clocks,
                    "logits_sha256": b3.sha(logits.read_bytes())}
            reference = by_compiler["gcc"]
            need(all(row["frames"] == reference["frames"] and
                     row["ctc_event_clocks"] == reference["ctc_event_clocks"]
                     for row in by_compiler.values()),
                 "C11_CLANG_ARM32_CONTINUOUS_EVENT_CLOCK_MISMATCH")
            hits = classify(reference["ctc_event_clocks"], segments)
            reports.append({"case": case, "stream_reset_between_clips": False,
                 "duration_samples": len(pcm) // 2, "pcm_sha256": b3.sha(pcm),
                 "reused_originals": len(segments),
                 "positive_original_segments": sum(s["class"] == 1 for s in segments),
                 "ordinary_original_segments": sum(s["class"] == 0 for s in segments),
                 "segments": segments, "observed_events": hits,
                 "unattributed_events": sum(x["segment_class"] is None for x in hits),
                 "positive_segments_with_event": len({
                     x["segment_id"] for x in hits if x["segment_class"] == 1}),
                 "ordinary_segments_with_event": len({
                     x["segment_id"] for x in hits if x["segment_class"] == 0}),
                 "compilers": by_compiler})
            shutil.rmtree(scratch)
        need(len(reports) == len(CASES), "FULL_FOUR_CASE_CONTINUOUS_DIAGNOSTIC")
        result = {"schema": SCHEMA,
           "status": "RESEARCH_CONTINUOUS_EXPOSED_DEV_BEHAVIOR_OBSERVED_NO_PROMOTION",
           "receipt": receipt, "source": source_receipt,
           "model_artifact": b3.B3["id"], "model_sha256": b3.B3["pt"],
           "original_independent_count": 14, "positive": 6, "ordinary": 8,
           "unknown_excluded": 18, "no_new_independent_source": True,
           "cases": reports, "training_updates": 0,
           "new_tts_asr_calls": 0, "model_selection_allowed": False,
           "natural_far_per_hour": "NOT_MEASURED",
           "human_3_5m_frr": "NOT_MEASURED",
           "word_end_p95": "NOT_MEASURED", "ssc305": "NOT_RUN",
           "product_promotion": False}
        store(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"],
                          "per_case": [{"case": x["case"],
                              "events": len(x["observed_events"]),
                              "positive_segments_with_event": x["positive_segments_with_event"],
                              "ordinary_segments_with_event": x["ordinary_segments_with_event"]}
                                      for x in reports],
                          "product_promotion": False}, sort_keys=True))
    except BaseException as err:
        store(output / "STOP.json", {"schema": SCHEMA, "status": "STOP_CONTINUOUS_NO_REPLAY",
              "error_type": type(err).__name__, "reason": str(err)[:400],
              "training_updates": 0, "product_promotion": False})
        raise
    finally:
        for scratch in output.glob("scratch-*"):
            if scratch.is_dir():
                shutil.rmtree(scratch)
        for file in output.glob("bin-*"):
            if file.is_file():
                file.unlink()
        if (output / "temp.pack").is_file():
            (output / "temp.pack").unlink()

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check", "--guard", "--execute"):
        group.add_argument(flag, action="store_true")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--screen", type=Path)
    ap.add_argument("--b3-zip", type=Path)
    ap.add_argument("--pipeline", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.source_check:
        print(json.dumps(source_check(), sort_keys=True))
    elif args.guard:
        print(json.dumps(guard(), sort_keys=True))
    else:
        need(all(x is not None for x in (args.source, args.screen,
              args.b3_zip, args.pipeline, args.output)), "MISSING_EXACT_INPUT")
        execute(args.source, args.screen, args.b3_zip, args.pipeline, args.output)

if __name__ == "__main__":
    main()
