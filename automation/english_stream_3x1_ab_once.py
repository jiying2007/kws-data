#!/usr/bin/env python3
"""One new bounded A/B hypothesis on saved PCM; preserve the installed baseline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile

import english_stream_resource_once as saved

RESOURCE = {"run_id": 44000006165, "artifact_id": 14292027025,
            "name": "english-stream-resource-v41-44000006165",
            "head_sha": "00f3a81524308d8e7f5fc43015a26486de593760",
            "bytes": 4547376, "sha256": "ce4100bc444cfe99ef18c62e8a1bb17be655c7b57340949049e3abc5867a065f"}
TRACKS = {"libri-speaker-1272": "a192f737578dd259c447caf07a06a9566326aabd489faa96d4ccfb8854fc33ab",
          "libri-speaker-1462": "9a93bc5842da1ad5f3d6af3e9987ec7a157127d0eea6f6d83282df7afe37ffa7"}
MEMBERS = ["ekws_research_stream.c.o", "fixed_frontend_mel.c.o", "english_causal_head.c.o", "fixed_models.c.o"]
FLAGS = ["-std=c11", "-O2", "-DNDEBUG", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
need, sha, identity, read_json, write_json = saved.need, saved.sha, saved.file_identity, saved.read_json, saved.write_json


def validate_plan(plan):
    need(plan.get("schema") == "english-stream-3x1-ab-once-v1", "plan schema")
    need(plan.get("main_before") == RESOURCE["head_sha"], "new main successor only")
    need(re.fullmatch(r"[0-9a-f]{40}", plan.get("pipeline_commit", "")) is not None, "exact pipeline source required")
    need(type(plan.get("planned_full_model_passes")) is int and plan["planned_full_model_passes"] == 8
         and type(plan.get("maximum_execution_seconds")) is int and plan["maximum_execution_seconds"] == 720,
         "fixed eight-pass execution budget")
    need(plan.get("new_audio_or_model_training") is False and type(plan.get("retry_count")) is int
         and plan["retry_count"] == 0,
         "saved-input no-training no-retry boundary")


def parse_trace(raw):
    """Independent structural parser; full numeric equality is a raw-byte comparison."""
    need(len(raw) >= 93, "trace too short")
    need(struct.unpack_from("<8s6IQ", raw) == (b"EKWSTRC1", 1, 1, 16000, 320, 96, 3, 256000), "trace header")
    offset, records, events, ready_count = 40, 0, 0, 0

    def record(kind, length):
        nonlocal offset, records
        need(offset + 5 <= len(raw), "missing trace record")
        actual_kind, actual_length = struct.unpack_from("<BI", raw, offset)
        need((actual_kind, actual_length) == (kind, length), "trace record kind/length/order")
        offset += 5
        need(offset + length <= len(raw), "truncated trace payload")
        payload = raw[offset:offset + length]
        offset += length
        records += 1
        return payload

    for index in range(188):
        sample = 16000 + index * 1280
        embedding = record(1, 400)
        need(struct.unpack_from("<QQ", embedding) == (index, sample), "embedding ordinal/clock")
        need(all(math.isfinite(x) for x in struct.unpack_from("<96f", embedding, 16)), "nonfinite embedding")
        score = record(2, 48)
        ordinal, ready, keyword, observed = struct.unpack_from("<QIIQ", score)
        need((ordinal, observed) == (index, sample) and ready == int(index >= 15) and keyword in (0, 1, 2), "score counters")
        need(ready or keyword == 0, "unready event")
        need(all(math.isfinite(x) for x in struct.unpack_from("<6f", score, 24)), "nonfinite score")
        ready_count += ready
        if keyword:
            event = record(3, 32)
            event_index, embed_index, event_keyword, event_sample, confidence = struct.unpack("<QQIQf", event)
            need((event_index, embed_index, event_keyword, event_sample) == (events, index, keyword, sample)
                 and math.isfinite(confidence), "event identity/clock")
            need(event[28:32] == score[36 + 4 * keyword:40 + 4 * keyword], "event confidence/score bits")
            events += 1
    end = struct.unpack("<6Q", record(255, 48))
    need(end == (256000, 188, 188, 173, events, 800) and ready_count == 173, "END counters")
    need(offset == len(raw) == 86197 + events * 37, "trace trailing bytes/size")
    return {"input_samples": 256000, "embeddings": 188, "scores": 188, "ready": 173,
            "events": events, "feed_calls": 800, "records": records, "bytes": len(raw), "sha256": sha(raw)}


def verify_trace_metadata(meta, parsed):
    need(meta.get("schema") == "english-stream-numeric-trace-v1" and meta.get("trace_completed") is True
         and meta.get("end_record_written") is True and meta.get("score_observer_enabled") is True
         and meta.get("original_head_link_wrap") is True and meta.get("performance_measured") is False,
         "trace metadata boundary")
    need(meta.get("format_magic") == "EKWSTRC1" and meta.get("encoding") == "s16le_mono", "trace metadata encoding")
    expected = {"format_version": 1, "abi": 1, "sample_rate_hz": 16000, "chunk_samples": 320,
                "input_samples": 256000, "input_bytes": 512000, "feed_calls": 800,
                "embedding_records": 188, "real_head_calls": 188, "score_records": 188, "ready_outputs": 173,
                "event_records": parsed["events"], "trace_records": parsed["records"], "trace_bytes": parsed["bytes"],
                "input_repetitions": 1, "resets": 0, "eof_padding_samples": 0}
    need(all(type(meta.get(k)) is int and meta[k] == v for k, v in expected.items()), "trace metadata counters")


def verify_measurement(report, case_id):
    need(report.get("passed") is True and report.get("measurement_completed") is True
         and report.get("actual_input_seconds") == 16 and len(report.get("cases", [])) == 1, "performance result")
    case = report["cases"][0]
    need(case["input"]["id"] == case_id and case["status"] == "completed" and case["exit_code"] == 0
         and case["pcm_sha256_after"] == TRACKS[case_id], "performance input/exit")
    m = case["measurements"]
    need(m["score_observer_enabled"] is False and m["input_samples"] == 256000 and m["embeddings"] == 188
         and m["ready_outputs"] == 173 and m["resets"] == 0 and m["eof_padding_samples"] == 0
         and m["steady_output_counts"]["embeddings"] == 172 and m["steady_output_counts"]["ready_outputs"] == 172,
         "performance stream counters")
    need(m["steady"]["audio_seconds"] == 13.8 and m["steady"]["calls"] == 690
         and all(math.isfinite(m["steady"][k]) and m["steady"][k] > 0
                 for k in ("accept_wall_seconds", "accept_process_cpu_seconds", "wall_rtf", "cpu_seconds_per_audio_second")),
         "performance denominator/timing")
    return m


def extract_resource(archive, output):
    need(identity(archive) == {"bytes": RESOURCE["bytes"], "sha256": RESOURCE["sha256"]}, "resource ZIP identity")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        need(len(infos) == len({x.filename for x in infos}) == 35 and sum(x.file_size for x in infos) < 64 * 1024**2,
             "closed resource archive")
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xffff
            need(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                 and path.as_posix() == info.filename and not info.is_dir()
                 and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), "unsafe resource member")
        output.mkdir()
        for info in infos:
            target = output / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(info))
    result = read_json(output / "evidence/RESULT.json")
    need(result["status"] == "RESOURCE_MEASUREMENT_COMPLETED" and result["model_passes_completed"] == 3
         and result["github_sha"] == RESOURCE["head_sha"] and result["run_id"] == str(RESOURCE["run_id"])
         and result["baseline_input_seconds"] == 32 and result["kws_model_delivered"] is False, "source execution boundary")
    manifest = read_json(output / "inputs/inputs.json")
    need([x["id"] for x in manifest["cases"]] == list(TRACKS), "fixed source cases")
    for case in manifest["cases"]:
        need(case["path"] == case["id"] + ".pcm" and case["sha256"] == TRACKS[case["id"]]
             and case["samples"] == 256000 and case["exposure"] == "previously_exposed", "fixed source input metadata")
        need(identity(output / "inputs" / case["path"]) == {"bytes": 512000, "sha256": case["sha256"]}, "source PCM bytes")
    return manifest


def execute(args, plan):
    saved.guard(plan)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    evidence = root / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    state = {"schema": "english-stream-3x1-ab-execution-v1", "status": "RUNNING",
             "run_id": os.environ["GITHUB_RUN_ID"], "github_sha": os.environ["GITHUB_SHA"],
             "pipeline_commit": plan["pipeline_commit"], "sdk": saved.SDK, "resource_source": RESOURCE,
             "model_passes_planned": 8, "model_runner_invocations_started": 0, "verified_model_passes_completed": 0,
             "actual_model_audio_seconds_verified": 0, "retry_count": 0, "tts_asr_phone_calls": 0,
             "optimizer_updates": 0, "independent_far_test_hours": 0, "formal_training_admitted": False,
             "model_promotion_allowed": False, "kws_model_delivered": False, "ssc305": "NOT_RUN",
             "far_frr": "NOT_MEASURED", "word_end_latency": "NOT_MEASURED", "performance_budget_evaluated": False}

    def checkpoint():
        write_json(evidence / "RESULT.json", state)

    def command(label, argv, timeout=180, cwd=None, auth=False, model=False):
        remaining = 720 - (time.monotonic() - started)
        need(remaining > 0, "experiment deadline")
        env = dict(os.environ)
        if not auth:
            for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
                env.pop(key, None)
        if model:
            need(state["model_runner_invocations_started"] < 8, "full-model pass budget")
            state["model_runner_invocations_started"] += 1
        state["stage"] = label
        checkpoint()
        begin, code, error_name = time.monotonic(), None, None
        try:
            with (evidence / (label + ".stdout")).open("xb") as out, (evidence / (label + ".stderr")).open("xb") as err:
                process = subprocess.Popen(list(map(str, argv)), cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
                try:
                    code = process.wait(timeout=min(timeout, remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            need(code == 0, "command failed: " + label)
            return evidence / (label + ".stdout")
        except Exception as error:
            error_name = type(error).__name__
            raise
        finally:
            write_json(evidence / (label + "-exit.json"), {"argv": list(map(str, argv)), "cwd": str(cwd) if cwd else None,
                       "returncode": code, "error_type": error_name, "wall_seconds": time.monotonic() - begin,
                       "full_model_process": model})

    def verified_pass():
        state["verified_model_passes_completed"] += 1
        state["actual_model_audio_seconds_verified"] += 16
        checkpoint()

    def download(tag, pin):
        run = read_json(command(tag + "-run", ["gh", "api", "repos/" + saved.REPO + "/actions/runs/" + str(pin["run_id"])], auth=True))
        artifact = read_json(command(tag + "-artifact", ["gh", "api", "repos/" + saved.REPO + "/actions/artifacts/" + str(pin["artifact_id"])], auth=True))
        need(run["id"] == pin["run_id"] and run["status"] == "completed" and run["conclusion"] == "success"
             and run["head_sha"] == pin["head_sha"] and run["head_branch"] == "main" and run["event"] == "push"
             and run["run_attempt"] == 1 and run["repository"]["full_name"] == saved.REPO
             and run["repository"]["private"] is False, "authenticated source run")
        need(artifact["id"] == pin["artifact_id"] and artifact["name"] == pin["name"] and artifact["expired"] is False
             and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
             and artifact["workflow_run"]["id"] == pin["run_id"] and artifact["workflow_run"]["head_sha"] == pin["head_sha"],
             "authenticated source artifact")
        return command(tag + "-zip", ["gh", "api", "repos/" + saved.REPO + "/actions/artifacts/" + str(pin["artifact_id"]) + "/zip"], auth=True)

    checkpoint()
    try:
        pipeline = args.pipeline_root.resolve()
        need(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() == plan["pipeline_commit"],
             "exact candidate source commit")
        component = pipeline / "research/english_stream_performance"
        write_json(evidence / "PLAN.json", plan)
        write_json(evidence / "SOURCE.json", {p.name: identity(p) for p in sorted(component.iterdir()) if p.is_file()})
        write_json(evidence / "HOST.json", {"uname": list(os.uname()), "affinity": sorted(os.sched_getaffinity(0)),
                   "logical_cpu_count": os.cpu_count(), "cpuinfo": Path("/proc/cpuinfo").read_text(), "realtime_pacing": False})
        saved.extract_sdk(download("sdk", saved.SDK), root / "sdk")
        manifest = extract_resource(download("resource", RESOURCE), root / "resource")
        for name, pin in read_json(root / "resource/evidence/SOURCE.json").items():
            need(identity(component / name) == pin, "original performance source changed: " + name)
        shutil.copytree(root / "resource/inputs", root / "inputs")
        sdk = root / "sdk"
        baseline_sdk = sdk / "sdk-native"
        library = baseline_sdk / "lib/libekws_research_stream.a"
        head = baseline_sdk / "models/causal-head.bin"
        head_pin = identity(head)
        source = sdk / "source/research"
        candidate = root / "candidate"
        candidate.mkdir()
        command("specialize", [sys.executable, "-B", component / "specialize_3x1.py", "--source",
                source / "english_mel_specialization/fixed_frontend_mel.c", "--manifest", sdk / "generated/export_manifest.json",
                "--output", candidate / "frontend.c", "--receipt", candidate / "SOURCE_COPY.json"])
        need(identity(candidate / "frontend.c")["sha256"] == "b1d8011c099ed8d29f1423fb89409c4b9dcc5a7a3b63f12724e90aa87ff08a3d", "fixed candidate source identity")
        objects = candidate / "objects"
        objects.mkdir()
        archive_list = command("original-archive-list", ["ar", "t", library]).read_text().splitlines()
        need(archive_list == MEMBERS, "original archive member roster")
        command("extract-original-objects", ["ar", "x", library], cwd=objects)
        before = {name: identity(objects / name) for name in MEMBERS}
        command("compiler", ["gcc", "--version"])
        includes = ["-I" + str(source / "english_final_embedding_frontend"), "-I" + str(component)]
        command("build-candidate-frontend", ["gcc", *FLAGS, *includes, "-fopt-info-vec-optimized=" + str(candidate / "vectorization.txt"),
                "-c", candidate / "frontend.c", "-o", objects / "fixed_frontend_mel.c.o"])
        command("candidate-disassembly", ["objdump", "-d", objects / "fixed_frontend_mel.c.o"])
        candidate_sdk = candidate / "sdk-native"
        for directory in ("lib", "include", "models"):
            (candidate_sdk / directory).mkdir(parents=True)
        candidate_library = candidate_sdk / "lib/libekws_research_stream.a"
        command("build-candidate-archive", ["ar", "rcs", candidate_library, *[objects / name for name in MEMBERS]])
        need(command("candidate-archive-list", ["ar", "t", candidate_library]).read_text().splitlines() == MEMBERS, "candidate archive roster")
        after = {name: identity(objects / name) for name in MEMBERS}
        for member in MEMBERS:
            retained = command("candidate-member-" + member, ["ar", "p", candidate_library, member])
            need(identity(retained) == after[member], "candidate archived object bytes")
        need(all(before[n] == after[n] for n in MEMBERS if n != "fixed_frontend_mel.c.o"), "unchanged objects differ")
        write_json(candidate / "OBJECT_IDENTITIES.json", {"before": before, "after": after,
                   "only_recompiled_member": "fixed_frontend_mel.c.o", "archived_members_verified": True, "original_archive": identity(library),
                   "candidate_archive": identity(candidate_library), "flags": FLAGS})
        shutil.copy2(baseline_sdk / "include/ekws_research_stream.h", candidate_sdk / "include/ekws_research_stream.h")
        shutil.copy2(head, candidate_sdk / "models/causal-head.bin")
        need(identity(candidate_sdk / "models/causal-head.bin") == head_pin, "head changed")
        shutil.copytree(baseline_sdk / "share", candidate_sdk / "share")
        licenses = {str(p.relative_to(baseline_sdk)): identity(p)
                    for p in sorted((baseline_sdk / "share").rglob("*")) if p.is_file()}
        need(set(licenses) == {"share/KwsEnglishResearch/licenses/kws-pipeline-LICENSE",
                               "share/KwsEnglishResearch/licenses/openWakeWord-LICENSE"}, "original license roster")
        need(all(identity(candidate_sdk / name) == pin for name, pin in licenses.items()), "retained license bytes")
        write_json(candidate / "LICENSE_IDENTITIES.json", licenses)
        kernel = candidate / "kernel-equivalence"
        oracle = source / "english_mel_specialization/fixed_frontend_mel.c"
        command("build-kernel-check", ["gcc", *FLAGS, *includes, '-DEKWS_BASELINE_FRONTEND="' + str(oracle) + '"',
                component / "kernel_equivalence.c", "-lm", "-o", kernel])
        numeric = read_json(command("kernel-check", [kernel]))
        need(numeric.get("passed") is True and numeric.get("cases") == 54 and numeric.get("optimized_cases") == 54
             and numeric.get("failures") == 0 and numeric.get("FLT_EVAL_METHOD") == 0
             and numeric.get("full_model_executed") is False and numeric.get("compared_values") == 903168
             and numeric.get("fallback_matcher_checks") == 144 and numeric.get("guard_checks") == 171
             and numeric.get("checks") == 334, "synthetic kernel equivalence")
        state["synthetic_kernel_pairs_passed"] = 54
        performance = {}
        orders = [("libri-speaker-1272", ("baseline", "candidate")), ("libri-speaker-1462", ("candidate", "baseline"))]
        state["performance_execution_order"] = [[case, variant] for case, variants in orders for variant in variants]
        for case_id, variants in orders:
            selected = next(c for c in manifest["cases"] if c["id"] == case_id)
            one_manifest = root / "inputs" / (case_id + ".json")
            write_json(one_manifest, {"schema": manifest["schema"], "cases": [selected]})
            for variant in variants:
                label = "performance-" + case_id + "-" + variant
                output = root / "performance" / (case_id + "-" + variant)
                command(label, [sys.executable, "-B", component / "run_benchmark.py", "--sdk-root",
                        baseline_sdk if variant == "baseline" else candidate_sdk, "--manifest", one_manifest,
                        "--output", output, "--case-timeout", "180"], timeout=200, model=True)
                measurement = verify_measurement(read_json(output / "RESULT.json"), case_id)
                need(identity(root / "inputs" / selected["path"])["sha256"] == TRACKS[case_id]
                     and identity(head) == head_pin, "input/head changed during performance")
                performance[(case_id, variant)] = measurement
                verified_pass()
        trace_root = root / "traces"
        trace_root.mkdir()
        trace_binaries = {}
        for variant, lib in (("baseline", library), ("candidate", candidate_library)):
            command("symbols-" + variant, ["nm", "--print-file-name", lib])
            binary = trace_root / ("trace-" + variant)
            command("build-trace-" + variant, ["gcc", *FLAGS, "-I" + str(baseline_sdk / "include"),
                    "-I" + str(source / "english_final_embedding"), "-I" + str(component),
                    component / "ekws_stream_trace.c", lib, "-Wl,--wrap=ehead_step", "-lm", "-lrt", "-o", binary])
            trace_binaries[variant] = binary
        pairs = []
        for case_id, variants in orders:
            files, parsed = {}, {}
            for variant in variants:
                output = trace_root / (case_id + "-" + variant + ".trace")
                pcm = root / "inputs" / (case_id + ".pcm")
                need(identity(pcm) == {"bytes": 512000, "sha256": TRACKS[case_id]} and identity(head) == head_pin,
                     "pre-trace input/head identity")
                meta = read_json(command("trace-" + case_id + "-" + variant,
                                 [trace_binaries[variant], head, pcm, output], model=True))
                parsed[variant] = parse_trace(output.read_bytes())
                verify_trace_metadata(meta, parsed[variant])
                need(parsed[variant]["events"] == performance[(case_id, variant)]["events"], "trace/performance events")
                need(identity(pcm) == {"bytes": 512000, "sha256": TRACKS[case_id]} and identity(head) == head_pin,
                     "post-trace input/head identity")
                files[variant] = output
                verified_pass()
            need(files["baseline"].read_bytes() == files["candidate"].read_bytes(), "full numeric trace mismatch: " + case_id)
            base, opt = performance[(case_id, "baseline")], performance[(case_id, "candidate")]
            for key in ("events", "event_digest_fnv1a64", "embeddings", "ready_outputs", "resets", "eof_padding_samples"):
                need(base[key] == opt[key], "paired stream counters/events: " + key)
            pairs.append({"case_id": case_id, "baseline": base, "candidate": opt, "numeric_trace": parsed,
                          "trace_bytes_identical": True,
                          "observed_steady_cpu_speedup": base["steady"]["accept_process_cpu_seconds"] / opt["steady"]["accept_process_cpu_seconds"],
                          "observed_steady_wall_speedup": base["steady"]["accept_wall_seconds"] / opt["steady"]["accept_wall_seconds"]})
        base_cpu = sum(p["baseline"]["steady"]["accept_process_cpu_seconds"] for p in pairs)
        opt_cpu = sum(p["candidate"]["steady"]["accept_process_cpu_seconds"] for p in pairs)
        state.update(status="BOUNDED_AB_COMPLETED", measurement_completed=True, pairs=pairs,
                     full_numeric_trace_parity=True, distinct_previously_exposed_input_seconds=32,
                     performance_audio_seconds=64, numeric_trace_replay_audio_seconds=64,
                     steady_audio_seconds_per_variant=27.6, baseline_steady_cpu_seconds_per_audio_second=base_cpu / 27.6,
                     candidate_steady_cpu_seconds_per_audio_second=opt_cpu / 27.6,
                     observed_steady_cpu_speedup=base_cpu / opt_cpu,
                     nonempty_event_sequences_compared=any(p["numeric_trace"]["baseline"]["events"] for p in pairs),
                     overall_wall_seconds=time.monotonic() - started)
    except Exception as error:
        state.update(status="FAILED_NO_RETRY", failure={"type": type(error).__name__, "message": str(error)[:1000]})
        raise
    finally:
        state["overall_wall_seconds"] = time.monotonic() - started
        checkpoint()
    print(json.dumps({k: v for k, v in state.items() if k != "pairs"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if args.execute:
        need(args.pipeline_root is not None and args.output is not None, "explicit execution paths required")
        execute(args, plan)
    else:
        print("Validated the fixed saved-input 3x1 A/B plan; no model or network calls.")
