#!/usr/bin/env python3
"""New V4.1 resource experiment; consume authenticated SDK and exposed speech."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

REPO = "jiying2007/kws-data"
SDK = {"run_id": 43984579132, "artifact_id": 14285590605,
       "name": "english-snapshot160-installed-sdk-43984579132",
       "head_sha": "cf6dbed0fb95501111820d7130bfb8ccd90e162b",
       "bytes": 10810227, "sha256": "b6411b7b859fa59147c7cd6091113ce6d83e544b75bc55bafa5726626b2ab511",
       "manifest_sha256": "09ed60d3f6701837e0e376b58bccefc4c008833c4743d53bef973b5f63d70b09",
       "result_sha256": "1647eb276f9cd9c0e67dc2e10a86bd5c2e99d621d38dec406207ad8e1705621c"}
OP = {name: i + 1 for i, name in enumerate(("Add", "Sub", "Mul", "Div", "Pow", "Max", "Log", "Clip", "LeakyRelu", "Transpose", "ReduceMax", "MatMul", "Conv", "MaxPool"))}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plan(plan):
    tracks = plan["actual_research_tracks"]
    need([t["id"] for t in tracks] == ["libri-speaker-1272", "libri-speaker-1462"], "fixed two exposed tracks required")
    need(plan["source_pool_is_previously_exposed"] is True, "exposure must remain explicit")
    need(plan["upstream_training_run_id"] == 43867764366, "unexpected source training identity")
    need(plan["corpus"]["url"] == "https://www.openslr.org/resources/12/dev-clean.tar.gz", "fixed corpus URL required")
    need(plan["corpus"]["bytes"] == 337926286 and plan["corpus"]["sha256"] == "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3", "fixed corpus identity required")
    for track in tracks:
        need(track["frames"] == 256000 and track["role"] == "train" and track["trusted_negative"] is False, "resource-only input contract")
        offset = 0
        for fragment in track["fragments"]:
            need(fragment["track_start_sample"] == offset and fragment["source_start_sample"] == 0 and fragment["samples"] > 0, "fragment geometry")
            p = PurePosixPath(fragment["source_id"])
            need(not p.is_absolute() and ".." not in p.parts and p.as_posix() == fragment["source_id"], "fragment path")
            offset += fragment["samples"]
        need(offset == track["frames"], "track sample denominator")
    need(all(re.fullmatch(r"[0-9a-f]{40}", plan[k]) for k in ("pipeline_commit", "main_before")), "source commits required")
    need(all(re.fullmatch(r"[0-9a-f]{64}", t["pcm_sha256"]) and
             all(re.fullmatch(r"[0-9a-f]{64}", f["source_pcm_sha256"]) for f in t["fragments"])
             for t in tracks), "PCM hashes must be lowercase SHA256")
    need(plan["decoder"]["numpy"] == "1.26.4" and plan["decoder"]["soundfile"] == "0.13.1"
         and plan["decoder"]["libsndfile"] == "1.2.2" and plan["decoder"]["resampling"] is False
         and plan["decoder"]["normalization"] is False and plan["decoder"]["inserted_samples"] == 0,
         "original PCM decoder contract")


def guard(plan):
    event = read_json(Path(os.environ["GITHUB_EVENT_PATH"]))
    need(os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REPOSITORY") == REPO
         and os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main"
         and os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and event.get("before") == plan["main_before"]
         and event.get("after") == os.environ.get("GITHUB_SHA")
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False, "first bounded public main successor only")


def extract_sdk(archive, output):
    need(file_identity(archive) == {"bytes": SDK["bytes"], "sha256": SDK["sha256"]}, "SDK archive identity")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [i.filename for i in infos]
        need(len(names) == 128 and len(set(names)) == len(names), "SDK closed inventory")
        need(sum(i.file_size for i in infos) < 64 * 1024**2, "SDK expansion bound")
        for info in infos:
            p = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xffff
            need(not p.is_absolute() and ".." not in p.parts and "\\" not in info.filename
                 and p.as_posix() == info.filename and not info.is_dir()
                 and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), "unsafe SDK member")
        manifest_raw = bundle.read("MANIFEST.json")
        need(sha(manifest_raw) == SDK["manifest_sha256"], "SDK manifest identity")
        manifest = json.loads(manifest_raw)
        need(set(names) == set(manifest["files"]) | {"MANIFEST.json"}, "SDK member set")
        output.mkdir()
        for name, expected in manifest["files"].items():
            raw = bundle.read(name)
            need({"bytes": len(raw), "sha256": sha(raw)} == expected, "SDK member identity: " + name)
            if name.startswith(("sdk-native/", "source/", "generated/", "licenses/")) or name == "evidence/RESULT.json":
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
        result_raw = (output / "evidence/RESULT.json").read_bytes()
        need(sha(result_raw) == SDK["result_sha256"], "SDK result identity")
        result = json.loads(result_raw)
        need(result["installed_sdk_software_verified"] is True and result["github_sha"] == SDK["head_sha"]
             and str(result["run_id"]) == str(SDK["run_id"]) and result["kws_model_delivered"] is False
             and result["formal_training_admitted"] is False, "SDK evidence boundary")
        write_json(output / "MANIFEST.json", manifest)


def prepare_pcm(corpus, plan, output):
    import numpy as np
    import soundfile as sf
    need(np.__version__ == "1.26.4" and sf.__version__ == "0.13.1" and sf.__libsndfile_version__ == "1.2.2", "fixed decoder versions")
    need(file_identity(corpus) == {"bytes": plan["corpus"]["bytes"], "sha256": plan["corpus"]["sha256"]}, "corpus identity")
    output.mkdir()
    cases = []
    with tarfile.open(corpus, "r:gz") as archive:
        for track in plan["actual_research_tracks"]:
            parts = []
            for fragment in track["fragments"]:
                member = archive.getmember(fragment["source_id"])
                need(member.isfile() and member.size < 32 * 1024**2, "regular bounded FLAC required")
                stream = archive.extractfile(member)
                need(stream is not None, "missing corpus member")
                with sf.SoundFile(io.BytesIO(stream.read())) as audio:
                    need(audio.samplerate == 16000 and audio.channels == 1, "native mono16k required")
                    raw = np.asarray(audio.read(dtype="int16"), dtype="<i2").tobytes()
                need(sha(raw) == fragment["source_pcm_sha256"], "original decoded PCM mismatch")
                count = fragment["samples"] * 2
                need(len(raw) >= count, "source fragment too short")
                parts.append(raw[:count])
            pcm = b"".join(parts)
            need(len(pcm) == track["frames"] * 2 and sha(pcm) == track["pcm_sha256"], "reconstructed existing track mismatch")
            name = track["id"] + ".pcm"
            (output / name).write_bytes(pcm)
            cases.append({"id": track["id"], "path": name, "sha256": track["pcm_sha256"], "samples": track["frames"],
                          "sample_rate_hz": 16000, "encoding": "s16le_mono", "speech_kind": "natural_speech",
                          "continuity": "concatenated_recordings", "role": "train", "exposure": "previously_exposed",
                          "kind": track["kind"], "source": {"training_run_id": plan["upstream_training_run_id"],
                          "corpus": plan["corpus"], "track": track, "upstream_member_sha256": plan["upstream_member_sha256"]}})
    manifest = {"schema": "english-stream-performance-inputs-v1", "cases": cases}
    write_json(output / "inputs.json", manifest)
    return manifest


def verify_diagnostic(profile, measured, baseline, exported):
    need(measured.get("measurement_completed") is True and measured.get("score_observer_enabled") is False, "diagnostic did not complete")
    for key in ("input_samples", "embeddings", "ready_outputs", "events", "event_digest_fnv1a64", "resets", "eof_padding_samples"):
        need(measured[key] == baseline[key], "diagnostic event/counter mismatch: " + key)
    expected = {}
    for name, file in (("mel", "melspectrogram.onnx"), ("embedding", "embedding_model.onnx")):
        expected[(name, -1)] = -1
        for index, node in enumerate(exported["graphs"][file]["nodes"]):
            expected[(name, index)] = OP[node["op"]]
    need(profile.get("schema") == "english-graph-diagnostic-v1" and profile.get("valid") is True and profile.get("instrumented") is True
         and profile.get("headline_performance") is False and len(profile.get("rows", [])) == len(expected) == 81, "complete diagnostic required")
    observed = set()
    for row in profile["rows"]:
        key = (row["graph"], row["node"])
        need(key not in observed and key in expected and row["op"] == expected[key]
             and row["calls"] == measured["embeddings"], "diagnostic node identity/call count")
        observed.add(key)
        for field in ("wall_seconds", "process_cpu_seconds"):
            need(math.isfinite(row[field]) and row[field] >= 0, "invalid diagnostic timing")


def execute(args, plan):
    guard(plan)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    state = {"schema": "english-stream-resource-execution-v1", "status": "RUNNING", "measurement_completed": False,
             "run_id": os.environ["GITHUB_RUN_ID"], "github_sha": os.environ["GITHUB_SHA"], "pipeline_commit": plan["pipeline_commit"],
             "sdk": SDK, "tts_asr_phone_calls": 0, "optimizer_updates": 0, "independent_far_test_hours": 0,
             "formal_training_admitted": False, "pronunciation_verified": False, "model_promotion_allowed": False,
             "kws_model_delivered": False, "ssc305": "NOT_RUN", "far_frr": "NOT_MEASURED", "word_end_latency": "NOT_MEASURED",
             "retry_count": 0, "performance_budget_evaluated": False, "model_passes_planned": 3,
             "model_passes_completed": 0}
    write_json(evidence / "RESULT.json", state)

    def command(label, argv, limit=120, auth=False, extra_env=None):
        remaining = 720 - (time.monotonic() - started)
        need(remaining > 0, "experiment deadline")
        env = dict(os.environ)
        if not auth:
            for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
                env.pop(key, None)
        if extra_env:
            env.update(extra_env)
        begin = time.monotonic()
        returncode = None
        error_type = None
        try:
            with (evidence / (label + ".stdout")).open("xb") as stdout, (evidence / (label + ".stderr")).open("xb") as stderr:
                process = subprocess.Popen(list(map(str, argv)), stdout=stdout, stderr=stderr,
                                           env=env, start_new_session=True)
                try:
                    returncode = process.wait(timeout=min(limit, remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
        except Exception as error:
            error_type = type(error).__name__
            raise
        finally:
            write_json(evidence / (label + "-exit.json"), {"argv": list(map(str, argv)), "returncode": returncode,
                       "error_type": error_type, "wall_seconds": time.monotonic() - begin})
        need(returncode == 0, "command failed: " + label)
        return evidence / (label + ".stdout")

    try:
        pipeline = args.pipeline_root.resolve()
        commit = subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip()
        need(commit == plan["pipeline_commit"], "performance source commit mismatch")
        component = pipeline / "research/english_stream_performance"
        write_json(evidence / "PLAN.json", plan)
        write_json(evidence / "SOURCE.json", {p.name: file_identity(p) for p in sorted(component.iterdir()) if p.is_file()})
        for name, target in (("sdk-run", "runs/" + str(SDK["run_id"])), ("sdk-artifact", "artifacts/" + str(SDK["artifact_id"]))):
            command(name, ["gh", "api", "repos/" + REPO + "/actions/" + target], auth=True)
        run = read_json(evidence / "sdk-run.stdout")
        artifact = read_json(evidence / "sdk-artifact.stdout")
        need(run["id"] == SDK["run_id"] and run["head_sha"] == SDK["head_sha"] and run["conclusion"] == "success"
             and run["status"] == "completed" and run["head_branch"] == "main" and run["event"] == "push"
             and run["run_attempt"] == 1 and run["repository"]["full_name"] == REPO
             and run["repository"]["private"] is False, "source run mismatch")
        need(artifact["id"] == SDK["artifact_id"] and artifact["name"] == SDK["name"] and not artifact["expired"]
             and artifact["size_in_bytes"] == SDK["bytes"] and artifact["digest"] == "sha256:" + SDK["sha256"]
             and artifact["workflow_run"]["id"] == SDK["run_id"] and artifact["workflow_run"]["head_sha"] == SDK["head_sha"], "source artifact mismatch")
        sdk_zip = command("sdk-archive", ["gh", "api", "repos/" + REPO + "/actions/artifacts/" + str(SDK["artifact_id"]) + "/zip"], limit=120, auth=True)
        sdk = output / "sdk"
        extract_sdk(sdk_zip, sdk)
        corpus = output / "dev-clean.tar.gz"
        with urllib.request.urlopen(plan["corpus"]["url"], timeout=45) as response, corpus.open("xb") as stream:
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                need(count <= plan["corpus"]["bytes"] and time.monotonic() - started < 360, "corpus download bound")
                stream.write(block)
        manifest = prepare_pcm(corpus, plan, output / "inputs")
        host = {"uname": list(os.uname()), "cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
                "cpuinfo": Path("/proc/cpuinfo").read_text(), "steady_pacing": False}
        write_json(evidence / "HOST.json", host)
        state["stage"] = "baseline_execution"
        write_json(evidence / "RESULT.json", state)
        command("baseline", [sys.executable, component / "run_benchmark.py", "--sdk-root", sdk / "sdk-native",
                "--manifest", output / "inputs/inputs.json", "--output", output / "baseline", "--case-timeout", "180"], limit=420)
        baseline = read_json(output / "baseline/RESULT.json")
        need(baseline["passed"] is True and baseline["actual_input_seconds"] == 32 and len(baseline["cases"]) == 2, "baseline result")
        state["model_passes_completed"] = 2
        diagnostic = output / "diagnostic"
        diagnostic.mkdir()
        source = sdk / "source/research"
        command("instrument", [sys.executable, component / "instrument_graph.py", source / "english_mel_specialization/fixed_frontend_mel.c", diagnostic / "frontend.c"])
        includes = [component, source / "english_stream_integration", source / "english_final_embedding_frontend", source / "english_final_embedding", sdk / "generated"]
        executable = diagnostic / "benchmark"
        command("build-diagnostic", ["gcc", "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                *["-I" + str(p) for p in includes], component / "ekws_stream_benchmark.c", component / "ekws_graph_profile.c",
                diagnostic / "frontend.c", source / "english_stream_integration/ekws_research_stream.c",
                source / "english_final_embedding/english_causal_head.c", sdk / "generated/fixed_models.c", "-lm", "-lrt", "-o", executable])
        state["stage"] = "diagnostic_execution"
        write_json(evidence / "RESULT.json", state)
        measured_path = command("diagnostic", [executable, sdk / "sdk-native/models/causal-head.bin", output / "inputs" / manifest["cases"][0]["path"]],
                                limit=180, extra_env={"EKWS_GRAPH_PROFILE_PATH": str(diagnostic / "graph-profile.json")})
        measured = read_json(measured_path)
        profile = read_json(diagnostic / "graph-profile.json")
        exported = read_json(sdk / "generated/export_manifest.json")
        verify_diagnostic(profile, measured, baseline["cases"][0]["measurements"], exported)
        state["model_passes_completed"] = 3
        write_json(diagnostic / "VERIFIED.json", {"event_and_counter_parity": True, "logit_parity_checked": False,
                   "instrumented": True, "node_rows": len(profile["rows"]), "binary": file_identity(executable),
                   "source_copy": file_identity(diagnostic / "frontend.c"), "measurements": measured})
        state.update(status="RESOURCE_MEASUREMENT_COMPLETED", measurement_completed=True,
                     baseline_input_seconds=32, additional_diagnostic_replay_seconds=16,
                     independent_input_seconds_claimed=0, diagnostic_event_and_counter_parity=True,
                     overall_wall_seconds=time.monotonic() - started)
    except Exception as error:
        state.update(status="FAILED_NO_RETRY", failure={"type": type(error).__name__, "message": str(error)[:1000]})
        raise
    finally:
        partial_baseline = output / "baseline/RESULT.json"
        if partial_baseline.is_file() and state["model_passes_completed"] < 2:
            partial = read_json(partial_baseline)
            state["model_passes_completed"] = sum(c.get("status") == "completed" for c in partial.get("cases", []))
        write_json(evidence / "RESULT.json", state)
    print(json.dumps(state, sort_keys=True))


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
        need(args.pipeline_root is not None and args.output is not None, "execution paths required")
        execute(args, plan)
    else:
        print("Validated fixed V4.1 resource input plan; no model or network calls.")
