#!/usr/bin/env python3
"""One bounded, source-pinned English63 research request; no generic command runner."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

REPOSITORY = "jiying2007/kws-data"
PIPELINE_REPOSITORY = "jiying2007/kws-pipeline"
ENTRYPOINT = "research/english63_online_pilot/online_pilot.py"
REQUEST_DIR = ".github/research-v4/requests"
REQUEST_SCHEMA = "kws-research-v4-request-v1"
ARTIFACT_LIMIT = 64 * 1024 * 1024
INPUTS = {
    "tts": {
        "artifact_id": 13954744587,
        "zip_sha256": "f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc",
    },
    "review": {
        "artifact_id": 14001747172,
        "zip_sha256": "f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8",
        "member": "english63-three-evidence-review.json",
        "member_sha256": "89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12",
    },
}
FIXED = {
    "schema": REQUEST_SCHEMA,
    "experiment": "english63-online-weak-mil",
    "pipeline_repository": PIPELINE_REPOSITORY,
    "entrypoint": ENTRYPOINT,
    "source_artifacts": INPUTS,
    "optimizer_updates_per_model": 48,
    "max_training_seconds": 1200,
    "max_replay_seconds": 300,
    "new_audio_generated": False,
    "new_asr_calls": 0,
    "formal_training_admitted": False,
    "model_promotion_allowed": False,
}
ROLES = {
    "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY": 30,
    "EXPOSED_DEVELOPMENT_NO_TUNE": 12,
    "QUARANTINE_AMBIGUOUS_NEARPHONE": 21,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def valid_sha(value, length=40):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) and set(value) != {"0"}


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    require(len(raw) < 16384, "request too large")
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(value, dict) and set(value) == set(FIXED) | {"request_id", "pipeline_commit", "reason"}, "request keys differ from fixed experiment")
    for key, expected in FIXED.items():
        require(value[key] == expected and type(value[key]) is type(expected), "unsupported request field: " + key)
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", value["request_id"] or ""), "invalid request_id")
    require(path.name == value["request_id"] + ".json", "request filename must match request_id")
    require(valid_sha(value["pipeline_commit"]), "pipeline_commit must be a reviewed full commit SHA")
    require(isinstance(value["reason"], str) and 1 <= len(value["reason"].strip()) <= 500, "request needs a bounded execution reason")
    return value, raw, sha256(raw)


def output(**values):
    destination = os.environ.get("GITHUB_OUTPUT")
    for key, value in values.items():
        text = str(value).lower() if isinstance(value, bool) else str(value)
        require("\n" not in text, "invalid action output")
        if destination:
            with open(destination, "a", encoding="utf-8") as stream:
                stream.write(key + "=" + text + "\n")
        print(key + "=" + text)


def changed_request(lines, allow_empty):
    rows = [line.split("\t") for line in lines.splitlines() if line]
    require(all(len(row) == 2 and row[0] == "A" for row in rows), "consumed requests are immutable: add a new version instead")
    require(len(rows) <= 1 and (allow_empty or len(rows) == 1), "one new request per main event is required")
    if not rows:
        return None
    path = rows[0][1]
    require(re.fullmatch(re.escape(REQUEST_DIR) + r"/[a-z0-9][a-z0-9-]{2,79}\.json", path), "request must be a versioned JSON directly in requests/")
    return path


def native_consumer(request_path):
    """Only this named, no-training consumer may extend the original task."""
    value = read_json(request_path)
    if not isinstance(value, dict) or value.get("experiment") != "english63-saved-native-c-parity":
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "research_v4_native_c", Path(__file__).with_name("research_v4_native_c.py"))
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    return consumer


def source_check():
    event = read_json(os.environ["GITHUB_EVENT_PATH"])
    name = os.environ["GITHUB_EVENT_NAME"]
    require(name in ("push", "pull_request"), "only source PR or main request push is supported")
    before = event["pull_request"]["base"]["sha"] if name == "pull_request" else event["before"]
    after = event["pull_request"]["head"]["sha"] if name == "pull_request" else event["after"]
    require(valid_sha(before) and valid_sha(after), "source comparison needs full nonzero SHAs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(head == after, "checkout is not the event's exact source commit")
    changes = subprocess.check_output(["git", "diff", "--name-status", "--no-renames", before, after, "--", REQUEST_DIR], text=True)
    path = changed_request(changes, allow_empty=name == "pull_request")
    if path:
        consumer = native_consumer(path)
        request, _, digest = (consumer.load_request if consumer else load_request)(path)
        output(request_path=path, request_sha256=digest, pipeline_commit=request["pipeline_commit"],
               experiment=request["experiment"], job_timeout_minutes=40 if consumer else 45)
    else:
        output(request_path="", request_sha256="", pipeline_commit="", experiment="", job_timeout_minutes=45)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def github_get(path, *, archive=False):
    token = os.environ.get("GH_TOKEN", "")
    require(token, "same-repository Actions token is required only in prepare")
    request = urllib.request.Request("https://api.github.com/repos/" + REPOSITORY + path, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kws-data-research-v4",
    })
    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        require(archive and exc.code in (301, 302, 303, 307, 308), "GitHub artifact/API request failed: HTTP " + str(exc.code))
        target = exc.headers.get("Location", "")
        require(target.startswith("https://"), "artifact redirect must use HTTPS")
        # Signed blob URL receives no GitHub Authorization header.
        response = urllib.request.urlopen(urllib.request.Request(target, headers={"User-Agent": "kws-data-research-v4"}), timeout=60)
    with response:
        data = response.read(ARTIFACT_LIMIT + 1)
    require(len(data) <= ARTIFACT_LIMIT, "artifact/API response exceeds fixed byte budget")
    return data if archive else json.loads(data, object_pairs_hook=unique_object)


def artifact_metadata(artifact_id, expected_digest=None):
    item = github_get("/actions/artifacts/" + str(artifact_id))
    require(item.get("id") == artifact_id and item.get("expired") is False, "original artifact missing or expired; do not regenerate")
    digest = item.get("digest", "")
    require(isinstance(digest, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", digest), "artifact has no verifiable digest")
    if expected_digest:
        require(digest == "sha256:" + expected_digest, "source artifact metadata digest changed")
    return item


def artifact_bytes(item):
    require(item.get("expired") is False, "prior artifact expired; execution remains consumed")
    data = github_get("/actions/artifacts/" + str(item["id"]) + "/zip", archive=True)
    require("sha256:" + sha256(data) == item.get("digest"), "downloaded artifact digest mismatch")
    return data


def named_artifacts(name):
    listing = github_get("/actions/artifacts?per_page=100&name=" + name)
    require(listing.get("total_count", 0) <= 100, "too many prior artifacts; stop without retraining")
    return [item for item in listing["artifacts"] if item["name"] == name]


def restore_archive(archive, destination):
    destination.mkdir()
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        require(len(infos) <= 32 and sum(info.file_size for info in infos) <= ARTIFACT_LIMIT, "prior result exceeds fixed output budget")
        names = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename and info.filename not in names, "unsafe/duplicate result member")
            require((info.external_attr >> 16) & 0o170000 != 0o120000, "result symlinks forbidden")
            names.add(info.filename)
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(handle.read(info))


def validate_model_result(result):
    model = result / "model"
    summary = read_json(model / "SUMMARY.json")
    replay = read_json(model / "REPLAY.json")
    require(summary.get("schema") == "english63-bounded-online-weak-mil-v1" and summary.get("status") == "EXECUTED_WEAK_MIL_ONLINE_PROPOSALS_NOT_KWS_QUALIFICATION", "missing successful model research receipt")
    for key in ("formal_training_admitted", "human_gold", "model_promotion_allowed"):
        require(summary.get(key) is False, "summary attempts scientific promotion: " + key)
    for key in ("independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile"):
        require(summary.get(key) == "NOT_RUN", "pilot cannot claim qualification: " + key)
    require(summary.get("source_audio_clips") == 63 and summary.get("source_pcm_samples") == 2101760 and summary.get("role_counts") == ROLES, "source/role counts changed")
    require(replay.get("schema") == "english63-online-saved-weight-replay-v1" and replay.get("status") == "REPRODUCED_SAVED_WEAK_PROPOSALS", "saved-weight replay did not succeed")
    require(replay.get("optimizer_updates") == 0 and replay.get("formal_training_admitted") is False and replay.get("model_promotion_allowed") is False, "replay must not train/promote")
    for filename, key in (("SUMMARY.json", "summary_sha256"), ("events.jsonl", "events_sha256")):
        require(sha256((model / filename).read_bytes()) == replay.get(key), "replay hash mismatch: " + filename)
    require(len(summary.get("models", [])) == len(replay.get("models", [])) == 2, "exactly two fixed candidates required")
    summaries = {row.get("kind"): row for row in summary["models"]}
    replays = {row.get("kind"): row for row in replay["models"]}
    require(set(summaries) == set(replays) == {"gru", "dsc_tcn"}, "candidate roster changed")
    for kind, row in summaries.items():
        filename = kind + "-weights.npz"
        require(row.get("weight_file") == filename and (model / filename).stat().st_size > 0, "actual saved checkpoint missing")
        digest = sha256((model / filename).read_bytes())
        require(row.get("weights_sha256") == digest and replays[kind].get("weights_sha256") == digest, "checkpoint identity differs from summary/replay")
        require(row.get("optimizer_updates") == 48 and row.get("train_rows") == 30 and row.get("train_nearphone_rows") == 0 and row.get("train_development_rows") == 0 and row.get("model_promotion_allowed") is False, "training budget or role isolation changed")
        require(replays[kind].get("saved_predictions_reproduced") is True and replays[kind].get("clip_comparisons") == 63 and replays[kind].get("continuous_comparisons") == 1, "incomplete saved-weight readback")
    evidence = read_json(result / "keyword-evidence" / "report.json")
    require(evidence.get("schema") == "kws-v4-existing-keyword-evidence-v1" and evidence.get("audio_rows") == 63 and evidence.get("keyword_rows") == 126 and evidence.get("asr_observations_reused") == 189, "keyword evidence incomplete")
    require(evidence.get("new_model_calls") == 0 and evidence.get("training_admitted") is False and evidence.get("human_gold") is False and evidence.get("accepted_negative_seconds") == 0, "keyword evidence cannot promote labels")
    require((result / "keyword-evidence" / "rows.jsonl").is_file(), "word-level evidence rows missing")


def file_manifest(result):
    files = [path for path in result.rglob("*") if path.is_file() and path != result / "TERMINAL.json"]
    require(all(not path.is_symlink() for path in result.rglob("*")), "result symlinks forbidden")
    require(len(files) <= 24 and sum(path.stat().st_size for path in files) <= ARTIFACT_LIMIT, "result exceeds fixed output budget")
    return {path.relative_to(result).as_posix(): sha256(path.read_bytes()) for path in sorted(files)}


def validate_terminal(result, request, digest):
    terminal = read_json(result / "TERMINAL.json")
    require(terminal.get("schema") == "kws-research-v4-terminal-v1" and terminal.get("request_sha256") == digest and terminal.get("pipeline_commit") == request["pipeline_commit"], "terminal belongs to another request/code")
    require(terminal.get("request_id") == request["request_id"] and all(terminal.get(key) is False for key in ("formal_training_admitted", "human_gold", "model_promotion_allowed")), "terminal scientific boundary changed")
    require(sha256((result / "REQUEST.json").read_bytes()) == digest, "terminal request bytes changed")
    require(terminal.get("files_sha256") == file_manifest(result), "terminal file/checkpoint manifest changed")
    require(terminal.get("status") == "SUCCEEDED", "prior execution failed; consumed request must not auto-retry")
    execution = read_json(result / "EXECUTION.json")
    require(execution.get("request_sha256") == digest and execution.get("train_exit_code") == execution.get("replay_exit_code") == 0, "terminal lacks actual execution and no-training readback")
    validate_model_result(result)
    return terminal


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = read_json(os.environ["GITHUB_EVENT_PATH"])
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY and os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main" and event["repository"]["private"] is False, "actual execution is public main only")
    state.mkdir()
    result_name, claim_name = "research-v4-result-" + digest, "research-v4-claim-" + digest
    terminals, claims = named_artifacts(result_name), named_artifacts(claim_name)
    require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous duplicate receipts; stop without retraining")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(artifact_bytes(terminals[0]))
        restore_archive(archive, state / "reused")
        validate_terminal(state / "reused", request, digest)
        output(execute=False, request_sha256=digest, result_artifact=result_name, prior_artifact_id=terminals[0]["id"])
        return
    require(not claims, "prior start claim exists without a valid terminal; do not repeat an unknown/failed execution")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "rerun without retained receipts is blocked; no silent repeated training")
    (state / "inputs").mkdir()
    (state / "result").mkdir()
    (state / "claim").mkdir()
    (state / "result" / "REQUEST.json").write_bytes(raw)
    for key, specification in INPUTS.items():
        metadata = artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        if key == "tts":
            require(metadata.get("workflow_run", {}).get("id") == 43242278962 and metadata.get("workflow_run", {}).get("head_sha") == "bec1a1421d1db2de17afb9c6e3155343becdb980", "original audio run identity changed")
        (state / "inputs" / (key + ".zip")).write_bytes(artifact_bytes(metadata))
    with zipfile.ZipFile(state / "inputs" / "review.zip") as handle:
        names = handle.namelist()
        require(names.count(INPUTS["review"]["member"]) == 1, "review member absent or duplicated")
        info = handle.getinfo(INPUTS["review"]["member"])
        require(info.file_size <= 8 * 1024 * 1024, "review member exceeds byte budget")
        review = handle.read(info)
    require(sha256(review) == INPUTS["review"]["member_sha256"], "review JSON digest changed")
    (state / "inputs" / "review.json").write_bytes(review)
    write_json(state / "claim" / "CLAIM.json", {
        "schema": "kws-research-v4-start-claim-v1", "request_sha256": digest,
        "pipeline_commit": request["pipeline_commit"], "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "status": "CONSUMED_BEFORE_MODEL_START",
    })
    output(execute=True, request_sha256=digest, result_artifact=result_name, claim_artifact=claim_name, prior_artifact_id="")


def child_environment(python, state):
    # Deliberate allowlist: GitHub, artifact, provider and checkout credentials are absent.
    temporary = state / "child-temp"
    temporary.mkdir(exist_ok=True)
    return {
        "PATH": str(Path(python).parent) + ":/usr/bin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": str(temporary),
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    }


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() == request["pipeline_commit"], "pipeline checkout SHA mismatch")
    script = pipeline / ENTRYPOINT
    require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()), "fixed model entrypoint missing")
    require(read_json(state / "claim" / "CLAIM.json")["request_sha256"] == digest, "start claim differs")
    result = state / "result"
    environment = child_environment(python, state)
    base = [str(python), "-I", "-B", str(script), "--tts-zip", str(state / "inputs" / "tts.zip"), "--review-json", str(state / "inputs" / "review.json")]
    execution = {"schema": "kws-research-v4-execution-v1", "request_sha256": digest, "train_exit_code": None, "replay_exit_code": None, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        for phase, argument, seconds in (("train", "--output-dir", request["max_training_seconds"]), ("replay", "--verify-result-dir", request["max_replay_seconds"])):
            with (result / (phase + ".log")).open("xb") as log:
                done = subprocess.run(base + [argument, str(result / "model")], cwd=pipeline, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=seconds, check=False)
            execution[phase + "_exit_code"] = done.returncode
            require(done.returncode == 0, phase + " process failed; retain partial output")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(result / "EXECUTION.json", execution)
    require(execution["train_exit_code"] == execution["replay_exit_code"] == 0, "bounded model execution/readback failed; inspect retained logs")


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {
        "schema": "kws-research-v4-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": request["request_id"],
        "pipeline_commit": request["pipeline_commit"], "data_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        execution = read_json(result / "EXECUTION.json")
        require(execution.get("train_exit_code") == execution.get("replay_exit_code") == 0, "model execution/readback incomplete")
        validate_model_result(result)
        terminal["status"] = "SUCCEEDED"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        terminal["error"] = type(exc).__name__ + ": " + str(exc)
    terminal["files_sha256"] = file_manifest(result)
    write_json(result / "TERMINAL.json", terminal)
    output(terminal_status=terminal["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "prepare", "execute", "seal", "check"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.command == "source":
        source_check()
        return
    require(args.request is not None and args.state is not None and args.state.is_absolute(), "request and absolute state directory required")
    consumer = native_consumer(args.request)
    if consumer:
        consumer.dispatch(args)
        return
    if args.command == "prepare":
        prepare(args.request, args.state)
    elif args.command == "execute":
        require(args.pipeline is not None and args.pipeline.is_absolute() and args.python is not None and args.python.is_absolute(), "absolute checkout and Python executable required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest, "formal_training_admitted": False, "model_promotion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(1)
