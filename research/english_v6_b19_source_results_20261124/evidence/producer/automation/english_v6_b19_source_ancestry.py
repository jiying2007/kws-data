"""Original B18 STOP proof and complete exposure history before any B19 call."""
from __future__ import annotations
import array
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import english_v6_b18_source as previous

qa, native = previous.qa, previous.observers
PLANS = Path(__file__).with_name("english_v6_b19_panels")
ASSESSMENT = PLANS / "assessment.json"
ASSESSMENT_SHA = "b88f91dcc9ca98c548a34be746c13de3f815663310ce093fd5559ca38e04c365"
BASE_ANCESTRY_SHA = previous.lineage.BASE_ANCESTRY_SHA
SCHEMA = "english-v6-b19-original-stop-proof-and-source-history-v1"
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b19-source-feasibility-once-20261124"
sha, need = previous.lineage.sha, qa.need


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def read_plan():
    raw = ASSESSMENT.read_bytes()
    need(sha(raw) == ASSESSMENT_SHA, "prospective source-only canary assessment")
    design = json.loads(raw)
    previous.lineage.read_plan()
    need(design["schema"] == "english-v6-b19-source-only-feasibility-assessment-v1"
         and design["B18_source_assessment_sha256"] == previous.ASSESSMENT_SHA
         and design["base_source_ancestry_sha256"] == BASE_ANCESTRY_SHA
         and design["source_original_slots"] == 48 and design["source_speakers"] == 8
         and design["maximum_logical_TTS_requests"] == design["maximum_public_Piper_synthesize_calls"]
             == design["maximum_native_observer_attempts"] == 48
         and design["source_gate_before_observer"] == dict(all_48_basic_audio_ok=True,
             all_positive_originals=32, positive_min_samples=6400, positive_max_samples=28672,
             whole_originals_only=True, padding_allowed=False, cropping_allowed=False, replacement_allowed=False)
         and all(design[key] is False for key in ("new_training_allowed", "source_admission_eligible",
             "model_evaluation_allowed", "promotion_allowed", "replication_opened", "threshold_search",
             "future_source_execution_allowed", "B17_missing_raw_capture_cause_inferred"))
         and design["new_KWS_forwards"] == design["new_natural_source_calls"] == design["new_natural_observer_calls"] == 0
         and all(design[key] is True for key in ("B17_source_failure_preserved", "B18_source_STOP_preserved",
             "all_old_replication_sealed", "canary_permanently_excluded_from_training_and_KWS_assessment",
             "all_8_planned_identities_excluded_even_if_never_called", "all_generated_PCM_permanently_excluded")),
         "bounded diagnostic cannot authorize source admission, model scoring or release")
    for role, first, last in (("feasibility", 790, 798), ("future_primary", 798, 822), ("future_replication", 822, 846)):
        need(set(design["source_maps"][role]) == {str(i) for i in range(first, last)}, "only preregistered canonical source roles")
    return design


def validate_identity(identity):
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and isinstance(identity["run_id"], str) and re.fullmatch(r"[1-9][0-9]*", identity["run_id"]) is not None
         and re.fullmatch(r"[0-9a-f]{40}", identity["run_head"]) is not None
         and re.fullmatch(r"[0-9a-f]{40}", identity["prepared_parent"]) is not None
         and identity["run_head"] != identity["prepared_parent"]
         and type(identity["run_attempt"]) is int and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH, "actual first public diagnostic source identity")
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "reviewed exact predecessor required; inert placeholder cannot run")
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "first designated public B19 diagnostic only")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
                                     prepared_parent=prepared_parent, run_branch=BRANCH, run_attempt=1))
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event.get("repository", {}).get("private") is False,
         "one public reviewed predecessor, no retry, force push or multi-commit jump")
    return identity


def proof_locks():
    proof = read_plan()["original_B18_source_proof"]
    return {"source": proof["source"], **proof["observers"]}


def required_archives():
    return tuple(dict.fromkeys((*previous.required_archives(), *(str(pin["id"]) for pin in proof_locks().values()))))


def verify_b18(directory):
    """Revalidate saved returns only; no renderer, backend loader or model call."""
    design, pins = read_plan(), proof_locks()
    proof = design["original_B18_source_proof"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for role, pin in pins.items():
            with previous.lineage.opened(directory, pin) as archive:
                previous.lineage.extract_original(archive, root / role)
        rows = previous.read_source(root / "source")
        registration = previous.registration(root / "source")
        source_sha = qa.sha(root / "source/MANIFEST.json")
        need(registration["run_id"] == str(proof["run_id"]) and registration["run_head"] == proof["run_head"]
             and registration["prepared_parent"] == proof["prepared_parent"], "actual original B18 source execution")
        totals = dict(attempted=0, returned=0, valid=0, failed=0, not_started=0)
        for provider in native.PROVIDERS:
            folder = root / provider
            report, start, model = [json.loads((folder / name).read_bytes()) for name in ("RESULT.json", "STARTED.json", "MODEL.json")]
            need(json.loads((folder / "FILES.json").read_bytes()) == {
                 "schema": "english-v6-b18-observer-file-closure-v1", "self_excluded": "FILES.json",
                 "files": native.file_closure(folder)}, "complete original native byte closure")
            need(start["registration"] == registration and start["provider"] == provider
                 and report["source_manifest_sha256"] == source_sha and report["denominator"] == 96
                 and report["observer_code_sha256"] == registration["observer_sha256"]
                 and report["asr_adapter_sha256"] == registration["asr_adapter_sha256"]
                 and report["kws_forward_calls"] == report["new_tts_calls"] == report["retry_count"] == 0,
                 "original native run, raw parser and no replay")
            ledger = native.row_ledger(folder, rows)
            counts = native.ledger_counts(ledger)
            need(json.loads((folder / "LEDGER.json").read_bytes()) == {
                 "schema": "english-v6-b18-native-attempt-return-valid-ledger-v1", "provider": provider,
                 "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
                 "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
                 "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
                 "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}, "actual original attempt/return/valid ledger")
            observed = dict(attempted=counts["forward_attempted"], returned=counts["native_returned"],
                            valid=counts["payload_valid"], failed=counts["failed"], not_started=96-counts["input_started"])
            need(observed == proof["provider_counts"][provider]
                 and report["actual_native_forward_attempts"] == observed["attempted"]
                 and report["native_forward_returns"] == observed["returned"]
                 and report["valid_native_payloads"] == report["complete"] == observed["valid"], "same actual B18 accounting")
            captured = {r["id"]: r for r in report["rows"]}
            need(len(captured) == len(report["rows"]) == 96 and set(captured) == {r["id"] for r in rows}, "all original B18 slots retained")
            for row in rows:
                capture = captured[row["id"]]
                if capture["status"] != "success":
                    continue
                path = folder / "rows" / row["id"]
                binding = native.binding(provider, row, source_sha)
                need(json.loads((path / "STARTED.json").read_bytes()) == binding
                     and json.loads((path / "CAPTURE.json").read_bytes()) == capture,
                     "unchanged source-bound native capture")
                if provider in native.raw_asr.TEXT_PROVIDERS:
                    native.raw_asr.validate_capture(provider, path, binding, capture)
                else:
                    native._verify_phone(folder, path, binding, capture, model)
                    values = array.array("f", (folder / "logits" / (row["id"] + ".f32")).read_bytes())
                    if sys.byteorder != "little": values.byteswap()
                    need(all(math.isfinite(value) for value in values), "complete finite original native phone bytes")
            if provider == "parakeet":
                need(report["status"] == "STOP_OBSERVER_FAILURE" and report["whole_panel_stop"] is True
                     and report["row_error"] == "ValueError: EMPTY_NATIVE_TRANSCRIPT"
                     and sum(value["status"] == "not_started" for value in captured.values()) == 90,
                     "original invalid sixth return stopped remaining calls")
            else:
                need(native.complete_ledger(ledger) and report["status"] == "COMPLETE"
                     and report["whole_panel_stop"] is False, "all four other native observers really completed")
            for key in totals: totals[key] += observed[key]
        need(totals == proof["total_native_observers"] == dict(attempted=390, returned=390, valid=389, failed=1, not_started=90),
             "original B18 STOP cannot become a completed five-by-96 panel")
        failed = proof["failed_original"]
        path = root / "parakeet/rows" / failed["id"]
        row = next(r for r in rows if r["id"] == failed["id"])
        binding = native.binding("parakeet", row, source_sha)
        attempted, returned, process = [json.loads((path / name).read_bytes()) for name in
                                       ("FORWARD_STARTED.json", "FORWARD_RETURNED.json", "RAW_PROCESS.json")]
        need(all(all(receipt[k] == value for k, value in binding.items()) for receipt in (attempted, returned, process))
             and attempted["attempt"] == 1 and returned["native_exit_code"] == process["native_exit_code"] == 0
             and returned["native_return_observed"] is process["native_return_observed"] is True
             and process["process_started"] is True and process["timed_out"] is False and process["exception_type"] is None,
             "actual retained native return, no inferred old B17 cause")
        command = native.raw_asr._read_pinned(path, process["command"])
        stdout = native.raw_asr._read_pinned(path, process["stdout"])
        stderr = native.raw_asr._read_pinned(path, process["stderr"])
        need(attempted["native_command_sha256"] == sha(command) and stdout == b"\n"
             and len(stderr) == failed["raw_stderr_bytes"] == 203 and sha(stderr) == failed["raw_stderr_sha256"]
             and sha(stdout) == failed["raw_stdout_sha256"] and row["pcm_sha256"] == failed["pcm_sha256"]
             and row["samples"] == failed["source_samples"] == 11518
             and not (path / "CAPTURE.json").exists() and not (path / "PAYLOAD_VALID.json").exists(),
             "exact saved one-byte LF remains invalid under the unchanged B18 rule")
        return ({r["pcm_sha256"] for r in rows}, {r["source_group_id"] for r in rows},
                {r["source_speaker_id"].removeprefix("p") for r in rows})


def validate_sets(seen, groups, people):
    expected = read_plan()["actual_complete_history_inventory"]
    for name, values in zip(("all_pcm_sha256", "all_source_groups", "canonical_people"), (seen, groups, people), strict=True):
        need(len(values) == expected[name]["count"] and sha(canonical(sorted(values))) == expected[name]["sorted_canonical_json_sha256"],
             "complete frozen historical PCM/group/canonical identity set " + name)
    return seen, groups, people


def registered_history():
    proof = read_plan()["original_B18_source_proof"]
    documents = {}
    for name in ("MANIFEST.json", "HISTORY.json"):
        pin = proof["durable_evidence"][name]
        raw = (PLANS.parent.parent / pin["path"]).read_bytes()
        need(sha(raw) == pin["sha256"], "retained actual B18 source member " + name)
        documents[name] = json.loads(raw)
    seen, groups, people = previous.lineage.validate_history(documents["HISTORY.json"])
    rows = documents["MANIFEST.json"]["rows"]
    seen.update(row["pcm_sha256"] for row in rows)
    groups.update(row["source_group_id"] for row in rows)
    people.update(row["source_speaker_id"].removeprefix("p") for row in rows)
    return validate_sets(seen, groups, people)


def history(directory):
    seen, groups, people = previous.lineage.history(Path(directory))
    new_pcm, new_groups, new_people = verify_b18(directory)
    need(not (seen & new_pcm) and not (groups & new_groups) and not (people & new_people), "actual B18 full originals were independent")
    return validate_sets(seen | new_pcm, groups | new_groups, people | new_people)


def history_receipt(directory):
    seen, groups, people = history(directory)
    archives = {ident: dict(bytes=(Path(directory) / (ident + ".zip")).stat().st_size,
                  sha256=qa.sha(Path(directory) / (ident + ".zip"))) for ident in required_archives()}
    return dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, original_archives=archives,
                all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups), canonical_people=sorted(people),
                old_B17_B18_source_failures_preserved=True, all_prior_replication_sealed=True)


def validate_history(receipt):
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and receipt["old_B17_B18_source_failures_preserved"] is True and receipt["all_prior_replication_sealed"] is True
         and set(receipt["original_archives"]) == set(required_archives()), "complete old raw ancestry and no replica opening")
    known = dict(previous.lineage.read_plan()["base_original_archive_sha256"])
    known.update({str(pin["id"]): pin["sha256"] for pin in previous.lineage.proof_locks().values()})
    known.update({str(pin["id"]): pin["sha256"] for pin in proof_locks().values()})
    for ident, digest in known.items():
        need(receipt["original_archives"][ident]["sha256"] == digest
             and type(receipt["original_archives"][ident]["bytes"]) is int
             and receipt["original_archives"][ident]["bytes"] > 0, "every historical original ZIP SHA remains fixed")
    for pin in proof_locks().values():
        need(receipt["original_archives"][str(pin["id"])]["bytes"] == pin["bytes"], "original B18 ZIP length")
    values = tuple(set(receipt[key]) for key in ("all_pcm_sha256", "all_source_groups", "canonical_people"))
    need(all(len(value) == len(receipt[key]) for value, key in zip(values,
         ("all_pcm_sha256", "all_source_groups", "canonical_people"), strict=True)), "unique full ancestry inventory")
    return validate_sets(*values)


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    proof = read_plan()["original_B18_source_proof"]
    need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "fresh proof output before original downloads")
    run = json.loads(gh("actions/runs/" + str(proof["run_id"])))
    commit = json.loads(gh("commits/" + proof["run_head"]))
    zero = json.loads(gh("actions/runs?branch=research%2Fen-v6-b18-primary-scoring-once-20261124&per_page=100"))
    need(run["id"] == proof["run_id"] and run["head_sha"] == proof["run_head"]
         and run["head_branch"] == proof["run_branch"] and run["status"] == "completed"
         and run["conclusion"] == "failure" and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "actual first public B18 failure before any new source access")
    need(commit["sha"] == proof["run_head"] and len(commit["parents"]) == 1
         and commit["parents"][0]["sha"] == proof["prepared_parent"]
         and [row["filename"] for row in commit["files"]] == [".github/workflows/english-v6-b18-synthetic-primary-once.yml"],
         "unchanged original B18 preregistration code at its reviewed parent")
    need(zero == {"total_count": 0, "workflow_runs": []}, "no B18 KWS scoring exposure")
    out.mkdir()
    for role, pin in proof_locks().items():
        artifact = json.loads(gh("actions/artifacts/" + str(pin["id"])))
        need(artifact["id"] == pin["id"] and artifact["expired"] is False
             and artifact["size_in_bytes"] == pin["bytes"] and artifact["digest"] == "sha256:" + pin["sha256"]
             and artifact["workflow_run"]["id"] == pin["run_id"] and artifact["workflow_run"]["head_sha"] == pin["run_head"],
             "original B18 artifact metadata " + role)
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original B18 ZIP bytes " + role)
        (out / (str(pin["id"]) + ".zip")).write_bytes(raw)
    verify_b18(out)
    receipt = dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, source_execution=identity,
        original_B18_run={key: run[key] for key in ("id", "head_sha", "head_branch", "status", "conclusion", "run_attempt")},
        original_B18_source_proof=proof_locks(), original_B18_prepared_parent=proof["prepared_parent"],
        B18_KWS_runs=zero, old_source_failures_preserved=True, all_prior_replication_sealed=True,
        diagnostic_only=True, source_admission_eligible=False, new_KWS_forwards=0)
    (out / "AUTHORIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n")


def require_authorization(directory):
    proof = read_plan()["original_B18_source_proof"]
    receipt = json.loads((Path(directory) / "AUTHORIZATION.json").read_bytes())
    validate_identity(receipt["source_execution"])
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and receipt["original_B18_source_proof"] == proof_locks()
         and receipt["original_B18_prepared_parent"] == proof["prepared_parent"]
         and receipt["original_B18_run"] == dict(id=proof["run_id"], head_sha=proof["run_head"],
             head_branch=proof["run_branch"], status="completed", conclusion="failure", run_attempt=1)
         and receipt["B18_KWS_runs"] == {"total_count": 0, "workflow_runs": []}
         and receipt["old_source_failures_preserved"] is receipt["all_prior_replication_sealed"] is receipt["diagnostic_only"] is True
         and receipt["source_admission_eligible"] is False and receipt["new_KWS_forwards"] == 0,
         "same original B18 STOP authorizes only this fixed diagnostic")
    return receipt
