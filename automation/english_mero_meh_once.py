"""One exact-main Mero-v1 corpus run; PR and default checks are model-free."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import english_pocket12_once as utility

need, digest, read_json, write_json, within = (
    utility.need, utility.digest, utility.read_json, utility.write_json, utility.within)
PLAN = HERE / "english_mero_meh_plan.json"
REPOSITORY = "jiying2007/kws-data"
WORKFLOW = ".github/workflows/english-mero-meh-once.yml"
CAMPAIGN = "english-mero-meh-corpus-v1"
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
BUDGETS = {"initialization": 1200, "generation": 5400, "observation": 7200, "combine": 600}


def component(name):
    return importlib.import_module("english_mero_meh_" + name)


def source_preflight(plan=None):
    plan = read_json(PLAN) if plan is None else plan
    fixed = {"schema": "english-mero-meh-plan-v1", "campaign_id": CAMPAIGN,
        "repository": REPOSITORY, "workflow_path": WORKFLOW, "native_tts_request_budget": 144,
        "maximum_calls_per_observer": 144, "maximum_total_model_calls": 720,
        "providers": list(PROVIDERS), "retry_count": 0, "warmup_count": 0,
        "automatic_expansion": False, "training_source_candidate": True, "time_budget_seconds": BUDGETS}
    need(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in fixed.items()), "FIXED_NEW_CAMPAIGN_PLAN")
    need(type(plan.get("enabled")) is bool, "BOOLEAN_ENABLED")
    before = plan.get("main_before")
    need(before is None or (type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before)), "MAIN_PREDECESSOR")
    need(not plan["enabled"] or before is not None, "ENABLED_REQUIRES_PREDECESSOR")
    pins = plan.get("source_files_sha256", {})
    required = {WORKFLOW, "EXCLUSION_EVIDENCE.json", "automation/english_pocket12_once.py",
                "automation/english_pocket12_observe.py", "automation/english_mero_meh_profile_v1.json"}
    required |= {"automation/english_mero_meh_" + name + ".py" for name in
                 ("once", "generate", "observe", "recipe", "policy")}
    required.add("automation/test_english_mero_meh_once.py")
    required |= set(component("observe").native.HELPER_PINS) | set(component("generate").LEGACY_PINS)
    need(type(pins) is dict and required <= pins.keys(), "COMPLETE_FROZEN_SOURCE_SET")
    for name, expected in pins.items():
        need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected)
             and digest(within(ROOT, name)) == expected, "SOURCE_BYTES: " + name)
    component("generate").source_preflight(plan)
    component("observe").source_check(plan)
    need(len(component("recipe").rows()) == 144, "ALL_144_PREREGISTERED_ROWS")
    return {"source_check": "PASS", "enabled": plan["enabled"], "pinned_files": len(pins),
            "denominator": 144, "new_model_calls": 0, "new_network_calls": 0}


def execution_identity(plan, *, environ=None, event=None, head=None, parent=None):
    env = os.environ if environ is None else environ
    need(plan.get("enabled") is True, "DISABLED_CAMPAIGN")
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW_REF": REPOSITORY + "/" + WORKFLOW + "@refs/heads/main"}
    need(all(env.get(k) == v for k, v in expected.items()), "FIRST_PUBLIC_MAIN_PUSH_ONLY")
    run_id, commit = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[1-9][0-9]{0,19}", run_id) and re.fullmatch(r"[0-9a-f]{40}", commit), "RUN_ID_AND_COMMIT")
    before = plan.get("main_before")
    need(type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before), "BOUND_PREDECESSOR")
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPOSITORY and repo.get("private") is False
         and event.get("ref") == "refs/heads/main" and event.get("before") == before
         and event.get("after") == commit and event.get("forced") is False and event.get("deleted") is False,
         "EXACT_PUBLIC_MAIN_SUCCESSOR")
    if head is None:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], cwd=ROOT, text=True).strip()
    need(head == commit and parent == before, "CHECKOUT_AND_PARENT")
    return {"repository": REPOSITORY, "run_id": run_id, "commit": commit, "run_attempt": "1", "event": "push"}


def source_readback(source_root, plan, identity):
    try:
        value = component("generate").source_readback(source_root, plan, identity)
        need(value.get("denominator") == 144, "SOURCE_READBACK_DENOMINATOR")
        return value
    except Exception as exc:
        return {"source_ready": False, "denominator": 144, "run": identity, "successful_count": None,
                "failed_count": None, "not_started_count": None, "tts_attempts": None, "tts_returned": None,
                "status": "UNKNOWN_OR_INCOMPLETE", "missing_terminal_is_zero": False,
                "error": type(exc).__name__ + ": " + str(exc), "new_model_calls": 0}


def download_wheel(assets, runtime, deadline):
    assets = Path(assets)
    need(assets.is_absolute() and not assets.exists(), "FRESH_PRIVATE_ASSETS")
    assets.mkdir(parents=True)
    recipe = component("recipe")
    package = recipe.PACKAGE
    pins = [{"path": package["filename"], **{k: package[k] for k in ("url", "bytes", "sha256")}}]
    receipts = []
    for pin in pins:
        destination = within(assets, pin["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {**pin, "status": "STARTED", "received_bytes": 0}
        receipts.append(record)
        write_json(Path(runtime) / "DOWNLOADS.json", receipts)
        try:
            request = urllib.request.Request(pin["url"], headers={"User-Agent": "kws-mero-meh-once/1"})
            with urllib.request.urlopen(request, timeout=min(60, utility.remaining(deadline))) as response:
                with destination.open("xb") as output:
                    while block := response.read(1024 * 1024):
                        utility.remaining(deadline)
                        record["received_bytes"] += len(block)
                        need(record["received_bytes"] <= pin["bytes"], "ASSET_SIZE_BOUND")
                        output.write(block)
            need(record["received_bytes"] == pin["bytes"] and digest(destination) == pin["sha256"], "EXACT_ASSET_BYTES")
            record["status"] = "VERIFIED"
        except Exception as exc:
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            write_json(Path(runtime) / "DOWNLOADS.json", receipts)


def install_runtime(provider, venv, runtime, assets, *, source_root, plan, identity):
    need(sys.version_info[:2] == (3, 12), "PINNED_PYTHON_3_12")
    venv, runtime = Path(venv), Path(runtime)
    need(venv.is_absolute() and not venv.exists(), "FRESH_RUNTIME_NO_RETRY")
    runtime.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + BUDGETS["initialization"]
    result = {"provider": provider, "run": identity, "status": "STARTED", "retry_count": 0,
              "new_model_calls": 0, "commands": []}
    py = venv / "bin/python"
    def execute(args, label):
        record = {"step": label, "status": "STARTED"}
        result["commands"].append(record)
        write_json(runtime / "INSTALLATION.json", result)
        with (runtime / (label + ".stdout.txt")).open("xb") as out, (runtime / (label + ".stderr.txt")).open("xb") as err:
            process = subprocess.run([str(x) for x in args], env=utility.clean_environment(), stdout=out, stderr=err,
                                     timeout=utility.remaining(deadline), check=False)
        record.update(status="RETURNED", returncode=process.returncode)
        need(process.returncode == 0, "INSTALL_STEP_FAILED: " + label)
    try:
        if provider != "generation":
            check = source_readback(source_root, plan, identity)
            write_json(runtime / "SOURCE_READBACK.json", check)
            need(check["source_ready"] is True, "COMPLETE_144_SOURCE_BEFORE_INSTALL")
        execute([sys.executable, "-m", "venv", venv], "venv")
        pip = [py, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--retries", "0", "--timeout", "60"]
        cpu = ["--index-url", "https://download.pytorch.org/whl/cpu"]
        if provider == "generation":
            package_assets = runtime / "package-assets"
            download_wheel(package_assets, runtime, deadline)
            recipe = component("recipe")
            pins = recipe.RUNTIME_VERSIONS
            constraints = runtime / "CONSTRAINTS.txt"
            constraints.write_text("".join(k + "==" + v + "\n" for k, v in sorted(pins.items())))
            pip += ["--constraint", constraints]
            execute(pip + cpu + ["torch==" + pins["torch"]], "torch")
            execute(pip + [k + "==" + v for k, v in pins.items() if k not in ("torch", "kokoro")], "dependencies")
            execute(pip + [package_assets / recipe.PACKAGE["filename"]], "kokoro_wheel")
        elif provider == "parakeet":
            execute(pip + ["huggingface-hub==0.36.2", "hf-xet==1.2.0"], "observer_dependencies")
        else:
            packages = {"qwen": ["qwen-asr==0.0.6", "huggingface-hub==0.36.2"],
                        "whisper": ["openai-whisper==20250625"],
                        "phoneme": ["transformers==4.46.3", "huggingface-hub==0.36.2", "numpy==1.26.4"]}
            execute(pip + cpu + ["torch==2.9.1+cpu"] + ([] if provider == "phoneme" else ["torchaudio==2.9.1+cpu"]), "torch")
            execute(pip + packages[provider], "observer_dependencies")
        execute([py, "-I", "-m", "pip", "check"], "pip_check")
        result["status"] = "COMPLETE"
    except Exception as exc:
        result.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        write_json(runtime / "INSTALLATION.json", result)
    return result


def _missing_observations(provider, rows, error=None):
    return {"summary": {"provider": provider, "denominator": 144, "terminal_receipt_present": False,
        "actual_forward_attempts": None, "attempts_lower_bound": 0, "attempts_upper_bound": 144,
        "native_forward_returns": None, "completed_count": 0, "failed_count": 0,
        "not_started_count": None, "unknown_count": 144, "status": "INCOMPLETE_UNKNOWN_COUNTERS", "error": error},
        "rows": [{"id": row["id"], "status": "missing"} for row in rows]}


def combine(source_root, observer_roots, output, plan, identity):
    """Keep a complete packet even when source or observers did not complete."""
    source_preflight(plan)
    output, source_root = Path(output), Path(source_root)
    need(not output.exists(), "FRESH_COMBINED_OUTPUT")
    output.mkdir(parents=True)
    issues = []
    source_status = source_readback(source_root, plan, identity)
    try:
        corpus, _ = component("generate").validate_source(source_root, identity)
        utility.copy_packet(source_root, output)
        for name in ("RESULT.json", "FILES.json"):
            if (output / name).exists():
                saved = output / ("SOURCE_" + name)
                if saved.exists():
                    (output / name).unlink()
                else:
                    (output / name).rename(saved)
    except Exception as exc:
        issues.append("source: " + type(exc).__name__ + ": " + str(exc))
        utility.copy_packet(source_root, output / "partial_source")
        corpus = {"schema": "english-mero-meh-corpus-v1", "campaign_id": CAMPAIGN,
                  "run": identity, "status": "UNVERIFIED_SOURCE", "rows": component("recipe").rows()}
        write_json(output / "CORPUS_MANIFEST.json", corpus)
    rows = corpus["rows"]
    need(len(rows) == 144, "FIXED_COMBINED_DENOMINATOR")
    packets = {}
    for provider in PROVIDERS:
        destination = output / "observers" / provider
        try:
            utility.copy_packet(observer_roots[provider], destination)
            packets[provider] = component("observe").readback(provider=provider, source_root=source_root,
                observer_root=destination, plan=plan, run_identity=identity)
        except Exception as exc:
            message = provider + ": " + type(exc).__name__ + ": " + str(exc)
            issues.append(message)
            packets[provider] = _missing_observations(provider, rows, message)
    observations = {provider: {r["id"]: r for r in packet["rows"]} for provider, packet in packets.items()}
    exact = [source_status.get("tts_attempts")] + [packets[p]["summary"]["actual_forward_attempts"] for p in PROVIDERS]
    lower = [source_status.get("tts_attempts_lower_bound", exact[0] if type(exact[0]) is int else 0)]
    lower += [packets[p]["summary"]["attempts_lower_bound"] for p in PROVIDERS]
    total = sum(exact) if all(type(x) is int for x in exact) else None
    need(total is None or total <= plan["maximum_total_model_calls"], "HIGH_LEVEL_CALL_BUDGET")
    machine = {"schema": "english-mero-meh-machine-evidence-v1", "campaign_id": CAMPAIGN,
        "run": identity, "denominator": 144, "source_readback": source_status,
        "observer_summaries": {p: packets[p]["summary"] for p in PROVIDERS},
        "model_call_count_unit": "ONE_HIGH_LEVEL_SYNTHESIS_OR_RECOGNITION_REQUEST",
        "actual_total_model_calls": total, "actual_total_model_calls_lower_bound": sum(lower),
        "maximum_total_model_calls": 720, "missing_terminal_is_zero": False, "retry_count": 0,
        "rows": [{"id": row["id"], "observations": {p: {k: v for k, v in observations[p][row["id"]].items()
                  if k != "id"} for p in PROVIDERS}} for row in rows]}
    contract = component("recipe").identity_contract()
    exclusions = read_json(ROOT / "EXCLUSION_EVIDENCE.json")
    forbidden = exclusions["excluded_identities"]
    fields = {"row_ids": "id", "request_sha256": "request_sha256", "pcm_sha256": "pcm_sha256",
              "parent_pcm_sha256": "parent_pcm_sha256"}
    acoustic_exclusions = set(forbidden["pcm_sha256"]) | set(forbidden["parent_pcm_sha256"])
    overlap = {key: sorted({r[field] for r in rows if r.get(field)} &
                          (acoustic_exclusions if key in ("pcm_sha256", "parent_pcm_sha256") else set(forbidden[key])))
               for key, field in fields.items()}
    if any(overlap.values()):
        issues.append("OLD_EXCLUDED_IDENTITY_OVERLAP")
    try:
        admission = component("policy").build_admission(corpus, machine, contract)
    except Exception as exc:
        issues.append("admission: " + type(exc).__name__ + ": " + str(exc))
        admission = {"status": "STOP_INVALID_OR_INCOMPLETE_EVIDENCE", "training_allowed": False,
                     "training_snapshot_allowed": False, "denominator": 144,
                     "rows": [{"id": r["id"], "admitted": False, "status": "unknown"} for r in rows]}
    if issues or source_status["source_ready"] is not True:
        admission.update(training_allowed=False, training_snapshot_allowed=False,
                         status="STOP_INVALID_OR_INCOMPLETE_EVIDENCE")
    for name, value in (("MACHINE_EVIDENCE.json", machine), ("ADMISSION.json", admission),
                        ("SOURCE_IDENTITY_CONTRACT.json", contract), ("EXECUTION_PLAN.json", plan),
                        ("EXCLUSION_CHECK.json", overlap)):
        write_json(output / name, value)
    shutil.copyfile(ROOT / "EXCLUSION_EVIDENCE.json", output / "EXCLUSION_EVIDENCE.json")
    for name, expected in plan["source_files_sha256"].items():
        original = within(ROOT, name)
        need(digest(original) == expected, "FROZEN_COMBINED_SOURCE")
        target = within(output / "frozen_source", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    write_json(output / "frozen_source/automation/english_mero_meh_plan.json", plan)
    result = {"schema": "english-mero-meh-combined-result-v1", "stage": "combine",
        "status": "MACHINE_EVIDENCE_PACKET_RETAINED", "campaign_id": CAMPAIGN, "run": identity,
        "denominator": 144, "admission_status": admission["status"], "training_allowed": admission["training_allowed"],
        "model_call_count_unit": "ONE_HIGH_LEVEL_SYNTHESIS_OR_RECOGNITION_REQUEST",
        "actual_total_model_calls": total, "maximum_total_model_calls": 720, "issues": issues,
        "new_model_calls": 0, "new_network_calls": 0}
    write_json(output / "RESULT.json", result)
    write_json(output / "FILES.json", {"schema": "english-mero-meh-corpus-files-v1",
        "files": component("observe").file_records(output)})
    return result


def stage_status(stage, output, runtime, identity, outcome):
    output = Path(output)
    status = {"stage": stage, "run": identity, "command_step_outcome": outcome,
              "has_terminal_result": (output / "RESULT.json").is_file(), "retry_count": 0,
              "missing_terminal_is_zero": False}
    if not output.exists() or not any(output.iterdir()):
        write_json(output / "UNOBSERVED.json", status | {"actual_model_calls": None})
    write_json(Path(runtime) / "STAGE_STATUS.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source-check", action="store_true")
    actions.add_argument("--guard", action="store_true")
    actions.add_argument("--install-runtime", choices=("generation", *PROVIDERS))
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--read-source", action="store_true")
    actions.add_argument("--observe", choices=PROVIDERS)
    actions.add_argument("--combine", action="store_true")
    actions.add_argument("--stage-status", choices=("generation", *PROVIDERS))
    for name in ("output", "assets", "source", "observers", "venv", "runtime"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--outcome", choices=("success", "failure", "skipped", "cancelled"))
    args = parser.parse_args(argv)
    plan = read_json(PLAN)
    check = source_preflight(plan)
    if args.source_check:
        print(json.dumps(check, sort_keys=True))
        return 0
    identity = execution_identity(plan)
    utility.strip_credentials()
    if args.guard:
        value = identity
    elif args.install_runtime:
        value = install_runtime(args.install_runtime, args.venv, args.runtime, args.assets,
                                source_root=args.source, plan=plan, identity=identity)
    elif args.generate:
        value = component("generate").generate(plan, args.output, args.assets, identity)
    elif args.read_source:
        value = source_readback(args.source, plan, identity)
        if args.output:
            write_json(args.output, value)
    elif args.observe:
        value = component("observe").run(provider=args.observe, source_root=args.source,
            output_root=args.output, asset_root=args.assets, plan=plan, run_identity=identity)
    elif args.combine:
        roots = {p: args.observers / ("mero-meh-observer-" + p) for p in PROVIDERS}
        value = combine(args.source, roots, args.output, plan, identity)
    else:
        value = stage_status(args.stage_status, args.output, args.runtime, identity, args.outcome)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return int((args.generate or args.observe) and value.get("status") != "COMPLETE")


if __name__ == "__main__":
    sys.exit(main())
