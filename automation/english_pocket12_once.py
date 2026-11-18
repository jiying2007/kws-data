"""One guarded Pocket12 probe; default and PR entry points are source-only.

No training, dispatch, retry, warmup, credential export or automatic expansion.
The two activation fields are enabled and main_before; all other inputs freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = HERE / "english_pocket12_plan.json"
REPOSITORY = "jiying2007/kws-data"
WORKFLOW = ".github/workflows/english-pocket12-once.yml"
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
BUDGETS = {"initialization": 840, "generation": 1140, "observation": 2040, "combine": 480}
MODULES = {"source": "english_pocket12_generate", "observer": "english_pocket12_observe",
           "evaluate": "english_pocket12_evaluate"}
_COMPONENTS = {}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024**2,
         "BOUNDED_REGULAR_JSON")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=lambda _: need(False, "FINITE_JSON"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def within(root, name):
    need(type(name) is str and "\\" not in name, "SAFE_RELATIVE_PATH")
    rel = PurePosixPath(name)
    need(rel.parts and not rel.is_absolute() and rel.as_posix() == name
         and all(x not in (".", "..") for x in rel.parts), "SAFE_RELATIVE_PATH")
    path = Path(root).absolute().joinpath(*rel.parts)
    need(not any(x.is_symlink() for x in (path, *path.parents)), "NO_SYMLINK")
    return path


def component(kind):
    if kind in _COMPONENTS:
        return _COMPONENTS[kind]
    for directory in (ROOT, HERE):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    name = MODULES[kind]
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _COMPONENTS[kind] = module
    return module


def pinned_files(plan):
    pins = {"automation/english_pocket12_once.py": plan["worker_sha256"],
            "automation/english_pocket12_generate.py": plan["source_code_sha256"],
            "automation/english_pocket12_observe.py": plan["observer_code_sha256"],
            "automation/english_pocket12_evaluate.py": plan["evaluate_code_sha256"],
            "automation/english_pocket12_recipe.json": plan["recipe_sha256"],
            WORKFLOW: plan["workflow_sha256"]}
    for group in ("helper_sha256", "test_sha256"):
        need(type(plan[group]) is dict and plan[group], "NONEMPTY_SOURCE_PINS")
        for path, sha in plan[group].items():
            need(path not in pins, "UNIQUE_SOURCE_PIN")
            pins[path] = sha
    return pins


def source_preflight(plan=None):
    plan = read_json(PLAN) if plan is None else plan
    fixed = {"schema": "english-pocket12-plan-v1", "campaign_id": "pocket12-native-short-v1",
             "repository": REPOSITORY, "workflow_path": WORKFLOW,
             "diagnostic_only": True, "training_allowed": False,
             "native_tts_request_budget": 12, "maximum_calls_per_observer": 12,
             "maximum_total_model_calls": 60, "retry_count": 0, "warmup_count": 0,
             "automatic_expansion": False, "providers": list(PROVIDERS),
             "time_budget_seconds": BUDGETS}
    for key, value in fixed.items():
        need(type(plan.get(key)) is type(value) and plan[key] == value, "FIXED_PLAN: " + key)
    need(type(plan.get("enabled")) is bool, "BOOLEAN_ENABLED")
    before = plan.get("main_before")
    need(before is None or (type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before)),
         "MAIN_PREDECESSOR_FORMAT")
    need(not plan["enabled"] or before is not None, "ENABLED_REQUIRES_PREDECESSOR")
    pins = pinned_files(plan)
    for name, sha in pins.items():
        path = within(ROOT, name)
        need(type(sha) is str and re.fullmatch(r"[0-9a-f]{64}", sha)
             and path.is_file() and digest(path) == sha, "SOURCE_BYTES: " + name)
    component("source").source_preflight(plan)
    component("observer").source_check(plan)
    component("evaluate").source_check(plan)
    return {"source_check": "PASS", "pinned_files": len(pins), "enabled": plan["enabled"],
            "new_model_calls": 0, "new_network_calls": 0}


def execution_identity(plan, *, environ=None, event=None, head=None, parent=None):
    env = os.environ if environ is None else environ
    need(plan.get("enabled") is True, "DISABLED_PROBE")
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPOSITORY,
                "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_WORKFLOW_REF": REPOSITORY + "/" + WORKFLOW + "@refs/heads/main"}
    need(all(env.get(k) == v for k, v in expected.items()), "PUBLIC_FIRST_MAIN_WORKFLOW_ONLY")
    run_id, commit = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[1-9][0-9]{0,19}", run_id) and re.fullmatch(r"[0-9a-f]{40}", commit),
         "RUN_ID_AND_COMMIT")
    before = plan.get("main_before")
    need(type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before), "BOUND_PREDECESSOR")
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPOSITORY and repo.get("private") is False
         and event.get("ref") == "refs/heads/main" and event.get("before") == before
         and event.get("after") == commit and event.get("deleted") is False
         and event.get("forced") is False, "EXACT_PUBLIC_MAIN_PUSH")
    if head is None:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], cwd=ROOT, text=True).strip()
    need(head == commit and parent == before, "CHECKOUT_AND_FIRST_PARENT")
    return {"repository": REPOSITORY, "run_id": run_id, "commit": commit,
            "run_attempt": "1", "event": "push"}


def clean_environment():
    """Install/model subprocesses receive no inherited auth or package index variables."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "RUNNER_TEMP", "SYSTEMROOT",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    return env


def strip_credentials():
    for name in tuple(os.environ):
        if (re.search(r"(^|_)(TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|ACCESS_KEY|PRIVATE_KEY)(_|$)", name)
                and name != "TOKENIZERS_PARALLELISM"):
            del os.environ[name]
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def remaining(deadline):
    value = deadline - time.monotonic()
    need(value > 0, "FIXED_TIME_BUDGET_EXHAUSTED")
    return value


def download_assets(assets, runtime, deadline):
    assets = Path(assets)
    need(assets.is_absolute() and not assets.exists(), "FRESH_PRIVATE_ASSET_DIRECTORY")
    assets.mkdir(parents=True)
    receipts = []
    for pin in component("source").recipe()["assets"]:
        destination = within(assets, pin["path"])
        record = {**pin, "status": "STARTED", "received_bytes": 0}
        receipts.append(record)
        write_json(Path(runtime) / "DOWNLOADS.json", receipts)
        try:
            request = urllib.request.Request(pin["url"], headers={"User-Agent": "kws-pocket12-once/1"})
            with urllib.request.urlopen(request, timeout=min(60, remaining(deadline))) as response:
                with destination.open("xb") as output:
                    while block := response.read(1024 * 1024):
                        remaining(deadline)
                        record["received_bytes"] += len(block)
                        need(record["received_bytes"] <= pin["bytes"], "EXACT_ASSET_SIZE_BOUND")
                        output.write(block)
            need(record["received_bytes"] == pin["bytes"] and digest(destination) == pin["sha256"],
                 "EXACT_ASSET_BYTES: " + pin["path"])
            record["status"] = "VERIFIED"
        except Exception as exc:
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            write_json(Path(runtime) / "DOWNLOADS.json", receipts)


def source_readback(source_root, plan, identity):
    try:
        value = component("source").source_readback(source_root, plan, identity)
        value.update(denominator=12, run=identity)
        return value
    except Exception as exc:
        return {"source_ready": False, "denominator": 12, "run": identity,
                "successful_count": None, "failed_count": None, "not_started_count": None,
                "tts_attempts": None, "tts_returned": None, "new_model_calls": 0,
                "error": type(exc).__name__ + ": " + str(exc),
                "status": "UNKNOWN_OR_INCOMPLETE", "missing_terminal_is_zero": False}


def install_runtime(provider, venv, runtime, assets, *, source_root, plan, identity):
    need(sys.version_info[:2] == (3, 12), "FIXED_PYTHON_3_12")
    venv, runtime = Path(venv), Path(runtime)
    need(venv.is_absolute() and not venv.exists(), "FRESH_RUNTIME_NO_RETRY")
    runtime.mkdir(parents=True, exist_ok=True)
    deadline, commands = time.monotonic() + BUDGETS["initialization"], []
    result = {"provider": provider, "run": identity, "status": "STARTED", "retry_count": 0,
              "new_model_calls": 0, "commands": commands}
    py = venv / "bin/python"

    def execute(args, label):
        commands.append({"step": label, "status": "STARTED"})
        write_json(runtime / "INSTALLATION.json", result)
        with (runtime / (label + ".stdout.txt")).open("xb") as out:
            with (runtime / (label + ".stderr.txt")).open("xb") as err:
                proc = subprocess.run([str(x) for x in args], env=clean_environment(),
                    stdout=out, stderr=err, timeout=remaining(deadline), check=False)
        commands[-1].update(status="RETURNED", returncode=proc.returncode)
        need(proc.returncode == 0, "INSTALL_STEP_FAILED: " + label)

    try:
        if provider != "generation":
            readback = source_readback(source_root, plan, identity)
            write_json(runtime / "SOURCE_READBACK.json", readback)
            need(readback["source_ready"] is True, "TWELVE_VERIFIED_SOURCES_BEFORE_INSTALL")
        execute([sys.executable, "-m", "venv", venv], "venv")
        pip = [py, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
               "--retries", "0", "--timeout", "60"]
        cpu = ["--index-url", "https://download.pytorch.org/whl/cpu"]
        if provider == "generation":
            download_assets(assets, runtime, deadline)
            pins = component("source").recipe()["runtime_versions"]
            constraints = runtime / "CONSTRAINTS.txt"
            constraints.write_text("".join(k + "==" + v + "\n" for k, v in sorted(pins.items())))
            pip += ["--constraint", constraints]
            execute(pip + cpu + ["torch==2.9.1+cpu"], "torch")
            execute(pip + [k + "==" + v for k, v in pins.items() if k not in ("torch", "pocket-tts")],
                    "numerical_dependencies")
            wheel = Path(assets) / "pocket_tts-4.0.1-py3-none-any.whl"
            execute(pip + [wheel], "pocket_wheel")
        elif provider == "parakeet":
            execute(pip + ["huggingface-hub==0.36.2", "hf-xet==1.2.0"], "observer_dependencies")
        else:
            torch = ["torch==2.9.1+cpu"] + ([] if provider == "phoneme" else ["torchaudio==2.9.1+cpu"])
            execute(pip + cpu + torch, "torch")
            packages = {"qwen": ["qwen-asr==0.0.6", "huggingface-hub==0.36.2"],
                        "whisper": ["openai-whisper==20250625"],
                        "phoneme": ["transformers==4.46.3", "huggingface-hub==0.36.2", "numpy==1.26.4"]}
            execute(pip + packages[provider], "observer_dependencies")
        execute([py, "-I", "-m", "pip", "check"], "pip_check")
        result["status"] = "COMPLETE"
    except Exception as exc:
        result.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if py.is_file():
            with (runtime / "PIP_FREEZE.txt").open("wb") as stream:
                try:
                    subprocess.run([str(py), "-I", "-m", "pip", "freeze"], stdout=stream,
                                   stderr=subprocess.STDOUT, env=clean_environment(), timeout=10, check=False)
                except Exception:
                    pass
        write_json(runtime / "INSTALLATION.json", result)
    return result


def copy_packet(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return False
    need(source.is_dir() and not source.is_symlink(), "REGULAR_PACKET_DIRECTORY")
    total = 0
    files = []
    for path in sorted(source.rglob("*")):
        need(not path.is_symlink(), "NO_PACKET_SYMLINK")
        if path.is_file():
            size = path.stat().st_size
            total += size
            need(size <= 64 * 1024**2 and total <= 512 * 1024**2, "PACKET_BYTE_BOUND")
            files.append(path)
    for path in files:
        target = within(destination, path.relative_to(source).as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return True


def combine(source_root, observer_roots, output, plan, identity):
    output = Path(output)
    need(not output.exists(), "FRESH_COMBINED_OUTPUT")
    output.mkdir(parents=True)
    write_json(output / "RUN.json", identity)
    write_json(output / "PLAN.json", plan)
    failure = None
    try:
        for name, sha in pinned_files(plan).items():
            original = within(ROOT, name)
            need(digest(original) == sha, "COMBINED_FROZEN_SOURCE")
            target = within(output / "frozen_source", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        copy_packet(source_root, output / "source")
        copied = {}
        for provider in PROVIDERS:
            copied[provider] = output / "observers" / provider
            copy_packet(observer_roots[provider], copied[provider])
        evaluation = component("evaluate").evaluate(output / "source", copied, plan, identity)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
        evaluation = {"schema": "english-pocket12-evaluation-failure-v1", "run": identity,
            "status": "INCOMPLETE", "diagnostic_only": True, "training_allowed": False,
            "denominator": 12, "maximum_total_model_calls": 60, "actual_total_model_calls": None,
            "missing_terminal_is_zero": False, "error": failure}
    write_json(output / "EVALUATION.json", evaluation)
    result = {"schema": "english-pocket12-combined-result-v1", "run": identity,
              "diagnostic_only": True, "training_allowed": False, "retry_count": 0,
              "evaluation_status": evaluation.get("status"), "combination_error": failure,
              "new_model_calls": 0, "new_network_calls": 0}
    write_json(output / "RESULT.json", result)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "FILES.json", {"files": files, "self_excluded": "FILES.json"})
    return result


def stage_status(stage, output, runtime, identity, outcome):
    output = Path(output)
    status = {"stage": stage, "run": identity, "command_step_outcome": outcome,
              "has_terminal_result": (output / "RESULT.json").is_file(),
              "diagnostic_only": True, "training_allowed": False, "retry_count": 0,
              "missing_terminal_is_zero": False}
    if not output.exists() or not any(output.iterdir()):
        write_json(output / "UNOBSERVED.json", {**status, "actual_model_calls": None})
    write_json(Path(runtime) / "STAGE_STATUS.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source-check", action="store_true")
    action.add_argument("--guard", action="store_true")
    action.add_argument("--install-runtime", choices=("generation", *PROVIDERS))
    action.add_argument("--generate", action="store_true")
    action.add_argument("--read-source", action="store_true")
    action.add_argument("--observe", choices=PROVIDERS)
    action.add_argument("--combine", action="store_true")
    action.add_argument("--stage-status", choices=("generation", *PROVIDERS))
    for name in ("output", "assets", "source", "observers", "venv", "runtime"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--outcome", choices=("success", "failure", "skipped", "cancelled"))
    args = parser.parse_args(argv)
    need(args.plan.resolve() == PLAN, "CHECKED_IN_PLAN_ONLY")
    plan = read_json(PLAN)
    check = source_preflight(plan)
    if args.source_check:
        print(json.dumps(check, sort_keys=True))
        return 0
    identity = execution_identity(plan)
    strip_credentials()
    if args.guard:
        value = identity
    elif args.install_runtime:
        value = install_runtime(args.install_runtime, args.venv, args.runtime, args.assets,
                                source_root=args.source, plan=plan, identity=identity)
    elif args.generate:
        value = component("source").generate(plan, args.output, args.assets, identity)
    elif args.read_source:
        value = source_readback(args.source, plan, identity)
        write_json(args.output, value)
    elif args.observe:
        need(source_readback(args.source, plan, identity)["source_ready"] is True,
             "TWELVE_VERIFIED_SOURCES_BEFORE_OBSERVATION")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        value = component("observer").run(provider=args.observe, source_root=args.source,
            output_root=args.output, asset_root=args.assets, plan=plan, run_identity=identity)
    elif args.combine:
        roots = {p: args.observers / ("pocket12-observer-" + p) for p in PROVIDERS}
        value = combine(args.source, roots, args.output, plan, identity)
    else:
        value = stage_status(args.stage_status, args.output, args.runtime, identity, args.outcome)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))
    if args.generate or args.observe:
        return 0 if value.get("status") == "COMPLETE" else 1
    if args.combine:
        return 0 if not value["combination_error"] and value["evaluation_status"] == "PASS_POCKET12_SOURCE_FEASIBILITY" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
