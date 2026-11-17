"""One shared CPU runtime installation and import-only check for PR and production.

No model is constructed. No model asset is downloaded. The package source is
imported with Hugging Face offline flags; production inference is a later step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_kokoro48_recipe as recipe

SCHEMA = "english-kokoro48-runtime-import-v1"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def allowed_event(env, event):
    need(env.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and event.get("repository", {}).get("private") is False,
         "PUBLIC_OWN_REPOSITORY_RUNTIME_ONLY")
    if env.get("GITHUB_EVENT_NAME") == "pull_request":
        need(event.get("pull_request", {}).get("head", {}).get("repo", {}).get("full_name") == "jiying2007/kws-data",
             "SAME_REPOSITORY_PR_ONLY")
    else:
        need(env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
             and env.get("GITHUB_RUN_ATTEMPT") == "1", "FIRST_MAIN_PRODUCTION_RUNTIME_ONLY")


def install_arguments(venv):
    pip = str(Path(venv) / "bin/pip")
    cpu = [pip, "install", "--disable-pip-version-check", "--index-url", "https://download.pytorch.org/whl/cpu",
           "torch==" + recipe.RUNTIME_VERSIONS["torch"]]
    packages = [recipe.PACKAGE["url"] + "#sha256=" + recipe.PACKAGE["sha256"]]
    for name, version in recipe.RUNTIME_VERSIONS.items():
        if name not in ("torch", "kokoro"):
            packages.append(("misaki[en]" if name == "misaki" else name) + "==" + version)
    return [cpu, [pip, "install", "--disable-pip-version-check", *packages], [pip, "check"]]


IMPORT_CODE = '''
import importlib.metadata, json, os, sys
expected = json.loads(sys.argv[1])
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected or sys.version_info[:2] != (3, 12):
    raise ValueError('EXACT_RUNTIME_VERSIONS')
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    if os.environ.get(key) != '1':
        raise ValueError('OFFLINE_IMPORT_REQUIRED')
from kokoro import KModel
if KModel.__module__ != 'kokoro.model':
    raise ValueError('NATIVE_KMODEL_IMPORT')
print(json.dumps({'status':'IMPORT_ONLY_PASSED','required_versions':expected,
 'actual_versions':actual,'python':sys.version,'KModel_module':KModel.__module__,
 'model_instances':0,'model_asset_downloads':0,'model_forward_calls':0,
 'offline_flags':{k:os.environ[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE')}}))
'''


def run(venv, output):
    env = dict(os.environ)
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    allowed_event(env, event)
    venv, output = Path(venv).absolute(), Path(output).absolute()
    need(venv != output and not venv.exists() and not output.exists()
         and venv.parent.is_dir() and output.parent.is_dir()
         and not any(p.is_symlink() for p in (venv, output, *venv.parents, *output.parents)), "NEW_RUNTIME_AND_RECEIPT_DIRECTORIES")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    deadline = started + 780
    commands, failure = [], None
    base = {"schema": SCHEMA, "github_sha": env.get("GITHUB_SHA"), "run_id": env.get("GITHUB_RUN_ID"),
        "event": env.get("GITHUB_EVENT_NAME"), "runtime_versions": recipe.RUNTIME_VERSIONS,
        "model_instances": 0, "model_asset_downloads": 0, "model_forward_calls": 0}
    def save(name, value):
        with (output / name).open("x") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    def command(argv, label, execution_env=None):
        remaining = deadline - time.monotonic()
        need(remaining > 0, "RUNTIME_INSTALL_DEADLINE")
        with (output / (label + ".log")).open("xb") as stream:
            result = subprocess.run(argv, env=execution_env or env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=min(600, remaining), check=False)
        commands.append({"phase": label, "returncode": result.returncode})
        need(result.returncode == 0, "RUNTIME_PHASE_FAILED: " + label)
    save("STARTED.json", base | {"status": "STARTED_IMPORT_ONLY", "deadline_seconds": 780})
    try:
        command([sys.executable, "-m", "venv", str(venv)], "create-venv")
        for argv, label in zip(install_arguments(venv), ("install-cpu-torch", "install-pinned-packages", "pip-check")):
            command(argv, label)
        offline = env | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1"}
        command([str(venv / "bin/python"), "-I", "-B", "-c", IMPORT_CODE,
            json.dumps(recipe.RUNTIME_VERSIONS, sort_keys=True)], "import-only", offline)
        lines = (output / "import-only.log").read_text().splitlines()
        documents = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("status") == "IMPORT_ONLY_PASSED":
                documents.append(value)
        need(len(documents) == 1 and documents[0]["actual_versions"] == recipe.RUNTIME_VERSIONS, "ONE_REAL_IMPORT_RESULT")
        save("IMPORT_RESULT.json", documents[0])
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:300]}
    finally:
        pip = venv / "bin/pip"
        freeze_failure = None
        if pip.is_file():
            try:
                with (output / "pip-freeze.txt").open("xb") as stream:
                    freeze = subprocess.run([str(pip), "freeze"], stdout=stream, stderr=subprocess.STDOUT,
                        check=False, timeout=30, env=env)
                    if freeze.returncode != 0:
                        freeze_failure = {"type": "NonzeroExit", "returncode": freeze.returncode}
            except BaseException as exc:
                freeze_failure = {"type": type(exc).__name__, "message": str(exc)[:300]}
        result = base | {"status": "IMPORT_ONLY_PASSED" if failure is None else "IMPORT_ONLY_FAILED",
            "commands": commands, "failure": failure, "pip_freeze_failure": freeze_failure,
            "elapsed_seconds": time.monotonic() - started}
        save("RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.venv, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "IMPORT_ONLY_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
