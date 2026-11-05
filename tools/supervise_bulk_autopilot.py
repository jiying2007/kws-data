#!/usr/bin/env python3
"""Hard-deadline Linux supervisor for the existing resident Qwen bulk autopilot.

This is NOT another generator. It executes the SHA-pinned canonical controller
once in a process group, preserves its per-cell durable started receipts and
kills the whole group if a model call or model load exceeds the hard wall clock.
No invocation is retried and no training or shipping authority is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys

CONTROLLER_RELATIVE = Path("research/auto-tts-voice-clone-20261102/bulk_autopilot.py")
CONTROLLER_BLOB_SHA1 = "b77f992a1c990a5eb052b3bab491cb35605edbac"
MAX_CONTROLLER_BYTES = 128 * 1024
MAX_HARD_SECONDS = 7200
SAFE_CHILD_ENV = frozenset((
    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
    "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_controller(repository):
    path = Path(repository) / CONTROLLER_RELATIVE
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            "CONTROLLER_SYMLINK")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            0 < info.st_size <= MAX_CONTROLLER_BYTES, "CONTROLLER_REGULAR_BOUNDED")
    with path.open("rb") as stream:
        content = stream.read(MAX_CONTROLLER_BYTES + 1)
        final = os.fstat(stream.fileno())
    require(len(content) == info.st_size and
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) ==
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns),
            "CONTROLLER_CHANGED_DURING_READ")
    digest = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") +
                          b"\0" + content).hexdigest()
    require(digest == CONTROLLER_BLOB_SHA1, "CONTROLLER_SOURCE_DRIFT")
    return path


def child_environment(parent):
    # No token, cloud credential or network proxy is needed for pinned offline
    # local inference. The model subprocess must not inherit arbitrary secrets.
    allowed = {key: value for key, value in parent.items() if key in SAFE_CHILD_ENV}
    allowed.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                   PYTHONDONTWRITEBYTECODE="1")
    return allowed


def terminate_group(process, grace_seconds):
    # start_new_session=True makes the child its own process-group leader.
    # Terminate descendants too; a hanging/grandchild model must not survive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(command, *, hard_seconds, grace_seconds, env):
    require(sys.platform.startswith("linux"), "LINUX_PROCESS_GROUP_REQUIRED")
    require(type(hard_seconds) is int and 1 <= hard_seconds <= MAX_HARD_SECONDS,
            "HARD_TIMEOUT_BUDGET")
    require(type(grace_seconds) is int and 1 <= grace_seconds <= 15,
            "TERMINATION_GRACE_BUDGET")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                               start_new_session=True, close_fds=True,
                               env=child_environment(env))
    try:
        code = process.wait(timeout=hard_seconds)
    except subprocess.TimeoutExpired:
        terminate_group(process, grace_seconds)
        return {"status": "HARD_TIMEOUT_HOLD_NO_REPLAY", "child_exit_code": process.returncode,
                "hard_seconds": hard_seconds, "training_admitted": False,
                "qualification_allowed": False}
    except KeyboardInterrupt:
        terminate_group(process, grace_seconds)
        return {"status": "OPERATOR_INTERRUPT_HOLD_NO_REPLAY",
                "child_exit_code": process.returncode, "hard_seconds": hard_seconds,
                "training_admitted": False, "qualification_allowed": False}
    return {"status": "CHILD_SUCCESS" if code == 0 else "CHILD_FAILED_NO_AUTO_RETRY",
            "child_exit_code": code, "hard_seconds": hard_seconds,
            "training_admitted": False, "qualification_allowed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sources", "audio-root", "workspace", "adapter-lock",
                 "pipeline-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("model-folder", "model-lock"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--max-tts", type=int, required=True)
    parser.add_argument("--max-asr", type=int, required=True)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--hard-seconds", type=int, default=600)
    parser.add_argument("--kill-grace-seconds", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN_EXECUTE_REQUIRED", "model_calls": 0,
                          "training_admitted": False}, sort_keys=True))
        return 0
    if args.model_folder is None or args.model_lock is None:
        print(json.dumps({"status": "NOT_RUN_MODEL_ASSETS_MISSING", "model_calls": 0,
                          "training_admitted": False}, sort_keys=True))
        return 0
    require(os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "OFFLINE_EXECUTION_REQUIRED")
    require(type(args.max_tts) is int and 0 <= args.max_tts <= 16384 and
            type(args.max_asr) is int and 0 <= args.max_asr <= 49152,
            "CALL_BUDGET")
    require(math.isfinite(args.wall_seconds) and 0 < args.wall_seconds <= 3600,
            "LAUNCH_WINDOW")
    require(1 <= args.hard_seconds <= MAX_HARD_SECONDS and
            1 <= args.kill_grace_seconds <= 15, "HARD_BOUND")
    repository = Path(__file__).resolve().parents[1]
    work = args.workspace.resolve()
    require(work != repository and repository not in work.parents,
            "PRIVATE_WORKSPACE_OUTSIDE_REPO_REQUIRED")
    controller = checked_controller(repository)
    command = [sys.executable, "-B", str(controller)]
    for name in ("plan", "sources", "audio_root", "workspace", "adapter_lock",
                 "pipeline_root"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for name in ("model_folder", "model_lock"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    command += ["--max-tts", str(args.max_tts), "--max-asr", str(args.max_asr),
                "--wall-seconds", str(args.wall_seconds), "--execute"]
    status = supervise(command, hard_seconds=args.hard_seconds,
                       grace_seconds=args.kill_grace_seconds, env=os.environ)
    print(json.dumps(status, sort_keys=True))
    if status["status"] == "HARD_TIMEOUT_HOLD_NO_REPLAY":
        return 124
    if status["status"] == "OPERATOR_INTERRUPT_HOLD_NO_REPLAY":
        return 130
    code = status["child_exit_code"]
    return code if code is not None and 0 <= code <= 125 else 125


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("BULK_SUPERVISOR_HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(2)
