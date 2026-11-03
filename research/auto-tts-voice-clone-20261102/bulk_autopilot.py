"""Bounded local bulk TTS -> native ASR -> independent pipeline readback.

Only explicit callers provide a backend/executor. Saved starts are consumed,
including failed/unknown calls. No training, model promotion or network setup.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import stat
from pathlib import Path
import sys
import subprocess
import time

import bulk_pool as p
import bulk_runner
import bulk_asr
import campaign
import receipt_contract as receipts
import runner

PIPELINE_COMMIT = "8b7d62b355b714f27aae177f676fc80aedbf6cf4"
PIPELINE_BLOBS = {
    "corpus_identity": "5744097a6c244197410fbd75112d65e8180fdb57",
    "kws_data_bulk_effect_boundary": "26652a5f3d0aaf58d6be809ad8b954af14204ad9",
    "prioritize_kws_data_feedback": "eae711270a9ac5ee59589d0ef17e535891c759eb",
}
DATA_COMMIT = "143eb4e8c0c3baeb41ebbfac36774968c7ba8f6d"
SCHEMA = "bounded-bulk-autopilot-v1"

def digest(value):
    return hashlib.sha256(p.canonical(value)).hexdigest()

def regular(path, limit=16 * 1024 * 1024):
    runner._safe_file(path, limit=limit)
    return Path(path).read_bytes()

def once_or_equal(path, value):
    if path.exists():
        p.require(p.canonical(p.read_json(path)) == p.canonical(value), "CONTROLLER_IDENTITY_DRIFT")
    else:
        runner._once(path, value)

@contextmanager
def exclusive(root):
    """Kernel-held lock, released on process exit; it does not authorize replay."""
    path = root / "controller.lock"
    p.require(not path.is_symlink(), "CONTROLLER_LOCK_ALIAS")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        p.require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, "CONTROLLER_LOCK_LINK")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("CONTROLLER_ALREADY_RUNNING") from exc
        yield
    finally:
        os.close(fd)

class PipelineConsumer:
    """Execute only the three fixed, reviewed local consumer source blobs."""
    identity = {"commit": PIPELINE_COMMIT, "blobs": PIPELINE_BLOBS}

    def __init__(self, root):
        self.root = Path(root)
        self.verify()

    def verify(self):
        for name, expected in PIPELINE_BLOBS.items():
            raw = regular(self.root / "tools" / (name + ".py"))
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            p.require(actual == expected, "PIPELINE_CONSUMER_SOURCE_DRIFT")

    def __call__(self, plan, screen, generation):
        self.verify()
        saved = {name: sys.modules.get(name) for name in PIPELINE_BLOBS}
        old_path = sys.path[:]
        loaded = {}
        try:
            for name in PIPELINE_BLOBS:
                spec = importlib.util.spec_from_file_location(
                    name, self.root / "tools" / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                loaded[name] = module
            effect = loaded["kws_data_bulk_effect_boundary"].measure(plan, screen, generation)
            feedback = loaded["prioritize_kws_data_feedback"].derive(effect)
        finally:
            sys.path[:] = old_path
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.verify()
        return effect, feedback

def counts(plan, generation, asr, binding):
    tts = asr_count = unknown = failed = 0
    ids = {cell['cell_id'] for cell in plan['cells']}
    for base, engines in ((generation / 'cells', False), (asr / 'clips', True)):
        if base.exists():
            p.require(not base.is_symlink(), 'RECEIPT_DIRECTORY_ALIAS')
            for folder in base.iterdir():
                p.require(folder.name in ids and folder.is_dir() and not folder.is_symlink(),
                          'FOREIGN_ATTEMPT_DIRECTORY')
                if engines:
                    p.require(all(x.name in campaign.ENGINES and x.is_dir() and not x.is_symlink()
                                  for x in folder.iterdir()), 'FOREIGN_ASR_ENGINE_DIRECTORY')
    for cell in plan["cells"]:
        folders = [generation / "cells" / cell["cell_id"]]
        folders += [asr / "clips" / cell["cell_id"] / e for e in campaign.ENGINES]
        for index, folder in enumerate(folders):
            start = folder / "started.json"
            if not start.exists():
                p.require(not folder.exists(), "ORPHAN_ATTEMPT_DIRECTORY")
                continue
            regular(start, 65536)
            record = p.read_json(start, 65536)
            p.require(record.get('cell_id') == cell['cell_id'] and
                      record.get('plan_sha256') == binding['plan_sha256'] and
                      type(record.get('attempts')) is int and record['attempts'] == 1 and
                      record.get('training_admitted') is False, 'START_RECEIPT_IDENTITY')
            if index:
                p.require(record.get('schema') == 'kws-bulk-asr-start-v1' and
                          record.get('engine') == campaign.ENGINES[index - 1] and
                          record.get('adapter_lock_sha256') == binding['adapter_lock_sha256'],
                          'ASR_START_RECEIPT_IDENTITY')
            else:
                p.require(record.get('schema') == 'kws-bulk-clone-start-v1',
                          'TTS_START_RECEIPT_IDENTITY')
            if index == 0:
                tts += 1
            else:
                asr_count += 1
            if not (folder / "completed.json").exists():
                if (folder / "failed.json").exists():
                    failed += 1
                else:
                    unknown += 1
    return {"tts_started": tts, "asr_started": asr_count,
            "consumed_unknown": unknown, "consumed_failed": failed}

def execute(plan, sources, audio_root, workspace, adapters, *,
            max_tts, max_asr, wall_seconds, consumer, backend_identity,
            backend=None, executor=None, writer=runner._write_waves,
            clock=time.monotonic, stop_on_confusion=True):
    """Resume a frozen campaign. Wall time is a launch cutoff, not preemption.

    The caller must bound each supplied backend/executor call. This controller
    never starts another call after its deadline and always reserves via the
    existing durable started receipt before invocation.
    """
    p.verify_plan(plan)
    expected = p.plan(sources, audio_root, repetitions=plan["repetitions"],
        batch_id=plan["batch_id"], revision=plan["model_revision"],
        max_calls=plan["budgets"]["max_total_calls"])
    p.require(p.canonical(expected) == p.canonical(plan), "SOURCE_OR_PLAN_CHANGED")
    binding = receipts.binding(plan, adapters)
    p.require(type(backend_identity) is dict and set(backend_identity) ==
              {'model_id', 'model_revision', 'asset_lock_sha256', 'provider_kind'} and
              backend_identity['model_id'] == plan['model_id'] and
              backend_identity['model_revision'] == plan['model_revision'] and
              type(backend_identity['asset_lock_sha256']) is str and
              p._SHA.fullmatch(backend_identity['asset_lock_sha256']) and
              backend_identity['provider_kind'] in ('qwen-offline', 'synthetic-test-double'),
              'BACKEND_IDENTITY_REQUIRED')
    p.require(type(max_tts) is int and 0 <= max_tts <= plan["budgets"]["max_total_calls"]
              and type(max_asr) is int and 0 <= max_asr <= 3 * max_tts, "GLOBAL_CALL_BUDGET")
    p.require(type(wall_seconds) in (int, float) and math.isfinite(wall_seconds)
              and 0 < wall_seconds <= 3600, "WALL_BUDGET")
    p.require(type(stop_on_confusion) is bool, "STOP_POLICY")
    p.require(isinstance(consumer, PipelineConsumer), "PINNED_PIPELINE_CONSUMER_REQUIRED")
    consumer.verify()
    root = Path(workspace)
    p.require(not any(x.is_symlink() for x in (root, *root.parents)), "WORKSPACE_ALIAS")
    generation, asr = root / "tts", root / "asr"
    config = {"schema": SCHEMA, **binding, "sources_sha256": digest(sources),
              "audio_root": str(Path(audio_root).resolve()),
              "max_tts": max_tts, "max_asr": max_asr,
              "wall_seconds": wall_seconds, "stop_on_confusion": stop_on_confusion,
              "data_commit": DATA_COMMIT, "consumer": consumer.identity,
              "backend_identity": backend_identity,
              "training_admitted": False}
    # Validate existing identities before constructing or calling any model.
    if root.exists():
        p.require(root.is_dir(), "WORKSPACE_NOT_DIRECTORY")
    else:
        root.mkdir(parents=True)
    with exclusive(root):
        once_or_equal(root / "controller-config.json", config)
        receipts.verify_root(asr, binding, "kws-bulk-asr-lock-v1")
        if generation.exists():
            lock = p.read_json(generation / "plan-lock.json")
            p.require(lock.get("plan_sha256") == binding["plan_sha256"], "GENERATION_PLAN_DRIFT")
        before = counts(plan, generation, asr, binding)
        p.require(before["tts_started"] <= max_tts and before["asr_started"] <= max_asr,
                  "GLOBAL_BUDGET_ALREADY_EXCEEDED")
        started_at = clock()
        deadline = started_at + wall_seconds
        status = "NO_MORE_ELIGIBLE_CALLS"
        latest = None
        preflight_hold = False
        if generation.exists():
            prior_screen = bulk_asr.screen_results(plan, generation, asr, adapters)
            prior_effect, prior_feedback = consumer(plan, prior_screen, generation)
            latest = {'schema': SCHEMA, **binding, 'screen': prior_screen,
                      'effect': prior_effect, 'feedback': prior_feedback, 'counts': before,
                      'training_admitted': False, 'qualification_allowed': False}
            once_or_equal(root / 'checkpoints' / (digest(latest) + '.json'), latest)
            preflight_hold = stop_on_confusion and bool(prior_screen['counts'].get(
                'NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT', 0))
            if preflight_hold:
                status = 'CONFUSION_HOLD'
        while not preflight_hold:
            current = counts(plan, generation, asr, binding)
            p.require(current["tts_started"] <= max_tts and current["asr_started"] <= max_asr,
                      "GLOBAL_BUDGET_ALREADY_EXCEEDED")
            if clock() >= deadline:
                status = "DEADLINE_HOLD"
                break
            progress = False
            if current["tts_started"] < max_tts:
                if backend is None:
                    status = "NOT_RUN_BACKEND_MISSING"
                    break
                bulk_runner.run(plan, sources, audio_root, generation, backend,
                                max_new_calls=1, writer=writer)
                progress = True
            if clock() >= deadline:
                status = "DEADLINE_HOLD"
                break
            current = counts(plan, generation, asr, binding)
            # A single new ASR request per dispatch makes deadline and budget
            # observation independent of the number of remaining cells.
            if current["asr_started"] < max_asr:
                if executor is None:
                    status = "NOT_RUN_ASR_EXECUTOR_MISSING"
                    break
                result = bulk_asr.dispatch(plan, generation, asr, adapters,
                    max_new_calls=1, executor=executor, execute=True)
                progress = progress or result["actual_calls"] > 0
            if generation.exists():
                screen = bulk_asr.screen_results(plan, generation, asr, adapters)
                effect, feedback = consumer(plan, screen, generation)
                # A content-addressed immutable checkpoint can be reused on
                # resume, but never overwritten with a different partial state.
                latest = {"schema": SCHEMA, **binding, "screen": screen,
                          "effect": effect, "feedback": feedback,
                          "counts": counts(plan, generation, asr, binding),
                          "training_admitted": False, "qualification_allowed": False}
                once_or_equal(root / "checkpoints" / (digest(latest) + ".json"), latest)
                if stop_on_confusion and screen["counts"].get("NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT", 0):
                    status = "CONFUSION_HOLD"
                    break
            if not progress:
                break
        final = counts(plan, generation, asr, binding)
        # A zero-new-call resume still revalidates raw bytes and produces the
        # same independently consumed output, not a cached success assertion.
        if generation.exists():
            screen = bulk_asr.screen_results(plan, generation, asr, adapters)
            effect, feedback = consumer(plan, screen, generation)
            latest = {"schema": SCHEMA, **binding, "screen": screen,
                      "effect": effect, "feedback": feedback, "counts": final,
                      "training_admitted": False, "qualification_allowed": False}
            once_or_equal(root / "checkpoints" / (digest(latest) + ".json"), latest)
        result = {"schema": SCHEMA, "status": status, **binding, **final,
                  "new_tts_calls": final["tts_started"] - before["tts_started"],
                  "new_asr_calls": final["asr_started"] - before["asr_started"],
                  "checkpoint_sha256": digest(latest) if latest is not None else None,
                  "training_admitted": False, "qualification_allowed": False}
        return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sources", "audio-root", "workspace", "adapter-lock", "pipeline-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-tts", required=True, type=int)
    parser.add_argument("--max-asr", required=True, type=int)
    parser.add_argument("--wall-seconds", type=float, default=60,
                        help="Stop launching new calls after this interval; does not preempt a running TTS/ASR call. Use an external process supervisor for a hard runtime limit.")
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument("--model-lock", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = p.read_json(args.plan)
    sources = p.read_json(args.sources)
    adapters = p.read_json(args.adapter_lock)
    p.verify_plan(plan)
    receipts.check_adapters(adapters)
    consumer = PipelineConsumer(args.pipeline_root)
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN_EXECUTE_REQUIRED", "model_calls": 0,
                          "pipeline_commit": PIPELINE_COMMIT, "training_admitted": False}))
        return 0
    if args.model_folder is None or args.model_lock is None:
        print(json.dumps({"status": "NOT_RUN_MODEL_ASSETS_MISSING", "model_calls": 0,
                          "training_admitted": False}))
        return 0
    p.require(os.environ.get("HF_HUB_OFFLINE") == "1" and
              os.environ.get("TRANSFORMERS_OFFLINE") == "1", "OFFLINE_RUNTIME_REQUIRED")
    model_lock = p.read_json(args.model_lock)
    bulk_runner.asset_lock(args.model_folder, model_lock, plan["model_revision"])
    backend = bulk_runner.QwenBulkClone(args.model_folder, device="cpu")
    result = execute(plan, sources, args.audio_root, args.workspace, adapters,
                     max_tts=args.max_tts, max_asr=args.max_asr,
                     wall_seconds=args.wall_seconds, consumer=consumer,
                     backend_identity={'model_id': plan['model_id'],
                         'model_revision': plan['model_revision'],
                         'asset_lock_sha256': digest(model_lock), 'provider_kind': 'qwen-offline'},
                     backend=backend, executor=subprocess.run)
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("BULK_AUTOPILOT_HOLD: " + str(exc), file=sys.stderr)
        raise SystemExit(2)

