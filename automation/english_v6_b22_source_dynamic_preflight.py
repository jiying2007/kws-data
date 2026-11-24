"""Check the real text-observer import path without assets, models or audio.

Run this file from the captured producer root. The unchanged raw ASR loader is
allowed through its actual helper checksum, version and provider import checks,
then stopped at the first asset acquisition boundary. The dynamically imported
runner/campaign pair is checked separately with tiny temporary metadata files.
This is an execution-dependency check, never acoustic or model-load evidence.
"""
from __future__ import annotations

import ast
import builtins
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import urllib.request
import wave
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
POCKET = "automation/english_pocket12_observe.py"
CAMPAIGN = "research/auto-tts-voice-clone-20261102/campaign.py"
RUNNER = "research/auto-tts-voice-clone-20261102/runner.py"
FORBIDDEN = frozenset(("numpy", "torch", "torchaudio", "onnxruntime", "soundfile",
    "transformers", "piper", "qwen_asr", "qwen_tts", "whisper", "huggingface_hub",
    "requests", "httpx"))


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def required_paths(repository=ROOT):
    """Read the existing runtime checksum authority without importing backends."""
    root = Path(repository).resolve()
    tree = ast.parse((root / POCKET).read_bytes())
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "HELPER_PINS"
                           for target in node.targets)]
    need(len(assignments) == 1, "one original runtime helper checksum authority")
    pins = ast.literal_eval(assignments[0])
    need(type(pins) is dict and len(pins) == 19 and {CAMPAIGN, RUNNER} <= set(pins),
         "all nineteen original helpers, including both dynamic research modules")
    for name, digest in pins.items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts and name.endswith(".py")
             and type(digest) is str and re.fullmatch(r"[a-f0-9]{64}", digest),
             "safe immutable runtime helper pin")
    return pins


class AssetBoundary(Exception):
    """Expected stop before any original asset acquisition or model construction."""


def check(repository=ROOT):
    root = Path(repository).resolve()
    need(root == ROOT, "run the actual preflight file inside its captured producer root")
    pins = required_paths(root)
    for name, digest in pins.items():
        path = root / name
        need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
             and sha(path.read_bytes()) == digest, "complete actual runtime helper: " + name)
    need(not (set(sys.modules) & FORBIDDEN), "run dependency preflight before any model runtime import")
    sys.path.insert(0, str(root / "automation"))
    sys.path.insert(1, str(root))
    old_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        need(name.split(".")[0] not in FORBIDDEN, "no model or network runtime during dependency preflight")
        return old_import(name, *args, **kwargs)

    def prohibited(*args, **kwargs):
        raise ValueError("no network, native process or audio read during dependency preflight")

    def boundary(label):
        def stop(*args, **kwargs):
            raise AssetBoundary(label)
        return stop

    with tempfile.TemporaryDirectory(prefix="b22-runtime-dependency-metadata-") as temp, ExitStack() as stack:
        stack.enter_context(mock.patch("builtins.__import__", side_effect=import_guard))
        for name in ("socket.socket", "socket.create_connection", "urllib.request.urlopen",
                     "subprocess.Popen", "os.system", "wave.open"):
            stack.enter_context(mock.patch(name, side_effect=prohibited))
        stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
        raw = importlib.import_module("english_v6_b18_asr_adapter")
        pocket = importlib.import_module("english_pocket12_observe")
        need(pocket.ROOT.resolve() == root and pocket.HELPER_PINS == pins,
             "the actual loader checks these captured files, without checkout fallback")
        initial = pocket.source_check()
        need(initial["helper_sha256"] == pins and initial["new_model_calls"] == initial["new_network_calls"] == 0,
             "complete original runtime source_check")
        qwen = importlib.import_module("english_qwen_asr_real_once")
        whisper = importlib.import_module("english_whisper_real_once")
        parakeet = importlib.import_module("english_parakeet_real_once")
        english = importlib.import_module("english_qwen_capture")
        runner = english.media()
        campaign = runner.c
        need(Path(runner.__file__).resolve() == root / RUNNER
             and Path(campaign.__file__).resolve() == root / CAMPAIGN,
             "dynamic runner and campaign imports resolve inside the captured root")
        fixture = Path(temp) / "metadata.json"
        fixture.write_bytes(b'{"dependency_fixture":true}\n')
        need(campaign.read_json(fixture) == {"dependency_fixture": True}
             and runner._safe_file(fixture, limit=1024) == (fixture.stat().st_size, sha(fixture.read_bytes())),
             "actual campaign JSON and runner bounded-file operations, with metadata only")
        entries = (("qwen", qwen, "lock_model"), ("whisper", whisper, "pinned_model"),
                   ("parakeet", parakeet, "verify_runtime"))
        checked = {}
        for provider, module, operation in entries:
            label = provider + "." + operation
            assets = Path(temp) / (provider + "-assets")
            with mock.patch.object(raw.importlib.metadata, "version", side_effect=pocket.TEXT_RUNTIME[provider].__getitem__), \
                 mock.patch.object(pocket, "source_check", wraps=pocket.source_check) as source_check, \
                 mock.patch.object(module, operation, side_effect=boundary(label)) as stop:
                try:
                    raw._load(provider, assets, Path(temp))
                except AssetBoundary as exc:
                    need(str(exc) == label, "exact expected provider asset boundary")
                else:
                    raise ValueError("loader did not stop before original asset acquisition")
                need(source_check.call_count == stop.call_count == 1 and assets.is_dir()
                     and not any(assets.iterdir()), "source check preceded one empty asset-boundary fixture")
            checked[provider] = dict(original_source_check=True, boundary=label,
                assets_acquired=0, models_constructed=0, audio_reads=0, forwards=0)
        local_modules = {}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if not isinstance(filename, str):
                continue
            path = Path(filename).resolve()
            if path.is_relative_to(root) and path.is_file():
                local_modules[path.relative_to(root).as_posix()] = sha(path.read_bytes())
            if name.startswith(("english_", "multilingual_campaign", "kws_data_contract")) or name in ("runner", "campaign"):
                need(path.is_relative_to(root), "no original-checkout module leakage: " + name)
        need(not (set(sys.modules) & FORBIDDEN), "no model runtime was imported")
    return dict(schema="english-v6-b22-original-text-runtime-dependency-preflight-v1",
        status="PASS_IMPORT_LOCK_RUNNER_ONLY", code_reads_only=True, no_model=True, no_network=True,
        helper_pins=pins, loaded_repository_modules=dict(sorted(local_modules.items())),
        provider_boundaries=checked, new_model_calls=0, new_network_calls=0, new_audio_reads=0,
        model_construction_verified=False, acoustic_evidence=False)


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True, separators=(",", ":")))
