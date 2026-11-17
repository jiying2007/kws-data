"""Public-CI-only native processor check; no speech model or model weights.

The fixed integer PCM fixture is independent of all campaign audio.  Production
and this check call the same prepare_native_pcm helper and preserve its batch.
Importing this module or --source-check uses the standard library only.
"""
from __future__ import annotations
import argparse
import base64
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import sys

HERE = Path(__file__).resolve().parent
OBSERVER_PATH = HERE / "english_kokoro_context32_mask_observe.py"
spec = importlib.util.spec_from_file_location("context32_native_mask_observer", OBSERVER_PATH)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)

SCHEMA = "english-kokoro-context32-processor-check-v1"
WORKFLOW = "jiying2007/kws-data/.github/workflows/english-kokoro-context32-maskfix-once.yml"
JOB = "processor-native-smoke"
TASK = "public-native-int32-smoke-v1"
DEADLINE_SECONDS = 120
FIXTURE = {"id": "context32-processor-int16-fixture-v1", "samples": 640,
    "sample_rate": 16000, "channels": 1, "sample_width": 2,
    "integer_rule": "((index*73+19)%16001)-8000; index=0..639",
    "campaign_audio": False, "model_input_task": False}
PREPROCESSOR_CONFIG_BYTES = base64.b64decode(
    "ewogICJkb19ub3JtYWxpemUiOiB0cnVlLAogICJmZWF0dXJlX2V4dHJhY3Rvcl90eXBlIjogIldhdjJWZWMyRmVhdHVyZUV4dHJhY3RvciIsCiAgImZlYXR1cmVfc2l6ZSI6IDEsCiAgInBhZGRpbmdfc2lkZSI6ICJyaWdodCIsCiAgInBhZGRpbmdfdmFsdWUiOiAwLAogICJyZXR1cm5fYXR0ZW50aW9uX21hc2siOiB0cnVlLAogICJzYW1wbGluZ19yYXRlIjogMTYwMDAKfQo=",
    validate=True,
)


class ProcessorCheckInterrupted(BaseException):
    pass


def interrupted(signum, frame):
    raise ProcessorCheckInterrupted("PUBLIC_PROCESSOR_CHECK_INTERRUPTED: " + str(signum))


def fixture_pcm():
    return struct.pack("<640h", *[((index * 73 + 19) % 16001) - 8000 for index in range(640)])


def source_preflight():
    pin = {"bytes": len(PREPROCESSOR_CONFIG_BYTES), "sha256": observer.sha(PREPROCESSOR_CONFIG_BYTES)}
    observer.need(pin == observer.COMMON_FILES["preprocessor_config.json"], "EXACT_OFFICIAL_PREPROCESSOR_BYTES")
    config = json.loads(PREPROCESSOR_CONFIG_BYTES)
    observer.need(config == {"do_normalize": True, "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1, "padding_side": "right", "padding_value": 0,
        "return_attention_mask": True, "sampling_rate": 16000}, "EXACT_OFFICIAL_PREPROCESSOR_FIELDS")
    raw = fixture_pcm()
    return {"schema": SCHEMA, "fixture": FIXTURE,
        "fixture_pcm": {"bytes": len(raw), "sha256": observer.sha(raw)},
        "preprocessor_config": pin, "native_processor_source_files": observer.PROCESSOR_SOURCE_FILES,
        "runtime_versions": observer.RUNTIME_VERSIONS, "native_mask_dtype": "torch.int32",
        "processor_calls": 0, "model_constructions": 0, "model_forwards": 0, "network_calls": 0}


def public_guard(env=None):
    env = os.environ if env is None else env
    observer.need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == "jiying2007/kws-data"
        and env.get("GITHUB_RUN_ATTEMPT") == "1" and env.get("GITHUB_JOB") == JOB
        and env.get("KWS_CONTEXT32_PROCESSOR_TASK") == TASK, "FIRST_PUBLIC_PROCESSOR_SOURCE_TASK_ONLY")
    observer.need(type(env.get("GITHUB_WORKFLOW_REF")) is str
        and env["GITHUB_WORKFLOW_REF"].startswith(WORKFLOW + "@refs/"), "EXACT_PROCESSOR_CHECK_WORKFLOW")
    observer.need(re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None
        and re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")) is not None, "PUBLIC_RUN_IDENTITY")
    event = json.loads(observer.regular(Path(env["GITHUB_EVENT_PATH"])))
    repository = event.get("repository", {})
    observer.need(repository.get("full_name") == "jiying2007/kws-data" and repository.get("private") is False,
        "PUBLIC_REPOSITORY_EVENT")
    kind, ref = env.get("GITHUB_EVENT_NAME"), env.get("GITHUB_REF")
    if kind == "pull_request":
        pr = event.get("pull_request", {})
        observer.need(type(event.get("number")) is int and event["number"] > 0
            and ref == "refs/pull/" + str(event["number"]) + "/merge"
            and pr.get("base", {}).get("ref") == "main", "PUBLIC_MAIN_TARGET_PR")
        for side in ("head", "base"):
            repo = pr.get(side, {}).get("repo", {})
            observer.need(repo.get("full_name") == "jiying2007/kws-data" and repo.get("private") is False,
                "SAME_PUBLIC_REPOSITORY_PR_ONLY")
    else:
        observer.need(kind == "push" and ref == "refs/heads/main" and event.get("ref") == ref
            and event.get("after") == env["GITHUB_SHA"], "PUBLIC_MAIN_PUSH_SOURCE_TASK_ONLY")
    return {"schema": SCHEMA, "repository": env["GITHUB_REPOSITORY"], "commit": env["GITHUB_SHA"],
        "run_id": env["GITHUB_RUN_ID"], "run_attempt": env["GITHUB_RUN_ATTEMPT"], "event": kind,
        "workflow_ref": env["GITHUB_WORKFLOW_REF"], "job": JOB, "task": TASK,
        "source_task_only": True, "campaign_audio_consumed": False}


def _load_runtime(output):
    """Only the feature extractor is imported; no model class is constructed."""
    versions = {name: importlib.metadata.version(name) for name in observer.RUNTIME_VERSIONS}
    observer.once(output / "RUNTIME_VERSIONS.json", versions)
    observer.need(versions == observer.RUNTIME_VERSIONS, "EXACT_PUBLIC_PROCESSOR_RUNTIME")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import torch
    import transformers
    from transformers import Wav2Vec2FeatureExtractor
    sources = observer.verify_processor_sources(Path(transformers.__file__).resolve().parent)
    observer.once(output / "NATIVE_PROCESSOR_SOURCE_FILES.json", sources)
    return np, torch, Wav2Vec2FeatureExtractor, versions, sources


def verify_saved_inputs(output, facts, normalized, mask):
    samples = FIXTURE["samples"]
    observer.need(facts["normalized_dtype"] == "torch.float32" and facts["attention_mask_dtype"] == "torch.int32"
        and facts["normalized_shape"] == facts["attention_mask_shape"] == [1, samples]
        and facts["attention_mask_sum"] == samples and facts["attention_mask_unique_values"] == [1]
        and facts["right_padding_samples"] == 0, "ACTUAL_FIXED_NATIVE_PROCESSOR_CONTRACT")
    norm_raw = observer.regular(output / normalized["path"])
    mask_raw = observer.regular(output / mask["path"])
    observer.need(len(norm_raw) == len(mask_raw) == samples * 4
        and observer.sha(norm_raw) == normalized["sha256"] and observer.sha(mask_raw) == mask["sha256"],
        "NATIVE_PROCESSOR_BYTES_RETAINED")
    observer.need(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", norm_raw))
        and all(x[0] == 1 for x in struct.iter_unpack("<i", mask_raw)), "NATIVE_ARRAY_READBACK")


def run(output_root, env=None):
    guard = public_guard(env)
    source = source_preflight()
    output = Path(output_root)
    observer.need(output.is_absolute() and not output.exists() and output.parent.is_dir()
        and not any(p.is_symlink() for p in output.parents), "CREATE_ONLY_PROCESSOR_OUTPUT")
    output.mkdir(mode=0o700)
    observer.once(output / "GUARD.json", guard)
    observer.once(output / "STARTED.json", {"schema": SCHEMA, "run": guard,
        "processor_check_code_sha256": observer.sha(observer.regular(Path(__file__))),
        "observer_code_sha256": observer.sha(observer.regular(OBSERVER_PATH)), "fixture": FIXTURE,
        "model_constructions": 0, "model_forwards": 0, "maximum_processor_calls": 1, "retry_count": 0})
    counters = {"processor_construction_attempts": 0, "processor_instances": 0,
        "processor_call_attempts": 0, "processor_native_returns": 0}
    failure, capture = None, None
    versions, sources = None, None
    old_alarm = signal.signal(signal.SIGALRM, interrupted)
    old_term = signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(DEADLINE_SECONDS)
    try:
        observer.put(output / "preprocessor_config.json", PREPROCESSOR_CONFIG_BYTES)
        fixture_pin = observer.put(output / "fixture.pcm", fixture_pcm())
        np, torch, factory, versions, sources = _load_runtime(output)
        observer.once(output / "PROCESSOR_CREATE_STARTED.json", {"schema": SCHEMA, "attempt": 1})
        counters["processor_construction_attempts"] = 1
        extractor = factory.from_pretrained(str(output), local_files_only=True)
        counters["processor_instances"] = 1
        observer.once(output / "PROCESSOR_CREATED.json", {"schema": SCHEMA, "processor_instances": 1,
            "class": "transformers.Wav2Vec2FeatureExtractor", "model_constructions": 0})
        observer.need(extractor.sampling_rate == 16000 and extractor.do_normalize is True
            and extractor.return_attention_mask is True, "UNCHANGED_OFFICIAL_FEATURE_EXTRACTOR")

        def counted_extractor(values, **kwargs):
            observer.need(counters["processor_call_attempts"] == 0, "ONE_PROCESSOR_CALL_NO_RETRY")
            observer.once(output / "PROCESSOR_CALL_STARTED.json", {"schema": SCHEMA, "attempt": 1})
            counters["processor_call_attempts"] = 1
            batch = extractor(values, **kwargs)
            counters["processor_native_returns"] = 1
            observer.once(output / "PROCESSOR_RETURNED.json", {"schema": SCHEMA, "returned": True,
                "input_values_dtype": str(batch["input_values"].dtype),
                "attention_mask_dtype": str(batch["attention_mask"].dtype),
                "input_values_shape": list(batch["input_values"].shape),
                "attention_mask_shape": list(batch["attention_mask"].shape)})
            return batch

        _, facts, normalized_raw, mask_raw = observer.prepare_native_pcm(
            np, torch, counted_extractor, fixture_pcm(), {"pcm_samples": FIXTURE["samples"]})
        facts = facts | {"normalization_scope": "Entire deterministic non-corpus PCM fixture."}
        normalized = observer.put(output / "normalized-input.f32", normalized_raw) | {
            "dtype": "float32-little-endian", "order": "C", "shape": [1, FIXTURE["samples"]]}
        mask = observer.put(output / "attention-mask.i32", mask_raw) | {
            "dtype": "int32-little-endian", "order": "C", "shape": [1, FIXTURE["samples"]]}
        verify_saved_inputs(output, facts, normalized, mask)
        capture = {"schema": SCHEMA, "run": guard, "status": "NATIVE_PROCESSOR_CAPTURED",
            "fixture": FIXTURE, "fixture_pcm": fixture_pin, "preprocessor_config": source["preprocessor_config"],
            "runtime_versions": versions, "native_processor_source_files": sources, "processor": facts,
            "normalized_input": normalized, "attention_mask": mask,
            "model_batch_modified": False, "campaign_audio_consumed": False,
            "model_constructions": 0, "model_forwards": 0, "weights_downloaded": False,
            "observer_code_sha256": observer.sha(observer.regular(OBSERVER_PATH)),
            "processor_check_code_sha256": observer.sha(observer.regular(Path(__file__)))}
        observer.once(output / "CAPTURE.json", capture)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old_term)
        passed = failure is None and capture is not None and counters == {
            "processor_construction_attempts": 1, "processor_instances": 1,
            "processor_call_attempts": 1, "processor_native_returns": 1}
        result = {"schema": SCHEMA, "run": guard,
            "status": "PASSED_NATIVE_INT32_PROCESSOR_ONLY" if passed else "FAILED_NATIVE_PROCESSOR_CHECK_NO_RETRY",
            "passed": passed, **counters, "model_constructions": 0, "model_forwards": 0,
            "weights_downloaded": False, "campaign_audio_consumed": False, "retry_count": 0,
            "fixture": FIXTURE, "runtime_versions": versions, "native_processor_source_files": sources,
            "observer_code_sha256": observer.sha(observer.regular(OBSERVER_PATH)),
            "processor_check_code_sha256": observer.sha(observer.regular(Path(__file__))), "failure": failure}
        observer.once(output / "RESULT.json", result)
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["--source-check"]:
        print(json.dumps(source_preflight(), sort_keys=True))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        outcome = run(args.output)
        print(json.dumps(outcome, sort_keys=True))
        raise SystemExit(0 if outcome["passed"] else 1)
