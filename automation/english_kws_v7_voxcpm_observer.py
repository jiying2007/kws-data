"""One-pass blind ASR/phoneme observers for the saved 12-clip VoxCPM pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import english_kws_v7_voxcpm_pilot as pilot
import english_kws_v7_voxcpm_policy as policy

PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
SCHEMA = "english-kws-v7-voxcpm-pilot-observer-v1"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    return pilot.read_json(path)


def write_once(path, value):
    return pilot.write_once(path, value)


def source_check():
    plan, _ = pilot.load_plan()
    result = policy.source_check()
    need(policy.UPSTREAM == {
        "repository": "jiying2007/kws-pipeline",
        "commit": "ec414dc89c52fb5756e9560e79517bcd41784248",
        "path": "training/english_kws_dual_track_v7_silver.py",
        "blob_sha": "d7c0d68b7ad7ed7e79f67b0e15037cd42ab74ffb",
    }, "EXACT_UPSTREAM_V7_ADMISSION_POLICY")
    need(set(policy.MODEL_PINS) == set(PROVIDERS), "EXACT_FOUR_OBSERVER_SET")
    return result | {
        "source_check": "PASS",
        "source_schema": plan["schema"],
        "planned_calls_per_provider": 12,
        "new_model_calls": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def validate_source(root):
    root = Path(root)
    pilot.readback(root)
    source = read_json(root / "SOURCE.json")
    need(source.get("schema") == pilot.SCHEMA
         and source.get("denominator") == 12
         and isinstance(source.get("rows"), list) and len(source["rows"]) == 12
         and source.get("training_allowed") is False
         and source.get("human_gold") is False
         and source.get("calibration_audio_opened") is False
         and source.get("locked_test_audio_opened") is False,
         "PILOT_SOURCE_DENOMINATOR_AND_FLAGS")
    return source, sha(pilot.regular(root / "SOURCE.json", 4 * 1024 * 1024))


def _load_asr(provider, model_root):
    import english_seven_voice_asr_once as seven
    model_root = Path(model_root)
    model_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    decoder = seven._load(provider, model_root)
    provenance = seven.PROVENANCE[provider]
    identity = policy.observer_identity(provider)
    return decoder, {
        **identity,
        "model_id": provenance[1],
        "model_revision": provenance[2],
        "adapter": "english_seven_voice_asr_once._load",
        "target_conditioned": False,
        "initialization_model_calls": 0,
        "training_allowed": False,
    }


def _load_phone(model_root):
    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    import english_hifi_micro as micro
    import english_hifi_phones_once as phones

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_id, revision = micro.PHONE_MODELS[0]
    need(model_id == "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
         and revision == "2c733782da5604684829819a5eb744c193fe9398",
         "EXACT_PINNED_XLSR_PHONE_MODEL")
    info = HfApi(token=False).model_info(model_id, revision=revision)
    need(info.sha.startswith(revision), "PINNED_XLSR_MODEL_REVISION")
    cache_root = Path(model_root) / "hf-cache"
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder = Path(snapshot_download(
        repo_id=model_id, revision=info.sha,
        cache_dir=str(cache_root), max_workers=2, token=False,
    ))
    weights = folder / "pytorch_model.bin"
    need(weights.is_file() and weights.stat().st_size == 1263535127,
         "PINNED_XLSR_WEIGHT_SIZE")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    need(digest.hexdigest() == phones.WEIGHT_SHA[0], "PINNED_XLSR_WEIGHT_SHA256")
    vocab = phones.read_verified_hf_metadata(folder, cache_root, "vocab.json")
    feature = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    config = Wav2Vec2Config.from_json_file(str(folder / "config.json"))
    model = Wav2Vec2ForCTC(config)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    need(config.pad_token_id == 0 and feature.sampling_rate == 16000,
         "PINNED_XLSR_INPUT_AND_BLANK")
    identity = policy.observer_identity("phoneme")
    lock = {
        **identity, "model_id": model_id, "model_revision": info.sha,
        "weight_sha256": digest.hexdigest(), "weight_bytes": weights.stat().st_size,
        "input": "entire uncropped PCM16 mono 16 kHz converted to float32",
        "attention_mask": "all valid samples; no padding",
        "target_conditioned": False, "initialization_model_calls": 0,
        "training_allowed": False,
    }

    def forward(pcm):
        values = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        inputs = feature(values, sampling_rate=16000, return_tensors="pt",
                         padding=False, return_attention_mask=True)
        need(inputs.input_values.shape[-1] == len(values)
             and inputs.attention_mask.shape[-1] == len(values),
             "WHOLE_PCM_UNPADDED_PHONE_INPUT")
        with torch.inference_mode():
            logits = model(input_values=inputs.input_values,
                           attention_mask=inputs.attention_mask.int()).logits
        frame_ids = logits.argmax(dim=-1)[0].tolist()
        tokens = phones.ctc(frame_ids, vocab, config.pad_token_id)
        need(isinstance(tokens, list) and tokens, "NONEMPTY_PHONE_OBSERVATION")
        return tokens, {
            "input_samples": len(values),
            "logit_frames": int(logits.shape[1]),
            "vocab_size": int(logits.shape[2]),
            "target_conditioned": False,
            "input_coverage_complete": True,
        }

    return forward, lock


def _source_audio(root, row):
    wav = pilot.regular(pilot.within(root, row["wav_path"]), pilot.MAX_WAV_BYTES)
    pcm = pilot.regular(pilot.within(root, row["pcm_path"]), pilot.MAX_PCM_BYTES)
    decoded, samples = pilot.audio_info(wav, expected_rate=16000)
    need(decoded == pcm and sha(wav) == row["wav_sha256"]
         and sha(pcm) == row["pcm_sha256"]
         and samples == row["pcm_samples"],
         "OBSERVER_INPUT_SOURCE_BYTE_IDENTITY")
    return wav, pcm


def file_inventory(root):
    return pilot.file_inventory(root)


def run(provider, source_root, output_root, model_root):
    need(provider in PROVIDERS, "FIXED_PROVIDER_ONLY")
    plan, plan_raw = pilot.load_plan()
    run_identity = pilot.guard(plan)
    source_root = Path(source_root)
    source, source_sha = validate_source(source_root)
    output = Path(output_root).absolute()
    model_root = Path(model_root).absolute()
    need(output.parent.is_dir() and not output.exists()
         and model_root.parent.is_dir() and not model_root.exists()
         and output != model_root and output not in model_root.parents
         and model_root not in output.parents,
         "FRESH_SEPARATE_OBSERVER_AND_MODEL_ROOTS")
    output.mkdir(mode=0o700)
    model_root.mkdir(mode=0o700)
    (output / "rows").mkdir(mode=0o700)
    started = {
        "schema": SCHEMA, "provider": provider, "run": run_identity,
        "source_sha256": source_sha, "denominator": 12,
        "maximum_calls": 12, "retry_count": 0,
        "target_conditioned": False, "training_allowed": False,
    }
    write_once(output / "STARTED.json", started)
    ledger = []
    attempts = returned = completed = 0
    initialization_error = None
    lock = None
    asr_decoder = None
    phone_forward = None
    try:
        if provider == "phoneme":
            phone_forward, lock = _load_phone(model_root)
        else:
            asr_decoder, lock = _load_asr(provider, model_root)
        write_once(output / "MODEL_LOCK.json", lock)
    except Exception as exc:
        initialization_error = type(exc).__name__ + ":" + str(exc)[:1000]
    for row in source["rows"]:
        item = {
            "id": row["id"], "status": "unknown",
            "input_started": False, "forward_attempts": 0,
            "forward_returned": False, "capture_saved": False,
        }
        directory = output / "rows" / row["id"]
        directory.mkdir(mode=0o700)
        if row.get("generation_status") != "success" or initialization_error is not None:
            item["status"] = "not_started"
            write_once(directory / "UNOBSERVED.json", {
                **item, "reason": initialization_error or "SOURCE_AUDIO_NOT_VALID",
                "retry_count": 0, "training_allowed": False,
            })
            ledger.append(item)
            continue
        base = {
            "schema": policy.CAPTURE_SCHEMA,
            **policy.observer_identity(provider),
            "id": row["id"],
            "source_pcm_sha256": row["pcm_sha256"],
            "source_wav_sha256": row["wav_sha256"],
            "source_samples": row["pcm_samples"],
            "target_conditioned": False,
            "input_coverage_complete": True,
            "decoder_output_complete": True,
            "human_gold": False,
            "training_allowed": False,
        }
        write_once(directory / "STARTED.json", base)
        item["input_started"] = True
        attempts += 1
        item["forward_attempts"] = 1
        write_once(directory / "FORWARD_STARTED.json", base)
        try:
            wav, pcm = _source_audio(source_root, row)
            if provider == "phoneme":
                phone_tokens, facts = phone_forward(pcm)
                capture = base | {"phone_tokens": phone_tokens, "phone_input_facts": facts,
                                  "status": "success"}
            else:
                transcript = asr_decoder(pilot.within(source_root, row["wav_path"]), directory)
                need(isinstance(transcript, str) and 0 < len(transcript.strip()) <= 4096
                     and not any(ord(ch) < 32 for ch in transcript.strip()),
                     "RAW_COMPLETE_BLIND_ASR_TEXT")
                capture = base | {"raw_text": transcript, "status": "success"}
            item["forward_returned"] = True
            returned += 1
            write_once(directory / "FORWARD_RETURNED.json", base)
            pin = write_once(directory / "CAPTURE.json", capture)
            item.update(status="success", capture_saved=True,
                        capture_path="rows/" + row["id"] + "/CAPTURE.json",
                        capture_sha256=pin["sha256"])
            completed += 1
        except Exception as exc:
            item.update(status="failed", error=type(exc).__name__ + ":" + str(exc)[:1000])
            write_once(directory / "FAILED.json", {
                **base, "forward_attempts": 1,
                "forward_returned": item["forward_returned"],
                "capture_saved": False, "error": item["error"],
                "retry_count": 0,
            })
        ledger.append(item)
    result = {
        "schema": SCHEMA, "provider": provider, "run": run_identity,
        "source_sha256": source_sha, "denominator": 12,
        "actual_forward_attempts": attempts, "native_forward_returns": returned,
        "completed_count": completed,
        "failed_count": sum(item["status"] == "failed" for item in ledger),
        "not_started_count": sum(item["status"] == "not_started" for item in ledger),
        "retry_count": 0, "initialization_error": initialization_error,
        "rows": ledger, "model_lock": lock,
        "target_conditioned": False, "human_gold": False,
        "training_allowed": False, "promotion_allowed": False,
    }
    need(len(ledger) == 12 and 0 <= completed <= returned <= attempts <= 12,
         "OBSERVER_COUNTER_INVARIANTS")
    write_once(output / "RESULT.json", result)
    write_once(output / "FILES.json", {
        "schema": SCHEMA + "/files", "self_excluded": "FILES.json",
        "files": file_inventory(output),
    })
    return result


def readback(provider, source_root, observer_root):
    need(provider in PROVIDERS, "FIXED_PROVIDER")
    source, source_sha = validate_source(source_root)
    root = Path(observer_root)
    result = read_json(root / "RESULT.json")
    inventory = read_json(root / "FILES.json")
    need(inventory == {
        "schema": SCHEMA + "/files", "self_excluded": "FILES.json",
        "files": file_inventory(root),
    }, "OBSERVER_FILE_INVENTORY_CLOSURE")
    need(result.get("schema") == SCHEMA and result.get("provider") == provider
         and result.get("source_sha256") == source_sha
         and result.get("denominator") == 12 and result.get("retry_count") == 0
         and result.get("training_allowed") is False
         and result.get("human_gold") is False
         and len(result.get("rows", [])) == 12,
         "OBSERVER_TERMINAL_IDENTITY_AND_DENOMINATOR")
    source_by_id = {row["id"]: row for row in source["rows"]}
    for item in result["rows"]:
        row = source_by_id[item["id"]]
        if item["status"] != "success":
            continue
        cap = read_json(pilot.within(root, item["capture_path"]))
        need(sha(pilot.regular(pilot.within(root, item["capture_path"]), 1024*1024))
             == item["capture_sha256"]
             and cap.get("id") == row["id"]
             and cap.get("source_pcm_sha256") == row["pcm_sha256"]
             and cap.get("source_wav_sha256") == row["wav_sha256"]
             and cap.get("target_conditioned") is False
             and cap.get("input_coverage_complete") is True,
             "OBSERVER_CAPTURE_READBACK")
    return {
        "provider": provider,
        "status": "PASS" if result["completed_count"] == 12 else "INCOMPLETE",
        "denominator": 12,
        "actual_forward_attempts": result["actual_forward_attempts"],
        "native_forward_returns": result["native_forward_returns"],
        "completed_count": result["completed_count"],
        "failed_count": result["failed_count"],
        "not_started_count": result["not_started_count"],
        "retry_count": 0,
        "training_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("source-check")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--provider", choices=PROVIDERS, required=True)
    run_parser.add_argument("--source", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model-root", type=Path, required=True)
    check = sub.add_parser("readback")
    check.add_argument("--provider", choices=PROVIDERS, required=True)
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--observer", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source-check":
        print(json.dumps(source_check(), sort_keys=True))
    elif args.action == "run":
        result = run(args.provider, args.source, args.output, args.model_root)
        print(json.dumps({
            "provider": args.provider,
            "completed": result["completed_count"],
            "attempts": result["actual_forward_attempts"],
            "training_allowed": False,
        }, sort_keys=True))
    else:
        print(json.dumps(readback(args.provider, args.source, args.observer), sort_keys=True))


if __name__ == "__main__":
    main()
