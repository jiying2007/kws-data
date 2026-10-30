"""One new 16-cell preset batch in the retained offline container.

No old campaign launcher is invoked. No human/ASR label is synthesized.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import signal
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intake_contract import (BUDGET, CAMPAIGN, MODEL_ID, MODEL_REVISION, RECIPE, VOICES,
                      expected_plan, file_sha256, require, save_json, signal_metrics)


def main() -> None:
    output = Path("/output")
    require(output.is_dir() and not list(output.iterdir()), "new empty output mount required")
    sys.path.insert(0, str(HERE.parent / "source-screen32-v1"))
    from runtime_scope import verify_runtime_scope
    kernel = verify_runtime_scope()  # Before any inference dependency import.
    save_json(output / "kernel-scope.json", kernel)
    save_json(output / "progress.json", {"stage": "load", "index": 0,
                                        "started_monotonic": time.monotonic()})
    receipt = {"schema": "qwen16-intake-receipt-v1", "campaign": CAMPAIGN,
               "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "status": "started", "outcomes": [], "asr_calls": 0,
               "training_runs": 0, "training_admitted": False,
               "source_rights_status": "PRESET_SOURCE_RETAINED_NOT_INDEPENDENTLY_AUDITED"}
    receipt_path = output / "generation-receipt.json"
    save_json(receipt_path, receipt)
    original_generate = old_handler = talker = None
    try:
        # Original adapter and its locked 0.6B model selection are copied to a
        # fresh read-only staging tree; original Git files are never edited.
        sys.path.insert(0, str(HERE.parent / "tts-quality-pilot-v1"))
        from setup_adapter import verify_runtime, engine
        require(engine == "qwen06", "fixed 0.6B engine required")
        runtime_proof = verify_runtime(Path("/runtime"), "tts")
        # Retain a digest, not the full setup's paths/logs.
        from intake_contract import canonical
        receipt["runtime_verification_sha256"] = hashlib.sha256(canonical(runtime_proof)).hexdigest()
        gate = sys.modules["runtime_gate"]
        torch, transformers, network_events = gate.prepare()
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from qwen_tts import Qwen3TTSModel
        model_dir = Path("/runtime/models/qwen_tts")
        sidecars = gate.inspect_model_sidecars(model_dir)
        loader_calls = []
        with gate.local_loader_policy(model_dir, torch, transformers, loader_calls):
            start_load = time.monotonic()
            model = Qwen3TTSModel.from_pretrained(
                str(model_dir), device_map="cpu", dtype=torch.float32,
                attn_implementation="eager", use_safetensors=True, weights_only=True,
                local_files_only=True, trust_remote_code=False)
            receipt["model_load_seconds"] = time.monotonic() - start_load
            require(model.model.tts_model_type == "custom_voice" and
                    model.model.tts_model_size == "0b6", "loaded model identity")
            speakers = {name.lower() for name in model.get_supported_speakers()}
            require(all(name.lower() in speakers for name in VOICES), "missing preset")
            for owner in (model.model, model.model.speech_tokenizer.model):
                require(not owner.training, "eval mode required")
                require({str(p.device) for p in owner.parameters()} == {"cpu"} and
                        {str(p.dtype) for p in owner.parameters()} == {"torch.float32"},
                        "CPU FP32 parameters required")
            save_json(output / "runtime-observations.json",
                      {"sidecars": sidecars, "loader_calls": loader_calls,
                       "network_events": network_events})
            talker = model.model.talker
            original_generate = talker.generate
            termination = {}
            def capture(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                eos = int(model.model.config.talker_config.codec_eos_token_id)
                last = result.sequences[:, -1].detach().cpu().tolist()
                termination.update(last_token_ids=last, eos_token_id=eos,
                                   ended_with_eos=bool(last) and all(x == eos for x in last))
                return result
            def timeout(*_):
                raise TimeoutError("bounded cell time exceeded")
            talker.generate = capture
            old_handler = signal.signal(signal.SIGALRM, timeout)
            for index, cell in enumerate(expected_plan()):
                verify_runtime_scope()
                cell_id = cell["cell_id"]
                with (output / (cell_id + ".claim")).open("x") as stream:
                    stream.write(cell_id + "\n")
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                row = dict(cell, status="attempt_started", attempts=1)
                receipt["outcomes"].append(row)
                save_json(receipt_path, receipt)
                started = time.monotonic()
                save_json(output / "progress.json", {"stage": "cell", "index": index + 1,
                                                    "started_monotonic": started})
                termination.clear()
                random.seed(cell["seed"])
                np.random.seed(cell["seed"])
                torch.manual_seed(cell["seed"])
                signal.alarm(BUDGET["per_cell_seconds"])
                try:
                    with torch.inference_mode():
                        wavs, rate = model.generate_custom_voice(
                            text=cell["intended_text"], speaker=cell["speaker"], **RECIPE)
                    require(len(wavs) == 1, "one output required")
                    native = np.asarray(wavs[0])
                    require(native.ndim == 1 and native.dtype == np.dtype("float32"),
                            "mono native float32 output required")
                    require(type(rate) is int and rate == 24000, "native 24k integer rate required")
                    native_metrics = signal_metrics(native, rate,
                                                    eos=termination.get("ended_with_eos"))
                    raw_path = output / (cell_id + ".raw-float.wav")
                    # Preserve original floating values, not trimmed/normalized audio.
                    sf.write(raw_path, native, 24000, subtype="FLOAT")
                    row.update(native_metrics=native_metrics, native_termination=dict(termination),
                               raw_wav_sha256=file_sha256(raw_path),
                               native_float_values_sha256=hashlib.sha256(
                                   np.asarray(native, dtype="<f4").tobytes()).hexdigest())
                    save_json(receipt_path, receipt)
                    require(native_metrics["technically_valid"], "native audio technical failure")
                    derived = resample_poly(native.astype(np.float64), 2, 3)
                    require(derived.ndim == 1 and len(derived) == math.ceil(len(native) * 2 / 3),
                            "resampler geometry")
                    derived_metrics = signal_metrics(derived, 16000)
                    row["derived_metrics"] = derived_metrics
                    save_json(receipt_path, receipt)
                    # Do not hide resampling overshoot through PCM16 clipping.
                    require(derived_metrics["technically_valid"], "derived audio technical failure")
                    pcm_path = output / (cell_id + ".wav")
                    sf.write(pcm_path, derived, 16000, subtype="PCM_16")
                    with wave.open(str(pcm_path), "rb") as stream:
                        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth())
                                == (16000, 1, 2), "PCM16 geometry")
                        frames = stream.getnframes()
                        require(frames == len(derived), "PCM16 frame count")
                        pcm = stream.readframes(frames)
                        require(len(pcm) == 2 * frames, "complete PCM16 bytes")
                    row.update(status="generated_pending_review", wall_seconds=time.monotonic() - started,
                               wav_sha256=file_sha256(pcm_path),
                               pcm_sha256=hashlib.sha256(pcm).hexdigest(), frames=frames,
                               resample_policy="scipy-resample-poly-f64-2-over-3-no-gain-no-trim-v1")
                    save_json(receipt_path, receipt)
                except BaseException as error:
                    row.update(status="failed_no_retry", error_type=type(error).__name__)
                    save_json(receipt_path, receipt)
                    raise
                finally:
                    signal.alarm(0)
        receipt["status"] = "generated_16_pending_review"
    except BaseException as error:
        receipt.update(status="stopped_no_retry", error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        if talker is not None and original_generate is not None:
            talker.generate = original_generate
        save_json(receipt_path, receipt)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        import json
        print(json.dumps({"status": "stopped_no_retry", "error_type": type(error).__name__}))
        raise SystemExit(1)
