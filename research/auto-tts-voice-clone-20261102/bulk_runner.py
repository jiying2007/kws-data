"""Local, explicitly budgeted Qwen Base clone for many reviewed source clips.

Per source reference, extract the speaker prompt once. Generate multiple
positive/negative intended utterances without exposing raw source audio in Git.
All outputs remain unadmitted synthetic candidate material.
"""
from __future__ import annotations
import argparse
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import wave

import bulk_pool as p
import campaign as old_campaign
import runner as media

LOCK_SCHEMA = "kws-clone-base-model-assets-v1"
MAX_ASSET_TOTAL = 8 * 1024**3
MAX_ASSET_FILES = 256


def asset_lock(folder, lock, revision):
    p.require(type(lock) is dict and set(lock) ==
              {"schema", "model_id", "revision", "files"} and
              lock["schema"] == LOCK_SCHEMA and lock["model_id"] == p.MODEL and
              lock["revision"] == revision, "BASE_ASSET_LOCK")
    files = lock["files"]
    p.require(type(files) is list and 1 <= len(files) <= MAX_ASSET_FILES,
              "MODEL_ASSET_COUNT")
    folder = Path(folder)
    p.require(folder.is_dir() and not any(f.is_symlink()
              for f in (folder, *folder.parents)), "MODEL_ROOT_ALIAS")
    total = 0
    seen = set()
    for row in files:
        p.require(type(row) is dict and set(row) == {"path", "sha256", "bytes"},
                  "ASSET_FIELDS")
        name = row["path"]
        p.require(type(name) is str and 1 <= len(name) <= 180 and
                  not name.startswith("/") and "\\" not in name and
                  ".." not in Path(name).parts and name not in seen, "ASSET_PATH")
        seen.add(name)
        p._sha(row["sha256"])
        p.require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_ASSET_TOTAL,
                  "ASSET_LENGTH")
        size, digest = media._safe_file(folder/name, limit=MAX_ASSET_TOTAL)
        p.require(size == row["bytes"] and digest == row["sha256"],
                  "PINNED_MODEL_ASSET_CHANGED")
        total += size
        p.require(total <= MAX_ASSET_TOTAL, "MODEL_SIZE_BUDGET")
    found = set()
    for f in folder.rglob("*"):
        p.require(not f.is_symlink(), "MODEL_SYMLINK")
        if f.is_file():
            found.add(f.relative_to(folder).as_posix())
    p.require(found == seen, "UNLISTED_MODEL_ASSET")
    return folder


class QwenBulkClone:
    """Offline inference API, not invoked by no-model source tests."""

    def __init__(self, model_folder, *, device="cpu"):
        p.require(device in ("cpu", "cuda:0"), "DEVICE")
        self.folder = Path(model_folder)
        self.device = device
        self.model = None
        self.cache = {}
        self.prompt_creations = 0

    def _model(self):
        if self.model is not None:
            return self.model
        p.require(os.environ.get("HF_HUB_OFFLINE") == "1" and
                  os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                  "MODEL_OFFLINE_ONLY")
        import torch
        from qwen_tts import Qwen3TTSModel
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        if self.device != "cpu":
            p.require(torch.cuda.is_available(), "NO_CUDA_AVAILABLE")
        self.model = Qwen3TTSModel.from_pretrained(str(self.folder),
            device_map=self.device, dtype=dtype, attn_implementation="sdpa")
        return self.model

    def _reference(self, ref, audio_root):
        # WAV must be decoded from the same SHA-bound reference checked above.
        import numpy as np
        raw_path = Path(audio_root)/ref["audio_path"]
        current = p.measure_reference(audio_root,ref["audio_path"])
        p.require(all(current[k]==ref[k] for k in
                  ("wav_sha256","pcm_sha256","frames","sample_rate_hz")),
                  "REFERENCE_CHANGED_BEFORE_PROMPT")
        with wave.open(str(raw_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            b = w.readframes(frames)
        p.require(rate == ref["sample_rate_hz"] and frames == ref["frames"]
                  and hashlib.sha256(b).hexdigest() == ref["pcm_sha256"],
                  "REFERENCE_CHANGED_BEFORE_PROMPT")
        return np.frombuffer(b, dtype="<i2").astype("float32") / 32768.0, rate

    def _prompt(self, ref, audio_root):
        p.reference_text.validate_reference(ref)
        cache_id = (ref["source_family_id"], ref["wav_sha256"],
                    ref["reference_text_status"], ref["reference_text"],
                    hashlib.sha256(p.canonical(ref.get("reference_text_provenance"))).hexdigest())
        if cache_id in self.cache:
            return self.cache[cache_id]
        model = self._model()
        clip, rate = self._reference(ref, audio_root)
        if ref["reference_text_status"] in ("verified", p.reference_text.CHECKED):
            args = {"ref_audio": (clip, rate), "ref_text": ref["reference_text"],
                    "x_vector_only_mode": False}
        else:
            args = {"ref_audio": (clip, rate), "ref_text": None,
                    "x_vector_only_mode": True}
        prompt = model.create_voice_clone_prompt(**args)
        self.cache[cache_id] = prompt
        self.prompt_creations += 1
        return prompt

    def synthesize(self, cell, ref, audio_root):
        import torch
        torch.manual_seed(cell["seed"])
        if self.device == "cuda:0":
            torch.cuda.manual_seed_all(cell["seed"])
        model = self._model()
        prompt = self._prompt(ref, audio_root)
        wavs, sr = model.generate_voice_clone(
            text=cell["prompt"], language="Chinese",
            voice_clone_prompt=prompt, non_streaming_mode=True,
            do_sample=True, top_k=50, top_p=1.0,
            temperature=0.9, repetition_penalty=1.05)
        p.require(len(wavs) == 1, "ONE_WAV_PER_CALL")
        return wavs[0], int(sr)


def run(plan, source_manifest, audio_root, output, backend, *,
        max_new_calls, writer=media._write_waves):
    p.verify_plan(plan)
    expected = p.plan(source_manifest, audio_root,
        repetitions=plan["repetitions"], batch_id=plan["batch_id"],
        revision=plan["model_revision"],
        max_calls=plan["budgets"]["max_total_calls"], layout=plan.get("cell_layout"))
    p.require(p.canonical(expected) == p.canonical(plan),
              "SOURCE_OR_PLAN_CHANGED")
    p.require(type(max_new_calls) is int and
              0 <= max_new_calls <= plan["budgets"]["max_calls_per_invocation"],
              "RUN_INVOCATION_BUDGET")
    root = Path(output)
    plan_digest = hashlib.sha256(p.canonical(plan)).hexdigest()
    if not root.exists():
        root.mkdir(parents=True)
        media._once(root/"plan-lock.json", {
            "schema": "kws-bulk-clone-execution-lock-v1",
            "plan_sha256": plan_digest,
            "training_admitted": False, "model_id": p.MODEL})
    else:
        old = p.read_json(root/"plan-lock.json")
        p.require(type(old) is dict and
                  old.get("plan_sha256") == plan_digest and
                  old.get("training_admitted") is False, "EXECUTION_LOCK_DRIFT")
    refmap = {r["reference_id"]: r for r in plan["references"]}
    started_count = sum(
        (root/"cells"/cell["cell_id"]/"started.json").exists()
        for cell in plan["cells"])
    p.require(started_count <= plan["budgets"]["max_total_calls"],
              "BUDGET_ALREADY_EXCEEDED")
    remaining = plan["budgets"]["max_total_calls"] - started_count
    allowed = min(max_new_calls, remaining)
    invoked = 0
    # Group all samples from each source reference. Backend caches the expensive
    # Qwen speaker prompt once across its N positive/negative utterances.
    for cell in plan["cells"]:
        folder = root/"cells"/cell["cell_id"]
        if (folder/"started.json").exists():
            continue
        if invoked >= allowed:
            break
        p.require(not folder.exists(), "UNKNOWN_CELL_DIRECTORY")
        folder.mkdir(parents=True)
        media._once(folder/"started.json", {
            "schema": "kws-bulk-clone-start-v1", "cell_id": cell["cell_id"],
            "reference_id": cell["reference_id"],
            "reference_wav_sha256": refmap[cell["reference_id"]]["wav_sha256"],
            "plan_sha256": plan_digest, "source_family_id": cell["source_family_id"],
            "seed": cell["seed"], "attempts": 1,
            "training_admitted": False})
        invoked += 1
        try:
            audio, rate = backend.synthesize(
                cell, refmap[cell["reference_id"]], audio_root)
            metrics = media.quality(audio, rate)
            rendered = writer(folder, audio, rate, metrics)
            required = {"native_sha256", "native_bytes", "wav_sha256", "wav_bytes",
                        "pcm_sha256", "audio_frames", "sample_rate_hz"}
            p.require(type(rendered) is dict and set(rendered) == required and
                      all(type(rendered[k]) is int and rendered[k] > 0
                          for k in ("native_bytes","wav_bytes","audio_frames")) and
                      rendered["sample_rate_hz"] == 16000 and
                      all(type(rendered[k]) is str and p._SHA.fullmatch(rendered[k])
                          for k in ("native_sha256","wav_sha256","pcm_sha256")),
                      "GENERATED_MEDIA_INVALID")
            media._once(folder/"completed.json", {
                "schema":"kws-bulk-clone-complete-v1", "cell_id":cell["cell_id"],
                "status":"GENERATED_PENDING_ASR",
                "technical":metrics, "audio":rendered,
                "source_family_id":cell["source_family_id"],
                "intended_kind":cell["intended_kind"],
                "actual_text":None, "training_admitted":False,
                "qualification_allowed":False})
        except Exception as error:
            media._once(folder/"failed.json", {
                "schema":"kws-bulk-clone-failure-v1", "cell_id":cell["cell_id"],
                "status":"FAILED_NO_RETRY", "error_type":type(error).__name__,
                "training_admitted":False})
    return {
        "attempts_this_call":invoked,
        "total_started":sum((root/"cells"/x["cell_id"]/"started.json").exists()
                             for x in plan["cells"]),
        "total_completed":sum((root/"cells"/x["cell_id"]/"completed.json").exists()
                               for x in plan["cells"]),
        "model_calls_requested":invoked,
        "qualification_allowed":False, "training_admitted":False,
        "status":"BULK_SYNTHETIC_UNADMITTED"
    }


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sources",type=Path,required=True)
    cli.add_argument("--audio-root",type=Path,required=True)
    cli.add_argument("--plan",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--model-root",type=Path,required=True)
    cli.add_argument("--model-lock",type=Path,required=True)
    cli.add_argument("--max-new-calls",type=int,default=32)
    cli.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    cli.add_argument("--execute",action="store_true")
    args=cli.parse_args(argv)
    try:
        source=p.read_json(args.sources)
        plan=p.read_json(args.plan)
        p.verify_plan(plan)
        if not args.execute:
            p.require(plan["source_manifest_sha256"] ==
                      hashlib.sha256(p.canonical(source)).hexdigest(),
                      "PLAN_SOURCE_MISMATCH")
            print(json.dumps({"status":"VALIDATED_NO_MODEL_CALL",
                "planned_cells":len(plan["cells"]),
                "training_admitted":False}))
            return 0
        p.require(os.environ.get("HF_HUB_OFFLINE") == "1" and
                  os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                  "OFFLINE_EXECUTION_ONLY")
        folder=asset_lock(args.model_root,p.read_json(args.model_lock),
                          plan["model_revision"])
        result=run(plan,source,args.audio_root,args.out,
            QwenBulkClone(folder,device=args.device),
            max_new_calls=args.max_new_calls)
    except (ValueError, OSError) as error:
        print("BULK_CLONE_NOT_ADMITTED: "+type(error).__name__,file=sys.stderr)
        return 2
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
