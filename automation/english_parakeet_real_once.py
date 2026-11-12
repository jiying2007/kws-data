"""Third independent English ASR on original ten WAVs: Parakeet TDT v3 Q8 GGUF.

Uses the pinned NVIDIA NeMo-Speech.cpp CPU binary and original NVIDIA GGUF.
Qwen and Whisper transcripts are never read by this recognizer. No data
admission, human gold, new TTS, or production model claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request

import english_blind_asr as blind
import english_qwen_asr_sourcefix_once as sourcefix
import english_qwen_capture as english
import multilingual_campaign as market

POLICY = "english-parakeet-tdt-v3-real-once-20261112-v1"
MODEL = "nvidia/parakeet-tdt-0.6b-v3"
PROVIDER = "parakeet-tdt-v3-en"
MODEL_COMMIT = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_FILE = "parakeet-tdt-0.6b-v3.q8_0.gguf"
MODEL_SHA256 = "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e"
MAX_MODEL_BYTES = 850 * 1024**2

RUNTIME = "NVIDIA/NeMo-Speech.cpp"
RUNTIME_TAG = "v0.3.0"
RUNTIME_SHA256 = "255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e"
RUNTIME_BYTES = 9605101
RUNTIME_FILE = "nemo-speech-0.3.0-linux-x86_64-cpu.tar.gz"
RUNTIME_URL = ("https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/"
               + RUNTIME_TAG + "/" + RUNTIME_FILE)
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-parakeet-real-20261112.json"
MAX_AUDIO = 10


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def need(condition: bool, code: str):
    if not condition:
        raise ValueError(code)


def request() -> dict:
    value = market.read(TRIGGER)
    expected = {
        "schema": POLICY,
        "original_tts_run_id": 43214159818,
        "original_tts_head_sha": "f2b91001e4fd56a03bbce9bc6b0c1cae7fad1dc8",
        "original_audio_artifact_id": 13942568634,
        "original_audio_artifact_sha256": "0235b0064006f79bfe9c8fa551c2a1e41da61c86e0fd987a90ee2d47842a9f21",
        "model_id": MODEL, "model_revision": MODEL_COMMIT,
        "model_gguf_sha256": MODEL_SHA256,
        "runtime_repository": RUNTIME, "runtime_tag": RUNTIME_TAG,
        "runtime_release_sha256": RUNTIME_SHA256,
        "max_new_asr_calls": MAX_AUDIO,
        "human_gold": False, "training_admitted": False,
        "three_asr_completed": False,
    }
    need(value == expected, "PARAKEET_FIXED_RESEARCH_REQUEST_DRIFT")
    return expected


def source_inputs(parent: Path):
    root, plan, count = sourcefix.original_audio_root(parent)
    need(count == MAX_AUDIO and plan["asr_calls"] == 0 and
         plan["training_admitted"] is False, "PARAKEET_SOURCE_ROLES_DRIFT")
    cells = sourcefix.core.verified_source(root)[1]
    need(len(cells) == MAX_AUDIO, "PARAKEET_SOURCE_COUNT_DRIFT")
    return plan, cells


def fetch_bounded(url: str, target: Path, max_bytes: int, expected_sha: str,
                  expected_length: int | None = None):
    need(target.parent.is_dir() and not target.exists() and
         not target.parent.is_symlink(), "DOWNLOAD_DESTINATION_NOT_NEW")
    hasher = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent":"research-kws-pinned/1"}),
        timeout=120,
    ) as stream, target.open("xb") as out:
        for block in iter(lambda: stream.read(1024*1024), b""):
            count += len(block)
            need(count <= max_bytes, "PINNED_DOWNLOAD_OVERSIZE")
            hasher.update(block)
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    need(count > 1024**2 and
         (expected_length is None or count == expected_length) and
         hasher.hexdigest() == expected_sha, "PINNED_DOWNLOAD_SHA_OR_SIZE")
    return {"bytes": count, "sha256": hasher.hexdigest()}


def verify_runtime(root: Path) -> tuple[Path, dict]:
    """Extract a checksum-pinned CPU release without following links or unsafe paths."""
    need(root.is_dir(), "RUNTIME_DIR_MISSING")
    archive = root / RUNTIME_FILE
    row = fetch_bounded(RUNTIME_URL, archive, 25*1024**2,
                        RUNTIME_SHA256, RUNTIME_BYTES)
    extracted = root / "extract"
    extracted.mkdir(mode=0o700)
    names = set()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        need(1 <= len(members) <= 1024, "RUNTIME_ARCHIVE_MEMBER_BUDGET")
        for member in members:
            rel = PurePosixPath(member.name)
            # Tar archives may contain a harmless leading "./" root member.
            if member.isdir() and not rel.parts:
                continue
            key = rel.as_posix()
            need(not rel.is_absolute() and rel.parts and
                 all(x not in (".", "..", "") for x in rel.parts) and
                 key not in names and
                 (member.isdir() or member.isfile()),
                 "RUNTIME_ARCHIVE_UNSAFE_MEMBER:" +
                 repr((member.name[:100],str(member.type)[:20],
                       member.linkname[:100])))
            names.add(key)
            location = extracted.joinpath(*rel.parts)
            if member.isdir():
                location.mkdir(parents=True, exist_ok=True)
                continue
            need(0 <= member.size <= 25*1024**2,
                 "RUNTIME_MEMBER_SIZE_BUDGET")
            location.parent.mkdir(parents=True, exist_ok=True)
            handle = tar.extractfile(member)
            need(handle is not None, "RUNTIME_ARCHIVE_FILE_MISSING")
            with handle, location.open("xb") as out:
                copied = 0
                for piece in iter(lambda: handle.read(1024*1024), b""):
                    copied += len(piece)
                    need(copied <= member.size, "RUNTIME_MEMBER_OVERSIZE")
                    out.write(piece)
            need(copied == member.size, "RUNTIME_MEMBER_INCOMPLETE")
    candidates = [p for p in extracted.rglob("nemo-speech")
                  if p.is_file() and not p.is_symlink()]
    need(len(candidates) == 1, "NEMO_SPEECH_BINARY_MISSING_OR_AMBIGUOUS")
    executable = candidates[0]
    executable.chmod(0o700)
    need(stat.S_ISREG(executable.stat().st_mode) and
         executable.stat().st_size > 1024, "NEMO_SPEECH_BINARY_INCOMPLETE")
    return executable, row


def verify_gguf(root: Path) -> tuple[Path, dict]:
    from huggingface_hub import hf_hub_download
    need(root.is_dir() and
         shutil.disk_usage(root).free >= MAX_MODEL_BYTES + 1024**3,
         "GGUF_DISK_BUDGET")
    # Materialize a fresh regular file rather than trusting the HF cache alias.
    cached = hf_hub_download(repo_id=MODEL, filename=MODEL_FILE,
                            revision=MODEL_COMMIT, token=False)
    model_path = root / MODEL_FILE
    need(not model_path.exists(), "GGUF_ALREADY_EXISTS")
    cached = Path(cached)
    need(cached.is_file(), "PINNED_GGUF_CACHE_NOT_FILE")
    # HF's cache may point to a content-addressed blob via a symlink.
    # Stream into our own create-once regular file and trust only verified bytes.
    count = 0
    digest = hashlib.sha256()
    with cached.open("rb") as src, model_path.open("xb") as dst:
        for piece in iter(lambda: src.read(1024*1024), b""):
            count += len(piece)
            need(count <= MAX_MODEL_BYTES, "PINNED_GGUF_COPY_OVERSIZE")
            digest.update(piece)
            dst.write(piece)
        dst.flush()
        os.fsync(dst.fileno())
    need(count > 100*1024**2 and digest.hexdigest() == MODEL_SHA256,
         "PINNED_GGUF_CHECKSUM_CHANGED")
    actual_size, actual_sha = english.media()._safe_file(
        model_path, limit=MAX_MODEL_BYTES)
    need(actual_size == count and actual_sha == MODEL_SHA256,
         "PINNED_GGUF_LOCAL_COPY_CHANGED")
    return model_path, {"bytes":actual_size,"sha256":actual_sha}


def run(parent: Path, output: Path, *, executor=subprocess.run) -> dict:
    claim = request()
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "PARAKEET_ONCE_MAIN_ONLY")
    blind.private_path(output, "PARAKEET_RESULT_MUST_BE_PRIVATE")
    need(output.parent.is_dir() and not output.exists(), "PARAKEET_OUTPUT_CREATE_ONCE")
    plan, items = source_inputs(parent)
    output.mkdir(mode=0o700)
    english.media()._once(output / "started.json", {
        "schema": POLICY, "model": MODEL, "model_revision": MODEL_COMMIT,
        "gguf_sha256": MODEL_SHA256, "runtime_release_sha256": RUNTIME_SHA256,
        "source_audio_artifact_sha256": claim["original_audio_artifact_sha256"],
        "source_plan_sha256": sha(market.canonical(plan)),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "training_admitted": False, "no_retry": True})
    model_root = output / "private-models"
    model_root.mkdir(mode=0o700)
    observations = []
    attempts = 0
    fatal = None
    runtime_receipt = None
    model_receipt = None
    try:
        binary, runtime_receipt = verify_runtime(model_root)
        model, model_receipt = verify_gguf(model_root)
        env = {"PATH":"/usr/bin:/bin", "HOME":str(model_root),
               "LANG":"C.UTF-8", "HF_HUB_OFFLINE":"1",
               "TRANSFORMERS_OFFLINE":"1",
               "NEMO_SPEECH_MODEL_DIR":str(model_root), "OMP_NUM_THREADS":"4"}
        for cell, wav, pcm in items:
            folder = output / "cells" / cell["cell_id"]
            need(not folder.exists(), "PARAKEET_CELL_ALREADY_CONSUMED")
            folder.mkdir(parents=True, mode=0o700)
            english.media()._once(folder / "started.json", {
                "schema": POLICY, "cell_id":cell["cell_id"],
                "wav_sha256":pcm["wav_sha256"],"provider_id":PROVIDER,
                "model_revision":MODEL_COMMIT,
                "retry_count":0,"training_admitted":False})
            attempts += 1
            status = "FAILED_NO_RETRY"
            try:
                result = executor([str(binary), "transcribe", str(wav),
                                   "--model",str(model),"--device","cpu"],
                                  env=env, cwd=str(folder), timeout=180,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  check=False, text=True)
                need(result.returncode == 0, "PARAKEET_NATIVE_PROCESS_FAILED")
                text = result.stdout.strip()
                need(type(text) is str and 0 < len(text) <= 512 and
                     not any(ord(c) < 32 for c in text),
                     "PARAKEET_TRANSCRIPT_INVALID")
                native = {"schema":blind.RAW_SCHEMA,"provider_id":PROVIDER,
                          "model_id":MODEL,"model_revision_sha":MODEL_COMMIT,
                          "audio_wav_sha256":pcm["wav_sha256"],
                          "processed_start_frame":0,
                          "processed_end_frame":pcm["frames"],
                          "decoder_completed":True,"actual_text":text}
                english.media()._once(folder/"native.json",native)
                need(blind.native_result(folder/"native.json",{
                    "provider_id":PROVIDER,"model_id":MODEL,
                    "model_revision_sha":MODEL_COMMIT
                },cell,pcm)==native,"PARAKEET_RESULT_NATIVE_BINDING")
                english.media()._once(folder/"completed.json",{
                    "schema":POLICY,"cell_id":cell["cell_id"],
                    "provider_id":PROVIDER,"status":"BLIND_ASR_UNADMITTED",
                    "wav_sha256":pcm["wav_sha256"],
                    "native_sha256":sha((folder/"native.json").read_bytes()),
                    "training_admitted":False,"human_gold":False})
                status = "BLIND_ASR_UNADMITTED"
            except Exception as exc:
                english.media()._once(folder/"failed.json",{
                    "schema":POLICY,"cell_id":cell["cell_id"],
                    "status":"FAILED_NO_RETRY",
                    "error_type":type(exc).__name__,
                    "training_admitted":False})
            observations.append({"cell_id":cell["cell_id"],
                                 "wav_sha256":pcm["wav_sha256"],"status":status})
    except BaseException as exc:
        fatal = type(exc).__name__
    finally:
        successful = sum(x["status"]=="BLIND_ASR_UNADMITTED" for x in observations)
        summary = {"schema":POLICY,
                   "status":"COMPLETE" if successful==MAX_AUDIO and fatal is None
                            else "INCOMPLETE_NO_RETRY",
                   "model_id":MODEL,"model_revision_sha":MODEL_COMMIT,
                   "model_weight_sha256":MODEL_SHA256,
                   "runtime_release_sha256":RUNTIME_SHA256,
                   "runtime_receipt":runtime_receipt,
                   "gguf_receipt":model_receipt,
                   "input_audio_count":MAX_AUDIO,"asr_attempts":attempts,
                   "completed_asr":successful,"error_type":fatal,
                   "source_plan_sha256":sha(market.canonical(plan)),
                   "source_tts_run_id":claim["original_tts_run_id"],
                   "rows":observations,"training_admitted":False,
                   "human_gold":False,"three_asr_completed":False}
        english.media()._once(output/"summary.json",summary)
        if model_root.exists():
            shutil.rmtree(model_root)
    return summary


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--artifact-parent",type=Path)
    cli.add_argument("--output",type=Path)
    cli.add_argument("--verify-source",action="store_true")
    cli.add_argument("--execute",action="store_true")
    args=cli.parse_args(argv)
    try:
        request()
        need(args.verify_source!=args.execute and args.artifact_parent is not None,
             "PARAKEET_EXPLICIT_MODE_REQUIRED")
        if args.verify_source:
            plan,rows=source_inputs(args.artifact_parent)
            print(json.dumps({"schema":POLICY,"status":"SOURCE_VERIFIED_ONLY",
                "input_audio_count":len(rows),"asr_calls":0,
                "source_plan_sha256":sha(market.canonical(plan)),
                "training_admitted":False},sort_keys=True))
            return 0
        need(args.output is not None,"PARAKEET_OUTPUT_REQUIRED")
        result=run(args.artifact_parent,args.output)
        print(json.dumps({k:result[k] for k in ("status","asr_attempts",
            "completed_asr","training_admitted")},sort_keys=True))
        return 0 if result["status"]=="COMPLETE" else 2
    except (OSError,ValueError) as exc:
        print("PARAKEET_REAL_STOP:"+type(exc).__name__,file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
