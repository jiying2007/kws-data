"""One-shot independent phoneme-controlled Kokoro English wake research pilot.

Two fresh Apache-2.0 Kokoro voices, not the previous nine Qwen synthetic
source families. Explicit IPA/phoneme prompts are INPUT HYPOTHESES only.
Only real generated WAV+hash receipts are evidence; no training admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import english_qwen_capture as media
import multilingual_campaign as market

POLICY = "english-kokoro-controlled-ipa-pilot-20261112-v1"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-kokoro-ipa-pilot-20261112.json"
MODEL_ID = "hexgrad/Kokoro-82M"
MODEL_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MODEL_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
MODEL_FILE = "kokoro-v1_0.pth"
VOICES = (
    ("af_heart", "a", "train-candidate", "0ab5709b"),
    ("bf_emma", "b", "development-only", "d0a423de"),
)
# /O/ is Kokoro's English GOAT /oʊ/ token in its official phoneme vocabulary.
# /ɛ/ and /ɹ/ fix the desired second phrase (MEH-roh); no automatic G2P.
# Include short and sentence-carried positives; upstream warns ultra-short TTS
# may be unstable. These are phonetic experimental assumptions, not gold labels.
PROMPTS = (
    ("Hey Vosen", "hˈeɪ vˈOsən.", "wake", 1, "isolated"),
    ("Hey Vosen, can you hear me?", "hˈeɪ vˈOsən, kən ju hˈiɹ mi?", "wake", 1, "carrier"),
    ("Hey Mero", "hˈeɪ mˈɛɹO.", "wake", 2, "isolated"),
    ("Hey Mero, can you hear me?", "hˈeɪ mˈɛɹO, kən ju hˈiɹ mi?", "wake", 2, "carrier"),
    ("Hey Wosen", "hˈeɪ wˈOsən.", "near_miss", None, "other_confusable"),
    ("Hey Miro", "hˈeɪ mˈiɹO.", "near_miss", None, "other_confusable"),
    ("What time is it?", "wˈʌt tˈaɪm ɪz ɪt?", "ordinary_negative", None, "conversation"),
)
MAX_CALLS = len(VOICES) * len(PROMPTS)
MAX_MODEL_BYTES = 600 * 1024**2


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def claim() -> dict:
    expected = {
        "schema": POLICY, "model_id": MODEL_ID,
        "model_revision": MODEL_REV, "model_sha256": MODEL_SHA256,
        "voices": [v[0] for v in VOICES],
        "wake_phrases": ["Hey Vosen", "Hey Mero"],
        "max_calls": MAX_CALLS, "previous_qwen_audio_regenerated": 0,
        "human_gold": False, "training_admitted": False, "asr_calls": 0,
    }
    need(market.read(TRIGGER) == expected, "KOKORO_FIXED_CLAIM_DRIFT")
    need(MAX_CALLS == 14 and {v[0] for v in VOICES} == {"af_heart","bf_emma"},
         "KOKORO_RESEARCH_BUDGET_DRIFT")
    return expected


def plan(asset_lock_sha: str) -> dict:
    need(type(asset_lock_sha) is str and
         market.HEX64.fullmatch(asset_lock_sha) is not None,
         "PINNED_KOKORO_MODEL_LOCK_REQUIRED")
    need(tuple(market.normalize(x, "en-US") for x in ("Hey Vosen","Hey Mero")) ==
         market.MARKET_ENGLISH_CANDIDATES, "KOKORO_MARKET_ROSTER_DRIFT")
    cells = []
    for voice, accent, role, _ in VOICES:
        for prompt, phonemes, intent, keyword, variant in PROMPTS:
            label={"policy":POLICY,"voice":voice,"phonemes":phonemes,"revision":MODEL_REV}
            cells.append({
                "cell_id":"kko-"+sha(market.canonical(label))[:24],
                "voice_id":voice, "source_family_id":"kokoro-v1-"+voice,
                "role":role,"accent":accent, "requested_text":prompt,
                "phonemes_requested":phonemes,"intent":intent,
                "requested_keyword_index":keyword, "variant":variant,
                "human_gold":False,"training_admitted":False,
            })
    need(len(cells)==MAX_CALLS and
         len({x["cell_id"] for x in cells})==MAX_CALLS,
         "KOKORO_CELL_IDENTITY")
    return {
        "schema":POLICY, "locale":"en-US", "tts_model_id":MODEL_ID,
        "tts_model_revision":MODEL_REV, "model_assets_lock_sha256":asset_lock_sha,
        "wake_phrases_requested":["Hey Vosen","Hey Mero"],
        "voices":[{"id":v,"accent":a,"role":r,"source_family_id":"kokoro-v1-"+v}
                  for v,a,r,_ in VOICES],
        "cells":cells,"human_gold":False,"training_admitted":False,
        "qualification_allowed":False,"asr_calls":0,
    }


def frozen_assets(root: Path) -> dict:
    """One exact upstream model commit, four precise files, local hash readback."""
    from huggingface_hub import HfApi, snapshot_download
    info=HfApi().model_info(MODEL_ID, revision=MODEL_REV,
                            files_metadata=True, token=False)
    need(info.sha==MODEL_REV and info.private is False and
         not getattr(info,"gated",False), "KOKORO_SOURCE_REVISION")
    files=[MODEL_FILE,"config.json"] + ["voices/"+v+".pt" for v,_,_,_ in VOICES]
    siblings={x.rfilename:x.size for x in (info.siblings or [])}
    need(all(type(siblings.get(f)) is int and 0 < siblings[f] <= MAX_MODEL_BYTES
             for f in files) and sum(siblings[f] for f in files)<=MAX_MODEL_BYTES,
         "KOKORO_DOWNLOAD_BUDGET")
    need(shutil.disk_usage(root).free >= sum(siblings[f] for f in files)+1024**3,
         "KOKORO_DISK_RESERVE")
    folder=root/"upstream"
    need(not folder.exists(),"KOKORO_MODEL_ROOT_MUST_BE_NEW")
    folder.mkdir(mode=0o700)
    result=snapshot_download(repo_id=MODEL_ID,revision=MODEL_REV,
                             allow_patterns=files,local_dir=str(folder),token=False)
    need(Path(result).resolve()==folder.resolve(),"KOKORO_DOWNLOAD_ROOT_DRIFT")
    cache=folder/".cache"
    if cache.exists():
        need(cache.is_dir() and not cache.is_symlink(),"KOKORO_CACHE_NOT_DIRECTORY")
        shutil.rmtree(cache)
    actual=set()
    for f in folder.rglob("*"):
        need(not f.is_symlink(),"KOKORO_MODEL_ALIAS")
        if f.is_file():actual.add(f.relative_to(folder).as_posix())
    need(actual==set(files),"KOKORO_MODEL_INVENTORY_DRIFT")
    rows=[]
    for name in sorted(files):
        n,digest=media.media()._safe_file(folder/name,limit=MAX_MODEL_BYTES)
        need(n==siblings[name],"KOKORO_MODEL_FILE_BYTES_DRIFT")
        if name==MODEL_FILE:
            need(digest==MODEL_SHA256,"KOKORO_OFFICIAL_MODEL_SHA_DRIFT")
        for v,_,_,prefix in VOICES:
            if name=="voices/"+v+".pt":
                need(digest.startswith(prefix),"KOKORO_OFFICIAL_VOICE_SHA_DRIFT")
        rows.append({"path":name,"bytes":n,"sha256":digest})
    return {"schema":"english-kokoro-model-lock-v1","repo_id":MODEL_ID,
            "revision":MODEL_REV,"rows":rows}


class KokoroPhoneticBackend:
    """Heavy imports occur only after exact model/voice hashes were verified."""
    def __init__(self,root:Path):
        import torch
        from kokoro import KModel,KPipeline
        torch.set_num_threads(min(os.cpu_count() or 1,4))
        self.model=KModel(repo_id=MODEL_ID,config=str(root/"config.json"),
                          model=str(root/MODEL_FILE)).to("cpu").eval()
        self.pipes={}
        for v,lang,_,_ in VOICES:
            pipe=KPipeline(lang_code=lang,repo_id=MODEL_ID,model=False)
            pack=torch.load(root/"voices"/(v+".pt"),
                            weights_only=True,map_location="cpu")
            need(isinstance(pack,torch.Tensor) and pack.isfinite().all(),
                 "KOKORO_VOICEPACK_INVALID")
            pipe.voices[v]=pack
            self.pipes[v]=pipe

    def synthesize(self,cell:dict):
        import torch
        voice=cell["voice_id"]
        pipe=self.pipes[voice]
        allowed=set(self.model.vocab)
        ps=cell["phonemes_requested"]
        need(all(ch in allowed for ch in ps),"KOKORO_IPA_OUTSIDE_PINNED_VOCAB")
        with torch.inference_mode():
            results=list(pipe.generate_from_tokens(ps,voice=voice,speed=1.0,
                                                   model=self.model))
        need(len(results)==1 and results[0].audio is not None and
             results[0].phonemes==ps,"KOKORO_RAW_PHONEME_NOT_CONSUMED")
        return results[0].audio.detach().float().cpu().numpy(),24000


def capture(plan_data:dict, root:Path,backend,*, writer=None,limit:int=MAX_CALLS):
    """Private immutable source WAVs; one attempt per cell, even on failure."""
    need(type(limit) is int and 0<=limit<=MAX_CALLS,"KOKORO_CALL_BUDGET")
    need(root.is_absolute() and root.parent.is_dir() and
         root!=market.ROOT and market.ROOT not in root.parents and
         not any(p.is_symlink() for p in (root,*root.parents)),
         "KOKORO_PRIVATE_RESULT_ROOT")
    expected={"schema":POLICY,"plan_sha256":sha(market.canonical(plan_data)),
              "training_admitted":False}
    if not root.exists():
        root.mkdir(mode=0o700)
        media.media()._once(root/"plan-lock.json",expected)
    else:
        need(market.read(root/"plan-lock.json")==expected,"KOKORO_RESUME_DRIFT")
    if writer is None:writer=media.media()._write_waves
    attempts=0
    for cell in plan_data["cells"]:
        dest=root/"cells"/cell["cell_id"]
        if (dest/"started.json").exists():continue
        if attempts>=limit:break
        need(not dest.exists(),"KOKORO_FOREIGN_ATTEMPT")
        dest.mkdir(mode=0o700,parents=True)
        media.media()._once(dest/"started.json",{
            "schema":POLICY,"cell_id":cell["cell_id"],"plan_sha256":expected["plan_sha256"],
            "source_family_id":cell["source_family_id"],
            "retry_count":0,"training_admitted":False})
        attempts+=1
        try:
            audio,rate=backend.synthesize(cell)
            technical=media.media().quality(audio,rate)
            files=writer(dest,audio,rate,technical)
            need(type(files) is dict and
                 all(type(files.get(k)) is str and
                     market.HEX64.fullmatch(files[k]) is not None
                     for k in ("wav_sha256","pcm_sha256","native_sha256")) and
                 files.get("sample_rate_hz")==16000 and
                 type(files.get("audio_frames")) is int and
                 files["audio_frames"]>0 and type(files.get("wav_bytes")) is int,
                 "KOKORO_WAV_RECEIPT_INVALID")
            media.media()._once(dest/"completed.json",{
                "schema":POLICY,"cell_id":cell["cell_id"],"status":"GENERATED_UNADMITTED",
                "plan_sha256":expected["plan_sha256"],"technical":technical,"audio":files,
                "human_gold":False,"training_admitted":False,"qualification_allowed":False})
        except Exception as exc:
            media.media()._once(dest/"failed.json",{
                "schema":POLICY,"cell_id":cell["cell_id"],"status":"FAILED_NO_RETRY",
                "error_type":type(exc).__name__,"training_admitted":False})
    return {"attempts_this_call":attempts,
            "total_started":sum((root/"cells"/x["cell_id"]/"started.json").exists()
                                for x in plan_data["cells"]),
            "total_completed":sum((root/"cells"/x["cell_id"]/"completed.json").exists()
                                  for x in plan_data["cells"]),
            "training_admitted":False}


def run(root:Path)->dict:
    claim()
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_EVENT_NAME")=="push" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1",
         "KOKORO_ONCE_PROTECTED_MAIN_ONLY")
    need(root.is_absolute() and not root.exists() and root.parent.is_dir(),
         "KOKORO_NEW_PRIVATE_OUTPUT")
    root.mkdir(mode=0o700)
    media.media()._once(root/"started.json",{
        "schema":POLICY,"head_sha":os.environ.get("GITHUB_SHA"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "trigger_sha256":sha(TRIGGER.read_bytes()),
        "retry_count":0,"training_admitted":False})
    locked=root/"locked-model";locked.mkdir(mode=0o700)
    complete=None
    try:
        lock=frozen_assets(locked)
        media.media()._once(root/"asset-lock.json",lock)
        p=plan(sha((root/"asset-lock.json").read_bytes()))
        media.media()._once(root/"source-plan.json",p)
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        result=capture(p,root/"capture",KokoroPhoneticBackend(locked/"upstream"))
        rows=[]
        for cell in p["cells"]:
            folder=root/"capture"/"cells"/cell["cell_id"]
            done=folder/"completed.json"
            status="GENERATED_UNADMITTED" if done.is_file() else (
                "FAILED_NO_RETRY" if (folder/"failed.json").is_file() else "NOT_RUN")
            row={"cell_id":cell["cell_id"],"status":status,
                 "source_family_id":cell["source_family_id"],
                 "training_admitted":False}
            if done.is_file():
                receipt=market.read(done)
                row.update(wav_sha256=receipt["audio"]["wav_sha256"],
                           pcm_sha256=receipt["audio"]["pcm_sha256"])
            rows.append(row)
        ok=sum(r["status"]=="GENERATED_UNADMITTED" for r in rows)
        complete={"schema":POLICY,"status":"COMPLETE" if ok==MAX_CALLS else "INCOMPLETE_NO_RETRY",
                  "model_id":MODEL_ID,"model_revision":MODEL_REV,
                  "asset_lock_sha256":sha((root/"asset-lock.json").read_bytes()),
                  "planned_audio_count":MAX_CALLS,"generated_audio_count":ok,
                  "attempted_audio_count":result["total_started"],"source_rows":rows,
                  "asr_calls":0,"training_admitted":False,"human_gold":False}
    except BaseException as exc:
        complete={"schema":POLICY,"status":"FAILED_NO_RETRY",
                  "error_type":type(exc).__name__,"asr_calls":0,
                  "training_admitted":False,"human_gold":False}
        media.media()._once(root/"failed.json",complete)
        raise
    finally:
        if complete is not None:media.media()._once(root/"summary.json",complete)
        if locked.exists():shutil.rmtree(locked)
    return complete


def main(argv=None)->int:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path)
    cli.add_argument("--execute",action="store_true")
    args=cli.parse_args(argv)
    try:
        claim()
        if not args.execute:
            print(json.dumps({"schema":POLICY,"planned_audio_count":MAX_CALLS,
                              "model_calls":0,"training_admitted":False}))
            return 0
        need(args.output is not None,"KOKORO_EXPLICIT_RESULT_REQUIRED")
        result=run(args.output)
        print(json.dumps({"status":result["status"],
                          "generated_audio_count":result.get("generated_audio_count",0),
                          "asr_calls":0,"training_admitted":False},sort_keys=True))
        return 0 if result["status"]=="COMPLETE" else 2
    except (ValueError,OSError) as exc:
        print("KOKORO_PILOT_STOP:"+type(exc).__name__,file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
