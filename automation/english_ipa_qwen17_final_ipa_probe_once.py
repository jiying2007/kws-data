"""One-shot pronunciation-controlled Qwen 1.7B English KWS probe; no training/admission."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_qwen_capture as capture
import multilingual_campaign as market

POLICY="english-ipa-qwen17-final-ipa-probe-v1"
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-qwen17-final-ipa-20261116.json"
MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
REV="0c0e3051f131929182e2c023b9537f8b1c68adfe"
MAX_MODEL_BYTES=6*1024**3
MAX_DISK_RESERVE=1024**3

def sha(b): return hashlib.sha256(b).hexdigest()
def need(ok,msg):
    if not ok: raise ValueError(msg)
def read_trigger():
    x=market.read(TRIGGER)
    expected = [
      {"phrase":"Hey Mero","ipa":"/heɪ ˈmiːroʊ/","speaker":"aiden",
       "instruction":"Pronounce the name Mero exactly as /miːroʊ/, with a long stressed /iː/ vowel in its first syllable, not /ɪ/. Keep the phrase natural."},
      {"phrase":"Hey Mero","ipa":"/heɪ ˈmiːroʊ/","speaker":"ryan",
       "instruction":"Pronounce the name Mero exactly as /miːroʊ/, with a long stressed /iː/ vowel in its first syllable, not /ɪ/. Keep the phrase natural."},
      {"phrase":"Hey Vosen","ipa":"/heɪ ˈvoʊzən/","speaker":"aiden",
       "instruction":"Pronounce the name Vosen exactly as /voʊzən/, with a voiced /z/ in the middle, not an unvoiced /s/. Keep the phrase natural."},
      {"phrase":"Hey Vosen","ipa":"/heɪ ˈvoʊzən/","speaker":"ryan",
       "instruction":"Pronounce the name Vosen exactly as /voʊzən/, with a voiced /z/ in the middle, not an unvoiced /s/. Keep the phrase natural."}
    ]
    need(x.get("schema")==POLICY and x.get("model_id")==MODEL and
         x.get("model_revision")==REV and x.get("model_license")=="Apache-2.0" and
         x.get("voices")==["aiden","ryan"] and x.get("cells")==expected and
         x.get("max_calls")==4 and x.get("asr_calls")==0 and
         x.get("training_admitted") is False and x.get("human_gold") is False and
         x.get("qualification_allowed") is False and x.get("no_retry") is True,
         "TRIGGER_IDENTITY_OR_PRONUNCIATION_DRIFT")
    return x

def materialize(root):
    from huggingface_hub import HfApi, snapshot_download
    info=HfApi().model_info(repo_id=MODEL,revision=REV,files_metadata=True,token=False)
    need(info.sha==REV and info.private is False and not getattr(info,"gated",False),
         "UPSTREAM_MODEL_REVISION_OR_ACCESS")
    sizes=[getattr(f,"size",None) for f in (info.siblings or [])]
    need(sizes and all(type(n) is int and n>=0 for n in sizes) and
         sum(sizes)<=MAX_MODEL_BYTES, "MODEL_SIZE_BUDGET")
    need(shutil.disk_usage(root).free>=sum(sizes)+MAX_DISK_RESERVE,
         "MODEL_DISK_BUDGET")
    folder=root/"model"
    need(not folder.exists(),"MODEL_DOWNLOAD_ONCE")
    folder.mkdir(mode=0o700)
    path=snapshot_download(repo_id=MODEL,revision=REV,local_dir=str(folder),token=False)
    need(Path(path).resolve()==folder.resolve(),"MODEL_PATH")
    cache=folder/".cache"
    if cache.exists():
        need(cache.is_dir() and not cache.is_symlink(),"MODEL_CACHE")
        shutil.rmtree(cache)
    rows=[]
    for f in sorted(folder.rglob("*")):
        need(not f.is_symlink(),"MODEL_SYMLINK")
        if f.is_file() and ".cache" not in f.relative_to(folder).parts:
            n,h=capture.media()._safe_file(f,limit=MAX_MODEL_BYTES)
            rows.append({"path":f.relative_to(folder).as_posix(),"bytes":n,"sha256":h})
    need(rows and sum(x["bytes"] for x in rows)<=MAX_MODEL_BYTES,"MODEL_SNAPSHOT")
    return folder,{"schema":"english-ipa-qwen17-model-lock-v1","model_id":MODEL,
                   "revision":REV,"license":"Apache-2.0","files":rows}

class Backend:
    def __init__(self,folder):
        self.folder=folder
        self.model=None
    def synthesize(self,cell):
        import torch
        from qwen_tts import Qwen3TTSModel
        if self.model is None:
            torch.set_num_threads(min(4,os.cpu_count() or 1))
            self.model=Qwen3TTSModel.from_pretrained(
                str(self.folder),device_map="cpu",dtype=torch.bfloat16,
                attn_implementation="sdpa")
        torch.manual_seed(int(sha(cell["cell_id"].encode())[:12],16))
        wavs,rate=self.model.generate_custom_voice(
            text=cell["phrase"],language="English",speaker=cell["speaker"].title(),
            instruct=cell["instruction"],do_sample=False)
        need(len(wavs)==1,"ONE_WAV_PER_CELL")
        return wavs[0],int(rate)

def execute(out):
    trigger=read_trigger()
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","MAIN_ATTEMPT_ONE_ONLY")
    need(out.is_absolute() and out.parent.is_dir() and not out.exists() and
         not any(p.is_symlink() for p in (out,out.parent,*out.parent.parents)),
         "CREATE_ONLY_PRIVATE_OUTPUT")
    out.mkdir(mode=0o700)
    runner=capture.media()
    runner._once(out/"started.json",{"schema":POLICY,"status":"started",
      "source_sha":os.environ.get("GITHUB_SHA"),"run_id":os.environ.get("GITHUB_RUN_ID"),
      "trigger_sha256":sha(TRIGGER.read_bytes()),"max_calls":4,"no_retry":True,
      "training_admitted":False,"human_gold":False})
    result={"schema":POLICY,"status":"FAILED_NO_RETRY","attempts":0,"completed":0}
    try:
        folder,lock=materialize(out)
        runner._once(out/"model-lock.json",lock)
        model_lock_sha=sha((out/"model-lock.json").read_bytes())
        plan=[]
        for i,c in enumerate(trigger["cells"]):
            row=dict(c)
            row["cell_id"]="enipa-"+sha(json.dumps({"i":i,"cell":c,"revision":REV},
                                      sort_keys=True,separators=(",",":")).encode())[:24]
            row["model_lock_sha256"]=model_lock_sha
            plan.append(row)
        runner._once(out/"source-plan.json",{"schema":POLICY,"model":MODEL,
          "revision":REV,"model_lock_sha256":model_lock_sha,"cells":plan,
          "training_admitted":False,"human_gold":False,"qualification_allowed":False})
        backend=Backend(folder)
        rows=[]
        for cell in plan:
            d=out/"cells"/cell["cell_id"]
            d.mkdir(parents=True,mode=0o700)
            runner._once(d/"started.json",{"schema":POLICY,"cell_id":cell["cell_id"],
              "attempts":1,"retry_count":0,"training_admitted":False})
            result["attempts"]+=1
            try:
                samples,rate=backend.synthesize(cell)
                metrics=runner.quality(samples,rate)
                audio=runner._write_waves(d,samples,rate,metrics)
                runner._once(d/"completed.json",{"schema":POLICY,"cell_id":cell["cell_id"],
                  "phrase":cell["phrase"],"ipa_target":cell["ipa"],
                  "instruction":cell["instruction"],"speaker":cell["speaker"],
                  "audio":audio,"technical":metrics,"pronunciation_verified":False,
                  "human_gold":False,"training_admitted":False})
                rows.append({"cell_id":cell["cell_id"],"status":"GENERATED_UNADMITTED",
                             "wav_sha256":audio["wav_sha256"],"pcm_sha256":audio["pcm_sha256"]})
                result["completed"]+=1
            except Exception as e:
                runner._once(d/"failed.json",{"schema":POLICY,"cell_id":cell["cell_id"],
                  "status":"FAILED_NO_RETRY","error_type":type(e).__name__,
                  "training_admitted":False})
                rows.append({"cell_id":cell["cell_id"],"status":"FAILED_NO_RETRY"})
        result.update({"status":"COMPLETE" if result["completed"]==4 else "INCOMPLETE_NO_RETRY",
          "rows":rows,"model_lock_sha256":model_lock_sha,"qwen_tts_version":version("qwen-tts"),
          "asr_calls":0,"training_admitted":False,"human_gold":False})
    except BaseException as e:
        result.update({"error_type":type(e).__name__,"training_admitted":False})
    runner._once(out/"result.json",result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args(argv)
    if not a.execute:
        read_trigger()
        print("PLAN_VALID_NO_MODEL_OR_AUDIO_EXECUTION")
        return 0
    execute(a.output)
    return 0
if __name__=="__main__":
    raise SystemExit(main())
