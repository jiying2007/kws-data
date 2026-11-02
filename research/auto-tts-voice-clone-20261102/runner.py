"""Run a strictly bounded offline Qwen3-TTS campaign, including synthetic-only clone.

Heavy libraries are loaded only in the explicit execute lane. A consumed cell
is never retried by resume, even after interruption. This is NOT label admission.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import wave

import campaign as c

MODEL_LOCK_SCHEMA="qwen3-tts-offline-asset-lock-v1"
MAX_MODEL_FILES=250
MAX_MODEL_BYTES=8*1024**3
MAX_OUTPUT_SECONDS=12
MIN_OUTPUT_SECONDS=0.35


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _safe_file(path, *, limit, at_least=1):
    path=Path(path)
    c.require(not any(p.is_symlink() for p in (path,*path.parents)),
              "ALIAS_OR_SYMLINK")
    st=path.stat()
    c.require(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and
              at_least<=st.st_size<=limit,"FILE_TYPE_OR_SIZE")
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        bytes_read=0
        for block in iter(lambda:fh.read(1024*1024),b""):
            bytes_read+=len(block)
            c.require(bytes_read<=limit,"STREAM_BOUNDS")
            digest.update(block)
        end=os.fstat(fh.fileno())
    c.require(bytes_read==st.st_size and
              (st.st_dev,st.st_ino,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink)==
              (end.st_dev,end.st_ino,end.st_mtime_ns,end.st_ctime_ns,end.st_nlink),
              "MUTATED_INPUT")
    return bytes_read,digest.hexdigest()


def validate_models(plan,lock,asset_root):
    c.validate_plan(plan)
    c.require(type(lock) is dict and set(lock)=={"schema","models"} and
              lock["schema"]==MODEL_LOCK_SCHEMA,"MODEL_ASSET_LOCK_SCHEMA")
    rows=lock["models"]
    c.require(type(rows) is list and 1<=len(rows)<=3,"MODEL_LOCK_COUNT")
    assets={}
    declared={(v["model_id"],v["model_revision"]) for v in plan["voices"]}
    for row in rows:
        c.require(type(row) is dict and set(row)=={"model_id","revision","relative_dir","files"},
                  "MODEL_LOCK_ROW")
        mid,revision=row["model_id"],row["revision"]
        c.require((mid,revision) in declared and
                  (mid,revision) not in assets,"UNREQUESTED_OR_DUPLICATE_MODEL")
        rel=row["relative_dir"]
        c.require(type(rel) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_/-]{0,120}",rel)
                  and ".." not in Path(rel).parts,"MODEL_PATH")
        folder=Path(asset_root)/rel
        c.require(folder.is_dir() and not any(x.is_symlink() for x in
                  (folder,*folder.parents)),"MODEL_DIRECTORY_ALIAS")
        files=row["files"]
        c.require(type(files) is list and 1<=len(files)<=MAX_MODEL_FILES,
                  "MODEL_FILE_COUNT")
        names=set();total=0
        for item in files:
            c.require(type(item) is dict and
                      set(item)=={"path","bytes","sha256"},"MODEL_FILE_ROW")
            name=item["path"]
            c.require(type(name) is str and
                      re.fullmatch(r"[a-zA-Z0-9_.-][a-zA-Z0-9_./-]{0,180}",name) and
                      name not in (".","..") and not name.startswith(".cache/") and
                      ".." not in Path(name).parts and name not in names,
                      "MODEL_FILE_PATH")
            names.add(name)
            c.require(type(item["bytes"]) is int and 1<=item["bytes"]<=MAX_MODEL_BYTES,
                      "MODEL_MEMBER_BOUND")
            c.require(type(item["sha256"]) is str and c._SHA.fullmatch(item["sha256"]),
                      "MODEL_DIGEST")
            size,digest=_safe_file(folder/name,limit=MAX_MODEL_BYTES)
            c.require(size==item["bytes"] and digest==item["sha256"],
                      "MODEL_ASSET_MISMATCH")
            total+=size
            c.require(total<=MAX_MODEL_BYTES,"MODEL_TOTAL_BUDGET")
        # Reject additional regular files not represented by the exact lock.
        present=set()
        for f in folder.rglob("*"):
            c.require(not f.is_symlink(),"MODEL_EXTRA_ALIAS")
            if f.is_file():
                present.add(f.relative_to(folder).as_posix())
        c.require(present==names,"MODEL_EXTRA_OR_MISSING_FILES")
        assets[(mid,revision)]=folder
    c.require(set(assets)==declared,"MODEL_SET_INCOMPLETE")
    return assets


def validate_reference(ref,root):
    c.require(type(ref) is dict and
              ref["source_kind"]=="synthesized_and_machine_screened" and
              ref["rights_basis"]=="synthetic_model_output_research_only" and
              ref["real_person_identity"] is None and
              ref["training_admitted"] is False,
              "HUMAN_REFERENCE_NOT_ALLOWED")
    root=Path(root)
    audio=root/(ref["audio_id"]+".wav")
    screen=root/(ref["audio_id"]+".screen.json")
    size,sha=_safe_file(audio,limit=1024*1024)
    c.require(sha==ref["wav_sha256"],"REFERENCE_WAV_MISMATCH")
    _,screen_sha=_safe_file(screen,limit=65536)
    c.require(screen_sha==ref["screening_sha256"],"REFERENCE_SCREEN_MISMATCH")
    result=c.read_json(screen,65536)
    c.require(type(result) is dict and
              set(result)=={"schema","status","audio_id","wav_sha256","pcm_sha256",
                            "source_family_id","normalized_actual_text",
                            "engines","machine_only","training_admitted"} and
              result["schema"]=="synthetic-clone-reference-screen-v1" and
              result["status"]=="MACHINE_SCREENED_REFERENCE_ONLY" and
              result["audio_id"]==ref["audio_id"] and
              result["wav_sha256"]==ref["wav_sha256"] and
              result["pcm_sha256"]==ref["pcm_sha256"] and
              result["source_family_id"]==ref["source_family_id"] and
              result["normalized_actual_text"]==c.normalize(ref["reference_text"]) and
              result["engines"]==list(c.ENGINES) and
              result["machine_only"] is True and
              result["training_admitted"] is False,
              "REFERENCE_SCREEN_INCONSISTENT")
    with wave.open(str(audio),"rb") as stream:
        c.require((stream.getnchannels(),stream.getsampwidth(),stream.getframerate(),
                   stream.getnframes(),stream.getcomptype())==
                  (1,2,ref["sample_rate_hz"],ref["frames"],"NONE"),
                  "REFERENCE_WAV_GEOMETRY")
        data=stream.readframes(ref["frames"])
        c.require(not stream.readframes(1),"REFERENCE_EXTRA_PCM")
    c.require(_hash(data)==ref["pcm_sha256"] and len(data)==2*ref["frames"],
              "REFERENCE_PCM_MISMATCH")
    return data,ref["sample_rate_hz"]


def quality(samples,rate):
    c.require(type(rate) is int and rate==24000,"REQUIRE_NATIVE_24KHZ")
    c.require(hasattr(samples,"__len__") and
              int(MIN_OUTPUT_SECONDS*rate)<=len(samples)<=MAX_OUTPUT_SECONDS*rate,
              "NATIVE_WAVE_DURATION")
    peak=0.0
    square=0.0
    for v in samples:
        x=float(v)
        c.require(math.isfinite(x),"NONFINITE_WAVE")
        peak=max(peak,abs(x))
        square+=x*x
    rms=math.sqrt(square/len(samples))
    c.require(0.008<=rms<=0.6,"SILENT_OR_TOO_LOUD")
    c.require(peak<0.999,"CLIPPED_AUDIO")
    return {"native_frames":len(samples),"native_rate":rate,"duration_s":len(samples)/rate,
            "rms":rms,"peak":peak,"technical_audio_pass":True}


def _write_waves(path,output,rate,metrics):
    """Actual scientific conversion is lazy; CI mock never imports these libs."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    vec=np.asarray(output,dtype=np.float32)
    c.require(vec.ndim==1 and vec.shape[0]==metrics["native_frames"],"WAVE_SHAPE")
    down=resample_poly(vec,2,3)
    c.require(0<len(down)<=192000 and np.all(np.isfinite(down)) and
              float(np.max(np.abs(down)))<0.999,"RESAMPLE_CLIPPING_OR_NONFINITE")
    # No automatic gain, trimming or overrange clipping to create false positives.
    pcm=np.rint(down.astype(np.float64)*32767).astype("<i2").tobytes()
    native=path/"native-f32.wav"
    derived=path/"audio-16k.wav"
    sf.write(str(native),vec,rate,subtype="FLOAT")
    with wave.open(str(derived),"wb") as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
        writer.writeframes(pcm)
    nsize,nhash=_safe_file(native,limit=6*1024*1024)
    wsize,whash=_safe_file(derived,limit=2*1024*1024)
    return {"native_sha256":nhash,"native_bytes":nsize,
            "wav_sha256":whash,"wav_bytes":wsize,
            "pcm_sha256":_hash(pcm),"audio_frames":len(pcm)//2,
            "sample_rate_hz":16000}


def _once(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(c.canonical(record)+b"\n")
        stream.flush();os.fsync(stream.fileno())


class QwenOfflineBackend:
    """Qwen official APIs; strictly local pinned files; reuse clone prompt once."""
    def __init__(self,assets,reference_root,device="cpu"):
        c.require(device in ("cpu","cuda:0"),"DEVICE_SELECTION")
        self.assets=assets
        self.refs=Path(reference_root)
        self.device=device
        self.model=None
        self.current=None
        self.prompts={}

    def _load(self,cell):
        key=(cell["model_id"],cell["model_revision"])
        if self.current==key:return self.model
        if self.model is not None:
            del self.model
            self.model=None
            self.prompts.clear()  # prompt embeddings belong to a particular model revision
            gc.collect()
        import torch
        from qwen_tts import Qwen3TTSModel
        torch.set_num_threads(min(os.cpu_count() or 1,4))
        if self.device=="cuda:0":
            c.require(torch.cuda.is_available(),"NO_CUDA")
            dtype=torch.bfloat16
        else:
            dtype=torch.float32
        self.model=Qwen3TTSModel.from_pretrained(str(self.assets[key]),
            device_map=self.device,dtype=dtype,attn_implementation="sdpa")
        self.current=key
        return self.model

    def synthesize(self,cell,voice):
        import torch
        model=self._load(cell)
        torch.manual_seed(cell["seed"])
        if self.device=="cuda:0":torch.cuda.manual_seed_all(cell["seed"])
        if voice["mode"]=="preset":
            wavs,sr=model.generate_custom_voice(text=cell["text"],
                language="Chinese",speaker=voice["preset"])
        elif voice["mode"]=="design":
            wavs,sr=model.generate_voice_design(text=cell["text"],
                language="Chinese",instruct=voice["description"])
        else:
            ref=voice["reference"]
            cachekey=ref["wav_sha256"]
            if cachekey not in self.prompts:
                import numpy as np
                pcm,rate=validate_reference(ref,self.refs)
                arr=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
                self.prompts[cachekey]=model.create_voice_clone_prompt(
                    ref_audio=(arr,rate),ref_text=ref["reference_text"],
                    x_vector_only_mode=False)
            wavs,sr=model.generate_voice_clone(text=cell["text"],
                language="Chinese",voice_clone_prompt=self.prompts[cachekey])
        c.require(len(wavs)==1,"ONE_WAVE_PER_CELL")
        return wavs[0],int(sr)


def run_batch(plan,output_dir,backend,*,max_calls,writer=_write_waves):
    c.validate_plan(plan)
    c.require(type(max_calls) is int and 0<=max_calls<=min(len(plan["cells"]),c.MAX_CELLS),
              "CALL_BUDGET")
    root=Path(output_dir)
    expected_hash=_hash(c.canonical(plan))
    if not root.exists():
        root.mkdir(parents=True)
        _once(root/"plan-lock.json",{"schema":"tts-autonomous-plan-lock-v1",
            "canonical_sha256":expected_hash,"training_admitted":False})
    else:
        lock=c.read_json(root/"plan-lock.json")
        c.require(type(lock) is dict and lock.get("canonical_sha256")==expected_hash
                  and lock.get("training_admitted") is False,"RESUME_WRONG_PLAN")
    voices={v["voice_id"]:v for v in plan["voices"]}
    # Load one model at a time. A clone never manufactures an independent speaker.
    queue=sorted(plan["cells"],key=lambda r:(r["model_id"],r["voice_id"],r["cell_id"]))
    attempts=0
    for cell in queue:
        folder=root/"cells"/cell["cell_id"]
        if (folder/"started.json").exists():
            continue  # inclusive of failed/incomplete: NEVER retry a consumed cell
        if attempts>=max_calls:
            break
        c.require(not folder.exists(),"UNEXPLAINED_CELL_DIRECTORY")
        folder.mkdir(parents=True)
        _once(folder/"started.json",{"schema":"tts-generated-cell-start-v1",
            "cell_id":cell["cell_id"],"seed":cell["seed"],
            "voice_id":cell["voice_id"],"source_family_id":cell["source_family_id"],
            "model_id":cell["model_id"],"revision":cell["model_revision"],
            "plan_sha256":expected_hash,"retry_count":0})
        attempts+=1
        try:
            output,rate=backend.synthesize(cell,voices[cell["voice_id"]])
            metrics=quality(output,rate)
            rendered=writer(folder,output,rate,metrics)
            c.require(type(rendered) is dict and
                      all(type(rendered.get(k)) is int for k in
                          ("native_bytes","wav_bytes","audio_frames","sample_rate_hz")) and
                      all(type(rendered.get(k)) is str and
                          c._SHA.fullmatch(rendered[k]) for k in
                          ("native_sha256","wav_sha256","pcm_sha256")),
                      "WRITER_IDENTITY")
            _once(folder/"completed.json",{
                "schema":"tts-generated-cell-completed-v1",
                "status":"GENERATED_UNADMITTED","cell_id":cell["cell_id"],
                "plan_sha256":expected_hash,"technical":metrics,"audio":rendered,
                "human_gold":False,"actual_text":None,
                "training_admitted":False,"qualification_allowed":False})
        except Exception as err:
            _once(folder/"failed.json",{"schema":"tts-generated-cell-failure-v1",
                "cell_id":cell["cell_id"],"error_type":type(err).__name__,
                "status":"FAILED_NO_RETRY","training_admitted":False})
    return {"attempts_this_call":attempts,
        "total_started":sum((root/"cells"/r["cell_id"]/"started.json").exists()
                            for r in plan["cells"]),
        "total_completed":sum((root/"cells"/r["cell_id"]/"completed.json").exists()
                            for r in plan["cells"]),
        "training_admitted":False,"qualification_allowed":False,
        "status":"SYNTHETIC_GENERATION_NOT_TRAINING"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-lock",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--max-calls",type=int,default=24)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        plan=c.validate_plan(c.read_json(args.plan))
        if not args.execute:
            print(json.dumps({"validated":True,"prospective_cells":len(plan["cells"]),
                "model_calls":0,"training_admitted":False}))
            return 0
        c.require(os.environ.get("HF_HUB_OFFLINE")=="1" and
                  os.environ.get("TRANSFORMERS_OFFLINE")=="1",
                  "OFFLINE_RUNTIME_REQUIRED")
        assets=validate_models(plan,c.read_json(args.model_lock),args.model_root)
        backend=QwenOfflineBackend(assets,args.references,args.device)
        result=run_batch(plan,args.out,backend,max_calls=args.max_calls)
    except (ValueError,OSError) as err:
        print("AUTOTTS_STOP: "+type(err).__name__,file=sys.stderr)
        return 2
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
