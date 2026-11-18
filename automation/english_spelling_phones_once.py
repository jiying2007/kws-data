"""Independent complete-audio phoneme checks for prospective homophone renderings.

XLSR+LV60 raw CTC are not pronunciation gold. No requested text/target is
passed to models; record unprompted output and refuse to create training labels.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
HERE=Path(__file__).resolve().parent
def load(n):
    spec=importlib.util.spec_from_file_location("spelling_"+n,HERE/(n+".py"))
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x
s=load("english_spelling_micro_once")
p=load("english_hifi_phones_once")
c=load("english_hifi_micro")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--custom")
    ap.add_argument("--clone")
    ap.add_argument("--output")
    a=ap.parse_args()
    if a.source_check:
        s.source_check()
        print(json.dumps({"status":"SOURCE_ONLY","phone_calls":0,"train_labels":0}))
        return
    s.guard()
    custom=s.verify(a.custom,"custom")
    clone=s.verify(a.clone,"clone")
    rows=s.CUSTOM+s.CLONE
    folders={"custom":Path(a.custom),"clone":Path(a.clone)}
    names=[s.item(x,"custom") for x in s.CUSTOM]+[s.item(x,"clone") for x in s.CLONE]
    s.need(len(names)==10 and len({x["id"] for x in names})==10,
           "FROZEN_TEN_DIAGNOSTIC_WAV")
    root=Path(a.output)
    s.need(root.is_absolute() and not root.exists(),"FRESH_PHONE_OUTPUT")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":s.SCHEMA,"planned":20,
        "target_conditioned":False,"training_allowed":False,
        "synthesis_calls_this_stage":0,"source_groups_for_training":0})
    outputs=[]
    try:
        import numpy as np
        import torch
        from huggingface_hub import HfApi,snapshot_download
        from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        for idx,(model_id,revision) in enumerate(c.PHONE_MODELS):
            actual=HfApi(token=False).model_info(model_id,revision=revision)
            s.need(actual.sha.startswith(revision),"PINNED_ORIGINAL_PHONE_MODEL")
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"orthographic-phone-cache"
            modelpath=Path(snapshot_download(repo_id=model_id,revision=actual.sha,
                    cache_dir=str(cache),max_workers=2,token=False))
            weights=modelpath/"pytorch_model.bin"
            s.need(weights.is_file() and weights.stat().st_size==1263535127,
                   "ORIGINAL_PHONE_CHECKPOINT_SIZE")
            with weights.open("rb") as f:
                h=hashlib.sha256()
                for b in iter(lambda:f.read(1_048_576),b""):h.update(b)
            s.need(h.hexdigest()==p.WEIGHT_SHA[idx],"PINNED_PHONE_WEIGHT_HASH")
            feat=Wav2Vec2FeatureExtractor.from_pretrained(str(modelpath),
                       local_files_only=True)
            vocab=p.read_verified_hf_metadata(modelpath,cache,"vocab.json")
            cfg=Wav2Vec2Config.from_json_file(str(modelpath/"config.json"))
            model=Wav2Vec2ForCTC(cfg)
            model.load_state_dict(torch.load(str(weights),map_location="cpu",weights_only=True),strict=True)
            model.eval()
            s.need(cfg.pad_token_id==0 and feat.sampling_rate==16000,"FIXED_NATIVE_CTC")
            for row in names:
                stage="clone" if row["id"].startswith("clone-") else "custom"
                filepath=folders[stage]/"wav"/(row["id"]+".wav")
                wavinfo=c.inspect_wav(filepath)
                pcm,ns=p.pcm_of(filepath)
                s.need(wavinfo["pcm_sha256"]==c.digest(pcm),
                       "SAME_COMPLETE_WAV_AND_PCM")
                x=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
                inp=feat(x,sampling_rate=16000,return_tensors="pt",
                          padding=False,return_attention_mask=True)
                with torch.inference_mode():
                    logits=model(input_values=inp.input_values,
                                 attention_mask=inp.attention_mask.int()).logits
                b=logits[0].detach().cpu().numpy().astype("<f4").tobytes()
                ids=logits.argmax(dim=-1)[0].tolist()
                result={"id":row["id"],"source_group":row["source_group"],
                    "phoneme_model":model_id,"revision":actual.sha,
                    "wav_sha256":wavinfo["wav_sha256"],
                    "pcm_sha256":wavinfo["pcm_sha256"],
                    "sample_count":ns,"phone_tokens":p.ctc(ids,vocab,cfg.pad_token_id),
                    "frame_count":logits.shape[1],"vocab_size":logits.shape[2],
                    "logits_sha256":c.digest(b),
                    "ctc_argmax_sha256":c.digest(struct.pack("<%di"%len(ids),*ids)),
                    "target_conditioned":False,"full_pcm_input":True,
                    "training_allowed":False}
                outputs.append(result)
                c.record(root/(f"{idx}-{row['id']}.json"),result)
            del model,feat
            import gc
            gc.collect()
        s.need(len(outputs)==20,"ALL_TWENTY_OBSERVATIONS")
        c.record(root/"OBSERVATIONS.json",{"schema":s.SCHEMA,
                "status":"ACTUAL_UNPROMPTED_PHONEME_OBSERVATIONS",
                "rows":outputs,"phone_calls":20,"new_tts_calls":0,
                "new_asr_calls":0,"training_allowed":False,
                "canonical_admission":"NOT_RUN"})
        print(json.dumps({"status":"PHONE_OBSERVATION_COMPLETE",
                          "count":len(outputs),"training_allowed":False}))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":s.SCHEMA,
          "error_type":type(exc).__name__,"error_message":str(exc)[:500],
          "completed_forward_returns":len(outputs),"missing":20-len(outputs),
          "training_allowed":False,"retry":False})
        raise
if __name__=="__main__":main()
