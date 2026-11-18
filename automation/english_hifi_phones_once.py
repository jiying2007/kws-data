"""Two unprompted complete-PCM phoneme observers for 7 fresh TTS research clips.

Outputs are observations, not labels. Model input never receives the requested
text, label, brand pronunciation or source role. No training/admission here.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import wave
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("hifi_micro_contract",HERE/"english_hifi_micro.py")
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

WEIGHT_SHA=(
    "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551",
    "3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e")

def pcm_of(path):
    import io
    raw=c.read(path,24*1024*1024)
    with wave.open(io.BytesIO(raw),"rb") as w:
        c.need((w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000),
               "FULL_PCM_NATIVE_WAV_ONLY")
        count=w.getnframes()
        pcm=w.readframes(count)
    c.need(len(pcm)==count*2 and count>=3200,"FULL_PCM_BYTES")
    return pcm,count

def ctc(ids,vocab,pad):
    inverse={v:k for k,v in vocab.items()}
    c.need(len(inverse)==len(vocab),"ONE_TO_ONE_PHONEME_VOCAB")
    from itertools import groupby
    return [inverse[x] for x in (p for p,_ in groupby(ids)) if x!=pad]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--custom",required=True)
    ap.add_argument("--clone",required=True)
    ap.add_argument("--output",required=True)
    opts=ap.parse_args()
    identity=c.guard()
    custom=c.verify_stage(opts.custom,"custom")
    cloned=c.verify_stage(opts.clone,"clone")
    rows=[r for r in custom["rows"] if r["requested_kind"]!="reference"]+cloned["rows"]
    c.need(len(rows)==7 and len({r["id"] for r in rows})==7,
           "EXACT_SEVEN_NONREFERENCE_ROWS")
    root=Path(opts.output)
    c.need(root.is_absolute() and not root.exists(),"FRESH_PHONE_PACKET")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":c.SCHEMA,"run":identity,
        "candidate_rows":len(rows),"models":c.PHONE_MODELS,
        "expected_forwards":c.MAX_PHONE_CALLS,
        "target_text_sent_to_observers":False,
        "training_allowed":False})
    results=[]
    try:
        import numpy as np
        import torch
        from huggingface_hub import HfApi,snapshot_download
        from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        sources={"custom":Path(opts.custom),"clone":Path(opts.clone)}
        for index,(model_id,revision) in enumerate(c.PHONE_MODELS):
            model_info=HfApi(token=False).model_info(model_id,revision=revision)
            c.need(model_info.sha.startswith(revision),"PINNED_PHONE_COMMIT")
            folder=Path(snapshot_download(repo_id=model_id,revision=model_info.sha,
                cache_dir=str(Path(os.environ.get("RUNNER_TEMP","/tmp"))/"hifi-private-phone-cache"),max_workers=2,token=False))
            weights=folder/"pytorch_model.bin"
            c.need(weights.is_file() and weights.stat().st_size==1263535127,
                   "PINNED_PHONE_MODEL_SIZE")
            with weights.open("rb") as f:
                digest=hashlib.sha256()
                for b in iter(lambda:f.read(1048576),b""):
                    digest.update(b)
            c.need(digest.hexdigest()==WEIGHT_SHA[index],"PINNED_PHONE_WEIGHT_DIGEST")
            feat=Wav2Vec2FeatureExtractor.from_pretrained(str(folder),local_files_only=True)
            vocab=json.loads(c.read(folder/"vocab.json"))
            cfg=Wav2Vec2Config.from_json_file(str(folder/"config.json"))
            model=Wav2Vec2ForCTC(cfg)
            state=torch.load(str(weights),map_location="cpu",weights_only=True)
            model.load_state_dict(state,strict=True)
            del state
            model.eval()
            c.need(cfg.pad_token_id==0 and feat.sampling_rate==16000,
                   "FIXED_PHONE_AUDIO_AND_BLANK")
            for row in rows:
                stage="clone" if row["id"].startswith("clone-") else "custom"
                pcm,count=pcm_of(sources[stage]/"wav"/(row["id"]+".wav"))
                x=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
                inputs=feat(x,sampling_rate=16000,return_tensors="pt",
                            padding=False,return_attention_mask=True)
                with torch.inference_mode():
                    logits=model(input_values=inputs.input_values,
                                 attention_mask=inputs.attention_mask.int()).logits
                raw=logits[0].detach().cpu().numpy().astype("<f4").tobytes()
                ids=logits.argmax(dim=-1)[0].tolist()
                token_seq=ctc(ids,vocab,cfg.pad_token_id)
                result={"model":model_id,"revision":model_info.sha,
                        "id":row["id"],"wav_sha256":row["wav_sha256"],
                        "pcm_sha256":row["pcm_sha256"],
                        "sample_count":count,"input_coverage_complete":True,
                        "target_conditioned":False,"decoder_output_complete":True,
                        "logits_sha256":c.digest(raw),"frames":logits.shape[1],
                        "vocab_size":logits.shape[2],"raw_phone_tokens":token_seq,
                        "raw_phone_string":" ".join(token_seq),
                        "ctc_frame_ids_sha256":c.digest(struct.pack("<%di"%len(ids),*ids)),
                        "human_gold":False,"training_allowed":False}
                results.append(result)
                c.record(root/(f"{index}-{row['id']}.json"),result)
            del model,feat
            import gc
            gc.collect()
        c.need(len(results)==c.MAX_PHONE_CALLS,"NO_MISSING_PHONE_FORWARD")
        c.record(root/"OBSERVATIONS.json",{
            "schema":c.SCHEMA,"status":"COMPLETE_INDEPENDENT_PHONE_OBSERVATIONS",
            "models":c.PHONE_MODELS,"rows":results,
            "phone_calls":len(results),"new_asr_calls":0,
            "human_gold":False,"training_allowed":False,
            "manual_audio_review":"NOT_RUN",
            "three_asr_admission":"NOT_RUN"})
        print(json.dumps({"status":"PHONE_OBSERVATION_COMPLETE",
                          "observed":len(results),"training_allowed":False}))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":c.SCHEMA,
            "type":type(exc).__name__,"reason":str(exc)[:500],
            "completed_phone_forwards":len(results),
            "missing":c.MAX_PHONE_CALLS-len(results),
            "no_retry":True,"training_allowed":False})
        raise

if __name__=="__main__":main()
