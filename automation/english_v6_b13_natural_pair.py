"""Train two matched 43K arms using original TRAIN and nine owner-reviewed natural windows."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_hardnegative_pair as paired
import english_v6_b11_train_hardness as base_data
import english_v6_four_model_panel as scoring

b10 = paired.b10
need = b10.need
BASE = Path(__file__).resolve().parents[1]
PINS = BASE / "automation/english_v6_b13_natural_pair_inputs.json"
REVIEW = BASE / "research/english_hardneg_increment_20261117/OWNER_REVIEW.json"
SCHEMA = "english-v6-b13-train-only-owner-natural-pair-v1"
STEMS = {name: f"mero43-b13-{name}192" for name in ("control", "challenger")}


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/jiying2007/kws-data/" + endpoint])


def fetch(pins, work):
    """Fetch two frozen plans, two gate archives and the original B9 training inventory."""
    work.mkdir()
    reg = pins["registered_panels"]
    for k in ("primary", "replication", "natural-design"):
        path, sha = reg["files"][k]["path"], reg["files"][k]["sha256"]
        meta = json.loads(gh("contents/" + path + "?ref=" + reg["commit"]))
        raw = base64.b64decode(meta["content"])
        need(b10.digest(raw) == sha, "registered independent panel " + k)
        (work/(k+".json")).write_bytes(raw)
    gate = pins["replay"]
    run = json.loads(gh("actions/runs/" + str(gate["run_id"])))
    need(run["status"]=="completed" and run["conclusion"]=="success" and run["run_attempt"]==1
         and run["head_sha"]==gate["head"], "exact completed natural-window replay run")
    sources = {name:(spec[0],spec[1]) for name,spec in base_data.ARCHIVES.items()}
    for k in ("natural", "replay"):
        sources[k] = pins[k]["id"], pins[k]["sha256"]
    for name,(aid,sha) in sources.items():
        data = gh(f"actions/artifacts/{aid}/zip")
        need(b10.digest(data)==sha, "input ZIP "+name)
        (work/(name+".zip")).write_bytes(data)


def reviewed_gate(inputs, pins, np):
    """Re-read the reviewed windows and attest every frozen native B9 hard example."""
    need(b10.digest(REVIEW.read_bytes())==pins["review_sha256"],"authenticated owner review")
    review=json.loads(REVIEW.read_bytes())
    need(review["owner_statement_verbatim"]=="没有唤醒词" and not review["full_original_utterance_labels_asserted"]
         and review["reviewed_negative_samples"]==488880 and not review["independent_test_admitted"],"scope of physical listening")
    members = {r["candidate_id"]:r for r in review["candidates"]}
    need(len(members)==len(review["candidates"])==9 and len({r["speaker_id"] for r in members.values()})==8,"natural window roster")
    with scoring.opened(inputs/"replay.zip",pins["replay"]) as z:
        result=json.loads(z.read("RESULT.json"))
        raw=z.read("TRAIN_FRAME_LOGITS.npz")
    need(b10.digest(raw)==result["all_logits_sha256"] and result["actual_forwards"]==54 and result["gain"]==1.0
         and result["optimizer_updates"]==result["dev_cal_locked_forwards"]==result["full_utterance_model_forwards"]==0
         and result["model_pt_sha256"]["B9"]==b10.b9.MODELS["B9_control"]["pt"]
         and result["schema"]=="english-v6-owner-negative-window-replay-v1" and not result["promotion_allowed"],
         "frozen TRAIN-only gate")
    scores = {s["candidate_id"]:s for s in result["records"]}
    need(len(scores)==len(result["records"]) and set(scores)==set(members),"exact reviewed score subjects")
    hard, events, windows, waves, rows = set(), 0, 0, {}, []
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays, scoring.opened(inputs/"natural.zip",pins["natural"]) as archive:
        for uid,r in sorted(members.items()):
            need(r["source_role"]=="train" and r["research_training_admitted"] and r["trusted_negative"] and
                 r["review_decision"]=="N" and "Hey Mero" in r["negative_targets"],"approved exact window")
            wav=archive.read("candidates/wav/"+uid+".wav")
            pcm=archive.read("candidates/pcm/"+uid+".pcm")
            score=scores[uid]
            need(b10.digest(pcm)==r["window_pcm_sha256"]==score["pcm_sha256"] and b10.digest(wav)==r["wav_sha256"],
                 "authenticated window PCM and WAV")
            with wave.open(io.BytesIO(wav)) as stream:
                need(stream.getparams()[:3]==(1,2,16000) and stream.readframes(stream.getnframes())==pcm, "16k mono PCM")
            need(r["speaker_id"]==score["speaker_id"] and r["recording_id"]==score["recording_id"] and
                 r["start_sample"]==score["start_sample"] and r["end_sample_exclusive"]==score["end_sample_exclusive"] and
                 r["end_sample_exclusive"]-r["start_sample"]==r["window_samples"]==len(pcm)//2,"identity and boundaries")
            state=score["models"]["B9"]
            arr=arrays[state["key"]]
            need(b10.digest(arr.tobytes())==state["logits_sha256"] and np.isfinite(arr).all()
                 and arr.shape==(1+(len(pcm)//2-512)//160,4),"verified native B9 frame array")
            path=arr.argmax(1).tolist()
            runs=[(i,t) for i,t in enumerate(path) if t and (i==0 or path[i-1]!=t)]
            tokens=[t for _,t in runs]
            fired=[runs[i+2][0] for i in range(len(tokens)-2) if tokens[i:i+3]==[1,2,3]]
            need(fired==state["event_frames"] and tokens==state["tokens"],"whole-word B9 replay")
            if fired:
                hard.add(r["speaker_id"])
                windows+=1
                events+=len(fired)
            # The B10 helper's 'silver' name selects its full-weight loss slot.
            # The original owner review is the label; no synthetic confidence tier was assigned.
            rows.append(dict(id=uid,role="train",mono_class=0,tier="silver",source_group_id="librispeech:"+r["speaker_id"],
                             source_pcm_sha256=r["window_pcm_sha256"],samples=r["window_samples"],
                             start_sample=r["start_sample"],end_sample_exclusive=r["end_sample_exclusive"],
                             recording_id=r["recording_id"],speaker_id=r["speaker_id"],b9_original_gain_events=len(fired),
                             label_basis="owner-reviewed exact interval",loss_scaling=1.0))
            waves[uid]=pcm
    need(len(hard)>=2 and result["two_group_hardness_diagnostic"] and sorted(hard)==result["summary"]["B9"]["speakers"]
         and events==result["summary"]["B9"]["events"] and windows==result["summary"]["B9"]["windows"],"registered >=2 real-speaker hardness gate")
    return rows,waves


def execute(inputs, out, pins):
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    extra, extra_waves = reviewed_gate(inputs, pins, np)
    rows, waves, groups = paired.original_training(inputs)
    need(len(rows)==79 and all(len(g)==35 for g in groups.values()) and
         not ({r['source_pcm_sha256'] for r in extra} & {r['source_pcm_sha256'] for r in rows}),"distinct base/original intervals")
    for k in ("primary","replication","natural-design"):
        raw = (inputs/(k+".json")).read_bytes()
        need(b10.digest(raw)==pins["registered_panels"]["files"][k]["sha256"],"fresh independent "+k)
    draws=paired.freeze_draw(extra,b10.STEPS)
    schedule=[]
    for i,r in enumerate(draws):
        slot=dict(step=i+1,group=r["source_group_id"],id=r["id"],source_pcm_sha256=r["source_pcm_sha256"],
                  prefix=b10.PREFIXES[(i//b10.TRAIN_GROUPS+i%3)%len(b10.PREFIXES)])
        for klass,key in ((0,'ordinary'),(1,'positive')):
            group=sorted(groups[klass])[i%b10.TRAIN_GROUPS]
            existing=sorted(groups[klass][group],key=lambda row:row['id'])
            old=existing[(i//b10.TRAIN_GROUPS)%len(existing)]
            slot[key]={'id':old['id'],'group':group,'pcm_sha256':old['source_pcm_sha256']}
        schedule.append(slot)
    out.mkdir()
    b10.write(out/'COMMON_DRAW.json',schedule)
    draw_sha=b10.digest((out/'COMMON_DRAW.json').read_bytes())
    b10.write(out/'SOURCE.json',dict(schema=SCHEMA,natural_train=extra,base_train=[{'id':r['id'],'pcm_sha256':r['source_pcm_sha256'],'group':r['source_group_id']} for r in rows],
                                    exact_inputs=pins,natural_groups=8,base_groups=35,loss_scaling_tier_semantics="owner-reviewed natural negatives use unattenuated B10 negative CTC scaling"))
    weights,_=b10.b9.verify_b9_archive(inputs/'b9.zip',torch,np)
    model,*frontend=weights['B9_control']
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    base_cache=paired.make_features(rows,waves,frontend,torch,np)
    extra_cache=paired.make_features(extra,extra_waves,frontend,torch,np)
    arms, states={},{}
    for arm,mode in (('control','control'),('challenger','hard_negative')):
        path=out/arm
        path.mkdir()
        r=b10.fit(mode,initial,rows,groups,base_cache,draws,extra_cache,*frontend,path,torch,np,F)
        for s in r['trace']:
            slot=schedule[s['step']-1]
            need(s['prefix']==slot['prefix'] and s['hard_negative_source_id']==(slot['id'] if arm=='challenger' else None),"optimizer follows locked schedule")
        files={}
        for ext in ('pt','npz','onnx'):
            old=path/'models'/(b10.MODEL_STEMS[mode]+'.'+ext)
            target=old.with_name(STEMS[arm]+'.'+ext)
            old.rename(target)
            files[target.name]=r['files'][old.name]
            need(b10.digest(target.read_bytes())==files[target.name]['sha256'],"native model serialization identity")
        states[arm]=torch.load(path/'models'/(STEMS[arm]+'.pt'),map_location='cpu',weights_only=True)
        arms[arm]=dict(optimizer_updates=r['optimizer_updates'],files=files,
                       hard_negative_loss_weight=r['hard_negative_loss_weight'],trace=r['trace'],
                       stem=arm+'/models/'+STEMS[arm],shared_hard_draw_sha256=draw_sha)
    difference=sum((t!=states['challenger'][k]).sum().item() for k,t in states['control'].items())
    need(difference>0 and b10.STEPS==192 and sum(len(v['trace']) for v in arms.values())==26 and
         arms['control']['hard_negative_loss_weight']==0 and arms['challenger']['hard_negative_loss_weight']==0.08,"genuine paired training outcome")
    b10.write(out/'RESULT.json',dict(schema=SCHEMA,**arms,parent_pt_sha256=b10.b9.MODELS['B9_control']['pt'],
                 actual_optimizer_updates=384,trained_model_parameters=b10.PARAMS,different_trained_float_values=difference,
                 gains=list(b10.GAINS),natural_train_originals=9,natural_train_speakers=8,seed=b10.SEED,
                 common_hard_draw_sha256=draw_sha,common_learning_rate=b10.LR,full_utterance_natural_forwards=0,
                 dev_cal_locked_forwards=0,promotion_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('fetch','execute'))
    parser.add_argument('inputs',type=Path)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    need(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_REPOSITORY')=='jiying2007/kws-data'
         and os.environ.get('GITHUB_RUN_ATTEMPT')=='1','standard one-shot public runner')
    pins=json.loads(PINS.read_bytes())
    if a.mode=='fetch':
        fetch(pins,a.inputs)
    else:
        need(a.output is not None,'output required')
        execute(a.inputs,a.output,pins)
