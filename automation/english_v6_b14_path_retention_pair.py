"""One B13-reproducing control and one paired B14 positive path-retention ablation."""
import argparse
import base64
from collections import Counter, defaultdict
import io
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b13_natural_pair as b13
import english_v6_b13_train_positive_retention as diagnostic

b10=b13.b10
need=b13.need
DESIGN_FILE=Path(__file__).with_name("english_v6_b14_positive_retention_design.json")
DESIGN_SHA="28daae60ebcd7191f53fc8b6ab11a8f7e9bd88046fcc22e96cacef29b66da21b"
REGISTRATION="1a7b7ae571cdc33361d704d0ae4f3b5f406ef7b6"
EVAL_GATE=Path(__file__).with_name("english_v6_b14_independent_registration.json")
SCHEMA="english-v6-b14-train-positive-path-retention-pair-v1"


def read_design():
    need(b10.digest(DESIGN_FILE.read_bytes())==DESIGN_SHA,"immutable training protocol")
    return json.loads(DESIGN_FILE.read_bytes())


def fetch(inputs):
    design=read_design()
    inputs.mkdir()
    gate=json.loads(EVAL_GATE.read_bytes())
    for entry in gate["files"]:
        name=entry["path"]
        response=json.loads(b13.gh("contents/"+name+"?ref="+entry["commit"]))
        raw=base64.b64decode(response["content"])
        need(b10.digest(raw)==entry["sha256"],"independent preregistration "+name)
    b10.write(inputs/"independent_registration.json",gate)
    old_pins=json.loads(b13.PINS.read_bytes())
    pins=dict(b13.base_data.ARCHIVES)
    for name in ("natural","replay"):
        pin=old_pins[name];pins[name]=(pin["id"],pin["sha256"])
    for key,name in (("diagnostic","diagnostic"),("historical_b13","pair")):
        p=design[key]
        run=json.loads(b13.gh("actions/runs/"+str(p["run_id"])))
        need(run["head_sha"]==p["head"] and run["conclusion"]=="success" and run["run_attempt"]==1,"completed frozen "+name)
        pins[name]=(p["artifact_id"],p["sha256"])
    for name,(ident,sha) in pins.items():
        raw=b13.gh("actions/artifacts/"+str(ident)+"/zip")
        need(b10.digest(raw)==sha,"original TRAIN-only input "+name)
        (inputs/(name+".zip")).write_bytes(raw)


def completed_word(matrix, np):
    """Return native nonblank frame runs for the earliest full ordered word."""
    labels=matrix.argmax(axis=1)
    points=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    runs=[(int(labels[left]),int(left),int(right)) for left,right in zip(points[:-1],points[1:]) if labels[left]]
    for index in range(len(runs)-2):
        triplet=runs[index:index+3]
        if [v[0] for v in triplet]==[1,2,3]:
            return triplet
    return []


def prepare_paths(inputs, rows, waves, frontend, design, np, torch):
    original={r["id"]:r for r in rows if r["mono_class"]==1}
    need(len(original)==44,"original 44 positives")
    p=design["diagnostic"]
    by_group=defaultdict(dict)
    audit=[]
    mel,mean,scale=frontend
    with zipfile.ZipFile(inputs/"diagnostic.zip") as archive:
        raw=archive.read("ALL_FRAME_LOGITS.npz")
        doc=json.loads(archive.read("RESULT.json"))
    need(b10.digest(raw)==doc["logits_sha256"] and doc["retention_hypothesis"]=="TRAIN_SIGNAL" and
         doc["head"]==p["head"] and doc["actual_forwards"]==396 and doc["originals"]==44,"authentic diagnostic")
    need({r["id"] for r in doc["records"]}==original.keys(),"exact 44 originals")
    gains=Counter()
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
        for measured in doc["records"]:
            uid=measured["id"];row=original[uid];group=measured["group"]
            need(row["role"]=="train" and row["source_pcm_sha256"]==measured["pcm_sha256"] and
                 row["source_group_id"]==group and b10.digest(waves[uid])==measured["pcm_sha256"],"exact positive TRAIN waveform")
            for gain in diagnostic.GAINS:
                slot=measured["gains"][gain]
                paths={}
                for teacher in ("B9","control"):
                    state=slot[teacher]
                    mat=arrays[state["array"]]
                    need(mat.shape==((row["samples"]-512)//160+1,4) and np.isfinite(mat).all() and
                         b10.digest(mat.tobytes())==state["sha256"],"complete raw teacher array")
                    paths[teacher]=completed_word(mat,np)
                    need(bool(paths[teacher])==bool(state["events"]),"frozen word admission")
                if not all(paths.values()):
                    continue
                b9scores=arrays[slot["B9"]["array"]]
                pure=diagnostic.frozen.ref.gain_pcm(waves[uid],float(gain),np)
                f=diagnostic.frozen.v5.features(pure,row,torch,np,mel)
                x=torch.from_numpy(((f-mean)/scale).astype(np.float32))[None]
                need(len(f)==len(b9scores),"teacher/student native shared source frame grid")
                aligned=[]
                for token,start,end in paths["B9"]:
                    others=[c for c in range(4) if c!=token]
                    ref=b9scores[start:end,token]-b9scores[start:end][:,others].max(axis=1)
                    need(len(ref)>0 and np.all(ref>=0),"B9 complete native nonblank span")
                    aligned.append((token,start,end,others,torch.from_numpy(ref.copy())))
                by_group[group].setdefault(uid,[]).append((x,aligned))
                gains[gain]+=1
                audit.append(dict(id=uid,group=group,gain=gain,pcm_sha256=measured["pcm_sha256"],
                                  b9_array_sha256=slot["B9"]["sha256"],matched_array_sha256=slot["control"]["sha256"],
                                  b9_registered_spans=paths["B9"],matched_complete_spans=paths["control"]))
    registered=design["retention"]
    need(dict(gains)==registered["eligible_combinations_by_gain"] and len(by_group)==registered["eligible_groups"]
         and len({r["id"] for r in audit})==registered["eligible_originals"] and len(audit)==120,"all predeclared eligible teacher paths")
    return by_group,audit


def retention_loss(by_group, weight, trace, torch):
    groups=sorted(by_group)
    def calculate(step, model):
        group=groups[step%len(groups)]
        original_losses=[]
        for uid in sorted(by_group[group]):
            gain_losses=[]
            for x,spans in by_group[group][uid]:
                matrix=model(x)[0]
                token_losses=[]
                for token,start,end,others,reference in spans:
                    part=matrix[start:end]
                    student=part[:,token]-part[:,others].max(dim=1).values
                    token_losses.append(torch.relu(reference-student).mean())
                gain_losses.append(torch.stack(token_losses).mean())
            original_losses.append(torch.stack(gain_losses).mean())
        base=torch.stack(original_losses).mean()
        need(bool(torch.isfinite(base)),"finite positive path loss")
        trace.append({"step":step+1,"group":group,"eligible_originals":len(original_losses),"unweighted_loss":float(base.detach())})
        return base*weight
    return calculate


def execute(inputs,output):
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.mkldnn.enabled=False
    design=read_design()
    gate=json.loads((inputs/"independent_registration.json").read_bytes())
    need(gate==json.loads(EVAL_GATE.read_bytes()),"prospective assessment registered")
    p=design["historical_b13"]
    need(b10.digest((inputs/"pair.zip").read_bytes())==p["sha256"],"B13 reference bytes")
    with zipfile.ZipFile(inputs/"pair.zip") as prior:
        past=json.loads(prior.read("RESULT.json"))
        source13=json.loads(prior.read("SOURCE.json"))
        schedule=prior.read("COMMON_DRAW.json")
        old_stem=past["challenger"]["stem"]
        raw=prior.read(old_stem+".pt")
        need(b10.digest(raw)==p["challenger_pt_sha256"],"B13 original natural challenger")
        previous=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    old_pins=json.loads(b13.PINS.read_bytes())
    extra,wextra=b13.reviewed_gate(inputs,old_pins,np)
    rows,waves,groups=b13.paired.original_training(inputs)
    need(set(r["source_pcm_sha256"] for r in rows)=={r["pcm_sha256"] for r in source13["base_train"]}
         and extra==source13["natural_train"],"same 79 base originals and nine reviewed windows")
    draws=b13.paired.freeze_draw(extra,b10.STEPS)
    slots=json.loads(schedule)
    for step,r in enumerate(draws):
        slot=slots[step]
        need(slot["id"]==r["id"] and slot["source_pcm_sha256"]==r["source_pcm_sha256"] and
             slot["prefix"]==b10.PREFIXES[(step//b10.TRAIN_GROUPS+step%3)%len(b10.PREFIXES)],"natural original/prefix draw")
        for klass,key in ((0,"ordinary"),(1,"positive")):
            group=sorted(groups[klass])[step%b10.TRAIN_GROUPS]
            choices=sorted(groups[klass][group],key=lambda entry:entry["id"])
            base=choices[(step//b10.TRAIN_GROUPS)%len(choices)]
            need(slot[key]==dict(id=base["id"],group=group,pcm_sha256=base["source_pcm_sha256"]),"base source draw")
    need(len(draws)==len(slots)==192 and b10.digest(schedule)==past["common_hard_draw_sha256"],"192 frozen steps")
    output.mkdir()
    (output/"COMMON_DRAW.json").write_bytes(schedule)
    nets,_=b10.b9.verify_b9_archive(inputs/"b9.zip",torch,np)
    parent,*frontend=nets["B9_control"]
    original={k:v.detach().clone() for k,v in parent.state_dict().items()}
    cache=b13.paired.make_features(rows,waves,frontend,torch,np)
    hard_cache=b13.paired.make_features(extra,wextra,frontend,torch,np)
    by_group,audit=prepare_paths(inputs,rows,waves,frontend,design,np,torch)
    b10.write(output/"SOURCE.json",dict(base_train=source13["base_train"],natural_train=extra,retention=audit,
                                        independent_design_commit=gate["commit"],registered_optimizer=REGISTRATION))
    trace=[]; arms={}; states={}
    for arm,weight in (("control",0.0),("challenger",design["pair"]["challenger_aux_weight"])):
        dest=output/arm;dest.mkdir()
        aux=retention_loss(by_group,weight,trace,torch) if weight else None
        report=b10.fit("hard_negative",original,rows,groups,cache,draws,hard_cache,*frontend,dest,torch,np,F,auxiliary_loss=aux)
        stem="mero43-b14-"+arm+"192"
        files={}
        for ext in ("pt","npz","onnx"):
            old=b10.MODEL_STEMS["hard_negative"]+"."+ext
            current=stem+"."+ext
            path=dest/"models"/old
            path.rename(path.with_name(current))
            files[current]=report["files"][old]
            need(b10.digest((dest/"models"/current).read_bytes())==files[current]["sha256"],"export identity")
        state=torch.load(dest/"models"/(stem+".pt"),map_location="cpu",weights_only=True)
        states[arm]=state
        if arm=="control":
            exact={k:bool(torch.equal(t,previous[k])) for k,t in state.items()}
            delta=max(float(torch.max(torch.abs(t-previous[k]))) for k,t in state.items())
            diffs=sum(int(torch.count_nonzero(t-previous[k])) for k,t in state.items())
            b10.write(output/"B13_CONTROL_COMPARISON.json",dict(numerically_exact=all(exact.values()),different_float_values=diffs,
                       max_abs=delta,per_tensor=exact,source_pt_sha256=p["challenger_pt_sha256"]))
            need(all(exact.values()) and set(state)==set(previous),"B14 control must exactly reproduce every B13 challenger tensor")
        arms[arm]=dict(files=files,stem=arm+"/models/"+stem,optimizer_updates=report["optimizer_updates"],
                       natural_loss_weight=report["hard_negative_loss_weight"],retention_weight=weight,
                       sampled_training_trace=report["trace"])
    different=sum(int(torch.count_nonzero(t-states["control"][k])) for k,t in states["challenger"].items())
    need(len(trace)==192 and different>0 and all(v["optimizer_updates"]==192 and v["natural_loss_weight"]==0.08 for v in arms.values()),"distinct paired optimization")
    b10.write(output/"RETENTION_TRACE.json",trace)
    b10.write(output/"RESULT.json",dict(schema=SCHEMA,run_id=os.environ["GITHUB_RUN_ID"],head=os.environ["GITHUB_SHA"],
              registration=REGISTRATION,independent_design_commit=gate["commit"],parent_pt_sha256=design["pair"]["parent_pt_sha256"],
              **arms,actual_optimizer_updates=384,trained_model_parameters=43876,different_trained_float_values=different,
              b13_challenger_reproduced_bit_exact=True,common_draw_sha256=b10.digest(schedule),positive_path_combinations=len(audit),
              positive_originals=43,positive_families=34,baseline_originals=79,natural_originals=9,
              dev_cal_locked_forwards=0,new_tts_asr_calls=0,promotion_allowed=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("fetch","execute"));ap.add_argument("inputs",type=Path);ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    if a.mode=="fetch":
        fetch(a.inputs)
    else:
        need(a.output is not None,"expected output directory")
        execute(a.inputs,a.output)
