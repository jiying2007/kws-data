"""V5 English: ONE keyword Hey Mero, causal FSMN-CTC real research training.

The eight original GitHub Actions ZIPs are consumed read-only. Earlier two-word
audio/screen code is used solely to AUTHENTICATE original provenance; labels are
reconstructed as a NEW single-target snapshot: Mero=>CTC(HEY,MEH,ROH),
independently verified ordinary=>empty CTC. Neither historical Vosen nor
uncertain/legacy-variant audio is made a new positive or a background negative.
No old calibration or previously exposed locked audio is opened by the model.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CFG=HERE/"english_v5_mero_only.json"
SCHEMA="english-v5-mero-only-fsmn-ctc-run-v1"
REPO="jiying2007/kws-data"
BEFORE="61b2cbe28cb018ebd5a0e359a33e149bd8cb5c07"
PAST_SOURCE=("fresh24","fresh18","piper24")
ARCHIVES={
 "fresh24-source":14422271973,"fresh24-screen":14429385634,
 "fresh18-source":14429578824,"fresh18-screen":14429884666,
 "piper24-source":14434542792,"piper24-screen":14436377961,
 "newtrain24-source":14486048125,"newtrain24-screen":14489594100,
}


def load(name):
    spec=importlib.util.spec_from_file_location(
        "en_v5_auth_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


# Historical modules are NEVER modified and never used as the V5 classifier.
prior=load("english_newtrain24_real_fit_once")
binding=load("english_canonical_multisource_bind_once")
base=load("english_canonical_multisource_fit_once")


def need(ok,reason):
    if not ok:raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode("utf-8")


def config():
    cfg=json.loads(CFG.read_bytes())
    need(cfg.get("schema")=="english-v5-mero-only-fsmn-ctc-research-v1"
         and cfg.get("active_keyword_count")==1
         and cfg.get("target")=={
             "id":"hey_mero","text":"Hey Mero","accent":"en-US",
             "ipa":"/heɪ ˈmɛroʊ/","hint":"Hey MEH-roh",
             "phoneme_tokens":["HEY","MEH","ROH"],
             "ctc_token_ids":[1,2,3],"class_id":1}
         and cfg.get("background",{}).get("class_id")==0
         and cfg["background"]["silence_ctc_target"]==[]
         and cfg["vocabulary"]=={"0":"<blank>","1":"HEY",
                                  "2":"MEH","3":"ROH"}
         and cfg["positive_historical_review_class"]==2
         and cfg["ordinary_historical_review_class"]==0
         and cfg["source_role_policy"]=={
              "allowed":["train","development"],
              "minimum_groups":{"train":6,"development":2},
              "split_by":"upstream_original_speaker_source_group_id",
              "reassignment":False}
         and cfg["archived_artifact_sets"]["historical_six"]==[
              ARCHIVES[x] for x in
              ("fresh24-source","fresh24-screen","fresh18-source",
               "fresh18-screen","piper24-source","piper24-screen")]
         and cfg["archived_artifact_sets"]["new_source"]==
             ARCHIVES["newtrain24-source"]
         and cfg["archived_artifact_sets"]["new_machine_screen"]==
             ARCHIVES["newtrain24-screen"],
         "ONLY_ONE_CANONICAL_MERO_AND_EIGHT_PINNED_HISTORIC_ARTIFACTS")
    m=cfg["model"]
    need(m["architecture"]=="causal_fsmn_ctc_v1"
         and m["input"]=="original_whole_pcm16_16khz"
         and m["mel_bins"]==40
         and m["fft_size"]==512 and m["window_samples"]==400
         and m["hop_samples"]==160 and m["center"] is False
         and m["hidden"]==96 and m["layers"]==4
         and m["left_memory"]==4 and m["ctc_outputs"]==4
         and m["seed"]==20261119 and m["optimizer"]=="AdamW"
         and m["learning_rate"]==0.0008
         and m["max_updates"]==240 and m["max_grad_norm"]==5.0
         and m["negative_ctc_blank_weight"]==0.6
         and m["positive_ctc_weight"]==1.0
         and m["training_augment_snr_db"]==[None,10]
         and m["dev_augmentation"]=="none"
         and m["checkpoint_selection"]=="fixed_final_step_no_test_or_dev_sweep"
         and cfg["execution"]["new_tts_forwards"]==0
         and cfg["execution"]["new_asr_forwards"]==0
         and cfg["execution"]["new_phone_forwards"]==0
         and cfg["execution"]["retry"] is False
         and cfg["completion"]["fresh_calibration_required"] is True
         and cfg["completion"]["fresh_locked_test_required"] is True
         and cfg["completion"]["ssc305_physical_bsp_required"] is True
         and cfg["excluded_input"]["old_exposed_locked_test"]==
             "never_reused_for_calibration_selection_or_training",
         "V5_CAUSAL_FSMN_CTC_NO_LEGACY_OR_HOLDOUT_REUSE")
    need(set(ARCHIVES)=={
        "fresh24-source","fresh24-screen","fresh18-source","fresh18-screen",
        "piper24-source","piper24-screen","newtrain24-source","newtrain24-screen"},
         "EXACT_EIGHT_IMMUTABLE_ZIPS")
    return cfg


def source_check():
    c=config()
    old=prior.source_check()
    need(old["old_train_dev"]==[28,13]
         and old["new_train_dev"]==[8,3]
         and old["new_piper_source_artifact"]==ARCHIVES["newtrain24-source"]
         and old["new_five_native_screen_artifact"]==ARCHIVES["newtrain24-screen"]
         and len(PAST_SOURCE)==3,
         "PREVIOUSLY_VERIFIED_52_SOURCE_ROWS_REQUIRED")
    return {"status":"FROZEN_MONO_MERO_RESEARCH_SOURCE",
            "active_target_ids":["hey_mero"],"positive_ctc":[1,2,3],
            "negative_ctc":[],"historical_source_zip_count":8,
            "new_tts_asr_ctc_calls":0,"new_optimizer_updates":0,
            "profile_sha256":sha(canonical(c)),
            "training_enabled_only_on_public_main":True,
            "old_locked_reopened":False,"ssc305_measured":False}


def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PUBLIC_MAIN_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_PROTECTED_MAIN_NUMERIC_RUN")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False,
         "FIRST_V5_MAIN_SUCCESSOR_NO_REPLAY")
    if parent is None:
        parent=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"V5_MAIN_PARENT_PROTECTION")
    return {"run_id":env["GITHUB_RUN_ID"],"head_sha":head,
            "parent_sha":BEFORE,"profile_sha256":source_check()["profile_sha256"]}


def reviewed_roles(rows):
    """Only explicitly reviewed ordinary and canonical Mero; never Vosen."""
    allowed=[]
    all_ids=set();all_pcm=set();groups=defaultdict(set);source_roles={}
    for row in rows:
        need(row["role"] in ("train","development")
             and row["tier"] in ("silver","weak")
             and row["weak_class"] in (0,1,2)
             and row["weak_class"]==row["requested_class"]
             and row["id"] not in all_ids
             and row["source_pcm_sha256"] not in all_pcm
             and row["human_gold"] is False
             and row["training_allowed_in_original"] is False,
             "FULL_MACHINE_REVIEWED_TRAIN_DEV_ONLY")
        all_ids.add(row["id"]);all_pcm.add(row["source_pcm_sha256"])
        gid=row["source_group_id"]
        need(gid not in source_roles or source_roles[gid]==row["role"],
             "SOURCE_SPLIT_LEAKAGE")
        source_roles[gid]=row["role"]
        # Original class 1 was a separate deprecated target: it is NOT
        # background and contributes NO V5 gradient or evaluation event.
        if row["weak_class"]==1:
            continue
        kind=1 if row["weak_class"]==2 else 0
        need(kind in (0,1),"UNEXPECTED_MERO_TARGET_CLASS")
        groups[(row["role"],gid)].add(kind)
        allowed.append({**row,"mono_class":kind,
                        "ctc_targets":[1,2,3] if kind==1 else [],
                        "mono_keyword_id":"hey_mero" if kind==1 else None})
    qualified={k for k,v in groups.items() if v=={0,1}}
    eligible=[r for r in allowed if
              (r["role"],r["source_group_id"]) in qualified]
    counts=Counter(role for role,_ in qualified)
    need(counts["train"]>=6 and counts["development"]>=2,
         "MONO_MERO_NEEDS_INDEPENDENT_POSITIVE_AND_ORDINARY_GROUPS")
    need({r["mono_class"] for r in eligible if r["role"]=="train"}=={0,1}
         and {r["mono_class"] for r in eligible if r["role"]=="development"}=={0,1},
         "EACH_ROLE_MUST_HAVE_MERO_AND_ORDINARY")
    new_groups={role:sum(role==k[0] for k in qualified
                         if k[1] in {x["source_group_id"] for x in rows
                                     if x["corpus"]=="newtrain24"})
                for role in ("train","development")}
    need(new_groups["train"]>=1 and new_groups["development"]>=1,
         "V5_TRAIN_AND_DEV_MUST_INCLUDE_NEW_UNEXPOSED_SOURCE_GROUP")
    need(all(r["mono_class"] in (0,1)
             and (r["mono_class"]==0)==(r["ctc_targets"]==[])
             for r in eligible),"NO_VOSEN_OR_UNKNOWN_RELABEL")
    return eligible,{"input_trinary_train_dev":len(rows),
                    "excluded_historical_other_target":
                         sum(r["weak_class"]==1 for r in rows),
                    "selected_mono_rows":len(eligible),
                    "selected_by_role":dict(Counter(x["role"] for x in eligible)),
                    "selected_by_class":dict(Counter(
                         "mero" if x["mono_class"]==1 else "ordinary"
                         for x in eligible)),
                    "qualified_groups":{k:counts[k] for k in ("train","development")},
                    "qualified_new_groups":new_groups,
                    "old_calibration_or_locked_audio_used":False,
                    "new_tts_asr_phone_calls":0}


def admission(folder):
    need(source_check()["status"]=="FROZEN_MONO_MERO_RESEARCH_SOURCE",
         "ACTIVE_V5_POLICY_REQUIRED")
    old,origins=prior.all_train_dev(folder)
    eligible,summary=reviewed_roles(old)
    need(len(old)==52 and
         Counter(r["role"] for r in old)==
         {"train":36,"development":16},
         "ORIGINAL_52_READ_ONLY_ADMISSION_INVARIANT")
    return eligible,{**summary,"upstream_new_source":origins,
                     "source_artifacts":dict(ARCHIVES)}


def make_mel_bank(np):
    """Exact 40 HTK mel triangles for a no-center causal 512-sample STFT."""
    high=8000.0;nfft=512;nmel=40
    def mel(f):
        return 2595*math.log10(1+f/700)
    def hz(m):
        return 700*(10**(m/2595)-1)
    anchors=np.asarray([hz(mel(high)*i/(nmel+1))
                        for i in range(nmel+2)],dtype=np.float64)
    hz_bins=np.linspace(0,high,nfft//2+1,dtype=np.float64)
    weights=np.zeros((nfft//2+1,nmel),dtype=np.float32)
    for j in range(nmel):
        left,center,right=anchors[j:j+3]
        rise=(hz_bins-left)/(center-left)
        fall=(right-hz_bins)/(right-center)
        weights[:,j]=np.maximum(np.minimum(rise,fall),0).astype(np.float32)
    need(weights.shape==(257,40)
         and np.isfinite(weights).all()
         and (weights.sum(axis=0)>0).all(),
         "FINITE_FIXED_CAUSAL_LOGMEL_FILTERBANK")
    return weights


def build_model(torch):
    """Trainable causal 4-block FSMN; no look-ahead, no old dual-word head."""
    import torch.nn.functional as F
    class Block(torch.nn.Module):
        def __init__(self,dim,left):
            super().__init__()
            self.linear=torch.nn.Linear(dim,dim)
            self.memory=torch.nn.Conv1d(dim,dim,left+1,groups=dim)
            self.left=left
        def forward(self,x):
            y=self.linear(x)
            delayed=self.memory(F.pad(y.transpose(1,2),
                                       (self.left,0))).transpose(1,2)
            return F.relu(x+delayed)
    class CausalFsmnCtc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input=torch.nn.Linear(40,96)
            self.layers=torch.nn.ModuleList(Block(96,4) for _ in range(4))
            self.output=torch.nn.Linear(96,4)
        def forward(self,frames):
            x=F.relu(self.input(frames))
            for layer in self.layers:
                x=layer(x)
            return self.output(x)
    model=CausalFsmnCtc()
    need(sum(x.numel() for x in model.parameters())==43876,
         "EXPECTED_V5_CAUSAL_FSMN_PARAM_COUNT")
    return model


def features(pcm,row,torch,np,bank,*,noise_db=None):
    need(type(pcm) is bytes and len(pcm)==row["samples"]*2,
         "BYTE_BOUND_ORIGINAL_PCM")
    wave=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768.0
    if noise_db is not None:
        rng=np.random.default_rng(int(sha(
            (row["source_pcm_sha256"]+"/v5/snr10").encode())[:16],16))
        drawn=rng.normal(0,1,size=wave.shape).astype(np.float32)
        signal_rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
        noise_rms=max(signal_rms,0.002)/(10**(noise_db/20))
        drawn*=noise_rms/max(float(np.sqrt(np.mean(drawn**2))),1e-10)
        wave=np.clip(wave+drawn,-1,1)
    need(len(wave)>=6400,"WHOLE_PCM_TOO_SHORT_FOR_TARGET")
    with torch.no_grad():
        original=torch.from_numpy(wave.copy())
        window=torch.hann_window(400,periodic=True)
        power=torch.stft(original,n_fft=512,hop_length=160,
                 win_length=400,window=window,center=False,
                 return_complex=True).abs().square().transpose(0,1)
        mel=power @ torch.from_numpy(bank)
        log=torch.log1p(mel)
    values=log.cpu().numpy().astype(np.float32)
    need(values.shape[1]==40 and values.shape[0]>=35 and
         np.isfinite(values).all(),"CAUSAL_LOGMEL_FRAME_GEOMETRY")
    return values


def ctc_greedy(logits,torch):
    indices=logits.argmax(-1).detach().cpu().reshape(-1).tolist()
    tokens=[];last=-1
    for token in indices:
        if token!=last and token!=0:
            tokens.append(token)
        last=token
    # A complete ordered phrase is needed; arbitrary single phonemes do
    # not count as a wake and Vosen is not an output vocabulary entry.
    phrase=[1,2,3]
    hits=sum(tokens[i:i+3]==phrase for i in range(max(0,len(tokens)-2)))
    return {"collapsed_tokens":tokens,"mero_event":hits>0,
            "extra_events":max(0,hits-1)}


def scored_model(model,rows,feature_cache,mean,scale,torch,np):
    model.eval()
    recordings=[]
    with torch.no_grad():
        for row in sorted(rows,key=lambda x:x["id"]):
            frames=feature_cache[row["id"]]["clean"]
            normalized=((frames-mean)/scale).astype(np.float32)
            logits=model(torch.from_numpy(normalized).unsqueeze(0))
            receipt=ctc_greedy(logits,torch)
            # CTC token-sequence observations are NOT independent event
            # timing, word-end boundaries, or FAR per hour.
            recordings.append({"id":row["id"],"role":row["role"],
                 "source_group_id":row["source_group_id"],
                 "positive":row["mono_class"]==1,
                 "event":receipt["mero_event"],
                 "extra_events":receipt["extra_events"]})
    positives=[x for x in recordings if x["positive"]]
    negatives=[x for x in recordings if not x["positive"]]
    need(positives and negatives,"BOTH_CLASSES_DEVELOPMENT_REQUIRED")
    missed=sum(not x["event"] for x in positives)
    false_events=sum(x["event"]+x["extra_events"] for x in negatives)
    return {"status":"DEV_ONLY_CTC_TOKEN_REPLAY_NOT_FAR",
            "dev_source_groups":len({x["source_group_id"] for x in rows}),
            "positive_original_wavs":len(positives),
            "ordinary_original_wavs":len(negatives),
            "positive_misses":missed,
            "ordinary_false_or_extra":false_events,
            "positive_miss_fraction":missed/len(positives),
            "model_frozen_before_dev":True,
            "no_threshold_tuning":True,
            "no_word_end_latency_assessed":True,
            "not_independent_far_per_hour":True,
            "human_gold":False,"rows":recordings}


def train(rows,archives,out):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(config()["model"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    bank=make_mel_bank(np)
    model=build_model(torch)
    trainrows=[r for r in rows if r["role"]=="train"]
    devrows=[r for r in rows if r["role"]=="development"]
    cache={}
    for row in sorted(rows,key=lambda x:x["id"]):
        original=base.load_source_pcm(row,archives)
        cache[row["id"]]={"clean":features(original,row,torch,np,bank)}
        if row["role"]=="train":
            cache[row["id"]]["snr10"]=features(
                original,row,torch,np,bank,noise_db=10)
    denom=sum(len(cache[r["id"]]["clean"]) for r in trainrows)
    values=sum((cache[r["id"]]["clean"].astype(np.float64).sum(axis=0)
                for r in trainrows),np.zeros(40,dtype=np.float64))
    squares=sum(((cache[r["id"]]["clean"].astype(np.float64)**2).sum(axis=0)
                for r in trainrows),np.zeros(40,dtype=np.float64))
    mean=(values/denom).astype(np.float32)
    scale=np.maximum(np.sqrt(np.maximum(squares/denom-
                         (values/denom)**2,0)),1e-4).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all()
         and (scale>0).all(),"TRAIN_ONLY_NORMALIZATION")
    source_by_class={0:defaultdict(list),1:defaultdict(list)}
    for row in trainrows:
        source_by_class[row["mono_class"]][row["source_group_id"]].append(row)
    need(all(len(source_by_class[k])>=6 for k in (0,1)),
         "SIX_INDEPENDENT_TRAIN_SOURCE_GROUPS_PER_CLASS")
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0008,weight_decay=0.0001)
    start_losses=[];draws=[]
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        objectives=[];chosen=[]
        for clazz in (0,1):
            groups=sorted(source_by_class[clazz])
            group=groups[step%len(groups)]
            eligible=sorted(source_by_class[clazz][group],key=lambda x:x["id"])
            row=eligible[(step//len(groups))%len(eligible)]
            version="clean" if step%2==0 else "snr10"
            frames=cache[row["id"]][version]
            normalized=((frames-mean)/scale).astype(np.float32)
            logits=model(torch.from_numpy(normalized).unsqueeze(0))
            log=logits.log_softmax(-1).transpose(0,1)
            targets=torch.tensor(row["ctc_targets"],dtype=torch.long)
            raw=torch.nn.functional.ctc_loss(
                log,targets,torch.tensor([log.shape[0]],dtype=torch.long),
                torch.tensor([len(targets)],dtype=torch.long),blank=0,
                reduction="none",zero_infinity=False)[0]
            need(bool(torch.isfinite(raw)),"FINITE_REAL_CTC_LOSS")
            tier_weight=1.0 if row["tier"]=="silver" else 0.5
            class_weight=1.0 if clazz else 0.6
            objectives.append(raw*class_weight*tier_weight)
            chosen.append({"id":row["id"],"class":clazz,"source_group":group,
                           "tier":row["tier"],"version":version})
        total=sum(objectives)/2
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        start_losses.append(float(total.detach()))
        draws.append(chosen)
    model.eval()
    modeldir=out/"models";modeldir.mkdir()
    path=modeldir/"mero-fsmn-ctc-v5.pt"
    torch.save(model.state_dict(),path)
    loaded=torch.load(path,map_location="cpu",weights_only=True)
    check=build_model(torch)
    check.load_state_dict(loaded,strict=True)
    with torch.no_grad():
        check.eval()
        probe=torch.from_numpy(((cache[devrows[0]["id"]]["clean"]-
                          mean)/scale).astype(np.float32)).unsqueeze(0)
        diff=(model(probe)-check(probe)).abs().max().item()
    need(diff==0,"WEIGHT_RELOAD_EXACT_NUMERIC_PARITY")
    npz=modeldir/"mero-fsmn-ctc-v5.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy() for k,v in loaded.items()})
    onnx=modeldir/"mero-fsmn-ctc-v5.onnx"
    torch.onnx.export(check,probe,onnx.as_posix(),input_names=["fbank40"],
        output_names=["frame_ctc_logits"],opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(onnx.exists() and onnx.stat().st_size>10000,
         "ACTUAL_EXPORTED_CAUSAL_ONNX_MODEL")
    development=scored_model(check,devrows,cache,mean,scale,torch,np)
    evidence={"schema":SCHEMA,
        "status":"ACTUAL_MONO_MERO_FSMN_CTC_RESEARCH_MODEL_TRAINED",
        "keyword_id":"hey_mero","active_keyword_count":1,
        "parameters":sum(p.numel() for p in check.parameters()),
        "optimizer_steps":len(draws),"data":{
            "train_wavs":len(trainrows),"dev_wavs":len(devrows),
            "train_source_groups":len(source_by_class[0]),
            "dev_source_groups":development["dev_source_groups"],
            "train_snr10_additive_augmentation":True,
            "dev_original_full_pcm_only":True},
        "artifacts":{
            p.name:{"bytes":p.stat().st_size,
                    "sha256":sha(p.read_bytes())}
            for p in (path,npz,onnx)},
        "ctc_token_ids":[1,2,3],"ctc_blank":0,
        "development":development,
        "quality_promotion":False,"new_locked_or_calibration":False,
        "ssc305": "NOT_RUN","human_far_frr":"NOT_MEASURED",
        "no_test_or_dev_checkpoint_sweep":True}
    binding.once(out/"TRAINING.json",{"schema":SCHEMA,
        "updates":len(draws),"losses":start_losses,
        "draws":draws,"final_loss":start_losses[-1],
        "model_seed":20261119,
        "train_only_normalization":True,
        "historical_other_keyword_grads":0,
        "historical_locked_audio_read":False,
        "new_tts_asr_ctc_calls":0})
    binding.once(out/"RESULT.json",evidence)
    return evidence


def execute(folder,out):
    identity=guard()
    out=Path(out)
    need(out.is_absolute() and not out.exists()
         and out.parent.is_dir(),"CREATE_ONCE_PROVENANCE_OUTPUT")
    out.mkdir()
    binding.once(out/"STARTED.json",{"schema":SCHEMA,"run":identity,
              "only_keyword":"hey_mero","actual_optimizer_steps":0})
    source={}
    try:
        records,audit=admission(folder)
        binding.once(out/"ADMISSION.json",{"schema":SCHEMA,
             "profile_sha256":identity["profile_sha256"],
             "status":"V5_REVIEWED_MONO_SOURCE_BOUND",
             "summary":audit,"rows":records,
             "training_calibration_or_locked":False})
        for spec in binding.config()["corpus"]:
            key=spec["key"]
            source[key]=binding.archive(Path(folder)/(key+"-source.zip"),
                                       spec["source"])
        source["newtrain24"]=binding.archive(
            Path(folder)/"newtrain24-source.zip",prior.SOURCE)
        result=train(records,source,out)
        print(json.dumps({"status":result["status"],
             "keyword":"Hey Mero","parameters":result["parameters"],
             "actual_optimizer_steps":result["optimizer_steps"],
             "dev":{k:v for k,v in result["development"].items()
                    if k!="rows"},
             "model_sha256":result["artifacts"]["mero-fsmn-ctc-v5.onnx"]["sha256"],
             "production_promotion":False},sort_keys=True))
    except BaseException as e:
        binding.once(out/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_MONO_MERO_MODEL_NOT_QUALIFIED",
             "error_type":type(e).__name__,"error":str(e)[:450],
             "automatic_retry":False,"test_audio_reopened":False})
        raise
    finally:
        for z in source.values():z.close()


def main():
    p=argparse.ArgumentParser()
    for f in ("--source-check","--audit-inputs","--guard","--execute"):
        p.add_argument(f,action="store_true")
    p.add_argument("--sources");p.add_argument("--output")
    a=p.parse_args()
    need(sum([a.source_check,a.audit_inputs,a.guard,a.execute])==1,
         "EXACT_ONE_MONO_MERO_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.sources,"EIGHT_ORIGINAL_SAVED_SOURCE_ZIPS_REQUIRED")
    if a.audit_inputs:
        rows,audit=admission(Path(a.sources))
        print(json.dumps({"status":"AUTHENTICATED_MONO_MERO_TRAIN_DEV",
            "actual_rows":len(rows),"summary":audit,
            "numerical_model_forwards":0},sort_keys=True))
        return
    need(a.output,"NEW_ACTUAL_RESEARCH_MODEL_OUTPUT_REQUIRED")
    execute(Path(a.sources),a.output)


if __name__=="__main__":
    main()
