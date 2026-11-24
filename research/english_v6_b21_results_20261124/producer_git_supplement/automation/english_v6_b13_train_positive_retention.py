"""Score the complete frozen 44-positive TRAIN roster under three retained checkpoints."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_train_hardness as source
import english_v6_b12_positive_train as roster
import english_v6_b9_dev528_four_frozen_once as frozen

ARMS = ("B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
LOCKS = {name:pin for name,pin in source.ARCHIVES.items() if not name.startswith("b10")}
LOCKS["baseline_gain"] = (14782234724, "59e4f39122debbcb3151977a5f6481ba59cb5766a826840a5abaa8653e215bc3")
LOCKS["pair"] = (14789216108, "e66de6d2683c93e434e3d426ef88ffa46c9eba63d89d6940454a81d17e7d29af")
CHECKPOINTS = {"B9": "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655",
               "control": "40c09e038ad731b8f860087bc5983ff39d993249b749402be140d8439928d555",
               "challenger": "643fbbc21f5e2ca7a22c8e320966ddd43c3f2985137e1949ae1e8884565db553"}
require = source.require


def fetch(inputs):
    inputs.mkdir()
    for name, (ident, sha) in LOCKS.items():
        raw = subprocess.check_output(["gh", "api", f"repos/jiying2007/kws-data/actions/artifacts/{ident}/zip"])
        require(source.digest(raw) == sha, "sealed original ZIP "+name)
        (inputs/(name+".zip")).write_bytes(raw)


def pair_models(archives, frontend, np, torch):
    archive = archives["pair"]
    report = json.loads(archive.read("RESULT.json"))
    require(report["schema"] == "english-v6-b13-train-only-owner-natural-pair-v1"
            and report["parent_pt_sha256"] == CHECKPOINTS["B9"] and report["actual_optimizer_updates"] == 384
            and report["dev_cal_locked_forwards"] == 0 and report["trained_model_parameters"] == 43876, "exact completed B13")
    models = {}
    for arm in ("control","challenger"):
        stem = arm+"/models/mero43-b13-"+arm+"192"
        raw = archive.read(stem+".pt")
        require(source.digest(raw) == CHECKPOINTS[arm] and report[arm]["stem"] == stem, "immutable arm")
        weights = torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
        with np.load(io.BytesIO(archive.read(stem+".npz")),allow_pickle=False) as bundle:
            require(set(bundle.files)==set(weights)|{"mel_filters","train_mean","train_scale"} and
                    all(np.array_equal(bundle[k],t.numpy()) for k,t in weights.items()), "same numerical weights")
            require(all(np.array_equal(bundle[k],ref) for k,ref in zip(("mel_filters","train_mean","train_scale"),frontend,strict=True)), "frozen parent frontend")
        model = frozen.v5.build_model(torch)
        model.load_state_dict(weights,strict=True)
        model.eval()
        models[arm] = model
    return models


def decode(logits, torch):
    frame_ids = logits.argmax(-1)[0].tolist()
    starts = [(i,tok) for i,tok in enumerate(frame_ids) if tok!=0 and (i==0 or tok!=frame_ids[i-1])]
    tokens = [t for _,t in starts]
    times = [starts[i][0] for i in range(2,len(tokens)) if tokens[i-2:i+1]==[1,2,3]]
    reference = frozen.v5.ctc_greedy(logits,torch)
    require(tokens==reference["collapsed_tokens"] and len(times)==int(reference["mero_event"])+reference["extra_events"],"complete ordered phrase")
    return tokens, times


def summarize(records):
    regressions = {}
    counts = {}
    unique_ids = set()
    union_groups = set()
    for gain in GAINS:
        matches = [r for r in records if r["gains"][gain]["B9"]["events"]>0 and
                   r["gains"][gain]["control"]["events"]>0 and r["gains"][gain]["challenger"]["events"]==0]
        groups = sorted({r["group"] for r in matches})
        regressions[gain] = {"originals":sorted(r["id"] for r in matches),"groups":groups}
        unique_ids.update(r["id"] for r in matches)
        union_groups.update(groups)
        counts[gain] = {arm:{"originals_with_event":sum(r["gains"][gain][arm]["events"]>0 for r in records),
                             "total_events":sum(r["gains"][gain][arm]["events"] for r in records),
                             "groups_with_event":len({r["group"] for r in records if r["gains"][gain][arm]["events"]})}
                        for arm in ARMS}
    passing = [gain for gain in GAINS if len(regressions[gain]["groups"])>=2]
    return {"positive_summary":counts,"same_original_comparisons_by_gain":regressions,
            "unique_regressed_originals":sorted(unique_ids),"unique_regressed_groups":sorted(union_groups),
            "gains_with_at_least_two_regressed_groups":passing,"two_group_same_gain_gate":bool(passing),
            "retention_hypothesis":"TRAIN_SIGNAL" if passing else "STOP"}


def execute(inputs, output):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    with ExitStack() as stack:
        archives = {}
        for name, (_,sha) in LOCKS.items():
            path=inputs/(name+".zip")
            require(source.digest(path.read_bytes())==sha, "exact bytes for "+name)
            archives[name]=stack.enter_context(zipfile.ZipFile(path))
        originals = roster.select(archives)
        baseline = json.loads(archives["baseline_gain"].read("RESULT.json"))
        require(baseline["actual_forwards"]==132 and baseline["dev_cal_locked_forwards"]==baseline["optimizer_updates"]==0
                and baseline["model_pt_sha256"]==CHECKPOINTS["B9"], "historical TRAIN roster")
        saved = {r["id"]:r for r in baseline["records"]}
        require(saved.keys()=={r["id"] for r in originals} and len(saved)==len(originals)==44, "exact original roster")
        b9, _ = frozen.verify_b9_archive(inputs/"b9.zip",torch,np)
        model, *frontend = b9["B9_control"]
        models = {"B9":model, **pair_models(archives,frontend,np,torch)}
        bank, mean, scale = frontend
        records, arrays = [], {}
        for row in originals:
            ident = row["id"]
            previous = saved[ident]
            require(row["role"]=="train" and row["source_pcm_sha256"]==previous["pcm_sha256"] and
                    row["wav_sha256"]==previous["wav_sha256"] and row["delivery"]==previous["delivery"] and
                    row["source_group_id"]==previous["group"] and row["tier"]==previous["tier"], "historically admitted original")
            pcm=source.read_train_pcm(archives[row["shard"]],row)
            scores={}
            with torch.inference_mode():
                for gain in GAINS:
                    audio=frozen.ref.gain_pcm(pcm,float(gain),np)
                    features=frozen.v5.features(audio,row,torch,np,bank)
                    tensor=torch.from_numpy(((features-mean)/scale).astype(np.float32))[None]
                    scores[gain]={}
                    for arm in ARMS:
                        logits=models[arm](tensor)
                        matrix=logits[0].cpu().numpy().astype("<f4")
                        require(matrix.shape==((row["samples"]-512)//160+1,4) and np.isfinite(matrix).all(), "complete four-class frame clock")
                        tokens,fires=decode(logits,torch)
                        key=ident+"__"+gain+"__"+arm
                        arrays[key]=matrix
                        h=source.digest(matrix.tobytes())
                        scores[gain][arm]=dict(array=key,sha256=h,frames=len(matrix),tokens=tokens,events=len(fires),
                                               event_frames=fires,event_sample_ends_exclusive=[512+160*i for i in fires])
                    b=scores[gain]["B9"]
                    prior=previous["gains"][gain]
                    require(b["tokens"]==prior["tokens"] and b["events"]==prior["events"] and b["frames"]==prior["frames"],"authentic B9 reference events")
                    b["bit_exact_historical_logits"]=b["sha256"]==prior["logits_sha256"]
            records.append(dict(id=ident,role="train",group=row["source_group_id"],tier=row["tier"],
                                samples=row["samples"],source_shard=row["shard"],wav_sha256=row["wav_sha256"],
                                pcm_sha256=row["source_pcm_sha256"],delivery=row["delivery"],gains=scores))
    require(len(records)==44 and len({r["group"] for r in records})==35 and len(arrays)==396,"44 originals, 35 groups, 396 actual model calls")
    output.mkdir()
    file=output/"ALL_FRAME_LOGITS.npz"
    np.savez_compressed(file,**arrays)
    result=dict(schema="english-v6-b13-train-positive-retention-diagnostic-v1",
                run=os.environ["GITHUB_RUN_ID"],head=os.environ["GITHUB_SHA"],originals=44,source_groups=35,
                actual_forwards=len(arrays),model_pt_sha256=CHECKPOINTS,source_archives=LOCKS,optimizer_updates=0,
                new_tts_calls=0,dev_cal_locked_forwards=0,promotion_allowed=False,
                frame_clock=dict(sample_rate=16000,fft=512,hop=160,center=False,complete_frame_sample_end="512+160*frame_index"),
                logits_sha256=source.digest(file.read_bytes()),historical_b9_numeric_exact=sum(r["gains"][g]["B9"]["bit_exact_historical_logits"] for r in records for g in GAINS),
                **summarize(records),records=records)
    (output/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print({k:result[k] for k in ("actual_forwards","positive_summary","same_original_comparisons_by_gain","retention_hypothesis")})


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("command", choices=("fetch","execute"))
    ap.add_argument("inputs",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.command=="fetch":
        fetch(args.inputs)
    else:
        require(args.output is not None,"output directory")
        execute(args.inputs,args.output)
