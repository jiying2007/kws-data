"""Recompute the preregistered B13 TRAIN retention decision using saved raw logits."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def digest(raw):
    return sha256(raw).hexdigest()


def need(test,message):
    if not test:
        raise ValueError(message)


def check(paths, sha, out):
    diagnostic, prior = paths
    need(digest(diagnostic.read_bytes()) == sha and digest(prior.read_bytes()) == "59e4f39122debbcb3151977a5f6481ba59cb5766a826840a5abaa8653e215bc3","original artifact hashes")
    with ZipFile(diagnostic) as zz, ZipFile(prior) as bb:
        need(zz.testzip() is None and bb.testzip() is None,"input CRCs")
        data=json.loads(zz.read("RESULT.json"))
        roster={r["id"]:r for r in json.loads(bb.read("RESULT.json"))["records"]}
        all_bytes=zz.read("ALL_FRAME_LOGITS.npz");prev_bytes=bb.read("TRAIN_POSITIVE_FRAME_LOGITS.npz")
    need(data["head"]=="cee2389ab3765b01947f485d8a5cf95dc4e3568f" and str(data["run"])=="45111029245" and data["actual_forwards"]==396 and data["optimizer_updates"]==data["dev_cal_locked_forwards"]==0 and digest(all_bytes)==data["logits_sha256"],"original job/run payload")
    distinct={}
    totals=defaultdict(lambda:defaultdict(int))
    reported_shapes=[]
    matched=0
    event_diffs=defaultdict(dict)
    ids={r["id"] for r in data["records"]}
    need(len(ids)==44==len(data["records"]) and ids==roster.keys(),"all original sources")
    by_gain={g:[] for g in ("1.0","0.75","0.5")}
    with np.load(BytesIO(all_bytes), allow_pickle=False) as zz, np.load(BytesIO(prev_bytes), allow_pickle=False) as bb:
        need(len(zz.files)==396 and len(bb.files)==132,"complete original tensors")
        for r in data["records"]:
            p=roster[r["id"]]
            need(r["role"]=="train" and all(r[k]==p[k] for k in ("id","group","tier","pcm_sha256","wav_sha256")),"untouched frozen positive")
            distinct[r["id"]]=r["group"]
            for g,count in r["gains"].items():
                evt={}
                for m,score in count.items():
                    a=zz[score["array"]]
                    need(a.dtype==np.dtype("<f4") and np.isfinite(a).all() and a.shape==(1+(r["samples"]-512)//160,4)
                         and digest(a.tobytes())==score["sha256"],"original numeric frame clock")
                    reported_shapes.append(len(a))
                    best=a.argmax(1);beg=np.flatnonzero(np.r_[True,np.diff(best)!=0] & (best!=0))
                    tok=best[beg].tolist();end=[int(beg[i]) for i in range(2,len(tok)) if tok[i-2:i+1]==[1,2,3]]
                    evt[m]=len(end)
                    need(tok==score["tokens"] and end==score["event_frames"] and evt[m]==score["events"] and [512+160*i for i in end]==score["event_sample_ends_exclusive"],"CTC path and sample ends")
                    totals[g][m]+=bool(end)
                    if m=="B9":
                        old=p["gains"][g]
                        need(tok==old["tokens"] and evt[m]==old["events"], "old reference events")
                        matched+=int(np.array_equal(a,bb[old["array"]]))
                if evt["B9"] and evt["control"] and not evt["challenger"]:
                    by_gain[g].append(r["id"])
                    event_diffs[r["id"]][g]={m:count[m]["tokens"] for m in ("B9","control","challenger")}
    result={}
    for g, group in by_gain.items():
        result[g]=dict(originals=sorted(group),groups=sorted({distinct[idx] for idx in group}))
    good=[g for g,v in result.items() if len(v["groups"])>=2]
    gate="TRAIN_SIGNAL" if good else "STOP"
    need(result==data["same_original_comparisons_by_gain"] and gate==data["retention_hypothesis"]
         and matched==data["historical_b9_numeric_exact"],"independent gate")
    summary=dict(schema="english-v6-b13-positive-retention-independent-readback",archive_sha256=sha,raw_npz_sha256=digest(all_bytes),
                 authentic_run=int(data["run"]),head=data["head"],originals=len(roster),source_groups=len(set(distinct.values())),
                 actual_forwards=len(reported_shapes),complete_frames=sum(reported_shapes),historical_b9_bit_exact_arrays=matched,
                 hits_by_gain=totals,regressions_by_gain=result,regression_paths=event_diffs,
                 unique_regressed_originals=sorted(event_diffs),unique_regressed_groups=sorted({distinct[i] for i in event_diffs}),
                 gains_meeting_two_group_rule=good,gate=gate,optimizer_updates=0,dev_cal_locked_forwards=0,promotion_allowed=False)
    out.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("actual_forwards","complete_frames","historical_b9_bit_exact_arrays","gate","hits_by_gain","regressions_by_gain")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("diag",type=Path);p.add_argument("sha");p.add_argument("prior",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    check((a.diag,a.prior),a.sha,a.out)
