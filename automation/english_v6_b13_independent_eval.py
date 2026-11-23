"""Frozen B13 decision on natural ordinary speech and paired synthetic originals."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_four_model_panel as base
import english_v6_b13_natural_sources as natural

need=base.need
GAINS=base.GAINS
ARMS=base.NAMES


def count(row,gain,arm):
    return row["gains"][gain][arm]["events"]


def decision(rows):
    positives=[r for r in rows if r["positive"]]
    ordinary=[r for r in rows if not r["positive"]]
    readers=[r for r in ordinary if r["cohort"]=="natural"]
    added=[[r["id"],g,m] for r in ordinary for g in GAINS for m in ("B9","control")
           if count(r,g,"challenger")>count(r,g,m)]
    lost=[]
    for g in GAINS:
        hits={m:{p["group"] for p in positives if count(p,g,m)} for m in ARMS}
        lost.extend([gid,g,m] for m in ("B0","control") for gid in sorted(hits[m]-hits["challenger"]))
    improved=[]
    for group in sorted({r["group"] for r in readers}):
        events={m:sum(count(r,"1.0",m) for r in readers if r["group"]==group) for m in ARMS}
        if events["challenger"]<min(events["B9"],events["control"]):
            improved.append(group)
    summary={}
    for key,cohort in (("natural_ordinary",readers),("synthetic_ordinary",[r for r in ordinary if r["cohort"]=="synthetic"]),
                       ("synthetic_positive",positives)):
        summary[key]={m:{g:{"originals":len(cohort),"with_event":sum(count(r,g,m)>0 for r in cohort),
                            "events":sum(count(r,g,m) for r in cohort)} for g in GAINS} for m in ARMS}
    native={m:summary["natural_ordinary"][m]["1.0"]["events"] for m in ARMS}
    enough=len(improved)>=2 and native["challenger"]<min(native["control"],native["B9"])
    passed=enough and native["challenger"]<=native["B0"] and not added and not lost
    return dict(summary=summary,natural_speakers_improved=improved,sufficient_natural_advantage=enough,
                no_worse_than_B0_natural_total=native["challenger"]<=native["B0"],
                added_ordinary_events=added,lost_positive_group_gains=lost,panel_gate="PASS" if passed else "NO_PROMOTION")


def human_rows(source,review,panel,contract=natural):
    raw=source.read("MANIFEST.json")
    manifest=json.loads(raw)
    labels=json.loads(review.read("SCREEN.json"))
    design=contract.plan()
    need(manifest["schema"]==labels["schema"]==contract.SCHEMA and labels["source_manifest_sha256"]==base.sha(raw)
         and labels["kws_forwards"]==manifest["kws_forwards"]==0 and labels["training_allowed"] is False
         and manifest["design_sha256"]==contract.DESIGN_SHA and manifest["corpus_sha256"]==design["upstream"]["sha256"]
         and labels["panel"]==manifest["panel"]==panel,"human source-screen provenance")
    roster_raw=source.read("ROSTERS.json")
    roster=json.loads(roster_raw)
    need(manifest["rosters_sha256"]==base.sha(roster_raw),"two-panel roster digest")
    ids={key:{r["source_id"] for r in roster[key]} for key in ("primary","replication")}
    speakers={key:{r["speaker"] for r in roster[key]} for key in ids}
    need(len(ids[panel])==24 and len(speakers["primary"])==len(speakers["replication"])==12 and
         not (speakers["primary"] & speakers["replication"]),"separate natural panels")
    byid={q["id"]:q for q in labels["decisions"]}
    need(set(byid)=={r["id"] for r in manifest["rows"]} and len(byid)==labels["source_rows"],"natural row identities")
    kept=[]
    for r in manifest["rows"]:
        q=byid[r["id"]]
        need(r["source_id"] in ids[panel] and r["source_id"]==q["source_id"] and r["pcm_sha256"]==q["pcm_sha256"]
             and q["group"]==r["source_group_id"] and r["role"]=="development","native identity")
        if q["tier"]=="unknown":
            continue
        need(q["tier"] in ("silver","weak") and not q["full_wake_veto"] and q["exact_votes"]>=2,"admission")
        kept.append(dict(r,cell_id="read-speech",observed_class=0,tier=q["tier"]))
    need(len(kept)==labels["admitted"] and len({r["speaker"] for r in kept})==labels["admitted_speakers"],"natural denominators")
    return kept


def run(spec_path,inputs,output):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    spec=json.loads(spec_path.read_bytes())
    design=natural.plan()
    panel=spec["panel"]
    need(panel in ("primary","replication") and spec["design_commit"]=="24a170a88eb2d08a7e3a5fee62018cbbec8d61a9","registered design")
    registry=Path(__file__).with_name("english_v6_b13_panels")/(panel+".json")
    need(base.sha(registry.read_bytes())==design["synthetic_panel_sha256"][panel],"synthetic roster")
    output.mkdir()
    with ExitStack() as stack:
        opened={k:stack.enter_context(base.opened(inputs/(k+".zip"),spec["inputs"][k]))
                for k in ("synth_source","synth_screen","natural_source","natural_screen")}
        review=json.loads(opened["synth_screen"].read("SCREEN.json"))
        synthetic_candidates=[r for r in review["rows"] if r.get("observed_class") in (0,2) and r["tier"] in ("weak","silver")]
        ss=[r for r in synthetic_candidates if r["observed_class"]==2]
        so=[r for r in synthetic_candidates if r["observed_class"]==0]
        human=human_rows(opened["natural_source"],opened["natural_screen"],panel)
        coverage=dict(synthetic_min_paired_families=len(review["paired_source_groups"]),
                      synthetic_min_positive_sources=len({r["source_group_id"] for r in ss}),
                      synthetic_min_positive_originals=len(ss),synthetic_min_ordinary_originals=len(so),
                      synthetic_min_ordinary_forms=len({r["requested_text"] for r in so}),
                      natural_min_admitted_speakers=len({r["source_group_id"] for r in human}),
                      natural_min_admitted_originals=len(human))
        missing={k:[n,design["source_gate"][k]] for k,n in coverage.items() if n<design["source_gate"][k]}
        common=dict(trial=spec,source_coverage=coverage,source_shortfall=missing,optimizer_updates=0,
                    promotion_allowed=False,field_far_per_hour=None,physical_board_measured=False)
        if missing:
            stop=dict(common,panel_gate="INSUFFICIENT_SOURCE",actual_forwards=0)
            (output/"RESULT.json").write_text(json.dumps(stop,indent=2)+"\n")
            return
        synthetic,paired,faster=base.select(opened["synth_source"],opened["synth_screen"],panel,json.loads(registry.read_bytes()))
        nets=base.snapshots(inputs,spec,torch,np)
        rows=[]
        arrays={}
        for cohort,eligible in (("synthetic",synthetic),("natural",human)):
            archive=opened["synth_source" if cohort=="synthetic" else "natural_source"]
            for item in eligible:
                result=base.infer(archive.read("wav/"+item["id"]+".wav"),item,nets,arrays,torch,np)
                rows.append(dict(id=item["id"],group=item["source_group_id"],cohort=cohort,cell=item["cell_id"],
                     positive=item["observed_class"]==2,tier=item["tier"],pcm_sha256=item["pcm_sha256"],gains=result))
    need(len({r["pcm_sha256"] for r in rows})==len(rows) and len(arrays)==12*len(rows),"unique originals and 12 forwards per original")
    np.savez_compressed(output/"ALL_FRAME_LOGITS.npz",**arrays)
    report=dict(common,**decision(rows),schema="english-v6-b13-net-benefit-v1",actual_forwards=len(arrays),records=rows,
                model_pt_sha256={k:v[-1] for k,v in nets.items()},logits_sha256=base.sha((output/"ALL_FRAME_LOGITS.npz").read_bytes()))
    (output/"RESULT.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    run(*map(Path,sys.argv[1:]))
