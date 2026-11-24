"""B16 independent primary: B0/B9/B15-control/B15-nonblank on new original speech.

Candidate weights are frozen by successful B15 paired TRAIN 45296300913.
New 24-voice B16 source and unscored B15 dev-clean readers are independent.
No training, threshold fitting, CAL/LOCKED, field FAR claim or auto-promotion.
"""
import json
from contextlib import ExitStack
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

import english_v6_four_model_panel as common
import english_v6_b16_source as synth
import english_v6_b15_natural_sources as natural
import english_v6_b14_decision as old_decision

SCHEMA="english-v6-b16-b15-unseen-primary-four-frozen-models-v1"
REGISTERED=Path(__file__).with_name("english_v6_b16_b15_primary_inputs.json")
B15_ASSESSMENT_SHA="daa45200d226dd004a41e74a3cf5ff876a1f3fc41dd2def7c5ed937efbf085f8"
UNCHANGED_QUALITY_SHA="944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
GATE={"synthetic_positive":12,"synthetic_ordinary":24,"synthetic_paired":8,
      "natural_originals":12,"natural_speakers":8}

def protocol():
    d=json.loads(REGISTERED.read_bytes())
    study=REGISTERED.with_name("english_v6_b15_panels")/"assessment.json"
    common.need(common.sha(study.read_bytes())==B15_ASSESSMENT_SHA and
                common.sha(old_decision.ASSESSMENT.read_bytes())==UNCHANGED_QUALITY_SHA,
                "no changes to B15 preregistration or B14 acoustic quality safeguard")
    common.need(d["schema"]==SCHEMA and d["source_gate"]==GATE
                and d["scoring_allowed"] in (True,False)
                and d["B14_replication_opened"] is d["B15_replication_opened"] is d["B16_replication_opened"] is False
                and d["promotion_allowed"] is d["physical_ssc305"] is False
                and d["pair_schema"]=="english-v6-b15-train-nonblank-contrast-pair-v1"
                and d["pair_stems"]=={"control":"control/models/mero43-b15-control192",
                                    "challenger":"challenger/models/mero43-b15-challenger192"},
                "source first, genuine frozen B15 pair, no release")
    common.need(synth.source_plan("primary")["frozen_source_gate"]==
               {"positive_originals_min":12,"ordinary_originals_min":24,
                "paired_source_groups_min":8},
                "B16 source gate cannot be weakened after generation")
    return d

def verified_synthetic(sources,screen):
    planned=synth.source_plan("primary")
    raw=sources.read("MANIFEST.json")
    doc=json.loads(raw)
    lab=json.loads(screen.read("SCREEN.json"))
    common.need(doc["schema"]==planned["schema"]==synth.SCHEMA
                and doc["panel"]=="primary" and doc["tts_calls"]==96 and
                doc["registered_plan_file_sha256"]==common.sha(synth.PLANS/"primary.json")
                and doc["new_kws_forwards"]==0 and doc["human_gold"] is False
                and lab["schema"]==synth.BATCH and lab["panel"]=="primary"
                and lab["source_manifest_sha256"]==common.sha(raw)
                and lab["registered_plan_file_sha256"]==doc["registered_plan_file_sha256"]
                and lab["coverage"]==dict.fromkeys(synth.qa.observers.PROVIDERS,96)
                and lab["tts_calls_at_screen"]==lab["new_model_forwards"]==lab["optimizer_updates"]==0
                and lab["training_allowed"] is lab["replication_opened"] is lab["promotion_allowed"] is False,
                "frozen 96 original B16 DEV source + all five unprompted native observers")
    original=doc["rows"]; decisions={r["id"]:r for r in lab["rows"]}
    common.need(len(original)==len(planned["rows"])==len(decisions)==96 and
                len({r["pcm_sha256"] for r in original})==96 and
                len({r["source_group_id"] for r in original})==24,
                "96 full-source unique original PCM and 24 speakers")
    chosen=[];groups={}
    for r,locked in zip(original,planned["rows"],strict=True):
        q=decisions[r["id"]]
        common.need(all(r[k]==v for k,v in locked.items()) and r["role"]=="development"
                    and r["training_allowed"] is False and r["whole_native_resample"] is True
                    and r["crop_calls"]==0 and r["basic_audio_ok"] is True,
                    "source/speaker/IPA/audio original immutable")
        common.need(all(q[k]==v for k,v in r.items() if k!="observed_class"),
                    "machine label did not mutate original source bytes or split")
        if q["observed_class"] is None:
            common.need(q["tier"]=="unknown","reject unlabeled sound")
            continue
        common.need(q["tier"] in ("silver","weak")
                    and q["observed_class"]==r["requested_class"],"blind source label independent")
        groups.setdefault(r["source_group_id"],set()).add(q["observed_class"])
        chosen.append(q)
    pairs=sorted(g for g,c in groups.items() if c=={0,2})
    counts=dict(synthetic_positive=sum(r["observed_class"]==2 for r in chosen),
                synthetic_ordinary=sum(r["observed_class"]==0 for r in chosen),
                synthetic_paired=len(pairs))
    common.need(counts=={"synthetic_positive":lab["source_coverage"]["positive_originals"],
                         "synthetic_ordinary":lab["source_coverage"]["ordinary_originals"],
                         "synthetic_paired":lab["source_coverage"]["paired_source_groups"]}
                and pairs==lab["paired_source_groups"],
                "no source screening omissions or distorted paired numerator")
    return chosen,counts

def verified_human(sources,screen):
    raw=sources.read("MANIFEST.json")
    doc=json.loads(raw);lab=json.loads(screen.read("SCREEN.json"))
    roster_raw=sources.read("ROSTERS.json")
    rosters=json.loads(roster_raw)
    design=natural.plan()
    common.need(doc["schema"]==lab["schema"]==natural.SCHEMA and
                doc["panel"]==lab["panel"]=="primary"
                and doc["design_sha256"]==B15_ASSESSMENT_SHA
                and doc["corpus_sha256"]==design["upstream"]["sha256"]
                and doc["rosters_sha256"]==common.sha(roster_raw)
                and lab["source_manifest_sha256"]==common.sha(raw)
                and lab["coverage"]==dict.fromkeys(natural.qa.observers.PROVIDERS,24)
                and doc["kws_forwards"]==lab["kws_forwards"]==0
                and lab["training_allowed"] is False
                and set(rosters)=={"primary"} and len(rosters["primary"])==24,
                "new real human dev-clean source, five observers, no model selection exposure")
    ref={r["source_id"]:r for r in rosters["primary"]}
    labels={r["id"]:r for r in lab["decisions"]}
    common.need(len(doc["rows"])==len(labels)==24 and
                len({r["pcm_sha256"] for r in doc["rows"]})==24,
                "24 complete human originals")
    admitted=[]
    for row in doc["rows"]:
        q=labels[row["id"]]
        common.need(row["source_id"] in ref and
                    all(row[k]==v for k,v in ref[row["source_id"]].items())
                    and row["role"]=="development" and row["source_group_id"]=="librispeech-dev-clean-"+str(row["speaker"])
                    and q["pcm_sha256"]==row["pcm_sha256"] and
                    q["source_id"]==row["source_id"] and q["group"]==row["source_group_id"],
                    "unchanged human source and five-observer identity")
        if q["tier"]=="unknown": continue
        common.need(q["tier"] in ("silver","weak") and not q["full_wake_veto"]
                    and q["exact_votes"]>=2,"admit only independent verified human nonwake")
        admitted.append(dict(row,cell_id="read-speech",observed_class=0,tier=q["tier"]))
    common.need(len(admitted)==lab["admitted"] and
                len({r["speaker"] for r in admitted})==lab["admitted_speakers"],
                "true source counts and independent human family denominator")
    return admitted,dict(natural_originals=len(admitted),
                         natural_speakers=len({r["speaker"] for r in admitted}))

def source_gate(observed):
    common.need(set(observed)==set(GATE),"all B16 frozen source dimensions")
    deficits={k:{"observed":observed[k],"minimum":v}
              for k,v in GATE.items() if observed[k]<v}
    return "INSUFFICIENT_SOURCE" if deficits else "SOURCE_GATE_PASS",deficits

def run(specfile,inputs,out):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    meta=json.loads(specfile.read_bytes());fixed=protocol()
    common.need(meta==fixed and meta["scoring_allowed"] is True,
                "unopened source role: NEVER infer on incomplete B16 original archive")
    common.need(set(meta["inputs"])=={"B0","B9","pair","synth_source","synth_screen",
                                     "natural_source","natural_screen"},"exact seven frozen sources")
    out.mkdir()
    with ExitStack() as stack:
        zs={k:stack.enter_context(common.opened(inputs/(k+".zip"),meta["inputs"][k]))
            for k in ("synth_source","synth_screen","natural_source","natural_screen")}
        sr,sc=verified_synthetic(zs["synth_source"],zs["synth_screen"])
        hr,hc=verified_human(zs["natural_source"],zs["natural_screen"])
        coverage={**sc,**hc}
        gate,short=source_gate(coverage)
        result={"schema":SCHEMA,"source_coverage":coverage,"source_gate":gate,"source_shortfall":short,
                "panel_gate":"INSUFFICIENT_SOURCE","actual_forwards":0,
                "optimizer_updates":0,"threshold_search":False,"promotion_allowed":False,
                "product_released":False,"B14_replication_opened":False,"B15_replication_opened":False,
                "B16_replication_opened":False,"ssc305_physical":False,
                "natural_field_far_per_hour":None,"field_3_5m_frr":None,
                "installed_software_baseline":"B0"}
        if not short:
            nets=common.snapshots(inputs,meta,torch,np)
            rows=[];arrays={}
            for source,cohort,observed in (("synth_source","synthetic",sr),
                                           ("natural_source","natural",hr)):
                for row in observed:
                    gains=common.infer(zs[source].read("wav/"+row["id"]+".wav"),
                                       row,nets,arrays,torch,np)
                    rows.append(dict(id=row["id"],group=row["source_group_id"],
                                     cohort=cohort,cell=row["cell_id"],
                                     positive=row["observed_class"]==2,tier=row["tier"],
                                     pcm_sha256=row["pcm_sha256"],gains=gains))
            common.need(len({r["pcm_sha256"] for r in rows})==len(rows)
                        and len(arrays)==12*len(rows),
                        "complete 4-model x3-gain full original PCM matrix")
            file=out/"ALL_FRAME_LOGITS.npz"
            np.savez_compressed(file,**arrays)
            decision=old_decision.decide(rows)
            result.update(decision)
            result.update(records=rows,actual_forwards=len(arrays),
                          logits_sha256=common.sha(file.read_bytes()),
                          model_pt_sha256={k:v[-1] for k,v in nets.items()},
                          replication_open_allowed=decision["panel_gate"]=="PASS",
                          promotion_allowed=False)
    (out/"RESULT.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    run(*map(Path,sys.argv[1:]))
