"""One fixed B16 new-voice Piper source cohort: 24 new families, 96 PCM originals.

Prospective source-only five-blind-observer screening. B15 model weights remain
frozen and unopened until an independent whole-source gate passes. B15's failed
primary/unknowns and both reserved replication panels remain closed.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b14_primary_source_qa as b14
import english_v6_b15_synthetic_source as b15

qa=b14.qa
B13=b14.previous
PLANS=Path(__file__).with_name("english_v6_b16_panels")
BLOBS={"primary":"afb985d9e82d1b12e5a9bcb253ee682a650ac8ce",
       "replication":"1a4ec1f1d3dc1481d743f0ed752d7e9f3b6ba3c2"}
SCHEMA="english-v6-b16-frozen-independent-source-v1"
BATCH="english-v6-b16-96-original-blind-source-screen-v1"
B15_SOURCE_ID=14878153098
B15_SOURCE_SHA="8b8ad3109cf35a80455a369aeaa65389b45f4d1aee3403c6286cba9ea7c12ed7"


def blob_sha(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def source_plan(panel):
    qa.need(panel in BLOBS,"one preregistered source role")
    raw=(PLANS/(panel+".json")).read_bytes()
    qa.need(blob_sha(raw)==BLOBS[panel],"preregistered B16 Git source blob")
    plan=json.loads(raw)
    min_index=670 if panel=="primary" else 694
    sources={(r["speaker_index"],r["source_speaker_id"]) for r in plan["rows"]}
    qa.need(plan["schema"]==SCHEMA and plan["panel"]==panel
         and plan["max_source_calls"]==96 and plan["max_native_observer_calls"]==480
         and plan["training_allowed"] is False and plan["promotion_allowed"] is False
         and len(plan["rows"])==96 and len(sources)==24
         and {i for i,_ in sources}==set(range(min_index,min_index+24))
         and len({r["id"] for r in plan["rows"]})==96
         and all(r["role"]=="development" and r["training_allowed"] is False
                 and r["observed_class"] is None for r in plan["rows"]),
         "source-identity and 96 original B16 input contract")
    for group in sources:
        speaker=[r for r in plan["rows"] if r["source_speaker_id"]==group[1]]
        qa.need(len(speaker)==4
                and sorted(r["cell_id"] for r in speaker)==["mero-fast","mero-slow","question","request"]
                and sum(r["requested_class"]==2 for r in speaker)==2
                and sum(r["requested_class"]==0 for r in speaker)==2,
                "frozen 2-2 canonical/ordinary voice source geometry")
    return plan


def gate(original_screen):
    """Retain the exact historical B13 question-IPA prerequisite; no model calls."""
    b14.gate(original_screen)

def history(directory):
    # B15.history includes B14 original 48-source primary before the B15 primary.
    # Going directly to B14.history would omit 48 prior PCM and 12 source groups.
    seen,groups,people=b15.history(directory)
    p=directory/(str(B15_SOURCE_ID)+".zip")
    qa.need(qa.sha(p)==B15_SOURCE_SHA,"completed B15 original source archive, no regeneration")
    with zipfile.ZipFile(p) as z:
        qa.need(z.testzip() is None,"B15 full source ZIP CRC")
        manifest=json.loads(z.read("MANIFEST.json"))
    entries=manifest["rows"]
    qa.need(len(entries)==48 and manifest["tts_calls"]==48
            and len({r["source_group_id"] for r in entries})==12,"prior primary source denominator")
    seen.update(r["pcm_sha256"] for r in entries)
    groups.update(r["source_group_id"] for r in entries)
    people.update(r["source_speaker_id"].removeprefix("p") for r in entries)
    # Previous B15 reservation is not opened or regenerated as a B16 source.
    b15=json.loads((Path(__file__).with_name("english_v6_b15_panels")/"replication-synthetic.json").read_bytes())
    people.update(r["source_speaker_id"].removeprefix("p") for r in b15["rows"])
    readers=json.loads((Path(__file__).with_name("english_v6_b14_panels")/"natural-rosters.json").read_bytes())
    people.update(str(r["speaker"]) for x in readers.values() for r in x)
    b15_readers=json.loads((Path(__file__).with_name("english_v6_train_context24_roster.json")).read_bytes())
    people.update(str(r["speaker"]) for r in b15_readers["source_rows"])
    qa.need(len(seen)==1300 and len(groups)==252,
            "full previous B14/B15 source groups accounted for; no duplicate")
    return seen,groups,people


def required_archives():
    """One immutable source lineage, including the previously verified B14 primary."""
    return (*B13.locks(),"14788464792","14788972705",
            str(b15.B14_PRIMARY_ID),str(B15_SOURCE_ID))


def validate_config(config,history_groups,people):
    # Reuse exact SHA-pinned 904-voice config and previously exposed speaker checks,
    # in four disjoint fixed ranges of 12 to avoid weakening the mature upstream guard.
    all_rows=source_plan("primary")["rows"]+source_plan("replication")["rows"]
    def reader_at(start):
        def reader(role):
            lo=start+(0 if role=="primary" else 12)
            rows=[r for r in all_rows if lo<=r["speaker_index"]<lo+12]
            return {"rows":rows}
        return reader
    B13.groups_check(config,history_groups,plan_reader=reader_at(670),first=670,extra_people=people)
    B13.groups_check(config,history_groups,plan_reader=reader_at(694),first=694,extra_people=people)
    qa.need(len({r["source_speaker_id"] for r in all_rows})==48,
            "no duplicated primary/replication or source ancestry")


def generate(out,archives):
    seen,groups,people=history(archives)
    B13.render_panel(out,source_plan("primary"),
                     qa.sha(PLANS/"primary.json"),seen,
                     lambda config:validate_config(config,groups,people))
    read_source(out)


def read_source(directory):
    plan=source_plan("primary")
    manifest=json.loads((directory/"MANIFEST.json").read_bytes())
    rows=manifest["rows"]
    qa.need(manifest["schema"]==SCHEMA and manifest["panel"]=="primary"
            and manifest["tts_calls"]==len(rows)==96
            and manifest["registered_plan_file_sha256"]==qa.sha(PLANS/"primary.json")
            and manifest["plan_sha256"]==B13.b10.sha(B13.b10.canon(plan))
            and manifest["new_kws_forwards"]==0 and manifest["human_gold"] is False
            and len({r["pcm_sha256"] for r in rows})==96,
            "96 full original PCM from locked plans, zero KWS forwards")
    for row,frozen in zip(rows,plan["rows"],strict=True):
        qa.need(all(row[k]==v for k,v in frozen.items()) and row["role"]=="development"
                and row["training_allowed"] is False and row["whole_native_resample"] is True
                and row["crop_calls"]==0 and row["basic_audio_ok"] is True,
                "original immutable B16 source and split")
        facts=qa.pcm_tools.inspect_wav(directory/"wav"/(row["id"]+".wav"))
        qa.need(all(row[k]==v for k,v in facts.items()),
                "complete 16k PCM WAV independently verified")
        native=b""
        for chunk in row["native_chunks"]:
            raw=(directory/"native"/row["id"]/chunk["file"]).read_bytes()
            qa.need(len(raw)==2*chunk["samples"] and B13.b10.sha(raw)==chunk["native_pcm_sha256"],
                    "original TTS native phoneme chunk")
            native+=raw
        qa.need(B13.b10.sha(native)==row["native_pcm_sha256"],
                "whole original native speech before one-shot resampling")
    return rows


def screen(source,observers,output):
    rows=read_source(source)
    texts,phones,coverage=qa.load_observations(source,observers,rows,BATCH)
    accepted=defaultdict(set)
    decisions=[]
    for row in rows:
        ident=row["id"]
        # Existing native five-observer policy checks canonical segmental phones;
        # no requested TTS text becomes gold by itself.
        outcome=qa.admit(row,texts[ident],phones[ident])
        klass=outcome["class"]
        qa.need(outcome["training_allowed"] is False
                and (klass is None or klass==row["requested_class"]),
                "DEV role and label class")
        if klass is not None:
            accepted[row["source_group_id"]].add(klass)
        decisions.append(dict(row,observed_class=klass,tier=outcome["tier"],
                              reason=outcome["reason"],phone_margins=outcome["margins"]))
    qa.need(coverage==dict.fromkeys(qa.observers.PROVIDERS,96),
            "exact five original native model forwards x 96")
    paired=sorted(g for g,classes in accepted.items() if classes=={0,2})
    counts=Counter(r["observed_class"] for r in decisions)
    minimum=source_plan("primary")["frozen_source_gate"]
    mins={key.removesuffix("_min"):value for key,value in minimum.items()}
    measured={"positive_originals":counts[2],"ordinary_originals":counts[0],
              "paired_source_groups":len(paired)}
    qa.need(set(measured)==set(mins),"unaltered frozen source gate keys")
    short={k:{"observed":measured[k],"min":v}
           for k,v in mins.items() if measured[k]<v}
    report={
     "schema":BATCH,"panel":"primary","registered_plan_file_sha256":qa.sha(PLANS/"primary.json"),
     "source_manifest_sha256":qa.sha(source/"MANIFEST.json"),
     "rows":decisions,"coverage":coverage,"unknown_count":counts[None],
     "tier_counts":dict(Counter(r["tier"] for r in decisions)),
     "paired_source_groups":paired,"source_families":24,
     "source_coverage":measured,"frozen_minima":mins,"source_shortfall":short,
     "source_gate":"INSUFFICIENT_SOURCE" if short else "SOURCE_GATE_PASS",
     "tts_calls_at_screen":0,"new_model_forwards":0,"optimizer_updates":0,
     "human_gold":False,"training_allowed":False,
     "threshold_search":False,"replication_opened":False,"promotion_allowed":False
    }
    qa.record(output,report)


def self_test():
    p=source_plan("primary");r=source_plan("replication")
    a={row["source_speaker_id"] for row in p["rows"]}
    b={row["source_speaker_id"] for row in r["rows"]}
    qa.need(len(a)==len(b)==24 and not (a & b),"frozen disjoint 24/24 voice source roles")
    qa.need(all(row["requested_class"] in (0,2) for row in p["rows"])
         and p["frozen_source_gate"]=={"positive_originals_min":12,
             "ordinary_originals_min":24,"paired_source_groups_min":8},
         "one fixed B16 quality source gate")
    print("PASS: 24 independent new primary voices and reserved 24-source replication")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    sub.add_parser("self-test")
    x=sub.add_parser("gate");x.add_argument("archive",type=Path)
    sub.add_parser("history")
    x=sub.add_parser("generate");x.add_argument("out",type=Path);x.add_argument("previous",type=Path)
    x=sub.add_parser("observe");x.add_argument("provider",choices=qa.observers.PROVIDERS)
    x.add_argument("source",type=Path);x.add_argument("out",type=Path)
    x=sub.add_parser("screen");x.add_argument("source",type=Path)
    x.add_argument("observers",type=Path);x.add_argument("out",type=Path)
    a=p.parse_args()
    if a.mode=="self-test":self_test()
    elif a.mode=="gate":gate(a.archive)
    elif a.mode=="history":print("\n".join(required_archives()))
    elif a.mode=="generate":generate(a.out,a.previous)
    elif a.mode=="observe":qa.observers.observe(a.provider,a.source,a.out,read_rows=read_source,schema=BATCH)
    elif a.mode=="screen":screen(a.source,a.observers,a.out)
