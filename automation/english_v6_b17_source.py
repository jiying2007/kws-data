"""One frozen B17 fresh-voice primary; all earlier sources and reserves excluded."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b16_source as previous
import english_v6_b17_source_ancestry as lineage

qa = previous.qa
B13 = previous.B13
PLANS = lineage.PLANS
FILES = {"primary": "e118466cf7cfb8ae5e3de1ac763af6274a7e89f29fe22cd921f89331b5f92b11",
         "replication": "7e083f1c187bcc187469879078f919b600b047e87adaf7ca31e069d484e6bdc3"}
SCHEMA = "english-v6-b17-frozen-independent-source-v1"
BATCH = "english-v6-b17-96-original-blind-source-screen-v1"


def source_plan(panel):
    assessment, _ = lineage.read_plan()
    qa.need(panel in FILES, "one preregistered B17 role")
    raw = (PLANS / (panel + ".json")).read_bytes()
    qa.need(lineage.digest(raw) == FILES[panel], "immutable B17 24-voice source roster")
    plan = json.loads(raw)
    first = 718 if panel == "primary" else 742
    sources = {(r["speaker_index"], r["source_speaker_id"]) for r in plan["rows"]}
    qa.need(plan["schema"] == SCHEMA and plan["panel"] == panel
            and plan["immutable_assessment_sha256"] == lineage.ASSESSMENT_SHA
            and plan["source_ancestry_sha256"] == lineage.ANCESTRY_SHA
            and plan["max_source_calls"] == 96 and plan["max_native_observer_calls"] == 480
            and not plan["training_allowed"] and not plan["promotion_allowed"]
            and len(plan["rows"]) == 96 and len(sources) == 24
            and {i for i, _ in sources} == set(range(first, first + 24))
            and len({r["id"] for r in plan["rows"]}) == 96,
            "fixed source geometry and immutable provenance")
    for index, person in sources:
        rows = [r for r in plan["rows"] if r["source_speaker_id"] == person]
        qa.need(len(rows) == 4 and sorted(r["cell_id"] for r in rows) ==
                ["mero-fast", "mero-slow", "question", "request"]
                and sum(r["requested_class"] == 2 for r in rows) == 2
                and sum(r["requested_class"] == 0 for r in rows) == 2
                and all(r["role"] == "development" and not r["training_allowed"]
                        and r["observed_class"] is None and r["panel"] == panel
                        and r["speaker_index"] == index for r in rows),
                "unlabeled two-positive/two-ordinary prospective source requests")
    return plan


def gate(original_screen):
    previous.gate(original_screen)


def required_archives():
    _, history = lineage.read_plan()
    return tuple(dict.fromkeys((*previous.required_archives(),
                  str(history["b16_synthetic_source"]["artifact_id"]), *lineage.required_archives())))


def history(directory):
    seen, groups, people = previous.history(directory)
    _, registered = lineage.read_plan()
    lock = registered["b16_synthetic_source"]
    raw = lineage.member(directory, lock, "MANIFEST.json")
    prior = json.loads(raw)
    rows = prior["rows"]
    qa.need(prior["schema"] == previous.SCHEMA and prior["panel"] == "primary"
            and prior["tts_calls"] == len(rows) == 96
            and len({r["source_group_id"] for r in rows}) == 24
            and all(r["role"] == "development" for r in rows),
            "complete actual B16 primary ancestry, no regenerated old source")
    seen.update(r["pcm_sha256"] for r in rows)
    groups.update(r["source_group_id"] for r in rows)
    qa.need(len(seen) == 1396 and len(groups) == 276,
            "full B16 96-original/24-family increment closes historical source chain")
    for panel in ("primary", "replication"):
        people.update(r["source_speaker_id"].removeprefix("p")
                      for r in previous.source_plan(panel)["rows"])
    old_pcm, old_people = lineage.verify_history(directory)
    seen.update(old_pcm)
    people.update(old_people)
    return seen, groups, people


def validate_config(config, history_groups, people):
    all_rows = source_plan("primary")["rows"] + source_plan("replication")["rows"]
    def reader_at(start):
        def reader(role):
            first = start + (0 if role == "primary" else 12)
            return {"rows": [r for r in all_rows if first <= r["speaker_index"] < first + 12]}
        return reader
    B13.groups_check(config, history_groups, plan_reader=reader_at(718), first=718, extra_people=people)
    B13.groups_check(config, history_groups, plan_reader=reader_at(742), first=742, extra_people=people)
    qa.need(len({r["source_speaker_id"] for r in all_rows}) == 48,
            "disjoint fixed new primary and reserved replication voices")


def generate(out, archives):
    seen, groups, people = history(archives)
    B13.render_panel(out, source_plan("primary"), qa.sha(PLANS / "primary.json"), seen,
                     lambda config: validate_config(config, groups, people))
    manifest = out / "MANIFEST.json"
    document = json.loads(manifest.read_bytes())
    document.update(source_ancestry_sha256=lineage.ANCESTRY_SHA,
                    assessment_sha256=lineage.ASSESSMENT_SHA,
                    all_prior_replication_sealed=True)
    manifest.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
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
            and manifest["source_ancestry_sha256"]==lineage.ANCESTRY_SHA
            and manifest["assessment_sha256"]==lineage.ASSESSMENT_SHA
            and manifest["all_prior_replication_sealed"] is True
            and len({r["pcm_sha256"] for r in rows})==96,
            "96 full original PCM from locked plans, zero KWS forwards")
    for row,frozen in zip(rows,plan["rows"],strict=True):
        qa.need(all(row[k]==v for k,v in frozen.items()) and row["role"]=="development"
                and row["training_allowed"] is False and row["whole_native_resample"] is True
                and row["crop_calls"]==0 and row["basic_audio_ok"] is True,
                "original immutable B17 source and split")
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
     "source_ancestry_sha256":lineage.ANCESTRY_SHA,"assessment_sha256":lineage.ASSESSMENT_SHA,
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
         "one fixed B17 quality source gate")
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
