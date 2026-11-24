"""B15 prospectively frozen Piper voice primary; one model-family synthetic DEV source-only admission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b14_primary_source_qa as b14

qa=b14.qa
B13=b14.previous
PIN_ROOT=Path(__file__).with_name("english_v6_b15_panels")
FILES={"primary":"afde489a5a87a7ea2cc037ab21e38d779270cf1b","replication":"0de46d409e23bada6bf1d7e434dbe168d1c5f7f4f5ea34e8c656bcc28ce7d6e2"}
SCHEMA="english-v6-b15-primary-blind-source-review-v1"
B14_PRIMARY_ID=14789951699
B14_PRIMARY_SHA="940d9bd7b4798094840d9d7fd1c7909c404428087f5ca8d8d5f8d181cc283a86"
STUDY_SHA="daa45200d226dd004a41e74a3cf5ff876a1f3fc41dd2def7c5ed937efbf085f8"


def source_plan(panel):
    qa.need(panel in FILES,"only preregistered source roles")
    path=PIN_ROOT/(panel+"-synthetic.json")
    qa.need(qa.sha(path)==FILES[panel],"immutable B15 synthetic speaker/utterance roster")
    plan=json.loads(path.read_bytes())
    qa.need(plan["schema"]=="english-v6-b15-frozen-dev-panel-v1"
            and plan["panel"]==panel and len(plan["rows"])==48
            and plan["immutable_assessment_sha256"]==STUDY_SHA
            and plan["max_native_observer_calls"]==240
            and plan["training_allowed"] is False,"frozen source-only panel contract")
    expected=set(range(646 if panel=="primary" else 658,658 if panel=="primary" else 670))
    sources={(r["speaker_index"],r["source_speaker_id"]) for r in plan["rows"]}
    qa.need({i for i,_ in sources}==expected and len(sources)==12,"fixed 12 source families with no substitutions")
    qa.need(len({r["id"] for r in plan["rows"]})==48
            and all(r["role"]=="development" and r["training_allowed"] is False
                    and r["observed_class"] is None for r in plan["rows"]),
            "unique prospective data role and labels")
    return plan


def gate(original_screen):
    # Previously sealed B14 independent source comparison is historical context, never a B15 model-selection panel.
    b14.gate(original_screen)


def history(directory):
    seen,groups,people=b14.history(directory)
    path=directory/(str(B14_PRIMARY_ID)+".zip")
    qa.need(qa.sha(path)==B14_PRIMARY_SHA,"B14 earlier primary source ZIP identity")
    with zipfile.ZipFile(path) as z:
        qa.need(z.testzip() is None,"previous primary source CRC")
        meta=json.loads(z.read("MANIFEST.json"))
    rows=meta["rows"]
    qa.need(len(rows)==48 and len({r["source_group_id"] for r in rows})==12
            and meta["tts_calls"]==48,"prior B14 source population")
    seen.update(r["pcm_sha256"] for r in rows)
    groups.update(r["source_group_id"] for r in rows)
    for role in ("primary","replication"):
        people.update(r["source_speaker_id"].removeprefix("p") for r in b14.source_plan(role)["rows"])
    # Natural research readers and new unadmitted TRAIN source people are forbidden as speaker ancestors too.
    natural=json.loads((Path(__file__).with_name("english_v6_b14_panels")/"natural-rosters.json").read_bytes())
    people.update(str(r["speaker"]) for rs in natural.values() for r in rs)
    train=json.loads(Path(__file__).with_name("english_v6_train_context24_roster.json").read_bytes())
    people.update(str(r["speaker"]) for r in train["source_rows"])
    qa.need(len(seen)==1252 and len(groups)==240,"complete B14 primary historical hashes and groups")
    return seen,groups,people


def generate(out,directory):
    seen,groups,people=history(directory)
    def verify(config):
        # Validate complete exact upstream Piper config SHA and 904-element speaker_id_map.
        # Verify all 24 prospective source IDs against TRAIN, previous DEV and reserved replication voices.
        B13.groups_check(config,groups,plan_reader=source_plan,first=646,extra_people=people)
    B13.render_panel(out,source_plan("primary"),FILES["primary"],seen,verify)
    read_source(out)


def read_source(path):
    return B13.read_source(path,plan_reader=source_plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    sub.add_parser("history")
    job=sub.add_parser("gate");job.add_argument("archive",type=Path)
    job=sub.add_parser("generate");job.add_argument("out",type=Path);job.add_argument("original_archives",type=Path)
    job=sub.add_parser("observe")
    job.add_argument("provider",choices=qa.observers.PROVIDERS)
    job.add_argument("source",type=Path);job.add_argument("out",type=Path)
    job=sub.add_parser("screen")
    for field in ("source","evidence","out"):job.add_argument(field,type=Path)
    args=p.parse_args()
    if args.mode=="history":
        print("\n".join((*B13.locks(),"14788464792","14788972705",str(B14_PRIMARY_ID))))
    elif args.mode=="gate":
        gate(args.archive)
    elif args.mode=="generate":
        generate(args.out,args.original_archives)
    elif args.mode=="observe":
        qa.observers.observe(args.provider,args.source,args.out,read_rows=read_source,schema=SCHEMA)
    else:
        qa.screen(args.source,args.evidence,args.out,schema=SCHEMA,recipe=source_plan)


if __name__=="__main__":
    main()
