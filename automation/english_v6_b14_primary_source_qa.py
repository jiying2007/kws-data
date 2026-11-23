"""Source-only B14 companion. Only the first 12 registered speakers may be rendered here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b13_primary_source_qa as previous

qa = previous.qa
PINS = Path(__file__).with_name("english_v6_b14_panels")
FILES = {"primary": "7b2ad480a5e536dd180831308ab7f3e106376ebf581735fec25113e4a6430406",
         "replication": "82188b54daaf27dd2264e37d54c12ff20f5d6ae11b6a34412d2cfa88c243a434"}
SCHEMA = "english-v6-b14-primary-blind-source-review-v1"
LAST_SOURCE = "1cc208dd1b83ac6c72b308b1d536bb19b41e5b4d09a4052eefd6d5a7ebdbcfb8"
LAST_NATURAL = "cc647c2164218dfdc314e5505c94f1f72389def90787cbce7d6d4f24d0c2a611"
LAST_SCREEN = "96eee49777cbdc7e6ef2f6b25dcf625cd53490067d1dae8a86e311af7d9ee7ad"


def source_plan(panel):
    qa.need(panel in FILES, "registered split")
    path = PINS / (panel + ".json")
    qa.need(qa.sha(path) == FILES[panel], "registered B14 roster")
    doc = json.loads(path.read_bytes())
    qa.need(doc["schema"] == "english-v6-b14-frozen-dev-panel-v1" and doc["panel"] == panel
            and len(doc["rows"]) == 48, "complete four-cell B14 roster")
    return doc


def gate(path):
    qa.need(qa.sha(path) == LAST_SCREEN, "paired source change confirmed on independent original B13 panel")
    with zipfile.ZipFile(path) as z:
        qa.need(z.testzip() is None, "prior source-screen archive integrity")
        r = json.loads(z.read("SCREEN.json"))
    questions = [x for x in r["rows"] if x["cell_id"] == "question"]
    qa.need(r["schema"] == previous.SCHEMA and r["kws_forwards"] == 0
            and sum(r["coverage"].values()) == 239 and len(questions) == 6
            and all(x["observed_class"] == 0 for x in questions), "original six qualified direct-IPA questions")


def history(directory):
    seen, groups = previous.history(directory)
    p = directory / "14788464792.zip"
    qa.need(qa.sha(p) == LAST_SOURCE, "B13 source archive integrity")
    with zipfile.ZipFile(p) as z:
        rows = json.loads(z.read("MANIFEST.json"))["rows"]
    seen.update(r["pcm_sha256"] for r in rows)
    groups.update(r["source_group_id"] for r in rows)
    qa.need((len(seen), len(groups)) == (1204, 228), "synthetic source history")
    p = directory / "14788972705.zip"
    qa.need(qa.sha(p) == LAST_NATURAL, "B13 natural source archive integrity")
    with zipfile.ZipFile(p) as z:
        natural = json.loads(z.read("ROSTERS.json"))
    people = {str(r["speaker"]) for panel in natural.values() for r in panel}
    qa.need(len(people) == 24, "B13 primary and reserved human source roles")
    people.update(r["source_speaker_id"].removeprefix("p") for r in previous.source_plan("replication")["rows"])
    return seen, groups, people


def generate(out, directory):
    seen, groups, people = history(directory)
    def verify(config):
        previous.groups_check(config, groups, plan_reader=source_plan, first=622, extra_people=people)
    previous.render_panel(out, source_plan("primary"), FILES["primary"], seen, verify)
    read_source(out)


def read_source(path):
    return previous.read_source(path, plan_reader=source_plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("history")
    job = sub.add_parser("gate")
    job.add_argument("archive", type=Path)
    job = sub.add_parser("generate")
    job.add_argument("out", type=Path)
    job.add_argument("original_archives", type=Path)
    job = sub.add_parser("observe")
    job.add_argument("provider", choices=qa.observers.PROVIDERS)
    job.add_argument("source", type=Path)
    job.add_argument("out", type=Path)
    job = sub.add_parser("screen")
    for field in ("source", "evidence", "out"):
        job.add_argument(field, type=Path)
    a = p.parse_args()
    if a.mode == "history":
        print("\n".join((*previous.locks(), "14788464792", "14788972705")))
    elif a.mode == "gate":
        gate(a.archive)
    elif a.mode == "generate":
        generate(a.out, a.original_archives)
    elif a.mode == "observe":
        qa.observers.observe(a.provider, a.source, a.out, read_rows=read_source, schema=SCHEMA)
    else:
        qa.screen(a.source, a.evidence, a.out, schema=SCHEMA, recipe=source_plan)


if __name__ == "__main__":
    main()
