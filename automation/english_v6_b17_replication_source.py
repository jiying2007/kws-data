"""The frozen B17 742--765 replication, executable only after actual primary PASS."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_replication_gate as gate
import english_v6_b17_replication_history as chain
import english_v6_b17_source as primary

qa, B13, PLANS = primary.qa, primary.B13, primary.PLANS
SCHEMA = primary.SCHEMA
BATCH = "english-v6-b17-replication-96-original-blind-source-screen-v1"
ASSESSMENT_SHA = gate.ASSESSMENT_SHA
history = chain.history
required_archives = chain.required_archives


def source_plan(panel="replication"):
    qa.need(panel == "replication", "only the originally frozen new B17 replication")
    return primary.source_plan(panel)


def validate_config(config, groups, people):
    rows = source_plan()["rows"]
    def reader(role):
        first = 742 if role == "primary" else 754
        return {"rows": [r for r in rows if first <= r["speaker_index"] < first + 12]}
    B13.groups_check(config, groups, plan_reader=reader, first=742, extra_people=people)


def generate(out, archives, authorization):
    permit = gate.require(authorization)
    seen, groups, people = history(archives, authorization)
    B13.render_panel(out, source_plan(), primary.FILES["replication"], seen,
                     lambda config: validate_config(config, groups, people))
    path = out / "MANIFEST.json"
    document = json.loads(path.read_bytes())
    document.update(chain.binding(permit), all_prior_replication_sealed=True)
    qa.record(path, document)
    read_source(out, authorization)


def read_source(directory, authorization):
    permit = gate.require(authorization)
    directory = Path(directory)
    plan = source_plan()
    meta = json.loads((directory / "MANIFEST.json").read_bytes())
    rows = meta["rows"]
    qa.need(all(meta[k] == v for k, v in chain.binding(permit).items())
            and meta["schema"] == SCHEMA and meta["panel"] == "replication"
            and meta["tts_calls"] == len(rows) == 96
            and meta["registered_plan_file_sha256"] == primary.FILES["replication"]
            and meta["plan_sha256"] == B13.b10.sha(B13.b10.canon(plan))
            and meta["new_kws_forwards"] == 0 and meta["human_gold"] is False
            and meta["all_prior_replication_sealed"] is True
            and len({r["pcm_sha256"] for r in rows}) == 96,
            "complete 96-original replica bound to the actual primary PASS")
    for row, frozen in zip(rows, plan["rows"], strict=True):
        qa.need(all(row[k] == v for k, v in frozen.items()) and row["role"] == "development"
                and row["training_allowed"] is False and row["whole_native_resample"] is True
                and row["crop_calls"] == 0 and row["basic_audio_ok"] is True,
                "unaltered original reserved source and split")
        facts = qa.pcm_tools.inspect_wav(directory / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == v for k, v in facts.items()), "whole original PCM16 WAV")
        native = bytearray()
        qa.need(bool(row["native_chunks"]), "original native speech chunks remain present")
        for i, chunk in enumerate(row["native_chunks"]):
            qa.need(chunk["file"] == f"{i:03d}.s16le", "original consecutive native chunk identity")
            raw = (directory / "native" / row["id"] / chunk["file"]).read_bytes()
            qa.need(len(raw) == 2 * chunk["samples"] and B13.b10.sha(raw) == chunk["native_pcm_sha256"],
                    "original native chunk bytes")
            native.extend(raw)
        qa.need(B13.b10.sha(native) == row["native_pcm_sha256"], "whole native original before one resample")
    return rows


def observe(provider, source, output, authorization):
    gate.require(authorization)  # Runs before observer/model initialization.
    qa.observers.observe(provider, source, output,
                         read_rows=lambda path: read_source(path, authorization), schema=BATCH)


def screen(source, providers, output, authorization):
    permit = gate.require(authorization)
    rows = read_source(source, authorization)
    texts, phones, coverage = qa.load_observations(source, providers, rows, BATCH)
    accepted = defaultdict(set)
    decisions = []
    for row in rows:
        outcome = qa.admit(row, texts[row["id"]], phones[row["id"]])
        klass = outcome["class"]
        qa.need(outcome["training_allowed"] is False and (klass is None or klass == row["requested_class"]),
                "unaltered original DEV admission rule")
        if klass is not None:
            accepted[row["source_group_id"]].add(klass)
        decisions.append(dict(row, observed_class=klass, tier=outcome["tier"],
                              reason=outcome["reason"], phone_margins=outcome["margins"]))
    qa.need(coverage == dict.fromkeys(qa.observers.PROVIDERS, 96), "complete five-model native coverage")
    paired = sorted(group for group, classes in accepted.items() if classes == {0, 2})
    counts = Counter(r["observed_class"] for r in decisions)
    mins = {k.removesuffix("_min"): v for k, v in source_plan()["frozen_source_gate"].items()}
    measured = dict(positive_originals=counts[2], ordinary_originals=counts[0], paired_source_groups=len(paired))
    qa.need(mins == {"positive_originals": 12, "ordinary_originals": 24, "paired_source_groups": 8},
            "no change to pre-optimizer replica source minima")
    short = {key: {"observed": measured[key], "min": value}
             for key, value in mins.items() if measured[key] < value}
    qa.record(output, dict(schema=BATCH, panel="replication", **chain.binding(permit),
              registered_plan_file_sha256=primary.FILES["replication"],
              source_manifest_sha256=qa.sha(source / "MANIFEST.json"), rows=decisions,
              coverage=coverage, unknown_count=counts[None], tier_counts=dict(Counter(r["tier"] for r in decisions)),
              paired_source_groups=paired, source_families=24, source_coverage=measured,
              frozen_minima=mins, source_shortfall=short,
              source_gate="INSUFFICIENT_SOURCE" if short else "SOURCE_GATE_PASS",
              tts_calls_at_screen=0, new_model_forwards=0, optimizer_updates=0, human_gold=False,
              training_allowed=False, threshold_search=False, replication_opened=True,
              all_prior_replication_sealed=True, promotion_allowed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    item = modes.add_parser("history"); item.add_argument("directory", nargs="?", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path); item.add_argument("archives", type=Path)
    item = modes.add_parser("observe"); item.add_argument("provider", choices=qa.observers.PROVIDERS)
    item.add_argument("source", type=Path); item.add_argument("out", type=Path)
    item = modes.add_parser("screen"); item.add_argument("source", type=Path)
    item.add_argument("providers", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "history":
        if args.directory:
            history(args.directory, args.authorization)
            print("PASS full old source chain plus actual successful B17 primary; old replicas remain sealed")
        else:
            print("\n".join(required_archives(args.authorization)))
    elif args.mode == "generate":
        generate(args.out, args.archives, args.authorization)
    elif args.mode == "observe":
        observe(args.provider, args.source, args.out, args.authorization)
    elif args.mode == "screen":
        screen(args.source, args.providers, args.out, args.authorization)
