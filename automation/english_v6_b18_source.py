"""A separately preregistered B18 primary; never retry or relabel B17 originals."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_source as previous
import english_v6_b18_source_ancestry as lineage
import english_v6_b18_source_observe as observers

qa, B13 = previous.qa, previous.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = "english-v6-b18-frozen-independent-source-v1"
BATCH = "english-v6-b18-96-original-blind-source-screen-v1"
FILES = {"primary": "7ee93b5c0810fe5fc10f58d67326484e426f35a76a8f9dd22ab05589d2a68ede",
         "replication": "c0a64de1c75d94f5ed0b1fd712e26592279b359584535a98a9de55108ac13b7a"}
history, required_archives = lineage.history, lineage.required_archives


def source_plan(panel):
    design = lineage.read_plan()
    qa.need(panel in FILES, "registered B18 primary or unopened inherited reserve")
    raw = (PLANS / (panel + ".json")).read_bytes()
    qa.need(lineage.sha(raw) == FILES[panel], "immutable prospective source plan")
    plan = json.loads(raw)
    first = 766 if panel == "primary" else 742
    qa.need(plan["schema"] == SCHEMA and plan["panel"] == panel
            and plan["immutable_assessment_sha256"] == ASSESSMENT_SHA
            and plan["source_ancestry_sha256"] == lineage.BASE_ANCESTRY_SHA
            and len(plan["rows"]) == len({r["id"] for r in plan["rows"]}) == 96
            and {r["speaker_index"] for r in plan["rows"]} == set(range(first, first + 24))
            and len({r["source_speaker_id"] for r in plan["rows"]}) == 24
            and {str(r["speaker_index"]): r["source_speaker_id"] for r in plan["rows"]}
                == design[panel]["synthetic"]["speaker_id_map"]
            and plan["model"] == design["primary"]["synthetic"]["model"]
            and plan["config_sha256"] == design["primary"]["synthetic"]["config_sha256"]
            and plan["max_source_calls"] == 96 and plan["max_native_observer_calls"] == 480
            and plan["frozen_source_gate"] == {"positive_originals_min": 12,
                "ordinary_originals_min": 24, "paired_source_groups_min": 8}
            and plan["training_allowed"] is False and plan["promotion_allowed"] is False,
            "96 fixed original source slots, not internal TTS-forward accounting")
    for speaker in {r["source_speaker_id"] for r in plan["rows"]}:
        rows = [r for r in plan["rows"] if r["source_speaker_id"] == speaker]
        qa.need(len(rows) == 4 and sorted(r["cell_id"] for r in rows) ==
                ["mero-fast", "mero-slow", "question", "request"]
                and sorted(r["requested_class"] for r in rows) == [0, 0, 2, 2]
                and all(r["role"] == "development" and r["observed_class"] is None
                        and r["training_allowed"] is False and r["panel"] == panel for r in rows),
                "original two-positive/two-ordinary source geometry")
    if panel == "replication":
        original = previous.source_plan("replication")
        qa.need(plan["rows"] == original["rows"] and plan["model"] == original["model"],
                "inherited reserve requests, voices, seeds, text and IPA remain byte-equivalent")
    return plan


def validate_config(config, groups, people):
    reserved_natural = set(lineage.read_plan()["replication"]["natural"]["speakers"])
    excluded_people = set(people) | reserved_natural
    for panel, first in (("primary", 766), ("replication", 742)):
        rows = source_plan(panel)["rows"]
        def reader(role):
            start = first + (0 if role == "primary" else 12)
            return {"rows": [r for r in rows if start <= r["speaker_index"] < start + 12]}
        B13.groups_check(config, groups, plan_reader=reader, first=first, extra_people=excluded_people)
    current = {r["source_speaker_id"] for r in source_plan("primary")["rows"]}
    reserve = {r["source_speaker_id"] for r in source_plan("replication")["rows"]}
    qa.need(not current & reserve
            and not ({person.removeprefix("p") for person in current | reserve} & reserved_natural),
            "new primary and inherited synthetic/natural reserves are distinct canonical sources")


def code_contract():
    design = lineage.read_plan()
    hashes = observers.source_check()
    qa.need(hashes == design["observer_code_contract"], "reviewed raw-before-validation observer implementation")
    return hashes


def generate(out, archives, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    qa.need(out.is_absolute() and not out.exists() and out.parent.is_dir(),
            "fresh absolute source output before any source/model access")
    code_contract()
    permission = lineage.require_authorization(archives)
    qa.need(permission["source_execution"] == identity,
            "source authorization belongs to this exact first run, head and parent before TTS")
    receipt = lineage.history_receipt(archives)
    seen, groups, people = lineage.validate_history(receipt)
    validate_config(PLANS / "piper-config.json", groups, people)
    B13.render_panel(out, source_plan("primary"), FILES["primary"], seen,
                     lambda config: validate_config(config, groups, people))
    shutil.copyfile(archives / "AUTHORIZATION.json", out / "AUTHORIZATION.json")
    qa.record(out / "HISTORY.json", receipt)
    path = out / "MANIFEST.json"
    doc = json.loads(path.read_bytes())
    doc.update(assessment_sha256=ASSESSMENT_SHA, source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
               source_execution=identity,
               B17_source_stop_retained=True, original_B17_source_proof=lineage.read_plan()["original_B17_source_proof"]["source_inputs"],
               history_receipt_sha256=qa.sha(out / "HISTORY.json"),
               source_authorization_sha256=qa.sha(out / "AUTHORIZATION.json"),
               tts_logical_originals=96, tts_public_synthesize_api_invocations=96,
               native_output_chunks=sum(len(r["native_chunks"]) for r in doc["rows"]),
               internal_tts_model_forwards=None, internal_onnx_session_run_calls=None,
               internal_tts_forwards_instrumented=False,
               all_prior_replication_sealed=True, B18_replication_opened=False,
               new_natural_source_calls=0, new_natural_observer_calls=0)
    # B13 wrote the provisional manifest exactly once; this local augmentation
    # freezes its B18 provenance before any artifact or observer can consume it.
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    read_source(out)


def validate_manifest(meta, receipt, permission):
    plan = source_plan("primary")
    rows = meta["rows"]
    seen, groups, people = lineage.validate_history(receipt)
    qa.need(meta["schema"] == SCHEMA and meta["panel"] == "primary"
            and meta["registered_plan_file_sha256"] == FILES["primary"]
            and meta["plan_sha256"] == B13.b10.sha(B13.b10.canon(plan))
            and meta["assessment_sha256"] == ASSESSMENT_SHA
            and meta["source_ancestry_sha256"] == lineage.BASE_ANCESTRY_SHA
            and meta["tts_calls"] == meta["tts_logical_originals"] == meta["tts_public_synthesize_api_invocations"] == len(rows) == 96
            and meta["new_kws_forwards"] == meta["new_natural_source_calls"] == meta["new_natural_observer_calls"] == 0
            and meta["human_gold"] is False and meta["B17_source_stop_retained"] is True
            and meta["B18_replication_opened"] is False and meta["all_prior_replication_sealed"] is True
            and meta["original_B17_source_proof"] == lineage.read_plan()["original_B17_source_proof"]["source_inputs"]
            and meta["native_output_chunks"] == sum(len(r["native_chunks"]) for r in rows)
            and meta["internal_tts_model_forwards"] is None and meta["internal_onnx_session_run_calls"] is None
            and meta["internal_tts_forwards_instrumented"] is False
            and meta["source_execution"] == permission["source_execution"]
            and permission["assessment_sha256"] == ASSESSMENT_SHA,
            "complete new original provenance and honest source-call accounting")
    qa.need(len({r["pcm_sha256"] for r in rows}) == 96
            and not ({r["pcm_sha256"] for r in rows} & seen)
            and not ({r["source_group_id"] for r in rows} & groups)
            and not ({r["source_speaker_id"].removeprefix("p") for r in rows} & people),
            "all historical B17 failed/unqualified originals and canonical identities excluded")
    for row, expected in zip(rows, plan["rows"], strict=True):
        qa.need(all(row[k] == value for k, value in expected.items())
                and row["whole_native_resample"] is True and row["crop_calls"] == 0
                and row["basic_audio_ok"] is True, "unchanged frozen whole-original source request")
    return rows


def read_source(directory):
    directory = Path(directory)
    permission = lineage.require_authorization(directory)
    receipt = json.loads((directory / "HISTORY.json").read_bytes())
    meta = json.loads((directory / "MANIFEST.json").read_bytes())
    qa.need(meta["history_receipt_sha256"] == qa.sha(directory / "HISTORY.json")
            and meta["source_authorization_sha256"] == qa.sha(directory / "AUTHORIZATION.json"),
            "unchanged full source-history and opening receipts")
    rows = validate_manifest(meta, receipt, permission)
    qa.need(json.loads((directory / "STARTED.json").read_bytes()) ==
            {"plan_sha256": FILES["primary"], "one_shot": True}, "one original source-generation start")
    for row in rows:
        qa.need(json.loads((directory / "starts" / (row["id"] + ".json")).read_bytes()) ==
                {"id": row["id"], "attempt": 1, "seed": row["seed"]}, "one fixed original TTS request")
        facts = qa.pcm_tools.inspect_wav(directory / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == v for k, v in facts.items()), "original full PCM16 WAV bytes")
        native = bytearray()
        qa.need(bool(row["native_chunks"]), "saved native TTS chunks")
        for i, chunk in enumerate(row["native_chunks"]):
            qa.need(chunk["file"] == f"{i:03d}.s16le", "original native chunk numbering")
            raw = (directory / "native" / row["id"] / chunk["file"]).read_bytes()
            qa.need(len(raw) == 2 * chunk["samples"] and lineage.sha(raw) == chunk["native_pcm_sha256"],
                    "native TTS chunk bytes before the original one-shot resampling")
            native.extend(raw)
        qa.need(lineage.sha(native) == row["native_pcm_sha256"], "full native original")
    return rows


def registration(source):
    source = Path(source)
    read_source(source)
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    design = lineage.read_plan()
    code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
                source_manifest_sha256=qa.sha(source / "MANIFEST.json"), source_rows=96, max_calls=96,
                **meta["source_execution"],
                parser_contract_sha256=design["parser_contract_sha256"],
                observer_sha256=design["observer_code_contract"]["observer_sha256"],
                asr_adapter_sha256=design["observer_code_contract"]["asr_adapter_sha256"])


def observe(provider, source, out, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    expected = registration(source)
    qa.need(all(expected[k] == v for k, v in identity.items()),
            "observer belongs to the actual source generation run")
    observers.observe(provider, source, out, read_rows=read_source, schema=BATCH, registration=expected)


def screen(source, providers, output):
    rows = read_source(source)
    # The new strict raw/attempt/return/valid ledger is mandatory before old admission.
    texts, phones, coverage = observers.load_observations(source, providers, rows, BATCH,
                                                         registration=registration(source))
    qa.need(coverage == dict.fromkeys(qa.observers.PROVIDERS, 96), "five valid complete native outputs per original, no abstention substitution")
    decisions, accepted = [], defaultdict(set)
    for row in rows:
        outcome = qa.admit(row, texts[row["id"]], phones[row["id"]])
        value = outcome["class"]
        qa.need(outcome["training_allowed"] is False and (value is None or value == row["requested_class"]),
                "unchanged B14 source-admission rules")
        if value is not None:
            accepted[row["source_group_id"]].add(value)
        decisions.append(dict(row, observed_class=value, tier=outcome["tier"], reason=outcome["reason"],
                              phone_margins=outcome["margins"]))
    counts = Counter(r["observed_class"] for r in decisions)
    paired = sorted(k for k, v in accepted.items() if v == {0, 2})
    mins = dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8)
    measured = dict(positive_originals=counts[2], ordinary_originals=counts[0], paired_source_groups=len(paired))
    missing = {k: {"observed": measured[k], "min": v} for k, v in mins.items() if measured[k] < v}
    design = lineage.read_plan()
    qa.record(output, dict(schema=BATCH, panel="primary", assessment_sha256=ASSESSMENT_SHA,
              source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA, source_manifest_sha256=qa.sha(source / "MANIFEST.json"),
              registered_plan_file_sha256=FILES["primary"], rows=decisions, coverage=coverage,
              source_coverage=measured, frozen_minima=mins, source_shortfall=missing,
              paired_source_groups=paired, source_families=24, unknown_count=counts[None],
              tier_counts=dict(Counter(r["tier"] for r in decisions)),
              source_gate="INSUFFICIENT_SOURCE" if missing else "SOURCE_GATE_PASS",
              parser_contract_sha256=design["parser_contract_sha256"],
              observer_code_contract=design["observer_code_contract"], strict_raw_receipt_readback=True,
              original_B17_source_proof=design["original_B17_source_proof"]["source_inputs"],
              B17_source_stop_retained=True, tts_calls_at_screen=0, new_model_forwards=0,
              optimizer_updates=0, training_allowed=False, human_gold=False, threshold_search=False,
              all_prior_replication_sealed=True, replication_opened=False, promotion_allowed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    item = modes.add_parser("authorize"); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("guard"); item.add_argument("prepared_parent")
    modes.add_parser("history")
    item = modes.add_parser("verify-history"); item.add_argument("archives", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path); item.add_argument("archives", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("observe"); item.add_argument("provider", choices=qa.observers.PROVIDERS)
    item.add_argument("source", type=Path); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("screen"); item.add_argument("source", type=Path); item.add_argument("providers", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "authorize": lineage.authorize(args.out, args.prepared_parent)
    elif args.mode == "guard": lineage.execution_guard(args.prepared_parent)
    elif args.mode == "history": print("\n".join(required_archives()))
    elif args.mode == "verify-history": history(args.archives)
    elif args.mode == "generate": generate(args.out, args.archives, args.prepared_parent)
    elif args.mode == "observe": observe(args.provider, args.source, args.out, args.prepared_parent)
    else: screen(args.source, args.providers, args.out)
