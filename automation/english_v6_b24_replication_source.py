"""B24 fixed replica sources; original PRIMARY PASS is required before opening."""
from __future__ import annotations
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b24_source as original
from english_v6_b22_replication_reuse import inherit
inherit(original,globals())
import english_v6_b24_replication_source_ancestry as lineage
import english_v6_b24_replication_source_observe as observers
import english_v6_b24_replication_natural_sources as natural
import english_v6_b24_replication_capacity as capacity
SCHEMA='english-v6-b24-fresh-replication-source-v1'
BATCH='english-v6-b24-120-original-replication-screen-v1'


def source_plan(panel='replication'):
    qa.need(panel=='replication','only the fixed sealed replica metadata')
    return original.source_plan('replication')


def code_contract():
    return observers.source_check()


def source_binding():
    return dict(assessment_sha256=ASSESSMENT_SHA,source_ancestry_sha256=lineage.ANCESTRY_SHA,
        all_prior_replication_sealed=True,replication_opened=True,original_primary_pass_required=True,
        opening_sha256=lineage.opening_sha256(),training_allowed=False,promotion_allowed=False,
        threshold_search=False,new_kws_forwards=0,project_role='development',natural_upstream_partition='train-clean-100',
        transport_continuation=False,transport_plan_sha256=lineage.transport_plan_sha(),
        training_binding_sha256=lineage.training_binding_sha())

def generate(out, archives, corpus, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    code_contract()
    permission = lineage.require_authorization(archives)
    qa.need(permission["source_execution"] == identity, "actual same source event before generation")
    out, archives = Path(out), Path(archives)
    qa.need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "one fresh absolute source output")
    receipt = json.loads((archives / "HISTORY.json").read_bytes())
    seen, groups, people = lineage.validate_history(receipt)
    lineage.require_generation_permit(archives, corpus, prepared_parent)
    natural.verify_corpus(corpus)  # Complete fixed archive before the first real TTS call.
    B13.render_panel(out, source_plan(), FILES["replication"], seen, validate_config)
    original_manifest = out / "MANIFEST.json"
    synthetic = json.loads(original_manifest.read_bytes())
    # The reused renderer creates an immutable original receipt. Retain it
    # byte-for-byte under its own name, then create the combined receipt once.
    qa.need(not (out / "SYNTHETIC_MANIFEST.json").exists(), "fresh original synthetic receipt destination")
    original_manifest.rename(out / "SYNTHETIC_MANIFEST.json")
    lineage.retain_proof(archives, out)
    natural_meta = natural.extract(out, corpus, receipt)
    meta = dict(schema=SCHEMA, panel="replication", **source_binding(), source_execution=identity,
        synthetic=synthetic, synthetic_manifest_sha256=qa.sha(out / "SYNTHETIC_MANIFEST.json"),
        natural=natural_meta, rows=synthetic["rows"] + natural_meta["rows"],
        history_receipt_sha256=qa.sha(out / "HISTORY.json"), source_authorization_sha256=qa.sha(out / "AUTHORIZATION.json"),
        selected_original_slots=120, tts_calls=96, natural_original_decodes=24,
        observer_eligible_originals=96 + len(natural_meta["rows"]),
        tts_public_synthesize_api_invocations=96, internal_tts_model_forwards=None,
        internal_tts_forwards_instrumented=False, human_gold=False)
    qa.record(out / "MANIFEST.json", meta)
    read_source(out)

def verify_synthetic(source, synthetic):
    plan = source_plan()
    rows = synthetic["rows"]
    qa.need(json.loads((source / "SYNTHETIC_MANIFEST.json").read_bytes()) == synthetic
            and synthetic["schema"] == plan["schema"] and synthetic["panel"] == "replication"
            and synthetic["registered_plan_file_sha256"] == FILES["replication"]
            and synthetic["plan_sha256"] == B13.b10.sha(B13.b10.canon(plan))
            and synthetic["tts_calls"] == len(rows) == 96 and synthetic["new_kws_forwards"] == 0
            and synthetic["human_gold"] is False and synthetic["fresh_cpu_session_each_input"] is True,
            "all96 full synthetic originals from unchanged renderer")
    qa.need(json.loads((source / "STARTED.json").read_bytes()) == {"plan_sha256": FILES["replication"], "one_shot": True},
            "one fixed source-rendering start")
    for row, frozen in zip(rows, plan["rows"], strict=True):
        qa.need(all(row[k] == value for k, value in frozen.items())
                and row["whole_native_resample"] is True and row["crop_calls"] == 0
                and row["basic_audio_ok"] is True and 3200 <= row["samples"] <= 240000,
                "each frozen whole original without cropping or replacement")
        qa.need(json.loads((source / "starts" / (row["id"] + ".json")).read_bytes()) ==
                {"id": row["id"], "attempt": 1, "seed": row["seed"]}, "one public native request per original")
        facts = qa.pcm_tools.inspect_wav(source / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == v for k, v in facts.items()), "original full WAV/PCM bytes")
        chunks = bytearray()
        qa.need(bool(row["native_chunks"]), "whole native output retained")
        for index, part in enumerate(row["native_chunks"]):
            qa.need(part["file"] == f"{index:03d}.s16le", "contiguous original native chunks")
            raw = (source / "native" / row["id"] / part["file"]).read_bytes()
            qa.need(len(raw) == 2 * part["samples"] and lineage.digest(raw) == part["native_pcm_sha256"],
                    "original native chunk identity")
            chunks.extend(raw)
        qa.need(lineage.digest(chunks) == row["native_pcm_sha256"], "whole native output checksum")
    return rows

def _read(directory):
    lineage.read_plan()
    directory = Path(directory)
    capacity.source_members(directory)
    permission = lineage.require_authorization(directory)
    lineage.require_saved_consumption(directory)
    history = json.loads((directory / "HISTORY.json").read_bytes())
    seen, groups, people = lineage.validate_history(history)
    meta = json.loads((directory / "MANIFEST.json").read_bytes())
    qa.need(meta["schema"] == SCHEMA and meta["panel"] == "replication"
            and all(meta[k] == v for k, v in source_binding().items())
            and meta["source_execution"] == permission["source_execution"]
            and meta["history_receipt_sha256"] == qa.sha(directory / "HISTORY.json")
            and meta["source_authorization_sha256"] == qa.sha(directory / "AUTHORIZATION.json")
            and meta["selected_original_slots"] == 120 and meta["tts_calls"] == meta["tts_public_synthesize_api_invocations"] == 96
            and meta["synthetic_manifest_sha256"] == qa.sha(directory / "SYNTHETIC_MANIFEST.json")
            and meta["natural_original_decodes"] == 24 and meta["internal_tts_model_forwards"] is None
            and meta["internal_tts_forwards_instrumented"] is meta["human_gold"] is False,
            "complete bounded mixed-source primary identity")
    synthetic = verify_synthetic(directory, meta["synthetic"])
    retained_natural, all_natural = natural.read_source(directory, meta["natural"], history)
    rows, all_rows = synthetic + retained_natural, synthetic + all_natural
    qa.need(meta["rows"] == rows and meta["observer_eligible_originals"] == len(rows)
            and len(all_rows) == len({r["id"] for r in all_rows}) == len({r["pcm_sha256"] for r in all_rows}) == 120,
            "all120 unique raw originals retained and exact observer denominator")
    actual_people = {r["source_speaker_id"].removeprefix("p") if r["cohort"] == "synthetic" else r["speaker"] for r in all_rows}
    qa.need(len(actual_people) == 36 and not actual_people & people
            and not {r["pcm_sha256"] for r in all_rows} & seen
            and not {r["source_group_id"] for r in all_rows} & groups, "all new canonical readers and full original PCM disjoint from complete history")
    return rows, all_rows, history, meta


def evaluation_preflight(directory):
    rows,all_rows,history,meta=_read(directory)
    return dict(rows=rows,all_original_rows=all_rows,history=history,source_execution=meta['source_execution'],
        assessment_sha256=ASSESSMENT_SHA,history_sha256=qa.sha(Path(directory)/'HISTORY.json'),
        primary_quality=history['primary_quality'],training_binding_sha256=meta['training_binding_sha256'],
        transport_plan_sha256=meta['transport_plan_sha256'])


def check():
    lineage.read_opening(False)
    lineage.read_plan();source_plan();code_contract()
    validate_config(PIPER_CONFIG)
    print('PASS: sealed B24 replica metadata; zero new audio, observations or KWS forwards')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    item = modes.add_parser("guard"); item.add_argument("prepared_parent")
    item = modes.add_parser("authorize"); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("verify-corpus"); item.add_argument("corpus", type=Path)
    for mode in ("transport", "consume"):
        item = modes.add_parser(mode); item.add_argument("corpus", type=Path); item.add_argument("archives", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("verify-source"); item.add_argument("source", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path); item.add_argument("archives", type=Path)
    item.add_argument("corpus", type=Path); item.add_argument("prepared_parent")
    for mode in ("consume-observer", "observe"):
        item = modes.add_parser(mode); item.add_argument("provider", choices=observers.PROVIDERS)
        item.add_argument("source", type=Path); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("screen"); item.add_argument("source", type=Path); item.add_argument("providers", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "check": check()
    elif args.mode == "guard": lineage.execution_guard(args.prepared_parent)
    elif args.mode == "authorize": lineage.authorize(args.out, args.prepared_parent)
    elif args.mode == "verify-corpus": natural.verify_corpus(args.corpus)
    elif args.mode == "transport":
        from english_v6_b24_replication_transport import download
        download(args.corpus, args.archives, args.prepared_parent)
    elif args.mode == "consume": lineage.consumption(args.archives, args.corpus, args.prepared_parent)
    elif args.mode == "verify-source": registration(args.source)
    elif args.mode == "generate": generate(args.out, args.archives, args.corpus, args.prepared_parent)
    elif args.mode == "consume-observer": lineage.consume_observer(args.provider, args.source, args.out, args.prepared_parent)
    elif args.mode == "observe": observe(args.provider, args.source, args.out, args.prepared_parent)
    else: screen(args.source, args.providers, args.out)
