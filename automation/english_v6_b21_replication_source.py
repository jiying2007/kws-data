"""B21 fixed replication sources, opened only by original primary PASS evidence.

All 96 frozen synthetic slots and 24 whole natural slots remain unchanged.
The original PASS, complete source history and all raw native returns are
reauthenticated without KWS inference, training or threshold selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_replication_source_ancestry as lineage
import english_v6_b21_source as primary
import english_v6_b21_replication_source_observe as observers
import english_v6_b21_natural_amendment_replication_natural_sources as natural

qa, B13 = lineage.qa, primary.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = "english-v6-b21-fresh-replication-source-v1"
BATCH = "english-v6-b21-120-original-replication-screen-v1"
FILES = primary.FILES


def source_plan(panel="replication"):
    qa.need(panel == "replication", "this entry point only exposes the fixed B21 replication plan")
    # Reuse the already frozen request bytes; do not rewrite their schema/hash.
    return primary.source_plan("replication")


def code_contract():
    frozen, actual = primary.code_contract(), observers.source_check()
    expected = dict(frozen, schema="english-v6-b21-replication-five-observer-contract-v1",
        observer_sha256=actual["observer_sha256"], source_indices=list(range(870, 894)),
        frozen_primary_observer_sha256=frozen["observer_sha256"])
    qa.need(actual == expected, "only replica identity wrapper changes; exact frozen native/parsing/admission contract")
    return actual


def validate_config(path):
    return primary.validate_config(path)


def source_binding():
    return dict(assessment_sha256=ASSESSMENT_SHA, source_ancestry_sha256=lineage.ANCESTRY_SHA,
        all_prior_replication_sealed=True, replication_opened=True,
        original_primary_pass_required=True, opening_sha256=lineage.opening_sha256(),
        training_allowed=False,
        promotion_allowed=False, threshold_search=False, new_kws_forwards=0,
        project_role="development", natural_upstream_partition="train-clean-100",
        natural_panel_amendment=True, natural_amendment=lineage.amendment_binding())


def generate(out, archives, corpus, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    code_contract()
    permission = lineage.require_authorization(archives)
    qa.need(permission["source_execution"] == identity, "actual same source event before generation")
    out, archives = Path(out), Path(archives)
    qa.need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "one fresh absolute source output")
    receipt = json.loads((archives / "HISTORY.json").read_bytes())
    seen, groups, people = lineage.validate_history(receipt)
    lineage.require_generation_permit(archives, prepared_parent)
    natural.verify_corpus(corpus)  # Source availability before the first TTS call.
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
    qa.need(json.loads((source / "SYNTHETIC_MANIFEST.json").read_bytes()) == synthetic,
            "preserved original renderer receipt exactly matches nested source document")
    qa.need(synthetic["schema"] == plan["schema"] and synthetic["panel"] == "replication"
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
                "every fixed whole source, no crop, padding, substitute or repeat")
        qa.need(json.loads((source / "starts" / (row["id"] + ".json")).read_bytes()) ==
                {"id": row["id"], "attempt": 1, "seed": row["seed"]}, "exact one request per source")
        facts = qa.pcm_tools.inspect_wav(source / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == v for k, v in facts.items()), "original WAV and PCM byte identity")
        chunks = bytearray()
        qa.need(bool(row["native_chunks"]), "whole original native chunks retained")
        for index, part in enumerate(row["native_chunks"]):
            qa.need(part["file"] == f"{index:03d}.s16le", "fixed contiguous native chunks")
            raw = (source / "native" / row["id"] / part["file"]).read_bytes()
            qa.need(len(raw) == 2 * part["samples"] and lineage.digest(raw) == part["native_pcm_sha256"], "native chunk bytes")
            chunks.extend(raw)
        qa.need(lineage.digest(chunks) == row["native_pcm_sha256"], "full original native source hash")
    return rows


def _read(directory):
    lineage.read_plan()
    directory = Path(directory)
    permission = lineage.require_authorization(directory)
    lineage.require_saved_generation(directory)
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
            "complete bounded mixed-source replication identity")
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


def read_source(directory):
    return _read(directory)[0]


def all_original_rows(directory):
    return _read(directory)[1]


def evaluation_preflight(directory):
    rows, all_rows, history, meta = _read(directory)
    return dict(rows=rows, all_original_rows=all_rows, history=history,
        source_execution=meta["source_execution"], assessment_sha256=ASSESSMENT_SHA,
        primary_quality=history["primary_quality"], history_sha256=qa.sha(Path(directory) / "HISTORY.json"))


def registration(source):
    rows, all_rows, history, meta = _read(source)
    contract = code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
        source_manifest_sha256=qa.sha(Path(source) / "MANIFEST.json"), source_rows=len(rows), max_calls=len(rows),
        **meta["source_execution"], **{k: contract[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256")})


def observe(provider, source, out, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    expected = registration(source)
    qa.need(all(expected[k] == v for k, v in identity.items()), "same actual source job before native load")
    return observers.observe(provider, source, out, read_rows=read_source, schema=BATCH, registration=expected)


def screen_document(source, providers):
    source = Path(source)
    rows, all_rows, history, meta = _read(source)
    texts, phones, coverage = observers.load_observations(source, providers, rows, BATCH, registration=registration(source))
    qa.need(coverage == dict.fromkeys(observers.PROVIDERS, len(rows)), "five valid native payloads per eligible whole original")
    decisions, accepted = [], defaultdict(set)
    for row in rows:
        sid = row["id"]
        if row["cohort"] == "synthetic":
            outcome = qa.admit(row, texts[sid], phones[sid])
            value = outcome["class"]
            qa.need(outcome["training_allowed"] is False and (value is None or value == row["requested_class"]), "unchanged synthetic source-admission policy")
            if value is not None: accepted[row["source_group_id"]].add(value)
            decisions.append(dict(row, observed_class=value, tier=outcome["tier"], reason=outcome["reason"], phone_margins=outcome["margins"]))
        else:
            all_obs = set(texts[sid]) == set(qa.policy.TEXT_MODELS) and set(phones[sid]) == set(qa.policy.PHONE_MODELS)
            heard = [qa.policy.norm(text) for text in texts[sid].values()]
            votes = heard.count(qa.policy.norm(row["transcript"]))
            veto = any(natural.BLOCK.intersection(text.upper().split()) for text in heard) or any(
                qa.policy.greedy_full_wake(value["logits"], value["vocab"]) for value in phones[sid].values())
            tier = "silver" if all_obs and not veto and votes == 3 else "weak" if all_obs and not veto and votes >= 2 else "unknown"
            decisions.append(dict(row, observed_class=0 if tier != "unknown" else None, tier=tier,
                reason="BLIND_ORDINARY_NO_WAKE" if tier != "unknown" else "UNKNOWN_NATURAL_SOURCE",
                exact_votes=votes, full_wake_veto=veto, native_text=texts[sid],
                phone_sha256={p: value["sha256"] for p, value in phones[sid].items()}))
    synthetic = [r for r in decisions if r["cohort"] == "synthetic"]
    admitted_natural = [r for r in decisions if r["cohort"] == "natural" and r["observed_class"] == 0]
    paired = sorted(group for group, classes in accepted.items() if classes == {0, 2})
    measured = dict(synthetic_positive=sum(r["observed_class"] == 2 for r in synthetic),
        synthetic_ordinary=sum(r["observed_class"] == 0 for r in synthetic), synthetic_paired=len(paired),
        natural_originals=len(admitted_natural), natural_speakers=len({r["speaker"] for r in admitted_natural}))
    minima = lineage.read_plan()["source_gate"]
    missing = {k: {"observed": measured[k], "min": v} for k, v in minima.items() if measured[k] < v}
    return dict(schema=BATCH, panel="replication", **source_binding(), source_execution=meta["source_execution"],
        source_manifest_sha256=qa.sha(source / "MANIFEST.json"), rows=decisions, coverage=coverage,
        source_rows=len(rows), selected_original_slots=120, original_quality_rejections=120-len(rows),
        source_coverage=measured, frozen_minima=minima, source_shortfall=missing, paired_source_groups=paired,
        unknown_count=sum(r["observed_class"] is None for r in decisions), tier_counts=dict(Counter(r["tier"] for r in decisions)),
        source_gate="INSUFFICIENT_SOURCE" if missing else "SOURCE_GATE_PASS",
        parser_contract_sha256=code_contract()["parser_contract_sha256"], observer_code_contract=code_contract(),
        strict_raw_receipt_readback=True, new_model_forwards=0, optimizer_updates=0, human_gold=False)


def verify_screen_contract(source, document, providers):
    actual = screen_document(source, providers)
    qa.need(actual == document and actual["source_gate"] == "SOURCE_GATE_PASS", "exact complete raw native readback and unchanged source minima")
    return actual


def screen(source, providers, output):
    qa.record(output, screen_document(source, providers))


def check():
    lineage.read_plan(); lineage.read_opening(require_bound=False); source_plan(); code_contract()
    validate_config(primary.lineage.previous.lineage.canary.previous.PLANS / "piper-config.json")
    print("PASS: B21 replication metadata and unchanged strict policy; zero source/model calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    item = modes.add_parser("guard"); item.add_argument("prepared_parent")
    item = modes.add_parser("authorize"); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("verify-corpus"); item.add_argument("corpus", type=Path); item.add_argument("archives", type=Path)
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
    elif args.mode == "verify-corpus":
        lineage.require_authorization(args.archives)
        natural.verify_corpus(args.corpus)
    elif args.mode == "verify-source": registration(args.source)
    elif args.mode == "generate": generate(args.out, args.archives, args.corpus, args.prepared_parent)
    elif args.mode == "consume-observer": lineage.consume_observer(args.provider, args.source, args.out, args.prepared_parent)
    elif args.mode == "observe": observe(args.provider, args.source, args.out, args.prepared_parent)
    else: screen(args.source, args.providers, args.out)
