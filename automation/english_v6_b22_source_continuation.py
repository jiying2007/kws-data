"""Complete original B22 sources with only the three zero-call text observers resumed.

No audio is generated or decoded; both completed phone observer artifacts remain
byte-identical. The original source identity and continuation identity coexist.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_source as original_source
import english_v6_b22_source_continuation_ancestry as lineage
import english_v6_b22_source_continuation_observe as observers
import english_v6_b22_natural_sources as natural

qa, B13 = lineage.qa, original_source.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA, BATCH, FILES = original_source.SCHEMA, original_source.BATCH, original_source.FILES
source_plan, validate_config = original_source.source_plan, original_source.validate_config


def code_contract():
    return observers.source_check()


def source_binding():
    return dict(original_source.source_binding(), observer_continuation=True,
                origin_binding_sha256=lineage.origin_binding_sha())


def _read(directory):
    directory = Path(directory)
    continuation = lineage.require_continuation(directory)
    rows, all_rows, history, meta = original_source._read(directory)
    qa.need(meta['source_execution'] == continuation['original_source_execution'],
            'original audio generation identity remains exact and separate')
    return rows, all_rows, history, dict(meta,
        original_source_execution=meta['source_execution'],
        source_execution=continuation['source_execution'],
        continuation_manifest_sha256=qa.sha(directory / 'CONTINUATION.json'))

def read_source(directory):
    return _read(directory)[0]


def all_original_rows(directory):
    return _read(directory)[1]


def evaluation_preflight(directory):
    rows, all_rows, history, meta = _read(directory)
    return dict(rows=rows, all_original_rows=all_rows, history=history,
        source_execution=meta["source_execution"], original_source_execution=meta["original_source_execution"],
        continuation_manifest_sha256=meta["continuation_manifest_sha256"], assessment_sha256=ASSESSMENT_SHA,
        history_sha256=qa.sha(Path(directory) / "HISTORY.json"),
        training_binding_sha256=meta["training_binding_sha256"], transport_plan_sha256=meta["transport_plan_sha256"])


def registration(source):
    rows, all_rows, history, meta = _read(source)
    contract = code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
        source_manifest_sha256=qa.sha(Path(source) / "MANIFEST.json"), source_rows=len(rows), max_calls=len(rows),
        original_source_execution=meta["original_source_execution"],
        continuation_manifest_sha256=meta["continuation_manifest_sha256"],
        origin_binding_sha256=lineage.origin_binding_sha(),
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
    return dict(schema=BATCH, panel="primary", **source_binding(), source_execution=meta["source_execution"],
        original_source_execution=meta["original_source_execution"],
        continuation_manifest_sha256=meta["continuation_manifest_sha256"],
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
    original_source.check()
    code_contract()
    from english_v6_b22_source_dynamic_preflight import check as dynamic_check
    dynamic_check()
    print('PASS: B22 text-only continuation; complete dynamic imports; no native calls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('check')
    item = modes.add_parser('guard'); item.add_argument('prepared_parent')
    item = modes.add_parser('fetch-origin'); item.add_argument('directory', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('consume'); item.add_argument('directory', type=Path); item.add_argument('output', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('reuse')
    for name in ('directory', 'source', 'phones', 'consumed'): item.add_argument(name, type=Path)
    item.add_argument('prepared_parent')
    item = modes.add_parser('verify-source'); item.add_argument('source', type=Path)
    for mode in ('consume-observer', 'observe'):
        item = modes.add_parser(mode); item.add_argument('provider', choices=lineage.PROVIDERS)
        item.add_argument('source', type=Path); item.add_argument('output', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('screen'); item.add_argument('source', type=Path); item.add_argument('providers', type=Path); item.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.mode == 'check': check()
    elif args.mode == 'guard': lineage.execution_guard(args.prepared_parent)
    elif args.mode == 'fetch-origin': lineage.fetch_origin(args.directory, args.prepared_parent)
    elif args.mode == 'consume': lineage.consume(args.directory, args.output, args.prepared_parent)
    elif args.mode == 'reuse': lineage.reuse(args.directory, args.source, args.phones, args.consumed, args.prepared_parent)
    elif args.mode == 'verify-source': registration(args.source)
    elif args.mode == 'consume-observer': lineage.consume_observer(args.provider, args.source, args.output, args.prepared_parent)
    elif args.mode == 'observe': observe(args.provider, args.source, args.output, args.prepared_parent)
    else: screen(args.source, args.providers, args.output)
