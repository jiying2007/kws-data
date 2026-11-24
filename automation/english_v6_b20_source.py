"""New B20 full source distribution for the unchanged four B17 model files."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b20_source_ancestry as lineage
import english_v6_b20_source_observe as observers

qa, B13 = lineage.qa, lineage.canary.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = "english-v6-b20-b17-frozen-new-format-source-v1"
BATCH = "english-v6-b20-96-original-strict-source-screen-v1"
FILES = {'primary': '7e197c8cd15d8a738fb559db2858202089f3428d25e1a4529d1b0211b0958727', 'replication': '8ce1eab43499b5d3a8ad414637bdfb27719e0e01974c403f15e44b7bcc1b089e'}
history, required_archives = lineage.history, lineage.required_archives


def source_plan(panel="primary"):
    design = lineage.draft_plan()
    qa.need(panel in FILES, "only the two newly registered B20 metadata panels")
    raw = (PLANS / (panel + ".json")).read_bytes()
    qa.need(FILES[panel] is None and ASSESSMENT_SHA is None or lineage.sha(raw) == FILES[panel],
            "exact B20 source request file")
    plan = json.loads(raw)
    family = design[panel]["synthetic"]
    rows = plan["rows"]
    qa.need(plan["schema"] == SCHEMA and plan["panel"] == panel
            and plan["immutable_assessment_sha256"] == ASSESSMENT_SHA
            and plan["source_ancestry_sha256"] == lineage.BASE_ANCESTRY_SHA
            and plan["model"] == lineage.previous.read_plan()["source_model"]
            and plan["config_sha256"] == lineage.previous.read_plan()["config_sha256"]
            and plan["max_source_calls"] == plan["original_slots"] == len(rows) == 96
            and plan["maximum_native_observer_calls"] == 480
            and plan["expected_providers"] == list(observers.PROVIDERS)
            and plan["training_allowed"] is plan["promotion_allowed"] is plan["existing_replication_opened"] is False,
            "all fixed new original requests; no old replication or model authority")
    qa.need(len({r["id"] for r in rows}) == 96
            and {r["speaker_index"] for r in rows} == set(family["speaker_indices"])
            and {str(r["speaker_index"]): r["source_speaker_id"] for r in rows} == family["speaker_id_map"],
            "24 exact canonical voices and all 96 slots")
    seed_start = 20262300 if panel == "primary" else 20262400
    first = min(family["speaker_indices"])
    for index in family["speaker_indices"]:
        per_reader = [r for r in rows if r["speaker_index"] == index]
        qa.need(len(per_reader) == 4 and {r["cell_id"]: r["length_scale"] for r in per_reader}
                == {"mero-fast": 1.4, "mero-slow": 1.8, "question": 1.0, "request": 1.08},
                "fixed new positive rate pair; no third rate, rerender or substituted cell")
        for row in per_reader:
            target = 2 if row["cell_id"].startswith("mero-") else 0
            text = "Hey Mero." if target == 2 else "What time is it?" if row["cell_id"] == "question" else "Please set a timer."
            native = "[[heɪ mˈɛɹoʊ]]" if target == 2 else "[[wˈʌt tˈaɪm ɪz ɪt]]" if row["cell_id"] == "question" else text
            qa.need(row["id"] == f"b20-{panel}-{index}-{row['cell_id']}" and row["panel"] == panel
                    and row["seed"] == seed_start + index - first and row["requested_class"] == target
                    and row["requested_text"] == text and row["native_input"] == native
                    and row["source_group_id"] == "piper-libritts-" + row["source_speaker_id"]
                    and row["role"] == "development" and row["observed_class"] is None and row["training_allowed"] is False,
                    "unchanged exact phonetics and honest unobserved development request")
    return plan


def code_contract():
    expected = lineage.read_plan()["observer_code_contract"]
    actual = observers.source_check()
    qa.need(actual == expected, "reviewed five-observer implementation and unchanged native parser/admission")
    return actual


def validate_config(config, groups, people):
    B13.b12.verify_config(config)
    mapping = json.loads(config.read_bytes())["speaker_id_map"]
    design = lineage.draft_plan()
    old = lineage.previous.previous.lineage.read_plan()
    excluded = set(people) | set(old["replication"]["natural"]["speakers"])
    excluded.update(p.removeprefix("p") for p in old["replication"]["synthetic"]["speaker_id_map"].values())
    excluded.update(design["B19_all_source_exclusions"]["canonical_people"])
    excluded.update(design["replication"]["natural"]["speakers"])
    legacy = B13.b10.previous.old
    excluded.update(str(p) for p in (*legacy.OLD_PIPER_IDS, *legacy.OLD_LIBRI_IDS, *(p for _, p in legacy.NEW_VOICES)))
    assigned = set()
    for role in ("primary", "replication"):
        for index, person in design[role]["synthetic"]["speaker_id_map"].items():
            reader = person.removeprefix("p")
            qa.need(mapping.get(person) == int(index) and reader not in excluded and person not in assigned
                    and not any(re.search(r"(?<!\d)" + reader + r"(?!\d)", group) for group in groups),
                    "all real old exposure, every sealed reserve, canary and new natural reader excluded")
            assigned.add(person)
    qa.need(len(assigned) == 48, "disjoint new primary and newly reserved synthetic voices")


def source_binding():
    design = lineage.read_plan()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
        B19_feasibility_result_sha256=design["B19_original_proof"]["feasibility_result_sha256"],
        B19_original_artifacts=lineage.proof_locks(), B19_canary_source_admission_allowed=False,
        source_format=design["source_format"], B17_source_stop_retained=True, B18_source_stop_retained=True,
        all_prior_replication_sealed=True, replication_opened=False, promotion_allowed=False)


def generate(out, archives, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    code_contract()
    permission = lineage.require_authorization(archives)
    qa.need(permission["source_execution"] == identity, "actual B20 run/head/parent authorization before TTS")
    qa.need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "fresh absolute original source output")
    receipt = lineage.history_receipt(archives)
    seen, groups, people = lineage.validate_history(receipt)
    config = lineage.canary.previous.PLANS / "piper-config.json"
    validate_config(config, groups, people)
    B13.render_panel(out, source_plan("primary"), FILES["primary"], seen,
                     lambda path: validate_config(path, groups, people))
    shutil.copyfile(Path(archives) / "AUTHORIZATION.json", out / "AUTHORIZATION.json")
    lineage.retain_proof(archives, out / "B19_PROOF")
    qa.record(out / "HISTORY.json", receipt)
    path = out / "MANIFEST.json"
    document = json.loads(path.read_bytes())
    document.update(**source_binding(), source_execution=identity,
        history_receipt_sha256=qa.sha(out / "HISTORY.json"),
        source_authorization_sha256=qa.sha(out / "AUTHORIZATION.json"),
        tts_logical_originals=96, tts_public_synthesize_api_invocations=96,
        native_output_chunks=sum(len(row["native_chunks"]) for row in document["rows"]),
        internal_tts_model_forwards=None, internal_onnx_session_run_calls=None, internal_tts_forwards_instrumented=False,
        new_natural_source_calls=0, new_natural_observer_calls=0)
    # One local provenance augmentation after the unchanged renderer, before
    # artifact publication or any native observation.
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    read_source(out)


def validate_manifest(meta, receipt, permission):
    lineage.read_plan()
    plan = source_plan("primary")
    rows = meta["rows"]
    seen, groups, people = lineage.validate_history(receipt)
    qa.need(meta["schema"] == SCHEMA and meta["panel"] == "primary"
            and meta["registered_plan_file_sha256"] == FILES["primary"]
            and meta["plan_sha256"] == B13.b10.sha(B13.b10.canon(plan))
            and all(meta[k] == value for k, value in source_binding().items())
            and meta["source_execution"] == permission["source_execution"]
            and meta["tts_calls"] == meta["tts_logical_originals"] == meta["tts_public_synthesize_api_invocations"] == len(rows) == 96
            and meta["new_kws_forwards"] == meta["new_natural_source_calls"] == meta["new_natural_observer_calls"] == 0
            and meta["human_gold"] is meta["internal_tts_forwards_instrumented"] is False
            and meta["internal_tts_model_forwards"] is meta["internal_onnx_session_run_calls"] is None
            and meta["native_output_chunks"] == sum(len(row["native_chunks"]) for row in rows),
            "complete new original provenance and honest public source-call accounting")
    qa.need(len({r["pcm_sha256"] for r in rows}) == 96 and not ({r["pcm_sha256"] for r in rows} & seen)
            and not ({r["source_group_id"] for r in rows} & groups)
            and not ({r["source_speaker_id"].removeprefix("p") for r in rows} & people),
            "all full historical, unadmitted and canary originals excluded before any KWS")
    for row, frozen in zip(rows, plan["rows"], strict=True):
        qa.need(all(row[k] == value for k, value in frozen.items())
                and row["whole_native_resample"] is True and row["crop_calls"] == 0
                and row["basic_audio_ok"] is True and 3200 <= row["samples"] <= 240000,
                "all 96 fixed whole original source requests, without padding/crop/replacement")
    return rows


def read_source(directory):
    lineage.read_plan()
    directory = Path(directory)
    permission = lineage.require_authorization(directory)
    receipt = json.loads((directory / "HISTORY.json").read_bytes())
    meta = json.loads((directory / "MANIFEST.json").read_bytes())
    qa.need(meta["history_receipt_sha256"] == qa.sha(directory / "HISTORY.json")
            and meta["source_authorization_sha256"] == qa.sha(directory / "AUTHORIZATION.json"),
            "unchanged complete raw history and original B19 prerequisite")
    rows = validate_manifest(meta, receipt, permission)
    qa.need(json.loads((directory / "STARTED.json").read_bytes()) == {"plan_sha256": FILES["primary"], "one_shot": True},
            "one original source-generation start")
    for row in rows:
        qa.need(json.loads((directory / "starts" / (row["id"] + ".json")).read_bytes()) ==
                {"id": row["id"], "attempt": 1, "seed": row["seed"]}, "exactly one fixed source request per original")
        actual = qa.pcm_tools.inspect_wav(directory / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == value for k, value in actual.items()), "whole original WAV/PCM16 bytes")
        native = bytearray()
        qa.need(bool(row["native_chunks"]), "all native chunks retained")
        for index, chunk in enumerate(row["native_chunks"]):
            qa.need(chunk["file"] == f"{index:03d}.s16le", "original contiguous native chunk numbering")
            raw = (directory / "native" / row["id"] / chunk["file"]).read_bytes()
            qa.need(len(raw) == 2 * chunk["samples"] and lineage.sha(raw) == chunk["native_pcm_sha256"], "exact native chunk bytes")
            native.extend(raw)
        qa.need(lineage.sha(native) == row["native_pcm_sha256"], "whole native source before the single resampling")
    return rows


def registration(source):
    source = Path(source)
    read_source(source)
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    code = code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
        source_manifest_sha256=qa.sha(source / "MANIFEST.json"), source_rows=96, max_calls=96,
        B19_feasibility_result_sha256=meta["B19_feasibility_result_sha256"], **meta["source_execution"],
        **{k: code[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256")})


def evaluation_preflight(source):
    source = Path(source)
    rows = read_source(source)  # Includes the three original B19 ZIPs and full raw return recomputation.
    history_doc = json.loads((source / "HISTORY.json").read_bytes())
    lineage.validate_history(history_doc)
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    design = lineage.read_plan()
    return dict(rows=rows, history=history_doc, assessment_sha256=ASSESSMENT_SHA,
        source_execution=meta["source_execution"], history_sha256=qa.sha(source / "HISTORY.json"),
        b19_feasibility_status="SOURCE_FORMAT_FEASIBLE_REQUIRES_NEW_PREREGISTRATION",
        b19_result_sha256=design["B19_original_proof"]["feasibility_result_sha256"],
        b19_canary_pcm_sha256=design["B19_all_source_exclusions"]["all_PCM_sha256"],
        b19_canary_people=design["B19_all_source_exclusions"]["canonical_people"])


def observe(provider, source, out, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    expected = registration(source)
    qa.need(all(expected[k] == value for k, value in identity.items()), "same actual source run before any observer load")
    return observers.observe(provider, source, out, read_rows=read_source, schema=BATCH, registration=expected)


def screen_document(source, providers):
    source = Path(source)
    rows = read_source(source)
    texts, phones, coverage = observers.load_observations(source, providers, rows, BATCH, registration=registration(source))
    qa.need(coverage == dict.fromkeys(observers.PROVIDERS, 96), "five valid complete native payloads per original; no diagnostic exception")
    decisions, accepted = [], defaultdict(set)
    for row in rows:
        outcome = qa.admit(row, texts[row["id"]], phones[row["id"]])
        value = outcome["class"]
        qa.need(outcome["training_allowed"] is False and (value is None or value == row["requested_class"]), "unchanged B14 source admission")
        if value is not None: accepted[row["source_group_id"]].add(value)
        decisions.append(dict(row, observed_class=value, tier=outcome["tier"], reason=outcome["reason"], phone_margins=outcome["margins"]))
    counts = Counter(r["observed_class"] for r in decisions)
    paired = sorted(group for group, classes in accepted.items() if classes == {0, 2})
    minima = dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8)
    measured = dict(positive_originals=counts[2], ordinary_originals=counts[0], paired_source_groups=len(paired))
    missing = {key: {"observed": measured[key], "min": value} for key, value in minima.items() if measured[key] < value}
    design = lineage.read_plan()
    return dict(schema=BATCH, panel="primary", **source_binding(),
        source_manifest_sha256=qa.sha(source / "MANIFEST.json"), registered_plan_file_sha256=FILES["primary"],
        rows=decisions, coverage=coverage, source_coverage=measured, frozen_minima=minima, source_shortfall=missing,
        paired_source_groups=paired, source_families=24, unknown_count=counts[None],
        tier_counts=dict(Counter(r["tier"] for r in decisions)), source_gate="INSUFFICIENT_SOURCE" if missing else "SOURCE_GATE_PASS",
        parser_contract_sha256=design["parser_contract_sha256"], observer_code_contract=design["observer_code_contract"],
        strict_raw_receipt_readback=True, tts_calls_at_screen=0, new_model_forwards=0, optimizer_updates=0,
        training_allowed=False, human_gold=False, threshold_search=False)


def verify_screen_contract(source, screen_doc, providers):
    actual = screen_document(source, providers)
    qa.need(screen_doc == actual, "exact recomputed five-native raw evidence and every unchanged source-admission decision")
    qa.need(actual["source_gate"] == "SOURCE_GATE_PASS", "unchanged source minima must pass before KWS")
    return actual


def screen(source, providers, output):
    qa.record(output, screen_document(source, providers))


def check():
    design = lineage.draft_plan()
    source_plan("primary"); source_plan("replication")
    if ASSESSMENT_SHA is not None:
        lineage.read_plan(); code_contract()
    _, groups, people = lineage.previous.registered_history()
    groups.update(design["B19_all_source_exclusions"]["source_groups"])
    people.update(design["B19_all_source_exclusions"]["canonical_people"])
    validate_config(lineage.canary.previous.PLANS / "piper-config.json", groups, people)
    print("PASS: B20 fixed 96/96 new-format requests; new nine-reader reserve sealed; no source/model calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check"); modes.add_parser("history")
    item = modes.add_parser("guard"); item.add_argument("prepared_parent")
    item = modes.add_parser("authorize"); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("verify-history"); item.add_argument("archives", type=Path)
    item = modes.add_parser("verify-source"); item.add_argument("source", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path); item.add_argument("archives", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("observe"); item.add_argument("provider", choices=observers.PROVIDERS)
    item.add_argument("source", type=Path); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("screen"); item.add_argument("source", type=Path); item.add_argument("providers", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "check": check()
    elif args.mode == "history": print("\n".join(required_archives()))
    elif args.mode == "guard": lineage.execution_guard(args.prepared_parent)
    elif args.mode == "authorize": lineage.authorize(args.out, args.prepared_parent)
    elif args.mode == "verify-history": history(args.archives)
    elif args.mode == "verify-source": registration(args.source)
    elif args.mode == "generate": generate(args.out, args.archives, args.prepared_parent)
    elif args.mode == "observe": observe(args.provider, args.source, args.out, args.prepared_parent)
    else: screen(args.source, args.providers, args.out)
