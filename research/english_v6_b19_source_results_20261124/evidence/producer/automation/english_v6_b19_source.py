"""One fixed 48-original canary; never a KWS training or evaluation source."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b19_source_ancestry as lineage
import english_v6_b19_source_observe as observer

previous = lineage.previous
qa, B13 = previous.qa, previous.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = "english-v6-b19-source-feasibility-originals-v1"
BATCH = "english-v6-b19-parakeet-source-feasibility-v1"
PLAN_SHA = "c5d17c4902c827a69048c8fd1f91a0d8eded3abe093707d9611d4f80e7a31f34"
history, required_archives = lineage.history, lineage.required_archives


def source_plan(panel="feasibility"):
    design = lineage.read_plan()
    qa.need(panel == "feasibility", "only the new fixed diagnostic may execute; future cohorts remain sealed")
    raw = (PLANS / "feasibility.json").read_bytes()
    qa.need(lineage.sha(raw) == PLAN_SHA, "48 prospective whole source requests")
    plan = json.loads(raw)
    qa.need(plan["schema"] == SCHEMA and plan["panel"] == "feasibility"
            and plan["immutable_assessment_sha256"] == ASSESSMENT_SHA
            and plan["source_ancestry_sha256"] == lineage.BASE_ANCESTRY_SHA
            and plan["model"] == design["source_model"] and plan["config_sha256"] == design["config_sha256"]
            and plan["max_source_calls"] == plan["max_native_observer_calls"] == len(plan["rows"]) == 48
            and plan["diagnostic_only"] is True and plan["all_canary_identities_and_PCM_permanently_excluded"] is True
            and all(plan[key] is False for key in ("source_admission_eligible", "training_allowed", "evaluation_allowed",
                     "promotion_allowed", "future_source_execution_allowed")), "only bounded source-format feasibility")
    rows = plan["rows"]
    qa.need(len({r["id"] for r in rows}) == 48 and {r["speaker_index"] for r in rows} == set(range(790, 798))
            and {str(r["speaker_index"]): r["source_speaker_id"] for r in rows} == design["source_maps"]["feasibility"],
            "all eight exact canonical Piper identities frozen")
    for index in range(790, 798):
        family = [row for row in rows if row["speaker_index"] == index]
        qa.need(len(family) == 6 and {r["cell_id"] for r in family} == set(observer.CELLS), "all six cells for every canary reader")
        for row in family:
            arm, scale, target = observer.CELLS[row["cell_id"]]
            qa.need((row["feasibility_arm"], row["length_scale"], row["requested_class"]) == (arm, scale, target)
                    and row["seed"] == 20262200 + index - 790 and row["role"] == "source-feasibility"
                    and row["panel"] == "feasibility" and row["observed_class"] is None
                    and all(row[key] is False for key in ("training_allowed", "evaluation_allowed", "admission_eligible")),
                    "fixed paired rates and permanent diagnostic-only source role")
    return plan


def code_contract():
    actual = observer.code_contract()
    qa.need(actual == lineage.read_plan()["observer_code_contract"], "frozen reviewed one-observer canary implementation")
    return actual


def validate_config(config, groups, people):
    B13.b12.verify_config(config)
    mapping = json.loads(config.read_bytes())["speaker_id_map"]
    design = lineage.read_plan()
    b18 = previous.lineage.read_plan()
    excluded = set(people) | set(b18["replication"]["natural"]["speakers"])
    excluded.update(person.removeprefix("p") for person in b18["replication"]["synthetic"]["speaker_id_map"].values())
    old = B13.b10.previous.old
    excluded.update(str(person) for person in (*old.OLD_PIPER_IDS, *old.OLD_LIBRI_IDS, *(p for _, p in old.NEW_VOICES)))
    assigned = set()
    for role in ("feasibility", "future_primary", "future_replication"):
        for index, person in design["source_maps"][role].items():
            digits = person.removeprefix("p")
            qa.need(mapping.get(person) == int(index) and digits not in excluded and person not in assigned
                    and not any(re.search(r"(?<!\d)" + digits + r"(?!\d)", group) for group in groups),
                    "canonical source was never old TRAIN/DEV/reserved or another new cohort")
            assigned.add(person)
    qa.need(len(assigned) == 56, "eight permanently excluded canary readers plus disjoint unopened 24/24 future voices")


def generate(out, archives, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    code_contract()
    permission = lineage.require_authorization(archives)
    qa.need(permission["source_execution"] == identity, "actual authorization must match this run/head/parent before TTS")
    qa.need(out.is_absolute() and not out.exists() and out.parent.is_dir(), "fresh absolute source destination before TTS")
    receipt = lineage.history_receipt(archives)
    seen, groups, people = lineage.validate_history(receipt)
    validate_config(previous.PLANS / "piper-config.json", groups, people)
    B13.render_panel(out, source_plan(), PLAN_SHA, seen, lambda config: validate_config(config, groups, people))
    shutil.copyfile(Path(archives) / "AUTHORIZATION.json", out / "AUTHORIZATION.json")
    qa.record(out / "HISTORY.json", receipt)
    path = out / "MANIFEST.json"
    document = json.loads(path.read_bytes())
    document.update(assessment_sha256=ASSESSMENT_SHA, source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
        source_execution=identity, history_receipt_sha256=qa.sha(out / "HISTORY.json"),
        source_authorization_sha256=qa.sha(out / "AUTHORIZATION.json"),
        diagnostic_only=True, source_admission_eligible=False, evaluation_allowed=False, training_allowed=False,
        promotion_allowed=False, future_source_execution_allowed=False, all_prior_replication_sealed=True,
        B17_source_failure_preserved=True, B18_source_STOP_preserved=True,
        planned_permanent_canonical_exclusions=sorted(p.removeprefix("p") for p in lineage.read_plan()["source_maps"]["feasibility"].values()),
        generated_permanent_PCM_exclusions=sorted(row["pcm_sha256"] for row in document["rows"]),
        tts_logical_originals=48, tts_public_synthesize_api_invocations=48,
        native_output_chunks=sum(len(row["native_chunks"]) for row in document["rows"]),
        internal_tts_model_forwards=None, internal_onnx_session_run_calls=None, internal_tts_forwards_instrumented=False)
    # The old renderer's provisional manifest gets this one local provenance
    # augmentation before publication. It is then immutable for all observers.
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    read_source(out)  # All 48 bounds must pass before any observer job can begin.


def validate_manifest(meta, receipt, permission):
    plan = source_plan()
    rows = meta["rows"]
    seen, groups, people = lineage.validate_history(receipt)
    qa.need(meta["schema"] == SCHEMA and meta["panel"] == "feasibility"
            and meta["registered_plan_file_sha256"] == PLAN_SHA
            and meta["plan_sha256"] == B13.b10.sha(B13.b10.canon(plan))
            and meta["assessment_sha256"] == ASSESSMENT_SHA
            and meta["source_ancestry_sha256"] == lineage.BASE_ANCESTRY_SHA
            and meta["source_execution"] == permission["source_execution"]
            and meta["tts_calls"] == meta["tts_logical_originals"] == meta["tts_public_synthesize_api_invocations"] == len(rows) == 48
            and meta["new_kws_forwards"] == 0 and meta["human_gold"] is False
            and all(meta[key] is False for key in ("source_admission_eligible", "evaluation_allowed", "training_allowed",
                    "promotion_allowed", "future_source_execution_allowed", "internal_tts_forwards_instrumented"))
            and all(meta[key] is True for key in ("diagnostic_only", "all_prior_replication_sealed",
                    "B17_source_failure_preserved", "B18_source_STOP_preserved"))
            and meta["internal_tts_model_forwards"] is None and meta["internal_onnx_session_run_calls"] is None
            and meta["native_output_chunks"] == sum(len(row["native_chunks"]) for row in rows),
            "48 fixed diagnostic originals and honest public-source API accounting")
    qa.need(len({row["pcm_sha256"] for row in rows}) == 48
            and not ({row["pcm_sha256"] for row in rows} & seen)
            and not ({row["source_group_id"] for row in rows} & groups)
            and not ({row["source_speaker_id"].removeprefix("p") for row in rows} & people)
            and meta["planned_permanent_canonical_exclusions"] == sorted(p.removeprefix("p") for p in
                lineage.read_plan()["source_maps"]["feasibility"].values())
            and meta["generated_permanent_PCM_exclusions"] == sorted(row["pcm_sha256"] for row in rows),
            "every original canary identity and PCM stays separate and permanently excluded")
    for row, frozen in zip(rows, plan["rows"], strict=True):
        qa.need(all(row[key] == value for key, value in frozen.items())
                and row["whole_native_resample"] is True and row["crop_calls"] == 0
                and row["basic_audio_ok"] is True and 3200 <= row["samples"] <= 240000
                and (row["requested_class"] != 2 or 6400 <= row["samples"] <= 28672),
                "all 48 original bounds, including baseline, pass before any observation")
    return rows


def read_source(directory):
    directory = Path(directory)
    permission = lineage.require_authorization(directory)
    receipt = json.loads((directory / "HISTORY.json").read_bytes())
    meta = json.loads((directory / "MANIFEST.json").read_bytes())
    qa.need(meta["history_receipt_sha256"] == qa.sha(directory / "HISTORY.json")
            and meta["source_authorization_sha256"] == qa.sha(directory / "AUTHORIZATION.json"),
            "unchanged complete raw history and original STOP authorization")
    rows = validate_manifest(meta, receipt, permission)
    qa.need(json.loads((directory / "STARTED.json").read_bytes()) ==
            {"plan_sha256": PLAN_SHA, "one_shot": True}, "one original source-generation start")
    for row in rows:
        qa.need(json.loads((directory / "starts" / (row["id"] + ".json")).read_bytes()) ==
                {"id": row["id"], "attempt": 1, "seed": row["seed"]}, "one fixed diagnostic source request")
        facts = qa.pcm_tools.inspect_wav(directory / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[key] == value for key, value in facts.items()), "unchanged complete original PCM16")
        native_pcm = bytearray()
        qa.need(bool(row["native_chunks"]), "all original native chunks retained")
        for index, chunk in enumerate(row["native_chunks"]):
            qa.need(chunk["file"] == f"{index:03d}.s16le", "original native chunk sequence")
            raw = (directory / "native" / row["id"] / chunk["file"]).read_bytes()
            qa.need(len(raw) == 2 * chunk["samples"] and lineage.sha(raw) == chunk["native_pcm_sha256"], "original native TTS bytes")
            native_pcm.extend(raw)
        qa.need(lineage.sha(native_pcm) == row["native_pcm_sha256"], "whole native source before the single resample")
    observer.validate_rows(rows)
    return rows


def registration(source):
    source = Path(source)
    read_source(source)
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    code = code_contract()
    return dict(assessment_sha256=ASSESSMENT_SHA, source_schema=SCHEMA, batch_schema=BATCH,
        source_manifest_sha256=qa.sha(source / "MANIFEST.json"), source_rows=48, max_calls=48,
        diagnostic_only=True, source_admission_eligible=False, **meta["source_execution"],
        observer_sha256=code["observer_sha256"], asr_adapter_sha256=code["asr_adapter_sha256"],
        parser_contract_sha256=code["parser_contract_sha256"])


def observe(source, out, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    expected = registration(source)
    qa.need(all(expected[key] == value for key, value in identity.items()), "same original diagnostic source execution before model load")
    return observer.observe(source, out, read_rows=read_source, registration=expected)


def readback(source, evidence, out):
    result = observer.readback(source, evidence, read_source, registration(source))
    qa.need(result["source_admission_eligible"] is False and result["KWS_evaluation_allowed"] is False
            and result["promotion_allowed"] is False, "feasibility cannot become model authority")
    qa.record(out, result)
    qa.need(result["prospective_pair_feasible"] is True,
            "source-format pair not qualified; diagnostic evidence retained, no retry or third rate")


def check():
    source_plan()
    code_contract()
    _, groups, people = lineage.registered_history()
    validate_config(previous.PLANS / "piper-config.json", groups, people)
    qa.need(len(required_archives()) == 44, "all original old/B17/B18 archives, no source substitution")
    print("PASS: fixed 48-original/8-reader diagnostic; future 24/24 identities sealed; zero new source or model calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check"); modes.add_parser("history")
    item = modes.add_parser("guard"); item.add_argument("prepared_parent")
    item = modes.add_parser("authorize"); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("verify-history"); item.add_argument("archives", type=Path)
    item = modes.add_parser("verify-source"); item.add_argument("source", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path); item.add_argument("archives", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("observe"); item.add_argument("source", type=Path); item.add_argument("out", type=Path); item.add_argument("prepared_parent")
    item = modes.add_parser("readback"); item.add_argument("source", type=Path); item.add_argument("evidence", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "check": check()
    elif args.mode == "history": print("\n".join(required_archives()))
    elif args.mode == "guard": lineage.execution_guard(args.prepared_parent)
    elif args.mode == "authorize": lineage.authorize(args.out, args.prepared_parent)
    elif args.mode == "verify-history": history(args.archives)
    elif args.mode == "verify-source": registration(args.source)
    elif args.mode == "generate": generate(args.out, args.archives, args.prepared_parent)
    elif args.mode == "observe": observe(args.source, args.out, args.prepared_parent)
    else: readback(args.source, args.evidence, args.out)
