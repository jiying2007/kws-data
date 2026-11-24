"""Saved-only B20 primary source admission, with no execution guard emulation.

The body repeats the frozen primary's pure pre-Torch checks against its original
SPEC, FETCH and actual producer bytes. It never calls primary.preflight, changes
its globals, impersonates a GitHub run, loads a model, or invokes an observer.
"""
from __future__ import annotations
from contextlib import ExitStack
import io
import json
from pathlib import Path
import sys
import tempfile
import wave


def authenticate_import_roots(primary, producer):
    """A preloaded module may not redirect frozen relative resource reads."""
    known = [(Path(name).parts, name, digest) for name, digest in producer["primary"]["files"].items()]
    roots = set()
    for module in tuple(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        matched = [(parts, name, digest) for parts, name, digest in known
                   if path.parts[-len(parts):] == parts]
        if not matched:
            continue
        primary.need(len(matched) == 1, "unambiguous first-party module path")
        parts, name, digest = matched[0]
        primary.need(primary.sha(path.read_bytes()) == digest, "already imported frozen scientific module " + name)
        roots.add(path.parents[len(parts) - 1])
    for root in roots:
        for name, digest in {**producer["primary"]["files"], **producer["resources"]}.items():
            primary.need(primary.sha((root / name).read_bytes()) == digest,
                         "same authenticated scientific resources under the actual import root " + name)


def authenticate_sources(primary, inputs, spec, identity, producer, workflow, fetched):
    need, sha, helpers, original_zip = primary.need, primary.sha, primary.helpers, primary.original_zip
    archive_pins = primary.archive_pins
    validate_new_format, exact_admitted_join = primary.validate_new_format, primary.exact_admitted_join
    reject_overlap, person, score_identity = primary.reject_overlap, primary.person, primary.score_identity
    valid_hash = primary.valid_hash
    TRAINING_DESIGN, TRAINING_ASSESSMENT = primary.TRAINING_DESIGN, primary.TRAINING_ASSESSMENT
    B0_NPZ = primary.B0_NPZ
    SOURCE_SCHEMA, SCREEN_SCHEMA = primary.SOURCE_SCHEMA, primary.SCREEN_SCHEMA
    SOURCE_BRANCH, SOURCE_WORKFLOW = primary.SOURCE_BRANCH, primary.SOURCE_WORKFLOW
    PROVIDERS, SCHEMA, SPEC = primary.PROVIDERS, primary.SCHEMA, primary.SPEC
    authenticate_import_roots(primary, producer)
    old, decision, source, plan = helpers(spec)
    authenticate_import_roots(primary, producer)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as tmp:
        archives = {key: stack.enter_context(original_zip(inputs / (key + ".zip"), pin)) for key, pin in archive_pins(spec).items()}
        pair = archives["pair"]
        receipt = json.loads(pair.read("RESULT.json"))
        train = json.loads(pair.read("SOURCE.json"))
        old.validate_pair_receipt(receipt)
        need(train["parent_pt_sha256"] == spec["parent_pt_sha256"]
             and train["design_sha256"] == TRAINING_DESIGN and train["independent_assessment_sha256"] == TRAINING_ASSESSMENT,
             "original actual TRAIN provenance before Torch")
        need(len(train["base_train"]) == 79 and len(train["natural_train"]) == 9,
             "all original 79 base and 9 natural TRAIN sources before Torch")
        for arm in ("control", "challenger"):
            for ext in ("pt", "npz", "onnx"):
                member = old.STEMS[arm] + "." + ext
                raw = pair.read(member)
                need(receipt[arm]["files"][Path(member).name] == dict(bytes=len(raw), sha256=sha(raw)), "complete original paired model member")
                if ext == "pt":
                    need(sha(raw) == spec["model_pt_sha256"][arm], "same frozen trained arm")
        need(sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.pt")) == spec["model_pt_sha256"]["B0"]
             and sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.npz")) == B0_NPZ
             and sha(archives["B9"].read(old.common.b9.MODELS["B9_control"]["path"] + ".pt")) == spec["model_pt_sha256"]["B9"],
             "original baseline/teacher identity before Torch")
        directory = Path(tmp) / "synthetic"
        directory.mkdir()
        archives["synth_source"].extractall(directory)  # original_zip already rejects unsafe members.
        owner = source.evaluation_preflight(directory)  # Complete original B19 proof; no new model/source call.
        synthetic = owner["rows"]
        validate_new_format(synthetic)
        history = owner["history"]
        need(owner["assessment_sha256"] == spec["assessment_sha256"]
             and owner["b19_feasibility_status"] == "SOURCE_FORMAT_FEASIBLE_REQUIRES_NEW_PREREGISTRATION"
             and valid_hash(owner["b19_result_sha256"]), "actual re-read B19 PASS, never a CI-success proxy")
        execution = owner["source_execution"]
        pin = spec["inputs"]["synth_source"]
        meta = fetched["source_metadata"]["synth_source"]
        need(str(execution["run_id"]) == str(pin["run_id"]) and execution["run_head"] == pin["run_head"]
             and execution["run_attempt"] == 1 and execution["run_branch"] == "refs/heads/" + SOURCE_BRANCH
             and meta["run"]["head_branch"] == SOURCE_BRANCH and meta["run"]["path"] == SOURCE_WORKFLOW
             and meta["run"]["event"] == "push" and len(meta["commit"]["parents"]) == 1
             and meta["commit"]["parents"][0]["sha"] == execution["prepared_parent"]
             and [entry["filename"] for entry in meta["commit"]["files"]] == [SOURCE_WORKFLOW],
             "source MANIFEST/AUTH binds actual first public run/head/parent")
        manifest_raw = archives["synth_source"].read("MANIFEST.json")
        manifest = json.loads(manifest_raw)
        screen = json.loads(archives["synth_screen"].read("SCREEN.json"))
        planned = source.source_plan("primary")["rows"]
        need(manifest["rows"] == synthetic and manifest["schema"] == SOURCE_SCHEMA
             and manifest["panel"] == screen["panel"] == "primary" and screen["schema"] == SCREEN_SCHEMA
             and screen["source_manifest_sha256"] == sha(manifest_raw)
             and screen["assessment_sha256"] == spec["assessment_sha256"]
             and screen["coverage"] == dict.fromkeys(source.qa.observers.PROVIDERS, 96)
             and screen["tts_calls_at_screen"] == screen["new_model_forwards"] == screen["optimizer_updates"] == 0
             and screen["training_allowed"] is screen["promotion_allowed"] is screen["replication_opened"] is False
             and len(synthetic) == len(planned) == 96
             and {r["speaker_index"] for r in synthetic} == set(range(798, 822))
             and all(all(row[k] == v for k, v in frozen.items()) for row, frozen in zip(synthetic, planned, strict=True)),
             "all 96 exact new planned originals and original five-by-96 strict SCREEN")
        synthetic_providers = Path(tmp) / "synthetic-providers"
        natural_providers = Path(tmp) / "natural-providers"
        for role, target in (("synthetic", synthetic_providers), ("natural", natural_providers)):
            for provider in PROVIDERS:
                dest = target / provider; dest.mkdir(parents=True)
                archives[role + "_native_" + provider].extractall(dest)
        source.verify_screen_contract(directory, screen, synthetic_providers)
        selected_synth = exact_admitted_join(synthetic, screen["rows"], "synthetic")
        natural_manifest = json.loads(archives["natural_source"].read("MANIFEST.json"))
        natural_screen = json.loads(archives["natural_screen"].read("SCREEN.json"))
        natural = natural_manifest["rows"]
        natural_directory = Path(tmp) / "natural-source"
        natural_directory.mkdir()
        archives["natural_source"].extractall(natural_directory)
        reproduced = Path(tmp) / "original-natural-screen.json"
        old.natural.screen(natural_directory, natural_providers, reproduced)
        need(json.loads(reproduced.read_bytes()) == natural_screen,
             "all original natural native receipts reproduce the unchanged original SCREEN exactly; no new observation")
        selected_natural = exact_admitted_join(natural, natural_screen["decisions"], "natural")
        unchanged, natural_counts = old.verified_human(archives["natural_source"], archives["natural_screen"])
        need(selected_natural == unchanged and natural_counts == dict(natural_originals=21, natural_speakers=11)
             and len(natural) == 24 and sum(r["tier"] == "unknown" for r in natural_screen["decisions"]) == 3,
             "only exact originally admitted 21/11 natural; three UNKNOWN cannot substitute")
        for row in natural:
            raw = archives["natural_source"].read("wav/" + row["id"] + ".wav")
            with wave.open(io.BytesIO(raw)) as audio:
                need(audio.getparams()[:3] == (1, 2, 16000) and audio.getnframes() == row["samples"], "all 24 full original natural PCM")
                pcm = audio.readframes(audio.getnframes())
            need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "natural UNKNOWN raw bytes are also authenticated")
        reserve = {person(r) for r in source.source_plan("replication")["rows"]}
        reserve.update(str(int(p)) for p in plan["replication"]["natural"]["speakers"])
        reject_overlap(synthetic, natural, train["base_train"] + train["natural_train"], history,
                       set(owner["b19_canary_pcm_sha256"]), set(owner["b19_canary_people"]), reserve)
        groups = {}
        for row in selected_synth:
            groups.setdefault(row["source_group_id"], set()).add(row["observed_class"])
        pairs = sorted(k for k, values in groups.items() if values == {0, 2})
        counts = dict(synthetic_positive=sum(r["observed_class"] == 2 for r in selected_synth),
                      synthetic_ordinary=sum(r["observed_class"] == 0 for r in selected_synth),
                      synthetic_paired=len(pairs), **natural_counts)
        need(screen["paired_source_groups"] == pairs and screen["source_coverage"] == dict(
             positive_originals=counts["synthetic_positive"], ordinary_originals=counts["synthetic_ordinary"],
             paired_source_groups=counts["synthetic_paired"]), "exact original screened source denominators")
        gate, missing = old.source_gate(counts)
        need(gate == "SOURCE_GATE_PASS" and not missing and screen["source_gate"] == gate and not screen["source_shortfall"],
             "INSUFFICIENT_SOURCE: no model work is permitted")
    chosen = [(r, "synthetic") for r in selected_synth] + [(r, "natural") for r in selected_natural]
    admission = dict(schema=SCHEMA, identity=identity, assessment_sha256=spec["assessment_sha256"],
        registered_inputs_sha256=sha(SPEC.read_bytes()), producer_manifest_sha256=spec["producer_manifest_sha256"],
        workflow_sha256=sha(workflow), source_gate=gate, source_shortfall=missing, source_coverage=counts,
        original_source_rows=120, admitted_rows=[score_identity(r, c) for r, c in chosen],
        training_source_sha256=sha(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()),
        model_calls=0, B19_canary_excluded=True, promotion_allowed=False)
    inventory = dict(source_manifest_sha256=sha(manifest_raw), original_slots=96,
        all_pcm_sha256=sorted(r["pcm_sha256"] for r in synthetic),
        all_source_groups=sorted({r["source_group_id"] for r in synthetic}),
        canonical_people=sorted({person(r) for r in synthetic}, key=int))
    return dict(admission=admission, selected_rows=[dict(r, cohort=c) for r, c in chosen],
        all_original_rows=[dict(r, cohort="synthetic") for r in synthetic]
                          + [dict(r, cohort="natural") for r in natural],
        primary_source_inventory=inventory, training=train, plan=plan)
