"""Freeze B23 primary and sealed replica using only authenticated metadata.

This adds every original B22 source and all its reserved people to the complete
historical exclusions. It never reads audio, observations, or KWS results and
does not authorize B23 execution. Original B22 NO_PROMOTION must be separately
authenticated before any new scientific budget can be consumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANELS = HERE / "english_v6_b23_panels"
DESIGN_PATH = "automation/english_v6_b23_prefix_peak_design.json"
DESIGN_SHA = "8485e152f0ba93e22c510731dc4acd7823b36df033856b4145aa971aa96ebe56"
PREVIOUS = "automation/english_v6_b22_source_metadata.py"
PREVIOUS_SHA = "94ec4723da84bf6403acb2df4fa26cb2b2e9be232d8d9f9f118d79a74b61f58c"
MANIFEST = "research/english_v6_b22_source_20261124/initial_failure/source/MANIFEST.json"
ORIGIN = "automation/english_v6_b22_source_origin.json"
ARTIFACTS = "research/english_v6_b22_source_20261124/initial_failure/ARTIFACTS.json"
RUN = "research/english_v6_b22_source_20261124/initial_failure/RUN.json"
EXTRA_INPUTS = {
    MANIFEST: "24d2161e4e3696daee4fe72b56ee665cc0d5d099835f54139542c6f155e19b6c",
    ORIGIN: "c790abd7ff62d5aaeb3124a031e3ed6f7960cf115db939fc808d10583cd091fb",
    ARTIFACTS: "3a4e785b7e37c39f9a087c4b14605a66bbc6ae98c624cb60e456ccfc949366c1",
    RUN: "2bd7c791356d91f2170f27742b204e62ded8278a6a45d5fd20a11688c166db40",
    DESIGN_PATH: DESIGN_SHA,
}
SALTS = {
    role: {kind: f"english-v6-b23-prefix-peak-{role}-{kind}-20261124-v1"
           for kind in ("piper", "natural")}
    for role in ("primary", "replication")
}
CONTROL_PT = "9b1900312430c5c92dee7f47608fc2be866778387d4ed3baedb11f4bcda6a655"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def inputs():
    need(digest((ROOT / PREVIOUS).read_bytes()) == PREVIOUS_SHA,
         "unchanged complete B22 metadata validator")
    spec = importlib.util.spec_from_file_location("b23_original_b22_metadata", ROOT / PREVIOUS)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    prior_validation = previous.validate()
    hashes = dict(prior_validation["hashes"])
    extra = {}
    for name, expected in EXTRA_INPUTS.items():
        raw = (ROOT / name).read_bytes()
        need(digest(raw) == expected, "unaltered B23 metadata input: " + name)
        hashes[name] = expected
        extra[name] = json.loads(raw)
    old = json.loads((previous.PANELS / "assessment.json").read_bytes())
    history = json.loads((previous.PANELS / "ancestry.json").read_bytes())
    manifest, origin, run = extra[MANIFEST], extra[ORIGIN], extra[RUN]
    source_pin = origin["artifacts"]["source"]
    actual = [row for row in extra[ARTIFACTS]["artifacts"] if row["id"] == source_pin["id"]]
    need(len(actual) == 1 and actual[0]["digest"] == "sha256:" + source_pin["sha256"]
         and actual[0]["size_in_bytes"] == source_pin["bytes"]
         and actual[0]["workflow_run"]["id"] == source_pin["run_id"] == run["id"]
         and actual[0]["workflow_run"]["head_sha"] == source_pin["run_head"] == run["head_sha"]
         and run["run_attempt"] == 1 and run["repository"]["private"] is False,
         "original public B22 source artifact identity; failed observer outcome does not erase exposure")
    need(manifest["schema"] == "english-v6-b22-fresh-combined-source-v1"
         and manifest["panel"] == "primary" and manifest["selected_original_slots"] == 120
         and len(manifest["rows"]) == 120 and manifest["source_ancestry_sha256"] == old["ancestry_sha256"]
         and manifest["assessment_sha256"] == digest((previous.PANELS / "assessment.json").read_bytes())
         and manifest["history_receipt_sha256"] == digest((previous.PANELS / "ancestry.json").read_bytes())
         and origin["source_manifest_sha256"] == EXTRA_INPUTS[MANIFEST]
         and str(manifest["source_execution"]["run_id"]) == str(source_pin["run_id"])
         and manifest["source_execution"]["run_head"] == source_pin["run_head"],
         "all original B22 rows including rejected and unobserved sources")
    design = extra[DESIGN_PATH]
    need(design["historical_control"]["pt_sha256"] == CONTROL_PT
         and design["opening_condition"] == "only_after_actual_B22_NO_PROMOTION"
         and design["updates"] == 192 and design["parameters"] == 43876
         and design["contrast"] == "known_silent_prefix_softplus_blank_loss_time_mean_to_amax_only",
         "B23 is the separately registered conditional prefix-only intervention")
    return previous, old, history, manifest, source_pin, hashes


def build_documents():
    previous, old, history, manifest, source_pin, hashes = inputs()
    original = manifest["rows"]
    exposed = {previous.source_person(row) for row in original}
    expected = {previous.canonical(value) for value in old["primary"]["synthetic"]["speaker_id_map"].values()}
    expected.update(old["primary"]["natural"]["speakers"])
    reserved = {previous.canonical(value) for value in old["replication"]["synthetic"]["speaker_id_map"].values()}
    reserved.update(old["replication"]["natural"]["speakers"])
    need(exposed == expected and len(exposed) == len(reserved) == 36 and not exposed & reserved
         and len({row["id"] for row in original}) == len({row["pcm_sha256"] for row in original}) == 120
         and len({row["source_group_id"] for row in original}) == 36
         and sum(row["cohort"] == "synthetic" for row in original) == 96
         and sum(row["cohort"] == "natural" for row in original) == 24,
         "complete original120 and all72 B22 exposed or reserved canonical people")
    newly_excluded = exposed | reserved
    synth_blocked = set(history["synthetic_excluded_people"]) | newly_excluded
    natural_blocked = set(history["natural_excluded_people"]) | newly_excluded
    need(not set(history["all_pcm_sha256"]) & {row["pcm_sha256"] for row in original},
         "B22 original120 remain independent from prior PCM inventory")
    ancestry = dict(schema="english-v6-b23-complete-exposure-and-sealed-reserves-v1",
        inherited_history_sha256=old["ancestry_sha256"],
        inherited_assessment_sha256=digest((previous.PANELS / "assessment.json").read_bytes()),
        original_B22_source=source_pin, B22_manifest_sha256=EXTRA_INPUTS[MANIFEST],
        all_pcm_sha256=sorted(set(history["all_pcm_sha256"]) | {row["pcm_sha256"] for row in original}),
        all_source_groups=sorted(set(history["all_source_groups"]) | {row["source_group_id"] for row in original}),
        canonical_people=previous.ordered(set(history["canonical_people"]) | exposed),
        synthetic_excluded_people=previous.ordered(synth_blocked),
        natural_excluded_people=previous.ordered(natural_blocked),
        prior_explicit_and_conservative_roster_people=history["historical_explicit_and_conservative_roster_people"],
        prior_never_reclaimed_reserves=history["additional_never_reclaimed_reserves"],
        B22_exposed_primary_people=previous.ordered(exposed),
        B22_sealed_replication_people=previous.ordered(reserved),
        all_B22_120_raw_originals_added=True, all_prior_replication_sealed=True,
        legacy_Piper_index_aliases_resolved_through_frozen_config=True,
        B22_quality_results_read=False, B23_execution_authorized=False,
        metadata_input_sha256=dict(hashes))
    prior_inputs = previous.read_inputs()
    config, catalog = prior_inputs["piper-config.json"], prior_inputs["FULL_METADATA_CATALOG.json"]
    mapping = config["speaker_id_map"]
    by_index = {index: name for name, index in mapping.items()}
    available_synth = {previous.canonical(name): index for index, name in by_index.items()
                       if previous.canonical(name) not in synth_blocked}
    census, eligible = previous.catalog_rows(catalog)
    available_natural = census - natural_blocked
    usable_natural = {speaker for speaker in available_natural if len(eligible[speaker]) >= 2}
    need(len(available_synth) >= 48 and len(usable_natural) >= 24,
         "sufficient remaining people for two independent complete B23 panels")
    panels, selected = {}, set()
    for role in ("primary", "replication"):
        # One shared canonical-person set prevents overlap across source types,
        # not only overlap between two synthetic or two natural cohorts.
        speakers = sorted(set(available_synth) - selected,
                          key=lambda p: previous.rank(SALTS[role]["piper"], "speaker", p))[:24]
        need(len(speakers) == 24, "exact24 new synthetic source people")
        selected.update(speakers)
        indices = [available_synth[p] for p in speakers]
        readers = sorted(usable_natural - selected,
                         key=lambda p: previous.rank(SALTS[role]["natural"], "speaker", p))[:12]
        need(len(readers) == 12, "exact12 new natural readers, disjoint across both source types")
        selected.update(readers)
        roster = [row for p in readers for row in sorted(
            eligible[p], key=lambda row: previous.rank(SALTS[role]["natural"], "utterance", row["source_id"]))[:2]]
        natural = copy.deepcopy(old[role]["natural"])
        natural.update(opened=False, salt=SALTS[role]["natural"], speakers=readers, roster=roster,
            excluded_canonical_speakers=previous.ordered(natural_blocked | (selected - set(readers))),
            available_census=previous.ordered(available_natural),
            selected_utterance_ids=[row["source_id"] for row in roster])
        panels[role] = dict(synthetic=dict(salt=SALTS[role]["piper"], speaker_indices=indices,
            speaker_id_map={str(i): by_index[i] for i in indices}, original_slots=96,
            seed_base=20262900 if role == "primary" else 20263000), natural=natural)
    need(len(selected) == 72 and not selected & (synth_blocked & natural_blocked),
         "all72 B23 canonical people unique and independent of shared exclusions")
    assessment = copy.deepcopy(old)
    assessment.update(schema="english-v6-b23-prospective-prefix-peak-assessment-v1",
        purpose="TRAIN known-silent-prefix mean-to-amax only, immutable B22 historical control, fresh primary and sealed replica.",
        ancestry_sha256=digest(encoded(ancestry)), metadata_validator_sha256=digest(Path(__file__).read_bytes()),
        metadata_inputs_sha256=dict(hashes), primary=panels["primary"], replication=panels["replication"],
        training=dict(design_path=DESIGN_PATH, design_sha256=DESIGN_SHA, parameters=43876,
            shared_parent="B0", shared_parent_pt_sha256=old["training"]["shared_parent_pt_sha256"],
            new_challenger_updates=192, control="immutable_original_B22_challenger", control_pt_sha256=CONTROL_PT,
            new_historical_control_forwards=0, new_teacher_forwards=0, new_training_audio=False,
            dev_cal_locked_forwards_in_training=0, threshold_search=False,
            only_numerical_change="known_silent_prefix_softplus_time_mean_to_amax"),
        model_roles=dict(B0="unchanged_B0", B9="unchanged_B9", control="original_B22_challenger",
                         challenger="B23_prefix_peak_challenger"),
        quality_control_role="exact original B22 challenger as immutable historical control; numerical B14 decision unchanged",
        promotion_prerequisites="Fresh B23 primary PASS and untouched B23 replica PASS with identical frozen weights, actual PT/ONNX/C11/Cortex-A32 parity, and target SSC305 AFE/callback/FRR/FAR/latency evidence.",
        replication_opening="Only authenticated actual B23 primary numerical PASS opens its separately registered replica; all older reserves remain sealed.",
        B22_quality_results_read=False, original_B22_NO_PROMOTION_required_before_any_new_budget=True,
        primary_scoring_requires_frozen_actual_B23_weights_and_actual_source_gate_pass=True,
        metadata_selection=dict(algorithm="SHA256(salt + '/speaker/' + canonical_speaker), then canonical identity; primary synthetic, primary natural, replica synthetic, replica natural excluding all previously selected canonical people; utterances SHA256(salt + '/utterance/' + original_id)",
            salts=SALTS, corpus_census=251, eligible_synthetic_voices=len(available_synth),
            available_natural_census=len(available_natural), eligible_natural_readers=len(usable_natural),
            synthetic_excluded_canonical_people=len(synth_blocked), natural_excluded_canonical_people=len(natural_blocked),
            metadata_only=True, selection_before_audio=True, selection_before_optimizer=True,
            source_replacements=0, reserves_reclaimed=0, cross_source_type_identity_exclusion=True))
    assessment.pop("primary_scoring_requires_frozen_actual_B22_weights_and_actual_source_gate_pass")
    documents = {"ancestry.json": ancestry, "assessment.json": assessment}
    for role in ("primary", "replication"):
        synthetic, rows = panels[role]["synthetic"], []
        for ordinal, index in enumerate(synthetic["speaker_indices"]):
            speaker = by_index[index]
            for cell, scale, cls, text, native in (
                    ("mero-fast", 1.4, 2, "Hey Mero.", "[[heɪ mˈɛɹoʊ]]"),
                    ("mero-slow", 1.8, 2, "Hey Mero.", "[[heɪ mˈɛɹoʊ]]"),
                    ("question", 1.0, 0, "What time is it?", "[[wˈʌt tˈaɪm ɪz ɪt]]"),
                    ("request", 1.08, 0, "Please set a timer.", "Please set a timer.")):
                rows.append(dict(id=f"b23-{role}-{index}-{cell}", speaker_index=index, source_speaker_id=speaker,
                    source_group_id="piper-libritts-" + speaker, role="development", panel=role, cell_id=cell,
                    cohort="synthetic", length_scale=scale, seed=synthetic["seed_base"] + ordinal,
                    requested_class=cls, requested_text=text, native_input=native, observed_class=None, training_allowed=False))
        documents[role + ".json"] = dict(schema="english-v6-b23-fresh-combined-source-v1", panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)), source_ancestry_sha256=digest(encoded(ancestry)),
            model=previous.PIPER, config_sha256=previous.INPUTS["piper-config.json"], max_source_calls=96,
            maximum_native_observer_calls=480, original_slots=96, expected_providers=previous.PROVIDERS,
            source_format=old["source_format"], source_admission_rule="unchanged full B14 five-observer policy",
            existing_replication_opened=False, training_allowed=False, promotion_allowed=False, rows=rows)
        documents["natural-" + role + ".json"] = panels[role]["natural"]
    return documents, hashes


def validate():
    documents, hashes = build_documents()
    for name, value in documents.items():
        need((PANELS / name).read_bytes() == encoded(value), "complete deterministic frozen B23 metadata: " + name)
        hashes[(PANELS / name).relative_to(ROOT).as_posix()] = digest(encoded(value))
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    return dict(schema="english-v6-b23-offline-metadata-validation-v1", status="PASS_METADATA_ONLY",
        hashes=hashes, selection=documents["assessment.json"]["metadata_selection"],
        all_B22_120_originals_excluded=True, all_B22_72_people_excluded=True,
        B22_quality_results_read=False, source_calls=0, observer_calls=0, kws_forwards=0, optimizer_updates=0,
        source_execution_authorized=False, replica_opening_authorized=False, promotion_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="initial metadata-only freeze; refuses overwrite")
    args = parser.parse_args()
    if args.freeze:
        documents, _ = build_documents()
        need(not PANELS.exists(), "freeze only once; never redraw selected panels")
        PANELS.mkdir()
        for name, value in documents.items():
            (PANELS / name).write_bytes(encoded(value))
    print(json.dumps(validate(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
