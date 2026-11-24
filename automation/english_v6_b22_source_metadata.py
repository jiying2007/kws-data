"""Frozen B22 source registration; metadata only, with no model/audio imports.

The six original metadata resources are byte-pinned checkpoints, independently
authenticated during the B21 source/PRIMARY readbacks.  Every old exclusion and
every exposed B21 original survives this transition.  No reserve is reclaimed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PANELS = HERE / "english_v6_b22_panels"
DESIGN_PATH = "automation/english_v6_b22_union_retention_design.json"
DESIGN_SHA = "6a73fe3996397784e8f6c8df0e5d22692f6ba17d516c66236d5adc37d12a1964"
INPUTS = {
    "FULL_METADATA_CATALOG.json": "f05ee9d8f20dbf0f2747764a36c8cce4dfce3d1ae934987082409b8e40d5f846",
    "b21-assessment.json": "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc",
    "b21-complete-history.json": "404428caff8c76a06d5cbaf789bfd0725957501bb38bd8b103739db00fb4ea5f",
    "b21-natural-amendment-inputs.json": "67f944bd7775c56007ad92ff58d4a8f66e25e312b3ba0ebdfb8ffcf84a976210",
    "b21-source-manifest.json": "4c6dd87ec464d60bf57a454954e5aefcf93ca235131f10d9cafde98b6b5fe9ff",
    "piper-config.json": "2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d",
    "legacy-source-rosters.json": "7e28ca69070526bb7b155545596f16a971aac61dc5ee16767a2c2654ef3cc9f7",
}
PIPER = {
    "commit": "de3dcfdf1912bb49726dc4aa11c26017ce2ac62a",
    "config": "en/en_US/libritts/high/en_US-libritts-high.onnx.json",
    "config_bytes": 20163, "config_md5": "cb1564bb779f3b1f87e2de778adeb669",
    "dataset": "LibriTTS train-clean-360", "dataset_license": "CC-BY-4.0",
    "derived_rate": 16000, "engine_license": "GPL-3.0",
    "file": "en/en_US/libritts/high/en_US-libritts-high.onnx", "file_bytes": 136673811,
    "file_sha256": "9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f",
    "native_rate": 22050, "num_speakers": 904, "piper_tts_version": "1.8.0",
    "repo": "rhasspy/piper-voices",
}
PROVIDERS = ["qwen", "whisper", "parakeet", "xlsr", "lv60"]
BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
SALTS = {
    role: {kind: f"english-v6-b22-teacher-union-{role}-{kind}-20261124-v1"
           for kind in ("piper", "natural")}
    for role in ("primary", "replication")
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def read_inputs():
    result = {}
    for name, expected in INPUTS.items():
        raw = (PANELS / name).read_bytes()
        need(digest(raw) == expected, "unaltered authenticated metadata: " + name)
        result[name] = json.loads(raw)
    return result


def canonical(value):
    text = str(value)
    if text.startswith("p"):
        text = text[1:]
    need(re.fullmatch(r"[0-9]+", text) is not None, "canonical corpus speaker identity")
    return str(int(text))


def source_person(row):
    for key in ("source_speaker_id", "speaker", "speaker_id"):
        if row.get(key) is not None:
            return canonical(row[key])
    return canonical(row["source_group_id"].rsplit("-", 1)[-1])


def ordered(values):
    return sorted(set(values), key=lambda value: (int(value), value))


def rank(salt, category, identity):
    return digest((salt + "/" + category + "/" + identity).encode()), identity


def legacy_people(bundle, mapping):
    """Preserve explicit old rosters and the earlier conservative pool exclusion.

    B21's canonical_people alone is insufficient for moving backwards through
    the Piper index map: old piper-libritts-pNNNN groups were not all expanded
    into that field.  Full groups are independently normalized below as well.
    """
    need(bundle["schema"] == "english-v6-b22-authenticated-legacy-source-rosters-v1"
         and bundle["original_source_producer"]["sha256"]
             == "4cc1e427c26d771b8e75194b2d5bbd0b5399423f632b54f6ef6e7e583b1bc957"
         and bundle["old_conservative_index_reservations"] == [list(range(58)), list(range(400, 570))],
         "original B12 conservative exclusions remain excluded")
    people = {canonical(name) for name, index in mapping.items() if index < 58 or 400 <= index < 570}
    need(len(bundle["files"]) == 20, "all B12 through B21 explicit synthetic rosters")
    for name, item in bundle["files"].items():
        raw = item["text"].encode()
        need(len(raw) == item["bytes"] and digest(raw) == item["sha256"]
             and hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == item["git_blob_sha1"],
             "unchanged original source-producer roster bytes: " + name)
        if name.endswith(".json"):
            for row in json.loads(raw)["rows"]:
                speaker, index = row["source_speaker_id"], row["speaker_index"]
                need(mapping[speaker] == index, "historical canonical speaker matches full current map")
                people.add(canonical(speaker))
    return people


def catalog_rows(catalog):
    need(catalog["schema"] == "english-v6-b21-full-train-clean-100-text-metadata-v1"
         and catalog["metadata_only"] is True
         and catalog["corpus_bytes"] == 6387309499
         and catalog["corpus_sha256"] == "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
         "authenticated full corpus text catalog")
    for key in ("audio_headers_interpreted", "audio_members_extracted", "kws_forwards",
                "natural_decodes", "new_source_calls", "observer_calls", "optimizer_updates",
                "replacement_reader_draws", "source_panel_openings"):
        need(catalog[key] == 0, "metadata collection did not open or score sources")
    need(digest(catalog["speakers_text"].encode()) == catalog["speakers_sha256"]
         == "3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76",
         "whole official speaker census")
    census = set()
    for line in catalog["speakers_text"].splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        if fields[2] == "train-clean-100":
            census.add(canonical(fields[0]))
    paths = [item["path"] for item in catalog["transcripts"]]
    need(paths == sorted(set(paths)) and len(paths) == 585, "complete canonical transcript order")
    rows, seen, seen_readers, byte_count = defaultdict(list), set(), set(), len(catalog["speakers_text"].encode())
    for item in catalog["transcripts"]:
        match = re.fullmatch(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", item["path"])
        need(match is not None, "canonical transcript path")
        speaker, chapter, sp2, ch2 = match.groups()
        raw = item["text"].encode()
        need((speaker, chapter) == (sp2, ch2) and speaker in census
             and len(raw) == item["bytes"] and digest(raw) == item["sha256"],
             "full unchanged transcript bytes")
        seen_readers.add(speaker)
        byte_count += len(raw)
        for line in item["text"].splitlines():
            fields = line.split(" ", 1)
            need(len(fields) == 2, "original transcript line")
            ident, text = fields
            need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident) is not None and ident not in seen,
                 "unique original utterance identity")
            seen.add(ident)
            words = text.split(" ")
            if (4 <= len(words) <= 14 and not BLOCK.intersection(words)
                    and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words)):
                rows[speaker].append(dict(speaker=speaker, chapter=chapter, source_id=ident,
                    transcript=text, transcript_file_sha256=item["sha256"], transcript_member=item["path"]))
    need(seen_readers == census and len(census) == 251 and len(seen) == 28539
         and byte_count == catalog["metadata_bytes"] == 5886217,
         "full census and every original transcript, not a cherry-picked subset")
    return census, rows


def build_documents():
    inputs = read_inputs()
    old = inputs["b21-assessment.json"]
    amendment = inputs["b21-natural-amendment-inputs.json"]
    history = inputs["b21-complete-history.json"]
    manifest = inputs["b21-source-manifest.json"]
    config = inputs["piper-config.json"]
    catalog = inputs["FULL_METADATA_CATALOG.json"]
    need(manifest["panel"] == "primary" and manifest["selected_original_slots"] == 120
         and len(manifest["rows"]) == 120 and history["all_prior_replication_sealed"] is True,
         "complete exposed B21 source checkpoint including UNKNOWN and rejected originals")
    exposed = {source_person(row) for row in manifest["rows"]}
    need(len(exposed) == 36 and len({row["pcm_sha256"] for row in manifest["rows"]}) == 120,
         "all120 actual B21 originals from36 independent source people")
    reserves = {
        "B21_original_primary_natural": ordered(old["primary"]["natural"]["speakers"]),
        "B21_original_replication_natural": ordered(old["replication"]["natural"]["speakers"]),
        "B21_amended_primary_natural": ordered(amendment["selection"]["panels"]["primary"]["design"]["speakers"]),
        "B21_amended_replication_natural": ordered(amendment["selection"]["panels"]["replication"]["design"]["speakers"]),
        "B21_replication_synthetic": ordered(canonical(v) for v in old["replication"]["synthetic"]["speaker_id_map"].values()),
    }
    reserved = set().union(*(set(values) for values in reserves.values()))
    mapping = config["speaker_id_map"]
    historical_by_index = {index: name for name, index in mapping.items()}
    group_people = set()
    for group in history["all_source_groups"]:
        index_match = re.fullmatch(r"piper-libritts-index-(\d+)", group)
        match = re.fullmatch(r"(?:piper-libritts-p|librispeech-(?:train-clean-100-|test-clean-|dev-other-)?)(\d+)", group)
        if index_match:
            group_people.add(canonical(historical_by_index[int(index_match.group(1))]))
        elif match:
            group_people.add(canonical(match.group(1)))
        else:
            need(group.startswith("kokoro-v1_0-preset:"), "every nonnumeric source group is an explicit Kokoro preset")
    older_roster_people = legacy_people(inputs["legacy-source-rosters.json"], mapping)
    common = set(history["canonical_people"]) | group_people | older_roster_people | exposed | reserved
    synth_blocked = set(history["synthetic_excluded_people"]) | common
    natural_blocked = set(history["natural_excluded_people"]) | common | set(amendment["selection"]["base_excluded_people"])
    ancestry = dict(schema="english-v6-b22-complete-exposure-and-sealed-reserves-v1",
        inherited_history_sha256=INPUTS["b21-complete-history.json"],
        B21_original_source=dict(run_id=45378709327, run_head="a9b059e928021cb77248dd5e50617de3f5a2d60a",
            id=14911649527, bytes=30668376, sha256="9dbb738ba9f25af58e367f83a383d234a1fb138816bcdb624195778a9e5ce797"),
        B21_manifest_sha256=INPUTS["b21-source-manifest.json"],
        all_pcm_sha256=sorted(set(history["all_pcm_sha256"]) | {row["pcm_sha256"] for row in manifest["rows"]}),
        all_source_groups=sorted(set(history["all_source_groups"]) | {row["source_group_id"] for row in manifest["rows"]}),
        canonical_people=ordered(set(history["canonical_people"]) | group_people | exposed),
        synthetic_excluded_people=ordered(synth_blocked), natural_excluded_people=ordered(natural_blocked),
        normalized_historical_source_group_people=ordered(group_people),
        historical_explicit_and_conservative_roster_people=ordered(older_roster_people),
        historical_source_rosters_sha256=INPUTS["legacy-source-rosters.json"],
        legacy_Piper_index_aliases_resolved_through_frozen_config=True,
        additional_never_reclaimed_reserves=reserves, all_prior_replication_sealed=True,
        all_B21_120_raw_originals_added=True, original_B21_PRIMARY_verdict="NO_PROMOTION",
        original_producer_archive_complete=False,
        independent_complete_readback_with_separate_git_supplement_sha256="483be35deb1634bb2d60ffb5c6b539d7b7708cf96e2177b6eec75d5fd275d415",
        primary_producer_git_supplement_sha256="c1b7036ff4e96ce83b33db2b9932bb156ce299bfe13907834e2458b7bae72a50")
    need(config["num_speakers"] == len(mapping) == 904 and set(mapping.values()) == set(range(904))
         and len({canonical(value) for value in mapping}) == 904, "actual complete904 voice map")
    by_index = {index: name for name, index in mapping.items()}
    available_synth = [index for index, name in by_index.items() if canonical(name) not in synth_blocked]
    census, eligible = catalog_rows(catalog)
    available_natural = census - natural_blocked
    usable_natural = {speaker for speaker in available_natural if len(eligible[speaker]) >= 2}
    need(len(available_synth) >= 48 and len(usable_natural) >= 24, "two new independent complete panels exist")
    panels, used_synth, used_natural = {}, set(), set()
    for role in ("primary", "replication"):
        indices = sorted((i for i in available_synth if i not in used_synth),
                         key=lambda i: rank(SALTS[role]["piper"], "speaker", canonical(by_index[i])))[:24]
        used_synth.update(indices)
        readers = sorted(usable_natural - used_natural,
                         key=lambda speaker: rank(SALTS[role]["natural"], "speaker", speaker))[:12]
        used_natural.update(readers)
        roster = []
        for speaker in readers:
            roster.extend(sorted(eligible[speaker], key=lambda row: rank(SALTS[role]["natural"], "utterance", row["source_id"]))[:2])
        natural = dict(corpus_split="train-clean-100", project_role="development", training_allowed=False,
            opened=False, salt=SALTS[role]["natural"], speakers=readers, requested_speakers=12,
            utterances_per_speaker=2, source_slots=24, excluded_canonical_speakers=ordered(natural_blocked | (used_natural - set(readers))),
            available_census=ordered(available_natural), metadata_speakers_sha256=catalog["speakers_sha256"],
            full_metadata_catalog_sha256=INPUTS["FULL_METADATA_CATALOG.json"],
            upstream=old[role]["natural"]["upstream"], minimum_approved_speakers=8,
            minimum_approved_utterances=12, required_valid_observers_per_retained_original=5,
            utterance_selection=dict(rank_before_audio=True, no_replacement_after_quality_rejection=True,
                whole_original_only=True, minimum_words=4, maximum_words=14, maximum_native_decodes=24),
            selected_utterance_ids=[row["source_id"] for row in roster], roster=roster)
        panels[role] = dict(synthetic=dict(salt=SALTS[role]["piper"], speaker_indices=indices,
            speaker_id_map={str(i): by_index[i] for i in indices}, original_slots=96,
            seed_base=20262700 if role == "primary" else 20262800), natural=natural)
    all_selected_people = [canonical(by_index[i]) for i in used_synth] + list(used_natural)
    need(len(all_selected_people) == len(set(all_selected_people)) == 72,
         "both panels have72 distinct canonical people across synthetic and natural sources")
    assessment = dict(schema="english-v6-b22-prospective-teacher-union-assessment-v1",
        keyword=old["keyword"], canonical_ipa=old["canonical_ipa"],
        purpose="TRAIN-only two-condition teacher-union coverage extension with immutable B21 historical control, fresh primary and separately opened independent replication.",
        freeze_before_optimizer_and_new_source=True,
        ancestry_sha256=digest(encoded(ancestry)), metadata_validator_sha256=digest(Path(__file__).read_bytes()),
        metadata_inputs_sha256=INPUTS,
        training=dict(design_path=DESIGN_PATH, design_sha256=DESIGN_SHA, parameters=43876,
            shared_parent="B0", shared_parent_pt_sha256="1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7",
            new_challenger_updates=192, control="immutable_original_B21_challenger",
            control_pt_sha256="4947d017142f3b01f5dd4cf7706cc3c6978cf1a072ca022c26ea847ec0dc2f04",
            new_historical_control_forwards=0, new_teacher_forwards=0, new_training_audio=False,
            dev_cal_locked_forwards_in_training=0, threshold_search=False),
        model_roles=dict(B0="unchanged_B0", B9="unchanged_B9", control="original_B21_challenger", challenger="B22_union_retention_challenger"),
        primary=panels["primary"], replication=panels["replication"],
        source_gate=old["source_gate"], gains=old["gains"], quality_criteria=old["quality_criteria"],
        quality_assessment_sha256=old["quality_assessment_sha256"],
        quality_control_role="exact original B21 challenger as immutable historical control; numerical B14 decision unchanged",
        promotion_prerequisites="Fresh B22 primary PASS and untouched fresh B22 replication PASS using identical frozen weights, actual PT/ONNX/C11/Cortex-A32 numerical and event parity, and target SSC305 AFE/callback/FRR/FAR/latency evidence.",
        target=old["target"], false_alarm_reporting=old["false_alarm_reporting"],
        metadata_speakers_sha256=catalog["speakers_sha256"], source_format=old["source_format"],
        observer_scientific_contract={key: value for key, value in old["observer_code_contract"].items()
                                      if key not in ("schema", "observer_sha256", "source_indices")},
        maximum_public_Piper_synthesize_calls=96, maximum_natural_original_decodes=24,
        maximum_native_observer_calls=600, maximum_primary_KWS_forwards=1440,
        training_allowed_for_sources=False, threshold_search=False, promotion_allowed=False,
        replication_opened=False, all_prior_replication_sealed=True,
        primary_scoring_requires_frozen_actual_B22_weights_and_actual_source_gate_pass=True,
        replication_opening="Only authenticated actual B22 primary numerical PASS, followed by a new separately registered replication execution; all old reserves remain sealed.",
        source_only_generation_before_model_training_completion_allowed=False,
        execution_authorized=False, new_source_calls=0, new_observer_calls=0,
        metadata_selection=dict(algorithm="SHA256(salt + '/speaker/' + canonical_speaker), then canonical identity; first primary, then replica excluding primary; natural utterances SHA256(salt + '/utterance/' + original_id)",
            salts=SALTS, corpus_census=251, eligible_synthetic_voices=len(available_synth),
            available_natural_census=len(available_natural), eligible_natural_readers=len(usable_natural),
            synthetic_excluded_canonical_people=len(synth_blocked), natural_excluded_canonical_people=len(natural_blocked),
            metadata_only=True, selection_before_audio=True, selection_before_optimizer=True,
            source_replacements=0, reserves_reclaimed=0))
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
                rows.append(dict(id=f"b22-{role}-{index}-{cell}", speaker_index=index, source_speaker_id=speaker,
                    source_group_id="piper-libritts-" + speaker, role="development", panel=role, cell_id=cell,
                    cohort="synthetic", length_scale=scale, seed=synthetic["seed_base"] + ordinal,
                    requested_class=cls, requested_text=text, native_input=native, observed_class=None, training_allowed=False))
        documents[role + ".json"] = dict(schema="english-v6-b22-fresh-combined-source-v1", panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)), source_ancestry_sha256=digest(encoded(ancestry)),
            model=PIPER, config_sha256=INPUTS["piper-config.json"], max_source_calls=96,
            maximum_native_observer_calls=480, original_slots=96, expected_providers=PROVIDERS,
            source_format="fixed positive length-scale pair1.4/1.8 and ordinary1.0/1.08; no crop/pad/retry",
            source_admission_rule="unchanged full B14 five-observer policy", existing_replication_opened=False,
            training_allowed=False, promotion_allowed=False, rows=rows)
        documents["natural-" + role + ".json"] = panels[role]["natural"]
    return documents


def validate():
    documents = build_documents()
    for name, value in documents.items():
        need((PANELS / name).read_bytes() == encoded(value), "complete deterministic frozen registration: " + name)
    hashes = {"automation/english_v6_b22_panels/" + name: digest((PANELS / name).read_bytes())
              for name in sorted(set(INPUTS) | set(documents))}
    hashes["automation/english_v6_b22_source_metadata.py"] = digest(Path(__file__).read_bytes())
    return dict(schema="english-v6-b22-offline-metadata-validation-v1", status="PASS_METADATA_ONLY",
        hashes=hashes, selection=documents["assessment.json"]["metadata_selection"],
        source_calls=0, observer_calls=0, kws_forwards=0, optimizer_updates=0,
        source_execution_authorized=False, replica_opening_authorized=False, promotion_allowed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="initial local metadata freeze; refuses to overwrite")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.freeze:
        documents = build_documents()
        need(not any((PANELS / name).exists() for name in documents), "freeze only once; no regeneration of selected panels")
        for name, value in documents.items():
            (PANELS / name).write_bytes(encoded(value))
    report = validate()
    if args.out:
        args.out.write_bytes(encoded(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
