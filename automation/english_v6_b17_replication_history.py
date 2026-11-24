"""The immutable historical chain plus the actual authenticated B17 primary."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import wave
import zipfile

import english_v6_b17_replication_gate as opening
import english_v6_b17_source as primary
import english_v6_b17_natural_sources as natural

lineage = primary.lineage
qa = primary.qa


def binding(permit):
    return dict(primary_result_sha256=permit["primary_result_sha256"],
                opening_lock_sha256=permit["opening_lock_sha256"],
                primary_source_inputs=permit["primary_source_inputs"],
                assessment_sha256=lineage.ASSESSMENT_SHA,
                source_ancestry_sha256=lineage.ANCESTRY_SHA,
                old_replication_opened=False, B17_replication_source_opened=True)


def required_archives(authorization):
    permit = opening.require(authorization)
    return tuple(dict.fromkeys((*primary.required_archives(),
                 *(str(pin["id"]) for pin in permit["primary_source_inputs"].values()))))


def original(directory, lock):
    """Hash the actual prior source archive before inspecting any member."""
    raw = (Path(directory) / (str(lock["id"]) + ".zip")).read_bytes()
    qa.need(len(raw) == lock["bytes"] and lineage.digest(raw) == lock["sha256"],
            "actual B17 primary source ZIP identity")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    qa.need(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())),
            "primary source archive CRC and unique original members")
    return archive


def primary_sources(directory, authorization):
    permit = opening.require(authorization)
    seen, groups, people = set(), set(), set()
    with original(directory, permit["primary_source_inputs"]["synth_source"]) as archive:
        meta = json.loads(archive.read("MANIFEST.json"))
        plan = primary.source_plan("primary")
        rows = meta["rows"]
        qa.need(meta["schema"] == primary.SCHEMA and meta["panel"] == "primary"
                and meta["tts_calls"] == len(rows) == 96 and meta["new_kws_forwards"] == 0
                and meta["source_ancestry_sha256"] == lineage.ANCESTRY_SHA
                and meta["assessment_sha256"] == lineage.ASSESSMENT_SHA
                and meta["all_prior_replication_sealed"] is True
                and meta["registered_plan_file_sha256"] == primary.FILES["primary"]
                and meta["plan_sha256"] == primary.B13.b10.sha(primary.B13.b10.canon(plan))
                and len({r["pcm_sha256"] for r in rows}) == 96,
                "complete fixed B17 primary original-source ancestry")
        for row, expected in zip(rows, plan["rows"], strict=True):
            qa.need(all(row[k] == v for k, v in expected.items()) and row["basic_audio_ok"] is True
                    and row["whole_native_resample"] is True and row["crop_calls"] == 0,
                    "actual primary source equals its pre-optimizer roster")
            seen.add(row["pcm_sha256"])
            groups.add(row["source_group_id"])
            people.add(row["source_speaker_id"].removeprefix("p"))
    with original(directory, permit["primary_source_inputs"]["natural_source"]) as archive:
        meta = json.loads(archive.read("MANIFEST.json"))
        raw = archive.read("ROSTERS.json")
        rosters = json.loads(raw)
        design = natural.plan()
        qa.need(meta["schema"] == natural.SCHEMA and meta["panel"] == "primary"
                and meta["design_sha256"] == lineage.ASSESSMENT_SHA
                and meta["source_ancestry_sha256"] == lineage.ANCESTRY_SHA
                and meta["all_prior_replication_sealed"] is True
                and meta["corpus_split"] == "test-clean"
                and meta["corpus_sha256"] == design["upstream"]["sha256"]
                and meta["rosters_sha256"] == lineage.digest(raw) and raw == natural.roster_bytes(rosters)
                and set(rosters) == {"primary"} and len(rosters["primary"]) == 24
                and meta["selected_original_slots"] == 24 and meta["selected_speakers"] == 12,
                "all original natural primary slots, including any quality exclusions")
        selected = rosters["primary"]
        expected = {r["source_id"]: r for r in selected}
        qa.need(len(expected) == 24 and [r["speaker"] for r in selected] ==
                [sp for sp in design["speakers"] for _ in range(2)], "actual primary twelve-reader roster")
        rows, rejected = meta["rows"], meta["source_rejections"]
        keys = [r["source_id"] for r in rows + rejected]
        qa.need(len(keys) == len(set(keys)) == 24 and set(keys) == set(expected)
                and all(r["reason"] in ("duration", "audio_quality") for r in rejected),
                "no selected primary source was replaced or erased")
        for row in rows:
            qa.need(all(row[k] == v for k, v in expected[row["source_id"]].items())
                    and row["id"] == "b17-natural-" + row["source_id"]
                    and row["source_group_id"] == "librispeech-test-clean-" + row["speaker"]
                    and row["role"] == "development" and row["training_allowed"] is False,
                    "actual primary reader and original utterance")
            seen.add(row["pcm_sha256"])
        # An audio-quality rejection can leave its original WAV in the source ZIP.
        # Exclude those bytes as well, not only the rows that survived admission.
        wav_names = [n for n in archive.namelist() if n.startswith("wav/") and n.endswith(".wav")]
        for name in wav_names:
            match = re.fullmatch(r"wav/b17-natural-(\d+-\d+-\d+)\.wav", name)
            qa.need(match is not None and match[1] in expected, "only originally selected primary waveform members")
            with wave.open(io.BytesIO(archive.read(name)), "rb") as wav:
                qa.need((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) ==
                        (1, 2, 16000, "NONE"), "whole prior natural PCM16")
                pcm = wav.readframes(wav.getnframes())
                seen.add(lineage.digest(pcm))
        people.update(design["speakers"])
        groups.update("librispeech-test-clean-" + person for person in design["speakers"])
    qa.need(len(people) == 36, "twenty-four primary Piper identities and twelve actual natural readers")
    return seen, groups, people


def history(directory, authorization):
    opening.require(authorization)  # The primary result gate precedes all historical archive reads.
    seen, groups, people = primary.history(Path(directory))
    new_pcm, new_groups, new_people = primary_sources(directory, authorization)
    qa.need(not (new_people & people), "authenticated B17 primary identities were historically unexposed")
    seen.update(new_pcm)
    groups.update(new_groups)
    people.update(new_people)
    assessment, _ = lineage.read_plan()
    replica_people = set(assessment["replication"]["natural"]["speakers"])
    replica_people.update(r["source_speaker_id"].removeprefix("p")
                          for r in primary.source_plan("replication")["rows"])
    qa.need(not (replica_people & people), "only the fixed new B17 replication source families remain eligible")
    return seen, groups, people
