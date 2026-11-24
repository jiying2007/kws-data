"""Eighteen frozen dev-other source slots, gated by authenticated B17 primary PASS."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_replication_gate as gate
import english_v6_b17_replication_history as chain
import english_v6_b17_natural_sources as primary

qa, lineage, BLOCK = primary.qa, primary.lineage, primary.BLOCK
ASSESSMENT, ASSESSMENT_SHA = lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
CORPUS_SPLIT = "dev-other"
SCHEMA = "english-v6-b17-ordinary-human-dev-other-replication-source-v1"
digest = lineage.digest
history, required_archives = chain.history, chain.required_archives
roster_bytes = primary.roster_bytes


def plan():
    assessment, ancestry = lineage.read_plan()
    source = assessment["replication"]["natural"]
    qa.need(source["corpus_split"] == CORPUS_SPLIT and source["requested_speakers"] == 9
            and source["source_slots"] == 18 and source["utterances_per_speaker"] == 2
            and source["no_replacement_after_quality_rejection"] is True
            and source["minimum_approved_speakers"] == 8 and source["minimum_approved_utterances"] == 12,
            "exact pre-optimizer nine-reader/eighteen-slot replica; unchanged admission minima")
    excluded = set(ancestry["natural_excluded_speakers"]) | set(assessment["primary"]["natural"]["speakers"])
    qa.need(not (set(source["speakers"]) & excluded), "all old and actual primary readers excluded")
    return dict(upstream=source["upstream"], corpus_split=CORPUS_SPLIT, salt=source["salt"],
                speakers=source["speakers"], excluded_speakers=sorted(excluded, key=int),
                source_utterances_per_speaker=2, replication_speaker_slots=9, source_slots=18,
                metadata_speakers_sha256=ancestry["metadata_census"]["speakers_sha256"],
                independent_study_sha256=ASSESSMENT_SHA, source_ancestry_sha256=lineage.ANCESTRY_SHA,
                project_role="development", train_allowed=False)


def pick(archive, design, authorization):
    gate.require(authorization)  # Before reading even source transcript/census members.
    metadata = archive.extractfile("LibriSpeech/SPEAKERS.TXT").read()
    qa.need(digest(metadata) == design["metadata_speakers_sha256"], "exact original official speaker census")
    people = set()
    for line in metadata.decode("utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        if fields[2] == CORPUS_SPLIT:
            people.add(fields[0])
    requested = design["speakers"]
    qa.need(len(requested) == len(set(requested)) == 9 and set(requested) <= people
            and not (set(requested) & set(design["excluded_speakers"])),
            "fixed unexposed reader identities, no substitutions")
    by_speaker = defaultdict(list)
    for member in archive.getmembers():
        match = re.fullmatch(r"LibriSpeech/dev-other/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", member.name)
        if not match:
            continue
        speaker, chapter, sp2, ch2 = match.groups()
        qa.need((speaker, chapter) == (sp2, ch2), "original transcript path")
        if speaker not in requested:
            continue
        raw = archive.extractfile(member).read()
        for line in raw.decode("utf-8").splitlines():
            ident, text = line.split(maxsplit=1)
            qa.need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident), "original speaker/chapter source ID")
            words = text.upper().split()
            if (4 <= len(words) <= 14 and not BLOCK.intersection(words)
                    and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words)):
                by_speaker[speaker].append(dict(speaker=speaker, source_id=ident, chapter=chapter,
                                                transcript=text, transcript_file_sha256=digest(raw)))
    rows = []
    for speaker in requested:
        candidates = by_speaker[speaker]
        qa.need(len(candidates) >= 2, "frozen reader lacks two eligible transcripts; STOP without replacement")
        ranked = sorted(candidates, key=lambda row: (
            digest((design["salt"] + "/utterance/" + row["source_id"]).encode()), row["source_id"]))
        rows.extend(ranked[:2])
    qa.need(len(rows) == len({r["source_id"] for r in rows}) == 18, "all eighteen original fixed slots")
    return {"replication": rows}


def verify_corpus(corpus, authorization):
    gate.require(authorization)
    pins = plan()["upstream"]
    with corpus.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    qa.need(corpus.stat().st_size == pins["bytes"] and actual == pins["sha256"],
            "complete exact dev-other original archive before transcript/audio access")
    return actual


def extract(source, corpus, archives, authorization):
    permit = gate.require(authorization)
    design = plan()
    old_pcm, _, old_people = history(archives, authorization)
    qa.need(not (set(design["speakers"]) & old_people), "all actual old/primary/reserved identities excluded")
    archive_sha = verify_corpus(corpus, authorization)
    import soundfile as sf
    source.mkdir()
    (source / "wav").mkdir()
    (source / "flac").mkdir()
    rawtar = corpus.with_suffix(".b17-replication-uncompressed")
    with gzip.open(corpus, "rb") as packed, rawtar.open("xb") as output:
        shutil.copyfileobj(packed, output)
    rows, rejected = [], []
    try:
        with tarfile.open(rawtar, "r:") as archive:
            rosters = pick(archive, design, authorization)
            (source / "ROSTERS.json").write_bytes(roster_bytes(rosters))
            for row in rosters["replication"]:
                ident = row["source_id"]
                sid = "b17-replication-natural-" + ident
                member = f"LibriSpeech/{CORPUS_SPLIT}/{row['speaker']}/{row['chapter']}/{ident}.flac"
                raw = archive.extractfile(member).read()
                (source / "flac" / (sid + ".flac")).write_bytes(raw)
                provenance = dict(source_id=ident, source_member=member, flac_sha256=digest(raw))
                with sf.SoundFile(io.BytesIO(raw)) as audio:
                    qa.need((audio.samplerate, audio.channels, audio.subtype) == (16000, 1, "PCM_16"),
                            "whole original mono PCM16 recording")
                    if not 8000 <= len(audio) <= 240000:
                        rejected.append(dict(provenance, reason="duration", samples=len(audio)))
                        continue
                    pcm = audio.read(dtype="int16").astype("<i2").tobytes()
                qa.need(digest(pcm) not in old_pcm, "an old or primary original cannot become new replica DEV")
                old_pcm.add(digest(pcm))
                path = source / "wav" / (sid + ".wav")
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(pcm)
                facts = qa.pcm_tools.inspect_wav(path)
                if not facts["basic_audio_ok"]:
                    rejected.append(dict(provenance, reason="audio_quality", samples=facts["samples"],
                                         pcm_sha256=facts["pcm_sha256"]))
                    continue
                rows.append(dict(row, **facts, id=sid, source_member=member, flac_sha256=digest(raw),
                                 source_group_id="librispeech-dev-other-" + row["speaker"],
                                 role="development", requested_text=row["transcript"], requested_class=0,
                                 human_gold=False, training_allowed=False))
    finally:
        rawtar.unlink()
    pins = design["upstream"]
    qa.record(source / "MANIFEST.json", dict(schema=SCHEMA, panel="replication", **chain.binding(permit),
              design_sha256=ASSESSMENT_SHA, corpus_split=CORPUS_SPLIT, corpus_sha256=archive_sha,
              rosters_sha256=qa.sha(source / "ROSTERS.json"), rows=rows, source_rejections=rejected,
              selected_original_slots=18, selected_speakers=9,
              metadata_speakers_sha256=design["metadata_speakers_sha256"],
              legacy_natural_member_sha256=lineage.read_plan()[1]["legacy_natural"]["member_sha256"],
              attribution=pins["attribution"], license=pins["license"], physical_microphone_test=False,
              kws_forwards=0, training_allowed=False, human_gold=False, all_prior_replication_sealed=True))
    read_source(source, authorization)


def read_source(source, authorization):
    permit = gate.require(authorization)
    source = Path(source)
    design = plan()
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    raw = (source / "ROSTERS.json").read_bytes()
    rosters = json.loads(raw)
    qa.need(all(meta[k] == v for k, v in chain.binding(permit).items())
            and meta["schema"] == SCHEMA and meta["panel"] == "replication"
            and meta["corpus_split"] == CORPUS_SPLIT and meta["corpus_sha256"] == design["upstream"]["sha256"]
            and meta["design_sha256"] == ASSESSMENT_SHA and meta["rosters_sha256"] == digest(raw)
            and raw == roster_bytes(rosters) and set(rosters) == {"replication"}
            and len(rosters["replication"]) == 18
            and meta["selected_original_slots"] == 18 and meta["selected_speakers"] == 9
            and meta["metadata_speakers_sha256"] == design["metadata_speakers_sha256"]
            and meta["kws_forwards"] == 0 and meta["training_allowed"] is False
            and meta["human_gold"] is False and meta["all_prior_replication_sealed"] is True,
            "immutable natural replica identity and actual primary authorization")
    expected = {r["source_id"]: r for r in rosters["replication"]}
    people = [r["speaker"] for r in rosters["replication"]]
    qa.need(len(expected) == 18 and people == [sp for sp in design["speakers"] for _ in range(2)]
            and not (set(people) & set(design["excluded_speakers"])), "all original nine readers in fixed order")
    for ident, row in expected.items():
        qa.need(re.fullmatch(row["speaker"] + "-" + row["chapter"] + r"-\d+", ident)
                and gate.valid_hash(row["transcript_file_sha256"]), "canonical selected transcript identity")
    rows, rejected = meta["rows"], meta["source_rejections"]
    keys = [r["source_id"] for r in rows + rejected]
    qa.need(len(keys) == len(set(keys)) == 18 and set(keys) == set(expected)
            and 0 <= len(rows) <= 18 and len({r["pcm_sha256"] for r in rows}) == len(rows)
            and all(r["reason"] in ("duration", "audio_quality") for r in rejected),
            "every original slot is retained or explicitly quality-rejected; never replaced")
    for row in rows + rejected:
        fixed = expected[row["source_id"]]
        sid = "b17-replication-natural-" + row["source_id"]
        member = f"LibriSpeech/{CORPUS_SPLIT}/{fixed['speaker']}/{fixed['chapter']}/{row['source_id']}.flac"
        qa.need(row["source_member"] == member
                and digest((source / "flac" / (sid + ".flac")).read_bytes()) == row["flac_sha256"],
                "saved original FLAC for every selected slot, including quality exclusions")
        if row in rejected and row["reason"] == "duration":
            qa.need(type(row["samples"]) is int and not 8000 <= row["samples"] <= 240000,
                    "duration exclusion cannot erase an eligible original")
            continue
        facts = qa.pcm_tools.inspect_wav(source / "wav" / (sid + ".wav"))
        if row in rejected:
            qa.need(not facts["basic_audio_ok"] and row["samples"] == facts["samples"]
                    and row["pcm_sha256"] == facts["pcm_sha256"], "actual retained audio-quality rejection")
        else:
            qa.need(all(row[k] == v for k, v in fixed.items()) and all(row[k] == v for k, v in facts.items())
                    and facts["basic_audio_ok"] is True and row["id"] == sid
                    and row["source_group_id"] == "librispeech-dev-other-" + fixed["speaker"]
                    and row["role"] == "development" and row["requested_class"] == 0
                    and row["requested_text"] == fixed["transcript"]
                    and row["training_allowed"] is False and row["human_gold"] is False,
                    "unchanged full original PCM16 and unexposed canonical reader")
    return rows


def observe(provider, source, output, authorization):
    gate.require(authorization)
    qa.observers.observe(provider, source, output,
                         read_rows=lambda path: read_source(path, authorization), schema=SCHEMA)


def screen(source, providers, output, authorization):
    permit = gate.require(authorization)
    rows = read_source(source, authorization)
    texts, phones, counts = qa.load_observations(source, providers, rows, SCHEMA)
    qa.need(counts == dict.fromkeys(qa.observers.PROVIDERS, len(rows)),
            "five native observers cover every retained full original")
    decisions = []
    for row in rows:
        sid = row["id"]
        all_obs = set(texts[sid]) == set(qa.policy.TEXT_MODELS) and set(phones[sid]) == set(qa.policy.PHONE_MODELS)
        heard = [qa.policy.norm(text) for text in texts[sid].values()]
        votes = heard.count(qa.policy.norm(row["transcript"]))
        veto = any(BLOCK.intersection(text.upper().split()) for text in heard) or any(
            qa.policy.greedy_full_wake(value["logits"], value["vocab"]) for value in phones[sid].values())
        tier = "silver" if all_obs and not veto and votes == 3 else (
            "weak" if all_obs and not veto and votes >= 2 else "unknown")
        decisions.append(dict(id=sid, group=row["source_group_id"], source_id=row["source_id"], tier=tier,
              exact_votes=votes, full_wake_veto=veto, native_text=texts[sid],
              phone_sha256={p: value["sha256"] for p, value in phones[sid].items()}, pcm_sha256=row["pcm_sha256"]))
    admitted = [r for r in decisions if r["tier"] != "unknown"]
    people = len({r["group"] for r in admitted})
    result = "SOURCE_GATE_PASS" if len(admitted) >= 12 and people >= 8 else "INSUFFICIENT_SOURCE"
    qa.record(output, dict(schema=SCHEMA, panel="replication", **chain.binding(permit),
              source_manifest_sha256=qa.sha(source / "MANIFEST.json"), source_rows=len(rows), coverage=counts,
              attempted_per_provider=len(rows), selected_original_slots=18, selected_speakers=9,
              admitted=len(admitted), admitted_speakers=people, decisions=decisions, source_gate=result,
              minimum_approved_originals=12, minimum_approved_speakers=8,
              kws_forwards=0, field_far=None, training_allowed=False, field_human_frr=None,
              all_prior_replication_sealed=True, promotion_allowed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    item = modes.add_parser("history"); item.add_argument("directory", nargs="?", type=Path)
    item = modes.add_parser("generate"); item.add_argument("out", type=Path)
    item.add_argument("corpus", type=Path); item.add_argument("archives", type=Path)
    item = modes.add_parser("observe"); item.add_argument("provider", choices=qa.observers.PROVIDERS)
    item.add_argument("source", type=Path); item.add_argument("out", type=Path)
    item = modes.add_parser("screen"); item.add_argument("source", type=Path)
    item.add_argument("providers", type=Path); item.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.mode == "history":
        if args.directory:
            history(args.directory, args.authorization)
            print("PASS original historical source chain and actual B17 primary ancestry")
        else:
            print("\n".join(required_archives(args.authorization)))
    elif args.mode == "generate":
        extract(args.out, args.corpus, args.archives, args.authorization)
    elif args.mode == "observe":
        observe(args.provider, args.source, args.out, args.authorization)
    elif args.mode == "screen":
        screen(args.source, args.providers, args.out, args.authorization)
