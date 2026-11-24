"""New test-clean reader primary with complete historical speaker ancestry."""
from __future__ import annotations
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
import english_v6_primary_source_qa as qa
import english_v6_b17_source_ancestry as lineage

ASSESSMENT = lineage.ASSESSMENT
ASSESSMENT_SHA = lineage.ASSESSMENT_SHA
CORPUS_SPLIT = "test-clean"
SCHEMA = "english-v6-b17-ordinary-human-test-clean-source-v1"
BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
digest = lineage.digest


def plan():
    assessment, ancestry = lineage.read_plan()
    source = assessment["primary"]["natural"]
    qa.need(source["corpus_split"] == CORPUS_SPLIT and source["requested_speakers"] == 12
            and source["source_slots"] == 24 and source["no_replacement_after_quality_rejection"],
            "fixed 12 genuinely new readers and 24 prospective whole-utterance slots")
    return dict(upstream=source["upstream"], corpus_split=CORPUS_SPLIT,
                salt=source["salt"], speakers=source["speakers"],
                excluded_speakers=ancestry["natural_excluded_speakers"],
                source_utterances_per_speaker=2, primary_speaker_slots=12,
                metadata_speakers_sha256=ancestry["metadata_census"]["speakers_sha256"],
                independent_study_sha256=ASSESSMENT_SHA,
                source_ancestry_sha256=lineage.ANCESTRY_SHA,
                project_role="development", train_allowed=False)


def pick(archive, design):
    """Choose transcript identities for the fixed readers before reading any FLAC."""
    metadata = archive.extractfile("LibriSpeech/SPEAKERS.TXT").read()
    qa.need(digest(metadata) == design["metadata_speakers_sha256"], "exact official speaker census member")
    people = set()
    for line in metadata.decode("utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [part.strip() for part in line.split("|")]
        if fields[2] == design["corpus_split"]:
            people.add(fields[0])
    requested = design["speakers"]
    qa.need(len(requested) == len(set(requested)) == 12 and set(requested) <= people
            and not (set(requested) & set(design["excluded_speakers"])),
            "source-independent frozen reader identity, no substitutions")
    by_speaker = defaultdict(list)
    split = re.escape(design["corpus_split"])
    for member in archive.getmembers():
        match = re.fullmatch(r"LibriSpeech/" + split + r"/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", member.name)
        if not match:
            continue
        speaker, chapter, sp2, ch2 = match.groups()
        qa.need((speaker, chapter) == (sp2, ch2), "transcript path identity")
        if speaker not in requested:
            continue
        raw = archive.extractfile(member).read()
        for line in raw.decode("utf-8").splitlines():
            ident, text = line.split(maxsplit=1)
            qa.need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident), "original speaker/chapter transcript")
            words = text.upper().split()
            if (4 <= len(words) <= 14 and not BLOCK.intersection(words)
                    and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words)):
                by_speaker[speaker].append(dict(speaker=speaker, source_id=ident, chapter=chapter,
                                                transcript=text, transcript_file_sha256=digest(raw)))
    rows = []
    for speaker in requested:
        candidates = by_speaker[speaker]
        qa.need(len(candidates) >= 2, "frozen reader lacks two eligible transcripts; STOP without replacement")
        candidates = sorted(candidates, key=lambda row: (
            digest((design["salt"] + "/utterance/" + row["source_id"]).encode()), row["source_id"]))
        rows.extend(candidates[:2])
    qa.need(len(rows) == len({row["source_id"] for row in rows}) == 24, "24 unique fixed source slots")
    return {"primary": rows}


def verify_corpus(corpus):
    pins = plan()["upstream"]
    with corpus.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    qa.need(corpus.stat().st_size == pins["bytes"] and actual == pins["sha256"],
            "complete exact official test-clean archive before reading any source")
    return actual


def roster_bytes(rosters):
    return (json.dumps(rosters, indent=2) + "\n").encode()


def extract(source, corpus, panel, previous):
    import soundfile as sf
    qa.need(panel == "primary", "B17 replication still sealed")
    design = plan()
    old_pcm, old_people = lineage.verify_history(previous)
    qa.need(not (set(design["speakers"]) & old_people), "all old TRAIN/DEV/reserved source identities excluded")
    archive_sha = verify_corpus(corpus)
    source.mkdir()
    (source / "wav").mkdir()
    (source / "flac").mkdir()
    rawtar = corpus.with_suffix(".b17-uncompressed")
    with gzip.open(corpus, "rb") as packed, rawtar.open("xb") as output:
        shutil.copyfileobj(packed, output)
    rows, rejected = [], []
    try:
        with tarfile.open(rawtar, "r:") as archive:
            rosters = pick(archive, design)
            (source / "ROSTERS.json").write_bytes(roster_bytes(rosters))
            for row in rosters[panel]:
                ident = row["source_id"]
                member = f"LibriSpeech/{CORPUS_SPLIT}/{row['speaker']}/{row['chapter']}/{ident}.flac"
                raw = archive.extractfile(member).read()
                with sf.SoundFile(io.BytesIO(raw)) as audio:
                    qa.need((audio.samplerate, audio.channels, audio.subtype) == (16000, 1, "PCM_16"),
                            "unchanged whole original PCM16 recording")
                    if not 8000 <= len(audio) <= 240000:
                        rejected.append(dict(source_id=ident, reason="duration", samples=len(audio)))
                        continue
                    pcm = audio.read(dtype="int16").astype("<i2").tobytes()
                qa.need(digest(pcm) not in old_pcm, "prior original PCM cannot become new DEV")
                sid = "b17-natural-" + ident
                (source / "flac" / (sid + ".flac")).write_bytes(raw)
                path = source / "wav" / (sid + ".wav")
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(pcm)
                facts = qa.pcm_tools.inspect_wav(path)
                if not facts["basic_audio_ok"]:
                    rejected.append(dict(source_id=ident, reason="audio_quality", samples=facts["samples"]))
                    continue
                rows.append(dict(row, **facts, id=sid, source_member=member, flac_sha256=digest(raw),
                                 source_group_id="librispeech-" + CORPUS_SPLIT + "-" + row["speaker"],
                                 role="development", requested_text=row["transcript"], requested_class=0,
                                 human_gold=False, training_allowed=False))
    finally:
        rawtar.unlink()
    pins = design["upstream"]
    qa.record(source / "MANIFEST.json", dict(schema=SCHEMA, panel=panel, design_sha256=ASSESSMENT_SHA,
              source_ancestry_sha256=lineage.ANCESTRY_SHA, corpus_split=CORPUS_SPLIT,
              corpus_sha256=archive_sha, rosters_sha256=qa.sha(source / "ROSTERS.json"), rows=rows,
              source_rejections=rejected, selected_original_slots=24, selected_speakers=12,
              metadata_speakers_sha256=design["metadata_speakers_sha256"],
              legacy_natural_member_sha256=lineage.read_plan()[1]["legacy_natural"]["member_sha256"],
              attribution=pins["attribution"], license=pins["license"],
              physical_microphone_test=False, kws_forwards=0, training_allowed=False, human_gold=False,
              all_prior_replication_sealed=True))
    read_source(source)


def read_source(source):
    design = plan()
    meta = json.loads((source / "MANIFEST.json").read_bytes())
    raw = (source / "ROSTERS.json").read_bytes()
    rosters = json.loads(raw)
    qa.need(meta["schema"] == SCHEMA and meta["panel"] == "primary"
            and meta["corpus_split"] == CORPUS_SPLIT and meta["corpus_sha256"] == design["upstream"]["sha256"]
            and meta["design_sha256"] == ASSESSMENT_SHA and meta["source_ancestry_sha256"] == lineage.ANCESTRY_SHA
            and meta["rosters_sha256"] == digest(raw) and raw == roster_bytes(rosters)
            and set(rosters) == {"primary"} and len(rosters["primary"]) == 24
            and meta["selected_original_slots"] == 24 and meta["selected_speakers"] == 12
            and not meta["kws_forwards"] and not meta["training_allowed"] and not meta["human_gold"]
            and meta["all_prior_replication_sealed"], "frozen source identity and provenance")
    expected = {row["source_id"]: row for row in rosters["primary"]}
    people = [row["speaker"] for row in rosters["primary"]]
    qa.need(len(expected) == 24 and people == [sp for sp in design["speakers"] for _ in range(2)]
            and not (set(people) & set(design["excluded_speakers"])), "original preselected speaker order")
    rows = meta["rows"]
    rejected = meta["source_rejections"]
    keys = [row["source_id"] for row in rows + rejected]
    qa.need(len(keys) == len(set(keys)) == 24 and set(keys) == set(expected)
            and 0 < len(rows) <= 24
            and all(row["reason"] in ("duration", "audio_quality") for row in rejected),
            "quality exclusions never substitute or fabricate an original")
    for row in rows:
        qa.need(all(row[k] == v for k, v in expected[row["source_id"]].items())
                and row["role"] == "development" and not row["training_allowed"] and not row["human_gold"]
                and row["source_group_id"] == "librispeech-" + CORPUS_SPLIT + "-" + row["speaker"],
                "fixed new independent reader/source role")
        facts = qa.pcm_tools.inspect_wav(source / "wav" / (row["id"] + ".wav"))
        qa.need(all(row[k] == v for k, v in facts.items()) and facts["basic_audio_ok"], "full unchanged PCM16 source")
        qa.need(digest((source / "flac" / (row["id"] + ".flac")).read_bytes()) == row["flac_sha256"],
                "saved original FLAC digest")
    return rows


def screen(source, providers, output):
    rows = read_source(source)
    texts, phones, counts = qa.load_observations(source, providers, rows, SCHEMA)
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
    admitted = [row for row in decisions if row["tier"] != "unknown"]
    people = len({row["group"] for row in admitted})
    gate = "SOURCE_GATE_PASS" if len(admitted) >= 12 and people >= 8 else "INSUFFICIENT_SOURCE"
    qa.record(output, dict(schema=SCHEMA, panel="primary", source_manifest_sha256=qa.sha(source / "MANIFEST.json"),
              source_ancestry_sha256=lineage.ANCESTRY_SHA, source_rows=len(rows), coverage=counts,
              attempted_per_provider=len(rows), admitted=len(admitted), admitted_speakers=people,
              decisions=decisions, source_gate=gate, minimum_approved_originals=12, minimum_approved_speakers=8,
              kws_forwards=0, field_far=None, training_allowed=False, field_human_frr=None,
              all_prior_replication_sealed=True))


if __name__ == "__main__":
    method, *args = sys.argv[1:]
    if method == "history":
        if args:
            lineage.verify_history(Path(args[0]))
            print("PASS complete original speaker ancestry; no model or audio inference")
        else:
            print("\n".join(lineage.required_archives()))
    elif method == "generate":
        source, corpus, panel, previous = args
        extract(Path(source), Path(corpus), panel, Path(previous))
    elif method == "observe":
        provider, source, output = args
        qa.observers.observe(provider, Path(source), Path(output), read_rows=read_source, schema=SCHEMA)
    elif method == "screen":
        screen(*map(Path, args))
    else:
        raise ValueError(method)
