"""Fixed B21 replication natural originals after an original primary PASS.

Only official transcript metadata ranks the 24 source slots. The frozen replication metadata determines the reader and utterance draw.
No model score chooses or replaces an original source.
"""
from __future__ import annotations
from array import array
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import wave

import english_v6_b21_replication_source_ancestry as lineage

qa, digest = lineage.qa, lineage.digest
CORPUS_SPLIT = "train-clean-100"
BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())


def plan():
    return lineage.read_plan()["replication"]["natural"]


def roster_bytes(rows):
    return (json.dumps({"replication": rows}, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def rank_metadata(catalog, design):
    """Pure ranking of retained transcript text; never reads audio or grants access."""
    qa.need(catalog["schema"] == "english-v6-b21-replication-natural-original-metadata-v1"
            and digest(catalog["speakers_text"].encode("utf-8")) == design["metadata_speakers_sha256"],
            "exact official speaker census before any utterance selection")
    pool = set()
    for line in catalog["speakers_text"].splitlines():
        if not line.strip() or line.startswith(";"): continue
        fields = [part.strip() for part in line.split("|")]
        if fields[2] == CORPUS_SPLIT: pool.add(fields[0])
    requested = design["speakers"]
    qa.need(len(requested) == len(set(requested)) == 12 and set(requested) <= pool
            and not set(requested) & set(design["excluded_canonical_speakers"]),
            "all twelve fixed canonical readers remain independent; no reader substitution")
    by_speaker, seen = defaultdict(list), set()
    qa.need(type(catalog["transcripts"]) is list and catalog["transcripts"], "all selected-reader transcript metadata retained")
    paths = [item["path"] for item in catalog["transcripts"]]
    qa.need(paths == sorted(set(paths)), "unique canonical transcript member order")
    for item in catalog["transcripts"]:
        match = re.fullmatch(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", item["path"])
        qa.need(match is not None, "original transcript member path")
        speaker, chapter, sp2, ch2 = match.groups()
        raw = item["text"].encode("utf-8")
        qa.need((speaker, chapter) == (sp2, ch2) and speaker in requested and digest(raw) == item["sha256"],
                "unchanged selected-reader original transcript bytes")
        for line in item["text"].splitlines():
            fields = line.split(" ", 1)
            qa.need(len(fields) == 2, "original transcript line with source identifier")
            ident, text = fields
            qa.need(re.fullmatch(speaker + "-" + chapter + r"-\d+", ident) is not None and ident not in seen,
                    "unique original speaker/chapter utterance identity")
            seen.add(ident)
            words = text.split(" ")
            if (4 <= len(words) <= 14 and not BLOCK.intersection(words)
                    and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words)):
                by_speaker[speaker].append(dict(speaker=speaker, chapter=chapter, source_id=ident,
                    transcript=text, transcript_file_sha256=item["sha256"], transcript_member=item["path"]))
    rows = []
    for speaker in requested:
        candidates = by_speaker[speaker]
        qa.need(len(candidates) >= 2, "fixed reader has fewer than two eligible transcripts; STOP without replacement")
        candidates.sort(key=lambda row: (digest((design["salt"] + "/utterance/" + row["source_id"]).encode()), row["source_id"]))
        rows.extend(candidates[:2])
    qa.need(len(rows) == len({r["source_id"] for r in rows}) == 24, "all 24 immutable pre-audio slots")
    return rows


def pick(archive, design):
    metadata, transcripts = None, []
    for member in archive:
        if member.name == "LibriSpeech/SPEAKERS.TXT":
            qa.need(metadata is None and member.isfile(), "one original speaker census")
            metadata = archive.extractfile(member).read().decode("utf-8", errors="strict")
        match = re.fullmatch(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", member.name)
        if match and match.group(1) in design["speakers"]:
            qa.need(member.isfile(), "ordinary transcript member")
            raw = archive.extractfile(member).read()
            transcripts.append(dict(path=member.name, sha256=digest(raw), text=raw.decode("utf-8", errors="strict")))
    qa.need(metadata is not None, "official corpus speaker census is present")
    catalog = dict(schema="english-v6-b21-replication-natural-original-metadata-v1", speakers_text=metadata,
                   transcripts=sorted(transcripts, key=lambda item: item["path"]))
    return rank_metadata(catalog, design), catalog


def flac_header(raw):
    qa.need(type(raw) is bytes and len(raw) >= 42 and raw[:4] == b"fLaC"
            and raw[4] & 0x7f == 0 and int.from_bytes(raw[5:8], "big") == 34,
            "original FLAC begins with its complete 34-byte STREAMINFO")
    info = raw[8:42]
    packed = int.from_bytes(info[10:18], "big")
    facts = dict(sample_rate=packed >> 44, channels=((packed >> 41) & 7) + 1,
                 bits_per_sample=((packed >> 36) & 31) + 1, samples=packed & ((1 << 36) - 1),
                 native_pcm_md5=info[18:34].hex())
    qa.need(facts["sample_rate"] == 16000 and facts["channels"] == 1 and facts["bits_per_sample"] == 16
            and facts["samples"] > 0 and facts["native_pcm_md5"] != "0" * 32,
            "whole native mono PCM16 recording with explicit sample count and original PCM checksum")
    return facts


def whole_wav_facts(path):
    """Retain even out-of-window originals; eligible ones use the old inspector."""
    raw = Path(path).read_bytes()
    with wave.open(io.BytesIO(raw), "rb") as stream:
        qa.need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                (1, 2, 16000, "NONE") and stream.getnframes() > 0, "whole native PCM16 mono original")
        samples = stream.getnframes()
        pcm = stream.readframes(samples)
    qa.need(len(pcm) == samples * 2, "every full native PCM sample retained")
    if 3200 <= samples <= 240000:
        return qa.pcm_tools.inspect_wav(path)
    values = array("h", pcm)
    if sys.byteorder != "little": values.byteswap()
    rms = math.sqrt(sum(value * value for value in values) / samples) / 32768
    rails = sum(value in (-32768, 32767) for value in values) / samples
    return dict(wav_sha256=digest(raw), pcm_sha256=digest(pcm), samples=samples, rms=rms,
                rail_fraction=rails, basic_audio_ok=rms >= 0.0001 and rails <= 0.01)


def verify_corpus(corpus):
    pins = plan()["upstream"]
    corpus = Path(corpus)
    qa.need(corpus.stat().st_size == pins["bytes"], "exact whole train-clean-100 archive length")
    with corpus.open("rb") as stream: actual = hashlib.file_digest(stream, "sha256").hexdigest()
    qa.need(actual == pins["sha256"], "original fixed corpus bytes before metadata access")
    return actual


def extract(source, corpus, history):
    source, corpus = Path(source), Path(corpus)
    design = plan()
    old_pcm, old_groups, old_people = lineage.validate_history(history)
    qa.need(not set(design["speakers"]) & old_people, "fresh canonical readers")
    archive_sha = verify_corpus(corpus)
    with tarfile.open(corpus, "r|gz") as archive: roster, catalog = pick(archive, design)
    qa.record(source / "NATURAL_METADATA.json", catalog)
    (source / "NATURAL_ROSTERS.json").write_bytes(roster_bytes(roster))
    (source / "flac").mkdir()
    requests = {f"LibriSpeech/{CORPUS_SPLIT}/{r['speaker']}/{r['chapter']}/{r['source_id']}.flac": r for r in roster}
    originals, seen_pcm = {}, set(old_pcm)
    import soundfile as sf
    with tarfile.open(corpus, "r|gz") as archive:
        for member in archive:
            if member.name not in requests: continue
            row = requests[member.name]
            ident, sid = row["source_id"], "b21-replication-natural-" + row["source_id"]
            qa.need(member.isfile() and ident not in originals, "each fixed original FLAC once")
            raw = archive.extractfile(member).read()
            header = flac_header(raw)
            (source / "flac" / (sid + ".flac")).write_bytes(raw)
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                qa.need((audio.samplerate, audio.channels, audio.subtype, len(audio)) == (16000, 1, "PCM_16", header["samples"]),
                        "native original codec matches complete STREAMINFO")
                pcm = audio.read(dtype="int16").astype("<i2").tobytes()
            qa.need(len(pcm) == 2 * header["samples"] and hashlib.md5(pcm).hexdigest() == header["native_pcm_md5"],
                    "whole native PCM exactly matches original checksum")
            qa.need(digest(pcm) not in seen_pcm, "unseen complete natural PCM")
            seen_pcm.add(digest(pcm))
            with wave.open(str(source / "wav" / (sid + ".wav")), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed")); output.writeframes(pcm)
            facts = whole_wav_facts(source / "wav" / (sid + ".wav"))
            group = "librispeech-train-clean-100-" + row["speaker"]
            qa.need(group not in old_groups, "independent source group")
            originals[ident] = dict(row, **facts, id=sid, source_member=member.name, flac_sha256=digest(raw),
                native_header=header, source_group_id=group, cohort="natural", role="development", requested_text=row["transcript"],
                requested_class=0, observed_class=None, human_gold=False, training_allowed=False,
                whole_original=True, crop_calls=0, pad_calls=0, retry_count=0)
    qa.need(set(originals) == {r["source_id"] for r in roster}, "all24 exact whole originals retained")
    rows, rejected = [], []
    for fixed in roster:
        row = originals[fixed["source_id"]]
        if not 8000 <= row["samples"] <= 240000: rejected.append(dict(row, reason="duration"))
        elif not row["basic_audio_ok"]: rejected.append(dict(row, reason="audio_quality"))
        else: rows.append(row)
    return dict(rows=rows, source_rejections=rejected, corpus_sha256=archive_sha,
        corpus_bytes=design["upstream"]["bytes"], rosters_sha256=qa.sha(source / "NATURAL_ROSTERS.json"),
        metadata_sha256=qa.sha(source / "NATURAL_METADATA.json"), all_original_pcm_retained=True,
        selected_original_slots=24, selected_speakers=12)


def read_source(source, meta, history):
    source, design = Path(source), plan()
    old_pcm, old_groups, old_people = lineage.validate_history(history)
    catalog = json.loads((source / "NATURAL_METADATA.json").read_bytes())
    roster = rank_metadata(catalog, design)
    qa.need((source / "NATURAL_ROSTERS.json").read_bytes() == roster_bytes(roster)
            and meta["rosters_sha256"] == qa.sha(source / "NATURAL_ROSTERS.json")
            and meta["metadata_sha256"] == qa.sha(source / "NATURAL_METADATA.json")
            and meta["corpus_sha256"] == design["upstream"]["sha256"]
            and meta["corpus_bytes"] == design["upstream"]["bytes"]
            and meta["selected_original_slots"] == 24 and meta["selected_speakers"] == 12
            and meta["all_original_pcm_retained"] is True,
            "complete metadata ranking, raw original slots and fixed upstream bytes")
    qa.need([r["speaker"] for r in roster] == [sp for sp in design["speakers"] for _ in range(2)]
            and not set(design["speakers"]) & old_people, "12 original reader pairs; no speaker substitution")
    rows, rejected = meta["rows"], meta["source_rejections"]
    keys = [r["source_id"] for r in rows + rejected]
    qa.need(len(keys) == len(set(keys)) == 24 and set(keys) == {r["source_id"] for r in roster}
            and all(r["reason"] in ("duration", "audio_quality") for r in rejected),
            "all24 slots admitted or rejected without dropping raw originals")
    rejected_by_id = {r["source_id"]: r for r in rejected}
    by_id, seen_pcm = {r["source_id"]: r for r in rows + rejected}, set(old_pcm)
    for item in roster:
        row = by_id[item["source_id"]]
        sid = "b21-replication-natural-" + item["source_id"]
        member = f"LibriSpeech/{CORPUS_SPLIT}/{item['speaker']}/{item['chapter']}/{item['source_id']}.flac"
        raw = (source / "flac" / (sid + ".flac")).read_bytes()
        header = flac_header(raw)
        path = source / "wav" / (sid + ".wav")
        facts = whole_wav_facts(path)
        with wave.open(str(path), "rb") as audio: pcm = audio.readframes(audio.getnframes())
        qa.need(all(row[k] == v for k, v in item.items()) and all(row[k] == v for k, v in facts.items())
                and row["id"] == sid and row["source_member"] == member and row["flac_sha256"] == digest(raw)
                and row["native_header"] == header and header["samples"] == facts["samples"]
                and len(pcm) == 2 * header["samples"] and hashlib.md5(pcm).hexdigest() == header["native_pcm_md5"],
                "all unchanged full raw FLAC/PCM/WAV originals, including rejected ones")
        group = "librispeech-train-clean-100-" + item["speaker"]
        qa.need(facts["pcm_sha256"] not in seen_pcm and group not in old_groups
                and row["source_group_id"] == group and row["role"] == "development" and row["cohort"] == "natural"
                and row["requested_class"] == 0 and row["requested_text"] == item["transcript"] and row["observed_class"] is None
                and row["training_allowed"] is row["human_gold"] is False and row["whole_original"] is True
                and row["crop_calls"] == row["pad_calls"] == row["retry_count"] == 0,
                "canonical reader and project DEV identity, no crop/pad/retry")
        seen_pcm.add(facts["pcm_sha256"])
        reason = "duration" if not 8000 <= facts["samples"] <= 240000 else "audio_quality" if not facts["basic_audio_ok"] else None
        qa.need((reason is None and item["source_id"] not in rejected_by_id) or
                (reason is not None and rejected_by_id.get(item["source_id"], {}).get("reason") == reason),
                "quality rejects determined only from complete original audio")
    qa.need([r["source_id"] for r in rows] == [r["source_id"] for r in roster if r["source_id"] not in rejected_by_id],
            "native observers see precisely eligible originals in fixed roster order")
    return rows, [by_id[r["source_id"]] for r in roster]
