"""B23 whole natural originals; the B22 decoder and rejection semantics are exact.

Only the frozen B23 ancestry/profile and literal original ID prefix differ.
Every selected FLAC, full PCM/WAV, rejected original and decode journal remains.
"""
from __future__ import annotations
import english_v6_b22_natural_sources as baseline
from english_v6_b22_replication_reuse import inherit
inherit(baseline, globals())
import english_v6_b23_source_ancestry as lineage
qa, digest = lineage.qa, lineage.digest


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
    decode_dir = source / "natural-decodes"
    decode_dir.mkdir()
    import soundfile as sf
    with tarfile.open(corpus, "r|gz") as archive:
        for member in archive:
            if member.name not in requests: continue
            row = requests[member.name]
            ident, sid = row["source_id"], "b23-primary-natural-" + row["source_id"]
            qa.need(member.isfile() and ident not in originals, "each fixed original FLAC once")
            raw = archive.extractfile(member).read()
            header = flac_header(raw)
            (source / "flac" / (sid + ".flac")).write_bytes(raw)
            qa.record(decode_dir / (sid + "-STARTED.json"), dict(source_id=ident, source_member=member.name,
                flac_sha256=digest(raw), attempt=1, status="NATIVE_DECODE_CONSUMED_UNKNOWN"))
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                qa.need((audio.samplerate, audio.channels, audio.subtype, len(audio)) == (16000, 1, "PCM_16", header["samples"]),
                        "native original codec matches complete STREAMINFO")
                pcm = audio.read(dtype="int16").astype("<i2").tobytes()
            qa.need(len(pcm) == 2 * header["samples"] and hashlib.md5(pcm).hexdigest() == header["native_pcm_md5"],
                    "whole native PCM exactly matches original checksum")
            qa.record(decode_dir / (sid + "-RETURNED.json"), dict(source_id=ident, pcm_sha256=digest(pcm),
                samples=header["samples"], attempt=1, status="COMPLETE_NATIVE_DECODE"))
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
        sid = "b23-primary-natural-" + item["source_id"]
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
        qa.need(json.loads((source / "natural-decodes" / (sid + "-STARTED.json")).read_bytes()) ==
                dict(source_id=item["source_id"], source_member=member, flac_sha256=digest(raw),
                     attempt=1, status="NATIVE_DECODE_CONSUMED_UNKNOWN")
                and json.loads((source / "natural-decodes" / (sid + "-RETURNED.json")).read_bytes()) ==
                dict(source_id=item["source_id"], pcm_sha256=facts["pcm_sha256"], samples=header["samples"],
                     attempt=1, status="COMPLETE_NATIVE_DECODE"), "all24 original native decode starts and returns retained")
        seen_pcm.add(facts["pcm_sha256"])
        reason = "duration" if not 8000 <= facts["samples"] <= 240000 else "audio_quality" if not facts["basic_audio_ok"] else None
        qa.need((reason is None and item["source_id"] not in rejected_by_id) or
                (reason is not None and rejected_by_id.get(item["source_id"], {}).get("reason") == reason),
                "quality rejects determined only from complete original audio")
    qa.need([r["source_id"] for r in rows] == [r["source_id"] for r in roster if r["source_id"] not in rejected_by_id],
            "native observers see precisely eligible originals in fixed roster order")
    return rows, [by_id[r["source_id"]] for r in roster]
