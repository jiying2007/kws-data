"""B15 prospective independent dev-clean natural ordinary speech and five blind observers."""
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

ASSESSMENT = Path(__file__).with_name("english_v6_b15_panels") / "assessment.json"
ASSESSMENT_SHA = "daa45200d226dd004a41e74a3cf5ff876a1f3fc41dd2def7c5ed937efbf085f8"
OLD_DESIGN = Path(__file__).with_name("english_v6_b14_panels") / "natural-design.json"
OLD_ROSTERS = OLD_DESIGN.with_name("natural-rosters.json")
TRAIN_ROSTER = Path(__file__).with_name("english_v6_train_context24_roster.json")
SCHEMA = "english-v6-b15-ordinary-human-dev-clean-source-v1"
BLOCK = set("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
digest = lambda b: hashlib.sha256(b).hexdigest()


def plan():
    qa.need(qa.sha(ASSESSMENT)==ASSESSMENT_SHA, "B15 assessment frozen before optimization")
    assessment=json.loads(ASSESSMENT.read_bytes())
    qa.need(assessment["schema"]=="english-v6-b15-prospective-nonblank-contrast-assessment-v1"
            and assessment["primary"]["natural"]["corpus"]=="LibriSpeech dev-clean"
            and assessment["primary"]["natural"]["requested_speakers"]==12
            and assessment["constraints"]["preexisting_dev_not_reused_for_selection"],"source-independent planned role")
    old=json.loads(OLD_DESIGN.read_bytes())
    old_rosters=json.loads(OLD_ROSTERS.read_bytes())
    training=json.loads(TRAIN_ROSTER.read_bytes())
    excluded=set(old["excluded_speakers"])
    excluded.update(str(r["speaker"]) for panel in old_rosters.values() for r in panel)
    excluded.update(str(r["speaker"]) for r in training["source_rows"])
    return {
      "upstream":{"bytes":337926286,"sha256":"76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
                 "md5":"42e2234ba48799c1f50f24a7926300a1","corpus":"LibriSpeech dev-clean",
                 "license":"CC BY 4.0","attribution":"V. Panayotov, G. Chen, D. Povey and S. Khudanpur, LibriSpeech, OpenSLR 12"},
      "excluded_speakers": sorted(excluded),
      "salt": assessment["primary"]["natural"]["salt"],
      "primary_speaker_slots": 12,
      "source_utterances_per_speaker":2,
      "project_role":"development",
      "train_allowed":False,
      "independent_study_sha256":ASSESSMENT_SHA
    }


def pick(archive, design):
    by_speaker = defaultdict(list)
    excluded = set(design["excluded_speakers"])
    for f in archive.getmembers():
        match = re.fullmatch(r"LibriSpeech/dev-clean/(\d+)/(\d+)/(\d+)-(\d+)\.trans\.txt", f.name)
        if not match:
            continue
        sp, chapter, sp2, ch2 = match.groups()
        qa.need((sp, chapter) == (sp2, ch2), "metadata identity")
        if sp in excluded:
            continue
        native = archive.extractfile(f).read()
        for line in native.decode("utf-8").splitlines():
            ident, text = line.split(maxsplit=1)
            qa.need(re.fullmatch(sp+"-"+chapter+r"-\d+", ident), "speaker and chapter identity")
            words = text.upper().split()
            if (4 <= len(words) <= 14 and not BLOCK.intersection(words)
                    and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", w) for w in words)):
                by_speaker[sp].append({"speaker":sp, "source_id":ident, "chapter":chapter,
                                       "transcript":text, "transcript_file_sha256":digest(native)})
    salt = design["salt"]
    def rank(role, identifier):
        return digest(f"{salt}/{role}/{identifier}".encode()), identifier
    speakers = sorted((k for k,v in by_speaker.items() if len(v)>=2), key=lambda sp:rank("speaker", sp))
    qa.need(len(speakers) >= 12, "12 new independent dev-clean readers or fail closed")
    all_panels = {}
    for panel, pool in [("primary", speakers[:12])]:
        rows = []
        for sp in pool:
            ordered = sorted(by_speaker[sp], key=lambda r:rank("utterance",r["source_id"]))
            rows.extend(ordered[:2])
        all_panels[panel] = rows
    return all_panels


def verify_corpus(corpus):
    pins = plan()["upstream"]
    with corpus.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    qa.need(corpus.stat().st_size == pins["bytes"] and sha == pins["sha256"] and md5 == pins["md5"],
            "complete pinned dev-clean archive")
    return sha


def roster_bytes(panels):
    return (json.dumps(panels, indent=2) + "\n").encode()


def register(corpus, output):
    verify_corpus(corpus)
    rawtar = corpus.with_suffix(".transcript-index.tar")
    with gzip.open(corpus, "rb") as packed, rawtar.open("xb") as out:
        shutil.copyfileobj(packed, out)
    with tarfile.open(rawtar, mode="r:") as ar:
        panels = pick(ar, plan())
    rawtar.unlink()
    output.write_bytes(roster_bytes(panels))


def extract(source, corpus, panel):
    import soundfile as sf
    design = plan()
    pins = design["upstream"]
    sha = verify_corpus(corpus)
    source.mkdir()
    (source/"wav").mkdir()
    (source/"flac").mkdir()
    rawtar = corpus.with_suffix(".uncompressed")
    with gzip.open(corpus,"rb") as packed, rawtar.open("xb") as out:
        shutil.copyfileobj(packed,out)
    rows = []
    skipped = []
    with tarfile.open(rawtar, mode="r:") as ar:
        panels = pick(ar, design)
        # Source roster derives from the predeclared salt and authenticated full transcript archive.
        # No KWS model or score is read before registration.
        registered=roster_bytes(panels)
        (source/"ROSTERS.json").write_bytes(registered)
        for r in panels[panel]:
            sp, ch, ident = r["speaker"], r["chapter"], r["source_id"]
            member = f"LibriSpeech/dev-clean/{sp}/{ch}/{ident}.flac"
            raw = ar.extractfile(member).read()
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                qa.need((audio.samplerate,audio.channels,audio.subtype)==(16000,1,"PCM_16"),"native original format")
                if not 8000<=len(audio)<=240000:
                    skipped.append(dict(source_id=ident, reason="duration",samples=len(audio)))
                    continue
                pcm=audio.read(dtype="int16").astype("<i2").tobytes()
            sid="b15-natural-"+ident
            (source/"flac"/(sid+".flac")).write_bytes(raw)
            wav=source/"wav"/(sid+".wav")
            with wave.open(str(wav),"wb") as f:
                f.setparams((1,2,16000,0,"NONE","not compressed"))
                f.writeframes(pcm)
            facts=qa.pcm_tools.inspect_wav(wav)
            if not facts["basic_audio_ok"]:
                skipped.append(dict(source_id=ident,reason="audio_quality",samples=facts["samples"]))
                continue
            rows.append(dict(r,**facts,id=sid,source_member=member,flac_sha256=digest(raw),
                             source_group_id="librispeech-dev-clean-"+sp,role="development",
                             requested_text=r["transcript"],requested_class=0))
    rawtar.unlink()
    (source/"MANIFEST.json").write_text(json.dumps(dict(schema=SCHEMA,panel=panel,design_sha256=ASSESSMENT_SHA,
                    corpus_sha256=sha,rosters_sha256=qa.sha(source/"ROSTERS.json"),rows=rows,
                    source_rejections=skipped,attribution=pins["attribution"],license=pins["license"],
                    physical_microphone_test=False,kws_forwards=0,training_allowed=False,human_gold=False),indent=2)+"\n")


def read_source(source):
    design = plan()
    meta=json.loads((source/"MANIFEST.json").read_bytes())
    rosters=json.loads((source/"ROSTERS.json").read_bytes())
    qa.need(meta["schema"] == SCHEMA and meta["corpus_sha256"] == design["upstream"]["sha256"] and
            meta["rosters_sha256"] == qa.sha(source/"ROSTERS.json") and meta["design_sha256"] == ASSESSMENT_SHA, "roster bindings")
    qa.need(roster_bytes(rosters)==(source/"ROSTERS.json").read_bytes(),"selection roster preserves deterministic sample order")
    qa.need(meta["panel"]=="primary" and set(rosters)=={"primary"} and len({r["speaker"] for r in rosters["primary"]})==12,"frozen new source role and speaker denominator")
    rows=meta["rows"]
    qa.need(len({r["source_id"] for r in rows})==len(rows) and 0<len(rows)<=24,"eligible original identities")
    expected={r["source_id"]:r for r in rosters[meta["panel"]]}
    for r in rows:
        qa.need(all(r[k]==v for k,v in expected[r["source_id"]].items()),"preselected row")
        actual=qa.pcm_tools.inspect_wav(source/"wav"/(r["id"]+".wav"))
        qa.need(all(r[k]==v for k,v in actual.items()) and actual["basic_audio_ok"],"full PCM16")
    return rows


def screen(source, providers, output):
    rows=read_source(source)
    texts, phones, counts=qa.load_observations(source, providers, rows, SCHEMA)
    decisions=[]
    for r in rows:
        sid = r["id"]
        all_obs=set(texts[sid])==set(qa.policy.TEXT_MODELS) and set(phones[sid])==set(qa.policy.PHONE_MODELS)
        heard=[qa.policy.norm(x) for x in texts[sid].values()]
        target=qa.policy.norm(r["transcript"])
        votes=heard.count(target)
        veto=any(BLOCK.intersection(t.upper().split()) for t in heard) or any(
            qa.policy.greedy_full_wake(x["logits"],x["vocab"]) for x in phones[sid].values())
        tier="unknown"
        if all_obs and not veto and votes>=2:
            tier="silver" if votes==3 else "weak"
        decisions.append(dict(id=sid,group=r["source_group_id"],source_id=r["source_id"],tier=tier,
                  exact_votes=votes,full_wake_veto=veto,native_text=texts[sid],
                  phone_sha256={p:x["sha256"] for p,x in phones[sid].items()},pcm_sha256=r["pcm_sha256"]))
    ok=[r for r in decisions if r["tier"]!="unknown"]
    output.write_text(json.dumps(dict(schema=SCHEMA,panel=json.loads((source/"MANIFEST.json").read_bytes())["panel"],
          source_manifest_sha256=qa.sha(source/"MANIFEST.json"),source_rows=len(rows),coverage=counts,
          admitted=len(ok),admitted_speakers=len({r["group"] for r in ok}),decisions=decisions,kws_forwards=0,
          field_far=None,training_allowed=False,field_human_frr=None),indent=2)+"\n")


if __name__=="__main__":
    method,*args=sys.argv[1:]
    if method=="register":
        register(*map(Path, args))
    elif method=="generate":
        source,corp,panel=args
        qa.need(panel=="primary","replication requires independently registered opening")
        extract(Path(source),Path(corp),panel)
    elif method=="observe":
        provider,source,out=args
        qa.observers.observe(provider,Path(source),Path(out),read_rows=read_source,schema=SCHEMA)
    elif method=="screen":
        screen(*map(Path,args))
    else:
        raise ValueError(method)
