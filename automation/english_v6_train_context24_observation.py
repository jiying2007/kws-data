"""Extract registered immutable original TRAIN FLAC and screen against three blind ASRs."""
from collections import defaultdict
from hashlib import file_digest
import io
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_train_context24_sources as registry
import english_v6_b11_supplement_observe as observers
import english_fresh24_ctc_admission as text_policy
import english_hifi_micro as pcm_tools
import english_pocket12_observe as asr

SCHEMA="english-v6-new-natural-train-context24-v1"
PROVIDERS=("qwen","whisper","parakeet")
ROSTER_SHA="a6ac3da422fc98cdc826608ee66a11a44606feb19a98a0102648169f006851e9"
ROSTER=Path(__file__).with_name("english_v6_train_context24_roster.json")
need=registry.need
sha=registry.sha
write=registry.write


def generate(out):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    plan=json.loads(registry.PLAN_PATH.read_bytes())
    need(sha(ROSTER.read_bytes())==ROSTER_SHA,"immutable pre-audio roster SHA256")
    roster=json.loads(ROSTER.read_bytes())
    need(roster["schema"]==SCHEMA and roster["plan_sha256"]==sha(registry.PLAN_PATH.read_bytes()) and roster["raw_audio_decodes"]==0,
         "source-only immutable roster")
    ids=roster["source_rows"]
    need(len(ids)==len({v["source_id"] for v in ids})==24 and len({v["speaker"] for v in ids})==12 and
         not set(plan["excluded_speakers"]).intersection(v["speaker"] for v in ids),"12 disjoint prospective TRAIN identities")
    wanted=defaultdict(lambda:defaultdict(list))
    for row in ids:
        wanted[row["parquet_path"]][row["row_group"]].append(row)
    out.mkdir();(out/"wav").mkdir();(out/"flac").mkdir()
    all_rows=[]; skipped=[]
    for name,groups in wanted.items():
        pin=next(x for x in roster["scanned_shards"] if x["path"]==name)
        path=Path(hf_hub_download(repo_id=roster["hf_repo"],repo_type="dataset",revision=roster["hf_revision"],filename=name,token=False))
        with path.open("rb") as stream:
            need(path.stat().st_size==pin["bytes"] and file_digest(stream,"sha256").hexdigest()==pin["lfs_sha256"],"verified complete original parquet shard")
        file=pq.ParquetFile(path)
        for group,requested in groups.items():
            table=file.read_row_group(group,columns=["speaker_id","chapter_id","id","text","audio"],use_threads=False)
            for row in requested:
                got=table.slice(row["row_in_group"],1).to_pylist()[0]
                need(got["id"]==row["source_id"] and got["text"].strip()==row["transcript"]
                     and str(got["speaker_id"])==row["speaker"] and str(got["chapter_id"])==row["chapter"],"frozen full row identity")
                raw=got["audio"]["bytes"]
                need(type(raw) is bytes and raw.startswith(b"fLaC"),"whole original encoded FLAC bytes")
                with sf.SoundFile(io.BytesIO(raw)) as st:
                    need((st.samplerate,st.channels,st.subtype)==(16000,1,"PCM_16"),"native unaltered 16-kHz mono PCM16")
                    count=len(st)
                    if not 8000<=count<=240000:
                        skipped.append(dict(row,samples=count,flac_sha256=sha(raw),reason="duration"))
                        continue
                    samples=st.read(dtype="int16").astype("<i2").tobytes()
                sid="train-context24-"+row["source_id"]
                (out/"flac"/(sid+".flac")).write_bytes(raw)
                wav=out/"wav"/(sid+".wav")
                with wave.open(str(wav),"wb") as f:
                    f.setparams((1,2,16000,0,"NONE","not compressed"));f.writeframes(samples)
                facts=pcm_tools.inspect_wav(wav)
                need(facts["pcm_sha256"]==sha(samples) and facts["samples"]==count,"sample-exact PCM16 fallback")
                all_rows.append(dict(row,**facts,id=sid,source_group_id="librispeech-"+row["speaker"],
                    native_filename=got["audio"]["path"],full_original_audio=True,role="train",requested_class=0,
                    requested_text=row["transcript"],shard_sha256=pin["lfs_sha256"],flac_sha256=sha(raw),
                    human_review_completed=False,proposed_training_eligibility=False))
    indexed={r["source_id"]:r for r in all_rows}
    rows=[indexed[v["source_id"]] for v in ids if v["source_id"] in indexed]
    need(len(rows)+len(skipped)==24 and len({s["pcm_sha256"] for s in rows})==len(rows),"no replacement or duplicate")
    (out/"ROSTER.json").write_bytes(ROSTER.read_bytes())
    write(out/"MANIFEST.json",dict(schema=SCHEMA,role="train",roster_sha256=sha(ROSTER.read_bytes()),
        registry_plan_sha256=roster["plan_sha256"],clock_preregistration_commit="9db369b83e12234048e321cd12759d3b55c643a0",
        hf_repo=roster["hf_repo"],hf_revision=roster["hf_revision"],partition=roster["partition"],
        attribution=plan["attribution"],license=plan["license"],raw_audio_decodes=len(ids),kws_forwards=0,
        model_updates=0,rows=rows,rejections=skipped,human_review_completed=False,formal_training_allowed=False,
        independent_evaluation_allowed=False,field_false_alarm_per_hour=None))


def read_rows(source):
    m=json.loads((source/"MANIFEST.json").read_bytes())
    need(m["schema"]==SCHEMA and m["roster_sha256"]==sha(ROSTER.read_bytes())==sha((source/"ROSTER.json").read_bytes())
         and m["kws_forwards"]==0 and not m["human_review_completed"],"authenticated source roster")
    for r in m["rows"]:
        facts=pcm_tools.inspect_wav(source/"wav"/(r["id"]+".wav"))
        need(r["role"]=="train" and all(r[k]==v for k,v in facts.items()),"full original TRAIN PCM")
    return m["rows"]


def screen(source, providers, out):
    rows=read_rows(source)
    byid={r["id"]:r for r in rows}
    heard=defaultdict(dict);coverage={}
    manifest_sha=sha((source/"MANIFEST.json").read_bytes())
    for p in PROVIDERS:
        d=providers/p; native=json.loads((d/"RESULT.json").read_bytes())
        model=json.loads((d/"MODEL.json").read_bytes())
        need(model["observer"]==asr.identity(p) and not model["target_conditioned"] and not native["target_conditioned"] and
             native["schema"]==SCHEMA and native["provider"]==p and native["source_manifest_sha256"]==manifest_sha
             and native["attempted"]==native["denominator"]==len(rows) and not native["initialization_error"],"original blind ASR evidence")
        need({s["id"] for s in native["rows"]}==byid.keys() and len(native["rows"])==len(rows),"full ASR roster")
        coverage[p]=0
        for item in native["rows"]:
            uid=item["id"];r=byid[uid]
            need(item["source_pcm_sha256"]==r["pcm_sha256"] and item["source_wav_sha256"]==r["wav_sha256"]
                 and item["source_samples"]==r["samples"] and item["target_conditioned"] is False,"PCM-bound ASR capture")
            if item["status"]=="failed":
                need(json.loads((d/"rows"/uid/"FAILED.json").read_bytes())==item,"authentic failure")
                continue
            need(item["status"]=="success" and item["input_coverage_complete"] and
                 json.loads((d/"rows"/uid/"CAPTURE.json").read_bytes())==item,"authentic native text")
            heard[uid][p]=item["raw_transcript"];coverage[p]+=1
        need(coverage[p]==native["complete"],"true completed-ASR count")
    blocked=set(json.loads(registry.PLAN_PATH.read_bytes())["excluded_words"])
    decisions=[]
    for uid,r in byid.items():
        texts=heard[uid];normalized=[text_policy.norm(s) for s in texts.values()]
        votes=normalized.count(text_policy.norm(r["transcript"]))
        veto=any(blocked.intersection(s.upper().split()) for s in normalized)
        tier="unknown"
        if set(texts)==set(PROVIDERS) and votes>=2 and not veto:
            tier="silver" if votes==3 else "weak"
        decisions.append(dict(id=uid,role="train",speaker=r["speaker"],source_id=r["source_id"],source_group_id=r["source_group_id"],
               tier=tier,reviewed_text_votes=votes,blocked_native_transcript=veto,raw_text_by_provider=texts,
               pcm_sha256=r["pcm_sha256"],human_review_completed=False,proposed_training_eligibility=False))
    ok=[d for d in decisions if d["tier"]!="unknown"]
    out.mkdir()
    write(out/"SCREEN.json",dict(schema=SCHEMA,role="train",source_manifest_sha256=manifest_sha,roster_sha256=sha(ROSTER.read_bytes()),
            source_rows=len(rows),coverage=coverage,admitted_for_diagnostic=len(ok),admitted_speakers=len({r["speaker"] for r in ok}),
            decisions=decisions,kws_forwards=0,optimizer_updates=0,human_review_completed=False,
            proposed_training_eligibility=False,formal_training_allowed=False,independent_evaluation_allowed=False))


if __name__=="__main__":
    mode,*args=sys.argv[1:]
    if mode=="generate":
        generate(Path(args[0]))
    elif mode=="observe":
        provider,source,out=args
        need(provider in PROVIDERS,"three preregistered unprompted text observers")
        observers.observe(provider,Path(source),Path(out),read_rows=read_rows,schema=SCHEMA)
    elif mode=="screen":
        screen(*map(Path,args))
    else:
        raise ValueError("unsupported mode")
