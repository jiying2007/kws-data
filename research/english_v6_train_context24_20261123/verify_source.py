"""Offline verification of native source and ASR review from released public archives."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import unicodedata
import subprocess
import wave
from zipfile import ZipFile


def need(condition,why):
    if not condition:
        raise ValueError(why)


def sha(b):
    return sha256(b).hexdigest()


def obj(z,path):
    return json.loads(z.read(path))


def norm(text):
    # Same specified English word convention as the source-only admission.
    plain=unicodedata.normalize("NFKC",text).casefold()
    return " ".join("".join(" " if unicodedata.category(c)[0]=="P" else c for c in plain).split())


def check(inputs, pins, result=None):
    snap=json.loads(pins.read_bytes())
    with ExitStack() as stack:
        bundles={}
        for k,s in snap["artifacts"].items():
            if k not in ("roster","source","screen","qwen","whisper","parakeet"):
                continue
            p=inputs/(k+".zip")
            need(len(p.read_bytes())==s["bytes"] and sha(p.read_bytes())==s["sha256"],"source archive "+k)
            z=stack.enter_context(ZipFile(p))
            need(z.testzip() is None and len(z.namelist())==len(set(z.namelist())),"unique ZIP members")
            bundles[k]=z
        src=bundles["source"];screen=obj(bundles["screen"],"SCREEN.json")
        roster=obj(bundles["roster"],"ROSTER.json"); mraw=src.read("MANIFEST.json"); m=json.loads(mraw)
        need(sha(src.read("ROSTER.json"))==m["roster_sha256"]==sha(bundles["roster"].read("ROSTER.json"))=="a6ac3da422fc98cdc826608ee66a11a44606feb19a98a0102648169f006851e9","prior roster")
        need(m["hf_revision"]==roster["hf_revision"]=="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1" and m["partition"]==roster["partition"]=="clean/train.100","immutable training-only corpus")
        need(roster["scanned_shards"][0]["lfs_sha256"]=="3098c6e44d1d49f8c62bd123775f1c492bda2cabd80f3ee70fe7800572371401","original HF shard")
        need(m["kws_forwards"]==screen["kws_forwards"]==screen["optimizer_updates"]==0 and not m["formal_training_allowed"]
             and not screen["human_review_completed"] and not screen["formal_training_allowed"] and screen["source_manifest_sha256"]==sha(mraw),"scope")
        ordered=roster["source_rows"]
        need(len(ordered)==24 and len({v["speaker"] for v in ordered})==12 and all(n==2 for n in Counter(v["speaker"] for v in ordered).values()),"original 12x2 roster")
        planned={r["source_id"]:r for r in ordered}; actual={r["id"]:r for r in m["rows"]}
        need({r["source_id"] for r in actual.values()}|{r["source_id"] for r in m["rejections"]}==planned.keys(),"all planned originals accounted for")
        native={};manifest_sha=sha(mraw);basic=Counter()
        for sid,r in actual.items():
            ref=planned[r["source_id"]]
            need(all(r[k]==v for k,v in ref.items()) and r["role"]=="train" and r["source_group_id"]=="librispeech-"+ref["speaker"] and
                 not r["proposed_training_eligibility"],"authentic source role")
            flac=src.read("flac/"+sid+".flac"); wav=src.read("wav/"+sid+".wav")
            need(sha(flac)==r["flac_sha256"] and sha(wav)==r["wav_sha256"],"original media")
            # Independent system decoder for the archived FLAC, not the generator's libsndfile implementation.
            proc=subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-f","flac","-i","pipe:0","-f","s16le","pipe:1"],input=flac,capture_output=True,check=True)
            with wave.open(BytesIO(wav),"rb") as f:
                need(f.getparams()[:3]==(1,2,16000) and f.getnframes()==r["samples"],"native PCM format")
                pcm=f.readframes(f.getnframes())
            need(pcm==proc.stdout and sha(pcm)==r["pcm_sha256"] and len(pcm)==2*r["samples"],"native FLAC-to-PCM equivalence")
            basic[bool(r["basic_audio_ok"])]+=1
        for provider in ("qwen","whisper","parakeet"):
            z=bundles[provider];doc=obj(z,"RESULT.json");model=obj(z,"MODEL.json")
            need(doc["source_manifest_sha256"]==manifest_sha and doc["provider"]==provider and doc["target_conditioned"] is False and
                 model["target_conditioned"] is False and doc["denominator"]==doc["attempted"]==len(actual),"unprompted original model")
            native[provider]={}
            for item in doc["rows"]:
                uid=item["id"];c=actual[uid]
                need(item["source_pcm_sha256"]==c["pcm_sha256"] and item["source_samples"]==c["samples"] and not item["target_conditioned"],"input joins")
                if item["status"]=="failed":
                    need(item==obj(z,"rows/"+uid+"/FAILED.json"),"failed source receipt")
                    continue
                need(item["status"]=="success" and item==obj(z,"rows/"+uid+"/CAPTURE.json"),"genuine original transcript")
                native[provider][uid]=item["raw_transcript"]
        decisions={r["id"]:r for r in screen["decisions"]}
        need(decisions.keys()==actual.keys(),"complete frozen screen")
        veto_words=set("hey hay mero meero miro marrow merrow".split())
        kept=[];stats=Counter();reasons=Counter()
        for ident,r in actual.items():
            texts={k:d[ident] for k,d in native.items() if ident in d}
            item=decisions[ident]
            need(item["raw_text_by_provider"]==texts and not item["human_review_completed"] and not item["proposed_training_eligibility"],"observed transcripts and human status")
            target=norm(r["transcript"]); votes=sum(norm(t)==target for t in texts.values())
            veto=any(veto_words.intersection(norm(t).split()) for t in texts.values())
            qualified=len(texts)==3 and votes>=2 and not veto
            tier=("silver" if votes==3 else "weak") if qualified else "unknown"
            need(item["tier"]==tier and item["reviewed_text_votes"]==votes and item["blocked_native_transcript"]==veto,"independent source admission")
            stats[tier]+=1
            if qualified:kept.append(r)
            else:reasons["missing_model" if len(texts)<3 else "wake_like_ASR" if veto else "insufficient_exact_votes"]+=1
        coverage={p:len(v) for p,v in native.items()}
        need(coverage==screen["coverage"] and len(kept)==screen["admitted_for_diagnostic"] and len({r["speaker"] for r in kept})==screen["admitted_speakers"],"independent denominators")
        out=dict(schema="english-v6-verified-natural-train-source-readback-v1",source_head=snap["source_head"],registration_head=snap["registry_head"],
                 actual_originals=len(actual),actual_readers=len({r["speaker"] for r in actual.values()}),
                 independently_decoded_flac=len(actual),distinct_original_pcm=len({r["pcm_sha256"] for r in actual.values()}),quality_flag_counts=dict(basic),
                 source_seconds=sum(r["samples"] for r in actual.values())/16000,three_asr_capture_counts=coverage,independent_tiers=dict(stats),
                 excludes=dict(reasons),admitted_originals=len(kept),admitted_readers=len({r["speaker"] for r in kept}),admitted_seconds=sum(r["samples"] for r in kept)/16000,
                 human_review_completed=False,formal_training_allowed=False,independent_evaluation_allowed=False,field_far_per_hour=None)
        if result:
            result.write_text(json.dumps(out,indent=2)+"\n")
        return out


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("inputs",type=Path);p.add_argument("pins",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();print(check(a.inputs,a.pins,a.output))
