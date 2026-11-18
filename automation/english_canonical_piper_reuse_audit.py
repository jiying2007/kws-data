"""Screen only canonical cells in an already generated, immutable Piper corpus.

No model calls, TTS, retries, model training or retrospective resplitting.
Historic /i:/ Mero and /z/ Vosen source rows are excluded by predeclared cell ID.
"""
import argparse, collections, hashlib, io, json, os, re, sys, unicodedata, urllib.request, wave, zipfile
from pathlib import Path, PurePosixPath

RUN=44107239806
HEAD="8b591ace9e426971749e099a001f124137c3f29c"
ART=14339980867
NAME="english-phonetic192-corpus-44107239806"
BYTES=24295068
SHA="529c09953fa54914b600de0d23cca1477a78dddff620ac5be7a297b0b308d744"
REPO="jiying2007/kws-data"
CANONICAL={"mero-direct-eh":(2,"mero-us-eh-v1","Hey Mero.","hˈeɪ mˈɛɹoʊ."),
           "vosen-direct-s":(1,"vosen-us-s-v1","Hey Vosen.","hˈeɪ vˈoʊsən.")}
ORDINARY={"ordinary-time":"What time is it?",
          "ordinary-light":"Please turn on the kitchen light.",
          "ordinary-package":"The package is by the front door.",
          "ordinary-window":"Could you open the window?"}
LEGACY={"mero-direct-i","vosen-direct-z"}
PROVIDERS=("qwen","whisper","parakeet")
M_ALIAS={"hey mero","hey merrow","hey marrow","hay mero","hay merrow","hay marrow"}
V_ALIAS={"hey vosen","hey vosin","hey vohsen","hay vosen","hay vosin","hay vohsen"}
OLD_TEXT={"hey meero","hay meero","hey miro","hay miro",
          "hey vozen","hay vozen","hey vozin","hay vozin",
          "hey vozan","hay vozan","hey vohzen","hay vohzen"}
M_PHONE=re.compile(r"h?eɪmɛr(?:oʊ|ou|o)\Z")
V_PHONE=re.compile(r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n\Z")
ROLES={"train":6,"development":2,"calibration":2,"locked-test":2}

def must(ok, reason):
    if not ok: raise ValueError(reason)

def h(b): return hashlib.sha256(b).hexdigest()

def norm(t):
    t=unicodedata.normalize("NFKC",t).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
                            for c in t).split())

def phones(tokens):
    must(isinstance(tokens,list) and tokens and all(isinstance(x,str) for x in tokens),
         "PHONES_MISSING")
    value=re.sub(r"[\sˈˌːˑ͜͡]","",unicodedata.normalize("NFKC","".join(tokens)))
    return value.replace("ɹ","r").replace("ɾ","r")

def blind(texts, raw_phone):
    """Classifier never sees requested label or source role."""
    if set(texts)!=set(PROVIDERS) or raw_phone is None or any(not texts.get(p) for p in PROVIDERS):
        return ("unknown",None,"MISSING_OBSERVATION")
    t=[norm(texts[p]) for p in PROVIDERS]
    if any(v in OLD_TEXT for v in t): return ("unknown",None,"OLD_SPELLING_NOT_SUPPORTED")
    p=phones(raw_phone)
    if M_PHONE.fullmatch(p): label,aliases,other=2,M_ALIAS,V_ALIAS
    elif V_PHONE.fullmatch(p): label,aliases,other=1,V_ALIAS,M_ALIAS
    else: label,aliases,other=None,None,None
    if label is not None:
        if any(v in other for v in t): return ("unknown",None,"OTHER_WAKE")
        n=sum(v in aliases for v in t)
        if n<2 or (not p.startswith("h") and n!=3):
            return ("unknown",None,"BLIND_VOTES_INSUFFICIENT")
        return ("silver" if n==3 else "weak",label,"COMPLETE_PHONES_AND_BLIND_TEXT")
    if any(re.search(r"\b(?:hey|hay)\b",v) for v in t):
        return ("unknown",None,"POSSIBLE_WAKE_NOT_NEGATIVE")
    n=max((sum(v==norm(s) for v in t) for s in ORDINARY.values()),default=0)
    if n>=2: return ("silver" if n==3 else "weak",0,"FULL_ORDINARY_NO_WAKE")
    return ("unknown",None,"NO_TRUSTED_NEGATIVE")

def fetch(token):
    must(token,"TOKEN_REQUIRED")
    def get(url,limit):
        req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json","User-Agent":"canonical-reuse-once"})
        with urllib.request.urlopen(req,timeout=90) as f: data=f.read(limit+1)
        must(0<len(data)<=limit,"DOWNLOAD_LIMIT")
        return data
    api=f"https://api.github.com/repos/{REPO}/actions/artifacts/{ART}"
    meta=json.loads(get(api,1024*1024))
    must(meta["id"]==ART and meta["name"]==NAME and meta["expired"] is False
         and meta["size_in_bytes"]==BYTES and meta["digest"]=="sha256:"+SHA
         and meta["workflow_run"]["id"]==RUN and meta["workflow_run"]["head_sha"]==HEAD,
         "ORIGINAL_ARTIFACT_IDENTITY")
    data=get(api+"/zip",40*1024*1024)
    must(len(data)==BYTES and h(data)==SHA,"ORIGINAL_ZIP_SHA256")
    return data

def read_zip(raw):
    data={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        total=0
        for item in z.infolist():
            p=PurePosixPath(item.filename)
            must(not item.is_dir() and not p.is_absolute()
                 and p.as_posix()==item.filename and ".." not in p.parts
                 and item.file_size<=4*1024*1024 and item.filename not in data
                 and (item.external_attr>>16)&0o170000!=0o120000,"ARCHIVE_PATH_OR_SIZE")
            total+=item.file_size
            must(total<180*1024*1024,"EXPANDED_ARCHIVE_LIMIT")
            data[item.filename]=z.read(item)
    manifest=json.loads(data["FILES.json"])["files"]
    must(len(data)==len(manifest)+1
         and {r["path"] for r in manifest}==set(data)-{"FILES.json"},"SOURCE_FILE_CLOSURE")
    for entry in manifest:
        b=data[entry["path"]]
        must(len(b)==entry["bytes"] and h(b)==entry["sha256"],"SOURCE_MEMBER_DIGEST")
    return data

def evaluate(data):
    corpus=json.loads(data["CORPUS_MANIFEST.json"])
    evidence=json.loads(data["MACHINE_EVIDENCE.json"])
    must(len(corpus["rows"])==len(evidence["rows"])==192
         and corpus["campaign_id"]=="piper-libritts-direct-34-57-v1"
         and corpus["valid_count"]==192 and corpus["duplicate_pcm_count"]==0,
         "EXPECTED_192_UNIQUE_CORPUS")
    packets={r["id"]:r for r in evidence["rows"]}
    must(len(packets)==192 and set(packets)=={r["id"] for r in corpus["rows"]},
         "EXACT_OBSERVER_ROSTER")
    selected=[]; skipped=[]; groups=collections.defaultdict(list)
    for row in corpus["rows"]:
        cell=row["cell_id"]
        if cell in LEGACY:
            skipped.append(row["id"]); continue
        must(cell in set(CANONICAL)|set(ORDINARY)
             and row["generation_status"]=="success","EXACT_CANONICAL_CELL")
        if cell in CANONICAL:
            label,variant,text,phon=CANONICAL[cell]
            must((row["requested_class"],row["pronunciation_hypothesis"],
                  row["requested_text"],row["raw_phonemes"])==(label,variant,text,phon)
                 and "".join(row["actual_phonemes"])==phon,"ONLY_BRAND_DIRECT_REQUEST")
        else:
            must(row["requested_class"]==0
                 and row["requested_text"]==ORDINARY[cell],"ORDINARY_INPUT")
        pcm=data[row["pcm_path"]]; wav=data[row["wav_path"]]
        must(h(pcm)==row["pcm_sha256"] and h(wav)==row["wav_sha256"],"AUDIO_HASH")
        with wave.open(io.BytesIO(wav)) as stream:
            must((stream.getnchannels(),stream.getsampwidth(),stream.getframerate())
                 ==(1,2,16000) and stream.getnframes()==row["pcm_samples"]
                 and stream.readframes(stream.getnframes())==pcm,"WAV_PCM_MATCH")
        packet=packets[row["id"]]
        trans={}; phonetic=None
        for provider in PROVIDERS+("phoneme",):
            xs=([packet.get("phone_observation")] if provider=="phoneme"
                else [a for a in packet.get("observations",[])
                      if a.get("recognizer_id")==provider])
            if len(xs)!=1 or not isinstance(xs[0],dict): continue
            obs=xs[0]; path=obs.get("capture_path")
            if path not in data or h(data[path])!=obs.get("capture_sha256"): continue
            record=json.loads(data[path])
            must(record.get("source_pcm_sha256")==row["pcm_sha256"]
                 and record.get("source_wav_sha256")==row["wav_sha256"]
                 and record.get("source_samples")==row["pcm_samples"]
                 and record.get("target_conditioned") is False,"CAPTURE_SOURCE_BINDING")
            if record.get("status")!="success" or not record.get("input_coverage_complete") \
                    or not record.get("decoder_output_complete"): continue
            if provider=="phoneme": phonetic=record.get("phone_tokens")
            else: trans[provider]=record.get("raw_text")
        tier,label,reason=blind(trans,phonetic)
        if tier!="unknown" and label!=row["requested_class"]:
            tier,label,reason="unknown",None,"REQUEST_CONTRADICTED_BY_OBSERVATION"
        if tier!="unknown" and label in (1,2) and row["pcm_samples"]>28672:
            tier,label,reason="unknown",None,"POSITIVE_DURATION_OVER_1792_MS"
        item={"id":row["id"],"cell_id":cell,"role":row["role"],
              "source_group_id":row["source_group_id"],"tier":tier,
              "class":label,"reason":reason,"pcm_sha256":row["pcm_sha256"],
              "pcm_samples":row["pcm_samples"],"complete_observers":
                  len(trans)==3 and phonetic is not None}
        selected.append(item);groups[row["source_group_id"]].append(item)
    must(len(selected)==144 and len(skipped)==48 and len(groups)==24
         and all(len(rows)==6 for rows in groups.values()),"CANONICAL_ONLY_144_ROSTER")
    complete={g for g,rows in groups.items()
              if {r["class"] for r in rows if r["tier"]!="unknown"}=={0,1,2}}
    counts=collections.Counter(groups[g][0]["role"] for g in complete)
    passes=all(counts[role]>=n for role,n in ROLES.items())
    return {"schema":"english-canonical-piper-reuse-screen-v1",
            "status":"PASS_CANONICAL_SOURCE_SCREEN" if passes else "STOP_CANONICAL_SOURCE_COVERAGE",
            "source":{"run_id":RUN,"artifact_id":ART,"sha256":SHA},
            "selected_rows":144,"excluded_old_mero_and_voiced_vosen_rows":48,
            "excluded_row_id_hash":h("".join(sorted(skipped)).encode()),
            "balanced_groups_by_role":{r:counts[r] for r in ROLES},
            "minimum_balanced_groups":ROLES,
            "tier_counts":dict(collections.Counter(x["tier"] for x in selected)),
            "reason_counts_by_cell":{cell:dict(collections.Counter(
                r["reason"] for r in selected if r["cell_id"]==cell))
                for cell in sorted(set(CANONICAL)|set(ORDINARY))},
            "training_snapshot_allowed":passes,
            "training_rows_when_stopped":0 if not passes else sum(
                r["role"]=="train" and r["tier"]!="unknown"
                and r["source_group_id"] in complete for r in selected),
            "rows":selected,"new_tts_calls":0,"new_asr_calls":0,
            "new_phone_calls":0,"optimizer_updates":0,"new_model_weights":False,
            "human_gold":False,"independent_real_far_frr":False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--output")
    a=ap.parse_args()
    must(a.source_check!=a.run,"ONE_MODE")
    must(len(CANONICAL)==2 and len(ORDINARY)==4
         and not set(CANONICAL)&LEGACY and not (M_ALIAS|V_ALIAS)&OLD_TEXT,
         "CANONICAL_ONLY_POLICY")
    if a.source_check:
        print('{"status":"SOURCE_ONLY","new_model_calls":0}')
        return 0
    must(a.output and os.environ.get("GITHUB_REPOSITORY")==REPO
         and os.environ.get("GITHUB_EVENT_NAME")=="push"
         and os.environ.get("GITHUB_RUN_ATTEMPT")=="1","PUBLIC_ONCE_ONLY")
    data=read_zip(fetch(os.environ.get("GH_TOKEN")))
    result=evaluate(data)
    path=Path(a.output)
    must(not path.exists(),"CREATE_ONCE")
    path.mkdir(parents=True)
    (path/"SCREEN.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({x:result[x] for x in ("status","selected_rows","tier_counts",
           "balanced_groups_by_role","training_snapshot_allowed")}))
    return 0

if __name__=="__main__":
    sys.exit(main())
