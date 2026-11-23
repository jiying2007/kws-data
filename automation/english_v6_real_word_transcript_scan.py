"""Look for existing whole-word English wake-phrase leads in LibriSpeech TRAIN transcripts."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

REPO = "openslr/librispeech_asr"
REV = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
PARTITIONS = ("clean/train.100", "clean/train.360", "other/train.500")
SPELLINGS = ("MERO", "MEERO", "MIRO", "MERROW", "MARROW")
PAIR = re.compile(r"\bHEY (" + "|".join(SPELLINGS) + r")\b")
ALONE = re.compile(r"\bMERO\b")
DIGEST = re.compile(r"[0-9a-f]{64}")


def normalized(text):
    raw=unicodedata.normalize("NFKD",text).upper()
    return " ".join(re.findall("[A-Z]+", raw))


def inspect(entry):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs=HfFileSystem(token=False)
    key=f"datasets/{REPO}@{REV}/{entry.path}"
    with fs.open(key,"rb",block_size=65536,cache_type="none") as stream:
        pf=pq.ParquetFile(stream,pre_buffer=False)
        seen=0
        digest=sha256()
        leads=[]
        mero_only=0
        for rg in range(pf.num_row_groups):
            texts=pf.read_row_group(rg,columns=["text"]).column("text").to_pylist()
            matches=[]
            for i,raw in enumerate(texts):
                if not isinstance(raw,str):
                    raise ValueError("missing source transcript")
                utf=raw.encode()
                digest.update(len(utf).to_bytes(4,"little"))
                digest.update(utf)
                words=normalized(raw)
                mero_only+=bool(ALONE.search(words))
                variants=sorted(set(PAIR.findall(words)))
                if variants:
                    matches.append((i,raw,variants))
            if matches:
                ids=pf.read_row_group(rg,columns=["id","speaker_id"]).to_pylist()
                for i,raw,v in matches:
                    leads.append(dict(id=ids[i]["id"],speaker_id=str(ids[i]["speaker_id"]),text=raw,spellings=v,row_group=rg,row=i))
            seen+=len(texts)
        if seen!=pf.metadata.num_rows or seen==0:
            raise ValueError("incomplete metadata scan")
    return dict(shard=entry.path,file_bytes=entry.size,upstream_lfs_sha256=entry.lfs.sha256,rows=seen,
                text_stream_sha256=digest.hexdigest(),contains_standalone_mero=mero_only,transcript_leads=leads)


def retry(entry):
    for attempt in range(3):
        try:
            return inspect(entry)
        except Exception:
            if attempt==2:
                raise
            time.sleep(2**attempt)


def scan(output):
    from huggingface_hub import HfApi
    api=HfApi(token=False)
    shards=[]
    for partition in PARTITIONS:
        items=sorted((e for e in api.list_repo_tree(repo_id=REPO,path_in_repo=partition,revision=REV,repo_type="dataset")
                      if re.fullmatch(re.escape(partition)+r"/\d+\.parquet",e.path)),key=lambda x:x.path)
        if not items or any(not e.lfs or not DIGEST.fullmatch(e.lfs.sha256) for e in items):
            raise ValueError("missing or unversioned shards: "+partition)
        shards.extend(items)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(retry,shards))
    leads=[dict(shard=r["shard"],**x) for r in results for x in r["transcript_leads"]]
    if len({r["id"] for r in leads})!=len(leads):
        raise ValueError("duplicate official utterance id")
    report=dict(schema="english-v6-librispeech-train-word-transcript-search-v1",upstream_repo=REPO,upstream_revision=REV,
                normalization="NFKD, uppercase, whole consecutive ASCII word tokens within one official utterance",
                searches=["HEY "+w for w in SPELLINGS],role="upstream TRAIN only; leads require provenance and direct audio review",
                totals={p:sum(r["rows"] for r in results if r["shard"].startswith(p+"/")) for p in PARTITIONS},
                total_utterances=sum(r["rows"] for r in results),matched_utterances_by_spelling={w:sum(w in x["spellings"] for x in leads) for w in SPELLINGS},
                mero_standalone_utterances=sum(r["contains_standalone_mero"] for r in results),
                shard_count=len(results),shards=results,transcript_leads=leads,decoded_audio=0,model_calls=0,admitted_human_positives=0)
    output.mkdir()
    (output/"RESULT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(report["totals"],report["matched_utterances_by_spelling"])


if __name__=="__main__":
    scan(Path(sys.argv[1]))
