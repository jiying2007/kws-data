"""Pre-register a distinct public TRAIN-only roster from the read-speech corpus metadata."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

PLAN_PATH = Path(__file__).with_name("english_v6_train_context24_plan.json")
sha = lambda data: hashlib.sha256(data).hexdigest()


def need(value, label):
    if not value:
        raise ValueError(label)


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def register(output):
    from huggingface_hub import HfApi,HfFileSystem
    import pyarrow.parquet as pq
    plan=json.loads(PLAN_PATH.read_bytes())
    name=plan["upstream_repository"]
    api=HfApi(token=False)
    rev=api.dataset_info(name).sha
    need(re.fullmatch(r"[a-f0-9]{40}",rev),"one immutable upstream revision")
    fs=HfFileSystem(token=False)
    exclusions=set(plan["excluded_speakers"]);blocked=set(plan["excluded_words"])
    need(len(exclusions)==976,"all earlier identifiable readers excluded")
    salt=plan["selection_salt"]
    def rank(role,ident):
        return sha((salt+"/"+role+"/"+ident).encode()),ident
    scanned=[]
    for partition in plan["partitions_in_order"]:
        entries=sorted((f for f in api.list_repo_tree(name,path_in_repo=partition,revision=rev,repo_type="dataset")
                        if re.fullmatch(re.escape(partition)+r"/\d+\.parquet",f.path)),key=lambda f:f.path)
        need(entries,"official HF sharded partition "+partition)
        grouped=defaultdict(list)
        for entry in entries[:plan["max_shards_per_partition"]]:
            lock=dict(path=entry.path,bytes=entry.size,lfs_sha256=entry.lfs.sha256)
            need(re.fullmatch("[a-f0-9]{64}",lock["lfs_sha256"]),"original immutable LFS SHA")
            path=f"datasets/{name}@{rev}/{entry.path}"
            with fs.open(path,"rb",block_size=64*1024,cache_type="none") as remote:
                file=pq.ParquetFile(remote)
                dest=[]
                for rg in range(file.num_row_groups):
                    cols=file.read_row_group(rg,columns=["id","speaker_id","chapter_id","text"],use_threads=False)
                    need(set(cols.column_names)=={"id","speaker_id","chapter_id","text"},"transcript-only columns")
                    for within,c in enumerate(cols.to_pylist()):
                        sid=str(c["speaker_id"]);chapter=str(c["chapter_id"]);uid=c["id"]
                        need(re.fullmatch(sid+"-"+chapter+r"-\d+",uid),"official numerical utterance identity")
                        transcript=c["text"].strip()
                        words=transcript.upper().split()
                        if sid in exclusions or not 8<=len(words)<=18 or blocked.intersection(words):
                            continue
                        if not all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*",word) for word in words):
                            continue
                        rec=dict(speaker=sid,chapter=chapter,source_id=uid,transcript=transcript,
                                 parquet_path=entry.path,row_group=rg,row_in_group=within)
                        grouped[sid].append(rec);dest.append(rec)
                meta_sha=sha(json.dumps(dest,sort_keys=True,separators=(",",":")).encode())
                scanned.append(dict(lock,partition=partition,row_groups=file.num_row_groups,total_rows=file.metadata.num_rows,
                                    text_eligible_rows=len(dest),eligible_metadata_sha256=meta_sha,
                                    available_speakers_so_far=len([s for s in grouped if len(grouped[s])>=2])))
            eligible=sorted((s for s,v in grouped.items() if len(v)>=2),key=lambda k:rank("speaker",k))
            if len(eligible)<12:
                continue
            selected=[]
            for sp in eligible[:12]:
                selected.extend(sorted(grouped[sp],key=lambda r:rank("utterance",r["source_id"]))[:2])
            need(len(selected)==len({r["source_id"] for r in selected})==24,"24 distinct preselected sources")
            output.mkdir()
            write(output/"ROSTER.json",dict(schema=plan["schema"],plan_sha256=sha(PLAN_PATH.read_bytes()),hf_repo=name,hf_revision=rev,
                  partition=partition,scanned_shards=scanned,source_rows=selected,raw_audio_decodes=0,kws_forwards=0,
                  new_asr_calls=0,optimizer_updates=0,role="train",training_allowed=False))
            return
    raise ValueError("No capacity in either permitted metadata-only prefix")


if __name__=="__main__":
    register(Path(sys.argv[1]))
