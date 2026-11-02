"""Optional one-time pinned public Qwen asset prefetch + exact offline file lock.

No text-to-speech or ASR inference is performed by this setup command.
Model downloads require explicit --download; normal CI never invokes it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import campaign as c
import runner

MODEL_DOWNLOAD_CEILING=12*1024**3
MIN_DISK_RESERVE=2*1024**3


def destination_name(model_id,revision):
    c.require(model_id in c.MODEL_FAMILIES.values() and
              type(revision) is str and c._REV.fullmatch(revision),
              "UNPINNED_PUBLIC_MODEL")
    kind=next(k for k,v in c.MODEL_FAMILIES.items() if v==model_id)
    return kind+"-"+revision[:16]


def exact_files(root,model_dir):
    path=Path(model_dir)
    c.require(path.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)),
              "MODEL_DIR_ALIAS")
    out=[]
    for target in sorted(path.rglob("*")):
        c.require(not target.is_symlink(),"MODEL_SYMLINK")
        if target.is_dir():continue
        rel=target.relative_to(path).as_posix()
        c.require(target.is_file() and ".cache" not in target.relative_to(path).parts,
                  "UNEXPECTED_MODEL_METADATA")
        # File sha256 will be recomputed by runner.validate_models before inference.
        length,digest=runner._safe_file(target,limit=runner.MAX_MODEL_BYTES)
        out.append({"path":rel,"bytes":length,"sha256":digest})
    c.require(1<=len(out)<=runner.MAX_MODEL_FILES,"MODEL_FILE_COUNT")
    c.require(sum(x["bytes"] for x in out)<=runner.MAX_MODEL_BYTES,"MODEL_SIZE_LIMIT")
    return out


def prefetch(plan,root,*,max_download_bytes):
    c.validate_plan(plan)
    c.require(type(max_download_bytes) is int and
              0<max_download_bytes<=MODEL_DOWNLOAD_CEILING,
              "EXPLICIT_PUBLIC_DOWNLOAD_BUDGET")
    try:
        from huggingface_hub import HfApi,snapshot_download
    except ImportError as error:
        raise ValueError("HUGGINGFACE_HUB_REQUIRED_FOR_DOWNLOAD") from error
    models=sorted({(v["model_id"],v["model_revision"]) for v in plan["voices"]})
    plan_rows=[]
    total=0
    api=HfApi()
    for mid,revision in models:
        metadata=api.model_info(repo_id=mid,revision=revision,
                                files_metadata=True,token=False)
        c.require(metadata.sha==revision and getattr(metadata,"gated",False) is False and
                  getattr(metadata,"private",False) is False,"PUBLIC_PIN_NOT_AUTHENTICATED")
        sizes=[getattr(s,"size",None) for s in metadata.siblings or ()]
        c.require(sizes and all(type(s) is int and 0<=s<MODEL_DOWNLOAD_CEILING
                                for s in sizes),"MODEL_SIZE_METADATA_REQUIRED")
        count=sum(sizes)
        total+=count
        c.require(total<=max_download_bytes,"EXCEEDED_DOWNLOAD_BUDGET")
        plan_rows.append((mid,revision,count))
    c.require(root.is_dir() and
              shutil.disk_usage(root).free>=total+MIN_DISK_RESERVE,
              "INSUFFICIENT_LOCAL_DISK")
    for mid,revision,count in plan_rows:
        dest=root/destination_name(mid,revision)
        c.require(not dest.exists(),"MODEL_DESTINATION_MUST_BE_NEW")
        dest.mkdir()
        result=snapshot_download(repo_id=mid,revision=revision,local_dir=str(dest),
                                 token=False)
        c.require(Path(result).resolve()==dest.resolve(),"UNEXPECTED_HF_DOWNLOAD_ROOT")
        # Metadata directory is generated only in the new owned destination.
        # It is not loaded by the runtime and must never be an undeclared input.
        metadata=dest/".cache"
        if metadata.exists():
            c.require(metadata.is_dir() and not metadata.is_symlink(),
                      "DOWNLOAD_CACHE_ALIAS")
            shutil.rmtree(metadata)
    return {"downloaded_models":len(plan_rows),"download_budget_bytes":total,
            "model_calls":0,"training_admitted":False}


def lock_existing(plan,root):
    c.validate_plan(plan)
    records=[]
    for mid,revision in sorted({(v["model_id"],v["model_revision"])
                                for v in plan["voices"]}):
        rel=destination_name(mid,revision)
        records.append({"model_id":mid,"revision":revision,
            "relative_dir":rel,"files":exact_files(root,root/rel)})
    lock={"schema":runner.MODEL_LOCK_SCHEMA,"models":records}
    runner.validate_models(plan,lock,root)
    return lock


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--max-download-bytes",type=int,default=8*1024**3)
    args=parser.parse_args(argv)
    try:
        plan=c.validate_plan(c.read_json(args.plan))
        c.require(args.output.parent.is_dir() and not args.output.exists(),
                  "NEW_MODEL_LOCK_REQUIRED")
        if args.download:prefetch(plan,args.model_root,max_download_bytes=args.max_download_bytes)
        value=lock_existing(plan,args.model_root)
        runner._once(args.output,value)
    except (OSError,ValueError) as error:
        print("MODEL_ASSETS_NOT_LOCKED: "+type(error).__name__,file=sys.stderr)
        return 2
    print(json.dumps({"status":"MODEL_INPUT_BYTES_LOCKED_NOT_TRAINING",
        "models":len(value["models"]),"model_calls":0,
        "training_admitted":False}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
