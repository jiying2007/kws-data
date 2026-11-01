"""Prepare a source-only Git object proposal, never update any branch or authority.

Exact closed-retention inventory, original source bytes and history are enforced.
The output commit must be independently read back and applied with a head lease.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REPO = "jiying2007/kws-data"
BRANCH = "research/field-collection-pilot-20261102"
BASE = "docs/BRANCH_RETENTION_20261007.json"
FOLDER = "research/field-collection-pilot-20261102"
PATHS = {
    ".github/workflows/field-collection-pilot-20261102.yml",
    *(FOLDER+"/"+n for n in ("plan.py","test_plan.py","README.md","register_sources.py"))
}


def require(ok, why):
    if not ok: raise ValueError(why)


def pairs(items):
    out={}
    for key,value in items:
        require(key not in out,"DUPLICATE_JSON_KEY")
        out[key]=value
    return out


def git(*args):
    return subprocess.check_output(["git",*args],timeout=30).decode().strip()


def rest(path,record):
    response=subprocess.check_output(["gh","api","--method","POST",
        "repos/"+REPO+"/git/"+path,"--input","-"],
        input=json.dumps(record,ensure_ascii=False,allow_nan=False).encode("utf-8"),timeout=60)
    obj=json.loads(response,object_pairs_hook=pairs)
    require(type(obj.get("sha")) is str and re.fullmatch(r"[a-f0-9]{40}",obj["sha"]),"INVALID_GIT_OBJECT")
    return obj["sha"]


def propose(old,tree,head,old_body):
    require(type(old) is dict and old.get("schema")=="kws-public-branch-retention-v1"
            and old.get("repository")==REPO,"EXISTING_RETENTION_REQUIRED")
    retained=old["retained_files"]
    rows={x["path"]:x for x in retained}
    integrated=set(old["integration_files"])
    require(len(rows)==len(retained)==old["expected_retained_files"] and
            len(integrated)==len(old["integration_files"]) and
            set(rows).isdisjoint(integrated) and
            len(rows)+len(integrated)==old["expected_tracked_files"],"OLD_INVENTORY_DRIFT")
    require(set(tree)==set(rows)|integrated|PATHS and PATHS<=set(tree),
            "UNRELATED_FILE_CHANGED_OR_MISSING")
    for name,record in rows.items():
        require(tree[name]==(record["mode"],record["git_blob_sha1"],record["bytes"]),
                "HISTORICAL_SOURCE_CHANGED")
    changed=sorted(PATHS-set(rows))
    require(len(changed)==len(PATHS) or len(changed)==0,"PARTIAL_SOURCE_REGISTRATION")
    if not changed:
        return None
    new=copy.deepcopy(old)
    for name in changed:
        mode,sha,size=tree[name]
        require(mode=="100644" and type(size) is int and 0<size<=65536,
                "BOUNDED_TEXT_SOURCE_REQUIRED")
        new["retained_files"].append({"path":name,"mode":mode,
            "git_blob_sha1":sha,"bytes":size,"source_commit":head})
    new["expected_retained_files"]=len(new["retained_files"])
    new["expected_tracked_files"]=len(tree)
    new.setdefault("source_registration_history",[]).append({
        "campaign":"field-collection-pilot-20261102",
        "source_commit":head,
        "paths":changed,
        "previous_manifest_sha256":hashlib.sha256(old_body).hexdigest(),
        "previous_campaign_rows":[],
        "scope":"private field recording planning and structural audit only; "
                "no rights, label, real audio, model or release admission"})
    return new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get("GITHUB_REPOSITORY")==REPO and
            os.environ.get("GITHUB_EVENT_NAME")=="pull_request" and
            os.environ.get("GITHUB_HEAD_REF")==BRANCH,"FIXED_SOURCE_BRANCH_REQUIRED")
    event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes(),object_pairs_hook=pairs)
    require(event["pull_request"]["head"]["ref"]==BRANCH and
            event["pull_request"]["head"]["repo"]["full_name"]==REPO,
            "FIXED_PR_HEAD_REQUIRED")
    head=git("rev-parse","HEAD")
    require(head==event["pull_request"]["head"]["sha"] and
            git("status","--porcelain")=="","CLEAN_EXACT_HEAD_REQUIRED")
    root=Path.cwd().resolve()
    output=args.output.resolve()
    require(output.is_dir() and output != root and
            not root in output.parents,"OUTPUT_OUTSIDE_REPO_REQUIRED")
    tree={}
    for item in subprocess.check_output(["git","ls-tree","-r","-l","-z","HEAD"],timeout=30).split(b"\0"):
        if not item:continue
        meta,path=item.split(b"\t",1)
        mode,kind,blob,size=meta.decode("utf-8").split()
        require(kind=="blob" and mode in ("100644","100755"),"NO_LINKS_OR_SUBMODULES")
        tree[path.decode("utf-8")]=(mode,blob,int(size))
    old_body=Path(BASE).read_bytes()
    old=json.loads(old_body,object_pairs_hook=pairs)
    proposal=propose(old,tree,head,old_body)
    result={"schema":"field-pilot-retention-proposal-v1","parent_head":head,
        "ref_updated":False,"model_calls":0,
        "status":"already_registered" if proposal is None else "creating_objects"}
    if proposal is not None:
        body=(json.dumps(proposal,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
        (output/"registered-manifest.json").write_bytes(body)
        blob=rest("blobs",{"content":body.decode("utf-8"),"encoding":"utf-8"})
        require(blob==hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest(),
                "REGISTERED_MANIFEST_BLOB_MISMATCH")
        newtree=rest("trees",{"base_tree":git("rev-parse","HEAD^{tree}"),
            "tree":[{"path":BASE,"mode":"100644","type":"blob","sha":blob}]})
        candidate=rest("commits",{"message":"chore(research): register exact source-only field corpus pilot",
            "tree":newtree,"parents":[head]})
        result.update(status="proposal_created_ref_not_updated",commit_sha=candidate,
            tree_sha=newtree,manifest_blob=blob,manifest_sha256=hashlib.sha256(body).hexdigest(),
            previous_manifest_sha256=hashlib.sha256(old_body).hexdigest(),
            original_retained_files=old["expected_retained_files"],
            resulting_retained_files=proposal["expected_retained_files"])
    (output/"registration-proposal.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
