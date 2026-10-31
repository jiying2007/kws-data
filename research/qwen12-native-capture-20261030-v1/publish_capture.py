"""Post-capture-only publication to a new evidence Git ref; no model execution.

The job token exists only in this separate host process. Publish only the
bounded, hash-checked artifact snapshot, never runtime/download directories.
Create-only Git refs make existing evidence immutable to this publisher.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fresh_contract as fresh
from run_capture import REPO,EXECUTION_BRANCH,CLAIM_REF,validate_execution
EVIDENCE_REF='refs/heads/evidence/qwen12-native-capture-once-20261031-v1'
WORKFLOW='.github/workflows/qwen12-native-capture-20261030-v1.yml'
MIB=1024**2


def snapshot(root):
    """No reconstruction of observations; recheck the exact published snapshot."""
    root=Path(root)
    fresh.require(root.is_dir() and not root.is_symlink(),'artifact snapshot required')
    for path in root.rglob('*'):
        fresh.require(not path.is_symlink() and (path.is_dir() or path.is_file()),'regular artifact members')
    names={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if names=={'publication-failure.json'}:
        raw=fresh.bounded_read(root/'publication-failure.json',65536)
        value=json.loads(raw)
        fresh.require(fresh.canonical(value)==raw,'canonical failure receipt')
        fresh.require(value.get('status')=='PUBLICATION_FAILED' and value.get('training_admitted') is False,'failure-only artifact')
        members={'publication-failure.json':raw}
    else:
        raw=fresh.bounded_read(root/'artifact-freeze.json',2*MIB)
        value=json.loads(raw)
        fresh.require(fresh.canonical(value)==raw,'canonical artifact freeze')
        fresh.require(value.get('schema')=='fresh-qwen12-native-artifact-v1' and
            value.get('training_admitted') is False and value.get('qualification_allowed') is False and
            type(value.get('files')) is dict and len(value['files'])<=4096,'artifact freeze shape')
        fresh.require(names==set(value['files'])|{'artifact-freeze.json'},'closed artifact membership')
        members={'artifact-freeze.json':raw};total=0
        for name,record in value['files'].items():
            path=Path(name)
            fresh.require(type(name) is str and not path.is_absolute() and path.as_posix()==name and
                all(p not in ('','.','..') for p in name.split('/')),'safe artifact member')
            body=fresh.bounded_read(root/path,32*MIB);total+=len(body)
            fresh.require(total<=256*MIB and type(record.get('bytes')) is int and record['bytes']==len(body) and
                record.get('sha256')==hashlib.sha256(body).hexdigest(),'artifact byte binding')
            members[name]=body
        fresh.require(type(value.get('raw_bytes')) is int and value['raw_bytes']==total,'artifact total bytes')
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,body in sorted(members.items()):
            entry=zipfile.ZipInfo(name,(2026,10,31,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o100644<<16;archive.writestr(entry,body)
    payload=target.getvalue()
    fresh.require(0<len(payload)<=64*MIB,'bounded Git evidence ZIP')
    return payload,len(members)


def publish(env,event,claim,root,api):
    head,run=validate_execution(env,event,claim)
    repo=api('GET','')
    fresh.require(repo.get('full_name')==REPO and repo.get('private') is False,'public repository required')
    actual=api('GET','/actions/runs/'+run)
    fresh.require(actual.get('id')==int(run) and actual.get('head_sha')==head and
        actual.get('head_branch')==EXECUTION_BRANCH and actual.get('event')=='push' and
        actual.get('run_attempt')==1 and actual.get('path')==WORKFLOW,'actual workflow identity required')
    source=api('GET','/git/commits/'+head)
    fresh.require(source.get('sha')==head and re.fullmatch('[0-9a-f]{40}',source.get('tree',{}).get('sha','')),'source commit identity')
    payload,count=snapshot(root)
    receipt={'schema':'fresh-qwen12-native-git-publication-v1','source_commit':head,'run_id':int(run),
        'run_attempt':1,'repository':REPO,'workflow':WORKFLOW,'evidence_ref':EVIDENCE_REF,
        'source_tree':source['tree']['sha'],'zip_bytes':len(payload),'zip_sha256':hashlib.sha256(payload).hexdigest(),'member_count':count,
        'training_admitted':False,'qualification_allowed':False,'model_completion_established':False}
    entries=[]
    for name,body in [('native-evidence.zip',payload),('publication.json',fresh.canonical(receipt))]:
        blob=api('POST','/git/blobs',{'content':base64.b64encode(body).decode(),'encoding':'base64'})
        fresh.require(blob.get('sha')==hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),'created blob readback')
        entries.append({'path':'research/'+fresh.CAMPAIGN+'/evidence/'+name,'mode':'100644','type':'blob','sha':blob['sha']})
    tree=api('POST','/git/trees',{'base_tree':source['tree']['sha'],'tree':entries})
    fresh.require(re.fullmatch('[0-9a-f]{40}',tree.get('sha','')),'created tree identity')
    commit=api('POST','/git/commits',{'message':'evidence: retain single fresh Qwen12 native run '+run,
        'tree':tree['sha'],'parents':[head]})
    fresh.require(re.fullmatch('[0-9a-f]{40}',commit.get('sha','')),'created evidence commit identity')
    reference=api('POST','/git/refs',{'ref':EVIDENCE_REF,'sha':commit['sha']})
    fresh.require(reference.get('ref')==EVIDENCE_REF and reference.get('object',{}).get('sha')==commit['sha'],'new evidence ref readback')
    return {**receipt,'evidence_commit':commit['sha'],'status':'PUBLISHED_NATIVE_OUTCOME_REQUIRES_REVIEW'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim',required=True,type=Path);parser.add_argument('--evidence',required=True,type=Path)
    args=parser.parse_args();env=os.environ
    event=json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes());claim=json.loads(args.claim.read_bytes())
    head,run=validate_execution(env,event,claim)
    allowed={('GET',''),('GET','/actions/runs/'+run),('GET','/git/commits/'+head)}
    allowed|={('POST','/git/'+name) for name in ('blobs','trees','commits','refs')}
    def api(method,path,body=None):
        fresh.require((method,path) in allowed,'fixed publication API route')
        command=['gh','api','--method',method,'repos/'+REPO+path]
        encoded=None
        if body is not None:command+=['--input','-'];encoded=fresh.canonical(body)
        result=subprocess.run(command,input=encoded,capture_output=True,timeout=90)
        fresh.require(result.returncode==0 and len(result.stdout)<=2*MIB,'publication API failed; no retry')
        return json.loads(result.stdout)
    result=publish(env,event,claim,args.evidence,api)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'PUBLICATION_FAILED_NO_RETRY','error_type':type(error).__name__}),file=sys.stderr)
        raise SystemExit(1)
