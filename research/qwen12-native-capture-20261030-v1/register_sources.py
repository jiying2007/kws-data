"""Propose a manifest-only Git commit after source tests; never move a ref.

Only the exact seven capture files are mutable here. Original retained evidence,
verification policy and main remain untouched. The operator reads back the
proposal and updates the existing PR branch with an expected-head lease.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REPO='jiying2007/kws-data'
CAMPAIGN='qwen12-native-capture-20261030-v1'
BRANCH='research/'+CAMPAIGN
MANIFEST='docs/BRANCH_RETENTION_20261007.json'
PATHS={'.github/workflows/'+CAMPAIGN+'.yml',*[BRANCH+'/'+name for name in
    ('fresh_contract.py','test_fresh_contract.py','register_sources.py','worker.py','run_capture.py','test_capture_entry.py')]}


def require(value,message):
    if not value:raise ValueError(message)


def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON key');result[key]=value
    return result


def git(*args):
    return subprocess.check_output(['git',*args],timeout=30)


def gh(endpoint,value):
    raw=subprocess.check_output(['gh','api','--method','POST','repos/'+REPO+'/git/'+endpoint,'--input','-'],
        input=json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),timeout=60)
    result=json.loads(raw,object_pairs_hook=unique)
    require(type(result.get('sha')) is str and re.fullmatch('[0-9a-f]{40}',result['sha']),'Git object identity')
    return result['sha']


def proposed_manifest(old,tree,head,old_raw):
    require(old['schema']=='kws-public-branch-retention-v1' and old['repository']==REPO,'retention identity')
    rows={r['path']:r for r in old['retained_files']};integration=set(old['integration_files'])
    require(len(rows)==len(old['retained_files'])==old['expected_retained_files'] and
        len(integration)==len(old['integration_files']) and not rows.keys()&integration and
        len(rows)+len(integration)==old['expected_tracked_files'],'existing inventory shape')
    require(PATHS<=set(tree) and set(tree)==set(rows)|integration|PATHS,'only exact capture source paths may be registered')
    for name,row in rows.items():
        if name not in PATHS:
            require(tree[name]==(row['mode'],row['git_blob_sha1'],row['bytes']),'retained evidence changed: '+name)
    changed=sorted(name for name in PATHS if name not in rows or
        tree[name]!=(rows[name]['mode'],rows[name]['git_blob_sha1'],rows[name]['bytes']))
    if not changed:return None
    result=copy.deepcopy(old);current={r['path']:r for r in result['retained_files']}
    for name in changed:
        mode,blob,size=tree[name]
        require(mode=='100644' and 0<size<=65536,'bounded regular campaign source required')
        row={'path':name,'mode':mode,'git_blob_sha1':blob,'bytes':size,'source_commit':head}
        if name in current:current[name].update(row)
        else:result['retained_files'].append(row)
    result['expected_retained_files']=len(result['retained_files'])
    result['expected_tracked_files']=len(tree)
    result.setdefault('source_registration_history',[]).append({
        'campaign':CAMPAIGN,'source_commit':head,'paths':changed,
        'previous_manifest_sha256':hashlib.sha256(old_raw).hexdigest(),
        'previous_campaign_rows':[copy.deepcopy(rows[n]) for n in changed if n in rows],
        'scope':'prospective fresh36 capture source; source tests do not execute ASR, training or qualification'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get('GITHUB_REPOSITORY')==REPO and os.environ.get('GITHUB_EVENT_NAME')=='pull_request' and
        os.environ.get('GITHUB_HEAD_REF')==BRANCH,'fixed same-repository PR context')
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_bytes())
    require(event.get('number')==36 and event['pull_request']['head']['repo']['full_name']==REPO,'fixed PR36')
    head=git('rev-parse','HEAD').decode().strip()
    require(head==event['pull_request']['head']['sha'] and not git('status','--porcelain'),'clean exact PR head')
    output=args.output.resolve();root=Path.cwd().resolve()
    require(output.is_dir() and output!=root and root not in output.parents,'output outside checkout required')
    tree={}
    for item in git('ls-tree','-r','-l','-z','HEAD').split(b'\0'):
        if not item:continue
        meta,name=item.split(b'\t',1);mode,kind,blob,size=meta.decode().split()
        require(kind=='blob','no submodules');tree[name.decode()]=(mode,blob,int(size))
    raw=Path(MANIFEST).read_bytes();old=json.loads(raw,object_pairs_hook=unique)
    proposed=proposed_manifest(old,tree,head,raw)
    receipt={'schema':'qwen12-source-registration-proposal-v1','parent_head':head,'ref_updated':False,
        'model_calls':0,'status':'already_registered' if proposed is None else 'object_creation_started'}
    if proposed is not None:
        body=(json.dumps(proposed,ensure_ascii=False,indent=2)+'\n').encode()
        (output/'registered-manifest.json').write_bytes(body)
        blob=gh('blobs',{'content':body.decode(),'encoding':'utf-8'})
        require(blob==hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(),'created manifest blob mismatch')
        newtree=gh('trees',{'base_tree':git('rev-parse','HEAD^{tree}').decode().strip(),
            'tree':[{'path':MANIFEST,'mode':'100644','type':'blob','sha':blob}]})
        commit=gh('commits',{'message':'fix(research): register exact fresh Qwen12 capture source identities',
            'tree':newtree,'parents':[head]})
        receipt.update(status='proposal_created_ref_not_updated',commit_sha=commit,tree_sha=newtree,
            manifest_blob=blob,manifest_sha256=hashlib.sha256(body).hexdigest(),
            previous_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            retained_files=proposed['expected_retained_files'],tracked_files=proposed['expected_tracked_files'])
    (output/'registration-proposal.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
