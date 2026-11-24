"""Capture the actual Git-bound B21 scientific producer before execution.

This reads code and registered text resources only. No archive, waveform,
network, native observer or KWS model is opened or executed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b20_primary_eval as prior

ROOT = Path(__file__).resolve().parents[1]
REPO = 'jiying2007/kws-data'
ROLES = {
 'train': dict(branch='research/en-v6-b21-blank-retention-train-once-20261124',
  workflow='.github/workflows/english-v6-b21-blank-retention-train-once.yml',
  roots=('english_v6_b21_competing_margin_pair','test_english_v6_b21_competing_margin_pair')),
 'source': dict(branch='research/en-v6-b21-primary-sources-once-20261124',
  workflow='.github/workflows/english-v6-b21-primary-sources-once.yml',
  roots=('english_v6_b21_source','test_english_v6_b21_sources'))}
SUFFIXES = {'.json','.txt','.yml','.yaml','.toml','.csv','.tsv','.md','.c'}
EXTRA = ('automation/english_v6_b21_panels',
         'automation/english_v6_b21_competing_margin_design.json',
         'automation/english_v6_b21_train_roster.json',
         'automation/english_v6_b21_source_workflow_template.yml')


def need(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def blob(raw): return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args])


def tree_entries(raw, allowed=None):
    result={}
    for item in raw.split(b'\0'):
        if not item: continue
        left,name=item.split(b'\t',1); mode,kind,digest=left.decode().split()
        name=name.decode('utf-8',errors='strict')
        if allowed is not None and name not in allowed: continue
        need(kind == 'blob' and mode in ('100644','100755') and re.fullmatch(r'[0-9a-f]{40}',digest),
             'regular immutable Git source entries only')
        need(name not in result and not Path(name).is_absolute() and '..' not in Path(name).parts, 'unique safe Git source paths')
        result[name]=dict(mode=mode,git_blob_sha1=digest)
    return result


def validate_event(role,prepared,head,parents,changed,env,event):
    spec=ROLES[role]
    need(re.fullmatch(r'[0-9a-f]{40}',prepared or '') and re.fullmatch(r'[0-9a-f]{40}',head or '')
         and parents == [prepared] and head != prepared and changed == [spec['workflow']],
         'one reviewed Git predecessor and workflow-only activation')
    ref='refs/heads/'+spec['branch']
    need(all(env.get(k)==v for k,v in dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY=REPO,
         GITHUB_EVENT_NAME='push',GITHUB_REF=ref,GITHUB_RUN_ATTEMPT='1',GITHUB_SHA=head).items())
         and re.fullmatch(r'[1-9][0-9]*',env.get('GITHUB_RUN_ID','')),
         'actual first public B21 run checkout')
    need(event.get('before')==prepared and event.get('after')==head and event.get('ref')==ref
         and event.get('forced') is event.get('deleted') is False
         and event.get('repository',{}).get('full_name')==REPO and event['repository'].get('private') is False,
         'exact public push identity before producer capture')
    return dict(role=role,run_id=env['GITHUB_RUN_ID'],run_head=head,run_attempt=1,
                run_branch=spec['branch'],prepared_parent=prepared,runtime_workflow=spec['workflow'])


def resource_paths(root, python):
    root=Path(root)
    values={n.value for path in python for n in ast.walk(ast.parse((root/path).read_bytes()))
            if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    values={v for v in values if re.fullmatch(r'[A-Za-z0-9_./-]{1,180}',v)
            and not Path(v).is_absolute() and '..' not in Path(v).parts}
    resources=set()
    def add(relative):
        path=root/relative
        if path.is_file() and path.suffix.lower() in SUFFIXES: resources.add(str(relative))
        elif path.is_dir() and 'panel' in path.name:
            resources.update(str(p.relative_to(root)) for p in path.rglob('*')
                if p.is_file() and p.suffix.lower() in SUFFIXES)
    for value in values:
        add(Path(value));add(Path('automation')/value)
        if Path(value).name==value and Path(value).suffix.lower() in SUFFIXES:
            for path in (root/'automation').rglob(value): add(path.relative_to(root))
    for value in EXTRA:
        need((root/value).exists(),'complete B21 registered resource '+value)
        add(Path(value))
    # The reviewed B20 producer already closes paths carried indirectly through
    # B18/B19 registrations, including durable original proof metadata.
    legacy='automation/english_v6_b20_primary_producer.json'
    old=json.loads((root/legacy).read_bytes()); resources.add(legacy)
    for mapping in (old['resources'],old['source'].get('resources',{})):
        for path,expected in mapping.items():
            need((root/path).is_file() and sha((root/path).read_bytes())==expected,
                 'unchanged indirect scientific resource '+path)
            resources.add(path)
    def refs(node):
        if type(node) is dict:
            if type(node.get('path')) is str and re.fullmatch(r'[0-9a-f]{64}',str(node.get('sha256',''))):
                yield node['path'],node['sha256']
            for v in node.values(): yield from refs(v)
        elif type(node) is list:
            for v in node: yield from refs(v)
    pending=list(resources);visited=set()
    while pending:
        name=pending.pop()
        if name in visited or Path(name).suffix.lower()!='.json': continue
        visited.add(name)
        for ref,expected in refs(json.loads((root/name).read_bytes())):
            p=Path(ref)
            if p.is_absolute() or '..' in p.parts or p.suffix.lower() not in SUFFIXES: continue
            if not (root/p).is_file(): continue  # Remote/archive-member paths are not local producer files.
            need(sha((root/p).read_bytes())==expected,'registered indirect resource bytes '+ref)
            if ref not in resources: resources.add(ref);pending.append(ref)
    return resources


def capture(role,output,prepared,root=ROOT):
    root=Path(root).resolve();output=Path(output)
    head=git(root,'rev-parse','HEAD').decode().strip()
    parents=git(root,'rev-list','--parents','-n','1','HEAD').decode().split()[1:]
    changed=git(root,'diff-tree','--no-commit-id','--name-only','-r','HEAD').decode().splitlines()
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_bytes())
    identity=validate_event(role,prepared,head,parents,changed,os.environ,event)
    roots=(*ROLES[role]['roots'],'english_v6_b21_producer_snapshot','test_english_v6_b21_producer_snapshot')
    python=prior.static_closure(root/'automation',roots)
    resources=resource_paths(root,python)
    paths=python|resources|{ROLES[role]['workflow']}
    tracked=tree_entries(git(root,'ls-tree','-rz','--full-tree','HEAD'),paths)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh producer output; no overwrite')
    files={};payloads={}
    for name in sorted(paths):
        path=root/name
        need(name in tracked and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
             'regular tracked producer source '+name)
        raw=path.read_bytes()
        need(blob(raw)==tracked[name]['git_blob_sha1'], 'working source differs from actual Git HEAD '+name)
        files[name]=dict(**tracked[name],bytes=len(raw),sha256=sha(raw))
        payloads[name]=raw
    output.mkdir()
    for name,raw in payloads.items():
        dest=output/'repository'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    manifest=dict(schema='english-v6-b21-original-git-producer-snapshot-v1',**identity,
        git_tree=git(root,'rev-parse','HEAD^{tree}').decode().strip(),files=files,
        python_files=sorted(python),resource_files=sorted(resources),
        runtime_workflow_preserved_hidden=True,source_code_reads_only=True,network_calls=0,
        new_source_calls=0,new_model_forwards=0,optimizer_updates=0,promotion_allowed=False)
    raw=(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    (output/'PRODUCER.json').write_bytes(raw)
    print(json.dumps(dict(role=role,files=len(files),bytes=sum(v['bytes'] for v in files.values()),
        manifest_sha256=sha(raw),run_head=head,prepared_parent=prepared,new_model_forwards=0)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=tuple(ROLES));parser.add_argument('output',type=Path)
    parser.add_argument('--prepared-parent',required=True)
    args=parser.parse_args();capture(args.role,args.output,args.prepared_parent)
