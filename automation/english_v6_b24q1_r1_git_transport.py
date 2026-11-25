"""Transport two unchanged historical Git JSON bodies; no scientific authority.

The new read-only transport artifact is additional evidence, not one of the
fourteen original R1 artifacts. Historical snapshot bytes and current gh API
bytes are separately pinned and preserved. Only strict typed JSON value
identity connects them; current responses are never reserialized into old ones.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import stat
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BINDING = HERE / 'english_v6_b24q1_r1_git_transport.json'
BINDING_SHA = '40aa8743c97492a7c7493f0cc5fb1b78b769374ac6f44c83ceba83ba08dc0dd7'
TRANSPORT_DIR = 'R1_STOP_TRANSPORT'
SCHEMA = 'english-v6-b24q1-r1-historical-git-snapshot-transport-v1'
MANIFEST_SCHEMA = 'english-v6-b24q1-historical-r1-git-metadata-transport-v1'
REPO = 'jiying2007/kws-data'
BRANCH = 'review/en-v6-b24q1-r1-git-snapshots-20261125'
WORKFLOW = '.github/workflows/english-v6-b24q1-r1-git-snapshots.yml'
PREPARED_PARENT = '4e046f40bde54bb37ef381f03b79b310227bb07e'
ORIGINAL_RUN = 45534020313
ORIGINAL_HEAD = 'b65bfe4974432b3d922738d7931d1c2d6d6f367c'
ORIGINAL_TREE = 'c3c804008ae68d7fb1166becdd11469bcf4d5e1d'
ORIGINAL_LOCK_SHA = '56c10ef6a669650050fd1d821871489c030f13f224fc85d1c9dc537c69ef37f8'
ORIGINAL_LOCK = ROOT / 'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json'
SNAPSHOT_ROOT = 'research/english_v6_b24q1_original_git_transport_20261125/'
HISTORICAL = {
    'commit': {'bytes':5900,'sha256':'8d34e7c54aa1af030b67f66f460fb143d9473fa25de6e94306108d5bd836e107'},
    'tree': {'bytes':1540157,'sha256':'834c2233915936aec21d71c47ba89cc41ccb990f6b5ada6131dd0ad08a940b78'}}
CURRENT = {
    'commit': {'bytes':5300,'sha256':'c354913941be8605f6f3b52fc24ee6944bd89350264937a443d503124c978292'},
    'tree': {'bytes':1279866,'sha256':'0bc8212a636224c59b5fa2a92dd942e32b3ae0db4180d3fa0d0048ece6583e07'}}
API_ROLES = ('run','artifacts','commit','tree')
DIRECTORY_FILES = {'BINDING.json','RECEIPT.json','snapshots.zip'} | {
    'origin_'+name+'.json' for name in API_ROLES} | {'current_commit.json','current_tree.json'}
MAX_METADATA = 4_000_000
MAX_ZIP_BYTES = 4_000_000
MAX_EXPANDED = 2_000_000


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw):
    return dict(bytes=len(raw),sha256=sha(raw))


def canonical(value):
    return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def _pairs(values):
    result={}
    for key,value in values:
        need(key not in result,'no duplicate JSON object keys')
        result[key]=value
    return result


def _constant(value):
    raise ValueError('no nonfinite JSON constant: '+value)


def _float(value):
    number=float(value)
    need(math.isfinite(number),'no overflowed nonfinite JSON number')
    return number


def document(raw):
    need(type(raw) is bytes and 0<len(raw)<=MAX_METADATA,'bounded original UTF-8 JSON bytes')
    return json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=_constant,parse_float=_float)


def same_json(left,right):
    """Full value identity with bool/int/float distinctions and ordered lists."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(same_json(left[k],right[k]) for k in left)
    if type(left) is list:
        return len(left)==len(right) and all(same_json(a,b) for a,b in zip(left,right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left==right and (
            left!=0 or math.copysign(1,left)==math.copysign(1,right))
    return type(left) in (str,int,bool,type(None)) and left==right


def regular(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
         'regular original transport file: '+str(path))
    return path


def read_file(path,limit):
    path=regular(path)
    need(0<path.stat().st_size<=limit,'bounded complete transport file: '+str(path))
    return path.read_bytes()


def valid_hash(value,length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


def _pin(value,limit):
    need(type(value) is dict and set(value)=={'bytes','sha256'}
         and type(value['bytes']) is int and 0<value['bytes']<=limit
         and valid_hash(value['sha256']),'exact bounded byte pin')


def _original_lock():
    raw=read_file(ORIGINAL_LOCK,MAX_METADATA)
    need(sha(raw)==ORIGINAL_LOCK_SHA,'unchanged complete original R1 lock')
    value=document(raw)
    need(len(value['artifacts'])==14 and all(value['api'][r]==HISTORICAL[r] for r in HISTORICAL),
         'the two snapshots are exactly the original independently authenticated Git bodies')
    return value


def validate_binding(value):
    need(type(value) is dict and set(value)=={'schema','bound','transport','snapshots','manifest','live_git',
         'original_lock_sha256','original_run_id','original_head','original_tree','scientific_execution_authorized'},
         'complete narrowly scoped historical Git transport binding')
    need(value['schema']==SCHEMA and value['bound'] is True
         and value['original_lock_sha256']==ORIGINAL_LOCK_SHA and type(value['original_run_id']) is int
         and value['original_run_id']==ORIGINAL_RUN and value['original_head']==ORIGINAL_HEAD
         and value['original_tree']==ORIGINAL_TREE and value['scientific_execution_authorized'] is False,
         'only two existing historical Git snapshots; no scientific execution grant')
    need(type(value['snapshots']) is dict and set(value['snapshots'])=={'commit','tree'}
         and type(value['live_git']) is dict and value['live_git']==CURRENT,
         'both immutable original and separately pinned current Git responses')
    for role in HISTORICAL:
        member='source_'+role+'.snapshot.json'
        need(value['snapshots'][role]==dict(member=member,repository_path=SNAPSHOT_ROOT+member,**HISTORICAL[role]),
             'exact historical snapshot identity '+role)
        _pin(value['live_git'][role],MAX_METADATA)
    manifest=value['manifest']
    need(type(manifest) is dict and set(manifest)=={'member','bytes','sha256'}
         and manifest['member']=='SNAPSHOTS.json','one explicit additional-transport manifest')
    _pin({k:manifest[k] for k in ('bytes','sha256')},16_384)
    t=value['transport']
    need(type(t) is dict and set(t)=={'repository','run_id','head','prepared_parent','branch','workflow',
         'artifact','api','producer_files'} and t['repository']==REPO and type(t['run_id']) is int
         and t['run_id']>0 and t['run_id']!=ORIGINAL_RUN and valid_hash(t['head'],40)
         and t['head'] not in (ORIGINAL_HEAD,PREPARED_PARENT) and t['prepared_parent']==PREPARED_PARENT
         and t['branch']==BRANCH and t['workflow']==WORKFLOW,'one new public read-only snapshot transport run')
    artifact=t['artifact']
    need(type(artifact) is dict and set(artifact)=={'id','name','bytes','sha256'}
         and type(artifact['id']) is int and artifact['id']>0
         and artifact['name']=='english-v6-b24q1-historical-r1-git-metadata-'+str(t['run_id']),
         'additional named transport artifact')
    _pin({k:artifact[k] for k in ('bytes','sha256')},MAX_ZIP_BYTES)
    old=_original_lock()
    need(artifact['id'] not in {a['id'] for a in old['artifacts'].values()},
         'the new transport artifact is not one of the original fourteen artifacts')
    need(type(t['api']) is dict and set(t['api'])==set(API_ROLES),'all four original transport API bodies')
    for expected in t['api'].values():
        _pin(expected,MAX_METADATA)
    paths={value['snapshots'][r]['repository_path'] for r in HISTORICAL}|{WORKFLOW}
    need(type(t['producer_files']) is dict and set(t['producer_files'])==paths,
         'exact two historical snapshot paths and read-only transport workflow')
    for name,facts in t['producer_files'].items():
        need(type(facts) is dict and set(facts)=={'bytes','sha256','git_blob_sha1'}
             and valid_hash(facts['git_blob_sha1'],40),'complete original Git producer entry '+name)
        _pin({k:facts[k] for k in ('bytes','sha256')},MAX_METADATA)
    return value


def read_binding(path=None):
    raw=read_file(path or BINDING,MAX_METADATA)
    need(sha(raw)==BINDING_SHA,'exact frozen actual transport artifact and origin binding')
    return validate_binding(document(raw))


def _require_pin(raw,expected,role):
    need(type(raw) is bytes and pin(raw)=={k:expected[k] for k in ('bytes','sha256')},
         'exact original transport bytes: '+role)
    return raw


def _origin(binding,raws):
    t=binding['transport']
    need(type(raws) is dict and set(raws)==set(API_ROLES),'complete raw transport origin')
    api={k:document(_require_pin(raws[k],t['api'][k],'origin '+k)) for k in API_ROLES}
    run,artifacts,commit,tree=(api[k] for k in API_ROLES)
    need(type(run['id']) is int and run['id']==t['run_id'] and run['head_sha']==t['head']
         and type(run['run_attempt']) is int and run['run_attempt']==1
         and run['event']=='push' and run['head_branch']==BRANCH and run['path']==WORKFLOW
         and run['status']=='completed' and run['conclusion']=='success'
         and all(run[k]['full_name']==REPO and run[k]['private'] is False for k in ('repository','head_repository')),
         'actual completed first public read-only snapshot transport')
    a=t['artifact']
    need(type(artifacts['total_count']) is int and artifacts['total_count']==len(artifacts['artifacts'])==1,
         'all and only one transport artifact')
    actual=artifacts['artifacts'][0]
    need(actual['id']==a['id'] and actual['name']==a['name'] and actual['size_in_bytes']==a['bytes']
         and actual['digest']=='sha256:'+a['sha256'] and actual['expired'] is False
         and actual['workflow_run']['id']==t['run_id'] and actual['workflow_run']['head_sha']==t['head'],
         'exact public snapshot artifact and producer run')
    need(commit['sha']==t['head'] and len(commit['parents'])==1
         and commit['parents'][0]['sha']==PREPARED_PARENT
         and len(commit['files'])==3 and {f['filename'] for f in commit['files']}==set(t['producer_files'])
         and all(f['status']=='added' for f in commit['files']),
         'transport commit adds only two snapshots and its workflow to the reviewed parent')
    need(tree['sha']==commit['commit']['tree']['sha'] and tree['truncated'] is False
         and type(tree['tree']) is list and len({e['path'] for e in tree['tree']})==len(tree['tree']),
         'complete original transport Git tree')
    entries={e['path']:e for e in tree['tree']}
    changed={f['filename']:f for f in commit['files']}
    for name,facts in t['producer_files'].items():
        e=entries[name]
        need(e['type']=='blob' and e['mode']=='100644' and e['size']==facts['bytes']
             and e['sha']==changed[name]['sha']==facts['git_blob_sha1'],
             'exact original transport producer Git identity: '+name)
    return api


def _expected_manifest(binding):
    t=binding['transport']
    return dict(schema=MANIFEST_SCHEMA,source_repository=REPO,original_source_run=ORIGINAL_RUN,
        original_source_head=ORIGINAL_HEAD,original_source_tree=ORIGINAL_TREE,
        transport_run_id=t['run_id'],transport_run_head=t['head'],transport_branch=BRANCH,transport_run_attempt=1,
        original_lock_sha256=ORIGINAL_LOCK_SHA,
        files={binding['snapshots'][r]['member']:HISTORICAL[r] for r in HISTORICAL},
        transport_only=True,part_of_original_fourteen_artifacts=False,historical_snapshot_bytes_unchanged=True,
        current_API_response_raw_bytes_claimed=False,reserialization_performed=False,
        new_source_calls=0,new_observer_calls=0,new_KWS_forwards=0,new_optimizer_updates=0,
        scientific_execution_authorized=False)


def _snapshots(binding,raw):
    _require_pin(raw,binding['transport']['artifact'],'complete additional snapshot ZIP')
    need(len(raw)<=MAX_ZIP_BYTES,'bounded whole snapshot archive')
    names={s['member'] for s in binding['snapshots'].values()}|{'SNAPSHOTS.json'}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries=archive.infolist()
        need(len(entries)==3 and {e.filename for e in entries}==names
             and sum(e.file_size for e in entries)<=MAX_EXPANDED and len(archive.comment)<=512,
             'exact three-member snapshot archive closure')
        for e in entries:
            need(not e.is_dir() and not stat.S_ISLNK(e.external_attr>>16)
                 and stat.S_IFMT(e.external_attr>>16) in (0,stat.S_IFREG)
                 and not(e.flag_bits&1) and e.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
                 and len(e.extra)<=128 and not e.comment,'safe bounded regular snapshot member')
            local=raw[e.header_offset:e.header_offset+30]
            need(len(local)==30 and local[:4]==b'PK\x03\x04'
                 and int.from_bytes(local[26:28],'little')<=128
                 and int.from_bytes(local[28:30],'little')<=128,'bounded original local ZIP header')
        need(archive.testzip() is None,'every snapshot ZIP member CRC')
        members={name:archive.read(name) for name in names}
    manifest=document(_require_pin(members['SNAPSHOTS.json'],binding['manifest'],'transport manifest'))
    need(same_json(manifest,_expected_manifest(binding)),'honest complete historical-snapshot transport manifest')
    historical={}
    for role,expected in binding['snapshots'].items():
        original=_require_pin(members[expected['member']],expected,'historical '+role)
        facts=binding['transport']['producer_files'][expected['repository_path']]
        need(pin(original)=={k:facts[k] for k in ('bytes','sha256')}
             and hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()==facts['git_blob_sha1'],
             'original snapshot bytes exactly match the transported Git blob')
        historical[role]=original
    return historical


def _compare(binding,historical,current):
    need(type(current) is dict and set(current)=={'commit','tree'},'both actual current gh response bodies')
    for role in HISTORICAL:
        _require_pin(current[role],binding['live_git'][role],'actual current Git '+role)
        need(same_json(document(historical[role]),document(current[role])),
             'complete type-aware historical/current Git value identity: '+role)
    commit,tree=(document(historical[r]) for r in ('commit','tree'))
    need(commit['sha']==ORIGINAL_HEAD and commit['commit']['tree']['sha']==tree['sha']==ORIGINAL_TREE
         and tree['truncated'] is False,'same immutable original R1 Git commit and complete tree')


def _receipt(binding,raw_binding):
    t=binding['transport']
    return dict(schema=SCHEMA+'-receipt',binding_sha256=sha(raw_binding),transport_only=True,
        original_source_run=ORIGINAL_RUN,original_source_head=ORIGINAL_HEAD,original_source_tree=ORIGINAL_TREE,
        transport_run_id=t['run_id'],transport_run_head=t['head'],transport_artifact=dict(t['artifact']),
        historical_raw_pins=HISTORICAL,current_raw_pins=CURRENT,origin_raw_pins=t['api'],
        complete_archive_digest_and_all_member_CRCs_verified=True,complete_origin_API_and_three_Git_entries_verified=True,
        two_historical_snapshot_raw_Git_blobs_recomputed=True,transport_workflow_raw_body_reopened=False,
        strict_type_aware_JSON_values_equal=True,byte_equality_between_historical_and_current_claimed=False,
        historical_snapshot_bytes_unchanged=True,current_raw_bytes_retained=True,reserialization_performed=False,
        part_of_original_fourteen_artifacts=False,old_reader_lock_and_report_unchanged=True,
        new_source_calls=0,new_observer_calls=0,new_KWS_forwards=0,new_optimizer_updates=0,
        scientific_execution_authorized=False)


def verify_transport(parent_dir):
    """Pure saved-byte verification; return the original historical raw bodies."""
    binding=read_binding();raw_binding=read_file(BINDING,MAX_METADATA)
    directory=Path(parent_dir)/TRANSPORT_DIR
    need(directory.is_dir() and not directory.is_symlink()
         and {p.name for p in directory.iterdir()}==DIRECTORY_FILES,'complete exact original transport directory')
    raws={p.name:read_file(p,MAX_ZIP_BYTES if p.name=='snapshots.zip' else MAX_METADATA) for p in directory.iterdir()}
    need(raws['BINDING.json']==raw_binding,'same exact captured transport binding')
    _origin(binding,{r:raws['origin_'+r+'.json'] for r in API_ROLES})
    historical=_snapshots(binding,raws['snapshots.zip'])
    _compare(binding,historical,{r:raws['current_'+r+'.json'] for r in HISTORICAL})
    need(raws['RECEIPT.json']==canonical(_receipt(binding,raw_binding)),
         'exact honest transport receipt from retained original evidence')
    return historical


def gh(endpoint):
    return subprocess.run(['gh','api','repos/'+REPO+'/'+endpoint,'--allow-escape-sequences'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout


def assemble(parent_dir,live_git_raw,origin_raw,snapshot_zip_raw):
    """Validate and retain supplied original bytes, without any network call.

    This supports independent readback of already collected API/artifact
    evidence. It does not claim that this process downloaded those bodies.
    """
    binding=read_binding();raw_binding=read_file(BINDING,MAX_METADATA)
    need(type(live_git_raw) is dict and set(live_git_raw)==set(HISTORICAL),'complete separately captured current Git bodies')
    for role in HISTORICAL:
        _require_pin(live_git_raw[role],binding['live_git'][role],'actual current Git '+role)
    parent=Path(parent_dir);directory=parent/TRANSPORT_DIR
    need(parent.is_dir() and not parent.is_symlink() and not directory.exists(),'fresh explicit snapshot transport directory')
    _origin(binding,origin_raw)
    historical=_snapshots(binding,snapshot_zip_raw);_compare(binding,historical,live_git_raw)
    files={'BINDING.json':raw_binding,'snapshots.zip':snapshot_zip_raw,
        'RECEIPT.json':canonical(_receipt(binding,raw_binding)),
        **{'origin_'+r+'.json':v for r,v in origin_raw.items()},
        **{'current_'+r+'.json':v for r,v in live_git_raw.items()}}
    need(set(files)==DIRECTORY_FILES,'complete transport evidence before saving')
    directory.mkdir()
    for name,raw in files.items():
        with (directory/name).open('xb') as stream:
            stream.write(raw)
    return verify_transport(parent)


def fetch(parent_dir,live_git_raw):
    """Collect one separately pinned existing transport artifact and its origin."""
    binding=read_binding();t=binding['transport']
    need(type(live_git_raw) is dict and set(live_git_raw)==set(HISTORICAL),'both current raw Git responses before download')
    for role in HISTORICAL:
        _require_pin(live_git_raw[role],binding['live_git'][role],'actual current Git '+role)
    raws={r:gh(endpoint) for r,endpoint in (
        ('run','actions/runs/'+str(t['run_id'])),
        ('artifacts','actions/runs/'+str(t['run_id'])+'/artifacts?per_page=100'),
        ('commit','commits/'+t['head']))}
    commit=document(_require_pin(raws['commit'],t['api']['commit'],'transport commit'))
    raws['tree']=gh('git/trees/'+commit['commit']['tree']['sha']+'?recursive=1')
    _origin(binding,raws)
    raw_zip=gh('actions/artifacts/'+str(t['artifact']['id'])+'/zip')
    return assemble(parent_dir,live_git_raw,raws,raw_zip)


def retain_transport(parent_dir,output):
    historical=verify_transport(parent_dir)
    output=Path(output)
    need(output.is_dir() and not output.is_symlink() and not(output/TRANSPORT_DIR).exists(),
         'fresh retained additional transport evidence directory')
    shutil.copytree(Path(parent_dir)/TRANSPORT_DIR,output/TRANSPORT_DIR)
    verified=verify_transport(output)
    need(verified==historical,'all original historical snapshot bytes survive retention')
    return verified
