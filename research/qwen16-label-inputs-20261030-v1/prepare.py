"""Derive blind labeling inputs from one pinned, already archived public batch.

This module never imports a model, makes a network request, supplies a transcript,
changes a data role, or grants training admission. Only blind/ may be mounted in
an eventual recognizer. control/ is a separate source/role mapping for consumers.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import zipfile

SCOPE = 'research/qwen16-label-inputs-20261030-v1/'
BRANCH = 'research/qwen16-label-inputs-20261030-v1'
WORKFLOW = '.github/workflows/qwen16-label-inputs-20261030-v1.yml'
SOURCE_PATHS = {SCOPE+n for n in ('prepare.py','test_prepare.py','README.md')} | {WORKFLOW}
ARCHIVE_COMMIT = '1ec9c6b20387620026bdbbcac6c705f3912f004e'
ARCHIVE_PATH = 'research/qwen16-archive-20261030-v1/evidence/run-41008894901/original.zip'
ARCHIVE_SHA256 = '78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1'
ARCHIVE_SIZE = 4602393
SOURCE_COMMIT = '1398f5e3c6a608ab865cd20fcad9d1b9e622c30a'
RUN_ID = 41008894901
MANIFEST = 'docs/BRANCH_RETENTION_20261007.json'
MAX_MEMBER = 2*1024**2
MAX_TOTAL = 32*1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid)


def read_regular(path, cap):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= cap, 'bounded unaliased file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stamp = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    try:
        require(stamp(os.fstat(fd)) == stamp(before), 'file changed at open')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(cap+1)
        require(len(raw) == before.st_size and stamp(os.fstat(fd)) == stamp(before), 'file changed during read')
    finally:
        os.close(fd)
    require(stamp(path.lstat()) == stamp(before), 'file replaced after read')
    return raw


def archive_members(path):
    raw = read_regular(path, ARCHIVE_SIZE)
    require(len(raw) == ARCHIVE_SIZE and sha(raw) == ARCHIVE_SHA256, 'exact archived ZIP required')
    files, total = {}, 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(len(archive.infolist()) == 59, 'original 59 members required')
        for item in archive.infolist():
            name = item.filename
            require(not PurePosixPath(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/')) and '\\' not in name and name not in files, 'ZIP path or duplicate')
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16) and not item.flag_bits & 1 and 0 <= item.file_size <= MAX_MEMBER, 'ZIP member type/size')
            total += item.file_size
            require(total <= MAX_TOTAL, 'ZIP expanded bound')
            with archive.open(item) as stream:
                body = stream.read(MAX_MEMBER+1)
            require(len(body) == item.file_size, 'ZIP member length')
            files[name] = body
    return files


def pcm_identity(raw):
    require(44 < len(raw) <= 44+2*192000 and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE', 'PCM WAV identity')
    require(struct.unpack_from('<I',raw,4)[0]+8 == len(raw), 'PCM RIFF extent')
    require(raw[12:16] == b'fmt ' and struct.unpack_from('<I',raw,16)[0] == 16 and raw[36:40] == b'data', 'canonical PCM chunks')
    require(struct.unpack_from('<HHIIHH',raw,20) == (1,1,16000,32000,2,16), 'PCM geometry')
    size = struct.unpack_from('<I',raw,40)[0]
    require(size == len(raw)-44 and size % 2 == 0, 'PCM complete samples')
    return {'file_sha256':sha(raw),'pcm_sha256':sha(raw[44:]),'audio_frames':size//2}


def derive(files):
    """Pure projection; callers must first authenticate the exact archive."""
    freeze = parse(files['artifact-freeze.json'])
    require(freeze['source_commit'] == SOURCE_COMMIT and freeze['run_id'] == str(RUN_ID) and type(freeze['run_attempt']) is int and freeze['run_attempt'] == 1, 'original execution identity')
    require(set(files) == set(freeze['files']) | {'artifact-freeze.json'}, 'original member set')
    require(all(freeze[k] is False for k in ('labels_verified','qualification_allowed','training_admitted')), 'archive cannot supply labels')
    for name, row in freeze['files'].items():
        require(type(row['bytes']) is int and row == {'bytes':len(files[name]),'sha256':sha(files[name])}, 'original member digest')
    plan = parse(files['plan.json'])
    receipt = parse(files['output/generation-receipt.json'])
    require(type(plan) is list and len(plan) == 16 and len(receipt['outcomes']) == 16 and receipt['status'] == 'generated_16_pending_review' and receipt['training_admitted'] is False, 'original denominator/status')
    selected, excluded, seen = [], [], set()
    for i, (intent, result) in enumerate(zip(plan,receipt['outcomes'])):
        cell = f'qwen16-20261030-{i:03d}'
        voice = ('Vivian','Serena','Uncle_Fu','Dylan')[i//4]
        role = 'development_a' if voice == 'Serena' else 'train'
        require(intent['cell_id'] == cell and intent['speaker'] == voice and intent['proposed_split'] == role and intent['source_family_id'] == 'qwen3-customvoice:preset:'+voice.lower(), 'fixed source role/family')
        require(canonical({k:result.get(k) for k in intent}) == canonical(intent), 'plan/outcome identity')
        require(result['actual_text'] is None and result['actual_tokens'] is None and result['label_status'] == 'PENDING_REVIEW' and result['synthetic'] is True and result['source_type'] == 'tts', 'pending actual labels and synthetic origin')
        require(all(result[k] is False for k in ('ctc_training_allowed','evaluation_gold_allowed','qualification_allowed')), 'no source admission')
        require(type(result['attempts']) is int and result['attempts'] == 1 and result['status'] == 'generated_pending_review' and files['output/'+cell+'.claim'] == (cell+'\n').encode(), 'consumed source attempt')
        body = files['output/'+cell+'.wav']
        measured = pcm_identity(body)
        require(measured['file_sha256'] == result['wav_sha256'] and measured['pcm_sha256'] == result['pcm_sha256'] and type(result['frames']) is int and measured['audio_frames'] == result['frames'], 'measured original audio identity')
        require(measured['pcm_sha256'] not in seen, 'duplicate PCM')
        seen.add(measured['pcm_sha256'])
        mapping = {'source_id':cell,'speaker':voice,'proposed_split':role,'source_family_id':intent['source_family_id'], **measured,
            'synthetic':True,'source_type':'tts','actual_text':None,'actual_tokens':None,'ctc_training_allowed':False,
            'evaluation_gold_allowed':False,'qualification_allowed':False,
            'source_plan_record_sha256':sha(canonical(intent)),'source_outcome_record_sha256':sha(canonical(result))}
        (selected if role == 'train' else excluded).append((mapping,body))
    require(len(selected) == 12 and len(excluded) == 4, 'fixed train/development counts')
    outputs, clips, mapping = {}, [], []
    for i, (row, body) in enumerate(sorted(selected,key=lambda x:x[0]['file_sha256']),1):
        opaque = f'clip-{i:06d}'
        name = 'audio/'+row['file_sha256']+'.wav'
        clips.append({'audio_id':opaque,'audio_path':name,'wav_sha256':row['file_sha256']})
        mapping.append({'audio_id':opaque,**row})
        outputs['blind/'+name] = body
    outputs['blind/job.json'] = canonical({'schema':'blind-source-screen32-job-v1','clips':clips})
    outputs['control/source-map.json'] = canonical({'schema':'qwen16-label-source-map-v1','archive_commit':ARCHIVE_COMMIT,
        'archive_path':ARCHIVE_PATH,'archive_sha256':ARCHIVE_SHA256,'generation_source_commit':SOURCE_COMMIT,'generation_run_id':RUN_ID,
        'purpose':'pending-label-verification-only','rights_policy_review':'PENDING','rows':mapping,
        'excluded_development_a':[row for row,_ in excluded]})
    outputs['control/capture-request.json'] = canonical({'schema':'qwen16-prospective-native-capture-v1','status':'NOT_RUN',
        'input_job_sha256':sha(outputs['blind/job.json']),'model_families':['qwen3-asr','sensevoice','whisper'],
        'clips_per_family':12,'maximum_recognizer_calls':36,'maximum_retries':0,'new_tts_calls':0,'training_runs':0,
        'requires_reviewed_runtime_source_asset_and_native_config_freeze':True,
        'worker_mount':'blind/ only; never control/ or original archive','target_conditioning_allowed':False,
        'executed':False,'ctc_training_allowed':False,'evaluation_gold_allowed':False,'qualification_allowed':False})
    outputs['handoff.json'] = canonical({'schema':'qwen16-blind-label-handoff-v1','status':'INPUTS_READY_LABELS_PENDING',
        'archive_sha256':ARCHIVE_SHA256,'selected_train_candidates':12,'excluded_development_a':4,
        'selected_seconds':sum(r['audio_frames'] for r in mapping)/16000,'actual_asr_calls':0,'actual_training_runs':0,
        'ctc_training_allowed':False,'evaluation_gold_allowed':False,'qualification_allowed':False,
        'files':{name:{'bytes':len(raw),'sha256':sha(raw)} for name,raw in sorted(outputs.items())}})
    return outputs


def verify_outputs(archive, output):
    expected = derive(archive_members(archive))
    root = Path(output)
    require(root.is_dir() and not root.is_symlink(), 'handoff directory required')
    actual = set()
    for base, dirs, files in os.walk(root,followlinks=False):
        require(not any((Path(base)/n).is_symlink() for n in dirs+files), 'handoff aliases forbidden')
        for name in files:
            path = Path(base)/name
            relative = path.relative_to(root).as_posix()
            require(relative in expected and read_regular(path,MAX_MEMBER) == expected[relative], 'handoff drift/extra file')
            actual.add(relative)
    require(actual == set(expected), 'missing handoff file')
    return parse(expected['handoff.json'])


def materialize(archive, output):
    expected = derive(archive_members(archive))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)  # Do not overwrite any prior handoff.
    for name, raw in sorted(expected.items(), key=lambda pair:(pair[0]=='handoff.json',pair[0])):
        path = output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    return verify_outputs(archive,output)


def register():
    """Append exactly four new source paths; never revise retained evidence."""
    def git(*args):
        return subprocess.check_output(['git',*args],timeout=30)
    require(git('branch','--show-current').decode().strip() == BRANCH and not git('status','--porcelain'), 'clean fixed source branch')
    commit = git('rev-parse','HEAD').decode().strip()
    tree = {}
    for item in git('ls-tree','-r','-l','-z','HEAD').split(b'\0'):
        if item:
            meta,name = item.split(b'\t',1)
            mode,kind,blob,size = meta.decode().split()
            require(kind == 'blob','no submodules')
            tree[name.decode()] = (mode,blob,int(size))
    raw = Path(MANIFEST).read_bytes()
    old = parse(raw)
    require(old['schema'] == 'kws-public-branch-retention-v1','retention schema')
    rows = {r['path']:r for r in old['retained_files']}
    integration = set(old['integration_files'])
    require(len(rows) == len(old['retained_files']) == old['expected_retained_files'] and len(integration) == len(old['integration_files']) and not rows.keys() & integration and len(rows)+len(integration) == old['expected_tracked_files'],'existing inventory')
    for name,row in rows.items():
        require(tree.get(name) == (row['mode'],row['git_blob_sha1'],row['bytes']),'retained source changed')
    require(set(rows)|integration <= set(tree),'existing file removed')
    fresh = set(tree)-set(rows)-integration
    if not fresh:
        print('source paths already registered; no mutation')
        return
    require(fresh == SOURCE_PATHS,'only exact four new source paths')
    result = copy.deepcopy(old)
    for name in sorted(fresh):
        mode,blob,size = tree[name]
        require(mode == '100644','regular new source')
        result['retained_files'].append({'path':name,'mode':mode,'git_blob_sha1':blob,'bytes':size,'source_commit':commit})
    result['expected_retained_files'] = len(result['retained_files'])
    result['expected_tracked_files'] = len(tree)
    result.setdefault('source_registration_history',[]).append({'campaign':'qwen16-blind-label-inputs-20261030',
        'source_commit':commit,'paths':sorted(fresh),'previous_manifest_sha256':sha(raw),
        'scope':'blind input preparation only; no ASR, labels, training or new generation'})
    Path(MANIFEST).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'new_paths':len(fresh),'source_commit':commit,'retained_before':len(rows),'retained_after':len(result['retained_files'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path(ARCHIVE_PATH))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path)
    group.add_argument('--verify',type=Path)
    group.add_argument('--register',action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
    else:
        result = materialize(args.archive,args.output) if args.output else verify_outputs(args.archive,args.verify)
        print(json.dumps({k:v for k,v in result.items() if k!='files'},sort_keys=True))
