"""Read-only verification and exact archival of ONE already completed TTS run.

No inference, retries, labels, promotion, or changes to the production catalog.
The pinned ZIP is the authority; extracted payloads and observer reports are separate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import zipfile

REPO = 'jiying2007/kws-data'
RUN = 41008894901
ARTIFACT = 12956997278
SOURCE = '1398f5e3c6a608ab865cd20fcad9d1b9e622c30a'
ZIP_SHA = '78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1'
ZIP_SIZE = 4602393
FREEZE_SHA = 'e5122781aa979f7bf10f151fab17f58a2569df6f610fb935a43fa07e8fa81983'
CAMPAIGN = 'qwen16-intake-20261030-v1'
BRANCH = 'research/archive-qwen16-41008894901'
SCOPE = 'research/qwen16-archive-20261030-v1/'
DEST = SCOPE + 'evidence/run-41008894901'
WORKFLOW = '.github/workflows/qwen16-archive-20261030-v1.yml'
MANIFEST = 'docs/BRANCH_RETENTION_20261007.json'
MAX_MEMBER = 2 * 1024**2
MAX_TOTAL = 32 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def parse(raw):
    def pairs(rows):
        result = {}
        for k, v in rows:
            require(k not in result, 'duplicate JSON field')
            result[k] = v
        return result
    def number(text):
        result = float(text)
        require(math.isfinite(result), 'nonfinite JSON number')
        return result
    def reject(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=reject)


def wav_payload(raw, rate, tag, bits):
    require(44 <= len(raw) <= MAX_MEMBER and raw[:4] == b'RIFF' and raw[8:12] == b'WAVE', 'WAV identity')
    require(struct.unpack_from('<I', raw, 4)[0] + 8 == len(raw), 'RIFF extent')
    pos, fmt, data = 12, None, None
    while pos < len(raw):
        require(pos + 8 <= len(raw), 'truncated chunk header')
        kind, count = struct.unpack_from('<4sI', raw, pos)
        start, end = pos + 8, pos + 8 + count
        require(end + (count & 1) <= len(raw), 'truncated chunk')
        if kind == b'fmt ':
            require(fmt is None and count in (16, 18), 'format chunk')
            fmt = struct.unpack_from('<HHIIHH', raw, start)
            require(count == 16 or raw[start+16:end] == b'\0\0', 'format extension')
        elif kind == b'data':
            require(data is None, 'duplicate data')
            data = raw[start:end]
        pos = end + (count & 1)
    width = bits // 8
    require(fmt == (tag, 1, rate, rate * width, width, bits), 'audio geometry')
    require(data is not None and 0 < len(data) <= 12 * rate * width and len(data) % width == 0, 'whole samples')
    return data


def inspect_members(files):
    names = {'artifact-freeze.json', 'container-setup.json', 'container-tts.json', 'host-receipt.json',
             'image-transfer.json', 'ledger.json', 'plan.json', 'output/generation-receipt.json',
             'output/kernel-scope.json', 'output/progress.json', 'output/runtime-observations.json'}
    ids = [f'qwen16-20261030-{i:03d}' for i in range(16)]
    names |= {'output/' + k + s for k in ids for s in ('.claim', '.wav', '.raw-float.wav')}
    require(set(files) == names and len(files) == 59, 'exact artifact membership')
    freeze = parse(files['artifact-freeze.json'])
    require(set(freeze['files']) == names - {'artifact-freeze.json'}, 'frozen membership')
    for name, spec in freeze['files'].items():
        require(type(spec['bytes']) is int and spec['bytes'] == len(files[name]) and spec['sha256'] == sha(files[name]), 'member identity: ' + name)
    for flag in ('labels_verified', 'qualification_allowed', 'training_admitted'):
        require(freeze.get(flag) is False, 'false archive admission flags')
    require(freeze['campaign'] == CAMPAIGN and freeze['source_commit'] == SOURCE and
            freeze['source_freeze_sha256'] == FREEZE_SHA and freeze['run_id'] == str(RUN) and
            type(freeze['run_attempt']) is int and freeze['run_attempt'] == 1, 'execution identity')
    host = parse(files['host-receipt.json'])
    require(host['status'] == 'complete_pending_review' and host['source_commit'] == SOURCE and
            host['run_id'] == str(RUN) and host['source_freeze_sha256'] == FREEZE_SHA and
            host['training_admitted'] is False, 'host receipt identity')
    for key, expected in (('consumed_cells',16), ('generated_pending_review',16), ('not_run_cells',0),
                          ('asr_calls',0), ('training_runs',0), ('run_attempt',1)):
        require(type(host[key]) is int and host[key] == expected, 'host count: ' + key)
    for stage in ('setup', 'tts'):
        d = parse(files['container-' + stage + '.json'])
        require(d['status'] == 'complete' and d['primary_failure'] is None and d['cleanup_failure'] is None and
                canonical(d['container_state']) == canonical({'Running':False,'ExitCode':0,'OOMKilled':False}) and
                all(x['status'] == 'complete' for x in d['operations'].values()), 'container execution failure')
    receipt = parse(files['output/generation-receipt.json'])
    require(receipt['status'] == 'generated_16_pending_review' and receipt['training_admitted'] is False and
            receipt['model_id'] == 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice' and
            receipt['model_revision'] == '85e237c12c027371202489a0ec509ded67b5e4b5', 'generation model/status')
    plan = parse(files['plan.json'])
    rows = receipt['outcomes']
    require(type(plan) is list and type(rows) is list and len(plan) == len(rows) == 16, 'denominator')
    ledger = parse(files['ledger.json'])
    require(ledger['training_admitted'] is False and len(ledger['cells']) == 16, 'ledger shape')
    measured, seen_pcm, total = [], set(), 0
    voices = ('Vivian','Serena','Uncle_Fu','Dylan')
    for i, (p, row, cell) in enumerate(zip(plan, rows, ledger['cells'])):
        ident, voice, repeat = ids[i], voices[i//4], i % 2
        require(p['cell_id'] == row['cell_id'] == cell['cell_id'] == ident and p['speaker'] == voice and
                p['seed'] == 20261030 + repeat and p['repeat'] == repeat and
                p['intended_keyword_id'] == (i//2)%2+1 and
                p['intended_text'] == ('你好小窝','小窝小窝')[(i//2)%2] and
                p['source_family_id'] == 'qwen3-customvoice:preset:' + voice.lower() and
                p['proposed_split'] == ('development_a' if voice == 'Serena' else 'train'), 'fixed cell/split')
        require(canonical({k:row.get(k) for k in p}) == canonical(p), 'row/plan binding')
        require(row['actual_text'] is None and row['actual_tokens'] is None and row['label_status'] == 'PENDING_REVIEW', 'no actual labels')
        require(row['synthetic'] is True and row['source_type'] == 'tts' and
                all(row[k] is False for k in ('ctc_training_allowed','evaluation_gold_allowed','qualification_allowed')), 'no relabel/admission')
        require(type(row['attempts']) is int and row['attempts'] == 1 and row['status'] == 'generated_pending_review' and
                canonical(cell) == canonical({'cell_id':ident,'attempts':1,'status':'GENERATED_PENDING_REVIEW','ctc_training_allowed':False}), 'attempt ledger')
        require(files['output/' + ident + '.claim'] == (ident + '\n').encode(), 'durable claim')
        raw, pcm_wav = files['output/' + ident + '.raw-float.wav'], files['output/' + ident + '.wav']
        require(sha(raw) == row['raw_wav_sha256'] and sha(pcm_wav) == row['wav_sha256'], 'WAV hash')
        native = wav_payload(raw, 24000, 3, 32)
        pcm = wav_payload(pcm_wav, 16000, 1, 16)
        require(sha(native) == row['native_float_values_sha256'] and sha(pcm) == row['pcm_sha256'], 'sample identity')
        require(type(row['frames']) is int and len(pcm)//2 == row['frames'] and
                row['frames'] == math.ceil(len(native)//4*2/3), 'resample geometry')
        require(row['pcm_sha256'] not in seen_pcm, 'duplicate PCM'); seen_pcm.add(row['pcm_sha256'])
        values = [v[0] for v in struct.iter_unpack('<f', native)]
        require(all(math.isfinite(v) for v in values), 'nonfinite native values')
        peak = max(map(abs, values))
        energy = 0.0
        for value in values:
            energy += value * value
        rms = math.sqrt(energy/len(values))
        metrics = row['native_metrics']
        require(peak < 1 and rms >= 1e-4 and metrics['technically_valid'] is True and metrics['flags'] == [] and
                metrics['peak'] == peak and metrics['rms'] == rms and metrics['frames'] == len(values), 'native metrics')
        eos = row['native_termination']
        require(eos['ended_with_eos'] is True and type(eos['eos_token_id']) is int and
                eos['last_token_ids'] == [eos['eos_token_id']], 'recorded EOS')
        total += row['frames']
        measured.append({'cell_id':ident, 'proposed_split':p['proposed_split'], 'frames':row['frames'],
                         'duration_seconds':row['frames']/16000, 'wav_sha256':sha(pcm_wav),
                         'pcm_sha256':sha(pcm), 'native_peak':peak, 'native_rms':rms,
                         'actual_text':None, 'actual_tokens':None, 'ctc_training_allowed':False})
    return {'schema':'qwen16-independent-archive-verification-v1','run_id':RUN,'artifact_id':ARTIFACT,
            'artifact_sha256':ZIP_SHA,'source_commit':SOURCE,'verified_members':len(files),
            'generated_pending_review':16,'train_candidates':12,'development_a_candidates':4,
            'measured_seconds':total/16000,'unique_pcm':len(seen_pcm),'audio_checks':'PASS',
            'labels_verified':False,'ctc_training_allowed':False,'qualification_allowed':False,
            'human_review':'NOT_RUN','asr_review':'NOT_RUN','new_model_runs':0,'rows':measured}


def read_archive(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == ZIP_SIZE, 'pinned ZIP size/type')
    raw = path.read_bytes(); require(sha(raw) == ZIP_SHA, 'pinned ZIP digest')
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(len(archive.infolist()) == 59, 'ZIP member count')
        total = 0
        for item in archive.infolist():
            name = item.filename; p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and str(p) == name and '\\' not in name and name not in files, 'ZIP path/duplicate')
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16) and not item.flag_bits & 1 and
                    0 <= item.file_size <= MAX_MEMBER, 'ZIP file type/size')
            total += item.file_size; require(total <= MAX_TOTAL, 'ZIP total size')
            with archive.open(item) as stream:
                payload = stream.read(MAX_MEMBER+1)
            require(len(payload) == item.file_size, 'ZIP actual member size'); files[name] = payload
    report = inspect_members(files)
    return raw, files, report


def materialize(archive, output):
    require(not output.exists(), 'fresh archive destination required')
    raw, files, report = read_archive(archive)  # Validate everything before writing.
    output.mkdir(parents=True)
    (output/'original.zip').write_bytes(raw)
    for name, payload in files.items():
        target = output/'artifact'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(payload)
    (output/'verification.json').write_bytes(canonical(report)+b'\n')
    ordered = sorted(report['rows'],key=lambda row:row['wav_sha256'])
    mapping = [{'review_id':f'R{i+1:02d}', 'cell_id':row['cell_id'], 'wav_sha256':row['wav_sha256'],
                'actual_text':None, 'verdict':'PENDING', 'review_origin':None, 'ctc_training_allowed':False}
               for i,row in enumerate(ordered)]
    (output/'review-map.json').write_bytes(canonical(mapping)+b'\n')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>16条音频盲审</title><h1>16条待审合成音频</h1><p>先记录实际听到的内容、完整性和自然程度。此页不展示请求文本；未做任何标签准入。</p>'
    for item in mapping:
        page += '<section><h2>'+item['review_id']+'</h2><audio controls preload="none" src="artifact/output/'+html.escape(item['cell_id'])+'.wav"></audio></section>'
    (output/'review.html').write_text(page+'</html>\n',encoding='utf-8')
    return report


def register():
    """Register ONLY new source/archive paths against their already-created commit."""
    def git(*args):
        return subprocess.check_output(['git',*args],timeout=30)
    require(git('branch','--show-current').decode().strip() == BRANCH and not git('status','--porcelain'), 'clean fixed archive branch')
    commit = git('rev-parse','HEAD').decode().strip()
    tree = {}
    for item in git('ls-tree','-r','-l','-z','HEAD').split(b'\0'):
        if item:
            meta,name=item.split(b'\t',1); mode,kind,blob,size=meta.decode().split()
            require(kind == 'blob','no submodules'); tree[name.decode()] = (mode,blob,int(size))
    original=Path(MANIFEST).read_bytes(); value=parse(original)
    require(value['schema']=='kws-public-branch-retention-v1','retention schema')
    rows={r['path']:r for r in value['retained_files']}; integration=set(value['integration_files'])
    require(len(rows)==len(value['retained_files'])==value['expected_retained_files'] and
            len(integration)==len(value['integration_files']) and not integration & rows.keys() and
            len(rows)+len(integration)==value['expected_tracked_files'], 'original inventory')
    for name,row in rows.items():
        require(tree.get(name)==(row['mode'],row['git_blob_sha1'],row['bytes']),'retained file changed: '+name)
    require(set(rows)|integration <= set(tree),'original file removed')
    fresh=set(tree)-set(rows)-integration
    members = {'artifact-freeze.json','container-setup.json','container-tts.json','host-receipt.json','image-transfer.json','ledger.json','plan.json','output/generation-receipt.json','output/kernel-scope.json','output/progress.json','output/runtime-observations.json'}
    members |= {'output/qwen16-20261030-%03d%s' % (i,suffix) for i in range(16) for suffix in ('.claim','.wav','.raw-float.wav')}
    expected = {WORKFLOW} | {SCOPE+n for n in ('verify_archive.py','test_archive.py','README.md')}
    expected |= {DEST+'/'+n for n in ('original.zip','verification.json','review-map.json','review.html')}
    expected |= {DEST+'/artifact/'+n for n in members}
    require(fresh == expected, 'only exact source and original archive additions')
    result=copy.deepcopy(value)
    for name in sorted(fresh):
        mode,blob,size=tree[name]; require(mode=='100644','regular new file')
        result['retained_files'].append({'path':name,'mode':mode,'git_blob_sha1':blob,'bytes':size,'source_commit':commit})
    result['expected_retained_files']=len(result['retained_files']); result['expected_tracked_files']=len(tree)
    result.setdefault('source_registration_history',[]).append({'campaign':'qwen16-original-archive-41008894901',
        'source_commit':commit,'paths':sorted(fresh),'previous_manifest_sha256':sha(original),
        'scope':'exact original TTS evidence; labels pending; no new inference/training'})
    Path(MANIFEST).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'new_paths':len(fresh),'retained_before':len(rows),'retained_after':len(result['retained_files']),'source_commit':commit}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    if args.register:
        require(args.archive is None and args.output is None,'separate registration mode');register()
    else:
        require(args.archive is not None,'archive required')
        report=materialize(args.archive,args.output) if args.output else read_archive(args.archive)[2]
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,sort_keys=True))
