#!/usr/bin/env python3
"""Stdlib archive integrity/count checks; no training, inference or corpus admission."""
import hashlib
import json
import math
import pathlib
import struct
import statistics

BASE = pathlib.Path(__file__).resolve().parent
SPEC_SHA = 'c0022e6abee8555d1abfa982d7c947da8a662e09f5bc39c88fd41d1b2edf6dc5'
RUNS = ('A-seed1337','A-seed2346','B-seed1337','B-seed2346')
COPIED = {'spec.json','review-approval.json','RESULTS.md','tests.log','postcheck.json','feature-audit.json',
          'data/rows.json','data/native-export-receipt.json',
          'runtime/install-report.json','runtime/install.log','runtime/requirements.research.lock'}
COPIED |= {'historical-source/'+x for x in ('models.py','test_models.py','prepare.py','run.py','postcheck.py','feature_audit.py')}
COPIED |= {'revisions/preregistered-v1/'+x for x in ('spec.json','models.py','test_models.py','prepare.py','run.py')}
COPIED |= {'results/'+r+'/'+x for r in RUNS for x in ('invocation.json','progress.json','summary.json','clip-scores.json')}
GENERATED = {'README.md','verify.py','test_verify.py','feature-fingerprints.json','copied-file-origins.json',
             'checkpoint_conversion.py','PIPELINE_ROADMAP_UPDATE.md','results-recomputed.json'}
GENERATED |= {'results/'+r+'/checkpoint.weights.json' for r in RUNS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def safe_path(root, name):
    p = pathlib.Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe archive path')
    current = root
    for part in p.parts:
        current /= part
        require(not current.is_symlink(), 'symlink archive path')
    require(root.resolve() in current.resolve().parents, 'archive path escapes root')
    return current


def recalculate(details):
    groups = {}
    for method in ('human','asr'):
        for role in ('train','development_a','development_b'):
            subset=[d for d in details if d['review_method']==method and d['split']==role]
            if not subset:
                continue
            positive=[d for d in subset if d['kind']=='positive']
            negative=[d for d in subset if d['kind']=='confusable']
            groups[method+':'+role] = {
                'clips':len(subset),'positives':len(positive),'confusables':len(negative),
                'positive_target_hits':sum(d['keyword_id'] in d['predicted_keywords'] for d in positive),
                'positive_wrong_keyword_clips':sum(any(k!=d['keyword_id'] for k in d['predicted_keywords']) for d in positive),
                'confusable_triggered_clips':sum(bool(d['predicted_keywords']) for d in negative),
                'exact_clip_labels_correct':sum(d['predicted_keywords']==([d['keyword_id']] if d['kind']=='positive' else []) for d in subset),
                'per_keyword':{str(k):{'positive_clips':sum(d['keyword_id']==k for d in positive),
                    'target_hits':sum(d['keyword_id']==k and k in d['predicted_keywords'] for d in positive),
                    'confusable_triggered_clips':sum(k in d['predicted_keywords'] for d in negative)} for k in (1,2)}}
    return groups


def tensor_shapes(architecture):
    shapes={'stem.weight':[48,32,1],'stem.bias':[48],'head.weight':[2,48,1],'head.bias':[2]}
    for i in range(5 if architecture=='A' else 4):
        prefix='blocks.'+str(i)+'.'
        layers=({'depthwise.weight':[48,1,5],'depthwise.bias':[48],
                 'pointwise.weight':[48,48,1],'pointwise.bias':[48]} if architecture=='A' else
                {'project.weight':[24,48,1],'memory.weight':[24,1,9],
                 'expand.weight':[48,24,1],'expand.bias':[48]})
        shapes.update({prefix+k:v for k,v in layers.items()})
    return shapes


def verify(root=BASE):
    manifest=read(root,'archive-manifest.json')
    require(type(manifest.get('schema_version')) is int and manifest['schema_version']==1 and manifest.get('purpose')=='historical-evidence-integrity-not-data-catalog', 'archive purpose')
    seen=set()
    for entry in manifest['files']:
        name=entry['path'];require(name not in seen,'duplicate archive member');seen.add(name)
        p=safe_path(root,name)
        require(p.is_file() and type(entry['bytes']) is int and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'], 'archive hash mismatch: '+name)
        p.read_text(encoding='utf-8')
    require(seen==COPIED|GENERATED,'mandatory archive coverage mismatch')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(actual==seen|{'archive-manifest.json'},'unindexed archive files')
    require(not any(pathlib.Path(p).suffix in {'.wav','.npz','.npy','.pt','.whl','.so','.onnx'} for p in actual),'binary/audio/cache payload forbidden')
    origins=read(root,'copied-file-origins.json')
    require(len(origins)==len(COPIED) and {x['archive_path'] for x in origins}==COPIED,'copied origins coverage')
    for item in origins:
        p=root/item['archive_path']
        require(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'copied bytes changed')
    spec=read(root,'spec.json')
    require(sha(root/'spec.json')==SPEC_SHA,'preregistered spec identity')
    require(spec['steps']==1000 and spec['checkpoint_selection']=='last-step-only' and spec['asr_training_allowed'] is False,'protocol drift')
    for name,value in spec['source_hashes'].items():
        require(sha(root/'historical-source'/name)==value,'approved source identity')
    require(sha(root/'runtime/install-report.json')==spec['install_report_sha256'],'runtime receipt identity')
    old=read(root,'revisions/preregistered-v1/spec.json')
    require(sha(root/'revisions/preregistered-v1/spec.json')==spec['superseded_untrained_spec_sha256'],'untrained revision identity')
    for name,value in old['source_hashes'].items():
        require(sha(root/'revisions/preregistered-v1'/name)==value,'untrained revision source identity')
    approval=read(root,'review-approval.json')
    require(approval['approved'] is True and approval['spec_sha256']==SPEC_SHA and
            approval['approved_runs']==[['A',1337],['A',2346],['B',1337],['B',2346]],'review scope')
    receipt=read(root,'data/native-export-receipt.json')
    require(sha(root/'data/native-export-receipt.json')==spec['native_export_sha256'] and
            receipt['data_repository_commit']==spec['data_commit'] and receipt['qualification_allowed'] is False,'native receipt pin/scope')
    native={r['recording']:dict(r,dataset_id=d['dataset_id']) for d in receipt['datasets'] for r in d['recordings']}
    rows=read(root,'data/rows.json')
    require(sha(root/'data/rows.json')==spec['rows_sha256'] and len(rows)==42 and len(native)==42,'native row identity')
    by_name={r['recording']:r for r in rows}
    require(len(by_name)==42 and set(by_name)==set(native),'row coverage')
    for r in rows:
        require(all(r[k]==v for k,v in native[r['recording']].items()),'native field changed')
        require(r['feature_frames']==max(0,(r['frames']-400)//320+1),'feature frame geometry')
    fingerprints=read(root,'feature-fingerprints.json')
    require(fingerprints['spec_sha256']==SPEC_SHA and fingerprints['feature_cache_sha256']==spec['feature_cache_sha256'],'feature cache identity')
    fp={r['recording']:r for r in fingerprints['recordings']}
    require(len(fingerprints['recordings'])==42 and set(fp)==set(native),'feature fingerprint coverage')
    require(fingerprints['frame_count_total']==3886==sum(r['feature_frames'] for r in rows),'total feature frames')
    for name,f in fp.items():
        require(all(f[k]==by_name[name][k] for k in ('file_sha256','pcm_sha256','frames','feature_frames','feature_jsonl_sha256')),'feature source binding')
        require(f['features_shape']==[f['feature_frames'],32],'feature shape')
    outcomes={}
    for run in RUNS:
        prefix='results/'+run+'/'
        architecture=run[0];seed=int(run.split('seed')[1])
        summary=read(root,prefix+'summary.json');details=read(root,prefix+'clip-scores.json')
        require(summary['completed'] is True and summary['steps']==1000 and summary['threshold_logit']==0 and summary['spec_sha256']==SPEC_SHA,'run completion/protocol')
        invocation=read(root,prefix+'invocation.json')
        require(invocation=={'spec_sha256':SPEC_SHA,'review_sha256':sha(root/'review-approval.json'),'architecture':architecture,'seed':seed},'invocation binding')
        progress=read(root,prefix+'progress.json')
        require([p['step'] for p in progress]==[1]+list(range(100,1001,100)) and all(math.isfinite(p['batch_loss']) for p in progress),'training progress')
        require(len(details)==42 and {d['recording'] for d in details}==set(native),'score coverage')
        for d in details:
            original=native[d['recording']]
            require(all(d[k]==original[k] for k in ('dataset_id','split','review_method','kind','keyword_id','intended_text','speaker_id')),'prediction native labels changed')
            scores=d['max_logits']
            require(len(scores)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in scores),'nonfinite/malformed scores')
            require(d['predicted_keywords']==[i+1 for i,x in enumerate(scores) if x>=0],'score threshold mismatch')
            require(len(d['argmax_frames'])==2 and all(type(x) is int and 0<=x<by_name[d['recording']]['feature_frames'] for x in d['argmax_frames']),'argmax range')
        counts=recalculate(details)
        require(counts==summary['groups'],'recomputed summary mismatch')
        weights=read(root,prefix+'checkpoint.weights.json')
        require(type(weights.get('schema_version')) is int and weights['schema_version']==1,'checkpoint schema')
        require(weights['format']=='finite-float32-tensor-json-v1' and weights['qualification_allowed'] is False and
                weights['architecture']==architecture and weights['seed']==seed and weights['completed'] is True and
                weights['spec_sha256']==SPEC_SHA and weights['original_checkpoint_sha256']==summary['checkpoint_sha256'],'checkpoint identity')
        expected=tensor_shapes(architecture)
        require(set(weights['tensors'])==set(expected),'checkpoint tensor names')
        total=0
        for name,shape in expected.items():
            tensor=weights['tensors'][name];values=tensor['values'];total+=len(values)
            require(tensor['dtype']=='float32' and tensor['shape']==shape and
                    all(type(d) is int for d in tensor['shape']) and len(values)==math.prod(shape),'tensor shape/count')
            require(all(type(x) in (float,int) and math.isfinite(x) for x in values),'nonfinite tensor value')
            packed=struct.pack('<'+str(len(values))+'f',*values)
            require(hashlib.sha256(packed).hexdigest()==tensor['little_endian_float32_sha256'],'tensor data identity')
        require(total==spec['architectures'][architecture]['parameters'],'parameter count')
        outcomes[run]=counts
    require(read(root,'results-recomputed.json')==outcomes,'compact result mismatch')
    parity=read(root,'postcheck.json')
    require(len(parity)==4 and {x['run'] for x in parity}==set(RUNS) and
            all(x['clips']==42 and x['chunk_parity_passed'] is True and x['threshold_decisions_equal'] is True and
                math.isfinite(x['max_abs_logit_difference']) and 0<=x['max_abs_logit_difference']<2e-5 for x in parity),'retained parity results')
    audit=read(root,'feature-audit.json')
    require(audit['scope']=='post-result-descriptive-not-causal-not-significance-test','feature audit scope')
    expected_groups={method+':'+role:[r for r in rows if r['review_method']==method and r['split']==role]
                     for method in ('human','asr') for role in ('train','development_a','development_b')}
    expected_groups={k:v for k,v in expected_groups.items() if v}
    speakers={r['speaker_id'] for r in rows if r['review_method']=='human'}
    require(set(audit['groups'])==set(expected_groups) and set(audit['human_speakers'])==speakers,'feature audit coverage')
    selections=list((audit['groups'][k],v) for k,v in expected_groups.items())
    selections += [(audit['human_speakers'][speaker], [r for r in rows if r['review_method']=='human' and r['speaker_id']==speaker]) for speaker in speakers]
    for values,selected in selections:
        require(type(values['clips']) is int and values['clips']==len(selected) and
                type(values['frames']) is int and values['frames']==sum(r['feature_frames'] for r in selected), 'feature audit counts')
        require(math.isclose(values['seconds_total'],sum(r['duration_s'] for r in selected),abs_tol=1e-9) and
                math.isclose(values['duration_median_s'],statistics.median(r['duration_s'] for r in selected),abs_tol=1e-9), 'feature audit durations')
        fields=('feature_std','feature_min','feature_max','rms_train_standardized_centroid_shift',
                'frame_fraction_dbfs_above_minus50','seconds_total','duration_median_s')
        require(all(type(values[k]) in (float,int) and math.isfinite(values[k]) for k in fields), 'nonfinite feature audit')
        require(values['feature_std']>=0 and values['feature_min']<=values['feature_max'] and
                values['rms_train_standardized_centroid_shift']>=0 and
                0<=values['frame_fraction_dbfs_above_minus50']<=1,'feature audit range')
        q=values['dbfs_p10_p50_p90']
        require(len(q)==3 and all(type(x) in (float,int) and math.isfinite(x) for x in q) and q==sorted(q), 'feature audit quantiles')
    return {'verified':True,'scope':'archive byte/tensor/count checks only; no numeric inference replay',
            'runs':4,'steps_per_run':1000,'recordings_per_run':42,'feature_frames':3886,
            'qualification_allowed':False}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
