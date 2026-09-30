#!/usr/bin/env python3
"""Verify this historical evidence archive, not the canonical audio catalog."""
import collections
import hashlib
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
CONDITIONS = ('raw', 'tail500ms', 'head500ms_tail500ms')
COPIED_MEMBERS = frozenset(['boundary/condition-comparison.json',
 'boundary/head500ms_tail500ms-derived-receipt.json',
 'boundary/probe-spec.json',
 'boundary/tail500ms-derived-receipt.json',
 'data/native-export-receipt.json',
 'external/REPORT.md',
 'external/config.json',
 'external/head500ms_tail500ms/raw-events.jsonl',
 'external/head500ms_tail500ms/readback.json',
 'external/head500ms_tail500ms/run-provenance.json',
 'external/historical-scripts/prepare.py',
 'external/historical-scripts/run.py',
 'external/historical-scripts/run_boundary.py',
 'external/keywords.txt',
 'external/license/model-README.md',
 'external/raw/raw-events.jsonl',
 'external/raw/readback.json',
 'external/raw/run-provenance.json',
 'external/sources/downloads.json',
 'external/sources/numpy-2.2.6-pypi.json',
 'external/sources/pip-freeze.txt',
 'external/sources/runtime-LICENSE',
 'external/sources/runtime-keyword-spotter.py',
 'external/sources/runtime-keyword_spotter.py',
 'external/sources/sherpa-onnx-1.13.8-pypi.json',
 'external/sources/sherpa-onnx-core-1.13.8-pypi.json',
 'external/tail500ms/raw-events.jsonl',
 'external/tail500ms/readback.json',
 'external/tail500ms/run-provenance.json',
 'ours/artifact-index.json',
 'ours/generated/kws_build_config.h',
 'ours/generated/kws_parameter_limits.h',
 'ours/head500ms_tail500ms/clip-readback.json',
 'ours/head500ms_tail500ms/detections.jsonl',
 'ours/head500ms_tail500ms/execution-inputs.jsonl',
 'ours/head500ms_tail500ms/run-provenance.json',
 'ours/manual-build-receipt.json',
 'ours/raw/clip-readback.json',
 'ours/raw/detections.jsonl',
 'ours/raw/execution-inputs.jsonl',
 'ours/raw/run-provenance.json',
 'ours/readback-spec.json',
 'ours/registry/MODEL_SHA256SUMS',
 'ours/registry/registry.json',
 'ours/registry/tokens.example.txt',
 'ours/registry/xiaowo-keywords.tsv',
 'ours/source-main-comparison.json',
 'ours/tail500ms/clip-readback.json',
 'ours/tail500ms/detections.jsonl',
 'ours/tail500ms/execution-inputs.jsonl',
 'ours/tail500ms/run-provenance.json'])
GENERATED_MEMBERS = frozenset({'README.md','verify.py','test_verify.py','copied-file-origins.json'})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root, path):
    return json.loads((root / path).read_text())


def lines(root, path):
    return [json.loads(x) for x in (root / path).read_text().splitlines() if x.strip()]


def aggregate(rows, event_key):
    counts = collections.Counter()
    for row in rows:
        events = row[event_key]
        counts['clips'] += 1
        counts['events'] += len(events)
        if row['kind'] == 'positive':
            hits = sum(e['keyword_id'] == row['keyword_id'] for e in events)
            counts['positives'] += 1
            counts['target_hits'] += bool(hits)
            counts['target_misses'] += not bool(hits)
            counts['wrong_keyword_events'] += len(events) - hits
            counts['additional_target_events'] += max(0, hits - 1)
            require(row['target_hit'] == bool(hits), 'target-hit field mismatch')
            require(row['target_miss'] == (not bool(hits)), 'target-miss field mismatch')
            require(row['wrong_keyword_events'] == len(events)-hits, 'wrong-keyword field mismatch')
            require(row['additional_target_events'] == max(0,hits-1), 'duplicate-target field mismatch')
        else:
            require(row['kind'] == 'confusable', 'unknown clip kind')
            counts['confusables'] += 1
            counts['confusable_clips_with_events'] += bool(events)
            counts['confusable_events'] += len(events)
    return dict(counts)


def verify(root=BASE):
    manifest = read(root, 'archive-manifest.json')
    require(manifest['purpose'] == 'historical-evidence-integrity-not-data-catalog', 'manifest purpose')
    seen = set()
    for entry in manifest['files']:
        relative = pathlib.Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe archive path')
        require(str(relative) not in seen, 'duplicate manifest path')
        seen.add(str(relative))
        p = root / relative
        require(p.is_file() and not p.is_symlink(), 'missing or symlink archive file')
        require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'archive hash mismatch: ' + str(relative))
        try:
            p.read_bytes().decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('non-UTF8 archive payload: '+str(relative)) from exc
    require(seen == COPIED_MEMBERS | GENERATED_MEMBERS, 'mandatory artifact coverage mismatch')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(actual == seen | {'archive-manifest.json'}, 'unindexed archive files')
    require(not any(pathlib.Path(x).suffix.lower() in {'.wav','.onnx','.whl','.so','.bin','.tflite'} for x in actual), 'binary/audio payload forbidden')
    origins=read(root, 'copied-file-origins.json')
    require(len(origins)==len(COPIED_MEMBERS) and {x['archive_path'] for x in origins}==COPIED_MEMBERS,'copied origins coverage mismatch')
    for entry in origins:
        p = root / entry['archive_path']
        require(sha(p) == entry['sha256'] and p.stat().st_size == entry['bytes'], 'copied source bytes changed')
    receipt = read(root, 'data/native-export-receipt.json')
    require(sha(root/'data/native-export-receipt.json') == '8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5', 'native receipt pin')
    require(receipt['data_repository_commit'] == '2f9658ffa9568076ef547615c76861abed84f56e', 'native data commit')
    native = {x['recording']: x for d in receipt['datasets'] for x in d['recordings']}
    require(len(native) == 42, 'native clip count')
    for d in receipt['datasets']:
        encoded = json.dumps(d['identity'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        require(d['content_id'] == 'sha256:' + hashlib.sha256(encoded).hexdigest(), 'dataset content ID')
    result = {}
    for engine, filename, eventkey, eventfile in [('ours','clip-readback.json','detections','detections.jsonl'),('external','readback.json','events','raw-events.jsonl')]:
        for condition in CONDITIONS:
            prefix = f'{engine}/{condition}'
            report = read(root, prefix+'/'+filename)
            rows = report['recordings']
            require(len(rows)==42 and {x['recording'] for x in rows}==set(native), 'report clip coverage')
            for row in rows:
                original = native[row['recording']]
                require(all(row[k] == original[k] for k in ['keyword_id','kind','split','review_method','speaker_id','intended_text']), 'native labels/roles changed')
                if condition == 'raw':
                    require(row['file_sha256']==original['file_sha256'] and row['pcm_sha256']==original['pcm_sha256'], 'raw audio identity')
                else:
                    require(row['source_file_sha256']==original['file_sha256'] and row['source_pcm_sha256']==original['pcm_sha256'], 'derived source identity')
            counts = aggregate(rows, eventkey)
            expected = dict(clips=42,events=0 if engine=='ours' else 19,positives=20,target_hits=0 if engine=='ours' else 19,target_misses=20 if engine=='ours' else 1,wrong_keyword_events=0,additional_target_events=0,confusables=22,confusable_clips_with_events=0,confusable_events=0)
            require(counts == expected, 'recomputed semantic count mismatch: '+prefix)
            raw_events = lines(root,prefix+'/'+eventfile)
            flattened = [dict(e,recording=row['recording']) for row in rows for e in row[eventkey]]
            require(raw_events == flattened, 'raw event/report mismatch: '+prefix)
            provenance=read(root,prefix+'/run-provenance.json')
            if engine == 'external':
                bindings={'config_sha256':'external/config.json','keywords_sha256':'external/keywords.txt','native_export_sha256':'data/native-export-receipt.json','downloads_sha256':'external/sources/downloads.json','readback_sha256':prefix+'/readback.json','script_sha256':'external/historical-scripts/'+('run.py' if condition=='raw' else 'run_boundary.py')}
                require(all(provenance[k]==sha(root/v) for k,v in bindings.items()),'external provenance binding mismatch')
                require(provenance['sherpa_onnx']=='1.13.8' and provenance['numpy']=='2.2.6' and provenance['data_commit']==receipt['data_repository_commit'],'external provenance version mismatch')
                require(all(not e['eof_flush'] for e in raw_events), 'unexpected EOF-only event')
                require([x['recording'] for x in rows if x['target_miss']] == ['qwen3-kw2-eric'], 'external miss identity')
                require(report['config']['keywords_threshold']==0.25 and report['config']['keywords_score']==1.0, 'external frozen thresholds')
                total = next(g for g in report['groups'] if g['dimension']=='all')
                require(all(total[k]==v for k,v in counts.items()), 'external aggregate mismatch')
                expected_groups={('all','all')}|{('role',str(x['split'])) for x in rows}|{('review',str(x['review_method'])) for x in rows}|{('dataset',str(x['dataset_id'])) for x in rows}|{('keyword',str(x['keyword_id'])) for x in rows}
                group_ids=[(g['dimension'],g['value']) for g in report['groups']]
                require(len(group_ids)==len(set(group_ids)) and set(group_ids)==expected_groups,'external subgroup coverage mismatch')
                for group in report['groups']:
                    dimension,value=group['dimension'],group['value']
                    field={'role':'split','review':'review_method','dataset':'dataset_id','keyword':'keyword_id'}.get(dimension)
                    selected=rows if dimension=='all' else [x for x in rows if str(x[field])==value]
                    calculated=aggregate(selected,eventkey)
                    require(all(group.get(k,0)==v for k,v in calculated.items()), 'external subgroup mismatch')
                frozen=read(root,'external/config.json')
                require(all(report['config'][k]==v for k,v in frozen.items()), 'external config changed')
            else:
                require(provenance['detections_sha256']==sha(root/(prefix+'/detections.jsonl')) and provenance['references_sha256']==sha(root/(prefix+'/execution-inputs.jsonl')),'canonical provenance data binding mismatch')
                spec=read(root,'ours/readback-spec.json')
                require(provenance['model_sha256']==spec['model_sha256'] and provenance['keyword_pack_sha256']==spec['keyword_pack_sha256'] and provenance['runner_sha256']==spec['runner_sha256'],'canonical frozen model binding mismatch')
                report_key='execution_provenance_sha256' if condition=='raw' else 'run_provenance_sha256'
                require(report[report_key]==sha(root/(prefix+'/run-provenance.json')),'canonical report provenance binding mismatch')
                measured={x['recording']:x for x in provenance['audio_files']}
                require(len(provenance['audio_files'])==42 and set(measured)==set(native),'canonical provenance audio coverage')
                require(all(all(row[k]==measured[row['recording']][k] for k in ['file_sha256','pcm_sha256','frames','duration_s','path']) for row in rows),'canonical provenance audio identity')
                input_rows=lines(root,prefix+'/execution-inputs.jsonl')
                require(len(input_rows)==42 and {x['recording'] for x in input_rows}==set(native),'canonical input coverage')
                report_rows={x['recording']:x for x in rows}
                require(all(all(report_rows[x['recording']][k]==v for k,v in x.items()) for x in input_rows),'canonical input/report mismatch')
                require(provenance['detections']==len(raw_events) and provenance['recordings']==42,'canonical provenance counts')
                require(report['qualification_allowed'] is False, 'qualification not permitted')
                keys={'recordings':'clips','positives':'positives','target_hits':'target_hits','target_misses':'target_misses','additional_target_events':'additional_target_events','wrong_keyword_events':'wrong_keyword_events','confusables':'confusables','confusable_clips_with_events':'confusable_clips_with_events','confusable_event_count':'confusable_events'}
                group_ids=set()
                for group in report['groups']:
                    key=tuple(group[k] for k in ['dataset_id','split','review_method','keyword_id'])
                    require(key not in group_ids,'duplicate canonical subgroup')
                    group_ids.add(key)
                    selected=[x for x in rows if tuple(x[k] for k in ['dataset_id','split','review_method','keyword_id'])==key]
                    calculated=aggregate(selected,eventkey)
                    require(all(group[k]==calculated.get(v,0) for k,v in keys.items()),'canonical subgroup counter mismatch')
                require(group_ids=={tuple(x[k] for k in ['dataset_id','split','review_method','keyword_id']) for x in rows},'canonical subgroup coverage')
            result[prefix]=counts
    for condition in CONDITIONS[1:]:
        derived=read(root,'boundary/'+condition+'-derived-receipt.json')
        require(derived['canonical_source_receipt_sha256']==sha(root/'data/native-export-receipt.json'),'padding receipt native pin')
        rows={x['recording']:x for x in derived['recordings']}
        require(set(rows)==set(native),'padding coverage')
        expected_prefix=0 if condition=='tail500ms' else 8000
        ours={x['recording']:x for x in read(root,'ours/'+condition+'/clip-readback.json')['recordings']}
        external={x['recording']:x for x in read(root,'external/'+condition+'/readback.json')['recordings']}
        for name,row in rows.items():
            require(row['prefix_frames']==expected_prefix and row['tail_frames']==8000,'padding recipe')
            require(row['derived_frames']==native[name]['frames']+expected_prefix+8000,'padding frame count')
            require(row['source_pcm_sha256']==native[name]['pcm_sha256'],'padding source PCM')
            require(row['derived_pcm_sha256']==ours[name]['pcm_sha256']==external[name]['pcm_sha256'],'cross-engine derived PCM identity')
            require(row['derived_file_sha256']==ours[name]['file_sha256']==external[name]['file_sha256'],'cross-engine derived WAV identity')
    return result


if __name__=='__main__':
    try:
        result=verify()
        print(json.dumps({'verified':True,'scope':'archive integrity and semantic consistency; no inference rerun','conditions':result},indent=2))
    except (ValueError,KeyError,OSError,StopIteration) as exc:
        print('verification failed: '+str(exc),file=sys.stderr)
        sys.exit(1)
