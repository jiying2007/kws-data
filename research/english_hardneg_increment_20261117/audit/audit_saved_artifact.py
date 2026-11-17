"""Independent stdlib byte/record audit; executes no model or candidate code."""
import ast
import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

EXPECTED = '5912bec75bedc33def9c07c6013b497b1999849aec5f6ff54107e70ab5a3acd6'


def check(value, reason):
    if not value:
        raise ValueError(reason)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def arrays(raw):
    shapes = {'mean96': (96,), 'scale96': (96,), 'w1': (32, 1536),
              'b1': (32,), 'w2': (3, 32), 'b2': (3,)}
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        check(len(z.namelist()) == 6 and set(z.namelist()) == {k+'.npy' for k in shapes}, 'npz keys')
        for key, shape in shapes.items():
            data = z.read(key + '.npy')
            check(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00'), 'npy magic')
            offset, fmt = (10, '<H') if data[6] == 1 else (12, '<I')
            count = struct.unpack_from(fmt, data, 8)[0]
            header = ast.literal_eval(data[offset:offset+count].decode('latin1'))
            check(header == {'descr': '<f4', 'fortran_order': False, 'shape': shape}, 'npy header')
            body = data[offset+count:]
            check(len(body) == 4*math.prod(shape), 'npy size')
            check(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', body)), 'npy finite')
            result[key] = body
    return result


def trace_scores(raw):
    check(raw[:8] == b'EKWSTRC1' and struct.unpack_from('<6I', raw, 8) == (1,1,16000,320,96,3), 'trace header')
    scores, pos, expected_tag, ordinal = [], 40, 1, 0
    while pos < len(raw):
        tag, count = struct.unpack_from('<BI', raw, pos)
        pos += 5
        body = raw[pos:pos+count]
        check(len(body) == count, 'trace record truncated')
        pos += count
        if tag == 1:
            check(expected_tag == 1 and count == 400, 'embedding record')
            o, t = struct.unpack_from('<QQ', body)
            check(o == ordinal and t == 16000+1280*ordinal, 'embedding clock')
            expected_tag = 2
        elif tag == 2:
            check(expected_tag == 2 and count == 48, 'score record')
            o, ready, keyword, t = struct.unpack_from('<QIIQ', body)
            nums = struct.unpack_from('<6f', body, 24)
            check(o == ordinal and t == 16000+1280*ordinal and ready == int(ordinal >= 15), 'score clock')
            check(keyword == 0 and all(math.isfinite(x) for x in nums), 'natural baseline events/finite')
            scores.append({'sample': t, 'ready': bool(ready), 'keyword': keyword,
                           'logits': list(nums[:3]), 'probabilities': list(nums[3:])})
            ordinal += 1
            expected_tag = 1
        elif tag == 255:
            check(expected_tag == 1 and count == 48 and pos == len(raw), 'trace terminator')
            total, embeds, rows, ready, events, calls = struct.unpack('<6Q', body)
            check(total == struct.unpack_from('<Q', raw, 32)[0] and embeds == rows == len(scores), 'trace totals')
            check(ready == sum(s['ready'] for s in scores) and events == 0, 'trace totals ready/events')
            break
        else:
            raise ValueError('unexpected natural trace tag')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'output already exists')
    artifact = args.artifact.read_bytes()
    check(pin(artifact)['sha256'] == EXPECTED, 'artifact SHA')
    with zipfile.ZipFile(io.BytesIO(artifact)) as z:
        load = lambda p: json.loads(z.read(p))
        lines = lambda p: [json.loads(l) for l in z.read(p).splitlines()]
        manifest = load('MANIFEST.json')
        check(len(z.namelist()) == len(set(z.namelist())) == 190, 'unique archive names')
        check(set(z.namelist()) == set(manifest['files']) | {'MANIFEST.json'}, 'manifest closure')
        for name, identity in manifest['files'].items():
            check(pin(z.read(name)) == identity, 'manifest hash ' + name)
        base = arrays(z.read('inputs/coverage/models/coverage-head.npz'))
        candidate = arrays(z.read('models/incremental-head.npz'))
        check(z.read('models/incremental-head.bin') == b'EHKWS01\0'+b''.join(candidate.values()), 'npz/bin exact')
        check(all(base[k] == candidate[k] for k in ('mean96','scale96')), 'normalizer unchanged')
        check(all(base[k] != candidate[k] for k in ('w1','b1','w2','b2')), 'all trainable arrays changed')
        frozen = load('evidence/CANDIDATE_FROZEN.json')
        check(frozen['npz'] == pin(z.read('models/incremental-head.npz')) and frozen['bin'] == pin(z.read('models/incremental-head.bin')), 'frozen weights')
        check(frozen['evaluation_model_calls_before_freeze'] == 0, 'freeze before evaluation')
        old, old_frames = [], {}
        for cond in ('clean','snr10','snr5','new8'):
            old += [dict(r,condition=cond) for r in load('inputs/coverage/predictions/coverage-'+cond+'-events.json')]
            for r in lines('inputs/coverage/predictions/coverage-'+cond+'-frames.jsonl'):
                old_frames.setdefault((cond,r['id']),[]).append(r)
        new = load('predictions/candidate-events.json')
        candidate_frames = collections.defaultdict(list)
        for r in lines('predictions/candidate-frames.jsonl'):
            candidate_frames[(r['condition'],r['id'])].append(r)
        check(len(old) == len(new) == 291, 'paired case count')
        recomputed, grouped, changed = [], {}, []
        for a,b in zip(old,new):
            identity = ('condition','id','role','weak_class','observed_samples','clip_start_sample','original_clip_eof_sample')
            check(all(a[k] == b[k] for k in identity), 'paired identities')
            key = (a['condition'],a['id'])
            for table,row in ((old_frames,a),(candidate_frames,b)):
                events = [{'sample': f['sample'], 'keyword': f['keyword']} for f in table[key] if f['keyword']]
                check(events == row['events'], 'events recompute')
            counts = ['prefix_events','wrong_keyword_after_clip_start']
            numerics = ['extra_same_keyword_events','weak_negative_intent_events']
            increased = [k for k in counts if len(b[k]) > len(a[k])] + [k for k in numerics if b[k] > a[k]]
            row = {k:a[k] for k in ('condition','id','role','weak_class')}
            row.update(prior_matched_intent_lost=a['matched_intent_after_clip_start'] and not b['matched_intent_after_clip_start'],
                prior_correct_case_regressed=not a['research_error'] and b['research_error'],
                event_error_categories_increased=increased,old_events=a['events'],new_events=b['events'])
            recomputed.append(row)
            g = (a['condition'],a['role'],a['weak_class'])
            d = grouped.setdefault(g,{'cases':0,'old_matches':0,'new_matches':0,'old_errors':0,'new_errors':0})
            d['cases']+=1
            d['old_matches']+=bool(a['matched_intent_after_clip_start']);d['new_matches']+=bool(b['matched_intent_after_clip_start'])
            d['old_errors']+=bool(a['research_error']);d['new_errors']+=bool(b['research_error'])
            if row['prior_matched_intent_lost'] or row['prior_correct_case_regressed'] or increased:
                diag = dict(row)
                for label,table in [('old',old_frames),('new',candidate_frames)]:
                    fs = [f for f in table[key] if f['ready'] and (a['clip_start_sample'] is None or f['sample'] > a['clip_start_sample'])]
                    k = a['weak_class']; ps = [f['probabilities'][k] for f in fs]
                    triples = [(min(ps[i:i+3]),fs[i+2]['sample']) for i in range(len(ps)-2)]
                    diag[label+'_target_peak'] = max(ps)
                    diag[label+'_target_max_three_frame_min'],diag[label+'_best_three_frame_end_sample'] = max(triples)
                changed.append(diag)
        check(recomputed == load('evidence/PAIRED_REGRESSION.json')['cases'], 'paired report exactly reproduced')
        check(len(changed) == 5 and all(r['prior_matched_intent_lost'] for r in changed), 'five lost cases')
        selection = load('evidence/ADMITTED_HEAD_INPUTS.json')
        source_ce = []
        for s in selection['sources']:
            rid = s['recording_id']
            a = {f['sample']:f for f in trace_scores(z.read('inputs/natural/traces/'+rid+'.trace'))}
            b = {f['sample']:f for f in candidate_frames[('natural',rid)]}
            avgs = [sum(-math.log(max(t[ts]['probabilities'][0],1e-12)) for ts in s['head_ends'])/len(s['head_ends']) for t in (a,b)]
            source_ce.append({'recording_id':rid,'head_inputs':len(s['head_ends']),'old_background_ce':avgs[0],'new_background_ce':avgs[1]})
        fit = load('evidence/REVIEWED_FIT.json')
        check(source_ce == fit['sources'], 'source CE exactly recomputed')
        check(sum(s['old_background_ce'] for s in source_ce)/8 == fit['old_source_balanced_ce'], 'old CE exact')
        check(sum(s['new_background_ce'] for s in source_ce)/8 == fit['new_source_balanced_ce'], 'new CE exact')
        ref = lines('runtime/reference-frames.jsonl')
        native = z.read('evidence/head-native.stdout').decode('ascii').splitlines()
        check(len(ref) == len(native) == 18982, 'native total')
        maxlogit,maxprob = 0.,0.
        for r,line in zip(ref,native):
            fields=line.split();check(len(fields)==9,'native row columns')
            check(tuple(map(int,fields[:3]))==(r['sample'],int(r['ready']),r['keyword']),'native clock ready event')
            vals=list(map(float,fields[3:]));check(all(math.isfinite(v) for v in vals),'native finite')
            maxlogit=max(maxlogit,max(abs(a-b) for a,b in zip(vals[:3],r['logits'])))
            maxprob=max(maxprob,max(abs(a-b) for a,b in zip(vals[3:],r['probabilities'])))
        parity=load('evidence/HEAD_NATIVE_PARITY.json')
        check(maxlogit==parity['max_abs_logit_error'] and maxprob==parity['max_abs_probability_error'],'native maxima exact')
        check(maxlogit<=1e-4 and maxprob<=2e-6,'native tolerance')
        natural_events=sum(bool(r['keyword']) for key,rs in candidate_frames.items() if key[0]=='natural' for r in rs)
        check(natural_events==0,'natural no events')
        training=load('evidence/TRAINING.json');progress=load('evidence/NUMERIC_PROGRESS.json')
        check(training['optimizer_updates']==len(training['losses'])==len(training['draws'])==96,'96 updates')
        check(progress['training_bag_forwards_completed']==672 and progress['candidate_head_cases_completed']==323,'forward budgets')
        check(progress['candidate_embedding_rows']==18982 and progress['candidate_ready_rows']==14137,'row budgets')
        draw_counts=collections.Counter(r['reviewed_source'] for r in training['draws'])
        check(len(draw_counts)==8 and set(draw_counts.values())=={12},'8 sources equal')
        train_only=[]
        for g in load('evidence/TRAINING_GEOMETRY.json')['original_bags']:
            if g['role']!='train' or not g['weak_class'] or g['condition'] not in ('clean','snr10'):
                continue
            key=(g['condition'],g['id']); target=g['weak_class']
            row={'condition':g['condition'],'id':g['id'],'weak_class':target}
            for label,table in [('old',old_frames),('new',candidate_frames)]:
                frames={r['sample']:r for r in table[key]}
                row[label+'_bag_mean']=sum(frames[t]['probabilities'][target] for t in g['head_instance_ends'])/len(g['head_instance_ends'])
                probabilities=[r['probabilities'][target] for r in table[key] if r['ready']]
                row[label+'_best_triple']=max(min(probabilities[i:i+3]) for i in range(len(probabilities)-2))
            train_only.append(row)
        train_only_summary=[]
        for target in (1,2):
            rows=[r for r in train_only if r['weak_class']==target]
            train_only_summary.append({'weak_class':target,'bags':len(rows),
                'bag_mean_decreased':sum(r['new_bag_mean']<r['old_bag_mean'] for r in rows),
                'best_triple_decreased':sum(r['new_best_triple']<r['old_best_triple'] for r in rows),
                'mean_old_best_triple':sum(r['old_best_triple'] for r in rows)/len(rows),
                'mean_new_best_triple':sum(r['new_best_triple'] for r in rows)/len(rows),
                'max_best_triple_drop':max(r['old_best_triple']-r['new_best_triple'] for r in rows)})
        output = {'schema':'independent-saved-increment-audit-v1','artifact':pin(artifact),
            'artifact_verified':True,'manifest_files_verified':189,'execution':'SAVED_BYTES_ONLY_ZERO_MODEL_CALLS',
            'npz_bin_exact':True,'normalization_unchanged':True,'candidate_frozen_before_evaluation':True,
            'numeric_execution_complete':True,'candidate_accepted':False,'reason':'FIVE_PREVIOUS_MERO_MATCHES_LOST',
            'regressed_cases':changed,'paired_regression_recomputed_exactly':True,
            'grouped_counts':[dict(condition=k[0],role=k[1],weak_class=k[2],**v) for k,v in sorted(grouped.items())],
            'reviewed_fit':{'old_source_balanced_ce':fit['old_source_balanced_ce'],'new_source_balanced_ce':fit['new_source_balanced_ce'],
                'relative_reduction':1-fit['new_source_balanced_ce']/fit['old_source_balanced_ce'],'all_eight_source_ce_decreased':all(s['new_background_ce']<s['old_background_ce'] for s in source_ce)},
            'natural32_events':{'old':0,'new':0,'independent_far_improvement':False},
            'native_parity':{'rows':len(ref),'max_logit_error':maxlogit,'max_probability_error':maxprob,'clock_ready_event_exact':True},
            'optimizer_updates':96,'training_bag_forwards':672,'source_draw_counts':dict(draw_counts),
            'train_only_saved_confidence_analysis':{'conditions':['clean','snr10'],'role':'train','bags':35,'by_class':train_only_summary,
                'no_development_or_snr5_new8_used':True},
            'mechanism_observation':'ALL_LOSSES_ARE_MERO_EVENT_SENSITIVITY; COMPLETE_SUPPORT_NEGATIVE_CE_IMPROVES; NO_EXTRA_EVENT_ERRORS',
            'causal_limit':'No no-hardnegative continued-training control or saved gradient decomposition; cannot assign sole causality to new loss weighting.',
            'candidate_not_promoted':load('evidence/RESULT.json')['current_default_model_replaced'] is False,
            'no_independent_far_frr_or_ssc305_acceptance':True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'artifact_verified': True, 'candidate_accepted': False,
                          'reason': output['reason'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
