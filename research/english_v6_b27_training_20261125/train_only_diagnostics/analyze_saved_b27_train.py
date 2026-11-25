"""Read the registered B27 TRAIN mechanism predictions from saved logits only.

The original diagnostic source is preserved as inert text. Its exact event,
CTC and summary functions and its per-observation loop are reused below.
An already completed, independently verified TRAIN report must be explicitly
hash-bound; this diagnostic does not repeat its full Q1 provenance readback.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METHOD = HERE / 'ORIGINAL_B24_METHOD.py.txt'
METHOD_SHA = 'a4c7eca1f2643c94cb940bb5856944430b790842fa9f944d5194c8f6bf4dcc45'
BASELINE = ROOT / 'research/english_v6_b26_preparation_20261125/TRAIN_ONLY_EVIDENCE.json'
BASELINE_SHA = 'f4bd7e2e525764c25d9d38bd8d6c225e6ade52eeab70638505284ddbf37f3579'
PRIOR = ROOT / 'research/english_v6_b26_preparation_20261125/FROZEN_TRAIN_ONLY_PROPOSAL.json'
PRIOR_SHA = '4f2df013204098f1014e366def6e8b0b6b983fcaba73805db1204a415f116b8c'
B27_PROPOSAL = ROOT / 'research/english_v6_b27_preparation_20261125/PROPOSAL.json'
B27_PROPOSAL_SHA = '587b9a404fa9209d10f1364826cf0c37879669f3a56545e1e6576473670a10c3'
READER = ROOT / 'automation/english_v6_b27_training_readback.py'
READER_SHA = '9b952dc1787f5b88afd038b29e881b006fde5a3b64c7829074f765f26b829195'
DRAW_SHA = '90b41a650ad5439524bef4bc6d13394fd65034e98c2c5a7862e0a0c1de78a83d'
DESIGN_SHA = '956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf'
ASSESSMENT_SHA = '8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee'
SCHEMA = 'english-v6-b27-saved-train-mechanism-readback-v1'
WINDOWS = ((1, 35), (36, 70), (71, 105), (106, 140), (141, 175), (176, 192))
KINDS = ('natural_ordinary', 'positive_retention', 'synthetic_ordinary', 'synthetic_positive')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mean(x): return float(np.mean(x)) if len(x) else None
def median(x): return float(np.median(x)) if len(x) else None
def summary(x):
    return dict(n=len(x), mean=mean(x), median=median(x), min=min(x) if x else None, max=max(x) if x else None)
def collapse(a):
    runs=[]
    for frame, token in enumerate(a):
        token=int(token)
        if runs and runs[-1][0] == token: runs[-1][2]=frame+1
        else:runs.append([token, frame, frame+1])
    nonblank=[v for v in runs if v[0]]
    events=[nonblank[i:i+3] for i in range(len(nonblank)-2) if [v[0] for v in nonblank[i:i+3]] == [1,2,3]]
    return nonblank,events
def logp(x):
    x=x.astype(np.float64)
    high=x.max(axis=1,keepdims=True)
    return x-high-np.log(np.exp(x-high).sum(axis=1,keepdims=True))
def ctc(lp, positive):
    if not positive:return float(-lp[:,0].sum())
    labels=[0,1,0,2,0,3,0]
    prev=np.full(7,-np.inf)
    prev[:2]=lp[0,:2]
    for frame in lp[1:]:
        cur=np.full(7,-np.inf)
        for j,t in enumerate(labels):
            terms=[prev[j]]
            if j:terms.append(prev[j-1])
            if j>1 and t and t != labels[j-2]:terms.append(prev[j-2])
            cur[j]=np.logaddexp.reduce(terms)+frame[t]
        prev=cur
    return -float(np.logaddexp(prev[-1],prev[-2]))
def event_summary(rr):
    out=dict(observations=len(rr),full_event_observations=sum(bool(r['full_events']) for r in rr),full_event_total=sum(r['full_events'] for r in rr),frame0_hey=sum(r['frame0_token']==1 for r in rr))
    if rr and rr[0]['kind']!='positive_retention':out.update(suffix_event_observations=sum(bool(r['suffix_events']) for r in rr),prefix_shortcut_observations=sum(r['first_event_prefix_hey'] for r in rr),silent_nonblank_frames=sum(r['silent_nonblank_frames'] for r in rr),ctc=summary([r['weighted_ctc'] for r in rr]),prefix_loss=summary([r['prefix_loss'] for r in rr]))
    return out


def rows_from_saved(index, arrays, draw):
    rows=[]
    logps={}
    for r in index['records']:
        a=arrays[r['array']]
        if a.shape != (r['frames'],4) or not np.isfinite(a).all() or sha(a.tobytes())!=r['sha256']:raise ValueError('array hash/shape '+r['array'])
        labels=a.argmax(axis=1)
        tokens, events=collapse(labels)
        item=dict(r,full_events=len(events),frame0_token=int(labels[0]),tokens=[x[0] for x in tokens],full_first_event=events[0] if events else None)
        if r['kind']!='positive_retention':
            slot=draw[r['step']-1]
            source=slot[{'synthetic_positive':'positive','synthetic_ordinary':'ordinary','natural_ordinary':'natural'}[r['kind']]]
            if source['id']!=r['id'] or r['prefix']!=slot['prefix']:raise ValueError('exact draw')
            at=r['prefix']//160-3
            blank=r['prefix']//160-8
            lp=logp(a);logps[(r['step'],r['kind'],r['gain'])]=lp
            suf_tokens,suf_events=collapse(labels[at:])
            tier=1.0 if source['tier']=='silver' else .75
            loss_ctc=ctc(lp[at:],r['kind']=='synthetic_positive')
            prefix_loss=.5*float((-lp[:blank,0]).max())
            item.update(tier=source['tier'],suffix_frames=len(a)-at,suffix_events=len(suf_events),suffix_tokens=[x[0] for x in suf_tokens],ctc=loss_ctc,weighted_ctc=loss_ctc*tier,prefix_loss=prefix_loss,base_loss=(loss_ctc*(1 if r['kind']=='synthetic_positive' else .6)+prefix_loss)*tier,
                        first_event_prefix_hey=bool(events and events[0][0][2]<=blank),silent_nonblank_frames=int((labels[:blank]!=0).sum()),known_silence=blank,
                        prefix_max_nonblank_probability=float(1-np.exp(lp[:blank,0]).min()),
                        suffix_best_hey_margin=float((a[at:,1]-np.max(a[at:][:,[0,2,3]],axis=1)).max()))
        rows.append(item)
    return rows, logps


def method_check():
    for path, expected in ((METHOD, METHOD_SHA), (BASELINE, BASELINE_SHA), (PRIOR, PRIOR_SHA),
                           (B27_PROPOSAL, B27_PROPOSAL_SHA), (READER, READER_SHA)):
        need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == expected,
             'exact previously frozen method, evidence or TRAIN reader: ' + str(path))
    original = ast.parse(METHOD.read_text())
    current = ast.parse(Path(__file__).read_text())
    original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
    current_functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
    reused = ('mean', 'median', 'summary', 'collapse', 'logp', 'ctc', 'event_summary')
    for name in reused:
        need(ast.dump(original_functions[name], include_attributes=False) ==
             ast.dump(current_functions[name], include_attributes=False), 'unchanged original helper ' + name)
    old_loop = next(node for node in original.body if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == "index['records']")
    new_loop = next(node for node in current_functions['rows_from_saved'].body if isinstance(node, ast.For))
    need(ast.dump(old_loop, include_attributes=False) == ast.dump(new_loop, include_attributes=False),
         'unchanged exact original per-observation statistical definition')
    old_loss = next(node for node in original.body if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name) and node.iter.id == 'trace')
    new_loss = next(node for node in current_functions['loss_residuals'].body if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name) and node.iter.id == 'trace')
    natural = next(node for node in old_loss.body if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'natural')
    need(isinstance(natural.value.left.left, ast.Constant) and natural.value.left.left.value == 0.08,
         'original natural reconstruction coefficient')
    natural.value.left.left = ast.Name(id='natural_weight', ctx=ast.Load())
    need(ast.dump(old_loss, include_attributes=False) == ast.dump(new_loss, include_attributes=False),
         'same original loss reconstruction except the registered natural coefficient')
    original_windows = next(node.iter for node in original.body if isinstance(node, ast.For)
                            and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(lo, hi)')
    need(tuple(ast.literal_eval(original_windows)) == WINDOWS, 'same original six descriptive step windows')
    baseline = json.loads(BASELINE.read_bytes())
    events = baseline['overall_events']
    need(events['natural_ordinary']['observations'] == events['synthetic_positive']['observations'] == 384
         and events['natural_ordinary']['full_event_total'] == 47
         and events['natural_ordinary']['full_event_observations'] == 35
         and events['synthetic_positive']['suffix_event_observations'] == 202
         and baseline['readback']['saved_arrays'] == 1844
         and baseline['readback']['optimizer_updates'] == 192, 'unchanged frozen TRAIN comparison')
    need(not {'torch', 'onnxruntime', 'tensorflow', 'jax'} & set(sys.modules), 'no acoustic or gradient runtime imported')
    return dict(schema=SCHEMA, status='INERT_AWAIT_COMPLETE_ACTUAL_TRAIN_READBACK',
                original_functions_reused=list(reused), original_observation_loop_identical=True,
                original_loss_reconstruction_preserved_except_registered_coefficient=True,
                baseline_read_only=True, new_model_calls=0, new_optimizer_updates=0, new_source_calls=0)


def loss_residuals(rows, logps, trace, margins, retention, natural_weight):
    """Original saved-output reconstruction, with the registered B27 0.16 scalar."""
    need(natural_weight == 0.16, 'only the already registered B27 natural coefficient')
    by_step=defaultdict(list)
    for r in rows:by_step[r['step']].append(r)
    residuals=defaultdict(list)
    for t in trace:
        step=t['step']; rr=by_step[step]
        synthetic=sum(r['base_loss'] for r in rr if r['kind'].startswith('synthetic_'))/4
        natural=natural_weight*sum(r['base_loss'] for r in rr if r['kind']=='natural_ordinary')/2
        kl=[]
        for kind in ['synthetic_ordinary','synthetic_positive']:
            at=t['prefix']//160-3
            lp1,lp05=[logps[(step,kind,g)][at:] for g in [1.,.5]]
            kl.append(min(max(float((np.exp(lp1)*(lp1-lp05)).sum(axis=1).mean()),0),8))
        consistency=.03*sum(kl)/2
        mm=[r for r in margins if r['step']==step]
        ceiling=mean([r['mean'] for r in mm])
        rgroup=defaultdict(list)
        for r in retention:
            if r['step']==step:rgroup[r['id']].append(r['unweighted_loss'])
        positive=.08*mean([mean(v) for v in rgroup.values()])
        expected=synthetic+natural+consistency+positive+.08*ceiling
        for key,val in [('synthetic_ctc',synthetic),('natural_ctc',natural),('gain_consistency',consistency),('positive_retention',positive),('ordinary_ceiling',ceiling),('loss',expected)]:
            residuals[key].append(abs(t[key]-val))
    return {key: max(values) for key, values in residuals.items()}


def analyze(models_path, lock_path, report_path, expected_report_sha, output):
    method_check()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new absolute diagnostic output directory')
    need(len(expected_report_sha) == 64 and all(c in '0123456789abcdef' for c in expected_report_sha),
         'explicit independently obtained complete TRAIN report SHA-256')
    for path in (models_path, lock_path, report_path):
        need(path.is_file() and not path.is_symlink(), 'regular original bound input: ' + str(path))
    report_raw = report_path.read_bytes()
    need(sha(report_raw) == expected_report_sha, 'exact already independently verified original TRAIN report')
    sys.path.insert(0, str(ROOT / 'automation'))
    reader = importlib.import_module('english_v6_b27_training_readback')
    lock = reader.document(lock_path.read_bytes())
    bound = reader.validate_lock(lock)
    verified = reader.document(report_raw)
    need(verified['schema'] == reader.REPORT_SCHEMA and verified['original_lock_sha256'] == sha(reader.canonical(lock))
         and verified['result_sha256'] == bound['result_sha256']
         and verified['producer_manifest_sha256'] == lock['training_producer_manifest_sha256']
         and all(verified[key] is True for key in ('all_original_producer_git_verified', 'all_original_inputs_verified',
             'all_saved_training_arrays_verified', 'complete_original_Q1_NO_PROMOTION_verified', 'original_B24_control_unchanged'))
         and all(verified[key] == 0 for key in ('new_model_calls', 'new_source_calls', 'new_optimizer_updates'))
         and verified['promotion_allowed'] is verified['product_released'] is False,
         'complete previously verified TRAIN qualification and saved-array report, with no promotion claim')
    models_raw = models_path.read_bytes()
    model_pin = bound['artifacts']['models']
    need(len(models_raw) == model_pin['bytes'] and sha(models_raw) == model_pin['sha256'], 'complete original B27 TRAIN models archive')
    with reader.train.failure.opened(models_raw) as models:
        result_raw, source_raw = models.read('RESULT.json'), models.read('SOURCE.json')
        result, source = reader.document(result_raw), reader.document(source_raw)
        need(result == verified['result'] and source == verified['training_source']
             and sha(result_raw) == bound['result_sha256'] and sha(source_raw) == verified['source_sha256'] == result['source_sha256']
             and result['schema'] == reader.train.SCHEMA and result['status'] == reader.train.RESULT_STATUS
             and result['natural_weight'] == 0.16 and result['actual_optimizer_updates'] == 192
             and result['actual_new_model_forwards'] == 1847 and result['model_pt_sha256'] == bound['model_pt_sha256']
             and result['design_sha256'] == result['recipe_sha256'] == DESIGN_SHA
             and result['assessment_sha256'] == ASSESSMENT_SHA
             and sha(models.read('DESIGN.json')) == DESIGN_SHA
             and sha(models.read('COMMON_DRAW.json')) == result['common_draw_sha256'] == DRAW_SHA,
             'same complete actual final B27 TRAIN and unchanged prospectively fixed recipe/draw')
        observed = reader.saved_training_arrays(models, result, source)
        need(observed == verified['saved_training_observations'], 'same complete1844 arrays and192 traces as independent original readback')
        index = reader.document(models.read('challenger/TRAIN_FORWARD_INDEX.json'))
        draw = reader.sequence(models.read('COMMON_DRAW.json'))
        trace = reader.sequence(models.read('challenger/OPTIMIZER_TRACE.json'))
        margins = reader.sequence(models.read('challenger/ORDINARY_MARGIN_TRACE.json'))
        retention = reader.sequence(models.read('challenger/POSITIVE_RETENTION_TRACE.json'))
        globals()['np'] = importlib.import_module('numpy')
        with np.load(io.BytesIO(models.read('challenger/TRAIN_FORWARD_LOGITS.npz')), allow_pickle=False) as arrays:
            rows, logps = rows_from_saved(index, arrays, draw)
        members = {name: sha(models.read(name)) for name in ('RESULT.json', 'SOURCE.json', 'DESIGN.json', 'COMMON_DRAW.json')}
        members.update({'challenger/' + name: sha(models.read('challenger/' + name)) for name in reader.OBSERVATIONS})
    overall = {kind: event_summary([r for r in rows if r['kind'] == kind]) for kind in KINDS}
    need({kind: value['observations'] for kind, value in overall.items()} ==
         dict(natural_ordinary=384, positive_retention=692, synthetic_ordinary=384, synthetic_positive=384),
         'complete identical original observation populations')
    windows = [dict(start=lo, end=hi, events={kind: event_summary([r for r in rows
               if lo <= r['step'] <= hi and r['kind'] == kind]) for kind in KINDS}) for lo, hi in WINDOWS]
    base = [overall[kind] for kind in KINDS if kind != 'positive_retention']
    silence = dict(known_silent_prefix_nonblank_frames=sum(row['silent_nonblank_frames'] for row in base),
        prefix_shortcut_observations=sum(row['prefix_shortcut_observations'] for row in base),
        all_base_frame0_hey_observations=sum(row['frame0_hey'] for row in base))
    natural, positive = overall['natural_ordinary'], overall['synthetic_positive']
    predictions = dict(
        natural_full_events=dict(observed=natural['full_event_total'], baseline=47, relation='<', supported=natural['full_event_total'] < 47),
        natural_event_bearing_observations=dict(observed=natural['full_event_observations'], baseline=35, relation='<', supported=natural['full_event_observations'] < 35),
        positive_suffix_event_observations=dict(observed=positive['suffix_event_observations'], baseline=202, relation='>=', supported=positive['suffix_event_observations'] >= 202),
        no_known_silence_shortcut_relapse=dict(observed=silence, baseline={key: 0 for key in silence},
                                             supported=all(value == 0 for value in silence.values())))
    baseline = json.loads(BASELINE.read_bytes())
    diagnostic = dict(schema=SCHEMA, scope='already_saved_pre_update_TRAIN_trajectory_only',
        method=dict(original_script_sha256=METHOD_SHA, original_public_evidence_sha256=BASELINE_SHA,
            original_preexisting_proposal_sha256=PRIOR_SHA, B27_registered_proposal_sha256=B27_PROPOSAL_SHA,
            analyzer_sha256=sha(Path(__file__).read_bytes()), training_reader_sha256=READER_SHA,
            event_helpers_and_observation_loop_identical=True, original_diagnostic_reexecuted=False,
            original_windows=[list(w) for w in WINDOWS], loss_reconstruction_natural_coefficient=0.16),
        original_training=dict(run_id=bound['run_id'], run_head=bound['run_head'], models_archive=model_pin,
            original_lock_sha256=sha(reader.canonical(lock)), independent_readback_sha256=expected_report_sha,
            independent_readback_reexecuted=False, result_sha256=bound['result_sha256'], member_sha256=members,
            model_pt_sha256=bound['model_pt_sha256']),
        saved_observations=observed, overall_events=overall, original_B24_overall_events=baseline['overall_events'],
        fixed_windows=windows, registered_predictions=predictions,
        registered_support_pattern_observed=all(value['supported'] for value in predictions.values()),
        max_loss_reconstruction_residual=loss_residuals(rows, logps, trace, margins, retention, result['natural_weight']),
        interpretation=dict(final_checkpoint_accuracy_estimated=False, independent_quality_result=False,
            promotion_gate=False, parameter_or_checkpoint_selection=False, same_job_causal_isolation=False,
            note='Repeated fixed-draw observations from changing pre-update TRAIN weights; descriptive mechanism support only. Historical B24 is not a same-job control.'),
        new_model_calls=0, new_backward_calls=0, new_optimizer_updates=0, new_source_calls=0,
        evaluation_arrays_used=False, evaluation_metrics_used_for_statistics=False,
        promotion_allowed=False, product_released=False)
    need(not {'torch', 'onnxruntime', 'tensorflow', 'jax'} & set(sys.modules), 'saved-array arithmetic only; no model runtime')
    encoded_rows = (json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    diagnostic['rows'] = dict(path='rows.json', bytes=len(encoded_rows), sha256=sha(encoded_rows))
    encoded_report = (json.dumps(diagnostic, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    output.mkdir()
    (output / 'rows.json').write_bytes(encoded_rows)
    (output / 'READBACK.json').write_bytes(encoded_report)
    return dict(path=str(output / 'READBACK.json'), bytes=len(encoded_report), sha256=sha(encoded_report),
                registered_predictions=predictions, registered_support_pattern_observed=diagnostic['registered_support_pattern_observed'],
                new_model_calls=0, new_backward_calls=0, promotion_allowed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    run = commands.add_parser('analyze')
    run.add_argument('--models', type=Path, required=True)
    run.add_argument('--training-lock', type=Path, required=True)
    run.add_argument('--training-readback', type=Path, required=True)
    run.add_argument('--readback-sha256', required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        value = method_check()
    else:
        value = analyze(args.models, args.training_lock, args.training_readback, args.readback_sha256, args.output)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
