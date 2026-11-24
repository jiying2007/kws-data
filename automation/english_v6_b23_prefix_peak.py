"""One conditional B23 TRAIN candidate: penalize the worst known-silent frame.

All B22 TRAIN data, teacher arrays, draws, optimizer and loss coefficients are
reused unchanged. B22 is an immutable historical control. Only the temporal
reduction of its existing artificial-silence blank term changes mean to amax.
Both new panels must be registered and an actual original B22 NO_PROMOTION must
be independently reconstructed before any new budget can be consumed.
"""
from __future__ import annotations
import argparse
import ast
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_union_retention_continuation as baseline
import english_v6_b23_b22_failure as failure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DESIGN = HERE / 'english_v6_b23_prefix_peak_design.json'
DESIGN_SHA = '8485e152f0ba93e22c510731dc4acd7823b36df033856b4145aa971aa96ebe56'
BINDING = HERE / 'english_v6_b23_assessment_binding.json'
OPENING = HERE / 'english_v6_b23_opening.json'
REPO = 'jiying2007/kws-data'
BRANCH = 'refs/heads/research/en-v6-b23-prefix-peak-train-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b23-prefix-peak-train-once.yml'
MARKER = 'english-v6-b23-prefix-peak-train-consumed'
SCHEMA = 'english-v6-b23-train-prefix-peak-v1'
need, sha, encoded, write, gh = baseline.need, baseline.sha, baseline.encoded, baseline.write, baseline.gh
old = baseline.old
Recorder, positive_loss, margin_statistics = baseline.Recorder, baseline.positive_loss, baseline.margin_statistics


def design():
    need(sha(DESIGN.read_bytes()) == DESIGN_SHA, 'fixed TRAIN-only prefix-maximum intervention')
    value = json.loads(DESIGN.read_bytes())
    before = baseline.design()
    retained = set(before) - {'schema', 'contrast', 'historical_control', 'b21_rerun', 'aggregation', 'scope'}
    need(all(value[key] == before[key] for key in retained), 'every B22 training recipe field remains unchanged')
    need(sha((ROOT / value['baseline_engine_path']).read_bytes()) == value['baseline_engine_sha256']
         and baseline.DESIGN_SHA == value['baseline_recipe_sha256']
         and value['opening_condition'] == 'only_after_actual_B22_NO_PROMOTION'
         and value['historical_control']['pt_sha256'] == '9b1900312430c5c92dee7f47608fc2be866778387d4ed3baedb11f4bcda6a655'
         and value['budget']['total_new_model_forwards'] == 1847
         and value['new_gradient_diagnostic_backwards'] == 0,
         'exact immutable B22 scientific parent, control and forward/update budget')
    baseline.architecture_contract()
    return value


def assessment_binding(require_frozen=True):
    value = json.loads(BINDING.read_bytes())
    need(value['schema'] == 'english-v6-b23-prospective-assessment-binding-v1'
         and value['recipe_path'] == DESIGN.relative_to(ROOT).as_posix()
         and value['recipe_sha256'] == DESIGN_SHA, 'prospective panels bind this exact prefix intervention')
    if not require_frozen and value['frozen'] is False:
        need(value['assessment'] is None and value['files'] == {} and value['optimizer_allowed'] is False,
             'inert incomplete metadata grants no execution budget')
        return value
    need(value['frozen'] is True and value['optimizer_allowed'] is False
         and type(value['files']) is dict and bool(value['files']),
         'complete metadata freeze is separate from failed-primary permission')
    for name, digest in value['files'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts
             and re.fullmatch(r'[0-9a-f]{64}', digest or ''), 'safe complete metadata resource pin')
        target = ROOT / path
        need(target.is_file() and not target.is_symlink() and sha(target.read_bytes()) == digest,
             'unchanged prospective assessment member ' + name)
    pin = value['assessment']
    need(set(pin) == {'path', 'sha256'} and value['files'].get(pin['path']) == pin['sha256'],
         'assessment is covered by complete metadata inventory')
    report = json.loads((ROOT / pin['path']).read_bytes())
    need(report['freeze_before_optimizer_and_new_source'] is True
         and report['training']['design_path'] == DESIGN.relative_to(ROOT).as_posix()
         and report['training']['design_sha256'] == DESIGN_SHA,
         'fresh independent panels are prospectively frozen')
    import english_v6_b23_source_metadata as metadata
    need(Path(metadata.__file__).resolve() == HERE / 'english_v6_b23_source_metadata.py'
         and metadata.validate()['hashes'] == value['files'], 'complete deterministic fresh metadata validation')
    return value


def opening(require_enabled=True):
    value = json.loads(OPENING.read_bytes())
    need(set(value) == {'schema', 'condition', 'enabled', 'quality_lock', 'baseline_challenger_pt_sha256'}
         and value['schema'] == 'english-v6-b23-after-original-b22-no-promotion-v1'
         and value['condition'] == 'only_after_actual_B22_NO_PROMOTION'
         and value['baseline_challenger_pt_sha256'] == design()['historical_control']['pt_sha256'],
         'specific original failed-candidate dependency; no generic approval switch')
    if not require_enabled and value['enabled'] is False:
        need(value['quality_lock'] is None, 'unopened experiment has no fabricated original result identity')
        return value
    need(value['enabled'] is True, 'B23 is sealed until actual B22 NO_PROMOTION is bound')
    failure.validate_lock(value['quality_lock'])
    return value


def guard(parent, env=None, event=None, git_parent=None):
    design(); assessment_binding(); opening()
    e = os.environ if env is None else env
    need(re.fullmatch(r'[0-9a-f]{40}', parent or ''), 'exact reviewed B23 prepared predecessor')
    need(all(e.get(k) == v for k, v in dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=REPO,
         GITHUB_EVENT_NAME='push', GITHUB_REF=BRANCH, GITHUB_RUN_ATTEMPT='1').items()),
         'only first public B23 execution on its prospectively named branch')
    head, run = e.get('GITHUB_SHA', ''), e.get('GITHUB_RUN_ID', '')
    need(re.fullmatch(r'[0-9a-f]{40}', head) and re.fullmatch(r'[1-9][0-9]*', run), 'actual public execution identity')
    if event is None:
        event = json.loads(Path(e['GITHUB_EVENT_PATH']).read_bytes())
    if git_parent is None:
        git_parent = subprocess.check_output(['git', '-C', e['GITHUB_WORKSPACE'], 'rev-parse', 'HEAD^1'], text=True).strip()
    need(event.get('before') == git_parent == parent and event.get('after') == head
         and event.get('ref') == BRANCH and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPO and event['repository'].get('private') is False,
         'single workflow-only reviewed public activation; no force or retry')
    return dict(run_id=run, run_head=head, run_attempt=1, prepared_parent=parent, run_branch=BRANCH)


def previous_failure(inputs):
    bound = opening()
    report = failure.verify_failure_original(Path(inputs) / 'no-promotion', bound['quality_lock'])
    need(report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['model_pt_sha256']['challenger'] == bound['baseline_challenger_pt_sha256']
         and report['all_original_arrays_verified'] is report['original_source_admission_recomputed']
             is report['original_producer_git_verified'] is True
         and report['promotion_allowed'] is False, 'actual complete B22 failure with independent original-source/tensor readback')
    return report


def historical_control(inputs):
    path = Path(inputs) / 'control/B22.zip'
    pin = design()['historical_control']
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes']
         and sha(path.read_bytes()) == pin['sha256'], 'original B22 control ZIP unchanged')
    with zipfile.ZipFile(path) as archive:
        need(archive.testzip() is None, 'complete original B22 control archive')
        receipt = json.loads(archive.read('RESULT.json'))
        need(receipt['schema'] == 'english-v6-b22-train-union-retention-v1'
             and receipt['identity']['run_id'] == str(pin['run_id'])
             and receipt['identity']['run_head'] == pin['run_head']
             and receipt['actual_optimizer_updates'] == 192
             and receipt['actual_new_model_forwards'] == 1847
             and receipt['model_pt_sha256']['challenger'] == pin['pt_sha256']
             and receipt['challenger']['stem'] == pin['stem'], 'same original trained B22 historical control')
        expected = {'mero43-b22-challenger192' + ext for ext in ('.pt', '.npz', '.onnx')}
        need(set(receipt['challenger']['files']) == expected, 'all three original control formats')
        for name, file_pin in receipt['challenger']['files'].items():
            raw = archive.read('challenger/models/' + name)
            need(len(raw) == file_pin['bytes'] and sha(raw) == file_pin['sha256'], 'unaltered original B22 control ' + name)
    return receipt['challenger']


def source_preflight(inputs):
    inputs = Path(inputs)
    proof = previous_failure(inputs)
    control = historical_control(inputs)
    report, *values = baseline.source_preflight(inputs / 'training')
    value = dict(report, schema='english-v6-b23-authenticated-train-source-v1',
        original_b22_training_source_sha256=sha(encoded(report)), previous_failure=proof,
        original_control=control, all_original_train_sources_and_teachers_unchanged=True,
        original_retention_conditions=120, union_retention_conditions=122,
        new_model_forwards=0, new_source_calls=0, new_dev_model_forwards=0,
        saved_B22_dev_readback_for_authorization=True, dev_rows_used_for_training=0,
        prospective_assessment_sha256=assessment_binding()['assessment']['sha256'])
    value.pop('new_dev_reads')
    return value, *values


def fetch(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), 'fresh immutable B23 input root')
    inputs.mkdir()
    proof = failure.fetch_failure_original(inputs / 'no-promotion', opening()['quality_lock'])
    need(proof['model_pt_sha256']['challenger'] == design()['historical_control']['pt_sha256'], 'failure belongs to original B22 control')
    receipts = {}
    for subdir, pins in (('training', design()['input_archives']), ('control', {'B22.zip': design()['historical_control']})):
        target = inputs / subdir
        target.mkdir()
        for name, pin in pins.items():
            artifact = json.loads(gh('actions/artifacts/' + str(pin['artifact_id'])))
            need(artifact['id'] == pin['artifact_id'] and artifact['expired'] is False
                 and artifact['digest'] == 'sha256:' + pin['sha256'], 'live original input artifact ' + name)
            raw = gh('actions/artifacts/' + str(pin['artifact_id']) + '/zip')
            need(sha(raw) == pin['sha256'] and len(raw) == artifact['size_in_bytes'], 'exact original input bytes ' + name)
            (target / name).write_bytes(raw)
            receipts[subdir + '/' + name] = artifact
    historical_control(inputs)
    write(inputs / 'FETCH.json', dict(identity=identity, artifacts=receipts, recipe_sha256=DESIGN_SHA,
        assessment_binding_sha256=sha(BINDING.read_bytes()), opening_sha256=sha(OPENING.read_bytes()), previous_failure=proof,
        new_model_forwards=0))


def consume(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    report, *_ = source_preflight(inputs)
    marker = inputs / 'CONSUMPTION.json'
    need(not marker.exists(), 'one B23 budget marker; original B22 markers remain untouched')
    prior = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(prior['total_count'] == 0 and prior['artifacts'] == [], 'B23 budget has never been consumed')
    write(inputs / 'SOURCE_PREFLIGHT.json', report)
    value = dict(schema=SCHEMA, status='TRAINING_BUDGET_CONSUMED_UNKNOWN', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        opening_sha256=sha(OPENING.read_bytes()), source_preflight_sha256=sha((inputs / 'SOURCE_PREFLIGHT.json').read_bytes()),
        budget=design()['budget'], promotion_allowed=False)
    write(marker, value)
    return value


def verify_consumed(inputs, parent):
    identity = guard(parent)
    local = json.loads((inputs / 'CONSUMPTION.json').read_bytes())
    need(local['identity'] == identity and local['budget'] == design()['budget']
         and local['recipe_sha256'] == DESIGN_SHA and local['assessment_binding_sha256'] == sha(BINDING.read_bytes())
         and local['opening_sha256'] == sha(OPENING.read_bytes())
         and local['source_preflight_sha256'] == sha((inputs / 'SOURCE_PREFLIGHT.json').read_bytes()),
         'same frozen input receipt and preoptimizer budget')
    runs = json.loads(gh('actions/workflows/' + Path(WORKFLOW).name + '/runs?branch=' + BRANCH.removeprefix('refs/heads/') + '&per_page=100'))
    need(runs['total_count'] == len(runs['workflow_runs']) == 1
         and str(runs['workflow_runs'][0]['id']) == identity['run_id']
         and runs['workflow_runs'][0]['head_sha'] == identity['run_head']
         and runs['workflow_runs'][0]['run_attempt'] == 1, 'exactly one first B23 training execution')
    items = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(items['total_count'] == len(items['artifacts']) == 1, 'one durably uploaded B23 budget')
    item = items['artifacts'][0]
    need(item['expired'] is False and item['workflow_run']['id'] == int(identity['run_id'])
         and item['workflow_run']['head_sha'] == identity['run_head'], 'same first execution owns the marker')
    raw = gh('actions/artifacts/' + str(item['id']) + '/zip')
    need(sha(raw) == item['digest'].removeprefix('sha256:') and len(raw) == item['size_in_bytes'], 'original consumed artifact bytes')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [v.filename for v in archive.infolist() if not v.is_dir()]
        need(names == ['CONSUMPTION.json'] and json.loads(archive.read(names[0])) == local,
             'uploaded exact marker precedes first optimization')
    return identity


def prefix_peak_suffix_loss(model,x,klass,prefix,tier,torch,F,return_logp=False):
    # Exact B4/B5-control CTC suffix and blank-prefix loss; no frame pseudolabels.
    logit=model(x).log_softmax(-1)
    start=prefix//160-3
    blank=prefix//160-8
    need(0<blank<start<logit.shape[1]-5,
         "IMMUTABLE_B3_SUFFIX_CTC_START")
    tail=logit[:,start:,:]
    labels=torch.tensor([1,2,3] if klass else [],dtype=torch.long)
    ctc=F.ctc_loss(tail.transpose(0,1),labels,
                   torch.tensor([tail.shape[1]],dtype=torch.long),
                   torch.tensor([len(labels)],dtype=torch.long),
                   blank=0,reduction="none",zero_infinity=False)[0]
    silent=logit[:,:blank,:]
    blank_margin=torch.logsumexp(silent[...,1:],dim=-1)-silent[...,0]
    loss=ctc*(1.0 if klass else 0.6)+0.5*F.softplus(blank_margin).amax()
    weighted=loss*(1.0 if tier=="silver" else 0.75)
    return (weighted,logit) if return_logp else weighted

def fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    torch.manual_seed(old.SEED)
    net = old.b10.b6.b4.v5.build_model(torch)
    net.load_state_dict(initial, strict=True)
    net.train()
    need(not any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for m in net.modules()),
         'no forward RNG or running statistics in the fixed model')
    optimizer = torch.optim.AdamW(net.parameters(), lr=old.LR, weight_decay=.001)
    recorder, trace, positive_trace, ordinary_trace = Recorder(np), [], [], []
    handle = net.register_forward_hook(recorder.hook)
    try:
        for step, slot in enumerate(slots):
            optimizer.zero_grad(set_to_none=True)
            prefix, losses, consistency, ordinary = slot['prefix'], [], [], []
            for klass, kind in ((0, 'ordinary'), (1, 'positive')):
                row, pair = slot[kind], {}
                for gain in old.b10.GAINS:
                    recorder.before(step=step + 1, kind='synthetic_' + kind, id=row['id'], gain=gain, prefix=prefix)
                    base, logp = prefix_peak_suffix_loss(net, base_cache[row['id']][gain][prefix], klass,
                                                     prefix, row['tier'], torch, F, True)
                    losses.append(base)
                    pair[gain] = logp
                    if klass == 0:
                        ordinary.append((row, gain, prefix, logp))
                at = prefix // 160 - 3
                kl = F.kl_div(pair[.5][:, at:, :], pair[1.][:, at:, :].detach(),
                              reduction='none', log_target=True).sum(-1).mean()
                need(bool(torch.isfinite(kl)), 'finite unchanged gain consistency')
                consistency.append(kl.clamp(min=0., max=old.b10.b6.KL_CLIP))
            natural, row = [], slot['natural']
            for gain in old.b10.GAINS:
                recorder.before(step=step + 1, kind='natural_ordinary', id=row['id'], gain=gain, prefix=prefix)
                loss, logp = prefix_peak_suffix_loss(net, natural_cache[row['id']][gain][prefix], 0,
                                                 prefix, row['tier'], torch, F, True)
                natural.append(loss)
                ordinary.append((row, gain, prefix, logp))
            positive = positive_loss(step, net, positive_paths, recorder, positive_trace, torch)
            ceiling = old.b17.ceiling_loss(ordinary, teachers, torch)
            loss = (sum(losses) / 4 + old.b10.b6.WEIGHT * sum(consistency) / 2
                    + old.NATURAL_WEIGHT * sum(natural) / 2 + positive + old.CEILING_WEIGHT * ceiling)
            need(bool(torch.isfinite(loss)), 'finite unchanged base objective plus union retention')
            for row, gain, prefix, logp in ordinary:
                ordinary_trace.append(dict(step=step + 1, id=row['id'], group=row['source_group_id'], gain=gain,
                    prefix=prefix, **margin_statistics(logp[0], teachers[(row['id'], gain, prefix)], torch)))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            trace.append(dict(step=step + 1, prefix=prefix, ordinary_id=slot['ordinary']['id'],
                positive_id=slot['positive']['id'], natural_id=slot['natural']['id'], loss=float(loss.detach()),
                synthetic_ctc=float((sum(losses) / 4).detach()), gain_consistency=float((old.b10.b6.WEIGHT * sum(consistency) / 2).detach()),
                natural_ctc=float((old.NATURAL_WEIGHT * sum(natural) / 2).detach()), positive_retention=float(positive.detach()),
                ordinary_ceiling=float(ceiling.detach()), ceiling_coefficient=old.CEILING_WEIGHT,
                preclip_gradient_norm=float(grad_norm), gradient_clipped=bool(grad_norm > 5.)))
    finally:
        handle.remove()
        recorder.save(output)
        write(output / 'OPTIMIZER_TRACE.json', trace)
        write(output / 'POSITIVE_RETENTION_TRACE.json', positive_trace)
        write(output / 'ORDINARY_MARGIN_TRACE.json', ordinary_trace)
    need(len(trace) == 192 and len(recorder.rows) == 1844 and len(ordinary_trace) == 768 and len(positive_trace) == 692,
         'one exact 192-update challenger; all existing training forwards and gradient statistics saved')
    net.eval()
    model_dir = output / 'models'
    model_dir.mkdir()
    stem = 'mero43-b23-challenger192'
    pt = model_dir / (stem + '.pt')
    torch.save(net.state_dict(), pt)
    state = torch.load(pt, map_location='cpu', weights_only=True)
    rebuilt = old.b10.b6.b4.v5.build_model(torch)
    rebuilt.load_state_dict(state, strict=True)
    rebuilt.eval()
    need(sum(v.numel() for v in state.values()) == 43876 and all(torch.isfinite(v).all() for v in state.values())
         and all(torch.equal(v, net.state_dict()[k]) for k, v in state.items()), 'exact finite full checkpoint reload')
    mel, mean, scale = frontend
    npz = model_dir / (stem + '.npz')
    np.savez(npz, mel_filters=mel, train_mean=mean, train_scale=scale, **{k: v.cpu().numpy() for k, v in state.items()})
    example = base_cache[slots[0]['ordinary']['id']][1.][slots[0]['prefix']]
    with torch.inference_mode():
        expected, observed = net(example), rebuilt(example)
    need(torch.equal(expected, observed), 'two real checkpoint reload verification forwards')
    np.savez_compressed(output / 'CHECKPOINT_RELOAD_LOGITS.npz', saved=expected[0].cpu().numpy(), reloaded=observed[0].cpu().numpy())
    with np.load(npz, allow_pickle=False) as encoded_state:
        need(all(np.array_equal(encoded_state[k], v.cpu().numpy()) for k, v in state.items()), 'exact PT/NPZ tensors')
    onnx = model_dir / (stem + '.onnx')
    count = [0]
    hook = net.register_forward_hook(lambda *_: count.__setitem__(0, count[0] + 1))
    try:
        torch.onnx.export(net, example, onnx.as_posix(), opset_version=17, dynamo=False,
                          input_names=['fbank40'], output_names=['frame_ctc_logits'],
                          dynamic_axes={'fbank40': {1: 'frames'}, 'frame_ctc_logits': {1: 'frames'}})
    finally:
        hook.remove()
    need(count[0] == 1, 'one bounded ONNX export forward')
    import onnx as api
    api.checker.check_model(api.load(onnx))
    return dict(stem='challenger/models/' + stem, optimizer_updates=192,
        parent_pt_sha256=old.B0_PT, optimization_forwards=1844, checkpoint_reload_forwards=2,
        onnx_export_forwards=1, actual_new_model_forwards=1847,
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in (pt, npz, onnx)})

def execute(inputs, output, parent):
    inputs, output = Path(inputs), Path(output)
    identity = verify_consumed(inputs, parent)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'one fresh output; no training retry')
    report, rows, waves, groups, extra, extra_waves, slots = source_preflight(inputs)
    need(sha(encoded(report)) == json.loads((inputs / 'CONSUMPTION.json').read_bytes())['source_preflight_sha256'],
         'same complete source proof as the durable consumption marker')
    output.mkdir()
    write(output / 'STARTED.json', dict(schema=SCHEMA, identity=identity, budget=design()['budget'], recipe_sha256=DESIGN_SHA))
    write(output / 'SOURCE.json', report)
    write(output / 'COMMON_DRAW.json', slots)
    (output / 'DESIGN.json').write_bytes(DESIGN.read_bytes())
    (output / 'ASSESSMENT_BINDING.json').write_bytes(BINDING.read_bytes())
    (output / 'OPENING.json').write_bytes(OPENING.read_bytes())
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    b0net, _unused_b9, frontend = old.b17.load_parents(inputs / 'training', torch, np)
    base_cache = old.b13.paired.make_features(rows, waves, frontend, torch, np)
    natural_cache = old.b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    paths, old_audit = baseline.prepare_union_paths(inputs / 'training', rows, waves, frontend, np, torch)
    teachers = baseline.saved_ordinary_teachers(inputs / 'training', torch, np)
    write(output / 'UNION_EVIDENCE.json', report['union'])
    write(output / 'OLD120_POSITIVE_REFERENCES.json', old_audit)
    control_dir = output / 'control/models'
    control_dir.mkdir(parents=True)
    control_files = {}
    with zipfile.ZipFile(inputs / 'control/B22.zip') as archive:
        receipt = json.loads(archive.read('RESULT.json'))['challenger']
        for name, pin in receipt['files'].items():
            raw = archive.read('challenger/models/' + name)
            need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'unchanged historical control export')
            (control_dir / name).write_bytes(raw)
            control_files[name] = pin
    target = output / 'challenger'
    target.mkdir()
    candidate = fit({k: v.detach().clone() for k, v in b0net.state_dict().items()}, slots, base_cache, natural_cache,
                    paths, teachers, frontend, target, torch, np, F)
    result = dict(schema=SCHEMA, status='TRAINED_PREFIX_PEAK_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        design_sha256=DESIGN_SHA, assessment_sha256=assessment_binding()['assessment']['sha256'],
        source_sha256=sha((output / 'SOURCE.json').read_bytes()),
        prefix_blank_reduction='amax', original_B22_NO_PROMOTION_reauthenticated=True,
        opening_sha256=sha(OPENING.read_bytes()),
        common_draw_sha256=sha((output / 'COMMON_DRAW.json').read_bytes()),
        union_evidence_sha256=sha((output / 'UNION_EVIDENCE.json').read_bytes()),
        old120_references_sha256=sha((output / 'OLD120_POSITIVE_REFERENCES.json').read_bytes()),
        model_pt_sha256=dict(B0=old.B0_PT, B9=old.b10.b9.MODELS['B9_control']['pt'],
                            control=design()['historical_control']['pt_sha256'],
                            challenger=candidate['files']['mero43-b23-challenger192.pt']['sha256']),
        control=dict(**design()['historical_control'], output_stem='control/models/mero43-b22-challenger192', files=control_files),
        challenger=candidate, actual_optimizer_updates=192, actual_new_model_forwards=1847,
        new_control_optimizer_updates=0, new_control_forwards=0, new_teacher_forwards=0,
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        dev_cal_locked_forwards=0, new_tts_asr_calls=0, historical_numerical_replay_claimed=False,
        same_job_matched_control=False, architecture=baseline.architecture_contract(), promotion_allowed=False,
        physical_ssc305=False, field_far_per_hour=None)
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('check')
    for mode in ('fetch', 'consume', 'execute', 'preflight'):
        item = sub.add_parser(mode)
        item.add_argument('--inputs', type=Path, required=True)
        if mode != 'preflight': item.add_argument('--prepared-parent', required=True)
        if mode == 'execute': item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'check':
        design(); assessment_binding(False); opening(False)
        print('PASS: B23 prefix-maximum preparation; only actual B22 NO_PROMOTION can open execution')
    elif args.mode == 'fetch': fetch(args.inputs, args.prepared_parent)
    elif args.mode == 'consume': print(json.dumps(consume(args.inputs, args.prepared_parent), sort_keys=True))
    elif args.mode == 'preflight': print(json.dumps(source_preflight(args.inputs)[0], sort_keys=True))
    else: print(json.dumps(execute(args.inputs, args.output, args.prepared_parent), sort_keys=True))
