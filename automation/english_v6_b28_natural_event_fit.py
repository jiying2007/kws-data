"""B28 TRAIN-only fit: replace only the natural suffix objective.

The two mathematical files are byte-for-byte copies of the previously frozen
source; importing this module does not import Torch or run any model. This
module is not an execution permit. A separately reviewed B28 TRAIN authority
must authenticate the original inputs, frozen design and once-only budget.

The original B23 fit is retained as an explicit auditable loop, with only its
natural_ordinary loss call changed and forward-free observation records added.
The original synthetic losses, retention, teachers, updates, checkpoint checks
and export remain fixed. Only the new suffix NLL computes in float64; original
raw returns and shared logp remain FP32. Its mixed scalar contribution promotes
the summed training loss, without converting the old computational branches.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import sys
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OBJECTIVE_SCHEMA = 'english-v6-b28-natural-suffix-no-event-v1'
NATURAL_TRACE_SCHEMA = 'english-v6-b28-natural-no-event-trace-v1'
OPTIMIZER_SCHEMA = 'english-v6-b28-optimizer-trace-v1'
OBJECTIVE_SHA = '1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a'
REFERENCE_SHA = '034a4827fb9ff3091a5e23ea25e2eab0bc23ee7263bce644fd52741e6b80f969'
COMMON_DRAW_SHA = '90b41a650ad5439524bef4bc6d13394fd65034e98c2c5a7862e0a0c1de78a83d'
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
NATURAL_WEIGHT = 0.16
STEM = 'mero43-b28-challenger192'
DEPENDENCIES = {
    'english_v6_b23_prefix_peak.py': '4c61ffafb00fa75478424179b80d004eb4370e8050b1841caa47c51c669ccbae',
    'english_v6_b24_b23_warm_start.py': '6259ae829a32655822350caee2608c9d162fdadc6366a3fa6ecf3fb6ab452cd7',
    'english_v6_b22_union_retention_continuation.py': '5afb83c3c0b840135c1d6e8bb0f97b2b4bb0988c3e4fdb87f1d6c1e347c2001a',
    'english_v6_b17_b0_margin_pair.py': '7473e468ce55e96718b6599c7b9187377710258238b330758bec79c9d96e8df3',
    'english_v6_b28_no_event_objective.py': OBJECTIVE_SHA,
    'english_v6_b28_no_event_reference.py': REFERENCE_SHA,
}
for _name, _digest in DEPENDENCIES.items():
    _path = HERE / _name
    if _path.is_symlink() or hashlib.sha256(_path.read_bytes()).hexdigest() != _digest:
        raise ValueError('exact frozen scientific dependency before import: ' + _name)
sys.path.insert(0, str(HERE))
import english_v6_b24_b23_warm_start as b24
import english_v6_b28_no_event_objective as objective

need, sha, encoded, write = b24.need, b24.sha, b24.encoded, b24.write
old = SimpleNamespace(**vars(b24.old))
old.NATURAL_WEIGHT = NATURAL_WEIGHT
Recorder, positive_loss, margin_statistics = b24.Recorder, b24.positive_loss, b24.margin_statistics
prefix_peak_suffix_loss = b24.prefix_peak_suffix_loss
natural_no_event_suffix_loss = objective.natural_no_event_suffix_loss


def contract():
    """Read source/constant identities only; this grants no execution authority."""
    for name, digest in DEPENDENCIES.items():
        path = HERE / name
        need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == digest,
             'unchanged scientific dependency ' + name)
    need(old.NATURAL_WEIGHT == NATURAL_WEIGHT == 0.16 and b24.old.NATURAL_WEIGHT == 0.08
         and old.SEED == 20261120 and old.LR == 0.00006 and old.CEILING_WEIGHT == 0.08
         and old.b10.GAINS == (1.0, 0.5) and old.b10.PREFIXES == (6400, 12800, 19200)
         and old.b10.b6.WEIGHT == 0.03 and old.b10.b6.KL_CLIP == 8
         and b24.INITIAL_PT == INITIAL_PT,
         'one private natural weight with every original common training constant unchanged')
    need(Recorder is b24.b23.Recorder and positive_loss is b24.b23.positive_loss
         and margin_statistics is b24.b23.margin_statistics
         and prefix_peak_suffix_loss is b24.b23.prefix_peak_suffix_loss
         and natural_no_event_suffix_loss is objective.natural_no_event_suffix_loss,
         'unchanged shared scientific helpers and one explicit natural objective')
    return dict(objective_schema=OBJECTIVE_SCHEMA, objective_sha256=OBJECTIVE_SHA,
                reference_sha256=REFERENCE_SHA, fit_adapter_sha256=sha(Path(__file__).read_bytes()),
                common_draw_sha256=COMMON_DRAW_SHA)


def natural_observation(step, row, gain, prefix, returned, weighted_loss):
    """Link an already returned raw array to its existing loss; no recomputation."""
    need(type(step) is int and 1 <= step <= 192 and gain in (1.0, 0.5)
         and type(prefix) is int and prefix in (6400, 12800, 19200),
         'one existing natural schedule slot')
    need(set(returned) == {'step', 'kind', 'id', 'gain', 'prefix', 'array', 'frames', 'sha256'}
         and returned['step'] == step and returned['kind'] == 'natural_ordinary'
         and returned['id'] == row['id'] and returned['gain'] == gain and returned['prefix'] == prefix,
         'natural scalar links the immediately preceding exact scheduled raw return')
    need(type(returned['frames']) is int and re.fullmatch(r'forward_[0-9]{5}', returned['array'])
         and re.fullmatch(r'[0-9a-f]{64}', returned['sha256'])
         and type(weighted_loss) is float and math.isfinite(weighted_loss) and weighted_loss >= 0.0,
         'finite weighted loss and complete recorded raw-logit identity')
    start, blank = prefix // 160 - 3, prefix // 160 - 8
    need(0 < blank < start < returned['frames'] - 5, 'unchanged suffix and prefix grid')
    return dict(step=step, id=row['id'], group=row['source_group_id'], gain=gain, prefix=prefix,
                tier=row['tier'], array=returned['array'], raw_logits_sha256=returned['sha256'],
                frames=returned['frames'], suffix_start=start, prefix_blank_end=blank,
                weighted_loss=weighted_loss)


def natural_trace_document(records):
    """Retain also a partial execution as observations, never as a completed fit."""
    return dict(schema=NATURAL_TRACE_SCHEMA, objective_schema=OBJECTIVE_SCHEMA,
                objective_sha256=OBJECTIVE_SHA, reference_sha256=REFERENCE_SHA,
                common_draw_sha256=COMMON_DRAW_SHA, count=len(records), records=records)


def fit_loop(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    contract()
    need(sha(encoded(slots)) == COMMON_DRAW_SHA, 'same original complete 192 COMMON_DRAW before any model')
    natural_trace = []
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
                loss, logp = natural_no_event_suffix_loss(net, natural_cache[row['id']][gain][prefix],
                                                        prefix, row['tier'], torch, F, True)
                natural_trace.append(natural_observation(step + 1, row, gain, prefix, recorder.rows[-1],
                                                          float(loss.detach())))
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
                natural_no_event=float((old.NATURAL_WEIGHT * sum(natural) / 2).detach()), positive_retention=float(positive.detach()),
                ordinary_ceiling=float(ceiling.detach()), ceiling_coefficient=old.CEILING_WEIGHT,
                preclip_gradient_norm=float(grad_norm), gradient_clipped=bool(grad_norm > 5.)))
    finally:
        handle.remove()
        recorder.save(output)
        write(output / 'OPTIMIZER_TRACE.json', dict(schema=OPTIMIZER_SCHEMA, count=len(trace), records=trace))
        write(output / 'NATURAL_OBJECTIVE_TRACE.json', natural_trace_document(natural_trace))
        write(output / 'POSITIVE_RETENTION_TRACE.json', positive_trace)
        write(output / 'ORDINARY_MARGIN_TRACE.json', ordinary_trace)
    need(len(trace) == 192 and len(recorder.rows) == 1844 and len(ordinary_trace) == 768 and len(positive_trace) == 692,
         'one exact 192-update challenger; all existing training forwards and gradient statistics saved')
    need(len(natural_trace) == 384, 'all original natural forwards have linked objective observations')
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


def fit_engine():
    contract()
    return fit_loop


def fit(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F):
    """Preserve the original export wrapper and correct the actual B23 parent."""
    original = b24.fit
    namespace = dict(original.__globals__, b23=SimpleNamespace(fit=fit_engine()), STEM=STEM)
    wrapper = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    value = wrapper(initial, slots, base_cache, natural_cache, positive_paths, teachers, frontend, output, torch, np, F)
    return dict(value, natural_weight=NATURAL_WEIGHT, historical_control_pt_sha256=CONTROL_PT,
                **contract(), natural_objective_returns=384,
                original_raw_logits_dtype='float32', natural_suffix_objective_dtype='float64',
                natural_scalar_promotes_total_loss=True, old_B24_optimizer_updates_reexecuted=0,
                new_teacher_forwards=0, new_control_forwards=0)
