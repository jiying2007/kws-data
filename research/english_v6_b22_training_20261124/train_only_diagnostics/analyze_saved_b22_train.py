"""Read only original B22 TRAIN outputs; no acoustic model or framework import."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'continuation-models.zip'
EXPECTED_SHA = 'b5b81de69cdd80c102cdfcc8aa83dea3e685a22e99fb5dbff7613ead35fbda3b'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), total=float(a.sum()),
                min=float(a.min()), q25=float(np.quantile(a, .25)),
                median=float(np.median(a)), q75=float(np.quantile(a, .75)),
                q90=float(np.quantile(a, .90)), q95=float(np.quantile(a, .95)),
                max=float(a.max()))


def lse(a):
    return np.logaddexp.reduce(np.asarray(a, dtype=np.float64))


def ctc(logp, positive):
    """Exact fixed [1,2,3]/empty-target CTC likelihood, output-space derivative."""
    prob = np.exp(logp)
    if not positive:
        grad = prob.copy()
        grad[:, 0] -= 1
        return -float(logp[:, 0].sum()), grad
    labels = (0, 1, 0, 2, 0, 3, 0)
    n, states = len(logp), len(labels)
    alpha = np.full((n, states), -np.inf)
    alpha[0, :2] = logp[0, :2]
    for t in range(1, n):
        for s in range(states):
            terms = [alpha[t - 1, s]]
            if s:
                terms.append(alpha[t - 1, s - 1])
            if s > 1 and labels[s] and labels[s] != labels[s - 2]:
                terms.append(alpha[t - 1, s - 2])
            alpha[t, s] = logp[t, labels[s]] + lse(terms)
    logz = lse(alpha[-1, -2:])
    beta = np.full((n, states), -np.inf)
    beta[-1, -2:] = 0
    for t in range(n - 2, -1, -1):
        for s in range(states):
            successors = [s]
            if s + 1 < states:
                successors.append(s + 1)
            if s + 2 < states and labels[s + 2] and labels[s + 2] != labels[s]:
                successors.append(s + 2)
            beta[t, s] = lse([logp[t + 1, labels[k]] + beta[t + 1, k] for k in successors])
    posterior = np.zeros_like(logp)
    for s, label in enumerate(labels):
        posterior[:, label] += np.exp(alpha[:, s] + beta[:, s] - logz)
    assert np.max(np.abs(posterior.sum(axis=1) - 1)) < 1e-9
    return -float(logz), prob - posterior


def greedy(matrix):
    labels = matrix.argmax(axis=1)
    points = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    runs = [[int(labels[left]), int(left), int(right)]
            for left, right in zip(points[:-1], points[1:]) if labels[left]]
    events = [runs[i:i + 3] for i in range(len(runs) - 2)
              if [r[0] for r in runs[i:i + 3]] == [1, 2, 3]]
    return dict(events=len(events), first_event=events[0] if events else [],
                runs=runs, blank_argmax_fraction=float(np.mean(labels == 0)))


def margin_spans(matrix, spans):
    out = []
    for token, start, end in spans:
        part = matrix[start:end]
        other = [x for x in range(4) if x != token]
        competitors = part[:, other]
        maximum = competitors.max(axis=1)
        margin = part[:, token] - maximum
        winner = np.argmax(part, axis=1)
        out.append(dict(token=token, start=start, end=end,
                        student_margin_mean=float(margin.mean()),
                        student_margin_min=float(margin.min()),
                        student_margin_max=float(margin.max()),
                        target_winning_frames=int(np.sum(winner == token)),
                        blank_winning_frames=int(np.sum(winner == 0)),
                        all_competitor_winner_counts={str(k): int(np.sum(winner == k)) for k in range(4)}))
    return out


raw = ARCHIVE.read_bytes()
assert digest(raw) == EXPECTED_SHA
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    assert z.testzip() is None
    wanted = ['DESIGN.json', 'SOURCE.json', 'COMMON_DRAW.json',
              'challenger/OPTIMIZER_TRACE.json', 'challenger/ORDINARY_MARGIN_TRACE.json',
              'challenger/POSITIVE_RETENTION_TRACE.json', 'challenger/TRAIN_FORWARD_INDEX.json',
              'challenger/TRAIN_FORWARD_LOGITS.npz']
    files = {name: z.read(name) for name in wanted}

design, source, draws, optim, ordinary, positive, index = [json.loads(files[n]) for n in wanted[:-1]]
assert len(optim) == len(draws) == 192 and len(ordinary) == 768 and len(positive) == 692
assert index['count'] == len(index['records']) == 1844
assert index['arrays_sha256'] == digest(files[wanted[-1]]) and index['pending_unreturned_forward'] is None
assert all(r['role'] == 'train' for r in source['train_originals'])
assert all(d[k]['role'] == 'train' for d in draws for k in ('ordinary', 'positive', 'natural'))

trace_ret = {(r['step'], r['id'], float(r['gain'])): r for r in positive}
trace_ord = {(r['step'], r['id'], float(r['gain']), r['prefix']): r for r in ordinary}
spans = {(r['id'], float(r['gain'])): r['b9_registered_spans'] for r in source['positive_retention']}
spans.update({(r['id'], float(r['gain'])): r['teacher_spans'] for r in source['retention_additions']})
by_step = defaultdict(list)
retention_records, ctc_records, all_records = [], [], []
logps = {}

with np.load(io.BytesIO(files[wanted[-1]]), allow_pickle=False) as arrays:
    assert len(arrays.files) == 1844 and set(arrays.files) == {r['array'] for r in index['records']}
    for rec in index['records']:
        stored = arrays[rec['array']]
        assert stored.shape == (rec['frames'], 4) and np.isfinite(stored).all()
        assert digest(stored.tobytes()) == rec['sha256']
        matrix = stored.astype(np.float64)
        out = {**rec, **greedy(matrix)}
        logp = matrix - np.logaddexp.reduce(matrix, axis=1, keepdims=True)
        out['mean_blank_posterior'] = float(np.exp(logp[:, 0]).mean())
        if rec['kind'] == 'positive_retention':
            trace = trace_ret[(rec['step'], rec['id'], rec['gain'])]
            out['unweighted_loss'] = trace['unweighted_loss']
            out['teacher_spans'] = margin_spans(matrix, spans[(rec['id'], rec['gain'])])
            retention_records.append(out)
        else:
            slot = draws[rec['step'] - 1]
            positive_class = rec['kind'] == 'synthetic_positive'
            key = 'positive' if positive_class else ('natural' if rec['kind'] == 'natural_ordinary' else 'ordinary')
            row = slot[key]
            assert rec['id'] == row['id'] and rec['prefix'] == slot['prefix']
            tier = 1 if row['tier'] == 'silver' else .75
            start, blank = rec['prefix'] // 160 - 3, rec['prefix'] // 160 - 8
            nll, derivative = ctc(logp[start:], positive_class)
            ctc_factor = 1 if positive_class else .6
            outer = .04 if key == 'natural' else .25
            blank_loss = -.5 * float(logp[:blank, 0].mean())
            weighted_ctc = outer * tier * ctc_factor * nll
            weighted_prefix = outer * tier * blank_loss
            out.update(tier=row['tier'], suffix_start=start, blank_prefix_end=blank,
                       suffix_frames=len(logp) - start, ctc_nll=nll,
                       diagnostic_ctc_nll_per_suffix_frame=nll / (len(logp) - start),
                       suffix_events=greedy(matrix[start:])['events'],
                       weighted_ctc=weighted_ctc, weighted_prefix=weighted_prefix,
                       weighted_base_total=weighted_ctc + weighted_prefix,
                       weighted_ctc_output_gradient_l1=float(np.abs(derivative).sum()) * outer * tier * ctc_factor,
                       weighted_prefix_output_gradient_l1=outer * tier * float(np.mean(1 - np.exp(logp[:blank, 0]))))
            if not positive_class:
                o = trace_ord[(rec['step'], rec['id'], rec['gain'], rec['prefix'])]
                out.update(ordinary_margin_mean=o['mean'], ordinary_margin_max=o['maximum'],
                           ordinary_active_fraction=o['active_fraction'],
                           weighted_ceiling=.02 * o['mean'],
                           weighted_ceiling_output_gradient_l1=.04 * o['active_cells'] / o['cells'])
            logps[(rec['step'], rec['kind'], rec['gain'])] = logp
            ctc_records.append(out)
        by_step[rec['step']].append(out)
        all_records.append(out)

comparisons = []
for row in optim:
    step, records = row['step'], by_step[row['step']]
    c = defaultdict(float)
    ret = defaultdict(list)
    for r in records:
        if r['kind'] == 'positive_retention':
            ret[r['id']].append(r['unweighted_loss'])
        else:
            k = 'natural_ctc' if r['kind'] == 'natural_ordinary' else 'synthetic_ctc'
            c[k] += r['weighted_base_total']
            c[r['kind'] + '_ctc_only'] += r['weighted_ctc']
            c[r['kind'] + '_prefix_only'] += r['weighted_prefix']
            if 'weighted_ceiling' in r:
                c['ordinary_ceiling_weighted'] += r['weighted_ceiling']
    c['positive_retention'] = .08 * float(np.mean([np.mean(v) for v in ret.values()]))
    for kind in ('synthetic_ordinary', 'synthetic_positive'):
        start = row['prefix'] // 160 - 3
        clean = logps[(step, kind, 1.)][start:]
        half = logps[(step, kind, .5)][start:]
        kl = float(np.mean(np.sum(np.exp(clean) * (clean - half), axis=1)))
        c['gain_consistency'] += .015 * min(8, max(0, kl))
    c['loss'] = sum(c[k] for k in ('synthetic_ctc', 'natural_ctc', 'positive_retention',
                                 'ordinary_ceiling_weighted', 'gain_consistency'))
    errors = {k: abs(c[k] - row[k]) for k in ('loss', 'synthetic_ctc', 'natural_ctc', 'positive_retention', 'gain_consistency')}
    errors['ordinary_ceiling_weighted'] = abs(c['ordinary_ceiling_weighted'] - .08 * row['ordinary_ceiling'])
    comparisons.append(dict(step=step, reconstructed=dict(c), absolute_errors=errors))

error_max = {k: max(r['absolute_errors'][k] for r in comparisons) for k in comparisons[0]['absolute_errors']}
assert max(error_max.values()) < 5e-4, error_max

component_names = ['synthetic_ctc', 'natural_ctc', 'positive_retention', 'ordinary_ceiling_weighted', 'gain_consistency']
components = [{k: (.08 * r['ordinary_ceiling'] if k == 'ordinary_ceiling_weighted' else r[k]) for k in component_names}
              for r in optim]
loss_sum = sum(r['loss'] for r in optim)
component_summary = {k: {**stats([r[k] for r in components]),
                         'share_of_sum_loss': sum(r[k] for r in components) / loss_sum}
                     for k in component_names}
blocks = []
for lo in range(1, 193, 34):
    hi = min(lo + 33, 192)
    rows = optim[lo - 1:hi]
    block = dict(step_first=lo, step_last=hi, loss=stats([r['loss'] for r in rows]),
                 preclip_gradient_norm=stats([r['preclip_gradient_norm'] for r in rows]))
    block['components'] = {k: stats([r[k] for r in components[lo - 1:hi]]) for k in component_names}
    blocks.append(block)


def match_summary(records, keyfields, metric, event='events'):
    grouped = defaultdict(list)
    for r in records:
        grouped[tuple(r[k] for k in keyfields)].append(r)
    pairs = []
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r['step'])
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        pair = {k: v for k, v in zip(keyfields, key)}
        pair.update(visits=len(rows), first_step=first['step'], last_step=last['step'],
                    first_value=first[metric], last_value=last[metric],
                    delta=last[metric] - first[metric],
                    first_hit=bool(first[event]), last_hit=bool(last[event]),
                    first_blank_fraction=first['blank_argmax_fraction'],
                    last_blank_fraction=last['blank_argmax_fraction'])
        pairs.append(pair)
    transitions = Counter(('hit' if p['first_hit'] else 'miss') + '_to_' + ('hit' if p['last_hit'] else 'miss') for p in pairs)
    return dict(event_scope=event, conditions=len(grouped), paired_conditions=len(pairs),
                single_visit_conditions=len(grouped) - len(pairs),
                first_value=stats([p['first_value'] for p in pairs]),
                last_value=stats([p['last_value'] for p in pairs]),
                delta=stats([p['delta'] for p in pairs]),
                increased=sum(p['delta'] > 1e-6 for p in pairs),
                decreased=sum(p['delta'] < -1e-6 for p in pairs),
                event_transitions=dict(transitions), pairs=pairs)


retention_summary = {}
for name, selected in [('all122', retention_records),
                        ('old120', [r for r in retention_records if not r['newly_protected']]),
                        ('new2', [r for r in retention_records if r['newly_protected']])]:
    retention_summary[name] = match_summary(selected, ['id', 'gain'], 'unweighted_loss')

ctc_summary = {}
for kind in ('synthetic_positive', 'synthetic_ordinary', 'natural_ordinary'):
    selected = [r for r in ctc_records if r['kind'] == kind]
    summary = match_summary(selected, ['id', 'gain', 'prefix'], 'ctc_nll', event='suffix_events')
    summary['full_array_event_transitions'] = match_summary(
        selected, ['id', 'gain', 'prefix'], 'ctc_nll', event='events')['event_transitions']
    summary['all_observations'] = dict(n=len(selected), events=sum(r['events'] for r in selected),
                                     observations_with_events=sum(bool(r['events']) for r in selected),
                                     suffix_observations_with_events=sum(bool(r['suffix_events']) for r in selected),
                                     ctc_nll=stats([r['ctc_nll'] for r in selected]),
                                     ctc_nll_per_suffix_frame=stats([r['diagnostic_ctc_nll_per_suffix_frame'] for r in selected]),
                                     weighted_ctc=stats([r['weighted_ctc'] for r in selected]),
                                     weighted_prefix=stats([r['weighted_prefix'] for r in selected]),
                                     weighted_ctc_output_gradient_l1=stats([r['weighted_ctc_output_gradient_l1'] for r in selected]))
    if kind != 'synthetic_positive':
        summary['all_observations'].update(ordinary_margin_mean=stats([r['ordinary_margin_mean'] for r in selected]),
                                           ordinary_margin_max=stats([r['ordinary_margin_max'] for r in selected]),
                                           ordinary_active_fraction=stats([r['ordinary_active_fraction'] for r in selected]),
                                           weighted_ceiling_output_gradient_l1=stats([r['weighted_ceiling_output_gradient_l1'] for r in selected]))
    ctc_summary[kind] = summary

prefix_dependency = {}
for kind in sorted({r['kind'] for r in all_records}):
    rows = [r for r in all_records if r['kind'] == kind]
    hits = [r for r in rows if r['events']]
    prefix_dependency[kind] = dict(observations=len(rows),
        observations_with_full_array_event=len(hits),
        first_nonblank_run_is_token1_at_frame0=sum(bool(r['runs'] and r['runs'][0][:2] == [1, 0]) for r in rows),
        first_full_array_event_uses_token1_at_frame0=sum(r['first_event'][0][1] == 0 for r in hits))
    if kind != 'positive_retention':
        prefix_dependency[kind].update(
            first_event_token1_entirely_in_known_silent_prefix=sum(
                r['first_event'][0][2] <= r['blank_prefix_end'] for r in hits),
            observations_with_suffix_only_event=sum(bool(r['suffix_events']) for r in rows))

positive_by_original = defaultdict(list)
for r in ctc_records:
    if r['kind'] == 'synthetic_positive':
        positive_by_original[r['id']].append(r)
total_positive_ctc = sum(r['weighted_ctc'] for rows in positive_by_original.values() for r in rows)
positive_concentration = []
for uid, rows in positive_by_original.items():
    amount = sum(r['weighted_ctc'] for r in rows)
    positive_concentration.append(dict(id=uid, observations=len(rows), tier=rows[0]['tier'],
        weighted_ctc_total=amount, share_of_all_positive_ctc=amount / total_positive_ctc,
        suffix_frames=sorted({r['suffix_frames'] for r in rows}),
        full_array_hits=sum(bool(r['events']) for r in rows),
        suffix_only_hits=sum(bool(r['suffix_events']) for r in rows),
        step_records=[dict(step=r['step'], gain=r['gain'], prefix=r['prefix'],
                           ctc_nll=r['ctc_nll'], weighted_ctc=r['weighted_ctc'],
                           preclip_norm=optim[r['step'] - 1]['preclip_gradient_norm']) for r in rows]))
positive_concentration.sort(key=lambda r: r['weighted_ctc_total'], reverse=True)

new_conditions = []
for spec in source['retention_additions']:
    selected = [r for r in retention_records if r['id'] == spec['id'] and r['gain'] == float(spec['gain'])]
    group_rows = [r for r in source['positive_retention'] + source['retention_additions'] if r['group'] == spec['group']]
    originals = {r['id'] for r in group_rows}
    gains = sum(r['id'] == spec['id'] for r in group_rows)
    coefficient = .08 / len(originals) / gains
    old_gains = sum(r['id'] == spec['id'] for r in source['positive_retention'])
    observations = []
    for r in selected:
        o = optim[r['step'] - 1]
        comp = components[r['step'] - 1]
        observations.append({**r, 'effective_condition_coefficient': coefficient,
                             'weighted_condition_loss': coefficient * r['unweighted_loss'],
                             'share_of_step_loss': coefficient * r['unweighted_loss'] / o['loss'],
                             'step_loss_components': comp,
                             'preclip_gradient_norm': o['preclip_gradient_norm']})
    new_conditions.append(dict(id=spec['id'], gain=float(spec['gain']), teacher=spec['teacher'],
                               effective_condition_coefficient=coefficient,
                               old_condition_coefficient=.08 / len(originals) / old_gains,
                               inherited_condition_weight_multiplier=old_gains / gains,
                               observations=observations,
                               same_original_ctc_observations=[r for r in ctc_records if r['id'] == spec['id']]))

output_gradient = []
for step in range(1, 193):
    records = by_step[step]
    ctc_l1 = sum(r.get('weighted_ctc_output_gradient_l1', 0) for r in records)
    prefix_l1 = sum(r.get('weighted_prefix_output_gradient_l1', 0) for r in records)
    ceiling_l1 = sum(r.get('weighted_ceiling_output_gradient_l1', 0) for r in records)
    output_gradient.append(dict(step=step, weighted_ctc_output_gradient_l1=ctc_l1,
                                weighted_prefix_output_gradient_l1=prefix_l1,
                                weighted_ceiling_output_gradient_l1=ceiling_l1,
                                positive_retention_output_gradient_l1_upper_bound=.16,
                                ctc_to_positive_retention_upper_bound_ratio=ctc_l1 / .16))

report = dict(schema='english-v6-b22-saved-train-only-mechanism-analysis-v1',
              provenance=dict(original_archive=str(ARCHIVE), original_archive_sha256=EXPECTED_SHA,
                              files_sha256={n: digest(b) for n, b in files.items()},
                              arrays_verified=1844, optimizer_steps_observed=192,
                              train_originals=88, retained_conditions=122),
              new_model_forwards=0, new_backward_calls=0, optimizer_updates=0, new_audio=0,
              dev_reads=0, network_calls=0, thresholds_changed=False, b22_modified=False,
              numerical_reconstruction=dict(method='pure NumPy float64 CTC dynamic programming from original raw logits',
                                             max_absolute_errors=error_max),
              component_summary=component_summary,
              largest_component_per_step=dict(Counter(max(r, key=r.get) for r in components)),
              blocks=blocks,
              clipping=dict(clipped_steps=sum(r['gradient_clipped'] for r in optim),
                            threshold=5., preclip_norm=stats([r['preclip_gradient_norm'] for r in optim]),
                            clipping_multiplier=stats([5 / r['preclip_gradient_norm'] for r in optim]),
                            largest_norm_steps=sorted(optim, key=lambda r: r['preclip_gradient_norm'], reverse=True)[:12]),
              positive_retention=retention_summary, base_ctc=ctc_summary,
              prefix_dependency=prefix_dependency,
              teacher_token1_at_frame0=dict(
                  old120=sum(r['b9_registered_spans'][0][1] == 0 for r in source['positive_retention']),
                  additions=[dict(id=r['id'], gain=r['gain'], teacher=r['teacher'],
                                  first_span=r['teacher_spans'][0]) for r in source['retention_additions']]),
              positive_ctc_concentration=positive_concentration,
              new_conditions=new_conditions,
              output_space_derivatives=dict(
                  limitation='Analytic derivatives only with respect to saved independent output logits; no model Jacobians, parameter gradients, Adam moments, or per-component gradient cosines exist in these artifacts. These are not parameter-gradient dominance or conflict measurements.',
                  ctc=stats([r['weighted_ctc_output_gradient_l1'] for r in output_gradient]),
                  ordinary_ceiling=stats([r['weighted_ceiling_output_gradient_l1'] for r in output_gradient]),
                  prefix=stats([r['weighted_prefix_output_gradient_l1'] for r in output_gradient]),
                  positive_retention_total_upper_bound=.16,
                  ctc_to_retention_upper_bound_ratio=stats([r['ctc_to_positive_retention_upper_bound_ratio'] for r in output_gradient]),
                  per_step=output_gradient),
              limitations=[
                  'All observations are pre-update training outputs at different optimizer steps, not final checkpoint evaluations.',
                  'Exactly matched condition pairs fix original, gain and prefix but span different optimizer steps.',
                  'Scalar loss magnitude and output-space derivatives do not determine shared-parameter gradient directions or optimizer effects.',
                  'Global gradient clipping scales the summed gradient; clipping alone does not prove inter-objective conflict.',
                  'A teacher-span hinge can grow while the actual greedy full-word event survives or moves.',
                  'No B22 promotion conclusion, no new B23 experiment, and no unseen holdout inspection is authorized by this report.'])

(ROOT / 'B22_TRAIN_ONLY_TRACE_ANALYSIS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
(ROOT / 'B22_TRAIN_ONLY_ARRAY_READBACK.json').write_text(json.dumps(dict(schema='english-v6-b22-original-train-array-readback-v1',
    source_zip_sha256=EXPECTED_SHA, new_model_forwards=0, dev_reads=0, records=all_records), ensure_ascii=False, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(report=str(ROOT / 'B22_TRAIN_ONLY_TRACE_ANALYSIS.json'),
                     arrays=1844, reconstruction_error=error_max,
                     component_summary=component_summary,
                     clipping=report['clipping']['clipping_multiplier'],
                     retention_transitions={k: v['event_transitions'] for k, v in retention_summary.items()},
                     retained_loss_first_last={k: [v['first_value']['mean'], v['last_value']['mean']] for k, v in retention_summary.items()},
                     ctc_transitions={k: v['event_transitions'] for k, v in ctc_summary.items()},
                     ctc_gradient=report['output_space_derivatives']['ctc']), indent=2))
