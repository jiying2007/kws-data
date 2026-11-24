"""One B22 TRAIN challenger; reuse B21 control and every original teacher array.

The sole scientific change is inclusion of the two previously unprotected
TRAIN source/gain conditions already completed by original B0 or B9. No
previous training is replayed. New independent panels must be frozen first.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_competing_margin_pair as old

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DESIGN = HERE / 'english_v6_b22_union_retention_design.json'
DESIGN_SHA = '6a73fe3996397784e8f6c8df0e5d22692f6ba17d516c66236d5adc37d12a1964'
BINDING = HERE / 'english_v6_b22_assessment_binding.json'
OLD_PRODUCER = HERE / 'english_v6_b22_b21_producer_manifest.json'
REPO = 'jiying2007/kws-data'
BRANCH = 'refs/heads/research/en-v6-b22-union-retention-train-continuation-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b22-union-retention-train-continuation-once.yml'
MARKER = 'english-v6-b22-union-retention-train-continuation-consumed'
SCHEMA = 'english-v6-b22-train-union-retention-v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write(path, value):
    Path(path).write_bytes(encoded(value))


def design():
    raw = DESIGN.read_bytes()
    need(sha(raw) == DESIGN_SHA, 'exact frozen TRAIN-only single-factor recipe')
    value = json.loads(raw)
    need(value['updates'] == 192 and value['parameters'] == 43876
         and value['budget']['total_new_model_forwards'] == 1847
         and value['historical_control']['new_optimizer_updates'] == 0
         and value['existing_retention_conditions'] == 120 and value['union_retention_conditions'] == 122,
         'one new challenger, original control and fixed two-condition expansion')
    return value


def assessment_binding(require_frozen=True):
    value = json.loads(BINDING.read_bytes())
    need(value['schema'] == 'english-v6-b22-prospective-assessment-binding-v1'
         and value['recipe_path'] == DESIGN.relative_to(ROOT).as_posix()
         and value['recipe_sha256'] == DESIGN_SHA, 'assessment binds this immutable recipe')
    if not require_frozen and value['frozen'] is False:
        need(value['assessment'] is None and value['files'] == {} and value['optimizer_allowed'] is False,
             'explicit inert metadata; never an execution permission')
        return value
    need(value['frozen'] is value['optimizer_allowed'] is True and type(value['assessment']) is dict
         and type(value['files']) is dict and bool(value['files']), 'new independent assessment must be frozen before optimizer')
    for name, digest in value['files'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts and type(digest) is str
             and re.fullmatch(r'[0-9a-f]{64}', digest), 'complete safe assessment resource pin')
        target = ROOT / path
        need(target.is_file() and not target.is_symlink() and sha(target.read_bytes()) == digest,
             'exact prospective assessment resource ' + name)
    pin = value['assessment']
    need(set(pin) == {'path', 'sha256'} and value['files'].get(pin['path']) == pin['sha256'],
         'main assessment is part of its complete resource inventory')
    assessment = json.loads((ROOT / pin['path']).read_bytes())
    need(assessment['freeze_before_optimizer_and_new_source'] is True
         and assessment['training']['design_path'] == DESIGN.relative_to(ROOT).as_posix()
         and assessment['training']['design_sha256'] == DESIGN_SHA,
         'new independent panels frozen for this exact TRAIN intervention')
    import english_v6_b22_source_metadata as metadata
    need(Path(metadata.__file__).resolve() == HERE / 'english_v6_b22_source_metadata.py'
         and metadata.validate()['hashes'] == value['files'],
         'complete original metadata qualification from the declared minimal producer')
    return value


def architecture_contract():
    path = HERE / 'english_v5_mero_fsmn_ctc_once.py'
    node = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_model')
    need(sha(ast.dump(node, include_attributes=False).encode()) == design()['model_build_ast_sha256'],
         'unchanged 43876-parameter architecture')
    names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    need(not names & {'Dropout', 'BatchNorm1d', 'BatchNorm2d', 'rand', 'randn', 'bernoulli', 'multinomial'},
         'no stochastic forward or running normalization state')
    return dict(unchanged=True, dropout=False, batchnorm=False, random_forward=False,
                historical_numerical_replay_claimed=False)


def gh(endpoint):
    return subprocess.run(['gh', 'api', 'repos/' + REPO + '/' + endpoint, '--allow-escape-sequences'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def guard(parent, env=None, event=None, git_parent=None):
    assessment_binding()
    if env is None:
        import english_v6_b22_training_origin as origin
        origin.verify_live()
    e = os.environ if env is None else env
    need(re.fullmatch(r'[0-9a-f]{40}', parent or ''), 'exact prepared predecessor')
    need(all(e.get(k) == v for k, v in dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=REPO,
         GITHUB_EVENT_NAME='push', GITHUB_REF=BRANCH, GITHUB_RUN_ATTEMPT='1').items()),
         'one public original B22 execution only')
    head = e.get('GITHUB_SHA', '')
    need(re.fullmatch(r'[0-9a-f]{40}', head) and re.fullmatch(r'[1-9][0-9]*', e.get('GITHUB_RUN_ID', '')),
         'actual run identity')
    if event is None:
        event = json.loads(Path(e['GITHUB_EVENT_PATH']).read_bytes())
    if git_parent is None:
        git_parent = subprocess.check_output(['git', '-C', e['GITHUB_WORKSPACE'], 'rev-parse', 'HEAD^1'], text=True).strip()
    need(event.get('before') == git_parent == parent and event.get('after') == head
         and event.get('ref') == BRANCH and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPO and event['repository'].get('private') is False,
         'exact original nonforced single-parent public activation')
    return dict(run_id=e['GITHUB_RUN_ID'], run_head=head, run_attempt=1, prepared_parent=parent, run_branch=BRANCH)


def original_producer(inputs):
    value = design()
    with zipfile.ZipFile(inputs / 'b21_producer.zip') as archive:
        raw = archive.read('PRODUCER.json')
        need(raw == OLD_PRODUCER.read_bytes() and sha(raw) == value['original_b21_producer_manifest_sha256'],
             'entire actual original B21 producer manifest, not an AST guess')
        manifest = json.loads(raw)
        need(len(manifest['files']) == 221, 'all original B21 producer dependencies')
        for name, pin in manifest['files'].items():
            data = archive.read('repository/' + name)
            need(len(data) == pin['bytes'] and sha(data) == pin['sha256'], 'exact original producer member ' + name)
            current = (ROOT / name).read_bytes()
            if name == manifest['runtime_workflow']:
                token = b'__B21_REVIEWED_TRAIN_PARENT__'
                need(current.count(token) == 2 and current.replace(token, manifest['prepared_parent'].encode()) == data,
                     'unchanged old workflow-only activation template')
            else:
                need(current == data, 'original TRAIN dependency changed: ' + name)
    return manifest


def verify_archives(inputs):
    inputs = Path(inputs)
    pins = design()['input_archives']
    need({p.name for p in inputs.glob('*.zip')} == set(pins), 'exact 18 original archives; no added DEV data')
    for name, pin in pins.items():
        path = inputs / name
        need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == pin['sha256'], 'original archive ' + name)
        if 'bytes' in pin:
            need(path.stat().st_size == pin['bytes'], 'original archive size ' + name)
        with zipfile.ZipFile(path) as archive:
            need(archive.testzip() is None, 'complete original archive CRC ' + name)
    original_producer(inputs)
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
        source_raw = archive.read('SOURCE.json')
        need(sha(source_raw) == design()['source_manifest_sha256'], 'original B21 source receipt')
        source = json.loads(source_raw)
        need(source['input_zip_sha256'] == {name: pin['sha256'] for name, pin in pins.items() if not name.startswith('b21_')},
             'all sixteen TRAIN inputs unchanged from actual B21')
    return source


def union_evidence(inputs, np):
    value = design()
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
        source = json.loads(archive.read('SOURCE.json'))
        mechanism = json.loads(archive.read('TRAIN_MECHANISM.json'))
    need(old.saved_train_mechanism(inputs / 'b17.zip', np) == mechanism, 'same original 528-array TRAIN mechanism')
    existing = {(r['id'], str(float(r['gain']))) for r in source['positive_retention']}
    need(len(existing) == 120, 'every original B21 retention condition preserved')
    additions, complete = [], set()
    with zipfile.ZipFile(inputs / 'b17.zip') as archive:
        report = json.loads(archive.read('TRAIN_DIAGNOSTIC.json'))
        raw = archive.read('TRAIN_DIAGNOSTIC_LOGITS.npz')
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        for row in report['records']:
            if not row['positive']:
                continue
            for gain, state in row['gains'].items():
                key = row['id'], gain
                teacher = 'B9' if state['B9']['events'] else ('B0' if state['B0']['events'] else None)
                if teacher is None:
                    need(key not in existing, 'no invented correct teacher')
                    continue
                complete.add(key)
                if key in existing:
                    continue
                pin = state[teacher]
                matrix = arrays[pin['array']]
                need(sha(matrix.tobytes()) == pin['sha256'], 'exact successful union teacher')
                spans = old.b15.completed_word(matrix, np)
                need(bool(spans), 'native full phrase from original TRAIN array')
                additions.append(dict(id=row['id'], group=row['group'], gain=gain, teacher=teacher,
                    pcm_sha256=row['pcm_sha256'], teacher_array=pin['array'], teacher_array_sha256=pin['sha256'],
                    teacher_spans=[list(v) for v in spans]))
    additions.sort(key=lambda r: (r['group'], r['id'], r['gain']))
    need(additions == value['retention_additions'] and len(complete) == 122 and existing <= complete,
         'exactly the two preregistered union-success omissions, no score-based subset')
    return dict(schema='english-v6-b22-original-train-union-evidence-v1', existing_conditions=120,
                union_conditions=122, additions=additions, new_model_forwards=0, new_dev_reads=0)


def source_preflight(inputs, require_registered=True):
    """Complete original source admission with NumPy arrays; no Torch/model import."""
    inputs = Path(inputs)
    assessment_binding(require_registered)
    source = verify_archives(inputs)
    architecture = architecture_contract()
    import numpy as np
    extra, extra_waves = old.b13.reviewed_gate(inputs, json.loads(old.b13.PINS.read_bytes()), np)
    rows, waves, groups = old.b13.paired.original_training(inputs)
    need(len(rows) == 79 and extra == source['natural_train'] and len(extra) == 9,
         'exact approved synthetic and owner-reviewed natural TRAIN sources')
    observed = {r['id']: (r['source_group_id'], r['source_pcm_sha256']) for r in rows}
    expected = {r['id']: (r['group'], r['pcm_sha256']) for r in source['base_train']}
    need(observed == expected and not set(waves) & set(extra_waves), 'exact original 88-source inventory')
    for row in rows + extra:
        pcm = (waves if row['id'] in waves else extra_waves)[row['id']]
        need(row['role'] == 'train' and sha(pcm) == row['source_pcm_sha256'], 'TRAIN identity and original PCM bytes')
    slots = old.shared_schedule(groups, extra)
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
        raw = archive.read('COMMON_DRAW.json')
        need(sha(raw) == design()['common_draw_sha256'] and json.loads(raw) == slots, 'exact original 192 source/prefix draws')
        ordinary = json.loads(archive.read('B0_ORDINARY_TEACHER.json'))
        teacher_raw = archive.read('B0_ORDINARY_TEACHER_LOGITS.npz')
        need(sha(teacher_raw) == ordinary['logits_sha256'] and ordinary['parent_pt_sha256'] == old.B0_PT,
             'authentic original 264 B0 ordinary teachers')
        with np.load(io.BytesIO(teacher_raw), allow_pickle=False) as arrays:
            keys = set()
            for row in ordinary['rows']:
                matrix = arrays[row['array']]
                need(matrix.shape == (row['frames'], 4) and np.isfinite(matrix).all()
                     and sha(matrix.tobytes()) == row['sha256'], 'each original B0 teacher matrix')
                keys.add((row['id'], float(row['gain']), row['prefix_samples']))
            desired = {(s[k]['id'], gain, s['prefix']) for s in slots for k in ('ordinary', 'natural') for gain in old.b10.GAINS}
            need(keys == desired and len(keys) == len(ordinary['rows']) == len(arrays.files) == 264,
                 'complete shared ordinary teachers without new forwards')
    union = union_evidence(inputs, np)
    positive_counts = defaultdict(int)
    for row in source['positive_retention'] + union['additions']:
        positive_counts[row['group']] += 1
    positive_groups = sorted(positive_counts)
    retention_forwards = sum(positive_counts[positive_groups[step % len(positive_groups)]] for step in range(192))
    need(len(positive_groups) == 34 and retention_forwards == 692
         and 192 * 6 + retention_forwards == design()['budget']['optimization_forwards'],
         'exact forward budget derived from all original memberships and the fixed schedule')
    report = dict(schema='english-v6-b22-authenticated-train-source-v1', originals=88, synthetic_originals=79,
        natural_originals=9, natural_readers=8, ordinary_teachers_reused=264, same_common_draw=True,
        architecture=architecture, union=union, input_archives=design()['input_archives'],
        base_train=source['base_train'], natural_train=extra,
        positive_retention=source['positive_retention'], retention_additions=union['additions'],
        train_originals=[dict(id=r['id'], source_group_id=r['source_group_id'],
            source_pcm_sha256=r['source_pcm_sha256'], role=r['role'], mono_class=r['mono_class'])
            for r in sorted(rows + extra, key=lambda r: r['id'])],
        registered_positive_retention_forwards=retention_forwards,
        registered_base_ctc_forwards=192 * 6,
        new_model_forwards=0, new_source_calls=0, new_dev_reads=0,
        prospective_assessment_frozen=bool(json.loads(BINDING.read_bytes())['frozen']))
    need('torch' not in sys.modules, 'complete TRAIN preflight precedes acoustic runtime import')
    return report, rows, waves, groups, extra, extra_waves, slots


def fetch(inputs, parent):
    identity = guard(parent)
    inputs = Path(inputs)
    need(inputs.is_absolute() and inputs.parent.is_dir() and not inputs.exists(), 'fresh original-input directory')
    inputs.mkdir()
    receipts = {}
    for name, pin in design()['input_archives'].items():
        artifact = json.loads(gh('actions/artifacts/' + str(pin['artifact_id'])))
        need(artifact['id'] == pin['artifact_id'] and artifact['expired'] is False
             and artifact['digest'] == 'sha256:' + pin['sha256'], 'original live artifact identity ' + name)
        raw = gh('actions/artifacts/' + str(pin['artifact_id']) + '/zip')
        need(sha(raw) == pin['sha256'] and len(raw) == artifact['size_in_bytes'], 'exact original downloaded archive ' + name)
        (inputs / name).write_bytes(raw)
        receipts[name] = artifact
    write(inputs / 'FETCH.json', dict(identity=identity, artifacts=receipts, recipe_sha256=DESIGN_SHA,
                                   assessment_binding_sha256=sha(BINDING.read_bytes()), new_model_forwards=0))


def consume(inputs, parent):
    identity = guard(parent)
    report, *_ = source_preflight(inputs)
    marker = Path(inputs) / 'CONSUMPTION.json'
    need(not marker.exists(), 'single original training budget; never replay')
    previous = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(previous['total_count'] == 0 and previous['artifacts'] == [], 'B22 training budget already consumed')
    write(Path(inputs) / 'SOURCE_PREFLIGHT.json', report)
    value = dict(schema=SCHEMA, status='TRAINING_BUDGET_CONSUMED_UNKNOWN', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        source_preflight_sha256=sha((Path(inputs) / 'SOURCE_PREFLIGHT.json').read_bytes()),
        budget=design()['budget'], promotion_allowed=False)
    write(marker, value)
    return value


def verify_consumed(inputs, parent):
    identity = guard(parent)
    local = json.loads((inputs / 'CONSUMPTION.json').read_bytes())
    need(local['identity'] == identity and local['budget'] == design()['budget']
         and local['recipe_sha256'] == DESIGN_SHA and local['assessment_binding_sha256'] == sha(BINDING.read_bytes())
         and local['source_preflight_sha256'] == sha((inputs / 'SOURCE_PREFLIGHT.json').read_bytes()),
         'exact preoptimizer consumption and source proof')
    runs = json.loads(gh('actions/workflows/' + Path(WORKFLOW).name + '/runs?branch=' + BRANCH.removeprefix('refs/heads/') + '&per_page=100'))
    need(runs['total_count'] == len(runs['workflow_runs']) == 1
         and str(runs['workflow_runs'][0]['id']) == identity['run_id']
         and runs['workflow_runs'][0]['head_sha'] == identity['run_head']
         and runs['workflow_runs'][0]['run_attempt'] == 1, 'single original public B22 training run')
    items = json.loads(gh('actions/artifacts?name=' + MARKER + '&per_page=100'))
    need(items['total_count'] == len(items['artifacts']) == 1, 'one durably uploaded consumed budget')
    item = items['artifacts'][0]
    need(item['expired'] is False and item['workflow_run']['id'] == int(identity['run_id'])
         and item['workflow_run']['head_sha'] == identity['run_head'], 'consumption belongs to this exact run')
    raw = gh('actions/artifacts/' + str(item['id']) + '/zip')
    need(sha(raw) == item['digest'].removeprefix('sha256:') and len(raw) == item['size_in_bytes'], 'original consumption archive')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [v.filename for v in archive.infolist() if not v.is_dir()]
        need(names == ['CONSUMPTION.json'] and json.loads(archive.read(names[0])) == local,
             'uploaded original consumption matches the verified local budget')
    return identity


def prepare_union_paths(inputs, rows, waves, frontend, np, torch):
    paired, audit = old.prepare_paths(inputs, rows, waves, frontend, old.b15.read_design(), np, torch)
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
        need(json.loads(encoded(audit)) == json.loads(archive.read('SOURCE.json'))['positive_retention'], 'all original120 references unchanged')
    indices = defaultdict(list)
    for row in audit:
        indices[(row['group'], row['id'])].append(row)
    result = defaultdict(dict)
    for group, originals in paired['challenger'].items():
        for uid, entries in originals.items():
            result[group][uid] = [(x, spans, dict(meta, teacher='B9', newly_protected=False))
                                 for (x, spans), meta in zip(entries, indices[(group, uid)])]
    index = {row['id']: row for row in rows}
    mel, mean, scale = frontend
    with zipfile.ZipFile(inputs / 'b17.zip') as archive:
        raw = archive.read('TRAIN_DIAGNOSTIC_LOGITS.npz')
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        for item in design()['retention_additions']:
            row = index[item['id']]
            need(row['source_pcm_sha256'] == item['pcm_sha256'] and row['source_group_id'] == item['group'], 'same missing TRAIN positive')
            pcm = old.b15.diagnostic.frozen.ref.gain_pcm(waves[row['id']], float(item['gain']), np)
            f = old.b15.diagnostic.frozen.v5.features(pcm, row, torch, np, mel)
            x = torch.from_numpy(((f - mean) / scale).astype(np.float32))[None]
            matrix = arrays[item['teacher_array']]
            need(matrix.shape == (x.shape[1], 4) and sha(matrix.tobytes()) == item['teacher_array_sha256'], 'exact union teacher and native frame grid')
            spans = []
            for token, start, end in item['teacher_spans']:
                other = old.competitors(token, 'challenger')
                reference = matrix[start:end, token] - matrix[start:end][:, other].max(axis=1)
                need(len(reference) > 0 and np.all(reference >= 0), 'complete full-competitor teacher span')
                spans.append((token, start, end, other, torch.from_numpy(reference.copy())))
            result[item['group']][item['id']].append((x, spans, dict(item, newly_protected=True)))
    need(len(result) == 34 and sum(len(v) for v in result.values()) == 43
         and sum(len(entries) for originals in result.values() for entries in originals.values()) == 122,
         'one exact 120-to122 coverage intervention')
    return result, audit


def saved_ordinary_teachers(inputs, torch, np):
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
        rows = json.loads(archive.read('B0_ORDINARY_TEACHER.json'))['rows']
        raw = archive.read('B0_ORDINARY_TEACHER_LOGITS.npz')
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        return {(r['id'], float(r['gain']), r['prefix_samples']): torch.from_numpy(arrays[r['array']].copy()) for r in rows}


def margin_statistics(student, teacher, torch):
    delta = (student[:, 1:] - student[:, :1]) - (teacher[:, 1:] - teacher[:, :1])
    positive = torch.relu(delta.detach())
    return dict(frames=len(student), mean=float(positive.mean()), maximum=float(positive.max()),
                active_fraction=float((delta.detach() > 0).float().mean()),
                active_cells=int((delta.detach() > 0).sum()), cells=delta.numel())


class Recorder:
    def __init__(self, np):
        self.np, self.arrays, self.rows, self.pending = np, {}, [], None

    def before(self, **metadata):
        need(self.pending is None, 'previous forward must return before the next')
        self.pending = metadata

    def hook(self, module, args, output):
        need(self.pending is not None, 'every acoustic forward must be budgeted')
        key = 'forward_' + str(len(self.rows)).zfill(5)
        matrix = output[0].detach().cpu().numpy().copy()
        need(matrix.ndim == 2 and matrix.shape[1] == 4 and self.np.isfinite(matrix).all(), 'finite original full-grid student output')
        self.arrays[key] = matrix
        self.rows.append(dict(self.pending, array=key, frames=len(matrix), sha256=sha(matrix.tobytes())))
        self.pending = None

    def save(self, output):
        self.np.savez_compressed(output / 'TRAIN_FORWARD_LOGITS.npz', **self.arrays)
        write(output / 'TRAIN_FORWARD_INDEX.json', dict(records=self.rows, count=len(self.rows),
              arrays_sha256=sha((output / 'TRAIN_FORWARD_LOGITS.npz').read_bytes()),
              pending_unreturned_forward=self.pending, logits_are_raw=True))


def positive_loss(step, model, by_group, recorder, trace, torch):
    group = sorted(by_group)[step % len(by_group)]
    original_losses = []
    for uid in sorted(by_group[group]):
        gains = []
        for x, spans, meta in by_group[group][uid]:
            recorder.before(step=step + 1, kind='positive_retention', id=uid, group=group,
                            gain=float(meta['gain']), prefix=0, teacher=meta['teacher'], newly_protected=meta['newly_protected'])
            matrix = model(x)[0]
            tokens = []
            for token, start, end, other, reference in spans:
                part = matrix[start:end]
                margin = part[:, token] - part[:, other].max(dim=1).values
                tokens.append(torch.relu(reference.detach() - margin).mean())
            term = torch.stack(tokens).mean()
            gains.append(term)
            trace.append(dict(step=step + 1, id=uid, group=group, gain=meta['gain'], teacher=meta['teacher'],
                              newly_protected=meta['newly_protected'], unweighted_loss=float(term.detach())))
        original_losses.append(torch.stack(gains).mean())
    return torch.stack(original_losses).mean() * old.POSITIVE_WEIGHT


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
                    base, logp = old.b10.b6.suffix_loss(net, base_cache[row['id']][gain][prefix], klass,
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
                loss, logp = old.b10.b6.suffix_loss(net, natural_cache[row['id']][gain][prefix], 0,
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
    stem = 'mero43-b22-challenger192'
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
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    b0net, _unused_b9, frontend = old.b17.load_parents(inputs, torch, np)
    base_cache = old.b13.paired.make_features(rows, waves, frontend, torch, np)
    natural_cache = old.b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    paths, old_audit = prepare_union_paths(inputs, rows, waves, frontend, np, torch)
    teachers = saved_ordinary_teachers(inputs, torch, np)
    write(output / 'UNION_EVIDENCE.json', report['union'])
    write(output / 'OLD120_POSITIVE_REFERENCES.json', old_audit)
    control_dir = output / 'control/models'
    control_dir.mkdir(parents=True)
    control_files = {}
    with zipfile.ZipFile(inputs / 'b21_weights.zip') as archive:
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
    result = dict(schema=SCHEMA, status='TRAINED_UNION_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY', identity=identity,
        recipe_sha256=DESIGN_SHA, assessment_binding_sha256=sha(BINDING.read_bytes()),
        design_sha256=DESIGN_SHA, assessment_sha256=assessment_binding()['assessment']['sha256'],
        source_sha256=sha((output / 'SOURCE.json').read_bytes()),
        common_draw_sha256=sha((output / 'COMMON_DRAW.json').read_bytes()),
        union_evidence_sha256=sha((output / 'UNION_EVIDENCE.json').read_bytes()),
        old120_references_sha256=sha((output / 'OLD120_POSITIVE_REFERENCES.json').read_bytes()),
        model_pt_sha256=dict(B0=old.B0_PT, B9=old.b10.b9.MODELS['B9_control']['pt'],
                            control=design()['historical_control']['pt_sha256'],
                            challenger=candidate['files']['mero43-b22-challenger192.pt']['sha256']),
        control=dict(**design()['historical_control'], output_stem='control/models/mero43-b21-challenger192', files=control_files),
        challenger=candidate, actual_optimizer_updates=192, actual_new_model_forwards=1847,
        new_control_optimizer_updates=0, new_control_forwards=0, new_teacher_forwards=0,
        reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
        dev_cal_locked_forwards=0, new_tts_asr_calls=0, historical_numerical_replay_claimed=False,
        same_job_matched_control=False, architecture=architecture_contract(), promotion_allowed=False,
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
        if mode != 'preflight':
            item.add_argument('--prepared-parent', required=True)
        if mode == 'execute':
            item.add_argument('--output', type=Path, required=True)
        if mode == 'preflight':
            item.add_argument('--offline-check', action='store_true')
    args = parser.parse_args()
    if args.mode == 'check':
        design(); assessment_binding(False); architecture_contract()
        print('PASS: fixed B22 single new challenger, inert until fresh assessment binding')
    elif args.mode == 'fetch':
        fetch(args.inputs, args.prepared_parent)
    elif args.mode == 'consume':
        print(json.dumps(consume(args.inputs, args.prepared_parent), sort_keys=True))
    elif args.mode == 'preflight':
        print(json.dumps(source_preflight(args.inputs, not args.offline_check)[0], sort_keys=True))
    else:
        print(json.dumps(execute(args.inputs, args.output, args.prepared_parent), sort_keys=True))
