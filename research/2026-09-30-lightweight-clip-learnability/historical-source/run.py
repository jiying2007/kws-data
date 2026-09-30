"""Four preregistered CPU runs, invoked only after independent pre-training review."""
import argparse
import hashlib
import json
import pathlib
import time
import sys
import numpy as np
import torch
from models import CausalClassifier, clip_loss

ROOT = pathlib.Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

def batch(rows, features):
    arrays = [torch.from_numpy(features[r['recording']]) for r in rows]
    lengths = torch.tensor([len(x) for x in arrays], dtype=torch.long)
    x = torch.zeros(len(rows), int(lengths.max()), 32)
    targets = torch.zeros(len(rows), 2)
    for i, (r, values) in enumerate(zip(rows, arrays)):
        x[i, :len(values)] = values
        if r['kind'] == 'positive':
            targets[i, r['keyword_id'] - 1] = 1
    return x, lengths, targets

def evaluate(model, rows, features):
    details = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            x = torch.from_numpy(features[row['recording']])[None]
            logits = model(x)[0]
            scores, where = logits.max(0)
            details.append({key: row[key] for key in ('recording', 'dataset_id', 'split', 'review_method',
                                                     'kind', 'keyword_id', 'intended_text', 'speaker_id')})
            details[-1].update(max_logits=scores.tolist(), argmax_frames=where.tolist(),
                               predicted_keywords=[i + 1 for i in range(2) if float(scores[i]) >= 0])
    groups = {}
    for method in ('human', 'asr'):
        for role in ('train', 'development_a', 'development_b'):
            subset = [d for d in details if d['review_method'] == method and d['split'] == role]
            if not subset:
                continue
            positive = [d for d in subset if d['kind'] == 'positive']
            negative = [d for d in subset if d['kind'] == 'confusable']
            groups[method + ':' + role] = {
                'clips': len(subset), 'positives': len(positive), 'confusables': len(negative),
                'positive_target_hits': sum(d['keyword_id'] in d['predicted_keywords'] for d in positive),
                'positive_wrong_keyword_clips': sum(any(k != d['keyword_id'] for k in d['predicted_keywords'])
                                                     for d in positive),
                'confusable_triggered_clips': sum(bool(d['predicted_keywords']) for d in negative),
                'exact_clip_labels_correct': sum(d['predicted_keywords'] == ([d['keyword_id']] if d['kind'] == 'positive' else [])
                                                for d in subset),
                'per_keyword': {str(k): {'positive_clips': sum(d['keyword_id'] == k for d in positive),
                                        'target_hits': sum(d['keyword_id'] == k and k in d['predicted_keywords'] for d in positive),
                                        'confusable_triggered_clips': sum(k in d['predicted_keywords'] for d in negative)}
                                for k in (1, 2)}}
    return details, groups

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=['A', 'B'], required=True)
    parser.add_argument('--seed', type=int, choices=[1337, 2346], required=True)
    parser.add_argument('--review-receipt', type=pathlib.Path, required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / 'spec.json').read_text())
    if (sys.version, torch.__version__, np.__version__) != (spec['research_python'], spec['research_torch'], spec['research_numpy']):
        raise ValueError('research runtime version drift')
    review = json.loads(args.review_receipt.read_text())
    if review.get('approved') is not True or review.get('spec_sha256') != sha(ROOT / 'spec.json'):
        raise ValueError('independent review approval must bind this exact spec')
    for name, expected in spec['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('preregistered source drift: ' + name)
    for name, key in [('features.npz', 'feature_cache_sha256'), ('rows.json', 'rows_sha256')]:
        if sha(ROOT / name) != spec[key]:
            raise ValueError('input identity drift: ' + name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    features = np.load(ROOT / 'features.npz')
    rows = json.loads((ROOT / 'rows.json').read_text())
    train = [r for r in rows if r['review_method'] == 'human' and r['split'] == 'train']
    if len(train) != 12:
        raise ValueError('training membership drift')
    model = CausalClassifier(args.architecture)
    if sum(p.numel() for p in model.parameters()) != spec['architectures'][args.architecture]['parameters']:
        raise ValueError('architecture parameter drift')
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['lr'])
    order = torch.Generator().manual_seed(args.seed + 10000)
    output = ROOT / 'results' / (args.architecture + '-seed' + str(args.seed))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'invocation.json', {'spec_sha256': sha(ROOT / 'spec.json'), 'review_sha256': sha(args.review_receipt),
                                    'architecture': args.architecture, 'seed': args.seed})
    started = time.monotonic()
    indices = []
    progress = []
    complete = False
    for step in range(1, spec['steps'] + 1):
        if time.monotonic() - started > spec['per_run_wall_cap_seconds']:
            break
        if not indices:
            indices = torch.randperm(len(train), generator=order).tolist()
        chosen, indices = indices[:4], indices[4:]
        x, lengths, labels = batch([train[i] for i in chosen], features)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = clip_loss(model(x), lengths, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            point = {'step': step, 'batch_loss': float(loss.detach()), 'wall_seconds': time.monotonic() - started}
            progress.append(point)
            save(output / 'progress.json', progress)
            print(json.dumps(dict(point, architecture=args.architecture, seed=args.seed)), flush=True)
        if step == spec['steps']:
            complete = True
    torch.save({'state_dict': model.state_dict(), 'architecture': args.architecture, 'seed': args.seed,
                'spec_sha256': sha(ROOT / 'spec.json'), 'completed': complete}, output / 'last.pt')
    details, groups = evaluate(model, rows, features)
    save(output / 'clip-scores.json', details)
    save(output / 'summary.json', {'completed': complete, 'steps': step if complete else step - 1,
                                  'wall_seconds': time.monotonic() - started, 'groups': groups,
                                  'checkpoint_sha256': sha(output / 'last.pt'),
                                  'spec_sha256': sha(ROOT / 'spec.json'), 'threshold_logit': 0,
                                  'evidence_scope': 'observed-development-whole-clip-scores-only'})

if __name__ == '__main__':
    main()
