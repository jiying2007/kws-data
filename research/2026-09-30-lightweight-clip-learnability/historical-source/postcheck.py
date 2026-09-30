"""Read-only trained-checkpoint streaming parity audit, no selection or training."""
import hashlib
import json
import pathlib
import numpy as np
import torch
from models import CausalClassifier

ROOT = pathlib.Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
spec_sha = hashlib.sha256((ROOT / 'spec.json').read_bytes()).hexdigest()
features = np.load(ROOT / 'features.npz')
report = []
with torch.no_grad():
    for folder in sorted((ROOT / 'results').iterdir()):
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        assert checkpoint['completed'] and checkpoint['spec_sha256'] == spec_sha
        model = CausalClassifier(checkpoint['architecture']).eval()
        model.load_state_dict(checkpoint['state_dict'])
        maximum = 0.
        for key in sorted(features.files):
            x = torch.from_numpy(features[key])[None]
            expected = model(x)
            chunks, state, offset = [], None, 0
            sizes = (1, 7, 23, 3, 41)
            j = 0
            while offset < x.shape[1]:
                size = min(sizes[j % len(sizes)], x.shape[1] - offset)
                y, state = model.chunk(x[:, offset:offset + size], state)
                chunks.append(y)
                offset += size
                j += 1
            actual = torch.cat(chunks, 1)
            delta = float((expected - actual).abs().max())
            maximum = max(maximum, delta)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            assert torch.equal(actual.max(1).values >= 0, expected.max(1).values >= 0)
        report.append({'run': folder.name, 'clips': len(features.files),
                       'max_abs_logit_difference': maximum, 'chunk_parity_passed': True,
                       'threshold_decisions_equal': True})
(ROOT / 'postcheck.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
