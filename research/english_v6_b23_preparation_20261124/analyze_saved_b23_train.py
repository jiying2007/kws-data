"""Reuse the exact saved B22 TRAIN analyzer with the two B23 prefix reductions.

Reads only an explicitly pinned completed B23 TRAIN ZIP. No acoustic framework,
checkpoint evaluation, parameter backward, source generation or DEV scoring.
All 1,844 saved TRAIN outputs retain the original full-array/CTC verification.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'research/english_v6_b22_training_20261124/train_only_diagnostics/analyze_saved_b22_train.py'
ORIGINAL_SHA = '0fc46ba5f4c26272e61665c1a95710c8360953f937f12f1dac921b661631fd3d'
DESIGN_SHA = '8485e152f0ba93e22c510731dc4acd7823b36df033856b4145aa971aa96ebe56'
EDITS = (
    ('Read only original B22 TRAIN outputs', 'Read only original B23 TRAIN outputs'),
    ('ROOT = Path(__file__).resolve().parent', 'ROOT = ANALYSIS_OUTPUT'),
    ("ARCHIVE = ROOT / 'continuation-models.zip'", 'ARCHIVE = ORIGINAL_ARCHIVE'),
    ("EXPECTED_SHA = 'b5b81de69cdd80c102cdfcc8aa83dea3e685a22e99fb5dbff7613ead35fbda3b'", 'EXPECTED_SHA = ORIGINAL_ARCHIVE_SHA'),
    ('blank_loss = -.5 * float(logp[:blank, 0].mean())', 'blank_loss = -.5 * float(logp[:blank, 0].min())'),
    ('float(np.mean(1 - np.exp(logp[:blank, 0])))', 'float(np.max(1 - np.exp(logp[:blank, 0])))'),
    ("schema='english-v6-b22-saved-train-only-mechanism-analysis-v1'", "schema='english-v6-b23-saved-train-only-mechanism-analysis-v1'"),
    ("schema='english-v6-b22-original-train-array-readback-v1'", "schema='english-v6-b23-original-train-array-readback-v1'"),
    ('No B22 promotion conclusion, no new B23 experiment, and no unseen holdout inspection is authorized by this report.',
     'No B23 promotion conclusion, further experiment, or unseen holdout inspection is authorized by this report.'),
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepared_source():
    raw = ORIGINAL.read_bytes()
    need(sha(raw) == ORIGINAL_SHA, 'exact immutable original full TRAIN-array analyzer')
    text = raw.decode()
    for before, after in EDITS:
        need(text.count(before) == 1, 'one explicit existing adapter site: ' + before)
        text = text.replace(before, after)
    for name in ('TRACE_ANALYSIS', 'ARRAY_READBACK'):
        old = 'B22_TRAIN_ONLY_' + name + '.json'
        need(text.count(old) in (1, 2), 'original report destinations remain explicit')
        text = text.replace(old, 'B23_TRAIN_ONLY_' + name + '.json')
    ast.parse(text)
    return text


def analyze(archive, expected, output):
    need(sys.flags.optimize == 0, 'the original complete-array assertions must remain enabled')
    need(re.fullmatch(r'[0-9a-f]{64}', expected or '') is not None, 'actual original TRAIN artifact SHA-256 is mandatory')
    archive, output = Path(archive).resolve(), Path(output).resolve()
    need(archive.is_file() and not archive.is_symlink() and output.parent.is_dir() and not output.exists(),
         'regular original ZIP and fresh analysis output')
    raw = archive.read_bytes()
    need(sha(raw) == expected, 'same complete downloaded original B23 TRAIN bytes')
    with zipfile.ZipFile(io.BytesIO(raw)) as original:
        need(original.testzip() is None, 'all original TRAIN archive CRCs')
        result = json.loads(original.read('RESULT.json'))
        source = json.loads(original.read('SOURCE.json'))
        need(result['schema'] == 'english-v6-b23-train-prefix-peak-v1'
             and result['status'] == 'TRAINED_PREFIX_PEAK_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
             and result['prefix_blank_reduction'] == 'amax' and result['actual_optimizer_updates'] == 192
             and result['actual_new_model_forwards'] == 1847
             and result['source_sha256'] == sha(original.read('SOURCE.json'))
             and result['design_sha256'] == result['recipe_sha256'] == sha(original.read('DESIGN.json')) == DESIGN_SHA
             and source['schema'] == 'english-v6-b23-authenticated-train-source-v1'
             and source['dev_rows_used_for_training'] == 0,
             'actual complete prefix-maximum TRAIN receipt and original source/design')
    text = prepared_source()
    output.mkdir()
    state = dict(__name__='b23_saved_array_analysis', __file__=str(ORIGINAL),
                 ANALYSIS_OUTPUT=output, ORIGINAL_ARCHIVE=archive, ORIGINAL_ARCHIVE_SHA=expected)
    captured = io.StringIO()
    with redirect_stdout(captured):
        exec(compile(text, str(ORIGINAL) + '[B23 two-reduction adapter]', 'exec'), state)
    (output / 'ANALYZER_STDOUT.json').write_text(captured.getvalue())
    for name in ('TRACE_ANALYSIS', 'ARRAY_READBACK'):
        target = output / ('B23_TRAIN_ONLY_' + name + '.json')
        report = json.loads(target.read_bytes())
        report.pop('dev_reads', None)
        report.update(new_candidate_dev_arrays_read=0, embedded_prior_failure_metadata_retained=True,
                      original_analyzer_sha256=ORIGINAL_SHA, adapter_sha256=sha(Path(__file__).read_bytes()),
                      prefix_loss_reduction='amax', original_array_count=1844,
                      new_model_forwards=0, new_backward_calls=0, optimizer_updates=0,
                      promotion_allowed=False, further_experiment_authorized=False)
        target.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    result = dict(status='SAVED_TRAIN_ARRAYS_VERIFIED', original_archive_sha256=expected,
                  arrays=1844, new_model_forwards=0, new_backward_calls=0,
                  report=str(output / 'B23_TRAIN_ONLY_TRACE_ANALYSIS.json'),
                  arrays_report=str(output / 'B23_TRAIN_ONLY_ARRAY_READBACK.json'))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.archive, args.sha256, args.output)
