"""Package a completed real research run; default invocation copies nothing.

This helper never executes a model. It requires saved-weight host/ARM parity,
the exact pipeline source and original upstream license, then writes a new
self-contained offline inference bundle with a per-file SHA256 manifest.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_offline_bundle_contract', HERE / 'english_final_embedding_infer.py')
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)
need, read_json, sha = contract.need, contract.read_json, contract.sha
identity, canonical = contract.file_identity, contract.canonical
EVIDENCE = ('STARTED.json', 'SOURCE_BINDING.json', 'SELECTION.json',
            'MINIMUM_AFTER_DEDUPLICATION.json', 'BUILD.json', 'LIBRISPEECH_SOURCES.json',
            'FEATURE_PREFLIGHT.json', 'FEATURE_EVIDENCE.json', 'EXCLUSIONS.json',
            'MINIMUM_AFTER_FEATURE_GEOMETRY.json', 'STREAM_ROSTER.json', 'TRAINING_STARTED.json',
            'PARITY.json', 'PER_CASE_EVENTS.json', 'RESULT.json')
AUTOMATION_FILES = ('english_final144_embedding_once.py', 'final144_runner.py', 'final144_recipe.py',
                    'final144_training_pins.json', 'final144_execution_pins.json')


def training_evidence(root):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink() and not (root / 'FAILURE.json').exists(),
         'COMPLETED_TRAINING_DIRECTORY_REQUIRED')
    evidence = {name: read_json(root / name) for name in EVIDENCE}
    result, started = evidence['RESULT.json'], evidence['STARTED.json']
    trained = read_json(root / 'train/RESULT.json')
    need(result.get('status') == 'EXECUTED_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY' and
         trained.get('status') == 'EXECUTED_WEAK_RESEARCH_ONLY' and
         result.get('optimizer_updates') == trained.get('optimizer_updates') == 96 and
         len(trained.get('losses', [])) == len(trained.get('optimizer_bag_draws', [])) == 96,
         'REAL_COMPLETE_FIXED_96_UPDATE_RESULT')
    spec = trained.get('spec', {})
    need(all(spec.get(key) == value for key, value in
             {'history': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
              'seed': 20261116, 'optimizer_updates': 96, 'checkpoint_selection': False}.items()) and
         trained.get('saved_replay') and all(0 <= row.get('max_abs_logit_error', -1) <= 1e-4
                                             for row in trained['saved_replay']), 'FIXED_HEAD_AND_SAVED_NPZ_REPLAY')
    need(started.get('test_backend') is False and result.get('test_backend') is False and
         all(document.get(flag) is False for document in (started, result, trained)
             for flag in contract.FALSE_FLAGS), 'UNADMITTED_REAL_RESEARCH_ONLY')
    need(all(result.get(key) == value for key, value in started.items()) and
         result.get('source_denominator') == 144 and result.get('minimum', {}).get('passed') is True and
         evidence['MINIMUM_AFTER_DEDUPLICATION.json'].get('passed') is True and
         evidence['MINIMUM_AFTER_FEATURE_GEOMETRY.json'] == result['minimum'], 'SOURCE_MINIMUM_AND_RUN_BINDING')
    need(result.get('arm_scope') == 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU' and
         all(result.get(key) == 'NOT_RUN' for key in ('independent_frr', 'independent_far',
             'word_end_latency', 'ssc305', 'deployment_frontend_arm_parity')), 'RESEARCH_SCOPE_UNCHANGED')
    weights, binary = identity(root / 'train/causal-head.npz'), identity(root / 'causal-head.bin')
    need(result.get('weights_sha256') == trained.get('weights_sha256') == weights['sha256'] and
         result.get('head_binary_sha256') == binary['sha256'] and binary['bytes'] == 197908 and
         (root / 'causal-head.bin').read_bytes()[:8] == b'EHKWS01\0', 'REAL_SAVED_WEIGHT_IDENTITIES')
    reports = evidence['PARITY.json']
    need(set(reports) == {'host-parity', 'arm-parity'}, 'BOTH_HOST_AND_ARM_REPORTS')
    for name, report in reports.items():
        need(report.get('schema') == 'english-causal-head-c-parity-v1' and
             report.get('head_binary_sha256') == binary['sha256'] and
             identity(root / name / 'head.bin') == binary and
             report.get('threshold') == 0.8 and report.get('confirm_embeddings') == 3 and
             report.get('cooldown_embeddings') == 13 and
             report.get('formal_qualification') is False and report.get('ssc305_qualification') is False and
             0 <= report.get('max_abs_logit_error', -1) <= 1e-4 and
             0 <= report.get('max_abs_probability_error', -1) <= 2e-6 and
             type(report.get('input_embeddings')) is int and 1 <= report['input_embeddings'] <= 40000 and
             type(report.get('head_outputs')) is int and 1 <= report['head_outputs'] <= report['input_embeddings'],
             'COMPLETE_FIXED_POLICY_SAVED_HEAD_PARITY')
        stream = identity(root / name / 'stream.bin')
        with (root / name / 'stream.bin').open('rb') as source:
            header = source.read(12)
        need(stream['sha256'] == report.get('stream_binary_sha256') and
             stream['bytes'] == 12 + 396 * report['input_embeddings'] and header[:8] == b'EHSTR01\0' and
             struct.unpack('<I', header[8:])[0] == report['input_embeddings'], 'ACTUAL_REPLAY_STREAM_IDENTITY')
    need(reports['host-parity'] == reports['arm-parity'] or
         all(reports['host-parity'][key] == reports['arm-parity'][key] for key in
             ('head_binary_sha256', 'stream_binary_sha256', 'events', 'input_embeddings', 'head_outputs')),
         'HOST_ARM_SAVED_INPUT_AND_EVENT_IDENTITY')
    binding, selected, preflight = (evidence[name] for name in
                                    ('SOURCE_BINDING.json', 'SELECTION.json', 'FEATURE_PREFLIGHT.json'))
    bound = result.get('input_binding', {})
    need(bound == trained.get('input_binding') == evidence['TRAINING_STARTED.json'].get('input_binding') and
         bound.get('source_manifest_sha256') == binding.get('source_manifest_sha256') and
         bound.get('observation_manifest_sha256') == selected.get('observer_artifact_manifest_sha256') and
         bound.get('recipe_sha256') == binding.get('recipe_sha256') and
         bound.get('selection_sha256') == sha(canonical(selected)) and
         bound.get('pipeline_source_sha256') == sha(canonical(preflight.get('pipeline_source'))) and
         bound.get('feature_evidence_sha256') == identity(root / 'FEATURE_EVIDENCE.json')['sha256'] and
         bound.get('feature_stream_sha256') == identity(root / 'actual-embedding-streams.npz')['sha256'] and
         bound.get('worker_code_sha256') == started.get('worker_code_sha256'), 'ACTUAL_TRAINING_INPUT_BINDINGS')
    need(binding.get('test_backend') is False and binding.get('run_id') == result.get('run_id') and
         binding.get('github_sha') == result.get('github_sha') and
         preflight.get('geometry', {}).get('status') == 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED' and
         preflight.get('asset_identities') == {name: list(value) for name, value in contract.ASSETS.items()},
         'REAL_SOURCE_RUN_AND_FEATURE_PREFLIGHT')
    need(all(type(document.get(key)) is str and re.fullmatch('[0-9a-f]{40}', document[key])
             for document, key in ((result, 'github_sha'), (preflight['pipeline_source'], 'commit'))),
         'EXACT_DATA_AND_PIPELINE_SOURCE_COMMITS')
    return evidence, trained, weights, binary


def model_card(result, trained, preflight, model_bytes):
    counts = result['minimum']['counts']
    rows = '\n'.join('| %s | %d | %s |' % (group, item['count'], ', '.join(item['families']))
                     for group, item in sorted(counts.items()))
    return f"""# English frozen embedding research bundle

This bundle contains actual saved weights from run {result['run_id']} in
`jiying2007/kws-data`, source commit `{result['github_sha']}`. The pipeline source
commit is `{preflight['pipeline_source']['commit']}`. It is an offline research
candidate for **Hey Vosen** and **Hey Mero**. It is not SSC305 qualified.

## What was trained and checked

The two unchanged feature ONNX files feed a 16 × 96 causal history, flattened
oldest to newest, with train-only mean/scale, Linear(1536,32), ReLU, and
Linear(32,3). The class order is background=0, Hey Vosen=1, Hey Mero=2.
There were exactly {result['optimizer_updates']} optimizer updates, seed
{trained['spec']['seed']}; no development checkpoint or threshold selection.
The saved NPZ was reopened, and the actual cached embeddings were replayed with
NumPy, native C, and ARM C under QEMU. `evidence/PARITY.json` records their exact
input and weight identities and numerical errors. This checks the causal head;
it does not establish frontend parity or device performance on SSC305.

The source denominator is all 144 generated cells. The fixed weak-intent rule
requires two of three exact normalized ASR texts plus the requested contiguous
machine phone core for wake candidates. Ordinary negative intents require two
matching ASR texts and neither target core. Unmatched rows are quarantined.
Hey acoustics, vowel length and stress remain unverified. These source intents
are not human gold, trusted negatives, or pronunciation admissions.

| Role / weak class | Eligible source cases | Preset families |
| --- | ---: | --- |
{rows}

The additional read-speech material is the previously exposed LibriSpeech
dev-clean source pool. The actual research track observation was
{result['exposed_read_speech_tracks']['seconds']:.3f} seconds over
{result['exposed_read_speech_tracks']['tracks']} speaker tracks, with resets per
speaker and at most 16 seconds per speaker. It is not an independent FAR test.
No independent FRR, qualified FAR, acoustic word-end latency, or SSC305 result
is claimed. Raw source audio is retained in the source run's separate artifacts;
this bundle includes its identities, selection records and saved head replay
stream. Source code and recipe pins locate the complete data-owned experiment.

## Fixed startup, clock and EOF behavior

Input must already be **16,000 Hz, mono, signed PCM16 WAV**. There is no implicit
resampling, gain normalization, leading padding, or tail silence. The frontend
uses float32 PCM16-scale values. Its trailing 16,000 observed samples produce
the first embedding at sample 16,000; subsequent embeddings arrive every
1,280 samples (80 ms). The 16th embedding and first permitted head output occur
at sample **35,200 (2.20 seconds)**. Threshold 0.8 needs three consecutive head
outputs of the same foreground class; earliest confirmation is therefore
sample 37,760 (2.36 seconds). Cooldown is fixed at 13 embedding steps (1.04 s).
These are observation clocks, not measured acoustic word-end latency.

The original file ends where it ends. No EOF output is fabricated. A short WAV
or any finite clip with no event is recorded as **CENSORED_NOT_FRR**; absence of
an event is not a measured false reject. Each invocation begins with empty
history and writes all embedding/head clocks to `frames.jsonl`.

## Verify and run offline

Run from the unpacked bundle. Verification uses only the Python standard library
and checks the complete file roster, byte sizes, SHA256 values and fixed policy:

```sh
python3 -I -S -B english_final_embedding_infer.py --bundle . --verify-only
```

Install the fixed CPU inference dependencies ahead of offline use (Python 3.12
was used by the research worker). Torch is not needed for user inference:

```sh
python3 -m venv ../english-kws-venv
../english-kws-venv/bin/pip install -r requirements.txt
../english-kws-venv/bin/python -I -B english_final_embedding_infer.py \\
  --bundle . --wav /absolute/path/input-16k-mono.wav \\
  --output ../english-kws-result --execute
```

The output directory must be new and outside this bundle. `RESULT.json` lists
candidate events with sample clocks and EOF status. No runtime download occurs.
The default command without a mode executes no model.

The actual saved embedding stream can also be replayed using only the C head:

```sh
cc -std=c99 -O2 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror \\
  runtime/english_causal_head.c runtime/head_replay.c -lm -o ../head-replay
../head-replay models/causal-head.bin evidence/head-replay-stream.bin
```

## Model size, sources and licenses

The two feature models alone occupy 2,414,536 bytes, exceeding a 2 MiB model
budget. The actual two ONNX files plus the saved NPZ occupy {model_bytes} bytes.
The NPZ is the Python head; `causal-head.bin` is the equivalent portable C form.
Target CPU/memory budgets require separate measurements on the intended device.

The feature files are the fixed openWakeWord v0.5.1 mel frontend and shared
speech embedding assets, declared Apache-2.0 by the audited source. No upstream
pretrained wakeword classifier weights are included. Upstream source commit:
`{contract.UPSTREAM_COMMIT}`. Its original `LICENSE` is copied verbatim to
`licenses/openWakeWord-LICENSE`. That source tree has no NOTICE file. The exact
downstream repository license is in `licenses/kws-pipeline-LICENSE`. The
LibriSpeech source corpus is CC-BY-4.0 and is identified in
`evidence/LIBRISPEECH_SOURCES.json`; no raw corpus recording is redistributed here.
The manifest records per-file identities, source commits, assets, weight hashes
and input bindings. Generated explanatory files are not original upstream notices.
"""


def build(args):
    training, pipeline, assets, output = (Path(getattr(args, name)).absolute()
                                         for name in ('training_result', 'pipeline_source', 'assets', 'output'))
    evidence, trained, weights, binary = training_evidence(training)
    result, preflight = evidence['RESULT.json'], evidence['FEATURE_PREFLIGHT.json']
    code = preflight['pipeline_source']
    actual = subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip()
    need(actual == code['commit'] and not subprocess.check_output(
        ['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(), 'EXACT_CLEAN_PIPELINE_CHECKOUT')
    pairs = {'english_final_embedding_infer.py': HERE / 'english_final_embedding_infer.py',
             'models/causal-head.npz': training / 'train/causal-head.npz',
             'models/causal-head.bin': training / 'causal-head.bin',
             'licenses/openWakeWord-LICENSE': Path(args.upstream_license),
             'licenses/kws-pipeline-LICENSE': pipeline / 'LICENSE',
             'evidence/TRAIN_RESULT.json': training / 'train/RESULT.json',
             'evidence/head-replay-stream.bin': training / 'host-parity/stream.bin',
             'source/librispeech_screen.py': pipeline / 'research/english63_online_c/librispeech_screen.py',
             'source/feature-asset-audit.json': pipeline / 'configs/research/english-embedding-backbone-assets-20261116.json',
             'source/package_final_embedding.py': Path(__file__)}
    for name in contract.RUNTIME_FILES:
        source = pipeline / 'research/english_final_embedding' / name
        need(identity(source)['sha256'] == code['feature_head_files_sha256'].get(name), 'EXACT_TRAINING_RUNTIME_SOURCE')
        pairs['runtime/' + name] = source
    need(identity(pairs['source/librispeech_screen.py'])['sha256'] == code['librispeech_source_helper_sha256'],
         'EXACT_LIBRI_SOURCE_HELPER')
    for name, (size, digest) in contract.ASSETS.items():
        pairs['models/' + name] = assets / name
        need(identity(assets / name) == {'bytes': size, 'sha256': digest}, 'FIXED_FEATURE_ASSET')
    need(identity(args.upstream_license) == {'bytes': contract.UPSTREAM_LICENSE[0],
                                            'sha256': contract.UPSTREAM_LICENSE[1]}, 'ORIGINAL_UPSTREAM_LICENSE')
    pairs.update({'evidence/' + name: training / name for name in EVIDENCE})
    pairs.update({'source/' + name: HERE / name for name in AUTOMATION_FILES})
    data_head = subprocess.check_output(['git', '-C', str(HERE.parent), 'rev-parse', 'HEAD'], text=True).strip()
    need(data_head == result['github_sha'], 'EXACT_DATA_CHECKOUT_COMMIT')
    for name in (*AUTOMATION_FILES, 'english_final_embedding_infer.py', 'package_final_embedding.py'):
        committed = subprocess.check_output(['git', '-C', str(HERE.parent), 'show',
                                             data_head + ':automation/' + name])
        need(identity(HERE / name)['sha256'] == sha(committed), 'EXACT_COMMITTED_DATA_SOURCE:' + name)
    need(identity(HERE / 'english_final144_embedding_once.py')['sha256'] == result['worker_code_sha256'] and
         identity(HERE / 'final144_runner.py')['sha256'] == evidence['SOURCE_BINDING.json']['code_sha256'],
         'EXACT_DATA_WORKER_AND_SOURCE_READER')
    pins, batch_pins = (read_json(HERE / name) for name in
                        ('final144_training_pins.json', 'final144_execution_pins.json'))
    need(sha(canonical(pins)) == result['training_pins_sha256'] and pins['pipeline_commit'] == code['commit'] and
         sha(canonical(batch_pins)) == evidence['SOURCE_BINDING.json']['execution_pins_sha256'] and
         pins['recipe_sha256'] == evidence['SOURCE_BINDING.json']['recipe_sha256'], 'EXACT_EXECUTION_RECIPE_PINS')
    pip_freeze = training.parent / 'training-pip-freeze.txt'
    need(pip_freeze.is_file(), 'ACTUAL_RUNTIME_FREEZE_REQUIRED')
    pairs['evidence/training-pip-freeze.txt'] = pip_freeze
    originals = {relative: identity(source) for relative, source in pairs.items()}
    need(output.parent.is_dir() and not output.exists() and not output.is_symlink() and
         not any(path.is_symlink() for path in output.parents), 'CREATE_ONLY_BUNDLE_DIRECTORY')
    output.mkdir(mode=0o700)
    for relative, source in pairs.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        need(identity(destination) == originals[relative], 'COPY_IDENTITY')
    model_bytes = weights['bytes'] + sum(size for size, _ in contract.ASSETS.values())
    (output / 'MODEL_CARD.md').write_text(model_card(result, trained, preflight, model_bytes), encoding='utf-8')
    (output / 'requirements.txt').write_text('numpy==1.26.4\nonnx==1.17.0\nonnxruntime==1.20.1\n', encoding='utf-8')
    files = {path.relative_to(output).as_posix(): identity(path)
             for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'schema': contract.SCHEMA, 'status': 'COMPLETE_RESEARCH_BUNDLE', 'test_backend': False,
        'policy': contract.POLICY, 'files': files, 'model_calls_during_packaging': 0,
        'provenance': {'data_repository': 'jiying2007/kws-data', 'data_commit': result['github_sha'],
            'run_id': result['run_id'], 'pipeline_commit': code['commit'],
            'upstream_repository': 'dscripka/openWakeWord', 'upstream_commit': contract.UPSTREAM_COMMIT,
            'upstream_release': 'v0.5.1', 'upstream_notice_present': False,
            'weights_sha256': weights['sha256'], 'head_binary_sha256': binary['sha256'],
            'training_input_binding': result['input_binding'], 'optimizer_updates': 96,
            'arm_scope': result['arm_scope']}, **dict.fromkeys(contract.FALSE_FLAGS, False)}
    contract.write_json(output / 'manifest.json', manifest)
    verified = contract.verify_bundle(output)
    return {'status': 'BUILT_AND_VERIFIED_RESEARCH_BUNDLE', 'files': len(verified['files']),
            'manifest_sha256': identity(output / 'manifest.json')['sha256'], 'model_calls': 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--build', action='store_true')
    for name in ('training-result', 'pipeline-source', 'assets', 'upstream-license', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if not args.build:
        print(canonical({'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
                         'requires': 'actual 96-update result, saved-weight host/ARM parity, fixed assets and sources'}).decode())
        return 0
    need(all(getattr(args, name) is not None for name in
             ('training_result', 'pipeline_source', 'assets', 'upstream_license', 'output')), 'EXPLICIT_PACKAGE_INPUTS')
    print(canonical(build(args)).decode())
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
