"""Package actual terminal-source research weights; default is zero-model.

The original 144-cell generation remains incomplete. Source, observation and
training identities are separate; only the new completed training result can
produce a bundle. No model, source re-rendering or training runs here.
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
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data' / 'automation'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_file('_terminal_package_infer', AUTOMATION / 'english_final_embedding_infer.py')
legacy = load_file('_terminal_package_template', AUTOMATION / 'package_final_embedding.py')
need, read_json, sha = contract.need, contract.read_json, contract.sha
identity, canonical = contract.file_identity, contract.canonical
EVIDENCE = legacy.EVIDENCE + ('SOURCE_LEDGER.json', 'COMBINED_OBSERVATIONS.json',
                            'OBSERVER_BINDING.json', 'TRAINING_PINS.json', 'OBSERVATION_PINS.json')
AUTOMATION_FILES = legacy.AUTOMATION_FILES + (
    'terminal144_train_once.py', 'terminal144_training_pins.json',
    'terminal144_observe_once.py', 'terminal144_observation_pins.json')
SUCCESS = 'EXECUTED_TERMINAL_SOURCE_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'


def source_evidence(root):
    """The worker's pure validator rechecks saved source/observer/selection pins."""
    return load_file('_terminal_saved_source_validator', HERE / 'terminal144_train_once.py').saved_source_evidence(root)


def training_evidence(root):
    """Pure readback; same return shape as the former all-complete packager."""
    root = Path(root).absolute()
    need(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)) and
         not (root / 'FAILURE.json').exists(), 'COMPLETED_TRAINING_DIRECTORY_REQUIRED')
    saved = source_evidence(root)
    evidence = {name: read_json(root / name) for name in EVIDENCE}
    need(saved and all(name in evidence and evidence[name] == value for name, value in saved.items()),
         'SAVED_SOURCE_VALIDATION_READBACK')
    result, started = evidence['RESULT.json'], evidence['STARTED.json']
    trained = read_json(root / 'train/RESULT.json')
    need(result.get('status') == SUCCESS and trained.get('status') == 'EXECUTED_WEAK_RESEARCH_ONLY' and
         result.get('optimizer_updates') == trained.get('optimizer_updates') == 96 and
         len(trained.get('losses', [])) == len(trained.get('optimizer_bag_draws', [])) == 96,
         'REAL_COMPLETE_TERMINAL_FIXED_96_UPDATE_RESULT')
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
    binding, selected, preflight, pins = (evidence[name] for name in
        ('SOURCE_BINDING.json', 'SELECTION.json', 'FEATURE_PREFLIGHT.json', 'TRAINING_PINS.json'))
    need(result.get('source_identity') == binding.get('source_identity') and
         result.get('observer_identity') == binding.get('observer_identity') and
         result.get('original_campaign_status') == binding.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY' and
         result.get('valid_source_count') == binding.get('valid_source_count') and
         result.get('source_dispositions') == binding.get('source_dispositions'), 'THREE_STAGE_SOURCE_COUNTS')
    source, observer = result['source_identity'], result['observer_identity']
    need(source.get('conclusion') == 'failure' and observer.get('test_backend') is False and
         all(document.get('run_attempt') == '1' and str(document.get('run_id', '')).isdigit() and
             re.fullmatch('[0-9a-f]{40}', str(document.get('github_sha', '')))
             for document in (source, observer, result)) and
         len({document['run_id'] for document in (source, observer, result)}) == 3 and
         len({document['github_sha'] for document in (source, observer, result)}) == 3,
         'DISTINCT_SOURCE_OBSERVER_TRAINING_IDENTITIES')
    need(result.get('arm_scope') == 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU' and
         result.get('new_tts_asr_phone_calls') == 0 and
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
        with (root / name / 'stream.bin').open('rb') as source_file:
            header = source_file.read(12)
        need(stream['sha256'] == report.get('stream_binary_sha256') and
             stream['bytes'] == 12 + 396 * report['input_embeddings'] and header[:8] == b'EHSTR01\0' and
             struct.unpack('<I', header[8:])[0] == report['input_embeddings'], 'ACTUAL_REPLAY_STREAM_IDENTITY')
    need(all(reports['host-parity'][key] == reports['arm-parity'][key] for key in
             ('head_binary_sha256', 'stream_binary_sha256', 'events', 'input_embeddings', 'head_outputs')),
         'HOST_ARM_SAVED_INPUT_AND_EVENT_IDENTITY')
    bound = result.get('input_binding', {})
    required = {
        'source_ledger_canonical_sha256': binding.get('source_ledger_canonical_sha256'),
        'source_binding_sha256': sha(canonical(binding)),
        'observation_pins_sha256': sha(canonical(evidence['OBSERVATION_PINS.json'])),
        'combined_observations_sha256': identity(root / 'COMBINED_OBSERVATIONS.json')['sha256'],
        'observer_artifact_manifest_sha256': sha(canonical(evidence['OBSERVER_BINDING.json'])),
        'recipe_sha256': binding.get('recipe_sha256'), 'selection_sha256': sha(canonical(selected)),
        'selection_policy_sha256': selected.get('policy_sha256'),
        'legacy_worker_sha256': pins.get('legacy_worker_sha256'),
        'terminal_observer_code_sha256': pins.get('terminal_observer_code_sha256'),
        'pipeline_source_sha256': sha(canonical(preflight.get('pipeline_source'))),
        'feature_evidence_sha256': identity(root / 'FEATURE_EVIDENCE.json')['sha256'],
        'feature_stream_sha256': identity(root / 'actual-embedding-streams.npz')['sha256'],
        'worker_code_sha256': started.get('worker_code_sha256'),
        'corpus_sha256': evidence['LIBRISPEECH_SOURCES.json'].get('corpus', {}).get('sha256'),
        'reference_sha256': evidence['LIBRISPEECH_SOURCES.json'].get('reference', {}).get('member_sha256')}
    need(bound == trained.get('input_binding') == evidence['TRAINING_STARTED.json'].get('input_binding') and
         all(re.fullmatch('[0-9a-f]{64}', str(value)) and bound.get(key) == value
             for key, value in required.items()) and
         'source_manifest_sha256' not in bound, 'ACTUAL_TERMINAL_TRAINING_INPUT_BINDINGS')
    if 'observation_manifest_sha256' in bound:
        need(bound['observation_manifest_sha256'] == selected.get('observer_artifact_manifest_sha256'),
             'FOUR_SUMMARY_MANIFEST_IS_NOT_THE_FIVE_ARTIFACT_MANIFEST')
    need(sha(canonical(pins)) == result.get('training_pins_sha256') and
         pins.get('code_sha256') == result.get('worker_code_sha256') and
         pins.get('pipeline_commit') == preflight.get('pipeline_source', {}).get('commit') and
         preflight.get('geometry', {}).get('status') == 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED' and
         preflight.get('asset_identities') == {name: list(value) for name, value in contract.ASSETS.items()} and
         re.fullmatch('[0-9a-f]{40}', str(preflight.get('pipeline_source', {}).get('commit'))),
         'REAL_PINNED_FEATURE_PREFLIGHT')
    return evidence, trained, weights, binary


def model_card(result, trained, preflight, model_bytes):
    """Reuse the fixed inference explanation, explicitly replace old provenance."""
    card = legacy.model_card(result, trained, preflight, model_bytes)
    old = 'The source denominator is all 144 generated cells.'
    need(card.count(old) == 1, 'KNOWN_LEGACY_CARD_TEMPLATE')
    source, observer = result['source_identity'], result['observer_identity']
    text = (f"The source denominator is all 144 declared cells, including failed or unavailable rows. "
            f"There are {result['valid_source_count']} verified original audio sources; the preserved "
            f"source dispositions are `{canonical(result['source_dispositions']).decode()}`. "
            "The original campaign remains **INCOMPLETE_NO_RETRY**, not a completed generation campaign.\n\n"
            f"Original generation: run `{source['run_id']}`, commit `{source['github_sha']}`, attempt 1, failure. "
            f"Separate observation: run `{observer['run_id']}`, commit `{observer['github_sha']}`, attempt 1. "
            f"Separate training: run `{result['run_id']}`, commit `{result['github_sha']}`, attempt 1. "
            "The bundle retains the complete source ledger, observer evidence bindings and selection; "
            "failed, invalid, unknown and excluded sources do not become negative examples.")
    return card.replace(old, text).replace('actual saved weights from run ',
        'actual saved weights from training run ', 1).replace('source commit ', 'training code commit ', 1)


def build(args):
    training, pipeline, assets, output = (Path(getattr(args, name)).absolute()
        for name in ('training_result', 'pipeline_source', 'assets', 'output'))
    evidence, trained, weights, binary = training_evidence(training)
    result, preflight = evidence['RESULT.json'], evidence['FEATURE_PREFLIGHT.json']
    code = preflight['pipeline_source']
    actual = subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip()
    need(actual == code['commit'] and not subprocess.check_output(
        ['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(), 'EXACT_CLEAN_PIPELINE_CHECKOUT')
    pairs = {'english_final_embedding_infer.py': AUTOMATION / 'english_final_embedding_infer.py',
        'models/causal-head.npz': training / 'train/causal-head.npz',
        'models/causal-head.bin': training / 'causal-head.bin',
        'licenses/openWakeWord-LICENSE': Path(args.upstream_license),
        'licenses/kws-pipeline-LICENSE': pipeline / 'LICENSE',
        'evidence/TRAIN_RESULT.json': training / 'train/RESULT.json',
        'evidence/head-replay-stream.bin': training / 'host-parity/stream.bin',
        'source/librispeech_screen.py': pipeline / 'research/english63_online_c/librispeech_screen.py',
        'source/feature-asset-audit.json': pipeline / 'configs/research/english-embedding-backbone-assets-20261116.json',
        'source/terminal144_package.py': Path(__file__),
        'source/package_final_embedding.py': AUTOMATION / 'package_final_embedding.py'}
    for name in contract.RUNTIME_FILES:
        path = pipeline / 'research/english_final_embedding' / name
        need(identity(path)['sha256'] == code['feature_head_files_sha256'].get(name), 'EXACT_TRAINING_RUNTIME_SOURCE')
        pairs['runtime/' + name] = path
    need(identity(pairs['source/librispeech_screen.py'])['sha256'] == code['librispeech_source_helper_sha256'],
         'EXACT_LIBRI_SOURCE_HELPER')
    for name, (size, digest) in contract.ASSETS.items():
        pairs['models/' + name] = assets / name
        need(identity(assets / name) == {'bytes': size, 'sha256': digest}, 'FIXED_FEATURE_ASSET')
    need(identity(args.upstream_license) == {'bytes': contract.UPSTREAM_LICENSE[0],
        'sha256': contract.UPSTREAM_LICENSE[1]}, 'ORIGINAL_UPSTREAM_LICENSE')
    pairs.update({'evidence/' + name: training / name for name in EVIDENCE})
    pairs.update({'source/' + name: AUTOMATION / name for name in AUTOMATION_FILES})
    data_head = subprocess.check_output(['git', '-C', str(AUTOMATION.parent), 'rev-parse', 'HEAD'], text=True).strip()
    need(data_head == result['github_sha'], 'EXACT_TRAINING_DATA_CHECKOUT_COMMIT')
    for name in (*AUTOMATION_FILES, 'english_final_embedding_infer.py', 'package_final_embedding.py',
                 'terminal144_package.py'):
        committed = subprocess.check_output(['git', '-C', str(AUTOMATION.parent), 'show',
                                             data_head + ':automation/' + name])
        path = Path(__file__) if name == 'terminal144_package.py' else AUTOMATION / name
        need(identity(path)['sha256'] == sha(committed), 'EXACT_COMMITTED_DATA_SOURCE:' + name)
    pins = read_json(AUTOMATION / 'terminal144_training_pins.json')
    need(pins == evidence['TRAINING_PINS.json'] and
         read_json(AUTOMATION / 'terminal144_observation_pins.json') == evidence['OBSERVATION_PINS.json'] and
         identity(AUTOMATION / 'terminal144_train_once.py')['sha256'] == pins['code_sha256'] and
         identity(AUTOMATION / 'english_final144_embedding_once.py')['sha256'] == pins['legacy_worker_sha256'] and
         identity(AUTOMATION / 'terminal144_observe_once.py')['sha256'] == pins['terminal_observer_code_sha256'],
         'EXACT_COMMITTED_TERMINAL_SOURCE_AND_TRAINING_PINS')
    pip_freeze = training.parent / 'training-pip-freeze.txt'
    need(pip_freeze.is_file(), 'ACTUAL_RUNTIME_FREEZE_REQUIRED')
    pairs['evidence/training-pip-freeze.txt'] = pip_freeze
    originals = {relative: identity(path) for relative, path in pairs.items()}
    need(output.parent.is_dir() and not output.exists() and not output.is_symlink() and
         not any(path.is_symlink() for path in output.parents), 'CREATE_ONLY_BUNDLE_DIRECTORY')
    output.mkdir(mode=0o700)
    for relative, path in pairs.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
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
            'source_identity': result['source_identity'], 'observer_identity': result['observer_identity'],
            'training_identity': {key: result[key] for key in ('run_id', 'run_attempt', 'github_sha')},
            'original_campaign_status': result['original_campaign_status'], 'source_denominator': 144,
            'valid_source_count': result['valid_source_count'], 'source_dispositions': result['source_dispositions'],
            'upstream_repository': 'dscripka/openWakeWord', 'upstream_commit': contract.UPSTREAM_COMMIT,
            'upstream_release': 'v0.5.1', 'upstream_notice_present': False,
            'weights_sha256': weights['sha256'], 'head_binary_sha256': binary['sha256'],
            'training_input_binding': result['input_binding'], 'optimizer_updates': 96,
            'arm_scope': result['arm_scope']}, **dict.fromkeys(contract.FALSE_FLAGS, False)}
    contract.write_json(output / 'manifest.json', manifest)
    verified = contract.verify_bundle(output)
    return {'status': 'BUILT_AND_VERIFIED_TERMINAL_RESEARCH_BUNDLE', 'files': len(verified['files']),
        'manifest_sha256': identity(output / 'manifest.json')['sha256'], 'model_calls': 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--build', action='store_true')
    for name in ('training-result', 'pipeline-source', 'assets', 'upstream-license', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if not args.build:
        print(canonical({'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
            'requires': 'separate source/observer/training identities; actual 96-update saved weights and host/ARM parity'}).decode())
        return 0
    need(all(getattr(args, name) is not None for name in
             ('training_result', 'pipeline_source', 'assets', 'upstream_license', 'output')), 'EXPLICIT_PACKAGE_INPUTS')
    print(canonical(build(args)).decode())
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
