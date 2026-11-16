"""Package one real snapshot160 requested-intent weak candidate; no model calls."""
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


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load('_snapshot_package_infer', 'english_final_embedding_infer.py')
legacy = load('_snapshot_package_legacy', 'package_final_embedding.py')
worker = load('_snapshot_package_worker', 'english_snapshot160_train_once.py')
need, read_json, sha = contract.need, contract.read_json, contract.sha
identity, canonical = contract.file_identity, contract.canonical
SUCCESS = worker.SUCCESS
EVIDENCE = legacy.EVIDENCE + ('TRAINING_PINS.json',)
AUTOMATION_FILES = legacy.AUTOMATION_FILES + (
    'terminal144_train_once.py', 'terminal144_training_pins.json',
    'terminal144_observe_once.py', 'terminal144_observation_pins.json',
    'english_snapshot160_train_once.py', 'english_snapshot160_training_pins.json',
    'vosen_schwa16_once.py', 'vosen_schwa16_recipe.py', 'english_vosen_schwa16_pins.json')


def source_evidence(root):
    return worker.saved_source_evidence(root)


def training_evidence(root):
    root = Path(root).absolute()
    need(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)) and
         not (root / 'FAILURE.json').exists(), 'COMPLETED_TRAINING_DIRECTORY_REQUIRED')
    saved = source_evidence(root)
    evidence = {name: read_json(root / name) for name in EVIDENCE}
    need(saved and all(evidence.get(name) == value for name, value in saved.items()), 'SAVED_SOURCE_VALIDATOR')
    result, started = evidence['RESULT.json'], evidence['STARTED.json']
    trained = read_json(root / 'train/RESULT.json')
    need(result.get('status') == SUCCESS and trained.get('status') == 'EXECUTED_WEAK_RESEARCH_ONLY' and
         result.get('optimizer_updates') == trained.get('optimizer_updates') == 96 and
         len(trained.get('losses', [])) == len(trained.get('optimizer_bag_draws', [])) == 96,
         'REAL_COMPLETE_FIXED_96_UPDATES')
    spec = trained.get('spec', {})
    need(all(spec.get(key) == value for key, value in
             {'history': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
              'seed': 20261116, 'optimizer_updates': 96, 'checkpoint_selection': False}.items()) and
         trained.get('saved_replay') and all(0 <= r.get('max_abs_logit_error', -1) <= 1e-4
                                            for r in trained['saved_replay']), 'FIXED_HEAD_AND_SAVED_REPLAY')
    need(all(document.get(k) is False for document in (started, result, trained) for k in contract.FALSE_FLAGS)
         and started.get('test_backend') is False and result.get('test_backend') is False,
         'REQUESTED_INTENT_WEAK_RESEARCH_ONLY')
    need(all(result.get(k) == value for k, value in started.items()) and result.get('source_denominator') == 160
         and result.get('minimum', {}).get('passed') is True
         and evidence['MINIMUM_AFTER_DEDUPLICATION.json'].get('passed') is True
         and evidence['MINIMUM_AFTER_FEATURE_GEOMETRY.json'] == result['minimum'], 'BOTH_MINIMUM_GATES')
    binding, pins, preflight = (evidence[name] for name in
                               ('SOURCE_BINDING.json', 'TRAINING_PINS.json', 'FEATURE_PREFLIGHT.json'))
    need(result.get('source_binding') == binding and binding.get('source_denominator') == 160
         and result.get('original_campaign_status') == binding.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY'
         and result.get('new_batch_status') == binding.get('new_batch_status')
         and result.get('snapshot_sha256') == binding.get('snapshot_sha256') == pins.get('snapshot_sha256'),
         'MIXED_SOURCE_PROVENANCE_NOT_OLD_CAMPAIGN_SUCCESS')
    need(result.get('arm_scope') == 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU' and result.get('new_tts_asr_phone_calls') == 0
         and all(result.get(k) == 'NOT_RUN' for k in ('independent_frr', 'independent_far',
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
        'snapshot_sha256': pins['snapshot_sha256'],
        'snapshot_transport_manifest_sha256': binding['transport_manifest_sha256'],
        'source_binding_sha256': sha(canonical(binding)),
        'selection_sha256': sha(canonical(evidence['SELECTION.json'])),
        'selection_policy_sha256': worker.SELECTION_SHA,
        'old_source_ledger_sha256': worker.OLD_LEDGER_SHA, 'old_selection_sha256': worker.OLD_SELECTION_SHA,
        'new_recipe_sha256': pins['new_recipe_sha256'],
        'new_execution_pins_sha256': pins['new_execution_pins_sha256'],
        'new_worker_sha256': pins['new_worker_sha256'], 'new_recipe_code_sha256': pins['new_recipe_code_sha256'],
        'corpus_sha256': worker.old.CORPUS['sha256'], 'reference_sha256': worker.old.REFERENCE['member_sha256'],
        'pipeline_source_sha256': sha(canonical(preflight['pipeline_source'])),
        'feature_evidence_sha256': identity(root / 'FEATURE_EVIDENCE.json')['sha256'],
        'feature_stream_sha256': identity(root / 'actual-embedding-streams.npz')['sha256'],
        'worker_code_sha256': started['worker_code_sha256'],
        'legacy_worker_sha256': worker.terminal.LEGACY_SHA, 'terminal_worker_sha256': worker.TERMINAL_SHA}
    need(bound == required == trained.get('input_binding') == evidence['TRAINING_STARTED.json'].get('input_binding')
         and all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in required.values()), 'ACTUAL_SNAPSHOT_INPUT_BINDING')
    need(preflight.get('geometry', {}).get('status') == 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED' and
         preflight.get('asset_identities') == {k: list(v) for k, v in contract.ASSETS.items()} and
         preflight['pipeline_source'].get('commit') == worker.PIPELINE_COMMIT and
         sha(canonical(pins)) == result['training_pins_sha256'], 'EXACT_FEATURE_AND_TRAINING_PINS')
    return evidence, trained, weights, binary


def model_card(result, trained, preflight, model_bytes):
    card = legacy.model_card(result, trained, preflight, model_bytes)
    original = 'The source denominator is all 144 generated cells.'
    need(card.count(original) == 1, 'FROZEN_CARD_TEMPLATE')
    binding = result['source_binding']
    text = ("This is an **L1 requested-intent weak research candidate**, not a pronunciation-positive "
        "or formally admitted keyword model. Independent acoustic L2 and physical SSC305 L3 remain NOT_RUN. "
        "The planned denominator is **160 = 144 original + 16 distinct new synthesis requests**, including every failure. "
        "The additional requests reuse the original Aiden/Eric voice families; they add no independent human speakers. "
        "The original 144 campaign remains **INCOMPLETE_NO_RETRY**, with 143 verified original sources. "
        f"The new batch retains status **{binding['new_batch_status']}** and source dispositions "
        f"`{canonical(binding['new_source_dispositions']).decode()}`. Missing or failed observations and whole "
        "duplicate PCM groups are excluded from the derived training view; they are not negative targets.\n\n"
        f"Original source/observer identities: `{canonical(binding['old_source']).decode()}`.\n\n"
        f"Separate new source-stage identities: `{canonical(binding['new_source']).decode()}`.\n\n"
        f"The separate training identity is run {result['run_id']}, commit {result['github_sha']}, attempt 1. "
        "SOURCE_SNAPSHOT.zip preserves the original saved artifacts, semantic snapshot and transport byte manifest. "
        "Both frozen minimum gates passed for this completed fit; no prior failure or source role was rewritten.")
    return card.replace(original, text).replace('source commit ', 'training code commit ', 1)


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
        'licenses/openWakeWord-LICENSE': Path(args.upstream_license), 'licenses/kws-pipeline-LICENSE': pipeline / 'LICENSE',
        'evidence/TRAIN_RESULT.json': training / 'train/RESULT.json',
        'evidence/head-replay-stream.bin': training / 'host-parity/stream.bin',
        'evidence/actual-embedding-streams.npz': training / 'actual-embedding-streams.npz',
        'evidence/SOURCE_SNAPSHOT.zip': training / 'SOURCE_SNAPSHOT.zip',
        'source/librispeech_screen.py': pipeline / 'research/english63_online_c/librispeech_screen.py',
        'source/feature-asset-audit.json': pipeline / 'configs/research/english-embedding-backbone-assets-20261116.json',
        'source/english_snapshot160_package.py': Path(__file__),
        'source/package_final_embedding.py': HERE / 'package_final_embedding.py'}
    for name in contract.RUNTIME_FILES:
        path = pipeline / 'research/english_final_embedding' / name
        need(identity(path)['sha256'] == code['feature_head_files_sha256'].get(name), 'EXACT_TRAINING_RUNTIME')
        pairs['runtime/' + name] = path
    need(identity(pairs['source/librispeech_screen.py'])['sha256'] == code['librispeech_source_helper_sha256'], 'EXACT_LIBRI_HELPER')
    for name, (size, digest) in contract.ASSETS.items():
        need(identity(assets / name) == {'bytes': size, 'sha256': digest}, 'FIXED_FEATURE_ASSET')
        pairs['models/' + name] = assets / name
    need(identity(args.upstream_license) == {'bytes': contract.UPSTREAM_LICENSE[0],
         'sha256': contract.UPSTREAM_LICENSE[1]}, 'ORIGINAL_UPSTREAM_LICENSE')
    pairs.update({'evidence/' + name: training / name for name in EVIDENCE})
    pairs.update({'source/' + name: HERE / name for name in AUTOMATION_FILES})
    head = subprocess.check_output(['git', '-C', str(HERE.parent), 'rev-parse', 'HEAD'], text=True).strip()
    need(head == result['github_sha'], 'EXACT_TRAINING_DATA_COMMIT')
    for name in (*AUTOMATION_FILES, 'english_final_embedding_infer.py', 'package_final_embedding.py', 'english_snapshot160_package.py'):
        committed = subprocess.check_output(['git', '-C', str(HERE.parent), 'show', head + ':automation/' + name])
        need(identity(HERE / name)['sha256'] == sha(committed), 'EXACT_COMMITTED_DATA_SOURCE:' + name)
    pins = read_json(HERE / 'english_snapshot160_training_pins.json')
    need(pins == evidence['TRAINING_PINS.json'] and identity(HERE / 'english_snapshot160_train_once.py')['sha256'] == pins['code_sha256']
         and identity(HERE / 'vosen_schwa16_once.py')['sha256'] == pins['new_worker_sha256']
         and identity(HERE / 'vosen_schwa16_recipe.py')['sha256'] == pins['new_recipe_code_sha256']
         and identity(HERE / 'english_vosen_schwa16_pins.json')['sha256'] == pins['new_execution_pins_sha256'],
         'EXACT_NEW_TRAINING_SOURCE_PINS')
    pairs['evidence/training-pip-freeze.txt'] = training.parent / 'training-pip-freeze.txt'
    originals = {relative: identity(path) for relative, path in pairs.items()}
    need(output.parent.is_dir() and not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_BUNDLE')
    output.mkdir(mode=0o700)
    for relative, path in pairs.items():
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        need(identity(destination) == originals[relative], 'COPY_IDENTITY')
    (output / 'MODEL_CARD.md').write_text(model_card(result, trained, preflight,
        weights['bytes'] + sum(size for size, _ in contract.ASSETS.values())), encoding='utf-8')
    (output / 'requirements.txt').write_text('numpy==1.26.4\nonnx==1.17.0\nonnxruntime==1.20.1\n', encoding='utf-8')
    files = {path.relative_to(output).as_posix(): identity(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'schema': contract.SCHEMA, 'status': 'COMPLETE_RESEARCH_BUNDLE', 'test_backend': False,
        'policy': contract.POLICY, 'files': files, 'model_calls_during_packaging': 0,
        'provenance': {'data_repository': 'jiying2007/kws-data', 'data_commit': result['github_sha'],
            'run_id': result['run_id'], 'run_attempt': result['run_attempt'], 'pipeline_commit': code['commit'],
            'source_denominator': 160, 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
            'source_snapshot': result['source_binding'], 'training_input_binding': result['input_binding'],
            'upstream_repository': 'dscripka/openWakeWord', 'upstream_commit': contract.UPSTREAM_COMMIT,
            'upstream_release': 'v0.5.1', 'upstream_notice_present': False,
            'weights_sha256': weights['sha256'], 'head_binary_sha256': binary['sha256'],
            'optimizer_updates': 96, 'arm_scope': result['arm_scope']}, **dict.fromkeys(contract.FALSE_FLAGS, False)}
    contract.write_json(output / 'manifest.json', manifest)
    verified = contract.verify_bundle(output)
    return {'status': 'BUILT_AND_VERIFIED_SNAPSHOT160_WEAK_RESEARCH_BUNDLE',
            'files': len(verified['files']), 'manifest_sha256': identity(output / 'manifest.json')['sha256'], 'model_calls': 0}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--build', action='store_true')
    for name in ('training-result', 'pipeline-source', 'assets', 'upstream-license', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if not args.build:
        print('{"status":"SOURCE_CONTRACT_ONLY","model_calls":0}'); return 0
    need(all(getattr(args, name) is not None for name in
             ('training_result', 'pipeline_source', 'assets', 'upstream_license', 'output')), 'EXPLICIT_PACKAGE_INPUTS')
    import json
    print(json.dumps(build(args), sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
