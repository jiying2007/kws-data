"""Compose one actual snapshot160 head with the already verified mel512 C frontend.

Default is source-only. A new public Actions first-main push may build twice and
run six C processes. The three ORT reference traces are saved evidence, never new
ORT inference. Artificial composition checks cannot establish acoustic accuracy.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data/automation'
FULL_HELPER_SHA = '30dd4b20671bb27ef7907e3c3d5c6d96635b70acabac0d0b131690eafe6e87e7'
MEL_HELPER_SHA = 'fbce05b6364402f3eaa16be54cea4a97a445188062b1518da855266c4890b9ba'
PIPELINE_COMMIT = '6ed95e5ffcb680d2b9c68778c14289025eeefebd'
CANDIDATE_SHA = '5a476954887b59032ea5d419e97d6fb913695e3ad99f5d09164513a4cbd87a89'
SCHEMA = 'english-snapshot160-fullchain-saved-reference-v1'
PROFILE = 'SNAPSHOT160_HEAD_WITH_VERIFIED_MEL512_SAVED_REFERENCE'
REPOSITORY = 'jiying2007/kws-data'
TRAINING_SUCCESS = 'EXECUTED_SNAPSHOT160_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'
PIN_FILE = HERE / 'snapshot160_frontend_pins.json'
MEL = {'run_id': '43836587076', 'github_sha': PIPELINE_COMMIT, 'run_attempt': 1,
    'artifact_id': 14211845823, 'artifact_name': 'english-mel-specialization-43836587076',
    'zip_bytes': 55470166, 'zip_sha256': '6d27ec26868a545fa5e2c930eb03fbd20173961c605656cfe033f653988a553f',
    'result_member': 'mel-specialization-result/RESULT.json',
    'result_sha256': 'a02142d7f8b37b5207bd1fab152f10f196f3822025b86e9c5dae5b60e9bdb9d9',
    'manifest_sha256': 'fe8cbead0b052bbcf542467e2a7677fc740306d3c79115e0a06c0e409922629d',
    'zip_members': 62, 'manifest_files': 50}
PACKAGE_FILES = ('english_snapshot160_package.py', 'english_snapshot160_train_once.py')
TRAINING_SUPPORT = dict(zip(PACKAGE_FILES, (
    '8e0c3dd1c7a9f981c6ae0fe56e4723d068b040cebc6822306355015222940ffb',
    'bb986a78f25247a14af3d177f5c6b504b9ea02acb9f550e3a79adcf9139aff7a')))
UPSTREAM_LICENSE = {'bytes': 11357, 'sha256': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}
PIPELINE_LICENSE = {'bytes': 9108, 'sha256': '06d4c76347e397987c4951179b1c65fbd17bc948e09b5617b38be02bc1f93eab'}


def load_checked(name, path, digest):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or \
            hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('FROZEN_SOURCE:' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = load_checked('_snapshot160_full_trace_contract', AUTOMATION / 'verify_terminal_frontend_once.py', FULL_HELPER_SHA)
need, sha, canonical, regular, once = full.need, full.sha, full.canonical, full.regular, full.once
read_json, hex_id = full.read_json, full.hex_id
FALSE_FLAGS = (*full.FALSE_FLAGS, 'kws_model_delivered', 'ssc305_qualification')
CASES, BACKENDS = ('zero', 'sine400', 'lcg14'), ('native', 'arm')
PLAN = {**copy.deepcopy(full.PLAN), 'protocol': PROFILE,
    'scope': 'THREE_FIXED_ARTIFICIAL_SIGNALS_COMPOSITION_ONLY_NO_ACOUSTIC_QUALIFICATION',
    'inherited_full_head_plan_sha256': full.FIXED_PLAN_SHA,
    'reference_input': 'exact saved ORT float32 stages1-3 from nested completed baseline artifact',
    'verified_mel_run_id': MEL['run_id'], 'maximum_reference_onnx_session_runs': 0,
    'historical_reference_onnx_session_runs_not_reexecuted': 156,
    'maximum_build_processes_per_backend': 1, 'maximum_c_processes_per_backend': 3,
    'maximum_reference_head_streams': 9, 'reference_head_embedding_steps': 234,
    'reference_head_ready_outputs': 99,
    'same_backend_frontend_trace_identity': 'BIT_FOR_BIT_STAGES1_2_3_INCLUDING_HEADERS_AND_SIGNED_ZERO',
    'separate_upstream_onnx_downloads': 0, 'new_ort_inference_calls': 0, 'no_retry': True,
    'saved_training_archive_onnx_scope': 'archived ONNX files may be transported and hashed; never loaded or executed',
    'profiling': 'whole host C process including startup and trace I/O; ARM includes QEMU; not SSC305 timing'}
FIXED_PLAN_SHA = '5d92ab35810c5eaa0d1ed4b1957bc54fbdc51d62a5a7ac8ff40400370f3a890a'


def pins_template():
    return {'schema': SCHEMA + '/pins', 'profile': PROFILE, 'enabled': False,
        'main_before': None, 'worker_code_sha256': None, 'plan_sha256': FIXED_PLAN_SHA,
        'pipeline_commit': PIPELINE_COMMIT, 'full_helper_sha256': FULL_HELPER_SHA,
        'mel_helper_sha256': MEL_HELPER_SHA, 'candidate_source_sha256': CANDIDATE_SHA,
        'mel': copy.deepcopy(MEL), 'training_support_sha256': TRAINING_SUPPORT.copy(),
        'head_training': {'run_id': None, 'github_sha': None, 'run_attempt': 1,
            'artifact_id': None, 'artifact_name': None, 'zip_bytes': None, 'zip_sha256': None,
            'result_member': None, 'result_sha256': None, 'weights_member': None, 'weights_sha256': None,
            'head_binary_member': None, 'head_binary_sha256': None, 'snapshot_sha256': None},
        'upstream_license': UPSTREAM_LICENSE.copy(), 'pipeline_license': PIPELINE_LICENSE.copy(),
        'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}


def validate_pins(pins, *, execution=False):
    template = pins_template()
    dynamic = {'enabled', 'main_before', 'worker_code_sha256', 'head_training'}
    need(type(pins) is dict and set(pins) == set(template) and type(pins['enabled']) is bool and
         all(canonical(pins[k]) == canonical(v) for k, v in template.items() if k not in dynamic) and
         sha(canonical(PLAN)) == FIXED_PLAN_SHA, 'FROZEN_FULLCHAIN_PROFILE')
    need(full.FIXED_PLAN_SHA == 'cdb8cce44b3ba88c92afeaf2b1d64124190c46198091f375d036a099964a16fb' and
         sha(canonical(full.PLAN)) == full.FIXED_PLAN_SHA, 'ORIGINAL_FULL_HEAD_PLAN_UNCHANGED')
    need(type(pins['training_support_sha256']) is dict and
         set(pins['training_support_sha256']) == set(PACKAGE_FILES) and
         type(pins['head_training']) is dict and set(pins['head_training']) == set(template['head_training']),
         'EXPLICIT_PENDING_TRAINING_SCHEMA')
    if not execution and not pins['enabled']:
        return
    need(pins['enabled'] is True and hex_id(pins['main_before'], 40) and
         hex_id(pins['worker_code_sha256'], 64), 'FULLCHAIN_DISABLED_OR_MISSING_EXECUTION_PINS')
    support = pins['training_support_sha256']
    need(type(support) is dict and set(support) == set(PACKAGE_FILES) and
         all(hex_id(value, 64) for value in support.values()), 'ACTUAL_TRAINING_SUPPORT_SOURCE_PINS')
    trained = pins['head_training']
    need(type(trained) is dict and set(trained) == set(template['head_training']) and
         type(trained['run_id']) is str and trained['run_id'].isdigit() and
         type(trained['run_attempt']) is int and trained['run_attempt'] == 1 and
         hex_id(trained['github_sha'], 40) and type(trained['artifact_name']) is str and trained['artifact_name'] and
         all(hex_id(trained[key], 64) for key in ('weights_sha256', 'head_binary_sha256', 'snapshot_sha256')),
         'ACTUAL_SAVED_SNAPSHOT160_HEAD_REQUIRED')
    full.artifact_pin(trained)
    parent = PurePosixPath(trained['result_member']).parent
    need(trained['weights_member'] == str(parent / 'train/causal-head.npz') and
         trained['head_binary_member'] == str(parent / 'causal-head.bin'), 'SAVED_HEAD_MEMBER_LAYOUT')


def execution_guard(pins, env, event, worker_sha=None):
    validate_pins(pins, execution=True)
    need(pins['worker_code_sha256'] == sha(regular(Path(__file__), 1024**2)) and
         (worker_sha is None or worker_sha == pins['worker_code_sha256']), 'EXACT_FULLCHAIN_WORKER')
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPOSITORY and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == event.get('ref') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and str(env.get('GITHUB_RUN_ID', '')).isdigit() and
         env['GITHUB_RUN_ID'] != pins['head_training']['run_id'] and hex_id(env.get('GITHUB_SHA'), 40) and
         env['GITHUB_SHA'] not in (pins['head_training']['github_sha'], PIPELINE_COMMIT) and
         event.get('before') == pins['main_before'] and event.get('after') == env['GITHUB_SHA'] != event['before'] and
         event.get('repository', {}).get('full_name') == REPOSITORY and event['repository'].get('private') is False,
         'FIRST_PUBLIC_MAIN_ACTIONS_ATTEMPT_ONE_ONLY')


def api_identity(run, artifact, pin, repository):
    need(str(run.get('id')) == pin['run_id'] and run.get('head_sha') == pin['github_sha'] and
         run.get('run_attempt') == pin['run_attempt'] == 1 and run.get('head_branch') == 'main' and
         run.get('event') == 'push' and run.get('status') == 'completed' and run.get('conclusion') == 'success' and
         run.get('repository', {}).get('full_name') == repository and run['repository'].get('private') is False,
         'SUCCESSFUL_PUBLIC_INPUT_RUN')
    need(artifact.get('id') == pin['artifact_id'] and artifact.get('name') == pin['artifact_name'] and
         artifact.get('size_in_bytes') == pin['zip_bytes'] and artifact.get('digest') == 'sha256:' + pin['zip_sha256'] and
         artifact.get('expired') is False and str(artifact.get('workflow_run', {}).get('id')) == pin['run_id'] and
         artifact['workflow_run'].get('head_sha') == pin['github_sha'] and
         artifact['workflow_run'].get('head_branch') == 'main', 'EXACT_INPUT_ARTIFACT_METADATA')


def mel_helpers(pipeline):
    mel = load_checked('_snapshot160_frozen_mel_reader', Path(pipeline) /
        'research/english_mel_specialization/mel_specialization_once.py', MEL_HELPER_SHA)
    need(sha(canonical(mel.PLAN)) == mel.FIXED_PLAN_SHA and mel.LIMITS ==
         {k: full.LIMITS[k] for k in ('raw_mel', 'normalized_mel', 'embedding')}, 'ORIGINAL_MEL_LIMITS')
    return mel, mel.load_trace_helpers(pipeline)


def read_mel_archive(path, pipeline, temporary):
    """Read the exact completed mel artifact and nested baseline; zero inference."""
    mel, backbone = mel_helpers(pipeline)
    with full.checked_archive(path, MEL) as archive:
        need(len(archive.infolist()) == MEL['zip_members'], 'EXACT_MEL_ARCHIVE_ROSTER')
        prefix = str(PurePosixPath(MEL['result_member']).parent)
        get = lambda name, size=64 * 1024**2: full.archive_bytes(archive, name, size)
        raw = get(MEL['result_member'])
        need(sha(raw) == MEL['result_sha256'], 'ORIGINAL_MEL_RESULT')
        result = full.json_bytes(raw)
        need(result['passed'] is True and result['status'] == 'EXECUTED_NEW_MEL_IMPLEMENTATION_WITHIN_FIXED_LIMITS' and
             result['run_id'] == MEL['run_id'] and result['github_sha'] == MEL['github_sha'] and
             result['plan_sha256'] == mel.FIXED_PLAN_SHA and result['test_backend'] is False,
             'ACTUALLY_VERIFIED_MEL_PROFILE')
        manifest_raw = get(prefix + '/MANIFEST.json')
        need(sha(manifest_raw) == MEL['manifest_sha256'], 'ORIGINAL_MEL_MANIFEST')
        files = full.json_bytes(manifest_raw)['files']
        actual_names = {name[len(prefix) + 1:] for name in archive.namelist()
                        if name.startswith(prefix + '/') and name != prefix + '/MANIFEST.json'}
        need(len(files) == MEL['manifest_files'] and set(files) == actual_names, 'CLOSED_MEL_MANIFEST')
        for name, identity in files.items():
            item = get(prefix + '/' + name)
            need(identity == {'bytes': len(item), 'sha256': sha(item)}, 'MEL_MANIFEST_MEMBER:' + name)
        nested = Path(temporary) / 'original-baseline.zip'
        with nested.open('xb') as stream:
            stream.write(get('mel-specialization-evidence/baseline.zip'))
        saved = mel.read_baseline(nested, backbone)
        need(result['source_binding']['candidate'][mel.CANDIDATE]['sha256'] == CANDIDATE_SHA and
             result['source_binding']['worker'][mel.PROFILE_DIR + '/mel_specialization_once.py']['sha256'] == MEL_HELPER_SHA,
             'VERIFIED_MEL_SOURCE_IDENTITIES')
        for name, value in saved['generated'].items():
            need(get(prefix + '/generated/' + name) == value, 'SAME_ORIGINAL_GENERATED_C')
        for case in CASES:
            entry = saved['cases'][case]
            need(get(prefix + '/' + case + '/input.pcm') == entry['pcm'] and
                 get(prefix + '/' + case + '/reference.bin') == entry['reference'], 'SAME_SAVED_PCM_AND_ORT_REFERENCE')
            entry['mel'] = {}
            for backend in BACKENDS:
                trace = get(prefix + '/' + case + '/' + backend + '.bin', 1024**2)
                need(trace == entry['old'][backend]['trace'], 'VERIFIED_MEL_SAME_BACKEND_BIT_IDENTITY')
                backbone.parse_trace(trace)
                entry['mel'][backend] = trace
        saved.update(mel_result=result, mel_manifest_sha256=sha(manifest_raw))
    return saved, mel, backbone


def extract_training(path, pin, destination):
    """All validator inputs come from one exact ZIP, not an independently edited directory."""
    destination.mkdir(mode=0o700)
    with full.checked_archive(path, pin) as archive:
        for info in archive.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                need(info.file_size <= 256 * 1024**2, 'BOUNDED_TRAINING_MEMBER')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(archive.read(info))
    root = destination / PurePosixPath(pin['result_member']).parent
    for member, digest in (('result_member', 'result_sha256'), ('weights_member', 'weights_sha256'),
                           ('head_binary_member', 'head_binary_sha256')):
        need(sha(regular(destination / pin[member])) == pin[digest], 'PINNED_TRAINING_MEMBER:' + member)
    return root


def source_preflight(args, pins, temporary):
    pipeline = args.pipeline_source.resolve()
    need(subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip() == PIPELINE_COMMIT and
         not subprocess.check_output(['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(),
         'EXACT_CLEAN_VERIFIED_PIPELINE_CHECKOUT')
    data = AUTOMATION.parent
    need(Path(__file__).resolve() == data / 'automation/verify_snapshot160_frontend_once.py' and
         args.pins.resolve() == data / 'automation/snapshot160_frontend_pins.json' and
         subprocess.check_output(['git', '-C', str(data), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['GITHUB_SHA'] and
         not subprocess.check_output(['git', '-C', str(data), 'status', '--porcelain'], text=True).strip(),
         'EXACT_CLEAN_EXECUTION_SOURCE')
    for name, digest in full.SUPPORT.items():
        need(sha(regular(AUTOMATION / name)) == digest, 'FROZEN_LEGACY_TRAINING_SUPPORT')
    for name, digest in pins['training_support_sha256'].items():
        need(sha(regular(AUTOMATION / name)) == digest, 'FROZEN_SNAPSHOT160_TRAINING_SUPPORT')
    for prefix, pin, repository in (('mel', MEL, 'jiying2007/kws-pipeline'), ('training', pins['head_training'], REPOSITORY)):
        api_identity(read_json(getattr(args, prefix + '_run_json')), read_json(getattr(args, prefix + '_artifact_json')), pin, repository)
    saved, mel, backbone = read_mel_archive(args.mel_zip, pipeline, temporary)
    frozen = {**mel.FROZEN_SOURCES, mel.CANDIDATE: CANDIDATE_SHA}
    need(all(sha(regular(pipeline / name)) == digest for name, digest in frozen.items()), 'VERIFIED_C_SOURCE_BYTES')
    training = extract_training(args.training_zip, pins['head_training'], Path(temporary) / 'training')
    archived_onnx = {p.relative_to(Path(temporary) / 'training').as_posix():
        {'bytes': p.stat().st_size, 'sha256': sha(regular(p))}
        for p in sorted((Path(temporary) / 'training').rglob('*.onnx'))}
    package = load_checked('_snapshot160_actual_saved_head', AUTOMATION / 'english_snapshot160_package.py',
                           pins['training_support_sha256']['english_snapshot160_package.py'])
    evidence, trained, weights, binary = package.training_evidence(training)
    result, wanted = evidence['RESULT.json'], pins['head_training']
    need(result.get('schema') == 'english-snapshot160-weak-research-v1' and result.get('status') == TRAINING_SUCCESS and
         all(str(result.get(k)) == str(wanted[k]) for k in ('run_id', 'github_sha', 'run_attempt')) and
         result.get('source_denominator') == 160 and result.get('test_backend') is False and
         result.get('snapshot_sha256') == wanted['snapshot_sha256'] and
         result.get('worker_code_sha256') == pins['training_support_sha256']['english_snapshot160_train_once.py'] and
         weights['sha256'] == wanted['weights_sha256'] and binary['sha256'] == wanted['head_binary_sha256'],
         'REAL_SAVED_SNAPSHOT160_HEAD_BINDING')
    head = pipeline / 'research/english_final_embedding'
    head_hashes = {name: sha(regular(head / name)) for name in full.HEAD_FILES}
    need(head_hashes == evidence['FEATURE_PREFLIGHT.json']['pipeline_source']['feature_head_files_sha256'],
         'SAME_HEAD_IMPLEMENTATION_AS_TRAINING')
    for path, expected in ((args.upstream_license, UPSTREAM_LICENSE), (pipeline / 'LICENSE', PIPELINE_LICENSE)):
        raw = regular(path)
        need({'bytes': len(raw), 'sha256': sha(raw)} == expected, 'ORIGINAL_LICENSE_BYTES')
    binding = {'pipeline_commit': PIPELINE_COMMIT, 'head_training': wanted, 'mel': MEL,
        'training_result': result, 'training_input_binding': result['input_binding'],
        'full_helper_sha256': FULL_HELPER_SHA, 'mel_helper_sha256': MEL_HELPER_SHA,
        'head_sources_sha256': head_hashes, 'frontend_sources_sha256': frozen,
        'training_support_sha256': pins['training_support_sha256'], 'saved_reference_only': True,
        'separate_upstream_onnx_downloads': 0, 'new_ort_inference_calls': 0,
        'saved_training_archive_contains_onnx': bool(archived_onnx), 'saved_training_archive_onnx_members': archived_onnx,
        'new_onnx_session_runs': 0, 'generated': mel.GENERATED,
        'api_files_sha256': {name: sha(regular(getattr(args, name))) for name in
            ('mel_run_json', 'mel_artifact_json', 'training_run_json', 'training_artifact_json')}}
    return training, head, saved, mel, backbone, binding


def frontend_projection(raw):
    """Drop only stage4 records, preserving every original stage1-3 bit and header."""
    full.parse_trace(raw)
    output, position = bytearray(full.TRACE_MAGIC), 8
    for _ in full.SAMPLES:
        for stage, count in full.STAGE_COUNTS.items():
            size = 16 + 4 * count
            if stage != 4:
                output.extend(raw[position:position + size])
            position += size
    return bytes(output)


def build_command(compiler, backend, pipeline, generated, program):
    frontend = Path(pipeline) / 'research/english_final_embedding_frontend'
    head = Path(pipeline) / 'research/english_final_embedding'
    need(backend in BACKENDS, 'FIXED_BUILD_BACKEND')
    return [compiler, '-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
        '-DEF_WITH_HEAD', *(['-static'] if backend == 'arm' else []), '-I', frontend, '-I', head, '-I', generated,
        Path(pipeline) / 'research/english_mel_specialization/fixed_frontend_mel.c', frontend / 'frontend_replay.c',
        Path(generated) / 'fixed_models.c', head / 'english_causal_head.c', '-lm', '-o', program]


def fresh_accounting():
    return {'new_onnx_session_attempts': 0, 'new_onnx_session_completions': 0, 'new_ort_inference_calls': 0,
        'separate_upstream_onnx_downloads': 0, 'saved_training_archive_contains_onnx': None,
        'build_attempts': dict.fromkeys(BACKENDS, 0), 'build_completions': dict.fromkeys(BACKENDS, 0),
        'c_process_attempts': dict.fromkeys(BACKENDS, 0), 'c_process_completions': dict.fromkeys(BACKENDS, 0),
        'c_frontend_windows_completed': dict.fromkeys(BACKENDS, 0), 'c_head_ready_outputs': dict.fromkeys(BACKENDS, 0),
        'reference_head_stream_attempts': 0, 'reference_head_stream_completions': 0,
        'reference_head_embedding_steps': 0, 'reference_head_ready_outputs': 0,
        'optimizer_updates': 0, 'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}


def attach_once(rows, portable, weights, accounting, receipt):
    need(accounting['reference_head_stream_attempts'] < 9, 'NINE_HEAD_REFERENCE_STREAMS_MAXIMUM')
    accounting['reference_head_stream_attempts'] += 1
    once(receipt, {'operation': 'ATTACH_SAVED_HEAD_TO_SAVED_EMBEDDINGS', 'attempt': 1, 'embeddings': 26,
                   'retry_count': 0, 'new_onnx_session_runs': 0})
    value = full.attach_head(rows, portable, weights)
    need(len(value) == 26 and sum(row[4][0] == 1 for row in value) == 11, 'FIXED_HEAD_STREAM_COUNTS')
    accounting['reference_head_stream_completions'] += 1
    accounting['reference_head_embedding_steps'] += 26
    accounting['reference_head_ready_outputs'] += 11
    return value


def output_manifest(output):
    files = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(regular(p))}
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    once(output / 'MANIFEST.json', {'schema': SCHEMA, 'profile': PROFILE, 'plan': PLAN,
        'plan_sha256': FIXED_PLAN_SHA, 'files': files, **dict.fromkeys(FALSE_FLAGS, False)})


def execute(args):
    pins = read_json(args.pins)
    execution_guard(pins, os.environ, read_json(os.environ.get('GITHUB_EVENT_PATH', '')))
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and not any(p.is_symlink() for p in (output, *output.parents)),
         'FRESH_CREATE_ONLY_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'profile': PROFILE, 'run_id': os.environ['GITHUB_RUN_ID'],
        'github_sha': os.environ['GITHUB_SHA'], 'run_attempt': '1', 'plan': PLAN, 'plan_sha256': FIXED_PLAN_SHA,
        'worker_code_sha256': sha(regular(Path(__file__))),
        'pins_sha256': sha(canonical(pins)), 'test_backend': False, 'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    once(output / 'EXECUTION_PINS.json', pins)
    accounting, reports, stage = fresh_accounting(), [], 'saved-input-preflight'
    def stop(signum, frame):
        raise TimeoutError('FIXED_TOTAL_DEADLINE_OR_TERMINATION')
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(PLAN['total_timeout_seconds'])
    try:
        with tempfile.TemporaryDirectory(prefix='snapshot160-composition-') as temporary:
            training, head, saved, mel, backbone, binding = source_preflight(args, pins, temporary)
            accounting['saved_training_archive_contains_onnx'] = binding['saved_training_archive_contains_onnx']
            once(output / 'SOURCE_BINDING.json', binding)
            for name, path in {'causal-head.npz': training / 'train/causal-head.npz',
                               'causal-head.bin': training / 'causal-head.bin'}.items():
                with (output / name).open('xb') as stream:
                    stream.write(regular(path))
            for directory in ('generated', 'source', 'licenses'):
                (output / directory).mkdir(mode=0o700)
            for name, raw in saved['generated'].items():
                (output / 'generated' / name).write_bytes(raw)
            frontend = args.pipeline_source / mel.FRONTEND_DIR
            for path in (args.pipeline_source / mel.CANDIDATE, frontend / 'fixed_frontend.h',
                         frontend / 'frontend_replay.c', head / 'english_causal_head.c', head / 'english_causal_head.h'):
                (output / 'source' / path.name).write_bytes(regular(path))
            (output / 'licenses/openWakeWord-LICENSE').write_bytes(regular(args.upstream_license))
            (output / 'licenses/kws-pipeline-LICENSE').write_bytes(regular(args.pipeline_source / 'LICENSE'))
            tools = {name: shutil.which(name) for name in ('gcc', 'arm-linux-gnueabihf-gcc', 'qemu-arm')}
            need(all(tools.values()), 'FIXED_COMPILERS_AND_QEMU')
            programs, build = {}, {}
            for backend, compiler, machine in (('native', 'gcc', 62), ('arm', 'arm-linux-gnueabihf-gcc', 40)):
                stage = 'build:' + backend
                program = output / ('frontend-' + backend)
                command = build_command(tools[compiler], backend, args.pipeline_source, output / 'generated', program)
                mel.reserve_attempt(accounting, 'build', backend)
                once(output / ('compile-' + backend + '-started.json'), {'attempt': 1, 'no_retry': True,
                    'command': [str(x) for x in command], 'head_binary_sha256': pins['head_training']['head_binary_sha256']})
                build[backend] = mel.timed_process(command, PLAN['compile_timeout_seconds'], output / ('compile-' + backend + '.json'))
                accounting['build_completions'][backend] += 1
                raw = regular(program)
                need(len(raw) >= 20 and raw[:4] == b'\x7fELF' and raw[5] == 1 and
                     int.from_bytes(raw[18:20], 'little') == machine, 'EXPECTED_EXECUTABLE_ELF')
                build[backend]['executable_identity'] = {'bytes': len(raw), 'sha256': sha(raw)}
                programs[backend] = [str(program)] if backend == 'native' else [tools['qemu-arm'], str(program)]
            build['head_compiled_or_linked'] = True
            build['tool_versions'] = {k: subprocess.check_output([v, '--version'], text=True, timeout=10).splitlines()[0] for k, v in tools.items()}
            once(output / 'BUILD.json', build)
            import numpy as np
            need(np.__version__ == '1.26.4' and not any(name in sys.modules for name in ('torch', 'onnxruntime', 'onnx')),
                 'NUMPY_ONLY_HEAD_REFERENCE_RUNTIME')
            portable = load_checked('_snapshot160_portable_head', head / 'portable_head.py', binding['head_sources_sha256']['portable_head.py'])
            weights = portable.load_weights(output / 'causal-head.npz')
            need(sha(portable.export_blob(weights)) == pins['head_training']['head_binary_sha256'], 'NPZ_AND_C_HEAD_BINARY_AGREE')
            once(output / 'RUNTIME.json', {'numpy': np.__version__, 'python': sys.version, 'onnxruntime_imported': False,
                'new_onnx_session_runs': 0, 'head_reference_stream_budget': 9})
            for case in CASES:
                directory = output / case
                directory.mkdir(mode=0o700)
                item = saved['cases'][case]
                (directory / 'input.pcm').write_bytes(item['pcm'])
                (directory / 'saved-ort-backbone.bin').write_bytes(item['reference'])
                stage = 'saved-reference-head:' + case
                reference = attach_once(backbone.parse_trace(item['reference']), portable, weights, accounting,
                                        directory / 'reference-head-started.json')
                reference_raw = full.trace_bytes(reference)
                (directory / 'reference.bin').write_bytes(reference_raw)
                reference = full.parse_trace(reference_raw)
                for backend in BACKENDS:
                    stage = backend + ':' + case
                    trace = directory / (backend + '.bin')
                    mel.reserve_attempt(accounting, 'c_process', backend)
                    once(directory / (backend + '-started.json'), {'attempt': 1, 'no_retry': True, 'backend': backend,
                        'pcm_sha256': sha(item['pcm']), 'head_binary_sha256': pins['head_training']['head_binary_sha256']})
                    timing = mel.timed_process([*programs[backend], directory / 'input.pcm', trace,
                        ','.join(map(str, full.CHUNKS)), output / 'causal-head.bin'], PLAN['replay_timeout_seconds'],
                        directory / (backend + '-timing.json'))
                    accounting['c_process_completions'][backend] += 1
                    raw = regular(trace, 1024**2)
                    actual = full.parse_trace(raw)
                    summary = full.replay_summary(timing['stdout'], actual)
                    same_frontend = frontend_projection(raw) == item['mel'][backend]
                    whole = full.comparisons(reference, actual)
                    attached = attach_once(actual, portable, weights, accounting, directory / (backend + '-same-embedding-head-started.json'))
                    strict = full.head_comparison(attached, actual, strict=True)
                    state = summary['workspace_bytes'] == 528384 and summary['frontend_state_bytes'] == (32080 if backend == 'native' else 32056)
                    accounting['c_frontend_windows_completed'][backend] += 26
                    accounting['c_head_ready_outputs'][backend] += 11
                    report = {'case': case, 'backend': backend, 'pcm_sha256': sha(item['pcm']), 'embeddings': 26,
                        'ready_head_outputs': 11, 'same_backend_verified_mel_frontend_bitexact': same_frontend,
                        'unchanged_frontend_workspace_and_state': state, 'saved_ort_backbone_plus_new_head': whole,
                        'head_with_same_c_embeddings': strict, 'process_summary': summary, 'timing': timing,
                        'passed': same_frontend and state and whole['passed'] and strict['passed']}
                    reports.append(report)
                    once(directory / (backend + '-comparison.json'), report)
            need(len(reports) == 6 and accounting['reference_head_stream_attempts'] == accounting['reference_head_stream_completions'] == 9 and
                 accounting['reference_head_embedding_steps'] == 234 and accounting['reference_head_ready_outputs'] == 99 and
                 all(accounting['build_attempts'][b] == accounting['build_completions'][b] == 1 and
                     accounting['c_process_attempts'][b] == accounting['c_process_completions'][b] == 3 and
                     accounting['c_frontend_windows_completed'][b] == 78 and accounting['c_head_ready_outputs'][b] == 33 for b in BACKENDS),
                 'EXACT_COMPOSITION_BUDGET')
            passed = all(r['passed'] for r in reports)
            (output / 'RUN_C.md').write_text('# Fixed English C research chain\n\n'
                'Input is 16 kHz mono signed PCM16LE. No resampling or EOF padding occurs.\n\n'
                'Native: `./frontend-native INPUT.pcm TRACE.bin 1,1279,257,4096 causal-head.bin`\n\n'
                'ARM/QEMU: `qemu-arm ./frontend-arm INPUT.pcm TRACE.bin 1,1279,257,4096 causal-head.bin`\n\n'
                'TRACE.bin must not exist. The trace contains per-window stages1–3 and stage4 '
                '[ready,event,logits3,probabilities3]. Event IDs: 0 none, 1 Hey Vosen, 2 Hey Mero. '
                'These are uncalibrated research proposals. A fixed three-signal numerical comparison '
                'does not establish pronunciation, FRR/FAR, latency or physical SSC305 performance. '
                'Inspect RESULT.json before relying on a claimed composition pass.\n', encoding='utf-8')
            signal.alarm(0)
            once(output / 'RESULT.json', {**started, 'status': 'EXECUTED_FULLCHAIN_WITHIN_FIXED_LIMITS' if passed else
                'EXECUTED_FULLCHAIN_WITH_MISMATCH', 'passed': passed, 'accounting': accounting,
                'source_binding': binding, 'comparisons': reports, 'reference_frontend_reexecuted': False,
                'arm_scope': 'FULL_FIXED_FRONTEND_AND_SAVED_SNAPSHOT160_HEAD_UNDER_HOST_QEMU_ONLY',
                'independent_frr': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'word_end_latency': 'NOT_RUN', 'ssc305': 'NOT_RUN'})
            output_manifest(output)
            return 0 if passed else 2
    except BaseException as error:
        signal.alarm(0)
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': str(error)[:2000], 'accounting': accounting,
            'completed_comparisons': reports})
        output_manifest(output)
        return 2
    finally:
        signal.alarm(0)
        for sig, previous in handlers.items():
            signal.signal(sig, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pins', type=Path, default=PIN_FILE)
    for name in ('pipeline-source', 'training-zip', 'mel-zip', 'training-run-json', 'training-artifact-json',
                 'mel-run-json', 'mel-artifact-json', 'upstream-license', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        pins = read_json(args.pins) if args.pins.exists() else pins_template()
        validate_pins(pins)
        print(json.dumps({'status': 'SOURCE_CONTRACT_ONLY', 'plan_sha256': FIXED_PLAN_SHA,
            'new_model_calls': 0, 'new_c_processes': 0, 'enabled': pins['enabled']}, sort_keys=True))
        return 0
    need(all(getattr(args, name.replace('-', '_')) is not None for name in
         ('pipeline-source', 'training-zip', 'mel-zip', 'training-run-json', 'training-artifact-json',
          'mel-run-json', 'mel-artifact-json', 'upstream-license', 'output')), 'EXPLICIT_INPUT_PATHS_REQUIRED')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
