"""Bind one retained coverage head to the immutable English research SDK.

All operations in this file use the Python standard library. No inference,
compiler, subprocess, download, training, or dynamic source import is performed.
The caller must supply post-run byte pins for the actual artifact and independent
audit. A rejected experiment is never a packageable candidate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

SCHEMA = 'english-coverage-package-inputs-v1'
BUNDLE_SCHEMA = 'english-final-embedding-offline-bundle-v1'
TRAINING_ZIP = {'bytes': 61290521, 'sha256': '2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1'}
BASE_PREFIX = 'english-snapshot160-embedding-research-bundle/'
BASE_MANIFEST_SHA = '1dda5e4504cbff66b0b373e6583501e7d61aff3f69ec0167ce188d97b779eaa2'
PIPELINE_COMMIT = '31238cafe3d6668874d446eae5dfa7c55028fb13'
UPSTREAM_COMMIT = '368c03716d1e92591906a84949bc477f3a834455'
POLICY = {'sample_rate': 16000, 'embedding_hop_samples': 1280, 'first_embedding_sample': 16000,
          'history_embeddings': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
          'first_head_sample': 35200, 'threshold': 0.8, 'confirm_embeddings': 3,
          'cooldown_embeddings': 13, 'eof_padding_samples': 0, 'dynamic_threshold': False}
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
BASE_FILES = {
    'english_final_embedding_infer.py': (14910, '479971406dd1277c4008f6b795571b55145480334ef477742ad7151e90f9498a'),
    'requirements.txt': (47, 'ee51a74b48a0853bc41df666170bee6da06bb2eb1174143f4355b12616d07fca'),
    'licenses/kws-pipeline-LICENSE': (9108, '06d4c76347e397987c4951179b1c65fbd17bc948e09b5617b38be02bc1f93eab'),
    'licenses/openWakeWord-LICENSE': (11357, 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
    'models/embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'),
    'models/melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'),
    'runtime/causal_embedding.py': (6883, 'bb623ca627c0394f900638c90eee3a1ec71ff6901fa59713319fd3132de84f37'),
    'runtime/causal_head.py': (7178, '37a93e253ee29b819ced11ae86d648dd319f7e2ede96ef3f624699656c109b98'),
    'runtime/english_causal_head.c': (4413, 'e1e4a4cffd1758677ef820bd2183ec2ab3ab65d450af4d7eea35f565ebefa551'),
    'runtime/english_causal_head.h': (1913, '557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9'),
    'runtime/head_replay.c': (2273, '36fb7cb1732247bc118a9f5707bae2d642fb6d23192201d535f311faa47a8798'),
    'runtime/portable_head.py': (8227, 'f6117a27a67d8b0b5b263c2db316b5825af62d10609f18bfe8488b52f79bef44'),
    'runtime/weak_audio_cases.py': (10405, '345a3361d4aa4f1dd1c0aa6b29454e4e7924f515666daaf0059f6a54dca011d7'),
    'runtime/weak_embedding_contract.py': (10513, 'ba9567b0303b1883f4d4834fec2b499abb8697b8fe5773519f07105f5d55388c'),
}
SHAPES = {'mean96': (96,), 'scale96': (96,), 'w1': (32, 1536),
          'b1': (32,), 'w2': (3, 32), 'b2': (3,)}
C_SOURCE_SHA = {name: BASE_FILES['runtime/' + name][1] for name in
                ('english_causal_head.c', 'english_causal_head.h', 'head_replay.c')}
ARTIFACT_EVIDENCE = ('evidence/PLAN.json', 'evidence/CANDIDATE_FROZEN.json', 'evidence/PHASES.json', 'evidence/RESULT.json',
    'evidence/RESEARCH_RETENTION.json', 'evidence/PAIRED_COMPARISONS.json', 'evidence/INPUTS.json', 'evidence/WORKER_MANIFEST.json',
    'evidence/TRAINING.json', 'evidence/HOST_PARITY.json', 'evidence/ARM_PARITY.json',
    'runtime/head-stream.bin', 'runtime/reference-frames.jsonl', 'runtime/CASE_MAP.json',
    'runtime/head-native', 'runtime/head-arm',
    'evidence/build-head-native.exit.json', 'evidence/build-head-arm.exit.json',
    'evidence/head-native.exit.json', 'evidence/head-arm.exit.json',
    'evidence/host-compiler-version.stdout', 'evidence/arm-compiler-version.stdout', 'evidence/qemu-version.stdout',
    'evidence/host-compiler-version.exit.json', 'evidence/arm-compiler-version.exit.json', 'evidence/qemu-version.exit.json',
    'evidence/head-native.stdout', 'evidence/head-arm.stdout')
HERE = Path(__file__).resolve().parent


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode() + b'\n'


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(raw):
    need(isinstance(raw, bytes) and len(raw) <= 32 * 1024**2, 'JSON_SIZE_LIMIT')
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def file_bytes(path, limit=384 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    return path.read_bytes()


def valid_identity(value):
    return (type(value) is dict and set(value) == {'bytes', 'sha256'} and
            type(value['bytes']) is int and 0 <= value['bytes'] <= 384 * 1024**2 and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']))


def path_ok(name):
    return (type(name) is str and name and '\\' not in name and '\x00' not in name and
            not PurePosixPath(name).is_absolute() and
            all(p not in ('', '.', '..') for p in name.split('/')))


def open_zip(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members = archive.infolist()
    names = [m.filename for m in members]
    need(len(names) == len(set(names)) <= 10000, 'DUPLICATE_OR_EXCESS_ZIP_MEMBERS')
    total = 0
    for member in members:
        name = member.filename.rstrip('/') if member.is_dir() else member.filename
        need(path_ok(name), 'UNSAFE_ZIP_PATH')
        mode = (member.external_attr >> 16) & 0xffff
        need(stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), 'UNSAFE_ZIP_FILE_TYPE')
        need(not member.flag_bits & 1 and member.file_size <= 256 * 1024**2, 'UNSAFE_ZIP_MEMBER')
        total += member.file_size
    need(total <= 1024**3, 'ZIP_EXPANDED_SIZE_LIMIT')
    return archive


def npy_body(raw, expected_shape):
    need(raw[:6] == b'\x93NUMPY' and len(raw) >= 10, 'NPY_MAGIC')
    major, minor = raw[6], raw[7]
    need((major, minor) in ((1, 0), (2, 0), (3, 0)), 'NPY_VERSION')
    length_bytes = 2 if major == 1 else 4
    header_start = 8 + length_bytes
    header_size = int.from_bytes(raw[8:header_start], 'little')
    need(0 < header_size <= 65536 and header_start + header_size <= len(raw), 'NPY_HEADER_SIZE')
    header = ast.literal_eval(raw[header_start:header_start + header_size].decode('utf-8' if major == 3 else 'latin1').strip())
    need(type(header) is dict and set(header) == {'descr', 'fortran_order', 'shape'} and
         header['descr'] == '<f4' and header['fortran_order'] is False and
         header['shape'] == expected_shape, 'NPY_F32_C_GEOMETRY')
    body = raw[header_start + header_size:]
    need(len(body) == 4 * math.prod(expected_shape), 'NPY_PAYLOAD_LENGTH')
    need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', body)), 'NPY_NONFINITE')
    return body


def export_head_bytes(npz_raw):
    """Copy six validated LE f32 array payloads into the existing C ABI."""
    with open_zip(npz_raw) as archive:
        need(set(archive.namelist()) == {key + '.npy' for key in SHAPES}, 'EXACT_NPZ_ARRAYS')
        parts = {key: npy_body(archive.read(key + '.npy'), shape) for key, shape in SHAPES.items()}
    need(all(x[0] > 0 for x in struct.iter_unpack('<f', parts['scale96'])), 'NONPOSITIVE_SCALE')
    result = b'EHKWS01\0' + b''.join(parts.values())
    need(len(result) == 197908, 'C_HEAD_ABI_SIZE')
    return result


def require_false_flags(document, name):
    need(all(document.get(key) is False for key in FALSE_FLAGS), 'RESEARCH_FLAGS:' + name)


def validate_retention(result, audit, pins, npz_raw, binary_raw):
    need(result.get('status') == 'COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH' and
         result.get('research_retention', {}).get('passed') is True, 'QUALITY_CANDIDATE_NOT_RETAINED')
    need(result.get('planned_new_development_conditions') == result.get('eligible_new8_conditions') == 27 and
         result.get('research_retention', {}).get('criteria', {}).get(
             'new8_all_27_planned_development_cases_eligible') is True and
         result.get('independent_source_count_increment') == 0 and
         result.get('new8_new_source_holdout') is False, 'COMPLETE_NEW8_SAME_SOURCE_CONDITION_GATE')
    require_false_flags(result, 'RESULT')
    need(result.get('coverage_candidate') == identity(npz_raw) and
         result.get('coverage_candidate_binary') == identity(binary_raw), 'RESULT_NEW_HEAD_BINDING')
    need(str(result.get('run_id')) == str(pins['run_id']) and
         result.get('head_sha') == pins['head_sha'] and
         str(result.get('run_attempt')) == '1', 'ACTUAL_RESULT_RUN_BINDING')
    need(audit.get('schema') == 'english-coverage-independent-audit-v1' and
         audit.get('status') == 'VERIFIED' and
         audit.get('expected_external_binding') is True and
         audit.get('research_retention', {}).get('passed') is True,
         'INDEPENDENT_AUDIT_AND_QUALITY_REQUIRED')
    require_false_flags(audit, 'AUDIT')
    need(audit.get('quality_artifact') == pins['quality_artifact'] and
         str(audit.get('expected_run')) == str(pins['run_id']) and
         audit.get('expected_head') == pins['head_sha'] and
         audit.get('candidate_head') == identity(npz_raw) and
         audit.get('candidate_binary') == identity(binary_raw), 'AUDIT_NEW_ARTIFACT_AND_HEAD_BINDING')
    need(audit.get('auditor_sha256') == pins['auditor_sha256'], 'FROZEN_AUDITOR_CODE_BINDING')
    need(all(audit.get('parity', {}).get(kind, {}).get('passed') is True for kind in ('host', 'arm')),
         'AUDITED_NEW_HOST_AND_ARM_PARITY_REQUIRED')


def parse_reference(raw):
    lines = raw.splitlines()
    need(1 <= len(lines) <= 50000, 'REFERENCE_FRAME_COUNT')
    result = [read_json(line) for line in lines]
    for row in result:
        need(type(row.get('sample')) is int and row['sample'] >= 16000 and
             type(row.get('ready')) is bool and type(row.get('keyword')) is int and
             row['keyword'] in (0, 1, 2), 'REFERENCE_FRAME_HEADER')
        for key in ('logits', 'probabilities'):
            need(type(row.get(key)) is list and len(row[key]) == 3 and
                 all(type(x) in (int, float) and math.isfinite(x) for x in row[key]),
                 'REFERENCE_FINITE_TRIPLET')
        need(all(0 <= p <= 1 for p in row['probabilities']), 'REFERENCE_PROBABILITY_RANGE')
    return result


def validate_parity(kind, report, blobs, npz_raw, binary_raw):
    need(kind in ('host', 'arm'), 'PARITY_KIND')
    need(report.get('schema') == 'english-coverage-' + kind + '-c-head-parity-v1' and
         report.get('passed') is True and report.get('scope') == 'CANDIDATE_CACHED_EMBEDDING_HEAD_ONLY',
         'NEW_HEAD_PARITY_SCOPE:' + kind)
    need(report.get('candidate') == identity(npz_raw) and
         report.get('candidate_binary') == identity(binary_raw), 'PARITY_NEW_HEAD_IDENTITY:' + kind)
    need(report.get('source_hashes') == C_SOURCE_SHA and all(report.get(key) is True for key in
         ('exact_samples', 'exact_ready', 'exact_events')), 'PARITY_FIXED_SOURCE_AND_EVENTS:' + kind)
    need(report.get('threshold') == 0.8 and report.get('confirm_embeddings') == 3 and
         report.get('cooldown_embeddings') == 13 and report.get('c_frontend') == report.get('ssc305') == 'NOT_RUN',
         'PARITY_FIXED_EVENT_AND_HEAD_ONLY_SCOPE')
    backend = 'native' if kind == 'host' else 'arm'
    executable = blobs['runtime/head-' + backend]
    need(report.get('executable') == identity(executable) and len(executable) >= 64 and
         executable[:4] == b'\x7fELF' and executable[5] == 1 and
         executable[4] == (2 if kind == 'host' else 1) and
         struct.unpack_from('<H', executable, 18)[0] == (62 if kind == 'host' else 40),
         'PARITY_ACTUAL_EXECUTABLE_ARCHITECTURE:' + kind)
    for prefix, argv_field in (('build-head-', 'compiler_argv'), ('head-', 'runner_argv')):
        receipt = read_json(blobs['evidence/' + prefix + backend + '.exit.json'])
        need(receipt.get('returncode') == 0 and receipt.get('error') is None and
             receipt.get('process_group_killed') is False and
             receipt.get('process_group_terminated') is False and
             receipt.get('argv') == report.get(argv_field) and
             type(receipt.get('argv')) is list and len(receipt['argv']) >= 3,
             'PARITY_SUCCESSFUL_BUILD_AND_REPLAY_RECEIPTS:' + kind)
    compiler, runner = report['compiler_argv'], report['runner_argv']
    need({'-std=c99', '-O2', '-ffp-contract=off', '-fno-fast-math', '-Wall', '-Wextra', '-Werror'} <= set(compiler),
         'PARITY_STRICT_COMPILER_FLAGS')
    need(Path(compiler[0]).name == ('cc' if kind == 'host' else 'arm-linux-gnueabihf-gcc') and
         Path(runner[0]).name == ('head-native' if kind == 'host' else 'qemu-arm') and
         Path(runner[-2]).name == 'coverage-head.bin' and Path(runner[-1]).name == 'head-stream.bin',
         'PARITY_BACKEND_COMMAND_BINDING')
    version_commands = {'host-compiler-version': ['cc', '--version'],
                        'arm-compiler-version': ['arm-linux-gnueabihf-gcc', '--version'],
                        'qemu-version': ['qemu-arm', '--version']}
    need(report.get('tool_versions') == {name: identity(blobs['evidence/' + name + '.stdout'])
         for name in version_commands}, 'PARITY_TOOL_VERSION_IDENTITIES')
    for name, argv in version_commands.items():
        receipt = read_json(blobs['evidence/' + name + '.exit.json'])
        need(receipt.get('argv') == argv and receipt.get('returncode') == 0 and
             receipt.get('error') is None and receipt.get('process_group_killed') is False and
             receipt.get('process_group_terminated') is False, 'PARITY_TOOL_VERSION_RECEIPT')
    mapping = {'stream_binary': 'runtime/head-stream.bin',
               'reference_frames': 'runtime/reference-frames.jsonl', 'case_map': 'runtime/CASE_MAP.json',
               'raw_stdout': 'evidence/head-' + backend + '.stdout'}
    need(all(report.get(key) == identity(blobs[path]) for key, path in mapping.items()),
         'PARITY_SAVED_BYTES:' + kind)
    refs = parse_reference(blobs[mapping['reference_frames']])
    stream = blobs[mapping['stream_binary']]
    need(stream[:8] == b'EHSTR01\0' and len(stream) == 12 + 396 * len(refs) and
         struct.unpack('<I', stream[8:12])[0] == len(refs), 'PARITY_STREAM_ABI')
    for index, row in enumerate(refs):
        sample, gap = struct.unpack_from('<QI', stream, 12 + index * 396)
        need(sample == row['sample'] and gap in (0, 1), 'PARITY_STREAM_REFERENCE_CLOCK')
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',
             stream[24 + index * 396:408 + index * 396])), 'PARITY_STREAM_NONFINITE')
    case_map = read_json(blobs[mapping['case_map']])
    need(type(case_map) is list and len(case_map) == report.get('cases') and
         len({(case.get('condition'), case.get('id')) for case in case_map}) == len(case_map), 'PARITY_CASE_MAP')
    cursor = origin = 0
    for case_index, case in enumerate(case_map):
        records = case.get('records')
        need(type(records) is int and 16 <= records <= 50000 and
             case.get('first_record') == cursor and case.get('offset') == origin and
             case.get('head_outputs') == records - 15 and
             type(case.get('observed_samples')) is int and
             16000 + (records - 1) * 1280 <= case['observed_samples'] < 16000 + records * 1280,
             'PARITY_CASE_LOCAL_OBSERVED_GEOMETRY')
        for index in range(records):
            need(cursor + index < len(refs), 'PARITY_CASE_MAP_OVERRUN')
            row = refs[cursor + index]
            sample, gap = struct.unpack_from('<QI', stream, 12 + (cursor + index) * 396)
            local = 16000 + 1280 * index
            need(row.get('case_index') == case_index and row.get('id') == case.get('id') and
                 row.get('condition') == case.get('condition') and row.get('local_sample') == local and
                 sample == origin + local and gap == int(index == 0) and row['ready'] is (index >= 15),
                 'PARITY_EXACT_CASE_RESET_AND_TRANSLATED_CLOCK')
        cursor += records
        origin += case['observed_samples']
    need(cursor == len(refs), 'PARITY_COMPLETE_CASE_MAP')
    lines = blobs[mapping['raw_stdout']].decode('ascii').splitlines()
    need(len(lines) == len(refs) == report.get('input_embeddings') and
         sum(row['ready'] for row in refs) == report.get('head_outputs') and
         type(report.get('cases')) is int and 1 <= report['cases'] <= 291, 'PARITY_COUNTS')
    max_logit = max_probability = 0.0
    for line, row in zip(lines, refs):
        tokens = line.split()
        need(len(tokens) == 9 and tokens[1] in ('0', '1') and
             (int(tokens[0]), bool(int(tokens[1])), int(tokens[2])) ==
             (row['sample'], row['ready'], row['keyword']), 'PARITY_RAW_CLOCK_READY_EVENT')
        values = [float(token) for token in tokens[3:]]
        need(all(math.isfinite(x) for x in values), 'PARITY_RAW_NONFINITE')
        max_logit = max(max_logit, *(abs(a - b) for a, b in zip(values[:3], row['logits'])))
        max_probability = max(max_probability, *(abs(a - b) for a, b in zip(values[3:], row['probabilities'])))
    need(max_logit <= 1e-4 and max_probability <= 2e-6, 'PARITY_SAVED_NUMERICAL_DIFFERENCES')
    for key, actual, limit in (('max_abs_logit_error', max_logit, 1e-4),
                               ('max_abs_probability_error', max_probability, 2e-6)):
        declared = report.get(key)
        need(type(declared) in (float, int) and math.isfinite(declared) and 0 <= declared <= limit and
             abs(declared - actual) <= max(1e-12, abs(actual) * 1e-6), 'PARITY_ERROR_RECEIPT')
    return {'passed': True, 'scope': report['scope'], 'input_embeddings': len(refs),
            'head_outputs': sum(row['ready'] for row in refs), 'cases': report['cases'],
            'stream_sha256': digest(stream), 'max_abs_logit_error': max_logit,
            'max_abs_probability_error': max_probability}


def model_card(pins, result):
    return f'''# English coverage research SDK

This package binds the actually retained coverage candidate from public Actions
run {pins['run_id']} at `{pins['head_sha']}` to the unchanged English inference
runtime from pipeline commit `{PIPELINE_COMMIT}`. The original snapshot160 SDK
is a source of runtime bytes, frontend weights and original licenses only.
Its old head, old quality result and old runtime parity are not this candidate's
proof. The coverage artifact SHA256 is `{pins['quality_artifact']['sha256']}`.

Classes are 0 background, 1 Hey Vosen (/heɪ ˈvoʊzən/), 2 Hey Mero
(/heɪ ˈmiːroʊ/). Hey is intended to be light, the first name syllable stressed.
The source intents remain weak research labels, not human pronunciation gold.
The source population adds zero independent source recordings. New SNR and
background pairings are additional conditions of already exposed recordings.
No independent natural FAR/FRR, acoustic word-end latency or SSC305 claim follows.

The saved head has mean96/scale96, Linear(16*96,32), ReLU, Linear(32,3).
NPZ and portable C binary are verified to contain the same six LE f32 payloads.
Both actual native C and ARM C under QEMU were checked against saved Python
probabilities on identical cached embeddings, with identical sample/ready/event
outputs. These checks are head-only. New complete PCM frontend replay and
physical SSC305 acceptance are NOT_RUN. Packaging executes zero models.

The retained source uses the fixed original runtime ABI: 16 kHz mono PCM16;
first embedding at sample 16000, hop 1280 (80 ms), first head at sample 35200.
Threshold is nominal 0.8 (float32 0.800000011920929), confirmation is three
consecutive ready outputs, cooldown is 13 hops, EOF padding is zero. File EOF
is not an acoustic word end; a non-triggered finite file is censored.

## Verify without numerical dependencies

    python3 -I -S -B english_coverage_sdk.py --bundle . --verify-only
    python3 -I -S -B english_final_embedding_infer.py --bundle . --verify-only

## Explicit offline research inference

Install requirements.txt ahead of offline use. No Torch or runtime download is
required by the unchanged inference CLI. Input must already have the required
PCM format; the CLI does not resample, normalize gain or append EOF silence.

    python3 -I -B english_coverage_sdk.py --bundle . --wav /absolute/input.wav --output /absolute/new-result --execute

The output must be a new directory outside the bundle. Default invocation and
verification run no models. Replacing a head or any listed byte invalidates the
manifest; external candidate heads enter only through this artifact-and-audit
bound packager, never an unchecked runtime path override.

## Evidence and licenses

evidence/RESULT.json and evidence/RESEARCH_RETENTION.json preserve the actual
quality decision. evidence/INDEPENDENT_AUDIT.json and PACKAGE_INPUTS.json bind
the independent check to the exact artifact and head. HOST_PARITY.json and
ARM_PARITY.json bind the actual new head to saved stream and replay receipts.
The upstream openWakeWord and downstream project licenses are copied verbatim.
The unchanged frontend assets total 2,414,536 bytes and exceed a 2 MiB model
budget before adding the head. A physical-device performance/quality gate is
still required for product release. All formal admission and promotion flags
remain false; retention here authorizes only this research candidate package.
'''.encode()


def prepare(training_raw, quality_raw, audit_raw, pins_raw):
    pins, audit = read_json(pins_raw), read_json(audit_raw)
    need(pins.get('schema') == SCHEMA and pins.get('runtime_base') == TRAINING_ZIP and
         pins.get('runtime_base_manifest_sha256') == BASE_MANIFEST_SHA and
         re.fullmatch('[0-9a-f]{40}', str(pins.get('head_sha'))) and
         re.fullmatch('[0-9]+', str(pins.get('run_id'))) and
         re.fullmatch('[0-9a-f]{64}', str(pins.get('auditor_sha256'))), 'POSTRUN_PACKAGE_PINS')
    need(valid_identity(pins.get('quality_artifact')) and valid_identity(pins.get('independent_audit')) and
         identity(training_raw) == TRAINING_ZIP and identity(quality_raw) == pins['quality_artifact'] and
         identity(audit_raw) == pins['independent_audit'], 'EXACT_PACKAGE_INPUT_BYTES')
    files = {}
    with open_zip(training_raw) as archive:
        manifest_raw = archive.read(BASE_PREFIX + 'manifest.json')
        need(digest(manifest_raw) == BASE_MANIFEST_SHA, 'FIXED_BASE_MANIFEST')
        manifest = read_json(manifest_raw)
        need(manifest.get('policy') == POLICY and
             manifest.get('provenance', {}).get('pipeline_commit') == PIPELINE_COMMIT,
             'FIXED_BASE_POLICY_AND_SOURCE')
        for path, (size, sha) in BASE_FILES.items():
            raw = archive.read(BASE_PREFIX + path)
            need(identity(raw) == {'bytes': size, 'sha256': sha} == manifest['files'][path],
                 'IMMUTABLE_RUNTIME_BYTE:' + path)
            files[path] = raw
    with open_zip(quality_raw) as archive:
        blobs = {path: archive.read(path) for path in ARTIFACT_EVIDENCE}
        npz_raw = archive.read('models/coverage-head.npz')
        binary_raw = archive.read('models/coverage-head.bin')
        need(export_head_bytes(npz_raw) == binary_raw, 'NEW_NPZ_AND_C_ABI_BYTES')
        for path in ('models/melspectrogram.onnx', 'models/embedding_model.onnx'):
            need(archive.read(path) == files[path], 'CANDIDATE_FRONTEND_UNCHANGED')
        for name, sha in C_SOURCE_SHA.items():
            need(digest(archive.read('sources/' + name)) == sha, 'ACTUAL_C_REPLAY_SOURCE')
    result = read_json(blobs['evidence/RESULT.json'])
    worker_files = read_json(blobs['evidence/WORKER_MANIFEST.json']).get('files', {})
    need(all(worker_files.get(path) == identity(raw) for path, raw in blobs.items()
             if path != 'evidence/WORKER_MANIFEST.json') and
         worker_files.get('models/coverage-head.npz') == identity(npz_raw) and
         worker_files.get('models/coverage-head.bin') == identity(binary_raw), 'ACTUAL_WORKER_MANIFEST_BINDING')
    validate_retention(result, audit, pins, npz_raw, binary_raw)
    need(read_json(blobs['evidence/RESEARCH_RETENTION.json']) == result['research_retention'], 'ACTUAL_RETENTION_COPY')
    proofs = {kind: validate_parity(kind, read_json(blobs['evidence/' + kind.upper() + '_PARITY.json']),
                  blobs, npz_raw, binary_raw) for kind in ('host', 'arm')}
    need(all(proofs['host'][key] == proofs['arm'][key] for key in
             ('stream_sha256', 'cases', 'input_embeddings', 'head_outputs')), 'HOST_ARM_SAME_CASES_AND_STREAM')
    need(proofs['host']['cases'] == 291, 'ALL_RETAINED_CONDITIONS_HAVE_NATIVE_AND_ARM_PROOF')
    files.update({'models/causal-head.npz': npz_raw, 'models/causal-head.bin': binary_raw,
                  'evidence/PACKAGE_INPUTS.json': pins_raw, 'evidence/INDEPENDENT_AUDIT.json': audit_raw,
                  'MODEL_CARD.md': model_card(pins, result)})
    for path, raw in blobs.items():
        destination = ('evidence/' + path if '/' not in path else path)
        if destination.startswith('runtime/'):
            destination = 'evidence/' + destination
        files[destination] = raw
    files['evidence/TRAIN_RESULT.json'] = blobs['evidence/TRAINING.json']
    files['evidence/PARITY.json'] = canonical({'host': read_json(blobs['evidence/HOST_PARITY.json']),
                                            'arm': read_json(blobs['evidence/ARM_PARITY.json'])})
    provenance = {'data_repository': 'jiying2007/kws-data', 'data_commit': pins['head_sha'],
        'run_id': str(pins['run_id']), 'run_attempt': 1, 'pipeline_commit': PIPELINE_COMMIT,
        'runtime_base_archive': TRAINING_ZIP, 'runtime_base_manifest_sha256': BASE_MANIFEST_SHA,
        'upstream_repository': 'dscripka/openWakeWord', 'upstream_commit': UPSTREAM_COMMIT,
        'upstream_release': 'v0.5.1', 'upstream_notice_present': False,
        'weights_sha256': digest(npz_raw), 'head_binary_sha256': digest(binary_raw),
        'quality_artifact': pins['quality_artifact'], 'independent_audit': pins['independent_audit'],
        'independent_source_count_increment': 0, 'arm_scope': 'NEW_CANDIDATE_HEAD_ONLY_UNDER_QEMU',
        'new_full_pcm_frontend_parity': 'NOT_RUN', 'ssc305': 'NOT_RUN',
        'new_head_parity': proofs, 'quality_retention': 'RETAINED_WEAK_RESEARCH_ONLY'}
    return files, provenance


def verify_bundle(directory):
    """Static SDK acceptance including quality, audit, runtime and head ABI."""
    directory = Path(directory).absolute()
    need(directory.is_dir() and not any(p.is_symlink() for p in (directory, *directory.parents)), 'BUNDLE_DIRECTORY')
    manifest = read_json(file_bytes(directory / 'manifest.json', 1024**2))
    need(manifest.get('schema') == BUNDLE_SCHEMA and manifest.get('status') == 'COMPLETE_RESEARCH_BUNDLE' and
         manifest.get('package_schema') == 'english-coverage-research-sdk-v1' and
         manifest.get('policy') == POLICY and manifest.get('test_backend') is False, 'COVERAGE_BUNDLE_SCHEMA')
    require_false_flags(manifest, 'MANIFEST')
    roster = manifest.get('files')
    need(type(roster) is dict and 20 <= len(roster) <= 100 and 'manifest.json' not in roster, 'BUNDLE_ROSTER')
    actual = set()
    for path in directory.rglob('*'):
        need(not path.is_symlink(), 'BUNDLE_SYMLINK')
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    need(actual == set(roster) | {'manifest.json'}, 'BUNDLE_UNLISTED_OR_MISSING_FILE')
    files = {}
    for path, expected in roster.items():
        need(path_ok(path) and valid_identity(expected), 'BUNDLE_FILE_DECLARATION')
        files[path] = file_bytes(directory / path)
        need(identity(files[path]) == expected, 'BUNDLE_CHANGED_BYTE:' + path)
    for path, (size, sha) in BASE_FILES.items():
        need(identity(files[path]) == {'bytes': size, 'sha256': sha}, 'BUNDLE_FROZEN_RUNTIME:' + path)
    npz_raw, binary_raw = files['models/causal-head.npz'], files['models/causal-head.bin']
    need(export_head_bytes(npz_raw) == binary_raw, 'BUNDLE_NPZ_C_ABI')
    pins = read_json(files['evidence/PACKAGE_INPUTS.json'])
    need(identity(files['evidence/INDEPENDENT_AUDIT.json']) == pins['independent_audit'], 'BUNDLE_AUDIT_RAW_IDENTITY')
    result = read_json(files['evidence/RESULT.json'])
    audit = read_json(files['evidence/INDEPENDENT_AUDIT.json'])
    validate_retention(result, audit, pins, npz_raw, binary_raw)
    blobs = {path: files['evidence/' + path if '/' not in path or path.startswith('runtime/') else path]
             for path in ARTIFACT_EVIDENCE}
    proofs = {kind: validate_parity(kind, read_json(blobs['evidence/' + kind.upper() + '_PARITY.json']),
                  blobs, npz_raw, binary_raw) for kind in ('host', 'arm')}
    need(proofs['host']['cases'] == proofs['arm']['cases'] == 291, 'BUNDLE_ALL_291_HEAD_CASES')
    provenance = manifest.get('provenance', {})
    need(provenance.get('weights_sha256') == digest(npz_raw) and
         provenance.get('head_binary_sha256') == digest(binary_raw) and
         provenance.get('new_head_parity') == proofs and
         provenance.get('quality_artifact') == pins['quality_artifact'] and
         provenance.get('independent_audit') == pins['independent_audit'] and
         provenance.get('pipeline_commit') == PIPELINE_COMMIT and
         provenance.get('upstream_commit') == UPSTREAM_COMMIT and
         provenance.get('independent_source_count_increment') == 0 and
         provenance.get('new_full_pcm_frontend_parity') == provenance.get('ssc305') == 'NOT_RUN',
         'BUNDLE_QUALITY_RUNTIME_SCOPE_BINDING')
    return {'status': 'VERIFIED_COVERAGE_RESEARCH_SDK_BYTES', 'files': len(files),
            'manifest_sha256': digest(file_bytes(directory / 'manifest.json')),
            'weights_sha256': digest(npz_raw), 'head_binary_sha256': digest(binary_raw),
            'model_calls': 0, 'ssc305': 'NOT_RUN'}


def build(args):
    files, provenance = prepare(file_bytes(args.training), file_bytes(args.quality),
                                file_bytes(args.audit), file_bytes(args.pins))
    # The package contains this static verifier and a thin explicit-execution CLI.
    files['english_coverage_package.py'] = file_bytes(Path(__file__))
    files['english_coverage_sdk.py'] = file_bytes(HERE / 'english_coverage_sdk.py')
    manifest = {'schema': BUNDLE_SCHEMA, 'package_schema': 'english-coverage-research-sdk-v1',
        'status': 'COMPLETE_RESEARCH_BUNDLE', 'test_backend': False, 'policy': POLICY,
        'model_calls_during_packaging': 0, 'provenance': provenance,
        'files': {path: identity(raw) for path, raw in sorted(files.items())},
        **dict.fromkeys(FALSE_FLAGS, False)}
    output = Path(args.output).absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_BUNDLE')
    output.mkdir(mode=0o700)
    for path, raw in files.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target:
            target.write(raw)
    with (output / 'manifest.json').open('xb') as target:
        target.write(canonical(manifest))
    return verify_bundle(output)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--check-inputs', action='store_true')
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--verify-bundle', type=Path)
    for name in ('training', 'quality', 'audit', 'pins', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.verify_bundle:
        result = verify_bundle(args.verify_bundle)
    elif args.build or args.check_inputs:
        need(all(getattr(args, name) for name in ('training', 'quality', 'audit', 'pins')), 'EXPLICIT_ACTUAL_INPUTS')
        if args.build:
            need(args.output is not None, 'EXPLICIT_OUTPUT')
            result = build(args)
        else:
            files, provenance = prepare(file_bytes(args.training), file_bytes(args.quality),
                                        file_bytes(args.audit), file_bytes(args.pins))
            result = {'status': 'ACTUAL_RETAINED_INPUTS_VERIFIED_NO_OUTPUT', 'model_calls': 0,
                      'files_prepared': len(files), 'provenance': provenance}
    else:
        result = {'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
            'required': ['retained coverage artifact', 'independent audit', 'new native + ARM head parity',
                         'immutable original runtime archive', 'post-run byte pins'],
            'execution_enabled': False}
    print(canonical(result).decode(), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
