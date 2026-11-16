"""Disabled-by-default, once-guarded verification of a fixed C feature port.

Only a separately pinned public Actions main push may execute real models.
The three artificial PCM signals and numerical limits are fixed before results.
This terminal-source profile changes evidence inputs only, not the numerical plan.
This is not arbitrary-audio equivalence, acoustic qualification, or SSC305 proof.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data' / 'automation'
SCHEMA = 'english-terminal-fixed-frontend-verification-v1'
PINS_SCHEMA = 'english-terminal-fixed-frontend-verification-pins-v1'
INPUT_PROFILE = 'TERMINAL_SOURCE_WITH_DISTINCT_OBSERVER_AND_TRAINING_IDENTITIES'
FIXED_PLAN_SHA = 'cdb8cce44b3ba88c92afeaf2b1d64124190c46198091f375d036a099964a16fb'
PIPELINE_COMMIT = 'cfc69dcd7535ea909403f10a91bcfe26ccc1b331'
TRAINING_SUCCESS = 'EXECUTED_TERMINAL_SOURCE_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'
TRAINING_SCHEMA = 'english-terminal144-embedding-intent-research-v1'
STATIC_SCHEMA = 'english-fixed-frontend-static-export-clip-bound-fix-v2'
SOURCE_IDENTITY = {'run_id': '43816869386', 'github_sha': '8286ee743a2690d168c3e56f5b31ccc64ae490ed',
                   'run_attempt': '1', 'conclusion': 'failure'}
GENERATED_FILES = ('fixed_models.c', 'fixed_models.h', 'export_manifest.json')
MAX_ARTIFACT_ZIP = 256 * 1024**2
FRONTEND_FILES = ('fixed_frontend.h', 'fixed_frontend.c', 'export_fixed_frontend.py',
                  'frontend_replay.c', 'test_fixed_frontend.py', 'README.md')
HEAD_FILES = ('weak_embedding_contract.py', 'weak_audio_cases.py', 'causal_embedding.py',
              'causal_head.py', 'portable_head.py', 'english_causal_head.c',
              'english_causal_head.h', 'head_replay.c')
SUPPORT = {
    'terminal144_package.py': 'dce03049c6ebc3620b54c374cadd348ae17f30a6e1793d7d86e3ea8e65e5adbb',
    'terminal144_train_once.py': '1c5d73568042723da723403665edaa1a9569f3b056fee080346ae3677cd68bce',
    'package_final_embedding.py': '641785a7203e7fcd641d27dc37544c7f5b0c92b9453aadfa41a6898c6fb9429a',
    'english_final_embedding_infer.py': '479971406dd1277c4008f6b795571b55145480334ef477742ad7151e90f9498a',
}
ASSETS = {
    'melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'),
    'embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'),
}
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
SINE_CYCLE = (0, 1282, 2531, 3719, 4815, 5793, 6627, 7299, 7791, 8091,
              8192, 8091, 7791, 7299, 6627, 5793, 4815, 3719, 2531, 1282,
              0, -1282, -2531, -3719, -4815, -5793, -6627, -7299, -7791, -8091,
              -8192, -8091, -7791, -7299, -6627, -5793, -4815, -3719, -2531, -1282)
PCM_SHA = {
    'zero': '55873fecc61a79e87ca550c7072e38ccdd7ecb600ace286fe4717952a97c42b0',
    'sine400': 'f81681f58003475f0e81e7625f1bf7d76fb018585d3757c0e3083af83c1a8b01',
    'lcg14': '4e97af4846861db16ce7461517d405f60dca4473b515ae1efcb0f505221142dc',
}
CHUNKS = (1, 1279, 257, 4096)
SAMPLES = tuple(range(16000, 48001, 1280))
STAGE_COUNTS = {1: 3104, 2: 2432, 3: 96, 4: 8}
TRACE_MAGIC = b'EFFTR01\0'
LIMITS = {'raw_mel': (0.01, 2e-5), 'normalized_mel': (0.001, 2e-5),
          'embedding': (0.001, 2e-4), 'head_logits': (0.003, 2e-4),
          'head_probabilities': (0.0005, 0.0)}
PLAN = {
    'scope': 'THREE_FIXED_ARTIFICIAL_SIGNALS_ONLY_NOT_ARBITRARY_AUDIO_OR_ACOUSTIC_QUALIFICATION',
    'pcm16_samples_per_case': 48000, 'sample_rate': 16000, 'pcm_sha256': PCM_SHA,
    'signals': {'zero': 'all signed PCM16 zeros',
                'sine400': 'fixed 40-integer cycle, 400Hz, peak8192, repeated1200',
                'lcg14': 'update uint32 seed0x12345678 by1664525*x+1013904223; (x>>18)-8192'},
    'chunk_pattern_samples': list(CHUNKS), 'reset_between_cases': True, 'eof_padding_samples': 0,
    'first_embedding_sample': 16000, 'embedding_hop_samples': 1280, 'first_head_sample': 35200,
    'cases': 3, 'embedding_windows_per_backend': 78, 'ready_head_outputs_per_backend': 33,
    'maximum_reference_onnx_session_runs': 156,
    'error_rule': 'abs(candidate-reference) <= atol + rtol*abs(reference), every finite element',
    'limits': {name: {'atol': a, 'rtol': r} for name, (a, r) in LIMITS.items()},
    'same_embedding_head_limits': {'logits_atol': 1e-4, 'probabilities_atol': 2e-6, 'rtol': 0.0},
    'exact_clock_ready_event_identity': True, 'threshold': 0.8,
    'confirm_embeddings': 3, 'cooldown_embeddings': 13,
    'compile_timeout_seconds': 180, 'replay_timeout_seconds': 300, 'total_timeout_seconds': 1800,
    'rationale': 'Predeclared float32 engineering limits for accumulation-order and layer propagation; '
        'absolute bounds cover near-zero values, mel normalization divides the absolute bound by10, '
        'probability error is at most0.05 percentage points, and decisions must still match exactly. '
        'These bounds are not a mathematical guarantee for arbitrary weights or speech. No widening after results.',
    'optimizer_updates': 0, 'tts_asr_phone_calls': 0, 'real_audio_cases': 0,
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def regular(path, maximum=256 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
         path.stat().st_size <= maximum, 'REGULAR_BOUNDED_FILE:' + str(path))
    raw = path.read_bytes()
    need(len(raw) <= maximum, 'FILE_GREW')
    return raw


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(path):
    return json.loads(regular(path, 32 * 1024**2), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return sha(raw)


def hex_id(value, size):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % size, value) is not None


def pins_template():
    """Known source/plan only; actual training and export receipts stay unknown."""
    return {'schema': PINS_SCHEMA, 'enabled': False, 'main_before': None,
        'pipeline_commit': PIPELINE_COMMIT, 'worker_code_sha256': None,
        'plan_sha256': FIXED_PLAN_SHA, 'input_profile': INPUT_PROFILE, 'no_retry': True,
        'frontend_sources_sha256': None,
        'head_training': {'run_id': None, 'github_sha': None, 'run_attempt': '1',
            'artifact_id': None, 'zip_bytes': None, 'zip_sha256': None,
            'result_member': None, 'result_sha256': None, 'weights_member': None, 'weights_sha256': None,
            'head_binary_member': None, 'head_binary_sha256': None,
            'worker_code_sha256': SUPPORT['terminal144_train_once.py'],
            'source_identity': None, 'observer_identity': None},
        'static_export': {'run_id': None, 'github_sha': None, 'run_attempt': 1,
            'artifact_id': None, 'zip_bytes': None, 'zip_sha256': None,
            'result_member': None, 'result_sha256': None,
            'generated': {name: {'member': None, 'bytes': None, 'sha256': None} for name in GENERATED_FILES}},
        **dict.fromkeys(FALSE_FLAGS, False)}


def member_path(name):
    return (type(name) is str and '\\' not in name and not PurePosixPath(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')))


def artifact_pin(pin):
    need(type(pin.get('artifact_id')) is int and pin['artifact_id'] > 0 and
         type(pin.get('zip_bytes')) is int and 0 < pin['zip_bytes'] <= MAX_ARTIFACT_ZIP and
         hex_id(pin.get('zip_sha256'), 64) and member_path(pin.get('result_member')) and
         PurePosixPath(pin['result_member']).name == 'RESULT.json' and
         hex_id(pin.get('result_sha256'), 64), 'ACTUAL_ARTIFACT_ZIP_AND_RESULT_PINS')


def validate_pins(pins):
    need(type(pins) is dict and pins.get('schema') == PINS_SCHEMA and pins.get('enabled') is True,
         'ACTUAL_FRONTEND_VERIFICATION_DISABLED')
    need(set(pins) == set(pins_template()) and hex_id(pins.get('main_before'), 40) and
         pins.get('pipeline_commit') == PIPELINE_COMMIT and pins.get('input_profile') == INPUT_PROFILE and
         hex_id(pins.get('worker_code_sha256'), 64) and pins.get('plan_sha256') ==
         sha(canonical(PLAN)) == FIXED_PLAN_SHA and
         pins.get('no_retry') is True and all(pins.get(flag) is False for flag in FALSE_FLAGS),
         'COMPLETE_FIXED_EXECUTION_PINS')
    sources = pins.get('frontend_sources_sha256')
    need(type(sources) is dict and set(sources) == set(FRONTEND_FILES) and
         all(hex_id(value, 64) for value in sources.values()), 'FROZEN_FRONTEND_SOURCE_PINS')
    trained = pins.get('head_training', {})
    need(type(trained) is dict and set(trained) == set(pins_template()['head_training']) and
         type(trained.get('run_id')) is str and trained['run_id'].isdigit() and trained.get('run_attempt') == '1' and
         hex_id(trained.get('github_sha'), 40) and all(hex_id(trained.get(key), 64) for key in
             ('result_sha256', 'weights_sha256', 'head_binary_sha256')) and
         trained.get('worker_code_sha256') == SUPPORT['terminal144_train_once.py'], 'ACTUAL_SAVED_HEAD_PINS_REQUIRED')
    artifact_pin(trained)
    need(all(member_path(trained.get(key)) and PurePosixPath(trained[key]).name == filename for key, filename in
             (('weights_member', 'causal-head.npz'), ('head_binary_member', 'causal-head.bin'))),
         'SAVED_WEIGHT_ARCHIVE_MEMBER_PINS')
    source, observer = trained.get('source_identity'), trained.get('observer_identity')
    need(source == SOURCE_IDENTITY and type(observer) is dict and
         set(observer) == {'run_id', 'github_sha', 'run_attempt', 'test_backend'} and
         observer.get('test_backend') is False and observer.get('run_attempt') == '1' and
         type(observer.get('run_id')) is str and observer['run_id'].isdigit() and
         hex_id(observer.get('github_sha'), 40) and
         len({p['run_id'] for p in (source, observer, trained)}) == 3 and
         len({p['github_sha'] for p in (source, observer, trained)}) == 3, 'THREE_DISTINCT_PINNED_INPUT_IDENTITIES')
    exported = pins.get('static_export')
    need(type(exported) is dict and set(exported) == set(pins_template()['static_export']) and
         type(exported.get('run_id')) is str and exported['run_id'].isdigit() and
         exported.get('github_sha') == PIPELINE_COMMIT and type(exported.get('run_attempt')) is int and
         exported['run_attempt'] == 1, 'ACTUAL_STATIC_EXPORT_IDENTITY_REQUIRED')
    artifact_pin(exported)
    generated = exported.get('generated')
    need(type(generated) is dict and set(generated) == set(GENERATED_FILES) and
         all(type(entry) is dict for entry in generated.values()) and
         len({entry.get('member') for entry in generated.values()}) == len(GENERATED_FILES),
         'THREE_EXACT_GENERATED_FILE_PINS')
    for name, entry in generated.items():
        need(type(entry) is dict and set(entry) == {'member', 'bytes', 'sha256'} and
             member_path(entry.get('member')) and PurePosixPath(entry['member']).name == name and
             type(entry.get('bytes')) is int and 0 < entry['bytes'] <= 64 * 1024**2 and
             hex_id(entry.get('sha256'), 64), 'EXACT_GENERATED_MEMBER_PIN')


def execution_guard(pins, env, event, worker_sha):
    validate_pins(pins)
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and re.fullmatch('[0-9]+', env.get('GITHUB_RUN_ID', '')) and
         event.get('before') == pins['main_before'] and event.get('after') == env.get('GITHUB_SHA') and
         hex_id(event.get('after'), 40) and event.get('after') != event.get('before') and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False and worker_sha == pins['worker_code_sha256'],
         'FIRST_PINNED_PUBLIC_MAIN_PUSH_ONLY')
    previous = (pins['head_training'], pins['head_training']['source_identity'],
                pins['head_training']['observer_identity'])
    need(env['GITHUB_RUN_ID'] not in {p['run_id'] for p in previous} and
         env['GITHUB_SHA'] not in {p['github_sha'] for p in previous}, 'NEW_VERIFICATION_AFTER_THREE_INPUT_RUNS')


def synthetic_pcm():
    state, noise = 0x12345678, []
    for _ in range(48000):
        state = (1664525 * state + 1013904223) & 0xffffffff
        noise.append((state >> 18) - 8192)
    values = {'zero': (0,) * 48000, 'sine400': SINE_CYCLE * 1200, 'lcg14': noise}
    result = {name: struct.pack('<48000h', *samples) for name, samples in values.items()}
    need({name: sha(raw) for name, raw in result.items()} == PCM_SHA, 'FIXED_ARTIFICIAL_PCM_IDENTITIES')
    return result


def chunks(raw):
    need(len(raw) == 96000, 'THREE_SECOND_PCM_BYTES')
    offset = index = 0
    while offset < 48000:
        end = min(48000, offset + CHUNKS[index % len(CHUNKS)])
        yield struct.unpack('<%dh' % (end - offset), raw[2 * offset:2 * end])
        offset, index = end, index + 1


def trace_bytes(rows):
    output = bytearray(TRACE_MAGIC)
    for row in rows:
        for stage, count in STAGE_COUNTS.items():
            values = row[stage]
            need(len(values) == count and all(math.isfinite(x) for x in values), 'FINITE_TRACE_VALUES')
            output.extend(struct.pack('<IIQ', stage, count, row['sample']))
            output.extend(struct.pack('<%df' % count, *values))
    return bytes(output)


def parse_trace(raw, samples=SAMPLES):
    row_bytes = sum(16 + 4 * count for count in STAGE_COUNTS.values())
    need(len(raw) == 8 + len(samples) * row_bytes and raw[:8] == TRACE_MAGIC, 'TRACE_LENGTH_OR_MAGIC')
    rows, position = [], 8
    for index, sample in enumerate(samples):
        row = {'sample': sample}
        for stage, count in STAGE_COUNTS.items():
            header = struct.unpack_from('<IIQ', raw, position)
            need(header == (stage, count, sample),
                 'TRACE_STAGE_COUNT_CLOCK:%d:%s!=%s' % (index, header, (stage, count, sample)))
            position += 16
            values = struct.unpack_from('<%df' % count, raw, position)
            position += 4 * count
            need(all(math.isfinite(x) for x in values), 'NONFINITE_C_TRACE')
            row[stage] = values
        need(row[4][0] in (0.0, 1.0) and row[4][1] in (0.0, 1.0, 2.0), 'EXACT_HEAD_DISCRETE_FIELDS')
        rows.append(row)
    return rows


def errors(reference, candidate, samples, atol, rtol):
    need(len(reference) == len(candidate) == len(samples), 'COMPARISON_ROW_COUNT')
    maximum = {'abs': (0.0, None), 'relative': (0.0, None), 'ratio': (0.0, None)}
    violations = count = zero_mismatches = nonzero_count = 0
    for index, (expected, actual, sample) in enumerate(zip(reference, candidate, samples)):
        need(len(expected) == len(actual), 'COMPARISON_ELEMENT_COUNT')
        for component, (a, b) in enumerate(zip(expected, actual)):
            need(math.isfinite(a) and math.isfinite(b), 'FINITE_COMPARISON')
            delta, allowed = abs(b - a), atol + rtol * abs(a)
            count += 1
            violations += int(delta > allowed)
            zero_mismatches += int(a == 0 and delta != 0)
            nonzero_count += int(a != 0)
            where = {'embedding_index': index, 'sample': sample, 'component': component,
                     'reference': a, 'candidate': b}
            values = [('abs', delta), ('ratio', delta / allowed)]
            if a:
                values.append(('relative', delta / abs(a)))
            for key, value in values:
                if maximum[key][1] is None or value > maximum[key][0]:
                    maximum[key] = (value, where)
    return {'atol': atol, 'rtol': rtol, 'elements': count, 'violations': violations,
            'max_abs_error': maximum['abs'][0], 'max_abs_at': maximum['abs'][1],
            'max_relative_error_nonzero_reference': maximum['relative'][0] if nonzero_count else None,
            'max_relative_at': maximum['relative'][1], 'zero_reference_mismatch_count': zero_mismatches,
            'nonzero_reference_elements': nonzero_count,
            'max_tolerance_ratio': maximum['ratio'][0], 'max_tolerance_ratio_at': maximum['ratio'][1]}


def head_comparison(reference, candidate, strict=False):
    need(len(reference) == len(candidate) and
         all(a['sample'] == b['sample'] for a, b in zip(reference, candidate)), 'EXACT_HEAD_CLOCKS')
    clocks = [row['sample'] for row in reference]
    discrete = {name: [{'sample': a['sample'], 'reference': a[4][column], 'candidate': b[4][column]}
                       for a, b in zip(reference, candidate) if a[4][column] != b[4][column]]
                for name, column in (('ready', 0), ('event', 1))}
    metrics = {}
    for name, section, tolerance in (
        ('head_logits', slice(2, 5), (1e-4, 0.0) if strict else LIMITS['head_logits']),
        ('head_probabilities', slice(5, 8), (2e-6, 0.0) if strict else LIMITS['head_probabilities'])):
        metrics[name] = errors([row[4][section] for row in reference],
                               [row[4][section] for row in candidate], clocks, *tolerance)
    expected_events = [{'sample': r['sample'], 'keyword': int(r[4][1])} for r in reference if r[4][1]]
    actual_events = [{'sample': r['sample'], 'keyword': int(r[4][1])} for r in candidate if r[4][1]]
    return {'metrics': metrics, 'ready_mismatches': len(discrete['ready']),
            'event_mismatches': len(discrete['event']), 'first_discrete_mismatches':
                {name: rows[:8] for name, rows in discrete.items()},
            'reference_event_count': len(expected_events), 'candidate_event_count': len(actual_events),
            'reference_events': expected_events, 'candidate_events': actual_events,
            'acoustic_trigger_coverage_established': False,
            'passed': not any(discrete.values()) and all(m['violations'] == 0 for m in metrics.values())}


def comparisons(reference, candidate):
    report = head_comparison(reference, candidate)
    clocks = [row['sample'] for row in reference]
    for stage, name in ((1, 'raw_mel'), (2, 'normalized_mel'), (3, 'embedding')):
        report['metrics'][name] = errors([r[stage] for r in reference], [r[stage] for r in candidate],
                                         clocks, *LIMITS[name])
    report['passed'] = report['passed'] and all(x['violations'] == 0 for x in report['metrics'].values())
    return report


def replay_summary(stdout, rows):
    summary = json.loads(stdout, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    need(summary.get('samples') == 48000 and summary.get('embeddings') == len(rows) == 26 and
         summary.get('head_ready') == sum(row[4][0] == 1 for row in rows) == 11 and
         summary.get('events') == sum(row[4][1] != 0 for row in rows) and
         summary.get('head_enabled') is True and summary.get('ssc305_qualification') is False and
         all(type(summary.get(key)) is int and 0 < summary[key] <= 256 * 1024**2
             for key in ('workspace_bytes', 'frontend_state_bytes')), 'C_PROCESS_SAMPLE_AND_OUTPUT_ACCOUNTING')
    return summary


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_archive(path, pin):
    raw = regular(path, MAX_ARTIFACT_ZIP)
    need(len(raw) == pin['zip_bytes'] and sha(raw) == pin['zip_sha256'], 'EXACT_ARTIFACT_ZIP_BYTES')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members = archive.infolist()
    need(len(members) == len({m.filename for m in members}) and
         sum(m.file_size for m in members) <= 1024**3, 'UNIQUE_BOUNDED_ARTIFACT_MEMBERS')
    for item in members:
        name = item.filename[:-1] if item.is_dir() else item.filename
        need(member_path(name) and not item.flag_bits & 1 and
             stat.S_IFMT(item.external_attr >> 16) in
             ((0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)), 'SAFE_ARTIFACT_MEMBER')
    return archive


def archive_bytes(archive, member, limit):
    info = archive.getinfo(member)
    need(not info.is_dir() and 0 < info.file_size <= limit, 'BOUNDED_REQUIRED_ARCHIVE_MEMBER')
    return archive.read(member)


def json_bytes(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def training_archive(args, pins):
    """Bind the extracted result/weights to the exact retained training artifact."""
    wanted = pins['head_training']
    with checked_archive(args.training_zip, wanted) as archive:
        for member_key, digest_key, local in (
            ('result_member', 'result_sha256', args.training_result / 'RESULT.json'),
            ('weights_member', 'weights_sha256', args.training_result / 'train/causal-head.npz'),
            ('head_binary_member', 'head_binary_sha256', args.training_result / 'causal-head.bin')):
            raw = archive_bytes(archive, wanted[member_key], 32 * 1024**2)
            need(sha(raw) == wanted[digest_key] and raw == regular(local, 32 * 1024**2),
                 'ACTUAL_TRAINING_ARTIFACT_MEMBER:' + member_key)


def static_export_archive(args, pins, frontend_hashes):
    """Return original generated bytes; no parser export or model inference."""
    wanted = pins['static_export']
    with checked_archive(args.static_export_zip, wanted) as archive:
        base = PurePosixPath(wanted['result_member']).parent
        need(str(base / 'FAILURE.json') not in archive.namelist(), 'COMPLETED_STATIC_EXPORT_WITHOUT_FAILURE')
        raw = archive_bytes(archive, wanted['result_member'], 32 * 1024**2)
        need(sha(raw) == wanted['result_sha256'], 'EXACT_STATIC_EXPORT_RESULT_BYTES')
        result = json_bytes(raw)
        need(result.get('schema') == STATIC_SCHEMA and result.get('status') == 'EXPORTED_STATIC_GRAPHS_ONLY' and
             result.get('repository') == 'jiying2007/kws-pipeline' and
             result.get('run_id') == wanted['run_id'] and result.get('run_attempt') == 1 and
             result.get('source_head') == wanted['github_sha'] == pins['pipeline_commit'] and
             result.get('export_attempted') is True and
             result.get('parser_versions') == {'numpy': '1.26.4', 'onnx': '1.17.0'} and
             all(result.get(k) == 0 for k in ('model_calls', 'onnx_inference_calls', 'c_inference_calls',
                                             'optimizer_updates', 'tts_asr_phone_calls')) and
             result.get('numerical_parity_established') is False and result.get('ssc305_qualification') is False,
             'COMPLETED_ZERO_INFERENCE_STATIC_EXPORT_REQUIRED')
        assets = {name: {'bytes': size, 'sha256': digest} for name, (size, digest) in ASSETS.items()}
        need(result.get('assets') == result.get('expected_assets') == assets, 'STATIC_EXPORT_SAME_ASSETS')
        prefix = args.frontend_source.resolve().relative_to(args.pipeline_source.resolve()).as_posix()
        code = result.get('code_files')
        need(type(code) is dict and {prefix + '/' + name for name in FRONTEND_FILES} <= set(code),
             'STATIC_EXPORT_FULL_FRONTEND_SOURCE_BINDING')
        for name, entry in code.items():
            need(member_path(name) and type(entry) is dict and set(entry) == {'bytes', 'sha256'},
                 'STATIC_EXPORT_SOURCE_PATH')
            source_raw = regular(args.pipeline_source / name, 1024**2)
            need(entry == {'bytes': len(source_raw), 'sha256': sha(source_raw)}, 'STATIC_EXPORT_SOURCE_BYTES')
        need(all(code[prefix + '/' + name]['sha256'] == digest for name, digest in frontend_hashes.items()),
             'STATIC_EXPORT_USED_EXACT_FRONTEND')
        generated = {}
        expected = result.get('generated')
        need(type(expected) is dict and set(expected) == {'generated/' + name for name in GENERATED_FILES},
             'STATIC_EXPORT_EXACT_GENERATED_ROSTER')
        for name, entry in wanted['generated'].items():
            need(entry['member'] == str(base / 'generated' / name), 'EXACT_STATIC_GENERATED_MEMBER_PATH')
            content = archive_bytes(archive, entry['member'], 64 * 1024**2)
            identity = {'bytes': len(content), 'sha256': sha(content)}
            need(identity == expected['generated/' + name] ==
                 {k: entry[k] for k in ('bytes', 'sha256')}, 'EXACT_STATIC_GENERATED_BYTES')
            generated[name] = content
    manifest = json_bytes(generated['export_manifest.json'])
    need(manifest.get('schema') == 'fixed-feature-pure-export-v1' and
         manifest.get('real_inference_performed') is False and manifest.get('numerical_parity_established') is False and
         manifest.get('ssc305_qualification') is False and
         manifest.get('generated_c_sha256') == sha(generated['fixed_models.c']) and
         set(manifest.get('graphs', {})) == set(ASSETS) and set(result.get('graphs', {})) == set(ASSETS),
         'ORIGINAL_STATIC_GENERATED_MANIFEST')
    for name, (size, digest) in ASSETS.items():
        graph = manifest['graphs'][name]
        need(graph.get('asset_sha256') == digest and graph.get('asset_bytes') == size and
             result['graphs'][name] == {k: graph[k] for k in
                ('asset_sha256', 'asset_bytes', 'input', 'output', 'constant_bytes', 'workspace_bytes')},
             'STATIC_GRAPH_ASSET_AND_GEOMETRY_BINDING')
    return result, generated


def verify_training_profile(original, wanted):
    dispositions = original.get('source_dispositions')
    need(original.get('schema') == TRAINING_SCHEMA and original.get('status') == TRAINING_SUCCESS and
         original.get('run_id') == wanted['run_id'] and original.get('github_sha') == wanted['github_sha'] and
         original.get('run_attempt') == wanted['run_attempt'] and
         original.get('source_identity') == wanted['source_identity'] and
         original.get('observer_identity') == wanted['observer_identity'] and
         original.get('worker_code_sha256') == wanted['worker_code_sha256'] and
         original.get('source_denominator') == 144 and
         original.get('original_campaign_status') == 'INCOMPLETE_NO_RETRY' and
         type(dispositions) is dict and all(type(n) is int and n > 0 for n in dispositions.values()) and
         sum(dispositions.values()) == 144 and type(original.get('valid_source_count')) is int and
         0 < original['valid_source_count'] == dispositions.get('VALID_UNOBSERVED_SOURCE', 0) <= 144 and
         original.get('test_backend') is False and all(original.get(k) is False for k in FALSE_FLAGS),
         'ACTUAL_TERMINAL_TRAINING_PROFILE')


def source_preflight(args, pins):
    pipeline, frontend = args.pipeline_source.resolve(), args.frontend_source.resolve()
    need(pipeline.is_dir() and frontend.is_relative_to(pipeline), 'FRONTEND_IN_PIPELINE_CHECKOUT')
    head = pipeline / 'research/english_final_embedding'
    commit = subprocess.check_output(['git', '-C', str(pipeline), 'rev-parse', 'HEAD'], text=True).strip()
    need(commit == pins['pipeline_commit'] and not subprocess.check_output(
        ['git', '-C', str(pipeline), 'status', '--porcelain'], text=True).strip(), 'EXACT_CLEAN_PIPELINE_SOURCE')
    front_hashes = {name: sha(regular(frontend / name, 1024**2)) for name in FRONTEND_FILES}
    need(front_hashes == pins['frontend_sources_sha256'], 'FROZEN_FRONTEND_SOURCE_BYTES')
    paths = [str((frontend / name).relative_to(pipeline)) for name in FRONTEND_FILES]
    paths += [str((head / name).relative_to(pipeline)) for name in HEAD_FILES]
    subprocess.run(['git', '-C', str(pipeline), 'ls-files', '--error-unmatch', *paths],
                   check=True, stdout=subprocess.DEVNULL)
    for name, digest in SUPPORT.items():
        need(sha(regular(AUTOMATION / name, 256 * 1024)) == digest, 'FROZEN_SUPPORT_SOURCE')
    training_archive(args, pins)
    exported, generated = static_export_archive(args, pins, front_hashes)
    package = load_module('_terminal_frontend_saved_training_evidence', AUTOMATION / 'terminal144_package.py')
    evidence, trained, weights, binary = package.training_evidence(args.training_result)
    original, wanted = evidence['RESULT.json'], pins['head_training']
    verify_training_profile(original, wanted)
    need(sha(regular(args.training_result / 'RESULT.json', 32 * 1024**2)) == wanted['result_sha256'] and
         weights['sha256'] == wanted['weights_sha256'] and binary['sha256'] == wanted['head_binary_sha256'],
         'PINNED_ACTUAL_TRAINING_RESULT_AND_WEIGHTS')
    head_hashes = {name: sha(regular(head / name, 256 * 1024)) for name in HEAD_FILES}
    need(head_hashes == evidence['FEATURE_PREFLIGHT.json']['pipeline_source']['feature_head_files_sha256'],
         'UNCHANGED_REFERENCE_AND_HEAD_FROM_TRAINING')
    for name, (size, digest) in ASSETS.items():
        raw = regular(args.assets / name, 2 * 1024**2)
        need(len(raw) == size and sha(raw) == digest, 'SAME_TWO_PINNED_ONNX_ASSETS')
    return head, frontend, {'verification_pipeline_commit': commit,
        'training_pipeline_commit': evidence['FEATURE_PREFLIGHT.json']['pipeline_source']['commit'],
        'frontend_sources_sha256': front_hashes, 'head_sources_sha256': head_hashes,
        'support_sources_sha256': SUPPORT, 'head_training': wanted,
        'input_profile': INPUT_PROFILE, 'source_identity': original['source_identity'],
        'observer_identity': original['observer_identity'], 'training_identity':
            {k: original[k] for k in ('run_id', 'github_sha', 'run_attempt')},
        'original_campaign_status': original['original_campaign_status'], 'source_denominator': 144,
        'valid_source_count': original['valid_source_count'], 'source_dispositions': original['source_dispositions'],
        'static_export': pins['static_export'], 'static_export_result': exported,
        'consumed_original_static_generated_bytes': True,
        'training_input_binding': original['input_binding'], 'source_optimizer_updates': trained['optimizer_updates']}, generated


def timed_process(command, timeout, log):
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    try:
        result = subprocess.run([str(value) for value in command], timeout=timeout,
                                capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired as error:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        once(log, {'status': 'TIMEOUT', 'timeout_seconds': timeout,
                   'wall_seconds': time.perf_counter() - start,
                   'host_child_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime})
        raise TimeoutError('BOUNDED_CHILD_PROCESS_TIMEOUT') from error
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {'returncode': result.returncode, 'wall_seconds': time.perf_counter() - start,
              'host_child_cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
              'stdout': result.stdout[:32768], 'stderr': result.stderr[:32768],
              'scope': 'whole host process including startup and trace I/O; ARM includes QEMU emulation'}
    once(log, report)
    need(result.returncode == 0, 'CHILD_PROCESS_FAILED:' + str(log.name))
    return report


def build_replays(frontend, head, generated, output):
    tools = {name: shutil.which(name) for name in ('gcc', 'arm-linux-gnueabihf-gcc', 'qemu-arm')}
    need(all(tools.values()), 'NATIVE_ARM_COMPILER_AND_QEMU_REQUIRED')
    programs, build = {}, {}
    for name, compiler, static, machine in (('native', 'gcc', [], 62),
                                           ('arm', 'arm-linux-gnueabihf-gcc', ['-static'], 40)):
        program = output / ('frontend-' + name)
        command = [tools[compiler], '-std=c11', '-O2', '-fno-fast-math', '-ffp-contract=off',
                   '-Wall', '-Wextra', '-Werror', '-DEF_WITH_HEAD', *static,
                   '-I', frontend, '-I', generated, '-I', head,
                   frontend / 'fixed_frontend.c', frontend / 'frontend_replay.c', generated / 'fixed_models.c',
                   head / 'english_causal_head.c', '-lm', '-o', program]
        build[name] = timed_process(command, PLAN['compile_timeout_seconds'], output / ('compile-' + name + '.json'))
        raw = regular(program, 64 * 1024**2)
        need(raw[:4] == b'\x7fELF' and raw[5] == 1 and int.from_bytes(raw[18:20], 'little') == machine,
             'EXPECTED_NATIVE_OR_ARM_ELF')
        programs[name] = [str(program)] if name == 'native' else [tools['qemu-arm'], str(program)]
        build[name]['executable_sha256'] = sha(raw)
    build['tool_versions'] = {name: subprocess.check_output([path, '--version'], text=True).splitlines()[0]
                              for name, path in tools.items()}
    once(output / 'BUILD.json', build)
    return programs


def attach_head(rows, portable, weights):
    import numpy as np
    matrix = np.asarray([row[3] for row in rows], dtype=np.float32)
    heads = portable.reference(weights, matrix, [row['sample'] for row in rows])
    return [{**row, 4: [float(head['ready']), float(head['keyword']), *head['logits'], *head['probabilities']]}
            for row, head in zip(rows, heads)]


def reference_trace(raw, embedding, causal, portable, weights, accounting):
    import numpy as np
    rows = []
    def callback(pcm):
        accounting['onnx_session_attempts'] += 1
        mel = embedding.mel.run(None, {'input': np.asarray(pcm, dtype=np.float32).reshape(1, 16000)})[0]
        accounting['onnx_session_completions'] += 1
        need(mel.shape == (1, 1, 97, 32) and mel.dtype == np.float32 and bool(np.isfinite(mel).all()),
             'REFERENCE_MEL_GEOMETRY')
        normalized = ((mel[0, 0, -76:, :] / np.float32(10)) + np.float32(2)).reshape(1, 76, 32, 1)
        accounting['onnx_session_attempts'] += 1
        vector = embedding.embedding.run(None, {'input_1': normalized})[0]
        accounting['onnx_session_completions'] += 1
        need(vector.shape == (1, 1, 1, 96) and vector.dtype == np.float32 and bool(np.isfinite(vector).all()),
             'REFERENCE_EMBEDDING_GEOMETRY')
        rows.append({'sample': 16000 + len(rows) * 1280, 1: mel.reshape(-1).tolist(),
                     2: normalized.reshape(-1).tolist(), 3: vector.reshape(-1).tolist()})
        accounting['reference_frontend_windows_completed'] += 1
        return vector.reshape(96)
    stream, observed_clocks = causal.CausalFeatureStream(callback), []
    cpu, start = time.process_time(), time.perf_counter()
    for chunk in chunks(raw):
        observed_clocks.extend(r['available_sample'] for r in stream.accept(chunk))
    need(stream.observed == 48000 and not stream.finish() and tuple(observed_clocks) == SAMPLES and
         tuple(row['sample'] for row in rows) == SAMPLES, 'REFERENCE_ACTUAL_CLOCKS_AND_NO_EOF_PADDING')
    rows = attach_head(rows, portable, weights)
    return rows, {'wall_seconds': time.perf_counter() - start, 'process_cpu_seconds': time.process_time() - cpu,
                  'scope': 'loaded ORT sessions plus Python stream/head; excludes session initialization and file I/O'}


def timeout_handler(_signum, _frame):
    raise TimeoutError('TOTAL_VERIFICATION_DEADLINE')


def manifest(output):
    files = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(regular(p))}
             for p in sorted(output.rglob('*')) if p.is_file()}
    once(output / 'MANIFEST.json', {'schema': SCHEMA, 'scope': PLAN['scope'], 'plan': PLAN,
         'plan_sha256': sha(canonical(PLAN)), 'input_profile': INPUT_PROFILE, 'files': files, 'test_backend': False,
         **dict.fromkeys(FALSE_FLAGS, False)})


def execute(args):
    pins = read_json(args.pins)
    validate_pins(pins)
    execution_guard(pins, os.environ, read_json(Path(os.environ.get('GITHUB_EVENT_PATH', ''))),
                    sha(regular(Path(__file__), 256 * 1024)))
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'FRESH_VERIFICATION_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'run_id': os.environ['GITHUB_RUN_ID'], 'github_sha': os.environ['GITHUB_SHA'],
               'pins_sha256': sha(canonical(pins)), 'plan_sha256': sha(canonical(PLAN)), 'plan': PLAN,
               'input_profile': INPUT_PROFILE, 'run_attempt': '1',
               'source_identity': pins['head_training']['source_identity'],
               'observer_identity': pins['head_training']['observer_identity'],
               'training_identity': {k: pins['head_training'][k] for k in ('run_id', 'github_sha', 'run_attempt')},
               'test_backend': False, 'no_retry': True, **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    stage, reports = 'source-and-saved-head-preflight', []
    accounting = {'onnx_session_attempts': 0, 'onnx_session_completions': 0,
                  'reference_frontend_windows_completed': 0, 'optimizer_updates': 0,
                  'new_tts_asr_phone_calls': 0, 'real_audio_cases': 0}
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PLAN['total_timeout_seconds'])
    try:
        head, frontend, source, generated = source_preflight(args, pins)
        once(output / 'SOURCE_BINDING.json', source)
        for name, source_path in (('causal-head.npz', args.training_result / 'train/causal-head.npz'),
                                  ('causal-head.bin', args.training_result / 'causal-head.bin')):
            (output / name).write_bytes(regular(source_path))
        stage = 'original-static-generated-bytes-and-build'
        (output / 'generated').mkdir(mode=0o700)
        for name, raw in generated.items():
            with (output / 'generated' / name).open('xb') as stream:
                stream.write(raw)
        programs = build_replays(frontend, head, output / 'generated', output)
        sys.path.insert(0, str(head))
        causal, portable = importlib.import_module('causal_embedding'), importlib.import_module('portable_head')
        need(all(Path(sys.modules[name].__file__).resolve() == head / (name + '.py') for name in
                 ('causal_embedding', 'portable_head', 'weak_embedding_contract')), 'PINNED_REFERENCE_MODULES')
        import numpy as np
        import onnx
        import onnxruntime as ort
        versions = {'numpy': np.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}
        need(versions == {'numpy': '1.26.4', 'onnx': '1.17.0', 'onnxruntime': '1.20.1'}, 'FIXED_REFERENCE_RUNTIME')
        once(output / 'RUNTIME.json', {'versions': versions, 'python': sys.version,
             'timing_is_profiling_only_not_matched_benchmark_or_target_measurement': True})
        weights = portable.load_weights(output / 'causal-head.npz')
        need(sha(portable.export_blob(weights)) == pins['head_training']['head_binary_sha256'], 'NPZ_C_BINARY_IDENTITY')
        embedding = causal.FrozenOnnxEmbedding(args.assets)
        once(output / 'GEOMETRY.json', embedding.geometry)
        for case, raw in synthetic_pcm().items():
            directory = output / case
            directory.mkdir()
            pcm = directory / 'input.pcm'
            pcm.write_bytes(raw)
            stage = 'reference:' + case
            reference, timing = reference_trace(raw, embedding, causal, portable, weights, accounting)
            (directory / 'reference.bin').write_bytes(trace_bytes(reference))
            # The comparison uses an independently parsed saved float32 trace.
            reference = parse_trace(regular(directory / 'reference.bin', 1024**2))
            once(directory / 'reference-timing.json', timing)
            for backend, runner in programs.items():
                stage = backend + ':' + case
                trace = directory / (backend + '.bin')
                timing = timed_process([*runner, pcm, trace, ','.join(map(str, CHUNKS)), output / 'causal-head.bin'],
                                       PLAN['replay_timeout_seconds'], directory / (backend + '-timing.json'))
                actual = parse_trace(regular(trace, 1024**2))
                process_summary = replay_summary(timing['stdout'], actual)
                full = comparisons(reference, actual)
                same_embedding = head_comparison(attach_head(actual, portable, weights), actual, strict=True)
                report = {'case': case, 'backend': backend, 'pcm_sha256': sha(raw),
                          'embeddings': len(actual), 'ready_head_outputs': sum(r[4][0] == 1 for r in actual),
                          'full_pcm_to_head': full, 'head_with_same_c_embeddings': same_embedding,
                          'process_summary': process_summary, 'timing': timing,
                          'passed': full['passed'] and same_embedding['passed']}
                reports.append(report)
                once(directory / (backend + '-comparison.json'), report)
        need(accounting['onnx_session_attempts'] == accounting['onnx_session_completions'] == 156 and
             accounting['reference_frontend_windows_completed'] == 78 and len(reports) == 6,
             'EXACT_PREDECLARED_WORK_ACCOUNTING')
        passed = all(report['passed'] for report in reports)
        once(output / 'RESULT.json', {**started, 'status': 'EXECUTED_WITHIN_PREDECLARED_ENGINEERING_LIMITS'
             if passed else 'EXECUTED_WITH_NUMERICAL_OR_EVENT_MISMATCH', 'passed': passed,
             'accounting': accounting, 'comparisons': reports, 'source_binding': source,
             'acoustic_trigger_coverage_established': False, 'independent_frr': 'NOT_RUN',
             'independent_far': 'NOT_RUN', 'word_end_latency': 'NOT_RUN', 'ssc305': 'NOT_RUN',
             'arm_scope': 'FULL_FIXED_FRONTEND_AND_SAVED_HEAD_UNDER_HOST_QEMU_ONLY'})
        manifest(output)
        return 0 if passed else 2
    except Exception as error:
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
             'error_type': type(error).__name__, 'reason': str(error)[:1000],
             'accounting': accounting, 'completed_comparisons': reports})
        manifest(output)
        return 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--template', action='store_true')
    cli.add_argument('--pins', type=Path, default=AUTOMATION / 'terminal_frontend_verification_pins.json')
    for name in ('pipeline-source', 'frontend-source', 'training-result', 'training-zip',
                 'static-export-zip', 'assets', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.template:
        print(json.dumps(pins_template(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.execute:
        print(canonical({'schema': SCHEMA, 'status': 'SOURCE_CONTRACT_ONLY', 'execution_enabled': False,
                         'plan_sha256': sha(canonical(PLAN)), 'input_profile': INPUT_PROFILE,
                         'plan': PLAN, 'model_calls': 0}).decode())
        return 0
    need(all(getattr(args, name) is not None for name in
             ('pipeline_source', 'frontend_source', 'training_result', 'training_zip',
              'static_export_zip', 'assets', 'output')), 'EXPLICIT_INPUTS_REQUIRED')
    return execute(args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
