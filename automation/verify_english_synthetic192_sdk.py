"""Verify a synthetic English research SDK from saved bytes only.

No network requests, subprocesses, audio decoding, numerical model forwards,
training, or compilation are performed. The two imported parsers have fixed
source hashes and are used only for their byte-format parsing functions.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import zipfile

REPO = 'jiying2007/kws-data'
SCHEMA = 'english-synthetic192-sdk-delivery-v1'
TRAIN_SCHEMA = 'english-synthetic192-training-once-v1'
PCM_SCHEMA = 'english-synthetic192-fullpcm-once-v1'
BASE_SDK = {'bytes': 13148833, 'sha256': '912ed6143a794bcee8cc3b10c53a61eac1240330d4f319ea81daf33981175587'}
BASE_RECORDS = {'PACKAGE_MANIFEST.json': {'bytes': 13705, 'sha256': '9ff4405e3283dbc8c03cfbda225474559a2349ea01be8af85fe60d501ec0eb47'},
    'PROVENANCE.json': {'bytes': 1827, 'sha256': 'f500eb194c2e31779650a36522b731aef4e506925780ae4ebf4d04bce117451e'}}
COVERAGE = {'run_id': 44049954828, 'artifact_id': 14313603148,
    'name': 'english-coverage-fullpcm-44049954828',
    'head_sha': '95aec73b0ac8b58fa74e62df35c3e443107619b1', 'bytes': 13172135,
    'sha256': 'ad5494d3a7456d9c7e1fcd51fec727fa9d9a377a508459db6f88af5256170ba6'}
HELPERS = {
    'english_coverage_fullpcm_once.py': {'bytes': 44356, 'sha256': '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'},
    'english_hardneg_increment_once.py': {'bytes': 69681, 'sha256': 'ac89f4564bb3f93661beb02f01f7766616322a39096c94a4bdac84560f5a9ec1'}}
CONSUMERS = {'native': {'bytes': 2452080, 'sha256': 'ae5cdef5223be6274bb61b8ed6ab4ce5fd067cdb8f1c95faeb8f38fab6bbf9d5'},
    'arm': {'bytes': 2862184, 'sha256': 'cdee7e5a2a918315abde4953894e04a8ff8136879df5088d8d81d8a7f9dc9dd3'}}
GATES = ('legacy_all_291_case_regression', 'legacy_natural32_zero_events',
    'fresh_development_aggregate_nonregression', 'calibration_zero_false_events',
    'locked_test_per_word_nonregression', 'locked_test_strict_positive_gain',
    'weights_changed', 'normalization_byte_identical', 'native_saved_npz_bin_parity',
    'source_balanced_reviewed_ce_decreased')
WORKING_POINT = {'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13}
MAX_ZIP, MAX_EXPANDED, MAX_MEMBER = 512*1024**2, 1024*1024**2, 128*1024**2


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:' + value)))


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe(name):
    return (type(name) is str and name and '\\' not in name and '\x00' not in name
        and not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/'))
        and PurePosixPath(name).as_posix() == name)


def valid_pin(value, maximum=MAX_MEMBER):
    return (type(value) is dict and set(value) == {'bytes', 'sha256'}
        and type(value['bytes']) is int and 0 <= value['bytes'] <= maximum
        and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None)


def file_pin(path, maximum=MAX_ZIP):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE')
    need(path.stat().st_size <= maximum, 'FILE_BOUND')
    digest = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            count += len(block); digest.update(block)
            need(count <= maximum, 'READ_BOUND')
    return {'bytes': count, 'sha256': digest.hexdigest()}


class Directory:
    def __init__(self, root):
        self.root = Path(root).absolute()
        need(self.root.is_dir() and not any(p.is_symlink() for p in (self.root, *self.root.parents)), 'REGULAR_DIRECTORY')
        self.names = set()
        for path in self.root.rglob('*'):
            need(not path.is_symlink(), 'SYMLINK_IN_DIRECTORY')
            if path.is_file():
                name = path.relative_to(self.root).as_posix()
                need(safe(name), 'SAFE_DIRECTORY_MEMBER')
                self.names.add(name)
        need(len(self.names) <= 12000, 'DIRECTORY_MEMBER_BOUND')

    def read(self, name):
        need(safe(name) and name in self.names, 'DIRECTORY_MEMBER:' + name)
        path = self.root / name
        need(path.stat().st_size <= MAX_MEMBER and not path.is_symlink(), 'DIRECTORY_READ_BOUND')
        return path.read_bytes()

    def json(self, name):
        return loads(self.read(name))


class Archive:
    def __init__(self, path, expected):
        need(file_pin(path) == {k: expected[k] for k in ('bytes', 'sha256')}, 'ARCHIVE_BYTES')
        self.z = zipfile.ZipFile(path)
        infos = self.z.infolist(); names = [x.filename for x in infos]
        need(len(names) == len(set(names)) <= 10000, 'ARCHIVE_UNIQUE_NAMES')
        need(sum(x.file_size for x in infos) <= MAX_EXPANDED, 'ARCHIVE_EXPANSION_BOUND')
        for item in infos:
            need(safe(item.filename) and not item.is_dir() and not item.flag_bits & 1
                and 0 <= item.file_size <= MAX_MEMBER
                and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), 'ARCHIVE_SAFE_REGULAR_MEMBER')
        self.names = set(names)

    def read(self, name):
        need(safe(name) and name in self.names, 'ARCHIVE_MEMBER:' + name)
        return self.z.read(name)

    def json(self, name):
        return loads(self.read(name))

    def close(self):
        self.z.close()


def check_manifest(reader, schema, *, complete=True, name='MANIFEST.json'):
    doc = reader.json(name); files = doc.get('files')
    need(doc.get('schema') == schema and type(files) is dict and 1 <= len(files) <= 10000, 'SOURCE_MANIFEST_SCHEMA')
    for member, expected in files.items():
        need(safe(member) and valid_pin(expected), 'SOURCE_MANIFEST_PIN')
    actual = reader.names - {name}
    need(actual == set(files) if complete else actual <= set(files), 'SOURCE_MANIFEST_INVENTORY')
    for member in actual:
        need(pin(reader.read(member)) == files[member], 'SOURCE_MEMBER_BYTES:' + member)
    need(doc.get('supervised_success') is True
        and doc.get('authoritative_completion_receipt') == 'evidence/SUPERVISOR_EXIT.json', 'SOURCE_SUPERVISED_MANIFEST')
    return doc


def source_pin(value, kind):
    need(type(value) is dict and set(value) == set(COVERAGE), 'SOURCE_PIN_FIELDS')
    need(type(value['run_id']) is int and value['run_id'] > 0
        and type(value['artifact_id']) is int and value['artifact_id'] > 0
        and type(value['head_sha']) is str and re.fullmatch('[0-9a-f]{40}', value['head_sha']) is not None
        and valid_pin({k: value[k] for k in ('bytes', 'sha256')}, MAX_ZIP), 'SOURCE_PIN_IDENTITY')
    if kind == 'coverage':
        need(value == COVERAGE, 'EXACT_COVERAGE_SOURCE')
    else:
        expected = 'english-synthetic192-' + ('training' if kind == 'training' else 'fullpcm') + '-' + str(value['run_id'])
        need(value['name'] == expected and value['run_id'] != COVERAGE['run_id'], 'NEW_SYNTHETIC_SOURCE_REQUIRED')


def audit_api(run_raw, artifact_raw, binding, kind):
    need(set(binding) == {'source', 'run_json', 'artifact_json'}, 'API_BINDING_FIELDS')
    p = binding['source']; source_pin(p, kind)
    need(valid_pin(binding['run_json']) and valid_pin(binding['artifact_json'])
        and pin(run_raw) == binding['run_json'] and pin(artifact_raw) == binding['artifact_json'], 'EXACT_API_RESPONSE_BYTES')
    run, artifact = loads(run_raw), loads(artifact_raw)
    need(run.get('id') == p['run_id'] and run.get('head_sha') == p['head_sha']
        and run.get('status') == 'completed' and run.get('conclusion') == 'success'
        and run.get('run_attempt') == 1 and run.get('head_branch') == 'main'
        and run.get('event') == 'push' and run.get('repository', {}).get('private') is False
        and run.get('repository', {}).get('full_name') == REPO, 'ACTUAL_SUCCESSFUL_PUBLIC_RUN:' + kind)
    need(artifact.get('id') == p['artifact_id'] and artifact.get('name') == p['name']
        and artifact.get('expired') is False and artifact.get('size_in_bytes') == p['bytes']
        and artifact.get('digest') == 'sha256:' + p['sha256']
        and artifact.get('workflow_run', {}).get('id') == p['run_id']
        and artifact.get('workflow_run', {}).get('head_sha') == p['head_sha'], 'ACTUAL_SOURCE_ARTIFACT:' + kind)


def load_helpers(directory):
    modules = []
    for index, (name, expected) in enumerate(HELPERS.items()):
        path = Path(directory) / name
        need(file_pin(path, MAX_MEMBER) == expected, 'FROZEN_PARSER_BYTES:' + name)
        spec = importlib.util.spec_from_file_location('synthetic_sdk_frozen_' + str(index), path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def audit_supervisor(doc, source, schema):
    need(doc.get('schema') == schema + '/supervisor' and doc.get('passed') is True
        and doc.get('returncode') == 0 and doc.get('timed_out') is False and doc.get('error') is None
        and type(doc.get('cleanup')) is dict and doc['cleanup'].get('descendants_found') is False
        and doc.get('retry_count') == 0 and doc.get('partial_receipt_errors') == []
        and doc.get('repository') == REPO and doc.get('run_attempt') == 1
        and str(doc.get('run_id')) == str(source['run_id']) and doc.get('head_sha') == source['head_sha'], 'SUCCESSFUL_SUPERVISOR_AND_EXECUTION_IDENTITY')


def audit_training(reader, source, head, npz, h, inc):
    result = reader.json('evidence/RESULT.json')
    need(result.get('schema') == TRAIN_SCHEMA and result.get('status') == 'RETAINED_RESEARCH_SYNTHETIC192_HEAD'
        and result.get('passed') is True and result.get('research_retention_allowed') is True
        and result.get('numerical_execution_completed') is True and result.get('locked_test_opened') is True,
        'RETAINED_NEW_SYNTHETIC_TRAINING_REQUIRED')
    need(result.get('repository') == REPO and result.get('public_repository') is True
        and str(result.get('run_id')) == str(source['run_id']) and result.get('head_sha') == source['head_sha']
        and result.get('run_attempt') == 1, 'TRAINING_EXECUTION_BINDING')
    need(set(result.get('gates', {})) == set(GATES) and all(value is True for value in result['gates'].values()), 'ALL_TEN_TRAINING_GATES')
    need(result.get('process_attempts') == {'numeric': 1, 'native': 1}
        and result.get('strong_english_fsmn_ctc') is False and result.get('ssc305_qualification') is False,
        'ACTUAL_TRAINING_PROCESS_AND_ARCHITECTURE')
    need(result.get('candidate', {}).get('bin') == head and result['candidate'].get('npz') == npz, 'TRAINING_CANDIDATE_PIN')
    audit_supervisor(reader.json('evidence/SUPERVISOR_EXIT.json'), source, TRAIN_SCHEMA)
    raw, weights = reader.read('models/synthetic-head.bin'), reader.read('models/synthetic-head.npz')
    need(pin(raw) == head and pin(weights) == npz and head != h.HEAD, 'NEW_SELECTED_WEIGHT_BYTES')
    arrays = inc.verify_saved_npz_bin(weights, raw)
    need(arrays == result['candidate'].get('arrays'), 'SAVED_HEAD_ARRAY_PINS')
    parity = reader.json('evidence/HEAD_NATIVE_PARITY.json')
    need(parity.get('passed') is True and parity.get('candidate_bin') == head and parity.get('candidate_npz') == npz
        and parity.get('head_cases', 0) >= 323 and parity.get('embedding_rows', 0) >= 18982
        and parity.get('ready_rows', 0) >= 14137, 'SUCCESSFUL_NATIVE_SELECTED_HEAD_PARITY')
    frozen = reader.json('evidence/CANDIDATE_FROZEN.json')
    need(frozen.get('bin') == head and frozen.get('npz') == npz and frozen.get('arrays') == arrays
        and frozen.get('calibration_or_test_scores_before_freeze') == 0
        and frozen.get('legacy_development_used_for_selection') is False, 'CHECKPOINT_FROZEN_BEFORE_CALIBRATION_TEST')
    trained = reader.json('evidence/TRAINING.json')
    need(trained.get('schema') == TRAIN_SCHEMA + '/training' and trained.get('candidate') == result['candidate']
        and trained.get('base_bin') == h.HEAD and trained.get('base_npz') == h.NPZ
        and 16 <= trained.get('epochs', 0) <= 64 and 1 <= trained.get('selected_epoch', 0) <= trained['epochs']
        and trained.get('original_train_bags_replayed_every_epoch') == 225
        and trained.get('original_development_optimizer_uses') == 0
        and trained.get('calibration_optimizer_uses') == trained.get('locked_test_optimizer_uses') == 0
        and trained.get('epochs') == result.get('actual_epochs')
        and trained.get('selected_epoch') == result.get('selected_epoch')
        and trained.get('optimizer_updates') == result.get('optimizer_updates')
        and trained.get('optimizer_updates', 0) > 0, 'ACTUAL_CONVERGENCE_TRAINING_AND_ROLE_ISOLATION')
    paired = reader.json('evidence/PAIRED_REGRESSION.json')
    need(paired.get('passed') is True and len(paired.get('cases', [])) == 291
        and all(row.get('prior_matched_intent_lost') is False and row.get('prior_correct_case_regressed') is False
            and row.get('event_error_categories_increased') == [] for row in paired['cases']), 'ALL_291_PAIRED_NONREGRESSION')
    natural = reader.json('evidence/NATURAL32_PAIRED.json')
    need(natural.get('old_event_count') == natural.get('new_event_count') == 0 and natural.get('new_events') == [], 'NATURAL32_ZERO_SAVED_EVENTS')
    need(reader.json('evidence/REVIEWED_FIT.json').get('passed') is True, 'SOURCE_BALANCED_REVIEWED_FIT')
    evaluated = reader.json('evidence/FRESH_EVALUATION.json')
    need(evaluated.get('schema') == TRAIN_SCHEMA + '/fresh-evaluation' and evaluated.get('working_point') == WORKING_POINT
        and set(evaluated.get('gates_before_native', {})) == set(GATES) - {'native_saved_npz_bin_parity'}
        and all(v is True for v in evaluated['gates_before_native'].values()), 'FROZEN_WORKING_POINT_FRESH_GATES')
    roles = evaluated['roles']
    for role in ('development', 'locked-test'):
        old, new = roles[role]['baseline'], roles[role]['candidate']
        need(roles[role].get('aggregate_nonregression') is True and old['cases'] == new['cases']
            and new['false_and_extra_events'] <= old['false_and_extra_events'], 'FRESH_NONREGRESSION:' + role)
        for word in ('1', '2'):
            a, b = old['per_word'][word], new['per_word'][word]
            need(a['cases'] == b['cases'] > 0 and b['missed'] <= a['missed']
                and a['matched'] + a['missed'] == a['cases'] and b['matched'] + b['missed'] == b['cases'], 'PER_WORD_NONREGRESSION')
    need(roles['calibration']['candidate']['false_and_extra_events'] == 0
        and sum(roles['locked-test']['candidate']['per_word'][w]['matched'] for w in ('1', '2'))
        > sum(roles['locked-test']['baseline']['per_word'][w]['matched'] for w in ('1', '2')), 'ACTUAL_CALIBRATION_AND_LOCKED_GAIN')
    opening = reader.json('evidence/LOCKED_TEST_OPENED.json')
    need(opening.get('candidate') == result['candidate'] and opening.get('head_sha256_checked') == head['sha256']
        and opening.get('opened_once') is True and opening.get('no_subsequent_optimizer_or_selection') is True
        and all(v is True for v in opening.get('pretest_gates', {}).values()), 'ONE_FROZEN_LOCKED_TEST_OPENING')
    return result, evaluated


def selected_fresh(doc, candidate):
    need(doc.get('schema') == TRAIN_SCHEMA + '/fresh-cases' and doc.get('candidate') == candidate
        and doc.get('references_bound_to_selected_weights') is True and doc.get('locked_test_opened') is True
        and type(doc.get('rows')) is list and 9 <= len(doc['rows']) <= 576, 'FROZEN_FRESH_ROSTER')
    ids = [row['id'] for row in doc['rows']]
    need(len(ids) == len(set(ids)) and all(re.fullmatch('[A-Za-z0-9_.-]+', rid) for rid in ids), 'UNIQUE_FRESH_IDS')
    selected = []
    for label in (0, 1, 2):
        for condition in ('clean', 'snr10', 'snr5'):
            candidates = [r for r in doc['rows'] if r.get('role') == 'locked-test' and r.get('usable') is True
                and r.get('candidate_reference_available') is True and r.get('weak_class') == label and r.get('condition') == condition]
            need(candidates, 'FRESH_CASE_PER_CLASS_AND_CONDITION')
            selected.append(min(candidates, key=lambda r: r['id']))
    return selected


def candidate_legacy_refs(reader, h):
    mapping = reader.json('runtime/CASE_MAP.json')
    rows = [loads(line) for line in reader.read('runtime/reference-frames.jsonl').splitlines()]
    stream = reader.read('runtime/candidate-head-stream.bin')
    need(323 <= len(mapping) <= 899 and 18982 <= len(rows) <= 110000 and stream[:8] == b'EHSTR01\0'
        and len(stream) == 12 + 396*len(rows) and struct.unpack_from('<I', stream, 8)[0] == len(rows)
        and sum(m['embedding_rows'] for m in mapping) == len(rows), 'COMPLETE_CANDIDATE_SAVED_STREAM')
    wanted = {(x['condition'], x['id']) for x in h.CASES}; result = {}; seen = set(); cursor = 0
    for index, item in enumerate(mapping[:323]):
        key = item['condition'], item['id']; count = item['embedding_rows']
        need(key not in seen and item['case_index'] == index and type(count) is int and count >= 16
            and item['ready_rows'] == count - 15, 'LEGACY_CASE_GEOMETRY')
        seen.add(key); scores = []; vectors = []
        for ordinal, row in enumerate(rows[cursor:cursor+count]):
            sample, gap = struct.unpack_from('<QI', stream, 12+396*(cursor+ordinal))
            vec = list(struct.unpack_from('<96f', stream, 24+396*(cursor+ordinal)))
            local = 16000 + 1280*ordinal
            need((row['condition'], row['id']) == key and row['case_index'] == index and row['local_sample'] == local
                and row['sample'] == sample == item['offset'] + local and gap == int(ordinal == 0)
                and row['ready'] is (ordinal >= 15), 'CANDIDATE_SAVED_CLOCK')
            need(row['keyword'] in (0, 1, 2) and len(row['logits']) == len(row['probabilities']) == 3
                and all(math.isfinite(v) for v in vec + row['logits'] + row['probabilities']), 'FINITE_CANDIDATE_SAVED_REFERENCE')
            scores.append({k: row[k] for k in ('ready', 'keyword', 'logits', 'probabilities')} | {'sample': local})
            vectors.append(vec)
        need(item['observed_samples'] >= scores[-1]['sample'] and item['observed_samples'] < scores[-1]['sample']+1280, 'CANDIDATE_EXACT_EOF')
        if key in wanted:
            result[key] = {'mapping': item, 'scores': scores, 'embeddings': vectors}
        cursor += count
    need(cursor == 18982 and sum(m['ready_rows'] for m in mapping[:323]) == 14137 and set(result) == wanted, 'EXACT_323_LEGACY_PREFIX')
    return result


def audit_fullpcm(reader, source, training, training_reader, head, npz, h):
    doc = reader.json('evidence/RESULT.json')
    need(doc.get('schema') == PCM_SCHEMA + '/result' and doc.get('status') == 'VERIFIED_SYNTHETIC192_HEAD_FULLPCM_NATIVE_ARM'
        and doc.get('passed') is True and doc.get('software_research_delivery_verified') is True
        and doc.get('candidate_source') == training and doc.get('candidate_head') == head and doc.get('candidate_npz') == npz,
        'SUCCESSFUL_SAME_NEW_MODEL_FULLPCM')
    need(doc.get('repository') == REPO and str(doc.get('run_id')) == str(source['run_id'])
        and doc.get('head_sha') == source['head_sha'] and doc.get('run_attempt') == 1,
        'NEW_FULLPCM_RUN_IDENTITY')
    audit_supervisor(reader.json('evidence/SUPERVISOR_EXIT.json'), source, PCM_SCHEMA)
    need(doc.get('fullpcm_process_attempts') == doc.get('completed_comparisons') == 28
        and doc.get('legacy_cases') == 5 and doc.get('locked_synthetic_cases') == 9
        and doc.get('strong_english_fsmn_ctc') is False and doc.get('ssc305_qualification') is False
        and doc.get('ssc305') == 'NOT_RUN', 'FULLPCM_EXACT_SCOPE')
    for name in ('new_compilations', 'optimizer_updates', 'new_python_or_onnx_forwards', 'new_audio_mixes', 'new_tts_asr_calls', 'retry_count'):
        need(doc.get(name) == 0, 'FULLPCM_NO_ADDITIONAL_COMPUTE:' + name)
    need(pin(reader.read('models/synthetic-head.bin')) == head and pin(reader.read('models/synthetic-head.npz')) == npz, 'FULLPCM_SELECTED_WEIGHT_BYTES')
    preflight = reader.json('evidence/INPUT_PREFLIGHT.json')
    need(preflight.get('status') == 'AUTHENTICATED_SYNTHETIC192_HEAD_FIXED_FULLPCM_INPUTS'
        and preflight.get('candidate_source') == training and preflight.get('candidate_head') == head
        and preflight.get('outcome_based_case_selection') is False and preflight.get('new_locked_synthetic_cases') == 9,
        'DETERMINISTIC_NEW_CASE_SELECTION')
    cases = preflight['cases']; need(len(cases) == 14 and len({r['id'] for r in cases}) == 14, 'FOURTEEN_DISTINCT_CASES')
    retained = training_reader.json('evidence/RESULT.json')
    selected = selected_fresh(training_reader.json('fresh/CASE_ROSTER.json'), retained['candidate'])
    need(cases[5:] == selected, 'EXACT_NINE_LEXICOGRAPHIC_LOCKED_CASES')
    refs = candidate_legacy_refs(training_reader, h)
    for index, original in enumerate(h.CASES):
        current = refs[(original['condition'], original['id'])]; case = cases[index]
        expected = dict(original); expected.pop('first_record')
        expected['events'] = [[s['keyword'], s['sample']] for s in current['scores'] if s['keyword']]
        if original['kind'] == 'known_vosen_miss_preserved': expected['kind'] = 'historical_vosen_miss_regression_case'
        expected['reference_origin'] = 'NEW_SYNTHETIC192_CANDIDATE_SAVED_OUTPUT'
        need(case == expected and current['mapping']['observed_samples'] == case['samples'], 'FIVE_SAME_PCM_WITH_NEW_CANDIDATE_REFERENCES')
        reference = reader.json('inputs/' + case['id'] + '.reference.json')
        need(reference == {'case': case, 'scores': current['scores'], 'embeddings': current['embeddings']}, 'NEW_LEGACY_REFERENCE_FROM_TRAINING_BYTES')
    for case in selected:
        reference = reader.json('inputs/' + case['id'] + '.reference.json')
        need(reference == training_reader.json('fresh/references/' + case['id'] + '.json')
            and reference['case'] == case and case.get('kind') == 'synthetic_source_heldout'
            and case.get('tier') in ('silver', 'weak') and case.get('context_kind') == 'generated_non_speech',
            'FRESH_REFERENCE_FROM_FROZEN_TRAINING_BYTES')
    reports = []; parsed_by_backend = {}; process_names = set()
    for backend in ('native', 'arm'):
        binary = reader.read('bin/trace-' + backend)
        need(pin(binary) == CONSUMERS[backend], 'ORIGINAL_PINNED_TRACE_BINARY')
        h.validate_elf(binary, backend); parsed_by_backend[backend] = []
        for index, case in enumerate(cases):
            rid = case['id']; stem = 'evidence/' + backend + '-case' + str(index)
            process_names.add(stem + '.started.json')
            started, ended = reader.json(stem + '.started.json'), reader.json(stem + '.exit.json')
            need(started.get('model_process') is True and started.get('consumer_build') is False
                and started.get('attempt') == 1 and started.get('retry_count') == 0
                and ended.get('model_process') is True and ended.get('consumer_build') is False
                and ended.get('attempt') == 1 and ended.get('retry_count') == 0
                and ended.get('returncode') == 0 and ended.get('timed_out') is False and ended.get('error') is None
                and ended.get('argv') == started.get('argv') and ended.get('descendants_supervised_by_parent_process_group') is True,
                'ONE_SUCCESSFUL_REPLAY_RECEIPT')
            argv = started['argv']; offset = 1 if backend == 'arm' else 0
            need(len(argv) == 4+offset and (backend == 'native' or argv[0] == 'qemu-arm')
                and Path(argv[offset]).name == 'trace-' + backend
                and Path(argv[offset+1]).name == 'synthetic-head.bin'
                and Path(argv[offset+2]).name == rid + '.pcm'
                and Path(argv[offset+3]).name == backend + '-' + rid + '.trace', 'EXACT_REPLAY_MODEL_AND_INPUT_ARGV')
            pcm = reader.read('inputs/' + rid + '.pcm'); reference = reader.json('inputs/' + rid + '.reference.json')
            need(pin(pcm) == {'bytes': 2*case['samples'], 'sha256': case['pcm_sha256']} and reference['case'] == case, 'EXACT_REPLAY_PCM')
            trace = reader.read('traces/' + backend + '-' + rid + '.trace'); parsed = h.parse_trace(trace, case)
            report = h.compare_trace({'case': case, 'reference': reference['scores'], 'embeddings': reference['embeddings']}, parsed, reader.json(stem + '.stdout'))
            report['backend'] = backend
            need(report['passed'] is True and report == reader.json(stem + '.comparison.json'), 'RECOMPUTED_FULLPCM_COMPARISON')
            reports.append(report); parsed_by_backend[backend].append(parsed)
    actual_processes = {n for n in reader.names if n.startswith('evidence/') and n.endswith('.started.json') and reader.json(n).get('model_process') is True}
    need(actual_processes == process_names and len(reports) == 28 and reports == doc.get('comparisons'), 'EXACT_28_MODEL_PROCESSES_AND_COMPARISONS')
    paired = []
    for index, case in enumerate(cases):
        a, b = parsed_by_backend['native'][index], parsed_by_backend['arm'][index]
        same = a['events'] == b['events'] and all((x['sample'], x['ready'], x['keyword']) == (y['sample'], y['ready'], y['keyword']) for x, y in zip(a['scores'], b['scores']))
        need(same, 'NATIVE_ARM_EXACT_STATES_AND_EVENTS')
        paired.append({'case_id': case['id'], 'exact_states_and_events': same, 'whole_trace_bitexact': a['trace_identity'] == b['trace_identity']})
    need(paired == doc.get('paired_backends'), 'FOURTEEN_NATIVE_ARM_PAIRS')
    return cases


def verify(root):
    reader = Directory(root); manifest = reader.json('PACKAGE_MANIFEST.json')
    need(manifest.get('schema') == SCHEMA and type(manifest.get('files')) is dict, 'PACKAGE_SCHEMA')
    files = manifest['files']
    need(reader.names == set(files) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS'}, 'EXACT_PACKAGE_INVENTORY')
    for name, expected in files.items():
        need(safe(name) and valid_pin(expected) and pin(reader.read(name)) == expected, 'PACKAGE_MEMBER_BYTES:' + name)
    expected_checksums = ''.join(hashlib.sha256(reader.read(name)).hexdigest() + '  ' + name + '\n' for name in sorted(set(files) | {'PACKAGE_MANIFEST.json'}))
    need(reader.read('SHA256SUMS').decode() == expected_checksums, 'EXACT_SHA256SUMS')
    plan = reader.json('ASSEMBLY_PLAN.json'); need(plan.get('enabled') is True and plan.get('base_sdk') == BASE_SDK, 'ACTUALLY_BOUND_ASSEMBLY_PLAN')
    for kind in ('coverage', 'training', 'fullpcm'):
        audit_api(reader.read('evidence/api/' + kind + '-run.json'), reader.read('evidence/api/' + kind + '-artifact.json'), plan['api'][kind], kind)
    h, inc = load_helpers(reader.root / 'verification')
    card = reader.json('MODEL_CARD.json'); head, npz = card['head'], card['npz']
    need(manifest.get('selected_model') == head and plan.get('candidate_head') == head and plan.get('candidate_npz') == npz
        and pin(reader.read('models/selected-head.bin')) == head and pin(reader.read('models/selected-head.npz')) == npz,
        'PACKAGE_SELECTED_MODEL_BINDING')
    need(pin(reader.read('verify_package.py')) == plan['verifier'], 'PINNED_STANDALONE_VERIFIER')
    source = reader.json('PROVENANCE.json')
    need(source.get('api') == plan['api'] and source.get('base_sdk') == BASE_SDK and source.get('assembly_execution') == 'OFFLINE_SAVED_BYTES_ONLY', 'PROVENANCE_INPUT_BINDING')
    base_manifest = reader.json('evidence/base-sdk/PACKAGE_MANIFEST.json')
    base_provenance = reader.json('evidence/base-sdk/PROVENANCE.json')
    for name, expected in BASE_RECORDS.items():
        need(pin(reader.read('evidence/base-sdk/' + name)) == expected, 'EXACT_ACCEPTED_BASE_RECORD')
    need(base_provenance.get('selected_fullpcm_source') == COVERAGE and source.get('original_sdk_source') == base_provenance['sdk_source'], 'ORIGINAL_INSTALLED_ASSET_LINEAGE')
    asset_prefixes = ('sdk-native/', 'sdk-arm/', 'diagnostics/')
    need({n for n in reader.names if n.startswith(asset_prefixes)} == {n for n in base_manifest['files'] if n.startswith(asset_prefixes)}, 'COMPLETE_ORIGINAL_INSTALLED_ASSET_SET')
    for name in reader.names:
        if name.startswith(('sdk-native/', 'sdk-arm/', 'diagnostics/')) and not name.endswith('/models/causal-head.bin'):
            need(name in base_manifest['files'] and pin(reader.read(name)) == base_manifest['files'][name], 'UNCHANGED_INSTALLED_ASSET:' + name)
    for backend in ('native', 'arm'):
        need(reader.read('sdk-' + backend + '/models/causal-head.bin') == reader.read('models/selected-head.bin'), 'SDK_NEW_HEAD_INSTALLED')
        need(pin(reader.read('sdk-' + backend + '/lib/libekws_research_stream.a')) == h.SDK_LIBRARIES[backend]
            and hashlib.sha256(reader.read('sdk-' + backend + '/include/ekws_research_stream.h')).hexdigest() == h.PUBLIC_HEADER_SHA
            and pin(reader.read('diagnostics/trace-' + backend)) == CONSUMERS[backend], 'PINNED_INSTALLED_LIBRARY_HEADER_AND_TRACE_CONSUMER')
        need((reader.root / ('sdk-' + backend) / 'bin/ekws_pcm_stream').stat().st_mode & 0o111
            and (reader.root / ('diagnostics/trace-' + backend)).stat().st_mode & 0o111, 'PRESERVED_EXECUTABLE_MODE')
    old_arrays = inc.verify_saved_npz_bin(reader.read('evidence/base-sdk/selected-head.npz'), reader.read('evidence/base-sdk/selected-head.bin'))
    need(pin(reader.read('evidence/base-sdk/selected-head.bin')) == h.HEAD and pin(reader.read('evidence/base-sdk/selected-head.npz')) == h.NPZ, 'EXACT_BASELINE_HEAD')
    new_arrays = inc.verify_saved_npz_bin(reader.read('models/selected-head.npz'), reader.read('models/selected-head.bin'))
    need(all(new_arrays[k] == old_arrays[k] for k in ('mean96', 'scale96')), 'BYTE_IDENTICAL_NORMALIZATION')
    training = Directory(reader.root / 'evidence/training'); fullpcm = Directory(reader.root / 'evidence/fullpcm')
    check_manifest(training, TRAIN_SCHEMA + '/artifact', complete=False)
    check_manifest(fullpcm, PCM_SCHEMA + '/artifact', complete=True)
    result, metrics = audit_training(training, plan['api']['training']['source'], head, npz, h, inc)
    cases = audit_fullpcm(fullpcm, plan['api']['fullpcm']['source'], plan['api']['training']['source'], training, head, npz, h)
    for original in h.CASES:
        name = 'tests/' + original['id'] + '.reference.json'
        raw = reader.read('evidence/base-sdk/' + name)
        need(pin(raw) == base_manifest['files'][name], 'ORIGINAL_REFERENCE_IDENTITY')
        old, new = loads(raw), fullpcm.json('inputs/' + original['id'] + '.reference.json')
        need(old['case'] == original and len(old['embeddings']) == len(new['embeddings'])
            and all(struct.pack('<96f', *a) == struct.pack('<96f', *b) for a, b in zip(old['embeddings'], new['embeddings'])),
            'ORIGINAL_FIVE_EMBEDDINGS_PRESERVED')
    need(card.get('schema') == SCHEMA + '/model-card' and card.get('selection') == 'retained-synthetic192-research'
        and card.get('head') == head and card.get('npz') == npz and card.get('cases') == cases
        and card.get('training_source') == plan['api']['training']['source'] and card.get('fullpcm_source') == plan['api']['fullpcm']['source']
        and card.get('working_point') == WORKING_POINT and card.get('quality_gates') == result['gates']
        and card.get('fresh_evaluation') == metrics, 'MODEL_CARD_MATCHES_ACTUAL_EVIDENCE')
    need(card.get('model_kind') == 'frozen_acoustic_embedding_plus_1536_to_32_to_3_research_head'
        and card.get('primary_track') == 'frozen_embedding_plus_head_synthetic_research'
        and card.get('software_research_delivery_verified') is True and card.get('strong_english_fsmn_ctc') is False
        and card.get('ssc305_qualified') is False and card.get('human_recording_prerequisite') is False
        and card.get('physical_ssc305_prerequisite') is False and card.get('final_mero_pronunciation_frozen') is False
        and card.get('independent_real_far') == card.get('independent_real_frr') == card.get('human_word_end_latency') == 'NOT_MEASURED',
        'HONEST_SYNTHETIC_RESEARCH_AND_HARDWARE_SCOPE')
    return {'status': 'VERIFIED_SYNTHETIC192_RESEARCH_SDK_SAVED_BYTES_ONLY', 'files': len(reader.names),
        'selected_model': head, 'fullpcm_cases_rechecked': 14, 'fullpcm_comparisons_rechecked': 28,
        'training_gates_rechecked': 10, 'new_model_calls': 0, 'new_audio_decodes': 0,
        'software_research_delivery_verified': True, 'physical_ssc305_tested': False}


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(canonical(verify(args.package)).decode(), end='')
