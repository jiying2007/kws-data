"""Audit a saved teacher-retention artifact using only standard-library readers.

Never imports repository code, loads a tensor runtime, or executes a model.
The ZIP identity and Actions run/head must be supplied independently by caller.
Exit 0 means the audit verified the evidence, including an honestly rejected
candidate; it does not imply model acceptance. Exit 2 means incomplete/invalid
evidence and the output never permits acceptance.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

SCHEMA = 'independent-teacher-retention-artifact-audit-v1'
REPOSITORY = 'jiying2007/kws-data'
RUN_SCHEMA = 'english-train-teacher-retention-once-v1'
BASE_NPZ_SHA = '75c1d13203f77fcec9b66013bd6fc8c1979cb3397207931eaf31241eeb396a35'
BASE_BIN_SHA = 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'
TRAIN_CONDITIONS = ('clean', 'snr10', 'new5', 'new15')
EVAL_CONDITIONS = ('clean', 'snr10', 'snr5', 'new8')
FORBIDDEN_PROMOTIONS = ('formal_training_admitted', 'human_gold', 'pronunciation_verified',
                        'model_promotion_allowed', 'qualification_allowed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'DUPLICATE_JSON_KEY:' + key)
        value[key] = item
    return value


def json_read(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:' + value)))


def safe_path(name):
    return (bool(name) and not PurePosixPath(name).is_absolute() and '\\' not in name
            and '\x00' not in name and all(part not in ('', '.', '..') for part in name.split('/')))


def read_npz_arrays(raw):
    shapes = {'mean96': (96,), 'scale96': (96,), 'w1': (32, 1536),
              'b1': (32,), 'w2': (3, 32), 'b2': (3,)}
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(len(archive.namelist()) == 6 and set(archive.namelist()) == {k+'.npy' for k in shapes}, 'NPZ_SIX_KEYS')
        for key, shape in shapes.items():
            data = archive.read(key + '.npy')
            require(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00'), 'NPY_HEADER')
            offset, fmt = (10, '<H') if data[6] == 1 else (12, '<I')
            count = struct.unpack_from(fmt, data, 8)[0]
            header = ast.literal_eval(data[offset:offset+count].decode('latin1'))
            require(header == {'descr': '<f4', 'fortran_order': False, 'shape': shape}, 'NPY_DTYPE_SHAPE:' + key)
            body = data[offset+count:]
            require(len(body) == 4*math.prod(shape), 'NPY_BODY_SIZE:' + key)
            require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', body)), 'NPY_FINITE:' + key)
            arrays[key] = body
    return arrays


def read_natural_trace(raw):
    require(raw[:8] == b'EKWSTRC1' and struct.unpack_from('<6I', raw, 8) == (1, 1, 16000, 320, 96, 3), 'TRACE_HEADER')
    scores, offset, next_tag, ordinal, terminated = [], 40, 1, 0, False
    while offset < len(raw):
        tag, length = struct.unpack_from('<BI', raw, offset)
        offset += 5
        body = raw[offset:offset+length]
        require(len(body) == length, 'TRACE_TRUNCATION')
        offset += length
        if tag == 1:
            require(next_tag == 1 and length == 400, 'TRACE_EMBEDDING_RECORD')
            number, clock = struct.unpack_from('<QQ', body)
            require(number == ordinal and clock == 16000 + 1280*ordinal, 'TRACE_EMBEDDING_CLOCK')
            next_tag = 2
        elif tag == 2:
            require(next_tag == 2 and length == 48, 'TRACE_SCORE_RECORD')
            number, ready, keyword, clock = struct.unpack_from('<QIIQ', body)
            values = struct.unpack_from('<6f', body, 24)
            require(number == ordinal and clock == 16000+1280*ordinal and ready == int(ordinal >= 15), 'TRACE_SCORE_CLOCK')
            require(keyword == 0 and all(math.isfinite(v) for v in values), 'BASELINE_NATURAL_ZERO_EVENTS')
            require(all(0 <= p <= 1 for p in values[3:]), 'TRACE_PROBABILITY_RANGE')
            scores.append({'sample': clock, 'ready': bool(ready), 'keyword': keyword,
                           'logits': list(values[:3]), 'probabilities': list(values[3:])})
            ordinal += 1
            next_tag = 1
        elif tag == 255:
            require(next_tag == 1 and length == 48 and offset == len(raw), 'TRACE_TERMINATOR')
            total, embeddings, rows, ready, events, calls = struct.unpack('<6Q', body)
            require(total == struct.unpack_from('<Q', raw, 32)[0] and embeddings == rows == len(scores), 'TRACE_TOTALS')
            require(ready == sum(r['ready'] for r in scores) and events == 0, 'TRACE_READY_EVENT_TOTALS')
            require(calls == (total+319)//320, 'TRACE_FEED_TOTAL')
            terminated = True
        else:
            raise ValueError('UNEXPECTED_NATURAL_TRACE_RECORD')
    require(terminated, 'TRACE_MISSING_TERMINATOR')
    return scores


def summarize(events, row):
    expected, start = row['weak_class'], row['clip_start_sample']
    require(expected in (0, 1, 2), 'EVENT_CLASS')
    require(all(type(e['sample']) is int and 0 < e['sample'] <= row['observed_samples']
                and e['keyword'] in (1, 2) for e in events), 'EVENT_RANGE')
    require(all(a['sample'] < b['sample'] for a, b in zip(events, events[1:])), 'EVENT_ORDER')
    prefix = [e for e in events if start is not None and e['sample'] <= start]
    after = [e for e in events if start is None or e['sample'] > start]
    matched = [e for e in after if expected and e['keyword'] == expected]
    wrong = [e for e in after if expected and e['keyword'] != expected]
    extra = max(0, len(matched)-1)
    error = bool(prefix or wrong or (expected and not matched) or extra or (expected == 0 and events))
    return {'matched_intent_after_clip_start': bool(matched), 'prefix_events': prefix,
            'wrong_keyword_after_clip_start': wrong, 'extra_same_keyword_events': extra,
            'weak_negative_intent_events': len(events) if expected == 0 else 0,
            'research_error': error}


def audit(args, report):
    path = args.artifact.absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_ARTIFACT_REQUIRED')
    require(path.stat().st_size <= 512*1024**2, 'ARTIFACT_SIZE_LIMIT')
    raw = path.read_bytes()
    report['artifact'] = identity(raw)
    require(report['artifact']['sha256'] == args.artifact_sha256, 'EXPECTED_ARTIFACT_SHA256')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        require(len(names) == len(set(names)) <= 10000, 'UNIQUE_BOUNDED_MEMBERS')
        require(sum(m.file_size for m in members) <= 512*1024**2, 'UNCOMPRESSED_SIZE_LIMIT')
        for member in members:
            require(safe_path(member.filename) and not member.is_dir() and not member.flag_bits & 1,
                    'REGULAR_SAFE_ZIP_MEMBER')
            require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG), 'ZIP_TYPE')
        load = lambda name: json_read(z.read(name))
        json_lines = lambda name: [json_read(line) for line in z.read(name).splitlines()]
        manifest = load('MANIFEST.json')
        require(set(names) == set(manifest['files']) | {'MANIFEST.json'}, 'MANIFEST_EXACT_CLOSURE')
        for name, pin in manifest['files'].items():
            require(identity(z.read(name)) == pin, 'MANIFEST_IDENTITY:' + name)
        report['manifest_files_verified'] = len(manifest['files'])
        report['archive_integrity_verified'] = True

        plan = load('source/automation/english_teacher_retention_plan.json')
        require(plan['schema'] == RUN_SCHEMA and plan['enabled'] is True, 'ENABLED_TEACHER_PLAN')
        worker_raw = z.read('source/automation/english_teacher_retention_once.py')
        require(identity(worker_raw)['sha256'] == plan['worker_sha256'], 'WORKER_PLAN_IDENTITY')
        if args.worker_sha256:
            require(plan['worker_sha256'] == args.worker_sha256, 'EXPECTED_WORKER_SHA256')
        for name, pin in plan['source_pins'].items():
            require(identity(z.read('source/'+name)) == pin, 'RETAINED_SOURCE_IDENTITY:' + name)
        require(plan['configuration'] == load('source/CONFIG.json'), 'FROZEN_CONFIGURATION')
        config = plan['configuration']
        require(config['threshold_float32'] == struct.unpack('<f',struct.pack('<f',0.8))[0]
                and config['confirm_embeddings'] == 3 and config['cooldown_embeddings'] == 13, 'UNCHANGED_EVENT_POLICY')
        binding = {'repository': REPOSITORY, 'run_id': str(args.run_id), 'head_sha': args.head_sha,
                   'main_before': plan['main_before'], 'run_attempt': 1, 'event': 'push', 'public_repository': True}
        result, supervisor = load('evidence/RESULT.json'), load('evidence/SUPERVISOR_EXIT.json')
        for item in (result, supervisor, load('STARTED.json')['binding']):
            require(all(item.get(k) == v for k,v in binding.items()), 'EXPECTED_EXECUTION_BINDING')
        report['binding'] = binding
        require(all(result.get(k) is False for k in FORBIDDEN_PROMOTIONS), 'NO_FORMAL_PROMOTION')
        require(result['current_default_model_replaced'] is False, 'BASELINE_NOT_REPLACED')
        require(result['numerical_execution_completed'] is True, 'NUMERICAL_EXECUTION_INCOMPLETE')
        require(supervisor['error'] is None and not supervisor['timed_out'] and not supervisor['partial_receipt_errors']
                and not supervisor['cleanup'].get('descendants_found'), 'SUPERVISOR_NOT_CLEAN')

        base_npz = z.read('inputs/coverage/models/coverage-head.npz')
        base_bin = z.read('inputs/coverage/models/coverage-head.bin')
        require(identity(base_npz)['sha256'] == BASE_NPZ_SHA and identity(base_bin)['sha256'] == BASE_BIN_SHA, 'EXACT_COVERAGE_BASE')
        base = read_npz_arrays(base_npz)
        require(base_bin == b'EHKWS01\0'+b''.join(base.values()), 'BASE_NPZ_BIN')
        candidate_npz, candidate_bin = z.read('models/retention-head.npz'), z.read('models/retention-head.bin')
        candidate = read_npz_arrays(candidate_npz)
        require(candidate_bin == b'EHKWS01\0'+b''.join(candidate.values()), 'CANDIDATE_NPZ_BIN')
        normalization_same = all(base[k] == candidate[k] for k in ('mean96', 'scale96'))
        weights_changed = any(base[k] != candidate[k] for k in ('w1', 'b1', 'w2', 'b2'))
        frozen = load('evidence/CANDIDATE_FROZEN.json')
        candidate_pin = {'npz': identity(candidate_npz), 'bin': identity(candidate_bin),
                         'arrays': {k: identity(v) for k,v in candidate.items()}}
        require(all(frozen[k] == v for k,v in candidate_pin.items()) and result['candidate'] == candidate_pin, 'CANDIDATE_BINDING')
        require(frozen['evaluation_model_calls_before_freeze'] == 0 and frozen['checkpoint_selection'] is False, 'CANDIDATE_FROZEN_FIRST')
        report['candidate'] = candidate_pin

        input_pins = load('source/research/english_hardneg_increment_20261117/INPUT_PINS.json')
        for name,pin in input_pins['coverage_members'].items():
            require(identity(z.read('inputs/coverage/'+name)) == pin, 'ORIGINAL_COVERAGE_MEMBER:'+name)
        old_natural_manifest = load('inputs/natural/MANIFEST.json')
        natural_roster = load('source/research/english_hardneg_natural32_20261117/natural32-roster.json')['rows']
        natural_baseline = {}
        require(len(natural_roster) == 32, 'FROZEN_NATURAL_ROSTER32')
        for source in natural_roster:
            rid = source['recording_id']; name = 'traces/'+rid+'.trace'
            trace = z.read('inputs/natural/'+name)
            require(identity(trace) == old_natural_manifest['files'][name], 'ORIGINAL_NATURAL_TRACE:'+rid)
            natural_baseline[rid] = {r['sample']:r for r in read_natural_trace(trace)}
        require(len(natural_baseline) == 32, 'UNIQUE_NATURAL_BASELINE32')

        geometry = load('evidence/TRAINING_GEOMETRY.json')['original_bags']
        geometry_by_key = {(g['condition'],g['id']):g for g in geometry}
        require(len(geometry) == len(geometry_by_key) == 225, 'TRAIN_GEOMETRY_225_UNIQUE')
        require(all(g['role'] == 'train' and g['condition'] in TRAIN_CONDITIONS for g in geometry), 'NO_DEVELOPMENT_TEACHER_INPUT')
        require(sum(len(g['head_instance_ends']) for g in geometry) == 8370, 'TEACHER_8370_WINDOWS')
        training = load('evidence/TRAINING.json')
        draws = training['draws']
        require(len(draws) == 96, 'NINETY_SIX_DRAWS')
        schedule_keys = set()
        reviewed_counts = collections.Counter()
        prior_draws = load('inputs/coverage/evidence/TRAINING.json')['draws']
        for i, draw in enumerate(draws):
            require(draw['step'] == i and draw['original'] == prior_draws[i], 'UNCHANGED_ORIGINAL_DRAWS')
            require(len(draw['original']) == 6 and [r['weak_class'] for r in draw['original']] == [0,1,2,0,1,2], 'SIX_OLD_CLASS_BALANCED_BAGS')
            require(all(r['role'] == 'train' and r['condition'] in TRAIN_CONDITIONS for r in draw['original']), 'DRAW_TRAIN_ONLY')
            schedule_keys.update((r['condition'],r['id']) for r in draw['original'])
            reviewed_counts[draw['reviewed_source']] += 1
        require(schedule_keys == set(geometry_by_key), 'TEACHER_KEYS_EQUAL_SCHEDULE_KEYS')
        require(len(reviewed_counts) == 8 and set(reviewed_counts.values()) == {12}, 'REVIEWED_SOURCE_EQUAL_WEIGHT')
        teacher = load('evidence/TEACHER_CACHE.json')
        require(teacher['base_npz'] == identity(base_npz) and teacher['base_bin'] == identity(base_bin), 'TEACHER_BASE')
        require(teacher['model_calls'] == 225 and teacher['head_windows'] == 8370
                and teacher['student_updates_before_freeze'] == teacher['development_calls'] == teacher['reviewed_calls'] == 0, 'TEACHER_TIMING_SCOPE')
        param_map = {'0.weight':'w1','0.bias':'b1','2.weight':'w2','2.bias':'b2'}
        require(teacher['parameter_pins'] == {k:identity(base[v]) for k,v in param_map.items()}, 'FROZEN_TEACHER_PARAMETERS')
        expected_keys = sorted(geometry_by_key)
        require(len(teacher['bags']) == 225, 'TEACHER_CACHE_BAG_COUNT')
        maximum_logp_normalization_error, teacher_rows = 0.0, 0
        for i, bag in enumerate(teacher['bags']):
            key = (bag['condition'],bag['id'])
            require(key == expected_keys[i] and bag['role'] == 'train' and bag['temperature'] == 2.0, 'TEACHER_CACHE_MEMBERSHIP')
            rows = len(geometry_by_key[key]['head_instance_ends'])
            require(bag['shape'] == [rows,3] and bag['member'] == f'cache/teacher-logp-{i:03d}.f32le', 'TEACHER_CACHE_GEOMETRY')
            data = z.read(bag['member'])
            require(identity(data) == bag['identity'] and len(data) == rows*12, 'TEACHER_CACHE_BYTES')
            for row in struct.iter_unpack('<3f',data):
                require(all(math.isfinite(x) and x <= 1e-6 for x in row), 'TEACHER_LOGP_FINITE')
                error = abs(sum(math.exp(x) for x in row)-1.0)
                maximum_logp_normalization_error = max(maximum_logp_normalization_error,error)
                require(error <= 1e-6, 'TEACHER_LOGP_NORMALIZED')
            teacher_rows += rows
        require(teacher_rows == 8370, 'COMPLETE_TEACHER_CACHE')
        recipe = teacher['recipe']
        require(recipe == plan['recipe']['distillation'] == training['recipe']['distillation'], 'DISTILLATION_RECIPE')
        require(recipe['temperature'] == 2.0 and recipe['weight'] == 10.0 and recipe['direction'] == 'KL_TEACHER_TO_STUDENT'
                and recipe['teacher_on_development_or_snr5_new8'] is False and recipe['teacher_on_reviewed_negatives'] is False,
                'FIXED_DISTILLATION_T2_WEIGHT10')
        require(all(len(training[k]) == 96 for k in ('losses','supervised_losses','distillation_losses')), 'LOSS_COUNTS')
        loss_arithmetic_error = max(abs(t-(s+10*d)) for t,s,d in zip(training['losses'],training['supervised_losses'],training['distillation_losses']))
        require(loss_arithmetic_error <= 1e-6 and all(math.isfinite(x) for k in ('losses','supervised_losses','distillation_losses') for x in training[k]), 'LOSS_ARITHMETIC_FINITE')
        report['teacher_cache'] = {'unique_train_bags':225,'windows':8370,'development_or_reviewed_inputs':0,
            'parameter_identity_verified':True,'maximum_logp_normalization_error':maximum_logp_normalization_error,
            'loss_arithmetic_max_error':loss_arithmetic_error,'temperature':2.0,'weight':10.0,
            'cache_contents_verified_without_recomputing_teacher_outputs':True}

        progress = load('evidence/NUMERIC_PROGRESS.json')
        expected_progress = {'optimizer_updates_started':96,'optimizer_updates_completed':96,
            'training_bag_forwards_started':672,'training_bag_forwards_completed':672,
            'candidate_head_cases_started':323,'candidate_head_cases_completed':323,
            'candidate_embedding_rows':18982,'candidate_ready_rows':14137,
            'teacher_train_forwards_started':225,'teacher_train_forwards_completed':225,
            'teacher_train_head_windows':8370,'teacher_development_forwards':0,'teacher_reviewed_forwards':0,
            'new_frontend_calls':0,'new_audio_decodes':0}
        require(progress == expected_progress and result['progress'] == progress, 'EXACT_EXECUTION_COUNTS')
        require(result['process_attempts'] == {'native':1,'numeric':1}, 'EXACT_PROCESS_COUNTS')

        old_metrics, old_frames = [], collections.defaultdict(list)
        for condition in EVAL_CONDITIONS:
            old_metrics += [dict(r,condition=condition) for r in load(f'inputs/coverage/predictions/coverage-{condition}-events.json')]
            for row in json_lines(f'inputs/coverage/predictions/coverage-{condition}-frames.jsonl'):
                old_frames[(condition,row['id'])].append(row)
        new_metrics = load('predictions/candidate-events.json')
        frames = collections.defaultdict(list)
        for row in json_lines('predictions/candidate-frames.jsonl'):
            require(all(math.isfinite(v) for v in row['logits']+row['probabilities']), 'CANDIDATE_FINITE')
            frames[(row['condition'],row['id'])].append(row)
        require(len(old_metrics) == len(new_metrics) == 291 and len(frames) == 323, 'COMPLETE_CASE_SETS')
        case_map = load('runtime/CASE_MAP.json')
        require(len(case_map) == 323, 'CASE_MAP_COUNT')
        reference, offset, ready_count = [], 0, 0
        for i, case in enumerate(case_map):
            key = (case['condition'],case['id']); rows = frames[key]
            require(case['case_index'] == i and case['offset'] == offset, 'CASE_MAP_ORDER_OFFSET')
            require([r['sample'] for r in rows] == list(range(16000,case['observed_samples']+1,1280)), 'COMPLETE_CAUSAL_CLOCKS')
            require(case['embedding_rows'] == len(rows), 'CASE_MAP_ROW_COUNT')
            for ordinal,row in enumerate(rows):
                require(row['ready'] is (ordinal >= 15), 'EXACT_READY_STATE')
                require(row['keyword'] in (0,1,2) and (row['ready'] or row['keyword'] == 0), 'EVENT_REQUIRES_READY')
                if not row['ready']:
                    require(all(x == 0 for x in row['logits']+row['probabilities']), 'NONREADY_ZERO_OUTPUT')
                else:
                    require(all(0 <= x <= 1 for x in row['probabilities']) and abs(sum(row['probabilities'])-1) <= 1e-6,'CANDIDATE_PROBABILITIES')
                reference.append(dict(row,case_index=i,local_sample=row['sample'],sample=offset+row['sample']))
            count = sum(r['ready'] for r in rows)
            require(case['ready_rows'] == count, 'CASE_READY_COUNT')
            ready_count += count
            offset += case['observed_samples']
        require(len(reference) == 18982 and ready_count == 14137, 'REFERENCE_COUNTS')
        require(reference == json_lines('runtime/reference-frames.jsonl'), 'SAVED_REFERENCE_MATCHES_CANDIDATE_FRAMES')

        paired, grouped, regressions, improvements = [], {}, [], []
        for a,b in zip(old_metrics,new_metrics):
            metadata = ('condition','id','role','weak_class','observed_samples','clip_start_sample','original_clip_eof_sample')
            require(all(a[k] == b[k] for k in metadata), 'PAIRED_CASE_IDENTITIES')
            key = (a['condition'],a['id'])
            summaries = []
            for table,item in ((old_frames,a),(frames,b)):
                events = [{'sample':r['sample'],'keyword':r['keyword']} for r in table[key] if r['keyword']]
                require(events == item['events'], 'EVENTS_FROM_SAVED_FRAMES')
                summary = summarize(events,item)
                require(all(item[k] == v for k,v in summary.items()), 'SUMMARY_FROM_SAVED_EVENTS')
                summaries.append(summary)
            old_summary,new_summary = summaries
            increases = [k for k in ('prefix_events','wrong_keyword_after_clip_start') if len(new_summary[k]) > len(old_summary[k])]
            increases += [k for k in ('extra_same_keyword_events','weak_negative_intent_events') if new_summary[k] > old_summary[k]]
            row = {k:a[k] for k in ('condition','id','role','weak_class')}
            row.update(prior_matched_intent_lost=old_summary['matched_intent_after_clip_start'] and not new_summary['matched_intent_after_clip_start'],
                prior_correct_case_regressed=not old_summary['research_error'] and new_summary['research_error'],
                event_error_categories_increased=increases,old_events=a['events'],new_events=b['events'])
            paired.append(row)
            if row['prior_matched_intent_lost'] or row['prior_correct_case_regressed'] or increases:
                regressions.append(row)
            if (not old_summary['matched_intent_after_clip_start'] and new_summary['matched_intent_after_clip_start']) or (old_summary['research_error'] and not new_summary['research_error']):
                improvements.append(row)
            group = (a['condition'],a['role'],a['weak_class'])
            counts = grouped.setdefault(group,{'cases':0,'old_matches':0,'new_matches':0,'old_errors':0,'new_errors':0})
            counts['cases'] += 1
            for label,s in zip(('old','new'),summaries):
                counts[label+'_matches'] += int(s['matched_intent_after_clip_start'])
                counts[label+'_errors'] += int(s['research_error'])
        require(paired == load('evidence/PAIRED_REGRESSION.json')['cases'], 'ALL_291_RECOMPUTED_PAIRED_ROWS')

        selection = load('evidence/ADMITTED_HEAD_INPUTS.json')
        require(selection['head_inputs'] == 134 and len(selection['sources']) == 8 and len(selection['candidates']) == 9, 'REVIEWED_INPUT_COUNTS')
        require(sum(len(s['head_ends']) for s in selection['sources']) == 134, 'EXACT_REVIEWED_INPUT_TOTAL')
        source_fit = []
        for source in selection['sources']:
            rid = source['recording_id']; ends = source['head_ends']
            old = natural_baseline[rid]
            new = {r['sample']:r for r in frames[('natural',rid)]}
            ce = [sum(-math.log(max(table[t]['probabilities'][0],1e-12)) for t in ends)/len(ends) for table in (old,new)]
            source_fit.append({'recording_id':rid,'head_inputs':len(ends),'old_background_ce':ce[0],'new_background_ce':ce[1]})
        fit = load('evidence/REVIEWED_FIT.json')
        require(source_fit == fit['sources'], 'RECOMPUTED_EIGHT_SOURCE_FIT')
        old_ce = sum(s['old_background_ce'] for s in source_fit)/8
        new_ce = sum(s['new_background_ce'] for s in source_fit)/8
        require(old_ce == fit['old_source_balanced_ce'] and new_ce == fit['new_source_balanced_ce'], 'RECOMPUTED_SOURCE_BALANCED_CE')
        natural_keys = [key for key in frames if key[0] == 'natural']
        require(len(natural_keys) == 32 and {k[1] for k in natural_keys} == set(natural_baseline), 'NATURAL32_CASES')
        natural_events = [{'id':key[1],'sample':r['sample'],'keyword':r['keyword']} for key in natural_keys for r in frames[key] if r['keyword']]
        natural_result = load('evidence/NATURAL32_PAIRED.json')
        require(natural_result['old_event_count'] == 0 and natural_result['new_event_count'] == len(natural_events)
                and natural_result['new_events'] == natural_events, 'RECOMPUTED_NATURAL_EVENTS')

        native = z.read('evidence/head-native.stdout').decode('ascii').splitlines()
        require(len(native) == len(reference), 'NATIVE_ROW_COUNT')
        maxlogit,maxprob = 0.0,0.0
        native_events = []
        for row,line in zip(reference,native):
            fields = line.split(); require(len(fields) == 9, 'NATIVE_COLUMNS')
            require(tuple(map(int,fields[:3])) == (row['sample'],int(row['ready']),row['keyword']), 'NATIVE_CLOCK_READY_EVENT')
            values = list(map(float,fields[3:])); require(all(math.isfinite(x) for x in values), 'NATIVE_FINITE')
            maxlogit = max(maxlogit,max(abs(a-b) for a,b in zip(values[:3],row['logits'])))
            maxprob = max(maxprob,max(abs(a-b) for a,b in zip(values[3:],row['probabilities'])))
            if row['keyword']:
                native_events.append({'case_index':row['case_index'],'condition':row['condition'],'id':row['id'],
                                      'sample':row['local_sample'],'keyword':row['keyword']})
        native_passed = maxlogit <= 1e-4 and maxprob <= 2e-6
        parity = load('evidence/HEAD_NATIVE_PARITY.json')
        require(maxlogit == parity['max_abs_logit_error'] and maxprob == parity['max_abs_probability_error']
                and native_events == parity['events'], 'NATIVE_REPORT_REPRODUCED')
        for receipt in ('evidence/retention-numeric.exit.json','evidence/head-native.exit.json'):
            data = load(receipt)
            require(data['returncode'] == 0 and data['error'] is None and data['timed_out'] is False and data['retry_count'] == 0, 'COMPLETED_CHILD_PROCESS')
        gates = {'actual_weights_changed':weights_changed,'all_291_case_regression':not regressions,
                 'native_saved_npz_bin_parity':native_passed,'natural32_zero_events':not natural_events,
                 'normalization_byte_identical':normalization_same,'source_balanced_reviewed_ce_decreased':new_ce < old_ce}
        require(gates == result['gates'], 'RECOMPUTED_FINAL_GATES')
        accepted = all(gates.values())
        expected_status = 'RETAINED_RESEARCH_TEACHER_RETENTION_HEAD' if accepted else 'REJECTED_RESEARCH_REGRESSION_GATE'
        require(result['passed'] is accepted and result['research_retention_allowed'] is accepted
                and result['status'] == expected_status, 'FINAL_RESEARCH_VERDICT')
        require(supervisor['passed'] is accepted and supervisor['returncode'] == (0 if accepted else 1)
                and manifest['supervised_success'] is accepted, 'SUPERVISOR_VERDICT')
        report.update(audit_verified=True,status='VERIFIED_RESEARCH_RETAINED' if accepted else 'VERIFIED_RESEARCH_REJECTED',
            candidate_accepted_for_research=accepted,product_model_accepted=False,
            gates=gates,regression_cases=regressions,improved_cases=improvements,
            grouped_counts=[dict(condition=k[0],role=k[1],weak_class=k[2],**v) for k,v in sorted(grouped.items())],
            reviewed_fit={'old_source_balanced_ce':old_ce,'new_source_balanced_ce':new_ce,
                'relative_decrease':1-new_ce/old_ce,'all_eight_sources_improved':all(s['new_background_ce'] < s['old_background_ce'] for s in source_fit)},
            natural32_events={'old':0,'new':len(natural_events),'independent_far_claim':False},
            native_parity={'rows':len(reference),'ready_rows':ready_count,'max_logit_error':maxlogit,
                'max_probability_error':maxprob,'clock_ready_event_exact':True},
            progress=progress,default_model_replaced=False,independent_far='NOT_MEASURED',independent_frr='NOT_MEASURED',
            new_full_pcm_chain='NOT_RUN',new_arm='NOT_RUN',ssc305='NOT_RUN')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New JSON file; never overwritten.')
    parser.add_argument('--artifact-sha256',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--head-sha',required=True)
    parser.add_argument('--worker-sha256')
    args = parser.parse_args(argv)
    output = args.output.absolute()
    require(not output.exists() and not any(p.is_symlink() for p in (output,*output.parents)), 'NEW_REGULAR_OUTPUT_REQUIRED')
    require(re.fullmatch('[0-9a-f]{64}',args.artifact_sha256) and re.fullmatch('[0-9a-f]{40}',args.head_sha)
            and args.run_id.isdigit(), 'EXPLICIT_EXPECTED_IDENTITIES')
    if args.worker_sha256:
        require(re.fullmatch('[0-9a-f]{64}',args.worker_sha256), 'EXPLICIT_WORKER_IDENTITY')
    report = {'schema':SCHEMA,'audit_verified':False,'status':'INVALID_OR_INCOMPLETE_EVIDENCE',
              'candidate_accepted_for_research':False,'product_model_accepted':False,
              'model_executions':0,'network_calls':0,'method':'INDEPENDENT_STANDARD_LIBRARY_SAVED_BYTE_AND_RECORD_READBACK',
              'expected':{'artifact_sha256':args.artifact_sha256,'run_id':args.run_id,'head_sha':args.head_sha,
                          'worker_sha256':args.worker_sha256},'auditor':identity(Path(__file__).read_bytes())}
    try:
        audit(args,report)
    except (OSError,ValueError,KeyError,TypeError,IndexError,struct.error,zipfile.BadZipFile) as error:
        report['error'] = type(error).__name__ + ':' + str(error)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as target:
        json.dump(report,target,ensure_ascii=False,indent=2,allow_nan=False)
        target.write('\n')
    print(json.dumps({k:report[k] for k in ('status','audit_verified','candidate_accepted_for_research','product_model_accepted','model_executions')},sort_keys=True))
    return 0 if report['audit_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
