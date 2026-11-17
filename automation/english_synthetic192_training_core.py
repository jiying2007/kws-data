"""One source-separated synthetic expansion experiment for the existing C head.

Imported by the separately supervised public Actions worker. Importing this file
does not import numeric runtimes, decode audio, fit a model, or access a network.
Historic experiments and their consumed guards are never changed.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import sys

SCHEMA = 'english-synthetic192-training-once-v1'
ROLES = ('train', 'development', 'calibration', 'locked-test')
CONDITIONS = ('clean', 'snr10', 'snr5')
CONTEXT = {'prefix_samples': 38400, 'suffix_samples': 16000,
    'kind': 'generated_non_speech',
    'clean_context_rms_pcm16': 32.0, 'snr10_db': 10.0, 'snr5_db': 5.0,
    'snr_definition': 'WHOLE_ORIGINAL_PCM_RMS_NOT_ACTIVE_SPEECH_RMS',
    'peak_headroom_pcm16': 30000, 'sample_rate': 16000,
    'no_runtime_eof_padding': True, 'human_word_end_annotation': False}
RECIPE = {'model': 'FROZEN_EMBEDDING_1536_32_RELU_3', 'parameters': 49283,
    'warmstart': 'RETAINED_COVERAGE_NOT_REJECTED_CANDIDATE',
    'normalization': 'COVERAGE_BYTES_FROZEN', 'seed': 20261117,
    'optimizer': 'AdamW', 'learning_rate': 0.0001, 'weight_decay': 0.0001,
    'gradient_clip_norm': 1.0, 'batch_bags': 24,
    'maximum_epochs': 64, 'minimum_epochs': 16, 'event_patience_epochs': 12,
    'checkpoint_selection': 'NEW_DEVELOPMENT_ONLY_LEXICOGRAPHIC_EVENT_THEN_LOSS_EARLIEST',
    'selection_key': ['false_and_extra_events', 'worst_per_word_miss_rate',
        'macro_per_word_miss_rate', 'source_balanced_development_loss', 'epoch'],
    'stop_rule': 'AT_LEAST_16_EPOCHS_AND_12_WITHOUT_EVENT_KEY_IMPROVEMENT_OR_MAXIMUM64',
    'plateau_claim': 'DEVELOPMENT_EVENT_PLATEAU_NOT_GLOBAL_OPTIMIZATION_CONVERGENCE',
    'epoch_sampler': 'EVERY_ELIGIBLE_TRAIN_BAG_ONCE_SEEDED_PERMUTATION',
    'weights': 'EQUAL_CLASSES_THEN_EQUAL_SOURCE_GROUPS_THEN_BAGS_TIMES_FIXED_TIER_WEIGHT',
    'legacy_weak_tier_weight': 0.5, 'owner_reviewed_negative_weight': 1.0,
    'positive_loss': 'UNCHANGED_WEAK_NEGATIVE_LOG_MEAN_TARGET_PROBABILITY',
    'negative_loss': 'UNCHANGED_MEAN_WINDOW_BACKGROUND_CROSS_ENTROPY',
    'teacher_loss': False, 'parameter_sweep': False, 'active_training_configurations': 1,
    'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
    'calibration': 'SEPARATE_AFTER_FREEZE_FIXED_WORKING_POINT_FEASIBILITY_NO_TUNING',
    'locked_test': 'OPEN_ONCE_ONLY_IF_FROZEN_DEVELOPMENT_LEGACY_CALIBRATION_GATES_PASS',
    'minimum_balanced_groups': {'train': 6, 'development': 2, 'calibration': 2, 'locked-test': 2},
    'cpu_threads': 1, 'maximum_feature_calls': 90000,
    'maximum_total_bags': 1200, 'maximum_embedding_rows_native': 110000}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def seeded(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'little')


def libri_identity(value):
    """Piper pNNN and both Libri corpus names refer to the same numeric person."""
    text = str(value)
    match = re.fullmatch(r'(?:p|librispeech:|libritts-speaker-|piper-libritts-|librispeech-dev-clean-speaker-|libri-speaker-)?([0-9]+)', text)
    need(match is not None, 'KNOWN_LIBRI_NUMERIC_IDENTITY')
    return str(int(match.group(1)))


def require_libri_disjoint(old, fresh):
    need(not {libri_identity(x) for x in old}.intersection(libri_identity(x) for x in fresh),
        'CROSS_LIBRITTS_LIBRISPEECH_NUMERIC_SPEAKER_EXPOSURE')


def frame_ends(samples):
    need(type(samples) is int and samples >= 35200, 'COMPLETE_CAUSAL_HISTORY')
    return list(range(16000, samples + 1, 1280))


def positive_ends(start, original_samples, observed):
    """Actual frozen mel support, whole synthetic clip; transitions are masked."""
    return [end for end in frame_ends(observed) if end >= 35200
            and end - 31840 <= start and end - 128 >= start + original_samples]


def contextualize(raw, row, condition):
    """Construct and retain a complete synthetic stream, never an EOF flush."""
    need(condition in CONDITIONS and len(raw) == 2 * row['pcm_samples'], 'SOURCE_PCM_GEOMETRY')
    samples = list(struct.unpack('<' + 'h' * row['pcm_samples'], raw))
    need(len(samples) >= 6400, 'MINIMUM_SYNTHETIC_SOURCE')
    prefix, suffix = CONTEXT['prefix_samples'], CONTEXT['suffix_samples']
    count = prefix + len(samples) + suffix
    rng = random.Random(seeded('synthetic192-context-v1', row['role'], row['source_group_id'], row['id'], condition))
    noise = [rng.randrange(-32767, 32768) for _ in range(count)]
    nmean = sum(noise) / count
    noise = [x - nmean for x in noise]
    nrms = math.sqrt(sum(x*x for x in noise) / count)
    rms = math.sqrt(sum(x*x for x in samples) / len(samples))
    amplitude = CONTEXT['clean_context_rms_pcm16'] if condition == 'clean' else rms / (10 ** (CONTEXT[condition + '_db']/20))
    signal = [0.0] * prefix + [float(x) for x in samples] + [0.0] * suffix
    values = [s + n * amplitude / nrms if condition != 'clean' or i < prefix or i >= prefix + len(samples)
              else s for i, (s, n) in enumerate(zip(signal, noise))]
    peak = max(abs(x) for x in values)
    gain = min(1.0, CONTEXT['peak_headroom_pcm16'] / peak) if peak else 1.0
    pcm = struct.pack('<' + 'h' * count, *(int(round(x * gain)) for x in values))
    return pcm, {'original_pcm': identity(raw), 'condition': condition,
        'clip_start_sample': prefix, 'original_clip_eof_sample': prefix + len(samples),
        'samples': count, 'prefix_samples': prefix, 'suffix_samples': suffix,
        'context_kind': CONTEXT['kind'], 'whole_clip_rms_pcm16': rms,
        'noise_rms_pcm16_before_headroom': amplitude, 'headroom_gain': gain,
        'generated_context_is_natural_recording': False, 'eof_padding_samples': 0,
        'word_end_latency': 'NOT_MEASURED'}


def postgeometry_admission(rows):
    """Do not repair failed groups by moving speakers between their frozen roles."""
    by_group = collections.defaultdict(set)
    for row in rows:
        if row['geometry_eligible']:
            by_group[(row['role'], row['source_group_id'])].add(row['weak_class'])
    good = {key for key, labels in by_group.items() if labels == {0, 1, 2}}
    counts = {role: sum(key[0] == role for key in good) for role in ROLES}
    for row in rows:
        row['usable'] = row['geometry_eligible'] and (row['role'], row['source_group_id']) in good
    return {'schema': SCHEMA + '/postgeometry', 'balanced_source_groups': counts,
        'minimum_balanced_groups': RECIPE['minimum_balanced_groups'],
        'passed': all(counts[r] >= RECIPE['minimum_balanced_groups'][r] for r in ROLES),
        'planned_roles_preserved': True,
        'counts_by_role_class_tier': [dict(role=k[0], weak_class=k[1], tier=k[2], cases=v)
            for k, v in sorted(collections.Counter((r['role'], r['weak_class'], r['tier'])
                for r in rows if r['usable']).items())]}


def bag_weights(bags):
    """Every bag is visited; a long recording or many augmentations buys no extra source weight."""
    groups = collections.defaultdict(set)
    counts = collections.Counter((b['weak_class'], b['source_group_id']) for b in bags)
    for label, group in counts:
        groups[label].add(group)
    need(set(groups) == {0, 1, 2}, 'ALL_THREE_TRAIN_CLASSES')
    weights = [b['tier_weight'] / (3 * len(groups[b['weak_class']]) * counts[(b['weak_class'], b['source_group_id'])]) for b in bags]
    total = sum(weights)
    return [w / total for w in weights]


def event_summary(rows):
    grouped = collections.defaultdict(lambda: {'cases': 0, 'matched': 0, 'events': 0, 'errors': 0})
    false_events = 0
    for row in rows:
        key = (row['weak_class'], row.get('tier', 'legacy-weak'), row.get('condition', 'clean'))
        g = grouped[key]; g['cases'] += 1; g['matched'] += int(row['matched_intent_after_clip_start'])
        g['events'] += len(row['events']); g['errors'] += int(row['research_error'])
        false_events += (row['weak_negative_intent_events'] if row['weak_class'] == 0 else
            len(row['prefix_events']) + len(row['wrong_keyword_after_clip_start']) + row['extra_same_keyword_events']
            + len(row.get('early_same_keyword_events', [])) + len(row.get('late_same_keyword_events', [])))
    words = {}
    for label in (1, 2):
        selected = [r for r in rows if r['weak_class'] == label]
        matched = sum(r['matched_intent_after_clip_start'] for r in selected)
        words[str(label)] = {'cases': len(selected), 'matched': matched,
                             'missed': len(selected)-matched,
                             'miss_rate': (len(selected)-matched)/len(selected) if selected else None}
    need(all(words[str(k)]['cases'] for k in (1, 2)), 'BOTH_WORDS_IN_FRESH_EVALUATION')
    return {'cases': len(rows), 'source_groups': len({r.get('source_group_id') for r in rows}),
        'false_and_extra_events': false_events, 'per_word': words,
        'grouped_class_tier_condition': [dict(weak_class=k[0], tier=k[1], condition=k[2], **v) for k, v in sorted(grouped.items())],
        'pronunciation_variant_counts': [dict(weak_class=k[0], requested_hypothesis=k[1],
            matched_hypothesis=k[2], cases=v['cases'], matched=v['matched'])
            for k, v in sorted(variant_counts(rows).items())],
        'negative_events': sum(r['weak_negative_intent_events'] for r in rows),
        'independent_real_far': 'NOT_MEASURED', 'independent_real_frr': 'NOT_MEASURED'}


def event_key(summary):
    rates = [summary['per_word'][str(k)]['miss_rate'] for k in (1, 2)]
    return (summary['false_and_extra_events'], max(rates), sum(rates)/2)


def variant_counts(rows):
    result = collections.defaultdict(lambda: {'cases': 0, 'matched': 0})
    for row in rows:
        key = (row['weak_class'], row.get('pronunciation_hypothesis') or 'ordinary-negative-v1',
            row.get('matched_pronunciation_hypothesis') or 'none')
        result[key]['cases'] += 1
        result[key]['matched'] += int(row['matched_intent_after_clip_start'])
    return result


def aggregate_nonregression(old, new):
    need(old['cases'] == new['cases'] and all(old['per_word'][str(k)]['cases'] == new['per_word'][str(k)]['cases'] for k in (1, 2)), 'SAME_FRESH_DENOMINATORS')
    return new['false_and_extra_events'] <= old['false_and_extra_events'] and all(
        new['per_word'][str(k)]['missed'] <= old['per_word'][str(k)]['missed'] for k in (1, 2))


def should_stop(epoch, event_best_epoch):
    if epoch >= RECIPE['minimum_epochs'] and epoch-event_best_epoch >= RECIPE['event_patience_epochs']:
        return 'DEVELOPMENT_EVENT_PLATEAU'
    if epoch == RECIPE['maximum_epochs']:
        return 'BUDGET_EXHAUSTED_NOT_CONVERGENCE'
    return None


def feature_packet(output, frozen, progress, save_progress, np):
    """Inspect all source identities, then extract frozen features without any KWS head."""
    root = output / 'inputs/fresh'
    corpus = frozen.loads(frozen.regular(root/'CORPUS_MANIFEST.json'))
    admission = frozen.loads(frozen.regular(root/'ADMISSION.json'))
    need(admission['training_snapshot_allowed'] is True, 'MACHINE_ADMISSION_MINIMUM')
    byid = {r['id']: r for r in admission['rows']}
    sys.path.insert(0, str(output/'runtime/backbone'))
    import causal_embedding
    embedding = causal_embedding.FrozenOnnxEmbedding(output/'assets')
    need(embedding.geometry['head_support_start_before_available'] == 31840 and
         embedding.geometry['support_end_before_available'] == 128, 'EXACT_BACKBONE_SUPPORT')
    frozen.once(output/'evidence/FEATURE_GEOMETRY.json', embedding.geometry)
    class CountedSession:
        def __init__(self, session): self.session = session
        def run(self, *args, **kwargs):
            progress['new_onnx_forwards_started'] += 1; save_progress()
            result = self.session.run(*args, **kwargs)
            progress['new_onnx_forwards_completed'] += 1; save_progress()
            return result
    embedding.mel, embedding.embedding = CountedSession(embedding.mel), CountedSession(embedding.embedding)
    rows = []
    for source in corpus['rows']:
        allowed = byid[source['id']]
        if not allowed['supervision_allowed']:
            continue
        need(source['role'] == allowed['role'] and source['source_group_id'] == allowed['source_group_id'], 'FRESH_SOURCE_ROLE_BINDING')
        raw = frozen.regular(root/source['pcm_path'])
        need(identity(raw)['sha256'] == source['pcm_sha256'], 'FRESH_PCM_HASH')
        for condition in CONDITIONS:
            pcm, context = contextualize(raw, source, condition)
            ident = 'ens192-' + hashlib.sha256((source['id']+'/'+condition).encode()).hexdigest()[:24]
            ends = frame_ends(context['samples'])
            label = allowed['research_class']
            support = [x for x in ends if x >= 35200] if label == 0 else positive_ends(context['clip_start_sample'], source['pcm_samples'], context['samples'])
            row = {'id': ident, 'parent_id': source['id'], 'source_group_id': source['source_group_id'],
                'source_speaker_id': source['source_speaker_id'], 'role': source['role'],
                'tier': allowed['tier'], 'weak_class': label,
                'pronunciation_hypothesis': allowed['pronunciation_hypothesis'],
                'matched_pronunciation_hypothesis': allowed['matched_pronunciation_hypothesis'],
                'tier_weight': allowed['training_weight'] if source['role'] == 'train' else 0.0,
                'head_instance_ends': support, 'geometry_eligible': len(support) >= (3 if label else 1),
                'condition': condition, 'records': len(ends), 'ready': max(0, len(ends)-15),
                'pcm_sha256': identity(pcm)['sha256'], 'pcm_path': 'fresh/pcm/'+ident+'.pcm',
                'kind': 'synthetic_source_heldout' if source['role'] != 'train' else 'synthetic_train', **context}
            frozen.put(output/row['pcm_path'], pcm)
            rows.append(row)
    post = postgeometry_admission(rows)
    frozen.once(output/'evidence/FRESH_POSTGEOMETRY.json', post)
    need(post['passed'], 'FROZEN_POSTGEOMETRY_SOURCE_MINIMUM')
    required = sum(r['records'] for r in rows if r['usable'])
    need(required <= RECIPE['maximum_feature_calls'], 'BOUNDED_NEW_FEATURE_CALLS')
    arrays, streams = {}, {}
    for index, row in enumerate(rows):
        if not row['usable']:
            continue
        pcm = frozen.regular(output/row['pcm_path'])
        values = np.frombuffer(pcm, dtype='<i2'); ends = frame_ends(len(values)); vectors = []
        for end in ends:
            progress['new_embedding_calls_started'] += 1; save_progress()
            vectors.append(embedding(values[end-16000:end]))
            progress['new_embedding_calls_completed'] += 1
            save_progress()
        matrix = np.asarray(vectors, dtype=np.float32)
        need(matrix.shape == (row['records'], 96) and bool(np.isfinite(matrix).all()), 'NEW_FEATURE_MATRIX')
        arrays['embeddings_'+str(index)] = matrix
        arrays['samples_'+str(index)] = np.asarray(ends, dtype='<u8')
        row['cache_index'] = index
        row['embedding_raw'] = identity(matrix.astype('<f4').tobytes())
        row['sample_clock_raw'] = identity(arrays['samples_'+str(index)].tobytes())
        streams[('fresh', row['id'])] = (matrix, ends)
    np.savez(output/'cache/synthetic-embeddings.npz', **arrays)
    frozen.once(output/'cache/SYNTHETIC_FEATURE_ROSTER.json', {'schema': SCHEMA+'/fresh-feature-cases', 'context': CONTEXT,
        'corpus_sha256': identity(frozen.regular(root/'CORPUS_MANIFEST.json'))['sha256'],
        'admission_sha256': identity(frozen.regular(root/'ADMISSION.json'))['sha256'],
        'feature_cache': frozen.file_identity(output/'cache/synthetic-embeddings.npz'),
        'rows': rows, 'head_evaluations_during_extraction': 0, 'locked_test_outcomes_opened': False})
    need(progress['new_embedding_calls_completed'] == required, 'NEW_FEATURE_CALL_ACCOUNTING')
    return rows, streams


def fresh_summary(row, scores):
    need(row['role'] in ROLES, 'FROZEN_FRESH_ROLE')
    events = [{'sample': r['sample'], 'keyword': r['keyword']} for r in scores if r['keyword']]
    label, start = row['weak_class'], row['clip_start_sample']
    prefix = [e for e in events if e['sample'] <= start]
    after = [e for e in events if e['sample'] > start]
    minimum_match = row['original_clip_eof_sample'] + 128
    maximum_match = row['original_clip_eof_sample'] + 16000
    correct = [e for e in after if label and e['keyword'] == label and minimum_match <= e['sample'] <= maximum_match]
    early = [e for e in after if label and e['keyword'] == label and e['sample'] < minimum_match]
    late = [e for e in after if label and e['keyword'] == label and e['sample'] > maximum_match]
    wrong = [e for e in after if label and e['keyword'] != label]
    extra = max(0, len(correct)-1)
    return {k: row[k] for k in ('id', 'parent_id', 'role', 'source_group_id', 'weak_class', 'tier', 'condition',
        'pronunciation_hypothesis', 'matched_pronunciation_hypothesis')} | {
        'events': events, 'matched_intent_after_clip_start': bool(correct),
        'matched_complete_synthetic_phrase': bool(correct), 'early_same_keyword_events': early,
        'late_same_keyword_events': late, 'synthetic_match_min_sample': minimum_match,
        'synthetic_match_max_sample': maximum_match,
        'prefix_events': prefix, 'wrong_keyword_after_clip_start': wrong,
        'extra_same_keyword_events': extra, 'weak_negative_intent_events': len(events) if label == 0 else 0,
        'no_matching_event_in_observed_context': bool(label and not correct),
        'research_error': bool(events) if label == 0 else bool(prefix or wrong or early or late or not correct or extra),
        'observed_samples': row['samples'], 'clip_start_sample': start,
        'original_clip_eof_sample': row['original_clip_eof_sample'],
        'event_minus_synthesis_clip_eof_samples': [e['sample']-row['original_clip_eof_sample'] for e in correct],
        'word_end_latency': 'NOT_MEASURED', 'label_kind': 'SYNTHETIC_MACHINE_RESEARCH',
        'independent_real_acoustic_evaluation': False}


def train_and_evaluate(output, frozen, progress, save_progress):
    """Actual execution is reachable only from the new worker's supervised numeric child."""
    import numpy as np
    import torch
    import onnx
    import onnxruntime
    need(np.__version__ == '1.26.4' and torch.__version__ == '2.9.1+cpu', 'EXACT_NUMERIC_RUNTIME')
    need(onnx.__version__ == '1.17.0' and onnxruntime.__version__ == '1.20.1', 'EXACT_FROZEN_FRONTEND_RUNTIME')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(RECIPE['seed']); torch.use_deterministic_algorithms(True)
    frozen.once(output/'evidence/RUNTIME.json', {'python': sys.version, 'numpy': np.__version__,
        'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': onnxruntime.__version__,
        'cpu_threads': 1, 'interop_threads': 1})
    portable = frozen.load_module(output/'inputs/coverage/sources/portable_head.py', 'synthetic192_portable')
    baseline = portable.load_weights(output/'inputs/coverage/models/coverage-head.npz')
    need(identity(portable.export_blob(baseline)) == frozen.BASE_BIN, 'ACTUAL_COVERAGE_WARMSTART')
    allrows, streams, legacy_keys, geometry, _ = frozen.restore_streams(output, np)
    source_roster = frozen.loads(frozen.regular(output/'source'/frozen.ROSTER_PATH))
    selection = frozen.loads(frozen.regular(output/'evidence/ADMITTED_HEAD_INPUTS.json'))
    parser = frozen.load_module(output/'source'/frozen.FROZEN_PARSER, 'synthetic192_trace_parser')
    old_natural, natural_keys = {}, []
    for row in source_roster['rows']:
        rid = row['recording_id']
        parsed = parser.parse_trace(frozen.regular(output/'inputs/natural/traces'/(rid+'.trace')),
            {'samples': row['frames'], 'records': row['expected_embedding_rows'], 'ready': row['expected_ready_rows']})
        key = ('natural', rid); natural_keys.append(key)
        streams[key] = (np.asarray(parsed['embeddings'], dtype=np.float32), [r['sample'] for r in parsed['scores']])
        allrows[key] = {'id': rid, 'role': row['source_role'], 'weak_class': None,
            'observed_samples': row['frames'], 'source': row}
        old_natural[rid] = parsed['scores']
    fresh_corpus = frozen.loads(frozen.regular(output/'inputs/fresh/CORPUS_MANIFEST.json'))
    old_libri_speakers = {libri_identity(r['source']['speaker']) for r in allrows.values()
        if r.get('source', {}).get('speaker') is not None}
    old_libri_speakers.update(libri_identity(r['speaker_id']) for r in source_roster['rows'])
    require_libri_disjoint(old_libri_speakers, (r['source_speaker_id'] for r in fresh_corpus['rows']))
    fresh_rows, fresh_streams = feature_packet(output, frozen, progress, save_progress, np)
    streams.update(fresh_streams)
    fresh_by_role = {role: [r for r in fresh_rows if r['usable'] and r['role'] == role] for role in ROLES}
    # No head has seen calibration or test during feature extraction.
    need(progress['calibration_head_cases'] == progress['locked_test_head_cases'] == 0, 'NO_PRETRAIN_TEST_OUTCOMES')

    def windows(key, ends):
        matrix, times = streams[key]; positions = {t: i for i, t in enumerate(times)}
        need(ends and all(t in positions and positions[t] >= 15 for t in ends), 'ACTUAL_16_CONTIGUOUS_EMBEDDINGS')
        raw = np.asarray([matrix[positions[t]-15:positions[t]+1] for t in ends], dtype=np.float32)
        return torch.from_numpy(((raw-baseline['mean96'])/baseline['scale96']).reshape(len(raw), 1536))

    parent_families = {}
    for row in allrows.values():
        parent = row.get('parent_id', row['id'])
        family = row.get('source_family_id', row.get('mix', {}).get('source_family_id'))
        if family is not None:
            need(parent not in parent_families or parent_families[parent] == family, 'LEGACY_PARENT_FAMILY_CONSISTENCY')
            parent_families[parent] = family
    bags = []
    for g in geometry:
        key = (g['condition'], g['id']); row = allrows[key]
        need(row['role'] == g['role'] == 'train', 'NO_LEGACY_DEVELOPMENT_TRAINING')
        parent = g.get('parent_id', row.get('parent_id', g['id']))
        if parent.startswith('libri-speaker-'):
            group = 'librispeech:'+parent.removeprefix('libri-speaker-')
        else:
            need(parent in parent_families, 'LEGACY_SYNTHETIC_SOURCE_FAMILY_REQUIRED')
            group = parent_families[parent]
        bags.append({'key': key, 'id': 'old/'+g['condition']+'/'+g['id'], 'role': 'train',
            'weak_class': g['weak_class'], 'source_group_id': group, 'parent_id': parent,
            'tier': 'legacy-weak', 'tier_weight': RECIPE['legacy_weak_tier_weight'],
            'head_instance_ends': g['head_instance_ends'], 'origin': 'ORIGINAL_TRAIN_REPLAY'})
    need(len(bags) == 225, 'COMPLETE_ORIGINAL_225_TRAIN_BAGS')
    for s in selection['sources']:
        rid = s['recording_id']; original = next(r for r in source_roster['rows'] if r['recording_id'] == rid)
        need(original['source_role'] == original['current_coverage_role'] == 'train', 'REVIEWED_SOURCE_TRAIN_ONLY')
        bags.append({'key': ('natural', rid), 'id': 'reviewed/'+rid, 'role': 'train',
            'weak_class': 0, 'source_group_id': 'librispeech:'+original['speaker_id'], 'parent_id': rid,
            'tier': 'owner-reviewed-negative', 'tier_weight': RECIPE['owner_reviewed_negative_weight'],
            'head_instance_ends': s['head_ends'], 'origin': 'EXACT_REVIEWED_WINDOWS_ONLY'})
    for row in fresh_by_role['train']:
        need(row['tier_weight'] in (0.5, 1.0), 'EXPLICIT_NEW_TRAIN_TIER_WEIGHT')
        bags.append({**row, 'key': ('fresh', row['id']), 'origin': 'NEW_SYNTHETIC_TRAIN'})
    need(len(bags) <= RECIPE['maximum_total_bags'], 'BOUNDED_TRAINING_BAGS')
    weights = bag_weights(bags)
    prepared = [windows(b['key'], b['head_instance_ends']) for b in bags]
    for b, w in zip(bags, weights):
        b['objective_weight'] = w
    frozen.once(output/'evidence/TRAINING_GEOMETRY.json', {'schema': SCHEMA+'/training-geometry',
        'bags': bags, 'bag_count': len(bags), 'training_windows': sum(len(x) for x in prepared),
        'all_training_roles': ['train'], 'legacy_development_used': False,
        'weight_sum': sum(weights), 'recipe': RECIPE})
    train_source_groups = {b['source_group_id'] for b in bags}
    need(not train_source_groups.intersection(r['source_group_id'] for r in fresh_rows if r['role'] != 'train'), 'SOURCE_GROUP_SPLIT_BARRIER')
    model = torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    need(sum(p.numel() for p in model.parameters()) == 49283, 'SAME_C_HEAD_ABI')
    with torch.no_grad():
        for parameter, name in ((model[0].weight, 'w1'), (model[0].bias, 'b1'), (model[2].weight, 'w2'), (model[2].bias, 'b2')):
            parameter.copy_(torch.from_numpy(baseline[name]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])

    def current_weights():
        return {'mean96': baseline['mean96'].copy(), 'scale96': baseline['scale96'].copy(),
            'w1': model[0].weight.detach().numpy().copy(), 'b1': model[0].bias.detach().numpy().copy(),
            'w2': model[2].weight.detach().numpy().copy(), 'b2': model[2].bias.detach().numpy().copy()}

    def weak_loss(logp, label):
        return -logp[:, 0].mean() if label == 0 else math.log(len(logp))-torch.logsumexp(logp[:, label], dim=0)

    def evaluate_fresh(candidate, role, tag, *, save=False):
        if role in ('calibration', 'locked-test'):
            need((output/'evidence/CANDIDATE_FROZEN.json').is_file(), 'TEST_ONLY_AFTER_HASHED_CANDIDATE_FREEZE')
        metrics, scores_by_id = [], {}
        for row in fresh_by_role[role]:
            matrix, times = streams[('fresh', row['id'])]
            scores = portable.reference(candidate, matrix, times)
            metrics.append(fresh_summary(row, scores)); scores_by_id[row['id']] = scores
            progress['fresh_head_case_evaluations'] += 1
            if role == 'calibration': progress['calibration_head_cases'] += 1
            if role == 'locked-test': progress['locked_test_head_cases'] += 1
        save_progress()
        if save:
            frozen.once(output/'predictions'/(tag+'-events.json'), metrics)
        return event_summary(metrics), metrics, scores_by_id

    def diagnostic_loss(candidate):
        devbags = [{**r, 'tier_weight': 1.0 if r['tier'] == 'silver' else 0.5} for r in fresh_by_role['development']]
        dweights = bag_weights(devbags); total = 0.0
        for row, weight in zip(devbags, dweights):
            x = windows(('fresh', row['id']), row['head_instance_ends']).numpy()
            logits = np.maximum(x @ candidate['w1'].T + candidate['b1'], np.float32(0)) @ candidate['w2'].T + candidate['b2']
            logits = logits.astype(np.float64); logits -= np.max(logits, axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            label = row['weak_class']
            value = -np.log(np.maximum(p[:, 0], 1e-12)).mean() if label == 0 else -math.log(max(float(p[:, label].mean()), 1e-12))
            total += weight * float(value)
        return total

    baseline_dev, _, _ = evaluate_fresh(baseline, 'development', 'baseline-development', save=True)
    frozen.once(output/'evidence/BASELINE_NEW_DEVELOPMENT.json', baseline_dev)
    best_weights, best_key, best_epoch, best_events, best_event_epoch = None, None, None, None, 0
    curves = []; stop_reason = None
    with (output/'evidence/TRAINING_DRAWS.jsonl').open('x') as draw_file:
        for epoch in range(1, RECIPE['maximum_epochs']+1):
            order = list(range(len(bags))); random.Random(seeded(RECIPE['seed'], 'epoch', epoch)).shuffle(order)
            need(sorted(order) == list(range(len(bags))), 'COMPLETE_ONCE_PER_EPOCH_REPLAY')
            epoch_loss = 0.0
            for begin in range(0, len(order), RECIPE['batch_bags']):
                indices = order[begin:begin+RECIPE['batch_bags']]
                progress['optimizer_updates_started'] += 1; save_progress()
                optimizer.zero_grad(set_to_none=True); terms = []
                for index in indices:
                    need(bags[index]['role'] == 'train', 'BATCH_TRAIN_ONLY')
                    progress['training_bag_forwards_started'] += 1
                    logp = torch.log_softmax(model(prepared[index]), dim=-1)
                    progress['training_bag_forwards_completed'] += 1
                    loss = weak_loss(logp, bags[index]['weak_class']); terms.append(weights[index] * loss)
                    epoch_loss += weights[index] * float(loss.detach())
                # Uniform sampling of bags, inverse batch-size scaling estimates
                # the predeclared complete source-balanced objective.
                total = torch.stack(terms).sum() * (len(bags) / len(indices))
                need(bool(torch.isfinite(total)), 'FINITE_TRAINING_LOSS')
                total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); progress['optimizer_updates_completed'] += 1; save_progress()
                draw_file.write(canonical({'epoch': epoch, 'step': progress['optimizer_updates_completed'],
                    'bag_indices': indices, 'bag_ids': [bags[i]['id'] for i in indices],
                    'loss': float(total.detach())}).decode()+'\n'); draw_file.flush()
            candidate = current_weights()
            summary, _, _ = evaluate_fresh(candidate, 'development', 'epoch-'+str(epoch))
            dloss = diagnostic_loss(candidate); ekey = event_key(summary); key = (*ekey, dloss, epoch)
            need(math.isfinite(dloss), 'FINITE_DEVELOPMENT_DIAGNOSTIC')
            if best_events is None or ekey < best_events:
                best_events, best_event_epoch = ekey, epoch
            if best_key is None or key < best_key:
                best_key, best_weights, best_epoch = key, candidate, epoch
                # This is a development-selected working checkpoint, never a
                # final test-selected checkpoint; all epochs remain in curves.
                np.savez(output/'models/development-best.npz', **best_weights)
            progress['epochs_completed'] = epoch; save_progress()
            curves.append({'epoch': epoch, 'optimizer_updates': progress['optimizer_updates_completed'],
                'train_epoch_weighted_objective': epoch_loss, 'development_diagnostic_loss': dloss,
                'development_event_summary': summary, 'event_key': ekey, 'selection_key': key,
                'selected_epoch_so_far': best_epoch, 'event_best_epoch': best_event_epoch,
                'all_train_bags_visited': len(order)})
            frozen.checkpoint(output/'evidence/LEARNING_CURVE.json', {'recipe': RECIPE, 'epochs': curves})
            stop_reason = should_stop(epoch, best_event_epoch)
            if stop_reason:
                break
    need(best_weights is not None and best_epoch is not None, 'ACTUAL_TRAINED_CHECKPOINT')
    np.savez(output/'models/synthetic-head.npz', **best_weights)
    selected = portable.load_weights(output/'models/synthetic-head.npz')
    blob = portable.export_blob(selected); frozen.put(output/'models/synthetic-head.bin', blob)
    pins = {'npz': frozen.file_identity(output/'models/synthetic-head.npz'), 'bin': identity(blob),
        'arrays': frozen.verify_saved_npz_bin(frozen.regular(output/'models/synthetic-head.npz'), blob)}
    need(all(selected[k].tobytes() == baseline[k].tobytes() for k in ('mean96', 'scale96')), 'NORMALIZATION_UNCHANGED')
    need(progress['calibration_head_cases'] == progress['locked_test_head_cases'] == 0, 'NO_CALIBRATION_OR_TEST_SELECTION')
    frozen.once(output/'evidence/CANDIDATE_FROZEN.json', {**pins, 'selected_epoch': best_epoch,
        'actual_epochs': len(curves), 'optimizer_updates': progress['optimizer_updates_completed'],
        'stop_reason': stop_reason, 'selection_rule': RECIPE['selection_key'],
        'calibration_or_test_scores_before_freeze': 0, 'legacy_development_used_for_selection': False,
        'new_human_or_board_data_required_for_research': False})

    # No alternate checkpoint may be tried after these independent accept/reject gates.
    old_metrics = []
    for condition in ('clean', 'snr10', 'snr5', 'new8'):
        old_metrics.extend([dict(r, condition=condition) for r in frozen.loads(frozen.regular(
            output/'inputs/coverage/predictions'/('coverage-'+condition+'-events.json')))])
    new_metrics, new_natural, retained_scores = [], {}, {}
    for key in legacy_keys + natural_keys:
        condition, rid = key; matrix, times = streams[key]
        scores = portable.reference(selected, matrix, times); retained_scores[key] = scores
        events = [{'sample': r['sample'], 'keyword': r['keyword']} for r in scores if r['keyword']]
        if condition == 'natural': new_natural[rid] = scores
        else: new_metrics.append(dict(frozen.summarize_case(allrows[key], events), condition=condition))
        progress['legacy_final_head_cases'] += 1; save_progress()
    paired = frozen.paired_case_gate(old_metrics, new_metrics)
    natural_events = [{'id': rid, 'sample': r['sample'], 'keyword': r['keyword']}
        for rid, scores in new_natural.items() for r in scores if r['keyword']]
    fit = frozen.reviewed_fit(selection, old_natural, new_natural)
    frozen.once(output/'evidence/PAIRED_REGRESSION.json', paired)
    frozen.once(output/'evidence/REVIEWED_FIT.json', fit)
    frozen.once(output/'evidence/NATURAL32_PAIRED.json', {'old_event_count': 0,
        'new_event_count': len(natural_events), 'new_events': natural_events,
        'other_sources': 24, 'other_source_labels': 'UNKNOWN', 'whole_originals_labeled_negative': False})
    frozen.once(output/'predictions/candidate-events.json', new_metrics)
    summaries, fresh_scores, final_fresh_rows = {}, {}, []
    for role in ('train', 'development', 'calibration'):
        baseline_summary, _, _ = evaluate_fresh(baseline, role, 'baseline-'+role, save=role != 'development')
        current_summary, _, saved = evaluate_fresh(selected, role, 'selected-'+role, save=True)
        fresh_scores.update(saved); final_fresh_rows.extend(fresh_by_role[role])
        summaries[role] = {'baseline': baseline_summary, 'candidate': current_summary,
            'aggregate_nonregression': aggregate_nonregression(baseline_summary, current_summary)}
    gates = {'legacy_all_291_case_regression': paired['passed'],
        'legacy_natural32_zero_events': not natural_events,
        'source_balanced_reviewed_ce_decreased': fit['passed'],
        'fresh_development_aggregate_nonregression': summaries['development']['aggregate_nonregression'],
        'calibration_zero_false_events': summaries['calibration']['candidate']['false_and_extra_events'] == 0,
        'weights_changed': pins['bin'] != frozen.BASE_BIN, 'normalization_byte_identical': True}
    if all(gates.values()):
        frozen.once(output/'evidence/LOCKED_TEST_OPENED.json', {'candidate': pins, 'pretest_gates': gates,
            'head_sha256_checked': frozen.file_identity(output/'models/synthetic-head.bin')['sha256'],
            'opened_once': True, 'no_subsequent_optimizer_or_selection': True})
        baseline_summary, _, _ = evaluate_fresh(baseline, 'locked-test', 'baseline-locked-test', save=True)
        current_summary, _, saved = evaluate_fresh(selected, 'locked-test', 'selected-locked-test', save=True)
        fresh_scores.update(saved); final_fresh_rows.extend(fresh_by_role['locked-test'])
        summaries['locked-test'] = {'baseline': baseline_summary, 'candidate': current_summary,
            'aggregate_nonregression': aggregate_nonregression(baseline_summary, current_summary)}
        gates['locked_test_per_word_nonregression'] = summaries['locked-test']['aggregate_nonregression']
        gates['locked_test_strict_positive_gain'] = sum(current_summary['per_word'][str(k)]['matched'] for k in (1, 2)) > sum(
            baseline_summary['per_word'][str(k)]['matched'] for k in (1, 2))
    else:
        summaries['locked-test'] = {'status': 'NOT_OPENED_PRETEST_GATE_FAILED', 'head_evaluations': 0}
        gates['locked_test_per_word_nonregression'] = False
        gates['locked_test_strict_positive_gain'] = False
    frozen.once(output/'evidence/FRESH_EVALUATION.json', {'schema': SCHEMA+'/fresh-evaluation',
        'working_point': {'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13},
        'roles': summaries, 'gates_before_native': gates,
        'scope': 'SOURCE_GROUP_HELDOUT_SYNTHETIC_SILVER_AND_EXPLICIT_WEAK_RESEARCH'})
    final_cases = []
    evaluated_ids = {r['id'] for r in final_fresh_rows}
    for row in fresh_rows:
        if row['id'] not in evaluated_ids:
            final_cases.append({**row, 'events': None, 'candidate_reference_available': False})
            continue
        scores = fresh_scores[row['id']]; key = ('fresh', row['id']); retained_scores[key] = scores
        case = {**row, 'events': [[r['keyword'], r['sample']] for r in scores if r['keyword']],
            'candidate_reference_available': True}
        final_cases.append(case)
        frozen.once(output/'fresh/references'/(row['id']+'.json'), {'case': case,
            'scores': scores, 'embeddings': streams[key][0].tolist()})
    frozen.once(output/'fresh/CASE_ROSTER.json', {'schema': SCHEMA+'/fresh-cases',
        'candidate': pins, 'rows': final_cases, 'context': CONTEXT,
        'references_bound_to_selected_weights': True,
        'locked_test_opened': (output/'evidence/LOCKED_TEST_OPENED.json').exists()})
    # A single native process sees exactly the same cached rows, including fresh
    # heldout rows only when their one-shot evaluation was actually opened.
    native_keys = legacy_keys + natural_keys + [('fresh', r['id']) for r in final_fresh_rows]
    total_rows = sum(len(streams[k][0]) for k in native_keys)
    need(total_rows <= RECIPE['maximum_embedding_rows_native'], 'NATIVE_FRAME_BUDGET')
    case_map, offset = [], 0
    with (output/'runtime/candidate-head-stream.bin').open('xb') as stream, (output/'runtime/reference-frames.jsonl').open('x') as refs, (output/'predictions/candidate-frames.jsonl').open('x') as predictions:
        stream.write(b'EHSTR01\0'+struct.pack('<I', total_rows))
        fresh_index = {r['id']: r for r in final_fresh_rows}
        for index, key in enumerate(native_keys):
            condition, rid = key; matrix, _ = streams[key]; scores = retained_scores[key]
            observed = fresh_index[rid]['samples'] if condition == 'fresh' else allrows[key]['observed_samples']
            for ordinal, score in enumerate(scores):
                stream.write(struct.pack('<QI', offset+score['sample'], int(ordinal == 0)))
                stream.write(matrix[ordinal].astype('<f4', copy=False).tobytes())
                refs.write(canonical({**score, 'case_index': index, 'condition': condition, 'id': rid,
                    'local_sample': score['sample'], 'sample': offset+score['sample']}).decode()+'\n')
                predictions.write(canonical({'condition': condition, 'id': rid, **score}).decode()+'\n')
            case_map.append({'case_index': index, 'condition': condition, 'id': rid, 'offset': offset,
                'observed_samples': observed, 'embedding_rows': len(scores), 'ready_rows': sum(r['ready'] for r in scores)})
            offset += observed
    frozen.once(output/'runtime/CASE_MAP.json', case_map)
    need(frozen.file_identity(output/'models/synthetic-head.bin') == pins['bin'] and
         frozen.file_identity(output/'models/synthetic-head.npz') == pins['npz'], 'FROZEN_CANDIDATE_POST_EVALUATION')
    frozen.once(output/'evidence/TRAINING.json', {'schema': SCHEMA+'/training', 'recipe': RECIPE,
        'context': CONTEXT, 'candidate': pins, 'base_npz': frozen.BASE_NPZ, 'base_bin': frozen.BASE_BIN,
        'epochs': len(curves), 'optimizer_updates': progress['optimizer_updates_completed'],
        'selected_epoch': best_epoch, 'selected_key': best_key, 'stop_reason': stop_reason,
        'training_bag_forwards': progress['training_bag_forwards_completed'], 'bags_per_epoch': len(bags),
        'original_train_bags_replayed_every_epoch': 225, 'original_development_optimizer_uses': 0,
        'calibration_optimizer_uses': 0, 'locked_test_optimizer_uses': 0,
        'convergence_scope': RECIPE['plateau_claim'], 'fresh_role_summaries': summaries})
    return {'passed': True, 'status': 'NUMERICAL_SYNTHETIC192_TRAINING_COMPLETED', 'candidate': pins,
        'gates': gates, 'selected_epoch': best_epoch, 'actual_epochs': len(curves), 'stop_reason': stop_reason,
        'optimizer_updates': progress['optimizer_updates_completed'], 'progress': progress,
        'native_cases': len(native_keys), 'native_rows': total_rows,
        'locked_test_opened': (output/'evidence/LOCKED_TEST_OPENED.json').exists()}


def verify_native(raw, references):
    lines = raw.decode('ascii').splitlines()
    need(len(lines) == len(references) <= RECIPE['maximum_embedding_rows_native'], 'NATIVE_COMPLETE_DYNAMIC_ROWS')
    le = pe = 0.0; events = []
    for line, row in zip(lines, references):
        parts = line.split(); need(len(parts) == 9, 'NATIVE_ROW_WIDTH')
        need(tuple(map(int, parts[:3])) == (row['sample'], int(row['ready']), row['keyword']), 'NATIVE_EXACT_CLOCK_READY_EVENT')
        values = list(map(float, parts[3:])); need(all(math.isfinite(x) for x in values), 'NATIVE_FINITE')
        a = max(abs(x-y) for x, y in zip(values[:3], row['logits']))
        b = max(abs(x-y) for x, y in zip(values[3:], row['probabilities']))
        need(a <= 1e-4 and b <= 2e-6, 'UNCHANGED_NATIVE_NUMERIC_TOLERANCES')
        le, pe = max(le, a), max(pe, b)
        if row['keyword']:
            events.append({k: row[k] for k in ('case_index', 'condition', 'id', 'keyword', 'local_sample')})
    return {'passed': True, 'embedding_rows': len(lines), 'ready_rows': sum(r['ready'] for r in references),
        'head_cases': len({r['case_index'] for r in references}), 'max_abs_logit_error': le,
        'max_abs_probability_error': pe, 'exact_clock_ready_events': True, 'events': events,
        'scope': 'NATIVE_CACHED_HEAD_LEGACY_PLUS_NEW_SYNTHETIC', 'full_pcm_chain': 'NOT_RUN', 'arm': 'NOT_RUN'}
