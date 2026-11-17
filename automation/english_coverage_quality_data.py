"""New train acoustic conditions and one fixed development challenge.

No model imports or calls. Existing originals, role assignments, labels, prefix,
context, and head-instance clocks are immutable. New conditions are not new
independent sources. Old 5 dB evaluation condition IDs never become training.
"""
from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCHEMA = 'english-coverage-condition-data-v1'
FROZEN_HELPER_SHA256 = 'c2dfae52eda3a3adda952dfd23a250054540f517d6e6fc516aaf9dd2bc915d88'
FROZEN_HELPER_BYTES = 16372
DOMAIN = 'english-coverage-new-acoustic-conditions-20261117-v1'
PURPOSES = {
    'train': {'role': 'train', 'conditions': (('new5', 5.0), ('new15', 15.0)),
              'sources': 50, 'planned_conditions': 100, 'planned_rows': 3666,
              'maximum_new_embedding_calls': 2066},
    'development_challenge': {'role': 'development', 'conditions': (('new8', 8.0),),
              'sources': 27, 'planned_conditions': 27, 'planned_rows': 1001,
              'maximum_new_embedding_calls': 569},
}
FALSE_FLAGS = ('formal_training_admitted', 'training_admitted', 'human_gold',
               'trusted_negative', 'pronunciation_verified',
               'model_promotion_allowed', 'qualification_allowed')
PREFIX_REUSE_ROWS = 16


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def _shared():
    here = Path(__file__).resolve()
    choices = (here.with_name('english_temporal_quality_data.py'),
               here.parents[2] / 'english_temporal/automation/english_temporal_quality_data.py')
    path = next((p for p in choices if p.is_file()), None)
    need(path is not None and not path.is_symlink(), 'frozen data helper available')
    raw = path.read_bytes()
    need(len(raw) == FROZEN_HELPER_BYTES and sha(raw) == FROZEN_HELPER_SHA256,
         'frozen data helper byte identity')
    spec = importlib.util.spec_from_file_location('_coverage_frozen_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _shared()


def mix_pcm16(source_pcm, noise_pcm, snr_db):
    """Existing full-clip RMS arithmetic, one prespecified condition at a time."""
    need(snr_db in (5.0, 8.0, 15.0), 'only frozen new SNR values')
    signal, noise = shared.pcm_values(source_pcm), shared.pcm_values(noise_pcm)
    need(len(signal) == len(noise), 'same signal/noise sample count')
    ps = math.fsum(float(x) ** 2 for x in signal) / len(signal)
    pn = math.fsum(float(x) ** 2 for x in noise) / len(noise)
    receipt = {'snr_definition': 'full-original-clip-rms-before-mixture',
        'target_snr_db': snr_db, 'samples': len(signal), 'signal_power': ps,
        'noise_power': pn, 'source_pcm_sha256': sha(source_pcm),
        'noise_pcm_sha256': sha(noise_pcm), 'gain_on_signal': 1.0,
        'clipped_samples': 0, 'fade_samples': 0}
    if ps == 0 or pn == 0:
        return None, {**receipt, 'status': 'MIX_ZERO_ENERGY_EXCLUDED'}
    gain = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    values = [round(float(s) + gain * float(n)) for s, n in zip(signal, noise)]
    receipt.update(noise_gain=gain, unbounded_peak=max(abs(x) for x in values))
    if min(values) < -32768 or max(values) > 32767:
        return None, {**receipt, 'status': 'MIX_HEADROOM_EXCLUDED'}
    actual_power = math.fsum(float(m - s) ** 2 for m, s in zip(values, signal)) / len(signal)
    if actual_power == 0:
        return None, {**receipt, 'status': 'MIX_QUANTIZED_NOISE_ZERO_EXCLUDED'}
    actual_snr = 10 * math.log10(ps / actual_power)
    if abs(actual_snr - snr_db) > 0.05:
        return None, {**receipt, 'status': 'MIX_QUANTIZATION_SNR_EXCLUDED',
                      'actual_snr_db': actual_snr}
    result = array('h', values)
    if sys.byteorder != 'little':
        result.byteswap()
    raw = result.tobytes()
    return raw, {**receipt, 'status': 'MIXED', 'actual_snr_db': actual_snr,
                 'mixed_pcm_sha256': sha(raw)}


def select_noise(parent_id, purpose, condition, frames, tracks, excluded_tracks):
    need(purpose in PURPOSES, 'frozen purpose')
    spec = PURPOSES[purpose]
    need(condition in dict(spec['conditions']), 'condition belongs to purpose')
    eligible = sorted((row for row in tracks
        if row['role'] == spec['role'] and row['frames'] >= frames
        and row['id'] not in excluded_tracks), key=lambda row: row['id'])
    need(eligible and len({r['id'] for r in eligible}) == len(eligible),
         'same-role distinct eligible noise tracks')
    digest = sha((DOMAIN + '/' + purpose + '/' + condition + '/' + parent_id).encode())
    track = eligible[int(digest[:16], 16) % len(eligible)]
    offset = int(digest[16:32], 16) % (track['frames'] - frames + 1)
    return track, offset


def condition_id(parent_id, purpose, condition, source_sha, track, offset):
    request = {'schema': SCHEMA, 'domain': DOMAIN, 'parent_id': parent_id,
        'purpose': purpose, 'condition': condition,
        'source_pcm_sha256': source_sha, 'noise_track': track['id'],
        'noise_track_pcm_sha256': track['pcm_sha256'], 'noise_start_sample': offset,
        'target_snr_db': dict(PURPOSES[purpose]['conditions'])[condition]}
    return 'encov-' + condition + '-' + sha(canonical(request))[:24]


def prefix_reuse_windows(clean_pcm, new_pcm):
    """Only exact one-second inputs up to original clip start are reusable."""
    need(len(clean_pcm) >= 70400 and len(new_pcm) == len(clean_pcm), 'same observed geometry')
    need(clean_pcm[:70400] == new_pcm[:70400], 'unchanged complete original prefix')
    records = []
    for end in range(16000, 35201, 1280):
        left, right = 2 * (end - 16000), 2 * end
        original, new = clean_pcm[left:right], new_pcm[left:right]
        need(len(original) == 32000 and original == new, 'byte-equal frozen frontend input')
        records.append({'available_sample': end, 'input_pcm_bytes': 32000,
                        'input_pcm_sha256': sha(original)})
    need(len(records) == PREFIX_REUSE_ROWS, 'sixteen prefix cache rows')
    return records


def _group_counts(rows):
    return dict(sorted(Counter(str(r['condition']) + '/' + r['role'] + '/' +
                               str(r['weak_class']) for r in rows).items()))


def training_coverage(rows):
    groups = {}
    checks = []
    for condition in ('new5', 'new15'):
        for label in range(3):
            group = [r for r in rows if r['condition'] == condition and r['weak_class'] == label]
            families = sorted({r['mix']['source_family_id'] for r in group})
            sources = sorted({r['parent_id'] for r in group})
            texts = sorted({r['mix']['source_synthesis_text'] for r in group})
            accepted = len(sources) >= 2 and len(families) >= 2 if label else \
                len(families) == 4 and len(texts) == 8
            checks.append(accepted)
            groups[condition + '/' + str(label)] = {'sources': len(sources),
                'families': families, 'source_texts': texts, 'coverage_passed': accepted}
    return {'passed': all(checks), 'groups': groups,
            'minimum_positive_source_count': 2, 'minimum_positive_family_count': 2,
            'negative_all_four_families_and_eight_templates_required': True}


def build_conditions(training_zip, incumbent_zip, *, purpose):
    """Return summary, all planned receipts, eligible rows, PCM keyed by new ID.

    Called on public Actions for construction. Independent auditing may rebuild
    the same PCM with these standard-library operations; no model is invoked.
    The challenge call must be made only after the worker freezes its candidate.
    """
    need(purpose in PURPOSES, 'frozen purpose required')
    spec = PURPOSES[purpose]
    training = shared.pinned_archive(training_zip, shared.TRAINING_IDENTITY)
    previous = shared.pinned_archive(incumbent_zip, shared.PREVIOUS_IDENTITY)
    selection = training.json('SELECTION.json')
    selected_ids = set(selection['eligible_cell_ids'])
    roster = training.json('STREAM_ROSTER.json')
    parents = [r for r in roster if r['id'] in selected_ids]
    need(len(roster) == 117 and len(parents) == len(selected_ids) == 77,
         'unchanged 117 streams and 77 admitted weak research intentions')
    need(shared.grouped_counts(parents) == {'train/0': 32, 'train/1': 8, 'train/2': 10,
        'development/0': 16, 'development/1': 3, 'development/2': 8}, 'unchanged role/class roster')
    selected_parents = [r for r in parents if r['role'] == spec['role']]
    need(len(selected_parents) == spec['sources'], 'purpose source denominator')
    source_rows, originals = shared.source_pcm(training, {r['id'] for r in selected_parents})
    libri = training.json('LIBRISPEECH_SOURCES.json')
    tracks = libri['actual_research_tracks']
    track_by_id = {r['id']: r for r in tracks}
    need(len(tracks) == len(track_by_id) == 40 and Counter(r['role'] for r in tracks)
         == {'train': 30, 'development': 10}, 'original noise role partition')
    need(libri['source_pool_is_previously_exposed'] is True, 'exposed background source status')
    background_manifest = previous.json('background_tracks/MANIFEST.json')['files']
    background = {}
    for track in tracks:
        name = track['id'] + '.pcm'
        raw = previous.read('background_tracks/' + name)
        recorded = background_manifest[name]
        need(len(raw) == track['frames'] * 2 == recorded['bytes'] == 512000
             and sha(raw) == track['pcm_sha256'] == recorded['sha256']
             and recorded['role'] == track['role'] and recorded['speaker'] == track['speaker'],
             'exact background PCM source and role')
        background[track['id']] = raw
    old_receipts = previous.json('evidence/MIX_ROSTER.json')
    old_by_id = {r['id']: r for r in old_receipts}
    need(len(old_by_id) == len(old_receipts) == 77 and set(old_by_id) == selected_ids,
         'all original noisy receipts including exclusions')
    receipts, eligible, pcm_by_key = [], [], {}
    for parent in selected_parents:
        pid, src, role = parent['id'], parent['source'], parent['role']
        original, original_meta, old = originals[pid], source_rows[pid], old_by_id[pid]
        need(sha(original) == src['source_pcm_sha256'] == old['source_pcm_sha256']
             and original_meta['role'] == role == old['role']
             and original_meta['source_family_id'] == old['source_family_id']
             and parent['weak_class'] == old['weak_class'], 'parent identity/role/label/family unchanged')
        def span(kind):
            track = track_by_id[src[kind + '_track']]
            start, count = src[kind + '_start_sample'], src[kind + '_samples']
            need(track['role'] == src[kind + '_role'] == role
                 and 0 <= start < start + count <= track['frames'], 'same-role real prefix/context')
            return background[track['id']][2 * start:2 * (start + count)]
        prefix, context = span('prefix'), span('context')
        clean = prefix + original + context
        need(len(prefix) == 70400 and len(context) == 10240
             and len(clean) == 2 * parent['observed_samples']
             and sha(clean) == src['combined_pcm_sha256'] == old['original_combined_pcm_sha256'],
             'original complete source/real-context stream')
        excluded_tracks = {old['noise_track']}
        need(track_by_id[old['noise_track']]['role'] == role, 'old noisy track role')
        for condition, snr_db in spec['conditions']:
            track, offset = select_noise(pid, purpose, condition, len(original) // 2,
                                         tracks, excluded_tracks)
            excluded_tracks.add(track['id'])  # Preserve the selection even if excluded for headroom.
            noise = background[track['id']][offset * 2:offset * 2 + len(original)]
            cid = condition_id(pid, purpose, condition, sha(original), track, offset)
            mixed, receipt = mix_pcm16(original, noise, snr_db)
            receipt.update(id=cid, condition_id=cid, parent_id=pid, condition=condition,
                purpose=purpose, role=role, weak_class=parent['weak_class'],
                source_family_id=original_meta['source_family_id'],
                source_synthesis_text=original_meta['synthesis_text'],
                source_variant=original_meta['variant'],
                canonical_phrase=original_meta['canonical_phrase'],
                target_ipa=original_meta['target_ipa'],
                noise_track=track['id'], noise_role=track['role'], noise_start_sample=offset,
                noise_track_pcm_sha256=track['pcm_sha256'],
                old_snr10_noise_track=old['noise_track'],
                old_snr10_noise_start_sample=old['noise_start_sample'],
                old_snr10_receipt_sha256=sha(canonical(old)),
                original_combined_pcm_sha256=sha(clean), new_noise_domain=DOMAIN,
                prefix_pcm_sha256=sha(prefix), context_pcm_sha256=sha(context),
                prefix_samples=35200, context_samples=5120,
                independent_source_count_increment=0,
                condition_training_allowed=(purpose == 'train'),
                old_snr5_cache_used_for_training=False,
                **{k: False for k in FALSE_FLAGS})
            if mixed is not None:
                combined = prefix + mixed + context
                windows = prefix_reuse_windows(clean, combined)
                receipt.update(combined_mixed_pcm_sha256=sha(combined),
                    prefix_cache_rows_reused=16, prefix_input_windows=windows)
                need(cid not in pcm_by_key, 'unique new condition identity')
                # first_record/offset remain source metadata; feature writer reindexes its new cache.
                row = {**parent, 'id': cid, 'condition_id': cid, 'parent_id': pid,
                       'condition': condition, 'mix': receipt}
                eligible.append(row)
                pcm_by_key[cid] = combined
            receipts.append(receipt)
    need(len(receipts) == len({r['id'] for r in receipts}) == spec['planned_conditions'],
         'complete planned condition denominator')
    need(all(r['id'] not in selected_ids for r in receipts), 'new IDs distinct from all original IDs')
    need(set(pcm_by_key) == {r['id'] for r in eligible}, 'eligible PCM and roster closure')
    planned_rows = sum(parent['records'] for parent in selected_parents) * len(spec['conditions'])
    need(planned_rows == spec['planned_rows'], 'fixed original clock row ceiling')
    eligible_rows = sum(r['records'] for r in eligible)
    reused_rows = PREFIX_REUSE_ROWS * len(eligible)
    needed_rows = eligible_rows - reused_rows
    need(0 <= needed_rows <= spec['maximum_new_embedding_calls'], 'bounded new suffix embeddings')
    summary = {'schema': SCHEMA, 'status': 'FIXED_NEW_CONDITIONS_CONSTRUCTED',
        'purpose': purpose, 'role': spec['role'], 'noise_domain': DOMAIN,
        'source_cases': spec['sources'], 'planned_conditions': len(receipts),
        'eligible_conditions': len(eligible), 'excluded_conditions': len(receipts) - len(eligible),
        'planned_embedding_rows': planned_rows, 'eligible_embedding_calls': eligible_rows,
        'eligible_embedding_rows': eligible_rows, 'prefix_cache_rows_reused': reused_rows,
        'new_embedding_calls_required': needed_rows,
        'maximum_new_embedding_calls': spec['maximum_new_embedding_calls'],
        'onnx_forward_calls_required': 2 * needed_rows,
        'planned_by_condition_role_class': _group_counts(receipts),
        'eligible_by_condition_role_class': _group_counts(eligible),
        'excluded_statuses': dict(sorted(Counter(r['status'] for r in receipts
                                               if r['status'] != 'MIXED').items())),
        'condition_ids': [r['id'] for r in receipts],
        'eligible_condition_ids': [r['id'] for r in eligible],
        'receipts_semantic_sha256': sha(canonical(receipts)),
        'roster_semantic_sha256': sha(canonical(eligible)),
        'original_training_source': shared.TRAINING_IDENTITY,
        'incumbent_source': shared.PREVIOUS_IDENTITY,
        'same_source_condition_is_not_independent_audio': True,
        'independent_source_count_increment': 0, 'new_tts_asr_phone_calls': 0,
        'old_snr5_cache_used_for_training': False,
        'development_challenge_training_allowed': False,
        'condition_training_allowed': purpose == 'train',
        'overflow_policy': 'EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING',
        **{k: False for k in FALSE_FLAGS}}
    if purpose == 'train':
        summary['training_coverage'] = training_coverage(eligible)
    return summary, receipts, eligible, pcm_by_key
