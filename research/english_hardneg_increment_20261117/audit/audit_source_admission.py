"""Read saved bytes only; derive head inputs covered by nine owner-reviewed clips.

No audio codec, model, compiler, subprocess, network, or training library is used.
The owner statement is a separate provenance record, not an edit of the original
UNREVIEWED ledger/form. Do not extend it to the eight complete source utterances.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

REVIEW_PIN = {'bytes': 1865208, 'sha256': '23ff12e76b2a3c0cf2919f48c16712a374d186c8f5c2ac01af7add7e64005682'}
SCAN_PIN = {'bytes': 34035367, 'sha256': '03e2d7e1953ead05736dc2ff1de60c115bc24ee45442a2322886fc3d02050807'}
SDK_PIN = {'bytes': 10810227, 'sha256': 'b6411b7b859fa59147c7cd6091113ce6d83e544b75bc55bafa5726626b2ab511'}
OWNER_PIN = {'bytes': 10313, 'sha256': '0bec70e613449d04a744b7f07e2b0257bde7ed749bc82fe3a7800c9ffba7aaba'}
OWNER_REPOSITORY_PATH = 'research/english_hardneg_increment_20261117/OWNER_REVIEW.json'
MODEL_SHA = 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'
CONFIG_SHA = '13ff76c46b35d8d1c427bc237cfc3153771f281379ad86df8203f685aa2e7af4'
CODE_PINS = {
    'source/research/english_mel_specialization/fixed_frontend_mel.c': '5a476954887b59032ea5d419e97d6fb913695e3ad99f5d09164513a4cbd87a89',
    'source/research/english_final_embedding_frontend/fixed_frontend.h': '55ffd7edffd06744f9146088c4431635cd57498195cb05019df4e0d476e6cef9',
    'source/research/english_final_embedding/english_causal_head.c': 'e1e4a4cffd1758677ef820bd2183ec2ab3ab65d450af4d7eea35f565ebefa551',
    'source/research/english_final_embedding/english_causal_head.h': '557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9',
    'source/research/english_stream_integration/ekws_research_stream.c': 'a9f8339f483c1e635b25f710fcb0f982ec06cf1d95598da29e973c1771da3b25',
    'source/research/english_stream_integration/CMakeLists.txt': None,
    'generated/fixed_models.c': '9bf740a692370dfed02f034f87f6f0dfcaac42f3f83152c79fd27088ccc71941',
    'generated/export_manifest.json': 'b94549aec824a8eddb014e19f01629d8733c35f809d763927b712df845b5677b',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def pairs(values):
    out = {}
    for key, value in values:
        need(key not in out, 'duplicate JSON key: ' + key)
        out[key] = value
    return out


def loads(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


class Archive:
    def __init__(self, path, pin, expected_payloads=None):
        raw = Path(path).read_bytes()
        need(identity(raw) == pin, 'archive byte identity: ' + str(path))
        self.identity = pin
        self.files = {}
        with zipfile.ZipFile(Path(path)) as archive:
            for info in archive.infolist():
                name = info.filename
                need(not name.startswith('/') and '\\' not in name and
                     all(x not in ('', '.', '..') for x in name.rstrip('/').split('/')),
                     'unsafe ZIP member')
                need(not info.flag_bits & 1 and not stat.S_ISLNK(info.external_attr >> 16),
                     'encrypted/symlink ZIP member')
                if info.is_dir():
                    continue
                need(name not in self.files, 'duplicate ZIP member')
                self.files[name] = archive.read(info)
        self.manifest = loads(self.files['MANIFEST.json'])
        entries = self.manifest['files']
        if isinstance(entries, list):
            pins = {x['path']: {'bytes': x['bytes'], 'sha256': x['sha256']} for x in entries}
            need(len(pins) == len(entries), 'duplicate manifest path')
        else:
            pins = entries
        need(set(self.files) == set(pins) | {'MANIFEST.json'}, 'manifest exact closure')
        if expected_payloads is not None:
            need(len(pins) == expected_payloads, 'unexpected manifest payload count')
        for name, pin in pins.items():
            need(identity(self.files[name]) == pin, 'manifest hash: ' + name)
        self.payloads = len(pins)

    def read(self, name):
        return self.files[name]

    def json(self, name):
        return loads(self.read(name))


def pcm_wav_payload(raw):
    """Inspect RIFF headers and copy the already uncompressed PCM payload only."""
    need(raw[:4] == b'RIFF' and raw[8:12] == b'WAVE' and
         struct.unpack_from('<I', raw, 4)[0] + 8 == len(raw), 'WAV RIFF framing')
    pos, chunks = 12, {}
    while pos < len(raw):
        need(pos + 8 <= len(raw), 'WAV chunk header')
        tag, count = struct.unpack_from('<4sI', raw, pos)
        pos += 8
        need(tag not in chunks and pos + count <= len(raw), 'WAV chunk identity/bounds')
        chunks[tag] = raw[pos:pos + count]
        pos += count + (count & 1)
    need(pos == len(raw) and set(chunks) == {b'fmt ', b'data'}, 'WAV chunks')
    need(len(chunks[b'fmt ']) == 16 and
         struct.unpack('<HHIIHH', chunks[b'fmt ']) == (1, 1, 16000, 32000, 2, 16),
         'uncompressed mono 16k PCM16 WAV only')
    return chunks[b'data']


def parse_trace(raw, samples):
    need(raw[:8] == b'EKWSTRC1' and struct.unpack_from('<6IQ', raw, 8) ==
         (1, 1, 16000, 320, 96, 3, samples), 'trace header')
    count, pos = 1 + (samples - 16000) // 1280, 40
    embeddings, scores = [], []

    def take(kind, length):
        nonlocal pos
        need(pos + 5 + length <= len(raw) and struct.unpack_from('<BI', raw, pos) ==
             (kind, length), 'trace record order/length')
        start = pos + 5
        payload = raw[start:start + length]
        pos = start + length
        return payload, start

    for i in range(count):
        payload, offset = take(1, 400)
        ordinal, sample = struct.unpack_from('<QQ', payload)
        values = struct.unpack_from('<96f', payload, 16)
        need(ordinal == i and sample == 16000 + 1280 * i and all(map(math.isfinite, values)),
             'embedding finite/ordinal/clock')
        embeddings.append({'sample': sample, 'ordinal': i, 'raw': payload[16:],
                           'trace_vector_offset': offset + 16})
        payload, offset = take(2, 48)
        ordinal, ready, keyword, when = struct.unpack_from('<QIIQ', payload)
        values = struct.unpack_from('<6f', payload, 24)
        need((ordinal, ready, keyword, when) == (i, int(i >= 15), 0, sample),
             'zero-event saved score state/clock')
        need(all(map(math.isfinite, values)) and all(0 <= p <= 1 for p in values[3:]),
             'saved score numeric domain')
        need(abs(sum(values[3:]) - 1) < 1e-5 if ready else all(x == 0 for x in values),
             'saved score probability/zero startup')
        scores.append({'sample': sample, 'ready': bool(ready), 'probabilities': list(values[3:])})
    payload, _ = take(255, 48)
    need(struct.unpack('<6Q', payload) ==
         (samples, count, count, count - 15, 0, (samples + 319) // 320), 'trace END')
    need(pos == len(raw), 'trace trailing bytes')
    return embeddings, scores


def source_proof(sdk, scan):
    need(sdk.read('MANIFEST.json') == scan.read('installed-base/ORIGINAL_MANIFEST.json'),
         'SDK original manifest identity')
    library = 'sdk-native/lib/libekws_research_stream.a'
    need(sdk.read(library) == scan.read('installed-base/' + library), 'linked native library bytes')
    tree = ast.parse(scan.read('source/automation/english_hardneg_natural32_retained_once.py'))
    sources = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'SOURCES' for x in node.targets):
            sources = ast.literal_eval(node.value)
    need(sources is not None and {x: sources['sdk'][x] for x in ('bytes', 'sha256')} == SDK_PIN,
         'consumed SDK source archive pin')
    files = {}
    for path, expected in CODE_PINS.items():
        raw = sdk.read(path)
        need(expected is None or sha(raw) == expected, 'pinned C source: ' + path)
        files[path] = identity(raw)
    runtime = sdk.read('source/research/english_mel_specialization/fixed_frontend_mel.c').decode()
    graph = sdk.read('generated/fixed_models.c').decode()
    head = sdk.read('source/research/english_final_embedding/english_causal_head.c').decode()
    for literal in (
        'pcm[i] = (float)s->ring[(s->write + i) % EF_PCM_WINDOW]',
        's->next_end += EF_HOP',
        'mel[(EF_MEL_FRAMES - EF_EMBED_FRAMES) * EF_MEL_BINS + i] / 10.0f + 2.0f',
        'for (i = 0; i < x->count; ++i)',
        'for (frame = 0; frame < 97; ++frame)',
        'const float *window = a + frame * 160',
        'for (tap = 0; tap < 512; ++tap)',
    ):
        need(literal in runtime, 'frontend source anchor: ' + literal)
    need('{EF_REDUCE_MAX, 1, {25, EF_NONE' in graph and
         '26, {15, 0, 0, 0' in graph and
         '{EF_CLIP, 3, {25, 29, EF_NONE' in graph and
         '{4, {1, 1, 97, 32, 0, 0}, 3104' in graph,
         'global 97x32 reduction and output clipping')
    need('s->history[(s->next + t) % EHEAD_HISTORY]' in head and
         'memcpy(s->history[s->next], normalized, sizeof normalized)' in head,
         'head circular history anchors')
    return {
        'sdk_archive': SDK_PIN, 'sdk_consumed_source_pin': sources['sdk'],
        'sdk_manifest_sha256': sha(sdk.read('MANIFEST.json')),
        'consumed_native_library': identity(sdk.read(library)), 'source_files': files,
        'pipeline_source_commit': sdk.json('evidence/RESULT.json')['source_commit'],
        'sample_rate': 16000, 'embedding_hop_samples': 1280,
        'embedding_width': 96, 'head_embedding_rows': 16,
        'frontend_exact_read_buffer_samples': 16000,
        'frontend_input_interval_at_embedding_sample_t': '[t-16000,t)',
        'head_embedding_samples_at_ready_s': '[s-15*1280,...,s] in ascending order',
        'head_complete_frontend_input_union_samples': 16000 + 15 * 1280,
        'admission_interval_at_ready_s': '[s-35200,s)',
        'admission_interval_is_conservative_complete_input_envelope': True,
        'minimal_nonzero_weight_receptive_field_claimed': False,
        'graph_structural_conv_support_within_one_read_buffer': [0, 15872],
        'trailing_read_samples_not_used_by_any_512_stride160_conv': 128,
        'trailing_128_samples_removed_for_admission': False,
        'global_mel_dependency': 'Both 512-tap stride160 convolutions produce 97 frames; '
            'ReduceMax over all 97x32 log-mel values sets the floor of the final Clip before '
            'the last 76 frames are selected. Earlier 21 frames may affect selected features.',
        'last76_local_only_support_is_rejected': True,
        'proof_is_not_inferred_from_first_ready_sample': True,
    }


def audit(review_path, scan_path, sdk_path, review_receipt):
    review = Archive(review_path, REVIEW_PIN, 29)
    scan = Archive(scan_path, SCAN_PIN, 659)
    sdk = Archive(sdk_path, SDK_PIN)
    proof = source_proof(sdk, scan)
    reference = review.json('03_reference/REFERENCE.json')
    owner_raw = Path(review_receipt).read_bytes()
    need(identity(owner_raw) == OWNER_PIN, 'public owner review receipt pin')
    owner = loads(owner_raw)
    need(owner['schema'] == 'english-natural32-owner-negative-review-v1' and
         owner['owner_statement_verbatim'] == '没有唤醒词' and
         {k: owner['review_packet'][k] for k in ('bytes', 'sha256')} == REVIEW_PIN and
         owner['full_original_utterance_labels_asserted'] is False and
         owner['per_file_listening_process_asserted'] is False and
         owner['positive_pronunciation_gold_created'] is False and
         owner['independent_test_admitted'] is False and
         owner['assistant_listening_performed'] is False,
         'owner receipt scope and nonclaims')
    ledger_raw = scan.read('candidates/LEDGER.json')
    ledger = loads(ledger_raw)
    need(reference['original_artifact_sha256'] == SCAN_PIN['sha256'] and
         reference['original_artifact_id'] == 14318639588 and
         reference['original_ledger_sha256'] == sha(ledger_raw), 'review to successful scan binding')
    need(ledger['model_sha256'] == MODEL_SHA and ledger['config_sha256'] == CONFIG_SHA,
         'frozen model/config binding')
    need(sha(scan.read('models/coverage-head.bin')) == MODEL_SHA, 'actual model byte binding')
    candidates = sorted(ledger['candidates'], key=lambda c: c['global_priority_rank'])
    need(len(candidates) == len(reference['candidates']) == 9, 'nine review windows only')
    owner_by_id = {c['candidate_id']: c for c in owner['candidates']}
    need(len(owner_by_id) == len(owner['candidates']) == 9 and
         set(owner_by_id) == {c['candidate_id'] for c in candidates}, 'nine owner decisions')
    ref_by_id = {c['candidate_id']: c for c in reference['candidates']}
    need(set(ref_by_id) == {c['candidate_id'] for c in candidates}, 'review candidate IDs')
    copied = {c['path']: c for c in reference['copied_files']}
    need(len(copied) == len(reference['copied_files']) == 25, 'review copied file set')
    for path, record in copied.items():
        raw = review.read(path)
        need(identity(raw) == {k: record[k] for k in ('bytes', 'sha256')} and
             raw == scan.read(record['original_member']), 'original copied bytes: ' + path)
    cache, rows, by_source = {}, [], defaultdict(list)
    for candidate in candidates:
        ref = ref_by_id[candidate['candidate_id']]
        decision = owner_by_id[candidate['candidate_id']]
        for key in ('recording_id', 'source_member', 'start_sample', 'end_sample_exclusive', 'global_priority_rank'):
            need(ref[key] == candidate[key], 'review candidate coordinate: ' + key)
        for key in ('recording_id', 'source_member', 'source_pcm_sha256', 'window_pcm_sha256',
                    'start_sample', 'end_sample_exclusive', 'speaker_id', 'source_role'):
            need(decision[key] == candidate[key], 'owner candidate coordinate/hash: ' + key)
        need(decision['review_decision'] == 'N' and decision['trusted_negative'] is True and
             decision['research_training_admitted'] is True and
             decision['negative_targets'] == ['Hey Vosen', 'Hey Mero'], 'owner N for both targets')
        rid = candidate['recording_id']
        start, end = candidate['start_sample'], candidate['end_sample_exclusive']
        need(candidate['source_role'] == 'train' and candidate['source_utterance_continuity'] == 'natural_individual',
             'original train role/natural continuity')
        if rid not in cache:
            pcm = scan.read('selectedinputs/pcm/' + rid + '.pcm')
            trace = scan.read('traces/' + rid + '.trace')
            need(sha(pcm) == candidate['source_pcm_sha256'] and
                 len(pcm) == candidate['source_samples'] * 2 and
                 sha(trace) == candidate['original_c_trace_sha256'], 'source PCM/trace identity')
            embeddings, scores = parse_trace(trace, candidate['source_samples'])
            cache[rid] = (pcm, trace, embeddings, scores)
        pcm, trace, embeddings, scores = cache[rid]
        clip_pcm = pcm_wav_payload(review.read(ref['audio_path']))
        need(identity(review.read(ref['audio_path'])) ==
             {'bytes': decision['wav_bytes'], 'sha256': decision['wav_sha256']}, 'owner reviewed WAV identity')
        need(clip_pcm == pcm[start * 2:end * 2] and sha(clip_pcm) == candidate['window_pcm_sha256'],
             'review WAV exactly equals source PCM candidate slice')
        eligible = []
        for index, row in enumerate(scores):
            sample = row['sample']
            if not (row['ready'] and sample - 35200 >= start and sample <= end):
                continue
            selected = embeddings[index - 15:index + 1]
            need(len(selected) == 16 and [r['sample'] for r in selected] ==
                 list(range(sample - 15 * 1280, sample + 1, 1280)), 'complete contiguous saved 16x96')
            need(all(start <= r['sample'] - 16000 and r['sample'] <= end for r in selected),
                 'each actual frontend read buffer stays inside human-reviewed window')
            raw = b''.join(r['raw'] for r in selected)
            eligible.append({
                'sample': sample, 'embedding_first_ordinal': index - 15, 'embedding_last_ordinal': index,
                'first_embedding_sample': selected[0]['sample'], 'last_embedding_sample': sample,
                'full_pcm_support_start': sample - 35200, 'full_pcm_support_end_exclusive': sample,
                'full_pcm_support_sha256': sha(pcm[(sample - 35200) * 2:sample * 2]),
                'raw_embedding_f32le_bytes': len(raw), 'raw_embedding_f32le_sha256': sha(raw),
            })
        need(eligible, 'reviewed window has no contained head input')
        eligible_samples = {r['sample'] for r in eligible}
        need(all(seed['sample'] in eligible_samples for seed in candidate['seeds']),
             'all original candidate seeds retained by complete-input containment')
        row = {k: candidate[k] for k in (
            'candidate_id', 'recording_id', 'source_member', 'speaker_id', 'source_family_id',
            'source_pcm_sha256', 'original_c_trace_sha256', 'window_pcm_sha256',
            'start_sample', 'end_sample_exclusive', 'global_priority_rank',
            'pre_context_clipped_at_source_start', 'post_context_clipped_at_source_end')}
        row.update({
            'review_packet_audio_path': ref['audio_path'],
            'review_packet_audio': identity(review.read(ref['audio_path'])),
            'source_role': 'train', 'use_role': 'research_train_only',
            'review_status': 'OWNER_CONFIRMED_NO_WAKE_WORD', 'admitted_class_id': 0,
            'owner_review_decision': 'N', 'trusted_negative': True,
            'research_training_admitted': True,
            'label_scope': 'THIS_REVIEWED_PCM_WINDOW_ONLY',
            'whole_original_negative_admitted': False, 'human_pronunciation_gold': False,
            'eligible_ready_samples': [r['sample'] for r in eligible],
            'eligible_head_input_count': len(eligible), 'eligible_head_inputs': eligible,
            'original_candidate_seeds': candidate['seeds'],
            'all_original_candidate_seeds_retained': True,
            'trace_member': 'traces/' + rid + '.trace', 'source_samples': len(pcm) // 2,
        })
        rows.append(row)
        by_source[rid].append(row)
    sources = []
    for rid, group in sorted(by_source.items()):
        selected = {}
        for row in group:
            for head in row['eligible_head_inputs']:
                need(head['sample'] not in selected or selected[head['sample']] == head, 'same-source inconsistent duplicate')
                selected[head['sample']] = head
        raw = cache[rid][0]
        sources.append({
            'recording_id': rid, 'source_member': group[0]['source_member'],
            'speaker_id': group[0]['speaker_id'], 'source_role': 'train',
            'candidate_ids': [r['candidate_id'] for r in group],
            'candidate_windows': [[r['start_sample'], r['end_sample_exclusive']] for r in group],
            'original_samples': len(raw) // 2, 'original_seconds': len(raw) / 32000,
            'whole_original_negative_admitted': False,
            'eligible_ready_samples': sorted(selected), 'deduplicated_head_input_count': len(selected),
            'duplicate_head_inputs_removed': sum(r['eligible_head_input_count'] for r in group) - len(selected),
            'recommended_equal_source_weight': {'numerator': 1, 'denominator': 8},
            'same_source_windows_independent_examples': False,
        })
    need([r['eligible_head_input_count'] for r in rows] == [17,17,17,16,16,15,5,16,15], 'fixed candidate admission counts')
    need(len(sources) == 8 and sum(r['deduplicated_head_input_count'] for r in sources) == 134,
         'fixed source/admitted input totals')
    return {
        'schema': 'english-reviewed-candidate-source-admission-v1',
        'status': 'OWNER_SCOPE_AND_SAVED_INPUTS_VERIFIED',
        'source_audit_passed': True,
        'review_evidence': {
            'source': 'explicit owner/user feedback bound by the frozen public OWNER_REVIEW.json',
            'user_statement_zh': '没有唤醒词',
            'accepted_scope': 'exactly the nine PCM windows in the identified review package',
            'review_package': REVIEW_PIN,
            'owner_review_receipt': {'repository_path': OWNER_REPOSITORY_PATH, 'identity': OWNER_PIN},
            'materialized_review_package_exactly_matches_owner_receipt': True,
            'private_download_identity_published': False,
            'assistant_listening_performed': False, 'per_file_listening_times': None,
            'per_file_review_method': None, 'listening_to_complete_originals_claimed': False,
            'individual_phonetic_gold_claimed': False,
            'historical_unreviewed_ledger_and_review_form_rewritten': False,
            'current_target_policy': owner['target_policy'],
        },
        'source_artifact': dict(SCAN_PIN, run_id=44063710088, artifact_id=14318639588,
            head_sha='4f998e89d9f344677b8a97fdb78352e42b8cf6bd'),
        'source_ledger_sha256': sha(ledger_raw),
        'review_reference_sha256': sha(review.read('03_reference/REFERENCE.json')),
        'manifest_closed_payload_counts': {'review': review.payloads, 'successful_scan': scan.payloads, 'sdk': sdk.payloads},
        'model_sha256': MODEL_SHA, 'config_sha256': CONFIG_SHA, 'roster_sha256': ledger['roster_sha256'],
        'receptive_field_source_proof': proof,
        'selection': {
            'rule': 'Use each original ready row iff start_sample <= sample-35200 and sample <= end_sample_exclusive; require all 16 original saved embeddings.',
            'window_shape': [16,96], 'dtype': 'IEEE754 little-endian float32',
            'embedding_hop_samples': 1280, 'head_input_pcm_support_samples': 35200,
            'original_clock_preserved': True, 'raw_embedding_bytes_preserved': True,
            'new_frontend_calls': 0, 'crop_and_reset_frontend': False,
            'padding_or_synthetic_history': False, 'selection_uses_scores_or_peaks': False,
            'deduplication_key': ['recording_id', 'sample'],
            'source_balance_recommendation': 'Eight source bags with equal source weight or deterministic round-robin; mean over all admitted head inputs within each source. The two 5536 windows belong to one bag.',
        },
        'counts': {
            'reviewed_candidate_windows': 9, 'source_originals': 8,
            'reviewed_window_samples': sum(c['end_sample_exclusive']-c['start_sample'] for c in rows),
            'reviewed_window_seconds': sum(c['end_sample_exclusive']-c['start_sample'] for c in rows)/16000,
            'original_seconds_for_reference_only': sum(s['original_samples'] for s in sources)/16000,
            'eligible_head_inputs_before_deduplication': sum(c['eligible_head_input_count'] for c in rows),
            'eligible_head_inputs_after_deduplication': sum(s['deduplicated_head_input_count'] for s in sources),
            'original_candidate_seeds_retained': sum(len(c['original_candidate_seeds']) for c in rows),
            'original_candidate_seeds_dropped': 0,
            'full_original_negative_labels_admitted': 0,
        },
        'candidates': rows, 'sources': sources,
        'boundaries': {
            'research_training_source_admission_only': True,
            'independent_evaluation_source': False, 'independent_far_denominator': False,
            'positive_pronunciation_admission': False, 'historical_IPA_promoted_to_formal_standard': False,
            'whole_source_utterance_labels_extended': False,
            'optimizer_updates_executed': 0, 'model_calls_executed': 0,
            'audio_decodes_executed': 0, 'compiler_calls_executed': 0,
            'original_artifact_or_labels_modified': False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-package', type=Path, required=True)
    parser.add_argument('--scan-artifact', type=Path, required=True)
    parser.add_argument('--installed-sdk', type=Path, required=True)
    parser.add_argument('--review-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'output must be new')
    result = audit(args.review_package, args.scan_artifact, args.installed_sdk, args.review_receipt)
    raw = canonical(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'output': str(args.output), 'identity': identity(raw), 'status': result['status'], 'counts': result['counts']}, sort_keys=True))


if __name__ == '__main__':
    main()
