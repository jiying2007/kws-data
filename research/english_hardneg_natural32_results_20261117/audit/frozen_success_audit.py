#!/usr/bin/env python3
"""Independent stdlib audit of saved natural32 evidence. No imported worker.

Only ZIP/JSON/FLAC header/PCM/WAV/trace bytes are read. There is no subprocess,
network access, codec execution, compilation, inference or model construction.
The original SDK event policy is recomputed from saved float32 probabilities;
UNKNOWN review candidates never become labelled false accepts or negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import wave
import zipfile

RESEARCH = 'research/english_hardneg_natural32_20261117'
HEAD_SHA = 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'
LIB_SHA = '040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a'
FALSE_FLAGS = ('human_gold', 'trusted_negative', 'training_admitted', 'formal_training_admitted',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
THRESHOLD = struct.unpack('<f', struct.pack('<I', 0x3f4ccccd))[0]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def canonical(value, pretty=False):
    options = {'indent': 2} if pretty else {'separators': (',', ':')}
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, **options) + '\n').encode()


def load(raw):
    def unique(items):
        obj = {}
        for key, value in items:
            check(key not in obj, 'duplicate JSON key: ' + key)
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def false_flags(obj, where):
    check(all(obj.get(k) is False for k in FALSE_FLAGS), 'admission flag changed: ' + where)


def unknown(obj, where):
    false_flags(obj, where)
    check(obj.get('review') == 'UNREVIEWED' and obj.get('label_status') == 'UNKNOWN_REVIEW_REQUIRED',
          'unknown review status changed: ' + where)


class Artifact:
    def __init__(self, path):
        self.raw = path.read_bytes()
        self.archive = zipfile.ZipFile(io.BytesIO(self.raw))
        infos = self.archive.infolist()
        check(0 < len(infos) <= 3000, 'artifact member bound')
        names = [i.filename for i in infos if not i.is_dir()]
        check(len(names) == len(set(names)), 'duplicate artifact member')
        check(sum(i.file_size for i in infos) <= 256 * 1024**2, 'expanded artifact size bound')
        self.files = {}
        for info in infos:
            name = info.filename
            check(not PurePosixPath(name).is_absolute() and '\\' not in name and '\x00' not in name and
                  all(p not in ('.', '..', '') for p in name.rstrip('/').split('/')), 'unsafe artifact name')
            check(not info.flag_bits & 1 and not stat.S_ISLNK(info.external_attr >> 16), 'encrypted or symlink member')
            if info.is_dir():
                continue
            check(info.file_size <= 64 * 1024**2, 'oversized artifact member')
            self.files[name] = self.archive.read(info)
        check('MANIFEST.json' in self.files, 'root manifest absent')
        self.manifest = self.json('MANIFEST.json')
        entries = self.manifest['files']
        check(type(entries) is dict and set(self.files) == set(entries) | {'MANIFEST.json'}, 'manifest exact closed file set')
        for name, pin in entries.items():
            check(identity(self.files[name]) == pin, 'manifest byte mismatch: ' + name)

    def read(self, name):
        check(name in self.files, 'missing artifact member: ' + name)
        return self.files[name]

    def json(self, name):
        return load(self.read(name))

    def exact_directory(self, prefix, names):
        check({p[len(prefix):] for p in self.files if p.startswith(prefix)} == set(names),
              'unexpected/missing directory members: ' + prefix)


def parse_trace(raw, samples):
    """Parse independently from the binary record grammar, preserving float32."""
    expected_n = (samples - 16000) // 1280 + 1
    expected_ready = expected_n - 15
    check(samples >= 35200 and len(raw) >= 40 and raw[:8] == b'EKWSTRC1', 'trace header/magic')
    check(struct.unpack_from('<6IQ', raw, 8) == (1, 1, 16000, 320, 96, 3, samples), 'trace header geometry')
    pos = 40
    def take(kind, length):
        nonlocal pos
        check(pos + 5 + length <= len(raw), 'trace truncated')
        tag, declared = struct.unpack_from('<BI', raw, pos)
        check((tag, declared) == (kind, length), 'trace order/length')
        pos += 5
        payload = raw[pos:pos + length]
        pos += length
        return payload
    scores, events = [], []
    for index in range(expected_n):
        payload = take(1, 400)
        ordinal, sample = struct.unpack_from('<QQ', payload)
        values = struct.unpack_from('<96f', payload, 16)
        check(ordinal == index and sample == 16000 + 1280 * index and all(map(math.isfinite, values)), 'embedding ordinal/clock/finite')
        payload = take(2, 48)
        ordinal, ready, keyword, when = struct.unpack_from('<QIIQ', payload)
        values = struct.unpack_from('<6f', payload, 24)
        check(ordinal == index and ready == int(index >= 15) and when == sample and keyword in (0, 1, 2), 'score state/clock')
        check(all(map(math.isfinite, values)) and all(0 <= x <= 1 for x in values[3:]), 'score finite/probability')
        if not ready:
            check(keyword == 0 and all(x == 0 for x in values), 'nonready score is not zero')
        else:
            check(abs(sum(values[3:]) - 1) <= 1e-5, 'probability normalization')
        scores.append({'sample': when, 'ready': bool(ready), 'keyword': keyword,
                       'logits': list(values[:3]), 'probabilities': list(values[3:])})
        if keyword:
            ordinal2, embed_ordinal, which, event_clock, probability = struct.unpack('<QQIQf', take(3, 32))
            check((ordinal2, embed_ordinal, which, event_clock) == (len(events), index, keyword, sample), 'event identity/clock')
            check(struct.pack('<f', probability) == struct.pack('<f', values[3 + keyword]), 'event confidence bits')
            events.append([keyword, sample])
    end = list(struct.unpack('<6Q', take(255, 48)))
    check(end == [samples, expected_n, expected_n, expected_ready, len(events), (samples + 319) // 320], 'END counters')
    check(pos == len(raw), 'trace trailing bytes')
    # Independent fixed policy: maximum tie chooses the first class; cooldown
    # discards confirmations. It never filters a different decoder's events.
    recent, run, eligible_at = None, 0, 0
    derived = []
    for row in scores[15:]:
        probs = row['probabilities']
        best = max(range(3), key=lambda k: (probs[k], -k))
        emission = 0
        if row['sample'] < eligible_at or best == 0 or probs[best] < THRESHOLD:
            recent, run = None, 0
        else:
            run = run + 1 if recent == best else 1
            recent = best
            if run == 3:
                emission = best
                derived.append([best, row['sample']])
                eligible_at = row['sample'] + 16640
                recent, run = None, 0
        check(row['keyword'] == emission, 'saved score keyword disagrees with fixed float32 event policy')
    check(derived == events, 'independent saved event replay mismatch')
    return {'scores': scores, 'events': events, 'end': end, 'trace': identity(raw),
            'embedding_rows': expected_n, 'ready_rows': expected_ready}


def check_metadata(meta, parsed, frames):
    n, r, events = parsed['embedding_rows'], parsed['ready_rows'], parsed['events']
    expected = dict(schema='english-stream-numeric-trace-v1', format_magic='EKWSTRC1', format_version=1, abi=1,
                    sample_rate_hz=16000, encoding='s16le_mono', chunk_samples=320,
                    trace_bytes=parsed['trace']['bytes'], trace_records=2*n+len(events)+1,
                    trace_completed=True, end_record_written=True, input_samples=frames, input_bytes=2*frames,
                    embedding_records=n, real_head_calls=n, score_records=n, ready_outputs=r, event_records=len(events),
                    feed_calls=(frames+319)//320, input_repetitions=1, resets=0, eof_padding_samples=0,
                    score_observer_enabled=True, original_head_link_wrap=True, performance_measured=False)
    check(all(type(meta.get(k)) is type(v) and meta.get(k) == v for k, v in expected.items()), 'native stdout metadata mismatch')


def keyword_detail(rows, keyword):
    peak = sorted(rows, key=lambda r: (-r['probabilities'][keyword], r['sample']))[0] if rows else None
    return {'keyword_id': keyword, 'keyword_name': {1: 'Hey Vosen', 2: 'Hey Mero'}[keyword],
            'peak_probability': peak['probabilities'][keyword] if peak else None,
            'peak_sample': peak['sample'] if peak else None,
            'proposal_threshold_samples': [r['sample'] for r in rows if r['probabilities'][keyword] >= .5],
            'event_samples': [r['sample'] for r in rows if r['keyword'] == keyword]}


def derive_windows(roster_rows, ready_by_id, pcm_by_id, config_sha):
    """Independently construct windows and both caps from discrete saved scores."""
    windows = []
    def priority(c):
        return (not bool(c['event_count']), -c['maximum_keyword_probability'], c['recording_id'],
                c['start_sample'], c['end_sample_exclusive'])
    for source in roster_rows:
        rid, frames = source['recording_id'], source['frames']
        scores = ready_by_id[rid]
        intervals = []
        for row in scores:
            ids = [k for k in (1, 2) if row['probabilities'][k] >= .5]
            if not ids and row['keyword'] == 0:
                continue
            point = row['sample']
            seed = {'sample': point, 'proposal_threshold_keywords': ids, 'emitted_keyword': row['keyword']}
            left, right = max(0, point - 38400), min(frames, point + 16000)
            if intervals and left < intervals[-1]['end_sample_exclusive']:
                intervals[-1]['end_sample_exclusive'] = max(right, intervals[-1]['end_sample_exclusive'])
                intervals[-1]['seeds'].append(seed)
            else:
                intervals.append({'start_sample': left, 'end_sample_exclusive': right, 'seeds': [seed]})
        source_windows = []
        for interval in intervals:
            left, right = interval['start_sample'], interval['end_sample_exclusive']
            context = [row for row in scores if left < row['sample'] <= right]
            details = {str(k): keyword_detail(context, k) for k in (1, 2)}
            key = dict(source_pcm_sha256=source['pcm_sha256'], model_sha256=HEAD_SHA, config_sha256=config_sha,
                       recording_id=rid, start_sample=left, end_sample_exclusive=right)
            candidate = dict(interval, recording_id=rid, candidate_id='hardneg-review-'+sha(canonical(key, True))[:24],
                             window_samples=right-left, window_pcm_sha256=sha(pcm_by_id[rid][2*left:2*right]),
                             by_keyword=details, maximum_keyword_probability=max(v['peak_probability'] for v in details.values()),
                             event_count=sum(len(v['event_samples']) for v in details.values()),
                             triggering_keywords=sorted({k for s in interval['seeds'] for k in s['proposal_threshold_keywords']+
                                                         ([s['emitted_keyword']] if s['emitted_keyword'] else [])}),
                             pre_context_clipped_at_source_start=any(s['sample'] < 38400 for s in interval['seeds']),
                             post_context_clipped_at_source_end=any(s['sample']+16000 > frames for s in interval['seeds']))
            source_windows.append(candidate)
        for rank, candidate in enumerate(sorted(source_windows, key=priority), 1):
            candidate['source_priority_rank'] = rank
            candidate['selection'] = 'ELIGIBLE_FOR_GLOBAL_CAP' if rank <= 2 else 'EXCLUDED_PER_SOURCE_CAP'
            windows.append(candidate)
    eligible = sorted((c for c in windows if c['source_priority_rank'] <= 2), key=priority)
    for rank, candidate in enumerate(eligible, 1):
        candidate['global_priority_rank'] = rank
        candidate['selection'] = 'SELECTED_UNREVIEWED' if rank <= 32 else 'EXCLUDED_GLOBAL_CAP'
    selected = [c for c in eligible if c['selection'] == 'SELECTED_UNREVIEWED']
    excluded = sorted((c for c in windows if c['selection'].startswith('EXCLUDED_')), key=priority)
    return windows, selected, excluded


def verify_ledger(a, roster, parsed_by_id, pcm_by_id, config_sha, result):
    ledger = a.json('candidates/LEDGER.json')
    false_flags(ledger, 'ledger')
    check(ledger['review'] == 'UNREVIEWED' and ledger['training_tsv_emitted'] is False and
          ledger['verified_negative_seconds'] == 0 and ledger['independent_far'] == ledger['independent_frr'] == 'NOT_MEASURED', 'ledger became labelled quality evidence')
    check(ledger['new_model_forwards'] == ledger['new_tts_asr_phone_calls'] == ledger['optimizer_updates'] == 0, 'ledger computation count')
    ready = {rid: p['scores'][15:] for rid, p in parsed_by_id.items()}
    roster_sha = sha(canonical(roster, True))
    check(ledger['model_sha256'] == HEAD_SHA and ledger['config_sha256'] == config_sha and ledger['roster_sha256'] == roster_sha, 'ledger model/config/roster')
    evidence = ledger['source_evidence']
    check(ledger['source_evidence_sha256'] == sha(canonical(evidence, True)), 'ledger source evidence bytes')
    check(evidence['trace_sha256_by_id'] == {rid: p['trace']['sha256'] for rid, p in parsed_by_id.items()}, 'ledger trace sources')
    for key in ('run_id', 'head_sha', 'run_attempt', 'main_before', 'repository'):
        check(evidence[key] == result[key], 'ledger run binding')
    check(evidence['model_processes'] == evidence['validated_trace_count'] == 32 and evidence['retry_count'] == 0, 'ledger actual invocation counts')
    check(evidence['native_consumer'] == identity(a.read('bin/trace-native')) and
          evidence['installed_library'] == {'bytes':2441224,'sha256':LIB_SHA}, 'ledger actual executable/library')
    windows, selected, excluded = derive_windows(roster['rows'], ready, pcm_by_id, config_sha)
    actual_all = ledger['candidates'] + ledger['excluded_candidates']
    actual = {c['candidate_id']: c for c in actual_all}
    check(len(actual) == len(actual_all) == len(windows) and set(actual) == {c['candidate_id'] for c in windows}, 'exact independent candidate set')
    check([c['candidate_id'] for c in ledger['candidates']] == [c['candidate_id'] for c in selected] and
          [c['candidate_id'] for c in ledger['excluded_candidates']] == [c['candidate_id'] for c in excluded], 'independent candidate priority/caps/order')
    source_rows = {r['recording_id']: r for r in roster['rows']}
    normalized = {rid: [{k: row[k] for k in ('sample', 'probabilities', 'keyword')} for row in rows] for rid, rows in ready.items()}
    def common(entry):
        rid = entry['recording_id']
        check(rid in source_rows, 'ledger unknown recording')
        source = source_rows[rid]
        unknown(entry, rid)
        for key in ('source_member', 'speaker_id', 'source_family_id'):
            check(entry[key] == source[key], 'ledger source lineage: '+key)
        check(entry['source_role'] == 'train' and entry['source_utterance_continuity'] == 'natural_individual' and
              entry['source_pcm_sha256'] == source['pcm_sha256'] and entry['source_samples'] == source['frames'], 'ledger source PCM/role')
        check(entry['model_sha256'] == HEAD_SHA and entry['config_sha256'] == config_sha and entry['roster_sha256'] == roster_sha and
              entry['original_c_trace_sha256'] == parsed_by_id[rid]['trace']['sha256'], 'ledger row evidence binding')
        check(entry['input_ready_rows_sha256'] == sha(canonical(ready[rid], True)) and
              entry['normalized_ready_rows_sha256'] == sha(canonical(normalized[rid], True)), 'ledger ready score hashes')
    for expected in windows:
        actual_c = actual[expected['candidate_id']]
        common(actual_c)
        check(all(actual_c.get(k) == v for k, v in expected.items()), 'candidate window/score/rank mismatch')
        check(actual_c['training_tsv_emitted'] is False and actual_c['word_end_annotation'] is None, 'candidate training/timing promotion')
        check(actual_c['start_seconds'] == expected['start_sample']/16000 and
              actual_c['end_seconds_exclusive'] == expected['end_sample_exclusive']/16000, 'candidate seconds coordinate')
    summaries = {s['recording_id']: s for s in ledger['source_summaries']}
    check(len(summaries) == len(ledger['source_summaries']) == 32 and set(summaries) == set(ready), 'complete source summaries')
    for rid, summary in summaries.items():
        common(summary)
        check(summary['by_keyword'] == {str(k): keyword_detail(ready[rid], k) for k in (1, 2)}, 'source keyword peaks/events')
        n_windows = sum(c['recording_id'] == rid for c in windows)
        check(summary['actual_event_count'] == len(parsed_by_id[rid]['events']) and summary['ready_rows'] == len(ready[rid]) and
              summary['source_seconds'] == source_rows[rid]['frames']/16000 and summary['merged_candidate_windows'] == n_windows and
              summary['windows_after_per_source_cap'] == min(n_windows, 2) and
              summary['selected_candidate_windows'] == sum(c['recording_id'] == rid for c in selected), 'source summary counts')
    counts = dict(sources=32, source_samples=3270080, source_seconds=204.38, ready_rows=1694,
                  actual_events=sum(len(p['events']) for p in parsed_by_id.values()), merged_candidates=len(windows),
                  selected_candidates=len(selected), excluded_by_per_source_cap=sum(c['selection']=='EXCLUDED_PER_SOURCE_CAP' for c in windows),
                  excluded_by_global_cap=sum(c['selection']=='EXCLUDED_GLOBAL_CAP' for c in windows))
    check(ledger['counts'] == counts and len(selected) <= 32, 'ledger total counts')
    check(ledger['status'] == ('UNREVIEWED_CANDIDATES_SAVED' if selected else 'NO_CANDIDATE_IN_SAVED_OBSERVATION'), 'empty queue semantics')
    files = a.json('candidates/CLIP_FILES.json')
    check(files['count'] == len(selected) and files['training_tsv_emitted'] is False, 'candidate file count/training')
    file_rows = {f['candidate_id']: f for f in files['clips']}
    check(len(file_rows) == len(files['clips']) and set(file_rows) == {c['candidate_id'] for c in selected}, 'candidate file identities')
    a.exact_directory('candidates/pcm/', [c['candidate_id']+'.pcm' for c in selected])
    a.exact_directory('candidates/wav/', [c['candidate_id']+'.wav' for c in selected])
    for c in selected:
        cid = c['candidate_id']
        raw = a.read('candidates/pcm/'+cid+'.pcm')
        expected = pcm_by_id[c['recording_id']][2*c['start_sample']:2*c['end_sample_exclusive']]
        check(raw == expected, 'candidate PCM is not exact source slice')
        wav = a.read('candidates/wav/'+cid+'.wav')
        with wave.open(io.BytesIO(wav), 'rb') as stream:
            check((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes(), stream.getcomptype()) ==
                  (1, 2, 16000, c['window_samples'], 'NONE') and stream.readframes(c['window_samples']+1) == raw, 'candidate WAV PCM/geometry')
        record = file_rows[cid]
        false_flags(record, cid)
        check(record['review'] == 'UNREVIEWED' and record['pcm'] == identity(raw) and record['wav'] == identity(wav), 'candidate file receipt')
    check(result['actual_event_count'] == counts['actual_events'] and result['candidate_count'] == len(selected) and
          result['ledger_identity'] == identity(a.read('candidates/LEDGER.json')), 'worker aggregate vs independent ledger')
    return counts


def audit(path, bindings, api_receipt=None):
    a = Artifact(path)
    check(a.manifest['supervised_success'] is True and a.manifest['downloads_excluded'] is True and
          a.manifest['authoritative_completion_receipt'] == 'evidence/SUPERVISOR_EXIT.json', 'artifact completion authority')
    false_flags(a.manifest, 'manifest')
    for source, pin in {**bindings['source_pins'], **bindings['files']}.items():
        check(identity(a.read('source/'+source)) == pin, 'frozen source mismatch: '+source)
    plan = a.json('source/automation/english_hardneg_natural32_plan.json')
    check(plan['worker_sha256'] == bindings['worker_sha256'] and plan['main_before'] == bindings['main_before'] and
          plan['source_pins'] == bindings['source_pins'], 'frozen plan/source binding')
    config_raw = a.read('source/CONFIG.json')
    config = load(config_raw)
    check(config == bindings['configuration'] == plan['configuration'], 'frozen configuration')
    config_sha = sha(canonical(config))
    check(config_sha == plan['configuration_sha256'], 'configuration SHA')
    check(identity(a.read('models/coverage-head.bin')) == {'bytes':197908,'sha256':HEAD_SHA}, 'actual model bytes')
    check(identity(a.read('installed-base/sdk-native/lib/libekws_research_stream.a')) == {'bytes':2441224,'sha256':LIB_SHA}, 'installed library bytes')
    consumer = a.json('evidence/NATIVE_CONSUMER.json')
    check(consumer['identity'] == identity(a.read('bin/trace-native')) and consumer['installed_library'] ==
          {'bytes':2441224,'sha256':LIB_SHA} and consumer['consumer_builds'] == 1, 'native consumer build receipt')
    start, supervisor, result = (a.json(p) for p in ['STARTED.json', 'evidence/SUPERVISOR_EXIT.json', 'evidence/RESULT.json'])
    check(supervisor['passed'] is True and supervisor['returncode'] == 0 and supervisor['timed_out'] is False and supervisor['error'] is None,
          'supervisor did not succeed')
    check(supervisor['cleanup'].get('descendants_found') is False and not supervisor.get('partial_receipt_errors') and
          not supervisor['cleanup'].get('error'), 'supervisor cleanup/partial evidence')
    check(result['passed'] is True and result['status'] == 'COMPLETED_NATURAL32_RESEARCH_SCAN', 'worker did not succeed')
    for obj in (supervisor, result, start['binding']):
        check(obj['repository'] == 'jiying2007/kws-data' and obj['main_before'] == bindings['main_before'] and obj['run_attempt'] == 1,
              'run identity/budget binding')
        check(obj['head_sha'] == result['head_sha'] and obj['run_id'] == result['run_id'] and obj.get('public_repository') is True,
              'run/head/public mismatch')
    check(start['worker_sha256'] == bindings['worker_sha256'] and start['monotonic_deadline']-start['monotonic_started'] == 1200 and
          0 < supervisor['wall_seconds'] <= 1210 and supervisor['retry_count'] == result['retry_count'] == 0, 'deadline/no-retry evidence')
    false_flags(result, 'RESULT')
    check(result['verified_negative_seconds'] == result['verified_negative_count'] == result['new_independent_speaker_count'] == 0 and
          result['independent_far'] == result['independent_frr'] == 'NOT_MEASURED' and result['natural_continuous_far_denominator'] is False,
          'UNKNOWN source falsely promoted to FAR/FRR')
    check(result['quality_improvement_claim'] is False and result['word_end_latency'] == 'NOT_MEASURED', 'unsupported quality/timing claim')
    if api_receipt is not None:
        receipt = load(api_receipt.read_bytes())
        check(receipt['sha256'] == sha(a.raw) and receipt['bytes'] == len(a.raw) and
              str(receipt['run_id']) == str(result['run_id']) and receipt['head_sha'] == result['head_sha'] and
              receipt['run_attempt'] == 1 and receipt['status'] == 'completed' and receipt['conclusion'] == 'success', 'external API receipt mismatch')
    roster_raw = a.read('source/'+RESEARCH+'/natural32-roster.json')
    roster = load(roster_raw)
    check(roster['data_repository_commit'] == bindings['main_before'] and roster['coverage_head_binary_sha256'] == HEAD_SHA, 'roster source/model')
    for name, expected in roster['metadata_inputs'].items():
        check(sha(a.read('source-metadata/'+name)) == expected, 'original training metadata hash: '+name)
    rows = roster['rows']
    check(len(rows) == 32 and len({r['recording_id'] for r in rows}) == 32 and
          len({r['speaker_id'] for r in rows}) == 24 and sum(r['frames'] for r in rows) == 3270080, 'frozen roster cardinality')
    ids = [r['recording_id'] for r in rows]
    transcript_names = {PurePosixPath(r['source_member']).parts[-3]+'-'+PurePosixPath(r['source_member']).parts[-2]+'.trans.txt'
                        for r in rows}
    a.exact_directory('selectedinputs/transcript-sources/', transcript_names)
    for prefix, suffix in [('selectedinputs/pcm/', '.pcm'), ('selectedinputs/flac/', '.flac'),
                           ('selectedinputs/transcripts/', '.txt'), ('traces/', '.trace'), ('scores/', '.json')]:
        a.exact_directory(prefix, [rid+suffix for rid in ids])
    parsed, pcm = {}, {}
    for index, row in enumerate(rows):
        rid, frames = row['recording_id'], row['frames']
        false_flags(row, rid)
        check(row['index'] == index and row['source_role'] == row['old_bg48_role'] == row['current_coverage_role'] == 'train' and
              row['current_coverage_original_source_consumed'] is False and row['historically_exposed_to_other_models'] is True and
              row['label_status'] == 'UNKNOWN_REVIEW_REQUIRED' and row['expected_events'] is None, 'roster role/exposure/UNKNOWN')
        check((row['sample_rate'],row['channels'],row['sample_width']) == (16000,1,2), 'roster audio geometry')
        pcm[rid] = a.read('selectedinputs/pcm/'+rid+'.pcm')
        check(len(pcm[rid]) == 2*frames and sha(pcm[rid]) == row['pcm_sha256'], 'original PCM identity')
        flac = a.read('selectedinputs/flac/'+rid+'.flac')
        check(sha(flac) == row['flac_sha256'] and len(flac) >= 42 and flac[:4] == b'fLaC' and flac[4]&127 == 0 and
              int.from_bytes(flac[5:8],'big') == 34, 'original FLAC identity/header')
        info = int.from_bytes(flac[18:26], 'big')
        check((info>>44,((info>>41)&7)+1,((info>>36)&31)+1,info&((1<<36)-1)) == (16000,1,16,frames), 'FLAC STREAMINFO geometry')
        transcript = a.read('selectedinputs/transcripts/'+rid+'.txt')
        check(sha(transcript) == row['transcript_sha256'], 'original transcript SHA')
        member = PurePosixPath(row['source_member'])
        transcript_source = a.read('selectedinputs/transcript-sources/'+member.parts[-3]+'-'+member.parts[-2]+'.trans.txt')
        pairs = [line.split(maxsplit=1) for line in transcript_source.decode().splitlines()]
        check(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'original transcript ID uniqueness')
        source_text = dict(pairs)
        check(source_text[member.stem].encode() == transcript, 'selected transcript not exact original utterance')
        p = parse_trace(a.read('traces/'+rid+'.trace'), frames)
        check(p['embedding_rows'] == row['expected_embedding_rows'] and p['ready_rows'] == row['expected_ready_rows'], 'roster/trace counts')
        check_metadata(a.json('evidence/scan-'+str(index)+'.stdout'), p, frames)
        score_doc = a.json('scores/'+rid+'.json')
        check(score_doc['recording_id'] == rid and score_doc['source'] == row and score_doc['scores'] == p['scores'] and
              score_doc['events'] == p['events'] and score_doc['trace'] == p['trace'] and score_doc['configuration_sha256'] == config_sha,
              'saved JSON scores diverge from raw trace')
        check(score_doc['model'] == {'bytes':197908,'sha256':HEAD_SHA}, 'saved score model identity')
        verified = a.json('evidence/scan-'+str(index)+'.verified.json')
        false_flags(verified, 'verified trace')
        check(verified['recording_id'] == rid and verified['input_pcm_sha256'] == row['pcm_sha256'] and verified['events'] == p['events'] and
              verified['trace'] == p['trace'] and verified['saved_score_event_replay_exact'] is True and
              verified['numeric_reference_comparison'] == 'NOT_RUN' and verified['label_status'] == 'UNKNOWN_REVIEW_REQUIRED' and
              verified['frames'] == frames and verified['embedding_rows'] == p['embedding_rows'] and verified['ready_rows'] == p['ready_rows'],
              'per-trace verification claim')
        parsed[rid] = p
    check(sum(p['embedding_rows'] for p in parsed.values()) == 2174 and sum(p['ready_rows'] for p in parsed.values()) == 1694, 'aggregate trace counts')
    kinds = Counter()
    for name in sorted(a.files):
        if not name.startswith('evidence/') or not name.endswith('.started.json'):
            continue
        started = a.json(name)
        ended = a.json(name[:-len('.started.json')]+'.exit.json')
        check(ended.get('error') is None and ended.get('retry_count') == 0, 'failed/interrupted step')
        if 'argv' in started:
            check(started['argv'] == ended['argv'] and started['attempt'] == ended['attempt'] == 1 and
                  ended['returncode'] == 0 and ended['timed_out'] is False, 'command start/exit pairing')
            if started['kind'] is not None:
                kinds[started['kind']] += 1
        if started.get('kind') == 'model':
            i = int(name.split('/')[-1].removeprefix('scan-').removesuffix('.started.json'))
            check(0 <= i < 32, 'unexpected model invocation')
            argv = started['argv']
            check(len(argv) == 4 and argv[0].endswith('/bin/trace-native') and argv[1].endswith('/models/coverage-head.bin') and
                  argv[2].endswith('/selectedinputs/pcm/'+ids[i]+'.pcm') and argv[3].endswith('/traces/'+ids[i]+'.trace'), 'model command/source pairing')
    check(kinds == {'build':1, 'decode':32, 'model':32}, 'once invocation budget')
    check(result['consumer_build_attempts'] == 1 and result['decoder_process_attempts'] == result['model_process_attempts'] ==
          result['completed_validated_traces'] == 32 and result['embedding_rows'] == 2174 and result['ready_rows'] == 1694, 'RESULT invocation counters')
    counts = verify_ledger(a, roster, parsed, pcm, config_sha, result)
    return {'passed': True, 'artifact': identity(a.raw), 'manifest_files_verified': len(a.files)-1,
            'source_bindings_sha256': sha(canonical(bindings, True)), 'worker_sha256': bindings['worker_sha256'],
            'run_id': result['run_id'], 'head_sha': result['head_sha'], 'run_attempt': 1,
            'api_identity_verification': 'MATCHED_PROVIDED_RECEIPT' if api_receipt else 'NOT_PERFORMED_BY_OFFLINE_AUDITOR',
            'originals': 32, 'source_samples': 3270080, 'source_seconds': 204.38, 'embedding_rows':2174, 'ready_rows':1694,
            'independent_fixed_event_replay_exact': True, 'independent_candidate_windows_exact': True,
            'counts':counts, 'process_start_exit_pairs':dict(kinds), 'supervisor_success_without_descendants':True,
            'per_source':[{'recording_id':rid,'events':p['events'],'END':p['end'],'trace':p['trace']} for rid,p in parsed.items()],
            'models_executed':0,'audio_decodes_executed':0,'compiler_calls':0,'network_calls':0,
            'verified_negative_seconds':0,'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED',
            'candidate_status':'UNKNOWN_REVIEW_REQUIRED','numeric_reference_comparison':'NOT_RUN',
            'limits':['This replays saved event policy; it does not recompute embeddings/logits or decode FLAC.',
                      'Archive checks bind retained bytes and invocations; candidates remain unreviewed and are not false-accept labels.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-receipt', type=Path)
    args = parser.parse_args()
    check(not args.output.exists(), 'output must be create-only')
    bindings = load(Path(__file__).with_name('FROZEN_INPUT_BINDINGS.json').read_bytes())
    try:
        report = audit(args.artifact, bindings, args.api_receipt)
    except Exception as exc:
        report = {'passed':False,'error':type(exc).__name__+':'+str(exc),'models_executed':0,'audio_decodes_executed':0,
                  'compiler_calls':0,'network_calls':0,'candidate_status':'UNKNOWN_REVIEW_REQUIRED',
                  'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED'}
    report['schema'] = 'independent-natural32-saved-artifact-audit-v1'
    with args.output.open('xb') as out:
        out.write(canonical(report, True))
    print(json.dumps({'passed':report['passed'],'output':str(args.output),'error':report.get('error')},sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
