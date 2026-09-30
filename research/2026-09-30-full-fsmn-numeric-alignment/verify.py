"""Stdlib-only archive integrity and retained-scalar consistency, never inference.

The original NPZ arrays are deliberately excluded. Elementwise error summaries
are aggregated from the retained original scalar reports, not independently
recalculated from tensor values. Events and timing sums ARE recomputed here.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zlib

BASE = Path(__file__).absolute().parent
CHUNK_BYTES = 48 * 1024
MAX_LOGICAL_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = 512 * 1024
GENERATED = {'README.md', 'verify.py', 'test_verify.py', 'logical-files.json',
             'summary.json', 'LICENSE.wekws', 'NOTICE.md'}
# Inventory is fixed independently of either editable manifest.
ORIGINALS = {
    'RESULTS.md', 'full-oracle-preflight.json', 'full-oracle-result.json',
    'pcm-compression-manifest.json', 'pcm-native-exit-code.txt',
    'pcm-native-preflight.json', 'pcm-native-result.json', 'pcm-native-run.log',
    'pcm-reference-exit-code.txt', 'pcm-reference-preflight.json',
    'pcm-reference-receipt.json', 'pcm-reference-run.log',
    'resource-profile/RESULTS.md', 'resource-profile/compression.json',
    'resource-profile/exit-code.txt', 'resource-profile/preflight.json',
    'resource-profile/result.json', 'resource-profile/run.log',
    *('prior-stages/' + name for name in (
        'fsmn-double-diagnostic.c', 'fsmn-v1.c', 'fsmn-v2.c',
        'guard-cmvn-result.json', 'local-affine-prereg.json',
        'local-affine-result.json', 'network-result-double-diagnostic.json',
        'network-result-v1.json', 'network-result-v2.json',
        'reference-metadata-supplement.json', 'reference-receipt.json',
        'splice-build-receipt.json', 'splice-result.json',
        'torch-self-parity.json', 'compare-network-v1.py',
        'compare-network-v2.py', 'compare-double-executed.py',
        'fsmn-initial-executed.h', 'fsmn-before-full-oracle.c')),
}
COMPRESSED = {
    'full-oracle-result.json', 'pcm-native-result.json',
    'pcm-reference-receipt.json', 'resource-profile/result.json',
    'prior-stages/network-result-v1.json',
    'prior-stages/network-result-v2.json',
    'prior-stages/network-result-double-diagnostic.json',
}
PINNED = {
    'full-oracle-result.json': 'ddd7eb6c92ba7f04df4859b1b9ae3627f1f22bcae9acc6231e1d59f12eaee0bc',
    'pcm-native-result.json': '0a4a927610d603b5be3703a0cef880e370dfb61ee367a7457dc3cc9369e9daec',
    'pcm-reference-receipt.json': '1e9c5acd3dc2268ca79189043780ddf1f8674b95bcad6322faa0c03fa6f3ec6d',
    'resource-profile/result.json': 'ba58cc18037d56884731e0646f46d013e2dbd08fb5de288d249032ec4bc57629',
}
SOURCE_SNAPSHOT = {'repository': 'jiying2007/kws-pipeline', 'commit': 'fe501f51243b40015fa4c17c267e66e9274c20d8', 'path': 'research/donor_fsmn', 'url': 'https://github.com/jiying2007/kws-pipeline/tree/fe501f51243b40015fa4c17c267e66e9274c20d8/research/donor_fsmn', 'commit_role': 'source-snapshot-not-merge-sha'}
CONDITIONS = ('raw', 'tail500ms', 'head500ms_tail500ms')
COMPONENTS = {'model_including_cmvn', 'frontend_splice_inclusive',
              'callback_copy', 'torch_softmax', 'python_decoder',
              'standalone_cmvn_micro'}
METRICS = {'fbank', 'splice', 'cmvn', 'cache', 'logits', 'probabilities',
           *('stage' + str(i) for i in range(21))}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonnegative_int(value):
    return type(value) is int and value >= 0


def checked_path(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'unsafe path')
    parts = name.split('/')
    require(not name.startswith('/') and all(p not in ('', '.', '..') for p in parts),
            'unsafe path')
    require(str(PurePosixPath(name)) == name, 'noncanonical path')
    return name


def safe_read(root, name):
    checked_path(name)
    # Do not resolve first: that would erase evidence of a symlinked ancestor.
    path = root / name
    for p in (path, *path.parents):
        require(not p.is_symlink(), 'symlink member or parent')
    mode = path.stat().st_mode
    require(stat.S_ISREG(mode), 'non-regular member')
    require(path.stat().st_size <= CHUNK_BYTES, 'physical member size cap')
    return path.read_bytes()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    result = json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
                        parse_constant=bad_constant)
    # Also rejects exponent overflow (e.g. 1e999), not just NaN tokens.
    def check(value):
        if isinstance(value, float):
            require(math.isfinite(value), 'nonfinite JSON number')
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(result)
    return result


def identity(data, size, sha, message):
    require(type(size) is int and len(data) == size and
            isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) and
            digest(data) == sha, message)


def decompress_bounded(data, expanded_bytes):
    require(type(expanded_bytes) is int and 0 < expanded_bytes <= MAX_LOGICAL_BYTES,
            'expanded size budget')
    require(0 < len(data) <= MAX_COMPRESSED_BYTES, 'compressed size budget')
    # Single deterministic gzip stream: no timestamp, optional name or comments.
    require(data[:8] == b'\x1f\x8b\x08\x00\x00\x00\x00\x00' and data[8] == 2,
            'non-deterministic gzip header')
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(data, expanded_bytes + 1)
    require(len(raw) <= expanded_bytes and not decoder.unconsumed_tail,
            'expanded size budget')
    require(decoder.eof and not decoder.unused_data, 'truncated/concatenated/trailing gzip')
    require(len(raw) == expanded_bytes, 'expanded size mismatch')
    return raw


def read_archive(root=BASE):
    root = Path(root).absolute()
    for p in (root, *root.parents):
        require(not p.is_symlink(), 'symlink root or ancestor')
    inventory = strict_json(safe_read(root, 'logical-files.json'))
    require(inventory['schema_version'] == 1 and type(inventory['schema_version']) is int
            and inventory['chunk_max_bytes'] == CHUNK_BYTES, 'logical inventory schema')
    rows = inventory['files']
    require(len(rows) == len(ORIGINALS) and {r['path'] for r in rows} == ORIGINALS,
            'required logical inventory')
    physical = set(GENERATED)
    payloads = {}
    for row in rows:
        name = checked_path(row['path'])
        if name in COMPRESSED:
            require(row['encoding'] == 'gzip-chunks', 'required compressed encoding')
            n = row['gzip_bytes']
            require(type(n) is int and 0 < n <= MAX_COMPRESSED_BYTES, 'compressed size budget')
            parts = row['parts']
            require(len(parts) == (n + CHUNK_BYTES - 1) // CHUNK_BYTES, 'chunk count')
            chunks = []
            for i, piece in enumerate(parts):
                path = checked_path(piece['path'])
                require(path == name + '.gz.parts/' + f'{i:04d}.bin', 'chunk path/order')
                require(piece['bytes'] == min(CHUNK_BYTES, n - CHUNK_BYTES * i), 'chunk size')
                b = safe_read(root, path)
                identity(b, piece['bytes'], piece['sha256'], 'chunk identity')
                physical.add(path)
                chunks.append(b)
            packed = b''.join(chunks)
            identity(packed, n, row['gzip_sha256'], 'whole compressed identity')
            raw = decompress_bounded(packed, row['original_bytes'])
        else:
            require(row['encoding'] == 'utf-8' and 'parts' not in row, 'required UTF-8 encoding')
            physical.add(name)
            raw = safe_read(root, name)
        raw.decode('utf-8')
        identity(raw, row['original_bytes'], row['original_sha256'], 'original logical identity')
        if name in PINNED:
            require(digest(raw) == PINNED[name], 'approved original identity')
        payloads[name] = raw
    manifest = strict_json(safe_read(root, 'archive-manifest.json'))
    require(manifest['schema_version'] == 1 and type(manifest['schema_version']) is int
            and manifest['purpose'] == 'historical-research-not-data-catalog', 'archive schema')
    require(manifest['source_snapshot'] == SOURCE_SNAPSHOT, 'fixed source snapshot metadata')
    require(SOURCE_SNAPSHOT['url'].encode() in safe_read(root, 'README.md'), 'source snapshot README link')
    rows = manifest['files']
    require(len(rows) == len(physical) and {r['path'] for r in rows} == physical,
            'required physical inventory')
    actual = set()
    allowed_dirs = {str(p) for n in physical for p in PurePosixPath(n).parents if str(p) != '.'}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(directory) / name
            require(not p.is_symlink(), 'symlink directory')
            require(p.relative_to(root).as_posix() in allowed_dirs, 'unexpected directory')
        for name in files:
            p = Path(directory) / name
            require(not p.is_symlink(), 'symlink member')
            actual.add(p.relative_to(root).as_posix())
    require(actual == physical | {'archive-manifest.json'}, 'unexpected/missing physical inventory')
    for row in rows:
        data = safe_read(root, row['path'])
        identity(data, row['bytes'], row['sha256'], 'physical identity')
        if not row['path'].endswith('.bin'):
            data.decode('utf-8')
    require(digest(safe_read(root, 'LICENSE.wekws')) == 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
            'historical Apache license identity')
    notice = safe_read(root, 'NOTICE.md').decode('utf-8')
    require(all(text in notice for text in ('6a45aeb994dd81c0969ff877a5a7c46d60ed0c86',
                'Apache-2.0', 'Yueyue Nyy', 'Jing Du', 'not distributed')),
            'historical attribution notice')
    return payloads


def verify_history(payloads):
    load = lambda name: strict_json(payloads[name])
    receipt = load('prior-stages/reference-receipt.json')
    require(receipt['gate'] == {'atol': 1e-4, 'rtol': 1e-5}, 'old B gate unchanged')
    histories = {}
    for variant, source, comparator, count in (
        ('v1', 'fsmn-v1.c', 'compare-network-v1.py', 33),
        ('v2', 'fsmn-v2.c', 'compare-network-v2.py', 35),
        ('double-diagnostic', 'fsmn-double-diagnostic.c', 'compare-double-executed.py', 29),
    ):
        r = load('prior-stages/network-result-' + variant + '.json')
        for key, file in [('source_sha256', source), ('header_sha256', 'fsmn-initial-executed.h'),
                          ('comparison_sha256', comparator), ('reference_receipt_sha256', 'reference-receipt.json')]:
            require(r[key] == digest(payloads['prior-stages/' + file]), 'historical source/receipt binding')
        checks = r['checks']
        require(len(checks) == 264 and len({x['name'] for x in checks}) == 264, 'historical coverage')
        require(all(type(x['passed']) is bool and x['passed'] == (x['fail_elements'] == 0)
                    for x in checks), 'historical per-check outcome')
        failed = sum(not x['passed'] for x in checks)
        require(failed == count and r['passed'] is False, 'historical B failure')
        histories[variant] = {'checks': len(checks), 'failed_checks': failed}
    cmvn = load('prior-stages/guard-cmvn-result.json')
    require(cmvn['guard_checks_passed'] is True and cmvn['cmvn_passed'] is True and
            len(cmvn['cmvn_checks']) == 12 and all(x['passed'] is True and x['max_abs'] == 0
            for x in cmvn['cmvn_checks']), 'CMVN historical result')
    require(load('prior-stages/local-affine-prereg.json')['source_sha256'] ==
            digest(payloads['prior-stages/fsmn-before-full-oracle.c']), 'local affine historical source binding')
    local = load('prior-stages/local-affine-result.json')
    require(local['prereg_sha256'] == digest(payloads['prior-stages/local-affine-prereg.json'])
            and local['C_vs_frozen'] == {'max_abs': 3.814697265625e-05, 'B_failures': 0},
            'local affine binding/outcome')
    torch = load('prior-stages/torch-self-parity.json')
    require(len(torch['checks']) == 63 and sum(not x['passed'] for x in torch['checks']) == 3
            and torch['passed'] is False, 'Torch whole/split failure')
    splice = load('prior-stages/splice-result.json')
    require((splice['cases'], splice['calls'], splice['tests']) == (29, 67, 8)
            and splice['passed'] is True, 'splice historical outcome')
    oracle = load('full-oracle-result.json')
    require(oracle['preflight_sha256'] == digest(payloads['full-oracle-preflight.json']), 'oracle preflight binding')
    pre = load('full-oracle-preflight.json')
    require(pre['old_B'] == receipt['gate'], 'oracle old B gate unchanged')
    records, caches = oracle['records'], oracle['cache_checks']
    names = {case + '_' + route + '_' + stage for case in ('zero', 'impulse', 'random')
             for route in ('whole', 'split') for stage in ['cmvn'] + ['stage' + str(i) for i in range(21)]}
    require(len(records) == 132 and {x['name'] for x in records} == names and len(caches) == 12
            and len({x['name'] for x in caches}) == 12, 'oracle coverage')
    require(all(x['local_C_bound_failures'] == x['local_Torch_bound_failures'] ==
                x['envelope_exceedances'] == 0 for x in records) and
            all(x['source_copy_exact'] is True and x['envelope_exceedances'] == 0 for x in caches),
            'oracle local consistency')
    failed = sum(x['old_B_failures'] > 0 for x in records + caches)
    envelope = max(x['envelope_max'] for x in records)
    require(failed == oracle['old_B_actual_failing_records'] == 27 and
            oracle['old_B_still_failed'] is True and oracle['mathematical_consistency_passed'] is True and
            oracle['stage_C_executed'] is False, 'oracle does not overwrite B')
    require(envelope == 2308069467687359.5 and
            all(x['old_B_failures'] == 0 for x in records if x['name'].endswith('stage20')),
            'oracle envelope/final-logit outcome')
    return {'old_B_variants': histories, 'full_oracle_records': 132,
            'full_oracle_cache_records': 12, 'full_oracle_B_failing_records': failed,
            'full_oracle_max_envelope': envelope, 'envelope_establishes_fidelity': False}


def verify_pcm(payloads):
    load = lambda name: strict_json(payloads[name])
    native, ref = load('pcm-native-result.json'), load('pcm-reference-receipt.json')
    np, rp = load('pcm-native-preflight.json'), load('pcm-reference-preflight.json')
    require(native['preflight_sha256'] == digest(payloads['pcm-native-preflight.json']) and
            ref['preflight_sha256'] == digest(payloads['pcm-reference-preflight.json']) and
            np['reference_sha256'] == digest(payloads['pcm-reference-receipt.json']), 'PCM receipt bindings')
    require(np['gates'] == {
        'logits': {'atol': 1e-4, 'rtol': 1e-5},
        'probabilities': {'atol': 1e-5, 'rtol': 1e-5, 'row_sum_error': 1e-5},
        'event_fields_exact_including_score': True, 'score_tolerance_report_only': 1e-5}, 'PCM fixed gates')
    require(native['clips'] == ref['clips'] == 126 and len(native['recordings']) ==
            len(ref['recordings']) == 126, 'PCM coverage')
    require(ref['historical_logits_and_events_exact'] is True, 'reference historical claim')
    key = lambda r: (r['condition'], r['recording'])
    refs = {key(r): r for r in ref['recordings']}
    require(len(refs) == 126 and len({key(r) for r in native['recordings']}) == 126 and
            {key(r) for r in native['recordings']} == set(refs), 'PCM unique identities')
    source = {(condition, r['recording']): r for condition, rows in rp['conditions'].items() for r in rows}
    require(set(source) == set(refs), 'PCM source coverage')
    metrics = {name: {'elements': 0, 'fail_elements': 0, 'failing_calls': 0,
                      'failing_recordings': 0, 'max_abs': 0} for name in sorted(METRICS)}
    counts = {c: {'clips': 0, 'positive_clips': 0, 'target_hits': 0,
                  'confusable_clips': 0, 'confusable_triggered_clips': 0} for c in CONDITIONS}
    changed, max_score, row_sum = 0, 0, 0
    without_score = lambda e: {k: v for k, v in e.items() if k != 'score'}
    for r in native['recordings']:
        rr = refs[key(r)]
        s = source[key(r)]
        require(r['condition'] in CONDITIONS and r['kind'] in ('positive', 'confusable') and
                r['kind'] == rr['kind'] and r['keyword_id'] == rr['keyword_id'], 'PCM label identity')
        require(rr['source_sha256'] == s['file_sha256'] and rr['pcm_sha256'] == s['pcm_sha256'] and
                np['reference_arrays'][rr['arrays_file']] == rr['arrays_sha256'], 'PCM source/reference array bindings')
        for a in (r, rr):
            require(all(finite_number(e['score']) and 0 <= e['score'] <= 1 for e in a['events']), 'event score range')
        require([without_score(e) for e in r['events']] == [without_score(e) for e in rr['events']], 'non-score event equality')
        delta = max((abs(a['score'] - b['score']) for a, b in zip(r['events'], rr['events'])), default=0)
        exact = r['events'] == rr['events']
        require(r['events_exact'] is exact and r['events_non_score_exact'] is True and
                r['score_max_difference'] == delta and r['score_within_1e_minus_5'] is (delta <= 1e-5), 'event score summary')
        changed += not exact
        max_score = max(max_score, delta)
        c = counts[r['condition']]
        c['clips'] += 1
        if r['kind'] == 'positive':
            c['positive_clips'] += 1
            c['target_hits'] += any(e['keyword_id'] == r['keyword_id'] for e in r['events'])
        else:
            c['confusable_clips'] += 1
            c['confusable_triggered_clips'] += bool(r['events'])
        require(len(r['calls']) == len(rr['calls']), 'PCM call coverage')
        failing = set()
        for call, reference in zip(r['calls'], rr['calls']):
            require(all(call[a] == reference[b] for a, b in (
                ('index', 'index'), ('wave_samples', 'wave_remained'),
                ('feature_count', 'feature_remained'), ('offset', 'offset'),
                ('centers', 'centers'), ('available_audio_samples', 'available_audio_samples'))),
                'PCM discrete call state')
            require(without_score(call['result']) == without_score(reference['result']), 'PCM call decisions')
            m = call['metrics']
            require(type(call['has_acoustic_output']) is bool and call['has_acoustic_output'] == bool(call['centers']), 'PCM acoustic output')
            if not call['has_acoustic_output']:
                require(m == {}, 'no-output metrics')
                continue
            require(set(m) == METRICS | {'probability_row_sum_max_error'}, 'PCM metric inventory')
            require(finite_number(m['probability_row_sum_max_error']) and 0 <= m['probability_row_sum_max_error'] <= 1e-5, 'probability row sums')
            row_sum = max(row_sum, m['probability_row_sum_max_error'])
            for name in METRICS:
                v, total = m[name], metrics[name]
                require(set(v) == {'max_abs', 'fail_elements', 'elements'} and
                        nonnegative_int(v['elements']) and nonnegative_int(v['fail_elements']) and
                        v['fail_elements'] <= v['elements'] and finite_number(v['max_abs']) and v['max_abs'] >= 0,
                        'PCM scalar metric validity')
                if name in ('probabilities', 'logits', 'stage20'):
                    require(v['elements'] == len(call['centers']) * 2599, 'full2599 element coverage')
                total['elements'] += v['elements']
                total['fail_elements'] += v['fail_elements']
                total['failing_calls'] += v['fail_elements'] > 0
                total['max_abs'] = max(total['max_abs'], v['max_abs'])
                if v['fail_elements']:
                    failing.add(name)
            require(m['logits'] == m['stage20'], 'logit/stage20 metric consistency')
        for name in failing:
            metrics[name]['failing_recordings'] += 1
    require([counts[c]['target_hits'] for c in CONDITIONS] == [11, 11, 12] and
            all(c['clips'] == 42 and c['positive_clips'] == 20 and c['confusable_clips'] == 22 and
                c['confusable_triggered_clips'] == 0 for c in counts.values()), 'PCM observed decision counts')
    require(changed == 34 and max_score == 2.694541942516171e-6 and max_score <= 1e-5,
            'PCM event score differences retained')
    require(native['full_fields_events_exact'] is False and native['non_score_event_fields_exact'] is True
            and native['event_scores_within_1e_minus_5'] is True and native['old_B_still_failed'] is True,
            'PCM qualification flags')
    require(native['logit_gate_passed'] is False and native['probability_gate_passed'] is True,
            'PCM gate flags')
    expected = {
        'fbank': (2368160, 6, 6, 0.0012373924255371094),
        'splice': (None, 11, None, None), 'cmvn': (None, 0, 0, 0.0001863241195678711),
        'logits': (25529977, 6126, 63, 0.005767822265625),
        'probabilities': (25529977, 0, 0, 8.046627044677734e-6),
    }
    for name, values in expected.items():
        for field, value in zip(('elements', 'fail_elements', 'failing_calls', 'max_abs'), values):
            if value is not None:
                require(metrics[name][field] == value, 'PCM retained scalar aggregate: ' + name + '/' + field)
    require(metrics['logits']['failing_recordings'] == 42 and metrics['cache']['fail_elements'] > 0,
            'PCM strict numerical failure remains')
    return {'conditions': counts, 'score_different_recordings': changed, 'max_score_difference': max_score,
            'full_fields_events_exact': False, 'non_score_event_fields_exact': True,
            'reported_per_element_metrics': metrics, 'reported_probability_row_sum_max_error': row_sum,
            'tensor_arrays_independently_recomputed': False}


def verify_profile(payloads):
    profile = strict_json(payloads['resource-profile/result.json'])
    pre = strict_json(payloads['resource-profile/preflight.json'])
    source = strict_json(payloads['pcm-reference-preflight.json'])['conditions']['raw']
    samples = {r['recording']: r['frames'] for r in source}
    require(profile['preflight_sha256'] == digest(payloads['resource-profile/preflight.json']) and
            pre['native_result_sha256'] == digest(payloads['pcm-native-result.json']) and
            pre['native_preflight_sha256'] == digest(payloads['pcm-native-preflight.json']), 'profile input bindings')
    require(pre['threads'] == 1 and pre['passes'] == 3 and pre['clips_per_pass'] == 42 and
            pre['warmup'] == 'first3rawclips in original receipt order', 'profile fixed protocol')
    require(pre['CMVN'] == 'inside model timed calls; standalone microcall outside pipeline, no double count/subtraction',
            'profile disjoint CMVN scope')
    require(profile['measurement_notes'][0] == 'frontend_splice_inclusive already includes callback_copy; never add both',
            'profile nested callback scope')
    require(profile['scope'] == 'Python+Torch+ctypes process on x86; numerical candidate failed; no SSC305 extrapolation; no all-C pipeline claim'
            and profile['old_B_still_failed'] is True and profile['PCM_logit_gate_still_failed'] is True,
            'profile qualification scope')
    whole_cpu = whole_wall = init_cpu = audio_samples = 0
    components = dict.fromkeys(sorted(COMPONENTS), 0)
    require(len(profile['passes']) == 3, 'profile pass coverage')
    io_deltas = []
    for index, pass_ in enumerate(profile['passes']):
        require(pass_['index'] == index and pass_['threads_before'] == pass_['threads_after'] == 1,
                'profile sampled threads/pass identity')
        rows = pass_['recordings']
        require(len(rows) == 42 and [r['recording'] for r in rows] == list(samples), 'profile raw42 source order')
        for r in rows:
            require(r['audio_samples'] == samples[r['recording']] and r['logits_and_events_exact'] is True,
                    'profile saved native parity/denominator')
            require(set(r['components']) == COMPONENTS, 'profile component inventory')
            for interval in [r['whole'], r['clip_initialization']] + [v for vs in r['components'].values() for v in vs]:
                require(set(interval) == {'cpu_ns', 'wall_ns'} and all(nonnegative_int(v) for v in interval.values()),
                        'profile nonnegative timings')
            whole_cpu += r['whole']['cpu_ns']
            whole_wall += r['whole']['wall_ns']
            init_cpu += r['clip_initialization']['cpu_ns']
            audio_samples += r['audio_samples']
            for name, intervals in r['components'].items():
                components[name] += sum(v['cpu_ns'] for v in intervals)
        delta = {k: pass_['io_after'][k] - v for k, v in pass_['io_before'].items()}
        require(delta == {'rchar': 106, 'wchar': 0, 'syscr': 2, 'syscw': 0, 'read_bytes': 0,
                          'write_bytes': 0, 'cancelled_write_bytes': 0}, 'profile pass I/O deltas')
        io_deltas.append(delta)
    audio = audio_samples / 16000
    cpu = whole_cpu / 1e9
    rtf = cpu / audio
    require(audio == profile['audio_seconds'] == 235.68 and cpu == profile['pipeline_cpu_seconds'] == 5.758814368
            and rtf == profile['pipeline_cpu_RTF'], 'profile RTF numerator/denominator')
    require(whole_wall == 5734398696 and init_cpu == 74619186, 'profile whole/init timing totals')
    require(components == {'model_including_cmvn': 5201670355, 'frontend_splice_inclusive': 152644713,
                          'callback_copy': 8003267, 'torch_softmax': 27976982, 'python_decoder': 172264473,
                          'standalone_cmvn_micro': 60044159}, 'profile component timing totals')
    # The four disjoint boundary groups are only a subset of the whole interval.
    # Callback is nested; CMVN micro and initialization are explicitly outside.
    disjoint = sum(components[k] for k in ('model_including_cmvn', 'frontend_splice_inclusive',
                                          'torch_softmax', 'python_decoder'))
    require(disjoint <= whole_cpu and components['callback_copy'] <= components['frontend_splice_inclusive'],
            'profile non-double-counted timing scope')
    require((profile['parameter_payload_bytes'], profile['pcm_C_state_bytes'], profile['model_C_ctypes_mirror_bytes'],
             profile['declared_df_step_scratch_float_bytes'], profile['peak_RSS_bytes']) ==
            (3027732, 62264, 22544, 4624, 273309696), 'profile retained size measurements')
    return {'passes': 3, 'clips_per_pass': 42, 'audio_seconds': audio, 'pipeline_cpu_seconds': cpu,
            'pipeline_wall_seconds': whole_wall / 1e9, 'pipeline_cpu_RTF': rtf,
            'component_cpu_ns': components, 'disjoint_component_cpu_ns': disjoint,
            'clip_initialization_cpu_ns_outside_pipeline': init_cpu, 'pass_io_deltas': io_deltas,
            'SSC305_measured': False, 'numerically_qualified': False}


def verify_semantics(payloads):
    for path in ('pcm-native-exit-code.txt', 'pcm-reference-exit-code.txt', 'resource-profile/exit-code.txt'):
        require(payloads[path] == b'0\n', 'original execution exit status')
    return {'schema_version': 1, 'scope': 'retained scalar reports, event records and x86 timing only; no inference or NPZ reconstruction',
            'history': verify_history(payloads), 'pcm': verify_pcm(payloads), 'profile': verify_profile(payloads)}


def verify(root=BASE):
    payloads = read_archive(root)
    summary = verify_semantics(payloads)
    require(summary == strict_json(safe_read(Path(root).absolute(), 'summary.json')), 'recomputed summary mismatch')
    return {'verified': True, 'original_logical_files': len(payloads),
            'scope': summary['scope'], 'strict_numerical_qualification': False,
            'pcm_target_hits': [summary['pcm']['conditions'][c]['target_hits'] for c in CONDITIONS],
            'native_profile_cpu_RTF': summary['profile']['pipeline_cpu_RTF']}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
