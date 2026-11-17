"""Independent stdlib reader for saved full-PCM traces; never executes a model."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

WORKER_SHA = '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'
PLAN_SHA = 'fd4fe7fe456eea2dab03bfc56362c4aa6311f86e28dfe98837ec398af750799f'
QUALITY = (50040751, '56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1')
INCUMBENT = (32878537, 'acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15')
HEAD = (197908, 'd3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4')
NPZ = (199356, '75c1d13203f77fcec9b66013bd6fc8c1979cb3397207931eaf31241eeb396a35')
AUDIT = (314019, '9c1f1769a41552ec21a07d5cc02a645c73ca689b7f901037b28a53766454057d')
LIBS = {'native': (2441224, '040089f781d87366a3d8a27933043aa96e891849f2bf9ce6bbd2321d0d70152a'),
        'arm': (2431812, '7865cd7fcb81cdd900899a307d304862580ee3f57e51746c47cbc07e93bb9218')}
REF_PINS = {'runtime/head-stream.bin': '614bb845016bbf7e3ea87a859972659e71628acc8b1e968e69577e8c117df72a',
            'runtime/reference-frames.jsonl': 'b50484d125c0492bd43f2e01e473124e6a8104192f824fd5917aa7ae1c4a5f07',
            'runtime/CASE_MAP.json': '7d7f64e7f0c1b7950fe7e1af19a74c82319f38f0d61cf4a159e37354d7b1ac3a'}
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed',
               'kws_model_delivered', 'ssc305_qualification')
CASES = [
    ('encov-new8-7ec79719666821c08c694dde', 'new8', 54400, 31, 16, 15873,
     '4295f2257a0ce0b1495e55685e1366bde41bc52974cb1f489b826b1fc8593643', [[2, 54400]]),
    ('encov-new8-69e0f996e6049f0d58d78d65', 'new8', 56960, 33, 18, 15904,
     '5d764e3f68c46b3cb049ebf2e4e5b8acb56d4598c2fa54eb374b104e71c5be8d', [[1, 55680]]),
    ('encov-new8-28b17376eefb7d7a62334bab', 'new8', 58240, 34, 19, 15937,
     'ada5d3fea5aea3083880edb6b5a77abf38c52358bbe5535f8b774e4bba5c2a42', []),
    ('encov-new8-1556744d0babcec74d18c76c', 'new8', 63360, 38, 23, 16003,
     '247bc5b97ad44be4ed56a6d3654debec405a191ed0ed9aae810ee642b177854f', []),
    ('enfinal144-5602dd573b0afb64e36b51b2', 'snr10', 55680, 32, 17, 12614,
     'c21fb94514ec0b9efaf8383664a1c6c9a28e65d96d090ff05ca80379d89bb95e', [[2, 54400]])]
LIMITS = {'embedding': (0.001, 0.0002), 'logits': (0.003, 0.0002), 'probabilities': (0.0005, 0.0)}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def matches(raw, pin):
    return (len(raw), digest(raw)) == pin


def unique(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, 'duplicate JSON key')
        d[k] = v
    return d


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class Evidence:
    def __init__(self, path, pin):
        self.raw = Path(path).read_bytes()
        require(matches(self.raw, pin), 'external artifact identity')
        self.z = zipfile.ZipFile(io.BytesIO(self.raw))
        members = self.z.infolist()
        self.names = {m.filename for m in members}
        require(len(self.names) == len(members) <= 10000, 'duplicate ZIP entries')
        require(sum(m.file_size for m in members) <= 512 * 1024**2, 'ZIP expanded size')
        for m in members:
            path = m.filename.rstrip('/')
            require(path and not PurePosixPath(path).is_absolute() and '\\' not in path and '\0' not in path
                    and all(p not in ('', '.', '..') for p in path.split('/'))
                    and not m.flag_bits & 1 and m.file_size <= 128 * 1024**2
                    and stat.S_IFMT(m.external_attr >> 16) in (0, stat.S_IFDIR, stat.S_IFREG), 'unsafe ZIP member')

    def read(self, name):
        return self.z.read(name)

    def json(self, name):
        return loads(self.read(name))


def check_reference(f, q, old, case):
    cid, condition, samples, count, ready, first, pcm_sha, events = case
    original_pcm = q.read('audio/new/' + cid + '.pcm') if condition == 'new8' else old.read('audio/' + cid + '.pcm')
    pcm = f.read('inputs/' + cid + '.pcm')
    require(pcm == original_pcm and matches(pcm, (2 * samples, pcm_sha)), 'saved PCM source identity: ' + cid)
    data = f.json('inputs/' + cid + '.reference.json')
    c = data['case']
    require((c['id'], c['condition'], c['samples'], c['records'], c['ready'], c['first_record'], c['pcm_sha256'], c['events']) == case,
            'selected case facts: ' + cid)
    mapping = [r for r in q.json('runtime/CASE_MAP.json') if r['id'] == cid and r['condition'] == condition]
    require(len(mapping) == 1, 'unique original case mapping')
    mapping = mapping[0]
    require((mapping['first_record'], mapping['records'], mapping['observed_samples'], mapping['head_outputs']) ==
            (first, count, samples, ready), 'original case geometry')
    original = [loads(line) for line in q.read('runtime/reference-frames.jsonl').splitlines()]
    stream = q.read('runtime/head-stream.bin')
    require(stream[:8] == b'EHSTR01\0' and len(stream) == 12 + 396 * len(original)
            and struct.unpack_from('<I', stream, 8)[0] == len(original), 'reference embedding binary ABI')
    refs, embeddings = [], []
    for i in range(count):
        idx = first + i
        row = original[idx]
        global_sample, reset = struct.unpack_from('<QI', stream, 12 + 396 * idx)
        vec = list(struct.unpack_from('<96f', stream, 24 + 396 * idx))
        local = 16000 + 1280 * i
        require(row['id'] == cid and row['condition'] == condition and row['local_sample'] == local
                and global_sample == row['sample'] == mapping['offset'] + local
                and reset == int(i == 0) and row['ready'] is (i >= 15), 'independent local/global clock')
        refs.append({'sample': local, 'ready': row['ready'], 'keyword': row['keyword'],
                     'logits': row['logits'], 'probabilities': row['probabilities']})
        embeddings.append(vec)
    require(data['scores'] == refs and data['embeddings'] == embeddings, 'saved reference exact original extraction')
    require([[r['keyword'], r['sample']] for r in refs if r['keyword']] == events, 'frozen original events')
    return data


def numeric_summary(expected, actual, limit):
    require(len(expected) == len(actual), 'numeric vector count')
    absolute, violations, n = 0.0, 0, 0
    for a, b in zip(expected, actual):
        require(len(a) == len(b), 'numeric vector width')
        for x, y in zip(a, b):
            require(math.isfinite(x) and math.isfinite(y), 'finite saved numeric values')
            error = abs(x - y)
            absolute = max(absolute, error)
            violations += error > limit[0] + limit[1] * abs(x)
            n += 1
    return {'elements': n, 'max_abs_error': absolute, 'violations': violations,
            'atol': limit[0], 'rtol': limit[1], 'passed': violations == 0}


def inspect_trace(raw, case, reference):
    cid, condition, samples, count, ready, first, pcm_sha, expected_events = case
    require(raw[:8] == b'EKWSTRC1' and len(raw) >= 40, 'full trace header')
    require(struct.unpack_from('<6IQ', raw, 8) == (1, 1, 16000, 320, 96, 3, samples), 'trace ABI and geometry')
    pos = 40
    def record(kind, size):
        nonlocal pos
        require(pos + 5 + size <= len(raw), 'truncated trace')
        require(struct.unpack_from('<BI', raw, pos) == (kind, size), 'trace record order and length')
        value = raw[pos + 5:pos + 5 + size]
        pos += 5 + size
        return value
    embeddings, logits, probabilities, states, events = [], [], [], [], []
    for i in range(count):
        local = 16000 + i * 1280
        e = record(1, 400)
        require(struct.unpack_from('<QQ', e) == (i, local), 'trace embedding sequence/clock')
        embeddings.append(list(struct.unpack_from('<96f', e, 16)))
        s = record(2, 48)
        ordinal, rdy, keyword, sample = struct.unpack_from('<QIIQ', s)
        values = list(struct.unpack_from('<6f', s, 24))
        require((ordinal, rdy, sample) == (i, int(i >= 15), local) and keyword in (0, 1, 2), 'trace ready and score clock')
        require(all(math.isfinite(v) for v in values) and all(0 <= v <= 1 for v in values[3:]), 'finite trace score')
        require(rdy or keyword == 0 and all(v == 0 for v in values), 'nonready state must be zero')
        states.append((sample, bool(rdy), keyword))
        logits.append(values[:3]); probabilities.append(values[3:])
        if keyword:
            ev = record(3, 32)
            ei, frame, cls, stamp, confidence = struct.unpack('<QQIQf', ev)
            require((ei, frame, cls, stamp) == (len(events), i, keyword, local)
                    and math.isfinite(confidence) and struct.pack('<f', confidence) == struct.pack('<f', values[3 + keyword]),
                    'event callback must match score and exact confidence bits')
            events.append([keyword, local])
    end = list(struct.unpack('<6Q', record(255, 48)))
    require(end == [samples, count, count, ready, len(events), (samples + 319) // 320] and pos == len(raw),
            'exact END; no missing/extra/trailing records')
    expected_states = [(r['sample'], r['ready'], r['keyword']) for r in reference['scores']]
    require(states == expected_states and events == expected_events, 'exact reference states/events')
    comparisons = {'embedding': numeric_summary(reference['embeddings'], embeddings, LIMITS['embedding']),
        'logits': numeric_summary([r['logits'] for r in reference['scores']], logits, LIMITS['logits']),
        'probabilities': numeric_summary([r['probabilities'] for r in reference['scores']], probabilities, LIMITS['probabilities'])}
    require(all(r['passed'] for r in comparisons.values()), 'frozen numerical tolerance violated: ' + cid)
    return {'case_id': cid, 'condition': condition, 'trace': identity(raw), 'input_pcm_sha256': pcm_sha,
            'observed_samples': samples, 'embedding_records': count, 'ready_outputs': ready,
            'exact_clock_ready_keyword': True, 'exact_events': True, 'events': events,
            'exact_end': True, 'end': end, 'numeric': comparisons, 'passed': True}


def command_receipts(f, tag, model=False, build=False):
    a, b = f.json('evidence/' + tag + '.started.json'), f.json('evidence/' + tag + '.exit.json')
    require(a['argv'] == b['argv'] and a['attempt'] == b['attempt'] == 1
            and a['retry_count'] == b['retry_count'] == 0 and a['model_process'] is model
            and b['model_process'] is model and a['consumer_build'] is build and b['consumer_build'] is build,
            'single started/completed command receipt: ' + tag)
    require(a['new_session'] is False and a['cpu_threads'] == 1 and b['returncode'] == 0
            and b['error'] is None and b['timed_out'] is False and b['descendants_supervised_by_parent_process_group'] is True,
            'successful supervised command: ' + tag)
    require(0 <= b['wall_seconds'] <= 1200 and a['monotonic_started'] < a['global_deadline'], 'command deadline receipt')
    return a['argv']


def audit(args):
    f = Evidence(args.artifact, (args.expected_bytes, args.expected_sha256))
    q, old = Evidence(args.quality, QUALITY), Evidence(args.incumbent, INCUMBENT)
    manifest = f.json('MANIFEST.json')
    require(manifest['schema'] == 'english-coverage-fullpcm-once-v1/artifact'
            and manifest['supervised_success'] is True and manifest['downloads_excluded'] is True
            and all(manifest.get(k) is False for k in FALSE_FLAGS), 'successful fullPCM manifest scope')
    require(set(manifest['files']) | {'MANIFEST.json'} == f.names, 'all artifact members covered exactly')
    for name, pin in manifest['files'].items():
        require(identity(f.read(name)) == pin, 'manifest byte mismatch: ' + name)
    require(digest(f.read('source/english_coverage_fullpcm_once.py')) == WORKER_SHA
            and digest(f.read('source/PLAN.json')) == PLAN_SHA, 'reviewed exact source and frozen plan')
    result, supervisor, started = f.json('evidence/RESULT.json'), f.json('evidence/SUPERVISOR_EXIT.json'), f.json('STARTED.json')
    for d in (result, supervisor, started['binding']):
        require(str(d['run_id']) == str(args.expected_run) and d['head_sha'] == args.expected_head
                and d['run_attempt'] == 1 and d['main_before'] == '4f4d3f15214613c0c3a8f92c871da0b419b3f7f7'
                and d['repository'] == 'jiying2007/kws-data' and d['public_repository'] is True and d['event'] == 'push',
                'external run/commit/main-successor binding')
    require(supervisor['passed'] is True and supervisor['returncode'] == 0 and supervisor['error'] is None
            and supervisor['timed_out'] is False and supervisor['retry_count'] == 0
            and supervisor['cleanup'] == {'descendants_found': False, 'terminated': False, 'killed': False}
            and 0 <= supervisor['wall_seconds'] <= 1200, 'supervisor success and no descendant leak')
    require(started['worker_sha256'] == WORKER_SHA and started['maximum_seconds'] == 1200 and started['retry_count'] == 0
            and started['monotonic_deadline'] - started['monotonic_started'] == 1200, 'frozen supervision budget')
    require(result['passed'] is True and result['status'] == 'VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM'
            and result['consumer_build_attempts'] == 2 and result['full_pcm_process_attempts'] == result['completed_comparisons'] == 10
            and result['retry_count'] == 0 and result['plan_sha256'] == PLAN_SHA, 'exact ten fullPCM passes and two builds')
    require(all(result.get(k) is False for k in FALSE_FLAGS) and result['ssc305'] == result['independent_far'] == result['independent_frr'] == 'NOT_RUN'
            and result['word_end_latency'] == 'NOT_MEASURED' and result['performance_measured'] is False,
            'preserved unmeasured acoustic/device criteria')
    for k in ('new_onnx_calls', 'python_head_forwards', 'optimizer_updates', 'new_tts_asr_phone_calls', 'new_mixes'):
        require(result[k] == 0, 'no new model reference/training/mixing: ' + k)
    require(matches(f.read('models/coverage-head.bin'), HEAD) and matches(f.read('models/coverage-head.npz'), NPZ)
            and matches(f.read('evidence/INDEPENDENT_QUALITY_AUDIT.json'), AUDIT), 'candidate and independent quality audit bytes')
    require(f.read('models/coverage-head.bin') == q.read('models/coverage-head.bin')
            and f.read('models/coverage-head.npz') == q.read('models/coverage-head.npz'), 'exact original coverage weight identity')
    for path, pin in REF_PINS.items():
        require(digest(q.read(path)) == pin, 'original reference provenance')
    original_manifest = f.read('installed-base/ORIGINAL_MANIFEST.json')
    require(digest(original_manifest) == '09ed60d3f6701837e0e376b58bccefc4c008833c4743d53bef973b5f63d70b09', 'original SDK manifest pin')
    original_files = loads(original_manifest)['files']
    for name in f.names:
        if name.startswith('installed-base/') and name != 'installed-base/ORIGINAL_MANIFEST.json':
            relative = name[len('installed-base/'):]
            require(identity(f.read(name)) == original_files[relative], 'unchanged installed byte: ' + relative)
    reports, declared_reports = [], []
    models, model_exits, builds = [], [], []
    for name in f.names:
        if name.startswith('evidence/') and name.endswith('.started.json'):
            doc = f.json(name)
            if doc.get('model_process'):
                models.append(name)
            if doc.get('consumer_build'):
                builds.append(name)
        if name.startswith('evidence/') and name.endswith('.exit.json'):
            doc = f.json(name)
            if doc.get('model_process'):
                model_exits.append(name)
    require(set(models) == {'evidence/' + b + '-case' + str(i) + '.started.json' for b in ('native', 'arm') for i in range(5)},
            'closed ten model-start receipts')
    require(set(model_exits) == {n.replace('.started.json', '.exit.json') for n in models}
            and set(builds) == {'evidence/build-native.started.json', 'evidence/build-arm.started.json'},
            'closed ten model exits and two build starts')
    references = [check_reference(f, q, old, case) for case in CASES]
    for backend in ('native', 'arm'):
        library = f.read('installed-base/sdk-' + backend + '/lib/libekws_research_stream.a')
        require(matches(library, LIBS[backend]), 'frozen installed library')
        executable = f.read('bin/trace-' + backend)
        require(executable[:4] == b'\x7fELF' and executable[5] == 1
                and executable[4] == (2 if backend == 'native' else 1)
                and struct.unpack_from('<H', executable, 18)[0] == (62 if backend == 'native' else 40), 'consumer executable architecture')
        elf = f.json('evidence/' + backend + '-ELF.json')
        require(elf['identity'] == identity(executable) and elf['installed_library'] == identity(library), 'consumer/library binding')
        build = command_receipts(f, 'build-' + backend, build=True)
        require('-Wl,--wrap=ehead_step' in build and '-fno-fast-math' in build and '-ffp-contract=off' in build
                and ('-static' in build) == (backend == 'arm'), 'fixed trace link and numerical flags')
        for i, (case, reference) in enumerate(zip(CASES, references)):
            tag = backend + '-case' + str(i)
            argv = command_receipts(f, tag, model=True)
            require(len(argv) == (4 if backend == 'native' else 5)
                    and (backend == 'native' or argv[0] == 'qemu-arm')
                    and Path(argv[-4]).name == 'trace-' + backend and Path(argv[-3]).name == 'coverage-head.bin'
                    and Path(argv[-2]).name == case[0] + '.pcm' and Path(argv[-1]).name == backend + '-' + case[0] + '.trace',
                    'exact backend/head/PCM/trace invocation')
            raw = f.read('traces/' + backend + '-' + case[0] + '.trace')
            report = inspect_trace(raw, case, reference)
            meta = f.json('evidence/' + tag + '.stdout')
            expected_meta = {'schema': 'english-stream-numeric-trace-v1', 'trace_completed': True,
                'format_magic': 'EKWSTRC1', 'format_version': 1, 'abi': 1, 'sample_rate_hz': 16000,
                'encoding': 's16le_mono', 'chunk_samples': 320, 'input_samples': case[2], 'input_bytes': case[2] * 2,
                'input_repetitions': 1, 'resets': 0, 'eof_padding_samples': 0, 'feed_calls': report['end'][-1],
                'embedding_records': case[3], 'real_head_calls': case[3], 'score_records': case[3], 'ready_outputs': case[4],
                'event_records': len(case[7]), 'trace_records': 2 * case[3] + len(case[7]) + 1, 'trace_bytes': len(raw),
                'end_record_written': True, 'score_observer_enabled': True, 'original_head_link_wrap': True, 'performance_measured': False}
            require(meta == expected_meta, 'trace stdout independently matches complete raw trace')
            declared = f.json('evidence/' + tag + '.comparison.json')
            declared_reports.append(declared)
            require(declared['passed'] is True and declared['exact_clock_ready_keyword'] is True
                    and declared['exact_events'] is True and declared['actual_events'] == report['events']
                    and declared['trace'] == report['trace'], 'saved comparison exact events and trace identity')
            for kind in LIMITS:
                for key in ('elements', 'max_abs_error', 'violations', 'atol', 'rtol', 'passed'):
                    require(declared['numeric'][kind][key] == report['numeric'][kind][key], 'independent error recomputation: ' + kind)
            report['backend'] = backend
            reports.append(report)
    require(result['comparisons'] == declared_reports and len(declared_reports) == 10
            and all(x['passed'] is True for x in declared_reports), 'exact result comparison roster')
    paired = []
    for i, case in enumerate(CASES):
        native, arm = reports[i], reports[5 + i]
        pair = {'case_id': case[0], 'exact_native_arm_events_and_states': True,
                'whole_trace_bitexact': native['trace'] == arm['trace']}
        paired.append(pair)
    require(result['paired_backends'] == paired, 'paired backend decisions')
    rescue = f.json('evidence/SAVED_RESCUE.json')
    require(rescue['id'] == CASES[-1][0] and rescue['previous_events'] == []
            and rescue['coverage_events'] == [{'keyword': 2, 'sample': 54400}]
            and rescue['development_miss_rescued'] is True, 'only original 10dB rescue is preserved')
    totals = {k: {'max_abs_error': max(r['numeric'][k]['max_abs_error'] for r in reports),
                  'elements': sum(r['numeric'][k]['elements'] for r in reports), 'violations': 0,
                  'atol': LIMITS[k][0], 'rtol': LIMITS[k][1]} for k in LIMITS}
    return {'schema': 'english-coverage-fullpcm-independent-saved-trace-audit-v1', 'status': 'VERIFIED',
        'artifact': identity(f.raw), 'expected_run': str(args.expected_run), 'expected_head': args.expected_head,
        'expected_external_binding': True, 'auditor_sha256': digest(Path(__file__).read_bytes()),
        'worker_sha256': WORKER_SHA, 'plan_sha256': PLAN_SHA, 'manifest_members_verified': len(manifest['files']),
        'candidate_binary': identity(f.read('models/coverage-head.bin')), 'candidate_npz': identity(f.read('models/coverage-head.npz')),
        'reference_source_quality_zip': identity(q.raw), 'rescued_pcm_source_incumbent_zip': identity(old.raw),
        'original_saved_reference_members': REF_PINS, 'all_five_reference_extractions_and_pcm_bytes_verified': True,
        'supervisor_success_no_live_descendants': True, 'full_pcm_process_start_exit_pairs': 10,
        'consumer_build_start_exit_pairs': 2, 'retry_count': 0, 'reports': reports, 'paired_backends': paired,
        'numeric_totals': totals, 'distinct_previously_exposed_pcm_seconds': 18.04, 'replay_pcm_seconds': 36.08,
        'embeddings_per_backend': 168, 'ready_outputs_per_backend': 93, 'events_per_backend': 3,
        'unique_saved_10db_mero_rescue_verified': True, 'model_forward_reexecutions_by_auditor': 0,
        'compilers_or_c_model_processes_by_auditor': 0, 'new_onnx_calls_by_auditor': 0,
        'new_training_or_mixing_by_auditor': 0, 'independent_acoustic_evaluation': False,
        'performance_measured': False, 'ssc305': 'NOT_RUN', 'independent_far': 'NOT_RUN',
        'independent_frr': 'NOT_RUN', 'word_end_latency': 'NOT_MEASURED', **dict.fromkeys(FALSE_FLAGS, False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'quality', 'incumbent', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-bytes', type=int, required=True)
    for name in ('expected-sha256', 'expected-run', 'expected-head'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    try:
        result = audit(args)
    except Exception as exc:
        result = {'schema': 'english-coverage-fullpcm-independent-saved-trace-audit-v1', 'status': 'FAILED',
                  'error': type(exc).__name__ + ': ' + str(exc), 'model_forward_reexecutions_by_auditor': 0,
                  'compilers_or_c_model_processes_by_auditor': 0, 'expected_run': args.expected_run, 'expected_head': args.expected_head}
    with args.output.open('x', encoding='utf-8') as out:
        out.write(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': digest(args.output.read_bytes()),
                      'error': result.get('error'), 'model_forward_reexecutions_by_auditor': 0}))
    return 0 if result['status'] == 'VERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
