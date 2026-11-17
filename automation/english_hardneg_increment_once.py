"""One reviewed-window incremental research head; default is source-only.

No local model entry is available. The guarded public Actions worker authenticates
all saved inputs, performs one fixed warm-start, freezes actual weights, and
retains both passing and rejected regression evidence. It never decodes audio,
runs the frontend, relabels complete original utterances, or selects a checkpoint.
"""
from __future__ import annotations
import argparse
import ast
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REPO='jiying2007/kws-data'
BEFORE='c312a6ef2335af0b4d16c17c598693fc95273c0c'
SCHEMA='english-reviewed-hardneg-increment-once-v1'
WORKER_PATH='automation/english_hardneg_increment_once.py'
PLAN_PATH='automation/english_hardneg_increment_plan.json'
WORKFLOW_PATH='.github/workflows/english-hardneg-increment-once.yml'
RESEARCH='research/english_hardneg_increment_20261117'
ADMISSION_PATH=RESEARCH+'/SOURCE_ADMISSION.json'
REVIEW_PATH=RESEARCH+'/OWNER_REVIEW.json'
PINS_PATH=RESEARCH+'/INPUT_PINS.json'
FROZEN_PARSER='automation/english_coverage_fullpcm_once.py'
ROSTER_PATH='research/english_hardneg_natural32_20261117/natural32-roster.json'
FALSE_FLAGS=('formal_training_admitted','human_gold','pronunciation_verified','model_promotion_allowed','qualification_allowed')
CONFIG={'threshold_float32':struct.unpack('<f',struct.pack('<f',0.8))[0],
        'confirm_embeddings':3,'cooldown_embeddings':13,'sample_rate':16000,
        'embedding_hop_samples':1280,'head_history':16,'first_embedding_sample':16000,
        'first_ready_sample':35200,'support_samples':35200,'tie_policy':'lowest_class_id',
        'reset_between_originals_only':True,'eof_padding_samples':0}
RECIPE={'updates':96,'seed':20261117,'learning_rate':0.0001,'weight_decay':0.0001,
        'gradient_clip_norm':1.0,'optimizer':'AdamW','optimizer_state':'RESET_NO_OLD_MOMENTS',
        'cpu_threads':1,'positive_loss':'UNCHANGED_WEAK_NEGATIVE_LOG_MEAN_TARGET_PROBABILITY',
        'ordinary_negative_loss':'MEAN_WINDOW_BACKGROUND_CROSS_ENTROPY',
        'reviewed_negative_loss':'MEAN_UNIQUE_FULL_SUPPORT_WINDOWS_PER_ORIGINAL_SOURCE',
        'aggregation':'ORIGINAL_SIX_TERMS_PLUS_ONE_SOURCE_TERM_DIVIDED_BY_SEVEN',
        'ordinary_draws':'EXACT_PREVIOUS_COVERAGE_96_BY_6_DRAWS',
        'reviewed_draws':'SORTED_EIGHT_SOURCE_IDS_STEP_MOD_EIGHT',
        'normalization':'FROZEN_BYTE_IDENTICAL','head_parameter_count':49283,
        'reviewed_sources':8,'reviewed_windows':9,'reviewed_head_inputs':134,
        'training_bag_forwards':672,'checkpoint_selection':False,'parameter_sweeps':False}
SOURCES={
 'quality':{'run_id':44042856615,'artifact_id':14310138100,'name':'english-coverage-quality-44042856615',
  'head_sha':'4f4d3f15214613c0c3a8f92c871da0b419b3f7f7','bytes':50040751,
  'sha256':'56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1'},
 'natural':{'run_id':44063710088,'artifact_id':14318639588,'name':'english-hardneg-natural32-retained-44063710088',
  'head_sha':'4f998e89d9f344677b8a97fdb78352e42b8cf6bd','bytes':34035367,
  'sha256':'03e2d7e1953ead05736dc2ff1de60c115bc24ee45442a2322886fc3d02050807'}}
BASE_NPZ={'bytes':199356,'sha256':'75c1d13203f77fcec9b66013bd6fc8c1979cb3397207931eaf31241eeb396a35'}
BASE_BIN={'bytes':197908,'sha256':'d3e39adf72495034de0405abb997b76744885f91868eccfa97ba09b929bc63e4'}
SOURCE_PINS={'automation/english_coverage_fullpcm_once.py': {'bytes': 44356, 'sha256': '70a9ae5c36191a75f58e8819ac4e6e12f3f7e464e5891d1b1e5fdc648e9ae422'}, 'research/english_hardneg_natural32_20261117/natural32-roster.json': {'bytes': 56281, 'sha256': '31457faf8efd0cf77629c1a6f8de67bc101190eb118bcfe9ac2f256acca41239'}}
SOURCE_PINS[WORKFLOW_PATH]={'bytes':3483,'sha256':'7e5bb73ef255d7ccee314faefa1261eec288d6539eba25066e466b9855c8c050'}

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode() + b'\n'

def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result

def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))

def regular(path, limit=128 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    return path.read_bytes()

def file_identity(path, limit=512 * 1024**2):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'REGULAR_FILE_REQUIRED')
    need(path.stat().st_size <= limit, 'FILE_SIZE_LIMIT')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            size += len(chunk)
            need(size <= limit, 'FILE_SIZE_LIMIT')
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}

def put(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'OUTPUT_SYMLINK')
    with path.open('xb') as target:
        target.write(raw)

def once(path, value):
    # Publish complete JSON create-only. A killed write leaves a visible pending
    # file, not an apparently complete receipt; link never overwrites a target.
    path = Path(path)
    # Parent sealing must not collide with an interrupted child's pending file.
    pending = path.with_name(path.name + '.pending-' + str(os.getpid()))
    put(pending, canonical(value))
    try:
        os.link(pending, path)
    finally:
        pending.unlink()

def safe_member(name):
    return (type(name) is str and name and not PurePosixPath(name).is_absolute()
            and '\\' not in name and '\x00' not in name
            and all(x not in ('', '.', '..') for x in name.split('/')))

class Archive:
    def __init__(self, path, pin):
        need(file_identity(path) == {k: pin[k] for k in ('bytes', 'sha256')}, 'ARCHIVE_PIN')
        self.z = zipfile.ZipFile(path)
        infos = self.z.infolist()
        need(len(infos) == len({i.filename for i in infos}) <= 10000, 'UNIQUE_ZIP_MEMBERS')
        need(sum(i.file_size for i in infos) <= 384 * 1024**2, 'ZIP_EXPANSION_LIMIT')
        for item in infos:
            name = item.filename[:-1] if item.is_dir() else item.filename
            need(safe_member(name) and not item.flag_bits & 1 and item.file_size <= 128 * 1024**2,
                 'UNSAFE_ZIP_MEMBER')
            need(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'UNSAFE_ZIP_TYPE')

    def read(self, name):
        need(safe_member(name), 'UNSAFE_ZIP_READ')
        return self.z.read(name)

    def json(self, name):
        return loads(self.read(name))

    def close(self):
        self.z.close()

def execution_identity(plan, env=None, event=None, head=None, parent=None):
    env = os.environ if env is None else env
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPO and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1', 'PUBLIC_MAIN_FIRST_PUSH_ONLY')
    commit, run_id = env.get('GITHUB_SHA', ''), env.get('GITHUB_RUN_ID', '')
    need(re.fullmatch('[0-9a-f]{40}', commit) is not None and run_id.isdigit(), 'ACTUAL_RUN_AND_HEAD')
    if event is None:
        event = loads(regular(env['GITHUB_EVENT_PATH'], 8 * 1024**2))
    need(event.get('before') == plan['main_before'] == BEFORE and event.get('after') == commit and
         event.get('ref') == 'refs/heads/main' and event.get('repository', {}).get('private') is False and
         event.get('repository', {}).get('full_name') == REPO and event.get('deleted') is False and
         event.get('forced') is False, 'EXACT_PUBLIC_SUCCESSOR_EVENT')
    if head is None:
        head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    if parent is None:
        parent = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD^1'], text=True, timeout=10).strip()
    need(head == commit and parent == BEFORE, 'WORKTREE_EXACT_SUCCESSOR_OF_FROZEN_BASE')
    return {'repository': REPO, 'run_id': run_id, 'head_sha': commit, 'main_before': BEFORE,
            'run_attempt': 1, 'event': 'push', 'public_repository': True}

def command_env(auth=False):
    env = dict(os.environ)
    if not auth:
        for key in tuple(env):
            if key in ('GH_TOKEN', 'GITHUB_TOKEN') or key.endswith('_TOKEN') or key.startswith('ACTIONS_RUNTIME_'):
                env.pop(key, None)
    env.update(DEBIAN_FRONTEND='noninteractive', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return env

class Runner:
    """All commands inherit the single supervised child's process group."""
    def __init__(self, output, deadline):
        self.output, self.deadline = Path(output), deadline
        self.counts = {'numeric': 0, 'native': 0}

    def command(self, tag, argv, timeout=120, auth=False, maximum_stdout=16 * 1024**2,
                kind=None, stdout_path=None):
        need(re.fullmatch('[a-z0-9_.-]+', tag) is not None, 'COMMAND_TAG')
        need(time.monotonic() < self.deadline, 'GLOBAL_DEADLINE')
        if kind is not None:
            need(kind in self.counts, 'COMMAND_KIND')
            need(self.counts[kind] < {'numeric': 1, 'native': 1}[kind], 'COMMAND_BUDGET:' + kind)
            self.counts[kind] += 1
        argv = [str(x) for x in argv]
        evidence, started = self.output / 'evidence', time.monotonic()
        stdout = evidence / (tag + '.stdout') if stdout_path is None else Path(stdout_path)
        need(stdout.is_relative_to(self.output) and not stdout.exists() and
             not any(p.is_symlink() for p in (stdout, *stdout.parents)), 'NEW_CONFINED_COMMAND_OUTPUT')
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stderr = evidence / (tag + '.stderr')
        once(evidence / (tag + '.started.json'), {'argv': argv, 'attempt': 1, 'retry_count': 0,
             'kind': kind, 'monotonic_started': started, 'global_deadline': self.deadline,
             'new_session': False, 'cpu_threads': 1, 'stdout': stdout.relative_to(self.output).as_posix()})
        process, error, timed_out = None, None, False
        with stdout.open('xb') as out, stderr.open('xb') as err:
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    env=command_env(auth), start_new_session=False, cwd=str(ROOT))
                end = min(self.deadline, started + timeout)
                while process.poll() is None:
                    if time.monotonic() >= end:
                        timed_out = True
                        raise TimeoutError('COMMAND_DEADLINE')
                    need(stdout.stat().st_size <= maximum_stdout and stderr.stat().st_size <= 4 * 1024**2,
                         'COMMAND_OUTPUT_LIMIT')
                    time.sleep(0.05)
                need(stdout.stat().st_size <= maximum_stdout and stderr.stat().st_size <= 4 * 1024**2,
                     'COMMAND_OUTPUT_LIMIT')
                need(process.returncode == 0, 'COMMAND_NONZERO_EXIT')
            except BaseException as exc:
                error = type(exc).__name__ + ':' + str(exc)
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                once(evidence / (tag + '.exit.json'), {'argv': argv, 'returncode': None if process is None else process.returncode,
                    'error': error, 'timed_out': timed_out, 'wall_seconds': time.monotonic() - started,
                    'attempt': 1, 'retry_count': 0, 'kind': kind,
                    'descendants_supervised_by_parent_process_group': True})
        need(error is None, error or 'COMMAND_FAILED')
        return stdout

    @contextlib.contextmanager
    def stage(self, name):
        need(time.monotonic() < self.deadline and re.fullmatch('[a-z0-9-]+', name), 'STAGE_DEADLINE_AND_NAME')
        start = time.monotonic()
        once(self.output / 'evidence' / ('stage-' + name + '.started.json'), {'stage': name, 'monotonic_started': start})
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__ + ':' + str(exc)
            raise
        finally:
            once(self.output / 'evidence' / ('stage-' + name + '.exit.json'),
                 {'stage': name, 'error': error, 'wall_seconds': time.monotonic() - start, 'retry_count': 0})

    def download(self, name, pin=None, expected_conclusion='success'):
        pin = SOURCES[name] if pin is None else pin
        run = loads(regular(self.command(name + '-run', ['gh', 'api', 'repos/' + REPO + '/actions/runs/' + str(pin['run_id'])], auth=True)))
        need(run.get('id') == pin['run_id'] and run.get('head_sha') == pin['head_sha'] and
             run.get('conclusion') == expected_conclusion and run.get('status') == 'completed' and run.get('run_attempt') == 1 and
             run.get('repository', {}).get('private') is False and run.get('repository', {}).get('full_name') == REPO,
             'PINNED_PUBLIC_SOURCE_RUN')
        artifact = loads(regular(self.command(name + '-artifact', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id'])], auth=True)))
        need(artifact.get('id') == pin['artifact_id'] and artifact.get('name') == pin['name'] and
             artifact.get('expired') is False and artifact.get('size_in_bytes') == pin['bytes'] and
             artifact.get('digest') == 'sha256:' + pin['sha256'] and
             artifact.get('workflow_run', {}).get('id') == pin['run_id'] and
             artifact.get('workflow_run', {}).get('head_sha') == pin['head_sha'], 'PINNED_SOURCE_ARTIFACT')
        target = self.output / 'downloads' / (name + '.zip')
        self.command(name + '-zip', ['gh', 'api', 'repos/' + REPO + '/actions/artifacts/' + str(pin['artifact_id']) + '/zip'],
                     auth=True, timeout=240, maximum_stdout=pin['bytes'], stdout_path=target)
        need(file_identity(target) == {k: pin[k] for k in ('bytes', 'sha256')}, 'DOWNLOADED_EXACT_ZIP')
        once(self.output / 'evidence' / (name + '-download.json'), {'source': pin, 'actual': file_identity(target), 'retry_count': 0})
        return target

def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False

def cleanup_group(pid):
    if not group_exists(pid):
        return {'descendants_found': False, 'terminated': False, 'killed': False}
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {'descendants_found': False, 'terminated': False, 'killed': False}
    end = time.monotonic() + 2
    while time.monotonic() < end and group_exists(pid):
        time.sleep(0.05)
    killed = group_exists(pid)
    if killed:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {'descendants_found': True, 'terminated': True, 'killed': killed}

def close_interrupted_receipts(output, supervisor):
    """A killed process has an unknown exit code, never an invented success."""
    errors = []
    for path in sorted((Path(output) / 'evidence').glob('*.started.json')):
        target = path.with_name(path.name.removesuffix('.started.json') + '.exit.json')
        raw = regular(path)
        try:
            started = loads(raw)
            need(type(started) is dict, 'STARTED_RECEIPT_OBJECT')
            damaged = None
        except (ValueError, TypeError) as exc:
            started = {}
            damaged = {'identity': identity(raw), 'parse_error': type(exc).__name__ + ':' + str(exc)}
            errors.append({'path': path.name, **damaged})
        if target.exists():
            continue
        once(target, {'schema': SCHEMA + '/interrupted-step', 'stage': started.get('stage'),
            'kind': started.get('kind'), 'argv': started.get('argv'), 'returncode': None,
            'error': 'INTERRUPTED_BEFORE_STEP_EXIT_RECEIPT', 'completed': False,
            'damaged_started_receipt': damaged,
            'supervisor_cleanup': supervisor['cleanup'], 'timed_out': supervisor['timed_out'],
            'retry_count': 0, 'no_step_retry': True})
    return errors

def supervise(args, plan):
    binding = execution_identity(plan)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_SUPERVISED_OUTPUT_ONLY')
    output.mkdir(mode=0o700)
    (output / 'evidence').mkdir()
    started, process, timed_out, error, cleanup = time.monotonic(), None, False, None, {}
    deadline = started + 1200
    once(output / 'STARTED.json', {'schema': SCHEMA + '/started', 'binding': binding,
        'parent_pid': os.getpid(), 'worker_sha256': plan['worker_sha256'], 'monotonic_started': started,
        'monotonic_deadline': deadline, 'maximum_seconds': 1200, 'retry_count': 0, 'network_calls_at_start': 0})
    env = dict(os.environ, KWS_INCREMENT_PARENT_PID=str(os.getpid()))
    argv = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--_worker', '--output', str(output)]
    def cancelled(signum, _frame):
        raise InterruptedError('SUPERVISOR_SIGNAL_' + str(signum))
    handlers = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / 'evidence/SUPERVISOR.stdout').open('xb') as out, (output / 'evidence/SUPERVISOR.stderr').open('xb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=env, cwd=str(ROOT), start_new_session=True)
            once(output / 'evidence/WORKER_LAUNCH.json', {'pid': process.pid, 'process_group': process.pid,
                 'argv': argv, 'attempt': 1, 'retry_count': 0})
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise TimeoutError('SUPERVISOR_1200_SECONDS')
                time.sleep(0.1)
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process is not None:
            try:
                cleanup = cleanup_group(process.pid)
            except BaseException as exc:
                cleanup = {'error': type(exc).__name__ + ':' + str(exc), 'descendants_found': True}
                error = error or 'PROCESS_GROUP_CLEANUP_FAILED'
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                error = error or 'WORKER_NOT_REAPED'
        receipt = {'schema': SCHEMA + '/supervisor', 'returncode': None if process is None else process.returncode,
            'timed_out': timed_out, 'error': error, 'wall_seconds': time.monotonic() - started,
            'retry_count': 0, 'cleanup': cleanup, **binding}
        worker_result = None
        if (output / 'evidence/RESULT.json').exists():
            try:
                worker_result = loads(regular(output / 'evidence/RESULT.json'))
            except (ValueError, OSError) as exc:
                receipt['error'] = 'WORKER_RESULT_UNREADABLE:' + str(exc)
        receipt['passed'] = (process is not None and process.returncode == 0 and not timed_out and
            receipt['error'] is None and not cleanup.get('descendants_found') and
            type(worker_result) is dict and worker_result.get('passed') is True and
            all(worker_result.get(k) == v for k, v in binding.items()))
        receipt['partial_receipt_errors'] = close_interrupted_receipts(output, receipt)
        if receipt['partial_receipt_errors']:
            receipt['passed'] = False
        once(output / 'evidence/SUPERVISOR_EXIT.json', receipt)
        if type(worker_result) is not dict:
            fallback = result_boundary(binding)
            fallback.update(error='SUPERVISOR_TERMINATED_BEFORE_WORKER_RESULT', supervisor=receipt,
                            numeric_or_native_process_attempts=None if receipt['partial_receipt_errors'] else
                            sum(loads(regular(p)).get('kind') in ('numeric', 'native') for p in (output / 'evidence').glob('*.started.json')),
                            optimizer_updates_completed='UNKNOWN_IF_INTERRUPTED')
            result_path = output / 'evidence/RESULT.json'
            if result_path.exists():
                fallback['preserved_damaged_worker_result'] = file_identity(result_path)
                result_path = output / 'evidence/RESULT_SUPERVISOR_FAILURE.json'
            once(result_path, fallback)
        files = {p.relative_to(output).as_posix(): file_identity(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and 'downloads' not in p.relative_to(output).parts}
        once(output / 'MANIFEST.json', {'schema': SCHEMA + '/artifact', 'files': files,
             'downloads_excluded': True, 'supervised_success': receipt['passed'],
             'authoritative_completion_receipt': 'evidence/SUPERVISOR_EXIT.json', **dict.fromkeys(FALSE_FLAGS, False)})
    print(canonical(receipt).decode(), end='')
    return 0 if receipt['passed'] else 1


def summarize_case(row, events):
    source = row.get("source", {})
    start = source.get("clip_start_sample")
    eof = source.get("original_clip_eof_sample")
    expected = row["weak_class"]
    need(row.get("role") in ("train", "development") and type(row.get("observed_samples")) is int
         and row["observed_samples"] > 0, "valid row role and observed length")
    need((start is None and eof is None) or
         (type(start) is int and type(eof) is int and 0 < start < eof <= row["observed_samples"]),
         "clip interval is inside observed PCM")
    need(type(expected) is int and expected in (0, 1, 2), "weak intent class")
    need(start is not None or expected == 0, "positive intent requires a source clip interval")
    need(all(type(e.get("sample")) is int and 0 < e["sample"] <= row["observed_samples"]
             and type(e.get("keyword")) is int and e["keyword"] in (1, 2) for e in events), "observed event clocks")
    need(all(a["sample"] < b["sample"] for a, b in zip(events, events[1:])), "strictly ordered unique events")
    prefix = [e for e in events if start is not None and e["sample"] <= start]
    after = [e for e in events if start is None or e["sample"] > start]
    correct = [e for e in after if expected and e["keyword"] == expected]
    wrong = [e for e in after if expected and e["keyword"] != expected]
    deficit = bool(expected and not correct)
    extra = max(0, len(correct) - 1)
    error = bool(prefix or wrong or deficit or extra or (expected == 0 and events))
    return {"id": row["id"], "role": row["role"], "weak_class": expected,
            "events": events, "matched_intent_after_clip_start": bool(correct),
            "prefix_events": prefix, "wrong_keyword_after_clip_start": wrong,
            "extra_same_keyword_events": extra, "no_matching_event_in_observed_context": deficit,
            "weak_negative_intent_events": len(events) if expected == 0 else 0,
            "research_error": error, "observed_samples": row["observed_samples"],
            "clip_start_sample": start, "original_clip_eof_sample": eof,
            "event_minus_clip_eof_samples": [e["sample"] - eof for e in correct] if eof is not None else [],
            "eof_prefix_event_count": sum(e["sample"] <= eof for e in events) if eof is not None else None,
            "no_event_status": "CENSORED_NOT_FRR" if not events and expected else None,
            "word_end_latency": "NOT_MEASURED", "independent_acoustic_evaluation": False}


def expected_plan(worker_sha, source_pins):
    return {'schema': SCHEMA, 'repository': REPO, 'main_before': BEFORE,
        'worker_sha256': worker_sha, 'source_pins': source_pins, 'sources': SOURCES,
        'base_npz': BASE_NPZ, 'base_bin': BASE_BIN, 'configuration': CONFIG, 'recipe': RECIPE,
        'maximum_execution_seconds': 1200, 'numeric_processes': 1, 'native_head_processes': 1,
        'new_optimizer_updates': 96, 'training_bag_forwards': 672,
        'candidate_head_cases': 323, 'candidate_embedding_rows': 18982, 'candidate_ready_rows': 14137,
        'native_head_cases': 323, 'native_embedding_rows': 18982, 'native_ready_rows': 14137,
        'new_frontend_calls': 0, 'new_audio_decodes': 0, 'new_compilations': 0,
        'old_model_forward_calls': 0, 'new_tts_asr_calls': 0, 'new_audio_mixes': 0,
        'native_consumer': 'PINNED_PREVIOUS_COVERAGE_NATIVE_HEAD_REPLAY',
        'arm': 'NOT_RUN', 'new_full_pcm_chain': 'NOT_RUN', 'ssc305': 'NOT_RUN',
        'native_logit_atol': 0.0001, 'native_probability_atol': 0.000002,
        'native_clock_ready_event': 'EXACT', 'retry_count': 0,
        'independent_far': 'NOT_MEASURED', 'independent_frr': 'NOT_MEASURED',
        'regression_gates': ['all_291_prior_matched_intents_retained',
            'all_291_prior_correct_cases_retained', 'each_291_event_error_category_not_increased',
            'natural32_actual_events_remain_zero', 'source_balanced_reviewed_background_ce_decreased',
            'weights_changed_and_normalization_unchanged', 'native_saved_npz_bin_parity'],
        'checkpoint_selection': False, 'automatic_retrain_after_gate_failure': False,
        'trigger': 'public main exact successor push run_attempt=1',
        'scope': 'NINE_OWNER_REVIEWED_WINDOWS_ONLY_RESEARCH_INCREMENTAL_HEAD',
        **dict.fromkeys(FALSE_FLAGS, False)}


def source_preflight(root, plan, worker_path=None):
    root=Path(root); worker_path=root/WORKER_PATH if worker_path is None else Path(worker_path)
    pins=plan.get('source_pins', {})
    need(set(pins)==set(SOURCE_PINS)|{ADMISSION_PATH,REVIEW_PATH,PINS_PATH}, 'EXACT_SOURCE_PIN_SET')
    need(all(pins[k]==v for k,v in SOURCE_PINS.items()), 'FROZEN_CONSUMED_SOURCE_PINS')
    need(canonical(plan)==canonical(expected_plan(sha(regular(worker_path)),pins)), 'EXACT_FIXED_PLAN')
    for name,pin in pins.items():
        need(identity(regular(root/name))==pin, 'SOURCE_PIN:'+name)
    admission=loads(regular(root/ADMISSION_PATH));review=loads(regular(root/REVIEW_PATH))
    need(admission.get('schema')=='english-reviewed-candidate-source-admission-v1', 'ADMISSION_SCHEMA')
    need(len(admission.get('candidates', []))==9, 'EXACT_NINE_REVIEWED_WINDOWS')
    need(len({r['recording_id'] for r in admission['candidates']})==8, 'EXACT_EIGHT_REVIEWED_SOURCES')
    validate_owner_review(review,admission)
    need(admission['review_evidence']['owner_review_receipt']=={
        'repository_path':REVIEW_PATH,'identity':identity(regular(root/REVIEW_PATH))}, 'OWNER_REVIEW_RECEIPT_BINDING')
    return admission


def validate_owner_review(review,admission):
    need(review.get('schema')=='english-natural32-owner-negative-review-v1' and
         review.get('owner_statement_verbatim')=='没有唤醒词' and review.get('candidate_count')==9 and
         review.get('distinct_source_originals')==8 and review.get('reviewed_negative_samples')==488880 and
         review.get('full_original_utterance_labels_asserted') is False and
         review.get('positive_pronunciation_gold_created') is False and review.get('independent_test_admitted') is False,
         'EXACT_OWNER_REVIEW_SCOPE')
    owners={c['candidate_id']:c for c in review['candidates']}
    need(len(owners)==9 and set(owners)=={c['candidate_id'] for c in admission['candidates']},'REVIEW_ADMISSION_SAME_NINE')
    for c in admission['candidates']:
        o=owners[c['candidate_id']]
        need(o.get('review_decision')=='N' and o.get('trusted_negative') is True and
             o.get('research_training_admitted') is True and o.get('source_role')=='train' and
             c.get('owner_review_decision')=='N' and c.get('admitted_class_id')==0 and
             c.get('trusted_negative') is True and c.get('research_training_admitted') is True and
             c.get('whole_original_negative_admitted') is False and c.get('use_role')=='research_train_only',
             'REVIEWED_WINDOW_RESEARCH_NEGATIVE_ONLY')
        need(all(c[k]==o[k] for k in ('candidate_id','recording_id','start_sample','end_sample_exclusive',
             'source_pcm_sha256','window_pcm_sha256')), 'OWNER_EXACT_PCM_INTERVAL_IDENTITY')


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,Path(path))
    need(spec is not None and spec.loader is not None, 'PINNED_MODULE_SPEC')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def npy_member(raw, member):
    """Inspect dtype/shape and literal saved bytes with stdlib only."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        need(len(z.namelist())==len(set(z.namelist())), 'UNIQUE_NPZ_KEYS')
        b=z.read(member+'.npy')
    need(b[:6]==b'\x93NUMPY' and b[6:8] in (b'\x01\x00',b'\x02\x00'), 'NPY_HEADER')
    length=struct.unpack_from('<H' if b[6]==1 else '<I',b,8)[0]
    offset=10 if b[6]==1 else 12
    header=ast.literal_eval(b[offset:offset+length].decode('latin1'))
    need(type(header) is dict and set(header)=={'descr','fortran_order','shape'} and
         header['fortran_order'] is False and type(header['shape']) is tuple,
         'NPY_C_ORDER_HEADER')
    return header,b[offset+length:]


def verify_saved_npz_bin(npzraw, blob):
    shapes={'mean96':(96,),'scale96':(96,),'w1':(32,1536),'b1':(32,),'w2':(3,32),'b2':(3,)}
    with zipfile.ZipFile(io.BytesIO(npzraw)) as z:
        need(set(z.namelist())=={k+'.npy' for k in shapes}, 'SIX_HEAD_ARRAYS')
    arrays=[]
    for key,shape in shapes.items():
        h,b=npy_member(npzraw,key)
        need(h['descr']=='<f4' and h['shape']==shape and len(b)==4*math.prod(shape), 'HEAD_ARRAY:'+key)
        arrays.append(b)
    need(blob==b'EHKWS01\0'+b''.join(arrays), 'NPZ_BIN_EXACT_SIX_ARRAY_BYTES')
    return {key:identity(b) for key,b in zip(shapes,arrays)}


def choose_reviewed_inputs(candidates, roster, ledger):
    """Original continuous 16-embedding windows, only within an owner-reviewed PCM interval."""
    byid={r['recording_id']:r for r in roster['rows']}
    prior={c['candidate_id']:c for c in ledger['candidates']}
    need(len(candidates)==len(prior)==9 and {c['candidate_id'] for c in candidates}==set(prior), 'EXACT_REVIEW_SET')
    result=[]; bysource={}
    for c in candidates:
        cid=c['candidate_id']; old=prior[cid]; rid=c['recording_id']
        need(rid in byid and all(c[k]==old[k] for k in
             ('candidate_id','recording_id','window_pcm_sha256','start_sample','end_sample_exclusive')), 'REVIEW_WINDOW_BINDING')
        r=byid[rid]; start,end=c['start_sample'],c['end_sample_exclusive']
        need(type(start) is int and type(end) is int and 0<=start<end<=r['frames'], 'REVIEW_HALF_OPEN_BOUNDS')
        need(r['source_role']==r['old_bg48_role']==r['current_coverage_role']=='train', 'UNCHANGED_TRAIN_SOURCE_ROLE')
        ends=[t for t in range(35200,r['frames']+1,1280) if t-35200>=start and t<=end]
        need(ends and c['eligible_ready_samples']==ends, 'EXACT_COMPLETE_SUPPORT_MASK')
        need(c['source_pcm_sha256']==r['pcm_sha256'], 'ORIGINAL_PCM_IDENTITY')
        row={'candidate_id':cid,'recording_id':rid,'reviewed_start':start,'reviewed_end_exclusive':end,
             'head_ends':ends,'head_support_samples':35200,'input_count':len(ends),'target_class':0,
             'scope':'REVIEWED_WINDOW_ONLY','whole_utterance_label':'UNKNOWN_OUTSIDE_REVIEWED_WINDOWS'}
        result.append(row);bysource.setdefault(rid,set()).update(ends)
    source_rows=[{'recording_id':rid,'head_ends':sorted(ends),'input_count':len(ends)} for rid,ends in sorted(bysource.items())]
    need(len(source_rows)==8 and sum(x['input_count'] for x in source_rows)==134,
         'EXACT_134_INPUTS_EIGHT_SOURCES')
    return {'candidates':result,'sources':source_rows,'head_inputs':134,'deduplicate_key':['recording_id','sample'],
            'source_weighting':'EQUAL_SOURCE_THEN_MEAN_UNIQUE_WINDOWS','reviewed_seconds':30.555,
            'unreviewed_original_portions_admitted':False,'independent_negative_source_count':0}


def prepare_saved_inputs(root, quality_path, natural_path, admission, output=None):
    """All-authentication stdlib preparation, no numpy, decoding, compiler or model."""
    root=Path(root); pins=loads(regular(root/PINS_PATH))
    quality=Archive(quality_path,SOURCES['quality']);natural=Archive(natural_path,SOURCES['natural'])
    try:
        for name,pin in pins['coverage_members'].items():
            need(identity(quality.read(name))==pin, 'COVERAGE_MEMBER:'+name)
        need(identity(quality.read('models/coverage-head.npz'))==BASE_NPZ and
             identity(quality.read('models/coverage-head.bin'))==BASE_BIN, 'CURRENT_COVERAGE_WARMSTART')
        verify_saved_npz_bin(quality.read('models/coverage-head.npz'),quality.read('models/coverage-head.bin'))
        qr=quality.json('evidence/RESULT.json')
        need(qr['status']=='COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH' and qr['research_retention']['passed'] is True,
             'SAVED_COVERAGE_RETAINED_RESEARCH')
        manifest_raw=natural.read('MANIFEST.json'); manifest=loads(manifest_raw)
        need(sha(manifest_raw)==pins['natural_manifest_sha256'] and manifest['supervised_success'] is True and
             set(natural.z.namelist())==set(manifest['files'])|{'MANIFEST.json'}, 'STRICT_NATURAL_ARTIFACT_CLOSURE')
        for name,pin in manifest['files'].items():
            need(identity(natural.read(name))==pin, 'NATURAL_MEMBER:'+name)
        nr=natural.json('evidence/RESULT.json');supervisor=natural.json('evidence/SUPERVISOR_EXIT.json')
        need(nr.get('passed') is True and nr.get('candidate_count')==9 and nr.get('actual_event_count')==0 and
             nr.get('model_process_attempts')==32 and nr.get('completed_validated_traces')==32 and
             nr.get('candidate_head')==BASE_BIN and supervisor.get('passed') is True and
             str(nr.get('run_id'))==str(SOURCES['natural']['run_id']) and
             nr.get('head_sha')==SOURCES['natural']['head_sha'], 'COMPLETE_PRIOR_NATURAL_SCAN')
        raw_roster=regular(root/ROSTER_PATH)
        need(natural.read('source/'+ROSTER_PATH)==raw_roster, 'FROZEN_ROSTER_BYTES')
        roster=loads(raw_roster);ledger=natural.json('candidates/LEDGER.json')
        selected=choose_reviewed_inputs(admission['candidates'],roster,ledger)
        parser=load_module(root/FROZEN_PARSER,'increment_frozen_trace_parser')
        traces={};scores={}
        for row in roster['rows']:
            rid=row['recording_id'];raw=natural.read('traces/'+rid+'.trace')
            parsed=parser.parse_trace(raw,{'samples':row['frames'],'records':row['expected_embedding_rows'],
                                         'ready':row['expected_ready_rows']})
            need(parsed['events']==[], 'PRIOR_NATURAL_ZERO_EVENTS')
            traces[rid]=parsed;scores[rid]=parsed['scores']
        for c in admission['candidates']:
            rid=c['recording_id'];pcm=natural.read('selectedinputs/pcm/'+rid+'.pcm')
            need(sha(pcm)==c['source_pcm_sha256'] and len(pcm)==2*next(r['frames'] for r in roster['rows'] if r['recording_id']==rid), 'REVIEW_SOURCE_PCM_BYTES')
            clip=pcm[2*c['start_sample']:2*c['end_sample_exclusive']]
            need(sha(clip)==c['window_pcm_sha256'] and
                 clip==natural.read('candidates/pcm/'+c['candidate_id']+'.pcm') and
                 c['original_c_trace_sha256']==traces[rid]['trace_identity']['sha256'], 'REVIEW_CLIP_TRACE_BINDING')
            need([h['sample'] for h in c['eligible_head_inputs']]==c['eligible_ready_samples'], 'ADMITTED_ROWS_ORDER')
            for h in c['eligible_head_inputs']:
                end=h['sample'];last=(end-16000)//1280;first=last-15
                need(h['embedding_first_ordinal']==first and h['embedding_last_ordinal']==last and
                     h['first_embedding_sample']==end-15*1280 and h['last_embedding_sample']==end and
                     h['full_pcm_support_start']==end-35200 and h['full_pcm_support_end_exclusive']==end,
                     'ADMITTED_ROW_EXACT_CONTEXT_GEOMETRY')
                embedding_raw=b''.join(struct.pack('<96f',*r) for r in traces[rid]['embeddings'][first:last+1])
                need(len(embedding_raw)==h['raw_embedding_f32le_bytes']==6144 and
                     sha(embedding_raw)==h['raw_embedding_f32le_sha256'] and
                     sha(pcm[2*(end-35200):2*end])==h['full_pcm_support_sha256'], 'ADMITTED_RAW_INPUT_BYTES')
        if output is not None:
            output=Path(output)
            for name in pins['coverage_members']:
                put(output/'inputs/coverage'/name,quality.read(name))
            for name in ('MANIFEST.json','evidence/RESULT.json','evidence/SUPERVISOR_EXIT.json','candidates/LEDGER.json','candidates/CLIP_FILES.json'):
                put(output/'inputs/natural'/name,natural.read(name))
            for row in roster['rows']:
                rid=row['recording_id'];put(output/'inputs/natural/traces'/(rid+'.trace'),natural.read('traces/'+rid+'.trace'))
            for c in admission['candidates']:
                name='candidates/pcm/'+c['candidate_id']+'.pcm';put(output/'inputs/natural'/name,natural.read(name))
            once(output/'evidence/ADMITTED_HEAD_INPUTS.json',selected)
            once(output/'evidence/INPUTS_VERIFIED.json',{'status':'VERIFIED_FROZEN_INPUTS_NO_NUMERIC_CALLS',
                'sources':SOURCES,'base_npz':BASE_NPZ,'base_bin':BASE_BIN,'candidate_windows':9,'original_sources':8,
                'admitted_head_inputs':134,'natural_trace_count':32,'audio_decodes':0,'frontend_calls':0})
        return {'quality':quality,'natural':natural,'roster':roster,'ledger':ledger,'selection':selected,
                'traces':traces,'scores':scores,'pins':pins}
    except BaseException:
        quality.close();natural.close();raise


def paired_case_gate(old,new):
    need(len(old)==len(new)==291, 'ALL_291_REGRESSION_CASES')
    result=[]
    for a,b in zip(old,new):
        need(all(a[k]==b[k] for k in ('condition','id','role','weak_class','observed_samples','clip_start_sample','original_clip_eof_sample')),
             'PAIRED_REGRESSION_IDENTITY')
        increases=[]
        for k in ('prefix_events','wrong_keyword_after_clip_start'):
            if len(b[k])>len(a[k]):increases.append(k)
        for k in ('extra_same_keyword_events','weak_negative_intent_events'):
            if b[k]>a[k]:increases.append(k)
        result.append({'condition':a['condition'],'id':a['id'],'role':a['role'],'weak_class':a['weak_class'],
            'prior_matched_intent_lost':a['matched_intent_after_clip_start'] and not b['matched_intent_after_clip_start'],
            'prior_correct_case_regressed':not a['research_error'] and b['research_error'],
            'event_error_categories_increased':increases,'old_events':a['events'],'new_events':b['events']})
    return {'cases':result,'passed':all(not r['prior_matched_intent_lost'] and not r['prior_correct_case_regressed'] and
                                       not r['event_error_categories_increased'] for r in result)}


def fixed_draws(old_draws, selected_sources):
    need(len(old_draws)==96 and len(selected_sources)==8 and selected_sources==sorted(set(selected_sources)), 'FIXED_DRAW_GEOMETRY')
    draws=[]
    for step,rows in enumerate(old_draws):
        need(len(rows)==6 and all(r['role']=='train' and r['condition'] in ('clean','snr10','new5','new15') for r in rows),
             'ORIGINAL_SIX_TRAIN_ONLY_DRAWS')
        need([r['weak_class'] for r in rows]==[0,1,2,0,1,2] and
             all(r['condition']=='clean' for r in rows[:3]) and
             all(r['condition']==('snr10','new5','new15')[step%3] for r in rows[3:]), 'ORIGINAL_CONDITION_CLASS_ORDER')
        draws.append({'step':step,'original':rows,'reviewed_source':selected_sources[step%8]})
    return draws


def result_boundary(binding):
    return {'schema':SCHEMA,**binding,'status':'FAILED_NO_RETRY','passed':False,'retry_count':0,
        'new_audio_decodes':0,'new_frontend_calls':0,'new_tts_asr_calls':0,'new_compilations':0,
        'human_review_scope':'NINE_WINDOW_PCM_IDENTITIES_ONLY','whole_utterance_labels_changed':0,
        'new_independent_speaker_count':0,'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED',
        'full_pcm_new_head':'NOT_RUN','arm_new_head':'NOT_RUN','ssc305':'NOT_RUN',
        'strong_english_fsmn_ctc':False,'weak_original_positive_labels_preserved':True,
        'formal_quality_improvement_claim':False,**dict.fromkeys(FALSE_FLAGS,False)}


def checkpoint(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.pending')
    with temp.open('xb') as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def restore_streams(output, np):
    base=Path(output)/'inputs/coverage'
    clean=loads(regular(base/'sources/STREAM_ROSTER.json'))
    geometry=loads(regular(base/'evidence/TRAINING_GEOMETRY.json'))
    previous_training=loads(regular(base/'evidence/TRAINING.json'))
    allrows={};streams={};evaluation=[]
    with np.load(base/'cache/clean-original.npz',allow_pickle=False) as archive:
        need(set(archive.files)=={'embeddings','samples','discontinuity_before'}, 'ORIGINAL_CLEAN_CACHE_KEYS')
        matrix,times,gaps=archive['embeddings'].copy(),archive['samples'].copy(),archive['discontinuity_before'].copy()
    need(matrix.dtype==np.float32 and matrix.shape==(10354,96) and bool(np.isfinite(matrix).all()) and
         times.dtype==np.uint64 and times.shape==(10354,) and gaps.dtype==np.bool_ and gaps.shape==(10354,), 'ORIGINAL_CLEAN_GEOMETRY')
    cursor=0
    for row in clean:
        count=row['records'];need(row['first_record']==cursor,'CLEAN_CONTIGUOUS_RECORDS')
        local=[int(x)-row['offset'] for x in times[cursor:cursor+count]]
        need(local==list(range(16000,row['observed_samples']+1,1280)) and
             gaps[cursor:cursor+count].tolist()==[True]+[False]*(count-1), 'EXACT_CLEAN_LOCAL_CLOCK')
        key=('clean',row['id']);allrows[key]=row;streams[key]=(matrix[cursor:cursor+count].copy(),local)
        evaluation.append(key);cursor+=count
    need(cursor==10354 and len(clean)==117,'ALL_CLEAN_ROWS')
    conditions=[('snr10','snr10-roster.json','mixed-snr10.npz',76,2804),
                ('snr5','snr5-roster.json','snr5-embeddings.npz',71,2649),
                ('new8','new8-roster.json','new8-embeddings.npz',27,1001),
                (None,'train-new-roster.json','train-new-embeddings.npz',96,3544)]
    for fixed,roster_name,cache_name,count,total in conditions:
        rows=loads(regular(base/'cache'/roster_name));need(len(rows)==count,'FIXED_CONDITION_COUNT')
        with np.load(base/'cache'/cache_name,allow_pickle=False) as archive:
            need(set(archive.files)=={f'{kind}{i:03d}' for i in range(count) for kind in ('e','t')}, 'EXACT_CACHE_ARRAY_SET')
            observed=0
            for i,row in enumerate(rows):
                m,t=archive[f'e{i:03d}'].copy(),archive[f't{i:03d}'].copy()
                local=[int(x) for x in t];condition=fixed or row['condition'];key=(condition,row['id'])
                need(m.dtype==np.float32 and m.shape==(len(local),96) and bool(np.isfinite(m).all()) and
                     t.dtype==np.uint64 and t.ndim==1 and local==list(range(16000,row['observed_samples']+1,1280)),
                     'CONDITION_CAUSAL_GEOMETRY')
                need(key not in streams and row['role'] in ('train','development'), 'UNIQUE_ORIGINAL_CONDITION')
                if fixed is None:need(row['role']=='train' and condition in ('new5','new15'), 'NEW_CONDITION_TRAIN_ONLY')
                allrows[key]=row;streams[key]=(m,local);observed+=len(m)
                if fixed is not None:evaluation.append(key)
        need(observed==total,'EXACT_CONDITION_EMBEDDING_ROWS')
    need(len(evaluation)==291 and sum(len(streams[k][0]) for k in evaluation)==16808 and
         len(geometry)==225, 'COMPLETE_FROZEN_EVALUATION_AND_TRAIN_GEOMETRY')
    geometry_keys=set()
    for g in geometry:
        key=(g['condition'],g['id']);need(key in streams and key not in geometry_keys,'UNIQUE_TRAINING_BAG')
        row=allrows[key];ends=g['head_instance_ends'];times=streams[key][1]
        need(g['role']==row['role']=='train' and g['weak_class']==row['weak_class'] and
             g['condition'] in ('clean','snr10','new5','new15') and ends==sorted(set(ends)) and ends,
             'ORIGINAL_BAG_LABEL_ROLE_GEOMETRY')
        need(all(t in times and times.index(t)>=15 for t in ends),'ORIGINAL_FULL_EMBEDDING_HISTORY')
        geometry_keys.add(key)
    for draw in previous_training['draws']:
        need(all((r['condition'],r['id']) in geometry_keys for r in draw),'PREVIOUS_DRAW_MEMBERSHIP')
    return allrows,streams,evaluation,geometry,previous_training


def native_parity(raw, reference):
    lines=raw.decode('ascii').splitlines();need(len(lines)==len(reference)==18982,'NATIVE_COMPLETE_ROW_COUNT')
    maxlogit=maxprob=0.0;events=[]
    for line,r in zip(lines,reference):
        p=line.split();need(len(p)==9,'NATIVE_ROW_SHAPE')
        need((int(p[0]),int(p[1]),int(p[2]))==(r['sample'],int(r['ready']),r['keyword']), 'NATIVE_EXACT_CLOCK_READY_EVENT')
        values=[float(x) for x in p[3:]];need(all(math.isfinite(x) for x in values),'NATIVE_FINITE_VALUES')
        le=max(abs(a-b) for a,b in zip(values[:3],r['logits']));pe=max(abs(a-b) for a,b in zip(values[3:],r['probabilities']))
        maxlogit=max(maxlogit,le);maxprob=max(maxprob,pe)
        need(le<=1e-4 and pe<=2e-6,'NATIVE_SAVED_NPZ_BIN_TOLERANCE')
        if r['keyword']:events.append({'case_index':r['case_index'],'condition':r['condition'],'id':r['id'],
                                      'sample':r['local_sample'],'keyword':r['keyword']})
    return {'passed':True,'head_cases':323,'embedding_rows':18982,'ready_rows':sum(r['ready'] for r in reference),
        'max_abs_logit_error':maxlogit,'max_abs_probability_error':maxprob,'events':events,
        'exact_clock_ready_events':True,'frontend_calls':0,'scope':'NATIVE_HEAD_ONLY_SAVED_EMBEDDINGS',
        'arm':'NOT_RUN','full_pcm_chain':'NOT_RUN','ssc305':'NOT_RUN'}


def reviewed_fit(selection,old_scores,new_scores):
    byold={rid:{r['sample']:r for r in rows if r['ready']} for rid,rows in old_scores.items()}
    bynew={rid:{r['sample']:r for r in rows if r['ready']} for rid,rows in new_scores.items()}
    source_rows=[];candidate_rows=[]
    def ce(rows):return sum(-math.log(max(float(r['probabilities'][0]),1e-12)) for r in rows)/len(rows)
    for s in selection['sources']:
        rid=s['recording_id'];ends=s['head_ends']
        a=[byold[rid][t] for t in ends];b=[bynew[rid][t] for t in ends]
        source_rows.append({'recording_id':rid,'head_inputs':len(ends),'old_background_ce':ce(a),'new_background_ce':ce(b)})
    for c in selection['candidates']:
        rid=c['recording_id'];ends=c['head_ends'];allends=[t for t in byold[rid] if c['reviewed_start']<t<=c['reviewed_end_exclusive']]
        a=[byold[rid][t] for t in ends];b=[bynew[rid][t] for t in ends]
        row={'candidate_id':c['candidate_id'],'recording_id':rid,'fit_head_ends':ends,
             'complete_support_old_background_ce':ce(a),'complete_support_new_background_ce':ce(b),
             'reviewed_window_all_ready_endpoints':allends,'diagnostic_context':'ORIGINAL_STREAM_CONTEXT_AVAILABLE_IN_(START,END]_NOT_ADDITIONAL_TRAINING_LABELS','by_keyword':{}}
        for k in (1,2):
            series=[]
            for label,table in [('old',byold),('new',bynew)]:
                probs=[table[rid][t]['probabilities'][k] for t in allends]
                triples=[min(probs[i:i+3]) for i in range(len(probs)-2)]
                row['by_keyword'][str(k)+'_'+label]={'peak_probability':max(probs),
                    'at_or_above_proposal_0_5':sum(p>=0.5 for p in probs),
                    'max_three_consecutive_min_probability':max(triples) if triples else None,
                    'events':[t for t in allends if table[rid][t]['keyword']==k]}
                series.append(probs)
            row['by_keyword'][str(k)+'_paired_probabilities']=[{'sample':t,'old':x,'new':y} for t,x,y in zip(allends,*series)]
        candidate_rows.append(row)
    old=sum(r['old_background_ce'] for r in source_rows)/8;new=sum(r['new_background_ce'] for r in source_rows)/8
    return {'status':'TRAINING_FIT_DIAGNOSTIC_NOT_INDEPENDENT_EVALUATION','old_source_balanced_ce':old,
        'new_source_balanced_ce':new,'passed':new<old,'sources':source_rows,'candidates':candidate_rows,
        'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED','fit_inputs':134}


def numeric(args,plan):
    output=args.output.absolute();binding=execution_identity(plan)
    need(os.environ.get('KWS_INCREMENT_NUMERIC_PARENT_PID')==str(os.getppid()) and
         os.getpgrp()==os.getppid(), 'NUMERIC_ONLY_IN_SUPERVISED_WORKER_GROUP')
    worker=loads(regular(output/'evidence/WORKER_STARTED.json'))
    need(worker['pid']==os.getppid() and worker['binding']==binding,'NUMERIC_PARENT_BINDING')
    need(not (output/'evidence/NUMERIC_STARTED.json').exists(),'NO_NUMERIC_REPLAY')
    once(output/'evidence/NUMERIC_STARTED.json',{'pid':os.getpid(),'parent_pid':os.getppid(),
        'attempt':1,'retry_count':0,'binding':binding,'recipe':RECIPE})
    progress={'optimizer_updates_started':0,'optimizer_updates_completed':0,'training_bag_forwards_started':0,
        'training_bag_forwards_completed':0,'candidate_head_cases_started':0,'candidate_head_cases_completed':0,
        'candidate_embedding_rows':0,'candidate_ready_rows':0,'old_model_forward_calls':0,
        'new_frontend_calls':0,'new_audio_decodes':0}
    def ck():checkpoint(output/'evidence/NUMERIC_PROGRESS.json',progress)
    ck();losses=[];draws=[];outcome={'status':'NUMERIC_FAILED_NO_RETRY','passed':False}
    try:
        import numpy as np
        import torch
        need(np.__version__=='1.26.4' and torch.__version__=='2.9.1+cpu','EXACT_NUMERIC_RUNTIME')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261117);torch.use_deterministic_algorithms(True)
        once(output/'evidence/RUNTIME.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
            'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'frontend_runtime_imported':False})
        portable=load_module(output/'inputs/coverage/sources/portable_head.py','increment_portable_head')
        baseline=portable.load_weights(output/'inputs/coverage/models/coverage-head.npz')
        need(identity(portable.export_blob(baseline))==BASE_BIN,'LOADED_BASE_NPZ_BIN')
        allrows,streams,evaluation,geometry,previous_training=restore_streams(output,np)
        roster=loads(regular(output/'source'/ROSTER_PATH));selection=loads(regular(output/'evidence/ADMITTED_HEAD_INPUTS.json'))
        parser=load_module(output/'source'/FROZEN_PARSER,'increment_numeric_trace_parser')
        old_natural={};natural_order=[]
        for row in roster['rows']:
            rid=row['recording_id'];parsed=parser.parse_trace(regular(output/'inputs/natural/traces'/(rid+'.trace')),
                {'samples':row['frames'],'records':row['expected_embedding_rows'],'ready':row['expected_ready_rows']})
            key=('natural',rid);streams[key]=(np.asarray(parsed['embeddings'],dtype=np.float32),[r['sample'] for r in parsed['scores']])
            allrows[key]={'id':rid,'role':'train','weak_class':None,'observed_samples':row['frames'],'source':row}
            old_natural[rid]=parsed['scores'];natural_order.append(key)
        prepared={}
        def make_windows(key,ends):
            matrix,times=streams[key];positions={t:i for i,t in enumerate(times)}
            need(all(t in positions and positions[t]>=15 for t in ends),'COMPLETE_ORIGINAL_HISTORY')
            windows=np.asarray([matrix[positions[t]-15:positions[t]+1] for t in ends],dtype=np.float32)
            return torch.from_numpy(((windows-baseline['mean96'])/baseline['scale96']).reshape(len(windows),1536))
        for g in geometry:prepared[(g['condition'],g['id'])]=make_windows((g['condition'],g['id']),g['head_instance_ends'])
        for s in selection['sources']:prepared[('reviewed',s['recording_id'])]=make_windows(('natural',s['recording_id']),s['head_ends'])
        all_draws=fixed_draws(previous_training['draws'],[s['recording_id'] for s in selection['sources']])
        once(output/'evidence/TRAINING_DRAWS.json',{'draws':all_draws,'recipe':RECIPE,'source_order':[s['recording_id'] for s in selection['sources']]})
        once(output/'evidence/TRAINING_GEOMETRY.json',{'original_bags':geometry,'reviewed':selection,
            'original_positive_label_kind':'WEAK_CLIP_INTENT','all_original_roles_unchanged':True})
        model=torch.nn.Sequential(torch.nn.Linear(1536,32),torch.nn.ReLU(),torch.nn.Linear(32,3))
        need(sum(p.numel() for p in model.parameters())==49283,'UNCHANGED_HEAD_PARAMETER_COUNT')
        with torch.no_grad():
            for parameter,key in ((model[0].weight,'w1'),(model[0].bias,'b1'),(model[2].weight,'w2'),(model[2].bias,'b2')):
                parameter.copy_(torch.from_numpy(baseline[key]))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.0001)
        need(len(optimizer.state)==0,'ADAMW_STATE_RESET')
        for draw in all_draws:
            need(progress['optimizer_updates_started']<96,'ONE_FIXED_TRAINING_BUDGET')
            progress['optimizer_updates_started']+=1;ck();optimizer.zero_grad(set_to_none=True);terms=[]
            uses=[((r['condition'],r['id']),r['weak_class']) for r in draw['original']]
            uses.append((('reviewed',draw['reviewed_source']),0))
            for key,label in uses:
                need(progress['training_bag_forwards_started']<672,'EXACT_SEVEN_BAG_BUDGET')
                progress['training_bag_forwards_started']+=1;ck()
                logp=torch.log_softmax(model(prepared[key]),dim=-1)
                progress['training_bag_forwards_completed']+=1;ck()
                terms.append(-logp[:,0].mean() if label==0 else math.log(len(logp))-torch.logsumexp(logp[:,label],dim=0))
            loss=torch.stack(terms).mean();need(bool(torch.isfinite(loss)),'FINITE_INCREMENT_LOSS')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()));draws.append(draw);progress['optimizer_updates_completed']+=1;ck()
        weights={'w1':model[0].weight.detach().numpy(),'b1':model[0].bias.detach().numpy(),
            'w2':model[2].weight.detach().numpy(),'b2':model[2].bias.detach().numpy(),
            'mean96':baseline['mean96'].copy(),'scale96':baseline['scale96'].copy()}
        (output/'models').mkdir(exist_ok=True)
        np.savez(output/'models/incremental-head.npz',**weights)
        candidate=portable.load_weights(output/'models/incremental-head.npz')
        binary=portable.export_blob(candidate);put(output/'models/incremental-head.bin',binary)
        array_pins=verify_saved_npz_bin(regular(output/'models/incremental-head.npz'),binary)
        need(all(candidate[k].tobytes()==baseline[k].tobytes() for k in ('mean96','scale96')),'NORMALIZATION_BYTE_IDENTICAL')
        need(any(candidate[k].tobytes()!=baseline[k].tobytes() for k in ('w1','b1','w2','b2')),'ACTUAL_UPDATED_WEIGHTS')
        frozen={'npz':file_identity(output/'models/incremental-head.npz'),'bin':identity(binary),'arrays':array_pins}
        once(output/'evidence/CANDIDATE_FROZEN.json',{**frozen,'progress':dict(progress),
            'evaluation_model_calls_before_freeze':0,'checkpoint_selection':False,'all_later_evaluation_uses_frozen_weights':True})
        once(output/'evidence/TRAINING.json',{'recipe':RECIPE,'base_npz':BASE_NPZ,'base_bin':BASE_BIN,
            'candidate':frozen,'optimizer_state_reset':True,'optimizer_updates':96,'training_bag_forwards':672,
            'losses':losses,'draws':draws,'normalization_byte_identical':True,'original_roles_unchanged':True,
            'original_positive_labels':'WEAK','snr5_new8_training_used':False})
        old_metrics=[];new_metrics=[];new_natural={};reference=[];case_map=[];offset=0
        for condition in ('clean','snr10','snr5','new8'):
            old_metrics.extend([dict(r,condition=condition) for r in loads(regular(output/'inputs/coverage/predictions'/('coverage-'+condition+'-events.json')))])
        with (output/'predictions/candidate-frames.jsonl').open('x') as prediction, (output/'runtime/candidate-head-stream.bin').open('xb') as stream:
            stream.write(b'EHSTR01\0'+struct.pack('<I',18982))
            for case_index,key in enumerate(evaluation+natural_order):
                condition,rid=key;row=allrows[key];matrix,times=streams[key]
                need(file_identity(output/'models/incremental-head.npz')==frozen['npz'],'FROZEN_BEFORE_EACH_EVAL')
                need(progress['candidate_head_cases_started']<323,'FIXED_CANDIDATE_CASE_BUDGET')
                progress['candidate_head_cases_started']+=1;ck()
                scores=portable.reference(candidate,matrix,times)
                events=[{'sample':r['sample'],'keyword':r['keyword']} for r in scores if r['keyword']]
                if condition=='natural':new_natural[rid]=scores
                else:new_metrics.append(dict(summarize_case(row,events),condition=condition))
                for ordinal,r in enumerate(scores):
                    prediction.write(json.dumps({'condition':condition,'id':rid,**r},separators=(',',':'),allow_nan=False)+'\n')
                    stream.write(struct.pack('<QI',offset+r['sample'],int(ordinal==0)))
                    stream.write(matrix[ordinal].astype('<f4',copy=False).tobytes())
                    reference.append(dict(r,case_index=case_index,condition=condition,id=rid,
                                          local_sample=r['sample'],sample=offset+r['sample']))
                case_map.append({'case_index':case_index,'condition':condition,'id':rid,'offset':offset,
                    'observed_samples':row['observed_samples'],'embedding_rows':len(scores),'ready_rows':sum(r['ready'] for r in scores)})
                offset+=row['observed_samples'];progress['candidate_head_cases_completed']+=1
                progress['candidate_embedding_rows']+=len(scores);progress['candidate_ready_rows']+=sum(r['ready'] for r in scores);ck()
        need(progress['candidate_head_cases_completed']==323 and progress['candidate_embedding_rows']==18982 and
             progress['candidate_ready_rows']==14137,'ALL_FIXED_CANDIDATE_REPLAYS_COMPLETE')
        once(output/'runtime/CASE_MAP.json',case_map)
        with (output/'runtime/reference-frames.jsonl').open('x') as f:
            for row in reference:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        paired=paired_case_gate(old_metrics,new_metrics);fit=reviewed_fit(selection,old_natural,new_natural)
        natural_events=[{'id':rid,'sample':r['sample'],'keyword':r['keyword']} for rid,rows in new_natural.items() for r in rows if r['keyword']]
        once(output/'predictions/candidate-events.json',new_metrics)
        once(output/'evidence/PAIRED_REGRESSION.json',paired)
        once(output/'evidence/REVIEWED_FIT.json',fit)
        once(output/'evidence/NATURAL32_PAIRED.json',{'old_event_count':0,'new_event_count':len(natural_events),'new_events':natural_events,
            'whole_originals_labeled_negative':False,'reviewed_sources':8,'other_sources':24,
            'other_source_labels':'UNKNOWN','independent_far':'NOT_MEASURED'})
        outcome={'status':'FIXED_TRAINING_AND_SAVED_CANDIDATE_REPLAYS_COMPLETE','passed':True,
            'candidate':frozen,'progress':progress,'research_gate_before_native':paired['passed'] and fit['passed'] and not natural_events,
            'gates':{'all_291_case_regression':paired['passed'],'natural32_zero_events':not natural_events,
                     'source_balanced_reviewed_ce_decreased':fit['passed'],'actual_weights_changed':True,
                     'normalization_byte_identical':True},'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED'}
    except BaseException as exc:
        outcome['error']=type(exc).__name__+':'+str(exc)
        put(output/'evidence/NUMERIC_TRACEBACK.txt',traceback.format_exc().encode())
    finally:
        once(output/'evidence/TRAINING_PARTIAL.json',{'progress':progress,'losses':losses,'draws':draws,'retry_count':0})
        once(output/'evidence/NUMERIC_RESULT.json',outcome)
    return 0 if outcome['passed'] else 1


def prepare_dependencies(runner):
    tools={n:shutil.which(n) for n in ('gh','curl')}
    once(runner.output/'evidence/DEPENDENCIES_BEFORE.json',{'tools':tools,'python':sys.version,
        'new_model_calls':0,'retry_count':0,'compiler_or_decoder_required':False})
    need(all(tools.values()) and sys.version_info[:2]==(3,12),'BASE_TOOLS_AND_PYTHON312')
    runner.command('gh-version',['gh','--version'],timeout=15)
    runner.command('curl-version',['curl','--version'],timeout=15)
    common=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--retries','0','--timeout','30','--only-binary=:all:']
    runner.command('install-numpy',common+['numpy==1.26.4'],timeout=120)
    runner.command('install-torch',common+['--index-url','https://download.pytorch.org/whl/cpu','torch==2.9.1+cpu'],timeout=360)
    runner.command('numeric-runtime-version',[sys.executable,'-I','-B','-c',
        "import numpy,torch; assert numpy.__version__=='1.26.4' and torch.__version__=='2.9.1+cpu'; print(numpy.__version__); print(torch.__version__)"],timeout=60)
    once(runner.output/'evidence/DEPENDENCIES_READY.json',{'numpy':'1.26.4','torch':'2.9.1+cpu',
        'pip_retries':0,'compiler_or_decoder_installed':False,'native_consumer':'REUSE_PINNED_BINARY'})


def child(args,plan,admission):
    output=args.output.absolute();binding=execution_identity(plan)
    need(os.environ.get('KWS_INCREMENT_PARENT_PID')==str(os.getppid()) and os.getpgrp()==os.getpid(),
         'ONLY_SUPERVISED_PUBLIC_WORKER')
    started=loads(regular(output/'STARTED.json'))
    need(started['binding']==binding and started['worker_sha256']==plan['worker_sha256'],'CHILD_SOURCE_BINDING')
    runner=Runner(output,started['monotonic_deadline']);result=result_boundary(binding);prepared=None
    once(output/'evidence/WORKER_STARTED.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'binding':binding,'retry_count':0})
    try:
        for name in ('source','inputs','cache','predictions','runtime','models','downloads'):(output/name).mkdir()
        with runner.stage('source-binding'):
            run=loads(regular(runner.command('current-run',['gh','api','repos/'+REPO+'/actions/runs/'+binding['run_id']],auth=True)))
            need(run.get('id')==int(binding['run_id']) and run.get('head_sha')==binding['head_sha'] and
                 run.get('run_attempt')==1 and run.get('event')=='push' and run.get('head_branch')=='main' and
                 run.get('repository',{}).get('private') is False and run.get('repository',{}).get('full_name')==REPO,
                 'CURRENT_PUBLIC_RUN_BINDING')
            for name in plan['source_pins']:put(output/'source'/name,regular(ROOT/name))
            for name in (WORKER_PATH,PLAN_PATH):put(output/'source'/name,regular(ROOT/name))
            once(output/'source/CONFIG.json',CONFIG);once(output/'source/RECIPE.json',RECIPE)
        with runner.stage('authenticate-saved-inputs'):
            quality=runner.download('quality');natural=runner.download('natural')
            prepared=prepare_saved_inputs(ROOT,quality,natural,admission,output)
        with runner.stage('numeric-runtime-preparation'):
            prepare_dependencies(runner)
        with runner.stage('fixed-incremental-training-and-regression'):
            os.environ['KWS_INCREMENT_NUMERIC_PARENT_PID']=str(os.getpid())
            runner.command('incremental-numeric',[sys.executable,'-I','-B',str(Path(__file__).resolve()),
                '--_numeric','--output',str(output)],timeout=600,kind='numeric')
            numeric_result=loads(regular(output/'evidence/NUMERIC_RESULT.json'))
            need(numeric_result['passed'] is True,'NUMERIC_COMPLETED_RESULT')
        with runner.stage('native-saved-binary-head-parity'):
            raw=regular(output/'inputs/coverage/runtime/head-native')
            need(raw[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',raw,18)[0]==62,'PINNED_NATIVE_X86_64_ELF')
            consumer=output/'runtime/head-native';put(consumer,raw);consumer.chmod(0o500)
            npz=regular(output/'models/incremental-head.npz');binary=regular(output/'models/incremental-head.bin')
            verify_saved_npz_bin(npz,binary)
            stdout=runner.command('head-native',[str(consumer),str(output/'models/incremental-head.bin'),
                str(output/'runtime/candidate-head-stream.bin')],timeout=180,kind='native')
            reference=[loads(line) for line in regular(output/'runtime/reference-frames.jsonl').splitlines()]
            parity=native_parity(regular(stdout),reference)
            need(regular(consumer)==raw and regular(output/'models/incremental-head.npz')==npz and
                 regular(output/'models/incremental-head.bin')==binary,'NATIVE_INPUTS_UNCHANGED')
            once(output/'evidence/HEAD_NATIVE_PARITY.json',{**parity,'consumer':identity(raw),
                 'candidate_bin':identity(binary),'candidate_npz':identity(npz),'command_receipt':'head-native.exit.json'})
        need(runner.counts=={'numeric':1,'native':1},'EXACT_ONE_TRAINING_ONE_NATIVE_EXECUTION')
        gates={**numeric_result['gates'],'native_saved_npz_bin_parity':True}
        passed=all(gates.values())
        result.update(status='RETAINED_RESEARCH_INCREMENTAL_HEAD' if passed else 'REJECTED_RESEARCH_REGRESSION_GATE',
            passed=passed,numerical_execution_completed=True,weights_retained=True,candidate=numeric_result['candidate'],
            progress=numeric_result['progress'],gates=gates,optimizer_updates=96,training_bag_forwards=672,
            research_retention_allowed=passed,current_default_model_replaced=False,
            owner_reviewed_negative_windows=9,owner_reviewed_negative_seconds=30.555,
            admitted_full_context_head_inputs=134,owner_reviewed_original_sources=8,
            reviewed_source_total_equal_weight=True,old_model_forward_calls=0)
    except BaseException as exc:
        result['error']=type(exc).__name__+':'+str(exc)
        put(output/'evidence/FAILURE_TRACEBACK.txt',traceback.format_exc().encode())
        if (output/'models/incremental-head.npz').exists():
            result['retained_npz']=file_identity(output/'models/incremental-head.npz')
        if (output/'models/incremental-head.bin').exists():
            result['retained_bin']=file_identity(output/'models/incremental-head.bin')
    finally:
        if prepared is not None:prepared['quality'].close();prepared['natural'].close()
        result['process_attempts']=runner.counts
        once(output/'evidence/RESULT.json',result)
    return 0 if result['passed'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    mode.add_argument('--_numeric',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path);args=parser.parse_args(argv)
    plan=loads(regular(ROOT/PLAN_PATH));admission=source_preflight(ROOT,plan)
    if args.execute or args._worker or args._numeric:
        need(args.output is not None,'EXPLICIT_EXECUTION_OUTPUT')
        if args._numeric:return numeric(args,plan)
        return child(args,plan,admission) if args._worker else supervise(args,plan)
    need(args.output is None,'SOURCE_ONLY_HAS_NO_EXECUTION_OUTPUT')
    print(canonical({'status':'SOURCE_CONTRACT_ONLY','new_model_calls':0,'new_audio_decodes':0,
        'new_network_calls':0,'planned_updates':96,'planned_reviewed_head_inputs':134,
        'candidate_cases':323,'native_processes':1,'maximum_seconds':1200,
        'worker_sha256':plan['worker_sha256']}).decode(),end='')
    return 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
