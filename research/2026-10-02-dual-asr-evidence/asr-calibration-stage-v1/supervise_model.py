#!/usr/bin/env python3
"""Sampled process-group guard for local serial model experiments, not a cgroup."""
import argparse
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import time

GIB = 1024 ** 3


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('available memory measurement unavailable')


def group_usage(pgid):
    rss, threads, members = 0, 0, []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) != pgid:
                continue
            status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            rss += int(status.get('VmRSS', '0 kB').split()[0]) * 1024
            threads += int(status.get('Threads', '0'))
            members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, threads, members


def live_group_members(pgid):
    """Zombies have exited and consume no workload resources; list live members."""
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--max-rss-mib', type=int, default=6656)
    parser.add_argument('--inherit-platform-proxy', action='store_true',
                        help='Preserve only proxy route environment for reviewed input downloads')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 1 <= args.seconds <= 7200 or not 16 <= args.max_rss_mib <= 7168:
        raise SystemExit('invalid bounded command request')
    root = Path(__file__).resolve().parent
    receipt_path = Path(args.receipt)
    log = receipt_path.with_suffix('.log')
    if any(path.exists() or path.is_symlink() for path in (receipt_path, log)):
        raise SystemExit('receipt/log destination is not fresh')
    initial_available = available_memory()
    if initial_available < 2 * GIB:
        raise SystemExit('less than 2 GiB host headroom before start')
    cpus = sorted(os.sched_getaffinity(0))[:2]
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
           'HOME': str(root / 'home'), 'USER': 'agent', 'LOGNAME': 'agent',
           'TMPDIR': str(root / 'tmp'), 'XDG_CACHE_HOME': str(root / 'cache'),
           'TORCHINDUCTOR_CACHE_DIR': str(root / 'cache/torchinductor'),
           'HF_HOME': str(root / 'cache/huggingface'), 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'NUMBA_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PIP_CONFIG_FILE': '/dev/null', 'PIP_NO_INDEX': '1',
           'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'SOURCE_DATE_EPOCH': '1704067200', 'PYTHONHASHSEED': '0'}
    if args.inherit_platform_proxy:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                     'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            if name in os.environ:
                env[name] = os.environ[name]
    for name in ('home', 'tmp', 'cache', 'cache/torchinductor', 'cache/huggingface'):
        (root / name).mkdir(parents=True, exist_ok=True)

    def prepare_child():
        os.sched_setaffinity(0, cpus)
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (16 * GIB, 16 * GIB))
        resource.setrlimit(resource.RLIMIT_CPU, (args.seconds * 2, args.seconds * 2 + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * GIB, 4 * GIB))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum

    previous_handlers = {signum: signal.signal(signum, request_stop)
                         for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    start = time.monotonic()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, preexec_fn=prepare_child)
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output, total_output, peak_rss, peak_threads = bytearray(), 0, 0, 0
    minimum_available, reason = initial_available, None
    cleanup_errors = []

    def signal_group(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except OSError as exc:
            cleanup_errors.append('signal_' + str(signum) + ': ' + str(exc))

    def wait_child(seconds):
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            return False
        except OSError as exc:
            cleanup_errors.append('wait: ' + str(exc))
            return False
        return True

    try:
        while True:
            rss, threads, members = group_usage(process.pid)
            available = available_memory()
            peak_rss = max(peak_rss, rss)
            peak_threads = max(peak_threads, threads)
            minimum_available = min(minimum_available, available)
            elapsed = time.monotonic() - start
            if stop_signal is not None:
                reason = 'signal_' + signal.Signals(stop_signal).name
            elif rss > args.max_rss_mib * 1024 ** 2:
                reason = 'sampled_process_group_rss_limit'
            elif available < 2 * GIB:
                reason = 'host_available_memory_floor'
            elif threads > 128:
                reason = 'sampled_process_group_thread_limit'
            elif elapsed > args.seconds:
                reason = 'wall_deadline'
            if reason:
                break
            for key, _ in selector.select(0.2):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total_output += len(chunk)
                output.extend(chunk)
                if len(output) > 4 * 1024 ** 2:
                    del output[:len(output) - 4 * 1024 ** 2]
            if process.poll() is not None:
                break
    except BaseException:
        reason = 'supervisor_error_or_interrupt'
        raise
    finally:
        if process.poll() is None:
            signal_group(signal.SIGTERM)
            if not wait_child(3):
                signal_group(signal.SIGKILL)
                if not wait_child(3):
                    cleanup_errors.append('child_exit_not_confirmed')
        # The group is dedicated to this launched command, never ambient work.
        signal_group(signal.SIGKILL)
        cleanup_deadline = time.monotonic() + 3
        remaining_members = live_group_members(process.pid)
        while remaining_members and time.monotonic() < cleanup_deadline:
            time.sleep(0.05)
            remaining_members = live_group_members(process.pid)
        if remaining_members:
            cleanup_errors.append('live_process_group_members_remain')
        for _ in range(100):
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            total_output += len(chunk)
            output.extend(chunk)
            if len(output) > 4 * 1024 ** 2:
                del output[:len(output) - 4 * 1024 ** 2]
        selector.close()
        process.stdout.close()
        with log.open('xb') as destination:
            destination.write(output)
        record = {'command': command, 'returncode': process.returncode, 'stop_reason': reason,
                  'cleanup_errors': cleanup_errors,
                  'remaining_live_process_group_members': remaining_members,
                  'platform_proxy_route_inherited': args.inherit_platform_proxy,
                  'wall_seconds': time.monotonic() - start, 'cpu_affinity': cpus,
                  'per_process_address_space_limit_bytes': 16 * GIB,
                  'sampled_process_group_rss_peak_bytes': peak_rss,
                  'sampled_process_group_threads_peak': peak_threads,
                  'host_available_memory_minimum_bytes': minimum_available,
                  'output_bytes': total_output, 'output_tail_truncated': total_output > len(output),
                  'log_path': str(log), 'status': 'pass' if process.returncode == 0 and reason is None and not cleanup_errors else 'failed',
                  'limitations': 'Sampled RSS/headroom/process-group guard and per-process limits; not aggregate cgroup memory enforcement or network isolation.'}
        with receipt_path.open('x') as destination:
            destination.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
        print(json.dumps(record), flush=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0 if process.returncode == 0 and reason is None and not cleanup_errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
