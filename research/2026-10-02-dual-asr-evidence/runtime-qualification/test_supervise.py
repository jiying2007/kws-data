#!/usr/bin/env python3
"""Dependency-free, independent subprocess tests; every fixture is cleaned up."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def live(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1]
        return value.split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    work = Path(tempfile.mkdtemp(prefix='supervisor-regression-', dir=ROOT))
    fixture = work / 'fixture.py'
    fixture.write_text('''import json,os,pathlib,signal,subprocess,sys,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
p=subprocess.Popen([sys.executable,'-I','-S','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(90)'])
pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]))
time.sleep(90)
''')
    cases = []
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        pidfile = work / (sig.name + '.pids.json')
        receipt = work / (sig.name + '.json')
        process = subprocess.Popen([sys.executable, str(ROOT / 'supervise.py'),
            '--receipt', str(receipt), '--seconds', '60', '--',
            sys.executable, '-I', '-S', str(fixture), str(pidfile)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pids = []
        try:
            deadline = time.monotonic() + 8
            while not pidfile.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(process.communicate()[0])
                time.sleep(.05)
            pids = json.loads(pidfile.read_text())
            process.send_signal(sig)
            process.communicate(timeout=12)
            result = json.loads(receipt.read_text())
            assert process.returncode == 1
            assert result['stop_reason'] == 'signal_' + sig.name
            assert not result['cleanup_errors']
            assert not any(live(pid) for pid in pids)
            assert not result['remaining_live_process_group_members']
            cases.append({'name': sig.name, 'passed': True, 'receipt': str(receipt)})
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for pid in pids:
                if live(pid):
                    os.kill(pid, signal.SIGKILL)
    for name, options, code, reason in (
        ('positive', [], 'print("positive test")', None),
        ('deadline', ['--seconds', '1'], 'import time;time.sleep(90)', 'wall_deadline'),
        ('rss', ['--max-rss-mib', '16'], 'import time; x=bytearray(64*1024*1024);time.sleep(90)', 'sampled_process_group_rss_limit'),
    ):
        receipt = work / (name + '.json')
        process = subprocess.run([sys.executable, str(ROOT / 'supervise.py'),
            '--receipt', str(receipt), *options, '--', sys.executable, '-I', '-S', '-c', code],
            capture_output=True, text=True, timeout=12)
        result = json.loads(receipt.read_text())
        assert result['stop_reason'] == reason, result
        assert process.returncode == (0 if reason is None else 1), result
        assert not result['cleanup_errors'] and not result['remaining_live_process_group_members']
        cases.append({'name': name, 'passed': True, 'receipt': str(receipt)})
    summary = {'status': 'passed', 'cases': cases,
        'supervisor_sha256': hashlib.sha256((ROOT / 'supervise.py').read_bytes()).hexdigest(),
        'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
