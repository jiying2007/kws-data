"""Public same-repository PR initialization audit, with zero audio forwards.

Use the production generator's exact _load_native path once. Never call its
returned forward, generate(), or _run_rows(). A failed strict readback remains
failed and retains all available per-component diagnostics. This command does
not change the 48 requests, permit training, or consume a TTS request.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time

_spec = importlib.util.spec_from_file_location('kokoro48_initialization_generator',
    Path(__file__).with_name('english_kokoro48_generate.py'))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)
SCHEMA = 'english-kokoro48-initialization-v1'
RECIPE_SHA256 = 'e104a64717ea195b5e7544285e07fa66441be24f3d7cdc9ea8493050cd715a83'
COMPONENTS = ('bert', 'bert_encoder', 'decoder', 'predictor', 'text_encoder')


class InitializationInterrupted(BaseException):
    """Abort through readback's Exception handler without retrying a component."""


def read_diagnostic_packet(output):
    path = output/'source/weight-diagnostics/SUMMARY.json'
    if not path.is_file():
        return None, False
    summary = json.loads(g.regular(path, 2 * 1024 * 1024))
    g.need(type(summary) is dict and summary.get('schema') == g.SCHEMA + '/checkpoint-readback',
        'CHECKPOINT_DIAGNOSTIC_SUMMARY_SCHEMA')
    records = summary.get('component_records', [])
    complete = (summary.get('component_count') == len(COMPONENTS)
        and type(records) is list and len(records) == len(COMPONENTS)
        and [r.get('component') for r in records] == list(COMPONENTS))
    if complete:
        for record in records:
            relative = 'source/weight-diagnostics/' + record['component'] + '.json'
            g.need(record.get('path') == relative, 'EXACT_COMPONENT_DIAGNOSTIC_PATH')
            raw = g.regular(g.within(output, relative), 2 * 1024 * 1024)
            g.need(len(raw) == record.get('bytes') and g.sha(raw) == record.get('sha256'),
                'EXACT_COMPONENT_DIAGNOSTIC_BYTES')
            component = json.loads(raw)
            g.need(component.get('schema') == g.SCHEMA + '/checkpoint-component'
                and component.get('component') == record['component']
                and type(component.get('passed')) is bool
                and component['passed'] == record.get('passed'), 'COMPONENT_DIAGNOSTIC_IDENTITY')
    return summary, complete


def allowed_event(env, event):
    g.need(env.get('GITHUB_ACTIONS') == 'true'
        and env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data'
        and env.get('GITHUB_EVENT_NAME') == 'pull_request'
        and env.get('GITHUB_RUN_ATTEMPT') == '1'
        and event.get('repository', {}).get('full_name') == 'jiying2007/kws-data'
        and event.get('repository', {}).get('private') is False
        and event.get('pull_request', {}).get('head', {}).get('repo', {}).get('full_name') == 'jiying2007/kws-data',
        'PUBLIC_SAME_REPOSITORY_FIRST_PR_INITIALIZATION_ONLY')
    g.need(re.fullmatch(r'[0-9a-f]{40}', env.get('GITHUB_SHA', '')) is not None
        and env.get('GITHUB_RUN_ID', '').isdigit(), 'ACTUAL_PUBLIC_PR_RUN_IDENTITY')


def run(output):
    env = dict(os.environ)
    event = json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes())
    allowed_event(env, event)
    g.need(g.recipe.digest(g.recipe.recipe()) == RECIPE_SHA256, 'UNCHANGED_RUNTIME_RECIPE_AND_48_REQUESTS')
    output = Path(output).absolute()
    g.need(not output.exists() and output.parent.is_dir()
        and not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_INITIALIZATION_RECEIPT_DIRECTORY')
    output.mkdir(mode=0o700)
    (output/'source').mkdir(mode=0o700)
    started = time.monotonic()
    binding = {'repository': env['GITHUB_REPOSITORY'], 'github_sha': env['GITHUB_SHA'],
        'run_id': env['GITHUB_RUN_ID'], 'run_attempt': env['GITHUB_RUN_ATTEMPT'],
        'event': env['GITHUB_EVENT_NAME'], 'pull_request_head_sha': event['pull_request']['head'].get('sha')}
    g.once(output/'STARTED.json', {'schema': SCHEMA, 'status': 'INITIALIZATION_ONLY_STARTED',
        'binding': binding, 'recipe_sha256': RECIPE_SHA256, 'deadline_seconds': 600,
        'maximum_constructor_calls': 1, 'model_forward_calls': 0,
        'tts_request_attempts': 0, 'training_allowed': False})
    private, failure, cleanup_failure = None, None, None
    def interrupted(signum, frame):
        raise InitializationInterrupted('INITIALIZATION_ONLY_DEADLINE' if signum == signal.SIGALRM
            else 'INITIALIZATION_ONLY_SIGTERM')
    previous_handlers = {number: signal.signal(number, interrupted)
        for number in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(600)
    try:
        private = Path(tempfile.mkdtemp(prefix='kokoro48-initialization-assets-', dir=output.parent))
        unused_forward, unused_vocab = g._load_native(output, private)
        # No phoneme input is passed to this closure. Dropping it releases the
        # successfully constructed model when the audit process exits.
        del unused_forward, unused_vocab
        lock = json.loads(g.regular(output/'source/MODEL_LOCK.json', 2 * 1024 * 1024))
        proof, complete = read_diagnostic_packet(output)
        g.need(lock.get('all_component_weights_verified') is True
            and lock.get('weight_diagnostics') == 'source/weight-diagnostics/SUMMARY.json'
            and lock.get('weight_diagnostics_sha256') == g.sha(g.regular(output/'source/weight-diagnostics/SUMMARY.json', 2 * 1024 * 1024))
            and complete
            and proof.get('passed') is True and proof.get('component_count') == 5
            and proof.get('passed_components') == 5 and proof.get('failed_components') == []
            and proof.get('allowed_key_exceptions') == [], 'EXACT_PRODUCTION_MODEL_PROOF_REQUIRED')
        g.need((output/'source/MODEL_LOAD_STARTED.json').is_file()
            and (output/'source/MODEL_CONSTRUCTED.json').is_file(), 'ONE_RETURNED_CONSTRUCTOR_RECEIPT')
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'code': str(exc)[:300]}
    finally:
        signal.alarm(0)
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        if private is not None:
            try:
                shutil.rmtree(private)
            except Exception as exc:
                cleanup_failure = {'type': type(exc).__name__, 'code': str(exc)[:300]}
        files = [p for p in output.rglob('*') if p.is_file()]
        audio = [p.relative_to(output).as_posix() for p in files if p.suffix.lower() in ('.wav', '.pcm', '.f32')]
        if audio and failure is None:
            failure = {'type': 'ValueError', 'code': 'INITIALIZATION_MUST_NOT_CREATE_AUDIO'}
        model_started = (output/'source/MODEL_LOAD_STARTED.json').is_file()
        model_returned = (output/'source/MODEL_CONSTRUCTED.json').is_file()
        summary, diagnostic_packet_complete, summary_read_failure = None, False, None
        try:
            summary, diagnostic_packet_complete = read_diagnostic_packet(output)
        except Exception as exc:
            summary_read_failure = {'type': type(exc).__name__, 'code': str(exc)[:300]}
        final_proof_passed = (diagnostic_packet_complete and summary is not None
            and summary.get('passed') is True and summary.get('passed_components') == 5
            and summary.get('failed_components') == [] and summary.get('allowed_key_exceptions') == [])
        if failure is None and summary_read_failure is None and not final_proof_passed:
            failure = {'type': 'ValueError', 'code': 'FINAL_DIAGNOSTIC_PACKET_NOT_VERIFIED'}
        succeeded = failure is None and cleanup_failure is None and summary_read_failure is None and final_proof_passed
        result = {'schema': SCHEMA, 'binding': binding,
            'status': 'INITIALIZATION_ONLY_PASSED' if succeeded else 'INITIALIZATION_ONLY_FAILED',
            'failure': failure, 'cleanup_failure': cleanup_failure,
            'summary_read_failure': summary_read_failure,
            'model_construction_attempts': int(model_started),
            'model_construction_and_runtime_transforms_returned': int(model_returned),
            'constructor_count_basis': {'attempted': 'source/MODEL_LOAD_STARTED.json',
                'constructor_and_runtime_transforms_returned': 'source/MODEL_CONSTRUCTED.json',
                'standalone_constructor_return_unobserved_if_runtime_transforms_fail': True},
            'model_forward_calls': 0, 'tts_request_attempts': 0,
            'zero_forward_basis': 'production loader called once; returned forward discarded without invocation; no generate or _run_rows call',
            'audio_files': audio, 'checkpoint_readback_summary': summary,
            'diagnostic_packet_complete': diagnostic_packet_complete,
            'model_verified': succeeded and diagnostic_packet_complete,
            'strict_model_lock_created': (output/'source/MODEL_LOCK.json').is_file(),
            'recipe_sha256': RECIPE_SHA256, 'elapsed_seconds': time.monotonic()-started,
            'same_production_loader': 'english_kokoro48_generate._load_native',
            'training_allowed': False, 'diagnostic_only': True, 'automatic_retry': False}
        g.once(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'INITIALIZATION_ONLY_PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
