"""One new sixteen-request experiment, then four audio-only observers; never training.

Original high-level execution gates are not called or changed. The frozen capture,
receipt readers, and model loaders are reused below their original campaign APIs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vosen_schwa16_recipe as recipe
need, sha, canonical = recipe.need, recipe.sha, recipe.canonical
POLICY = 'english-vosen-schwa16-once-v1'
PIN_FILE = HERE / 'english_vosen_schwa16_pins.json'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
VALID = 'VALID_UNOBSERVED_SOURCE'
TTS_LOCK = '06abd719b12f4191e1552194fc386d5a0b813318d7f8dc4dac719c589c59ac32'
XLSR_LOCK = '71a30ee3d506cd10911a531c7a97918757ad7c924fbb01316e0f126fc0dffce1'
HELPER_SHA256 = {
    'final144_recipe': recipe.OLD_RECIPE_CODE_SHA256,
    'final144_runner': '16814d7cc89d2fe744bae51b9672aca45f360e2307d3f6f76f57fe1e2546aae2',
    'terminal144_observe_once': '67dac982beed528d8a867558e50222712b73a56a93c77e65e614020a37a2dd0d',
    'english_ipa_qwen17_final_ipa_probe_once': '15207db7182e4cb6d85349059b4aff91e721c70e8782fe901d4893f55224be76',
    'english_qwen_capture': 'be99d27dcff84107feca42d648afa863863cd5c9a3ffffaa505d287c29b2104f',
    'english_blind_asr': 'e232cb219cabcc3877366bc446e5180d203d43b094695dba18fddee9b584669e',
    'english_ipa_qwen17_probe_once': 'e5f6bce2e17c3755f7b910d00c84b8d2f3d36ecca3af082c9c62a86cd689b4ed',
    'english_seven_voice_asr_once': '17771ca272204771982dbdc8e959f7fd75838b6598e84f09ac5819d8e3c79184',
    'english_final_ipa_xlsr_once': '716928f7707fc662f62e59a988c0c264123be301abac4c23fe835bc09c052b3a',
    'english_qwen_asr_real_once': '14e8150ee76bf356fad0927e93d20981ccfaf32696beacf871810e26e8ff7990',
    'english_whisper_real_once': '0f455c46061c6e3e4da2eec781234875fe2f59f257f11bacea8105e951593b6d',
    'english_parakeet_real_once': '366cd278f570a37ce68dd20e47e2bee1422daaa109c693ac42ab4bd4c2418881',
    'english_seven_voice_tts_once': 'e9dcf10dd94909edc13bf65ebcc31b2f3df6bdc81ab1876c94cbe28ec936ad0e',
    'english_qwen_real_once': '60e995b4cc39818b8795d8730f1eba81bc57a8bd7c778fc2df040b92f0bca95e',
    'multilingual_campaign': '3cda7b79d1cf4e00f7b2a4aeff6c254f8b49d2adcf527a5ca51093ef60d1ef4f',
}
MEDIA_SHA256 = 'df3e8858e25f041ecbc1ef7f10e7b7715722ad7ad8a2f5e53f120c172d3d40c1'


def check_helpers():
    for name, expected in HELPER_SHA256.items():
        path = HERE / (name + '.py')
        need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
             sha(path.read_bytes()) == expected, 'FROZEN_HELPER_CHANGED:' + name)
    path = HERE.parent / 'research/auto-tts-voice-clone-20261102/runner.py'
    need(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == MEDIA_SHA256,
         'FROZEN_MEDIA_CHANGED')


def helpers():
    check_helpers()
    names = {'original': 'final144_runner', 'terminal': 'terminal144_observe_once',
             'capture': 'english_qwen_capture', 'tts': 'english_ipa_qwen17_probe_once',
             'seven': 'english_seven_voice_asr_once', 'xlsr': 'english_final_ipa_xlsr_once',
             'utility': 'english_ipa_qwen17_final_ipa_probe_once'}
    result = {key: importlib.import_module(name) for key, name in names.items()}
    for key, module in result.items():
        need(Path(module.__file__).resolve() == HERE / (names[key] + '.py'), 'HELPER_IMPORT_LOCATION')
    return result


def pins_template():
    return {'schema': 'english-vosen-schwa16-execution-pins-v1', 'profile': POLICY,
        'enabled': False, 'repository': 'jiying2007/kws-data', 'guard_before': None, 'worker_sha256': None,
        'recipe_code_sha256': sha((HERE / 'vosen_schwa16_recipe.py').read_bytes()),
        'recipe_sha256': sha(canonical(recipe.recipe())), 'helper_sha256': dict(HELPER_SHA256),
        'old_recipe_sha256': recipe.OLD_RECIPE_SHA256,
        'old_request_fingerprints_sha256': sha(canonical(recipe.old_request_fingerprints())),
        'tts_model_lock_sha256': TTS_LOCK, 'xlsr_model_lock_sha256': XLSR_LOCK,
        'max_tts_calls': 16, 'max_calls_per_provider': 16, 'generation_deadline_seconds': 3480,
        'observation_deadline_seconds': 840, 'no_retry': True, **recipe.FLAGS}


def hex_id(value, count=64):
    return type(value) is str and len(value) == count and all(c in '0123456789abcdef' for c in value)


def validate_pins(pins):
    expected = pins_template()
    need(type(pins) is dict and set(pins) == set(expected), 'EXECUTION_PIN_FIELDS')
    mutable = {'enabled', 'guard_before', 'worker_sha256'}
    need(canonical({k: v for k, v in pins.items() if k not in mutable}) ==
         canonical({k: v for k, v in expected.items() if k not in mutable}), 'FROZEN_PROFILE_PINS')
    need(type(pins['enabled']) is bool and
         (pins['guard_before'] is None or hex_id(pins['guard_before'], 40)) and
         (pins['worker_sha256'] is None or hex_id(pins['worker_sha256'])), 'EXECUTION_PIN_TYPES')
    if pins['enabled']:
        need(hex_id(pins['guard_before'], 40) and hex_id(pins['worker_sha256']), 'ENABLED_PINS_INCOMPLETE')


def execution_guard(pins, env, event, worker_sha=None):
    validate_pins(pins)
    check_helpers()
    actual = sha(Path(__file__).read_bytes())
    need(pins['enabled'] is True and pins['worker_sha256'] == actual and
         (worker_sha is None or worker_sha == actual), 'DISABLED_OR_UNPINNED_WORKER')
    need(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == pins['repository'] and
         env.get('GITHUB_EVENT_NAME') == 'push' and event.get('ref') == 'refs/heads/main' and
         env.get('GITHUB_REF') == 'refs/heads/main' and env.get('GITHUB_RUN_ATTEMPT') == '1' and
         type(env.get('GITHUB_RUN_ID')) is str and env['GITHUB_RUN_ID'].isdigit() and
         hex_id(env.get('GITHUB_SHA'), 40) and event.get('before') == pins['guard_before'] and
         event.get('after') == env['GITHUB_SHA'] and event['after'] != event['before'] and
         event.get('repository', {}).get('full_name') == pins['repository'] and
         event.get('repository', {}).get('private') is False, 'FIRST_PUBLIC_MAIN_ATTEMPT_ONE_ONLY')
    return {'run_id': env['GITHUB_RUN_ID'], 'github_sha': env['GITHUB_SHA'],
            'run_attempt': '1', 'test_backend': False}


def run_identity(pins, simulated):
    if simulated:
        return {'run_id': 'TEST_ONLY', 'github_sha': '0' * 40, 'run_attempt': '1', 'test_backend': True}
    utility = helpers()['utility']
    event = utility.loads(utility.regular(Path(os.environ.get('GITHUB_EVENT_PATH', '')), 8 * 1024**2))
    return execution_guard(pins, os.environ, event)


def base_identity(pins, identity):
    return {'schema': POLICY, 'profile': POLICY, 'source_identity': identity,
        'recipe_sha256': sha(canonical(recipe.recipe())), 'code_sha256': sha(Path(__file__).read_bytes()),
        'recipe_code_sha256': pins['recipe_code_sha256'], 'pins_sha256': sha(canonical(pins)),
        'helper_sha256': dict(HELPER_SHA256), 'test_backend': identity['test_backend'],
        'source_labels_are_intents_only': True, 'retry_count': 0, **recipe.FLAGS}


class DeadlineExceeded(BaseException):
    pass


@contextmanager
def deadline(seconds):
    def stop(signum, frame):
        raise DeadlineExceeded('FIXED_WORKER_DEADLINE_OR_TERMINATION')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def make_plan(pins, identity, model_lock_sha):
    return {**recipe.recipe(), 'capture_identity': base_identity(pins, identity),
            'model_lock_sha256': model_lock_sha}


def make_ledger(rows, base):
    counts = Counter(r.get('pcm_sha256') for r in rows if r['status'] == 'GENERATED_UNADMITTED')
    values = []
    for row in rows:
        status = row['status']
        disposition = (VALID if counts[row['pcm_sha256']] == 1 else 'DUPLICATE_PCM_QUARANTINED') \
            if status == 'GENERATED_UNADMITTED' else ('NOT_STARTED_NO_RETRY' if status == 'NOT_STARTED' else status)
        values.append({**row, 'source_status': status, 'source_disposition': disposition,
            'source_attempted': row['attempted'], 'wav_sha256': row.get('wav_sha256'),
            'pcm_sha256': row.get('pcm_sha256'), 'frames': row.get('frames'), **recipe.FLAGS})
    return {**base, 'schema': POLICY + '/source-ledger', 'source_denominator': 16,
        'valid_count': sum(r['source_disposition'] == VALID for r in values),
        'source_dispositions': dict(Counter(r['source_disposition'] for r in values)), 'rows': values}


def generate(output, pins, *, runtime=None):
    """A test-only (backend, media, model_lock) seam never reaches the CLI."""
    validate_pins(pins)
    h = helpers()
    simulated = runtime is not None
    identity = run_identity(pins, simulated)
    output = h['original'].new_output(Path(output))
    base = base_identity(pins, identity)
    once, regular, loads = (getattr(h['utility'], name) for name in ('once', 'regular', 'loads'))
    expected_lock = sha((json.dumps(runtime[2], ensure_ascii=False, sort_keys=True) + '\n').encode()) \
        if simulated else pins['tts_model_lock_sha256']
    plan = make_plan(pins, identity, expected_lock)
    once(output / 'started.json', {**base, 'operation': 'generate', 'call_budget': 16,
                                   'deadline_seconds': pins['generation_deadline_seconds']})
    once(output / 'source-plan.json', plan)
    error, message, stage, lock_sha = None, None, None, None
    with tempfile.TemporaryDirectory(prefix='schwa16-tts-') as private:
        try:
            with deadline(pins['generation_deadline_seconds']):
                if runtime is None:
                    folder, lock = h['tts'].materialize(Path(private))
                    runtime = h['tts'].Backend(folder), h['capture'].media(), lock
                backend, media, lock = runtime
                need(lock.get('model_id') == recipe.MODEL and lock.get('revision') == recipe.REVISION,
                     'TTS_MODEL_IDENTITY')
                once(output / 'model-lock.json', lock)
                lock_sha = sha(regular(output / 'model-lock.json', 1024**2))
                need(lock_sha == expected_lock, 'FROZEN_TTS_MODEL_LOCK')
                stage = h['capture'].capture(plan, output / 'capture', recipe.CaptureAdapter(backend),
                                             max_calls=16, writer=media._write_waves)
                need(stage['attempts_this_call'] == stage['total_started'] <= 16, 'SIXTEEN_CAPTURE_BUDGET')
                once(output / 'capture-stage.json', {**base, **stage})
        except BaseException as exc:
            error, message = type(exc).__name__, h['terminal'].error_message(exc)
        finally:
            rows = h['original'].capture_rows(plan, output / 'capture')
            ledger = make_ledger(rows, base)
            once(output / 'SOURCE_LEDGER.json', ledger)
            attempts = sum(row['attempted'] for row in rows)
            completed = sum(row['status'] == 'GENERATED_UNADMITTED' for row in rows)
            complete = stage is not None and attempts == completed == ledger['valid_count'] == 16 and error is None
            result = {**base, 'status': ('SIMULATED_COMPLETE' if simulated else 'COMPLETE') if complete
                      else 'INCOMPLETE_NO_RETRY', 'terminal': True, 'planned_audio': 16,
                'tts_attempts': attempts, 'generated_audio': completed, 'valid_count': ledger['valid_count'],
                'capture_stages_completed': int(stage is not None), 'source_plan_sha256': sha(regular(output / 'source-plan.json', 1024**2)),
                'model_lock_sha256': lock_sha, 'source_ledger_canonical_sha256': sha(canonical(ledger)),
                'asr_calls': 0, 'phoneme_calls': 0, 'optimizer_steps': 0,
                'error_type': error, 'error_message': message, 'rows': rows}
            once(output / 'result.json', result)
    return result


def source_evidence(source_root, pins, *, allow_simulated=False):
    """Pure receipt readback. Never reads current Actions environment or old campaign gates."""
    validate_pins(pins)
    h = helpers()
    root = Path(source_root)
    regular, loads = h['utility'].regular, h['utility'].loads
    start = loads(regular(root / 'started.json', 131072))
    identity = start.get('source_identity')
    need(type(identity) is dict and set(identity) == {'run_id', 'github_sha', 'run_attempt', 'test_backend'} and
         type(identity['test_backend']) is bool and identity['run_attempt'] == '1', 'SOURCE_IDENTITY_SHAPE')
    simulated = identity['test_backend']
    need(allow_simulated or not simulated, 'SIMULATED_SOURCE_REJECTED')
    if not simulated:
        need(type(identity['run_id']) is str and identity['run_id'].isdigit() and hex_id(identity['github_sha'], 40) and
             identity['github_sha'] != pins['guard_before'] and pins['enabled'] is True and
             pins['worker_sha256'] == sha(Path(__file__).read_bytes()), 'SOURCE_PUBLIC_RUN_AND_CODE')
    base = base_identity(pins, identity)
    need(start == {**base, 'operation': 'generate', 'call_budget': 16,
                   'deadline_seconds': pins['generation_deadline_seconds']}, 'SOURCE_STARTED_BINDING')
    raw_result = regular(root / 'result.json', 2 * 1024**2)
    result = loads(raw_result)
    need(all(result.get(k) == v for k, v in base.items()) and result.get('terminal') is True and
         result.get('planned_audio') == 16 and result.get('asr_calls') == result.get('phoneme_calls') ==
         result.get('optimizer_steps') == 0, 'SOURCE_NOT_TERMINAL_OR_BOUND')
    plan_raw = regular(root / 'source-plan.json', 1024**2)
    plan = loads(plan_raw)
    expected_lock = plan.get('model_lock_sha256') if simulated else pins['tts_model_lock_sha256']
    need(hex_id(expected_lock) and plan == make_plan(pins, identity, expected_lock) and
         result.get('source_plan_sha256') == sha(plan_raw), 'SOURCE_PLAN_CHANGED')
    capture = root / 'capture'
    if capture.exists():
        need(loads(regular(capture / 'plan-lock.json', 32768)) == {
            'policy': h['capture'].POLICY, 'plan_sha256': sha(canonical(plan)),
            'model_calls_admitted_to_training': False}, 'CAPTURE_PLAN_LOCK')
        cells_root = capture / 'cells'
        if cells_root.exists():
            need(not cells_root.is_symlink() and {p.name for p in cells_root.iterdir()} <=
                 {c['cell_id'] for c in plan['cells']}, 'EXTRA_SOURCE_CELL')
    rows = h['original'].capture_rows(plan, capture)
    attempts = sum(row['attempted'] for row in rows)
    completed = sum(row['status'] == 'GENERATED_UNADMITTED' for row in rows)
    need(rows == result.get('rows') and result.get('tts_attempts') == attempts <= 16 and
         result.get('generated_audio') == completed, 'SOURCE_RECEIPTS_CHANGED')
    if result.get('model_lock_sha256') is not None:
        lock_raw = regular(root / 'model-lock.json', 1024**2)
        lock = loads(lock_raw)
        need(sha(lock_raw) == result['model_lock_sha256'] == expected_lock and
             lock.get('model_id') == recipe.MODEL and lock.get('revision') == recipe.REVISION,
             'SOURCE_MODEL_LOCK_CHANGED')
    else:
        need(attempts == 0 and not (root / 'model-lock.json').exists(), 'ATTEMPT_WITHOUT_MODEL_LOCK')
    stages = result.get('capture_stages_completed')
    need(type(stages) is int and stages in (0, 1), 'CAPTURE_STAGE_COUNT')
    if stages:
        stage = loads(regular(root / 'capture-stage.json', 131072))
        need(all(stage.get(k) == v for k, v in base.items()) and stage.get('policy') == h['capture'].POLICY and
             stage.get('attempts_this_call') == stage.get('total_started') == attempts and
             stage.get('total_completed') == completed and stage.get('asr_calls') == 0 and
             stage.get('training_admitted') is False, 'CAPTURE_STAGE_RECEIPTS')
    else:
        need(not (root / 'capture-stage.json').exists(), 'UNDECLARED_CAPTURE_STAGE')
    ledger = make_ledger(rows, base)
    need(loads(regular(root / 'SOURCE_LEDGER.json', 2 * 1024**2)) == ledger and
         result.get('valid_count') == ledger['valid_count'] and
         result.get('source_ledger_canonical_sha256') == sha(canonical(ledger)), 'SOURCE_LEDGER_CHANGED')
    complete = stages == 1 and attempts == completed == ledger['valid_count'] == 16 and result.get('error_type') is None
    expected_status = ('SIMULATED_COMPLETE' if simulated else 'COMPLETE') if complete else 'INCOMPLETE_NO_RETRY'
    need(result.get('status') == expected_status, 'FALSE_SOURCE_COMPLETION')
    verified = [{'cell': cell, 'row': row, 'wav': capture / 'cells' / cell['cell_id'] / 'audio-16k.wav'}
                for cell, row in zip(plan['cells'], ledger['rows']) if row['source_status'] == 'GENERATED_UNADMITTED']
    valid = [item for item in verified if item['row']['source_disposition'] == VALID]
    binding = {'source_identity': identity, 'recipe_sha256': base['recipe_sha256'],
        'source_ledger_canonical_sha256': sha(canonical(ledger)), 'source_result_sha256': sha(raw_result),
        'source_plan_sha256': sha(plan_raw), 'source_model_lock_sha256': result.get('model_lock_sha256'),
        'execution_pins_sha256': sha(canonical(pins)), 'test_backend': simulated, **recipe.FLAGS}
    return {'recipe': recipe.recipe(), 'plan': plan, 'ledger': ledger, 'valid': valid,
            'verified_generated': verified, 'binding': binding}


def observer_base(pins, source, provider):
    return {**base_identity(pins, source['binding']['source_identity']), **source['binding'],
        'observer_identity': source['binding']['source_identity'], 'provider': provider,
        'source_denominator': 16, 'call_budget': len(source['valid']), 'new_tts_calls': 0,
        'optimizer_steps': 0, 'deadline_seconds': pins['observation_deadline_seconds']}


def observe(provider, source_root, pins, output, *, backend=None):
    need(provider in PROVIDERS, 'UNKNOWN_OBSERVER')
    validate_pins(pins)
    h = helpers()
    simulated = backend is not None
    identity = run_identity(pins, simulated)
    source = source_evidence(source_root, pins, allow_simulated=simulated)
    need(identity == source['binding']['source_identity'], 'SOURCE_OBSERVER_SAME_RUN_REQUIRED')
    output = h['original'].new_output(Path(output))
    once, regular = h['utility'].once, h['utility'].regular
    base = observer_base(pins, source, provider)
    once(output / 'started.json', base)
    (output / 'cells').mkdir(mode=0o700)
    attempts, error, message, lock_sha = 0, None, None, None
    with tempfile.TemporaryDirectory(prefix='schwa16-observer-') as private_name:
        private = Path(private_name)
        try:
            with deadline(pins['observation_deadline_seconds']):
                model_root = private / 'model'
                model_root.mkdir(mode=0o700)
                if simulated:
                    lock = {'test_backend': True, 'not_a_real_model': True}
                elif provider == 'phones':
                    predict, lock = h['xlsr'].load_runtime(model_root)
                    backend = lambda wav, _: predict(wav)
                else:
                    backend = h['seven']._load(provider, model_root)
                    lock = {'provenance': list(h['seven'].PROVENANCE[provider]),
                            'loader_git_blob': h['original'].MODULES['english_seven_voice_asr_once']}
                once(output / 'model-lock.json', lock)
                lock_sha = h['terminal'].verify_observer_lock(output, provider, simulated)
                for item in source['valid']:
                    row = item['row']
                    need(attempts < base['call_budget'] <= pins['max_calls_per_provider'], 'OBSERVER_CALL_BUDGET')
                    raw = regular(item['wav'], 2 * 1024**2)
                    need(sha(raw) == row['wav_sha256'], 'SOURCE_CHANGED_BEFORE_OBSERVATION')
                    folder = output / 'cells' / row['cell_id']
                    folder.mkdir(mode=0o700)
                    anonymous = private / 'audio.wav'
                    with anonymous.open('xb') as stream:
                        stream.write(raw)
                    receipt = h['terminal'].receipt_identity(base, row)
                    once(folder / 'started.json', receipt)
                    attempts += 1
                    try:
                        value = h['terminal'].output_value(provider, backend(anonymous, folder), row['frames'])
                        once(folder / 'completed.json', {**receipt, **value,
                            'observation_status': 'COMPLETE', 'decoder_completed': True})
                    except Exception as exc:
                        once(folder / 'failed.json', {**receipt, 'observation_status': 'FAILED_NO_RETRY',
                            'error_type': type(exc).__name__, 'error_message': h['terminal'].error_message(exc)})
                    finally:
                        anonymous.unlink()
        except BaseException as exc:
            error, message = type(exc).__name__, h['terminal'].error_message(exc)
        finally:
            rows = h['terminal'].read_observation_rows(output, base, source['ledger'])
            completed = sum(r['observation_status'] == 'COMPLETE' for r in rows)
            need(attempts == sum(r['observation_attempted'] for r in rows), 'OBSERVER_ATTEMPT_RECEIPTS')
            status = ('SIMULATED_COMPLETE' if simulated else 'OBSERVATIONS_COMPLETE') \
                if completed == len(source['valid']) and error is None else 'INCOMPLETE_NO_RETRY'
            result = {**base, 'status': status, 'terminal': True, 'attempts': attempts, 'completed': completed,
                'model_lock_sha256': lock_sha, 'error_type': error, 'error_message': message, 'rows': rows}
            once(output / 'summary.json', result)
    return result


def combine(source_root, observation_parent, pins, *, allow_simulated=False):
    source = source_evidence(source_root, pins, allow_simulated=allow_simulated)
    h = helpers()
    regular, loads = h['utility'].regular, h['utility'].loads
    ledger = source['ledger']
    values, summaries, locks = {}, {}, {}
    for provider in PROVIDERS:
        root = Path(observation_parent) / ('observer-' + provider)
        base = observer_base(pins, source, provider)
        if not (root / 'started.json').exists():
            need(not root.exists() or not any(root.iterdir()), 'OBSERVER_OUTPUT_WITHOUT_STARTED')
            values[provider] = {'status': 'UNKNOWN_MISSING_PROVIDER_OUTPUT', 'attempts': None, 'completed': None,
                'rows': [{**r, 'observation_attempted': None, 'observation_status':
                    'UNKNOWN_MISSING_PROVIDER_OUTPUT' if r['source_disposition'] == VALID else 'NOT_ELIGIBLE_SOURCE'}
                    for r in ledger['rows']]}
            summaries[provider], locks[provider] = None, None
            continue
        need(loads(regular(root / 'started.json', 131072)) == base, 'CROSS_RUN_OR_OBSERVER_BINDING')
        rows = h['terminal'].read_observation_rows(root, base, ledger)
        attempts = sum(r['observation_attempted'] for r in rows)
        completed = sum(r['observation_status'] == 'COMPLETE' for r in rows)
        path = root / 'summary.json'
        if path.exists():
            raw = regular(path, 2 * 1024**2)
            value = loads(raw)
            need(all(value.get(k) == v for k, v in base.items()) and value.get('terminal') is True and
                 value.get('rows') == rows and value.get('attempts') == attempts <= len(source['valid']) and
                 value.get('completed') == completed, 'OBSERVER_SUMMARY_RECEIPTS')
            complete = completed == len(source['valid']) and value.get('error_type') is None
            expected_status = ('SIMULATED_COMPLETE' if base['test_backend'] else 'OBSERVATIONS_COMPLETE') \
                if complete else 'INCOMPLETE_NO_RETRY'
            need(value.get('status') == expected_status, 'FALSE_OBSERVER_COMPLETION')
            summaries[provider] = sha(raw)
        else:
            value = {'status': 'UNKNOWN_MISSING_PROVIDER_TERMINAL', 'attempts': attempts,
                     'completed': completed, 'rows': rows}
            summaries[provider] = None
        if attempts:
            expected_sha = value.get('model_lock_sha256')
            need(not path.exists() or hex_id(expected_sha), 'ATTEMPTED_MODEL_LOCK_NOT_BOUND')
            locks[provider] = h['terminal'].verify_observer_lock(root, provider, base['test_backend'], expected_sha)
        else:
            locks[provider] = None
        values[provider] = value
    complete = all(v['status'] in ('OBSERVATIONS_COMPLETE', 'SIMULATED_COMPLETE') for v in values.values())
    simulated = source['binding']['test_backend']
    return {'schema': POLICY + '/combined', **source['binding'],
        'status': ('SIMULATED_COMPLETE' if simulated else 'FOUR_OBSERVERS_COMPLETE') if complete else 'INCOMPLETE_NO_RETRY',
        'source_denominator': 16, 'valid_count': ledger['valid_count'],
        'observer_identity': source['binding']['source_identity'],
        'observer_summaries_sha256': summaries, 'observer_model_locks_sha256': locks,
        'providers': {p: {k: value.get(k) for k in ('status', 'attempts', 'completed', 'error_type', 'error_message')}
                      for p, value in values.items()},
        'model_calls_by_combiner': 0, 'new_tts_calls': 0, 'optimizer_steps': 0,
        'source_labels_are_intents_only': True,
        'rows': [{**row, 'observations': {p: values[p]['rows'][i] for p in PROVIDERS}}
                 for i, row in enumerate(ledger['rows'])]}


def review_rows(source_root, observations_root, pins, *, allow_simulated=False):
    """Return every disposition; consumers must require four complete same-PCM observations per eligible row."""
    source = source_evidence(source_root, pins, allow_simulated=allow_simulated)
    combined = combine(source_root, observations_root, pins, allow_simulated=allow_simulated)
    return {**source, 'rows': combined['rows'], 'binding': {**source['binding'],
        **{k: combined[k] for k in ('observer_identity', 'observer_summaries_sha256', 'observer_model_locks_sha256')}},
        'combined': combined, 'status': combined['status']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-check', action='store_true')
    mode.add_argument('--generate', action='store_true')
    mode.add_argument('--observe', choices=PROVIDERS)
    mode.add_argument('--combine', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--pins', type=Path, default=PIN_FILE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    check_helpers()
    active = args.generate or args.observe or args.combine is not None
    if not active:
        need(args.output is None and args.source_root is None, 'SOURCE_CHECK_ARGUMENTS')
        pins = json.loads(args.pins.read_bytes()) if args.pins.exists() else pins_template()
        validate_pins(pins)
        print(json.dumps({'status': 'SOURCE_CONTRACT_ONLY', 'recipe_sha256': pins['recipe_sha256'],
                          'planned_audio': 16, 'model_calls': 0, **recipe.FLAGS}, sort_keys=True))
        return 0
    h = helpers()
    pins = h['utility'].loads(h['utility'].regular(args.pins, 131072))
    need(args.output is not None, 'OUTPUT_REQUIRED')
    if args.generate:
        need(args.source_root is None, 'GENERATION_DOES_NOT_CONSUME_OLD_AUDIO')
        result = generate(args.output, pins)
    else:
        need(args.source_root is not None, 'NEW_SOURCE_ROOT_REQUIRED')
        if args.observe:
            result = observe(args.observe, args.source_root, pins, args.output)
        else:
            need(args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists() and
                 not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'CREATE_ONLY_COMBINED_OUTPUT')
            result = combine(args.source_root, args.combine, pins)
            h['utility'].once(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'helper_sha256')}, sort_keys=True))
    return 2 if result['status'] == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
