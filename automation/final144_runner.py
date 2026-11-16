"""Bounded 144-cell adapter; real execution requires exact evidence and run pins.

Each of nine fixed shards invokes capture once with its sixteen-attempt limit.
All source labels remain intents, and no observer changes admission or gold.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if HERE.name == 'automation' else HERE.parent / 'kws-data'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'automation'))
import final144_recipe as recipe
from english_ipa_qwen17_final_ipa_probe_once import regular, loads, once
need, sha, canonical = recipe.need, recipe.sha, recipe.canonical
POLICY = 'english-final144-same-run-capture-observe-v1'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
SHARDS, SHARD_SIZE = 9, 16
MODULES = {
    'english_qwen_capture': '380c3f2ef2bc789ef1e653a8c21d216b28911d23',
    'english_blind_asr': '35d076f20fc9413675e60d3e6b4a6e2bee13d0a0',
    'english_ipa_qwen17_probe_once': '9204ddf194fdf2f1d15852f5b0283012c2706280',
    'english_seven_voice_asr_once': 'f3bf07e6237dfaeebf7a92d49a821ac0efcdc09b',
    'english_final_ipa_xlsr_once': '42c198952cab9409acc17e39c20c98178f610d97',
    'english_ipa_qwen17_final_ipa_probe_once': 'c8de4cfd4b743e787ae1b8d9728375aef28babba',
}


def check_modules():
    for name, expected in MODULES.items():
        raw = regular(REPO / 'automation' / (name + '.py'), 256 * 1024)
        need(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected,
             'REUSED_MODULE_CHANGED:' + name)


def guard(pins, pilot_review, xlsr_review):
    """Read exact evidence; the separate frozen useful decision is never inferred."""
    recipe.execution_guard(pins, loads(regular(Path(os.environ['GITHUB_EVENT_PATH']), 8 * 1024**2)), os.environ)
    validate_evidence(pins, pilot_review, xlsr_review)


def validate_evidence(pins, pilot_review, xlsr_review):
    """Pure readback of immutable evidence; this function does not authorize work."""
    check_modules()
    old_raw = regular(pilot_review, 2 * 1024**2)
    old = loads(old_raw)
    need(sha(old_raw) == pins['pilot_review_json_sha256'] and old.get('test_backend') is False and
         old.get('status') == 'FOUR_OBSERVERS_COMPLETE_REVIEW_REQUIRED' and
         old.get('run_id') == str(pins['pilot_run_id']) and old.get('github_sha') == pins['pilot_head_sha'] and
         old.get('source_plan_sha256') == pins['pilot_source_plan_sha256'] and
         old.get('model_lock_sha256') == pins['pilot_model_lock_sha256'] and
         old.get('code_sha256') == pins['pilot_code_sha256'], 'PILOT_REVIEW_EVIDENCE')
    for key in ('xlsr_run_id', 'xlsr_artifact_id'):
        need(type(pins.get(key)) is int and pins[key] > 0, 'MISSING_XLSR_PIN:' + key)
    for key in ('xlsr_summary_sha256', 'xlsr_zip_sha256', 'xlsr_model_lock_sha256'):
        value = pins.get(key)
        need(type(value) is str and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), 'MISSING_XLSR_PIN:' + key)
    raw = regular(xlsr_review, 2 * 1024**2)
    value = loads(raw)
    xlsr = importlib.import_module('english_final_ipa_xlsr_once')
    need(sha(raw) == pins['xlsr_summary_sha256'] and value.get('schema') == xlsr.POLICY and
         value.get('run_id') == str(pins['xlsr_run_id']) and value.get('status') == 'COMPLETE' and
         value.get('test_backend') is False and value.get('attempts') == value.get('completed') == 8 and
         value.get('model_revision') == xlsr.REVISION and value.get('model_weights_sha256') == xlsr.WEIGHTS_SHA and
         all(value.get(k) is False for k in xlsr.FLAGS), 'XLSR_EVIDENCE_NOT_COMPLETE_OR_BOUND')
    expected = {r[0]: r[4] for r in xlsr.original.ROSTER + xlsr.RENDER_ROSTER}
    rows = value.get('rows', [])
    need(len(rows) == 8 and {r['cell_id']: r['wav_sha256'] for r in rows} == expected and
         all(r.get('status') == 'COMPLETE' and r.get('test_backend') is False for r in rows),
         'XLSR_EIGHT_SOURCE_BINDING')
    return {'pilot_review_json_sha256': sha(old_raw), 'xlsr_summary_sha256': sha(raw),
            'model_calls': 0, 'pronunciation_pass': False}


def identity(pins, simulated):
    return {'schema': POLICY, 'recipe_sha256': sha(canonical(recipe.recipe())),
            'code_sha256': sha(regular(Path(__file__), 256 * 1024)),
            'execution_pins_sha256': sha(canonical(pins)), 'reused_module_blobs': MODULES,
            'github_sha': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
            'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'test_backend': simulated,
            'source_labels_are_intents_only': True, 'acoustic_review': 'UNCERTAIN_MACHINE_OBSERVATIONS_ONLY',
            'retry_count': 0, **recipe.FLAGS}


def shard_plan(index, lock_sha, pins, simulated):
    need(type(index) is int and 0 <= index < SHARDS, 'FIXED_SHARD_INDEX')
    whole = recipe.recipe()
    cells = whole['cells'][index * SHARD_SIZE:(index + 1) * SHARD_SIZE]
    counts = {'cells': SHARD_SIZE, 'train': sum(c['role'] == 'train' for c in cells),
              'development': sum(c['role'] == 'development' for c in cells),
              'wake_candidates': sum(c['intent'] == 'wake-candidate' for c in cells),
              'ordinary_negative_candidates': sum(c['intent'] == 'ordinary-negative-candidate' for c in cells),
              'independent_humans': 0, 'independent_test_cells': 0}
    return {**whole, 'cells': cells, 'counts': counts, 'global_counts': whole['counts'],
            'global_recipe_sha256': sha(canonical(whole)), 'shard_index': index, 'shard_count': SHARDS,
            'max_tts_calls': SHARD_SIZE, 'capture_stage_budgets': [SHARD_SIZE],
            'capture_identity': identity(pins, simulated), 'model_lock_sha256': lock_sha}


def new_output(output):
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         REPO.resolve() not in output.resolve().parents and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_PRIVATE_OUTPUT')
    output.mkdir(mode=0o700)
    return output


def capture_rows(plan, root):
    """Receipt readback does not turn missing/failed/unknown audio into a label."""
    blind = importlib.import_module('english_blind_asr')
    rows = []
    for cell in plan['cells']:
        folder = root / 'cells' / cell['cell_id']
        row = {**cell, 'status': 'NOT_STARTED', 'attempted': False, 'source_labels_are_intents_only': True,
               'acoustic_review': 'UNCERTAIN_MACHINE_OBSERVATIONS_ONLY'}
        if folder.exists():
            row['status'] = 'INVALID_SOURCE_RECEIPT'
        if (folder / 'started.json').exists():
            row.update(status='STARTED_UNKNOWN_NO_RETRY', attempted=True)
            try:
                started = loads(regular(folder / 'started.json', 32768))
                need(started.get('schema') == blind.english.POLICY and
                     started.get('cell_id') == cell['cell_id'] and type(started.get('model_calls')) is int and
                     started.get('model_calls') == 1 and type(started.get('retry_count')) is int and
                     started.get('plan_sha256') == sha(canonical(plan)) and started.get('retry_count') == 0 and
                     started.get('source_family_id') == cell['source_family_id'] and
                     started.get('training_admitted') is False, 'CAPTURE_STARTED_IDENTITY')
                if (folder / 'failed.json').exists():
                    failed = loads(regular(folder / 'failed.json', 32768))
                    need(not (folder / 'completed.json').exists() and failed.get('cell_id') == cell['cell_id'] and
                         failed.get('schema') == blind.english.POLICY and
                         failed.get('training_admitted') is False and
                         failed.get('status') == 'FAILED_NO_RETRY', 'CAPTURE_FAILED_IDENTITY')
                    row.update(status='FAILED_NO_RETRY', error_type=failed.get('error_type'))
                else:
                    checked = blind.verify_capture(plan, root, cell)
                    if checked is not None:
                        wav, info = checked
                        receipt = loads(regular(folder / 'completed.json', 65536))
                        audio = receipt['audio']
                        need(receipt.get('human_gold') is False and
                             sha(regular(folder / 'native-f32.wav', 4 * 1024**2)) == audio['native_sha256'],
                             'CAPTURE_NATIVE_RECEIPT')
                        row.update(status='GENERATED_UNADMITTED', **info, pcm_sha256=audio['pcm_sha256'])
            except Exception as error:
                row.update(status='INVALID_SOURCE_RECEIPT', error_type=type(error).__name__)
        rows.append(row)
    return rows


def generate(output, pins, shard, pilot_review=None, xlsr_review=None, *, runtime=None):
    """Offline test seam: (Qwen17-compatible backend, capture-media helper, model lock)."""
    simulated = runtime is not None
    plan = shard_plan(shard, None, pins, simulated)
    if not simulated:
        guard(pins, pilot_review, xlsr_review)
    capture = importlib.import_module('english_qwen_capture')
    output = new_output(output)
    base = {**identity(pins, simulated), 'shard_index': shard, 'shard_count': SHARDS}
    once(output / 'started.json', {**base, 'operation': 'generate', 'call_budget': SHARD_SIZE})
    error, lock_sha, plan_sha, stages = None, None, None, []
    with tempfile.TemporaryDirectory(prefix='final144-tts-') as private:
        try:
            if runtime is None:
                tts = importlib.import_module('english_ipa_qwen17_probe_once')
                folder, lock = tts.materialize(Path(private))
                runtime = tts.Backend(folder), capture.media(), lock
            backend, media, lock = runtime
            need(lock.get('model_id') == recipe.MODEL and lock.get('revision') == recipe.REVISION,
                 'TTS_MODEL_IDENTITY')
            once(output / 'model-lock.json', lock)
            lock_sha = sha(regular(output / 'model-lock.json', 1024**2))
            need(simulated or lock_sha == pins['pilot_model_lock_sha256'], 'EXACT_PREVIOUSLY_OBSERVED_TTS_MODEL_LOCK')
            plan = {**plan, 'model_lock_sha256': lock_sha}
            once(output / 'source-plan.json', plan)
            plan_sha = sha(regular(output / 'source-plan.json', 2 * 1024**2))
            adapter = recipe.RenderingCaptureAdapter(backend)
            stage = capture.capture(plan, output / 'capture', adapter, max_calls=16, writer=media._write_waves)
            need(0 <= stage['attempts_this_call'] <= 16 and stage['total_started'] <= SHARD_SIZE,
                 'CAPTURE_SIXTEEN_BUDGET')
            stages.append(stage)
            once(output / 'capture-stage.json', {**base, **stage})
        except BaseException as exc:
            error = type(exc).__name__
            if not isinstance(exc, Exception):
                raise
        finally:
            rows = capture_rows(plan, output / 'capture')
            attempts = sum(r['attempted'] for r in rows)
            completed = sum(r['status'] == 'GENERATED_UNADMITTED' for r in rows)
            status = 'COMPLETE' if attempts == completed == SHARD_SIZE and len(stages) == 1 and error is None else 'INCOMPLETE_NO_RETRY'
            if simulated and status == 'COMPLETE':
                status = 'SIMULATED_COMPLETE'
            result = {**base, 'status': status, 'planned_audio': SHARD_SIZE, 'global_planned_audio': 144,
                      'tts_attempts': attempts,
                      'generated_audio': completed, 'source_plan_sha256': plan_sha, 'model_lock_sha256': lock_sha,
                      'asr_calls': 0, 'phoneme_calls': 0, 'optimizer_steps': 0, 'error_type': error,
                      'capture_stages_completed': len(stages), 'rows': rows}
            once(output / 'result.json', result)
    return result


def source_shard(root, pins, shard, *, allow_simulated=False):
    """Verify the original sixteen-cell plan/receipts; do not rewrite its lock."""
    root = Path(root)
    started = loads(regular(root / 'started.json', 65536))
    raw_result = regular(root / 'result.json', 2 * 1024**2)
    result = loads(raw_result)
    simulated = started.get('test_backend') is True
    need(allow_simulated or not simulated, 'SIMULATED_SOURCE_REJECTED')
    base = {**identity(pins, simulated), 'shard_index': shard, 'shard_count': SHARDS}
    for value in (started, result):
        need(all(value.get(k) == v for k, v in base.items()), 'SOURCE_RUN_CODE_RECIPE_PIN_BINDING')
    need(started.get('operation') == 'generate' and started.get('call_budget') == SHARD_SIZE and
         result.get('status') == ('SIMULATED_COMPLETE' if simulated else 'COMPLETE') and
         result.get('tts_attempts') == result.get('generated_audio') == result.get('planned_audio') == SHARD_SIZE and
         result.get('global_planned_audio') == 144 and
         result.get('asr_calls') == result.get('phoneme_calls') == result.get('optimizer_steps') == 0 and
         result.get('capture_stages_completed') == 1, 'SOURCE_NOT_COMPLETE')
    lock_raw = regular(root / 'model-lock.json', 1024**2)
    lock = loads(lock_raw)
    need(lock.get('model_id') == recipe.MODEL and lock.get('revision') == recipe.REVISION and
         sha(lock_raw) == result['model_lock_sha256'], 'SOURCE_MODEL_LOCK')
    plan_raw = regular(root / 'source-plan.json', 2 * 1024**2)
    plan = loads(plan_raw)
    need(plan == shard_plan(shard, sha(lock_raw), pins, simulated) and
         sha(plan_raw) == result['source_plan_sha256'], 'SOURCE_PLAN_CHANGED')
    rows = capture_rows(plan, root / 'capture')
    need(rows == result['rows'] and all(r['status'] == 'GENERATED_UNADMITTED' for r in rows) and
         {p.name for p in (root / 'capture/cells').iterdir()} == {c['cell_id'] for c in plan['cells']} and
         len({r['wav_sha256'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == SHARD_SIZE,
         'SOURCE_RECEIPTS_OR_AUDIO_NOT_COMPLETE_UNIQUE')
    binding = {'shard_index': shard, 'source_plan_sha256': sha(plan_raw), 'source_result_sha256': sha(raw_result),
               'source_model_lock_sha256': sha(lock_raw)}
    return binding, [(r, root / 'capture/cells' / r['cell_id'] / 'audio-16k.wav') for r in rows]


def source_cells(root, pins, *, allow_simulated=False):
    """Read exactly nine complete shards from this run before any observation."""
    check_modules()
    root = Path(root)
    need(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)) and
         {p.name for p in root.iterdir()} == {'shard-%02d' % i for i in range(SHARDS)},
         'EXACT_NINE_SOURCE_SHARDS')
    bindings, cells = [], []
    for index in range(SHARDS):
        binding, rows = source_shard(root / ('shard-%02d' % index), pins, index,
                                    allow_simulated=allow_simulated)
        bindings.append(binding)
        cells.extend(rows)
    need(len({b['source_model_lock_sha256'] for b in bindings}) == 1, 'SHARD_MODEL_LOCK_BYTES_DIFFER')
    need([r['cell_id'] for r, _ in cells] == [c['cell_id'] for c in recipe.recipe()['cells']] and
         len({r['wav_sha256'] for r, _ in cells}) == len({r['pcm_sha256'] for r, _ in cells}) == 144,
         'GLOBAL_EXACT_144_UNIQUE_SOURCE_CELLS')
    return {'source_shards': bindings, 'source_manifest_sha256': sha(canonical(bindings)),
            'source_model_lock_sha256': bindings[0]['source_model_lock_sha256']}, cells


def review_rows(source_parent, observations_parent, pins, *, allow_simulated=False):
    """Join four complete raw observers; never assign pronunciation admission."""
    binding, cells = source_cells(source_parent, pins, allow_simulated=allow_simulated)
    parent = Path(observations_parent)
    need(parent.is_dir() and not parent.is_symlink() and
         {p.name for p in parent.iterdir()} == {'observer-' + p for p in PROVIDERS},
         'EXACT_FOUR_OBSERVER_DIRECTORIES')
    summaries, values, simulated_any = {}, {}, False
    for provider in PROVIDERS:
        raw = regular(parent / ('observer-' + provider) / 'summary.json', 2 * 1024**2)
        value = loads(raw)
        simulated = value.get('test_backend') is True
        simulated_any |= simulated
        need(allow_simulated or not simulated, 'SIMULATED_OBSERVER_REJECTED')
        expected = {**identity(pins, simulated), **binding, 'provider': provider}
        need(all(value.get(k) == v for k, v in expected.items()) and
             value.get('status') == ('SIMULATED_COMPLETE' if simulated else 'COMPLETE') and
             value.get('attempts') == value.get('completed') == 144 and
             value.get('new_tts_calls') == value.get('optimizer_steps') == 0, 'OBSERVER_NOT_COMPLETE_OR_BOUND')
        rows = value.get('rows', [])
        need(len(rows) == 144 and [r['cell_id'] for r in rows] == [r['cell_id'] for r, _ in cells],
             'OBSERVER_EXACT_144_CELL_ROSTER')
        for observed, (cell, _) in zip(rows, cells):
            need(observed.get('status') == 'COMPLETE' and observed.get('decoder_completed') is True and
                 observed.get('test_backend') is simulated and
                 all(observed.get(k) == v for k, v in cell.items() if k != 'status'), 'OBSERVER_SOURCE_ROW_DRIFT')
            data = observed.get('raw_ipa_tokens') if provider == 'phones' else observed.get('text')
            need((type(data) is list and 1 <= len(data) <= 300 and
                  all(type(t) is str and 0 < len(t) <= 40 and not any(ord(c) < 32 for c in t) for t in data))
                 if provider == 'phones' else
                 (type(data) is str and data.strip() and len(data) <= 512 and not any(ord(c) < 32 for c in data)),
                 'OBSERVER_RAW_OUTPUT')
        summaries[provider], values[provider] = sha(raw), rows
    joined = [({**cell, 'transcripts': {p: values[p][i]['text'] for p in PROVIDERS[:3]},
                'raw_ipa_tokens': values['phones'][i]['raw_ipa_tokens'], 'test_backend': simulated_any}, wav)
              for i, (cell, wav) in enumerate(cells)]
    return {**identity(pins, simulated_any), **binding, 'observer_summaries_sha256': summaries,
            'review_status': 'FOUR_OBSERVERS_COMPLETE_INTENT_REVIEW_ONLY'}, joined


def observe(provider, source_parent, output, pins, pilot_review=None, xlsr_review=None, *, backend=None):
    need(provider in PROVIDERS, 'PROVIDER')
    simulated = backend is not None
    if not simulated:
        guard(pins, pilot_review, xlsr_review)
    binding, cells = source_cells(source_parent, pins, allow_simulated=simulated)
    output = new_output(output)
    base = {**identity(pins, simulated), **binding, 'provider': provider}
    once(output / 'started.json', {**base, 'call_budget': 144})
    attempts, rows, error = 0, [], None
    with tempfile.TemporaryDirectory(prefix='final144-' + provider + '-') as private:
        try:
            if backend is None:
                if provider == 'phones':
                    xlsr = importlib.import_module('english_final_ipa_xlsr_once')
                    predict, lock = xlsr.load_runtime(Path(private))
                    once(output / 'model-lock.json', lock)
                    need(sha(regular(output / 'model-lock.json', 1024**2)) == pins['xlsr_model_lock_sha256'],
                         'EXACT_PREVIOUSLY_OBSERVED_XLSR_MODEL_LOCK')
                    backend = lambda wav, _: predict(wav)
                else:
                    seven = importlib.import_module('english_seven_voice_asr_once')
                    backend = seven._load(provider, Path(private))
                    once(output / 'model-lock.json', {'provenance': list(seven.PROVENANCE[provider]),
                                                     'loader_git_blob': MODULES['english_seven_voice_asr_once']})
            for index, (cell, wav) in enumerate(cells):
                need(attempts < 144, 'MAX_144_OBSERVER_CALLS')
                folder = output / ('observation-%03d' % index)
                folder.mkdir(mode=0o700)
                audio = regular(wav, 2 * 1024**2)
                need(sha(audio) == cell['wav_sha256'], 'SOURCE_CHANGED_BEFORE_OBSERVATION')
                anonymous = folder / 'audio.wav'
                with anonymous.open('xb') as target:
                    target.write(audio)
                row = {**cell, 'test_backend': simulated}
                once(folder / 'started.json', {**row, 'attempt': 1})
                attempts += 1
                try:
                    value = backend(anonymous, folder)
                    if provider == 'phones':
                        tokens, frames = value
                        need(type(frames) is int and frames == cell['frames'] and type(tokens) is list and
                             1 <= len(tokens) <= 300 and all(type(t) is str and 0 < len(t) <= 40 and
                             not any(ord(c) < 32 for c in t) for t in tokens), 'PHONE_OUTPUT')
                        row['raw_ipa_tokens'] = tokens
                    else:
                        need(type(value) is str and value.strip() and len(value) <= 512 and
                             not any(ord(c) < 32 for c in value), 'ASR_OUTPUT')
                        row['text'] = value
                    row.update(status='COMPLETE', decoder_completed=True)
                    once(folder / 'completed.json', row)
                except BaseException as exc:
                    row.update(status='FAILED_NO_RETRY', error_type=type(exc).__name__)
                    once(folder / 'failed.json', row)
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    rows.append(row)
                    anonymous.unlink()
        except BaseException as exc:
            error = type(exc).__name__
            if not isinstance(exc, Exception):
                raise
        finally:
            completed = sum(r.get('status') == 'COMPLETE' for r in rows)
            status = 'COMPLETE' if completed == 144 and error is None else 'INCOMPLETE_NO_RETRY'
            if simulated and status == 'COMPLETE':
                status = 'SIMULATED_COMPLETE'
            result = {**base, 'status': status, 'attempts': attempts, 'completed': completed,
                      'new_tts_calls': 0, 'optimizer_steps': 0, 'rows': rows, 'error_type': error}
            once(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--generate', action='store_true')
    modes.add_argument('--observe', choices=PROVIDERS)
    parser.add_argument('--pins', type=Path, default=recipe.PIN_FILE)
    parser.add_argument('--pilot-review', type=Path)
    parser.add_argument('--xlsr-review', type=Path)
    parser.add_argument('--source-parent', type=Path)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    pins = loads(regular(args.pins, 65536))
    if not (args.generate or args.observe):
        need(args.output is None and args.source_parent is None and args.shard is None, 'SOURCE_ONLY_ARGUMENTS')
        print(json.dumps({'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
                          'recipe_sha256': sha(canonical(recipe.recipe())), **recipe.FLAGS}))
        return 0
    recipe.validate_execution_pins(pins)
    need(args.output is not None and args.pilot_review is not None and args.xlsr_review is not None,
         'EXPLICIT_OUTPUT_AND_EVIDENCE_REQUIRED')
    if args.generate:
        need(args.source_parent is None and args.shard is not None, 'EXPLICIT_SHARD_NO_OLD_AUDIO')
        value = generate(args.output, pins, args.shard, args.pilot_review, args.xlsr_review)
    else:
        need(args.source_parent is not None and args.shard is None, 'EXPLICIT_NINE_SHARD_SOURCE_REQUIRED')
        value = observe(args.observe, args.source_parent, args.output, pins, args.pilot_review, args.xlsr_review)
    print(json.dumps({k: v for k, v in value.items() if k != 'rows'}, ensure_ascii=False, sort_keys=True))
    return 2 if value['status'] == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
