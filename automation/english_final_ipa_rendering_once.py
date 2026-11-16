"""Four new orthographic renderings followed by existing audio-only observers.

Canonical wake names and IPA stay unchanged. Meero/Vozen are synthesis inputs,
not new wake labels. This bounded research operation never admits training rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from english_ipa_qwen17_final_ipa_probe_once import need, regular, loads, once, sha, FALSE_FLAGS

POLICY = 'english-final-ipa-orthographic-rendering-v1'
PREDECESSOR = 'b0810b837c8c31d66bfbfeb1181e7b7585b12f21'
MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
REV = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
PINS = {
    'english_ipa_qwen17_probe_once': '9204ddf194fdf2f1d15852f5b0283012c2706280',
    'english_seven_voice_asr_once': 'f3bf07e6237dfaeebf7a92d49a821ac0efcdc09b',
    'english63_independent_ipa': '016df0222bf74f356399c30a9d7049b6ba5d4480',
    'english_ipa_qwen17_final_ipa_probe_once': 'c8de4cfd4b743e787ae1b8d9728375aef28babba',
}
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
RENDERINGS = (
    ('Hey Mero', 'Hey Meero.', '/heɪ ˈmiːroʊ/',
     'Say only Hey Meero. Use MEE-roh, /miːroʊ/, with the long ee vowel in meet. '
     'Keep Hey light and unstressed; stress MEE, the first syllable of the name. '
     'Use natural American English. Do not add any words.'),
    ('Hey Vosen', 'Hey Vozen.', '/heɪ ˈvoʊzən/',
     'Say only Hey Vozen. Use VOH-zuhn, /voʊzən/, with a voiced z as in zoo. '
     'Keep Hey light and unstressed; stress VOH, the first syllable of the name. '
     'Use natural American English. Do not add any words.'),
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def recipe():
    cells = []
    for phrase, text, ipa, instruction in RENDERINGS:
        for voice in ('aiden', 'ryan'):
            row = {'canonical_phrase': phrase, 'synthesis_text': text, 'target_ipa': ipa,
                   'speaker': voice, 'instruction': instruction}
            row['cell_id'] = 'enrender-' + sha(canonical({'policy': POLICY, 'revision': REV,
                                                        'cell': row}))[:24]
            cells.append(row)
    return {'schema': POLICY, 'model_id': MODEL, 'model_revision': REV,
            'module_pins': PINS, 'max_tts_calls': 4, 'max_calls_per_observer': 4,
            'no_retry': True, 'cells': cells, **FALSE_FLAGS}


def check_pins():
    for name, expected in PINS.items():
        raw = regular(HERE / (name + '.py'), 256 * 1024)
        need(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected,
             'REUSED_MODULE_CHANGED:' + name)


def execution_guard(event, env):
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and event.get('before') == PREDECESSOR and
         event.get('after') == env.get('GITHUB_SHA') and
         type(event.get('after')) is str and len(event['after']) == 40 and
         all(c in '0123456789abcdef' for c in event['after']) and
         event['after'] != PREDECESSOR and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False and
         str(env.get('GITHUB_RUN_ID', '')).isdigit(), 'ONE_PUBLIC_MAIN_SUCCESSOR_ONLY')


def guard():
    execution_guard(loads(regular(Path(os.environ['GITHUB_EVENT_PATH']), 8 * 1024 * 1024)), os.environ)
    check_pins()


def identity(test_backend):
    return {'schema': POLICY, 'recipe_sha256': sha(canonical(recipe())),
            'code_sha256': sha(regular(Path(__file__), 256 * 1024)),
            'github_sha': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
            'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'test_backend': test_backend,
            'retry_count': 0, **FALSE_FLAGS}


def new_output(path):
    path = Path(path)
    need(path.is_absolute() and path.parent.is_dir() and not path.exists() and
         not any(p.is_symlink() for p in (path, *path.parents)), 'CREATE_ONLY_OUTPUT')
    path.mkdir(mode=0o700)
    return path


def wave_identity(path):
    raw = regular(path, 1024 * 1024)
    with wave.open(io.BytesIO(raw), 'rb') as stream:
        frames = stream.getnframes()
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
              stream.getcomptype()) == (1, 2, 16000, 'NONE') and 400 <= frames <= 192000,
             'SOURCE_WAV_GEOMETRY')
        pcm = stream.readframes(frames)
    need(len(pcm) == frames * 2, 'SOURCE_PCM_LENGTH')
    return {'wav_sha256': sha(raw), 'pcm_sha256': sha(pcm), 'frames': frames, 'wav_bytes': len(raw)}


def generate(output, *, runtime=None):
    """runtime is an offline test seam: (backend, wave writer, model lock)."""
    simulated = runtime is not None
    if not simulated:
        guard()
    output = new_output(output)
    base = identity(simulated)
    once(output / 'started.json', {**base, 'operation': 'generate', 'call_budget': 4})
    attempts, rows, error, lock_sha = 0, [], None, None
    try:
        if runtime is None:
            tts = importlib.import_module('english_ipa_qwen17_probe_once')
            need((tts.MODEL, tts.REV) == (MODEL, REV), 'TTS_MODEL_IDENTITY')
            model_folder, lock = tts.materialize(output)
            runtime = (tts.Backend(model_folder), tts.capture.media(), lock)
        backend, writer, lock = runtime
        need(lock.get('model_id') == MODEL and lock.get('revision') == REV, 'MODEL_LOCK_IDENTITY')
        once(output / 'model-lock.json', lock)
        lock_sha = sha(regular(output / 'model-lock.json', 1024 * 1024))
        once(output / 'source-plan.json', {**recipe(), 'model_lock_sha256': lock_sha})
        for cell in recipe()['cells']:
            need(attempts < 4, 'TTS_CALL_BUDGET')
            folder = output / 'cells' / cell['cell_id']
            folder.mkdir(parents=True, mode=0o700)
            once(folder / 'started.json', {**base, 'cell_id': cell['cell_id'], 'attempt': 1})
            attempts += 1
            row = {'cell_id': cell['cell_id']}
            try:
                # The old Backend is reused only as a synthesizer, never its old execute/trigger.
                render_cell = {'cell_id': cell['cell_id'], 'phrase': cell['synthesis_text'],
                               'speaker': cell['speaker'], 'instruction': cell['instruction']}
                samples, rate = backend.synthesize(render_cell)
                technical = writer.quality(samples, rate)
                audio = writer._write_waves(folder, samples, rate, technical)
                observed = wave_identity(folder / 'audio-16k.wav')
                need(observed['wav_sha256'] == audio['wav_sha256'] and
                     observed['pcm_sha256'] == audio['pcm_sha256'] and
                     observed['frames'] == audio['audio_frames'] and
                     observed['wav_bytes'] == audio['wav_bytes'] and
                     sha(regular(folder / 'native-f32.wav', 4 * 1024 * 1024)) == audio['native_sha256'],
                     'GENERATED_AUDIO_RECEIPT')
                row.update(status='GENERATED_UNADMITTED', **observed)
                once(folder / 'completed.json', {**base, **cell, **row, 'audio': audio,
                                                'technical': technical})
            except BaseException as exc:
                row.update(status='FAILED_NO_RETRY', error_type=type(exc).__name__)
                once(folder / 'failed.json', {**base, **row})
                if not isinstance(exc, Exception):
                    raise
            finally:
                rows.append(row)
    except BaseException as exc:
        error = type(exc).__name__
        if not isinstance(exc, Exception):
            raise
    finally:
        completed = sum(r.get('status') == 'GENERATED_UNADMITTED' for r in rows)
        status = 'COMPLETE' if completed == 4 and error is None else 'INCOMPLETE_NO_RETRY'
        if simulated and status == 'COMPLETE':
            status = 'SIMULATED_COMPLETE'
        result = {**base, 'status': status, 'tts_attempts': attempts, 'generated_audio': completed,
                  'model_lock_sha256': lock_sha, 'rows': rows, 'error_type': error,
                  'asr_calls': 0, 'phoneme_calls': 0, 'optimizer_steps': 0}
        once(output / 'result.json', result)
        if (output / 'model').exists():
            shutil.rmtree(output / 'model')
    return result


def source_cells(parent, *, allow_simulated=False):
    """Read the exact direct artifact root; bind its plan/receipts/audio to this run."""
    parent = Path(parent)
    base = identity(False)
    started = loads(regular(parent / 'started.json', 32768))
    result = loads(regular(parent / 'result.json', 65536))
    simulated = started.get('test_backend') is True
    need(allow_simulated or not simulated, 'SIMULATED_SOURCE_REJECTED')
    for obj in (started, result):
        need(all(obj.get(k) == base[k] for k in
                 ('schema', 'recipe_sha256', 'code_sha256', 'github_sha', 'run_id', 'run_attempt')) and
             all(obj.get(k) is False for k in FALSE_FLAGS) and
             obj.get('test_backend') is simulated and obj.get('retry_count') == 0,
             'SOURCE_RUN_OR_RECIPE_IDENTITY')
    need(started.get('operation') == 'generate' and started.get('call_budget') == 4 and
         result.get('status') == ('SIMULATED_COMPLETE' if simulated else 'COMPLETE') and
         result.get('tts_attempts') == result.get('generated_audio') == 4 and
         result.get('asr_calls') == result.get('phoneme_calls') == result.get('optimizer_steps') == 0,
         'SOURCE_NOT_COMPLETE')
    lock_raw = regular(parent / 'model-lock.json', 1024 * 1024)
    lock_sha = sha(lock_raw)
    lock = loads(lock_raw)
    need(lock.get('model_id') == MODEL and lock.get('revision') == REV and
         result.get('model_lock_sha256') == lock_sha, 'SOURCE_MODEL_LOCK')
    plan_raw = regular(parent / 'source-plan.json', 65536)
    need(loads(plan_raw) == {**recipe(), 'model_lock_sha256': lock_sha}, 'SOURCE_PLAN_CHANGED')
    rows = result.get('rows', [])
    need(type(rows) is list and len(rows) == 4 and len({r['cell_id'] for r in rows}) == 4,
         'SOURCE_ROW_ROSTER')
    by_id = {r['cell_id']: r for r in rows}
    cells = recipe()['cells']
    need(set(by_id) == {c['cell_id'] for c in cells}, 'SOURCE_CELL_SET')
    need({p.name for p in (parent / 'cells').iterdir()} == set(by_id), 'SOURCE_CELL_DIRECTORIES')
    output = []
    for cell in cells:
        folder = parent / 'cells' / cell['cell_id']
        receipt = loads(regular(folder / 'completed.json', 65536))
        start = loads(regular(folder / 'started.json', 32768))
        need(not (folder / 'failed.json').exists() and start.get('attempt') == 1 and
             start.get('cell_id') == cell['cell_id'], 'SOURCE_CELL_ATTEMPT')
        for obj in (receipt, start):
            need(all(obj.get(k) == started[k] for k in base) and
                 obj.get('test_backend') is simulated, 'SOURCE_CELL_RUN_BINDING')
        need(all(receipt.get(k) == v for k, v in cell.items()) and
             receipt.get('status') == by_id[cell['cell_id']].get('status') == 'GENERATED_UNADMITTED',
             'SOURCE_CELL_RENDERING')
        wav = folder / 'audio-16k.wav'
        observed = wave_identity(wav)
        audio = receipt['audio']
        need(all(receipt.get(k) == by_id[cell['cell_id']].get(k) == v for k, v in observed.items()) and
             observed['wav_sha256'] == audio['wav_sha256'] and
             observed['pcm_sha256'] == audio['pcm_sha256'] and
             observed['frames'] == audio['audio_frames'] and
             observed['wav_bytes'] == audio['wav_bytes'] and
             sha(regular(folder / 'native-f32.wav', 4 * 1024 * 1024)) == audio['native_sha256'],
             'SOURCE_WAV_PCM_RECEIPT')
        output.append((cell, wav, observed))
    need(len({r[2]['wav_sha256'] for r in output}) == 4 and
         len({r[2]['pcm_sha256'] for r in output}) == 4, 'SOURCE_DUPLICATE_AUDIO')
    binding = {'source_plan_sha256': sha(plan_raw), 'model_lock_sha256': lock_sha,
               'source_cells': [{'cell_id': c['cell_id'], **info} for c, _, info in output]}
    return binding, output


def observe(provider, source_parent, output, *, backend=None):
    need(provider in PROVIDERS, 'PROVIDER')
    simulated = backend is not None
    if not simulated:
        guard()
    binding, cells = source_cells(source_parent, allow_simulated=simulated)
    output = new_output(output)
    base = {**identity(simulated), **binding, 'provider': provider}
    once(output / 'started.json', {**base, 'call_budget': 4})
    model_root = output / 'private-model'
    attempts, rows, error, provenance = 0, [], None, None
    try:
        if backend is None:
            model_root.mkdir(mode=0o700)
            if provider == 'phones':
                ipa = importlib.import_module('english63_independent_ipa')
                processor, model, torch, asset = ipa.load_model_from_pinned_assets(model_root)
                once(output / 'model-lock.json', asset)
                provenance = [ipa.MODEL_ID, ipa.MODEL_REV, ipa.WEIGHTS_SHA256]
                backend = lambda wav, _: ipa.predict_phone(wav, processor, model, torch)
            else:
                seven = importlib.import_module('english_seven_voice_asr_once')
                provenance = list(seven.PROVENANCE[provider])
                backend = seven._load(provider, model_root)
        for i, (cell, wav, info) in enumerate(cells):
            need(attempts < 4, 'OBSERVER_CALL_BUDGET')
            folder = output / ('observation-%02d' % i)
            folder.mkdir(mode=0o700)
            anonymous = folder / 'audio.wav'
            with anonymous.open('xb') as target:
                target.write(regular(wav, 1024 * 1024))
            row = {'cell_id': cell['cell_id'], **info, **FALSE_FLAGS}
            once(folder / 'started.json', {**row, 'attempt': 1, 'test_backend': simulated})
            attempts += 1
            try:
                value = backend(anonymous, folder)
                if provider == 'phones':
                    raw, phones, frames = value
                    need(type(frames) is int and frames == info['frames'], 'PHONE_FRAME_BINDING')
                    need(all(type(ts) is list and 2 <= len(ts) <= 300 and
                             all(type(t) is str and 0 < len(t) <= 20 and
                                 not any(ord(c) < 32 for c in t) for t in ts)
                             for ts in (raw, phones)), 'PHONE_TOKENS')
                    row.update(raw_ipa_tokens=raw, stressless_phone_tokens=phones)
                else:
                    need(type(value) is str and value.strip() and len(value) <= 512 and
                         not any(ord(c) < 32 for c in value), 'ASR_TEXT')
                    row['text'] = value
                row.update(status='COMPLETE', decoder_completed=True)
                once(folder / 'completed.json', {**row, 'test_backend': simulated})
            except BaseException as exc:
                row.update(status='FAILED_NO_RETRY', error_type=type(exc).__name__)
                once(folder / 'failed.json', {**row, 'test_backend': simulated})
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
        status = 'COMPLETE' if completed == 4 and error is None else 'INCOMPLETE_NO_RETRY'
        if simulated and status == 'COMPLETE':
            status = 'SIMULATED_COMPLETE'
        result = {**base, 'status': status, 'attempts': attempts, 'completed': completed,
                  'provenance': provenance, 'rows': rows, 'error_type': error,
                  'new_tts_calls': 0, 'new_optimizer_steps': 0}
        once(output / 'summary.json', result)
        if model_root.exists():
            shutil.rmtree(model_root)
    return result


def combine(parent, output):
    """Join same-run authenticated observations without labeling or promotion."""
    parent, output = Path(parent), Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'COMBINED_CREATE_ONLY')
    common, values, hashes = None, {}, {}
    expected_base = identity(False)
    for provider in PROVIDERS:
        raw = regular(parent / provider / 'summary.json', 128 * 1024)
        value = loads(raw)
        need(all(value.get(k) == v for k, v in expected_base.items()) and
             value.get('test_backend') is False and all(value.get(k) is False for k in FALSE_FLAGS) and
             value.get('provider') == provider and value.get('status') == 'COMPLETE' and
             value.get('attempts') == value.get('completed') == 4 and
             value.get('new_tts_calls') == value.get('new_optimizer_steps') == 0,
             'OBSERVER_IDENTITY_OR_COMPLETION')
        binding = {k: value.get(k) for k in ('source_plan_sha256', 'model_lock_sha256', 'source_cells')}
        need(all(type(binding[k]) is str and len(binding[k]) == 64 and
                 all(c in '0123456789abcdef' for c in binding[k])
                 for k in ('source_plan_sha256', 'model_lock_sha256')), 'SOURCE_HASH_SHAPE')
        expected_plan = (json.dumps({**recipe(), 'model_lock_sha256': binding['model_lock_sha256']},
                                   ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
        need(binding['source_plan_sha256'] == sha(expected_plan), 'COMBINED_PLAN_BINDING')
        need(type(binding['source_cells']) is list and len(binding['source_cells']) == 4 and
             len({r['cell_id'] for r in binding['source_cells']}) == 4 and
             {r['cell_id'] for r in binding['source_cells']} == {c['cell_id'] for c in recipe()['cells']} and
             len({r['wav_sha256'] for r in binding['source_cells']}) == 4 and
             len({r['pcm_sha256'] for r in binding['source_cells']}) == 4,
             'OBSERVER_SOURCE_ROSTER')
        if common is None:
            common = binding
        need(binding == common, 'CROSS_PROVIDER_SOURCE_DRIFT')
        rows = value.get('rows', [])
        need(type(rows) is list and len(rows) == 4 and len({r['cell_id'] for r in rows}) == 4,
             'OBSERVER_ROW_ROSTER')
        by_id = {r['cell_id']: r for r in rows}
        need(set(by_id) == {r['cell_id'] for r in common['source_cells']}, 'OBSERVER_ROW_CELLS')
        for info in common['source_cells']:
            row = by_id[info['cell_id']]
            need(all(row.get(k) == v for k, v in info.items()) and row.get('status') == 'COMPLETE' and
                 row.get('decoder_completed') is True and all(row.get(k) is False for k in FALSE_FLAGS),
                 'OBSERVER_ROW_BINDING')
            if provider == 'phones':
                need(type(row.get('raw_ipa_tokens')) is list and
                     type(row.get('stressless_phone_tokens')) is list, 'PHONES_REQUIRED')
            else:
                need(type(row.get('text')) is str and bool(row['text'].strip()), 'ASR_REQUIRED')
        values[provider] = by_id
        hashes[provider] = sha(raw)
    result = {**expected_base, **common, 'status': 'FOUR_OBSERVERS_COMPLETE_REVIEW_REQUIRED',
              'input_summary_sha256': hashes, 'tts_calls': 4, 'asr_calls': 12, 'phoneme_calls': 4,
              'optimizer_steps': 0, 'machine_observations_are_not_acoustic_gold': True,
              'cells': [{**cell, 'wav_sha256': values['phones'][cell['cell_id']]['wav_sha256'],
                         'transcripts': {p: values[p][cell['cell_id']]['text'] for p in PROVIDERS[:3]},
                         'raw_ipa_tokens': values['phones'][cell['cell_id']]['raw_ipa_tokens'],
                         'stressless_phone_tokens': values['phones'][cell['cell_id']]['stressless_phone_tokens'],
                         'acoustic_review': 'PENDING', **FALSE_FLAGS} for cell in recipe()['cells']]}
    once(output, result)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument('--generate', action='store_true')
    modes.add_argument('--observe', choices=PROVIDERS)
    modes.add_argument('--combine', type=Path)
    cli.add_argument('--source-parent', type=Path)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args(argv)
    if args.generate:
        need(args.output is not None and args.source_parent is None, 'GENERATE_ARGUMENTS')
        result = generate(args.output)
    elif args.observe:
        need(args.source_parent is not None and args.output is not None, 'OBSERVE_ARGUMENTS')
        result = observe(args.observe, args.source_parent, args.output)
    elif args.combine:
        need(args.output is not None and args.source_parent is None, 'COMBINE_ARGUMENTS')
        guard()
        result = combine(args.combine, args.output)
    else:
        need(args.source_parent is None and args.output is None, 'CONTRACT_ARGUMENTS')
        check_pins()
        result = {'schema': POLICY, 'recipe_sha256': sha(canonical(recipe())),
                  'status': 'SOURCE_CONTRACT_ONLY', 'executed': 0, **FALSE_FLAGS}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get('status') == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
