"""Consume the completed Qwen17 probe once; no regeneration or label admission.

The original generator is retained at SOURCE_HEAD. Successors reuse its exact
four WAVs for separately bounded blind-ASR and independent phoneme observations.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import wave
import zipfile

HERE = Path(__file__).resolve().parent
TRIGGER = HERE.parent / '.github/triggers/english-ipa-qwen17-final-ipa-20261116.json'
POLICY = 'english-ipa-qwen17-final-ipa-probe-v1'
ASR_POLICY = 'english-final-ipa-qwen17-blind-asr-v1'
MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
REV = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
SOURCE_HEAD = '5273e92135f71245dca531221f2f51f1415113c7'
SOURCE_RUN = 43793684582
SOURCE_ARTIFACT = 14193452864
SOURCE_SHA = 'e737354ad8ed69d511f061781f5b6cc3c86ff1ea70ba9c72084c5ac4cd1c738f'
LOCK_SHA = 'be0906da0f79d8af218e2cbbdd269992c428232dd8bb96cdca8359d337396da7'
BACKEND_BLOB = 'f3bf07e6237dfaeebf7a92d49a821ac0efcdc09b'
PHONE_POLICY = 'english-final-ipa-qwen17-independent-phones-v1'
PHONE_PREDECESSOR = 'e844bfbff4d579397eb93de0f2e27bf38fe1af8a'
PHONE_BACKEND_BLOB = '016df0222bf74f356399c30a9d7049b6ba5d4480'
PHONE_MODEL = 'bookbot/wav2vec2-ljspeech-gruut'
PHONE_REV = '7be1e6f27c8163ce790688d62fd9ce1731177c2e'
PHONE_WEIGHTS_SHA = '9bc8cd20e1348fc384ba4b9cd60a5c761dbb3f0b7071d1bdc33331ed1b358f29'
PROVIDERS = ('qwen', 'whisper', 'parakeet')
# Cell, requested phrase, preset, frame count, original WAV SHA, PCM SHA.
ROSTER = (
 ('enipa-6f78627d1f7517b32f22603a', 'Hey Mero', 'aiden', 20480,
  'dcc263af734f33f563c5cb6f71687d67a8b94bd8e882b36684042b31202c200a',
  '5959b0ffbfdeaaa9689e6328006f990155134802945fd964f28018c8cce0e723'),
 ('enipa-abdd93d1968b502793ca8548', 'Hey Mero', 'ryan', 16640,
  '92c71d03069e699c85e448066163b8e682944d3147cfc877f646c3162a8e1804',
  '700cf60dcc34f12d88a59e2e8252767f001fb3ec9a6ab5fb7e5124d756facad0'),
 ('enipa-2166f344d7858e6c40de6ea6', 'Hey Vosen', 'aiden', 11520,
  'd4d72b357d5078d77c96c31b94f9fc41e010c17e850e92be5ddc4904adec7aee',
  '1aa87c92786698d3ce5d6e41b68ff4e64271e61d45370ab51d71f60a1b26cc10'),
 ('enipa-2dd614fa331bb5096e46466a', 'Hey Vosen', 'ryan', 17920,
  'afa1160bc7e8c18f225522eddac72b10e79613b233490c11810f7185174398a5',
  '8cacc21ac162941025156f6134e7d88261b7856d35716e2931c778792d1fb812'),
)
IPAS = {'Hey Mero': '/heɪ ˈmiːroʊ/', 'Hey Vosen': '/heɪ ˈvoʊzən/'}
FALSE_FLAGS = {'training_admitted': False, 'human_gold': False,
               'pronunciation_verified': False, 'model_promotion_allowed': False}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, 'DUPLICATE_JSON_KEY')
        out[key] = value
    return out


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def regular(path, limit):
    path = Path(path)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_PATH')
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode)
         and path.stat().st_size <= limit, 'FILE_OR_SIZE')
    with path.open('rb') as src:
        data = src.read(limit + 1)
    need(len(data) <= limit, 'FILE_SIZE_CHANGED')
    return data


def once(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as dst:
        dst.write(raw)
        dst.flush()
        os.fsync(dst.fileno())


def read_trigger():
    x = loads(regular(TRIGGER, 16384))
    need(x.get('schema') == POLICY and x.get('model_id') == MODEL and
         x.get('model_revision') == REV and x.get('max_calls') == 4 and
         x.get('asr_calls') == 0 and x.get('no_retry') is True and
         x.get('training_admitted') is False and x.get('human_gold') is False,
         'ORIGINAL_TRIGGER_IDENTITY')
    need([(c['phrase'], c['speaker'], c['ipa']) for c in x['cells']] ==
         [(r[1], r[2], IPAS[r[1]]) for r in ROSTER], 'ORIGINAL_TARGET_ROSTER')
    return x


def source_cells(path):
    """Authenticate before returning any bytes; never extract arbitrary members."""
    raw = regular(path, 1024 * 1024)
    need(sha(raw) == SOURCE_SHA, 'SOURCE_ZIP_SHA')
    prefix = 'en-ipa-probe/'
    names = {prefix + x for x in ('started.json', 'result.json', 'model-lock.json', 'source-plan.json')}
    names.add('en-ipa-pip-freeze.txt')
    for row in ROSTER:
        names.update(prefix + 'cells/' + row[0] + '/' + x for x in
                     ('audio-16k.wav', 'native-f32.wav', 'started.json', 'completed.json'))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        need(len(infos) == len(names) and {i.filename for i in infos} == names and
             all(not i.is_dir() and not i.flag_bits & 1 and
                 stat.S_IFMT(i.external_attr >> 16) in (0, stat.S_IFREG)
                 and i.file_size <= 1024 * 1024 for i in infos) and
             sum(i.file_size for i in infos) <= 4 * 1024 * 1024, 'SOURCE_MEMBER_SET')
        summary = loads(z.read(prefix + 'result.json'))
        plan = loads(z.read(prefix + 'source-plan.json'))
        started = loads(z.read(prefix + 'started.json'))
        lock_raw = z.read(prefix + 'model-lock.json')
        lock = loads(lock_raw)
        need(summary.get('schema') == POLICY and summary.get('status') == 'COMPLETE' and
             summary.get('attempts') == summary.get('completed') == 4 and
             summary.get('asr_calls') == 0 and summary.get('training_admitted') is False and
             summary.get('human_gold') is False, 'SOURCE_TERMINAL_STATUS')
        need(str(started.get('run_id')) == str(SOURCE_RUN) and
             started.get('source_sha') == SOURCE_HEAD and sha(lock_raw) == LOCK_SHA and
             lock.get('model_id') == MODEL and lock.get('revision') == REV and
             plan.get('model') == MODEL and plan.get('revision') == REV and
             plan.get('model_lock_sha256') == LOCK_SHA and
             summary.get('model_lock_sha256') == LOCK_SHA and
             plan.get('training_admitted') is False and plan.get('human_gold') is False,
             'SOURCE_MODEL_OR_RUN_BINDING')
        need(len(plan['cells']) == len(summary['rows']) == 4, 'SOURCE_ROW_COUNT')
        need(len({c['cell_id'] for c in plan['cells']}) == 4 and
             len({c['cell_id'] for c in summary['rows']}) == 4, 'SOURCE_DUPLICATE_CELL')
        plans = {c['cell_id']: c for c in plan['cells']}
        reports = {c['cell_id']: c for c in summary['rows']}
        need(set(plans) == set(reports) == {r[0] for r in ROSTER}, 'SOURCE_CELL_SET')
        output = []
        for cid, phrase, speaker, frames, wav_sha, pcm_sha in ROSTER:
            folder = prefix + 'cells/' + cid + '/'
            receipt = loads(z.read(folder + 'completed.json'))
            need(plans[cid]['phrase'] == receipt.get('phrase') == phrase and
                 plans[cid]['speaker'] == receipt.get('speaker') == speaker and
                 plans[cid]['ipa'] == receipt.get('ipa_target') == IPAS[phrase] and
                 receipt.get('cell_id') == cid and receipt.get('training_admitted') is False and
                 receipt.get('human_gold') is False and receipt.get('pronunciation_verified') is False,
                 'SOURCE_CELL_RECEIPT')
            wav = z.read(folder + 'audio-16k.wav')
            audio = receipt['audio']
            with wave.open(io.BytesIO(wav), 'rb') as w:
                need((w.getnchannels(), w.getsampwidth(), w.getframerate(),
                      w.getnframes(), w.getcomptype()) == (1, 2, 16000, frames, 'NONE'), 'WAV_GEOMETRY')
                pcm = w.readframes(frames)
            need(len(pcm) == frames * 2 and sha(wav) == wav_sha == audio['wav_sha256'] ==
                 reports[cid]['wav_sha256'] and sha(pcm) == pcm_sha == audio['pcm_sha256'] ==
                 reports[cid]['pcm_sha256'] and len(wav) == audio['wav_bytes'] and
                 frames == audio['audio_frames'] and reports[cid]['status'] == 'GENERATED_UNADMITTED',
                 'SOURCE_WAV_PCM_BINDING')
            need(sha(z.read(folder + 'native-f32.wav')) == audio['native_sha256'], 'NATIVE_WAV_BINDING')
            output.append((cid, wav, frames, wav_sha))
    return output


def execution_guard(event, env):
    """Only the first protected-main successor push can consume this campaign."""
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and
         event.get('before') == SOURCE_HEAD and event.get('after') == env.get('GITHUB_SHA') and
         event.get('after') not in (None, SOURCE_HEAD) and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'ONE_SUCCESSOR_PUSH_ONLY')


def capture(provider, source_zip, output, *, backend=None):
    need(provider in PROVIDERS, 'PROVIDER')
    cells = source_cells(source_zip)
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_OUTPUT')
    if backend is None:
        event = loads(regular(Path(os.environ['GITHUB_EVENT_PATH']), 8 * 1024 * 1024))
        execution_guard(event, os.environ)
    output.mkdir(mode=0o700)
    once(output / 'started.json', {'schema': ASR_POLICY, 'provider': provider,
         'source_zip_sha256': SOURCE_SHA, 'source_run_id': SOURCE_RUN,
         'github_sha': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
         'call_budget': 4, 'retry_count': 0, 'test_backend': backend is not None, **FALSE_FLAGS})
    model_root = output / 'model'
    model_root.mkdir(mode=0o700)
    injected_backend = backend is not None
    attempts, rows, failure = 0, [], None
    provenance = None
    try:
        if backend is None:
            source = regular(HERE / 'english_seven_voice_asr_once.py', 128 * 1024)
            need(hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
                 == BACKEND_BLOB, 'REUSED_BACKEND_SOURCE_CHANGED')
            sys.path.insert(0, str(HERE))
            import english_seven_voice_asr_once as seven
            provenance = list(seven.PROVENANCE[provider])
            backend = seven._load(provider, model_root)
        for i, (cid, wav, frames, wav_sha) in enumerate(cells):
            folder = output / ('observation-%02d' % i)
            folder.mkdir(mode=0o700)
            # Decoder sees an anonymous WAV, never target text, IPA or another transcript.
            audio_path = folder / 'audio.wav'
            with audio_path.open('xb') as dst:
                dst.write(wav)
            once(folder / 'started.json', {'cell_id': cid, 'wav_sha256': wav_sha, 'attempt': 1})
            attempts += 1
            try:
                text = backend(audio_path, folder)
                need(type(text) is str and text.strip() and len(text) <= 512 and
                     not any(ord(c) < 32 for c in text), 'ASR_TEXT_INVALID')
                row = {'cell_id': cid, 'wav_sha256': wav_sha, 'frames': frames,
                       'status': 'COMPLETE', 'text': text, 'decoder_completed': True}
                once(folder / 'completed.json', {**row, **FALSE_FLAGS})
            except Exception as exc:
                row = {'cell_id': cid, 'wav_sha256': wav_sha, 'status': 'FAILED_NO_RETRY',
                       'error_type': type(exc).__name__}
                once(folder / 'failed.json', row)
            finally:
                audio_path.unlink()
            rows.append(row)
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        completed = sum(r['status'] == 'COMPLETE' for r in rows)
        result = {'schema': ASR_POLICY, 'status': 'COMPLETE' if completed == 4 and failure is None
                  else 'INCOMPLETE_NO_RETRY', 'provider': provider, 'provenance': provenance,
                  'source_run_id': SOURCE_RUN, 'source_artifact_id': SOURCE_ARTIFACT,
                  'source_zip_sha256': SOURCE_SHA, 'asr_attempts': attempts, 'completed_asr': completed,
                  'error_type': failure, 'retry_count': 0, 'source_audio_count': 4,
                  'test_backend': injected_backend, 'rows': rows, **FALSE_FLAGS}
        once(output / 'summary.json', result)
    return result


def combine(parent, output):
    """Join authenticated observations; spelling agreement never grants training."""
    parent, output = Path(parent), Path(output)
    need(not output.exists(), 'COMBINED_CREATE_ONLY')
    inputs, observations = {}, {}
    for provider in PROVIDERS:
        raw = regular(parent / provider / 'summary.json', 65536)
        value = loads(raw)
        need(value.get('schema') == ASR_POLICY and value.get('provider') == provider and
             value.get('source_run_id') == SOURCE_RUN and value.get('source_artifact_id') == SOURCE_ARTIFACT and
             value.get('source_zip_sha256') == SOURCE_SHA and value.get('status') == 'COMPLETE' and
             value.get('asr_attempts') == value.get('completed_asr') == 4 and
             value.get('test_backend') is False and
             all(value.get(k) is False for k in FALSE_FLAGS), 'ASR_SUMMARY_IDENTITY')
        rows = value.get('rows', [])
        need(len(rows) == 4 and len({r['cell_id'] for r in rows}) == 4, 'ASR_ROWS')
        by_id = {r['cell_id']: r for r in rows}
        need(set(by_id) == {r[0] for r in ROSTER}, 'ASR_CELL_SET')
        for cid, _, _, frames, wav_sha, _ in ROSTER:
            row = by_id[cid]
            need(row.get('wav_sha256') == wav_sha and row.get('frames') == frames and
                 row.get('status') == 'COMPLETE' and row.get('decoder_completed') is True and
                 type(row.get('text')) is str and bool(row['text'].strip()), 'ASR_ROW_BINDING')
            observations.setdefault(cid, {})[provider] = row['text']
        inputs[provider] = sha(raw)
    result = {'schema': ASR_POLICY, 'status': 'THREE_ASR_COMPLETE_REVIEW_REQUIRED',
              'source_run_id': SOURCE_RUN, 'source_artifact_id': SOURCE_ARTIFACT,
              'source_zip_sha256': SOURCE_SHA, 'input_summary_sha256': inputs,
              'asr_calls': 12, 'cells': [
                  {'cell_id': r[0], 'requested_phrase': r[1], 'target_ipa': IPAS[r[1]],
                   'source_preset': r[2], 'wav_sha256': r[4], 'transcripts': observations[r[0]],
                   'acoustic_review': 'PENDING', **FALSE_FLAGS} for r in ROSTER],
              'machine_observations_are_not_acoustic_gold': True, **FALSE_FLAGS}
    once(output, result)
    return result


def phone_execution_guard(event, env):
    """The new phoneme operation has its own first-successor bound, not an ASR replay."""
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and
         event.get('before') == PHONE_PREDECESSOR and
         event.get('after') == env.get('GITHUB_SHA') and
         event.get('after') not in (None, PHONE_PREDECESSOR) and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False,
         'PHONE_ONE_SUCCESSOR_PUSH_ONLY')


def capture_phones(source_zip, output, *, backend=None):
    """Four verified anonymous WAVs, one phone-model load, no target input or retries.

    The injected callable is exclusively an offline test seam. Its output is
    explicitly simulated and cannot be confused with completed real inference.
    """
    cells = source_cells(source_zip)
    need(len(cells) == 4, 'PHONE_SOURCE_COUNT')
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_OUTPUT')
    injected_backend = backend is not None
    if not injected_backend:
        event = loads(regular(Path(os.environ['GITHUB_EVENT_PATH']), 8 * 1024 * 1024))
        phone_execution_guard(event, os.environ)
    output.mkdir(mode=0o700)
    identity = {'schema': PHONE_POLICY, 'source_run_id': SOURCE_RUN,
                'source_artifact_id': SOURCE_ARTIFACT, 'source_zip_sha256': SOURCE_SHA,
                'model_id': PHONE_MODEL, 'model_revision': PHONE_REV,
                'model_safetensors_sha256': PHONE_WEIGHTS_SHA,
                'test_backend': injected_backend, 'call_budget': 4, 'retry_count': 0,
                'source_audio_count': 4, 'machine_observation_only': True,
                'new_tts_calls': 0, 'new_asr_calls': 0,
                'new_kws_model_forwards': 0, 'new_optimizer_steps': 0, **FALSE_FLAGS}
    once(output / 'started.json', {**identity, 'github_sha': os.environ.get('GITHUB_SHA'),
                                 'run_id': os.environ.get('GITHUB_RUN_ID')})
    model_root = output / 'private-model'
    attempts, rows, failure, model_manifest = 0, [], None, None
    try:
        if not injected_backend:
            source = regular(HERE / 'english63_independent_ipa.py', 128 * 1024)
            need(hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
                 == PHONE_BACKEND_BLOB, 'REUSED_PHONE_BACKEND_SOURCE_CHANGED')
            sys.path.insert(0, str(HERE))
            import english63_independent_ipa as ipa
            need((ipa.MODEL_ID, ipa.MODEL_REV, ipa.WEIGHTS_SHA256) ==
                 (PHONE_MODEL, PHONE_REV, PHONE_WEIGHTS_SHA), 'PHONE_BACKEND_MODEL_IDENTITY')
            model_root.mkdir(mode=0o700)
            processor, model, torch, asset = ipa.load_model_from_pinned_assets(model_root)
            model_manifest = asset['manifest_sha256']
            once(output / 'model-lock.json', asset)
            backend = lambda wav: ipa.predict_phone(wav, processor, model, torch)
        for i, (cid, wav, frames, wav_sha) in enumerate(cells):
            need(attempts < 4, 'PHONE_FORWARD_BUDGET')
            folder = output / ('observation-%02d' % i)
            folder.mkdir(mode=0o700)
            audio_path = folder / 'audio.wav'
            with audio_path.open('xb') as dst:
                dst.write(wav)
            row = {'cell_id': cid, 'wav_sha256': wav_sha, 'frames': frames,
                   'test_backend': injected_backend, 'target_is_not_passed_to_model': True,
                   **FALSE_FLAGS}
            once(folder / 'started.json', {**row, 'attempt': 1})
            attempts += 1
            try:
                # Only the WAV path reaches the decoder; target metadata is joined later.
                raw_tokens, phone_tokens, observed_frames = backend(audio_path)
                need(type(observed_frames) is int and observed_frames == frames,
                     'PHONE_FRAME_BINDING')
                need(all(type(tokens) is list and 2 <= len(tokens) <= 300 and
                         all(type(t) is str and 0 < len(t) <= 20 and
                             not any(ord(c) < 32 for c in t) for t in tokens)
                         for tokens in (raw_tokens, phone_tokens)), 'PHONE_TOKEN_SHAPE')
                row.update(status='COMPLETE', raw_ipa_tokens=raw_tokens,
                           stressless_phone_tokens=phone_tokens, decoder_completed=True)
                once(folder / 'completed.json', row)
            except BaseException as exc:
                row.update(status='FAILED_NO_RETRY', error_type=type(exc).__name__)
                once(folder / 'failed.json', row)
                if not isinstance(exc, Exception):
                    raise
            finally:
                rows.append(row)
                audio_path.unlink()
    except BaseException as exc:
        failure = type(exc).__name__
        if not isinstance(exc, Exception):
            raise
    finally:
        completed = sum(r.get('status') == 'COMPLETE' for r in rows)
        status = 'COMPLETE' if completed == 4 and failure is None else 'INCOMPLETE_NO_RETRY'
        if injected_backend and status == 'COMPLETE':
            status = 'SIMULATED_COMPLETE'
        result = {**identity, 'status': status, 'phoneme_attempts': attempts,
                  'completed_phonemes': completed, 'error_type': failure,
                  'model_manifest_sha256': model_manifest, 'rows': rows,
                  'limitations': ['Single-narrator/Gruut model domain bias',
                                  'Machine phonemes do not establish pronunciation or prosodic gold']}
        once(output / 'summary.json', result)
        if model_root.exists():
            shutil.rmtree(model_root)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--source-zip', type=Path)
    cli.add_argument('--provider', choices=PROVIDERS)
    cli.add_argument('--combine', type=Path)
    cli.add_argument('--execute', action='store_true', help='Retired TTS mode; always rejected')
    cli.add_argument('--execute-asr', action='store_true')
    cli.add_argument('--execute-phones', action='store_true')
    args = cli.parse_args(argv)
    need(not args.execute, 'TTS_ALREADY_CONSUMED_DO_NOT_REGENERATE')
    need(not (args.execute_asr and args.execute_phones), 'EXCLUSIVE_EXECUTION_MODE')
    if args.combine:
        need(args.output is not None and not args.execute_asr and not args.execute_phones and
             args.provider is None,
             'COMBINE_ARGUMENTS')
        result = combine(args.combine, args.output)
    elif args.execute_asr:
        need(args.provider is not None and args.source_zip is not None and args.output is not None,
             'ASR_ARGUMENTS')
        result = capture(args.provider, args.source_zip, args.output)
    elif args.execute_phones:
        need(args.provider is None and args.source_zip is not None and args.output is not None,
             'PHONE_ARGUMENTS')
        result = capture_phones(args.source_zip, args.output)
    elif args.source_zip:
        need(args.provider is None, 'SOURCE_VERIFY_ONLY')
        cells = source_cells(args.source_zip)
        result = {'status': 'EXACT_FOUR_WAVS_VERIFIED_NO_MODEL', 'source_audio_count': len(cells),
                  'asr_calls': 0, **FALSE_FLAGS}
    else:
        read_trigger()
        result = {'status': 'SOURCE_CONTRACT_ONLY', 'asr_calls': 0, **FALSE_FLAGS}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get('status') == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
