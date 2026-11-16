"""Eight fixed existing WAVs, one independent XLSR phoneme observer, no replay.

Original and orthographically rendered samples retain their exact identities.
Machine phonemes neither establish pronunciation gold nor authorize expansion.
"""
from __future__ import annotations
import argparse
import hashlib
import io
from itertools import groupby
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import wave
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_ipa_qwen17_final_ipa_probe_once as original
need, regular, loads, once, sha = original.need, original.regular, original.loads, original.once, original.sha
FLAGS = original.FALSE_FLAGS
POLICY = 'english-final-ipa-eight-xlsr-observations-v1'
PREDECESSOR = '0641690f2af5468992485319d04ba7c2b4fe6a61'
SOURCE_HELPER_BLOB = 'c8de4cfd4b743e787ae1b8d9728375aef28babba'
MODEL = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft'
REVISION = '2c733782da5604684829819a5eb744c193fe9398'
WEIGHTS_BYTES = 1263535127
WEIGHTS_SHA = '04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551'
MODEL_FILES = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'vocab.json', 'pytorch_model.bin')
RENDER_RUN = 43809825038
RENDER_ARTIFACT = 14199609338
RENDER_ZIP_SHA = '93b47d8885bf592f219a658bd27090c9af3c1a80bc5a3cae98bb49b9af7b7661'
RENDER_PLAN_SHA = '022c87a4b32178c45c903b73d181cc6ba0e064b1cebdacc9ebfc8e299210bfd7'
RENDER_LOCK_SHA = '06abd719b12f4191e1552194fc386d5a0b813318d7f8dc4dac719c589c59ac32'
RENDER_CODE_SHA = 'cb0d71bea6caab54227835e1ebe86e6fbadd09004baf0ed913c6454535180e42'
RENDER_RECIPE_SHA = '6f8b94521335699fcaeb85ff6990c84fb614a243e69355c74c71310848207ad8'
RENDER_ROSTER = (
 ('enrender-41ea399c68e39d4dad0ab507', 'Hey Mero', 'aiden', 14080,
  '4ad2d37f20d6e5874994160b2c7eafc97c87ed890d3e65e596b8005caa469ade',
  '085a4c248592f7226ce964fe8cc902fb765d75f175fbdf086d8be1dcf9ac4e67'),
 ('enrender-1de7418fdc72eec5bed01900', 'Hey Mero', 'ryan', 15360,
  '6893e9163802a713f4b69c87a3065362a36280fb73971e19b8ab2d3b36a16ad2',
  'e39b53d0fe32e1e8c26c31e2c15c297ce399ef9629e5b34959ef0f9caded3274'),
 ('enrender-3d4ac8c02322a25a1606f9b6', 'Hey Vosen', 'aiden', 14080,
  '2831d3fa932c1af0b1a81d97bdbc522e050564a06df1680052ba3f7455dea7d7',
  '8c534fe9c4212a646f8bdf0afcd07e39e77462673dcc96f86c9816d43f0c03db'),
 ('enrender-f909bb55a16c6aedf0ddc466', 'Hey Vosen', 'ryan', 15360,
  'a432bdb68b53a990fcaf13898411c723f2aa03235dbc45eaa5416398bfeb1278',
  'f3615a2ade7c31a34694310f7cff8dc5701db886c021157c075546c98a4f7909'),
)


def check_helper():
    raw = regular(HERE / 'english_ipa_qwen17_final_ipa_probe_once.py', 128 * 1024)
    need(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() ==
         SOURCE_HELPER_BLOB, 'ORIGINAL_SOURCE_HELPER_CHANGED')


def execution_guard(event, env):
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and event.get('before') == PREDECESSOR and
         event.get('after') == env.get('GITHUB_SHA') and type(event.get('after')) is str and
         len(event['after']) == 40 and all(c in '0123456789abcdef' for c in event['after']) and
         event['after'] != PREDECESSOR and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'XLSR_FIRST_PUBLIC_MAIN_SUCCESSOR_ONLY')


def rendered_cells(path):
    raw = regular(path, 1024 * 1024)
    need(sha(raw) == RENDER_ZIP_SHA, 'RENDER_ZIP_SHA')
    names = {'started.json', 'source-plan.json', 'model-lock.json', 'result.json'}
    for row in RENDER_ROSTER:
        names.update('cells/' + row[0] + '/' + name for name in
                     ('started.json', 'completed.json', 'audio-16k.wav', 'native-f32.wav'))
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == len(names) and {x.filename for x in infos} == names and
             all(not x.is_dir() and not x.flag_bits & 1 and x.file_size <= 1024 * 1024 and
                 stat.S_IFMT(x.external_attr >> 16) in (0, stat.S_IFREG) for x in infos) and
             sum(x.file_size for x in infos) <= 4 * 1024 * 1024, 'RENDER_MEMBER_SET')
        plan_raw, lock_raw = archive.read('source-plan.json'), archive.read('model-lock.json')
        need(sha(plan_raw) == RENDER_PLAN_SHA and sha(lock_raw) == RENDER_LOCK_SHA, 'RENDER_PLAN_LOCK_SHA')
        plan, result = loads(plan_raw), loads(archive.read('result.json'))
        identity = {'schema': 'english-final-ipa-orthographic-rendering-v1',
                    'github_sha': PREDECESSOR, 'run_id': str(RENDER_RUN), 'run_attempt': '1',
                    'code_sha256': RENDER_CODE_SHA, 'recipe_sha256': RENDER_RECIPE_SHA}
        started = loads(archive.read('started.json'))
        for value in (started, result):
            need(all(value.get(k) == v for k, v in identity.items()) and
                 value.get('test_backend') is False and all(value.get(k) is False for k in FLAGS),
                 'RENDER_SOURCE_IDENTITY')
        need(result.get('status') == 'COMPLETE' and
             result.get('generated_audio') == result.get('tts_attempts') == 4 and
             result.get('asr_calls') == result.get('phoneme_calls') == result.get('optimizer_steps') == 0,
             'RENDER_NOT_COMPLETE')
        need(len(plan['cells']) == len(result['rows']) == 4 and
             len({c['cell_id'] for c in plan['cells']}) == len({r['cell_id'] for r in result['rows']}) == 4,
             'RENDER_ROW_ROSTER')
        cells, rows = {c['cell_id']: c for c in plan['cells']}, {r['cell_id']: r for r in result['rows']}
        need(set(cells) == set(rows) == {r[0] for r in RENDER_ROSTER}, 'RENDER_CELL_SET')
        output = []
        for cid, phrase, voice, frames, wav_sha, pcm_sha in RENDER_ROSTER:
            folder = 'cells/' + cid + '/'
            receipt = loads(archive.read(folder + 'completed.json'))
            cell_start = loads(archive.read(folder + 'started.json'))
            need(all(receipt.get(k) == v for k, v in identity.items()) and
                 receipt.get('cell_id') == cid and
                 receipt.get('test_backend') is False and all(receipt.get(k) is False for k in FLAGS) and
                 cell_start.get('cell_id') == cid and cell_start.get('attempt') == 1 and
                 cell_start.get('test_backend') is False and all(cell_start.get(k) is False for k in FLAGS) and
                 all(cell_start.get(k) == v for k, v in identity.items()), 'RENDER_CELL_RUN')
            need(receipt.get('canonical_phrase') == cells[cid]['canonical_phrase'] == phrase and
                 receipt.get('speaker') == cells[cid]['speaker'] == voice and
                 receipt.get('target_ipa') == cells[cid]['target_ipa'] == original.IPAS[phrase] and
                 receipt.get('status') == rows[cid].get('status') == 'GENERATED_UNADMITTED', 'RENDER_CELL_TARGET')
            wav = archive.read(folder + 'audio-16k.wav')
            with wave.open(io.BytesIO(wav), 'rb') as stream:
                need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                      stream.getnframes(), stream.getcomptype()) == (1, 2, 16000, frames, 'NONE'),
                     'RENDER_WAV_GEOMETRY')
                pcm = stream.readframes(frames)
            audio = receipt['audio']
            need(len(pcm) == frames * 2 and sha(wav) == wav_sha == audio['wav_sha256'] == rows[cid]['wav_sha256'] and
                 sha(pcm) == pcm_sha == audio['pcm_sha256'] == rows[cid]['pcm_sha256'] and
                 len(wav) == audio['wav_bytes'] and frames == audio['audio_frames'] and
                 sha(archive.read(folder + 'native-f32.wav')) == audio['native_sha256'], 'RENDER_WAV_PCM_BINDING')
            output.append((cid, wav, frames, wav_sha))
    return output


def source_cells(original_zip, rendered_zip):
    check_helper()
    output = []
    for group, cells, roster, run, artifact, digest in (
        ('original', original.source_cells(original_zip), original.ROSTER,
         original.SOURCE_RUN, original.SOURCE_ARTIFACT, original.SOURCE_SHA),
        ('rendered', rendered_cells(rendered_zip), RENDER_ROSTER,
         RENDER_RUN, RENDER_ARTIFACT, RENDER_ZIP_SHA),
    ):
        meta = {r[0]: r for r in roster}
        for cid, wav, frames, wav_sha in cells:
            r = meta[cid]
            output.append(({'cell_id': cid, 'source_group': group, 'source_run_id': run,
                            'source_artifact_id': artifact, 'source_zip_sha256': digest,
                            'canonical_phrase': r[1], 'speaker': r[2], 'target_ipa': original.IPAS[r[1]],
                            'synthesis_text': ({'Hey Mero': 'Hey Meero.', 'Hey Vosen': 'Hey Vozen.'}[r[1]]
                                               if group == 'rendered' else r[1]),
                            'frames': frames, 'wav_sha256': wav_sha, 'pcm_sha256': r[5]}, wav))
    need(len(output) == 8 and len({r[0]['cell_id'] for r in output}) == 8 and
         len({r[0]['wav_sha256'] for r in output}) == 8, 'EXACT_EIGHT_UNIQUE_SOURCES')
    return output


def ctc_tokens(ids, vocab, skip):
    reverse = {number: token for token, number in vocab.items()}
    need(type(ids) is list and 1 <= len(ids) <= 2000 and
         all(type(i) is int and i in reverse for i in ids), 'CTC_IDS')
    # Collapse adjacent IDs before deleting blank. A blank-separated repeat survives.
    tokens = [reverse[i] for i, _ in groupby(ids) if i not in skip]
    need(1 <= len(tokens) <= 300 and all(type(t) is str and 0 < len(t) <= 40 and
         not any(ord(c) < 32 for c in t) for t in tokens), 'CTC_TOKENS')
    return tokens


def load_runtime(folder):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL, revision=REVISION, allow_patterns=list(MODEL_FILES),
                      local_dir=str(folder), max_workers=2, token=False)
    manifest = {}
    for name in MODEL_FILES:
        path = folder / name
        need(path.is_file() and not path.is_symlink(), 'MODEL_FILE')
        h = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                h.update(block)
        manifest[name] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
    need(manifest['pytorch_model.bin'] == {'bytes': WEIGHTS_BYTES, 'sha256': WEIGHTS_SHA} and
         all(0 < manifest[n]['bytes'] <= 512 * 1024 for n in MODEL_FILES if n != 'pytorch_model.bin'),
         'MODEL_ASSET_IDENTITY')
    vocab = loads(regular(folder / 'vocab.json', 512 * 1024))
    config_json = loads(regular(folder / 'config.json', 512 * 1024))
    tokenizer = loads(regular(folder / 'tokenizer_config.json', 512 * 1024))
    special = loads(regular(folder / 'special_tokens_map.json', 512 * 1024))
    need(len(vocab) == 392 and all(type(v) is int for v in vocab.values()) and
         set(vocab.values()) == set(range(392)) and vocab.get('iː') == 30 and vocab.get('z') == 21 and
         config_json.get('pad_token_id') == 0 and config_json.get('vocab_size') == 392, 'XLSR_VOCAB')
    skip = {0}
    for value in list(special.values()) + [tokenizer.get('word_delimiter_token')]:
        token = value.get('content') if isinstance(value, dict) else value
        if type(token) is str and token in vocab:
            skip.add(vocab[token])
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    need(extractor.sampling_rate == 16000, 'XLSR_SAMPLE_RATE')
    config = Wav2Vec2Config.from_json_file(str(folder / 'config.json'))
    model = Wav2Vec2ForCTC(config)
    state = torch.load(str(folder / 'pytorch_model.bin'), map_location='cpu', weights_only=True)
    need(isinstance(state, dict) and all(type(k) is str and isinstance(v, torch.Tensor)
                                       for k, v in state.items()), 'TENSOR_STATE_DICT_ONLY')
    model.load_state_dict(state, strict=True)
    del state
    model.to('cpu').eval()
    def predict(wav):
        with wave.open(str(wav), 'rb') as stream:
            frames = stream.getnframes()
            need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                 (1, 2, 16000, 'NONE') and 400 <= frames <= 192000, 'INPUT_PCM_GEOMETRY')
            pcm = stream.readframes(frames)
        need(len(pcm) == frames * 2, 'INPUT_PCM_BYTES')
        values = np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768.0
        need(bool(np.isfinite(values).all()), 'INPUT_NONFINITE')
        features = extractor(values, sampling_rate=16000, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**features).logits
            need(logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[2] == 392 and
                 1 <= logits.shape[1] <= 2000 and bool(torch.isfinite(logits).all()), 'OUTPUT_SHAPE_OR_NONFINITE')
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        return ctc_tokens(ids, vocab, skip), frames
    return predict, {'model_id': MODEL, 'revision': REVISION, 'files': manifest,
                     'blank_id': 0, 'skipped_special_ids': sorted(skip), 'weights_only': True,
                     'trust_remote_code': False, 'phonemizer_used': False}


def execute(original_zip, rendered_zip, output, *, backend=None):
    cells = source_cells(original_zip, rendered_zip)
    simulated = backend is not None
    if not simulated:
        execution_guard(loads(regular(Path(os.environ['GITHUB_EVENT_PATH']), 8 * 1024 * 1024)), os.environ)
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_OUTPUT')
    output.mkdir(mode=0o700)
    base = {'schema': POLICY, 'model_id': MODEL, 'model_revision': REVISION,
            'model_weights_sha256': WEIGHTS_SHA, 'test_backend': simulated, 'call_budget': 8,
            'code_sha256': sha(regular(Path(__file__), 128 * 1024)),
            'source_helper_git_blob': SOURCE_HELPER_BLOB,
            'github_sha': os.environ.get('GITHUB_SHA'), 'run_id': os.environ.get('GITHUB_RUN_ID'),
            'retry_count': 0, 'new_tts_calls': 0, 'new_asr_calls': 0, 'new_optimizer_steps': 0,
            'machine_observation_only': True, **FLAGS}
    once(output / 'started.json', base)
    attempts, rows, failure = 0, [], None
    model_root = None
    try:
        if backend is None:
            model_root = Path(tempfile.mkdtemp(prefix='english-xlsr-assets-'))
            backend, lock = load_runtime(model_root)
            once(output / 'model-lock.json', lock)
        for index, (meta, wav) in enumerate(cells):
            need(attempts < 8, 'EIGHT_FORWARD_BUDGET')
            folder = output / ('observation-%02d' % index)
            folder.mkdir(mode=0o700)
            audio = folder / 'audio.wav'
            with audio.open('xb') as stream:
                stream.write(wav)
            row = {**meta, **FLAGS, 'test_backend': simulated}
            once(folder / 'started.json', {**row, 'attempt': 1})
            attempts += 1
            try:
                tokens, frames = backend(audio)
                need(type(frames) is int and frames == meta['frames'], 'OUTPUT_FRAME_BINDING')
                need(type(tokens) is list and 1 <= len(tokens) <= 300 and
                     all(type(t) is str and 0 < len(t) <= 40 and not any(ord(c) < 32 for c in t)
                         for t in tokens), 'OUTPUT_TOKEN_SHAPE')
                row.update(status='COMPLETE', raw_ipa_tokens=tokens, decoder_completed=True)
                once(folder / 'completed.json', row)
            except BaseException as exc:
                row.update(status='FAILED_NO_RETRY', error_type=type(exc).__name__)
                once(folder / 'failed.json', row)
                if not isinstance(exc, Exception):
                    raise
            finally:
                rows.append(row)
                audio.unlink()
    except BaseException as exc:
        failure = type(exc).__name__
        if not isinstance(exc, Exception):
            raise
    finally:
        completed = sum(r.get('status') == 'COMPLETE' for r in rows)
        status = 'COMPLETE' if completed == 8 and failure is None else 'INCOMPLETE_NO_RETRY'
        if simulated and status == 'COMPLETE':
            status = 'SIMULATED_COMPLETE'
        result = {**base, 'status': status, 'attempts': attempts, 'completed': completed,
                  'rows': rows, 'error_type': failure, 'observer_agreement_is_not_gold': True}
        once(output / 'summary.json', result)
        if model_root is not None and model_root.exists():
            shutil.rmtree(model_root)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--original-zip', type=Path)
    cli.add_argument('--rendered-zip', type=Path)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args(argv)
    check_helper()
    if args.execute:
        need(all(v is not None for v in (args.original_zip, args.rendered_zip, args.output)), 'EXPLICIT_INPUTS')
        result = execute(args.original_zip, args.rendered_zip, args.output)
    elif args.original_zip or args.rendered_zip:
        need(args.original_zip is not None and args.rendered_zip is not None and args.output is None,
             'READBACK_ARGUMENTS')
        result = {'status': 'EIGHT_SOURCES_VERIFIED_NO_MODEL', 'source_count':
                  len(source_cells(args.original_zip, args.rendered_zip)), 'model_calls': 0, **FLAGS}
    else:
        need(args.output is None, 'SOURCE_ONLY_ARGUMENTS')
        result = {'status': 'SOURCE_CONTRACT_ONLY', 'max_new_phoneme_forwards': 8, 'model_calls': 0, **FLAGS}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get('status') == 'INCOMPLETE_NO_RETRY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
