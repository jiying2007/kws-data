"""Offline research inference for one manifest-bound English embedding head.

The default command only describes the source contract. --verify-only uses the
standard library. --execute explicitly runs the bundled fixed ONNX frontend and
saved head on a user-supplied 16 kHz mono PCM16 WAV, without EOF padding.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import wave

SCHEMA = 'english-final-embedding-offline-bundle-v1'
ASSETS = {
    'melspectrogram.onnx': (1087958, 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f'),
    'embedding_model.onnx': (1326578, '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f'),
}
UPSTREAM_COMMIT = '368c03716d1e92591906a84949bc477f3a834455'
UPSTREAM_LICENSE = (11357, 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4')
POLICY = {'sample_rate': 16000, 'embedding_hop_samples': 1280, 'first_embedding_sample': 16000,
          'history_embeddings': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
          'first_head_sample': 35200, 'threshold': 0.8, 'confirm_embeddings': 3,
          'cooldown_embeddings': 13, 'eof_padding_samples': 0, 'dynamic_threshold': False}
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
RUNTIME_FILES = ('weak_embedding_contract.py', 'weak_audio_cases.py', 'causal_embedding.py',
                 'causal_head.py', 'portable_head.py', 'english_causal_head.c',
                 'english_causal_head.h', 'head_replay.c')
THRESHOLD_F32 = struct.unpack('<f', struct.pack('<f', 0.8))[0]
LABELS = {0: 'background', 1: 'Hey Vosen', 2: 'Hey Mero'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_json(path):
    need(file_identity(path)['bytes'] <= 32 * 1024**2, 'JSON_SIZE_LIMIT')
    raw = Path(path).read_bytes()
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def file_identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE_REQUIRED:' + str(path))
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def write_json(path, value):
    with Path(path).open('xb') as output:
        output.write(canonical(value) + b'\n')


def verify_bundle(directory):
    """Verify every delivered byte without importing NumPy, ONNX, or Torch."""
    directory = Path(directory).absolute()
    need(directory.is_dir() and not directory.is_symlink(), 'REGULAR_BUNDLE_DIRECTORY')
    manifest_path = directory / 'manifest.json'
    file_identity(manifest_path)
    manifest = read_json(manifest_path)
    need(manifest.get('schema') == SCHEMA and manifest.get('status') == 'COMPLETE_RESEARCH_BUNDLE'
         and manifest.get('policy') == POLICY and manifest.get('test_backend') is False
         and all(manifest.get(key) is False for key in FALSE_FLAGS), 'FIXED_RESEARCH_BUNDLE_CONTRACT')
    files = manifest.get('files')
    need(type(files) is dict and 15 <= len(files) <= 100, 'MANIFEST_FILE_ROSTER')
    required = {'english_final_embedding_infer.py', 'requirements.txt', 'MODEL_CARD.md',
                'models/causal-head.npz', 'models/causal-head.bin',
                'licenses/openWakeWord-LICENSE', 'licenses/kws-pipeline-LICENSE',
                'evidence/RESULT.json', 'evidence/TRAIN_RESULT.json', 'evidence/PARITY.json'}
    required.update('models/' + name for name in ASSETS)
    required.update('runtime/' + name for name in RUNTIME_FILES)
    need(required <= set(files) and 'manifest.json' not in files, 'REQUIRED_BUNDLE_FILES')
    for relative, identity in files.items():
        need(type(relative) is str and '\\' not in relative and
             not PurePosixPath(relative).is_absolute() and
             all(part not in ('', '.', '..') for part in relative.split('/')), 'MANIFEST_PATH')
        path = directory.joinpath(*relative.split('/'))
        need(not any(p.is_symlink() for p in (path, *path.parents)), 'BUNDLE_SYMLINK')
        need(type(identity) is dict and set(identity) == {'bytes', 'sha256'} and
             type(identity['bytes']) is int and 0 <= identity['bytes'] <= 256 * 1024**2 and
             re.fullmatch('[0-9a-f]{64}', str(identity['sha256'])) and
             file_identity(path) == identity, 'BUNDLE_FILE_IDENTITY:' + relative)
    actual = set()
    for path in directory.rglob('*'):
        need(not path.is_symlink(), 'BUNDLE_SYMLINK')
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    need(actual == set(files) | {'manifest.json'}, 'UNLISTED_BUNDLE_FILE')
    for name, (size, digest) in ASSETS.items():
        need(files['models/' + name] == {'bytes': size, 'sha256': digest}, 'FIXED_FEATURE_ASSET')
    need(files['licenses/openWakeWord-LICENSE'] ==
         {'bytes': UPSTREAM_LICENSE[0], 'sha256': UPSTREAM_LICENSE[1]}, 'ORIGINAL_UPSTREAM_LICENSE')
    provenance = manifest.get('provenance', {})
    need(provenance.get('upstream_commit') == UPSTREAM_COMMIT and
         provenance.get('weights_sha256') == files['models/causal-head.npz']['sha256'] and
         provenance.get('head_binary_sha256') == files['models/causal-head.bin']['sha256'],
         'MODEL_PROVENANCE_BINDING')
    return manifest


class FixedEvents:
    def __init__(self):
        self.candidate = self.confirmations = self.blocked_until = 0

    def step(self, probabilities, sample):
        need(len(probabilities) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities),
             'THREE_FINITE_PROBABILITIES')
        selected = max(range(3), key=lambda index: probabilities[index])
        if selected == 0 or probabilities[selected] < THRESHOLD_F32 or sample < self.blocked_until:
            self.candidate = self.confirmations = 0
            return 0
        self.confirmations = self.confirmations + 1 if selected == self.candidate else 1
        self.candidate = selected
        if self.confirmations < 3:
            return 0
        self.blocked_until = sample + 13 * 1280
        self.candidate = self.confirmations = 0
        return selected


class HistoryHead:
    """Bounded real history; the score callback is the unchanged saved head math."""
    def __init__(self, score):
        self.score, self.history, self.events = score, deque(maxlen=16), FixedEvents()
        self.last_sample = 0

    def step(self, embedding, sample):
        need(type(sample) is int and sample == (self.last_sample + 1280 if self.last_sample else 16000),
             'CONTIGUOUS_OBSERVED_SAMPLE_CLOCK')
        need(len(embedding) == 96 and all(math.isfinite(x) for x in embedding), 'FINITE_EMBEDDING_96')
        self.history.append(tuple(embedding))
        self.last_sample = sample
        result = {'sample': sample, 'ready': len(self.history) == 16, 'keyword': 0,
                  'logits': [0.0] * 3, 'probabilities': [0.0] * 3}
        if result['ready']:
            logits, probabilities = self.score(self.history)
            need(len(logits) == 3 and all(math.isfinite(x) for x in logits), 'FINITE_HEAD_LOGITS')
            result.update(logits=logits, probabilities=probabilities,
                          keyword=self.events.step(probabilities, sample))
        return result


def numpy_scorer(weights):
    import numpy as np
    def score(history):
        normalized = ((np.asarray(history, dtype=np.float32) - weights['mean96']) /
                      weights['scale96']).astype(np.float32)
        need(bool(np.isfinite(normalized).all()), 'FINITE_NORMALIZED_EMBEDDING')
        flattened = normalized.reshape(1536)
        hidden = np.maximum(weights['w1'] @ flattened + weights['b1'], 0)
        logits = weights['w2'] @ hidden + weights['b2']
        exp = np.exp((logits - logits.max()).astype(np.float64))
        probabilities = (exp / exp.sum()).astype(np.float32)
        return logits.tolist(), probabilities.tolist()
    return score


def consume_wav(source, features, head, trace):
    """The injectable callbacks support source fixtures without any real model."""
    need((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) ==
         (1, 2, 16000, 'NONE'), 'WAV_MUST_BE_MONO_PCM16_16000')
    declared, observed, embeddings, outputs, events = source.getnframes(), 0, 0, 0, []
    pcm_hash = hashlib.sha256()
    while True:
        raw = source.readframes(1280)
        if not raw:
            break
        need(len(raw) % 2 == 0, 'TRUNCATED_PCM16')
        pcm_hash.update(raw)
        observed += len(raw) // 2
        for record in features.accept(struct.unpack('<%dh' % (len(raw) // 2), raw)):
            row = head.step(record['embedding'], record['available_sample'])
            trace(row)
            embeddings += 1
            outputs += int(row['ready'])
            if row['keyword']:
                events.append({'sample': row['sample'], 'audio_seconds': row['sample'] / 16000,
                               'keyword': row['keyword'], 'label': LABELS[row['keyword']]})
    need(observed == declared and features.observed == observed and not features.finish(),
         'OBSERVED_WAV_COUNT_AND_NO_EOF_PADDING')
    last = head.last_sample or None
    return {'observed_samples': observed, 'pcm_sha256': pcm_hash.hexdigest(),
            'embedding_outputs': embeddings, 'head_outputs': outputs, 'events': events,
            'first_embedding_sample': 16000 if embeddings else None,
            'first_head_sample': 35200 if outputs else None, 'last_embedding_sample': last,
            'eof': {'padding_samples': 0, 'observed_tail_after_last_embedding': observed - (last or 0),
                    'future_samples_unobserved': True, 'nontrigger_status': 'CENSORED_NOT_FRR',
                    'complete_head_history_observed': outputs > 0}}


def create_output_directory(bundle, requested):
    requested = Path(requested).absolute()
    need(not any(path.is_symlink() for path in (requested, *requested.parents)), 'OUTPUT_SYMLINK')
    output, bundle = requested.resolve(), Path(bundle).resolve()
    need(output.parent.is_dir() and not output.exists() and output != bundle and bundle not in output.parents,
         'NEW_OUTPUT_DIRECTORY_OUTSIDE_BUNDLE')
    output.mkdir(mode=0o700)
    return output


def execute(bundle, wav_path, output):
    manifest = verify_bundle(bundle)
    bundle, wav_path = Path(bundle).resolve(), Path(wav_path)
    wav_identity = file_identity(wav_path)
    with wave.open(str(wav_path), 'rb') as source:
        need((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) ==
             (1, 2, 16000, 'NONE'), 'WAV_MUST_BE_MONO_PCM16_16000')
    output = create_output_directory(bundle, output)
    started = {'schema': 'english-final-embedding-offline-inference-v1', 'test_backend': False,
               'bundle_manifest_sha256': file_identity(bundle / 'manifest.json')['sha256'],
               'weights_sha256': manifest['provenance']['weights_sha256'],
               'wav': wav_identity, 'policy': POLICY, **dict.fromkeys(FALSE_FLAGS, False)}
    write_json(output / 'STARTED.json', started)
    try:
        sys.path.insert(0, str(bundle / 'runtime'))
        causal, portable = importlib.import_module('causal_embedding'), importlib.import_module('portable_head')
        need(all(Path(sys.modules[name].__file__).resolve() == bundle / 'runtime' / (name + '.py')
                 for name in ('causal_embedding', 'portable_head', 'weak_embedding_contract')),
             'BUNDLED_RUNTIME_MODULE_IDENTITIES')
        weights = portable.load_weights(bundle / 'models/causal-head.npz')
        need(sha(portable.export_blob(weights)) == manifest['provenance']['head_binary_sha256'],
             'SAVED_NPZ_AND_C_BINARY_IDENTITY')
        embedding = causal.FrozenOnnxEmbedding(bundle / 'models')
        features, head = causal.CausalFeatureStream(embedding), HistoryHead(numpy_scorer(weights))
        with (output / 'frames.jsonl').open('xb') as trace, wave.open(str(wav_path), 'rb') as source:
            result = consume_wav(source, features, head, lambda row: trace.write(canonical(row) + b'\n'))
        need(file_identity(wav_path) == wav_identity, 'WAV_CHANGED_DURING_INFERENCE')
        result.update(started, status='EXECUTED_OFFLINE_RESEARCH_ONLY', geometry=embedding.geometry,
                      independent_frr='NOT_MEASURED', independent_far='NOT_MEASURED',
                      word_end_latency='NOT_MEASURED', ssc305='NOT_QUALIFIED',
                      trace=file_identity(output / 'frames.jsonl'))
        write_json(output / 'RESULT.json', result)
        return result
    except Exception as error:
        write_json(output / 'FAILURE.json', {**started, 'status': 'FAILED',
                   'error_type': type(error).__name__, 'reason': str(error)[:500]})
        raise


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    for name in ('bundle', 'wav', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if args.verify_only:
        need(args.bundle is not None, 'EXPLICIT_BUNDLE_REQUIRED')
        manifest = verify_bundle(args.bundle)
        print(json.dumps({'status': 'VERIFIED_FILE_IDENTITIES_ONLY', 'files': len(manifest['files']),
                          'manifest_sha256': file_identity(args.bundle / 'manifest.json')['sha256'], 'model_calls': 0}))
    elif args.execute:
        need(all(x is not None for x in (args.bundle, args.wav, args.output)), 'EXPLICIT_INFERENCE_INPUTS')
        print(json.dumps(execute(args.bundle, args.wav, args.output), ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps({'schema': SCHEMA, 'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0,
                          'policy': POLICY, 'execution_enabled': False}))
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
