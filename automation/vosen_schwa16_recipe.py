"""Sixteen new frozen-rhyme requests; no model execution or label admission."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = 'english-vosen-schwa16-frozen-rhyme-v1'
MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
REVISION = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
OLD_RECIPE_CODE_SHA256 = '519d02747e8a8519f7b1f6086c2b92a8a562254477d3d01e4c7164967e7d59ac'
OLD_RECIPE_SHA256 = 'd1fe7973324bf35f9fcd1c20472786e121c18bb3e4bdbe8967953a0f88243b83'
PHONETIC_INSTRUCTION = (
    'The name Vosen rhymes with the American English word "frozen". '
    'Replace only the initial /fr/ of "frozen" with a voiced /v/, preserving the /oʊzən/ ending. '
    'Use two syllables, /ˈvoʊ.zən/. In the second, unstressed syllable, make the short relaxed '
    'schwa /ə/ audible between /z/ and the final /n/.')
DECODING = {'language': 'English', 'do_sample': False, 'device_map': 'cpu',
            'dtype': 'bfloat16', 'attn_implementation': 'sdpa',
            'seed_rule': 'int(sha256(cell_id.encode())[:12],16)',
            'other_generation_settings': 'unchanged-pinned-qwen-tts-0.1.1-and-model-generation-config'}
FLAGS = dict.fromkeys(('human_gold', 'training_admitted', 'formal_training_admitted',
    'qualification_allowed', 'model_promotion_allowed', 'trusted_negative', 'pronunciation_verified'), False)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def legacy_recipe():
    path = HERE / 'final144_recipe.py'
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
         sha(path.read_bytes()) == OLD_RECIPE_CODE_SHA256, 'FROZEN_144_RECIPE_SOURCE')
    spec = importlib.util.spec_from_file_location('_schwa16_original_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.recipe()
    need(sha(canonical(value)) == OLD_RECIPE_SHA256 and len(value['cells']) == 144, 'FROZEN_144_RECIPE')
    return value


def request_fingerprint(cell):
    """Do not let a new cell id or seed disguise an otherwise repeated request."""
    return sha(canonical({'model_id': MODEL, 'model_revision': REVISION,
        'voice': cell['voice_id'], 'text': cell['synthesis_text'],
        'instruction': cell['instruction'], 'decoding': DECODING}))


def old_request_fingerprints():
    return [request_fingerprint(cell) for cell in legacy_recipe()['cells']]


def assert_new_requests(cells):
    fingerprints = [request_fingerprint(cell) for cell in cells]
    need(len(fingerprints) == 16 and len(set(fingerprints)) == 16, 'EXACT_SIXTEEN_UNIQUE_REQUESTS')
    need(not set(fingerprints).intersection(old_request_fingerprints()), 'REPEATED_OLD_144_REQUEST')
    return fingerprints


def recipe():
    old = legacy_recipe()
    selected = [cell for cell in old['cells'] if cell['voice_id'] in ('aiden', 'eric') and
                cell['canonical_phrase'] == 'Hey Vosen']
    original_phrase = 'The name is VOH-zuhn, /voʊzən/, with voiced z as in zoo.'
    cells = []
    for previous in selected:
        cell = dict(previous)
        del cell['cell_id']
        need(cell['instruction'].count(original_phrase) == 1, 'ORIGINAL_PHONETIC_INSTRUCTION')
        cell['instruction'] = cell['instruction'].replace(original_phrase, PHONETIC_INSTRUCTION)
        fingerprint = request_fingerprint(cell)
        cell.update(historical_comparator_cell_id=previous['cell_id'],
                    request_fingerprint_sha256=fingerprint,
                    cell_id='envosenschwa16-' + sha(canonical({'policy': POLICY, 'request': fingerprint}))[:24])
        cells.append(cell)
    assert_new_requests(cells)
    need([c['voice_id'] for c in cells] == ['aiden'] * 8 + ['eric'] * 8 and
         all(c['role'] == 'train' and c['source_family_id'] == 'qwen3-preset-' + c['voice_id'] and
             c['target_ipa'] == '/heɪ ˈvoʊzən/' for c in cells), 'SOURCE_ROLE_OR_TARGET_DRIFT')
    return {'schema': POLICY, 'model_id': MODEL, 'model_revision': REVISION,
        'old_recipe_sha256': OLD_RECIPE_SHA256,
        'old_request_fingerprints_sha256': sha(canonical(old_request_fingerprints())),
        'hypothesis': 'FROZEN_RHYME_UNSTRESSED_SCHWA_MACHINE_OBSERVATION_ONLY',
        'decoding': dict(DECODING), 'cells': cells,
        'voice_sources': [v for v in old['voice_sources'] if v['id'] in ('aiden', 'eric')],
        'counts': {'cells': 16, 'train': 16, 'development': 0, 'wake_candidates': 16,
                   'ordinary_negative_candidates': 0, 'independent_humans': 0},
        'max_tts_calls': 16, 'max_calls_per_observer': 16,
        'observer_names': ['qwen', 'whisper', 'parakeet', 'phones'],
        'source_labels_are_intents_only': True, 'no_retry': True, **FLAGS}


class CaptureAdapter:
    def __init__(self, backend):
        self.backend = backend

    def synthesize(self, cell, speaker):
        need(speaker.lower() == cell['voice_id'], 'CAPTURE_SPEAKER_IDENTITY')
        return self.backend.synthesize({'cell_id': cell['cell_id'], 'phrase': cell['synthesis_text'],
            'speaker': cell['voice_id'], 'instruction': cell['instruction']})
