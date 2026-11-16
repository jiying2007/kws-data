"""Prepared 144-cell research recipe; no model/network/execution implementation.

The four-cell pilot must first be read back and its exact useful evidence frozen.
Missing pins are deliberately fatal to the proposed execution guard.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

POLICY = 'english-final-ipa-rendering-144-research-v1'
PILOT_RUN = 43809825038
MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
REVISION = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
KEYWORD_INDICES = {'Hey Vosen': 1, 'Hey Mero': 2}
VOICES = (('aiden', 'train'), ('dylan', 'train'), ('eric', 'train'),
          ('serena', 'train'), ('ryan', 'development'), ('vivian', 'development'))
WAKE_NAMES = (
    ('Hey Mero', 'Meero', '/heɪ ˈmiːroʊ/',
     'The name is MEE-roh, /miːroʊ/, with the long ee vowel of meet.'),
    ('Hey Vosen', 'Vosen', '/heɪ ˈvoʊzən/',
     'The name is VOH-zuhn, /voʊzən/, with voiced z as in zoo.'),
)
VARIANTS = (
    ('neutral', 'Hey {name}.', 'Use neutral everyday conversational speech.'),
    ('gentle', 'Hey, {name}.', 'Use a gentle, relaxed tone.'),
    ('calling', 'Hey {name}!', 'Call naturally to someone nearby, without shouting.'),
    ('questioning', 'Hey {name}?', 'Use a natural mild questioning intonation.'),
    ('slightly_slow', 'Hey {name}.', 'Speak slightly slowly but keep the words connected.'),
    ('slightly_brisk', 'Hey {name}.', 'Speak slightly briskly and naturally, without rushing.'),
    ('soft_audible', 'Hey, {name}.', 'Speak softly but clearly with a voiced, audible tone; do not whisper.'),
    ('warm', 'Hey {name}!', 'Use a warm, cheerful everyday tone.'),
)
ORDINARY_NEGATIVES = (
    'What time is it?',
    'Please turn on the kitchen light.',
    'The package is by the front door.',
    'Could you open the window?',
    'I will be back this afternoon.',
    "Let's have dinner at seven.",
    'Where did I put my keys?',
    'Thank you for your help.',
)
FLAGS = {'human_gold': False, 'training_admitted': False,
         'formal_training_admitted': False, 'qualification_allowed': False,
         'model_promotion_allowed': False, 'trusted_negative': False,
         'pronunciation_verified': False}
PIN_FILE = Path(__file__).with_name('final144_execution_pins.json')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def recipe():
    cells = []
    for voice, role in VOICES:
        voice_fields = {'voice_id': voice, 'speaker': voice, 'role': role,
                        'source_family_id': 'qwen3-preset-' + voice,
                        'origin_kind': 'synthetic-preset', 'independent_human': False,
                        'family_exposed_in_rendering_pilot': voice in ('aiden', 'ryan')}
        for phrase, name, ipa, phonetics in WAKE_NAMES:
            for variant, pattern, style in VARIANTS:
                text = pattern.format(name=name)
                cells.append({**voice_fields, 'intent': 'wake-candidate',
                              'canonical_phrase': phrase, 'target_ipa': ipa,
                              'synthesis_text': text, 'text_requested': text,
                              'requested_keyword_index': KEYWORD_INDICES[phrase], 'variant': variant,
                              'instruction': 'Say only ' + text + ' ' + phonetics +
                              ' Keep Hey light and unstressed; stress the first syllable of the name. '
                              'Use natural American English. ' + style + ' Do not add any words.',
                              **FLAGS})
        for number, text in enumerate(ORDINARY_NEGATIVES, start=1):
            cells.append({**voice_fields, 'intent': 'ordinary-negative-candidate',
                          'canonical_phrase': None, 'target_ipa': None,
                          'synthesis_text': text, 'text_requested': text,
                          'requested_keyword_index': None, 'variant': 'ordinary-%02d' % number,
                          'instruction': 'Say only the supplied sentence in natural American English. '
                          'Use neutral conversational speech and do not add any words.', **FLAGS})
    for cell in cells:
        cell['cell_id'] = 'enfinal144-' + sha(canonical({'policy': POLICY, 'model_revision': REVISION,
                                                       'cell': cell}))[:24]
    need(len(cells) == len({c['cell_id'] for c in cells}) == 144, 'EXACT_144_UNIQUE_CELLS')
    return {'schema': POLICY, 'pilot_run_id': PILOT_RUN, 'model_id': MODEL,
            'model_revision': REVISION, 'keyword_indices': KEYWORD_INDICES, 'voice_sources': [
                {'id': v, 'source_family_id': 'qwen3-preset-' + v, 'role': r,
                 'origin_kind': 'synthetic-preset', 'independent_human': False} for v, r in VOICES],
            'cells': cells, 'counts': {'cells': 144, 'train': 96, 'development': 48,
                                     'wake_candidates': 96, 'ordinary_negative_candidates': 48,
                                     'independent_humans': 0, 'independent_test_cells': 0},
            'max_tts_calls': 144, 'max_calls_per_observer': 144, 'no_retry': True,
            'capture_stage_budgets': [16] * 9,
            'observer_names': ['qwen', 'whisper', 'parakeet', 'phones'],
            'nearphone_negative_cells': 0, 'source_labels_are_intents_only': True,
            'execution_state': 'PREPARED_NOT_EXECUTABLE_WITHOUT_PILOT_PINS', **FLAGS}


def validate_execution_pins(pins):
    """Shape/intent validation only; the eventual runner must read the pinned evidence."""
    need(pins.get('schema') == 'english-final144-execution-pins-v1' and
         pins.get('pilot_run_id') == PILOT_RUN, 'PILOT_IDENTITY')
    for field in ('pilot_head_sha', 'next_main_before'):
        value = pins.get(field)
        need(type(value) is str and len(value) == 40 and
             all(c in '0123456789abcdef' for c in value), 'MISSING_OR_INVALID_PIN:' + field)
    for field in ('pilot_source_artifact_id', 'pilot_review_artifact_id'):
        need(type(pins.get(field)) is int and pins[field] > 0, 'MISSING_OR_INVALID_PIN:' + field)
    for field in ('pilot_source_zip_sha256', 'pilot_review_zip_sha256',
                  'pilot_review_json_sha256', 'pilot_source_plan_sha256',
                  'pilot_model_lock_sha256', 'pilot_code_sha256', 'recipe_sha256'):
        value = pins.get(field)
        need(type(value) is str and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), 'MISSING_OR_INVALID_PIN:' + field)
    need(pins.get('recipe_sha256') == sha(canonical(recipe())), 'RECIPE_PIN_CHANGED')
    need(pins.get('pilot_review_status') == 'USEFUL_FOR_WEAK_RESEARCH' and
         pins.get('pilot_machine_review_useful') is True, 'PILOT_USEFUL_EVIDENCE_REQUIRED')
    need(all(pins.get(k) is False for k in FLAGS) and pins.get('no_retry') is True and
         pins.get('max_tts_calls') == pins.get('max_calls_per_observer') == 144,
         'BOUNDED_UNADMITTED_RESEARCH_ONLY')
    return pins


def execution_guard(pins, event, env):
    validate_execution_pins(pins)
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and event.get('before') == pins['next_main_before'] and
         event.get('after') == env.get('GITHUB_SHA') and
         type(event.get('after')) is str and len(event['after']) == 40 and
         all(c in '0123456789abcdef' for c in event['after']) and
         event['after'] != pins['next_main_before'] and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'FIRST_PUBLIC_MAIN_SUCCESSOR_ONLY')


class RenderingCaptureAdapter:
    """Adapt the existing capture(cell, speaker) call to the pinned TTS Backend."""
    def __init__(self, backend):
        self.backend = backend

    def synthesize(self, cell, speaker):
        need(cell['voice_id'] in {v for v, _ in VOICES} and
             speaker == cell['voice_id'].title(), 'CAPTURE_PRESET_BINDING')
        return self.backend.synthesize({'cell_id': cell['cell_id'],
                    'phrase': cell['synthesis_text'], 'speaker': cell['voice_id'],
                    'instruction': cell['instruction']})


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan', action='store_true')
    cli.add_argument('--check-execution-pins', action='store_true')
    cli.add_argument('--pins', type=Path, default=PIN_FILE)
    args = cli.parse_args(argv)
    if args.check_execution_pins:
        validate_execution_pins(json.loads(args.pins.read_text(encoding='utf-8')))
        print('PINS_SHAPE_CHECKED_NO_EVIDENCE_CONSUMED_OR_MODEL_EXECUTED')
    else:
        plan = recipe()
        print(json.dumps(plan if args.plan else {'schema': POLICY, 'counts': plan['counts'],
                         'recipe_sha256': sha(canonical(plan)), 'model_calls': 0,
                         'execution_enabled': False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
