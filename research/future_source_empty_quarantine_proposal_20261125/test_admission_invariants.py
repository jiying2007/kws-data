"""Invented proposal examples only; no real evidence, people, runtime or approval.

These local stdlib fixtures illustrate decision boundaries for human review.
They are not a production observer, original-evidence verifier or source gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unicodedata
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROPOSAL = json.loads((HERE / 'PROPOSAL.json').read_bytes())
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60')
TEXT_PROVIDERS = frozenset(PROVIDERS[:3])
WHITE_SPACE = frozenset(int(item[2:], 16)
    for item in PROPOSAL['native_empty_boundary']['white_space_codepoints'])
ROWS = tuple(f'INVENTED-{i:03}' for i in range(120))
VALID, EMPTY, FATAL = 'VALID_NONEMPTY', 'INVALID_EMPTY_PROVEN', 'FATAL'


def invented_text_outcome(provider, raw, **changes):
    """Symbolic evidence flags, never an authenticator of actual native data."""
    proof = dict(attempt=1, input_bound=True, native_return_observed=True,
        raw_complete=True, native_shape_valid=True, timed_out=False,
        exception_type=None, native_exit_code=0, untransformed=True)
    proof.update(changes)
    if (provider not in TEXT_PROVIDERS or type(raw) is not bytes
            or type(proof['attempt']) is not int or proof['attempt'] != 1
            or any(proof[key] is not True for key in
                ('input_bound', 'native_return_observed', 'raw_complete', 'native_shape_valid', 'untransformed'))
            or proof['timed_out'] is not False or proof['exception_type'] is not None
            or (provider == 'parakeet' and (type(proof['native_exit_code']) is not int
                or proof['native_exit_code'] != 0))):
        return FATAL
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return FATAL
    if len(text) > 4096 or any(ord(c) not in WHITE_SPACE
            and unicodedata.category(c) in ('Cc', 'Cf', 'Cs') for c in text):
        return FATAL
    return EMPTY if all(ord(c) in WHITE_SPACE for c in text) else VALID


def invented_panel_eligibility(records, observer_rows=ROWS):
    """Only the full original IDs eligible to be given to an unchanged gate."""
    if (len(set(observer_rows)) != len(observer_rows) or not 96 <= len(observer_rows) <= 120
            or not set(ROWS[:96]) <= set(observer_rows) <= set(ROWS)):
        raise ValueError('SOURCE_STOP: changed pre-observer eligibility')
    expected = {(row, provider) for row in observer_rows for provider in PROVIDERS}
    keys = [(row, provider) for row, provider, _ in records]
    if len(keys) != 5 * len(observer_rows) or len(set(keys)) != len(expected) or set(keys) != expected:
        raise ValueError('SOURCE_STOP: incomplete, duplicated or unplanned slot')
    for _row, provider, outcome in records:
        if outcome not in (VALID, EMPTY) or (outcome == EMPTY and provider not in TEXT_PROVIDERS):
            raise ValueError('SOURCE_STOP: fatal return; quarantine cannot conceal it')
    quarantined = {row for row, _provider, outcome in records if outcome == EMPTY}
    return dict(original_denominator=120, observer_denominator=len(observer_rows),
        slot_denominator=5 * len(observer_rows),
        pre_observer_rejected=[row for row in ROWS if row not in observer_rows],
        eligible=[row for row in observer_rows if row not in quarantined],
        quarantined=[row for row in ROWS if row in quarantined])


def invented_records(changes=None, observer_rows=ROWS):
    changes = {} if changes is None else changes
    return [(row, provider, changes.get((row, provider), VALID))
            for row in observer_rows for provider in PROVIDERS]


class ProposalInvariants(unittest.TestCase):
    def test_no_authority_and_original_rules_are_still_exact(self):
        self.assertEqual(PROPOSAL['status'], 'PROPOSAL_ONLY_NOT_APPROVED_NOT_REGISTERED_NOT_ACTIVATED')
        self.assertFalse(PROPOSAL['approval']['approved'])
        self.assertTrue(all(value is False for value in PROPOSAL['execution_authority'].values()))
        for name, pin in PROPOSAL['evidence_scope']['immutable_reference_files'].items():
            raw = (ROOT / name).read_bytes()
            self.assertEqual({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, pin, name)
        old = json.loads((ROOT / 'automation/english_v6_b24r1_evaluation_design.json').read_bytes())
        for key in ('source_gate', 'gains', 'frozen_model_pt_sha256',
                    'quality_assessment_sha256', 'quality_function_sha256'):
            self.assertEqual(PROPOSAL['unchanged_science'][key], old[key])
        protected = PROPOSAL['protected_existing_state']
        self.assertEqual(protected['B26_opening_unchanged'],
            json.loads((ROOT / 'automation/english_v6_b26_opening.json').read_bytes()))
        self.assertFalse(protected['future_other_panel_result_satisfies_B26_opening'])
        self.assertFalse(protected['automatic_R2_allowed'])

    def test_genuine_zero_byte_and_explicit_whitespace_returns_are_invalid(self):
        for provider in TEXT_PROVIDERS:
            for raw in (b'', b'\n', b' \t\r\n', '\u00a0'.encode(), '\u3000'.encode()):
                self.assertEqual(invented_text_outcome(provider, raw), EMPTY)
        for codepoint in WHITE_SPACE:
            self.assertEqual(invented_text_outcome('parakeet', chr(codepoint).encode()), EMPTY)

    def test_nonempty_text_is_not_turned_into_an_empty_or_a_label(self):
        for raw in (b'Hey Mero.', b'What time is it?', b'...', b'[no speech]', b'\nlog line\n'):
            self.assertEqual(invented_text_outcome('parakeet', raw), VALID)
        # VALID here only means native nonempty; this fixture assigns no class.
        self.assertFalse(PROPOSAL['single_policy_change']['empty_return_has_positive_or_negative_vote'])

    def test_corrupt_oversize_or_non_whitespace_control_is_fatal(self):
        for raw in (b'\xff', b'\x00', b' ' * 4097, '\u200b'.encode(), '\ufeff'.encode()):
            self.assertEqual(invented_text_outcome('parakeet', raw), FATAL)
        self.assertEqual(invented_text_outcome('parakeet', b' ' * 4096), EMPTY)

    def test_an_exception_name_or_empty_looking_value_is_not_proof(self):
        for changes in ({'attempt': 2}, {'attempt': True}, {'input_bound': False},
                {'native_return_observed': False}, {'raw_complete': False},
                {'native_shape_valid': False}, {'timed_out': True}, {'untransformed': False},
                {'exception_type': 'EMPTY_NATIVE_TRANSCRIPT'}, {'native_exit_code': 1},
                {'native_exit_code': False}):
            self.assertEqual(invented_text_outcome('parakeet', b'\n', **changes), FATAL)
        for provider in ('qwen', 'whisper'):
            self.assertEqual(invented_text_outcome(provider, b'', native_shape_valid=False), FATAL)
        for raw in (None, [], {}, '', 0):
            self.assertEqual(invented_text_outcome('qwen', raw), FATAL)

    def test_empty_native_phone_array_cannot_take_the_text_exception(self):
        self.assertEqual(invented_text_outcome('xlsr', b''), FATAL)
        self.assertEqual(invented_text_outcome('lv60', b''), FATAL)
        with self.assertRaises(ValueError):
            invented_panel_eligibility(invented_records({(ROWS[0], 'xlsr'): EMPTY}))

    def test_one_empty_removes_whole_original_even_with_other_four_valid(self):
        for provider in TEXT_PROVIDERS:
            result = invented_panel_eligibility(invented_records({(ROWS[7], provider): EMPTY}))
            self.assertEqual(result['quarantined'], [ROWS[7]])
            self.assertEqual(len(result['eligible']), 119)
            self.assertNotIn(ROWS[7], result['eligible'])
            self.assertEqual((result['original_denominator'], result['slot_denominator']), (120, 600))
        self.assertEqual(PROPOSAL['quarantine_rule']['applies_to_both_requested_classes'], [0, 2])
        self.assertEqual(PROPOSAL['quarantine_rule']['kws_forwards_allowed'], 0)

    def test_fatal_on_an_already_quarantined_original_still_stops_panel(self):
        changes = {(ROWS[7], 'parakeet'): EMPTY, (ROWS[7], 'qwen'): FATAL}
        with self.assertRaises(ValueError):
            invented_panel_eligibility(invented_records(changes))

    def test_missing_duplicate_unknown_and_plan_changes_keep_panel_closed(self):
        records = invented_records({(ROWS[7], 'parakeet'): EMPTY})
        variants = (records[:-1], records + [records[0]], [records[0]] + records[:-1],
            records[:-1] + [('UNPLANNED', 'lv60', VALID)],
            invented_records({(ROWS[0], 'whisper'): 'UNKNOWN'}))
        for changed in variants:
            with self.assertRaises(ValueError):
                invented_panel_eligibility(changed)

    def test_all_valid_only_means_eligible_not_source_or_quality_pass(self):
        result = invented_panel_eligibility(invented_records())
        self.assertEqual(result['eligible'], list(ROWS))
        self.assertEqual(result['quarantined'], [])
        self.assertNotIn('source_gate_pass', result)
        self.assertNotIn('observed_class', result)
        self.assertNotIn('promotion_allowed', result)

    def test_existing_pre_observer_rejects_are_retained_without_extra_calls(self):
        for count in (96, 108, 118, 120):
            observer_rows = ROWS[:count]
            records = invented_records(observer_rows=observer_rows)
            result = invented_panel_eligibility(records, observer_rows)
            self.assertEqual(result['original_denominator'], 120)
            self.assertEqual(result['observer_denominator'], count)
            self.assertEqual(result['slot_denominator'], 5 * count)
            self.assertEqual(result['pre_observer_rejected'], list(ROWS[count:]))
            self.assertEqual(result['eligible'], list(observer_rows))
            self.assertEqual(result['quarantined'], [])
            with self.assertRaises(ValueError):
                invented_panel_eligibility(records[:-1], observer_rows)
            if count < 120:
                with self.assertRaises(ValueError):
                    invented_panel_eligibility(records + [(ROWS[count], 'parakeet', VALID)], observer_rows)

    def test_minimum_counts_can_coexist_with_72_quarantined_originals(self):
        # Entirely invented cells/identities; no real roster or data selection.
        positive = {2 * i for i in range(8)} | {2 * i + 1 for i in range(4)}
        ordinary = set(range(48, 72))
        natural = {96 + 2 * i for i in range(8)} | {97 + 2 * i for i in range(4)}
        retained = positive | ordinary | natural
        changes = {(ROWS[i], 'parakeet'): EMPTY for i in range(120) if i not in retained}
        result = invented_panel_eligibility(invented_records(changes))
        self.assertEqual((len(result['eligible']), len(result['quarantined'])), (48, 72))
        floors = PROPOSAL['unchanged_science']['source_gate']
        self.assertEqual((len(positive), len(ordinary), len(natural)),
            (floors['synthetic_positive'], floors['synthetic_ordinary'], floors['natural_originals']))
        paired = {i // 2 for i in positive} & {(i - 48) // 2 for i in ordinary}
        readers = {(i - 96) // 2 for i in natural}
        self.assertEqual(len(paired), floors['synthetic_paired'])
        self.assertEqual(len(readers), floors['natural_speakers'])


if __name__ == '__main__':
    unittest.main()
