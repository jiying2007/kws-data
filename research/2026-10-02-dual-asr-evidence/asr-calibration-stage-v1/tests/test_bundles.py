"""Fixture-only retained-bundle checks; never load weights, audio, or runtimes.

Every fabricated execution has a fixture-prefixed run ID, fake model/PCM roots,
loader, binder, and infer function. Only immutable plan/decoder metadata is read
from the project. Successful validation does not authenticate actual inference.
Run with: python3 -B -I -S tests/test_bundles.py
"""
import builtins
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL_IMPORT = builtins.__import__
FORBIDDEN_IMPORTS = {'torch', 'numpy', 'transformers', 'qwen_asr', 'funasr',
                     'yaml', 'kaldi_native_fbank', 'sentencepiece'}


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN_IMPORTS:
        raise AssertionError('Real model/runtime import forbidden in bundle fixtures')
    return ORIGINAL_IMPORT(name, *args, **kwargs)


with patch('builtins.__import__', guarded_import):
    from asr_stage.adapters import QWEN, SENSE
    from asr_stage.architecture import canonical_sha
    from asr_stage.bundles import validate_primary_bundle
    from asr_stage.decoding import qwen_transcript, sense_transcript
    from asr_stage.execution import ClipDeadline, execute_primary, json_bytes
    from asr_stage.results import output_envelope, scientific_manifest
    from asr_stage.assets import MODELS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch('builtins.__import__', guarded_import)
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='fixture-asr-bundles-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.decoder = (ROOT / 'pcm/decoder-inputs.json').read_bytes()
        self.inputs = {r['opaque_id']: r for r in json.loads(self.decoder)['clips']}
        self.counter = 0

    def fake_evidence(self, model_id, bound, decoded=None, ids=None):
        if model_id == QWEN:
            decoded = decoded if decoded is not None else 'language Chinese<asr_text>fixture-' + bound.opaque_id
            ids = [11, 151645] if ids is None else ids
            row, metadata = qwen_transcript(decoded, ids)
            evidence = {'token_ids': ids,
                'decoded_with_special_tokens': decoded + '<|im_end|>',
                'decoded_skip_special_tokens': decoded, 'metadata': metadata,
                'input_binding': bound.descriptor, 'prompt_context': '',
                'force_language': None, 'transcript_repair_applied': False,
                'processor_tensor_evidence': {
                    'input_ids': {'shape': [1, 3], 'dtype': 'torch.int64', 'sha256': 'a' * 64},
                    'input_features': {'shape': [1, 128, 20], 'dtype': 'torch.float32', 'sha256': 'b' * 64}},
                'processor_input_layout': {
                    'input_ids': {'original_stride': [3, 1], 'contiguous_copy': False},
                    'input_features': {'original_stride': [2560, 20, 1], 'contiguous_copy': False}}}
        else:
            decoded = decoded if decoded is not None else '<|zh|><|NEUTRAL|><|Speech|><|woitn|>fixture-' + bound.opaque_id
            ids = [1, 2, 3, 4, 11] if ids is None else ids
            row, metadata = sense_transcript(decoded)
            evidence = {'raw_result': ([{'key': bound.opaque_id, 'text': decoded}], {'fixture_only': True}),
                'token_evidence': {'token_ids': ids, 'decoded': decoded,
                    'token_scope': 'Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits'},
                'metadata': metadata, 'input_binding': bound.descriptor,
                'use_itn': False, 'rich_postprocess_applied': False,
                'completeness_scope': 'Returned full-input CTC inference; not acoustic transcript completeness'}
        return row, evidence

    def make_bundle(self, model_id=QWEN, *, fail_at=None, timeout=False,
                    load_fail=False, input_fail_at=None, decoded=None, ids=None):
        self.counter += 1
        name = 'qwen06' if model_id == QWEN else 'sensevoice'
        plan = json.loads((ROOT / 'execution-plans-v2' / (name + '.plan.json')).read_bytes())
        plan['model_root'] = '/fixture-model-never-opened'
        plan['pcm_root'] = '/fixture-pcm-never-opened'
        plan['clip_seconds'] = 1
        plan['contract']['run_id'] = 'fixture-bundle-' + name + '-' + str(self.counter)
        plan['contract_sha256'] = canonical_sha(plan['contract'])
        frozen_sha = sha(json_bytes(plan))  # Frozen before fake execution, never from retained outputs.
        path = self.root / ('fixture-bundle-' + str(self.counter))
        seen = []

        def loader(*args):
            self.assertEqual(args[0], '/fixture-model-never-opened')
            if load_fail:
                raise RuntimeError('fixture-only loader failure')
            return types.SimpleNamespace(receipt={'fixture_only': True})

        def binder(root, expectation):
            self.assertEqual(root, '/fixture-pcm-never-opened')
            self.assertEqual(expectation.basename, expectation.opaque_id + '.wav')
            if expectation.opaque_id == input_fail_at:
                raise RuntimeError('fixture-only input failure')
            item = self.inputs[expectation.opaque_id]
            return types.SimpleNamespace(opaque_id=expectation.opaque_id,
                descriptor=copy.deepcopy(item['descriptor']), descriptor_sha256=item['binding_sha256'])

        def infer(runtime, bound, decoder_sha):
            self.assertEqual(decoder_sha, sha(self.decoder))
            seen.append(bound.opaque_id)
            if bound.opaque_id == fail_at:
                raise (ClipDeadline if timeout else RuntimeError)('fixture-only decode failure')
            return self.fake_evidence(model_id, bound, decoded, ids)

        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute_primary(plan, path, self.decoder, loader=loader,
                binder=binder, infer=infer, check_environment=False)
        return types.SimpleNamespace(path=path, plan=plan, sha=frozen_sha, seen=seen, summary=summary)

    def validate(self, bundle):
        return validate_primary_bundle(bundle.path, bundle.sha)

    def read(self, bundle, name):
        return json.loads((bundle.path / name).read_bytes())

    def write(self, bundle, name, value):
        raw = json_bytes(value)
        (bundle.path / name).write_bytes(raw)
        return sha(raw)

    def rehash_chain(self, bundle, *, oid='clip-0001', sidecar=None, receipt=None,
                     outcomes=None):
        """Attacker-controlled local chain; the independent plan SHA stays frozen."""
        receipt = self.read(bundle, oid + '.receipt.json') if receipt is None else receipt
        if sidecar is not None:
            receipt['decoder_evidence_sha256'] = self.write(bundle, oid + '.decoder.json', sidecar)
        receipt_sha = self.write(bundle, oid + '.receipt.json', receipt)
        outcomes = self.read(bundle, 'outcomes.json') if outcomes is None else outcomes
        for outcome in outcomes:
            if outcome['opaque_id'] == oid:
                outcome['execution_receipt_sha256'] = receipt_sha
        self.write(bundle, 'outcomes.json', outcomes)

    def expect_bad(self, bundle, pattern=None):
        if pattern:
            with self.assertRaisesRegex((ValueError, RuntimeError, OSError), pattern):
                self.validate(bundle)
        else:
            with self.assertRaises((ValueError, RuntimeError, OSError)):
                self.validate(bundle)

    def snapshot(self, bundle):
        return {p.name: p.read_bytes() for p in bundle.path.iterdir() if p.is_file()}

    def restore(self, bundle, snapshot):
        for path in bundle.path.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        for name, raw in snapshot.items():
            (bundle.path / name).write_bytes(raw)

    def test_both_models_return_exact_retained_bytes_and_scoped_receipt(self):
        for model_id in (QWEN, SENSE):
            with self.subTest(model=model_id):
                bundle = self.make_bundle(model_id)
                outcomes, receipts, verification = self.validate(bundle)
                self.assertEqual(bundle.seen, list(self.inputs))
                self.assertEqual(outcomes, self.read(bundle, 'outcomes.json'))
                self.assertEqual(verification['counts'], {'success': 16, 'error': 0, 'timeout': 0, 'not_run': 0})
                self.assertEqual(verification['transcript_extractions_verified'], 16)
                self.assertEqual(verification['plan_sha256'], bundle.sha)
                self.assertEqual(verification['execution_contract_sha256'], bundle.plan['contract_sha256'])
                self.assertEqual(verification['model']['run_id'], bundle.plan['contract']['run_id'])
                self.assertTrue(verification['model']['run_id'].startswith('fixture-'))
                self.assertEqual(len(receipts), 16)
                self.assertEqual(len(verification['files']), 36)
                for row in outcomes:
                    raw = (bundle.path / (row['opaque_id'] + '.receipt.json')).read_bytes()
                    self.assertEqual(receipts[row['execution_receipt_sha256']], raw)
                    self.assertEqual(sha(raw), row['execution_receipt_sha256'])
                for name, details in verification['files'].items():
                    raw = (bundle.path / name).read_bytes()
                    self.assertEqual(details, {'sha256': sha(raw), 'bytes': len(raw)})
                self.assertIn('Not cryptographic proof that inference occurred', verification['scope'])
                self.assertIn('independent tokenizer re-decoding', verification['scope'])

    def test_verified_receipt_bytes_bridge_to_both_declared_envelopes(self):
        bundles = [self.make_bundle(model_id) for model_id in (QWEN, SENSE)]
        human = json.loads((ROOT / 'human-labels.json').read_bytes())
        models = [{'model_id': bundle.plan['model_id'],
            'revision': MODELS[bundle.plan['model_id']][0],
            'run_id': bundle.plan['contract']['run_id']} for bundle in bundles]
        manifest = scientific_manifest(human, models, 'fixture-bundle-probe')
        manifest_raw = json_bytes(manifest)
        for slot, bundle in enumerate(bundles):
            with self.subTest(model=bundle.plan['model_id']):
                outcomes, receipts, verification = self.validate(bundle)
                envelope, bindings = output_envelope(outcomes, human, manifest,
                    sha(manifest_raw), 'a' * 64, slot, manifest_bytes=manifest_raw,
                    execution_receipts=receipts, decoder_manifest_bytes=self.decoder,
                    decoder_manifest_sha256=verification['decoder_manifest_sha256'])
                self.assertEqual(envelope['model'], models[slot])
                self.assertEqual(len(envelope['records']), 16)
                self.assertEqual(len(bindings['records']), 16)
                self.assertEqual([r['raw_text'] for r in envelope['records']],
                                 [r['raw_text'] for r in outcomes])
                self.assertEqual([r['execution_receipt_sha256'] for r in bindings['records']],
                                 [r['execution_receipt_sha256'] for r in verification['records']])

    def test_fixture_pass_does_not_authenticate_tokens_or_live_model_bodies(self):
        bundle = self.make_bundle(SENSE)
        sidecar = self.read(bundle, 'clip-0001.decoder.json')
        sidecar['evidence']['token_evidence']['token_ids'] = [999]
        self.rehash_chain(bundle, sidecar=sidecar)
        self.validate(bundle)  # In range, but not re-decoded with a real tokenizer.
        self.assertFalse(Path(bundle.plan['model_root']).exists())
        self.assertFalse(Path(bundle.plan['pcm_root']).exists())

    def test_independent_plan_sha_is_required(self):
        bundle = self.make_bundle()
        for invalid in (None, '', 'A' * 64, '0' * 63, 1):
            with self.subTest(sha=invalid):
                with self.assertRaises(ValueError):
                    validate_primary_bundle(bundle.path, invalid)
        with self.assertRaisesRegex(ValueError, 'independently frozen'):
            validate_primary_bundle(bundle.path, '0' * 64)

    def test_exact_plan_bytes_reject_whitespace_and_rehashed_plan_mutation(self):
        bundle = self.make_bundle()
        (bundle.path / 'plan.json').write_bytes(json_bytes(bundle.plan) + b'\n')
        self.expect_bad(bundle, 'Plan bytes differ')
        plan = copy.deepcopy(bundle.plan)
        plan['contract']['run_id'] = 'fixture-attacker-run'
        plan['contract_sha256'] = canonical_sha(plan['contract'])
        self.write(bundle, 'plan.json', plan)
        self.write(bundle, 'contract.json', plan['contract'])
        self.expect_bad(bundle, 'Plan bytes differ')

    def test_exact_contract_and_decoder_bytes_reject_harmless_whitespace(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        for name in ('contract.json', 'decoder-inputs.json'):
            with self.subTest(file=name):
                (bundle.path / name).write_bytes(original[name] + b'\n')
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_trusted_but_malformed_plan_declarations_fail_closed(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        mutations = (
            ('asset hash', lambda p: p['asset_lock']['files'][0].update(bytes=1)),
            ('source hash', lambda p: p['source_lock'][0].update(sha256='0' * 64)),
            ('contract hash', lambda p: p['contract'].update(run_id='fixture-changed')),
            ('batch bool', lambda p: p['contract'].update(batch_size=True)),
            ('model mismatch', lambda p: p['contract'].update(model_id=SENSE)),
            ('manifest mismatch', lambda p: p['contract'].update(decoder_manifest_sha256='0' * 64)),
            ('schema mismatch', lambda p: p['contract'].update(expected_schema_sha256='0' * 64)),
            ('deadline bool', lambda p: p.update(clip_seconds=True)),
            ('unknown field', lambda p: p.update(unexpected='fixture')),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                plan = copy.deepcopy(bundle.plan)
                mutate(plan)
                if label not in ('asset hash', 'source hash', 'contract hash'):
                    plan['contract_sha256'] = canonical_sha(plan['contract'])
                malformed_frozen_sha = self.write(bundle, 'plan.json', plan)
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_primary_bundle(bundle.path, malformed_frozen_sha)
                self.restore(bundle, original)

    def test_receipt_and_decoder_edits_are_rejected_without_rehashing(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        for name in ('clip-0001.receipt.json', 'clip-0001.decoder.json'):
            with self.subTest(file=name):
                (bundle.path / name).write_bytes(original[name] + b'\n')
                self.expect_bad(bundle, 'bytes changed')
                self.restore(bundle, original)

    def test_receipt_swaps_fail_even_if_outcome_hash_is_updated(self):
        bundle = self.make_bundle()
        replacement = self.read(bundle, 'clip-0002.receipt.json')
        self.rehash_chain(bundle, receipt=replacement)
        self.expect_bad(bundle, 'Receipt opaque_id')

    def test_decoder_swaps_fail_even_after_rehashing_the_chain(self):
        for model_id in (QWEN, SENSE):
            with self.subTest(model=model_id):
                bundle = self.make_bundle(model_id)
                replacement = self.read(bundle, 'clip-0002.decoder.json')
                self.rehash_chain(bundle, sidecar=replacement)
                self.expect_bad(bundle, 'Decoder sidecar opaque_id')

    def test_cross_model_and_cross_run_sidecar_swaps_fail_when_rehashed(self):
        original = self.make_bundle(QWEN)
        for source in (self.make_bundle(SENSE), self.make_bundle(QWEN)):
            with self.subTest(source=source.plan['contract']['run_id']):
                self.rehash_chain(original, sidecar=self.read(source, 'clip-0001.decoder.json'))
                self.expect_bad(original, 'Decoder sidecar model')

    def test_rehashed_receipt_association_and_execution_flags_are_checked(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        mutations = {
            'model_id': lambda r: r['model'].update(model_id=SENSE),
            'revision': lambda r: r['model'].update(revision='0' * 40),
            'run': lambda r: r['model'].update(run_id='fixture-other-run'),
            'contract': lambda r: r.update(execution_contract_sha256='0' * 64),
            'input': lambda r: r.update(input_binding_sha256='0' * 64),
            'manifest': lambda r: r.update(decoder_manifest_sha256='0' * 64),
            'wav': lambda r: r.update(wav_sha256='0' * 64),
            'attempt false': lambda r: r.update(attempt_started=False),
            'attempt integer': lambda r: r.update(attempt_started=1),
            'forward false': lambda r: r.update(model_forward_completed=False),
            'forward integer': lambda r: r.update(model_forward_completed=1),
            'extra field': lambda r: r.update(extra='fixture'),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label):
                receipt = self.read(bundle, 'clip-0001.receipt.json')
                mutate(receipt)
                self.rehash_chain(bundle, receipt=receipt)
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_rehashed_sidecar_association_duration_and_shape_are_checked(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        mutations = (
            ('run', lambda s: s['model'].update(run_id='fixture-other')),
            ('input', lambda s: s.update(input_binding_sha256='0' * 64)),
            ('wav', lambda s: s.update(wav_sha256='0' * 64)),
            ('outcome', lambda s: s['outcome'].update(raw_text='invented')),
            ('negative seconds', lambda s: s.update(wall_seconds=-1)),
            ('boolean seconds', lambda s: s.update(wall_seconds=True)),
            ('string seconds', lambda s: s.update(wall_seconds='0')),
            ('extra field', lambda s: s.update(extra='fixture')),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                sidecar = self.read(bundle, 'clip-0001.decoder.json')
                mutate(sidecar)
                self.rehash_chain(bundle, sidecar=sidecar)
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_outcome_order_missing_extra_and_duplicate_membership_fail(self):
        bundle = self.make_bundle()
        original = self.read(bundle, 'outcomes.json')
        for label, rows in (
                ('reordered', [original[1], original[0]] + original[2:]),
                ('missing', original[:-1]), ('extra', original + [original[0]]),
                ('duplicate', [original[0]] + original[:-1])):
            with self.subTest(case=label):
                self.write(bundle, 'outcomes.json', rows)
                self.expect_bad(bundle)

    def test_outcome_shape_states_and_flags_are_strict(self):
        bundle = self.make_bundle()
        original = self.read(bundle, 'outcomes.json')
        mutations = (
            {'status': 'failed'}, {'completeness': 'partial'}, {'quality_flags': ['fake']},
            {'quality_flags': ['ambiguous', 'ambiguous']}, {'quality_flags': [True]},
            {'quality_flags': 'ambiguous'}, {'raw_text': None}, {'raw_text': 'x' * 1025},
            {'wav_sha256': '0' * 64}, {'execution_receipt_sha256': None}, {'extra': 'fixture'},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                outcomes = copy.deepcopy(original)
                outcomes[0].update(mutation)
                self.write(bundle, 'outcomes.json', outcomes)
                self.expect_bad(bundle)

    def test_reextraction_rejects_coherently_rehashed_invented_text_flags_completeness(self):
        for model_id in (QWEN, SENSE):
            bundle = self.make_bundle(model_id)
            original = self.snapshot(bundle)
            for field, value in (('raw_text', 'invented fixture text'),
                                 ('quality_flags', ['ambiguous']), ('completeness', 'incomplete')):
                with self.subTest(model=model_id, field=field):
                    sidecar = self.read(bundle, 'clip-0001.decoder.json')
                    receipt = self.read(bundle, 'clip-0001.receipt.json')
                    outcomes = self.read(bundle, 'outcomes.json')
                    sidecar['outcome'][field] = value
                    receipt[field] = value
                    outcomes[0][field] = value
                    self.rehash_chain(bundle, sidecar=sidecar, receipt=receipt, outcomes=outcomes)
                    self.expect_bad(bundle, 'Re-extracted transcript outcome')
                    self.restore(bundle, original)

    def test_qwen_reextraction_and_evidence_declarations_reject_rehashed_edits(self):
        bundle = self.make_bundle(QWEN)
        original = self.snapshot(bundle)
        mutations = (
            ('raw extract', lambda e: e.update(decoded_skip_special_tokens='language Chinese<asr_text>edited')),
            ('token termination', lambda e: e.update(token_ids=[11, 12])),
            ('bad tokens', lambda e: e.update(token_ids=[True])),
            ('empty tokens', lambda e: e.update(token_ids=[])),
            ('token bound', lambda e: e.update(token_ids=[11] * 257)),
            ('metadata', lambda e: e['metadata'].update(eos_terminated=1)),
            ('binding', lambda e: e['input_binding']['wav'].update(sha256='0' * 64)),
            ('context', lambda e: e.update(prompt_context='fixture hotword')),
            ('language', lambda e: e.update(force_language='Chinese')),
            ('repair', lambda e: e.update(transcript_repair_applied=True)),
            ('repair integer', lambda e: e.update(transcript_repair_applied=0)),
            ('full decoded', lambda e: e.update(decoded_with_special_tokens=None)),
            ('no features', lambda e: e.update(processor_tensor_evidence={})),
            ('layout mismatch', lambda e: e['processor_input_layout'].pop('input_ids')),
            ('shape bool', lambda e: e['processor_tensor_evidence']['input_ids'].update(shape=[True, 3])),
            ('shape negative', lambda e: e['processor_tensor_evidence']['input_ids'].update(shape=[1, -3])),
            ('dtype', lambda e: e['processor_tensor_evidence']['input_ids'].update(dtype='torch.float16')),
            ('tensor sha', lambda e: e['processor_tensor_evidence']['input_ids'].update(sha256='x' * 64)),
            ('stride', lambda e: e['processor_input_layout']['input_ids'].update(original_stride=[3])),
            ('copy boolean', lambda e: e['processor_input_layout']['input_ids'].update(contiguous_copy=0)),
            ('extra', lambda e: e.update(extra='fixture')),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                sidecar = self.read(bundle, 'clip-0001.decoder.json')
                mutate(sidecar['evidence'])
                self.rehash_chain(bundle, sidecar=sidecar)
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_sense_reextraction_and_evidence_declarations_reject_rehashed_edits(self):
        bundle = self.make_bundle(SENSE)
        original = self.snapshot(bundle)
        def edit_both_raw_views(evidence):
            evidence['raw_result'][0][0]['text'] += ' edited'
            evidence['token_evidence']['decoded'] += ' edited'

        mutations = (
            ('both raw views', edit_both_raw_views),
            ('raw result text', lambda e: e['raw_result'][0][0].update(text='edited')),
            ('raw result key', lambda e: e['raw_result'][0][0].update(key='clip-0002')),
            ('raw result shape', lambda e: e.update(raw_result=[])),
            ('raw result metadata', lambda e: e['raw_result'].__setitem__(1, [])),
            ('decoded', lambda e: e['token_evidence'].update(decoded='edited')),
            ('bad tokens', lambda e: e['token_evidence'].update(token_ids=[True])),
            ('negative tokens', lambda e: e['token_evidence'].update(token_ids=[-1])),
            ('token bound', lambda e: e['token_evidence'].update(token_ids=[1] * 4097)),
            ('token scope', lambda e: e['token_evidence'].update(token_scope='framewise logits')),
            ('metadata', lambda e: e.update(metadata=['zh'])),
            ('binding', lambda e: e['input_binding']['wav'].update(sha256='0' * 64)),
            ('itn', lambda e: e.update(use_itn=True)),
            ('itn integer', lambda e: e.update(use_itn=0)),
            ('postprocess', lambda e: e.update(rich_postprocess_applied=True)),
            ('scope', lambda e: e.update(completeness_scope='Acoustically complete')),
            ('extra', lambda e: e.update(extra='fixture')),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                sidecar = self.read(bundle, 'clip-0001.decoder.json')
                mutate(sidecar['evidence'])
                self.rehash_chain(bundle, sidecar=sidecar)
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_extraction_preserves_warnings_and_non_speech_without_repair(self):
        cases = (
            (QWEN, 'language None<asr_text>', [151645], 'complete', ['non_speech']),
            (QWEN, 'language Chinese<asr_text>fixture', [11], 'incomplete', ['incomplete']),
            (QWEN, 'unrecognized fixture prefix', [151643], 'complete', ['decoding_warning']),
            (SENSE, '', [], 'unknown', ['decoding_warning']),
            (SENSE, '<|nospeech|><|EMO_UNKNOWN|><|Event_UNK|><|woitn|>', [], 'complete', ['ambiguous', 'non_speech']),
            (SENSE, '<|zh|><|NEUTRAL|><|Speech|><|withitn|>一', [1], 'complete', ['decoding_warning']),
        )
        for model_id, decoded, ids, completeness, flags in cases:
            with self.subTest(model=model_id, text=decoded):
                bundle = self.make_bundle(model_id, decoded=decoded, ids=ids)
                outcomes, _, _ = self.validate(bundle)
                self.assertEqual(outcomes[0]['completeness'], completeness)
                self.assertEqual(outcomes[0]['quality_flags'], flags)

    def test_error_and_timeout_preserve_unattempted_tail(self):
        for model_id in (QWEN, SENSE):
            for timeout in (False, True):
                with self.subTest(model=model_id, timeout=timeout):
                    bundle = self.make_bundle(model_id, fail_at='clip-0003', timeout=timeout)
                    outcomes, receipts, verification = self.validate(bundle)
                    status = 'timeout' if timeout else 'error'
                    self.assertEqual(bundle.seen, ['clip-0001', 'clip-0002', 'clip-0003'])
                    self.assertEqual(verification['counts'], {'success': 2, 'error': int(not timeout),
                        'timeout': int(timeout), 'not_run': 13})
                    self.assertEqual(len(receipts), 3)
                    self.assertEqual(verification['transcript_extractions_verified'], 2)
                    self.assertEqual(outcomes[2]['status'], status)
                    for row in outcomes[2:]:
                        self.assertIsNone(row['raw_text'])
                        self.assertEqual(row['completeness'], 'unknown')
                        self.assertEqual(row['quality_flags'], [])
                    for row in outcomes[3:]:
                        self.assertIsNone(row['execution_receipt_sha256'])

    def test_all_not_run_is_preserved_but_cannot_be_declared_execution(self):
        for model_id in (QWEN, SENSE):
            with self.subTest(model=model_id):
                bundle = self.make_bundle(model_id, load_fail=True)
                outcomes, receipts, verification = self.validate(bundle)
                self.assertEqual(bundle.seen, [])
                self.assertEqual(receipts, {})
                self.assertEqual(verification['counts']['not_run'], 16)
                self.assertEqual(verification['records'], [])
                self.assertEqual(len(verification['files']), 4)
                human = json.loads((ROOT / 'human-labels.json').read_bytes())
                models = [{'model_id': mid, 'revision': MODELS[mid][0],
                    'run_id': bundle.plan['contract']['run_id'] if mid == model_id else 'fixture-other'}
                    for mid in (QWEN, SENSE)]
                manifest = scientific_manifest(human, models, 'fixture-bundle-probe')
                manifest_raw = json_bytes(manifest)
                with self.assertRaisesRegex(ValueError, 'All-not-run cannot claim declared execution'):
                    output_envelope(outcomes, human, manifest, sha(manifest_raw), 'a' * 64,
                        0 if model_id == QWEN else 1, manifest_bytes=manifest_raw,
                        execution_receipts=receipts, decoder_manifest_bytes=self.decoder,
                        decoder_manifest_sha256=sha(self.decoder))

    def test_input_failure_is_not_invented_as_decode_attempt(self):
        bundle = self.make_bundle(SENSE, input_fail_at='clip-0002')
        outcomes, receipts, verification = self.validate(bundle)
        self.assertEqual(bundle.seen, ['clip-0001'])
        self.assertEqual(len(receipts), 1)
        self.assertEqual(verification['counts']['not_run'], 15)
        self.assertEqual(outcomes[1]['status'], 'not_run')

    def test_continuing_after_error_or_not_run_is_rejected(self):
        for status in ('error', 'not_run'):
            with self.subTest(status=status):
                bundle = self.make_bundle()
                if status == 'error':
                    failed = self.make_bundle(fail_at='clip-0001')
                    # Same frozen run identity; keep later success rows to test stopping semantics.
                    receipt = self.read(failed, 'clip-0001.receipt.json')
                    sidecar = self.read(failed, 'clip-0001.decoder.json')
                    receipt['model']['run_id'] = bundle.plan['contract']['run_id']
                    receipt['execution_contract_sha256'] = bundle.plan['contract_sha256']
                    sidecar['model']['run_id'] = bundle.plan['contract']['run_id']
                    outcomes = self.read(bundle, 'outcomes.json')
                    outcomes[0] = self.read(failed, 'outcomes.json')[0]
                    self.rehash_chain(bundle, sidecar=sidecar, receipt=receipt, outcomes=outcomes)
                else:
                    outcomes = self.read(bundle, 'outcomes.json')
                    outcomes[0].update(status='not_run', raw_text=None, completeness='unknown',
                                       quality_flags=[], execution_receipt_sha256=None)
                    self.write(bundle, 'outcomes.json', outcomes)
                self.expect_bad(bundle, 'continued after stopping')

    def test_failure_evidence_status_and_forward_flags_reject_rehashed_edits(self):
        bundle = self.make_bundle(fail_at='clip-0001')
        original = self.snapshot(bundle)
        mutations = (
            ('wrong exception status', lambda e: e.update(exception_type='ClipDeadline')),
            ('forward true', lambda e: e.update(forward_completion_confirmed=True)),
            ('forward integer', lambda e: e.update(forward_completion_confirmed=0)),
            ('empty exception', lambda e: e.update(exception_type='')),
            ('bad detail', lambda e: e.update(detail=None)),
            ('large detail', lambda e: e.update(detail='x' * 4097)),
            ('extra', lambda e: e.update(extra='fixture')),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                sidecar = self.read(bundle, 'clip-0001.decoder.json')
                mutate(sidecar['evidence'])
                self.rehash_chain(bundle, sidecar=sidecar)
                self.expect_bad(bundle)
                self.restore(bundle, original)
        receipt = self.read(bundle, 'clip-0001.receipt.json')
        receipt['model_forward_completed'] = True
        self.rehash_chain(bundle, receipt=receipt)
        self.expect_bad(bundle, 'forward/attempt')

    def test_failed_or_not_run_outcomes_cannot_invent_transcript_evidence(self):
        for status in ('error', 'not_run'):
            bundle = self.make_bundle(fail_at='clip-0001') if status == 'error' else self.make_bundle(load_fail=True)
            original = self.read(bundle, 'outcomes.json')
            for change in ({'raw_text': ''}, {'completeness': 'complete'},
                           {'quality_flags': ['ambiguous']}):
                with self.subTest(status=status, change=change):
                    rows = copy.deepcopy(original)
                    rows[0].update(change)
                    self.write(bundle, 'outcomes.json', rows)
                    self.expect_bad(bundle, 'invented transcript')
            if status == 'not_run':
                rows = copy.deepcopy(original)
                rows[0]['execution_receipt_sha256'] = '0' * 64
                self.write(bundle, 'outcomes.json', rows)
                self.expect_bad(bundle, 'invented execution')

    def test_required_files_and_attempted_sidecars_must_exist(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        for name in ('plan.json', 'contract.json', 'decoder-inputs.json', 'outcomes.json',
                     'clip-0001.receipt.json', 'clip-0001.decoder.json'):
            with self.subTest(file=name):
                (bundle.path / name).unlink()
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_extra_and_not_run_receipt_decoder_sidecars_are_rejected(self):
        bundle = self.make_bundle(fail_at='clip-0001')
        for name in ('extra.receipt.json', 'extra.decoder.json',
                     'clip-0002.receipt.json', 'clip-0002.decoder.json'):
            with self.subTest(file=name):
                self.write(bundle, name, {'fixture_only': True})
                self.expect_bad(bundle, 'unused or not-run')
                (bundle.path / name).unlink()

    def test_auxiliary_files_are_explicitly_outside_validation_scope(self):
        bundle = self.make_bundle()
        for name in ('model-load.json', 'environment.json', 'summary.json',
                     'outcomes.initial.json', 'clip-0001.started.json', 'extra.txt'):
            (bundle.path / name).write_bytes(b'not JSON; fixture outside declared scope')
        _, _, verification = self.validate(bundle)
        self.assertNotIn('summary.json', verification['files'])
        self.assertIn('outside this validation scope', verification['scope'])

    def test_duplicate_json_keys_and_nonfinite_numbers_fail_before_use(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        for name in ('plan.json', 'contract.json', 'decoder-inputs.json', 'outcomes.json',
                     'clip-0001.receipt.json', 'clip-0001.decoder.json'):
            for malformed in (b'{"duplicate":1,"duplicate":2}', b'{"bad":NaN}',
                              b'{"bad":Infinity}', b'{"bad":1e999}'):
                with self.subTest(file=name, malformed=malformed):
                    (bundle.path / name).write_bytes(malformed)
                    self.expect_bad(bundle, 'Duplicate JSON|Nonfinite JSON')
                    self.restore(bundle, original)

    def test_symlink_directory_and_file_are_rejected(self):
        bundle = self.make_bundle()
        linked_directory = self.root / 'fixture-linked-bundle'
        linked_directory.symlink_to(bundle.path, target_is_directory=True)
        with self.assertRaises(OSError):
            validate_primary_bundle(linked_directory, bundle.sha)
        original = self.snapshot(bundle)
        target = self.root / 'fixture-external-json'
        for name in ('plan.json', 'clip-0001.receipt.json', 'clip-0001.decoder.json'):
            with self.subTest(file=name):
                target.write_bytes(original[name])
                (bundle.path / name).unlink()
                (bundle.path / name).symlink_to(target)
                self.expect_bad(bundle)
                self.restore(bundle, original)

    def test_nonregular_and_oversized_files_are_rejected_without_blocking(self):
        bundle = self.make_bundle()
        original = self.snapshot(bundle)
        name = 'clip-0001.decoder.json'
        (bundle.path / name).unlink()
        os.mkfifo(bundle.path / name)
        self.expect_bad(bundle, 'Nonregular or oversized')
        self.restore(bundle, original)
        (bundle.path / name).unlink()
        (bundle.path / name).mkdir()
        self.expect_bad(bundle)
        self.restore(bundle, original)
        for name, limit in (('clip-0001.decoder.json', 8 * 1024 * 1024),
                            ('outcomes.json', 1024 * 1024)):
            with self.subTest(file=name):
                with (bundle.path / name).open('wb') as stream:
                    stream.truncate(limit + 1)
                self.expect_bad(bundle, 'Nonregular or oversized')
                self.restore(bundle, original)

    def test_detects_metadata_change_during_file_read(self):
        bundle = self.make_bundle()
        real_fstat = os.fstat
        calls = []

        def changed_fstat(fd):
            result = real_fstat(fd)
            calls.append(fd)
            if len(calls) == 2:
                return types.SimpleNamespace(st_size=result.st_size,
                    st_mtime_ns=result.st_mtime_ns + 1, st_ctime_ns=result.st_ctime_ns)
            return result

        with patch('asr_stage.bundles.os.fstat', changed_fstat):
            self.expect_bad(bundle, 'changed while reading')


if __name__ == '__main__':
    unittest.main(verbosity=2)
