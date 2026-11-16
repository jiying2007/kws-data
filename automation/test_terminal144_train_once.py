"""Invented source/observation/API/ZIP fixtures only; no models or Actions env."""
from __future__ import annotations
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_training_under_test', HERE / 'terminal144_train_once.py')
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)
old, observer = worker.contracts()
recipe = observer.recipe.recipe()


def digest(text):
    return worker.sha(text.encode())


def fixture():
    """All identities/observations here are explicitly invented test values."""
    rows, joined, old_rows = [], [], []
    summaries = {p: digest('invented summary ' + p) for p in worker.PROVIDERS}
    for index, cell in enumerate(recipe['cells']):
        label = cell['requested_keyword_index'] or 0
        text = {1: 'Hey Vosen', 2: 'Hey Mero'}.get(label, cell['synthesis_text'])
        tokens = {1: ['v', 'oʊ', 'z', 'ə', 'n'], 2: ['m', 'i', 'ɹ', 'oʊ']}.get(label, ['s', 'ɛ', 'n', 't'])
        row = {**cell, 'source_status': 'GENERATED_UNADMITTED', 'source_disposition': observer.VALID,
               'wav_sha256': digest('invented WAV ' + str(index)), 'pcm_sha256': digest('invented PCM ' + str(index)),
               'frames': 16000, 'source_shard': index // 16, 'source_attempted': True,
               'source_error_type': None, 'readback_status': 'GENERATED_UNADMITTED',
               'readback_error_type': None, 'readback_equals_original': True}
        rows.append(row)
        values = {p: {'observation_status': 'COMPLETE', 'observation_attempted': True,
                      'pcm_sha256': row['pcm_sha256'], 'text': text} for p in worker.PROVIDERS[:3]}
        values['phones'] = {'observation_status': 'COMPLETE', 'observation_attempted': True,
                           'pcm_sha256': row['pcm_sha256'], 'raw_ipa_tokens': tokens}
        joined.append({**copy.deepcopy(row), 'observations': values})
        old_rows.append({**copy.deepcopy(row), 'status': row['source_status'],
                         'transcripts': {p: text for p in worker.PROVIDERS[:3]}, 'raw_ipa_tokens': tokens})
    terminal = {'fixture': 'invented original terminal API'}
    ledger = {'schema': observer.POLICY + '/source-ledger', **observer.SOURCE, 'source_denominator': 144,
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'original_run_conclusion': 'failure',
        'terminal_evidence': terminal, 'terminal_evidence_canonical_sha256': worker.sha(worker.canonical(terminal)),
        'source_artifacts': [], 'test_backend': False, 'rows': rows,
        'source_labels_are_intents_only': True, **dict.fromkeys(worker.FALSE_FLAGS, False)}
    ledger.update(valid_cells=observer.cohort(ledger), valid_count=144, source_dispositions={observer.VALID: 144})
    identity = {'run_id': '99000002', 'github_sha': '2' * 40, 'run_attempt': '1', 'test_backend': False}
    combined = {'schema': observer.POLICY + '/combined', **observer.SOURCE, 'status': 'FOUR_OBSERVERS_COMPLETE',
        'test_backend': False, 'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'source_denominator': 144,
        'valid_count': 144, 'source_ledger_canonical_sha256': worker.sha(worker.canonical(ledger)),
        'observer_summaries_sha256': summaries, 'observer_identity': identity,
        'observer_model_locks_sha256': {p: digest('invented lock ' + p) for p in worker.PROVIDERS},
        'providers': {p: {'status': 'OBSERVATIONS_COMPLETE', 'attempts': 144, 'completed': 144,
                          'error_type': None, 'error_message': None} for p in worker.PROVIDERS},
        'new_tts_calls': 0, 'optimizer_steps': 0, 'model_calls_by_combiner': 0,
        'source_labels_are_intents_only': True, **dict.fromkeys(worker.FALSE_FLAGS, False), 'rows': joined}
    return ledger, combined, old_rows


def pinned(ledger, combined):
    observation_pins = observer.pins_template()
    observation_pins.update(enabled=True, next_main_before='1' * 40, code_sha256=worker.OBSERVER_SHA,
        source_artifacts=[{'shard_index': i, 'artifact_id': 7000 + i, 'zip_bytes': 20,
            'zip_sha256': digest('invented source ZIP ' + str(i)),
            'source_plan_sha256': digest('invented plan ' + str(i)),
            'source_result_sha256': digest('invented result ' + str(i)),
            'model_lock_sha256': observer.SOURCE['source_model_lock_sha256']} for i in range(9)],
        terminal_evidence_canonical_sha256=ledger['terminal_evidence_canonical_sha256'],
        source_ledger_canonical_sha256=worker.sha(worker.canonical(ledger)),
        valid_cells=ledger['valid_cells'], max_calls_per_provider=ledger['valid_count'])
    pins = worker.pins_template()
    pins.update(observer_run={'run_id': 99000002, 'github_sha': '2' * 40, 'run_attempt': 1},
        observation_pins_sha256=worker.sha(worker.canonical(observation_pins)),
        observer_artifacts={p: {'artifact_id': 8000 + i, 'zip_bytes': 20,
            'zip_sha256': digest('invented observer ZIP ' + p),
            'summary_sha256': combined['observer_summaries_sha256'][p],
            'model_lock_sha256': combined['observer_model_locks_sha256'][p]} for i, p in enumerate(worker.PROVIDERS)},
        combined_artifact={'artifact_id': 8004, 'zip_bytes': 20, 'zip_sha256': digest('invented combined ZIP'),
                          'member': 'terminal144-observations-144.json',
                          'member_sha256': worker.sha(worker.canonical(combined) + b'\n')})
    terminal = {'fixture': 'invented observer terminal API'}
    pins['observer_terminal_evidence_canonical_sha256'] = worker.sha(worker.canonical(terminal))
    binding = {'observer_identity': worker.observer_identity(pins), 'artifacts': worker.artifact_roster(pins),
        'observer_terminal_evidence': terminal,
        'observer_terminal_evidence_canonical_sha256': pins['observer_terminal_evidence_canonical_sha256'],
        'api_files_sha256': {name: digest('invented ' + name) for name in ('run.json', 'jobs.json', 'artifacts.json')}}
    return pins, observation_pins, binding


class TerminalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='terminal-training-invented-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger, self.combined, self.old_rows = fixture()
        self.pins, self.opins, self.binding = pinned(self.ledger, self.combined)

    def select(self):
        return worker.select_terminal(recipe, self.ledger, self.combined, old)

    def test_default_and_template_are_disabled_without_loading_models_or_contracts(self):
        before = dict(os.environ)
        with mock.patch.object(worker, 'contracts', side_effect=AssertionError('default must be stdlib only')):
            for argv in ([], ['--template']):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(worker.main(argv), 0)
                result = json.loads(stream.getvalue())
                self.assertIs(result.get('enabled', result.get('execution_enabled')), False)
        template = worker.pins_template()
        self.assertTrue(all(template[k] is None for k in ('observer_run', 'observer_artifacts',
            'combined_artifact', 'selected_rows_sha256', 'valid_count', 'next_main_before')))
        self.assertEqual(os.environ, before)
        self.assertFalse({'torch', 'numpy', 'onnxruntime'} & set(sys.modules))

    def test_fully_observed_predicate_matches_unchanged_legacy_for_all_144(self):
        previous = old.select_intents(recipe, self.old_rows, {'test_backend': False,
            'observer_summaries_sha256': self.combined['observer_summaries_sha256']})
        actual = self.select()
        self.assertEqual(actual['eligible_cell_ids'], previous['eligible_cell_ids'])
        self.assertEqual(actual['observations'], previous['observations'])
        for before, after in zip(previous['decisions'], actual['decisions']):
            self.assertEqual(before, {key: after[key] for key in before})
        self.assertEqual(actual['policy_sha256'], worker.SELECTION_SHA)

    def test_failed_source_is_retained_and_never_faked_as_generated(self):
        self.ledger['rows'][0].update(source_status='FAILED_NO_RETRY', source_disposition='FAILED_NO_RETRY',
                                     pcm_sha256=None, wav_sha256=None, frames=None)
        self.combined['rows'][0].update(self.ledger['rows'][0])
        self.ledger.update(valid_cells=observer.cohort(self.ledger), valid_count=143,
                           source_dispositions={observer.VALID: 143, 'FAILED_NO_RETRY': 1})
        result = self.select()
        self.assertEqual(result['source_denominator'], 144)
        self.assertEqual(len(result['decisions']), 144)
        self.assertEqual(result['decisions'][0]['source_status_unchanged'], 'FAILED_NO_RETRY')
        self.assertEqual(worker.library_source_rows(self.ledger)[0]['status'], 'FAILED_NO_RETRY')
        self.assertNotIn(self.ledger['rows'][0]['cell_id'], result['eligible_cell_ids'])

    def test_entire_duplicate_group_is_excluded_across_classes_and_roles(self):
        first = next(r for r in self.ledger['rows'] if r['role'] == 'train' and r['requested_keyword_index'] == 1)
        other = next(r for r in self.ledger['rows'] if r['role'] == 'development' and r['requested_keyword_index'] is None)
        other['pcm_sha256'] = first['pcm_sha256']
        joined = next(r for r in self.combined['rows'] if r['cell_id'] == other['cell_id'])
        joined.update(other)
        for values in joined['observations'].values():
            values['pcm_sha256'] = first['pcm_sha256']
        result = self.select()
        for cid in (first['cell_id'], other['cell_id']):
            self.assertNotIn(cid, result['eligible_cell_ids'])
            self.assertNotIn(cid, result['observations'])
            self.assertEqual(next(d for d in result['decisions'] if d['cell_id'] == cid)['quarantine_reason'],
                             'WHOLE_DUPLICATE_PCM_GROUP_QUARANTINED')

    def test_empty_phones_cannot_support_ordinary_negative(self):
        row = next(r for r in self.combined['rows'] if r['requested_keyword_index'] is None)
        row['observations']['phones']['raw_ipa_tokens'] = []
        decision = next(d for d in self.select()['decisions'] if d['cell_id'] == row['cell_id'])
        self.assertEqual(decision['evidence_status'], 'UNKNOWN')
        self.assertFalse(decision['eligible_for_requested_intent_research'])

    def test_exact_asr_and_phone_rule_not_retuned(self):
        row = next(r for r in self.combined['rows'] if r['requested_keyword_index'] == 1)
        row['observations']['phones']['raw_ipa_tokens'] = ['v', 'oʊ', 'z', 'ɪ', 'n']
        decision = next(d for d in self.select()['decisions'] if d['cell_id'] == row['cell_id'])
        self.assertEqual(decision['asr_votes'], 3)
        self.assertFalse(decision['eligible_for_requested_intent_research'])
        row['observations']['phones']['raw_ipa_tokens'] = ['v', 'oʊ', 'z', 'ə', 'n']
        for provider in worker.PROVIDERS[:2]:
            row['observations'][provider]['text'] = 'Hey Vosen please'
        decision = next(d for d in self.select()['decisions'] if d['cell_id'] == row['cell_id'])
        self.assertEqual(decision['asr_votes'], 1)
        self.assertFalse(decision['eligible_for_requested_intent_research'])

    def test_missing_provider_or_mismatched_pcm_is_rejected(self):
        row = self.combined['rows'][0]
        row['observations']['whisper']['pcm_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'SAME_PCM'):
            self.select()
        row['observations']['whisper']['pcm_sha256'] = row['pcm_sha256']
        del row['observations']['phones']
        with self.assertRaisesRegex(ValueError, 'SAME_PCM'):
            self.select()

    def test_four_complete_observers_and_frozen_selection_are_required(self):
        actual = worker.validate_source_documents(self.pins, self.opins, self.ledger, self.combined,
                                                   self.binding, old, observer)
        self.pins['selected_rows_sha256'] = worker.sha(worker.canonical(actual))
        self.combined['providers']['whisper']['completed'] = 143
        with self.assertRaisesRegex(ValueError, 'ACTUALLY_COMPLETE'):
            worker.validate_source_documents(self.pins, self.opins, self.ledger, self.combined, self.binding, old, observer)
        self.combined['providers']['whisper']['completed'] = 144
        self.pins['selected_rows_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SELECTION_CHANGED'):
            worker.validate_source_documents(self.pins, self.opins, self.ledger, self.combined, self.binding, old, observer)

    def test_minimum_counts_are_source_only_and_rechecked_after_feature_losses(self):
        rows = [{'cell_id': c['cell_id'], 'role': c['role'], 'weak_class': c['requested_keyword_index'] or 0,
                 'source_family_id': c['source_family_id']} for c in recipe['cells']]
        self.assertTrue(old.minimum_evidence(rows)['passed'])
        remaining = [r for r in rows if not (r['role'] == 'development' and r['weak_class'] == 2)]
        self.assertFalse(old.minimum_evidence(remaining)['passed'])
        remaining = [r for r in rows if r['weak_class'] != 0]
        result = old.minimum_evidence(remaining)
        self.assertFalse(result['passed'])
        self.assertEqual(result['counts']['train/0']['count'], 0)
        self.assertTrue(result['ordinary_negative_excludes_librispeech_substitution'])

    def execution_pins(self):
        selected = self.select()
        pins = copy.deepcopy(self.pins)
        pins.update(enabled=True, code_sha256=worker.sha(Path(worker.__file__).read_bytes()),
            next_main_before='2' * 40, source_ledger_canonical_sha256=worker.sha(worker.canonical(self.ledger)),
            valid_count=144, valid_cells_sha256=worker.sha(worker.canonical(self.ledger['valid_cells'])),
            selected_rows_sha256=worker.sha(worker.canonical(selected)))
        return pins

    def test_distinct_first_public_main_guard_uses_only_passed_dictionaries(self):
        before = dict(os.environ)
        pins = self.execution_pins()
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
               'GITHUB_RUN_ID': '99000003', 'GITHUB_SHA': '3' * 40}
        event = {'before': '2' * 40, 'after': '3' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        worker.execution_guard(pins, event, env)
        for key, value in (('GITHUB_RUN_ID', worker.SOURCE_IDENTITY['run_id']),
                           ('GITHUB_RUN_ID', '99000002'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_REF', 'refs/heads/fixture')):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                worker.execution_guard(pins, event, {**env, key: value})
        pins['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'DISABLED'):
            worker.execution_guard(pins, event, env)
        self.assertEqual(os.environ, before)

    def test_fixed_old_dependencies_recipe_and_no_qualification_cannot_change(self):
        for key, value in (('pipeline_commit', 'f' * 40), ('legacy_worker_sha256', 'f' * 64),
                           ('optimizer_updates', 95), ('seed', 1), ('proposal_threshold', 0.79),
                           ('selection_policy_sha256', 'f' * 64), ('formal_training_admitted', True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FIXED_TERMINAL'):
                worker.validate_pins({**self.pins, key: value})

    def test_zip_hash_and_safe_members_reject_mutation_traversal_and_symlink(self):
        path = self.root / 'invented.zip'
        with zipfile.ZipFile(path, 'x') as archive:
            archive.writestr('good.json', '{}')
        pin = {'zip_bytes': path.stat().st_size, 'zip_sha256': worker.sha(path.read_bytes())}
        with worker.checked_zip(path, pin) as archive:
            worker.safe_members(archive, {'good.json'}, {'good.json'})
        with path.open('ab') as output:
            output.write(b'invented corruption')
        with self.assertRaisesRegex(ValueError, 'ZIP_BYTES'):
            worker.checked_zip(path, pin)
        for name, symlink in (('../escape.json', False), ('link.json', True)):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                info = zipfile.ZipInfo(name)
                if symlink:
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, '{}')
            with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
                with self.assertRaisesRegex(ValueError, 'UNSAFE'):
                    worker.safe_members(archive, {name}, {name})

    def metadata(self):
        run = {'id': self.pins['observer_run']['run_id'], 'head_sha': '2' * 40, 'run_attempt': 1,
            'event': 'push', 'head_branch': 'main', 'status': 'completed', 'conclusion': 'success',
            'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        names = ['source-contract', 'source-evidence', 'combine'] + ['observe (' + p + ')' for p in worker.PROVIDERS]
        jobs = [{'id': 9000 + i, 'name': name, 'run_id': run['id'], 'run_attempt': 1, 'head_sha': run['head_sha'],
                 'status': 'completed', 'conclusion': 'success'} for i, name in enumerate(names)]
        terminal = {'run_id': run['id'], 'head_sha': run['head_sha'], 'run_attempt': 1, 'status': 'completed',
                    'conclusion': 'success', 'jobs': sorted(jobs, key=lambda j: j['name'])}
        self.pins['observer_terminal_evidence_canonical_sha256'] = worker.sha(worker.canonical(terminal))
        artifacts = [{'id': p['artifact_id'], 'name': p['name'], 'size_in_bytes': p['zip_bytes'],
            'digest': 'sha256:' + p['zip_sha256'], 'expired': False,
            'workflow_run': {'id': run['id'], 'head_sha': run['head_sha'], 'head_branch': 'main'}}
            for p in worker.artifact_roster(self.pins)]
        documents = {'run.json': run, 'jobs.json': {'total_count': len(jobs), 'jobs': jobs},
                     'artifacts.json': {'total_count': len(artifacts), 'artifacts': artifacts}}
        for name, doc in documents.items():
            worker.once(self.root / name, doc)
        return documents

    def test_native_observer_api_must_be_complete_and_artifacts_belong_to_it(self):
        docs = self.metadata()
        actual = worker.observer_metadata(self.root, self.pins)
        self.assertEqual(actual['observer_identity'], worker.observer_identity(self.pins))
        self.assertEqual(actual['artifacts'], worker.artifact_roster(self.pins))
        docs['artifacts.json']['artifacts'][0]['workflow_run']['id'] = int(worker.SOURCE_IDENTITY['run_id'])
        (self.root / 'artifacts.json').write_bytes(worker.canonical(docs['artifacts.json']))
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_IDENTITY'):
            worker.observer_metadata(self.root, self.pins)
        docs['run.json'].update(status='in_progress', conclusion=None)
        (self.root / 'run.json').write_bytes(worker.canonical(docs['run.json']))
        with self.assertRaisesRegex(ValueError, 'COMPLETED_OBSERVER_RUN'):
            worker.observer_metadata(self.root, self.pins)

    def test_saved_join_is_compared_to_independent_receipt_readback(self):
        class Args:
            observation_pins = self.root / 'observation-pins.json'
            source_bundle = self.root / 'source-bundle'
            observer_bundle = self.root / 'observer-bundle'
        worker.once(Args.observation_pins, self.opins)
        Args.observer_bundle.mkdir()
        for name in ['observer-' + p + '.zip' for p in worker.PROVIDERS] + ['combined.zip']:
            (Args.observer_bundle / name).write_bytes(b'invented ZIP placeholder; extraction is stubbed')
        out = self.root / 'readback'
        out.mkdir()
        changed = copy.deepcopy(self.combined)
        changed['status'] = 'INCOMPLETE_NO_RETRY'
        with mock.patch.object(observer, 'read_sources', return_value=(self.ledger, [])), \
             mock.patch.object(worker, 'observer_metadata', return_value=self.binding), \
             mock.patch.object(worker, 'extract_observers', return_value=worker.canonical(self.combined) + b'\n'), \
             mock.patch.object(observer, 'combine', return_value=changed) as recompute:
            with self.assertRaisesRegex(ValueError, 'SAVED_JOIN_MUST_EQUAL'):
                worker.readback(Args, self.pins, old, observer, self.root / 'private', out)
            recompute.assert_called_once()
        self.assertEqual(list(out.iterdir()), [])

    def test_saved_source_validator_recomputes_selection_and_keeps_three_runs(self):
        pins = self.execution_pins()
        selected = self.select()
        combined_raw = worker.canonical(self.combined) + b'\n'
        binding = worker.make_source_binding(pins, self.opins, self.ledger, self.combined,
                                             worker.sha(combined_raw), self.binding)
        started = {'schema': worker.SCHEMA, 'test_backend': False, 'source_identity': worker.SOURCE_IDENTITY,
            'observer_identity': worker.observer_identity(pins), 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
            'source_denominator': 144, 'valid_source_count': 144, 'source_dispositions': self.ledger['source_dispositions'],
            'training_pins_sha256': worker.sha(worker.canonical(pins)), 'worker_code_sha256': pins['code_sha256'],
            'selection_policy_sha256': worker.SELECTION_SHA, 'maximum_optimizer_updates': 96,
            'no_retry': True, 'run_id': '99000003', 'run_attempt': '1', 'github_sha': '3' * 40,
            **dict.fromkeys(worker.FALSE_FLAGS, False)}
        docs = {'STARTED.json': started, 'SOURCE_BINDING.json': binding, 'OBSERVER_BINDING.json': self.binding,
                'SOURCE_LEDGER.json': self.ledger, 'COMBINED_OBSERVATIONS.json': self.combined,
                'SELECTION.json': selected, 'TRAINING_PINS.json': pins, 'OBSERVATION_PINS.json': self.opins}
        for name, value in docs.items():
            worker.once(self.root / name, value)
        before = dict(os.environ)
        saved = worker.saved_source_evidence(self.root)
        self.assertEqual(saved, docs)
        self.assertEqual(os.environ, before)
        started['run_id'] = worker.SOURCE_IDENTITY['run_id']
        (self.root / 'STARTED.json').write_bytes(worker.canonical(started))
        with self.assertRaisesRegex(ValueError, 'THREE_DISTINCT_RUN'):
            worker.saved_source_evidence(self.root)


if __name__ == '__main__':
    unittest.main()
