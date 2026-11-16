"""Bounded invented metadata/byte fixtures; no models, training, or real C.

Original source readers retain their own tests. Package tests explicitly stub
the saved-source validator and exercise only downstream saved-byte checks.
"""
from __future__ import annotations
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load('_snapshot160_tests_worker', 'english_snapshot160_train_once.py')
p = load('_snapshot160_tests_package', 'english_snapshot160_package.py')
new_recipe = load('_snapshot160_tests_recipe', 'vosen_schwa16_recipe.py')
fixtures = load('_snapshot160_tests_bytes', 'test_final_embedding_package.py')


def ready_pins():
    pins = w.pins_template()
    for key in ('new_worker_sha256', 'new_recipe_code_sha256', 'new_recipe_sha256',
                'new_execution_pins_sha256', 'new_terminal_evidence_canonical_sha256'):
        pins[key] = '1' * 64
    pins.update(new_run={'run_id': 1000, 'github_sha': 'a' * 40, 'run_attempt': 1, 'conclusion': 'failure'},
        new_artifacts={key: {'artifact_id': i + 1, 'artifact_name': 'schwa16-source' if key == 'source'
            else 'schwa16-observer-' + key, 'zip_bytes': 100, 'zip_sha256': str(i + 1) * 64}
            for i, key in enumerate(('source', *w.PROVIDERS))})
    return pins


def cohort_fixture():
    recipe = w.old_observer.recipe.recipe()
    fresh = new_recipe.recipe()
    sources, decisions, observations = [], [], {}
    voice_vosen = CounterFixture()
    failed = next(c['cell_id'] for c in recipe['cells'] if c['voice_id'] == 'aiden' and c.get('requested_keyword_index') == 2)
    for cell in recipe['cells']:
        cid, label = cell['cell_id'], cell.get('requested_keyword_index') or 0
        source = {**cell, 'source_status': 'GENERATED_UNADMITTED', 'source_disposition': 'VALID_UNOBSERVED_SOURCE',
            'frames': 16000, 'wav_sha256': w.sha(('invented-wave/' + cid).encode()),
            'pcm_sha256': w.sha(('invented-pcm/' + cid).encode())}
        useful = cid != failed
        if label == 1:
            voice_vosen[cell['voice_id']] += 1
            useful = voice_vosen[cell['voice_id']] <= {'aiden': 3, 'eric': 2, 'ryan': 3}.get(cell['voice_id'], 0)
        if cid == failed:
            source.update(source_status='FAILED_NO_RETRY', source_disposition='FAILED_NO_RETRY',
                          frames=None, wav_sha256=None, pcm_sha256=None)
        sources.append(source)
        decision = {'cell_id': cid, 'role': cell['role'], 'weak_intent_class': label,
            'source_status_unchanged': source['source_status'], 'source_disposition': source['source_disposition'],
            'eligible_for_requested_intent_research': useful,
            'evidence_status': 'USEFUL_FOR_INTENT_RESEARCH' if useful else 'UNKNOWN',
            'phone_review_status': 'ARTIFICIAL_UNVERIFIED_FIXTURE', **dict.fromkeys(w.FALSE_FLAGS, False)}
        decisions.append(decision)
        if useful:
            observations[cid] = {'source_pcm_sha256': source['pcm_sha256'],
                'observer_artifact_sha256': '2' * 64, 'observation_sha256': '3' * 64,
                'evidence_status': 'USEFUL_FOR_INTENT_RESEARCH', 'phone_review_status': 'ARTIFICIAL_UNVERIFIED_FIXTURE'}
    joined = []
    for cell in fresh['cells']:
        pcm = w.sha(('invented-new-pcm/' + cell['cell_id']).encode())
        row = {**cell, 'source_status': 'GENERATED_UNADMITTED', 'source_disposition': 'VALID_UNOBSERVED_SOURCE',
            'frames': 16000, 'wav_sha256': w.sha(('invented-new-wave/' + cell['cell_id']).encode()), 'pcm_sha256': pcm}
        values = {provider: {'observation_status': 'COMPLETE', 'observation_attempted': True,
            'pcm_sha256': pcm, 'text': 'Hey Vosen'} for provider in w.PROVIDERS}
        values['phones'] = {'observation_status': 'COMPLETE', 'observation_attempted': True,
                           'pcm_sha256': pcm, 'raw_ipa_tokens': ['h', 'eɪ', 'v', 'oʊ', 'z', 'ə', 'n']}
        joined.append({**row, 'observations': values})
    ledger = {'rows': sources, 'valid_count': 143}
    selection = {'decisions': decisions, 'observations': observations}
    reviewed = {'recipe': fresh, 'rows': joined, 'binding': {'observer_summaries_sha256':
        {provider: '4' * 64 for provider in w.PROVIDERS}}, 'status': 'FOUR_OBSERVERS_COMPLETE'}
    return recipe, ledger, selection, reviewed


class CounterFixture(dict):
    def __missing__(self, key):
        return 0


def api_fixture(root):
    pins = ready_pins()
    run = {'id': 1000, 'head_sha': 'a' * 40, 'run_attempt': 1, 'status': 'completed', 'conclusion': 'failure',
        'event': 'push', 'head_branch': 'main', 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
    names = ['source-contract', 'generate', 'combine'] + ['observe (' + x + ')' for x in w.PROVIDERS]
    rows = [{'id': i + 50, 'name': name, 'run_id': 1000, 'run_attempt': 1, 'head_sha': 'a' * 40,
             'status': 'completed', 'conclusion': 'failure' if name in ('generate', 'observe (qwen)', 'combine')
             else 'success'} for i, name in enumerate(names)]
    terminal = {'run_id': 1000, 'head_sha': 'a' * 40, 'run_attempt': 1, 'status': 'completed',
                'conclusion': 'failure', 'jobs': sorted(rows, key=lambda r: r['name'])}
    pins['new_terminal_evidence_canonical_sha256'] = w.sha(w.canonical(terminal))
    artifacts = [{'id': entry['artifact_id'], 'name': entry['artifact_name'], 'expired': False,
        'size_in_bytes': entry['zip_bytes'], 'digest': 'sha256:' + entry['zip_sha256'],
        'workflow_run': {'id': 1000, 'head_sha': 'a' * 40}} for entry in pins['new_artifacts'].values()]
    values = {'run.json': run, 'jobs.json': {'total_count': 7, 'jobs': rows},
              'artifacts.json': {'total_count': 5, 'artifacts': artifacts}}
    for name, value in values.items():
        (root / name).write_bytes(w.canonical(value) + b'\n')
    return pins, values


def fake_training(root):
    docs = fixtures.fake_training(root)
    pins = ready_pins(); pins.update(code_sha256='b' * 64, snapshot_sha256='5' * 64)
    started = docs['STARTED.json']
    started.update(schema=w.SCHEMA, source_denominator=160, original_campaign_status='INCOMPLETE_NO_RETRY',
                   run_attempt='1', snapshot_sha256=pins['snapshot_sha256'], new_tts_asr_phone_calls=0)
    started['training_pins_sha256'] = w.sha(w.canonical(pins))
    binding = {'source_denominator': 160, 'original_campaign_status': 'INCOMPLETE_NO_RETRY',
        'new_batch_status': 'INCOMPLETE_NO_RETRY', 'snapshot_sha256': pins['snapshot_sha256'],
        'transport_manifest_sha256': '6' * 64, 'new_source_dispositions': {'VALID_UNOBSERVED_SOURCE': 15, 'FAILED_NO_RETRY': 1},
        'old_source': {'source_identity': {'run_id': '1'}, 'observer_identity': {'run_id': '2'}},
        'new_source': {'source_identity': {'run_id': '3'}, 'observer_identity': {'run_id': '3'}}}
    docs['SOURCE_BINDING.json'] = binding
    docs['TRAINING_PINS.json'] = pins
    docs['FEATURE_PREFLIGHT.json']['pipeline_source']['commit'] = w.PIPELINE_COMMIT
    docs['RESULT.json'].update(started, status=w.SUCCESS, source_binding=binding,
        new_batch_status=binding['new_batch_status'], exposed_read_speech_tracks={'seconds': 640, 'tracks': 40})
    bound = {'snapshot_sha256': pins['snapshot_sha256'], 'snapshot_transport_manifest_sha256': binding['transport_manifest_sha256'],
        'source_binding_sha256': w.sha(w.canonical(binding)), 'selection_sha256': w.sha(w.canonical(docs['SELECTION.json'])),
        'selection_policy_sha256': w.SELECTION_SHA, 'old_source_ledger_sha256': w.OLD_LEDGER_SHA,
        'old_selection_sha256': w.OLD_SELECTION_SHA, 'new_recipe_sha256': pins['new_recipe_sha256'],
        'new_execution_pins_sha256': pins['new_execution_pins_sha256'], 'new_worker_sha256': pins['new_worker_sha256'],
        'new_recipe_code_sha256': pins['new_recipe_code_sha256'], 'corpus_sha256': w.old.CORPUS['sha256'],
        'reference_sha256': w.old.REFERENCE['member_sha256'],
        'pipeline_source_sha256': w.sha(w.canonical(docs['FEATURE_PREFLIGHT.json']['pipeline_source'])),
        'feature_evidence_sha256': p.identity(root / 'FEATURE_EVIDENCE.json')['sha256'],
        'feature_stream_sha256': p.identity(root / 'actual-embedding-streams.npz')['sha256'],
        'worker_code_sha256': started['worker_code_sha256'], 'legacy_worker_sha256': w.terminal.LEGACY_SHA,
        'terminal_worker_sha256': w.TERMINAL_SHA}
    docs['RESULT.json']['input_binding'] = bound
    docs['TRAINING_STARTED.json']['input_binding'] = bound
    trained = p.read_json(root / 'train/RESULT.json'); trained['input_binding'] = bound
    for name, doc in docs.items():
        fixtures.put(root / name, doc)
    fixtures.put(root / 'train/RESULT.json', trained)
    return docs


class SnapshotTests(unittest.TestCase):
    def test_default_and_disabled_template_do_not_execute(self):
        with patch.object(w.old.subprocess, 'run', side_effect=AssertionError('NO PROCESSES')), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(w.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())['model_calls'], 0)
        self.assertFalse(w.pins_template()['enabled'])
        with self.assertRaises(ValueError):
            w.validate_pins(w.pins_template(), execution=True)

    def test_guard_requires_new_main_once_and_immutable_budget(self):
        pins = ready_pins(); pins.update(enabled=True, code_sha256=w.sha(Path(w.__file__).read_bytes()),
            snapshot_sha256='5' * 64, next_main_before='b' * 40)
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '2000', 'GITHUB_SHA': 'c' * 40}
        event = {'before': 'b' * 40, 'after': 'c' * 40, 'ref': 'refs/heads/main',
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        w.execution_guard(pins, event, env)
        for key, value in [('GITHUB_ACTIONS', 'false'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_RUN_ID', '1000'), ('GITHUB_REF', 'refs/pull/1/merge')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.execution_guard(pins, event, {**env, key: value})
        with self.assertRaises(ValueError):
            w.execution_guard(pins, {**event, 'ref': 'refs/heads/not-main'}, env)
        with self.assertRaises(ValueError):
            w.execution_guard({**pins, 'optimizer_updates': 97}, event, env)

    def test_original_order_failure_and_roles_remain(self):
        data = cohort_fixture(); originals = copy.deepcopy(data)
        result = w.combine_snapshot(*data)
        self.assertEqual(data, originals)
        self.assertEqual(result['selection']['source_order'], [c['cell_id'] for c in data[0]['cells'] + data[3]['recipe']['cells']])
        self.assertEqual(len(result['source_rows']), 160)
        self.assertEqual(sum(r['source_status'] == 'FAILED_NO_RETRY' for r in result['source_rows']), 1)
        self.assertTrue(result['minimum']['passed'])

    def test_role_or_family_inflation_is_rejected(self):
        for key, value in [('role', 'development'), ('source_family_id', 'new-batch-new-family')]:
            data = cohort_fixture(); data[3]['recipe']['cells'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.combine_snapshot(*data)

    def test_partial_new_batch_keeps_other_complete_rows(self):
        data = cohort_fixture(); data[3]['status'] = 'INCOMPLETE_NO_RETRY'
        row = data[3]['rows'][0]; row['observations']['qwen']['observation_status'] = 'FAILED_NO_RETRY'
        result = w.combine_snapshot(*data)
        self.assertEqual(result['selection']['new_batch_status'], 'INCOMPLETE_NO_RETRY')
        self.assertNotIn(row['cell_id'], result['selection']['eligible_cell_ids'])
        self.assertIn(data[3]['rows'][1]['cell_id'], result['selection']['eligible_cell_ids'])
        self.assertTrue(result['minimum']['passed'])

    def test_missing_provider_means_unknown_not_negative(self):
        data = cohort_fixture()
        for row in data[3]['rows']:
            del row['observations']['whisper']
        result = w.combine_snapshot(*data)
        self.assertFalse(result['minimum']['passed'])
        self.assertEqual(result['minimum']['counts']['train/1']['count'], 5)
        self.assertFalse(any(d['eligible_for_requested_intent_research'] for d in result['selection']['decisions'][144:]))

    def test_schwa_predicate_is_unchanged(self):
        data = cohort_fixture()
        for row in data[3]['rows']:
            row['observations']['phones']['raw_ipa_tokens'] = ['v', 'oʊ', 'z', 'ɪ', 'n']
        result = w.combine_snapshot(*data)
        self.assertEqual(result['minimum']['counts']['train/1']['count'], 5)

    def test_same_pcm_is_mandatory_for_complete_observation(self):
        data = cohort_fixture(); data[3]['rows'][0]['observations']['qwen']['pcm_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SAME_PCM'):
            w.combine_snapshot(*data)

    def test_global_duplicate_includes_internally_quarantined_new_rows(self):
        data = cohort_fixture()
        old_row = next(r for r in data[1]['rows'] if r['voice_id'] == 'aiden' and r.get('requested_keyword_index') == 1)
        for row in data[3]['rows'][:2]:
            row.update(pcm_sha256=old_row['pcm_sha256'], source_disposition='DUPLICATE_PCM_QUARANTINED')
        result = w.combine_snapshot(*data)
        self.assertEqual(len(result['selection']['duplicate_groups'][old_row['pcm_sha256']]), 3)
        self.assertNotIn(old_row['cell_id'], result['selection']['eligible_cell_ids'])
        self.assertTrue(next(d for d in data[2]['decisions'] if d['cell_id'] == old_row['cell_id'])['eligible_for_requested_intent_research'])

    def test_duplicate_against_old_unknown_also_excludes_new(self):
        data = cohort_fixture()
        old_row = next(r for r in data[1]['rows'] if r['voice_id'] == 'dylan' and r.get('requested_keyword_index') == 1)
        row = data[3]['rows'][0]; row['pcm_sha256'] = old_row['pcm_sha256']
        for obs in row['observations'].values():
            obs['pcm_sha256'] = row['pcm_sha256']
        result = w.combine_snapshot(*data)
        self.assertNotIn(row['cell_id'], result['selection']['eligible_cell_ids'])

    def test_actual_feature_ids_reapply_minimum_without_libri_substitution(self):
        result = w.combine_snapshot(*cohort_fixture()); rows = result['rows']
        self.assertTrue(result['minimum']['passed'])
        targets = [r['cell_id'] for r in rows if r['role'] == 'train' and r['weak_class'] == 1]
        keep = {r['cell_id'] for r in rows} - set(targets[7:])
        after = w.feature_minimum(rows, keep | {'libri-speaker-invented-' + str(n) for n in range(40)})
        self.assertFalse(after['passed']); self.assertEqual(after['counts']['train/1']['count'], 7)
        self.assertFalse(w.feature_minimum(rows, {'libri-speaker-only'})['passed'])

    def test_transport_format_changes_manifest_not_semantic_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pins, docs = api_fixture(root)
            before = w.new_metadata(root, pins); manifest = w.closed_manifest(root)
            for name, doc in docs.items():
                if name == 'jobs.json':
                    doc['jobs'] = list(reversed(doc['jobs']))
                if name == 'artifacts.json':
                    doc['artifacts'] = list(reversed(doc['artifacts']))
                (root / name).write_text(json.dumps(doc, indent=4, sort_keys=True) + '\n')
            after = w.new_metadata(root, pins)
            self.assertEqual(w.sha(w.canonical(before['semantic'])), w.sha(w.canonical(after['semantic'])))
            self.assertNotEqual(before['api_raw_sha256'], after['api_raw_sha256'])
            self.assertNotEqual(w.sha(w.canonical(manifest)), w.sha(w.canonical(w.closed_manifest(root))))
            # The old reader also hashes its raw API binding. Neither that hash
            # nor the raw API identities may leak indirectly into the snapshot.
            old = {key: '1' * 64 for key in ('source_identity', 'observer_identity',
                'original_campaign_status', 'source_denominator', 'valid_source_count', 'source_dispositions',
                'source_ledger_canonical_sha256', 'terminal_evidence_canonical_sha256', 'recipe_sha256',
                'combined_observations_sha256', 'observer_summaries_sha256')}
            observed = {'artifacts': before['semantic']['artifacts'],
                'observer_terminal_evidence_canonical_sha256': '2' * 64,
                'api_files_sha256': before['api_raw_sha256']}
            fresh = {key: '3' * 64 for key in ('source_identity', 'observer_identity', 'recipe_sha256',
                'source_ledger_canonical_sha256', 'source_result_sha256', 'source_plan_sha256',
                'source_model_lock_sha256', 'observer_summaries_sha256', 'observer_model_locks_sha256')}
            old['observer_artifact_manifest_sha256'] = w.sha(w.canonical(observed))
            semantic_before = w.canonical(w.semantic_source_bindings(old, observed, fresh))
            observed['api_files_sha256'] = after['api_raw_sha256']
            old['observer_artifact_manifest_sha256'] = w.sha(w.canonical(observed))
            self.assertEqual(semantic_before, w.canonical(w.semantic_source_bindings(old, observed, fresh)))

    def test_run_head_and_artifact_semantic_changes_rejected(self):
        for mode in ('run', 'head', 'artifact'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); pins, docs = api_fixture(root)
                if mode == 'run': docs['run.json']['id'] = 1001
                if mode == 'head': docs['run.json']['head_sha'] = 'd' * 40
                if mode == 'artifact': docs['artifacts.json']['artifacts'][0]['digest'] = 'sha256:' + '0' * 64
                for name, value in docs.items(): (root / name).write_bytes(w.canonical(value))
                with self.assertRaises(ValueError): w.new_metadata(root, pins)

    def test_no_missing_artifact_can_be_invented_or_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pins, _ = api_fixture(root)
            pins['new_artifacts']['qwen'] = None
            with self.assertRaisesRegex(ValueError, 'ACTUALLY_ABSENT'):
                w.new_metadata(root, pins)

    def test_zip_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / 'bad.zip'
            with zipfile.ZipFile(path, 'w') as z: z.writestr('../outside', b'invented')
            data = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'UNSAFE_ZIP_MEMBER'):
                w.extract_zip(path, root / 'extract', {'zip_bytes': len(data), 'zip_sha256': w.sha(data)})
            self.assertFalse((root / 'outside').exists())

    def test_package_keeps_partial_source_and_requires_saved_byte_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'training'; docs = fake_training(root)
            with patch.object(p, 'source_evidence', return_value={'STARTED.json': docs['STARTED.json']}):
                evidence, trained, weights, binary = p.training_evidence(root)
                self.assertEqual(evidence['RESULT.json']['new_batch_status'], 'INCOMPLETE_NO_RETRY')
                self.assertEqual(binary['bytes'], 197908)
                (root / 'train/causal-head.npz').write_bytes(b'CHANGED INVENTED BYTES')
                with self.assertRaisesRegex(ValueError, 'SAVED_WEIGHT'):
                    p.training_evidence(root)

    def test_package_stops_on_second_gate_even_with_96_reported_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'training'; docs = fake_training(root)
            docs['MINIMUM_AFTER_FEATURE_GEOMETRY.json'] = {'passed': False}
            fixtures.put(root / 'MINIMUM_AFTER_FEATURE_GEOMETRY.json', docs['MINIMUM_AFTER_FEATURE_GEOMETRY.json'])
            with patch.object(p, 'source_evidence', return_value={'STARTED.json': docs['STARTED.json']}), self.assertRaisesRegex(ValueError, 'BOTH_MINIMUM'):
                p.training_evidence(root)

    def test_package_source_validator_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'training'; fake_training(root)
            with patch.object(p, 'source_evidence', side_effect=ValueError('BAD_REAL_SOURCE')), self.assertRaisesRegex(ValueError, 'BAD_REAL_SOURCE'):
                p.training_evidence(root)


if __name__ == '__main__':
    unittest.main()
