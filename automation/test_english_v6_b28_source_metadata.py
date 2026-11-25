"""Invented metadata/ZIP fixtures only; never select from a real new catalog."""
from __future__ import annotations

import ast
import base64
import builtins
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


PATH = Path(__file__).with_name('english_v6_b28_source_metadata.py')
m = load(PATH, 'b28_metadata_fixtures')


def fixture_api(pin=None):
    pin = m.SOURCE_PIN if pin is None else pin
    return dict(run=dict(id=pin['run_id'], head_sha=pin['run_head'], run_attempt=1, event='push',
        status='completed', conclusion='success', head_branch=m.SOURCE_BRANCH, path=m.SOURCE_WORKFLOW,
        repository=dict(full_name='jiying2007/kws-data', private=False)),
        artifact=dict(id=pin['id'], size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'], expired=False,
            name='english-v6-b27-primary-source-' + str(pin['run_id']),
            workflow_run=dict(id=pin['run_id'], head_sha=pin['run_head'])),
        commit=dict(sha=pin['run_head'], parents=[dict(sha=m.SOURCE_PARENT)], files=[dict(filename=m.SOURCE_WORKFLOW)]))


def fixture_source():
    old, synthetic, natural = {}, [], []
    for role, start in (('primary', 7000), ('replication', 8000)):
        speakers = [str(i) for i in range(start, start + 24)]
        readers = [str(i) for i in range(start + 24, start + 36)]
        roster = [dict(speaker=p, source_id=p + '-1-' + str(i)) for p in readers for i in (0, 1)]
        old[role] = dict(synthetic=dict(speaker_id_map={str(i): 'p' + p for i, p in enumerate(speakers)}),
            natural=dict(speakers=readers, roster=roster))
        if role == 'primary':
            for p in speakers:
                for cell in range(4):
                    ident = 'fixture-synthetic-' + p + '-' + str(cell)
                    synthetic.append(dict(id=ident, cohort='synthetic', source_group_id='piper-libritts-p' + p,
                        source_speaker_id='p' + p, pcm_sha256=m.digest(ident.encode())))
            for row in roster:
                ident = 'fixture-natural-' + row['source_id']
                natural.append(dict(row, id=ident, cohort='natural', source_group_id='librispeech-train-clean-100-' + row['speaker'],
                    pcm_sha256=m.digest(ident.encode())))
    history = dict(all_prior_replication_sealed=True, all_pcm_sha256=[])
    history_raw = m.encoded(history)
    execution = dict(prepared_parent=m.SOURCE_PARENT, run_attempt=1, run_branch='refs/heads/' + m.SOURCE_BRANCH,
        run_head=m.SOURCE_PIN['run_head'], run_id=str(m.SOURCE_PIN['run_id']))
    # Two rejected originals remain in the complete exposure. They are invented.
    manifest = dict(schema='english-v6-b27-fresh-combined-source-v1', panel='primary', selected_original_slots=120,
        source_execution=execution, assessment_sha256=m.ORIGINAL_ASSESSMENT['sha256'],
        original_training_assessment_sha256=m.ORIGINAL_ASSESSMENT['sha256'],
        source_ancestry_sha256=m.digest(history_raw), history_receipt_sha256=m.digest(history_raw),
        new_kws_forwards=0, replication_opened=False, synthetic=dict(rows=synthetic),
        natural=dict(rows=natural[:-2], source_rejections=[dict(r, reason='duration') for r in natural[-2:]]),
        rows=synthetic + natural[:-2])
    raw = {'MANIFEST.json': m.encoded(manifest), 'HISTORY.json': history_raw}
    return old, raw, {n: m.digest(v) for n, v in raw.items()}


def source_documents():
    old, raw, pins = fixture_source()
    with mock.patch.object(m, 'SOURCE_MEMBERS', pins), mock.patch.object(m, 'original_assessment', return_value=old):
        result = m.derive_original_metadata(raw, fixture_api(), m.SOURCE_PIN)
    return result, old, raw, pins


def row_fixture(speaker, number, split, text='ONE TWO THREE FOUR'):
    ident = speaker + '-1-' + str(number)
    member = 'LibriSpeech/' + split + '/' + speaker + '/1/' + speaker + '-1.trans.txt'
    return dict(speaker=speaker, chapter='1', source_id=ident, transcript=text,
        transcript_file_sha256=m.digest(member.encode()), transcript_member=member)


def text_catalog_fixture(speakers, split='train-clean-360', texts=None):
    transcripts, members = [], []
    for speaker in speakers:
        lines = texts if texts is not None else ['ONE TWO THREE FOUR', 'FIVE SIX SEVEN EIGHT', 'ANOTHER FOUR WORDS HERE']
        text = ''.join(speaker + '-1-' + str(i) + ' ' + line + '\n' for i, line in enumerate(lines))
        path = 'LibriSpeech/' + split + '/' + speaker + '/1/' + speaker + '-1.trans.txt'
        transcripts.append(dict(path=path, text=text, bytes=len(text.encode()), sha256=m.digest(text.encode())))
        for i in range(len(lines)):
            member = path.rsplit('/', 1)[0] + '/' + speaker + '-1-' + str(i) + '.flac'
            members.append(dict(kind='opaque_flac', path=member, bytes=30, sha256=m.digest(member.encode())))
    return dict(transcripts=transcripts, corpus=dict(split=split)), dict(members=members)


def panel_inputs(natural_count=40):
    documents, _, _, _ = source_documents()
    exposure, origin = documents['b27-source-exposure.json'], documents['b27-source-origin.json']
    mapping = {'p' + str(p): i for i, p in enumerate(range(1000, 1064))}
    readers100 = {str(p) for p in range(2000, 2018)}
    readers360 = {str(p) for p in range(3000, 3000 + max(0, natural_count - 18))}
    catalog, index = text_catalog_fixture(sorted(readers360))
    eligible100 = {p: [row_fixture(p, i, 'train-clean-100') for i in range(3)] for p in readers100}
    base = SimpleNamespace(canonical=lambda p: str(int(str(p).removeprefix('p'))),
        source_person=m.canonical_person, ordered=lambda values: sorted(set(values), key=int),
        rank=lambda salt, category, identity: (m.digest((salt + '/' + category + '/' + identity).encode()), identity),
        read_inputs=lambda: {'piper-config.json': {'speaker_id_map': mapping}, 'FULL_METADATA_CATALOG.json': {}},
        catalog_rows=lambda value: (readers100, eligible100), BLOCK=set('HEY HAY MERO MEERO MIRO MARROW MERROW'.split()),
        INPUTS={'FULL_METADATA_CATALOG.json': '1' * 64, 'piper-config.json': '2' * 64},
        PIPER={'fixture': True}, PROVIDERS=['qwen', 'whisper', 'parakeet', 'xlsr', 'lv60'])
    history = dict(all_pcm_sha256=[m.digest(('old-' + str(i)).encode()) for i in range(3063)],
        all_source_groups=['prior-group'], canonical_people=['6000'], synthetic_excluded_people=['6000'],
        natural_excluded_people=['6000', *map(str, range(1000, 1064))], all_prior_replication_sealed=True,
        prior_never_reclaimed_reserves={'fixture': ['9000']}, prior_explicit_and_conservative_roster_people=['6000'])
    old = dict(source_format={'fixture': 'whole'}, metadata_selection={'algorithm': 'fixed hash ordering'},
        training={'optimizer_allowed': False, 'new_training_audio': False},
        frozen_model_pt_sha256={'B0': '3' * 64, 'B9': '4' * 64, 'control': m.CONTROL_PT, 'challenger': None},
        source_gate=dict(synthetic_positive=12, synthetic_ordinary=24, synthetic_paired=8, natural_originals=12, natural_speakers=8),
        gains=['1.0', '0.75', '0.5'], quality_criteria={'unchanged': True},
        execution_authorized=False, source_execution_authorized=False, primary_scoring_authorized=False,
        replica_opening_authorized=False, engineering_authorized=False, optimizer_allowed=False, promotion_allowed=False)
    for role in ('primary', 'replication'):
        old[role] = dict(natural=dict(upstream={'split': 'train-clean-100'}, corpus_split='train-clean-100',
            full_metadata_catalog_sha256='1' * 64, utterances_per_speaker=2, requested_speakers=12,
            source_slots=24, minimum_approved_speakers=8, minimum_approved_utterances=12, training_allowed=False))
    return dict(base=base, old=old, history=history, exposure=exposure, origin=origin, hashes={'fixture/input.json': '5' * 64},
        design={'objective_schema': 'english-v6-b28-natural-suffix-no-event-v1'}, opening={},
        catalog_origin={'catalog_members': {'FULL_METADATA_CATALOG.json': {'sha256': '6' * 64}}},
        complete=dict(catalog=catalog, index=index, report={'invented_fixture': True}))


def zip_bytes(path, members):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    return dict(bytes=path.stat().st_size, sha256=m.digest(path.read_bytes()))


def catalog_proof_fixture(directory):
    """Real capture mechanics; every stream byte and utterance is invented."""
    collector = load(m.ROOT / m.COLLECTOR, 'b28_collector_fixture_source')
    helpers = load(m.HERE / 'test_english_v6_train360_metadata.py', 'b28_collector_fixture_helpers')
    compressed, spec = helpers.all_census_transport_fixture()
    catalog, index, summary = collector.scan_stream(io.BytesIO(compressed), spec, (m.ROOT / collector.CENSUS).read_bytes())
    # Only the invented compressed fixture identity changes; all captured source
    # bytes, real complete census and original artifact closure checks remain.
    adapter = SimpleNamespace(**vars(collector))
    adapter.registration = lambda: copy.deepcopy(spec)
    execution = dict(run_id='123', run_head='a' * 40, run_branch=collector.BRANCH, run_attempt=1, prepared_parent='b' * 40)
    producer_members, file_rows = {}, []
    for name in collector.paths():
        raw = ((m.ROOT / collector.TEMPLATE).read_text().replace(collector.TOKEN, execution['prepared_parent']).encode()
            if name == collector.WORKFLOW else (m.ROOT / name).read_bytes())
        producer_members['repository/' + name] = raw
        file_rows.append(dict(path=name, bytes=len(raw), sha256=m.digest(raw),
            git_blob_sha1=hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()))
    proof = dict(schema='english-v6-train360-metadata-producer-v1', execution=execution, files=file_rows,
        frozen_inputs=collector.FROZEN, metadata_only_counters=spec['metadata_only_counters'])
    proof_raw = collector.canonical(proof)
    producer_members['PRODUCER.json'] = proof_raw
    authorization = dict(schema='english-v6-train360-metadata-authorization-v1', execution=execution,
        producer_sha256=m.digest(proof_raw), registration_sha256=collector.FROZEN[collector.REGISTRATION],
        maximum_archive_gets=1, scientific_panel_budget_consumed=False, metadata_only_counters=spec['metadata_only_counters'])
    run = dict(id=123, head_sha=execution['run_head'], head_branch=collector.BRANCH.removeprefix('refs/heads/'),
        path=collector.WORKFLOW, event='push', run_attempt=1, status='completed', conclusion='success',
        repository=dict(full_name=collector.REPO, private=False))
    marker = dict(name=collector.MARKER, expired=False, workflow_run=dict(id=123, head_sha=execution['run_head']))
    prepare_api = dict(runs=dict(total_count=1, workflow_runs=[run]), artifacts=dict(total_count=0, artifacts=[]))
    collect_api = dict(runs=dict(total_count=1, workflow_runs=[run]), artifacts=dict(total_count=1, artifacts=[marker]))
    catalog_raw, index_raw = collector.canonical(catalog), collector.canonical(index)
    pins = {'FULL_METADATA_CATALOG.json': dict(bytes=len(catalog_raw), sha256=m.digest(catalog_raw)),
        'FULL_MEMBER_INDEX.json': dict(bytes=len(index_raw), sha256=m.digest(index_raw))}
    validation = dict(schema='english-v6-train360-metadata-validation-v1', status='COMPLETE_AUTHENTICATED_METADATA',
        execution=execution, corpus=spec['corpus'], summary=summary, catalog=pins['FULL_METADATA_CATALOG.json'],
        member_index=pins['FULL_MEMBER_INDEX.json'], metadata_only_counters=spec['metadata_only_counters'],
        eligibility_computed=False, selected_people=None, source_policy_change=False,
        scientific_panel_budget_consumed=False, source_continuation_authorized=False)
    validation_raw = collector.canonical(validation)
    pins['VALIDATION.json'] = dict(bytes=len(validation_raw), sha256=m.digest(validation_raw))
    result = dict(schema='english-v6-train360-metadata-acquisition-result-v1', status='COMPLETE_AUTHENTICATED_METADATA',
        execution=execution, summary=summary, catalog=validation['catalog'], member_index=validation['member_index'],
        validation_sha256=m.digest(validation_raw), archive_get_attempted=1, archive_response_returned=1,
        algorithm_result=None, scientific_panel_budget_consumed=False, metadata_only_counters=spec['metadata_only_counters'])
    receipts = {'AUTHORIZATION.json': collector.canonical(authorization),
        'PREPARATION_API.json': collector.canonical(prepare_api), 'COLLECTION_API.json': collector.canonical(collect_api),
        'TRANSFER_STARTED.json': b'{}', 'GET_ATTEMPT.json': b'{}', 'HTTP_RESPONSE.json': b'{}', 'PROGRESS.jsonl': b'',
        'RESULT.json': collector.canonical(result), 'COMPLETE/FULL_METADATA_CATALOG.json': catalog_raw,
        'COMPLETE/FULL_MEMBER_INDEX.json': index_raw, 'COMPLETE/VALIDATION.json': validation_raw}
    directory = Path(directory)
    (directory / 'originals').mkdir()
    (directory / 'api').mkdir()
    artifacts, files, transport, api_artifacts = {}, {}, {}, []
    names = dict(producer='english-v6-train360-metadata-producer-123', consumed=collector.MARKER,
        metadata='english-v6-train360-metadata-123')
    for number, (role, members) in enumerate((('producer', producer_members),
        ('consumed', {'CONSUMPTION.json': collector.canonical(dict(authorization, status='METADATA_TRANSPORT_CONSUMED_UNKNOWN'))}),
        ('metadata', receipts)), 1):
        name = 'originals/' + role + '.zip'
        pin = zip_bytes(directory / name, members)
        files[name] = pin
        transport[name] = dict(artifact_id=number)
        artifacts[role] = dict(pin, id=number, run_id=123, run_head=execution['run_head'])
        api_artifacts.append(dict(id=number, name=names[role], size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'],
            expired=False, workflow_run=dict(id=123, head_sha=execution['run_head'])))
    api = dict(run=run, commit=dict(sha=execution['run_head'], parents=[dict(sha=execution['prepared_parent'])],
        files=[dict(filename=collector.WORKFLOW)]), artifacts=dict(total_count=3, artifacts=api_artifacts))
    for key, value in api.items():
        raw = collector.canonical(value)
        name = 'api/' + key + '.json'
        (directory / name).write_bytes(raw)
        files[name] = dict(bytes=len(raw), sha256=m.digest(raw))
        transport[name] = dict(original_api_base64=base64.b64encode(raw).decode())
    origin = dict(schema='english-v6-b28-original-train360-catalog-registration-v1', execution=execution,
        files=files, transport=transport, artifacts=artifacts, producer_manifest_sha256=m.digest(proof_raw),
        catalog_members=pins, metadata_only=True, scientific_execution_authorized=False)
    return origin, adapter, receipts, producer_members


class MetadataTests(unittest.TestCase):
    def test_five_unbound_pins_refuse_before_any_selection_or_catalog_read(self):
        for function in (m.inputs, m.validate, m.validate_registration):
            with self.subTest(function=function.__name__), mock.patch.object(m, '_previous', side_effect=AssertionError('no old or new rank')), \
                 self.assertRaisesRegex(ValueError, 'remain unbound'):
                function()

    def test_exact_prespecified_salts(self):
        self.assertEqual(m.SALTS, {role: {kind: 'english-v6-b28-natural-event-' + role + '-' + kind + '-20261125-v1'
            for kind in ('piper', 'natural')} for role in ('primary', 'replication')})
        self.assertEqual(len({value for row in m.SALTS.values() for value in row.values()}), 4)

    def test_invented_full120_includes_rejected_natural_and_all72_people(self):
        documents, _, _, _ = source_documents()
        exposure = documents['b27-source-exposure.json']
        self.assertEqual(len(exposure['rows']), 120)
        self.assertEqual(len(exposure['all_excluded_people']), 72)
        self.assertEqual(len(exposure['sealed_replication_people']), 36)
        self.assertTrue(any(row['id'] == 'fixture-natural-7035-1-1' for row in exposure['rows']))
        self.assertFalse(documents['b27-source-origin.json']['quality_results_read'])
        self.assertTrue(all(set(r) == set(exposure['selection_fields']) for r in exposure['rows']))

    def test_source_identity_rejects_wrong_pin_attempt_branch_and_parent_before_open(self):
        api = fixture_api()
        with mock.patch.object(m, 'archive', side_effect=AssertionError('no unbound archive read')):
            with self.assertRaises(ValueError):
                m.read_original_metadata('/missing', api, dict(m.SOURCE_PIN, id=1))
            for key, value in (('run_attempt', 2), ('conclusion', 'failure'), ('head_branch', 'other')):
                changed = copy.deepcopy(api)
                changed['run'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m.read_original_metadata('/missing', changed, m.SOURCE_PIN)

    def test_source_archive_reads_only_two_original_metadata_members(self):
        old, raw, pins = fixture_source()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zip'
            pin = dict(m.SOURCE_PIN, **zip_bytes(path, dict(raw, **{'forbidden.wav': b'not audio'})))
            actual_read, seen = zipfile.ZipFile.read, []
            def read(archive, name, *args, **kwargs):
                seen.append(name)
                self.assertIn(name, ('MANIFEST.json', 'HISTORY.json'))
                return actual_read(archive, name, *args, **kwargs)
            with mock.patch.object(m, 'SOURCE_PIN', pin), mock.patch.object(m, 'SOURCE_MEMBERS', pins), \
                 mock.patch.object(m, 'original_assessment', return_value=old), mock.patch.object(zipfile.ZipFile, 'read', read):
                result = m.read_original_metadata(path, fixture_api(pin), pin)
            self.assertEqual(seen, ['MANIFEST.json', 'HISTORY.json'])
            self.assertEqual(len(result['b27-source-exposure.json']['rows']), 120)

    def test_changed_or_dropped_original_rows_stop_even_with_fixture_hash_repin(self):
        old, raw, _ = fixture_source()
        for change in ('drop_reject', 'duplicate', 'person'):
            changed = m.document(raw['MANIFEST.json'])
            if change == 'drop_reject': changed['natural']['source_rejections'].pop()
            elif change == 'duplicate': changed['synthetic']['rows'][0]['pcm_sha256'] = changed['synthetic']['rows'][1]['pcm_sha256']
            else: changed['synthetic']['rows'][0]['source_speaker_id'] = 'p999999'
            altered = dict(raw, **{'MANIFEST.json': m.encoded(changed)})
            pins = {n: m.digest(v) for n, v in altered.items()}
            with self.subTest(change=change), mock.patch.object(m, 'SOURCE_MEMBERS', pins), \
                 mock.patch.object(m, 'original_assessment', return_value=old), self.assertRaises(ValueError):
                m.derive_original_metadata(altered, fixture_api(), m.SOURCE_PIN)

    def test_unchanged_text_eligibility_all_rows_preserved_before_filter(self):
        texts = ['A B C D', ' '.join(['WORD'] * 14), 'A B C', ' '.join(['WORD'] * 15),
            'HEY A B C', 'A B C D.', 'A  B C D', "DON'T CHANGE THESE WORDS"]
        catalog, index = text_catalog_fixture(['3000'], texts=texts)
        census, eligible = m.train360_rows(catalog, index, panel_inputs()['base'])
        self.assertEqual(census, {'3000'})
        self.assertEqual([r['transcript'] for r in eligible['3000']], [texts[0], texts[1], texts[-1]])
        self.assertTrue(all(r['corpus_split'] == r['archive_key'] == 'train-clean-360' for r in eligible['3000']))
        self.assertTrue(all(set(r['original_flac']) == {'bytes', 'sha256'} for r in eligible['3000']))
        self.assertEqual(len(catalog['transcripts'][0]['text'].splitlines()), 8)

    def test_invented_two_panels_keep_full_counts_gates_and_sealed_history(self):
        context = panel_inputs()
        with mock.patch.object(m, 'inputs', return_value=context):
            documents, _ = m.build_documents()
        ancestry, assessment = documents['ancestry.json'], documents['assessment.json']
        self.assertEqual(len(ancestry['all_pcm_sha256']), 3183)
        self.assertEqual(ancestry['prior_never_reclaimed_reserves'], context['history']['prior_never_reclaimed_reserves'])
        self.assertEqual(assessment['source_gate'], context['old']['source_gate'])
        self.assertEqual(assessment['quality_criteria'], context['old']['quality_criteria'])
        people = []
        for role in ('primary', 'replication'):
            selected = assessment[role]
            synthetic = [str(int(v.removeprefix('p'))) for v in selected['synthetic']['speaker_id_map'].values()]
            natural = selected['natural']['speakers']
            self.assertEqual((len(synthetic), len(natural), len(selected['natural']['roster'])), (24, 12, 24))
            self.assertEqual(len(documents[role + '.json']['rows']), 96)
            self.assertFalse(selected['natural']['opened'])
            self.assertTrue(all(sum(r['speaker'] == p for r in selected['natural']['roster']) == 2 for p in natural))
            people.extend(synthetic + natural)
        self.assertEqual(len(people), len(set(people)))
        self.assertEqual(len(people), 72)
        self.assertFalse(set(people) & set(context['exposure']['all_excluded_people']))
        self.assertFalse(assessment['optimizer_allowed'])

    def test_insufficient_pool_fails_before_any_new_ranking(self):
        context = panel_inputs(natural_count=23)
        context['base'].rank = mock.Mock(side_effect=AssertionError('no rank before full pool'))
        with mock.patch.object(m, 'inputs', return_value=context), self.assertRaisesRegex(ValueError, 'two full independent panels'):
            m.build_documents()
        context['base'].rank.assert_not_called()

    def test_cross_alias_pool_rejected_before_selection(self):
        context = panel_inputs()
        context['history']['natural_excluded_people'].remove('1000')
        with mock.patch.object(m, 'inputs', return_value=context), self.assertRaisesRegex(ValueError, 'Piper alias'):
            m.build_documents()

    def test_freeze_cannot_reenter_after_any_document_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'primary.json').write_text('{}')
            with mock.patch.object(m, 'PANELS', folder), mock.patch.object(m, 'build_documents', side_effect=AssertionError('no redraw')), \
                 self.assertRaisesRegex(ValueError, 'freeze only once'):
                m.freeze()

    def test_bound_registration_is_not_a_catalog_context(self):
        with mock.patch.object(m, 'catalog_registration', return_value={'invented': True}):
            with self.assertRaisesRegex(ValueError, 'catalog_scope required'):
                m.require_catalog()

    def test_original_catalog_fixture_complete_scope_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, collector, _, _ = catalog_proof_fixture(root)
            with mock.patch.object(m, 'catalog_registration', return_value=origin), mock.patch.object(m, '_collector', return_value=collector), \
                 mock.patch.object(socket, 'socket', side_effect=AssertionError('no network')), \
                 mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('no subprocess')):
                with m.catalog_scope(root) as report:
                    self.assertEqual(report['summary']['census_readers'], 921)
                    self.assertEqual(report['summary']['utterances'], 921)
                    self.assertEqual(len(m.require_catalog()['catalog']['transcripts']), 921)
                    self.assertFalse(report['scientific_execution_authorized'])
                with self.assertRaises(ValueError):
                    m.require_catalog()

    def test_original_catalog_changed_file_missing_file_and_extra_file_stop(self):
        for change in ('altered', 'missing', 'extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                origin, collector, _, _ = catalog_proof_fixture(root)
                if change == 'altered': (root / 'api/run.json').write_bytes(b'{}')
                elif change == 'missing': (root / 'originals/consumed.zip').unlink()
                else: (root / 'invented-supplement.json').write_bytes(b'{}')
                with mock.patch.object(m, 'catalog_registration', return_value=origin), \
                     mock.patch.object(m, '_collector', return_value=collector), self.assertRaises(ValueError):
                    with m.catalog_scope(root): pass

    def test_selected_only_catalog_rejected_even_after_member_pins_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            origin, collector, receipts, _ = catalog_proof_fixture(Path(directory))
            raw = {n: receipts['COMPLETE/' + n] for n in m.CATALOG_MEMBERS}
            catalog = m.document(raw['FULL_METADATA_CATALOG.json'])
            catalog['transcripts'].pop()
            raw['FULL_METADATA_CATALOG.json'] = collector.canonical(catalog)
            origin['catalog_members']['FULL_METADATA_CATALOG.json'] = dict(bytes=len(raw['FULL_METADATA_CATALOG.json']),
                sha256=m.digest(raw['FULL_METADATA_CATALOG.json']))
            with self.assertRaises(RuntimeError):
                m._catalog_complete(raw, origin, collector)

    def test_catalog_registration_checks_full_transport_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            origin, _, _, _ = catalog_proof_fixture(Path(directory))
            with mock.patch.object(m, 'frozen', return_value=m.encoded(origin)):
                self.assertEqual(m.catalog_registration(), origin)
            changed = copy.deepcopy(origin)
            changed['transport']['api/run.json']['original_api_base64'] = base64.b64encode(b'{}').decode()
            with mock.patch.object(m, 'frozen', return_value=m.encoded(changed)), self.assertRaisesRegex(ValueError, 'API bytes'):
                m.catalog_registration()

    def test_no_scientific_or_network_imports_and_no_assert_guards(self):
        allowed = {'__future__', 'argparse', 'base64', 'contextlib', 'contextvars', 'copy', 'hashlib',
            'importlib', 'json', 'pathlib', 're', 'stat', 'zipfile'}
        tree = ast.parse(PATH.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom): imported.add(node.module.split('.')[0])
        self.assertLessEqual(imported, allowed)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        self.assertNotIn('extractall', PATH.read_text())


if __name__ == '__main__':
    unittest.main()
