"""Invented B28 SOURCE contracts; no real corpus, codec, model, draw or network."""
from __future__ import annotations

from array import array
import base64
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_profile as profile
import english_v6_b28_source as source
import english_v6_b28_replication_source as replica
import english_v6_b28_natural_sources as natural
import english_v6_b28_replication_natural_sources as replica_natural
import english_v6_b28_source_transport as transport
import english_v6_b28_capacity as capacity
import english_v6_b28_source_producer as producer
import english_v6_b28_original_proof as original_proof
import english_v6_b28_multicorpus_source as multicorpus
import test_english_v6_b28_multicorpus_source as fixtures

HERE = Path(__file__).resolve().parent


@contextmanager
def invented_catalogs():
    census, corpora, texts, indices, payloads, _ = fixtures.fixture()
    with patch.object(multicorpus, 'CORPORA', corpora), patch.object(multicorpus, 'SPEAKERS_SHA', multicorpus.sha(census)):
        values = {key: multicorpus.normalized_catalog(texts[key], key, census, indices[key]) for key in corpora}
        # All twelve invented readers already have two admissible rows. No
        # ranking function is used by this SOURCE integration fixture.
        roster = [r for value in values.values() for r in value['rows']]
        plan = dict(schema='english-v6-b28-multicorpus-staging-plan-v1', panel='primary',
            assessment_sha256='1' * 64, training_readback_sha256='2' * 64, roster=roster,
            corpora=corpora, catalog_pins={key: dict(bytes=len(multicorpus.encoded(value)),
                sha256=multicorpus.sha(multicorpus.encoded(value))) for key, value in values.items()}, limits=multicorpus.LIMITS)
        multicorpus.validate_plan(plan, values)
        yield plan, values, payloads


def write_http(directory, plan, release):
    directory.mkdir()
    for key, corpus in plan['corpora'].items():
        multicorpus.write(directory / (key + '-GET.json'), dict(method='GET', url=corpus['url'],
            headers=[['Accept-encoding', 'identity']], attempt=1, maximum_gets=1, retries=0, redirects=0, ranges=0))
        multicorpus.write(directory / (key + '-RESPONSE.json'), dict(url=corpus['url'], status=200,
            headers=[['Content-Length', str(corpus['bytes'])]], body_saved=False, redirect_rejected=False))
    multicorpus.write(directory / 'RESULT.json', dict(schema='english-v6-b28-whole-original-http-result-v1', panel='primary',
        plan_sha256=multicorpus.sha(multicorpus.encoded(plan)), status='COMPLETE_VERIFIED',
        attempts=sorted(plan['corpora']), responses=sorted(plan['corpora']), retries=0, redirects=0, ranges=0,
        source_budget_authority_granted=False, native_decodes=0, model_calls=0,
        release_sha256=multicorpus.sha(multicorpus.encoded(release))))


def invented_native(root, panel='primary', long_samples=240001):
    """Construct STREAMINFO + opaque invented bytes; SoundFile is always mocked."""
    items, pcms, roster = [], {}, []
    for index in range(24):
        person = str(index // 2 + 1)
        ident = person + '-1-' + str(index % 2 + 1).zfill(4)
        split = 'train-clean-100' if index < 12 else 'train-clean-360'
        member = 'LibriSpeech/' + split + '/' + person + '/1/' + ident + '.flac'
        row = dict(archive_key=split, corpus_split=split, speaker=person, chapter='1', source_id=ident,
            transcript='ONE TWO THREE FOUR', transcript_file_sha256='3' * 64,
            transcript_member=member.rsplit('/', 1)[0] + '/' + person + '-1.trans.txt', flac_member=member, original_flac=None)
        count = long_samples if index == 0 else 7999 if index == 1 else 8000 + index
        pcm = array('h', [1000 + index, -1000 - index] * (count // 2) + ([1000 + index] if count % 2 else []))
        if sys.byteorder != 'little':
            pcm.byteswap()
        pcm = pcm.tobytes()
        packed = (16000 << 44) | (15 << 36) | count
        raw = b'fLaC\x80\x00\x00\x22' + bytes(10) + packed.to_bytes(8, 'big') + hashlib.md5(pcm).digest() + b'INVENTED-NOT-FLAC' + ident.encode()
        path = root / 'NATURAL_STAGING/COMPLETE' / split / (ident + '.flac')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        pin = dict(bytes=len(raw), sha256=profile.sha(raw))
        row['original_flac'] = pin if split == 'train-clean-360' else None
        items.append(dict(row=row, path=path, flac_sha256=pin['sha256'], flac_bytes=pin['bytes'],
            source_group_id=multicorpus.source_group(row), id=multicorpus.natural_id(panel, row)))
        pcms[raw] = pcm
        roster.append(row)
    (root / 'wav').mkdir()
    multicorpus.write(root / 'STAGING_PLAN.json', dict(invented=True))
    multicorpus.write(root / 'NATURAL_STAGING/COMPLETE/RELEASE.json', dict(invented=True))
    return items, pcms, dict(roster=roster, speakers=sorted({r['speaker'] for r in roster}))


class SourceContracts(unittest.TestCase):
    def test_unchanged_native_quarantine_admission_and_whole_wav_code(self):
        frozen = profile.preserved('english_v6_b24q1_source_observe')
        for owner in (source, replica):
            for name in ('observe', 'read_valid', 'load_observations', 'started_document', 'report_document'):
                self.assertIs(getattr(owner.observers, name).__code__, getattr(frozen, name).__code__)
            self.assertIs(owner.observers.owner(), owner)
            self.assertIs(owner.observe.__globals__['lineage'], owner.lineage)
            self.assertIs(owner.screen_document.__code__, profile.preserved('english_v6_b24q1_source').screen_document.__code__)
        for owner in (natural, replica_natural):
            self.assertIs(owner.flac_header.__code__, natural.baseline.flac_header.__code__)
            self.assertIs(owner.whole_wav_facts.__code__, natural.baseline.whole_wav_facts.__code__)
        self.assertEqual(source.observers.POLICY_KEYS, frozen.POLICY_KEYS + ('b28_execution_profile_sha256',))
        self.assertEqual(source.FILES, profile.profile(False)['panel_sha256'])
        self.assertEqual(capacity.SOURCE_LIMIT, 256_000_000)
        self.assertEqual(capacity.CORE_LIMIT, 128_000_000)

    def test_inert_registration_stops_before_output_network_tts_decode_or_observers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            current = profile.profile(False)
            # Invented future pin values only: this is not a bound production
            # profile or evidence of any TRAIN/quality result.
            future = dict(current, training_design_sha256='a' * 64, assessment_sha256='b' * 64,
                          ancestry_sha256='c' * 64, panel_sha256=dict(primary='d' * 64, replication='e' * 64))
            for state, registered in (('current', current), ('future_bound_fixture', future)):
                inert = copy.deepcopy(registered)
                inert.update(training_design_sha256=None, assessment_sha256=None, ancestry_sha256=None,
                             panel_sha256=dict(primary=None, replication=None))
                fixture = root / (state + '-inert-profile.json')
                fixture.write_bytes(profile.canonical(inert))
                for owner in (source, replica):
                    event = root / (state + '-' + owner.PANEL + '-event.json')
                    event.write_bytes(profile.canonical(dict(before='a' * 40, after='b' * 40,
                        ref=owner.lineage.BRANCH, forced=False, deleted=False,
                        repository=dict(full_name=owner.lineage.REPO, private=False))))
                    env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=owner.lineage.REPO,
                        GITHUB_EVENT_NAME='push', GITHUB_REF=owner.lineage.BRANCH, GITHUB_RUN_ATTEMPT='1',
                        GITHUB_RUN_ID='123', GITHUB_SHA='b' * 40, PREPARED_PARENT='a' * 40,
                        GITHUB_WORKSPACE=str(root), GITHUB_EVENT_PATH=str(event))
                    with self.subTest(state=state, panel=owner.PANEL), \
                         patch.object(profile, 'PROFILE', fixture), patch.object(profile, 'DESIGN_SHA', None), \
                         patch.dict(owner.lineage.os.environ, env, clear=True), \
                         patch.object(subprocess, 'Popen', side_effect=AssertionError('no process before admission')) as process, \
                         patch('socket.socket', side_effect=AssertionError('no network before admission')) as socket_call, \
                         patch('socket.create_connection', side_effect=AssertionError('no connection before admission')) as connect, \
                         patch.object(owner.B13, 'render_panel', side_effect=AssertionError('no TTS')) as render, \
                         patch.object(owner.natural, 'extract', side_effect=AssertionError('no native decode')) as decode, \
                         patch.object(owner.observers, 'observe', side_effect=AssertionError('no observer')) as observe, \
                         patch.object(owner.lineage, 'gh', side_effect=AssertionError('no acquisition')) as network:
                        with self.assertRaisesRegex(ValueError, 'INERT: prospective B28 quality metadata is unbound'):
                            owner.generate(root / owner.PANEL, root / 'prior', root / 'prior/NATURAL_STAGING', 'a' * 40)
                        self.assertFalse((root / owner.PANEL).exists())
                        for entry in (process, socket_call, connect, render, decode, observe, network):
                            entry.assert_not_called()
        with self.assertRaises(ValueError):
            profile.require_proof('primary')

    def test_replica_functions_only_use_replica_globals(self):
        self.assertIs(replica_natural.staging_plan.__globals__['lineage'], replica.lineage)
        self.assertIs(replica_natural.read_source.__globals__['plan'], replica_natural.plan)
        self.assertIs(replica.lineage.require_saved_consumption.__globals__['consumption_document'], replica.lineage.consumption_document)
        for owner in (source, replica):
            self.assertEqual(owner.lineage.marker_name('staging'), 'english-v6-b28-' + owner.PANEL + '-whole-original-staging-consumed')
            self.assertIn('/en-v6-b28-' + owner.PANEL + '-sources-once-', owner.lineage.BRANCH)
            with patch.object(owner.lineage, 'read_plan', return_value={owner.PANEL: {'natural': {'identity': owner.PANEL}}}):
                self.assertEqual(owner.natural.plan(), {'identity': owner.PANEL})
        with patch.object(replica.lineage, 'read_plan', return_value={}):
            with self.assertRaises(ValueError):
                replica.source_plan('primary')

    def test_primary_runtime_never_traverses_future_b28_quality(self):
        paths = producer.runtime_paths(HERE.parent, 'primary')
        for name in producer.DORMANT:
            self.assertNotIn('automation/' + name + '.py', paths)
        for name in ('english_v6_b28_training_readback', 'english_v6_b28_source_metadata', 'english_v6_b28_multicorpus_source',
                     'english_v6_b28_original_proof', 'english_v6_b27_primary_no_promotion_readback'):
            self.assertIn('automation/' + name + '.py', paths)
        self.assertTrue({'automation/' + name for name in profile.profile(False)['fixed_base_sha256']} <= paths)

    def test_piper_map_uses_original_config_and_all48_frozen_b28_people(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'invented-piper.json'
            mapping = {'p' + str(10000 + n): n for n in range(48)}
            path.write_text(json.dumps(dict(speaker_id_map=mapping)))
            assessment = {role: dict(synthetic=dict(speaker_id_map={str(n): 'p' + str(10000 + n)
                for n in range(0 if role == 'primary' else 24, 24 if role == 'primary' else 48)}))
                for role in ('primary', 'replication')}
            expected = source.qa.sha(path)
            for owner in (source, replica):
                with patch.object(owner.B13.b12, 'verify_config'), patch.dict(owner.piper_metadata.INPUTS, {'piper-config.json': expected}), \
                     patch.object(owner.lineage, 'read_plan', return_value=assessment), \
                     patch.object(owner.lineage, 'read_ancestry', return_value={'synthetic_excluded_people': []}):
                    owner.validate_config(path)
                    with patch.object(owner.lineage, 'read_ancestry', return_value={'synthetic_excluded_people': ['10000']}):
                        with self.assertRaises(ValueError): owner.validate_config(path)
                    duplicated = copy.deepcopy(assessment)
                    duplicated['replication']['synthetic']['speaker_id_map']['0'] = 'p10000'
                    with patch.object(owner.lineage, 'read_plan', return_value=duplicated):
                        with self.assertRaises(ValueError): owner.validate_config(path)

    def test_transport_cannot_construct_opener_without_consumed_permission(self):
        with patch.object(transport.lineage, 'require_staging_permit', side_effect=ValueError('inert')) as permission, \
             patch.object(transport.urllib.request, 'build_opener') as opener, patch.object(multicorpus, 'stage_panel') as stage:
            with self.assertRaises(ValueError):
                transport.download('/invented/NATURAL_STAGING', '/invented', '0' * 40)
            permission.assert_called_once(); opener.assert_not_called(); stage.assert_not_called()

    def test_complete_two_stream_staging_is_reopened_with_original_http_headers(self):
        with invented_catalogs() as (fixed, catalogs, payloads), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            multicorpus.write(root / 'STAGING_PLAN.json', fixed)
            release = multicorpus.stage_panel(fixed, root / 'NATURAL_STAGING', catalogs,
                lambda corpus: fixtures.Response(payloads[corpus['split']], corpus))
            write_http(root / 'STAGING_HTTP', fixed, release)
            with patch.object(natural, 'staging_plan', return_value=(fixed, catalogs)), \
                 patch.object(multicorpus, 'select_natural_rows', side_effect=AssertionError('SOURCE cannot draw')):
                self.assertEqual(len(natural.read_staging(root)), 24)
                response = root / 'STAGING_HTTP/train-clean-360-RESPONSE.json'
                raw = response.read_bytes()
                changed = json.loads(raw); changed['headers'].append(['Content-Length', str(fixed['corpora']['train-clean-360']['bytes'])])
                response.write_bytes(multicorpus.encoded(changed))
                with self.assertRaises(ValueError): natural.read_staging(root)
                response.write_bytes(raw)
                (root / 'NATURAL_STAGING/extra.flac').write_bytes(b'UNREGISTERED')
                with self.assertRaises(ValueError): natural.read_staging(root)

    def test_staging_request_cannot_hide_retry_or_alternate_origin(self):
        with invented_catalogs() as (fixed, catalogs, payloads), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            multicorpus.write(root / 'STAGING_PLAN.json', fixed)
            release = multicorpus.stage_panel(fixed, root / 'NATURAL_STAGING', catalogs,
                lambda corpus: fixtures.Response(payloads[corpus['split']], corpus))
            write_http(root / 'STAGING_HTTP', fixed, release)
            with patch.object(natural, 'staging_plan', return_value=(fixed, catalogs)):
                path = root / 'STAGING_HTTP/train-clean-100-GET.json'
                raw = path.read_bytes()
                for key, value in (('attempt', 2), ('url', 'https://example.invalid/alternate'), ('ranges', 1)):
                    changed = json.loads(raw); changed[key] = value
                    path.write_bytes(multicorpus.encoded(changed))
                    with self.assertRaises(ValueError): natural.read_staging(root)
                path.write_bytes(raw)
                self.assertEqual(len(natural.read_staging(root)), 24)

    def test_24_whole_streaminfos_measured_without_native_decode_and_no_duration_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            items, _, _ = invented_native(root)
            with patch.object(natural, 'read_staging', return_value=items):
                value = capacity.natural_envelope(root, 'primary')
                self.assertEqual(value['original_count'], 24)
                self.assertEqual(value['members'][items[0]['path'].relative_to(root).as_posix()]['wav_bytes'], 480046)
                self.assertTrue(value['duration_rejects_retained'])
                raw = items[0]['path'].read_bytes(); items[0]['path'].write_bytes(raw[:26] + bytes(16) + raw[42:])
                with self.assertRaises(ValueError): capacity.natural_envelope(root, 'primary')

    def test_oversize_complete_original_stops_before_native_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            items, _, design = invented_native(root)
            for name in ('AUTHORIZATION', 'HISTORY', 'PROOF_BINDING', 'STAGING_CONSUMPTION', 'STAGING_CONSUMPTION_UPLOAD'):
                multicorpus.write(root / (name + '.json'), {'invented': True})
            with patch.object(natural, 'read_staging', return_value=items), patch.object(profile, 'require_proof', return_value={}):
                value = capacity.before_consumption(root, root / 'NATURAL_STAGING', design, 'primary')
                self.assertLess(value['maximum_zip_bytes'], capacity.SOURCE_LIMIT)
                raw = items[0]['path'].read_bytes()
                count = 80_000_000
                packed = (16000 << 44) | (15 << 36) | count
                items[0]['path'].write_bytes(raw[:18] + packed.to_bytes(8, 'big') + raw[26:])
                with self.assertRaises(ValueError):
                    capacity.before_consumption(root, root / 'NATURAL_STAGING', design, 'primary')
                self.assertFalse((root / 'CONSUMPTION.json').exists())

    def test_each_whole_original_is_decoded_once_and_duration_rejects_remain_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); items, pcms, design = invented_native(root)
            calls = []
            class Pcm:
                def __init__(self, raw): self.raw = raw
                def astype(self, dtype):
                    if dtype != '<i2': raise ValueError('wrong dtype')
                    return self
                def tobytes(self): return self.raw
            class SoundFile:
                def __init__(self, stream):
                    self.raw = stream.read(); calls.append(self.raw)
                    self.samplerate, self.channels, self.subtype = 16000, 1, 'PCM_16'
                def __len__(self): return len(pcms[self.raw]) // 2
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self, dtype):
                    if dtype != 'int16': raise ValueError('wrong read')
                    return Pcm(pcms[self.raw])
            with patch.object(natural, 'plan', return_value=design), patch.object(natural, 'read_staging', return_value=items), \
                 patch.object(natural.lineage, 'validate_history', return_value=(set(), set(), set())), \
                 patch.object(natural.lineage, 'require_saved_consumption'), patch.dict(sys.modules, {'soundfile': SimpleNamespace(SoundFile=SoundFile)}):
                meta = natural.extract(root, root / 'NATURAL_STAGING', {})
                self.assertEqual(len(calls), 24)
                self.assertEqual([r['samples'] for r in meta['source_rejections']], [240001, 7999])
                self.assertEqual(len(meta['rows']), 22)
                rows, originals = natural.read_source(root, meta, {})
                self.assertEqual((len(rows), len(originals), len(calls)), (22, 24, 24))
                self.assertFalse((root / 'flac').exists())
                self.assertEqual(len(list((root / 'NATURAL_STAGING').rglob('*.flac'))), 24)
                self.assertEqual((root / 'wav' / (items[0]['id'] + '.wav')).stat().st_size, 480046)
                tampered = copy.deepcopy(meta); tampered['source_rejections'][0]['samples'] = 240000
                with self.assertRaises(ValueError): natural.read_source(root, tampered, {})
                with self.assertRaises((ValueError, FileExistsError)): natural.extract(root, root / 'NATURAL_STAGING', {})
                self.assertEqual(len(calls), 24)

    def test_native_decode_failure_keeps_start_without_retry_or_later_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); items, _, design = invented_native(root)
            with patch.object(natural, 'plan', return_value=design), patch.object(natural, 'read_staging', return_value=items), \
                 patch.object(natural.lineage, 'validate_history', return_value=(set(), set(), set())), \
                 patch.object(natural.lineage, 'require_saved_consumption'), \
                 patch.dict(sys.modules, {'soundfile': SimpleNamespace(SoundFile=lambda _stream: (_ for _ in ()).throw(RuntimeError('fixture failure')))}):
                with self.assertRaises(RuntimeError): natural.extract(root, root / 'NATURAL_STAGING', {})
                self.assertEqual([p.name for p in (root / 'natural-decodes').iterdir()], [items[0]['id'] + '-STARTED.json'])
                self.assertEqual(list((root / 'wav').iterdir()), [])
                with self.assertRaises((ValueError, FileExistsError)): natural.extract(root, root / 'NATURAL_STAGING', {})

    def test_original_proof_transport_reuses_only_complete_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); original = root / 'original'; original.mkdir()
            (original / 'run.json').write_bytes(b'{"fixture":true}\n')
            (original / 'original.zip').write_bytes(b'PK\x03\x04INVENTED')
            files = profile.inventory(original)
            transport_map = {'run.json': dict(original_api_base64=base64.b64encode((original / 'run.json').read_bytes()).decode()),
                'original.zip': dict(artifact_id=123)}
            with patch.object(original_proof.subprocess, 'run') as network:
                original_proof.fetch_files(root / 'copy', files, transport_map, original)
                network.assert_not_called()
            self.assertEqual(profile.inventory(root / 'copy'), files)
            (original / 'run.json').write_bytes(b'{}')
            with self.assertRaises(ValueError): original_proof.fetch_files(root / 'changed', files, transport_map, original)
            self.assertFalse((root / 'changed').exists())

    def test_workflow_staging_consumes_before_get_and_native_consumes_after_capacity(self):
        for panel in ('primary', 'replication'):
            prefix = '' if panel == 'primary' else 'replication_'
            template = HERE / ('english_v6_b28_' + prefix + 'source_workflow_template.yml')
            raw = template.read_text(); token = '__B28_REVIEWED_' + panel.upper() + '_SOURCE_PARENT__'
            self.assertEqual(raw.count(token), 2)
            runtime = (HERE.parent / '.github/workflows' / ('english-v6-b28-' + panel + '-sources-once.yml')).read_text()
            if token in runtime: self.assertEqual(runtime, raw)
            else:
                parent = re.search(r"PREPARED_PARENT: '([0-9a-f]{40})'", runtime).group(1)
                self.assertEqual(runtime, raw.replace(token, parent))
            self.assertLess(raw.index(' stage-consume '), raw.index('whole-original-staging-consumed\n'))
            self.assertLess(raw.index('whole-original-staging-consumed\n'), raw.index(' transport '))
            self.assertLess(raw.index(' transport '), raw.index(' consume '))
            self.assertLess(raw.index(' consume '), raw.index(' generate '))
            self.assertIn('timeout-minutes: 245', raw)
            self.assertIn('timeout-minutes: 360', raw)
            self.assertIn('github.run_attempt == 1', raw)
            self.assertIn('cancel-in-progress: false', raw)
            self.assertNotIn('workflow_dispatch', raw)
            self.assertNotIn('magicdatatech', raw)
            self.assertNotIn('train-clean-500', raw)
            self.assertNotIn('test_english_v6_b28_quality_execution', raw)

    def test_imports_are_stdlib_only_and_source_never_calls_selector(self):
        code = 'import sys;sys.path.insert(0,' + repr(str(HERE)) + ');import english_v6_b28_source,english_v6_b28_replication_source;'
        code += 'blocked={"torch","numpy","scipy","soundfile","piper","onnxruntime"};'
        code += 'raise SystemExit(bool(blocked & set(sys.modules)))'
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('english_v6_b28_source', 'english_v6_b28_source_ancestry', 'english_v6_b28_natural_sources', 'english_v6_b28_source_transport'):
            raw = (HERE / (name + '.py')).read_text()
            self.assertNotIn('select_natural_rows(', raw)
            self.assertNotIn('freeze(', raw)


if __name__ == '__main__':
    unittest.main()
