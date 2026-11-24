"""B21 source contract regressions, without audio engines, network or KWS."""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_source as source
import english_v6_b21_source_ancestry as lineage
import english_v6_b21_natural_sources as natural
import english_v6_b21_source_observe as observers


class SourceContract(unittest.TestCase):
    def test_frozen_metadata_is_disjoint_and_bounded(self):
        source.check()
        design, history = lineage.read_plan(), lineage.read_ancestry()
        a = design['primary']; b = design['replication']
        self.assertEqual((len(history['all_pcm_sha256']), len(history['all_source_groups']), len(history['canonical_people'])), (2343, 380, 300))
        self.assertFalse(set(a['natural']['speakers']) & set(b['natural']['speakers']))
        self.assertFalse(set(a['synthetic']['speaker_id_map'].values()) & set(b['synthetic']['speaker_id_map'].values()))
        self.assertEqual(design['maximum_native_observer_calls'], 600)
        self.assertFalse(design['replication_opened'])

    def test_history_omission_of_unknown_original_is_rejected(self):
        history = lineage.read_ancestry()
        history['all_pcm_sha256'].pop()
        with self.assertRaises(ValueError): lineage.validate_history(history)

    def test_event_guard_rejects_replay_and_private_before_fetch(self):
        prepared, head = '1'*40, '2'*40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=lineage.REPO,
                   GITHUB_REF=lineage.BRANCH, GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1',
                   GITHUB_RUN_ID='123', GITHUB_SHA=head)
        event = dict(before=prepared, after=head, ref=lineage.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=lineage.REPO, private=False))
        self.assertEqual(lineage.execution_guard(prepared, env, event, prepared)['run_id'], '123')
        for key, bad in [('GITHUB_RUN_ATTEMPT','2'), ('GITHUB_REPOSITORY','someone/private')]:
            changed = dict(env); changed[key] = bad
            with self.assertRaises(ValueError): lineage.execution_guard(prepared, changed, event, prepared)
        changed = copy.deepcopy(event); changed['repository']['private'] = True
        with self.assertRaises(ValueError): lineage.execution_guard(prepared, env, changed, prepared)
        with self.assertRaises(ValueError): lineage.execution_guard(prepared, env, event, '3'*40)

    def test_source_checkpoint_hash_is_checked_before_unzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'B20_SOURCE.zip').write_bytes(b'not a genuine checkpoint')
            with patch('zipfile.ZipFile', side_effect=AssertionError('must not unzip bad source')):
                with self.assertRaises(ValueError): lineage.authenticate_previous(p)

    def make_catalog(self):
        design=natural.plan()
        transcripts=[]
        for sp in design['speakers']:
            text=''.join(f'{sp}-1-{i:04d} PLEASE BRING THE PACKAGE INSIDE\n' for i in range(5))
            transcripts.append(dict(path=f'LibriSpeech/train-clean-100/{sp}/1/{sp}-1.trans.txt',
                                    text=text, sha256=lineage.digest(text.encode())))
        return dict(schema='english-v6-b21-primary-natural-original-metadata-v1',
                    speakers_text=(lineage.PLANS/'SPEAKERS.TXT').read_text(),
                    transcripts=sorted(transcripts,key=lambda r:r['path']))

    def test_natural_roster_rank_is_independent_and_no_replacement(self):
        catalog=self.make_catalog(); design=natural.plan()
        rows=natural.rank_metadata(catalog,design)
        self.assertEqual(len(rows),24)
        self.assertEqual([r['speaker'] for r in rows], [sp for sp in design['speakers'] for _ in range(2)])
        changed=copy.deepcopy(catalog); changed['transcripts'].pop()
        with self.assertRaises(ValueError): natural.rank_metadata(changed,design)
        changed=copy.deepcopy(catalog); changed['transcripts'][0]['text'] += 'injected'
        with self.assertRaises(ValueError): natural.rank_metadata(changed,design)
        changed=copy.deepcopy(catalog); changed['transcripts'].append(changed['transcripts'][0])
        with self.assertRaises(ValueError): natural.rank_metadata(changed,design)

    def test_flac_full_length_and_checksum_required(self):
        info=bytearray(34)
        packed=(16000<<44)|(0<<41)|(15<<36)|12345
        info[10:18]=packed.to_bytes(8,'big')
        info[18:34]=bytes.fromhex('0123456789abcdef0123456789abcdef')
        raw=b'fLaC'+bytes([0x80])+int(34).to_bytes(3,'big')+info
        actual=natural.flac_header(raw)
        self.assertEqual(actual['samples'],12345)
        self.assertEqual(actual['bits_per_sample'],16)
        with self.assertRaises(ValueError): natural.flac_header(raw[:41])
        with self.assertRaises(ValueError): natural.flac_header(raw[:-16]+bytes(16))

    def observer_rows(self):
        rows=copy.deepcopy(source.source_plan()['rows'])
        for i,row in enumerate(rows): row['pcm_sha256']=hashlib.sha256(str(i).encode()).hexdigest()
        return rows

    def test_observer_denominator_cannot_drop_replace_or_relabel_source(self):
        rows=self.observer_rows(); reg=dict(source_rows=96, source_manifest_sha256='f'*64)
        observers.validate_rows(rows,'f'*64,reg)
        with self.assertRaises(ValueError): observers.validate_rows(rows[:-1],'f'*64,reg)
        changed=copy.deepcopy(rows); changed[-1]=changed[0]
        with self.assertRaises(ValueError): observers.validate_rows(changed,'f'*64,reg)
        changed=copy.deepcopy(rows); changed[0]['role']='train'
        with self.assertRaises(ValueError): observers.validate_rows(changed,'f'*64,reg)
        changed=copy.deepcopy(rows); changed[0]['observed_class']=2
        with self.assertRaises(ValueError): observers.validate_rows(changed,'f'*64,reg)

    def test_mixed_observer_budget_cannot_add_unregistered_extra_rows(self):
        rows=self.observer_rows()
        for i in range(24):
            rows.append(dict(id=f'b21-primary-natural-f{i}', pcm_sha256=hashlib.sha256(('n'+str(i)).encode()).hexdigest(),
                cohort='natural', role='development', training_allowed=False, observed_class=None,
                requested_class=0, whole_original=True))
        reg=dict(source_rows=120,source_manifest_sha256='f'*64)
        observers.validate_rows(rows,'f'*64,reg)
        rows.append(dict(rows[-1],id='b21-primary-natural-extra',pcm_sha256='1'*64))
        with self.assertRaises(ValueError): observers.validate_rows(rows,'f'*64,dict(reg,source_rows=121))

    def test_real_observe_completion_uses_all120_retained_originals(self):
        # No native model or audio engine is invoked. A deterministic fake
        # backend writes the same attempt/return/payload marker files; the real
        # observer ledger and completion control flow must accept all120.
        rows=self.observer_rows()
        for i in range(24):
            rows.append(dict(id=f'b21-primary-natural-fixture{i}', pcm_sha256=hashlib.sha256(('n'+str(i)).encode()).hexdigest(),
                cohort='natural', role='development', training_allowed=False, observed_class=None,
                requested_class=0, whole_original=True))
        for row in rows: row.update(wav_sha256=row['pcm_sha256'],samples=16000)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'source';src.mkdir()
            (src/'MANIFEST.json').write_text('{}')
            source_sha=source.qa.sha(src/'MANIFEST.json')
            registration=dict(source_rows=120,source_manifest_sha256=source_sha)
            by_id={r['id']:r for r in rows}
            called=[]
            def fake_backend(wav,directory,start):
                called.append(wav.stem)
                for name in ('FORWARD_STARTED.json','FORWARD_RETURNED.json','RAW_PHONE.json','PAYLOAD_VALID.json'):
                    source.qa.record(directory/name,{'fixture':True})
                return {'fixture':True}
            import english_hifi_micro as micro
            with ExitStack() as stack:
                stack.enter_context(patch.object(observers,'guard',return_value={'run':'123','sha':'1'*40}))
                stack.enter_context(patch.object(observers,'expected_registration',return_value=registration))
                stack.enter_context(patch.object(observers,'require_original_rows',return_value=None))
                stack.enter_context(patch.object(observers,'_load_phone',return_value=(fake_backend,{'fixture':True})))
                stack.enter_context(patch.object(micro,'inspect_wav',side_effect=lambda p:by_id[p.stem]))
                report=observers.observe('xlsr',src,root/'observer',read_rows=lambda p:rows,
                    schema=observers.BATCH,registration=registration)
            self.assertEqual(len(called),120)
            self.assertEqual(report['status'],'COMPLETE')
            self.assertEqual(report['valid_native_payloads'],120)
            ledger=observers.row_ledger(root/'observer',rows)
            self.assertTrue(observers.complete_ledger(ledger,120))
            self.assertFalse(observers.complete_ledger(ledger[:-1],120))
            damaged=copy.deepcopy(ledger);damaged[-1]['payload_valid']=False;damaged[-1]['capture_saved']=False
            self.assertFalse(observers.complete_ledger(damaged,120))

    def test_generate_preserves_immutable_renderer_receipt(self):
        # Actual qa.record refuses overwrites. This exercises the generation
        # wiring with no source engine, corpus decode or model calls.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archives=root/'archives';archives.mkdir()
            history=lineage.read_ancestry()
            source.qa.record(archives/'HISTORY.json',history)
            source.qa.record(archives/'AUTHORIZATION.json',{'fixture':True})
            (archives/'B20_SOURCE.zip').write_bytes(b'fixture-unused-proof')
            identity=dict(run_id='123',run_head='1'*40,run_attempt=1,run_branch=lineage.BRANCH,prepared_parent='2'*40)
            synthetic={'rows':[],'immutable_fixture':True}
            def render(out,*args):
                out.mkdir();source.qa.record(out/'MANIFEST.json',synthetic)
            with ExitStack() as stack:
                stack.enter_context(patch.object(lineage,'execution_guard',return_value=identity))
                stack.enter_context(patch.object(lineage,'require_authorization',return_value={'source_execution':identity}))
                stack.enter_context(patch.object(natural,'verify_corpus',return_value='fixture'))
                stack.enter_context(patch.object(source.B13,'render_panel',side_effect=render))
                stack.enter_context(patch.object(natural,'extract',return_value={'rows':[]}))
                read=stack.enter_context(patch.object(source,'read_source',return_value=[]))
                source.generate(root/'out',archives,root/'unused-corpus','2'*40)
            original=(root/'out/SYNTHETIC_MANIFEST.json').read_bytes()
            combined=json.loads((root/'out/MANIFEST.json').read_bytes())
            self.assertEqual(json.loads(original),synthetic)
            self.assertEqual(combined['synthetic'],synthetic)
            self.assertEqual(combined['synthetic_manifest_sha256'],lineage.digest(original))
            self.assertEqual(read.call_count,1)

    def test_native_policy_and_strict_return_contract_remain_frozen(self):
        contract=source.code_contract()
        self.assertTrue(contract['strict_valid_five_per_source'])
        self.assertTrue(contract['no_retry'])
        self.assertFalse(contract['diagnostic_empty_exception'])
        self.assertEqual(contract['maximum_native_calls'],600)
        self.assertEqual(contract['frozen_native_observer_sha256'],observers.FROZEN_NATIVE_SHA)


if __name__ == '__main__':
    unittest.main()
