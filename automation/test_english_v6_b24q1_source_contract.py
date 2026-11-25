"""Invented complete ledgers and source contracts, with no scientific execution."""
from __future__ import annotations
import ast
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24q1_source as source
import english_v6_b24q1_source_observe as observer
import english_v6_b24q1_quarantine as q
import english_v6_b18_asr_adapter as raw
import english_v6_b18_source_observe as native
import english_hifi_phones_once as phone
from test_english_v6_b24q1_quarantine import original_row, native_text_files, text_model, write_json


def phone_files(root, source_dir, row, provider):
    """Explicit invented finite blank tensors; empty CTC tokens remain valid."""
    index=source.qa.policy.PHONE_MODELS.index(provider)
    name, revision=source.qa.pcm_tools.PHONE_MODELS[index]
    revision=revision+'0'*(40-len(revision))
    model=dict(model=name, revision=revision, weights_sha256=phone.WEIGHT_SHA[index], target_conditioned=False,
        conv_kernel=[10,3,3,3,3,2,2], conv_stride=[5,2,2,2,2,2,2], runtime_versions={'invented_fixture':'stdlib'})
    vocab={('blank' if index==0 else 'invented-'+str(index)):index for index in range(392)}
    if not (root/'MODEL.json').exists():
        raw.record(root/'MODEL.json',model)
        raw.record(root/'VOCAB.json',dict(model=name,revision=revision,vocab=vocab,pad_token_id=0))
        (root/'logits').mkdir()
    start=native.binding(provider,row,raw.sha((source_dir/'MANIFEST.json').read_bytes()))
    directory=root/'rows'/row['id'];directory.mkdir(parents=True)
    raw.record(directory/'STARTED.json',start)
    raw.record(directory/'FORWARD_STARTED.json',dict(start,attempt=1))
    raw.record(directory/'FORWARD_RETURNED.json',dict(start,native_return_observed=True))
    frames=row['samples']
    for kernel,stride in zip(model['conv_kernel'],model['conv_stride'],strict=True):frames=(frames-kernel)//stride+1
    matrix=struct.pack('<392f',1.,*([0.]*391))*frames
    saved=raw.put(root/'logits'/(row['id']+'.f32'),matrix);saved['path']='logits/'+saved['path']
    payload=raw.record(directory/'RAW_PHONE.json',dict(start,schema=raw.RAW_SCHEMA,logits=saved,
        shape=[frames,392],dtype='float32-little-endian',order='C'))
    valid=raw.record(directory/'PAYLOAD_VALID.json',dict(start,schema=raw.RAW_SCHEMA,payload_valid=True,
        raw_phone=payload,logit_frames=frames,vocab_size=392,logits_sha256=saved['sha256']))
    capture=dict(start,raw_phone_tokens=[],logit_frames=frames,vocab_size=392,logits_sha256=saved['sha256'],
        raw_evidence={'payload':payload,'validation':valid},status='success',input_coverage_complete=True)
    raw.record(directory/'CAPTURE.json',capture)
    return capture


def refresh(root, rows, provider, registration):
    """Update invented receipt pins to test semantics beyond a stale-checksum failure."""
    source_sha=registration['source_manifest_sha256']
    ledger=q.ledger_document(root,rows,provider,source_sha)
    write_json(root/'LEDGER.json',ledger)
    captures={}
    for row in rows:
        directory=root/'rows'/row['id']
        for name in ('CAPTURE.json','INVALID_EMPTY.json','FAILED.json'):
            if (directory/name).exists(): captures[row['id']]=json.loads((directory/name).read_bytes());break
    run={'run':registration['run_id'],'sha':registration['run_head']}
    write_json(root/'RESULT.json',observer.report_document(provider,observer.BATCH,run,source_sha,rows,captures,ledger['rows']))
    write_json(root/'FILES.json',{'schema':'english-v6-b18-observer-file-closure-v1','self_excluded':'FILES.json',
                                'files':observer.file_closure(root)})


class SourceAuditScope(unittest.TestCase):
    def test_one_scope_covers_entire_cli_dispatch_for_each_panel(self):
        for filename in ('english_v6_b24q1_source.py','english_v6_b24q1_replication_source.py'):
            with self.subTest(filename=filename):
                tree=ast.parse((HERE/filename).read_text())
                main=tree.body[-1]
                self.assertIsInstance(main,ast.If)
                scope=main.body[-1]
                self.assertIsInstance(scope,ast.With)
                self.assertEqual(ast.unparse(scope.items[0].context_expr),'scope()')
                self.assertEqual(len(scope.body),1)
                self.assertIsInstance(scope.body[0],ast.If)
                branch=scope.body[0]
                modes=[]
                while isinstance(branch,ast.If):
                    modes.append(branch.test.comparators[0].value)
                    branch=branch.orelse[0]
                self.assertEqual(modes,['check','guard','authorize','verify-corpus','transport',
                    'consume','verify-source','generate','consume-observer','observe'])
                self.assertIn('screen(',ast.unparse(branch))
                read=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_read')
                self.assertFalse(any(isinstance(n,ast.With) and any(
                    isinstance(item.context_expr,ast.Call) and isinstance(item.context_expr.func,ast.Name)
                    and item.context_expr.func.id=='scope' for item in n.items) for n in ast.walk(read)))


class CompletePanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=source.lineage.read_plan()
        cls.contract=observer.source_check()
        cls.synthetic_plan=source.source_plan()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.source=self.base/'source';self.source.mkdir()
        (self.source/'MANIFEST.json').write_bytes(b'{"invented_contract_fixture":true}\n')
        self.rows=[];self.all_rows=[]
        for index,planned in enumerate(self.synthetic_plan['rows']):
            row=dict(planned,**{k:v for k,v in original_row(self.source,planned['id'],samples=8000,value=100+index).items() if k!='id'})
            self.rows.append(row);self.all_rows.append(row)
        for index,planned in enumerate(self.plan['primary']['natural']['roster']):
            sid='b24q1-primary-natural-'+planned['source_id']
            row=dict(planned,**original_row(self.source,sid,samples=8000 if index<4 else 4000,value=200+index),
                cohort='natural',role='development',requested_class=0,observed_class=None,training_allowed=False,
                source_group_id='librispeech-train-clean-100-'+planned['speaker'],whole_original=True,
                requested_text=planned['transcript'])
            self.all_rows.append(row)
            if index<4:self.rows.append(row)
        self.execution=dict(run_id='12345',run_head='b'*40,prepared_parent='a'*40,run_attempt=1,
            run_branch=source.lineage.BRANCH)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(source.lineage,'read_plan',return_value=self.plan))
        self.stack.enter_context(mock.patch.object(observer,'source_check',return_value=self.contract))
        self.stack.enter_context(mock.patch.object(source,'_read',return_value=(self.rows,self.all_rows,{},
            {'source_execution':self.execution})))
        self.stack.enter_context(mock.patch.object(source,'source_binding',return_value={'training_allowed':False}))
        self.registration=source.registration(self.source)
        def budget(actual,provider,registration):
            q.need(q.equal(actual,{'invented_only':True,'provider':provider,'registration':registration}),
                   'invented whole-registration budget proof')
        self.stack.enter_context(mock.patch.object(source.lineage,'validate_observer_budget',side_effect=budget))
        self.providers=self.base/'providers';self.providers.mkdir()

    def panel(self, empties=None):
        empties=empties if empties is not None else {'qwen':{self.rows[0]['id']},'parakeet':{self.rows[1]['id']}}
        for provider in observer.PROVIDERS:
            root=self.providers/provider;root.mkdir()
            raw.record(root/'OBSERVER_BUDGET.json',dict(invented_only=True,provider=provider,registration=self.registration))
            raw.record(root/'STARTED.json',observer.started_document(provider,observer.BATCH,
                {'run':self.execution['run_id'],'sha':self.execution['run_head']},
                self.registration['source_manifest_sha256'],self.registration,len(self.rows)))
            raw.record(root/'MODEL_LOAD_STARTED.json',{'provider':provider,'actual_forward_attempts':0})
            for row in self.rows:
                if provider in raw.TEXT_PROVIDERS:
                    empty=row['id'] in empties.get(provider,set())
                    directory,start=native_text_files(root,self.source,row,provider,
                        '\n' if empty else row['requested_text'],self.contract,valid=not empty)
                    if empty:raw.record(directory/'INVALID_EMPTY.json',q.empty_receipt(self.source,root,row,start,self.contract))
                else:phone_files(root,self.source,row,provider)
            refresh(root,self.rows,provider,self.registration)

    def load(self):
        return observer.load_observations(self.source,self.providers,self.rows,observer.BATCH,registration=self.registration)

    def mutate_json(self,relative,change,provider):
        path=self.providers/provider/relative;value=json.loads(path.read_bytes());change(value);write_json(path,value)
        return path

    def test_complete_5_times_n_includes_two_distinct_invalid_empties(self):
        self.panel();value=self.load()
        self.assertEqual(value['completed_coverage'],dict.fromkeys(observer.PROVIDERS,100))
        self.assertEqual(value['coverage'],dict(qwen=99,whisper=100,parakeet=99,xlsr=100,lv60=100))
        self.assertEqual(value['invalid_empty_coverage'],dict(qwen=1,whisper=0,parakeet=1,xlsr=0,lv60=0))
        self.assertEqual(value['quarantined_ids'],sorted([self.rows[0]['id'],self.rows[1]['id']]))
        self.assertEqual(value['native_slot_accounting']['total'],
            dict(planned=500,attempted=500,returned=500,valid=498,invalid_empty=2,fatal=0,unknown=0))
        self.assertEqual(value['phones'][self.rows[0]['id']]['xlsr']['logits'][:4],struct.pack('<f',1.))
        self.assertEqual(json.loads((self.providers/'xlsr/rows'/self.rows[0]['id']/'CAPTURE.json').read_bytes())['raw_phone_tokens'],[])

    def test_same_row_empty_union_counts_once_but_preserves_each_provider_slot(self):
        sid=self.rows[0]['id'];self.panel({'qwen':{sid},'whisper':{sid},'parakeet':{sid}})
        value=self.load();self.assertEqual(value['quarantined_ids'],[sid])
        self.assertEqual(set(value['quarantine'][sid]),set(raw.TEXT_PROVIDERS))
        self.assertEqual(value['native_slot_accounting']['total']['invalid_empty'],3)
        self.assertEqual(value['native_slot_accounting']['total']['returned'],500)

    def test_fatal_other_provider_on_quarantined_row_still_stops(self):
        self.panel();sid=self.rows[0]['id'];root=self.providers/'lv60';directory=root/'rows'/sid
        (directory/'CAPTURE.json').unlink()
        raw.record(directory/'FAILED.json',dict(native.binding('lv60',self.rows[0],self.registration['source_manifest_sha256']),
            status='failed',error='invented fatal',retry_count=0,whole_panel_stop=True))
        refresh(root,self.rows,'lv60',self.registration)
        with self.assertRaisesRegex(ValueError,'ALL_FIXED_NATIVE_ATTEMPTS'):self.load()

    def test_omitted_last_slot_after_early_empty_does_not_complete(self):
        self.panel();root=self.providers/'qwen';shutil.rmtree(root/'rows'/self.rows[-1]['id'])
        refresh(root,self.rows,'qwen',self.registration)
        with self.assertRaisesRegex(ValueError,'ONLY_FIXED_PLANNED_NATIVE_SLOTS'):self.load()

    def test_duplicate_or_reordered_report_rows_cannot_hide_missing_slot(self):
        self.panel();root=self.providers/'whisper';path=root/'RESULT.json';saved=path.read_bytes()
        for change in (lambda v:v['rows'].__setitem__(-1,v['rows'][0]),lambda v:v['rows'].reverse()):
            with self.subTest(change=repr(change)):
                self.mutate_json('RESULT.json',change,'whisper')
                write_json(root/'FILES.json',{'schema':'english-v6-b18-observer-file-closure-v1','self_excluded':'FILES.json','files':observer.file_closure(root)})
                with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_REPORT'):self.load()
                path.write_bytes(saved)

    def test_extra_unplanned_slot_is_fatal_even_if_report_and_closure_updated(self):
        self.panel();root=self.providers/'whisper'
        (root/'rows'/'b24q1-unplanned').mkdir();(root/'rows'/'b24q1-unplanned'/'STARTED.json').write_bytes(b'{}')
        refresh(root,self.rows,'whisper',self.registration)
        with self.assertRaisesRegex(ValueError,'ONLY_FIXED_PLANNED_NATIVE_SLOTS'):self.load()

    def test_forged_summary_or_budget_cannot_relabel_empty_as_valid(self):
        self.panel();root=self.providers/'qwen'
        self.mutate_json('RESULT.json',lambda v:v.update(valid_native_payloads=100,invalid_empty_native_payloads=0),'qwen')
        write_json(root/'FILES.json',{'schema':'english-v6-b18-observer-file-closure-v1','self_excluded':'FILES.json','files':observer.file_closure(root)})
        with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_REPORT'):self.load()
        refresh(root,self.rows,'qwen',self.registration)
        self.mutate_json('OBSERVER_BUDGET.json',lambda v:v['registration'].update(max_calls=99),'qwen')
        with self.assertRaisesRegex(ValueError,'invented whole-registration'):self.load()

    def test_phone_nonfinite_wrong_geometry_and_missing_matrix_stay_fatal(self):
        self.panel();root=self.providers/'xlsr';sid=self.rows[0]['id'];path=root/'logits'/(sid+'.f32');saved=path.read_bytes()
        # Even if raw file/receipt hashes are consistently rewritten, the original
        # finite tensor validator rejects NaN on an already quarantined original.
        path.write_bytes(struct.pack('<f',float('nan'))+saved[4:])
        directory=root/'rows'/sid;payload=json.loads((directory/'RAW_PHONE.json').read_bytes())
        payload['logits']=dict(raw.pin(path),path='logits/'+path.name);write_json(directory/'RAW_PHONE.json',payload)
        valid=json.loads((directory/'PAYLOAD_VALID.json').read_bytes());valid.update(raw_phone=raw.pin(directory/'RAW_PHONE.json'),logits_sha256=raw.sha(path.read_bytes()))
        write_json(directory/'PAYLOAD_VALID.json',valid)
        capture=json.loads((directory/'CAPTURE.json').read_bytes());capture.update(logits_sha256=valid['logits_sha256'],
            raw_evidence={'payload':raw.pin(directory/'RAW_PHONE.json'),'validation':raw.pin(directory/'PAYLOAD_VALID.json')})
        write_json(directory/'CAPTURE.json',capture);refresh(root,self.rows,'xlsr',self.registration)
        with self.assertRaisesRegex(ValueError,'NONFINITE_NATIVE_CTC_LOGIT'):self.load()
        path.unlink();refresh(root,self.rows,'xlsr',self.registration)
        with self.assertRaisesRegex(ValueError,'EVERY_FIXED_PHONE_MATRIX'):self.load()

    def test_screen_uses_every_remaining_five_valid_row_and_preserves_all120_strata(self):
        self.panel();seen=[]
        def admit(row,texts,phones):
            seen.append(row['id']);self.assertEqual(set(texts),set(raw.TEXT_PROVIDERS))
            self.assertEqual(set(phones),{'xlsr','lv60'})
            return {'class':row['requested_class'],'tier':'silver','reason':'invented','margins':{},'training_allowed':False}
        with mock.patch.object(source.qa,'admit',side_effect=admit):value=source.screen_document(self.source,self.providers)
        self.assertEqual(set(seen),{row['id'] for row in self.rows[:96]}-set(value['quarantined_ids']))
        self.assertEqual(value['source_rows'],100);self.assertEqual(value['selected_original_slots'],120)
        self.assertEqual(value['partition'],dict(pre_observer_rejected=20,quarantined_invalid_empty=2,
            five_valid_unqualified=0,qualified=98))
        self.assertEqual(value['source_gate'],'INSUFFICIENT_SOURCE')
        self.assertEqual(value['source_shortfall']['natural_originals'],{'observed':4,'min':12})
        for dimension in ('cohort','requested_class','cell_id','canonical_person'):
            self.assertEqual(sum(v['original_slots'] for v in value['attrition'][dimension].values()),120)
            self.assertEqual(sum(v['observer_eligible'] for v in value['attrition'][dimension].values()),100)
        for row in value['rows']:
            if row['id'] in value['quarantined_ids']:
                self.assertIsNone(row['observed_class']);self.assertEqual(row['tier'],'unknown')
                self.assertEqual(row['reason'],q.EMPTY_REASON);self.assertIs(row['source_admission_eligible'],False)
        self.assertEqual(value['source_policy'],source.source_policy())

    def test_full_panel_is_reopened_before_screen_calls_any_admission(self):
        self.panel();root=self.providers/'lv60'
        (root/'rows'/self.rows[-1]['id']/'FORWARD_RETURNED.json').unlink()
        write_json(root/'FILES.json',{'schema':'english-v6-b18-observer-file-closure-v1','self_excluded':'FILES.json',
            'files':observer.file_closure(root)})
        with mock.patch.object(source.qa,'admit',side_effect=AssertionError('must not admit before complete readback')) as admit:
            with self.assertRaises(ValueError):source.screen_document(self.source,self.providers)
            admit.assert_not_called()

    def test_pass_screen_replay_rejects_relabeled_quarantine_and_omitted_attrition(self):
        # A separate invented fixture meets every unchanged source minimum. This
        # tests the successful replay gate, not merely an always-insufficient panel.
        for index in range(100,112):
            row=self.all_rows[index]
            row.update(original_row(self.source,row['id'],8000,400+index))
            self.rows.append(row)
        self.registration=source.registration(self.source)
        self.panel()
        def admit(row,_texts,_phones):
            return {'class':row['requested_class'],'tier':'silver','reason':'invented','margins':{},'training_allowed':False}
        with mock.patch.object(source.qa,'admit',side_effect=admit):
            document=source.screen_document(self.source,self.providers)
            self.assertEqual(document['source_gate'],'SOURCE_GATE_PASS')
            self.assertEqual(source.verify_screen_contract(self.source,document,self.providers),document)
            for change in (lambda v:v['rows'][0].update(observed_class=2,tier='silver'),
                           lambda v:v.update(quarantined_ids=[]),lambda v:v['partition'].update(quarantined_invalid_empty=0)):
                altered=copy.deepcopy(document);change(altered)
                with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_RAW_Q1'):
                    source.verify_screen_contract(self.source,altered,self.providers)


class OrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=observer.source_check()

    def experiment(self, failure=None):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        base=Path(temporary.name);src=base/'source';src.mkdir();(src/'MANIFEST.json').write_bytes(b'{}\n')
        rows=[original_row(src,'b24q1-primary-invented-'+str(index),8000,100+index) for index in range(3)]
        output=base/'observer';calls=[]
        registration={'source_rows':3,'source_manifest_sha256':raw.sha((src/'MANIFEST.json').read_bytes())}
        def backend(wav,directory,start):
            index=len(calls);calls.append(start['id'])
            if failure=='exception' and index==0:raise RuntimeError('EMPTY_NATIVE_TRANSCRIPT')
            if failure=='forged_name' and index==0:raise ValueError('EMPTY_NATIVE_TRANSCRIPT')
            if failure=='later_fatal' and index==2:raise ValueError('other invalid return')
            return raw.capture_python('whisper',directory,start,lambda:{'text':'\n' if index==0 else 'invented ordinary'})
        with ExitStack() as stack:
            for target,name,value in ((observer.lineage,'read_plan',{}),(observer,'source_check',self.contract),
                (observer,'guard',{'run':'1','sha':'b'*40}),(observer.lineage,'require_observer_permit',{'invented':True}),
                (observer,'expected_registration',registration),(observer,'require_original_rows',None),
                (observer,'validate_rows',None)):
                stack.enter_context(mock.patch.object(target,name,return_value=value))
            stack.enter_context(mock.patch.object(raw,'_load',return_value=(backend,text_model('whisper',self.contract))))
            if failure:
                with self.assertRaisesRegex(ValueError,'WHOLE_PANEL_STOP_OTHER_NATIVE_FAILURE'):
                    observer.observe('whisper',src,output,read_rows=lambda _:rows,schema=observer.BATCH,registration=registration)
            else:observer.observe('whisper',src,output,read_rows=lambda _:rows,schema=observer.BATCH,registration=registration)
        return output,rows,calls

    def test_proved_empty_continues_every_later_fixed_slot_once(self):
        output,rows,calls=self.experiment()
        self.assertEqual(calls,[row['id'] for row in rows]);report=json.loads((output/'RESULT.json').read_bytes())
        self.assertEqual(report['status'],'COMPLETE_NATIVE_SLOTS');self.assertEqual(report['valid_native_payloads'],2)
        self.assertEqual(report['invalid_empty_native_payloads'],1);self.assertEqual(report['completed_native_slots'],3)
        directory=output/'rows'/rows[0]['id']
        self.assertTrue((directory/'INVALID_EMPTY.json').exists());self.assertFalse((directory/'CAPTURE.json').exists())
        self.assertFalse((directory/'PAYLOAD_VALID.json').exists());self.assertFalse((directory/'FAILED.json').exists())

    def test_exception_name_without_complete_raw_proof_never_authorizes_continue(self):
        for failure in ('exception','forged_name'):
            with self.subTest(failure=failure):
                output,rows,calls=self.experiment(failure)
                self.assertEqual(calls,[rows[0]['id']])
                self.assertTrue((output/'rows'/rows[0]['id']/'FAILED.json').exists())
                self.assertFalse((output/'rows'/rows[0]['id']/'INVALID_EMPTY.json').exists())

    def test_later_unrelated_failure_is_not_swallowed_after_an_empty(self):
        output,rows,calls=self.experiment('later_fatal')
        self.assertEqual(calls,[row['id'] for row in rows]);report=json.loads((output/'RESULT.json').read_bytes())
        self.assertEqual(report['status'],'STOP_OBSERVER_FAILURE');self.assertIs(report['whole_panel_stop'],True)
        self.assertEqual(report['invalid_empty_native_payloads'],1)
        self.assertTrue((output/'rows'/rows[2]['id']/'FAILED.json').exists())


class FrozenPrimitives(unittest.TestCase):
    def test_old_native_and_admission_function_objects_and_bytes_still_exact(self):
        self.assertIs(observer._load_phone,native._load_phone);self.assertIs(observer._verify_phone,native._verify_phone)
        self.assertIs(observer.raw_asr,raw)
        contract=observer.source_check()
        self.assertEqual(contract['source_admission_function_sha256'],native.POLICY_FUNCTIONS)
        self.assertEqual(contract['asr_adapter_sha256'],raw.sha(Path(raw.__file__).read_bytes()))
        self.assertIs(contract['orchestration_changed'],True)
        self.assertEqual(contract['orchestration_contract_sha256'],q.ORCHESTRATION_SHA)

    def test_old_numerical_synthetic_and_natural_branches_are_ast_identical(self):
        def branches(path):
            tree=ast.parse(path.read_text());screen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='screen_document')
            loop=next(n for n in screen.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='row')
            branch=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                and isinstance(n.test.left,ast.Subscript) and isinstance(n.test.left.slice,ast.Constant)
                and n.test.left.slice.value=='cohort')
            return ast.dump(branch,include_attributes=False)
        self.assertEqual(branches(HERE/'english_v6_b24q1_source.py'),branches(HERE/'english_v6_b22_source.py'))

    def test_natural_generation_and_rejection_only_rename_new_experiment(self):
        for name in ('natural_sources.py','replication_natural_sources.py'):
            old=(HERE/('english_v6_b24r1_'+name)).read_text().replace('B24R1','B24Q1').replace('b24r1','b24q1')
            self.assertEqual((HERE/('english_v6_b24q1_'+name)).read_text(),old)

    def test_source_registration_retains_n_and_all_new_policy_pins(self):
        contract=observer.source_check();policy=source.source_policy()
        self.assertEqual({key:policy[key] for key in observer.POLICY_KEYS},
                         {key:contract[key] for key in observer.POLICY_KEYS})
        self.assertEqual(policy['approval_sha256'],source.lineage.metadata.APPROVAL_SHA)
        self.assertEqual(source.FILES['primary'],'4735d1142c24d74ce1815f924be7b189bcb55643230be73631d899e8b88249ee')
        self.assertEqual(source.FILES['replication'],'50d8d6ba0828032f2d3585a24f86c8fdfa223fae073761873d8f09c960661e96')

    def test_replica_reuses_identical_new_policy_and_original_training_globals(self):
        import english_v6_b24q1_replication_source as replica
        import english_v6_b24q1_replication_source_observe as replica_observer
        import english_v6_b24_source_ancestry as training
        self.assertIs(replica.screen_document.__code__,source.screen_document.__code__)
        self.assertIs(replica_observer.observe.__code__,observer.observe.__code__)
        self.assertIs(replica_observer.load_observations.__code__,observer.load_observations.__code__)
        self.assertIs(replica_observer.quarantine,q)
        self.assertIs(replica_observer._load_phone,native._load_phone)
        self.assertEqual(replica.PANEL,replica_observer.PANEL)
        self.assertEqual(replica.PANEL,'replication')
        for name in ('training_binding','verify_training','validate_training_live','training_binding_sha'):
            self.assertIs(getattr(source.lineage,name),getattr(training,name))
            self.assertIs(getattr(replica.lineage,name),getattr(training,name))

    def test_phone_validity_is_full_matrix_even_for_empty_greedy_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);src=base/'source';src.mkdir();(src/'MANIFEST.json').write_bytes(b'{}\n')
            row=original_row(src,samples=8000);root=base/'xlsr';root.mkdir()
            phone_files(root,src,row,'xlsr')
            start=native.binding('xlsr',row,raw.sha((src/'MANIFEST.json').read_bytes()))
            capture,value=observer.read_valid(src,root,row,start,{})
            self.assertEqual(capture['raw_phone_tokens'],[]);self.assertGreater(len(value['logits']),0)
            path=root/'VOCAB.json';vocab=json.loads(path.read_bytes());saved=path.read_bytes()
            vocab['vocab'].pop(next(iter(vocab['vocab'])));write_json(path,vocab)
            with self.assertRaisesRegex(ValueError,'EXACT_NATIVE_PHONE_VOCAB'):
                observer.read_valid(src,root,row,start,{})
            path.write_bytes(saved)
            path=root/'rows'/row['id']/'CAPTURE.json';capture['logit_frames']-=1;write_json(path,capture)
            with self.assertRaisesRegex(ValueError,'COMPLETE_ORIGINAL_PHONE_ARRAY_RETENTION'):
                observer.read_valid(src,root,row,start,{})


if __name__=='__main__':unittest.main()
