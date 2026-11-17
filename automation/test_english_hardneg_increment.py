"""Stdlib/source/fictional-fixture contracts; never execute numeric code or a process."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

HERE=Path(__file__).resolve().parent
WORKER=HERE/'incremental_head_once.py'
if not WORKER.exists():WORKER=HERE/'english_hardneg_increment_once.py'
spec=importlib.util.spec_from_file_location('increment_under_test',WORKER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture_sources():
    roster={'rows':[]};candidates=[];ledger={'candidates':[]}
    starts=[5760,5760,103040,60800,44160,8320,33920,13440,26240]
    ends=[62720,62720,160000,116480,99840,62720,76080,69120,80640]
    ids=['source1','source2','source3','source4','source5','source6','source7','source3','source8']
    for rid in sorted(set(ids)):
        roster['rows'].append({'recording_id':rid,'frames':200000,'source_role':'train','old_bg48_role':'train',
                              'current_coverage_role':'train','pcm_sha256':'0'*64})
    for index,(start,end,rid) in enumerate(zip(starts,ends,ids)):
        row={'candidate_id':'c'+str(index),'recording_id':rid,'start_sample':start,'end_sample_exclusive':end,
             'window_pcm_sha256':'1'*64,'source_pcm_sha256':'0'*64,
             'eligible_ready_samples':[t for t in range(35200,200001,1280) if t-35200>=start and t<=end]}
        candidates.append(row);ledger['candidates'].append(copy.deepcopy(row))
    return candidates,roster,ledger


def old_draws():
    return [[{'condition':cond,'id':cond+str(label),'weak_class':label,'role':'train'}
             for cond in ('clean',('snr10','new5','new15')[step%3]) for label in range(3)] for step in range(96)]


def paired_rows():
    return [{'condition':'clean','id':'case'+str(i),'role':'development','weak_class':1,
             'observed_samples':64000,'clip_start_sample':35200,'original_clip_eof_sample':60000,
             'matched_intent_after_clip_start':True,'research_error':False,'prefix_events':[],
             'wrong_keyword_after_clip_start':[],'extra_same_keyword_events':0,'weak_negative_intent_events':0,
             'events':[{'sample':62080,'keyword':1}]} for i in range(291)]


class IncrementTests(unittest.TestCase):
    def test_import_has_no_numeric_library(self):
        import ast
        tree=ast.parse(WORKER.read_text())
        imported=[]
        for n in tree.body:
            if isinstance(n,ast.Import):imported.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):imported.append(n.module)
        self.assertFalse(set(imported)&{'numpy','torch','onnx','onnxruntime'})
        self.assertNotIn('numpy',__import__('sys').modules)
        self.assertNotIn('torch',__import__('sys').modules)

    def test_recipe_and_limits_fixed(self):
        p=m.expected_plan('a'*64,{})
        self.assertEqual((p['new_optimizer_updates'],p['training_bag_forwards'],p['candidate_head_cases']),(96,672,323))
        self.assertEqual((p['candidate_embedding_rows'],p['candidate_ready_rows']),(18982,14137))
        self.assertEqual(p['native_head_processes'],1)
        self.assertEqual(p['maximum_execution_seconds'],1200)
        self.assertEqual(p['new_frontend_calls'],0)
        self.assertEqual(p['new_audio_decodes'],0)
        self.assertEqual(p['new_compilations'],0)
        self.assertFalse(p['checkpoint_selection'])
        self.assertEqual(m.CONFIG['threshold_float32'],0.800000011920929)
        self.assertEqual((m.CONFIG['confirm_embeddings'],m.CONFIG['cooldown_embeddings']),(3,13))
        self.assertEqual(m.RECIPE['optimizer_state'],'RESET_NO_OLD_MOMENTS')

    def test_complete_context_has_134_rows_not_291(self):
        c,r,l=fixture_sources();s=m.choose_reviewed_inputs(c,r,l)
        self.assertEqual([x['input_count'] for x in s['candidates']],[17,17,17,16,16,15,5,16,15])
        self.assertEqual(sum(x['input_count'] for x in s['sources']),134)
        self.assertEqual(next(x['input_count'] for x in s['sources'] if x['recording_id']=='source3'),33)
        self.assertFalse(s['unreviewed_original_portions_admitted'])

    def test_support_boundary_is_half_open_pcm_and_inclusive_clock(self):
        c,r,l=fixture_sources();s=m.choose_reviewed_inputs(c,r,l)
        for row in s['candidates']:
            for end in row['head_ends']:
                self.assertGreaterEqual(end-35200,row['reviewed_start']);self.assertLessEqual(end,row['reviewed_end_exclusive'])
        c[0]['eligible_ready_samples'].insert(0,c[0]['eligible_ready_samples'][0]-1280)
        with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_SUPPORT_MASK'):m.choose_reviewed_inputs(c,r,l)

    def test_any_window_or_source_label_expansion_is_rejected(self):
        for mutate,reason in [(lambda c,r,l:c[0].update(start_sample=0),'REVIEW_WINDOW_BINDING'),
                              (lambda c,r,l:r['rows'][0].update(source_role='development'),'UNCHANGED_TRAIN_SOURCE_ROLE'),
                              (lambda c,r,l:c[0].update(source_pcm_sha256='9'*64),'ORIGINAL_PCM_IDENTITY')]:
            c,r,l=fixture_sources();mutate(c,r,l)
            with self.assertRaisesRegex(ValueError,reason):m.choose_reviewed_inputs(c,r,l)

    def test_duplicate_candidate_or_missing_review_rejected(self):
        c,r,l=fixture_sources();c[-1]=copy.deepcopy(c[0])
        with self.assertRaisesRegex(ValueError,'EXACT_REVIEW_SET'):m.choose_reviewed_inputs(c,r,l)

    def test_owner_review_cannot_admit_unknown_or_complete_originals(self):
        c,_,_=fixture_sources()
        for row in c:
            row.update(owner_review_decision='N',admitted_class_id=0,trusted_negative=True,
                research_training_admitted=True,whole_original_negative_admitted=False,use_role='research_train_only')
        owners=[dict(row,review_decision='N',source_role='train') for row in c]
        review={'schema':'english-natural32-owner-negative-review-v1','owner_statement_verbatim':'没有唤醒词',
            'candidate_count':9,'distinct_source_originals':8,'reviewed_negative_samples':488880,
            'full_original_utterance_labels_asserted':False,'positive_pronunciation_gold_created':False,
            'independent_test_admitted':False,'candidates':owners}
        m.validate_owner_review(review,{'candidates':c})
        for change in ({'review_decision':'UNKNOWN'},{'trusted_negative':False},{'source_role':'development'}):
            altered=copy.deepcopy(review);altered['candidates'][0].update(change)
            with self.assertRaises(ValueError):m.validate_owner_review(altered,{'candidates':c})
        altered=copy.deepcopy(review);altered['full_original_utterance_labels_asserted']=True
        with self.assertRaisesRegex(ValueError,'EXACT_OWNER_REVIEW_SCOPE'):m.validate_owner_review(altered,{'candidates':c})

    def test_fixed_draws_equal_eight_sources_and_three_conditions(self):
        sources=['s'+str(i) for i in range(8)];old=old_draws();draws=m.fixed_draws(old,sources)
        self.assertEqual([d['original'] for d in draws],old)
        for source in sources:
            uses=[d for d in draws if d['reviewed_source']==source];self.assertEqual(len(uses),12)
            for cond in ('snr10','new5','new15'):
                self.assertEqual(sum(d['original'][3]['condition']==cond for d in uses),4)

    def test_draws_reject_development_snr5_and_changed_class_order(self):
        for key,value in [('role','development'),('condition','snr5'),('weak_class',2)]:
            old=old_draws();old[0][0][key]=value
            with self.assertRaises(ValueError):m.fixed_draws(old,['s'+str(i) for i in range(8)])

    def test_regression_gate_preserves_all_cases(self):
        a=paired_rows();self.assertTrue(m.paired_case_gate(a,copy.deepcopy(a))['passed'])
        for key,value in [('matched_intent_after_clip_start',False),('research_error',True),
                          ('prefix_events',[{'sample':32000,'keyword':1}]),
                          ('wrong_keyword_after_clip_start',[{'sample':50000,'keyword':2}]),
                          ('extra_same_keyword_events',1),('weak_negative_intent_events',1)]:
            b=copy.deepcopy(a);b[0][key]=value
            self.assertFalse(m.paired_case_gate(a,b)['passed'],key)

    def test_regression_cannot_hide_loss_with_another_improvement(self):
        a=paired_rows();a[1]['matched_intent_after_clip_start']=False;a[1]['research_error']=True
        b=copy.deepcopy(a);b[0]['matched_intent_after_clip_start']=False;b[0]['research_error']=True
        b[1]['matched_intent_after_clip_start']=True;b[1]['research_error']=False
        self.assertFalse(m.paired_case_gate(a,b)['passed'])

    def test_regression_rejects_reordered_or_relabelled_rows(self):
        a=paired_rows();b=copy.deepcopy(a);b[0],b[1]=b[1],b[0]
        with self.assertRaisesRegex(ValueError,'PAIRED_REGRESSION_IDENTITY'):m.paired_case_gate(a,b)
        b=copy.deepcopy(a);b[0]['role']='train'
        with self.assertRaisesRegex(ValueError,'PAIRED_REGRESSION_IDENTITY'):m.paired_case_gate(a,b)

    def test_review_fit_uses_equal_source_means_and_correct_available_clock(self):
        c,r,l=fixture_sources();s=m.choose_reviewed_inputs(c,r,l)
        old={};new={}
        for row in r['rows']:
            rid=row['recording_id'];old[rid]=[];new[rid]=[]
            for t in range(35200,row['frames']+1,1280):
                old[rid].append({'sample':t,'ready':True,'keyword':0,'probabilities':[0.25,0.6,0.15]})
                new[rid].append({'sample':t,'ready':True,'keyword':0,'probabilities':[0.75,0.2,0.05]})
        f=m.reviewed_fit(s,old,new);self.assertTrue(f['passed'])
        self.assertAlmostEqual(f['old_source_balanced_ce'],-__import__('math').log(.25))
        for ci,diag in zip(s['candidates'],f['candidates']):
            self.assertNotIn(ci['reviewed_start'],diag['reviewed_window_all_ready_endpoints'])
            self.assertTrue(all(ci['reviewed_start']<t<=ci['reviewed_end_exclusive'] for t in diag['reviewed_window_all_ready_endpoints']))
        self.assertEqual(f['status'],'TRAINING_FIT_DIAGNOSTIC_NOT_INDEPENDENT_EVALUATION')

    def test_npy_bin_validation_requires_all_six_arrays_in_exact_order(self):
        shapes={'mean96':(96,),'scale96':(96,),'w1':(32,1536),'b1':(32,),'w2':(3,32),'b2':(3,)}
        raw=io.BytesIO();arrays=[]
        with zipfile.ZipFile(raw,'w') as z:
            for i,(key,shape) in enumerate(shapes.items()):
                count=__import__('math').prod(shape);body=struct.pack('<f',float(i+1))*count;arrays.append(body)
                header=repr({'descr':'<f4','fortran_order':False,'shape':shape}).encode()+b'\n'
                z.writestr(key+'.npy',b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+body)
        blob=b'EHKWS01\0'+b''.join(arrays)
        self.assertEqual(len(m.verify_saved_npz_bin(raw.getvalue(),blob)),6)
        with self.assertRaisesRegex(ValueError,'NPZ_BIN_EXACT'):m.verify_saved_npz_bin(raw.getvalue(),blob[:-1]+b'x')

    def test_native_comparison_checks_tail_exact_events_and_numeric_tolerance(self):
        row={'sample':35200,'ready':True,'keyword':0,'logits':[0.,0.,0.],
             'probabilities':[1.,0.,0.],'case_index':0,'condition':'fake','id':'case','local_sample':35200}
        refs=[row]*18982;line=b'35200 1 0 0 0 0 1 0 0\n'
        self.assertTrue(m.native_parity(line*18982,refs)['passed'])
        with self.assertRaisesRegex(ValueError,'NATIVE_COMPLETE_ROW_COUNT'):m.native_parity(line*18981,refs)
        with self.assertRaisesRegex(ValueError,'NATIVE_EXACT_CLOCK_READY_EVENT'):m.native_parity(line*18981+b'35200 1 1 0 0 0 1 0 0\n',refs)
        with self.assertRaisesRegex(ValueError,'NATIVE_SAVED_NPZ_BIN_TOLERANCE'):m.native_parity(line*18981+b'35200 1 0 .01 0 0 1 0 0\n',refs)

    def test_execution_requires_exact_public_first_successor(self):
        env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':m.REPO,'GITHUB_EVENT_NAME':'push',
             'GITHUB_REF':'refs/heads/main','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123'}
        event={'before':m.BEFORE,'after':'a'*40,'ref':'refs/heads/main','repository':{'private':False,'full_name':m.REPO},'deleted':False,'forced':False}
        with mock.patch.object(m.subprocess,'check_output',side_effect=AssertionError('NO_PROCESS')):
            self.assertEqual(m.execution_identity({'main_before':m.BEFORE},env,event,'a'*40,m.BEFORE)['run_attempt'],1)
            for changes in ({'GITHUB_RUN_ATTEMPT':'2'},{'GITHUB_EVENT_NAME':'workflow_dispatch'},{'GITHUB_REF':'refs/heads/x'}):
                with self.assertRaises(ValueError):m.execution_identity({'main_before':m.BEFORE},dict(env,**changes),event,'a'*40,m.BEFORE)
            with self.assertRaises(ValueError):m.execution_identity({'main_before':m.BEFORE},env,dict(event,forced=True),'a'*40,m.BEFORE)

    def test_private_numeric_refuses_external_invocation_before_import(self):
        args=type('Args',(),{'output':Path('/tmp/not-created')})()
        with mock.patch.object(m,'execution_identity',return_value={}),mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'NUMERIC_ONLY'):m.numeric(args,{})

    def test_dependency_commands_are_bounded_fixed_and_never_compile(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);(output/'evidence').mkdir();runner=mock.Mock(output=output)
            with mock.patch.object(m.shutil,'which',return_value='/fake/tool'):
                m.prepare_dependencies(runner)
            calls=runner.command.call_args_list
            self.assertEqual(len(calls),5)
            for call in calls:
                self.assertLessEqual(call.kwargs['timeout'],360)
                self.assertFalse(any(x in call.args[1] for x in ('apt-get','gcc','ffmpeg','onnxruntime')))
            for call in calls[2:4]:
                self.assertIn('--retries',call.args[1]);self.assertEqual(call.args[1][call.args[1].index('--retries')+1],'0')

    def test_dependency_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);(output/'evidence').mkdir();runner=mock.Mock(output=output)
            runner.command.side_effect=[None,None,ValueError('install failed')]
            with mock.patch.object(m.shutil,'which',return_value='/fake/tool'):
                with self.assertRaisesRegex(ValueError,'install failed'):m.prepare_dependencies(runner)
            self.assertEqual(runner.command.call_count,3)

    def test_completed_compute_budgets_cannot_launch_again(self):
        with tempfile.TemporaryDirectory() as td:
            runner=m.Runner(Path(td),10);runner.counts={'numeric':1,'native':1}
            with mock.patch.object(m.time,'monotonic',return_value=0),mock.patch.object(m.subprocess,'Popen',side_effect=AssertionError('FORBIDDEN')) as popen:
                for kind in ('numeric','native'):
                    with self.assertRaisesRegex(ValueError,'COMMAND_BUDGET'):
                        runner.command('second-'+kind,['never-run'],kind=kind)
                popen.assert_not_called()

    def test_cleanup_disappearing_group_is_safe(self):
        with mock.patch.object(m,'group_exists',return_value=True),mock.patch.object(m.os,'killpg',side_effect=ProcessLookupError):
            self.assertEqual(m.cleanup_group(123),{'descendants_found':False,'terminated':False,'killed':False})

    def test_damaged_partial_receipt_is_preserved_and_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'evidence').mkdir();p=root/'evidence/test.started.json';p.write_bytes(b'{')
            result=m.close_interrupted_receipts(root,{'cleanup':{},'timed_out':True})
            self.assertEqual(len(result),1);self.assertEqual(p.read_bytes(),b'{')
            receipt=json.loads((root/'evidence/test.exit.json').read_bytes())
            self.assertIsNone(receipt['returncode']);self.assertFalse(receipt['completed'])

    def test_result_boundary_does_not_claim_full_pcm_or_formal_quality(self):
        b=m.result_boundary({})
        self.assertFalse(b['passed']);self.assertFalse(b['formal_quality_improvement_claim'])
        self.assertFalse(b['strong_english_fsmn_ctc'])
        self.assertEqual(b['full_pcm_new_head'],'NOT_RUN');self.assertEqual(b['independent_far'],'NOT_MEASURED')

if __name__=='__main__':unittest.main()
