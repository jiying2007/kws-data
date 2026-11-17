"""Source-only leakage/launch boundary checks; no tensors, models or network."""
from pathlib import Path
import copy
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('teacher_retention_under_test',ROOT/'automation/english_teacher_retention_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def invented_geometry():
    rows=[]
    for i in range(225):
        rows.append({'condition':'clean','id':'invented-train-'+str(i),'role':'train',
            'head_instance_ends':list(range(82 if i==224 else 37))})
    draws=[{'original':[{'condition':r['condition'],'id':r['id']} for r in rows]}]
    return rows,draws


class TeacherInputBoundary(unittest.TestCase):
    def test_complete_training_membership(self):
        geometry,draws=invented_geometry()
        self.assertEqual(len(m.teacher_keys(geometry,draws)),225)

    def test_development_role_cannot_be_teacher_input(self):
        geometry,draws=invented_geometry();geometry[0]['role']='development'
        with self.assertRaisesRegex(ValueError,'TEACHER_TRAIN_ONLY'):
            m.teacher_keys(geometry,draws)

    def test_development_conditions_cannot_be_relabelled_train(self):
        for condition in ('snr5','new8','natural','reviewed'):
            with self.subTest(condition=condition):
                geometry,draws=invented_geometry()
                geometry[0]['condition']=draws[0]['original'][0]['condition']=condition
                with self.assertRaisesRegex(ValueError,'TEACHER_TRAIN_ONLY'):
                    m.teacher_keys(geometry,draws)

    def test_extra_or_missing_teacher_bag_fails(self):
        geometry,draws=invented_geometry();geometry.pop()
        with self.assertRaisesRegex(ValueError,'ALL_225'):
            m.teacher_keys(geometry,draws)

    def test_teacher_windows_must_remain_frozen(self):
        geometry,draws=invented_geometry();geometry[0]['head_instance_ends'].append(999)
        with self.assertRaisesRegex(ValueError,'EXACT_8370'):
            m.teacher_keys(geometry,draws)


class SeparateExecutionBoundary(unittest.TestCase):
    def setUp(self):
        self.before='a'*40;self.head='b'*40
        self.plan={'enabled':True,'main_before':self.before}
        self.env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':m.REPO,'GITHUB_EVENT_NAME':'push',
            'GITHUB_REF':'refs/heads/main','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':self.head,'GITHUB_RUN_ID':'123'}
        self.event={'before':self.before,'after':self.head,'ref':'refs/heads/main','deleted':False,'forced':False,
            'repository':{'private':False,'full_name':m.REPO}}

    def call(self):
        return m.execution_identity(self.plan,self.env,self.event,self.head,self.before)

    def test_new_source_identity_is_separate(self):
        self.assertEqual(self.call()['main_before'],self.before)
        self.assertEqual(m.frozen.BEFORE,'c312a6ef2335af0b4d16c17c598693fc95273c0c')
        self.assertNotEqual(m.frozen.SCHEMA,m.SCHEMA)

    def test_disabled_plan_cannot_execute(self):
        self.plan['enabled']=False
        with self.assertRaisesRegex(ValueError,'NEW_PUBLIC_MAIN_ONCE_ONLY'):self.call()

    def test_reexecution_attempt_is_rejected(self):
        self.env['GITHUB_RUN_ATTEMPT']='2'
        with self.assertRaisesRegex(ValueError,'NEW_PUBLIC_MAIN_ONCE_ONLY'):self.call()

    def test_private_or_forced_source_is_rejected(self):
        for mutation in ('private','forced'):
            with self.subTest(mutation=mutation):
                old=copy.deepcopy(self.event)
                if mutation=='private':self.event['repository']['private']=True
                else:self.event['forced']=True
                with self.assertRaisesRegex(ValueError,'EXACT_NEW_PUBLIC_SUCCESSOR'):self.call()
                self.event=old

    def test_stale_parent_cannot_trigger(self):
        with self.assertRaisesRegex(ValueError,'NEW_WORKTREE_EXACT_SUCCESSOR'):
            m.execution_identity(self.plan,self.env,self.event,self.head,'c'*40)

    def test_local_execution_has_no_bypass(self):
        with self.assertRaisesRegex(ValueError,'NEW_PUBLIC_MAIN_ONCE_ONLY'):
            m.execution_identity(self.plan,{},self.event,self.head,self.before)


class RetainedEvidenceContracts(unittest.TestCase):
    def test_interrupted_teacher_progress_never_claims_planned_completion(self):
        progress={'teacher_train_forwards_started':3,'teacher_train_forwards_completed':2,
                  'teacher_train_head_windows':74}
        result=m.partial_training_receipt(progress,[],[],[],[])
        self.assertEqual(result['teacher_train_bag_forwards_started'],3)
        self.assertEqual(result['teacher_train_bag_forwards_completed'],2)
        self.assertEqual(result['teacher_train_head_windows_completed'],74)
        self.assertEqual(result['planned_teacher_train_bag_forwards'],225)
        self.assertEqual(result['planned_teacher_train_head_windows'],8370)

    def test_hidden_source_upload_and_bound_launch(self):
        source=(ROOT/m.WORKFLOW_PATH).read_text()
        actual=m.loads(m.regular(ROOT/m.PLAN_PATH))
        m.workflow_contract(source,actual)
        with self.assertRaisesRegex(ValueError,'MANIFEST_HIDDEN'):
            m.workflow_contract(source.replace('include-hidden-files: true','include-hidden-files: false'),actual)
        enabled={'enabled':True,'main_before':'a'*40}
        bound=source.replace('false && github.repository','github.repository').replace(
            actual['main_before'] if actual['enabled'] else 'DISABLED_PENDING_NEW_SOURCE_REVIEW',enabled['main_before'])
        m.workflow_contract(bound,enabled)
        with self.assertRaisesRegex(ValueError,'WORKFLOW_EXACT_PLAN_MAIN_BEFORE'):
            m.workflow_contract(bound,{'enabled':True,'main_before':'b'*40})
        with self.assertRaisesRegex(ValueError,'WORKFLOW_EXACT_PLAN_MAIN_BEFORE'):
            m.workflow_contract(bound,{'enabled':False,'main_before':None})

    def test_runtime_version_and_seed_cannot_silently_drift(self):
        source=(ROOT/m.WORKFLOW_PATH).read_text()
        actual=m.loads(m.regular(ROOT/m.PLAN_PATH))
        with self.assertRaisesRegex(ValueError,'PINNED_PYTHON312'):
            m.workflow_contract(source.replace("python-version: '3.12'","python-version: '3.13'"),actual)
        with self.assertRaisesRegex(ValueError,'FROZEN_NUMERIC_ENVIRONMENT'):
            m.workflow_contract(source.replace("PYTHONHASHSEED: '20261117'","PYTHONHASHSEED: '0'"),actual)


if __name__=='__main__':
    unittest.main()
