"""Offline contracts on the exact archived public audio; no model imports."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('label_inputs', HERE/'prepare.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ARCHIVE = Path(os.environ.get('QWEN16_ARCHIVE', str(HERE.parents[1]/m.ARCHIVE_PATH)))


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = m.archive_members(ARCHIVE)
        cls.outputs = m.derive(cls.files)

    def mutate(self, member, change):
        files = dict(self.files)
        value = m.parse(files[member]); change(value)
        files[member] = m.canonical(value)
        freeze = m.parse(files['artifact-freeze.json'])
        freeze['files'][member] = {'bytes':len(files[member]),'sha256':m.sha(files[member])}
        files['artifact-freeze.json'] = m.canonical(freeze)
        return files

    def test_exact_archive(self):
        self.assertEqual(len(self.files),59)
        self.assertEqual(m.sha(ARCHIVE.read_bytes()),m.ARCHIVE_SHA256)

    def test_twelve_blind_inputs(self):
        job = m.parse(self.outputs['blind/job.json'])
        self.assertEqual(len(job['clips']),12)
        self.assertEqual(set(job),{'schema','clips'})
        for row in job['clips']:
            self.assertEqual(set(row),{'audio_id','audio_path','wav_sha256'})
        self.assertEqual([r['wav_sha256'] for r in job['clips']],sorted({r['wav_sha256'] for r in job['clips']}))

    def test_roles_and_family_preserved(self):
        control = m.parse(self.outputs['control/source-map.json'])
        self.assertEqual({r['speaker'] for r in control['rows']},{'Vivian','Uncle_Fu','Dylan'})
        self.assertEqual({r['speaker'] for r in control['excluded_development_a']},{'Serena'})
        for row in control['rows']+control['excluded_development_a']:
            self.assertEqual(row['source_family_id'],'qwen3-customvoice:preset:'+row['speaker'].lower())

    def test_no_labels_or_execution(self):
        handoff = m.parse(self.outputs['handoff.json'])
        self.assertEqual(handoff['actual_asr_calls'],0)
        self.assertEqual(handoff['actual_training_runs'],0)
        self.assertEqual(handoff['selected_seconds'],30.08)
        for row in m.parse(self.outputs['control/source-map.json'])['rows']:
            self.assertIsNone(row['actual_text']); self.assertIsNone(row['actual_tokens'])
            for key in ('ctc_training_allowed','qualification_allowed','evaluation_gold_allowed'):
                self.assertIs(row[key],False)

    def test_no_control_file_in_blind(self):
        blind = [p for p in self.outputs if p.startswith('blind/')]
        self.assertEqual(len(blind),13)
        self.assertFalse(any(p.endswith('.json') and p!='blind/job.json' for p in blind))
        text = self.outputs['blind/job.json'].decode()
        for forbidden in ('你好','小窝','Serena','Vivian','speaker','text','train','prompt'):
            self.assertNotIn(forbidden,text)

    def test_source_row_hashes(self):
        plan = {r['cell_id']:r for r in m.parse(self.files['plan.json'])}
        rows = {r['cell_id']:r for r in m.parse(self.files['output/generation-receipt.json'])['outcomes']}
        for item in m.parse(self.outputs['control/source-map.json'])['rows']:
            self.assertEqual(item['source_plan_record_sha256'],m.sha(m.canonical(plan[item['source_id']])))
            self.assertEqual(item['source_outcome_record_sha256'],m.sha(m.canonical(rows[item['source_id']])))

    def test_audio_bytes_unchanged(self):
        for row in m.parse(self.outputs['control/source-map.json'])['rows']:
            self.assertEqual(self.outputs['blind/audio/'+row['file_sha256']+'.wav'],self.files['output/'+row['source_id']+'.wav'])

    def test_36_calls_are_only_prospective(self):
        request = m.parse(self.outputs['control/capture-request.json'])
        self.assertEqual(request['status'],'NOT_RUN')
        self.assertEqual(request['maximum_recognizer_calls'],36)
        self.assertIs(request['executed'],False)
        self.assertIs(request['requires_reviewed_runtime_source_asset_and_native_config_freeze'],True)

    def test_mutated_zip_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.zip';raw=bytearray(ARCHIVE.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaises(ValueError):m.archive_members(p)

    def test_truncated_zip_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.zip';p.write_bytes(ARCHIVE.read_bytes()[:-1])
            with self.assertRaises(ValueError):m.archive_members(p)

    def test_archive_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.zip';p.symlink_to(ARCHIVE.resolve())
            with self.assertRaises(ValueError):m.archive_members(p)

    def test_duplicate_json_key_refused(self):
        with self.assertRaises(ValueError):m.parse(b'{"a":1,"a":2}')

    def test_nonfinite_json_refused(self):
        for value in ('NaN','Infinity','-Infinity','1e999'):
            with self.subTest(value=value),self.assertRaises(ValueError):m.parse(('{"a":'+value+'}').encode())

    def test_run_boolean_refused(self):
        files=dict(self.files);freeze=m.parse(files['artifact-freeze.json']);freeze['run_attempt']=True
        files['artifact-freeze.json']=m.canonical(freeze)
        with self.assertRaises(ValueError):m.derive(files)

    def test_archive_admission_refused(self):
        files=dict(self.files);freeze=m.parse(files['artifact-freeze.json']);freeze['training_admitted']=True
        files['artifact-freeze.json']=m.canonical(freeze)
        with self.assertRaises(ValueError):m.derive(files)

    def test_role_change_refused(self):
        files=self.mutate('plan.json',lambda rows:rows[4].update(proposed_split='train'))
        with self.assertRaises(ValueError):m.derive(files)

    def test_family_change_refused(self):
        files=self.mutate('plan.json',lambda rows:rows[0].update(source_family_id='new-batch-local'))
        with self.assertRaises(ValueError):m.derive(files)

    def test_invented_label_refused(self):
        files=self.mutate('output/generation-receipt.json',lambda r:r['outcomes'][0].update(actual_text='你好小窝'))
        with self.assertRaises(ValueError):m.derive(files)

    def test_boolean_attempt_refused(self):
        files=self.mutate('output/generation-receipt.json',lambda r:r['outcomes'][0].update(attempts=True))
        with self.assertRaises(ValueError):m.derive(files)

    def test_missing_claim_refused(self):
        files=dict(self.files);del files['output/qwen16-20261030-000.claim']
        with self.assertRaises(ValueError):m.derive(files)

    def test_original_member_drift_refused(self):
        files=dict(self.files);files['output/qwen16-20261030-000.wav']+=b'x'
        with self.assertRaises(ValueError):m.derive(files)

    def test_pcm_geometry_refused(self):
        raw=bytearray(self.files['output/qwen16-20261030-000.wav']);raw[22]=2
        with self.assertRaises(ValueError):m.pcm_identity(bytes(raw))

    def test_pcm_truncation_refused(self):
        with self.assertRaises(ValueError):m.pcm_identity(self.files['output/qwen16-20261030-000.wav'][:-1])

    def test_materialize_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';m.materialize(ARCHIVE,output)
            self.assertEqual(m.verify_outputs(ARCHIVE,output)['selected_train_candidates'],12)

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';output.mkdir();(output/'keep').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):m.materialize(ARCHIVE,output)
            self.assertEqual((output/'keep').read_bytes(),b'keep')

    def test_output_extra_control_in_blind_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';m.materialize(ARCHIVE,output)
            (output/'blind/labels.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):m.verify_outputs(ARCHIVE,output)

    def test_output_role_edit_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';m.materialize(ARCHIVE,output)
            p=output/'control/source-map.json';v=m.parse(p.read_bytes());v['rows'][0]['ctc_training_allowed']=True;p.write_bytes(m.canonical(v))
            with self.assertRaises(ValueError):m.verify_outputs(ARCHIVE,output)

    def test_output_missing_audio_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';m.materialize(ARCHIVE,output)
            next((output/'blind/audio').iterdir()).unlink()
            with self.assertRaises(ValueError):m.verify_outputs(ARCHIVE,output)

    def test_output_audio_alias_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';m.materialize(ARCHIVE,output)
            p=next((output/'blind/audio').iterdir());body=p.read_bytes();p.unlink();target=Path(directory)/'other.wav';target.write_bytes(body);p.symlink_to(target)
            with self.assertRaises(ValueError):m.verify_outputs(ARCHIVE,output)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.old=Path.cwd();os.chdir(self.root)
        self.git('init','-q');self.git('config','user.name','fixture');self.git('config','user.email','fixture@example.invalid')
        self.git('checkout','-qb',m.BRANCH)
        Path('old.txt').write_text('old\n');Path('docs').mkdir();Path(m.MANIFEST).write_text('{}')
        self.git('add','.');self.git('commit','-qm','initial fixture')
        initial=self.git('rev-parse','HEAD').strip()
        blob=self.git('rev-parse','HEAD:old.txt').strip()
        value={'schema':'kws-public-branch-retention-v1','retained_files':[{'path':'old.txt','mode':'100644','git_blob_sha1':blob,'bytes':4,'source_commit':initial}],
            'integration_files':[m.MANIFEST],'expected_retained_files':1,'expected_tracked_files':2}
        Path(m.MANIFEST).write_text(json.dumps(value))
        self.git('add','.');self.git('commit','-qm','manifest fixture')
        self.original=copy.deepcopy(value)
        for name in m.SOURCE_PATHS:
            p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture\n')
        self.git('add','.');self.git('commit','-qm','new source fixture')
        self.source=self.git('rev-parse','HEAD').strip()

    def tearDown(self):
        os.chdir(self.old);self.tmp.cleanup()

    def git(self,*args):
        return subprocess.check_output(['git',*args],stderr=subprocess.PIPE,text=True)

    def test_exact_registration(self):
        m.register();v=json.loads(Path(m.MANIFEST).read_text())
        self.assertEqual(v['retained_files'][0],self.original['retained_files'][0])
        self.assertEqual(len(v['retained_files']),5)
        for r in v['retained_files'][1:]:self.assertEqual(r['source_commit'],self.source)
        self.assertEqual(self.git('diff','--name-only').strip(),m.MANIFEST)

    def test_registration_idempotent(self):
        m.register();self.git('add','.');self.git('commit','-qm','register fixture');before=Path(m.MANIFEST).read_bytes();m.register()
        self.assertEqual(Path(m.MANIFEST).read_bytes(),before)
        self.assertEqual(self.git('status','--porcelain'),'')

    def test_unrelated_addition_refused(self):
        Path('unexpected').write_text('no');self.git('add','.');self.git('commit','-qm','unrelated')
        with self.assertRaises(ValueError):m.register()

    def test_old_source_change_refused(self):
        Path('old.txt').write_text('modified');self.git('add','.');self.git('commit','-qm','bad change')
        with self.assertRaises(ValueError):m.register()

    def test_dirty_tree_refused(self):
        Path('untracked').write_text('no')
        with self.assertRaises(ValueError):m.register()

    def test_wrong_branch_refused(self):
        self.git('checkout','-qb','other')
        with self.assertRaises(ValueError):m.register()


if __name__=='__main__':
    unittest.main(verbosity=2)
