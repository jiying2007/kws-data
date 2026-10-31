"""No-model tests for the prospective capture entry and retained API adapter."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fresh_contract as fresh
import run_capture as host
import worker


class ExecutionAdmission(unittest.TestCase):
    def setUp(self):
        self.env={'GITHUB_REPOSITORY':host.REPO,'GITHUB_EVENT_NAME':'push',
            'GITHUB_REF':'refs/heads/'+host.EXECUTION_BRANCH,'GITHUB_RUN_ATTEMPT':'1',
            'GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123'}
        self.event={'created':True,'deleted':False,'after':'a'*40,'repository':{'full_name':host.REPO,'private':False}}
        self.claim={'ref':host.CLAIM_REF,'object':{'type':'commit','sha':'a'*40}}

    def test_fixed_admission(self):
        self.assertEqual(host.validate_execution(self.env,self.event,self.claim),('a'*40,'123'))

    def test_reject_replay_or_foreign_context(self):
        for key,value in [('GITHUB_RUN_ATTEMPT','2'),('GITHUB_EVENT_NAME','pull_request'),
            ('GITHUB_REF','refs/heads/main'),('GITHUB_REPOSITORY','other/repo'),('GITHUB_SHA','bad'),('GITHUB_RUN_ID','0')]:
            env=dict(self.env);env[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):host.validate_execution(env,self.event,self.claim)

    def test_reject_existing_or_private_branch(self):
        for key,value in [('created',False),('created',1),('deleted',True),('after','b'*40)]:
            event=copy.deepcopy(self.event);event[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):host.validate_execution(self.env,event,self.claim)
        event=copy.deepcopy(self.event);event['repository']['private']=True
        with self.assertRaises(ValueError):host.validate_execution(self.env,event,self.claim)

    def test_claim_bound_to_exact_source(self):
        for claim in ({},{'ref':host.CLAIM_REF,'object':{'type':'commit','sha':'b'*40}},
            {'ref':'refs/tags/other','object':{'type':'commit','sha':'a'*40}}):
            with self.subTest(claim=claim),self.assertRaises(ValueError):host.validate_execution(self.env,self.event,claim)


class SourceStaging(unittest.TestCase):
    def setUp(self):
        self.pipeline=Path(os.environ['KWS_PINNED_PIPELINE'])
        self.freeze=json.loads((self.pipeline/'research/qwen12-native-admission-v1/execution-freeze.json').read_bytes())

    def test_exact_profiles_and_explicit_new_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile in ('setup_code','inference_code'):
                destination=Path(tmp)/profile
                bindings=host.stage_sources(self.pipeline,self.freeze,profile,destination)
                self.assertEqual(len(bindings),6)
                for name,row in self.freeze[profile].items():
                    self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(),row['sha256'])
                for name,row in bindings.items():
                    body=(destination/name).read_bytes()
                    self.assertEqual(len(body),row['bytes']);self.assertEqual(hashlib.sha256(body).hexdigest(),row['sha256'])

    def test_reject_upstream_source_drift(self):
        freeze=copy.deepcopy(self.freeze);name=next(iter(freeze['inference_code']))
        freeze['inference_code'][name]['sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
            host.stage_sources(self.pipeline,freeze,'inference_code',Path(tmp)/'code')

    def test_five_original_archives_bind_both_profiles(self):
        self.assertEqual(len(host.SOURCE_ARCHIVES),5)
        for profile,expected_count in (('setup_code',52),('inference_code',47)):
            members=host.source_members(self.pipeline,self.freeze,profile)
            self.assertEqual(len(members),expected_count)
            self.assertEqual(set(members),set(self.freeze[profile]))
            for name,archive in host.SOURCE_ARCHIVES.items():
                with self.subTest(profile=profile,name=name):
                    self.assertEqual(members[name],(self.pipeline/archive).read_bytes())
                    self.assertNotEqual(members[name],(self.pipeline/name).read_bytes())

    def test_archive_not_replaced_by_live_bytes_or_missing_file(self):
        read=fresh.bounded_read
        for name,archive in host.SOURCE_ARCHIVES.items():
            for kind in ('live','modified','missing'):
                def changed(path,maximum,archive=archive,name=name,kind=kind):
                    if path==self.pipeline/archive:
                        if kind=='missing':raise FileNotFoundError(archive)
                        return read(self.pipeline/name,maximum) if kind=='live' else read(path,maximum)+b' '
                    return read(path,maximum)
                with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as tmp:
                    destination=Path(tmp)/'code'
                    with patch.object(fresh,'bounded_read',changed),self.assertRaises((OSError,ValueError)):
                        host.stage_sources(self.pipeline,self.freeze,'inference_code',destination)
                    self.assertFalse(destination.exists())

    def test_source_freeze_bytes_are_not_replaced(self):
        read=fresh.bounded_read
        target=self.pipeline/'research/qwen12-native-admission-v1/execution-freeze.json'
        def changed(path,maximum):
            raw=read(path,maximum)
            return raw+b' ' if path==target else raw
        with patch.object(fresh,'bounded_read',changed),self.assertRaisesRegex(ValueError,'original source freeze'):
            host.source_members(self.pipeline,self.freeze,'inference_code')

    def test_only_two_source_profiles(self):
        for profile in ('host_code','other','',None):
            with self.subTest(profile=profile),self.assertRaisesRegex(ValueError,'fixed source profile'):
                host.source_members(self.pipeline,self.freeze,profile)

    def test_embedded_dependency_map_is_independently_checked(self):
        for pins in ({},[],{'core/sense_binding.py':'0'*64},{'missing.py':'0'*64}):
            with self.subTest(pins=pins),patch.object(host.ast,'literal_eval',return_value=pins),self.assertRaises(ValueError):
                host.source_members(self.pipeline,self.freeze,'inference_code')

    def test_staging_preserves_live_and_archived_checkout_bytes(self):
        names=set(self.freeze['setup_code'])|set(host.SOURCE_ARCHIVES.values())
        names.add('research/qwen12-native-admission-v1/execution-freeze.json')
        before={name:hashlib.sha256((self.pipeline/name).read_bytes()).hexdigest() for name in names}
        with tempfile.TemporaryDirectory() as tmp:
            host.stage_sources(self.pipeline,self.freeze,'setup_code',Path(tmp)/'code')
        after={name:hashlib.sha256((self.pipeline/name).read_bytes()).hexdigest() for name in names}
        self.assertEqual(before,after)

    def test_fresh_worker_import_binds_all_12_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            code=Path(tmp)/'code';host.stage_sources(self.pipeline,self.freeze,'inference_code',code)
            script=r'''
import importlib.util,pathlib,sys
path=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('new_worker',path);w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
native=w.bind_fresh_api()
if native.validate_original_job is not w.fresh.validate_original_job:raise RuntimeError('historical job binding')
if any(len(native.planned_clips(w.fresh.original_job(),f))!=12 for f in w.fresh.FAMILIES):raise RuntimeError('lost fresh slot')
if any(n in sys.modules for n in ('torch','transformers','whisper','funasr','qwen_asr')):raise RuntimeError('model imported')
print('staged fresh worker import and all36 plan PASS; zero runtime imports')
'''
            result=subprocess.run([sys.executable,'-I','-B','-c',script,str(code/'research'/fresh.CAMPAIGN/'native_worker.py')],
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


class SavedCompletion(unittest.TestCase):
    def fixture(self,root):
        for row in fresh.remaining_plan(fresh.original_job(),'qwen06'):
            directory=root/row['audio_id'];directory.mkdir()
            start={'schema':'retained-asr-started-v1','file_sha256':row['wav_sha256'],
                'pcm_sha256':row['pcm_sha256'],'audio_frames':row['audio_frames'],'audio_id':row['audio_id'],
                'source_id':row['source_id'],'run_id':'123','executed_head_sha':'a'*40,'recognizer_id':'qwen06',
                'request_id':'qwen12-123-qwen06-'+row['audio_id'],
                'original_job_sha256':hashlib.sha256(fresh.canonical(fresh.original_job())).hexdigest()}
            raw=fresh.canonical({'invented':True});text=fresh.canonical({'actual_text':'invented fixture only'})
            started=fresh.canonical(start)
            end={**start,'schema':'retained-asr-completed-v1','status':'success','decoder_completed':True,
                'processed_start_frame':0,'processed_end_frame':row['audio_frames'],
                'started_sha256':hashlib.sha256(started).hexdigest(),'raw_output_sha256':hashlib.sha256(text).hexdigest(),
                'raw_native_output_sha256':hashlib.sha256(raw).hexdigest()}
            for name,body in [('started.json',started),('completed.json',fresh.canonical(end)),
                    ('raw-transcript.json',text),('raw-native-output.json',raw)]:
                (directory/name).write_bytes(body)

    def test_bound_saved_receipt_count_not_acoustic_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(worker.verify_completed(root,fresh.original_job(),'qwen06')['completed'],12)

    def test_reject_wrong_expected_run_or_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(worker.verify_completed(root,fresh.original_job(),'qwen06',expected_run='123',expected_head='a'*40)['completed'],12)
            for kwargs in ({'expected_run':'124'},{'expected_head':'b'*40}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                    worker.verify_completed(root,fresh.original_job(),'qwen06',**kwargs)

    def test_reject_missing_or_changed_raw_receipt(self):
        for name in ('started.json','completed.json','raw-transcript.json','raw-native-output.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root);(root/'clip-000001'/name).unlink()
                with self.assertRaises((OSError,ValueError)):worker.verify_completed(root,fresh.original_job(),'qwen06')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'clip-000001/raw-native-output.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):worker.verify_completed(root,fresh.original_job(),'qwen06')

    def test_reject_incomplete_decoder_or_frame_count(self):
        for field,value in [('decoder_completed',1),('processed_start_frame',False),('processed_end_frame',1)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.fixture(root);p=root/'clip-000001/completed.json'
                row=json.loads(p.read_bytes());row[field]=value;p.write_bytes(fresh.canonical(row))
                with self.assertRaises(ValueError):worker.verify_completed(root,fresh.original_job(),'qwen06')

    def test_reject_extra_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'clip-000013').mkdir()
            with self.assertRaises(ValueError):worker.verify_completed(root,fresh.original_job(),'qwen06')

    def test_export_never_includes_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'asr-runtime').mkdir();(root/'asr-runtime/weights.bin').write_bytes(b'not public')
            (root/'host-receipt.json').write_bytes(b'{}');public=root/'artifact'
            host.retain(root,public,{'status':'STOPPED_NO_RETRY'})
            self.assertEqual({p.name for p in public.iterdir()},{'host-receipt.json','artifact-freeze.json'})

    def test_export_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'decode').mkdir();(root/'secret').write_bytes(b'not public')
            (root/'decode/alias').symlink_to(root/'secret')
            with self.assertRaises(ValueError):host.retain(root,root/'artifact',{'status':'STOPPED_NO_RETRY'})
            self.assertEqual({p.name for p in (root/'artifact').iterdir()},{'publication-failure.json'})


class PublicationAndSetup(unittest.TestCase):
    def test_native_output_paths_and_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'decode/qwen06/clip-000001/raw-native-output.json'
            source.parent.mkdir(parents=True);body=b'{"fixture":true}';source.write_bytes(body)
            host.retain(root,root/'artifact',{'status':'STOPPED_NO_RETRY'})
            target=root/'artifact/output/qwen06/clip-000001/raw-native-output.json'
            self.assertEqual(target.read_bytes(),body)
            self.assertFalse((root/'artifact/decode').exists())
            manifest=json.loads((root/'artifact/artifact-freeze.json').read_bytes())
            self.assertEqual(manifest['files']['output/qwen06/clip-000001/raw-native-output.json'],
                {'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
            self.assertFalse(manifest['training_admitted'])
            self.assertEqual(list(root.glob('.capture-public-*')),[])

    def test_directory_alias_leaves_only_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'private').mkdir();(root/'private/do-not-copy').write_bytes(b'fixture')
            (root/'decode').symlink_to(root/'private',target_is_directory=True)
            with self.assertRaises(ValueError):host.retain(root,root/'artifact',{'status':'STOPPED_NO_RETRY'})
            self.assertEqual({p.name for p in (root/'artifact').iterdir()},{'publication-failure.json'})
            self.assertEqual((root/'private/do-not-copy').read_bytes(),b'fixture')
            self.assertEqual(list(root.glob('.capture-public-*')),[])

    def test_existing_artifact_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/'artifact';public.mkdir();(public/'keep').write_bytes(b'original')
            with self.assertRaises(ValueError):host.retain(root,public,{'status':'STOPPED_NO_RETRY'})
            self.assertEqual({p.name for p in public.iterdir()},{'keep'})
            self.assertEqual((public/'keep').read_bytes(),b'original')

    def test_copy_error_and_copy_drift_do_not_publish_partial_evidence(self):
        write=Path.write_bytes
        for kind in ('error','drift'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'host-receipt.json').write_bytes(b'{}')
                def damaged(path,body):
                    if path.name=='host-receipt.json' and path.parent.name=='artifact':
                        if kind=='error':raise OSError('fixture write error')
                        return write(path,body+b' ')
                    return write(path,body)
                with patch.object(Path,'write_bytes',damaged),self.assertRaises((OSError,ValueError)):
                    host.retain(root,root/'artifact',{'status':'STOPPED_NO_RETRY'})
                self.assertEqual({p.name for p in (root/'artifact').iterdir()},{'publication-failure.json'})
                self.assertEqual((root/'host-receipt.json').read_bytes(),b'{}')
                self.assertEqual(list(root.glob('.capture-public-*')),[])

    def test_shared_setup_uses_intact_package_and_restores_fresh_scope(self):
        for fails in (False,True):
            with self.subTest(fails=fails):
                library=Mock();library.EXPERIMENT=fresh.CAMPAIGN
                arguments=[object() for _ in range(7)]
                def supervise(*args):
                    self.assertEqual(library.EXPERIMENT,'qwen12-native-admission-v1')
                    self.assertEqual(args,(*arguments[:-1],'setup',arguments[-1]))
                    if fails:raise RuntimeError('fixture setup failure')
                    return 1.25
                library.supervise=supervise
                if fails:
                    with self.assertRaises(RuntimeError):host.supervise_shared_setup(library,*arguments)
                else:self.assertEqual(host.supervise_shared_setup(library,*arguments),1.25)
                self.assertEqual(library.EXPERIMENT,fresh.CAMPAIGN)

    def test_output_location_precedes_host_library_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);event=root/'event.json';claim=root/'claim.json';output=root/'job-output'
            event.write_text(json.dumps({'created':True,'deleted':False,'after':'a'*40,
                'repository':{'full_name':host.REPO,'private':False}}))
            claim.write_text(json.dumps({'ref':host.CLAIM_REF,'object':{'sha':'a'*40,'type':'commit'}}))
            env={'GITHUB_REPOSITORY':host.REPO,'GITHUB_EVENT_NAME':'push',
                'GITHUB_REF':'refs/heads/'+host.EXECUTION_BRANCH,'GITHUB_RUN_ATTEMPT':'1',
                'GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123','GITHUB_EVENT_PATH':str(event),
                'RUNNER_TEMP':str(root),'GITHUB_OUTPUT':str(output)}
            with patch.dict(os.environ,env,clear=True),patch.object(host,'git',side_effect=['a'*40,'',host.PIPELINE_SHA,'']),patch.object(host,'load',side_effect=RuntimeError('fixture before imports')):
                with self.assertRaises(RuntimeError):host.execute(root/'pipeline',claim)
            self.assertEqual(output.read_text(),'evidence_root='+str(root/(fresh.CAMPAIGN+'-123')/'artifact')+'\n')


if __name__=='__main__':unittest.main()
