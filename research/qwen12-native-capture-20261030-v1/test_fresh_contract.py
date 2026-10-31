"""Offline contracts only. Never imports a model or creates ASR observations."""
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_contract as fresh


class FreshSlots(unittest.TestCase):
    def test_all_36_slots(self):
        rows = fresh.remaining_plan(fresh.original_job())
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({(r['family'], r['wav_sha256']) for r in rows}), 36)
        for family in fresh.FAMILIES:
            self.assertEqual(len(fresh.remaining_plan(fresh.original_job(), family)), 12)
            self.assertEqual(len(fresh.planned_clips(fresh.original_job(), family)), 12)
        self.assertEqual(sum(r[2] for r in fresh.SOURCES), 481280)

    def test_every_native_start_binding(self):
        for row in fresh.remaining_plan(fresh.original_job()):
            with self.subTest(family=row['family'], audio=row['audio_id']):
                result = fresh.validate_start_slot(fresh.original_job(), row['family'],
                    audio_id=row['audio_id'], source_id=row['source_id'], run_id='123',
                    request_id='qwen12-123-'+row['family']+'-'+row['audio_id'],
                    measured={'file_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256'],'audio_frames':row['audio_frames']})
                self.assertEqual(result, row)

    def test_no_shared_mutable_job(self):
        job = fresh.original_job(); job['clips'].clear()
        self.assertEqual(len(fresh.original_job()['clips']), 12)

    def test_metadata_contamination(self):
        for key in ('text','speaker','intended_text','tokens','seed','proposed_split'):
            job = fresh.original_job(); job['clips'][0][key] = 'invented'
            with self.subTest(key=key), self.assertRaises(ValueError):
                fresh.validate_original_job(job)

    def test_removed_repeated_reordered_or_foreign_clip(self):
        for action in ('remove','repeat','reverse','foreign'):
            job = fresh.original_job()
            if action == 'remove': job['clips'].pop()
            elif action == 'repeat': job['clips'][1] = copy.deepcopy(job['clips'][0])
            elif action == 'reverse': job['clips'].reverse()
            else: job['clips'][0]['wav_sha256'] = '0'*64
            with self.subTest(action=action), self.assertRaises(ValueError):
                fresh.validate_original_job(job)

    def test_invalid_family(self):
        for family in ('other', '', True):
            with self.subTest(family=family), self.assertRaises(ValueError):
                fresh.remaining_plan(fresh.original_job(), family)

    def test_invalid_native_start(self):
        row = fresh.remaining_plan(fresh.original_job(), 'qwen06')[0]
        kwargs = {'audio_id':row['audio_id'],'source_id':row['source_id'],'run_id':'123',
            'request_id':'qwen12-123-qwen06-'+row['audio_id'],
            'measured':{'file_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256'],'audio_frames':row['audio_frames']}}
        for key in ('source_id','request_id','run_id','audio_id'):
            case=copy.deepcopy(kwargs);case[key]='invalid'
            with self.subTest(key=key),self.assertRaises(ValueError):
                fresh.validate_start_slot(fresh.original_job(),'qwen06',**case)
        for key,value in [('audio_frames',True),('audio_frames',float(row['audio_frames'])),
                          ('pcm_sha256','0'*64),('file_sha256','0'*64)]:
            case=copy.deepcopy(kwargs);case['measured'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                fresh.validate_start_slot(fresh.original_job(),'qwen06',**case)

    def test_no_labels_in_fresh_job(self):
        job=fresh.original_job()
        self.assertEqual(set(job),{'schema','clips'})
        for row in job['clips']:
            self.assertEqual(set(row),{'audio_id','audio_path','wav_sha256'})


class RealBlindInput(unittest.TestCase):
    def setUp(self):
        # CI must rederive this directory from the unchanged archived audio.
        self.root=Path(os.environ['KWS_FRESH_BLIND_DIR'])

    def test_exact_archived_audio(self):
        self.assertEqual(fresh.validate_blind(self.root),fresh.original_job())

    def test_reject_extra_root_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'blind';shutil.copytree(self.root,root)
            (root/'answers.json').write_text('{}')
            with self.assertRaises(ValueError):fresh.validate_blind(root)

    def test_reject_extra_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'blind';shutil.copytree(self.root,root)
            (root/'audio/other.wav').write_bytes(b'not audio')
            with self.assertRaises(ValueError):fresh.validate_blind(root)

    def test_reject_changed_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'blind';shutil.copytree(self.root,root)
            path=next((root/'audio').iterdir());body=path.read_bytes()
            path.write_bytes(body[:-1]+bytes([body[-1]^1]))
            with self.assertRaises(ValueError):fresh.validate_blind(root)

    def test_reject_noncanonical_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'blind';shutil.copytree(self.root,root)
            path=root/'job.json';path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):fresh.validate_blind(root)

    def test_reject_file_alias(self):
        for alias in ('symlink','hardlink'):
            with self.subTest(alias=alias),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'blind';shutil.copytree(self.root,root)
                path=next((root/'audio').iterdir());saved=Path(tmp)/'saved.wav'
                path.rename(saved)
                if alias=='symlink':path.symlink_to(saved)
                else:os.link(saved,path)
                with self.assertRaises(ValueError):fresh.validate_blind(root)

    def test_reject_directory_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            link=Path(tmp)/'blind';link.symlink_to(self.root,target_is_directory=True)
            with self.assertRaises(ValueError):fresh.validate_blind(link)

    def test_retained_receipt_api_uses_fresh_plan_without_running_old_launcher(self):
        pipeline=Path(os.environ['KWS_PINNED_PIPELINE'])
        script=r'''
import importlib.util, pathlib, sys
fresh_dir,pipeline=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(fresh_dir))
import fresh_contract as fresh
spec=importlib.util.spec_from_file_location('historical_contract',pipeline/'research/qwen12-native-admission-v1/campaign_contract.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
try:fresh.validate_original_job(old.original_job())
except ValueError:pass
else:raise RuntimeError('historical audio accepted')
# Explicit prospective API injection only; never old main/execute/fresh_run.
sys.modules['campaign_contract']=fresh
spec=importlib.util.spec_from_file_location('fresh_native_receipts',pipeline/'research/qwen12-native-admission-v1/worker_receipts.py')
receipts=importlib.util.module_from_spec(spec);sys.modules[spec.name]=receipts;spec.loader.exec_module(receipts)
if receipts.validate_start_slot is not fresh.validate_start_slot:raise RuntimeError('old slot binding')
if len(receipts.remaining_plan(fresh.original_job()))!=36:raise RuntimeError('fresh slot lost')
if any(len(receipts.remaining_plan(fresh.original_job(),family))!=12 for family in fresh.FAMILIES):raise RuntimeError('family omitted')
if any(name in sys.modules for name in ('torch','transformers','funasr','whisper','qwen_asr')):raise RuntimeError('model runtime imported')
print('fresh36 native receipt API PASS; no historical launcher or model import')
'''
        result=subprocess.run([sys.executable,'-I','-B','-c',script,str(HERE),str(pipeline)],
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
