"""Offline tests only: no network, credentials or model calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fresh_contract as fresh
import run_capture as host

class PublicationProtocol(unittest.TestCase):
    def fixture(self,root):
        import publish_capture as publisher
        (root/'host-receipt.json').write_bytes(b'{}')
        host.retain(root,root/'artifact',{'status':'STOPPED_NO_RETRY'})
        env={'GITHUB_REPOSITORY':host.REPO,'GITHUB_EVENT_NAME':'push',
            'GITHUB_REF':'refs/heads/'+host.EXECUTION_BRANCH,'GITHUB_RUN_ATTEMPT':'1',
            'GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123'}
        event={'created':True,'deleted':False,'after':'a'*40,'repository':{'full_name':host.REPO,'private':False}}
        claim={'ref':host.CLAIM_REF,'object':{'type':'commit','sha':'a'*40}}
        return publisher,env,event,claim

    def test_snapshot_reproducible_and_no_runtime_members(self):
        import io,zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pub,env,event,claim=self.fixture(root)
            first,count=pub.snapshot(root/'artifact');second,_=pub.snapshot(root/'artifact')
            self.assertEqual(first,second);self.assertEqual(count,2)
            with zipfile.ZipFile(io.BytesIO(first)) as archive:
                self.assertEqual(set(archive.namelist()),{'host-receipt.json','artifact-freeze.json'})

    def test_snapshot_rejects_member_change_and_extra_file(self):
        for mode in ('changed','extra'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);pub,env,event,claim=self.fixture(root)
                (root/'artifact'/('host-receipt.json' if mode=='changed' else 'unexpected')).write_bytes(b'changed')
                with self.assertRaises(ValueError):pub.snapshot(root/'artifact')

    def test_failure_only_snapshot_preserves_unknown_outcome(self):
        import publish_capture as pub
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'publication-failure.json').write_bytes(fresh.canonical(
                {'status':'PUBLICATION_FAILED','training_admitted':False,'model_outcome':'UNKNOWN'}))
            payload,count=pub.snapshot(root)
            self.assertGreater(len(payload),0);self.assertEqual(count,1)

    def test_exact_run_and_create_only_publication(self):
        import base64
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pub,env,event,claim=self.fixture(root);calls=[]
            def api(method,path,body=None):
                calls.append((method,path,body))
                if path=='':return {'full_name':host.REPO,'private':False}
                if path=='/actions/runs/123':return {'id':123,'head_sha':'a'*40,'head_branch':host.EXECUTION_BRANCH,
                    'event':'push','run_attempt':1,'path':pub.WORKFLOW}
                if path=='/git/commits/'+'a'*40:return {'sha':'a'*40,'tree':{'sha':'b'*40}}
                if path=='/git/blobs':
                    raw=base64.b64decode(body['content']);return {'sha':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
                if path=='/git/trees':
                    self.assertEqual(set(body),{'tree'});self.assertEqual(len(body['tree']),2);return {'sha':'c'*40}
                if path=='/git/commits':
                    self.assertEqual(body['parents'],['a'*40]);return {'sha':'d'*40}
                if path=='/git/refs':return {'ref':pub.EVIDENCE_REF,'object':{'sha':'d'*40}}
                raise AssertionError('unexpected API')
            result=pub.publish(env,event,claim,root/'artifact',api)
            self.assertEqual(result['evidence_commit'],'d'*40)
            self.assertFalse(result['model_completion_established'])
            self.assertFalse(any(method in ('PATCH','PUT','DELETE') for method,path,body in calls))
            self.assertEqual(sum(path=='/git/refs' for method,path,body in calls),1)

    def test_wrong_workflow_has_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pub,env,event,claim=self.fixture(root);calls=[]
            def api(method,path,body=None):
                calls.append((method,path))
                if path=='':return {'full_name':host.REPO,'private':False}
                return {'id':123,'head_sha':'a'*40,'head_branch':host.EXECUTION_BRANCH,
                    'event':'push','run_attempt':1,'path':'wrong-workflow'}
            with self.assertRaises(ValueError):pub.publish(env,event,claim,root/'artifact',api)
            self.assertTrue(all(method=='GET' for method,path in calls))

    def test_publication_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pub,env,event,claim=self.fixture(root);calls=[]
            def api(method,path,body=None):
                calls.append((method,path));raise RuntimeError('fixture API failure')
            with self.assertRaises(RuntimeError):pub.publish(env,event,claim,root/'artifact',api)
            self.assertEqual(len(calls),1)

    def test_workflow_only_activates_fresh_once_branch_after_source_tests(self):
        raw=(HERE.parents[1]/'.github/workflows/qwen12-native-capture-20261030-v1.yml').read_text()
        self.assertIn('  capture:\n',raw)
        capture=raw.split('  capture:\n',1)[1]
        for expected in ('needs: source-preparation',"github.event_name == 'push'",host.EXECUTION_BRANCH,
                         'github.event.deleted == false','github.run_attempt == 1',
                         'github.event.repository.private == false','timeout-minutes: 70','runs-on: ubuntu-24.04',host.CLAIM_REF):
            self.assertIn(expected,capture)
        self.assertNotIn('github.event.created == true',capture)
        push=raw.split('  push:',1)[1].split('  pull_request:',1)[0]
        self.assertNotIn('paths:',push)
        self.assertNotIn('paths-ignore:',push)
        self.assertNotIn('actions/upload-artifact',capture)
        self.assertNotIn('continue-on-error',capture)
        inference=capture.split('      - name: Run the single native three-family capture',1)[1].split('      - name:',1)[0]
        self.assertNotIn('GH_TOKEN',inference);self.assertNotIn('github.token',inference)


if __name__=='__main__':unittest.main()
