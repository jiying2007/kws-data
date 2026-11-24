"""Small producer binding fixtures; no repository mutation, model or network."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b21_producer_snapshot as producer

class ProducerSnapshot(unittest.TestCase):
    def test_git_blob_matches_known_empty_file_object(self):
        self.assertEqual(producer.blob(b''),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
    def test_hidden_runtime_workflow_is_preserved_in_tree_parser(self):
        name=producer.ROLES['source']['workflow']
        value=producer.tree_entries(('100644 blob '+'1'*40+'\t'+name+'\0').encode())
        self.assertEqual(value[name]['git_blob_sha1'],'1'*40)
        for bad in (('120000 blob '+'1'*40+'\tlink\0').encode(),('100644 blob '+'1'*40+'\t../escape\0').encode()):
            with self.assertRaises(ValueError):producer.tree_entries(bad)
    def test_only_original_workflow_child_can_be_snapshotted(self):
        before,head='1'*40,'2'*40;role='train';spec=producer.ROLES[role];ref='refs/heads/'+spec['branch']
        env=dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY=producer.REPO,GITHUB_EVENT_NAME='push',
                 GITHUB_REF=ref,GITHUB_RUN_ATTEMPT='1',GITHUB_SHA=head,GITHUB_RUN_ID='123')
        event=dict(before=before,after=head,ref=ref,forced=False,deleted=False,
                   repository=dict(full_name=producer.REPO,private=False))
        self.assertEqual(producer.validate_event(role,before,head,[before],[spec['workflow']],env,event)['run_id'],'123')
        for parents,changed in (([before,'3'*40],[spec['workflow']]),([before],[spec['workflow'],'automation/model.py']),([],[])):
            with self.assertRaises(ValueError):producer.validate_event(role,before,head,parents,changed,env,event)
        bad=dict(env,GITHUB_RUN_ATTEMPT='2')
        with self.assertRaises(ValueError):producer.validate_event(role,before,head,[before],[spec['workflow']],bad,event)

if __name__=='__main__':unittest.main()
