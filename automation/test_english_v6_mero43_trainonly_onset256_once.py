"""Strict source-only tests for V6 TRAIN-only causal onset experiment."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v6_trainonly_onset_once', HERE/'english_v6_mero43_trainonly_onset256_once.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class SourceContract(unittest.TestCase):
    def test_exact_machine_qualified_train_and_no_model(self):
        x=mod.source_check()
        self.assertEqual(x['status'],'SOURCE_ONLY_NO_NEW_OPTIMIZER')
        self.assertEqual(x['target'],'Hey Mero')
        self.assertEqual(x['parameters'],43876)
        self.assertEqual(x['train_groups'],35)
        self.assertEqual(x['new_steps'],256)
        self.assertEqual(x['prefix_samples'],[6400,12800,19200])
        self.assertFalse(x['dev_model_inference'])
        self.assertFalse(x['cal_locked_access'])
        self.assertFalse(x['product_promotion'])

    def test_prefix_blanks_safe_geometry(self):
        self.assertTrue(all(x%160==0 and x//160-8>0 for x in mod.PREFIX_SAMPLES))
        self.assertEqual(mod.BEFORE,'c3cf9807a1aa1b7bedb2dbe01918e51af45af63b')
        self.assertEqual(mod.PREVIOUS_SOURCE_TEST_STOP_RUN,44647610854)
        self.assertEqual(mod.source_check()['previous_optimizer_updates'],0)
        self.assertEqual(mod.B1_ZIP,'bab018a63aaf4cc760ee835b8554ebacf73e6ee77c37adc7c9d5008b1a03d7ce')
        self.assertEqual(mod.B1_PT,'20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05')

    def test_no_replay_from_pull_request_context(self):
        # The source-only PR must NEVER receive a training guard grant.
        with patch.dict(os.environ,{'GITHUB_EVENT_NAME':'pull_request'}):
            with self.assertRaises(ValueError):
                mod.guard()

    def test_exact_first_main_grant_is_valid_not_misreported_as_failure(self):
        # This regression catches the previous broken expectation that ALL
        # source-contract tests must raise, even on the valid first-main push.
        head='a'*40
        event={'before':mod.BEFORE,'after':head,
               'ref':'refs/heads/main',
               'repository':{'full_name':mod.REPO,'private':False},
               'forced':False,'deleted':False}
        with tempfile.TemporaryDirectory(prefix='v6-r2-guard-test-') as tmp:
            event_path=Path(tmp)/'event.json'
            event_path.write_text(json.dumps(event),encoding='utf-8')
            variables={
               'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':mod.REPO,
               'GITHUB_EVENT_NAME':'push','GITHUB_REF':'refs/heads/main',
               'GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':head,
               'GITHUB_RUN_ID':'12345','GITHUB_EVENT_PATH':str(event_path)}
            with patch.dict(os.environ,variables), patch.object(
                    mod.subprocess,'check_output',return_value=mod.BEFORE+'\n'):
                grant=mod.guard()
            self.assertEqual(grant['parent'],mod.BEFORE)
            self.assertEqual(grant['source_commit'],head)
            self.assertEqual(grant['previous_optimizer_updates'],0)
            event['before']='b'*40
            event_path.write_text(json.dumps(event),encoding='utf-8')
            with patch.dict(os.environ,variables), patch.object(
                    mod.subprocess,'check_output',return_value=mod.BEFORE+'\n'):
                with self.assertRaises(ValueError):
                    mod.guard()

if __name__=='__main__':unittest.main()
