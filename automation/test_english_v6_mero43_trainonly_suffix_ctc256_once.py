"""No-work-model, source-only constraints for V6 suffix-constrained CTC."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    'v6_suffix_ctc_once',HERE/'english_v6_mero43_trainonly_suffix_ctc256_once.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Contract(unittest.TestCase):
    def test_only_one_loss_factor_and_immutability(self):
        x=m.source_check()
        self.assertEqual(x['status'],'SOURCE_ONLY_NO_NEW_OPTIMIZER')
        self.assertEqual(x['target'],'Hey Mero')
        self.assertEqual(x['parameters'],43876)
        self.assertEqual(x['train_groups'],35)
        self.assertEqual(x['new_steps'],256)
        self.assertEqual(x['only_changed_factor'],'SUFFIX_CONSTRAINED_CTC_TARGET')
        self.assertEqual(x['prior_B2_real_training_run'],44648437304)
        self.assertEqual(x['prior_B2_optimizer_updates'],256)
        self.assertEqual(x['prefix_samples'],[6400,12800,19200])
        self.assertEqual(m.BEFORE,'9804b3fe816b503346d19c5f4d5a3627a413e840')
        self.assertEqual(m.B1_PT,'20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05')
        self.assertEqual(m.B1_NPZ,'76f413772e2820b3aab27124e1f6b627d9de19e9520068ac205bdea497f0b648')
        self.assertFalse(x['dev_model_inference'])
        self.assertFalse(x['cal_locked_access'])
        self.assertFalse(x['product_promotion'])

    def test_exact_earliest_speech_touch_geometry(self):
        for prefix in m.PREFIX_SAMPLES:
            self.assertEqual(prefix%160,0)
            blank=prefix//160-8
            suffix=prefix//160-3
            self.assertGreater(suffix,blank)
            self.assertEqual(suffix-blank,5)
        src=(HERE/'english_v6_mero43_trainonly_suffix_ctc256_once.py').read_text()
        self.assertIn('speech=lprobs[:,suffix_start:,:]',src)
        self.assertIn('prefix_logits=logits[:, :n_blank, :]',src)
        self.assertIn('SUFFIX_CONSTRAINED_CTC_TARGET',src)
        self.assertNotIn('new DEV PCM used for gradient',src)

    def test_guard_rejects_pull_request_and_missing_authority(self):
        with patch.dict(os.environ,{'GITHUB_EVENT_NAME':'pull_request'}):
            with self.assertRaises(ValueError): m.guard()
        with patch.dict(os.environ,{'GH_TOKEN':''}):
            with self.assertRaises(ValueError): m.verify_prior_b2_evidence()

    def test_first_main_grant_positive_and_parent_mismatch(self):
        head='a'*40
        event={'before':m.BEFORE,'after':head,'ref':'refs/heads/main',
            'repository':{'full_name':m.REPO,'private':False},
            'forced':False,'deleted':False}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'event.json';p.write_text(json.dumps(event))
            env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':m.REPO,
                'GITHUB_EVENT_NAME':'push','GITHUB_REF':'refs/heads/main',
                'GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':head,
                'GITHUB_RUN_ID':'123456','GITHUB_EVENT_PATH':str(p)}
            with patch.dict(os.environ,env),patch.object(
                    m.subprocess,'check_output',return_value=m.BEFORE+'\n'),patch.object(
                    m,'verify_prior_b2_evidence',
                    return_value={'B2_run':44648437304}) as proof:
                grant=m.guard()
                self.assertEqual(grant['prior_real_train']['B2_run'],44648437304)
                proof.assert_called_once()
            event['before']='b'*40;p.write_text(json.dumps(event))
            with patch.dict(os.environ,env),patch.object(
                    m.subprocess,'check_output',return_value=m.BEFORE+'\n'):
                with self.assertRaises(ValueError):m.guard()

if __name__=='__main__':
    unittest.main()
