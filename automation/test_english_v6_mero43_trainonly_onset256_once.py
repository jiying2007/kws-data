"""Strict source-only tests for V6 TRAIN-only causal onset experiment."""
import importlib.util
from pathlib import Path
import unittest
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
        self.assertEqual(mod.BEFORE,'0e3957370215ddf3d502441b6b859a684d92bf21')
        self.assertEqual(mod.B1_ZIP,'bab018a63aaf4cc760ee835b8554ebacf73e6ee77c37adc7c9d5008b1a03d7ce')
        self.assertEqual(mod.B1_PT,'20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05')

    def test_does_not_replay_outside_exact_main(self):
        with self.assertRaises(ValueError):
            mod.guard()

if __name__=='__main__':unittest.main()
