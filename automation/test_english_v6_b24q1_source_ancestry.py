"""Q1 source entry keeps original TRAIN and immutable fail-only proof authority."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b24q1_source_ancestry as lineage


class SourceAncestryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((lineage.metadata.ROOT/lineage.metadata.STOP_REPORT_PATH).read_bytes())
        cls.lock=json.loads((lineage.metadata.ROOT/'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json').read_bytes())
        cls.opening=dict(original_source_lock=cls.lock,expected_failure_report_sha256=lineage.metadata.STOP_REPORT_SHA,
            assessment=dict(path=lineage.metadata.PREFIX+'assessment.json',sha256=lineage.ASSESSMENT_SHA))

    def test_original_train_functions_retain_original_module_globals(self):
        for name in ('training_binding','training_binding_sha','validate_training_live','verify_training'):
            self.assertIs(getattr(lineage,name),getattr(lineage.original,name))
        training=lineage.training_binding()
        self.assertEqual(training['assessment_sha256'],lineage.metadata.TRAINING_ASSESSMENT_SHA)
        self.assertNotEqual(training['assessment_sha256'],lineage.ASSESSMENT_SHA)
        self.assertEqual(training['training']['model_pt_sha256'],lineage.metadata.FROZEN_MODELS)

    def test_q1_plan_is_approved_and_distinct_from_original_train(self):
        value=lineage.read_plan()
        self.assertEqual(lineage.digest(lineage.ASSESSMENT.read_bytes()),lineage.ASSESSMENT_SHA)
        self.assertEqual(value['training_assessment_sha256'],lineage.metadata.TRAINING_ASSESSMENT_SHA)
        self.assertEqual(value['approval']['sha256'],lineage.metadata.APPROVAL_SHA)
        self.assertTrue(value['original_whole_panel_empty_return_policy_changed'])
        self.assertFalse(value['execution_authorized'])
        self.assertFalse(value['automatic_R2_allowed'])

    def test_entry_rejects_a_different_approved_assessment_binding(self):
        identity={'invented':'test-only'}
        with mock.patch.object(lineage,'_original_execution_guard',return_value=identity), \
             mock.patch.object(lineage.failure,'read_binding',return_value=self.opening):
            self.assertEqual(lineage.execution_guard('invented'),identity)
        bad=copy.deepcopy(self.opening);bad['assessment']['sha256']='0'*64
        with mock.patch.object(lineage,'_original_execution_guard',return_value=identity), \
             mock.patch.object(lineage.failure,'read_binding',return_value=bad),self.assertRaises(ValueError):
            lineage.execution_guard('invented')

    def test_original_fail_only_report_requires_exact_corrected_bytes(self):
        self.assertIs(lineage.validate_stop_report(self.report,self.opening),self.report)
        for key,value in (('primary_allowed',True),('total_valid_payloads_reparsed',489),
                          ('original_phone_model_vocab_finite_logits_and_greedy_ctc_recomputed',False),
                          ('original_candidate_pt_sha256','0'*64),('R1_source_stop_permanent',False)):
            bad=copy.deepcopy(self.report);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                lineage.validate_stop_report(bad,self.opening)

    def test_saved_stop_calls_only_the_exact_immutable_direct_worker(self):
        # This mock tests child-process identity and output binding only. It
        # does not claim a new complete original readback or scientific call.
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);binding=directory/'bound.json';binding.write_bytes(b'{"fixture":true}')
            (directory/lineage.STOP_BINDING).write_bytes(binding.read_bytes())
            raw=lineage.failure.canonical(self.report)
            (directory/lineage.STOP_REPORT).write_bytes(raw)
            (directory/'ORIGINAL_B24R1_STOP').mkdir()
            def direct_worker(argv):
                self.assertEqual(argv[:4],[sys.executable,'-I','-S','-B'])
                self.assertEqual(argv[4:7],[str(lineage.failure.READER_PATH.resolve()),'--verify-worker',
                                          str((directory/'ORIGINAL_B24R1_STOP').resolve())])
                self.assertEqual(len(argv),8)
                self.assertEqual(Path(argv[7]).read_bytes(),lineage.failure.canonical(self.lock))
                return raw
            with mock.patch.object(lineage.failure,'BINDING',binding), \
                 mock.patch.object(lineage.failure,'read_binding',return_value=self.opening), \
                 mock.patch.object(lineage.failure,'verify_transport'), \
                 mock.patch.object(lineage.failure,'verify_in_scope',side_effect=lambda directory,lock,execute:execute()), \
                 mock.patch.object(lineage.subprocess,'check_output',side_effect=direct_worker) as child:
                opening,report=lineage.saved_stop(directory)
                self.assertEqual((opening,report),(self.opening,self.report))
                child.assert_called_once()
            with mock.patch.object(lineage.failure,'BINDING',binding), \
                 mock.patch.object(lineage.failure,'read_binding',return_value=self.opening), \
                 mock.patch.object(lineage.failure,'verify_transport'), \
                 mock.patch.object(lineage.failure,'verify_in_scope',side_effect=lambda directory,lock,execute:execute()), \
                 mock.patch.object(lineage.subprocess,'check_output',return_value=raw+b' '),self.assertRaises(ValueError):
                lineage.saved_stop(directory)

    def test_observer_consumption_binds_entire_new_policy_registration(self):
        identity=dict(run_id='1',run_head='a'*40,run_attempt=1,run_branch=lineage.BRANCH,prepared_parent='b'*40)
        registration=dict(identity,source_rows=96,max_calls=96,source_manifest_sha256='c'*64,
            orchestration_contract_sha256=lineage.metadata.ORCHESTRATION_SHA,
            quarantine_helper_sha256='d'*64,observer_implementation_sha256='e'*64)
        consumed=lineage.observer_document('parakeet',registration)
        self.assertEqual(consumed['registration'],registration)
        self.assertEqual(consumed['maximum_native_calls'],96)
        changed=dict(registration,orchestration_contract_sha256='f'*64)
        self.assertNotEqual(lineage.observer_document('parakeet',changed),consumed)
        self.assertEqual(consumed['new_kws_forwards'],0)

    def test_capacity_failure_precedes_any_consumption_record(self):
        transport=types.ModuleType('english_v6_b24q1_source_transport')
        transport.verify_transport=mock.Mock()
        capacity=types.ModuleType('english_v6_b24q1_capacity')
        capacity.primary_source_budget=mock.Mock(side_effect=ValueError('fixture capacity STOP'))
        identity={'test_only':'identity'}
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.dict(sys.modules,{transport.__name__:transport,capacity.__name__:capacity}), \
             mock.patch.object(lineage,'execution_guard',return_value=identity), \
             mock.patch.object(lineage,'require_authorization',return_value={'source_execution':identity}), \
             mock.patch.object(lineage,'one_execution'), \
             mock.patch.object(lineage,'read_plan',return_value={'primary':{'natural':{'fixture':True}}}), \
             mock.patch.object(lineage.qa,'record') as record, \
             mock.patch.object(lineage,'consumption_document',side_effect=AssertionError('no consumption after failure')):
            with self.assertRaisesRegex(ValueError,'fixture capacity STOP'):
                lineage.consumption(temporary,'invented-corpus','invented-parent')
            record.assert_not_called()
            self.assertFalse((Path(temporary)/'CONSUMPTION.json').exists())

    def test_consumption_binds_exact_capacity_bytes(self):
        transport=types.ModuleType('english_v6_b24q1_source_transport')
        transport.read_receipt=mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name in ('CAPACITY.json','AUTHORIZATION.json','CORPUS_TRANSPORT.json'):
                (directory/name).write_bytes(b'{"fixture":true}\n')
            with mock.patch.dict(sys.modules,{transport.__name__:transport}), \
                 mock.patch.object(lineage,'require_authorization',return_value={'source_execution':{'test_only':True}}):
                original=lineage.consumption_document(directory)
                self.assertEqual(original['capacity_sha256'],lineage.qa.sha(directory/'CAPACITY.json'))
                (directory/'CAPACITY.json').write_bytes(b'{"changed":true}\n')
                self.assertNotEqual(lineage.consumption_document(directory),original)

    def test_changed_generation_capacity_stops_before_entry_receipt(self):
        transport=types.ModuleType('english_v6_b24q1_source_transport')
        transport.verify_transport=mock.Mock()
        capacity=types.ModuleType('english_v6_b24q1_capacity')
        capacity.primary_generation_capacity=mock.Mock(side_effect=ValueError('fixture altered capacity'))
        identity={'test_only':'identity'};expected={'source_execution':identity}
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary)/'CONSUMPTION.json').write_text(json.dumps(expected))
            with mock.patch.dict(sys.modules,{transport.__name__:transport,capacity.__name__:capacity}), \
                 mock.patch.object(lineage,'execution_guard',return_value=identity), \
                 mock.patch.object(lineage,'consumption_document',return_value=expected), \
                 mock.patch.object(lineage,'read_plan',return_value={'primary':{'natural':{'fixture':True}}}), \
                 mock.patch.object(lineage,'uploaded_marker') as marker, \
                 mock.patch.object(lineage.qa,'record') as record:
                with self.assertRaisesRegex(ValueError,'fixture altered capacity'):
                    lineage.require_generation_permit(temporary,'invented-corpus','invented-parent')
                marker.assert_not_called();record.assert_not_called()
                self.assertFalse((Path(temporary)/'CONSUMPTION_UPLOAD.json').exists())

    def test_complete_capacity_receipt_survives_original_proof_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'authorization';output=Path(temporary)/'source'
            directory.mkdir();output.mkdir();(directory/'ORIGINAL_B24R1_STOP').mkdir()
            for name in (lineage.STOP_BINDING,lineage.STOP_REPORT,'CAPACITY.json'):
                (directory/name).write_bytes(b'exact fixture bytes')
            with mock.patch.object(lineage,'_original_retain_proof'),mock.patch.object(lineage,'saved_stop'), \
                 mock.patch.object(lineage.failure,'retain_transport'):
                lineage.retain_proof(directory,output)
            self.assertEqual((output/'CAPACITY.json').read_bytes(),(directory/'CAPACITY.json').read_bytes())
            self.assertTrue((output/'ORIGINAL_B24R1_STOP').is_dir())


if __name__=='__main__':
    unittest.main()
