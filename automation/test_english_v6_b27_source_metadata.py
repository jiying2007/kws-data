"""Original Q1 exclusion, immutable B27 population and zero-call metadata contracts."""
import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

PATH=Path(__file__).with_name('english_v6_b27_source_metadata.py')
SPEC=importlib.util.spec_from_file_location('b27_metadata_test_target',PATH)
metadata=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(metadata)


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=metadata.inputs()
        cls.base,cls.old,cls.history,cls.exposure,cls.origin,cls.hashes=cls.inputs
        cls.documents={n:json.loads((metadata.PANELS/n).read_bytes()) for n in metadata.OUTPUT_NAMES}

    def test_exact_original_ancestry_and_all120_q1_and72_people_excluded(self):
        a=self.documents['ancestry.json']
        self.assertEqual(len(a['all_pcm_sha256']),3063)
        for key in ('all_pcm_sha256','all_source_groups','canonical_people','synthetic_excluded_people','natural_excluded_people'):
            self.assertLessEqual(set(self.history[key]),set(a[key]),key)
        self.assertLessEqual({r['pcm_sha256'] for r in self.exposure['rows']},set(a['all_pcm_sha256']))
        self.assertEqual(a['B24Q1_exposed_primary_people'],self.exposure['primary_people'])
        self.assertEqual(a['B24Q1_sealed_replication_people'],self.exposure['sealed_replication_people'])
        for key in ('synthetic_excluded_people','natural_excluded_people'):
            self.assertLessEqual(set(self.exposure['all_excluded_people']),set(a[key]))
        for key in ('prior_never_reclaimed_reserves','prior_explicit_and_conservative_roster_people'):
            self.assertEqual(a[key],self.history[key])
        for generation in ('B22','B23','B24','B24R1'):
            for kind in ('exposed_primary_people','sealed_replication_people'):
                self.assertEqual(a[generation+'_'+kind],self.history[generation+'_'+kind])

    def test_all72_new_people_disjoint_across_both_panels_and_types(self):
        assessment,ancestry=self.documents['assessment.json'],self.documents['ancestry.json']
        people=[]
        for role in ('primary','replication'):
            synthetic=[self.base.canonical(p) for p in assessment[role]['synthetic']['speaker_id_map'].values()]
            natural=assessment[role]['natural']['speakers']
            self.assertEqual((len(synthetic),len(natural)),(24,12))
            self.assertFalse(set(synthetic)&set(ancestry['synthetic_excluded_people']))
            self.assertFalse(set(natural)&set(ancestry['natural_excluded_people']))
            self.assertFalse(assessment[role]['natural']['opened'])
            people.extend(synthetic+natural)
        self.assertEqual(len(people),len(set(people)))
        self.assertEqual(len(people),72)
        self.assertFalse(set(people)&set(self.exposure['all_excluded_people']))
        self.assertEqual(assessment['metadata_selection']['eligible_synthetic_voices'],112)
        self.assertEqual(assessment['metadata_selection']['eligible_natural_readers'],42)

    def test_original_quality_observer_policy_and_whole_source_format_unchanged(self):
        a=self.documents['assessment.json']
        for key in ('source_gate','gains','quality_criteria','quality_assessment_sha256','observer_scientific_contract',
                    'source_orchestration_contract','maximum_native_observer_calls','maximum_primary_KWS_forwards',
                    'maximum_public_Piper_synthesize_calls','maximum_natural_original_decodes','source_format','target'):
            self.assertEqual(a[key],self.old[key],key)
        for role in ('primary','replication'):
            rows=self.documents[role+'.json']['rows'];natural=self.documents['natural-'+role+'.json']
            self.assertEqual(len(rows),96)
            self.assertEqual({(r['cell_id'],r['length_scale']) for r in rows},
                             {('mero-fast',1.4),('mero-slow',1.8),('question',1.),('request',1.08)})
            self.assertTrue(all(r['observed_class'] is None and r['training_allowed'] is False for r in rows))
            self.assertEqual(len(natural['roster']),24)
            self.assertEqual(len(set(natural['selected_utterance_ids'])),24)
            self.assertTrue(all(sum(r['speaker']==p for r in natural['roster'])==2 for p in natural['speakers']))

    def test_preoptimizer_contract_and_b24_control_with_all_stages_closed(self):
        a=self.documents['assessment.json']
        self.assertTrue(a['freeze_before_optimizer_and_new_source'])
        self.assertFalse(a['freeze_after_training_before_new_source'])
        self.assertEqual(a['training']['design_sha256'],metadata.DESIGN_SHA)
        self.assertEqual(a['training']['initial_pt_sha256'],metadata.INITIAL_PT)
        self.assertEqual(a['training']['historical_control_pt_sha256'],metadata.CONTROL_PT)
        self.assertEqual(a['frozen_model_pt_sha256']['control'],metadata.CONTROL_PT)
        self.assertIsNone(a['frozen_model_pt_sha256']['challenger'])
        self.assertEqual(a['training']['new_challenger_updates'],192)
        self.assertEqual(a['training']['new_model_forwards'],1847)
        self.assertEqual(a['maximum_total_KWS_model_and_backend_calls'],4867)
        self.assertEqual(a['B26_opener'],'only_after_actual_complete_B24R1_primary_NO_PROMOTION')
        self.assertEqual(a['opening_condition'],'only_after_actual_complete_B24Q1_primary_NO_PROMOTION')
        for key in ('execution_authorized','source_execution_authorized','primary_scoring_authorized','replica_opening_authorized',
                    'engineering_authorized','optimizer_allowed','promotion_allowed','automatic_R2_allowed',
                    'automatic_next_assessment_allowed','replication_opened','old_B24Q1_replication_opened'):
            self.assertFalse(a[key],key)

    def test_metadata_revalidation_without_network_scientific_import_or_subprocess(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','numpy','torchaudio','onnxruntime','transformers','piper','soundfile','requests','httpx'}:
                raise AssertionError('no scientific/network import: '+name)
            return original(name,*args,**kwargs)
        with mock.patch('builtins.__import__',side_effect=guarded),mock.patch.object(socket,'socket',side_effect=AssertionError('no network')), \
             mock.patch.object(subprocess,'run',side_effect=AssertionError('no subprocess')):
            result=metadata.validate()
        binding=json.loads((metadata.HERE/'english_v6_b27_assessment_binding.json').read_bytes())
        self.assertEqual(result['hashes'],binding['files'])
        self.assertEqual(result['original_B24Q1_source_exclusion'],binding['original_B24Q1_source_exclusion'])
        self.assertEqual(result['status'],'PASS_METADATA_ONLY')
        self.assertFalse(result['execution_authorized'])

    def test_changed_frozen_population_or_execution_permission_rejected(self):
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temporary:
            folder=Path(temporary)
            for name,value in self.documents.items():(folder/name).write_bytes(metadata.encoded(value))
            for change in ('roster','execution','replica'):
                altered=copy.deepcopy(self.documents['assessment.json'])
                if change=='roster':altered['primary']['natural']['speakers'][0]=self.exposure['primary_people'][0]
                elif change=='execution':altered['optimizer_allowed']=True
                else:altered['replication']['natural']['opened']=True
                (folder/'assessment.json').write_bytes(metadata.encoded(altered))
                with mock.patch.object(metadata,'PANELS',folder),mock.patch.object(metadata,'build_documents',return_value=(self.documents,dict(self.hashes))), \
                     self.assertRaisesRegex(ValueError,'complete deterministic frozen B27 metadata'):
                    metadata.validate()

    def test_attempted_second_freeze_rejected_before_ranking(self):
        with mock.patch.object(metadata,'build_documents',side_effect=AssertionError('no redraw')):
            with self.assertRaisesRegex(ValueError,'freeze only once'):
                metadata.freeze()

    def test_source_pin_or_failure_misclassification_rejected_before_read(self):
        api=self.origin['source_api_identity']
        bad=copy.deepcopy(metadata.SOURCE_PIN);bad['id']+=1
        with mock.patch.object(Path,'open',side_effect=AssertionError('no read before identity')):
            with self.assertRaises(ValueError):metadata.read_original_metadata('/missing',api,bad)
            changed=copy.deepcopy(api);changed['run']['conclusion']='failure'
            with self.assertRaises(ValueError):metadata.read_original_metadata('/missing',changed,metadata.SOURCE_PIN)
            changed=copy.deepcopy(api);changed['commit']['parents'][0]['sha']='f'*40
            with self.assertRaises(ValueError):metadata.read_original_metadata('/missing',changed,metadata.SOURCE_PIN)

    def test_metadata_archive_reader_only_interprets_two_whitelisted_members(self):
        history=(metadata.ROOT/'automation/english_v6_b24q1_panels/ancestry.json').read_bytes()
        manifest=dict(schema='english-v6-b24q1-fresh-combined-source-v1',panel='primary',selected_original_slots=120,
            rows=self.exposure['rows'],source_execution=self.exposure['source_execution'],assessment_sha256=metadata.PREVIOUS_ASSESSMENT_SHA,
            original_training_assessment_sha256=metadata.TRAINING_ASSESSMENT_SHA,
            source_ancestry_sha256=metadata.SOURCE_MEMBERS['HISTORY.json'],history_receipt_sha256=metadata.SOURCE_MEMBERS['HISTORY.json'],
            new_kws_forwards=0,replication_opened=False)
        raw=metadata.encoded(manifest);members={'MANIFEST.json':metadata.digest(raw),'HISTORY.json':metadata.digest(history)}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fixture.zip'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('MANIFEST.json',raw);z.writestr('HISTORY.json',history);z.writestr('forbidden.wav',b'not audio')
            pin={**metadata.SOURCE_PIN,'bytes':path.stat().st_size,'sha256':metadata.digest(path.read_bytes())}
            api=copy.deepcopy(self.origin['source_api_identity']);api['artifact']['size_in_bytes']=pin['bytes'];api['artifact']['digest']='sha256:'+pin['sha256']
            read=zipfile.ZipFile.read;seen=[]
            def permitted(z,name,*args,**kwargs):
                seen.append(name)
                self.assertIn(name,('MANIFEST.json','HISTORY.json'))
                return read(z,name,*args,**kwargs)
            with mock.patch.object(metadata,'SOURCE_PIN',pin),mock.patch.object(metadata,'SOURCE_MEMBERS',members), \
                 mock.patch.object(zipfile.ZipFile,'read',permitted):
                value=metadata.read_original_metadata(path,api,pin)
            self.assertEqual(seen,['MANIFEST.json','HISTORY.json'])
            self.assertEqual(len(value['b24q1-source-exposure.json']['rows']),120)
            self.assertFalse(value['b24q1-source-origin.json']['quality_results_read'])
            self.assertFalse(value['b24q1-source-origin.json']['scientific_execution_authorized'])

    def test_original_metadata_digest_and_all120_whitelist_cannot_be_forged(self):
        with self.assertRaisesRegex(ValueError,'exact original Q1 MANIFEST and HISTORY bytes'):
            metadata.derive_original_metadata({'MANIFEST.json':b'{}','HISTORY.json':b'{}'},self.origin['source_api_identity'],metadata.SOURCE_PIN)
        changed=list(self.inputs);changed[3]=copy.deepcopy(self.exposure);changed[3]['rows'].pop()
        with mock.patch.object(metadata,'inputs',return_value=changed),self.assertRaisesRegex(ValueError,'complete original Q1 population'):
            metadata.build_documents()

    def test_insufficient_fresh_population_stops_without_salt_or_sampler_retry(self):
        with mock.patch.object(metadata,'inputs',return_value=self.inputs), \
             mock.patch.object(self.base,'catalog_rows',return_value=(set(),{})), \
             mock.patch.object(self.base,'rank',side_effect=AssertionError('no ranking an insufficient pool')), \
             self.assertRaisesRegex(ValueError,'two full independent panels'):
            metadata.build_documents()

    def test_changed_prior_hypothesis_or_design_dependency_rejected_before_selection(self):
        read=Path.read_bytes
        def altered(path):
            raw=read(path)
            if path==metadata.ROOT/metadata.DESIGN_PATH:
                value=json.loads(raw);value['scientific_change']['after']=.32
                return metadata.encoded(value)
            return raw
        with mock.patch.object(Path,'read_bytes',altered),self.assertRaisesRegex(ValueError,'fixed B27 metadata input'):
            metadata.inputs()


if __name__=='__main__':
    unittest.main()
