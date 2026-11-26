"""Invented original-byte and refusal fixtures; no real evidence is qualified."""
from __future__ import annotations

import base64
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bind_b28_source as source_binder
import bind_b28_quality as binder
import english_v6_b28_profile as profile
import english_v6_b28_primary_eval as primary
import english_v6_b28_replication_eval as replica
import english_v6_b28_quality_readback as quality
import english_v6_b28_source_producer as source_producer
import english_v6_b28_training_readback as training_reader
import test_english_v6_b28_training_readback as train_fixtures

HERE = Path(__file__).resolve().parent
ASSESSMENT = 'a' * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(profile.canonical(value))


def training_originals(root):
    """71 explicitly fictional files, with real byte checks and no audio."""
    origin = root / 'training-origin'
    names = {'inputs/proof/training-origin/old-' + str(i) + '.zip' for i in range(41)}
    names |= {'inputs/proof/primary-quality/' + name for name in profile.QUALITY_FILES}
    names |= {'inputs/metadata/train360/' + name for name in profile.CATALOG_FILES}
    expected = {}
    for name in sorted(names):
        raw = (b'{"INVENTED":true}\n' if name.endswith('.json') else b'PK\x03\x04INVENTED:') + (b'' if name.endswith('.json') else name.encode())
        path = origin / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        expected[name] = dict(bytes=len(raw), sha256=profile.sha(raw))
    lock = train_fixtures.lock()
    for role, pin in lock['training']['artifacts'].items():
        raw = b'PK\x03\x04INVENTED TRAIN:' + role.encode()
        path = origin / 'trained' / ('TRAIN_' + role + '.zip')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        pin.update(bytes=len(raw), sha256=profile.sha(raw))
    (origin / 'trained/TRAIN_LIVE.json').write_bytes(b'{"INVENTED-original-whole-API":true}\n')
    report = dict(schema=training_reader.REPORT_SCHEMA, original_lock_sha256=profile.sha(profile.canonical(lock)),
        result_sha256=lock['training']['result_sha256'], result=dict(schema=source_binder.train.SCHEMA,
            status=source_binder.train.RESULT_STATUS, assessment_sha256=ASSESSMENT,
            model_pt_sha256=copy.deepcopy(lock['training']['model_pt_sha256'])),
        saved_training_observations=dict(natural_objective_returns_verified=384))
    report.update(dict.fromkeys(('all_original_producer_git_verified','all_original_inputs_verified',
        'all_saved_training_arrays_verified','natural_no_event_terms_recomputed',
        'complete_original_B27_NO_PROMOTION_verified','original_B24_control_unchanged'),True))
    report.update(dict.fromkeys(('new_model_calls','new_source_calls','new_optimizer_updates'),0))
    report.update(dict.fromkeys(('promotion_allowed','product_released','gradients_recomputed','optimizer_updates_replayed'),False))
    return lock, expected, report


@contextmanager
def training_scope(expected, report):
    with ExitStack() as stack:
        for name in ('design', 'opening', 'assessment_binding'):
            stack.enter_context(patch.object(source_binder.train, name, return_value={}))
        stack.enter_context(patch.object(training_reader, 'validate_lock', side_effect=lambda lock: lock['training']))
        stack.enter_context(patch.object(profile, 'expected_training_inputs', return_value=expected))
        stack.enter_context(patch.object(profile, 'profile', return_value={'assessment_sha256':ASSESSMENT}))
        reader = stack.enter_context(patch.object(profile, '_training_readback', return_value=profile.canonical(report)))
        yield reader


def api_bundle(root, owner=primary):
    bundle = dict(schema='english-v6-b28-' + owner.PANEL + '-original-quality-binding-inputs-v1',groups={},archives={})
    groups, baselines, role_pins = {}, {}, {}
    for offset, group in enumerate(binder.GROUPS, 1):
        run_id, head = 100 + offset, str(offset) * 40
        run = dict(id=run_id,head_sha=head,run_attempt=1,status='completed',conclusion='success',event='push',
            repository=dict(full_name=owner.REPO,private=False),head_repository=dict(full_name=owner.REPO,private=False),
            head_branch=owner.TRAIN_BRANCH if group=='train' else owner.SOURCE_BRANCH if group=='source' else 'INVENTED-baseline',
            path=owner.TRAIN_WORKFLOW if group=='train' else owner.SOURCE_WORKFLOW if group=='source' else 'INVENTED.yml')
        tree = dict(sha='5'*40,truncated=False,tree=[])
        commit = dict(sha=head,commit=dict(tree=dict(sha=tree['sha'])),parents=[dict(sha='6'*40)],files=[dict(filename=run['path'])])
        groups[group] = dict(run=run,commit=commit,artifacts=dict(total_count=0,artifacts=[]))
        if group in ('train','source'): groups[group]['tree']=tree
    for index, role in enumerate(binder.ROLES, 1):
        group = 'train' if role in ('pair','train_producer') else role if role in ('B0','B9') else 'source'
        run = groups[group]['run']
        raw = b'PK\x03\x04INVENTED ARCHIVE:' + role.encode()
        member = 'archives/' + role + '.zip'
        path = root / member; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
        bundle['archives'][role] = member
        pin = dict(id=1000+index,run_id=run['id'],run_head=run['head_sha'],bytes=len(raw),sha256=profile.sha(raw))
        item = dict(id=pin['id'],name='INVENTED-'+role if role in ('B0','B9') else binder.artifact_name(role,run['id'],owner.PANEL),
            size_in_bytes=len(raw),digest='sha256:'+pin['sha256'],expired=False,
            workflow_run=dict(id=run['id'],head_sha=run['head_sha']))
        groups[group]['artifacts']['artifacts'].append(item)
        role_pins[role] = pin
        if role in ('B0','B9'): baselines[role] = pin
    for name, group in groups.items():
        group['artifacts']['total_count'] = len(group['artifacts']['artifacts'])
        bundle['groups'][name] = {}
        for key, value in group.items():
            member='api/'+name+'/'+key+'.json';write(root/member,value);bundle['groups'][name][key]=member
    return bundle,groups,baselines,role_pins


class BindingContracts(unittest.TestCase):
    def test_registrations_are_exactly_inert_or_authentically_bound(self):
        for owner in (primary,replica):
            self.assertTrue(all(v is None for v in binder.inert_inputs(owner.PANEL)['model_pt_sha256'].values()))
            current=profile.document(owner.SPEC.read_bytes())
            if current['scoring_allowed'] is False:
                self.assertEqual(current,binder.inert_inputs(owner.PANEL))
                self.assertEqual(profile.document(owner.PRODUCER.read_bytes()),binder.inert_producer(owner.PANEL))
                with self.assertRaises(ValueError): owner.protocol()
            else:
                self.assertEqual(owner.protocol(),current)
                self.assertTrue(profile.document(owner.PRODUCER.read_bytes())['frozen'])
            bound=profile.binding(owner.PANEL,False)
            if bound['status']==profile.inert_binding(owner.PANEL)['status']:
                self.assertEqual(bound,profile.inert_binding(owner.PANEL))
            else:
                self.assertEqual(profile.validate_binding(bound,owner.PANEL),bound)

    def test_real_unbound_source_and_quality_binders_never_start_readers_or_write_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();proof=root/'proof';(proof/'training-origin').mkdir(parents=True)
            bundle=root/'bundle.json';write(bundle,{'INVENTED':True})
            with (patch.object(profile,'_training_readback') as training,patch.object(quality,'verify_quality_original') as scored,
                 patch.object(binder.subprocess,'check_output') as child):
                with self.assertRaises(ValueError): source_binder.bind('primary',proof,{},root/'source-review')
                with self.assertRaises(ValueError): binder.bind('primary',bundle,proof,root/'quality-review')
                training.assert_not_called();scored.assert_not_called();child.assert_not_called()
            self.assertFalse((root/'source-review').exists());self.assertFalse((root/'quality-review').exists())

    def test_exact71_byte_inventory_precedes_training_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();lock,expected,report=training_originals(root)
            with training_scope(expected,report) as reader:
                files,returned,_=source_binder.verify_training(root,lock)
                self.assertEqual(len(files),71);self.assertEqual(returned,report);reader.assert_called_once()
                reader.reset_mock()
                extra=root/'training-origin/unregistered.zip';extra.write_bytes(b'INVENTED')
                with self.assertRaises(ValueError): source_binder.verify_training(root,lock)
                reader.assert_not_called();extra.unlink()
                (root/'training-origin/inputs/metadata/train360/api/tree.json').unlink()
                with self.assertRaises(ValueError): source_binder.verify_training(root,lock)
                reader.assert_not_called()

    def test_changed_original_or_old_candidate_and_incomplete_objective_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();lock,expected,report=training_originals(root)
            for changed in ('candidate','objective','failure'):
                bad=copy.deepcopy(report)
                if changed=='candidate':bad['result']['model_pt_sha256']['challenger']=profile.B27_PT
                elif changed=='objective':bad['saved_training_observations']['natural_objective_returns_verified']=383
                else:bad['complete_original_B27_NO_PROMOTION_verified']=False
                with training_scope(expected,bad):
                    with self.assertRaises(ValueError):source_binder.verify_training(root,lock)
            with training_scope(expected,report) as reader:
                changed_path=root/'training-origin/trained/TRAIN_models.zip';changed_path.write_bytes(b'REPLACEMENT')
                with self.assertRaises(ValueError):source_binder.verify_training(root,lock)
                reader.assert_not_called()

    def test_training_report_does_not_hide_mutation_during_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();lock,expected,report=training_originals(root)
            with training_scope(expected,report) as reader:
                def mutate(*args):
                    (root/'training-origin/trained/TRAIN_LIVE.json').write_bytes(b'{}')
                    return profile.canonical(report)
                reader.side_effect=mutate
                with self.assertRaises(ValueError):source_binder.verify_training(root,lock)

    def test_source_review_never_retargets_existing_destination_or_borrows_primary_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();proof=root/'proof';(proof/'training-origin').mkdir(parents=True)
            output=root/'existing';output.mkdir()
            with patch.object(source_binder,'verify_training') as reader:
                with self.assertRaises(ValueError):source_binder.bind('primary',proof,{},output)
                (proof/'primary-quality').mkdir()
                with self.assertRaises(ValueError):source_binder.bind('primary',proof,{},root/'new')
                reader.assert_not_called()

    def test_transport_retains_exact_whole_api_bytes_and_no_unregistered_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'run.json').write_bytes(b'{ "INVENTED": true }\n')
            (root/'original.zip').write_bytes(b'PK\x03\x04INVENTED')
            files=profile.inventory(root)
            result=source_binder.transport(root,files,{'original.zip':7})
            self.assertEqual(base64.b64decode(result['run.json']['original_api_base64']),b'{ "INVENTED": true }\n')
            with self.assertRaises(ValueError):source_binder.transport(root,files,{'original.zip':7,'extra.zip':8})
            (root/'run.json').write_bytes(b'{"INVENTED":true}\n')
            with self.assertRaises(ValueError):source_binder.transport(root,files,{'original.zip':7})

    def test_training_transport_joins_old59_catalog7_and_new5_without_reserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();lock,_,_=training_originals(root)
            origin=root/'training-origin';files=profile.inventory(origin)
            def locations(prefix):
                result={}
                for index,(name,pin) in enumerate(sorted(files.items()),100):
                    if name.startswith(prefix):
                        raw=(origin/name).read_bytes()
                        result[name.removeprefix(prefix)]=(dict(original_api_base64=base64.b64encode(raw).decode())
                            if name.endswith('.json') else dict(artifact_id=index))
                return result
            old=dict(transport=locations('inputs/proof/training-origin/'))
            opening=dict(original_primary_transport=locations('inputs/proof/primary-quality/'))
            catalog=dict(transport=locations('inputs/metadata/train360/'))
            with (patch.object(training_reader,'validate_lock',side_effect=lambda value:value['training']),
                  patch.object(source_binder.train.profile,'binding',return_value=old),
                  patch.object(source_binder.train,'opening',return_value=opening),
                  patch.object(source_binder.metadata,'catalog_registration',return_value=catalog)):
                actual=source_binder.training_transport(origin,files,lock)
                self.assertEqual(len(actual),71)
                self.assertEqual(base64.b64decode(actual['inputs/metadata/train360/api/tree.json']['original_api_base64']),
                                 (origin/'inputs/metadata/train360/api/tree.json').read_bytes())
                catalog['transport']['api/tree.json']['original_api_base64']=base64.b64encode(b'{ "INVENTED":true}\n').decode()
                with self.assertRaises(ValueError):source_binder.training_transport(origin,files,lock)

    def test_replica_cannot_open_from_old_failure_or_different_same_named_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();proof=root/'proof';(proof/'training-origin').mkdir(parents=True);(proof/'primary-quality').mkdir()
            report=dict(result=dict(model_pt_sha256=dict(control=profile.CONTROL_PT,challenger='7'*64),assessment_sha256=ASSESSMENT))
            good=dict(panel='primary',panel_gate='PASS',replication_open_allowed=True,
                model_pt_sha256=report['result']['model_pt_sha256'],assessment_sha256=ASSESSMENT,
                original_training_report_sha256=profile.sha(profile.canonical(report)))
            for field,value in (('panel_gate','NO_PROMOTION'),('model_pt_sha256',dict(control=profile.CONTROL_PT,challenger=profile.B27_PT)),
                                ('assessment_sha256','8'*64),('original_training_report_sha256','9'*64)):
                bad=dict(good,**{field:value})
                with (patch.object(source_binder,'verify_training',return_value=({},report,'a'*64)),
                     patch.object(profile,'binding',side_effect=lambda panel,require_bound=True:profile.inert_binding(panel)),
                     patch.object(source_binder,'training_transport',return_value={}),
                     patch.object(quality,'verify_quality_original',return_value=bad),patch.object(quality,'validate_report',return_value=bad)):
                    with self.assertRaises(ValueError):source_binder.bind('replication',proof,{},root/'review',dict(panel='primary'))
                self.assertFalse((root/'review').exists())

    def test_four_api_groups_bind_all_twelve_distinct_original_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();bundle,groups,baselines,pins=api_bundle(root)
            with patch.object(primary,'BASELINES',baselines):
                actual,selected,evidence=binder.metadata(bundle,root,primary,'primary')
            self.assertEqual(actual,groups);self.assertEqual(selected,pins);self.assertEqual(set(evidence),set(binder.ROLES))
            self.assertEqual(len({pin['id'] for pin in selected.values()}),12)

    def test_missing_or_truncated_original_git_tree_rejects_before_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();bundle,groups,baselines,_=api_bundle(root)
            with patch.object(primary,'BASELINES',baselines):
                bad=copy.deepcopy(bundle);del bad['groups']['source']['tree']
                with self.assertRaises(ValueError):binder.metadata(bad,root,primary,'primary')
                tree=dict(groups['source']['tree'],truncated=True);write(root/bundle['groups']['source']['tree'],tree)
                with self.assertRaises(ValueError):binder.metadata(bundle,root,primary,'primary')

    def test_old_run_wrong_attempt_and_private_source_never_qualify(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();bundle,groups,baselines,_=api_bundle(root)
            with patch.object(primary,'BASELINES',baselines):
                for field,value in (('head_branch','research/en-v6-b27-primary-sources-once-20261125'),('run_attempt',2),
                                    ('repository',dict(full_name=primary.REPO,private=True))):
                    run=copy.deepcopy(groups['source']['run']);run[field]=value;write(root/bundle['groups']['source']['run'],run)
                    with self.assertRaises(ValueError):binder.metadata(bundle,root,primary,'primary')

    def test_missing_native_or_duplicate_id_and_baseline_substitution_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();bundle,groups,baselines,_=api_bundle(root)
            with patch.object(primary,'BASELINES',baselines):
                path=root/bundle['groups']['source']['artifacts']
                bad=copy.deepcopy(groups['source']['artifacts']);bad['artifacts'].pop();bad['total_count']-=1;write(path,bad)
                with self.assertRaises(ValueError):binder.metadata(bundle,root,primary,'primary')
                bad=copy.deepcopy(groups['source']['artifacts']);bad['artifacts'][1]['id']=bad['artifacts'][0]['id'];write(path,bad)
                with self.assertRaises(ValueError):binder.metadata(bundle,root,primary,'primary')
                write(path,groups['source']['artifacts'])
                bad=copy.deepcopy(groups['B0']['artifacts']);bad['artifacts'][0]['digest']='sha256:'+'f'*64
                write(root/bundle['groups']['B0']['artifacts'],bad)
                with self.assertRaises(ValueError):binder.metadata(bundle,root,primary,'primary')

    def test_bundle_relative_paths_cannot_escape_or_follow_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'real').write_bytes(b'INVENTED');(root/'link').symlink_to(root/'real')
            for name in ('../real','/etc/hostname','link','a\\b','a/../real'):
                with self.assertRaises(ValueError):binder.member_path(root,name)
            self.assertEqual(binder.member_path(root,'real'),root/'real')

    def test_quality_registration_joins_actual_train_artifacts_and_new_objective_contract(self):
        import english_v6_b28_source as source
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();_,_,_,pins=api_bundle(root)
            models=dict(B0=primary.B0_PT,B9=primary.B9_PT,control=profile.CONTROL_PT,challenger='d'*64)
            training=dict(result=dict(model_pt_sha256=models,assessment_sha256=ASSESSMENT))
            actual=dict(run_id=pins['pair']['run_id'],run_head=pins['pair']['run_head'],model_pt_sha256=models,
                artifacts={role:{key:pins[target][key] for key in ('id','bytes','sha256')}
                    for role,target in (('models','pair'),('producer','train_producer'))})
            bound=dict(training_lock=dict(training=actual),training_report_sha256='e'*64,primary_quality_lock=None)
            with (patch.object(profile,'binding',return_value=bound),patch.object(source,'source_policy',return_value={'INVENTED':True}),
                  patch.object(primary,'EVALUATION_ASSESSMENT',ASSESSMENT)):
                value=binder.registration(primary,'primary',pins,training)
                self.assertTrue(value['complete_original_B27_failure_required'])
                self.assertNotIn('complete_original_B24Q1_failure_required',value)
                self.assertEqual(value['original_training_report_sha256'],'e'*64)
                for key,expected in primary.objective_contract().items():self.assertEqual(value[key],expected)
                self.assertFalse(value['scoring_allowed']);self.assertIsNone(value['producer_manifest_sha256'])
                bad=copy.deepcopy(pins);bad['pair']['run_id']+=1
                with self.assertRaises(ValueError):binder.registration(primary,'primary',bad,training)

    def test_complete_actual_producer_layout_reopens_every_captured_dynamic_resource(self):
        repository=HERE.parent
        for owner in (primary,replica):
            with self.subTest(panel=owner.PANEL),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary).resolve()
                source_code=source_producer.runtime_paths(repository,owner.PANEL)
                code=owner.static_closure(repository/'automation',owner.PRODUCER_ROOTS)|source_code
                required={'automation/english_v6_b14_panels/assessment.json','automation/english_v6_b28_panels/assessment.json',
                    'automation/english_v6_b28_natural_event_design.json','automation/english_v6_b28_execution_profile.json',
                    'automation/english_v6_b28_opening.json','automation/english_v6_b28_assessment_binding.json',
                    'automation/english_v6_b28_'+owner.PANEL+'_proof_binding.json',
                    'automation/'+owner.WORKFLOW_TEMPLATE.name,owner.SOURCE_WORKFLOW,owner.TRAIN_WORKFLOW}
                if owner.PANEL=='replication':required|={'automation/english_v6_b28_primary_inputs.json',
                    'automation/english_v6_b28_primary_producer.json','automation/english_v6_b28_primary_proof_binding.json'}
                for name in code|required:
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes((repository/name).read_bytes() if (repository/name).is_file() else b'{"INVENTED-unbound-assessment":true}\n')
                source_names=source_code|required-{owner.TRAIN_WORKFLOW}
                def pins(names):return {name:dict(bytes=(root/name).stat().st_size,sha256=profile.sha((root/name).read_bytes())) for name in names}
                snapshots=dict(source=dict(files=pins(source_names),runtime_workflow=owner.SOURCE_WORKFLOW),
                    train=dict(files=pins({'automation/english_v6_b14_decision.py',owner.TRAIN_WORKFLOW}),runtime_workflow=owner.TRAIN_WORKFLOW))
                spec={}
                if owner.PANEL=='replication':spec['primary_quality_lock']=dict(inputs_sha256=profile.sha(primary.SPEC.read_bytes()),
                    producer_manifest_sha256=profile.sha(primary.PRODUCER.read_bytes()))
                doc,_,resources=binder.producer_document(owner,owner.PANEL,snapshots,spec,root)
                self.assertIn('automation/english_v6_b28_opening.json',resources)
                self.assertIn('automation/english_v6_b28_assessment_binding.json',resources)
                parent='a'*40;runtime=root/owner.WORKFLOW;runtime.parent.mkdir(parents=True,exist_ok=True)
                runtime.write_bytes(owner.WORKFLOW_TEMPLATE.read_bytes().replace(owner.PARENT_TOKEN.encode(),parent.encode()))
                destination=root/'automation'/owner.PRODUCER.name
                def store(value):
                    destination.write_bytes(profile.canonical(value));spec['producer_manifest_sha256']=profile.sha(destination.read_bytes())
                store(doc)
                actual,workflow=owner.producer_contract(spec,parent,root)
                self.assertEqual(actual,doc);self.assertEqual(workflow,runtime.read_bytes())
                bad=copy.deepcopy(doc);del bad['resources']['automation/english_v6_b28_opening.json'];store(bad)
                with self.assertRaises(ValueError):owner.producer_contract(spec,parent,root)
                store(doc);member=sorted(doc['source']['files'])[0]
                (root/member).write_bytes((root/member).read_bytes()+b'\n# changed original fixture\n')
                with self.assertRaises(ValueError):owner.producer_contract(spec,parent,root)

    def test_quality_workflows_are_inert_and_consume_before_any_runtime_install(self):
        for owner in (primary,replica):
            raw=owner.WORKFLOW_TEMPLATE.read_text();runtime=(HERE.parent/owner.WORKFLOW).read_text()
            self.assertEqual(raw.count(owner.PARENT_TOKEN),2)
            if owner.PARENT_TOKEN in runtime:self.assertEqual(raw,runtime)
            else:
                parent=re.search(r"PREPARED_PARENT: '([0-9a-f]{40})'",runtime).group(1)
                self.assertEqual(runtime,raw.replace(owner.PARENT_TOKEN,parent))
            self.assertIn('github.run_attempt == 1',raw);self.assertIn('cancel-in-progress: false',raw)
            self.assertLess(raw.index(' preflight '),raw.index('name: '+owner.MARKER+'\n'))
            self.assertLess(raw.index('name: '+owner.MARKER+'\n'),raw.index('pip install'))
            self.assertLess(raw.index('pip install'),raw.index(' score '))
            self.assertNotIn('workflow_dispatch',raw)
            self.assertIn('english_v6_b28_original_proof.py '+owner.PANEL,raw)
            self.assertIn('include-hidden-files: true',raw)


class FreshProtocolContracts(unittest.TestCase):
    def prepare(self,root,body='return json.loads(SPEC.read_bytes())'):
        directory=root/'automation';directory.mkdir()
        profile_code='import hashlib,json\ndef sha(raw):return hashlib.sha256(raw).hexdigest()\ndef document(raw):return json.loads(raw)\n'
        driver="from pathlib import Path\nimport json\nHERE=Path(__file__).resolve().parent\nSPEC=HERE/'english_v6_b28_primary_inputs.json'\nPRODUCER=HERE/'english_v6_b28_primary_producer.json'\nCANDIDATE_PT=json.loads(SPEC.read_bytes())['model_pt_sha256']['challenger']\ndef protocol():\n    "+body+'\n'
        (directory/'english_v6_b28_profile.py').write_text(profile_code)
        (directory/'english_v6_b28_primary_eval.py').write_text(driver)
        (directory/'english_v6_b28_primary_workflow_template.yml').write_text('INVENTED TEMPLATE\n')
        (directory/'resource.json').write_bytes(b'{"INVENTED":true}')
        owner=SimpleNamespace(SPEC=directory/'english_v6_b28_primary_inputs.json',PRODUCER=directory/'english_v6_b28_primary_producer.json',
            WORKFLOW_TEMPLATE=directory/'english_v6_b28_primary_workflow_template.yml',CANDIDATE_PT=None)
        code={str(path.relative_to(root)):profile.sha(path.read_bytes()) for path in directory.glob('*.py')}
        producer=dict(primary=dict(files=code,workflow_template_sha256=profile.sha(owner.WORKFLOW_TEMPLATE.read_bytes())),
            resources={'automation/resource.json':profile.sha((directory/'resource.json').read_bytes())})
        spec=dict(model_pt_sha256=dict(challenger='d'*64),producer_manifest_sha256=profile.sha(profile.canonical(producer)))
        return owner,spec,producer

    def test_fresh_import_reads_candidate_from_proposed_bytes_without_cached_global_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();owner,spec,producer=self.prepare(root)
            result=binder.validate_fresh_draft(root,'primary',owner,spec,producer)
            self.assertEqual(result['candidate_pt_sha256'],'d'*64);self.assertIsNone(owner.CANDIDATE_PT)
            self.assertFalse(owner.SPEC.exists());self.assertFalse(owner.PRODUCER.exists())

    def test_fresh_worker_blocks_science_network_subprocess_write_and_external_fallback(self):
        for body in ('import torch','import socket; socket.socket()','import subprocess; subprocess.run(["true"])',
                     "(HERE/'changed').write_text('forbidden')","Path('/etc/hostname').read_bytes()"):
            with self.subTest(body=body),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary).resolve();owner,spec,producer=self.prepare(root,body)
                with self.assertRaises(subprocess.CalledProcessError):binder.validate_fresh_draft(root,'primary',owner,spec,producer)
                self.assertFalse((root/'automation/changed').exists())

    def test_changed_draft_dependency_rejects_before_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();owner,spec,producer=self.prepare(root)
            (root/'automation/resource.json').write_bytes(b'{}')
            with patch.object(binder.subprocess,'check_output') as child:
                with self.assertRaises(ValueError):binder.validate_fresh_draft(root,'primary',owner,spec,producer)
                child.assert_not_called()


if __name__=='__main__':
    unittest.main()
