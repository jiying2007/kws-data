"""Bounded sealed-replica fixtures: no source/model load or external calls."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b22_replication_source_ancestry as lineage
import english_v6_b22_replication_source as source
import english_v6_b22_replication_source_observe as observe
import english_v6_b22_replication_natural_sources as natural
import english_v6_b22_replication_source_producer as producer
import english_v6_b22_source as original
import english_v6_b22_source_observe as original_observe


class SealedSources(unittest.TestCase):
    def sealed_fixture(self):
        value=json.loads(lineage.OPENING.read_bytes())
        value['status']='SEALED_UNTIL_ORIGINAL_PRIMARY_PASS'
        for key in ('primary_quality_lock','quality_readback_sha256','pair_pin','model_pt_sha256'):
            value[key]=None
        return value

    def test_opening_null_and_blocks_before_any_external_operation(self):
        # The actual config may already be bound when this pre-execution fixture
        # runs. Test a sealed config without mutating or weakening that binding.
        fixture=self.sealed_fixture()
        with patch.object(lineage,'OPENING') as opening, \
             patch.object(lineage,'gh',side_effect=AssertionError('no network')), \
             patch.object(lineage.reader,'fetch_quality_original',side_effect=AssertionError('no fetch')):
            opening.read_bytes.return_value=json.dumps(fixture).encode()
            self.assertEqual(lineage.read_opening(False)['status'],'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS')
            with tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp)/'unopened'
                with self.assertRaisesRegex(ValueError,'sealed'):
                    lineage.authorize(dest,'a'*40)
                self.assertFalse(dest.exists())

    def test_partial_lock_cannot_open_the_panel(self):
        value=self.sealed_fixture()
        value['primary_quality_lock']={}
        with patch.object(lineage,'OPENING') as path:
            path.read_bytes.return_value=json.dumps(value).encode()
            with self.assertRaises(ValueError):lineage.read_opening(False)

    def test_same_frozen72_disjoint_people_and_only_replica_requests(self):
        plan=lineage.read_plan()
        people=[]
        for panel in ('primary','replication'):
            group={v.removeprefix('p') for v in plan[panel]['synthetic']['speaker_id_map'].values()}
            group.update(plan[panel]['natural']['speakers'])
            self.assertEqual(len(group),36);people.append(group)
        self.assertFalse(people[0]&people[1])
        rows=source.source_plan()['rows']
        self.assertEqual(len(rows),96)
        self.assertTrue(all(r['id'].startswith('b22-replication-') for r in rows))
        with self.assertRaises(ValueError):source.source_plan('primary')
        self.assertEqual(natural.plan()['roster'],plan['replication']['natural']['roster'])

    def test_numerical_admission_and_native_observer_bodies_are_reused(self):
        self.assertIs(source.screen_document.__code__,original.screen_document.__code__)
        self.assertIs(observe.observe.__code__,original_observe.observe.__code__)
        self.assertIs(observe.load_observations.__code__,original_observe.load_observations.__code__)
        self.assertIsNot(source.screen_document.__globals__,original.screen_document.__globals__)
        self.assertEqual(source.screen_document.__globals__['SCHEMA'],source.SCHEMA)
        self.assertEqual(original.SCHEMA,'english-v6-b22-fresh-combined-source-v1')

    def test_replica_observer_uses_fixed_noncontiguous_voice_roster(self):
        contract=observe.source_check()
        self.assertEqual(contract['source_indices'],lineage.read_plan()['replication']['synthetic']['speaker_indices'])
        self.assertEqual(contract['schema'],'english-v6-b22-replication-five-observer-contract-v1')

    def test_template_remains_inert_except_two_reviewed_parent_substitutions(self):
        template=(HERE.parent/producer.TEMPLATE).read_bytes()
        actual=(HERE.parent/lineage.WORKFLOW).read_bytes()
        self.assertEqual(template.count(producer.TOKEN),2)
        if actual!=template:
            import re
            parents=re.findall(rb"(?:PREPARED_PARENT: ')([0-9a-f]{40})'",actual)
            self.assertEqual(len(parents),1)
            self.assertEqual(template.replace(producer.TOKEN,parents[0]),actual)
        text=template.decode()
        self.assertIn('include-hidden-files: true',text)
        self.assertIn('b22-replica-source-producer/repository/automation/english_v6_b22_replication_source.py',text)
        self.assertNotIn('workflow_dispatch',text)

    def test_source_producer_keeps_primary_and_excludes_future_replica_bindings(self):
        self.assertFalse('automation/english_v6_b22_primary_inputs.json'.startswith(producer.FUTURE_PREFIXES))
        for name in ('english_v6_b22_replication_inputs.json','english_v6_b22_replication_producer.json','english_v6_b22_replication_eval.py'):
            self.assertTrue(('automation/'+name).startswith(producer.FUTURE_PREFIXES))
        self.assertEqual(producer.TOKEN,b'__B22_REVIEWED_REPLICATION_SOURCE_PARENT__')

    def test_source_producer_inherits_complete_text_runtime_helper_authority(self):
        pins=producer.dynamic.required_paths()
        self.assertEqual(len(pins),19)
        self.assertTrue({producer.dynamic.CAMPAIGN,producer.dynamic.RUNNER} <= set(pins))
        root=HERE.parent
        original=json.loads((HERE/'english_v6_b22_b21_producer_manifest.json').read_bytes())
        tracked=set(original['files']) | set(pins) | producer.EXTRA
        for prefix in ('automation','.github/workflows','research/english_v6_b22_training_20261124/initial_failure'):
            tracked.update(path.relative_to(root).as_posix() for path in (root/prefix).rglob('*') if path.is_file())
        binding=json.loads((HERE/'english_v6_b22_source_inputs.json').read_bytes())
        if binding['bound']:
            tracked.update(json.loads((HERE/'english_v6_b22_source_training_producer.json').read_bytes())['files'])
        self.assertTrue(set(pins) <= producer.select_paths(tracked))
        with self.assertRaisesRegex(ValueError,'remain tracked'):
            producer.select_paths(tracked-{producer.dynamic.CAMPAIGN})


if __name__=='__main__':unittest.main()
