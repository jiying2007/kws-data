"""Exact captured-source entry and prospective activation contracts, no audio."""
from __future__ import annotations
import ast
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24q1_source_producer as producer
import english_v6_b24q1_source_ancestry as lineage
import english_v6_b24q1_source_transport as transport
import english_v6_b22_source_transport as old_transport
import english_v6_b24_source_ancestry as original_train


class SourceEntry(unittest.TestCase):
    def test_no_original_train_assessment_or_corpus_transport_rebound(self):
        for name in ('training_binding', 'training_binding_sha', 'verify_training', 'validate_training_live'):
            self.assertIs(getattr(lineage, name), getattr(original_train, name), name)
        self.assertNotEqual(lineage.ASSESSMENT_SHA, original_train.ASSESSMENT_SHA)
        self.assertEqual(lineage.ASSESSMENT_SHA, lineage.failure.ASSESSMENT_SHA)
        self.assertIs(transport.download.__code__, old_transport.download.__code__)
        self.assertEqual(lineage.training_binding()['training']['model_pt_sha256'], lineage.metadata.FROZEN_MODELS)

    def test_new_workflow_has_only_one_reviewed_workflow_child_and_durable_markers(self):
        raw = (producer.ROOT / producer.TEMPLATE).read_bytes()
        text = raw.decode()
        self.assertEqual(raw.count(producer.TOKEN), 2)
        self.assertNotIn('workflow_dispatch', text)
        self.assertIn('research/en-v6-b24q1-primary-sources-once-20261125', text)
        self.assertIn('github.run_attempt == 1', text)
        self.assertIn('cancel-in-progress: false', text)
        self.assertIn('provider: [qwen, parakeet, xlsr, lv60, whisper]', text)
        self.assertIn('check-live-metadata', text)
        self.assertEqual(text.count('include-hidden-files: true'), text.count('retention-days: 90'))
        self.assertLess(text.index('source.py" authorize'), text.index('source.py" consume'))
        self.assertLess(text.index('name: english-v6-b24q1-primary-source-consumed'), text.index('source.py" generate'))
        self.assertLess(text.index('name: english-v6-b24q1-primary-observer-${{ matrix.provider }}-consumed'),
                        text.index('Install one pinned native observer runtime'))
        self.assertIn('prior-source/B24R1_SOURCE_STOP_BINDING.json', text)
        self.assertIn('prior-source/B24R1_SOURCE_STOP_REPORT.json', text)
        for line in text.splitlines():
            if 'source.py' in line and ('python' in line or 'bin/python' in line):
                self.assertIn('$RUNNER_TEMP/b24q1-source-producer/repository/automation/english_v6_b24q1_source.py', line)

    def test_captured_closure_includes_all_approved_metadata_and_immutable_stop_reader(self):
        tracked = subprocess.check_output(['git', 'ls-files'], cwd=producer.ROOT, text=True).splitlines()
        selected = producer.select_paths(tracked)
        self.assertTrue(set(lineage.metadata.validate()['hashes']) <= selected)
        self.assertTrue({lineage.WORKFLOW, producer.TEMPLATE,
            'automation/english_v6_b24q1_capacity.py', 'automation/english_v6_b24q1_quarantine.py',
            'automation/english_v6_b24q1_r1_source_failure.py',
            'automation/english_v6_b24r1_source_stop_readback.py',
            'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json'} <= selected)
        self.assertFalse(any(name.startswith(producer.FUTURE_PREFIXES) for name in selected))
        # A later scoring-only preparation must leave the original source closure stable.
        future = ['automation/english_v6_b24q1_primary_future.json',
                  'automation/english_v6_b24q1_replication_future.py',
                  '.github/workflows/english-v6-b24q1-engineering-future.yml']
        self.assertEqual(producer.select_paths([*tracked, *future]), selected)

    def test_saved_stop_spawns_only_the_exact_original_reader(self):
        tree = ast.parse(Path(lineage.__file__).read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'saved_stop')
        source = ast.unparse(node)
        self.assertIn('READER_PATH', source)
        self.assertIn('ORIGINAL_B24R1_STOP', source)
        self.assertNotIn('Path(failure.__file__)', source)


if __name__ == '__main__':
    unittest.main()
