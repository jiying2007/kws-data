"""Standard-library adapter checks; no model or actual TRAIN artifact is read."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b23_saved_analyzer_fixture', HERE / 'analyze_saved_b23_train.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def test_complete_original_analyzer_is_pinned_and_only_explicit_sites_change(self):
        original = adapter.ORIGINAL.read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), adapter.ORIGINAL_SHA)
        changed = adapter.prepared_source()
        for before, after in reversed(adapter.EDITS):
            self.assertEqual(changed.count(after), 1)
            changed = changed.replace(after, before)
        changed = changed.replace('B23_TRAIN_ONLY_', 'B22_TRAIN_ONLY_')
        self.assertEqual(changed, original)

    def test_ctc_and_all_greedy_functions_remain_identical(self):
        def functions(text):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions(adapter.prepared_source()), functions(adapter.ORIGINAL.read_text()))

    def test_missing_or_changed_original_never_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'invented.zip'
            source.write_bytes(b'fixture is not a model artifact')
            for digest in (None, 'a' * 64):
                with self.assertRaises(ValueError): adapter.analyze(source, digest, root / 'output')
                self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
