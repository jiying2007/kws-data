"""Private state migration and split stability; no real speaker material."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest

from kws_data_contract import source_registry as registry


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "sources.json"
        self.roster = ["speaker-" + str(i) for i in range(8)]
        self.state = registry.extend(registry.new_registry(), "test-data", self.roster)

    def test_initial_eight_and_repeated_subsets_keep_six_train_two_dev(self):
        self.assertEqual(sum(r["role"] == "development-only"
                             for r in self.state["entries"]), 2)
        self.assertEqual(self.state,
                         registry.extend(self.state, "test-data", self.roster[::-1]))
        extended = registry.extend(self.state, "test-data", ["another-speaker"])
        self.assertEqual(self.state["entries"], extended["entries"][:8])

    def test_save_is_private_and_existing_assignments_cannot_change(self):
        registry.save(self.path, self.state, None)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.state, registry.read_json(self.path))
        changed = copy.deepcopy(self.state)
        changed["entries"][0]["role"] = "development-only"
        with self.assertRaisesRegex(ValueError, "EXISTING_SOURCE_ASSIGNMENT_CHANGED"):
            registry.save(self.path, changed, self.state)
        with self.assertRaisesRegex(ValueError, "REGISTRY_CHANGED_CONCURRENTLY"):
            registry.save(self.path, self.state, None)

    def test_legacy_migration_preserves_old_roles_and_ids_without_retaining_key(self):
        key = b"r" * 40
        index = {"dataset_id": "test-data", "rows": []}
        legacy = {"schema": "kws-bulk-clone-sources-v1", "references": []}
        for n, speaker in enumerate(self.roster):
            token = hmac.new(key, ("test-data\0" + speaker).encode(),
                             hashlib.sha256).hexdigest()
            digest = f"{n:064x}"
            index["rows"].append({"speaker_key": speaker, "original_sha256": digest})
            legacy["references"].append({
                "reference_id": "ref-" + str(n), "dataset_id": "test-data",
                "speaker_group_id": "spk-" + token[:24],
                "source_family_id": "fam-" + token[:24],
                "role": "development-only" if int(token[:4], 16) % 10 < 2
                        else "train-candidate", "parent_source_sha256": digest})
        before = copy.deepcopy(legacy)
        migrated = registry.migrate_legacy(index, legacy, key)
        self.assertEqual(before, legacy)
        for entry, old in zip(migrated["entries"], legacy["references"]):
            for field in ("speaker_group_id", "source_family_id", "role"):
                self.assertEqual(old[field], entry[field])
        self.assertNotIn(key.decode(), json.dumps(migrated))
        # A new public HMAC alias/key is not an input to lookup or extension.
        registry.save(self.path, migrated, None)
        self.assertEqual(migrated, registry.extend(registry.read_json(self.path),
                                                   "test-data", self.roster))
        with self.assertRaisesRegex(ValueError, "LEGACY_REFERENCE_JOIN_INCOMPLETE"):
            registry.migrate_legacy(index, legacy, b"x" * 40)

    def test_cli_requires_explicit_initialization_and_dry_run_is_read_only(self):
        index = self.root / "index.json"
        index.write_text(json.dumps({"dataset_id": "test-data", "rows": [
            {"speaker_key": s} for s in self.roster]}))
        args = ["--registry", str(self.path), "--index", str(index)]
        with self.assertRaisesRegex(ValueError, "INITIALIZE_ONLY_NEW_REGISTRY"):
            registry.main(args)
        before = set(self.root.iterdir())
        self.assertEqual(registry.main(args + ["--initialize"]), 0)
        self.assertEqual(before, set(self.root.iterdir()))
        self.assertEqual(registry.main(args + ["--initialize", "--execute"]), 0)
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "INITIALIZE_ONLY_NEW_REGISTRY"):
            registry.main(args + ["--initialize", "--execute"])
        self.assertEqual(registry.main(args + ["--execute"]), 0)
        self.assertEqual(original, self.path.read_bytes())

    def test_registry_aliases_and_duplicate_rows_are_rejected(self):
        changed = copy.deepcopy(self.state)
        changed["entries"].append(copy.deepcopy(changed["entries"][0]))
        with self.assertRaisesRegex(ValueError, "REGISTRY_DUPLICATE_SPEAKER"):
            registry.validate(changed)
        self.path.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "REGISTRY_PATH_ALIAS"):
            registry.save(self.path, self.state, None)


if __name__ == "__main__":
    unittest.main()
