"""B15 prospectively frozen synthetic voices contract tests (no synthesis or inference)."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b15_synthetic_source as source

class FrozenSourceTest(unittest.TestCase):
    def test_two_disjoint_frozen_source_roles(self):
        primary=source.source_plan("primary");secondary=source.source_plan("replication")
        self.assertEqual(primary["config_sha256"],"2efdc6d7f954588b8180132cbd9b8001933fdd00932c92bc92fd0d2028a9eb3d")
        self.assertEqual(len(primary["rows"]),len(secondary["rows"]),48)
        for panel,lo in ((primary,646),(secondary,658)):
            self.assertEqual({r["speaker_index"] for r in panel["rows"]},set(range(lo,lo+12)))
            self.assertEqual(len({r["source_group_id"] for r in panel["rows"]}),12)
            self.assertEqual(sum(r["requested_class"]==2 for r in panel["rows"]),24)
            self.assertEqual(sum(r["requested_class"]==0 for r in panel["rows"]),24)
            self.assertFalse(panel["training_allowed"])
            self.assertEqual(panel["max_source_calls"],48)
        self.assertFalse({r["source_speaker_id"] for r in primary["rows"]} &
                         {r["source_speaker_id"] for r in secondary["rows"]})
    def test_native_ipa_and_provenance(self):
        for panel in ("primary","replication"):
            doc=source.source_plan(panel)
            self.assertEqual(doc["immutable_assessment_sha256"],source.STUDY_SHA)
            self.assertTrue(all(r["native_input"]=="[[heɪ mˈɛɹoʊ]]" for r in doc["rows"]
                                if r["requested_class"]==2))
            self.assertTrue(all(r["native_input"]=="[[wˈʌt tˈaɪm ɪz ɪt]]" for r in doc["rows"]
                                if r["cell_id"]=="question"))
            self.assertEqual(len({r["id"] for r in doc["rows"]}),48)
        self.assertEqual(source.B14_PRIMARY_ID,14789951699)
    def test_unsafe_roster_mutation_denied(self):
        import unittest.mock as mock
        original=source.qa.sha
        with mock.patch.object(source.qa,"sha",side_effect=lambda p: "0"*64):
            with self.assertRaises(ValueError):
                source.source_plan("primary")

if __name__=="__main__":
    unittest.main()
