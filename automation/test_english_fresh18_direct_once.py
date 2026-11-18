"""No-model prospective source/negative control tests for 18 fresh preset voices."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fresh18_direct",HERE/"english_fresh18_direct_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Direct18Contracts(unittest.TestCase):
    def test_unexposed_native_preset_roster_and_source_roles(self):
        actual=m.source_check()
        self.assertEqual(actual["planned_tts_forwards"],126)
        self.assertEqual(actual["roles"],{"train":10,"development":4,"locked-test":4})
        self.assertFalse(actual["training_allowed"])
        plan=m.plan()
        self.assertEqual(len(plan["groups"]),18)
        self.assertEqual(len({g["source_group_id"] for g in plan["groups"]}),18)
        self.assertTrue({x["preset_id"] for x in plan["groups"]}.isdisjoint(
            set(plan["original_source_excluded"])))
        self.assertEqual(plan["minimum_full_cumulative_6_2_2_2"],
                         {"train":6,"development":2,"calibration":2,"locked-test":2})
        self.assertTrue(all(row["asset_sha256"] in {x["asset_sha256"]
                        for x in plan["groups"]} for row in plan["groups"]))
        self.assertEqual([g["preset_id"] for g in plan["groups"]],
                         sorted((g["preset_id"] for g in plan["groups"]),
                                key=lambda name:(m.crc131(name),name)))
    def test_target_canonical_phones_are_not_request_labels(self):
        p=m.plan()
        self.assertEqual(len(p["cells"]),7)
        self.assertEqual([x["requested_class"] for x in p["cells"]],
                         [2,2,1,1,0,None,None])
        self.assertEqual(p["cells"][0]["raw_phonemes"],"hˈA mˈɛɹO.")
        self.assertEqual(p["cells"][2]["raw_phonemes"],"hˈA vˈOsən.")
        self.assertTrue(all("Meero" not in x["render_text"]
             for x in p["cells"] if x["requested_class"] in (1,2)))
        data=m.rows()
        self.assertEqual(len(data),126)
        self.assertEqual(len({x["id"] for x in data}),126)
        self.assertEqual(len([x for x in data if x["requested_class"] is None]),36)
        self.assertTrue(all(x["training_allowed"] is False and
              x["observed_class"] is None and x["human_gold"] is False
              for x in data))
        self.assertEqual({x["source_group_id"] for x in data},
                         {g["source_group_id"] for g in p["groups"]})
    def test_public_main_exact_successor_one_attempt_only(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"443888001",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["before"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY"):
            m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"ONE_ACTUAL_PREDECLARED_MAIN_SUCCESSOR"):
            m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_PARENT_NO_REPLAY"):
            m.guard(env,event,"b"*40)
    def test_existing_full_ctc_rule_never_rewritten(self):
        original=m.plan()
        self.assertEqual(original["original_immutable_rule_git_blob_sha1"],
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertEqual(original["immutable_prior_quality_report"]["qualification_by_role"],
                         {"train":4,"development":1,"calibration":3,"locked-test":0})
        self.assertEqual(m.source_check()["status"],"FROZEN_SOURCE_ONLY")
    def test_no_prior_wavs_or_transcripts_are_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            fake=Path(root)/"unstarted"
            with self.assertRaises(FileNotFoundError):
                m.readback(fake)
        self.assertTrue(m.plan()["no_reuse_of_old_wavs"])
        self.assertFalse(m.plan()["human_gold"])
        self.assertFalse(m.plan()["fresh_model_training_allowed"])
if __name__=="__main__":
    unittest.main()
