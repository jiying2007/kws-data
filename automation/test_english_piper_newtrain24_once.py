"""Pure, no-model checks for new original Piper voice-group source contract."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("piper_new_source",HERE/"english_piper_newtrain24_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FreshTrainDevSource(unittest.TestCase):
    def test_contract_preapproved_ipa_and_minimum(self):
        p=m.protocol()
        r=m.source_check()
        self.assertEqual(r["voice_groups"],24)
        self.assertEqual(r["planned_tts_calls"],168)
        self.assertFalse(r["training_allowed"])
        self.assertEqual([x["role"] for x in p["groups"]],
            ["train"]*18+["development"]*6)
        self.assertEqual(p["minimum_qualified_groups"],{"train":6,
            "development":2,"calibration":2,"locked-test":2})
        self.assertEqual([x["class"] for x in p["cells"]],[2,2,1,1,0,None,None])
        self.assertEqual([x["raw_ipa"] for x in p["cells"][:4]],
            ["heɪ mˈɛɹoʊ","heɪ mˈɛɹoʊ","heɪ vˈoʊsən","heɪ vˈoʊsən"])
        self.assertNotIn("Meero",str(p["cells"][:4]))
        self.assertTrue(all(x["speaker_index"]>=424 for x in p["groups"]))
    def test_source_mapping_rejects_prior_numeric_speakers(self):
        speakers={f"p{9000+i}":i for i in range(904)}
        roster=m.native_speakers({"num_speakers":904,"speaker_id_map":speakers})
        self.assertEqual(len(roster),24)
        self.assertEqual(m.planned(roster)[0]["role"],"train")
        self.assertEqual(len(m.planned(roster)),168)
        self.assertTrue(all(x["role"] in ("train","development") for x in m.planned(roster)))
        self.assertTrue(all(x["observed_class"] is None and
                            x["training_allowed"] is False
                            for x in m.planned(roster)))
        speakers["p422"]=424
        speakers.pop("p9424")
        with self.assertRaisesRegex(ValueError,
             "NO_REPLAY_OF_OLD_LIBRI_SOURCE_SPEAKER_IDS"):
            m.native_speakers({"num_speakers":904,"speaker_id_map":speakers})
    def test_attempt_1_public_main_parent_only(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"443999000",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["before"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaises(ValueError):m.guard(env,event,"b"*40)
    def test_no_unreviewed_wav_is_a_training_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                m.source_readback(tmp)
        p=m.protocol()
        self.assertFalse(p["speaker_data_human_ancestry_independently_verified"])
        self.assertFalse(p["training_allowed"])
        self.assertFalse(p["new_audio_request_is_label"])
        self.assertFalse(p["old_wavs_reused"])
if __name__=="__main__":unittest.main()
