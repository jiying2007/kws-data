"""No-model source, SHA, source-role and replay tests for 12×5 clone cohort."""
import importlib.util
from pathlib import Path
import hashlib
from array import array
import json
import math
import tempfile
import unittest
import wave

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("cohort_test",HERE/"english_clone_cohort12_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.c

def reference_fixture(root):
    root=Path(root)
    root.mkdir(parents=True);(root/"wav").mkdir()
    rows=[]
    for idx,name in enumerate(sorted(m.ROLES)):
        signal=array("h",(int(4000*math.sin(k/9)+(idx*20)) for k in range(44000)))
        p=root/"wav"/("synthetic-"+name+".wav")
        with wave.open(str(p),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(signal.tobytes())
        stats=c.inspect_wav(p)
        rows.append({"id":"synthetic-"+name,"preset_id":name,"role":m.ROLES[name],
          "source_group_id":"kokoro-v1_0-preset:"+name,
          "reference_text":"Hello there. Good morning.",
          "reference_kind":"SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY",
          "wav_sha256":stats["wav_sha256"],"joined_pcm_sha256":stats["pcm_sha256"],
          "pcm_samples":stats["samples"],
          "training_allowed":False,"no_human_identity_gold":True,
          "independent_human_source_claim":False})
    c.record(root/"MANIFEST.json",{"schema":m.SOURCE_REF_SCHEMA,
        "record_kind":"SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY",
        "reference_audio_count":12,"roles":m.ROLE_FLOOR,
        "human_gold":False,"training_allowed":False,
        "source":{"run":44218734553,"artifact_id":14389757284,
                  "archive_sha256":"712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da",
                  "head_sha":"7e9797789cba7d1476d02f14e21af91dc3cd0712"},
        "rows":rows})
    return root

class CohortContract(unittest.TestCase):
    def test_exact_five_cells_and_twelve_source_roles(self):
        r=m.source_contract()
        self.assertEqual((r["donor_groups"],r["planned_new_tts_calls"]),(12,60))
        self.assertFalse(r["training_allowed"])
        self.assertNotIn("Meero",str(m.CELLS))
    def test_original_artifact_pin_and_bad_digest_stop(self):
        good={**m.ARTIFACT,"expired":False,
              "workflow_run":{"id":m.SOURCE_RUN,"head_sha":m.SOURCE_COMMIT}}
        self.assertEqual(m.source_metadata(good)["source_run"],m.SOURCE_RUN)
        for field,value in [("digest","sha256:"+"0"*64),
                            ("size_in_bytes",1),("expired",True),
                            ("workflow_run",{"id":m.SOURCE_RUN+1,"head_sha":m.SOURCE_COMMIT})]:
            bad={**good,field:value}
            with self.assertRaises(ValueError):m.source_metadata(bad)
    def test_first_main_identity_only_once(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44299990",
             "GITHUB_SHA":"a"*40}
        event={"before":m.PREDECESSOR,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":m.REPO,"private":False}}
        self.assertEqual(m.guard(env,event,m.PREDECESSOR)["source_previous"],m.PREDECESSOR)
        for x in ["2","3"]:
            changed=dict(env,GITHUB_RUN_ATTEMPT=x)
            with self.assertRaises(ValueError):m.guard(changed,event,m.PREDECESSOR)
        event["before"]="b"*40
        with self.assertRaises(ValueError):m.guard(env,event,m.PREDECESSOR)
    def test_all_reference_audio_fixed_and_requested_targets_not_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=reference_fixture(Path(temp)/"source")
            refs=m.reference_rows(root)
            rows=m.planned(refs)
            self.assertEqual(len(rows),60)
            self.assertEqual(len({x["id"] for x in rows}),60)
            self.assertTrue(all(x["observed_class"] is None and
                                x["training_allowed"] is False for x in rows))
            self.assertEqual({x["source_group_id"] for x in rows
                              if x["role"]=="locked-test"},
                              {"kokoro-v1_0-preset:af_nova",
                               "kokoro-v1_0-preset:am_onyx"})
            self.assertTrue(all(x["clone_reference_text"]==
                                "Hello there. Good morning." for x in rows))
            manifest=json.loads((root/"MANIFEST.json").read_text())
            manifest["rows"][0]["role"]="train"
            (root/"MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):m.reference_rows(root)
    def test_refuse_old_source_training_label_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=reference_fixture(Path(temp)/"source")
            manifest=json.loads((root/"MANIFEST.json").read_text())
            manifest["training_allowed"]=True
            (root/"MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):m.reference_rows(root)

if __name__=="__main__":unittest.main()
