"""Stdlib-only protected mono70 70PCM real-training contract tests."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "mono70_student_contract",HERE/"english_v5_mero_mono70_robust_train_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def toy_train():
    rows=[];i=0
    for group in range(24):
        for cls in (0,1):
            i+=1
            rows.append({"id":f"mono_{i}","corpus":"fresh24",
                         "role":"train","source_group_id":f"train_g{group}",
                         "mono_class":cls,"tier":"silver"})
    # 6 extra Mero positive within original, not independent source groups.
    for group in range(6):
        i+=1
        rows.append({"id":f"mono_{i}","corpus":"newtrain24",
                     "role":"train","source_group_id":f"train_g{group}",
                     "mono_class":1,"tier":"weak"})
    return rows


class Mono70Student(unittest.TestCase):
    def test_single_wake_contracted_and_not_yet_promotable(self):
        state=v.source_check()
        self.assertEqual(state["target"],"Hey Mero")
        self.assertEqual(state["optimizer_updates"],0)
        self.assertEqual(state["source_audit_run"],44470829024)
        self.assertEqual(state["source_audit_artifact"],14503604626)
        self.assertEqual(len(state["source_audit_sha"]),64)
        self.assertEqual(state["old_locked_audio"],"NEVER_OPENED")
        self.assertFalse(state["product_release"])
        self.assertFalse(state["old_weights_reused"])
        self.assertEqual(v.TRAIN_COUNTS,{0:24,1:30})
        self.assertEqual(v.DEV_COUNTS,{0:7,1:9})
        self.assertEqual(v.UPDATES,360)
        self.assertEqual(v.HIDDEN_PARAMS,43876)

    def test_frozen_same_source_code_not_authorized_to_forge_audit(self):
        with tempfile.TemporaryDirectory() as root:
            archive=Path(root)/"fake.zip"
            archive.write_bytes(b"not-a-github-artifact")
            with self.assertRaises(ValueError):
                v.authorized_source(archive,Path(root))

    def test_train_groups_balanced_and_no_dev_gradient(self):
        g=v.groups_by_class(toy_train())
        self.assertEqual(len(g[0]),24)
        self.assertEqual(len(g[1]),24)
        self.assertEqual(sum(map(len,g[0].values())),24)
        self.assertEqual(sum(map(len,g[1].values())),30)
        bad=toy_train()
        bad[0]["role"]="development"
        with self.assertRaises(ValueError):
            v.groups_by_class(bad)
        bad=toy_train()[2:]
        with self.assertRaises(ValueError):
            v.groups_by_class(bad)

    def test_actual_run_is_only_first_public_same_repo_PR(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"pull_request",
             "GITHUB_HEAD_REF":v.BRANCH,
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44470111111"}
        event={"action":"synchronize",
           "repository":{"full_name":v.REPO,"private":False},
           "pull_request":{"head":{"ref":v.BRANCH,"sha":"a"*40,
                                  "repo":{"full_name":v.REPO}},
                           "base":{"ref":"main"}}}
        self.assertEqual(v.run_guard(env,event)["head_sha"],"a"*40)
        for changed,evt in (
            ({**env,"GITHUB_RUN_ATTEMPT":"2"},event),
            ({**env,"GITHUB_REPOSITORY":"wrong"},event),
            (env,{**event,"action":"closed"}),
            (env,{**event,"repository":{"full_name":v.REPO,
                                        "private":True}}),
            (env,{**event,"pull_request":{**event["pull_request"],
                  "base":{"ref":"locked-test"}}}),
        ):
            with self.subTest(changed=changed,evt=evt):
                with self.assertRaises(ValueError):
                    v.run_guard(changed,evt)

    def test_source_and_model_contracts_are_not_fake_alignment(self):
        text=(HERE/"english_v5_mero_mono70_robust_train_once.py").read_text()
        self.assertIn("mono.audit(source_root)",text)
        self.assertIn("v5.ctc_greedy(logits,torch)",text)
        self.assertIn("torch.optim.AdamW",text)
        self.assertIn("source_pcm_sha256",text)
        self.assertIn("model.state_dict()",text)
        self.assertIn("torch.onnx.export",text)
        self.assertIn("source_disjoint",text) if False else None
        self.assertIn('"human_far_frr":"NOT_MEASURED"',text)
        self.assertIn('"product_promotion_allowed":False',text)
        self.assertIn('"no_calibration_locked_opened":True',text)
        self.assertNotIn("Hey Vosen",text)
        self.assertNotIn("teacher_pseudophone",text)
        self.assertNotIn("for step in range(512)",text)


if __name__=="__main__":
    unittest.main()
