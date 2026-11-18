"""No-model regression for one-shot source-bound five-native fresh24 intake."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("fresh24_review_test_"+name,HERE/(name+".py"))
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x
review=load("english_fresh24_five_native_once")
combine=load("english_fresh24_combine_once")


class Fresh24SourceReview(unittest.TestCase):
    def test_original_source_policy_frozen_and_never_recycled(self):
        result=review.source_check()
        self.assertEqual(result["source_rows"],168)
        self.assertEqual(result["source_run"],44289263966)
        self.assertFalse(result["training_allowed"])
        self.assertEqual(combine.source_check()["policy_git_blob_sha1"],
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertEqual(combine.PLAN_ORIGINAL_BLOB_SHA1,
                         "dbe911fbcf8a697492ad29c7ca11858e707a72ff")
        self.assertEqual(review.REFERENCE["id"],14419694053)
        self.assertEqual(review.SOURCE,{"id":14422271973,
            "name":"english-fresh24-clone168-44289263966",
            "size":4505217,
            "digest":"sha256:26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583"})
        self.assertEqual(result["status"],"SOURCE_PINNED")
        self.assertEqual(review.original_identity(),{
            "run_id":"44289263966","commit":review.SOURCE_COMMIT,
            "before":"dfbb95f003b3845439ccbe5e5eb4cb80e58e83f8"})
        self.assertEqual(len(review.PROVIDERS),5)
        self.assertEqual(review.source_worker.source_check()["planned_clone_calls"],168)
    def test_unpinned_source_may_never_call_any_native_observer(self):
        pinned=review.SOURCE
        review.SOURCE={"id":0,"name":pinned["name"],
                       "size":0,"digest":""}
        try:
            self.assertEqual(review.source_check()["status"],
                             "BLOCKED_UNPINNED_CLONE_ARTIFACT")
            with self.assertRaisesRegex(ValueError,
                    "NO_NUMERICAL_EXECUTION_WITHOUT_VERIFIED_CLONE_ARCHIVE"):
                review.observe("qwen","/bad","/bad","/bad")
            with self.assertRaisesRegex(ValueError,
                    "NO_NUMERICAL_EXECUTION_WITHOUT_VERIFIED_CLONE_ARCHIVE"):
                review.guard()
        finally:review.SOURCE=pinned
    def test_first_public_main_only_when_really_pinned(self):
        existing=review.SOURCE
        review.SOURCE={"id":9001,"name":existing["name"],
                      "size":168000,"digest":"sha256:"+"1"*64}
        try:
            env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":review.REPO,
                 "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
                 "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"443001122",
                 "GITHUB_SHA":"a"*40}
            event={"before":review.BEFORE,"after":"a"*40,
                "ref":"refs/heads/main",
                "repository":{"full_name":review.REPO,"private":False},
                "forced":False,"deleted":False}
            self.assertEqual(review.guard(env,event,review.BEFORE)["source_run"],
                             review.SOURCE_RUN)
            env["GITHUB_RUN_ATTEMPT"]="2"
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY"):
                review.guard(env,event,review.BEFORE)
            env["GITHUB_RUN_ATTEMPT"]="1"
            event["repository"]["private"]=True
            with self.assertRaisesRegex(ValueError,"ONE_EXACT_MAIN_SUCCESSOR_REVIEW"):
                review.guard(env,event,review.BEFORE)
            event["repository"]["private"]=False
            with self.assertRaisesRegex(ValueError,"EXACT_FIRST_MAIN_PARENT"):
                review.guard(env,event,"b"*40)
        finally:review.SOURCE=existing
    def test_five_model_absence_never_becomes_negative_or_quality_pass(self):
        names=review.source_worker.neutral_requests(review.source_worker.source_plan())
        refs={r["id"]:{**r,"wav_sha256":"a"*64,
                "pcm_sha256":"b"*64} for r in names}
        rows=review.source_worker.clone_requests(refs)
        for i,row in enumerate(rows):
            row.update(samples=16000,positive_duration_pass=True,
                wav_sha256=f"{i:064x}",pcm_sha256=f"{i+250:064x}")
        missing={r["id"]:{k:None for k in review.PROVIDERS[:3]}
                 for r in rows}
        missing_phones={r["id"]:{k:None for k in review.PROVIDERS[3:]}
                       for r in rows}
        result=combine.evaluate(rows,missing,missing_phones)
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertEqual(result["qualified_groups_by_role"],
                         {"train":0,"development":0,"calibration":0,
                          "locked-test":0})
        self.assertEqual(result["tier_counts"],{"unknown":168})
        self.assertEqual(result["new_model_weights"],False)
        self.assertTrue(all(r["training_allowed"] is False for r in result["rows"]))
    def test_synthetic_fixture_cannot_be_misreported_as_real_admission(self):
        names=review.source_worker.neutral_requests(review.source_worker.source_plan())
        refs={r["id"]:{**r,"wav_sha256":"a"*64,
                "pcm_sha256":"b"*64} for r in names}
        rows=review.source_worker.clone_requests(refs)
        for i,row in enumerate(rows):
            row.update(samples=16000,positive_duration_pass=True,
                wav_sha256=f"{i:064x}",pcm_sha256=f"{i+250:064x}")
        texts={r["id"]:{} for r in rows}
        phones={r["id"]:{} for r in rows}
        def completely_artificial_consensus(row,*ignored):
            if row["requested_class"] in (0,1,2):
                return {"tier":"silver","class":row["requested_class"],
                        "reason":"UNIT_TEST_ONLY_NOT_AN_ACOUSTIC_RECEIPT",
                        "margins":{},"training_allowed":False}
            return {"tier":"unknown","class":None,
                    "reason":"CONTRAST_DIAGNOSTIC","margins":{},
                    "training_allowed":False}
        with patch.object(combine.policy,"decide",completely_artificial_consensus):
            result=combine.evaluate(rows,texts,phones)
        self.assertEqual(result["qualified_groups_by_role"],
                         {"train":12,"development":4,"calibration":4,
                          "locked-test":4})
        self.assertTrue(result["machine_source_qualification_passed"])
        self.assertTrue(all(r["training_allowed"] is False for r in result["rows"]))
        self.assertEqual(result["new_optimizer_updates"],0)
        self.assertFalse(result["locked_kws_evaluation_consumed"])
        self.assertTrue(result["source_groups_are_verified_distinct_synthetic_presets_not_humans"])
        bad={r["source_group_id"] for r in rows if r["role"]=="train"}
        broken=sorted(bad)[:7]
        def incomplete(row,*ignored):
            if row["source_group_id"] in broken and row["requested_class"]==2:
                return {"tier":"unknown","class":None,"reason":"MISSING_POSITIVE",
                        "margins":{},"training_allowed":False}
            return completely_artificial_consensus(row)
        with patch.object(combine.policy,"decide",incomplete):
            rejected=combine.evaluate(rows,texts,phones)
        self.assertEqual(rejected["qualified_groups_by_role"]["train"],5)
        self.assertFalse(rejected["training_snapshot_allowed"])

if __name__=="__main__":
    unittest.main()
