"""No-network tests for the exact #210 source artifact pins."""
import importlib.util
import json
from pathlib import Path
import unittest
p=Path(__file__).with_name("english_hifi_phone_recovery.py")
spec=importlib.util.spec_from_file_location("hifi_original_artifacts",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FrozenAssets(unittest.TestCase):
    def test_exact_two_artifact_ids(self):
        self.assertEqual(set(m.FILES),{"custom","clone"})
        self.assertEqual(len({x["id"] for x in m.FILES.values()}),2)
    def test_good_original_identity(self):
        records={}
        for name,x in m.FILES.items():
            records[name]={**x,"size_in_bytes":x["bytes"],"expired":False,
                           "workflow_run":{"id":m.RUN,"head_sha":m.HEAD}}
        got=m.validate_all(records)
        self.assertEqual(len(got),2)
        self.assertTrue(all(x["independent_training_group_count"]==0 for x in got.values()))
    def test_reject_different_source_run_digest_or_expiry(self):
        good={}
        for name,x in m.FILES.items():
            good[name]={**x,"size_in_bytes":x["bytes"],"expired":False,
                        "workflow_run":{"id":m.RUN,"head_sha":m.HEAD}}
        for field,value in (("digest","sha256:"+"0"*64),("expired",True),
                            ("size_in_bytes",1),("workflow_run",{"id":m.RUN+1,"head_sha":m.HEAD})):
            trial=json.loads(json.dumps(good))
            trial["clone"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                m.validate_all(trial)
    def test_unregistered_data_source_rejected(self):
        good={}
        for name,x in m.FILES.items():
            good[name]={**x,"size_in_bytes":x["bytes"],"expired":False,
                        "workflow_run":{"id":m.RUN,"head_sha":m.HEAD}}
        good["new"]=good["clone"]
        with self.assertRaises(ValueError):
            m.validate_all(good)

if __name__=="__main__":unittest.main()
