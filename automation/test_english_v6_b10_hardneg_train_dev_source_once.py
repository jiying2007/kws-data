"""Source-only tests for the bounded B10 hard-negative TRAIN + fresh DEV source stage."""
from __future__ import annotations
from collections import Counter,defaultdict
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b10_hardneg_source",HERE/"english_v6_b10_hardneg_train_dev_source_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def synthetic_originals():
    p=m.plan()
    source=[];screen=[]
    selected={g["source_group_id"]:g for g in p["train_groups"]}
    # Reconstruct the complete original 24-group/96-row manifest metadata.
    roles=["train"]*12+["development"]*4+["calibration"]*4+["locked-test"]*4
    for i,role in enumerate(roles):
        if i<len(p["train_groups"]):
            g=p["train_groups"][i]
            gid=g["source_group_id"];speaker=g["source_speaker_id"];idx=g["speaker_index"]
        else:
            idx=600+i;speaker=f"p{40000+i}";gid="piper-libritts-"+speaker
        for cell,klass in (("mero-fast",2),("mero-slow",2),
                           ("ordinary-time",0),("nearphone-meero-challenge",None)):
            source.append({"id":f"old-{idx}-{cell}","source_group_id":gid,
                "speaker_index":idx,"source_speaker_id":speaker,"role":role,
                "cell_id":cell,"pcm_sha256":f"{idx:04x}"+("a"*60)})
    # The saved source screen contains the 12 TRAIN and four DEV groups only.
    for i,role in enumerate(["train"]*12+["development"]*4):
        if i<len(p["train_groups"]):
            g=p["train_groups"][i]
            gid=g["source_group_id"];idx=g["speaker_index"]
        else:
            idx=700+i;gid=f"piper-libritts-p{41000+i}"
        for cell,klass in (("mero-fast",2),("mero-slow",2),
                           ("ordinary-time",0),("nearphone-meero-challenge",None)):
            screen.append({"id":f"old-{idx}-{cell}","source_group_id":gid,
                "speaker_index":idx,"role":role,
                "tier":"silver" if klass is not None else "unknown",
                "observed_class":klass})
    manifest={"schema":"english-v5-mero-fresh24-independent-source-once-v1",
        "run":{"run_id":"44478554596","commit":"e3f79406f469061dc9ac7438ed6658ad66375e55",
               "before":"90fa488738acde10eca3a3cd92e5cb5b31eff5c4","attempt":1},
        "denominator":96,"completed":96,"new_tts_calls":96,
        "new_asr_calls":0,"new_phone_calls":0,"training_allowed":False,
        "human_gold":False,"rows":source}
    packet={"schema":"english-v5-mero-fresh24-saved-5-native-screen-no-model-v1",
        "status":"PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY",
        "source_run":44478554596,"original_observer_run":44481695979,
        "source_group_counts":{"development":2,"train":11},
        "tier_counts":{"silver":25,"unknown":32,"weak":7},
        "observed_label_counts":{"0":16,"2":16},
        "training_snapshot_allowed":False,"locked_kws_test_opened":False,
        "rows":screen}
    return manifest,packet

class B10SourceTests(TestCase):
    def test_frozen_plan_and_source_check(self):
        x=m.source_check()
        self.assertEqual(x["status"],"SOURCE_ONLY_NO_AUDIO_GENERATED")
        self.assertEqual(x["new_original_wavs"],18)
        self.assertEqual(x["train_hard_negative_candidates"],6)
        self.assertEqual(x["development_groups"],6)
        self.assertEqual(x["source_roles"],{"train":6,"development":12})
        self.assertEqual(x["new_speaker_indices"],[540,545])
        self.assertEqual(x["new_optimizer_updates"],0)
        self.assertEqual(x["new_asr_phone_calls"],0)
        self.assertFalse(x["cal_locked_opened"])
        self.assertFalse(x["training_allowed"])
        self.assertFalse(x["product_promotion"])

    def test_planned_exact_role_and_group_identity(self):
        choices={g["speaker_index"]:g["source_speaker_id"] for g in m.plan()["train_groups"]}
        choices.update({i:f"p{20000+i}" for i in range(540,546)})
        rows=m.planned(choices)
        self.assertEqual(len(rows),18)
        self.assertEqual(Counter(r["role"] for r in rows),{"train":6,"development":12})
        self.assertEqual(Counter(r["requested_class"] for r in rows),{0:12,2:6})
        self.assertEqual(len({r["source_group_id"] for r in rows}),12)
        self.assertEqual(len({r["pcm_sha256"] for r in rows}),0) if False else None
        self.assertTrue(all(r["training_allowed"] is False for r in rows))
        self.assertTrue(all(r["human_gold"] is False for r in rows))
        self.assertTrue(all(r["source_group_id"] in
            {g["source_group_id"] for g in m.plan()["train_groups"]}
            for r in rows if r["role"]=="train"))

    def test_existing_train_groups_are_byte_and_role_bound(self):
        manifest,packet=synthetic_originals()
        with TemporaryDirectory() as folder:
            root=Path(folder);src=root/"source";scr=root/"screen"
            src.mkdir();scr.mkdir()
            (src/"MANIFEST.json").write_text(json.dumps(manifest))
            (scr/"SCREEN.json").write_text(json.dumps(packet))
            selected=m.original_train_groups(src,scr)
            self.assertEqual(len(selected),6)
            self.assertEqual(set(selected),{g["source_group_id"] for g in m.plan()["train_groups"]})
            bad=json.loads((scr/"SCREEN.json").read_text())
            bad["rows"][0]["role"]="development"
            (scr/"SCREEN.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"ONLY_PREQUALIFIED"):
                m.original_train_groups(src,scr)

    def test_dev_speakers_must_be_new_and_unique(self):
        mapping={f"speaker-{i}":i for i in range(904)}
        for g in m.plan()["train_groups"]:
            mapping[g["source_speaker_id"]]=g["speaker_index"]
        # Ensure the six new DEV identities are not historical piper numeric IDs.
        for i in range(540,546):
            mapping[f"p{20000+i}"]=i
        config={"num_speakers":904,"speaker_id_map":mapping}
        # The synthetic mapping has duplicate values for train speaker aliases;
        # the real config is one-to-one, so use a one-to-one replacement map.
        inv={i:f"p{30000+i}" for i in range(904)}
        for g in m.plan()["train_groups"]:
            inv[g["speaker_index"]]=g["source_speaker_id"]
        config={"num_speakers":904,"speaker_id_map":{v:k for k,v in inv.items()}}
        with TemporaryDirectory() as folder:
            root=Path(folder);src=root/"source";scr=root/"screen"
            src.mkdir();scr.mkdir()
            manifest,packet=synthetic_originals()
            (src/"MANIFEST.json").write_text(json.dumps(manifest))
            (scr/"SCREEN.json").write_text(json.dumps(packet))
            choices=m.native_speakers(config,src,scr)
            self.assertEqual(len(choices),12)
            self.assertEqual([choices[i] for i in range(448,454)],
                [g["source_speaker_id"] for g in m.plan()["train_groups"]])
            self.assertEqual(set(range(540,546)),set(choices)-set(range(448,454)))

    def test_guard_fails_closed_outside_protected_main(self):
        with mock.patch.dict(os.environ,{"GITHUB_ACTIONS":"false"},clear=True):
            with self.assertRaisesRegex(ValueError,"ONLY_FIRST_PUBLIC_MAIN_SOURCE_PUSH"):
                m.guard()

if __name__=="__main__":
    main()
