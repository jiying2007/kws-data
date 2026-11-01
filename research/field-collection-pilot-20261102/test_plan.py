"""Pure stdlib, synthetic-waveform tests. No real people, model calls or networks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import plan as field


def p(speaker, role="train-candidate", suffix="1", source=None):
    return {"speaker_id":speaker,"session_id":"session-"+speaker+suffix,
            "source_family_id":source or "source-"+speaker,"role":role,
            "room_id":"room-"+speaker,"device_id":"device-"+speaker,
            "afe_profile_sha256":None}


def fixture():
    return {"schema":field.PARTICIPANTS_SCHEMA,"participants":[
        p("participant-a","train-candidate"),
        p("participant-b","development"),
        p("participant-c","sealed-evaluation")
    ]}


def wav(pcm):
    # Real canonical 16kHz/mono/PCM16 RIFF bytes, synthetic data ONLY.
    return (b"RIFF"+struct.pack("<I",36+len(pcm))+b"WAVEfmt "+
            struct.pack("<IHHIIHH",16,1,1,16000,32000,2,16)+
            b"data"+struct.pack("<I",len(pcm))+pcm)


class PilotPlan(unittest.TestCase):
    def test_fixed_plan_count_and_roles(self):
        a=field.build_plan(fixture())
        self.assertEqual(a["participants"],3)
        self.assertEqual(len(a["slots"]),111)
        self.assertEqual(len({s["slot_id"] for s in a["slots"]}),111)
        self.assertEqual({s["role"] for s in a["slots"]},set(field.SPLITS))
        self.assertEqual({s["distance_m"] for s in a["slots"]},{1,3,5})
        self.assertEqual(sum(s["intent_kind"]=="wake-intent" for s in a["slots"]),72)
        self.assertEqual(sum(s["intent_kind"]=="confusable-intent" for s in a["slots"]),36)
        self.assertEqual(sum(s["intent_kind"]=="continuous-negative-candidate" for s in a["slots"]),3)
        self.assertTrue(all(not s["training_admitted"] and not s["rights_approved"] for s in a["slots"]))
        self.assertFalse(a["qualification_allowed"])

    def test_output_order_permutation_is_stable(self):
        x=fixture()
        a=field.build_plan(x)
        x["participants"].reverse()
        self.assertEqual(field.build_plan(x),a)

    def test_same_speaker_across_roles_rejected(self):
        x=fixture()
        x["participants"][1]["speaker_id"]="participant-a"
        with self.assertRaises(ValueError):field.build_plan(x)

    def test_same_source_family_across_roles_rejected(self):
        x=fixture()
        x["participants"][1]["source_family_id"]=x["participants"][0]["source_family_id"]
        with self.assertRaises(ValueError):field.build_plan(x)

    def test_duplicate_sessions_rejected(self):
        x=fixture()
        x["participants"][1]["session_id"]=x["participants"][0]["session_id"]
        with self.assertRaises(ValueError):field.build_plan(x)

    def test_invalid_role_and_opaque_metadata_rejected(self):
        for key,value in (("role","final-admitted"),("speaker_id","../../me"),
                          ("session_id","S P A C E"),("device_id","secret@example.com"),
                          ("afe_profile_sha256","broken")):
            x=fixture();x["participants"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):field.build_plan(x)

    def test_new_schema_fields_are_not_silently_accepted(self):
        x=fixture()
        x["participants"][0]["self_reported_training_approved"]=True
        with self.assertRaises(ValueError):field.build_plan(x)

    def test_trust_boundaries_do_not_change_even_with_all_roles(self):
        value=field.build_plan(fixture())
        for k in ("human_rights_authenticated","audio_labels_verified",
                  "training_admitted","qualification_allowed","shipping_approved"):
            self.assertIs(value[k],False)
        self.assertEqual(value["model_calls"],0)


class PrivateAudioAudit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=field.build_plan(fixture())

    def clip(self,filename="one.wav",pcm=None):
        if pcm is None:pcm=struct.pack("<h",128)*320
        data=wav(pcm)
        (self.root/filename).write_bytes(data)
        return {
            "slot_id":self.plan["slots"][0]["slot_id"],
            "path":filename,
            "wav_sha256":hashlib.sha256(data).hexdigest(),
            "pcm_sha256":hashlib.sha256(pcm).hexdigest(),
            "frames":len(pcm)//2,
            "human_verbatim_reviewed":False,
            "rights_approved":False,
            "training_admitted":False,
        }

    def records(self,rows):
        return {"schema":field.RECORDINGS_SCHEMA,"recordings":rows}

    def test_real_wav_hashes_are_checked_but_not_admitted(self):
        row=self.clip()
        summary=field.audit_recordings(self.plan,self.records([row]),self.root)
        self.assertEqual(summary["verified_recording_bytes"],1)
        self.assertFalse(summary["all_slots_present"])
        self.assertEqual(sum(v for role in summary["count_by_role_and_declared_intent"].values() for v in role.values()),1)
        for k in ("training_admitted","rights_approved","actual_spoken_text_verified",
                  "audio_labels_verified","qualification_allowed","shipping_approved",
                  "physical_ssc305_qualification"):
            self.assertIs(summary[k],False)

    def test_empty_recordings_not_ready(self):
        x=field.audit_recordings(self.plan,self.records([]),self.root)
        self.assertEqual(x["verified_recording_bytes"],0)
        self.assertFalse(x["all_slots_present"])
        self.assertFalse(x["training_admitted"])

    def test_exact_wav_sha_and_pcm_sha_and_frames(self):
        for key,wrong in (("wav_sha256","0"*64),("pcm_sha256","f"*64),("frames",321),("frames",True)):
            row=self.clip();row[key]=wrong
            with self.subTest(key=key),self.assertRaises(ValueError):
                field.audit_recordings(self.plan,self.records([row]),self.root)

    def test_empty_or_broken_wav_rejected(self):
        for data in (b"",b"garbage",wav(b"\0\0")[:-1],wav(b"\0\0")+b"!",
                     wav(b"\0\0")[:24]+b"\x00"+wav(b"\0\0")[25:]):
            self.root.joinpath("bad.wav").write_bytes(data)
            row=self.clip("good.wav")
            row.update(path="bad.wav",wav_sha256=hashlib.sha256(data).hexdigest())
            with self.subTest(size=len(data)),self.assertRaises(ValueError):
                field.audit_recordings(self.plan,self.records([row]),self.root)

    def test_reject_symlink_to_file(self):
        row=self.clip()
        (self.root/"alias.wav").symlink_to(self.root/"one.wav")
        row["path"]="alias.wav"
        with self.assertRaises(ValueError):
            field.audit_recordings(self.plan,self.records([row]),self.root)

    def test_reject_hardlink_to_file(self):
        row=self.clip()
        os.link(self.root/"one.wav",self.root/"two.wav")
        with self.assertRaises(ValueError):
            field.audit_recordings(self.plan,self.records([row]),self.root)

    def test_reject_audio_root_alias(self):
        row=self.clip()
        alias=self.root.parent/(self.root.name+"-alias")
        alias.symlink_to(self.root,target_is_directory=True)
        self.addCleanup(lambda:alias.unlink(missing_ok=True))
        with self.assertRaises(ValueError):
            field.audit_recordings(self.plan,self.records([row]),alias)

    def test_unsafe_paths_are_rejected(self):
        row=self.clip()
        for value in ("../one.wav","/one.wav","one.wav/../two.wav",
                      "dir\\one.wav","one.mp3","./one.wav","one%2fwav.wav"):
            x=dict(row,path=value)
            with self.subTest(value=value),self.assertRaises(ValueError):
                field.audit_recordings(self.plan,self.records([x]),self.root)

    def test_duplicate_recording_and_foreign_slot_rejected(self):
        row=self.clip()
        for collection in ([row,row],[dict(row,slot_id="s-"+"0"*24)]):
            with self.assertRaises(ValueError):
                field.audit_recordings(self.plan,self.records(collection),self.root)

    def test_reused_audio_payload_is_rejected(self):
        a=self.clip("one.wav")
        b=self.clip("two.wav")
        b["slot_id"]=self.plan["slots"][1]["slot_id"]
        with self.assertRaises(ValueError):
            field.audit_recordings(self.plan,self.records([a,b]),self.root)

    def test_false_rights_review_or_training_admission_rejected(self):
        for key in ("human_verbatim_reviewed","rights_approved","training_admitted"):
            for value in (True,1,"true"):
                row=self.clip()
                row[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                    field.audit_recordings(self.plan,self.records([row]),self.root)

    def test_recording_extra_metadata_rejected(self):
        row=self.clip();row["actual_text"]="你好小窝"
        with self.assertRaises(ValueError):
            field.audit_recordings(self.plan,self.records([row]),self.root)


class ParsersAndOutput(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"input.json"
            for sample in (b'{"a":1,"a":2}',b'{"k":NaN}',b'{"k":1e999}',b'\xff'):
                path.write_bytes(sample)
                with self.subTest(sample=sample),self.assertRaises(ValueError):
                    field._json_file(path)

    def test_output_create_only_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"audit.json"
            field._publish(path,{"training_admitted":False})
            orig=path.read_bytes()
            with self.assertRaises(ValueError):
                field._publish(path,{"training_admitted":True})
            self.assertEqual(path.read_bytes(),orig)

    def test_cli_plan_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"participants.json";out=Path(tmp)/"plan.json"
            src.write_text(json.dumps(fixture()))
            self.assertEqual(field.main(["plan","--participants",str(src),"--output",str(out)]),0)
            value=json.loads(out.read_bytes())
            self.assertEqual(value["participants"],3)
            self.assertEqual(len(value["slots"]),111)
            self.assertFalse(value["training_admitted"])

    def test_cli_verify_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=field.build_plan(fixture())
            (root/"plan.json").write_text(json.dumps(plan))
            (root/"items.json").write_text(json.dumps({"schema":field.RECORDINGS_SCHEMA,"recordings":[]}))
            self.assertEqual(field.main(["verify","--plan",str(root/"plan.json"),
                "--recordings",str(root/"items.json"),"--audio-root",str(root),
                "--output",str(root/"audit.json")]),0)
            self.assertFalse(json.loads((root/"audit.json").read_bytes())["training_admitted"])


if __name__ == "__main__":
    unittest.main()
