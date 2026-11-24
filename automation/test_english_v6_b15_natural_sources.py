"""Source-only B15 dev-clean reader-selection contract without audio/model calls."""
from pathlib import Path
import io
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b15_natural_sources as b15

class FakeMember:
    def __init__(self,name,data): self.name=name;self.data=data
class FakeArchive:
    def __init__(self,members): self.members=members
    def getmembers(self): return self.members
    def extractfile(self,item): return io.BytesIO(item.data)

class NaturalSelection(unittest.TestCase):
    def test_pinned_corpus_and_exposed_speaker_exclusion(self):
        p=b15.plan()
        self.assertEqual(p["upstream"]["bytes"],337926286)
        self.assertEqual(p["upstream"]["sha256"],"76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3")
        self.assertEqual(p["primary_speaker_slots"],12)
        self.assertFalse(p["train_allowed"])
        self.assertGreaterEqual(len(p["excluded_speakers"]),900)
    def test_deterministic_selection_from_only_source_metadata(self):
        p=b15.plan(); used=set(p["excluded_speakers"]);members=[]
        for n in range(16):
            speaker=str(90000+n)
            self.assertNotIn(speaker,used)
            name=f"LibriSpeech/dev-clean/{speaker}/1000/{speaker}-1000.trans.txt"
            raw=(f"{speaker}-1000-0001 ONE TWO THREE FOUR\n{speaker}-1000-0002 FIVE SIX SEVEN EIGHT\n").encode()
            members.append(FakeMember(name,raw))
        ar=FakeArchive(members)
        a=b15.pick(ar,p);b=b15.pick(ar,p)
        self.assertEqual(a,b)
        self.assertEqual(set(a),{"primary"})
        self.assertEqual(len(a["primary"]),24)
        self.assertEqual(len({x["speaker"] for x in a["primary"]}),12)
        self.assertTrue(all(x["source_id"].startswith(x["speaker"]+"-") for x in a["primary"]))

if __name__=="__main__": unittest.main()
