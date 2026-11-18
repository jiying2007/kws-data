"""Standard-library checks: no audio generation, model load or training."""
import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name("english_canonical_piper_reuse_audit.py")
spec=importlib.util.spec_from_file_location("canonical_screen",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CanonicalOnly(unittest.TestCase):
    def test_old_cells_not_part_of_new_source(self):
        self.assertTrue(m.LEGACY.isdisjoint(m.CANONICAL))
        self.assertEqual(len(m.CANONICAL)+len(m.ORDINARY),6)
    def test_mero_meh_only(self):
        texts=dict.fromkeys(m.PROVIDERS,"Hey Mero.")
        self.assertEqual(m.blind(texts,["h","eɪ","m","ɛ","r","oʊ"])[1],2)
        self.assertEqual(m.blind(texts,["h","eɪ","m","iː","r","oʊ"])[0],"unknown")
    def test_voiceless_vosen_only(self):
        texts=dict.fromkeys(m.PROVIDERS,"Hey Vosen.")
        self.assertEqual(m.blind(texts,["h","eɪ","v","oʊ","s","ə","n"])[1],1)
        self.assertEqual(m.blind(texts,["h","eɪ","v","oʊ","z","ə","n"])[0],"unknown")
    def test_meero_alias_cannot_create_label(self):
        texts=dict.fromkeys(m.PROVIDERS,"Hey Meero.")
        self.assertEqual(m.blind(texts,["h","eɪ","m","ɛ","r","oʊ"])[0],"unknown")
    def test_unknown_not_a_negative(self):
        self.assertEqual(m.blind({"qwen":"Hello"},None)[0],"unknown")
        self.assertEqual(m.blind(dict.fromkeys(m.PROVIDERS,"Hey Nero."),
                         ["h","eɪ","n","ɛ","r","oʊ"])[0],"unknown")
    def test_ordinary_requires_consensus(self):
        self.assertEqual(m.blind(dict.fromkeys(m.PROVIDERS,"What time is it?"),
                         ["w","ʌ","t","t","aɪ","m"])[1],0)

if __name__=="__main__": unittest.main()
