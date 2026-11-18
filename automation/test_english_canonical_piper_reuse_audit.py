"""Standard-library checks: no audio generation, model load or training."""
import importlib.util
from pathlib import Path
import unittest
import urllib.request
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

    def test_cross_origin_signed_blob_get_drops_bearer(self):
        req=urllib.request.Request(
            "https://api.github.com/repos/jiying2007/kws-data/actions/artifacts/3/zip",
            headers={"Authorization":"Bearer SENTINEL",
                     "Accept":"application/vnd.github+json"})
        url="https://productionresultssa0.blob.core.windows.net/actions-results/x.zip?sv=1&sig=xyz"
        redirected=m.SafeArtifactRedirect().redirect_request(
            req,None,302,"Found",{},url)
        self.assertEqual(redirected.full_url,url)
        self.assertNotIn("authorization",
                         {k.casefold() for k,v in redirected.header_items()})
        self.assertEqual(redirected.get_header("Accept"),
                         "application/vnd.github+json")

    def test_same_api_origin_retains_bearer(self):
        req=urllib.request.Request(
            "https://api.github.com/repos/jiying2007/kws-data/actions/artifacts/3",
            headers={"Authorization":"Bearer SENTINEL"})
        redirected=m.SafeArtifactRedirect().redirect_request(
            req,None,302,"Found",{},
            "https://api.github.com/repos/jiying2007/kws-data/actions/artifacts/3/zip")
        self.assertEqual(redirected.get_header("Authorization"),"Bearer SENTINEL")

    def test_untrusted_or_insecure_artifact_redirect_fails_closed(self):
        req=urllib.request.Request(
            "https://api.github.com/repos/jiying2007/kws-data/actions/artifacts/3/zip",
            headers={"Authorization":"Bearer SENTINEL"})
        for url in ("http://productionresultssa0.blob.core.windows.net/x.zip",
                    "https://attacker.example/x.zip",
                    "https://api.github.com.attacker.example/x.zip"):
            with self.subTest(url=url),self.assertRaises(ValueError):
                m.SafeArtifactRedirect().redirect_request(
                    req,None,302,"Found",{},url)

if __name__=="__main__": unittest.main()
