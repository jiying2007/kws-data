"""Offline contract tests; integration tests use the exact retained original ZIP."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile
import verify_archive as v

class FormatTests(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaises(ValueError): v.parse(b'{"x":1,"x":2}')
    def test_nonfinite_json(self):
        for token in (b'NaN',b'Infinity',b'-Infinity',b'1e9999'):
            with self.subTest(token=token), self.assertRaises(ValueError): v.parse(token)
    def test_finite_json(self):
        self.assertEqual(v.parse(b'{"x":1.25}'),{'x':1.25})
    def test_canonical_distinguishes_types(self):
        self.assertNotEqual(v.canonical(False),v.canonical(0))
        self.assertNotEqual(v.canonical(1),v.canonical(1.0))
    @staticmethod
    def wav():
        data=struct.pack('<hh',10,-10)
        body=b'WAVEfmt '+struct.pack('<IHHIIHH',16,1,1,16000,32000,2,16)+b'data'+struct.pack('<I',len(data))+data
        return b'RIFF'+struct.pack('<I',len(body))+body
    def test_valid_pcm(self):
        self.assertEqual(v.wav_payload(self.wav(),16000,1,16),struct.pack('<hh',10,-10))
    def test_truncated_pcm(self):
        with self.assertRaises(ValueError):v.wav_payload(self.wav()[:-1],16000,1,16)
    def test_trailing_pcm(self):
        with self.assertRaises(ValueError):v.wav_payload(self.wav()+b'junk',16000,1,16)
    def test_wrong_geometry(self):
        with self.assertRaises(ValueError):v.wav_payload(self.wav(),24000,3,32)
    def test_odd_pcm(self):
        b=bytearray(self.wav()[:-1]);struct.pack_into('<I',b,4,len(b)-8);struct.pack_into('<I',b,40,3)
        with self.assertRaises(ValueError):v.wav_payload(bytes(b),16000,1,16)

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=Path(os.environ.get('QWEN16_ARCHIVE',str(Path(__file__).parent/'evidence/run-41008894901/original.zip')))
        if not cls.archive.exists():
            if os.environ.get('REQUIRE_ARCHIVE')=='1':raise RuntimeError('required original archive missing')
            raise unittest.SkipTest('source-only check: original archive not yet imported')
        cls.raw,cls.files,cls.report=v.read_archive(cls.archive)
    def setUp(self): self.files=dict(self.files)
    def change(self,name,value):
        self.files[name]=v.canonical(value)+b'\n'
        freeze=v.parse(self.files['artifact-freeze.json'])
        freeze['files'][name]={'bytes':len(self.files[name]),'sha256':v.sha(self.files[name])}
        self.files['artifact-freeze.json']=v.canonical(freeze)+b'\n'
    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError)):v.inspect_members(self.files)
    def test_exact_archive(self):
        self.assertEqual(self.report['verified_members'],59);self.assertEqual(self.report['unique_pcm'],16)
        self.assertEqual(self.report['measured_seconds'],43.2);self.assertFalse(self.report['ctc_training_allowed'])
    def test_missing_member(self):self.files.pop('ledger.json');self.reject()
    def test_extra_member(self):self.files['extra']=b'x';self.reject()
    def test_bad_member_hash(self):self.files['plan.json']+=b' ';self.reject()
    def test_training_flag(self):
        x=v.parse(self.files['artifact-freeze.json']);x['training_admitted']=True;self.files['artifact-freeze.json']=v.canonical(x);self.reject()
    def test_host_boolean_count(self):
        x=v.parse(self.files['host-receipt.json']);x['run_attempt']=True;self.change('host-receipt.json',x);self.reject()
    def test_source_commit(self):
        x=v.parse(self.files['host-receipt.json']);x['source_commit']='a'*40;self.change('host-receipt.json',x);self.reject()
    def test_ledger_reset(self):
        x=v.parse(self.files['ledger.json']);x['cells'][0]['attempts']=0;self.change('ledger.json',x);self.reject()
    def test_claim_missing(self):
        self.files['output/qwen16-20261030-000.claim']=b'';self.reject()
    def test_serena_train(self):
        x=v.parse(self.files['plan.json']);x[4]['proposed_split']='train';self.change('plan.json',x);self.reject()
    def test_actual_label_invented(self):
        x=v.parse(self.files['output/generation-receipt.json']);x['outcomes'][0]['actual_text']='你好小窝';self.change('output/generation-receipt.json',x);self.reject()
    def test_false_eos(self):
        x=v.parse(self.files['output/generation-receipt.json']);x['outcomes'][0]['native_termination']['ended_with_eos']=False;self.change('output/generation-receipt.json',x);self.reject()
    def test_missing_outcome(self):
        x=v.parse(self.files['output/generation-receipt.json']);x['outcomes'].pop();self.change('output/generation-receipt.json',x);self.reject()
    def test_wrong_pcm_identity(self):
        x=v.parse(self.files['output/generation-receipt.json']);x['outcomes'][0]['pcm_sha256']='0'*64;self.change('output/generation-receipt.json',x);self.reject()
    def test_raw_statistics(self):
        x=v.parse(self.files['output/generation-receipt.json']);x['outcomes'][0]['native_metrics']['rms']=0.5;self.change('output/generation-receipt.json',x);self.reject()
    def test_corrupt_original_zip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(self.raw[:-1]+bytes([self.raw[-1]^1]))
            with self.assertRaises(ValueError):v.read_archive(p)
    def test_materialization_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';v.materialize(self.archive,out)
            self.assertEqual((out/'original.zip').read_bytes(),self.raw)
            for name,raw in self.files.items():self.assertEqual((out/'artifact'/name).read_bytes(),raw)
            page=(out/'review.html').read_text();self.assertNotIn('你好小窝',page);self.assertNotIn('小窝小窝',page)
            self.assertEqual(page.count('<audio '),16)
            with self.assertRaises(ValueError):v.materialize(self.archive,out)
            self.assertEqual((out/'original.zip').read_bytes(),self.raw)

if __name__=='__main__':unittest.main()
