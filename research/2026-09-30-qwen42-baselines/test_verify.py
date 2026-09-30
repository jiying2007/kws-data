import copy
import json
import hashlib
import importlib.util
import pathlib
import tempfile
import shutil
import unittest

ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('archive_verify',ROOT/'verify.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def rebind(root):
    origins=json.loads((root/'copied-file-origins.json').read_text())
    for item in origins:
        p=root/item['archive_path'];item['bytes']=p.stat().st_size;item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'copied-file-origins.json').write_text(json.dumps(origins))
    manifest=json.loads((root/'archive-manifest.json').read_text())
    for item in manifest['files']:
        p=root/item['path'];item['bytes']=p.stat().st_size;item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'archive-manifest.json').write_text(json.dumps(manifest))


class ArchiveTests(unittest.TestCase):
    def test_intact_pack(self):
        result=module.verify(ROOT)
        self.assertEqual(result['ours/raw']['target_hits'],0)
        self.assertEqual(result['external/head500ms_tail500ms']['target_hits'],19)

    def test_modified_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            with (root/'external/raw/raw-events.jsonl').open('a') as handle:handle.write('\n')
            with self.assertRaisesRegex(ValueError,'archive hash mismatch'):
                module.verify(root)

    def test_summary_must_match_events(self):
        report=module.read(ROOT,'external/raw/readback.json')
        row=copy.deepcopy(next(x for x in report['recordings'] if x['target_hit']))
        row['target_hit']=False
        with self.assertRaisesRegex(ValueError,'target-hit field mismatch'):
            module.aggregate([row],'events')

    def test_rebound_bad_group_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            p=root/'ours/raw/clip-readback.json';report=json.loads(p.read_text())
            report['groups'][0]['target_misses']+=1
            p.write_text(json.dumps(report))
            rebind(root)
            with self.assertRaisesRegex(ValueError,'canonical subgroup counter mismatch'):
                module.verify(root)

    def test_joint_removal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            victim='external/license/model-README.md';(root/victim).unlink()
            p=root/'copied-file-origins.json';items=json.loads(p.read_text());p.write_text(json.dumps([x for x in items if x['archive_path']!=victim]))
            p=root/'archive-manifest.json';manifest=json.loads(p.read_text());manifest['files']=[x for x in manifest['files'] if x['path']!=victim];p.write_text(json.dumps(manifest))
            rebind(root)
            with self.assertRaisesRegex(ValueError,'mandatory artifact coverage mismatch'):
                module.verify(root)

    def test_rebound_binary_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            (root/'external/license/model-README.md').write_bytes(b'\xff\xfe')
            rebind(root)
            with self.assertRaisesRegex(ValueError,'non-UTF8 archive payload'):
                module.verify(root)

    def test_rebound_wrong_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            p=root/'external/raw/run-provenance.json';value=json.loads(p.read_text())
            value['keywords_sha256']='0'*64;p.write_text(json.dumps(value));rebind(root)
            with self.assertRaisesRegex(ValueError,'external provenance binding mismatch'):
                module.verify(root)

    def test_rebound_missing_subgroup_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'pack'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            p=root/'external/raw/readback.json';report=json.loads(p.read_text())
            report['groups']=[g for g in report['groups'] if not(g['dimension']=='review' and g['value']=='human')]
            p.write_text(json.dumps(report))
            q=root/'external/raw/run-provenance.json';value=json.loads(q.read_text());value['readback_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();q.write_text(json.dumps(value))
            rebind(root)
            with self.assertRaisesRegex(ValueError,'external subgroup coverage mismatch'):
                module.verify(root)

if __name__=='__main__':unittest.main()
