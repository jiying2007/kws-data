import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest

BASE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lightweight_archive_verify',BASE/'verify.py')
archive=importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=pathlib.Path(self.temp.name)/'archive'
        shutil.copytree(BASE,self.root,ignore=shutil.ignore_patterns('__pycache__'))

    def tearDown(self):
        self.temp.cleanup()

    def rebind(self,name):
        path=self.root/name
        origins=json.loads((self.root/'copied-file-origins.json').read_text())
        for entry in origins:
            if entry['archive_path']==name:
                entry.update(bytes=path.stat().st_size,sha256=archive.sha(path))
        (self.root/'copied-file-origins.json').write_text(json.dumps(origins))
        manifest=json.loads((self.root/'archive-manifest.json').read_text())
        for entry in manifest['files']:
            p=self.root/entry['path']
            entry.update(bytes=p.stat().st_size,sha256=archive.sha(p))
        (self.root/'archive-manifest.json').write_text(json.dumps(manifest))

    def change(self,name,callback):
        path=self.root/name
        value=json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value))
        self.rebind(name)

    def test_clean(self):
        self.assertEqual(archive.verify(self.root)['runs'],4)

    def test_unindexed_audio_rejected(self):
        (self.root/'surprise.wav').write_bytes(b'not allowed')
        with self.assertRaisesRegex(ValueError,'unindexed archive'):
            archive.verify(self.root)

    def test_rebound_summary_count_rejected(self):
        self.change('results/A-seed1337/summary.json',
                    lambda x:x['groups']['human:development_a'].update(positive_target_hits=2))
        with self.assertRaisesRegex(ValueError,'recomputed summary'):
            archive.verify(self.root)

    def test_rebound_prediction_mismatch_rejected(self):
        self.change('results/A-seed1337/clip-scores.json',lambda x:x[0].update(predicted_keywords=[1,2]))
        with self.assertRaisesRegex(ValueError,'score threshold mismatch'):
            archive.verify(self.root)

    def test_native_role_relabel_rejected(self):
        self.change('results/B-seed2346/clip-scores.json',lambda x:x[0].update(split='train'))
        with self.assertRaisesRegex(ValueError,'native labels changed'):
            archive.verify(self.root)

    def test_weight_payload_tamper_rejected(self):
        def mutate(x):
            key=next(iter(x['tensors']))
            x['tensors'][key]['values'][0]+=1
        self.change('results/B-seed1337/checkpoint.weights.json',mutate)
        with self.assertRaisesRegex(ValueError,'tensor data identity'):
            archive.verify(self.root)

    def test_missing_mandatory_member_rejected(self):
        name='feature-audit.json'
        (self.root/name).unlink()
        manifest=json.loads((self.root/'archive-manifest.json').read_text())
        manifest['files']=[e for e in manifest['files'] if e['path']!=name]
        (self.root/'archive-manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'mandatory archive coverage'):
            archive.verify(self.root)

    def test_parent_symlink_rejected(self):
        original=self.root/'runtime'
        saved=pathlib.Path(self.temp.name)/'runtime'
        original.rename(saved)
        original.symlink_to(saved,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink archive path'):
            archive.verify(self.root)


    def test_feature_audit_count_tamper_rejected(self):
        self.change('feature-audit.json',lambda x:x['groups']['human:train'].update(clips=99))
        with self.assertRaisesRegex(ValueError,'feature audit counts'):
            archive.verify(self.root)

    def test_checkpoint_schema_bool_rejected(self):
        self.change('results/A-seed1337/checkpoint.weights.json',lambda x:x.update(schema_version=True))
        with self.assertRaisesRegex(ValueError,'checkpoint schema'):
            archive.verify(self.root)

    def test_manifest_schema_bool_rejected(self):
        path=self.root/'archive-manifest.json'
        value=json.loads(path.read_text());value['schema_version']=True;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'archive purpose'):
            archive.verify(self.root)


if __name__=='__main__':
    unittest.main()
