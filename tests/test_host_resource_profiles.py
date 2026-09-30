"""Offline archive checks; standard library only, no model/runtime downloads."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resource_evidence',ROOT/'tools/verify_host_resource_profiles.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class HostEvidenceTests(unittest.TestCase):
    def test_retained_evidence(self):
        self.assertGreaterEqual(v.verify()['verified_files'],29)

    def test_invalid_numbers(self):
        for x in (True, False, None, 0, -1, float('nan'), float('inf')):
            with self.subTest(x=x), self.assertRaises(ValueError):v.positive(x)
        with self.assertRaises(ValueError):v.close(float('nan'),1)
        with self.assertRaises(ValueError):v.close(1,2)

    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            with (p/'cfsmn/profile-corrected.json').open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):v.verify(p)

    def test_denominator_detected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            f=p/'frozen-c/cpu-io-run-1.json';d=json.loads(f.read_text());d['measured_audio_seconds']=0;f.write_text(json.dumps(d))
            mf=p/'manifest.json';manifest=json.loads(mf.read_text())
            for item in manifest['files']:
                if item['path']=='frozen-c/cpu-io-run-1.json':item.update(bytes=f.stat().st_size,sha256=v.sha(f))
            mf.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'denominator'):v.verify(p)

    def test_rebound_bad_percentile_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            f=p/'lightweight-ab/A-profile.json';d=json.loads(f.read_text());d['process_cpu_per_frame']['p99_us']*=2;f.write_text(json.dumps(d))
            mf=p/'manifest.json';manifest=json.loads(mf.read_text())
            for item in manifest['files']:
                if item['path']=='lightweight-ab/A-profile.json':item.update(bytes=f.stat().st_size,sha256=v.sha(f))
            mf.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Arithmetic'):v.verify(p)

    def test_joint_deletion_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            (p/'frozen-c/PROFILE_NOTES.md').unlink()
            f=p/'manifest.json';m=json.loads(f.read_text());m['files']=[i for i in m['files'] if i['path']!='frozen-c/PROFILE_NOTES.md'];f.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'mandatory inventory'):v.verify(p)

    def test_parent_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            external=Path(tmp)/'outside';shutil.move(p/'cfsmn',external);(p/'cfsmn').symlink_to(external,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink/containment'):v.verify(p)

    def test_manifest_bool_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pack';shutil.copytree(v.DEFAULT,p)
            f=p/'manifest.json';m=json.loads(f.read_text());m['schema_version']=True;f.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'schema/purpose'):v.verify(p)

    def test_extra_zip_member_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'times.npz';shutil.copyfile(v.DEFAULT/'lightweight-ab/A-timings.npz',p)
            with zipfile.ZipFile(p,'a') as z:z.writestr('unexpected.txt','extra')
            with self.assertRaisesRegex(ValueError,'ZIP members'):v.read_ns(p,'cpu_ns')

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'manifest.json').write_text(json.dumps({'schema_version':1,'purpose':'host-resource-evidence-not-target-qualification','files':[{'path':'../escape'}]}))
            with self.assertRaisesRegex(ValueError,'Unsafe'):v.verify(p)

if __name__=='__main__':unittest.main()
