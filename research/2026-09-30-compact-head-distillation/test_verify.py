import gzip,hashlib,importlib.util,json,pathlib,shutil,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v',ROOT/'verify.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class ArchiveTests(unittest.TestCase):
 def copy(self,tmp):
  p=pathlib.Path(tmp)/'archive';shutil.copytree(ROOT,p,ignore=shutil.ignore_patterns('__pycache__'));return p
 def rebind(self,p,name):
  f=p/'archive-manifest.json';m=json.loads(f.read_text());target=p/name
  for item in m['files']:
   if item['path']==name:item.update(bytes=target.stat().st_size,sha256=v.sha(target))
  f.write_text(json.dumps(m))
 def rebind_original(self,p,name):
  f=p/'copied-file-origins.json';m=json.loads(f.read_text());target=p/name
  for item in m:
   if item['path']==name:item.update(source_bytes=len(v.logical_bytes(target)),source_sha256=v.sha(target))
  f.write_text(json.dumps(m));self.rebind(p,name if target.exists() else name+'.gz');self.rebind(p,'copied-file-origins.json')
 def test_copied_inventory_deletion(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'copied-file-origins.json';d=json.loads(f.read_text());d.pop();f.write_text(json.dumps(d));self.rebind(p,'copied-file-origins.json')
   with self.assertRaisesRegex(ValueError,'Copied source coverage'):v.verify(p)
 def test_script_provenance_even_rebound(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'student.py';f.write_text(f.read_text()+'\n# changed\n');self.rebind_original(p,'student.py')
   with self.assertRaisesRegex(ValueError,'provenance'):v.verify(p)
 def test_target_flag_events_contradiction(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'student-results/raw.json';d=json.loads(v.logical_bytes(f));row=next(x for x in d['recordings'] if x['kind']=='positive' and not x['events']);row['target_hit']=True;self.write_gzip(p,'student-results/raw.json',json.dumps(d).encode());self.rebind_original(p,'student-results/raw.json')
   with self.assertRaisesRegex(ValueError,'Target-hit'):v.verify(p)
 def write_gzip(self,p,name,data,declared_size=None):
  packed=gzip.compress(data,mtime=0);(p/(name+'.gz')).write_bytes(packed)
  f=p/'compression-manifest.json';m=json.loads(f.read_text())
  for item in m['files']:
   if item['logical_path']==name:item.update(expanded_bytes=len(data) if declared_size is None else declared_size,expanded_sha256=hashlib.sha256(data).hexdigest(),compressed_bytes=len(packed),compressed_sha256=hashlib.sha256(packed).hexdigest())
  f.write_text(json.dumps(m));self.rebind(p,name+'.gz');self.rebind(p,'compression-manifest.json')
 def test_expansion_limit(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);self.write_gzip(p,'student-results/raw.json',b'a'*(v.MAX_EXPANDED+1),declared_size=v.MAX_EXPANDED)
   with self.assertRaisesRegex(ValueError,'Expanded size limit'):v.verify(p)
 def test_expanded_hash_rebound(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'compression-manifest.json';m=json.loads(f.read_text());m['files'][0]['expanded_sha256']='0'*64;f.write_text(json.dumps(m));self.rebind(p,'compression-manifest.json')
   with self.assertRaisesRegex(ValueError,'identity'):v.verify(p)
 def test_gzip_corruption(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'student-results/raw.json.gz';f.write_bytes(f.read_bytes()[:-8]+b'badcrc!!');self.rebind(p,'student-results/raw.json.gz')
   with self.assertRaises((ValueError,OSError,EOFError)):v.verify(p)
 def test_valid(self):self.assertTrue(v.verify()['candidate_failed'])
 def test_head_nonnumeric_even_rebound(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'head-float32.json';d=json.loads(f.read_text());next(iter(d['tensors'].values()))['values'][0]=True;f.write_text(json.dumps(d));self.rebind(p,'head-float32.json')
   with self.assertRaises(ValueError):v.verify(p)
 def test_joint_delete(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);(p/'RESULTS.md').unlink();f=p/'archive-manifest.json';d=json.loads(f.read_text());d['files']=[x for x in d['files'] if x['path']!='RESULTS.md'];f.write_text(json.dumps(d))
   with self.assertRaisesRegex(ValueError,'inventory'):v.verify(p)
 def test_parent_symlink(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);outside=pathlib.Path(tmp)/'outside';shutil.move(p/'student-results',outside);(p/'student-results').symlink_to(outside,target_is_directory=True)
   with self.assertRaisesRegex(ValueError,'Symlink'):v.verify(p)
 def test_rebound_false_pass(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);f=p/'student-results/failure-readback.json';d=json.loads(f.read_text());d['compression_diagnostic_passed']=True;f.write_text(json.dumps(d));self.rebind(p,'student-results/failure-readback.json')
   with self.assertRaises(ValueError):v.verify(p)
 def test_no_extra_full_checkpoint(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=self.copy(tmp);(p/'student-results/compact-final.pt').write_bytes(b'not permitted')
   with self.assertRaisesRegex(ValueError,'inventory'):v.verify(p)
if __name__=='__main__':unittest.main()
