import unittest,pathlib,tempfile,shutil,json,struct,io,zipfile
import verify
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'pack';shutil.copytree(verify.BASE,self.root,ignore=shutil.ignore_patterns('__pycache__'));self.mapping,self.read=verify.make_reader(self.root)
 def tearDown(self):self.tmp.cleanup()
 def test_valid(self):self.assertTrue(verify.verify(self.root)['verified'])
 def test_corrupt_plain(self):
  (self.root/'fft/result.json').write_text('{}')
  with self.assertRaises(ValueError):verify.verify(self.root)
 def test_root_symlink(self):
  q=pathlib.Path(self.tmp.name)/'link';q.symlink_to(self.root,target_is_directory=True)
  with self.assertRaises(ValueError):verify.verify(q)
 def test_ancestor_symlink(self):
  q=pathlib.Path(self.tmp.name)/'alias';q.symlink_to(self.root.parent,target_is_directory=True)
  with self.assertRaises(ValueError):verify.verify(q/'pack')
 def test_generated_binary_rebound(self):
  p=self.root/'README.md';p.write_bytes(b'\xff');m=self.root/'archive-manifest.json';x=json.loads(m.read_text())
  for f in x['files']:
   if f['path']=='README.md':f.update(bytes=1,sha256=verify.sha(b'\xff'))
  m.write_text(json.dumps(x))
  with self.assertRaises(ValueError):verify.verify(self.root)
 def test_strict_json(self):
  for b in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}']:
   with self.assertRaises(ValueError):verify.strict_json(b)
 def test_gzip_trailing(self):
  p=self.root/'logical-files.json';x=json.loads(p.read_text());e=next(e for e in x['files'] if e['codec']=='gzip-utf8');old=self.root/e['parts'][0]['path'];b=old.read_bytes()+b'trailing';h=verify.sha(b);name='objects/'+h+'/0000.bin';q=self.root/name;q.parent.mkdir(parents=True);q.write_bytes(b);e.update(encoded_bytes=len(b),encoded_sha256=h,parts=[dict(path=name,bytes=len(b),sha256=h)]);p.write_text(json.dumps(x,separators=(',',':')))
  _,read=verify.make_reader(self.root)
  with self.assertRaises(ValueError):read(e['path'])
 def test_part_order(self):
  p=self.root/'logical-files.json';x=json.loads(p.read_text());e=next(e for e in x['files'] if e['codec']=='identity-npz');e['parts'].reverse();p.write_text(json.dumps(x,separators=(',',':')))
  _,read=verify.make_reader(self.root)
  with self.assertRaises(ValueError):read(e['path'])
 def test_mapping_escape(self):
  p=self.root/'logical-files.json';x=json.loads(p.read_text());x['files'][0]['parts'][0]['path']='../outside';p.write_text(json.dumps(x,separators=(',',':')))
  _,read=verify.make_reader(self.root)
  with self.assertRaises(ValueError):read(x['files'][0]['path'])
 def altered(self,name,change):
  def read(n):
   b=self.read(n)
   if n==name:
    x=json.loads(b);change(x);return json.dumps(x).encode()
   return b
  return read
 def test_fft_failure_not_erased(self):
  with self.assertRaises(ValueError):verify.check_fft(self.altered('fft/execution-status.json',lambda x:x.update(exit_code=0)))
 def test_fft_unfixed_gate(self):
  r=json.loads(self.read('fft/result.json'));r['cases'][0]['energies']['c']['stored_log']=r['cases'][0]['energies']['python']['stored_log']
  with self.assertRaises(ValueError):verify.check_fft(lambda n:json.dumps(r).encode() if n=='fft/result.json' else self.read(n))
 def test_fleurs_global_gate(self):
  with self.assertRaises(ValueError):verify.check_head(self.altered('head/dataset.json',lambda x:next(r for r in x if r['source']=='FLEURS')['source_record'].update(admitted_for_training=True)))
 def test_strict_gate(self):
  with self.assertRaises(ValueError):verify.check_head(self.altered('head/results/summary.json',lambda x:x['gates'].update(P_primary_all8_exact=True)))
 def test_100_steps(self):
  with self.assertRaises(ValueError):verify.check_head(self.altered('head/results/P-training.json',lambda x:x.update(optimizer_updates=101)))
 def test_npz_extra_member(self):
  b=io.BytesIO(self.read('head/head-parameters-and-normalization.npz'))
  with zipfile.ZipFile(b,'a') as z:z.writestr('unexpected.npy',b'bad')
  with self.assertRaises(ValueError):verify.arrays(b.getvalue(),False)
 def test_npz_nonfinite(self):
  source=zipfile.ZipFile(io.BytesIO(self.read('head/head-parameters-and-normalization.npz')));b=io.BytesIO()
  with zipfile.ZipFile(b,'w') as z:
   for name in source.namelist():
    raw=bytearray(source.read(name))
    if name=='P_head_bias.npy':raw[-4:]=struct.pack('<f',float('nan'))
    z.writestr(name,raw)
  with self.assertRaises(ValueError):verify.arrays(b.getvalue(),False)
if __name__=='__main__':unittest.main()
