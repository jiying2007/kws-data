import tempfile,shutil,pathlib,json,gzip,unittest
import verify
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'pack';shutil.copytree(verify.BASE,self.root,ignore=shutil.ignore_patterns('__pycache__'))
 def tearDown(self):self.tmp.cleanup()
 def rebind(self):
  p=self.root/'archive-manifest.json';v=json.loads(p.read_text())
  v['files']=[dict(path=str(q.relative_to(self.root)),bytes=q.stat().st_size,sha256=verify.digest(q.read_bytes())) for q in sorted(self.root.rglob('*')) if q.is_file() and '__pycache__' not in q.parts and q.name!='archive-manifest.json']
  p.write_text(json.dumps(v))
 def replace(self,name,raw):
  if name in verify.SHARDED:
   packed=gzip.compress(raw,mtime=0);p=self.root/'gzip-shards.json';index=json.loads(p.read_text());entry=next(x for x in index['files'] if x['path']==name)
   for part in entry['parts']:(self.root/part['path']).unlink()
   entry.update(gzip_bytes=len(packed),gzip_sha256=verify.digest(packed),expanded_bytes=len(raw),expanded_sha256=verify.digest(raw),parts=[])
   for i,start in enumerate(range(0,len(packed),131072)):
    b=packed[start:start+131072];n=name+'.parts/'+f'{i:04d}.bin';(self.root/n).write_bytes(b);entry['parts'].append(dict(index=i,path=n,bytes=len(b),sha256=verify.digest(b)))
   p.write_text(json.dumps(index))
  else:(self.root/name).write_bytes(gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw)
  p=self.root/'copied-file-origins.json';origins=json.loads(p.read_text())
  for x in origins:
   if x['path']==name:x['original_bytes']=len(raw);x['original_sha256']=verify.digest(raw)
  p.write_text(json.dumps(origins));self.rebind()
 def reject(self):
  with self.assertRaises((ValueError,KeyError,AssertionError)):verify.verify(self.root)
 def test_valid(self):self.assertTrue(verify.verify(self.root)['verified'])
 def test_bytes(self):(self.root/'README.md').write_text('changed');self.reject()
 def test_boolean_schema(self):
  p=self.root/'archive-manifest.json';v=json.loads(p.read_text());v['schema_version']=True;p.write_text(json.dumps(v));self.reject()
 def test_gzip_original_identity(self):
  p=self.root/'gzip-shards.json';s=json.loads(p.read_text());s['files'][0]['expanded_sha256']='0'*64;p.write_text(json.dumps(s));self.rebind();self.reject()
 def test_missing_origin(self):
  p=self.root/'copied-file-origins.json';v=json.loads(p.read_text());p.write_text(json.dumps(v[:-1]));self.rebind();self.reject()
 def test_rebound_trigger(self):
  n='runs/c/records.jsonl.gz';r=[json.loads(x) for x in verify.payload(self.root,n).splitlines()];r[42]['events']=[{'keyword_id':1}];self.replace(n,b''.join((json.dumps(x)+'\n').encode() for x in r));self.reject()
 def test_rebound_duplicate(self):
  n='runs/c/records.jsonl.gz';r=verify.payload(self.root,n).splitlines();r[43]=r[42];self.replace(n,b'\n'.join(r)+b'\n');self.reject()
 def test_rebound_spec(self):
  n='runs/c/complete.json';r=json.loads(verify.payload(self.root,n));r['spec_sha256']='0'*64;self.replace(n,json.dumps(r).encode());self.reject()
 def test_rebound_thread_sample(self):
  n='runs/sherpa-api1/worker-thread-samples.json.gz';r=json.loads(verify.payload(self.root,n));r[0]['worker_threads']=2;self.replace(n,json.dumps(r).encode());self.reject()
 def test_rebound_source(self):
  n='original-three-paths/driver.py';self.replace(n,verify.payload(self.root,n)+b'\n# changed\n');self.reject()
 def test_false_returncode(self):
  n='runs/c/supervisor.json';r=json.loads(verify.payload(self.root,n));r['returncode']=False;self.replace(n,json.dumps(r).encode());self.reject()
 def test_non_utf8_text(self):
  (self.root/'README.md').write_bytes(b'\xff');self.rebind();self.reject()
 def test_symlink_root(self):
  link=pathlib.Path(self.tmp.name)/'linked';link.symlink_to(self.root,target_is_directory=True)
  with self.assertRaises(ValueError):verify.verify(link)
 def test_shard_order(self):
  p=self.root/'gzip-shards.json';s=json.loads(p.read_text());s['files'][0]['parts'].reverse();p.write_text(json.dumps(s));self.rebind();self.reject()
 def test_shard_path(self):
  p=self.root/'gzip-shards.json';s=json.loads(p.read_text());s['files'][0]['parts'][0]['path']='../escape';p.write_text(json.dumps(s));self.rebind();self.reject()
 def test_part_size_cap(self):
  p=self.root/'gzip-shards.json';s=json.loads(p.read_text());s['files'][0]['parts'][0]['bytes']=131073;p.write_text(json.dumps(s));self.rebind();self.reject()
 def test_duplicate_whole_gzip_rejected(self):
  n=next(iter(verify.SHARDED));(self.root/n).write_bytes(verify.read(self.root,n));self.rebind();self.reject()
 def test_expansion_cap(self):
  p=self.root/'gzip-shards.json';s=json.loads(p.read_text());s['files'][0]['expanded_bytes']=32*1024*1024+1;p.write_text(json.dumps(s));self.rebind();self.reject()
 def test_rebound_resource(self):
  n='runs/c/complete.json';r=json.loads(verify.payload(self.root,n));r['resource_usage']['cpu_s']=999;self.replace(n,json.dumps(r).encode());self.reject()
if __name__=='__main__':unittest.main()
