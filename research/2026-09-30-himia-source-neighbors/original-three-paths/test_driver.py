import unittest,tempfile,pathlib,wave,io,hashlib,os,json
import driver,prepare,subprocess,sys,time
class Tests(unittest.TestCase):
 def raw(self):
  b=io.BytesIO()
  with wave.open(b,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(640))
  return b.getvalue()
 def test_pcm_identity(self):
  b=self.raw();m=prepare.inspect(b)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'x.wav';p.write_bytes(b);self.assertEqual(len(driver.pcm(dict(path=str(p),**m))),640)
   p.write_bytes(b[:-1])
   with self.assertRaises(AssertionError):driver.pcm(dict(path=str(p),**m))
 def test_empty(self):
  b=io.BytesIO()
  with wave.open(b,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'')
  with self.assertRaises(ValueError):prepare.inspect(b.getvalue())
 def test_truncated(self):
  with self.assertRaises(ValueError):prepare.inspect(self.raw()[:-1])
 def test_labels(self):
  for k in [True,0,3,1.0]:
   with self.assertRaises(ValueError):driver.checked_events([{'keyword_id':k}])
 def test_nan(self):
  with self.assertRaises(ValueError):driver.checked_events([{'keyword_id':1,'score':float('nan')}])
 def test_event_order(self):
  a=[{'keyword_id':1},{'keyword_id':2}];self.assertNotEqual(driver.event_identity(a),driver.event_identity(list(reversed(a))))
 def test_group_resources(self):
  x=driver.process_group(os.getpgrp());self.assertGreaterEqual(x['processes'],1);self.assertGreater(x['rss_bytes'],0);self.assertGreaterEqual(x['cpu_s'],0)
 def test_native_stall_hard_kill(self):
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'clip.json'
   code="import ctypes,json,pathlib,time;pathlib.Path("+repr(str(path))+").write_text(json.dumps({'active':True,'started':time.monotonic()}));ctypes.CDLL(None).sleep(20)"
   p=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
   r=driver.monitor(p,time.monotonic(),path,clip_limit=.15,wall_limit=2)
   self.assertEqual(r['stop_reason'],'hard per-clip wall budget');self.assertLess(r['wall_s'],2);self.assertNotEqual(r['returncode'],0)
 def test_monitor_read_error_cleans_group(self):
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'bad.json';path.write_text('{bad')
   p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
   r=driver.monitor(p,time.monotonic(),path)
   self.assertIn('monitor error:',r['stop_reason']);self.assertIsNotNone(p.poll());self.assertNotEqual(p.returncode,0)
 def test_timeout(self):
  with self.assertRaises(TimeoutError):driver.timeout(None,None)
if __name__=='__main__':unittest.main()
