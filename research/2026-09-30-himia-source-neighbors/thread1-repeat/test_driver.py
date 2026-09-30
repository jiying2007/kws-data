import unittest,os,json,pathlib,subprocess,sys,tempfile,time
import driver
class Tests(unittest.TestCase):
 def test_worker_only_threads(self):
  x=driver.observe_worker(os.getpid());self.assertEqual(x['worker_pid'],os.getpid());self.assertGreaterEqual(x['worker_threads'],1)
 def test_monitor_bad_heartbeat_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   q=pathlib.Path(d)/'clip.json';q.write_text('{bad')
   p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
   r=driver.monitor(p,time.monotonic(),q);self.assertEqual(r['status'],'invalid');self.assertIn('monitor error',r['stop_reason']);self.assertIsNotNone(p.poll())
 def test_monitor_completed_threads(self):
  with tempfile.TemporaryDirectory() as d:
   p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'],start_new_session=True)
   r=driver.monitor(p,time.monotonic(),pathlib.Path(d)/'clip.json');self.assertEqual(r['status'],'completed');self.assertGreaterEqual(r['worker_threads_max'],1)
if __name__=='__main__':unittest.main()
