"""One-shot execution journal and bounded single-thread CPU/RSS/wall watchdog."""
import argparse, hashlib, json, os, pathlib, resource, signal, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parent
PYTHON='/workspace/shared/kws-lightweight-prototypes/venv/bin/python'
ENV={k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']}
ENV.update({'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
DEFAULT={'cpu_seconds_process_group':600,'rlimit_cpu_soft':600,'rlimit_cpu_hard':605,'wall_seconds':900,'rss_bytes':2147483648,'minimum_launch_available_bytes':3221225472,'minimum_available_bytes':1073741824,'monitor_interval_seconds':.1,'termination_grace_seconds':3}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with open(p,'x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def available():
 host=next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
 limits=[]
 for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
  left,right=line.split(' - ',1)
  if right.split()[0]!='cgroup2':continue
  root=pathlib.Path(left.split()[4])
  try:
   limit=(root/'memory.max').read_text().strip()
   if limit!='max':limits.append(max(0,int(limit)-int((root/'memory.current').read_text())))
  except (FileNotFoundError,PermissionError):pass
 return min([host]+limits),bool(limits)

def group_state(pgid):
 rss=0;cpu=0.;active=0;threads=0;hz=os.sysconf('SC_CLK_TCK')
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if os.getpgid(int(p.name))!=pgid:continue
   s=(p/'stat').read_text().rsplit(')',1)[1].split()
   if s[0]=='Z':continue
   cpu+=(int(s[11])+int(s[12]))/hz;active+=1
   for line in (p/'status').read_text().splitlines():
    if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
    if line.startswith('Threads:'):threads+=int(line.split()[1])
  except (FileNotFoundError,ProcessLookupError):pass
 return {'rss_bytes':rss,'cpu_seconds':cpu,'processes':active,'threads':threads}

def stop(p,grace):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 deadline=time.monotonic()+grace
 while time.monotonic()<deadline and group_state(p.pid)['processes']:time.sleep(.02)
 if group_state(p.pid)['processes']:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['tests','preflight','run']);args=a.parse_args()
 limits=DEFAULT.copy()
 if args.action=='run':
  lock=json.loads((ROOT/'preflight-lock.json').read_text())
  for name,h in lock.items():assert sha(ROOT/name)==h,('preflight drift',name)
  review=json.loads((ROOT/'independent-approval.json').read_text())
  assert review['approved'] is True and review['preflight_lock_sha256']==sha(ROOT/'preflight-lock.json')
  limits=json.loads((ROOT/'spec.json').read_text())['resource_limits'];assert limits==DEFAULT
  save(ROOT/'execution-started.json',{'action':'run','one_authorized_pair_only':True,'preflight_lock_sha256':sha(ROOT/'preflight-lock.json'),'independent_approval_sha256':sha(ROOT/'independent-approval.json'),'timestamp_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 effective,cgroup=available();assert effective>=limits['minimum_launch_available_bytes'],'insufficient launch memory'
 def child_limits():resource.setrlimit(resource.RLIMIT_CPU,(limits['rlimit_cpu_soft'],limits['rlimit_cpu_hard']))
 command=[PYTHON,str(ROOT/'test_contract.py')] if args.action=='tests' else [PYTHON,str(ROOT/'experiment.py'),args.action]
 start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak={'rss_bytes':0,'cpu_seconds':0.,'processes':0,'threads':0};reason=None
 with open(ROOT/f'{args.action}.log','x') as log:
  p=subprocess.Popen(command,env={**os.environ,**ENV},cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_limits)
  try:
   while p.poll() is None:
    state=group_state(p.pid)
    for k,v in state.items():peak[k]=max(peak[k],v)
    memory,_=available()
    if state['cpu_seconds']>=limits['cpu_seconds_process_group']:reason='CPU_budget_exceeded'
    elif time.monotonic()-start>=limits['wall_seconds']:reason='wall_watchdog'
    elif state['rss_bytes']>=limits['rss_bytes']:reason='RSS_watchdog'
    elif memory<limits['minimum_available_bytes']:reason='low_memory_watchdog'
    elif state['processes']>1:reason='unexpected_child_process'
    if reason:break
    time.sleep(limits['monitor_interval_seconds'])
  finally:
   if group_state(p.pid)['processes']:stop(p,limits['termination_grace_seconds'])
   p.wait()
 after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
 if cpu>limits['cpu_seconds_process_group'] and reason is None:reason='CPU_budget_exceeded_at_exit'
 result={'action':args.action,'exit_code':p.returncode,'stop_reason':reason,'wall_seconds':time.monotonic()-start,'child_CPU_seconds':cpu,
 'peak_sampled':peak,'limits':limits,'effective_memory_at_launch':effective,'finite_cgroup_limit_visible':cgroup,
 'guard_scope':'child process group plus process RLIMIT_CPU; Python main is sole permitted process; sampled RSS/CPU may overshoot; no SSC305 inference',
 'environment_allowlist':ENV,'success':p.returncode==0 and reason is None}
 save(ROOT/f'{args.action}-resource.json',result);print(json.dumps(result),flush=True)
 if not result['success']:raise SystemExit(1)

if __name__=='__main__':main()
