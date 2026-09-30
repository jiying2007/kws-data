"""Standard-library preservation/count checks; no model or audio execution."""
import collections,gzip,hashlib,json,math,pathlib,re
BASE=pathlib.Path(__file__).resolve().parent
ROUTES=('c','sherpa-api2','cfsmn','sherpa-api1')
COPIED=set(['data/source-input-identities.json.gz']+['original-three-paths/'+n for n in ['PLAN.md','RESULTS.md','run-spec.json','prepare.py','driver.py','freeze.py','test_driver.py','summarize.py']]+['thread1-repeat/'+n for n in ['PLAN.md','RESULTS.md','run-spec.json','driver.py','test_driver.py']]+['runs/'+r+'/'+n for r in ROUTES for n in ['records.jsonl.gz','complete.json','supervisor.json','recomputed.json']]+['runs/sherpa-api1/worker-thread-samples.json.gz'])
GENERATED={'README.md','verify.py','test_verify.py','copied-file-origins.json','gzip-shards.json'}
def digest(b):return hashlib.sha256(b).hexdigest()
def fail(ok,msg):
 if not ok:raise ValueError(msg)
def raw_read(root,name):
 p=root/name;fail(not root.is_symlink() and not p.is_symlink() and p.is_file() and root.resolve() in p.resolve().parents,'unsafe/missing member')
 for parent in p.parents:
  fail(not parent.is_symlink(),'symlink parent')
  if parent==root:break
 fail(p.stat().st_size<=(131072 if name.endswith('.bin') else 16*1024*1024),'physical member size cap')
 return p.read_bytes()
SHARDED={x for x in COPIED if x.endswith('.gz') and not x.endswith('worker-thread-samples.json.gz')}
def shard_index(root):
 s=json.loads(raw_read(root,'gzip-shards.json'));fail(type(s['schema_version']) is int and s['schema_version']==1 and type(s['part_max_bytes']) is int and s['part_max_bytes']==131072,'shard schema')
 fail({x['path'] for x in s['files']}==SHARDED and len(s['files'])==len(SHARDED),'shard inventory')
 for x in s['files']:
  fail(type(x['gzip_bytes']) is int and 0<x['gzip_bytes']<=16*1024*1024 and type(x['expanded_bytes']) is int and 0<x['expanded_bytes']<=32*1024*1024,'shard size cap')
  fail(len(x['parts'])==(x['gzip_bytes']+131071)//131072,'part count')
  for i,p in enumerate(x['parts']):
   fail(type(p['index']) is int and p['index']==i and p['path']==x['path']+'.parts/'+f'{i:04d}.bin','part order/path')
   fail(type(p['bytes']) is int and p['bytes']==min(131072,x['gzip_bytes']-i*131072),'part size')
 return {x['path']:x for x in s['files']}
def read(root,name):
 if name not in SHARDED:return raw_read(root,name)
 x=shard_index(root)[name];chunks=[]
 for p in x['parts']:
  b=raw_read(root,p['path']);fail(len(b)==p['bytes'] and digest(b)==p['sha256'],'part digest');chunks.append(b)
 b=b''.join(chunks);fail(len(b)==x['gzip_bytes'] and digest(b)==x['gzip_sha256'],'assembled gzip identity');return b
def load(root,name):return json.loads(read(root,name))
def payload(root,name):
 raw=read(root,name)
 if not name.endswith('.gz'):return raw
 import io
 with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
  data=f.read(32*1024*1024+1);fail(len(data)<=32*1024*1024,'expanded size budget')
  if name in SHARDED:
   x=shard_index(root)[name];fail(len(data)==x['expanded_bytes'] and digest(data)==x['expanded_sha256'],'expanded original identity')
  return data

def verify(root=BASE):
 root=pathlib.Path(root);manifest=load(root,'archive-manifest.json');fail(type(manifest['schema_version']) is int and manifest['schema_version']==1,'schema');fail(manifest['purpose']=='historical-evidence-integrity-not-data-catalog','purpose')
 shards=shard_index(root);parts={p['path'] for x in shards.values() for p in x['parts']};expected=(COPIED-SHARDED)|GENERATED|parts;files=manifest['files'];fail({x['path'] for x in files}==expected and len(files)==len(expected),'mandatory inventory')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts};fail(actual==expected|{'archive-manifest.json'},'unexpected inventory')
 for x in files:
  b=read(root,x['path'])
  if not x['path'].endswith('.gz') and x['path'] not in parts:b.decode('utf-8')
  fail(type(x['bytes']) is int and len(b)==x['bytes'] and digest(b)==x['sha256'],'file digest')
 origins=load(root,'copied-file-origins.json');fail({x['path'] for x in origins}==COPIED and len(origins)==len(COPIED),'origins coverage')
 for x in origins:
  b=payload(root,x['path']);b.decode('utf-8');fail(x['encoding']==('gzip' if x['path'].endswith('.gz') else 'identity'),'encoding');fail(type(x['original_bytes']) is int and len(b)==x['original_bytes'] and digest(b)==x['original_sha256'],'original bytes')
 origin_map={x['historical_path']:x for x in origins}
 specs=[load(root,'original-three-paths/run-spec.json'),load(root,'thread1-repeat/run-spec.json')]
 for spec in specs:
  fail(type(spec['schema_version']) is int and spec['schema_version']==1,'historical spec schema')
  for historical,h in spec['bindings'].items():
   if historical in origin_map:fail(origin_map[historical]['original_sha256']==h,'historical spec cross-binding')
 required=[x for x in origins if (x['path'].startswith(('original-three-paths/','thread1-repeat/')) and x['path'].endswith(('.py','PLAN.md'))) or x['path'] in ('data/source-input-identities.json.gz','runs/sherpa-api2/records.jsonl.gz')]
 for x in required:fail(any(spec['bindings'].get(x['historical_path'])==x['original_sha256'] for spec in specs),'required historical binding')
 data=json.loads(payload(root,'data/source-input-identities.json.gz'));fail(type(data['schema_version']) is int and data['schema_version']==1,'input schema');rows=data['rows'];ids=[r['recording'] for r in rows]
 fail(len(rows)==7006 and ids==sorted(set(ids)),'source coverage');fail(digest(('\n'.join(ids)+'\n').encode())=='5011b2faa1eb5bbb3212c65a48a88f384748af1b034a7517cb910d5e010bcc5a','selected source IDs')
 for r in rows:
  fail(type(r['frames']) is int and r['frames']>0 and r['split']=='observed_development' and r['label_authority']=='official-source-weak-text','source fields')
  fail(all(isinstance(r[k],str) and re.fullmatch('[0-9a-f]{64}',r[k]) for k in ['file_sha256','pcm_sha256']),'source digest field')
 fail(sum(r['frames'] for r in rows)==161688053 and len({r['pcm_sha256'] for r in rows})==7006 and data['duplicates']==[],'source frames/duplicates')
 fail(len({r['speaker_id'] for r in rows})==15 and len({r['official_text'] for r in rows})==12,'source breadth');sources={r['recording']:r for r in rows};prior_sherpa=None;report={}
 for route in ROUTES:
  recs=[json.loads(l) for l in payload(root,'runs/'+route+'/records.jsonl.gz').splitlines()];fail(len(recs)==7090,'run coverage');stages={'qwen_before':recs[:42],'himia':recs[42:7048],'qwen_after':recs[7048:]};groups={};counts={}
  for stage,rs in stages.items():
   fail(all(r['stage']==stage for r in rs) and len({r['recording'] for r in rs})==len(rs),'stage identity')
   counts[stage]=dict(clips=len(rs),events=0,triggered_clips=0,frames=0,positive_hits=0)
   for r in rs:
    ev=r['events'];json.dumps(ev,allow_nan=False);fail(all(type(e['keyword_id']) is int and e['keyword_id'] in (1,2) for e in ev),'event keyword')
    if stage=='himia':
     fail(r['recording'] in sources and all(r[k]==v for k,v in sources[r['recording']].items()),'source row identity');fail(ev==[],'retained zero-event result');dims=[('all','all'),('speaker',r['speaker_id']),('official_weak_text',r['official_text']),('source_speed',r['source_speed'])]
    else:dims=[('all','all'),('native_role',r['split']),('review',r['review_method'])]
    c=counts[stage];c['events']+=len(ev);c['triggered_clips']+=bool(ev);c['frames']+=r['frames'];c['positive_hits']+=int(r.get('kind')=='positive' and any(e['keyword_id']==r['keyword_id'] for e in ev))
    for dim,value in dims:
     g=groups.setdefault((stage,dim,value),collections.Counter());g['clips']+=1;g['events']+=len(ev);g['triggered_clips']+=bool(ev);g['frames']+=r['frames']
     for kw in [1,2]:g[f'keyword_{kw}_events']+=sum(e['keyword_id']==kw for e in ev)
     if r.get('kind')=='positive':
      hits=sum(e['keyword_id']==r['keyword_id'] for e in ev);g['positive_clips']+=1;g['positive_hits']+=bool(hits);g['wrong_keyword_events']+=len(ev)-hits;g['additional_target_events']+=max(0,hits-1)
     elif stage!='himia':g['confusable_clips']+=1;g['confusable_triggered_clips']+=bool(ev)
  before,after=stages['qwen_before'],stages['qwen_after'];identity=lambda r:{k:v for k,v in r.items() if k not in ['stage','wall_s']}
  fail([identity(r) for r in before]==[identity(r) for r in after],'Qwen pre/post parity');fail(sum(r['kind']=='positive' for r in before)==20 and counts['qwen_before']['positive_hits']=={'c':0,'sherpa-api2':19,'cfsmn':11,'sherpa-api1':19}[route],'Qwen known counts')
  done=load(root,'runs/'+route+'/complete.json');guard=load(root,'runs/'+route+'/supervisor.json');fail(done['status']==guard['status']=='completed' and type(guard['returncode']) is int and guard['returncode']==0 and guard['stop_reason'] is None,'completion');fail(done['counts']==counts,'saved counts')
  specpath=('thread1-repeat/' if route=='sherpa-api1' else 'original-three-paths/')+'run-spec.json'
  fail(done['spec_sha256']==digest(read(root,specpath)),'run spec binding')
  fail(done['qualification_allowed'] is False and done['source_label_only'] is True,'authority')
  for value,limit in [(done['resource_usage']['cpu_s'],900),(guard['wall_s'],1200),(guard['observed_group_peak_rss_bytes'],1024**3)]:fail(type(value) in (int,float) and math.isfinite(value) and 0<=value<=limit,'resource caps')
  saved=load(root,'runs/'+route+'/recomputed.json');fail(saved['raw_sha256']==digest(payload(root,'runs/'+route+'/records.jsonl.gz')),'raw binding');fail(saved['groups']==[dict(stage=k[0],dimension=k[1],value=k[2],**v) for k,v in sorted(groups.items())],'group counts')
  observations=[identity(r) for r in recs]
  if route=='sherpa-api2':prior_sherpa=observations
  if route=='sherpa-api1':
   fail(observations==prior_sherpa,'API1/API2 full observations parity');samples=json.loads(payload(root,'runs/'+route+'/worker-thread-samples.json.gz'));fail(bool(samples),'thread samples')
   fail(all(type(x['worker_threads']) is int and x['worker_threads']>=1 and type(x['worker_pid']) is int for x in samples),'thread fields');fail(len({x['worker_pid'] for x in samples})==1,'worker PID identity');fail(min(x['worker_threads'] for x in samples)==guard['worker_threads_min']==1 and max(x['worker_threads'] for x in samples)==guard['worker_threads_max']==1,'thread extrema')
  report[route]=dict(source_clips=7006,source_events=0,qwen_positive_hits=counts['qwen_before']['positive_hits'])
 return dict(verified=True,scope='retained compressed evidence identities/counts/parity only; no audio/inference or SSC305 verification',routes=report)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
