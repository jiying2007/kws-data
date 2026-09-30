"""Standard-library historical archive checks. No acoustic inference or training."""
import ast,collections,gzip,hashlib,io,json,math,pathlib,re,struct,zipfile,zlib
BASE=pathlib.Path(__file__).resolve().parent
GENERATED={'README.md','verify.py','test_verify.py','logical-files.json'}
SOURCE_INDEXES={'fft/public-scalar-evidence-index.json':'a9fb4bca9ffdbd762f1b669c6f645aaa511cc32160fce31d7067e890dff7c6a4','head/artifact-index.json':'4b38c24db83fd684e10696186fa3b7f9dbf7df55b6adf79605b747387d50f7b0'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def close(a,b,tol=2e-6):require(type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=tol,abs_tol=tol),'numeric mismatch')
def physical(root,name):
 p=pathlib.PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'unsafe path');q=root/name
 require(not root.is_symlink() and q.is_file() and not q.is_symlink(),'missing/symlink')
 for a in q.parents:
  require(not a.is_symlink(),'symlink parent')
 require(q.stat().st_size<=49152,'physical 48KiB cap');return q.read_bytes()
def strict_json(data):
 def pairs(items):
  d={}
  for k,v in items:
   require(k not in d,'duplicate JSON key');d[k]=v
  return d
 def number(s):
  v=float(s);require(math.isfinite(v),'nonfinite JSON number');return v
 def invalid(s):raise ValueError('nonfinite JSON constant')
 return json.loads(data,object_pairs_hook=pairs,parse_float=number,parse_constant=invalid)
def make_reader(root):
 index=strict_json(physical(root,'logical-files.json'));require(type(index['schema_version']) is int and index['schema_version']==1 and type(index['physical_limit_bytes']) is int and index['physical_limit_bytes']==49152,'mapping schema');entries=index['files'];require(len(entries)==80 and len({e['path'] for e in entries})==80,'logical coverage');mapping={e['path']:e for e in entries}
 def read(name):
  e=mapping[name];require(type(e['bytes']) is int and 0<e['bytes']<=1024*1024,'logical cap');require(type(e['encoded_bytes']) is int and 0<e['encoded_bytes']<=1024*1024,'encoded cap');parts=e['parts'];require(bool(parts) and len(parts)<=22,'parts cap');chunks=[]
  for i,p in enumerate(parts):
   expected=name if e['codec']=='identity' else 'objects/'+e['encoded_sha256']+'/'+f'{i:04d}.bin';require(p['path']==expected,'part path/order')
   b=physical(root,p['path']);require(type(p['bytes']) is int and len(b)==p['bytes'] and sha(b)==p['sha256'],'part identity');chunks.append(b)
  b=b''.join(chunks);require(len(b)==e['encoded_bytes'] and sha(b)==e['encoded_sha256'],'encoded identity')
  if e['codec']=='gzip-utf8':
   d=zlib.decompressobj(16+zlib.MAX_WBITS);b=d.decompress(b,1024*1024+1);require(d.eof and not d.unused_data and not d.unconsumed_tail,'single bounded gzip member')
  else:require(e['codec'] in ('identity','identity-npz'),'codec')
  require(len(b)==e['bytes'] and sha(b)==e['sha256'],'original logical identity')
  if e['kind']=='utf8':b.decode('utf8')
  else:require(e['kind']=='finite-derived-float32-npz' and name in ('head/head-parameters-and-normalization.npz','head/results/pooled-normalization-head-artifacts.npz'),'binary whitelist')
  return b
 return mapping,read

def arrays(raw,full):
 shapes={}
 for arm in 'PR':
  shapes.update({arm+'_head_weight.npy':(2,140),arm+'_head_bias.npy':(2,),arm+'_norm_mean.npy':(140,),arm+'_norm_std.npy':(140,),arm+'_norm_raw_std.npy':(140,)})
  if full:shapes.update({arm+'_train_pool.npy':(44,140),arm+'_qwen42_pool.npy':(42,140)})
 result={}
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  require(set(z.namelist())==set(shapes) and len(z.infolist())==len(shapes),'NPZ member whitelist')
  require(sum(i.file_size for i in z.infolist())<=256000,'NPZ expansion cap')
  for info in z.infolist():
   require(info.file_size<=25000 and info.compress_type in (0,8) and not info.flag_bits&1,'NPZ member cap/type');b=z.read(info);require(b[:8]==b'\x93NUMPY\x01\x00','NPY version');n=int.from_bytes(b[8:10],'little');require(n<=4096,'NPY header cap');h=ast.literal_eval(b[10:10+n].decode('ascii'));shape=shapes[info.filename]
   require(h=={'descr':'<f4','fortran_order':False,'shape':shape},'NPY float32 shape');count=math.prod(shape);data=b[10+n:];require(len(data)==4*count,'NPY payload length');values=struct.unpack('<'+str(count)+'f',data);require(all(math.isfinite(v) for v in values),'finite derived features');result[info.filename[:-4]]=(shape,values,data)
 return result

def check_fft(read):
 j=lambda n:strict_json(read('fft/'+n));r=j('result.json');p=j('preflight.json');status=j('execution-status.json')
 require(type(status['exit_code']) is int and status['exit_code']==1 and status['rerun'] is False and status['frontend_executions']==2 and 'TypeError' in status['error'],'retain serialization failure')
 for field,name in [('preflight_sha256','preflight.json'),('executed_script_sha256','run.py'),('summary_script_sha256','summarize_saved.py')]:require(r[field]==sha(read('fft/'+name)),'FFT source binding')
 require(len(r['cases'])==len(p['cases'])==2,'two-window limit')
 for i,c in enumerate(r['cases']):
  require(c['case']==p['cases'][i] and c['case']['end_sample']-c['case']['start_sample']==400,'fixed window identity')
  for layer in ('dc','preemphasis','windowed'):require(c['layers'][layer]=={'different_elements':0,'max_abs':0.0,'first_difference':None},'pre-FFT equality')
  for layer in ('power','mel','logfbank'):require(c['layers'][layer]['different_elements']>0,'retain mismatch')
  for side in ('reference','native'):
   h=c['historical_reproduction'][side];require(h['selected_old']==h['selected_now'],'selected historical reproduction')
  if i==1:require(c['historical_reproduction']['reference']['all80_bit_exact'] is False,'single-window limitation')
  energies=c['energies'];require(abs(energies['python']['stored_log']-energies['c']['stored_log'])>1e-3,'original log gate stays failed')
  for side,fft_side in [('python','python'),('c','c')]:
   e=energies[side];total=math.fsum(weight*c['fft'][k][side+'_power'] for k,(_,weight) in enumerate(c['mel_weights']));close(total,e['double_sum_saved_power'],1e-12);close(math.log(e['double_dft_mel']),e['double_dft_log'],1e-12)
   for f in c['fft']:
    z=f[fft_side];d=f['dft_'+side+'_window'];close(math.hypot(z[0]-d[0],z[1]-d[1]),f[side+'_abs_error_own_dft'],1e-12)
 return {'windows':2,'original_log_gate':'FAIL','correction_applied':False,'serialization_exit':1}

def check_head(read):
 j=lambda n:strict_json(read('head/'+n));spec=j('spec.json');data=j('dataset.json');lock=j('preflight-lock.json');approval=j('independent-approval.json');start=j('execution-started.json');provenance=j('results/provenance.json');summary=j('results/summary.json');review=j('independent-results-review.json')
 for name,h in lock.items():require(sha(read('head/'+name))==h,'preflight lock binding')
 require(approval['approved'] is True and approval['preflight_lock_sha256']==sha(read('head/preflight-lock.json')),'approval lock')
 require(start['one_authorized_pair_only'] is True and start['preflight_lock_sha256']==approval['preflight_lock_sha256'] and start['independent_approval_sha256']==sha(read('head/independent-approval.json')),'one-run chronology binding')
 require(provenance['approval_sha256']==start['independent_approval_sha256'] and provenance['preflight_lock_sha256']==start['preflight_lock_sha256'],'result provenance')
 for name,h in provenance['results_sha256'].items():require(sha(read('head/results/'+name))==h,'saved result binding')
 for name,h in review['bindings'].items():require(sha(read('head/'+name))==h,'result review binding')
 for x in j('reference-source-index.json'):
  require(sha(read('head/'+x['path']))==x['sha256'],'upstream identity')
  if x['original_path'] in spec['source_paths_sha256']:require(x['sha256']==spec['source_paths_sha256'][x['original_path']],'upstream spec identity')
 require(len(data)==74 and len({x['recording'] for x in data})==74,'74 source identities');train=[x for x in data if x['recipe_role']=='train'];dev=[x for x in data if x['recipe_role']=='primary_observed_development'];qwen=[x for x in data if x['source']=='Qwen'];fleurs=[x for x in data if x['source']=='FLEURS']
 require(len(train)==44 and len(dev)==8 and len(qwen)==42 and len(fleurs)==32,'fixed data roles');require([x['recording'] for x in train]==spec['train_recording_order'],'TRAIN order')
 for x in fleurs:require(x['source_receipt_admitted_for_training'] is False and x['source_record']['admitted_for_training'] is False and x['human_reviewed_by_us'] is False and x['recipe_role']=='train' and x['labels']==[0,0] and bool(x['recipe_specific_authorization']),'recipe-only FLEURS scope')
 require(sum(x['source']=='Qwen' and x['review_method']=='human' for x in train)==12,'human TRAIN')
 require(spec['optimizer']['steps']==100 and spec['trainable_parameters_per_arm']==282 and spec['evaluation']['threshold']==.5 and spec['evaluation']['qualification_allowed'] is False and spec['normalization']['dev_updates'] is False,'fixed recipe')
 require('AttributeError' in read('head/preflight-revisions/preflight.log').decode() and 'AssertionError' in read('head/preflight-revisions/tests.log').decode(),'earlier failures retained')
 small=arrays(read('head/head-parameters-and-normalization.npz'),False);full=arrays(read('head/results/pooled-normalization-head-artifacts.npz'),True)
 for k in small:require(small[k]==full[k],'head-only tensor agreement')
 scores={};source={x['recording']:x for x in data};groups={}
 for arm in 'PR':
  rows=j('results/'+arm+'-qwen42-readout.json');require(len(rows)==42 and [x['recording'] for x in rows]==[x['recording'] for x in qwen],'Qwen readout order')
  history=j('results/'+arm+'-training.json');require(type(history['optimizer_updates']) is int and history['optimizer_updates']==100,'100 updates');steps=[strict_json(l) for l in read('head/results/'+arm+'-steps.jsonl').splitlines()];require([x['step'] for x in steps]==list(range(1,101)) and steps==history['training_history'],'100 recorded steps');require(history['normalization_fit_recordings']==spec['train_recording_order'],'normalization TRAIN-only')
  vals=lambda k:full[arm+'_'+k][1];mean=vals('norm_mean');std=vals('norm_std');rawstd=vals('norm_raw_std');pool=vals('train_pool');weights=vals('head_weight');bias=vals('head_bias');readout=vals('qwen42_pool')
  for i in range(140):
   mu=math.fsum(pool[k*140+i] for k in range(44))/44;sd=math.sqrt(math.fsum((pool[k*140+i]-mu)**2 for k in range(44))/44);close(mean[i],mu,2e-5);close(rawstd[i],sd,2e-5);close(std[i],max(rawstd[i],1e-5),1e-7)
  primary=[]
  for i,row in enumerate(rows):
   src=source[row['recording']]
   for k in ['labels','recipe_role','review_method','speaker_id','split','intended_text']:require(row[k]==src[k],'preserve native label/role')
   require(all(type(x) is int and x in (0,1) for x in row['labels']+row['decisions']),'typed bits');logits=[math.fsum(((readout[i*140+k]-mean[k])/std[k])*weights[o*140+k] for k in range(140))+bias[o] for o in range(2)]
   for o,x in enumerate(logits):
    close(x,row['logits'][o],2e-5);bce=max(x,0)-row['labels'][o]*x+math.log1p(math.exp(-abs(x)));close(bce,row['unweighted_BCE_by_output'][o],2e-5);require(row['decisions'][o]==int(row['logits'][o]>=0),'fixed threshold')
   exact=row['decisions']==row['labels'];require(row['both_bits_correct'] is exact,'both-bit decision');g=groups.setdefault((arm,row['split'],row['review_method']),[0,0]);g[0]+=1;g[1]+=exact
   if row['recipe_role']=='primary_observed_development':primary.append(row)
  require(len(primary)==8,'primary clips');exact=sum(x['both_bits_correct'] for x in primary);bits=sum(a==b for r in primary for a,b in zip(r['labels'],r['decisions']));means=[math.fsum(r['unweighted_BCE_by_output'][o] for r in primary)/8 for o in range(2)];scores[arm]=means
  require(exact=={'P':4,'R':1}[arm]==summary[arm]['primary_exact_clips'] and bits==summary[arm]['primary_exact_bits'],'strict development failure counts')
  for o in range(2):close(means[o],summary[arm]['primary_unweighted_BCE_by_output'][o],1e-12)
  require(groups[(arm,'train','human')]==[12,{'P':0,'R':3}[arm]],'human TRAIN failures visible')
 gates=summary['gates'];require(all(scores['P'][i]<scores['R'][i] for i in range(2)) and gates['representation_advantage_both_outputs'] is True and gates['P_lower_BCE_than_R_by_output']==[True,True],'narrow BCE comparison');require(gates['P_primary_all8_exact'] is False and gates['R_primary_all8_exact'] is False and gates['streaming_or_product_qualification'] is False,'no strict/product promotion')
 guard=j('run-resource.json');require(type(guard['exit_code']) is int and guard['exit_code']==0 and guard['success'] is True and guard['stop_reason'] is None and 0<guard['child_CPU_seconds']<600,'resource closure')
 return {'P_dev_exact':4,'R_dev_exact':1,'dev_clips':8,'P_human_TRAIN_exact':0,'human_TRAIN_clips':12,'strict_gates':'FAIL/FAIL','BCE_comparison':'P lower on both outputs','feature_scope':'finite derived pools and small trained heads; no donor weights'}

def verify(root=BASE):
 root=pathlib.Path(root);manifest=strict_json(physical(root,'archive-manifest.json'));require(type(manifest['schema_version']) is int and manifest['schema_version']==1 and manifest['purpose']=='historical-evidence-integrity-not-data-catalog','manifest schema');mapping,read=make_reader(root);expected=GENERATED|{p['path'] for e in mapping.values() for p in e['parts']};files=manifest['files'];require({x['path'] for x in files}==expected and len(files)==len(expected),'physical inventory');actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts};require(actual==expected|{'archive-manifest.json'},'unexpected file')
 for f in files:
  b=physical(root,f['path'])
  if not f['path'].endswith(('.bin','.npz')):b.decode('utf8')
  require(type(f['bytes']) is int and len(b)==f['bytes'] and sha(b)==f['sha256'],'physical digest')
 for name in mapping:read(name)
 names=set(SOURCE_INDEXES)
 for name,h in SOURCE_INDEXES.items():
  b=read(name);require(sha(b)==h,'original approved index');index=strict_json(b);prefix=name.split('/')[0]
  for f in index['files']:
   key=prefix+'/'+f['path'];b=read(key);require(len(b)==f['bytes'] and sha(b)==f['sha256'],'original whitelist binding');names.add(key)
 require(set(mapping)==names,'no extra logical payload')
 return {'verified':True,'logical_originals':len(mapping),'fft':check_fft(read),'frozen_head':check_head(read),'scope':'retained byte/count/scalar arithmetic checks; no acoustic replay or training'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
