import collections,hashlib,json,pathlib,subprocess,sys,wave,platform
import numpy as np
import sherpa_onnx
ROOT=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path('/workspace/shared/kws-data-pinned')
RECEIPT=pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback/native-export-receipt.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'config.json')=='c7c14a26522b0dacc0846e03ba98c68aa6624e91e81df3eddaf20b8c69cca6c4'
assert sha(ROOT/'keywords.txt')=='bdbeeb12c2b95b9a71f09994024c8ef86d88bdaa8146b758116d48471fd91e1a'
assert sha(RECEIPT)=='8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5'
assert subprocess.check_output(['git','-C',str(DATA),'rev-parse','HEAD'],text=True).strip()=='2f9658ffa9568076ef547615c76861abed84f56e'
assert not subprocess.check_output(['git','-C',str(DATA),'status','--porcelain'],text=True).strip()
config=json.loads((ROOT/'config.json').read_text());r=json.loads(RECEIPT.read_text())
rows=[dict(x,dataset_id=d['dataset_id']) for d in r['datasets'] for x in d['recordings']]
assert len(rows)==42
for d in json.loads((ROOT/'evidence/downloads.json').read_text()):assert sha(ROOT/d['path'])==d['expected_sha256']
vocab={line.split()[0] for line in (ROOT/'models/tokens.txt').read_text().splitlines()}
for line in (ROOT/'keywords.txt').read_text().splitlines():assert set(line.split()[:-1])<=vocab
kw={x:config[x] for x in ['num_threads','sample_rate','feature_dim','max_active_paths','keywords_score','keywords_threshold','num_trailing_blanks','provider']}
for comp in ['encoder','decoder','joiner']:kw[comp]=str(ROOT/'models'/f'{comp}-epoch-12-avg-2-chunk-16-left-64.onnx')
kw.update(tokens=str(ROOT/'models/tokens.txt'),keywords_file=str(ROOT/'keywords.txt'))
kws=sherpa_onnx.KeywordSpotter(**kw)
phrase_id={'你好小窝':1,'小窝小窝':2};outputs=[]
for row in rows:
 path=DATA/row['path'];assert sha(path)==row['file_sha256']
 with wave.open(str(path)) as w:
  assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,row['frames'])
  pcm=w.readframes(w.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
 samples=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0
 stream=kws.create_stream();events=[]
 def drain(fed,eof=False):
  while kws.is_ready(stream):
   kws.decode_stream(stream);phrase=kws.get_result(stream)
   if phrase:
    events.append({'phrase':phrase,'keyword_id':phrase_id[phrase],'available_audio_samples':fed,'eof_flush':eof,'tokens':kws.tokens(stream),'token_timestamps':kws.timestamps(stream)})
    kws.reset_stream(stream)
 for start in range(0,len(samples),config['feed_samples_per_block']):
  end=min(len(samples),start+config['feed_samples_per_block']);stream.accept_waveform(16000,samples[start:end]);drain(end)
 stream.input_finished();drain(len(samples),True)
 positive=row['kind']=='positive';target=sum(e['keyword_id']==row['keyword_id'] for e in events) if positive else 0
 out=dict(row,events=events,target_hit=bool(target) if positive else None,target_miss=not bool(target) if positive else None,wrong_keyword_events=len(events)-target if positive else 0,additional_target_events=max(0,target-1) if positive else 0,confusable_events=len(events) if not positive else 0)
 outputs.append(out);print(json.dumps({'recording':row['recording'],'events':events},ensure_ascii=False),flush=True)
groups={}
for row in outputs:
 for key in [('all','all'),('role',row['split']),('review',row['review_method']),('dataset',row['dataset_id']),('keyword',str(row['keyword_id']))]:
  g=groups.setdefault(key,collections.Counter());g['clips']+=1;g['events']+=len(row['events'])
  if row['kind']=='positive':g['positives']+=1;g['target_hits']+=row['target_hit'];g['target_misses']+=row['target_miss'];g['wrong_keyword_events']+=row['wrong_keyword_events'];g['additional_target_events']+=row['additional_target_events']
  else:g['confusables']+=1;g['confusable_clips_with_events']+=bool(row['events']);g['confusable_events']+=len(row['events'])
report={'config':config,'groups':[{'dimension':k[0],'value':k[1],**v} for k,v in groups.items()],'recordings':outputs}
(ROOT/'results/readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(ROOT/'results/raw-events.jsonl').write_text(''.join(json.dumps({'recording':x['recording'],**e},ensure_ascii=False)+'\n' for x in outputs for e in x['events']))
receipt={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sherpa_onnx':sherpa_onnx.__version__,'config_sha256':sha(ROOT/'config.json'),'keywords_sha256':sha(ROOT/'keywords.txt'),'script_sha256':sha(ROOT/'run.py'),'native_export_sha256':sha(RECEIPT),'data_commit':r['data_repository_commit'],'readback_sha256':sha(ROOT/'results/readback.json'),'downloads_sha256':sha(ROOT/'evidence/downloads.json')}
(ROOT/'results/run-provenance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(report['groups'],ensure_ascii=False,indent=2))
