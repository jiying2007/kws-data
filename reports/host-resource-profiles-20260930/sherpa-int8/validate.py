import json,hashlib,pathlib
R=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/workspace/shared/kws-external-baseline')
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=j(R/'evidence/pinned-files.json'); files={x['Name']:x for x in f['Data']['Files']}; assert f['Code']==200
for d in j(R/'evidence/downloads.json'):
 p=R/d['path'];assert sha(p)==d['expected_sha256']==files[p.name]['Sha256'];assert p.stat().st_size==files[p.name]['Size']
q=j(R/'results/readback.json'); old=j(B/'results/readback.json');c=j(R/'config.json');oldc=j(B/'config.json')
assert {k:v for k,v in c.items() if k not in ('precision','encoder_file')}=={k:v for k,v in oldc.items() if k!='precision'}
assert sha(R/'keywords.txt')==sha(B/'keywords.txt')
assert len(q['recordings'])==42 and q['groups']==old['groups']
for x,y in zip(q['recordings'],old['recordings']):
 assert {k:v for k,v in x.items() if k!='events'}=={k:v for k,v in y.items() if k!='events'}
 assert [e['keyword_id'] for e in x['events']]==[e['keyword_id'] for e in y['events']]
p=j(R/'results/run-provenance.json')
for field,path in [('config_sha256','config.json'),('keywords_sha256','keywords.txt'),('script_sha256','run.py'),('readback_sha256','results/readback.json'),('downloads_sha256','evidence/downloads.json')]:assert p[field]==sha(R/path)
m=j(R/'resource-profile/measurement.json');raw=j(R/'resource-profile/raw-timings.json')
assert m['script_sha256']==sha(R/'resource-profile/profile.py')
assert m['config_source_sha256']==sha(R/'config.json')
assert m['asset_total_bytes']==5737545 and len(m['runs'])==3
assert m['config']['num_threads']==1
assert {k:v for k,v in m['config'].items() if k!='num_threads'}=={k:v for k,v in c.items() if k!='num_threads'}
for x in m['assets']:assert sha(R/'models'/x['file'])==x['sha256']
for phase in ['model_load','warmup','steady']:
 for pos in ['before','after']:assert m[phase][pos]['status']['Threads']=='1'
assert m['steady']['feed_plus_ready_decode']['count']==11784
for rr, timing in zip(m['runs'],raw):
 assert rr['clips']==42 and rr['events']==19 and len(timing['events'])==19
 assert all(not e['eof'] for e in timing['events'])
 assert [(e['recording'],e['phrase']) for e in timing['events']]==[(x['recording'],e['phrase']) for x in q['recordings'] for e in x['events']]
print('PASS: published asset hashes/sizes, unchanged decoding and labels, quality groups, provenance, one-thread resource protocol, all 3 repeated event identities')
