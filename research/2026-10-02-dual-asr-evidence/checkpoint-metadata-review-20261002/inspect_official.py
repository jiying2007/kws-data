from collections import OrderedDict, Counter
from pathlib import Path
import hashlib, json, os, resource
import torch
ROOT=Path('/workspace/scratch/6c2ef8b5a46e')
p=ROOT/'asr-model-metadata-lock-v1/verified-model-bodies-20261001T2348Z/FunAudioLLM--SenseVoiceSmall--3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/model.pt'
assert p.stat().st_size < 2*1024**3
h=hashlib.sha256()
with p.open('rb') as f:
    while chunk:=f.read(8*1024**2):h.update(chunk)
assert h.hexdigest()=='833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea'
s=torch.load(p,weights_only=True,map_location='cpu')
assert type(s) is OrderedDict
attrs=s.__dict__
report={'torch_version':str(torch.__version__), 'checkpoint_sha256':h.hexdigest(), 'checkpoint_class':type(s).__name__, 'checkpoint_keys':len(s), 'attribute_names':list(attrs), 'attribute_types':{k:type(v).__name__ for k,v in attrs.items()}}
assert set(attrs)=={'_metadata'}
m=attrs['_metadata']; assert type(m) in (dict,OrderedDict)
assert all(type(k) is str and type(v) is dict for k,v in m.items())
assert all(set(v)=={'version'} and type(v['version']) is int for v in m.values())
report['metadata']=dict(m)
report['metadata_entries']=len(m)
report['metadata_versions']=dict(Counter(v['version'] for v in m.values()))
report['metadata_sha256']=hashlib.sha256(json.dumps(dict(m),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
report['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(ROOT/'checkpoint-metadata-review-20261002/official-metadata.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='metadata'},sort_keys=True),flush=True)
