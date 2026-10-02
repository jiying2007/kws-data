from collections import Counter
import hashlib, inspect, json, sys
from pathlib import Path
ROOT=Path('/workspace/scratch/6c2ef8b5a46e')
sys.path.insert(0,str(ROOT/'asr-calibration-stage-v1'))
from asr_stage.runtime_support import require_runtime_versions, verify_source_members
from asr_stage.architecture import canonical_sha,validate_sense_geometry,sensevoice_schema
versions=require_runtime_versions()
sources=json.loads((ROOT/'asr-calibration-stage-v1/locks/sensevoice-package-source.candidate.json').read_text())
verify_source_members(sources,canonical_sha(sources),'FunAudioLLM/SenseVoiceSmall')
import torch,yaml
from funasr.models.sense_voice.model import SenseVoiceSmall
p=ROOT/'asr-model-metadata-lock-v1/verified-model-bodies-20261001T2348Z/FunAudioLLM--SenseVoiceSmall--3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/config.yaml'
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19'
data=yaml.safe_load(raw);validate_sense_geometry(data)
kwargs=dict(data['model_conf']);kwargs.update({name:data[name] for name in ('encoder','encoder_conf','specaug','specaug_conf') if name in data});kwargs.update(input_size=560,vocab_size=25055)
with torch.device('meta'):model=SenseVoiceSmall(**kwargs)
state=model.state_dict();assert all(t.is_meta for t in state.values());assert {k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in state.items()}==sensevoice_schema()
metadata=dict(state._metadata)
official=json.loads((ROOT/'checkpoint-metadata-review-20261002/official-metadata.json').read_text())['metadata']
classes={};hook_modules=[]
for name,module in model.named_modules(remove_duplicate=False):
    kind=type(module)
    if kind not in classes:
        loader=kind._load_from_state_dict
        path=Path(inspect.getsourcefile(kind));loadpath=Path(inspect.getsourcefile(loader))
        classes[kind]={'class':kind.__module__+'.'+kind.__qualname__,'version':module._version,'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'loader':loader.__module__+'.'+loader.__qualname__,'loader_source_sha256':hashlib.sha256(loadpath.read_bytes()).hexdigest()}
    if any(getattr(module,attr) for attr in ('_state_dict_hooks','_state_dict_pre_hooks','_load_state_dict_pre_hooks','_load_state_dict_post_hooks')):hook_modules.append(name)
report={'versions':versions,'metadata':metadata,'metadata_entries':len(metadata),'metadata_sha256':canonical_sha(metadata),'checkpoint_only':{k:official[k] for k in set(official)-set(metadata)},'architecture_only':{k:metadata[k] for k in set(metadata)-set(official)},'changed':{k:{'architecture':metadata[k],'checkpoint':official[k]} for k in set(metadata)&set(official) if metadata[k]!=official[k]},'classes':list(classes.values()),'hook_modules':hook_modules,'forward_executed':False,'checkpoint_read':False}
(ROOT/'checkpoint-metadata-review-20261002/architecture-metadata.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','classes')},sort_keys=True),flush=True)
