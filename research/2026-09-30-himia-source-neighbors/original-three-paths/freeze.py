"""Bind inputs and existing model artifacts, no model execution."""
import json,pathlib,subprocess,sys
from driver import ROOT,C,B,S,F,sha
files=[ROOT/'PLAN.md',ROOT/'prepare.py',ROOT/'inputs.json',ROOT/'driver.py',ROOT/'test_driver.py',ROOT/'freeze.py',ROOT/'summarize.py',B/'readback-spec.json',B/'manual-build-receipt.json',B/'native-export-receipt.json',B/'kws_wav',S/'config.json',S/'keywords.txt',S/'run.py',S/'evidence/downloads.json',F/'run_baseline.py',F/'baseline-spec.json']
files += list(pathlib.Path('/workspace/shared/kws-external-baseline/venv/lib/python3.12/site-packages/sherpa_onnx/lib').glob('*.so'))
files += list((C/'models/registry/model-749187ec1d66').glob('*.kwm'))+list((C/'models/registry/model-749187ec1d66').glob('*.kwk'))
for r in json.loads((S/'evidence/downloads.json').read_text()):
 p=S/r['path'];assert sha(p)==r['expected_sha256'];files.append(p)
fs=json.loads((F/'baseline-spec.json').read_text())
for p,h in fs['sources'].items():assert sha(F/p)==h;files.append(F/p)
assert sha(F/'run_baseline.py')==fs['driver_sha256']
refs={'c':str(B/'run-qwen42/clip-readback.json'),'sherpa':str(S/'results/readback.json'),'cfsmn':str(F/'results/hamming-raw.json')};files += [pathlib.Path(p) for p in refs.values()]+[B/'run-qwen42/detections.jsonl']
assert (B/'run-qwen42/detections.jsonl').read_text().strip()==''
paths={'c':sys.executable,'sherpa':'/workspace/shared/kws-external-baseline/venv/bin/python','cfsmn':'/workspace/shared/kws-lightweight-prototypes/venv/bin/python'}
runtimes={k:{'executable':p,'python':subprocess.check_output([p,'-c','import sys;print(sys.version,end="")'],text=True)} for k,p in paths.items()}
spec=dict(schema_version=1,scope='source-labelled-neighbor-clip-events-observed-development',bindings={str(p):sha(p) for p in sorted(set(files))},reference_files=refs,runtimes=runtimes,model_training=False,qualification_allowed=False,routes=['c','sherpa','cfsmn'],counts=dict(himia=7006,qwen_before=42,qwen_after=42),limits=dict(route_cpu_s=900,route_wall_s=1200,clip_wall_s=10,rss_bytes=1073741824,poll_interval_s=.05),ready_for_review=True)
with (ROOT/'run-spec.json').open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
print(sha(ROOT/'run-spec.json'))
