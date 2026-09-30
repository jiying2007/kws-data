"""Read-only recomputation from saved representations; no encoder run or updates."""
import hashlib,json,os,pathlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
import torch.nn.functional as F
R=pathlib.Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 spec=read(R/'spec.json');data=read(R/'dataset.json');summary=read(R/'results/summary.json');provenance=read(R/'results/provenance.json')
 for n,h in read(R/'preflight-lock.json').items():assert sha(R/n)==h
 for n,h in provenance['results_sha256'].items():assert sha(R/'results'/n)==h
 a=np.load(R/'results/pooled-normalization-head-artifacts.npz',allow_pickle=False)
 checks={}
 for name in ['P','R']:
  train=torch.from_numpy(a[f'{name}_train_pool']);mean=torch.from_numpy(a[f'{name}_norm_mean']);std=torch.from_numpy(a[f'{name}_norm_std']);raw=torch.from_numpy(a[f'{name}_norm_raw_std'])
  assert torch.equal(mean,train.mean(0));assert torch.equal(raw,train.std(0,correction=0));assert torch.equal(std,raw.clamp_min(1e-5))
  q=torch.from_numpy(a[f'{name}_qwen42_pool']);w=torch.from_numpy(a[f'{name}_head_weight']);b=torch.from_numpy(a[f'{name}_head_bias'])
  logits=F.linear((q-mean)/std,w,b);records=read(R/f'results/{name}-qwen42-readout.json');truth=torch.tensor([r['labels'] for r in records],dtype=torch.float32)
  saved=torch.tensor([r['logits'] for r in records]);assert torch.equal(logits,saved)
  loss=F.binary_cross_entropy_with_logits(logits,truth,reduction='none');assert torch.equal(loss,torch.tensor([r['unweighted_BCE_by_output'] for r in records]))
  assert (logits.sigmoid()>=.5).int().tolist()==[r['decisions'] for r in records]
  training=read(R/f'results/{name}-training.json');steps=[json.loads(l) for l in (R/f'results/{name}-steps.jsonl').read_text().splitlines()]
  assert [x['step'] for x in steps]==list(range(1,101)) and training['optimizer_updates']==100
  assert training['normalization_fit_recordings']==spec['train_recording_order']
  primary=[r for r in records if r['recipe_role']=='primary_observed_development']
  means=np.array([r['unweighted_BCE_by_output'] for r in primary]).mean(0).tolist()
  assert means==summary[name]['primary_unweighted_BCE_by_output']
  assert sum(r['both_bits_correct'] for r in primary)==summary[name]['primary_exact_clips']
  groups={}
  for split,method in sorted({(r['split'],r['review_method']) for r in records}):
   group=[r for r in records if r['split']==split and r['review_method']==method]
   groups[f'{split}/{method}']={'count':len(group),'both_bits_correct':sum(r['both_bits_correct'] for r in group),
     'unweighted_BCE_by_output':np.array([r['unweighted_BCE_by_output'] for r in group]).mean(0).tolist()}
  checks[name]={'saved_logits_BCE_decisions_bitwise_reproduced':True,'normalization_bitwise_reproduced_from_44TRAIN':True,
    'updates_exactly100':True,'pooled_Qwen42_readout_groups':groups,'first_step_weighted_train_BCE':steps[0]['weighted_train_BCE_before_update'],
    'last_step_weighted_train_BCE_before_update':steps[-1]['weighted_train_BCE_before_update']}
 g=read(R/'run-resource.json');assert g['success'] and g['peak_sampled']['threads']==1 and g['child_CPU_seconds']<600
 assert read(R/'git-precheck-run.json')['clean'] and read(R/'git-postcheck-run.json')['clean']
 f=read(pathlib.Path('/workspace/shared/kws-fleurs-train-preflight/local-source-package.json'));assert f['admitted_for_training'] is False
 result={'passed':True,'checks':checks,'source_and_frozen_locks_unchanged':True,'resource_guard_success':True,
  'read_only_scope':'saved pooled vectors and head only; no encoder inference, no optimizer, no data selection, no threshold change',
  'script_sha256':sha(pathlib.Path(__file__))}
 with open(R/'postrun-audit.json','x') as o:json.dump(result,o,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False);o.write('\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
