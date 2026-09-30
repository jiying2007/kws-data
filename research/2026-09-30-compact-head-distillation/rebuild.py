"""Overlay 846 head values on a separately supplied pinned donor; no inference."""
import argparse,hashlib,json,math,pathlib,struct
ROOT=pathlib.Path(__file__).resolve().parent
HEAD_NAMES={'backbone.out_linear2.linear.weight':(6,140),'backbone.out_linear2.linear.bias':(6,)}
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read_head(path):
 obj=json.loads(pathlib.Path(path).read_text())
 if type(obj.get('schema_version')) is not int or obj['schema_version']!=1 or obj.get('purpose')!='research-failed-distillation-head-only' or type(obj.get('parameter_count')) is not int or obj['parameter_count']!=846 or set(obj['tensors'])!=set(HEAD_NAMES):raise ValueError('Invalid head schema')
 for name,shape in HEAD_NAMES.items():
  item=obj['tensors'][name];values=item['values']
  if item['dtype']!='float32' or item['shape']!=list(shape) or len(values)!=math.prod(shape):raise ValueError('Invalid head tensor dimensions')
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Nonfinite/non-numeric head')
  data=struct.pack('<'+str(len(values))+'f',*values)
  if list(struct.unpack('<'+str(len(values))+'f',data))!=values:raise ValueError('Head numbers not exact finite float32')
  if hashlib.sha256(data).hexdigest()!=item['data_sha256']:raise ValueError('Head tensor digest mismatch')
 return obj

def reconstruct(donor_path,head_path=ROOT/'head-float32.json',proof_path=ROOT/'reconstruction-proof.json'):
 import torch
 head=read_head(head_path);proof=json.loads(pathlib.Path(proof_path).read_text())
 if sha(donor_path)!=head['donor_checkpoint_sha256'] or head['donor_checkpoint_sha256']!=proof['donor_checkpoint_sha256'] or head['original_full_checkpoint_sha256']!=proof['original_full_checkpoint_sha256']:raise ValueError('Donor/proof identity mismatch')
 state=torch.load(donor_path,weights_only=True,map_location='cpu')
 if set(state)!=set(proof['tensors']):raise ValueError('Donor tensor inventory mismatch')
 for name,item in head['tensors'].items():state[name]=torch.tensor(item['values'],dtype=torch.float32).reshape(item['shape'])
 for name,value in state.items():
  expected=proof['tensors'][name]
  if value.dtype!=torch.float32 or list(value.shape)!=expected['shape'] or not torch.isfinite(value).all():raise ValueError('Rebuilt tensor type/shape mismatch: '+name)
  values=value.flatten().tolist();data=struct.pack('<'+str(len(values))+'f',*values)
  if hashlib.sha256(data).hexdigest()!=expected['data_sha256']:raise ValueError('Rebuilt tensor hash mismatch: '+name)
 return state

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--donor',required=True);parser.add_argument('--output',help='optional new checkpoint path; refuses overwrite');args=parser.parse_args();state=reconstruct(args.donor)
 if args.output:
  import torch
  with open(args.output,'xb') as f:torch.save(state,f)
 print(json.dumps({'verified':True,'tensor_count':len(state),'head_parameters':846,'all_tensors_match_original_proof':True,'inference_executed':False,'serialization_bytes_need_not_match_original':True}))
