"""Local authoring utility; weights_only load, export only the two trained tensors."""
import argparse,hashlib,json,math,pathlib
import torch
HEAD=('backbone.out_linear2.linear.weight','backbone.out_linear2.linear.bias')
DONOR_SHA='d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233'
FINAL_SHA='b61fed9ef590a9c922ee2a6ddc6584b3f29330c45bd4c07d386db459a5514bba'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tensor_record(value):
 assert value.dtype==torch.float32 and torch.isfinite(value).all()
 return {'shape':list(value.shape),'dtype':'float32','data_sha256':hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()}
def export(donor_path,checkpoint_path,destination):
 assert sha(donor_path)==DONOR_SHA and sha(checkpoint_path)==FINAL_SHA
 donor=torch.load(donor_path,weights_only=True,map_location='cpu');final=torch.load(checkpoint_path,weights_only=True,map_location='cpu');assert set(donor)==set(final)
 tensors={};head={}
 for key,value in final.items():
  tensors[key]=tensor_record(value)
  if key in HEAD:
   head[key]=dict(tensors[key],values=value.flatten().tolist())
   rebuilt=torch.tensor(head[key]['values'],dtype=torch.float32).reshape(value.shape);assert torch.equal(rebuilt,value)
  else:assert torch.equal(value,donor[key]),key
 assert sum(len(v['values']) for v in head.values())==846
 obj={'schema_version':1,'purpose':'research-failed-distillation-head-only','parameter_count':846,'encoding':'JSON finite numbers exactly round-trip to IEEE754 float32; tensor order row-major','donor_checkpoint_sha256':DONOR_SHA,'original_full_checkpoint_sha256':FINAL_SHA,'tensors':head}
 proof={'schema_version':1,'original_full_checkpoint_sha256':FINAL_SHA,'original_full_checkpoint_bytes':pathlib.Path(checkpoint_path).stat().st_size,'donor_checkpoint_sha256':DONOR_SHA,'all_non_head_tensors_equal_donor':True,'head_float32_roundtrip_exact':True,'tensor_count':len(tensors),'total_model_parameters':390520,'tensors':tensors,'scope':'exact tensor identity; PyTorch serialization-container bytes are not expected to reproduce original checkpoint SHA'}
 destination=pathlib.Path(destination);(destination/'head-float32.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');(destination/'reconstruction-proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--donor',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--destination',required=True);a=p.parse_args();export(a.donor,a.checkpoint,a.destination)
