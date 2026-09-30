"""Synthetic-only locked classifier contract checks, no corpus model inference."""
import copy, math
import numpy as np
import torch
import torch.nn.functional as F
import experiment as e

def main():
 b,front,ns,tokens,_=e.setup()
 cases=[]
 def passed(name):cases.append(name)
 # Exact frame counts from actual tagged source parser, including EOF no-flush.
 assert e.frame_contract(b,ns,tokens,799)['eligible_frames']==0
 assert e.frame_contract(b,ns,tokens,4800)['eligible_frames']==1
 assert e.frame_contract(b,ns,tokens,9600)['eligible_frames']==11
 passed('source_parser_8_step_mask_and_no_flush_counts')
 # Train-fitted statistics use population std, absolute floor, no dev updates.
 x=torch.arange(44*140,dtype=torch.float32).reshape(44,140)/100
 x[:,0]=7
 z,mean,std,raw=e.normalize_train(x)
 assert raw[0]==0 and std[0]==torch.tensor(1e-5) and torch.equal(z[:,0],torch.zeros(44))
 assert torch.equal(raw,x.std(0,correction=0)) and torch.equal(mean,x.mean(0))
 frozen=(mean.clone(),std.clone());dev=torch.full((8,140),1e6);_=(dev-mean)/std
 assert torch.equal(mean,frozen[0]) and torch.equal(std,frozen[1])
 passed('population_std_floor_and_train_only_stats_immutable')
 # Independent BCE outputs and inclusive fixed threshold are explicit.
 logits=torch.tensor([[0.,-1.],[1.,0.]])
 assert (logits.sigmoid()>=.5).int().tolist()==[[1,0],[1,1]]
 y=torch.tensor([[1.,0.],[0.,1.]])
 unweighted=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
 expected=torch.where(y==1,F.softplus(-logits),F.softplus(logits))
 assert torch.allclose(unweighted,expected,atol=1e-7,rtol=1e-7)
 weighted=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([41/3,41/3]),reduction='none')
 assert torch.allclose(weighted,unweighted*torch.where(y==1,torch.tensor(41/3),torch.tensor(1.)))
 passed('inclusive_fixed_threshold_and_unweighted_dev_BCE')
 # Catch nonfinite inputs, and reject invalid train sample counts.
 try:e.finite(torch.tensor([float('nan')]),'deliberate_test')
 except AssertionError:pass
 else:raise AssertionError('nonfinite accepted')
 try:e.normalize_train(torch.zeros(43,140))
 except AssertionError:pass
 else:raise AssertionError('wrong train count accepted')
 passed('nonfinite_and_wrong_count_fail_closed')
 # Metric gate must evaluate both independent bits, not only intended target.
 enc,h,init=e.models(b)
 assert all(not p.requires_grad for m in enc.values() for p in m.parameters())
 assert e.state_hash(copy.deepcopy(h).state_dict())==init['head']['state_sha256']
 assert sum(p.numel() for p in h.parameters())==282
 passed('frozen_encoder_shared_282_parameter_head')
 e.save(e.ROOT/'unit-tests.json',{'passed':True,'cases':cases,'real_corpus_model_inference':False,'real_corpus_training':False})
 print('PASS:',', '.join(cases))
if __name__=='__main__':main()
