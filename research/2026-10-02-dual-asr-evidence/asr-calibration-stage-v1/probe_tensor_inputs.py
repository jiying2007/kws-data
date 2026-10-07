"""Tiny installed-tensor/processor test; no ASR model or recorded clip used."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from asr_stage.runtime_support import describe_cpu_tensor, verify_pcm_array

parser=argparse.ArgumentParser()
parser.add_argument('--qwen-assets')
args=parser.parse_args()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
checked = []
for dtype in (torch.float32,torch.float64,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
    for value in (torch.zeros(0,dtype=dtype),torch.tensor(0,dtype=dtype),torch.zeros((2,3),dtype=dtype)):
        row=describe_cpu_tensor(value,True)
        expected=hashlib.sha256(memoryview(value.detach().reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()
        assert row['sha256']==expected
        checked.append({'shape':list(value.shape),'dtype':str(dtype)})
large=torch.zeros(262145,dtype=torch.float32)
describe_cpu_tensor(large,True)
large[-1]=float('nan')
try:describe_cpu_tensor(large,True)
except RuntimeError:pass
else:raise AssertionError('Nonfinite second chunk accepted')
extractor=WhisperFeatureExtractor(feature_size=128,return_attention_mask=True)
inputs=extractor([np.zeros(1600,dtype=np.float32)],sampling_rate=16000,
                 return_tensors='pt',padding=True,return_attention_mask=True)
layouts={}
for key,value in inputs.items():
    layouts[key]={'original_stride':list(value.stride()),'original_contiguous':value.is_contiguous()}
    converted=value.contiguous()
    assert torch.equal(value,converted)
    describe_cpu_tensor(converted,True)
qwen_layouts=None
if args.qwen_assets:
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(args.qwen_assets,fix_mistral_regex=True,
        local_files_only=True,trust_remote_code=False)
    from types import SimpleNamespace
    dummy=SimpleNamespace(device=torch.device('cpu'),dtype=torch.float32)
    wrapper=Qwen3ASRModel(backend='transformers',model=dummy,processor=processor,
        forced_aligner=None,max_inference_batch_size=1,max_new_tokens=256)
    prompt=wrapper._build_text_prompt(context='',force_language=None)
    values=processor(text=[prompt],audio=[np.zeros(1600,dtype=np.float32)],
        sampling_rate=16000,return_tensors='pt',padding=True)
    values=values.to('cpu').to(torch.float32)
    qwen_layouts={}
    for key,value in values.items():
        converted=value.contiguous()
        assert torch.equal(value,converted)
        qwen_layouts[key]={'original_stride':list(value.stride()),
            'original_contiguous':value.is_contiguous(),'descriptor':describe_cpu_tensor(converted,True)}
print(json.dumps({'status':'PASS','tiny_tensor_cases':len(checked),
    'chunked_nonfinite_rejection':True,'whisper_processor_layouts':layouts,
    'actual_qwen_processor_layouts':qwen_layouts,
    'contiguous_conversion_exact':True,'ASR_weights_loaded':False,
    'ASR_model_forward':False,'recorded_clip_consumed':False}),flush=True)
