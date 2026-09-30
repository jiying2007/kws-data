"""Optional exact weight-only JSON restoration; requires declared research PyTorch.

CI never imports this module. No pickle deserialization, training or inference.
"""
import argparse
import hashlib
import json
import pathlib
import math
import struct
import torch


def restore(path):
    payload=json.loads(pathlib.Path(path).read_text())
    if (type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or
            payload.get('format')!='finite-float32-tensor-json-v1'):
        raise ValueError('unsupported weights format')
    state={}
    for name,tensor in payload['tensors'].items():
        values=tensor['values']
        shape=tensor['shape']
        if (tensor.get('dtype')!='float32' or not isinstance(shape,list) or not shape or
                not all(type(x) is int and x>0 for x in shape) or
                len(values)!=math.prod(shape) or
                not all(type(x) in (float,int) and math.isfinite(x) for x in values)):
            raise ValueError('invalid finite float32 tensor')
        raw=struct.pack('<'+str(len(values))+'f',*values)
        if hashlib.sha256(raw).hexdigest()!=tensor['little_endian_float32_sha256']:
            raise ValueError('tensor bytes mismatch')
        state[name]=torch.tensor(values,dtype=torch.float32).reshape(tensor['shape'])
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args=parser.parse_args()
    state=restore(args.checkpoint)
    print(json.dumps({'tensor_count':len(state),'parameters':sum(v.numel() for v in state.values())}))
