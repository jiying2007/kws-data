#!/usr/bin/env python3
"""Compare installed no-weight meta constructors to independent schemas.

Run exactly one model in a fresh supervised process. No model checkpoint is read
or downloaded. Config and package source members must match retained source locks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from asr_stage.architecture import (canonical_sha, qwen06_schema, sensevoice_schema,
    validate_qwen_geometry, validate_sense_geometry)
from asr_stage.runtime_support import require_runtime_versions, verify_source_members, assert_effective_qwen_attention
from asr_stage.execution import require_offline_flags, retain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen','sense'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    config_raw = Path(args.config).read_bytes()
    if len(config_raw) > 1024*1024 or hashlib.sha256(config_raw).hexdigest() != args.config_sha256:
        raise ValueError('Config identity mismatch')
    mid = 'Qwen/Qwen3-ASR-0.6B' if args.model == 'qwen' else 'FunAudioLLM/SenseVoiceSmall'
    prefix = 'qwen06' if args.model == 'qwen' else 'sensevoice'
    sources = json.loads((ROOT/'locks'/f'{prefix}-package-source.candidate.json').read_bytes())
    versions = require_runtime_versions()
    verify_source_members(sources, canonical_sha(sources), mid)
    environment = require_offline_flags()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('Default dtype changed')
    if args.model == 'qwen':
        import qwen_asr
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        from transformers import AutoModel
        data = json.loads(config_raw)
        validate_qwen_geometry(data)
        config = Qwen3ASRConfig.from_dict(data)
        for cfg in (config, config.thinker_config, config.thinker_config.audio_config, config.thinker_config.text_config):
            cfg._attn_implementation = 'eager'
            cfg.dtype = torch.float32
        with torch.device('meta'):
            model = AutoModel.from_config(config, trust_remote_code=False, dtype=torch.float32)
        attention = assert_effective_qwen_attention(model)
        expected = qwen06_schema()
    else:
        import yaml
        from funasr.models.sense_voice.model import SenseVoiceSmall
        data = yaml.safe_load(config_raw)
        validate_sense_geometry(data)
        kwargs = dict(data['model_conf'])
        kwargs.update({name:data[name] for name in ('encoder','encoder_conf','specaug','specaug_conf') if name in data})
        kwargs.update(input_size=560, vocab_size=25055)
        with torch.device('meta'):
            model = SenseVoiceSmall(**kwargs)
        attention = None
        expected = sensevoice_schema()
    observed = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor) or not value.is_meta:
            raise RuntimeError('Architecture probe allocated non-meta retained state')
        observed[key] = {'shape':list(value.shape), 'dtype':str(value.dtype)}
    if observed != expected:
        missing = sorted(set(expected)-set(observed))
        extra = sorted(set(observed)-set(expected))
        changed = {k:{'expected':expected[k],'observed':observed[k]} for k in set(expected)&set(observed) if expected[k] != observed[k]}
        raise RuntimeError(json.dumps({'missing':missing,'extra':extra,'changed':changed}))
    output = Path(args.output_directory)
    output.mkdir(exist_ok=False)
    result = {'schema':'asr-installed-architecture-qualification-v1','model_id':mid,
        'status':'PASS','config_sha256':args.config_sha256,'source_lock_sha256':canonical_sha(sources),
        'expected_schema_sha256':canonical_sha(expected),'observed_schema_sha256':canonical_sha(observed),
        'state_keys':len(observed),'versions':versions,'environment':environment,
        'effective_attention':attention,'checkpoint_read':False,'forward_executed':False,
        'scope':'No-weight installed meta constructor shape/dtype comparison; tokenizer/checkpoint body and real forward remain separate'}
    retain(output, 'receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
