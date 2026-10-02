
import importlib, importlib.metadata, json, resource
from pathlib import Path
import torch
expected = json.loads(Path('/workspace/scratch/6c2ef8b5a46e/cloud-asr-dependency-probe/runtime-versions.json').read_text())
actual = {name: importlib.metadata.version(name) for name in expected}
assert actual == expected, 'installed package versions differ from complete closure'
plan = json.loads(Path('/workspace/scratch/6c2ef8b5a46e/cloud-asr-dependency-probe/build-plan.json').read_text())
modules = ['torch', 'numpy', 'safetensors', 'transformers', 'qwen_asr', 'funasr', 'kaldi_native_fbank'] + plan['required_native_imports']
if 'torchaudio' in expected:
    modules.append('torchaudio')
for name in modules:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(Path('/workspace/scratch/6c2ef8b5a46e/cloud-asr-dependency-probe/venv')), 'import escaped isolated environment'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
a = torch.arange(16, dtype=torch.float32, device='cpu').reshape(4, 4)
b = torch.eye(4, dtype=torch.float32, device='cpu')
assert torch.equal(a @ b, a)
x = a.to(torch.bfloat16)
assert torch.equal((x @ b.to(torch.bfloat16)).float(), a)
# Small representative operations only, not an ASR/KWS/TTS forward pass.
y = torch.nn.functional.layer_norm(x, (4,))
z = torch.nn.functional.conv1d(torch.ones(1, 2, 8, dtype=torch.bfloat16), torch.ones(2, 2, 3, dtype=torch.bfloat16))
assert y.dtype == torch.bfloat16 and z.dtype == torch.bfloat16
assert torch.isfinite(y).all() and torch.isfinite(z).all()
record = {'schema': 'kws.asr-runtime-cpu-probe.v1', 'event':'cpu_probe', 'imports': modules, 'torchaudio_status': 'imported_locked_package' if 'torchaudio' in expected else 'not_in_locked_knf_route', 'installed_versions': actual, 'torch_version': torch.__version__, 'torch_build_cuda': torch.version.cuda,
                 'device':'cpu', 'float32_matmul':True, 'bf16_matmul_layer_norm_conv1d':True,
                 'interpretation':'Tiny CPU operations only; no model compatibility or latency claim',
                 'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'target_speech_model_weights_loaded':0, 'user_audio_loaded':0, 'speech_model_forward_calls':0,
                 'dependency_import_side_effects':'Package-bundled tokenizers/statistical state may initialize; not counted as target ASR/TTS/KWS weights. No unlocked asset download permitted.'}
Path('/workspace/scratch/6c2ef8b5a46e/cloud-asr-dependency-probe/cpu-probe.json').write_text(json.dumps(record, sort_keys=True) + '\n')
print(json.dumps(record), flush=True)
