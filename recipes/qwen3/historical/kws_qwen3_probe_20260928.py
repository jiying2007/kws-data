import hashlib
import json
import pathlib
import math
import time
import wave

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from qwen_tts import Qwen3TTSModel

ROOT = pathlib.Path('/work/build/qwen-tts-20260928')
MODEL = ROOT / 'model'
OUT = ROOT / 'samples'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(1337)
np.random.seed(1337)

started = time.monotonic()
model = Qwen3TTSModel.from_pretrained(
    str(MODEL), device_map='cpu', dtype=torch.float32,
    attn_implementation='eager', local_files_only=True,
)
loaded_s = time.monotonic() - started
started = time.monotonic()
wavs, rate = model.generate_custom_voice(
    text='小窝小窝', speaker='Vivian', language='Chinese',
    do_sample=True, max_new_tokens=120,
)
generated_s = time.monotonic() - started
if len(wavs) != 1:
    raise ValueError('unexpected number of generated waveforms')
raw = OUT / 'qwen3-kw2-vivian-raw.wav'
normalized = OUT / 'qwen3-kw2-vivian-16k.wav'
sf.write(str(raw), wavs[0], rate, subtype='PCM_16')
divisor = math.gcd(rate, 16000)
samples_16k = resample_poly(np.asarray(wavs[0], dtype=np.float64),
                            16000 // divisor, rate // divisor)
sf.write(str(normalized), samples_16k, 16000, subtype='PCM_16')
with wave.open(str(normalized), 'rb') as audio:
    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
        raise ValueError('invalid normalized WAV')
    duration_s = audio.getnframes() / 16000
print(json.dumps({
    'model_revision': '85e237c12c027371202489a0ec509ded67b5e4b5',
    'speaker': 'Vivian', 'text': '小窝小窝', 'seed': 1337,
    'source_sample_rate_hz': rate, 'duration_s': duration_s,
    'model_load_s': loaded_s, 'generate_s': generated_s,
    'normalized_wav_sha256': hashlib.sha256(normalized.read_bytes()).hexdigest(),
}, ensure_ascii=False))
