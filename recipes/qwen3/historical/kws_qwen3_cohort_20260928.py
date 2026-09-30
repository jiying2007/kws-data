import hashlib
import json
import math
import pathlib
import time
import wave

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from scipy.signal import resample_poly

ROOT = pathlib.Path('/work/build/qwen-tts-20260928')
MODEL = ROOT / 'model'
OUT = ROOT / 'samples'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
model = Qwen3TTSModel.from_pretrained(
    str(MODEL), device_map='cpu', dtype=torch.float32,
    attn_implementation='eager', local_files_only=True,
)
cases = [
    ('Vivian', 'kw1', '你好小窝', 1, 1338),
    ('Vivian', 'suffix-kw2', '窝小窝', None, 1339),
    ('Serena', 'kw1', '你好小窝', 1, 1340),
    ('Serena', 'kw2', '小窝小窝', 2, 1341),
    ('Uncle_Fu', 'kw1', '你好小窝', 1, 1342),
    ('Uncle_Fu', 'kw2', '小窝小窝', 2, 1343),
]
rows = []
first = OUT / 'qwen3-kw2-vivian-16k.wav'
with wave.open(str(first), 'rb') as audio:
    first_duration = audio.getnframes() / audio.getframerate()
rows.append({'recording': 'qwen3-kw2-vivian', 'audio_path': str(first),
             'text': '小窝小窝', 'keyword_id': 2, 'speaker_id': 'Vivian',
             'seed': 1337, 'duration_s': first_duration,
             'file_sha256': hashlib.sha256(first.read_bytes()).hexdigest()})
for speaker, kind, text, keyword_id, seed in cases:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.monotonic()
    wavs, rate = model.generate_custom_voice(
        text=text, speaker=speaker, language='Chinese',
        do_sample=True, max_new_tokens=120,
    )
    elapsed = time.monotonic() - started
    if len(wavs) != 1:
        raise ValueError('unexpected number of outputs')
    stem = f'qwen3-{kind}-{speaker.lower()}'
    raw = OUT / (stem + '-raw.wav')
    normalized = OUT / (stem + '-16k.wav')
    sf.write(str(raw), wavs[0], rate, subtype='PCM_16')
    divisor = math.gcd(rate, 16000)
    samples_16k = resample_poly(np.asarray(wavs[0], dtype=np.float64),
                                16000 // divisor, rate // divisor)
    sf.write(str(normalized), samples_16k, 16000, subtype='PCM_16')
    with wave.open(str(normalized), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError('bad normalized WAV')
        duration = audio.getnframes() / 16000
    row = {'recording': stem, 'audio_path': str(normalized), 'text': text,
           'keyword_id': keyword_id, 'speaker_id': speaker, 'seed': seed,
           'duration_s': duration,
           'file_sha256': hashlib.sha256(normalized.read_bytes()).hexdigest()}
    rows.append(row)
    print(json.dumps({'recording': stem, 'duration_s': duration,
                      'generation_s': elapsed, 'sha256': row['file_sha256']},
                     ensure_ascii=False), flush=True)
manifest = ROOT / 'qwen3-probe-manifest.jsonl'
manifest.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                            for row in rows), encoding='utf-8')
print(json.dumps({'recordings': len(rows), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()},
                 ensure_ascii=False), flush=True)
