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

ROOT = pathlib.Path('/work/build/software-closure-20260928/qwen3-holdout')
MODEL = pathlib.Path('/work/build/qwen-tts-20260928/model')
REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
MANIFEST = ROOT / 'manifest.jsonl'
if ROOT.exists():
    raise ValueError('frozen holdout output already exists')
ROOT.mkdir(parents=True)
texts = (('kw1', '你好小窝', 1), ('kw2', '小窝小窝', 2),
         ('suffix-kw2', '窝小窝', None), ('repeat-nihao', '你好你好', None))
cases = [(speaker, label, text, keyword_id, 6200 + index)
         for index, (speaker, (label, text, keyword_id)) in enumerate(
             ((speaker, case) for speaker in ('Serena', 'Eric') for case in texts), start=1)]
torch.set_num_threads(4)
model = Qwen3TTSModel.from_pretrained(
    str(MODEL), device_map='cpu', dtype=torch.float32,
    attn_implementation='eager', local_files_only=True,
)
with MANIFEST.open('w', encoding='utf-8') as output:
    for speaker, label, text, keyword_id, seed in cases:
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = time.monotonic()
        waves, rate = model.generate_custom_voice(
            text=text, speaker=speaker, language='Chinese',
            do_sample=True, max_new_tokens=120,
        )
        if len(waves) != 1:
            raise ValueError('wrong output count')
        factor = math.gcd(rate, 16000)
        samples = resample_poly(np.asarray(waves[0], dtype=np.float64),
                                16000 // factor, rate // factor)
        stem = f'qwen3-closure-{speaker.lower()}-{label}-s{seed}'
        path = ROOT / f'{stem}.wav'
        sf.write(str(path), samples, 16000, subtype='PCM_16')
        with wave.open(str(path), 'rb') as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                raise ValueError('bad wav format')
            frames = reader.getnframes()
            pcm = reader.readframes(frames)
        row = {
            'recording': stem, 'audio_path': str(path), 'text': text,
            'kind': 'positive' if keyword_id else 'confusable',
            'keyword_id': keyword_id, 'speaker_id': speaker, 'seed': seed,
            'source_id': f'qwen3:{REVISION}:{stem}:{seed}',
            'model_revision': REVISION, 'duration_s': frames / 16000,
            'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
            'generation_s': round(time.monotonic() - start, 3),
        }
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        output.flush()
        print(json.dumps({'recording': stem, 'duration_s': row['duration_s']}, ensure_ascii=False), flush=True)
print(json.dumps({'rows': len(cases), 'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}), flush=True)
