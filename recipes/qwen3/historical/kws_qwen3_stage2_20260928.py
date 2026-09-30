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
REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
OUT = ROOT / 'samples'
OLD = ROOT / 'qwen3-probe-manifest.jsonl'
NEW = ROOT / 'qwen3-stage2-manifest.jsonl'
if NEW.exists():
    raise ValueError('stage2 manifest already exists')

cases = [
    ('Dylan', 'kw1', '你好小窝', 1, 1344),
    ('Dylan', 'kw2', '小窝小窝', 2, 1345),
    ('Eric', 'kw1', '你好小窝', 1, 1346),
    ('Eric', 'kw2', '小窝小窝', 2, 1347),
    ('Serena', 'suffix-kw2', '窝小窝', None, 1348),
    ('Uncle_Fu', 'suffix-kw2', '窝小窝', None, 1349),
    ('Dylan', 'suffix-kw2', '窝小窝', None, 1350),
    ('Eric', 'suffix-kw2', '窝小窝', None, 1351),
    ('Vivian', 'repeat-nihao', '你好你好', None, 1352),
    ('Serena', 'repeat-nihao', '你好你好', None, 1353),
    ('Uncle_Fu', 'repeat-nihao', '你好你好', None, 1354),
    ('Dylan', 'repeat-nihao', '你好你好', None, 1355),
    ('Eric', 'repeat-nihao', '你好你好', None, 1356),
]

rows = []
for line in OLD.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    audio = pathlib.Path(row['audio_path'])
    local = OUT / audio.name
    if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != row['file_sha256']:
        raise ValueError(f'existing audio identity mismatch: {row["recording"]}')
    row['audio_path'] = (pathlib.Path('samples') / local.name).as_posix()
    row['kind'] = 'positive' if row['keyword_id'] is not None else 'confusable'
    row['source_id'] = f'qwen3:{REVISION}:{row["recording"]}:{row["seed"]}'
    rows.append(row)

torch.set_num_threads(4)
model = Qwen3TTSModel.from_pretrained(
    str(ROOT / 'model'), device_map='cpu', dtype=torch.float32,
    attn_implementation='eager', local_files_only=True,
)
for speaker, kind, text, keyword_id, seed in cases:
    stem = f'qwen3-{kind}-{speaker.lower()}'
    raw = OUT / (stem + '-raw.wav')
    normalized = OUT / (stem + '-16k.wav')
    if raw.exists() or normalized.exists():
        raise ValueError(f'new audio path already exists: {stem}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.monotonic()
    wavs, rate = model.generate_custom_voice(
        text=text, speaker=speaker, language='Chinese',
        do_sample=True, max_new_tokens=120,
    )
    elapsed = time.monotonic() - started
    if len(wavs) != 1:
        raise ValueError(f'unexpected generated count: {stem}')
    sf.write(str(raw), wavs[0], rate, subtype='PCM_16')
    divisor = math.gcd(rate, 16000)
    samples_16k = resample_poly(np.asarray(wavs[0], dtype=np.float64),
                                16000 // divisor, rate // divisor)
    sf.write(str(normalized), samples_16k, 16000, subtype='PCM_16')
    with wave.open(str(normalized), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f'bad normalized WAV: {stem}')
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    if frames < 3200 or rms < 0.001:
        raise ValueError(f'silent or too short: {stem}')
    row = {
        'recording': stem,
        'audio_path': (pathlib.Path('samples') / normalized.name).as_posix(),
        'text': text, 'kind': 'positive' if keyword_id is not None else 'confusable',
        'keyword_id': keyword_id, 'speaker_id': speaker, 'seed': seed,
        'source_id': f'qwen3:{REVISION}:{stem}:{seed}',
        'duration_s': frames / 16000,
        'rms': rms, 'peak': float(np.max(np.abs(samples))),
        'file_sha256': hashlib.sha256(normalized.read_bytes()).hexdigest(),
        'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
    }
    rows.append(row)
    print(json.dumps({'recording': stem, 'duration_s': row['duration_s'],
                      'generation_s': elapsed, 'sha256': row['file_sha256']},
                     ensure_ascii=False), flush=True)

if (len(rows) != 20 or sum(row['keyword_id'] is not None for row in rows) != 10
        or len({row['source_id'] for row in rows}) != 20
        or len({row['file_sha256'] for row in rows}) != 20):
    raise ValueError('stage2 cohort identity/count mismatch')
NEW.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                       for row in rows), encoding='utf-8')
print(json.dumps({'recordings': len(rows), 'positives': 10, 'negatives': 10,
                  'manifest_sha256': hashlib.sha256(NEW.read_bytes()).hexdigest()},
                 ensure_ascii=False), flush=True)
