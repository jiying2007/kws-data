"""Fresh 2026-10-30 input/slot contract; stdlib only, no model execution.

This is NOT the consumed qwen12-native-admission remaining35 campaign. The
retained worker receipt API may consume these explicit fresh job/slot functions;
no historical launcher, release, result or consumed-slot record is modified.
"""
from pathlib import Path
import hashlib
import io
import json
import re
import stat
import wave

CAMPAIGN = 'qwen12-native-capture-20261030-v1'
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
# Independently measured from merged #35 artifact 12976253656, run 41055537577.
# Opaque identities only: never prompts, expected text, speakers or split labels.
SOURCES = (
 ('063378ac988d9e7251a25af3f6aa9c37bb06ba5b865e2aa2b346236c58daafc1','7e37ce1b2ae15422bec19d91a447dd490cd7a6b2115951d2995b25ab7a5b6b3f',70400),
 ('0861a3075f4f8079bf8f76395605d114930503ffee1b610585ed201b8001c4bb','c0aa606827e28c2c8c5d33f6a5cb4c740ac320d7c201d8cde96bb2797d1e1af0',32000),
 ('1cee8eac7d27ad0c7842a1f054d8c10f0979c61776277fbdfe0414599091cd0a','405372288dddd08257f087fb0d40a1b934c332a6a7204e135a57fcceb51f9476',29440),
 ('2c73f17624243df5b9f50794e83d112291c908d40835263d6e5c75f0e37a4809','ffb8d3d46512ed98fd362d43d3cdc78e0cb9293018d0426c6fdc77cc7d17c8f1',32000),
 ('4cbe9933c95cf927b57274865f79cd7523da6ccc5d748ef0e602b0b30bfaee75','142f3aba53d62919feea3a0c8b8aff6aa4118d34a18f4a6f80240de011c39d2c',49920),
 ('5898c6b1dccdb785f2ebc4d2551c979ac176fabf9a24e238ec6509d68717bfa7','8ac003a40d4a812b83bf75aec00b751bc7c6322f23cb4e9b5e076aed26ce01a7',53760),
 ('59d2968d1b016833e25ca284466258b5ef465c1824d1273c2fb749ea79d61495','a3c91ba55f7533901735a39c04d56b01c982287279b659942423263f79e88033',44800),
 ('8a260e0aac52fa88e9573c013dc4ca102f00ed116100bab3c15f269525bdcf0b','2a32beb64db80c3c84678668fc9b29539f3ff7c3216e6cc7f9fb59ca284157e8',39680),
 ('b491fbd04b8e5ae6e1ead3db23a1b0a1ad59dfd1a7e795891afd9258a424dd89','b50f16b3d165f67d46cd1333ca1403b3f7fca04f166f628b2fd572acb6cb91a9',32000),
 ('d01601234d539a741b03264938599b3aabe4f41f0955a48af6150389ce2929d1','5191a9be2682b79a82da6a828fa461e4aff60613823803672d66bb01d93bc021',37120),
 ('e748ba37f993dc6317cb2a5d44ba82da4321995cf4ac16a3030d6eb390783cc9','b1db70ef5f870cb4c8ee6683e1ebb3a0aa557cbb748698ac25c05c895d71e827',37120),
 ('fa34a53ddb4cbe515e2d8f2f9e54577998d7df680e2307b5432f7a32c03afc4a','5259de53127ee6531440ec6e695b4e4b6bada6319c2dfb7caeb1051a1a9ac499',23040),
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def original_job():
    return {'schema': 'blind-source-screen32-job-v1', 'clips': [
        {'audio_id': f'clip-{i:06d}', 'audio_path': 'audio/'+wav+'.wav',
         'wav_sha256': wav} for i, (wav, pcm, frames) in enumerate(SOURCES, 1)]}


def validate_original_job(job):
    require(type(job) is dict and canonical(job) == canonical(original_job()),
            'exact fresh twelve-clip job required; historical/other jobs rejected')
    return job


def remaining_plan(job, family=None):
    """Compatibility name: ALL 36 fresh slots, zero inherited consumed slots."""
    validate_original_job(job)
    require(family is None or family in FAMILIES, 'fixed recognizer family required')
    return [{'family': f, 'audio_id': f'clip-{i:06d}', 'wav_sha256': wav,
             'pcm_sha256': pcm, 'audio_frames': frames, 'source_id': 'qwen12-'+wav}
            for f in FAMILIES if family is None or f == family
            for i, (wav, pcm, frames) in enumerate(SOURCES, 1)]


def planned_clips(job, family):
    require(family in FAMILIES, 'fixed recognizer family required')
    validate_original_job(job)
    return [dict(row) for row in job['clips']]


def validate_start_slot(job, family, *, audio_id, measured, source_id, request_id, run_id):
    require(type(run_id) is str and re.fullmatch(r'[1-9][0-9]{0,19}', run_id), 'real run ID required')
    rows = [r for r in remaining_plan(job, family) if r['audio_id'] == audio_id]
    require(len(rows) == 1 and family in FAMILIES, 'one fixed fresh slot required')
    row = rows[0]
    expected = {'file_sha256': row['wav_sha256'], 'pcm_sha256': row['pcm_sha256'],
                'audio_frames': row['audio_frames']}
    require(type(measured) is dict and type(measured.get('audio_frames')) is int and
            canonical(measured) == canonical(expected), 'measured full-source identity drift')
    require(source_id == row['source_id'] and
            request_id == 'qwen12-'+run_id+'-'+family+'-'+audio_id, 'source/request binding drift')
    return row


def bounded_read(path, maximum):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'path aliases rejected')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum,
            'bounded unaliased regular file required')
    with path.open('rb') as stream:
        body = stream.read(maximum+1)
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    require(stamp(before) == stamp(after) and len(body) == before.st_size, 'file changed during read')
    return body


def validate_blind(root):
    root = Path(root)
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'unaliased blind directory')
    require({p.name for p in root.iterdir()} == {'job.json', 'audio'}, 'blind input may contain only job/audio')
    require(bounded_read(root/'job.json', 65536) == canonical(original_job()), 'canonical fresh job bytes required')
    audio = root/'audio'
    require(audio.is_dir() and not audio.is_symlink(), 'unaliased audio directory')
    require({p.name for p in audio.iterdir()} == {wav+'.wav' for wav, pcm, frames in SOURCES}, 'exact twelve audio members')
    for wav, pcm_sha, frames in SOURCES:
        raw = bounded_read(audio/(wav+'.wav'), 1024**2)
        require(hashlib.sha256(raw).hexdigest() == wav, 'original WAV bytes required')
        require(raw[:4] == b'RIFF' and raw[8:12] == b'WAVE' and
                int.from_bytes(raw[4:8], 'little')+8 == len(raw), 'complete original RIFF extent')
        with wave.open(io.BytesIO(raw), 'rb') as stream:
            require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                     stream.getcomptype(), stream.getnframes()) == (1, 2, 16000, 'NONE', frames), 'PCM geometry drift')
            pcm = stream.readframes(frames)
            require(not stream.readframes(1), 'unexpected trailing samples')
        require(len(pcm) == frames*2 and hashlib.sha256(pcm).hexdigest() == pcm_sha, 'measured PCM identity drift')
    return original_job()
