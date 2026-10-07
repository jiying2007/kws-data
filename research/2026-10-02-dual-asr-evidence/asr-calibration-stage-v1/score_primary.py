#!/usr/bin/env python3
"""Validate actual retained raw bundles, then run the unchanged frozen scorer."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from asr_stage.bundles import validate_primary_bundle
from asr_stage.results import output_envelope

MANIFEST_SHA='500a5fac10960bcbe3ead0be38a81f9cfe6c0b8a22e11b25cca5bb292cd86355'
RULES_SHA='1495ae540c56a20de1ba78074d10f488f741acca82be9e358d2a2b4c68828f8b'
DECODER_SHA='49e7b6b3a1bdbf9e48de698ea3eb6053fc7c13cb2be800b95f8e1e7a86fd8729'
HUMAN_SHA='400b11710507a2fd912b8400b21dba61fd1ab374fae317dd07da29b14c9f7b88'
SCORER_SHA='2bd566983f9846e19c3ac24d59cffe155cc4b45c39c1d64a9f1fae7772af104a'
PLANS=[('qwen06','0da6856b7342ad1c8f7b66dfd2a189bf4bb9fa32b9a8dc176b74a2c458df922c'),
       ('sensevoice','0ec850fb323cbdd42267e9a0c695107633a3f650757bfe7d307993f8e93d61cb')]

def checked(path,sha):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Frozen input identity changed')
    return data
def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    manifest_path=ROOT/'execution-plans-v2/scientific-manifest.json'
    manifest_raw=checked(manifest_path,MANIFEST_SHA);manifest=json.loads(manifest_raw)
    human=json.loads(checked(ROOT/'human-labels.json',HUMAN_SHA))
    decoder=checked(ROOT/'pcm/decoder-inputs.json',DECODER_SHA)
    rules=ROOT/'scoring/frozen/rules.json';checked(rules,RULES_SHA)
    scorer=ROOT/'scoring/scorer/calibrate.py';checked(scorer,SCORER_SHA)
    output=ROOT/'calibration-result-v1';output.mkdir(exist_ok=False)
    for slot,(name,sha) in enumerate(PLANS):
        run=ROOT/'primary-runs-v2'/name
        rows,receipts,validation=validate_primary_bundle(str(run),sha)
        envelope,bindings=output_envelope(rows,human,manifest,MANIFEST_SHA,RULES_SHA,slot,
            manifest_bytes=manifest_raw,execution_receipts=receipts,
            decoder_manifest_bytes=decoder,decoder_manifest_sha256=DECODER_SHA)
        write(output/(name+'-envelope.json'),envelope)
        write(output/(name+'-bundle-validation.json'),validation)
        write(output/(name+'-receipt-bindings.json'),bindings)
    command=[sys.executable,'-B','-I','-S',str(scorer),
        '--manifest',str(manifest_path),'--manifest-sha256',MANIFEST_SHA,
        '--rules',str(rules),'--rules-sha256',RULES_SHA,
        '--characters',str(ROOT/'scoring/frozen/characters.json'),
        '--phrases',str(ROOT/'scoring/frozen/phrases.json'),
        '--model-a',str(output/'qwen06-envelope.json'),
        '--model-b',str(output/'sensevoice-envelope.json'),
        '--output',str(output/'readout.json')]
    result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=120)
    (output/'scorer.stdout.json').write_bytes(result.stdout)
    (output/'scorer.stderr.txt').write_bytes(result.stderr)
    write(output/'invocation.json',{'command':command,'returncode':result.returncode,
        'scorer_sha256':SCORER_SHA,'manifest_sha256':MANIFEST_SHA,'rules_sha256':RULES_SHA})
    if result.returncode:raise RuntimeError('Frozen scorer rejected inputs; inspect retained stderr')
    print(result.stdout.decode(),end='')
if __name__=='__main__':main()
