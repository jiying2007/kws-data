#!/usr/bin/env python3
"""Explicitly post-hoc evaluation; never reruns models or replaces v1 artifacts."""
import argparse,ast,datetime,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from asr_stage.bundles import validate_primary_bundle

ORIGINAL_RULES_SHA='1495ae540c56a20de1ba78074d10f488f741acca82be9e358d2a2b4c68828f8b'
ORIGINAL_READOUT_SHA='c332fd6390de1ccf5f3b21f3f6a38724721deb45e28f44d74993c2195b43bac3'
MANIFEST_SHA='500a5fac10960bcbe3ead0be38a81f9cfe6c0b8a22e11b25cca5bb292cd86355'
SOURCES=[('qwen06','0da6856b7342ad1c8f7b66dfd2a189bf4bb9fa32b9a8dc176b74a2c458df922c',
    'bf99c7d209eaaa317b03c6f0d99c9f96eba5a46506ff34f3f5cfde8bb47545fa'),
    ('sensevoice','0ec850fb323cbdd42267e9a0c695107633a3f650757bfe7d307993f8e93d61cb',
    'c678aaef9312b6192dbacd503976cac19013f780862ec354a0bba137634c1453')]
def checked(path,sha):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Frozen source bytes changed: '+path.name)
    return data
def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as f:f.write(raw)
    return hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-scorer-sha256',required=True)
    args=parser.parse_args()
    scorer=HERE/'calibrate.py';source=checked(scorer,args.reviewed_scorer_sha256)
    tree=ast.parse(source)
    versions=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='RULE_VERSION' for t in n.targets)]
    if versions!=['posthoc-target-local-text-evidence-v2']:raise ValueError('Unexpected diagnostic rule version')
    original=ROOT/'calibration-result-v1'
    checked(original/'readout.json',ORIGINAL_READOUT_SHA)
    manifest=ROOT/'execution-plans-v2/scientific-manifest.json';checked(manifest,MANIFEST_SHA)
    rules=json.loads(checked(ROOT/'scoring/frozen/rules.json',ORIGINAL_RULES_SHA))
    rules['rule_version']=versions[0]
    output=HERE/'evaluation-v1';output.mkdir(exist_ok=False)
    rules_sha=write(output/'rules.json',rules)
    provenance={'schema':'asr-posthoc-evaluation-provenance-v1',
        'evaluation_scope':'post_hoc_diagnostic_reused_outputs',
        'prepared_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'policy_timing':'Designed after observing v1 zero coverage; synthetic general fixture tests and separate review preceded this evaluation. Not untouched-holdout qualification.',
        'source_execution_rules_sha256':ORIGINAL_RULES_SHA,
        'revised_evaluation_rules_sha256':rules_sha,
        'source_execution_manifest_sha256':MANIFEST_SHA,
        'source_frozen_readout_sha256':ORIGINAL_READOUT_SHA,
        'source_frozen_archive_sha256':'b629479217934db38792a7d6ddcb349f91a98f9a8924c8f073ec4905f03bc194',
        'revised_scorer_sha256':args.reviewed_scorer_sha256,
        'source_model_runs':[], 'new_model_forward_calls':0,
        'transcript_or_token_edits':0, 'human_label_edits':0,
        'envelope_change_scope':'Only rules_sha256 changes to identify this later evaluation; original execution declarations remain preserved'}
    for name,plan_sha,envelope_sha in SOURCES:
        _,_,validation=validate_primary_bundle(str(ROOT/'primary-runs-v2'/name),plan_sha)
        envelope=json.loads(checked(original/(name+'-envelope.json'),envelope_sha))
        if envelope['rules_sha256']!=ORIGINAL_RULES_SHA:raise ValueError('Original envelope rule identity changed')
        old=json.loads(json.dumps(envelope));envelope['rules_sha256']=rules_sha
        comparison=dict(envelope);comparison['rules_sha256']=ORIGINAL_RULES_SHA
        assert comparison==old
        new_sha=write(output/(name+'-reused-envelope.json'),envelope)
        write(output/(name+'-source-bundle-validation.json'),validation)
        provenance['source_model_runs'].append({'model':envelope['model'],
            'original_envelope_sha256':envelope_sha,'reused_envelope_sha256':new_sha,
            'raw_primary_outcomes_sha256':validation['outcomes_sha256']})
    write(output/'evaluation-provenance.json',provenance)
    command=[sys.executable,'-B','-I','-S',str(scorer),'--manifest',str(manifest),
        '--manifest-sha256',MANIFEST_SHA,'--rules',str(output/'rules.json'),
        '--rules-sha256',rules_sha,'--characters',str(ROOT/'scoring/frozen/characters.json'),
        '--phrases',str(ROOT/'scoring/frozen/phrases.json'),
        '--model-a',str(output/'qwen06-reused-envelope.json'),
        '--model-b',str(output/'sensevoice-reused-envelope.json'),
        '--output',str(output/'readout.json')]
    result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=120)
    (output/'scorer.stdout.json').write_bytes(result.stdout)
    (output/'scorer.stderr.txt').write_bytes(result.stderr)
    write(output/'invocation.json',{'command':command,'returncode':result.returncode})
    if result.returncode:raise RuntimeError('Revised diagnostic scorer rejected input')
    # Prove the preserved baseline remains unchanged after the diagnostic.
    checked(original/'readout.json',ORIGINAL_READOUT_SHA)
    checked(ROOT/'scoring/frozen/rules.json',ORIGINAL_RULES_SHA)
    print(result.stdout.decode(),end='')
if __name__=='__main__':main()
