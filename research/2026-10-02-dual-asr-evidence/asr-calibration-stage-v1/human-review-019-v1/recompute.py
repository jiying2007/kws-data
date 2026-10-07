#!/usr/bin/env python3
"""Apply one explicit human label correction; replay unchanged text scorers only."""
import copy,datetime,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RECORD='kws-temporal72-019-eric-kw2_wu_confusable'
TARGET='小窝小窝'
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_bytes())
def checked(path,digest):
    data=path.read_bytes()
    if sha(data)!=digest:raise ValueError('Original evidence changed: '+str(path))
    return json.loads(data)
def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw)
    return sha(raw)
def main():
    original_human=ROOT/'human-labels.json';old_human_sha='400b11710507a2fd912b8400b21dba61fd1ab374fae317dd07da29b14c9f7b88'
    human=checked(original_human,old_human_sha)
    original_manifest=ROOT/'execution-plans-v2/scientific-manifest.json';old_manifest_sha='500a5fac10960bcbe3ead0be38a81f9cfe6c0b8a22e11b25cca5bb292cd86355'
    manifest=checked(original_manifest,old_manifest_sha)
    for obj in [human,manifest]:
        matches=[r for r in obj['records'] if r['recording_id']==RECORD]
        assert len(matches)==1 and matches[0]['human_target_presence']==['negative','positive']
        matches[0]['human_target_presence'][1]='negative'
    counts={x:sum(r['human_target_presence'].count(x) for r in human['records']) for x in ['positive','negative','unknown']}
    assert counts=={'positive':8,'negative':18,'unknown':6}
    human['human_label_counts']=counts
    human_sha=write(HERE/'human-labels.corrected.json',human)
    manifest_sha=write(HERE/'scientific-manifest.corrected.json',manifest)
    overlay={'schema':'asr-human-label-correction-v1','scope':'One later human annotation correction; no scorer-policy or ASR-output change',
      'reviewed_at_utc':'2026-10-02T01:08:58Z','reviewer_role':'user human listener',
      'human_comment':'019发音不标准，听起来像小屋小窝，不是小窝小窝',
      'recording_id':RECORD,'opaque_id':'clip-0004','wav_sha256':'a9344deed334fa976e6879e14033166dcf67a2f516c70ab7b2065ba3e70cafdd',
      'target':TARGET,'target_index':1,'previous_label':'positive','corrected_label':'negative','other_target_label_unchanged':'negative',
      'previous_human_labels_sha256':old_human_sha,'corrected_human_labels_sha256':human_sha,
      'previous_execution_manifest_sha256':old_manifest_sha,'corrected_evaluation_manifest_sha256':manifest_sha,
      'corrected_human_label_counts':counts,'unknown_human_labels_promoted':0,'new_model_forward_calls':0,
      'scope_limit':'The comment is a pronunciation/phrase-presence judgment, not a complete transcript or measured tone/timing annotation.'}
    overlay_sha=write(HERE/'human-overlay.json',overlay)
    configurations=[('frozen-v1',ROOT/'calibration-result-v1',ROOT/'scoring/scorer/calibrate.py',ROOT/'scoring/frozen/rules.json','-envelope.json','2bd566983f9846e19c3ac24d59cffe155cc4b45c39c1d64a9f1fae7772af104a'),
      ('posthoc-v2',ROOT/'scoring/posthoc-v2/evaluation-v1',ROOT/'scoring/posthoc-v2/calibrate.py',ROOT/'scoring/posthoc-v2/evaluation-v1/rules.json','-reused-envelope.json','1bccdd66bca3cd67109df1eff63770b5d7e9ba3a9a10b578a665b7bbcc9575b4')]
    result_summary={};provenance=[]
    for name,source,scorer,rules,suffix,scorer_sha in configurations:
        assert sha(scorer.read_bytes())==scorer_sha
        out=HERE/name;out.mkdir(exist_ok=False);old_readout=read(source/'readout.json');source_readout_sha=sha((source/'readout.json').read_bytes());rules_sha=sha(rules.read_bytes())
        env_records=[]
        for model in ['qwen06','sensevoice']:
            original=read(source/(model+suffix));assert original['manifest_sha256']==old_manifest_sha
            new=copy.deepcopy(original);new['manifest_sha256']=manifest_sha
            assert dict(new,manifest_sha256=old_manifest_sha)==original
            new_sha=write(out/(model+'-envelope.json'),new)
            env_records.append({'model':model,'original_sha256':sha((source/(model+suffix)).read_bytes()),'corrected_evaluation_envelope_sha256':new_sha,'change':'manifest_sha256 only; model identity and all output records identical'})
        command=[sys.executable,'-B','-I','-S',str(scorer),'--manifest',str(HERE/'scientific-manifest.corrected.json'),'--manifest-sha256',manifest_sha,
          '--rules',str(rules),'--rules-sha256',rules_sha,'--characters',str(ROOT/'scoring/frozen/characters.json'),'--phrases',str(ROOT/'scoring/frozen/phrases.json'),
          '--model-a',str(out/'qwen06-envelope.json'),'--model-b',str(out/'sensevoice-envelope.json'),'--output',str(out/'readout.json')]
        p=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=120)
        (out/'scorer.stdout.json').write_bytes(p.stdout);(out/'scorer.stderr.txt').write_bytes(p.stderr)
        write(out/'invocation.json',{'command':command,'returncode':p.returncode})
        if p.returncode:raise RuntimeError('Scorer replay failed')
        new_readout=read(out/'readout.json')
        assert len(new_readout['records'])==len(old_readout['records'])==16
        for a,b in zip(old_readout['records'],new_readout['records']):
            assert a['recording_id']==b['recording_id'] and a['model_evidence']==b['model_evidence']
            if a['recording_id']!=RECORD:assert a==b
            else:assert b['human_target_presence']==['negative','negative']
        assert sha((source/'readout.json').read_bytes())==source_readout_sha
        result_summary[name]={k:new_readout[k] for k in ['human_label_counts','per_model_known_bit_readout','paired_known_bit_readout']}
        result_summary[name]['readout_sha256']=sha((out/'readout.json').read_bytes())
        provenance.append({'scorer_variant':name,'scorer_sha256':scorer_sha,'rules_sha256':rules_sha,'original_readout_sha256':source_readout_sha,'corrected_readout_sha256':result_summary[name]['readout_sha256'],'envelopes':env_records})
    checked(original_human,old_human_sha);checked(original_manifest,old_manifest_sha)
    write(HERE/'evaluation-provenance.json',{'schema':'asr-corrected-human-evaluation-v1','evaluation_scope':'Later human label correction, unchanged scorers and reused model outputs','prepared_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'human_overlay_sha256':overlay_sha,'changed_human_target_bits':1,'unchanged_model_evidence_rows':32,'raw_text_token_wav_edits':0,'new_model_forward_calls':0,'source_results_preserved':True,'scorer_policy_changes':0,'evaluations':provenance})
    write(HERE/'summary.json',result_summary)
    print(json.dumps(result_summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
