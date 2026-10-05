"""Bind this saved six-cell screen to the unchanged prospective panel gate.

CLI paths are local inputs only. Outputs contain logical IDs/hashes, never paths.
No model, normalization change, new admission rule or human label is introduced.
"""
import argparse,copy,hashlib,importlib.util,json,pathlib,sys,wave

PINS={'panel.py':'6da5cb2918270c0db13dffea280ac800f370ad0bd47805e9b7f28f59c1897aee',
      'vendor/__init__.py':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      'vendor/quality_gates.py':'e9a0a838e134746bbd0fdd60aeff48ca9ad2963b721e3ea45bd0d8b45690b15d',
      'declared-history.json':'3251c0fc2cf67aee04e475ac64e5c7fba66af0d20c84a78489097795a3799985',
      'proposed-panel.json':'e05f4ea950a56c7bdb349b6d6d6aece8b706e82a06712115030d58ba041e0e37'}
COMPARISON_SHA='132764b252f4e4f6191910d1053828e2d6a72c6a4d321ad2f74d0312a1a644ec'
RECEIPT_SHA='d8bef1fa08b7b79de20cec038e2d209e783fa0834a181c183f6b1f1ec59153d2'
GENERATION_FREEZE_SHA='ed9ccd99d41ea2f63d57bcd7eb49b603cd280cf10b3cdaa1f028220fc6dcab6c'
TTS_PLAN_SHA='1b0e7296877432ee24c6bf79d79573d78eda8f8f4277fd06edb9dba79cefac4c'
MODELS=['FunAudioLLM/SenseVoiceSmall','Qwen/Qwen3-ASR-0.6B']
MAPPING=[(f'qwen6-{i:03d}',f'clip-{i:06d}',voice,cell,role,text)
         for i,(voice,cell,role,text) in enumerate([
             ('Ryan','K1','train','你好小窝'),('Ryan','K2','train','小窝小窝'),
             ('Aiden','K1','train','你好小窝'),('Aiden','K2','train','小窝小窝'),
             ('Ono_Anna','K1','dev','你好小窝'),('Ono_Anna','K2','dev','小窝小窝')],1)]

def require(value,message):
    if not value:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read(path,expected=None):
    path=pathlib.Path(path);require(path.is_file() and not path.is_symlink(),'Expected regular bound input')
    raw=path.read_bytes();require(expected is None or sha(raw)==expected,'Input byte identity mismatch');return raw

def integrate(panel_root,comparison_path,generation_root):
    panel_root=pathlib.Path(panel_root).resolve();generation_root=pathlib.Path(generation_root).resolve()
    captured={};panel_bytes={}
    for name,expected in PINS.items():
        raw=read(panel_root/name,expected);panel_bytes[name]=raw;captured[panel_root/name]=sha(raw)
    comparison_raw=read(comparison_path,COMPARISON_SHA);captured[pathlib.Path(comparison_path)]=sha(comparison_raw)
    comparison=json.loads(comparison_raw);freeze_raw=read(generation_root/'generation-freeze.json',GENERATION_FREEZE_SHA)
    captured[generation_root/'generation-freeze.json']=sha(freeze_raw);freeze=json.loads(freeze_raw)
    require(freeze['schema']=='qwen6-generation-freeze-v1' and len(freeze['files'])==19,'Wrong generation freeze')
    names=set()
    for row in freeze['files']:
        name=row['path'];require(type(name) is str and pathlib.PurePosixPath(name).name==name and name not in names,'Generation member name')
        names.add(name);raw=read(generation_root/name,row['sha256']);require(len(raw)==row['bytes'],'Generation member size')
        captured[generation_root/name]=sha(raw)
    require({p.name for p in generation_root.iterdir()}==names|{'generation-freeze.json'},'Generation membership changed')
    receipt=json.loads(read(generation_root/'generation-receipt.json',RECEIPT_SHA))
    require(receipt['status']=='six_candidates_generated' and receipt['plan_sha256']==TTS_PLAN_SHA,'Generation plan/status')
    require(comparison['schema']=='private-asr6-comparison-v1' and comparison['models']==MODELS,'Comparison method/model scope')
    verification=comparison['saved_artifact_verification']
    require(verification['both_model_raw_freezes_verified_before_private_read'] is True and verification['generation_freeze_sha256']==GENERATION_FREEZE_SHA and verification['plan_sha256']==TTS_PLAN_SHA,'Frozen comparison association')
    clips={r['audio_id']:r for r in comparison['clips']};generated={r['source_id']:r for r in receipt['generation_rows']}
    outcomes={(r['audio_id'],r['model']):r for r in comparison['outcomes']}
    require(len(clips)==len(comparison['clips'])==6 and len(generated)==len(receipt['generation_rows'])==6 and len(outcomes)==len(comparison['outcomes'])==12,'Saved denominator')
    panel_data=json.loads(panel_bytes['proposed-panel.json']);rows={r['id']:r for r in panel_data['plan']['rows']};observations=[];mapping=[]
    for source_id,audio_id,voice,cell,role,text in MAPPING:
        row=generated[source_id];clip=clips[audio_id];panel_id=f'screen-{voice.lower()}-{cell}';planned=rows[panel_id]
        require((row['voice'],row['voice_group'],row['intended_keyword_id'],row['intended_text'])==(voice,'Qwen3-TTS-CustomVoice:'+voice,int(cell[1]),text),'Generation fixed cell changed')
        require({'development':'dev'}.get(row['prospective_role'],row['prospective_role'])==role,'Generation role changed')
        require((clip['preset'],clip['keyword'],clip['role'],clip['intended_text'])==(voice,cell,role,text),'Comparison fixed cell changed')
        require((planned['voice_identity'],planned['prospective_role'],planned['intended_text'],planned['stage'])==('Qwen3-stock:'+voice,role,text,'screen'),'Panel fixed cell changed')
        wav=read(generation_root/(source_id+'.wav'),row['audio_sha256'])
        with wave.open(str(generation_root/(source_id+'.wav')),'rb') as f:
            require((f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,16000),'WAV geometry changed');pcm=f.readframes(f.getnframes())
        require(sha(pcm)==row['pcm_sha256']==clip['signal_measurements']['pcm_sha256'],'PCM binding changed')
        asr=[copy.deepcopy(outcomes[(audio_id,model)]['raw_record']) for model in MODELS]
        require(all(r['audio_id']==audio_id and r['wav_sha256']==sha(wav) for r in asr),'ASR waveform binding changed')
        observations.append({'id':panel_id,'actual_text':None,'wav_sha256':sha(wav),'pcm_sha256':sha(pcm),
            'generation_receipt_sha256':RECEIPT_SHA,'review':{'status':'pending','independent_human':False,'complete':False},
            'asr_results':asr,'exposure':'EXPOSED','reference_sha256':None,'evidence_basis':'observed',
            'generator':copy.deepcopy(panel_data['plan']['generator']),'voice_identity':planned['voice_identity']})
        mapping.append({'source_id':source_id,'audio_id':audio_id,'panel_id':panel_id,'voice_identity':planned['voice_identity'],
            'cell':cell,'prospective_role':role,'comparison_handling':clip['handling'],
            'weak_machine_supported_candidate':clip['weak_machine_supported_candidate'],'quarantine_reasons':clip['quarantine_reasons']})
    require(all(not x['human_truth_established'] and not x['training_admitted'] for x in comparison['clips']),'Comparison wrongly promoted to truth')
    sys.path.insert(0,str(panel_root))
    if 'vendor.quality_gates' in sys.modules:
        require(pathlib.Path(sys.modules['vendor.quality_gates'].__file__).resolve()==panel_root/'vendor/quality_gates.py','Different gate already imported')
    spec=importlib.util.spec_from_file_location('saved_prospective_panel',panel_root/'panel.py');panel=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)
    report,code=panel.assess(panel_data,observations,history=json.loads(panel_bytes['declared-history.json']),require_data=True)
    require(code==1 and report['status']=='REJECTED' and report['counts']['observations']==6,'Expected real gate refusal')
    require(not report['coverage']['balanced_admission'] and not report['training_authorized'] and report['heldout_screen_attempts']==0,'Admission scope changed')
    unchanged=all(sha(read(path))==expected for path,expected in captured.items());require(unchanged,'Frozen input mutation')
    binding={'schema':'saved-qwen6-panel-integration-v1','run_id':37343458180,'comparison_sha256':COMPARISON_SHA,
        'generation_receipt_sha256':RECEIPT_SHA,'generation_freeze_sha256':GENERATION_FREEZE_SHA,'frozen_panel_sources':PINS,
        'asr_result_model_order':MODELS,'mapping':mapping,'gate_exit_code':code,'require_data':True,'screen_exposed_observations':6,
        'human_truth_observations':0,'training_rows':0,'sohee_screen_attempts':0,'frozen_inputs_unchanged':unchanged,
        'namespace_note':'Qwen3-TTS-CustomVoice and Qwen3-stock name the same pinned preset; development maps to dev. No acoustic identity authentication is inferred.',
        'no_new_model_calls':True,'no_new_human_review':True,'new_admission_rules':False}
    return observations,report,binding

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel-root',type=pathlib.Path,required=True);p.add_argument('--comparison',type=pathlib.Path,required=True);p.add_argument('--generation-root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    observations,report,binding=integrate(a.panel_root,a.comparison,a.generation_root)
    a.out.mkdir(parents=True,exist_ok=False)
    for name,value in [('OBSERVATIONS.json',observations),('GATE-REPORT.json',report),('BINDING-REPORT.json',binding)]:
        (a.out/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'integration':'complete','gate_status':report['status'],'gate_exit_code':binding['gate_exit_code'],'observations':6,'training_rows':0,'sohee_screen_attempts':0}))

if __name__=='__main__':main()
