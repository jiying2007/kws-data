"""Bounded TRAIN-only replay of original, unprompted 392-class frame logits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import wave
import zipfile

import numpy as np

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check(condition, label):
    if not condition:
        raise ValueError(label)

def read_archive(search, names, digest):
    for root in search:
        for n in names:
            path = root / n
            if not path.exists():
                continue
            data = path.read_bytes()
            check(sha(data) == digest, n)
            z = zipfile.ZipFile(io.BytesIO(data))
            check(len(z.namelist()) == len(set(z.namelist())), n)
            return z
    raise FileNotFoundError(names)

def obj(z, name):
    return json.loads(z.read(name))

def free_path(logits, vocab):
    seq = logits.argmax(1)
    logprob = logits.astype('float64')
    logprob -= logprob.max(axis=1,keepdims=True)
    logprob -= np.log(np.exp(logprob).sum(axis=1,keepdims=True))
    starts = np.flatnonzero(np.r_[True,seq[1:] != seq[:-1]])
    phones, peaks = [], []
    for start, end in zip(starts,[*starts[1:],len(seq)]):
        index = seq[start]
        if index == 0:
            continue
        phones.append(vocab[int(index)])
        peaks.append(int(start + np.argmax(logprob[start:end,index])))
    return phones, peaks

def find_matches(tokens, peaks):
    """Cover the explicit five phones (optional preceding h) without forcing any frame label."""
    out = []
    for k in range(len(tokens)-4):
        if (tokens[k:k+3]==['eɪ','m','ɛ'] and tokens[k+3] in ('r','ɹ') and tokens[k+4] in ('oʊ','o','ɔ')):
            start = k - 1 if k and tokens[k-1] == 'h' else k
            out.append({'full':start==0 and k+5==len(tokens), 'peak_frames':[peaks[k],peaks[k+2],peaks[k+4]]})
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    locks=json.loads((root/'OBSERVER_LOCK.json').read_bytes())
    gain_sha='59e4f39122debbcb3151977a5f6481ba59cb5766a826840a5abaa8653e215bc3'
    supp_sha='5b8f2dfab1285eb9f544e538dd3508c2c1c9405df12c06f99ba03b61ca29ee8c'
    g=read_archive(args.sources,['14782234724.zip','gain44.zip'],gain_sha)
    s=read_archive(args.sources,['14783505472.zip','supplement-result.zip'],supp_sha)
    gdoc=obj(g,'RESULT.json')
    negatives=obj(s,'score/RESULT.json')['scored']
    neg_screen={r['id']:r for r in obj(s,'SCREEN.json')['rows']}
    records=[dict(r,cls='positive') for r in gdoc['records']]
    records += [dict(r, cls='ordinary', wav_sha256=neg_screen[r['id']]['wav_sha256']) for r in negatives if r['gains']['1.0']['events']>0]
    check(len(records)==57 and sum(r['cls']=='positive' for r in records)==44,'expected source cohort')
    archives=gdoc['archives'].copy()
    archives['supplement']=[14782806405,'47d3076b74ea5c6392e00b2bf110afcb91791e91a4a4b9e1fe78b0d2e2cf9bec']
    source_index, packages, providers, identities, meta={},{},{},{},{}
    for key,pair in locks.items():
        art,digest=archives[key]
        z=read_archive(args.sources,[str(art)+'.zip',key+'.zip','train24.zip' if key=='supplement' else '__none'],digest)
        manifest=obj(z,'MANIFEST.json')['rows']
        for r in manifest:
            if r['role']=='train':
                source_index[r['id']]=(key,r)
        packages[key]=z
        for prov,lock in pair.items():
            obs=read_archive(args.sources,[str(lock['id'])+'.zip',('b11-observer-'+prov+'.zip') if key=='supplement' else '__none'],lock['sha256'])
            voc=obj(obs,'VOCAB.json')
            vv=voc['vocab'];m=obj(obs,'MODEL.json')
            check(voc['model']==m['model'] and voc['revision']==m['revision'] and voc['pad_token_id']==0 and sorted(vv.values())==list(range(392)),'original vocab')
            stamp={k:m[k] for k in ('model','revision','weights_sha256')}
            if prov in identities:
                check(identities[prov]==stamp, 'provider drift')
            identities[prov]=stamp
            meta[key+'/'+prov]={'vocab_sha256':sha(obs.read('VOCAB.json')),**lock}
            providers[key,prov]=(obs,{v:k for k,v in vv.items()})
    final=[]
    for row in records:
        uid=row['id'];key,src=source_index[uid]
        wavdata=packages[key].read('wav/'+uid+'.wav')
        with wave.open(io.BytesIO(wavdata)) as wav:
            check(wav.getframerate()==16000 and wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getnframes()==src['samples'],'PCM16 source')
            pcm=wav.readframes(wav.getnframes())
        check(sha(wavdata)==row['wav_sha256']==src['wav_sha256'] and sha(pcm)==row['pcm_sha256']==src['pcm_sha256'] and row['group']==src['source_group_id'] and row['tier'] in ('weak','silver'),'admitted identity')
        outs={}
        for prov in ('xlsr','lv60'):
            z,vocab=providers[key,prov]
            capture=obj(z,'rows/'+uid+'/CAPTURE.json');raw=z.read('logits/'+uid+'.f32')
            check(capture['status']=='success' and capture['input_coverage_complete'] and not capture['target_conditioned'] and capture['source_samples']==src['samples'] and capture['source_pcm_sha256']==row['pcm_sha256'] and capture['source_wav_sha256']==row['wav_sha256'] and capture['id']==uid and capture['provider']==prov and sha(raw)==capture['logits_sha256'],'native observation')
            teacher_frames=1+(src['samples']-400)//320
            mat=np.frombuffer(raw,dtype='<f4').reshape((-1,392));check(np.isfinite(mat).all() and len(mat)==capture['logit_frames']==teacher_frames,'complete frame grid')
            phones, peaks=free_path(mat,vocab)
            check(phones==capture['raw_phone_tokens'],'free decode vs saved')
            outs[prov]={'tokens':phones,'frames':len(mat),'path_matches':find_matches(phones,peaks),'logits_sha256':sha(raw)}
        is_pos=row['cls']=='positive'
        fast=row.get('delivery',{}).get('verified_faster',False)
        known=bool(is_pos and fast and row['gains']['1.0']['events'] and not row['gains']['0.5']['events'])
        matches=[[x for x in outs[k]['path_matches'] if x['full']] for k in ['xlsr','lv60']]
        deltas=None
        if all(matches):
            deltas=[abs(a-b) for a,b in zip(matches[0][0]['peak_frames'],matches[1][0]['peak_frames'])]
        final.append({'id':uid,'group':row['group'],'observed_class':row['cls'],'tier':row['tier'],'verified_faster':fast,'known_fast_half_miss':known,'wav_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256'],'samples':src['samples'],'source':key,'observers':outs,'joint_full':bool(all(matches)),'peak_difference_frames':deltas,'aligned_within_40ms':deltas is not None and max(deltas)<=2})
    def population(rows):
        return {'originals':len(rows),'groups':len({r['group'] for r in rows}), 'any_full':sum(any(any(m['full'] for m in arm['path_matches']) for arm in r['observers'].values()) for r in rows),'both_full':sum(r['joint_full'] for r in rows),'both_anchors_within_40ms':sum(r['aligned_within_40ms'] for r in rows),'aligned_distinct_groups':len({r['group'] for r in rows if r['aligned_within_40ms']}),'any_contiguous_wake':sum(any(arm['path_matches'] for arm in r['observers'].values()) for r in rows)}
    pops={'positive':population([r for r in final if r['observed_class']=='positive']),'fast':population([r for r in final if r['verified_faster']]),'known_fast_half_miss':population([r for r in final if r['known_fast_half_miss']]),'hard_ordinary':population([r for r in final if r['observed_class']=='ordinary'])}
    result={'schema':'english-mero-train-native-phone-free-anchors-v1','main':'8709eca5be5256034a09541b5ccad8e137b4c7a9','design_sha256':sha((root/'DESIGN.json').read_bytes()),'native_observer_runs':0,'optimizer_updates':0,'dev_cal_locked_opened':False,'input_receipts':{'positive':{'id':14782234724,'sha256':gain_sha},'ordinary':{'id':14783505472,'sha256':supp_sha}},'source_archives':{k:archives[k] for k in locks},'native_observers':identities,'observer_archives':meta,'measured_gain':1,'acoustic_human_gold':False,'grid':{'sample_rate':16000,'conv_kernel':[10,3,3,3,3,2,2],'conv_stride':[5,2,2,2,2,2,2],'feature_receptive_field_samples':400,'hop_samples':320,'nominal_frame_center_sample':'320*i+199.5','native_model_context':'bidirectional context over entire utterance; not a physical word endpoint'},'frame_arrays_replayed':2*len(final),'populations':pops,'rows':final}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(pops))
if __name__=='__main__': main()
