"""Check frame clocks, byte-exact listening clips and every frozen native logit trace."""
from collections import defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import unittest
import wave
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_natural_train_context_clock as clock


class Clocks(unittest.TestCase):
    def test_blanks_preserve_complete_phrase_and_split_repeated_tokens(self):
        a=clock.decode_events([1,1,0,2,0,3,0,1,0,1,2,3], "B13")
        self.assertEqual([e["token_frames"] for e in a],[[0,3,5],[9,10,11]])
        self.assertEqual([e["trigger_sample"] for e in a],[1312,2272])
        self.assertEqual(clock.decode_events([1,3,2,2], "B14"),[])

    def test_touching_ranges_and_deterministic_adjacent_controls(self):
        e=[dict(token_frames=[10,11,12],trigger_sample=2432),dict(token_frames=[54,55,56],trigger_sample=9472)]
        self.assertEqual(clock.intervals(e,30000),[[0,12672,[0,1]]])
        self.assertEqual(clock.adjacent([[24000,34000,[]]],0,60000),(8000,24000))
        self.assertEqual(clock.adjacent([[1000,3000,[]],[4300,6500,[]]],0,20000),None)


def hexd(raw):
    return sha256(raw).hexdigest()


def replay(folder, source_zip, screen_zip):
    import numpy as np
    raw_index=(folder/'REVIEW_INDEX.json').read_bytes()
    index=json.loads(raw_index);result=json.loads((folder/'RESULT.json').read_bytes())
    assert hexd(raw_index)==result['review_index_sha256']
    raw_logits=(folder/'ALL_FRAME_LOGITS.npz').read_bytes()
    assert hexd(raw_logits)==result['logits_sha256']
    assert index['role']=='train' and index['human_review_completed'] is False
    source=zipfile.ZipFile(source_zip); screen=zipfile.ZipFile(screen_zip)
    manifest=json.loads(source.read('MANIFEST.json'));labels=json.loads(screen.read('SCREEN.json'))
    assert hexd(source.read('MANIFEST.json'))==index['source_manifest_sha256']==labels['source_manifest_sha256']
    assert hexd(screen.read('SCREEN.json'))==index['screen_sha256']
    wanted={x['id'] for x in labels['decisions'] if x['tier'] in ('silver','weak')}
    assert wanted=={x['id'] for x in index['originals']}
    originals={r['id']:r for r in manifest['rows']}; byid={r['id']:r for r in index['originals']}
    records={r['id']:r['models'] for r in result['scored']}
    controls={r['id']:r for r in index['controls']}
    by_window=defaultdict(list)
    for w in index['windows']: by_window[w['original_id']].append(w)
    checked=0
    with np.load(io.BytesIO(raw_logits),allow_pickle=False) as traces:
        assert len(traces.files)==len(records)*4==result['forwards']
        for sid in sorted(wanted):
            r=byid[sid];src=originals[sid]
            raw_wav=(folder/r['wav_path']).read_bytes()
            assert source.read('wav/'+sid+'.wav')==raw_wav
            with wave.open(io.BytesIO(raw_wav)) as f:
                pcm=f.readframes(f.getnframes())
                assert f.getframerate()==16000 and f.getsampwidth()==2 and f.getnchannels()==1
            assert hexd(pcm)==src['pcm_sha256']==r['pcm_sha256'] and len(pcm)//2==r['samples']
            assert r['review_status']=='pending'
            clocks=[]
            for m in clock.NAMES:
                rec=records[sid][m]; values=traces[rec['array']]
                assert values.dtype==np.float32 and values.shape==(1+(r['samples']-512)//160,4) and np.isfinite(values).all()
                assert hexd(values.astype('<f4').tobytes())==rec['raw_f32_sha256']
                tokens=values.argmax(1); previous=np.concatenate(([255],tokens[:-1]))
                selected=np.flatnonzero((tokens!=previous)&(tokens!=0)); labels_=tokens[selected].tolist()
                assert labels_==rec['collapsed_tokens']
                found=[]
                for i in range(len(labels_)-2):
                    if labels_[i:i+3]==[1,2,3]:
                        frames=selected[i:i+3].tolist()
                        times=[j*160+512 for j in frames]
                        found.append(dict(model=m,token_frames=frames,token_ready_samples=times,trigger_sample=times[-1]))
                assert found==rec['events'] and len(found)==r['model_event_counts'][m]
                clocks.extend(found);checked+=1
            windows=sorted(by_window[sid],key=lambda v:v['start_sample'])
            assert sorted((e['model'],e['trigger_sample']) for e in clocks)==sorted((e['model'],e['trigger_sample']) for w in windows for e in w['events'])
            canonical=[]
            for a,b in sorted((max(0,e['token_frames'][0]*160-3200),min(r['samples'],e['trigger_sample']+3200)) for e in clocks):
                if canonical and a<=canonical[-1][1]: canonical[-1][1]=max(canonical[-1][1],b)
                else: canonical.append([a,b])
            assert canonical==[[w['start_sample'],w['end_sample']] for w in windows]
            def check_clip(c):
                a,b=c['start_sample'],c['end_sample'];raw=(folder/c['wav_path']).read_bytes()
                assert c['original_id']==sid and c['review_status']=='pending' and 0<=a<b<=r['samples'] and c['samples']==b-a
                with wave.open(io.BytesIO(raw)) as f:
                    assert (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getnframes())==(1,2,16000,b-a)
                    assert f.readframes(b-a)==pcm[a*2:b*2]
                assert hexd(raw)==c['wav_sha256'] and hexd(pcm[a*2:b*2])==c['pcm_sha256']
            for ix,w in enumerate(windows):
                check_clip(w)
                a,b=canonical[ix];l=canonical[ix-1][1] if ix else 0
                rr=canonical[ix+1][0] if ix+1<len(canonical) else r['samples']
                before=a-max(l,a-16000);after=min(rr,b+16000)-b
                if max(before,after)<8000:
                    assert w['control_id'] is None
                else:
                    c=controls[w['control_id']];check_clip(c)
                    assert [c['start_sample'],c['end_sample']]==([a-before,a] if before>=after else [b,b+after])
                    assert not any(c['start_sample']<=e['trigger_sample']<c['end_sample'] for e in clocks)
    assert result['optimizer_updates']==result['dev_cal_locked_test_forwards']==result['human_labels']==0 and checked==result['forwards']
    print("Verified",checked,"complete raw traces,",len(wanted),"original WAVs,",len(index['windows']),"event windows,",len(controls),"adjacent controls")


if __name__=="__main__":
    if len(sys.argv)==4: replay(*map(Path,sys.argv[1:]))
    else: unittest.main()
