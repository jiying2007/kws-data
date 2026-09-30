"""Stream retained JSONL and recompute source-labelled counts; no inference."""
import collections,json,pathlib
from driver import ROOT,dump,sha,event_identity
def summarize(route):
 p=ROOT/'runs'/route;done=json.loads((p/'complete.json').read_text());guard=json.loads((p/'supervisor.json').read_text());assert done['status']==guard['status']=='completed'
 manifest=json.loads((ROOT/'inputs.json').read_text());negative={r['recording']:r for r in manifest['rows']};seen=collections.defaultdict(set);groups={};qwen={}
 with (p/'records.jsonl').open() as f:
  for line in f:
   r=json.loads(line);stage=r['stage'];n=r['recording'];assert n not in seen[stage];seen[stage].add(n)
   if stage=='himia':
    assert all(r[k]==v for k,v in negative[n].items())
    dims=[('all','all'),('speaker',r['speaker_id']),('official_weak_text',r['official_text']),('source_speed',r['source_speed'])]
   else:
    ident=(r['file_sha256'],r['pcm_sha256'],event_identity(r['events']))
    if stage=='qwen_before':qwen[n]=ident
    else:assert ident==qwen[n]
    dims=[('all','all'),('native_role',r['split']),('review',r['review_method'])]
   for dim,value in dims:
    g=groups.setdefault((stage,dim,value),collections.Counter());g['clips']+=1;g['events']+=len(r['events']);g['triggered_clips']+=bool(r['events']);g['frames']+=r['frames']
    for kw in [1,2]:g[f'keyword_{kw}_events']+=sum(e['keyword_id']==kw for e in r['events'])
    if r.get('kind')=='positive':
     hits=sum(e['keyword_id']==r['keyword_id'] for e in r['events']);g['positive_clips']+=1;g['positive_hits']+=bool(hits);g['wrong_keyword_events']+=len(r['events'])-hits;g['additional_target_events']+=max(0,hits-1)
    elif stage!='himia':g['confusable_clips']+=1;g['confusable_triggered_clips']+=bool(r['events'])
 assert seen['himia']==set(negative) and len(seen['qwen_before'])==42 and seen['qwen_before']==seen['qwen_after']
 result=dict(scope='source-labelled-neighbor-clip-events-not-continuous-FAR',route=route,raw_sha256=sha(p/'records.jsonl'),groups=[dict(stage=k[0],dimension=k[1],value=k[2],**v) for k,v in sorted(groups.items())]);dump(p/'recomputed.json',result);return result
if __name__=='__main__':
 import sys
 summarize(sys.argv[1])
