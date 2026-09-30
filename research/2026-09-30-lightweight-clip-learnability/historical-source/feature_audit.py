"""Post-result descriptive audit only: never updates a model or selects thresholds."""
import json
import pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
rows = json.loads((ROOT/'rows.json').read_text())
cache = np.load(ROOT/'features.npz')
train_rows = [r for r in rows if r['review_method']=='human' and r['split']=='train']
train = np.concatenate([cache[r['recording']] for r in train_rows])
mean, std = train.mean(0), train.std(0)
std = np.maximum(std, .05)

def summarize(subset):
    frames = np.concatenate([cache[r['recording']] for r in subset])
    dbfs = np.asarray([json.loads(line)['dbfs'] for r in subset
                      for line in (ROOT/'features'/(r['recording']+'.jsonl')).read_text().splitlines()])
    centroid = frames.mean(0)
    return {'clips':len(subset),'frames':len(frames),'seconds_total':sum(r['duration_s'] for r in subset),
            'duration_median_s':float(np.median([r['duration_s'] for r in subset])),
            'feature_std':float(frames.std()),'feature_min':float(frames.min()),'feature_max':float(frames.max()),
            'rms_train_standardized_centroid_shift':float(np.sqrt(np.mean(((centroid-mean)/std)**2))),
            'dbfs_p10_p50_p90':np.quantile(dbfs,[.1,.5,.9]).tolist(),
            'frame_fraction_dbfs_above_minus50':float((dbfs>-50).mean())}
report={'scope':'post-result-descriptive-not-causal-not-significance-test',
        'current_normalization':'C per-frame mean across32mel dimensions then fixed configured scale; no fitted train statistics, no corpus CMVN or per-utterance CMVN',
        'diagnostic_standardization':'mean/std fitted to human12training frames only, std floor0.05; NEVER applied to existing trained models',
        'groups':{},'human_speakers':{}}
for method in ('human','asr'):
 for role in ('train','development_a','development_b'):
  subset=[r for r in rows if r['review_method']==method and r['split']==role]
  if subset:report['groups'][method+':'+role]=summarize(subset)
for speaker in sorted({r['speaker_id'] for r in rows if r['review_method']=='human'}):
 report['human_speakers'][speaker]=summarize([r for r in rows if r['review_method']=='human' and r['speaker_id']==speaker])
(ROOT/'feature-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
