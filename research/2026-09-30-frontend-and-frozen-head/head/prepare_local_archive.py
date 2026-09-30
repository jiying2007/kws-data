"""Build a local evidence-only archive; never contact or modify any remote."""
import hashlib,json,pathlib,shutil,tarfile
import numpy as np
R=pathlib.Path(__file__).resolve().parent
DEST=R/'local-archive'
BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def main():
 assert (R/'independent-results-review.json').is_file(),'wait for independent result receipt'
 assert json.loads((R/'run-resource.json').read_text())['success']
 DEST.mkdir(exist_ok=False)
 # Only these evidence extensions and directories are admitted; no input media.
 originals=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md','.log']]
 for p in originals:shutil.copy2(p,DEST/p.name)
 for dirname in ['results','preflight-revisions']:shutil.copytree(R/dirname,DEST/dirname)
 source_names=['run_baseline.py','upstream/wekws/model/fsmn.py','upstream/wekws/model/cmvn.py','upstream/wekws/bin/stream_kws_ctc.py','upstream/torchaudio/kaldi.py','upstream/LICENSE','upstream/torchaudio/LICENSE']
 refs=[]
 for name in source_names:
  p=BASE/name;to=DEST/'reference-source'/name;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,to)
  refs.append({'path':str(to.relative_to(DEST)),'original_path':str(p),'sha256':sha(p)})
 save(DEST/'reference-source-index.json',refs)
 (DEST/'REFERENCE-NOTICE.md').write_text('''# Input/reference boundary

This archive includes experiment source, pinned upstream Python reference source and upstream licenses, initialized-state hashes, saved trained heads and normalization, pooled vectors, all42 Qwen readouts, step curves, identity manifests and guard/review receipts. It contains no WAV/raw audio, donor checkpoint or complete donor parameter tensor payload. Dataset/checkpoint paths and SHA256 values are references to separately controlled inputs, not embedded inputs. Original source-provided FLEURS admitted_for_training=false labels are preserved in the experiment manifest with the narrow recipe-specific authorization.

The saved-head verification works from the cached pooled vectors without donor weights or audio. Full acoustic replay requires the separately pinned external inputs and a fresh explicitly approved execution; the one-shot original journal must not be removed to rerun. The archived experiment's strict8/8 criterion failed in both arms. Do not treat this artifact as a deployable streaming model. No upload, push, publish, pull request or integration with any active publishing package was performed.

Pinned WeKws and torchaudio source/license identities remain available in reference-source-index.json and spec.json. The published donor's original source/license provenance remains referenced by the baseline package and is not rewritten as self-authored work.
''')
 a=np.load(R/'results/pooled-normalization-head-artifacts.npz',allow_pickle=False)
 names=[f'{arm}_{field}' for arm in ['P','R'] for field in ['head_weight','head_bias','norm_mean','norm_std','norm_raw_std']]
 np.savez(DEST/'head-parameters-and-normalization.npz',**{n:a[n] for n in names})
 check=np.load(DEST/'head-parameters-and-normalization.npz',allow_pickle=False)
 assert all(np.array_equal(check[n],a[n]) for n in names)
 assert all(a[f'{arm}_head_weight'].size+a[f'{arm}_head_bias'].size==282 for arm in ['P','R'])
 files=[]
 for p in sorted(DEST.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink();assert p.suffix.lower() not in ['.wav','.flac','.pcm','.pt','.pth','.bin','.so','.parquet']
  files.append({'path':str(p.relative_to(DEST)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(DEST/'artifact-index.json',{'scope':'separate-local-batch-evidence-only','files':files,'excluded':['raw audio','donor checkpoint/weights','native C binary','current publishing workspace'],'external_writes':False})
 target=R/'kws-frozen-keyword-head-pr-v1-evidence.tar.gz'
 with tarfile.open(target,'x:gz') as t:t.add(DEST,arcname='kws-frozen-keyword-head-pr-v1')
 with tarfile.open(target,'r:gz') as t:
  for m in t.getmembers():
   assert not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts
   if m.isfile():
    rel=str(pathlib.PurePosixPath(m.name).relative_to('kws-frozen-keyword-head-pr-v1'))
    assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(DEST/rel)
 receipt={'archive_path':str(target),'archive_bytes':target.stat().st_size,'archive_sha256':sha(target),
          'artifact_index_path':str(DEST/'artifact-index.json'),'artifact_index_sha256':sha(DEST/'artifact-index.json'),
          'included_regular_files':len(files)+1,'all_tar_member_bytes_verified':True,'external_writes':False,
          'classification_result':'P/R strict8/8 FAIL; narrow P<R dev BCE both outputs only'}
 save(R/'local-archive-receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
