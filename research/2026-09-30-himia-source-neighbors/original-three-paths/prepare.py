"""Identity scan only. No model imports or inference; one WAV in memory."""
import hashlib,io,json,pathlib,tarfile,wave,time
ROOT=pathlib.Path(__file__).resolve().parent
SRC=pathlib.Path('/workspace/shared/kws-real-negative-preflight')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inspect(raw):
 with wave.open(io.BytesIO(raw)) as w:
  if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())!=(1,2,16000,'NONE'):raise ValueError('WAV format')
  n=w.getnframes();pcm=w.readframes(n+1)
 if not n or len(pcm)!=n*2:raise ValueError('empty/truncated PCM')
 return dict(frames=n,file_sha256=hashlib.sha256(raw).hexdigest(),pcm_sha256=hashlib.sha256(pcm).hexdigest())
def main():
 start=time.monotonic();out=ROOT/'inputs';out.mkdir(exist_ok=False)
 expected={'slr120-data.tgz':'5de169ac1931a0eab46546c477965edad23069f0a2c0c4eb14e798814f83c91a','slr120-resource.tgz':'8628c75e6ec534a9458229d5ac5267e696b953b5f905c26f889187f6de1ee4e4','transcription.original.txt':'4e2ab8e9bd098e066c0f4dbe86e77325957a63f719eeb74d78124c0bebe8c13c'}
 for p,h in expected.items():assert sha(SRC/p)==h,p
 lines=[l.split() for l in (SRC/'transcription.original.txt').read_text().splitlines()];labels=dict(lines);assert len(labels)==len(lines)==16343
 speakers=sorted({n.split('_')[0] for n in labels},key=lambda s:hashlib.sha256(('kws-himia-hardneg-20260928:'+s).encode()).hexdigest());dev=set(speakers[20:]);assert len(dev)==15
 selected={n for n in labels if n.split('_')[0] in dev};assert len(selected)==7006
 assert hashlib.sha256(('\n'.join(sorted(selected))+'\n').encode()).hexdigest()=='5011b2faa1eb5bbb3212c65a48a88f384748af1b034a7517cb910d5e010bcc5a'
 rows=[];seen=set();total=0
 with tarfile.open(SRC/'slr120-data.tgz','r|gz') as tf:
  for member in tf:
   if time.monotonic()-start>300:raise TimeoutError('identity scan wall budget')
   n=pathlib.PurePosixPath(member.name).name
   if n not in selected:continue
   if not member.isfile() or member.name!='16k_wav_file/'+n or n in seen:raise ValueError('duplicate/unsafe member')
   seen.add(n);total+=member.size
   if member.size>200000 or total>350000000:raise ValueError('size budget')
   raw=tf.extractfile(member).read();info=inspect(raw);(out/n).write_bytes(raw)
   rows.append(dict(recording=n,path=str(out/n),speaker_id=n.split('_')[0],official_text=labels[n],source_speed=n.split('_')[3],split='observed_development',label_authority='official-source-weak-text',**info))
 assert seen==selected;rows.sort(key=lambda r:r['recording']);assert sum(r['frames'] for r in rows)==161688053
 groups={}
 for r in rows:groups.setdefault(r['pcm_sha256'],[]).append(r['recording'])
 report=dict(schema_version=1,source_hashes=expected,rows=rows,duplicates=[v for v in groups.values() if len(v)>1],seconds=sum(r['frames'] for r in rows)/16000,model_inference=False)
 (ROOT/'inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(clips=len(rows),seconds=report['seconds'],duplicate_groups=len(report['duplicates']),sha256=sha(ROOT/'inputs.json'))))
if __name__=='__main__':main()
