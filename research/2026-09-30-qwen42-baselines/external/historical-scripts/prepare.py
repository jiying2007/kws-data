import urllib.request,json,hashlib,pathlib,concurrent.futures
root=pathlib.Path('/workspace/shared/kws-external-baseline')
rev='3787015f084cb241dfa0e4ba237703a2d4322d50'; base='https://modelscope.cn/api/v1/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/repo?Revision='+rev+'&FilePath='
model={
'encoder-epoch-12-avg-2-chunk-16-left-64.onnx':'24af8573b2425ee2e707d57653d4697a31fdf3a6528fef6e8dc9452d28dc9750',
'decoder-epoch-12-avg-2-chunk-16-left-64.onnx':'bb3d8640cc6a495088707173bc1707a8a4ffe014594fc9adcba8389e27d0339f',
'joiner-epoch-12-avg-2-chunk-16-left-64.onnx':'d990f5b67ae9d740048709f5d5f3e777af442061896a8322d5c3ab3ec480f3cb',
'tokens.txt':'72316508d9119696145abc6f1f8cdc46287535c34e5ce7e595f845cb1499cf2e',
'README.md':'dd0ec68b5d9ee736f016fc982078bd3cdbccc69057e9910bdd45580992d13367'}
jobs=[(base+n,root/'models'/n,h) for n,h in model.items()]
for package,version in [('sherpa-onnx','1.13.8'),('sherpa-onnx-core','1.13.8'),('numpy','2.2.6')]:
 u=f'https://pypi.org/pypi/{package}/{version}/json';r=json.load(urllib.request.urlopen(u,timeout=30));(root/'evidence'/f'{package}-{version}-pypi.json').write_text(json.dumps(r,indent=2))
 fs=[f for f in r['urls'] if 'manylinux' in f['filename'] and 'x86_64' in f['filename'] and ('cp312-cp312' in f['filename'] or 'py3-none' in f['filename'])]
 assert len(fs)==1,fs
 f=fs[0];jobs.append((f['url'],root/'wheels'/f['filename'],f['digests']['sha256']))
(root/'evidence'/'downloads.json').write_text(json.dumps([{'url':u,'path':str(p.relative_to(root)),'expected_sha256':h} for u,p,h in jobs],indent=2))
def get(job):
 u,p,h=job;b=urllib.request.urlopen(u,timeout=120).read();actual=hashlib.sha256(b).hexdigest();assert actual==h,(str(p),actual,h);p.write_bytes(b);print(p.name,len(b),actual,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(get,jobs))
