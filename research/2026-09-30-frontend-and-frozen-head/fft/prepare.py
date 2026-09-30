import pathlib,json,hashlib,wave,shutil,difflib
R=pathlib.Path(__file__).resolve().parent
C=pathlib.Path('/workspace/shared/kws-native-full-fsmn/research/donor_fbank')
P=pathlib.Path('/workspace/shared/kws-cfsmn-baseline/upstream/torchaudio/kaldi.py')
D=pathlib.Path('/workspace/shared/kws-fsmn-pcm-diagnostic-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback/native-export-receipt.json');assert sha(receipt)=='8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5'
rows=[r for d in json.loads(receipt.read_text())['datasets'] for r in d['recordings']]
ref=json.loads((D/'reference/receipt.json').read_text())['recordings'];nat=json.loads((D/'native/result.json').read_text())['recordings'];cases=[]
for idx,frame,call,row,mel,bins in [(1,91,3,3,4,[4,5]),(12,269,9,1,9,[8,9])]:
 r=ref[idx];original=next(x for x in rows if x['recording']==r['recording']);p=pathlib.Path('/workspace/shared/kws-data-pinned')/original['path'];assert sha(p)==r['source_sha256']
 with wave.open(str(p)) as w: assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000);pcm=w.readframes(w.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==r['pcm_sha256'];window=pcm[frame*320:frame*320+800];assert len(window)==800
 dest=R/'inputs'/f'{idx:02}.pcm16le';dest.write_bytes(window)
 archives={}
 for label,rr in [('reference',r),('native',nat[idx])]:
  npz=D/label/rr['arrays_file'];assert sha(npz)==rr['arrays_sha256'];archives[label]={'path':str(npz),'sha256':sha(npz)}
 cases.append(dict(recording=r['recording'],source_path=str(p),source_sha256=sha(p),source_pcm_sha256=r['pcm_sha256'],window_path=str(dest.relative_to(R)),window_sha256=sha(dest),frame=frame,start_sample=frame*160,end_sample=frame*160+400,call=call,row=row,mel=mel,fft_bins=bins,archives=archives))
for f in ['donor_fbank.c','donor_fbank.h','frontend_tables.h','fft_twiddles.h']:shutil.copyfile(C/f,R/'source'/f)
s=(C/'donor_fbank.c').read_text();anchor=' fft512(s);\n /* Reuse real scratch'
assert s.count(anchor)==1
s=s.replace(anchor,' fft512(s);\n memcpy(diag_re,s->re,sizeof(diag_re));memcpy(diag_im,s->im,sizeof(diag_im));\n /* Reuse real scratch')
s=s.replace('static void analyze(', 'float diag_re[512],diag_im[512];\nstatic void analyze(')
(R/'source/donor_fbank_instrumented.c').write_text(s)
(R/'source/c-instrumentation.diff').write_text(''.join(difflib.unified_diff((C/'donor_fbank.c').read_text().splitlines(True),s.splitlines(True),fromfile='original',tofile='instrumented')))
shutil.copyfile(P,R/'source/kaldi.original.py');s=P.read_text()
def add(anchor,extra):
 global s
 assert s.count(anchor)==1,(anchor,s.count(anchor));s=s.replace(anchor,anchor+extra)
add('        strided_input = strided_input - row_means\n','        TRACE["dc"] = strided_input.clone()\n')
add('        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]\n','        TRACE["preemphasis"] = strided_input.clone()\n')
add('    strided_input = strided_input * window_function  # size (m, window_size)\n','    TRACE["windowed400"] = strided_input.clone()\n')
anchor='    spectrum = torch.fft.rfft(strided_input).abs()';assert s.count(anchor)==1
s=s.replace(anchor,'    TRACE["windowed"] = strided_input.clone()\n    fft_complex = torch.fft.rfft(strided_input)\n    TRACE["fft_complex"] = fft_complex.clone()\n    spectrum = fft_complex.abs()')
add('        spectrum = spectrum.pow(2.0)\n','        TRACE["power"] = spectrum.clone()\n')
add('    mel_energies = torch.mm(spectrum, mel_energies.T)\n','    TRACE["mel"] = mel_energies.clone()\n')
(R/'source/kaldi.instrumented.py').write_text(s)
(R/'source/python-instrumentation.diff').write_text(''.join(difflib.unified_diff(P.read_text().splitlines(True),s.splitlines(True),fromfile='original',tofile='instrumented')))
spec=dict(scope='Two fixed 400-sample windows only; frontend-only diagnosis, no model/full-clip inference or numerical-gate changes',cases=cases,sources={str(p.relative_to(R)):sha(p) for p in sorted((R/'source').iterdir())},reference_receipt_sha256=sha(D/'reference/receipt.json'),native_result_sha256=sha(D/'native/result.json'),dft='Independent float64 direct DFT, math.cos/sin with modular bin*n phase and math.fsum; four selected bins; both independently captured window arrays',limitations=['Single-frame Torch FFT may differ from original batched execution. Compare with saved NPZ; do not silently substitute isolated results as historical trace.','Instrumentation copies intermediates only; no changed arithmetic.','No accuracy pass or correction claim from two selected windows.'])
(R/'preflight.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(cases,indent=2,ensure_ascii=False))
