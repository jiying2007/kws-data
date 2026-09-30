import pathlib,json,hashlib,math
import numpy as np
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((R/'preflight.json').read_text())
assert sha(R/'run.py')==spec['execution']['script_sha256']
assert not (R/'result.json').exists()
for name,digest in spec['sources'].items():assert sha(R/name)==digest
traces_hash=sha(R/'traces.npz')
assert traces_hash=='c0136ae5923df7b8a041039c4d8f3b69d327eb4728cb9be0d007800b89723bd4'
tracez=np.load(R/'traces.npz',allow_pickle=False)
def scalar(v):
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
# Read the exact positive mel coefficients from copied C tables, without new math.
import re
t=(R/'source/frontend_tables.h').read_text()
def table(name,cast):return [cast(x.strip()) for x in re.search(name+r'\[\d+\]\s*=\s*\{([^}]+)',t).group(1).split(',') if x.strip()]
o=table('donor_mel_offsets',int);bins=table('donor_mel_bins',int);weights=table('donor_mel_weights',lambda x:float.fromhex(x.removesuffix('f')))
def dft(x,k):
 return complex(math.fsum(float(v)*math.cos(2*math.pi*((i*k)%512)/512) for i,v in enumerate(x)),math.fsum(-float(v)*math.sin(2*math.pi*((i*k)%512)/512) for i,v in enumerate(x)))
def pair(z):return [float(z.real),float(z.imag)]
results=[];arrays={}
for ci,case in enumerate(spec['cases']):
 for entry in case['archives'].values():assert sha(pathlib.Path(entry['path']))==entry['sha256']
 py={k.removeprefix(f'case{ci}_python_'):tracez[k].copy() for k in tracez.files if k.startswith(f'case{ci}_python_')}
 c={k.removeprefix(f'case{ci}_c_'):tracez[k].copy() for k in tracez.files if k.startswith(f'case{ci}_c_')}
 historical={}
 for label,values in [('reference',py),('native',c)]:
  with np.load(case['archives'][label]['path']) as z:old=z[f"call{case['call']}_fbank"][case['row']]
  historical[label]={'all80_bit_exact':bool(np.array_equal(old.view('u4'),values['logfbank'].view('u4'))),'max_abs':float(np.max(abs(old.astype('d')-values['logfbank'].astype('d')))),'selected_old':float(old[case['mel']]),'selected_now':float(values['logfbank'][case['mel']])}
 layer={}
 for k in ['dc','preemphasis','windowed','power','mel','logfbank']:
  delta=c[k].astype('d')-py[k].astype('d');loc=np.flatnonzero(delta);layer[k]={'different_elements':len(loc),'max_abs':float(np.max(abs(delta))),'first_difference':None if not len(loc) else int(loc[0])}
 selected=[];m=case['mel'];dfts={side:{k:dft(values['windowed'],k) for k in case['fft_bins']} for side,values in [('python',py),('c',c)]}
 for k in case['fft_bins']:
  selected.append({'bin':k,'python':pair(py['fft_complex'][k]),'c':pair(c['fft_complex'][k]),'dft_python_window':pair(dfts['python'][k]),'dft_c_window':pair(dfts['c'][k]),'python_abs_error_own_dft':abs(complex(py['fft_complex'][k])-dfts['python'][k]),'c_abs_error_own_dft':abs(complex(c['fft_complex'][k])-dfts['c'][k]),'python_power':float(py['power'][k]),'c_power':float(c['power'][k])})
 energies={}
 for side,values in [('python',py),('c',c)]:
  energies[side]={'stored_mel':float(values['mel'][m]),'stored_log':float(values['logfbank'][m]),'double_sum_saved_power':math.fsum(float(values['power'][bins[j]])*weights[j] for j in range(o[m],o[m+1])),'double_dft_mel':math.fsum(abs(dfts[side][bins[j]])**2*weights[j] for j in range(o[m],o[m+1]))}
  energies[side]['double_dft_log']=math.log(energies[side]['double_dft_mel'])
 for side,values in [('python',py),('c',c)]:
  for key,val in values.items():
   assert np.all(np.isfinite(val)),(ci,side,key,'nonfinite')
   arrays[f'case{ci}_{side}_{key}']=val
 results.append({'case':case,'historical_reproduction':historical,'layers':layer,'fft':selected,'mel_weights':[(bins[j],weights[j]) for j in range(o[m],o[m+1])],'energies':energies})
assert sha(R/'traces.npz')==traces_hash
report={'scope':spec['scope'],'preflight_sha256':sha(R/'preflight.json'),'executed_script_sha256':sha(R/'run.py'),'summary_script_sha256':sha(pathlib.Path(__file__)),'compiler':spec['execution'],'library_sha256':sha(R/'instrumented.so'),'traces_sha256':traces_hash,'execution_note':'One frontend execution completed and saved all traces; original final JSON serialization failed on numpy.float32. This script summarizes existing arrays only; no repeated frontend execution.','cases':results}
(R/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n')
print(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False,default=scalar))
