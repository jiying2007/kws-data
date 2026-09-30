"""Execute reviewed two-window frontend instrumentation once; no acoustic model."""
import ast,ctypes as C,hashlib,json,math,pathlib,subprocess,sys,shutil
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((R/'preflight.json').read_text())
for name,digest in spec['sources'].items():assert sha(R/name)==digest
assert sha(pathlib.Path(__file__))==spec['execution']['script_sha256']
for name in ['result.json','traces.npz','instrumented.so']:assert not (R/name).exists(),('refuse overwrite',name)
compiler=pathlib.Path(shutil.which('gcc')).resolve()
assert str(compiler)==spec['execution']['compiler_path'] and sha(compiler)==spec['execution']['compiler_sha256']
assert subprocess.check_output([str(compiler),'--version'],text=True)==spec['execution']['compiler_version']
assert torch.__version__==spec['execution']['torch'] and np.__version__==spec['execution']['numpy']
torch.set_num_threads(1);torch.set_num_interop_threads(1)
cmd=[str(compiler),'-O2','-std=c11','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Wcast-qual','-Werror','-ffp-contract=off','-fPIC','-shared',str(R/'source/donor_fbank_instrumented.c'),'-lm','-o',str(R/'instrumented.so')]
subprocess.run(cmd,check=True)
lib=C.CDLL(str(R/'instrumented.so'));F=C.c_float
class Trace(C.Structure):_fields_=[('dc',F*400),('preemphasis',F*400),('windowed',F*512),('power',F*257),('mel',F*80),('logfbank',F*80)]
lib.donor_fbank_state_bytes.restype=C.c_size_t
lib.donor_fbank_init.argtypes=[C.c_void_p]
lib.donor_fbank_analyze_frame.argtypes=[C.c_void_p,C.POINTER(C.c_int16),C.POINTER(Trace)]
tree=ast.parse((R/'source/kaldi.instrumented.py').read_text())
tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='torchaudio' for a in n.names)) and not (isinstance(n,ast.FunctionDef) and n.name in ('mfcc','_get_dct_matrix'))]
ns={'TRACE':{}};exec(compile(tree,'kaldi.instrumented.py','exec'),ns)
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
 inp=R/case['window_path'];assert sha(inp)==case['window_sha256'];pcm=np.frombuffer(inp.read_bytes(),dtype='<i2').copy();assert pcm.shape==(400,)
 ns['TRACE'].clear()
 log=ns['fbank'](torch.from_numpy(pcm.astype(np.float32)).reshape(1,400),num_mel_bins=80,frame_length=25,frame_shift=10,dither=0,energy_floor=0.0,sample_frequency=16000,window_type='hamming')
 py={k:v.numpy().copy().reshape(-1) for k,v in ns['TRACE'].items()};py['logfbank']=log.numpy().reshape(-1).copy()
 state=(C.c_uint64*((lib.donor_fbank_state_bytes()+7)//8))();assert lib.donor_fbank_init(state)==0;tr=Trace();assert lib.donor_fbank_analyze_frame(state,pcm.ctypes.data_as(C.POINTER(C.c_int16)),C.byref(tr))==0
 c={name:np.ctypeslib.as_array(getattr(tr,name)).copy() for name,_ in Trace._fields_}
 c['fft_complex']=np.ctypeslib.as_array((F*512).in_dll(lib,'diag_re')).copy().astype('d')+1j*np.ctypeslib.as_array((F*512).in_dll(lib,'diag_im')).copy().astype('d')
 historical={}
 for label,values in [('reference',py),('native',c)]:
  with np.load(case['archives'][label]['path']) as z:old=z[f"call{case['call']}_fbank"][case['row']]
  historical[label]={'all80_bit_exact':bool(np.array_equal(old.view('u4'),values['logfbank'].view('u4'))),'max_abs':float(np.max(abs(old.astype('d')-values['logfbank'].astype('d')))),'selected_old':float(old[case['mel']]),'selected_now':float(values['logfbank'][case['mel']])}
 layer={}
 for k in ['dc','preemphasis','windowed','power','mel','logfbank']:
  delta=c[k].astype('d')-py[k].astype('d');loc=np.flatnonzero(delta);layer[k]={'different_elements':len(loc),'max_abs':float(np.max(abs(delta))),'first_difference':None if not len(loc) else int(loc[0])}
 selected=[];m=case['mel'];dfts={side:{k:dft(values['windowed'],k) for k in case['fft_bins']} for side,values in [('python',py),('c',c)]}
 for k in case['fft_bins']:
  selected.append({'bin':k,'python':pair(py['fft_complex'][k]),'c':pair(c['fft_complex'][k]),'dft_python_window':pair(dfts['python'][k]),'dft_c_window':pair(dfts['c'][k]),'python_abs_error_own_dft':abs(py['fft_complex'][k]-dfts['python'][k]),'c_abs_error_own_dft':abs(c['fft_complex'][k]-dfts['c'][k]),'python_power':float(py['power'][k]),'c_power':float(c['power'][k])})
 energies={}
 for side,values in [('python',py),('c',c)]:
  energies[side]={'stored_mel':float(values['mel'][m]),'stored_log':float(values['logfbank'][m]),'double_sum_saved_power':math.fsum(float(values['power'][bins[j]])*weights[j] for j in range(o[m],o[m+1])),'double_dft_mel':math.fsum(abs(dfts[side][bins[j]])**2*weights[j] for j in range(o[m],o[m+1]))}
  energies[side]['double_dft_log']=math.log(energies[side]['double_dft_mel'])
 for side,values in [('python',py),('c',c)]:
  for key,val in values.items():
   assert np.all(np.isfinite(val)),(ci,side,key,'nonfinite')
   arrays[f'case{ci}_{side}_{key}']=val
 results.append({'case':case,'historical_reproduction':historical,'layers':layer,'fft':selected,'mel_weights':[(bins[j],weights[j]) for j in range(o[m],o[m+1])],'energies':energies})
np.savez(R/'traces.npz',**arrays)
report={'scope':spec['scope'],'preflight_sha256':sha(R/'preflight.json'),'script_sha256':sha(pathlib.Path(__file__)),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'compiler':spec['execution'],'compile_command':cmd,'library_sha256':sha(R/'instrumented.so'),'traces_sha256':sha(R/'traces.npz'),'cases':results}
(R/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False))
