"""Read-only, fail-only authentication of the permanently stopped B24 SOURCE.

Only the one separate B24R1 evaluation registration may cite this proof. The
old source, its unentered rows and sealed reserve never resume. All14 original
ZIPs and complete Git bytes are verified before original captured pure readers
reparse the528 valid native payloads and the one preserved empty return. No
SCREEN is invented. This module grants no model, source or training execution.
"""
from __future__ import annotations
import argparse
import base64
import builtins
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
import urllib.request
import wave
import zipfile

HERE = Path(__file__).resolve().parent
BINDING = HERE / 'english_v6_b24r1_source_failure_inputs.json'
REPO = 'jiying2007/kws-data'
RUN = 45456888767
HEAD = '273e7bd0ae4b16972236f77aacae0eb5d1a12520'
PARENT = 'c76fbc535344cac7d84c0efaf9e6ae904cef9890'
BRANCH = 'research/en-v6-b24-primary-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b24-primary-sources-once.yml'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60')
FAILED_ID = 'b24-primary-284-mero-fast'
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
TRAIN_ASSESSMENT = '3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad'
LOCK_SCHEMA = 'english-v6-b24r1-original-b24-source-stop-lock-v1'
REPORT_SCHEMA = 'english-v6-b24r1-authenticated-original-b24-source-stop-v1'
MAX_ZIP_BYTES = 128_000_000
NAMES = dict(source='english-v6-b24-primary-source-' + str(RUN),
    source_producer='english-v6-b24-primary-source-producer-' + str(RUN),
    transport='english-v6-b24-primary-transport-' + str(RUN),
    source_consumed='english-v6-b24-primary-source-consumed',
    **{'native_' + p: 'b24-primary-dev-observer-' + p for p in PROVIDERS},
    **{'consumed_' + p: 'english-v6-b24-primary-observer-' + p + '-consumed' for p in PROVIDERS})
FALSE_PERMISSIONS = ('old_source_retry_allowed', 'old_source_continuation_allowed',
    'old_sealed_reserve_opened', 'training_allowed', 'primary_allowed',
    'replication_allowed', 'engineering_allowed', 'promotion_allowed',
    'product_released', 'scientific_execution_authorized')

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    """Stable ASCII JSON, sorted keys, compact separators, no newline or NaN."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')

def encoded(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('ascii')

def document(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, 'duplicate original JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON ' + value)))

def pin(raw):
    return dict(bytes=len(raw), sha256=sha(raw))

def validate_lock(lock):
    need(type(lock) is dict and set(lock) == {'schema', 'repo', 'run_id', 'run_head',
        'prepared_parent', 'branch', 'workflow', 'artifacts', 'source_producer_manifest_sha256',
        'source_manifest_sha256', 'history_sha256', 'source_assessment_sha256',
        'candidate_pt_sha256', 'train_result_sha256', 'logs'}, 'complete original STOP lock')
    need(lock['schema'] == LOCK_SCHEMA and lock['repo'] == REPO and lock['run_id'] == RUN
         and type(lock['run_id']) is int and lock['run_head'] == HEAD and lock['prepared_parent'] == PARENT
         and lock['branch'] == BRANCH and lock['workflow'] == WORKFLOW
         and lock['candidate_pt_sha256'] == CANDIDATE_PT
         and lock['source_assessment_sha256'] == TRAIN_ASSESSMENT,
         'only actual B24 SOURCE and unchanged actual B24 candidate')
    need(set(lock['artifacts']) == set(NAMES), 'all14 original artifacts; never a fabricated SCREEN')
    ids = set()
    for role, value in lock['artifacts'].items():
        need(set(value) == {'id', 'name', 'bytes', 'sha256'} and type(value['id']) is int
             and value['id'] > 0 and type(value['bytes']) is int and 0 < value['bytes'] <= MAX_ZIP_BYTES
             and value['name'] == NAMES[role] and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None,
             'exact original artifact pin ' + role)
        ids.add(value['id'])
    need(len(ids) == 14, 'fourteen distinct original artifact identities')
    need(set(lock['logs']) == {'parakeet', 'readback'}, 'both original failed job logs')
    for role, job, path in [('parakeet', 138672328824, 'PARAKEET_JOB.log'), ('readback', 138676788255, 'READBACK_JOB.log')]:
        value = lock['logs'][role]
        need(set(value) == {'path', 'job_id', 'bytes', 'sha256'} and value['path'] == path
             and value['job_id'] == job and type(value['bytes']) is int and value['bytes'] > 0
             and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None, 'exact failed job log ' + role)
    need(all(re.fullmatch('[0-9a-f]{64}', lock[key]) is not None for key in
        ('source_producer_manifest_sha256', 'source_manifest_sha256', 'history_sha256',
         'source_assessment_sha256', 'candidate_pt_sha256', 'train_result_sha256')), 'full frozen receipt hashes')
    return lock

def read_binding(require_enabled=True, path=None):
    value = document(Path(path or BINDING).read_bytes())
    need(set(value) == {'schema', 'enabled', 'condition', 'maximum_new_source_registrations',
        'candidate_pt_sha256', 'original_train_assessment_sha256', 'original_source_lock',
        'expected_failure_report_sha256', 'old_source_retry_allowed', 'old_source_continuation_allowed',
        'new_training_allowed', 'open_old_sealed_reserve', 'second_replacement_allowed'}, 'exact R1 failure opening shape')
    need(value['schema'] == 'english-v6-b24r1-source-stop-opening-v1'
         and value['condition'] == 'only_one_separate_B24R1_evaluation_after_authenticated_B24_SOURCE_STOP'
         and type(value['maximum_new_source_registrations']) is int and value['maximum_new_source_registrations'] == 1
         and value['candidate_pt_sha256'] == CANDIDATE_PT
         and value['original_train_assessment_sha256'] == TRAIN_ASSESSMENT
         and all(value[k] is False for k in ('old_source_retry_allowed', 'old_source_continuation_allowed',
                  'new_training_allowed', 'open_old_sealed_reserve', 'second_replacement_allowed')),
         'one distinct evaluation only; no old continuation, new training, old reserve or automatic R2')
    if value['enabled'] is False:
        need(not require_enabled and value['original_source_lock'] is None
             and value['expected_failure_report_sha256'] is None, 'R1 failure opening is inert')
    else:
        need(value['enabled'] is True and re.fullmatch('[0-9a-f]{64}', value['expected_failure_report_sha256'] or '')
             is not None, 'reviewed complete STOP proof required, never an enabled-only switch')
        validate_lock(value['original_source_lock'])
    return value

@contextmanager
def offline_only():
    old_import, old_popen, old_urlopen = builtins.__import__, subprocess.Popen, urllib.request.urlopen
    old_socket, old_connection = socket.socket, socket.create_connection
    denied = {'torch', 'numpy', 'torchaudio', 'transformers', 'onnxruntime', 'piper', 'soundfile',
              'requests', 'httpx', 'tensorflow', 'jax', 'huggingface_hub'}
    def guarded(name, *args, **kwargs):
        need(name.split('.')[0] not in denied, 'no scientific/network import during STOP readback: ' + name)
        return old_import(name, *args, **kwargs)
    def blocked(*args, **kwargs):
        raise RuntimeError('no process or network during original STOP readback')
    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs):
            blocked()
    builtins.__import__, subprocess.Popen, urllib.request.urlopen = guarded, blocked, blocked
    socket.socket, socket.create_connection = BlockedSocket, blocked
    try:
        yield
        need(not denied.intersection(sys.modules), 'no scientific runtime was imported')
    finally:
        builtins.__import__, subprocess.Popen, urllib.request.urlopen = old_import, old_popen, old_urlopen
        socket.socket, socket.create_connection = old_socket, old_connection

def original(path, expected):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular original ZIP')
    raw = path.read_bytes()
    need(pin(raw) == {key: expected[key] for key in ('bytes', 'sha256')}, 'original ZIP bytes and digest')
    value = zipfile.ZipFile(io.BytesIO(raw))
    infos = value.infolist()
    need(len(infos) <= 10000 and len({i.filename for i in infos}) == len(infos)
         and sum(i.file_size for i in infos) <= MAX_ZIP_BYTES and value.testzip() is None,
         'complete bounded original ZIP CRC and unique members')
    need(all(not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts
             and not (i.flag_bits & 1) and not stat.S_ISLNK(i.external_attr >> 16) for i in infos),
         'safe regular original ZIP members')
    return value

def validate_failure_pattern(counts):
    keys = ('input_started', 'forward_attempted', 'native_returned', 'raw_payload_saved',
            'payload_valid', 'capture_saved', 'failed', 'not_started', 'entered_without_return', 'entered_without_raw')
    need(set(counts) == set(PROVIDERS), 'all five original provider boundaries')
    for provider, row in counts.items():
        need(all(type(row[k]) is int and row[k] >= 0 for k in keys), 'integer original journal counters')
        expected = dict(input_started=120, forward_attempted=120, native_returned=120, raw_payload_saved=120,
                        payload_valid=120, capture_saved=120, failed=0, not_started=0,
                        entered_without_return=0, entered_without_raw=0)
        if provider == 'parakeet':
            expected.update(input_started=49, forward_attempted=49, native_returned=49, raw_payload_saved=49,
                            payload_valid=48, capture_saved=48, failed=1, not_started=71)
        need({key: row[key] for key in keys} == expected
             and row['status'] == ('STOP_OBSERVER_FAILURE' if provider == 'parakeet' else 'COMPLETE'),
             'only exact original known-empty STOP; no success, unknown return or other failure')
    return counts

# Remaining implementation authenticates archived provenance before importing
# the original captured validators; no live installation or model is possible.


def archive_facts(directory, lock):
    validate_lock(lock)
    ROOT = Path(directory).resolve()
    def meta(name):
        return document((ROOT / 'metadata' / ('source_' + name + '.json')).read_bytes())
    run,inventory,commit,tree,jobs=[meta(n) for n in ['run','artifacts','commit','tree','jobs']]
    need(run['id']==RUN and run['head_sha']==HEAD and run['head_branch']==BRANCH and run['status']=='completed' and run['conclusion']=='failure' and run['run_attempt']==1 and run['event']=='push' and run['path']==WORKFLOW,'actual closed failed first SOURCE')
    need(all(run[key]['full_name']==REPO and run[key]['private'] is False for key in ('repository','head_repository')), 'same public original repository')
    need(commit['sha']==HEAD and [p['sha'] for p in commit['parents']]==[PARENT] and [f['filename'] for f in commit['files']]==[WORKFLOW] and tree['sha']==commit['commit']['tree']['sha'] and tree['truncated'] is False,'complete original workflow-only Git')
    need(inventory['total_count']==len(inventory['artifacts'])==14 and {a['name'] for a in inventory['artifacts']}==set(NAMES.values()),'all14 actual artifacts, no SCREEN artifact')
    need({j['name']:j['conclusion'] for j in jobs['jobs']}=={'sources':'success','readback':'failure',**{'observers ('+p+')':('failure' if p=='parakeet' else 'success') for p in PROVIDERS}},'exact original source/provider/readback outcomes')
    need(jobs['total_count']==len(jobs['jobs'])==7 and all(j['run_id']==RUN and j['head_sha']==HEAD and j['run_attempt']==1 and j['status']=='completed' for j in jobs['jobs']), 'all seven original completed first-attempt jobs')
    need(next(j['id'] for j in jobs['jobs'] if j['name']=='observers (parakeet)')==lock['logs']['parakeet']['job_id'] and next(j['id'] for j in jobs['jobs'] if j['name']=='readback')==lock['logs']['readback']['job_id'], 'same two actual failed job log identities')
    archives={};pins={}
    for role,name in NAMES.items():
     item=next(a for a in inventory['artifacts'] if a['name']==name);path=ROOT/(role+'.zip');raw=path.read_bytes()
     need(len(raw)==item['size_in_bytes'] and item['digest']=='sha256:'+sha(raw) and item['expired'] is False and item['workflow_run']['id']==RUN and item['workflow_run']['head_sha']==HEAD,'original artifact identity '+role)
     z=zipfile.ZipFile(io.BytesIO(raw));need(z.testzip() is None and len(z.namelist())==len(set(z.namelist())),'CRC/unique '+role)
     need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'safe original member names')
     expanded=sum(i.file_size for i in z.infolist());need(len(raw)<=128000000 and expanded<=128000000,'original bounds '+role)
     pins[role]=dict(id=item['id'],name=name,run_id=RUN,run_head=HEAD,**pin(raw),members=len(z.infolist()),expanded_bytes=expanded,crc_verified=True);archives[role]=z
    for role in NAMES:
     need({k:pins[role][k] for k in ('id','name','bytes','sha256')}==lock['artifacts'][role], 'exact reviewed original artifact pin '+role)
    producer_raw=archives['source_producer'].read('PRODUCER.json');producer=document(producer_raw);need(sha(producer_raw)==lock['source_producer_manifest_sha256'], 'exact original complete source producer manifest');git={r['path']:r for r in tree['tree']}
    need(producer['schema']=='english-v6-b24-complete-source-producer-v1' and producer['role']=='source' and producer['run_head']==HEAD and str(producer['run_id'])==str(RUN) and producer['run_attempt']==1 and producer['run_branch']==BRANCH and producer['runtime_workflow']==WORKFLOW and producer['prepared_parent']==PARENT and producer['git_tree']==tree['sha'],'actual producer identity')
    need({i.filename for i in archives['source_producer'].infolist() if not i.is_dir()}=={'PRODUCER.json'}|{'repository/'+n for n in producer['files']},'exact full producer members')
    for name,p in producer['files'].items():
     raw=archives['source_producer'].read('repository/'+name);blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
     need(pin(raw)=={k:p[k] for k in ['bytes','sha256']} and blob==p['git_blob_sha1']==git[name]['sha'] and p['mode']==git[name]['mode'] and git[name]['type']=='blob','original producer file/Git '+name)
    template=archives['source_producer'].read('repository/automation/english_v6_b24_source_workflow_template.yml');token=b'__B24_REVIEWED_SOURCE_PARENT__'
    need(template.count(token)==2 and archives['source_producer'].read('repository/'+WORKFLOW)==template.replace(token,PARENT.encode()),'original two-token activation')
    source=archives['source'];manifest_raw=source.read('MANIFEST.json');manifest=json.loads(manifest_raw);rows=manifest['rows'];source_sha=sha(manifest_raw)
    need(source_sha==lock['source_manifest_sha256'] and sha(source.read('HISTORY.json'))==lock['history_sha256'] and manifest['assessment_sha256']==lock['source_assessment_sha256']==TRAIN_ASSESSMENT, 'original frozen source metadata hashes')
    need(manifest['source_execution']==dict(prepared_parent=PARENT,run_attempt=1,run_branch='refs/heads/'+BRANCH,run_head=HEAD,run_id=str(RUN)) and manifest['replication_opened'] is False, 'same original source execution and permanently sealed reserve')
    with zipfile.ZipFile(io.BytesIO(source.read('TRAIN_models.zip'))) as models:
     train_raw=models.read('RESULT.json');train=document(train_raw)
     need(sha(train_raw)==lock['train_result_sha256'] and train['model_pt_sha256']['challenger']==lock['candidate_pt_sha256']==CANDIDATE_PT, 'actual fixed B24 candidate and original completed TRAIN result')
    need(len(rows)==len({r['id'] for r in rows})==len({r['pcm_sha256'] for r in rows})==120 and manifest['tts_calls']==96 and manifest['natural_original_decodes']==24 and manifest['new_kws_forwards']==0,'all120 original fixed sources')
    for row in rows:
     raw=source.read('wav/'+row['id']+'.wav');need(sha(raw)==row['wav_sha256'],'whole WAV '+row['id'])
     with wave.open(io.BytesIO(raw)) as stream:
      need(stream.getnchannels()==1 and stream.getsampwidth()==2 and stream.getframerate()==16000 and stream.getnframes()==row['samples'],'original WAV dimensions')
      pcm=stream.readframes(stream.getnframes());need(not stream.readframes(1) and sha(pcm)==row['pcm_sha256'],'whole original PCM')
     if row['cohort']=='synthetic':
      chunks=[]
      for i,part in enumerate(row['native_chunks']):
       need(part['file']==f'{i:03d}.s16le','original chunk order');chunk=source.read('native/'+row['id']+'/'+part['file']);need(len(chunk)==part['samples']*2 and sha(chunk)==part['native_pcm_sha256'],'native chunk');chunks.append(chunk)
      need(sha(b''.join(chunks))==row['native_pcm_sha256'],'whole native original')
    need(source.read('CONSUMPTION.json')==archives['source_consumed'].read('CONSUMPTION.json') and base64.b64decode(json.loads(source.read('CONSUMPTION_UPLOAD.json'))['original_zip_base64'],validate=True)==(ROOT/'source_consumed.zip').read_bytes(),'original independent source budget')
    for info in archives['transport'].infolist():
     if not info.is_dir():need(archives['transport'].read(info.filename)==source.read(info.filename),'original transport receipt '+info.filename)
    counts={};failed_details=[]
    for provider in PROVIDERS:
     z=archives['native_'+provider];names=set(z.namelist());report=json.loads(z.read('RESULT.json'));stored=json.loads(z.read('LEDGER.json'));closure=json.loads(z.read('FILES.json'))
     files=[dict(path=n,**pin(z.read(n))) for n in sorted(names) if not n.endswith('/') and n!='FILES.json']
     need(closure==dict(schema='english-v6-b18-observer-file-closure-v1',self_excluded='FILES.json',files=files),'every original FILES member '+provider)
     need(base64.b64decode(json.loads(z.read('OBSERVER_BUDGET.json'))['original_zip_base64'],validate=True)==(ROOT/('consumed_'+provider+'.zip')).read_bytes(),'independent consumed ZIP '+provider)
     ledger=[]
     for row in rows:
      prefix='rows/'+row['id']+'/';entry={'id':row['id']}
      for key,name in [('input_started','STARTED.json'),('forward_attempted','FORWARD_STARTED.json'),('native_returned','FORWARD_RETURNED.json'),('payload_valid','PAYLOAD_VALID.json'),('capture_saved','CAPTURE.json'),('failed','FAILED.json')]:
       entry[key]=prefix+name in names
       if entry[key]:entry[key+'_sha256']=sha(z.read(prefix+name))
      entry['raw_payload_saved']=any(prefix+n in names for n in ['RAW_RETURN.json','RAW_PROCESS.json','RAW_PHONE.json']);ledger.append(entry)
     keys=['input_started','forward_attempted','native_returned','raw_payload_saved','payload_valid','capture_saved','failed'];actual={k:sum(int(r[k]) for r in ledger) for k in keys}
     need(stored==dict(schema='english-v6-b18-native-attempt-return-valid-ledger-v1',provider=provider,source_manifest_sha256=source_sha,rows=ledger,counts=actual,attempted_ids=[r['id'] for r in ledger if r['forward_attempted']],returned_ids=[r['id'] for r in ledger if r['native_returned']],payload_valid_ids=[r['id'] for r in ledger if r['payload_valid']]),'all120 original journal rows '+provider)
     need(report['provider']==provider and report['source_manifest_sha256']==source_sha and report['denominator']==120 and report['actual_native_forward_attempts']==actual['forward_attempted'] and report['native_forward_returns']==actual['native_returned'] and report['valid_native_payloads']==actual['payload_valid'] and report['complete']==actual['capture_saved'] and report['retry_count']==report['kws_forward_calls']==0,'original native summary '+provider)
     need(report['observer_code_sha256']==producer['files']['automation/english_v6_b24_source_observe.py']['sha256'] and report['asr_adapter_sha256']==producer['files']['automation/english_v6_b18_asr_adapter.py']['sha256'],'original observer/adapter code binding')
     actual.update(not_started=sum(not r['input_started'] for r in ledger),entered_without_return=sum(r['forward_attempted'] and not r['native_returned'] for r in ledger),entered_without_raw=sum(r['forward_attempted'] and not r['raw_payload_saved'] for r in ledger))
     counts[provider]=dict(status=report['status'],**actual)
     for item in ledger:
      if item['failed']:
       prefix='rows/'+item['id']+'/';process=json.loads(z.read(prefix+'RAW_PROCESS.json'));out=z.read(prefix+'RAW_STDOUT.bin')
       need(provider=='parakeet' and item['id']=='b24-primary-284-mero-fast' and process['native_exit_code']==0 and process['native_return_observed'] is True and process['timed_out'] is False and process['exception_type'] is None and out==b'\n' and report['row_error']=='ValueError: EMPTY_NATIVE_TRANSCRIPT','exact saved genuine empty native return')
       failed_details.append(dict(provider=provider,id=item['id'],native_exit_code=0,timed_out=False,stdout=pin(out),error=report['row_error'],complete_original_raw_return_saved=True))
    need(counts['parakeet']['forward_attempted']==counts['parakeet']['native_returned']==counts['parakeet']['raw_payload_saved']==49 and counts['parakeet']['payload_valid']==counts['parakeet']['capture_saved']==48 and counts['parakeet']['failed']==1 and counts['parakeet']['not_started']==71,'fixed actual Parakeet STOP boundary')
    need(all(counts[p]['status']=='COMPLETE' and all(counts[p][k]==120 for k in ['forward_attempted','native_returned','raw_payload_saved','payload_valid','capture_saved']) and counts[p]['failed']==counts[p]['not_started']==0 for p in PROVIDERS if p!='parakeet'),'four other complete original providers')
    validate_failure_pattern(counts)
    for role,value in lock['logs'].items():
     need(pin((ROOT/value['path']).read_bytes())=={k:value[k] for k in ('bytes','sha256')}, 'exact original failed log '+role)
    need(b'WHOLE_PANEL_STOP_NATIVE_RETURN_OR_PAYLOAD_FAILURE_NO_RETRY' in (ROOT/'PARAKEET_JOB.log').read_bytes(), 'original native process stopped without retry')
    log=(ROOT/'READBACK_JOB.log').read_bytes();need(b'ValueError: ALL_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES' in log and b'No artifacts will be uploaded.' in log and not (ROOT/'screen.zip').exists(),'original readback failure and absent SCREEN')
    for z in archives.values():z.close()
    result=dict(schema='english-v6-b24-original-source-stop-archive-checkpoint-v1',status='FAILED_EMPTY_NATIVE_TRANSCRIPT',run_id=RUN,head=HEAD,prepared_parent=PARENT,run_conclusion='failure',artifact_count=14,original_artifacts=pins,producer_paths=len(producer['files']),producer_manifest_sha256=sha(producer_raw),source_manifest_sha256=source_sha,whole_original_pcm_verified=120,all_original_native_chunks_verified=True,provider_counts=counts,planned_native_calls=600,actual_native_attempts=sum(c['forward_attempted'] for c in counts.values()),actual_native_returns=sum(c['native_returned'] for c in counts.values()),raw_payloads_saved=sum(c['raw_payload_saved'] for c in counts.values()),valid_native_payloads=sum(c['payload_valid'] for c in counts.values()),empty_native_returns=1,never_started=71,entered_without_return=sum(c['entered_without_return'] for c in counts.values()),entered_without_raw=sum(c['entered_without_raw'] for c in counts.values()),failure_details=failed_details,screen_artifact_present=False,source_gate=None,source_admission_performed=False,readback_log=pin(log),final_api={n:pin((ROOT/'metadata'/('source_'+n+'.json')).read_bytes()) for n in ['run','jobs','artifacts','commit','tree']},all14_original_zip_digests_crc_verified=True,all1399_original_producer_git_blobs_verified=len(producer['files'])==1399,all5_original_FILES_closures_and120_journals_recomputed=True,raw_payload_semantic_reparse_pending=True,new_model_calls=0,new_source_calls=0,new_observer_calls=0,primary_allowed=False,replication_allowed=False,engineering_allowed=False,training_allowed=False,continuation_allowed=False,promotion_allowed=False)
    return result


def semantic_readback(directory, lock):
    """Reparse real payload bytes with the immutable original captured readers."""
    directory = Path(directory)
    with tempfile.TemporaryDirectory(prefix='b24-source-stop-captured-') as temporary:
        root = Path(temporary)
        source_dir, providers = root / 'source', root / 'providers'
        for role, target in [('source_producer', root / 'producer'), ('source', source_dir),
                             *[('native_' + p, providers / p) for p in PROVIDERS]]:
            target.mkdir(parents=True)
            with original(directory / (role + '.zip'), lock['artifacts'][role]) as archive:
                archive.extractall(target)
        captured = root / 'producer' / 'repository'
        sys.path.insert(0, str(captured / 'automation'))
        import english_v6_b24_source as source
        owner = source.evaluation_preflight(source_dir)
        rows = owner['rows']
        need(len(rows) == len(owner['all_original_rows']) == 120,
             'original source reader verifies every120 raw whole input and full ancestry')
        obs, raw_asr = source.observers, source.observers.raw_asr
        contract = source.code_contract()
        registration = dict(assessment_sha256=source.ASSESSMENT_SHA, source_schema=source.SCHEMA,
            batch_schema=source.BATCH, source_manifest_sha256=lock['source_manifest_sha256'],
            source_rows=120, max_calls=120, **owner['source_execution'],
            **{key: contract[key] for key in ('observer_sha256', 'asr_adapter_sha256', 'parser_contract_sha256')})
        obs.validate_rows(rows, lock['source_manifest_sha256'], registration)
        parsed, failed, phone_frames = {}, [], 0
        for provider in PROVIDERS:
            folder = providers / provider
            obs.lineage.validate_observer_budget(document((folder / 'OBSERVER_BUDGET.json').read_bytes()), provider, registration)
            started = document((folder / 'STARTED.json').read_bytes())
            need(started['registration'] == registration and started['whole_panel_stop_on_failure'] is True
                 and started['old_admission_function_sha256'] == obs.POLICY_FUNCTIONS,
                 'original frozen registration and unchanged admission function ' + provider)
            report = document((folder / 'RESULT.json').read_bytes())
            model = document((folder / 'MODEL.json').read_bytes())
            by_id = {row['id']: row for row in report['rows']}
            need(len(by_id) == len(report['rows']) == 120 and set(by_id) == {row['id'] for row in rows},
                 'all120 original result slots ' + provider)
            count = 0
            for row in rows:
                point = folder / 'rows' / row['id']
                capture = by_id[row['id']]
                identity = obs.binding(provider, row, lock['source_manifest_sha256'])
                if capture['status'] == 'success':
                    need(capture == document((point / 'CAPTURE.json').read_bytes())
                         and document((point / 'STARTED.json').read_bytes()) == identity
                         and capture['input_coverage_complete'] is True
                         and all(capture.get(k) == v for k, v in identity.items()),
                         'original full capture identity ' + provider)
                    if provider in raw_asr.TEXT_PROVIDERS:
                        need(model['adapter_sha256'] == contract['asr_adapter_sha256']
                             and model['parser_contract_sha256'] == raw_asr.PARSER_CONTRACT_SHA256
                             and model['runtime_versions'] == contract['asr_runtime_versions'][provider]
                             and model['observer'] == contract['asr_models'][provider]
                             and model['raw_before_validation'] is True and model['decode_config_unchanged'] is True,
                             'original fixed ASR runtime and raw parser')
                        raw_asr.validate_capture(provider, point, identity, capture)
                    else:
                        obs._verify_phone(folder, point, identity, capture, model)
                        phone_frames += capture['logit_frames']
                    count += 1
                elif capture['status'] == 'failed':
                    need(provider == 'parakeet' and row['id'] == FAILED_ID
                         and capture == document((point / 'FAILED.json').read_bytes())
                         and capture['error'] == 'ValueError: EMPTY_NATIVE_TRANSCRIPT'
                         and not (point / 'CAPTURE.json').exists() and not (point / 'PAYLOAD_VALID.json').exists()
                         and (point / 'RAW_STDOUT.bin').read_bytes() == b'\n',
                         'one actual saved empty return; no invented successful capture')
                    # This pointer refers only to the actual pinned RAW_PROCESS.
                    # No CAPTURE or validity record is synthesized or written.
                    pointer = dict(raw_evidence=dict(parser_contract_sha256=raw_asr.PARSER_CONTRACT_SHA256,
                        payload=raw_asr.pin(point / 'RAW_PROCESS.json')))
                    try:
                        raw_asr.validate_capture(provider, point, identity, pointer)
                    except ValueError as exc:
                        need(str(exc) == 'EMPTY_NATIVE_TRANSCRIPT', 'original parser rejects exactly the real empty return')
                    else:
                        raise ValueError('a valid native return cannot authorize this failure opening')
                    failed.append(row['id'])
                else:
                    need(provider == 'parakeet' and capture['status'] == 'not_started' and not point.exists(),
                         'all71 never-entered observations remain absent, without reconstruction')
            parsed[provider] = count
        need(parsed == dict(qwen=120, whisper=120, parakeet=48, xlsr=120, lv60=120)
             and failed == [FAILED_ID], '528 real valid payloads and one original rejected empty return')
        modules = []
        for name, value in sys.modules.items():
            file = getattr(value, '__file__', None)
            if file and name.startswith(('english_', 'b24_original_', 'b24r1_original_')):
                need(Path(file).resolve().is_relative_to(captured), 'original captured repository only: ' + name)
                modules.append(name)
        return dict(valid_payloads_reparsed=parsed, total_valid_payloads_reparsed=sum(parsed.values()),
            empty_return_reparsed_by_original_parser=True, empty_return_source_id=FAILED_ID,
            full_original_phone_frames_reparsed=phone_frames, original_source_preflight_recomputed=True,
            all_original120_pcm_and_native_chunks_recomputed=True, captured_repository_modules=sorted(modules),
            original_captured_readers_only=True, checkout_fallback_used=False)


def verify_worker(directory, lock):
    facts = archive_facts(directory, lock)
    semantic = semantic_readback(directory, lock)
    # Raw API serialization and user/actor payloads stay private. The stable
    # report binds exact originals, immutable identity and computed facts only.
    facts.pop('final_api')
    facts.update(schema=REPORT_SCHEMA, raw_payload_semantic_reparse_pending=False,
        original_lock_sha256=sha(canonical(lock)), adapter_sha256=sha(Path(__file__).read_bytes()),
        canonical_encoding='ASCII JSON; sorted keys; compact separators; no newline; no NaN',
        original_metadata_sha256={'MANIFEST.json': lock['source_manifest_sha256'], 'HISTORY.json': lock['history_sha256']},
        original_assessment_sha256=lock['source_assessment_sha256'], candidate_pt_sha256=lock['candidate_pt_sha256'],
        original_train_result_sha256=lock['train_result_sha256'], original_source_lock_schema=LOCK_SCHEMA,
        all14_original_archives_reopened_by_this_helper=True, original_source_stop_permanent=True,
        new_source_registration_performed=False, **semantic, **dict.fromkeys(FALSE_PERMISSIONS, False))
    return facts


def verify(directory, lock):
    """Authenticate complete original failure in a fresh isolated stdlib process."""
    validate_lock(lock)
    need(not {'torch', 'numpy', 'onnxruntime'}.intersection(sys.modules), 'STOP authorization precedes model/array runtime imports')
    with tempfile.TemporaryDirectory(prefix='b24-source-stop-lock-') as temporary:
        path = Path(temporary) / 'lock.json'
        path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()),
            '--verify-worker', str(Path(directory).resolve()), str(path)])
    report = document(raw)
    need(report['schema'] == REPORT_SCHEMA and report['status'] == 'FAILED_EMPTY_NATIVE_TRANSCRIPT'
         and report['original_lock_sha256'] == sha(canonical(lock))
         and report['adapter_sha256'] == sha(Path(__file__).read_bytes())
         and report['total_valid_payloads_reparsed'] == 528 and report['empty_return_reparsed_by_original_parser'] is True
         and all(report[key] is False for key in FALSE_PERMISSIONS), 'complete fail-only isolated STOP proof')
    return report


def fetch(directory, lock):
    """Read-only original collection before a separate R1 source budget exists."""
    validate_lock(lock)
    directory = Path(directory).resolve()
    need(not directory.exists() and directory.parent.is_dir(), 'fresh original STOP evidence directory')
    def gh(endpoint):
        return subprocess.run(['gh', 'api', 'repos/' + REPO + '/' + endpoint, '--allow-escape-sequences'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    directory.mkdir()
    (directory / 'metadata').mkdir()
    for name, endpoint in [('run', 'actions/runs/' + str(RUN)), ('jobs', 'actions/runs/' + str(RUN) + '/jobs?per_page=100'),
                           ('artifacts', 'actions/runs/' + str(RUN) + '/artifacts?per_page=100'), ('commit', 'commits/' + HEAD)]:
        (directory / 'metadata' / ('source_' + name + '.json')).write_bytes(gh(endpoint))
    commit = document((directory / 'metadata/source_commit.json').read_bytes())
    need(commit['sha'] == HEAD, 'original SOURCE commit before following its Git tree')
    (directory / 'metadata/source_tree.json').write_bytes(gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1'))
    for role, expected in lock['artifacts'].items():
        raw = gh('actions/artifacts/' + str(expected['id']) + '/zip')
        need(pin(raw) == {key: expected[key] for key in ('bytes', 'sha256')}, 'complete original fetched ZIP ' + role)
        (directory / (role + '.zip')).write_bytes(raw)
    for role, expected in lock['logs'].items():
        raw = gh('actions/jobs/' + str(expected['job_id']) + '/logs')
        need(pin(raw) == {key: expected[key] for key in ('bytes', 'sha256')}, 'exact original failed job log ' + role)
        (directory / expected['path']).write_bytes(raw)
    return verify(directory, lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    args = parser.parse_args()
    original_lock = document(args.lock.read_bytes())
    if args.verify_worker:
        with offline_only():
            result = verify_worker(args.directory, original_lock)
    else:
        result = verify(args.directory, original_lock)
    sys.stdout.buffer.write(canonical(result))
