"""Read-only proof of the exact earlier acquisition failure and its one omission."""
import hashlib
import io
import json
import stat
import zipfile
from pathlib import PurePosixPath

FAILURE = {'bytes': 4257652, 'sha256': '514b70f69470620612d6ea3dfa84a53369468d0f86a659863a4301c108266e12'}
RUN_ID = '44057968882'
HEAD = 'd4edc97e4efb5612d16018f5cdeb42002d9eedfa'
ROSTER_BASE = '61e524e89c86950ef8e284ecaf18fddf82d11ca1'
MISSING = 'source/.github/workflows/english-hardneg-natural32-once.yml'
WORKFLOW = {'bytes': 4423, 'sha256': '2709d3cb7197624ca35e270974cd43bee2132d931d28dc109dda00d98b084d88'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(raw):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'prior duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('prior nonfinite JSON')))


def verify(raw, supplemental_workflow, expanded_files=None):
    need(identity(raw) == FAILURE, 'exact original failed ZIP identity')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        names = [i.filename for i in infos if not i.is_dir()]
        need(len(names) == len(set(names)) == 85, 'prior exact ZIP member count')
        payload = {}
        for i in infos:
            name = i.filename
            need(not PurePosixPath(name).is_absolute() and '\\' not in name and
                 all(p not in ('', '.', '..') for p in name.rstrip('/').split('/')) and
                 not stat.S_ISLNK(i.external_attr >> 16) and not i.flag_bits & 1, 'prior unsafe ZIP member')
            if not i.is_dir():
                payload[name] = z.read(i)
    manifest = load(payload['MANIFEST.json'])
    pins = manifest['files']
    need(len(pins) == 85 and set(pins) - set(payload) == {MISSING} and
         set(payload) - set(pins) == {'MANIFEST.json'}, 'only known prior workflow omission permitted')
    need(pins[MISSING] == WORKFLOW and identity(supplemental_workflow) == WORKFLOW, 'supplemental original workflow pin')
    for name, pin in pins.items():
        if name != MISSING:
            need(identity(payload[name]) == pin, 'prior manifest bytes: ' + name)
    if expanded_files is not None:
        need(set(expanded_files) == set(payload), 'prior expanded-copy exact member set')
        need(all(expanded_files[name] == value for name, value in payload.items()), 'prior expanded copy changed original bytes')
    result = load(payload['evidence/RESULT.json'])
    supervisor = load(payload['evidence/SUPERVISOR_EXIT.json'])
    start = load(payload['STARTED.json'])
    for obj in (result, supervisor, start['binding']):
        need(str(obj['run_id']) == RUN_ID and obj['head_sha'] == HEAD and obj['main_before'] == ROSTER_BASE and
             obj['run_attempt'] == 1 and obj['repository'] == 'jiying2007/kws-data' and obj['public_repository'] is True,
             'prior original run identity')
    need(manifest['supervised_success'] is False and manifest['authoritative_completion_receipt'] ==
         'evidence/SUPERVISOR_EXIT.json', 'prior failure authority')
    need(result['passed'] is False and result['status'] == 'FAILED_NO_RETRY' and result['reports'] == [] and
         result['consumer_build_attempts'] == result['decoder_process_attempts'] == result['model_process_attempts'] ==
         result['completed_validated_traces'] == result['retry_count'] == 0, 'prior compute was not zero')
    need(supervisor['passed'] is False and supervisor['returncode'] == 1 and supervisor['error'] is None and
         supervisor['timed_out'] is False and supervisor['wall_seconds'] == 307.016853929 and
         supervisor['cleanup'] == {'descendants_found':False,'terminated':False,'killed':False} and
         supervisor['retry_count'] == 0 and not supervisor['partial_receipt_errors'], 'prior supervisor terminal state')
    starts = {n: load(b) for n, b in payload.items() if n.endswith('.started.json')}
    need(not any(v.get('kind') in ('build','decode','model') for v in starts.values()), 'prior compute start exists')
    need(not any(n.startswith(('traces/','scores/','selectedinputs/','candidates/','bin/')) for n in payload), 'prior computed output exists')
    download = load(payload['evidence/corpus-download.exit.json'])
    download_start = load(payload['evidence/corpus-download.started.json'])
    need(download['returncode'] == 28 and download['timed_out'] is False and download['attempt'] == 1 and
         download['retry_count'] == 0 and download['argv'] == download_start['argv'], 'prior source acquisition failure')
    argv = download['argv']
    need(argv[argv.index('--max-time')+1] == '300' and argv[argv.index('--retry')+1] == '0', 'prior curl budget')
    need(b'curl: (28)' in payload['evidence/corpus-download.stderr'], 'prior curl timeout diagnosis')
    return {'status':'EXACT_PRIOR_ACQUISITION_FAILURE_AND_ZERO_COMPUTE_VERIFIED', 'original_archive':FAILURE,
            'run_id':RUN_ID,'head_sha':HEAD,'manifest_payload_entries':85,'available_payload_members_verified':84,
            'original_zip_closed_manifest':False,'fixed_missing_member':MISSING,'supplemental_original_workflow':WORKFLOW,
            'supplemental_bytes_repair_original_zip':False,'expanded_original_files_verified':expanded_files is not None,
            'source_acquisition_execution_attempt':1,'compute_attempts':0,'build_attempts':0,'decode_attempts':0,
            'completed_traces':0,'original_curl_returncode':28,'original_supervisor_seconds':307.016853929,
            'retry_count':0,'models_executed_by_this_audit':0}
