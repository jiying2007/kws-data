"""Independent closed second-failure and retained-original source proof.

No worker module, decoder, model, compiler or subprocess is used. The first
artifact's one documented omission is checked only inside that original ZIP.
"""
import hashlib
import struct
from pathlib import PurePosixPath

import prior_failure_proof

SECOND = {'bytes':16339221, 'sha256':'3b7a4ff6a59ba397eebac59a4f89feba8cc075a4f8881c9d531a3739bc1ec9a7'}
RUN_ID = '44060185229'
HEAD = 'cb6baa703f46451af73ec2cc0f85edfd1b159874'
PREVIOUS = 'd4edc97e4efb5612d16018f5cdeb42002d9eedfa'
ROSTER_BASE = '61e524e89c86950ef8e284ecaf18fddf82d11ca1'
RESEARCH = 'research/english_hardneg_natural32_20261117'
FALSE_FLAGS = ('human_gold','trusted_negative','training_admitted','formal_training_admitted',
               'pronunciation_verified','model_promotion_allowed','qualification_allowed')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def identity(raw):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


class BytesFile:
    def __init__(self, raw):
        self.raw = raw

    def read_bytes(self):
        return self.raw


def verify(raw, artifact_type, current_artifact=None):
    need(identity(raw) == SECOND, 'exact original second failure ZIP')
    second = artifact_type(BytesFile(raw))  # unchanged strict closed-manifest reader
    need(len(second.files) == 292 and len(second.manifest['files']) == 291 and
         second.manifest['supervised_success'] is False and second.manifest['authoritative_completion_receipt'] ==
         'evidence/SUPERVISOR_EXIT.json', 'second failure is not its closed original artifact')
    first = prior_failure_proof.verify(second.read('prior-failure/original-artifact.zip'),
        second.read('prior-failure/supplemental-source/.github/workflows/english-hardneg-natural32-once.yml'),
        {name[len('prior-failure/retained/'):]: value for name,value in second.files.items()
         if name.startswith('prior-failure/retained/')})
    prior = second.json('evidence/PRIOR_ACQUISITION_FAILURE_VERIFIED.json')
    need(second.read('evidence/PRIOR_ACQUISITION_FAILURE_VERIFIED.json') ==
         second.read('prior-failure/OMISSION_AND_ZERO_COMPUTE_PROOF.json') and prior['closed_manifest'] is False and
         prior['prior_compute_attempts'] == prior['prior_decode_attempts'] == prior['prior_build_attempts'] == prior['prior_trace_count'] == 0,
         'nested first-failure summary semantics')
    result, supervisor, started = (second.json(name) for name in
        ('evidence/RESULT.json','evidence/SUPERVISOR_EXIT.json','STARTED.json'))
    for obj in (result, supervisor, started['binding']):
        need(str(obj['run_id']) == RUN_ID and obj['head_sha'] == HEAD and obj['main_before'] == PREVIOUS and
             obj['run_attempt'] == 1 and obj['repository'] == 'jiying2007/kws-data' and obj['public_repository'] is True,
             'second failure run identity')
    need(result['passed'] is False and result['status'] == 'FAILED_NO_RETRY' and result['reports'] == [] and
         result['consumer_build_attempts'] == result['decoder_process_attempts'] == result['model_process_attempts'] ==
         result['completed_validated_traces'] == result['retry_count'] == result['prior_compute_attempts'] == 0 and
         result['acquisition_execution_attempt'] == 2 and result['compute_execution_attempt'] == 1, 'second compute was not zero')
    need(all(result[k] is False for k in FALSE_FLAGS) and result['verified_negative_seconds'] == 0, 'second failure admission boundary')
    need(supervisor['passed'] is False and supervisor['returncode'] == 1 and supervisor['error'] is None and
         supervisor['timed_out'] is False and 0 < supervisor['wall_seconds'] < 1200 and
         supervisor['cleanup'] == {'descendants_found':False,'terminated':False,'killed':False} and
         supervisor['retry_count'] == 0 and not supervisor['partial_receipt_errors'], 'second supervisor terminal state')
    command_starts = [second.json(name) for name in second.files
                      if name.startswith('evidence/') and name.endswith('.started.json')]
    need(not any(row.get('kind') in ('build','decode','model') for row in command_starts), 'second compute command exists')
    need(not any(name.startswith(('traces/','scores/','bin/','selectedinputs/pcm/','candidates/')) for name in second.files),
         'second computed output exists')
    exit = second.json('evidence/ffmpeg-version.exit.json')
    begin = second.json('evidence/ffmpeg-version.started.json')
    need(begin['argv'] == exit['argv'] == ['ffmpeg','-version'] and begin['kind'] is None and
         exit['returncode'] is None and exit['timed_out'] is False and exit['error'] ==
         "FileNotFoundError:[Errno 2] No such file or directory: 'ffmpeg'" and exit['retry_count'] == 0,
         'second missing-decoder failure diagnosis')
    corpus = second.json('evidence/CORPUS_DOWNLOAD.json')
    need(corpus['actual'] == {'bytes':337926286,'sha256':'76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3'} and
         corpus['retry_count'] == 0, 'saved full corpus acquisition receipt')
    download_exit = second.json('evidence/corpus-download.exit.json')
    need(download_exit['returncode'] == 0 and download_exit['error'] is None and download_exit['timed_out'] is False,
         'second corpus acquisition did not complete')
    roster_bytes = second.read('source/'+RESEARCH+'/natural32-roster.json')
    roster = second.json('source/'+RESEARCH+'/natural32-roster.json')
    need(hashlib.sha256(roster_bytes).hexdigest() == '31457faf8efd0cf77629c1a6f8de67bc101190eb118bcfe9ac2f256acca41239' and
         roster['data_repository_commit'] == ROSTER_BASE and len(roster['rows']) == 32 and
         sum(row['frames'] for row in roster['rows']) == 3270080, 'second frozen retained roster')
    ids = [row['recording_id'] for row in roster['rows']]
    second.exact_directory('selectedinputs/flac/', [rid+'.flac' for rid in ids])
    second.exact_directory('selectedinputs/transcripts/', [rid+'.txt' for rid in ids])
    transcript_sources = {PurePosixPath(r['source_member']).parts[-3]+'-'+PurePosixPath(r['source_member']).parts[-2]+'.trans.txt'
                          for r in roster['rows']}
    second.exact_directory('selectedinputs/transcript-sources/', transcript_sources)
    if current_artifact is not None:
        need(current_artifact.read('source/'+RESEARCH+'/natural32-roster.json') == roster_bytes, 'retained roster changed')
    for row in roster['rows']:
        rid = row['recording_id']
        flac_path, text_path = 'selectedinputs/flac/'+rid+'.flac', 'selectedinputs/transcripts/'+rid+'.txt'
        flac, text = second.read(flac_path), second.read(text_path)
        need(hashlib.sha256(flac).hexdigest() == row['flac_sha256'] and len(flac) >= 42 and flac[:4] == b'fLaC' and
             flac[4]&127 == 0 and int.from_bytes(flac[5:8],'big') == 34, 'retained original FLAC identity/header')
        header = int.from_bytes(flac[18:26], 'big')
        need((header>>44,((header>>41)&7)+1,((header>>36)&31)+1,header&((1<<36)-1)) ==
             (16000,1,16,row['frames']), 'retained original FLAC geometry')
        need(hashlib.sha256(text).hexdigest() == row['transcript_sha256'], 'retained selected transcript identity')
        if current_artifact is not None:
            need(current_artifact.read(flac_path) == flac and current_artifact.read(text_path) == text,
                 'retained FLAC/transcript bytes not reused exactly')
    for name in transcript_sources:
        if current_artifact is not None:
            need(current_artifact.read('selectedinputs/transcript-sources/'+name) ==
                 second.read('selectedinputs/transcript-sources/'+name), 'retained full transcript source changed')
    if current_artifact is not None:
        proof_members = ['MANIFEST.json','STARTED.json','evidence/RESULT.json','evidence/SUPERVISOR_EXIT.json',
                         'evidence/ffmpeg-version.started.json','evidence/ffmpeg-version.exit.json',
                         'evidence/CORPUS_DOWNLOAD.json','evidence/corpus-download.exit.json',
                         'evidence/SELECTED_ORIGINALS.json','evidence/PRIOR_ACQUISITION_FAILURE_VERIFIED.json']
        current_artifact.exact_directory('retained-source/proof/', proof_members)
        need(all(current_artifact.read('retained-source/proof/'+name) == second.read(name) for name in proof_members),
             'retained source proof fragments differ from original second ZIP')
    return {'status':'EXACT_CLOSED_SECOND_FAILURE_RETAINED_ORIGINALS_AND_DUAL_ZERO_COMPUTE_VERIFIED',
            'second_original_archive':SECOND,'second_run_id':RUN_ID,'second_head_sha':HEAD,
            'second_artifact_closed_manifest':True,'second_manifest_payload_files_verified':291,
            'second_supervisor_seconds':supervisor['wall_seconds'],'second_failure':'FFMPEG_VERSION_EXECUTABLE_ABSENT_BEFORE_DECODE',
            'second_compute_attempts':0,'second_build_attempts':0,'second_decode_attempts':0,'second_completed_traces':0,
            'retained_flac_originals_verified':32,'retained_original_samples':3270080,
            'current_retained_bytes_equal_second_artifact':current_artifact is not None,
            'first_failure_proof':first,'prior_model_total':0,'prior_source_archive_acquisitions':2,
            'full_corpus_tar_rehashed_by_this_audit':False,'models_executed_by_this_audit':0,'decodes_executed_by_this_audit':0}
