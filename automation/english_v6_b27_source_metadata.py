"""One B27 metadata-only assessment before any new optimizer or audio.

Preserve every original Q1 exposure and all sealed people. Reuse the unchanged
metadata ranking and exact approved Q1 native-empty policy. This module never
authenticates a quality verdict or opens a scientific execution budget.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANELS = HERE / 'english_v6_b27_panels'
PREFIX = 'automation/english_v6_b27_panels/'
MANIFEST = PREFIX + 'b24q1-source-exposure.json'
ORIGIN = PREFIX + 'b24q1-source-origin.json'
PREVIOUS = 'automation/english_v6_b24q1_source_metadata.py'
PREVIOUS_SHA = 'f1c850415a442912b0900b73795517e2b61570b2001587aa45426e4f9d575a0c'
PREVIOUS_ASSESSMENT_SHA = '681ddb42432bd11dfef0157d2f1c95f7e02824c3a2ed7da64ce9a12bdf9c3139'
ORIGINAL_ASSESSMENT = dict(path='automation/english_v6_b24q1_panels/assessment.json', bytes=87913,
    sha256=PREVIOUS_ASSESSMENT_SHA, git_blob_sha1='be013eddbeb825bd9cdf5f08c44e9604205b59ea')
DESIGN_PATH = 'automation/english_v6_b27_natural_weight_design.json'
DESIGN_SHA = '956f0695e6b4e3ac450bc3881c5ef13985d775210cf43cf4da98947a5a1846bf'
PROPOSAL_PATH = 'research/english_v6_b27_preparation_20261125/PROPOSAL.json'
PROPOSAL_SHA = '587b9a404fa9209d10f1364826cf0c37879669f3a56545e1e6576473670a10c3'
ORCHESTRATION_PATH = 'automation/english_v6_b24q1_orchestration_contract.json'
ORCHESTRATION_SHA = '75136b683c893e3d6457d93b2f9f74829c711e7b614b6a5b727380faf4f42f49'
SOURCE_PIN = dict(id=14996004110, run_id=45554635013,
    run_head='3f04370dde7c6c0d9521c7d9227d3dec37417929', bytes=367799776,
    sha256='a9f24a791583066ecb6fb78d2ac3f2f36bc172301145085de95b6ec14bce96df')
SOURCE_PRODUCER_PIN = dict(id=14995822013, run_id=45554635013,
    run_head=SOURCE_PIN['run_head'], bytes=18369647,
    sha256='8b518514ca29b491cde448a3b6c8b3c3409e97466cf76a2783999c3dfeaa733b')
SOURCE_PARENT = '5bdf5db0f632f92496df59de0a6fe70ce254cafd'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b24q1-primary-sources-once.yml'
SOURCE_BRANCH = 'research/en-v6-b24q1-primary-sources-once-20261125'
SOURCE_MEMBERS = {'MANIFEST.json':'d73ddafaa7af67fa4dfe55290d3e17baf7c6b7eaddafb60cd5c7c56ab58840ff',
                  'HISTORY.json':'5add13ba0bbd24cf5931998ec1866920cd02d7d0ec5b5743d10a267efa4dd6ec'}
TRAINING_ASSESSMENT_SHA = '3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad'
SALTS = {role:{kind:f'english-v6-b27-natural-weight-{role}-{kind}-20261125-v1'
    for kind in ('piper','natural')} for role in ('primary','replication')}
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
FROZEN_MODELS = {'B0':'1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7',
    'B9':'7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655',
    'control':CONTROL_PT,'challenger':None}
OUTPUT_NAMES = ('ancestry.json','assessment.json','primary.json','natural-primary.json',
                'replication.json','natural-replication.json')
# Filled from the exact original metadata reexport, before the single selection.
EXPOSURE_SHA = 'de581979b0b698a75145cae4ee8b76d6741aaf6d7fbeeb0f4c3c3a21669af9b3'
ORIGIN_SHA = 'a12976c2215543a15b5499013aaf961306c5a78bf1652620b27959be2fe45efc'


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _previous():
    need(digest((ROOT / PREVIOUS).read_bytes()) == PREVIOUS_SHA, 'unchanged complete Q1 metadata validator')
    spec = importlib.util.spec_from_file_location('b27_original_q1_metadata', ROOT / PREVIOUS)
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    return previous


def _source_api_identity(api, pin):
    need(pin == SOURCE_PIN, 'same exact original Q1 source pin')
    run, artifact, commit = (api[key] for key in ('run','artifact','commit'))
    need(run['id'] == pin['run_id'] and run['head_sha'] == pin['run_head'] == commit['sha']
         and run['run_attempt'] == 1 and run['event'] == 'push'
         and run['status'] == 'completed' and run['conclusion'] == 'success'
         and run['head_branch'] == SOURCE_BRANCH and run['path'] == SOURCE_WORKFLOW
         and run['repository']['full_name'] == 'jiying2007/kws-data' and run['repository']['private'] is False
         and len(commit['parents']) == 1 and commit['parents'][0]['sha'] == SOURCE_PARENT
         and [row['filename'] for row in commit['files']] == [SOURCE_WORKFLOW]
         and artifact['id'] == pin['id'] and artifact['size_in_bytes'] == pin['bytes']
         and artifact['digest'] == 'sha256:' + pin['sha256'] and artifact['expired'] is False
         and artifact['name'] == 'english-v6-b24q1-primary-source-' + str(pin['run_id'])
         and artifact['workflow_run']['id'] == pin['run_id'] and artifact['workflow_run']['head_sha'] == pin['run_head'],
         'original first public successful Q1 SOURCE and workflow-only identity')
    return dict(run={**{k:run[k] for k in ('id','run_attempt','head_sha','head_branch','event','path','status','conclusion')},
                     'repository':{k:run['repository'][k] for k in ('full_name','private')}},
        artifact={**{k:artifact[k] for k in ('id','name','size_in_bytes','digest','expired')},
                  'workflow_run':{k:artifact['workflow_run'][k] for k in ('id','head_sha')}},
        commit=dict(sha=commit['sha'],parents=[dict(sha=commit['parents'][0]['sha'])],files=[dict(filename=SOURCE_WORKFLOW)]))


def _canonical_person(row):
    value = next((row[k] for k in ('source_speaker_id','speaker','speaker_id') if row.get(k) is not None),
                 row['source_group_id'].rsplit('-',1)[-1])
    text = str(value).removeprefix('p')
    need(re.fullmatch(r'[0-9]+',text) is not None, 'canonical original source person')
    return str(int(text))


def original_assessment_people():
    raw = (ROOT / ORIGINAL_ASSESSMENT['path']).read_bytes()
    need(len(raw) == ORIGINAL_ASSESSMENT['bytes'] and digest(raw) == PREVIOUS_ASSESSMENT_SHA
         and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == ORIGINAL_ASSESSMENT['git_blob_sha1'],
         'exact original Q1 assessment bytes and Git blob')
    assessment, people = json.loads(raw), {}
    for role in ('primary','replication'):
        synthetic = {str(int(str(p).removeprefix('p'))) for p in assessment[role]['synthetic']['speaker_id_map'].values()}
        natural = set(assessment[role]['natural']['speakers'])
        need(len(synthetic)==24 and len(natural)==12 and not synthetic & natural, 'original Q1 36 people per panel')
        people[role] = synthetic | natural
    need(not people['primary'] & people['replication'], 'all72 original Q1 people remain excluded')
    return {k:sorted(v,key=int) for k,v in people.items()}


def _origin_document(exposure, api):
    identity = _source_api_identity(api,SOURCE_PIN); raw = encoded(exposure)
    return dict(schema='english-v6-b27-authenticated-q1-source-metadata-origin-v1',
        source_artifact=dict(SOURCE_PIN),source_producer_artifact=dict(SOURCE_PRODUCER_PIN),
        original_assessment=dict(ORIGINAL_ASSESSMENT),source_api_identity=identity,
        source_api_identity_sha256=digest(encoded(identity)),original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_exposure=dict(path=MANIFEST,bytes=len(raw),sha256=digest(raw)),
        extracted_member_allowlist=['MANIFEST.json','HISTORY.json'],metadata_only=True,
        audio_members_interpreted=0,observer_receipts_read=0,quality_results_read=False,
        model_calls=0,new_source_calls=0,original_SOURCE_run_success=True,
        full_original_Q1_NO_PROMOTION_proof_is_separate=True,scientific_execution_authorized=False)


def derive_original_metadata(members, api, source_pin):
    need(set(members)==set(SOURCE_MEMBERS) and all(type(members[n]) is bytes and digest(members[n])==h
         for n,h in SOURCE_MEMBERS.items()), 'exact original Q1 MANIFEST and HISTORY bytes')
    identity = _source_api_identity(api,source_pin)
    manifest,history = (json.loads(members[n]) for n in ('MANIFEST.json','HISTORY.json'))
    execution = dict(prepared_parent=SOURCE_PARENT,run_attempt=1,run_branch='refs/heads/'+SOURCE_BRANCH,
                     run_head=source_pin['run_head'],run_id=str(source_pin['run_id']))
    need(manifest['schema']=='english-v6-b24q1-fresh-combined-source-v1' and manifest['panel']=='primary'
         and manifest['selected_original_slots']==len(manifest['rows'])==120 and manifest['source_execution']==execution
         and manifest['assessment_sha256']==PREVIOUS_ASSESSMENT_SHA
         and manifest['original_training_assessment_sha256']==TRAINING_ASSESSMENT_SHA
         and manifest['source_ancestry_sha256']==manifest['history_receipt_sha256']==SOURCE_MEMBERS['HISTORY.json']
         and manifest['new_kws_forwards']==0 and manifest['replication_opened'] is False
         and history['all_prior_replication_sealed'] is True, 'complete pre-scoring original Q1 source and inherited history')
    fields = ['id','cohort','source_group_id','pcm_sha256','canonical_person']
    rows = [{**{k:r[k] for k in fields[:-1]},'canonical_person':_canonical_person(r)} for r in manifest['rows']]
    need(len({r['id'] for r in rows})==len({r['pcm_sha256'] for r in rows})==120
         and len({r['canonical_person'] for r in rows})==len({r['source_group_id'] for r in rows})==36
         and sum(r['cohort']=='synthetic' for r in rows)==96 and sum(r['cohort']=='natural' for r in rows)==24
         and all(re.fullmatch(r'[0-9a-f]{64}',r['pcm_sha256']) for r in rows), 'complete unfiltered original120 exposure')
    people = original_assessment_people()
    need(set(people['primary'])=={r['canonical_person'] for r in rows}, 'all original36 people match Q1 assessment')
    exposure = dict(schema='english-v6-b27-authenticated-q1-exposure-v1',original_source_schema=manifest['schema'],
        panel='primary',selected_original_slots=120,source_artifact=dict(source_pin),source_execution=execution,
        assessment_sha256=PREVIOUS_ASSESSMENT_SHA,source_ancestry_sha256=SOURCE_MEMBERS['HISTORY.json'],
        history_receipt_sha256=SOURCE_MEMBERS['HISTORY.json'],original_metadata_sha256=dict(SOURCE_MEMBERS),
        new_kws_forwards=0,replication_opened=False,rows=rows,metadata_only=True,selection_fields=fields,
        primary_people=people['primary'],sealed_replication_people=people['replication'],
        all_excluded_people=sorted(set(people['primary'])|set(people['replication']),key=int))
    return {'b24q1-source-exposure.json':exposure,'b24q1-source-origin.json':_origin_document(exposure,identity)}


def _archive(path,pin,maximum):
    path = Path(path).absolute()
    need(path.is_file() and all(not p.is_symlink() for p in (path,*path.parents))
         and path.stat().st_size==pin['bytes']<=maximum, 'regular exact original archive byte count')
    with path.open('rb') as stream:
        need(hashlib.file_digest(stream,'sha256').hexdigest()==pin['sha256'], 'whole original archive SHA256')
    archive = zipfile.ZipFile(path)
    try:
        infos = archive.infolist()
        need(len(infos)<=8192 and len(infos)==len({i.filename for i in infos})
             and sum(i.file_size for i in infos)<=maximum, 'bounded unique original archive members')
        for i in infos:
            p = PurePosixPath(i.filename)
            need(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
                 and not stat.S_ISLNK(i.external_attr>>16), 'safe original archive member')
        return archive
    except BaseException:
        archive.close(); raise


def read_original_metadata(source_zip,source_api,source_pin,source_producer_zip=None):
    _source_api_identity(source_api,source_pin)
    with _archive(source_zip,source_pin,768_000_000) as archive:
        members = {n:archive.read(n) for n in SOURCE_MEMBERS}
    if source_producer_zip is not None:
        with _archive(source_producer_zip,SOURCE_PRODUCER_PIN,128_000_000) as archive:
            raw = archive.read('repository/'+ORIGINAL_ASSESSMENT['path'])
            need(raw==(ROOT/ORIGINAL_ASSESSMENT['path']).read_bytes() and digest(raw)==PREVIOUS_ASSESSMENT_SHA,
                 'actual captured original Q1 assessment')
    return derive_original_metadata(members,source_api,source_pin)


def exclusion_summary(exposure,origin):
    return dict(schema='english-v6-b27-original-q1-source-exclusion-v1',
        source_artifact=dict(SOURCE_PIN),source_producer_artifact=dict(SOURCE_PRODUCER_PIN),
        original_assessment=dict(ORIGINAL_ASSESSMENT),original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_files={MANIFEST:digest(encoded(exposure)),ORIGIN:digest(encoded(origin))},
        original_rows=120,primary_people=36,sealed_replication_people=36,all72_people_excluded=True,
        prior_PCM_identities=2943,all_PCM_identities_after_Q1=3063,
        source_identity_authenticated=True,metadata_only=True,audio_members_interpreted=0,
        observer_receipts_read=0,quality_results_read=False,
        complete_original_Q1_NO_PROMOTION_readback_required_separately=True,
        captured_original_Q1_assessment_in_complete_producer_required=True,
        scientific_execution_authorized=False,model_calls=0,new_source_calls=0)


def verify_original_source(source_zip,source_api,source_pin,source_producer_zip=None):
    documents = read_original_metadata(source_zip,source_api,source_pin,source_producer_zip)
    for name,value in documents.items():
        need((PANELS/name).read_bytes()==encoded(value), 'original Q1 metadata reexport equals frozen whitelist: '+name)
    result = exclusion_summary(documents['b24q1-source-exposure.json'],documents['b24q1-source-origin.json'])
    need(result==validate()['original_B24Q1_source_exclusion'], 'same complete frozen original Q1 source exclusion')
    return result


def inputs():
    previous = _previous()
    hashes = dict(previous.validate()['hashes'])
    base = previous.inputs()[0]
    old_raw=(ROOT/ORIGINAL_ASSESSMENT['path']).read_bytes();history_raw=(previous.PANELS/'ancestry.json').read_bytes()
    need(digest(old_raw)==PREVIOUS_ASSESSMENT_SHA and digest(history_raw)==SOURCE_MEMBERS['HISTORY.json'],
         'whole original Q1 assessment and history')
    old,history=json.loads(old_raw),json.loads(history_raw)
    expected={MANIFEST:EXPOSURE_SHA,ORIGIN:ORIGIN_SHA,DESIGN_PATH:DESIGN_SHA,PROPOSAL_PATH:PROPOSAL_SHA}
    values={}
    for name,pin in expected.items():
        need(type(pin) is str and re.fullmatch(r'[0-9a-f]{64}',pin), 'original metadata inputs must be fixed before selection')
        raw=(ROOT/name).read_bytes();need(digest(raw)==pin,'fixed B27 metadata input '+name)
        hashes[name]=pin;values[name]=json.loads(raw)
    design,proposal=values[DESIGN_PATH],values[PROPOSAL_PATH]
    need(design['schema']=='english-v6-b27-natural-weight-single-factor-design-v1'
         and design['scientific_change']=={'field':'natural_weight','global':'NATURAL_WEIGHT','before':.08,'after':.16}
         and design['initial_pt_sha256']==INITIAL_PT and design['historical_control_pt_sha256']==CONTROL_PT
         and design['optimizer_updates']==192 and design['maximum_new_model_forwards']==1847
         and design['opening_condition']=='only_after_actual_complete_B24Q1_primary_NO_PROMOTION'
         and design['scientific_execution_authorized'] is False
         and design['bounded_proposal']==dict(path=PROPOSAL_PATH,sha256=PROPOSAL_SHA), 'exact prior-hypothesis B27 design')
    for value in design.values():
        if isinstance(value,dict) and set(value)=={'path','sha256'}:
            raw=(ROOT/value['path']).read_bytes();need(digest(raw)==value['sha256'],'entire design dependency '+value['path'])
            hashes[value['path']]=value['sha256']
    need(proposal['experiment_id']=='B27' and proposal['hypothesis']['scientific_change']==design['scientific_change']
         and proposal['entry_conditions']['condition']==design['opening_condition']
         and proposal['current_state']['scientific_execution_authorized'] is False
         and proposal['complete_route_caps']['total_new_KWS_model_and_backend_calls_maximum']==4867,
         'one bounded candidate and conditionally complete software route')
    previous.orchestration_contract()
    need(digest((ROOT/ORCHESTRATION_PATH).read_bytes())==ORCHESTRATION_SHA,'exact unchanged approved Q1 policy')
    exposure,origin=values[MANIFEST],values[ORIGIN]
    need(origin==_origin_document(exposure,origin['source_api_identity']), 'exact minimal original Q1 origin')
    fields=['id','cohort','source_group_id','pcm_sha256','canonical_person']
    need(set(exposure)=={'schema','original_source_schema','panel','selected_original_slots','source_artifact',
         'source_execution','assessment_sha256','source_ancestry_sha256','history_receipt_sha256','original_metadata_sha256',
         'new_kws_forwards','replication_opened','rows','metadata_only','selection_fields','primary_people',
         'sealed_replication_people','all_excluded_people'}
         and exposure['schema']=='english-v6-b27-authenticated-q1-exposure-v1'
         and exposure['original_source_schema']=='english-v6-b24q1-fresh-combined-source-v1'
         and exposure['panel']=='primary' and exposure['selected_original_slots']==len(exposure['rows'])==120
         and exposure['source_artifact']==SOURCE_PIN and exposure['original_metadata_sha256']==SOURCE_MEMBERS
         and exposure['source_ancestry_sha256']==exposure['history_receipt_sha256']==old['ancestry_sha256']==digest(history_raw)
         and exposure['assessment_sha256']==PREVIOUS_ASSESSMENT_SHA and exposure['new_kws_forwards']==0
         and exposure['replication_opened'] is False and exposure['metadata_only'] is True and exposure['selection_fields']==fields
         and all(set(r)==set(fields) and r['canonical_person']==base.source_person(r) for r in exposure['rows']),
         'all120 exposure identities only, no labels or numerical quality data')
    need(exposure['source_execution']==dict(prepared_parent=SOURCE_PARENT,run_attempt=1,run_branch='refs/heads/'+SOURCE_BRANCH,
         run_head=SOURCE_PIN['run_head'],run_id=str(SOURCE_PIN['run_id'])), 'exact original Q1 source execution')
    people=original_assessment_people()
    need(exposure['primary_people']==people['primary'] and exposure['sealed_replication_people']==people['replication']
         and exposure['all_excluded_people']==base.ordered(set(people['primary'])|set(people['replication'])),
         'all original36 primary plus36 unopened Q1 people retained')
    return base,old,history,exposure,origin,hashes


def build_documents():
    base,old,history,exposure,origin,hashes=inputs();rows=exposure['rows']
    exposed={base.source_person(r) for r in rows};reserved=set(exposure['sealed_replication_people'])
    need(exposed==set(exposure['primary_people']) and len(exposed)==len(reserved)==36 and not exposed & reserved
         and len({r['id'] for r in rows})==len({r['pcm_sha256'] for r in rows})==120
         and len({r['source_group_id'] for r in rows})==36
         and sum(r['cohort']=='synthetic' for r in rows)==96 and sum(r['cohort']=='natural' for r in rows)==24,
         'complete original Q1 population before any ranking')
    excluded=exposed|reserved
    need(excluded==set(exposure['all_excluded_people']), 'no sealed Q1 reserve reclaimed')
    synth_blocked=set(history['synthetic_excluded_people'])|excluded
    natural_blocked=set(history['natural_excluded_people'])|excluded
    need(not set(history['all_pcm_sha256']) & {r['pcm_sha256'] for r in rows}, 'all120 Q1 PCM identities actually fresh')
    ancestry=copy.deepcopy(history)
    ancestry.update(schema='english-v6-b27-complete-exposure-and-sealed-reserves-v1',
        inherited_history_sha256=old['ancestry_sha256'],inherited_assessment_sha256=PREVIOUS_ASSESSMENT_SHA,
        original_B24Q1_source=dict(SOURCE_PIN),B24Q1_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'],B24Q1_exposure_sha256=EXPOSURE_SHA,
        all_pcm_sha256=sorted(set(history['all_pcm_sha256'])|{r['pcm_sha256'] for r in rows}),
        all_source_groups=sorted(set(history['all_source_groups'])|{r['source_group_id'] for r in rows}),
        canonical_people=base.ordered(set(history['canonical_people'])|exposed),
        synthetic_excluded_people=base.ordered(synth_blocked),natural_excluded_people=base.ordered(natural_blocked),
        B24Q1_exposed_primary_people=base.ordered(exposed),B24Q1_sealed_replication_people=base.ordered(reserved),
        all_B24Q1_120_raw_originals_added=True,all_prior_replication_sealed=True,
        B24Q1_quality_metrics_read_for_selection=False,old_B24Q1_replication_opened=False,
        automatic_next_assessment_allowed=False,B27_execution_authorized=False,metadata_input_sha256=dict(hashes))
    need(len(history['all_pcm_sha256'])==2943 and len(ancestry['all_pcm_sha256'])==3063,
         'complete2943 prior plus all120 Q1 PCM identities')
    catalog=base.read_inputs();mapping=catalog['piper-config.json']['speaker_id_map']
    by_index={i:p for p,i in mapping.items()}
    available_synth={base.canonical(p):i for i,p in by_index.items() if base.canonical(p) not in synth_blocked}
    need({base.canonical(p) for p in mapping}<=natural_blocked,'whole potentially overlapping Piper population excluded from natural')
    census,eligible=base.catalog_rows(catalog['FULL_METADATA_CATALOG.json'])
    available_natural=census-natural_blocked;usable_natural={p for p in available_natural if len(eligible[p])>=2}
    need(len(available_synth)>=48 and len(usable_natural)>=24,'two full independent panels available without redraw')
    panels,selected={},set()
    for role in ('primary','replication'):
        speakers=sorted(set(available_synth)-selected,key=lambda p:base.rank(SALTS[role]['piper'],'speaker',p))[:24]
        need(len(speakers)==24,'exact24 new synthetic people');selected.update(speakers)
        indices=[available_synth[p] for p in speakers]
        readers=sorted(usable_natural-selected,key=lambda p:base.rank(SALTS[role]['natural'],'speaker',p))[:12]
        need(len(readers)==12,'exact12 new natural people');selected.update(readers)
        roster=[r for p in readers for r in sorted(eligible[p],key=lambda r:base.rank(SALTS[role]['natural'],'utterance',r['source_id']))[:2]]
        natural=copy.deepcopy(old[role]['natural'])
        natural.update(opened=False,salt=SALTS[role]['natural'],speakers=readers,roster=roster,
            excluded_canonical_speakers=base.ordered(natural_blocked|(selected-set(readers))),
            available_census=base.ordered(available_natural),selected_utterance_ids=[r['source_id'] for r in roster])
        panels[role]=dict(synthetic=dict(salt=SALTS[role]['piper'],speaker_indices=indices,
            speaker_id_map={str(i):by_index[i] for i in indices},original_slots=96,
            seed_base=20263900 if role=='primary' else 20264000),natural=natural)
    need(len(selected)==72 and not selected & excluded,'all72 new people independent of all original Q1 people')
    assessment=copy.deepcopy(old)
    for key in ('evaluation_design_path','evaluation_design_sha256','approval','approved_proposal',
                'training_assessment_sha256','primary_scoring_requires_frozen_actual_B24_weights_and_new_Q1_source_gate_pass',
                'original_B24R1_SOURCE_STOP_proof_required_before_any_new_budget'):
        assessment.pop(key,None)
    assessment.update(schema='english-v6-b27-preoptimizer-natural-weight-assessment-v1',
        purpose='One TRAIN-only natural-weight contrast, preoptimizer fresh primary and sealed replica, then conditional complete software landing; Q1 numerical results do not choose parameters or people.',
        bounded_proposal=dict(path=PROPOSAL_PATH,sha256=PROPOSAL_SHA),training_design_sha256=DESIGN_SHA,
        ancestry_sha256=digest(encoded(ancestry)),metadata_validator_sha256=digest(Path(__file__).read_bytes()),
        metadata_inputs_sha256=dict(hashes),primary=panels['primary'],replication=panels['replication'],
        evaluation_only=False,freeze_before_optimizer_and_new_source=True,freeze_after_training_before_new_source=False,
        frozen_model_pt_sha256=dict(FROZEN_MODELS),
        model_roles=dict(B0='unchanged_B0',B9='unchanged_B9',control='immutable_original_B24_challenger',challenger='only_final_B27_192_update_checkpoint'),
        training=dict(design_path=DESIGN_PATH,design_sha256=DESIGN_SHA,parameters=43876,initial_pt_sha256=INITIAL_PT,
            historical_control_pt_sha256=CONTROL_PT,new_challenger_updates=192,new_model_forwards=1847,
            new_training_audio=False,new_teacher_forwards=0,new_historical_control_forwards=0,
            existing_train_originals=88,positive_retention_conditions=122,saved_ordinary_teacher_arrays=264,
            restore_optimizer_state=False,optimizer='fresh_AdamW',optimizer_allowed=False,threshold_search=False,
            original_B24_training_assessment_sha256=TRAINING_ASSESSMENT_SHA,original_training_assessment_not_rebound=True),
        original_B24Q1_source_exclusion=exclusion_summary(exposure,origin),
        opening_condition='only_after_actual_complete_B24Q1_primary_NO_PROMOTION',
        original_B24Q1_complete_NO_PROMOTION_required_before_any_new_budget=True,
        original_Q1_result_is_eligibility_only_not_hypothesis_support=True,B24Q1_quality_metrics_read_for_selection=False,
        old_B24Q1_replication_opened=False,old_B26_opening_condition_unchanged=True,
        B26_opener='only_after_actual_complete_B24R1_primary_NO_PROMOTION',B27_source_failure_terminal=True,
        primary_scoring_requires_exact_final_B27_weights_and_new_source_gate_pass=True,
        quality_control_role='Exact original B24 challenger is the immutable historical control; original B14 criteria unchanged, no same-job rerun claim.',
        promotion_prerequisites='Both complete independent B27 quality PASSes, unchanged140-call engineering and full independent readback, then exact zero-forward software installation and candidate/B0 loader receipts; physical/field remain separate NOT_RUN.',
        replication_opening='Only complete authenticated B27 PRIMARY numerical PASS opens this frozen untouched replica; no old reserve or automatic next panel.',
        maximum_training_model_forwards=1847,maximum_optimizer_updates=192,maximum_engineering_backend_calls=140,
        maximum_total_KWS_model_and_backend_calls=4867,software_landing_in_same_bounded_scope=True,
        metadata_selection=dict(algorithm=old['metadata_selection']['algorithm'],salts=SALTS,corpus_census=251,
            eligible_synthetic_voices=len(available_synth),available_natural_census=len(available_natural),
            eligible_natural_readers=len(usable_natural),synthetic_excluded_canonical_people=len(synth_blocked),
            natural_excluded_canonical_people=len(natural_blocked),metadata_only=True,selection_before_audio=True,
            selection_before_optimizer=True,selection_after_training=False,selection_before_first_independent_KWS=True,
            source_replacements=0,reserves_reclaimed=0,cross_source_type_identity_exclusion=True))
    documents={'ancestry.json':ancestry,'assessment.json':assessment}
    for role in ('primary','replication'):
        synthetic,planned=panels[role]['synthetic'],[]
        for ordinal,index in enumerate(synthetic['speaker_indices']):
            speaker=by_index[index]
            for cell,scale,cls,text,native in (
                ('mero-fast',1.4,2,'Hey Mero.','[[heɪ mˈɛɹoʊ]]'),('mero-slow',1.8,2,'Hey Mero.','[[heɪ mˈɛɹoʊ]]'),
                ('question',1.,0,'What time is it?','[[wˈʌt tˈaɪm ɪz ɪt]]'),('request',1.08,0,'Please set a timer.','Please set a timer.')):
                planned.append(dict(id=f'b27-{role}-{index}-{cell}',speaker_index=index,source_speaker_id=speaker,
                    source_group_id='piper-libritts-'+speaker,role='development',panel=role,cell_id=cell,cohort='synthetic',
                    length_scale=scale,seed=synthetic['seed_base']+ordinal,requested_class=cls,requested_text=text,
                    native_input=native,observed_class=None,training_allowed=False))
        documents[role+'.json']=dict(schema='english-v6-b27-fresh-combined-source-v1',panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)),source_ancestry_sha256=digest(encoded(ancestry)),
            model=base.PIPER,config_sha256=base.INPUTS['piper-config.json'],max_source_calls=96,
            maximum_native_observer_calls=480,original_slots=96,expected_providers=base.PROVIDERS,
            source_format=old['source_format'],source_admission_rule='unchanged B14 five-valid admission and exact approved Q1 whole-original successful-empty quarantine',
            orchestration_contract_sha256=ORCHESTRATION_SHA,existing_replication_opened=False,training_allowed=False,
            promotion_allowed=False,rows=planned)
        documents['natural-'+role+'.json']=panels[role]['natural']
    need(set(documents)==set(OUTPUT_NAMES),'all and only six prospective frozen documents')
    return documents,hashes


def validate():
    documents,hashes=build_documents()
    for name,value in documents.items():
        raw=encoded(value);need((PANELS/name).read_bytes()==raw,'complete deterministic frozen B27 metadata: '+name)
        hashes[(PANELS/name).relative_to(ROOT).as_posix()]=digest(raw)
    hashes[Path(__file__).relative_to(ROOT).as_posix()]=digest(Path(__file__).read_bytes())
    return dict(schema='english-v6-b27-offline-metadata-validation-v1',status='PASS_METADATA_ONLY',hashes=hashes,
        selection=documents['assessment.json']['metadata_selection'],proposal_sha256=PROPOSAL_SHA,training_design_sha256=DESIGN_SHA,
        original_B24Q1_source_exclusion=documents['assessment.json']['original_B24Q1_source_exclusion'],
        all_B24Q1_120_originals_excluded=True,all_B24Q1_72_people_excluded=True,historical_PCM_identities_excluded=3063,
        quality_metrics_read_for_selection=False,freeze_before_optimizer_and_new_source=True,
        orchestration_contract_sha256=ORCHESTRATION_SHA,old_B24_and_R1_stops_preserved=True,old_B25_and_B26_openers_unchanged=True,
        automatic_next_assessment_allowed=False,model_calls=0,source_calls=0,observer_calls=0,optimizer_updates=0,
        execution_authorized=False,source_execution_authorized=False,primary_scoring_authorized=False,
        replica_opening_authorized=False,engineering_authorized=False,promotion_allowed=False)


def freeze():
    need(not any((PANELS/n).exists() for n in OUTPUT_NAMES),'freeze only once; never redraw or overwrite selected panels')
    documents,_=build_documents();PANELS.mkdir(exist_ok=True)
    for name,value in documents.items():
        with (PANELS/name).open('xb') as stream:stream.write(encoded(value))
    return validate()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='one reviewed metadata selection; zero scientific budget')
    args=parser.parse_args()
    print(json.dumps(freeze() if args.freeze else validate(),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
