"""Sealed B24 replica authorization using only original complete PRIMARY PASS."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24_source_ancestry as original
import english_v6_b24_quality_readback as reader
import english_v6_b24_replication_capacity as capacity
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())

BRANCH = 'refs/heads/research/en-v6-b24-replication-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b24-replication-sources-once.yml'
SCHEMA = 'english-v6-b24-replication-source-authorization-v1'
MARKER = 'english-v6-b24-replication-source-consumed'
OPENING = HERE / 'english_v6_b24_replication_opening.json'
READER = HERE / 'english_v6_b24_quality_readback.py'
DESIGN_SHA = original.metadata.DESIGN_SHA
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
_base_consumption_document = consumption_document


def read_opening(require_bound=True):
    value = json.loads(OPENING.read_bytes())
    need(value['schema'] == 'english-v6-b24-replication-original-primary-opening-v1'
         and value['panel'] == 'replication' and value['assessment_sha256'] == ASSESSMENT_SHA
         and value['training_design_sha256'] == DESIGN_SHA
         and value['original_primary_pass_required'] is True
         and value['training_allowed'] is value['promotion_allowed'] is value['same_job_matched_control'] is False,
         'exact frozen B24 replica and historical-control semantics')
    if value['status'] == 'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS':
        need(all(value[k] is None for k in ('primary_quality_lock','quality_readback_sha256','pair_pin','model_pt_sha256')),
             'all replica opening pins remain unset until actual PRIMARY PASS')
        need(not require_bound, 'STOP: replica remains sealed without original PRIMARY PASS')
        return value
    need(value['status'] == 'BOUND_ORIGINAL_PRIMARY_PASS'
         and digest(READER.read_bytes()) == value['quality_readback_sha256'], 'exact immutable original-quality reader')
    reader.validate_lock(value['primary_quality_lock'])
    need(value['primary_quality_lock']['panel'] == 'primary', 'original primary lock only')
    reader.valid_artifact(value['pair_pin'])
    model = value['model_pt_sha256']
    t = training_binding()['training']
    need(type(model) is dict and model == t['model_pt_sha256']
         and model['challenger'] == CANDIDATE_PT and model['control'] == original.metadata.CONTROL_PT
         and value['pair_pin'] == dict(id=t['artifacts']['models']['id'], run_id=t['run_id'], run_head=t['run_head'],
             bytes=t['artifacts']['models']['bytes'], sha256=t['artifacts']['models']['sha256']),
         'same successful original B24 challenger and immutable historical B23 control')
    return value


def opening_sha256():
    read_opening()
    return digest(OPENING.read_bytes())


def quality_reader():
    opening = read_opening()
    need(Path(reader.__file__).resolve() == READER and digest(READER.read_bytes()) == opening['quality_readback_sha256'],
         'exact reviewed original PRIMARY quality reader')
    return reader


def validate_primary(summary):
    opening = read_opening()
    need(summary['panel'] == 'primary' and summary['panel_gate'] == 'PASS'
         and summary['model_pt_sha256'] == opening['model_pt_sha256']
         and summary['pair_pin'] == opening['pair_pin']
         and summary['assessment_sha256'] == ASSESSMENT_SHA and summary['training_design_sha256'] == DESIGN_SHA
         and summary['original_result_sha256'] == opening['primary_quality_lock']['original_result_sha256']
         and summary['original_lock_sha256'] == digest(reader.canonical(opening['primary_quality_lock']))
         and summary['all_original_arrays_verified'] is summary['original_source_admission_recomputed'] is summary['original_producer_git_verified'] is True,
         'complete original independently reconstructed PRIMARY PASS')
    inventory = summary['source_inventory']
    plan = read_plan()
    expected = {p.removeprefix('p') for p in plan['primary']['synthetic']['speaker_id_map'].values()}
    expected.update(plan['primary']['natural']['speakers'])
    replica = {p.removeprefix('p') for p in plan['replication']['synthetic']['speaker_id_map'].values()}
    replica.update(plan['replication']['natural']['speakers'])
    need(inventory['original_count'] == 120 and len(inventory['pcm_sha256']) == 120
         and set(inventory['canonical_people']) == expected and len(expected) == len(replica) == 36
         and not expected & replica and len(inventory['source_group_id']) == 36,
         'all120 original PRIMARY sources and two fixed disjoint36-person panels')
    for key in ('pcm_sha256','source_group_id','canonical_people'):
        need(inventory[key] == sorted(set(inventory[key])), 'complete unique PRIMARY inventory '+key)
    return summary


def make_history(summary):
    validate_primary(summary)
    baseline = original.read_ancestry()
    inventory = summary['source_inventory']
    mapping = dict(all_pcm_sha256='pcm_sha256',all_source_groups='source_group_id',canonical_people='canonical_people')
    for key, value in mapping.items():
        need(not set(baseline[key]) & set(inventory[value]), 'PRIMARY independent of original history '+key)
    return dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, previous_ancestry_sha256=ANCESTRY_SHA,
        opening_sha256=opening_sha256(), primary_quality=summary,
        **{key:sorted(set(baseline[key]) | set(inventory[value])) for key,value in mapping.items()},
        all_prior_replication_sealed=True, b24_replication_opened=True, training_allowed=False,promotion_allowed=False)


def validate_history(value):
    need(value == make_history(value['primary_quality']), 'complete prior history plus every original PRIMARY source')
    return tuple(set(value[k]) for k in ('all_pcm_sha256','all_source_groups','canonical_people'))


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_opening()
    return _base_execution_guard(prepared_parent, env, event, parent)


# Retain exact parent code, bound to this replica's explicit branch and marker.
_base_execution_guard = type(original.execution_guard)(original.execution_guard.__code__, globals(),
    '_base_execution_guard', original.execution_guard.__defaults__, original.execution_guard.__closure__)


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    opening, t = read_opening(), training_binding()['training']
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh original-PASS replica authorization')
    out.mkdir()
    summary = validate_primary(quality_reader().fetch_quality_original(out/'PRIMARY_QUALITY', opening['primary_quality_lock']))
    commit = json.loads(gh('commits/' + t['run_head']))
    live = dict(run=json.loads(gh('actions/runs/'+str(t['run_id']))),
                artifacts=json.loads(gh('actions/runs/'+str(t['run_id'])+'/artifacts?per_page=100')),
                commit=commit, tree=json.loads(gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')))
    validate_training_live(live)
    qa.record(out/'TRAIN_LIVE.json',live)
    for name,pin in t['artifacts'].items():
        raw=gh('actions/artifacts/'+str(pin['id'])+'/zip')
        need(len(raw)==pin['bytes'] and digest(raw)==pin['sha256'],'unchanged original TRAIN evidence')
        (out/('TRAIN_'+name+'.zip')).write_bytes(raw)
    verify_training(out)
    qa.record(out/'HISTORY.json',make_history(summary))
    (out/'OPENING.json').write_bytes(OPENING.read_bytes())
    qa.record(out/'AUTHORIZATION.json',authorization_document(out,identity,summary))
    return require_authorization(out)


def authorization_document(directory,identity,summary):
    directory=Path(directory); t=training_binding()['training']
    return dict(schema=SCHEMA,source_execution=identity,assessment_sha256=ASSESSMENT_SHA,ancestry_sha256=ANCESTRY_SHA,
        training_binding_sha256=training_binding_sha(),train_result_sha256=t['result_sha256'],
        train_live_sha256=qa.sha(directory/'TRAIN_LIVE.json'),history_sha256=qa.sha(directory/'HISTORY.json'),
        transport_plan_sha256=transport_plan_sha(),maximum_tts_calls=96,maximum_natural_decodes=24,
        maximum_native_observer_calls=600,replication_audio_opened=True,training_allowed=False,promotion_allowed=False,
        new_kws_forwards=0,opening_sha256=opening_sha256(),primary_quality=summary)


def require_authorization(directory):
    opening=read_opening(); directory=Path(directory)
    need((directory/'OPENING.json').read_bytes()==OPENING.read_bytes(),'same original replica opening bytes')
    verify_training(directory)
    summary=validate_primary(quality_reader().verify_quality_original(directory/'PRIMARY_QUALITY',opening['primary_quality_lock']))
    validate_history(json.loads((directory/'HISTORY.json').read_bytes()))
    value=json.loads((directory/'AUTHORIZATION.json').read_bytes())
    need(value==authorization_document(directory,validate_identity(value['source_execution']),summary),
         'replica source requires complete original PRIMARY PASS and unchanged TRAIN identity')
    return value


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS,'registered native provider only')
    return MARKER if provider is None else 'english-v6-b24-replication-observer-'+provider+'-consumed'


def consumption_document(directory):
    value = _base_consumption_document(directory)
    value['source_capacity_sha256'] = qa.sha(Path(directory) / 'CAPACITY.json')
    return value


def consumption(directory, corpus, prepared_parent):
    from english_v6_b24_replication_transport import verify_transport
    import english_v6_b24_replication_natural_sources as natural
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same original PRIMARY-passed replica source identity')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    path = directory / 'CONSUMPTION.json'
    need(not path.exists(), 'single original replica source budget')
    # Full PRIMARY proof and fixed corpus are authenticated above. These reads
    # inspect only member bytes and FLAC STREAMINFO; no decode or observer call.
    qa.record(directory / 'CAPACITY.json', capacity.source_budget(directory, corpus, natural.plan()))
    value = consumption_document(directory)
    qa.record(path, value)
    return value


def consume_observer(provider,source,out,prepared_parent):
    import english_v6_b24_replication_source as owner
    identity=execution_guard(prepared_parent); registration=owner.registration(source)
    need(all(registration[k]==v for k,v in identity.items()),'same original replica before observer consumption')
    one_execution(identity,True); one_execution(identity,False,provider)
    out=Path(out);need(out.is_absolute() and out.parent.is_dir() and not out.exists(),'fresh replica observer budget')
    out.mkdir();value=observer_document(provider,registration);qa.record(out/'OBSERVER_CONSUMPTION.json',value)
    return value


def retain_proof(directory,output):
    directory,output=Path(directory),Path(output)
    require_saved_consumption(directory)
    names=('AUTHORIZATION.json','HISTORY.json','CONSUMPTION.json','CONSUMPTION_UPLOAD.json','CORPUS_TRANSPORT.json','CAPACITY.json',
           'TRAIN_LIVE.json','OPENING.json',*('TRAIN_'+k+'.zip' for k in training_binding()['training']['artifacts']))
    for name in names:
        need(not (output/name).exists(),'fresh original replica proof member')
        shutil.copyfile(directory/name,output/name)
    shutil.copytree(directory/'TRANSPORT',output/'TRANSPORT')
    shutil.copytree(directory/'PRIMARY_QUALITY',output/'PRIMARY_QUALITY')
