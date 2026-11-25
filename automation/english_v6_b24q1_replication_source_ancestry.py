"""Sealed B24Q1 replica authorization using only original complete PRIMARY PASS."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24q1_source_ancestry as original
import english_v6_b24q1_quality_readback as reader
import english_v6_b24q1_capacity as capacity
import english_v6_b24q1_evidence_scope as evidence_scope
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())

BRANCH = 'refs/heads/research/en-v6-b24q1-replication-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b24q1-replication-sources-once.yml'
SCHEMA = 'english-v6-b24q1-replication-source-authorization-v1'
MARKER = 'english-v6-b24q1-replication-source-consumed'
OPENING = HERE / 'english_v6_b24q1_replication_opening.json'
READER = HERE / 'english_v6_b24q1_quality_readback.py'
DESIGN_SHA = original.metadata.DESIGN_SHA
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
_base_consumption_document = consumption_document


def read_opening(require_bound=True):
    value = json.loads(OPENING.read_bytes())
    need(value['schema'] == 'english-v6-b24q1-replication-original-primary-opening-v1'
         and value['panel'] == 'replication' and value['assessment_sha256'] == ASSESSMENT_SHA
         and value['training_design_sha256'] == DESIGN_SHA
         and value['training_assessment_sha256'] == original.metadata.TRAINING_ASSESSMENT_SHA
         and value['evaluation_only'] is value['B24_original_source_stop_preserved'] is value['B24R1_original_source_stop_preserved'] is True
         and value['approved_proposal_sha256'] == original.metadata.PROPOSAL_SHA
         and value['old_B24R1_sources_used_for_scoring'] is value['old_B24R1_replication_opened'] is False
         and value['original_whole_panel_failure_contract_unchanged'] is False
         and type(value['new_optimizer_updates']) is int and value['new_optimizer_updates'] == 0
         and value['old_B24_sources_used_for_scoring'] is value['old_B24_replication_opened'] is value['automatic_R2_allowed'] is False
         and value['original_primary_pass_required'] is True
         and value['training_allowed'] is value['promotion_allowed'] is value['same_job_matched_control'] is False,
         'exact frozen B24Q1 replica and historical-control semantics')
    if value['status'] == 'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS':
        need(all(value[k] is None for k in ('primary_quality_lock','quality_readback_sha256','pair_pin','model_pt_sha256',
                                          'primary_quality_report_sha256','primary_quality_files')),
             'all replica opening pins remain unset until actual PRIMARY PASS')
        need(not require_bound, 'STOP: replica remains sealed without original PRIMARY PASS')
        return value
    need(value['status'] == 'BOUND_ORIGINAL_PRIMARY_PASS'
         and digest(READER.read_bytes()) == value['quality_readback_sha256'], 'exact immutable original-quality reader')
    reader.validate_lock(value['primary_quality_lock'])
    need(value['primary_quality_lock']['panel'] == 'primary', 'original primary lock only')
    validate_primary_files(value)
    reader.valid_artifact(value['pair_pin'])
    model = value['model_pt_sha256']
    t = training_binding()['training']
    need(type(model) is dict and model == t['model_pt_sha256']
         and model['challenger'] == CANDIDATE_PT and model['control'] == original.metadata.CONTROL_PT
         and value['pair_pin'] == dict(id=t['artifacts']['models']['id'], run_id=t['run_id'], run_head=t['run_head'],
             bytes=t['artifacts']['models']['bytes'], sha256=t['artifacts']['models']['sha256']),
         'same successful original B24Q1 challenger and immutable historical B23 control')
    return value


def opening_sha256():
    read_opening()
    return digest(OPENING.read_bytes())


def quality_reader():
    opening = read_opening()
    need(Path(reader.__file__).resolve() == READER and digest(READER.read_bytes()) == opening['quality_readback_sha256'],
         'exact reviewed original PRIMARY quality reader')
    return reader


def validate_primary_files(opening):
    files, lock = opening['primary_quality_files'], opening['primary_quality_lock']
    need(reader.valid_hash(opening['primary_quality_report_sha256'])
         and type(files) is dict and set(files) == capacity.PRIMARY_FILES,
         'frozen canonical PRIMARY report and all eighteen original file pins')
    for name, pin in files.items():
        maximum = capacity.PRIMARY_SOURCE_LIMIT if name == 'inputs/source.zip' else capacity.CORE_LIMIT
        need(type(pin) is dict and set(pin) == {'bytes','sha256'}
             and type(pin['bytes']) is int and 0 < pin['bytes'] <= maximum and reader.valid_hash(pin['sha256']),
             'exact original PRIMARY file and unchanged role cap: ' + name)
    for role in ('result','producer','consumed'):
        need(files['originals/'+role+'.zip'] == {key:lock[role][key] for key in ('bytes','sha256')},
             'three original quality archive pins equal the immutable opening lock')
    raw = (HERE / 'english_v6_b24q1_primary_inputs.json').read_bytes()
    need(digest(raw) == lock['inputs_sha256'], 'exact original PRIMARY producer SPEC, never later reconstructed input pins')
    spec = reader.document(raw)
    need(spec['schema'] == 'english-v6-b24q1-fresh-primary-four-frozen-models-v1' and spec['scoring_allowed'] is True,
         'actual bound original PRIMARY registration')
    pins = {**spec['inputs'], **{'native_'+p:pin for p,pin in spec['native_inputs'].items()},
            **{role+'_producer':pin for role,pin in spec['upstream_producers'].items()}}
    need({'inputs/'+role+'.zip' for role in pins} == {name for name in capacity.PRIMARY_FILES if name.startswith('inputs/')},
         'all twelve original PRIMARY input roles')
    for role, pin in pins.items():
        reader.valid_artifact(pin, panel='primary', role=role)
        need(files['inputs/'+role+'.zip'] == {key:pin[key] for key in ('bytes','sha256')},
             'each fixed input file is byte-identical to the original producer SPEC: '+role)
    return files


def validate_primary(summary, opening=None):
    opening = read_opening() if opening is None else opening
    need(summary['panel'] == 'primary' and summary['panel_gate'] == 'PASS'
         and digest(reader.canonical(summary)) == opening['primary_quality_report_sha256']
         and summary['model_pt_sha256'] == opening['model_pt_sha256']
         and summary['pair_pin'] == opening['pair_pin']
         and summary['assessment_sha256'] == ASSESSMENT_SHA and summary['training_design_sha256'] == DESIGN_SHA
         and summary['training_assessment_sha256'] == original.metadata.TRAINING_ASSESSMENT_SHA
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


def verify_primary(directory):
    """Rehash all eighteen originals on every call; reuse only one audit's full PASS."""
    opening = read_opening()
    verifier = quality_reader()
    raw = evidence_scope.verify_once('q1_primary_quality', Path(directory), opening['primary_quality_files'],
        verifier.canonical(opening['primary_quality_lock']), READER, opening['quality_readback_sha256'],
        opening['primary_quality_report_sha256'],
        lambda: verifier.canonical(verifier.verify_quality_original(Path(directory), opening['primary_quality_lock'])))
    return validate_primary(verifier.document(raw), opening)


def bind_opening(directory, lock, output):
    """Freeze an actual complete original PASS and its eighteen files into a new file."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    value = read_opening(False)
    need(value['status'] == 'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS' and not output.exists() and output.parent.is_dir(),
         'only sealed opening and fresh reviewed proposal output')
    reader.validate_lock(lock)
    need(lock['panel'] == 'primary', 'only original PRIMARY may open the replica')
    before = capacity._primary(directory)['files']
    with evidence_scope.scope():
        summary = reader.verify_quality_original(directory, lock)
    need(capacity._primary(directory)['files'] == before, 'all eighteen original files unchanged during full PRIMARY readback')
    with reader.original(directory/'originals/producer.zip',lock['producer'],panel='primary',role='producer') as archive:
        raw = archive.read('primary_producer/automation/english_v6_b24q1_primary_inputs.json')
        need(digest(raw) == lock['inputs_sha256']
             and raw == (HERE/'english_v6_b24q1_primary_inputs.json').read_bytes(),
             'actual original producer SPEC equals the exact captured registration used by future readers')
    value.update(status='BOUND_ORIGINAL_PRIMARY_PASS',primary_quality_lock=lock,
        quality_readback_sha256=digest(READER.read_bytes()),primary_quality_files=before,
        primary_quality_report_sha256=digest(reader.canonical(summary)),
        pair_pin=summary['pair_pin'],model_pt_sha256=summary['model_pt_sha256'])
    validate_primary_files(value)
    validate_primary(summary,value)
    with output.open('xb') as handle:
        handle.write(reader.canonical(value))
    return value


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
        all_prior_replication_sealed=True, b24q1_replication_opened=True, original_B24_replication_opened=False,
        B24_original_source_stop_preserved=True, B24R1_original_source_stop_preserved=True, original_B24R1_replication_opened=False,
        automatic_R2_allowed=False,
        training_allowed=False,promotion_allowed=False)


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
    quality_reader().fetch_quality_original(out/'PRIMARY_QUALITY', opening['primary_quality_lock'])
    summary = verify_primary(out/'PRIMARY_QUALITY')
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
        new_kws_forwards=0,opening_sha256=opening_sha256(),primary_quality=summary,
        evaluation_only=True, original_B24_source_stop_permanent=True,
        original_training_assessment_sha256=original.metadata.TRAINING_ASSESSMENT_SHA,
        automatic_R2_allowed=False)


def require_authorization(directory):
    opening=read_opening(); directory=Path(directory)
    need((directory/'OPENING.json').read_bytes()==OPENING.read_bytes(),'same original replica opening bytes')
    verify_training(directory)
    summary=verify_primary(directory/'PRIMARY_QUALITY')
    validate_history(json.loads((directory/'HISTORY.json').read_bytes()))
    value=json.loads((directory/'AUTHORIZATION.json').read_bytes())
    need(value==authorization_document(directory,validate_identity(value['source_execution']),summary),
         'replica source requires complete original PRIMARY PASS and unchanged TRAIN identity')
    return value


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS,'registered native provider only')
    return MARKER if provider is None else 'english-v6-b24q1-replication-observer-'+provider+'-consumed'


def consumption_document(directory):
    from english_v6_b24q1_replication_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), corpus_bytes=CORPUS_BYTES,
        corpus_sha256=CORPUS_SHA, transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        source_capacity_sha256=qa.sha(directory / 'CAPACITY.json'),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, corpus, prepared_parent):
    from english_v6_b24q1_replication_transport import verify_transport
    import english_v6_b24q1_replication_natural_sources as natural
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


def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b24q1_replication_transport import verify_transport
    import english_v6_b24q1_replication_natural_sources as natural
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity and json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed replica budget before TTS or decode')
    verify_transport(corpus, directory)
    capacity.replica_generation_capacity(directory, corpus, natural.plan())
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'one replica source entry, no retry')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected


def consume_observer(provider,source,out,prepared_parent):
    import english_v6_b24q1_replication_source as owner
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
