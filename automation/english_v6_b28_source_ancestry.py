"""B28 SOURCE from complete new TRAIN, all history, and separately consumed stages."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil

import english_v6_b28_profile as profile
original = profile.preserved('english_v6_b22_source_ancestry')
profile.inherit(original, globals())
import english_v6_b28_source_metadata as metadata
import english_v6_b28_multicorpus_source as multicorpus

HERE = Path(__file__).resolve().parent
PANEL = 'primary'
PLANS = metadata.PANELS
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = profile.profile(False)['assessment_sha256']
ANCESTRY_SHA = profile.profile(False)['ancestry_sha256']
INPUTS = HERE / 'english_v6_b28_primary_proof_binding.json'
BRANCH = 'refs/heads/research/en-v6-b28-primary-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b28-primary-sources-once.yml'
SCHEMA = 'english-v6-b28-complete-source-authorization-v1'
MARKER = 'english-v6-b28-primary-source-consumed'
TRAIN_BRANCH = 'research/en-v6-b28-natural-event-train-once-20261125'
TRAIN_WORKFLOW = '.github/workflows/english-v6-b28-natural-event-train-once.yml'
MULTICORPUS_SHA = '27c676ad1f03a1fadf5feb2d5fe27db28ab21ff3d5fef9fae3e2749147bbfef3'


def natural_module():
    return importlib.import_module('english_v6_b28_natural_sources' if PANEL == 'primary'
        else 'english_v6_b28_replication_natural_sources')


def read_plan():
    value = profile.profile()
    need(value['assessment_sha256'] == ASSESSMENT_SHA and value['ancestry_sha256'] == ANCESTRY_SHA,
         'same prospectively frozen B28 assessment and every historical exclusion')
    need(profile.sha(Path(multicorpus.__file__).read_bytes()) == MULTICORPUS_SHA,
         'one fixed complete whole-original staging implementation')
    return profile.document(ASSESSMENT.read_bytes())


def read_ancestry():
    read_plan()
    return profile.document((PLANS / 'ancestry.json').read_bytes())


def execution_history():
    value = read_ancestry()
    if PANEL == 'replication':
        passed = profile.require_proof(PANEL)['primary_quality']
        inventory = passed['source_inventory']
        value = dict(value, primary_quality=passed['original_lock_sha256'])
        for target, source in (('all_pcm_sha256', 'pcm_sha256'), ('all_source_groups', 'source_group_id'),
                               ('canonical_people', 'canonical_people')):
            value[target] = sorted(set(value[target]) | set(inventory[source]))
    return value


def validate_history(value):
    need(value == execution_history(), 'all old originals, aliases and unopened reserves remain excluded')
    return tuple(set(value[key]) for key in ('all_pcm_sha256', 'all_source_groups', 'canonical_people'))


def training_binding(require_bound=True):
    value = profile.binding(PANEL, require_bound)
    if value['status'] == 'INERT_UNTIL_COMPLETE_ORIGINAL_B28_TRAIN':
        return dict(bound=False, training=None, training_producer_manifest_sha256=None)
    lock = value['training_lock']
    training = lock['training']
    need(training['run_branch'] == TRAIN_BRANCH and training['runtime_workflow'] == TRAIN_WORKFLOW
         and training['model_pt_sha256']['control'] == profile.CONTROL_PT
         and training['model_pt_sha256']['challenger'] not in (profile.CONTROL_PT, profile.INITIAL_PT, profile.B27_PT)
         and set(training['model_pt_sha256']) == {'B0', 'B9', 'control', 'challenger'}
         and set(training['artifacts']) == {'models', 'producer', 'receipts', 'consumed'},
         'actual new B28 candidate and immutable original B24 historical control')
    return dict(bound=True, assessment_sha256=ASSESSMENT_SHA, training=training,
        training_producer_manifest_sha256=lock['training_producer_manifest_sha256'])


def training_binding_sha():
    training_binding()
    return digest(INPUTS.read_bytes())


def verify_training(directory=None):
    report = profile.require_proof(PANEL)
    need(report['result']['model_pt_sha256'] == training_binding()['training']['model_pt_sha256'],
         'complete original new TRAIN and same frozen model identities before source access')
    return report['result']


def transport_plan_sha():
    plan, _ = natural_module().staging_plan()
    return multicorpus.sha(multicorpus.encoded(plan))


def authorization_document(directory, identity):
    directory = Path(directory)
    result = verify_training(directory)
    return dict(schema=SCHEMA, panel=PANEL, source_execution=identity,
        assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        training_binding_sha256=training_binding_sha(), train_result_sha256=training_binding()['training']['result_sha256'],
        train_model_pt_sha256=result['model_pt_sha256'], original_training_assessment_sha256=ASSESSMENT_SHA,
        complete_external_training_report_sha256=profile.binding(PANEL)['training_report_sha256'],
        complete_original_B27_NO_PROMOTION_required=True, history_sha256=qa.sha(directory / 'HISTORY.json'),
        proof_binding_sha256=qa.sha(directory / 'PROOF_BINDING.json'), transport_plan_sha256=transport_plan_sha(),
        maximum_tts_calls=96, maximum_natural_decodes=24, maximum_native_observer_calls=600,
        planned_native_slot_rule='exact 5*N, N=96+unchanged preobserver-eligible natural, no more than 600',
        replication_audio_opened=PANEL == 'replication', original_primary_pass_required=PANEL == 'replication',
        training_allowed=False, promotion_allowed=False, new_kws_forwards=0, new_optimizer_updates=0,
        automatic_R2_allowed=False, automatic_next_assessment_allowed=False, all_old_stops_and_reserves_preserved=True)


def marker_name(provider=None):
    need(provider is None or provider in (*PROVIDERS, 'staging'), 'registered current-panel source, staging or native provider')
    if provider == 'staging':
        return 'english-v6-b28-' + PANEL + '-whole-original-staging-consumed'
    return MARKER if provider is None else 'english-v6-b28-' + PANEL + '-observer-' + provider + '-consumed'


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    verify_training()
    for provider in (None, 'staging', *PROVIDERS):
        one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B28 source authorization')
    out.mkdir()
    (out / 'PROOF_BINDING.json').write_bytes(INPUTS.read_bytes())
    qa.record(out / 'HISTORY.json', execution_history())
    plan, _ = natural_module().staging_plan()
    multicorpus.write(out / 'STAGING_PLAN.json', plan)
    qa.record(out / 'AUTHORIZATION.json', authorization_document(out, identity))
    return require_authorization(out)


def require_authorization(directory):
    directory = Path(directory)
    need((directory / 'PROOF_BINDING.json').read_bytes() == INPUTS.read_bytes(), 'exact external original proof binding')
    verify_training(directory)
    validate_history(profile.document((directory / 'HISTORY.json').read_bytes()))
    plan, _ = natural_module().staging_plan()
    need((directory / 'STAGING_PLAN.json').read_bytes() == multicorpus.encoded(plan), 'same 24 fixed whole originals before any stream')
    value = profile.document((directory / 'AUTHORIZATION.json').read_bytes())
    identity = validate_identity(value['source_execution'])
    need(value == authorization_document(directory, identity), 'complete actual B28 TRAIN and independently frozen new SOURCE')
    return value


def staging_consumption_document(directory):
    directory = Path(directory)
    permission = require_authorization(directory)
    plan = profile.document((directory / 'STAGING_PLAN.json').read_bytes())
    return dict(schema=SCHEMA + '-staging-consumption', marker_name=marker_name('staging'),
        source_execution=permission['source_execution'], authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'),
        staging_plan_sha256=transport_plan_sha(), corpus_keys=sorted(plan['corpora']),
        maximum_original_archive_gets=len(plan['corpora']),
        maximum_compressed_bytes=sum(p['bytes'] for p in plan['corpora'].values()),
        selected_original_slots=24, selected_people=12, status='WHOLE_ORIGINAL_STAGING_CONSUMED_UNKNOWN',
        retries=0, alternate_origins=0, native_decodes=0, new_tts_calls=0, new_observer_calls=0, new_kws_forwards=0)


def consume_staging(directory, prepared_parent):
    identity = execution_guard(prepared_parent)
    directory = Path(directory)
    need(require_authorization(directory)['source_execution'] == identity and not (directory / 'NATURAL_STAGING').exists(),
         'same fixed source before first archive stream')
    one_execution(identity, False, 'staging')
    value = staging_consumption_document(directory)
    qa.record(directory / 'STAGING_CONSUMPTION.json', value)
    return value


def require_staging_permit(directory, prepared_parent):
    identity = execution_guard(prepared_parent)
    directory = Path(directory)
    expected = staging_consumption_document(directory)
    need(expected['source_execution'] == identity
         and profile.document((directory / 'STAGING_CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed one-time original staging budget')
    need(not (directory / 'STAGING_CONSUMPTION_UPLOAD.json').exists()
         and not (directory / 'NATURAL_STAGING').exists(), 'single staging entry; no partial-stream retry')
    receipt = uploaded_marker(one_execution(identity, True, 'staging'), expected, 'STAGING_CONSUMPTION.json')
    qa.record(directory / 'STAGING_CONSUMPTION_UPLOAD.json', receipt)
    return expected


def require_saved_staging(directory):
    directory = Path(directory)
    expected = staging_consumption_document(directory)
    need(profile.document((directory / 'STAGING_CONSUMPTION.json').read_bytes()) == expected, 'same retained staging consumption')
    validate_uploaded_marker(profile.document((directory / 'STAGING_CONSUMPTION_UPLOAD.json').read_bytes()), expected,
        'STAGING_CONSUMPTION.json')
    return expected


def consumption_document(directory):
    directory = Path(directory)
    permission = require_authorization(directory)
    require_saved_staging(directory)
    natural_module().read_staging(directory)
    need((directory / 'CAPACITY.json').is_file(), 'exact complete originals fit before native consumption')
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), staging_plan_sha256=transport_plan_sha(),
        staging_release_sha256=qa.sha(directory / 'NATURAL_STAGING/COMPLETE/RELEASE.json'),
        capacity_sha256=qa.sha(directory / 'CAPACITY.json'), status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN',
        maximum_new_tts_calls=96, maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, staging, prepared_parent):
    import english_v6_b28_capacity as capacity
    identity = execution_guard(prepared_parent)
    directory = Path(directory)
    need(require_authorization(directory)['source_execution'] == identity and Path(staging) == directory / 'NATURAL_STAGING',
         'same current-panel complete staging before native budget')
    one_execution(identity, False)
    require_saved_staging(directory)
    value = capacity.before_consumption(directory, staging, read_plan()[PANEL]['natural'], PANEL)
    qa.record(directory / 'CAPACITY.json', value)
    receipt = consumption_document(directory)
    qa.record(directory / 'CONSUMPTION.json', receipt)
    return receipt


def require_generation_permit(directory, staging, prepared_parent):
    import english_v6_b28_capacity as capacity
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity and Path(staging) == directory / 'NATURAL_STAGING'
         and profile.document((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed B28 TTS/decode budget')
    capacity.before_generation(directory, staging, read_plan()[PANEL]['natural'], PANEL)
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'single native-generation entry; no retry')
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected


def consume_observer(provider, source, out, prepared_parent):
    owner = importlib.import_module('english_v6_b28_source' if PANEL == 'primary' else 'english_v6_b28_replication_source')
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[k] == value for k, value in identity.items()), 'same B28 native observer identity')
    one_execution(identity, True)
    one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh original observer budget')
    out.mkdir()
    value = observer_document(provider, registration)
    qa.record(out / 'OBSERVER_CONSUMPTION.json', value)
    return value


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_saved_consumption(directory)
    for name in ('AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json', 'STAGING_PLAN.json',
        'STAGING_CONSUMPTION.json', 'STAGING_CONSUMPTION_UPLOAD.json', 'CAPACITY.json', 'CONSUMPTION.json',
        'CONSUMPTION_UPLOAD.json'):
        need(not (output / name).exists(), 'fresh exact retained source evidence')
        shutil.copyfile(directory / name, output / name)
    shutil.copytree(directory / 'NATURAL_STAGING', output / 'NATURAL_STAGING')
    shutil.copytree(directory / 'STAGING_HTTP', output / 'STAGING_HTTP')
