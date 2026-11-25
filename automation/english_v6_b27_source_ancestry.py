"""B27 source entry from complete actual B27 TRAIN and closed Q1 originals."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import shutil

import english_v6_b27_profile as profile
original = profile.preserved('english_v6_b22_source_ancestry')
profile.inherit(original, globals())
from english_v6_b27_capacity import before_consumption, before_generation
import english_v6_b27_source_metadata as metadata

HERE = Path(__file__).resolve().parent
PANEL = 'primary'
PLANS = HERE / 'english_v6_b27_panels'
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = profile.profile(False)['assessment_sha256']
ANCESTRY_SHA = profile.profile(False)['ancestry_sha256']
INPUTS = HERE / 'english_v6_b27_primary_proof_binding.json'
BRANCH = 'refs/heads/research/en-v6-b27-primary-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-primary-sources-once.yml'
SCHEMA = 'english-v6-b27-complete-source-authorization-v1'
MARKER = 'english-v6-b27-primary-source-consumed'
TRAIN_BRANCH = 'research/en-v6-b27-natural-weight-train-once-20261125'
TRAIN_WORKFLOW = '.github/workflows/english-v6-b27-natural-weight-train-once.yml'


def read_plan():
    p = profile.profile()
    need(p['assessment_sha256'] == ASSESSMENT_SHA and p['ancestry_sha256'] == ANCESTRY_SHA,
         'same prospectively frozen B27 assessment throughout execution')
    return json.loads(ASSESSMENT.read_bytes())


def read_ancestry():
    read_plan()
    return json.loads((PLANS / 'ancestry.json').read_bytes())


def execution_history():
    value = read_ancestry()
    if PANEL == 'replication':
        proof = profile.require_proof(PANEL)['primary_quality']
        inventory = proof['source_inventory']
        value = dict(value, primary_quality=proof['original_lock_sha256'])
        for target, source in (('all_pcm_sha256', 'pcm_sha256'),
                               ('all_source_groups', 'source_group_id'),
                               ('canonical_people', 'canonical_people')):
            value[target] = sorted(set(value[target]) | set(inventory[source]))
    return value


def validate_history(value):
    need(value == execution_history(), 'all previous original PCM, people and unopened reserves remain excluded')
    return tuple(set(value[key]) for key in ('all_pcm_sha256', 'all_source_groups', 'canonical_people'))


def training_binding(require_bound=True):
    value = profile.binding(PANEL, require_bound)
    if value['status'] == 'INERT_UNTIL_COMPLETE_ORIGINAL_B27_TRAIN':
        return dict(bound=False, training=None, training_producer_manifest_sha256=None)
    lock = value['training_lock']
    t = lock['training']
    need(t['run_branch'] == TRAIN_BRANCH and t['runtime_workflow'] == TRAIN_WORKFLOW
         and t['model_pt_sha256']['control'] == profile.CONTROL_PT
         and t['model_pt_sha256']['challenger'] not in (profile.CONTROL_PT, profile.INITIAL_PT)
         and set(t['model_pt_sha256']) == {'B0', 'B9', 'control', 'challenger'}
         and set(t['artifacts']) == {'models', 'producer', 'receipts', 'consumed'},
         'actual separate B27 candidate and immutable historical B24 control')
    return dict(bound=True, assessment_sha256=ASSESSMENT_SHA, training=t,
                training_producer_manifest_sha256=lock['training_producer_manifest_sha256'])


def training_binding_sha():
    training_binding()
    return digest(INPUTS.read_bytes())


def verify_training(directory):
    # The source carries exact pins, not a recursively duplicated historical ZIP.
    report = profile.require_proof(PANEL)
    need(report['result']['model_pt_sha256'] == training_binding()['training']['model_pt_sha256'],
         'same actual original B27 TRAIN throughout source/quality execution')
    return report['result']


def authorization_document(directory, identity):
    directory = Path(directory)
    result = verify_training(directory)
    return dict(schema=SCHEMA, panel=PANEL, source_execution=identity,
        assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        training_binding_sha256=training_binding_sha(),
        train_result_sha256=training_binding()['training']['result_sha256'],
        train_model_pt_sha256=result['model_pt_sha256'],
        original_training_assessment_sha256=ASSESSMENT_SHA,
        complete_external_training_report_sha256=profile.binding(PANEL)['training_report_sha256'],
        original_B24Q1_failure_report_sha256=profile.Q1_REPORT_SHA,
        original_B24Q1_failure_lock_sha256=profile.Q1_LOCK_SHA,
        history_sha256=qa.sha(directory / 'HISTORY.json'),
        proof_binding_sha256=qa.sha(directory / 'PROOF_BINDING.json'),
        transport_plan_sha256=transport_plan_sha(), maximum_tts_calls=96,
        maximum_natural_decodes=24, maximum_native_observer_calls=600,
        planned_native_slot_rule='exact 5*N, N=96+unchanged preobserver-eligible natural, no more than 600',
        replication_audio_opened=PANEL == 'replication',
        original_primary_pass_required=PANEL == 'replication',
        training_allowed=False, promotion_allowed=False, new_kws_forwards=0,
        new_optimizer_updates=0, automatic_R2_allowed=False,
        automatic_next_assessment_allowed=False, all_old_stops_and_reserves_preserved=True)


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    verify_training(out)
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B27 source authorization')
    out.mkdir()
    (out / 'PROOF_BINDING.json').write_bytes(INPUTS.read_bytes())
    qa.record(out / 'HISTORY.json', execution_history())
    qa.record(out / 'AUTHORIZATION.json', authorization_document(out, identity))
    return require_authorization(out)


def require_authorization(directory):
    directory = Path(directory)
    need((directory / 'PROOF_BINDING.json').read_bytes() == INPUTS.read_bytes(),
         'exact Git-bound complete external original proof registration')
    verify_training(directory)
    validate_history(json.loads((directory / 'HISTORY.json').read_bytes()))
    value = json.loads((directory / 'AUTHORIZATION.json').read_bytes())
    identity = validate_identity(value['source_execution'])
    need(value == authorization_document(directory, identity),
         'actual complete B27 TRAIN, Q1 terminal failure and separately frozen new source assessment')
    return value


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'registered original provider')
    return MARKER if provider is None else 'english-v6-b27-' + PANEL + '-observer-' + provider + '-consumed'


def consumption_document(directory):
    from english_v6_b21_fixed_mirror_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    need((directory / 'CAPACITY.json').is_file(), 'prospective original capacity before consumption')
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER,
        source_execution=permission['source_execution'], authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'),
        corpus_bytes=CORPUS_BYTES, corpus_sha256=CORPUS_SHA,
        transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        capacity_sha256=qa.sha(directory / 'CAPACITY.json'),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, corpus, prepared_parent):
    from english_v6_b21_fixed_mirror_transport import verify_transport
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same new B27 source execution')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    capacity = before_consumption(directory, corpus, read_plan()[PANEL]['natural'], PANEL)
    qa.record(Path(directory) / 'CAPACITY.json', capacity)
    value = consumption_document(directory)
    qa.record(Path(directory) / 'CONSUMPTION.json', value)
    return value


def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b21_fixed_mirror_transport import verify_transport
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity
         and json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed B27 source budget')
    verify_transport(corpus, directory)
    before_generation(directory, corpus, read_plan()[PANEL]['natural'], PANEL)
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'once-only source entry; no retry')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected


def consume_observer(provider, source, out, prepared_parent):
    owner = importlib.import_module('english_v6_b27_source' if PANEL == 'primary'
                                   else 'english_v6_b27_replication_source')
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[key] == value for key, value in identity.items()), 'same B27 native observer identity')
    one_execution(identity, True); one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh original observer budget')
    out.mkdir()
    value = observer_document(provider, registration)
    qa.record(out / 'OBSERVER_CONSUMPTION.json', value)
    return value


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_saved_consumption(directory)
    for name in ('AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json',
                 'CAPACITY.json', 'CONSUMPTION.json', 'CONSUMPTION_UPLOAD.json', 'CORPUS_TRANSPORT.json'):
        need(not (output / name).exists(), 'fresh exact original retained metadata')
        shutil.copyfile(directory / name, output / name)
    shutil.copytree(directory / 'TRANSPORT', output / 'TRANSPORT')
