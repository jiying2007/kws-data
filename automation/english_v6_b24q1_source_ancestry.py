"""One approved fixed-weight Q1 assessment after the complete original R1 STOP.

The source-entry authorization binds explicit user approval, one new prospectively
frozen assessment and the complete unchanged R1 fail-only proof. The original B24
TRAIN assessment and four model weights remain bound to their historical evidence.
Saved readers reopen every R1 original in the exact immutable isolated reader;
neither stopped trial is reinterpreted under the new Q1 orchestration policy.
"""
from __future__ import annotations
import sys
import tempfile
import english_v6_b24_source_ancestry as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24q1_source_metadata as metadata
import english_v6_b24q1_r1_source_failure as failure

PLANS = metadata.PANELS
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = '681ddb42432bd11dfef0157d2f1c95f7e02824c3a2ed7da64ce9a12bdf9c3139'
ANCESTRY_SHA = '5add13ba0bbd24cf5931998ec1866920cd02d7d0ec5b5743d10a267efa4dd6ec'
BRANCH = 'refs/heads/research/en-v6-b24q1-primary-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b24q1-primary-sources-once.yml'
SCHEMA = 'english-v6-b24q1-complete-source-authorization-v1'
MARKER = 'english-v6-b24q1-primary-source-consumed'
# These exact aliases retain original module globals. The completed TRAIN is
# never relabelled with an assessment selected after its optimizer finished.
training_binding = original.training_binding
training_binding_sha = original.training_binding_sha
validate_training_live = original.validate_training_live
verify_training = original.verify_training
_original_execution_guard = execution_guard
_original_retain_proof = retain_proof
STOP_BINDING = 'B24R1_SOURCE_STOP_BINDING.json'
STOP_REPORT = 'B24R1_SOURCE_STOP_REPORT.json'

def execution_guard(prepared_parent, env=None, event=None, parent=None):
    identity = _original_execution_guard(prepared_parent, env, event, parent)
    opening = failure.read_binding()
    need(opening['assessment'] == dict(path=metadata.PREFIX+'assessment.json',sha256=ASSESSMENT_SHA),
         'same approved frozen Q1 assessment in source entry and original STOP opening')
    training_binding()
    return identity


def stop_scope():
    opening = failure.read_binding()
    return dict(original_B24R1_SOURCE_STOP_authority='actual_original_B24Q1_SOURCE_entry',
        original_B24R1_SOURCE_STOP_all14_archives_reopened_at_source_entry=True,
        original_B24R1_SOURCE_STOP_archives_reopened_by_saved_source_reader=True,
        original_B24R1_SOURCE_STOP_report_sha256=opening['expected_failure_report_sha256'],
        original_B24R1_SOURCE_STOP_binding_sha256=digest(failure.BINDING.read_bytes()),
        original_B24R1_source_stop_permanent=True,original_B24_source_stop_permanent=True,
        approved_proposal_sha256=metadata.PROPOSAL_SHA,approval_sha256=metadata.APPROVAL_SHA,
        orchestration_contract_sha256=metadata.ORCHESTRATION_SHA,
        original_whole_panel_empty_return_policy_changed=True,
        original_native_validity_and_numerical_admission_unchanged=True,
        automatic_R2_allowed=False,automatic_next_assessment_allowed=False,
        new_optimizer_updates=0,original_training_assessment_sha256=metadata.TRAINING_ASSESSMENT_SHA)


def validate_stop_report(report, opening):
    """The exact old fail-only report is required; Q1 never reclassifies it."""
    lock = opening['original_source_lock']
    need(digest(failure.canonical(report)) == opening['expected_failure_report_sha256'] == metadata.STOP_REPORT_SHA
         and report['schema'] == failure.REPORT_SCHEMA
         and report['status'] == 'SOURCE_STOP_EMPTY_NATIVE_TRANSCRIPT'
         and report['original_lock_sha256'] == digest(failure.canonical(lock))
         and report['reader_sha256'] == failure.READER_SHA == digest(failure.READER_PATH.read_bytes())
         and report['original_candidate_pt_sha256'] == metadata.CANDIDATE_PT
         and report['original_assessment_sha256'] == metadata.PREVIOUS_ASSESSMENT_SHA
         and report['original_training_assessment_sha256'] == metadata.TRAINING_ASSESSMENT_SHA
         and report['original_train_result_sha256'] == training_binding()['training']['result_sha256']
         and report['all14_original_zip_digests_crc_verified'] is True
         and report['R1_source_stop_permanent'] is report['original_B24_source_stop_permanent'] is True
         and report['new_source_registration_performed'] is False
         and report['total_valid_payloads_reparsed'] == 488
         and report['empty_return_reparsed_by_original_parser'] is True
         and report['original_phone_model_vocab_finite_logits_and_greedy_ctc_recomputed'] is True
         and all(report[key] is False for key in failure.FALSE_PERMISSIONS),
         'exact unchanged complete R1 SOURCE STOP, corrected original payload reparse and immutable B24 weights')
    return report


def saved_stop(directory):
    directory = Path(directory)
    opening = failure.read_binding()
    failure.verify_transport(directory)
    need((directory/STOP_BINDING).read_bytes() == failure.BINDING.read_bytes(),
         'exact Git-bound approved original R1 STOP opening retained in source')
    raw = (directory/STOP_REPORT).read_bytes()
    report = failure.document(raw)
    need(raw == failure.canonical(report),'one exact canonical original R1 STOP report')
    validate_stop_report(report,opening)
    # Direct immutable worker: no mutable adapter executable or nested process.
    with tempfile.TemporaryDirectory(prefix='b24q1-original-r1-stop-lock-') as temporary:
        lock_path = Path(temporary)/'lock.json'
        lock_path.write_bytes(failure.canonical(opening['original_source_lock']))
        raw_verified = failure.verify_in_scope(directory/'ORIGINAL_B24R1_STOP',opening['original_source_lock'],
            lambda: subprocess.check_output([sys.executable,'-I','-S','-B',str(failure.READER_PATH.resolve()),
                '--verify-worker',str((directory/'ORIGINAL_B24R1_STOP').resolve()),str(lock_path)]))
    need(raw_verified == raw,'all fourteen original R1 archives reopen to the exact frozen fail-only report')
    return opening,report


def authorization_document(directory, identity):
    directory = Path(directory)
    t = training_binding()['training']
    saved_stop(directory)
    return dict(schema=SCHEMA,source_execution=identity,assessment_sha256=ASSESSMENT_SHA,
        ancestry_sha256=ANCESTRY_SHA,training_binding_sha256=training_binding_sha(),
        original_training_assessment_sha256=metadata.TRAINING_ASSESSMENT_SHA,
        train_result_sha256=t['result_sha256'],train_live_sha256=qa.sha(directory/'TRAIN_LIVE.json'),
        history_sha256=qa.sha(directory/'HISTORY.json'),transport_plan_sha256=transport_plan_sha(),
        original_source_stop_binding_sha256=qa.sha(directory/STOP_BINDING),
        original_source_stop_report_sha256=qa.sha(directory/STOP_REPORT),
        approved_proposal_sha256=metadata.PROPOSAL_SHA,approval_sha256=metadata.APPROVAL_SHA,
        orchestration_contract_sha256=metadata.ORCHESTRATION_SHA,
        maximum_tts_calls=96,maximum_natural_decodes=24,maximum_native_observer_calls=600,
        planned_native_slot_rule='exact 5*N, N=96+unchanged preobserver-eligible natural, no more than 600',
        original_whole_panel_empty_return_policy_changed=True,
        original_native_validity_and_numerical_admission_unchanged=True,
        replication_audio_opened=False,training_allowed=False,promotion_allowed=False,
        new_kws_forwards=0,new_optimizer_updates=0,automatic_R2_allowed=False,automatic_next_assessment_allowed=False,
        original_B24_source_stop_permanent=True,original_B24R1_source_stop_permanent=True,
        original_B24_replication_opened=False,original_B24R1_replication_opened=False,
        old_B25_and_B26_openers_unchanged=True)


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    t = training_binding()['training']
    opening = failure.read_binding()
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B24Q1 source authorization')
    out.mkdir()
    # Every R1 ZIP retains its historical role cap. The separately registered
    # Q1 composite envelope does not alter any embedded archive parser.
    report = failure.fetch(out / 'ORIGINAL_B24R1_STOP', opening['original_source_lock'])
    validate_stop_report(report, opening)
    (out / STOP_BINDING).write_bytes(failure.BINDING.read_bytes())
    (out / STOP_REPORT).write_bytes(failure.canonical(report))
    commit = json.loads(gh('commits/' + t['run_head']))
    live = dict(run=json.loads(gh('actions/runs/' + str(t['run_id']))),
        artifacts=json.loads(gh('actions/runs/' + str(t['run_id']) + '/artifacts?per_page=100')),
        commit=commit, tree=json.loads(gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')))
    validate_training_live(live)
    qa.record(out / 'TRAIN_LIVE.json', live)
    for name, pin in t['artifacts'].items():
        raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'original TRAIN evidence download ' + name)
        (out / ('TRAIN_' + name + '.zip')).write_bytes(raw)
    verify_training(out)
    qa.record(out / 'HISTORY.json', read_ancestry())
    qa.record(out / 'AUTHORIZATION.json', authorization_document(out, identity))
    return require_authorization(out)

def require_authorization(directory):
    directory = Path(directory)
    verify_training(directory)
    validate_history(json.loads((directory / 'HISTORY.json').read_bytes()))
    value = json.loads((directory / 'AUTHORIZATION.json').read_bytes())
    identity = validate_identity(value['source_execution'])
    need(value == authorization_document(directory, identity),
         'complete actual B24 TRAIN, exact original R1 STOP and approved new bounded Q1 identity')
    return value

def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'registered native provider only')
    return MARKER if provider is None else 'english-v6-b24q1-primary-observer-' + provider + '-consumed'

def consumption_document(directory):
    from english_v6_b24q1_source_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    capacity = directory / 'CAPACITY.json'
    need(capacity.is_file() and not capacity.is_symlink(), 'original prospective capacity receipt before consumption')
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), corpus_bytes=CORPUS_BYTES,
        corpus_sha256=CORPUS_SHA, transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        capacity_sha256=qa.sha(capacity),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)

def consumption(directory, corpus, prepared_parent):
    from english_v6_b24q1_source_transport import verify_transport
    from english_v6_b24q1_capacity import primary_source_budget
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same new primary source identity')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    path = Path(directory) / 'CONSUMPTION.json'
    need(not path.exists(), 'single original source budget')
    capacity = primary_source_budget(directory, corpus, read_plan()['primary']['natural'])
    qa.record(Path(directory) / 'CAPACITY.json', capacity)
    value = consumption_document(directory)
    qa.record(path, value)
    return value

def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b24q1_source_transport import verify_transport
    from english_v6_b24q1_capacity import primary_generation_capacity
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity and json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed source budget before TTS or decode')
    verify_transport(corpus, directory)
    primary_generation_capacity(directory, corpus, read_plan()['primary']['natural'])
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'one source entry, no retry')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected

def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b24q1_source as owner
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[k] == value for k, value in identity.items()), 'same new source before observer consumption')
    one_execution(identity, True)
    one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh individual observer budget')
    out.mkdir()
    value = observer_document(provider, registration)
    qa.record(out / 'OBSERVER_CONSUMPTION.json', value)
    return value

def retain_proof(directory, output):
    _original_retain_proof(directory, output)
    saved_stop(directory)
    for name in (STOP_BINDING, STOP_REPORT, 'CAPACITY.json'):
        target = Path(output) / name
        need(not target.exists(), 'fresh immutable original STOP evidence member')
        shutil.copyfile(Path(directory) / name, target)
    shutil.copytree(Path(directory) / 'ORIGINAL_B24R1_STOP', Path(output) / 'ORIGINAL_B24R1_STOP')
    failure.retain_transport(directory, output)
