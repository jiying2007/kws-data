"""One new fixed-weight source panel after the complete original B24 SOURCE STOP.

The original B24 TRAIN retains its own assessment and complete original proof.
B24R1 has a separate post-training evaluation identity. At the actual source
entry all fourteen original STOP archives are reopened and semantically read.
Saved source readers independently reopen all fourteen retained original ZIPs
in a fresh standard-library process. Each original archive keeps its old cap;
only the separately registered B24R1 composite envelope is larger.
"""
from __future__ import annotations
import sys
import tempfile
import english_v6_b24_source_ancestry as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24r1_source_metadata as metadata
import english_v6_b24r1_b24_source_failure as failure

PLANS = metadata.PANELS
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = 'e783c982f66ed35f903680772842ca8068aad6fb18221e22b9512cd819efa075'
ANCESTRY_SHA = 'eca9b1fa081022f0efa4da79f4d398affdce01a6ebdd81e4cad0bcd6405d3c06'
BRANCH = 'refs/heads/research/en-v6-b24r1-primary-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b24r1-primary-sources-once.yml'
SCHEMA = 'english-v6-b24r1-complete-source-authorization-v1'
MARKER = 'english-v6-b24r1-primary-source-consumed'
# These functions deliberately retain the old module globals. The TRAIN
# assessment is never rebound to the new evaluation-only assessment.
training_binding = original.training_binding
training_binding_sha = original.training_binding_sha
validate_training_live = original.validate_training_live
verify_training = original.verify_training
_original_execution_guard = execution_guard
_original_retain_proof = retain_proof
STOP_BINDING = 'B24_SOURCE_STOP_BINDING.json'
STOP_REPORT = 'B24_SOURCE_STOP_REPORT.json'


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    identity = _original_execution_guard(prepared_parent, env, event, parent)
    failure.read_binding()
    training_binding()
    return identity


def stop_scope():
    opening = failure.read_binding()
    return dict(original_B24_SOURCE_STOP_authority='actual_original_B24R1_SOURCE_entry',
        original_B24_SOURCE_STOP_all14_archives_reopened_at_source_entry=True,
        original_B24_SOURCE_STOP_archives_reopened_by_saved_source_reader=True,
        original_B24_SOURCE_STOP_report_sha256=opening['expected_failure_report_sha256'],
        original_B24_SOURCE_STOP_binding_sha256=digest(failure.BINDING.read_bytes()),
        original_B24_source_stop_permanent=True, automatic_R2_allowed=False,
        new_optimizer_updates=0, original_training_assessment_sha256=metadata.TRAINING_ASSESSMENT_SHA)


def validate_stop_report(report, opening):
    """Check the exact report frozen after complete independent STOP readback.

    Every entry and saved reader also recomputes this exact report from all
    retained original archives before accepting its bound output.
    """
    lock = opening['original_source_lock']
    need(digest(failure.canonical(report)) == opening['expected_failure_report_sha256']
         and report['schema'] == failure.REPORT_SCHEMA
         and report['status'] == 'FAILED_EMPTY_NATIVE_TRANSCRIPT'
         and report['original_lock_sha256'] == digest(failure.canonical(lock))
         and report['adapter_sha256'] == digest(Path(failure.__file__).read_bytes())
         and report['candidate_pt_sha256'] == metadata.CANDIDATE_PT
         and report['original_assessment_sha256'] == metadata.TRAINING_ASSESSMENT_SHA
         and report['original_train_result_sha256'] == training_binding()['training']['result_sha256']
         and report['all14_original_archives_reopened_by_this_helper'] is True
         and report['original_source_stop_permanent'] is True
         and report['new_source_registration_performed'] is False
         and report['total_valid_payloads_reparsed'] == 528
         and report['empty_return_reparsed_by_original_parser'] is True
         and all(report[key] is False for key in failure.FALSE_PERMISSIONS),
         'exact independently frozen complete original STOP report, unchanged B24 weights and no old retry')
    return report


def saved_stop(directory):
    directory = Path(directory)
    opening = failure.read_binding()
    need((directory / STOP_BINDING).read_bytes() == failure.BINDING.read_bytes(),
         'exact Git-bound complete STOP opening retained in original source')
    raw = (directory / STOP_REPORT).read_bytes()
    report = failure.document(raw)
    need(raw == failure.canonical(report), 'one exact canonical saved STOP report')
    validate_stop_report(report, opening)
    # A fresh stdlib process keeps source readback free of loaded TTS/observer
    # runtimes even when this call follows generation in the same parent.
    with tempfile.TemporaryDirectory(prefix='b24r1-original-stop-lock-') as temporary:
        lock_path = Path(temporary) / 'lock.json'
        lock_path.write_bytes(failure.canonical(opening['original_source_lock']))
        raw_verified = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(failure.__file__).resolve()),
            '--verify-worker', str((directory / 'ORIGINAL_B24_STOP').resolve()), str(lock_path)])
    need(raw_verified == raw, 'complete original fourteen-archive STOP readback equals the frozen report')
    return opening, report


def authorization_document(directory, identity):
    directory = Path(directory)
    t = training_binding()['training']
    saved_stop(directory)
    return dict(schema=SCHEMA, source_execution=identity, assessment_sha256=ASSESSMENT_SHA,
        ancestry_sha256=ANCESTRY_SHA, training_binding_sha256=training_binding_sha(),
        original_training_assessment_sha256=metadata.TRAINING_ASSESSMENT_SHA,
        train_result_sha256=t['result_sha256'], train_live_sha256=qa.sha(directory / 'TRAIN_LIVE.json'),
        history_sha256=qa.sha(directory / 'HISTORY.json'), transport_plan_sha256=transport_plan_sha(),
        original_source_stop_binding_sha256=qa.sha(directory / STOP_BINDING),
        original_source_stop_report_sha256=qa.sha(directory / STOP_REPORT),
        maximum_tts_calls=96, maximum_natural_decodes=24, maximum_native_observer_calls=600,
        replication_audio_opened=False, training_allowed=False, promotion_allowed=False,
        new_kws_forwards=0, new_optimizer_updates=0, automatic_R2_allowed=False,
        original_B24_source_stop_permanent=True, original_B24_replication_opened=False)


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    t = training_binding()['training']
    opening = failure.read_binding()
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B24R1 source authorization')
    out.mkdir()
    # Every old ZIP remains individually pinned with its original 128 MB
    # limit; the new composite SOURCE has a separate prospective envelope.
    report = failure.fetch(out / 'ORIGINAL_B24_STOP', opening['original_source_lock'])
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
         'complete actual B24 TRAIN, exact original STOP report and new bounded evaluation identity')
    return value


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'registered native provider only')
    return MARKER if provider is None else 'english-v6-b24r1-primary-observer-' + provider + '-consumed'


def consumption_document(directory):
    from english_v6_b24r1_source_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), corpus_bytes=CORPUS_BYTES,
        corpus_sha256=CORPUS_SHA, transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, corpus, prepared_parent):
    from english_v6_b24r1_source_transport import verify_transport
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same new primary source identity')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    path = Path(directory) / 'CONSUMPTION.json'
    need(not path.exists(), 'single original source budget')
    value = consumption_document(directory)
    qa.record(path, value)
    return value


def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b24r1_source_transport import verify_transport
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity and json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed source budget before TTS or decode')
    verify_transport(corpus, directory)
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'one source entry, no retry')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected


def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b24r1_source as owner
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
    for name in (STOP_BINDING, STOP_REPORT):
        target = Path(output) / name
        need(not target.exists(), 'fresh immutable original STOP evidence member')
        shutil.copyfile(Path(directory) / name, target)
    shutil.copytree(Path(directory) / 'ORIGINAL_B24_STOP', Path(output) / 'ORIGINAL_B24_STOP')
