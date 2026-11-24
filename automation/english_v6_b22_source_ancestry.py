"""B22 primary authorization, frozen history and separately consumed source budgets.

The new assessment was committed before training. Only an authenticated,
completed B22 training run can open these new primary sources. All original
sources, including rejects, survive; historical replication remains sealed.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import english_v6_b21_source_ancestry as previous
import english_v6_b22_source_metadata as metadata

qa, need, digest, canonical = previous.qa, previous.need, previous.digest, previous.canonical
PLANS = metadata.PANELS
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = '1b4e043838c1313b14e31c45395fcce394526892562fb1c4e6c6793c5e45370e'
ANCESTRY_SHA = '5b8d154e18a1b18cbd3b89d413b8243214ab51a800fac0a25ff8fc92a76da3f7'
INPUTS = Path(__file__).with_name('english_v6_b22_source_inputs.json')
TRAIN_PRODUCER = Path(__file__).with_name('english_v6_b22_source_training_producer.json')
REPO = 'jiying2007/kws-data'
BRANCH = 'refs/heads/research/en-v6-b22-primary-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b22-primary-sources-once.yml'
SCHEMA = 'english-v6-b22-complete-source-authorization-v1'
MARKER = 'english-v6-b22-primary-source-consumed'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'xlsr', 'lv60')
CORPUS_BYTES = 6387309499
CORPUS_SHA = 'd4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2'


def read_plan():
    metadata.validate()
    need(digest(ASSESSMENT.read_bytes()) == ASSESSMENT_SHA
         and digest((PLANS / 'ancestry.json').read_bytes()) == ANCESTRY_SHA,
         'same prospectively frozen B22 assessment and complete history')
    return json.loads(ASSESSMENT.read_bytes())


def read_ancestry():
    read_plan()
    return json.loads((PLANS / 'ancestry.json').read_bytes())


def validate_history(value):
    need(value == read_ancestry(), 'every historical PCM, canonical person and sealed reserve remains excluded')
    return tuple(set(value[k]) for k in ('all_pcm_sha256', 'all_source_groups', 'canonical_people'))


def training_binding(require_bound=True):
    value = json.loads(INPUTS.read_bytes())
    need(value['schema'] == 'english-v6-b22-completed-train-source-binding-v1'
         and value['assessment_sha256'] == ASSESSMENT_SHA,
         'source execution binds exactly the assessment frozen before optimizer')
    if value['bound'] is False:
        need(value['training'] is None and not require_bound, 'actual successful B22 training must be bound first')
        return value
    need(value['bound'] is True and type(value['training']) is dict, 'explicit original training binding')
    need(re.fullmatch(r'[0-9a-f]{64}', value.get('training_producer_manifest_sha256') or '')
         and digest(TRAIN_PRODUCER.read_bytes()) == value['training_producer_manifest_sha256'],
         'entire actual successful training producer manifest is frozen before source capture')
    manifest = json.loads(TRAIN_PRODUCER.read_bytes())
    need(manifest['schema'] == 'english-v6-b22-complete-git-producer-v1'
         and type(manifest['files']) is dict and len(manifest['files']) > 243,
         'complete original first-fit continuation producer, including initial failure evidence')
    t = value['training']
    need(set(t) == {'run_id', 'run_head', 'prepared_parent', 'run_branch', 'runtime_workflow',
                   'artifacts', 'result_sha256', 'model_pt_sha256'}
         and type(t['run_id']) is int and t['run_id'] > 0
         and all(re.fullmatch(r'[0-9a-f]{40}', t[k]) for k in ('run_head', 'prepared_parent'))
         and re.fullmatch(r'[0-9a-f]{64}', t['result_sha256'])
         and set(t['artifacts']) == {'models', 'producer', 'receipts', 'consumed'},
         'complete original training execution and all four artifact pins')
    need(t['run_branch'] == 'research/en-v6-b22-union-retention-train-continuation-once-20261124'
         and t['runtime_workflow'] == '.github/workflows/english-v6-b22-union-retention-train-continuation-once.yml',
         'only the separately reviewed first-fit continuation after the preserved zero-forward STOP')
    for pin in t['artifacts'].values():
        need(set(pin) == {'id', 'bytes', 'sha256', 'name'} and type(pin['id']) is int and pin['id'] > 0
             and type(pin['bytes']) is int and pin['bytes'] > 0
             and re.fullmatch(r'[0-9a-f]{64}', pin['sha256']), 'immutable original training artifact identity')
    expected = read_plan()['training']
    hashes = t['model_pt_sha256']
    need(set(hashes) == {'B0', 'B9', 'control', 'challenger'}
         and hashes['B0'] == expected['shared_parent_pt_sha256']
         and hashes['control'] == expected['control_pt_sha256']
         and all(re.fullmatch(r'[0-9a-f]{64}', v) for v in hashes.values()),
         'complete original model identities, unchanged B0 and historical B21 control')
    return value


def training_binding_sha():
    training_binding()
    return digest(INPUTS.read_bytes())


def transport_plan_sha():
    from english_v6_b21_fixed_mirror_transport import PLAN, plan
    plan()
    return digest(PLAN.read_bytes())


def validate_identity(identity):
    need(set(identity) == {'run_id', 'run_head', 'run_attempt', 'run_branch', 'prepared_parent'}
         and re.fullmatch(r'[1-9][0-9]*', identity['run_id'] or '')
         and all(re.fullmatch(r'[0-9a-f]{40}', identity[k] or '') for k in ('run_head', 'prepared_parent'))
         and identity['run_head'] != identity['prepared_parent'] and identity['run_attempt'] == 1
         and identity['run_branch'] == BRANCH, 'one actual first public B22 primary source execution')
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=REPO,
         GITHUB_REF=BRANCH, GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1').items()),
         'only the separately registered first public source push')
    identity = validate_identity(dict(run_id=e.get('GITHUB_RUN_ID', ''), run_head=e.get('GITHUB_SHA', ''),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None:
        event = json.loads(Path(e['GITHUB_EVENT_PATH']).read_bytes())
    if parent is None:
        parent = subprocess.check_output(['git', '-C', e['GITHUB_WORKSPACE'], 'rev-parse', 'HEAD^1'], text=True).strip()
    need(event.get('before') == parent == prepared_parent and event.get('after') == identity['run_head']
         and event.get('ref') == BRANCH and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPO and event['repository'].get('private') is False,
         'exact reviewed predecessor, no retry or forced/private execution')
    return identity


def gh(endpoint):
    path = endpoint if endpoint.startswith('repos/' + REPO + '/') else 'repos/' + REPO + '/' + endpoint
    return subprocess.run(['gh', 'api', path, '--allow-escape-sequences'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def validate_artifact(artifact, pin, run_id, head):
    need(artifact['id'] == pin['id'] and artifact['name'] == pin['name'] and artifact['expired'] is False
         and artifact['size_in_bytes'] == pin['bytes'] and artifact['digest'] == 'sha256:' + pin['sha256']
         and artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == head,
         'exact original public artifact, never a replacement ZIP')


def validate_training_live(live):
    t = training_binding()['training']
    run, artifacts = live['run'], live['artifacts']
    need(run['id'] == t['run_id'] and run['head_sha'] == t['run_head'] and run['run_attempt'] == 1
         and run['event'] == 'push' and run['head_branch'] == t['run_branch']
         and run['path'] == t['runtime_workflow']
         and run['status'] == 'completed' and run['conclusion'] == 'success'
         and all(run[k]['full_name'] == REPO and run[k]['private'] is False for k in ('repository', 'head_repository')),
         'B22 sources remain closed until the original public training actually completes')
    need(artifacts['total_count'] == len(artifacts['artifacts']) == 4
         and {a['id'] for a in artifacts['artifacts']} == {p['id'] for p in t['artifacts'].values()},
         'complete original models, producer, receipts and consumed inventory')
    indexed = {a['id']: a for a in artifacts['artifacts']}
    for pin in t['artifacts'].values():
        validate_artifact(indexed[pin['id']], pin, t['run_id'], t['run_head'])
    return t


def verify_training(directory):
    directory = Path(directory)
    t = validate_training_live(json.loads((directory / 'TRAIN_LIVE.json').read_bytes()))
    for name, pin in t['artifacts'].items():
        raw = (directory / ('TRAIN_' + name + '.zip')).read_bytes()
        need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact saved training archive ' + name)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            need(archive.testzip() is None and all(not Path(n).is_absolute() and '..' not in Path(n).parts
                 for n in archive.namelist()), 'complete safe original training archive')
    with zipfile.ZipFile(directory / 'TRAIN_producer.zip') as archive:
        need(archive.read('PRODUCER.json') == TRAIN_PRODUCER.read_bytes(),
             'successful original producer exactly matches the source capture dependency inventory')
        manifest = json.loads(archive.read('PRODUCER.json'))
        need(manifest['source_code_reads_only'] is True and manifest['new_model_forwards'] == 0
             and manifest['optimizer_updates'] == 0 and manifest['new_source_calls'] == 0,
             'original training producer was captured before scientific execution')
        for name, pin in manifest['files'].items():
            raw = archive.read('repository/' + name)
            need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256']
                 and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == pin['git_blob_sha1'],
                 'all original training producer bytes and Git blobs')
        need(archive.read('repository/automation/english_v6_b22_panels/assessment.json') == ASSESSMENT.read_bytes(),
             'actual training captured the same unmodified prospective assessment')
    with zipfile.ZipFile(directory / 'TRAIN_receipts.zip') as receipts, zipfile.ZipFile(directory / 'TRAIN_consumed.zip') as consumed:
        need(receipts.read('CONSUMPTION.json') == consumed.read('CONSUMPTION.json'),
             'original before-fit consumed receipt retained identically in completed training')
    with zipfile.ZipFile(directory / 'TRAIN_models.zip') as archive:
        raw = archive.read('RESULT.json')
        result = json.loads(raw)
        identity = result['identity']
        need(digest(raw) == t['result_sha256']
             and result['status'] == 'TRAINED_UNION_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY'
             and identity == dict(run_id=str(t['run_id']), run_head=t['run_head'], run_attempt=1,
                 prepared_parent=t['prepared_parent'], run_branch='refs/heads/' + t['run_branch'])
             and result['recipe_sha256'] == result['design_sha256'] == metadata.DESIGN_SHA
             and result['assessment_sha256'] == ASSESSMENT_SHA and result['model_pt_sha256'] == t['model_pt_sha256']
             and result['actual_optimizer_updates'] == 192 and result['actual_new_model_forwards'] == 1847
             and result['original_retention_conditions'] == 120 and result['union_retention_conditions'] == 122
             and result['reused_ordinary_teacher_arrays'] == 264
             and all(result[k] == 0 for k in ('new_control_optimizer_updates', 'new_control_forwards',
                 'new_teacher_forwards', 'dev_cal_locked_forwards', 'new_tts_asr_calls'))
             and result['promotion_allowed'] is result['historical_numerical_replay_claimed'] is False,
             'actual single192-step TRAIN-only candidate and immutable historical control')
        for key, path in (('source_sha256', 'SOURCE.json'), ('common_draw_sha256', 'COMMON_DRAW.json'),
                          ('union_evidence_sha256', 'UNION_EVIDENCE.json'), ('old120_references_sha256', 'OLD120_POSITIVE_REFERENCES.json')):
            need(digest(archive.read(path)) == result[key], 'complete saved training result dependency ' + path)
        binding = json.loads(archive.read('ASSESSMENT_BINDING.json'))
        need(binding['assessment'] == {'path': 'automation/english_v6_b22_panels/assessment.json', 'sha256': ASSESSMENT_SHA}
             and digest(archive.read('ASSESSMENT_BINDING.json')) == result['assessment_binding_sha256'],
             'actual training opened only the prospectively frozen assessment')
        for role in ('control', 'challenger'):
            for name, pin in result[role]['files'].items():
                data = archive.read(role + '/models/' + name)
                need(len(data) == pin['bytes'] and digest(data) == pin['sha256'], 'all three original model formats')
                if name.endswith('.pt'):
                    need(pin['sha256'] == t['model_pt_sha256'][role], 'original frozen PT identity')
    return result


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    t = training_binding()['training']
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B22 source authorization')
    out.mkdir()
    live = dict(run=json.loads(gh('actions/runs/' + str(t['run_id']))),
                artifacts=json.loads(gh('actions/runs/' + str(t['run_id']) + '/artifacts?per_page=100')))
    validate_training_live(live)
    qa.record(out / 'TRAIN_LIVE.json', live)
    for name, pin in t['artifacts'].items():
        raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'original TRAIN evidence download ' + name)
        (out / ('TRAIN_' + name + '.zip')).write_bytes(raw)
    verify_training(out)
    history = read_ancestry()
    qa.record(out / 'HISTORY.json', history)
    qa.record(out / 'AUTHORIZATION.json', dict(schema=SCHEMA, source_execution=identity,
        assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        training_binding_sha256=training_binding_sha(), train_result_sha256=t['result_sha256'],
        train_live_sha256=qa.sha(out / 'TRAIN_LIVE.json'), history_sha256=qa.sha(out / 'HISTORY.json'),
        transport_plan_sha256=transport_plan_sha(), maximum_tts_calls=96, maximum_natural_decodes=24,
        maximum_native_observer_calls=600, replication_audio_opened=False, training_allowed=False,
        promotion_allowed=False, new_kws_forwards=0))
    return require_authorization(out)


def require_authorization(directory):
    directory = Path(directory)
    t = training_binding()['training']
    verify_training(directory)
    validate_history(json.loads((directory / 'HISTORY.json').read_bytes()))
    value = json.loads((directory / 'AUTHORIZATION.json').read_bytes())
    identity = validate_identity(value['source_execution'])
    need(value == dict(schema=SCHEMA, source_execution=identity, assessment_sha256=ASSESSMENT_SHA,
        ancestry_sha256=ANCESTRY_SHA, training_binding_sha256=training_binding_sha(),
        train_result_sha256=t['result_sha256'], train_live_sha256=qa.sha(directory / 'TRAIN_LIVE.json'),
        history_sha256=qa.sha(directory / 'HISTORY.json'), transport_plan_sha256=transport_plan_sha(),
        maximum_tts_calls=96, maximum_natural_decodes=24, maximum_native_observer_calls=600,
        replication_audio_opened=False, training_allowed=False, promotion_allowed=False, new_kws_forwards=0),
        'complete actual training proof and unchanged primary source budget')
    return value


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'registered native provider only')
    return MARKER if provider is None else 'english-v6-b22-primary-observer-' + provider + '-consumed'


def one_execution(identity, marker_present, provider=None):
    branch = BRANCH.removeprefix('refs/heads/')
    runs = json.loads(gh('actions/workflows/' + Path(WORKFLOW).name + '/runs?branch=' + branch + '&per_page=100'))
    need(runs['total_count'] == len(runs['workflow_runs']) == 1
         and str(runs['workflow_runs'][0]['id']) == identity['run_id']
         and runs['workflow_runs'][0]['head_sha'] == identity['run_head'] and runs['workflow_runs'][0]['run_attempt'] == 1,
         'exactly one primary source execution including failed or expired attempts')
    name = marker_name(provider)
    artifacts = json.loads(gh('actions/artifacts?name=' + name + '&per_page=100'))
    need(artifacts['total_count'] == len(artifacts['artifacts']) == int(marker_present), 'single consumed budget ' + name)
    if not marker_present:
        return None
    artifact = artifacts['artifacts'][0]
    need(artifact['name'] == name and artifact['expired'] is False
         and str(artifact['workflow_run']['id']) == identity['run_id']
         and artifact['workflow_run']['head_sha'] == identity['run_head'], 'same original source owns consumed budget')
    return artifact


def consumption_document(directory):
    from english_v6_b22_source_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), corpus_bytes=CORPUS_BYTES,
        corpus_sha256=CORPUS_SHA, transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, corpus, prepared_parent):
    from english_v6_b22_source_transport import verify_transport
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same new primary source identity')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    path = Path(directory) / 'CONSUMPTION.json'
    need(not path.exists(), 'single original source budget')
    value = consumption_document(directory)
    qa.record(path, value)
    return value


def uploaded_marker(artifact, expected, member):
    raw = gh('actions/artifacts/' + str(artifact['id']) + '/zip')
    result = dict(artifact=artifact, original_zip_base64=base64.b64encode(raw).decode(), receipt=expected)
    validate_uploaded_marker(result, expected, member)
    return result


def validate_uploaded_marker(value, expected, member):
    artifact = value['artifact']
    raw = base64.b64decode(value['original_zip_base64'], validate=True)
    identity = expected['source_execution']
    need(artifact['id'] > 0 and artifact['name'] == expected['marker_name'] and artifact['expired'] is False
         and str(artifact['workflow_run']['id']) == identity['run_id']
         and artifact['workflow_run']['head_sha'] == identity['run_head']
         and len(raw) == artifact['size_in_bytes'] and 'sha256:' + digest(raw) == artifact['digest']
         and value['receipt'] == expected, 'original durably uploaded budget ZIP')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(archive.namelist() == [member] and archive.testzip() is None
             and json.loads(archive.read(member)) == expected, 'uploaded budget exactly contains expected receipt')
    return expected


def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b22_source_transport import verify_transport
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


def require_saved_consumption(directory):
    directory = Path(directory)
    expected = consumption_document(directory)
    need(json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected, 'saved exact source consumption')
    validate_uploaded_marker(json.loads((directory / 'CONSUMPTION_UPLOAD.json').read_bytes()), expected, 'CONSUMPTION.json')
    return expected


def observer_document(provider, registration):
    need(provider in PROVIDERS, 'registered observer')
    identity = validate_identity({key: registration[key] for key in ('run_id', 'run_head', 'run_attempt', 'run_branch', 'prepared_parent')})
    need(96 <= registration['source_rows'] == registration['max_calls'] <= 120, 'all and only eligible originals')
    return dict(schema=SCHEMA + '-observer-consumption', marker_name=marker_name(provider), provider=provider,
        source_execution=identity, source_manifest_sha256=registration['source_manifest_sha256'],
        registration=registration, status='OBSERVER_BUDGET_CONSUMED_UNKNOWN', maximum_native_calls=registration['source_rows'],
        new_tts_calls=0, new_natural_decodes=0, new_kws_forwards=0, retry_count=0)


def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b22_source as owner
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


def require_observer_permit(provider, registration):
    identity = execution_guard(registration['prepared_parent'])
    need(all(registration[k] == value for k, value in identity.items()), 'same source observer identity')
    one_execution(identity, True)
    return uploaded_marker(one_execution(identity, True, provider), observer_document(provider, registration), 'OBSERVER_CONSUMPTION.json')


def validate_observer_budget(value, provider, registration):
    return validate_uploaded_marker(value, observer_document(provider, registration), 'OBSERVER_CONSUMPTION.json')


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_saved_consumption(directory)
    names = ('AUTHORIZATION.json', 'HISTORY.json', 'CONSUMPTION.json', 'CONSUMPTION_UPLOAD.json',
             'CORPUS_TRANSPORT.json', 'TRAIN_LIVE.json', *( 'TRAIN_' + k + '.zip' for k in training_binding()['training']['artifacts']))
    for name in names:
        need(not (output / name).exists(), 'fresh retained original source evidence member')
        shutil.copyfile(directory / name, output / name)
    shutil.copytree(directory / 'TRANSPORT', output / 'TRANSPORT')
