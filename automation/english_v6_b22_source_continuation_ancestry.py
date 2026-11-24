"""First text-only continuation of the frozen B22 source execution.

Original audio and the two completed phone observers are immutable inputs.
Only the three text observers proven to have made zero calls receive new budgets.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import english_v6_b22_source_ancestry as original
import english_v6_b22_source_origin as origin

qa, need, digest, canonical = original.qa, original.need, original.digest, original.canonical
PLANS, ASSESSMENT, ASSESSMENT_SHA = original.PLANS, original.ASSESSMENT, original.ASSESSMENT_SHA
ANCESTRY_SHA, metadata, REPO = original.ANCESTRY_SHA, original.metadata, original.REPO
read_plan, read_ancestry, validate_history = original.read_plan, original.read_ancestry, original.validate_history
training_binding, training_binding_sha = original.training_binding, original.training_binding_sha
transport_plan_sha, gh = original.transport_plan_sha, original.gh
uploaded_marker, validate_uploaded_marker = original.uploaded_marker, original.validate_uploaded_marker
BRANCH = 'refs/heads/research/en-v6-b22-primary-source-continuation-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b22-primary-source-continuation-once.yml'
SCHEMA = 'english-v6-b22-source-text-continuation-v1'
MARKER = 'english-v6-b22-primary-source-text-continuation-consumed'
PROVIDERS = ('qwen', 'whisper', 'parakeet')
PHONES = ('xlsr', 'lv60')
ORIGIN_DIRECTORY = 'CONTINUATION_ORIGIN'
ORIGIN_BINDING = Path(__file__).with_name('english_v6_b22_source_origin.json')
ORIGINAL_ARTIFACT_IDS = dict(source=14923907567, producer=14923562621,
    qwen=14924367664, parakeet=14924088037, whisper=14923418714,
    xlsr=14923518739, lv60=14924203180)
ADDED_SOURCE_FILES = {'CONTINUATION.json', 'CONTINUATION_CONSUMPTION.json',
                      'CONTINUATION_CONSUMPTION_UPLOAD.json'}


def origin_binding_sha():
    return digest(ORIGIN_BINDING.read_bytes())


def validate_identity(identity):
    need(set(identity) == {'run_id', 'run_head', 'run_attempt', 'run_branch', 'prepared_parent'}
         and re.fullmatch(r'[1-9][0-9]*', identity['run_id'] or '')
         and all(re.fullmatch(r'[0-9a-f]{40}', identity[k] or '') for k in ('run_head', 'prepared_parent'))
         and identity['run_head'] != identity['prepared_parent'] and identity['run_attempt'] == 1
         and identity['run_branch'] == BRANCH, 'one separately reviewed first text continuation')
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=REPO,
         GITHUB_REF=BRANCH, GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1').items()),
         'only the registered first public text-continuation push')
    identity = validate_identity(dict(run_id=e.get('GITHUB_RUN_ID', ''), run_head=e.get('GITHUB_SHA', ''),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None:
        event = json.loads(Path(e['GITHUB_EVENT_PATH']).read_bytes())
    if parent is None:
        parent = subprocess.check_output(['git', '-C', e['GITHUB_WORKSPACE'], 'rev-parse', 'HEAD^1'], text=True).strip()
    need(event.get('before') == parent == prepared_parent and event.get('after') == identity['run_head']
         and event.get('ref') == BRANCH and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPO and event['repository'].get('private') is False,
         'exact reviewed continuation predecessor, no retry, force or private run')
    return identity


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'only the three original zero-call text observers')
    return MARKER if provider is None else 'english-v6-b22-primary-text-continuation-' + provider + '-consumed'


def one_execution(identity, marker_present, provider=None):
    branch = BRANCH.removeprefix('refs/heads/')
    runs = json.loads(gh('actions/workflows/' + Path(WORKFLOW).name + '/runs?branch=' + branch + '&per_page=100'))
    need(runs['total_count'] == len(runs['workflow_runs']) == 1
         and str(runs['workflow_runs'][0]['id']) == identity['run_id']
         and runs['workflow_runs'][0]['head_sha'] == identity['run_head'] and runs['workflow_runs'][0]['run_attempt'] == 1,
         'exactly one continuation execution including failed attempts')
    name = marker_name(provider)
    artifacts = json.loads(gh('actions/artifacts?name=' + name + '&per_page=100'))
    need(artifacts['total_count'] == len(artifacts['artifacts']) == int(marker_present), 'single consumed budget ' + name)
    if not marker_present:
        return None
    artifact = artifacts['artifacts'][0]
    need(artifact['name'] == name and artifact['expired'] is False
         and str(artifact['workflow_run']['id']) == identity['run_id']
         and artifact['workflow_run']['head_sha'] == identity['run_head'], 'same continuation owns exact budget')
    return artifact


def verify_origin(directory):
    value = origin.verify_origin(Path(directory))
    need(value['original_source_count'] == 120
         and value['phone_forward_counts'] == dict.fromkeys(PHONES, 120)
         and value['text_forward_counts'] == dict.fromkeys(PROVIDERS, 0)
         and value['remaining_text_forward_calls'] == 360,
         'all120 original audio, all240 phone returns and exactly zero prior text calls')
    original.validate_identity(value['original_source_execution'])
    return value


def archive_files(directory, role):
    need(role in ORIGINAL_ARTIFACT_IDS, 'one of the seven immutable original artifacts')
    path = Path(directory) / ('source-' + str(ORIGINAL_ARTIFACT_IDS[role]) + '.zip')
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos if not i.is_dir()]
        need(z.testzip() is None and len(names) == len(set(names)) and len(names) > 0
             and sum(i.file_size for i in infos) < 128 * 1024 * 1024,
             'bounded original complete artifact without duplicate members')
        for info in infos:
            name = info.filename
            p = Path(name)
            need(not p.is_absolute() and '..' not in p.parts and '\\' not in name
                 and (info.external_attr >> 16) & 0o170000 != 0o120000,
                 'original regular archive member only')
        return {name: z.read(name) for name in names}


def consumption_document(identity, origin_report):
    validate_identity(identity)
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=identity,
        original_source_execution=origin_report['original_source_execution'],
        origin_binding_sha256=origin_binding_sha(), status='TEXT360_CONSUMED_UNKNOWN',
        maximum_new_text_calls=360, maximum_native_calls_per_provider=dict.fromkeys(PROVIDERS, 120),
        maximum_new_tts_calls=0, maximum_new_natural_decodes=0, maximum_new_phone_calls=0,
        new_kws_forwards=0, optimizer_updates=0, retry_count=0)


def fetch_origin(directory, prepared_parent):
    execution_guard(prepared_parent)
    origin.fetch_origin(Path(directory))
    return verify_origin(directory)


def consume(directory, output, prepared_parent):
    identity = execution_guard(prepared_parent)
    report = verify_origin(directory)
    one_execution(identity, False)
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new continuation-only budget directory')
    output.mkdir()
    value = consumption_document(identity, report)
    qa.record(output / 'CONSUMPTION.json', value)
    return value


def continuation_document(identity, report, source_files):
    return dict(schema=SCHEMA, source_execution=validate_identity(identity),
        original_source_execution=report['original_source_execution'],
        original_source_manifest_sha256=digest(source_files['MANIFEST.json']),
        original_artifact_ids=ORIGINAL_ARTIFACT_IDS, origin_binding_sha256=origin_binding_sha(),
        original_source_files={name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in sorted(source_files.items())},
        original_source_count=120, original_text_forward_counts=dict.fromkeys(PROVIDERS, 0),
        reused_phone_forward_counts=dict.fromkeys(PHONES, 120), maximum_new_text_calls=360,
        new_tts_calls=0, new_natural_decodes=0, new_phone_calls=0, new_kws_forwards=0,
        optimizer_updates=0, retry_count=0, promotion_allowed=False, replication_opened=False)


def write_archive(files, output):
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new exact artifact copy destination')
    output.mkdir()
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)


def reuse(directory, source, phones, consumed, prepared_parent):
    identity = execution_guard(prepared_parent)
    directory, source, phones, consumed = map(Path, (directory, source, phones, consumed))
    report = verify_origin(directory)
    expected = consumption_document(identity, report)
    need(json.loads((consumed / 'CONSUMPTION.json').read_bytes()) == expected, 'same360 text-only consumed budget')
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    source_files = archive_files(directory, 'source')
    need(not ADDED_SOURCE_FILES.intersection(source_files)
         and not any(n.startswith(ORIGIN_DIRECTORY + '/') for n in source_files), 'original source has no later continuation')
    write_archive(source_files, source)
    shutil.copytree(directory, source / ORIGIN_DIRECTORY)
    qa.record(source / 'CONTINUATION.json', continuation_document(identity, report, source_files))
    qa.record(source / 'CONTINUATION_CONSUMPTION.json', expected)
    qa.record(source / 'CONTINUATION_CONSUMPTION_UPLOAD.json', receipt)
    need(phones.is_absolute() and phones.parent.is_dir() and not phones.exists(), 'new exact phone-copy root')
    phones.mkdir()
    for provider in PHONES:
        write_archive(archive_files(directory, provider), phones / provider)
    require_continuation(source)


def require_continuation(source):
    source = Path(source)
    report = verify_origin(source / ORIGIN_DIRECTORY)
    original_files = archive_files(source / ORIGIN_DIRECTORY, 'source')
    value = json.loads((source / 'CONTINUATION.json').read_bytes())
    need(value == continuation_document(value['source_execution'], report, original_files),
         'complete original source and actual continuation are separate authenticated identities')
    actual = set()
    for path in source.rglob('*'):
        need(not path.is_symlink(), 'no substituted source symlink')
        if path.is_file():
            name = path.relative_to(source).as_posix()
            if not name.startswith(ORIGIN_DIRECTORY + '/') and name not in ADDED_SOURCE_FILES:
                actual.add(name)
                need(name in original_files and path.read_bytes() == original_files[name], 'original source byte identity: ' + name)
    need(actual == set(original_files), 'all and only original source files plus explicit continuation evidence')
    expected = consumption_document(value['source_execution'], report)
    need(json.loads((source / 'CONTINUATION_CONSUMPTION.json').read_bytes()) == expected,
         'exact retained continuation consumption')
    validate_uploaded_marker(json.loads((source / 'CONTINUATION_CONSUMPTION_UPLOAD.json').read_bytes()), expected, 'CONSUMPTION.json')
    return value


def require_phone_copy(source, provider, directory):
    need(provider in PHONES, 'only original completed phones may be reused')
    expected = archive_files(Path(source) / ORIGIN_DIRECTORY, provider)
    root = Path(directory)
    actual = {}
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'no reused phone symlink')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path.read_bytes()
    need(actual == expected, 'completed original phone raw payloads, identity and consumed permit remain byte-exact')


def observer_document(provider, registration):
    need(provider in PROVIDERS, 'only original zero-call text observers can receive a new budget')
    identity = validate_identity({key: registration[key] for key in ('run_id', 'run_head', 'run_attempt', 'run_branch', 'prepared_parent')})
    need(registration['source_rows'] == registration['max_calls'] == 120
         and registration['origin_binding_sha256'] == origin_binding_sha(), 'same120 originals and immutable zero-call origin')
    return dict(schema=SCHEMA + '-observer-consumption', marker_name=marker_name(provider), provider=provider,
        source_execution=identity, original_source_execution=registration['original_source_execution'],
        source_manifest_sha256=registration['source_manifest_sha256'], registration=registration,
        status='OBSERVER_BUDGET_CONSUMED_UNKNOWN', maximum_native_calls=120,
        new_tts_calls=0, new_natural_decodes=0, new_phone_calls=0, new_kws_forwards=0, retry_count=0)


def consume_observer(provider, source, output, prepared_parent):
    import english_v6_b22_source_continuation as owner
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[k] == v for k, v in identity.items()), 'same continuation before text budget')
    one_execution(identity, True)
    one_execution(identity, False, provider)
    output = Path(output)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'new individual text budget')
    output.mkdir()
    value = observer_document(provider, registration)
    qa.record(output / 'OBSERVER_CONSUMPTION.json', value)
    return value


def require_observer_permit(provider, registration):
    identity = execution_guard(registration['prepared_parent'])
    need(all(registration[k] == v for k, v in identity.items()), 'same continuation observer identity')
    one_execution(identity, True)
    return uploaded_marker(one_execution(identity, True, provider), observer_document(provider, registration), 'OBSERVER_CONSUMPTION.json')


def validate_observer_budget(value, provider, registration):
    return validate_uploaded_marker(value, observer_document(provider, registration), 'OBSERVER_CONSUMPTION.json')
