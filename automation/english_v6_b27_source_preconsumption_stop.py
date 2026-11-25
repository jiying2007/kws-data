"""Read the exact B27 SOURCE engineering stop before any scientific consumption.

This is a standalone, stdlib-only reader of original saved evidence. It never
fetches, imports captured code, constructs a model, or authorizes a new run.
The original incomplete producer remains incomplete. Complete TRAIN proof is
still independently required by the actual B27 SOURCE execution.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

SCHEMA = 'english-v6-b27-original-preconsumption-source-stop-v1'
FAILURE_DIR = Path(__file__).with_name('english_v6_b27_source_preconsumption_failure')
REPO = 'jiying2007/kws-data'
RUN = 45589429827
HEAD = '8c8577b5f480583870290a00ee0bc8ebc485b6ec'
PREPARED = 'dc395bb20afdcad34ea8e8b926951f9cb11b6d90'
TREE = '25688426d27f44aa530f35e4610d88b278d32c4d'
BRANCH = 'research/en-v6-b27-primary-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-primary-sources-once.yml'
TEMPLATE = 'automation/english_v6_b27_source_workflow_template.yml'
SOURCE_JOB = 139120307749
MISSING_RESOURCE = 'automation/english_v6_b21_fixed_mirror_transport.json'
WORKFLOW_SHA = '6a0cdbd70fef7280d78ede7521e3f5f91ab98095275c207570805b62500595ab'
TEMPLATE_SHA = '635adb6fa355dab440d694f6e65102ab00ca852e2a119b6fd7b4a78d8fd27aa9'
MANIFEST_SHA = 'f9b1cd9e23054730d59014b3188be2bf5a4d4e7ff5c981d20b24f9b70fca5dc8'
PROOF_BINDING_SHA = '6c6334a1d8d6452e186c41fe472eca6e072a8af42a02a81d9aac68b30e7d9a28'
HISTORY_SHA = '79e1181eb36fcb9b4a70438e0ceae9ca1fd34649e1db91c269abcd4722834a5a'
PROFILE_SHA = 'aed01e7b3f19066e001190e89e37858a3ec93251ffef09c2d2bf5e66c0481d22'
# These are the original connector-decoded bytes, not a synthesized job log.
LOG_BYTES = 35422
LOG_SHA = '8b6513cceb4a64258c9193f2b8f2a24890cb2fb5ed8fde06f2782b7a697da736'
PRODUCER = dict(id=15009453589, bytes=5955174,
    sha256='a648e5302a592cea612607728bc796e0a7d0495f3716a96f518710a8637fe77f',
    name='english-v6-b27-primary-source-producer-45589429827')
TRANSPORT = dict(id=15010614539, bytes=531171,
    sha256='022113db4dd99e883a0be51404cd016d0b2eedb691bfba1fbf87a90e8ec22b0d',
    name='english-v6-b27-primary-transport-45589429827')
FILES = frozenset(('run.json', 'jobs.json', 'artifacts.json', 'commit.json',
    'tree.json', 'comparison.json', 'producer.zip', 'transport.zip', 'source-job.log'))
CHECKOUT = 'Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'
PYTHON = 'Run actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1'
UPLOAD = 'Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02'
CONTRACT_STEPS = (
    (1, 'Set up job', 'success'), (2, CHECKOUT, 'success'), (3, PYTHON, 'success'),
    (4, 'Offline approval, whole-original quarantine and complete dependency contracts', 'success'),
    (7, 'Post ' + PYTHON, 'success'), (8, 'Post ' + CHECKOUT, 'success'),
    (9, 'Complete job', 'success'))
SOURCE_STEPS = (
    (1, 'Set up job', 'success'), (2, CHECKOUT, 'success'), (3, PYTHON, 'success'),
    (4, 'Capture exact Git-bound scientific producer', 'success'), (5, UPLOAD, 'success'),
    (6, 'Reopen every original B27 TRAIN and complete Q1 failure before new source permission', 'success'),
    (7, 'Authenticate the complete original candidate and separately frozen B27 source population', 'failure'),
    (8, 'Fetch the complete identical corpus over eight fixed official magicdatatech ranges', 'skipped'),
    (9, UPLOAD, 'success'),
    (10, 'Consume TTS96, natural24 and exact five-times-N observer budget after complete corpus and capacity verification', 'skipped'),
    (11, UPLOAD, 'skipped'),
    (12, 'Generate only the new frozen96 synthetic originals and decode only the committed24 whole natural originals', 'skipped'),
    (13, UPLOAD, 'success'), (25, 'Post ' + PYTHON, 'skipped'),
    (26, 'Post ' + CHECKOUT, 'success'), (27, 'Complete job', 'success'))
JOB_IDENTITIES = dict(contracts=(139119389797, 'success', CONTRACT_STEPS),
    sources=(SOURCE_JOB, 'failure', SOURCE_STEPS),
    observers=(139129956063, 'skipped', ()), readback=(139129956388, 'skipped', ()))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_object(kind, raw):
    return hashlib.sha1(kind.encode('ascii') + b' ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def document(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    def nonfinite(_):
        raise ValueError('nonfinite JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    need(type(value) is dict, 'explicit original JSON document')
    return value


def safe_path(name):
    need(type(name) is str and name and '\\' not in name and '\0' not in name,
         'regular relative original member')
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.as_posix() == name
         and all(part not in ('', '.', '..', '.git') for part in name.split('/')),
         'normalized relative original member')
    return path


def validate_remote(run, jobs, artifacts):
    """Validate complete live or saved API objects; perform no network operation."""
    need(type(run) is type(jobs) is type(artifacts) is dict, 'complete original API objects')
    need(run.get('id') == RUN and run.get('head_sha') == HEAD
         and type(run.get('run_attempt')) is int and run['run_attempt'] == 1
         and run.get('status') == 'completed' and run.get('conclusion') == 'failure'
         and run.get('event') == 'push' and run.get('head_branch') == BRANCH
         and run.get('path') == WORKFLOW
         and all(run.get(key, {}).get('full_name') == REPO
                 and run.get(key, {}).get('private') is False
                 for key in ('repository', 'head_repository'))
         and run.get('head_commit', {}).get('id') == HEAD
         and run.get('head_commit', {}).get('tree_id') == TREE,
         'only the exact completed original public first-attempt B27 engineering stop')
    rows = jobs.get('jobs', [])
    need(type(rows) is list and jobs.get('total_count') == len(rows) == 4
         and all(type(job) is dict for job in rows), 'complete original four-job graph')
    indexed = {job.get('name'): job for job in rows}
    need(len(indexed) == 4 and set(indexed) == set(JOB_IDENTITIES), 'no omitted, added or duplicate job')
    for name, (ident, conclusion, expected_steps) in JOB_IDENTITIES.items():
        job = indexed[name]
        need(job.get('id') == ident and job.get('run_id') == RUN
             and job.get('run_attempt') == 1 and job.get('head_sha') == HEAD
             and job.get('head_branch') == BRANCH and job.get('status') == 'completed'
             and job.get('conclusion') == conclusion, 'exact original terminal job ' + name)
        steps = job.get('steps')
        need(type(steps) is list and all(type(step) is dict for step in steps)
             and tuple((step.get('number'), step.get('name'), step.get('conclusion')) for step in steps) == expected_steps
             and all(step.get('status') == 'completed' for step in steps),
             'exact steps; transport, consumption, generation, observers and readback never entered')
    rows = artifacts.get('artifacts', [])
    need(type(rows) is list and artifacts.get('total_count') == len(rows) == 2,
         'only original producer and partial authorization artifacts; any consumption artifact forbids continuation')
    pins = {pin['id']: pin for pin in (PRODUCER, TRANSPORT)}
    need(all(type(item) is dict for item in rows) and {item.get('id') for item in rows} == set(pins),
         'exact two original artifact identities')
    for item in rows:
        pin = pins[item['id']]
        execution = item.get('workflow_run', {})
        need(item.get('name') == pin['name'] and item.get('expired') is False
             and item.get('size_in_bytes') == pin['bytes'] and item.get('digest') == 'sha256:' + pin['sha256']
             and execution.get('id') == RUN and execution.get('head_sha') == HEAD
             and execution.get('head_branch') == BRANCH,
             'unchanged complete original archive metadata')
    return dict(original_run_id=RUN, original_run_head=HEAD, source_job_id=SOURCE_JOB,
                run_attempt=1, complete_jobs=4, complete_artifacts=2,
                no_scientific_consumption_or_source_artifact=True,
                corpus_transport_and_all_scientific_stages_skipped=True)


def validate_tree(tree):
    """Reconstruct every Git tree object instead of trusting a declared root SHA."""
    need(tree.get('sha') == TREE and tree.get('truncated') is False
         and type(tree.get('tree')) is list, 'complete original recursive Git tree')
    entries, children = {}, {}
    kinds = {'040000': 'tree', '100644': 'blob', '100755': 'blob', '120000': 'blob', '160000': 'commit'}
    for entry in tree['tree']:
        need(type(entry) is dict, 'original Git tree entry')
        path = safe_path(entry.get('path'))
        name = path.as_posix()
        need(name not in entries and entry.get('mode') in kinds
             and entry.get('type') == kinds[entry['mode']]
             and type(entry.get('sha')) is str and re.fullmatch('[0-9a-f]{40}', entry['sha']),
             'unique original Git tree member and valid object identity')
        entries[name] = entry
        parent = '' if path.parent.as_posix() == '.' else path.parent.as_posix()
        children.setdefault(parent, []).append(entry)
    for name in entries:
        parent = str(PurePosixPath(name).parent)
        need(parent == '.' or parent in entries and entries[parent]['type'] == 'tree',
             'every complete original Git parent directory retained')
    directories = {''} | {name for name, item in entries.items() if item['type'] == 'tree'}
    for parent in directories:
        rows = children.get(parent, [])
        rows = sorted(rows, key=lambda entry: (PurePosixPath(entry['path']).name
                                               + ('/' if entry['type'] == 'tree' else '')).encode('utf-8'))
        body = b''.join(entry['mode'].lstrip('0').encode('ascii') + b' '
            + PurePosixPath(entry['path']).name.encode('utf-8') + b'\0' + bytes.fromhex(entry['sha']) for entry in rows)
        expected = TREE if not parent else entries[parent]['sha']
        need(git_object('tree', body) == expected, 'complete original Git tree bytes ' + (parent or '/'))
    return entries


def validate_git(commit, comparison, tree):
    need(commit.get('sha') == HEAD and commit.get('tree', {}).get('sha') == TREE
         and [item.get('sha') for item in commit.get('parents', [])] == [PREPARED],
         'exact original prepared parent and source Git tree')
    need(comparison.get('status') == 'ahead'
         and comparison.get('ahead_by') == comparison.get('total_commits') == 1
         and comparison.get('behind_by') == 0
         and comparison.get('base_commit', {}).get('sha') == PREPARED
         and comparison.get('merge_base_commit', {}).get('sha') == PREPARED
         and [item.get('sha') for item in comparison.get('commits', [])] == [HEAD]
         and [(item.get('filename'), item.get('status')) for item in comparison.get('files', [])] == [(WORKFLOW, 'modified')],
         'only the exact original workflow activation changed after its reviewed parent')
    return validate_tree(tree)


def original_archive(raw, pin):
    need(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'original whole ZIP identity')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    rows = archive.infolist()
    need(len(rows) == len({row.filename for row in rows}) and len(rows) <= 500,
         'unique bounded original archive members')
    total = 0
    for row in rows:
        safe_path(row.filename)
        mode = (row.external_attr >> 16) & 0xffff
        need(not row.is_dir() and not row.flag_bits & 1
             and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), 'regular unencrypted original ZIP member')
        total += row.file_size
        need(0 <= row.file_size <= 128_000_000 and total <= 128_000_000,
             'original complete archive expansion stays inside fixed core capacity')
    need(archive.testzip() is None, 'whole original ZIP CRC validation')
    return archive


def verify(directory=FAILURE_DIR):
    """Return the canonical zero-consumption finding for these exact originals."""
    directory = Path(directory)
    need(directory.is_absolute() and directory.is_dir()
         and not any(path.is_symlink() for path in (directory, *directory.parents)), 'regular absolute evidence root')
    found = {}
    for path in directory.rglob('*'):
        mode = path.lstat().st_mode
        need(stat.S_ISREG(mode), 'all evidence members are regular files with no extra directory or symlink')
        found[path.relative_to(directory).as_posix()] = path
    need(set(found) == FILES, 'all and only complete nine-file original stop evidence')
    payloads = {name: path.read_bytes() for name, path in found.items()}
    docs = {name: document(payloads[name + '.json']) for name in
            ('run', 'jobs', 'artifacts', 'commit', 'comparison', 'tree')}
    remote = validate_remote(docs['run'], docs['jobs'], docs['artifacts'])
    entries = validate_git(docs['commit'], docs['comparison'], docs['tree'])
    with original_archive(payloads['producer.zip'], PRODUCER) as archive:
        manifest_raw = archive.read('PRODUCER.json')
        need(len(manifest_raw) == 126003 and sha(manifest_raw) == MANIFEST_SHA, 'original incomplete producer manifest')
        manifest = document(manifest_raw)
        need(manifest.get('schema') == 'english-v6-b27-complete-source-producer-v1'
             and manifest.get('panel') == 'primary' and manifest.get('run_id') == str(RUN)
             and manifest.get('run_head') == HEAD and manifest.get('run_attempt') == 1
             and manifest.get('run_branch') == 'refs/heads/' + BRANCH
             and manifest.get('prepared_parent') == PREPARED and manifest.get('runtime_workflow') == WORKFLOW
             and manifest.get('git_tree') == TREE and len(manifest.get('files', {})) == 439
             and manifest.get('include_hidden_files') is manifest.get('runtime_workflow_preserved_hidden') is True
             and manifest.get('source_code_reads_only') is True
             and all(manifest.get(key) == 0 for key in ('new_source_calls', 'new_model_forwards', 'optimizer_updates'))
             and manifest.get('promotion_allowed') is False, 'exact original pre-science 439-file producer')
        files = manifest['files']
        need(set(archive.namelist()) == {'PRODUCER.json'} | {'repository/' + name for name in files}
             and set(manifest['python_files']) == {name for name in files if name.endswith('.py')}
             and set(manifest['resource_files']) == {name for name in files if not name.endswith('.py')},
             'all original producer files retained without supplement or omission')
        for name, pin in files.items():
            safe_path(name)
            body = archive.read('repository/' + name)
            entry = entries.get(name, {})
            need(entry.get('type') == 'blob' and entry.get('mode') == pin.get('mode') in ('100644', '100755')
                 and entry.get('sha') == pin.get('git_blob_sha1') == git_object('blob', body)
                 and len(body) == pin.get('bytes') and sha(body) == pin.get('sha256'), 'original Git-bound producer bytes ' + name)
        need(MISSING_RESOURCE not in files and entries.get(MISSING_RESOURCE, {}).get('type') == 'blob',
             'preserve the original capture omission; do not supplement its artifact retrospectively')
        workflow = archive.read('repository/' + WORKFLOW)
        template = archive.read('repository/' + TEMPLATE)
        token = b'__B27_REVIEWED_PRIMARY_SOURCE_PARENT__'
        need(sha(workflow) == WORKFLOW_SHA and sha(template) == TEMPLATE_SHA
             and template.count(token) == 2 and workflow == template.replace(token, PREPARED.encode('ascii')),
             'the exact original two-token workflow-only activation')
        binding_raw = archive.read('repository/automation/english_v6_b27_primary_proof_binding.json')
        history_raw = archive.read('repository/automation/english_v6_b27_panels/ancestry.json')
        need(sha(binding_raw) == manifest['external_proof_binding_sha256'] == PROOF_BINDING_SHA
             and sha(history_raw) == HISTORY_SHA
             and sha(archive.read('repository/automation/english_v6_b27_execution_profile.json'))
                 == manifest['execution_profile_sha256'] == PROFILE_SHA,
             'same unchanged original TRAIN binding, history and prospective execution profile')
    with original_archive(payloads['transport.zip'], TRANSPORT) as archive:
        need(set(archive.namelist()) == {'HISTORY.json', 'PROOF_BINDING.json'},
             'only the two original pre-authorization documents; no authorization, transport or consumption')
        need(archive.read('HISTORY.json') == history_raw and archive.read('PROOF_BINDING.json') == binding_raw,
             'two saved partial-authorization files equal their original producer bytes')
    raw_log = payloads['source-job.log']
    need(len(raw_log) == LOG_BYTES and sha(raw_log) == LOG_SHA, 'exact original connector-decoded source log')
    log = raw_log.decode('utf-8', errors='strict')
    stop = "FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/_temp/b27-source-producer/repository/" + MISSING_RESOURCE + "'"
    failed = '##[error]Process completed with exit code 1.'
    absent = 'No files were found with the provided path: /home/runner/work/_temp/source/. No artifacts will be uploaded.'
    need(log.count(stop) == log.count(failed) == log.count(absent) == 1
         and log.index(stop) < log.index(failed) < log.index(absent),
         'exact authorization-time missing resource exception and absent source output')
    return dict(schema=SCHEMA, status='VERIFIED_ORIGINAL_PRECONSUMPTION_ENGINEERING_STOP',
        **remote, prepared_parent=PREPARED, git_tree=TREE, workflow_sha256=WORKFLOW_SHA,
        original_producer=PRODUCER, original_transport=TRANSPORT,
        original_producer_manifest_sha256=MANIFEST_SHA, original_producer_files=439,
        original_source_log_sha256=LOG_SHA, missing_original_capture_resource=MISSING_RESOURCE,
        original_transport_members=['HISTORY.json', 'PROOF_BINDING.json'],
        original_proof_binding_sha256=PROOF_BINDING_SHA, original_history_sha256=HISTORY_SHA,
        original_execution_profile_sha256=PROFILE_SHA,
        source_authorization_created=False, consumption_started=False, source_generation_started=False,
        actual_new_tts_calls=0, actual_new_natural_original_decodes=0,
        actual_new_observer_calls=0, actual_new_kws_forwards=0, actual_new_optimizer_updates=0,
        original_train_preflight_step_succeeded=True, original_train_proof_reverified_by_this_reader=False,
        source_gate_created=False, model_quality_verdict=None, original_failure_preserved=True,
        new_scientific_execution_authorized=False, new_assessment_authorized=False,
        promotion_allowed=False, fresh_live_zero_consumption_confirmation_required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=FAILURE_DIR)
    args = parser.parse_args()
    print(canonical(verify(args.directory.resolve())).decode('ascii'))
