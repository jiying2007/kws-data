"""Bind the explicitly approved B24Q1 assessment to the immutable R1 STOP.

The original R1 reader and its complete, canonical report are unchanged. This
adapter collects original evidence and binds the separate user approval; it
does not reinterpret the stopped R1 observations under the new empty-return
policy. No scientific call, old-source continuation or automatic next panel
is granted by this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24r1_source_stop_readback as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BINDING = HERE / 'english_v6_b24q1_source_failure_inputs.json'
READER_PATH = Path(original.__file__).resolve()
READER_SHA = 'ac3e12810f72f68c3bdd4328dc5933b71c3e05da00524b508f91ce870b83d83d'
REPORT_SHA = 'e3ca7baa272609fc8a4b304fb56b52bd98c8a85a504dd2f4ab0fab298b777ffd'
LOCK_SHA = '56c10ef6a669650050fd1d821871489c030f13f224fc85d1c9dc537c69ef37f8'
APPROVAL_PATH = 'research/english_v6_b24q1_preparation_20261125/APPROVAL.json'
APPROVAL_SHA = '6b3a9303874606b17045ff1ee0b78f7a08123c9a1cdc80442056451110b3b13b'
PROPOSAL_PATH = 'research/english_v6_b24q1_preparation_20261125/APPROVED_PROPOSAL_SOURCE.json'
PROPOSAL_SHA = 'b5149d29049d5f308efd92a966ab36b0ccb81a9e04fbc64f8ff81e52d5fe7bea'
DESIGN_PATH = 'automation/english_v6_b24q1_evaluation_design.json'
DESIGN_SHA = 'f3d4079a685f65eddecafce7afa909150c6136bc842ce116279888d2b556c66d'
CONTRACT_PATH = 'automation/english_v6_b24q1_orchestration_contract.json'
CONTRACT_SHA = '75136b683c893e3d6457d93b2f9f74829c711e7b614b6a5b727380faf4f42f49'
ASSESSMENT_PATH = 'automation/english_v6_b24q1_panels/assessment.json'
ASSESSMENT_SHA = '681ddb42432bd11dfef0157d2f1c95f7e02824c3a2ed7da64ce9a12bdf9c3139'
SCHEMA = 'english-v6-b24q1-approved-r1-stop-opening-v1'
CONDITION = 'one_explicitly_approved_B24Q1_assessment_after_complete_original_B24R1_SOURCE_STOP'

REPO, RUN, HEAD = original.REPO, original.RUN, original.HEAD
REPORT_SCHEMA, FALSE_PERMISSIONS = original.REPORT_SCHEMA, original.FALSE_PERMISSIONS
CANDIDATE_PT, TRAIN_ASSESSMENT = original.CANDIDATE_PT, original.TRAIN_ASSESSMENT
need, sha, pin, canonical, document = original.need, original.sha, original.pin, original.canonical, original.document
DENIED = ('new_training_allowed', 'old_source_retry_allowed', 'old_source_continuation_allowed',
    'open_old_sealed_reserve', 'automatic_next_assessment_allowed', 'old_B25_or_B26_openers_changed')


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('ascii')


def exact_file(relative, digest, root=ROOT):
    path = original.regular(Path(root) / relative)
    raw = path.read_bytes()
    need(sha(raw) == digest, 'exact approved registration resource: ' + relative)
    return raw


def validate_lock(value):
    original.validate_lock(value)
    need(sha(canonical(value)) == LOCK_SHA, 'unchanged complete original R1 STOP lock')
    return value


def approval_resources(root=ROOT):
    """The immutable original proposal and the later approval are separate."""
    exact_file(PROPOSAL_PATH, PROPOSAL_SHA, root)
    approval = document(exact_file(APPROVAL_PATH, APPROVAL_SHA, root))
    design = document(exact_file(DESIGN_PATH, DESIGN_SHA, root))
    exact_file(CONTRACT_PATH, CONTRACT_SHA, root)
    need(approval['schema'] == 'english-v6-b24q1-explicit-user-approval-v1'
         and approval['approved'] is True and approval['experiment_id'] == 'B24Q1'
         and approval['user_statement'] == '批准提案，继续一次全新评估（推荐）'
         and approval['approved_proposal']['sha256'] == PROPOSAL_SHA
         and approval['scope']['maximum_new_assessments'] == 1
         and approval['scope']['successful_native_empty_orchestration_amendment_approved'] is True
         and approval['scope']['new_training_allowed'] is False
         and approval['scope']['automatic_R2_or_next_panel_allowed'] is False,
         'one explicit user-approved fresh fixed-weight assessment')
    need(design['experiment_id'] == 'B24Q1' and design['evaluation_only'] is True
         and design['frozen_model_pt_sha256'] == approval['scope']['frozen_model_pt_sha256']
         and design['approval'] == dict(path=APPROVAL_PATH, sha256=APPROVAL_SHA)
         and design['approved_proposal'] == dict(path=PROPOSAL_PATH, sha256=PROPOSAL_SHA)
         and design['orchestration_contract'] == dict(path=CONTRACT_PATH, sha256=CONTRACT_SHA),
         'exact separately approved new design, original weights and quarantine policy')
    need(sha(original.regular(READER_PATH).read_bytes()) == READER_SHA,
         'the original R1 STOP reader is immutable')
    return approval, design


def read_binding(require_enabled=True, path=None):
    value = document(original.regular(path or BINDING).read_bytes())
    need(type(value) is dict and set(value) == {
        'schema', 'enabled', 'condition', 'maximum_new_assessments', 'candidate_pt_sha256',
        'original_train_assessment_sha256', 'approval', 'approved_proposal', 'evaluation_design',
        'orchestration_contract', 'assessment', 'original_source_lock',
        'expected_failure_report_sha256', 'immutable_original_reader_sha256',
        'historical_git_snapshot_transport', *DENIED},
        'complete explicit-approval and original-STOP opening')
    need(value['schema'] == SCHEMA and value['condition'] == CONDITION
         and type(value['maximum_new_assessments']) is int and value['maximum_new_assessments'] == 1
         and value['candidate_pt_sha256'] == CANDIDATE_PT
         and value['original_train_assessment_sha256'] == TRAIN_ASSESSMENT
         and value['immutable_original_reader_sha256'] == READER_SHA
         and all(value[key] is False for key in DENIED),
         'only B24Q1; no training, old continuation, reclaimed reserve or automatic next assessment')
    for key, relative, digest in (
        ('approval', APPROVAL_PATH, APPROVAL_SHA), ('approved_proposal', PROPOSAL_PATH, PROPOSAL_SHA),
        ('evaluation_design', DESIGN_PATH, DESIGN_SHA), ('orchestration_contract', CONTRACT_PATH, CONTRACT_SHA)):
        need(value[key] == dict(path=relative, sha256=digest), 'fixed opening resource ' + key)
    approval_resources()
    if value['enabled'] is False:
        need(not require_enabled and value['original_source_lock'] is None
             and value['expected_failure_report_sha256'] is None and value['assessment'] is None
             and value['historical_git_snapshot_transport'] is None,
             'inert opening cannot fabricate an original report or a frozen assessment')
    else:
        need(value['enabled'] is True and value['expected_failure_report_sha256'] == REPORT_SHA,
             'the complete original corrected R1 STOP report must already be independently authenticated')
        validate_lock(value['original_source_lock'])
        assessment = value['assessment']
        need(assessment == dict(path=ASSESSMENT_PATH, sha256=ASSESSMENT_SHA),
             'one prospectively frozen new assessment')
        exact_file(ASSESSMENT_PATH, assessment['sha256'])
        import english_v6_b24q1_r1_git_transport as transport
        transport.read_binding()
        need(value['historical_git_snapshot_transport'] == dict(
            path=transport.BINDING.relative_to(ROOT).as_posix(), sha256=sha(transport.BINDING.read_bytes())),
            'same separately authenticated transport for unchanged historical Git snapshots')
    return value


def verify(directory, lock):
    """Run the unchanged isolated reader, then require its exact original report."""
    validate_lock(lock)
    need(sha(original.regular(READER_PATH).read_bytes()) == READER_SHA, 'unchanged original R1 reader')
    raw = verify_in_scope(directory, lock, lambda: canonical(original.verify(Path(directory), lock)))
    report = document(raw)
    need(sha(canonical(report)) == REPORT_SHA and report['reader_sha256'] == READER_SHA
         and report['original_candidate_pt_sha256'] == CANDIDATE_PT
         and report['original_training_assessment_sha256'] == TRAIN_ASSESSMENT
         and all(report[key] is False for key in FALSE_PERMISSIONS),
         'complete original failure proof remains fail-only and byte-identical')
    return report


def verify_in_scope(directory, lock, execute):
    """Reauthenticate identical old bytes once within one independent audit.

    Every invocation still reads and hashes every complete original file and
    the immutable reader. A fresh outer scope must execute the complete old
    reader; only its exact canonical report may be reused in that same scope.
    """
    import english_v6_b24q1_evidence_scope as evidence_scope
    validate_lock(lock)
    files = {role + '.zip': {k: value[k] for k in ('bytes', 'sha256')}
             for role, value in lock['artifacts'].items()}
    files.update({'metadata/source_' + role + '.json': dict(value) for role, value in lock['api'].items()})
    files.update({value['path']: {k: value[k] for k in ('bytes', 'sha256')}
                  for value in lock['logs'].values()})
    return evidence_scope.verify_once('r1_source_stop', Path(directory), files, canonical(lock),
        READER_PATH, READER_SHA, REPORT_SHA, execute)


def gh(endpoint):
    return subprocess.run(['gh', 'api', 'repos/' + REPO + '/' + endpoint, '--allow-escape-sequences'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def require_pin(raw, expected, role):
    need(type(raw) is bytes and pin(raw) == {key: expected[key] for key in ('bytes', 'sha256')},
         'exact unchanged original response bytes: ' + role)
    return raw


def live_metadata(lock, git_pins=None):
    """Seven read-only responses; no artifact, corpus, observer or model call.

    The three run APIs and two logs match the original saved byte pins. The
    immutable Git objects currently use a different JSON serialization; their
    actual raw pins are independently fixed by the read-only transport check.
    Bodies are never reserialized to make a historical byte pin match.
    """
    validate_lock(lock)
    if git_pins is None:
        import english_v6_b24q1_r1_git_transport as transport
        git_pins = transport.read_binding()['live_git']
    result = {}
    for name, endpoint in (
        ('run', 'actions/runs/' + str(RUN)),
        ('jobs', 'actions/runs/' + str(RUN) + '/jobs?per_page=100'),
        ('artifacts', 'actions/runs/' + str(RUN) + '/artifacts?per_page=100'),
        ('commit', 'commits/' + HEAD)):
        expected = git_pins[name] if name == 'commit' else lock['api'][name]
        result['metadata/source_' + name + '.json'] = require_pin(gh(endpoint), expected, name)
    commit = document(result['metadata/source_commit.json'])
    need(commit['sha'] == HEAD, 'the original commit chooses the original Git tree')
    result['metadata/source_tree.json'] = require_pin(
        gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1'), git_pins['tree'], 'tree')
    for role, expected in lock['logs'].items():
        result[expected['path']] = require_pin(gh('actions/jobs/' + str(expected['job_id']) + '/logs'), expected, role)
    return result


def fetch(directory, lock):
    """Collect and fully reauthenticate all fourteen original R1 ZIPs."""
    validate_lock(lock)
    directory = Path(directory).resolve()
    need(not directory.exists() and directory.parent.is_dir(), 'fresh complete original R1 evidence directory')
    metadata = live_metadata(lock)
    import english_v6_b24q1_r1_git_transport as transport
    historical = transport.fetch(directory.parent, {
        role: metadata['metadata/source_' + role + '.json'] for role in ('commit', 'tree')})
    # These are the pre-existing historical bytes, not a reserialization of
    # the current API body. Both originals are retained in the new package.
    for role in ('commit', 'tree'):
        metadata['metadata/source_' + role + '.json'] = require_pin(historical[role], lock['api'][role], role)
    directory.mkdir()
    (directory / 'metadata').mkdir()
    for name, raw in metadata.items():
        (directory / name).write_bytes(raw)
    for role, expected in lock['artifacts'].items():
        raw = require_pin(gh('actions/artifacts/' + str(expected['id']) + '/zip'), expected, role)
        (directory / (role + '.zip')).write_bytes(raw)
    return verify(directory, lock)


def verify_transport(directory):
    import english_v6_b24q1_r1_git_transport as transport
    directory = Path(directory)
    historical = transport.verify_transport(directory)
    for role in ('commit', 'tree'):
        need((directory / 'ORIGINAL_B24R1_STOP/metadata' / ('source_' + role + '.json')).read_bytes() == historical[role],
             'unchanged historical Git snapshot is the exact original-reader input: ' + role)
    return historical


def retain_transport(directory, output):
    import english_v6_b24q1_r1_git_transport as transport
    verify_transport(directory)
    transport.retain_transport(directory, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'check-live-metadata'))
    args = parser.parse_args()
    value = read_binding()
    result = dict(schema=SCHEMA + '-preflight', original_reader_sha256=READER_SHA,
        approved_assessment_sha256=value['assessment']['sha256'], new_source_calls=0,
        new_observer_calls=0, new_KWS_forwards=0, new_optimizer_updates=0,
        scientific_execution_authorized=False, original_failure_status_unchanged=True)
    if args.mode == 'check-live-metadata':
        responses = live_metadata(value['original_source_lock'])
        # Exercise the actual gh transport before any scientific SOURCE run.
        # This reads only the separately registered two-JSON metadata archive;
        # no original audio/model archive or scientific budget is opened.
        import english_v6_b24q1_r1_git_transport as transport
        with tempfile.TemporaryDirectory(prefix='b24q1-snapshot-transport-preflight-') as temporary:
            historical = transport.fetch(Path(temporary), {
                role: responses['metadata/source_' + role + '.json'] for role in ('commit', 'tree')})
            need(set(historical) == {'commit', 'tree'}, 'both complete historical metadata snapshots')
        result.update(exact_current_responses=len(responses),
            exact_historical_raw_matches=5, historical_git_snapshots_separately_transported=2,
            additional_metadata_transport_origin_responses=4,
            additional_metadata_snapshot_archives_reopened=1,
            additional_metadata_transport_independently_verified=True,
            exact_current_response_bytes=sum(map(len, responses.values())),
            all14_original_archives_reopened=False, complete_failure_readback_performed=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
