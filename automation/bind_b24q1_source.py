"""Bind exactly one approved B24Q1 registration to the complete original R1 STOP.

This standard-library original-evidence operation never creates audio, native
observations or KWS forwards. The approved policy applies only to new Q1 audio.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24q1_r1_source_failure as failure
import english_v6_b24_source_ancestry as train


def bind(directory, lock_path, output):
    lock = failure.document(Path(lock_path).read_bytes())
    failure.validate_lock(lock)
    inert = failure.read_binding(False)
    failure.need(inert['enabled'] is False, 'bind only the one new inert approved registration')
    # Metadata-only verification never reads a new native result or KWS score.
    import english_v6_b24q1_source_metadata as metadata
    metadata.validate()
    import english_v6_b24q1_r1_git_transport as transport
    transport.read_binding()
    assessment_raw = (failure.ROOT / failure.ASSESSMENT_PATH).read_bytes()
    report = failure.verify(Path(directory), lock)
    training = train.training_binding()['training']
    failure.need(report['original_candidate_pt_sha256'] == training['model_pt_sha256']['challenger']
                 and report['original_train_result_sha256'] == training['result_sha256'],
                 'same completed original B24 TRAIN result and fixed candidate weights')
    value = dict(inert, enabled=True, original_source_lock=lock,
        expected_failure_report_sha256=failure.sha(failure.canonical(report)),
        historical_git_snapshot_transport=dict(path=transport.BINDING.relative_to(failure.ROOT).as_posix(),
                                               sha256=failure.sha(transport.BINDING.read_bytes())),
        assessment=dict(path=failure.ASSESSMENT_PATH, sha256=failure.sha(assessment_raw)))
    output = Path(output)
    failure.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh binding output only')
    output.mkdir()
    target = output / failure.BINDING.name
    target.write_bytes(failure.encoded(value))
    failure.read_binding(path=target)
    (output / 'SOURCE_STOP_READBACK.json').write_bytes(failure.canonical(report))
    audit = dict(schema='english-v6-b24q1-approved-source-binding-audit-v1',
        source_failure_binding_sha256=failure.sha(target.read_bytes()),
        approval_sha256=failure.APPROVAL_SHA, approved_proposal_sha256=failure.PROPOSAL_SHA,
        approved_design_sha256=failure.DESIGN_SHA, assessment_sha256=failure.sha(assessment_raw),
        original_lock_sha256=failure.sha(failure.canonical(lock)),
        original_stop_report_sha256=value['expected_failure_report_sha256'],
        original_reader_sha256=failure.READER_SHA, original_artifacts_reopened=14,
        old_B24_and_R1_stops_permanent=True, old_reserves_opened=False,
        new_source_calls=0, new_observer_calls=0, new_KWS_forwards=0, new_optimizer_updates=0,
        workflow_activation_performed=False, promotion_allowed=False)
    (output / 'AUDIT.json').write_bytes(failure.encoded(audit))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(bind(args.directory, args.lock, args.output), sort_keys=True))
