"""Bind one B24R1 source entry to the complete original B24 SOURCE STOP.

This is a read-only original-evidence operation. It writes one new opening
manifest in a requested output directory, never source audio or model input.
The old failed source and its reserve remain permanently closed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24r1_b24_source_failure as failure
import english_v6_b24_source_ancestry as train


def bind(directory, lock_path, output):
    lock = failure.document(Path(lock_path).read_bytes())
    failure.validate_lock(lock)
    original = failure.read_binding(False)
    failure.need(original['enabled'] is False, 'binding is created once from the inert registration')
    report = failure.verify(Path(directory), lock)
    failure.need(report['candidate_pt_sha256'] == train.training_binding()['training']['model_pt_sha256']['challenger']
                 and report['original_train_result_sha256'] == train.training_binding()['training']['result_sha256'],
                 'same original trained B24 candidate and original TRAIN result')
    value = dict(original, enabled=True, original_source_lock=lock,
                 expected_failure_report_sha256=failure.sha(failure.canonical(report)))
    output = Path(output)
    failure.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), 'fresh binder output only')
    output.mkdir()
    target = output / failure.BINDING.name
    target.write_bytes(failure.encoded(value))
    failure.read_binding(path=target)
    (output / 'SOURCE_STOP_READBACK.json').write_bytes(failure.canonical(report))
    audit = dict(schema='english-v6-b24r1-source-stop-binding-audit-v1',
        source_failure_binding_sha256=failure.sha(target.read_bytes()),
        original_lock_sha256=failure.sha(failure.canonical(lock)),
        original_stop_report_sha256=value['expected_failure_report_sha256'],
        original_source_stop_permanent=True, original_artifacts_reopened=14,
        new_source_calls=0, new_observer_calls=0, new_kws_forwards=0, optimizer_updates=0,
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
