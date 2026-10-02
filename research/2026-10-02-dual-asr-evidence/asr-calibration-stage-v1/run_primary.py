#!/usr/bin/env python3
"""Execute an already acquired/qualified, hash-frozen local primary run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from asr_stage.execution import execute_primary


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate plan key')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--decoder-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = Path(args.plan).read_bytes()
    if len(raw) > 512*1024 or hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise ValueError('Plan bytes differ from independently frozen identity')
    plan = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite plan JSON')))
    decoder = Path(args.decoder_manifest).read_bytes()
    result = execute_primary(plan, args.output, decoder)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
