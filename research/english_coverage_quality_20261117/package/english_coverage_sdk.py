"""Explicit inference entry point for a manifest-bound coverage research SDK."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def load_pinned(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    for name in ('bundle', 'wav', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if not args.verify_only and not args.execute:
        print(json.dumps({'status': 'SOURCE_CONTRACT_ONLY', 'model_calls': 0, 'execution_enabled': False}))
        return 0
    if args.bundle is None:
        raise ValueError('EXPLICIT_BUNDLE_REQUIRED')
    bundle = args.bundle.absolute()
    if Path(__file__).resolve().parent != bundle.resolve():
        raise ValueError('ENTRY_POINT_MUST_BELONG_TO_BUNDLE')
    package = load_pinned(bundle / 'english_coverage_package.py', '_coverage_package_contract')
    receipt = package.verify_bundle(bundle)
    original = load_pinned(bundle / 'english_final_embedding_infer.py', '_coverage_original_inference')
    original.verify_bundle(bundle)
    if args.verify_only:
        print(json.dumps(receipt, sort_keys=True))
    else:
        if args.wav is None or args.output is None:
            raise ValueError('EXPLICIT_WAV_AND_NEW_OUTPUT_REQUIRED')
        print(json.dumps(original.execute(bundle, args.wav, args.output), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
