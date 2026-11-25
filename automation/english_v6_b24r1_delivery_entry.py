"""Read the reviewed public pipeline checkout identity; never select a model."""
import argparse
import json
from pathlib import Path
import re

SPEC = Path(__file__).resolve().with_name('english_v6_b24r1_delivery_inputs.json')
SCHEMA = 'english-v6-b24r1-software-delivery-entry-v1'


def need(ok, reason):
    if not ok: raise ValueError(reason)


def contract(value):
    need(type(value) is dict and set(value) == {'schema', 'enabled', 'pipeline_repository',
         'pipeline_commit', 'manifest_sha256', 'pipeline_files'} and value['schema'] == SCHEMA
         and value['pipeline_repository'] == 'jiying2007/kws-pipeline'
         and type(value['enabled']) is bool, 'exact reviewed public delivery entry')
    if value['enabled'] is False:
        need(all(value[key] is None for key in ('pipeline_commit', 'manifest_sha256', 'pipeline_files')),
             'inert entry has no invented candidate authority')
    else:
        need(type(value['pipeline_commit']) is str and re.fullmatch(r'[a-f0-9]{40}', value['pipeline_commit'])
             and type(value['manifest_sha256']) is str and re.fullmatch(r'[a-f0-9]{64}', value['manifest_sha256'])
             and type(value['pipeline_files']) is dict and bool(value['pipeline_files']),
             'exact future reviewed pipeline commit and file inventory')
        for name, pin in value['pipeline_files'].items():
            need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_./-]+', name)
                 and not name.startswith('/') and '..' not in Path(name).parts
                 and type(pin) is dict and set(pin) == {'bytes', 'sha256', 'git_blob_sha1', 'mode'}
                 and type(pin['bytes']) is int and 0 < pin['bytes'] <= 4_000_000
                 and type(pin['sha256']) is str and re.fullmatch(r'[a-f0-9]{64}', pin['sha256'])
                 and type(pin['git_blob_sha1']) is str and re.fullmatch(r'[a-f0-9]{40}', pin['git_blob_sha1'])
                 and pin['mode'] in ('100644', '100755'), 'complete regular pipeline producer member')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'pipeline-commit'))
    args = parser.parse_args()
    value = contract(json.loads(SPEC.read_bytes()))
    if args.mode == 'pipeline-commit':
        need(value['enabled'] is True, 'INERT: no reviewed selected pipeline checkout')
        print('pipeline_commit=' + value['pipeline_commit'])
    else:
        print(json.dumps(dict(status='INERT' if value['enabled'] is False else 'REVIEWED_PIPELINE_BOUND',
            model_forwards=0, packs_created=0, installation_started=False), sort_keys=True))


if __name__ == '__main__': main()
