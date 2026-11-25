"""Fresh B24R1 primary sources after authenticated warm-start training.

Scientific source generation and five-observer admission retain the exact B22
function bodies. Explicit B24R1 metadata, identities and authorization replace
only the experiment bindings. No prior source or reserve is reused.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_source as original
import english_v6_b22_source_metadata as piper_metadata
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24r1_source_ancestry as lineage
import english_v6_b24r1_source_observe as observers
import english_v6_b24r1_natural_sources as natural

qa, B13 = lineage.qa, original.B13
PLANS, ASSESSMENT, ASSESSMENT_SHA = lineage.PLANS, lineage.ASSESSMENT, lineage.ASSESSMENT_SHA
SCHEMA = 'english-v6-b24r1-fresh-combined-source-v1'
BATCH = 'english-v6-b24r1-120-original-strict-source-screen-v1'
FILES = {'primary': 'af074144cdb88435e659ec57d37e0d50bc14b4ed75469fd0c1a70e41bebda82d',
         'replication': '76a2d98f9f7e0f114f50f23ed3373a1bf9ef5f0cdbc9107aada5a975dd065eed'}
PIPER_CONFIG = piper_metadata.PANELS / 'piper-config.json'
original_source_binding = source_binding


def source_binding():
    from english_v6_b24_source_training_proof import SCOPE
    return dict(original_source_binding(), **SCOPE, **lineage.stop_scope())


def validate_config(path):
    B13.b12.verify_config(path)
    qa.need(qa.sha(Path(path)) == piper_metadata.INPUTS['piper-config.json'],
            'same complete original Piper map frozen before all B24R1 source draws')
    mapping = json.loads(Path(path).read_bytes())['speaker_id_map']
    assigned, excluded = set(), set(lineage.read_ancestry()['synthetic_excluded_people'])
    for role in ('primary', 'replication'):
        for index, person in lineage.read_plan()[role]['synthetic']['speaker_id_map'].items():
            canonical = piper_metadata.canonical(person)
            qa.need(mapping.get(person) == int(index) and canonical not in excluded
                    and canonical not in assigned, 'fresh canonical voice; every prior reserve stays excluded')
            assigned.add(canonical)
    qa.need(len(assigned) == 48, 'all48 new independently frozen synthetic identities')


def check():
    lineage.read_plan(); source_plan(); source_plan('replication'); code_contract()
    validate_config(PIPER_CONFIG)
    lineage.training_binding(False)
    lineage.failure.read_binding(False)
    print('PASS: B24R1 frozen sources; no TTS, decode, observer or KWS calls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('check')
    item = modes.add_parser('guard'); item.add_argument('prepared_parent')
    item = modes.add_parser('authorize'); item.add_argument('out', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('verify-corpus'); item.add_argument('corpus', type=Path)
    for mode in ('transport', 'consume'):
        item = modes.add_parser(mode); item.add_argument('corpus', type=Path); item.add_argument('archives', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('verify-source'); item.add_argument('source', type=Path)
    item = modes.add_parser('generate'); item.add_argument('out', type=Path); item.add_argument('archives', type=Path)
    item.add_argument('corpus', type=Path); item.add_argument('prepared_parent')
    for mode in ('consume-observer', 'observe'):
        item = modes.add_parser(mode); item.add_argument('provider', choices=observers.PROVIDERS)
        item.add_argument('source', type=Path); item.add_argument('out', type=Path); item.add_argument('prepared_parent')
    item = modes.add_parser('screen'); item.add_argument('source', type=Path); item.add_argument('providers', type=Path); item.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.mode == 'check': check()
    elif args.mode == 'guard': lineage.execution_guard(args.prepared_parent)
    elif args.mode == 'authorize': lineage.authorize(args.out, args.prepared_parent)
    elif args.mode == 'verify-corpus': natural.verify_corpus(args.corpus)
    elif args.mode == 'transport':
        from english_v6_b24r1_source_transport import download
        download(args.corpus, args.archives, args.prepared_parent)
    elif args.mode == 'consume': lineage.consumption(args.archives, args.corpus, args.prepared_parent)
    elif args.mode == 'verify-source': registration(args.source)
    elif args.mode == 'generate': generate(args.out, args.archives, args.corpus, args.prepared_parent)
    elif args.mode == 'consume-observer': lineage.consume_observer(args.provider, args.source, args.out, args.prepared_parent)
    elif args.mode == 'observe': observe(args.provider, args.source, args.out, args.prepared_parent)
    else: screen(args.source, args.providers, args.out)
