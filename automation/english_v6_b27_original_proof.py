"""Fetch only explicitly pinned full original proof; no scientific operation.

Complete original API bodies are preserved as exact base64 bytes in the reviewed
binding. Every ZIP is downloaded from its fixed original artifact ID and hashed
before it can enter a proof scope. Large old evidence is never nested in a new
SOURCE or producer archive.
"""
from __future__ import annotations
import argparse
import base64
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_profile as profile


def fetch_files(directory, files, locations, reuse=None):
    profile.validate_transport(locations, files)
    directory = Path(directory)
    profile.need(directory.is_absolute() and directory == directory.resolve()
                 and directory.parent.is_dir() and not directory.exists(), 'fresh complete original proof root')
    if reuse is not None:
        reuse = Path(reuse)
        profile.need(profile.inventory(reuse) == files, 'explicit complete reuse root matches every original byte')
    directory.mkdir()
    for name, pin in sorted(files.items()):
        location = locations[name]
        if reuse is not None:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(reuse / name, target)
            continue
        elif name.endswith('.json'):
            raw = base64.b64decode(location['original_api_base64'], validate=True)
        else:
            raw = subprocess.run(['gh', 'api', 'repos/jiying2007/kws-data/actions/artifacts/'
                + str(location['artifact_id']) + '/zip', '--allow-escape-sequences'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        profile.need(len(raw) == pin['bytes'] and profile.sha(raw) == pin['sha256'],
                     'exact entire external original ' + name)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    profile.need(profile.inventory(directory) == files, 'all and only complete original proof bytes')
    if reuse is not None:
        profile.need(profile.inventory(reuse) == files, 'complete reused originals unchanged after regular hardlinking')


def fetch(directory, panel, training_reuse=None, primary_reuse=None):
    bound = profile.binding(panel)
    directory = Path(directory)
    profile.need(directory.is_absolute() and directory == directory.resolve()
                 and directory.parent.is_dir() and not directory.exists(), 'fresh once-only proof acquisition')
    directory.mkdir()
    fetch_files(directory / 'training-origin', bound['files'], bound['transport'], training_reuse)
    if panel == 'replication':
        fetch_files(directory / 'primary-quality', bound['primary_quality_files'], bound['primary_quality_transport'], primary_reuse)
    else:
        profile.need(primary_reuse is None, 'no primary quality reused to open a new PRIMARY')
    with profile.proof_scope(directory, panel):
        return profile.require_proof(panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', choices=('primary', 'replication'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = fetch(args.directory, args.panel)
    print(profile.canonical(dict(schema='english-v6-b27-complete-proof-fetch-v1',
        panel=args.panel, training_report_sha256=profile.sha(profile.canonical(report)),
        new_model_calls=0, new_source_calls=0)).decode())
