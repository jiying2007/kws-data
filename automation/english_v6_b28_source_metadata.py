"""Prospective B28 metadata; complete original evidence precedes one selection.

The original B27 validator remains immutable. Complete TRAIN-clean-360 metadata
stays in its three original artifacts outside Git and outside later producers.
No catalog, terminal, or selection identity is invented while a pin is null.
This module has no network, audio, observer, optimizer, or model implementation.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANELS = HERE / 'english_v6_b28_panels'
PREFIX = 'automation/english_v6_b28_panels/'
MANIFEST = PREFIX + 'b27-source-exposure.json'
ORIGIN = PREFIX + 'b27-source-origin.json'
CATALOG_ORIGIN = PREFIX + 'train360-origin.json'
PREVIOUS = 'automation/english_v6_b27_source_metadata.py'
PREVIOUS_SHA = 'f247d4c041729f8c227798040657598345cda209ada4fcad8ae6edc157dae258'
ORIGINAL_ASSESSMENT = dict(path='automation/english_v6_b27_panels/assessment.json', bytes=92465,
    sha256='8c7b4571d2ac08546fd00789fc2809df8c4290ad53a38c1f6b33a993331ae2ee',
    git_blob_sha1='9c014798e8a695ae26fe47a6604a562ed6c25eaa')
ORIGINAL_HISTORY = 'automation/english_v6_b27_panels/ancestry.json'
SOURCE_INPUTS = 'automation/english_v6_b27_primary_inputs.json'
SOURCE_INPUTS_SHA = '66b39d8a0372d658e83a97f66db3edc421b954b242ecc9a40ca560951d904bda'
SOURCE_PIN = dict(id=15019445072, run_id=45597047117,
    run_head='7429fee92456bb04ada5187be04aadbb5b18d826', bytes=10903894,
    sha256='3892e46918861941c24e6c36a544fc05cc65a22be3c61dd2890b671dedf12642')
SOURCE_PRODUCER_PIN = dict(id=15014187560, run_id=45597047117,
    run_head=SOURCE_PIN['run_head'], bytes=16944258,
    sha256='c2b2334ab30ea1ebd12bf2322bb2ca51c4c42e608d0d53340c0727ff1cf6cc8b')
SOURCE_PARENT = 'b37932fbbf8579b7efb3f4789acf3f11ca495e87'
SOURCE_BRANCH = 'research/en-v6-b27-primary-sources-complete-resources-once-20261125'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b27-primary-sources-once.yml'
SOURCE_MEMBERS = {'MANIFEST.json': 'cc684d83f276400cff2c55e83ce3a3fda8f1713f41ee59bc126462b2dfd3dbee',
    'HISTORY.json': '79e1181eb36fcb9b4a70438e0ceae9ca1fd34649e1db91c269abcd4722834a5a'}
DESIGN_PATH = 'automation/english_v6_b28_natural_event_design.json'
OPENING_PATH = 'automation/english_v6_b28_opening.json'
BINDING_PATH = 'automation/english_v6_b28_assessment_binding.json'
COLLECTOR = 'automation/english_v6_train360_metadata.py'
COLLECTOR_SHA = 'b32fef85c966254b08b034704664d4dd04c9139657ff5046513fcea8bd22853e'
CATALOG_FILES = ('api/artifacts.json', 'api/commit.json', 'api/run.json',
    'originals/consumed.zip', 'originals/metadata.zip', 'originals/producer.zip')
CATALOG_MEMBERS = ('FULL_MEMBER_INDEX.json', 'FULL_METADATA_CATALOG.json', 'VALIDATION.json')
CATALOG_MAXIMUM = 224 * 1024 * 1024
OUTPUT_NAMES = ('ancestry.json', 'assessment.json', 'primary.json', 'natural-primary.json',
    'replication.json', 'natural-replication.json')
SALTS = {
    'primary': {'piper': 'english-v6-b28-natural-event-primary-piper-20261125-v1',
                'natural': 'english-v6-b28-natural-event-primary-natural-20261125-v1'},
    'replication': {'piper': 'english-v6-b28-natural-event-replication-piper-20261125-v1',
                    'natural': 'english-v6-b28-natural-event-replication-natural-20261125-v1'},
}
INITIAL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CONTROL_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
B27_PT = 'dd0d4bc08be1162886ab462dfb3ad7f4503150a57ee13272d01f41e80ca1069f'
ORCHESTRATION_SHA = '75136b683c893e3d6457d93b2f9f74829c711e7b614b6a5b727380faf4f42f49'
# Bound only after complete actual originals and the independent B27 terminal.
# Null is a hard refusal, not a wildcard, fallback, or synthetic identity.
DESIGN_SHA = None
OPENING_SHA = None
EXPOSURE_SHA = None
ORIGIN_SHA = None
CATALOG_ORIGIN_SHA = None
_CATALOG = ContextVar('b28_complete_original_train360_metadata', default=None)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def document(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    need(type(value) is dict, 'explicit metadata JSON object')
    return value


def valid_hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def safe_name(name):
    need(type(name) is str and name and '\\' not in name and '\0' not in name
         and not PurePosixPath(name).is_absolute() and str(PurePosixPath(name)) == name
         and all(p not in ('', '.', '..') for p in name.split('/')), 'canonical relative metadata path')


def frozen(name, expected):
    need(valid_hash(expected), 'complete actual metadata/terminal pins remain unbound: ' + name)
    safe_name(name)
    path = ROOT / name
    need(path.is_file() and all(not p.is_symlink() for p in (path, *path.parents)), 'ordinary frozen metadata file')
    raw = path.read_bytes()
    need(digest(raw) == expected, 'exact frozen metadata input: ' + name)
    return raw


def load_module(name, expected, identity):
    frozen(name, expected)
    spec = importlib.util.spec_from_file_location(identity, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _previous():
    return load_module(PREVIOUS, PREVIOUS_SHA, 'b28_unchanged_b27_metadata')


def _collector():
    return load_module(COLLECTOR, COLLECTOR_SHA, 'b28_original_train360_collector')


def original_assessment():
    raw = frozen(ORIGINAL_ASSESSMENT['path'], ORIGINAL_ASSESSMENT['sha256'])
    need(len(raw) == ORIGINAL_ASSESSMENT['bytes']
         and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == ORIGINAL_ASSESSMENT['git_blob_sha1'],
         'whole original B27 assessment and Git blob')
    return document(raw)


def canonical_person(row):
    text = str(next((row[k] for k in ('source_speaker_id', 'speaker', 'speaker_id') if row.get(k) is not None),
        row['source_group_id'].rsplit('-', 1)[-1])).removeprefix('p')
    need(re.fullmatch('[0-9]+', text) is not None, 'canonical original source person')
    return str(int(text))


def original_assessment_people(assessment=None):
    assessment = original_assessment() if assessment is None else assessment
    people = {}
    for role in ('primary', 'replication'):
        synthetic = {str(int(str(v).removeprefix('p'))) for v in assessment[role]['synthetic']['speaker_id_map'].values()}
        natural = {str(int(v)) for v in assessment[role]['natural']['speakers']}
        need(len(synthetic) == 24 and len(natural) == 12 and not synthetic & natural, 'original B27 24+12 distinct people')
        people[role] = synthetic | natural
    need(not people['primary'] & people['replication'], 'all72 original B27 exposed and sealed people')
    return {k: sorted(v, key=int) for k, v in people.items()}


def _source_api_identity(api, pin):
    need(pin == SOURCE_PIN, 'same exact original B27 SOURCE pin')
    run, artifact, commit = (api[k] for k in ('run', 'artifact', 'commit'))
    need(run['id'] == pin['run_id'] and run['head_sha'] == pin['run_head'] == commit['sha']
         and run['run_attempt'] == 1 and run['event'] == 'push'
         and run['status'] == 'completed' and run['conclusion'] == 'success'
         and run['head_branch'] == SOURCE_BRANCH and run['path'] == SOURCE_WORKFLOW
         and run['repository']['full_name'] == 'jiying2007/kws-data' and run['repository']['private'] is False
         and [p['sha'] for p in commit['parents']] == [SOURCE_PARENT]
         and [p['filename'] for p in commit['files']] == [SOURCE_WORKFLOW]
         and artifact['id'] == pin['id'] and artifact['size_in_bytes'] == pin['bytes']
         and artifact['digest'] == 'sha256:' + pin['sha256'] and artifact['expired'] is False
         and artifact['name'] == 'english-v6-b27-primary-source-' + str(pin['run_id'])
         and artifact['workflow_run']['id'] == pin['run_id'] and artifact['workflow_run']['head_sha'] == pin['run_head'],
         'original successful B27 SOURCE first attempt and workflow-only parent')
    return dict(run={**{k: run[k] for k in ('id', 'run_attempt', 'head_sha', 'head_branch', 'event', 'path', 'status', 'conclusion')},
        'repository': {k: run['repository'][k] for k in ('full_name', 'private')}},
        artifact={**{k: artifact[k] for k in ('id', 'name', 'size_in_bytes', 'digest', 'expired')},
            'workflow_run': {k: artifact['workflow_run'][k] for k in ('id', 'head_sha')}},
        commit=dict(sha=commit['sha'], parents=[dict(sha=SOURCE_PARENT)], files=[dict(filename=SOURCE_WORKFLOW)]))


def _source_origin(exposure, api):
    identity = _source_api_identity(api, SOURCE_PIN)
    raw = encoded(exposure)
    return dict(schema='english-v6-b28-authenticated-b27-source-metadata-origin-v1',
        source_artifact=dict(SOURCE_PIN), source_producer_artifact=dict(SOURCE_PRODUCER_PIN),
        original_assessment=dict(ORIGINAL_ASSESSMENT), source_api_identity=identity,
        source_api_identity_sha256=digest(encoded(identity)), original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_exposure=dict(path=MANIFEST, bytes=len(raw), sha256=digest(raw)),
        extracted_member_allowlist=['MANIFEST.json', 'HISTORY.json'], metadata_only=True,
        audio_members_interpreted=0, observer_receipts_read=0, quality_results_read=False,
        model_calls=0, new_source_calls=0, original_SOURCE_run_success=True,
        full_original_B27_NO_PROMOTION_proof_is_separate=True, scientific_execution_authorized=False)


def derive_original_metadata(members, api, source_pin):
    need(set(members) == set(SOURCE_MEMBERS) and all(type(members[n]) is bytes and digest(members[n]) == pin
         for n, pin in SOURCE_MEMBERS.items()), 'exact original B27 MANIFEST and HISTORY bytes')
    identity = _source_api_identity(api, source_pin)
    manifest, history = (document(members[n]) for n in ('MANIFEST.json', 'HISTORY.json'))
    execution = dict(prepared_parent=SOURCE_PARENT, run_attempt=1, run_branch='refs/heads/' + SOURCE_BRANCH,
        run_head=source_pin['run_head'], run_id=str(source_pin['run_id']))
    need(manifest['schema'] == 'english-v6-b27-fresh-combined-source-v1' and manifest['panel'] == 'primary'
         and manifest['selected_original_slots'] == 120 and manifest['source_execution'] == execution
         and manifest['assessment_sha256'] == manifest['original_training_assessment_sha256'] == ORIGINAL_ASSESSMENT['sha256']
         and manifest['source_ancestry_sha256'] == manifest['history_receipt_sha256'] == SOURCE_MEMBERS['HISTORY.json']
         and manifest['new_kws_forwards'] == 0 and manifest['replication_opened'] is False
         and history['all_prior_replication_sealed'] is True, 'complete original B27 SOURCE and inherited history')
    assessment = original_assessment()
    synthetic, natural = manifest['synthetic']['rows'], manifest['natural']
    originals = natural['rows'] + natural['source_rejections']
    by_id = {r['source_id']: r for r in originals}
    roster = assessment['primary']['natural']['roster']
    need(len(synthetic) == 96 and len(originals) == len(by_id) == len(roster) == 24
         and set(by_id) == {r['source_id'] for r in roster}
         and manifest['rows'] == synthetic + natural['rows'], 'all120 originals include every preobserver rejection')
    fields = ['id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person']
    rows = [{**{k: r[k] for k in fields[:-1]}, 'canonical_person': canonical_person(r)}
        for r in synthetic + [by_id[r['source_id']] for r in roster]]
    need(len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == 120
         and len({r['canonical_person'] for r in rows}) == len({r['source_group_id'] for r in rows}) == 36
         and sum(r['cohort'] == 'synthetic' for r in rows) == 96
         and sum(r['cohort'] == 'natural' for r in rows) == 24
         and all(valid_hash(r['pcm_sha256']) for r in rows), 'complete unfiltered original120 exposure')
    people = original_assessment_people(assessment)
    need(set(people['primary']) == {r['canonical_person'] for r in rows}, 'all original36 people match B27 assessment')
    exposure = dict(schema='english-v6-b28-authenticated-b27-exposure-v1', original_source_schema=manifest['schema'],
        panel='primary', selected_original_slots=120, source_artifact=dict(source_pin), source_execution=execution,
        assessment_sha256=ORIGINAL_ASSESSMENT['sha256'], source_ancestry_sha256=SOURCE_MEMBERS['HISTORY.json'],
        history_receipt_sha256=SOURCE_MEMBERS['HISTORY.json'], original_metadata_sha256=dict(SOURCE_MEMBERS),
        new_kws_forwards=0, replication_opened=False, rows=rows, metadata_only=True, selection_fields=fields,
        primary_people=people['primary'], sealed_replication_people=people['replication'],
        all_excluded_people=sorted(set(people['primary']) | set(people['replication']), key=int))
    return {'b27-source-exposure.json': exposure, 'b27-source-origin.json': _source_origin(exposure, identity)}


def archive(path, pin, maximum):
    path = Path(path).absolute()
    need(path.is_file() and all(not p.is_symlink() for p in (path, *path.parents))
         and type(pin['bytes']) is int and path.stat().st_size == pin['bytes'] <= maximum, 'exact ordinary original ZIP bytes')
    with path.open('rb') as stream:
        need(hashlib.file_digest(stream, 'sha256').hexdigest() == pin['sha256'], 'whole original ZIP SHA256')
    result = zipfile.ZipFile(path)
    try:
        infos = result.infolist()
        need(len(infos) <= 8192 and len(infos) == len({i.filename for i in infos})
             and sum(i.file_size for i in infos) <= maximum, 'bounded unique original ZIP members')
        for item in infos:
            safe_name(item.filename.rstrip('/') if item.is_dir() else item.filename)
            need(not stat.S_ISLNK(item.external_attr >> 16), 'no linked original ZIP member')
        return result
    except BaseException:
        result.close()
        raise


def read_original_metadata(source_zip, source_api, source_pin, source_producer_zip=None):
    _source_api_identity(source_api, source_pin)
    with archive(source_zip, source_pin, 256_000_000) as source:
        members = {name: source.read(name) for name in SOURCE_MEMBERS}
    if source_producer_zip is not None:
        with archive(source_producer_zip, SOURCE_PRODUCER_PIN, 128_000_000) as producer:
            raw = producer.read('repository/' + ORIGINAL_ASSESSMENT['path'])
            need(raw == frozen(ORIGINAL_ASSESSMENT['path'], ORIGINAL_ASSESSMENT['sha256']), 'actual captured B27 assessment')
    return derive_original_metadata(members, source_api, source_pin)


def exclusion_summary(exposure, origin):
    return dict(schema='english-v6-b28-original-b27-source-exclusion-v1', source_artifact=dict(SOURCE_PIN),
        source_producer_artifact=dict(SOURCE_PRODUCER_PIN), original_assessment=dict(ORIGINAL_ASSESSMENT),
        original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_files={MANIFEST: digest(encoded(exposure)), ORIGIN: digest(encoded(origin))},
        original_rows=120, primary_people=36, sealed_replication_people=36, all72_people_excluded=True,
        prior_PCM_identities=3063, all_PCM_identities_after_B27=3183,
        source_identity_authenticated=True, metadata_only=True, audio_members_interpreted=0,
        observer_receipts_read=0, quality_results_read=False,
        complete_original_B27_NO_PROMOTION_readback_required_separately=True,
        captured_original_B27_assessment_in_complete_producer_required=True,
        scientific_execution_authorized=False, model_calls=0, new_source_calls=0)


def verify_original_source(source_zip, source_api, source_pin, source_producer_zip=None):
    documents = read_original_metadata(source_zip, source_api, source_pin, source_producer_zip)
    for name, value in documents.items():
        need((PANELS / name).read_bytes() == encoded(value), 'whole original B27 metadata matches frozen reexport: ' + name)
    result = exclusion_summary(documents['b27-source-exposure.json'], documents['b27-source-origin.json'])
    need(result == validate()['original_B27_source_exclusion'], 'same complete original B27 source exclusion')
    return result


def catalog_registration():
    """Small frozen transport declaration only; this does not authenticate text."""
    value = document(frozen(CATALOG_ORIGIN, CATALOG_ORIGIN_SHA))
    need(set(value) == {'schema', 'execution', 'files', 'transport', 'artifacts', 'producer_manifest_sha256',
         'catalog_members', 'metadata_only', 'scientific_execution_authorized'}
         and value['schema'] == 'english-v6-b28-original-train360-catalog-registration-v1'
         and value['metadata_only'] is True and value['scientific_execution_authorized'] is False,
         'explicit metadata-only original catalog registration')
    execution = value['execution']
    need(set(execution) == {'run_id', 'run_head', 'run_branch', 'run_attempt', 'prepared_parent'}
         and re.fullmatch('[1-9][0-9]*', execution['run_id']) is not None and execution['run_attempt'] == 1
         and execution['run_branch'] == 'refs/heads/research/en-v6-train360-metadata-once-20261125'
         and all(re.fullmatch('[0-9a-f]{40}', execution[k]) for k in ('run_head', 'prepared_parent'))
         and execution['run_head'] != execution['prepared_parent'], 'one actually registered metadata run')
    need(set(value['files']) == set(value['transport']) == set(CATALOG_FILES), 'complete six original catalog files')
    for name, pin in value['files'].items():
        need(set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int and 0 < pin['bytes'] <= CATALOG_MAXIMUM
             and valid_hash(pin['sha256']), 'fixed whole external catalog file')
        location = value['transport'][name]
        if name.endswith('.zip'):
            need(set(location) == {'artifact_id'} and type(location['artifact_id']) is int and location['artifact_id'] > 0,
                 'fixed original catalog artifact ID')
        else:
            need(set(location) == {'original_api_base64'} and pin['bytes'] <= 8_000_000, 'bounded original catalog API bytes')
            raw = base64.b64decode(location['original_api_base64'], validate=True)
            need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact retained original catalog API bytes')
    need(sum(p['bytes'] for p in value['files'].values()) <= CATALOG_MAXIMUM
         and set(value['artifacts']) == {'producer', 'consumed', 'metadata'}
         and valid_hash(value['producer_manifest_sha256']) and set(value['catalog_members']) == set(CATALOG_MEMBERS),
         'bounded complete original catalog closure')
    for role, pin in value['artifacts'].items():
        name = 'originals/' + role + '.zip'
        need(set(pin) == {'id', 'run_id', 'run_head', 'bytes', 'sha256'}
             and pin['id'] == value['transport'][name]['artifact_id']
             and str(pin['run_id']) == execution['run_id'] and pin['run_head'] == execution['run_head']
             and {k: pin[k] for k in ('bytes', 'sha256')} == value['files'][name], 'same original catalog artifact')
    for name, pin in value['catalog_members'].items():
        need(set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int and 0 < pin['bytes'] <= 100663296
             and valid_hash(pin['sha256']), 'whole retained original catalog member pin')
    return value


def _catalog_api(api, origin, collector):
    execution = origin['execution']
    run, commit, artifacts = (api[k] for k in ('run', 'commit', 'artifacts'))
    need(str(run['id']) == execution['run_id'] and run['head_sha'] == execution['run_head'] == commit['sha']
         and run['head_branch'] == execution['run_branch'].removeprefix('refs/heads/')
         and run['path'] == collector.WORKFLOW and run['event'] == 'push' and run['run_attempt'] == 1
         and run['status'] == 'completed' and run['conclusion'] == 'success'
         and run['repository']['full_name'] == collector.REPO and run['repository']['private'] is False
         and [p['sha'] for p in commit['parents']] == [execution['prepared_parent']]
         and [p['filename'] for p in commit['files']] == [collector.WORKFLOW]
         and artifacts['total_count'] == len(artifacts['artifacts']) == 3, 'original successful workflow-only metadata run')
    by_id = {row['id']: row for row in artifacts['artifacts']}
    names = {'producer': 'english-v6-train360-metadata-producer-' + execution['run_id'],
        'consumed': collector.MARKER, 'metadata': 'english-v6-train360-metadata-' + execution['run_id']}
    for role, pin in origin['artifacts'].items():
        item = by_id[pin['id']]
        need(item['name'] == names[role] and item['size_in_bytes'] == pin['bytes']
             and item['digest'] == 'sha256:' + pin['sha256'] and item['expired'] is False
             and str(item['workflow_run']['id']) == execution['run_id']
             and item['workflow_run']['head_sha'] == execution['run_head'], 'original catalog artifact API identity')


def _catalog_complete(raw, origin, collector):
    spec = collector.registration()
    need(set(raw) == set(CATALOG_MEMBERS), 'three whole original catalog members')
    for name, payload in raw.items():
        need(type(payload) is bytes and dict(bytes=len(payload), sha256=digest(payload)) == origin['catalog_members'][name],
             'whole original catalog bytes: ' + name)
    catalog, index, validation = (document(raw[n]) for n in ('FULL_METADATA_CATALOG.json', 'FULL_MEMBER_INDEX.json', 'VALIDATION.json'))
    need(len(raw['FULL_METADATA_CATALOG.json']) <= spec['limits']['catalog_bytes']
         and len(raw['FULL_MEMBER_INDEX.json']) <= spec['limits']['member_index_bytes']
         and len(raw['VALIDATION.json']) <= 1048576, 'original per-member catalog capacities')
    summary = collector.validate_catalog(catalog, index, spec, (ROOT / collector.CENSUS).read_bytes())
    need(validation['schema'] == 'english-v6-train360-metadata-validation-v1'
         and validation['status'] == 'COMPLETE_AUTHENTICATED_METADATA' and validation['execution'] == origin['execution']
         and validation['corpus'] == spec['corpus']
         and all(validation['summary'][k] == v for k, v in summary.items())
         and validation['catalog'] == origin['catalog_members']['FULL_METADATA_CATALOG.json']
         and validation['member_index'] == origin['catalog_members']['FULL_MEMBER_INDEX.json']
         and validation['summary']['compressed_bytes'] == spec['corpus']['bytes']
         and validation['summary']['compressed_sha256'] == spec['corpus']['sha256']
         and validation['summary']['compressed_md5'] == spec['corpus']['md5']
         and summary['tar_end_blocks_offset'] + 1024 <= validation['summary']['expanded_tar_bytes'] <= spec['limits']['expanded_tar_bytes']
         and validation['summary']['expanded_tar_bytes'] % 512 == 0
         and validation['summary']['gzip_crc_and_trailer_verified'] is validation['summary']['compressed_eof_verified'] is True
         and validation['metadata_only_counters'] == spec['metadata_only_counters']
         and validation['selected_people'] is None
         and all(validation[k] is False for k in ('eligibility_computed', 'source_policy_change',
             'scientific_panel_budget_consumed', 'source_continuation_authorized')),
         'complete original compressed identity, every transcript, and no scientific authority')
    return catalog, index, validation


def _catalog_originals(directory, origin):
    collector = _collector()
    spec = collector.registration()
    api = {key: document((directory / ('api/' + key + '.json')).read_bytes()) for key in ('run', 'commit', 'artifacts')}
    _catalog_api(api, origin, collector)
    with ExitStack() as stack:
        archives = {role: stack.enter_context(archive(directory / ('originals/' + role + '.zip'), pin, CATALOG_MAXIMUM))
            for role, pin in origin['artifacts'].items()}
        producer = archives['producer']
        expected = {'PRODUCER.json', *('repository/' + p for p in collector.paths())}
        need({i.filename for i in producer.infolist() if not i.is_dir()} == expected, 'complete original metadata producer file set')
        proof_raw = producer.read('PRODUCER.json')
        proof = document(proof_raw)
        need(digest(proof_raw) == origin['producer_manifest_sha256']
             and proof['schema'] == 'english-v6-train360-metadata-producer-v1' and proof['execution'] == origin['execution']
             and proof['frozen_inputs'] == collector.FROZEN and proof['metadata_only_counters'] == spec['metadata_only_counters']
             and [r['path'] for r in proof['files']] == collector.paths(), 'actual captured metadata producer manifest')
        for item in proof['files']:
            raw = producer.read('repository/' + item['path'])
            expected = ((ROOT / collector.TEMPLATE).read_text().replace(collector.TOKEN, origin['execution']['prepared_parent']).encode()
                if item['path'] == collector.WORKFLOW else (ROOT / item['path']).read_bytes())
            need(raw == expected and len(raw) == item['bytes'] and digest(raw) == item['sha256']
                 and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['git_blob_sha1'],
                 'every exact original captured metadata implementation and resource')
        receipts = archives['metadata']
        needed = {'AUTHORIZATION.json', 'PREPARATION_API.json', 'COLLECTION_API.json', 'TRANSFER_STARTED.json',
            'GET_ATTEMPT.json', 'HTTP_RESPONSE.json', 'PROGRESS.jsonl', 'RESULT.json',
            *('COMPLETE/' + n for n in CATALOG_MEMBERS)}
        need({i.filename for i in receipts.infolist() if not i.is_dir()} == needed, 'complete successful original metadata receipts')
        authorization = dict(schema='english-v6-train360-metadata-authorization-v1', execution=origin['execution'],
            producer_sha256=digest(proof_raw), registration_sha256=collector.FROZEN[collector.REGISTRATION],
            maximum_archive_gets=1, scientific_panel_budget_consumed=False, metadata_only_counters=spec['metadata_only_counters'])
        need(document(receipts.read('AUTHORIZATION.json')) == authorization
             and {i.filename for i in archives['consumed'].infolist() if not i.is_dir()} == {'CONSUMPTION.json'}
             and document(archives['consumed'].read('CONSUMPTION.json')) == dict(authorization, status='METADATA_TRANSPORT_CONSUMED_UNKNOWN'),
             'same original metadata-only consumed marker and authorization')
        for name, consumed in (('PREPARATION_API.json', False), ('COLLECTION_API.json', True)):
            original_api = document(receipts.read(name))
            collector.once(origin['execution'], consumed,
                api=lambda endpoint: original_api['runs'] if '/runs?' in endpoint else original_api['artifacts'])
        raw = {n: receipts.read('COMPLETE/' + n) for n in CATALOG_MEMBERS}
        catalog, index, validation = _catalog_complete(raw, origin, collector)
        result = document(receipts.read('RESULT.json'))
        need(result['schema'] == 'english-v6-train360-metadata-acquisition-result-v1'
             and result['status'] == 'COMPLETE_AUTHENTICATED_METADATA' and result['execution'] == origin['execution']
             and result['summary'] == validation['summary'] and result['catalog'] == validation['catalog']
             and result['member_index'] == validation['member_index']
             and result['validation_sha256'] == digest(raw['VALIDATION.json'])
             and result['archive_get_attempted'] == result['archive_response_returned'] == 1
             and result['algorithm_result'] is None and result['scientific_panel_budget_consumed'] is False
             and result['metadata_only_counters'] == spec['metadata_only_counters'], 'same completed original metadata result')
    return dict(catalog=catalog, index=index, validation=validation, origin=origin,
        report=dict(schema='english-v6-b28-complete-original-train360-readback-v1', status='PASS_METADATA_ONLY',
            execution=origin['execution'], artifacts=origin['artifacts'], files=origin['files'],
            producer_manifest_sha256=origin['producer_manifest_sha256'], catalog_members=origin['catalog_members'],
            summary=validation['summary'], scientific_execution_authorized=False, model_calls=0, source_calls=0))


@contextmanager
def catalog_scope(directory):
    """Reopen all six fixed external originals; never use a selected-text cache."""
    origin = catalog_registration()
    directory = Path(directory)
    need(directory.is_absolute() and directory == directory.resolve() and directory.is_dir()
         and all(not p.is_symlink() for p in (directory, *directory.parents)), 'resolved original catalog proof root')
    actual = []
    for path in directory.rglob('*'):
        mode = path.lstat().st_mode
        need(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'no linked or special catalog proof member')
        if stat.S_ISREG(mode):
            name = path.relative_to(directory).as_posix()
            actual.append(name)
            need(name in origin['files'] and path.stat().st_size == origin['files'][name]['bytes'], 'complete catalog file length')
            with path.open('rb') as stream:
                need(hashlib.file_digest(stream, 'sha256').hexdigest() == origin['files'][name]['sha256'], 'complete catalog file SHA256')
    need(sorted(actual) == sorted(CATALOG_FILES), 'all and only six original catalog proof files')
    result = _catalog_originals(directory, origin)
    token = _CATALOG.set(result)
    try:
        yield result['report']
    finally:
        _CATALOG.reset(token)


def require_catalog():
    value = _CATALOG.get()
    need(type(value) is dict and value['origin'] == catalog_registration(), 'complete original train360 catalog_scope required')
    return value


def train360_rows(catalog, index, base):
    """Use the unchanged complete-text predicate after whole-catalog proof."""
    census, eligible, seen = set(), {}, set()
    flacs = {i['path']: i for i in index['members'] if i['kind'] == 'opaque_flac'}
    for item in catalog['transcripts']:
        match = re.fullmatch(r'LibriSpeech/train-clean-360/([1-9][0-9]*)/([1-9][0-9]*)/\1-\2\.trans\.txt', item['path'])
        need(match is not None and digest(item['text'].encode()) == item['sha256']
             and len(item['text'].encode()) == item['bytes'], 'whole original clean360 transcript')
        speaker, chapter = match.groups()
        census.add(speaker)
        eligible.setdefault(speaker, [])
        for line in item['text'].splitlines():
            ident, text = line.split(' ', 1)
            need(ident not in seen and re.fullmatch(speaker + '-' + chapter + r'-[0-9]+', ident), 'unique original clean360 utterance')
            seen.add(ident)
            words = text.split(' ')
            if 4 <= len(words) <= 14 and not base.BLOCK.intersection(words) and all(re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words):
                member = 'LibriSpeech/train-clean-360/' + speaker + '/' + chapter + '/' + ident + '.flac'
                pin = flacs[member]
                eligible[speaker].append(dict(speaker=speaker, chapter=chapter, source_id=ident, transcript=text,
                    transcript_file_sha256=item['sha256'], transcript_member=item['path'], archive_key='train-clean-360',
                    corpus_split='train-clean-360', flac_member=member,
                    original_flac={k: pin[k] for k in ('bytes', 'sha256')}))
    return census, eligible


def inputs(require_complete_catalog=True):
    fixed = {DESIGN_PATH: DESIGN_SHA, OPENING_PATH: OPENING_SHA, MANIFEST: EXPOSURE_SHA,
        ORIGIN: ORIGIN_SHA, CATALOG_ORIGIN: CATALOG_ORIGIN_SHA}
    # Refuse all unbound states before loading an old or new ranking function.
    need(all(valid_hash(pin) for pin in fixed.values()), 'complete actual metadata/terminal pins remain unbound')
    values = {name: document(frozen(name, pin)) for name, pin in fixed.items()}
    design, opening = values[DESIGN_PATH], values[OPENING_PATH]
    need(design['schema'] == 'english-v6-b28-train-natural-event-design-v1'
         and design['natural_weight'] == .16 and design['optimizer_updates'] == 192 and design['maximum_new_model_forwards'] == 1847
         and design['initial_pt_sha256'] == INITIAL_PT and design['historical_control_pt_sha256'] == CONTROL_PT
         and design['scientific_execution_authorized'] is False
         and design['source_protocol']['full_preoptimizer_assessment_required'] is True
         and design['source_protocol']['natural_people_per_panel'] == 12
         and design['source_protocol']['synthetic_people_per_panel'] == 24, 'unchanged prior B28 objective and full preoptimizer panels')
    need(opening['schema'] == 'english-v6-b28-after-original-b27-no-promotion-v1'
         and opening['condition'] == 'only_after_actual_complete_B27_primary_NO_PROMOTION' and opening['enabled'] is True
         and opening['original_B27_challenger_pt_sha256'] == B27_PT
         and all(valid_hash(opening[k]) for k in ('original_failure_report_sha256', 'original_training_binding_sha256', 'quality_reader_sha256'))
         and type(opening['quality_lock']) is dict and opening['quality_lock']['panel'] == 'primary'
         and bool(opening['original_primary_files']) and bool(opening['original_primary_transport']),
         'complete original B27 terminal explicitly bound before new selection')
    previous = _previous()
    inherited = previous.validate()
    base = previous.inputs()[0]
    old = original_assessment()
    history = document(frozen(ORIGINAL_HISTORY, SOURCE_MEMBERS['HISTORY.json']))
    source_spec = document(frozen(SOURCE_INPUTS, SOURCE_INPUTS_SHA))
    need(source_spec['inputs']['source'] == SOURCE_PIN and source_spec['upstream_producers']['source'] == SOURCE_PRODUCER_PIN,
         'same original SOURCE pins as the frozen B27 PRIMARY input declaration')
    exposure, origin = values[MANIFEST], values[ORIGIN]
    need(origin == _source_origin(exposure, origin['source_api_identity']), 'exact original B27 source origin')
    fields = ['id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person']
    people = original_assessment_people(old)
    need(exposure['schema'] == 'english-v6-b28-authenticated-b27-exposure-v1'
         and exposure['source_artifact'] == SOURCE_PIN and exposure['original_metadata_sha256'] == SOURCE_MEMBERS
         and exposure['assessment_sha256'] == ORIGINAL_ASSESSMENT['sha256']
         and exposure['source_ancestry_sha256'] == exposure['history_receipt_sha256'] == old['ancestry_sha256'] == SOURCE_MEMBERS['HISTORY.json']
         and exposure['selected_original_slots'] == len(exposure['rows']) == 120
         and exposure['selection_fields'] == fields and exposure['metadata_only'] is True
         and exposure['new_kws_forwards'] == 0 and exposure['replication_opened'] is False
         and all(set(row) == set(fields) and canonical_person(row) == row['canonical_person'] for row in exposure['rows'])
         and exposure['primary_people'] == people['primary'] and exposure['sealed_replication_people'] == people['replication']
         and exposure['all_excluded_people'] == base.ordered(set(people['primary']) | set(people['replication'])),
         'all120 unfiltered B27 identities and all72 exposed or sealed people')
    catalog_origin = catalog_registration()
    complete = require_catalog() if require_complete_catalog else None
    hashes = dict(inherited['hashes'])
    hashes.update(fixed)
    hashes[SOURCE_INPUTS] = SOURCE_INPUTS_SHA
    collector = _collector()
    for name in collector.paths():
        hashes[name] = digest((ROOT / name).read_bytes())
    hashes[COLLECTOR] = COLLECTOR_SHA
    return dict(base=base, old=old, history=history, exposure=exposure, origin=origin, hashes=hashes,
        design=design, opening=opening, catalog_origin=catalog_origin, complete=complete)


def build_documents():
    context = inputs()
    base, old, history, exposure, origin, hashes = (context[k] for k in ('base', 'old', 'history', 'exposure', 'origin', 'hashes'))
    rows = exposure['rows']
    exposed = {base.source_person(row) for row in rows}
    reserved = set(exposure['sealed_replication_people'])
    need(exposed == set(exposure['primary_people']) and len(exposed) == len(reserved) == 36 and not exposed & reserved
         and len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == 120
         and len({r['source_group_id'] for r in rows}) == 36
         and sum(r['cohort'] == 'synthetic' for r in rows) == 96 and sum(r['cohort'] == 'natural' for r in rows) == 24,
         'complete original B27 population before any ranking')
    excluded = exposed | reserved
    need(excluded == set(exposure['all_excluded_people'])
         and not set(history['all_pcm_sha256']) & {r['pcm_sha256'] for r in rows}, 'no old reserve or original PCM reclaimed')
    synth_blocked = set(history['synthetic_excluded_people']) | excluded
    natural_blocked = set(history['natural_excluded_people']) | excluded
    ancestry = copy.deepcopy(history)
    ancestry.update(schema='english-v6-b28-complete-exposure-and-sealed-reserves-v1',
        inherited_history_sha256=SOURCE_MEMBERS['HISTORY.json'], inherited_assessment_sha256=ORIGINAL_ASSESSMENT['sha256'],
        original_B27_source=dict(SOURCE_PIN), B27_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'], B27_exposure_sha256=EXPOSURE_SHA,
        all_pcm_sha256=sorted(set(history['all_pcm_sha256']) | {r['pcm_sha256'] for r in rows}),
        all_source_groups=sorted(set(history['all_source_groups']) | {r['source_group_id'] for r in rows}),
        canonical_people=base.ordered(set(history['canonical_people']) | exposed),
        synthetic_excluded_people=base.ordered(synth_blocked), natural_excluded_people=base.ordered(natural_blocked),
        B27_exposed_primary_people=base.ordered(exposed), B27_sealed_replication_people=base.ordered(reserved),
        all_B27_120_raw_originals_added=True, all_prior_replication_sealed=True,
        B27_quality_metrics_read_for_selection=False, old_B27_replication_opened=False,
        automatic_next_assessment_allowed=False, B28_execution_authorized=False, metadata_input_sha256=dict(hashes))
    need(len(history['all_pcm_sha256']) == 3063 and len(ancestry['all_pcm_sha256']) == 3183, 'complete3063 prior plus all120 B27 originals')
    prior_catalog = base.read_inputs()
    mapping = prior_catalog['piper-config.json']['speaker_id_map']
    by_index = {i: p for p, i in mapping.items()}
    need(len(by_index) == len(mapping) == len({base.canonical(p) for p in mapping}), 'complete nonaliasing Piper map')
    available_synth = {base.canonical(p): i for p, i in mapping.items() if base.canonical(p) not in synth_blocked}
    need({base.canonical(p) for p in mapping} <= natural_blocked, 'every potential Piper alias excluded from natural readers')
    census100, eligible100 = base.catalog_rows(prior_catalog['FULL_METADATA_CATALOG.json'])
    complete = context['complete']
    census360, eligible360 = train360_rows(complete['catalog'], complete['index'], base)
    need(not census100 & census360, 'upstream TRAIN split reader identities disjoint')
    eligible = {p: [dict(r, archive_key='train-clean-100', corpus_split='train-clean-100', original_flac=None,
        flac_member='LibriSpeech/train-clean-100/' + r['speaker'] + '/' + r['chapter'] + '/' + r['source_id'] + '.flac')
        for r in eligible100.get(p, [])] for p in census100}
    eligible.update(eligible360)
    census = census100 | census360
    available_natural = census - natural_blocked
    usable_natural = {p for p in available_natural if len(eligible[p]) >= 2}
    need(len(available_synth) >= 48 and len(usable_natural) >= 24, 'two full independent panels available without redraw')
    panels, selected = {}, set()
    for role in ('primary', 'replication'):
        speakers = sorted(set(available_synth) - selected, key=lambda p: base.rank(SALTS[role]['piper'], 'speaker', p))[:24]
        need(len(speakers) == 24, 'exact24 new synthetic people')
        selected.update(speakers)
        indices = [available_synth[p] for p in speakers]
        readers = sorted(usable_natural - selected, key=lambda p: base.rank(SALTS[role]['natural'], 'speaker', p))[:12]
        need(len(readers) == 12, 'exact12 new natural people')
        selected.update(readers)
        roster = [r for p in readers for r in sorted(eligible[p], key=lambda r: base.rank(SALTS[role]['natural'], 'utterance', r['source_id']))[:2]]
        natural = copy.deepcopy(old[role]['natural'])
        upstream100 = natural.pop('upstream')
        natural.pop('corpus_split')
        natural.pop('full_metadata_catalog_sha256')
        natural.update(opened=False, salt=SALTS[role]['natural'], speakers=readers, roster=roster,
            corpus_splits=['train-clean-100', 'train-clean-360'],
            upstreams={'train-clean-100': upstream100, 'train-clean-360': complete['catalog']['corpus']},
            full_metadata_catalog_sha256={'train-clean-100': base.INPUTS['FULL_METADATA_CATALOG.json'],
                'train-clean-360': context['catalog_origin']['catalog_members']['FULL_METADATA_CATALOG.json']['sha256']},
            complete_train360_metadata_origin_sha256=CATALOG_ORIGIN_SHA,
            excluded_canonical_speakers=base.ordered(natural_blocked | (selected - set(readers))),
            available_census=base.ordered(available_natural), selected_utterance_ids=[r['source_id'] for r in roster])
        panels[role] = dict(synthetic=dict(salt=SALTS[role]['piper'], speaker_indices=indices,
            speaker_id_map={str(i): by_index[i] for i in indices}, original_slots=96,
            seed_base=20264100 if role == 'primary' else 20264200), natural=natural)
    need(len(selected) == 72 and not selected & excluded, 'all72 new people distinct across both panels and corpus aliases')
    assessment = copy.deepcopy(old)
    for key in ('bounded_proposal', 'primary_scoring_requires_exact_final_B27_weights_and_new_source_gate_pass',
                'original_B24Q1_complete_NO_PROMOTION_required_before_any_new_budget'):
        assessment.pop(key, None)
    assessment.update(schema='english-v6-b28-preoptimizer-natural-event-assessment-v1',
        purpose='One fixed TRAIN-only natural suffix no-event objective with two fully frozen independent panels.',
        training_design_sha256=DESIGN_SHA, ancestry_sha256=digest(encoded(ancestry)),
        metadata_validator_sha256=digest(Path(__file__).read_bytes()), metadata_inputs_sha256=dict(hashes),
        primary=panels['primary'], replication=panels['replication'],
        freeze_before_optimizer_and_new_source=True, freeze_after_training_before_new_source=False,
        original_B27_source_exclusion=exclusion_summary(exposure, origin),
        complete_train360_metadata_origin_sha256=CATALOG_ORIGIN_SHA,
        original_B27_complete_NO_PROMOTION_required_before_any_new_budget=True,
        original_B27_result_is_eligibility_only_not_hypothesis_support=True, B27_quality_metrics_read_for_selection=False,
        opening_condition='only_after_actual_complete_B27_primary_NO_PROMOTION', old_B27_replication_opened=False,
        primary_scoring_requires_exact_final_B28_weights_and_new_source_gate_pass=True,
        model_roles=dict(B0='unchanged_B0', B9='unchanged_B9', control='immutable_original_B24_challenger',
            challenger='only_final_B28_192_update_checkpoint'),
        promotion_prerequisites='Both complete independent B28 quality PASSes and unchanged engineering/readback/installation gates.',
        replication_opening='Only complete authenticated B28 PRIMARY numerical PASS opens this frozen untouched replica.',
        metadata_selection=dict(algorithm=old['metadata_selection']['algorithm'], salts=SALTS,
            corpus_census=len(census), corpus_census_by_split={'train-clean-100': len(census100), 'train-clean-360': len(census360)},
            eligible_synthetic_voices=len(available_synth), available_natural_census=len(available_natural),
            eligible_natural_readers=len(usable_natural), synthetic_excluded_canonical_people=len(synth_blocked),
            natural_excluded_canonical_people=len(natural_blocked), metadata_only=True, selection_before_audio=True,
            selection_before_optimizer=True, selection_after_training=False, selection_before_first_independent_KWS=True,
            source_replacements=0, reserves_reclaimed=0, cross_source_type_identity_exclusion=True))
    assessment['training'] = dict(assessment['training'], design_path=DESIGN_PATH, design_sha256=DESIGN_SHA,
        initial_pt_sha256=INITIAL_PT, historical_control_pt_sha256=CONTROL_PT,
        new_challenger_updates=192, new_model_forwards=1847, optimizer_allowed=False,
        objective_schema=context['design']['objective_schema'], natural_weight=.16)
    assessment['frozen_model_pt_sha256'] = dict(assessment['frozen_model_pt_sha256'], challenger=None, control=CONTROL_PT)
    documents = {'ancestry.json': ancestry, 'assessment.json': assessment}
    for role in ('primary', 'replication'):
        synthetic, planned = panels[role]['synthetic'], []
        for ordinal, index in enumerate(synthetic['speaker_indices']):
            speaker = by_index[index]
            for cell, scale, cls, text, native in (
                ('mero-fast', 1.4, 2, 'Hey Mero.', '[[heɪ mˈɛɹoʊ]]'), ('mero-slow', 1.8, 2, 'Hey Mero.', '[[heɪ mˈɛɹoʊ]]'),
                ('question', 1., 0, 'What time is it?', '[[wˈʌt tˈaɪm ɪz ɪt]]'), ('request', 1.08, 0, 'Please set a timer.', 'Please set a timer.')):
                planned.append(dict(id=f'b28-{role}-{index}-{cell}', speaker_index=index, source_speaker_id=speaker,
                    source_group_id='piper-libritts-' + speaker, role='development', panel=role, cell_id=cell,
                    cohort='synthetic', length_scale=scale, seed=synthetic['seed_base'] + ordinal, requested_class=cls,
                    requested_text=text, native_input=native, observed_class=None, training_allowed=False))
        documents[role + '.json'] = dict(schema='english-v6-b28-fresh-combined-source-v1', panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)), source_ancestry_sha256=digest(encoded(ancestry)),
            model=base.PIPER, config_sha256=base.INPUTS['piper-config.json'], max_source_calls=96,
            maximum_native_observer_calls=480, original_slots=96, expected_providers=base.PROVIDERS,
            source_format=old['source_format'], source_admission_rule='unchanged B14 admission and Q1 whole-original successful-empty quarantine',
            orchestration_contract_sha256=ORCHESTRATION_SHA, existing_replication_opened=False,
            training_allowed=False, promotion_allowed=False, rows=planned)
        documents['natural-' + role + '.json'] = panels[role]['natural']
    need(set(documents) == set(OUTPUT_NAMES), 'all and only six prospective frozen documents')
    return documents, hashes


def validate():
    documents, hashes = build_documents()
    for name, value in documents.items():
        raw = encoded(value)
        need((PANELS / name).read_bytes() == raw, 'complete deterministic frozen B28 metadata: ' + name)
        hashes[(PANELS / name).relative_to(ROOT).as_posix()] = digest(raw)
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    return dict(schema='english-v6-b28-offline-metadata-validation-v1', status='PASS_METADATA_ONLY', hashes=hashes,
        selection=documents['assessment.json']['metadata_selection'], training_design_sha256=DESIGN_SHA,
        original_B27_source_exclusion=documents['assessment.json']['original_B27_source_exclusion'],
        complete_original_train360_catalog=require_catalog()['report'], all_B27_120_originals_excluded=True,
        all_B27_72_people_excluded=True, historical_PCM_identities_excluded=3183,
        quality_metrics_read_for_selection=False, freeze_before_optimizer_and_new_source=True,
        orchestration_contract_sha256=ORCHESTRATION_SHA, automatic_next_assessment_allowed=False,
        model_calls=0, source_calls=0, observer_calls=0, optimizer_updates=0,
        execution_authorized=False, source_execution_authorized=False, primary_scoring_authorized=False,
        replica_opening_authorized=False, engineering_authorized=False, promotion_allowed=False)


def validate_registration():
    """Frozen small files for pre-fetch guard only; full validate is still required."""
    context = inputs(False)
    value = document((ROOT / BINDING_PATH).read_bytes())
    need(value['schema'] == 'english-v6-b28-prospective-assessment-binding-v1' and value['frozen'] is True
         and value['optimizer_allowed'] is False and value['recipe_path'] == DESIGN_PATH and value['recipe_sha256'] == DESIGN_SHA,
         'explicit frozen B28 registration; no execution permission')
    hashes = dict(context['hashes'])
    for name in OUTPUT_NAMES:
        path = PANELS / name
        need(path.is_file() and not path.is_symlink(), 'all six frozen metadata documents')
        hashes[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    need(value['files'] == hashes and value['assessment'] == dict(path=PREFIX + 'assessment.json', sha256=hashes[PREFIX + 'assessment.json'])
         and value['original_B27_source_exclusion'] == exclusion_summary(context['exposure'], context['origin']),
         'whole exact static registration and explicit original catalog transport')
    assessment = document((PANELS / 'assessment.json').read_bytes())
    need(assessment['freeze_before_optimizer_and_new_source'] is True and assessment['training_design_sha256'] == DESIGN_SHA
         and assessment['metadata_selection']['salts'] == SALTS and assessment['metadata_inputs_sha256'] == context['hashes']
         and all(assessment[k] is False for k in ('execution_authorized', 'source_execution_authorized',
             'primary_scoring_authorized', 'replica_opening_authorized', 'engineering_authorized', 'optimizer_allowed', 'promotion_allowed')),
         'registered selection stays scientifically sealed')
    return dict(schema='english-v6-b28-frozen-metadata-registration-validation-v1', status='PASS_REGISTRATION_ONLY',
        hashes=hashes, original_B27_source_exclusion=value['original_B27_source_exclusion'],
        catalog_origin_sha256=CATALOG_ORIGIN_SHA, complete_catalog_readback_required_before_execution=True,
        execution_authorized=False, model_calls=0, source_calls=0, optimizer_updates=0)


def freeze():
    need(not any((PANELS / n).exists() for n in OUTPUT_NAMES), 'freeze only once; never redraw or overwrite selected panels')
    documents, _ = build_documents()
    PANELS.mkdir(exist_ok=True)
    for name, value in documents.items():
        with (PANELS / name).open('xb') as stream:
            stream.write(encoded(value))
    return validate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--registration-only', action='store_true')
    parser.add_argument('--catalog-proof', type=Path)
    args = parser.parse_args()
    need(not (args.freeze and args.registration_only), 'one explicit metadata operation')
    if args.registration_only:
        value = validate_registration()
    else:
        need(args.catalog_proof is not None, 'complete original catalog proof directory required')
        with catalog_scope(args.catalog_proof):
            value = freeze() if args.freeze else validate()
    print(json.dumps(value, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
