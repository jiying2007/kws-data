"""Freeze one evaluation-only B24R1 primary and sealed replica from metadata.

Actual B24 TRAIN design and assessment stay immutable. Original B24 SOURCE
STOP stays terminal; its 120 PCM identities and all 72 exposed or reserved
people extend the complete historical exclusion chain. Selection reads no
observation, KWS result, model, or audio. A separate complete original SOURCE
STOP proof and reviewed activation are required before any new source call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PANELS = HERE / 'english_v6_b24r1_panels'
DESIGN_PATH = 'automation/english_v6_b24_b23_warm_start_design.json'
DESIGN_SHA = '4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309'
PREVIOUS = 'automation/english_v6_b24_source_metadata.py'
PREVIOUS_SHA = '747fc9b5127707f982f62d232d9a7330218450bf0d3585a0c42fc000ac3d547c'
PREFIX = 'automation/english_v6_b24r1_panels/'
MANIFEST = PREFIX + 'b24-source-exposure.json'
ORIGIN = PREFIX + 'b24-source-origin.json'
EVALUATION_DESIGN_PATH = 'automation/english_v6_b24r1_evaluation_design.json'
EVALUATION_DESIGN_SHA = '79300b64e5870272ebeb54cc4364ee07b7e01518ad1b95752cc3a8ec2938d3a4'
EXTRA_INPUTS = {
    MANIFEST: 'b75a45b7b503460ba8dc04f79a2ab0de0319bfa5def16a39bbaf1cac09fd224d',
    ORIGIN: '2b49fe9d87d6fbb29853052508db866d4c739a55ee9d7464b18d4da6501b35cb',
    DESIGN_PATH: DESIGN_SHA,
    EVALUATION_DESIGN_PATH: EVALUATION_DESIGN_SHA,
}
SOURCE_PIN = dict(id=14950099270, run_id=45456888767,
    run_head='273e7bd0ae4b16972236f77aacae0eb5d1a12520', bytes=59393379,
    sha256='b4dd350b773ae925696f692e8f67caf7e83444220eab9cbb1f675ec20176ac4e')
SOURCE_PRODUCER_PIN = dict(id=14951855225, run_id=45456888767,
    run_head=SOURCE_PIN['run_head'], bytes=17486742,
    sha256='bcf47dc4e0d81f9678abf750ea184728a1cc8f7f1684ebbb9e71a36626bc6f7e')
SOURCE_WORKFLOW = '.github/workflows/english-v6-b24-primary-sources-once.yml'
SOURCE_BRANCH = 'research/en-v6-b24-primary-sources-once-20261124'
SOURCE_MEMBERS = {'MANIFEST.json': '5ab44377322029a99b105a998091550d2ef7a3768b01b67501c60d04b6256a12',
                  'HISTORY.json': 'fc5b2e3683222b5fd11136d42cd24949273b57148dd10a144bc6db6f34d8d72d'}
TRAINING_ASSESSMENT_SHA = '3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad'
PREVIOUS_ASSESSMENT_SHA = TRAINING_ASSESSMENT_SHA
ORIGINAL_ASSESSMENT = dict(path='automation/english_v6_b24_panels/assessment.json',
    bytes=81552, sha256=TRAINING_ASSESSMENT_SHA, git_blob_sha1='935045bb26cc847d7e34e1cddc70060c35ebe1ba')
SALTS = {role: {kind: f'english-v6-b24r1-frozen-b24-evaluation-{role}-{kind}-20261124-v1'
               for kind in ('piper', 'natural')} for role in ('primary', 'replication')}
CONTROL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
FROZEN_MODELS = {'B0': '1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7', 'B9': '7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655', 'control': '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f', 'challenger': 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'}
OUTPUT_NAMES = ('ancestry.json', 'assessment.json', 'primary.json', 'natural-primary.json',
                'replication.json', 'natural-replication.json')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _source_api_identity(api, pin):
    """Select only the identity fields already authenticated by the caller."""
    need(pin == SOURCE_PIN, 'same exact original B24 source pin')
    run, artifact, commit = (api[key] for key in ('run', 'artifact', 'commit'))
    need(run['id'] == pin['run_id'] and run['head_sha'] == pin['run_head'] == commit['sha']
         and run['run_attempt'] == 1 and run['event'] == 'push'
         and run['status'] == 'completed' and run['conclusion'] == 'failure'
         and run['head_branch'] == SOURCE_BRANCH and run['path'] == SOURCE_WORKFLOW
         and run['repository']['full_name'] == 'jiying2007/kws-data' and run['repository']['private'] is False
         and len(commit['parents']) == 1 and [row['filename'] for row in commit['files']] == [SOURCE_WORKFLOW]
         and artifact['id'] == pin['id'] and artifact['size_in_bytes'] == pin['bytes']
         and artifact['digest'] == 'sha256:' + pin['sha256'] and artifact['expired'] is False
         and artifact['name'] == 'english-v6-b24-primary-source-' + str(pin['run_id'])
         and artifact['workflow_run']['id'] == pin['run_id']
         and artifact['workflow_run']['head_sha'] == pin['run_head'],
         'original first public B24 SOURCE artifact, run and workflow-only Git identity')
    return dict(run={**{key: run[key] for key in ('id', 'run_attempt', 'head_sha', 'head_branch', 'event', 'path', 'status', 'conclusion')},
                     'repository': {key: run['repository'][key] for key in ('full_name', 'private')}},
        artifact={**{key: artifact[key] for key in ('id', 'name', 'size_in_bytes', 'digest', 'expired')},
                  'workflow_run': {key: artifact['workflow_run'][key] for key in ('id', 'head_sha')}},
        commit=dict(sha=commit['sha'], parents=[dict(sha=commit['parents'][0]['sha'])], files=[dict(filename=SOURCE_WORKFLOW)]))


def _canonical_person(row):
    value = next((row[key] for key in ('source_speaker_id', 'speaker', 'speaker_id') if row.get(key) is not None),
                 row['source_group_id'].rsplit('-', 1)[-1])
    text = str(value).removeprefix('p')
    need(re.fullmatch(r'[0-9]+', text) is not None, 'canonical original source person')
    return str(int(text))


def original_assessment_people():
    """The immutable old assessment supplies exposed and sealed people alike."""
    raw = (ROOT / ORIGINAL_ASSESSMENT['path']).read_bytes()
    need(len(raw) == ORIGINAL_ASSESSMENT['bytes'] and digest(raw) == TRAINING_ASSESSMENT_SHA
         and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
             == ORIGINAL_ASSESSMENT['git_blob_sha1'], 'exact original B24 assessment bytes and Git blob')
    assessment = json.loads(raw)
    people = {}
    for role in ('primary', 'replication'):
        synthetic = {str(int(str(p).removeprefix('p')))
                     for p in assessment[role]['synthetic']['speaker_id_map'].values()}
        natural = set(assessment[role]['natural']['speakers'])
        need(len(synthetic) == 24 and len(natural) == 12 and not synthetic & natural,
             'original B24 24 synthetic and 12 natural people per panel')
        people[role] = synthetic | natural
    need(not people['primary'] & people['replication'], 'all original B24 72 people distinct')
    return {key: sorted(value, key=int) for key, value in people.items()}


def _origin_document(exposure, api):
    identity = _source_api_identity(api, SOURCE_PIN)
    raw = encoded(exposure)
    return dict(schema='english-v6-b24r1-authenticated-b24-source-metadata-origin-v1',
        source_artifact=dict(SOURCE_PIN), source_producer_artifact=dict(SOURCE_PRODUCER_PIN),
        original_assessment=dict(ORIGINAL_ASSESSMENT), source_api_identity=identity,
        source_api_identity_sha256=digest(encoded(identity)), original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_exposure=dict(path=MANIFEST, bytes=len(raw), sha256=digest(raw)),
        extracted_member_allowlist=['MANIFEST.json', 'HISTORY.json'], metadata_only=True,
        audio_members_interpreted=0, observer_receipts_read=0, quality_results_read=False,
        model_calls=0, new_source_calls=0, original_source_gate_pass=False,
        full_original_SOURCE_STOP_proof_is_separate=True, scientific_execution_authorized=False)


def derive_original_metadata(members, api, source_pin):
    """Derive the two public whitelists from exact original metadata bytes.

    The archive wrapper verifies the whole ZIP SHA before calling this function.
    The API/lock inputs come from the caller's full original producer validation;
    this function also checks their immutable source and Git identity fields.
    """
    need(set(members) == set(SOURCE_MEMBERS)
         and all(type(members[name]) is bytes and digest(members[name]) == expected
                 for name, expected in SOURCE_MEMBERS.items()), 'exact original MANIFEST and HISTORY bytes')
    identity = _source_api_identity(api, source_pin)
    manifest, history = (json.loads(members[name]) for name in ('MANIFEST.json', 'HISTORY.json'))
    expected_execution = dict(prepared_parent=identity['commit']['parents'][0]['sha'], run_attempt=1,
        run_branch='refs/heads/' + SOURCE_BRANCH, run_head=source_pin['run_head'], run_id=str(source_pin['run_id']))
    need(manifest['schema'] == 'english-v6-b24-fresh-combined-source-v1'
         and manifest['panel'] == 'primary' and manifest['selected_original_slots'] == 120
         and len(manifest['rows']) == 120 and manifest['source_execution'] == expected_execution
         and manifest['assessment_sha256'] == PREVIOUS_ASSESSMENT_SHA
         and manifest['source_ancestry_sha256'] == manifest['history_receipt_sha256'] == SOURCE_MEMBERS['HISTORY.json']
         and manifest['new_kws_forwards'] == 0 and manifest['replication_opened'] is False
         and history['all_prior_replication_sealed'] is True,
         'complete pre-scoring original SOURCE identity and whole inherited history')
    rows = [{**{key: row[key] for key in ('id', 'cohort', 'source_group_id', 'pcm_sha256')},
             'canonical_person': _canonical_person(row)} for row in manifest['rows']]
    need(len({row['id'] for row in rows}) == len({row['pcm_sha256'] for row in rows}) == 120
         and len({row['canonical_person'] for row in rows}) == len({row['source_group_id'] for row in rows}) == 36
         and sum(row['cohort'] == 'synthetic' for row in rows) == 96
         and sum(row['cohort'] == 'natural' for row in rows) == 24
         and all(re.fullmatch(r'[0-9a-f]{64}', row['pcm_sha256']) for row in rows),
         'all120 original exposure identities, with no filtering or outcome fields')
    people = original_assessment_people()
    need(set(people['primary']) == {row['canonical_person'] for row in rows},
         'all original primary people match the immutable original assessment')
    exposure = dict(schema='english-v6-b24r1-authenticated-b24-exposure-v1',
        original_source_schema=manifest['schema'], panel='primary', selected_original_slots=120,
        source_artifact=dict(source_pin), source_execution=expected_execution,
        assessment_sha256=PREVIOUS_ASSESSMENT_SHA, source_ancestry_sha256=SOURCE_MEMBERS['HISTORY.json'],
        history_receipt_sha256=SOURCE_MEMBERS['HISTORY.json'], original_metadata_sha256=dict(SOURCE_MEMBERS),
        new_kws_forwards=0, replication_opened=False, rows=rows, metadata_only=True,
        primary_people=people['primary'], sealed_replication_people=people['replication'],
        all_excluded_people=sorted(set(people['primary']) | set(people['replication']), key=int),
        selection_fields=['id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person'])
    return {'b24-source-exposure.json': exposure, 'b24-source-origin.json': _origin_document(exposure, api)}


def verify_original_source(source_zip, source_api, source_pin, source_producer_zip=None):
    """Reexport actual failure-gated SOURCE without downloading or decoding.

    This metadata-only read does not certify the native failure predicate.
    The separate full SOURCE STOP reader authenticates all 14 original ZIPs.
    Optional source_producer_zip verifies the captured original assessment;
    otherwise its whole-original producer validation remains the caller's job.
    """
    source_zip = Path(source_zip)
    need(source_pin == SOURCE_PIN and source_zip.is_file() and not source_zip.is_symlink()
         and source_zip.stat().st_size == source_pin['bytes'], 'same complete original source ZIP before metadata read')
    with source_zip.open('rb') as stream:
        need(hashlib.file_digest(stream, 'sha256').hexdigest() == source_pin['sha256'], 'whole original source ZIP SHA256')
    with zipfile.ZipFile(source_zip) as archive:
        need(len(archive.infolist()) <= 8192 and len(archive.namelist()) == len(set(archive.namelist()))
             and sum(row.file_size for row in archive.infolist()) <= 128_000_000,
             'bounded unique original SOURCE members')
        # ZipFile.read verifies these original members' CRCs. No other member
        # is read, parsed, decoded, imported or used for choosing people.
        members = {name: archive.read(name) for name in SOURCE_MEMBERS}
    if source_producer_zip is not None:
        producer_path = Path(source_producer_zip)
        need(producer_path.is_file() and not producer_path.is_symlink()
             and producer_path.stat().st_size == SOURCE_PRODUCER_PIN['bytes'], 'exact original source producer bytes')
        with producer_path.open('rb') as stream:
            need(hashlib.file_digest(stream, 'sha256').hexdigest() == SOURCE_PRODUCER_PIN['sha256'],
                 'whole original source producer SHA256')
        with zipfile.ZipFile(producer_path) as archive:
            need(len(archive.infolist()) <= 8192 and len(archive.namelist()) == len(set(archive.namelist()))
                 and sum(row.file_size for row in archive.infolist()) <= 128_000_000,
                 'bounded unique original SOURCE producer members')
            raw = archive.read('repository/' + ORIGINAL_ASSESSMENT['path'])
            need(raw == (ROOT / ORIGINAL_ASSESSMENT['path']).read_bytes()
                 and digest(raw) == TRAINING_ASSESSMENT_SHA, 'captured original assessment equals immutable Git bytes')
    documents = derive_original_metadata(members, source_api, source_pin)
    published = {}
    for name, value in documents.items():
        path = PANELS / name
        raw = encoded(value)
        need(path.read_bytes() == raw and digest(raw) == EXTRA_INPUTS[path.relative_to(ROOT).as_posix()],
             'actual original metadata reexport equals frozen public whitelist: ' + name)
        published[path.relative_to(ROOT).as_posix()] = digest(raw)
    validation = validate()
    return dict(schema='english-v6-b24r1-original-source-metadata-reexport-v1', status='VERIFIED_ORIGINAL_METADATA_REEXPORT',
        source_artifact=dict(source_pin), original_metadata_sha256=dict(SOURCE_MEMBERS), published_files=published,
        original_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'], original_history_sha256=SOURCE_MEMBERS['HISTORY.json'],
        original_rows=120, prior_panel_people_excluded=72, complete_history_validated=validation['status'],
        original_source_identity_verified=True, whole_zip_sha256_verified=True, metadata_member_crc_verified=True,
        captured_original_assessment_verified=source_producer_zip is not None,
        full_original_SOURCE_STOP_proof_is_separate=True,
        audio_members_interpreted=0, observer_receipts_read=0, quality_results_read=False,
        model_calls=0, new_source_calls=0, scientific_execution_authorized=False)


def inputs():
    need(digest((ROOT / PREVIOUS).read_bytes()) == PREVIOUS_SHA, 'unchanged complete B24 metadata validator')
    spec = importlib.util.spec_from_file_location('b24r1_original_b24_metadata', ROOT / PREVIOUS)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    hashes = dict(previous.validate()['hashes'])
    base = previous.inputs()[0]
    old = json.loads((previous.PANELS / 'assessment.json').read_bytes())
    history_raw = (previous.PANELS / 'ancestry.json').read_bytes()
    history = json.loads(history_raw)
    extra = {}
    for name, expected in EXTRA_INPUTS.items():
        raw = (ROOT / name).read_bytes()
        need(digest(raw) == expected, 'unaltered B24R1 metadata input: ' + name)
        hashes[name] = expected
        extra[name] = json.loads(raw)
    exposure, origin = extra[MANIFEST], extra[ORIGIN]
    need(origin == _origin_document(exposure, origin['source_api_identity']), 'only necessary whitelisted original source identity fields')
    need(set(exposure) == {'schema', 'original_source_schema', 'panel', 'selected_original_slots', 'source_artifact',
         'source_execution', 'assessment_sha256', 'source_ancestry_sha256', 'history_receipt_sha256', 'original_metadata_sha256',
         'new_kws_forwards', 'replication_opened', 'rows', 'metadata_only', 'selection_fields',
         'primary_people', 'sealed_replication_people', 'all_excluded_people'}
         and exposure['schema'] == 'english-v6-b24r1-authenticated-b24-exposure-v1'
         and exposure['original_source_schema'] == 'english-v6-b24-fresh-combined-source-v1'
         and exposure['panel'] == 'primary' and exposure['selected_original_slots'] == 120 and len(exposure['rows']) == 120
         and exposure['source_artifact'] == SOURCE_PIN and exposure['metadata_only'] is True
         and exposure['original_metadata_sha256'] == SOURCE_MEMBERS
         and exposure['source_ancestry_sha256'] == exposure['history_receipt_sha256'] == old['ancestry_sha256'] == digest(history_raw)
         and exposure['assessment_sha256'] == PREVIOUS_ASSESSMENT_SHA == digest((previous.PANELS / 'assessment.json').read_bytes())
         and exposure['new_kws_forwards'] == 0 and exposure['replication_opened'] is False
         and exposure['selection_fields'] == ['id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person']
         and all(set(row) == set(exposure['selection_fields']) and row['canonical_person'] == base.source_person(row)
                 for row in exposure['rows']), 'complete source exclusion whitelist without raw API, audio metrics or observation payloads')
    identity = origin['source_api_identity']
    need(exposure['source_execution'] == dict(prepared_parent=identity['commit']['parents'][0]['sha'], run_attempt=1,
         run_branch='refs/heads/' + SOURCE_BRANCH, run_head=SOURCE_PIN['run_head'], run_id=str(SOURCE_PIN['run_id'])),
         'same immutable original source workflow and Git identity')
    original_people = original_assessment_people()
    need(exposure['primary_people'] == original_people['primary']
         and exposure['sealed_replication_people'] == original_people['replication']
         and exposure['all_excluded_people'] == sorted(set(original_people['primary']) | set(original_people['replication']), key=int),
         'complete original 36 exposed plus 36 sealed people, never only successful observations')
    design = extra[DESIGN_PATH]
    need(design['historical_control']['pt_sha256'] == design['initialization']['pt_sha256'] == CONTROL_PT
         and design['opening_condition'] == 'only_after_actual_complete_B23_primary_NO_PROMOTION'
         and design['updates'] == 192 and design['parameters'] == 43876
         and design['contrast'] == 'phase_initialization_B0_to_actual_B23_checkpoint_only'
         and design['initialization']['optimizer'] == 'fresh_AdamW'
         and design['initialization']['restore_optimizer_state'] is False
         and design['initialization']['exact_optimizer_continuation'] is False,
         'immutable actual B24 TRAIN retains its original B23 parent, design, and assessment')
    evaluation = extra[EVALUATION_DESIGN_PATH]
    need(evaluation['schema'] == 'english-v6-b24r1-frozen-models-new-source-assessment-v1'
         and evaluation['training_design_sha256'] == DESIGN_SHA
         and evaluation['training_assessment_sha256'] == TRAINING_ASSESSMENT_SHA
         and evaluation['frozen_model_pt_sha256'] == FROZEN_MODELS
         and evaluation['evaluation_only'] is True and evaluation['new_optimizer_updates'] == 0
         and evaluation['source_gate'] == old['source_gate'] and evaluation['gains'] == old['gains']
         and evaluation['automatic_R2_allowed'] is False and evaluation['R1_source_failure_terminal'] is True
         and evaluation['opening_condition'] == 'complete_original_B24_SOURCE_STOP_proof_and_separate_reviewed_activation'
         and all(evaluation[key] is False for key in ('execution_authorized', 'source_execution_authorized',
             'primary_scoring_authorized', 'replica_opening_authorized', 'engineering_authorized', 'optimizer_allowed',
             'promotion_allowed', 'old_B24_replication_opened', 'old_B24_sources_used_for_scoring', 'selection_before_optimizer')),
         'one separately registered evaluation of unchanged weights, without scientific execution authority')
    return base, old, history, exposure, dict(SOURCE_PIN), hashes


def build_documents():
    base, old, history, manifest, pin, hashes = inputs()
    rows = manifest['rows']
    exposed = {base.source_person(row) for row in rows}
    def people(role):
        value = {base.canonical(person) for person in old[role]['synthetic']['speaker_id_map'].values()}
        return value | set(old[role]['natural']['speakers'])
    reserved = people('replication')
    need(exposed == people('primary') and len(exposed) == len(reserved) == 36 and not exposed & reserved
         and len({row['id'] for row in rows}) == len({row['pcm_sha256'] for row in rows}) == 120
         and len({row['source_group_id'] for row in rows}) == 36
         and sum(row['cohort'] == 'synthetic' for row in rows) == 96
         and sum(row['cohort'] == 'natural' for row in rows) == 24,
         'complete original120 and all72 B24 exposed or reserved canonical people')
    newly_excluded = exposed | reserved
    need(newly_excluded == set(manifest['all_excluded_people']), 'all original B24 exposed and sealed people retained')
    synth_blocked = set(history['synthetic_excluded_people']) | newly_excluded
    natural_blocked = set(history['natural_excluded_people']) | newly_excluded
    need(not set(history['all_pcm_sha256']) & {row['pcm_sha256'] for row in rows},
         'all120 B24 original PCM identities remain independent of complete earlier history')
    ancestry = copy.deepcopy(history)
    ancestry.pop('B24_execution_authorized')
    ancestry.update(schema='english-v6-b24r1-complete-exposure-and-sealed-reserves-v1',
        inherited_history_sha256=old['ancestry_sha256'],
        inherited_assessment_sha256=manifest['assessment_sha256'], original_B24_source=pin,
        B24_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'], B24_exposure_sha256=EXTRA_INPUTS[MANIFEST],
        all_pcm_sha256=sorted(set(history['all_pcm_sha256']) | {row['pcm_sha256'] for row in rows}),
        all_source_groups=sorted(set(history['all_source_groups']) | {row['source_group_id'] for row in rows}),
        canonical_people=base.ordered(set(history['canonical_people']) | exposed),
        synthetic_excluded_people=base.ordered(synth_blocked), natural_excluded_people=base.ordered(natural_blocked),
        B24_exposed_primary_people=base.ordered(exposed), B24_sealed_replication_people=base.ordered(reserved),
        all_B24_120_raw_originals_added=True, all_prior_replication_sealed=True,
        B24_quality_results_read=False, B24_original_source_stop_preserved=True,
        old_B24_replication_opened=False, automatic_R2_allowed=False,
        B24R1_execution_authorized=False, metadata_input_sha256=dict(hashes))
    need(len(history['all_pcm_sha256']) == 2703 and len(ancestry['all_pcm_sha256']) == 2823,
         'all 2583 earlier plus 120 B23 plus 120 B24 PCM identities excluded')
    catalog_inputs = base.read_inputs()
    mapping = catalog_inputs['piper-config.json']['speaker_id_map']
    by_index = {index: person for person, index in mapping.items()}
    available_synth = {base.canonical(person): index for index, person in by_index.items()
                       if base.canonical(person) not in synth_blocked}
    census, eligible = base.catalog_rows(catalog_inputs['FULL_METADATA_CATALOG.json'])
    available_natural = census - natural_blocked
    usable_natural = {person for person in available_natural if len(eligible[person]) >= 2}
    need(len(available_synth) >= 48 and len(usable_natural) >= 24, 'two complete independent B24R1 panels remain available')
    panels, selected = {}, set()
    for role in ('primary', 'replication'):
        speakers = sorted(set(available_synth) - selected, key=lambda p: base.rank(SALTS[role]['piper'], 'speaker', p))[:24]
        need(len(speakers) == 24, 'exact24 new synthetic people')
        selected.update(speakers)
        indices = [available_synth[p] for p in speakers]
        readers = sorted(usable_natural - selected, key=lambda p: base.rank(SALTS[role]['natural'], 'speaker', p))[:12]
        need(len(readers) == 12, 'exact12 new natural readers, disjoint across both source types')
        selected.update(readers)
        roster = [row for person in readers for row in sorted(eligible[person],
                  key=lambda row: base.rank(SALTS[role]['natural'], 'utterance', row['source_id']))[:2]]
        natural = copy.deepcopy(old[role]['natural'])
        natural.update(opened=False, salt=SALTS[role]['natural'], speakers=readers, roster=roster,
            excluded_canonical_speakers=base.ordered(natural_blocked | (selected - set(readers))),
            available_census=base.ordered(available_natural), selected_utterance_ids=[row['source_id'] for row in roster])
        panels[role] = dict(synthetic=dict(salt=SALTS[role]['piper'], speaker_indices=indices,
            speaker_id_map={str(index): by_index[index] for index in indices}, original_slots=96,
            seed_base=20263300 if role == 'primary' else 20263400), natural=natural)
    need(len(selected) == 72 and not selected & newly_excluded, 'all72 new canonical people independent of all B24 primary and reserve people')
    assessment = copy.deepcopy(old)
    assessment.pop('original_B23_NO_PROMOTION_required_before_any_new_budget')
    assessment.pop('primary_scoring_requires_frozen_actual_B24_weights_and_actual_source_gate_pass')
    assessment.update(schema='english-v6-b24r1-frozen-models-new-source-assessment-v1',
        purpose='One bounded independent evaluation of the same actual B24 weights after original B24 SOURCE STOP; new primary and sealed replica selected after training and before audio or independent KWS.',
        ancestry_sha256=digest(encoded(ancestry)), metadata_validator_sha256=digest(Path(__file__).read_bytes()),
        metadata_inputs_sha256=dict(hashes), primary=panels['primary'], replication=panels['replication'],
        training_assessment_sha256=TRAINING_ASSESSMENT_SHA, training_design_sha256=DESIGN_SHA,
        evaluation_design_path=EVALUATION_DESIGN_PATH, evaluation_design_sha256=EVALUATION_DESIGN_SHA,
        frozen_model_pt_sha256=dict(FROZEN_MODELS), evaluation_only=True, new_optimizer_updates=0,
        training=dict(design_path=DESIGN_PATH, design_sha256=DESIGN_SHA,
            assessment_path=ORIGINAL_ASSESSMENT['path'], assessment_sha256=TRAINING_ASSESSMENT_SHA,
            original_run_id=45449190562, original_run_head='146b9e8022e067855f96e2e6a33f0c8d2e5d23af',
            original_completed_challenger_updates=192, new_challenger_updates=0, parameters=43876,
            immutable_challenger='actual_original_B24_challenger', challenger_pt_sha256=CANDIDATE_PT,
            control='immutable_original_B23_challenger', control_pt_sha256=CONTROL_PT,
            optimizer_allowed=False, restore_optimizer_state=False, new_historical_control_forwards=0,
            new_teacher_forwards=0, new_training_audio=False, dev_cal_locked_forwards_in_training=0,
            original_training_assessment_not_rebound=True, threshold_search=False),
        model_roles=dict(B0='unchanged_B0', B9='unchanged_B9', control='original_B23_challenger', challenger='unchanged_original_B24_challenger'),
        quality_control_role='exact original B23 challenger as immutable historical control; numerical B14 decision unchanged',
        promotion_prerequisites='Fresh B24R1 primary PASS and untouched B24R1 replica PASS with identical frozen weights, actual PT/ONNX/C11/Cortex-A32 parity, and target SSC305 AFE/callback/FRR/FAR/latency evidence.',
        replication_opening='Only authenticated actual B24R1 primary numerical PASS opens its separately registered replica; all older reserves remain sealed.',
        B24_quality_results_read=False, B24_original_source_stop_preserved=True,
        original_B24_SOURCE_STOP_proof_required_before_any_new_budget=True,
        separate_reviewed_activation_required=True, old_B24_sources_used_for_scoring=False,
        old_B24_replication_opened=False, old_B25_opening_condition_unchanged=True,
        automatic_R2_allowed=False, R1_source_failure_terminal=True,
        primary_scoring_requires_frozen_actual_B24_weights_and_actual_source_gate_pass=True,
        freeze_before_optimizer_and_new_source=False, freeze_after_training_before_new_source=True,
        source_execution_authorized=False, primary_scoring_authorized=False,
        replica_opening_authorized=False, engineering_authorized=False, optimizer_allowed=False,
        metadata_selection=dict(algorithm=old['metadata_selection']['algorithm'], salts=SALTS, corpus_census=251,
            eligible_synthetic_voices=len(available_synth), available_natural_census=len(available_natural),
            eligible_natural_readers=len(usable_natural), synthetic_excluded_canonical_people=len(synth_blocked),
            natural_excluded_canonical_people=len(natural_blocked), metadata_only=True,
            selection_before_audio=True, selection_before_optimizer=False, selection_after_training=True,
            selection_before_first_independent_KWS=True, source_replacements=0,
            reserves_reclaimed=0, cross_source_type_identity_exclusion=True))
    documents = {'ancestry.json': ancestry, 'assessment.json': assessment}
    for role in ('primary', 'replication'):
        synthetic, planned = panels[role]['synthetic'], []
        for ordinal, index in enumerate(synthetic['speaker_indices']):
            speaker = by_index[index]
            for cell, scale, cls, text, native in (
                    ('mero-fast', 1.4, 2, 'Hey Mero.', '[[heɪ mˈɛɹoʊ]]'),
                    ('mero-slow', 1.8, 2, 'Hey Mero.', '[[heɪ mˈɛɹoʊ]]'),
                    ('question', 1.0, 0, 'What time is it?', '[[wˈʌt tˈaɪm ɪz ɪt]]'),
                    ('request', 1.08, 0, 'Please set a timer.', 'Please set a timer.')):
                planned.append(dict(id=f'b24r1-{role}-{index}-{cell}', speaker_index=index, source_speaker_id=speaker,
                    source_group_id='piper-libritts-' + speaker, role='development', panel=role, cell_id=cell,
                    cohort='synthetic', length_scale=scale, seed=synthetic['seed_base'] + ordinal,
                    requested_class=cls, requested_text=text, native_input=native, observed_class=None, training_allowed=False))
        documents[role + '.json'] = dict(schema='english-v6-b24r1-fresh-combined-source-v1', panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)), source_ancestry_sha256=digest(encoded(ancestry)),
            model=base.PIPER, config_sha256=base.INPUTS['piper-config.json'], max_source_calls=96,
            maximum_native_observer_calls=480, original_slots=96, expected_providers=base.PROVIDERS,
            source_format=old['source_format'], source_admission_rule='unchanged full B14 five-observer policy',
            existing_replication_opened=False, training_allowed=False, promotion_allowed=False, rows=planned)
        documents['natural-' + role + '.json'] = panels[role]['natural']
    need(set(documents) == set(OUTPUT_NAMES), 'all and only the six prospective frozen documents')
    return documents, hashes


def validate():
    documents, hashes = build_documents()
    for name, value in documents.items():
        raw = encoded(value)
        need((PANELS / name).read_bytes() == raw, 'complete deterministic frozen B24R1 metadata: ' + name)
        hashes[(PANELS / name).relative_to(ROOT).as_posix()] = digest(raw)
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    return dict(schema='english-v6-b24r1-offline-metadata-validation-v1', status='PASS_METADATA_ONLY',
        hashes=hashes, selection=documents['assessment.json']['metadata_selection'],
        all_B24_120_originals_excluded=True, all_B24_72_people_excluded=True,
        historical_PCM_identities_excluded=2823, B24_quality_results_read=False,
        evaluation_only=True, training_assessment_sha256=TRAINING_ASSESSMENT_SHA,
        training_design_sha256=DESIGN_SHA, frozen_model_pt_sha256=dict(FROZEN_MODELS),
        B24_original_source_stop_preserved=True, full_original_SOURCE_STOP_proof_is_separate=True,
        automatic_R2_allowed=False, old_B25_opening_condition_unchanged=True,
        source_calls=0, observer_calls=0, kws_forwards=0, optimizer_updates=0,
        source_execution_authorized=False, primary_scoring_authorized=False,
        replica_opening_authorized=False, engineering_authorized=False, promotion_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='initial metadata-only freeze; refuses any output overwrite')
    args = parser.parse_args()
    if args.freeze:
        documents, _ = build_documents()
        need(not any((PANELS / name).exists() for name in OUTPUT_NAMES), 'freeze only once; never redraw selected panels')
        PANELS.mkdir(exist_ok=True)
        for name, value in documents.items():
            (PANELS / name).write_bytes(encoded(value))
    print(json.dumps(validate(), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
