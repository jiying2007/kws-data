"""One approved B24Q1 metadata freeze; no audio, observations, model or source call.

The actual B24 TRAIN and all B24/R1 terminal evidence remain immutable. Extend
complete historical exclusions with every R1 original and all exposed/reserved
people, then rank public metadata once for a fresh primary and sealed replica.
The successful-empty orchestration amendment is explicitly separate from the
unchanged native-payload and numerical-admission scientific snapshot.
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
PANELS = HERE / 'english_v6_b24q1_panels'
PREFIX = 'automation/english_v6_b24q1_panels/'
MANIFEST = PREFIX + 'b24r1-source-exposure.json'
ORIGIN = PREFIX + 'b24r1-source-origin.json'
PREVIOUS = 'automation/english_v6_b24r1_source_metadata.py'
PREVIOUS_SHA = 'c452f6a2fb5eae7d64d6040db29f372456286e43231cff72bdcaaa5496390b3f'
DESIGN_PATH = 'automation/english_v6_b24_b23_warm_start_design.json'
DESIGN_SHA = '4fa88cf4050aa276323a7306f21d53163d47c27e3c011c2547d5eeebe6980309'
EVALUATION_DESIGN_PATH = 'automation/english_v6_b24q1_evaluation_design.json'
EVALUATION_DESIGN_SHA = 'f3d4079a685f65eddecafce7afa909150c6136bc842ce116279888d2b556c66d'
PROPOSAL_PATH = 'research/english_v6_b24q1_preparation_20261125/APPROVED_PROPOSAL_SOURCE.json'
PROPOSAL_SHA = 'b5149d29049d5f308efd92a966ab36b0ccb81a9e04fbc64f8ff81e52d5fe7bea'
APPROVAL_PATH = 'research/english_v6_b24q1_preparation_20261125/APPROVAL.json'
APPROVAL_SHA = '6b3a9303874606b17045ff1ee0b78f7a08123c9a1cdc80442056451110b3b13b'
ORCHESTRATION_PATH = 'automation/english_v6_b24q1_orchestration_contract.json'
ORCHESTRATION_SHA = '75136b683c893e3d6457d93b2f9f74829c711e7b614b6a5b727380faf4f42f49'
STOP_REPORT_PATH = 'research/english_v6_b24r1_source_stop_20261125/READBACK.json'
STOP_REPORT_SHA = 'e3ca7baa272609fc8a4b304fb56b52bd98c8a85a504dd2f4ab0fab298b777ffd'
SOURCE_PIN = dict(id=14986233681, run_id=45534020313,
    run_head='b65bfe4974432b3d922738d7931d1c2d6d6f367c', bytes=213160927,
    sha256='802a70d9acd9d9ec0c096f201592fdcf5c8b5b5a34e31d8f300dea13edec814f')
SOURCE_PRODUCER_PIN = dict(id=14986820664, run_id=45534020313,
    run_head=SOURCE_PIN['run_head'], bytes=17859540,
    sha256='42655c7aa9e8844c6d4d6556a7895cbba7a0777aaff4befaaf750325514eaeb2')
SOURCE_PARENT = 'ea9f9e3b467231efe2746cbe64f867481177d09e'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b24r1-primary-sources-once.yml'
SOURCE_BRANCH = 'research/en-v6-b24r1-primary-sources-once-20261124'
SOURCE_MEMBERS = {'MANIFEST.json':'e8289607115d869797aef45bceca224d626107669bb84f0cd1a1a22a8661bae5',
                  'HISTORY.json':'eca9b1fa081022f0efa4da79f4d398affdce01a6ebdd81e4cad0bcd6405d3c06'}
TRAINING_ASSESSMENT_SHA = '3afdb17e822bb0b4a49757ac116a28101ff9f1447b22429305299047410fe2ad'
PREVIOUS_ASSESSMENT_SHA = 'e783c982f66ed35f903680772842ca8068aad6fb18221e22b9512cd819efa075'
ORIGINAL_ASSESSMENT = dict(path='automation/english_v6_b24r1_panels/assessment.json', bytes=84419,
    sha256=PREVIOUS_ASSESSMENT_SHA, git_blob_sha1='3848ead022b5be6c9a810880ec81aaf35d639d4a')
SALTS = {role:{kind:f'english-v6-b24q1-approved-empty-quarantine-{role}-{kind}-20261125-v1'
    for kind in ('piper','natural')} for role in ('primary','replication')}
CONTROL_PT = '784def9df6c2a8bfbecdb7ba2d7b70630337a1cda72561c05392327aa980297f'
CANDIDATE_PT = 'f6fce146324cab9286814bab26a594ef67cca58155fa2d2fb5f7fd9370992bb4'
FROZEN_MODELS = {'B0':'1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7',
    'B9':'7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655',
    'control':CONTROL_PT,'challenger':CANDIDATE_PT}
OUTPUT_NAMES = ('ancestry.json','assessment.json','primary.json','natural-primary.json',
                'replication.json','natural-replication.json')
EXTRA_INPUTS = {MANIFEST:'9a06e0d91eb6ba1aea9ec2fb76c6d7709d3eacdc1aafe9a93ba65218b8762f62',ORIGIN:'ae7ebd1d89b6e9a8e813748461c9445ae96f406d3418037695f68efa1b325f89',
    DESIGN_PATH:DESIGN_SHA,EVALUATION_DESIGN_PATH:EVALUATION_DESIGN_SHA,
    PROPOSAL_PATH:PROPOSAL_SHA,APPROVAL_PATH:APPROVAL_SHA,
    ORCHESTRATION_PATH:ORCHESTRATION_SHA,STOP_REPORT_PATH:STOP_REPORT_SHA}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _source_api_identity(api, pin):
    """Select only the identity fields already authenticated by the caller."""
    need(pin == SOURCE_PIN, 'same exact original B24R1 source pin')
    run, artifact, commit = (api[key] for key in ('run', 'artifact', 'commit'))
    need(run['id'] == pin['run_id'] and run['head_sha'] == pin['run_head'] == commit['sha']
         and run['run_attempt'] == 1 and run['event'] == 'push'
         and run['status'] == 'completed' and run['conclusion'] == 'failure'
         and run['head_branch'] == SOURCE_BRANCH and run['path'] == SOURCE_WORKFLOW
         and run['repository']['full_name'] == 'jiying2007/kws-data' and run['repository']['private'] is False
         and len(commit['parents']) == 1 and commit['parents'][0]['sha'] == SOURCE_PARENT and [row['filename'] for row in commit['files']] == [SOURCE_WORKFLOW]
         and artifact['id'] == pin['id'] and artifact['size_in_bytes'] == pin['bytes']
         and artifact['digest'] == 'sha256:' + pin['sha256'] and artifact['expired'] is False
         and artifact['name'] == 'english-v6-b24r1-primary-source-' + str(pin['run_id'])
         and artifact['workflow_run']['id'] == pin['run_id']
         and artifact['workflow_run']['head_sha'] == pin['run_head'],
         'original first public B24R1 SOURCE artifact, run and workflow-only Git identity')
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
    need(len(raw) == ORIGINAL_ASSESSMENT['bytes'] and digest(raw) == PREVIOUS_ASSESSMENT_SHA
         and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
             == ORIGINAL_ASSESSMENT['git_blob_sha1'], 'exact original B24R1 assessment bytes and Git blob')
    assessment = json.loads(raw)
    people = {}
    for role in ('primary', 'replication'):
        synthetic = {str(int(str(p).removeprefix('p')))
                     for p in assessment[role]['synthetic']['speaker_id_map'].values()}
        natural = set(assessment[role]['natural']['speakers'])
        need(len(synthetic) == 24 and len(natural) == 12 and not synthetic & natural,
             'original B24R1 24 synthetic and 12 natural people per panel')
        people[role] = synthetic | natural
    need(not people['primary'] & people['replication'], 'all original B24R1 72 people distinct')
    return {key: sorted(value, key=int) for key, value in people.items()}


def _origin_document(exposure, api):
    identity = _source_api_identity(api, SOURCE_PIN)
    raw = encoded(exposure)
    return dict(schema='english-v6-b24q1-authenticated-b24r1-source-metadata-origin-v1',
        source_artifact=dict(SOURCE_PIN), source_producer_artifact=dict(SOURCE_PRODUCER_PIN),
        original_assessment=dict(ORIGINAL_ASSESSMENT), source_api_identity=identity,
        source_api_identity_sha256=digest(encoded(identity)), original_metadata_sha256=dict(SOURCE_MEMBERS),
        published_exposure=dict(path=MANIFEST, bytes=len(raw), sha256=digest(raw)),
        extracted_member_allowlist=['MANIFEST.json', 'HISTORY.json'], metadata_only=True,
        audio_members_interpreted=0, observer_receipts_read=0, quality_results_read=False,
        model_calls=0, new_source_calls=0, original_source_gate_pass=False,
        full_original_SOURCE_STOP_proof_is_separate=True, scientific_execution_authorized=False,
        complete_original_STOP_report=dict(path=STOP_REPORT_PATH,bytes=12097,sha256=STOP_REPORT_SHA))


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
    need(manifest['schema'] == 'english-v6-b24r1-fresh-combined-source-v1'
         and manifest['panel'] == 'primary' and manifest['selected_original_slots'] == 120
         and len(manifest['rows']) == 120 and manifest['source_execution'] == expected_execution
         and manifest['assessment_sha256'] == PREVIOUS_ASSESSMENT_SHA
         and manifest['original_training_assessment_sha256'] == TRAINING_ASSESSMENT_SHA
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
    exposure = dict(schema='english-v6-b24q1-authenticated-b24r1-exposure-v1',
        original_source_schema=manifest['schema'], panel='primary', selected_original_slots=120,
        source_artifact=dict(source_pin), source_execution=expected_execution,
        assessment_sha256=PREVIOUS_ASSESSMENT_SHA, source_ancestry_sha256=SOURCE_MEMBERS['HISTORY.json'],
        history_receipt_sha256=SOURCE_MEMBERS['HISTORY.json'], original_metadata_sha256=dict(SOURCE_MEMBERS),
        new_kws_forwards=0, replication_opened=False, rows=rows, metadata_only=True,
        primary_people=people['primary'], sealed_replication_people=people['replication'],
        all_excluded_people=sorted(set(people['primary']) | set(people['replication']), key=int),
        selection_fields=['id', 'cohort', 'source_group_id', 'pcm_sha256', 'canonical_person'])
    return {'b24r1-source-exposure.json': exposure, 'b24r1-source-origin.json': _origin_document(exposure, api)}


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
             and sum(row.file_size for row in archive.infolist()) <= 512_000_000,
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
                 and digest(raw) == PREVIOUS_ASSESSMENT_SHA, 'captured original assessment equals immutable Git bytes')
    documents = derive_original_metadata(members, source_api, source_pin)
    published = {}
    for name, value in documents.items():
        path = PANELS / name
        raw = encoded(value)
        need(path.read_bytes() == raw and digest(raw) == EXTRA_INPUTS[path.relative_to(ROOT).as_posix()],
             'actual original metadata reexport equals frozen public whitelist: ' + name)
        published[path.relative_to(ROOT).as_posix()] = digest(raw)
    validation = validate()
    return dict(schema='english-v6-b24q1-original-source-metadata-reexport-v1', status='VERIFIED_ORIGINAL_METADATA_REEXPORT',
        source_artifact=dict(source_pin), original_metadata_sha256=dict(SOURCE_MEMBERS), published_files=published,
        original_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'], original_history_sha256=SOURCE_MEMBERS['HISTORY.json'],
        original_rows=120, prior_panel_people_excluded=72, complete_history_validated=validation['status'],
        original_source_identity_verified=True, whole_zip_sha256_verified=True, metadata_member_crc_verified=True,
        captured_original_assessment_verified=source_producer_zip is not None,
        full_original_SOURCE_STOP_proof_is_separate=True,
        audio_members_interpreted=0, observer_receipts_read=0, quality_results_read=False,
        model_calls=0, new_source_calls=0, scientific_execution_authorized=False)


def approval():
    raw = (ROOT / PROPOSAL_PATH).read_bytes()
    need(digest(raw) == PROPOSAL_SHA, 'exact approved source proposal, preserved without rewriting its historical status')
    proposal = json.loads(raw)
    approval_raw = (ROOT / APPROVAL_PATH).read_bytes()
    need(digest(approval_raw) == APPROVAL_SHA, 'exact separately recorded explicit user approval')
    value = json.loads(approval_raw)
    need(value['schema'] == 'english-v6-b24q1-explicit-user-approval-v1' and value['experiment_id'] == 'B24Q1'
         and value['user_statement'] == '批准提案，继续一次全新评估（推荐）' and value['approved'] is True
         and value['user_message_sent_at_utc'] is None
         and value['approved_proposal']['sha256'] == PROPOSAL_SHA
         and value['approved_proposal']['path'] == PROPOSAL_PATH
         and value['approved_proposal']['bytes'] == len(raw)
         and value['approved_proposal']['original_commit'] == 'd5b68dff01abcb66781496369458f7ee006baa87'
         and value['scope']['maximum_new_assessments'] == value['scope']['primary_panels'] == value['scope']['sealed_replication_panels'] == 1
         and value['scope']['metadata_selection_authorized'] is True
         and value['scope']['frozen_model_pt_sha256'] == FROZEN_MODELS
         and value['scope']['new_training_allowed'] is value['scope']['automatic_R2_or_next_panel_allowed'] is False
         and value['grants_immediate_scientific_execution'] is value['source_workflow_activated'] is False,
         'one explicitly approved new assessment; source entry still needs full proof and reviewed activation')
    need(proposal['protected_existing_state']['B26_opening_unchanged']['condition']
         == 'only_after_actual_complete_B24R1_primary_NO_PROMOTION'
         and proposal['protected_existing_state']['B26_opening_unchanged']['enabled'] is False,
         'old B25/B26 opening conditions are not satisfied or redefined by this evaluation')
    return value


def orchestration_contract():
    approval()
    proposal = json.loads((ROOT / PROPOSAL_PATH).read_bytes())
    raw = (ROOT / ORCHESTRATION_PATH).read_bytes()
    need(digest(raw) == ORCHESTRATION_SHA, 'fixed separately versioned successful-empty orchestration')
    value = json.loads(raw)
    expected = dict(schema='english-v6-b24q1-native-empty-quarantine-orchestration-v1',experiment_id='B24Q1',
        approved_proposal_sha256=PROPOSAL_SHA,
        **{k:proposal[k] for k in ('native_empty_boundary','quarantine_rule','return_dispositions','complete_denominator_and_readback')},
        fixed_plan={k:v for k,v in proposal['fixed_execution_plan_if_separately_approved'].items() if not k.startswith('future_')},
        native_payload_validity_and_numerical_admission_unchanged=True,
        old_whole_panel_empty_return_policy_changed=True,old_B24_and_B24R1_STOPs_remain_terminal=True,
        automatic_next_assessment_allowed=False)
    need(value == expected, 'only the exact explicitly approved orchestration amendment')
    return value


def inputs():
    need(digest((ROOT / PREVIOUS).read_bytes()) == PREVIOUS_SHA, 'unchanged complete R1 metadata validator')
    spec = importlib.util.spec_from_file_location('b24q1_original_r1_metadata', ROOT / PREVIOUS)
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    hashes = dict(previous.validate()['hashes'])
    base = previous.inputs()[0]
    old_raw = (previous.PANELS / 'assessment.json').read_bytes()
    history_raw = (previous.PANELS / 'ancestry.json').read_bytes()
    need(digest(old_raw) == PREVIOUS_ASSESSMENT_SHA and digest(history_raw) == SOURCE_MEMBERS['HISTORY.json'],
         'immutable R1 assessment and every prior exclusion')
    old, history = json.loads(old_raw), json.loads(history_raw)
    extra = {}
    for name, expected in EXTRA_INPUTS.items():
        raw = (ROOT / name).read_bytes()
        need(digest(raw) == expected, 'unaltered approved Q1 metadata input: ' + name)
        hashes[name] = expected; extra[name] = json.loads(raw)
    exposure, origin = extra[MANIFEST], extra[ORIGIN]
    need(origin == _origin_document(exposure, origin['source_api_identity']), 'minimal exact original R1 source origin')
    fields = ['id','cohort','source_group_id','pcm_sha256','canonical_person']
    need(set(exposure) == {'schema','original_source_schema','panel','selected_original_slots','source_artifact',
         'source_execution','assessment_sha256','source_ancestry_sha256','history_receipt_sha256','original_metadata_sha256',
         'new_kws_forwards','replication_opened','rows','metadata_only','selection_fields',
         'primary_people','sealed_replication_people','all_excluded_people'}
         and exposure['schema'] == 'english-v6-b24q1-authenticated-b24r1-exposure-v1'
         and exposure['original_source_schema'] == 'english-v6-b24r1-fresh-combined-source-v1'
         and exposure['panel'] == 'primary' and exposure['selected_original_slots'] == len(exposure['rows']) == 120
         and exposure['source_artifact'] == SOURCE_PIN and exposure['original_metadata_sha256'] == SOURCE_MEMBERS
         and exposure['source_ancestry_sha256'] == exposure['history_receipt_sha256'] == old['ancestry_sha256'] == digest(history_raw)
         and exposure['assessment_sha256'] == PREVIOUS_ASSESSMENT_SHA
         and exposure['new_kws_forwards'] == 0 and exposure['replication_opened'] is False
         and exposure['metadata_only'] is True and exposure['selection_fields'] == fields
         and all(set(r) == set(fields) and r['canonical_person'] == base.source_person(r) for r in exposure['rows']),
         'all original exposure identities only; no labels, observations or numerical KWS enter selection')
    need(exposure['source_execution'] == dict(prepared_parent=SOURCE_PARENT,run_attempt=1,
         run_branch='refs/heads/'+SOURCE_BRANCH,run_head=SOURCE_PIN['run_head'],run_id=str(SOURCE_PIN['run_id'])),
         'exact final R1 source identity')
    people = original_assessment_people()
    need(exposure['primary_people'] == people['primary'] and exposure['sealed_replication_people'] == people['replication']
         and exposure['all_excluded_people'] == base.ordered(set(people['primary'])|set(people['replication'])),
         'all 36 exposed and 36 never-opened R1 people excluded')
    permission = approval(); orchestration_contract()
    evaluation = extra[EVALUATION_DESIGN_PATH]
    need(evaluation['schema'] == 'english-v6-b24q1-approved-empty-quarantine-evaluation-v1'
         and evaluation['approval'] == dict(path=APPROVAL_PATH,sha256=APPROVAL_SHA)
         and evaluation['approved_proposal'] == dict(path=PROPOSAL_PATH,sha256=PROPOSAL_SHA)
         and evaluation['orchestration_contract'] == dict(path=ORCHESTRATION_PATH,sha256=ORCHESTRATION_SHA)
         and evaluation['training_design_sha256'] == DESIGN_SHA and evaluation['training_assessment_sha256'] == TRAINING_ASSESSMENT_SHA
         and evaluation['frozen_model_pt_sha256'] == FROZEN_MODELS
         and evaluation['evaluation_only'] is True and evaluation['new_optimizer_updates'] == 0
         and evaluation['metadata_selection_authorized'] is permission['scope']['metadata_selection_authorized'] is True
         and evaluation['original_whole_panel_empty_return_policy_changed'] is True
         and evaluation['original_native_validity_and_numerical_admission_unchanged'] is True
         and all(evaluation[k] == old[k] for k in ('source_gate','gains','quality_assessment_sha256'))
         and evaluation['quality_function_sha256'] == old['quality_criteria']['quality_function_sha256']
         and all(evaluation[k] is False for k in ('execution_authorized','source_execution_authorized','primary_scoring_authorized',
             'replica_opening_authorized','engineering_authorized','optimizer_allowed','promotion_allowed','automatic_R2_allowed')),
         'new approved orchestration and honest post-training evaluation; no changed weights, numerical gates or immediate execution')
    return base, old, history, exposure, dict(SOURCE_PIN), hashes


def build_documents():
    base, old, history, manifest, pin, hashes = inputs()
    rows = manifest['rows']
    exposed = {base.source_person(r) for r in rows}
    def people(role):
        return {base.canonical(p) for p in old[role]['synthetic']['speaker_id_map'].values()} | set(old[role]['natural']['speakers'])
    reserved = people('replication')
    need(exposed == people('primary') and len(exposed) == len(reserved) == 36 and not exposed & reserved
         and len({r['id'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == 120
         and len({r['source_group_id'] for r in rows}) == 36
         and sum(r['cohort']=='synthetic' for r in rows)==96 and sum(r['cohort']=='natural' for r in rows)==24,
         'complete original R1 120 identities and all 72 exposed/reserved people')
    newly_excluded = exposed | reserved
    need(newly_excluded == set(manifest['all_excluded_people']), 'no original R1 reserve reclaimed')
    synth_blocked = set(history['synthetic_excluded_people']) | newly_excluded
    natural_blocked = set(history['natural_excluded_people']) | newly_excluded
    need(not set(history['all_pcm_sha256']) & {r['pcm_sha256'] for r in rows}, '120 new R1 original PCM identities really distinct')
    ancestry = copy.deepcopy(history)
    ancestry.update(schema='english-v6-b24q1-complete-exposure-and-sealed-reserves-v1',
        inherited_history_sha256=old['ancestry_sha256'],inherited_assessment_sha256=PREVIOUS_ASSESSMENT_SHA,
        original_B24R1_source=pin,B24R1_manifest_sha256=SOURCE_MEMBERS['MANIFEST.json'],B24R1_exposure_sha256=EXTRA_INPUTS[MANIFEST],
        all_pcm_sha256=sorted(set(history['all_pcm_sha256'])|{r['pcm_sha256'] for r in rows}),
        all_source_groups=sorted(set(history['all_source_groups'])|{r['source_group_id'] for r in rows}),
        canonical_people=base.ordered(set(history['canonical_people'])|exposed),
        synthetic_excluded_people=base.ordered(synth_blocked),natural_excluded_people=base.ordered(natural_blocked),
        B24R1_exposed_primary_people=base.ordered(exposed),B24R1_sealed_replication_people=base.ordered(reserved),
        all_B24R1_120_raw_originals_added=True,all_prior_replication_sealed=True,
        B24R1_quality_results_read=False,B24R1_original_source_stop_preserved=True,old_B24R1_replication_opened=False,
        automatic_next_assessment_allowed=False,B24Q1_execution_authorized=False,metadata_input_sha256=dict(hashes))
    need(len(history['all_pcm_sha256']) == 2823 and len(ancestry['all_pcm_sha256']) == 2943,
         'actual complete 2823 prior plus all 120 R1 original PCM identities; not a guessed count')
    catalog_inputs = base.read_inputs()
    mapping = catalog_inputs['piper-config.json']['speaker_id_map']
    by_index = {index: person for person, index in mapping.items()}
    available_synth = {base.canonical(person): index for index, person in by_index.items()
                       if base.canonical(person) not in synth_blocked}
    need({base.canonical(p) for p in mapping} <= natural_blocked,
         'whole potentially overlapping Piper population remains excluded from natural selection')
    census, eligible = base.catalog_rows(catalog_inputs['FULL_METADATA_CATALOG.json'])
    available_natural = census - natural_blocked
    usable_natural = {person for person in available_natural if len(eligible[person]) >= 2}
    need(len(available_synth) >= 48 and len(usable_natural) >= 24, 'two complete independent B24Q1 panels remain available')
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
            seed_base=20263700 if role == 'primary' else 20263800), natural=natural)
    need(len(selected) == 72 and not selected & newly_excluded, 'all72 new canonical people independent of all B24R1 primary and reserve people')
    assessment = copy.deepcopy(old)
    assessment.pop('original_B24_SOURCE_STOP_proof_required_before_any_new_budget')
    assessment.pop('primary_scoring_requires_frozen_actual_B24_weights_and_actual_source_gate_pass')
    assessment.pop('R1_source_failure_terminal')
    assessment.update(schema='english-v6-b24q1-approved-empty-quarantine-evaluation-v1',
        purpose='One explicitly approved independent primary and sealed replica of the same B24 weights, with only proved-successful-empty whole-original quarantine; fresh metadata selected after training, before new audio and KWS.',
        ancestry_sha256=digest(encoded(ancestry)),metadata_validator_sha256=digest(Path(__file__).read_bytes()),
        metadata_inputs_sha256=dict(hashes),primary=panels['primary'],replication=panels['replication'],
        evaluation_design_path=EVALUATION_DESIGN_PATH,evaluation_design_sha256=EVALUATION_DESIGN_SHA,
        approved_proposal=dict(path=PROPOSAL_PATH,sha256=PROPOSAL_SHA),approval=dict(path=APPROVAL_PATH,sha256=APPROVAL_SHA),
        metadata_selection_authorized=True,source_orchestration_contract=dict(path=ORCHESTRATION_PATH,sha256=ORCHESTRATION_SHA),
        observer_scientific_contract_scope='Unchanged native return, payload validity and numerical admission primitives only; the full observer orchestration policy has changed explicitly.',
        original_whole_panel_empty_return_policy_changed=True,original_native_validity_and_numerical_admission_unchanged=True,
        source_observer_denominator='N=96+unchanged_preobserver_eligible_natural; exact 5*N planned calls <=600, retaining all120 exposure identities',
        proved_successful_empty_original_quarantined_without_vote=True,pre_observer_rejects_reported_separately=True,
        all_other_native_faults_whole_panel_stop=True,all_five_valid_nonempty_payloads_required_for_admission=True,
        original_B24R1_SOURCE_STOP_proof_required_before_any_new_budget=True,
        B24R1_quality_results_read=False,B24R1_original_source_stop_preserved=True,old_B24R1_replication_opened=False,
        old_B26_opening_condition_unchanged=True,B26_opener='only_after_actual_complete_B24R1_primary_NO_PROMOTION',
        automatic_next_assessment_allowed=False,Q1_source_failure_terminal=True,
        primary_scoring_requires_frozen_actual_B24_weights_and_new_Q1_source_gate_pass=True,
        quality_control_role='Exact original B23 challenger as immutable historical control; no same-job matched rerun and no changed numerical B14 decision.',
        promotion_prerequisites='Fresh B24Q1 primary PASS and untouched B24Q1 replica PASS under the same approved policy and identical fixed weights, then unchanged engineering, physical SSC305 and field evidence requirements.',
        replication_opening='Only complete authenticated numerical PASS of this new primary opens its prospectively frozen replica; no old reserve or automatic next panel.',
        metadata_selection=dict(algorithm=old['metadata_selection']['algorithm'],salts=SALTS,corpus_census=251,
            eligible_synthetic_voices=len(available_synth),available_natural_census=len(available_natural),
            eligible_natural_readers=len(usable_natural),synthetic_excluded_canonical_people=len(synth_blocked),
            natural_excluded_canonical_people=len(natural_blocked),metadata_only=True,selection_before_audio=True,
            selection_before_optimizer=False,selection_after_training=True,selection_before_first_independent_KWS=True,
            source_replacements=0,reserves_reclaimed=0,cross_source_type_identity_exclusion=True))
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
                planned.append(dict(id=f'b24q1-{role}-{index}-{cell}', speaker_index=index, source_speaker_id=speaker,
                    source_group_id='piper-libritts-' + speaker, role='development', panel=role, cell_id=cell,
                    cohort='synthetic', length_scale=scale, seed=synthetic['seed_base'] + ordinal,
                    requested_class=cls, requested_text=text, native_input=native, observed_class=None, training_allowed=False))
        documents[role + '.json'] = dict(schema='english-v6-b24q1-fresh-combined-source-v1', panel=role,
            immutable_assessment_sha256=digest(encoded(assessment)), source_ancestry_sha256=digest(encoded(ancestry)),
            model=base.PIPER, config_sha256=base.INPUTS['piper-config.json'], max_source_calls=96,
            maximum_native_observer_calls=480, original_slots=96, expected_providers=base.PROVIDERS,
            source_format=old['source_format'], source_admission_rule='unchanged B14 five-valid native admission; approved whole-original invalid-empty quarantine orchestration',
            orchestration_contract_sha256=ORCHESTRATION_SHA,
            existing_replication_opened=False, training_allowed=False, promotion_allowed=False, rows=planned)
        documents['natural-' + role + '.json'] = panels[role]['natural']
    need(set(documents) == set(OUTPUT_NAMES), 'all and only the six prospective frozen documents')
    return documents, hashes


def validate():
    documents, hashes = build_documents()
    for name, value in documents.items():
        raw = encoded(value)
        need((PANELS / name).read_bytes() == raw, 'complete deterministic frozen B24Q1 metadata: '+name)
        hashes[(PANELS / name).relative_to(ROOT).as_posix()] = digest(raw)
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__).read_bytes())
    return dict(schema='english-v6-b24q1-offline-metadata-validation-v1',status='PASS_METADATA_ONLY',hashes=hashes,
        selection=documents['assessment.json']['metadata_selection'],approval_sha256=APPROVAL_SHA,
        approved_proposal_sha256=PROPOSAL_SHA,orchestration_contract_sha256=ORCHESTRATION_SHA,
        all_B24R1_120_originals_excluded=True,all_B24R1_72_people_excluded=True,
        historical_PCM_identities_excluded=2943,quality_results_read_for_selection=False,
        evaluation_only=True,training_assessment_sha256=TRAINING_ASSESSMENT_SHA,training_design_sha256=DESIGN_SHA,
        frozen_model_pt_sha256=dict(FROZEN_MODELS),old_B24_and_R1_source_stops_permanent=True,
        original_whole_panel_empty_return_policy_changed=True,original_native_validity_and_numerical_admission_unchanged=True,
        old_B25_and_B26_openers_unchanged=True,automatic_next_assessment_allowed=False,
        source_calls=0,observer_calls=0,kws_forwards=0,optimizer_updates=0,
        source_execution_authorized=False,primary_scoring_authorized=False,replica_opening_authorized=False,
        engineering_authorized=False,promotion_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='one initial metadata freeze; never redraw or overwrite outputs')
    args=parser.parse_args()
    if args.freeze:
        need(not any((PANELS/n).exists() for n in OUTPUT_NAMES),'freeze only once; never redraw selected panels')
        documents,_=build_documents();PANELS.mkdir(exist_ok=True)
        for name,value in documents.items():(PANELS/name).write_bytes(encoded(value))
    print(json.dumps(validate(),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
