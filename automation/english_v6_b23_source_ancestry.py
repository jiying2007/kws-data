"""B23 source authorization, exact fresh ancestry and separately consumed budgets.

The completed original TRAIN proof is required before any new source call. It
retains original B22 failure receipts; old source admission was independently
verified at TRAIN entry and is not represented as repeated here.
"""
from __future__ import annotations
import english_v6_b22_source_ancestry as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b23_source_metadata as metadata

PLANS = metadata.PANELS
ASSESSMENT = PLANS / 'assessment.json'
ASSESSMENT_SHA = 'a5923fa64969ffa5d550fcdd320fd1c8d8aa68a9430f6de03fe5eab41b24e8ae'
ANCESTRY_SHA = '91e10721b8d936496db240ce4b17c91f0ad603b57065c43141e9d4ec5d20874c'
INPUTS = Path(__file__).with_name('english_v6_b23_source_inputs.json')
TRAIN_PRODUCER = Path(__file__).with_name('english_v6_b23_source_training_producer.json')
BRANCH = 'refs/heads/research/en-v6-b23-primary-sources-once-20261124'
WORKFLOW = '.github/workflows/english-v6-b23-primary-sources-once.yml'
SCHEMA = 'english-v6-b23-complete-source-authorization-v1'
MARKER = 'english-v6-b23-primary-source-consumed'
TRAIN_BRANCH = 'research/en-v6-b23-prefix-peak-train-once-20261124'
TRAIN_WORKFLOW = '.github/workflows/english-v6-b23-prefix-peak-train-once.yml'
original_validate_training_live = validate_training_live


def training_binding(require_bound=True):
    value = json.loads(INPUTS.read_bytes())
    need(set(value) == {'schema', 'assessment_sha256', 'bound', 'training', 'training_producer_manifest_sha256'}
         and value['schema'] == 'english-v6-b23-completed-train-source-binding-v1'
         and value['assessment_sha256'] == ASSESSMENT_SHA,
         'same B23 assessment frozen before optimizer and all new source draws')
    if value['bound'] is False:
        need(value['training'] is value['training_producer_manifest_sha256'] is None and not require_bound,
             'actual completed B23 training must be bound before source authorization')
        return value
    need(value['bound'] is True and type(value['training']) is dict
         and re.fullmatch(r'[0-9a-f]{64}', value.get('training_producer_manifest_sha256') or '')
         and digest(TRAIN_PRODUCER.read_bytes()) == value['training_producer_manifest_sha256'],
         'complete actual successful B23 producer is frozen before source capture')
    manifest = json.loads(TRAIN_PRODUCER.read_bytes())
    need(manifest['schema'] == 'english-v6-b23-complete-git-producer-v1'
         and type(manifest['files']) is dict and len(manifest['files']) > 264,
         'all264 historical TRAIN dependencies and complete explicit B23 additions')
    t = value['training']
    need(set(t) == {'run_id', 'run_head', 'prepared_parent', 'run_branch', 'runtime_workflow',
                   'artifacts', 'result_sha256', 'model_pt_sha256'}
         and type(t['run_id']) is int and t['run_id'] > 0
         and all(re.fullmatch(r'[0-9a-f]{40}', t[k]) for k in ('run_head', 'prepared_parent'))
         and t['run_head'] != t['prepared_parent'] and re.fullmatch(r'[0-9a-f]{64}', t['result_sha256'])
         and t['run_branch'] == TRAIN_BRANCH and t['runtime_workflow'] == TRAIN_WORKFLOW
         and set(t['artifacts']) == {'models', 'producer', 'receipts', 'consumed'},
         'one original prefix-peak TRAIN run and its four exact artifact identities')
    names = dict(models='english-v6-b23-prefix-peak-models-' + str(t['run_id']),
        producer='english-v6-b23-train-producer-' + str(t['run_id']),
        receipts='english-v6-b23-original-input-receipts-' + str(t['run_id']),
        consumed='english-v6-b23-prefix-peak-train-consumed')
    for role, pin in t['artifacts'].items():
        need(set(pin) == {'id', 'bytes', 'sha256', 'name'} and type(pin['id']) is int and pin['id'] > 0
             and type(pin['bytes']) is int and pin['bytes'] > 0
             and re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) and pin['name'] == names[role],
             'immutable original named training artifact ' + role)
    expected = read_plan()['training']
    hashes = t['model_pt_sha256']
    need(set(hashes) == {'B0', 'B9', 'control', 'challenger'} and len(set(hashes.values())) == 4
         and hashes['B0'] == expected['shared_parent_pt_sha256']
         and hashes['B9'] == '7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655'
         and hashes['control'] == expected['control_pt_sha256']
         and all(re.fullmatch(r'[0-9a-f]{64}', v) for v in hashes.values()),
         'same original B0/B9 and immutable B22 control; actual distinct trained B23 challenger')
    return value


def validate_training_live(live):
    value = original_validate_training_live(live)
    commit, tree = live['commit'], live['tree']
    need(commit['sha'] == value['run_head'] and len(commit['parents']) == 1
         and commit['parents'][0]['sha'] == value['prepared_parent']
         and [entry['filename'] for entry in commit['files']] == [TRAIN_WORKFLOW]
         and tree['sha'] == commit['commit']['tree']['sha'] and tree['truncated'] is False,
         'actual public TRAIN workflow-only child and complete original Git tree')
    return value


def verify_training(directory):
    from english_v6_b23_source_training_proof import verify
    return verify(Path(directory))


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    t = training_binding()['training']
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh B23 source authorization')
    out.mkdir()
    commit = json.loads(gh('commits/' + t['run_head']))
    live = dict(run=json.loads(gh('actions/runs/' + str(t['run_id']))),
        artifacts=json.loads(gh('actions/runs/' + str(t['run_id']) + '/artifacts?per_page=100')),
        commit=commit, tree=json.loads(gh('git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1')))
    validate_training_live(live)
    qa.record(out / 'TRAIN_LIVE.json', live)
    for name, pin in t['artifacts'].items():
        raw = gh('actions/artifacts/' + str(pin['id']) + '/zip')
        need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'original TRAIN evidence download ' + name)
        (out / ('TRAIN_' + name + '.zip')).write_bytes(raw)
    verify_training(out)
    qa.record(out / 'HISTORY.json', read_ancestry())
    qa.record(out / 'AUTHORIZATION.json', dict(schema=SCHEMA, source_execution=identity,
        assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        training_binding_sha256=training_binding_sha(), train_result_sha256=t['result_sha256'],
        train_live_sha256=qa.sha(out / 'TRAIN_LIVE.json'), history_sha256=qa.sha(out / 'HISTORY.json'),
        transport_plan_sha256=transport_plan_sha(), maximum_tts_calls=96, maximum_natural_decodes=24,
        maximum_native_observer_calls=600, replication_audio_opened=False, training_allowed=False,
        promotion_allowed=False, new_kws_forwards=0))
    return require_authorization(out)


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, 'registered native provider only')
    return MARKER if provider is None else 'english-v6-b23-primary-observer-' + provider + '-consumed'

def consumption_document(directory):
    from english_v6_b23_source_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    return dict(schema=SCHEMA + '-consumption', marker_name=MARKER, source_execution=permission['source_execution'],
        authorization_sha256=qa.sha(directory / 'AUTHORIZATION.json'), corpus_bytes=CORPUS_BYTES,
        corpus_sha256=CORPUS_SHA, transport_receipt_sha256=qa.sha(directory / 'CORPUS_TRANSPORT.json'),
        status='TTS96_NATURAL24_OBSERVER600_CONSUMED_UNKNOWN', maximum_new_tts_calls=96,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)

def consumption(directory, corpus, prepared_parent):
    from english_v6_b23_source_transport import verify_transport
    identity = execution_guard(prepared_parent)
    need(require_authorization(directory)['source_execution'] == identity, 'same new primary source identity')
    verify_transport(corpus, directory)
    one_execution(identity, False)
    path = Path(directory) / 'CONSUMPTION.json'
    need(not path.exists(), 'single original source budget')
    value = consumption_document(directory)
    qa.record(path, value)
    return value

def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b23_source_transport import verify_transport
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected['source_execution'] == identity and json.loads((directory / 'CONSUMPTION.json').read_bytes()) == expected,
         'same durably consumed source budget before TTS or decode')
    verify_transport(corpus, directory)
    receipt = uploaded_marker(one_execution(identity, True), expected, 'CONSUMPTION.json')
    need(not (directory / 'CONSUMPTION_UPLOAD.json').exists(), 'one source entry, no retry')
    qa.record(directory / 'CONSUMPTION_UPLOAD.json', receipt)
    return expected

def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b23_source as owner
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[k] == value for k, value in identity.items()), 'same new source before observer consumption')
    one_execution(identity, True)
    one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), 'fresh individual observer budget')
    out.mkdir()
    value = observer_document(provider, registration)
    qa.record(out / 'OBSERVER_CONSUMPTION.json', value)
    return value
