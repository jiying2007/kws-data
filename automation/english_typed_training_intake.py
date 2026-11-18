"""Pure checks between authenticated source admission and typed head training.

No policy, campaign, artifact or numerical job is enabled by this module.
The future worker must authenticate the actual source artifact, execute its
frozen admission policy on the saved evidence, and pass that independently
recomputed admission here. `expected_binding_sha256` comes from the reviewed
worker plan, never from the input packet. Equality to a caller-supplied digest
alone is not source authentication.

This layer checks the complete planned denominator, all-role source isolation,
admission semantics and frozen request/PCM exclusions before feature calls.
It neither labels audio nor turns context64 probe results into training data.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

SCHEMA = 'english-typed-training-intake-binding-v1'
IDENTITY_SHA256 = '3ba692d3bbf0777d326dd8f6a4b44d7b80b1357cd613599850b6c80e4bc2fc8d'
HEX64 = re.compile(r'[0-9a-f]{64}')
HEX40 = re.compile(r'[0-9a-f]{40}')
TIERS = {'silver': 1.0, 'weak': 0.5, 'context_supported_weak': 0.5}
EXCLUSION_KEYS = ('row_ids', 'request_sha256', 'pcm_sha256', 'parent_pcm_sha256')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _identity_module():
    path = Path(__file__).with_name('english_source_identity.py')
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'REGULAR_PINNED_IDENTITY_HELPER')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == IDENTITY_SHA256, 'PINNED_IDENTITY_BEFORE_IMPORT')
    spec = importlib.util.spec_from_file_location('typed_intake_identity', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_identity = _identity_module()


def validate_binding(binding, expected_binding_sha256):
    """The expected digest must be external to the received source packet."""
    need(type(expected_binding_sha256) is str and HEX64.fullmatch(expected_binding_sha256)
         and type(binding) is dict and digest(binding) == expected_binding_sha256,
         'EXTERNAL_FROZEN_INTAKE_BINDING_REQUIRED')
    fields = {'schema', 'campaign_id', 'producer_run', 'source_artifact',
        'planned_row_ids', 'source_identity_contract_sha256', 'corpus_sha256',
        'admission_sha256', 'machine_evidence_sha256', 'admission_policy_sha256',
        'legacy_libri_ids', 'exclusion_evidence_sha256', 'excluded_identities',
        'parent_context_weak_policy'}
    need(set(binding) == fields and binding['schema'] == SCHEMA,
         'EXACT_FUTURE_TRAINING_INTAKE_BINDING')
    need(type(binding['campaign_id']) is str and binding['campaign_id']
         and binding['campaign_id'] not in source_identity.PROBE_CAMPAIGNS,
         'FUTURE_NONPROBE_SOURCE_CAMPAIGN')
    for field in ('source_identity_contract_sha256', 'corpus_sha256', 'admission_sha256',
                  'machine_evidence_sha256', 'admission_policy_sha256', 'exclusion_evidence_sha256'):
        need(type(binding[field]) is str and HEX64.fullmatch(binding[field]), 'BOUND_DIGEST:' + field)
    need(type(binding['parent_context_weak_policy']) is bool, 'EXPLICIT_FUTURE_EVIDENCE_SCOPE')
    run = binding['producer_run']
    need(type(run) is dict and set(run) == {'repository', 'run_id', 'commit', 'run_attempt', 'event'}
         and run['repository'] == 'jiying2007/kws-data' and type(run['run_id']) is int and run['run_id'] > 0
         and type(run['commit']) is str and HEX40.fullmatch(run['commit'])
         and type(run['run_attempt']) is int and run['run_attempt'] == 1 and run['event'] == 'push',
         'ACTUAL_SOURCE_EXECUTION_IDENTITY')
    artifact = binding['source_artifact']
    need(type(artifact) is dict and set(artifact) == {'id', 'name', 'bytes', 'sha256'}
         and type(artifact['id']) is int and artifact['id'] > 0
         and type(artifact['name']) is str and bool(artifact['name'])
         and type(artifact['bytes']) is int and 0 < artifact['bytes'] <= 512 * 1024**2
         and type(artifact['sha256']) is str and HEX64.fullmatch(artifact['sha256']),
         'ACTUAL_SOURCE_ARTIFACT_IDENTITY')
    ids = binding['planned_row_ids']
    need(type(ids) is list and 0 < len(ids) <= 192 and all(type(x) is str and x for x in ids)
         and len(set(ids)) == len(ids), 'EXACT_PLANNED_DENOMINATOR')
    legacy = binding['legacy_libri_ids']
    need(type(legacy) is list and len(legacy) == len(set(legacy))
         and all(type(x) is str and re.fullmatch(r'0|[1-9][0-9]*', x) for x in legacy),
         'CANONICAL_LEGACY_SOURCE_EXCLUSIONS')
    exclusions = binding['excluded_identities']
    need(type(exclusions) is dict and set(exclusions) == set(EXCLUSION_KEYS),
         'EXACT_PROBE_EXCLUSION_IDENTITY_SETS')
    for key in EXCLUSION_KEYS:
        values = exclusions[key]
        need(type(values) is list and len(values) == len(set(values))
             and all(type(v) is str and v and (key == 'row_ids' or HEX64.fullmatch(v)) for v in values),
             'UNIQUE_PROBE_EXCLUSION:' + key)
    # A future Kokoro corpus cannot pretend that consumed probe identities did
    # not exist. The external binding authenticates the complete receipt/list.
    need(exclusions['request_sha256'] and exclusions['pcm_sha256']
         and exclusions['parent_pcm_sha256'], 'NONEMPTY_CONSUMED_PROBE_EXCLUSIONS')
    return binding


def _admission_semantics(row, allowed, *, parent_context_weak_policy):
    need(type(allowed.get('supervision_allowed')) is bool
         and type(allowed.get('training_allowed')) is bool, 'EXPLICIT_SUPERVISION_AND_TRAINING_FLAGS')
    supervised = allowed['supervision_allowed']
    tier, label, weight = allowed.get('tier'), allowed.get('research_class'), allowed.get('training_weight')
    need(type(weight) in (int, float) and not isinstance(weight, bool), 'FINITE_TIER_WEIGHT')
    if not supervised:
        need(tier == 'unknown' and label is None and allowed.get('training_class') is None
             and allowed['training_allowed'] is False and weight == 0.0,
             'UNKNOWN_IS_MASKED_NOT_BACKGROUND_OR_TRAINING')
        return
    need(row.get('generation_status') == 'success' and tier in TIERS
         and type(label) is int and label in (0, 1, 2), 'SUCCESSFUL_EXPLICIT_SUPERVISION')
    parent_scoped = allowed.get('phonetic_evidence_scope') == 'PARENT_FULL_INPUT_FIXED_CHILD_REGION'
    if tier == 'context_supported_weak':
        need(parent_context_weak_policy and parent_scoped
             and allowed.get('child_independent_phone_silver') is False
             and label in (1, 2), 'EXPLICIT_CONTEXT_WEAK_SCOPE_NO_SILVER_UPGRADE')
    else:
        need(not parent_scoped, 'PARENT_PHONE_CANNOT_IMPERSONATE_CHILD_INDEPENDENT_TIER')
    if row['role'] == 'train':
        need(allowed['training_allowed'] is True and type(allowed.get('training_class')) is int
             and allowed['training_class'] == label and weight == TIERS[tier], 'EXACT_TRAIN_TIER_SEMANTICS')
    else:
        need(allowed['training_allowed'] is False and allowed.get('training_class') is None
             and weight == 0.0, 'HELDOUT_NEVER_OPTIMIZER_INPUT')


def validate_packet(corpus, admission, identity_contract, machine_evidence,
                    recomputed_admission, binding, *, expected_binding_sha256,
                    normalize_libri):
    """Validate saved objects after the authenticated worker recomputes policy.

    Does not load a policy or verify an API/artifact receipt itself. Its caller
    owns those operations and must not substitute a copy of saved admission
    for actual policy recomputation. No numerical code is imported here.
    """
    validate_binding(binding, expected_binding_sha256)
    source_identity.require_training_corpus(corpus, admission)
    need(corpus.get('campaign_id') == corpus.get('source_design_id') == binding['campaign_id'],
         'EXACT_BOUND_FUTURE_CAMPAIGN')
    run = corpus.get('run', {})
    need(type(run) is dict and run.get('repository') == binding['producer_run']['repository']
         and str(run.get('run_id')) == str(binding['producer_run']['run_id'])
         and run.get('commit') == binding['producer_run']['commit']
         and str(run.get('run_attempt')) == '1' and run.get('event') == 'push', 'BOUND_SOURCE_RUN')
    for name, value in (('corpus', corpus), ('admission', admission),
                        ('machine_evidence', machine_evidence), ('source_identity_contract', identity_contract)):
        need(digest(value) == binding[name + '_sha256'], 'BOUND_PACKET_DIGEST:' + name)
    need(type(recomputed_admission) is dict and canonical(recomputed_admission) == canonical(admission),
         'ACTUAL_POLICY_RECOMPUTATION_MUST_MATCH')
    rows = corpus['rows']
    need([r['id'] for r in rows] == binding['planned_row_ids'], 'ALL_PLANNED_ROWS_IN_FROZEN_ORDER')
    need(type(machine_evidence) is dict and type(machine_evidence.get('rows')) is list
         and [r.get('id') for r in machine_evidence['rows']] == binding['planned_row_ids'],
         'COMPLETE_MACHINE_EVIDENCE_DENOMINATOR')
    proof = source_identity.validate_source_identities(corpus, admission, identity_contract,
        binding['legacy_libri_ids'], normalize_libri=normalize_libri)
    by_id = {r['id']: r for r in admission['rows']}
    deny = {k: set(v) for k, v in binding['excluded_identities'].items()}
    excluded_audio = deny['pcm_sha256'] | deny['parent_pcm_sha256']
    for row in rows:
        # Compare both request provenance and acoustic byte identities. Names,
        # caller labels and a different crop do not erase consumed provenance.
        need(row['id'] not in deny['row_ids'] and row.get('parent_id') not in deny['row_ids'],
             'CONSUMED_PROBE_ROW_OR_PARENT')
        for field in ('request_sha256', 'pcm_sha256', 'parent_pcm_sha256'):
            value = row.get(field)
            if value is not None:
                need(type(value) is str and HEX64.fullmatch(value), 'SOURCE_DIGEST:' + field)
                excluded = deny[field] if field == 'request_sha256' else excluded_audio
                need(value not in excluded, 'CONSUMED_PROBE_BYTES:' + field)
        need(row.get('request_sha256') is not None, 'EVERY_PLANNED_REQUEST_BOUND')
        if row.get('generation_status') == 'success':
            need(row.get('pcm_sha256') is not None, 'SUCCESSFUL_SOURCE_PCM_REQUIRED')
            if row['source_namespace'] == 'kokoro-v1_0-preset':
                need(row.get('parent_pcm_sha256') is not None, 'EXACT_PARENT_PCM_PROVENANCE')
        _admission_semantics(row, by_id[row['id']],
                             parent_context_weak_policy=binding['parent_context_weak_policy'])
    proof.update(intake_binding_sha256=expected_binding_sha256,
        admission_policy_sha256=binding['admission_policy_sha256'],
        source_artifact=binding['source_artifact'], producer_run=binding['producer_run'],
        planned_row_count=len(rows), all_planned_rows_checked=True,
        probe_reuse_allowed=False, actual_model_calls=0,
        future_worker_must_authenticate_artifact_and_recompute_policy=True)
    return proof
