"""Source-only candidate for typed synthetic-source isolation.

This module does not authenticate a source artifact or label audio. A future
worker must first bind the actual corpus, its recomputed admission, this module
and SOURCE_IDENTITY_CONTRACT.json to an independently frozen source contract.
No current corpus/plan/workflow adopts this candidate.

Contract v1 has exactly: schema, purpose, diagnostic_only, training_allowed,
human_speaker_identity_claimed, source_namespaces, known_lineage_edges and
train_side_exposed. Each edge has exactly left, right, relation, evidence_source
and evidence_sha256. Endpoints/exposures are canonical namespace:source_id keys.
Only explicitly evidenced edges connect sources; asset hashes/revisions do not.
Corpus and admission must bind the canonical SHA-256 of that same contract.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import re

SCHEMA = 'english-source-identity-contract-v1'
NAMESPACES = frozenset(('libri-speaker', 'vctk-speaker', 'kokoro-v1_0-preset'))
ROLES = frozenset(('train', 'development', 'calibration', 'locked-test'))
PROBE_CAMPAIGN = 'kokoro-v1_0-four-presets-short-carrier48-v1'
PROBE_CAMPAIGNS = frozenset((PROBE_CAMPAIGN,
    'kokoro-v1_0-original-carrier-full-context32-v1',
    'kokoro-full-context32-native-int32-first-forward-v1',
    'kokoro-exact-child-context-controls64-v1'))
PROBE_SCHEMA_PREFIXES = ('english-kokoro48-', 'english-kokoro-context32-',
    'english-kokoro-context64-')
PERMANENT_EXCLUSION_FIELDS = ('permanently_excluded_from_training',
    'all_probe_inputs_permanently_excluded_from_training')
PROBE_TRAIN_SIDE = frozenset('kokoro-v1_0-preset:' + name for name in
    ('af_bella', 'af_sarah', 'am_michael', 'am_puck'))
# The project already treats this as a mixture. No parent recipe is disclosed
# in the inspected fixed-revision docs, so none is invented here. This candidate
# cannot count it or an explicitly linked component as an independent source.
UNRESOLVED_MIXTURES = frozenset(('kokoro-v1_0-preset:af_heart',))
METADATA_FIELDS = ('source_namespace', 'source_identity_key', 'preset_id',
    'known_lineage', 'evaluation_scope', 'human_speaker_identity',
    'human_speaker_identity_verified')


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_key(namespace, source_id, normalize_libri):
    """The namespace, never the shape of a bare pNNN string, selects the corpus."""
    need(type(namespace) is str and namespace in NAMESPACES and type(source_id) is str and source_id,
         'EXPLICIT_SUPPORTED_SOURCE_NAMESPACE_AND_ID')
    if namespace == 'libri-speaker':
        ident = normalize_libri(source_id)
    elif namespace == 'vctk-speaker':
        match = re.fullmatch(r'p?([0-9]+)', source_id)
        need(match is not None, 'KNOWN_VCTK_NUMERIC_IDENTITY')
        ident = str(int(match.group(1)))
    else:
        need(re.fullmatch(r'[ab][fm]_[a-z]+', source_id) is not None,
             'KOKORO_ENGLISH_PRESET_NOT_NUMERIC_SPEAKER')
        ident = source_id
    return namespace + ':' + ident


def checked_key(value, normalize_libri):
    need(type(value) is str and ':' in value, 'TYPED_LINEAGE_KEY')
    namespace, source_id = value.split(':', 1)
    need(source_key(namespace, source_id, normalize_libri) == value,
         'CANONICAL_LINEAGE_KEY')
    return value


def metadata(row):
    """Preserve typed source semantics in events and final case/reference rows."""
    need(all(k in row for k in METADATA_FIELDS), 'COMPLETE_SOURCE_IDENTITY_METADATA')
    return copy.deepcopy({k: row[k] for k in METADATA_FIELDS})


def require_training_corpus(corpus, admission):
    """Consumed diagnostic campaigns never become eligible training inputs.

    This semantic check complements, and cannot replace, externally pinned
    request/PCM exclusions and authenticated source-policy recomputation.
    """
    need(type(corpus) is dict and type(admission) is dict
        and corpus.get('campaign_id') not in PROBE_CAMPAIGNS
        and corpus.get('source_design_id') not in PROBE_CAMPAIGNS
        and not str(corpus.get('schema', '')).startswith(PROBE_SCHEMA_PREFIXES)
        and not str(admission.get('schema', '')).startswith(PROBE_SCHEMA_PREFIXES)
        and not any(corpus.get(k) is True or admission.get(k) is True
                    for k in PERMANENT_EXCLUSION_FIELDS)
        and corpus.get('diagnostic_only') is False and corpus.get('training_allowed') is True
        and corpus.get('auto_admission_allowed') is not False
        and admission.get('training_snapshot_allowed') is True, 'NO_DIAGNOSTIC_PROBE_TRAINING')
    for packet in (corpus, admission):
        rows = packet.get('rows')
        need(type(rows) is list and all(type(row) is dict for row in rows),
             'COMPLETE_IDENTITY_ROSTERS')
        need(all(not any(row.get(k) is True for k in PERMANENT_EXCLUSION_FIELDS)
                 and row.get('campaign_id') not in PROBE_CAMPAIGNS
                 and row.get('source_campaign_id') not in PROBE_CAMPAIGNS
                 for row in rows), 'PERMANENT_DIAGNOSTIC_ROW_EXCLUSION')


def validate_source_identities(corpus, admission, contract, legacy_libri_ids, *, normalize_libri):
    """Reject exposure/split/group inflation before new feature-model calls.

All corpus rows participate, including uncertain/masked rows. One connected
known-lineage component must have one role and one consumed source_group_id;
the group must equal the lexicographically first canonical typed member of
that component. A producer cannot alias a fresh source to an old replay group.
One group cannot hide several disconnected components. A component is an
operational source grouping, not a claim about an independent human speaker.
"""
    required = {'schema', 'purpose', 'diagnostic_only', 'training_allowed',
        'human_speaker_identity_claimed', 'source_namespaces',
        'known_lineage_edges', 'train_side_exposed'}
    need(type(contract) is dict and set(contract) == required and contract['schema'] == SCHEMA
        and contract['purpose'] == 'synthetic_training_source_identity'
        and contract['diagnostic_only'] is False and contract['training_allowed'] is True
        and contract['human_speaker_identity_claimed'] is False, 'FROZEN_TRAINING_IDENTITY_CONTRACT')
    namespaces = contract['source_namespaces']
    need(type(namespaces) is list and namespaces
        and all(type(n) is str and n in NAMESPACES for n in namespaces)
        and len(set(namespaces)) == len(namespaces), 'DECLARED_SOURCE_NAMESPACES')
    require_training_corpus(corpus, admission)
    contract_sha = digest(contract)
    need(corpus.get('source_identity_contract_sha256') == contract_sha
        and admission.get('source_identity_contract_sha256') == contract_sha,
        'CORPUS_ADMISSION_SAME_IDENTITY_CONTRACT')
    rows, admitted = corpus.get('rows'), admission.get('rows')
    need(type(rows) is list and rows and type(admitted) is list
        and all(type(r) is dict and type(r.get('id')) is str and r['id'] for r in rows + admitted),
        'COMPLETE_IDENTITY_ROSTERS')
    by_id = {r['id']: r for r in admitted}
    need(len(by_id) == len(admitted) == len(rows) == len({r['id'] for r in rows})
        and set(by_id) == {r['id'] for r in rows}, 'IDENTITY_ROSTER_BIJECTION')

    parent = {}
    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def join(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    edges = contract['known_lineage_edges']
    need(type(edges) is list, 'DECLARED_LINEAGE_EDGE_LIST')
    seen_edges = set()
    for edge in edges:
        need(type(edge) is dict and set(edge) == {'left', 'right', 'relation', 'evidence_source', 'evidence_sha256'},
             'EXPLICIT_LINEAGE_EDGE_EVIDENCE')
        left, right = (checked_key(edge[k], normalize_libri) for k in ('left', 'right'))
        need(left != right and edge['relation'] in ('mix_parent', 'same_source_alias', 'shared_lineage')
            and type(edge['evidence_source']) is str and bool(edge['evidence_source'])
            and type(edge['evidence_sha256']) is str
            and re.fullmatch(r'[0-9a-f]{64}', edge['evidence_sha256']) is not None,
            'KNOWN_RELATION_REQUIRES_FROZEN_EVIDENCE')
        edge_key = tuple(sorted((left, right)))
        need(edge_key not in seen_edges, 'UNIQUE_LINEAGE_RELATION')
        seen_edges.add(edge_key); join(left, right)
    extra_exposed = contract['train_side_exposed']
    need(type(extra_exposed) is list and all(type(x) is str for x in extra_exposed)
        and len(extra_exposed) == len(set(extra_exposed)),
         'UNIQUE_DECLARED_TRAIN_EXPOSURE')
    exposed = PROBE_TRAIN_SIDE | {checked_key(x, normalize_libri) for x in extra_exposed}
    old_libri = {'libri-speaker:' + normalize_libri(x) for x in legacy_libri_ids}
    for key in exposed | old_libri | UNRESOLVED_MIXTURES:
        find(key)

    row_keys = {}
    for row in rows:
        allowed = by_id[row['id']]
        need(row.get('diagnostic_only') is False
            and row.get('auto_admission_allowed') is not False and row.get('human_gold') is False,
            'NO_DIAGNOSTIC_OR_HUMAN_GOLD_SOURCE_ROW')
        namespace, source_id = row.get('source_namespace'), row.get('source_id')
        need(namespace in namespaces, 'SOURCE_NAMESPACE_DECLARED_IN_CONTRACT')
        key = source_key(namespace, source_id, normalize_libri)
        need(row.get('role') in ROLES and type(row.get('source_group_id')) is str
            and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,200}', row['source_group_id']) is not None,
            'FROZEN_SOURCE_ROLE_AND_GROUP')
        need(allowed.get('role') == row['role'] and allowed.get('source_group_id') == row['source_group_id']
            and allowed.get('source_namespace') == namespace
            and source_key(namespace, allowed.get('source_id'), normalize_libri) == key,
            'ADMISSION_SOURCE_IDENTITY_ROLE_GROUP_BINDING')
        if namespace == 'kokoro-v1_0-preset':
            need(row.get('preset_id') == source_id == allowed.get('preset_id')
                and row.get('source_speaker_id') is None
                and row.get('human_speaker_identity') is None
                and row.get('human_speaker_identity_verified', False) is False
                and row.get('human_source_identity_verified', False) is False,
                'KOKORO_PRESET_IS_NOT_AUTHENTICATED_HUMAN')
        else:
            need(row.get('preset_id') is None and allowed.get('preset_id') is None,
                 'CORPUS_SPEAKER_IS_NOT_PRESET')
            if row.get('source_speaker_id') is not None:
                need(source_key(namespace, str(row['source_speaker_id']), normalize_libri) == key,
                     'TYPED_SOURCE_SPEAKER_ID_MATCH')
        row_keys[row['id']] = key; find(key)

    components = collections.defaultdict(set)
    for key in list(parent):
        components[find(key)].add(key)
    roles, groups, group_components, output = {}, {}, {}, {}
    for row in rows:
        key = row_keys[row['id']]; component = find(key); members = components[component]
        need(key not in old_libri, 'CROSS_LIBRITTS_LIBRISPEECH_NUMERIC_SPEAKER_EXPOSURE')
        need(not members.intersection(old_libri), 'LEGACY_LIBRI_KNOWN_LINEAGE_EXPOSURE')
        need(not members.intersection(UNRESOLVED_MIXTURES), 'UNRESOLVED_KNOWN_MIXTURE_NOT_INDEPENDENT_SOURCE')
        need(not members.intersection(exposed) or row['role'] == 'train', 'TRAIN_SIDE_PRESET_LINEAGE_EXPOSURE')
        role, group = row['role'], row['source_group_id']
        need(component not in roles or roles[component] == role, 'LINEAGE_CANNOT_CROSS_FROZEN_ROLES')
        need(component not in groups or groups[component] == group, 'ONE_LINEAGE_ONE_CONSUMED_SOURCE_GROUP')
        need(group not in group_components or group_components[group] == component, 'ONE_GROUP_ONE_KNOWN_LINEAGE')
        need(group == component, 'CANONICAL_TYPED_COMPONENT_SOURCE_GROUP')
        roles[component] = role; groups[component] = group; group_components[group] = component
        preset = row.get('preset_id')
        scope = ('SYNTHETIC_TRAIN_PRESET' if role == 'train' else 'HELDOUT_SYNTHETIC_PRESET_WITH_KNOWN_LINEAGE_EXCLUSIONS') if preset else (
            'SYNTHETIC_TRAIN_CORPUS_SOURCE' if role == 'train' else 'HELDOUT_SYNTHETIC_CORPUS_SOURCE_ID')
        value = {'source_namespace': row['source_namespace'], 'source_identity_key': key,
            'preset_id': preset, 'known_lineage': {'component_keys': sorted(members),
                'declared_relations': [copy.deepcopy(e) for e in edges if e['left'] in members],
                'human_ancestry_complete': False}, 'evaluation_scope': scope,
            'human_speaker_identity': None, 'human_speaker_identity_verified': False}
        # Do not silently overwrite conflicting scope/lineage claims supplied by
        # a source producer. The authenticated source contract must agree.
        for field in METADATA_FIELDS:
            if field in row:
                need(row[field] == value[field], 'NO_CONFLICTING_SOURCE_IDENTITY_METADATA')
        output[row['id']] = value
    return {'schema': SCHEMA + '/verified', 'contract_sha256': contract_sha,
        'corpus_sha256': digest(corpus), 'admission_sha256': digest(admission),
        'rows': output, 'source_group_id_preserved': True,
        'source_group_rule': 'lexicographically_first_canonical_typed_component_member',
        'groups_by_role': dict(collections.Counter(roles.values())),
        'known_component_count': len(roles), 'independent_human_speakers_claimed': False,
        'asset_revision_used_as_source_identity': False,
        'probe_train_side_keys': sorted(PROBE_TRAIN_SIDE),
        'not_a_label_or_acoustic_admission': True}
