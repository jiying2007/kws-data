#!/usr/bin/env python3
"""Read existing English research evidence; never train, infer, rank or retune.

This adapter preserves the original metric units and roles. It does not create
selection/holdout data to satisfy research/model_comparison/compare.py. Reuse
that module's identity/digest helpers, and leave event matching/confidence
bounds to eval/score_events.py and tools/statistical_bounds.py.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

PIPELINE_COMMIT = 'aec192dd7c5955a7fa2b709564cdacb185a590be'
EN63_SHA = 'ca16154bcc4100749870d264ade8594ccf416001ab66a3397fbc8d32afc94418'
COVERAGE_SHA = '56b7e9181cc0e8662b00dd1344ef414a1466a668428fec8bc709b600ba8e97e1'
PINS = {
    'research/model_comparison/compare.py': '6558bd0b80a1b5d3af066cc940d5e87103ae656ba869469ca7335676b64c711d',
    'tools/source_identity.py': 'fad7b9daefc5f865e9c064655253e79c0c0e0d055b861d2444796c1b3abcd2b1',
}
WORDS = {'background': 0, 'Hey Vosen': 1, 'Hey Mero': 2}
LEGACY = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def existing_helpers(root):
    """Only import the exact reviewed stdlib helpers; no numerical dependencies."""
    global LEGACY
    for name, expected in PINS.items():
        require(sha((root / name).read_bytes()) == expected, 'source pin mismatch: ' + name)
    spec = importlib.util.spec_from_file_location('saved_comparison_existing', root / 'research/model_comparison/compare.py')
    LEGACY = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LEGACY)


def digest(value):
    require(LEGACY is not None, 'existing identity helpers must be initialized')
    return LEGACY.digest(value)


def indexed(rows):
    require(isinstance(rows, list) and rows, 'nonempty records required')
    result = {}
    for row in rows:
        require(isinstance(row, dict), 'record must be an object')
        name = row.get('id')
        require(isinstance(name, str) and bool(name) and name not in result, 'invalid or duplicate record ID')
        result[name] = row
    return result


def card(model, contract, records, evidence):
    indexed(records)
    LEGACY.identity(model['weights_sha256'], 'pcm_sha256')  # existing strict SHA identity validator
    return dict(model=model, contract=contract, records=records, evidence=evidence)


def summarize(card):
    groups = defaultdict(Counter)
    for row in card['records']:
        name = row['role'] + '/' + str(row['expected_class'])
        group = groups[name]
        group['rows'] += 1
        for key, value in row['observation'].items():
            require(type(value) in (int, bool) and value >= 0, 'invalid discrete observation')
            group[key] += int(value)
    return {name: dict(values) for name, values in sorted(groups.items())}


def pair(left, right, *, use='research_diagnostic'):
    """Match exact tasks/rows. Frontend and architecture differences are reported.

    These two adapters have only exposed weak labels: an independent ranking
    is intentionally unsupported. Future admitted event evidence should go
    through the existing event scorer, not a renamed weak metric in this file.
    """
    require(use == 'research_diagnostic', 'independent ranking unsupported for these weak/exposed evidence adapters')
    a, b = indexed(left['records']), indexed(right['records'])
    reasons = []
    for key in ('language', 'target_profile_identity', 'dataset_identity', 'label_semantics',
                'metric_unit', 'scoring_identity', 'threshold_selection_identity', 'condition', 'class_map'):
        if left['contract'].get(key) != right['contract'].get(key):
            reasons.append('unmatched_' + key)
    if set(a) != set(b):
        reasons.append('unmatched_exact_record_set_no_implicit_intersection')
    identity_keys = ('source_identity', 'pcm_identity', 'role', 'expected_class', 'previously_exposed',
                     'independent_holdout', 'label_authority', 'target_profile_identity', 'clock')
    for name in sorted(set(a) & set(b)):
        for row in (a[name], b[name]):
            require(type(row['previously_exposed']) is bool and type(row['independent_holdout']) is bool,
                    'explicit exposure/holdout booleans required')
            require(not row['independent_holdout'], 'independent/protected holdout cannot enter an exposed diagnostic adapter')
            require(row['role'] in ('train', 'exposed_development', 'quarantine'), 'unsupported historical role')
        for key in identity_keys:
            if a[name].get(key) != b[name].get(key):
                reasons.append('unmatched_record_' + key)
    output = dict(left=left['model']['id'], right=right['model']['id'],
                  status='BLOCKED_INCOMPATIBLE' if reasons else 'PAIRED_RESEARCH_DIAGNOSTIC',
                  reasons=sorted(set(reasons)), formal_ranking_allowed=False,
                  inference_calls=0, selected_winner=None,
                  frontends={'left': left['model']['frontend'], 'right': right['model']['frontend']},
                  frontend_difference_is_not_rejection=True)
    if not reasons:
        output.update(metric_unit=left['contract']['metric_unit'], condition=left['contract']['condition'],
                      exact_paired_records=len(a), left_counts=summarize(left), right_counts=summarize(right),
                      identity_sha256=digest([{k: a[n][k] for k in ('id', *identity_keys)} for n in sorted(a)]))
        changes = defaultdict(lambda: Counter({'improved_requested_match': 0, 'lost_requested_match': 0, 'changed_observation_rows': 0}))
        for name in sorted(a):
            old, new = a[name]['observation'], b[name]['observation']
            group = changes[a[name]['role'] + '/' + str(a[name]['expected_class'])]
            group['changed_observation_rows'] += old != new
            if 'requested_match' in old:
                group['improved_requested_match'] += bool(new['requested_match']) and not old['requested_match']
                group['lost_requested_match'] += bool(old['requested_match']) and not new['requested_match']
        output['paired_changes'] = {key: dict(value) for key, value in sorted(changes.items())}
    return output


def english63_cards(root):
    source = root / 'research/english63_weak_pilot/RESULT.json'
    raw = source.read_bytes()
    require(sha(raw) == EN63_SHA, 'English63 result bytes differ from pinned evidence')
    data = parse(raw)
    require(data['schema'] == 'english63-weak-clip-two-architecture-pilot-v1', 'unexpected English63 schema')
    require(data['human_gold'] is False and data['formal_promotion_allowed'] is False, 'historical authority changed')
    profile = 'english63-original-requested-spellings-acoustic-IPA-unverified-20261113'
    contract = dict(language='en', target_profile_identity=profile, dataset_identity=data['exact_input_sha256'],
                    label_semantics='synthetic-requested-intent-not-acoustic-truth', metric_unit='whole_clip_weak_intent_class',
                    scoring_identity='whole-clip-class-argmax-v1', threshold_selection_identity='no-threshold-no-dev-early-stop-48-fixed-updates',
                    condition='original63', class_map=WORDS)
    cards = []
    actions = {'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY': 'train', 'EXPOSED_DEVELOPMENT_NO_TUNE': 'exposed_development',
               'QUARANTINE_AMBIGUOUS_NEARPHONE': 'quarantine'}
    for model in data['models']:
        require(model['class_roster'] == WORDS, 'class map changed')
        weight_source = root / 'research/english63_weak_pilot' / (model['weights_file'] + '.b64')
        weight_bytes = base64.b64decode(weight_source.read_bytes().strip(), validate=True)
        require(sha(weight_bytes) == model['weights_sha256'], 'English63 actual weight bytes mismatch')
        rows = []
        for r in model['per_record']:
            role = actions[r['action']]
            target, predicted = r['requested_weak_intent_id'], r['predicted_class']
            require(type(predicted) is int and predicted in (0, 1, 2), 'invalid predicted class')
            require(target is None if role == 'quarantine' else type(target) is int and target in (0, 1, 2), 'invalid weak target')
            require(r['human_gold'] is False and r['trusted_negative'] is False, 'weak label promoted')
            require(r['window_or_streaming_event'] is False, 'clip prediction promoted to event')
            observation = ({'requested_match': predicted == target} if target is not None else
                           {'quarantined_target_class_proposals_not_false_accepts': predicted in (1, 2)})
            rows.append(dict(id=r['cell_id'], source_identity=[data['source_tts_sha256'], r['cell_id'], r['source_family_id']],
                             pcm_identity=None, role=role, expected_class=target, previously_exposed=True, independent_holdout=False,
                             label_authority='requested-intent-only', target_profile_identity=profile, clock=None, observation=observation))
        require(len(rows) == 63, 'English63 roster changed')
        model_card = card(dict(id='english63-wholeclip-' + model['architecture'], architecture=model['architecture'],
                               weights_sha256=model['weights_sha256'], weight_check='exact_decoded_NPZ_bytes_SHA256',
                               frontend=dict(sample_rate_hz=16000, mel_bins=32, window_ms=25, hop_ms=10,
                                             train_normalization_sha256=digest(data['train_only_frontend_norm']))),
                          contract.copy(), rows, dict(result_sha256=sha(raw), original_tts_zip_sha256=data['source_tts_sha256'],
                                                     per_record_pcm_hashes_present=False))
        counts = summarize(model_card)
        for role, count_key in [('train', 'requested_weak_train_agreement'), ('exposed_development', 'exposed_development_weak_agreement_not_independent_accuracy')]:
            require(sum(v.get('requested_match', 0) for k, v in counts.items() if k.startswith(role + '/')) == model[count_key], 'saved aggregate mismatch')
        require(sum(v.get('quarantined_target_class_proposals_not_false_accepts', 0) for v in counts.values()) == model['quarantined_nearphone_wake_proposal_not_false_accept_count'], 'quarantine aggregate mismatch')
        cards.append(model_card)
    require({c['model']['architecture'] for c in cards} == {'gru', 'dsc_tcn'}, 'missing English63 arm')
    return cards


def event_observation(event, roster):
    """Recount saved discrete events only; no logits or decoder replay."""
    for key in ('id', 'role', 'weak_class', 'observed_samples', 'original_clip_eof_sample'):
        require(event[key] == roster[key], 'event/roster mismatch: ' + key)
    start = roster['source'].get('clip_start_sample')
    require(event['clip_start_sample'] == start, 'clip clock mismatch')
    require(event['independent_acoustic_evaluation'] is False and event['word_end_latency'] == 'NOT_MEASURED', 'event authority changed')
    events = event['events']
    require(isinstance(events, list), 'explicit saved event list required')
    last = -1
    for e in events:
        require(type(e['keyword']) is int and e['keyword'] in (1, 2), 'invalid event keyword')
        require(type(e['sample']) is int and last <= e['sample'] <= roster['observed_samples'] and e['sample'] >= 0, 'invalid event clock')
        last = e['sample']
    positive = roster['weak_class'] > 0
    before = [e for e in events if start is not None and e['sample'] < start]
    after = [e for e in events if start is None or e['sample'] >= start]
    matched = [e for e in after if e['keyword'] == roster['weak_class']] if positive else []
    wrong = [e for e in after if e['keyword'] != roster['weak_class']] if positive else []
    extra = max(0, len(matched) - 1)
    require(event['matched_intent_after_clip_start'] == bool(matched), 'saved matched count mismatch')
    require(event['prefix_events'] == before and event['wrong_keyword_after_clip_start'] == wrong, 'saved event partition mismatch')
    require(event['extra_same_keyword_events'] == extra, 'saved extra count mismatch')
    weak_negative = len(events) if not positive else 0
    require(event['weak_negative_intent_events'] == weak_negative, 'saved negative count mismatch')
    require(event['no_matching_event_in_observed_context'] == (positive and not matched), 'saved censored miss mismatch')
    observation = dict(events=len(events), prefix_events=len(before), wrong_keyword_events=len(wrong),
                       extra_same_keyword_events=extra, weak_negative_intent_events=weak_negative)
    if positive:
        observation.update(requested_match=bool(matched), censored_no_match_not_FRR=not matched)
    return observation


def coverage_cards_from_members(z):
    cards = []
    require(len(z.namelist()) == len(set(z.namelist())), 'duplicate ZIP members')
    result, plan = parse(z.read('evidence/RESULT.json')), parse(z.read('evidence/PLAN.json'))
    require(result['qualification_allowed'] is False and result['pronunciation_verified'] is False, 'coverage authority changed')
    profile = 'snapshot160-final-image-20261116-requested-IPA-miːroʊ-voʊzən-unverified-acoustics'
    event_rule = {k: plan[k] for k in ('threshold', 'confirm_embeddings', 'cooldown_embeddings')}
    for condition, roster_path, old_path in [
        ('clean', 'sources/STREAM_ROSTER.json', 'predictions/reused/candidate-clean-events.json'),
        ('snr10', 'cache/snr10-roster.json', 'predictions/reused/candidate-snr10-events.json'),
        ('snr5', 'cache/snr5-roster.json', 'predictions/reused/aug96-snr5-events.json'),
        ('new8', 'cache/new8-roster.json', 'predictions/aug96-new8-events.json')]:
        roster_raw = z.read(roster_path)
        roster = indexed(parse(roster_raw))
        contract = dict(language='en', target_profile_identity=profile, dataset_identity=sha(roster_raw),
                        label_semantics='synthetic-requested-intent-not-acoustic-truth', metric_unit='observed_context_weak_event',
                        scoring_identity=digest(event_rule), threshold_selection_identity='fixed0.8-confirm3-cooldown13-exposed-development-history-no-formal-calibration',
                        condition=condition, class_map=WORDS)
        for name, model_path, result_key, event_path in [
            ('aug96', 'models/aug96-control-head.npz', 'aug96_control', old_path),
            ('coverage', 'models/coverage-head.npz', 'coverage_candidate', 'predictions/coverage-' + condition + '-events.json')]:
            weights_sha = sha(z.read(model_path))
            require(weights_sha == result[result_key]['sha256'], 'saved model digest mismatch')
            event_raw = z.read(event_path)
            events = indexed(parse(event_raw))
            require(set(events) == set(roster), 'missing/extra saved event rows')
            rows = []
            for case_id, r in roster.items():
                source = r['source']
                current_pcm = r.get('mix', {}).get('combined_mixed_pcm_sha256') or source.get('combined_pcm_sha256') or source.get('pcm_sha256')
                LEGACY.identity(current_pcm, 'pcm_sha256')
                original_pcm = source.get('source_pcm_sha256') or source.get('pcm_sha256')
                rows.append(dict(id=case_id, source_identity=[r.get('parent_id', case_id), original_pcm], pcm_identity=current_pcm,
                                 role={'train': 'train', 'development': 'exposed_development'}[r['role']], expected_class=r['weak_class'],
                                 previously_exposed=True, independent_holdout=False, label_authority='requested-intent-only',
                                 target_profile_identity=profile,
                                 clock=dict(sample_rate_hz=16000, clip_start_sample=source.get('clip_start_sample'),
                                            observed_samples=r['observed_samples'], original_clip_eof_sample=r['original_clip_eof_sample']),
                                 observation=event_observation(events[case_id], r)))
            item = card(dict(id=name + '-' + condition, architecture='frozen-oWW-embedding96-MLP-16x96-32-3',
                             weights_sha256=weights_sha, weight_check='exact_NPZ_member_bytes_SHA256',
                             frontend=dict(sample_rate_hz=16000, features=96, embedding_hop_samples=1280,
                                           first_embedding_sample=16000, first_head_sample=35200)), contract.copy(), rows,
                        dict(artifact_sha256=COVERAGE_SHA, roster_member=roster_path, events_member=event_path, events_sha256=sha(event_raw),
                             per_record_pcm_hashes_present=True, event_rule=event_rule))
            counts = summarize(item)
            for key, saved in result[name + '_' + condition].items():
                group = key.replace('development/', 'exposed_development/')
                actual = counts[group]
                for field, source_field in [('rows', 'cases'), ('requested_match', 'matched_intent_cases'),
                                            ('prefix_events', 'prefix_events'), ('wrong_keyword_events', 'wrong_keyword_events'),
                                            ('extra_same_keyword_events', 'extra_same_keyword_events'), ('weak_negative_intent_events', 'weak_negative_intent_events')]:
                    require(actual.get(field, 0) == saved[source_field], 'coverage aggregate mismatch: ' + group + '/' + field)
            cards.append(item)
    return cards


def compare_cards(original, current):
    cards = original + current
    comparisons = [pair(*original)] + [pair(current[i], current[i + 1]) for i in range(0, len(current), 2)]
    incompatible = pair(original[0], current[1])
    return dict(schema='kws-existing-saved-evidence-comparison-v1', status='READBACK_COMPLETED_RESEARCH_ONLY',
                evidence_scope='previously_exposed_weak_requested_intent_diagnostics', human_gold=False,
                evaluation_source_previously_exposed=True,
                pipeline_commit=PIPELINE_COMMIT, model_calls=0, training_calls=0, decoder_replay_calls=0,
                independent_audio_increment=0, statistical_bounds_computed=0, formal_ranking_allowed=False,
                same_FAR_ranking='NOT_AVAILABLE_NO_COMMON_INDEPENDENT_EVENT_SET',
                observed_pairs=comparisons, rejected_cross_cohort_example=incompatible,
                model_cards=[{k: v for k, v in c.items() if k != 'records'} | {'counts': summarize(c)} for c in cards],
                existing_implementations=dict(receipt_compare='research/model_comparison/compare.py',
                                              event_matcher='eval/score_events.py', bounds='tools/statistical_bounds.py',
                                              far_aggregation='eval/aggregate_far.py'),
                limitations=['SHA/record pairing verifies saved identity, not acoustic label truth.',
                             'Historical source pronunciations are preserved; no current user target is chosen.',
                             'English63 per-record PCM hashes are absent from its public RESULT; original archive identity plus record IDs only.',
                             'SNR variants reuse sources and are separate conditions, not independent examples or FAR exposure.',
                             'Different frontends may be compared as complete systems on genuinely shared data; no architecture-only causal claim.',
                             'No global winner, FRR, FAR confidence, unseen-holdout or SSC305 qualification follows.'])


CAPSULE_SHA = '0d13167b2139352db23f126ace56a6d6045c8171e2244720dc4d60a5093e877d'


class MemberView:
    def __init__(self, archive, prefix='', path=''):
        self.archive, self.prefix, self.path = archive, prefix, path

    def __truediv__(self, part):
        return MemberView(self.archive, self.prefix, self.path + '/' + str(part))

    def read_bytes(self):
        name = self.path.lstrip('/').removeprefix('research/english63_weak_pilot/')
        return self.read(name)

    def read(self, name):
        return self.archive.read(self.prefix + name)

    def namelist(self):
        return [n[len(self.prefix):] for n in self.archive.namelist() if n.startswith(self.prefix)]


def build_capsule(package):
    existing_helpers(package / 'sources/pipeline')
    manifest = parse((package / 'SOURCE_MANIFEST.json').read_bytes())
    raw = (package / 'saved_inputs.zip').read_bytes()
    require(sha(raw) == CAPSULE_SHA == manifest['capsule_zip_sha256'], 'capsule ZIP hash mismatch')
    require(len(raw) == manifest['capsule_zip_bytes'], 'capsule ZIP size mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        expected = {r['path']: r for r in manifest['members']}
        require(len(expected) == len(manifest['members']), 'duplicate manifest member')
        require(len(names) == len(set(names)) and set(names) == set(expected), 'capsule exact member set mismatch')
        for name, row in expected.items():
            content = z.read(name)
            require(len(content) == row['bytes'] and sha(content) == row['sha256'], 'capsule member identity mismatch: ' + name)
        report = compare_cards(english63_cards(MemberView(z, 'english63/')),
                               coverage_cards_from_members(MemberView(z, 'coverage/')))
    report['readback_input'] = dict(kind='exact_saved_members_capsule', capsule_sha256=CAPSULE_SHA,
                                   manifest_sha256=sha((package / 'SOURCE_MANIFEST.json').read_bytes()),
                                   original_full_artifact_rechecked_in_this_replay=False,
                                   original_member_bytes_preserved=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = build_capsule(args.package.resolve())
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=output['status'], paired_groups=len(output['observed_pairs']), model_calls=0)))


if __name__ == '__main__':
    main()
