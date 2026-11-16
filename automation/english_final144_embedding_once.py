"""One explicitly guarded final144 requested-intent research training worker.

No model runs on import or default invocation. Four complete audio-only observers
support a fixed weak-intent selection; they do not verify acoustic pronunciation.
The worker consumes unchanged source artifacts and a pinned pipeline checkout.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE if HERE.name == 'automation' else HERE.parent / 'kws-data' / 'automation'
SCHEMA = 'english-final144-embedding-intent-research-v1'
PINS_SCHEMA = 'english-final144-training-pins-v1'
PROVIDERS = ('qwen', 'whisper', 'parakeet', 'phones')
FALSE_FLAGS = ('training_admitted', 'formal_training_admitted', 'human_gold', 'trusted_negative',
               'pronunciation_verified', 'model_promotion_allowed', 'qualification_allowed')
SELECTION = {
    'schema': 'english-final144-fixed-intent-support-v1',
    'normalization': 'delete-U0027-U2019-then-lowercase-ASCII-alphanumeric-tokens',
    'asr_votes_required': 2, 'asr_total': 3,
    'mero_names': ['mero', 'miro', 'meero'], 'vosen_names': ['vosen', 'vosin', 'vozin'],
    'mero_core': [['m'], ['i', 'iː'], ['ɹ', 'r'], ['oʊ', 'oː', 'o']],
    'vosen_core': [['v'], ['oʊ', 'oː'], ['z'], ['ə'], ['n']],
    'ordinary_negative': 'two-exact-requested-transcripts-and-neither-target-phone-core',
    'minimum': {'train_per_word': 8, 'train_families_per_word': 2,
                'development_per_word': 2, 'development_families_per_word': 1,
                'ordinary_train': 8, 'ordinary_development': 2},
    'unmatched': 'quarantine-without-negative-relabel',
    'length_stress_and_hey_acoustic_verification': False,
    'optimizer_updates': 96, 'proposal_threshold': 0.8,
    'proposal_confirm_embeddings': 3, 'proposal_cooldown_embeddings': 13,
    'development_tuning': False,
}
REFERENCE = {
    'repository': 'jiying2007/kws-data', 'run_id': 43538998200, 'artifact_id': 14090383839,
    'zip_bytes': 47199, 'zip_sha256': '7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4',
    'member': 'english-natural-read-speech-result.json',
    'member_sha256': '788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510'}
CORPUS = {'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz', 'bytes': 337926286,
          'sha256': '76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3'}
PIPELINE_FILES = ('weak_embedding_contract.py', 'weak_audio_cases.py', 'causal_embedding.py',
                  'causal_head.py', 'portable_head.py', 'english_causal_head.c',
                  'english_causal_head.h', 'head_replay.c')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def regular(path, limit):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and path.stat().st_size <= limit, 'REGULAR_FILE_SIZE')
    raw = path.read_bytes()
    need(len(raw) <= limit, 'FILE_SIZE_CHANGED')
    return raw


def read_json(path, limit=16 * 1024**2):
    return json.loads(regular(path, limit), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def once(path, value):
    raw = canonical(value) + b'\n'
    with Path(path).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha(raw)


def hex_id(value, length):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{%d}' % length, value) is not None


def validate_pins(pins, batch_pins):
    need(type(pins) is dict and pins.get('schema') == PINS_SCHEMA and pins.get('enabled') is True,
         'REAL_TRAINING_NOT_ENABLED')
    need(hex_id(pins.get('main_before'), 40) and hex_id(pins.get('pipeline_commit'), 40), 'MISSING_SOURCE_COMMIT_PINS')
    need(pins.get('batch_pins_sha256') == sha(canonical(batch_pins)) and
         pins.get('selection_sha256') == sha(canonical(SELECTION)) and
         pins.get('main_before') == batch_pins.get('next_main_before') and
         pins.get('recipe_sha256') == batch_pins.get('recipe_sha256'), 'TRAINING_RECIPE_OR_BATCH_BINDING')
    need(pins.get('optimizer_updates') == 96 and pins.get('max_source_cells') == 144 and
         pins.get('source_labels_are_intents_only') is True and pins.get('no_retry') is True and
         all(pins.get(key) is False for key in FALSE_FLAGS), 'WEAK_RESEARCH_LIMITS_ONLY')


def execution_guard(pins, batch_pins, env, event):
    validate_pins(pins, batch_pins)
    need(env.get('GITHUB_REPOSITORY') == 'jiying2007/kws-data' and
         env.get('GITHUB_EVENT_NAME') == 'push' and env.get('GITHUB_REF') == 'refs/heads/main' and
         env.get('GITHUB_RUN_ATTEMPT') == '1' and re.fullmatch(r'[0-9]+', env.get('GITHUB_RUN_ID', '')) and
         event.get('before') == pins['main_before'] and event.get('after') == env.get('GITHUB_SHA') and
         hex_id(event.get('after'), 40) and event.get('after') != event.get('before') and
         event.get('repository', {}).get('full_name') == 'jiying2007/kws-data' and
         event.get('repository', {}).get('private') is False, 'FIRST_PUBLIC_MAIN_PUSH_ONLY')


def normalized(text):
    need(type(text) is str, 'ASR_TEXT_TYPE')
    return re.findall(r'[a-z0-9]+', text.lower().replace("'", '').replace('’', ''))


def has_core(tokens, choices):
    return any(all(token in allowed for token, allowed in zip(tokens[start:start + len(choices)], choices))
               for start in range(len(tokens) - len(choices) + 1))


def select_intents(recipe, rows, binding):
    """Frozen machine support for source intents, never a pronunciation pass."""
    cells = recipe.get('cells')
    need(type(cells) is list and len(cells) == len(rows) == 144 and
         len({c['cell_id'] for c in cells}) == len({r['cell_id'] for r in rows}) == 144 and
         {c['cell_id'] for c in cells} == {r['cell_id'] for r in rows}, 'COMPLETE_144_REVIEW_ROSTER')
    summaries = binding.get('observer_summaries_sha256')
    need(binding.get('test_backend') is False and type(summaries) is dict and set(summaries) == set(PROVIDERS) and
         all(hex_id(value, 64) for value in summaries.values()), 'FOUR_BOUND_OBSERVER_SUMMARIES')
    manifest_sha = sha(canonical(summaries))
    by_id = {row['cell_id']: row for row in rows}
    eligible, observations, decisions = [], {}, []
    for cell in cells:
        row = by_id[cell['cell_id']]
        need(row.get('status') == 'GENERATED_UNADMITTED' and hex_id(row.get('pcm_sha256'), 64) and
             all(row.get(key) is False for key in FALSE_FLAGS), 'REAL_UNADMITTED_SOURCE_REQUIRED')
        transcripts, tokens = row.get('transcripts'), row.get('raw_ipa_tokens')
        need(type(transcripts) is dict and set(transcripts) == set(PROVIDERS[:3]) and
             all(type(x) is str for x in transcripts.values()) and type(tokens) is list and tokens and
             all(type(x) is str and x for x in tokens), 'COMPLETE_FOUR_OBSERVATIONS')
        cores = {1: has_core(tokens, SELECTION['vosen_core']), 2: has_core(tokens, SELECTION['mero_core'])}
        label = cell.get('requested_keyword_index') or 0
        normalized_asr = {provider: normalized(text) for provider, text in transcripts.items()}
        if cell.get('intent') == 'wake-candidate':
            need(label in (1, 2), 'WAKE_INTENT_CLASS')
            names = SELECTION['vosen_names' if label == 1 else 'mero_names']
            votes = sum(len(words) == 2 and words[0] == 'hey' and words[1] in names
                        for words in normalized_asr.values())
            useful = votes >= 2 and cores[label]
            status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else (
                'STRONG_CONFLICT' if cores[3 - label] and not cores[label] else 'UNKNOWN')
            phone_status = ('TARGET_CORE_MACHINE_SUPPORT_HEY_LENGTH_STRESS_UNVERIFIED' if cores[label] else
                            'TARGET_CORE_NOT_ESTABLISHED_BY_THIS_OBSERVER')
        else:
            need(cell.get('intent') == 'ordinary-negative-candidate' and label == 0, 'ORDINARY_INTENT_CLASS')
            wanted = normalized(cell['synthesis_text'])
            votes = sum(words == wanted for words in normalized_asr.values())
            useful = votes >= 2 and not any(cores.values())
            status = 'USEFUL_FOR_INTENT_RESEARCH' if useful else ('STRONG_CONFLICT' if any(cores.values()) else 'UNKNOWN')
            phone_status = 'NO_TARGET_CORE_MACHINE_OBSERVED_NOT_TRUSTED_NEGATIVE' if not any(cores.values()) else 'TARGET_CORE_IN_ORDINARY_INTENT_CONFLICT'
        observation = {'source_pcm_sha256': row['pcm_sha256'],
            # This field is a manifest of four original summary identities, not a
            # claim that it is the digest of an individual Actions ZIP artifact.
            'observer_artifact_sha256': manifest_sha,
            'observer_artifact_identity_kind': 'FOUR_ORIGINAL_SUMMARY_SHA256_MANIFEST',
            'observer_summaries_sha256': summaries,
            'observation_sha256': sha(canonical({'cell_id': row['cell_id'], 'pcm_sha256': row['pcm_sha256'],
                'transcripts': transcripts, 'raw_ipa_tokens': tokens})),
            'evidence_status': status, 'phone_review_status': phone_status}
        observations[row['cell_id']] = observation
        if useful:
            eligible.append(row['cell_id'])
        decisions.append({'cell_id': row['cell_id'], 'role': cell['role'], 'weak_intent_class': label,
            'source_status_unchanged': row['status'], 'asr_votes': votes,
            'machine_phone_core': {str(k): v for k, v in cores.items()},
            'transcripts': transcripts, 'raw_ipa_tokens': tokens,
            'evidence_status': status, 'phone_review_status': phone_status,
            'eligible_for_requested_intent_research': useful, **dict.fromkeys(FALSE_FLAGS, False)})
    return {'policy': SELECTION, 'policy_sha256': sha(canonical(SELECTION)),
            'observer_artifact_manifest_sha256': manifest_sha, 'source_denominator': 144,
            'eligible_cell_ids': eligible, 'observations': observations, 'decisions': decisions}


def minimum_evidence(rows):
    """Run after deduplication, then again after actual feature exclusions."""
    result, passed = {}, True
    for role in ('train', 'development'):
        for label in (0, 1, 2):
            subset = [row for row in rows if row['role'] == role and row['weak_class'] == label]
            families = sorted({row['source_family_id'] for row in subset})
            minimum = 8 if role == 'train' else 2
            family_minimum = (2 if role == 'train' else 1) if label else 1
            ok = len(subset) >= minimum and len(families) >= family_minimum
            passed = passed and ok
            result[role + '/' + str(label)] = {'count': len(subset), 'families': families,
                'minimum_count': minimum, 'minimum_families': family_minimum, 'passed': ok}
    return {'passed': passed, 'counts': result, 'ordinary_negative_excludes_librispeech_substitution': True}


def pipeline_libraries(root, expected_commit):
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'PIPELINE_CHECKOUT')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    need(actual == expected_commit and not subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain'], text=True).strip(), 'IMMUTABLE_CLEAN_PIPELINE_CHECKOUT')
    directory = root / 'research/english_final_embedding'
    hashes = {name: sha(regular(directory / name, 256 * 1024)) for name in PIPELINE_FILES}
    sys.path.insert(0, str(directory))
    libs = {name: importlib.import_module(name) for name in
            ('weak_embedding_contract', 'weak_audio_cases', 'causal_embedding', 'causal_head', 'portable_head')}
    path = root / 'research/english63_online_c/librispeech_screen.py'
    spec = importlib.util.spec_from_file_location('_final144_librispeech_sources', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    libs['librispeech'] = old
    return libs, {'commit': actual, 'feature_head_files_sha256': hashes,
                  'librispeech_source_helper_sha256': sha(regular(path, 256 * 1024))}


def reference_records(reference_zip, library):
    raw = regular(reference_zip, REFERENCE['zip_bytes'])
    need(len(raw) == REFERENCE['zip_bytes'] and sha(raw) == REFERENCE['zip_sha256'], 'EXACT_LIBRI_REFERENCE_ZIP')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == 1 and infos[0].filename == REFERENCE['member'] and
             not infos[0].is_dir() and infos[0].file_size < 1024**2, 'LIBRI_REFERENCE_MEMBER')
        member = archive.read(REFERENCE['member'])
    need(sha(member) == REFERENCE['member_sha256'], 'EXACT_LIBRI_REFERENCE_JSON')
    return library.reference_rows(member)


def decode_background(archive_path, reference, library):
    """Authenticate original FLAC/transcripts; decode original samples only."""
    import numpy as np
    import soundfile as sf
    need(np.__version__ == '1.26.4' and sf.__version__ == '0.13.1', 'PINNED_PCM_DECODER')
    library.old.verified_archive(Path(archive_path))
    rows = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        members, selected = library.old.source_members(archive)
        need(len({m.name for m in members}) == len(members) and len(selected) == 517 and
             {m.name for m in selected} == set(reference), 'ORIGINAL_LIBRI_SOURCE_ROSTER')
        transcripts = library.old.extract_transcripts(archive, members)
        for member in selected:
            prior = reference[member.name]
            speaker, _, utterance = library.old.NAME.fullmatch(member.name).groups()
            transcript = transcripts[utterance]
            need(speaker == prior['speaker_source_id'] and library.old.no_requested_wake(transcript) and
                 sha(transcript.encode()) == prior['transcript_sha256'], 'ORIGINAL_LIBRI_TRANSCRIPT')
            source = archive.extractfile(member)
            need(source is not None, 'ORIGINAL_LIBRI_FLAC_FILE')
            raw = source.read()
            need(sha(raw) == prior['flac_sha256'], 'ORIGINAL_LIBRI_FLAC_HASH')
            with sf.SoundFile(io.BytesIO(raw)) as sound:
                need(sound.samplerate == 16000 and sound.channels == 1 and sound.subtype == 'PCM_16' and
                     sound.frames == prior['sample_count'], 'ORIGINAL_LIBRI_PCM_FORMAT')
                samples = sound.read(dtype='int16', always_2d=False)
            need(samples.ndim == 1 and samples.dtype == np.int16 and len(samples) == prior['sample_count'], 'ORIGINAL_LIBRI_PCM_LENGTH')
            pcm = samples.astype('<i2', copy=False).tobytes()
            rows.append({'id': member.name, 'speaker': speaker, 'frames': len(samples),
                'flac_sha256': prior['flac_sha256'], 'transcript_sha256': prior['transcript_sha256'],
                'pcm_sha256': sha(pcm), 'pcm16le': pcm, 'trusted_negative': False})
    need(len(rows) == 517 and len({r['speaker'] for r in rows}) == 40 and
         sum(r['frames'] for r in rows) == 58646164, 'ORIGINAL_LIBRI_PCM_TOTAL')
    return rows, {'numpy': np.__version__, 'soundfile': sf.__version__,
                  'libsndfile': sf.__libsndfile_version__, 'old_pcm_pin_claimed': False,
                  'resampling': False, 'normalization': False, 'inserted_samples': 0}


def build_replays(portable, output):
    tools = {name: shutil.which(name) for name in ('gcc', 'arm-linux-gnueabihf-gcc', 'qemu-arm')}
    need(all(tools.values()), 'HOST_AND_ARM_REPLAY_TOOLS_REQUIRED')
    native, arm = output / 'head-native', output / 'head-armv7'
    portable.compile_replay(native, compiler=tools['gcc'])
    portable.compile_replay(arm, compiler=tools['arm-linux-gnueabihf-gcc'], extra_flags=('-static',))
    for path, machine in ((native, 62), (arm, 40)):
        raw = regular(path, 16 * 1024**2)
        need(raw[:4] == b'\x7fELF' and raw[5] == 1 and int.from_bytes(raw[18:20], 'little') == machine,
             'REPLAY_EXECUTABLE_ARCHITECTURE')
    report = {'executables': {p.name: sha(p.read_bytes()) for p in (native, arm)},
              'tool_versions': {name: subprocess.check_output([path, '--version'], text=True).splitlines()[0]
                                for name, path in tools.items()}, 'arm_execution_kind': 'QEMU_NOT_SSC305'}
    return [str(native)], [tools['qemu-arm'], str(arm)], report


def save_stream_cache(streams, cases, output):
    """Persist actual feature outputs before attempting any optimizer update."""
    import numpy as np
    by_case = {case['id']: case for case in cases}
    embeddings, samples, gaps, roster = [], [], [], []
    offset = 0
    for stream in streams:
        case = by_case[stream['id']]
        records = stream['records']
        need(records, 'NONEMPTY_ACTUAL_EMBEDDING_STREAM')
        start = len(samples)
        for index, row in enumerate(records):
            embeddings.append(row['embedding'])
            samples.append(offset + row['available_sample'])
            gaps.append(index == 0)
        count = len(case['pcm16le']) // 2
        roster.append({'id': stream['id'], 'role': case['role'], 'weak_class': case['weak_class'],
            'offset': offset, 'observed_samples': count, 'first_record': start, 'records': len(records),
            'original_clip_eof_sample': stream['original_clip_eof_sample'],
            'source': case['evidence']})
        offset += count
    matrix = np.asarray(embeddings, dtype=np.float32)
    # Persist the actual feature output before loading the saved head again.
    np.savez(output / 'actual-embedding-streams.npz', embeddings=matrix,
             samples=np.asarray(samples, dtype=np.uint64), discontinuity_before=np.asarray(gaps, dtype=np.bool_))
    once(output / 'STREAM_ROSTER.json', roster)
    return sha((output / 'actual-embedding-streams.npz').read_bytes())


def replay_saved_streams(portable, weights_path, output, native, arm):
    """Both C programs consume the independently reopened saved feature stream."""
    import numpy as np
    with np.load(output / 'actual-embedding-streams.npz', allow_pickle=False) as archive:
        matrix = archive['embeddings'].copy()
        samples = archive['samples'].copy().tolist()
        gaps = archive['discontinuity_before'].copy().tolist()
    roster = read_json(output / 'STREAM_ROSTER.json')
    weights = portable.load_weights(weights_path)
    reports = {name: portable.check_parity(weights, matrix, samples, output / name, runner, gaps)
               for name, runner in (('host-parity', native), ('arm-parity', arm))}
    need(reports['host-parity']['events'] == reports['arm-parity']['events'], 'HOST_ARM_EVENT_IDENTITY')
    (output / 'causal-head.bin').write_bytes(portable.export_blob(weights))
    events = reports['host-parity']['events']
    per_case = []
    for row in roster:
        found = [{'sample': event['sample'] - row['offset'], 'keyword': event['keyword']}
                 for event in events if row['offset'] < event['sample'] <= row['offset'] + row['observed_samples']]
        eof = row['original_clip_eof_sample']
        per_case.append({'id': row['id'], 'role': row['role'], 'weak_class': row['weak_class'],
            'constructed_observation_samples': row['observed_samples'], 'events': found,
            'original_eof_view': None if eof is None else {
                'observed_through_sample': eof, 'events': [event for event in found if event['sample'] <= eof],
                'view': 'CAUSAL_PREFIX_ONLY_NO_ADDED_SAMPLES', 'nontrigger_status': 'CENSORED_NOT_FRR'},
            'independent_acoustic_evaluation': False})
    return reports, per_case


def execute(args):
    pins, batch_pins = read_json(args.pins), read_json(args.batch_pins)
    execution_guard(pins, batch_pins, os.environ, read_json(Path(os.environ['GITHUB_EVENT_PATH'])))
    output = args.output
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'CREATE_ONLY_TRAINING_OUTPUT')
    output.mkdir(mode=0o700)
    started = {'schema': SCHEMA, 'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': '1',
        'github_sha': os.environ['GITHUB_SHA'], 'worker_code_sha256': sha(Path(__file__).read_bytes()),
        'training_pins_sha256': sha(canonical(pins)), 'selection_policy_sha256': sha(canonical(SELECTION)),
        'maximum_optimizer_updates': 96, 'no_retry': True, 'test_backend': False,
        **dict.fromkeys(FALSE_FLAGS, False)}
    once(output / 'STARTED.json', started)
    stage, training_started, training_result = 'source-readback', False, None
    clock = time.monotonic()
    try:
        sys.path.insert(0, str(AUTOMATION))
        runner, recipe_module = importlib.import_module('final144_runner'), importlib.import_module('final144_recipe')
        recipe = recipe_module.recipe()
        need(sha(canonical(recipe)) == pins['recipe_sha256'], 'FROZEN_RECIPE_BYTES')
        binding, source = runner.review_rows(args.source_parent, args.observations_parent, batch_pins)
        need(binding.get('test_backend') is False and binding.get('run_id') == os.environ['GITHUB_RUN_ID'] and
             binding.get('github_sha') == os.environ['GITHUB_SHA'], 'CURRENT_REAL_SOURCE_RUN_BINDING')
        source_rows, wavs = [row for row, _ in source], {row['cell_id']: regular(path, 512 * 1024) for row, path in source}
        selection = select_intents(recipe, source_rows, binding)
        once(output / 'SOURCE_BINDING.json', binding)
        once(output / 'SELECTION.json', selection)
        libs, code = pipeline_libraries(args.pipeline_source, pins['pipeline_commit'])
        excluded = []
        try:
            rows = libs['weak_embedding_contract'].weak_rows(recipe, source_rows,
                eligible_cell_ids=selection['eligible_cell_ids'], observations=selection['observations'], excluded=excluded)
        except ValueError as error:
            if str(error) not in ('EXPLICIT_INTENT_RESEARCH_SELECTION_REQUIRED', 'MISSING_WEAK_CLASS_OR_ROLE'):
                raise
            once(output / 'EXCLUSIONS.json', excluded)
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_MINIMUM_EVIDENCE_BEFORE_TRAINING',
                'optimizer_updates': 0, 'reason': str(error),
                'minimum_stage': 'LIBRARY_REQUIRES_ALL_SIX_WEAK_CLASS_ROLE_GROUPS'})
            return 2
        minimum = minimum_evidence(rows)
        once(output / 'MINIMUM_AFTER_DEDUPLICATION.json', minimum)
        if not minimum['passed']:
            once(output / 'EXCLUSIONS.json', excluded)
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_MINIMUM_EVIDENCE_BEFORE_TRAINING',
                'optimizer_updates': 0, 'minimum': minimum})
            return 2
        stage = 'runtime-and-background-preflight'
        portable = libs['portable_head']
        native, arm, build = build_replays(portable, output)
        once(output / 'BUILD.json', build)
        reference = reference_records(args.reference_zip, libs['librispeech'])
        background, decoder = decode_background(args.corpus_archive, reference, libs['librispeech'])
        tracks = libs['weak_audio_cases'].background_tracks(background)
        once(output / 'LIBRISPEECH_SOURCES.json', {'reference': REFERENCE, 'corpus': CORPUS, 'decoder': decoder,
            'decoded_originals': [{k: v for k, v in row.items() if k != 'pcm16le'} for row in background],
            'actual_research_tracks': [{k: v for k, v in track.items() if k != 'pcm16le'} for track in tracks],
            'unique_research_track_samples': sum(t['frames'] for t in tracks), 'prior_events_reused': False,
            'source_pool_is_previously_exposed': True})
        del background
        cases = libs['weak_audio_cases'].assemble_audio_cases(rows, wavs, tracks, excluded=excluded)
        stage = 'frozen-feature-extraction'
        embedding = libs['causal_embedding'].FrozenOnnxEmbedding(args.assets)
        expected_calls = sum(max(0, 1 + (len(c['pcm16le']) // 2 - 16000) // 1280) for c in cases)
        need(expected_calls <= 40000, 'FINITE_FROZEN_EMBEDDING_BUDGET')
        once(output / 'FEATURE_PREFLIGHT.json', {'geometry': embedding.geometry,
            'maximum_embedding_calls': expected_calls, 'asset_identities': libs['weak_embedding_contract'].ASSETS,
            'pipeline_source': code})
        bags, streams, evidence = libs['weak_audio_cases'].extract_feature_bags(cases, embedding, excluded=excluded)
        need(sum(e['embedding_calls'] for e in evidence) <= expected_calls, 'FEATURE_FORWARD_ACCOUNTING')
        evidence_sha = once(output / 'FEATURE_EVIDENCE.json', evidence)
        once(output / 'EXCLUSIONS.json', excluded)
        usable = {bag['id'] for bag in bags}
        final_minimum = minimum_evidence([row for row in rows if row['cell_id'] in usable])
        once(output / 'MINIMUM_AFTER_FEATURE_GEOMETRY.json', final_minimum)
        if not final_minimum['passed']:
            once(output / 'RESULT.json', {**started, 'status': 'STOPPED_FEATURE_ELIGIBILITY_BEFORE_TRAINING',
                'optimizer_updates': 0, 'minimum': final_minimum})
            return 2
        feature_stream_sha = save_stream_cache(streams, cases, output)
        input_binding = {'source_manifest_sha256': binding['source_manifest_sha256'],
            'observation_manifest_sha256': selection['observer_artifact_manifest_sha256'],
            'recipe_sha256': pins['recipe_sha256'], 'selection_sha256': sha(canonical(selection)),
            'corpus_sha256': CORPUS['sha256'], 'reference_sha256': REFERENCE['member_sha256'],
            'pipeline_source_sha256': sha(canonical(code)), 'feature_evidence_sha256': evidence_sha,
            'feature_stream_sha256': feature_stream_sha,
            'worker_code_sha256': started['worker_code_sha256']}
        stage = 'fixed-96-step-training'
        once(output / 'TRAINING_STARTED.json', {'maximum_optimizer_updates': 96, 'input_binding': input_binding,
                                               'development_tuning': False, 'no_retry': True})
        training_started = True
        training_result = libs['causal_head'].train_cached(bags, output / 'train', input_binding)
        need(training_result['optimizer_updates'] == 96, 'EXACT_96_OPTIMIZER_UPDATES')
        stage = 'saved-weight-host-and-arm-replay'
        reports, per_case = replay_saved_streams(portable, output / 'train/causal-head.npz', output, native, arm)
        once(output / 'PARITY.json', reports)
        once(output / 'PER_CASE_EVENTS.json', per_case)
        track_ids = {track['id'] for track in tracks}
        track_cases = [row for row in per_case if row['id'] in track_ids]
        background_events = sum(len(row['events']) for row in track_cases)
        track_seconds = sum(track['frames'] for track in tracks) / 16000
        result = {**started, 'status': 'EXECUTED_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY',
            'optimizer_updates': 96, 'weights_sha256': training_result['weights_sha256'],
            'head_binary_sha256': sha((output / 'causal-head.bin').read_bytes()),
            'frozen_embedding_calls': sum(e['embedding_calls'] for e in evidence),
            'source_denominator': 144, 'selected_after_deduplication': len(rows),
            'usable_source_cases': sum(row['cell_id'] in usable for row in rows),
            'excluded_rows': excluded, 'minimum': final_minimum, 'input_binding': input_binding,
            'exposed_read_speech_tracks': {'tracks': len(track_cases), 'seconds': track_seconds,
                'events': background_events, 'research_events_per_hour': background_events * 3600 / track_seconds,
                'reset_per_speaker': True, 'unique_exposure_includes_no_reused_prefixes': True,
                'independent_continuous_far': False},
            'wall_seconds': time.monotonic() - clock, 'new_tts_asr_phone_calls': 0,
            'independent_frr': 'NOT_RUN', 'independent_far': 'NOT_RUN', 'word_end_latency': 'NOT_RUN',
            'ssc305': 'NOT_RUN', 'deployment_frontend_arm_parity': 'NOT_RUN',
            'arm_scope': 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU'}
        once(output / 'RESULT.json', result)
        return 0
    except Exception as error:
        once(output / 'FAILURE.json', {**started, 'status': 'FAILED_NO_RETRY', 'stage': stage,
            'error_type': type(error).__name__, 'reason': str(error)[:500],
            'optimizer_updates': training_result['optimizer_updates'] if training_result is not None else
                ('UNKNOWN_AFTER_TRAINING_STARTED' if training_started else 0)})
        return 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--pins', type=Path, default=HERE / 'final144_training_pins.json')
    cli.add_argument('--batch-pins', type=Path, default=HERE / 'final144_execution_pins.json')
    for name in ('source-parent', 'observations-parent', 'pipeline-source', 'corpus-archive',
                 'reference-zip', 'assets', 'output'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'selection_sha256': sha(canonical(SELECTION)),
                          'model_calls': 0, 'optimizer_updates': 0, 'execution_enabled': False}))
        return 0
    need(all(getattr(args, name) is not None for name in ('source_parent', 'observations_parent',
         'pipeline_source', 'corpus_archive', 'reference_zip', 'assets', 'output')), 'EXPLICIT_WORKER_INPUTS_REQUIRED')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
