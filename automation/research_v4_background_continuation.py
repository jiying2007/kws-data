#!/usr/bin/env python3
"""One consumed, paired weak-background continuation; no generic training runner."""

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "background_fixed_input_helpers", Path(__file__).with_name("research_v4_native_c_librispeech.py"))
speech = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(speech)
core, native, state_consumer = speech.core, speech.native, speech.state_consumer
exact, fields, integer = speech.exact, speech.fields, speech.integer
file_sha256, validate_file = speech.file_sha256, speech.validate_file
download_corpus, validate_negative = speech.download_corpus, speech.validate_negative

FIXED = {'schema': 'kws-research-v4-background-continuation-request-v1',
 'experiment': 'english63-natural-background-continuation',
 'request_id': 'en63-background48-20261115-v1',
 'pipeline_repository': 'jiying2007/kws-pipeline',
 'entrypoint': 'research/english63_background_ab/run_ab.py',
 'source_artifacts': {'trained': {'run_id': 43595338774,
                                  'artifact_id': 14114695344,
                                  'zip_bytes': 100653,
                                  'zip_sha256': 'bc76805934df8eb158fa4259a75c26a4ea4276c15b83542acad5948a1f954b54',
                                  'data_commit': '14547faa42299ae9d819ba4bff9824c2471c412e',
                                  'pipeline_commit': 'fc59bef7fe108ce6accbf51d8de0bfcaae19e0bc',
                                  'terminal_sha256': '7c840f2877c0d80a147b5d1287261faeb01495e61f8a2d8ab8cf9053c6cc91a4',
                                  'summary_sha256': '09db16f5947d39c1824565b20defc16a2349f09fea5621b8a935989f65555295',
                                  'weights_sha256': {'gru': '71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623',
                                                     'dsc_tcn': '167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935'}},
                      'tts': {'artifact_id': 13954744587,
                              'zip_sha256': 'f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc'},
                      'review': {'artifact_id': 14001747172,
                                 'zip_sha256': 'f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8',
                                 'member': 'english63-three-evidence-review.json',
                                 'member_sha256': '89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12'},
                      'negative_reference': {'run_id': 43538998200,
                                             'artifact_id': 14090383839,
                                             'zip_bytes': 47199,
                                             'zip_sha256': '7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4',
                                             'member': 'english-natural-read-speech-result.json',
                                             'member_sha256': '788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510'},
                      'corpus': {'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
                                 'bytes': 337926286,
                                 'sha256': '76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3',
                                 'license': 'CC-BY-4.0'}},
 'profile': {'models': ['gru', 'dsc_tcn'],
             'variants': ['baseline', 'control', 'treatment'],
             'speaker_hash_domain': 'en63-bg48-speaker-v1/',
             'record_hash_domain': 'en63-bg48-record-v1/',
             'natural_source_clips': 517,
             'natural_source_speakers': 40,
             'natural_source_samples': 58646164,
             'train_speakers': 32,
             'study_dev_speakers': 8,
             'records_per_train_speaker': 4,
             'natural_train_windows': 128,
             'crop_samples': 48000,
             'crop_rule': 'floor((source_samples-48000)/2)',
             'crop_frames': 298,
             'natural_train_samples': 6144000,
             'natural_study_dev_clips': 102,
             'natural_study_dev_samples': 11124244,
             'natural_study_dev_frames': 69525,
             'natural_study_dev_order': 'lexical-source-member',
             'original_train_clips': 30,
             'tts_study_dev_clips': 12,
             'tts_study_dev_frames': 2440,
             'tts_study_dev_order': 'lexical-cell-id-independent-reset',
             'steps_per_arm': 48,
             'training_arms': 4,
             'optimizer_updates': 192,
             'train_microbatch_natural': 32,
             'natural_loss_weight': 0.5,
             'objective': 'control=old30-MIL-CE;treatment=old30-MIL-CE+0.5*new128-background-MIL-CE',
             'seeds': {'gru': 20261114, 'dsc_tcn': 20261215},
             'learning_rate': 0.001,
             'weight_decay': 0.01,
             'gradient_clip': 1,
             'cpu_threads': 1,
             'mkldnn': False,
             'training_arm_seconds': 180,
             'training_group_seconds': 720,
             'native_model_seconds': 120,
             'native_group_seconds': 180,
             'native_model_passes': 6,
             'native_segments_per_model': 13,
             'native_eval_frames': 431790,
             'feed_chunk_samples': 160,
             'pool_time_major': 0,
             'window_frames': 200,
             'minimum_observed_frames': 50,
             'mil_temperature': 0.25,
             'threshold': 0.8,
             'confirm_frames': 3,
             'cooldown_frames': 100,
             'baseline_header_sha256': '912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16',
             'benefit_gate': 'B>0,C>0,2*T<=B,2*T<=C;each-natural-class-T<=min(B,C);each-keyword-correct-clips>=max(B,C,2);each-keyword-wrong-other-clips<=min(B,C);ordinary-background-proposals<=min(B,C)'},
 'max_execution_seconds': 1200,
 'max_verification_seconds': 180,
 'new_audio_generated': False,
 'new_tts_asr_ipa_calls': 0,
 'new_independent_audio_seconds': 0,
 'formal_training_admitted': False,
 'human_gold': False,
 'model_promotion_allowed': False,
 'model_selection_made': False,
 'threshold_selection': False}
PIPELINE_COMMIT = "849040b48bb813d61722de7e1cdd0d0df20ca828"
SOURCE_SHA256 = {'research/english63_background_ab/ab_contract.py': '0ea193d7b2353788686a513146c30313825ff23c70c35b70fe60c02fb265cc9f',
 'research/english63_background_ab/ab_replay.c': '24d902be2de3a5b18efac68a41a107040f012332c5bce912137d7475944989aa',
 'research/english63_background_ab/protocol.json': 'ac7c7c80cd84b3ea2327c70f8eed06fb787680856b422ac8841f7c84fe6ba7a3',
 'research/english63_background_ab/run_ab.py': 'c00b51f5249755b9c5637c9fe8d9c1436c2164e5e78a6ed0bec569c72a0ae8c0',
 'research/english63_online_c/contract.py': '5a9c1d496daac3fc07dd80e2423c9a995a30f8da012abfac626ab8e5a9a265b5',
 'research/english63_online_c/en63_stream.c': 'b8cb0926c185d1200d27dae194dd83d27d7f08d83acd49d9526bc363d3cee500',
 'research/english63_online_c/en63_stream.h': '627f18dacede547d6aed9988a745071f4e4f988252d16a4646ebe4a09eeebeb7',
 'research/english63_online_c/export_weights.py': '670638aa458722acd0c6c46d5e189b13ca1568d8e1124ed52b14081167104411',
 'research/english63_online_c/librispeech_screen.py': 'e416b11e501a3f788c3d13a221a9b9747a4e32819e03aaaf3f65cf2dc4da99d2',
 'research/english63_online_c/pool_profile.py': 'cc9485e64e31e3f80d6a2f9b951346e7ebcf3c94b06406f61eebc0dc6d93a577',
 'research/english63_online_pilot/online_contract.py': '60eb2fa4ae8c5558824a35d085b27e879126b6b46b207b682605b0546d148969',
 'research/english63_online_pilot/online_pilot.py': '63748a1bedf975bbd3586e5cb53be98f190e93550e9a112c59fe847efb1ac128',
 'research/english63_weak_pilot/PCM_FRONTEND_PARITY.json': 'd0f4ab7c4911ec4d2bcf942197170b3f8668e94dc4ca9432431e1103fe1e61bd',
 'research/english63_weak_pilot/RESULT.json': 'ca16154bcc4100749870d264ade8594ccf416001ab66a3397fbc8d32afc94418',
 'research/english63_weak_pilot/english63_weak_train.py': '4ffa8b8c700b3d1413711be2a2329767dfc9bbdee13b77ca990a2c22a64e0719',
 'research/english63_weak_pilot/integrated_pcm_encoder.py': '37a1887e5822d73bb9f6c681d93feb4cceb383412406c6c5627bbd345caa944f',
 'research/english63_weak_pilot/librispeech_negative_events.py': '2ed629ddd2b8aa3ec3b2dc03e8e6e825e6608fa7719c27e9b253380f7267c32b',
 'research/english63_weak_pilot/pcm_frontend_parity.py': '693a282d490c2d2a7f8468a4ed9ae569f2c8b0dde3bc4714001ed537520a12ae',
 'research/english63_weak_pilot/pretrained_quant_events.py': '97714a6d9cd7fecf1c98d0b0f24a091d4644c4924dacdd564b527ad8b2962348',
 'research/english63_weak_pilot/streaming_parity.py': 'e53f23ee5a2ade0f33b8ac21f4ac67bcc8338d8b3f9a378c980ce312f19d87f0',
 'tools/corpus_identity.py': '0931b720e1be9d4a2ce8d54d9b8abcc748700ac6c805a66fe39d4cc5cd0e99af',
 'tools/source_identity.py': 'fad7b9daefc5f865e9c064655253e79c0c0e0d055b861d2444796c1b3abcd2b1',
 'training/audit_dataset.py': '40f01f7aaeb74a13670da385d9a830ddb8c2743afd566ed5293afe968320908d',
 'training/english_research_label_bridge.py': '5e1942dd989f0d9c8737cf1cb05090cba403780a2ca8b58816e05eb25249be2f'}
EXPERIMENT = "english63-natural-background-continuation"
REQUEST_ID = "en63-background48-20261115-v1"
ENTRYPOINT = "research/english63_background_ab/run_ab.py"
RESULT_DIR = "background-continuation"
KINDS, VARIANTS = ("gru", "dsc_tcn"), ("baseline", "control", "treatment")
TRACE_FILES = {variant + "-" + kind + ".trace" for variant in VARIANTS for kind in KINDS}
RESULT_FILES = {"PROTOCOL.json", "INPUTS.json", "SOURCE.zip", "TRAINING.json", "BUILD.json",
                "MEASUREMENTS.json", "SUMMARY.json", "REPLAY.json", "native.stderr", "WEIGHTS.zip", "RUNTIME.zip"} | TRACE_FILES
UPDATE_BUDGET, FORWARD_FRAMES = 192, 431790
WORK_COUNTERS = {"optimizer_updates": UPDATE_BUDGET, "native_forward_frames": FORWARD_FRAMES,
                 "native_initializations": 78, "native_finishes": 78,
                 "native_feed_calls": 431946, "native_supplied_samples": 69110904}
SEGMENT_SAMPLES = (11124244, 20480, 46080, 47360, 47360, 28160, 26880,
                   20480, 30720, 30720, 48640, 24320, 23040)
TRACE_BYTES = 2303208
WEIGHT_FILES = {variant + "-" + kind + ".npz" for variant in ("control", "treatment") for kind in KINDS}
RUNTIME_FILES = {variant + extension for variant in VARIANTS for extension in (".h", ".bin")}
FALSE_FIELDS = ("formal_training_admitted", "human_gold", "trusted_negative", "model_promotion_allowed",
                "model_selection_made", "threshold_selection", "runtime_default_changed", "new_audio_generated")
DOWNLOADS = {
    "trained": {**native.CHECKPOINT, "zip_bytes": 100653, "event": "push"},
    "tts": {**core.INPUTS["tts"], "run_id": 43242278962, "zip_bytes": 14167568,
            "data_commit": "bec1a1421d1db2de17afb9c6e3155343becdb980", "event": "push"},
    "review": {**core.INPUTS["review"], "run_id": 43346200939, "zip_bytes": 21217,
               "data_commit": "fb2b4deeb87dd24ce2796df7716bbb365445d58d", "event": "push"},
    "negative": {**speech.NEGATIVE, "data_commit": speech.NEGATIVE_HEAD, "event": "pull_request"},
}


def request_value(raw):
    core.require(len(raw) < 32768, "background request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(type(request) is dict and set(request) == set(FIXED) | {"pipeline_commit", "reason"},
                 "background request keys differ")
    for key, expected in FIXED.items():
        core.require(exact(request[key], expected), "unsupported background request field: " + key)
    core.require(request["pipeline_commit"] == PIPELINE_COMMIT and core.valid_sha(PIPELINE_COMMIT),
                 "background source differs from the reviewed commit")
    core.require(type(request["reason"]) is str and 1 <= len(request["reason"].strip()) <= 500,
                 "background request needs a bounded reason")
    return request


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    request = request_value(raw)
    core.require(path.name == REQUEST_ID + ".json", "background request filename differs")
    return request, raw, core.sha256(raw)


def validate_inputs(state):
    for name, specification in DOWNLOADS.items():
        validate_file(state / "inputs" / (name + ".zip"), specification["zip_bytes"],
                      specification["zip_sha256"], name + " source archive")
    native.validate_checkpoint(state / "inputs/trained-result")
    validate_negative(state / "inputs/negative-result")
    review = state / "inputs/review.json"
    core.require(review.is_file() and not review.is_symlink() and
                 file_sha256(review) == core.INPUTS["review"]["member_sha256"], "original review member changed")
    validate_file(state / "inputs/dev-clean.tar.gz", speech.CORPUS["bytes"], speech.CORPUS["sha256"], "original corpus")


def claim_record(request, digest, run=None):
    run = run or {"run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"]}
    return {"schema": "kws-research-v4-background-continuation-start-claim-v1", "request_sha256": digest,
            "pipeline_commit": request["pipeline_commit"], "source_checkpoint_artifact_id": 14114695344,
            "source_negative_artifact_id": 14090383839, "run_id": run["run_id"], "run_attempt": run["run_attempt"],
            "optimizer_updates": 0, "optimizer_update_budget": UPDATE_BUDGET,
            "status": "CONSUMED_BEFORE_BACKGROUND_CONTINUATION"}


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and
                 os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main" and
                 event["repository"]["private"] is False, "background execution is public main only")
    state.mkdir()
    result_name, claim_name = "research-v4-background-result-" + digest, "research-v4-background-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous background receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        terminal = validate_terminal(state / "reused", request, digest)
        fields(terminals[0].get("workflow_run"), {"id": int(terminal["run_id"]), "head_sha": terminal["data_commit"]},
               "background reused archive public run")
        core.output(execute=False, request_sha256=digest, result_artifact=result_name,
                    prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior background claim without valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "background rerun without receipts is blocked")
    for name in ("inputs", "result", "claim"):
        (state / name).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    for name, specification in DOWNLOADS.items():
        run = core.github_get("/actions/runs/" + str(specification["run_id"]))
        fields(run, {"status": "completed", "conclusion": "success", "run_attempt": 1,
                     "event": specification["event"], "head_sha": specification["data_commit"]},
               name + " original successful run")
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        fields(metadata, {"size_in_bytes": specification["zip_bytes"]}, name + " archive size")
        fields(metadata.get("workflow_run"), {"id": specification["run_id"], "head_sha": specification["data_commit"]},
               name + " archive public run")
        archive = state / "inputs" / (name + ".zip")
        archive.write_bytes(core.artifact_bytes(metadata))
        validate_file(archive, specification["zip_bytes"], specification["zip_sha256"], name + " archive")
        if name in ("trained", "negative"):
            core.restore_archive(archive, state / "inputs" / (name + "-result"))
    with zipfile.ZipFile(state / "inputs/review.zip") as archive:
        member = core.INPUTS["review"]["member"]
        core.require(archive.namelist().count(member) == 1, "review member absent or duplicated")
        info = archive.getinfo(member)
        core.require(not info.is_dir() and info.file_size < 8 * 1024 * 1024 and
                     (info.external_attr >> 16) & 0o170000 != 0o120000, "invalid review member")
        review = archive.read(member)
    core.require(core.sha256(review) == core.INPUTS["review"]["member_sha256"], "review JSON digest changed")
    (state / "inputs/review.json").write_bytes(review)
    native.validate_checkpoint(state / "inputs/trained-result")
    validate_negative(state / "inputs/negative-result")
    download_corpus(state / "inputs/dev-clean.tar.gz")
    validate_inputs(state)
    core.write_json(state / "claim/CLAIM.json", claim_record(request, digest))
    core.output(execute=True, request_sha256=digest, result_artifact=result_name,
                claim_artifact=claim_name, prior_artifact_id="")


def verified_source(archive_path):
    """Only the complete reviewed source closure may become an executable reader."""
    core.require(ENTRYPOINT in SOURCE_SHA256 and all(core.valid_sha(value, 64) for value in SOURCE_SHA256.values()),
                 "reviewed background source closure incomplete")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        core.require(len(infos) == len(SOURCE_SHA256) and set(archive.namelist()) == set(SOURCE_SHA256) and
                     all(not info.is_dir() and 0 < info.file_size < 4 * 1024 * 1024 and
                         (info.external_attr >> 16) & 0o170000 != 0o120000 for info in infos),
                     "background source path closure differs")
        source = {name: archive.read(name) for name in SOURCE_SHA256}
    core.require(all(core.sha256(raw) == SOURCE_SHA256[name] for name, raw in source.items()),
                 "background source bytes differ from the reviewed allowlist")
    return source


def run_saved_verifier(root, source):
    with tempfile.TemporaryDirectory(prefix="en63-background-saved-reader-") as directory:
        temporary = Path(directory)
        tree = temporary / "source"
        tree.mkdir()
        for name, raw in source.items():
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        python = Path(sys.executable).resolve()
        command = [str(python), "-I", "-S", "-B", str(tree / ENTRYPOINT), "--verify-result-dir", str(root.resolve())]
        process = subprocess.run(command, env=core.child_environment(python, temporary), cwd=tree,
                                 capture_output=True, text=True, timeout=180, check=False)
        core.require(type(process.returncode) is int and process.returncode == 0 and
                     len(process.stdout) < 65536 and process.stderr == "", "frozen background saved-only verifier failed")
        return json.loads(process.stdout, object_pairs_hook=core.unique_object)


def validate_checkout_sources(pipeline):
    core.require(ENTRYPOINT in SOURCE_SHA256, "reviewed background checkout source closure incomplete")
    for name, expected in SOURCE_SHA256.items():
        path = pipeline / name
        core.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(pipeline.resolve()) and
                     file_sha256(path) == expected, "background checkout differs from reviewed source: " + name)


def execution_ok(execution, digest):
    fields(execution, {"schema": "kws-research-v4-background-continuation-execution-v1", "request_sha256": digest,
           "execute_exit_code": 0, "verify_exit_code": 0, **WORK_COUNTERS,
           "optimizer_update_budget": UPDATE_BUDGET,
           "source_inputs_before_after_verified": True}, "background execution incomplete")


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() ==
                 request["pipeline_commit"], "background checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip(), "background tracked source modified")
    validate_checkout_sources(pipeline)
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()), "fixed background entrypoint missing")
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "show", request["pipeline_commit"] + ":" + ENTRYPOINT]) ==
                 script.read_bytes(), "background entrypoint differs from reviewed commit")
    core.require(exact(core.read_json(state / "claim/CLAIM.json"), claim_record(request, digest)), "background claim differs")
    result = state / "result"
    environment = state_consumer.state_environment(python, state)
    command = [str(python), "-I", "-B", str(script), "--trained-result-dir", str(state / "inputs/trained-result"),
               "--tts-zip", str(state / "inputs/tts.zip"), "--review-json", str(state / "inputs/review.json"),
               "--negative-evidence-dir", str(state / "inputs/negative-result"),
               "--source-tar", str(state / "inputs/dev-clean.tar.gz"), "--output-dir", str(result / RESULT_DIR)]
    execution = {"schema": "kws-research-v4-background-continuation-execution-v1", "request_sha256": digest,
                 "execute_exit_code": None, "verify_exit_code": None, **dict.fromkeys(WORK_COUNTERS),
                 "optimizer_update_budget": UPDATE_BUDGET,
                 "source_inputs_before_after_verified": False, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_inputs(state)
        with (result / "execute.log").open("xb") as log:
            execution["execute_exit_code"] = native.run_bounded(command, pipeline=pipeline, environment=environment,
                                                               log=log, seconds=1200)
        core.require(execution["execute_exit_code"] == 0, "background execution failed; retain partial evidence")
        source = verified_source(result / RESULT_DIR / "SOURCE.zip")
        verified = run_saved_verifier(result / RESULT_DIR, source)
        core.write_json(result / "verify.log", verified)
        execution["verify_exit_code"] = 0
        summary = core.read_json(result / RESULT_DIR / "SUMMARY.json")
        core.require(type(summary) is dict, "background summary must be an object")
        execution.update({key: summary.get(key) for key in WORK_COUNTERS})
        validate_inputs(state)
        execution["source_inputs_before_after_verified"] = True
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        core.write_json(result / "EXECUTION.json", execution)
    execution_ok(execution, digest)


def validate_trace_geometry(path):
    """Independently enforce six exact 13-segment workloads before saved decoding."""
    core.require(path.stat().st_size == TRACE_BYTES, "background trace byte count differs")
    with path.open("rb") as stream:
        core.require(stream.read(16) == b"EN63ABT1" + struct.pack("<II", 1, 13),
                     "background trace container header differs")
        total = 0
        for index, samples in enumerate(SEGMENT_SAMPLES):
            frames = (samples - 400) // 160 + 1
            core.require(stream.read(24) == struct.pack("<IIQQ", index, 32, samples, frames),
                         "background trace segment geometry differs")
            stream.seek(frames * 32, os.SEEK_CUR)
            total += frames
        core.require(stream.read(1) == b"" and total == 71965, "background trace trailing bytes or frame count")
    return {"native_forward_frames": total, "native_initializations": len(SEGMENT_SAMPLES),
            "native_finishes": len(SEGMENT_SAMPLES), "native_feed_calls": sum((n + 159) // 160 for n in SEGMENT_SAMPLES),
            "native_supplied_samples": sum(SEGMENT_SAMPLES)}


def closed_archive(path, expected, maximum):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        core.require(len(infos) == len(expected) and set(archive.namelist()) == expected and
                     all(not info.is_dir() and 0 < info.file_size <= maximum and
                         (info.external_attr >> 16) & 0o170000 != 0o120000 for info in infos),
                     "background nested archive file closure differs")
        return {name: archive.read(name) for name in expected}


def validate_result(result):
    root = result / RESULT_DIR
    core.require(root.is_dir() and not root.is_symlink() and
                 all(path.is_file() and not path.is_symlink() for path in root.iterdir()),
                 "background result must contain flat regular files")
    manifest = core.file_manifest(root)
    core.require(set(manifest) == RESULT_FILES and all((root / name).stat().st_size > 0
                 for name in RESULT_FILES if name != "native.stderr"), "background result file closure differs")
    outer = {"REQUEST.json", "EXECUTION.json", "execute.log", "verify.log", "runtime.txt"} | {
        RESULT_DIR + "/" + name for name in RESULT_FILES}
    core.require(set(core.file_manifest(result)) == outer, "background outer result closure differs")
    trace_work = [validate_trace_geometry(root / name) for name in TRACE_FILES]
    core.require(all(sum(row[key] for row in trace_work) == value for key, value in WORK_COUNTERS.items()
                     if key != "optimizer_updates"), "background total saved trace workload differs")
    weights = closed_archive(root / "WEIGHTS.zip", WEIGHT_FILES, 500000)
    core.require(all(raw.startswith(b"PK\x03\x04") for raw in weights.values()), "background saved NPZ missing")
    runtime = closed_archive(root / "RUNTIME.zip", RUNTIME_FILES, 1024 * 1024)
    core.require(core.sha256(runtime["baseline.h"]) == FIXED["profile"]["baseline_header_sha256"] and
                 all(runtime[variant + ".bin"].startswith(b"\x7fELF") for variant in VARIANTS),
                 "background frozen baseline header or native binaries differ")
    source = verified_source(root / "SOURCE.zip")
    summary, inputs, build, training, replay = [core.read_json(root / name) for name in (
        "SUMMARY.json", "INPUTS.json", "BUILD.json", "TRAINING.json", "REPLAY.json")]
    core.require(all(type(report) is dict for report in (summary, inputs, build, training, replay)),
                 "background receipts must be objects")
    request_raw = (result / "REQUEST.json").read_bytes()
    request, digest = request_value(request_raw), core.sha256(request_raw)
    run = build.get("public_run")
    core.require(type(run) is dict and set(run) == {"repository", "head", "run_id", "run_attempt"} and
                 run["repository"] == core.REPOSITORY and core.valid_sha(run["head"]) and
                 type(run["run_id"]) is str and run["run_id"].isdigit() and int(run["run_id"]) > 0 and
                 run["run_attempt"] == "1", "background run identity differs")
    for report in (summary, build):
        fields(report, {"public_run": run, "pipeline_head": request["pipeline_commit"]}, "background run/source binding")
    fields(inputs, {"request_raw_utf8": request_raw.decode("utf-8"), "request_sha256": digest},
           "background retained request identity")
    claim_raw = inputs.get("claim_raw_utf8")
    core.require(type(claim_raw) is str and core.sha256(claim_raw.encode()) == inputs.get("claim_sha256") and
                 exact(json.loads(claim_raw, object_pairs_hook=core.unique_object), claim_record(request, digest, run)),
                 "background consumed claim identity differs")
    fields(build, {"source_sha256": SOURCE_SHA256}, "background reviewed source closure")
    fields(training, {"optimizer_updates_started": UPDATE_BUDGET, "optimizer_updates_completed": UPDATE_BUDGET,
                      "active_training": None}, "background training group incomplete")
    fields(summary, {"schema": "english63-background-ab-result-v1", "experiment": EXPERIMENT,
           "request_sha256": digest, **WORK_COUNTERS,
           "new_independent_audio_seconds": 0, "new_tts_asr_ipa_calls": 0,
           **dict.fromkeys(FALSE_FIELDS, False)}, "background summary budget or scientific boundary")
    expected_hashes = {name: digest for name, digest in manifest.items() if name != "REPLAY.json"}
    expected_replay = {"schema": "english63-background-ab-replay-v1", "status": "VERIFIED_SAVED_BACKGROUND_AB",
        "files": {name: {"bytes": (root / name).stat().st_size, "sha256": value} for name, value in expected_hashes.items()},
        "files_sha256": expected_hashes, "summary_sha256": manifest["SUMMARY.json"], "model_forward_calls": 0,
        "optimizer_updates": 0, "new_audio_asr_calls": 0, "formal_training_admitted": False,
        "model_promotion_allowed": False}
    core.require(exact(replay, expected_replay), "background saved receipt manifest/scope differs")
    verified = run_saved_verifier(root, source)
    core.require(exact(verified, expected_replay), "frozen saved decoder did not reproduce the background receipt")


def validate_terminal_identity(result, terminal):
    run = core.read_json(result / RESULT_DIR / "SUMMARY.json")["public_run"]
    fields(terminal, {"data_commit": run["head"], "run_id": run["run_id"], "run_attempt": run["run_attempt"]},
           "background terminal public run binding")


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    fields(terminal, {"schema": "kws-research-v4-background-continuation-terminal-v1", "request_sha256": digest,
           "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"], **WORK_COUNTERS,
           "optimizer_update_budget": UPDATE_BUDGET,
           "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False},
           "background terminal identity/scope")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and
                 terminal.get("files_sha256") == core.file_manifest(result), "background terminal manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED", "prior background attempt failed; request stays consumed")
    execution_ok(core.read_json(result / "EXECUTION.json"), digest)
    validate_result(result)
    validate_terminal_identity(result, terminal)
    return terminal


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {"schema": "kws-research-v4-background-continuation-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), **dict.fromkeys(WORK_COUNTERS),
        "optimizer_update_budget": UPDATE_BUDGET,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    try:
        core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest, "retained background request changed")
        execution = core.read_json(result / "EXECUTION.json")
        execution_ok(execution, digest)
        validate_result(result)
        validate_terminal_identity(result, terminal)
        terminal.update(status="SUCCEEDED", **WORK_COUNTERS)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        terminal["error"] = type(exc).__name__ + ": " + str(exc)
    terminal["files_sha256"] = core.file_manifest(result)
    core.write_json(result / "TERMINAL.json", terminal)
    core.output(terminal_status=terminal["status"])


def dispatch(args):
    if args.command == "prepare":
        prepare(args.request, args.state)
    elif args.command == "execute":
        core.require(args.pipeline is not None and args.pipeline.is_absolute() and
                     args.python is not None and args.python.is_absolute(), "absolute background source/Python required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported background operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": UPDATE_BUDGET, "native_forward_frames": FORWARD_FRAMES,
                          "model_promotion_allowed": False}, sort_keys=True))
