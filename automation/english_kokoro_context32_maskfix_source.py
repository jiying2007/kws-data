"""Read the actual failed-preprocessing packet; reuse full PCM without derivation.

The previous two model constructions and thirty-two preprocessing attempts are
retained. No previous phone forward occurred. Only the exact closed packet below
authorizes this independent, bounded first-forward recovery.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "english-kokoro-context32-maskfix-v1"
CAMPAIGN_ID = "kokoro-full-context32-native-int32-first-forward-v1"
MAIN_BEFORE = "965bb0cc7150d244ac4f06f512f58a3824afbab2"
PRIOR_RUN = {"repository":"jiying2007/kws-data","commit":MAIN_BEFORE,
    "run_id":"44133547091","run_attempt":"1","event":"push"}
INPUT_MANIFEST_SHA256 = "f98935a551ea7d78d9118820e9b1c43d1f5cad840ba962df778d813881813cf3"
SOURCE_PLAN_SHA256 = "f5bf386a956bce3d1e1bdeee2ebf243026a284ba211d12bc780ca5a3d2a34d9a"
ROWS_SHA256 = "731e093b014efc49593f9502094f6cc6fd77bdd52af72452dc7bebafff301acf"
RECOVERY = {"recovery_id":"context32-native-int32-first-forward-after-44133547091-v1",
    "repository":"jiying2007/kws-data","prior_run_id":44133547091,"prior_head_sha":MAIN_BEFORE,
    "prior_workflow":".github/workflows/english-kokoro-context32-once.yml",
    "prior_artifact":{"id":14350388434,"name":"english-kokoro-context32-44133547091",
        "bytes":129737354,"sha256":"e62a5233353efd4d07974213bef0a4025b056ea4cac7d6d1035ebc96907be378"},
    "prior_prepared_inputs":16,"prior_model_constructions":2,"prior_processor_attempts":32,
    "prior_model_forward_attempts":0,"prior_model_forward_returns":0,"prior_captures":0,
    "prior_error":"ValueError: ACTUAL_UNPADDED_ATTENTION_MASK",
    "recovery_of_failed_preprocessing":True,"same_existing_full_pcm":True,
    "new_derivations":0,"new_tts_calls":0,"forward_replays":0,"automatic_retry":False}
PREPARED_SOURCE_PIN = "8e355666f70342b6cf422deaaa44e541eeaf450d3c0e8327e3f2ed9a825a36f7"
PRIOR_CONTEXT_PINS = {
    "automation/english_kokoro_context32_source.py":PREPARED_SOURCE_PIN,
    "automation/english_kokoro_context32_once.py":"f2f3b01e5b43b2008f71293fd2cccf49b969ebc2fdcd48d284c87091890cd211",
    "automation/english_kokoro_context32_observe.py":"aa4694b84cfaa88436303ea67a19da02d44be2a7e19e4b3202192a9dcbbeee9a",
    "automation/english_kokoro_context32_compare.py":"131af8474dd61d339efe1748f504fc9593e0da7462f9b0abcea0a2899d78904d"}
INPUT_FILES = ("prior-final.zip","prior-final.artifact.json","prior-run.json","prior-jobs.json")
MAX_FILE = 256*1024**2
MAX_EXPANDED = 1024*1024**2


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def unique(pairs):
    result={}
    for key,value in pairs:
        need(key not in result,"DUPLICATE_JSON_KEY"); result[key]=value
    return result


def loads(raw):
    return json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def regular(path,limit=MAX_FILE,allow_empty=False):
    path=Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents))
         and (0<=path.stat().st_size if allow_empty else 0<path.stat().st_size) and path.stat().st_size<=limit,
         "REGULAR_BOUNDED_FILE")
    return path.read_bytes()


def within(root,name):
    need(type(name) is str and "\\" not in name and "\0" not in name,"SAFE_RELATIVE_PATH")
    rel=PurePosixPath(name)
    need(name==rel.as_posix() and rel.parts and not rel.is_absolute()
         and all(p not in ("",".","..") for p in name.split("/")),"SAFE_RELATIVE_PATH")
    path=Path(root).joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path,*path.parents)),"NO_SYMLINK")
    return path


def put(path,raw):
    path=Path(path)
    need(type(raw) is bytes and len(raw)<=MAX_FILE and path.parent.is_dir()
         and not any(p.is_symlink() for p in (path,*path.parents)),"CREATE_ONLY_FILE")
    with path.open("xb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {"bytes":len(raw),"sha256":sha(raw)}


def once(path,value):
    return put(path,(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())


def new_directory(path):
    path=Path(path).absolute()
    need(path.parent.is_dir() and not path.exists() and not any(p.is_symlink() for p in (path,*path.parents)),"NEW_OUTPUT_DIRECTORY")
    path.mkdir(mode=0o700);return path


def prior_archive(raw):
    expected=RECOVERY["prior_artifact"]
    need(len(raw)==expected["bytes"] and sha(raw)==expected["sha256"],"EXACT_ACTUAL_PRIOR_FINAL_ZIP")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos=archive.infolist(); names=[i.filename for i in infos]
        need(len(names)==len(set(names))<=10000 and sum(i.file_size for i in infos)<=MAX_EXPANDED,"PRIOR_ZIP_BOUNDS")
        for info in infos:
            within(Path(tempfile.gettempdir()),info.filename)
            need(not info.is_dir() and not info.flag_bits&1 and info.file_size<=MAX_FILE
                 and stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG)
                 and "private-model" not in info.filename.split("/"),"SAFE_PRIOR_MEMBER")
        files={name:archive.read(name) for name in names}
    closed=loads(files["FILES.json"])
    rows=closed.get("files")
    need(closed.get("schema")=="english-kokoro-context32-v1/files" and type(rows) is list
         and len({r["path"] for r in rows})==len(rows)
         and {r["path"] for r in rows}|{"FILES.json"}==set(files),"CLOSED_PRIOR_FILES")
    for row in rows:
        raw=files[row["path"]]
        need(len(raw)==row["bytes"] and sha(raw)==row["sha256"],"EXACT_PRIOR_MEMBER_BYTES")
    return files


def metadata(inputs):
    inputs=Path(inputs)
    run=loads(regular(inputs/"prior-run.json"));jobs=loads(regular(inputs/"prior-jobs.json"))
    artifact=loads(regular(inputs/"prior-final.artifact.json")); expected=RECOVERY["prior_artifact"]
    need(run.get("id")==RECOVERY["prior_run_id"] and run.get("head_sha")==MAIN_BEFORE
         and run.get("run_attempt")==1 and run.get("event")=="push" and run.get("head_branch")=="main"
         and run.get("path")==RECOVERY["prior_workflow"] and run.get("status")=="completed"
         and run.get("conclusion")=="failure" and run.get("repository",{}).get("full_name")==RECOVERY["repository"]
         and run["repository"].get("private") is False,"EXACT_TERMINAL_PRIOR_FAILURE")
    need(artifact.get("id")==expected["id"] and artifact.get("name")==expected["name"]
         and artifact.get("size_in_bytes")==expected["bytes"] and artifact.get("digest")=="sha256:"+expected["sha256"]
         and artifact.get("expired") is False and artifact.get("workflow_run",{}).get("id")==RECOVERY["prior_run_id"]
         and artifact["workflow_run"].get("head_sha")==MAIN_BEFORE,"EXACT_PRIOR_ARTIFACT_API")
    actual=jobs.get("jobs")
    need(type(actual) is list and jobs.get("total_count")==len(actual) and len({j["id"] for j in actual})==len(actual),"COMPLETE_PRIOR_JOBS_API")
    for name,conclusion in (("prepare","success"),("observe (phoneme)","failure"),("observe (phoneme_lv60)","failure"),("combine","success")):
        found=[j for j in actual if j.get("name")==name]
        need(len(found)==1 and found[0].get("run_id")==RECOVERY["prior_run_id"] and found[0].get("head_sha")==MAIN_BEFORE
             and found[0].get("status")=="completed" and found[0].get("conclusion")==conclusion,"PRIOR_ACTUAL_JOB: "+name)
    return {name:{"bytes":len(raw),"sha256":sha(raw)} for name in INPUT_FILES if not name.endswith(".zip") for raw in [regular(inputs/name)]}


def immutable_readers(plan):
    helpers=plan.get("helper_sha256")
    need(type(helpers) is dict and len(helpers)==28 and all(helpers.get(k)==v for k,v in PRIOR_CONTEXT_PINS.items()),"EXACT_28_IMMUTABLE_HELPERS")
    for name,expected in helpers.items():
        need(sha(regular(within(ROOT,name),256*1024))==expected,"IMMUTABLE_HELPER_PIN: "+name)
    sys.path.insert(0,str(HERE))
    return importlib.import_module("english_kokoro_context32_once"),importlib.import_module("english_kokoro_context32_compare")


def zero_forward_evidence(files,manifest,comparison):
    result=loads(files["RESULT.json"])
    need(result.get("run")==PRIOR_RUN and result.get("counter_scope")=="complete_terminal_receipts"
         and result.get("declared_inputs")==16 and result.get("declared_observation_slots")==32
         and all(result.get(k)==0 for k in ("actual_forward_attempts","native_forward_returns","completed",
            "forward_attempts_upper_bound","retained_forward_attempts_lower_bound","retained_native_forward_returns_lower_bound",
            "retained_capture_files_lower_bound","new_tts_calls","new_text_asr_calls","new_embedding_calls","new_training_calls","retry_count"))
         and result.get("new_derivative_inputs")==16 and result.get("training_allowed") is False,
         "PRIOR_COMPLETE_ZERO_FORWARD_COUNTERS")
    need(comparison.get("status")=="INCOMPLETE_DIAGNOSTIC_NO_ADMISSION"
         and comparison.get("actual_forward_attempts")==comparison.get("native_forward_returns")==comparison.get("completed_new_captures")==0
         and comparison.get("forward_attempts_upper_bound")==0,"PRIOR_ZERO_FORWARD_COMPARISON")
    need(not any(name.startswith("observers/") and name.rsplit("/",1)[-1] in
        {"FORWARD_STARTED.json","FORWARD_RETURNED.json","CAPTURE.json","logits.f32","frame-ids.json", "normalized-input.f32", "attention-mask.i64"}
        for name in files),"NO_PRIOR_FORWARD_OR_SAVED_PROCESSOR_OUTPUT")
    per_provider={}
    for provider in ("phoneme","phoneme_lv60"):
        prefix="observers/"+provider+"/"
        terminal=loads(files[prefix+"RESULT.json"]);lock=loads(files[prefix+"MODEL_LOCK.json"])
        need(terminal.get("run")==PRIOR_RUN and terminal.get("input_manifest_sha256")==INPUT_MANIFEST_SHA256
             and terminal.get("status")=="INCOMPLETE_DIAGNOSTIC_NO_RETRY"
             and terminal.get("input_started")==terminal.get("failed")==terminal.get("declared_count")==16
             and terminal.get("not_started")==terminal.get("actual_forward_attempts")==terminal.get("native_forward_returns")==terminal.get("completed")==0
             and terminal.get("failure") is None and terminal.get("retry_count")==0,"PRIOR_FAILED_PREPROCESSOR_COUNTS")
        need(lock.get("new_model_constructions")==1 and lock.get("native_forward_attempts_at_initialization")==0
             and lock.get("input_manifest_sha256")==INPUT_MANIFEST_SHA256
             and prefix+"MODEL_LOAD_STARTED.json" in files,"PRIOR_MODEL_CONSTRUCTION_RETAINED")
        ledger=terminal.get("rows")
        need(type(ledger) is list and len(ledger)==16 and [r["id"] for r in ledger]==[r["id"] for r in manifest["rows"]],"PRIOR_16_FAILED_ROWS")
        for item,row in zip(ledger,manifest["rows"]):
            need(item.get("source_id")==row["source_id"] and item.get("status")=="failed" and item.get("input_started") is True
                 and item.get("forward_attempts")==0 and item.get("forward_returned") is False and item.get("capture_saved") is False
                 and item.get("failure")==RECOVERY["prior_error"] and item.get("retry_count")==0,"EXACT_PREPROCESSOR_FAILURE_ROW")
            base=prefix+row["id"]+"/"
            need({n for n in files if n.startswith(base)}=={base+"STARTED.json",base+"FAILED.json"},"EXACT_OLD_INPUT_STARTED_FAILED_ONLY")
            start,failed=loads(files[base+"STARTED.json"]),loads(files[base+"FAILED.json"])
            for doc in (start,failed):
                need(doc.get("run")==PRIOR_RUN and doc.get("id")==row["id"] and doc.get("source_id")==row["source_id"]
                     and doc.get("input_pcm_sha256")==row["pcm_sha256"] and doc.get("input_wav_sha256")==row["wav_sha256"]
                     and doc.get("input_samples")==row["pcm_samples"] and doc.get("forward_attempts")==0
                     and doc.get("forward_returned") is False and doc.get("retry_count")==0,"PRIOR_ZERO_FORWARD_INPUT_BINDING")
            need(failed.get("failure")==RECOVERY["prior_error"] and failed.get("input_started") is True
                 and failed.get("capture_saved") is False,"EXACT_FAILED_RECEIPT")
        per_provider[provider]={"prior_model_constructions":1,"prior_input_started":16,"prior_failed_preprocessing":16,
            "prior_forward_attempts":0,"prior_forward_returns":0,"prior_captures":0,
            "model_lock_sha256":sha(files[prefix+"MODEL_LOCK.json"]),"terminal_sha256":sha(files[prefix+"RESULT.json"])}
    return per_provider


@contextmanager
def verified_prior(inputs,plan):
    """Temporarily expose immutable prepared bytes; never write them to delivery."""
    need(plan.get("recovery")==RECOVERY and plan.get("input_manifest_sha256")==INPUT_MANIFEST_SHA256
         and plan.get("prior_run_identity")==PRIOR_RUN and plan.get("prior_source_plan_sha256")==SOURCE_PLAN_SHA256
         and plan.get("prepared_source_code_sha256")==PREPARED_SOURCE_PIN,"FIXED_RECOVERY_PLAN")
    api=metadata(inputs)
    files=prior_archive(regular(Path(inputs)/"prior-final.zip"))
    need(sha(files["prepared/INPUT_MANIFEST.json"])==INPUT_MANIFEST_SHA256
         and sha(files["prepared/EXECUTION_PLAN.json"])==SOURCE_PLAN_SHA256
         and files["EXECUTION_PLAN.json"]==files["prepared/EXECUTION_PLAN.json"],"ORIGINAL_PREPARED_PLAN_MANIFEST_BYTES")
    source_plan=loads(files["prepared/EXECUTION_PLAN.json"])
    prior_worker,prior_compare=immutable_readers(plan)
    prior_worker.validate_plan(source_plan)
    with tempfile.TemporaryDirectory(prefix="context32-maskfix-prior-") as folder:
        root=Path(folder)
        for name,raw in files.items():
            target=within(root,name);target.parent.mkdir(parents=True,exist_ok=True);put(target,raw)
        prepared=root/"prepared"
        manifest=prior_worker.source_packet_readback(prepared,source_plan)
        need(manifest["run"]==PRIOR_RUN and manifest["rows"]==plan.get("input_roster")
             and sha(canonical(manifest["rows"]))==ROWS_SHA256
             and manifest["old_observed_pcm_sha256"]==plan.get("old_observed_pcm_sha256"),"EXACT_EXISTING_16_FULL_PCM")
        prior_roots={provider:root/"observers"/provider for provider in ("phoneme","phoneme_lv60")}
        comparison=prior_compare.compare(prepared,prior_roots,source_plan)
        need(comparison==loads(files["COMPARISON.json"])
             and sha(files["COMPARISON.json"])==loads(files["RESULT.json"])["comparison_sha256"],"RECOMPUTED_PRIOR_FAILED_DIAGNOSTIC")
        provider_proof=zero_forward_evidence(files,manifest,comparison)
        proof={"schema":SCHEMA+"/prior-zero-forward","recovery":RECOVERY,"prior_run_identity":PRIOR_RUN,
            "input_manifest_sha256":INPUT_MANIFEST_SHA256,"prior_source_plan_sha256":SOURCE_PLAN_SHA256,
            "existing_rows_sha256":ROWS_SHA256,"raw_api_files":api,"prior_member_count":len(files),
            "prior_closed_byte_manifest_verified":True,"prior_model_constructions":2,"prior_processor_attempts":32,
            "prior_input_started":32,"prior_failed_preprocessing":32,"prior_forward_attempts":0,
            "prior_forward_returns":0,"prior_captures":0,"providers":provider_proof,
            "same_existing_pcm_verified":16,"new_derivations_during_verification":0,"new_model_calls_during_verification":0,
            "old_labels_modified":False,"training_allowed":False}
        yield {"prepared_root":prepared,"source_plan":source_plan,"manifest":manifest,"proof":proof}


def package_inputs(plan,inputs,output,run):
    with verified_prior(inputs,plan) as material:
        output=new_directory(output);(output/"inputs").mkdir(mode=0o700)
        for name in INPUT_FILES:put(output/"inputs"/name,regular(Path(inputs)/name))
        once(output/"EXECUTION_PLAN.json",plan)
        once(output/"STARTED.json",{"schema":SCHEMA,"stage":"verify-reuse","run":run,
            "recovery_of_failed_preprocessing":True,"new_derivations":0,"new_model_calls":0,"retry_count":0})
        proof_pin=once(output/"PRIOR_ZERO_FORWARD_PROOF.json",material["proof"])
        once(output/"INPUT_BINDING.json",{"schema":SCHEMA+"/input-binding","run":run,"source_run":PRIOR_RUN,
            "input_manifest_sha256":INPUT_MANIFEST_SHA256,"source_plan_sha256":SOURCE_PLAN_SHA256,
            "source_manifest_member":"prepared/INPUT_MANIFEST.json","prior_archive":"inputs/prior-final.zip",
            "existing_rows_sha256":ROWS_SHA256,"rows":material["manifest"]["rows"]})
    result={"schema":SCHEMA,"stage":"verify-reuse","run":run,"status":"COMPLETE_PRIOR_INPUT_REUSE",
        "declared_inputs":16,"reused_existing_full_inputs":16,"new_derivations":0,"new_model_calls":0,
        "new_tts_calls":0,"retry_count":0,"recovery_of_failed_preprocessing":True,
        "prior_model_constructions":2,"prior_processor_attempts":32,"prior_forward_attempts":0,
        "forward_replays":0,"training_allowed":False,"prior_zero_forward_proof_sha256":proof_pin["sha256"]}
    once(output/"RESULT.json",result)
    return result


@contextmanager
def readback_package(root,plan):
    root=Path(root)
    need(loads(regular(root/"EXECUTION_PLAN.json"))==plan,"NEW_PLAN_PACKET_BINDING")
    result=loads(regular(root/"RESULT.json"));binding=loads(regular(root/"INPUT_BINDING.json"))
    need(result.get("status")=="COMPLETE_PRIOR_INPUT_REUSE" and result.get("declared_inputs")==result.get("reused_existing_full_inputs")==16
         and result.get("new_derivations")==result.get("new_model_calls")==result.get("new_tts_calls")==result.get("retry_count")==result.get("forward_replays")==0
         and result.get("prior_model_constructions")==2 and result.get("prior_processor_attempts")==32
         and result.get("recovery_of_failed_preprocessing") is True and result.get("training_allowed") is False
         and result.get("run")==binding.get("run")==loads(regular(root/"STARTED.json")).get("run"),"REUSE_PACKET_COUNTS")
    with verified_prior(root/"inputs",plan) as material:
        need(loads(regular(root/"PRIOR_ZERO_FORWARD_PROOF.json"))==material["proof"]
             and result.get("prior_zero_forward_proof_sha256")==sha(regular(root/"PRIOR_ZERO_FORWARD_PROOF.json"))
             and binding.get("source_run")==PRIOR_RUN and binding.get("input_manifest_sha256")==INPUT_MANIFEST_SHA256
             and binding.get("source_plan_sha256")==SOURCE_PLAN_SHA256 and binding.get("rows")==material["manifest"]["rows"]
             and binding.get("existing_rows_sha256")==ROWS_SHA256,"ACTUAL_PRIOR_PROOF_AND_INPUT_BINDING")
        yield material
