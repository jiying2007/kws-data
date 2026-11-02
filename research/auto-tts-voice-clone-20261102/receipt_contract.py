"""Shared adapter and native-receipt identity checks; never invokes a model.

Old receipts without an exact adapter/plan binding are not upgraded or retried.
These checks establish consistency of retained bytes, not model authenticity.
"""
from __future__ import annotations
from pathlib import Path
import campaign as c
import runner

LOCK_SCHEMA="offline-kws-native-asr-adapters-v1"
ALLOWED_TOKENS={"{audio}","{output}"}
MODEL_IDS={"qwen06":"Qwen/Qwen3-ASR-0.6B",
           "sensevoice":"FunAudioLLM/SenseVoiceSmall",
           "whisper":"openai/whisper-small"}

def check_adapters(value):
    c.require(type(value) is dict and set(value)=={"schema","adapters"} and
              value["schema"]==LOCK_SCHEMA,"ASR_ADAPTER_LOCK_SCHEMA")
    entries=value["adapters"]
    c.require(type(entries) is list and len(entries)==3,"ALL_THREE_ASR_REQUIRED")
    found={}
    for row in entries:
        c.require(type(row) is dict and
                  set(row)=={"engine","model_id","model_revision",
                    "model_assets_sha256","executable","executable_sha256",
                    "argv","timeout_seconds"},"ASR_ADAPTER_FIELDS")
        engine=row["engine"]
        c.require(engine in c.ENGINES and engine not in found and
                  row["model_id"]==MODEL_IDS[engine] and
                  type(row["model_revision"]) is str and
                  c._REV.fullmatch(row["model_revision"]),"ASR_ENGINE_ID")
        for field in ("model_assets_sha256","executable_sha256"):
            c.require(type(row[field]) is str and c._SHA.fullmatch(row[field]),
                      "ASR_PIN_REQUIRED")
        path=Path(row["executable"])
        c.require(path.is_absolute(),"ASR_EXECUTABLE_ABSOLUTE")
        argc=row["argv"]
        c.require(type(argc) is list and 2<=len(argc)<=20 and
                  all(type(a) is str and 1<=len(a)<=128 for a in argc)
                  and set(ALLOWED_TOKENS)<=set(argc),"OPAQUE_AUDIO_AND_OUTPUT_REQUIRED")
        c.require(all("{" not in a and "}" not in a for a in argc if a not in ALLOWED_TOKENS),
                  "UNSUPPORTED_ASR_TEMPLATE")
        # No extra flags that can accidentally insert an expected keyword.
        for arg in argc:
            if arg not in ALLOWED_TOKENS:
                c.require(all(w not in arg for w in c.TARGETS),"TARGET_LEAKAGE_IN_ASR_CMD")
        c.require(type(row["timeout_seconds"]) is int and
                  5<=row["timeout_seconds"]<=300,"BOUNDED_ASR_TIMEOUT")
        found[engine]=row
    c.require(set(found)==set(c.ENGINES),"MISSING_ASR_ENGINE")
    return found


def binding(plan, adapters):
    check_adapters(adapters)
    return {"plan_sha256":runner._hash(c.canonical(plan)),
            "adapter_lock_sha256":runner._hash(c.canonical(adapters))}


def require_binding(record, expected):
    c.require(type(record) is dict and
              all(record.get(key)==value for key,value in expected.items()),
              "ASR_PLAN_OR_ADAPTER_LOCK_DRIFT")


def verify_root(root, expected, schema):
    root=Path(root)
    if not root.exists():
        return False  # No receipts exist; callers may only report incomplete.
    runner._safe_file(root/"plan-lock.json",limit=65536)
    lock=c.read_json(root/"plan-lock.json",65536)
    require_binding(lock,expected)
    c.require(lock.get("schema")==schema and
              lock.get("training_admitted") is False,"ASR_ROOT_LOCK_INVALID")
    return True


def verify_native(folder, observation, model, expected, *,
                  started_schema, completed_schema=None):
    folder=Path(folder)
    runner._safe_file(folder/"started.json",limit=65536)
    runner._safe_file(folder/"completed.json",limit=65536)
    started=c.read_json(folder/"started.json",65536)
    completed=c.read_json(folder/"completed.json",65536)
    require_binding(started,expected)
    require_binding(completed,expected)
    c.require(started.get("schema")==started_schema and
              started.get("engine")==observation["engine"] and
              started.get("model_id")==model["model_id"] and
              started.get("model_revision")==model["model_revision"] and
              started.get("model_assets_sha256")==model["model_assets_sha256"] and
              type(started.get("attempts")) is int and
              started["attempts"]==1 and
              started.get("training_admitted") is False,
              "ASR_STARTED_IDENTITY_DRIFT")
    c.require(all(started.get(key)==observation[key] for key in
              ("audio_id","wav_sha256","pcm_sha256","audio_frames")) and
              type(started.get("audio_frames")) is int,
              "ASR_STARTED_AUDIO_DRIFT")
    c.require(completed.get("training_admitted") is False and
              completed.get("observation_sha256")==
              runner._hash(c.canonical(observation)),"ASR_COMPLETION_DRIFT")
    if completed_schema is not None:
        c.require(completed.get("schema")==completed_schema,"ASR_COMPLETION_SCHEMA")
    else:
        c.require(completed.get("status")=="NATIVE_ASR_COMPLETE_UNADMITTED",
                  "ASR_COMPLETION_STATUS")
    size,digest=runner._safe_file(folder/"native.json",limit=8*1024*1024)
    c.require(digest==observation["raw_native_sha256"]==
              completed.get("native_sha256") and
              type(completed.get("native_bytes")) is int and
              size==completed["native_bytes"],"ASR_NATIVE_BYTES_DRIFT")
    native=c.read_json(folder/"native.json",8*1024*1024)
    fields={"actual_text","decoder_completed","processed_start_frame",
            "processed_end_frame","audio_id","model_id","model_revision","engine"}
    c.require(type(native) is dict and set(native)==fields and
              all(native[key]==observation[key] for key in fields),
              "ASR_NATIVE_OBSERVATION_DRIFT")
    c.require(native["model_id"]==model["model_id"] and
              native["model_revision"]==model["model_revision"] and
              native["decoder_completed"] is True and
              type(native["processed_start_frame"]) is int and
              native["processed_start_frame"]==0 and
              type(native["processed_end_frame"]) is int and
              native["processed_end_frame"]==observation["audio_frames"],
              "ASR_NATIVE_FULL_FRAME_IDENTITY")
    return native
