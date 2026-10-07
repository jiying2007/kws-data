"""Stdlib translation into PR458's externally declared evidence envelopes.

Scientific labels stay outside decoder execution and remain unchanged. This
module does no inference and never fills in missing outcomes from intended text.
"""
import json
import hashlib
import math
import re
from .assets import MODELS
from .adapters import validate_decoder_manifest
from .decoding import bounded_text

HEX40_64 = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
STATES = {"positive","negative","unknown"}
FLAGS = {"missing_characters","incomplete","ambiguous","non_speech","decoding_warning"}
OPAQUE_ID = re.compile(r"clip-[0-9]{4}\Z")


def _strict_json(raw):
    def unique(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError("Duplicate JSON key")
            obj[key]=value
        return obj
    def finite(value):
        number=float(value)
        if not math.isfinite(number):raise ValueError("Nonfinite JSON")
        return number
    return json.loads(raw.decode('utf-8'),object_pairs_hook=unique,parse_float=finite,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _identifier(value,name):
    if type(value) is not str or not 0<len(value)<=128 or any(c.isspace() for c in value):
        raise ValueError("Invalid "+name)
    bounded_text(value)


def scientific_manifest(human_projection,models,probe_set_id):
    if type(human_projection) is not dict or human_projection.get("schema")!="asr-independent-human-labels-v1":
        raise ValueError("Expected independently retained human projection")
    if human_projection.get("target_order")!=["你好小窝","小窝小窝"]:
        raise ValueError("Target order drift")
    _identifier(probe_set_id,"probe set identity")
    if type(models) is not list or len(models)!=2: raise ValueError("Exactly two declared models required")
    for row,model_id in zip(models,("Qwen/Qwen3-ASR-0.6B","FunAudioLLM/SenseVoiceSmall")):
        if type(row) is not dict or set(row)!={"model_id","revision","run_id"}:
            raise ValueError("Malformed model declaration")
        if row["model_id"]!=model_id or row["revision"]!=MODELS[model_id][0] or not HEX40_64.fullmatch(row["revision"]):
            raise ValueError("Model/revision drift")
        _identifier(row["run_id"],"run identity")
    rows=human_projection.get("records")
    if type(rows) is not list or len(rows)!=16: raise ValueError("Exactly16 human rows required")
    records=[];seen=set();opaque=set();counts={state:0 for state in STATES}
    for row in rows:
        if type(row) is not dict or not {"recording_id","opaque_id","wav_sha256","human_target_presence"}<=set(row):
            raise ValueError("Malformed human projection row")
        rid=row["recording_id"];oid=row["opaque_id"]
        _identifier(rid,"recording identity")
        if type(oid) is not str or not OPAQUE_ID.fullmatch(oid):raise ValueError("Invalid opaque identity")
        if rid in seen or oid in opaque: raise ValueError("Duplicate source identity")
        seen.add(rid);opaque.add(oid)
        if type(row["wav_sha256"]) is not str or not SHA.fullmatch(row["wav_sha256"]): raise ValueError("Invalid WAV identity")
        labels=row["human_target_presence"]
        if type(labels) is not list or len(labels)!=2 or any(type(s) is not str or s not in STATES for s in labels):
            raise ValueError("Invalid independent human labels")
        for state in labels:counts[state]+=1
        records.append({"recording_id":rid,"wav_sha256":row["wav_sha256"],"human_target_presence":list(labels)})
    if counts!={"positive":9,"negative":17,"unknown":6}: raise ValueError("Human known/unknown counts changed")
    return {"schema_version":"kws-asr-evidence-manifest-v1","probe_set_id":probe_set_id,
        "execution_kind":"declared_model_output","target_order":list(human_projection["target_order"]),
        "models":[dict(m) for m in models],"records":records}


def output_envelope(outcomes,human_projection,manifest,manifest_sha256,rules_sha256,slot,*,
                    manifest_bytes,execution_receipts,decoder_manifest_bytes,decoder_manifest_sha256):
    """Every exact clip must have an explicit outcome, including failures/not-run.

    SHA arguments must be frozen actual manifest/rules file SHA values supplied
    by the outer run plan, never recomputed here after seeing decoder outputs.
    Contract and decoder-evidence hashes are retained references only: the outer
    bundle validator must verify their bytes/relationships. Consistent caller
    declarations do not authenticate actual inference.
    """
    if type(slot) is not int or slot not in (0,1):raise ValueError("Invalid model slot")
    if any(type(s) is not str or not SHA.fullmatch(s) for s in (manifest_sha256,rules_sha256)):
        raise ValueError("Predeclared evidence hashes required")
    if (type(manifest_bytes) is not bytes or len(manifest_bytes)>512*1024 or
        hashlib.sha256(manifest_bytes).hexdigest()!=manifest_sha256 or
        _strict_json(manifest_bytes)!=manifest):
        raise ValueError("Scientific manifest bytes differ from predeclaration")
    manifest_fields={"schema_version","probe_set_id","execution_kind","target_order","models","records"}
    if type(manifest) is not dict or set(manifest)!=manifest_fields:
        raise ValueError("Unexpected scientific manifest schema")
    validated=scientific_manifest(human_projection,manifest["models"],manifest["probe_set_id"])
    if manifest!=validated:raise ValueError("Scientific manifest differs from validated human projection")
    if type(outcomes) is not list or len(outcomes)!=16:raise ValueError("Exactly16 explicit model outcomes required")
    if type(execution_receipts) is not dict or len(execution_receipts)>16:
        raise ValueError("Bounded raw execution receipt sidecars required")
    decoder_inputs=validate_decoder_manifest(decoder_manifest_bytes,decoder_manifest_sha256)
    crosswalk={r["opaque_id"]:r for r in human_projection["records"]}
    if set(crosswalk)!=set(decoder_inputs):raise ValueError("Human/decoder input membership differs")
    for oid,human in crosswalk.items():
        if human["wav_sha256"]!=decoder_inputs[oid]["descriptor"]["wav"]["sha256"]:
            raise ValueError("Human/decoder WAV association differs")
    expected={r["recording_id"]:r for r in manifest["records"]}
    records=[];seen=set();used_receipts=set();bindings=[]
    for outcome in outcomes:
        required={"opaque_id","wav_sha256","status","raw_text","completeness","quality_flags","execution_receipt_sha256"}
        if type(outcome) is not dict or set(outcome)!=required:raise ValueError("Malformed model outcome")
        oid=outcome["opaque_id"]
        if type(oid) is not str or oid not in crosswalk or oid in seen:raise ValueError("Unknown/duplicate model outcome")
        seen.add(oid);human=crosswalk[oid];reference=expected.get(human["recording_id"])
        if reference is None or outcome["wav_sha256"]!=human["wav_sha256"] or reference["wav_sha256"]!=outcome["wav_sha256"]:
            raise ValueError("Outcome WAV/recording association mismatch")
        if reference["human_target_presence"]!=human["human_target_presence"]:raise ValueError("Human label drift")
        status=outcome["status"];raw=outcome["raw_text"];complete=outcome["completeness"];flags=outcome["quality_flags"]
        if (type(status) is not str or status not in {"success","error","timeout","not_run"} or
            type(complete) is not str or complete not in {"complete","incomplete","unknown"}):
            raise ValueError("Invalid execution state")
        if type(flags) is not list or any(type(f) is not str or f not in FLAGS for f in flags) or len(set(flags))!=len(flags):
            raise ValueError("Invalid quality flags")
        if raw is not None and (type(raw) is not str or len(raw)>1024):raise ValueError("Invalid/oversized transcript")
        if raw is not None:bounded_text(raw)
        if status=="success" and type(raw) is not str:raise ValueError("Success requires raw transcript")
        if status=="not_run":
            if raw is not None or complete!="unknown" or flags or outcome["execution_receipt_sha256"] is not None:
                raise ValueError("Not-run outcome contains invented execution evidence")
        elif type(outcome["execution_receipt_sha256"]) is not str or not SHA.fullmatch(outcome["execution_receipt_sha256"]):
            raise ValueError("Attempted decode requires its external receipt hash")
        else:
            receipt_sha=outcome["execution_receipt_sha256"]
            raw_receipt=execution_receipts.get(receipt_sha)
            if (type(raw_receipt) is not bytes or len(raw_receipt)>1024*1024 or
                hashlib.sha256(raw_receipt).hexdigest()!=receipt_sha):
                raise ValueError("Missing or changed raw execution receipt")
            receipt=_strict_json(raw_receipt)
            required_receipt={'schema','model','opaque_id','wav_sha256','input_binding_sha256',
                'decoder_manifest_sha256','status','raw_text','completeness','quality_flags',
                'attempt_started','model_forward_completed','decoder_evidence_sha256','execution_contract_sha256'}
            if type(receipt) is not dict or set(receipt)!=required_receipt or receipt['schema']!='asr-clip-execution-receipt-v1':
                raise ValueError("Unexpected raw execution receipt schema")
            if receipt['model']!=manifest['models'][slot]:raise ValueError("Receipt model/revision/run differs from declared slot")
            for field in ('opaque_id','wav_sha256','status','raw_text','completeness','quality_flags'):
                if receipt[field]!=outcome[field]:raise ValueError("Receipt outcome association differs")
            expected_binding=decoder_inputs[oid]
            if (receipt['input_binding_sha256']!=expected_binding['binding_sha256'] or
                receipt['wav_sha256']!=expected_binding['descriptor']['wav']['sha256'] or
                receipt['decoder_manifest_sha256']!=decoder_manifest_sha256):
                raise ValueError("Receipt PCM/input manifest binding differs")
            if receipt['attempt_started'] is not True or type(receipt['model_forward_completed']) is not bool:
                raise ValueError("Receipt execution-state evidence malformed")
            if status=='success' and receipt['model_forward_completed'] is not True:
                raise ValueError("Success lacks model forward completion evidence")
            for field in ('decoder_evidence_sha256','execution_contract_sha256'):
                if type(receipt[field]) is not str or not SHA.fullmatch(receipt[field]):raise ValueError("Missing decoder/contract sidecar identity")
            used_receipts.add(receipt_sha)
            bindings.append({'recording_id':human['recording_id'],'opaque_id':oid,
                'wav_sha256':outcome['wav_sha256'],'execution_receipt_sha256':receipt_sha,
                'model':dict(receipt['model']),'input_binding_sha256':receipt['input_binding_sha256'],
                'decoder_evidence_sha256':receipt['decoder_evidence_sha256'],
                'execution_contract_sha256':receipt['execution_contract_sha256']})
        records.append({"recording_id":human["recording_id"],"wav_sha256":outcome["wav_sha256"],
            "status":status,"raw_text":raw,"completeness":complete,"quality_flags":list(flags)})
    if seen!=set(crosswalk):raise ValueError("Missing model outcome")
    if used_receipts!=set(execution_receipts):raise ValueError("Unexpected or unused raw receipt sidecars")
    if all(r["status"]=="not_run" for r in records):raise ValueError("All-not-run cannot claim declared execution")
    envelope={"schema_version":"kws-asr-evidence-input-v1","execution_kind":"declared_model_output",
        "manifest_sha256":manifest_sha256,"rules_sha256":rules_sha256,"model":dict(manifest["models"][slot]),
        "records":records}
    sidecar={'schema':'asr-model-receipt-bindings-v1','model':dict(manifest['models'][slot]),
        'manifest_sha256':manifest_sha256,'rules_sha256':rules_sha256,
        'decoder_manifest_sha256':decoder_manifest_sha256,'records':bindings,
        'scope':'Validated caller-produced execution declarations, not inference authenticity; raw receipts, '
                'execution contracts and decoder sidecars must be retained and independently verified'}
    return envelope,sidecar
