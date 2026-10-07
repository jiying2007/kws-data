"""Preparation-only exact CPU FP32 adapters. No launcher or automatic execution.

Real loading requires reviewed source/asset/schema/execution contracts and an
truthfully declared offline-configured, process-supervised environment. Tests use fake data;
these runtime paths have not been exercised against real models.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pcm.pcm_binding import validate_pcm_descriptor
from .architecture import (canonical_sha,qwen06_schema,sensevoice_schema,
                           validate_qwen_geometry,validate_sense_geometry)
from .assets import verify_assets,validate_asset_lock,read_small_locked
from .decoding import (CapturingTokenizer,qwen_generation_options,qwen_transcript,
                       sense_transcript,token_ids,bounded_text)
from .runtime_support import (require_runtime_versions,verify_source_members,
    describe_cpu_tensor,describe_storage,make_pcm_array,verify_pcm_array,
    assert_effective_qwen_attention)
from vendor import qwen_loading_gate_v2 as qgate
from vendor.checkpoint_guard.exact_apply import exact_apply,require_verified_receipt
from vendor.checkpoint_guard.checkpoint_contract import SENSEVOICE_METADATA_RECIPE

QWEN = "Qwen/Qwen3-ASR-0.6B"
SENSE = "FunAudioLLM/SenseVoiceSmall"
_MODEL_PROCESS_CLAIMED = False


def claim_fresh_model_process():
    """Exactly one model construction per fresh process, including failed loads."""
    global _MODEL_PROCESS_CLAIMED
    if _MODEL_PROCESS_CLAIMED:
        raise RuntimeError("Each model requires a fresh process; no same-process reload or second model")
    _MODEL_PROCESS_CLAIMED = True


def require_execution_declaration(declaration, expected_sha256, model_id,
                                  asset_sha256,source_sha256,schema_sha256):
    fields = {"schema","armed","model_id","run_id","asset_lock_sha256",
              "source_lock_sha256","expected_schema_sha256","decoder_manifest_sha256",
              "model_dtype","batch_size","network","scope"}
    if type(declaration) is not dict or set(declaration) != fields or canonical_sha(declaration) != expected_sha256:
        raise RuntimeError("Missing or changed predeclared model execution contract")
    expected = {"schema":"asr-calibration-model-execution-v1","armed":True,
        "model_id":model_id,"asset_lock_sha256":asset_sha256,"source_lock_sha256":source_sha256,
        "expected_schema_sha256":schema_sha256,"model_dtype":"float32","batch_size":1,
        "network":"offline_configured_not_kernel_isolated","scope":"single_primary_decode_each_of_16_synthetic_clips"}
    for key,value in expected.items():
        if type(declaration[key]) is not type(value) or declaration[key] != value:
            raise RuntimeError("Execution contract scope/recipe mismatch")
    for key in ("asset_lock_sha256","source_lock_sha256","expected_schema_sha256","decoder_manifest_sha256"):
        value = declaration[key]
        if type(value) is not str or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError("Malformed execution identity")
    run = declaration["run_id"]
    if type(run) is not str or not 0<len(run)<=128 or any(c.isspace() for c in run):
        raise RuntimeError("Malformed run ID")
    # A data declaration does not itself enforce isolation or grant user authority.
    return dict(declaration)


@dataclass
class PreparedRuntime:
    model_id: str
    wrapper: object
    receipt: dict
    declaration: dict
    consumed_ids: set
    decoder_inputs: dict


def validate_decoder_manifest(raw,expected_sha256):
    if type(raw) is not bytes or len(raw)>1024*1024 or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise RuntimeError("Decoder manifest bytes differ from predeclaration")
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result: raise ValueError("Duplicate JSON key")
            result[key]=value
        return result
    manifest = json.loads(raw.decode("utf-8"),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(manifest) is not dict or set(manifest)!={"schema","order","clips"} or manifest["schema"]!="asr-decoder-inputs-v1":
        raise RuntimeError("Unexpected decoder manifest schema")
    if type(manifest["clips"]) is not list or len(manifest["clips"])!=16:
        raise RuntimeError("Decoder manifest must contain exactly16 clips")
    index = {}
    for row in manifest["clips"]:
        if type(row) is not dict or set(row)!={"opaque_id","binding_sha256","descriptor"}:
            raise RuntimeError("Malformed decoder manifest row")
        rid = row["opaque_id"]
        if type(rid) is not str or len(rid)!=9 or not rid.startswith("clip-") or not rid[5:].isdigit() or rid in index:
            raise RuntimeError("Invalid/duplicate opaque input ID")
        validate_pcm_descriptor(row["descriptor"])
        if canonical_sha(row["descriptor"])!=row["binding_sha256"] or row["descriptor"].get("opaque_id")!=rid:
            raise RuntimeError("Decoder binding digest/ID mismatch")
        index[rid] = row
    if manifest["order"]!=list(index) or set(index)!={f"clip-{i:04d}" for i in range(1,17)}:
        raise RuntimeError("Decoder input order or membership differs")
    return index


def _preload(model_id,root,assets,asset_sha256,sources,source_sha256,
             declaration,declaration_sha256,decoder_manifest_bytes):
    schema = qwen06_schema() if model_id==QWEN else sensevoice_schema()
    schema_sha = canonical_sha(schema)
    validated = require_execution_declaration(declaration,declaration_sha256,model_id,
                                              asset_sha256,source_sha256,schema_sha)
    inputs = validate_decoder_manifest(decoder_manifest_bytes,validated["decoder_manifest_sha256"])
    # All filesystem/source/version checks precede importing/loading the ASR model.
    asset_receipt = verify_assets(root,assets,asset_sha256,model_id)
    versions = require_runtime_versions()
    verify_source_members(sources,source_sha256,model_id)
    return schema,schema_sha,validated,asset_receipt,versions,inputs


def load_qwen_after_approval(root,assets,asset_sha256,sources,source_sha256,
                             declaration,declaration_sha256,decoder_manifest_bytes):
    schema,schema_sha,dec,asset_receipt,versions,inputs = _preload(QWEN,root,assets,asset_sha256,
        sources,source_sha256,declaration,declaration_sha256,decoder_manifest_bytes)
    files = validate_asset_lock(assets,asset_sha256,QWEN)
    config_data = json.loads(read_small_locked(root,"config.json",files))
    validate_qwen_geometry(config_data)
    evidence = qgate.inspect_safetensors(Path(root)/"model.safetensors",schema,schema_sha,
        files["model.safetensors"]["sha256"],approved_aliases=[])
    claim_fresh_model_process()
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoConfig,AutoModel,AutoProcessor
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = AutoConfig.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
    # Explicit nested propagation. The top-level hint alone is not sufficient.
    for cfg in (config,config.thinker_config,config.thinker_config.audio_config,config.thinker_config.text_config):
        cfg._attn_implementation = "eager"
        # Nested _from_config calls consult their own dtype. The official
        # thinker BF16 hint otherwise overrides a top-level FP32 argument.
        cfg.dtype = torch.float32
    result = AutoModel.from_pretrained(str(root),config=config,dtype=torch.float32,
        device_map="cpu",attn_implementation="eager",local_files_only=True,
        trust_remote_code=False,output_loading_info=True,ignore_mismatched_sizes=False,
        use_safetensors=True,low_cpu_mem_usage=True)
    if type(result) is not tuple or len(result)!=2: raise RuntimeError("Qwen loader must return model/loading_info")
    model,loading_info = result
    qgate.clear_verification(model)
    qgate.validate_loading_info(loading_info)
    model.eval()
    attention = assert_effective_qwen_attention(model)
    callback = lambda value: describe_cpu_tensor(value,True)
    load_receipt = qgate.verify_loaded_model(model,loading_info,evidence,callback,describe_storage)
    processor = AutoProcessor.from_pretrained(str(root),fix_mistral_regex=True,
        local_files_only=True,trust_remote_code=False)
    wrapper = Qwen3ASRModel(backend="transformers",model=model,processor=processor,
        forced_aligner=None,max_inference_batch_size=1,max_new_tokens=256)
    receipt = {"assets":asset_receipt,"versions":versions,"load":load_receipt,
               "effective_attention":attention,"dtype":"float32","device":"cpu"}
    return PreparedRuntime(QWEN,wrapper,receipt,dec,set(),inputs)


def load_sense_after_approval(root,assets,asset_sha256,sources,source_sha256,
                              declaration,declaration_sha256,decoder_manifest_bytes):
    schema,schema_sha,dec,asset_receipt,versions,inputs = _preload(SENSE,root,assets,asset_sha256,
        sources,source_sha256,declaration,declaration_sha256,decoder_manifest_bytes)
    files = validate_asset_lock(assets,asset_sha256,SENSE)
    claim_fresh_model_process()
    import torch
    import yaml
    from funasr import AutoModel
    from funasr.utils import fbank
    if fbank._HAS_TORCHAUDIO or fbank._knf is None:
        raise RuntimeError("Exact native-fbank backend is required; no backend fallback")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    config = yaml.safe_load(read_small_locked(root,"config.yaml",files))
    validate_sense_geometry(config)
    names = ("model","model_conf","encoder","encoder_conf","tokenizer","tokenizer_conf",
             "frontend","frontend_conf","specaug","specaug_conf")
    kwargs = {key:config[key] for key in names if key in config}
    if config.get("tokenizer")!="SentencepiecesTokenizer" or config.get("frontend")!="WavFrontend":
        raise RuntimeError("Unreviewed SenseVoice components")
    kwargs["tokenizer_conf"] = dict(kwargs["tokenizer_conf"],bpemodel=str(Path(root)/"chn_jpn_yue_eng_ko_spectok.bpe.model"))
    kwargs["frontend_conf"] = dict(kwargs["frontend_conf"],cmvn_file=str(Path(root)/"am.mvn"),
                                   dither=1.0,snip_edges=True,upsacle_samples=True)
    kwargs.update(model_path=str(root),init_param=None,device="cpu",ncpu=1,batch_size=1,seed=0,
        disable_update=True,disable_pbar=True,disable_log=True,trust_remote_code=False,fp16=False,bf16=False)
    wrapper = AutoModel(**kwargs)
    if wrapper.kwargs["tokenizer"].get_vocab_size()!=25055:
        raise RuntimeError("Locked tokenizer vocabulary differs from independent schema")
    checkpoint = torch.load(Path(root)/"model.pt",map_location="cpu",weights_only=True)
    loaded = exact_apply(wrapper.model,checkpoint,describe_cpu_tensor,
                         expected_schema=schema,expected_schema_sha256=schema_sha,
                         metadata_recipe=SENSEVOICE_METADATA_RECIPE)
    del checkpoint
    wrapper.model.eval()
    receipt = {"assets":asset_receipt,"versions":versions,"load":loaded,
        "dtype":"float32","device":"cpu","frontend_backend":"kaldi-native-fbank",
        "dither":1.0,"dither_rng_reproducibility":"not established by torch/numpy seed",
        "no_vad":True,"use_itn":False}
    return PreparedRuntime(SENSE,wrapper,receipt,dec,set(),inputs)


def _begin_clip(runtime,bound,decoder_manifest_sha256):
    if decoder_manifest_sha256 != runtime.declaration["decoder_manifest_sha256"]:
        raise RuntimeError("Decoder manifest identity drift")
    declared = runtime.decoder_inputs.get(bound.opaque_id)
    if (declared is None or declared["descriptor"]!=bound.descriptor or
        declared["binding_sha256"]!=bound.descriptor_sha256 or
        hashlib.sha256(bound.descriptor_json).hexdigest()!=declared["binding_sha256"]):
        raise RuntimeError("Clip identity/PCM binding differs from approved decoder manifest")
    if bound.opaque_id in runtime.consumed_ids or len(runtime.consumed_ids)>=16:
        raise RuntimeError("Duplicate primary attempt or more than16 clips")
    # Mark before the attempt, including exceptions: a retry needs a new run ID.
    runtime.consumed_ids.add(bound.opaque_id)
    return make_pcm_array(bound)


def infer_qwen_once_after_approval(runtime,bound,decoder_manifest_sha256):
    if runtime.model_id!=QWEN: raise RuntimeError("Wrong runtime")
    wrapper = runtime.wrapper
    saved = qgate.require_verified_receipt(wrapper.model)
    # Read the actual current tensors again; a seal alone cannot detect mutation.
    qgate.verify_loaded_model(wrapper.model,saved["loading_info"],saved["checkpoint_evidence"],
                              lambda t:describe_cpu_tensor(t,True),describe_storage)
    assert_effective_qwen_attention(wrapper.model)
    pcm = _begin_clip(runtime,bound,decoder_manifest_sha256)
    import torch
    prompt = wrapper._build_text_prompt(context="",force_language=None)
    inputs = wrapper.processor(text=[prompt],audio=[pcm],sampling_rate=16000,
                               return_tensors="pt",padding=True)
    inputs = inputs.to(wrapper.model.device).to(wrapper.model.dtype)
    # Whisper's downsampled attention mask can retain a strided NumPy/torch
    # view. Canonicalize only processor inputs, preserving exact values; the
    # stricter weight descriptor continues to reject noncontiguous weights.
    input_layout = {key:{"original_stride":list(value.stride()),
                         "contiguous_copy":not value.is_contiguous()}
                    for key,value in inputs.items()}
    inputs = {key:value.contiguous() for key,value in inputs.items()}
    feature_evidence = {key:describe_cpu_tensor(value,True) for key,value in inputs.items()}
    with torch.inference_mode():
        output = wrapper.model.generate(**inputs,**qwen_generation_options())
    sequence = output.sequences
    if sequence.ndim!=2 or sequence.shape[0]!=1: raise RuntimeError("Qwen generated sequence contract mismatch")
    prefix = inputs["input_ids"].shape[1]
    if not torch.equal(sequence[:,:prefix],inputs["input_ids"]):
        raise RuntimeError("Generated sequence did not preserve the declared prompt prefix")
    ids = token_ids(sequence[0,prefix:].tolist(),256)
    decoded = bounded_text(wrapper.processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0])
    full = bounded_text(wrapper.processor.batch_decode([ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)[0])
    row,metadata = qwen_transcript(decoded,ids)
    verify_pcm_array(pcm,bound.descriptor["float32_pcm"])
    return row,{"token_ids":ids,"decoded_with_special_tokens":full,"decoded_skip_special_tokens":decoded,
        "metadata":metadata,"input_binding":bound.descriptor,"prompt_context":"","force_language":None,
        "processor_tensor_evidence":feature_evidence,"processor_input_layout":input_layout,
        "transcript_repair_applied":False}


def infer_sense_once_after_approval(runtime,bound,decoder_manifest_sha256):
    if runtime.model_id!=SENSE: raise RuntimeError("Wrong runtime")
    wrapper = runtime.wrapper
    require_verified_receipt(wrapper.model,describe_cpu_tensor)
    pcm = _begin_clip(runtime,bound,decoder_manifest_sha256)
    capture = CapturingTokenizer(wrapper.kwargs["tokenizer"])
    import torch
    with torch.inference_mode():
        output = wrapper.model.inference(data_in=pcm,key=[bound.opaque_id],tokenizer=capture,
            frontend=wrapper.kwargs["frontend"],device="cpu",language="auto",use_itn=False,
            text_norm="woitn",fs=16000,data_type="sound",output_timestamp=False,ban_emo_unk=False)
    if type(output) is not tuple or len(output)!=2 or type(output[0]) is not list or len(output[0])!=1:
        raise RuntimeError("SenseVoice single-clip result contract mismatch")
    item = output[0][0]
    if type(item) is not dict or item.get("key")!=bound.opaque_id or set(item)!={"key","text"}:
        raise RuntimeError("SenseVoice result ID/fields mismatch")
    decoded = bounded_text(item["text"])
    raw = capture.evidence(decoded)
    row,metadata = sense_transcript(decoded)
    verify_pcm_array(pcm,bound.descriptor["float32_pcm"])
    return row,{"raw_result":output,"token_evidence":raw,"metadata":metadata,
        "input_binding":bound.descriptor,"use_itn":False,"rich_postprocess_applied":False,
        "completeness_scope":"Returned full-input CTC inference; not acoustic transcript completeness"}


if __name__ == "__main__":
    raise SystemExit("Preparation module only. No command-line model execution or download route exists.")
