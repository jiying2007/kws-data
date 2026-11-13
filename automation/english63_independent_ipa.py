"""One-shot independent IPA/phone observations on the 63 saved English WAVs.

The speech model sees only verified PCM. It never receives TTS target texts,
wake labels, ASR hypotheses, or near-miss metadata. Phoneme outputs are a
fourth *machine* observation, NOT gold labels or training admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import wave

# Python -I -S omits the script directory: only checked-out sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_acoustic_contrast as acoustic
import english_seven_voice_review as review
import english_seven_voice_asr_once as seven
import multilingual_campaign as market

POLICY = "english63-independent-ipa-ctc-machine-review-v1"
TRIGGER = (Path(__file__).resolve().parents[1] /
           ".github/triggers/english63-independent-ipa-20261113.json")
MODEL_ID = "bookbot/wav2vec2-ljspeech-gruut"
MODEL_REV = "7be1e6f27c8163ce790688d62fd9ce1731177c2e"
WEIGHTS_SHA256 = "9bc8cd20e1348fc384ba4b9cd60a5c761dbb3f0b7071d1bdc33331ed1b358f29"
MODEL_FILES = (
    "model.safetensors", "config.json", "preprocessor_config.json",
    "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json",
)
CONTRASTS = acoustic.CONTRASTS
MAX_FORWARDS = 63


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def policy() -> dict:
    frozen = {
        "schema": POLICY, "source_tts_run_id": seven.SOURCE_RUN_ID,
        "source_tts_artifact_id": review.ARCHIVES["tts"][0],
        "source_tts_zip_sha256": review.ARCHIVES["tts"][1],
        "source_review_run_id": acoustic.REVIEW_RUN_ID,
        "source_review_artifact_id": acoustic.REVIEW_ARTIFACT_ID,
        "source_review_zip_sha256": acoustic.REVIEW_ZIP_SHA256,
        "source_review_json_sha256": acoustic.REVIEW_JSON_SHA256,
        "source_pcm_count": 63, "source_three_asr_observations": 189,
        "model_id": MODEL_ID, "model_revision": MODEL_REV,
        "model_safetensors_sha256": WEIGHTS_SHA256,
        "model_files": list(MODEL_FILES),
        "model_license_review": "apache-2.0-ljspeech-single-narrator-weak-observation",
        "max_new_phoneme_forwards": MAX_FORWARDS,
        "new_tts_calls": 0, "new_original_three_asr_calls": 0,
        "new_kws_model_forwards": 0, "new_optimizer_steps": 0,
        "human_gold": False, "training_admitted": False,
        "no_scheduled_tasks": True,
    }
    need(market.read(TRIGGER) == frozen, "IPA_TRIGGER_DRIFT")
    return frozen


def protected_main() -> None:
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_EVENT_NAME") == "push" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "IPA_PROTECTED_MAIN_ONCE_REQUIRED")


def manifest(model_path: Path) -> dict:
    need(model_path.is_dir() and not model_path.is_symlink(),
         "IPA_MODEL_FOLDER_UNSAFE")
    observed = {}
    for name in MODEL_FILES:
        path = model_path / name
        need(path.is_file() and not path.is_symlink(), "IPA_MODEL_FILE_MISSING")
        observed[name] = {"bytes":path.stat().st_size, "sha256":digest(path)}
    need(observed["model.safetensors"]["sha256"] == WEIGHTS_SHA256 and
         300_000_000 <= observed["model.safetensors"]["bytes"] <= 450_000_000,
         "IPA_PINNED_WEIGHTS_MISMATCH")
    need(all(0 < observed[name]["bytes"] <= 131_072
             for name in MODEL_FILES if name != "model.safetensors"),
         "IPA_TOKENIZER_METADATA_BUDGET")
    actual = {p.name for p in model_path.iterdir() if p.is_file()}
    need(actual == set(MODEL_FILES), "IPA_EXTRA_OR_MISSING_MODEL_ASSET")
    meta = {"model_id":MODEL_ID,"model_revision":MODEL_REV,
            "files":observed}
    return {"meta":meta, "manifest_sha256":hashlib.sha256(
        market.canonical(meta)).hexdigest()}


def load_model_from_pinned_assets(target: Path):
    """Network allowed ONLY for fixed revision/asset preflight; then offline."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REV,
                      allow_patterns=list(MODEL_FILES), local_dir=str(target),
                      max_workers=2)
    asset = manifest(target)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCTC, AutoProcessor
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    processor = AutoProcessor.from_pretrained(
        str(target), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCTC.from_pretrained(
        str(target), local_files_only=True, trust_remote_code=False,
        use_safetensors=True).to("cpu").eval()
    need(model.config.model_type == "wav2vec2" and
         processor.feature_extractor.sampling_rate == 16000,
         "IPA_MODEL_FRONTEND_GEOMETRY")
    return processor,model,torch,asset


def phone_tokens(predicted_ids: list[int], tokenizer) -> tuple[list[str],list[str]]:
    """Strict CTC collapse, preserve raw IPA and stressless comparison tokens."""
    need(type(predicted_ids) is list and 2 <= len(predicted_ids) <= 2000 and
         all(type(n) is int and n >= 0 for n in predicted_ids),
         "IPA_PREDICTED_IDS_INVALID")
    skip = set(tokenizer.all_special_ids)
    delimiter = tokenizer.word_delimiter_token_id
    if delimiter is not None:
        skip.add(delimiter)
    collapsed = [n for n,_ in groupby(predicted_ids) if n not in skip]
    raw = tokenizer.convert_ids_to_tokens(collapsed)
    need(type(raw) is list and len(raw) == len(collapsed) and
         all(type(t) is str and 0 < len(t) <= 20 and
             not re.search(r"[\x00-\x1f]",t) for t in raw),
         "IPA_TOKEN_SHAPE")
    phones = [t.replace("ˈ","").replace("ˌ","") for t in raw]
    phones = [t for t in phones if t]
    need(2 <= len(phones) <= 300, "IPA_NO_USABLE_PHONEMES")
    return raw,phones


def predict_phone(wav: Path, processor, model, torch) -> tuple[list[str],list[str],int]:
    """Audio-only input. No requested text, transcript, source role, or hint."""
    import numpy as np
    with wave.open(str(wav),"rb") as f:
        n = f.getnframes()
        need(f.getframerate() == 16000 and f.getnchannels() == 1 and
             f.getsampwidth() == 2 and f.getcomptype() == "NONE" and
             400 <= n <= 192_000, "IPA_WAV_GEOMETRY")
        pcm = f.readframes(n)
        need(len(pcm) == 2*n and not f.readframes(1), "IPA_PCM_INCOMPLETE")
    x = np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
    data = processor(x,sampling_rate=16000,return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_values=data.input_values).logits
        ids = torch.argmax(logits,dim=-1).squeeze(0).tolist()
    raw,phones = phone_tokens(ids,processor.tokenizer)
    return raw,phones,n


def edit_distance(a: list[str], b: list[str]) -> int:
    """Exact Levenshtein on phoneme-token sequences; no language-model bias."""
    need(type(a) is list and type(b) is list and
         1 <= len(a) <= 300 and 1 <= len(b) <= 300,
         "IPA_EDIT_INPUT_BUDGET")
    prev = list(range(len(b)+1))
    for i,x in enumerate(a,start=1):
        curr = [i]
        for j,y in enumerate(b,start=1):
            curr.append(min(prev[j]+1,curr[-1]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]


def compare_rows(rows: list[dict]) -> dict:
    need(len(rows) == 63 and len({r["cell_id"] for r in rows}) == 63,
         "IPA_ROW_ROSTER")
    by_voice = {}
    for r in rows:
        by_voice.setdefault(r["voice_id"], {})[r["text_requested"]] = r
    need(set(by_voice) == set(acoustic.VOICES),
         "IPA_VOICE_ROSTER")
    reports = []
    summary = {name:Counter() for name,_,_,_ in CONTRASTS}
    for voice,role in acoustic.source_tts.VOICES:
        cell = by_voice[voice]
        need(set(cell) == set(acoustic.PROMPTS) and
             all(r["role"] == role for r in cell.values()),
             "IPA_PROMPT_OR_SPLIT_LEAK")
        contrasts={}
        for name,plain,punctuation,nearphone in CONTRASTS:
            a=cell[plain]["stressless_phones"]
            b=cell[punctuation]["stressless_phones"]
            c=cell[nearphone]["stressless_phones"]
            within=edit_distance(a,b)
            near=min(edit_distance(a,c),edit_distance(b,c))
            flag=near <= within
            contrasts[name]={
                "same_wake_variant_phone_edit":within,
                "nearest_nearphone_phone_edit":near,
                "nearphone_not_farther_than_wake_variant":flag,
                "diagnostic_only":True, "trusted_negative":False,
                "training_admitted":False}
            summary[name]["families"]+=1
            summary[name]["nearphone_not_farther_than_wake_variant"]+=int(flag)
        reports.append({
            "voice_id":voice,"source_family_id":"qwen3-preset-"+voice,
            "role":role,"contrasts":contrasts,"independent_human":False,
            "training_admitted":False})
    return {
        "by_contrast":{k:dict(v) for k,v in summary.items()},
        "by_voice":reports, "independent_human_count":0}


def write_once(path: Path, data: dict) -> None:
    need(not path.exists(),"IPA_OUTPUT_ALREADY_EXISTS")
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,ensure_ascii=False,indent=2)
        f.write("\n")
    path.chmod(0o600)


def execute(tts_zip: Path, review_zip: Path, out: Path) -> dict:
    policy()
    protected_main()
    need(out.is_absolute() and out.parent.is_dir() and not out.exists() and
         not out.parent.is_symlink(), "IPA_CREATE_ONLY_OUTPUT")
    out.mkdir(mode=0o700)
    done=[]
    started={"schema":POLICY,"source_run_id":seven.SOURCE_RUN_ID,
             "model_id":MODEL_ID,"model_revision":MODEL_REV,
             "attempt_budget":MAX_FORWARDS,"attempted":0,
             "training_admitted":False,"human_gold":False}
    write_once(out/"started.json",started)
    try:
        with tempfile.TemporaryDirectory(prefix="english63-ipa-") as td:
            root=Path(td)
            plan,cells,ledger=acoustic.load_exact_sources(
                tts_zip,review_zip,root)
            need(plan["wake_phrases_requested"] == ["Hey Vosen","Hey Mero"] and
                 len(cells)==63 and
                 ledger["counts"]["three_way_completed_calls"]==189,
                 "IPA_SOURCE_ROSTER")
            model_root=root/"model"
            model_root.mkdir(mode=0o700)
            processor,model,torch,asset=load_model_from_pinned_assets(model_root)
            by_cell={r["cell_id"]:r for r in ledger["rows"]}
            for cell,wav,pcm in cells:
                need(len(done) < MAX_FORWARDS,"IPA_FORWARD_BUDGET")
                # The recognizer receives WAV only, not target/label/ASR/role.
                raw,phonemes,frames=predict_phone(wav,processor,model,torch)
                need(frames==by_cell[cell["cell_id"]].get("audio_frames",frames),
                     "IPA_FRAME_BINDING")
                done.append({
                    "cell_id":cell["cell_id"],"voice_id":cell["voice_id"],
                    "source_family_id":cell["source_family_id"],
                    "role":cell["role"], "requested_intent":cell["intent"],
                    "text_requested":cell["text_requested"],
                    "wav_sha256":pcm["wav_sha256"],
                    "raw_ipa_tokens":raw,"stressless_phones":phonemes,
                    "phoneme_model_id":MODEL_ID,
                    "phoneme_model_revision":MODEL_REV,
                    "phonetic_label_authenticated":False,
                    "trusted_negative":False, "human_gold":False,
                    "training_admitted":False,
                    "source_exposure":"synthetic-observed-development",
                })
            report={
                "schema":POLICY,"status":"INDEPENDENT_MACHINE_PHONES_OBSERVED_NOT_ADMITTED",
                "tts_artifact_id":review.ARCHIVES["tts"][0],
                "source_tts_zip_sha256":review.ARCHIVES["tts"][1],
                "review_artifact_id":acoustic.REVIEW_ARTIFACT_ID,
                "review_zip_sha256":acoustic.REVIEW_ZIP_SHA256,
                "model_id":MODEL_ID,"model_revision":MODEL_REV,
                "model_asset_manifest_sha256":asset["manifest_sha256"],
                "model_weights_sha256":WEIGHTS_SHA256,
                "model_assets":asset["meta"]["files"],
                "new_real_phoneme_forwards":len(done),
                "reused_prior_three_asr_calls":189,
                "new_tts_calls":0,"new_prior_three_asr_calls":0,
                "new_kws_model_forwards":0,"new_optimizer_steps":0,
                "independent_pronunciation_gold":"NOT_ESTABLISHED",
                "machine_training_admitted":False,
                "human_gold":False,"training_admitted":False,
                "physical_ssc305":"NOT_RUN",
                "limitations":["single-narrator phoneme model transfer bias",
                    "target text hidden from recognizer, not a gold transcript",
                    "synthetic preset voices are not independent humans",
                    "nearphone similarities are not KWS false alarms"],
                "rows":done,**compare_rows(done),
            }
            write_once(out/"summary.json",report)
            return report
    except BaseException as exc:
        write_once(out/"failed.json",{
            "schema":POLICY,"status":"FAILED_OR_INTERRUPTED_NO_IMPLICIT_RETRY",
            "error_type":type(exc).__name__,
            "new_real_phoneme_forwards_completed":len(done),
            "training_admitted":False,"human_gold":False})
        raise


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-zip",type=Path)
    parser.add_argument("--review-zip",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        if not args.execute:
            p=policy()
            print(json.dumps({"schema":POLICY,
                "maximum_new_phoneme_forwards":p["max_new_phoneme_forwards"],
                "executed":0,"training_admitted":False}))
            return 0
        need(all(v is not None for v in (args.tts_zip,args.review_zip,args.output)),
             "IPA_EXPLICIT_SOURCES_AND_OUTPUT_REQUIRED")
        result=execute(args.tts_zip,args.review_zip,args.output)
        print(json.dumps({"status":result["status"],
                          "new_real_phoneme_forwards":result["new_real_phoneme_forwards"],
                          "by_contrast":result["by_contrast"],
                          "training_admitted":False},sort_keys=True))
        return 0
    except (ValueError,OSError,wave.Error) as exc:
        print("ENGLISH_IPA_STOP: "+type(exc).__name__+": "+str(exc),
              file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
