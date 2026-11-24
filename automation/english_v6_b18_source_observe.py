"""Future B18 five-observer capture with raw-first, fail-closed return evidence.

Source admission remains the byte-frozen existing five-model policy. Public
execution needs a new B18 registration and first-push predecessor; import and
source_check make no calls, and no B17 source is an executable input here.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b18_asr_adapter as raw_asr

need, sha, record = raw_asr.need, raw_asr.sha, raw_asr.record
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
SOURCE_SCHEMA = "english-v6-b18-frozen-independent-source-v1"
BATCH = "english-v6-b18-96-original-blind-source-screen-v1"
COUNT = 96
OLD_FILES = {
    "english_v6_primary_source_qa.py": "ff1a19d5d821f778474523474ae4f494b9548b5ad97d9f33d9bb545db6468e94",
    "english_fresh24_ctc_admission.py": "204f6004d3b33bfdcf66c011c2eec34ff947ea5fbb2607b1b265e335a7366b4c",
    "english_v6_b11_supplement_observe.py": "83276ce5c208c3277ac9462d22fab024dd458248dd4279d8b49fdedd3d88e529",
    "english_v6_b11_supplement_screen.py": "7be739af313f69f926a84f3e0b37eb35a46618bb877e060ff1a2a3020c7514d9",
    "english_hifi_micro.py": "0251de2cddc112dc3c84c53cda46ed1a2a1d726d0a7653b9071872f50471fdcc",
    "english_hifi_phones_once.py": "920456b516dffbe878e389320065ced3af56ebc7d1e9494b14bf1994de39f8fd",
    "english_pocket12_observe.py": "07b93e626b5e822b09869c0f06d2179140d938c1c795f97c2d286e112b2c0426",
    "english_seven_voice_asr_once.py": "17771ca272204771982dbdc8e959f7fd75838b6598e84f09ac5819d8e3c79184",
    "english_parakeet_real_once.py": "366cd278f570a37ce68dd20e47e2bee1422daaa109c693ac42ab4bd4c2418881",
}
POLICY_FUNCTIONS = {
    "english_v6_primary_source_qa.py:admit": "f4e9c3ad6f575d857bd99baee046df232ac0638be921c58bcaaac5b74147efb3",
    "english_fresh24_ctc_admission.py:norm": "0ff8836785173f367af7e08833742eb825db863c8289ec39c65832dfa832684c",
    "english_fresh24_ctc_admission.py:verified_phone_input": "3316fdd71c66d59dd578d403874130ae6f74f41842c911d53f76a29137207d0d",
    "english_fresh24_ctc_admission.py:greedy_full_wake": "01afbd60c15d7ba0efafca62ec504abaa161c300a2ecf7b919ba6d64b60cc349",
    "english_fresh24_ctc_admission.py:decide": "388a84d138366710277fcd7605918f5e2f0c080be6883df213fcf8de62f8a20d",
}


def source_check(repository=None):
    root = HERE.parent if repository is None else Path(repository)
    observed = {name: sha((root / "automation" / name).read_bytes()) for name in OLD_FILES}
    need(observed == OLD_FILES, "OLD_PINNED_OBSERVER_AND_ADMISSION_FILES_UNCHANGED")
    functions = {}
    for key in POLICY_FUNCTIONS:
        name, function = key.split(":")
        code = (root / "automation" / name).read_text()
        node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == function)
        functions[key] = sha(ast.get_source_segment(code, node).encode("utf-8"))
    need(functions == POLICY_FUNCTIONS, "ORIGINAL_FIVE_MODEL_POLICY_FUNCTION_BYTES")
    asr_code = (root / "automation" / "english_pocket12_observe.py").read_text()
    constants = {target.id: ast.literal_eval(node.value) for node in ast.parse(asr_code).body
                 if isinstance(node, ast.Assign) for target in node.targets
                 if isinstance(target, ast.Name) and target.id in ("TEXT_MODELS", "TEXT_RUNTIME")}
    return {"schema": "english-v6-b18-observer-source-contract-v1",
            "observer_sha256": sha(Path(__file__).read_bytes()),
            "asr_adapter_sha256": sha((HERE / "english_v6_b18_asr_adapter.py").read_bytes()),
            "parser_contract_sha256": raw_asr.PARSER_CONTRACT_SHA256,
            "old_file_sha256": observed, "source_admission_function_sha256": functions,
            "asr_models": constants["TEXT_MODELS"], "asr_runtime_versions": constants["TEXT_RUNTIME"],
            "providers": list(PROVIDERS), "source_rows": COUNT,
            "maximum_calls_per_provider": COUNT, "maximum_native_calls": COUNT * 5,
            "source_indices": list(range(766, 790)), "strict_valid_five_per_source": True,
            "new_model_calls": 0, "new_network_calls": 0, "training_allowed": False,
            "promotion_allowed": False}


def guard(registration, *, env=None, event=None, parent=None):
    need(type(registration) is dict, "NEW_PREREGISTERED_B18_OBSERVER_REQUIRED")
    e = os.environ if env is None else env
    prepared = registration.get("prepared_parent", "")
    branch = registration.get("run_branch", "")
    need(re.fullmatch(r"[0-9a-f]{40}", prepared) is not None
         and re.fullmatch(r"refs/heads/research/en-v6-b18-[a-z0-9-]+", branch) is not None,
         "B18_EXACT_PREPARED_PREDECESSOR_AND_BRANCH")
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true",
         "GITHUB_REPOSITORY": "jiying2007/kws-data", "GITHUB_EVENT_NAME": "push",
         "GITHUB_RUN_ATTEMPT": "1", "GITHUB_REF": branch}.items()), "FIRST_PUBLIC_B18_PUSH_ONLY")
    need(registration.get("run_id") == e.get("GITHUB_RUN_ID")
         and registration.get("run_head") == e.get("GITHUB_SHA")
         and registration.get("run_attempt") == 1
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None
         and re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None,
         "ACTUAL_B18_RUN_IDENTITY")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared and event.get("after") == e["GITHUB_SHA"]
         and event.get("ref") == branch and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == "jiying2007/kws-data"
         and event.get("repository", {}).get("private") is False, "EXACT_UNREPLAYED_PUBLIC_B18_SUCCESSOR")
    contract = source_check()
    need(all(registration.get(k) == contract[k] for k in
             ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256"))
         and registration.get("source_schema") == SOURCE_SCHEMA and registration.get("batch_schema") == BATCH
         and registration.get("source_rows") == registration.get("max_calls") == COUNT
         and all(re.fullmatch(r"[0-9a-f]{64}", registration.get(k, "")) is not None
                 for k in ("assessment_sha256", "source_manifest_sha256")),
         "FROZEN_NEW_B18_SOURCE_AND_RAW_PARSER_REGISTRATION")
    return {"run": e["GITHUB_RUN_ID"], "sha": e["GITHUB_SHA"]}


def binding(provider, row, source_sha):
    return {"id": row["id"], "provider": provider, "source_wav_sha256": row["wav_sha256"],
            "source_pcm_sha256": row["pcm_sha256"], "source_samples": row["samples"],
            "source_manifest_sha256": source_sha, "target_conditioned": False,
            "training_allowed": False}


def row_ledger(root, rows):
    result = []
    for row in rows:
        directory = root / "rows" / row["id"]
        entry = {"id": row["id"]}
        for key, filename in (("input_started", "STARTED.json"),
                              ("forward_attempted", "FORWARD_STARTED.json"),
                              ("native_returned", "FORWARD_RETURNED.json"),
                              ("payload_valid", "PAYLOAD_VALID.json"),
                              ("capture_saved", "CAPTURE.json"), ("failed", "FAILED.json")):
            path = directory / filename
            entry[key] = path.is_file()
            if path.is_file():
                entry[key + "_sha256"] = raw_asr.pin(path)["sha256"]
        entry["raw_payload_saved"] = any((directory / n).is_file() for n in
                                           ("RAW_RETURN.json", "RAW_PROCESS.json", "RAW_PHONE.json"))
        result.append(entry)
    return result


def ledger_counts(ledger):
    need(len({r["id"] for r in ledger}) == len(ledger), "DISTINCT_SOURCE_ATTEMPT_LEDGER")
    counts = {name: sum(int(r[name]) for r in ledger) for name in
              ("input_started", "forward_attempted", "native_returned", "raw_payload_saved",
               "payload_valid", "capture_saved", "failed")}
    need(0 <= counts["capture_saved"] <= counts["payload_valid"] <= counts["native_returned"]
         <= counts["forward_attempted"] <= counts["input_started"] <= len(ledger),
         "REAL_ATTEMPT_RETURN_AND_VALID_PAYLOAD_COUNTER_ORDER")
    return counts


def complete_ledger(ledger, denominator=COUNT):
    counts = ledger_counts(ledger)
    return len(ledger) == denominator and counts["failed"] == 0 and all(
        counts[k] == denominator for k in ("input_started", "forward_attempted", "native_returned",
                                         "raw_payload_saved", "payload_valid", "capture_saved"))


def file_closure(root):
    found = []
    for path in sorted(root.rglob("*")):
        need(not path.is_symlink(), "NO_OBSERVER_EVIDENCE_SYMLINK")
        if path.is_file() and path != root / "FILES.json":
            found.append({**raw_asr.pin(path), "path": path.relative_to(root).as_posix()})
    return found


def _load_phone(provider, root):
    # Model/processor initialization and native call below match the original
    # english_v6_b11_supplement_observe path, with extra evidence around the call.
    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    import english_hifi_micro as c
    import english_hifi_phones_once as phones
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    index = 0 if provider == "xlsr" else 1
    name, revision = c.PHONE_MODELS[index]
    info = HfApi(token=False).model_info(name, revision=revision)
    need(info.sha.startswith(revision), "FROZEN_NATIVE_PHONE_COMMIT")
    cache = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / ("pipernew-phone-" + provider)
    folder = Path(snapshot_download(repo_id=name, revision=info.sha, cache_dir=str(cache), max_workers=2, token=False))
    weights = folder / "pytorch_model.bin"
    need(weights.is_file() and weights.stat().st_size == 1263535127, "ORIGINAL_PHONE_WEIGHT_SIZE")
    # HF's pinned snapshot weights may be a content-addressed cache symlink.
    # Preserve the original trusted-cache streaming hash rule, not artifact I/O rules.
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    need(digest.hexdigest() == phones.WEIGHT_SHA[index], "EXACT_NATIVE_PHONE_WEIGHT_SHA")
    vocab = phones.read_verified_hf_metadata(folder, cache, "vocab.json")
    feature = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    config = Wav2Vec2Config.from_json_file(str(folder / "config.json"))
    model = Wav2Vec2ForCTC(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    need(config.pad_token_id == 0 and feature.sampling_rate == 16000, "NATIVE_PCM16_RATE_AND_CTC_BLANK")
    lock = {"model": name, "revision": info.sha, "weights_sha256": phones.WEIGHT_SHA[index],
            "target_conditioned": False, "conv_kernel": list(config.conv_kernel),
            "conv_stride": list(config.conv_stride),
            "runtime_versions": {n: importlib.metadata.version(n) for n in
                                 ("torch", "transformers", "huggingface-hub", "numpy")}}
    record(root / "VOCAB.json", {"model": name, "revision": info.sha, "vocab": vocab, "pad_token_id": 0})
    (root / "logits").mkdir()
    def decode(wav, directory, start):
        pcm, count = phones.pcm_of(wav)
        wave = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        inp = feature(wave, sampling_rate=16000, return_tensors="pt", padding=False, return_attention_mask=True)
        need(inp.input_values.shape[-1] == count and inp.attention_mask.shape[-1] == count,
             "COMPLETE_UNPADDED_NATIVE_INPUT")
        record(directory / "FORWARD_STARTED.json", {**start, "attempt": 1})
        with torch.inference_mode():
            logits = model(input_values=inp.input_values, attention_mask=inp.attention_mask.int()).logits
        record(directory / "FORWARD_RETURNED.json", {**start, "native_return_observed": True})
        matrix = logits[0].detach().cpu().numpy().astype("<f4")
        raw = matrix.tobytes()
        saved = raw_asr.put(root / "logits" / (start["id"] + ".f32"), raw)
        original = {**start, "schema": raw_asr.RAW_SCHEMA,
                    "logits": {**saved, "path": "logits/" + saved["path"]}, "shape": list(matrix.shape),
                    "dtype": "float32-little-endian", "order": "C"}
        raw_pin = record(directory / "RAW_PHONE.json", original)
        frames = count
        for kernel, stride in zip(config.conv_kernel, config.conv_stride, strict=True):
            frames = (frames - kernel) // stride + 1
        need(matrix.shape == (frames, 392) and frames > 0 and np.isfinite(matrix).all(), "FULL_FINITE_NATIVE_PHONE_GRID")
        tokens = phones.ctc(logits.argmax(dim=-1)[0].tolist(), vocab, config.pad_token_id)
        valid_pin = record(directory / "PAYLOAD_VALID.json", {**start, "schema": raw_asr.RAW_SCHEMA,
                           "payload_valid": True, "raw_phone": raw_pin, "logit_frames": frames,
                           "vocab_size": 392, "logits_sha256": saved["sha256"]})
        return {"raw_phone_tokens": tokens, "logit_frames": frames, "vocab_size": 392,
                "logits_sha256": saved["sha256"], "raw_evidence": {"payload": raw_pin, "validation": valid_pin}}
    return decode, lock


def validate_rows(rows, source_sha, registration):
    need(source_sha == registration["source_manifest_sha256"] and len(rows) == COUNT
         and len({r["id"] for r in rows}) == COUNT
         and all(Path(r["id"]).name == r["id"] and r["id"] not in ("", ".", "..") for r in rows)
         and {r["speaker_index"] for r in rows} == set(range(766, 790))
         and all(r["role"] == "development" and r["training_allowed"] is False for r in rows),
         "NEW_FIXED_766_TO_789_ORIGINALS_ONLY")


def expected_registration(source, supplied=None):
    # Lazy import breaks the source/observer import cycle. The source module
    # authenticates its frozen assessment, implementation hashes, MANIFEST and
    # original source-generation authorization; provider agreement is insufficient.
    import english_v6_b18_source as source_module
    expected = source_module.registration(Path(source))
    need(supplied is None or supplied == expected,
         "OBSERVER_REGISTRATION_MUST_MATCH_AUTHENTICATED_SOURCE_AND_ASSESSMENT")
    return expected


def observe(provider, source, out, *, read_rows, schema, registration):
    need(provider in PROVIDERS and schema == BATCH, "ONLY_NEW_B18_FIVE_OBSERVER_SOURCE")
    run = guard(registration)
    source, root = Path(source), Path(out)
    rows = read_rows(source)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    registration = expected_registration(source, registration)
    validate_rows(rows, source_sha, registration)
    need(root.is_absolute() and not root.exists() and root.parent.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)), "FRESH_OBSERVER_OUTPUT_ONLY")
    root.mkdir(mode=0o700)
    record(root / "STARTED.json", {"run": run, "batch": schema, "provider": provider,
            "input_manifest": source_sha, "registration": registration,
            "parser_contract_sha256": raw_asr.PARSER_CONTRACT_SHA256,
            "old_admission_function_sha256": POLICY_FUNCTIONS,
            "denominator": COUNT, "whole_panel_stop_on_failure": True})
    captures, initialization_error, row_error = {}, None, None
    try:
        import english_hifi_micro as c
        record(root / "MODEL_LOAD_STARTED.json", {"provider": provider, "actual_forward_attempts": 0})
        if provider in ("xlsr", "lv60"):
            backend, lock = _load_phone(provider, root)
        else:
            assets = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / ("b18-asr-" + provider)
            backend, lock = raw_asr._load(provider, assets, root)
        record(root / "MODEL.json", lock)
    except BaseException as exc:
        initialization_error = type(exc).__name__ + ": " + str(exc)[:500]
    if initialization_error is None:
        for row in rows:
            directory = root / "rows" / row["id"]
            directory.mkdir(parents=True, exist_ok=False)
            start = binding(provider, row, source_sha)
            try:
                record(directory / "STARTED.json", start)
                wav = source / "wav" / (row["id"] + ".wav")
                measured = c.inspect_wav(wav)
                need(all(measured[k] == row[k] for k in ("wav_sha256", "pcm_sha256", "samples")),
                     "UNCHANGED_FULL_ORIGINAL_BEFORE_NATIVE_CALL")
                captured = backend(wav, directory, start)
                result = {**start, **captured, "status": "success", "input_coverage_complete": True}
                record(directory / "CAPTURE.json", result)
                captures[row["id"]] = result
            except BaseException as exc:
                row_error = type(exc).__name__ + ": " + str(exc)[:500]
                failed = {**start, "status": "failed", "error": row_error,
                          "retry_count": 0, "whole_panel_stop": True}
                record(directory / "FAILED.json", failed)
                captures[row["id"]] = failed
                break  # A missing/invalid return cannot become an abstention or a retry.
    ledger = row_ledger(root, rows)
    counts = ledger_counts(ledger)
    ledger_doc = {"schema": "english-v6-b18-native-attempt-return-valid-ledger-v1", "provider": provider,
                  "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
                  "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
                  "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
                  "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}
    record(root / "LEDGER.json", ledger_doc)
    complete = complete_ledger(ledger) and initialization_error is None and row_error is None
    summary = {"schema": schema, "provider": provider, "run": run,
        "source_manifest_sha256": source_sha, "denominator": COUNT,
        "attempted": counts["forward_attempted"], "actual_native_forward_attempts": counts["forward_attempted"],
        "native_forward_returns": counts["native_returned"], "valid_native_payloads": counts["payload_valid"],
        "complete": counts["capture_saved"], "missing_or_failed": COUNT - counts["capture_saved"],
        "status": "COMPLETE" if complete else "STOP_OBSERVER_FAILURE",
        "whole_panel_stop": not complete, "initialization_error": initialization_error, "row_error": row_error,
        "rows": [captures.get(r["id"], {**binding(provider, r, source_sha), "status": "not_started"}) for r in rows],
        "new_tts_calls": 0, "new_asr_calls": counts["forward_attempted"] if provider in raw_asr.TEXT_PROVIDERS else 0,
        "new_phone_calls": counts["forward_attempted"] if provider in ("xlsr", "lv60") else 0,
        "target_conditioned": False, "human_gold": False, "training_allowed": False,
        "retry_count": 0, "kws_forward_calls": 0, "promotion_allowed": False,
        "parser_contract_sha256": raw_asr.PARSER_CONTRACT_SHA256,
        "observer_code_sha256": sha(Path(__file__).read_bytes()),
        "asr_adapter_sha256": sha((HERE / "english_v6_b18_asr_adapter.py").read_bytes())}
    record(root / "RESULT.json", summary)
    record(root / "FILES.json", {"schema": "english-v6-b18-observer-file-closure-v1",
                                 "self_excluded": "FILES.json", "files": file_closure(root)})
    need(complete, "WHOLE_PANEL_STOP_NATIVE_RETURN_OR_PAYLOAD_FAILURE_NO_RETRY")
    return summary


def _verify_phone(root, directory, start, capture, model):
    for name in ("FORWARD_STARTED.json", "FORWARD_RETURNED.json"):
        value = json.loads((directory / name).read_bytes())
        need(all(value.get(k) == v for k, v in start.items()), "PHONE_CALL_SOURCE_BINDING")
        need(value.get("attempt") == 1 if name == "FORWARD_STARTED.json" else value.get("native_return_observed") is True,
             "PHONE_ACTUAL_ATTEMPT_AND_RETURN")
    evidence = capture["raw_evidence"]
    original = json.loads(raw_asr._read_pinned(directory, evidence["payload"]))
    valid = json.loads(raw_asr._read_pinned(directory, evidence["validation"]))
    path = root / "logits" / (start["id"] + ".f32")
    saved = {**raw_asr.pin(path), "path": "logits/" + path.name}
    frames = start["source_samples"]
    for kernel, stride in zip(model["conv_kernel"], model["conv_stride"], strict=True):
        need(type(kernel) is int and type(stride) is int and kernel > 0 and stride > 0, "NATIVE_CONV_GEOMETRY")
        frames = (frames - kernel) // stride + 1
    need(original == {**start, "schema": raw_asr.RAW_SCHEMA, "logits": saved, "shape": [frames, 392],
                      "dtype": "float32-little-endian", "order": "C"}
         and evidence["payload"]["path"] == "RAW_PHONE.json"
         and evidence["validation"]["path"] == "PAYLOAD_VALID.json"
         and valid == {**start, "schema": raw_asr.RAW_SCHEMA, "payload_valid": True,
                       "raw_phone": evidence["payload"], "logit_frames": frames, "vocab_size": 392,
                       "logits_sha256": saved["sha256"]}
         and capture["logit_frames"] == frames and capture["vocab_size"] == 392
         and capture["logits_sha256"] == saved["sha256"] and saved["bytes"] == 4 * frames * 392,
         "COMPLETE_ORIGINAL_PHONE_ARRAY_RETENTION")


def load_observations(source, evidence, rows, schema, *, registration=None):
    """Mandatory new evidence gate, followed by the unchanged five-model loader."""
    contract = source_check()
    need(schema == BATCH and len(rows) == len({r["id"] for r in rows}) == COUNT,
         "STRICT_B18_96_ORIGINAL_PANEL")
    source, evidence = Path(source), Path(evidence)
    registration = expected_registration(source, registration)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    registrations = []
    for provider in PROVIDERS:
        root = evidence / provider
        closure = json.loads((root / "FILES.json").read_bytes())
        need(closure == {"schema": "english-v6-b18-observer-file-closure-v1", "self_excluded": "FILES.json",
                         "files": file_closure(root)}, "FULL_RAW_OBSERVER_FILE_CLOSURE")
        report = json.loads((root / "RESULT.json").read_bytes())
        started = json.loads((root / "STARTED.json").read_bytes())
        need(started["input_manifest"] == source_sha and started["provider"] == provider and started["batch"] == schema
             and started["old_admission_function_sha256"] == POLICY_FUNCTIONS
             and started["parser_contract_sha256"] == raw_asr.PARSER_CONTRACT_SHA256
             and started["denominator"] == COUNT and started["whole_panel_stop_on_failure"] is True,
             "FROZEN_SOURCE_ADMISSION_AND_NEW_PARSER_START")
        reg = started["registration"]
        need(reg["source_schema"] == SOURCE_SCHEMA and reg["batch_schema"] == BATCH
             and reg["source_manifest_sha256"] == source_sha and reg["source_rows"] == reg["max_calls"] == COUNT
             and all(reg[k] == contract[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256"))
             and reg == registration, "PROSPECTIVE_REGISTRATION_BINDING")
        registrations.append(reg)
        ledger = row_ledger(root, rows)
        counts = ledger_counts(ledger)
        stored_ledger = json.loads((root / "LEDGER.json").read_bytes())
        need(stored_ledger == {"schema": "english-v6-b18-native-attempt-return-valid-ledger-v1", "provider": provider,
              "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
              "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
              "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
              "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}
              and complete_ledger(ledger), "96_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES")
        need(report["status"] == "COMPLETE" and report["whole_panel_stop"] is False
             and report["run"] == started["run"] == {"run": reg["run_id"], "sha": reg["run_head"]}
             and report["initialization_error"] is report["row_error"] is None
             and report["actual_native_forward_attempts"] == report["native_forward_returns"]
             == report["valid_native_payloads"] == report["attempted"] == report["complete"] == COUNT
             and report["missing_or_failed"] == report["retry_count"] == report["new_tts_calls"] == report["kws_forward_calls"] == 0
             and report["new_asr_calls"] == (COUNT if provider in raw_asr.TEXT_PROVIDERS else 0)
             and report["new_phone_calls"] == (COUNT if provider in ("xlsr", "lv60") else 0)
             and report["promotion_allowed"] is report["training_allowed"] is report["human_gold"] is False
             and report["observer_code_sha256"] == contract["observer_sha256"]
             and report["asr_adapter_sha256"] == contract["asr_adapter_sha256"],
             "NO_FAILED_EMPTY_OR_LOST_RETURN_CAN_COMPLETE_PANEL")
        model = json.loads((root / "MODEL.json").read_bytes())
        by_id = {r["id"]: r for r in report["rows"]}
        need(len(by_id) == len(report["rows"]) == COUNT, "96_UNIQUE_CAPTURE_RECORDS")
        for row in rows:
            directory = root / "rows" / row["id"]
            start = binding(provider, row, source_sha)
            capture = by_id[row["id"]]
            need(json.loads((directory / "STARTED.json").read_bytes()) == start
                 and capture == json.loads((directory / "CAPTURE.json").read_bytes())
                 and capture["status"] == "success" and capture["input_coverage_complete"] is True
                 and all(capture.get(k) == v for k, v in start.items()), "COMPLETE_ORIGINAL_CAPTURE_IDENTITY")
            if provider in raw_asr.TEXT_PROVIDERS:
                need(model["adapter_sha256"] == contract["asr_adapter_sha256"]
                     and model["parser_contract_sha256"] == raw_asr.PARSER_CONTRACT_SHA256
                     and model["runtime_versions"] == contract["asr_runtime_versions"][provider]
                     and model["observer"] == contract["asr_models"][provider]
                     and model["raw_before_validation"] is True and model["decode_config_unchanged"] is True,
                     "RAW_PRESERVING_FIXED_ASR_ADAPTER")
                raw_asr.validate_capture(provider, directory, start, capture)
            else:
                _verify_phone(root, directory, start, capture, model)
    need(all(r == registrations[0] for r in registrations), "SAME_FROZEN_SOURCE_REGISTRATION_ALL_FIVE")
    import english_v6_primary_source_qa as qa
    texts, phones, coverage = qa.load_observations(source, evidence, rows, schema)
    need(coverage == dict.fromkeys(PROVIDERS, COUNT), "ALL_480_VALID_NATIVE_PAYLOADS_REQUIRED")
    return texts, phones, coverage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-check", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(source_check(), indent=2, sort_keys=True))
