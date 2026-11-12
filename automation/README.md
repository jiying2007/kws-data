# Source-first bilingual KWS provider contract (zh-CN / en-US)

**Status: research planning and offline source-only CI; no pretrained weight
is downloaded or executed, no human sample is imported or cloned, and no
training root or new evaluation result is established.**

This is an additive **v1 research protocol**, not a replacement for existing
\`research/auto-tts-voice-clone-20261102\` and
\`research/field-collection-pilot-20261102\` producer/rights/receipt contracts.
Their existing real audio and ASR runners remain canonical. In particular,
never mutate previously consumed attempts or the source-retention archive.

## Decision: separate data synthesis from on-device KWS

| Language | Offline TTS | Source-authorized clone/voice conversion | Unbiased three-ASR screening | Candidate small embedded KWS |
| --- | --- | --- | --- | --- |
| Chinese | Qwen3-TTS CustomVoice / VoiceDesign | Qwen3-TTS Base; OpenVoice V2 as distinct experimental conversion | Qwen3-ASR / SenseVoiceSmall / Whisper | WeKWS FSMN-CTC, sherpa-onnx Zipformer zh/en |
| English | Kokoro-82M / Qwen3-TTS CustomVoice | Qwen3-TTS Base; OpenVoice V2 candidate | Whisper / Qwen3-ASR / NVIDIA Parakeet TDT v3 | sherpa-onnx English/zh-en Zipformer; custom trained openWakeWord framework |

These are **candidates to verify** on the target corpus. Qwen3-TTS,
OpenVoice V2 and Kokoro model cards document permissive licensing; Parakeet
v3 is CC-BY-4.0 with attribution conditions. SenseVoiceSmall's **source**
is MIT but **official model weights** use a separate FunASR Model Open
Source License Agreement: inspect the actual pinned revision and required
attribution/model-name terms. Similarly, sherpa/WeKWS **code** licenses
do not automatically license any separate model artifact, and each Piper
voice has its own license. Official F5-TTS pretrained weights have research-limited non-commercial terms;
bundled openWakeWord pretrained models are CC BY-NC-SA, even when framework
code has a permissive license. These are research candidates, not automatically
approved for use: verify exact model-weight terms and original-speaker
permissions before executing any model.

Source verification:
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/myshell-ai/OpenVoice
- https://huggingface.co/hexgrad/Kokoro-82M
- https://github.com/FunAudioLLM/SenseVoice
- https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://github.com/SWivid/F5-TTS
- https://github.com/dscripka/openWakeWord
- https://k2-fsa.github.io/sherpa/onnx/kws/index.html

## What is now machine checked

\`automation/providers.v1.json\` is an explicit reviewed *candidate*
catalog. New independent model entries require a source code change,
not a caller-supplied URL. Its checked-in SHA is included in each plan.

\`automation/multilingual_campaign.py\` validates a **private source request**
with:
- \`locale=zh-CN\`: exactly **你好小窝** / **小窝小窝**, both complete,
  one independent positive request per keyword. \`小窝\` alone is negative.
  No implicit Mandarin experiment config or frozen model change.
- \`locale=en-US\`: generic protocol remains **researcher-defined, multiword English wake phrases**;
  existing fixtures intentionally use an invented word. For the current US/European
  English-language study, use the explicit \`--market-candidates\` guard to require
  the ordered pair **Hey Vosen** (K1) and **Hey Mero** (K2), rather than silently
  substituting a generic fixture. This pair is a frozen *comparison roster*,
  not an accepted model, validated pronunciation or shipping decision.
- \`locale=zh-CN\`: the same optional guard asserts the existing ordered pair
  **你好小窝** (K1) / **小窝小窝** (K2) without changing historical data.
- Positive / phonetic near-miss / ordinary-negative classes, plus a
  requested \`challenge\` subtype. Requested words never certify what
  the WAV actually contains; samples remain machine-screen candidates.
- Version/asset hash and model-license review hash for each selected
  provider, exactly three distinct ASR family IDs and weight identities,
  all pretrained weights require release-specific provenance and appropriate research permissions.
- Original source-family, split and reference PCM/rights evidence binding
  for cloning. Public dataset license alone does NOT establish permission
  to clone someone's recognizable voice. A local JSON declaration of
  permission also does NOT authenticate actual legal authorization.
- A strict total generation cap (<=256 cells, <=16 original voice slots,
  <=48 requested utterances, <=4 repeats, \`max_paid_usd=0\`).
  Separate inference/capture/training executors must be authorized on a
  suitable private machine, not inside this source-only planner.

All generated IDs, TTS provider revisions, declared source and split,
original requests, catalog digest, planned 3-ASR count are deterministic.
All reports keep \`human_gold=false\`, \`training_admitted=false\`,
\`independent_acceptance_evidence=false\`,
\`model_calls_executed=0\` and \`next_campaign_authorized=false\`.
Three different ASR family strings and self-supplied hashes are **not**
proof of real engine/model/source execution or true transcript correctness.

## Run privately — example only

The caller prepares an access-controlled JSON request with the exact keys
required by \`automation/multilingual_campaign.py:plan\`. Its references
and speaker IDs must be pseudonymous. For each selected provider supply:

\`\`\`json
{
  "provider_id": "qwen3-custom-zh",
  "revision_sha": "ACTUAL_PINNED_GIT_40_HEX",
  "model_assets_sha256": "ACTUAL_CHECKED_MODEL_LOCK_64_HEX",
  "license_review_sha256": "ACTUAL_INDEPENDENT_REVIEW_64_HEX"
}
\`\`\`

Placeholders are deliberately invalid. The JSON \`voices\` array contains
\`id,source_family_id,role,origin_kind,reference_pcm_sha256,rights_evidence_sha256\`.
Human-derived and public-audio references need both immutable digests
and independent use-specific permission. \`utterances\` include
\`id,text,intent,challenge,keyword_index\`. Positive \`keyword_index\`
is 1-based; negative is null. Three distinct ASR models must also be pinned.

\`\`\`bash
python3 -I -S -B automation/multilingual_campaign.py \
  --request /restricted/kws/private-zh-campaign-request.json \
  --output /restricted/kws/new-zh-task-matrix.json

python3 -I -S -B automation/multilingual_campaign.py \
  --request /restricted/kws/private-en-campaign-request.json \
  --output /restricted/kws/new-en-task-matrix.json \
  --market-candidates
\`\`\`

The current English request must list \`["Hey Vosen", "Hey Mero"]\` in that
order and include a complete positive request for each word. Suggested
initial near-miss prompts include \`Hey Wosen\`, \`Hey Miro\` and partial
\`Hey\`, plus ordinary conversation negatives; TTS audio and unbiased ASR
judgments still need real acoustic verification. Test American, British and
other European English accents on independent speaker/source groups.
The \`en-US\` protocol name does not establish a US-only speaker limit.
Near-miss suppression must not erase legitimate accented positives.

\`--market-candidates\` checks only the requested keyword roster and adds
\`market_candidate_roster_verified=true\` and a fixed research policy tag
to the **planning-only** receipt. It never runs models or authenticates
voices or training eligibility. Generic English research still works without
the opt-in flag, preserving existing protocol and historical fixtures.

Output is immutable, private mode 0600 and create-only. **Do not** commit
the private request, speaker/voice identities, model artifacts, transcripts,
reference audio or generated WAVs. Normal CI tests use invented data only,
not real model inference, Internet, open dataset scraping or paid APIs.

## English Qwen CustomVoice real-audio capture (explicit opt-in)

The new \`automation/english_qwen_capture.py\` consumes the existing
**private** multilingual source request. Unlike the planner, its explicit
\`--execute\` mode runs a real local Qwen3 CustomVoice model and writes
private mono 16-kHz WAVs plus exact started/completed/failed receipts. It
does **not** run ASR, authenticate transcripts or train a KWS model. The
English two-keyword roster is enforced and the Chinese lane is untouched.

Only \`qwen3-custom-en\` is accepted; Kokoro, F5, voice clone and other
unimplemented provider adapters are **not** silently treated as executable.
Original voice entries must use approved Qwen preset IDs such as \`ryan\`
or \`aiden\` and identity \`source_family_id=qwen3-preset-ryan\`. A
new seed does not create an independent voice. Use distinct source-family
IDs across train/development, with rights reviewed for the actual pinned
model/source. A model file lock must follow the existing
\`qwen3-tts-offline-asset-lock-v1\` schema with one Qwen CustomVoice entry
and every local weight/config file included; its exact **file-byte** SHA256
must be the private request's \`model_assets_sha256\` pin. This is not
self-authenticating ownership or a model-license approval.

Example, after the pinned model is **already** available locally:

~~~sh
python3 -I -S -B automation/english_qwen_capture.py \
  --request /restricted/kws/private-en-campaign-request.json

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B automation/english_qwen_capture.py \
  --request /restricted/kws/private-en-campaign-request.json \
  --model-lock /restricted/models/qwen-custom-model-lock.json \
  --model-root /restricted/models \
  --out /restricted/kws/en-qwen-capture001 \
  --max-calls 8 --execute
~~~

The first command validates input and makes **zero** model calls.
The second runs at most 8 new cells on the local CPU with offline weights.
It is neither a scheduled Action nor an implicit GitHub model runner.
All source/request IDs and 24-kHz/16-kHz waveforms stay outside public Git.
The capture is private create-once; any started attempt, including a
failed/interrupted one, is consumed and never regenerated on resume.
No Qwen legacy Chinese batch, especially previously verified audio, is
revisited. The overall cap is the already planned finite cell set.

This stage only establishes **actual synthetic audio** and source identity,
not pronunciation truth. Near misses like \`Hey Wosen\` may sound like
\`Hey Vosen\`; do not assign trusted negative labels merely from intended
text. Feed generated audio into **three independently pinned blind-ASR
adapters** and independent source review before new training admission.
The older Chinese-only \`asr_dispatch.py\` is *not* compatible with this
English capture format and must not be called as though it were.
Actual model availability, pinned assets and runner costs remain explicit
prerequisites; a successful source-only CI proves none of those are present.

## English three-model blind ASR (after actual capture)

\`automation/english_blind_asr.py\` now reads only hash-verified, actually
captured \`audio-16k.wav\` files from the English Qwen runner. It rejects
missing, modified or incomplete source audio. It never sends the intended
wake text, expected/negative label, speaker identity or model output from
another recognizer to an ASR process.

The fixed English three-model roster is **qwen3-asr-en /
whisper-small-en / parakeet-tdt-v3-en**. The private adapter lock contains
exactly one entry per provider:

~~~json
{
  "schema": "en-market-three-asr-adapters-v1",
  "adapters": [{
    "provider_id": "qwen3-asr-en",
    "model_revision_sha": "ACTUAL_GIT_40_HEX",
    "model_assets_sha256": "ACTUAL_MODEL_MANIFEST_BYTES_SHA256",
    "model_manifest_path": "/restricted/models/qwen-asr-lock.json",
    "executable_path": "/restricted/bin/qwen-asr-native",
    "executable_sha256": "ACTUAL_NATIVE_EXECUTABLE_SHA256",
    "timeout_seconds": 300
  }]
}
~~~

The example shows only one entry for readability; a working lock requires
**all three**. Each model manifest has the exact schema
\`kws-native-asr-model-assets-v1\` and fields
\`schema,provider_id,revision,relative_dir,files\`; each file entry
has \`path,bytes,sha256\`. The manifest's actual byte SHA256 must match the
provider pin in the private source request. Each executable must exist,
be executable and match its locked SHA256. These are source/asset integrity
checks, not proof that the model's predictions are scientifically correct.

Adapters implement one fixed CLI:

~~~sh
/path/to/native-asr-adapter \
  --audio /restricted/capture/cells/<opaque-id>/audio-16k.wav \
  --model-dir /restricted/asr-models/<model-folder> \
  --output /restricted/asr/cells/<opaque-id>/<provider>/native.json
~~~

The adapter MUST write a create-once native JSON with exactly
\`schema="en-market-native-asr-result-v1"\`,
\`provider_id,model_id,model_revision_sha,audio_wav_sha256,
processed_start_frame,processed_end_frame,decoder_completed,actual_text\`.
For a successful full-file decode: frames run from 0 through the original
16-kHz WAV frame count, and \`decoder_completed=true\`. Recognizers must
not use hotword bias or target-text hints. Adapted model binaries and their
weights must already exist locally; this repository does **not** claim
three provider-specific ASR executable implementations are available.

~~~sh
python3 -I -S -B automation/english_blind_asr.py \
  --request /restricted/kws/private-en-campaign-request.json

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B automation/english_blind_asr.py \
  --request /restricted/kws/private-en-campaign-request.json \
  --capture-root /restricted/kws/en-qwen-capture001 \
  --adapter-lock /restricted/kws/en-three-asr-adapters.json \
  --model-root /restricted/asr-models \
  --out /restricted/kws/en-three-asr001 --max-calls 9 --execute
~~~

Each native ASR attempt writes a private started receipt **before**
calling the executable, then either a completed receipt or a terminal
failed-no-retry receipt. A 3/3 result creates a machine-screen record
for review; punctuation/word normalization disagreement, apparent false
wake in a negative prompt, and unsupported Unicode stay **review-required**.
Even 3/3 agreement does NOT become authenticated gold, a trusted
negative label, training admission, FAR/FRR or model qualification.
This new runner never reads/reprocesses the older Chinese legacy batches.
Offline/source-only CI uses invented WAVs and simulated native subprocess
output; it does not execute any pretrained model.

## Reusing the November English real-audio pilot without regenerating audio

The successful **main** one-shot English TTS run can be consumed by the
native three-ASR entry without editing its original source/plan identity.
Its original artifact contains \`source-plan.json\`, \`capture/plan-lock.json\`,
and per-cell actual audio, started/completed receipts and SHA256 checks. Keep
the downloaded artifact intact and outside the repository; do not rerun the
TTS job to obtain a new audio ID.

The optional \`english_blind_asr.py --pilot-source-plan\` mode now verifies
every byte and planned cell against the original
\`english_qwen_real_once.pilot_plan\` identity. ASR model IDs and revisions
are **separately pinned**, not injected into the immutable TTS source plan.
The private \`--pilot-asr-pins\` JSON has exactly:

~~~json
{
  "schema": "en-market-blind-asr-pins-v1",
  "providers": [
    {
      "provider_id": "qwen3-asr-en",
      "revision_sha": "REAL_GIT_COMMIT_40_HEX",
      "model_assets_sha256": "REAL_ASSET_LOCK_BYTES_SHA256",
      "license_review_sha256": "REAL_SOURCE_REVIEW_SHA256"
    }
  ]
}
~~~

Supply all three independently pinned Qwen3-ASR, Whisper-small and
Parakeet-TDT providers; this one-entry example is intentionally incomplete
and must fail. Each adapter must also pass the existing physical model/
executable-hash checks. This does not fabricate ASR binaries or model
licenses, and never treats intended text as observed transcript.

~~~sh
python3 -B automation/english_blind_asr.py \
  --pilot-source-plan /restricted/en-qwen-actual/source-plan.json \
  --pilot-asr-pins /restricted/en-qwen-three-asr-pins.json

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B automation/english_blind_asr.py \
  --pilot-source-plan /restricted/en-qwen-actual/source-plan.json \
  --pilot-asr-pins /restricted/en-qwen-three-asr-pins.json \
  --capture-root /restricted/en-qwen-actual/capture \
  --adapter-lock /restricted/en-qwen-three-asr-adapters.json \
  --model-root /restricted/models \
  --out /restricted/en-three-asr-attempt001 --max-calls 9 --execute
~~~

The original \`--request\` mode remains unchanged and compatible. The two
modes are mutually exclusive. Both produce create-only machine-review
results, never human gold, training admission or formal FAR/FRR. The
asynchronous real TTS job and the independent ASR models are distinct
scientific stages; do not count the 10 actual TTS WAVs as 30 completed
native ASR calls.

## True automatic execution, once legal and acoustic sources exist

Future execution should reuse the **existing** rights-checked Qwen local
runners, pinned ASR native adapters and no-retry source receipts. Unsupported
English/Kokoro/OpenVoice provider adapters are explicitly \`null\` and must
receive independent pinned implementation/CI contracts before being run;
\`adapter\` is metadata, not an executable invocation.

Do not let an ASR worker see the target transcript, reference voice text,
another recognizer output, or hotword hints. A 3/3 agreement is a machine
*screen*, not human-gold. Quarantine disagreement, false wake in negative
requests, duplicated PCM/source ancestry, and truncated words. Never
promote a machine-screen-only record to independent FRR/FAR acceptance.
Use only verified reviewed train sources for model optimization, source-
disjoint development for data/model comparisons, and sealed new real-human
final-AFE/SSC305 tests before drawing any independent model-quality conclusion.

Next path: \`kws-data\` candidate source → private actual TTS/ASR and
authenticated label source → \`kws-pipeline\` real training admission
→ frozen model/data one-variable comparison → resource-bound Pareto model
selection → independent long FAR, final AFE and SSC305 qualification.
