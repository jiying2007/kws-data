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
- \`locale=en-US\`: **researcher-defined, multiword English wake phrase**.
  No fixed English phrase has been adopted; do **not** invent one.
  English tests use an explicitly fabricated phrase as a protocol fixture.
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
  --output /restricted/kws/new-en-task-matrix.json
\`\`\`

Output is immutable, private mode 0600 and create-only. **Do not** commit
the private request, speaker/voice identities, model artifacts, transcripts,
reference audio or generated WAVs. Normal CI tests use invented data only,
not real model inference, Internet, open dataset scraping or paid APIs.

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
