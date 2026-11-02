# Bulk multi-speaker cloning for Xiaowo KWS (private local corpus)

## Purpose and hard boundaries

The previous source-only #47 Qwen TTS/VoiceDesign/synthetic-clone implementation
supports a few synthetic voice references. This extension allows many **distinct
source speaker groups**, each producing dozens of positive wake and negative
utterances, using the Qwen3-TTS Base model with one **cached prompt per
reference**. It handles legitimate open audio when dataset-specific voice-clone
rights are independently reviewed, plus separately consented human speech and
authorized synthetic references.

**No private reference audio, original dataset, speaker re-identification key,
consent files or trained model weights are ever uploaded by CI.** These tools
operate on a controlled local corpus and only store non-authoritative generated
synthetic candidates. Human voice identity cannot be inferred from a name or
speaker key. One voice with multiple clips/seeds/clones remains one source
family and cannot cross the train/development split.

Open audio is not automatically approved for cloning. Mozilla Data Collective
requires dataset-specific consumer terms and some derivative Common Voice
dataset pages expressly forbid voice-cloning even with CC0 metadata. Review:
https://commonvoice.mozilla.org/terms/en.html
https://mozilladatacollective.com/terms/consumers
https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5

An operator must preserve actual source/voice rights and use-specific approval.
The utility checks local rights-review record SHA256 per reference and rejects
nonempty prohibited uses, absent commercial/synthetic scope and audio/terms
mismatch. A user-authored review JSON still **cannot authenticate consent**.

## 1. Import a locally prepared approved audio catalog

Create a private catalog of 16/24 kHz mono PCM16 WAV references (1–15 sec),
containing a private source speaker key, relative audio path, original/source
audio SHA256, exact WAV SHA256 and optional genuinely verified transcript.
The importer uses a separate 32+ byte **private HMAC key file** to create stable
opaque speaker/source-family IDs, one representative ~5-second clip per speaker.
No name, original speaker key or person ID appears in output.

For each approved dataset, supply a private permit: schema
kws-external-voice-clone-permit-v1, dataset/source kind, source license and
terms SHA, independently reviewed evidence file SHA, explicit flags
voice_cloning_explicitly_allowed=true and
commercial_synthetic_kws_allowed=true, prohibited_uses=[], and
review_status=independently-reviewed. These are declarations whose human/legal
origin remains outside the tool; the importer does **not** automatically
authorize human voice cloning from a CC0 license alone. Any specific ban wins.

Import command (the directory contains only already authorized local data):

~~~sh
python3 -B bulk_ingest.py \
  --catalog /restricted/catalog.json \
  --permit /restricted/permit.json \
  --rights-evidence /restricted/rights-review-record.json \
  --audio-root /restricted/references \
  --pseudonym-key-file /restricted/secret-pseudonym-key.bin \
  --max-refs 100 --output /restricted/reference-pool.json --execute
~~~

Without --execute this command is a no-write dry run. The executed path creates
private exact per-reference rights metadata and the catalog with create-only
semantics, never downloads audio. Respect official provider access and
distribution controls; original MP3 must be lawfully obtained and normalized
externally with transformation provenance retained.

## 2. Generate many positive + negative intents per voice

~~~sh
python3 -B bulk_pool.py --sources /restricted/reference-pool.json \
  --audio-root /restricted/references \
  --model-revision EXACT_40_HEX_QWEN_BASE_REVISION \
  --batch-id pilot01 --repetitions 2 --max-total-calls 160 \
  --output /restricted/pilot01-plan.json
~~~

Default matrix per reference: 4 positive templates for the two wake words
including punctuation variants, 8 hard negatives (屋/吴/我/王/沃,
partial wake and repeated 你好), 4 ordinary conversational negatives.
With repetitions=2: **8 positive +16 hard negative +8 ordinary negative =32
intended utterances per reference**. For example, 5 references means 160
generator calls and at most 480 three-ASR calls. 100 references means
3200 generator calls and potentially 9600 ASR calls, **not** 100 independent
real-human wake evaluation subjects. Start with a small 2–5 source pilot.

Plan caps: 512 references, 16384 cells, 1–8 repetitions and explicit
total model call cap. Each invocation permits <=64 new TTS or ASR calls.
Started and failed cells are not reissued. No automatic model weight or
threshold sweeps, unbounded retries or paid runner escalation.

## 3. Run pinned Base model, cache the source prompt, generate many WAVs

Use an exact locally verified Qwen3-TTS 0.6B Base model folder and
kws-clone-base-model-assets-v1 file lock. The runtime rehashes all model assets.
The official Qwen API builds one voice_clone_prompt per source reference
and reuses it for every intended positive/negative sentence. If an actual
reference transcript is verified, use ICL; otherwise use x-vector-only.
Do not invent source transcripts from request prompts. x-vector-only may reduce
clone quality.

~~~sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B bulk_runner.py \
  --sources /restricted/reference-pool.json \
  --audio-root /restricted/references \
  --plan /restricted/pilot01-plan.json \
  --model-root /restricted/pinned-base-model \
  --model-lock /restricted/base-model-asset-lock.json \
  --out /restricted/generated-pilot01 \
  --max-new-calls 32 --execute
~~~

Retains original 24k float WAV, 16k PCM16 derivative and signal/identity
metrics in private output; per-cell started/completed/failure receipts are
content-addressed and non-retry. KWS on SSC305 does not run this large TTS
model; TTS is an offline data factory.

## 4. Run pinned three-ASR adapters and screen intended positives/negatives

The three already reviewed local adapter commands (Qwen3-ASR, SenseVoice,
Whisper) receive only opaque audio/output paths, not target phrases or TTS
prompts. Each has a separate exactly-once attempt, raw native JSON and
full-frame end receipt.

~~~sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B bulk_asr.py \
  --plan /restricted/pilot01-plan.json \
  --generated /restricted/generated-pilot01 \
  --asr-output /restricted/asr-pilot01 \
  --adapter-lock /restricted/three-asr-offline-lock.json \
  --max-new-calls 64 --execute
~~~

Create report with no new model calls:

~~~sh
python3 -B bulk_asr.py --plan /restricted/pilot01-plan.json \
  --generated /restricted/generated-pilot01 \
  --asr-output /restricted/asr-pilot01 \
  --adapter-lock /restricted/three-asr-offline-lock.json \
  --report /restricted/machine-screen-new.json
~~~

Three complete native ASRs exactly matching intended text yields only
machine-screened candidate positives/negatives. A negative whose audio is
transcribed as either wake phrase gets quarantined rather than called a clean
negative. If engines disagree or any output is incomplete, hold for review.
Words 窝/屋/沃/吴/王 are never auto-substituted. Intended prompts are
not verified actual acoustics, and these synthetic negative clips do not
replace long independently recorded real-world negative audio.

## Product qualification

All generated and screened data remains training_admitted=false and
qualification_allowed=false. This source-only PR neither adds an accepted
product source root nor creates CTC weights. External speech license/
speaker-rights review, representative true voice diversity, actual spoken-text
verification, real-human final AFE and SSC305 physical performance/FAR still
need separate authority. No human voice is impersonated in CI test fixtures.

## Offline tests

~~~sh
python3 -B test_bulk.py -v
python3 -B -O test_bulk.py -v
python3 -B -OO test_bulk.py -v
~~~

Tests use only artificial small PCM WAVs and mock ASR/Qwen outputs. No
provider login/download, microphone, human speaker, model import or inference.
