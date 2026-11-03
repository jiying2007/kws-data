# Bulk multi-speaker research KWS voice sources (private only)

## 1. Single noncommercial research source intake

The supported source importer is
[automation/research_only_bulk_ingest.py](../../automation/research_only_bulk_ingest.py).
It uses source-bound original/normalized WAVs, release-specific reviewed
**noncommercial research TTS** scope, HMAC pseudonyms and create-only private
receipts. There is **no commercial source importer** or backward-compatibility
alias on the current branch. Git history retains the old implementation for
historical audit without exposing a runnable production path.

See [research scope and invocation instructions](../../automation/RESEARCH_ONLY_REFERENCE_SCOPE.md)
and [open-source preparation](../../automation/OPEN_SOURCE_REFERENCES.md).
No generated or cloned voice establishes a new independent real-human
qualification speaker. Keep original audio, consent/review evidence, source
identities and generated recognizable voices in controlled private storage.
A publicly downloadable file alone does not grant rights beyond its release
terms, including distribution or any explicitly prohibited reuse.

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

## Research qualification and optional future product checks

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
