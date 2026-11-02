# Automated synthetic TTS and synthetic-reference voice cloning

This is a **development-only, non-shipping** pipeline to reduce manual review for
the Chinese Xiaowo KWS phrases 你好小窝 / 小窝小窝. The previous 12-clip
Qwen batch produced only 2/12 strict three-engine target matches and zero
three-engine target matches for 你好小窝. This pipeline automatically searches
new voices and prompts, screens native output and uses one bounded positive-only
refinement. It cannot substitute for real-human final-frontend/SSC305 evidence.

## Voice paths without requiring new human reference recordings

1. **Preset:** Qwen3-TTS 0.6B CustomVoice, already pinned at revision
   85e237c12c027371202489a0ec509ded67b5e4b5. Default voices Ryan,
   Aiden, Sohee and Ono_Anna, each retaining its stable preset ancestry.
2. **Design:** Qwen3-TTS 1.7B VoiceDesign describes new **fictional adult**
   synthetic voices; needs its own exact pinned 40-hex revision.
3. **Clone:** Qwen3-TTS 0.6B Base reuses a derived voice prompt ONLY from a
   local **synthetic** reference that already has exact 3/3 native ASR screen
   receipts. Never clone unconsented people, celebrities, scraped speech or
   unknown-licence human references. Every cloned voice stays in the reference
   source family and train/development role; a clone is not an independent
   real speaker.

The publisher lists Qwen3-TTS model/source rights as Apache-2.0. Output and
conditioning chain rights still need a separate commercial-product review.
Reference: https://github.com/QwenLM/Qwen3-TTS
F5-TTS pretrained weights have a separate non-commercial license despite MIT
code; that provider is not enabled automatically.

## Setup once, then run unattended bounded campaigns

One controlled Linux machine needs qwen-tts, torch, numpy, scipy, soundfile
and huggingface-hub installed, plus three independently configured local
Qwen3-ASR/SenseVoice/Whisper native adapters. These large pretrained models
are NOT deployed to SSC305; only the eventual small KWS inference engine is.

A. **Plan** (zero calls; create-only result):

~~~sh
python3 -B campaign.py --output /controlled/auto/plan.json --batch-id pilot001
~~~

The default campaign has 24 clips: 4 preset voices by 6 phrases, comprising
two 你好小窝 punctuation variants, one 小窝小窝 and three near-homophone
confusables per voice. A custom voices JSON may specify a designed synthetic
voice or a previously machine-screened synthetic reference clone.
All model revisions must be exact immutable 40-hex Git identifiers.
Seeds and IDs are derived deterministically. No second run is treated as an
independent speaker.

B. **Download public pinned model files ONCE and freeze every byte**:

~~~sh
python3 -B prepare_models.py --plan /controlled/auto/plan.json \
  --model-root /controlled/models --output /controlled/auto/model-lock.json \
  --download --max-download-bytes 8589934592
~~~

--download is explicit; the command verifies public repository identity,
exact revision, model size and available disk before any transfer, then
records original bytes and file hashes. Without --download it locks already
materialized files. The generation runtime rechecks the lock before loading
weights and requires HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1. No paid GPU
fallback, secrets or network at inference time. 1.7B design needs much
more RAM than the 0.6B preset and may not fit a basic CPU runner.

C. **Configure pinned blind native-ASR command adapters**. The offline JSON
adapter lock must contain exactly qwen06, sensevoice, whisper, each with
model ID/revision, model-asset hash, executable file hash, <=300-second timeout
and a fixed argv array with both {audio} and {output} literal placeholders.
A trusted adapter receives only the opaque 16-kHz mono PCM16 WAV and its
output path, never the intended target, TTS prompt, other ASR text or voice
identity. It writes a native JSON containing audio_id, engine, model_id,
model_revision, actual_text, decoder_completed, processed_start_frame and
processed_end_frame. Each adapter's actual model assets and code provenance
must be independently checked; the claimed hash is not by itself proof.

D. **Run the entire bounded development sequence with one command**:

~~~sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B autopilot.py --plan /controlled/auto/plan.json \
  --workspace /controlled/auto/run01 \
  --model-lock /controlled/auto/model-lock.json \
  --model-root /controlled/models \
  --adapter-lock /controlled/auto/asr-adapters.json \
  --max-tts 36 --max-asr 108 --rounds 2 --execute
~~~

Without --execute: validation only, zero model calls. With --execute: automatic
TTS, native/16k waveform metrics, three blinded ASR calls, complete native
results and strict machine-only classification. If a train K1 has weak/failed
recognition, an independent positive-only round may be produced ONCE.
No uncontrolled text/weight/seed/threshold sweep, auto retries, or promotion.
Run again with the same plan/workspace to continue untouched cells: started
but failed/interrupted cells remain consumed. The aggregate budget counts
already-started attempts across rounds and optional clone stages.

For optional **fully automatic synthetic clone** stage, additionally supply
--clone-revision with an exact Base model revision and have it in the model
assets lock. Only a technically valid 3/3 exact positive candidate can
produce a new synthetic-reference clone prompt. Its family and split are
unchanged. The resulting copies are not distinct human participants.

## Autonomous quality triage: not human gold

- 3/3 complete native outputs exactly equal target text: a machine-screened
  development candidate, NEVER automatic production training truth.
- 2/3 or contradictory outcomes: bounded disagreement review or new
  independent-generation candidate; never majority truth.
- silent, clipped, nonfinite, incomplete, unpinned or malformed WAV/ASR:
  reject or hold. Do not silently adjust gain, erase blank spans or change
  intended labels after examining a failure.
- Unicode typography only is normalized. 窝/屋/沃/吴/王 remain distinct.
- Proposed negative phrases require actual spoken-text review before being
  admitted as false-alarm negatives. Synthetic derivatives are never sealed
  independent qualification audio.

Human review should be concentrated on the few disputed near-homophones and
source-rights cases; formal real-human/final-AFE and physical SSC305
qualification cannot be automated away by model self-agreement. No source
root, trained weight, release tag, KWS threshold or shipping gate is modified.

## Source CI

~~~sh
python3 -B test_automation.py -v
python3 -B -O test_automation.py -v
python3 -B -OO test_automation.py -v
~~~

CI uses only synthetic short PCM data, fake backends, and locked adapter
fixtures. No pretrained models, ASR, TTS downloads, training or paid runners
are invoked. The source manifest registration workflow creates one proposal
commit OBJECT only and never changes the PR ref by itself. Actual generation
requires an explicitly bounded execution environment outside this source CI.
