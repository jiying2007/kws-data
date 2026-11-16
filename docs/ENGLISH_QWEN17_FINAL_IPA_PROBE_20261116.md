# Final-IPA Qwen 1.7B TTS probe — 2026-11-16

## Purpose
The final owner-confirmed American targets are:
- Hey Mero: /heɪ ˈmiːroʊ/ (long stressed /iː/)
- Hey Vosen: /heɪ ˈvoʊzən/ (voiced medial /z/)

The first final-IPA Piper v2 four-cell experiment produced strong machine evidence for Vosen but not Mero. This is a separate TTS route experiment, not a replacement for Piper audio.

## Frozen model and controls
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- revision 0c0e3051f131929182e2c023b9537f8b1c68adfe
- Apache-2.0 model declaration
- two named built-in voices: Aiden and Ryan
- exactly four calls, no retries
- pronunciation instructions explicitly distinguish long /iː/ from /ɪ/ and voiced /z/ from /s/

## Evidence boundary
Instruction text is a conditioning request, not proof that the generated audio obeys it. All four samples remain unverified, machine-only research audio until blind ASR, independent phoneme observation and human pronunciation review. No training admission or model qualification is granted by generation.

## Completed generation and exact-byte continuation

The generator completed at source `5273e92135f71245dca531221f2f51f1415113c7`,
Actions run `43793684582`, artifact `14193452864`. The 302,067-byte ZIP SHA-256 is
`e737354ad8ed69d511f061781f5b6cc3c86ff1ea70ba9c72084c5ac4cd1c738f`.
Independent local readback verified all 21 archive members and all four original
16-kHz mono PCM16 WAV/PCM identities. Durations are 1.28, 1.04, 0.72 and 1.12 seconds.
These are integrity observations, not listening or pronunciation approval.

The old generator, its trigger, and its actual run remain reconstructible from
that immutable source commit. The live module now consumes this existing ZIP;
`--execute` explicitly refuses another TTS run. There is no new audio generation.

The same workflow continues with three independent provider jobs using the
existing `english_seven_voice_asr_once._load` implementation (Git blob
`f3bf07e6237dfaeebf7a92d49a821ac0efcdc09b`). No new ASR engine is introduced.
Each decoder receives an anonymous original WAV, not a phrase, IPA, preset,
split or another recognizer's transcript. Four calls per provider / 12 maximum,
no retries, at most two simultaneous provider jobs. Calls are counted only when
an actual decoder is invoked; load failures cannot count as completed recognition.

Execution is limited to the first protected-main push whose `before` SHA is
`5273e92135f71245dca531221f2f51f1415113c7`, with run attempt 1 and a public repository.
Both workflow and Python enforce that bound. Later source edits, reruns, forks,
private repositories and other branches cannot silently replay this campaign.
A changed base requires an explicitly reviewed new execution decision, not a
weakened guard. The original generation job has been retired, not retriggered.

Successful provider jobs automatically join their exact-source observations in
`three-asr-summary.json`. Failed or unknown calls retain their receipts and block
the combined success result. Uploaded artifacts contain JSON/runtime receipts,
not duplicate audio or model assets. A successful summary still has
`training_admitted=false`, `human_gold=false`, `pronunciation_verified=false` and
`model_promotion_allowed=false`; it is input to acoustic review, not a KWS model.
The target includes light Hey and first-syllable name stress, which ASR spelling
alone cannot validate. No broad accent rejection or invented negative labels.

## Completed blind ASR and independent phoneme continuation

The blind ASR stage completed in run `43802227932` at commit
`e844bfbff4d579397eb93de0f2e27bf38fe1af8a`: each of Qwen3-ASR, Whisper-small
and Parakeet completed four calls, twelve in total. The combined artifact
`14196731830` has ZIP SHA-256
`d2d22f1d68d3a62e15e6856398fee2f5d8bbe17502c16706a0bb6c426acd791d`.
An independent ZIP readback recovered these exact observations:

| Requested phrase / preset | Qwen3-ASR | Whisper-small | Parakeet |
|---|---|---|---|
| Hey Mero / Aiden | Hey Mero. | Hey, Marrow! | Hey Merrow said. |
| Hey Mero / Ryan | Hey, Mero. | Hey, Mero! | Ei mero! |
| Hey Vosen / Aiden | Hey, Vosin. | Hey, Rosen. | Hey Vosin. |
| Hey Vosen / Ryan | Hey, Vozin. | Hey, Vosin! | Hey Vozin |

Spelling differences do not establish incorrect or correct acoustics. In
particular, no ASR transcript proves the required long /iː/, medial /z/, or
light Hey. The four original WAVs therefore continue into the existing
independent phoneme observer, without another TTS or ASR call.

The observer reuses `english63_independent_ipa.py` (Git blob
`016df0222bf74f356399c30a9d7049b6ba5d4480`) and the pinned
`bookbot/wav2vec2-ljspeech-gruut` revision
`7be1e6f27c8163ce790688d62fd9ce1731177c2e`, with safetensors SHA-256
`9bc8cd20e1348fc384ba4b9cd60a5c761dbb3f0b7071d1bdc33331ed1b358f29`.
It receives anonymous original audio only, with no target, transcript, voice
or split hints. At most four forwards are allowed, once per source WAV, with
no retries. Per-cell receipts, raw tokens, stressless tokens, source hashes,
the model lock and runtime package versions are retained.

This is a separate, reviewed execution stage: both the workflow and Python
require the first public-main successor push whose `before` commit is
`e844bfbff4d579397eb93de0f2e27bf38fe1af8a`, attempt 1. The older TTS and ASR
execution guards remain unchanged and cannot replay on this push. A moved
base or failed/unknown execution is not permission to weaken those guards.

Phoneme observations have single-narrator/domain bias and remain machine
evidence. They cannot grant acoustic gold, pronunciation verification,
training admission or model promotion. Simulated backends are explicitly
marked. Formal KWS training, independent evaluation and SSC305 remain pending.

## Observed phonemes and the next concrete data correction

Independent phoneme run `43808456802` completed all four real forwards, with
zero new TTS, ASR or optimizer calls. Its artifact `14199511253` has ZIP SHA-256
`cbdcaba3ba1c7dfa64dc825942887c3467d18c8349a323ee2f9c5be6717d446b`.
An independent byte-verified readback returned:

| Canonical phrase / voice | Raw machine tokens |
|---|---|
| Hey Mero / Aiden | eɪ p ɛ ɹ oʊ z |
| Hey Mero / Ryan | ə m ɛ ɹ ə l |
| Hey Vosen / Aiden | oʊ v oʊ z ə n |
| Hey Vosen / Ryan | eɪ eɪ v oʊ z ɪ n |

The fixed observer's vocabulary expresses the tense vowel as `i`, with no
length or stress tokens. The corresponding segmental comparison targets are
`h eɪ m i ɹ oʊ` and `h eɪ v oʊ z ə n`; the lack of a literal `iː` token is not
itself a mismatch. Both Mero observations nevertheless contain `ɛ`, not `i`.
Vosen has support for voiced `z`, but remaining prefix/vowel differences and
prosody are unresolved. No row is admitted or relabelled as a trusted negative.

The next finite experiment changes the text sent to the same pinned TTS
backend, not the user's names or pronunciation targets:

| Canonical phrase | New synthesis text | Target unchanged |
|---|---|---|
| Hey Mero | Hey Meero. | /heɪ ˈmiːroʊ/ |
| Hey Vosen | Hey Vozen. | /heɪ ˈvoʊzən/ |

`english_final_ipa_rendering_once.py` declares exactly these two renderings
for Aiden and Ryan, with light Hey and first-syllable name stress requested.
Its new `enrender-` cell identities keep the actual synthesis text distinct
from the canonical phrase. The old four WAVs and observations remain unchanged.

The new workflow performs one connected execution: four new synthesis calls,
then four audio-only calls per existing observer (three ASR providers plus the
phoneme observer), then an exact-source combined review artifact. Each stage
consumes only its successful same-run predecessor. The run, source commit,
recipe, code, source plan, model lock and each WAV/PCM identity are bound in
the receipts. No provider receives the desired phrase or another transcript.
The maximum is four TTS plus twelve ASR plus four phoneme calls, with no retry,
no prompt/seed sweep and at most two simultaneous observer jobs.

Both Python and workflow require the first public-main successor push whose
`before` is `b0810b837c8c31d66bfbfeb1181e7b7585b12f21`, attempt 1. Original
campaign guards are unchanged. Synthetic tests cannot become real source or
combined evidence. A failed or interrupted stage retains its receipts and
blocks automatic consumption. A complete combined result still grants no
formal training, human gold, pronunciation approval or model promotion.

This experiment addresses the observed Mero vowel conflict directly. Expansion
to a separate source-split research training batch depends on the observed
usefulness of these renderings; four probe clips are not a training dataset
or evidence of final FAR/FRR/SSC305 completion.

## Completed rendering observations and independent-domain diagnosis

The connected rendering run `43809825038` completed four real TTS calls,
twelve real ASR calls and four bookbot phoneme forwards. Its source artifact
`14199609338` has ZIP SHA-256
`93b47d8885bf592f219a658bd27090c9af3c1a80bc5a3cae98bb49b9af7b7661`;
combined review artifact `14200506304` has ZIP SHA-256
`b7af8b2a3151b81e9801cc21f0e2ef876cfc578847aa69ee571d6e879aa02b25`.
The original lengths are 0.88/0.96/0.88/0.96 seconds. Independent readback
verified the ZIPs and each WAV/PCM identity before interpreting the rows:

| Rendering / voice | Qwen3-ASR | Whisper-small | Parakeet | Bookbot raw phones |
|---|---|---|---|---|
| Meero / Aiden | Hey Miro. | Hey, Miro! | Hey Miro! | k j ʊ ɹ oʊ |
| Meero / Ryan | Hey Miro. | Hey, Miro! | Hey Miro | ɪ ɹ oʊ |
| Vozen / Aiden | Hey, Vosin. | Hey, Vosin! | Hey Vozin | p t͡ʃ eɪ v oʊ oʊ ɪ n |
| Vozen / Ryan | Hey, Vozin. | Hey, Vosin! | Hey Vozin! | p eɪ oʊ p v oʊ z ɪ n |

The three ASR word hypotheses are more consistent for Mero, but the phoneme
observer still does not establish the requested tense long vowel. Neither
spelling agreement nor an uncertain phoneme sequence confirms light Hey,
stress, duration, or pronunciation. Source-only inspection found no concrete
PCM-scaling or CTC-collapse bug in the bookbot adapter. Its single-narrator
LJSpeech domain and vocabulary without length/stress tokens remain limitations.
The prepared 144-cell expansion therefore remains inactive at this point.

One separately bounded observation now compares the same four original and
four rendered WAVs using `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, revision
`2c733782da5604684829819a5eb744c193fe9398`. The official model is trained for
phonetic recognition on multilingual Common Voice and requires 16 kHz input.
Its vocabulary distinguishes `iː`, `i`, `ɪ`, `ɛ`, `s` and `z`. This addresses a
specific observer-domain/phone-inventory uncertainty; it is not an automatic
tie-breaker, pronunciation approval, or replacement of the earlier evidence.

Official source: [model card](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/tree/2c733782da5604684829819a5eb744c193fe9398),
[weights pointer](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/blob/2c733782da5604684829819a5eb744c193fe9398/pytorch_model.bin),
and [vocabulary](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/blob/2c733782da5604684829819a5eb744c193fe9398/vocab.json).
The fixed PyTorch weights are 1,263,535,127 bytes, SHA-256
`04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551`.
The local loader permits weights only and disables remote code. No text or
phonemizer output enters the acoustic model. Exact metadata hashes and runtime
versions accompany raw CTC tokens and per-input attempt receipts.

The new `independent-xlsr` job in the existing rendering workflow and
`english_final_ipa_xlsr_once.py` require the first public-main successor push
whose `before` is `0641690f2af5468992485319d04ba7c2b4fe6a61`, attempt 1.
The maximum is eight real forwards, once per pinned WAV, no retries. Existing
TTS/ASR/bookbot guards remain unchanged. No new synthesis, ASR, KWS inference
or optimizer update is authorized by this stage. Partial failures retain
receipts and exit unsuccessfully. All gold, pronunciation, admission and
promotion flags stay false, including if the new model agrees with a target.

## Actual eight-WAV XLSR result and bounded-usefulness decision

[Independent XLSR run `43812073656`](https://github.com/jiying2007/kws-data/actions/runs/43812073656)
completed all eight declared forwards on the unchanged four original and four
rendered WAVs. It made zero new TTS, ASR or optimizer calls. Artifact
`14200951947` has ZIP SHA-256
`c7d8f1ce717218ca1490cd57d253a95ba95a9dca2c46e33a41cd9f61a40ae03d`.
The original `summary.json` bytes have SHA-256
`737ea6cf45ee9c5d32765771b6f9c84bc3a467174f994c222505b9e6e9886983`;
the original model-lock bytes have SHA-256
`71a30ee3d506cd10911a531c7a97918757ad7c924fbb01316e0f126fc0dffce1`.
Byte-verified artifact readback recovered the following raw CTC tokens:

| Source rendering | Canonical phrase | Preset | Raw XLSR tokens |
|---|---|---|---|
| Original Mero | Hey Mero | Aiden | `h eɪ m ɛ ɹ ɔ` |
| Original Mero | Hey Mero | Ryan | `e m e r o` |
| Rendered Meero | Hey Mero | Aiden | `eɪ m i ɹ oʊ` |
| Rendered Meero | Hey Mero | Ryan | `h e m i ɹ o` |
| Original Vosen | Hey Vosen | Aiden | `eɪ v oː z ə n` |
| Original Vosen | Hey Vosen | Ryan | `h eɪ v oː z ɪ n` |
| Rendered Vozen | Hey Vosen | Aiden | `p eɪ v oː z ə n` |
| Rendered Vozen | Hey Vosen | Ryan | `h eɪ v oː z ɪ n` |

For the two paired synthetic presets, the Mero spelling correction changes the
observed name-vowel category from original `ɛ`/`e` to rendered `i` in both cases.
This is directional machine acoustic evidence that the Meero rendering is useful
for a bounded follow-on experiment. It does not establish the required long
`iː`, first-syllable stress, light Hey, or acceptable whole-phrase pronunciation.
The XLSR inventory contains both `i` and `iː`; these observations are retained as
`i`, without adding a length or stress claim. Earlier contradictory bookbot
observations remain part of the record and are not overwritten.

Both Vosen spellings produce medial `z`. The paired Vozen results show no
observed improvement: Ryan's token sequence is unchanged and Aiden's rendered
sequence adds an initial `p`. Consequently the new recipe retains original
Vosen spelling while selecting Meero for Mero. The original and rendered pilot
audio is unchanged. Neither `oː` versus target `oʊ` nor Ryan's `ɪ` versus target
`ə` is silently accepted as a verified pronunciation variant.

The frozen decision is `USEFUL_FOR_WEAK_RESEARCH`, limited to one new
requested-intent data experiment. Its basis is the paired Mero vowel-category
improvement and the absence of a Vosen rendering gain. It is not a pronunciation
pass, an acoustic-gold decision, or satisfaction of the pronunciation-specific
positive-admission gate. The original rendering review used by this decision
is the exact ZIP member, SHA-256
`ebd3cccae358a4dcd2f439984a2d96fa124ee70c195d6f777420c70c454a269e`;
a locally reformatted JSON is not substituted for those evidence bytes.

## Declared 144-cell requested-intent experiment — execution pending

The next experiment is declared in
[`final144_recipe.py`](../automation/final144_recipe.py),
[`final144_runner.py`](../automation/final144_runner.py), and
[`final144_execution_pins.json`](../automation/final144_execution_pins.json).
It has not produced a completed 144-cell dataset or a trained successor at this
documentation stage. The global recipe SHA-256 is
`d1fe7973324bf35f9fcd1c20472786e121c18bb3e4bdbe8967953a0f88243b83`.

The canonical names and American targets remain Hey Mero `/heɪ ˈmiːroʊ/` and
Hey Vosen `/heɪ ˈvoʊzən/`. Synthesis text uses Meero and original Vosen, with
eight fixed natural delivery variants for each name. Every preset also receives
eight declared ordinary sentences, with no nearphone-negative intent. This
gives 24 cells per preset and 144 new cell identities:

| Source role | Synthetic presets | Wake-intent cells | Ordinary-intent cells | Total |
|---|---|---:|---:|---:|
| Train | Aiden, Dylan, Eric, Serena | 64 | 32 | 96 |
| Development | Ryan, Vivian | 32 | 16 | 48 |
| Total | Six presets | 96 | 48 | 144 |

Preset ancestry defines the source-family split. Aiden and Ryan were exposed in
the pilot; none of these presets is an independent human speaker or a locked
confirmation source. Canonical phrase, actual synthesis request, source family,
role, variant and requested-intent class remain separate fields. Ordinary
sentences are weak background intents, not trusted-negative acoustic labels.

Generation uses nine fixed consecutive partitions of 16 cells, with at most two
jobs running together. Each job invokes the existing
`english_qwen_capture.capture(..., max_calls=16)` once. The global ceiling is
144 TTS calls; each cell has one actual attempt and no retry. A failed or
started-but-unknown cell remains consumed. A missing or partial partition does
not authorize a replacement call or a new cell ID to fill the quota.

All nine partitions must be complete before any observer runs. Their original
partition plans, plan-locks and receipts remain intact. Verification binds the
same run, head, code, recipe and execution pins; requires identical TTS
model-lock bytes across partitions; and checks every original WAV, decoded PCM,
native-audio receipt and source role. It does not rewrite the nine plan-locks as
one global plan. The four observers then consume the same authenticated PCM:
144 calls each for Qwen3-ASR, Whisper-small, Parakeet and XLSR, for maxima of
**432 new ASR calls and 144 new XLSR forwards**. No observer receives the desired
text or another model's transcript. Old pilot TTS and observations are not
replayed by this experiment.

The source and observation stages are create-only, attempt-1 protected-main
operations. Incomplete sources or observer summaries block automatic downstream
consumption. All four complete 144-row summaries must bind to the same source
cells; their original byte hashes are retained as a four-summary manifest, not
misreported as the hash of a single ZIP artifact. Runtime versions, model locks,
raw observations and per-cell started/completed/failed receipts are retained.

## Frozen weak-intent selection and research-only head work

[`english_final144_embedding_once.py`](../automation/english_final144_embedding_once.py)
declares the selection policy, whose canonical SHA-256 is
`27a18716b5c0e7f29a937eeedde085917c04d4d45f30db0d9115dc3c881a9fd5`.
The feature/head implementation is reviewed in
[kws-pipeline PR #823](https://github.com/jiying2007/kws-pipeline/pull/823).
This is a separate requested-intent experiment, not a claim that current-target
pronunciation-specific positive admission has passed.

A wake-intent row requires at least two of the three ASR outputs to contain
exactly normalized `hey` plus a predeclared spelling: `mero`, `miro`, or `meero`
for Mero; `vosen`, `vosin`, or `vozin` for Vosen. Normalization only removes
straight/curly apostrophes, lowercases, and extracts ASCII alphanumeric tokens.
The same row must also contain the corresponding contiguous XLSR core:

| Weak requested intent | Required raw-token core |
|---|---|
| Mero | `m` + one of `i`, `iː` + one of `ɹ`, `r` + one of `oʊ`, `oː`, `o` |
| Vosen | `v` + one of `oʊ`, `oː` + `z ə n` |

These alternatives are a frozen machine-support screen for intent research;
they do not broaden the user's accepted pronunciation set. In particular,
Vosen `z ɪ n` does not satisfy this policy's required `z ə n`, even if ASR spells
the name as Vosin or Vozin. A Mero `i` can support a weak requested-intent class
without becoming evidence of verified `iː`, duration or stress.

An ordinary-intent row requires two exact normalized requested-sentence
transcripts and neither target phone core. Unknown, conflicting or unmatched
rows are quarantined; an uncertain wake candidate never becomes a negative.
The full 144-row source denominator, exclusions, raw observations and source
hashes remain visible. No selected row becomes human gold or a trusted negative.

Minimum evidence is checked twice: first after source/PCM deduplication, then
again after actual frontend/window geometry exclusions. At both checks, each
word requires at least eight train rows from at least two train source families,
and two development rows from at least one development source family. Ordinary
synthetic intents require at least eight train rows and two development rows.
Real read-speech background cannot substitute for these ordinary-intent minima.
Failure of either check stops before optimizer updates. Unsupported long clips
or feature windows are recorded as exclusions, without trimming away part of a
wake clip or generating replacement samples.

The declared background pool uses the authenticated, previously exposed 517
LibriSpeech clips from 40 real speakers. Deterministic speaker assignment keeps
30 speakers in train and 10 in development; at most 16 seconds of original PCM
per speaker form 40 disclosed read-speech tracks. Background prefixes and
context come only from the matching role, with exact source spans retained.
These constructed read-speech splices are not independent household continuous
negative gold. They add no invented silence and do not provide phonetic or
word-end annotations.

If all data and geometry gates pass, the worker will use the fixed frozen
embedding and train one small causal head with seed `20261116` and exactly
96 optimizer updates. Development data does not tune the configuration,
threshold, epoch count or stopping point. The research event rule is fixed at
probability 0.80, three consecutive confirmations and thirteen embedding steps
of cooldown; it remains uncalibrated. Saved actual weights and feature streams
will then be reopened for numerical/event replay in host C and ARM under QEMU.
That parity scope is the saved causal **head only**. It does not establish ARM
parity for the full audio frontend or embedding backbone, or execution on
SSC305 hardware.

All formal-training admission, human-gold, trusted-negative, pronunciation,
qualification and promotion flags remain false. Vowel length, stress and Hey
acoustics remain unverified. Independent FAR/FRR, word-end latency and SSC305
audio-chain/resource/stability measurements are `NOT_RUN`. Actual batch/run
identities, generated counts, eligible/excluded counts, optimizer completion,
saved weights and parity outcomes must be added from the future worker's
terminal artifacts; none is claimed by this declaration.
