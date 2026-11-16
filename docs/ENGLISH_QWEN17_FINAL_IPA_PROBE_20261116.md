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
