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

Twenty offline artificial-byte tests pass in normal, -O and -OO modes. A separate
real-ZIP readback passes without a model call. These tests do not claim real ASR
success. Formal KWS training, independent evaluation and SSC305 remain pending.
