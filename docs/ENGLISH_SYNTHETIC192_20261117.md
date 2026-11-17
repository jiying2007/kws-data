# English 192 fresh synthetic sources — finite first-pass campaign

This is the next real data expansion for English research training and a usable
software SDK. Newly collected human speech and a physical SSC305 are not entry
conditions. The source stage generates new audio and retains its first automatic
observations. Training consumes only the later byte-pinned corpus artifact.
Results remain synthetic research evidence; the campaign does not claim natural
human performance or that a historical IPA is the user's final American target.

## New material and immutable source roles

The existing Piper LibriTTS high model has 904 source identities. This campaign
uses the 24 untouched source indices 10–33. Previous Piper indices 0–9 and every
40 LibriSpeech source identities used by the current coverage/natural32 work are
excluded by their shared numeric LibriSpeech/LibriTTS identity.

| Role | Piper indices | Source speaker IDs | New clips |
| --- | --- | --- | ---: |
| Train | 10–21 | p3615, p318, p6104, p1382, p5400, p5712, p2769, p2573, p1463, p6458, p3274, p4356 | 96 |
| Development | 22–25 | p8498, p5570, p176, p339 | 32 |
| Calibration | 26–29 | p28, p5909, p3869, p4899 | 32 |
| Locked test | 30–33 | p64, p3368, p3307, p5618 | 32 |

Each voice gets exactly the same eight cells, including both keywords and ordinary
negative candidates. Conditions, windows, crops and later augmentation inherit
the source role. A failed or unknown cell is never replaced or moved to another
role. These are new source groups for KWS; all share one pretrained Piper engine.
They are not a held-out TTS engine or independent newly recorded human speakers.

| Cell | Exact synthesis request | Research hypothesis |
| --- | --- | --- |
| mero-natural | `Hey Mero.` | `mero-us-eh-v1` |
| mero-historical-i | `Hey Meero.` | `mero-us-i-v1` |
| vosen-natural | `Hey Vosen.` | `vosen-us-s-v1` |
| vosen-voiced-z | Direct phones `heɪ vˈoʊzən`, displayed request `Hey Vosen` | `vosen-us-z-v1` |
| ordinary-time | `What time is it?` | `ordinary-negative-v1` |
| ordinary-light | `Please turn on the kitchen light.` | `ordinary-negative-v1` |
| ordinary-package | `The package is by the front door.` | `ordinary-negative-v1` |
| ordinary-window | `Could you open the window?` | `ordinary-negative-v1` |

The official CMU pronunciation dictionary at commit
`74790861f652b15e4ac49015a90074ad62a27690`, dictionary blob
`2c0411740cce3e2026a80b90b650d5f6a7258164`, contains `mero M EH1 R OW0`.
This supplies the explicit US lexicon research branch. The previous long-i
rendering remains a separate research branch. Neither silently replaces the
active policy's unknown final owner IPA. The natural Vosen /s/ branch and voiced
/z/ branch are separately named hypotheses; disagreement with the old canonical
record does not turn one into a negative.

The worker captures the actual en-us G2P sequence and phoneme IDs for text cells.
For the raw Vosen cell it calls `PiperVoice.phonemes_to_ids` and then
`PiperVoice.phoneme_ids_to_audio` directly. It never depends on bracket-parser
semantics. All cells make exactly one synthesis forward at most. An input phone
sequence is generation intent, while the separate audio-to-phone model observes
the actual output. The initial Mero admission distinguishes /ɛ/ and /i/ strictly;
other observed vowels remain explicit unknowns under the pinned intake policy.

## Pinned runtime and rights provenance

Piper 1.8.0 uses `rhasspy/piper-voices` revision
`de3dcfdf1912bb49726dc4aa11c26017ce2ac62a`, model
`en/en_US/libritts/high/en_US-libritts-high.onnx` (136,673,811 bytes,
SHA256 `9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f`).
The existing exact 20,163-byte config MD5 is
`cb1564bb779f3b1f87e2de778adeb669`; its complete SHA256, all speaker mappings and
installed Piper API source are also retained by the real generation job.

LibriTTS train-clean-360 is CC-BY-4.0; attribution follows the
[upstream voice model card](https://huggingface.co/rhasspy/piper-voices/blob/de3dcfdf1912bb49726dc4aa11c26017ce2ac62a/en/en_US/libritts/high/MODEL_CARD).
The voice repository declares MIT. The invoked current
[Piper 1.8.0 engine source](https://github.com/OHF-Voice/piper1-gpl/tree/v1.8.0)
is GPL-3.0, a distinct code license. No TTS engine or checkpoint is embedded in
the resulting KWS SDK. The campaign records these identities separately.

Three existing blind ASR loaders are reused by exact source SHA256: Qwen3-ASR
0.6B, Whisper-small, and Parakeet TDT v3. The independent first phone pass is the
already executed 392-token XLSR observer, not the old LJSpeech-only observer:
`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, revision
`2c733782da5604684829819a5eb744c193fe9398`, weights SHA256
`04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551`.
Its existing loader verifies model bytes, uses tensor-only loading, and receives
only the original 16 kHz PCM. No target text, IPA, source role or KWS score enters
any observer. Raw text and every raw IPA token are retained.

## One executable public Actions chain

`english-synthetic192-once.yml` runs source-only tests for a PR. The actual path
requires an enabled plan, exact first-main predecessor, public repository,
main push and attempt 1; it has no schedule or manual dispatch. Every real stage
rechecks the same source pins before runtime installation.

The chain is source checks → generation → four parallel first-observer jobs →
byte-only corpus join and automatic admission. Bounds are 192 TTS calls,
192 calls for each of three ASRs, and 192 XLSR forwards. Generation has a 900-second
worker deadline and each observer a 3,600-second deadline. Outer workflow
timeouts and original started/completed/failed receipts preserve partial work.
An attempted cell is never retried. All 192 intended rows remain in the manifest,
including failures, incomplete observations and duplicate-PCM quarantine.

The final join downloads the exact already merged kws-pipeline profile and
`training/english_synthetic_silver.py` pinned by commit and file hashes. It
rechecks the original audio and captures, then emits the versioned admission
report. Three mapped ASR observations plus whole-phrase phone support produce
the silver research tier. A separately declared two-of-three tier is weak and
has weight 0.5. Unknowns are masked and never become background. Ordinary
negative candidates require complete observations and target-absence checks;
their requested text is insufficient. A source group contributes supervised
examples only when both keywords and ordinary negatives are supported. Minimum
balanced source groups are frozen at 6/2/2/2 for train/dev/cal/test; missing groups
never trigger resplitting or extra synthesis.

## Handoff to the actual model experiment

The immutable artifact is named `english-synthetic192-corpus-<actual run ID>`.
Its root contains `CORPUS_MANIFEST.json`, `MACHINE_EVIDENCE.json`, `ADMISSION.json`,
`EXECUTION_PLAN.json`, `RESULT.json` and a closed `FILES.json` inventory. The
original WAV, raw PCM, native 22,050 Hz WAV, phone IDs and receipts are under
`source/audio/<id>/`. Raw observer captures are under
`evidence/<provider>/<id>/CAPTURE.json`. The exact admission module, profile and
their source identity are under `policy/`.

The separate training job binds the real artifact ID, run metadata, size and
ZIP digest only after it exists, verifies the whole packet, and recomputes
admission before reading supervised examples. It uses the frozen embedding
frontend and portable head first. The checkpoint rule is fixed before training;
new calibration and locked-test source groups do not choose a model or alter the
pronunciation hypothesis. Previous exposed evaluations remain regression checks.
Successful retained weights can proceed to native/ARM software parity and SDK
assembly without waiting for new human recordings or a physical SSC305.

Existing Qwen final144 + schwa16, Qwen63, prior Piper16, and the 64 verified old
audio records keep their original bytes and labels. None is regenerated or
reobserved by this source workflow. Existing accepted coverage remains the
installed fallback until a new candidate passes its own fixed quality gate.
