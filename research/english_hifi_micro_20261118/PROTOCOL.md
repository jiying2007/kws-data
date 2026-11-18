# English KWS — high-fidelity phoneme and synthetic-reference clone micro experiment

**Research-only, 2026-11-18. Not a model, new training label, human recording or device result.**

## Exact observed blocker

Run [44238057637](https://github.com/jiying2007/kws-data/actions/runs/44238057637), artifact 14398960405, screened *existing* Piper-192 whole PCM with 0 new TTS/ASR/phone calls. The 144 canonical request cells comprise **71 silver / 8 weak / 65 unknown**, but *zero* source groups meet all three target classes. Mero: 0 silver, 1 weak, 23 unknown; Vosen: 5 silver, 3 weak, 16 unknown. No train/dev/calibration/locked-test source-group minimum is met. The immutable upstream source is run 44107239806, artifact 14339980867, archive SHA256 `529c09953fa54914b600de0d23cca1477a78dddff620ac5be7a297b0b308d744`.

Read-only review of the existing original phoneme captures (not synthesis labels) finds frequent Mero `/eː/` in place of `/ɛ/` and final `/ɔ/` in place of `/oʊ/`; this may reflect synthesis, observer bias, or both. Do **not** relax the admission regex based on this pilot.

## New experiment and decisions frozen **before** inference

Official open-source candidate: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache-2.0, CPU `0.6B-CustomVoice` and `0.6B-Base`, PyPI `qwen-tts==0.1.1`, model revision selectors `85e237c` and `5d83992`; resolve exact immutable HF commit and retain in runtime receipt. Both are **generation-side only** and require real native forward returns.

Exact **8 TTS** requests, no post-hoc re-selection or retry:

| Speaker / model | Requested utterance | Role |
| --- | --- | --- |
| Aiden / CustomVoice | The quiet garden gate creaked open as the morning sunlight arrived. | Synthetic reference only |
| Aiden / CustomVoice | Hey Mero. | Main brand target request, NOT a label |
| Aiden / CustomVoice | Hey Meero. | Old-vowel pronunciation contrast; never positive |
| Aiden / CustomVoice | Hey Vosen. | Main brand target request, NOT a label |
| Ryan / CustomVoice | Hey Mero. | Independent preset target request |
| Ryan / CustomVoice | Hey Vosen. | Independent preset target request |
| Synthetic Aiden / 0.6B-Base voice clone | Hey Mero. | Reference-audio cloning check |
| Synthetic Aiden / 0.6B-Base voice clone | Hey Vosen. | Reference-audio cloning check |

The Aiden reference is *generated within this campaign* from an unrelated English sentence. No person's existing voice, recording, identity, or consent is assumed; this measures **synthetic-reference cloning**, not real human identity cloning. Clone Aiden does **not** become a new independent speaker group. Only two preset source groups exist here and **neither** is eligible for the source-separated trainer.

Decode each entire original 24 kHz waveform once into a full 16 kHz mono PCM16 file. Preserve all eight WAVs, exact checksums, sample counts, native model commits, model attempts, and partial failures. Do not crop, concatenate, repeat, replace, normalize selectively, or infer phoneme labels from requested text. Resource preflight must be satisfied before weights download. No retrial after a failed inference.

## Independent phone observation

After authenticated whole-WAV readback, evaluate **seven nonreference clips × two** existing independent phoneme observers (XLSR-53 and LV60), with original upstream model revision and weight SHA256 pinned. A maximum of **14** whole-PCM phoneme calls, never prompted with target phonemes/text or source class. Retain raw CTC token sequences and complete-logit SHA256, native sample counts, and per-row model IDs. Do not reinterpret either observer as human phonetic truth, especially for `/ɛ/` versus `/eː/`, or `/oʊ/` versus `/ɔ/`. Preserve disagreements verbatim.

**Promotion gates remain unchanged:** this is diagnostic and may be usable only as a new research source. A further independent three-ASR pass, complete native evidence, canonical target matching, source-group independence, proper train/dev/calibration/locked test minimum, and unchanged model selection/FAR/FRR/deployment checks are still required. Silver/weak automatically inferred solely from generation request is forbidden. UNKNOWN is never a background negative. No SSC305 claims are made.

## Separate next candidate (not triggered by this experiment)

[IndexTTS 2.5](https://github.com/index-tts/index-tts) uniquely supports English CMU phoneme control. Prospective non-executed control strings are `Hey <Mero|M EH1 . R OW0>.` and `Hey <Vosen|V OW1 . S AH0 N>.` with `Hey <Meero|M IY1 . R OW0>.` as a contrasting source only. Before its **separate** executable lane, review the Bilibili Model Use License, exact upstream code/model hashes and runner memory requirements, and preregister independent data groups; do not quietly substitute this for the Qwen3 pilot or claim the phoneme annotations were obeyed.

The comparison is about **brand phonetic fidelity on short utterances**, not generic long-sentence TTS MOS or a benchmark's reported WER. No expensive scheduled polling, ChatGPT Work, Codex, private repository or repeated old-data generation is authorized.
