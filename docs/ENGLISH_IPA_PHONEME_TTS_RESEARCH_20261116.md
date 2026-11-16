# English exact-IPA TTS research probe — 2026-11-16

## Why a second TTS path is being tested

The four Qwen3-TTS 1.7B CustomVoice samples were generated successfully, but three independent blind ASR systems did not agree with the requested lexical forms. For Mero, Qwen3-ASR heard “Mero” while Whisper/Parakeet returned “Merrow/Marrow”. For Vosen, Qwen/Whisper returned “Vosin” and Parakeet “Vozin”. No sample is acoustically qualified or admitted.

The Qwen3-TTS CustomVoice API exposes an optional free-text instruction, not a hard phoneme-token input. The instruction did not produce stable evidence of the owner-confirmed readings. This probe therefore tests a different route: Piper's documented raw espeak-IPA phoneme blocks.

## Exact source pins

- Engine: OHF-Voice Piper TTS 1.8.0, source commit `639388b6317fc4731e91d53da42aea68fd4166ff` (GPL-3.0 engine, used only in this research data-generation worker; not included in any target firmware or shipped KWS artifact).
- Voice repo: `rhasspy/piper-voices`
- Voice revision: `de3dcfdf1912bb49726dc4aa11c26017ce2ac62a`
- Voice: `en_US-libritts-high`, 904-speaker model, 22.05 kHz, trained from scratch on LibriTTS train-clean-360.
- ONNX model: 136,673,811 bytes; SHA-256 `9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f`.
- Voice config: 20,163 bytes; MD5 `cb1564bb779f3b1f87e2de778adeb669`.
- Voice dataset license: CC BY 4.0; retain attribution to LibriTTS / OpenSLR 60 when redistributing generated audio. This is a research candidate and is not a production data-license approval.

The chosen model is not the LibriTTS-R voice that was fine-tuned from Lessac. It is the separate LibriTTS high voice explicitly described as trained from scratch on train-clean-360.

## Stress alignment correction

The v2 raw phoneme probe used an explicit primary stress marker on the word Hey. The image specifies Hey as light/unstressed and the name's first syllable as stressed. Therefore v2 is retained as historical evidence but is not the final stress-aligned target. The v3 probe uses raw eSpeak sequences without a stress mark on Hey: heɪ mˈiːɹoʊ and heɪ vˈoʊzən.

## Superseded probe notice

The earlier four-cell Piper probe and its ASR consumer were generated against the now-superseded readings /heɪ ˈmɪəroʊ/ and /heɪ ˈvoʊsən/. Those WAVs remain immutable historical evidence and are not positives for the current standard. The v2 probe used the right segmental vowels/consonants but placed primary stress on Hey; it is superseded for the image's prosodic target. The v3 probe is the final stress-aligned variant.

## Fixed four-cell probe v2

| Phrase | Exact IPA | Raw eSpeak IPA | Speaker IDs |
|---|---|---|---|
| Hey Mero | /heɪ ˈmiːroʊ/ | `hˈeɪ mˈiːɹoʊ` | 0, 1 |
| Hey Vosen | /heɪ ˈvoʊzən/ | `hˈeɪ vˈoʊzən` | 0, 1 |

Speaker IDs 0 and 1 correspond to LibriTTS speaker IDs p3922 and p8699 in the pinned voice configuration. They are dataset speakers, not user-supplied cloned identities.

Piper's documented `[[ ... ]]` syntax bypasses normal text-to-phoneme conversion and injects raw eSpeak IPA phonemes. The runner verifies that the phoneme sequence survives parsing and every symbol exists in the pinned voice's phoneme map before synthesis.

## Fixed execution budget and output boundary

- Exactly 4 TTS calls; one model load; no retries.
- No ASR, no optimizer, no training, no model qualification.
- 16-kHz derived WAV and native 22.05-kHz WAV with PCM/WAV SHA-256 receipts.
- Source audio remains `pronunciation_verified=false`, `human_gold=false`, `training_admitted=false`.
- Only the four WAVs, source plan, model lock, receipts and package freeze are uploaded; model weights remain in the ephemeral worker and are removed before artifact publication.

The result must still pass three independent blind ASR and acoustic review. Raw phoneme control reduces the text-to-phoneme ambiguity but does not guarantee a natural or correct acoustic realization.

## Final owner-confirmed source

The controlling standard is the user-provided image dated 2026-11-16. American target: Mero /heɪ ˈmiːroʊ/ (MEE-roh, long /iː/); Vosen /heɪ ˈvoʊzən/ (VOH-zən, voiced /z/). British variants are /heɪ ˈmiːrəʊ/ and /heɪ ˈvəʊzən/ and are tracked separately.
