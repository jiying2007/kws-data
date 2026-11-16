# Independent phoneme observations for final Piper v2 — 2026-11-16

## Final target
The controlling American targets from the user's final pronunciation-standard image are:
- Hey Mero: /heɪ ˈmiːroʊ/ (MEE-roh)
- Hey Vosen: /heɪ ˈvoʊzən/ (VOH-zən; voiced medial /z/)

## Frozen audio
Piper v2 run 43789385672, artifact 14191228436, SHA-256 `8ab7ff273ce5ee98e4c4bd83f43fd80b82c7ae03ac89449ded93a4cd61bedc61`. Four exact source WAV/PCM hashes are verified before inference.

## Independent phone observer
Reuse the already reviewed `bookbot/wav2vec2-ljspeech-gruut` checkpoint, exact revision and safetensors hash. Run exactly four audio-only CTC forwards. The model receives no target text, IPA, speaker ID, ASR transcript or source-role metadata. Preserve raw token output and bind it to the source WAV SHA.

This model has LJSpeech/Gruut domain and speaker bias. Its output is a secondary machine observation only, not phonetic truth, human gold or training admission. No TTS, ASR, KWS inference, training or optimizer steps occur in this workflow.
