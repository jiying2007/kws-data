# Piper exact-IPA stress correction probe — 2026-11-16

## Motivation
The original Piper probe used `hˈeɪ` for the word “Hey”, although the frozen target IPA is `/heɪ ˈ...` and marks primary stress only on the invented name. The original four WAVs and their 12 blind ASR observations remain immutable and unadmitted.

## One-variable correction
This separate four-cell probe changes only the raw eSpeak IPA stress on “Hey”:
- Hey Mero: `hˈeɪ mˈɪəɹoʊ` → `heɪ mˈɪəɹoʊ`
- Hey Vosen: `hˈeɪ vˈoʊsən` → `heɪ vˈoʊsən`

The model revision, two LibriTTS speaker IDs, synthesis settings, output sample rate and all other parameters remain fixed. The Piper voice is loaded once for all four cells.

## Execution boundary
Exactly four new TTS calls, one model load, no retries, no ASR or training. The new audio is a distinct stress-condition experiment, not a replacement for the original probe. It must receive its own blind three-ASR and independent phone review before any positive label is considered. Human gold and training admission remain false.
