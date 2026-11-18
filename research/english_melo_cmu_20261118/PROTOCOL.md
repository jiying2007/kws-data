# Controlled CMU phoneme input — independent generator study

This campaign deliberately changes **the TTS input and acoustic generator**, not the KWS training policy. The baseline [run 44259758593](https://github.com/jiying2007/kws-data/actions/runs/44259758593) returned 300/300 genuine five-observer checks on 60 Qwen3-0.6B voice-clone WAVs but left 41 UNKNOWN, 18 Silver, 1 Weak; source group coverage was 0/6, 0/2, 0/2, 1/2 and `training_snapshot_allowed=false`.

A stronger **Qwen3 1.7B Base** experiment [run 44261773165](https://github.com/jiying2007/kws-data/actions/runs/44261773165) completed eight actual new 1.7B synthesized WAVs and sixteen XLSR/LV60 whole-PCM forwards. Result: zero of eight candidates had both phoneme observers supporting the complete canonical phone family. A bigger language model is not sufficient to control new brand vowel/consonant output.

## New model / language frontend: independent of Qwen, Piper and Kokoro

Use **official [MyShell MeloTTS](https://github.com/myshell-ai/MeloTTS)**, exact Git commit `209145371cff8fc3bd60d7be902ea69cbdb7965a` (MIT), with **English VITS checkpoint** `myshell-ai/MeloTTS-English@17f1313104f4ad12d09bffc2fcbe2e72227ac161`, checkpoint SHA256 `acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c`. The upstream official project supports CPU inference and includes five trained English style IDs. Pin its English BERT frontend to `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`, including tokenizer and model weights. No unpinned alternative TTS model is imported.

Before any **numerical** TTS forward, set the English native CMU dictionary entries and inspect the **actual upstream phoneme tokens**, not merely a requested text string. Freeze:

| Word | Native CMU request | Interpretation |
| --- | --- | --- |
| `Mero` | M EH1 / R OW0 | brand /ˈmɛroʊ/ |
| `Vosen` | V OW1 / S AH0 N | brand /ˈvoʊsən/ |
| `Meero` | M IY1 / R OW0 | non-brand positive **contrast only** |
| `Vozen` | V OW1 / Z AH0 N | voiced **contrast only** |

Reject execution if the actual package G2P output does not contain exactly the declared critical phones in each respective requested utterance.

## Fixed source roster and call budget

Five public upstream **English style identities**: EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default. They are synthetic model styles and **not** five verified independent humans or new KWS source groups.

For every style, synthesize **exactly five new complete short utterances** using native inference: `Hey Mero.`, `Hey Vosen.`, `Hey Meero.` (negative phoneme contrast, never KWS positive), `Hey Vozen.` (voiced contrast, never KWS positive), and `What time is it?` (ordinary candidate). Total **25 actual source requests**, no iterative string search, no retries, no legacy PCM copy, no listener-selected sample rejection.

Save full original 44.1-kHz native waveform and one complete resample to 16 kHz mono PCM16 WAV; retain SHA256 and sample count, actual G2P phone/tone input, upstream checkpoint hash, native voice ID, and failure receipts. No original old Meero training bag or earlier Qwen3/Kokoro/Piper corpus is reprocessed.

## Next gate

First examine actual source outcomes. If G2P control, runtime and voice generation succeed, independently run both existing phoneme observers on the **same** 25 complete WAVs and save all logits, plus three ASRs if the phonetic pilot demonstrates promise. Do not treat correct G2P **input** as acoustic or phonetic **output** gold. Only new fully observed phoneme+ASR consensus can become silver/weak labels. The KWS trainer stays OFF until a separately sound 12-voice source-separated population reaches exactly train 6, development 2, calibration 2, locked-test 2 groups **with both brands and ordinary**, under a non-post-hoc prospective rule.

This branch is a bounded actual data-generation experiment and source-evidence pivot, **not** a claim that final training data or an SSC305 deployable model exists. No Work/Codex/private repo/cron. GitHub public Actions CPU only.
