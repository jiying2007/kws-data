# Strong Qwen3-TTS 1.7B Base phonetic clone — bounded actual research pilot

**Frozen before new inference.** No training, no gold labels, no legacy Meero replays, no test-set outcome selection.

## Motivation from actual 60-source results

The completed five-model run [44259758593](https://github.com/jiying2007/kws-data/actions/runs/44259758593) performed **300/300 real forwards** on the original 60 new Qwen3 Base 0.6B clone WAVs. Frozen source SCREEN artifact `14407568501` reports 18 Silver, 1 Weak, 41 UNKNOWN and `STOP_CANONICAL_SOURCE_COVERAGE`: complete groups train 0/6, development 0/2, calibration 0/2, locked-test 1/2. The original 60 WAVs and five observer artifacts are unchanged. Ordinary phrases are 12/12 silver but Mero only 2/24 and Vosen only 5/24 silver/weak. The short-phrase phonetic renderer, not the ordinary source pipeline, is the dominant measured blocker.

## New, actually different generative model

Official [Qwen3-TTS 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), exact immutable HF revision `fd4b254389122332181a7c3db7f27e918eec64e3`, replaces **only the generator** from the previous 0.6B Base. License: Apache-2.0. Original `qwen-tts==0.1.1`, CPU float32, sample output pinned to original entire 24k-native waveform resampled once to 16k PCM16. This is not a different trained KWS model.

IndexTTS-2.5 has a relevant explicit CMU phoneme facility, but its [official model card](https://huggingface.co/IndexTeam/IndexTTS-2.5) requires NVIDIA GPU with approximately 6 GB VRAM and has a separate bilibili Model Use License; no IndexTTS weights or outputs are silently imported into the CPU/GitHub-only experiment.

### Exactly 2 preselected *train-only* synthetic donor styles

Both are read only from frozen public donor run 44256271861, original artifact `14406282094`, SHA256 `8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5`. The two donor styles were fixed by the original source roster before this pilot, not picked from its locked-test predictions:
- `af_bella` (train)
- `am_puck` (train)

Each style uses exactly four newly modeled short requests: `Hey marrow.`, `Hey merrow.`, `Hey Vohsen.`, `Hey Vosen.`. Total **eight new 1.7B TTS forwards**, no retry, no prompt changes after observing results. The intended brand labels stay **Hey Mero /heɪ ˈmɛroʊ/** and **Hey Vosen /heɪ ˈvoʊsən/**, but no label is granted from request spelling.

## Independent assessment

After authentic whole-WAV source readback, exactly two existing pinned unprompted phone models perform eight original PCM forwards each (XLSR-53 and LV60, **16 actual phone forwards maximum**). Save raw CTC tokens, source WAV/PCM hashes, complete float32 logit arrays, model hashes and original donor identities, separately from requested brand target and without exposing it to models.

A byte-bound model-free reader applies the **already predeclared dual-phone** Mero/Vosen pattern from the previous 60-source policy as a research-only comparison. Even if dual phone support improves, this new 8-clip probe **cannot satisfy the 6/2/2/2 source-group requirement** and cannot enable new training. A separate actual *three ASR* + 12 original role group campaign is required before any dataset promotion. Both model-failure and insufficient source geometry are explicit STOP conditions.

Resource design: fresh public GitHub Actions CPU only, one 1.7B model loaded in one source job, one independent dual-phoneme follow-up job, bounded disk/RAM preflight; job failure retains original partial evidence. No ChatGPT Work/Codex, private GitHub repo, scheduler or repeated old datasets.
