# English KWS — 18 new direct-phoneme synthetic source groups

Pre-audio protocol. At registration, no fresh18 TTS, ASR, phoneme inference or model training has occurred. Historical exposed data are frozen and not relabeled.

## Original real scientific blocker

- [Fresh24 five-native source review 44306842000](https://github.com/jiying2007/kws-data/actions/runs/44306842000) completed 840/840 actual independent forwards and produced [artifact 14429385634](https://github.com/jiying2007/kws-data/actions/runs/44306842000/artifacts/14429385634), SHA256 5a9b226b8fe475331d9f03d3ca4f91d5bb94c5270113f6d0cf3e8b4c36fe8316.
- The authentic frozen result is STOP_CANONICAL_6_2_2_2_SOURCE_COVERAGE: 58/168 Silver, comprising 24 ordinary, 21 Mero and 13 Vosen, plus 110 UNKNOWN. Original qualified groups train4/development1/calibration3/locked0 versus minimum 6/2/2/2. These counts and rows remain immutable.
- Train/development observations show name-transcript contradictions as a major bottleneck. Original locked-test acoustic rows have not been used to choose the new generator, thresholds or roles.

## Newly pre-registered controlled source

- Select the last 18 previously unused Kokoro voice-pack assets by exact immutable LFS SHA256 and bytes in [protocol](../../automation/english_fresh18_direct_protocol.json). Every one of 36 older synthetic presets is excluded. Distinct voice packs mean distinct synthetic source groups, NOT independently verified humans; underlying Kokoro model is common ancestry.
- Assign roles using an outcome-independent, predeclared 32-bit base-131 rolling hash over source names: 10 train / 4 development / 4 locked-test; no new calibration groups because previously frozen calibration contains three qualified groups. No reshuffle after results.
- Call the official unmodified native KModel direct phoneme API at two fixed speeds 0.92/1.08 for each brand target. Hey Mero raw phones hˈA mˈɛɹO. (/heɪ ˈmɛroʊ/), Hey Vosen hˈA vˈOsən. (/heɪ ˈvoʊsən/, /s/ not /z/). Also generate one What time is it? ordinary candidate, one Meero /i/ control and one Vozen /z/ control per group.
- Seven total utterances per 18 groups = 126 maximum native CPU forwards. The old and voiced controls are never supervised training negatives and always UNKNOWN. None of the request strings or phones are acoustic gold.
- Pin official weights, model commit, 18 source asset SHA256, original whole 24k-to-16k PCM, seed, role, duration, full native phonemes, 16k WAV and PCM digests, no crop, no retry. A source model/load failure stops and preserves original failure receipt.

## Next evidence and training gate

- Independent follow-on observation of the 126 new WAVs will use the previously frozen CTC contrast rule Git blob 0a688325d0d36e83ef6623ac678511df24a92374 and the same unprompted Qwen3-ASR, Whisper-small, Parakeet, XLSR and LV60 model families. No tuning against any locked-test KWS material.
- Only if all original and new source groups qualify with exact lineage and full 6 train / 2 development / 2 calibration / 2 locked-test is fresh-seed English model training allowed. No old Meero-positive initialization, history mutations, human FAR/FRR or SSC305 claims.
- Research uses public GitHub Actions and open weights only; no Work, Codex, private repositories, paid API or scheduled patrol.
