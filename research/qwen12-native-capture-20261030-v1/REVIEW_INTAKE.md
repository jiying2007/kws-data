# Qwen12 blind-audition review intake (non-admitting)

Purpose: avoid copying human observations or ASR guesses into the product training lane without proof. This is a strictly offline **observation-to-immutable-audio binding** step for the already archived 12 Qwen16 training candidates. The four Serena `development_a` recordings do not enter this handoff, and are not moved to training by this step.

## Input

The original verified `qwen12-train-blind-main-41055537577` handoff has `blind/job.json`, twelve `blind/audio/*.wav`, and `control/source-map.json`. The only accepted source map is SHA-256 `14b1fa88d20acb88a4f05f214adebdc6a1762f0929cb6395e9f5098295793b01`; its original archive is SHA-256 `78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1`. The validator reopens and rehashes all twelve WAVs and decoded PCM via the pinned `fresh_contract` before reading a review.

The review file is **not** generated here. It is a local JSON exported by the separate, offline blind-listening page with schema `qwen12-manual-auditory-observation-v1`. A person must actually listen; generated default/fixture values are not evidence. Do not send human identity, raw audio or review observations to an external provider without permission. Keep `control/` isolated from ASR worker mounts.

Run on a stable, access-controlled local filesystem:

```sh
python3 -B research/qwen12-native-capture-20261030-v1/review_intake.py \
  --handoff-root /controlled/qwen12-train-blind-main \
  --review /controlled/qwen12-blind-human-observation.json \
  --output /controlled/new-observation-audit.json
```

Output is create-only and must not exist. It binds IDs, WAV/PCM/frame hashes, fixed source roles and all 12 audio identity records. Incorrect SHA, unknown files, duplicate JSON keys, nonfinite numbers, unsupported verdicts, token-like expected text injected by a client, output collisions, missing or modified original audio, symlinked input and false claimed approval all fail closed. Text normalization changes whitespace/punctuation only: it **never** treats `窝`, `屋`, `沃`, `吴` or `王` as equivalent.

## Hard boundary

The exported local browser identity and timestamp are **self-declared, not authenticated**. Even a full twelve-clip review, `complete` verdict, or observed wake string does not establish reviewer independence or veracity; ASR consensus likewise cannot replace actual acoustic evaluation. The validator always outputs `training_admitted=false`, `human_gold=false`, `qualification_allowed=false`, `release_authorized=false`, and `rights_policy_review=PENDING`. It cannot write a `speech-like-audio-review-v2` reviewer signature or change any source trust root, training source manifest, release config, threshold, model artifact, test/qualification holdout, or physical SSC305 outcome. **No model is loaded or run.**

Before any future training use, independently verify who heard the audio, actual spoken words and completeness, source/voice rights and consent, family lineage/split isolation, suitable negative corpus, and a code-reviewed production train-source root. Formal real-human final-AFE and target-board qualification remain separate.

## Reproduction

```sh
KWS_FRESH_BLIND_DIR=/controlled/qwen12-train-blind-main/blind \
  python3 -B research/qwen12-native-capture-20261030-v1/test_review_intake.py -v
```

Repeat with `-O` and `-OO`; the tests use the original source bytes and **synthetic reviewer fixtures only**. No actual review, training, ASR, TTS or paid resource is performed by the code or tests.
