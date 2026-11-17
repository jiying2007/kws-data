# Piper v4 existing-evidence joint review

This consumes the existing source audio and saved ASR/phone receipts. It performs
no generation, ASR, phone recognition, KWS inference, optimization, or admission.
The six original archives are pinned in `automation/english_piper_v4_joint_review.py`.

```sh
python3 -I -S -B automation/test_english_piper_v4_joint_review.py -v
python3 -I -S -B automation/english_piper_v4_joint_review.py \
  --input-dir INPUT_ARCHIVE_DIRECTORY --output NEW_RESULT_DIRECTORY
```

All source WAV/PCM hashes are recomputed. Native ASR receipt text, WAV hash and
full frame range are compared with the summary and source. Phone results must
refer to the same cell, phrase, final IPA, source WAV and frame count. Source
aggregate `INCOMPLETE_NO_RETRY` is preserved while actual 16-cell completion is
reported separately. No consumed source or launch identity is modified.

## Actual result from original archives

| Phrase | Source voices/audio | Saved ASR | Saved phone results | Segment matches ignoring stress/length | Formal train/dev |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hey Mero | 8/8 | 24 | 8 | 0 | 0/0 |
| Hey Vosen | 8/8 | 24 | 8 | 1 | 0/0 |

Total audio is 10.170875 seconds. All 16 rows remain `UNCERTAIN_REVIEW_REQUIRED`.
The one segment match is Vosen/p6544; its three ASR spellings differ. Segment
agreement is a review priority, not validation of lexical stress, vowel duration
or acoustic truth. Spelling disagreement never creates a trusted negative.

The actual observer vocabulary was checked against its original model-lock. The
official displayed 43-token vocabulary reconstructs to the exact 509-byte input
SHA256 `f8fb8fa1768ddf44c3ba04248be4dce226a7d9779bef329a825ba572720eb388`.
It distinguishes `i` and `ɪ`, uses `ɹ` rather than `r`, and contains no length or
stress marks. The two added tokens are only `<s>` and `</s>` (also byte-matched to
the original lock). The comparison therefore projects Mero to `h eɪ m i ɹ oʊ`.
Missing `iː` or `ˈ` in the observation cannot be used to reject Mero. Even the
projected sequence has no complete match among these eight clips; that remains a
machine observation, not a conclusion that every Mero clip is mispronounced.

Sources: [official vocabulary](https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut/blob/main/vocab.json),
[official training script](https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut/blob/main/run.sh),
and the already retained phone artifact model-lock. The exact source and observer
manifest hashes are retained in `joint-review.json`.

## Next executable data step

Use the existing WAVs and `joint-review.csv` for focused acoustic adjudication.
Do not regenerate these 16 cells or rerun their 64 saved model observations.
Keep explicit evidence for the complete phrase, key sounds, stress/naturalness,
and final use. The original source plan assigns no train/development roles;
speaker-family grouping must be explicit before later training. This exposed
diagnostic batch is not an independent final test.

The source clips have no verified word-end annotations or one-second real
post-word observation windows. They can supply engineering PCM or reviewed
training ingredients, but cannot directly establish word-end latency. Packaging
them into an explicitly synthetic continuous training stream is a separate step
after labels are resolved; it must retain the original audio and timing lineage.

The existing `english_snapshot160_train_once.py` entry accepts the old weak-intent
snapshot and the original consumed first-main identity. It must remain unchanged;
there is no evidence-backed final-pronunciation formal training snapshot ready
to launch from v4. The next formal training run requires admitted labels plus a
speaker-grouped snapshot, then a new actual execution identity. Host performance
and SSC305 interface work can proceed immediately with the existing SDK and PCM.

## Proposed repository paths

- `automation/english_piper_v4_joint_review.py`
- `automation/test_english_piper_v4_joint_review.py`
- `docs/ENGLISH_PIPER_V4_JOINT_REVIEW_20261117.md` (this readme)
- `research/english_piper_v4_joint_review_20261117/joint-review.json`
- `research/english_piper_v4_joint_review_20261117/joint-review.csv`

Raw archives remain at their original artifact IDs in the report. They are also
downloaded locally with verified bytes under `data_evidence/` for immediate reuse.
