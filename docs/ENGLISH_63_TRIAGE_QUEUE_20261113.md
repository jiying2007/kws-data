# English 63 original WAVs: reproducible three-evidence pronunciation queue

This research-only path rebuilds the already recorded 63-row / 14-pair
machine-review queue from the **three original and immutable** Actions ZIPs:

| Saved evidence | Source run | Artifact ID | ZIP SHA256 |
| --- | ---: | ---: | --- |
| Source-linked Qwen3/Whisper/Parakeet ASR | 43267739514 | 13965259534 | 3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133 |
| Real Log-Mel/DTW acoustic comparison | 43334622072 | 13996662526 | 64efe60c4b8c45cf130b2b55b4be6674f665dbcf7d667548c463dda15d951e18 |
| Independent raw-audio IPA model | 43340235981 | 13997774459 | a70ee89c76bf25f8a0a6b82a5a72c46128782e7f2020fb7655c06845de131be0 |

The original 63 WAVs came from TTS run 43242278962. None of these
archives should be rerun or regenerated. This program performs **zero**
TTS, ASR, IPA, KWS inference or model training.

automation/english63_evidence_review_queue.py checks each ZIP SHA256,
its expected inner JSON SHA256, original cell ID and WAV hash across
all three sources, PCM uniqueness, 7 preset voice families and
train/development roles. It reads only the saved JSON evidence, never
needs the original WAV bytes, and produces:

- 63-row machine evidence joined JSON with source SHA256/IPA/ASR
  observations, with all rows marked training_admitted=false,
  human_gold=false, trusted_negative=false.
- 14 same-synthetic-voice plain/punctuated wake pairs, split into
  10 train-role review priorities and 4 development-only diagnostics.
- A small CSV with source IDs and review ordering, not a truth label.

The review queue ranks *machine uncertainty*, not whether a phrase
was pronounced incorrectly or whether the KWS model would wake.
Nearphone Wosen/Miro/Rosen remains untrusted, especially under
regional accents. All original audio remains outside the repository.

Use three previously downloaded Actions ZIPs:

    python3 -I -S -B automation/english63_evidence_review_queue.py \
      --asr-zip /restricted/original-three-asr-review.zip \
      --dtw-zip /restricted/original-dtw-review.zip \
      --ipa-zip /restricted/original-ipa-review.zip \
      --output-dir /restricted/new-review

The output directory is create-only. Normal/-O/-OO CI uses invented
metadata, never pretrained models. A single protected-main trigger
runs the actual read-only original artifact join and checks the
already-known report SHA256:
89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12.

This is NOT acoustic ground truth or formal training admission.
Independent pronunciation adjudication and a separate English training
label contract are needed before treating any row as trusted speech.
Any actual completed join/CLI only demonstrates reproducible reuse of
existing evidence, not new model quality or independent FAR/FRR.
