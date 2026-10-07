# Sources, rights, changes, and privacy boundary

This is a small scientific record of already completed experiments, prepared for
separate publication review. It contains numerical outputs, source identities,
explicit projections of reports, and invented probability fixtures. It contains
no natural or synthetic recording, decoded PCM, model weight, compiled runtime,
fbank/MFCC/splice vector, source voice frame, full transcript, or model inversion.
It adds no training input or data-catalog entry.

## DEMAND

Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent, *DEMAND: a collection of
multi-channel recordings of acoustic noise in diverse environments*, v1.0 (2013).
Official record: https://zenodo.org/records/1227121
DOI: https://doi.org/10.5281/zenodo.1227121
Source license: CC BY-SA 3.0 Unported,
https://creativecommons.org/licenses/by-sa/3.0/ and
https://creativecommons.org/licenses/by-sa/3.0/legalcode.en

The measured source is exactly DLIVING/ch01.wav from DLIVING_16k.zip. The full
unmodified WAV and decoded PCM identities are retained. Original amplitude was
used, with no gain, resampling, clipping, trimming, looping, mixing or normalization
in this experiment. The publisher's 16 kHz release already involved upstream
resampling. Neither recording nor original documentation is redistributed here.
Source notices/attribution and ShareAlike requirements remain applicable to any
future redistribution of the source or an adaptation. Unrelated source code is
not assigned this source license by this record.

## Google FLEURS

Google FLEURS, cmn_hans_cn, official validation rows 0–19, revision
70bb2e84b976b7e960aa89f1c648e09c59f894dd.
Pinned official card:
https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md
Card SHA256: 688f79f2a5c731af3796e9f683eb02f9b3f09d040decd8c5625d0f37098e71c6
Authors: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod,
Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna.
*FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*
(2022): https://arxiv.org/abs/2205.12446
Source license as declared by that card: CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/
A historical supplementary legalcode download failed with HTTPError and was not
retried. This record does not turn that retrieval into a success or include the
unretrieved legal text.

The admitted float32 WAV identities refer to official Dataset Server served HTTP
bodies, not independently retrieved upstream archive bytes. Upstream cache
re-encoding was not audited. Each of the same 20 recordings was later converted
once to lossy PCM16 by nearest-even rounding of float32*32768 then int16 clipping.
No sample clipped in this batch; maximum observed error was 0.5 LSB. This is a
measured batch result, not a universal saturation error bound. Originals remained
unchanged. Only file/payload hashes, text hashes, numeric counts and saved A20
outputs are included. No signed or cached asset URL is published. Full transcript
strings and demographic gender fields are omitted. Exact UTF-8 transcript hashes
and text-presence counts are retained. Text absence does not prove spoken absence.
Source license notices and change disclosure apply; no endorsement is implied.

## Qwen and invented fixtures

Serena and Eric in Qwen recording identifiers denote pre-existing synthetic
CustomVoice presets, not known human speaker identities. Original Qwen asset
rights remain those of the already archived corpus and its separate rights record:
https://github.com/jiying2007/kws-data/blob/617b66bb30148a675805be2f17c61703d31e6e79/docs/RIGHTS.md
No Qwen recording or third-party model is duplicated here. The six witness tables
are authored mathematical fixtures, not extracted voices. New offline validation
code is authored for the Apache-2.0 source repository; its repository license does
not grant a new blanket license to source recordings or third-party assets.

## What the privacy review establishes

All nested fields and strings in the allowlisted raw JSONL were inspected.
Natural-source logs contain six logits per selected row, event/state fields,
integer geometry, source hashes/IDs and timing/resource counters. Fields named
wave_samples or waveform_samples contain scalar counts, never samples. Qwen
reconstruction stores six probability bit patterns, token/frame metadata, CTC
prefix masses and node indices. Witness rows contain invented probabilities.
Ephemeral process-local numeric pid values have no accompanying host or boot ID.

No direct person identity, source waveform or frontend feature vector was found
in the admitted fields. That is not a proof of anonymization, non-invertibility,
or speaker privacy. Hashes and IDs intentionally identify public source recordings;
their human speakers are unknown in these records. Low dimensionality alone is
not a privacy guarantee. No speech decoder, vocoder or inversion was attempted.
Natural recordings remain excluded pending a separate explicit public-file and
source-license review. Earlier synthetic-audio publication scope is not reused
as permission for natural recordings. No general commercial-rights claim is made.
