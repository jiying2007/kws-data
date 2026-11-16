# English Vosen schwa16 and the separate snapshot160 experiment

This protocol declares a new, bounded pronunciation-instruction hypothesis. It
does not retry the original final144 campaign or regenerate an existing request.
The original 144 declarations, 143 valid recordings, failed cell, source and
observer receipts, fixed selection rule and disabled training entry remain
unchanged. There are no results from this new experiment in this document yet.

## The new data hypothesis

The current targets remain **Hey Mero `/heɪ ˈmiːroʊ/`** and **Hey Vosen
`/heɪ ˈvoʊzən/`**. The saved original selection has five eligible train-role
Vosen weak-intent rows, below the unchanged minimum of eight. Aiden accounts
for three and Eric for two. The five other Aiden observations share a specific
machine-observed `/ɪ/` rather than `/ə/` difference; the other original failures
have additional causes. This hypothesis does not relabel any of them.

The new recipe uses Aiden and Eric, each with the eight existing delivery styles,
for **16 new requests**. Both voices remain train-role members of their original
`qwen3-preset-*` families. The canonical phrase, two-word synthesis text and
American target are unchanged. The phonetic instruction changes substantially:
it relates Vosen to American-English *frozen*, replacing initial `/fr/` with
`/v/` while retaining the unstressed `/zən/` ending. The recipe binds the exact
instruction bytes. A new cell ID, seed or policy name alone is not a new request;
the actual model/voice/text/instruction/decoding request is checked against the
retained original 144 requests.

This is a small representative hypothesis test under the V4 data-expansion
rules, not a declaration that the 128–256-item production stage has passed.
Requested spelling, an instruction, rhyme or machine agreement is not acoustic
gold. This experiment remains **requested-intent weak research**. It neither
changes the raw-phoneme requirements for pronunciation-specific positives nor
establishes Hey, vowel duration, stress or the owner's full pronunciation target.
All formal admission, human-gold, trusted-negative, pronunciation-verification
and model-promotion flags remain false.

## One new generation and first observation chain

`.github/workflows/english-vosen-schwa16-once.yml` has a source-only PR job and
one actual chain. The actual chain is limited to a public `jiying2007/kws-data`
`push` to `main`, attempt 1, whose exact predecessor is
`393b03e41eff68ce91a1943b18b3da9641378515`. Only a change to
`automation/english_vosen_schwa16_pins.json` triggers that main-push workflow.
The Python guard separately checks the enabled pins, event, code and recipe
identity before runtime installation or model work. There is no schedule,
manual dispatch, retry loop or continuation into an old experiment.

The source recipe and worker are `automation/vosen_schwa16_recipe.py` and
`automation/vosen_schwa16_once.py`; the source tests are
`automation/test_vosen_schwa16.py`. Generation invokes the existing bounded
capture helper once with a maximum of 16 attempts. Every attempted cell has a
started receipt before its model call. Native audio, the original derived WAV,
PCM identity, completed or failed receipts and the complete 16-row source
ledger are retained. A timeout or failed cell is not retried or replaced.

After the generation job has reached a terminal state, a pure source readback
may admit only its valid new recordings to their **first** four observations.
The source denominator remains 16 even if there are fewer valid recordings.
Failed, unknown, not-started or invalid rows remain explicit exclusions. A
partial generation result remains `INCOMPLETE_NO_RETRY`; allowing observations
of its valid rows does not report the generation job as successful.

Let **N** be the verified new valid-source count, with `0 <= N <= 16`. Qwen ASR,
Whisper, Parakeet and the same pinned XLSR phone observer each see only those
N new WAVs, once per provider. The maximum is 16 TTS calls, 48 ASR calls and
16 phone forwards; actual counts and unknown-after-start outcomes are saved.
Observers receive audio, not the requested phrase or target IPA. No old audio
is synthesized, re-observed or passed through another model by this chain.
The combined ledger preserves all 16 declarations and every provider's actual
completion state. Source, observer and runtime artifacts are uploaded even
after a partial failure. This workflow has **no training job**.

## Separate snapshot and fixed training continuation

The prospective combination is **160 declarations = original 144 + new 16**.
Its verified-source count is **143 + N**, before global duplicate quarantine;
the valid-source count is never substituted for the declared denominator.
The original failed campaign and separate original observer retain their
identities. New generation and observation stages retain their own stage/job
identities, including when they share one new Actions run. Any actual training
has a separate execution identity.

The combination reads the original nine source archives, four original
observer archives and saved join by exact bytes. It verifies the new source
and observer artifacts separately. It never treats the old disabled audit
directory as an actual training result. The fixed row order is the old recipe
followed by the new recipe. Global duplicate-PCM quarantine covers **all 143
original valid sources and all N new valid sources**, not only previously
selected rows. Batch, instruction and style do not create new source families.

The selection policy remains
`27a18716b5c0e7f29a937eeedde085917c04d4d45f30db0d9115dc3c881a9fd5`:
two of three exact normalized lexical observations and the same contiguous
phone core. Each candidate row must first have all four completed observations
of its exact PCM. A partial observer or combined job remains a recorded failure;
its other fully observed rows may be considered under this newly declared
snapshot protocol. Missing or failed observations do not count as votes or
negative evidence. For Vosen, `z ɪ n` is not substituted for `z ə n`. Unknown or
conflicting wake-intent rows remain excluded and do not become negatives.
The new small batch need not contain all six role/class groups; the combined
training view must satisfy them.

The reserved paths for the separate training follow-up are
`.github/workflows/english-snapshot160-train-once.yml` and
`automation/english_snapshot160_training_pins.json`. Its initial configuration
must be **disabled**, with no next-main predecessor or invented new artifact and
snapshot hashes. These files may follow the source-only protocol PR; they are
not an execution stage of the source workflow. A later, separately pinned first-main execution may proceed
only after the actual source/observer evidence and exact prepared snapshot
have been frozen. Source-only checks do not enable it.

Both existing minimum checks remain mandatory: after global source/PCM
deduplication and again after actual feature-window extraction. Each train
keyword requires at least eight original sources and two families; each
development keyword requires two sources and one family. Ordinary-intent
minimums remain eight train and two development sources. New windows,
augmentation or repeated playback cannot increase these source counts.
**Three generated recordings do not guarantee either minimum will pass.**

The fixed training implementation remains the pinned pipeline source
`31238cafe3d6668874d446eae5dfa7c55028fb13`: the same two feature assets,
96-dimensional embeddings, 16-frame causal history, 32-unit hidden layer and
three classes. The experiment retains 96 optimizer updates, seed `20261116`,
train-only normalization, threshold `0.8`, confirmation count `3` and cooldown
`13` embeddings. There is no architecture, threshold, seed or checkpoint search.
The exposed, speaker-split LibriSpeech sources remain research background and
context. Real context is declared and byte-bound; silence is not appended to
manufacture detections. Actual feature support and the second minimum must be
measured, and the existing finite feature-call budget is retained.

Successful training would save actual embedding caches, actual NPZ and binary
weights, optimizer draws and input bindings, then replay freshly loaded saved
weights and verify native/ARM **head** parity. A package must retain both source
cohorts, their observations, all exclusions, both minimum reports, saved replay
and licenses. No such success or new trained weights are claimed here.

The prospective snapshot identity binds canonical semantic source, observation,
selection and minimum evidence, including the exact original ZIP identities.
Original API response bytes are retained separately and are never rewritten to
make a pin match. A closed file manifest binds the actual prepared handoff;
its measured hash is passed between the preparation and training jobs. Changing
API JSON formatting cannot silently change the semantic training cohort, and
no execution-time patch of the enabled pins is permitted.

## Completion boundary

This chain is intended to make a reproducible L1 weak-research model possible.
The existing backbone and mel-specialization C results are separate measured
engineering evidence. Neither those results nor head parity under QEMU proves
the complete keyword system on SSC305. Independent pronunciation confirmation,
independent continuous acoustic evaluation, full-chain evidence and actual
SSC305 audio-chain/resource/latency/stability confirmation remain distinct
conditions for the corresponding L2/L3 claims.
