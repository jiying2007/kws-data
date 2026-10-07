# Post-hoc target-local diagnostic v2

This change was designed after v1 showed zero known-bit coverage. It is a post-hoc diagnostic on reused outputs, not untouched-holdout qualification. This implementation task did not open actual ASR outputs, human annotations, or audio, and did not create revised actual rules or execution envelopes.

## Frozen source

- `baseline_v1.py` is an exact copy of both preserved v1 scorer files
- V1 SHA256: `2bd566983f9846e19c3ac24d59cffe155cc4b45c39c1d64a9f1fae7772af104a`
- V2 `calibrate.py` SHA256: `1bccdd66bca3cd67109df1eff63770b5d7e9ba3a9a10b578a665b7bbcc9575b4`
- Rule version: `posthoc-target-local-text-evidence-v2`
- Complete source diff: `scorer-v1-to-v2.diff`

## Decision policy

Global execution/status/completeness/quality/uncertainty-marker/control/decoder guards are unchanged and still override exact text. Exact contiguous literal target text remains the only positive rule. Tokenization, greedy phrase dictionaries, normalization and marker requirements are unchanged.

Utterance-level OOV and multiple-reading diagnostics are retained as descriptive `lexical_diagnostics`, but no longer become blockers for every absent target. All-OOV/zero-lexical-evidence text still abstains, including a single unsupported character.

All pre-existing text, dictionary-phonetic and boundary-separated candidate fields are retained unchanged. A separate target-local matcher evaluates n-1, n and n+1 meaningful-token spans with set-valued phonetic Levenshtein distance at most one. All dictionary readings participate; no preferred reading is selected. OOV is a possible-match wildcard, never a phonetic observation. Insertions and deletions each cost one. A candidate only causes abstention; no missing word is reconstructed.

Candidate records retain the original normalized-text substring, start/end offsets, token positions, every intervening boundary position, OOV positions and ambiguous-reading positions. Both hard and soft boundary crossings remain uncertainty-only evidence. Normalized text is never globally rewritten, punctuation is never joined into positive evidence, and wildcard matches never populate dictionary-tone candidates.

## Synthetic validation

- `v2-target-local-results.txt`: 19/19 new invented-fixture tests pass
- `v1-regression-results.txt`: 38/38 original v1 tests pass unchanged
- `v2-contract-regression-results.txt`: 38/38 original synthetic contract tests pass when run against v2, with temporary fixture rule version/hash updates and the one deliberately changed irrelevant-OOV/heteronym expectation

The v2 contract run imported the original synthetic `test_calibrate.py`, substituted the v2 module and CLI path, changed `setUp` to update only the temporary synthetic rule version and refresh its hashes, and overrode the single global-lexical-abstention test to expect negative for irrelevant OOV/heteronym text. All other 37 test methods, including CLI identity/hash validation and gold preservation, were inherited unchanged. Original fixture files were not changed.

New coverage includes irrelevant heteronyms, OOV near/far, all-reading use, soft/hard boundary provenance, split homophones, short partials, inserted syllables, literal matches with unrelated ambiguity, existing candidates, independent targets, all-OOV abstention, global execution/quality/marker guards, explicit post-hoc readout scope and unknown-gold preservation.

## Limits

The extension can conservatively increase abstention for known-reading partial/split homophones. Short targets and OOV-rich text remain particularly abstention-prone. No dictionary reading or text-completeness declaration proves acoustic presence/absence. Character-based phonetic assumptions remain limited for non-Chinese material. Byte/input/output guards are unchanged. The caller must preserve original execution-rule identity separately from this revised evaluation identity; v2 never retroactively redefines the original model execution contract.
