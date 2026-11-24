# B21 TRAIN mechanism and paired implementation

This directory preserves a saved-data diagnostic and a prospective finite draw
coverage receipt. It contains no new waveform generation, ASR, KWS forward, optimizer
update, candidate quality result or deployment result.

## Proven TRAIN mechanism

The original B17 training archive (run 45319580193, artifact 14887048023, SHA256
`1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d`)
contains 528 authenticated preoptimizer B0/B9 TRAIN arrays. Restricting to the
same 120 previously admitted positive teacher paths yields 14 original/gain
conditions with a B9 whole-word event and no B0 event. In 13 conditions across
10 distinct source groups, 37 B9-token-span frames have B0 target logits above
both other nonblank tokens but below blank. The current nonblank-only objective
has no direct penalty for that blank competition. This supports a new TRAIN
mechanism experiment; it does not establish the cause of any prior DEV result.

## Shared source/prefix correction

B17 used `(step // 35 + step % 3) % 3` for prefix and `step % 35` for synthetic
group. For `step = group + 35 * epoch`, the prefix is always `group % 3`.
Its saved original draw confirms one prefix per synthetic source.

The prospective B21 draw fixes coverage in both arms. Five full 35-group passes
are followed by 17 groups ordered by source-group width, then lexical identity.
All nine two-original positive groups receive a sixth exposure. Within a group,
originals rotate by visit modulo original count and the prefix rotates by
`(group_index + visit // group_width) % 3`. No score enters this ordering.

The actual authenticated TRAIN row metadata produces all 237 synthetic
original/prefix pairs and all 27 natural original/prefix pairs in exactly 192
steps. Every one of the 79 synthetic and nine owner-reviewed natural originals
receives all three prefixes. Original natural selection remains unchanged.
This intentional common repair means B21 does not claim a numerical replay of
B17, even for its control arm.

## Paired scientific contrast

Both arms cold-start the same frozen B0 tensors, retain the ordinary B0 ceiling
at 0.08, and retain the same 120 B9 positive original/gain/frame paths at 0.08.
The control competes with the other two nonblank tokens. The challenger competes
with all three other classes, including blank. Sources, features, draws, losses,
optimizer and coefficients are otherwise identical between arms.

The independent B21 assessment and training design are finalized and byte-pinned
in the trainer. Training remains closed until the reviewed public first-attempt
Actions branch passes its exact predecessor guard. No previous frozen contract is changed.
Candidate C11/ARM engineering validation remains subject to independent quality
gates. B0 stays installed until those gates actually pass.

## Verification performed locally

- Eight offline protocol/arithmetic/actual-row-coverage tests passed.
- Four hand-constructed-tensor numerical tests are mandatory with `--numeric`
  in the pinned Actions job; they were not run in this environment.
- The original 528 TRAIN arrays were authenticated and inspected using NumPy.
- No Torch module was imported by the saved-data diagnostic or coverage check.
- No new KWS inference, optimizer, TTS or ASR operation was performed locally.

The trainer imports reviewed historical source/protocol helpers from B17/B15;
its producer closure must bind those dependencies and every literal data file,
including `english_v6_b21_competing_margin_design.json`, the independent
`english_v6_b21_panels/assessment.json`, and the metadata-only test fixture
`english_v6_b21_train_roster.json`.
