# B27 saved TRAIN mechanism readback

This adapter implements only the already registered TRAIN mechanism diagnostic.
It reads the 1,844 saved optimization outputs and complete traces from the
actual B27 TRAIN archive. It loads no checkpoint, performs no model forward or
backward, and changes no coefficient, sample, checkpoint or promotion gate.

The original B26 proposal already pinned the diagnostic source SHA-256
`a4c7eca1f2643c94cb940bb5856944430b790842fa9f944d5194c8f6bf4dcc45`
and the public original B24 result SHA-256
`f4bd7e2e525764c25d9d38bd8d6c225e6ade52eeab70638505284ddbf37f3579`.
That exact recovered source is retained as `ORIGINAL_B24_METHOD.py.txt`, inert
text. Its original output already matches the public evidence byte for byte;
the old diagnostic is not rerun. The new adapter checks AST equality of all
seven reused statistical helpers and the complete per-observation loop.
Its code was prepared during B27 execution; the predictions, method source
identity and old comparison values predate that execution.

## Fixed definitions

Frame labels are the first argmax of each complete raw four-token saved array.
Consecutive identical labels collapse to runs; blank runs are removed. Every
consecutive `1, 2, 3` sequence is a complete event, including each occurrence
when an observation has multiple events. The true suffix starts at
`prefix_samples // 160 - 3`. The known silent prefix ends at
`prefix_samples // 160 - 8`.

A prefix shortcut means the first complete event's `hey` run ends inside that
known silent prefix. It is not defined by subtracting suffix events from full
events. Nonblank frames within known silence and frame-zero `hey` observations
are retained as the original additional diagnostic counts. Positive-retention
observations have no injected silent prefix and are reported separately.

| Registered prediction | Original B24 value | B27 support pattern |
| --- | ---: | --- |
| Complete natural ordinary events | 47 | Fewer than 47 |
| Natural ordinary observations bearing a complete event | 35 of 384 | Fewer than 35 |
| Genuine positive suffix-event observations | 202 of 384 | At least 202 |
| Known-silence nonblank frames and shortcut observations | 0 | Remain 0 |

The same six original step windows are retained for description. Saved-loss
reconstruction uses the original method with the already registered natural
coefficient `0.16`; all other arithmetic is unchanged. Its residuals describe
rounding consistency and introduce no new threshold or selection rule.

## Input and execution

First complete the independent original TRAIN readback. Its full report and
canonical original training lock must bind the actual four original artifacts,
full TRAIN provenance and complete Q1 failure. Supply the exact report SHA-256
obtained from that completed readback. This diagnostic checks those bindings
and revalidates every saved optimization array and trace using the frozen TRAIN
reader's saved-array helper. It does not repeat the full Q1 proof subprocess.

The inert method check needs only the standard library:

```sh
python3 -I -S -B research/english_v6_b27_training_20261125/train_only_diagnostics/analyze_saved_b27_train.py check
```

Once the actual complete TRAIN archive and independently verified report exist,
the arithmetic command uses NumPy to apply the original saved-array method:

```sh
python3 -I -B research/english_v6_b27_training_20261125/train_only_diagnostics/analyze_saved_b27_train.py analyze \
  --models "$B27_TRAIN_MODELS_ZIP" \
  --training-lock "$B27_TRAIN_ORIGINAL_LOCK" \
  --training-readback "$B27_TRAIN_READBACK" \
  --readback-sha256 "$B27_TRAIN_READBACK_SHA256" \
  --output "$B27_TRAIN_DIAGNOSTIC_NEW_ABSOLUTE_DIRECTORY"
```

The output is `READBACK.json` and its hash-bound per-observation `rows.json`.
The original archive, report and all prior evidence remain unchanged. No
PRIMARY or replica arrays enter these calculations.

These observations concern changing pre-update training weights on repeated
fixed draws. They do not estimate final-checkpoint accuracy or independently
isolate a causal effect against the historical B24 execution. Whether the
registered pattern is supported or falsified, independent source, PRIMARY,
replica, engineering and delivery requirements keep their existing meanings.
