# Persistent V4 research executor

This executor accepts three explicit, bounded English63 research task types after
a reviewed, **new versioned request** reaches public `main`. The original online
weak-intent task produces GRU and DS-TCN weights and reloads them. The separate
saved-weight native C comparison consumes the existing successful checkpoints
with zero optimizer updates; its protocol is described below.
The third task measures new C discontinuity and state-isolation contracts using
the retained native header and original PCM, with no Python model forward.
PRs verify source and request boundaries only. They do not train, generate
audio, call ASR or download the original audio.

## Submit an execution

1. Review and merge the model source in `jiying2007/kws-pipeline` first. The
   only entrypoint is `research/english63_online_pilot/online_pilot.py`.
2. Copy `.github/research-v4/request.template.json` into
   `.github/research-v4/requests/<request_id>.json`. Set a new descriptive
   request ID, execution reason, and the reviewed **40-character commit SHA**.
   The placeholder is intentionally invalid. Repository, entrypoint, input
   artifacts and budgets cannot be overridden by a request.
3. Add that one request through the existing PR checks. Keep consumed requests
   unchanged. The main merge triggers `research-v4-executor.yml`; changing
   executor code or its documentation alone never launches training.

There is no schedule, patrol, workflow dispatch, TTS generation or external
agent task. The repository remains public. Execution uses this repository's
Actions token to read this repository's original artifacts; the pipeline
repository's token is not assumed to have cross-repository artifact access.
No Codex or ChatGPT Work compute is requested by this workflow.

## Fixed inputs and budget

| Item | Fixed value |
| --- | --- |
| Original audio | Artifact `13954744587`, source run `43242278962`; 63 original mono 16 kHz clips |
| Joined observations | Artifact `14001747172`, member `english63-three-evidence-review.json` |
| Original audio ZIP SHA256 | `f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc` |
| Review ZIP SHA256 | `f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8` |
| Review JSON SHA256 | `89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12` |
| Candidates | GRU and DS-TCN; 48 optimizer updates each; seeds fixed by the reviewed source |
| Roles | 30 weak-intent training rows; 12 exposed development rows; 21 quarantined nearphone rows |
| Model wall time | At most 1,200 seconds for both candidates and the driver's first replay |
| Independent saved-weight replay | At most 300 seconds; zero optimizer updates |
| GitHub job | Ubuntu 24.04, CPU, one compute thread, at most 45 minutes including setup |
| Runtime | Python 3.12.16, NumPy 2.3.5, PyTorch 2.10.0+cpu; actual runtime also retained |
| Output/archive size | At most 64 MiB; terminal artifacts retained 90 days |

This is a **new** experiment environment. It does not claim byte-identical
training reproduction of the earlier Python 3.13.5 pilot. The explicit saved
weight replay checks the new experiment in its own recorded environment.
Expired/missing original inputs cause a failure; audio and ASR are not
regenerated. The joined ASR, IPA and acoustic observations remain observations,
not human-authenticated pronunciation labels.

## Consumed requests and durable readback

The execution key is the SHA256 of the exact request bytes. A concurrency group
serializes this key. Before any model work, the workflow retains
`research-v4-claim-<request_sha256>` as an Actions artifact. The terminal artifact
is named `research-v4-result-<request_sha256>`.

On a rerun the executor first checks these durable artifacts. A prior success
is downloaded and verified against its exact request, pipeline commit, complete
file hash manifest, both actual NPZ checkpoints, the 48-update receipts and
successful no-training replay. It is reused without downloading the inputs or
executing the models. A prior failed terminal, a claim without a valid terminal,
an expired receipt or an ambiguous artifact set stops execution. It cannot be
interpreted as permission to repeat training. `run_attempt == 1` is an additional
fallback against missing/deleted receipts, **not** the idempotency mechanism.

Failures after a claim retain the terminal, logs, `FAILED.json` and any partial
weights that exist. They remain failures. If the runner is killed before its
terminal can be uploaded, the earlier claim still prevents automatic replay.
Recovery requires a reviewed new request with the failure diagnosis and changed
execution reason; the consumed request is never edited. Requests with new IDs
are explicit new execution authorizations, so they should not be added merely
to repeat an unchanged successful experiment.

The Actions download step alone receives `GH_TOKEN`. Authentication is removed
before following the signed blob download URL. Checkout credentials are not
persisted. The model and replay child processes receive a small environment
allowlist with offline flags and one-thread settings; GitHub, artifact and
provider credentials are absent.

## Result contents and scientific boundary

The terminal artifact retains the exact `REQUEST.json`, `TERMINAL.json`,
`EXECUTION.json`, `runtime.txt`, process logs, word-level `keyword-evidence/`,
and the model's `SUMMARY.json`, `REPLAY.json`, `events.jsonl`,
`gru-weights.npz` and `dsc_tcn-weights.npz`. It does not duplicate the original
WAVs or input archives. The word-level report reuses the 189 existing ASR
observations and creates 126 keyword rows; it runs no additional recognizer.

Success means the bounded experiment executed and its saved predictions were
reproduced. Zero proposals is a valid diagnostic outcome. It is not evidence
of independent English FRR, continuous-negative FAR/hour, word-end latency or
SSC305 performance; all four remain `NOT_RUN`. Formal training admission,
human gold and model promotion remain false. The exposed 12 rows never enter
optimization, and the 21 ambiguous nearphone rows never become trusted negative
training labels. Finite clip and diagnostic continuous-sequence results do not
constitute an independent real-world continuous-stream qualification.

After downloading and unpacking a successful terminal artifact, its checkpoint
manifest can be verified without models or tokens. Put the artifact under a
new absolute directory's `result/` subdirectory, retain the original named
request file, then run:

```sh
python3 -I -S -B automation/research_v4_executor.py check \
  --request .github/research-v4/requests/<request_id>.json \
  --state /absolute/downloaded-run
```

For a separately requested no-training numerical replay, reuse the exact
original inputs, checked-out pipeline commit and retained runtime:

```sh
python -I -B research/english63_online_pilot/online_pilot.py \
  --tts-zip /absolute/original-tts.zip \
  --review-json /absolute/english63-three-evidence-review.json \
  --verify-result-dir /absolute/downloaded-run/result/model
```

No local project-model execution is needed for source verification. The
executor tests use invented receipts/checkpoint bytes and mocked APIs; the
pipeline owns actual numeric/model tests. Register new files in the existing
branch-retention inventory as integration files when integrating this change.

## Independent task: saved-weight native C comparison

`english63-saved-native-c-parity` uses the same persistent workflow and immutable
request directory. Its separate consumer is `automation/research_v4_native_c.py`.
The existing `en63-online-20261114-v1.json` request remains unchanged. There is
no call to the online training entrypoint in this task.

Start from `.github/research-v4/native-c-request.template.json`, pin the reviewed
pipeline commit containing `research/english63_online_c/c_replay.py`, and add
one new named request through the normal PR checks. A code/documentation update
without a new request does not perform model computation. The only execution
entrypoint is:

```sh
python -I -B research/english63_online_c/c_replay.py \
  --tts-zip /absolute/original-tts.zip \
  --review-json /absolute/english63-three-evidence-review.json \
  --trained-result-dir /absolute/original-successful-artifact-root \
  --output-dir /absolute/new-comparison
```

The original successful checkpoint is
[run 43595338774](https://github.com/jiying2007/kws-data/actions/runs/43595338774),
artifact `14114695344`, from data commit
`14547faa42299ae9d819ba4bff9824c2471c412e`. Its exact byte identities are fixed:

| Original checkpoint component | SHA256 |
| --- | --- |
| Complete artifact ZIP | `bc76805934df8eb158fa4259a75c26a4ea4276c15b83542acad5948a1f954b54` |
| Original consumed request | `ffa74269a0537115462da17b5050462e867cdb9980690105d2a92a4bdc0b7e53` |
| Successful terminal | `7c840f2877c0d80a147b5d1287261faeb01495e61f8a2d8ab8cf9053c6cc91a4` |
| GRU NPZ | `71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623` |
| DS-TCN NPZ | `167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935` |

Before compiling or forwarding a model, the consumer verifies that the original
Actions run succeeded, the artifact belongs to that run, the ZIP digest matches,
and the **whole original terminal/checkpoint/replay manifest** passes the
existing online validator. It checks the fixed NPZ identities again after the
comparison and evidence verification. A missing or changed checkpoint never
causes retraining or re-creation of the original WAVs.

This task runs on one CPU thread with the same pinned Python/NumPy/Torch
versions. C compilation and reference/C comparison have a shared 1,200-second
wall limit. A second `--verify-result-dir` call has a 300-second limit and uses
only saved evidence: zero optimizer updates and zero model forward calls.
The full Actions job is capped at 40 minutes including setup. Timeout handling
kills the execution process group, including compiler/native descendants.

The comparison covers both existing models, all 63 isolated clips and one
continuous concatenation of the same source PCM, with fixed `160` and `mixed`
chunk plans. It requires 13,010 isolated frames and 13,134 continuous frames,
64 reference sequences per model, feature/logit/probability absolute errors
at most `0.0002`, identical chunk probabilities/events, and identical Python/C
events. EOF padding, EOF-only events and continuous-state resets must all be
zero. These checks establish equivalence for these retained inputs. They do
not establish independent real-world continuous-stream performance.

The 17 comparison files are `SUMMARY.json`, `REPLAY.json`, `BUILD.json`,
`RESOURCES.json`, the Linux `en63_replay` executable, two reference NPZ files,
eight aggregate binary traces, and `export/EXPORT.json` plus
`export/en63_generated.h`. Original PCM/WAVs and copies of the trained NPZ
weights are not included. All files are covered by the evidence manifest.
The consumer also retains its request, execution receipt, runtime and logs,
within the existing 24-file/64-MiB accepted-result limits.

Native C claims and terminal artifacts use independent names:
`research-v4-native-c-claim-<request_sha256>` and
`research-v4-native-c-result-<request_sha256>`. A successful terminal is verified
and reused without input downloads, compilation or model calls. A previous
failure, an incomplete claim or expired receipt blocks automatic repetition.
Failures retain available comparison evidence and logs. Any necessary correction
uses a reviewed new versioned request, preserving the earlier terminal.

No model or threshold is selected. Formal training admission, human gold and
model promotion remain false; English FRR, continuous-negative FAR/hour,
word-end latency and SSC305 profile remain `NOT_RUN`. Recorded Linux native
resources are host observations. The original synthetic source and exposed
development limitations continue to apply.

## Native C discontinuity and state contracts

`english63-native-c-state-contracts` addresses a different question from the
consumed parity experiment: after a dropped block, can the C stream clear its
acoustic, encoder and decoder history while preserving the session's supplied
sample clock? A full `en63_reset` still starts a new session at zero. A separate
`en63_discontinuity` starts a new acoustic segment, repeats the 50-frame warm-up,
and preserves cumulative supplied samples, produced frames and emitted events.
Missing capture time is not known and is never filled with silence or invented
samples. This research clock is not a hardware capture clock.

The fixed consumer is `automation/research_v4_native_c_state.py`, dispatched by
the same V4 executor and workflow. The only new pipeline entrypoint is
`research/english63_online_c/state_replay.py`. The request template is
`.github/research-v4/native-c-state-request.template.json`; its placeholder source
SHA cannot execute. Existing request bytes and claim names remain unchanged.

Inputs are the original two public audio/review artifacts above and the exact
successful native artifact `14117638750` from run `43605922752`, whose SHA256 is
`e85d14c87727f93b6e33edc196af00a6c9ce023d9e7d41d844109617fa5fb397`.
The baseline request, terminal, source commit, full evidence manifest and C
header are checked before reuse. Its exported weights are copied unchanged;
there is no weight export, optimizer update or Python reference-model forward.

Each of GRU and DS-TCN has ten fixed cases: saved baseline prefix, midstream
reset, reset after EOF, two interleaved instances, discontinuity at samples
399/400/32401, EOF without padding, invalid-call atomicity and invalid-model
atomicity. The source inventory remains 63 clips and 2,101,760 samples. The test
uses the first **128,246 unique existing samples**, divided into two 64,123-sample
streams. These are controlled state regressions, not a new 63-clip acoustic
evaluation. Repeated suffixes and resets add **zero independent audio exposure**.
The saved no-discontinuity prefix must remain byte-identical to its prior C
trace; any failure is retained rather than accepted with a looser threshold.

The new task uses Python 3.12.16 standard library and GCC on public Ubuntu 24.04
Actions. It does not install NumPy or PyTorch. Execution is capped at 600 seconds,
followed by a separate 180-second saved-evidence verification that has zero model
calls. Requests cannot override the source entrypoint, original inputs, model
roster, budget or scientific boundary. The child environment is the same
credential-free, offline, single-thread allowlist as the existing executor.

The workflow uploads `research-v4-c-state-claim-<request_sha256>` before executing
the native models and retains `research-v4-c-state-result-<request_sha256>` for
90 days. A successful saved terminal is reusable without model execution. A
failed terminal, orphan claim, ambiguous result or missing receipt on an Actions
rerun blocks repetition. Results retain actual case observations, raw numeric
traces, reviewed source snapshot, build/runtime identities, summary and complete
hash manifest; failed partial outputs remain failures. All successful output
files still fit the existing 24-file and 64-MiB outer budget.

Source PR checks use invented JSON and analytic fixtures only. The new state
contract proves neither target pronunciation nor independent FRR, continuous
negative FAR/hour, true word-end latency or SSC305 resources. These remain
`NOT_RUN`; model selection, formal admission and promotion remain false.

## Fixed paired C pool-access profile

`english63-native-c-pool-profile` measures one implementation variable:
`EN63_POOL_TIME_MAJOR=0/1` changes traversal of the 200-slot, 32-channel sliding
pool while preserving each channel's FP32 addition and maximum order. Its fixed
consumer is `automation/research_v4_native_c_pool.py`; the only pipeline entry is
`research/english63_online_c/pool_profile.py`. The same existing workflow handles
source checks and the public main request. The template
`.github/research-v4/native-c-pool-request.template.json` cannot execute until its
placeholder commit is replaced by the reviewed full source SHA. The sole request
ID is `en63-native-c-pool-20261115-v1`; this lane cannot create another campaign by
renaming a request. Existing requests, consumers and claims remain unchanged.

Inputs are the original TTS/review artifacts and the successful state result
artifact `14141171320` from run `43663731129`, attempt 1. The 1,597,111-byte state
ZIP has SHA256
`2fdd0c951d6cf964d6b73afff5726e7d82b365dae539876822e513d7af0ccaa7`.
Its request, data/source commits, terminal, file manifest and original C header
are checked without executing its binaries. The new paired baseline and
candidate are both compiled on the same current runner from one reviewed source
commit and the same retained header, driver and flags; the macro value is the
only algorithm variable. Historical timing or RSS is not the new baseline.

Each model has eight pairs: one trace pair, one warm-up pair and six timed pairs.
Timed order alternates baseline/candidate and candidate/baseline, three of each.
Each full pass supplies the same 2,101,760 original PCM samples in 13,136 calls of
160 samples, producing 13,134 frames. Both models together have exactly 32 full
passes, 420,288 frames and 67,256,320 supplied samples including repeats. The
unique original sample positions remain 2,101,760; repeated profiling adds zero
independent audio exposure. The original 30 train-role, 12 exposed-development
and 21 quarantine identities are not promoted to labels or optimizer inputs.

The profile uses the lowest allowed single CPU, an overall 180-second native
process-group budget and a 30-second budget for each child. The outer execution
and saved-only verification remain bounded at 600 and 180 seconds. It uses only
Python 3.12.16 standard library and GCC, with no NumPy or PyTorch installation.
The child receives the existing single-thread environment allowlist and five
public Actions identity fields, never download or checkout credentials.

Four complete traces must exactly reproduce the retained original continuous
trace hashes, including candidate versus baseline. Each trace has magic
`EN63TRC1`, version 1, 168-byte records and 2,206,528 total bytes. Resource timing
is recorded separately from trace writing. The fixed benefit rule for each
model requires at least 5% median paired CPU reduction, at least five positive
pairs out of six, a median paired p99 ratio no greater than 1.10, exact numerical
equality and no increase beyond 38,192 bytes per state. An otherwise valid
`NO_BENEFIT` result is still a successful experiment; no extra pairs or changed
thresholds may be used to turn it into a benefit result. It does not select or
promote a model, and it does not measure independent FRR, natural FAR/hour,
acoustic word-end latency or SSC305 resources.

The independent artifacts are `research-v4-c-pool-claim-<request_sha256>` and
`research-v4-c-pool-result-<request_sha256>`. The claim is uploaded before native
execution. A prior successful terminal is verified without model execution;
failed, incomplete, ambiguous or expired receipts do not authorize repetition.
The result's `pool-profile` directory has exactly 15 fixed files, within the
existing 24-file and 64-MiB outer budget. Original raw process measurements,
source/build/input identities, native binaries, traces and saved verification
are retained; the saved verifier has zero model calls. Partial failures keep
their failure terminal and available evidence. Source PR tests use only
invented JSON/binary fixtures and mocked execution.
