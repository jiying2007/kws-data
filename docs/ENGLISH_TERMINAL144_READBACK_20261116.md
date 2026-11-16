# Terminal English144: actual observations and a stopped training gate

The four independent observers completed on the 143 surviving original WAVs.
The fixed training minimum did **not** pass: train-role Vosen has five eligible
weak-intent rows, below eight. No feature extraction, optimizer updates, new
keyword weights or head parity were performed by this follow-up. The new
training configuration remains explicitly disabled. Its success-path source
is retained so that the exact boundary can be reviewed; source tests are not
a completed English model.

The owner-confirmed targets remain Hey Mero `/heɪ ˈmiːroʊ/` and Hey Vosen
`/heɪ ˈvoʊzən/`. Nothing here broadens the phonetic target, promotes synthetic
presets to real speakers, or grants acoustic gold or formal training admission.

## Original failure and separate observer identity

[Original source run 43816869386](https://github.com/jiying2007/kws-data/actions/runs/43816869386)
at `8286ee743a2690d168c3e56f5b31ccc64ae490ed` completed with failure. It made
144 first attempts and retained 143 valid WAVs, with 2,743,040 unique PCM16
sample positions / 171.44 seconds. Cell
`enfinal144-ae8bc9e78999b00ba2eef8ff` failed with `ValueError`; its original
receipt does not contain enough information to infer a more specific cause.
The original observer and training jobs were skipped. Those jobs, the nine
partition artifacts and the original all-nine-complete gate are unchanged.

[Separate observer run 43822597212](https://github.com/jiying2007/kws-data/actions/runs/43822597212)
at `21d807461c0afa30037362ddde7ea527d87cf3e6` completed successfully on attempt
one, with all seven actual jobs successful. Qwen3-ASR, Whisper, Parakeet and
XLSR each made 143 attempts and saved 143 completed observations. There were
429 ASR calls and 143 phone-recognizer forwards, no new TTS and no optimizer
updates. The combined evidence retains all 144 original dispositions,
including the failed source. The observers are machine measurements, not
independent human confirmation.

Original source, observer and any future training execution have distinct
identities. The source ledger canonical SHA-256 is
`b6e157eea8eeea38f9261d8756ab43d6570c35a6ef53b892436b7c97424efb38`.
The observer terminal evidence canonical SHA-256 is
`73e2b17227171ecc693d8bfb62b7c1f4b5d7a7f832e28ea4404341dc9b85a63c`.

## Exact frozen selection result

Selection policy
`27a18716b5c0e7f29a937eeedde085917c04d4d45f30db0d9115dc3c881a9fd5`
is unchanged. It requires two of three exact normalized lexical observations
and the same predeclared contiguous XLSR phone core. In particular, Vosen
`z ɪ n` does not substitute for `z ə n`. Unknown wake-intent rows remain
quarantined and never become negative examples.

| Role | Requested intent | Eligible rows | Source families | Minimum rows / families | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Train | Ordinary sentences | 32 | 4 | 8 / 1 | Pass |
| Train | Hey Vosen | 5 | 2 | 8 / 2 | **Fail** |
| Train | Hey Mero | 10 | 3 | 8 / 2 | Pass |
| Development | Ordinary sentences | 16 | 2 | 2 / 1 | Pass |
| Development | Hey Vosen | 3 | 1 | 2 / 1 | Pass |
| Development | Hey Mero | 8 | 2 | 2 / 1 | Pass |

There are 74 eligible requested-intent research rows and 70 unknown rows in
the full 144-row ledger. These counts follow source/PCM deduplication. The
second check after actual feature-window geometry has not run. The selected
decision object canonical SHA-256 is
`f51849b945cd49c4a726ce79a3d22b656408c9088532b6376d829a10c2baf6b2`.
The saved-only audit returned exit code 2 and zero model calls. It did not
forge GitHub environment variables or relabel the incomplete source campaign.

The machine-readable [readback summary](research-v4/en-final144-terminal-readback-20261116.json)
binds all five exact observer ZIP artifacts, native API evidence, the saved
selection/minimum/exclusion documents and their raw byte hashes. The original
audit input and its result remain unchanged. The separately frozen disabled
training pins add the measured source/selection identities; they do not
rewrite the audit's own configuration hash.

## Retained successor source, disabled execution

`automation/terminal144_train_once.py` verifies original nine ZIPs, native
source API files, the separate successful observer's four ZIPs and combined
ZIP, all per-cell receipts, all model-lock identities and the unchanged
source split. It reconstructs the join and selection from saved bytes before
any model import. The default command performs no work; `--audit-selection`
uses only standard-library source and evidence readback. `--execute` requires
an explicit enabled configuration, exact worker SHA, a new first public-main
execution and both frozen minimum checks.

The disabled workflow separates source tests, saved-input preflight and the
one prospective 96-update experiment. A failed audit or feature minimum
prevents optimization and package creation. There is no TTS/ASR stage, retry,
threshold search, seed sweep or quota-filling operation. Successful packaging
would require the exact saved NPZ, native/ARM head parity and three distinct
source/observer/training identities. None exists for this new head yet.

`terminal144_package.py` and the disabled
`verify_terminal_frontend_once.py` preserve that requirement. The latter binds
the actual successful static C export to the actual future training artifact;
it cannot substitute test tensors for the missing trained keyword head. Its
full-head numerical plan and tolerances remain fixed. Its training artifact,
weight and result pins are null and execution is disabled.

## Deployment work and completion boundary

[Pipeline static export 43823297841](https://github.com/jiying2007/kws-pipeline/actions/runs/43823297841)
at `cfc69dcd7535ea909403f10a91bcfe26ccc1b331` generated the fixed backbone C
source successfully with zero inference. The exact export ZIP is artifact
`14205188962`, 2,722,450 bytes, SHA-256
`1ffc968da201e07186929f05d9d8c651e18390a92f5436b6cdfec3a3cb1e9ac0`.
This retains the preceding failed export rather than replacing it. Its graph
manifest records mel workspace 299,184 bytes and embedding workspace 454,656
bytes. Those are graph scratch-buffer sizes, not complete process or SSC305
RAM measurements. Static generation alone does not establish numerical parity.

The present source/observation result does not authorize automatic replacement
samples or a lower minimum. Existing recordings can be reviewed without new
recognition calls; any independent pronunciation evidence must retain its own
reviewer/source identity and cannot silently change this consumed experiment.
Future inputs need an explicit source/role/variant protocol. The current
candidate has no new trained head and therefore no current-target L1 delivery.
Independent per-word FRR, combined continuous FAR, word-end callback latency
and physical SSC305 validation remain `NOT_RUN`; the V4 targets are unchanged.
