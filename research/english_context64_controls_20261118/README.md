# English exact-child context controls: observed stop

The first registered run completed on 2026-11-18 (Asia/Shanghai), but its data
admission method did not pass the preregistered controls. No KWS training or
model promotion followed this experiment.

- Source change: https://github.com/jiying2007/kws-data/pull/199
- Actual run: https://github.com/jiying2007/kws-data/actions/runs/44202856867
- Executed commit: `b64b575cff2f43f68ff0b8dade8ba00b7ee2fc3e`.
- Combined artifact: `14383037169`, 214589267 bytes, SHA256
  `2179c78aef1256dc11e94554ef6ad098b9c6ca0df351e207ad7a4398a9a861f0`.
- Exact saved evaluation SHA256:
  `d73cb08da7c50761b07e857f0edb83f937a6d23b7bc39f3b01d0cfbcaf521b93`.

## Actual results

All 48 TTS calls, 192 child ASR calls and 96 parent phone calls returned and
retained their outputs: 336/336, with no retry. All 64 child PCM slices were
verified against their parent bytes. These are execution counts, not measures
of KWS recall, false alarms or independent human pronunciation.

| Registered category | Count | Candidate positives | Corroborated controls |
| --- | ---: | ---: | ---: |
| Positive hypotheses | 16 | 7 | Not a control |
| Near-miss controls | 16 | 2 | 8 |
| Ordinary controls | 16 | 0 | 16 |
| Target-outside-child controls | 16 | 0 | 0 |

Positive support by declared hypothesis was Mero /ɛ/: 0/4; Mero /i/: 3/4;
Vosen /s/: 2/4; Vosen /z/: 2/4. These small synthetic counts do not select an
authoritative American pronunciation.

The two intended near-miss controls that became candidates were requested as
Vozin. Both phone observers reported a schwa-ending form and all three child
ASRs returned target-compatible spellings. This does not establish two errors
on acoustically verified negative examples: the generated contrast itself was
not corroborated. The target-outside-child crop also retained the beginning of
the following Please sentence; Parakeet included please in all 16 transcripts.
The original exact-text/control decisions remain failed.

`RESULT.json` preserves the worker's original result. `OUTCOME.json` preserves
all 64 row decisions, raw child texts, fixed-region phone summaries, original
denominators and failed gates. Full PCM, logits, frame IDs and native execution
receipts are in the combined artifact. Pure saved-byte independent audit
reproduced all 64 decisions and the stop without any model or resampler calls.

## Consequence for training

All probe parents and children remain permanently excluded from training.
No old labels or STOP decisions were changed, no new corpus was automatically
expanded, and the existing coverage model and 0.8 / 3 / 13 event rule remain
the current research baseline.

The accompanying typed-source intake supports explicit preset identities and
known lineage instead of interpreting every new source as a numeric LibriSpeech
speaker. It rejects heldout training weights, UNKNOWN-as-background and all
consumed context probes before numerical imports. Its numerical entry remains
disabled and unbound. The original training algorithm and ten quality gates
are unchanged.

Future execution still needs a separately authenticated eligible corpus and
an explicit admission policy, including trustworthy same-preset background
supervision. Parent-context evidence cannot silently use a child-independent
label tier. This source change does not claim a new trained English model or a
passing deployment-quality result.
