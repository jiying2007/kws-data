# English KWS Dual-Track V7.0

Status: **INERT / DESIGN-AND-REGISTRATION ONLY**  
Date: 2026-11-26  
Targets: **Hey Vosen** /heɪ ˈvoʊ.sən/ and **Hey Mero** /heɪ ˈmɛroʊ/

## Decision

The English wake-word program now evaluates two separately trained, separately packaged single-keyword models in a controlled competition. This does not create a multi-keyword model. Each model detects only its own brand phrase. The other brand phrase is a mandatory cross-trigger negative for that model.

The shared engineering substrate remains PCM16 → FFT512/Hann400 → 40 Mel → streaming FSMN-CTC and the existing C11 runtime. Model parameters, optimizer state, checkpoints, thresholds, candidate IDs, training receipts, score reports and release packages are strictly independent.

## Brand pronunciation lock

| Candidate | Exact phrase | Required US English reading | Stress |
|---|---|---|---|
| VOSEN | Hey Vosen | /heɪ ˈvoʊ.sən/ | Vosen first syllable |
| MERO | Hey Mero | /heɪ ˈmɛroʊ/ | Mero first syllable |

These are project-defined proper-name readings. Do not substitute legacy readings, spelling-based guesses, or other brand pronunciations. The phrase, IPA and locale are pinned in the candidate manifest.

## Candidate isolation

Each candidate receives its own:
- positive data registry and pronunciation audit;
- negative-data registry, including the other wake phrase as a cross-trigger negative;
- source-group and speaker assignment;
- train/dev/calibration/locked-test manifests;
- model initialization identity, optimizer state and training run;
- checkpoint lineage, quantization/export artifacts and candidate hash;
- independent primary assessment, sealed replication and engineering proof;
- release decision and package.

The two candidates may reuse only explicitly approved common ordinary-negative recordings and public environmental/noise material. Reuse must preserve original source identity and cannot cause speaker/source-group leakage. Neither candidate's positive recordings may be relabeled as the other's positive set. Cross-trigger phrases must be scored as negatives.

## Data qualification

Both candidates use the same admission policy:
1. Preserve original audio bytes and SHA-256, source/license, speaker/source-group identity, synthesis recipe or recording provenance, and exact text.
2. Apply bounded multi-ASR review; preserve each engine's raw transcript and disagreement. ASR agreement is evidence, not automatic human ground truth for a proper name.
3. Independently review pronunciation, phoneme sequence, stress, syllable count, clipping, silence, SNR, duration and audio integrity.
4. Distinguish genuine human recordings, TTS and voice-cloned speech. Never describe synthetic voices as independent human speakers.
5. Quarantine ambiguous, empty, corrupt, duplicate, mispronounced or untraceable samples. A true empty response quarantines the entire source item; no crop salvage or retry.
6. Reuse already verified audio; do not regenerate verified utterances.
7. Keep locked-test audio inaccessible to training, mining, tuning and checkpoint selection.

Each candidate must independently satisfy the 6/2/2/2 source-group partition (train/dev/calibration/locked test). A source group or canonical speaker must not cross partitions. The two candidate registries must each declare the same partition policy and minimum evidence requirements. A shared negative pool is permitted only where the exact same original source identity is preserved and leakage checks pass.

## Training and fairness

Use the same frontend, model family, optimizer implementation, reproducibility controls, evaluation protocol and hardware constraints. Start each candidate from the same approved immutable initializer, but instantiate independent model objects, optimizer state, random streams and output directories. Never copy trained weights or optimizer state between candidates.

Do not copy B28's single-candidate 192-update / 1,847-forward budget and pretend it is a dual-candidate budget. V7 training budgets are intentionally **null and execution-disabled** until both candidate-specific designs have explicit resource estimates, approved limits, immutable data identities and independent contract review. This avoids silently doubling compute or exhausting a one-shot budget.

## Evaluation

Run identical evaluation methodology separately for each candidate:
- target recall / false reject rate;
- false accept rate on ordinary speech and environmental negatives;
- cross-trigger rate on the other brand phrase;
- performance by speaker, accent, speed, distance, SNR and room condition;
- detection latency and event timing;
- confidence calibration and threshold stability;
- CPU, RAM, Flash, model size and thermal/resource behavior on SSC305.

Use independent fixed-weight primary and sealed replication. A primary PASS may open only the same candidate's sealed replication. Never transfer a PASS, threshold, calibration set, source inventory or candidate hash from one track to the other.

No numeric promotion threshold is invented in this registration. The existing approved product gates must be explicitly bound to V7 before any locked evaluation. Any new threshold needs a documented product rationale and approval before test access.

## Competition decision

First apply all hard safety, quality, evidence and device gates. Only candidates that pass every required gate may compete. Compare recall at the approved FAR limit, cross-trigger rate, far-field robustness, latency and resource use. Do not use a weighted average to compensate for a failed hard gate.

Allowed outcomes:
- VOSEN_ONLY;
- MERO_ONLY;
- BOTH_INDEPENDENTLY_QUALIFIED (only if product requirements justify two detectors);
- NO_PROMOTION.

## Deployment

Share the C11 runtime and frontend implementation, but package two independently hashed model payloads:
- `hey_vosen.pack`
- `hey_mero.pack`

Each pack must carry candidate ID, model hash, frontend/runtime compatibility, label/IPA, threshold/config identity, training and evaluation proof references, and rollback metadata. Verify each package independently on SSC305. B0 remains the production rollback baseline until the selected candidate has passed real-device acceptance and an explicit release record exists.

## Migration from B28

B28 remains an immutable historical single-candidate experiment. Do not rewrite its objective, design, budgets, registries or evidence to retrofit V7. Its current TRAIN360 one-shot acquisition (run 45804250909) remains untouched and must not be retried. If its completed metadata-only output becomes authenticated, it may be considered as a shared *metadata provenance source* only after the V7 license, identity and isolation checks; it does not automatically qualify any V7 audio or speaker.

Existing Vosen schwa16 work is also historical evidence only. It may be reused as original audio only if its exact pronunciation matches /heɪ ˈvoʊ.sən/, its source identity and license are intact, and it passes the V7 admission review. It is not automatically training-admitted.

## Execution gates

- G0: this dual-track specification and inert candidate registrations reviewed.
- G1: data inventory, licensing, old-exposure exclusion and source-group capacity proven for both tracks.
- G2: independently frozen candidate-specific datasets, 6/2/2/2 manifests and training budgets.
- G3: candidate-specific train workflow contracts; no run until all prerequisites bind.
- G4: one training run per candidate, with immutable receipts and no retry.
- G5: independent training readback and fixed-weight primary evaluation per candidate.
- G6: sealed replication per candidate after its own primary PASS.
- G7: C11/ONNX/C compiler/QEMU numerical and event parity plus SSC305 hardware tests.
- G8: independent release verification, package signing/hash, installation, rollback drill and explicit product selection.

Until G1-G3 are complete, both candidate registries remain inert. No training, source-generation, observer, locked-test or deployment budget is authorized by this design document alone.
