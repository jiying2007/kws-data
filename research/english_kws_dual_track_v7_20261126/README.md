# English KWS Dual-Track V7.0

Status: **INERT / DESIGN-AND-REGISTRATION ONLY**  
Date: 2026-11-26  
Targets: **Hey Vosen** /heɪ ˈvoʊ.sən/ and **Hey Mero** /heɪ ˈmɛroʊ/

## Product decision

Build two separately trained, separately packaged single-keyword detectors. This is not one multi-keyword model. The shared engineering substrate remains PCM16 → FFT512/Hann400 → 40 Mel → streaming FSMN-CTC and the existing C11 runtime. Model parameters, optimizer state, checkpoints, thresholds, candidate IDs, training receipts, score reports and release packages remain independent.

## Exact brand pronunciation lock

| Candidate | Exact phrase | Required US English reading | Required consonants |
|---|---|---|---|
| VOSEN | Hey Vosen | /heɪ ˈvoʊ.sən/ | voiced /v/, **voiceless /s/**, schwa /ə/ |
| MERO | Hey Mero | /heɪ ˈmɛroʊ/ | /m/, DRESS /ɛ/, rhotic /ɹ/ |

Vosen's medial consonant is explicitly **/s/**, not /z/. Do not rhyme it with *frozen*, do not use the voiced-z variant, and do not admit /z/ as a pronunciation variant. The Piper source phone string for Vosen is `heɪ vˈoʊsən`. Mero's source phone string is `heɪ mˈɛɹoʊ`.

## Matched paired source panel

V7 uses one prospectively frozen, matched 12-source-group panel for both tracks. The same original PCM is referenced by both candidate manifests, but its label is candidate-relative: Vosen audio is positive for VOSEN and a mandatory cross-trigger negative for MERO; Mero audio is positive for MERO and a mandatory cross-trigger negative for VOSEN. Ordinary and contrast negatives are shared with identical original hashes. This is not positive relabeling within either candidate.

The prospective source plan is `automation/english_kws_dual_track_v7_source_plan.json`:

- 12 official Piper/LibriTTS source identities, candidate indices 496–507.
- Fixed source roles: 6 train, 2 development, 2 calibration, 2 locked-test.
- 12 utterance requests per source group: three speeds for each wake phrase, four phonetic contrasts, and two ordinary negatives.
- 144 whole-utterance TTS requests; 3 blind ASR observers and one independent phone observer, each bounded to 144 observations.
- Maximum 720 high-level TTS/observer requests; zero KWS forwards and zero optimizer updates at source stage.
- The exact 904-speaker config and original-speaker-ID exclusion audit must PASS before any TTS. No fallback index, replacement speaker or retry is permitted.

These are Piper-generated synthetic utterances derived from LibriTTS speaker identities, **not real human microphone recordings**. They can support controlled synthetic training only after independent review. Real human recordings and real 3–5 m field tests remain mandatory for product qualification.

The audio gate requires full original WAV/PCM identity, no crop, no retries, whole-item quarantine for empty/invalid returns, no duplicate PCM, and positive geometry within the existing 28,672-sample / three-complete-window requirement. Locked-test audio remains sealed from training, mining and checkpoint selection.

## Historical data decisions

Historical data is preserved, not silently reclassified:

1. The Vosen schwa16 recipe requested /heɪ ˈvoʊzən/ with voiced /z/ (it rhymed with *frozen*) and its execution pins were disabled. It is **not** a V7 positive source.
2. The Mero MEH run `44218734553` generated 144 rows and completed 720 observer requests, but its authenticated result was `STOP_FRESH_CORPUS_COVERAGE`: final training rows 0 and all final weights 0. It is not training-admitted for V7. Its Vosen request used /s/ and remains historical pronunciation evidence only.
3. The older synthetic192 recipe contains a Vosen voiced-/z/ positive hypothesis. Its exposed source identities and mixed pronunciation cohort are excluded from V7 qualification.

Do not regenerate any previously verified audio. The new Piper panel uses new source identities only after the map audit proves they are unused. The B28 TRAIN360 one-shot run remains immutable and must never be retried.

## Candidate isolation and data admission

Each candidate receives its own:
- positive and negative registry and candidate-relative label projection;
- independently hashed manifest and training admission receipt;
- initialization identity, optimizer state, random stream and training run;
- checkpoint lineage, quantization/export artifacts and candidate hash;
- independent fixed-weight primary evaluation, sealed replication and engineering proof;
- release decision and package.

Both candidates apply the same bounded multi-ASR and independent phone review. ASR agreement is evidence, not human gold. Every original retains SHA-256, source/license, speaker/source-group identity, synthesis recipe, exact text and raw observer outputs. Empty returns quarantine the entire item. Failed positives cannot be relabelled as ordinary negatives. Locked-test audio never enters optimizer, mining, tuning or checkpoint selection.

Each candidate independently satisfies 6/2/2/2 source-group isolation. The matched panel uses the same group-to-role mapping for paired comparison, while candidate manifests and all model/evaluation state remain separate.

## Training and fairness

Use the same frontend, model family, optimizer implementation, reproducibility controls, evaluation protocol and hardware constraints. Start each candidate from the same approved immutable initializer, but instantiate independent model objects, optimizer state, random streams and output directories.

Do not copy B28's single-candidate 192-update / 1,847-forward budget and pretend it is a dual-candidate budget. V7 candidate training budgets remain null until both source manifests and their exact admission results are frozen, resource estimates are completed, and candidate-specific budgets pass review. No optimizer run is authorized by this plan.

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

No numeric promotion threshold is invented here. Existing approved product gates must be bound to V7 before locked evaluation. Real human recordings and SSC305 field validation are required; synthetic Piper voices cannot establish real-human far-field quality.

## Competition decision

Only candidates passing every hard safety, quality, evidence and device gate may compete. Compare recall at the approved FAR limit, cross-trigger rate, far-field robustness, latency and resource use. Do not use a weighted average to compensate for a failed hard gate.

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

## Execution gates

- G0: dual-track specification, exact pronunciations and inert registrations.
- G1: 904-speaker map and exposure audit; licensing and 12 source groups authenticated.
- G2: one-shot paired source generation and four-observer evidence; independent candidate-relative admission.
- G3: two frozen 6/2/2/2 manifests and candidate-specific training budgets.
- G4: one independent training run per candidate, with immutable receipts and no retry.
- G5: independent training readback and fixed-weight primary evaluation per candidate.
- G6: sealed replication per candidate after its own primary PASS.
- G7: C11/ONNX/C compiler/QEMU numerical and event parity plus SSC305 hardware tests.
- G8: independent release verification, package signing/hash, installation, rollback drill and explicit product selection.

Until G1–G3 are complete, both candidate registries remain inert. No source audio, observers, KWS training, locked-test scoring or deployment is authorized by this design alone.
