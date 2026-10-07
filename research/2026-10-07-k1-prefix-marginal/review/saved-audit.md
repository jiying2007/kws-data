# Saved receipt audit: completed candidate gate failure

The recovery receipts are internally consistent and match the reviewed source hashes. The candidate failed its fixed gate because M3, labeled `你好小屋`, received a certified K1 acceptance. N1 remained accepted. This is a completed negative scientific result, and the predeclared action is **STOP_CANDIDATE_NO_TUNING**.

| Observation | A mass | B mass | R mass | Certified action | Fixed-label gate |
|---|---:|---:|---:|---|---|
| M1 | 0.9980094357 | 0.0001679961 | 0.0018225682 | Accept | Pass |
| M2 | 4.9963e-10 | 4.8691e-12 | 0.9999999995 | Reject | Pass |
| M3 | 0.7411511569 | 0.1975905669 | 0.0612582763 | Accept | **Fail** |
| M4 | 4.2356e-19 | 1.7810e-12 | 0.999999999998 | Reject | Pass |
| M5 | 0.0000028052 | 0.0000097982 | 0.9999873966 | Reject | Pass |
| N1 | 0.8729288161 | 0.0004885542 | 0.1265826296 | Accept | Pass |

All six exact rational certificates were checked directly from the saved numerators and denominators. Accepts satisfy both lower A > 1/2 and lower A > upper B + upper R. Rejects satisfy upper A <= 1/2 and upper A <= lower B + lower R. No result is numerically unresolved. The aggregate bounds enclose unit total mass and satisfy the reviewed per-observation rounding budgets. Approximate masses and logs agree with one another and with the rational intervals to the declared numerical tolerance.

The result and ledger agree for the complete M1–M5,N1 order: 537 rows, 57 callbacks, six EOF records, then `attempt_complete`. The prior failed attempt remains intact, with no EOF or results file. Both old and new source hashes match their review records, and the recovery release/ledger/result identify cumulative attempt 2 and the same previous-attempt hash. The original 11,283-byte results file is unchanged.

The completed recovery records 81 exponential zeros across 19 rows and zero division-only zeros. The reviewed exact-difference guard and its ideal omitted-tail bound apply to those arithmetic zeros. This does not certify every other floating-point rounding step or acoustic calibration.

The receipts and prior static failure accounting give cumulative totals of 606 attempted softmax rows, 605 computed DP transitions, 596 committed rows, and six EOFs. The first attempt contributes 69/68/59/0 respectively; the complete recovery contributes 537/537/537/6. All 537 source rows had been loaded before each attempt. The first-attempt partial counts are reconstructed from program order and the coordinator-reported failure location, not directly logged counters.

Recovery resource receipts report 0.038824465 CPU seconds, 0.042778525 wall seconds, and 22,776 combined result/ledger bytes, below the configured caps. The 64 MiB address-space limit is verified in the reviewed source; peak memory is not measured in the receipt.

Audit scope was read-only receipt and source-hash inspection. No raw saved posterior files were reopened, no DP was rerun, and no model or audio operation occurred. Frozen raw-input hashes remain anchored by the unchanged observation manifest and the reviewed runner's successful pre-evaluation checks; this audit does not independently reproduce the posterior calculation.

Results SHA256: `76e35a4f16a004f6e3724b282c00775c350e3d02a7d833c4e8e5856bf68fc54a`

Recovery ledger SHA256: `e772664b181d69de66192669ea94ae2c652453db39768b4f6aeffdf4bede9d33`

Original failed ledger SHA256: `cf9c92a0e8db14cd187d8bbbd606a4dc0739fc62781e0f5dae69404c0313dbc2`

This single already exposed stock voice does not establish generalization, calibration, endpoint quality, continuous detection performance, or deployment readiness. The exact marginal remains mathematically useful, but this candidate is stopped as an M3-repair replacement.
