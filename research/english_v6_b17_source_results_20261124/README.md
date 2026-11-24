# B17 original source STOP and natural PASS audit

B17 is **not eligible for four-model scoring**. Synthetic run [45319586969](https://github.com/jiying2007/kws-data/actions/runs/45319586969) ended in failure at the preregistered five-by-96 valid-capture guard; no synthetic SCREEN artifact exists. Natural run [45319592567](https://github.com/jiying2007/kws-data/actions/runs/45319592567) succeeded and its original source gate passed.

Both public workflows ran once from approved parent `e46a54fb3e39b3a7a7abd8e8b1afdc00d1ec8cde`. Each source head changed only its workflow predecessor placeholder. Original ZIP digests, upload logs and authenticated GitHub artifact metadata agree. This audit loaded no observer/KWS model and made no new model calls.

## Original completion and admission

| Cohort | Raw originals | Original observer attempts | Valid saved captures | Original outcome |
|---|---:|---:|---:|---|
| Synthetic | 96 / 24 fixed voices | 480 | 478 | STOP: qwen 96, whisper 96, parakeet 94, xlsr 96, lv60 96 |
| Natural | 24 / 12 fixed readers | 120 | 120 | SOURCE_GATE_PASS: 21 admitted originals / 11 readers; 16 silver, 5 weak, 3 unknown |

All 96 full synthetic WAVs/native chunks match the frozen 718–741 plan and exact original Piper configuration map. All 24 natural slots remain present without replacements or source-quality drops; every retained FLAC decodes to the identical stored PCM16 WAV. The new natural and synthetic reader/PCM sets are disjoint. Natural readers/PCM are disjoint from the raw historical 100-reader/563-PCM evidence. Historical synthetic ancestry checks also completed within the original authenticated source job before TTS.

The ten observer archives were read back with the original model/vocabulary/capture bindings. All saved native phone float32 buffers passed byte hashes, finite-value checks and saved CTC parity. Natural SCREEN decisions agree with the unchanged original admission rule and raw captures. Its original SCREEN JSON SHA-256 is `1ef5129b5fff1ac97b7102efb59fbf1eb4adeb4fd0afc7e2c66009c1ae3a9709`.

## Unrecoverable original Parakeet capture gap

The two original failures are `b17-primary-725-mero-slow` and `b17-primary-730-mero-slow`, each recorded as `ValueError:RAW_UNPROMPTED_COMPLETE_ASR_TEXT`. Their artifact folders contain STARTED and FAILED only; raw stdout/transcript, explicit native exit-code/return receipts and CAPTURE were not retained. The exact original wrapper and adapter are retained in evidence. Pinned control flow implies the native process had returned exit code zero before text validation rejected its output, but does not establish whether the missing text was empty, oversized or contained controls. No empty/abstention value was fabricated and no failed original was retried or relabeled.

Original readback job `138200438377` failed at `2026-11-24T03:43:24.7319892Z`: `ValueError: exact five original native model forwards x 96`; exit code 1. The designated B17 model-scoring branch had zero Actions runs at the retained read-only API snapshot. Source/native artifacts record zero KWS evaluation of these originals. Acoustic quality and promotion were not evaluated.

## TTS accounting

The source ZIP has 290 members: 96 WAVs, 96 native PCM chunks, 96 per-original starts receipts, MANIFEST and root STARTED. Each original has exactly one saved native output chunk. The original B13 renderer invokes `PiperVoice.load` and `voice.synthesize` once per frozen row: 96 logical originals and 96 public synthesis API invocations. `tts_calls` / `max_source_calls` use this logical request denominator. Internal Piper/ONNX forward calls were not instrumented; their count remains null and is not inferred from native chunks.

## Exact original source input locks

| Role | Artifact | Run | Head | Bytes | ZIP SHA-256 |
|---|---:|---:|---|---:|---|
| synth_source | 14886149837 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 5065507 | `f3578f69bd0d933a982732cea2ca6dc4a02ad38880d1003116b9992b8de698d2` |
| synth_screen | absent | — | — | — | No original SCREEN: strict coverage STOP |
| natural_source | 14885854460 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 4065973 | `3798010ec657e5408a60edbd71d92c8c98588ebfd78e71766874e811a770d035` |
| natural_screen | 14887522111 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 5283 | `25ebcf834b491021baef934e47fd58f2b28bf00dc212fc6bf4da454dcef1c1a1` |

## Exact original observer locks

| Cohort/provider | Artifact | Run | Head | Bytes | ZIP SHA-256 |
|---|---:|---:|---|---:|---|
| synthetic/xlsr | 14888285170 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 5551061 | `9b4ffaa27f5c3cca7b651559d6dd474adfe679ad1b2be7038206e3f967c31d1d` |
| synthetic/qwen | 14888235488 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 91321 | `e86f3eceb134c5b15dab5271972089ae501e70a892478ecffbec4b3c0a2d687f` |
| synthetic/parakeet | 14887920318 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 91746 | `a2317ef83c0ebf028c615d8398719e9841b6c81e1967432bdb7a023de77104d6` |
| synthetic/whisper | 14887123917 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 91682 | `a437edaef19278a87f99b706724a27a07286cf1e0f6176d25c69333c9ff72a20` |
| synthetic/lv60 | 14886939867 | 45319586969 | `1a0f58f6f5ba190eb9746f690d4b7c23a3fb1668` | 5545463 | `cc630348f71214e04ec670638008c79904b66dec052466b452107114bea67d88` |
| natural/parakeet | 14888305385 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 25973 | `a88b0df9e67f08b8cf712fe414c0e5398539228e14693c89cb0a3bab4e552401` |
| natural/xlsr | 14887541548 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 6808589 | `642aadf12d25b8b3f38e696c1215008d6f914359244d3c504d3e73db0e4cf2d0` |
| natural/lv60 | 14887168543 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 6783044 | `8075ccd2fd9f4d1670b30df2cac1432fb1019c11a9c32184b7ee23eb05c5ea31` |
| natural/whisper | 14887054761 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 25977 | `5dd4f8daafce8e1001bbc244c62896b5e6224a2bbb84da79a709834e36d6a756` |
| natural/qwen | 14886994599 | 45319592567 | `5c031522a2c9e06117aee756490d7793ea6350aa` | 25952 | `7f27271fc082fa6e68d24eb9cbee1261e9264ab7bcf49f3e18e1f7323b28921d` |

The three source/screen ZIPs and ten observer ZIPs are retained locally under `/workspace/scratch/c21f90c2d979/b17_primary_source_readback/artifacts/`. The canonical machine-readable tuples are in ARTIFACT_LOCKS.json. This report records original failure/PASS outcomes only; it does not authorize B17 replication, model scoring or promotion. Any future reuse of unchanged zero-KWS natural sources needs a separate prospective evaluation protocol.
