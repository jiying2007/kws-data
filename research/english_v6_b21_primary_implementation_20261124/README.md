# B21 fresh primary evaluation implementation

This change prepares one primary evaluation of the new B21 TRAIN-only paired checkpoints. It is **inert**: `scoring_allowed` remains false and original artifact pins are absent. The independent assessment (`65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc`) and training design (`76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1`) match the jointly reviewed final source-observer and renderer corrections. No scoring run, source run, training operation or model import is performed by these files in their present state.

## Evaluation contract

The evaluator consumes ten exact original ZIPs: the fixed B0 and B9 baseline artifacts, the actual completed B21 pair, one new combined source artifact, its original SCREEN, and all five native observer artifacts. Source, SCREEN and the five native artifacts must share the same original successful source run/head. The pair must come from the separately registered B21 TRAIN event and match the final assessment and training design bytes.

All 120 raw primary originals are authenticated, including originals rejected by source quality or native screening. The source reader's eligible rows and all five saved native provider outputs must exactly reproduce SCREEN. Admission preserves the frozen row order, PCM identity, source group, requested class and cohort; UNKNOWN entries remain excluded. The source minima stay at 12 synthetic positives, 24 synthetic ordinary inputs, eight paired synthetic source groups, 12 natural ordinary inputs and eight natural readers.

The complete 120-row source population is checked against every historical PCM/group/canonical person, all 88 actual TRAIN originals and the unopened replication reserve. The source stage's metadata enumeration does not open the reserved source audio. No historical B20 source is scored, trained on, retuned or substituted into the new panel.

Only after these checks does the workflow install the pinned CPU runtime. B0, B9, the B21 control and the B21 challenger use the original common frontend and normalization. The evaluator checks the actual PT/NPZ/ONNX byte identities, PT↔NPZ learned tensor identity and the original finite 43,876-parameter architecture. It adds no warm-up inference. Each admitted original is scored at the unchanged gains 1.0, 0.75 and 0.5.

Every model entry writes a unique consumed/unknown journal before the forward. A hard **1,440**-forward limit is checked before the next entry, corresponding to the maximal 120 originals × three gains × four roles. Every raw output and saved frame array is retained. No successful or uncertain entry can run again. A separately uploaded consumed marker and the complete workflow-run inventory prevent replay after interrupted runs or expired artifacts.

The exact original B14 decision code and assessment SHA256 are checked before scoring. A primary PASS only permits separately authorized independent replication; it does not release a model or allow candidate engineering promotion. All other outcomes preserve B0. Physical SSC305 results, continuous natural FAR/h and field FRR are not inferred from this development panel.

## Producer evidence

The frozen producer manifest covers the primary and source Python closures, metadata resources and original workflows. Each fetched file is checked against both its SHA256 pin and its original commit's Git tree/blob identity. The actual source run's producer is saved separately from the scoring producer. The upload template sets `include-hidden-files: true`, retaining `.github` metadata and workflows in the original artifact. `FETCH.json` retains both original Git trees and all ten input metadata receipts.

## Activation handoff

Before activation, the owner must fill the real pair/source/SCREEN/five-native artifact tuples and the real control/challenger PT hashes, verify the final jointly frozen assessment/design hashes, and freeze the complete producer inventory. This is mechanical binding to already reviewed original evidence; it cannot alter the numerical quality function, source minima, gain list, model roles or source population.

`english_v6_b21_primary_inputs.json` names the ten archives through `inputs` (`B0`, `B9`, `pair`, `source`, `screen`) and `native_inputs` (`qwen`, `whisper`, `parakeet`, `xlsr`, `lv60`). Each tuple contains exactly `id`, `run_id`, `run_head`, `bytes` and `sha256`. The input directory keeps them under those role names, with native filenames prefixed `native_`.

`english_v6_b21_primary_producer.json` uses `primary.files`, `source.files`, `source.resources`, the union `resources`, the actual source workflow SHA256, and the primary workflow template SHA256. Source resources must include all inherited historical registration resources consumed by the source reader as well as the final B21 plans/census/history. The primary resource union additionally includes the B21 training design and B14 quality assessment. The spec and producer manifest themselves are retained separately, avoiding a self-referential content hash.

The reviewed template is activated only by replacing its two `__B21_REVIEWED_SCORING_PARENT__` placeholders with the exact reviewed main parent and adding the single workflow on the registered execution branch. Its first public push must have one parent and no other changed file. The CLI's `check` mode reports inert/bound state and performs zero model/source calls.

## Validation performed locally

Eleven standard-library fixtures pass. They exercise the inert boundary before any dependency/network work, original source identity and UNKNOWN handling, natural veto and label types, rejected-source overlap checks, consumed-before-forward and replay rejection, the pre-entry 1,440 ceiling, safe ZIP extraction, unchanged greedy CTC clocks, first-public-run identity, Git blob identity, and hidden-file retention in the workflow. These are implementation fixtures, not acoustic evidence or permission to execute a model.
