# Fresh MEH-roh corpus contract

Campaign: `english-mero-meh-corpus-v1`. This is a prospective training corpus;
it does not reclassify or admit any historical probe. All source, observer,
admission, training and held-out results must retain their actual status.

## Pronunciation and observations

The selected target is **Hey Mero /heɪ ˈmɛroʊ/**. Vosen remains
**/heɪ ˈvoʊsən/**. Kokoro receives `hˈA mˈɛɹO.` and `hˈA vˈOsən.`.
These are generation requests, not observed labels or pronunciation gold.

The independent XLSR decision uses the complete final 16 kHz mono int16 input.
The requested text, target phones, class and KWS scores never enter the four
observers. Qwen, Whisper and Parakeet produce blind complete transcripts; XLSR
retains the entire native input, mask, logits, argmax IDs and CTC output.

The whole-phone target families are `h?eɪmɛr(oʊ|ou|o)` and
`h?eɪv(oʊ|ou|o)s[əɐʌ]n`, after removal of stress/length/tie notation and mapping
rhotic symbols `ɹ/ɾ` to `r`. These observer tolerances are declared before the
new synthesis:

- Missing initial `h` is tolerated only when all three ASR outputs support the
  complete target spelling family. It is a CTC omission tolerance, not an
  additional generated wake pronunciation.
- The old Vosen **unstressed** vowel observation family `[əɐʌ]` is retained;
  the medial consonant remains strictly `s`. This does not admit Vosen-z.
- `oʊ/ou/o` is the declared GOAT output family. In particular, Mero final
  `ɔ`, `ə`, `ʌ` and `ɚ` are not converted into that family.
- The profile's Cambridge-style `e` notation for the DRESS vowel is not a
  blanket substitution of an XLSR `/e/` phone for `/ɛ/`.
- Stress is requested in synthesis; the stripped CTC decision does not claim
  independent verification of native stress, naturalness or human intent.

ASR aliases are spelling support only. A transcript such as Marrow or Merrow can
support the same observed Mero phone sequence. No strict Marrow/Merrow negative
is generated or included in strict false-positive scoring. Old MEER-family
requests are not new positives and do not automatically become negatives.

## Fixed rows and source roles

There are exactly **144 native requests: 12 presets × 12 cells**.

| Role | Presets |
| --- | --- |
| train | af_bella, af_sarah, am_michael, am_puck, af_nicole, am_fenrir |
| development | af_aoede, am_echo |
| calibration | af_kore, af_alloy |
| locked-test | af_nova, am_onyx |

The four previously exposed presets remain on the training side. Each new
preset has one fixed role, and all speed/noise variants inherit it. No preset
is mixed at runtime. `af_heart` is excluded because the existing typed source
contract marks its mixture lineage unresolved. These identities describe
upstream preset styles; they do not assert twelve independent people or
complete knowledge of human ancestry.

Each preset produces:

| Cells | Count | Native speeds |
| --- | ---: | --- |
| Mero main target | 2 | 0.90, 1.10 |
| Vosen main target | 2 | 0.90, 1.10 |
| Nero, Memo, Bosen-s, Bosen-z | 4 | 1.00 |
| Hello there; Good morning; Turn it off; Open the door | 4 | 1.00 |

The four contrasts have predeclared full-phone contracts with a genuinely
different consonant from the target. Both Bosen `/s/` and `/z/` are explicitly
allowed negative hypotheses because `/b/` is preserved. The actual matched
hypothesis is retained separately from the request. A negative-class request
can be admitted only as an independently observed registered negative kind;
a failed positive request cannot be relabelled background.

The roster was chosen before new synthesis using the upstream American-English
voice list and its published source-quality grades. In particular, this avoids
the initially considered `am_adam` F+ and uses `am_fenrir` C+ instead. These
grades are upstream estimates, not measurements of this campaign. The roster
will not be replaced in response to new outputs.

Asset revision: `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987`.
The new recipe includes each selected asset's actual LFS SHA-256 and byte count.
Before model construction the producer verifies downloaded asset bytes against
those pins. Source metadata was retrieved directly from:

- [Fixed-revision voice tree](https://huggingface.co/api/models/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987/voices?recursive=false&expand=false).
- [Fixed-revision voice documentation](https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/VOICES.md).

The exact retrieved bytes are retained as `KOKORO_VOICES_TREE.json` and
`KOKORO_VOICES_UPSTREAM.md`. Their SHA-256 values are also recorded in the recipe.
The upstream documentation notes a short-utterance weakness. It motivates
reporting this batch honestly; it is not permission to crop, repeat, replace
or regenerate examples after observing the results.

## Waveform provenance and finite budget

Each input is a complete native short phrase: no carrier, crop, parent-region
phone labels, appended speech or observer-selected boundaries. The source
retains the whole 24 kHz float32 output, then performs exactly one whole-output
resample to 16 kHz and one int16 quantization.

`parent_pcm_sha256` means the actual complete **24 kHz little-endian float32
bytes**. Its source format and scope are recorded explicitly; it is not an
int16 parent slice. `pcm_sha256` means the final complete 16 kHz int16 bytes
seen by all observers and later consumed by training. Both acoustic identities,
and the request and row identities, are checked against the authenticated
union of consumed probe exclusions.

The budget is 144 TTS requests plus 144 requests to each of four observers:
**720 high-level model requests**, no retry or automatic expansion. Missing,
failed or unknown rows remain in the 144-row denominator. A failed campaign
does not authorize replacement voices or another generation round.

## Admission and training eligibility

Complete matching phone output plus three compatible ASRs yields `silver`
with training weight 1.0. Two compatible ASRs plus complete phones, a complete
third output and no predeclared competing-word evidence yields `weak` with
weight 0.5. Insufficient or contradictory support is `unknown`, not a gold
judgment that the person or synthesizer pronounced a name incorrectly.

Contrast negatives require their own complete registered phone pattern and
the same two/three text-support rule. Ordinary negatives require a complete
registered ordinary sentence supported by two/three ASRs and no wake evidence
in any text or the complete phone output. Absence of a target match alone is
never a negative label.

Source quality requires 16 kHz mono int16, 6,400–160,000 samples, normalized RMS
at least 0.0001, and int16 rail fraction at most 0.01. Positive rows additionally
require at most **28,672 samples (1.792 seconds)** and at least three whole-clip
support windows under the unchanged 38,400-sample prefix, 16,000-sample suffix,
1,280-sample step and frozen causal support geometry. Longer positives are
masked, never cropped or regenerated.

The blind candidate is fixed before requested class/kind is consulted. Request
disagreement is unknown. Rows with incomplete group coverage retain their
observed evidence separately but have final unknown/null/zero admission fields.
Only a group with admitted classes `{0,1,2}` is usable. The unchanged minimum is
6 training, 2 development, 2 calibration and 2 locked-test groups. Failure to
reach any minimum stops training; no role reassignment or group replacement
is permitted.

Unlike the historical probes, successfully admitted fresh training-role rows
have `training_allowed=true`; other roles always have weight zero and cannot
enter the optimizer. At most 432 fresh clean/SNR10/SNR5 bags are derived by the
existing training core, within its 192-original-row bound. Source authentication,
saved-array checks, complete role isolation and policy recomputation precede
feature or training model execution. No improvement, independent-human quality
or device result is claimed until the actual later evaluations support it.
