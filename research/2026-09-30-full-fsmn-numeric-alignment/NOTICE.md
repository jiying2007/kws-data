# Historical implementation attribution

The historical FSMN C variants and initial header under `prior-stages/` implement and modify the behavior of WeKws Python FSMN sources at commit `6a45aeb994dd81c0969ff877a5a7c46d60ed0c86`, distributed under Apache-2.0. Upstream FSMN authors: Yueyue Nyy and Jing Du. Streaming keyword-spotter behavior is attributable to the upstream WeKws contributors. The applicable Apache License, Version 2.0 text is retained verbatim as `LICENSE.wekws`.

These files are research C translations and historical diagnostic arithmetic variants, not unmodified upstream source. Their original archived bytes are unchanged; the original scalar reports bind their exact source/header/comparator SHA-256 values. The initial header and double comparator were restored by hash-verified historical reconstruction, as described in README.md. Current reusable implementation code remains in the separate kws-pipeline repository.

The separately obtained `iic/speech_charctc_kws_phone-xiaoyun` checkpoint, donor weights/payload, complete tensor manifest, large derived arrays, corpus audio and third-party dependencies are not distributed in this archive. Model provenance does not grant rights to any undistributed upstream pretraining audio. These attribution notes apply to the historical implementation evidence, not a new license grant for the repository's datasets.
