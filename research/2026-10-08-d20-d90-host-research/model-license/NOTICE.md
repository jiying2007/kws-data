# Donor-derived research model notice

Original model: iic/speech_charctc_kws_phone-xiaoyun, published by the ModelScope iic/Speech team. Official model repository: https://modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun . The original model card at revision 68e1625545621d1dfb921866c4bc6b6a811b2685 explicitly declares Apache License 2.0. This is model-specific evidence, not an inference from a code license.

Exact original donor train/base.pt: 3,038,219 bytes, SHA-256 d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233. The original donor body is not newly included. Exact primary API responses, their URLs and hashes are preserved in DONOR-LICENSE-VERIFICATION.json and adjacent evidence. No separate LICENSE or NOTICE was found in that official pinned tree; the full Apache-2.0 text is supplied here. Existing original attribution in source and model cards is retained.

Modifications retained in this study: donor backbone plus six selected original head rows as the shared model initialization; fixed D20 and D90 full-encoder training for 1,200 updates per arm; FP32 flattened exports and associated state/CMVN identities. initial.pt is a donor-derived initialization artifact, not a trained D20/D90 endpoint. D20/D90 and converted exports are study derivatives, not official upstream model releases. MODEL-DERIVATIVES.json identifies every retained weight/flat-export member and original/public byte hash.

These files are research evidence only. Training, host-state and local numerical results do not qualify either derivative for product use. Original upstream pretraining data is not included, and this model license does not establish commercial rights in any voice or synthesized output.
