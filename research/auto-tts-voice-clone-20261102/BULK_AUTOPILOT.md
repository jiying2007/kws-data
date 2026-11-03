# Bounded bulk automation

bulk_autopilot.py connects the existing persistent bulk TTS backend, native ASR receipts, screen_results, and three exact-source pipeline consumers. It does not create training authority or promote a model.

The controller freezes plan, sources, adapters, backend asset-lock identity, total TTS/ASR call budgets, and consumer source hashes. Durable starts count against the same total on resume. Failed or uncertain starts are not retried. Existing evidence is rechecked before new calls; a confusion HOLD persists across resume. A process-held lock excludes concurrent controllers, but this is not a power-loss exactly-once guarantee.

--wall-seconds is a cutoff for launching new calls, not a hard process runtime limit. The existing Qwen TTS call has no per-call hard timeout. ASR adapters retain their own timeout. A real run requires an external process supervisor, approved local model assets, licensed references and explicit execution permission. This change was tested only with artificial WAVs and fake TTS/ASR backends; no real synthesis, transcription, training or measured KWS accuracy is claimed.

Without --execute the CLI returns NOT_RUN. Missing model assets also return NOT_RUN. Outputs remain training_admitted=false, qualification_allowed=false. ASR consensus remains weak source evidence, not evaluation gold. Existing language and engine contracts are unchanged; this slice does not add an English generation lane.

The normal test suite exercises actual pinned pipeline consumer code from b02edb7c1cc26c65a6e291f1e7661f44724133fd. CI checks out that exact public source separately, without persistent credentials, and supplies KWS_BULK_PIPELINE_ROOT. Source hashes are checked before and after consumption. Existing source-only tests and branch-specific registration guards remain intact.
