# B21 authorization continuation after the raw-log CLI STOP

The actual amended source run [45375018737](https://github.com/jiying2007/kws-data/actions/runs/45375018737), at commit `c662f81c8005e0d1108f5d5c8eb250b0a26398b2`, stopped while authenticating the earlier source job log. GitHub CLI refused the original log's terminal escape bytes. The failure happened before corpus transfer, source consumption, natural decoding, native observations, or KWS inference.

The original run remains failed. Its two original artifacts, complete job log, exact Git producer, and final API identities have been independently verified. The compact committed proof retains their content hashes, the complete original producer manifest, and snapshots of the operational files that change. The complete original producer ZIP remains pinned by artifact ID, size, and SHA256. The actual producer had 292 files: the reviewed 291 plus the unchanged historical metadata workflow. No reviewed scientific file was missing or changed.

This continuation uses a separately registered branch, workflow, and consumption markers. Before any new source work, it verifies the precise prior STOP and refreshes the failed run, jobs, artifact identities, and all six unused source/observer budgets. It then follows the existing authorization and independently consumed source/provider budgets. The failed workflow is not rerun.

The operational repair uses `gh api --allow-escape-sequences` with both stdout and stderr captured as bytes. Original logs and ZIPs are never decoded, normalized, filtered, or rendered to a terminal. The same repair is applied to the still inert PRIMARY, replica, saved-quality downloader, and engineering downloader. Byte fixtures include terminal controls and non-UTF8 values. All numerical evaluation bodies are unchanged.

## Unchanged scientific scope

- Reuse exactly the 96 original synthetic clips from failed run `45367984097`, including their original WAV, native PCM, and generation journals. That earlier run really consumed 96 TTS calls and remains a consumed STOP.
- Generate zero additional primary TTS clips. Decode only the prospectively registered 24 natural originals. The natural predicate, speaker exclusions, deterministic rankings, exact rosters, and source admission minima remain fixed.
- Retain the same five native observers, per-provider journals, call budgets, four frozen KWS models, gains, and B14 numerical promotion criteria.
- Keep replica audio sealed until an independently reproduced original PRIMARY numerical PASS. The inert replica preparation now routes to its already registered 24 natural originals and records source consumption before transport or generation.
- Keep engineering disabled until both independent original quality panels pass. B0 remains the installed baseline. No physical SSC305 or field validation is claimed.

The downstream changes received an independent AST review: only raw-byte download helpers changed their function bodies. PRIMARY retains 34 other function/class bodies, engineering retains 32, and the saved-quality reader retains all logic outside its nested download helper. The targeted 41 tests passed normally and under both `-O` and `-OO`; the replica preparation passed its 41 targeted offline tests. These are implementation checks, not quality evidence or promotion permits.
