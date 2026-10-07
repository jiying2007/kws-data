# Publication and final verification boundaries

This directory describes an optional transport, not permission to execute a
publication. Keep any older partial-upload queue paused while choosing one
publisher. Existing draft PRs, completed core content and confirmed Git objects
must not be deleted or force-rewritten.

1. Freeze and independently review every outer file, its exact SHA-256/size,
   Git blob SHA and serialized request size. Verify the 111 chunks reconstruct
   all six pinned ZIPs, the exact 2,243 public files, and all original logical
   assets. Review the pure invented negative tests and unchanged inner verifier.
   Before execution, durably save the exact source, protocol, file manifest and
   request plan in private recovery storage and verify their readback hashes.
   Obtain an explicit release for that exact frozen scope; preparation is not release.
2. Before any write, resolve the exact target repository, branch head and base
   tree again. If the user has already performed the manual import or the head
   otherwise differs from the approved head, stop and reconcile read-only.
   Do not silently update the old manual importer's expected head. Coordinate
   a single writer; no simultaneous manual and API publication.
3. Each binary request stores one exact chunk. The maximum binary request is
   1,398,183 bytes of compact base64 JSON. Large text metadata has a separately
   measured request size. These require an explicit change to any smaller
   request-size policy; a successful one-off capability probe alone is not a
   batch release or a measured speed guarantee.
4. Preserve a durable per-attempt ledger before calling GitHub, keyed by exact
   expected Git SHA, file SHA-256, size and payload identity. Use bounded rolling
   shared minute/hour budgets (at most 55 writes/minute and 440 writes/hour
   across all active publication work); include reservations, failures, duplicates
   and uncertain attempts. Respect stricter service limits and Retry-After. Confirm
   returned Git SHA before marking success. Preserve every failed/cancelled or
   duplicate attempt; never infer that cancelling a client undoes a server write.
5. On denial/cancel/error, stop new dispatch. Await already sent calls without
   cancelling them. An unknown write is reconciled read-only by expected Git
   SHA, then by exact tree entry and byte proof. Binary text-decoding errors do
   not establish absence. No blind reupload, route change, credential change,
   or automatic retry follows an uncertain or rejected call.
6. Build a tree that retains all existing unrelated files and the core checkpoint
   and adds only this reviewed outer transport. Before commit, read back its
   complete tree and prove every expected blob SHA, mode and size. Re-read the
   expected branch head immediately before commit creation, and again immediately
   before non-force ref update; any drift stops the operation for read-only
   reconciliation. A tree cannot
   substitute for actual remote SHA-256 byte verification. No main merge occurs.
7. After a normal non-force branch update, confirm the exact head/PR state and
   retrieve actual bytes for every new file at that immutable head. Verify file
   SHA-256/size and Git SHA, reconstruct the six packages and all original assets
   from those remote bytes, and rerun the invented tests. Read raw workflow data
   to bind every required successful CI run to that exact head and PR.
8. Report complete only when the full outer and inner remote chains and CI pass.
   Existence of uploaded blobs or a partial branch is not full delivery. The
   original failures, missing historical raw evidence and research-only limits
   remain unchanged.

Binary uploads are 111 requests before deduplication; large metadata, tree,
commit/ref and PR updates are additional operations. Small files may be grouped
in bounded tree requests. The request plan and checkpoint ledger are operational
records maintained outside this public scientific archive.
