# B27 complete original binding through public CI

The local execution environment became unavailable after the original B27 TRAIN
readback had completed and while SOURCE run 45597047117 was still running.
The already committed source repair and that original remote run are unchanged.

This companion moves only evidence acquisition and the existing offline binding
command into a public read-only GitHub Actions job. It is inert on main and on
pull requests. Its first activation is a separate workflow-only child of the
reviewed parent on research/en-v6-b27-primary-binding-ci-once-20261125.

## Fixed input and execution boundaries

The companion accepts only the original SOURCE run 45597047117 at commit
7429fee92456bb04ada5187be04aadbb5b18d826. It requires the complete successful
first run before large downloads. B0, B9, the actual B27 TRAIN and all source
artifacts retain their original provenance and complete byte hashes.

It fetches all 41 individually pinned TRAIN proof files using the unchanged
original_proof.fetch_files entry point. It fetches the twelve complete quality
input ZIPs and complete raw API bodies, including both actual Git trees.
The unchanged bind_b27_quality.bind performs the full independent TRAIN,
source, native SCREEN, producer, exclusion and capacity verification once.

No model or scientific runtime is installed, loaded or called. No audio is
generated, no observer is run, no budget marker is created and no quality
workflow is dispatched. Existing consumed marker names are never reused.

## Complete original output transport

The review artifact preserves the full raw APIs and tree responses together
with the exact generated registration, producer manifest and binding audit.
The original SCREEN member is copied directly from the authenticated archive.

Exactly five allowed export files, no more than 2,000,000 bytes in total, are
also emitted as indexed base64 chunks of at most 32,768 characters. Every record
carries the complete file byte count and SHA256; a final manifest covers every
file and chunk. A receiving client must reject missing, repeated, corrupted,
unexpected or truncated records and verify all bytes before making any commit.
The inventory hashes the other four files and is itself hashed in the final
transport manifest.

The workflow has contents:read and actions:read only, and checkout does not
persist credentials. It cannot push a branch, open a PR, merge, select a model
or activate scoring. The root reviewer receives and verifies the original bytes,
creates a separate registration PR, waits for its checks, and only then can
activate the unchanged once-only PRIMARY scoring workflow.

The new helper and workflow are not imported by any captured scientific
dependency. This preparation does not modify the frozen source, quality,
training, proof readers, model weights, source rosters or numerical gates.
