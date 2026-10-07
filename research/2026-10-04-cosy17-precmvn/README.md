# Cosy17 prepared feature archive

inputs.zip is the single exact 602,351-byte archive. ARCHIVE.json lists all 19 logical members and their SHA256/byte sizes: 17 little-endian FP32 PRE-CMVN feature arrays, an inner dataset manifest, and its README. The exact uncompressed total is 1,131,737 bytes, including 1,108,800 feature bytes. Archive SHA256: d43806895671140a22119021d409a7e1dbfb7facfde3392907573dc5e91d3ea3.

These are speech-derived acoustic features and human lexical technical labels from public CosyVoice synthetic speech. They retain speech-related spectral information; partial speech reconstruction may be possible. They are not anonymized or claimed to be non-invertible. No source WAV, model weight, conversation, approval record, private locator or execution receipt is included.

The inner manifest supplies fixed source order, actual words/targets, WAV/PCM/feature hashes, original public generation-code/config identities, and exact references to the previously published 32 prepared inputs at commit d0d54cf635189bdc522c9d0f6135082e81250fd8. Those old inputs are not retransmitted. Inaudible/development clips are excluded. Original source WAVs are not permanently archived by this package. Temporary Actions artifact URLs are not permanent source pins.

For use, verify archive size/hash and the complete member allowlist before bounded extraction into a new directory. Reject duplicate, extra, missing, encrypted, symlink or unsafe-path members. Only DEFLATE members are present; maximum individual member size is 107,200 bytes. Verify every restored member byte length/hash before reading the inner manifest. No archive member is executable; extraction does not execute code, train or run inference. Original model CMVN must later be applied exactly once to these PRE-CMVN features.

This is research input preservation only. Word boundaries and product-level FAR/FRR/device performance are not qualified. No new license grant or commercial-use clearance is asserted.
