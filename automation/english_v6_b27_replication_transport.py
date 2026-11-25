"""Same whole-corpus transport, sealed by B27 PRIMARY original PASS."""
import english_v6_b27_source_transport as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b27_replication_source_ancestry as lineage
need, digest, record = lineage.need, lineage.digest, lineage.qa.record
