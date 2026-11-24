"""New B23 authorization around the unchanged complete fixed-mirror transport."""
import english_v6_b22_source_transport as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b23_source_ancestry as lineage
need, digest, record = lineage.need, lineage.digest, lineage.qa.record
