"""The same two-corpus whole-original transport under the sealed replica identity."""
import english_v6_b28_source_transport as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b28_replication_source_ancestry as lineage
