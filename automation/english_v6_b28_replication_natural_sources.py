"""Same unchanged natural decoder under the fixed B28 replica profile."""
import english_v6_b28_natural_sources as primary_module
from english_v6_b22_replication_reuse import inherit
inherit(primary_module, globals())
import english_v6_b28_replication_source_ancestry as lineage
qa, digest = lineage.qa, lineage.digest
