"""Unchanged fixed-mirror transport guarded by the original PRIMARY PASS."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b23_source_transport as original
from english_v6_b22_replication_reuse import inherit
inherit(original,globals())
import english_v6_b23_replication_source_ancestry as lineage
