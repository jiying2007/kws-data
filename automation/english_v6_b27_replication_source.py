"""B27 replica profile; no source can open without original primary PASS."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_source as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b27_replication_source_ancestry as lineage
import english_v6_b27_replication_source_observe as observers
import english_v6_b27_replication_natural_sources as natural
PANEL = 'replication'
SCHEMA = 'english-v6-b27-fresh-replication-source-v1'
BATCH = 'english-v6-b27-120-original-quarantine-replication-screen-v1'
if __name__ == '__main__':
    main()
