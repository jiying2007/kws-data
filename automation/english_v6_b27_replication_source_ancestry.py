"""B27 replica identity; only complete same-candidate original PRIMARY PASS opens it."""
from pathlib import Path
import english_v6_b27_source_ancestry as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
PANEL = 'replication'
INPUTS = Path(__file__).with_name('english_v6_b27_replication_proof_binding.json')
BRANCH = 'refs/heads/research/en-v6-b27-replication-sources-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-replication-sources-once.yml'
SCHEMA = 'english-v6-b27-replication-source-authorization-v1'
MARKER = 'english-v6-b27-replication-source-consumed'
