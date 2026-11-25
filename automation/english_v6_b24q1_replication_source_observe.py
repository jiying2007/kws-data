"""The identical approved Q1 quarantine policy under sealed replica identities."""
from __future__ import annotations
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24q1_source_observe as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
import english_v6_b24q1_replication_source_ancestry as lineage
SOURCE_SCHEMA = 'english-v6-b24q1-fresh-replication-source-v1'
BATCH = 'english-v6-b24q1-120-original-quarantine-replication-screen-v1'
PANEL = 'replication'


def expected_registration(source, supplied=None):
    lineage.read_plan()
    import english_v6_b24q1_replication_source as owner
    expected = owner.registration(Path(source))
    need(supplied is None or quarantine.equal(supplied, expected), 'SOURCE_AUTHORIZED_REPLICA_REGISTRATION')
    return expected


def require_original_rows(source, rows):
    import english_v6_b24q1_replication_source as owner
    need(quarantine.equal(rows, owner.read_source(Path(source))), 'ALL_REPLICA_ORIGINALS_IN_REGISTERED_ORDER')
