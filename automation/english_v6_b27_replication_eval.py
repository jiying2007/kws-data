"""B27 sealed replica scoring profile, opened only by its complete original PRIMARY PASS."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_primary_eval as original
from english_v6_b22_replication_reuse import inherit
inherit(original, globals())
PANEL = 'replication'
SPEC = HERE / 'english_v6_b27_replication_inputs.json'
PRODUCER = HERE / 'english_v6_b27_replication_producer.json'
WORKFLOW_TEMPLATE = HERE / 'english_v6_b27_replication_workflow_template.yml'
SCHEMA = 'english-v6-b27-fresh-replication-four-frozen-models-v1'
SOURCE_MODULE = 'english_v6_b27_replication_source'
SOURCE_SCHEMA = 'english-v6-b27-fresh-replication-source-v1'
SCREEN_SCHEMA = 'english-v6-b27-120-original-quarantine-replication-screen-v1'
BRANCH = 'refs/heads/research/en-v6-b27-replication-scoring-once-20261125'
WORKFLOW = '.github/workflows/english-v6-b27-replication-model-eval-once.yml'
SOURCE_BRANCH = 'research/en-v6-b27-replication-sources-once-20261125'
SOURCE_WORKFLOW = '.github/workflows/english-v6-b27-replication-sources-once.yml'
SOURCE_TEMPLATE = 'automation/english_v6_b27_replication_source_workflow_template.yml'
SOURCE_PARENT_TOKEN = '__B27_REVIEWED_REPLICATION_SOURCE_PARENT__'
PARENT_TOKEN = '__B27_REVIEWED_REPLICATION_SCORING_PARENT__'
MARKER = 'english-v6-b27-replication-consumed'
PRODUCER_ROOTS = ('english_v6_b27_replication_eval', SOURCE_MODULE,
                  'english_v6_b27_quality_readback', 'english_v6_four_model_panel', 'english_v6_b14_decision')
CANDIDATE_PT = profile.document(SPEC.read_bytes()).get('model_pt_sha256', {}).get('challenger')
if __name__ == '__main__':
    main()
