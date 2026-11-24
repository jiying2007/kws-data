"""Small metadata fixtures for canonical-person ancestry; no audio/model dependency."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_source_lineage import canonical_person, cross_check


def fixture(speaker="1462"):
    old = {"id": "libri-speaker-1462", "speaker": "1462", "role": "train",
           "pcm_sha256": "old-splice", "frames": 16, "fragments": []}
    legacy = {"actual_research_tracks": [old], "decoded_originals": []}
    prior = [{"id": old["id"], "role": "train", "source": copy.deepcopy(old),
              "observed_samples": 16, "records": 1}]
    source_id = speaker + "-1-0001"
    row = {"id": "new-namespace-" + source_id, "speaker": speaker, "chapter": "1",
           "source_id": source_id, "source_member": "LibriSpeech/dev-clean/" + speaker + "/1/" + source_id + ".flac",
           "source_group_id": "librispeech-dev-clean-" + speaker, "pcm_sha256": "new-pcm",
           "flac_sha256": "new-flac", "role": "development"}
    current = {"rows": [row]}
    roster = {"primary": [{"source_id": source_id, "speaker": speaker, "chapter": "1"}]}
    scores = {"records": [{"id": row["id"], "cohort": "natural", "group": row["source_group_id"],
                           "pcm_sha256": row["pcm_sha256"], "positive": False}]}
    return legacy, prior, current, roster, scores


class LineageTests(unittest.TestCase):
    def test_different_group_names_and_new_pcm_do_not_make_person_independent(self):
        out = cross_check(*fixture())
        self.assertEqual(out["status"], "SOURCE_LEAKAGE")
        self.assertEqual(out["known_disjoint_natural_readers"], 0)
        self.assertEqual(out["same_historical_decoded_originals"], 0)

    def test_unseen_reader_is_distinct_from_only_audited_history(self):
        out = cross_check(*fixture("2412"))
        self.assertEqual(out["status"], "NO_KNOWN_OVERLAP_IN_AUDITED_ARCHIVE")
        self.assertEqual(out["known_disjoint_natural_readers"], 1)

    def test_mismatched_current_pcm_fails_closed(self):
        values = fixture()
        values[-1]["records"][0]["pcm_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            cross_check(*values)

    def test_metadata_reservation_without_observed_features_is_not_proof(self):
        values = fixture()
        values[1][0]["records"] = 0
        with self.assertRaises(ValueError):
            cross_check(*values)

    def test_numeric_canonical_identity_requires_no_alias_guessing(self):
        self.assertEqual(canonical_person(1462), "librispeech:1462")
        for wrong in ("p1462", "librispeech-1462", "01462"):
            with self.assertRaises(ValueError):
                canonical_person(wrong)


if __name__ == "__main__":
    unittest.main()
