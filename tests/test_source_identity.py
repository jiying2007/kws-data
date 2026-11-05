"""Identity consistency is independent of aliases and optional metadata."""
import unittest

from kws_data_contract.identity import SourceRoles


class SourceIdentityTests(unittest.TestCase):
    def test_missing_hashes_do_not_join_unrelated_sources(self):
        roles = SourceRoles()
        roles.add("train-candidate", family="first")
        roles.add("development-only", family="second")

    def test_shared_original_within_one_role_is_allowed(self):
        roles = SourceRoles()
        roles.add("train-candidate", family="first", parent_source_sha256="a" * 64)
        roles.add("train-candidate", family="alias", parent_source_sha256="a" * 64)

    def test_original_and_normalized_alias_are_joined(self):
        roles = SourceRoles()
        roles.add("train-candidate", wav_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:file"):
            roles.add("development-only", parent_source_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
