"""Unchanged whole-corpus fixed official transport under the explicit B27 profile."""
import english_v6_b27_profile as profile
original = profile.preserved('english_v6_b22_source_transport')
profile.inherit(original, globals())
import english_v6_b27_source_ancestry as lineage
need, digest, record = lineage.need, lineage.digest, lineage.qa.record
