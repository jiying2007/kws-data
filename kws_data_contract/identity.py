"""Keep declared and content-derived source identities in one data split."""
from __future__ import annotations

ROLES = frozenset(("train-candidate", "development-only"))


class SourceRoles:
    """Check aliases, PCM and original ancestry independently of caller names.

    This is a consistency check on supplied evidence, not speaker recognition.
    Missing evidence is not an identity and never joins unrelated sources.
    """

    def __init__(self):
        self._roles = {}

    def add(self, role, *, family=None, speaker=None, pcm_sha256=None,
            wav_sha256=None, parent_source_sha256=None):
        if type(role) is not str or role not in ROLES:
            raise ValueError("SOURCE_SPLIT_ROLE")
        identities = (("family", family), ("speaker", speaker),
                      ("pcm", pcm_sha256), ("file", wav_sha256),
                      ("file", parent_source_sha256))
        keys = []
        for kind, value in identities:
            if value is None:
                continue
            if type(value) is not str or not value:
                raise ValueError("SOURCE_IDENTITY_VALUE")
            key = (kind, value)
            if key in self._roles and self._roles[key] != role:
                raise ValueError("SOURCE_IDENTITY_SPLIT_LEAK:" + kind)
            keys.append(key)
        for key in keys:
            self._roles[key] = role
