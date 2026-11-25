"""B27 namespaced scope using the exact original full-byte cache algorithm."""
from contextvars import ContextVar
import english_v6_b27_profile as profile
original = profile.preserved('english_v6_b24q1_evidence_scope')
profile.inherit(original, globals())
KINDS = frozenset(('b27_complete_training', 'b27_primary_quality'))
_ACTIVE = ContextVar('b27_independent_complete_original_evidence', default=None)

# Rebind the actual generator, not contextlib's helper closure and globals.
from contextlib import contextmanager
from types import FunctionType
_original_generator = original.scope.__wrapped__
scope = contextmanager(FunctionType(_original_generator.__code__, globals(), 'scope',
    _original_generator.__defaults__, _original_generator.__closure__))
