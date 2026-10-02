"""Offline fake-model fixtures only. No torch, audio, model or network."""
from collections import OrderedDict, UserDict, namedtuple
from dataclasses import dataclass
import builtins
import gc
import hashlib
import json
from pathlib import Path
import sys
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "numpy", "funasr", "qwen_asr", "transformers"}:
        raise AssertionError("Tensor/model runtime import forbidden in stdlib fixtures")
    return _import(name, *args, **kwargs)


builtins.__import__ = guarded_import
from checkpoint_guard.checkpoint_contract import CheckpointContractError, digest
from checkpoint_guard.exact_apply import exact_apply, require_verified_receipt

Result = namedtuple("LoadResult", "missing_keys unexpected_keys")


@dataclass
class Tensor:
    shape: tuple
    dtype: str
    raw: bytes


def describe(tensor, include_content):
    if type(tensor) is not Tensor:
        raise TypeError("Not a tensor")
    row = {"shape": list(tensor.shape), "dtype": tensor.dtype}
    if include_content:
        row["sha256"] = hashlib.sha256(tensor.raw).hexdigest()
    return row


class Model:
    def __init__(self):
        self.state = OrderedDict((
            ("encoder.weight", Tensor((2,), "torch.float32", b"old-a")),
            ("decoder.bias", Tensor((), "torch.float32", b"old-b")),
        ))
        self._kws_weights_verified = True  # stale success must be invalidated
        self._kws_checkpoint_receipt = {"old": True}
        self.loads = []
        self.mode = "ok"
        self.loaded = False
        self.return_value = Result([], [])

    def state_dict(self):
        if self.loaded and self.mode == "readback_exception":
            raise RuntimeError("readback unavailable")
        result = OrderedDict(self.state)
        result._metadata = {"": {"version": 1}}  # actual torch state_dict convention
        return result

    def load_state_dict(self, checkpoint, strict):
        assert self._kws_weights_verified is False
        self.loads.append(strict)
        self.loaded = True
        if self.mode == "raise_before":
            raise RuntimeError("before write")
        if self.mode != "noop":
            self.state = OrderedDict((k, Tensor(v.shape, v.dtype, v.raw)) for k, v in checkpoint.items())
        if self.mode == "partial":
            self.state["decoder.bias"].raw = b"old-b"
            raise RuntimeError("after partial write")
        if self.mode == "wrong_content":
            self.state["decoder.bias"].raw += b"x"
        if self.mode == "wrong_shape":
            self.state["decoder.bias"].shape = (1,)
        if self.mode == "wrong_dtype":
            self.state["decoder.bias"].dtype = "torch.float64"
        if self.mode == "missing":
            self.state.pop("decoder.bias")
        if self.mode == "extra":
            self.state["other"] = Tensor((), "torch.float32", b"x")
        if self.mode == "mutate_both":
            checkpoint["decoder.bias"].raw = b"bad-both"
            self.state["decoder.bias"].raw = b"bad-both"
        if self.mode == "mutate_input_only":
            checkpoint["decoder.bias"].raw = b"bad-input"
        return self.return_value


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.cp = {"encoder.weight": Tensor((2,), "torch.float32", b"new-a"),
                   "decoder.bias": Tensor((), "torch.float32", b"new-b")}
        # Independent fixture architecture declaration, not generated from cp.
        self.schema = {"encoder.weight": {"shape": [2], "dtype": "torch.float32"},
                       "decoder.bias": {"shape": [], "dtype": "torch.float32"}}
        self.options = {"expected_schema": self.schema, "expected_schema_sha256": digest(self.schema)}

    def run_apply(self, checkpoint=None, descriptor=describe, **options):
        args = dict(self.options, **options)
        return exact_apply(self.model, self.cp if checkpoint is None else checkpoint,
                           descriptor, **args)

    def pre_reject(self, call):
        with self.assertRaises(CheckpointContractError) as caught:
            call()
        self.assertEqual(self.model.loads, [])
        self.assertIs(self.model._kws_weights_verified, False)
        self.assertIsNone(self.model._kws_checkpoint_receipt)
        self.assertFalse(caught.exception.mutation_may_have_occurred)

    def post_reject(self, call):
        with self.assertRaises(CheckpointContractError) as caught:
            call()
        self.assertEqual(self.model.loads, [True])
        self.assertIs(self.model._kws_weights_verified, False)
        self.assertIsNone(self.model._kws_checkpoint_receipt)
        self.assertTrue(caught.exception.mutation_may_have_occurred)

    def test_success_preserves_initially_different_contents(self):
        receipt = self.run_apply()
        self.assertEqual(self.model.loads, [True])
        self.assertIs(self.model._kws_weights_verified, True)
        self.assertEqual(self.model.state["decoder.bias"].raw, b"new-b")
        self.assertIs(receipt["execution_authorized"], False)
        self.assertEqual(receipt, require_verified_receipt(self.model, describe))

    def test_ordered_checkpoint_reordered_keys(self):
        self.run_apply(OrderedDict(reversed(list(self.cp.items()))))

    def test_scalar_zero_dimension_and_unicode_keys(self):
        self.cp = {"层.weight": Tensor((0, 3), "torch.int64", b"")}
        self.schema = {"层.weight": {"shape": [0, 3], "dtype": "torch.int64"}}
        self.model.state = OrderedDict({"层.weight": Tensor((0, 3), "torch.int64", b"")})
        self.run_apply(expected_schema=self.schema, expected_schema_sha256=digest(self.schema))

    def test_checkpoint_ordereddict_metadata_rejected(self):
        cp = OrderedDict(self.cp); cp._metadata = {}
        self.pre_reject(lambda: self.run_apply(cp))

    def test_missing_checkpoint_key(self):
        self.cp.pop("decoder.bias"); self.pre_reject(self.run_apply)

    def test_extra_checkpoint_key(self):
        self.cp["extra"] = Tensor((), "torch.float32", b"x"); self.pre_reject(self.run_apply)

    def test_same_count_different_name(self):
        self.cp["wrong.bias"] = self.cp.pop("decoder.bias"); self.pre_reject(self.run_apply)

    def test_architecture_missing_key(self):
        self.model.state.pop("decoder.bias"); self.pre_reject(self.run_apply)

    def test_architecture_extra_key(self):
        self.model.state["extra"] = Tensor((), "torch.float32", b"x"); self.pre_reject(self.run_apply)

    def test_checkpoint_wrong_shape(self):
        self.cp["decoder.bias"].shape = (1,); self.pre_reject(self.run_apply)

    def test_checkpoint_wrong_dtype(self):
        self.cp["decoder.bias"].dtype = "torch.float64"; self.pre_reject(self.run_apply)

    def test_architecture_wrong_shape(self):
        self.model.state["decoder.bias"].shape = (1,); self.pre_reject(self.run_apply)

    def test_architecture_wrong_dtype(self):
        self.model.state["decoder.bias"].dtype = "torch.float64"; self.pre_reject(self.run_apply)

    def test_nested_state_envelope(self):
        self.pre_reject(lambda: self.run_apply({"state_dict": self.cp}))

    def test_schema_hash_mismatch(self):
        self.pre_reject(lambda: self.run_apply(expected_schema_sha256="0" * 64))

    def test_schema_missing_key_even_rehashed(self):
        schema = {"encoder.weight": self.schema["encoder.weight"]}
        self.pre_reject(lambda: self.run_apply(expected_schema=schema, expected_schema_sha256=digest(schema)))

    def test_descriptor_exception_last_entry(self):
        def callback(t, include):
            if t.raw == b"new-a": raise TypeError("bad last entry")
            return describe(t, include)
        self.pre_reject(lambda: self.run_apply(descriptor=callback))

    def test_flag_false_throughout_prevalidation_and_readback(self):
        def callback(t, include):
            self.assertIs(self.model._kws_weights_verified, False)
            return describe(t, include)
        self.run_apply(descriptor=callback)

    def test_mutation_during_prevalidation(self):
        count = 0
        def callback(t, include):
            nonlocal count
            count += 1
            row = describe(t, include)
            if count == 4: self.cp["decoder.bias"].raw = b"changed"
            return row
        self.pre_reject(lambda: self.run_apply(descriptor=callback))

    def test_descriptor_does_not_leak_mutable_shape(self):
        returned = []
        def callback(t, include):
            for old in returned: old["shape"][:] = [999]
            row = describe(t, include); returned.append(row); return row
        self.run_apply(descriptor=callback)

    def test_success_then_prevalidation_failure_invalidates(self):
        self.run_apply(); self.model.loads.clear(); self.cp.pop("decoder.bias")
        self.pre_reject(self.run_apply)
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)

    def test_receipt_return_copy_cannot_mutate_seal(self):
        receipt = self.run_apply(); receipt["contract"]["checkpoint_state"].clear()
        require_verified_receipt(self.model, describe)

    def test_stored_receipt_tampering_invalidates(self):
        self.run_apply(); self.model._kws_checkpoint_receipt["execution_authorized"] = True
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)
        self.assertIs(self.model._kws_weights_verified, False)

    def test_copied_receipt_other_instance_fails(self):
        receipt = self.run_apply(); other = Model()
        other._kws_weights_verified = True; other._kws_checkpoint_receipt = receipt
        with self.assertRaises(CheckpointContractError): require_verified_receipt(other, describe)

    def test_boolean_only_cannot_pass(self):
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)

    def test_model_mutation_after_success_invalidates(self):
        self.run_apply(); self.model.state["decoder.bias"].raw += b"late"
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)
        self.assertIs(self.model._kws_weights_verified, False)

    def test_preinference_descriptor_exception_invalidates(self):
        self.run_apply()
        def fail(t, include): raise ValueError("broken readback")
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, fail)
        self.assertIs(self.model._kws_weights_verified, False)

    def test_runtime_imports_never_occur(self):
        self.assertFalse({"torch", "numpy", "funasr", "qwen_asr", "transformers"} & set(sys.modules))

    def test_none_checkpoint(self):
        self.pre_reject(lambda: exact_apply(self.model, None, describe, **self.options))

    def test_no_schema_lock_is_not_accepted(self):
        self.pre_reject(lambda: self.run_apply(expected_schema_sha256=None))

    def test_omitted_schema_lock_invalidates_before_rejecting(self):
        self.pre_reject(lambda: exact_apply(self.model, self.cp, describe))

    def test_omitted_preinference_descriptor_invalidates(self):
        self.run_apply()
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model)
        self.assertIs(self.model._kws_weights_verified, False)

    def test_unexpected_tensor_object(self):
        self.cp["decoder.bias"] = object(); self.pre_reject(self.run_apply)

    def test_success_then_postload_failure_invalidates(self):
        self.run_apply(); self.model.loads.clear(); self.model.mode = "wrong_content"
        self.post_reject(self.run_apply)
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)

    def test_readback_descriptor_exception(self):
        def callback(t, include):
            if self.model.loaded: raise ValueError("readback descriptor failed")
            return describe(t, include)
        self.post_reject(lambda: self.run_apply(descriptor=callback))

    def test_swapped_contents_are_rejected(self):
        original = self.model.load_state_dict
        def loader(cp, strict):
            r = original(cp, strict)
            left, right = self.model.state.values()
            left.raw, right.raw = right.raw, left.raw
            return r
        self.model.load_state_dict = loader
        self.post_reject(self.run_apply)

    def test_strict_keyword_rejection_has_no_fallback(self):
        def loader(cp): raise AssertionError("must not be called without strict")
        self.model.load_state_dict = loader
        with self.assertRaises(CheckpointContractError) as caught: self.run_apply()
        self.assertEqual(self.model.loads, [])
        self.assertTrue(caught.exception.mutation_may_have_occurred)
        self.assertIs(self.model._kws_weights_verified, False)

    def test_flag_clear_failure_prevents_weight_application(self):
        class FrozenFlagModel(Model):
            locked = False
            def __setattr__(self, name, value):
                if name == "_kws_weights_verified" and self.locked: raise RuntimeError("flag is immutable")
                super().__setattr__(name, value)
        self.model = FrozenFlagModel(); self.model.locked = True
        with self.assertRaises(CheckpointContractError) as caught: self.run_apply()
        self.assertEqual(self.model.loads, [])
        self.assertFalse(caught.exception.mutation_may_have_occurred)
        # Attribute cannot be cleared, but no registry seal exists and no load ran.
        with self.assertRaises(CheckpointContractError): require_verified_receipt(self.model, describe)

    def test_architecture_drift_before_application(self):
        calls = 0
        old = self.model.state_dict
        def state():
            nonlocal calls
            calls += 1
            if calls == 2: self.model.state["decoder.bias"].shape = (1,)
            return old()
        self.model.state_dict = state
        self.pre_reject(self.run_apply)

    def test_distinct_equal_hash_models_cannot_share_receipt(self):
        class EqualModel(Model):
            def __eq__(self, other): return isinstance(other, EqualModel)
            def __hash__(self): return 7
        self.model = EqualModel(); receipt = self.run_apply()
        other = EqualModel(); other.state = OrderedDict(self.model.state)
        other._kws_weights_verified = True; other._kws_checkpoint_receipt = receipt
        with self.assertRaises(CheckpointContractError): require_verified_receipt(other, describe)
        self.assertEqual(other.loads, [])
        self.assertIs(other._kws_weights_verified, False)
        require_verified_receipt(self.model, describe)

    def test_unhashable_model_success_and_invalidation(self):
        class Unhashable(Model):
            __hash__ = None
        self.model = Unhashable(); self.run_apply()
        require_verified_receipt(self.model, describe)
        self.model.loads.clear(); self.cp.pop("decoder.bias")
        self.pre_reject(self.run_apply)

    def test_equality_and_hash_never_called(self):
        class HostileEquality(Model):
            def __eq__(self, other): raise AssertionError("equality must not be consulted")
            def __hash__(self): raise AssertionError("hash must not be consulted")
        self.model = HostileEquality(); self.run_apply()
        require_verified_receipt(self.model, describe)
        self.model.loads.clear(); self.cp.pop("decoder.bias")
        self.pre_reject(self.run_apply)

    def test_equal_models_have_independent_valid_seals(self):
        class EqualModel(Model):
            def __eq__(self, other): return True
            def __hash__(self): return 1
        self.model = EqualModel(); self.run_apply()
        first = self.model; self.model = EqualModel(); self.run_apply()
        second = self.model
        require_verified_receipt(first, describe); require_verified_receipt(second, describe)
        self.model.loads.clear(); self.cp.pop("decoder.bias"); self.pre_reject(self.run_apply)
        require_verified_receipt(first, describe)

    def test_seal_cleanup_after_instance_deletion(self):
        from checkpoint_guard.exact_apply import _VERIFIED
        self.run_apply(); key = id(self.model); ref = weakref.ref(self.model)
        self.assertIn(key, _VERIFIED)
        self.model = None; gc.collect()
        self.assertIsNone(ref()); self.assertNotIn(key, _VERIFIED)

    def test_registry_entry_for_different_identity_rejected(self):
        from checkpoint_guard.exact_apply import _VERIFIED
        receipt = self.run_apply(); other = Model()
        other.state = OrderedDict(self.model.state)
        other._kws_weights_verified = True; other._kws_checkpoint_receipt = receipt
        # Synthetic ID-reuse/stale-entry adversary: weakref must still resolve to
        # the exact target object, independently of the integer registry key.
        _VERIFIED[id(other)] = _VERIFIED[id(self.model)]
        try:
            with self.assertRaises(CheckpointContractError): require_verified_receipt(other, describe)
            require_verified_receipt(self.model, describe)
        finally:
            _VERIFIED.pop(id(other), None)


def add_case(name, method):
    method.__name__ = name
    setattr(ContractTests, name, method)


class DictSubclass(dict): pass
class OrderedSubclass(OrderedDict): pass


for name, make in {
    "list": lambda cp: list(cp.items()), "empty": lambda cp: {},
    "userdict": lambda cp: UserDict(cp), "dict_subclass": lambda cp: DictSubclass(cp),
    "ordered_subclass": lambda cp: OrderedSubclass(cp), "integer": lambda cp: 1,
}.items():
    def test(self, make=make): self.pre_reject(lambda: self.run_apply(make(self.cp)))
    add_case("test_reject_checkpoint_type_" + name, test)

for name, value in {"bool": True, "negative": -1, "float": 2.0, "text": "2"}.items():
    def test(self, value=value):
        def callback(t, include):
            row = describe(t, include); row["shape"] = [value]; return row
        self.pre_reject(lambda: self.run_apply(descriptor=callback))
    add_case("test_reject_shape_dimension_" + name, test)

for name, change in {
    "missing_hash": lambda r: r.pop("sha256", None),
    "short_hash": lambda r: r.update(sha256="a" * 63),
    "uppercase_hash": lambda r: r.update(sha256="A" * 64),
    "extra_field": lambda r: r.update(extra=True),
    "bf16": lambda r: r.update(dtype="torch.bfloat16"),
    "shape_tuple": lambda r: r.update(shape=tuple(r["shape"])),
}.items():
    def test(self, change=change):
        def callback(t, include):
            row = describe(t, include)
            if include: change(row)
            return row
        self.pre_reject(lambda: self.run_apply(descriptor=callback))
    add_case("test_reject_descriptor_" + name, test)

for name, mapping in {"nonidentity": (("module.", ""),), "empty": (),
                      "list": [("", "")], "extra": (("", ""), ("x", "y"))}.items():
    def test(self, mapping=mapping): self.pre_reject(lambda: self.run_apply(prefix_mappings=mapping))
    add_case("test_reject_prefix_mapping_" + name, test)

for name, key in {"empty": "", "whitespace": "has space", "integer": 1,
                  "control": "bad\x00name"}.items():
    def test(self, key=key):
        self.cp[key] = self.cp.pop("decoder.bias"); self.pre_reject(self.run_apply)
    add_case("test_reject_key_" + name, test)

for mode in ("raise_before", "partial", "noop", "wrong_content", "wrong_shape",
             "wrong_dtype", "missing", "extra", "mutate_both", "mutate_input_only",
             "readback_exception"):
    def test(self, mode=mode):
        self.model.mode = mode; self.post_reject(self.run_apply)
    add_case("test_failed_loader_" + mode, test)

for name, value in {"none": None, "plain_tuple": ([], []), "dict": {},
                    "missing": Result(["x"], []), "unexpected": Result([], ["x"]),
                    "both": Result(["x"], ["y"]), "none_list": Result(None, []),
                    "tuple_list": Result((), [])}.items():
    def test(self, value=value):
        self.model.return_value = value; self.post_reject(self.run_apply)
    add_case("test_bad_load_return_" + name, test)

if __name__ == "__main__":
    unittest.main(verbosity=2)
