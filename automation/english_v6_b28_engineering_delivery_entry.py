"""B28 checkout identity bridge using the exact frozen B27-era entry contract.

The entry names only a reviewed public pipeline commit and complete file pins.
Actual B28 dual-quality and engineering evidence, the candidate identity and
immutable B0 rollback remain mandatory in that pipeline's bound manifest.
The committed null registration cannot select a checkout or start delivery.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "english_v6_b24r1_delivery_entry.py"
ORIGINAL_SHA = "bc8369986679d74599a023cb460dcde80ce49ba9f2159d6fec4c7d491254bf23"
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError("unchanged original explicit delivery-entry contract")

definition = importlib.util.spec_from_file_location("b28_frozen_delivery_entry", ORIGINAL)
original = importlib.util.module_from_spec(definition)
definition.loader.exec_module(original)

SPEC = HERE / "english_v6_b28_engineering_delivery_inputs.json"
SCHEMA = "english-v6-b28-software-delivery-entry-v1"

# Reuse both exact code objects in a private namespace. No existing reader,
# original contract global, model identity or execution guard is changed.
_namespace = dict(vars(original), SPEC=SPEC, SCHEMA=SCHEMA)
contract = FunctionType(original.contract.__code__, _namespace, "contract",
                        original.contract.__defaults__, original.contract.__closure__)
_namespace["contract"] = contract
main = FunctionType(original.main.__code__, _namespace, "main",
                    original.main.__defaults__, original.main.__closure__)

if __name__ == "__main__":
    main()
