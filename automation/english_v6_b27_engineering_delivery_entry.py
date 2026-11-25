"""Inert B27 checkout identity adapter for the already reviewed pipeline runner."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "english_v6_b24r1_delivery_entry.py"
ORIGINAL_SHA = "bc8369986679d74599a023cb460dcde80ce49ba9f2159d6fec4c7d491254bf23"
if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
    raise ValueError("unchanged original explicit delivery-entry contract")
definition = importlib.util.spec_from_file_location("q1_original_delivery_entry", ORIGINAL)
original = importlib.util.module_from_spec(definition)
definition.loader.exec_module(original)
SPEC = HERE / "english_v6_b27_engineering_delivery_inputs.json"
SCHEMA = "english-v6-b27-software-delivery-entry-v1"
original.SPEC, original.SCHEMA = SPEC, SCHEMA
contract = original.contract

if __name__ == "__main__":
    original.main()
