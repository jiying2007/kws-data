"""Offline adversarial fixtures for the independent FP32/CTC readback only."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

path = Path(__file__).with_name("independent_primary_output_audit.py")
spec = importlib.util.spec_from_file_location("b21_independent_audit_tested", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def npy(values, shape, dtype="<f4"):
    header = (repr(dict(descr=dtype, fortran_order=False, shape=shape)) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + struct.pack("<" + "f" * len(values), *values)


class OutputReadbackTests(unittest.TestCase):
    def test_ties_choose_the_first_index(self):
        raw = npy([2, 2, 2, 2, -1, 3, 3, 1], (2, 4))
        self.assertEqual(audit.independent_npy(raw, (2, 4))[1], [0, 1])

    def test_raw_batch_and_saved_matrix_match(self):
        values = [1, 2, 3, 4, 4, 3, 2, 1]
        self.assertEqual(audit.independent_npy(npy(values, (1, 2, 4)), (1, 2, 4)),
                         audit.independent_npy(npy(values, (2, 4)), (2, 4)))

    def test_truncated_or_extended_body_is_rejected(self):
        raw = npy([0, 1, 2, 3], (1, 4))
        for candidate in (raw[:-1], raw + b"\0"):
            with self.subTest(length=len(candidate)), self.assertRaises(ValueError):
                audit.independent_npy(candidate, (1, 4))

    def test_wrong_shape_or_dtype_is_rejected(self):
        for candidate in (npy([0, 1, 2, 3], (1, 1, 4)), npy([0, 1, 2, 3], (1, 4), ">f4")):
            with self.assertRaises(ValueError):
                audit.independent_npy(candidate, (1, 4))

    def test_nonfinite_frame_is_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.independent_npy(npy([0, 1, value, 3], (1, 4)), (1, 4))

    def test_ctc_blank_repetition_and_event_clock(self):
        result = audit.independent_decode([0, 1, 1, 0, 1, 2, 2, 0, 3, 0, 1, 2, 3])
        self.assertEqual(result, dict(tokens=[1, 1, 2, 3, 1, 2, 3], event_frames=[8, 12],
                                     event_samples=[1792, 2432], events=2))

    def test_no_acoustic_or_array_runtime_import(self):
        self.assertFalse({"torch", "numpy", "onnxruntime"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
