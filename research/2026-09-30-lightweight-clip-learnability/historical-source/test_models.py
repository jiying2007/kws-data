import unittest
import torch
from models import CausalClassifier, masked_max, clip_loss

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)


class CausalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_exact_parameter_and_cache_budgets(self):
        for arch, parameters, floats in [("A", 14882, 5952), ("B", 11954, 2880)]:
            model = CausalClassifier(arch)
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
            _, caches = model.chunk(torch.randn(1, 9, 32))
            self.assertEqual(sum(c.numel() for c in caches), floats)

    def test_future_frames_never_affect_past(self):
        for arch in ("A", "B"):
            model = CausalClassifier(arch).eval()
            x = torch.randn(1, 151, 32)
            y = model(x)
            modified = x.clone()
            modified[:, 70:] = torch.randn_like(modified[:, 70:]) * 100
            torch.testing.assert_close(y[:, :70], model(modified)[:, :70], atol=0, rtol=0)

    def test_frame_and_uneven_chunk_agree_with_sequence(self):
        for arch in ("A", "B"):
            model = CausalClassifier(arch).eval()
            x = torch.randn(2, 173, 32)
            expected = model(x)
            for sizes in ([1] * 173, [3, 17, 1, 64, 88]):
                cache, outputs, offset = None, [], 0
                for size in sizes:
                    y, cache = model.chunk(x[:, offset:offset + size], cache)
                    outputs.append(y)
                    offset += size
                torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=2e-6, rtol=2e-5)

    def test_reset_and_multiple_wraps(self):
        for arch in ("A", "B"):
            model = CausalClassifier(arch).eval()
            x = torch.randn(1, 401, 32)
            first, state = model.chunk(x)
            _, contaminated = model.chunk(torch.randn(1, 83, 32), state)
            reset, _ = model.chunk(x, None)
            torch.testing.assert_close(first, reset, atol=0, rtol=0)
            self.assertEqual(len(contaminated), len(model.blocks))

    def test_mask_ignores_padding_and_its_gradient(self):
        logits = torch.tensor([[[1., 2.], [3., 0.], [1e6, 1e6]],
                               [[4., 5.], [1e6, 1e6], [1e6, 1e6]]], requires_grad=True)
        lengths = torch.tensor([2, 1])
        pooled = masked_max(logits, lengths)
        torch.testing.assert_close(pooled, torch.tensor([[3., 2.], [4., 5.]]))
        clip_loss(logits, lengths, torch.tensor([[1., 0.], [0., 0.]])).backward()
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[1, 1:].abs().sum()), 0.)

    def test_batch_padding_cannot_change_valid_logits(self):
        for arch in ("A", "B"):
            model = CausalClassifier(arch).eval()
            x = torch.randn(1, 51, 32)
            padded = torch.cat([x, torch.randn(1, 33, 32) * 100], 1)
            torch.testing.assert_close(model(x), model(padded)[:, :51], atol=0, rtol=0)

    def test_invalid_masks_rejected(self):
        for lengths in (torch.tensor([0]), torch.tensor([4]), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                masked_max(torch.zeros(1, 3, 2), lengths)

    def test_gradients_finite_for_both(self):
        for arch in ("A", "B"):
            model = CausalClassifier(arch)
            loss = clip_loss(model(torch.randn(4, 133, 32)), torch.tensor([133, 99, 83, 51]),
                             torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
