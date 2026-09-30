"""Two fixed-size causal research encoders; no product/runtime integration."""
import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.history = 4 * dilation
        self.depthwise = nn.Conv1d(48, 48, 5, dilation=dilation, groups=48)
        self.pointwise = nn.Conv1d(48, 48, 1)

    def chunk(self, x, cache=None):
        if cache is None:
            cache = x.new_zeros(x.shape[0], 48, self.history)
        joined = torch.cat([cache, x], dim=2)
        y = self.pointwise(F.relu(self.depthwise(joined)))
        return F.relu(x + y), joined[:, :, -self.history:]


class MemoryBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.history = 8 * dilation
        self.project = nn.Conv1d(48, 24, 1, bias=False)
        self.memory = nn.Conv1d(24, 24, 9, dilation=dilation, groups=24, bias=False)
        self.expand = nn.Conv1d(24, 48, 1)

    def chunk(self, x, cache=None):
        z = self.project(x)
        if cache is None:
            cache = z.new_zeros(z.shape[0], 24, self.history)
        joined = torch.cat([cache, z], dim=2)
        y = z + self.memory(joined)
        return F.relu(x + self.expand(y)), joined[:, :, -self.history:]


class CausalClassifier(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture not in {"A", "B"}:
            raise ValueError("architecture must be A or B")
        self.architecture = architecture
        self.stem = nn.Conv1d(32, 48, 1)
        self.blocks = nn.ModuleList(
            [DepthwiseBlock(d) for d in (1, 2, 4, 8, 16)] if architecture == "A"
            else [MemoryBlock(d) for d in (1, 2, 4, 8)])
        self.head = nn.Conv1d(48, 2, 1)

    def chunk(self, x, caches=None):
        """Input B,T,32; return B,T,2 plus bounded causal state. None resets."""
        if x.ndim != 3 or x.shape[2] != 32 or x.shape[1] == 0:
            raise ValueError("expected nonempty B,T,32")
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError("invalid number of state caches")
        h = F.relu(self.stem(x.transpose(1, 2)))
        states = []
        for block, cache in zip(self.blocks, caches):
            h, state = block.chunk(h, cache)
            states.append(state)
        return self.head(h).transpose(1, 2), states

    def forward(self, x):
        return self.chunk(x, None)[0]


def masked_max(logits, lengths):
    if logits.ndim != 3 or logits.shape[2] != 2:
        raise ValueError("expected B,T,2 logits")
    if lengths.ndim != 1 or lengths.shape[0] != logits.shape[0] or lengths.dtype != torch.long:
        raise ValueError("expected int64 lengths per row")
    if bool(((lengths <= 0) | (lengths > logits.shape[1])).any()):
        raise ValueError("length outside valid frames")
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    return logits.masked_fill(~mask[:, :, None], float("-inf")).max(dim=1).values


def clip_loss(logits, lengths, targets):
    if targets.shape != (logits.shape[0], 2) or not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("expected binary two-keyword clip labels")
    return F.binary_cross_entropy_with_logits(masked_max(logits, lengths), targets)
