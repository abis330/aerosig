"""Cross-channel masked-patch Transformer (PatchTST-style patching + channel tokens).

Each (channel, patch) pair is a token. Pre-training hides
  * random patches (temporal context)                -> learns dynamics
  * entire channels for the whole window             -> learns *physical relations* between sensors
and reconstructs them. A channel that cannot be explained by the rest of the aircraft state
(an rpm that is zero while throttle is high, an airspeed that disagrees with ground speed +
attitude + power, GNSS innovations that grow while satellites look healthy) produces large
reconstruction error -> anomaly score with built-in per-channel attribution.
"""
from __future__ import annotations

import torch
from torch import nn


class CrossChannelPatchTST(nn.Module):
    def __init__(self, n_ch: int, win: int = 128, patch: int = 16, d: int = 64, depth: int = 3, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert win % patch == 0
        self.n_ch, self.win, self.patch, self.n_patch, self.d = n_ch, win, patch, win // patch, d
        self.embed = nn.Linear(patch, d)
        self.ch_emb = nn.Parameter(torch.randn(n_ch, d) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(self.n_patch, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, T); mask: (B, C, P) bool, True = hidden. Returns (B, C, P, d)."""
        B, C, _ = x.shape
        # non-overlapping patches: a reshape (== unfold with step=patch) exports cleanly to ONNX
        z = self.embed(x.reshape(B, C, self.n_patch, self.patch))  # B,C,P,d
        if mask is not None:
            z = torch.where(mask[..., None], self.mask_tok.expand_as(z), z)
        z = z + self.ch_emb[None, :, None, :] + self.pos_emb[None, None, :, :]
        h = self.enc(z.reshape(B, C * self.n_patch, self.d))
        return self.norm(h).reshape(B, C, self.n_patch, self.d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruction (B, C, T)."""
        B, C, _ = x.shape
        return self.head(self.encode(x, mask)).reshape(B, C, self.win)

    @torch.no_grad()
    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Window embedding (B, C*d): per-channel mean-pooled tokens, unmasked."""
        return self.encode(x).mean(dim=2).flatten(1)


def make_mask(B: int, C: int, P: int, p_patch: float = 0.25, max_ch: int = 6, gen=None) -> torch.Tensor:
    """Mixed temporal + channel masking used during pre-training."""
    m = torch.rand(B, C, P, generator=gen) < p_patch
    k = torch.randint(1, max_ch + 1, (B,), generator=gen)
    scores = torch.rand(B, C, generator=gen)
    ranks = scores.argsort(dim=1).argsort(dim=1)
    ch = ranks < k[:, None]
    return m | ch[:, :, None]


def group_masks(C: int, P: int, n_groups: int) -> list[torch.Tensor]:
    """Deterministic interleaved channel groups used at inference: channel i -> group i % n_groups."""
    out = []
    for g in range(n_groups):
        m = torch.zeros(C, P, dtype=torch.bool)
        m[g::n_groups] = True
        out.append(m)
    return out
