"""The FFN network: a 2-in / 1-out 3-D full-pre-activation ResNet.

Faithful to Januszewski et al. 2018 (Nat. Methods 15:605), Fig. 5:
    input module: Conv(2->F) - ReLU - Conv(F->F)
    depth x  full pre-activation residual module: y = x + Conv(ReLU(Conv(ReLU(x))))
    head: Conv(F->1, 1x1x1)  -> single-channel logit map, SAME size

At depth=8, F=32 the parameter count is exactly 472,353 (paper's number).
No batch-norm (FFN uses none; it keeps recurrent statistics stable).

The network consumes 2 channels [image, POM] and returns POM logits for the FOV.
See DESIGN.md 4.2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv3(cin: int, cout: int, k: int = 3) -> nn.Conv3d:
    return nn.Conv3d(cin, cout, kernel_size=k, padding=k // 2, bias=True)


class ResModule(nn.Module):
    """Residual module whose SKIP PATH is selectable, because this file has never had the one its
    docstring claimed.

    `nn.ReLU(inplace=True)` mutates its input. `self.act(x)` therefore overwrote the very tensor the
    skip added back, so `x + y` was really `relu(x) + y`. Verified numerically: the old forward
    matches `relu(x) + F(x)` to 0.0 and differs from `x + F(x)` by 2.90 on random input. The
    parameter count is unaffected (still 472,353), which is why `test_model_pom.py` -- which only
    asserts the count -- never saw it.

    Substituting a = relu(x) shows the old block is `a_{k+1} = relu(a_k + F(a_k))`, i.e. exactly a
    POST-activation ResNet-v1 block. So the bug silently traded v2 for v1; both are valid, and the
    v2 advantage is a very-deep-network phenomenon, which is why depth-8 training worked at all.

    The reference google/ffn `convstack_3d._predict_object_mask` captures `in_net = net` BEFORE its
    relu, i.e. a true identity skip (v2). `skip_alpha` interpolates between the two:

        alpha = 0  ->  relu(x) + F(x)     historical behaviour, bit-for-bit
        alpha = 1  ->  x + F(x)           the reference architecture

    Default 0.0 so every existing checkpoint keeps its exact function. Ramping alpha 0 -> 1 during
    training is a continuous homotopy onto the reference block, which is the only way to adopt it
    without the measured 55%-of-voxels decision flip a hard switch causes. It also matters for depth
    growth: zero-init block insertion is exact at alpha=1 and destructive at alpha=0 (44.9% flipped).
    """

    def __init__(self, f: int, skip_alpha: float = 0.0):
        super().__init__()
        self.conv1 = _conv3(f, f)
        self.conv2 = _conv3(f, f)
        # plain attribute, NOT a buffer: a buffer would add a state_dict key and break loading of
        # every checkpoint trained before this change.
        self.skip_alpha = float(skip_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)                      # non-in-place: the skip below must still see `x`
        y = self.conv2(torch.relu(self.conv1(a)))
        skip = a if self.skip_alpha == 0.0 else a + self.skip_alpha * (x - a)
        return skip + y


class FFNModel(nn.Module):
    """Recurrent FFN body. One forward = one flood-fill step's POM update."""

    def __init__(self, fmaps: int = 32, depth: int = 8, in_channels: int = 2,
                 head_bias_init: float | None = None, skip_alpha: float = 0.0,
                 move_head: bool = False, fov: int = 33, delta: int = 8):
        super().__init__()
        self.fmaps = fmaps
        self.depth = depth
        # input module: Conv - ReLU - Conv
        self.in_conv1 = _conv3(in_channels, fmaps)
        self.in_conv2 = _conv3(fmaps, fmaps)
        # in-place is safe HERE (in_conv1's output is not reused), unlike inside ResModule
        self.act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResModule(fmaps, skip_alpha) for _ in range(depth)])
        self.head = nn.Conv3d(fmaps, 1, kernel_size=1, bias=True)
        # Prior-bias init on the head (DESIGN.md R7): start biased toward "not me"
        # so the recurrence does not collapse into an early merge.
        if head_bias_init is not None:
            nn.init.constant_(self.head.bias, head_bias_init)
        # Learned movement gate (research/13 Lever #2). When enabled, forward returns
        # (logits, move_logits) -- a FIXED signature for the run, so torch.compile sees no dynamism.
        self.fov, self.delta = int(fov), int(delta)
        self.move_head = None
        if move_head:
            from .movement import MoveHead
            self.move_head = MoveHead(fmaps)

    def forward(self, x: torch.Tensor):
        """x: [B, C, D, H, W] -> logits [B, 1, D, H, W] (SAME size).

        With `move_head` enabled returns (logits, move_logits[B,6]) instead."""
        h = self.in_conv2(self.act(self.in_conv1(x)))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        if self.move_head is not None:
            return out, self.move_head(h, self.fov, self.delta)
        return out

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, head_bias_init: float | None = None) -> FFNModel:
    from math import log
    if head_bias_init is None:
        # logit(tgt_lo): background prior
        head_bias_init = log(cfg.tgt_lo / (1.0 - cfg.tgt_lo))
    return FFNModel(fmaps=cfg.fmaps, depth=cfg.depth,
                    in_channels=cfg.in_channels, head_bias_init=head_bias_init,
                    skip_alpha=getattr(cfg, "skip_alpha", 0.0),
                    move_head=bool(getattr(cfg, "move_head", False)),
                    fov=cfg.fov, delta=cfg.delta)


def grow_in_channels(model, in_channels: int) -> bool:
    """Grow `in_conv1` to accept more input channels, ZERO-INITIALISING the new ones.

    Function-preserving by construction: a zero weight slice contributes exactly 0 to the
    convolution, so the model computes a BIT-IDENTICAL output at step 0 and a checkpoint trained
    with 2 channels can be fine-tuned with 3 rather than needing a from-scratch run. Returns True
    if it grew. (12_next_moves item 4 requires exactly this discipline for any state-dict growth.)
    """
    m = getattr(model, "module", model)
    old_conv = m.in_conv1
    w = old_conv.weight
    old = w.shape[1]
    if in_channels <= old:
        return False
    with torch.no_grad():
        new_conv = _conv3(in_channels, m.fmaps).to(w.device, w.dtype)
        new_conv.weight.zero_()                       # new channels contribute exactly 0
        new_conv.weight[:, :old].copy_(w)             # existing channels verbatim
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)        # bias is per-OUTPUT: carry it over unchanged
        m.in_conv1 = new_conv
    return True


def set_skip_alpha(model, alpha: float) -> None:
    """Set the residual skip homotopy on every block (see ResModule)."""
    m = getattr(model, "module", model)
    for blk in m.blocks:
        blk.skip_alpha = float(alpha)
