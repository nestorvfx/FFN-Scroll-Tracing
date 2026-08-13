"""FOV movement policy.

Training-time movement (this module) is batched: from the current per-example
walk offset it evaluates the 6 face candidates at +/-delta, gates them by the
POM probability and, during the teacher-forced early curriculum, restricts to
candidates that stay inside the seed instance. The best-scoring valid candidate
is taken; otherwise the FOV stays put. The walk is bounded to +/-walk_radius so
every FOV stays in the per-example canvas (DESIGN.md 5.3, 6.1).

TWO GATE POLICIES (cfg.move_gate):

  "center"  -- the historical implementation: read the POM (and, under teacher
               forcing, the instance label) at the SINGLE voxel
               `centre + delta*e_i`.
  "facemax" -- Januszewski et al. / google-ffn: take the MAX over the whole
               `fov x fov` FACE PLANE at `+/-delta` along that axis.

Why the difference is large here, MEASURED model-free (move_reach.py, on real
training coordinates, teacher-forced, delta=8, walk_radius=16):

    policy      no first step   reachable walk-lattice   instance covered by
                                nodes                    the union of reached FOVs
    center          19.5%              6.5%                     55.9%
    facemax          0.0%             69.9%                     99.5%

A wrap ribbon's normal n is radial and roughly in-plane, so an axis step of
delta leaves a ~5-vox ribbon unless |n.e_i| <= t/(2*delta); at oblique azimuths
BOTH in-plane steps land off the ribbon and only +/-z survives. Face-max does
not have this problem because the ribbon always intersects the face PLANE
somewhere. Under "center" the trajectory therefore degenerates to a nearly
stationary FOV and the network is never asked to extend a mask -- the canonical
FFN failure mode.

NOTE the two policies are NOT interchangeable at inference with fixed weights: a
model trained under "center" has never been evaluated at the FOV positions
face-max opens up. (The old headline "merge 0.065 -> 0.630" is RETRACTED -- it was
an artifact of the since-fixed 33x33 face-size bug; re-measured: merge
0.0055 -> 0.120 with NERL improving 0.120 -> 0.179. See FINDINGS 3.8.) Train and
infer must still match; the shipped run-5b/run-6 checkpoints trained facemax.

Inference-time movement is a priority queue over face candidates and lives in
inference.py (it needs the full committed-segment canvas), but shares the
face-offset convention defined here.
"""
from __future__ import annotations

import torch

# 6 face directions (unit); scaled by delta at use.
FACE_DIRS = torch.tensor([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=torch.long)

_FACE_CACHE: dict = {}


def face_offsets(fov: int, delta: int, device) -> torch.Tensor:
    """[6, (2*delta+1)^2, 3] voxel offsets, relative to the FOV CENTRE, of the CUBOID FACE
    at +/-delta along each of the 6 axis directions.

    The face is (2*delta+1)^2, NOT fov^2. Januszewski et al. 2018 Methods: "A cuboid of POM values
    (x-dx <= x <= x+dx ...) ... the maximum value was identified on every one of its faces." Using the
    full fov x fov plane reaches +/-fov//2 transversely -- at fov=33 that is +/-16 voxels, ~1.7x the
    nearest-lamina spacing on the Scroll-4 core (lambda_nn 9.5), so the max would routinely be read on a
    DIFFERENT sheet. That would make the gate a merge generator rather than a movement rule."""
    key = (int(fov), int(delta), str(device))
    if key in _FACE_CACHE:
        return _FACE_CACHE[key]
    h = int(delta)
    r = torch.arange(-h, h + 1, device=device)
    gb, gc = torch.meshgrid(r, r, indexing="ij")
    gb, gc = gb.reshape(-1), gc.reshape(-1)
    z = torch.zeros_like(gb)
    out = []
    for d in FACE_DIRS.tolist():
        ax = max(range(3), key=lambda i: abs(d[i]))
        off = torch.stack([z, z, z], 1).clone()
        off[:, ax] = d[ax] * delta
        others = [i for i in range(3) if i != ax]
        off[:, others[0]] = gb
        off[:, others[1]] = gc
        out.append(off)
    t = torch.stack(out)
    _FACE_CACHE[key] = t
    return t


def _gather_voxel(vol_stack: torch.Tensor, vol_idx: torch.Tensor,
                  coords: torch.Tensor) -> torch.Tensor:
    """vol_stack [N,Z,Y,X], vol_idx [B], coords [B,K,3] -> [B,K] values."""
    B, K, _ = coords.shape
    Z, Y, X = vol_stack.shape[-3:]
    z = coords[..., 0].clamp(0, Z - 1)
    y = coords[..., 1].clamp(0, Y - 1)
    x = coords[..., 2].clamp(0, X - 1)
    vi = vol_idx.view(B, 1).expand(B, K)
    return vol_stack[vi, z, y, x]


def lattice_index(offset: torch.Tensor, delta: int, walk_radius: int) -> torch.Tensor:
    """Flat index of a walk offset on the (2n+1)^3 lattice, n = walk_radius//delta."""
    n = max(1, walk_radius // delta)
    q = (offset // delta + n).clamp_(0, 2 * n)
    s = 2 * n + 1
    return (q[:, 0] * s + q[:, 1]) * s + q[:, 2]


def training_move(offset: torch.Tensor, canvas: torch.Tensor, canvas_center: int,
                  delta: int, walk_radius: int, move_threshold: float,
                  inst_stack: torch.Tensor, vol_idx: torch.Tensor,
                  seed_abs: torch.Tensor, seed_instance: torch.Tensor,
                  teacher_force: bool, gate: str = "center",
                  fov: int = 33, visited: torch.Tensor | None = None,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """Return the next walk offset [B,3] for each example.

    offset:        [B,3] current displacement from the seed (vox)
    canvas:        [B,1,C,C,C] POM logits
    canvas_center: C//2
    inst_stack:    [N,Z,Y,X] instance labels (for teacher forcing / bounds)
    seed_abs:      [B,3] seed center in cube coords
    seed_instance: [B] the instance id being grown
    gate:          "center" (single voxel) or "facemax" (max over the face plane)
    visited:       [B,(2n+1)^3] bool, nodes already occupied by this trajectory. Without it the
                   walk revisits nodes and the FOV ping-pongs between two positions.
    gen:           RNG for the tie-break (see below)

    WHY THE TIE-BREAK MATTERS. `canvas` holds logits clamped to +-logit_clamp, so every face that
    clears the threshold reads sigmoid(6.9) = 0.99899... -- identical to float precision. A plain
    argmax therefore always returns index 0, i.e. +z, and the trained POM is essentially never
    extended in x/y. Measured over T=8: 84% of first moves were +-z and 1.6% were +-x/y, across only
    3.62 distinct lattice nodes of 9. The reference avoids this structurally (google/ffn queues ALL
    above-threshold faces via a `done`-set deque; the fixed policy walks a SHUFFLED offset list).
    """
    device = canvas.device
    B = offset.shape[0]
    dirs = FACE_DIRS.to(device) * delta          # [6,3]
    cand = offset.unsqueeze(1) + dirs.unsqueeze(0)  # [B,6,3]

    in_bounds = (cand.abs() <= walk_radius).all(dim=-1)   # [B,6]
    idx = torch.arange(B, device=device)

    if gate == "facemax":
        fo = face_offsets(fov, delta, device)             # [6,P,3]
        P = fo.shape[1]
        planes = offset.view(B, 1, 1, 3) + fo.unsqueeze(0)          # [B,6,P,3]
        loc = (planes + canvas_center).reshape(B, 6 * P, 3)
        pom_prob = torch.sigmoid(_gather_voxel(canvas[:, 0], idx, loc)
                                 ).view(B, 6, P).amax(dim=2)        # [B,6]
    else:
        local = cand + canvas_center                      # [B,6,3]
        pom_prob = torch.sigmoid(_gather_voxel(canvas[:, 0], idx, local))

    valid = in_bounds & (pom_prob >= move_threshold)
    if visited is not None:                      # never re-occupy a node this trajectory has used
        nxt = lattice_index(cand.reshape(B * 6, 3), delta, walk_radius).view(B, 6)
        valid = valid & ~torch.gather(visited, 1, nxt)

    if teacher_force:
        if gate == "facemax":
            abs_p = (seed_abs.view(B, 1, 1, 3) + planes).reshape(B, 6 * P, 3)
            inst_val = _gather_voxel(inst_stack, vol_idx, abs_p).view(B, 6, P)
            in_inst = (inst_val == seed_instance.view(B, 1, 1)).any(dim=2)
        else:
            abs_c = seed_abs.unsqueeze(1) + cand           # [B,6,3]
            in_inst = (_gather_voxel(inst_stack, vol_idx, abs_c)
                       == seed_instance.unsqueeze(1))
        valid = valid & in_inst

    # score = pom prob, masked to valid; if none valid, stay.
    # The jitter is what breaks the saturation tie (see docstring); it is far smaller than any real
    # POM difference, so a genuinely more confident face still wins.
    jitter = torch.rand(pom_prob.shape, device=device, generator=gen) * 1e-3
    score = torch.where(valid, pom_prob + jitter, torch.full_like(pom_prob, -1.0))
    best = score.argmax(dim=1)                            # [B]
    any_valid = valid.any(dim=1)                          # [B]
    new_off = torch.where(any_valid.unsqueeze(1),
                          cand[idx, best],
                          offset)
    if visited is not None:
        visited.scatter_(1, lattice_index(new_off, delta, walk_radius).unsqueeze(1), True)
    return new_off


def face_slabs(t: torch.Tensor, fov: int, delta: int) -> torch.Tensor:
    """Extract the 6 cuboid-face slabs from an FOV-sized tensor.

    t: [B, C, fov, fov, fov] -> [B, C, 6, (2*delta+1)^2], in FACE_DIRS order
    (+z, -z, +y, -y, +x, -x). The slab is the SAME region `face_offsets` gathers, so a head built on
    this sees exactly what the face-max gate reads -- and it lies entirely inside the FOV
    (fov 33, centre 16, delta 6 -> indices 10..22), so both the features and the labels are already
    in hand: no extra gather, no extra data.
    """
    c = fov // 2
    d = int(delta)
    s = slice(c - d, c + d + 1)
    f = [t[:, :, c + d, s, s], t[:, :, c - d, s, s],
         t[:, :, s, c + d, s], t[:, :, s, c - d, s],
         t[:, :, s, s, c + d], t[:, :, s, s, c - d]]
    return torch.stack(f, dim=2).flatten(3)          # [B,C,6,S]


class MoveHead(torch.nn.Module):
    """Learned movement gate (research/13 Lever #2): 6 face logits from the final feature map.

    WHY. The deployed gate is `max over the 169-voxel face >= 0.9` -- an EXTREME-VALUE statistic on
    a quantity trained under a MEAN criterion. For that gate to be even 50/50 safe at a blind
    contact, the per-voxel P(p>=0.9) on the neighbouring lamina must be below 1-0.5^(1/169) = 0.41%;
    the measured mean there is 0.45. No achievable improvement in a mean satisfies a tail that
    strict, which is why 200k steps moved the merge rate by exactly 0.0000 while changing the gate
    RULE alone (fixed weights) moves it 0.0055 -> 0.120.

    Pooling mean AND max over the slab turns that 169-fold test into ONE pooled, calibratable
    decision -- a mean-like requirement the model nearly meets already (blind-contact AUC 0.819:
    poor for a 169-fold extreme-value test, adequate for a single decision). It also gives a second
    knob, so merge control stops being the same scalar as growth control: abstaining at the GATE is
    paid in splits (cheap), abstaining at COMMIT discards voxels forever (measured: tau=0.921 halves
    coverage and the gap does not close as seeding doubles).
    """

    def __init__(self, fmaps: int, emb: int = 8, bias_init: float = -2.197):
        super().__init__()
        self.dir_emb = torch.nn.Parameter(torch.zeros(6, emb))
        torch.nn.init.normal_(self.dir_emb, std=0.02)
        self.fc = torch.nn.Linear(2 * fmaps + emb, 1)
        torch.nn.init.zeros_(self.fc.weight)
        # PRIOR-BIAS INIT, same argument as the POM head's (DESIGN R7: "start biased toward not-me
        # so the recurrence does not collapse into an early merge"). Zero-init would emit logit 0,
        # i.e. p = 0.5 exactly -- and the gate test is `p >= move_threshold_learned` with a 0.5
        # default, so an UNTRAINED head would open all six directions unconditionally and the fill
        # would run away. logit(0.1) starts the gate CLOSED; the aux BCE opens it where the label
        # says the sheet continues.
        torch.nn.init.constant_(self.fc.bias, float(bias_init))

    def forward(self, h: torch.Tensor, fov: int, delta: int) -> torch.Tensor:
        sl = face_slabs(h, fov, delta)               # [B,C,6,S]
        feat = torch.cat([sl.mean(-1), sl.amax(-1)], dim=1)      # [B,2C,6]
        feat = feat.permute(0, 2, 1)                             # [B,6,2C]
        e = self.dir_emb.unsqueeze(0).expand(feat.shape[0], -1, -1)
        return self.fc(torch.cat([feat, e], dim=-1)).squeeze(-1)  # [B,6]


def face_in_instance(inst_fov: torch.Tensor, seed_inst: torch.Tensor,
                     fov: int, delta: int) -> torch.Tensor:
    """The movement target, which `training_move` ALREADY computes and throws away (see the
    teacher_force branch): does the face slab contain any of the SEED instance?

    inst_fov: [B,1,fov,fov,fov] instance ids; seed_inst: [B]. Returns [B,6] float in {0,1}.
    """
    sl = face_slabs(inst_fov, fov, delta)                          # [B,1,6,S]
    return (sl == seed_inst.view(-1, 1, 1, 1)).any(-1).squeeze(1).float()
