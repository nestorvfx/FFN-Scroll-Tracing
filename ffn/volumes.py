"""GPU-resident volume cache, balanced sampler, and on-GPU augmentation.

A CPU-bound data pipeline is the known FFN failure mode (DESIGN.md 1, brief).
We avoid it entirely: all training cubes live as tensors on the GPU (image
uint8, instance int16 -- ~7 GB for 340 cubes), and every training example is a
single batched gather from that resident stack. Augmentation (octahedral
symmetry, intensity jitter, noise) runs on the GPU too. No dataloader workers,
no host round-trips inside the recurrent unroll.

The candidate-center index (skeleton voxels + partition bins + hard-neg flags)
is produced offline by scripts/preprocess.py and loaded here as flat GPU
tensors; the sampler draws locations balanced across the 17 partition bins with
an optional hard-negative fraction for Phase C.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .pom import batched_crop, make_offset_grid


class GpuVolumeCache:
    def __init__(self, image_stack: torch.Tensor, inst_stack: torch.Tensor,
                 index: dict, cfg, device):
        """image_stack [N,Z,Y,X] uint8 ; inst_stack [N,Z,Y,X] int16 (both on device).
        index: flat candidate arrays with keys vol,z,y,x,inst,pbin,hard.
        """
        self.image = image_stack
        self.inst = inst_stack
        self.cfg = cfg
        self.device = device
        self.patch = cfg.fov + 2 * cfg.walk_radius
        self._pgrid = make_offset_grid(self.patch, image_stack.device)
        self._check_index_margin(index, image_stack.shape[-3:])

        self.vol = torch.as_tensor(index["vol"], dtype=torch.long, device=device)
        self.ctr = torch.stack([
            torch.as_tensor(index["z"], dtype=torch.long, device=device),
            torch.as_tensor(index["y"], dtype=torch.long, device=device),
            torch.as_tensor(index["x"], dtype=torch.long, device=device)], dim=1)
        self.inst_id = torch.as_tensor(index["inst"], dtype=torch.long, device=device)
        self.pbin = torch.as_tensor(index["pbin"], dtype=torch.long, device=device)
        self.hard = torch.as_tensor(index["hard"], dtype=torch.bool, device=device)
        is_std = index.get("is_std")
        if is_std is None:
            is_std = np.ones(self.vol.shape[0], dtype=bool)
        self.is_std = torch.as_tensor(is_std, dtype=torch.bool, device=device)
        is_real = index.get("is_real")
        if is_real is None:
            is_real = np.zeros(self.vol.shape[0], dtype=bool)
        self.is_real = torch.as_tensor(is_real, dtype=torch.bool, device=device)

        # per-bin candidate pools (for balanced sampling), all + std-only
        self.bin_pools, self.bin_pools_std = [], []
        nbins = len(cfg.partition_bins) - 1
        for b in range(nbins):
            m = self.pbin == b
            self.bin_pools.append(torch.nonzero(m, as_tuple=False).squeeze(1))
            self.bin_pools_std.append(torch.nonzero(m & self.is_std, as_tuple=False).squeeze(1))
        self.hard_pool = torch.nonzero(self.hard, as_tuple=False).squeeze(1)
        self.real_pool = torch.nonzero(self.is_real, as_tuple=False).squeeze(1)  # AUDIT FIX 1
        self.all_pool = torch.arange(self.vol.shape[0], device=device)

    def _check_index_margin(self, index, shape):
        """Refuse to train on an index whose candidates do not fit the current training patch.

        `batched_crop` clamps out-of-bounds coordinates, so an overhanging patch is filled by EDGE
        REPLICATION -- the boundary slice extruded, instance label and all, into a prism reaching
        the patch edge. That teaches "objects extend forever", which is the one thing an FFN must
        not learn. It is silent: nothing crashes, the loss falls, and the model grows without bound.

        Run-3 (2026-07-27) hit exactly this. walk_radius 16 -> 48 took the patch 65 -> 129 while the
        cached index still carried margin 22, so 91.2% of patches were replicated and the model
        collapsed to one object per cube. Loud failure beats a wasted day.
        """
        h = self.patch // 2
        Z, Y, X = shape
        z, y, x = (np.asarray(index[k]) for k in ("z", "y", "x"))
        bad = ((z < h) | (z >= Z - h) | (y < h) | (y >= Y - h)
               | (x < h) | (x >= X - h))
        frac = float(bad.mean()) if bad.size else 0.0
        if frac <= 0.02:
            return
        if not getattr(self.cfg, "strict_margin", True):
            return                      # toy phantoms in tests: volume smaller than a real patch
        msg = (f"{frac:.1%} of training candidates sit closer than half a patch "
               f"({h} vox) to a volume face, so their patches would be EDGE-REPLICATED "
               f"(fov={self.cfg.fov} walk_radius={self.cfg.walk_radius} -> patch={self.patch}). "
               f"Replicated content extrudes sheets into infinite prisms and teaches unbounded "
               f"growth. Re-run scripts.preprocess with this config (margin is now derived from "
               f"the training patch), or lower walk_radius.")
        # HARD FAIL at anything above noise. The 0.02..0.35 band was a trap: it warned and
        # proceeded on exactly the condition that destroyed run-3 (which ran at 91%), and a
        # walk_radius=24 relaunch would have sat at 33% -- inside the band, silently.
        raise RuntimeError("REFUSING TO TRAIN: " + msg)

    def sample_indices(self, B: int, allowed_bins, hardneg_frac: float,
                       g: torch.Generator, std_only: bool = False,
                       real_frac: float = 0.0) -> torch.Tensor:
        """Return B flat candidate indices, balanced across allowed bins with an
        optional hard-negative (blind-contact) fraction, an optional real-cube
        fraction (AUDIT FIX 1), and an optional std-only restriction."""
        picks = []
        n_real = int(round(B * real_frac)) if self.real_pool.numel() > 0 else 0
        if n_real > 0:
            sel = torch.randint(self.real_pool.numel(), (n_real,),
                                generator=g, device=self.device)
            picks.append(self.real_pool[sel])
        n_hard = int(round(B * hardneg_frac)) if self.hard_pool.numel() > 0 else 0
        n_hard = min(n_hard, B - n_real)
        if n_hard > 0:
            sel = torch.randint(self.hard_pool.numel(), (n_hard,),
                                generator=g, device=self.device)
            picks.append(self.hard_pool[sel])
        remaining = B - n_hard - n_real
        src = self.bin_pools_std if std_only else self.bin_pools
        pools = [src[b] for b in allowed_bins if src[b].numel() > 0]
        if not pools:
            pools = [self.all_pool]
        # equal draw across the represented bins (balanced sampling, DESIGN.md 5.3)
        per = max(1, (remaining + len(pools) - 1) // len(pools))
        for pool in pools:
            sel = torch.randint(pool.numel(), (per,), generator=g, device=self.device)
            picks.append(pool[sel])
        out = torch.cat(picks)
        if out.numel() < B:  # top up from all candidates
            sel = torch.randint(self.all_pool.numel(), (B - out.numel(),),
                                generator=g, device=self.device)
            out = torch.cat([out, self.all_pool[sel]])
        # shuffle so bins are interleaved, then take B
        perm = torch.randperm(out.numel(), generator=g, device=self.device)
        return out[perm][:B]

    def get_patches(self, flat_idx: torch.Tensor):
        """Crop image+inst patches (patch^3) at the sampled centers.

        Returns image_patch [B,1,P,P,P] float32 normalized to [-1,1],
        inst_patch [B,1,P,P,P] long, seed_instance [B] long.
        """
        vidx = self.vol[flat_idx]
        centers = self.ctr[flat_idx]
        seed_inst = self.inst_id[flat_idx]
        if self.image.device != self.device:
            # CPU-resident stacks (cfg.cubes_on_gpu False): gather the patches on CPU, move only
            # the crops -- ~13 MB per optimizer step against ~75 GB of stacks.
            vc = vidx.to(self.image.device)
            cc = centers.to(self.image.device)
            img = batched_crop(self.image, vc, cc, self.patch, self._pgrid)                 .to(self.device, non_blocking=True).float()
            inst = batched_crop(self.inst, vc, cc, self.patch, self._pgrid)                 .to(self.device, non_blocking=True).long()
        else:
            img = batched_crop(self.image, vidx, centers, self.patch, self._pgrid).float()
            inst = batched_crop(self.inst, vidx, centers, self.patch, self._pgrid).long()
        img = (img / 255.0 - 0.5) / 0.5
        return img, inst, seed_inst


class AsyncPatchLoader:
    """Prefetch pipeline for CPU-resident stacks (cfg.cubes_on_gpu False).

    Measured cost of the synchronous path: the per-step CPU patch gather over a ~74 GB stack plus
    the H2D copy sat in the middle of every optimizer step and halved throughput (2050 -> ~1000
    fov-steps/s). Here the MAIN thread only samples indices (GPU, cheap, deterministic per-step
    generator); worker threads do the CPU gather + pinned staging + async H2D on a dedicated copy
    stream, `depth` steps ahead, so the transfer fully overlaps the previous step's compute.
    ATen CPU gathers release the GIL, so workers genuinely run in parallel with the training loop.
    """

    def __init__(self, cache: GpuVolumeCache, cfg, device, seed: int, rank: int, depth: int = 2):
        from concurrent.futures import ThreadPoolExecutor
        self.cache = cache
        self.cfg = cfg
        self.device = device
        self.seed = int(seed)
        self.rank = int(rank)
        self.depth = int(depth)
        self.pool = ThreadPoolExecutor(max_workers=depth)
        self.copy_stream = torch.cuda.Stream(device) if torch.cuda.is_available() else None
        self.pending = {}

    def _sample(self, step: int, sched):
        # per-step deterministic generator: sampling for step N+k may be drawn while step N is
        # still running, so it cannot share the trajectory generator without racing it
        gs = torch.Generator(device=self.device)
        gs.manual_seed(self.seed * 1000003 + self.rank * 7919 + step)
        return self.cache.sample_indices(
            self.cfg.batch_per_gpu, sched.allowed_bins, sched.hardneg_frac, gs,
            std_only=sched.std_only, real_frac=getattr(sched, "real_frac", 0.0))

    def _fetch(self, idx):
        c = self.cache
        vidx = c.vol[idx]
        centers = c.ctr[idx]
        seed_inst = c.inst_id[idx]
        vc = vidx.to(c.image.device)
        cc_ = centers.to(c.image.device)
        img_c = batched_crop(c.image, vc, cc_, c.patch, c._pgrid).pin_memory()
        inst_c = batched_crop(c.inst, vc, cc_, c.patch, c._pgrid).pin_memory()
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                img = img_c.to(self.device, non_blocking=True).float()
                img = (img / 255.0 - 0.5) / 0.5
                inst = inst_c.to(self.device, non_blocking=True).long()
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
        else:
            img = (img_c.float() / 255.0 - 0.5) / 0.5
            inst = inst_c.long()
            ev = None
        return img, inst, seed_inst, ev

    def schedule(self, step: int, sched):
        if step not in self.pending:
            idx = self._sample(step, sched)          # main thread: GPU sampling is cheap
            self.pending[step] = self.pool.submit(self._fetch, idx)

    def get(self, step: int, sched):
        self.schedule(step, sched)
        img, inst, seed_inst, ev = self.pending.pop(step).result()
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return img, inst, seed_inst


# ----------------------------------------------------------------------------
# On-GPU augmentation
#
# Runtime policy derived by empirical stress-testing on real+synth thin-sheet
# cubes (scratchpad/stress_aug.py, 2026-07): every augmentation was magnitude-
# swept for its safe operating range against sharper failure-mode metrics --
# gap-collapse (label gaps driven sub-voxel), sheet-thinning (<3 vox),
# label integrity (orphan/fragment), blind-contact preservation, and image<->
# label boundary drift. The catastrophe here is a MERGE, so the two hard limits
# are (1) never collapse an inter-wrap gap and (2) never thin a 5-6 vox sheet
# below ~3 vox. Appearance augs are geometry-safe by construction (labels are
# invariant); geometry augs interpolate, so their magnitudes are capped where
# nearest-label resampling starts to drift/thin/collapse.
#
# Order (per example, one transform held across the whole recurrent unroll):
#   1. spatial_augment  : octahedral(exact) x continuous rot/scale/shear, fused
#                         into ONE affine grid_sample (img bilinear, label nearest,
#                         reflection padding so no false border enters the FOV).
#   2. intensity_augment: gain/bias/gamma/contrast/blur/noise/bias-field, each
#                         probability-gated, continuous magnitudes, label-invariant.
# ----------------------------------------------------------------------------
def _rand(shape, g, dev, lo=0.0, hi=1.0):
    return torch.rand(shape, generator=g, device=dev) * (hi - lo) + lo


def octahedral_augment(img: torch.Tensor, inst: torch.Tensor, g: torch.Generator):
    """Label-exact base geometry: one random element of the 48-element octahedral
    group (axis permutation + per-axis flips) applied to the whole batch. Data is
    isotropic so this needs no interpolation. Kept as the exact base for the pure-
    geometry ablation and for tests; the wired training path uses spatial_augment
    (which folds a per-example octahedral orientation into its affine)."""
    dev = img.device
    perm = torch.randperm(3, generator=g, device=dev).tolist()
    dims = [2 + p for p in perm]
    img = img.permute(0, 1, *dims).contiguous()
    inst = inst.permute(0, 1, *dims).contiguous()
    flip = (torch.rand(3, generator=g, device=dev) < 0.5)
    fd = [2 + i for i in range(3) if flip[i]]
    if fd:
        img = torch.flip(img, fd)
        inst = torch.flip(inst, fd)
    return img, inst


def _octahedral_matrices(B, g, dev):
    """Per-example random signed-permutation matrix (one of the 48 octahedral
    orientations), returned as [B,3,3]. Integer-valued, so when composed into an
    affine_grid it maps voxel centers to voxel centers -> label-exact on its own."""
    # random axis permutation
    perms = torch.stack([torch.randperm(3, generator=g, device=dev) for _ in range(B)])
    P = torch.zeros(B, 3, 3, device=dev)
    P[torch.arange(B).unsqueeze(1), torch.arange(3).unsqueeze(0), perms] = 1.0
    signs = torch.where(_rand((B, 3), g, dev) < 0.5, -1.0, 1.0)
    return P * signs.unsqueeze(1)          # scale columns by ±1


def _axis_angle_matrix(axis, ang):
    """Rodrigues rotation matrices. axis [B,3] (unnormalized), ang [B]. -> [B,3,3]."""
    a = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c = torch.cos(ang); s = torch.sin(ang); C = 1 - c
    B = axis.shape[0]
    R = torch.stack([
        c + x*x*C,     x*y*C - z*s,   x*z*C + y*s,
        y*x*C + z*s,   c + y*y*C,     y*z*C - x*s,
        z*x*C - y*s,   z*y*C + x*s,   c + z*z*C], dim=1).view(B, 3, 3)
    return R


def spatial_augment(img: torch.Tensor, inst: torch.Tensor, g: torch.Generator,
                    rot_deg=16.0, scale=0.12, shear=0.06, p_continuous=0.9):
    """Per-example geometric augmentation, ONE affine grid_sample.

    Composes: a random octahedral orientation (label-exact on its own) with a
    continuous small rotation about a random 3D axis, anisotropic content-scaling,
    and shear. img sampled bilinear, inst nearest, both with BORDER (edge-replicate)
    padding so the walk region never sees a false zero border. (Docstring previously
    said "reflection", which the code has never used -- and the distinction is
    load-bearing: border replication is why large content-DOWNSCALES are unsafe
    here, they extrude edge slices with their labels into the patch.) Magnitudes are the measured
    safe range (rot<=~35deg, |scale-1|<=~0.15, shear<=~0.15 all keep gaps/thickness).

    img/inst: [B,1,P,P,P] (img float in [-1,1], inst long).
    """
    B = img.shape[0]; dev = img.device
    Roct = _octahedral_matrices(B, g, dev)                         # [B,3,3]
    apply_cont = _rand((B,), g, dev) < p_continuous
    # continuous rotation about a random axis
    axis = _rand((B, 3), g, dev, -1, 1)
    ang = _rand((B,), g, dev, -1, 1) * (rot_deg * math.pi / 180.0) * apply_cont
    Rc = _axis_angle_matrix(axis, ang)                             # [B,3,3]
    # anisotropic content-scale: coordinate multiplier = 1/content_scale
    cs = 1.0 + _rand((B, 3), g, dev, -1, 1) * scale * apply_cont.unsqueeze(1)
    Sinv = torch.zeros(B, 3, 3, device=dev)
    Sinv[:, 0, 0] = 1.0 / cs[:, 0]; Sinv[:, 1, 1] = 1.0 / cs[:, 1]; Sinv[:, 2, 2] = 1.0 / cs[:, 2]
    # shear (upper-triangular unit-diagonal)
    Sh = torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1)
    sh = _rand((B, 3), g, dev, -1, 1) * shear * apply_cont.unsqueeze(1)
    Sh[:, 0, 1] = sh[:, 0]; Sh[:, 0, 2] = sh[:, 1]; Sh[:, 1, 2] = sh[:, 2]
    M = torch.bmm(torch.bmm(Roct, Rc), torch.bmm(Sinv, Sh))        # [B,3,3]
    theta = torch.zeros(B, 3, 4, device=dev, dtype=img.dtype)
    theta[:, :, :3] = M.to(img.dtype)
    grid = F.affine_grid(theta, img.shape, align_corners=True)
    # 'border' (edge replicate): out-of-frame samples take the nearest edge value.
    # This never injects a false dark gap into the FOV (unlike 'zeros') and never
    # mirrors a border sheet into a DISCONNECTED duplicate (unlike 'reflection',
    # which inflates the per-label component count on cubes whose sheets touch the
    # border). Interior thickness/gaps are identical between the two; border keeps
    # label integrity clean.
    img_o = F.grid_sample(img, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
    inst_o = F.grid_sample(inst.float(), grid, mode="nearest",
                           padding_mode="border", align_corners=True)
    return img_o, inst_o.round().long()


def _sep_blur(img: torch.Tensor, sigma: torch.Tensor, rmax: int = 3):
    """Per-example separable Gaussian blur via grouped conv3d. sigma [B] (>=0)."""
    B = img.shape[0]; dev = img.device
    x = torch.arange(-rmax, rmax + 1, device=dev, dtype=torch.float32)     # [K]
    s = sigma.clamp(min=1e-3).view(B, 1)
    k = torch.exp(-(x.view(1, -1) ** 2) / (2 * s * s))                     # [B,K]
    k = k / k.sum(1, keepdim=True)
    K = 2 * rmax + 1
    out = img
    for ax in range(2, 5):
        shape = [B, 1, 1, 1, 1]; shape[ax] = K
        w = k.view(B, 1, K)[:, :, :].reshape(*shape)                       # [B,1,..K..]
        pad = [0, 0, 0, 0, 0, 0]
        out = out.reshape(1, B, *out.shape[2:])
        p = [0, 0, 0]; p[ax - 2] = rmax
        out = F.conv3d(F.pad(out, (p[2], p[2], p[1], p[1], p[0], p[0]), mode="reflect"),
                       w, groups=B).reshape(B, 1, *img.shape[2:])
    return out


def intensity_augment(img: torch.Tensor, g: torch.Generator,
                      gain=0.25, bias=0.10, gamma=0.5, contrast=0.3,
                      noise=0.05, blur_sigma=1.0, biasfield=0.3,
                      p_gain=0.8, p_bias=0.5, p_gamma=0.5, p_contrast=0.4,
                      p_noise=0.7, p_blur=0.3, p_biasfield=0.4):
    """Per-example, probability-gated, continuous-magnitude appearance jitter to
    close the synth->real gap. All ops are label-invariant. img in [-1,1].

    Ranges are the measured safe operating points (stress_aug.py): gamma exponent
    stays in [1/(1+g), 1+g] (>=2.0 starts erasing the gap cue), blur sigma<=~1.5
    (>=2 collapses the dark inter-wrap band), noise sigma<=~0.06*range.
    """
    B = img.shape[0]; dev = img.device
    def gate(p):
        return (_rand((B, 1, 1, 1, 1), g, dev) < p).to(img.dtype)
    # gain (multiplicative contrast about 0)
    if p_gain > 0:
        gg = 1.0 + _rand((B, 1, 1, 1, 1), g, dev, -1, 1) * gain * gate(p_gain)
        img = img * gg
    # contrast about the per-example mean
    if p_contrast > 0:
        mu = img.mean(dim=(2, 3, 4), keepdim=True)
        cc = 1.0 + _rand((B, 1, 1, 1, 1), g, dev, -1, 1) * contrast * gate(p_contrast)
        img = (img - mu) * cc + mu
    # bias (additive brightness)
    if p_bias > 0:
        bb = _rand((B, 1, 1, 1, 1), g, dev, -1, 1) * bias * gate(p_bias)
        img = img + bb
    # gamma (in [0,1] space)
    if p_gamma > 0:
        gm = torch.exp(_rand((B, 1, 1, 1, 1), g, dev, -1, 1) * math.log(1 + gamma) * gate(p_gamma))
        p01 = ((img + 1) * 0.5).clamp(0, 1)
        img = 2.0 * p01.pow(gm) - 1.0
    # smooth multiplicative bias field (low-freq inhomogeneity; TorchIO/N4 style)
    if p_biasfield > 0 and biasfield > 0:
        lo = _rand((B, 1, 4, 4, 4), g, dev, -1, 1)
        field = F.interpolate(lo, size=img.shape[2:], mode="trilinear", align_corners=True)
        amp = biasfield * gate(p_biasfield)
        img = img * (1.0 + amp * field)
    # gaussian blur (per-example sigma)
    if p_blur > 0 and blur_sigma > 0:
        sig = _rand((B,), g, dev, 0.4, blur_sigma)
        blurred = _sep_blur(img, sig)
        m = gate(p_blur)
        img = torch.where(m > 0, blurred, img)
    # additive gaussian noise
    if p_noise > 0 and noise > 0:
        sd = _rand((B, 1, 1, 1, 1), g, dev, 0, noise) * gate(p_noise)
        img = img + torch.randn(img.shape, generator=g, device=dev) * sd
    return img.clamp(-1.5, 1.5)
