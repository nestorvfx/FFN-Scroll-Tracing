"""Training curriculum (DESIGN.md 6.3).

FFN diverges if you start with long unrolls on hard scenes, so we schedule:
  Phase A (warm-up): T=1, std cubes only, easy bins (fill >= 0.05), teacher-forced.
  Phase B: ramp T 1->max_unroll, all bins, add deep cubes, teacher-force decays.
  Phase C: full T, all bins, policy movement, hard-negative mining at contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SchedState:
    T: int
    allowed_bins: List[int]
    teacher_force: bool
    hardneg_frac: float
    std_only: bool
    phase: str
    real_frac: float = 0.0
    global_step: int = 0        # absolute step; used by the move-head aux-loss ramp


class Curriculum:
    def __init__(self, cfg):
        self.cfg = cfg
        nbins = len(cfg.partition_bins) - 1
        # collapse fix: exclude degenerate near-empty-fill bins from ALL phases
        # (equal-per-bin draw otherwise gives their ~95%-background targets a huge
        # fixed batch share -> lazy predict-low solution)
        min_edge = getattr(cfg, "min_bin_edge", 0.0)
        self.all_bins = [i for i in range(nbins) if cfg.partition_bins[i] >= min_edge]
        if not self.all_bins:
            self.all_bins = list(range(nbins))
        # bins whose lower threshold >= 0.05 (the "easy"/higher-fill bins)
        self.easy_bins = [i for i in range(nbins) if cfg.partition_bins[i] >= 0.05]
        if not self.easy_bins:
            self.easy_bins = self.all_bins

    def state(self, step: int) -> SchedState:
        c = self.cfg
        _gs = int(step)
        if step < c.warmup_steps:
            # Phase A: pure synth-std easy warmup (std_only excludes real, AUDIT FIX 1)
            return SchedState(T=1, allowed_bins=self.easy_bins, teacher_force=True,
                              hardneg_frac=0.0, std_only=True, phase="A", real_frac=0.0,
                              global_step=_gs)
        if step < c.rampB_steps:
            # Phase B: ramp T, all bins/cubes. Real cubes already enter via bin pools;
            # ramp the real floor in over Phase B so appearance grounding starts early.
            frac = (step - c.warmup_steps) / max(1, c.rampB_steps - c.warmup_steps)
            T = 1 + round(frac * (c.max_unroll - 1))
            return SchedState(T=T, allowed_bins=self.all_bins,
                              teacher_force=step < c.teacher_force_steps,
                              hardneg_frac=0.0, std_only=False, phase="B",
                              real_frac=frac * c.real_frac, global_step=_gs)
        # Phase C: full T, policy movement, blind-contact hard mining + real floor
        return SchedState(T=c.max_unroll, allowed_bins=self.all_bins,
                          teacher_force=False, hardneg_frac=c.hardneg_frac,
                          std_only=False, phase="C", real_frac=c.real_frac,
                          global_step=_gs)
