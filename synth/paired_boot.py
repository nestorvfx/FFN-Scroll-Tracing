#!/usr/bin/env python3
"""Paired bootstrap over the shared held-out cubes: is an arm's MERGER drop real, or noise?

Why paired: every arm is scored on the SAME cubes, and per-cube difficulty dominates the variance (merger_rate
ranges ~0.0-0.67 across cubes). Resampling each arm independently would drown a real 0.02 effect in that
between-cube spread. Resampling the SAME cube indices for both arms cancels it -- the classic paired design.

Why resample cubes and RE-POOL: tstr_eval reports merger_rate = total_merged/total_covered (a pooled ratio over
sheets), NOT the mean of per-cube rates. The bootstrap must reproduce that estimator or its CI describes a
statistic we never report. So each replicate re-sums merged/covered over the resampled cubes.

The cube is the unit of resampling (sheets within a cube are not independent -- they share a scan region,
a density regime and a source segment).

Usage: python paired_boot.py --ctrl /root/score_ctrl.json --arms /root/score_crumple.json ...
"""
import json, argparse
import numpy as np


def load(p):
    d = json.load(open(p))
    return d["label"], {c["cube"]: c for c in d["per_cube"]}, d["overall"]


def pooled(rows, num, den):
    t = sum(r[den] for r in rows)
    return (sum(r[num] for r in rows) / t) if t else float("nan")


def boot(ctrl, arm, cubes, num, den, n_boot, rng):
    """Return (delta_hat, lo, hi, p_improve) for pooled `num/den`, arm - ctrl, paired on cube."""
    idx = np.arange(len(cubes))
    base = pooled([arm[c] for c in cubes], num, den) - pooled([ctrl[c] for c in cubes], num, den)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        take = [cubes[i] for i in rng.choice(idx, size=len(idx), replace=True)]   # SAME cubes both arms -> paired
        deltas[b] = pooled([arm[c] for c in take], num, den) - pooled([ctrl[c] for c in take], num, den)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return base, lo, hi, float((deltas < 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", default="/root/score_ctrl.json")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    clab, cmap, cov = load(a.ctrl)
    print(f"\nPAIRED BOOTSTRAP vs {clab}  ({a.n_boot} replicates, cube-level resampling, pooled ratio re-computed)")
    print(f"reference {clab}: merger {cov['merger_rate']:.4f}  split {cov['split_rate']:.4f}\n")
    print(f"{'ARM':<10} {'MERGER':>7} {'dMERGER':>8} {'95% CI':>17} {'P(better)':>9} | "
          f"{'SPLIT':>7} {'dSPLIT':>8} {'95% CI':>17} {'P(better)':>9}  VERDICT")

    for p in a.arms:
        lab, amap, aov = load(p)
        cubes = sorted(set(cmap) & set(amap))
        if len(cubes) != len(cmap):
            print(f"  ! {lab}: only {len(cubes)}/{len(cmap)} cubes shared -- comparing on the intersection")
        dm, mlo, mhi, pm = boot(cmap, amap, cubes, "merged_sheets", "n_covered", a.n_boot, rng)
        ds, slo, shi, ps = boot(cmap, amap, cubes, "split_sheets", "n_covered", a.n_boot, rng)
        # KEEP RULE: merger must drop with real confidence AND split must not credibly rise.
        # "not beyond noise" = the split CI's upper end stays within +0.01 (a 1-point sheet-level rise).
        keep = (pm >= 0.90) and (shi <= 0.01)
        verdict = "KEEP" if keep else ("borderline" if pm >= 0.75 and shi <= 0.01 else "DROP")
        print(f"{lab:<10} {aov['merger_rate']:>7.4f} {dm:>+8.4f} [{mlo:>+6.3f},{mhi:>+6.3f}] {pm:>9.2f} | "
              f"{aov['split_rate']:>7.4f} {ds:>+8.4f} [{slo:>+6.3f},{shi:>+6.3f}] {ps:>9.2f}  {verdict}")

    print("\nKEEP RULE: P(merger better) >= 0.90 AND split 95%-CI upper bound <= +0.01.")
    print("Merger and split are two halves of ONE tradeoff -- an arm that buys merger with split is not a win.\n")


if __name__ == "__main__":
    main()
