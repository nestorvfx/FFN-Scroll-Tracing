#!/usr/bin/env python3
"""Radial-blast-weighted maximin (MALIS) reweight -- aligns the structured merge-loss to the CTF eval cost.

THE PROBLEM (audit finding 1, both independent analyses): the current constrained-MALIS negative pass weights
each would-be-bridge edge by the PRODUCT OF COMPONENT SIZES (`csize[ra]*csize[rb]`, nnUNetTrainerAffG.py:124) --
the classic Turaga Rand-error weighting. But the eval metric CTF weights a bridge by its RADIAL BLAST RADIUS: a
bridge between wraps whose inner member has radial rank r corrupts every wrap radially outward of it, i.e.
`blast = n_ranks - r` (wrap_erl2.py:74-80). Rand weighting makes an inner-wrap merge and an outer-wrap merge of
equal contact area receive EQUAL gradient, though their downstream cost differs ~68x. So even a correctly-scheduled
MALIS optimizes the wrong topological cost.

THE FIX: weight each maximin tree edge by the blast radius of the merge it represents = `n_ranks - min_rank`, where
min_rank is the innermost (smallest radial rank) wrap across the two components being joined. This makes the
structured loss a differentiable surrogate of CTF and encodes inner >> outer for free. Requires a radial order over
the wrap ids (from the synthetic umbilicus / deposit order -- SOTA_PLAN Stage 3 dataset change; on the real slab,
wrap_erl2.radial_order supplies it).

This module is the PURE-PYTHON maximin core + its unit test, verified offline. The trainer wires it into
MalisNegativeAffinityLoss by passing `rank`/`n_ranks` through (backward compatible: no rank -> size-product).

Run:  python malis_blast.py --selftest
"""
import sys, argparse
import numpy as np


def kruskal_maximin(pairs, sizes, rank=None, n_ranks=None):
    """pairs: list of (lo_id, hi_id, aff, loc) = per-wrap-pair MAX-affinity bridge edge + its voxel location.
    sizes: {wrap_id: voxel_count}. Kruskal MAX-spanning-tree by affinity DESC over the (tiny) wrap graph; each tree
    edge is the maximin (would-be bridge) for the wrap-pairs it joins.

    weight of a tree edge:
      * rank given  -> RADIAL BLAST: n_ranks - min_rank_across_the_two_components (== CTF's cost of that bridge).
      * rank None   -> LEGACY Rand: product of the two components' voxel counts (backward compatible).
    Returns [(loc, weight)]."""
    parent = {i: i for i in sizes}
    csize = dict(sizes)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    for lo, hi, aff, loc in sorted(pairs, key=lambda e: -e[2]):
        ra, rb = find(lo), find(hi)
        if ra == rb:
            continue
        if rank is not None:
            # blast of THIS pairwise bridge = CTF cost of connecting wrap `lo` to wrap `hi`: everything radially
            # outward of the innermost of the two is corrupted. Use the edge's OWN endpoints (not the merged
            # component min), so the weight is order-independent and matches CTF's per-bridge definition exactly.
            inner = min(rank[lo], rank[hi])
            w = float(n_ranks - inner)
        else:
            w = float(csize[ra]) * float(csize[rb])
        out.append((loc, w))
        parent[rb] = ra
        csize[ra] += csize[rb]
    return out


def selftest():
    """Concentric-ring phantom: 5 wraps, radial rank 0 (innermost) .. 4 (outermost). Assert the blast weighting
    reproduces CTF's cost -- a bridge whose inner member is wrap r must weigh n_ranks - r -- and therefore an inner
    bridge weighs strictly more than an outer one, which the legacy size-product weighting does NOT guarantee."""
    n_ranks = 5
    rank = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}             # wrap id -> radial rank (ids arbitrary, not the rank)
    sizes = {10: 100, 20: 100, 30: 100, 40: 100, 50: 100}  # EQUAL sizes -> size-product is flat, isolating the fix

    # one bridge per adjacent radial pair; affinities descending so Kruskal adds them inner->outer deterministically
    def bridge(a, b, aff):
        return (a, b, aff, (a, b))
    pairs = [bridge(10, 20, 0.9), bridge(20, 30, 0.8), bridge(30, 40, 0.7), bridge(40, 50, 0.6)]

    blast = dict(kruskal_maximin(pairs, sizes, rank=rank, n_ranks=n_ranks))
    # bridge (10,20): inner rank 0 -> blast 5 ; (20,30): inner 1 -> 4 ; (30,40): inner 2 -> 3 ; (40,50): inner 3 -> 2
    exp = {(10, 20): 5.0, (20, 30): 4.0, (30, 40): 3.0, (40, 50): 2.0}
    for k, v in exp.items():
        assert abs(blast[k] - v) < 1e-9, f"blast weight {k} = {blast[k]}, expected {v} (CTF n_ranks-rank_inner)"
    # monotonic: inner bridge strictly heavier than any outer bridge
    order = [blast[(10, 20)], blast[(20, 30)], blast[(30, 40)], blast[(40, 50)]]
    assert order == sorted(order, reverse=True), f"blast must decrease outward, got {order}"
    ratio = blast[(10, 20)] / blast[(40, 50)]
    print(f"blast weights inner->outer: {order}  (inner/outer ratio {ratio:.2f})")

    # contrast: the LEGACY size-product, even with EQUAL wrap sizes, weights merges INCREASINGLY outward -- because
    # the inner component accumulates voxels as wraps join, so the product grows with each outward bridge. It gives
    # the OUTERMOST bridge the HEAVIEST gradient: exactly backwards from CTF (inner is worst). This is the bug.
    legacy = [w for _, w in kruskal_maximin(pairs, sizes, rank=None)]
    assert legacy == sorted(legacy), f"legacy should grow outward on nested rings, got {legacy}"
    assert order == sorted(order, reverse=True), "blast decreases outward"
    print(f"legacy size-product weights inner->outer: {[int(w) for w in legacy]}  "
          f"(GROWS outward -> weights the outer merge most; exactly backwards from CTF -- this is the bug)")

    # a size-DOMINATED case the legacy weighting gets exactly backwards: a huge OUTER wrap vs a small INNER one.
    # CTF says the inner bridge is worse (more text corrupted); size-product says the outer (bigger) is worse.
    sizes2 = {10: 10, 20: 10, 40: 1000, 50: 1000}
    rank2 = {10: 0, 20: 1, 40: 3, 50: 4}
    pairs2 = [bridge(10, 20, 0.9), bridge(40, 50, 0.6)]
    bl2 = dict(kruskal_maximin(pairs2, sizes2, rank=rank2, n_ranks=5))
    lg2 = dict(kruskal_maximin(pairs2, sizes2, rank=None))
    assert bl2[(10, 20)] > bl2[(40, 50)], "blast must rank the inner bridge worse"
    assert lg2[(40, 50)] > lg2[(10, 20)], "size-product ranks the (bigger) outer bridge worse -- exactly backwards"
    print(f"size-dominated case: blast inner {bl2[(10,20)]} > outer {bl2[(40,50)]} (correct); "
          f"legacy inner {lg2[(10,20)]:.0f} < outer {lg2[(40,50)]:.0f} (backwards)")
    print("SELFTEST_OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print("nothing to do; run --selftest")
