#!/usr/bin/env python3
"""Delamination census + single-vs-fused-lamina discriminator on the FINAL composer config.

Census (per instance, along the stacking axis ax=2):
  split_frac : fraction of occupied columns holding >= 2 papyrus runs (delamination present)
  gap_mode   : histogram mode of the inter-ply air-gap widths (1..12) -- the failure mode the
               bundling/squeeze fixes addressed produced mode 1-2 vs the donor mode ~7.

Fused-lamina discriminator (memory: sheet-mask-support-and-fusion-test -- no test existed and the
moat REWARDS a fused pair): an instance that is really TWO laminae under one label shows, in its
SINGLE-run columns, (a) anomalous thickness ~2x its own mode and (b) an internal CT dip (the dim
seam the scanner still resolves). Report the fraction of single-run columns that are thick AND
dipped, per corpus; the honest gate is synth <= donors (donor masks are the contamination source).

Usage:
  delam_census.py --corpus <dir with labelsTr_inst/> [--n 30]           # synth batches
  delam_census.py --bank <dir with *.pkl>                               # donor baseline (papbox)
"""
import argparse
import glob
import os
import pickle

import numpy as np
import tifffile


def col_runs(mf):
    """(col, start, end) of True-runs for rows of a 2-D bool array."""
    C, N = mf.shape
    pad = np.zeros((C, N + 2), np.int8)
    pad[:, 1:-1] = mf
    d = np.diff(pad.reshape(-1))
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    col = st // (N + 2)
    return col, st - col * (N + 2), en - col * (N + 2)


def census_mask(m, ct=None):
    """m: (H,W,E) bool for ONE instance (depth last). Returns per-instance census dict."""
    H, W, E = m.shape
    mf = m.reshape(-1, E)
    occ = mf.any(1)
    n_oc = int(occ.sum())
    if n_oc < 100:
        return None
    col, s, e = col_runs(mf)
    nruns = np.bincount(col, minlength=mf.shape[0])
    split = int((nruns[occ] >= 2).sum())
    same = col[1:] == col[:-1]
    g = (s[1:] - e[:-1])
    gaps = g[same & (g >= 1) & (g <= 12)]
    thick = (e - s).astype(np.int32)
    out = dict(split_frac=split / n_oc,
               gaps=gaps,
               med_thick=float(np.median(thick)) if thick.size else 0.0)
    if ct is not None:
        cf = ct.reshape(-1, E)
        single = np.flatnonzero(nruns == 1)
        one = np.isin(col, single)
        c1, s1, e1 = col[one], s[one], e[one]
        t1 = e1 - s1
        mode_t = float(np.median(t1)) if t1.size else 0.0
        thick_cols = t1 >= max(1.8 * mode_t, mode_t + 3)
        n_dip = n_thick = 0
        for c_, a_, b_ in zip(c1[thick_cols], s1[thick_cols], e1[thick_cols]):
            prof = cf[c_, a_:b_]
            if prof.size < 5:
                continue
            n_thick += 1
            edge = 0.5 * (prof[:2].mean() + prof[-2:].mean())
            if prof[1:-1].min() < edge - 8.0:
                n_dip += 1
        out["n_single"] = int(t1.size)
        out["n_thick"] = n_thick
        out["n_fused_suspect"] = n_dip
    return out


def report(rows, tag):
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"{tag}: no instances")
        return
    sf = np.array([r["split_frac"] for r in rows])
    gaps = np.concatenate([r["gaps"] for r in rows if r["gaps"].size]) if any(
        r["gaps"].size for r in rows) else np.array([0])
    gm = int(np.bincount(gaps).argmax()) if gaps.size else 0
    th = np.array([r["med_thick"] for r in rows])
    line = (f"{tag}: inst={len(rows)}  split_frac med {np.median(sf):.3f} "
            f"(p25 {np.percentile(sf,25):.3f} p75 {np.percentile(sf,75):.3f})  "
            f">10%-split {float((sf > 0.10).mean()):.2f}  gap-mode {gm}  "
            f"gap p50 {float(np.median(gaps)):.1f}  med-thick {float(np.median(th)):.1f}")
    if "n_fused_suspect" in rows[0]:
        ns = sum(r.get("n_single", 0) for r in rows)
        nt = sum(r.get("n_thick", 0) for r in rows)
        nd = sum(r.get("n_fused_suspect", 0) for r in rows)
        line += f"  | fused-suspect {nd}/{ns} single-cols ({100.0*nd/max(ns,1):.2f}%), thick {nt}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--bank")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.bank:
        rows = []
        for pk in sorted(glob.glob(os.path.join(a.bank, "*.pkl"))):
            sh, _air = pickle.load(open(pk, "rb"))
            for s_ in sh:
                pb = s_.get("papbox")
                cb = s_.get("ctbox")
                if pb is not None:
                    rows.append(census_mask(pb.astype(bool), cb))
        report(rows, "DONORS")
    if a.corpus:
        rows = []
        for p in sorted(glob.glob(os.path.join(a.corpus, "labelsTr_inst", "*.tif")))[: a.n]:
            inst = tifffile.imread(p).astype(np.int32)
            cp = p.replace("labelsTr_inst", "imagesTr").replace(".tif", "_0000.tif")
            ct = tifffile.imread(cp).astype(np.float32) if os.path.exists(cp) else None
            for iid in np.unique(inst[inst > 0]):
                m = inst == iid
                if m.sum() < 3000:
                    continue
                rows.append(census_mask(m, ct))
        report(rows, "SYNTH ")


if __name__ == "__main__":
    main()
