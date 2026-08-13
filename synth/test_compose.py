#!/usr/bin/env python3
"""Offline gate for the SHEET COMPOSER -- run before any corpus build. Targets from analyze_compact.py
(real compact windows, slice z=10192): fg 0.63-0.66, seam_w p50 ~10, persistence ~8-11, seams mostly thin.

T1 obstacle        : zero interpenetration violations (settle never pushes a sheet through the stack)
T2 determinism     : same seed -> byte-identical ct AND inst
T3 diversity       : sheets from >=2 source cubes; different seeds -> different cubes
T4 realism stats   : fg in [0.45,0.80]; seam_w p50 <= 12; SOME interfaces >60% fused and SOME <50% (a corpus of
                     all-fused or all-open teaches a constant, not a distribution)
T5 GT supervises   : >=10k voxel pairs where DIFFERENT ids are z-adjacent (fused contacts the carve can learn from)
T6 CT/label sanity : uint8, papyrus brighter than air, ids contiguous 1..K

Usage: python test_compose.py --cubes <dir>
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_compose import compose
from analyze_compact import window_stats

def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True)
    a = ap.parse_args()
    r = []

    ct1, in1, m1 = compose(a.cubes, 11)
    ct1b, in1b, m1b = compose(a.cubes, 11)
    ct2, in2, m2 = compose(a.cubes, 12)

    r.append(ok("T1 obstacle: zero violations", m1["violations"] == 0 and m2["violations"] == 0,
                f"viol {m1['violations']}/{m2['violations']}"))
    r.append(ok("T2 determinism", np.array_equal(ct1, ct1b) and np.array_equal(in1, in1b)))
    r.append(ok("T3 diversity: sources>=2 and seeds differ",
                len(m1["sources"]) >= 2 and not np.array_equal(ct1, ct2),
                f"sources {m1['sources']}"))

    st = window_stats(ct1[:, ct1.shape[1] // 2, :].astype(np.float32), force_angle=0.0)
    cf = [x["contact_frac"] for x in m1["interfaces"] if x["mode"] != "spacer"]
    r.append(ok("T4 realism: fg in [0.45,0.80]", 0.45 <= st["fg_frac"] <= 0.80, f"fg {st['fg_frac']}"))
    r.append(ok("T4 realism: seam_w p50 <= 12", st["seam_w_p"][1] <= 12, f"seam_w {st['seam_w_p']}"))
    r.append(ok("T4 realism: fused AND open interfaces both present",
                any(c > 0.6 for c in cf) and any(c < 0.5 for c in cf), f"contacts {sorted(cf)[:8]}"))

    z_adj = (in1[..., :-1] != in1[..., 1:]) & (in1[..., :-1] > 0) & (in1[..., 1:] > 0)
    r.append(ok("T5 GT: >=10k fused different-id contacts", int(z_adj.sum()) >= 10000, f"{int(z_adj.sum())} pairs"))

    ids = np.unique(in1[in1 > 0])
    pap = float(ct1[in1 > 0].mean()); air = float(ct1[in1 == 0].mean())
    r.append(ok("T6 sanity: uint8, papyrus > air, ids contiguous",
                ct1.dtype == np.uint8 and pap > air + 15 and ids.min() == 1 and ids.max() == len(ids),
                f"pap {pap:.0f} air {air:.0f} ids {len(ids)}"))

    print(f"\nCOMPOSER GATE: {sum(r)}/{len(r)} PASS")
    sys.exit(0 if all(r) else 1)


if __name__ == "__main__":
    main()
