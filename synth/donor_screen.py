#!/usr/bin/env python3
"""Build the donor fused-lamina exclusion list (SC_EXCL) from a preharvested bank: per donor
sheet, the fraction of single-run columns that are anomalously thick AND carry an internal CT
dip (two real laminae under one mask label). Exclude the tail above --thr."""
import argparse, glob, json, os, pickle
import numpy as np
from delam_census import census_mask

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.12)
a = ap.parse_args()
excl, rates = [], []
for pk in sorted(glob.glob(os.path.join(a.bank, "*.pkl"))):
    cname = os.path.basename(pk)[:-4]
    sh, _air = pickle.load(open(pk, "rb"))
    for s_ in sh:
        r = census_mask(s_["papbox"].astype(bool), s_.get("ctbox"))
        if r is None or r.get("n_single", 0) < 3000:
            continue
        rate = r["n_fused_suspect"] / max(r["n_single"], 1)
        rates.append(rate)
        if rate > a.thr:
            excl.append(f"{cname}:{s_['id']}")
json.dump(excl, open(a.out, "w"))
rates = np.array(rates)
print(f"sheets {rates.size}  suspect-rate p50 {np.median(rates):.3f} p90 {np.percentile(rates, 90):.3f}"
      f"  excluded {len(excl)} (> {a.thr})")
