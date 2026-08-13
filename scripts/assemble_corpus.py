#!/usr/bin/env python
"""Assemble a merged FFN training corpus from one or more source corpora (symlinks, no copies).

Merges imagesTr / labelsTr / labelsTr_inst by symlink, unions synthfuse_meta.json (later sources
win on id collisions -- ids are disjoint by seed-range convention: old synth 0xxxx, v4geo 4xxxx),
and copies real_meta.json from the first source that has one.

Usage:
  python -m scripts.assemble_corpus --out /root/surf/data/ffn_corpus_v4 \
      /root/surf/data/synthfuse_corpus /root/surf/data/synthfuse_v4geo
"""
import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for sub in ("imagesTr", "labelsTr", "labelsTr_inst"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        n = 0
        for src in a.sources:
            for f in glob.glob(os.path.join(src, sub, "*.tif")):
                lp = os.path.join(a.out, sub, os.path.basename(f))
                if not os.path.exists(lp):
                    os.symlink(os.path.abspath(f), lp)
                    n += 1
        print(f"{sub}: +{n} linked, total {len(os.listdir(os.path.join(a.out, sub)))}")
    merged = {}
    for src in a.sources:
        mp = os.path.join(src, "synthfuse_meta.json")
        if os.path.exists(mp):
            merged.update(json.load(open(mp)))
    json.dump(merged, open(os.path.join(a.out, "synthfuse_meta.json"), "w"), indent=1)
    for src in a.sources:
        rp = os.path.join(src, "real_meta.json")
        if os.path.exists(rp):
            shutil.copy(rp, os.path.join(a.out, "real_meta.json"))
            break
    missing = [k for k, v in merged.items()
               if v.get("stratum") is None or v.get("fg") is None]
    print(f"synth meta {len(merged)} entries, {len(missing)} missing stratum/fg"
          + (" -- FIX BEFORE preprocess" if missing else ""))


if __name__ == "__main__":
    main()
