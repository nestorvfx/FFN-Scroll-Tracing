#!/usr/bin/env python3
"""Bright silent-merge audit (workflow rec #1). Read-only. For each corpus cube, on the persisted per-sheet
instance volume vs the binary carved training label:
  (a) uncarved inter-wrap contact: 6-adjacent voxel pairs, both fiber in the binary label, DIFFERENT wrap ids
      (the carve should have zeroed these -> ~0 if carve works; nonzero = partial merge in the label).
  (b) LABEL BRIDGE (the one that matters): connected components (26-conn) of the binary fiber that span >=2
      distinct wrap ids with a bbox extent > 40 vox -> two wraps welded into one ridge the tracer would follow.
Stratified deep vs std. Light: small process pool."""
import glob, os, json
import numpy as np
import tifffile
from scipy import ndimage as ndi
from collections import defaultdict
import multiprocessing as mp

CORP = '/root/surf/data/synthfuse_corpus'
META = json.load(open(f'{CORP}/synthfuse_meta.json')) if os.path.exists(f'{CORP}/synthfuse_meta.json') else {}
CONN26 = np.ones((3, 3, 3), bool)


def audit_one(instf):
    cid = os.path.basename(instf)[:-4]
    try:
        inst = tifffile.imread(instf).astype(np.int32)
        lab = tifffile.imread(f'{CORP}/labelsTr/{cid}.tif')
    except Exception as e:
        return None
    fg = lab > 0
    tot = int(fg.sum())
    strat = META.get(cid, {}).get('stratum', '?')
    # (a) uncarved adjacent different-wrap pairs
    unc = 0
    for ax in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(None, -1); s2[ax] = slice(1, None)
        ai, bi = inst[tuple(s1)], inst[tuple(s2)]
        af, bf = fg[tuple(s1)], fg[tuple(s2)]
        unc += int((af & bf & (ai > 0) & (bi > 0) & (ai != bi)).sum())
    # (b) binary-fiber CCs spanning >=2 wrap ids over >40 vox
    cc, ncc = ndi.label(fg, structure=CONN26)
    K = int(inst.max()) + 1
    on = fg & (inst > 0)
    pair = np.unique(cc[on].astype(np.int64) * K + inst[on].astype(np.int64))
    cc_ids = defaultdict(set)
    for p in pair.tolist():
        c, i = divmod(p, K)
        cc_ids[c].add(i)
    multi = [c for c, ids in cc_ids.items() if len(ids) >= 2]
    slices = ndi.find_objects(cc)
    bridged, spans, bridged_vox = 0, [], 0
    for c in multi:
        sl = slices[c - 1]
        if sl is None:
            continue
        span = max(s.stop - s.start for s in sl)
        if span > 40:
            bridged += 1; spans.append(span)
            bridged_vox += int((cc == c).sum())
    return dict(cid=cid, strat=strat, fg=round(tot / inst.size, 3), n_inst=int(len(np.unique(inst[inst > 0]))),
                unc=unc, unc_frac=round(unc / max(1, tot), 6),
                bridged_ccs=bridged, max_span=round(max(spans), 1) if spans else 0,
                bridged_vox_frac=round(bridged_vox / max(1, tot), 5))


def main():
    insts = sorted(glob.glob(f'{CORP}/labelsTr_inst/*.tif'))
    # stratified sample: all deep (they're the risk) + a matched std sample, cap ~80 total for speed
    deep = [f for f in insts if META.get(os.path.basename(f)[:-4], {}).get('stratum') == 'deep']
    std = [f for f in insts if META.get(os.path.basename(f)[:-4], {}).get('stratum') != 'deep']
    pick = deep[:60] + std[:40]
    print(f"auditing {len(pick)} cubes ({min(60,len(deep))} deep + {min(40,len(std))} std) of {len(insts)}", flush=True)
    with mp.Pool(8) as pool:
        res = [r for r in pool.map(audit_one, pick) if r]
    for strat in ('deep', 'std'):
        rs = [r for r in res if r['strat'] == strat]
        if not rs:
            continue
        nb = sum(r['bridged_ccs'] for r in rs)
        cubes_with_bridge = sum(1 for r in rs if r['bridged_ccs'] > 0)
        unc = sum(r['unc'] for r in rs)
        print(f"\n=== {strat} (n={len(rs)}) ===")
        print(f"  uncarved diff-wrap voxel pairs: total {unc}  (mean frac {np.mean([r['unc_frac'] for r in rs]):.6f})")
        print(f"  LABEL BRIDGES (CC spans >=2 wraps >40vox): {nb} total, in {cubes_with_bridge}/{len(rs)} cubes")
        if nb:
            worst = sorted(rs, key=lambda r: -r['bridged_ccs'])[:5]
            for w in worst:
                if w['bridged_ccs']:
                    print(f"    {w['cid']} fg={w['fg']} inst={w['n_inst']}: {w['bridged_ccs']} bridges, "
                          f"max_span={w['max_span']}, bridged_vox_frac={w['bridged_vox_frac']}")
    print("\nMERGE_AUDIT_DONE")


if __name__ == '__main__':
    main()
