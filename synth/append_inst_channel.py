#!/usr/bin/env python3
"""Append the per-sheet instance volume as seg CHANNEL 1 ([binary, inst]) for SYNTH cases of a preprocessed
nnU-Net dataset, enabling the affinity/LSD aux head's on-the-fly GT computation. Real cases stay 1-channel
(no instances; the aff/LSD dataloaders pad ch1=0 -> masked loss). Synth cubes are 192^3 with no crop, so the
raw inst aligns with the preprocessed seg; an overlap check (carved binary subset of inst>0) verifies alignment
before writing. Parallel (recompression is CPU-bound), idempotent; removes stale unpacked .npy so training
re-unpacks the new 2-channel seg."""
import argparse, glob, os
import numpy as np
import tifffile
import multiprocessing as mp

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)            # e.g. Dataset230_059plus_synthfuse
ap.add_argument('--corp', required=True)               # corpus dir containing labelsTr_inst/
ap.add_argument('--jobs', type=int, default=32)
a = ap.parse_args()

PREP = os.environ.get('nnUNet_preprocessed', '/root/surf/data/nnUNet_preprocessed')
prep = sorted(glob.glob(f'{PREP}/{a.dataset}/*3d_fullres'))[0]
inst_dir = os.path.join(a.corp, 'labelsTr_inst')


def one(npz):
    cid = os.path.basename(npz)[:-4]
    try:
        dd = np.load(npz)
        data, seg = dd['data'], dd['seg']
    except Exception as e:
        return 'bad', (cid, 'corrupt', str(e)[:60])
    if seg.shape[0] >= 2:
        return 'skip', cid
    instf = os.path.join(inst_dir, cid + '.tif')
    if cid.startswith(('syn_', 'synthfuse_')) and os.path.exists(instf):
        inst = tifffile.imread(instf)
        if inst.shape != seg.shape[1:]:
            return 'bad', (cid, 'shape', inst.shape, seg.shape[1:])
        b = seg[0] > 0
        ov = float((b & (inst > 0)).sum()) / max(1, int(b.sum()))
        if ov < 0.5:                                   # aligned cubes ~0.8-1.0; gross misalignment <0.5
            return 'bad', (cid, 'align', round(ov, 3))
        seg2 = np.concatenate([seg, inst[None].astype(seg.dtype)], axis=0)
        tmp = npz + '.tmp.npz'
        np.savez_compressed(tmp, data=data, seg=seg2)      # atomic: a killed worker can't corrupt the target
        os.replace(tmp, npz)
        for stale in (npz[:-4] + '.npy', npz[:-4] + '_seg.npy'):
            if os.path.exists(stale):
                os.remove(stale)                       # stale unpacked arrays would shadow the new seg
        return 'syn', cid
    return 'real', cid


if __name__ == '__main__':
    counts = {'syn': 0, 'real': 0, 'skip': 0}
    bad = []
    files = glob.glob(f'{prep}/*.npz')
    with mp.Pool(a.jobs) as pool:
        for kind, payload in pool.imap_unordered(one, files):
            if kind == 'bad':
                bad.append(payload)
            else:
                counts[kind] += 1
    print(f"2-channel seg: synth {counts['syn']} appended | real {counts['real']} left 1-ch | "
          f"skipped {counts['skip']} | bad {len(bad)}")
    for x in bad[:20]:
        print('  bad', x)
    print('APPEND_INST_DONE')
