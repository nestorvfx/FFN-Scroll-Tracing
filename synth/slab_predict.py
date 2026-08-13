#!/usr/bin/env python3
"""Single plain (non-affinity) slab prediction -> fiber.npy. A real module (not a heredoc) so nnU-Net's
multiprocessing export workers can re-import __main__ without crashing.

Usage: python slab_predict.py --model_dir <dir> --ckpt checkpoint_final.pth --tag base [--gpu 0]"""
import os, sys, argparse


def main():
    import numpy as np, torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--tag", required=True); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    out = f"/root/slab_pred_{a.tag}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/fiber.npy"):
        print("exists", out); return
    torch.cuda.set_device(a.gpu)
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        device=torch.device(f"cuda:{a.gpu}"), verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(a.model_dir, use_folds=(0,), checkpoint_name=a.ckpt)
    p.predict_from_files([["/root/slab_case/slab_0000.tif"]], out, save_probabilities=True, overwrite=True,
                         num_processes_preprocessing=2, num_processes_segmentation_export=2)
    npz = [f for f in os.listdir(out) if f.endswith(".npz")][0]
    np.save(f"{out}/fiber.npy", np.load(os.path.join(out, npz))["probabilities"][1].astype(np.float32))
    print("pred", out)


if __name__ == "__main__":
    main()
