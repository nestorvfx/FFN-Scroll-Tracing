#!/usr/bin/env python
"""From-scratch instance bring-up (DESIGN.md 3, deliverable 5).

Detects usable GPUs / CPU cores / RAM, verifies the Python deps and the data
paths, installs any missing pip packages, and prints the exact one-command
preprocess + dual-GPU train + eval invocations for THIS machine.

  python -m scripts.setup_instance --corpus /root/surf/data/synthfuse_corpus \
      --slab /root/data/slab --work /root/surf/ffn_work
"""
import argparse
import os
import shutil
import subprocess
import sys


REQUIRED = ["torch", "numpy", "scipy", "skimage", "tifffile", "nrrd"]


def check_imports():
    missing = []
    for m in REQUIRED:
        try:
            __import__(m)
        except Exception:
            missing.append("pynrrd" if m == "nrrd" else
                           ("scikit-image" if m == "skimage" else m))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/surf/data/synthfuse_corpus")
    ap.add_argument("--slab", default="/root/data/slab")
    ap.add_argument("--work", default="/root/surf/ffn_work")
    ap.add_argument("--install", action="store_true", help="pip install missing deps")
    args = ap.parse_args()

    import multiprocessing
    cores = os.cpu_count() or multiprocessing.cpu_count()
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
    except Exception:
        ram = float("nan")
    ngpu = 0
    gpu_names = []
    try:
        import torch
        ngpu = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(ngpu)]
        print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        print(f"[env] torch import failed: {e}")
    print(f"[env] cores={cores} ram={ram:.0f}GB gpus={ngpu} {gpu_names}")

    missing = check_imports()
    if missing:
        print(f"[deps] missing: {missing}")
        if args.install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
            print("[deps] installed")
        else:
            print("[deps] rerun with --install to install them")
    else:
        print("[deps] all present")

    for name, p in [("corpus", args.corpus), ("slab", args.slab)]:
        ok = os.path.isdir(p)
        print(f"[data] {name}: {p} {'OK' if ok else 'MISSING'}")

    workers = max(1, cores - 2)
    nproc = max(1, ngpu)
    print("\n=== run these ===")
    print(f"# 1) preprocess (one-time, ~cores-parallel)")
    print(f"python -m scripts.preprocess --work {args.work} --corpus {args.corpus} "
          f"--workers {workers}")
    print(f"# 2) train (dual-GPU)" if nproc > 1 else "# 2) train (single GPU)")
    print(f"torchrun --nproc_per_node={nproc} -m scripts.train --work {args.work}")
    print(f"# 3) evaluate")
    print(f"python -m scripts.evaluate --gate A --work {args.work} "
          f"--ckpt {args.work}/ckpt_last.pt --cubes 20")
    print(f"python -m scripts.evaluate --gate B --work {args.work} "
          f"--ckpt {args.work}/ckpt_last.pt --slab {args.slab}")


if __name__ == "__main__":
    main()
