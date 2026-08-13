#!/usr/bin/env python
"""Push / pull a training run's checkpoints + state to HF, so a run survives losing the box.

Without this, "switch instance and resume" is not actually possible: the checkpoints live only on
the rented machine. `bootstrap.sh RESUME=1` pulls what this pushes.

  # on the training box, any time (safe while training runs -- it copies, never moves)
  HF_TOKEN=hf_... python -m scripts.ckpt_sync push --work /root/surf/ffn_run5

  # on a fresh box, after bootstrap.sh
  HF_TOKEN=hf_... python -m scripts.ckpt_sync pull --work /root/surf/ffn_run5

What travels: the three SELECTED checkpoints (ckpt_best / ckpt_merge_best / ckpt_core_best),
ckpt_last, **the newest numbered ckpt_NNNN.pt**, and the run state needed to continue exactly --
config.json, splits.json, train_index.npz, val_progress.jsonl. Older numbered snapshots are skipped
(pass --all for those).

The newest numbered snapshot is NOT optional: `ckpt_last.pt` is only written when a run ENDS, so
mid-run the numbered snapshots are the only resumable state that exists. Dropping them would make
a mid-run push carry no way to continue -- which is the whole point of this script.

config.json matters more than it looks: it is re-read at startup, so it is what makes `--resume`
reproduce the profile without re-passing --tracer/--core-dirs. Push it or a resumed run can
silently revert to run-2 settings.
"""
import argparse
import os
import subprocess
import sys
import tempfile

STATE = ["config.json", "splits.json", "train_index.npz", "val_progress.jsonl"]
SELECTED = ["ckpt_best.pt", "ckpt_merge_best.pt", "ckpt_core_best.pt", "ckpt_last.pt"]


def _hf():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("set HF_TOKEN=hf_... (write token for the ScrollData repo)")
    from huggingface_hub import HfApi
    return HfApi(token=tok), tok


def _numbered(work):
    """ckpt_NNNN.pt present in `work`, newest (highest step) first."""
    out = []
    for f in os.listdir(work):
        if f.startswith("ckpt_") and f.endswith(".pt"):
            stem = f[5:-3]
            if stem.isdigit():
                out.append((int(stem), f))
    return [f for _, f in sorted(out, reverse=True)]


def push(args):
    api, tok = _hf()
    num = _numbered(args.work)
    keep_num = num if args.all else num[:1]        # newest snapshot = the only mid-run resume point
    names = [f for f in os.listdir(args.work)
             if f in STATE or f in SELECTED or f in keep_num]
    if not any(f.endswith(".pt") for f in names):
        print("WARNING: no checkpoint found -- pushing run state only; this cannot resume training")
    if not names:
        sys.exit(f"nothing to push in {args.work}")
    tar = os.path.join(tempfile.gettempdir(), f"{args.run}_ckpts.tar.zst")
    print(f"packing {len(names)} files -> {tar}")
    subprocess.run(["tar", "-C", args.work, "-cf", "-", *sorted(names)], check=True,
                   stdout=open(tar + ".raw", "wb"))
    subprocess.run(["zstd", "-q", "-f", "-19", "-T0", tar + ".raw", "-o", tar], check=True)
    os.remove(tar + ".raw")
    dest = f"05_ffn/{args.run}/{args.run}_ckpts.tar.zst"
    print(f"uploading -> {args.repo}:{dest} ({os.path.getsize(tar)/1e9:.2f} GB)")
    api.upload_file(path_or_fileobj=tar, path_in_repo=dest, repo_id=args.repo,
                    repo_type="dataset")
    os.remove(tar)
    print("pushed. resume on a fresh box with:  RESUME=1 bash scripts/bootstrap.sh")


def pull(args):
    api, tok = _hf()
    from huggingface_hub import hf_hub_download
    src = f"05_ffn/{args.run}/{args.run}_ckpts.tar.zst"
    print(f"downloading {args.repo}:{src}")
    f = hf_hub_download(args.repo, src, repo_type="dataset", token=tok)
    # HF's cache stores the file as a SYMLINK into blobs/, and `zstd` refuses to follow symlinks --
    # it warns, writes nothing, and exits 1, so the tar that follows sees a 0-byte stream and the
    # whole restore fails with a misleading "does not look like a tar archive". Resolve it first.
    f = os.path.realpath(f)
    os.makedirs(args.work, exist_ok=True)
    subprocess.run(f'zstd -d -c "{f}" | tar -xf - -C "{args.work}"', shell=True, check=True)
    have = sorted(x for x in os.listdir(args.work) if x.startswith("ckpt_"))
    print(f"restored into {args.work}: {have}")
    print("continue with:  torchrun --nproc_per_node=2 -m scripts.train "
          f"--work {args.work} --resume")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--work", default="/root/surf/ffn_run5")
    ap.add_argument("--run", default=None, help="run name (default: basename of --work)")
    ap.add_argument("--repo", default="nestorvfx/ScrollData")
    ap.add_argument("--all", action="store_true",
                    help="also carry the numbered ckpt_NNNN.pt snapshots (much larger)")
    a = ap.parse_args()
    a.run = a.run or os.path.basename(a.work.rstrip("/"))
    (push if a.action == "push" else pull)(a)
