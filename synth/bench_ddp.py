#!/usr/bin/env python3
"""Quick empirical test: single-GPU vs 2-GPU DDP throughput on THIS box, with the real nnU-Net ResEnc net at
patch 192^3 (so the gradient all-reduce volume + compute are realistic). P2P is off on the 3090 -> DDP all-reduce
stages through host RAM; this measures whether that overhead makes DDP-on-1-run slower than just running 2 jobs.

Decision rule printed at the end:
  - DDP speedup = (2-GPU aggregate samples/s) / (1-GPU samples/s).
  - If DDP_speedup >= ~1.6  -> DDP is worth it for a SINGLE run.
  - If DDP_speedup <  ~1.6  -> run two single-GPU jobs instead (2x aggregate, no sync overhead).
"""
import os, time, torch, torch.nn as nn
import torch.distributed as dist, torch.multiprocessing as mp

PATCH = (192, 192, 192); ITERS = 20; WARMUP = 4


def make_net():
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=1, n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320], conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6], num_classes=2,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False)


def loop(net, rank):
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(1, 1, *PATCH, device=rank)
    y = torch.randint(0, 2, (1, *PATCH), device=rank)
    lossf = nn.CrossEntropyLoss()
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            out = net(x); out = out[0] if isinstance(out, (list, tuple)) else out
            l = lossf(out, y)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update()
    for _ in range(WARMUP): step()
    torch.cuda.synchronize(rank); t = time.time()
    for _ in range(ITERS): step()
    torch.cuda.synchronize(rank); return time.time() - t


def ddp_worker(rank, world, q):
    os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '12361'
    dist.init_process_group('nccl', rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    net = nn.parallel.DistributedDataParallel(make_net().cuda(rank), device_ids=[rank],
                                              find_unused_parameters=True)
    dt = loop(net, rank)
    if rank == 0: q.put(dt)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # single GPU
    net = make_net().cuda(0)
    dt1 = loop(net, 0)
    del net; torch.cuda.empty_cache()
    sps1 = ITERS / dt1
    # 2-GPU DDP
    q = mp.Queue(); p = [mp.Process(target=ddp_worker, args=(r, 2, q)) for r in range(2)]
    [x.start() for x in p]; dt2 = q.get(); [x.join() for x in p]
    sps_ddp = 2 * ITERS / dt2          # aggregate samples/s (2 GPUs)
    print(f"\n=== RESULT (patch {PATCH}, batch 1/GPU, {ITERS} iters) ===")
    print(f"single-GPU : {sps1:.3f} samples/s  ({dt1/ITERS*1000:.0f} ms/iter)")
    print(f"2-GPU DDP  : {sps_ddp:.3f} samples/s aggregate  ({dt2/ITERS*1000:.0f} ms/iter)")
    print(f"two single-GPU jobs = {2*sps1:.3f} samples/s aggregate")
    spd = sps_ddp / sps1
    print(f"DDP speedup vs 1 GPU: {spd:.2f}x")
    print("VERDICT:", "DDP worth it for a single run" if spd >= 1.6 else
          "run TWO single-GPU jobs (faster aggregate, no sync overhead)")
