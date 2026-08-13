"""Per-volume CT intensity statistics.

The air/papyrus boundary from a 2-Gaussian EM fit on the intensity histogram. Deliberately NOT a
percentile: a percentile threshold moves with the AIR FRACTION of whatever cube it is handed, so it
reports different physics for a tightly-packed core and a loose periphery. That is a real defect we hit
before -- percentile thresholds made Scroll-4 look "too compressed" (2 usable sheets) and a physical
threshold on the same data gave 49.
"""
import numpy as np


def air_papyrus_threshold(vol: np.ndarray) -> float:
    """Equal-likelihood boundary between the air and papyrus modes of a uint8 CT volume.

    Otsu for the initial split, then 200 EM iterations on a 2-component 1-D Gaussian mixture, then the
    grey level between the two means where the two component likelihoods cross.
    """
    v = vol[vol > 0]
    if v.size == 0:
        return 0.0
    h = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    x = np.arange(256.0)
    w = h / h.sum()
    om = np.cumsum(w)
    mu = np.cumsum(w * x)
    mt = mu[-1]
    den = om * (1 - om)
    den[den == 0] = 1e-12
    t0 = max(int(np.argmax((mt * om - mu) ** 2 / den)), 1)          # Otsu init

    wa, wp = w[:t0].sum(), w[t0:].sum()
    ma = (x[:t0] * w[:t0]).sum() / max(wa, 1e-9)
    mp = (x[t0:] * w[t0:]).sum() / max(wp, 1e-9)
    sa = np.sqrt(((x[:t0] - ma) ** 2 * w[:t0]).sum() / max(wa, 1e-9)) + 1e-6
    sp = np.sqrt(((x[t0:] - mp) ** 2 * w[t0:]).sum() / max(wp, 1e-9)) + 1e-6
    for _ in range(200):
        pa = wa * np.exp(-0.5 * ((x - ma) / sa) ** 2) / sa
        pp = wp * np.exp(-0.5 * ((x - mp) / sp) ** 2) / sp
        t = pa + pp + 1e-12
        ra, rp = pa / t, pp / t
        na, nb = (ra * w).sum(), (rp * w).sum()
        ma = (ra * w * x).sum() / max(na, 1e-12)
        mp = (rp * w * x).sum() / max(nb, 1e-12)
        sa = np.sqrt((ra * w * (x - ma) ** 2).sum() / max(na, 1e-12)) + 1e-6
        sp = np.sqrt((rp * w * (x - mp) ** 2).sum() / max(nb, 1e-12)) + 1e-6
        wa, wp = na, nb

    g = np.arange(int(ma), int(mp) + 1)
    if not len(g):
        return float(t0)
    la = wa * np.exp(-0.5 * ((g - ma) / sa) ** 2) / sa
    lp = wp * np.exp(-0.5 * ((g - mp) / sp) ** 2) / sp
    return float(g[np.argmin(np.abs(la - lp))])


def wrap_period(vol: np.ndarray) -> float:
    """Local wrap period lambda (voxels) from the 1-D autocorrelation across the layering.

    Measured: ~69 vox on Scroll-3 cores, ~19 on Scroll-4 cores, ~42 on Scroll-1 crops. Anything that
    wants to ask "did this object cross into the neighbouring wrap" must scale by THIS, not by a
    constant -- the same absolute thickness is one sheet in Scroll 3 and three in Scroll 4.

    Cross-layer axis = the axis of strongest mean |gradient| (intensity varies fastest across the
    sheets). Returns 0.0 when no periodicity is found, which callers must treat as "unknown".
    """
    from scipy import ndimage as _ndi
    v = vol.astype(np.float32)
    ax = int(np.argmax([np.abs(np.diff(_ndi.gaussian_filter(v, 1.0), axis=a)).mean()
                        for a in range(3)]))
    v = np.moveaxis(v, ax, -1)
    N = v.shape[-1]
    lines = v.reshape(-1, N)
    lines = lines[::max(1, lines.shape[0] // 4000)]
    lines = lines[lines.std(axis=1) > 3.0]
    if len(lines) < 20:
        return 0.0
    x = lines - lines.mean(axis=1, keepdims=True)
    F = np.fft.rfft(x, axis=1)
    ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :N // 2]
    ac /= (ac[:, :1] + 1e-9)
    m = ac.mean(axis=0)
    d = np.diff(m)
    pk = [i for i in range(3, len(m) - 1) if d[i - 1] > 0 >= d[i]]
    return float(pk[int(np.argmax(m[pk]))]) if pk else 0.0
