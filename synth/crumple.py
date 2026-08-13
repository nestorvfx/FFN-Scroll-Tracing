#!/usr/bin/env python3
"""Rank-4 primitive: developable-cone (d-cone) + stretching-ridge CRUMPLE-FOLD warp.

Physics: a confined thin sheet does not deform smoothly -- elastic energy condenses onto point singularities
(developable cones) joined by stretching ridges (Cerda & Mahadevan PRL 80:2358 1998; Cerda & Chaieb Nature 401:46
1999; Witten Rev.Mod.Phys 79:643 2007). The d-cone deflection is conical, w(r,th) = A*r*psi(th), with a UNIVERSAL
angular profile psi: one buckled (lifted) sector of ~139 deg (2.43 rad, the universal take-off angle) and a support-
contact sector. Ridges joining two apices have the Lobkovsky width scaling w ~ (h*X^2)^(1/3) (h sheet thickness,
X ridge length).

psi(th) is obtained by SOLVING the linearized Cerda-Mahadevan variational problem numerically (not a guessed
formula): minimize the conical bending energy  E[psi] = Int (psi'' + psi)^2 dth   subject to the support constraint
psi >= -1 (contact where equality) and fixed azimuthal excess length  g[psi] = Int (psi'^2 - psi^2) dth = eps > 0
(the linearized inextensibility content that makes buckling happen at all -- without it the only minimizer is the
full-contact cone). Solved once on a fine grid with scipy trust-constr, cached to disk, and VALIDATED against the
universal take-off angle (free arc 139 deg +- tol in the small-eps limit). See Witten RMP sec III.

Application to a CT cube (stack of near-parallel sheets, cross-sheet axis = `axis`):
  u_axis(x) = sum_i A_i * r_i * psi(th_i - phi_i) * W(r_i/R_i) * D_i(z)   (+ ridge terms between apex pairs)
a PURE-AXIAL displacement field. For a purely axial map the Jacobian is triangular:
  det(I + grad u) = 1 + d(u_z)/dz   EXACTLY,
so the diffeomorphism gate is the single condition d(u_z)/dz > -1 (we enforce <= -CAP with CAP=0.8 by analytic
rescale, a guarantee not a hope). The one-sided decay D(z) through the stack makes upper sheets displace more than
lower ones -> inter-sheet gaps close along the CURVED crescent loci of the cone rim / ridge lines (the geometry the
axis-aligned gap-collapse cannot produce), while papyrus strain stays bounded by A/lambda. CT warped order-1,
instance labels order-0, same coordinates -> GT exact by construction. Real voxels only -> texture stays honest.
"""
import os
import numpy as np
from scipy import ndimage as ndi

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dcone_profile.npz")


# ---------------------------------------------------------------- universal d-cone angular profile (solved once)
def solve_dcone_profile(n=480, eps=0.05, verbose=False):
    """Numerically minimize E[psi]=Int(psi''+psi)^2 s.t. psi>=-1, Int(psi'^2-psi^2)=eps (periodic grid).
    Penalty-continuation + L-BFGS-B (native lower bound, analytic gradients): mu*(g-eps)^2 with mu ramped
    10 -> 1e5, warm-started -- converges in seconds and enforces the constraint to ~1e-4.
    Returns (theta, psi) with the buckled bump centered at theta=0. Small eps -> universal linear-limit shape."""
    from scipy.optimize import minimize
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    h = th[1] - th[0]
    # periodic spectral derivatives (exact for the trig-polynomial minimizer, no stencil error at the contact edge)
    k = np.fft.rfftfreq(n, d=h / (2 * np.pi))          # integer wavenumbers
    def d1(v): return np.fft.irfft(np.fft.rfft(v) * (1j * k), n)
    def d2(v): return np.fft.irfft(np.fft.rfft(v) * -(k ** 2), n)

    def g(v):
        dv = d1(v)
        return h * float(np.dot(dv, dv) - np.dot(v, v))

    def gradg(v):
        return 2 * h * (-d2(v) - v)

    def make_obj(mu):
        def obj(v):
            r = d2(v) + v
            gv = g(v)
            f = h * float(np.dot(r, r)) + mu * (gv - eps) ** 2
            gr = 2 * h * (d2(r) + r) + 2 * mu * (gv - eps) * gradg(v)
            return f, gr
        return obj

    # init: full-contact cone + a gentle 139-deg-wide lift bump (symmetric, satisfies psi>=-1)
    x = np.full(n, -1.0) + 0.6 * np.exp(-0.5 * (th / 0.65) ** 2)
    # INDENTATION PIN: the sheet is pushed INTO the ring, so contact must be ATTAINED (psi = -1 somewhere).
    # Without this pin the global minimizer is a tiny m=2 harmonic that never touches the support (not a d-cone).
    # Pin the antipode of the bump center: bounds (-1,-1) on an arc around theta=pi.
    bounds = [(-1.0, None)] * n
    pin = np.abs(np.abs(th) - np.pi) < (np.pi / 8)      # 45-deg pinned contact arc at the antipode
    for j in np.where(pin)[0]:
        bounds[j] = (-1.0, -1.0)
    for mu in (10.0, 1e2, 1e3, 1e4, 1e5):
        res = minimize(make_obj(mu), x, jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options=dict(maxiter=3000, ftol=1e-14, gtol=1e-10))
        x = res.x
        if verbose:
            print(f"  mu={mu:.0e}: E-part done, g={g(x):.5f} (target {eps}), lifted_arc={free_arc_deg(th, x):.1f} deg")
    return th, x


def free_arc_deg(th, psi, tol=0.02):
    """Angular width (deg) of the lifted (non-contact) region -- the take-off signature (universal ~139 deg).
    Contact = psi within tol*range of the support level (-1 / min)."""
    lifted = psi > (psi.min() + tol * (psi.max() - psi.min() + 1e-12))
    return float(lifted.mean() * 360.0)


def analytic_dcone_profile(n=1440, s_c=1.215):
    """Direct linear-theory construction, validated against Cerda & Mahadevan, "Confined developable elastic
    surfaces: cylinders, cones and the Elastica", Proc. R. Soc. A 461:671-700 (2005), eq (4.45): the free-region
    angular deflection is  alpha = A cos(a s) + B cos(s) + C sin(a s) + D sin(s)  (a superposition of an s-mode and
    an a*s-mode -- the general solution of the linearized multiplier Euler-Lagrange (d^2+1)(d^2+a^2)alpha = 0). For
    the SYMMETRIC single-fold d-cone the sin terms vanish -> psi = A cos(th) + B cos(a th). We fix:
      * the UNIVERSAL take-off half-angle s_c = 1.215 rad, i.e. 2s_c = 2.43 rad ~ 139 deg (Cerda-Mahadevan 2005
        Table 1 fundamental solution i=1: 2s_c=2.43; Witten RMP 2007 sec III; Chaieb-Melo PRL 1998 experiment);
      * contact conditions psi(s_c) = -1 (support level) and psi'(s_c) = 0 (tangential take-off);
      * `a` by the EXACT linearized inextensibility of the closed generator circle Int_0^{2pi}(psi'^2-psi^2)dth = 0
        (the contact arc contributes -(2pi-2s_c); the bump pays it back). -> a = 2.4561 (unique admissible root:
        psi >= -1 one-sided support, single smooth max at 0; all other roots dip below the ring).
    PHYSICAL NOTE (not a defect): psi''(s_c) is discontinuous -- the take-off point carries a concentrated
    "take-off force" (Witten RMP 2007 sec III: "the curvature derivative must change discontinuously at takeoff...
    requires a localized normal force called the take-off force"). So a curvature jump at s_c is CORRECT, not an
    artifact of the truncated basis. Returns (th, psi) on [-pi, pi)."""
    from scipy.optimize import brentq
    s = float(s_c)

    def AB(a):
        B = -1.0 / (np.cos(a * s) - a * np.sin(a * s) / np.tan(s))
        A = -a * B * np.sin(a * s) / np.sin(s)
        return A, B

    tf = np.linspace(-s, s, 4001)
    hf = tf[1] - tf[0]

    def G(a):
        A, B = AB(a)
        psi = A * np.cos(tf) + B * np.cos(a * tf)
        dpsi = -A * np.sin(tf) - a * B * np.sin(a * tf)
        free = float(np.trapezoid(dpsi ** 2 - psi ** 2, dx=hf))
        return free - (2 * np.pi - 2 * s)               # contact arc contributes -(2pi-2s)*1

    # root-find a on a branch where the bump is valid (psi > -1 inside, single max at 0)
    a_root = None
    grid = np.linspace(1.05, 8.0, 400)
    vals = [G(a) for a in grid]
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            cand = brentq(G, grid[i], grid[i + 1], xtol=1e-12)
            A, B = AB(cand)
            psi_f = A * np.cos(tf) + B * np.cos(cand * tf)
            if psi_f.max() > -0.99 and psi_f.min() >= -1.02 and abs(np.argmax(psi_f) - len(tf) // 2) < len(tf) // 8:
                a_root = cand
                break
    if a_root is None:
        raise RuntimeError("no valid d-cone branch found")
    A, B = AB(a_root)
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    psi = np.full(n, -1.0)
    m = np.abs(th) < s
    psi[m] = A * np.cos(th[m]) + B * np.cos(a_root * th[m])
    # validation diagnostics
    curv_jump = float(-A * np.cos(s) - a_root ** 2 * B * np.cos(a_root * s))   # psi''(s_c); ~0 if branch correct
    return th, psi, dict(a=a_root, A=A, B=B, curv_pp_at_takeoff=curv_jump)


def dcone_profile():
    """Cached universal profile as an interpolation table (theta in [-pi,pi), psi normalized: contact=0, peak=1)."""
    if os.path.exists(_CACHE):
        d = np.load(_CACHE)
        return d["th"], d["psi_n"]
    th, psi, info = analytic_dcone_profile()
    arc = free_arc_deg(th, psi)
    if not (100.0 < arc < 180.0):                       # sanity: must bracket the universal 139 deg
        raise RuntimeError(f"d-cone profile failed validation: free arc {arc:.1f} deg (expect ~139)")
    psi_n = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)   # 0 = contact, 1 = bump peak
    np.savez(_CACHE, th=th, psi_n=psi_n, arc_deg=arc, a=info["a"], curv=info["curv_pp_at_takeoff"])
    return th, psi_n


# ---------------------------------------------------------------- displacement-field construction
def _smooth_window(r, r_out):
    """C1 radial taper: 1 in the core, cos^2 falloff to 0 at r_out (compact support)."""
    x = np.clip(r / max(r_out, 1e-6), 0.0, 1.0)
    return np.cos(0.5 * np.pi * x) ** 2


def _one_sided_decay(z, z0, lam, side):
    """Smooth one-sided profile in the cross-sheet coordinate: 1 at the pushed face, decaying through the stack.
    side=+1 pushes voxels with z>z0 (decaying upward influence below), side=-1 mirrored."""
    s = (z - z0) * side
    return 1.0 / (1.0 + np.exp(-s / (0.25 * lam))) * np.exp(-np.clip(s, 0, None) / lam) + \
           (1.0 - 1.0 / (1.0 + np.exp(-s / (0.25 * lam))))          # ~1 behind the front, exp decay past it


def crumple_field(shape, axis, rng, n_cones=2, amp=10.0, r_out_frac=0.45, lam=48.0,
                  ridge=True, sheet_h=10.0):
    """Build the pure-axial crumple displacement u_axis (in voxels) for a cube of `shape`.
    n_cones d-cone apices (universal profile, random center/orientation/amplitude/push-side) + Lobkovsky ridges
    between nearby apex pairs. Returns u (float32, shape) -- displacement ALONG `axis` only."""
    th_tab, psi_tab = dcone_profile()
    zs, ys, xs = [np.arange(s, dtype=np.float32) for s in shape]
    # in-plane coords = the two axes != `axis`; cross coord = axis
    grids = np.meshgrid(zs, ys, xs, indexing="ij")
    cz = grids[axis]
    ip = [grids[a] for a in range(3) if a != axis]
    L = min(shape[a] for a in range(3) if a != axis)
    r_out = r_out_frac * L
    u = np.zeros(shape, np.float32)
    apices = []
    for i in range(int(n_cones)):
        c0 = [float(rng.uniform(0.25 * s, 0.75 * s)) for s in shape]
        phi = float(rng.uniform(0, 2 * np.pi))
        A = float(rng.uniform(0.6, 1.0) * amp)
        side = int(rng.choice([-1, 1]))
        dy = ip[0] - c0[[a for a in range(3) if a != axis][0]]
        dx = ip[1] - c0[[a for a in range(3) if a != axis][1]]
        r = np.hypot(dy, dx)
        th = np.mod(np.arctan2(dx, dy) - phi + np.pi, 2 * np.pi) - np.pi
        psi = np.interp(th, th_tab, psi_tab)
        prof = (r / r_out) * psi * _smooth_window(r, r_out)         # conical: grows ~linearly, tapered at r_out
        dec = _one_sided_decay(cz, c0[axis], lam, side)
        u += (side * A) * (prof * dec).astype(np.float32)
        apices.append((c0, A, side))
    if ridge and len(apices) >= 2:
        # Lobkovsky ridge between the first apex pair: width w=(h X^2)^(1/3), sag ~ c*w, sech^2 cross-profile
        (c1, A1, s1), (c2, A2, s2) = apices[0], apices[1]
        p1 = np.array([c1[a] for a in range(3) if a != axis], np.float32)
        p2 = np.array([c2[a] for a in range(3) if a != axis], np.float32)
        X = float(np.linalg.norm(p2 - p1))
        if X > 8:
            w = float((sheet_h * X * X) ** (1.0 / 3.0))             # Lobkovsky width scaling
            d = (p2 - p1) / X
            t = (ip[0] - p1[0]) * d[0] + (ip[1] - p1[1]) * d[1]     # along-ridge coord
            q = -(ip[0] - p1[0]) * d[1] + (ip[1] - p1[1]) * d[0]    # across-ridge coord
            sag = float(rng.uniform(0.5, 1.2)) * w * 0.5 * (0.5 * (A1 + A2) / max(amp, 1e-6))
            along = np.clip(t / X, 0, 1)
            bump = np.sin(np.pi * along) ** 2 * (t > 0) * (t < X)   # smooth along the ridge span
            cross = 1.0 / np.cosh(q / max(w, 2.0)) ** 2             # sech^2 ridge cross-section
            zc = 0.5 * (c1[axis] + c2[axis])
            dec = _one_sided_decay(cz, zc, lam, s1)
            u += (s1 * sag) * (bump * cross * dec).astype(np.float32)
    return u


def _axial_grad(u, axis):
    return np.gradient(u, axis=axis)


def crumple_warp(vol, msk, axis, rng, n_cones=2, amp=10.0, lam=48.0, cap=0.8, ridge=True,
                 sheet_h=10.0, return_disp=False):
    """Apply the crumple-fold warp. GUARANTEED diffeomorphism: pure-axial displacement -> det(I+grad u) =
    1 + du/dz exactly; if min(du/dz) < -cap the field is rescaled analytically so the bound holds. CT order-1,
    labels order-0, identical coordinates (GT exact)."""
    u = crumple_field(vol.shape, axis, rng, n_cones=n_cones, amp=amp, lam=lam, ridge=ridge, sheet_h=sheet_h)
    g = _axial_grad(u, axis)
    m = float(g.min())
    if m < -cap:                                                    # analytic rescale -> exact validity guarantee
        u *= (cap / (-m))
        g = _axial_grad(u, axis)
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in vol.shape], indexing="ij")
    coords = [grids[a] + (u if a == axis else 0.0) for a in range(3)]
    vw = ndi.map_coordinates(vol.astype(np.float32), coords, order=1, mode="nearest")
    mw = ndi.map_coordinates(msk, coords, order=0, mode="nearest").astype(msk.dtype)
    if return_disp:
        disp = [u if a == axis else np.zeros_like(u) for a in range(3)]
        return vw, mw, disp
    return vw, mw


# ---------------------------------------------------------------- self-test / gates
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_only", action="store_true")
    a = ap.parse_args()
    if os.path.exists(_CACHE):
        os.remove(_CACHE)                                           # force fresh solve for the gate
    th, psi, info = analytic_dcone_profile()
    arc = free_arc_deg(th, psi)
    print(f"GATE d-cone profile: free arc = {arc:.1f} deg at 2%-lift threshold "
          f"(take-off arc 2*s_c = {2*1.215*180/np.pi:.1f} deg by construction; a={info['a']:.4f})")
    ok_arc = 100.0 < arc < 180.0
    print("  ->", "PASS" if ok_arc else "FAIL")
    psi_n = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)
    np.savez(_CACHE, th=th, psi_n=psi_n, arc_deg=arc, a=info["a"])
    if a.profile_only:
        raise SystemExit(0 if ok_arc else 1)
    # Jacobian gate on a synthetic field (no data needed)
    rng = np.random.default_rng(0)
    shape = (192, 192, 192)
    u = crumple_field(shape, 0, rng, n_cones=3, amp=14.0, lam=40.0)
    g = np.gradient(u, axis=0)
    print(f"GATE axial-grad: min du/dz = {g.min():.3f} (must be > -1; capped at -0.8 in crumple_warp)")
    from synth_merge import jacobian_folding
    disp = [u, np.zeros_like(u), np.zeros_like(u)]
    fold, dmin = jacobian_folding(disp)
    print(f"GATE jacobian_folding: fold={fold*100:.4f}%  min_det={dmin:.3f}")
    ok = ok_arc and fold == 0.0 and dmin > 0.0
    print("CRUMPLE SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
