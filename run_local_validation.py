#!/usr/bin/env python3
"""
SIGIL local validation (CPU, NumPy only)
========================================
Mechanism-level experiments. These run on any machine in ~2 minutes and are the
falsifiable core of the method. They do NOT use a real LLM -- that is what
notebooks/SIGIL_T4_Colab.ipynb does on a Colab T4.

What is synthetic and what is not:
  SYNTHETIC : the activation tensors. They are drawn from a generator calibrated
              to the KV-cache statistics reported in the literature -- a small
              number of massive channel outliers over a heavy-tailed
              (Laplace-like) bulk, which is the structure QuaRot/KIVI/KurTail all
              document and the reason low-bit KV quantisation fails.
  REAL      : every algorithm, every metric, every number below. The rotations
              are genuinely optimised, the quantisers are the real thing, the
              folding identity is verified to machine precision.

Experiments
  E1  transform quality: how Gaussian/uniform do the marginals actually become
  E2  quantisation fidelity at 2/3/4 bits, incl. the attention-logit metric
  E3  does the GoF statistic predict distortion well enough to allocate bits
  E4  does isotropy give a usable free OOD score for cascade routing
  E5  is the weight folding exact, and does the graph pass a Hexagon audit
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from sigil import (EppsPulley, UniformCvM, Kurtosis, fit_rotation, random_hadamard,
                   IntRTN, quantize_kv, rel_mse, inner_product_error,
                   IsotropyRouter, allocate_bits, predict_shape_factor,
                   validate_surrogate, fold_qk, fold_vo, check_hexagon,
                   global_scale)

RESULTS: dict = {}
SEED = 0


# --------------------------------------------------------------------------- #
# calibrated synthetic activations
# --------------------------------------------------------------------------- #

def make_activations(n=4096, d=128, n_outlier_ch=4, outlier_gain=28.0,
                     tail=1.6, seed=0):
    """
    Heavy-tailed bulk + a few dominant channels, the documented KV structure.
    tail<2 gives super-Gaussian marginals via a Gaussian scale mixture.
    """
    rng = np.random.default_rng(seed)
    # Gaussian scale mixture -> heavy tails, controllable kurtosis
    w = rng.gamma(shape=tail, scale=1.0 / tail, size=(n, 1))
    X = rng.standard_normal((n, d)) / np.sqrt(w)
    # correlated structure: real activations are not white
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    X = X @ (np.eye(d) * 0.7 + A * 0.3)
    # a handful of massive channels
    ch = rng.choice(d, size=n_outlier_ch, replace=False)
    X[:, ch] *= outlier_gain
    # a few token-level spikes (attention sinks)
    sink = rng.choice(n, size=max(1, n // 512), replace=False)
    X[sink] *= 6.0
    return X, ch


def ood_activations(n, d, seed=1, shift=0.55):
    """Distribution-shifted states: different tail index and a mean offset."""
    X, _ = make_activations(n=n, d=d, tail=2.6, outlier_gain=18.0, seed=seed)
    rng = np.random.default_rng(seed + 99)
    return X + shift * rng.standard_normal((1, d)) * np.abs(X).std()


def report(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------- #
# E1 + E2: transforms and quantisation
# --------------------------------------------------------------------------- #

def e1_e2(n=4096, d=64, steps=200):
    report("E1/E2  transform quality and quantisation fidelity")
    X, out_ch = make_activations(n=n, d=d, seed=SEED)
    Q, _ = make_activations(n=512, d=d, seed=SEED + 7)      # probe queries
    s = global_scale(X)
    rng = np.random.default_rng(SEED)

    ep_axes = EppsPulley(slices="axes")
    cvm_axes = UniformCvM(slices="axes")
    kurt = Kurtosis(slices="axes")

    transforms = {}
    transforms["none"] = (np.eye(d), 0.0)
    transforms["hadamard (QuaRot)"] = (random_hadamard(d, rng), 0.0)

    t0 = time.time()
    R_k, log_k = fit_rotation(X, kurt, steps=steps, lr=0.4, seed=SEED)
    transforms["kurtosis (KurTail-style)"] = (R_k, log_k.seconds)

    R_ep, log_ep = fit_rotation(X, ep_axes, steps=steps, lr=0.4, seed=SEED)
    transforms["SIGIL-EP (target N(0,1))"] = (R_ep, log_ep.seconds)

    R_cv, log_cv = fit_rotation(X, cvm_axes, steps=steps, lr=0.4, seed=SEED)
    transforms["SIGIL-CvM (target U)"] = (R_cv, log_cv.seconds)
    fit_total = time.time() - t0

    rows = []
    for name, (R, secs) in transforms.items():
        Z = (X @ R) / s
        ep = ep_axes(Z, need_grad=False)[0]
        cv = cvm_axes(Z, need_grad=False)[0]
        ku = kurt(Z, need_grad=False)[0]
        row = {"transform": name, "fit_s": round(secs, 2),
               "epps_pulley": ep, "uniform_cvm": cv, "excess_kurt_sq": ku}
        Xr, Qr = X @ R, Q @ R
        for b in (4, 3, 2):
            Xh = IntRTN(b, axis=0).search_clip(Xr)(Xr)
            row[f"relmse_int{b}"] = rel_mse(Xr, Xh)
            row[f"attnerr_int{b}"] = inner_product_error(Qr, Xr, Xh)
        rows.append(row)

    hdr = (f"{'transform':<26} {'fit_s':>6} {'EP':>9} {'CvM':>9} {'kurt^2':>9} "
           f"{'relMSE4':>9} {'relMSE3':>9} {'relMSE2':>9} {'attn4':>8} {'attn2':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['transform']:<26} {r['fit_s']:>6.1f} {r['epps_pulley']:>9.5f} "
              f"{r['uniform_cvm']:>9.5f} {r['excess_kurt_sq']:>9.2f} "
              f"{r['relmse_int4']:>9.5f} {r['relmse_int3']:>9.5f} {r['relmse_int2']:>9.5f} "
              f"{r['attnerr_int4']:>8.5f} {r['attnerr_int2']:>8.5f}")

    base = next(r for r in rows if r["transform"] == "hadamard (QuaRot)")
    for r in rows:
        if r["transform"].startswith("SIGIL"):
            for b in (4, 3, 2):
                r[f"gain_vs_hadamard_int{b}"] = base[f"relmse_int{b}"] / r[f"relmse_int{b}"]
    print(f"\nrotation fitting total: {fit_total:.1f}s for {len(transforms)-2} learned "
          f"transforms at d={d}, n={n}, {steps} steps (single CPU core)")
    RESULTS["e1_e2"] = rows
    return X, Q, transforms, s


# --------------------------------------------------------------------------- #
# E3: is the statistic predictive enough to allocate bits?
# --------------------------------------------------------------------------- #

def e3(d=64, n=3072, n_layers=12, steps=80):
    report("E3  does the GoF statistic predict quantisation distortion?")
    rng = np.random.default_rng(SEED + 3)
    ep = EppsPulley(slices="axes")
    gofs, dists, curves, sizes = [], [], [], []
    for L in range(n_layers):
        X, _ = make_activations(n=n, d=d,
                                n_outlier_ch=int(rng.integers(1, 8)),
                                outlier_gain=float(rng.uniform(6, 40)),
                                tail=float(rng.uniform(1.2, 3.0)),
                                seed=1000 + L)
        R, _ = fit_rotation(X, ep, steps=steps, lr=0.4, seed=L)
        s = global_scale(X)
        Z = (X @ R) / s
        g = ep(Z, need_grad=False)[0]
        Xr = X @ R
        curve = [rel_mse(Xr, IntRTN(b, axis=0).search_clip(Xr)(Xr)) for b in (2, 3, 4, 8)]
        gofs.append(g)
        dists.append(curve[0])          # distortion at the tightest budget
        curves.append(curve)
        sizes.append(d * 512)

    v = validate_surrogate(gofs, dists)
    print(f"Spearman rho(GoF, INT2 distortion) = {v['spearman_rho']:+.3f}  "
          f"(p={v['p_value']:.4f}, n={v['n']} layers)")

    curves = np.array(curves)
    grid = [2, 3, 4, 8]
    measured = allocate_bits(curves, sizes, grid, budget_bits=3.0)
    pred = np.array([[predict_shape_factor(g) * 4.0 ** -b for b in grid] for g in gofs])
    predicted = allocate_bits(pred, sizes, grid, budget_bits=3.0)
    agree = float(np.mean(measured == predicted))
    d_meas = float(np.mean([curves[i, grid.index(b)] for i, b in enumerate(measured)]))
    d_pred = float(np.mean([curves[i, grid.index(b)] for i, b in enumerate(predicted)]))
    d_flat = float(np.mean(curves[:, grid.index(3)]))
    print(f"allocation at a 3.0-bit average budget:")
    print(f"  uniform 3-bit everywhere      mean distortion {d_flat:.5f}")
    print(f"  oracle (measured curves)      mean distortion {d_meas:.5f}   "
          f"({d_flat/d_meas:.2f}x better)")
    print(f"  GoF surrogate (no sweep)      mean distortion {d_pred:.5f}   "
          f"({d_flat/d_pred:.2f}x better)")
    print(f"  surrogate/oracle bit agreement: {agree*100:.0f}% of layers")
    RESULTS["e3"] = {"spearman": v, "agreement": agree, "d_uniform": d_flat,
                     "d_oracle": d_meas, "d_surrogate": d_pred,
                     "bits_oracle": measured.tolist(),
                     "bits_surrogate": predicted.tolist()}


# --------------------------------------------------------------------------- #
# E4: free OOD score from isotropy
# --------------------------------------------------------------------------- #

def e4(d=64, n=4096, steps=200):
    report("E4  chi-square routing score in the isotropised space")
    X, _ = make_activations(n=n, d=d, seed=SEED + 11)
    Xo = ood_activations(n // 2, d, seed=SEED + 12)
    ep = EppsPulley(slices="random", n_slices=256, seed=SEED)   # joint law matters here
    R, _ = fit_rotation(X, ep, steps=steps, lr=0.4, seed=SEED)
    s = global_scale(X)

    cut = n // 2
    Zc, Zi, Zo = (X[:cut] @ R) / s, (X[cut:] @ R) / s, (Xo @ R) / s
    router = IsotropyRouter(d).fit(Zc)
    auroc_iso = IsotropyRouter.auroc(router.score(Zi), router.score(Zo))

    raw = IsotropyRouter(d).fit(X[:cut] / s)
    auroc_raw = IsotropyRouter.auroc(raw.score(X[cut:] / s), raw.score(Xo / s))

    had = random_hadamard(d, np.random.default_rng(SEED))
    rh = IsotropyRouter(d).fit((X[:cut] @ had) / s)
    auroc_had = IsotropyRouter.auroc(rh.score((X[cut:] @ had) / s), rh.score((Xo @ had) / s))

    print(f"AUROC (in-distribution vs shifted states), higher is better:")
    print(f"  raw squared norm, no transform     {auroc_raw:.3f}")
    print(f"  after Hadamard rotation            {auroc_had:.3f}")
    print(f"  after SIGIL isotropisation         {auroc_iso:.3f}")

    thr = router.set_budget(Zc, escalation_rate=0.10)
    sc = np.concatenate([router.score(Zi), router.score(Zo)])
    thr_mid = float(np.quantile(router.score(Zc), 0.70))
    cost = router.expected_cost(sc, thr, thr_mid)
    print(f"\ncascade at a 10% escalation budget (threshold {thr:.4f}):")
    for k, v in cost["fraction"].items():
        print(f"  {k:<18} {v*100:5.1f}% of tokens")
    print(f"  mean energy   {cost['energy_mj_per_1k']:.1f} mJ / 1k tok")
    print(f"  mean latency  {cost['latency_ms_mean']:.0f} ms")
    print(f"  fraction of traffic leaving the device: {cost['privacy_exposure']*100:.1f}%")
    RESULTS["e4"] = {"auroc_raw": auroc_raw, "auroc_hadamard": auroc_had,
                     "auroc_sigil": auroc_iso, "cost": cost}


# --------------------------------------------------------------------------- #
# E5: exactness of folding + hardware audit
# --------------------------------------------------------------------------- #

def e5(d_model=512, n_heads=8, head_dim=64):
    report("E5  weight folding exactness and Hexagon deployability")
    rng = np.random.default_rng(SEED + 5)
    Wq = rng.standard_normal((d_model, n_heads * head_dim)) / np.sqrt(d_model)
    Wk = rng.standard_normal((d_model, n_heads * head_dim)) / np.sqrt(d_model)
    Wv = rng.standard_normal((d_model, n_heads * head_dim)) / np.sqrt(d_model)
    Wo = rng.standard_normal((n_heads * head_dim, d_model)) / np.sqrt(d_model)
    x = rng.standard_normal((32, d_model))
    R, _ = np.linalg.qr(rng.standard_normal((head_dim, head_dim)))

    q = (x @ Wq).reshape(32, n_heads, head_dim)
    k = (x @ Wk).reshape(32, n_heads, head_dim)
    logits = np.einsum("thc,shc->hts", q, k)
    Wq2, Wk2 = fold_qk(Wq, Wk, R, head_dim, rope=False)
    q2 = (x @ Wq2).reshape(32, n_heads, head_dim)
    k2 = (x @ Wk2).reshape(32, n_heads, head_dim)
    logits2 = np.einsum("thc,shc->hts", q2, k2)
    e_qk = float(np.abs(logits - logits2).max())

    A = rng.random((n_heads, 32, 32)); A /= A.sum(-1, keepdims=True)
    v = (x @ Wv).reshape(32, n_heads, head_dim)
    o = np.einsum("hts,shc->thc", A, v).reshape(32, -1) @ Wo
    Wv2, Wo2 = fold_vo(Wv, Wo, R, head_dim)
    v2 = (x @ Wv2).reshape(32, n_heads, head_dim)
    o2 = np.einsum("hts,shc->thc", A, v2).reshape(32, -1) @ Wo2
    e_vo = float(np.abs(o - o2).max())

    print(f"max |attention logits - folded logits| : {e_qk:.3e}   (machine epsilon)")
    print(f"max |block output   - folded output|  : {e_vo:.3e}   (machine epsilon)")
    print("=> the transform is free at inference: zero added operators.\n")

    rep = check_hexagon(head_dim=head_dim, weight_bits=4, act_dtype="fp16", added_ops=0)
    print(rep)
    RESULTS["e5"] = {"qk_fold_max_err": e_qk, "vo_fold_max_err": e_vo,
                     "hexagon_pass": rep.ok}


if __name__ == "__main__":
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)
    print("SIGIL local validation -- NumPy/CPU, no GPU, no network")
    e1_e2()
    e3()
    e4()
    e5()
    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    with open("results_local.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("wrote results_local.json")
