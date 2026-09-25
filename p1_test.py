#!/usr/bin/env python3
"""
P1 -- PSDC's FATAL prediction, tested precisely.

The naive framing ("does a pruned subnet still work?") conflates two effects:
   (A) depth pruning degrades the model            -- KNOWN, Minitron/Shortened
                                                      LLaMA, recoverable by distillation
   (B) reading KV computed by the FULL network     -- PSDC'S ACTUAL CLAIM, untested

Only (B) is PSDC-specific. So the experiment holds pruning fixed and varies only
the KV source:

   pruned_own  : subnet computes its own K,V from its own residual stream
   pruned_full : subnet reads K,V cached by the full L-layer network  <- PSDC

If pruned_full ~= pruned_own, KV sharing is free and PSDC's mechanism is sound;
whatever degradation remains is ordinary depth pruning, which is Minitron's
problem and has a known fix. If pruned_full is much worse, PSDC is dead.

This needs no training: it is a question about residual-stream compatibility,
measured as divergence from the full model's own output distribution.
"""
import numpy as np
rng = np.random.default_rng(0)

V, T, D, H, L = 32, 24, 64, 4, 12
HD = D // H

def init():
    s = lambda *sh: rng.standard_normal(sh) / np.sqrt(sh[0])
    p = {"emb": s(V, D), "pos": rng.standard_normal((T, D)) * 0.02, "out": s(D, V)}
    for l in range(L):
        for n in ("q","k","v","o"): p[f"{n}{l}"] = s(D, D)
        p[f"w1{l}"], p[f"w2{l}"] = s(D, 2*D), s(2*D, D)
    return p

def ln(x): return (x - x.mean(-1,keepdims=True)) / (x.std(-1,keepdims=True) + 1e-5)

def block(x, p, l, cache=None, use_cached=False):
    h = ln(x); B,S,_ = h.shape
    q = (h @ p[f"q{l}"]).reshape(B,S,H,HD).transpose(0,2,1,3)
    if use_cached:
        k, v = cache[l]
    else:
        k = (h @ p[f"k{l}"]).reshape(B,S,H,HD).transpose(0,2,1,3)
        v = (h @ p[f"v{l}"]).reshape(B,S,H,HD).transpose(0,2,1,3)
        if cache is not None: cache[l] = (k, v)
    a = q @ k.transpose(0,1,3,2) / np.sqrt(HD)
    a = np.where(np.tril(np.ones((S,S)))[None,None] > 0, a, -1e9)
    a = np.exp(a - a.max(-1,keepdims=True)); a /= a.sum(-1,keepdims=True)
    x = x + (a @ v).transpose(0,2,1,3).reshape(B,S,D) @ p[f"o{l}"]
    m = ln(x) @ p[f"w1{l}"]
    return x + np.maximum(m, 0.01*m) @ p[f"w2{l}"]

def run(X, p, layers=None, cache=None, use_cached=False):
    layers = range(L) if layers is None else layers
    x = p["emb"][X] + p["pos"][None,:X.shape[1]]
    for l in layers: x = block(x, p, l, cache, use_cached)
    return ln(x) @ p["out"]

def probs(z):
    z = z - z.max(-1,keepdims=True); e = np.exp(z); return e / e.sum(-1,keepdims=True)

def kl(a, b):
    return float(np.mean((a * (np.log(a+1e-12) - np.log(b+1e-12))).sum(-1)))

print("P1 -- does sharing FULL-network KV cost anything beyond the pruning itself?\n")
print(f"transformer: {L} layers, d={D}, {H} heads, vocab={V}")
print("(untrained: this is a question about residual-stream compatibility,")
print(" not about task accuracy -- see the caveat at the end)\n")

X = rng.integers(0, V, size=(64, T))
p = init()
cache = {}
P_full = probs(run(X, p, cache=cache))
uniform = np.full_like(P_full, 1.0/V)
print(f"reference: KL(full || uniform) = {kl(P_full, uniform):.4f}  "
      f"(scale for what 'far' means)\n")

print(f"{'retained':<14}{'frac':>6}{'KL own-KV':>12}{'KL full-KV':>12}{'ratio':>9}  verdict")
print("-"*70)
rows=[]
for frac in (0.75, 0.5, 0.33, 0.25):
    step = int(round(1/frac))
    keep = list(range(0, L, step)) if frac < 0.75 else [l for l in range(L) if l % 4 != 3]
    P_own  = probs(run(X, p, layers=keep))
    P_shar = probs(run(X, p, layers=keep, cache=cache, use_cached=True))
    k_own, k_shar = kl(P_full, P_own), kl(P_full, P_shar)
    r = k_shar / max(k_own, 1e-12)
    rows.append((frac, k_own, k_shar, r))
    verdict = "KV sharing ~free" if r < 1.3 else ("mild cost" if r < 2 else "KV SHARING HURTS")
    print(f"{str(keep)[:13]:<14}{frac:>6.2f}{k_own:>12.4f}{k_shar:>12.4f}{r:>8.2f}x  {verdict}")

print()
worst = max(r for *_, r in rows)
print(f"worst ratio across depths: {worst:.2f}x")
print(f"PSDC kill condition: sharing full KV is much worse than own KV (>2x).")
print(f"=> P1 {'FAILS' if worst > 2 else 'SURVIVES'} on this test.\n")
print("CAVEAT, stated plainly: this is an UNTRAINED model, so it tests whether the")
print("residual stream stays compatible when layers are skipped -- the mechanism")
print("PSDC depends on -- NOT whether task accuracy survives. A trained model could")
print("behave differently. This narrows P1; it does not close it.")
