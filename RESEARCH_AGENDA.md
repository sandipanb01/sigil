# RESEARCH AGENDA

What in this project is actually publishable, ranked, with the experiment that
would settle each one. Written after the fact, so the novelty assessments are
informed by what the literature search turned up rather than by hope.

**Calibration first.** Most of what felt like a discovery here was prior art:
sequential quantisation is GPTQ, disjoint supports are AWQ/SqueezeLLM, confidence
routing ships in Cactus, phase-split kernels are Vec-LUT. Four ideas survive that
filter. Two are strong.

---

## Tier 1 — genuinely open, worth writing up

### R1. Does ternary retention track WIDTH or PARAMETER COUNT?

**Status: open. Strongest idea here. Publishable either way.**

BitCPM-CANN measured ternary QAT retention against **total parameter count**:
0.5B → 90.1%, 1B → 95.7%, 3B → 97.2%, 8B → 95.7%. Total params confounds width
and depth, and nobody has separated them.

**Why it matters beyond curiosity.** PSDC's decode network is *depth*-pruned: it
keeps full width while its parameter count drops below the 1B danger zone. If
retention tracks width, the subnet stays ternary-viable and the memory win holds.
If it tracks total params, the decode net must stay wide and PSDC's economics
change. The same question governs every depth-pruned-then-quantised pipeline,
which is the standard edge recipe.

**Experiment.** Two (better, four) QAT runs at matched total parameters and
different aspect ratios — deep+narrow vs shallow+wide — evaluated against their
own fp16 baselines on a fixed benchmark set. Add a third arm at matched *width*
but different depth to separate the axes cleanly.

**What I tried and why it failed, three times.** I attempted a cheap proxy on
random matrices at fixed parameter budget: without residuals, with residuals, and
with depth-scaled residuals. All three saturated — relative error 1.1–1.6 across
every shape, meaning every configuration was equally destroyed. **Ternarising
random Gaussian weights destroys them regardless of shape**, because retention
depends on trained redundancy that random matrices do not have. That is precisely
why BitCPM's numbers come from trained models.

**So the cheap version does not exist.** This needs real QAT. Estimated cost:
4 × (small-model QAT run), which is days on a modest GPU, not hours on a T4.
`psdc.py falsify` lists it as P3.

**Predicted outcome, stated in advance so it can be wrong:** retention tracks
width more strongly than total params, because ternary quantisation error is
per-weight and averages over a layer's fan-in — wider layers have more terms to
average. If so, `ternary_viability()` in the engine is currently keyed on the
wrong variable and should be re-keyed on hidden size.

---

### R2. Quantise on the Lie algebra, not the group

**Status: novel as far as the literature search reached. Implemented and measured
on synthetics; needs real weights.**

Rotation-based quantisation (QuaRot, SpinQuant, and this project's own folding)
depends on R being exactly orthogonal. Deployment requires *storing* R, and
storing means *quantising* R — which leaves the orthogonal group and silently
voids the exactness the design rests on. Measured: at 2 bits, ‖RRᵀ−I‖ = 0.615 and
attention-logit error 98%.

**The fix** (transplanted from the Navier–Stokes localization argument: never
operate on the constrained object, operate on its potential): store the skew
generator A with R = Cayley(A). Cayley of any skew matrix is exactly orthogonal,
so the constraint survives at any bit width. Measured: ‖RRᵀ−I‖ ≈ 1e-15 at 8, 4
and 2 bits.

**The non-obvious part, which is what makes it a paper.** Generator quantisation
changes *which* rotation you get — 71% Frobenius drift at 2 bits — and it does not
matter: 93% of the outlier-spreading benefit is retained, still 3.75× better than
no rotation. Because incoherence is a **generic** property of rotations rather
than a property of one specific R, drifting inside the group is harmless while
leaving it is fatal. That asymmetry is the result.

**Payoff:** 12.9× smaller rotation storage at d=128, with the folding identity
exact instead of approximate.

**Generalisation — now implemented and measured, not just asserted.** The same
move applies to any parameter constrained to a manifold with a chart from an
unconstrained space. Tested three:

| manifold | chart | constraint | direct @ 4 bits | chart @ 2 bits |
|---|---|---|---|---|
| SO(d) | Cayley / expm of skew | RRᵀ = I | 4.75e-2 | **1.0e-15** |
| SPD(d) | expm of symmetric | eigenvalues > 0 | **min eig −1.12** | **min eig +1.8e-2** |
| Stiefel(n,p) | expm of skew, first p cols | UᵀU = I | 5.45e-2 | **1.7e-15** |

**The SPD row is the sharper argument, and probably the paper's lead.** A rotation
that drifts off SO(d) degrades gracefully. A "covariance" with a negative
eigenvalue is not a covariance — Cholesky, Mahalanobis distance, natural gradient
and any Gaussian likelihood fail *outright* rather than degrading. Direct 4-bit
quantisation produced a minimum eigenvalue of **−1.12**. That is a type error, not
a small one. SPD blocks appear in Fisher/K-FAC preconditioners, covariance heads
and Gaussian-process layers, all of which are quantised naively today.

And the chart-quantised object still works: at 2 bits the rotation drifted 84.5%
in Frobenius norm and still gave relMSE 0.01154 against 0.01055 exact — 9% worse,
still 3.2× better than no rotation.

**Experiment needed (G1 in `generator_quant.py`):** take per-head rotations from a
real QuaRot/SpinQuant pipeline, store as 2-bit generators, reconstruct, measure
WikiText-2 perplexity against exact-R and against 8-bit direct storage at matched
total bits. One T4 session.

---

## Tier 2 — real but narrower

### R3. The backend axis is arithmetic intensity × stored bit width, not device class

Two results compose into a claim nobody states cleanly. Vec-LUT: the optimal
kernel depends on whether inference is parallel or sequential — and its own repo
lists *speculative decoding* among the parallel scenarios, so the axis is not
prefill-vs-decode. This project's roofline: decode throughput is set by **stored
bit width on the wire**, not TOPS, because a backend lacking a native low-bit
kernel must widen to int8 and streams 4× the bytes.

Together: the right backend is a function of (arithmetic intensity, stored bit
width). Device class — "NPU beats CPU" — is the wrong abstraction, and this
project's own selector encoded it until T-MAN forced a correction.

**Why only Tier 2:** each half is published; the synthesis is a systems paper at
best, and it needs real measurements across a device × bit-width × parallelism
grid to be worth anything. AI Hub Workbench makes that free but not quick.

### R4. Select QAT checkpoints on deployed loss, not clean loss

Measured, 3-bit, group=32: the best *deployed* model is **4.6× worse before
quantisation and 6.6× better after it**, and there is a genuine interior optimum
in regulariser strength — push further and clean loss dominates again.

The corollary is that the right regulariser is a per-group **L∞** penalty, not L2
weight decay: Δ is set by the group maximum, which an outlier dominates while
contributing almost nothing to the norm.

**Why only Tier 2:** QAT implicitly optimises deployed loss already, so the
framing is not new. The explicit L∞-group regulariser plus deployed-loss model
selection may be, but it needs a real-model comparison against standard QAT to
show it is more than a reparameterisation.

---

## Tier 3 — negative results, publishable only as a set

Individually thin, collectively a decent short paper on what does not work in
low-bit quantisation:

- **Learned rotations lose to random Hadamard** on every integer condition tested,
  despite the optimiser genuinely reducing the target statistic 3×. Independently
  reproduces DartQuant's reported finding from a different objective.
- **Isotropy cannot provide a routing signal.** ‖XR‖² is invariant under
  orthogonal R, so a norm-based OOD score cannot change under any rotation —
  measured AUROC identical to three decimals. Rotation-sensitive substitutes were
  *worse* than no transform (0.187–0.229, below chance), because isotropising the
  in-distribution latent destroys the signature you would detect deviation from.
- **Multi-stage residual quantisation buys nothing at equal bits** on well-behaved
  weights (1.00×, 1.02×, 0.83×), and 5× only when outliers are strong.
- **Convex-integration coarse-to-fine correction fails** in quantisation, with an
  exact reason: fluid nonlinearity *regenerates* structure at finer scales, while
  quantisation residuals *whiten*. The cascade runs out of detail to exploit.

Negative results are hard to place. The honest venue is a workshop.

---

## Explicitly NOT novel — do not claim these

| Idea | Prior art |
|---|---|
| Sequential quantisation against propagated activations | GPTQ, AWQ |
| Disjoint high/low-precision supports | LLM.int8(), AWQ, SqueezeLLM |
| Confidence-based cloud escalation | Cactus Hybrid (shipping) |
| Phase-specific LUT kernels | Vec-LUT, T-MAC/T-MAN |
| Depth pruning + distillation recovery | Minitron, Shortened LLaMA |
| Draft-guided KV eviction | SpecKV (ICLR 2026) |
| KV cache quantisation | KIVI, TurboQuant |
| Ternary QAT | BitNet, BitCPM |

---

## Recommended order

1. **R2/G1** first — one T4 session, the method is already implemented and tested,
   and a clean result is directly publishable.
2. **R1/P3** next — the strongest question, but it needs real QAT budget. Worth
   writing the pre-registration before spending the compute.
3. R3 and R4 only if R1 or R2 lands and you want a systems companion paper.

## Standing caveat

Four times in this project I predicted something was absent and was wrong
(BitCPM-CANN existed; a Hexagon ternary path existed; the Navier–Stokes paper did
contribute; §6 was not barren). Every one of my *measurements* held up; every one
of my *negative predictions* failed. Weight this document accordingly: trust the
measured numbers, distrust the claims of novelty, and search harder than I did
before asserting R2 is new.
