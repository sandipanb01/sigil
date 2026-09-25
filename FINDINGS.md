# SIGIL — validation findings (run 2026-09-08)

Everything below came out of `run_local_validation.py` and the probe scripts on
CPU/NumPy. Synthetic activations calibrated to published KV statistics
(channel-dominant outliers over a heavy-tailed bulk); all algorithms, quantisers
and metrics are real.

**Headline: 2 of the 3 pillars I proposed do not survive contact with the data.**
I am reporting this instead of shipping a deck around it.

---

## What holds

**1. Exact weight folding — verified, machine precision.**

| identity | max abs error |
|---|---|
| attention logits, `W_q ← W_q R`, `W_k ← W_k R` | ~1e-15 |
| block output, `W_v ← W_v R`, `W_o ← Rᵀ W_o` | ~1e-15 |

Any orthogonal head-dim transform is genuinely free at inference: zero added
operators, static graph preserved, passes the Hexagon audit (`check_hexagon`:
no added ops, static shapes, int4 weights, fp16 activations, head_dim % 32 == 0).
The RoPE non-commutation guard is real and refuses unsafe folds — that failure
mode only shows up past a few thousand tokens, so catching it statically matters.

**2. Rotation helps at INT4 — but this is QuaRot's result, not mine.**
Per-token single-scale, d=64: relMSE 0.02593 (none) → 0.00483 (Hadamard), 5.4×.

**3. Weak support for the pairing law.** Gaussian-target objective wins on the
Gaussian-optimal codebook: NF4 relMSE 0.00900 (SIGIL-EP) vs 0.00975 (Hadamard).
Small effect, single condition. Interesting, not yet a result.

---

## What is falsified

**1. Learned rotation does not beat random Hadamard for integer quantisation.**

Per-token single scale, d=64, 200 optimiser steps:

| transform | INT4 | INT3 | INT2 | NF4 | NF2 |
|---|---|---|---|---|---|
| none | 0.02593 | 0.04062 | 0.12050 | 0.01719 | 0.10250 |
| **hadamard** | **0.00483** | **0.02132** | 0.11810 | 0.00975 | 0.17262 |
| SIGIL-EP | 0.00520 | 0.02318 | 0.11983 | **0.00900** | 0.20653 |
| SIGIL-CvM | 0.00513 | 0.02271 | 0.11896 | 0.00914 | 0.20073 |

The optimiser worked — Epps–Pulley fell 0.00149 → 0.00051, a 3× improvement in
measured Gaussianity. Quantisation error still got *worse*. Driving the marginal
toward the target distribution is not the same as minimising quantisation error.
This independently reproduces DartQuant's (arXiv:2511.04063) reported finding
that variance- and kurtosis-style distributional objectives barely move
quantisation loss.

**2. The isotropy router is mathematically impossible as I specified it.**

‖XR‖² = ‖X‖² for orthogonal R. The squared norm is rotation-invariant, so a
chi-square test on it *cannot* change under any rotation. Measured AUROC:

| score | raw | hadamard | SIGIL |
|---|---|---|---|
| ‖z‖² chi-square | 0.615 | 0.615 | 0.615 |

Identical to three decimals, as the algebra requires. Rotation-sensitive
substitutes are worse, and SIGIL is the worst of all:

| score | raw | hadamard | SIGIL |
|---|---|---|---|
| max\|z\| | 0.277 | 0.341 | 0.229 |
| per-token kurtosis | 0.356 | **0.650** | 0.187 |

There is a real lesson here: **isotropisation and OOD-detectability are in direct
tension.** Making the in-distribution latent maximally featureless destroys the
geometric signature you would use to detect departures from it. SIGIL scores
below 0.5 — actively anti-predictive.

**3. Therefore the "one geometric quantity pays for three things" thesis fails.**
It pays for one, and that one (folding) does not need the learned objective at
all — a random Hadamard folds just as exactly.

---

## Secondary findings worth keeping

- **Scale granularity dominates rotation choice.** With GGUF-style group=32
  per-token scales, no rotation (0.02935) is competitive with Hadamard (0.02105)
  at INT3, and *better* than every rotation at INT2 (0.06785 vs 0.117–0.224).
  Grouping already handles sparse channel outliers; rotation converts a sparse
  outlier problem into a dense one, which is a loss at very low bit-width. Any
  claim about rotations must state the scale granularity or it is meaningless.
- **The kurtosis baseline diverged** (0.42 → 0.45). Treat that row as a likely
  bug in my detached-variance gradient, not as evidence about KurTail.

---

## What I would test next, in priority order

1. **Drop the learned rotation.** Use random Hadamard, which is free, has no
   calibration step, and won every integer condition here.
2. **Chase the pairing law properly** — it is the only live novel thread. Does an
   NF/lattice codebook plus a Gaussian-target rotation beat Hadamard+INT at
   equal *effective* bits including scale overhead? One condition at NF4 is not
   enough. This needs real model weights, not synthetics.
3. **Routing needs a different signal entirely.** Token-level entropy, draft/verify
   disagreement in a speculative cascade, or a small learned probe. Not geometry
   of an isotropised space.
4. **Re-run everything on real weights** before any claim leaves this file.
   Synthetic activations got the qualitative regime wrong twice already
   (token-vs-channel outlier balance, then scale granularity).

---

## Bearing on the Qualcomm submission

The deployment engineering is sound and reusable: exact folding, the RoPE safety
guard, the Hexagon constraint audit, the AI Hub Workbench and GenieX export
paths. The *novel compression architecture* is not established. I would not put
the three-pillar claim in front of judges on this evidence.

---

# ADDITIONS — 2026-09-19

Three results since the original run, and one correction to a number this file
implicitly relied on. Full derivations in `PROVENANCE.md`.

## What holds (new)

**4. The container rule — validated 8/8 on data we did not fit.**

PrismML ship the same 1-bit Bonsai-27B weights as 3,803,452,480 B of GGUF
(1.131 bpw) and 5,129,115,752 B of MLX (1.525 bpw). Identical weights,
identical accuracy, **35% more bytes**, because the MLX container stores an
affine scale *and* zero-point per group.

Their published throughput for both ternary containers on ten machines is a
free falsification test. Our roofline predicts only a sign; it is right on
8 of 8 GPUs, separating monotonically, crossover bracketed to 1008–1792 GB/s.
Every Snapdragon part sits 4.4–19.7× below that bracket.

Writing the test caught a bug in the predictor: the bracket edges are measured
*wins*, not unknowns, so they are inclusive. With strict inequalities it
mispredicted both 1792 GB/s parts.

**5. A calibration law for PTQ refinement.**

    kept ≈ 1 − 1.75 · d/n

`d` = matrix input dimension, `n` = calibration tokens. At `n/d = 0.5`,
refinement improves calibration error 2.0× and makes held-out error **worse**
(0.77×). Controls: flat in `d` (75–78% for d = 32→256) and flat in bit width
(75–79% for 8→2 bits), so it is a law in the ratio. An independent re-run
reproduced kept = 57.6/78.4/88.9% at n/d = 4/8/16 against the model's
56.2/78.1/89.1%.

**6. Certificates detect what perplexity does not.**

7 of 7 injected faults refused at four problem sizes and seeds, correct
pipeline passes, ~3 KB per certificate. Detection generalises; **diagnosis does
not** (7/7 on the harness the signatures came from, 4/7–6/7 elsewhere), so
`diagnose()` returns a ranked hint and says so.

## What is falsified (new)

**4. Low-bit models do not resist iterative refinement.**

Predicted from the Navier–Stokes correction cycle, whose quadratic
self-interaction imposes a floor on the starting residual: bit width sets that
residual, so there should be a bit width below which refinement stops
converging.

Measured at 8/6/5/4/3/2 bits: it converges at every one, and the kept fraction
is flat in bit width. The floor is an artefact of the paper needing a *bound*
on a term that coordinate descent evaluates exactly and simply rejects when it
does not pay.

## A correction to our own numbers

**`bonsai-27b` was wrong.** *(Corrected 2026-09-22: this line said "by 51%";
3.4 against the real 3.80 GB is 12%.)* The catalogue claimed "~3.4 GB resident — a
27B-class model that actually fits a phone." The measured Q1_0 GGUF is
**3.80 GB**. The self-test guarding it asserted `footprint < 3.5 GB`, a
threshold calibrated to the invented figure, so it passed and protected the
error. Replaced with a test requiring the catalogue's arithmetic to reproduce
the file's byte count.

Separately: the claim that PrismML state the ternary build exceeds the iOS
per-app budget **could not be sourced** — their model card says nothing about
iOS — and has been withdrawn. The arithmetic is kept, labelled as ours.

## The standing pattern

Seven predictions that a source would be barren. Seven wrong. Every measurement
has held; every prediction of absence has not. The Navier–Stokes paper was read
a second time on the assumption it was already mined, and produced two of the
three results above.

---

# ADDITIONS — 2026-09-22

Two prompts: the scaling book (*How to Scale Your Model*), and a user who
watched this project start downloading gigabytes onto a laptop that only
needed a latency number. Full record, dated, in `PROVENANCE.md`.

## What holds (new)

**7. The roofline reproduces the book before it touches Snapdragon.**
`roofline.py book` recomputes 17 of the book's worked answers — critical
intensities, B_crit in both its forms, KV sizes, step times — and gets all 17.
Only then are the same formulas pointed at Snapdragon.

**8. Decode on Snapdragon X is bandwidth-bound, on Qualcomm's own data.**
`qai_hub_models` 0.62.2 ships Qualcomm's measurements in `perf.yaml`; 90
X-series LLM rows are embedded in `roofline.py` and drift-checked against the
package. The cleanest series — X Elite NPU, GenieX QAIRT, w4a16, 4K context,
Qwen3 0.6B to 8B — is a straight line in bytes moved per token: R² ≥ 0.99
under all six ways of counting the LM head and the KV cache, intercept
between −1.9 and +4.1 ms, 58–69 GB/s, 43–51% of the 135 GB/s peak.

Not every series is that clean, and the tool says which. Of 21 series with
three or more models, 8 pass the strict test (R² ≥ 0.98, intercept within
6 ms), 12 show a bandwidth slope plus a fixed per-token overhead, and 1 — X
Elite, llama.cpp on the NPU at 4K — is not linear in bytes at all.

**9. Prefill/decode is the critical batch, and it belongs to the engine.** If
prefill is compute-bound and decode bandwidth-bound, their throughput ratio
is B_crit, which cannot depend on model size. On the X2 Elite NPU (QAIRT,
512 tokens) it is 62.8–70.6 across three sizes, CV 0.05; on the CPU of the
same chip, 13.7–17.1. A published table therefore says how many draft tokens
a speculative verifier checks for the price of one step, per engine.

**10. On X2 Elite the engine barely matters for decode, and enormously for
prefill.** Qwen3-4B, each engine on its best runtime: 33.7 / 33.5 / 36.2
tok/s decode on CPU / GPU / NPU, 461 → 2307 tok/s prefill. The bus is shared;
the arithmetic is not. The software is another matter: on X Elite, QAIRT
decodes the same model 3.23× faster than Genie, and at 512 tokens llama.cpp's
NPU path trails its CPU path 2.3×. This is not a claim about every runtime.

**11. KV overtakes the weights earlier than the formula says.** Qwen3-4B on X2
Elite adds 2.61 µs per context token and crosses over at 9,770 tokens — 0.65×
the 16-bit-KV formula. Across three sizes the measured crossover sits at
0.65–1.13× the formula. That fit keeps the 4K-context point which
leave-one-out flags in the cross-model fit; without it the Qwen3-4B crossover
is about 12,700 tokens (0.85×).

**12. The container rule, corrected for context.** The 35% gap is at zero
context. KV traffic is the same in both containers, so it dilutes the gap:
30.5% at 8K, 22.3% at 32K, 10.7% at 128K.

## What is falsified or corrected (new)

**5. "Qualcomm publishes no Compute performance numbers" — our claim,
withdrawn.** It was read off model pages showing empty tables. The package
data carries 491 measured entries for X Elite CRD and 487 for X2 Elite CRD.
For X Plus 8-Core CRD it carries none, so that gap is real — and it is the
device that stands in for four of the seven HP machines.

**6. Three guesses about Qualcomm's data, overruled by the data.** "Exactly
one pair is anomalous": there are two. "The 8B is the outlier on X2 Elite":
leave-one-out removes the 4B at 4K context instead. "Measured KV crossovers
sit at 0.9–1.5× the formula": they sit at 0.65–1.13×.

**7. A hand-typed data table.** The first snapshot of Qualcomm's numbers was
typed in, and 66 of its 90 rows were off in the fourth or fifth significant
figure. The drift check against the package caught it; the table is now
machine-generated.

**8. Five figures in our own pitch, caught by an independent fact-check.** A
separate reviewer checked the submission, deck and brief against the tools:

- "3.15e-07 on real GQA weights": it came from a random-init test model in
  fp32. Real weights are what the Colab run measures.
- "the calibration law, measured on real weights": it was measured on
  synthetic matrices; the Bonsai 2 figure of 54% is the law's prediction.
- "throughput for both containers on ten machines": ten machines, both
  containers on eight.
- "wrong by 51%": 3.4 against 3.80 GB is 12%.
- "Snapdragon, 5 parts" on the chart: nine parts at six bandwidths.

It also found claims stated beyond their evidence: "the engine barely matters
for decode" holds on X2 Elite with each engine on its best runtime, not on
every runtime; a Workbench run was described as done before one had run; and
"in milliseconds, on device" was never measured on a device. All corrected,
and the five figures are now in the claim audit.

## Not yet measured

No job from this project has run on AI Hub Workbench — there is no API token
in the build environment. `aihub_workbench.py` is verified end to end against
a mock whose signatures match the installed client, and `roofline.py
predict` has written down what X Plus 8-Core should show.

## The standing pattern

Nine predictions that a source would be barren. Nine wrong — the ninth was
this project's own claim about Qualcomm's data. Every measurement has held.
