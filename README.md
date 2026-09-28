# SIGIL-Edge

LLM deployment for the seven Snapdragon-powered HP PCs, built on one measured
fact: **on Snapdragon X, decode is bound by the memory bus -- its time is a
straight line in the bytes moved per token.** Qualcomm's own measurements
show it, the scaling-book roofline predicts it, and every design decision
here follows from it.

872 self-tests across 14 modules, a 292-check stress harness, and a 131-check
claim audit that re-derives the quoted numbers from the code. All passing. The
engine, the roofline and the HP mapping need only the standard library; the
science modules need NumPy; only the Colab benchmark needs torch. Apache-2.0.

**Nothing in this project downloads a model onto your laptop.** Qualcomm's
measurements ship inside the `qai_hub_models` pip package and are embedded in
`roofline.py`. New measurements come from AI Hub Workbench through
`aihub_workbench.py`, which uploads one decoder layer and brings back a few KB
of profile JSON. The stress harness reads every file, every command the
project emits and every document, and fails if a download comes back.

```powershell
python roofline.py law                              # Qualcomm's data vs the roofline
python roofline.py bcrit                            # the critical batch, measured
python roofline.py book                             # the scaling book's answers, 17/17
python aihub_workbench.py plan                      # the Workbench job, dry
python hp_snapdragon.py machines                    # the seven HP machines
python snapdragon_engine.py container --validate    # the container rule, 8/8
python stress_all.py                                # tries to break everything
python audit_claims.py                              # every number vs the code
```

---

## What is here

| File | Tests | What it does |
|---|---|---|
| `snapdragon_engine.py` | 141 | Hardware probe, 61-model catalogue, capability verdicts, deployment planner, zero-download AI Hub commands, identifier verification |
| `sigil_t4_benchmark.py` | 123 | Real-weights KV-quantisation benchmark for Colab. Refuses a >1 GB model download on a machine that is not a hosted notebook unless `--allow-download` |
| `dream_search.py` | 87 | Replay-simulator search over recorded benchmark history, after Dream-RSI. 4.5x fewer calls to the same answer, on pools it flags as too degenerate to rank policies |
| `roofline.py` | 83 | The scaling-book roofline, checked against the book's own worked answers, then tested on Qualcomm's measured X Elite and X2 Elite data |
| `aihub_workbench.py` | 64 | Zero-download testing on AI Hub Workbench: a decoder layer at real dimensions, built locally, compiled, quantised and profiled in the cloud |
| `psdc.py` | 60 | Phase-Split Depth Cascade: roofline, SoC budgets, **kill conditions**, MLA latent-rank analysis |
| `quant_certificate.py` | 58 | Certificates that prove a runtime decodes your model the way you quantised it; catches 7/7 injected faults |
| `hp_snapdragon.py` | 56 | The seven HP Snapdragon PCs mapped to silicon, RAM and AI Hub device; what AI Hub can and cannot prove |
| `residual_cascade.py` | 49 | When another round of PTQ refinement is worth running; the law `kept = 1 - 1.75*d/n` |
| `generator_quant.py` | 47 | **Quantise the chart, not the manifold** -- SO(d), SPD, Stiefel, Birkhoff |
| `qat_optimizer.py` | 43 | Quantisation damage model, group-L-inf regulariser, deployed-loss selection |
| `on_device_adaptation.py` | 23 | Can Snapdragon train locally? Activation memory, not FLOPs, is the blocker |
| `slt_compressibility.py` | 22 | Singular learning theory; documents an estimator failure |
| `aihub_bench.py` | 16 | Qualcomm's published numbers per model -- a few KB of metadata, no weights |
| `stress_all.py` | 292 | One command that tries to break everything: fuzzing, numeric sanity, determinism, portability, CLI smoke, and **network** |
| `audit_claims.py` | 131 | Re-derives the quoted numbers from the code, and checks the documents quote the counts the code reports |
| `sigil/` | -- | Rotation/quantiser primitives, exact folding, Hexagon audit (`run_local_validation.py`, `p1_test.py`) |

Documents: `SUBMISSION.md` (form text), `PROVENANCE.md` (every source and
dated correction), `FINDINGS.md` (what holds, what failed), `QUICKSTART.md`,
`RUNBOOK.md`, `RESEARCH_AGENDA.md`, `MANIFEST.md`.

---

## The result: the bus bounds decode, the engine sets prefill

`qai_hub_models` ships Qualcomm's own measurements inside the pip package.
`roofline.py` embeds the 90 X-series LLM rows (and checks them against the
package when it is installed) and fits each series against bytes moved per
token:

- **Decode is a straight line in bytes.** Qwen3 0.6B to 8B on X Elite's NPU
  (GenieX QAIRT, w4a16, 4K context): R^2 >= 0.99 under all six ways of counting
  the LM head and the KV cache, intercept within 5 ms of zero, 43-51% of the
  135 GB/s peak.
- **On X2 Elite the engine barely matters for decode; the software does.**
  Qwen3-4B decodes at 33.7 / 33.5 / 36.2 tok/s on CPU / GPU / NPU, each on its
  best runtime, while its prefill spans 461 -> 2307 tok/s. The bus is shared;
  the arithmetic is not. On X Elite the same model decodes 3.23x faster under
  QAIRT than under Genie, on the same NPU.
- **Prefill/decode is the critical batch.** If prefill is compute-bound and
  decode bandwidth-bound, their throughput ratio is the scaling book's B_crit,
  and it cannot depend on model size. On the X2 Elite NPU it is 62.8-70.6
  across three sizes (CV 0.05); on the CPU of the same chip, 13.7-17.1. That is
  how many draft tokens a speculative verifier checks for the price of one
  step, read straight off a published table.
- **Past ~10-13K tokens, KV outweighs the weights.** Qwen3-4B on X2 Elite
  adds 2.61 us per context token and crosses over at 9,770 tokens -- 0.65x
  the book's formula with a 16-bit cache -- or about 12,700 without the 4K
  point that leave-one-out flags. For long Indic documents, KV quantisation
  is the lever.

Two of Qualcomm's pairs are physically impossible (`roofline.py anomalies`):
a bigger model that is faster, and a marginal bandwidth above the chip's peak.
They are flagged and left out of the cross-model fits by leave-one-out, not
explained away.

## The scaling book, used as a tool

`roofline.py book` reproduces 17 worked answers from *How to Scale Your
Model* -- critical intensities, B_crit in its two forms, KV sizes, step times
-- before any of its formulas touch Snapdragon. Only then does it apply them:
the bandwidth law above, the critical batch, the KV crossover, where a
speculative verifier stops being free, and what an unmeasured device should
do (`roofline.py predict`).

## Testing on AI Hub Workbench without a download

Qualcomm's package carries 491 measured entries for X Elite CRD, 487 for X2
Elite CRD and **none for X Plus 8-Core CRD** -- the AI Hub device that stands
in for four of the seven HP machines. `aihub_workbench.py` fills that gap,
and has:

1. builds one Qwen3-4B decoder layer at its real dimensions -- attention with
   a 4K-token KV cache, RoPE, the gated MLP -- with grid-valued weights that
   zip 4.8x (83.8 MB);
2. uploads it once, with 33.6 MB of calibration data;
3. follows the job sequence of Qualcomm's own `qai_hub_models` pipeline:
   compile to QNN DLC and profile at fp16; compile to ONNX, quantise
   (min-max), compile with quantised I/O and profile at w8a16 and w4a16;
4. brings back only `download_profile()` JSON, against a hard 2 MB ceiling.

The client it drives exposes no download method at all. It is verified end to
end against a mock whose signatures match the installed `qai-hub` client
parameter by parameter, and the archive it uploads is accepted by the
client's own model-type classifier.

**It has run.** A Qwen3-1.7B decoder layer at 4K context, profiled on
Snapdragon X Plus 8-Core CRD: fp16 in **6.314 ms per layer, 53 operators,
every one of them on the NPU**, 81.7 MB peak (job `j5ql4e34p`). The two
quantised paths compiled and then failed on the device --
`QNN_COMMON_ERROR_MEM_ALLOC` at w8a16, "failed after compiling" at w4a16 --
which is the first public evidence of where that part's profiler stops. The
whole run uploaded once and brought back 7.1 KB of JSON.

---

## The one idea that is probably novel

**Quantise the chart, not the manifold** (`generator_quant.py`).

Constrained parameters break when quantised directly. Store the *unconstrained
generator* and reconstruct through the chart; the constraint then holds
structurally at any bit width.

| manifold | chart | direct @ 4 bits | chart @ 2 bits |
|---|---|---|---|
| SO(d) | Cayley / expm of skew | 4.75e-2 | **1.0e-15** |
| Birkhoff(d) | Sinkhorn-Knopp | dev 5.88e-2 | **dev 1.2e-2** |
| SPD(d) | expm of symmetric | **min eig -1.12** | **min eig +1.4e-2** |
| Stiefel(n,p) | expm of skew, first p cols | 5.45e-2 | **1.7e-15** |

The SPD row is the sharpest case: a "covariance" with a negative eigenvalue is
not a covariance, and Cholesky, Mahalanobis and natural gradient fail outright
rather than degrading. Direct 4-bit produced **-1.12**. That is a type error,
not a small one.

The non-obvious part: chart quantisation changes *which* point you land on --
71% Frobenius drift at 2 bits -- and it does not matter. 93% of the
outlier-spreading benefit survives, because incoherence is a **generic**
property of rotations, not a property of one specific R. Drifting inside the
group is harmless; leaving it is fatal. Payoff: **12.9x smaller rotation
storage** with exactness preserved. Needs validation on real transformer
weights: `falsification()` G1, one T4 session.

---

## Status of the proposed architecture

PSDC -- prefill full-depth on the NPU, decode on a depth-pruned ternary subset
sharing the prefill KV cache -- is a **design proposal**, not a result.

- **P1 (fatal)** -- NARROWED, not closed. Sharing full-network KV costs nothing
  and is often *better* (ratios 0.83 / 0.88 / 0.67 / 1.01), but measured on an
  untrained model, so it tests residual-stream compatibility, not accuracy.
- **P2** -- calibrates the LUT factor with AI Hub jobs.
- **P3** -- does ternary retention track width or parameter count? Three
  cheap-proxy attempts saturated; it needs QAT runs.
- **P4** -- does phase-split beat uniform compression at equal bytes?

Run `psdc.py falsify` before building on any of it.

---

## Claims deliberately NOT made

- No X Plus 8-Core numbers until a Workbench job produces them. `roofline.py
  predict` is a prediction and says so.
- No per-token figure from one layer is a model figure; `aihub_workbench.py
  results` labels every extrapolation.
- Speculative speed-ups assume independent acceptance, which real drafts do
  not have: they are upper bounds.
- No on-NPU training. QAIRT/QNN/Genie are inference-only.
- No closed-weight model running on device, at any compression ratio.
- No claim that a learned rotation beats a random Hadamard -- tested, it lost.
- Sequential quantisation, disjoint supports, confidence routing,
  phase-specific LUT kernels, depth pruning, KV quantisation and ternary QAT
  are **all prior art**. `RESEARCH_AGENDA.md` has the attribution table.

---

## Before you publish

1. `python snapdragon_engine.py verify` from an unrestricted network. 32 of the
   61 catalogue identifiers are registry-confirmed; the rest are marked `?`.
2. Run the Colab benchmark and replace every `[FILL]` in `SUBMISSION.md`.
3. Run `python aihub_workbench.py run --yes` with your API token, then put the
   job IDs in `SUBMISSION.md`.
4. Check licences. Shortened LLaMA checkpoints are **non-commercial**.

---

## How to read this work

Nine times in this project a source was predicted to be barren, and nine times
the prediction was wrong -- the latest being this project's own claim that
Qualcomm publishes no X-series numbers. The package data held 978.

**Every measurement here has held up. Every confident absence has not.**
Trust the numbers, push on the negatives, and search harder than this project
did before believing any novelty claim -- including the one above.

Licence: Apache-2.0.
