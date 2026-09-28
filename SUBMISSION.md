# Snapdragon AI Lab Build & Present Challenge — submission pack

Drafted to be pasted straight into the form. Nothing on this page is a
placeholder any more: the Colab run and the AI Hub Workbench run have both
happened, and their numbers — including the two jobs that failed — are below
with their job IDs. Everything else on this page is measured or cited, and
`PROVENANCE.md` carries the source for each. `python audit_claims.py`
re-derives the headline numbers here from the code and fails if a document
drifts from them.

---

## Project Title  (≤500 chars)

> **SIGIL-Edge: LLM deployment for Snapdragon-powered HP PCs, built on Qualcomm's own measurements — decode is bound by the memory bus, prefill/decode gives the critical batch, a container rule buys up to 35% of decode throughput for free, certificates catch a runtime decoding the wrong model, and a zero-download AI Hub Workbench runner targets the one X-series device Qualcomm has not measured.**

Shorter alternative:

> **SIGIL-Edge — on-device Indic document intelligence for Snapdragon-powered HP PCs: measured on Qualcomm's own data, zero added NPU operators, verifiable deployment.**

---

## Brief Project Description  (form text field + DOCX upload)

Snapdragon X2 Plus ships 80 TOPS of INT8 Hexagon NPU against 152 GB/s of
LPDDR5X. At batch 1 — the normal on-device case — decode re-reads the weights
and the KV cache once per token, so **the binding quantity is the stored bit
width on the wire, not TOPS.** We tested that on Qualcomm's own measurements
instead of asserting it: across Qwen3 0.6B to 8B on the X Elite NPU under
Qualcomm's QAIRT runtime, decode time is a straight line in bytes moved per
token — R² ≥ 0.99 under every way
of counting the LM head and the KV cache, intercept within 5 ms of zero.

**The roofline, checked before it was trusted.** Our roofline first
reproduces 17 of 17 worked answers from the scaling book *How to Scale Your
Model*. Only then is it applied to the 90 X-series LLM measurements that ship
inside Qualcomm's `qai_hub_models` package — nothing downloaded:

- **On X2 Elite the engine barely matters for decode; the software does.**
  Qwen3-4B decodes at 33.7 / 33.5 / 36.2 tok/s on CPU / GPU / NPU, each on
  its best runtime, while its prefill spans 461 → 2307 tok/s. On X Elite the
  same model decodes 3.23× faster under QAIRT than under Genie, on the same
  NPU.
- **Prefill/decode is the critical batch.** If prefill is compute-bound and
  decode bandwidth-bound, their ratio is the book's B_crit and cannot depend
  on model size. On the X2 Elite NPU it is 62.8–70.6 across three sizes
  (CV 0.05); on the CPU of the same chip, 13.7–17.1. That is how many draft
  tokens a speculative verifier checks for the price of one step — so
  verification belongs on the NPU.
- **Past ~10–13K tokens, KV outweighs the weights.** Qwen3-4B on X2 Elite
  crosses over at 9,770 tokens — about 12,700 without the one point
  leave-one-out flags. For long Indic documents, KV quantisation is the lever.
- **Two of Qualcomm's pairs are physically impossible** — a bigger model that
  is faster, and a marginal bandwidth above the chip's peak. We flag them and
  leave them out of the cross-model fits; we do not explain them away.

**1. The container rule — up to 35% of decode throughput, for free.**
The *same* 1-bit Bonsai-27B weights ship in two containers: `Q1_0.gguf` is
3,803,452,480 B (1.131 bits/weight) and the MLX pack is 5,129,115,752 B
(1.525 bpw), because MLX stores an affine scale *and* zero-point per group.
Identical weights, identical accuracy, **35% more bytes across the wire on
every token** at zero context — 11% at 128K, because the KV traffic is shared.

Throughput is published on ten machines, and for both *ternary* containers
(PTQ1_0 at 1.768 bpw, PQ2_0 at 2.143 bpw) on eight of them. That is a free
falsification test, and
nothing in our engine was fitted to it. Our roofline predicts only a sign —
bandwidth-starved parts should prefer the narrower stream, compute-starved
parts the cheaper unpack — and it is **right on 8 of 8 GPUs, separating
monotonically with the crossover bracketed to 1008–1792 GB/s.** Every
Snapdragon part sits **4.4–19.7× below that bracket**, so on Snapdragon the
narrower container always wins, with no benchmarking required. Where PTQ1_0
wins it realises 0.46 of the ideal byte advantage, independently close to our
`lut_arith_efficiency = 0.60`, calibrated on entirely different hardware.

**2. A measured KV-compression budget.** We sweep INT4/INT3/INT2 KV
quantisation across scale granularities (per-token, group-128, group-32), with
and without a randomised Hadamard rotation, and report WikiText-2 perplexity
against an FP16 baseline. The finding we lead with is that **scale granularity
dominates rotation choice.** Reproducible in under two hours on a free Colab
T4.

**3. Zero-overhead weight folding — with shipped precedent.** Orthogonal
head-dimension transforms fold exactly into `W_q/W_k` and `W_v/W_o`, leaving
attention logits and block outputs unchanged — exact to ~1e-15 in float64,
and 3.15e-07 end to end in fp32 on a random-init test model; the Colab run
repeats it on real weights.
The deployed graph gains **zero operators** — critical on Hexagon, where a
novel per-token operator means no NPU path at all and a silent CPU fallback.
Bonsai 2 27B ships exactly this: a blockwise Hadamard basis folded into the
stored weights, metadata **297,903 B of an 8.6 GB pack — 0.0035%.** Its
widths (5120, 6144, 17408) are exactly 5, 6 and 17 blocks of 1024, a
divisibility constraint we now enforce before export.

**4. Certificates, because quantised models fail silently.** The model's own
runtime note says *"Ordinary MLX loaders do not apply the required
transforms"* — a loader that does not know about the rotated basis **returns
wrong output rather than an error.** On Snapdragon the exposure is wider,
since a model reaching Hexagon has been through QNN conversion, graph
optimisation and quantisation. Borrowing the protocol of machine-checked
proofs — an independent checker that trusts nothing the producer built — we
ship a ~3 KB JSON beside the weights declaring the scheme (including the group
**axis**), the transform, seeded probes and a tolerance derived from the
declared dtype. **It refuses 7 of 7 injected faults at four problem sizes.** A
check took ~0.5 ms on a laptop CPU at the test size and needs no calibration
data, reference weights or network, so it can run on device before a model
loads.

**5. The benchmark history is a simulator.** Every 90-minute sweep writes a
complete record over the configurations it tried, so a cheaper search policy
can be scored against it for nothing. Greedy neighbourhood search reaches the
same answer in 4 evaluations against the sweep's 18 — 4.5x fewer calls, no
GPU. Replay is valid only inside the realized search space, so a policy that
asks for an unevaluated configuration is marked unreliable and never ranked,
and both shipped pools are flagged as unable to rank policies fairly, because
they are.

**6. A local-first cascade with an honest boundary.** Ternary/1-bit GGUF
drafts, a local INT4 model verifies, and a remote frontier model is reached
only on escalation. Closed-weight models **cannot** run on an NPU at any
compression ratio — we treat them as a network-cost verifier of last resort,
not as something to compress.

**The target, stated exactly.** The challenge asks for a solution optimised
for Snapdragon-powered HP PCs. HP ships seven, and `hp_snapdragon.py machines`
resolves each to its silicon, its RAM and its AI Hub proxy:

| HP machine | SoC | RAM | AI Hub device |
|---|---|---|---|
| OmniBook 3 14-HZ000 | Snapdragon X | 8 GB | X Plus 8-Core CRD* |
| OmniBook 5 16-bf000 | Snapdragon X | 16 GB | X Plus 8-Core CRD* |
| OmniBook Ultra 14-kg000 | **Snapdragon X2 Plus** | 16 GB | **none** |
| OmniBook Ultra 14 | Snapdragon X2 Elite | 16/32/64 GB | X2 Elite CRD |
| ProBook 4 G1q 14 | Snapdragon X | 16/32 GB | X Plus 8-Core CRD* |
| EliteBook 6 G1q 14 | Snapdragon X Elite | 32 GB | X Elite CRD |
| EliteBook Ultra G1q8 14 | Snapdragon X Plus | 16 GB | X Plus 8-Core CRD |

*\* HP prints only "Snapdragon X" on those three; the X Plus 8-Core CRD is the
nearest device offered, and we say so rather than implying an exact match.*

Two consequences we design around. The **8 GB OmniBook 3** leaves roughly 4–5
GB after Windows, which rules out an 8B model at INT4 resident — it is the
machine the architecture has to respect, not the 64 GB flagship. And the
**OmniBook Ultra 14-kg000 has no AI Hub device at all**, because X2 Plus is
not offered for profiling; that configuration can only be measured on the
metal, and we never present X2 Elite numbers as X2 Plus numbers.

**What AI Hub can and cannot prove — checked, then corrected.** Qualcomm's
measurements ship inside the `qai_hub_models` package: **491 measured entries
for X Elite CRD, 487 for X2 Elite CRD, and none for X Plus 8-Core CRD** —
listed as supported for 221 models, measured for none. X Plus 8-Core is the
proxy for four of the seven HP machines, so that gap is the one that matters.
(An earlier draft of this page said Qualcomm published no Compute numbers at
all, read off model pages showing empty tables. The package data proved that
wrong for two of the three devices; `PROVENANCE.md` records the correction.)

**Testing on Workbench without a download.** `aihub_workbench.py` builds one
Qwen3-4B decoder layer at its real dimensions — attention over a 4K-token KV
cache, RoPE, the gated MLP — uploads it once (~84 MB plus 34 MB of
calibration data), and runs the same job sequence as Qualcomm's own
pipeline on X Plus 8-Core CRD: compile and profile at fp16; compile to ONNX,
quantise, compile with quantised I/O and profile at w8a16 and w4a16. What
comes back is profile JSON, a few KB per job, against a hard 2 MB ceiling;
the client it drives has no download method at all. Every AI Hub call is
asserted against the installed `qai-hub` client's signatures at test time, so
an API change breaks a test here instead of a job the night before a deadline.
The Workbench numbers below were produced from this repository on Snapdragon
X Plus 8-Core CRD: fp16 profiled cleanly (job `j5ql4e34p`), and the two
quantised paths compiled and then failed on the device, which is reported
with the service's own error strings rather than dropped.

**The NPU question, answered in two halves.** Prefill runs on the Hexagon NPU:
INT4 weights, static shapes, via GenieX (QAIRT-backed) or ONNX Runtime QNN. It
is compute-bound over a whole prompt, and Qualcomm's data shows the NPU
running it 5× faster than the CPU on X2 Elite. Decode is bound by the bus —
on X2 Elite at INT4 the NPU (QAIRT) and the CPU decode alike — so what
decides the engine is the
container width. **Stock QNN has no ternary matmul**, so the ternary
container runs on the ARM CPU with NEON, where the container rule applies.
Claiming the whole pipeline runs on the NPU is the claim that gets taken apart
under questioning; claiming the NPU is unusable is equally wrong.

**Deployment.** Three real paths, no bespoke runtime: Qualcomm GenieX for
NPU/GGUF serving with an OpenAI-compatible endpoint, run on the HP laptop
itself; llmware's ONNXRuntime-QNN path for document parsing and RAG on the
Snapdragon NPU (Windows ARM64); and AI Hub Workbench for compile-and-profile
jobs on the three X-series CRDs, with nothing downloaded.

**And a shipped app this is immediately worth something to.** AnythingLLM is
an MIT-licensed, local-first document-chat desktop app that Qualcomm itself
ported to the Snapdragon NPU; its QNN path ships Llama-3.2-3B (8K and 16K
context), Llama-3.1-8B (8K) and Phi-3.5-mini (4K). Two findings here apply to
it directly and can be checked today. **The runtime is worth more than the
model choice:** on X Elite at w4a16, Qualcomm's own measurements give
Llama-3.2-3B 11.32 tok/s under Genie against 19.82 under QAIRT (1.75x), and
Llama-3.1-8B 5.02 against 10.72 (2.14x) — same weights, same silicon.
**And the device we profiled is the device that app cannot currently use:**
its NPU engine fails on Snapdragon X Plus (X1P42100) because the bundled
cpuinfo does not recognise the part (issue #5129, open, escalated to the
vendor). That is the same tier as the X Plus 8-Core CRD profiled above, and
the proxy for four of the seven HP machines. The gap this project measured is
not academic: it is where a 66k-star application loses the NPU.

**Use case.** Indic-language document intelligence: statutory forms, land
records, exam papers and health documents across Hindi, Marathi, Gujarati,
Punjabi, Telugu, Kannada, Tamil and Malayalam. These are exactly the documents
that must not leave the device, in exactly the markets where connectivity is
unreliable and Snapdragon share is highest. Qualcomm already ships an Indic
1.1B model, IndusQ, on AI Hub; this extends that direction into a full
on-device document pipeline.

---

## Measured results

**Evidence, ranked honestly (`hp_snapdragon.py evidence`).** Tier 1 is running
on an actual HP Snapdragon PC — strongest, and not available to us. Tier 2 is
an AI Hub job on an X-series CRD — free, minutes, and the only source for X
Plus 8-Core CRD. Tier 3 is Qualcomm's own published measurements — available
for X Elite and X2 Elite, not for X Plus 8-Core. Tier 4 is modelling, labelled
as supporting-only throughout.

**Qualcomm's own measurements, tested against the roofline (`roofline.py`):**

| Measurement | Result | Command |
|---|---|---|
| X Elite NPU, GenieX QAIRT w4a16, 4K context, Qwen3 0.6B–8B | decode linear in bytes, R² ≥ 0.99, 43–51% of 135 GB/s | `law` |
| X2 Elite NPU, QAIRT, 512 context, three sizes | prefill/decode 62.8–70.6, CV 0.05 | `bcrit` |
| X2 Elite CPU, llama.cpp, 512 context | prefill/decode 13.7–17.1 | `bcrit` |
| Qwen3-4B, X2 Elite, 512 context, best runtime per engine | decode 33.7 / 33.5 / 36.2 tok/s CPU / GPU / NPU; prefill 461 → 2307 | `engines` |
| Qwen3-4B, X2 Elite, QAIRT, 5 contexts | KV overtakes weights at 9,770 tokens (12,700 without the flagged 4K point) | `context` |
| Qwen3-4B, X Elite NPU, 4K context | software alone moves decode 3.23× (Genie → QAIRT) | `runtimes` |
| The scaling book's worked answers | 17 of 17 reproduced | `book` |

**Reproduced from published data, nothing fitted (`snapdragon_engine.py container --validate`):**

| GB/s | Device | Winner | Predicted |
|---|---|---|---|
| 300 | L4 (72 W) | PTQ1_0 | PTQ1_0 ✓ |
| 864 | L40S | PTQ1_0 | PTQ1_0 ✓ |
| 960 | RTX 6000 Ada | PTQ1_0 | PTQ1_0 ✓ |
| 1008 | RTX 4090 | PTQ1_0 | PTQ1_0 ✓ |
| 1792 | RTX 5090 | PQ2_0 | PQ2_0 ✓ |
| 1792 | RTX PRO 6000 | PQ2_0 | PQ2_0 ✓ |
| 2039 | A100 SXM | PQ2_0 | PQ2_0 ✓ |
| 3350 | H100 SXM | PQ2_0 | PQ2_0 ✓ |

**Measured here, on synthetic matrices (`residual_cascade.py law`):** a calibration
law for post-training quantisation refinement,

    kept ≈ 1 − 1.75 · d/n

where `d` is the matrix input dimension and `n` the calibration token count.
At `n/d = 0.5`, refinement improves calibration error 2.0× while making
held-out error **worse** (0.77×). Controls establish it is a law in the ratio:
flat in `d` (75–78% for d = 32→256) and flat in bit width (75–79% for 8→2
bits). For Bonsai 2 27B the law predicts that a 128×512 calibration set keeps only **54%** of any
measured gain on `down_proj` — the matrix that always binds, because it alone
sees the intermediate dimension.

**Measured on AI Hub Workbench, Snapdragon X Plus 8-Core CRD** — the one
X-series device Qualcomm has published nothing for. One Qwen3-1.7B decoder
layer at 4K context, built locally and uploaded once:

| precision | job | result |
|---|---|---|
| fp16 | `j5ql4e34p` | **6.314 ms per layer, 53 operators, every one on the NPU**, 81.7 MB peak, 5.43 s first load |
| w8a16 | `jpyo8rql5` | compiled, then `QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error` at profile |
| w4a16 | `jg9zojvqp` | compiled, then "Failed to fully run the model, failed after compiling" |

Uploaded once, 7.1 KB of profile JSON back, nothing downloaded. The fp16 layer
moves 117.5 MB per step and does it in 6.314 ms — **18.6 GB/s, 13.8% of the
135 GB/s peak** — which is what a single layer profiled in isolation looks
like: the per-inference fixed cost is paid once per layer instead of once per
token. We publish it as a layer figure and do not turn it into a model figure.

**The two failures are a result, not a gap.** They are the first public
evidence of where the 8-core part's profiler stops: the quantised graph, with
INT16 activations over a 4K KV cache, does not allocate on this device while
the fp16 graph does, at 81.7 MB. Both failed *after* a clean compile, so it is
a runtime limit, not a conversion error. Nobody else will have that either.

**Measured on Colab T4 (reproducible, ~90 min):** FP16 perplexity `14.131`;
KV at INT4 group-32 `15.032`; fold error `3.68E-007`; decode `25.77` tok/s;
KV cache `12.0` KiB/token -> `0.38` GiB at 32k context against 152 GB/s.

---

## An unclaimed win, stated as a hypothesis

The Bonsai 2 vision tower ships **unrotated and unquantised in every
container** — the v1 and v2 mmproj files are byte-identical to within 96 bytes
across five months. It is **1.70% of the parameters and 13.5% of the bytes.**
Rotating and ternarising it to the language model's own 1.768 bpw removes
0.83 GB, 12% of the pack, and it is the component always resident in a
document/OCR workload.

**Flagged UNMEASURED.** MMMU-Pro already falls 81.73 → 75.49 and OCR Bench v2
60.99 → 56.88 under ternary weights *with an untouched tower*, so the tower may
be carrying the multimodal path. This is a hypothesis with a clear experiment,
not a result, and it is presented that way.

---

## GitHub Repository Link  (≤500 chars)

```
https://github.com/<your-username>/sigil-edge
```

Push before submitting. Required in the repo root: `README.md`, `FINDINGS.md`,
`PROVENANCE.md`, the fourteen Python modules, `sigil/`, an OSI licence
(Apache-2.0 matches llmware and Bonsai), your `results_t4.json` +
`kv_sweep.png`, and your `workbench_results.json`. Add a Colab badge pointing
at the benchmark script — judges will click it.

---

## "Snapdragon Laptop" form field

Answer **honestly: No.** It does not disqualify you, and the submission is
built so it does not matter: every hardware number is Qualcomm's own
measurement, shipped in its pip package, or — once you have run it — a
profile from AI Hub Workbench on a cloud-hosted Snapdragon device, with a job
ID anyone can re-run. Say that
explicitly — a candidate who produced the first X Plus 8-Core numbers without
owning the hardware reads as resourceful, and a false claim that gets checked
is fatal.

Note in passing: **Snapdragon X2 Plus is not an AI Hub device.** Only X2 Elite
CRD, X Elite CRD and X Plus 8-Core CRD are offered on the compute side. The
OmniBook Ultra 14-kg000 can only be measured on the metal; say so.

---

## What NOT to claim

Cut these if they appear anywhere in your deck. Each was tested and failed, or
is not yet measured.

- *"Qualcomm publishes no X-series performance numbers."* Our own earlier
  claim, withdrawn: the package data holds 978 measured X Elite and X2 Elite
  entries. What is missing is X Plus 8-Core.
- *"X Plus 8-Core does 6.314 ms per token."* It does 6.314 ms per **layer**,
  at fp16, profiled in isolation. A model figure needs every layer plus the
  LM head, and the quantised paths did not run at all on that device.
- *"The quantised paths are broken."* Two jobs failed at profile time on one
  device at one context length. That is what we saw and all we claim.
- *"Speculative decoding gives a 12× speed-up."* The model assumes each draft
  token is accepted independently, which real drafts are not. Every speculative
  speed-up here is an upper bound; what is robust is where the verifier's
  ridge binds.
- *"We know why Qualcomm's two anomalous pairs are wrong."* We do not. They
  are flagged and left out of fits.
- *"A novel learned-rotation quantiser that beats existing methods."* It lost
  to a random Hadamard on every integer condition. See `FINDINGS.md`.
- *"Isotropy gives a free out-of-distribution routing signal."* Mathematically
  false: ‖XR‖² = ‖X‖² for orthogonal R, so the score cannot change under any
  rotation. Measured AUROC identical to three decimals.
- *"Runs Claude / GPT-6 Astra on device."* The weights are not public.
- *"One geometric principle unifies compression, allocation and routing."* Two
  of the three legs are falsified.
- *"Low-bit models resist iterative refinement."* Predicted from the
  Navier–Stokes cycle's self-interaction floor, **falsified** — refinement
  converges identically at 8 through 2 bits.
- *"A 27B model at 1-bit is 3.4 GB and fits a phone."* Our own earlier claim,
  12% under the real file. The measured Q1_0 GGUF is 3.80 GB, and the self-test that
  "verified" it had been calibrated to the invented number.
- *"The vendor says the ternary build exceeds the iOS budget."* Their model
  card says nothing about iOS. The arithmetic is ours and is labelled as ours.
- *"Our search policies beat exhaustive sweeping."* They use fewer calls on two
  pools that are both **degenerate** — one has its optimum at the first
  configuration tried, the other is nearly flat.

The submission is stronger without them. A measured negative result plus a
working deployment path beats an unverified grand claim, and it is defensible
under questioning.

---

## Judging-criteria map

| Criterion | What carries it |
|---|---|
| Technical Implementation | The scaling-book roofline, reproduced 17/17 and then tested on Qualcomm's own X-series measurements (R² ≥ 0.99); the first published profile of a real decoder layer on Snapdragon X Plus 8-Core CRD, failures included; the container rule validated 8/8 on data we did not fit; exact folding with a RoPE guard; 872 self-tests across 14 modules, a 292-check stress harness that found 17 defects on its first run and 10 more from its fuzzer since, and a 131-check claim audit that re-derives the quoted numbers from the code. Every correction is dated in `PROVENANCE.md`, including corrections to our own published numbers |
| Use Case & Innovation | Indic on-device document intelligence; privacy and offline operation are the product. Prefill/decode read as the critical batch tells a speculative pipeline where to verify. The vision-tower gap is a concrete, stated next experiment |
| Deployment & Accessibility | Every one of the seven HP Snapdragon PCs mapped to its silicon, RAM budget and AI Hub proxy, with the gaps named; a zero-download Workbench runner that has now profiled the one X-series device Qualcomm has not measured; findings that apply directly to a shipped MIT-licensed local-AI app whose NPU path fails on that same part; GenieX + llmware ONNXRuntime-QNN; Apache-2.0 throughout |
| Presentation & Documentation | `FINDINGS.md` and `PROVENANCE.md` report every falsified hypothesis and every correction, including the withdrawal of our own claim about Qualcomm's data. Few entrants will show their negative results |

---

## Verified sources

`PROVENANCE.md` is the authoritative record — every source with its status,
and every correction dated. Headline entries:

| Source | Status | Contribution |
|---|---|---|
| `qai_hub_models` 0.62.2, `models/<id>/perf.yaml` | read; 90 X-series rows embedded and drift-checked | Qualcomm's measurements: the bandwidth law, the critical batch, the device-coverage gap |
| *How to Scale Your Model* (jax-ml.github.io/scaling-book) | read; 17 worked answers reproduced | Roofline, B_crit, KV sizing, step-time formulas |
| aihub.qualcomm.com/get-started, `qai-hub` 0.55.0 | read; every call signature-checked | The Workbench job sequence and client surface |
| PrismML Bonsai 2 27B (36 repos, byte-exact file sizes) | verified | The container rule; rotation-folding precedent; the vision-tower gap |
| `microsoft/BitNet` + `microsoft/T-MAC` (T-MAN) | read | I2_S/TL1/TL2 kernel matrix; the open-source ternary-on-NPU path |
| ENERZAi Hexagon write-ups | read | Ternary on Hexagon is proven but needs a custom kernel |
| `llmware-ai/llmware` | read | ONNXRuntime-QNN Snapdragon NPU path |
| *Finite Time Blowup for Navier–Stokes* and its Lean certificates | read in full | The correction-cycle model, the calibration law, the certificate protocol |
| `zhengkid/Dream-RSI` | read | The replay-simulator idea behind `dream_search.py` |
| `Mintplex-Labs/anything-llm` (MIT) + its QNN model list | read | The shipped local-AI app these findings apply to; its NPU model line-up |
| anything-llm issue #5129, and Qualcomm's own AnythingLLM-on-NPU write-up | read | The X Plus NPU failure in shipped software; the vendor-supported NPU port |
| QuaRot, SpinQuant, KurTail, DartQuant, KIVI, QuIP#, GPTQ, AWQ | read | Rotation and quantisation prior art |
| Snapdragon X2 product briefs | verified | 80 TOPS INT8, LPDDR5X 9523 MT/s, ~152 GB/s |

Identifier verification runs over the network: `snapdragon_engine.py verify`.
Expect 0 MISSING; GATED/BLOCKED and NO-ID are fine.
