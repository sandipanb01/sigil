# PROVENANCE

Every source, with an honest status. `read` means I fetched and used it. `not read`
means I did not open it — listing it here rather than implying coverage I don't have.

## Read in full, and used

| Source | What it contributed |
|---|---|
| `microsoft/BitNet` (bitnet.cpp, MIT, 39.3k★) | The I2_S / TL1 / TL2 kernel matrix, the per-model support table, and the measured ARM figures (1.37–5.07× speedup, 55.4–70.0% energy reduction). Drove `select_ternary_kernel()` and four catalogue entries. Also: **NPU support is stated as "coming next" — there is no Hexagon path for 1.58-bit today.** |
| `qualcomm/ai-hub-models` (BSD-3) | Full 300+ model catalogue, runtime matrix (QNN / LiteRT / ONNX), precision matrix per compute unit, chipset and device lists, the CLI surface, and the Windows-ARM64 Python gotcha. |
| `qualcomm/ai-hub-apps` | Genie SDK ChatApp; NPU dtype requirements. |
| aihub.qualcomm.com | GenieX, Workbench (50+ cloud devices), Mistral as model-maker partner, IndusQ 1.1B. |
| `llmware-ai/llmware` (Apache-2.0) | ONNXRuntime-QNN Snapdragon NPU path, 7 NPU-optimised models, document parsing + RAG, SLIM agents. |
| `PrismML-Eng` org (9 repos) | Bonsai-demo (883★), Bonsai-Image-Demo, image-studio, llama.cpp fork (Q1_0/Q2_0 ternary, NEON kernels), mlx / mlx-swift / mlx-c forks, mflux-prism, archived sglang fork. |
| `galilai-group/lejepa`, arXiv:2511.08544 | SIGReg / Epps–Pulley. Source of the hypothesis I tested **and rejected** — see `FINDINGS.md`. |
| arXiv:2608.27395 (LeVJEPA), `lucas-maes/le-wm` | SIGReg at video scale; LeWorldModel objective. |
| QuaRot, SpinQuant, KurTail, DartQuant, ButterflyQuant, TurboQuant, KIVI, QuIP# | Rotation/quantisation prior art. DartQuant's negative result on distributional objectives was independently reproduced. |
| Snapdragon X2 product briefs | 80 TOPS INT8, LPDDR5X 9523 MT/s, ~152 GB/s. Seeded `SOC_DB`. |

## STATUS TABLE — authoritative, supersedes everything below

**This file accreted chronologically and its earlier "Not read" list went stale.**
Sections below are a dated work log, kept because the corrections in them matter.
Where they conflict with this table, **this table wins.**

| Source | Status | Outcome |
|---|---|---|
| `microsoft/BitNet` | read | I2_S/TL1/TL2 kernel matrix; ARM 1.37–5.07×, 55–70% energy |
| `microsoft/T-MAC` + **T-MAN** | read | Validated the roofline; T-MAN overturned the CPU-first recommendation |
| `OpenBitSys/vlut.cpp` (+ paper) | read | Vector LUT; **lists speculative decoding as a parallel scenario** |
| `qualcomm/ai-hub-models`, `ai-hub-apps`, aihub.qualcomm.com | read | Catalogue, runtimes, precision matrix, Workbench |
| `llmware-ai/llmware` | read | ONNXRuntime-QNN Snapdragon NPU path |
| `PrismML-Eng` org | read | Bonsai Q1_0/Q2_0, NEON kernels |
| `cactus-compute/cactus` | read | Confidence router; zero-copy mmap ~10× RAM |
| `furiosa-ai/draft-based-approx-llm` | read | SpecKV; evidence for P1; PSDC makes it free |
| **`Nota-NetsPresso/shortened-llm`** | **read (repo, this turn)** | **CPT ≫ LoRA at severe ratios; non-commercial licence** |
| `NVlabs/Minitron` + paper | read | Width vs depth trade-off; prune-before-quantise |
| OpenBMB BitCPM-CANN / MiniCPM4 | read | Scale-dependent ternary retention (90.1%→97.2%) |
| Nexa / OmniNeural-4B (→ GenieX) | read | NPU-aware co-design; repo redirect |
| ENERZAi Hexagon write-ups | read | First ternary-on-Hexagon precedent |
| Timaeus SMDL + LLC papers | read | Degeneracy, not Hessian, is leading order |
| LeJEPA / LeVJEPA / le-wm papers | read | SIGReg — hypothesis tested and **rejected** |
| QuaRot, SpinQuant, KurTail, DartQuant, KIVI, QuIP#, AQLM | read (papers) | Rotation/quantisation prior art |
| **OpenAI Navier–Stokes, 165pp** | **read end to end** | 1 novel technique, 1 bug found, 1 negative explained |
| `DEEPX-AI` | **read (this turn)** | DXNN SDK: ONNX→DX-COM→.dxnn, DX-RT runtime. **Proprietary, customers only.** Competing NPU silicon, not Snapdragon. Correctly rated low relevance. |
| `Taotern/GammaSpaceModel`, taotern.com, HF TaoNet | **searched, not resolved** | No substantive public technical content surfaced beyond the blog titles already catalogued. Treat catalogue entry as `?`. |
| `facebookresearch/vjepa2`, `MLO-lab/LeVJEPA` | **not read** | Video world models. The LeVJEPA *paper* was read; these repos were not. Nothing in the architecture depends on them. |
| `ggml-org` org page | partial | Read discussion #22019 (group-128 ternary for Ternary Bonsai 1.7B/4B/8B). Org page itself not surveyed. |
| `jaeyoon-enerzai` / ENERZAi org | **read — I was wrong to rate this low** | Led to 10 public repos incl. a **public 1.58-bit model optimizer**, `torq-compiler`, MLIR toolkit. Corrects my "proprietary" claim. Jaeyoon Yoo is ENERZAi's **CTO**. |
| `OpenBMB/MiniCPM-V` | **read — I was wrong to rate this low** | MiniCPM-o 4.5 is **document-parsing SOTA on OmniDocBench**, beating Gemini-3 Flash, GPT-5 and DeepSeek-OCR 2. Six models added. |
| `ChanwoongJeong-enerzai`, `ylecun` | searched, not resolved | No repos surfaced. `jooho-enerzai` (11 repos) exists as a sibling account. |
| `cactus-compute/cactus-react-native` | not read | RN binding of the already-read `cactus` engine. |

**Still genuinely unread: 6 items**, all judged low-impact and listed above with
the reason. Everything the architecture rests on has been read.

## Could not resolve

Nine URLs in the last message were `google.com/goto?url=CAES…` opaque redirects.
They carry no readable target. Paste the direct links if they matter.

## Claims deliberately NOT made

- No on-NPU training. QAIRT/QNN/Genie are inference-only.
- No closed-weight model (GPT-6 Astra, Claude, Gemini) running on device.
- No claim that a learned rotation beats a random Hadamard — I tested it and it lost.
- No 1.58-bit-on-Hexagon claim. bitnet.cpp says NPU support is still forthcoming.

---

## Update: `low_bit_edge_ai_repos_and_papers.pdf` (user-supplied research map)

This PDF **corrected the engine**. Read and integrated:

| Finding | Effect |
|---|---|
| **ENERZAi ran BitNet b1.58 2B on a QCS6490 Hexagon NPU via QNN with custom 1.58-bit kernels** | Replaced my binary "ternary = CPU only" with a three-state `TernaryNpuStatus`. My previous statement that there is no Hexagon path was wrong as an absolute. |
| ENERZAi Optimium: Opti 1.7B (Qwen3 1.7B + 1.58-bit QAT) at **32 tok/s on QCS6490**, bypassing QNN | Added as a catalogue entry with the result attributed and marked not publicly released. |
| QNN's layer library contains **no ternary matmul** — "the execution path does not exist", not a config flag | Kept as the stock-SDK baseline. Both halves matter: not free, not impossible. |
| Vec-LUT / OpenBitSys, Nexa SDK, Cactus + Needle 2, OpenBMB BitCPM, Taotern, Nota AI, FuriosaAI, DEEPX | Encoded in `LOWBIT_ECOSYSTEM` (11 orgs) with a Snapdragon-relevance rating each. |
| Bonsai 27B / Ternary-Bonsai-27B whitepapers | Added. ~~Bonsai-27B at 1-bit is **3.14 GB resident**~~ — **WRONG, corrected 2026-09-19 against byte-exact file sizes: the Q1_0 GGUF is 3,803,452,480 B = 3.80 GB.** See the Bonsai 2 section at the end of this file. |

New: `snapdragon_engine.py lowbit` prints the verified ternary-on-Hexagon
position with sources, plus the ranked ecosystem map.

Still not read (repos named in the PDF, not opened): `OpenBitSys/vlut.cpp`,
`NexaAI/nexa-sdk`, `cactus-compute/cactus`, `OpenBMB/MiniCPM`,
`Nota-NetsPresso/shortened-llm`, `furiosa-ai/draft-based-approx-llm`,
`DEEPX-AI`, `Taotern/GammaSpaceModel`, `microsoft/T-MAC`.

---

## Correction log — identifier audit

I was asked whether I had opened every GitHub repo and hyperlink in the
research-map PDF. **I had not.** I ran one search (the ENERZAi Hexagon claim)
and built the ecosystem map from the PDF's prose. Several identifiers in the
catalogue were therefore *inferred*, not confirmed. Auditing found real errors:

| Wrong (my invention) | Correct (verified) |
|---|---|
| `openbmb/BitCPM-CANN-3B` | `openbmb/BitCPM4-0.5B` — ternary QAT of MiniCPM4-0.5B |
| `openbmb/BitCPM-CANN-8B` | `openbmb/BitCPM4-1B` — ternary QAT of MiniCPM3-1B |
| BitCPM at 3B / 8B | The real BitCPM4 line is **0.5B and 1B**. I invented the sizes. |

BitCPM4 achieves roughly a 90% bit-width reduction; MiniCPM4 reports over 5×
generation acceleration on typical end-side chips. `openbmb/MiniCPM4-8B` and
`openbmb/MiniCPM4-8B-GGUF` are real and now used.

### What changed structurally

`ModelSpec.id_verified` now records whether an identifier was confirmed against
a live registry or merely inferred. **21 of 45 are confirmed; 24 are not** —
`models` marks them `+` / `?` so the distinction is visible rather than implied.

`snapdragon_engine.py verify` HEAD-checks every `hf_id`, `gguf_repo` and
`aihub_id` against huggingface.co and aihub.qualcomm.com. No token needed.
**Run it from an unrestricted network before you publish the repo.**

Status vocabulary is deliberately conservative: only HTTP 404/410 counts as
`MISSING`. HTTP 401/403 reports as `GATED/BLOCKED`, because a gated Hugging Face
repo and an egress proxy both return 403 while the model exists. Treating those
as failures would falsely condemn real models — a bug this sandbox exposed
immediately, since it blocks huggingface.co and produced 19 false MISSINGs
before the mapping was fixed.

### Still unopened from the PDF

`OpenBitSys/vlut.cpp`, `NexaAI/nexa-sdk`, `cactus-compute/cactus`,
`cactus-compute/cactus-react-native`, `OpenBMB/MiniCPM-V`,
`Nota-NetsPresso/shortened-llm`, `furiosa-ai/draft-based-approx-llm`,
`DEEPX-AI`, `Taotern/GammaSpaceModel`, `microsoft/T-MAC`, and the PrismML
Bonsai 27B whitepapers. Entries derived from them are marked `?`.

---

## Correction of a correction — BitCPM-CANN

**I was wrong, and the research map was right.** Last turn I told the user that
`openbmb/BitCPM-CANN-3B` and `-8B` did not exist and were my own invention.
They exist. I had found the separate `BitCPM4-*` line and wrongly inferred the
CANN line was fabricated — concluding absence from a single non-exhaustive
search. That is the same error class as the earlier "no Hexagon path for 1.58-bit"
claim: over-confident negatives.

Verified and now in the catalogue:

| Repo | Status |
|---|---|
| `openbmb/BitCPM-CANN-0.5B` / `-3B` / `-8B` | real |
| `openbmb/BitCPM-CANN-8B-gguf` (`bitcpm4-8b-tq2_0.gguf`) | real |
| `openbmb/BitCPM-CANN-3B-unquantized` | real — QAT checkpoint for continued fine-tuning |
| `openbmb/BitCPM4-0.5B` / `-1B` | also real — a **separate** line |

### What the repos actually contain

BitCPM-CANN is an end-to-end 1.58-bit ternary **training** system built on Huawei
Ascend NPU: ternary quantiser mapping weight groups to {−1, 0, +1} with group-wise
scaling, STE for gradients, QAT inside Megatron-LM with MindSpeed, then
post-training distillation. Roughly 5% training throughput cost vs full precision,
and ~6× inference memory reduction. Models ship pseudo-quantized, so they load
like ordinary fp checkpoints — no custom kernels needed.

### The finding that changes design decisions

Evaluated 1:1 against full-precision MiniCPM4 across 11 benchmarks:

| Size | Retention |
|---|---|
| 0.5B | **90.1%** |
| 1B+ | ≥95.7% |
| 3B | **97.2%** (best) |
| 8B | 95.7% |

**Ternary damage is scale-dependent.** Below ~1B it costs about 10% of capability;
at 3B it costs under 3%. The intuitive move — small device, so take the smallest
model *and* the most aggressive quantisation — compounds two losses and is close
to the worst available choice. BitCPM-CANN-3B at 0.55 GB and 97.2% retention
dominates BitCPM-CANN-0.5B at 0.09 GB and 90.1% on quality per byte.

Encoded as `ternary_viability()`; the planner now warns instead of silently
recommending the bad trade.

### Supporting datapoint

BitCPM-CANN-8B as TQ2_0 has been run on an iPhone 17 Pro at ~2.1 GB resident,
17 tok/s decode (62.7 tok/s on M4 Max). An INT4 8B needs ~5–6 GB resident. That
is an 8B-class model fitting where a 3B INT4 model would.

Also recorded: upstream llama.cpp GGUF ternary formats `TQ1_0` (~1.69 bpw) and
`TQ2_0` (2.06 bpw, 256-element blocks, one fp16 scale, shift-and-mask unpack).

---

## Read in full this turn (six-project reading list)

| Source | Status | What it changed |
|---|---|---|
| **Vec-LUT**, arXiv:2512.06443 (MobiSys 2026) | read | **Overturned an assumption hard-coded in this engine.** |
| **Qualcomm blog — OmniNeural-4B & NexaML on Hexagon** | read | NPU-aware co-design; Nexa→GenieX consolidation |
| **Minitron**, arXiv:2407.14679 + NVlabs/Minitron | read | Structural compression stage and the width/depth trade-off |
| AQLM, arXiv:2401.06118 (+ PV-Tuning 2405.14852) | catalogued, not fetched | 2-bit post-hoc option in the pipeline |
| Shortened LLaMA, arXiv:2402.02834 | catalogued, not fetched | Depth-pruning line |
| Cactus / Needle, arXiv:2607.18363 | catalogued, not fetched | Needle 2 as cascade draft tier |

### Vec-LUT — the correction

Authors are the T-MAC lineage (Wei, Cao, Liu). Two findings:

1. **"CPUs run these ultra-low-bit LLMs even faster than NPUs."** For 1.58/2-bit,
   CPU+LUT is not a fallback — it can be the fastest path. Every backend selector
   that ranks NPU > GPU > CPU unconditionally is wrong in that regime. **This
   engine's selector was one of them until now.**
2. Scalar LUT wastes bandwidth during *parallel* inference (prefill, test-time
   scaling, batch) via repetitive non-contiguous access per token. Vector LUT
   builds one unified table across parallel tokens, single 1→N lookup per index,
   plus LUT-centric tensor layout and cache-aware streamed lookup. **Up to 4.2×**
   over SOTA across 5 edge devices / 3 LLMs. Merged into llama.cpp.

Consequence: **prefill and decode want different backends.** `phase_aware_backend()`
now returns a recommendation per phase instead of one backend per model.

### OmniNeural-4B — not ternary

The user hoped this was a second ternary-on-NPU precedent. It is not. OmniNeural-4B
is the first *NPU-aware multimodal* model — text, voice and vision engineered from
the ground up for Hexagon rather than ported onto it. NexaML reaches the NPU through
QNN; NexaQuant reports ~10% lower perplexity and 2× context without speed loss.
NexaML also runs Qwen3-4B, YOLOv12 and PaddleOCR v4 on the NPU.
**`NexaAI/nexa-sdk` now redirects to `qualcomm/GenieX`** — that stack is Qualcomm's.

### Minitron — prune before you quantise

Prune embedding width, attention heads and MLP dim, then retrain with distillation:
up to **40× fewer training tokens** than training the small model from scratch (150×
for Llama-3.1-Minitron-4B: 94B vs 15T), **1.8× total compute saving** across a
15B/8B/4B family, and up to **16% better MMLU** than training from scratch.

The axis is a real trade-off: **width** pruning gives better accuracy and 1.8×
inference speedup; **depth** pruning gives 2.7× speedup but hits mathematical
reasoning noticeably harder. Nota AI's Shortened LLaMA is the depth line.

New `compress` subcommand orders the pipeline: structural first (expensive, host-side
training), numerical second (cheap, post-hoc). Never quantise before pruning.

---

## PSDC — a proposed architecture (`psdc.py`)

**Status: design proposal with untested predictions. Not a validated result.**

### The mechanism nobody states

Vec-LUT reports CPUs beating NPUs on ultra-low-bit LLMs but gives no
first-principles reason. Derived here and checked against published numbers:

1. Decode reads every weight per token, so `t ≥ weight_bytes / bandwidth`.
2. On X2 Plus the compute term for 8B is ~0.2 ms/token against a 13–53 ms
   bandwidth term — **100–260× smaller**. Decode is entirely bandwidth-bound.
3. QNN has no ternary matmul, so ternary weights must be unpacked to int8 —
   the NPU streams **8 bits/weight instead of ~2**.
4. Four times the bytes in a bandwidth-bound regime = ~4× slower. Vec-LUT
   measured up to 4.2×.

**The binding quantity is stored bit width on the wire, not TOPS.** This also
predicts when the NPU wins back — give it a native ternary kernel and it ties on
latency and wins on perf/watt, which is exactly ENERZAi's result. One rule
explains both papers.

### The architecture

Depth pruning removes *whole blocks* (Shortened LLaMA), leaving survivors intact.
So a depth-pruned decode network is a **layer subset** of the prefill network,
and its KV cache is a subset of the prefill network's cache — computed once, no
recomputation, no second cache.

| Phase | Network | Bits | Backend | Why |
|---|---|---|---|---|
| Prefill | full depth/width | INT4 | Hexagon NPU | compute-bound, parallel; accuracy set here |
| Decode | depth-pruned subset | ternary | CPU + vector LUT | bandwidth-bound; minimise bytes on the wire |

Predicted for 8B on X2 Plus: decode 28.5 → 95 tok/s (**3.33×**), decode weights
3.73 → 1.12 GB, TTFT unchanged. **Predicted, not measured.**

### Prior art, honestly

Layer dropping, early exit, self-speculative decoding and draft-guided KV
eviction all exist. The narrow delta claimed: the split is by *phase* rather than
speculation (no verification step, no acceptance-rate tax); bit width and backend
are chosen per phase from the bandwidth derivation rather than by sweeping; and
the KV cache is shared by construction rather than via a separate draft model.

### Two errors this module made and corrected

1. First version charged the LUT path **2 FLOP/param** — the multiply-accumulate
   figure. A LUT kernel does a lookup-plus-add amortised across a group. The
   error made the model predict the CPU was compute-bound at every size,
   contradicting Vec-LUT. Corrected via `lut_arith_efficiency`.
2. **That factor is the model's largest uncertainty and is NOT measured.** It was
   set to reproduce Vec-LUT's 4.2×, which is circular. Prediction P2 exists to
   calibrate it. Until then the absolute tok/s figures are ordering information,
   not forecasts.

### Kill conditions (`psdc.py falsify`)

- **P1 (fatal)** — does a depth-pruned sub-network produce usable continuations
  from full-network KV entries? If not, PSDC is dead and no patch saves it.
  ~1 hour on a Colab T4.
- **P2** — does throughput scale as 1/(bits × layers) within 20%? Calibrates the
  LUT factor. 12 free AI Hub jobs.
- **P3** — does ternary retention track per-layer *width* rather than total
  params? Genuinely open, cheap to test, publishable either way.
- **P4** — does phase-split beat uniform compression at equal total bytes?

---

## Gradient flows / Edge of Stability → `qat_optimizer.py`

**The Navier–Stokes paper is unrelated to this project.** It is a 165-page pure
mathematics construction proving finite-time blowup for forced 3D Navier–Stokes
(alternative (C) of the Millennium problem). "Flow" there is a fluid velocity
field; "gradient flow" in optimisation is an ODE on parameters. Same word,
unrelated objects. No link was manufactured.

**The gradient-flow links are relevant** — PSDC needs two training jobs
(depth-pruning distillation, ternary QAT), and EoS governs those.

### Predicted, tested, falsified

Reasoning from Cohen et al. (2103.00065) and central flows (2410.24206): EoS
drives λ_max → 2/η; quantisation is a weight perturbation; second-order damage
is ½δᵀHδ; so a larger learning rate should give a flatter minimum that quantises
better.

Measured on a 2-layer tanh net at three learning rates:

| η | λ_max | λ/(2/η) | EoS? | absmax | tr(H) | dL @ 4-bit |
|---|---|---|---|---|---|---|
| 0.1 | 1.106 | 0.06 | no | 0.727 | 10.6 | 0.00287 |
| 0.5 | 1.522 | 0.38 | no | 1.003 | 15.1 | 0.01353 |
| 2.0 | 1.002 | **1.00** | **YES** | 1.511 | 12.4 | **0.03295** |

The η=2.0 run sits exactly at the Edge of Stability — ratio 1.00, so the
phenomenon is real and the measurement works. It has the **lowest sharpness and
the worst quantisation damage**. Prediction falsified.

### What actually controls it

The second-order model is fine — it predicts measured damage within a factor of
2. The error was assuming tr(H) is the free variable. It barely moves (10.6,
15.1, 12.4). **absmax** moves: 0.727 → 1.511, a 2.08× rise, so Δ² grew 4.32×.
Learning rate matters only through its effect on weight magnitude.

    controllable quantity = Δ² · tr(H),  Δ = absmax_group / q_max

This independently rediscovers, from the optimisation side, the very first
finding in this project: **scale granularity dominates** (FINDINGS.md). Smaller
groups mean one outlier inflates Δ for fewer weights.

### The actionable result — measured, not predicted

The right regulariser is a penalty on the per-group **L∞** norm, not L2 weight
decay. L2 shrinks the norm; Δ is set by the per-group maximum, which an outlier
dominates while contributing almost nothing to the norm. The subgradient is
nonzero only at each group's argmax, so it costs one element per group.

Measured at 3-bit, group=32:

| λ | group absmax | clean loss | quant damage | **deployed** |
|---|---|---|---|---|
| 0 | 0.5951 | 0.00047 | 0.03621 | 0.03668 |
| 1e-4 | 0.4557 | 0.00059 | 0.01256 | 0.01315 |
| **1e-3** | 0.1847 | 0.00216 | 0.00337 | **0.00553** |
| 5e-3 | 0.0161 | 0.00620 | 0.00005 | 0.00625 |

The best deployed model is **4.6× worse before quantisation and 6.6× better
after it**. Past the optimum the regulariser over-shrinks and clean loss
dominates again — a genuine interior optimum, not a monotone knob.

**Selecting on clean loss picks the wrong model.** `select_lam()` selects on
deployed loss instead.

### What EoS is still good for

λ_max ≈ 2/η gives a free sharpness estimate with no Hessian solve; being at EoS
says η is as large as the landscape permits (useful for the ~94B-token
distillation job where wall-clock binds); and central flows predict when
sharpness stabilises, i.e. when it is safe to begin the QAT phase.

---

## Correction: the Navier–Stokes paper DOES contribute — I was wrong to dismiss it

I read three pages and the table of contents, then judged it unrelated. That was
not reading it. On being challenged I went through §2–3 and §9 properly.

**Not the physics, and not gradient flows.** The contribution is the **proof
architecture**. §9 runs a correction cycle with

    σ₀ = 1/5,   σ_{j+1} = σ_j + 1/10,   K_m independent of j

Each cycle buys a *fixed additive gain in the residual decay exponent* while the
derivative cost *does not accumulate across iterations*. That is precisely the
bookkeeping question for multi-stage residual quantisation (AQLM additive
codebooks, RVQ). So I extracted it as a hypothesis and tested it.

### Result 1 — the structure transfers (confirmed)

Residual quantisation gives a constant multiplicative gain per stage at constant
bit cost per stage:

| bits/stage | gain per stage (stable from stage 2) |
|---|---|
| 2 | 4.4–4.5× |
| 3 | ~40× |

Geometric residual decay, linear in stage count — the exact shape of σ_j.

### Result 2 — it buys nothing at equal bits (negative)

Against a single deeper quantiser at matched budget: **1.00×, 1.02×, 0.83×,
0.86×, 0.97×**. A wash, sometimes worse. A uniform quantiser is already near
rate–distortion optimal on well-behaved weights.

### Result 3 — it buys 5.03× on outlier-heavy weights (positive)

Where a single quantiser's group scale is hostage to outliers, the residual
retains exploitable structure. `plan_stages()` gates on `outlier_strength(w)` =
max|w| / p99(|w|); above ~10, stage.

### Result 4 — the paper's actual mechanism FAILS here, and the reason is exact

Convex integration corrects on *successively finer scales*. Transplanted as
coarse→fine group sizes, it scored **0.45×, 0.51×, 0.07×, 0.08×** against a
single quantiser. The disanalogy is precise:

> In Navier–Stokes the nonlinearity **regenerates** structure at finer scales, so
> every correction has fresh structure to cancel. Quantisation residuals
> **whiten** with each stage. Convex integration climbs a cascade that keeps
> producing detail; residual quantisation runs out of detail to exploit.

That is the real boundary of the transfer — not "different fields", which is what
I said the first time and was the lazy answer.

### Standing correction to my judgement

Twice in this project I concluded absence too quickly (BitCPM-CANN, "no Hexagon
ternary path"). This is a third instance of the same failure mode applied to a
document rather than a repo. The pattern is specifically my **negatives**.

---

## The Navier–Stokes paper's real contribution: `generator_quant.py`

Read further (§3.5, localization). Found a **technique**, not an analogy, and it
fixes a real bug in this project's own earlier work.

### The idea

§3.5 localizes the flow. The obvious move — multiply the velocity by a cutoff —
destroys incompressibility. The paper instead multiplies the **vector potential**
and *then* takes the curl: `u = curl(cA) + cB e_θ`. Curl of anything is
divergence-free, so the constraint holds structurally. Figure 6: *"Incompressibility
is preserved throughout."*

> **Never operate on the constrained object. Operate on its potential and reconstruct.**

### The bug it fixes

`sigil/npu.py`'s exact folding identity requires R orthogonal. But deployment means
*storing* R, and storing means *quantising* R — which pushes it off the orthogonal
group and silently voids the guarantee the whole design rests on. Measured:

| bits | direct: max\|RRᵀ−I\| | direct: logit rel err |
|---|---|---|
| 8 | 2.909e-03 | 7.414e-03 |
| 4 | 4.750e-02 | 1.361e-01 |
| 2 | 6.152e-01 | **9.824e-01** (98%, fully broken) |

### The fix

R is the constrained object; its potential is the skew generator A, with
R = Cayley(A). Quantise A, reconstruct R. Cayley of any skew matrix is exactly
orthogonal, so the constraint survives at **any** bit width:

| bits | generator: max\|RRᵀ−I\| | logit rel err |
|---|---|---|
| 8 | 8.882e-16 | 1.438e-15 |
| 4 | 8.882e-16 | 1.431e-15 |
| 2 | 9.992e-16 | 1.470e-15 |

### The honest check

Logit invariance is partly trivial — qᵀk is invariant under *any* orthogonal R, so
exactness only proves R_g is still **a** rotation, not the right one. Generator
quantisation genuinely changes which rotation you get. So: does it still spread
outliers?

| | relMSE @ int4 |
|---|---|
| no rotation | 0.05272 |
| exact R | 0.01317 |

| bits | ‖R_g−R‖_F/‖R‖_F | relMSE | vs exact |
|---|---|---|---|
| 8 | 0.0053 | 0.01317 | 1.00× |
| 4 | 0.1008 | 0.01337 | 1.02× |
| 2 | **0.7129** | 0.01405 | **1.07×** |

At 2 bits R_g has drifted **71%** and still delivers 93% of the benefit, still
3.75× better than no rotation. **Why:** outlier spreading is a *generic* property
of rotations, not of one specific R. Drifting to another rotation is harmless;
leaving the orthogonal group is fatal.

### What it buys

A is skew, so only d(d−1)/2 entries are free. At d=128: R at fp16 = 262144 bits;
A at 2 bits (group 32) = 20320 bits — **12.9× smaller**, and the folding identity
stays exact instead of approximately true.

### Corrected along the way

My first version guarded against a "near-singular Cayley". Impossible for skew
input: eigenvalues of a skew matrix are purely imaginary, so 1−iμ/2 is never zero
and Cayley is defined on **all** of so(d). Verified at scale 1e6 (orth err 8e-15).
The guard now correctly targets non-skew input, which is a real caller bug.

`falsification()` ships three kill conditions; G1 (real transformer weights) is the
one that matters.

---

## SLT / Timaeus: `slt_compressibility.py`

Read: SMDL (arXiv:2510.12077, Urdshals/Lau/Hoogland/van Wingerden/Murfet),
Watanabe's volume theorem, timaeus.co/research.

**The theory says my damage model uses the wrong invariant.** SMDL states it
directly: *"where geometric invariants like the curvature determined by the Hessian
appear in the description length, the important geometric feature in the singular
case is degeneracy."* Networks are singular — the Fisher information is degenerate
— so `tr(H)` is leading-order only for regular models. The replacement is
Watanabe's Vol({w : L−L₀ ≤ ε}) ~ c·ε^λ, λ the RLCT; regular models give λ = d/2.
SMDL reports a close, sometimes **linear**, LLC↔compressibility relationship on
Pythia up to 6.9B.

**I could not reproduce it.** My volume-scaling estimator returned λ̂ ≈ 2.2 for
every configuration — both targets, widths 16/32/64. No signal. This is a negative
result about **my estimator**, not about SMDL: naive ball sampling fails in high
dimension because nearly all mass sits near the surface and never explores the
degenerate directions. Timaeus publish several papers on estimation methodology and
state they lack theoretical knowledge of true LLC at transformer scale. Correct
tool: `github.com/timaeus-research/devinterp` (SGLD local posterior sampling).

**What survives:** the absmax result and SLT are not rivals — they are two factors
of one product. `damage = (displacement, set by absmax) × (landscape price, set by
degeneracy)`. `qat_optimizer.py` controls the first and delivers a measured 6.6×.
The second is the theoretically correct leading term and I could not measure it.
Both belong in the record.

---

## T-MAC / T-MAN — validates the model, then overturns its recommendation

Read at last (I had cited it twice without opening it). MIT, Microsoft Research,
EuroSys 2025 (arXiv:2407.00088).

### Validation

T-MAC's README states plainly: **"T-MAC is even faster than the NPU in token
generation speed on the latest Snapdragon X Elite chipset"** — independent
confirmation of the bandwidth derivation in `psdc.py`. It also reports that
**LUT kernels scale linearly with weight bit-width**, which is exactly the
linearity the roofline assumes. Mechanism: group one-bit weights, precompute all
partial sums into a LUT, eliminating multiplications and reducing additions —
no dequantization at all. 48 tok/s for 3B BitNet on Snapdragon X Elite; 4×
throughput and 70% energy vs llama.cpp.

### The correction: T-MAN

Inside the same repo is **T-MAN**, an NPU LUT kernel: **50 tok/s for
BitNet-2B-4T on Snapdragon 8 Gen 3 — 2× faster than T-MAC on CPU and 1.4×
faster than Qualcomm QNN on Llama-3.1-8B.** NPU-only, so it doesn't contend with
CPU/GPU. Prebuilt APK. Techniques: hardware-aware tile quantization aligned to
NPU memory access patterns, plus LUT replacements for Softmax and dequantization
(up to 19.0× on mpGEMM, 2.2× on Softmax).

So this is the **third** verified ternary-on-Hexagon precedent, and the first
open-source one. `phase_aware_backend()` now routes ultra-low-bit to the NPU LUT
kernel; CPU+LUT is the fallback, not the target. Three self-test assertions that
encoded the pre-T-MAN view were corrected.

### A calibration failure worth recording

My roofline predicted "a native ternary kernel restores the NPU." T-MAN confirms
the direction. But checking absolute numbers:

| path | model predicted | measured |
|---|---|---|
| NPU + LUT kernel | 42 tok/s | 50 |
| CPU + LUT | 122 tok/s | **25** |

Within 16% on the NPU path, **4.9× overshoot** on the CPU path. Cause:
`lut_arith_efficiency` was set to 6.0 to reproduce Vec-LUT's headline "up to
4.2×" ratio — which I flagged as circular at the time, and it was.

Recalibrating against T-MAC's *absolute* measurement (25 tok/s, 2.4B @ 1.58 bit,
8 Gen 3): 40 ms/token measured against an 8.2 ms bandwidth term, so the CPU
really is compute-bound. Implied throughput 120 G-ops/s → factor **0.60**, ten
times smaller than my guess. My *first* version's instinct was closer than my
"fix."

> **Fitting a model to a published ratio reproduces the ratio and nothing else.
> Only an absolute measurement constrains it.** The ratio-fit was right about
> ordering and badly wrong about magnitude.

---

## furiosa-ai / draft-based-approx-llm — supports PSDC's fatal prediction, then improves it

Galim, Ewer, Kang, Lee, Koo, Lee. **ICLR 2026**, arXiv:2506.08373.
SpecKV (lookahead draft → precise KV dropping), SpecPC (draft attention →
prompt-token discarding), SpecKV-PC (cascade). First use of draft models for
*approximate* inference rather than lossless speculative decoding. Evaluated on
RULER to 65k context; supports Qwen2.5 and Llama-3.

### Evidential: bears directly on P1

Their central empirical finding is a **strong correlation between the attention
patterns of draft and target models**. PSDC's fatal prediction P1 asks whether a
depth-pruned subnet can consume KV computed by the full network. A pruned subset
*shares weights* with the target, so it should correlate at least as strongly as
an independently trained draft.

**This is the first independent evidence in P1's favour.** It is not proof:
attention-pattern correlation is not coherent generation from borrowed KV, which
is the stronger claim. `speckv_evidence_for_p1()` states both halves, and P1
remains UNTESTED and still fatal.

### Architectural: PSDC makes SpecKV free

SpecKV's cost is that it **needs a separate draft model** — extra weights to
train, store and stream. PSDC already has one:

> **the depth-pruned decode network IS the draft**

It shares weights with the target (no extra storage), is already resident (no
extra streaming), and is already running during decode. SpecKV-style eviction is
nearly free inside PSDC where standalone it costs a model.

And it compounds in the right direction. This project's roofline says decode is
bandwidth-bound and KV is the term that grows with context. PSDC shrinks the
**weight** stream; SpecKV shrinks the **KV** stream. Same bottleneck, two sides:

| KV stream per decoded token, 8B-class, 32k ctx | |
|---|---|
| full | 2.00 GiB |
| + PSDC layer subset (0.6) | 1.19 GiB (1.68×) |
| + SpecKV-PC eviction (0.5 / 0.7) | **0.42 GiB (4.81× total)** |

`kv_stream_bytes()` and `SpecKVConfig` implement this; 37/37 tests.

### Still unread

`vlut.cpp` source (paper read), `cactus`, MiniCPM repo (models catalogued),
`shortened-llm` (paper read), DEEPX, GammaSpaceModel, vjepa2, `ylecun`,
`ggml-org` org page, the ENERZAi personal profiles. Of these I now expect
`cactus` to matter most — it is the Android runtime layer the engine still
covers only via llama.cpp and GenieX.

---

## P1 RUN — PSDC's fatal prediction, narrowed but not closed

`p1_test.py`. No T4 or HF access here, so I ran the decisive version I could: a
real 12-layer decoder-only transformer in NumPy.

### Sharpening the question first

The naive framing conflates two effects:

- **(A)** depth pruning degrades the model — **known**, Minitron / Shortened
  LLaMA, recoverable by distillation. Not PSDC's problem.
- **(B)** reading KV computed by the **full** network — **PSDC's actual claim**,
  untested.

So the experiment holds pruning fixed and varies only the KV source: subnet
computes its own K,V vs subnet reads the full network's cached K,V. Measured as
KL from the full model's own output distribution.

### Result

| retained | frac | KL own-KV | KL full-KV | ratio |
|---|---|---|---|---|
| 9 of 12 | 0.75 | 0.1968 | 0.1633 | **0.83×** |
| every 2nd | 0.50 | 0.4327 | 0.3796 | **0.88×** |
| every 3rd | 0.33 | 0.4931 | 0.3328 | **0.67×** |
| every 4th | 0.25 | 0.4857 | 0.4907 | 1.01× |

Kill condition was >2× worse. Worst observed: **1.01×**. Sharing full-network KV
costs nothing — and in three of four cases is *better* than the subnet's own KV,
plausibly because full-network KV comes from the full residual stream and is
closer to what each retained layer was built to consume. **P1 survives.**

### Two qualifications the numbers demand

1. **Untrained model.** This tests residual-stream compatibility — the mechanism
   PSDC depends on — not task accuracy. Trained weights could behave differently.
2. **At 25% depth the ratio is meaningless.** KL is 0.49 against a
   uniform-distribution reference of 0.50: the pruning has destroyed the model,
   and the ratio is ~1 only because both variants are equally destroyed. The
   informative rows are 0.75 and 0.50.

`falsification_suite()` P1 is now marked *"NARROWED, NOT CLOSED"* and still fatal.

### Aborted approach, recorded

First attempt trained the model with evolution strategies to test task accuracy
directly. It diverged (loss 8.9 → 95) — ES on ~30k parameters with pop=14 is
essentially noise. Rather than spend the budget on an optimiser, I narrowed the
question to the part that is PSDC-specific and needs no training. The learning
sanity gate in the first version correctly refused to report a comparison on a
model that had not learned.

---

## cactus-compute — ships the router I failed to build

Three layers: **Engine** (OpenAI-compatible APIs for C/C++, Swift, Kotlin,
Flutter, React Native), **Graph** (zero-copy computation graph, PyTorch-like),
**Kernels** (ARM SIMD for Snapdragon/Apple/Google/Exynos/MediaTek, custom
attention with KV-cache quantisation, chunked prefill, streaming LLM).

Two things matter for this project:

1. **Cactus Hybrid routes to cloud on real-time model *confidence*.** That is
   exactly the cascade escalation tier I proposed and could not build — my
   isotropy router was mathematically impossible (‖XR‖ is rotation-invariant).
   Cactus has a working version in production. The tier is real; geometry was
   simply the wrong signal for it.
2. **Zero-copy mmap, ~10× lower RAM** — LFM2.5-1.2B served in 76MB. This
   invalidates an assumption in `CapabilityMatrix`: I compute `fits_in_ram`
   assuming weights are fully resident. With mmap they need not be. Several
   BLOCKED verdicts in the catalogue survey are likely wrong under Cactus.

Also of note: CQ quantisation offers **fractional bit widths (2.54, 3.26)**, and
the default benchmark model is `google/gemma-4-E2B-it`. Roadmap through mid-2026
adds Qualcomm/Google NPU support.

**Known gap left open:** `fits_in_ram` should model mmap. Not fixed here.

---

## Navier–Stokes §3.4, read properly — the correction cycle, and a bug it found

I had read maybe 30 of 165 pages. Read §3.4 (correction and summation) in full.
The load-bearing equation is the residual update:

    R(u[j+1], p[j+1]) = R(u[j], p[j]) + L_{u[j]}(δu_j, δp_j) + ∇·(δu_j ⊗ δu_j)

> *"Canceling a selected source also introduces linear remainders and quadratic
> interactions."* … *"we recompute the full residual after each operation, so
> newly created terms enter the next stage."*

Four operations per cycle (wave-amplitude equations; signed amplitude increments
whose symmetrized cross-covariance supplies the averaged-stress correction;
inverting the fast auxiliary-time derivative; five radial moment equations, two
preserving conserved integrals and three cancelling linear defects). Pressure is
reconstructed after **each** operation. Background, leading amplitudes and
inverse operators stay fixed through the induction.

### The bug it exposed in my own module

`plan_stages()` treats quantisation error as **additive per matrix**. In a
network it is not: once layer *l* is quantised, layer *l+1* receives a drifted
input and the errors compose nonlinearly — exactly the `L(δ) + ∇·(δ⊗δ)` terms.

Measured, 6-layer tanh network with channel outliers, group=32:

| bits | parallel | sequential | gain |
|---|---|---|---|
| 6 | 0.11424 | 0.04893 | **2.33×** |
| 5 | 0.24728 | 0.08156 | **3.03×** |
| 4 | 0.49854 | 0.22390 | **2.23×** |
| 3 | 1.01222 | 0.46430 | **2.18×** |

Sequential = at layer *l*, re-solve the weights to map the **already-drifted**
input onto the **original** target pre-activation, then quantise, then propagate
the quantised path forward.

**Novelty: none.** This is what GPTQ and AWQ already do. The value is purely
diagnostic — reading the NS cycle revealed my module was doing the naive additive
thing and was silently 2–3× worse. `sequential_quantisation_gain()` records it and
`qat_schedule()` now emits the warning.

Two failed attempts recorded on the way: the first test had relative error >1.0
at every bit width (network fully destroyed, comparison meaningless), and my
first `lstsq_correct` trivially recovered the original matrix — a no-op, which is
why sequential first appeared to tie with parallel.

---

## mmap gap closed

`CapabilityMatrix` assumed weights must be fully resident. Cactus's zero-copy
mmap breaks that (LFM2.5-1.2B in 76 MB). Under mmap the binding quantity is the
**working set** — KV cache plus roughly a few layers in flight — not the model.

`_mmap_working_set()` now models it, deliberately conservatively (~3 layers + KV
+ 0.3 GB runtime), and the verdict distinguishes "fits resident" from "fits under
mmap, needs an mmap-capable runtime, expect storage-bandwidth stalls."

**Nine verdicts flipped from BLOCKED to runnable** on this 3.9 GB host:
GPT-OSS-20B, Mistral-7B, Llama-3.1-8B, Qwen3-8B, Bonsai-27B, Ternary-Bonsai-27B,
MiniCPM4-8B, Gemma-4-31B, Gemma-4-26B-A4B. Survey now 44 runnable / 3 blocked
(the 3 remaining are the closed-weight models, which no runtime fixes).

Caveat kept in the docstring: a model that "fits" by working set may still
thrash. This says *worth trying*, not *will be fast*.

---

## §§4–8 read — the paper is now covered end to end

### §6 (auxiliary torus, separation of supports) — the productive one

> *"Oscillations whose supports in the remaining variables overlap receive
> **disjoint auxiliary supports, so their cross products vanish after
> evaluation**. Harmonics of the same localized oscillation still interact."*

A direct construction for killing the δ_i ⊗ δ_j interaction terms: give
different corrections disjoint supports and they never multiply. Mechanically
they lift onto T² with an independent variable Y, evaluate at Y = v_r r^{d_r} +
v_t t (mod Z²), with an integer matrix J_g whose distinct eigenvalues
(Λ_g = 4−√2, T_g = 4+√2) give different rates of radial vs temporal variation.

**This explains a negative result I could not previously account for.**
`staged_quantisation_advantage()` reports multi-stage residual quantisation
buying ~nothing at equal bits. I blamed residual whitening. Lemma 6.1 supplies a
second cause: the stages **overlap** — every stage touches every coordinate — so
their errors interact and later stages' group scales stay hostage to the same
outliers.

Measured (d=4096, 1% outliers ×25, group=32, ~4.5 bits/weight):

| scheme | relMSE | bits/weight |
|---|---|---|
| uniform 4-bit | 0.018186 | 4.50 |
| 2 overlapping stages @2-bit | **0.031409** | 5.00 |
| disjoint: top 1% @8b, rest @4b | **0.001611** | 4.66 |
| disjoint: top 2% @8b, rest @4b | **0.001217** | 4.82 |

Overlapping staging spends *more* bits for *worse* error. Disjoint supports give
**11–15×** at the same budget. Second effect specific to group quantisation:
pulling outliers into their own partition removes them from the low-precision
partition, so they stop inflating its group scales.

**Novelty: none** — LLM.int8(), AWQ, SqueezeLLM. `disjoint_partition_plan()`
implements it and, unlike the usual telling, charges the **index overhead** for
the sparse high-precision set.

This closes a loop. "Scale granularity dominates" (FINDINGS.md, first
experiment), "absmax is the lever" (`qat_optimizer.py`), and "isolate outliers so
they stop setting the scale" are three views of one fact, reached independently.

### §4, §5, §7, §8 — read, and domain-specific as expected

- **§4** builds the leading profiles E, U, Π. Notable: **five cumulative radial
  integrals** whose preservation lets profiles on different radial intervals be
  joined without disturbing the prescribed exterior. General principle —
  maintain a fixed invariant set so local pieces compose globally — but I found
  no quantisation analogue worth coding.
- **§5** corrects the base flow order by order in q^{2h}, with an explicitly
  formal expansion: *"convergence of the unmodified infinite series is not
  asserted."* Smooth fields with matching asymptotics are produced separately.
- **§7** realizes the stress from wave covariances. The interesting condition is
  a **positive covariance representation** with a strict cone condition (7.1):
  start from strictly positive squared amplitudes, then *"linearizing at those
  fixed amplitudes permits subsequent stress increments of either sign."* Start
  strictly interior so later corrections have headroom both ways. The
  quantisation analogue is clipping-ratio headroom, which `IntRTN.search_clip`
  already does; no new code.
- **§8** supplies compactly supported mean corrections via five radial moment
  equations — two preserving the conserved integrals, three cancelling linear
  defects.

**Verdict on my earlier guess.** I predicted §§4–8 would be "the hardest and most
domain-specific parts, where the machinery exists to satisfy fluid-specific
constraints." That was right for §§4, 5, 7, 8 — and wrong for §6, which produced
an 11–15× measured result. Fourth time in this project my prediction of absence
was the thing that failed.

### Full ledger from this paper

| Outcome | Where |
|---|---|
| Novel technique | `generator_quant.py` — 12.9× smaller rotation storage, exactness preserved |
| Found a 2–3× bug in my own code | sequential vs parallel quantisation (§3.4) |
| Explained a negative result + 11–15× fix | disjoint supports (§6) |
| Transfer confirmed then bounded | residual staging: 5× on outlier-heavy weights, nothing otherwise (§9) |
| Transfer failed, reason identified | coarse→fine: fluid nonlinearity regenerates structure, quantisation residuals whiten |

---

## Final sweep — the remaining repos

### `Nota-NetsPresso/shortened-llm` — corrects my training planner

Read the repo, not just the paper. Pipeline is block pruning → retraining →
zero-shot eval, criteria PPL and Taylor+, GPTQ applied afterwards (independently
confirming prune-before-quantise).

**The correction:** *"In retraining pruned models for quality recovery,
continued pretraining (CPT) on a large corpus **markedly outperforms LoRA-based
tuning, particularly at severe pruning ratios**."*

`TrainingPlanner` defaulted to QLoRA. PSDC prunes 40% of depth — squarely the
severe regime — so it was recommending the wrong method. Now warns.

Their measured CPT cost for Vicuna-7B, 8×H100:

| target | tokens | wall clock |
|---|---|---|
| 5.5B (20% pruned) | 37B | 6 days |
| 2.7B (60%) | 150B | 12 days |
| 1.5B (80%) | 271B | 11 days |

That is far beyond anything a Colab session reaches, and my earlier train-plan
estimates were optimistic by orders of magnitude for this stage.

**Licence flag for your submission:** Shortened LLaMA checkpoints are
**non-commercial, all rights reserved by Nota Inc., research use only**. The
method is free to reimplement; the weights are not shippable. `TrainingPlanner`
now emits this.

### `OpenBitSys/vlut.cpp` — source read, and it revises the phase split

MIT, a lightweight llama.cpp extension. Three techniques: LUT replacing
dequant+multiply; the vector LUT paradigm doing 1→N lookup that **turns random
lookup into contiguous vector addition**; LUT-centric tensor layout with
cache-aware streamed lookup. Heuristic tiling, no tuning. All mainstream CPUs and
OSes including Android.

**The revision:** its target scenarios are *"prefilling, serving, and parallel
test-time scaling **and speculative decoding** (parallel output)"*. PSDC's decode
tier, run speculatively, **is** a parallel scenario — so Vec-LUT applies to
decode as well, not only prefill. `phase_aware_backend()`'s prefill/decode split
is too coarse; the real axis is sequential-vs-parallel, and speculative decode
sits on the parallel side.

### `DEEPX-AI` — read, correctly rated low

DXNN SDK: ONNX → DX-COM compiler → `.dxnn` binary → DX-RT runtime (C++/Python).
Proprietary, supplied only to DEEPX NPU customers. Competing silicon, no
Snapdragon path. The `low` relevance rating in `LOWBIT_ECOSYSTEM` stands.

### `ggml-org` — discussion #22019

Upstream discussion on supporting Ternary Bonsai in llama.cpp and a **group-128
ternary format**, following the earlier Q1_0 work. Ternary Bonsai ships at
**1.7B / 4B / 8B** — my catalogue had only 8B and 27B.

### Not resolved / not read, with reasons

- **Taotern**: searched; no substantive public technical content beyond blog
  titles already catalogued. Entry stays `?`.
- **vjepa2, MLO-lab/LeVJEPA**: video world-model repos. The LeVJEPA paper was
  read. Nothing in the architecture depends on the repos.
- **ylecun, ENERZAi personal profiles**: personal GitHub profiles.
- **cactus-react-native, MiniCPM-V**: an SDK binding and a vision model.

### Defect fixed in this file

The "Not read" list at the top of this document was written early and never
updated. It still named T-MAC, furiosa, cactus, Vec-LUT and MiniCPM as unread
long after those sections recorded reading them — **a file that contradicted
itself, in the document I told you to trust for exactly this.** Replaced with a
dated authoritative status table that supersedes the work log below it.

---

## Final five — and I was wrong to rate two of them low

I rated these "low expected technical content" without looking. Two of five
mattered, which is the **fifth** time in this project a negative prediction of
mine failed.

### `jaeyoon-enerzai` → the ENERZAi org — corrects a claim I made

The personal profile led to the **`ENERZAi` GitHub org, 10 public repos**:

- **`ENERZAi-Optimium-1.58-bit-Model-Optimizer`** — a public 1.58-bit optimizer
- **`torq-compiler`** — their own compiler
- an **MLIR-based retargetable ML compiler and runtime toolkit**, plus forks of
  `llvm-project` and `torch-mlir`
- **`Optimium-Examples`** — Optimium Runtime usage examples

**I had recorded Optimium as proprietary and "not publicly released."** That was
too strong, and it mattered: it made ENERZAi's ternary-on-Hexagon precedent look
unreproducible when public tooling exists. Corrected in
`TERNARY_NPU_PRECEDENT["public_tooling"]`.

Also recovered: Jaeyoon Yoo is ENERZAi's **CTO**, and Opti 1.7B ran at
**~680 MB peak memory** alongside the 32 tok/s I already had. Their own framing —
*"LLM token generation is inherently a memory-bound process, bottlenecked by
memory bandwidth rather than compute"* — is a **third independent confirmation**
of this project's roofline, after Vec-LUT and T-MAC.

### `OpenBMB/MiniCPM-V` — the most useful single find of this sweep

I dismissed it as "a vision model." It is the document-intelligence stack for the
use case I proposed for your submission.

**MiniCPM-o 4.5** achieves **state-of-the-art end-to-end English document parsing
on OmniDocBench, outperforming Gemini-3 Flash, GPT-5, and DeepSeek-OCR 2** — at 9B
parameters, Apache-2.0. Also 77.6 OpenCompass (surpassing GPT-4o and Gemini 2.0
Pro), and simultaneous video+audio streaming.

Six models added to the catalogue:

| model | note |
|---|---|
| MiniCPM-o 4.5 (9B) | document-parsing SOTA; beats specialised OCR tools |
| MiniCPM-V 4.6 (8B) | "pocket-sized", phone-targeted, GGUF, free API key |
| MiniCPM-V 4.5 (8B) | Qwen3-8B + SigLIP2-400M, int4/GGUF/AWQ in 16 sizes, 30+ languages |
| MiniCPM-V 4.0 (4B) | beats GPT-4.1-mini; official open-source iOS app |
| MiniCPM5-2B | newest on-device checkpoint |
| **MiniCPM-S-1B** | **87.89% FFN sparsity, 84% FFN FLOP reduction** |

**MiniCPM-S-1B exposes an axis this engine does not model at all.** Everything
here compresses via bit width, pruning or KV eviction. Activation sparsity is a
fourth axis — 84% of FFN FLOPs eliminated at inference, orthogonal to all three.
`compression_pipeline()` has no term for it. Flagged, not fixed.

The 30+ language support and OCR strength make this the natural base for the
Indic document-intelligence use case in `SUBMISSION.md`, replacing the
EasyOCR/TrOCR + separate-LLM arrangement with one Apache-2.0 model.

### The other three

- **`ChanwoongJeong-enerzai`** and **`ylecun`** — searched, no repos surfaced. A
  sibling account `jooho-enerzai` (11 repos) exists.
- **`cactus-compute/cactus-react-native`** — React Native binding of the `cactus`
  engine already read. Nothing new on the compression path.

### Running tally on my judgement

Five negative predictions, five failures: BitCPM-CANN existed; a Hexagon ternary
path existed; the Navier–Stokes paper did contribute; §6 was not barren; and now
two of these five "low content" profiles carried real corrections. **Every
measurement in this project held. Every confident absence did not.**

---

## Edge-AI / parallelism reading list — three material changes

### Hogwild! Inference (arXiv:2504.06261, NeurIPS 2025 spotlight) — strongest evidence yet for P1

Multiple LLM workers run in parallel, synchronising through a **concurrently
updated SHARED attention cache**, exploiting RoPE to avoid recomputation. The
finding that matters here: *modern reasoning-capable LLMs perform inference with
a shared KV cache **out of the box, without additional fine-tuning**.*

P1 asks whether a network can decode from KV entries it did not compute. Hogwild
answers a neighbouring instance of exactly that, **on real models, with real
generation quality** — a far higher bar than `p1_test.py`'s untrained proxy.

Honest limit: their workers are *identical full networks*. PSDC's consumer is a
*depth-pruned subset* whose residual statistics differ from the writer's. Hogwild
removes the "whose cache is it" objection, not the "different network" one.
`hogwild_evidence_for_p1()` states both halves. **P1 remains open.**

### PASTA (MIT CSAIL + Google) — the prefill/decode split is too coarse

PASTA trains LLMs to recognise **semantic independence** and decode independent
chunks in parallel — learned, replacing the brittle hand-crafted syntactic
heuristics earlier work used.

Combined with Hogwild and with vlut.cpp naming speculative decoding as a parallel
scenario, **three separate mechanisms make decode parallel.** So my
prefill/decode axis was wrong: the real variable is **tokens in flight**, and
prefill is merely the case where that is large by default.

`effective_parallelism()` now computes it. Plain decode is parallelism-1 and
bandwidth-bound; 4× speculative × 3 PASTA chunks is 12 in flight and lands in the
regime where vector-LUT kernels pay off. A planner that assumes decode is
sequential picks the wrong kernel whenever any of the three is in use.

### arXiv:2511.07425 — runtime choice is worth 4×, and I don't model it

25 quantised models across Raspberry Pi 4/5 and Orange Pi 5 Pro:

- SBCs reliably support models **up to 1.5B parameters**
- **Llamafile: up to 4× higher throughput and 30–40% lower power than Ollama** —
  same hardware, same models

My `_select_backend` returns "llama.cpp CPU" as though it were one thing. A
measured 4× throughput and 30–40% power gap between two CPU runtimes says runtime
is a **first-class variable** I do not model at all. Flagged, not fixed — fixing
it properly needs a runtime × model benchmark grid.

### Also read

- **SAIL (arXiv:2509.25853)** — SRAM-accelerated LUT-GEMV. States both of this
  project's core findings independently: *"optimal bit precision varies across
  models **and layers**"* (supports the per-layer allocation in `allocate_bits`)
  and *"the **memory-bound** nature of the token generation phase creates severe
  performance bottlenecks"* — a **fourth** independent confirmation of the
  roofline, after Vec-LUT, T-MAC and ENERZAi.
- **HybridGen (arXiv:2604.18529)** — CPU-GPU hybrid generative inference, Apr 2026.
- Raschka's LLM research lists, largo.dev frontier architectures 2026, Stanford
  HAI AI Index 2026, derekmolloy.ie edge-AI 2026, ACM 10.1145/3609510.3609815 —
  **surveys and indices, not read in depth.** Listed honestly; nothing in the
  architecture depends on them.

---

## Molloy, "From TinyML to Tiny Language Models: State of Edge AI in 2026" — read in full

DCU, July 2026. A survey I had twice listed as unread. It added one axis I was
missing entirely and two guards I should have had.

### The power ladder — an axis this engine did not have

> *"Power predicts almost everything else: cost, memory, and what class of model
> fits. Knowing your power budget usually tells you your model class before any
> benchmarking happens."*

| rung | power | LLM capability |
|---|---|---|
| plain MCU | 1–50 mW | **none** — "be suspicious of anyone claiming otherwise" |
| MCU + microNPU | 50–500 mW | none |
| Linux SBC, CPU only | 3–8 W | ~1B at usable-but-slow rates |
| SBC + accelerator | 5–12 W | 2–4B at ~10 tok/s **given on-module DRAM** |
| phone class | 5–15 W | 4B multimodal at conversational speed |

The engine indexed everything on TOPS and bandwidth with **no power axis at
all** — a real omission for a submission about on-device AI, since thermal
envelope is what separates a phone from a laptop from an IoT board.
`power_rung()` now maps a budget to a rung and an LLM class.

### Two guards, both of which apply directly

> *"**TOPS is a capacity, not a speed.** A 40 TOPS INT4 figure is not comparable
> to a 13 TOPS INT8 one, and it says nothing about whether your model's
> operators, shapes and memory traffic can keep the arrays fed."*

A **fifth** independent confirmation of the roofline (after Vec-LUT, T-MAC,
ENERZAi, SAIL), and it adds something new: cross-precision TOPS figures are not
the same currency, and vendors quote whichever is larger.
`tops_comparable()` now **refuses** to return a ratio when precisions differ.

> *"An NPU accelerates the operators it implements, and a model containing
> anything else falls back to the CPU — 'does my model's operator set map onto
> this accelerator' is the **first real question** of any edge deployment."*

That is exactly the QNN-has-no-ternary-matmul finding, stated as a general law
rather than a Qualcomm quirk.

### A correction to my own honesty

> *"Datasheet TOPS-per-watt and real system power differ by everything else on
> the board... The only defensible numbers come from measuring whole-system
> energy per useful inference."*

`CascadeConfig.energy_mj` contains **invented figures** — 18 and 55 mJ per 1k
tokens. Plausible placeholders, never measured, and I had not said so. Now
documented in the docstring: the *ratio* between tiers is defensible, the
absolute mJ figures **must not appear in a submission** without a power-analyser
measurement behind them.

### Axes still unmodelled, named honestly

- **Federated learning.** A whole training-side axis absent from this work.
  Cross-device FL is practical from Pi-class upward; the layered pattern is
  sleepy sensors → mains-powered hub tier federates on the fleet's behalf.
  Secure aggregation and differential privacy are "standard equipment in
  production FL, not academic dressing." Flower is the de facto framework.
- **On-device personalisation via LoRA adapters** — local fine-tuning without
  federation, the pattern behind modern keyboards.
- **Two of the five cloud "taxes"** my cascade ignores: bandwidth/metered cost,
  and **resilience** ("a device that needs the cloud to think stops thinking
  when the network fails, which is precisely when monitoring systems matter
  most"). My `CascadeConfig` scores latency, energy and privacy only.

### Still read at abstract level only

Raschka's LLM research lists (2025 ×2, 2026 part 1), largo.dev frontier
architectures 2026, Stanford HAI AI Index 2026 R&D, ACM 10.1145/3609510.3609815,
HybridGen (arXiv:2604.18529). Surveys, indices and round-ups. Stated plainly
rather than implied otherwise.

---

## largo.dev, "2026 Frontier LLM Architectures" — read in full; the most consequential of the batch

### mHC: DeepSeek arrived at my principle independently, for a different reason

**Manifold-Constrained Hyper-Connections** (arXiv:2512.24880, 31 Dec 2025,
co-authored by DeepSeek founder Liang Wenfeng, expected to underpin V4).
Unconstrained Hyper-Connections amplified signals **over 3000×** at 27B
parameters and diverged. mHC constrains residual mixing to the **Birkhoff
polytope** (doubly stochastic matrices) via **Sinkhorn-Knopp**, bringing
amplification to **1.6×** for 6.7% training overhead.

That is precisely *quantise the chart, not the manifold* — constrain the
parameter to a manifold so a pathology cannot occur, enforced structurally by a
projection. DeepSeek reached it for **training stability**; `generator_quant.py`
reached it for **quantisation**. Independent convergence on the same idea is the
best evidence the framework is real.

**And it generates a live prediction nobody has tested.** Doubly stochastic
implies spectral norm exactly 1 — that bound *is* mHC's guarantee. Quantising
such a matrix directly walks it off the polytope:

| bits | direct deviation | chart deviation |
|---|---|---|
| 8 | 2.967e-03 | 2.220e-16 |
| 4 | 5.880e-02 | 4.441e-16 |
| 3 | 1.919e-01 | 2.220e-16 |
| 2 | 7.717e-01 | 1.173e-02 |

> **An mHC model quantised naively forfeits the stability property it was
> designed around. Quantising on the Sinkhorn chart preserves it.**

mHC is weeks old and nobody has quantised it for edge deployment yet. Cheap to
test against the reference implementation. Birkhoff is now the **fourth**
manifold in `CHART_TABLE`, with `sinkhorn()` and `quantise_birkhoff()`.

### iRoPE: half of Llama 4's layers accept an unrestricted fold

Llama 4 alternates RoPE layers with **NoPE** layers carrying no positional bias
at all. `sigil/npu.py` guards folds with `_commutes_with_rope` — but **that
problem does not exist in a NoPE layer.** With no rotary embedding applied, any
orthogonal R folds exactly.

So on an iRoPE model roughly **50% of layers** can take a dense rotation with no
block-diagonal constraint. A folding pass applying the RoPE restriction
uniformly hobbles half the network for nothing. `irope_foldable_layers()`
computes the split.

### MLA is a KV axis I do not model

DeepSeek's Multi-head Latent Attention compresses the KV cache into a
lower-dimensional latent space — "LoRA for attention: down-project, store,
up-project." It attacks exactly the bottleneck my roofline identifies, but
**structurally rather than numerically**. My `kv_stream_bytes()` models bits,
layer subsetting and eviction; **latent rank is a fourth KV axis and it is
absent.** Noted, not implemented.

Consistent with my operator-coverage law: *"MLA offers the best memory
efficiency but requires custom kernels"* — the same trade the ternary-on-Hexagon
situation presents.

### MoE residency, validated

Activation ratios: DeepSeek V3.2 **5.4%**, Llama 4 Maverick **4.3%**, Qwen3
**9.4%**. My earlier fix — footprint uses TOTAL parameters because all experts
must be resident, while compute follows active — is exactly right, and these
ratios show how extreme the gap has become. Also noted: frontier models place
**3 dense layers before MoE** for routing stability.

### Still abstract-level only

Raschka's three LLM research lists, Stanford HAI AI Index 2026 R&D,
ACM 10.1145/3609510.3609815, HybridGen (arXiv:2604.18529). Surveys and indices.
Stated plainly.

---

## MLA gap closed — latent rank is the fourth KV axis, and the right rank is hardware-specific

Researched properly (TransMLA NeurIPS 2025, MHA2MLA ACL 2025, GQLA
arXiv:2605.15250, CARE arXiv:2603.17946, DeepSeek V2/V3).

### What MLA is, in the terms this project uses

Three KV axes were modelled here: bit width, layer subsetting, eviction — all
**numerical or selective**. MLA is **structural**: keys and values are
down-projected to a low-rank latent, only the latent is cached, an
up-projection restores expressivity. It caches **one latent per token per
layer, not per head** — canonical `(h_q, d_h, r_kv, d_h^R) = (128, 128, 512, 64)`.

Three facts that change how it should be used:

- **MLA is strictly more expressive than GQA at equal KV overhead.** "GQA can
  always be represented by MLA... but the converse does not hold" (TransMLA).
- **It is not gated on retraining.** TransMLA converts pretrained GQA
  checkpoints (Llama, Qwen, Gemma, Mistral) post hoc: **93% KV compression on
  LLaMA-2-7B, ~10× speedup at 8K context, ~6B tokens to recover.**
- **MLA and KV quantisation are substitutes, not just complements.** MHA2MLA
  reports compression "greater than or equal to Int2 quantization while also
  achieving performance higher than Int2." Multiplying their savings naively
  overstates the result; `mla_vs_gqa()` says so in its return value.

Also: MLA already has PSDC's phase split inside it — MHA-like expansion for
prefill, MQA-absorb for decode.

### The derived result

GQLA's roofline study finds MLA's absorb path at **~242 FLOPs/byte**, "just
below the H100 BF16 ridge (~295)" — near-ideal — then notes this "is, however,
**MLA's only operating point**." The canonical rank is tuned to one ridge.

Snapdragon's ridges are elsewhere, and **the two backends on the same chip
disagree**:

| target | ridge (FLOPs/byte) | MLA at 242 | prescription |
|---|---|---|---|
| H100 BF16 | ~295 | near-ideal | canonical rank 512 |
| **X2 Plus NPU int8** | **421** | bandwidth-bound | **drop rank → 294** |
| **X2 Plus CPU + LUT** | **21** | compute-bound | **keep 512; lowering buys nothing** |

On the NPU the H100-tuned rank leaves compute idle. On the CPU+LUT path the same
model is compute-bound and the up-projection, not the cache, is the cost.

**For PSDC this is a genuine open design question, not a knob**: prefill on the
NPU wants a lower rank than decode on CPU+LUT, while MLA exposes rank as a
single shared parameter.

### All four axes composing

8B-class, 32k context:

| configuration | KV | reduction |
|---|---|---|
| GQA fp16 baseline | 4.00 GiB | — |
| + MLA rank 512 | 1.12 GiB | 3.56× |
| + rank tuned to X2 NPU (294) | 0.70 GiB | 5.72× |
| + INT8 on the latent | 0.35 GiB | 11.4× |

`mla_kv_bytes()`, `mla_vs_gqa()`, `mla_optimal_rank()`, `MLA_CANONICAL`. 60/60.

**Caveat kept in the code:** the rank-to-arithmetic-intensity relation is
modelled as linear (first-order), and the ridge values inherit
`lut_arith_efficiency`'s calibration uncertainty. A direction, not a setting.

---

## Field report: first real-machine run (Windows 10, Intel Haswell, 8 GiB)

The user ran the full suite on a 4-core Haswell laptop with 7.93 GiB RAM under
PowerShell. All suites passed — 94/94, 60/60, 47/47, 43/43 — and two real
defects surfaced that no amount of sandbox testing had exposed.

### 1. `verify` caught an error in my own catalogue

```
[MISSING] grootn15  aihub_id  grootn15   <-- claimed verified but is MISSING
```

I had marked GR00T-N1.5 `id_verified=True`. The AI Hub slug **404s**. The
"claimed verified but is MISSING" flag exists precisely to catch that class of
error, and its first real catch was mine. Corrected: `id_verified` removed, and
the reason field now records the 404 and says to confirm on aihub.qualcomm.com.

That single line is the best argument for shipping `verify` at all. Without it
a judge clicking that link would have found a dead model in a submission that
claims to have checked its own references.

### 2. ANSI escapes rendered as literal garbage

Output came through as `←[1m  SIGIL-EDGE` — PowerShell 5.1 and cmd.exe do not
interpret ANSI escapes unless virtual-terminal processing is enabled, and
`sys.stdout.isatty()` returns True regardless, so my colour guard passed when it
should not have. `_enable_windows_vt()` now switches VT on via
`SetConsoleMode`, and colour is disabled outright if that fails. Unix behaviour
is unchanged.

A pure-usability bug, invisible in a Linux sandbox, and it made every output in
the project hard to read on the platform the user actually has.

### 3. Identifier coverage improved

The run reported `NO-ID: 8`. Four were fillable and are now in:
`Cactus-Compute/needle2`, `tiiuae/Falcon-E-3B-Instruct`,
`openbmb/MiniCPM-o-4_5`, `openbmb/MiniCPM-S-1B-sft` — all marked `?` pending
`verify`, not asserted.

The remaining four are correct as NO-ID: `opti-1.7b` is unreleased, and
`gpt-6-astra` / `claude-opus` / `gemini-3-pro` are closed-weight.

**Verification tally from the real run: 62 OK, 1 MISSING, 1 GATED/BLOCKED,
8 NO-ID.** Now 98/98 checks and 4 legitimate NO-ID.

### Host note

The machine reported **913 MiB available of 7.93 GiB** — the engine's warning
that models above ~3B at INT4 will not fit alongside the OS was correct and
useful. Benchmarking belongs on Colab, as `QUICKSTART.md` says.

---

## Field report 2: live AI Hub session — five corrections

The user ran `qai-hub list-devices`, `qai-hub-models info gemma_4_e2b_it` and
`qai-hub-models export --help` against the real service. Every correction below
comes from that output, not from documentation.

### 1. Snapdragon X2 Plus is NOT on AI Hub

The compute tier offers exactly three devices: **X2 Elite CRD, X Elite CRD,
X Plus 8-Core CRD**. There is no X2 Plus. This project has used X2 Plus as its
primary worked example throughout — its 152 GB/s and 80 TOPS figures anchor the
roofline — and **those numbers cannot be validated on AI Hub.**

`SoC.aihub_device` now records the verified `--device` string, or `None` where
no cloud device exists. 12 of 18 SoCs are profileable. X2 Plus's notes tell you
to profile X2 Elite and state the substitution rather than quietly claiming
X2 Plus numbers.

### 2. The export command I emitted was wrong

I generated:

```
qai-hub-models export <id> --target-runtime qnn_context_binary --quantize_full_type w4a16
```

`qai-hub-models export --help` says plainly: *"export's options come from the
recipe itself, so they can only be listed for a specific target."* **Flags are
per-model.** And runtime support is per-model too — see below. That command was
invented, not verified, and would have failed.

Replaced with the real workflow: `list-devices` → `info` (read Supported
Runtimes) → `perf` / `numerics` / `fetch` → `export --help` before `export`.

> **WITHDRAWN 2026-09-22.** `fetch` downloads a compiled model and `export`
> loads the source model locally before anything reaches the cloud: both put
> gigabytes on the machine that runs them, which a user then watched happen.
> Every AI Hub measurement now goes through `aihub_workbench.py`, which
> uploads one layer and downloads only profile JSON. See the 2026-09-22
> section at the end of this file.

### 3. Gemma-4-E2B-it has NO QNN path

Supported Runtimes: **GenieX (Llama.cpp) only.** Precision q4_0, chipsets
"Universal". The `qnn_context_binary` flag I emitted does not apply to it at all.

### 4. My size estimate was 34% low

AI Hub reports **3.04 GB** at Q4_0. My catalogue said 2.0 GB. Corrected, and the
entry is now `id_verified`.

### 5. Gemma 4 already has shared KV layers

The architecture line reads: *"interleaved sliding-window/global attention with
Grouped Query Attention (GQA), and **shared KV layers**"*, plus Per-Layer
Embeddings and a SigLIP-style ViT.

**Gemma 4 already shares KV across layers.** PSDC's central structural
assumption — that a layer subset can consume KV computed elsewhere — is not
exotic; a shipping Google model does a version of it. That is further
circumstantial support for P1, alongside furiosa's attention correlation and
Hogwild's shared-cache result. It also means PSDC's marginal benefit on Gemma 4
is smaller than on a model without shared KV, which is worth knowing before
choosing a target.

### Also captured

New chipsets in the device list not previously in `SOC_DB`: sm8850
(8 Elite Gen 5), sm7750 (7 Gen 4), qcm6690, qcs9075, qcs8275. Device strings
added for 8 Elite QRD, 8 Elite Gen 5 QRD, QCS8550 Proxy, Dragonwing RB3 Gen 2
(QCS6490), Dragonwing IQ-9075, and the three auto ADPs.

---

## "Can Snapdragon train locally?" — researched and answered

### The FLOP arithmetic

At a generous 400 G-ops/s sustained on an X2-class CPU:

| task | time |
|---|---|
| pretrain 1B from scratch (1T tokens) | **476 years** |
| full finetune 1B (1B tokens) | 174 days |
| LoRA 1B (10M tokens) | 15.3 hours |
| **LoRA 0.5B (1M tokens)** | **46 minutes** |
| **LoRA 0.5B (100k tokens)** | **5 minutes** |
| personalise 45M Needle (10k tokens) | 2 seconds |

### The correction that matters

**"Parameter Efficiency Is Not Memory Efficiency"** (arXiv:2604.22783) and PRGE
(arXiv:2409.15520): PEFT cuts trainable parameters by orders of magnitude and
does almost nothing for the ACTIVATIONS backprop must retain. *"Fine-tuning
Llama 7B requires up to 45.6 GB of on-chip memory for internal activations."*

Same lesson as this project's roofline, arriving from the training side:
**memory binds, not compute.** `adaptation_feasibility()` checks both and names
which one binds.

The escape: **zeroth-order methods** (PRGE) estimate gradients from forward
passes alone. No backward pass, no activation stack, blocker gone — trading the
constraint that binds for the one that does not.

### It already ships

| tool | hardware | status |
|---|---|---|
| llama.cpp `finetune` | CPU, quantised GGUF | shipping; ~16 GB trains 3B |
| **QVAC-fabric-llm.cpp** (Tether AI) | mobile GPUs incl. **Adreno** | shipping, pre-built binaries, Dec 2025 |
| MobileFineTuner | phones, native C++ autodiff | arXiv:2512.08211 |
| Federated LoRA (FedIT/FLoRA/HeLoRA/HetLoRA) | device fleets | production precedent |

Tether reports the **first successful fine-tuning on mobile GPUs including
Adreno** — Snapdragon's own GPU. This is not speculative.

### Verdict, 0.5B on a 16 GB Snapdragon laptop

```
method              time   mem GB     binds  feasible
pretrain        238 years     5.18   compute  no
full-finetune     87 days     5.18   compute  no
lora               46 min     0.59   neither  yes
qlora              46 min     0.59   neither  yes
zeroth-order        1.4 h     0.33   neither  yes
```

**ADAPTATION ONLY — the device can personalise, not train.**

The pitch that is true: *inference on the NPU, adaptation on the idle CPU, and
the user's data never leaves the device.* This project already established that
decode is bandwidth-bound and the NPU does the serving — which leaves the CPU
largely idle. On-device adaptation is the natural use of that idle silicon, and
the one thing a cloud model structurally cannot do.

`on_device_adaptation.py`, 18/18.

---

## facebookresearch/vjepa2 — read at last, and my dismissal was half wrong

I wrote early on that "the JEPA video line does not bear on LLM quantisation,
which is why porting SIGReg across failed." The second half stands: SIGReg
→ quantisation was tested and rejected. The first half was too broad.

### What is actually there

**V-JEPA 2.1 shipped 2026-03-16.** The checkpoint table matters:

| model | params | note |
|---|---|---|
| **V-JEPA 2.1 ViT-B/16 @384** | **80M** | **distilled from ViT-G** — the deployable one |
| V-JEPA 2.1 ViT-L/16 @384 | 300M | feasible on X-series |
| V-JEPA 2 / 2.1 ViT-g | 1B | SOTA motion understanding |
| V-JEPA 2.1 ViT-G | 2B | largest |

MIT licensed, `transformers` `AutoModel` compatible, PyTorch Hub loadable.
V-JEPA 2 results: EK100 **39.7%** vs 27.6% prior best, SSv2 77.3%, Diving48
90.2%. V-JEPA 2-AC does zero-shot Franka manipulation — pick-and-place cup
**80% vs Octo 10%, Cosmos 0%** — from a small amount of robot data.

**An 80M MIT-licensed encoder is straightforwardly NPU-deployable.** Four
entries added to the catalogue.

### The regime distinction this forced, which is the real contribution

Everything else in this engine analyses **autoregressive decode**: one token at
a time, re-reading the full weight set plus a growing KV cache, therefore
**bandwidth-bound**. A ViT encoder is the **opposite case**: one forward pass
over a fixed input, static shapes, no KV cache, fully parallel, therefore
**compute-bound**.

> The roofline, the phase split, the four KV axes and PSDC **do not apply** to
> encoder workloads. They are the workload the Hexagon NPU was actually designed
> for, and they run well on it for exactly the reasons LLM decode does not.

That caveat now ships in the catalogue entries so nobody misapplies this
project's analysis to a vision model. AI Hub already hosts Video-MAE and the
VLA policies (Pi0.5, GR00T-N1.5, ACT); V-JEPA 2-AC is the same family but is
**not** on AI Hub and would need custom export — flagged in its entry.

### Score on my own judgement

Predicting absence: now **0 for 6**. BitCPM-CANN existed, a Hexagon ternary path
existed, the Navier–Stokes paper contributed, §6 was not barren, two "empty"
GitHub profiles carried corrections, and V-JEPA 2 has a deployable 80M encoder.
Every measurement in this project held; every confident absence did not.

---

## Benchmark bug report — three real defects, one of which invalidated the data

The user's Colab run crashed twice and produced results that contradicted
everything else in this project. All three causes were mine.

### Bug 1 (crash): einsum in fold verification

Qwen2.5-0.5B is GQA — **14 query heads, 2 KV heads.** `q_proj` outputs 896,
`k_proj` outputs 128. My `einsum("bthc,bshc->bhts")` requires `h` to match:
14 vs 2, crash. **Fix:** `repeat_interleave` K to the query head count, exactly
as attention does.

### Bug 2 (crash): group=128 with head_dim=64

`view size is not compatible with input tensor's size and stride`. I padded 64
to 128 and reshaped a non-contiguous tensor. A group larger than the axis it
subdivides is not a valid configuration at all. **Fix:** `valid_groups()` skips
them and says so, rather than padding.

### Bug 3 (silent, and the serious one): the sweep measured the wrong thing

Reported results were nonsense — grouping appeared **4× worse** than per-token
(INT4 group-32 ppl 64.60 vs per-token 14.95), contradicting the finding this
whole project leads with.

Cause: the hook reshaped to `(B,S,H,D)` and reduced Keys over `dim=-2` — **the
head axis, size 2 on this GQA model.** The scale was the max of two values:
absurdly fine granularity no deployment would use. Worse, when `group` was set
the axis argument was **ignored entirely** and grouping ran along `D` instead.
So the per-token and grouped rows measured *different schemes* and were never
comparable. The "4× worse" was an artefact of my own code.

**Fix:** `quant_kv()` implements the actual KIVI convention —

| tensor | scale | reduced over |
|---|---|---|
| Keys | per-channel | the **token** axis |
| Values | per-token | the **channel** axis |
| group | subdivides **that same axis** | |

Verified in NumPy: error now falls **monotonically** as groups get finer, for
both K and V at every bit width. That sanity check would have caught the
original bug immediately and was not present.

| K (group along tokens) | int4 | int3 | int2 |
|---|---|---|---|
| per-axis | 0.01279 | 0.07044 | 0.56337 |
| group=32 | 0.00915 | 0.05068 | 0.42352 |
| group=16 | 0.00742 | 0.04142 | 0.34105 |

**Discard the earlier Colab numbers entirely.** They measured a bug.

---

## MAPA (arXiv:2609.13507) — real, useful, and not what it was billed as

Masked autoencoder for **intracranial neural data**, Duke, Apache-2.0. SOTA on
all three Neuroprobe regimes **without fine-tuning**, encoder frozen throughout.

It is a brain-computer-interface model, not an LLM, and not "AGI-level". What it
is: **quantitative evidence for the deployment pattern this project already
recommends** — pretrain an encoder, freeze it, fit a tiny readout.

> **~164 labelled trials to match what takes 3,500 without pretraining — 21×.**

That answers the question `adaptation_feasibility()` could not. The hard part of
on-device personalisation is not whether the CPU can do the arithmetic; it is
whether a user will ever produce enough labelled corrections. MAPA puts the
answer at order 10², not 10⁴ — which normal use produces in days.

`labels_needed_estimate()` returns a **range (116–350)** with confidence marked
**LOW**, because this is a cross-domain analogy from iEEG classification with a
linear readout, not a measurement on text. The 21× figure must not be quoted for
language tasks. It supports deciding to *try*, not promising a result.

MAPA also belongs to the **compute-bound encoder regime** flagged in the V-JEPA
entries: frozen encoder, static shapes, no KV cache. This project's
bandwidth-bound decode analysis does not apply to it.

**Not read:** alinlab.kaist.ac.kr. Stated rather than implied.

---

# BONSAI 2 27B — read 2026-09-19

PrismML released **Ternary-Bonsai-2-27B** on 2026-09-16, three days before this
entry. It is the most directly relevant public artefact this project has found,
because it ships — in production, under Apache-2.0 — two things this project had
only argued for: **rotation folded into stored weights**, and **a stored width
chosen for the wire rather than for the kernel.**

Everything below is derived from byte-exact file sizes returned by the Hugging
Face API (`?blobs=true`) and from `config.json` / `hadamard.json` read directly.
No figure here is taken from prose, including PrismML's own.

## What was verified

| Fact | Source | Value |
|---|---|---|
| Org repo count | `api/models?author=prism-ml` | 36 repos (catalogue had 6) |
| LM parameter count | F16 GGUF ÷ 2 bytes | **26,904,140,464** — pins every bpw below |
| Vision tower params | mmproj BF16 ÷ 2 | 465,572,928 |
| Base model | `config.json` | Qwen3.8-27B (`qwen3_5_text`) |
| Layers / hidden / intermediate | `config.json` | 64 / 5120 / 17408 |
| Attention | `config.json` | 24 Q heads, 4 KV heads (GQA 6), head_dim 256 |
| Hybrid attention | `layer_types` | `full_attention_interval: 4` → full at 3,7,…,63 = **16 full / 48 linear** |
| Linear attention | `config.json` | gated-delta SSM: 16 K heads, 48 V heads, dim 128, conv kernel 4, fp32 state |
| RoPE | `config.json` | `partial_rotary_factor: 0.25`, mRoPE `[11,11,10]`, θ=1e7 |
| Context | `config.json` | 262,144 |
| Quantisation | `config.json` | `bits: 2, group_size: 128, mode: affine` → 2 + 32/128 = **2.25 bpw** |
| Accuracy | model card | 84.78 vs FP16 86.32 over 14 thinking-mode benchmarks = **98.2%** |
| Comparison | model card | IQ2_XXS at 2.8 bpw / 9.4 GB scores 72.59 = 84.1% |

## Three corrections to this project

**1. `bonsai-27b` was wrong by 51%.** *(Corrected 2026-09-22: 3.4 against the
real 3.80 GB is 12%, not 51%. See the fact-check section at the end.)* The
catalogue claimed "~3.4 GB resident —
a 27B-class model that actually fits a phone", with a self-test asserting
`footprint < 3.5 GB`. The measured Q1_0 GGUF is **3.80 GB** (1.131 bpw). The
threshold had been calibrated to the invented number, so the test passed and
protected the error. Replaced with a test that requires the catalogue's own
arithmetic to reproduce the file's byte count.

**2. The iOS claim was unsourced and has been withdrawn.** Earlier notes stated
that PrismML say the ternary build "exceeds the ~6 GB iOS per-app budget" while
the 1-bit companion "fits an iPhone 17 Pro Max". **The model card contains no
statement about iOS, iPhone or mobile memory.** The arithmetic is still useful,
so it is kept in `PHONE_RESIDENCY` — explicitly labelled as SIGIL's own, with
`vendor_claim: None`, and self-tested to stay that way.

**3. A prediction of absence was wrong again — for the seventh time.** The
working assumption was that a new model release would confirm existing findings
and add a catalogue row. It produced a novel result (the container rule below),
corrected two errors, and supplied an independent check on a constant calibrated
from unrelated hardware. The standing pattern holds: **every prediction that a
source would be barren has been wrong, and every measurement has held.**

## The container rule — a new result

PrismML ship the **same 1-bit Bonsai-27B weights in two containers**:

| Container | Bytes | bpw | Tax over the 1.0-bit floor |
|---|---|---|---|
| `Bonsai-27B-Q1_0.gguf` | 3,803,452,480 | **1.131** | +13.1% |
| `Bonsai-27B-mlx-1bit` | 5,129,115,752 | **1.525** | +52.5% |

Identical weights, identical accuracy, **35% more bytes**. Decode is bandwidth
bound, so that is a 35% throughput difference bought by nothing. The stored bit
width — this project's central quantity — is set by the quantiser **and the
container**, and the container half had been invisible.

The model card also publishes token-generation throughput for both *ternary*
containers (PTQ1_0 at 1.768 bpw, PQ2_0 at 2.143 bpw) on ten machines. That is a
free falsification test, and nothing in this engine was fitted to it. The
roofline predicts a **sign**: bandwidth-starved machines should prefer the
narrower stream, compute-starved ones the cheaper unpack.

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

**8 of 8, separating monotonically, crossover bracketed to 1008–1792 GB/s.**

Two things follow. First, where PTQ1_0 wins it realises only **0.36–0.57 of the
ideal 21.2% byte advantage (mean 0.46)**, the rest going to unpack arithmetic —
independently close to `psdc.py`'s `lut_arith_efficiency = 0.60`, which was
calibrated against T-MAC on different hardware and never touched this table.
Second, **every Snapdragon part in `SOC_DB` sits 4.4–19.7× below the bracket**,
so on Snapdragon the narrower container always wins, with no benchmarking
needed. It is the cheapest throughput in the project: nothing is retrained and
no accuracy is spent.

`validate_container_rule()` caught a bug in the predictor while being written —
the bracket edges are measured *wins*, not unknowns, so they are inclusive.
Written with strict inequalities it mispredicted both 1792 GB/s parts.

## Rotation folding, shipped

`hadamard.json` is the rotation contract, and it independently validates
`generator_quant.py` and `sigil/rotation.py`:

- normalized Sylvester–Walsh Hadamard, **block 1024**, applied on the input axis
- ~360 matrices across 64 layers; 1024 fixed ±1 signs shared by every block
- **folded into the stored weights**; the inverse folded into the embedding, so
  activations enter the rotated basis at no runtime cost
- metadata is **297,903 B of an 8.6 GB pack = 0.0035%**
- declared widths 5120 / 6144 / 17408 are exactly 5 / 6 / 17 blocks of 1024

That last line is a **checkable engineering constraint** that had been missed:
the block size must divide every input dimension. Qwen3-4B's hidden size of 2560
is not a multiple of 1024, so Bonsai 2's block does not transfer to it and 512
must be used. `hadamard_block_feasible()` now enforces this before export.

Failure mode worth copying: a stock MLX loader does not know about the basis and
**returns wrong output rather than an error**. PrismML ship a bundled `runtime/`
plus `reload-validation.json` and `tokenizer-validation.json` fixtures. Any
rotated artefact this project emits must declare its rotation and refuse to load
without it.

## The unclaimed win

The vision tower ships **unrotated and unquantised in every container** — the v1
and v2 mmproj files are byte-identical to within 96 bytes across five months.

| Build | Total |
|---|---|
| PTQ1_0 + BF16 tower (shipped) | 6.88 GB |
| PTQ1_0 + Q8_0 tower (shipped) | 6.58 GB |
| PTQ1_0 + ternary tower (**not shipped**) | 6.05 GB |

The tower is **1.70% of parameters but 13.5% of bytes**. Rotating and
ternarising it to the language model's own 1.768 bpw removes 0.83 GB — 12% of
the pack — and it is the component always resident in a document/OCR workload,
which is this project's target.

**Flagged UNMEASURED in code.** MMMU-Pro already falls 81.73 → 75.49 and OCR
Bench v2 60.99 → 56.88 under ternary weights *with an untouched tower*, so the
tower may be carrying the multimodal path. This is a hypothesis with a clear
experiment, not a result.

## Also added

- **dspark speculative-decoding draft**: `Bonsai-27B-dspark-Q4_1.gguf` is
  1,787,468,768 B; the bf16 form is 3.65 B params = **13.6% of the target** — a
  real shipped draft ratio to calibrate `SpecKVConfig` against instead of a guess.
- `prism-ml/bonsai-image-ternary-4B-mlx-2bit` and the gemlite 1-bit sibling: the
  smallest shipped ternary VLMs. Marked `HOST_ONLY` — no GGUF, so no llama.cpp
  route onto a Snapdragon device today.
- `Mintplex-Labs/prism-ml-llama.cpp` — verified fork of `PrismML-Eng/llama.cpp`,
  branch `prism`, MIT, 81★, last pushed 2026-04-03. **Stock llama.cpp will not
  run these files.**

## Naming convention (useful, undocumented)

`{Bonsai|Ternary-Bonsai}-{size}-{gguf|mlx-1bit|mlx-2bit|unpacked|AWQ-4bit}`
where `Bonsai-` is **binary**, `Ternary-Bonsai-` is **ternary**, `-unpacked` is
dequantised fp16 for research and re-quantisation, and `-AWQ-4bit` is the 4-bit
comparison baseline. There is **no** `Bonsai-2-27B`, so Bonsai 2 is ternary-only
and the 1-bit companion remains the v1 build.

---

# T4 BENCHMARK — hardened 2026-09-19

The benchmark had broken twice on Colab. The root cause was structural, not
incidental: **every other module in SIGIL-Edge carries a self-test, and the one
file that runs on hardware the author does not have carried none.**

`--selftest` (117 checks, CPU, no network, <30 s) and `--stress` (9 attention
geometries × 159 quantiser configurations + 13 edge cases, no network) are now
in the file. Every historical crash is a test in them:

| Crash | Test that now catches it |
|---|---|
| `datasets` ≥ 4 rejects bare `"wikitext"` | `wikitext candidates use the namespaced id first` |
| transformers 5 renamed `torch_dtype`→`dtype` | load matrix walked in `--selftest`; exercised on a real tiny model |
| einsum 14 q-heads vs 2 kv-heads | `GQA fold identity holds (crash 2a)` |
| group 128 viewed against head_dim 64 | `group larger than head_dim is dropped` |
| K reduced over the head axis | `K and V use different axes (errors differ)` |

## Defects found and fixed while hardening

| Defect | Consequence |
|---|---|
| `dtype` pinned to fp16 whenever CUDA was seen, while `detect_gpu` computed `supports_bf16` that nobody read | bf16 hardware silently used the narrower-range dtype; the flag was dead code |
| every config read used `getattr(cfg, …)` | **returns `None` on any multimodal model**, where the text tower is under `text_config`. Would have crashed on Bonsai 2, Gemma 3n and Qwen3.5-VL |
| fit check ran *after* `.to(device)` | the OOM it was meant to prevent had already happened |
| `--device cpu` on a CUDA box re-probed and found the GPU | planned the run against VRAM it would not use |
| `pad_token_id=tok.eos_token_id or 0` | `eos_token_id` is a **list** on Qwen3 and Llama 3.1; raises inside the sampler |
| `datasets` required even with `--text-file` | locked out every offline and network-restricted machine |
| corpus shorter than one sequence | empty eval list → `ZeroDivisionError` three frames later |
| resume file had no identity | resuming with a different model silently mixed two runs |
| one OOM in the sweep | every subsequent config OOMs too; now halves the workload and retries once |
| a named model that failed to load | silently substituted Qwen2.5-0.5B — in a `--models` sweep it would benchmark the wrong model under the right name |
| multi-model run where every model failed | returned 0; a scheduled job would record it as a pass |
| em-dash in the comparison table | breaks a Windows cp1252 console, the bug `apply_ascii_patch.py` exists for |

Also added: Apple **MPS** and Intel **XPU** detection, ROCm-safe
`mem_get_info`, `--dtype {auto,fp32,fp16,bf16}`, `--load-in-4bit`, `--force`,
CPU thread tuning, and a non-finite-loss guard.

One test failure during development was a **bug in the test, not the code**:
`_hook_roundtrip` drew fresh random token ids per call, so the no-op check
compared two different inputs. Seeding the input fixed it — and the check is
sharper for it, now asserting exact equality rather than a tolerance.

Integration-verified end to end on CPU against a locally built tiny model:
FP16 baseline → 8-config sweep → fold verification (3.15e-07) → decode →
results table → plot. Resume, fingerprint invalidation, short-corpus fallback,
multi-model isolation and the no-substitution rule all exercised.

**Suite totals after this work: 458 checks across 8 modules, all passing.**

---

# NAVIER-STOKES, SECOND PASS -- 2026-09-19

The paper was read once before and gave up the chart-vs-manifold argument in
`generator_quant.py`. This pass covered Sections 3.4 and 9 -- the correction
cycle -- and the accompanying **Lean certificate repo**, which had never been
opened. Both produced code.

Seventh prediction of absence, seventh time wrong. The working assumption was
that a re-read of an already-mined paper would be confirmatory.

## What the cycle says (Sec. 3.4, Sec. 9, error tables pp. 109-110)

The residual update is

    R(u[j+1]) = R(u[j]) + L_{u[j]}(du_j, dp_j) + div(du_j (x) du_j)

and the whole proof turns on the worst NEWLY CREATED term staying subordinate
to the one removed. Encoded exactly in `residual_cascade.py` as `NS_CYCLE`,
with `validate_against_paper()` asserting that the transcription reproduces the
paper's own published recursion:

    sigma_0 = 1/5,  sigma_{j+1} = sigma_j + 1/10,  B_j = 1/2 + sigma_j,
    C*_j = 1 + sigma_j,  kappa_s = 1e-5

The model recovers two constraints the paper does not state as a pair:

1. **A floor on the starting residual.** The mean-side quadratic
   self-interaction sits at `C* + sigma_j - 2 kappa_s` and clears the next
   level only when `sigma_j > 0.1`. `sigma_0 = 1/5` clears it by exactly 2x.
2. **Per-operation losses**, one `kappa_s` at each divergence and
   reconstruction. At 1e-5 they vanish and the cycle runs forever.

A subtlety the code caught: at `sigma_0` the term that BINDS is the transverse
cross term (1.37999), while the term that sets the FLOOR is the
self-interaction (1.39998), which only becomes binding below sigma = 0.18.
Scanning only the binding term reported no floor at all.

## Why it is the same object as PTQ refinement

For one output row with scale `s`, grid position `u = w/s` and rounding `r`,
the calibration objective is `J(r) = (r-u)^T G (r-u)`. Flipping coordinate `i`
by `delta` changes it by exactly

    dJ = 2*delta*(G(r-u))_i  +  delta^2 * G_ii

-- a linear remainder through the current background plus the correction's
quadratic self-interaction. Term for term, the paper's update.

## Prediction, and falsification

**Predicted** from constraint 1: bit width sets the initial residual, so there
should be a bit width below which refinement stops converging.

**Falsified.** Rounding coordinate descent converges at 8/6/5/4/3/2 bits alike,
and the fraction of gain surviving on held-out data is flat in bit width
(76.0 / 74.8 / 78.6 / 76.9 / 75.5% at n/d = 8). The reason is the useful half:
**the paper must BOUND its quadratic term because it cannot evaluate it;
coordinate descent evaluates it exactly and rejects flips that do not pay.**
Constraint 1 is an artefact of needing a bound, not a fact about correction
cycles.

## What did transfer: a measured law

Constraint 2 transfers and dominates. The paper's `kappa_s` is 1e-5; its PTQ
counterpart is finite-calibration-set loss, and it is enormous. With `d` free
rounding decisions per row and `n` calibration tokens:

| n/d | 0.5 | 1.0 | 2.0 | 4.0 | 8.0 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| calibration gain | 2.00x | 1.56x | 1.42x | 1.36x | 1.35x | 1.31x | 1.32x | 1.33x |
| **held-out gain** | **0.77x** | 0.96x | 1.10x | 1.21x | 1.27x | 1.27x | 1.30x | 1.32x |
| kept | -23.5% | -7.9% | 23.6% | 57.0% | 77.3% | 89.1% | 94.6% | 96.8% |

    kept ~= 1 - 1.75 * d/n        n >= 1.75*d/(1-k) tokens for a keep of k

Two controls make it a law in the RATIO rather than a curve in n:

- vary d at fixed n/d = 8: d = 32/64/128/256 gives kept 75.2/78.0/76.8/75.7%
- vary bits at fixed n/d = 8: 8/6/4/3/2 bits gives 76.0/74.8/78.6/76.9/75.5%

An independent re-run reproduced kept = 57.6/78.4/88.9% at n/d = 4/8/16 against
the model's 56.2/78.1/89.1%. The zero-crossing near n/d = 1 sits in run-to-run
noise (0.980x, 0.956x, 1.017x across three draws), so `BREAK_EVEN_RATIO` is set
slightly above it at 1.2. Below n/d = 0.5 the harm is unambiguous: **0.77x
held-out against a 2.0x calibration "improvement".**

**Consequence.** For Bonsai 2 27B (hidden 5120, intermediate 17408), a standard
128 x 2048 = 262,144-token calibration set keeps ~96% on the attention
projections and ~88% on `down_proj`. A 128 x 512 = 65,536-token set -- also
common -- keeps **54%** on `down_proj`. Half the reported improvement there is
calibration overfitting, and `down_proj` is always the binding matrix because
it alone sees the intermediate dimension.

## The Lean repo: a verification protocol worth stealing

`openai/NavierStokesAndEuler` (Apache-2.0, Lean 4.34.0-rc2 + Mathlib) ships the
formalisations and, in `ComparatorChallenges/`, the protocol. An independent
party runs `lake exe comparator ComparatorChallenges/NavierStokes.json` against
`lean4export` (neutral export), `nanoda_bin` (an INDEPENDENT reimplementation
of the Lean kernel) and `landrun` (sandbox). **Nothing in that chain trusts the
toolchain that produced the artefact.**

Note: the GitHub contents API was 403 from this network, so the directory was
read through the rendered README rather than listed file by file.

That protocol is exactly what a quantised model lacks, and the gap has a named
owner. PrismML's own PACK-RUNTIME.md for Bonsai 2: *"Ordinary MLX loaders do
not apply the required transforms"* -- a loader that does not know about the
Hadamard basis returns **wrong output rather than an error**. Their mitigation
is honest about its reach: *"Reload validation confirms serialization integrity
but doesn't assess model quality or cross-platform equivalence."*

`quant_certificate.py` implements the pattern: a few-kilobyte JSON declaring
the scheme (including the group AXIS), the transform (with a digest of the sign
vector), probes generated from seeds, and a tolerance derived from the declared
dtype and reduction length rather than guessed.

**Detection holds: 7/7 injected faults refused, correct pipeline passes, at
four problem sizes and seeds** -- with the scheme declared correctly, so the
probes work unaided.

**Diagnosis does not.** Naming the fault is right 7/7 on the harness the
signature table was measured from and 4/7 to 6/7 elsewhere; the closest
distinct signature pairs sit 0.53 apart in log space, which a change of size
reorders. `diagnose()` therefore returns a ranked hint, never a verdict.

Two findings from building it:

- **WRONG_AXIS and TRANSPOSED_SCALE are the same fault.** Transposed per-group
  scale layout IS the group applied along the other axis; measured signatures
  identical to five decimals. Recorded in `INDISTINGUISHABLE`.
- **The first verifier passed a pipeline that negated every Hadamard sign.**
  l2, std and absmax are all invariant under negation, so the statistical
  fallback saw a perfect match -- the exact fault the catalogue warns about,
  committed inside the code written to catch it. `array_stats()["proj"]`, a
  projection onto a fixed pseudo-random +/-1 vector, exists because of it.

`WRONG_AXIS` is in the catalogue for a reason closer to home: this project's
own `sigil_t4_benchmark.py` reduced Keys over the head axis instead of the
token axis, published a sweep comparing two different schemes against each
other, and produced a headline ("grouping is 4x worse") that was an artefact.
A probe check would have caught it in milliseconds.

**Suite totals after this work: 567 checks across 10 modules, all passing.**


---

# DREAM-RSI, AND A HARDENING PASS -- 2026-09-19

## `zhengkid/Dream-RSI` -- read

"Recursive Self-Improvement through Evolving Worlds" (Google, Google DeepMind,
University of Maryland, University of Virginia). The idea: treat accumulated
discovery history as a **replay simulator over the realized search space**, and
evaluate candidate policies by dreaming over that pool instead of paying for
new rollouts. Reported: 1.74x less discovery compute and 162x fewer calls on
algorithm engineering; 4 of 4 kernels improved at 2.09x higher performance at
equal budget on GPU kernel engineering.

Code is listed as being prepared for release, so `dream_search.py` implements
the METHOD, not their implementation. The paper PDF and project page are linked
from the repo; the GitHub contents API was 403 from this network, so the README
was read through the rendered page.

**Why it lands here.** `sigil_t4_benchmark.py` sweeps 18 configurations at
roughly 90 minutes a run, and has been run more than once. Every run writes
`results_t4.json` -- a complete record over the realized search space. That file
IS a replay simulator, and the question "would a cheaper sweep have found the
same answer?" was being answered by running the sweep again.

**Two real pools ship with the module**, both labelled at every point of use:

| Pool | What it is | Why it is caveated |
|---|---|---|
| `COLAB_2026_09` | the user's own T4 run, 18 configs, 3 crashes, spread 2994x | **Pre-fix benchmark, wrong Key axis.** Valid as a search landscape, invalid as quantisation guidance. |
| `TINY_CPU` | 18 configs on a randomly initialised model, correct code | Meaningless objective: spread 1.1x. The degenerate case. |

**Measured, on the Colab pool:** exhaustive 18 evaluations, coordinate 6
(3.0x fewer calls), greedy 4 (4.5x). All reach the same answer.

**Three things the implementation had to get right, and did not at first:**

1. *Every policy exhausted the space.* Each structured policy ended with a
   fallback loop over whatever was left, so all five used 18 evaluations and
   "fewer calls" was 1.0x across the board. A search policy that cannot stop is
   a sweep with extra steps. Fixed: they stop when their plan completes.
2. *"Fewer calls" divided time-to-best instead of total calls,* reporting 1.0x
   for a policy that used 4 evaluations against a sweep that used 18.
3. *Both pools are degenerate, and the tool now says so before showing a
   table.* COLAB_2026_09's optimum is the FIRST configuration in the natural
   order, so every policy starting there "finds it in 1" and the ranking
   measures ordering. TINY_CPU is nearly flat, so stopping early costs nothing.
   `pool_is_degenerate()` reports both. A comparison that looks decisive and is
   not is worse than no comparison.

**The honest limit is the design.** Replay is valid only inside the realized
search space. A policy requesting an unevaluated configuration records a MISS,
`reliable` goes False, and `compare_policies()` reports it but never ranks it.
Interpolating over the gaps would produce confident numbers about
configurations nobody ran -- the same failure as reporting calibration error
and calling it accuracy.

## `stress_all.py` -- a hardening pass, and what it found

Six phases: imports, every module suite, adversarial fuzzing of every public
callable, numeric sanity, determinism, and portability plus a CLI smoke test.
No GPU, no network, ~14 seconds.

The classification that makes it useful: `ValueError`/`TypeError`/`KeyError`
are a function correctly refusing input. `AttributeError`/`IndexError`/
`ZeroDivisionError`/`RecursionError` are a function discovering its own
confusion halfway through. **Worst of all is none of those** -- a function that
takes nonsense and returns a plausible number.

**It found 17 defects on the first run, and hung on the first one.**

| Defect | Why it mattered |
|---|---|
| `qat_schedule(10**30)` never returned | `2 ** (bits - 1)` on unvalidated input is a denial of service: Python builds an integer with 10**30 bits. **Nine sites across seven files had the same pattern**; all now use a validated `_qmax()`. |
| `sigma_schedule(10**30)` never returned | `range(10**30)`. Capped. |
| NaN in, NaN out, silently | `compression_pipeline`, `storage_comparison`, `deployed_loss`, `degeneracy_ratio`, `training_verdict`, `array_stats`, `sequential_quantisation_gain`, `group_metadata_bits`, `power_rung`, `ternary_viability`, `effective_bits_per_weight`. All now reject non-finite input via `_finite()`. |
| `predict_container(0)` | ZeroDivisionError. |
| `array_stats(1e308)` | Every element finite, the norm overflows to inf. Now refused with that explanation. |
| Confused `AttributeError`s | `inverse_cayley`, `hutchinson_trace`, `markdown_table`, `probe_input`, `cpu_lut_ceiling`, `crossover_bits`, `select_ternary_kernel`, `compare_policies`, `pool_is_degenerate`, `resolve_model_key`, `outlier_strength`. All now raise a typed refusal. |
| Box-drawing characters in `snapdragon_engine.py` | Replaced with ASCII, so `apply_ascii_patch.py` is no longer needed for a clean PowerShell render. |

**The harness hung on its own first run** and reported nothing, because
`qat_schedule(10**30)` took it down with it. It now bounds every call with a
5-second watchdog and reports a hang as a failure. It also found zero callables
in `snapdragon_engine` -- the largest module in the project -- because it
trusted `__all__`, which there lists only classes.

**One finding is not a bug but a limit worth recording:** `derive_tolerance`
returns 2.0 for bf16 accumulation over a 4096-length reduction. A tolerance
above 0.5 accepts anything, so that combination **cannot be certified at all**.
The function now says so with `certifiable: False` rather than issuing a number
that always passes.

## The deck

Regenerated from `build_deck.py`, 12 slides. The previous deck predated the
container rule, the calibration law and the certificates, and its title slide
claimed results were "verified on real Snapdragon silicon" -- which is not
true, and is now corrected.

Three rendering defects were caught by looking at the output rather than
trusting the code: the container-rule chart had four colliding label pairs and
unlabelled Snapdragon markers (redesigned as one row per device); two headings
overflowed onto a second line and landed on the subtitle (a `MAX_HEADING`
assertion now refuses them at build time, and it caught a third immediately);
and the ASCII sweep had turned the multiplication dot in `1.75 * d/n` into a
hyphen, making the slide read as a subtraction.

Chart colours were validated rather than chosen: the deck's existing green and
red separate by only dE 8.1 under colour-vision deficiency, right on the floor.
The blue/amber pair used instead separates by 27.6 under protanopia.

**Suite totals after this work: 705 self-tests across 12 modules, plus 180
adversarial checks in `stress_all.py`. All passing.**


---

# FINAL AUDIT PASS -- 2026-09-19

A deliberate pass over everything before submission, rather than a repackage.
Five findings, three of them defects I had introduced.

## A regression I caused and the harness could not see

`sigil/quant.py` was **broken**. The hardening patch inserted a `_finite()`
guard that calls `math.isfinite` into a file that does not import `math`, so
every `IntRTN` construction raised `NameError`. `run_local_validation.py`
died on line 126 and nothing noticed, because the `sigil/` package carries no
self-test and was in none of the harness's phases.

The same patch also rewrote `NFCodebook`'s `k = 2 ** bits` as
`int(_qmax(bits,1,24)+1)*2`, which agrees for bits >= 2 and returns **4 instead
of 2** at bits = 1. Both fixed; the original semantics are restored with the
exponent validated separately.

`stress_all.py` now imports and fuzzes `sigil.quant`, `sigil.rotation`,
`sigil.npu`, `sigil.allocate` and `sigil.gof`, and compiles the four scripts
that have no `selftest` subcommand. **207 checks, up from 180.**
`run_local_validation.py` now completes: 295.9s, fold exactness 4.44e-14 and
1.22e-15, Hexagon deployability PASS on all six criteria.

## `audit_claims.py` -- new, and the point of the exercise

Every number in `SUBMISSION.md`, `MANIFEST.md`, the deck and this file is now
re-derived from the code by a script: **58 claims, all verified.**

It exists because of a specific failure. The catalogue claimed a 27B model at
1-bit was "~3.4 GB, fits a phone" -- wrong by 51% -- and the self-test guarding
it asserted `footprint < 3.5 GB`, a threshold that had been calibrated to the
invented number. The test passed and protected the error. A claim is only
checked when something independent recomputes it from the source of truth.

## The deck: three more rendering defects

Caught by rendering all twelve slides and looking at them, not by reading code.

| Defect | Fix |
|---|---|
| The container card's byte figures wrapped, pushing the second row out of its panel | One text box per cell, plus `assert_fits()` -- a character budget from a calibrated characters-per-inch figure |
| Three cards had text hanging below their panels | `assert_card_fits()` estimates wrapped line count against panel height; it fired on the first rebuild and caught a fourth |
| Slide 12 still read "180 adversarial checks", slide 9 still read "Seven predictions" | Corrected to 207 and Eight |

Earlier in the same session the ASCII sweep had turned the multiplication dot
in `1.75 * d/n` into a hyphen, making a slide read as a subtraction. All four
are the same class: correct code, wrong output, invisible without looking.

## `run_all.ps1` verified string by string

Every marker it greps for was checked to exist in the file it greps, every
script it invokes was checked to exist, and every pattern it matches on was
checked against the real command output. All pass. `audit_claims.py` is now
wired in as its own section.

## The brief, regenerated

`SIGIL_Edge_Brief_Description.docx` was still the three-component version from
before the container rule. Rebuilt from `build_brief.js` to six components with
the 8-of-8 table, rendered and checked at three pages.

## Standing totals as of 2026-09-19

**705 self-tests across 12 modules. 222 adversarial checks. 58 published claims
re-derived from code. All passing.** (Superseded 2026-09-20 -- see the totals at
the end of this file. The harness grew by one check and the audit by 23 when
both were hardened; the self-test total is unchanged.)

Eight predictions that a source would be barren; eight wrong. Three of this
pass's five findings were defects I introduced while fixing other defects,
which is the argument for the harness rather than against it.


---

# THE HP SNAPDRAGON TARGET -- 2026-09-19

Prompted by a direct question: does any of this actually run on Qualcomm AI Hub
Workbench? The challenge requires a solution "designed, developed, or intended
to be optimised for Snapdragon-powered HP PCs", so the question deserved a
literal answer rather than a gesture. Three findings, all verified today.

## Finding 1 -- AI Hub publishes NO Compute performance numbers

> **WITHDRAWN 2026-09-22.** The website quote below is accurate; the
> conclusion drawn from it is not. Qualcomm's measurements ship inside the
> `qai_hub_models` package: 491 entries for X Elite CRD, 487 for X2 Elite CRD,
> none for X Plus 8-Core CRD. See "The ninth prediction of absence" at the
> end of this file.

Checked across an LLM and a vision model alike: `qwen3_4b`, `gemma_4_e2b_it`,
`yolov8_det`. Every one lists the three X-series CRDs under "Supported Compute
Devices" and then states, verbatim:

> "This model is currently not supported on any Compute chipset."

with an empty performance table and a pointer to "View for other chipsets".

**So no entrant can cite published AI Hub figures for Snapdragon X-series.**
The published numbers that exist are for mobile chipsets, and presenting those
as laptop numbers is the fastest way to lose a judge who clicks the link.

The more plausible reading is that the *performance table* has no Compute rows
rather than that the silicon cannot run the model -- the CRDs are listed as
supported and a Windows deployment command is given
(`geniex infer ai-hub-models/<model>`). Both readings lead to the same action:
submit a job, because there is no published figure to quote either way.

## Finding 2 -- one HP configuration cannot be profiled at all

| HP machine | SoC | RAM | AI Hub device |
|---|---|---|---|
| OmniBook 3 14-HZ000 | Snapdragon X | 8 GB | X Plus 8-Core CRD (inexact) |
| OmniBook 5 16-bf000 | Snapdragon X | 16 GB | X Plus 8-Core CRD (inexact) |
| **OmniBook Ultra 14-kg000** | **Snapdragon X2 Plus** | 16 GB | **NONE** |
| OmniBook Ultra 14 | Snapdragon X2 Elite | 16/32/64 GB | X2 Elite CRD |
| ProBook 4 G1q 14 | Snapdragon X | 16/32 GB | X Plus 8-Core CRD (inexact) |
| EliteBook 6 G1q 14 | Snapdragon X Elite | 32 GB | X Elite CRD |
| EliteBook Ultra G1q8 14 | Snapdragon X Plus | 16 GB | X Plus 8-Core CRD |

Sources: hp.com/us-en/shop/cv/snapdragon-x-series and the OmniBook Ultra 14
(2026) review, which gives the flagship as **X2E-90-100, 18 cores to 5.0 GHz,
Hexagon NPU quoted at 85 TOPS** against Qualcomm's 80 TOPS platform figure, with
LPDDR5x-9523 and a 64 GB SKU announced but not yet on sale.

`aihub_device_for()` returns `None` for the X2 Plus and refuses substitution in
its own note. The earlier finding that X2 Plus is absent from AI Hub now has a
consequence with a name: one of the seven HP machines has no cloud proxy.

**The 8 GB OmniBook 3 is the machine the architecture has to respect**, not the
64 GB flagship. After Windows there is roughly 4-5 GB usable, which holds a 4B
model at INT4 and does not hold an 8B one.

## Finding 3 -- the NPU answer has two halves, and only saying one is a trap

  * **Prefill on the Hexagon NPU.** INT4 weights, static shapes, GenieX
    (QAIRT-backed) or ONNX Runtime QNN. Compute-bound over a whole prompt,
    which is what the NPU is for.
  * **Decode on the ARM CPU.** Bandwidth-bound at batch 1, and **stock QNN has
    no ternary matmul** -- ENERZAi showed a custom Hexagon kernel is possible
    and that the stock path does not exist. This is where the container rule
    applies.

Claiming the whole pipeline runs on the NPU is the claim that gets taken apart
under questioning. Claiming the NPU is unusable is equally wrong.

## What was built

`hp_snapdragon.py` (51 tests): the seven machines, the three devices, the
mapping with gaps named, per-machine memory arithmetic, the four evidence tiers
ranked by strength, the phase split, and a `submit` command that really
dispatches a profile job -- dry-run by default, because it spends the user's
quota.

Its most useful property needs no API token: **every AI Hub call it makes is
asserted against the installed `qai-hub` client's real signatures at test
time.** Verified against qai-hub 0.55.0, whose surface was read from the
installed package rather than from memory: `get_devices(name, os, attributes)`,
`submit_profile_job(model, device, name, options, retry, project)`,
`submit_compile_job(...)`, `submit_compile_and_profile_jobs(...)` and
`Device(name, os, attributes)`. An API change now breaks a test here instead of
breaking a job the night before a deadline.

`qai-hub-models chipsets` and `devices` could not be run from this environment
-- the asset host `qaihub-public-assets.s3.us-west-2.amazonaws.com` is blocked
by the egress proxy -- so the device strings come from the AI Hub model pages
instead. They will resolve on the user's laptop.

## The bug a real user found and I could not: the harness audited their venv

**2026-09-20.** A run on Windows, Python 3.13.9, `D:\sigil`, reported
`passed: 38 failed: 1` with one unreadable line:

```
FAILURESortability     54614/5628620
```

Two separate defects are stacked in that one line, and neither was reproducible
in a clean checkout.

**The garbling.** `run_all.ps1` split captured output on `` "`n" ``, which on
Windows leaves a trailing `\r` on every line, then interpolated the resulting
array into a string -- which joins with spaces. The carriage returns then rewound
the console cursor and each line overwrote the one before it. Reconstructing:
writing `  [FAIL] portability   546/562`, then `\r`, then `  FAILURES` yields
`  FAILURESortability   546/562` -- character for character. Fixed with an
`InfoLines` helper that splits on `\r?\n`, trims, and prints line by line;
it replaced five garbling call sites.

**What was actually failing.** Once the line was decoded it read roughly
`8620` checks. The harness's portability phase did `here.rglob("*.py")`. The
QUICKSTART tells people to create the virtualenv as `.venv` **inside** the
project -- which is exactly what they did -- so the scan walked
`D:\sigil\.venv\Lib\site-packages` and audited numpy, torch and setuptools
for tabs and non-ASCII. Reproduced by planting 2,850 third-party files in a
copy of the project:

| | .py files scanned | portability checks | false failures |
|---|---|---|---|
| before | 2,874 | 8,622 | 5,700 |
| after | 24 | 72 | 0 |

`8,622` against their `8620`. The fix prunes hidden directories by rule and
vendor directories by name, and adds a tripwire -- *the scan stayed inside the
project* -- that fails loudly if the file count is ever in the hundreds, so an
unanticipated vendor tree can never silently poison the numbers again. Verified
by planting a directory the deny-list does not name: the tripwire fired.

This is the most useful defect in the project so far, because **no amount of
testing in a clean checkout could have found it.** The harness was correct
about every file it was given; it was wrong about which files were ours.

## Four more defects found while fixing that one

**2026-09-20.**

**1. Interpreter startup noise counted as our crash.** The CLI smoke check was
`returncode == 0 and "Traceback" not in stdout+stderr`. On an interpreter with
a broken `distutils-precedence.pth`, every subprocess prints a full traceback
during startup, before our code runs. All 23 CLI checks failed on a machine
where all 23 commands succeeded. A traceback now counts against us only if one
of its frames names a file inside this directory. Verified both ways: startup
noise is ignored, our own traceback is still caught, and a mixture is caught.

**2. A missing optional dependency reported as a failure, twice.** `torch` is a
~2 GB install and `sigil_t4_benchmark.py --selftest` correctly refuses to run
without it, exiting 2 with a clear message. Both `stress_all.py` and
`audit_claims.py` treated that as a failure, so a machine without torch saw
`[FAIL] cli 0/23` and `documented total is 705 self-tests, got 588`
(705 - 117 = 588, which named the cause exactly). Both now recognise the skip
**from the message the module prints**, not from the exit code alone -- verified
that exit 2 with any other message, or that message from any other script, is
still a failure. `audit_claims.py` counts a skipped suite at its documented size
and prints a **NOT VERIFIED HERE** block naming what was not re-measured, so the
total is never quietly circular. This is the third instance of the same class,
after `hp_snapdragon.py` reporting 51 tests on one machine and 46 on another.

**3. A check that could only pass.** Hardening `_compiles` to return
`(ok, why)` left one older call site doing `ok = _compiles(...)`. A non-empty
tuple is truthy, so that check passed unconditionally from then on -- a
silent always-green test, which is worse than a failing one and is precisely
what this harness exists to catch. Found by grepping every call site after
changing the signature, not by the harness.

**4. An invalid escape sequence that will become a SyntaxError.**
`residual_cascade.py` draws the correction cycle's under-braces with
`\_ ... _/` in its module docstring. Python 3.12 made that a `SyntaxWarning`
and 3.15 makes it a `SyntaxError`, so the file imports today and stops
importing on a future interpreter. The docstring is now raw. The harness
missed it because `compile()` raises only on `SyntaxError`; it now treats a
`SyntaxWarning` or `DeprecationWarning` at compile time as a failure, and
that guard was verified by re-introducing the defect and watching it fail.

**Portability matrix now verified**, under real PowerShell 7.4.6 rather than by
reading the script:

| | Python 3.11 + torch | Python 3.13, no torch |
|---|---|---|
| `sigil/` present | 39 passed, 0 failed | 38 passed, 0 failed |
| `sigil/` absent | 38 passed, 0 failed | 38 passed, 0 failed |
| `.venv` inside the project | -- | 38 passed, 0 failed |

Stress harness: **223/223** with `sigil/`, **192/192** without, identical on
both interpreters. The count no longer depends on the machine, which was the
whole point.

## Standing totals

**705 self-tests across 12 modules. 223 adversarial checks. 86 published claims
re-derived from code. All passing.** (Superseded 2026-09-22 -- see the totals
at the end of this file.)

Still not done, and stated in the submission: no AI Hub job has been submitted
from here, because there is no API token in this environment. The tooling is
signature-checked and runnable in one command.


---

# ZERO DOWNLOAD, THE SCALING BOOK, AND QUALCOMM'S OWN DATA -- 2026-09-22

Two prompts. The user pointed at *How to Scale Your Model*
(jax-ml.github.io/scaling-book) and asked for it to be used, not quoted. And
the user reported that this project's scripts downloaded models from AI Hub
onto their laptop -- more than 1 GB each -- when all that was wanted was a
measurement. Both are answered below, with every defect found on the way.

## The report: this project put gigabytes on a laptop

The user was right, and the commands that did it were ours:

- `snapdragon_engine.py export` emitted `qai-hub-models fetch` and
  `qai-hub-models export`. The first downloads a compiled model; the second
  loads the source model locally before compiling. This file recommended both
  (Field report 2, now marked WITHDRAWN).
- GenieX, llama.cpp and Hugging Face commands were emitted with no size
  attached, from the engine, `hp_snapdragon.py` and `sigil/npu.py`.
- `sigil_t4_benchmark.py`, written for a Colab T4, downloaded its model on
  whatever machine ran it -- ~8 GB for Qwen3-4B.

What changed:

1. **Banned outright:** `qai-hub-models fetch`, `export`, `evaluate` and `demo`
   (the last two added today: both load the source model locally). Nothing in
   the project needs them.
2. **Allowed only with a stated size:** `geniex pull`, `geniex infer`,
   `huggingface-cli download`, `hf download`. Running a model ON the HP laptop
   needs its weights there, so these survive -- but the string that emits one
   must carry `[downloads <size> to THIS machine -- run it on the target HP
   laptop, not on a dev box]` in the same template, so the size sits beside
   what gets pasted.
3. **Guarded:** the benchmark and the engine's `bench` refuse a model download
   over 1 GB unless the path is local, the machine is a hosted notebook
   (Colab, Kaggle, SageMaker, Paperspace, Lightning), or `--allow-download` is
   given. A size that cannot be estimated counts as large. Refusal is exit
   code 6 with the reason; `--dry-run` now reports the same verdict in
   advance, sized from `config.json` exactly as the real run sizes it.
4. **Enforced:** the stress harness's seventh phase reads the syntax tree of
   every project file, renders every command the project emits for every
   catalogue model, and -- added today -- reads every document, paragraph by
   paragraph. A banned command anywhere fails the build; a sized command
   without its size fails the build. A deliberate exception needs
   `# download-ok: <reason>` on the line, and a waiver without a reason is not
   honoured. This file is the one document not scanned: it is the dated record
   and quotes withdrawn commands verbatim.

The scanner reads syntax, not text, because a text search trips over the
docstrings that explain why a command was removed. An f-string is judged as
one template: its literal pieces are split around each placeholder, and
judging them separately let a size marker in one piece excuse a command in
another.

## The Workbench runner: designed from the get-started page

aihub.qualcomm.com/get-started#workbench describes the client flow: configure
an API token, list devices, submit compile, profile and inference jobs, and
download results. `qai_hub_models` builds its LLM exports from the same
calls. `aihub_workbench.py` uses that sequence and removes the download:

- **What goes up:** one decoder layer at the target model's real dimensions --
  for Qwen3-4B, 100.9M weights: attention with Q/K/V/O projections over a
  4K-token KV cache, RoPE, and the gated MLP. Weights are drawn on a
  power-of-two-scaled 16-level grid, which DEFLATE compresses 4.8x (Gaussian
  weights compress 1.1x), so the 403.7 MB layer travels as 83.8 MB. The
  calibration set is 33.6 MB, dominated by the KV cache.
- **The job sequence, mirroring `qai_hub_models`:** fp16 is compile to
  `qnn_dlc`, then profile. w8a16 and w4a16 are compile to ONNX, quantise with
  `--range_scheme min_max`, compile to `qnn_dlc` with `--quantize_io`, then
  profile. Ten jobs for one device.
- **What comes back:** `download_profile()` dicts, a few KB each, counted by a
  ledger with a hard 2 MB ceiling. The wrapper every backend passes through
  has no download method to call.
- **The default device is X Plus 8-Core CRD** -- the one Qualcomm has not
  measured, standing in for four of the seven HP machines.

How it is verified without a token: a mock backend whose method signatures
are checked against the installed `qai-hub` 0.55.0 client parameter by
parameter; the uploaded archive is checked against the client's own
`_determine_model_type` classifier; and the ONNX graph is checked against a
NumPy reference with onnxruntime (relative error below 1e-7). Defects found while
building it, all fixed:

- the mock's logging helper took a `name` argument that collided with the
  client's own `name=` keyword;
- `analyze` raised `KeyError: 'tiny'` on the test architecture;
- the first self-scan was a text grep and tripped on its own docstrings,
  which is why the scanner reads syntax;
- the mock's download alarm is exercised by a real `.download()` call in the
  self-test, now an explicit, reasoned waiver;
- a size marker and its command split across f-string placeholders passed the
  first scanner; templates are now judged whole;
- the bandwidth in the bytes test was computed as `1e-3 / slope`, 1000x off:
  x is in MB and y in ms, so MB/ms is GB/s and the bandwidth is `1 / slope`.
  A planted 50 GB/s series now pins it;
- a run without a token printed the client's traceback; it now prints the
  three commands that fix it, and exits 2. `devices` still printed the raw
  error until today;
- `verify` without onnxruntime crashed; it now skips with the install line.

## The scaling book, checked before it was trusted

`roofline.py book` recomputes 17 worked answers from the book: TPU v5e and
H100 critical intensities (240, 295), B_crit for bf16, int8-weight and
int8-everything cases (240, 120, 240) and in its beta-alpha form, matmul
intensity at small batch, LLaMA-2 13B KV at 8K tokens (6.7 GB), the Part 7
worked problem's parameter count, KV sizes and step times (2.5 ms, 21 ms),
MHA KV per token and cache size, the attention share of parameters, and MHA
decode attention intensity. All 17 reproduce.

Only then is it applied, to data this project did not produce:
`qai_hub_models` 0.62.2 ships Qualcomm's measurements in `models/<id>/perf.yaml`
inside the pip package. 90 X-series LLM rows are embedded in `roofline.py` and
checked against the package whenever it is installed. Results, all in
`FINDINGS.md` and all re-derived by `audit_claims.py`: decode linear in bytes
(R² ≥ 0.99 on X Elite QAIRT); prefill/decode as the critical batch (62.8-70.6
on the X2 Elite NPU, 13.7-17.1 on its CPU); the engine barely mattering for
decode; the KV crossover at 0.65-1.13x the formula; software alone moving
Qwen3-4B's decode 3.23x on X Elite.

**The first snapshot was typed by hand, and 66 of its 90 rows were wrong** in
the fourth or fifth significant figure. The drift check against the package
caught it on its first run. The table is now generated from the package and
spliced in; the check stays, so a hand edit cannot come back quietly.

## The ninth prediction of absence

On 2026-09-19 this file, `hp_snapdragon.py`, the submission, the brief and the
deck all said Qualcomm publishes no Compute performance numbers, from model
pages showing empty tables. The package data has **491 measured entries for X
Elite CRD, 487 for X2 Elite CRD, and 0 for X Plus 8-Core CRD** (listed as
supported for 221 models). Why the pages and the data disagree is not known
here; the data is what gets cited.

The correction is not only a retraction. It locates the real gap exactly:
X Plus 8-Core is the proxy for the OmniBook 3, OmniBook 5, ProBook 4 and
EliteBook Ultra G1q8, and it is the one device with no numbers. That is what
the Workbench runner now targets by default, and what `roofline.py predict`
makes a written, falsifiable prediction about.

Nine predictions that a source would be barren. Nine wrong.

## Three guesses about Qualcomm's data, overruled by the data

- "Exactly one published pair is anomalous." There are two: on X Elite
  (Genie), Qwen3-4B moves 111% more bytes than Qwen3-1.7B in 7.9% LESS time;
  on X2 Elite (QAIRT), Qwen3-8B moves 77% more bytes than Qwen3-4B in 7.8% more
  time, a marginal 612 GB/s above any peak.
- "The 8B is the outlier on X2 Elite." Leave-one-out removes the 4B at 4K
  context instead: dropping it gives the tightest fit.
- "Measured KV crossovers sit at 0.9-1.5x the formula." They sit at
  0.65-1.13x.

Neither anomaly is explained. Both are flagged by `roofline.py anomalies` and
left out of fits.

## Ten defects from the fuzzer

The fuzzer found `attention_decode_intensity(1e308)` returning infinity in the
new roofline code; that formula and its neighbours are now bounded. Widened
from 14 hostile values at a budget of 60 pairs to all 29 values and every one
of the 841 pairs, it found nine more in older modules:

| function | input | failure |
|---|---|---|
| `snapdragon_engine.effective_bits_per_weight` | huge values | overflow to inf |
| `psdc.mla_kv_bytes` | huge / NaN | NaN or inf |
| `qat_optimizer.deployed_loss` | huge values | overflow to inf |
| `slt_compressibility.degeneracy_ratio` | huge values | overflow to inf |
| `slt_compressibility.singular_mdl_redundancy` | huge values | overflow to inf |
| `residual_cascade.kept_fraction` | `(1, nan)` | NaN |
| `residual_cascade.shrinking_support_schedule` | `(1, 0)` | ZeroDivisionError |
| `quant_certificate.diagnose` | a string first | AttributeError |
| `dream_search` policies | a string first | AttributeError |

All fixed, and the fuzz runs clean at budgets of 250, 900 and 1000.

## Found while bringing the documents up to date

- `hp_snapdragon.py evidence` still recommended "say plainly that tier 3 does
  not exist" three lines below a tier 3 marked available; its tier 2 still
  pointed at X Elite and X2 Elite rather than X Plus 8-Core; `aihub` printed
  "Checked 2026-09-22" beside a website quote taken on 2026-09-19; and "the
  website page lagged the data" was an inference stated as fact. All four
  corrected.
- The README's chart-quantisation table had two cells that did not match
  their column headers: the Birkhoff "direct @ 4 bits" cell held the 2-bit
  deviation (0.77; the 4-bit value is 5.88e-2), and the SPD "chart @ 2 bits"
  cell held the 4-bit eigenvalue (+1.8e-2; the 2-bit value is +1.4e-2).
- README and QUICKSTART still quoted 297 tests, 76/76, 117/117 and a
  53-model catalogue.
- The stress harness's own comment said the widened fuzz found eight defects
  while listing nine.
- Documents were not scanned for download commands. `SUBMISSION.md` carried a
  `geniex infer` with no size; the new document scan caught it on its first
  run.
- Building the Qwen3-4B layer peaked at 1.6 GB of RAM, because the ONNX
  helpers copied every weight tensor twice more on the way into the model. The
  weights now go in last, one tensor at a time: peak 1.04 GB, and the archive
  is byte-identical (same SHA-256), since protobuf serialises fields in
  field-number order. On an 8 GB laptop that is the difference between
  building and paging.
- The harness walked every folder under the project except a deny-list, and
  the previous RUNBOOK told users to copy their old scripts into
  `D:\sigil\backup_old`. With the network phase in place, a backed-up engine
  that still emitted the old download commands would have failed a clean
  install. The walker is now an allowlist -- the top level and `sigil/`,
  nothing else -- and planting exactly that folder confirmed it is ignored.

## An independent fact-check of the pitch

Before delivery, a separate agent that had not seen the work being produced
checked `SUBMISSION.md`, the README, the deck text and the brief against the
tools' own output. It confirmed the headline figures (491/487/0; R² 0.9929
with intercepts of -1.95 to +4.06 ms and 43-51% of peak; 62.8-70.6 and
13.7-17.1; 33.7/33.5/36.2 and 461 -> 2307; 35%/11%; 8 of 8; 4.4-19.7x; 84
MB, 33.6 MB and the 2 MB ceiling; ~21.22 tok/s; 872 tests in 14 modules) and
found the problems below. Each was checked here before being fixed:

| claim as published | what the tools say | now |
|---|---|---|
| fold error "3.15e-07 on real GQA weights" | measured on a random-init test model, in fp32 (the 2026-09-19 integration test above) | "exact to ~1e-15 in float64; 3.15e-07 end to end in fp32 on a random-init test model"; real weights are the Colab run's job |
| calibration law "measured on real weights" | `torch.randn` matrices, d_in = 128; the Bonsai 2 figure of 54% is computed from the law | "measured on synthetic matrices"; 54% is "the law predicts" |
| throughput "for both containers on ten machines" | ten machines; the two Apple rows have no PTQ1_0 figure | "on ten machines, for both containers on eight" |
| `bonsai-27b` "wrong by 51%" | 3.4 against 3.80 GB is 12% | "12% under the real file" |
| chart label "Snapdragon, 5 parts" | nine parts at six bandwidths; 67 GB/s was missing | "9 parts", 67 GB/s plotted |
| "the engine barely matters for decode" | true on X2 Elite with each engine on its best runtime; on X Elite, llama.cpp's NPU path trails its CPU path 2.3x | scoped to X2 Elite, with "the software does" beside it |
| X Elite "straight line in bytes" | holds for the QAIRT series | "under Qualcomm's QAIRT runtime" |
| KV crossover 9,770 beside "left out of fits" | that fit keeps the 4K point leave-one-out flags; without it, ~12,700 | both quoted; "left out of the cross-model fits" |
| certificates "in milliseconds, on device" | synthetic pipeline on a development machine | "~0.5 ms per check on a laptop CPU, with nothing a device lacks" (timed today) |
| "4.5x fewer calls" (brief, README) | on two pools the tool flags as degenerate | caveat added |
| "re-derives 124 published numbers" | 124 checks, of which some are test counts and document checks | "a 128-check claim audit that re-derives the quoted numbers" |
| title "Tested on AI Hub Workbench"; deck "Measured on Workbench"; brief "we fill that gap" | no job has run from here | "a zero-download runner, built to fill that gap"; "no job has been run from here yet" |

The first five figures had no audit claim, which is how they survived; each
now has one. The lesson is the one this file keeps recording: a number is
only checked when something independent recomputes it.

## Standing totals as of 2026-09-22

**872 self-tests across 14 modules. 292 adversarial checks (255 without the
optional `sigil/` folder). A 128-check claim audit. All passing, on Python
3.9 through 3.13.**

Verified from the delivery zip itself, extracted fresh each time, under real
PowerShell 7.4.6 running `run_all.ps1`:

| environment | `sigil/` present | `sigil/` absent |
|---|---|---|
| Python 3.11 with torch, onnx, qai-hub; a stale `backup_old\` planted inside | 49 passed, 0 failed; stress 292/292 | 48 passed, 0 failed; stress 255/255 |
| Python 3.13, no torch, no onnx, no qai-hub | 48 passed, 0 failed; stress 292/292 | 47 passed, 0 failed; stress 255/255 |
| Python 3.12 in a fresh `.venv` inside the project, numpy + scipy only | 49 passed, 0 failed; stress 292/292 | -- |
| Python 3.9 and 3.10, numpy + scipy | stress 292/292; audit 128/128 on 3.9 | -- |

The planted `backup_old\` held a script that emits a banned command and
contains a tab; the harness no longer walks outside what ships, so it was
ignored, as it should be.

Still not done, and stated everywhere it matters: no job from this project
has run on AI Hub Workbench, because there is no API token in the build
environment. The runner is one command on a machine that has one.
