# SIGIL-Edge — complete deliverable set

**Qualcomm Snapdragon AI Lab Build & Present Challenge.**
Apache-2.0. 872 self-tests across 14 modules, plus 292
adversarial checks in one stress harness (255 if the optional
`sigil/` subfolder is not extracted -- it is reported, not failed). All passing.
Last verified 2026-09-22.

**Nothing in this project downloads a model onto your laptop.** Every AI Hub
measurement runs through `aihub_workbench.py`, which brings back a few KB of
profile JSON and refuses anything over 2 MB. The stress harness reads the syntax
tree of every file, every command the project emits and every document that
tells you what to type, and fails the build if a download comes back.

---

## Start here

```powershell
cd D:\sigil
.\.venv\Scripts\Activate.ps1
.\run_all.ps1
```

`run_all.ps1` checks your files, runs every suite, the stress harness and the
claim audit, and tells you what to do next. Safe to run repeatedly.

If you only run a few commands, run these:

```powershell
python .\roofline.py law                              # Qualcomm's own X-series data vs the roofline
python .\roofline.py bcrit                            # prefill/decode = the critical batch, measured
python .\aihub_workbench.py plan                      # the zero-download Workbench job, dry
python .\snapdragon_engine.py container --validate    # the container rule, 8/8
python .\stress_all.py                                # tries to break everything
python .\audit_claims.py                              # every number vs the code
```

To measure on Qualcomm's real devices (needs a free API token; uploads an
84 MB layer and 34 MB of calibration data once, downloads only JSON):

```powershell
# token: workbench.aihub.qualcomm.com -> Account -> Settings -> API Token
pip install qai-hub onnx numpy
qai-hub configure --api_token PASTE_YOUR_TOKEN_HERE
python .\aihub_workbench.py run --yes                 # X Plus 8-Core CRD by default
python .\aihub_workbench.py results
```

---

## The fourteen modules, plus the two harnesses

| File | Tests | What it is | Try |
|---|---|---|---|
| `snapdragon_engine.py` | 141 | Hardware probe, 61-model catalogue, capability matrix, deployment planner, zero-download AI Hub commands, identifier verification. The spine. | `probe`, `models`, `container --validate`, `export --model qwen3-4b` |
| `sigil_t4_benchmark.py` | 123 | Real-weights KV-quantisation benchmark **for Colab**. Refuses a >1 GB model download on a machine that is not a hosted notebook unless `--allow-download`. | `--selftest`, then on Colab the full job |
| `dream_search.py` | 87 | Replay-simulator search over recorded benchmark history, after Dream-RSI. | `pools`, `compare`, `dream` |
| `roofline.py` | 83 | **The scaling-book roofline**, validated against the book's own 17 worked answers, then tested against Qualcomm's measured X Elite / X2 Elite data. Nothing downloaded. | `book`, `law`, `bcrit`, `context`, `engines`, `predict` |
| `aihub_workbench.py` | 64 | **Zero-download testing on AI Hub Workbench.** Builds a decoder layer at real dimensions locally, uploads it once, compiles/quantises/profiles in the cloud, fetches only profile JSON. | `plan`, `build`, `verify`, `run --yes`, `results` |
| `psdc.py` | 60 | Phase-Split Depth Cascade: roofline, SoC budgets, the P1–P4 falsification suite, MLA latent-rank analysis. | `plan`, `crossover`, `falsify` |
| `quant_certificate.py` | 58 | Certificates that prove a runtime decodes your model the way you quantised it. Catches 7/7 injected faults in ~3 KB. | `faults`, `experiment`, `demo` |
| `hp_snapdragon.py` | 56 | The seven HP Snapdragon PCs, each mapped to its silicon, RAM budget and AI Hub device, with the gaps named. | `machines`, `aihub`, `evidence`, `commands` |
| `residual_cascade.py` | 49 | When another round of PTQ refinement is worth running. The calibration law `kept = 1 − 1.75·d/n`. | `cycle`, `law`, `plan`, `experiment` |
| `generator_quant.py` | 47 | Quantise the chart, not the manifold. SO(d)/Cayley, SPD, Stiefel, Birkhoff. 12.9× storage. | run it |
| `qat_optimizer.py` | 43 | Training-side optimisation: quantisation damage, group-max-norm regulariser, deployed-loss selection. | run it |
| `on_device_adaptation.py` | 23 | Can Snapdragon train locally? Activation memory, not FLOPs, is the blocker. | run it |
| `slt_compressibility.py` | 22 | Singular-learning-theory compressibility. Documents an estimator failure honestly. | run it |
| `aihub_bench.py` | 16 | Qualcomm's **published** numbers per model -- a few KB of metadata, no weights. | `--models ...` |
| `stress_all.py` | 292 | One command that tries to break everything, in seven phases including **network**: nothing may download a model -- no file, no emitted command, no document. | run it, or `--quick` |
| `audit_claims.py` | 128 | Re-derives the quoted numbers from the code, and checks the documents quote the counts the code reports. | run it before quoting anything |

Plus the `sigil/` package (`npu.py`, `allocate.py`, `quant.py`, `rotation.py`,
`gof.py`) used by `run_local_validation.py` and `p1_test.py`.

---

## Documents

| File | What it is |
|---|---|
| `SUBMISSION.md` | The submission pack. Paste-ready form text, measured results, judging-criteria map, and an explicit **what NOT to claim** list. |
| `PROVENANCE.md` | **The authoritative record.** Every source with its status, every correction dated, including corrections to our own published numbers. |
| `FINDINGS.md` | What holds and what is falsified, with the numbers. |
| `RESEARCH_AGENDA.md` | Open questions and the experiments that would close them. |
| `QUICKSTART.md` | Setup, Colab, and per-machine guidance. |
| `RUNBOOK.md` | Operational steps end to end. |
| `README.md` | Repository front page. |
| `MANIFEST.md` | This file. |
| `LICENSE` | Apache-2.0, full text. |
| `SIGIL_Edge_Pitch.pptx` / `.pdf` / `SIGIL_Edge_Brief_Description.docx` | Slide deck and brief. Regenerate with `python build_deck.py` and `node build_brief.js`; the deck builder refuses to write text outside its panels. |

---

## The results worth defending

**1. Decode on Snapdragon X is bound by the memory bus -- measured, on
Qualcomm's own data.** `qai_hub_models` ships Qualcomm's measurements inside the pip
package. Across Qwen3 0.6B → 8B on X Elite's NPU (GenieX QAIRT, w4a16, 4K
context), decode time is linear in bytes moved per token: R² ≥ 0.99 under all
six ways of counting the LM head and KV cache, intercept within 5 ms of zero,
43–51% of the 135 GB/s peak. On X2 Elite, with each engine on its best
runtime, the engine barely matters for decode and enormously for prefill:
Qwen3-4B decodes at 33.7 / 33.5 / 36.2 tok/s on CPU / GPU / NPU while its
prefill spans 461 → 2307 tok/s. The software matters a great deal: on X
Elite, QAIRT decodes the same model 3.23× faster than Genie.

**2. Prefill/decode IS the critical batch.** If prefill is compute-bound and
decode bandwidth-bound, their throughput ratio is the scaling book's B_crit,
which cannot depend on model size. On the X2 Elite NPU it is 62.8–70.6 across
three model sizes (CV 0.05); on the CPU of the same chip, 13.7–17.1. Read off
a published table, it says how many draft tokens a speculative verifier checks
for the price of one step on each engine -- and so where verification belongs.

**3. The container rule.** The same 1-bit Bonsai-27B weights ship as 3.80 GB of
GGUF and 5.13 GB of MLX: 35% more bytes per token for nothing. The roofline
predicts which of two ternary containers wins on 8 of 8 published GPU
measurements; every Snapdragon part sits 4.4–19.7× below the crossover.
Corrected for context: the gap is 35% at zero context and 11% at 128K, because
the KV traffic is shared.

**4. Past ~10–13K tokens, KV traffic outweighs weights.** Qwen3-4B on X2 Elite
crosses over at 9,770 tokens (about 12,700 without the 4K point that
leave-one-out flags); measured crossovers sit at 0.65–1.13× the book's
formula with a 16-bit cache. For long Indic documents, KV quantisation is the
lever.

**5. Zero-overhead weight folding.** Exact to ~1e-15 in float64, and
3.15e-07 end to end in fp32 on a random-init test model -- real weights are
the Colab run's job -- with zero added operators. Bonsai 2 ships exactly this
at 0.0035% metadata cost.

**6. The calibration law.** `kept ≈ 1 − 1.75·d/n`, measured on synthetic
matrices. At `n/d = 0.5` refinement improves calibration 2.0× and makes
held-out error *worse* (0.77×).

**7. Certificates.** 7/7 injected faults refused in ~3 KB. A check took
~0.5 ms on a laptop CPU at the test size, with nothing a device lacks.

---

## The answer to "does it run on Qualcomm AI Hub?"

**Qualcomm has measured X Elite and X2 Elite. It has measured nothing on X Plus
8-Core, which stands in for four of the seven HP machines.** The package data
carries 491 measured entries for X Elite CRD, 487 for X2 Elite CRD, and zero
for X Plus 8-Core CRD -- listed as supported for 221 models, measured for none.

**This corrects an earlier claim.** On 2026-09-19 this file said Qualcomm
published no Compute numbers at all, from website pages showing empty tables.
The package data proves otherwise for two of the three devices. PROVENANCE.md
records it as the ninth prediction of absence to fail.

**So the gap is exact, and `aihub_workbench.py` is built to fill it without a
download.**
It profiles an architecture-faithful Qwen3-4B layer on X Plus 8-Core CRD at
fp16, w8a16 and w4a16, following the same job sequence as Qualcomm's own
qai_hub_models pipeline. `roofline.py predict` states in advance what it should
see: the same bus and NPU as X Elite, so the same decode rate.

**One HP configuration cannot be profiled at all.** The OmniBook Ultra 14-kg000
is Snapdragon X2 Plus, and AI Hub offers no X2 Plus device.

## Known gaps, stated plainly

- **No job from this project has run on Workbench yet.** There is no API token
  in the build environment. The runner is verified end to end against a mock
  whose signatures match the installed client parameter by parameter, and the
  exact archive it uploads is accepted by the client's own classifier.
- **One layer is not a model.** Per-token figures from the runner are
  extrapolated and labelled so; `results` calibrates them against Qualcomm's
  full-model numbers on the two devices where both exist.
- **The `[FILL]` markers in `SUBMISSION.md` need your Colab T4 run and your
  Workbench run.**
- **Two of Qualcomm's published pairs are unphysical** (`roofline.py
  anomalies`). They are flagged and left out of the cross-model fits by leave-one-out, not
  explained.
- **The vision-tower win is UNMEASURED.** 0.83 GB is available; the accuracy
  cost is unknown.
- **Certificate diagnosis does not generalise** (4/7–6/7 off its fitted
  harness). Detection does (7/7 everywhere).
- **Snapdragon X2 Plus is not an AI Hub device.** Profile X2 Elite CRD and say
  so.
- **Both replay pools are degenerate** and the tool says so.

---

## Standing pattern

Nine predictions that a source would be barren. Nine wrong -- the latest being
this project's own claim that Qualcomm publishes no X-series numbers. Every
measurement has held; every prediction of absence has not. `PROVENANCE.md`
records each one.
