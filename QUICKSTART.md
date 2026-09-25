# QUICKSTART — exactly what to run, where, in what order

Three places. Your laptop does nothing heavy, and **nothing here downloads a
model onto it.**

| Where | What runs there | Time |
|---|---|---|
| **Your laptop** | Every self-test, the roofline on Qualcomm's own data, the Workbench runner | ~20 min |
| **Google Colab (free T4)** | `sigil_t4_benchmark.py` — the only script that fetches model weights | ~90 min |
| **Qualcomm AI Hub Workbench** (cloud) | Profiles on real Snapdragon X devices, driven from your laptop | ~15–30 min |

---

# STEP 0 — Put the files somewhere (5 min, laptop)

Unzip `SIGIL-Edge.zip` somewhere simple and **keep the `sigil\` subfolder**:

- **Windows:** `D:\sigil` or `C:\sigil`
- **macOS / Linux:** `~/sigil`

Open a terminal in that folder and create a virtual environment inside it:

```powershell
cd D:\sigil
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # 3.9 or newer
```

On macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`.

> Throughout this guide, if `python` doesn't work, use `python3`.

---

# STEP 1 — Prove it works (2 min, NO internet needed)

```powershell
python snapdragon_engine.py selftest
```

**Expect:** `141/141 checks passed`. The engine needs no pip install at all.

On Windows, `.\run_all.ps1` runs every suite, the stress harness and the claim
audit in one go. If PowerShell refuses to run it:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`.

---

# STEP 2 — Look at your own machine (1 min)

```powershell
python snapdragon_engine.py probe
```

**Expect:** your CPU, cores, RAM, and `No Snapdragon SoC detected — this is a
development host`. That is correct: Step 8 measures real Snapdragon devices in
the cloud.

---

# STEP 3 — Qualcomm's own measurements (3 min, no download)

Qualcomm ships its measured X Elite and X2 Elite numbers inside the
`qai_hub_models` pip package. `roofline.py` carries the 90 X-series LLM rows,
so you do not even need that package installed.

```powershell
python roofline.py book        # the scaling book's worked answers: 17/17
python roofline.py law         # decode time vs bytes moved: R^2 >= 0.99
python roofline.py engines     # CPU / GPU / NPU: same decode, 5x prefill
python roofline.py bcrit       # prefill/decode = the critical batch
python roofline.py context     # where KV traffic overtakes the weights
python roofline.py predict     # what X Plus 8-Core should show -- Step 8 tests it
python roofline.py anomalies   # two of Qualcomm's pairs that cannot be physical
```

**`law` is the one to read.** It is the evidence that decode on Snapdragon X
is set by the memory bus, on Qualcomm's own data.

---

# STEP 4 — The views that matter (3 min)

```powershell
python hp_snapdragon.py machines
python hp_snapdragon.py evidence
python snapdragon_engine.py models
python snapdragon_engine.py lowbit
python snapdragon_engine.py capability --model gemma-4-e2b-it
```

- `machines` — the seven HP Snapdragon PCs, their RAM and their AI Hub device.
- `evidence` — what counts as proof, ranked, and the NPU/CPU split.
- `models` — 61 models. `+` means the ID was checked against a live registry,
  `?` means it wasn't.
- `lowbit` — the verified position on ternary-on-Hexagon, with sources.

`python snapdragon_engine.py capability --model gpt-astra` returns BLOCKED
with the reason: the engine telling the truth about a request that cannot be
satisfied.

---

# STEP 5 — Check every link before you publish (3 min, NEEDS INTERNET)

```powershell
python snapdragon_engine.py --json verify.json verify
```

**Expect:** `OK` / `MISSING` / `GATED/BLOCKED` lines and a summary. Any
**`MISSING`** line is a 404 — fix or delete that entry before you push.
`GATED/BLOCKED` is fine. Note the order: `--json` comes **before** `verify`.

---

# STEP 6 — Run the science (10 min)

```powershell
pip install numpy scipy
python psdc.py falsify
python psdc.py selftest
python generator_quant.py
python qat_optimizer.py
python slt_compressibility.py
python run_local_validation.py
```

**Expect:** `60/60`, `47/47`, `43/43` and `22/22 passed`, then a table of
numbers from the last one. **`psdc.py falsify` is the important one:** it
prints the four experiments that would *disprove* the architecture.

---

# STEP 7 — The benchmark run (90 min, on Colab — not your laptop)

This is the only script in the project that fetches model weights, so it
belongs on a hosted notebook. **On a laptop it refuses any model over 1 GB**
and tells you why (exit code 6). `--dry-run` says in advance what the real run
would do on the machine you are on.

## Option A: Google Colab (free T4, recommended)

1. **colab.research.google.com** → **New notebook**
2. **Runtime → Change runtime type → T4 GPU → Save** ← *do not skip this*
3. Upload `sigil_t4_benchmark.py` with the folder icon in the left sidebar
4. In a cell:

```
!pip install -q -U transformers datasets accelerate sentencepiece
!python sigil_t4_benchmark.py --selftest
```

   **Always run `--selftest` first.** Under 30 seconds, no GPU, no network. If
   it prints `123/123 checks passed`, the long job will not die on a shape bug
   an hour in. Then:

```
!python sigil_t4_benchmark.py
```

5. Walk away for ~90 minutes. Results are saved after **every** config, so a
   disconnect loses nothing — re-run and it resumes.
6. Download `results_t4.json` and `kv_sweep.png` from the left sidebar.

## Option B: your own GPU workstation

Only where you have the disk and the bandwidth. The script probes the
accelerator, the dtype and the attention geometry, then sizes the run to what
it finds — T4, A100, Apple Silicon, Intel XPU or CPU. Because this is not a
hosted notebook, the download needs your explicit consent:

```powershell
pip install -U torch transformers datasets accelerate sentencepiece
python sigil_t4_benchmark.py --selftest
python sigil_t4_benchmark.py --dry-run          # says what would be fetched
python sigil_t4_benchmark.py --allow-download   # [downloads ~1-16 GB of weights to THIS machine -- a workstation, not the laptop]
```

If the weights will not fit, the script says so **before** downloading them.

## Useful flags

```
python sigil_t4_benchmark.py --selftest   # 123 checks, no GPU, no network
python sigil_t4_benchmark.py --stress     # 9 attention geometries, no GPU, no network
python sigil_t4_benchmark.py --dry-run    # check setup and the download verdict
python sigil_t4_benchmark.py --quick      # ~15 min smoke run
python sigil_t4_benchmark.py --text-file x.txt   # skip the dataset hub entirely
python sigil_t4_benchmark.py --help
```

**What you get:** FP16 baseline perplexity, an 18-config KV-quantisation
sweep, exact-fold verification on real weights, decode throughput and the KV
memory budget. The script prints the exact numbers to paste into
`SUBMISSION.md`.

---

# STEP 8 — Real Snapdragon numbers, zero download (15–30 min)

Qualcomm has measured X Elite and X2 Elite. It has measured **nothing on X
Plus 8-Core CRD**, the device that stands in for four of the seven HP
machines. This step measures it.

1. Sign in at **workbench.aihub.qualcomm.com** (free) → **Account → Settings
   → API Token**, and copy the token.
2. On your laptop:

```powershell
pip install qai-hub onnx numpy
qai-hub configure --api_token PASTE_YOUR_TOKEN_HERE
qai-hub list-devices                       # confirms the token works
```

3. See exactly what will happen, then do it:

```powershell
python aihub_workbench.py plan             # dry: nothing is sent
python aihub_workbench.py run --yes        # X Plus 8-Core CRD, fp16 / w8a16 / w4a16
python aihub_workbench.py results          # ms per layer, GB/s, the bytes test
```

**What it costs:** it builds one Qwen3-4B decoder layer at real dimensions on
your laptop (~40 s, ~1 GB of RAM at peak, ~500 MB of scratch disk), uploads
it **once** (~84 MB,
plus 34 MB of calibration data), and runs 10 cloud jobs. What comes back is
profile JSON — a few KB per job, with a hard 2 MB ceiling. A re-run reuses
the upload.

Add `--all-devices` to profile X Elite and X2 Elite as well: there,
`results` can compare the one-layer figure with Qualcomm's full-model number.
`python aihub_workbench.py verify` (needs `pip install onnxruntime`) checks
the layer against a NumPy reference before you upload anything.

> **Windows on ARM:** if `pip install qai-hub onnx numpy` fails on a
> dependency, use an x64 (AMD64) Python, which runs under emulation. The
> heavier `qai_hub_models` package is not needed at all — `roofline.py`
> already carries its data.

---

# STEP 9 — Fill in the submission (30 min)

Open `SUBMISSION.md`. Replace each `[FILL]`:

| `[FILL]` | Where it comes from |
|---|---|
| Workbench job IDs, ms per layer, extrapolated tok/s | `python aihub_workbench.py results` |
| FP16 perplexity | `results_t4.json` → `ppl_fp16` |
| KV at INT4 group-32 | `results_t4.json` → `kv_sweep` |
| fold error | `results_t4.json` → `fold_rel_err` |
| decode tok/s | `results_t4.json` → `decode_tok_per_s_T4` |
| KV KiB/token | printed by the script at the end |

**Do not submit with `[FILL]` still in it.** Then:

- **Brief Project Description** → `SIGIL_Edge_Brief_Description.docx`
- **PDF pitch** → `SIGIL_Edge_Pitch.pdf`
- **PPT pitch** → `SIGIL_Edge_Pitch.pptx`
- **"Snapdragon Laptop?"** → **No**, honestly. Say your numbers are
  Qualcomm's own measurements plus your Workbench job IDs.

---

# STEP 10 — Push to GitHub (15 min)

```powershell
"`.venv/`n__pycache__/`n*.pyc`nworkbench_build/`n.aihub_workbench_cache.json" | Out-File .gitignore -Encoding utf8
git init
git add .
git commit -m "SIGIL-Edge"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sigil-edge.git
git push -u origin main
```

Keep `workbench_results.json` in the repo — it holds your job IDs. The build
folder and the cache are local scratch.

**Include `FINDINGS.md` and `PROVENANCE.md`.** They document experiments that
*failed* and claims that were *wrong*, including a withdrawn claim about
Qualcomm's own data. That is a credibility asset in front of judges.

---

# If something breaks

| Symptom | Fix |
|---|---|
| `python: command not found` | Use `python3` |
| `No module named numpy` | `pip install numpy scipy` |
| `aihub_workbench.py` says "not configured with an API token" | Step 8, parts 1–2 |
| `aihub_workbench.py run` says the upload is over budget | You picked a bigger layer: `--max-upload-mb 250`, or the default `--arch qwen3_4b` |
| Benchmark exits with `REFUSED` (code 6) | Working as designed: run it on Colab, or `--allow-download` on a workstation |
| "No CUDA GPU detected" on Colab | Runtime → Change runtime type → **T4 GPU** |
| Colab disconnects mid-run | Re-run — it resumes from `results_t4.json` |
| `verify` shows many GATED/BLOCKED | Normal. Only `MISSING` (404) matters |
| `--json` rejected | Put it **before** the subcommand |

---

# The shortest possible version

```
python snapdragon_engine.py selftest     # 1. proves it works
python roofline.py law                   # 2. the result, on Qualcomm's data
python aihub_workbench.py run --yes      # 3. the first X Plus 8-Core numbers
python snapdragon_engine.py verify       # 4. fix any MISSING before publishing
# 5. run sigil_t4_benchmark.py on a Colab T4 (90 min)
# 6. paste the numbers into SUBMISSION.md, upload the docx/pdf/pptx, push to GitHub
```
