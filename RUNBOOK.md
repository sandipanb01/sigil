# RUNBOOK — exact PowerShell commands, start to finish

Your terminal is at `D:\sigil` with the venv active. Everything below is
copy-paste; run one block at a time. **Nothing here downloads a model onto
this laptop** — the one script that fetches weights runs on Colab (Step 7).

---

## STEP 1 — Replace your files with this delivery

```powershell
cd D:\sigil
New-Item -ItemType Directory -Force backup_old | Out-Null
Copy-Item *.py, *.md, *.ps1 backup_old -Force -ErrorAction SilentlyContinue
```

Unzip `SIGIL-Edge.zip` over `D:\sigil`, overwriting. **Keep the `sigil\`
subfolder** — copying only the loose `.py` files leaves it out. New in this
delivery:

| file | why |
|---|---|
| `roofline.py` | **new** — the scaling-book roofline, tested on Qualcomm's own X-series measurements |
| `aihub_workbench.py` | **new** — zero-download testing on AI Hub Workbench |
| `snapdragon_engine.py`, `hp_snapdragon.py`, `sigil\npu.py` | no longer emit any command that downloads a model to this machine |
| `sigil_t4_benchmark.py` | refuses a >1 GB model download off Colab unless `--allow-download` |
| `stress_all.py`, `audit_claims.py`, `run_all.ps1` | now police downloads in every file, every emitted command and every document |

---

## STEP 2 — Verify everything in one command

```powershell
.\run_all.ps1
```

If PowerShell refuses to run it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\run_all.ps1
```

**Expect:** every line `[ OK ]`, `failed: 0`, then "Local setup is clean." It
runs all fourteen suites, the stress harness and the claim audit, and checks
that the harness ran as many checks as the documents say.

---

## STEP 3 — Qualcomm's own measurements (no download, no token)

```powershell
python .\roofline.py law     | Tee-Object .\roofline_law.txt
python .\roofline.py bcrit   | Tee-Object .\roofline_bcrit.txt
python .\roofline.py engines | Tee-Object .\roofline_engines.txt
python .\roofline.py context
python .\roofline.py book
```

These are Qualcomm's measurements from the `qai_hub_models` package, tested
against the roofline. `law` says decode time is a straight line in bytes
moved (R² ≥ 0.99); `bcrit` reads the critical batch off prefill/decode;
`engines` shows the CPU, GPU and NPU decoding alike while prefill spans 5×.

If you also have `qai_hub_models` installed, Qualcomm's own CLI prints the
same rows (metadata only, no weights):

```powershell
qai-hub-models perf qwen3_4b | Tee-Object .\aihub_perf_qwen.txt
```

---

## STEP 4 — AI Hub Workbench: the first X Plus 8-Core numbers

Qualcomm has measured nothing on X Plus 8-Core CRD — the device that stands
in for four of the seven HP machines. This measures it, without a download.

```powershell
# token: workbench.aihub.qualcomm.com -> Account -> Settings -> API Token
pip install qai-hub onnx numpy
qai-hub configure --api_token PASTE_YOUR_TOKEN_HERE
qai-hub list-devices
```

Then:

```powershell
python .\aihub_workbench.py plan
python .\aihub_workbench.py run --yes
python .\aihub_workbench.py results | Tee-Object .\workbench_results.txt
```

**Expect:** three lines (`fp16`, `w8a16`, `w4a16`) with `OK` and ms per layer,
then how much went up (~117 MB) and came back (a few KB of profile JSON).
`results` prints
`MEASURED on Workbench`, the effective GB/s, a bytes test across the three
precisions, and an extrapolated w4a16 decode rate next to the ceiling.

- It uploads once. Re-running reuses the upload recorded in
  `.aihub_workbench_cache.json` and submits the jobs again; `results` reads the
  latest record for each precision.
- A job that fails prints the Workbench error, saves it in
  `workbench_results.json`, and the run carries on with the rest.
- `--all-devices` adds X Elite and X2 Elite, where `results` compares the
  one-layer figure with Qualcomm's full-model number.
- `python .\aihub_workbench.py run --mock` rehearses the whole flow offline,
  and says `SIMULATED` everywhere it matters.

Put the job IDs from `workbench_results.json` into `SUBMISSION.md`.

---

## STEP 5 — Check your links before publishing

```powershell
python .\snapdragon_engine.py --json verify.json verify
```

**Expect:** `MISSING: 0`. `GATED/BLOCKED` and `NO-ID` are both fine — gated HF
repos and closed-weight models look like that legitimately.

---

## STEP 6 — See the training answer

```powershell
python .\on_device_adaptation.py verdict 0.5 16
python .\on_device_adaptation.py verdict 2.0 16
```

This is the answer to "can Snapdragon train locally". The last two lines of
each give pitch wording that is actually defensible.

---

## STEP 7 — The benchmark (Colab, NOT this laptop)

Your machine had 913 MiB free of 7.93 GiB. The benchmark fetches model
weights, so on this laptop it stops with `REFUSED` (exit code 6) for anything
over 1 GB — by design.

1. **colab.research.google.com** → **New notebook**
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Folder icon in the left sidebar → upload `sigil_t4_benchmark.py`
4. In a cell:

```python
!pip install -q -U transformers datasets accelerate sentencepiece
!python sigil_t4_benchmark.py --selftest
!python sigil_t4_benchmark.py --dry-run
```

**Expect:** `123/123 checks passed`, then "Environment is fine." On Colab the
download guard reports `hosted notebook`.

5. The real run, ~90 minutes, resumable:

```python
!python sigil_t4_benchmark.py
```

   Short on time: `!python sigil_t4_benchmark.py --quick` (~15 minutes).

6. Download `results_t4.json` and `kv_sweep.png` from the left sidebar.

---

## STEP 8 — Fill in the submission

```powershell
notepad .\SUBMISSION.md
```

| `[FILL]` | from |
|---|---|
| Workbench job IDs, ms per layer, extrapolated tok/s | `python .\aihub_workbench.py results` |
| FP16 perplexity | `results_t4.json` → `ppl_fp16` |
| KV at INT4 group-32 | `results_t4.json` → `kv_sweep` |
| fold error | `results_t4.json` → `fold_rel_err` |
| decode tok/s | `results_t4.json` → `decode_tok_per_s` |
| KV KiB/token | printed at the end of the run |

Check none are left, then re-run the audit:

```powershell
Select-String -Path .\SUBMISSION.md -Pattern '\[FILL\]'
python .\audit_claims.py
```

**Expect no output** from the first, and every claim `PASS` from the second.

---

## STEP 9 — Push to GitHub

```powershell
"backup_old/`n.venv/`n__pycache__/`n*.pyc`nworkbench_build/`n.aihub_workbench_cache.json" | Out-File .gitignore -Encoding utf8
git init
git add .
git commit -m "SIGIL-Edge: LLM deployment for Snapdragon-powered HP PCs"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sigil-edge.git
git push -u origin main
```

Keep `workbench_results.json` in the repo — it carries your job IDs.

---

## STEP 10 — Submit the form

| field | what to upload |
|---|---|
| Project Title | the title line in `SUBMISSION.md` |
| Brief Project Description | `SIGIL_Edge_Brief_Description.docx` |
| GitHub Repository Link | your repo URL |
| Short Pitch PDF | `SIGIL_Edge_Pitch.pdf` |
| Short Pitch PPT | `SIGIL_Edge_Pitch.pptx` |
| **Snapdragon Laptop** | **No** — answer honestly |

On that last one: say your hardware numbers are Qualcomm's own measurements
plus your own Workbench job IDs. Resourceful reads better than a claim that
gets checked.

---

## Quick reference

```powershell
.\run_all.ps1                                          # verify everything
python .\roofline.py law                                # the result
python .\aihub_workbench.py plan                        # the Workbench job, dry
python .\aihub_workbench.py run --yes                   # the Workbench job
python .\hp_snapdragon.py evidence                      # what counts as proof
python .\snapdragon_engine.py --json verify.json verify # check links
python .\psdc.py falsify                                # kill conditions
```

## If something breaks

| symptom | fix |
|---|---|
| `run_all.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` |
| "file is STALE" | re-extract that file from the zip, re-run `.\run_all.ps1` |
| dividers show as `←[1m` | `python apply_ascii_patch.py` |
| `aihub_workbench.py` says "not configured with an API token" | Step 4, first block |
| a Workbench job says `FAILED` | the service's message is saved in `workbench_results.json`; fix the cause and `run --yes` again -- the upload is reused |
| benchmark prints `REFUSED` | working as designed — run it on Colab (Step 7) |
| Colab: "No CUDA GPU detected" | Runtime → Change runtime type → **T4 GPU** |
| `verify` shows MISSING | that identifier 404s — fix or remove the entry |
