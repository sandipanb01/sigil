# =============================================================================
#  run_all.ps1 -- SIGIL-Edge local verification, start to finish
#
#  Usage, from D:\sigil with the venv active:
#      .\run_all.ps1
#
#  It checks which files you have, applies the ASCII divider patch if needed,
#  runs every test suite, the stress harness and the claim audit, and tells
#  you exactly what to do next. It downloads nothing and uploads nothing.
#  Safe to run repeatedly.
# =============================================================================

$ErrorActionPreference = "Continue"

# Run one of this project's scripts, whatever directory you invoked from.
#
# The first version wrote `python .\module.py`, which works only when the
# current directory happens to be the project folder -- and on any non-Windows
# PowerShell the backslash is not a separator at all, so the script could not
# be tested outside Windows. $PSScriptRoot fixes both.
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

function PyRun([string]$File, [string[]]$Arguments = @()) {
    $target = Join-Path $Root $File
    if (-not (Test-Path $target)) { return "MISSING FILE: $target" }
    try   { return (& python $target @Arguments 2>&1 | Out-String) }
    catch { return "python failed: $_" }
}

$pass = 0
$fail = 0

function Section($t) {
    Write-Host ""
    Write-Host ("-" * 70)
    Write-Host "  $t"
    Write-Host ("-" * 70)
}

function Ok($m)   { Write-Host "  [ OK ] $m"   -ForegroundColor Green; $script:pass++ }
function Bad($m)  { Write-Host "  [FAIL] $m"   -ForegroundColor Red;   $script:fail++ }
function Warn($m) { Write-Host "  [WARN] $m"   -ForegroundColor Yellow }
function Info($m) { Write-Host "         $m"   -ForegroundColor Gray }

# Print several lines of captured output WITHOUT mangling them.
#
# The obvious version -- Info ($out -split "`n" | Select-String "FAIL") --
# is wrong twice over. Out-String yields CRLF, so splitting on "`n" leaves a
# trailing carriage return on every element; and passing the resulting ARRAY
# into a string interpolation joins it with spaces. The CRs then rewind the
# cursor to column 0 mid-line and the terminal overwrites what it just drew,
# which is how a real failure list rendered as
#     FAILURESortability     54614/5628620
# Split on either line ending, trim, drop blanks, and print one at a time.
function InfoLines($text, $pattern = "", $first = 8) {
    if (-not $text) { return }
    $lines = $text -split "\r?\n" | ForEach-Object { $_.TrimEnd() } |
             Where-Object { $_ -ne "" }
    if ($pattern) { $lines = $lines | Where-Object { $_ -match $pattern } }
    $lines | Select-Object -First $first | ForEach-Object {
        Write-Host "         $_" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "  SIGIL-EDGE  local verification" -ForegroundColor Cyan

# --- 0. environment ----------------------------------------------------------
Section "ENVIRONMENT"
$py = (Get-Command python -ErrorAction SilentlyContinue)
if ($py) { Ok "python found: $(python --version 2>&1)" }
else     { Bad "python not on PATH. Activate the venv: .\.venv\Scripts\Activate.ps1"; exit 1 }

if ($env:VIRTUAL_ENV) { Ok "virtualenv active: $env:VIRTUAL_ENV" }
else { Warn "No virtualenv detected. Run: .\.venv\Scripts\Activate.ps1" }

# --- 1. required files -------------------------------------------------------
Section "FILES"
$core = @(
    "snapdragon_engine.py", "psdc.py", "qat_optimizer.py",
    "generator_quant.py", "slt_compressibility.py",
    "on_device_adaptation.py", "sigil_t4_benchmark.py", "aihub_bench.py",
    "residual_cascade.py", "quant_certificate.py", "dream_search.py",
    "stress_all.py", "audit_claims.py", "hp_snapdragon.py",
    "roofline.py", "aihub_workbench.py"
)
$docs = @("README.md", "QUICKSTART.md", "RUNBOOK.md", "SUBMISSION.md",
          "PROVENANCE.md", "FINDINGS.md", "RESEARCH_AGENDA.md", "MANIFEST.md")
$missing = @()
foreach ($f in $core) {
    if (Test-Path (Join-Path $Root $f)) { Ok $f } else { Bad "$f  <-- MISSING, re-download it"; $missing += $f }
}
foreach ($f in $docs) {
    if (Test-Path (Join-Path $Root $f)) { Info "$f" } else { Warn "$f missing (documentation only)" }
}
if (Test-Path (Join-Path $Root "sigil/__init__.py")) { Ok "sigil\ package present" }
else {
    Warn "sigil\ package folder is missing"
    Info "Optional: only run_local_validation.py and p1_test.py use it."
    Info "Everything else runs without it, and the stress harness says so"
    Info "rather than failing. To restore it, re-extract the zip and keep"
    Info "the sigil\ SUBFOLDER -- copying the loose .py files leaves it out."
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "  Download the missing files before continuing." -ForegroundColor Red
    exit 1
}

# --- 2. staleness check ------------------------------------------------------
Section "STALENESS"
$eng = Get-Content (Join-Path $Root "snapdragon_engine.py") -Raw
# Use POSITIVE markers of the corrected version. An earlier draft of this script
# searched for the old command string and produced a FALSE STALE on a correct
# file -- the phrase survives in the docstring that explains the old mistake.
$hasNew = ($eng -match "WHICH DEVICES EXIST") -and ($eng -match "qai-hub-models perf")
if ($hasNew) {
    Ok "export command is the corrected version"
} else {
    Bad "snapdragon_engine.py is STALE -- it emits the old export command."
    Info "Re-download snapdragon_engine.py, then re-run this script."
    exit 1
}

if ($eng -match "qai-hub list-devices") { Ok "export tells you to list devices first" }
else { Warn "export block looks unexpected; re-download snapdragon_engine.py" }

if ($eng -match "grootn15 no longer claims verification|404 on verification") {
    Ok "grootn15 correction present"
} else { Warn "grootn15 correction missing -- file may be stale" }

# Zero download, positively: the corrected engine points AI Hub work at the
# Workbench runner, and the corrected HP module no longer says tier 3 is empty.
if ($eng -match "ZERO-DOWNLOAD since 2026-09-22") {
    Ok "engine emits no model downloads (zero-download AI Hub path)"
} else {
    Bad "snapdragon_engine.py is STALE -- it predates the zero-download fix."
    Info "Re-extract it from the zip, then re-run this script."
}
$hpsrc = Get-Content (Join-Path $Root "hp_snapdragon.py") -Raw
if ($hpsrc -match "Cite tier 3 for X Elite and X2 Elite") {
    Ok "HP module carries the corrected AI Hub finding"
} else {
    Bad "hp_snapdragon.py is STALE -- it still says Qualcomm publishes nothing."
}

# --- 3. ASCII divider patch --------------------------------------------------
Section "ASCII DIVIDERS (PowerShell rendering)"
if ($eng -match 'return "-" \* width') {
    Ok "already ASCII-patched"
} elseif (Test-Path (Join-Path $Root "apply_ascii_patch.py")) {
    Info "applying patch..."
    & python (Join-Path $Root "apply_ascii_patch.py") | Out-Null
    if ($LASTEXITCODE -eq 0) { Ok "patch applied" } else { Bad "patch failed" }
} else {
    Warn "apply_ascii_patch.py not present; dividers may render as garbage"
}

# --- 4. test suites ----------------------------------------------------------
Section "TEST SUITES"
$suites = @(
    @{n="snapdragon_engine";    f="snapdragon_engine.py";    a=@("selftest")},
    @{n="psdc";                 f="psdc.py";                 a=@("selftest")},
    @{n="generator_quant";      f="generator_quant.py";      a=@()},
    @{n="qat_optimizer";        f="qat_optimizer.py";        a=@()},
    @{n="slt_compressibility";  f="slt_compressibility.py";  a=@()},
    @{n="on_device_adaptation"; f="on_device_adaptation.py"; a=@()},
    @{n="aihub_bench";          f="aihub_bench.py";          a=@("selftest")},
    @{n="residual_cascade";     f="residual_cascade.py";     a=@("selftest")},
    @{n="quant_certificate";    f="quant_certificate.py";    a=@("selftest")},
    @{n="dream_search";         f="dream_search.py";         a=@("selftest")},
    @{n="hp_snapdragon";        f="hp_snapdragon.py";        a=@("selftest")},
    @{n="roofline";             f="roofline.py";             a=@("selftest")},
    @{n="aihub_workbench";      f="aihub_workbench.py";      a=@("selftest")}
)
foreach ($s in $suites) {
    $out = PyRun $s.f $s.a
    $m = [regex]::Match($out, '(\d+)/(\d+) (?:checks )?passed')
    if ($m.Success -and $m.Groups[1].Value -eq $m.Groups[2].Value) {
        Ok ("{0,-22} {1}" -f $s.n, $m.Value)
    } else {
        Bad ("{0,-22} did not report a clean pass" -f $s.n)
        InfoLines $out "" 3
    }
}

# --- 5. benchmark script ------------------------------------------------------
# Grepping the source for marker strings only ever proved the file LOOKED right.
# The benchmark now carries its own tests, so run them: they need no GPU, no
# network and about half a minute, and they are what was missing when this
# script broke twice on Colab.
Section "BENCHMARK SCRIPT"
$bm = Get-Content (Join-Path $Root "sigil_t4_benchmark.py") -Raw
if ($bm -match "def selftest") {
    Ok "benchmark carries its own test suite"
    $torchOk = (& python -c "import torch" 2>&1 | Out-String).Trim() -eq ""
    if ($torchOk) {
        $out = PyRun "sigil_t4_benchmark.py" @("--selftest")
        $m = [regex]::Match($out, '(\d+)/(\d+) checks passed')
        if ($m.Success -and $m.Groups[1].Value -eq $m.Groups[2].Value) {
            Ok ("benchmark selftest       {0}" -f $m.Value)
        } else {
            Bad "benchmark selftest did not pass"
            InfoLines $out "FAIL" 5
        }
        Info "For the full shape matrix (2-4 min): python .\sigil_t4_benchmark.py --stress"
    } else {
        Warn "torch not installed here, so the benchmark tests were skipped."
        Info "That is fine on a laptop -- run them on Colab before the real job:"
        Info "  !python sigil_t4_benchmark.py --selftest"
    }
} else {
    Bad "sigil_t4_benchmark.py is STALE -- it has no test suite. Re-download it."
}

# --- 5b. Bonsai 2 / container rule -------------------------------------------
Section "CONTAINER RULE (Bonsai 2 27B)"
if ($eng -match "validate_container_rule") {
    $out = PyRun "snapdragon_engine.py" @("container","--validate")
    # Tolerate ANSI colour codes between the verdict and the count.
    if (($out -match "HOLDS") -and ($out -match "8/8 correct")) {
        Ok "container rule holds on all 8 measured GPUs"
        Info "Every Snapdragon part sits 4.4-19.7x below the crossover -> PTQ1_0"
    } else {
        Bad "container rule did not validate"
        InfoLines $out "" 4
    }
} else {
    Warn "snapdragon_engine.py predates the Bonsai 2 work; re-download it"
}

# --- 5b2. Qualcomm's own measurements -----------------------------------------
Section "QUALCOMM'S OWN DATA vs THE ROOFLINE (nothing downloaded)"
$bk = PyRun "roofline.py" @("book")
if ($bk -match "17/17 reproduced -- FAITHFUL") {
    Ok "the scaling book's worked answers reproduce, 17/17"
} else { Bad "roofline.py book did not reproduce the scaling book"; InfoLines $bk "" 4 }
$lw = PyRun "roofline.py" @("law")
if ($lw -match "BANDWIDTH-BOUND") {
    Ok "decode is linear in bytes on Qualcomm's X-series data"
    Info "python .\roofline.py law      # R^2 >= 0.99 on X Elite, QAIRT, w4a16"
    Info "python .\roofline.py bcrit    # prefill/decode = the critical batch"
} else { Bad "roofline.py law did not run"; InfoLines $lw "" 4 }

# --- 5b3. AI Hub Workbench, zero download --------------------------------------
Section "AI HUB WORKBENCH (uploads one layer, downloads only JSON)"
$pl = PyRun "aihub_workbench.py" @("plan")
if ($pl -match "profile JSON only") {
    Ok "Workbench plan: X Plus 8-Core CRD, profiles only, 2 MB download ceiling"
} else { Bad "aihub_workbench.py plan did not run"; InfoLines $pl "" 4 }
$cfg = Join-Path $HOME ".qai_hub/client.ini"
if (Test-Path $cfg) {
    Ok "AI Hub API token configured"
    Info "Measure X Plus 8-Core:  python .\aihub_workbench.py run --yes"
} else {
    Warn "No AI Hub API token yet -- the one step that needs it is RUNBOOK Step 4"
    Info "workbench.aihub.qualcomm.com -> Account -> Settings -> API Token, then"
    Info "  qai-hub configure --api_token PASTE_YOUR_TOKEN_HERE"
}
$wr = Join-Path $Root "workbench_results.json"
if (Test-Path $wr) {
    $res = PyRun "aihub_workbench.py" @("results")
    if ($res -match "MEASURED on Workbench") {
        Ok "Workbench measurements present -- put the job IDs in SUBMISSION.md"
    } elseif ($res -match "SIMULATED") {
        Info "workbench_results.json holds only a mock rehearsal, no measurement yet"
    }
}

# --- 5c. Navier-Stokes derived work -------------------------------------------
Section "CORRECTION CYCLE + CERTIFICATES"
$rc = PyRun "residual_cascade.py" @("cycle")
if ($rc -match "cycle closes: True") {
    Ok "the paper's correction cycle is reproduced and closes"
    Info "sigma_j = 1/5 + j/10, floor at sigma > 0.1 from the quadratic term"
} else { Bad "residual_cascade could not reproduce the cycle" }

$qc = PyRun "quant_certificate.py" @("experiment")
if ($qc -match "faults caught by probes alone: 7/7") {
    Ok "certificate catches all 7 injected faults"
} else { Bad "certificate fault detection did not reach 7/7" }
Info "Calibration sizing:  python .\residual_cascade.py plan --model bonsai-2-27b"
Info "Fault catalogue:     python .\quant_certificate.py faults"

# --- 5d. the stress harness --------------------------------------------------
Section "STRESS HARNESS"
$st = PyRun "stress_all.py"
if ($st -match "Nothing broke") {
    $m = [regex]::Match($st, "(\d+)/(\d+) checks passed")
    Ok ("stress harness clean   " + $m.Value)
    Info "fuzzing, numeric sanity, determinism, portability, CLI smoke, network"
    # The documents quote how many checks the harness runs. Read the number the
    # claim audit holds them to, and hold the harness to it as well: a count
    # that drifts from the documents is how a stale figure ships.
    $ac = Get-Content (Join-Path $Root "audit_claims.py") -Raw
    $d = [regex]::Match($ac, "STRESS_FULL, STRESS_NO_PKG, SELFTESTS, AUDIT_CLAIMS = (\d+), (\d+), (\d+), (\d+)")
    if ($d.Success -and $m.Success) {
        $hasPkg = Test-Path (Join-Path $Root "sigil/__init__.py")
        $want = if ($hasPkg) { $d.Groups[1].Value } else { $d.Groups[2].Value }
        if ($m.Groups[2].Value -eq $want) {
            Ok ("stress harness ran the documented $want checks")
        } else {
            Bad ("stress harness ran " + $m.Groups[2].Value + " checks; the documents say $want")
        }
    }
} else {
    Bad "stress harness reported failures"
    InfoLines $st "FAIL|failures" 8
}
Info "Replay search:  python .\dream_search.py compare"

# --- 5d2. the HP target -------------------------------------------------------
Section "SNAPDRAGON-POWERED HP PCs"
$hp = PyRun "hp_snapdragon.py" @("machines")
if ($hp -match "HP EliteBook 6 G1q") {
    Ok "HP machine catalogue resolves to AI Hub devices"
    Info "One configuration has NO AI Hub device: OmniBook Ultra 14-kg000 (X2 Plus)"
    Info "python .\hp_snapdragon.py evidence   # the NPU/CPU split, ranked"
} else { Bad "hp_snapdragon.py machines did not run" }

# --- 5e. claim audit ----------------------------------------------------------
Section "CLAIM AUDIT"
$ca = PyRun "audit_claims.py"
$m = [regex]::Match($ca, "(\d+)/(\d+) published claims")
if ($m.Success -and $m.Groups[1].Value -eq $m.Groups[2].Value) {
    Ok ("every published number matches the code   " + $m.Value)
    Info "Run this again before you quote any figure out loud."
} else {
    Bad "a published claim and the code disagree"
    InfoLines $ca "FAIL" 8
}

# --- 6. summary --------------------------------------------------------------
Section "SUMMARY"
Write-Host "  passed: $pass    failed: $fail"
Write-Host ""
if ($fail -eq 0) {
    Write-Host "  Local setup is clean. Nothing was downloaded. Next steps:" -ForegroundColor Green
    Write-Host ""
    Write-Host "    0.  python .\aihub_workbench.py run --yes"
    Write-Host "        (needs your AI Hub token; profiles X Plus 8-Core CRD, the"
    Write-Host "         device Qualcomm has not measured; downloads only JSON)"
    Write-Host ""
    Write-Host "    1.  python .\snapdragon_engine.py --json verify.json verify"
    Write-Host "        (expect 0 MISSING; GATED/BLOCKED and NO-ID are fine)"
    Write-Host ""
    Write-Host "    2.  Run the benchmark on a Colab T4 -- NOT on this laptop."
    Write-Host "        Upload sigil_t4_benchmark.py, set Runtime -> T4 GPU, then:"
    Write-Host "          !pip install -q -U transformers datasets accelerate sentencepiece"
    Write-Host "          !python sigil_t4_benchmark.py --selftest    # 30s, do this first"
    Write-Host "          !python sigil_t4_benchmark.py --stress      # 2-4 min, optional"
    Write-Host "          !python sigil_t4_benchmark.py               # the real job"
    Write-Host ""
    Write-Host "        It auto-detects the GPU, the dtype and the attention shape,"
    Write-Host "        so the same command works on a T4, an A100, a Mac or a CPU."
    Write-Host ""
    Write-Host "    3.  Paste the Colab and Workbench numbers into SUBMISSION.md over [FILL]"
    Write-Host ""
    Write-Host "    4.  git init; git add .; git commit -m 'SIGIL-Edge'; git push"
    Write-Host ""
} else {
    Write-Host "  Fix the failures above, then re-run .\run_all.ps1" -ForegroundColor Red
    Write-Host ""
}
