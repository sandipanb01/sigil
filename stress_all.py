#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 stress_all.py -- one command that tries to break everything
================================================================================

    python stress_all.py              # everything, ~2 min, no GPU, no network
    python stress_all.py --quick      # skip the slow module suites
    python stress_all.py --only fuzz  # one phase

Each module here carries its own self-test, and those prove the code does what
it claims on inputs it expects. This proves something different and, for code
that runs on someone else's machine, more useful: that nothing does something
WORSE THAN FAILING on inputs it does not expect.

Seven phases.

  1. IMPORT      every module imports on a bare interpreter
  2. SUITES      every self-test passes, and the totals are checked against
                 what the documentation claims
  3. FUZZ        every public callable is hit with adversarial arguments --
                 None, NaN, infinity, negatives, zero, enormous values, empty
                 containers, wrong types -- and must either work or fail
                 CLEANLY
  4. NUMERIC     nothing returns NaN or infinity quietly
  5. DETERMINISM the same call twice gives the same answer
  6. PORTABILITY no non-ASCII in any .py file, no tabs, every CLI runs
  7. NETWORK     nothing in the project downloads a model -- checked on the
                 syntax tree of every file, on every command it emits, and on
                 every document that tells a person what to type

WHAT COUNTS AS FAILING CLEANLY. ValueError, TypeError, KeyError and
FileNotFoundError are a function saying "you gave me something I do not
accept", which is correct behaviour. AttributeError, IndexError,
ZeroDivisionError, RecursionError and UnboundLocalError are a function
discovering its own confusion halfway through, which is a bug even when it
happens to raise.

The worst outcome is none of those: a function that takes nonsense and returns
a plausible-looking number. Phase 4 exists for that, and this project has
shipped that failure before -- a benchmark that reduced over the wrong axis and
produced perplexities nobody could tell were wrong.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import inspect
import io
import math
import pathlib
import subprocess
import sys
import time
import contextlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MODULES = [
    "snapdragon_engine", "psdc", "generator_quant", "qat_optimizer",
    "slt_compressibility", "on_device_adaptation", "aihub_bench",
    "residual_cascade", "quant_certificate", "dream_search",
    "hp_snapdragon", "roofline", "aihub_workbench",
]

# The sigil/ package carries no self-test of its own, so it was invisible to
# every phase here. A hardening patch then broke sigil/quant.py -- it used
# math.isfinite in a file that does not import math -- and nothing noticed
# until run_local_validation.py was run by hand. Anything importable is now
# imported and fuzzed, whether or not it has a suite.
SUBMODULES = [
    "sigil.quant", "sigil.rotation", "sigil.npu", "sigil.allocate",
    "sigil.gof",
]

# Scripts with no selftest subcommand. They are slow, so they are smoke-tested
# for import and syntax rather than executed here; RUNBOOK.md covers running
# them for real.
LEGACY_SCRIPTS = ["run_local_validation.py", "p1_test.py",
                  "apply_ascii_patch.py", "build_deck.py"]

EXPECTED_TESTS = {
    "snapdragon_engine": 141, "psdc": 60, "generator_quant": 47,
    "qat_optimizer": 43, "slt_compressibility": 22,
    "on_device_adaptation": 23, "aihub_bench": 16, "residual_cascade": 49,
    "quant_certificate": 58, "dream_search": 87,
    "hp_snapdragon": 56, "roofline": 83, "aihub_workbench": 64,
}

CLEAN_ERRORS = (ValueError, TypeError, KeyError, FileNotFoundError,
                NotImplementedError, ArithmeticError.__subclasses__ and
                OverflowError, StopIteration)
DIRTY_ERRORS = (AttributeError, IndexError, ZeroDivisionError,
                RecursionError, UnboundLocalError, AssertionError,
                MemoryError, SystemError)

# Adversarial values, in rough order of nastiness.
NASTY: List[Any] = [
    None, 0, -1, -0.0, 0.0, 1, 2, 0.5, -0.5,
    float("nan"), float("inf"), float("-inf"),
    1e308, -1e308, 10 ** 30, "", "x", "  ", [], (), {}, set(),
    [None], [float("nan")], {"a": None}, True, False, b"", object(),
]

# Callables that must not be fuzzed: they cost real time, touch the network,
# spawn work, or are the test harness itself.
SKIP_FUZZ = {
    "selftest", "main", "build_parser", "run", "stress",
    "verify_identifiers", "experiment_refinement_law",
    "experiment_fault_detection", "experiment_eos_vs_quantisation",
    "bench_model", "aihub", "probe", "HardwareProbe",
    # aihub_workbench: these WRITE files or build 400 MB models. Fuzzing
    # save_cache(None) would overwrite a real Workbench cache with "null".
    # Their argument checking is covered by that module's own self-test.
    "save_cache", "stand_in_archive", "build", "run_decode_experiment",
    "real_backend", "verify_graph",
}


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _watchdog(seconds: int):
    """
    Bound one call. A function that loops forever on hostile input is exactly
    what this harness is looking for, so it must never be able to take the
    harness down with it -- the first run of this file hung on
    qat_schedule(10**30) and reported nothing at all.

    SIGALRM is Unix-only. On Windows the body simply runs unguarded, which is
    acceptable because the guards it checks for are now in place.
    """
    try:
        import signal
        has = hasattr(signal, "SIGALRM")
    except ImportError:
        has = False
    if not has:
        yield
        return

    def _fire(signum, frame):
        raise _Timeout(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, bool, str]] = []
        self.t0 = time.time()

    def add(self, phase: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((phase, name, bool(ok), detail))

    def failures(self) -> List[Tuple[str, str, bool, str]]:
        return [r for r in self.rows if not r[2]]

    def render(self) -> int:
        by_phase: Dict[str, List[Tuple[str, str, bool, str]]] = {}
        for r in self.rows:
            by_phase.setdefault(r[0], []).append(r)
        print()
        print("=" * 74)
        print("  STRESS SUMMARY")
        print("=" * 74)
        for phase, rows in by_phase.items():
            npass = sum(1 for r in rows if r[2])
            mark = "ok  " if npass == len(rows) else "FAIL"
            print(f"  [{mark}] {phase:<16}{npass}/{len(rows)}")
        fails = self.failures()
        if fails:
            print()
            print("  FAILURES")
            for phase, name, _, detail in fails[:40]:
                print(f"    {phase}: {name}")
                if detail:
                    print(f"        {detail[:150]}")
            if len(fails) > 40:
                print(f"    ... and {len(fails) - 40} more")
        total = len(self.rows)
        npass = total - len(fails)
        print()
        print(f"  {npass}/{total} checks passed in {time.time() - self.t0:.0f}s")
        if not fails:
            print("  Nothing broke.")
        return 0 if not fails else 1


# --------------------------------------------------------------------------- #
# phase 1 -- imports
# --------------------------------------------------------------------------- #

def phase_import(rep: Report) -> Dict[str, Any]:
    """
    Import everything. The sigil/ package is OPTIONAL -- only
    run_local_validation.py and p1_test.py use it -- so its absence is a
    warning with a fix, not five red failures.

    That distinction is not pedantry. A user who copied the loose .py files
    instead of unzipping the folder saw five ModuleNotFoundErrors and a failed
    run, when the answer was one line: extract the sigil/ subfolder too.
    A package that is PRESENT but broken still fails, which is the case that
    matters -- a hardening patch once broke sigil/quant.py and nothing noticed.
    """
    mods: Dict[str, Any] = {}
    here = pathlib.Path(__file__).parent
    pkg_present = (here / "sigil" / "__init__.py").exists()

    for name in MODULES:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                mods[name] = __import__(name)
            rep.add("import", name, True)
        except Exception as exc:
            rep.add("import", name, False, f"{type(exc).__name__}: {exc}")

    if not pkg_present:
        rep.add("import", "sigil/ package (optional) is absent", True,
                "only run_local_validation.py and p1_test.py need it; "
                "re-extract the sigil/ subfolder from the zip to enable it")
        for name in SUBMODULES:
            rep.add("import", f"{name} skipped (package absent)", True)
    else:
        for name in SUBMODULES:
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    mods[name] = __import__(name)
                rep.add("import", name, True)
            except Exception as exc:
                rep.add("import", name, False, f"{type(exc).__name__}: {exc}")
    here = pathlib.Path(__file__).parent
    for script in LEGACY_SCRIPTS:
        f = here / script
        if not f.exists():
            rep.add("import", script, False, "missing")
            continue
        ok, why = _compiles(f.read_text(encoding="utf-8", errors="replace"),
                            script)
        rep.add("import", f"{script} compiles", ok, why)
    if not pkg_present:
        rep.add("import", "run_local_validation.py / p1_test.py need sigil/",
                True, "they will not run until the package is extracted")
    return mods


# --------------------------------------------------------------------------- #
# phase 2 -- the module suites
# --------------------------------------------------------------------------- #

def phase_suites(rep: Report, quick: bool) -> None:
    if quick:
        rep.add("suites", "skipped (--quick)", True)
        return
    import re
    # Anchor to THIS file's folder and run there. Passing a bare "name.py"
    # only works when the current directory happens to be the project folder,
    # so invoking the harness from anywhere else reported every suite as
    # failing -- 0/11 -- when nothing was wrong with any of them.
    here = pathlib.Path(__file__).resolve().parent
    for name in MODULES:
        if "." in name:
            continue
        script = here / f"{name}.py"
        if not script.exists():
            rep.add("suites", name, False, f"missing: {script}")
            continue
        args = [sys.executable, str(script)]
        args += ["--selftest"] if name == "sigil_t4_benchmark" else ["selftest"]
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=900, cwd=str(here))
        except subprocess.TimeoutExpired:
            rep.add("suites", name, False, "timed out after 900s")
            continue
        text = out.stdout + out.stderr
        m = re.search(r"(\d+)/(\d+) (?:checks )?passed", text)
        if not m:
            rep.add("suites", name, False,
                    "no pass line in output; exit " + str(out.returncode))
            continue
        got, tot = int(m.group(1)), int(m.group(2))
        rep.add("suites", f"{name} all pass", got == tot, f"{got}/{tot}")
        exp = EXPECTED_TESTS.get(name)
        if exp is not None:
            rep.add("suites", f"{name} test count matches documentation",
                    tot == exp, f"{tot} vs documented {exp}")
        rep.add("suites", f"{name} exits 0", out.returncode == 0,
                f"exit {out.returncode}")


# --------------------------------------------------------------------------- #
# phase 3 -- fuzzing every public callable
# --------------------------------------------------------------------------- #

def _public_callables(mod) -> List[Tuple[str, Callable]]:
    """
    Every public callable the module OWNS.

    Taking `__all__` when present found zero callables in snapdragon_engine,
    whose `__all__` lists only classes -- so the largest module in the project
    was silently not fuzzed at all. Scan both and take the union.
    """
    names = set(getattr(mod, "__all__", None) or ())
    names |= {n for n in dir(mod) if not n.startswith("_")}
    out = []
    for n in sorted(names):
        if n in SKIP_FUZZ:
            continue
        obj = getattr(mod, n, None)
        if not callable(obj):
            continue
        if inspect.isclass(obj) and not hasattr(obj, "__dataclass_fields__"):
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        out.append((n, obj))
    return out


def _arity(fn) -> int:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return 0
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is p.empty:
            n += 1
    return min(n, 3)


def phase_fuzz(rep: Report, mods: Dict[str, Any], budget: int) -> None:
    import itertools
    for mname, mod in mods.items():
        fns = _public_callables(mod)
        dirty: List[str] = []
        silent: List[str] = []
        hangs: List[str] = []
        calls = 0
        for fname, fn in fns:
            k = _arity(fn)
            if k == 0:
                combos: Sequence[Tuple] = [()]
            elif k == 1:
                combos = [(v,) for v in NASTY]
            else:
                # ALL 29 hostile values, and a default budget (900) that covers
                # every one of the 841 pairs. It used to be the first 14 values
                # and a budget of 60 -- under a third of even those pairs --
                # and that hid nine real defects: kept_fraction(1, nan) -> NaN,
                # shrinking_support_schedule(1, 0) -> ZeroDivisionError, float
                # overflow to inf in five functions, and two AttributeErrors
                # that only a string in the FIRST argument reaches.
                pool = NASTY
                combos = list(itertools.product(pool, repeat=min(k, 2)))[:budget]
            for args in combos:
                calls += 1
                try:
                    buf = io.StringIO()
                    with _watchdog(5), contextlib.redirect_stdout(buf), \
                            contextlib.redirect_stderr(io.StringIO()):
                        res = fn(*args)
                except _Timeout:
                    hangs.append(f"{fname}{_fmt(args)} did not return in 5s")
                except DIRTY_ERRORS as exc:
                    dirty.append(f"{fname}{_fmt(args)} -> "
                                 f"{type(exc).__name__}: {str(exc)[:60]}")
                except RecursionError:
                    dirty.append(f"{fname}{_fmt(args)} -> RecursionError")
                except Exception:
                    pass                      # a clean refusal
                else:
                    bad = _nonfinite(res)
                    if bad:
                        silent.append(f"{fname}{_fmt(args)} -> {bad}")
        rep.add("fuzz", f"{mname}: no confused exceptions ({calls} calls)",
                not dirty, "; ".join(dirty[:3]))
        rep.add("fuzz", f"{mname}: no silent NaN/inf returns",
                not silent, "; ".join(silent[:3]))
        rep.add("fuzz", f"{mname}: nothing hangs on hostile input",
                not hangs, "; ".join(hangs[:3]))


def _fmt(args: Sequence[Any]) -> str:
    def one(a):
        r = repr(a)
        return r if len(r) <= 14 else r[:11] + "..."
    return "(" + ", ".join(one(a) for a in args) + ")"


def _nonfinite(obj: Any, depth: int = 0) -> str:
    """Find a NaN or infinity hiding in a return value."""
    if depth > 3:
        return ""
    if isinstance(obj, bool):
        return ""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "inf"
        return ""
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:40]:
            found = _nonfinite(v, depth + 1)
            if found:
                return f"{k}={found}"
        return ""
    if isinstance(obj, (list, tuple)):
        for v in list(obj)[:40]:
            found = _nonfinite(v, depth + 1)
            if found:
                return found
        return ""
    return ""


# --------------------------------------------------------------------------- #
# phase 4 -- numeric sanity on the load-bearing functions
# --------------------------------------------------------------------------- #

def phase_numeric(rep: Report, mods: Dict[str, Any]) -> None:
    E = mods.get("snapdragon_engine")
    R = mods.get("residual_cascade")
    Q = mods.get("quant_certificate")
    D = mods.get("dream_search")

    if E:
        rep.add("numeric", "container rule still holds 8/8",
                E.validate_container_rule()["verdict"] == "HOLDS")
        rep.add("numeric", "bpw is positive for every measured container",
                all(E.container_tax(k)["effective_bpw"] > 0
                    for k in E.CONTAINER_MEASUREMENTS))
        rep.add("numeric", "F16 container measures 16.0 bpw",
                abs(E.container_tax("bonsai-2-27b F16 (gguf)"
                                    )["effective_bpw"] - 16.0) < 0.005)
        rep.add("numeric", "footprints are finite and positive",
                all((s.footprint_gb() or 1) > 0 and
                    math.isfinite(s.footprint_gb() or 1)
                    for s in E.MODEL_DB.values()))
        rep.add("numeric", "every catalogue entry has a licence",
                all(bool(s.licence) for s in E.MODEL_DB.values()))
        rep.add("numeric", "no catalogue key is duplicated",
                len(E.MODEL_DB) == len({s.key for s in E.MODEL_DB.values()}))
        rep.add("numeric", "predict_container is monotone in bandwidth",
                _monotone([E.predict_container(b)["container"]
                           for b in (10, 100, 500, 1000, 1500, 2000, 5000)]))
    if R:
        rep.add("numeric", "kept fraction never exceeds 1",
                all(R.kept_fraction(d, n) <= 1.0
                    for d in (1, 10, 1000) for n in (1, 100, 10 ** 9)))
        rep.add("numeric", "kept fraction increases with tokens",
                R.kept_fraction(1000, 10 ** 6) > R.kept_fraction(1000, 10 ** 3))
        rep.add("numeric", "tokens needed is a positive integer",
                all(isinstance(R.calibration_tokens_needed(d, k), int)
                    and R.calibration_tokens_needed(d, k) > 0
                    for d in (1, 5120) for k in (0.0, 0.5, 0.99)))
        rep.add("numeric", "the paper's cycle still validates",
                R.validate_against_paper()["verdict"] == "FAITHFUL")
    if Q:
        rep.add("numeric", "tolerance is positive and finite",
                all(Q.derive_tolerance(d, k)["rel_l2"] > 0
                    and math.isfinite(Q.derive_tolerance(d, k)["rel_l2"])
                    for d in ("fp32", "fp16", "bf16") for k in (1, 4096)))
        rep.add("numeric", "a vacuous tolerance is reported as uncertifiable",
                not Q.derive_tolerance("bf16", 4096)["certifiable"]
                and Q.derive_tolerance("fp32", 4096)["certifiable"])
        ex = Q.experiment_fault_detection(32, 16, 3, verbose=False)
        rep.add("numeric", "fault detection still catches everything",
                ex["caught"] == ex["total"] and ex["correct_passes"],
                f"{ex['caught']}/{ex['total']}")
    if D:
        for pool in D.POOLS.values():
            rep.add("numeric", f"{pool.name} best is finite",
                    pool.best() is not None
                    and math.isfinite(pool.best()[1].value))
        cmp = D.compare_policies(D.COLAB_2026_09)
        rep.add("numeric", "every policy stays inside the realized space",
                all(r["reliable"] for r in cmp["rows"]))
        rep.add("numeric", "no policy claims more calls than exhaustive",
                all((r["fewer_calls"] or 1) >= 1.0 for r in cmp["rows"]))


def _monotone(seq: Sequence[str]) -> bool:
    """Once the answer flips it must not flip back."""
    flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return flips <= 1


# --------------------------------------------------------------------------- #
# phase 5 -- determinism
# --------------------------------------------------------------------------- #

def phase_numeric_roofline(rep: Report, mods: Dict[str, Any]) -> None:
    RL = mods.get("roofline")
    if not RL:
        return
    rep.add("numeric", "the scaling book's worked answers still reproduce",
            RL.validate_against_book()["verdict"] == "FAITHFUL")
    rep.add("numeric", "decode is linear in bytes on X Elite QAIRT (R^2 >= 0.99)",
            RL.bandwidth_law("X Elite", "geniex_qairt", "w4a16", "npu", 4096)["min_r2"] >= 0.99)
    rep.add("numeric", "no Qualcomm measurement beats its roofline ceiling",
            all(r[6] < RL.decode_ceiling(r[0], r[1], r[5],
                                         RL.SOCS[r[2]].bw_alt_gbs or RL.SOCS[r[2]].bw_gbs, 4, 8)
                for r in RL.QUALCOMM_MEASURED))
    rep.add("numeric", "every critical batch is finite and positive",
            all(0 < r["b_crit"] < 1e6 for r in RL.critical_batch_table()))


def phase_determinism(rep: Report, mods: Dict[str, Any]) -> None:
    import json
    cases: List[Tuple[str, Callable[[], Any]]] = []
    RL = mods.get("roofline")
    if RL:
        cases.append(("roofline law table", lambda: RL.bandwidth_law_table()))
        cases.append(("effective critical batch", lambda: RL.effective_critical_batch()))
    W = mods.get("aihub_workbench")
    if W:
        cases.append(("Workbench plan byte estimate",
                      lambda: (W.estimated_upload_bytes(W.SliceSpec("qwen3_4b")),
                               W.calibration_bytes(W.SliceSpec("qwen3_4b")))))
    E, R, Q, D = (mods.get(k) for k in ("snapdragon_engine", "residual_cascade",
                                        "quant_certificate", "dream_search"))
    if E:
        cases.append(("container rule", lambda: E.validate_container_rule()))
        cases.append(("vision tower budget", lambda: E.vision_tower_budget()))
        cases.append(("ternary viability", lambda: E.ternary_viability(7.0)))
    if R:
        cases.append(("cycle analysis", lambda: R.analyse_cycle()))
        cases.append(("refinement plan",
                      lambda: R.layer_refinement_plan("bonsai-2-27b", 262144)))
    if Q:
        cases.append(("probe digest", lambda: Q.digest_array(
            Q.probe_input(Q.STANDARD_PROBES[0], 64))))
        cases.append(("tolerance", lambda: Q.derive_tolerance("fp32", 4096)))
    if D:
        cases.append(("policy comparison",
                      lambda: D.compare_policies(D.COLAB_2026_09)))
    for name, fn in cases:
        try:
            a = json.dumps(fn(), sort_keys=True, default=str)
            b = json.dumps(fn(), sort_keys=True, default=str)
            rep.add("determinism", name, a == b)
        except Exception as exc:
            rep.add("determinism", name, False,
                    f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# phase 6 -- portability and the command line
# --------------------------------------------------------------------------- #

CLI_SMOKE = [
    ("snapdragon_engine.py", ["container"]),
    ("snapdragon_engine.py", ["container", "--validate"]),
    ("snapdragon_engine.py", ["models"]),
    ("snapdragon_engine.py", ["lowbit"]),
    ("snapdragon_engine.py", ["capability", "--model", "bonsai-2-27b"]),
    ("residual_cascade.py", ["cycle"]),
    ("residual_cascade.py", ["law"]),
    ("residual_cascade.py", ["plan", "--model", "bonsai-2-27b"]),
    ("quant_certificate.py", ["faults"]),
    ("quant_certificate.py", ["experiment"]),
    ("quant_certificate.py", ["demo"]),
    ("dream_search.py", ["pools"]),
    ("dream_search.py", ["compare"]),
    ("dream_search.py", ["compare", "--pool", "TINY_CPU"]),
    ("dream_search.py", ["dream", "--policy", "greedy"]),
    ("psdc.py", ["plan"]),
    ("psdc.py", ["falsify"]),
    ("hp_snapdragon.py", ["machines"]),
    ("hp_snapdragon.py", ["aihub"]),
    ("hp_snapdragon.py", ["evidence"]),
    ("hp_snapdragon.py", ["commands"]),
    ("hp_snapdragon.py", ["submit"]),
    ("sigil_t4_benchmark.py", ["--selftest"]),
    ("roofline.py", ["book"]),
    ("roofline.py", ["law"]),
    ("roofline.py", ["bcrit"]),
    ("roofline.py", ["context"]),
    ("roofline.py", ["engines"]),
    ("roofline.py", ["predict"]),
    ("roofline.py", ["anomalies"]),
    ("aihub_workbench.py", ["plan"]),
    ("aihub_workbench.py", ["run"]),
    ("aihub_workbench.py", ["verify"]),
]


# Directory names that are never ours, however deep. A virtualenv created as
# D:\sigil\.venv -- which is what the QUICKSTART tells people to do -- puts
# ~2,900 third-party .py files INSIDE the project tree. A blind rglob then
# audited all of numpy, torch and setuptools for tabs and non-ASCII, produced
# ~8,620 checks instead of 72, and failed hundreds of them. The user who hit
# that saw one red line and no way to tell it was not their fault.
#
# Hidden directories are pruned by rule (a leading dot), the rest by name.
VENDOR_DIRS = {
    "__pycache__", "node_modules", "site-packages", "dist-packages",
    "venv", "env", "build", "dist", "wheels", "egg-info", "htmlcov",
    ".venv", ".git", ".tox", ".mypy_cache", ".pytest_cache", ".idea",
    "workbench_build",
}


PACKAGE_DIRS = ("sigil",)


def _project_py_files(here: pathlib.Path) -> List[pathlib.Path]:
    """
    Every .py file that is actually part of this project: the top level and
    the packages it ships, nothing else.

    This was a deny-list, and a deny-list is only as good as its guess about
    what users keep next to the project. The previous RUNBOOK told people to
    copy their old scripts into D:\\sigil\\backup_old -- which then got audited,
    and once the network phase existed, a backed-up engine that still emitted
    the old download commands would have failed a clean install. Anything not
    shipped here is not ours to judge, so only what ships is walked.
    """
    out: List[pathlib.Path] = sorted(here.glob("*.py"))
    for pkg in PACKAGE_DIRS:
        for f in sorted((here / pkg).rglob("*.py")):
            try:
                parts = f.relative_to(here).parts[:-1]
            except ValueError:                            # pragma: no cover
                continue
            if any(p in VENDOR_DIRS or p.startswith(".") or p.endswith(".egg-info")
                   for p in parts):
                continue
            out.append(f)
    return out


def _own_traceback(text: str, here: pathlib.Path) -> bool:
    """
    True only for a traceback raised inside OUR code.

    Scanning the whole of stdout+stderr for the word "Traceback" is too blunt.
    A broken distutils-precedence.pth, a deprecation shim, or any .pth loader
    error prints a full traceback during interpreter startup, before our script
    runs at all -- and that turned 23 passing CLI checks into 23 failures on an
    interpreter whose site-packages happened to be untidy. A traceback counts
    against us only if one of its frames names a file in this directory.
    """
    root = str(here).lower()
    blocks = text.split("Traceback (most recent call last)")
    for block in blocks[1:]:
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("File "):
                continue
            if root in line.lower():
                return True
    return False


def phase_portability(rep: Report, quick: bool) -> None:
    here = pathlib.Path(__file__).resolve().parent
    files = _project_py_files(here)
    # If this number is ever in the thousands, the scan has escaped the
    # project and every result below it is noise.
    rep.add("portability", "the scan stayed inside the project",
            0 < len(files) < 200, f"{len(files)} .py files")
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        bad = [i + 1 for i, ln in enumerate(text.splitlines())
               if any(ord(c) > 127 for c in ln)]
        rep.add("portability", f"{f.name} is pure ASCII", not bad,
                f"lines {bad[:6]}" if bad else "")
        rep.add("portability", f"{f.name} has no tabs", "\t" not in text)
        ok, why = _compiles(text, str(f))
        rep.add("portability", f"{f.name} compiles", ok, why)
    if quick:
        return
    for script, argv in CLI_SMOKE:
        label = f"{script} {' '.join(argv)}"
        if not (here / script).exists():
            rep.add("cli", label, False, "missing file")
            continue
        try:
            out = subprocess.run([sys.executable, str(here / script)] + argv,
                                 capture_output=True, text=True, timeout=600,
                                 cwd=str(here))
        except subprocess.TimeoutExpired:
            rep.add("cli", label, False, "timeout")
            continue
        text = out.stdout + out.stderr
        skip = _optional_dep_skip(script, out.returncode, text)
        if skip:
            rep.add("cli", label, True, skip)
            continue
        ok = out.returncode == 0 and not _own_traceback(text, here)
        rep.add("cli", label, ok,
                ("exit " + str(out.returncode)) if not ok else "")


# A module that refuses to run because a heavy optional dependency is absent
# is not broken -- it is doing the right thing, loudly. Treating that as a
# failure is the same mistake that made hp_snapdragon.py report 51 tests on
# one machine and 46 on another. The skip has to be recognised by the message
# the module itself prints, not guessed from an exit code alone.
OPTIONAL_DEP_SKIPS = [
    ("sigil_t4_benchmark.py", 2, "torch is required",
     "skipped: torch not installed"),
]


def _optional_dep_skip(script: str, rc: int, text: str) -> str:
    for want_script, want_rc, needle, note in OPTIONAL_DEP_SKIPS:
        if script == want_script and rc == want_rc and needle in text:
            return note
    return ""


def phase_network(rep: Report, mods: Dict[str, Any]) -> None:
    """
    Nothing in this project may download a model onto the machine it runs on.

    A user ran a command this project emitted and watched gigabytes arrive on
    a laptop that only needed a latency number. The fix removed the commands;
    this phase is what keeps them removed. It reads the SYNTAX TREE of every
    project file (a text grep trips over the docstrings that explain the
    history) and then renders every command the project emits and checks
    those too, because a command is data until someone pastes it.
    """
    W = mods.get("aihub_workbench")
    if W is None:
        rep.add("network", "aihub_workbench imports (it owns the scanner)", False)
        return
    here = pathlib.Path(__file__).resolve().parent
    all_waivers = []
    for f in _project_py_files(here):
        waived: List[Any] = []
        try:
            hits = W.download_calls_in(f.read_text(encoding="utf-8"), waived)
        except SyntaxError as exc:
            rep.add("network", f"{f.name}: parses", False, str(exc))
            continue
        all_waivers += [(f.name,) + tuple(w) for w in waived]
        rep.add("network", f"{f.name}: nothing downloads a model", not hits,
                "; ".join(f"line {ln}: {what}" for ln, what in hits[:3]))
    rep.add("network", "every waiver states its reason",
            all(w[3] for w in all_waivers), [w for w in all_waivers if not w[3]])

    banned = W.BANNED_COMMANDS
    emitted: List[Tuple[str, str]] = []
    E = mods.get("snapdragon_engine")
    if E:
        for key, spec in E.MODEL_DB.items():
            try:
                for kind, text in E.ExportGenerator.all(
                        spec, "Snapdragon X Plus 8-Core CRD", "tl1").items():
                    emitted.append((f"engine {key}/{kind}", text))
            except Exception as exc:
                emitted.append((f"engine {key}", f"RAISED {type(exc).__name__}"))
    HP = mods.get("hp_snapdragon")
    RL = mods.get("roofline")
    if HP and RL:
        for key in list(RL.ARCHS) + ["gemma_4_e2b_it"]:
            c = HP.aihub_commands(key, "Snapdragon X Plus 8-Core CRD")
            for sec, lines in c.items():
                if isinstance(lines, list):
                    emitted += [(f"hp {key}/{sec}", ln) for ln in lines]
    try:
        import sigil.npu as _npu
        emitted += [("sigil.npu aihub", _npu.aihub_export_snippet()),
                    ("sigil.npu geniex", _npu.geniex_snippet())]
    except Exception:
        pass
    bad = [w for w, t in emitted if any(b in t for b in banned)]
    rep.add("network", f"no emitted command fetches or exports a model "
                       f"({len(emitted)} rendered)", not bad, bad[:3])
    unsized = [w for w, t in emitted
               if any(c in t for c in W.SIZED_COMMANDS) and W.SIZE_MARKER not in t]
    rep.add("network", "every emitted on-device command states its download size",
            not unsized, unsized[:3])

    # Documents are commands too: a line in a README is a line someone pastes.
    # Each paragraph is judged on its own, so a size stated three screens away
    # does not excuse a command. PROVENANCE.md is the one file left out -- it is
    # the dated record of what this project got wrong, and it quotes withdrawn
    # commands verbatim so each correction can be checked against the original.
    docs = sorted(f for pat in ("*.md", "*.js", "*.ps1") for f in here.glob(pat)
                  if f.name != "PROVENANCE.md")
    doc_banned: List[str] = []
    doc_unsized: List[str] = []
    for d in docs:
        start, para = 1, []
        lines = d.read_text(encoding="utf-8").splitlines() + [""]
        for n, line in enumerate(lines, 1):
            if line.strip():
                if not para:
                    start = n
                para.append(line)
                continue
            block = "\n".join(para)
            para = []
            if any(b in block for b in banned):
                doc_banned.append(f"{d.name}:{start}")
            if (any(c in block for c in W.SIZED_COMMANDS)
                    and W.SIZE_MARKER not in block):
                doc_unsized.append(f"{d.name}:{start}")
    rep.add("network", f"no document tells you to fetch or export a model "
                       f"({len(docs)} read)", bool(docs) and not doc_banned,
            doc_banned[:3] or ("no documents found" if not docs else ""))
    rep.add("network", "every on-device command in a document states its "
                       "download size", not doc_unsized, doc_unsized[:3])
    rep.add("network", "the Workbench runner's download ceiling is profiles-only",
            W.MAX_DOWNLOAD_BYTES <= 2 * 1024 * 1024)


def _compiles(text: str, name: str) -> Tuple[bool, str]:
    """
    Compile, and treat a SyntaxWarning as a failure.

    Plain compile() raises only on SyntaxError, so it happily accepted a
    module docstring that drew under-braces with \\_ ... _/. Python 3.12 warns
    on that invalid escape sequence and 3.15 turns it into a SyntaxError --
    a file that imports today and stops importing on the next interpreter the
    judge happens to have. The warning IS the defect, so it is reported as one.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            compile(text, name, "exec")
        except SyntaxError as exc:
            return False, f"SyntaxError line {exc.lineno}"
        bad = [w for w in caught
               if issubclass(w.category, (SyntaxWarning, DeprecationWarning))]
        if bad:
            w = bad[0]
            return False, f"{w.category.__name__} line {w.lineno}: {w.message}"
    return True, ""


# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Try to break every module in the project.")
    p.add_argument("--quick", action="store_true",
                   help="skip the module suites and the CLI smoke tests")
    p.add_argument("--only", default="",
                   choices=["", "import", "suites", "fuzz", "numeric",
                            "determinism", "portability", "network"])
    p.add_argument("--fuzz-budget", type=int, default=900,
                   help="argument combinations per multi-argument function")
    args = p.parse_args(argv)

    rep = Report()
    run = (lambda ph: not args.only or args.only == ph)

    print()
    print("  SIGIL-Edge stress harness")
    print("  no GPU, no network, no side effects")
    print()

    mods = phase_import(rep) if run("import") else {}
    if not mods and args.only not in ("", "import"):
        mods = phase_import(Report())          # needed by later phases
    print(f"  [1/7] import        {len(mods)} modules "
          f"({len(MODULES)} required"
          + (", sigil/ absent" if len(mods) <= len(MODULES) else
             f" + {len(SUBMODULES)} from sigil/") + ")")

    if run("suites"):
        phase_suites(rep, args.quick)
    print("  [2/7] suites        done")

    if run("fuzz"):
        phase_fuzz(rep, mods, args.fuzz_budget)
    print("  [3/7] fuzz          done")

    if run("numeric"):
        phase_numeric(rep, mods)
        phase_numeric_roofline(rep, mods)
    print("  [4/7] numeric       done")

    if run("determinism"):
        phase_determinism(rep, mods)
    print("  [5/7] determinism   done")

    if run("portability"):
        phase_portability(rep, args.quick)
    print("  [6/7] portability   done")

    if run("network"):
        phase_network(rep, mods)
    print("  [7/7] network       done")

    return rep.render()


if __name__ == "__main__":
    sys.exit(main())
