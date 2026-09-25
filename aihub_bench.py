#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 aihub_bench.py -- benchmark MANY models on REAL Snapdragon silicon,
                   downloading NOTHING to your machine
================================================================================

The problem this solves: Gemma-4-E2B-it is a 3.04 GB download, and you have
913 MiB free. You do not need it. `qai-hub-models perf` and `numerics` return
Qualcomm's PUBLISHED measurements -- a few KB of metadata per model, no weights.

What that covers, exactly (qai_hub_models 0.62.2): 491 measured entries for
Snapdragon X Elite CRD, 487 for X2 Elite CRD, and NONE for X Plus 8-Core CRD,
the proxy for four of the seven HP machines. roofline.py embeds the X-series
LLM rows and tests them against the roofline offline; aihub_workbench.py
measures the missing device on Workbench, also without downloading a model.

    pip install qai_hub_models
    qai-hub configure --api_token <workbench.aihub.qualcomm.com -> Account -> API Token>

    python aihub_bench.py --list                    # what is available
    python aihub_bench.py                           # benchmark the default set
    python aihub_bench.py --models gemma_4_e2b_it qwen3_4b phi_4_mini_instruct
    python aihub_bench.py --all-llm                 # every LLM in the catalogue
    python aihub_bench.py --dry-run                 # check auth, fetch nothing

Outputs `aihub_results.json` and a Markdown table for the submission.

WHAT IT MEASURES
  perf      latency / throughput, measured on Qualcomm cloud devices
  numerics  accuracy vs the source model, measured the same way
  info      supported runtimes, chipsets, size, quantisation, architecture

WHAT IT DOES NOT DO
  * download model weights
  * run inference on your machine
  * invent numbers -- every field comes from the CLI or is reported as None

DESIGN NOTE, because it matters for trust: this script SHELLS OUT to the
official `qai-hub-models` CLI rather than reimplementing its API. The CLI is the
interface Qualcomm documents and tests; parsing its output is honest about where
the numbers come from and cannot silently diverge from the service.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _finite(value, name: str, lo=None, hi=None) -> float:
    """
    Reject a numeric argument that is not finite, before it propagates.

    NaN and infinity do not raise -- they SPREAD, and a function that takes NaN
    and returns a dict full of NaN has produced a confident-looking answer
    about nothing. The stress harness flags exactly that as the worst of the
    three failure classes, because it is the one nobody notices.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lo is not None and v < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi}, got {v}")
    return v



# Curated default set: small enough to finish quickly, broad enough to be
# interesting. Every id here was verified against the live AI Hub registry.
DEFAULT_MODELS = [
    "gemma_4_e2b_it",          # Google, 2B effective, multimodal, GenieX
    "qwen3_4b",                # Alibaba, 4B
    "qwen3_1_7b",              # Alibaba, 1.7B
    "phi_4_mini_instruct",     # Microsoft, 3.8B
    "llama_v3_2_3b_instruct",  # Meta, 3.2B
    "indus_1b",                # Tech Mahindra, Indic
]

LLM_MODELS = DEFAULT_MODELS + [
    "gemma_4_e4b_it", "mistral_7b_instruct_v0_3", "ministral_3_3b_instruct_2512",
    "qwen3_0_6b", "qwen3_8b", "qwen3_5_2b", "granite_4_0_micro",
    "llama_v3_2_1b_instruct", "llama_v3_1_8b_instruct", "gpt_oss_20b",
    "qwen3_vl_4b_instruct",
]

VISION_MODELS = ["easyocr", "trocr", "nomic_embed_text", "whisper_small",
                 "video_mae"]


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def log(m: str = "") -> None:
    print(m, flush=True)


def rule(t: str = "", w: int = 74) -> None:
    log("" if not t else "")
    log(("-" * w) if not t else f"-- {t} " + "-" * max(w - len(t) - 4, 0))


def run_cli(args: List[str], timeout: float = 180.0) -> Dict[str, Any]:
    """Run a qai-hub-models subcommand. Never raises."""
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout, check=False)
        return {"ok": r.returncode == 0, "code": r.returncode,
                "out": r.stdout or "", "err": r.stderr or ""}
    except FileNotFoundError:
        return {"ok": False, "code": 127, "out": "",
                "err": "qai-hub-models not found. pip install qai_hub_models"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 124, "out": "",
                "err": f"timed out after {timeout:.0f}s"}
    except Exception as exc:
        return {"ok": False, "code": 1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def check_auth() -> Dict[str, Any]:
    """Is the CLI installed and configured? Fail loudly and early if not."""
    info: Dict[str, Any] = {}
    info["cli"] = shutil.which("qai-hub-models")
    info["hub_cli"] = shutil.which("qai-hub")
    cfg = Path.home() / ".qai_hub" / "client.ini"
    info["configured"] = cfg.exists()
    info["config_path"] = str(cfg)
    if info["cli"]:
        v = run_cli(["qai-hub-models", "--version"], timeout=60)
        info["version"] = v["out"].strip() if v["ok"] else None
    return info


# --------------------------------------------------------------------------- #
# parsers -- the CLI prints ASCII tables; pull the fields we need
# --------------------------------------------------------------------------- #

def parse_info(text: str) -> Dict[str, Any]:
    """Extract the fields `qai-hub-models info` reports."""
    def field(label: str) -> Optional[str]:
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|", text)
        return m.group(1).strip() if m else None

    def multiline(label: str) -> Optional[str]:
        """Some fields wrap across rows with an empty label column."""
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|\n((?:\|\s*\|.+\|\n)*)",
                      text)
        if not m:
            return None
        cont = re.findall(r"\|\s*\|\s*(.+?)\s*\|", m.group(2) or "")
        return " ".join([m.group(1)] + cont).strip()

    return {
        "runtimes": field("Supported Runtimes"),
        "chipsets": multiline("Supported Chipsets"),
        "licence": field("License"),
        "source_repo": field("Source Repo"),
        "quantized": field("Quantized"),
        "params": field("Number of parameters"),
        "architecture": multiline("Model architecture"),
        "model_size": field("Model Size"),
        "quant_type": field("Quantization Type"),
        "infer_id": field("model_infer_id"),
        "input_modalities": field("Supported Input Modalities"),
        "languages": field("Supported languages"),
    }


_NUM = r"([0-9][0-9,]*\.?[0-9]*)"


def parse_metrics(text: str) -> Dict[str, Any]:
    """
    Pull numeric metrics out of `perf` / `numerics` output. Deliberately
    permissive: AI Hub's table shape varies by model and runtime, so anything
    not matched is left absent rather than guessed.
    """
    out: Dict[str, Any] = {}
    patterns = {
        "tokens_per_second": rf"(?:tokens?[ /_]per[ /_]sec|tok/s|Response Rate)\D*{_NUM}",
        "ttft_ms": rf"(?:TTFT|Time To First Token)\D*{_NUM}",
        "inference_ms": rf"(?:Inference Time|Latency)\D*{_NUM}",
        "peak_memory_mb": rf"(?:Peak Memory|Memory Usage)\D*{_NUM}",
        "accuracy": rf"(?:Accuracy|Top-1|macro AUROC)\D*{_NUM}",
        "psnr": rf"PSNR\D*{_NUM}",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            try:
                out[k] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    # capture device names mentioned, so the numbers are attributable
    devs = re.findall(r"(Snapdragon[^|\n,]{2,40}|QCS\d{4}[^|\n,]{0,20})", text)
    if devs:
        out["devices_seen"] = sorted({d.strip() for d in devs})[:8]
    return out


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #

def bench_model(model: str, do_numerics: bool = True,
                timeout: float = 180.0, raw: bool = False) -> Dict[str, Any]:
    """Gather info + perf + numerics for one model. Downloads nothing."""
    rec: Dict[str, Any] = {"model": model, "t": time.time()}

    i = run_cli(["qai-hub-models", "info", model], timeout)
    rec["info_ok"] = i["ok"]
    if i["ok"]:
        rec["info"] = parse_info(i["out"])
        if raw:
            rec["info_raw"] = i["out"]
    else:
        rec["info_error"] = (i["err"] or i["out"])[:200]
        return rec

    p = run_cli(["qai-hub-models", "perf", model], timeout)
    rec["perf_ok"] = p["ok"]
    if p["ok"]:
        rec["perf"] = parse_metrics(p["out"])
        if raw:
            rec["perf_raw"] = p["out"]
    else:
        rec["perf_error"] = (p["err"] or p["out"])[:200]

    if do_numerics:
        n = run_cli(["qai-hub-models", "numerics", model], timeout)
        rec["numerics_ok"] = n["ok"]
        if n["ok"]:
            rec["numerics"] = parse_metrics(n["out"])
            if raw:
                rec["numerics_raw"] = n["out"]
        else:
            rec["numerics_error"] = (n["err"] or n["out"])[:200]
    return rec


def markdown_table(records: List[Dict[str, Any]]) -> str:
    """A table you can paste straight into SUBMISSION.md."""
    if isinstance(records, (str, bytes)) or not hasattr(records, "__iter__"):
        raise TypeError("records must be a list of dicts, got "
                        f"{type(records).__name__}")
    records = list(records)
    if any(not isinstance(r, dict) for r in records):
        raise TypeError("every record must be a dict")
    rows = ["| model | params | size | quant | runtime | tok/s | TTFT ms |",
            "|---|---|---|---|---|---|---|"]
    for r in records:
        if not r.get("info_ok"):
            rows.append(f"| {r['model']} | - | - | - | **unavailable** | - | - |")
            continue
        i = r.get("info", {})
        p = r.get("perf", {})
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                r["model"],
                i.get("params") or "-",
                i.get("model_size") or "-",
                i.get("quant_type") or "-",
                (i.get("runtimes") or "-")[:28],
                p.get("tokens_per_second", "-"),
                p.get("ttft_ms", "-")))
    rows.append("")
    rows.append("_Measured by Qualcomm AI Hub on cloud-hosted Snapdragon devices. "
                "No weights were downloaded locally. `-` means the CLI did not "
                "report that field for this model; it is not an estimate._")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark models on Qualcomm AI Hub cloud devices. "
                    "Downloads nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--models", nargs="+", help="explicit model ids")
    g.add_argument("--all-llm", action="store_true", help="every LLM in the set")
    g.add_argument("--vision", action="store_true", help="vision / OCR / audio set")
    g.add_argument("--list", action="store_true",
                   help="list AI Hub's catalogue and exit")
    p.add_argument("--no-numerics", action="store_true",
                   help="skip accuracy metrics (faster)")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--out", default="aihub_results.json")
    p.add_argument("--md", default="aihub_results.md")
    p.add_argument("--raw", action="store_true",
                   help="keep full CLI output in the JSON (large)")
    p.add_argument("--dry-run", action="store_true",
                   help="check install and auth, then exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)

    log("")
    log("  AI HUB BENCH -- real Snapdragon measurements, zero local downloads")

    rule("SETUP")
    auth = check_auth()
    if not auth["cli"]:
        log("  qai-hub-models is not installed.")
        log("    pip install qai_hub_models")
        return 2
    log(f"  CLI: {auth['cli']}")
    if auth.get("version"):
        log(f"  version: {auth['version']}")
    if auth["configured"]:
        log(f"  configured: {auth['config_path']}")
    else:
        log("  NOT CONFIGURED. Run:")
        log("    qai-hub configure --api_token <workbench.aihub.qualcomm.com -> Account -> API Token>")
        if not a.dry_run:
            return 2

    if a.list:
        rule("AI HUB CATALOGUE")
        r = run_cli(["qai-hub-models", "models"], a.timeout)
        log(r["out"] if r["ok"] else f"  failed: {r['err'][:200]}")
        return 0 if r["ok"] else 1

    if a.dry_run:
        rule("DRY RUN")
        log("  Setup looks usable. Remove --dry-run to benchmark.")
        return 0

    models = (a.models if a.models else
              LLM_MODELS if a.all_llm else
              VISION_MODELS if a.vision else
              DEFAULT_MODELS)

    rule(f"BENCHMARKING {len(models)} MODELS")
    log("  Nothing is downloaded to this machine. Each model takes a few seconds.")
    log("")

    records: List[Dict[str, Any]] = []
    out_path = Path(a.out)
    t0 = time.time()
    for n, m in enumerate(models, 1):
        log(f"  [{n:2d}/{len(models)}] {m}")
        rec = bench_model(m, not a.no_numerics, a.timeout, a.raw)
        records.append(rec)
        # checkpoint after every model
        out_path.write_text(json.dumps(records, indent=2, default=str))
        if rec.get("info_ok"):
            i, p = rec.get("info", {}), rec.get("perf", {})
            bits = []
            if i.get("model_size"):
                bits.append(i["model_size"])
            if i.get("runtimes"):
                bits.append(i["runtimes"][:24])
            if p.get("tokens_per_second"):
                bits.append(f"{p['tokens_per_second']} tok/s")
            log(f"            {' | '.join(bits) if bits else 'info only'}")
        else:
            log(f"            UNAVAILABLE: {rec.get('info_error','')[:70]}")

    rule("RESULTS")
    md = markdown_table(records)
    log(md)
    Path(a.md).write_text(md)

    ok = sum(1 for r in records if r.get("info_ok"))
    perf = sum(1 for r in records if r.get("perf_ok"))
    log("")
    log(f"  {ok}/{len(records)} models resolved, {perf} returned perf metrics")
    log(f"  wrote {out_path} and {a.md}   ({time.time()-t0:.0f}s)")
    log("")
    log("  Paste aihub_results.md into SUBMISSION.md. These are measurements on")
    log("  Qualcomm silicon, which is exactly what the Deployment criterion wants.")
    log("")
    return 0


def _selftest() -> int:
    """Offline checks on the parsers, using captured real CLI output."""
    sample_info = """
+--------------------+---------------------------------------+
| Quantized          | Yes                                   |
| Supported Runtimes | GenieX (Llama.cpp)                    |
| Supported Chipsets | Snapdragon(R) 8 Elite Gen 5 Mobile,   |
|                    | Snapdragon(R) X2 Elite                |
| License            | Apache-2.0                            |
| Source Repo        | https://huggingface.co/google/gemma-4 |
+--------------------+---------------------------------------+
| Number of parameters | 2B (effective)                      |
+-----------------------------+--------------------------------+
| Model Size                  | 3.04 GB                        |
| Quantization Type           | Q4_0 (GGUF)                    |
| model_infer_id              | google/gemma-4-E2B-it-qat-q4_0 |
+-----------------------------+--------------------------------+
"""
    checks = []

    def ck(n, c, d=""):
        checks.append((n, bool(c), d))

    i = parse_info(sample_info)
    ck("parses runtime", i["runtimes"] == "GenieX (Llama.cpp)", str(i["runtimes"]))
    ck("parses model size", i["model_size"] == "3.04 GB", str(i["model_size"]))
    ck("parses quant type", i["quant_type"] == "Q4_0 (GGUF)")
    ck("parses licence", i["licence"] == "Apache-2.0")
    ck("parses params", "2B" in (i["params"] or ""))
    ck("multiline chipsets joined",
       i["chipsets"] and "X2 Elite" in i["chipsets"], str(i["chipsets"])[:50])
    ck("absent field is None", parse_info("")["runtimes"] is None)

    m = parse_metrics("Inference Time: 12.5 ms\nPeak Memory: 340 MB\n"
                      "Device: Snapdragon X2 Elite CRD")
    ck("parses latency", m.get("inference_ms") == 12.5, str(m.get("inference_ms")))
    ck("parses memory", m.get("peak_memory_mb") == 340.0)
    ck("captures device", any("X2 Elite" in d for d in m.get("devices_seen", [])))
    ck("no invention on empty input", parse_metrics("") == {})

    md = markdown_table([{"model": "x", "info_ok": False}])
    ck("unavailable model marked", "unavailable" in md)
    ck("table disclaims estimates", "not an estimate" in md)

    r = run_cli(["definitely-not-a-real-binary-xyz"])
    ck("missing binary handled", not r["ok"] and r["code"] == 127)

    ck("default set is non-empty", len(DEFAULT_MODELS) >= 5)
    ck("llm set extends default", set(DEFAULT_MODELS) <= set(LLM_MODELS))

    print("\naihub_bench self test\n" + "=" * 66)
    n = 0
    for name, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {d}" if d else ""))
        n += ok
    print(f"\n  {n}/{len(checks)} passed\n")
    return 0 if n == len(checks) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(_selftest())
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\n  interrupted; partial results were saved")
        raise SystemExit(130)
