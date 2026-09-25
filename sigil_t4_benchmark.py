#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 sigil_t4_benchmark.py -- real-weights KV-quantisation benchmark
================================================================================

Replaces the Colab notebook. Runs anywhere with a CUDA GPU: Colab, a rented box,
or your own machine. ~75-110 min on a free-tier T4.

    pip install -U "transformers>=4.44" "datasets>=2.20" accelerate sentencepiece

    python sigil_t4_benchmark.py --selftest   # 117 checks, CPU, no network, <30s
    python sigil_t4_benchmark.py --stress     # 9 attention geometries, no network
    python sigil_t4_benchmark.py --dry-run    # check the environment, no download
    python sigil_t4_benchmark.py --quick      # ~15 min smoke run
    python sigil_t4_benchmark.py             # the full job, Qwen2.5-0.5B

RUN --selftest FIRST. It takes half a minute, needs no GPU and no network, and
it is the thing that was missing when this script broke twice on Colab.

RUNS ANYWHERE
    --device  auto | cpu | cuda | mps | xpu      probed, in that order
    --dtype   auto | fp32 | fp16 | bf16          bf16 where it exists, fp16 on
                                                 pre-Ampere, fp32 on CPU
    --load-in-4bit                               4x more model on the same card
    --text-file mytext.txt                       no Hub, no `datasets`, offline
Apple Silicon, Intel XPU, ROCm, a CPU-only box and a T4 all work. The weights
are size-checked BEFORE loading, so a model too large to fit says so instead
of dying halfway through a download.

ON COLAB: upload this file, then in a cell:
    !pip install -q -U transformers datasets accelerate sentencepiece
    !python sigil_t4_benchmark.py --selftest && python sigil_t4_benchmark.py
Set Runtime -> Change runtime type -> T4 GPU first.

WHAT IT MEASURES
  1. FP16 baseline perplexity on WikiText-2
  2. An 18-config KV-quantisation sweep: bits {4,3,2} x rotation {off,Hadamard}
     x scale granularity {per-token, group-128, group-32}
  3. Exact-fold verification on real weights: q^T k is invariant under an
     orthogonal rotation of both, so folding is free at inference
  4. Decode throughput and the KV-cache memory budget against real bandwidth

The headline this project expects: SCALE GRANULARITY DOMINATES ROTATION CHOICE.
Grouped scales close most of the gap rotation is usually credited with. Report
whatever you measure, including if it contradicts that.

RESUMABLE. Results are written after every config, so a disconnect at config 15
does not lose configs 1-14. Re-running picks up where it stopped unless you pass
--no-resume.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _qmax(bits, lo: float = 1.0, hi: float = 64.0) -> float:
    """
    Quantiser level count, with the exponent VALIDATED.

    `2 ** (bits - 1)` on unvalidated input is a denial of service, not a
    rounding problem: given 10**30 Python will try to build an integer with
    10**30 bits and never come back. The stress harness found this by passing
    exactly that. Fractional widths (1.58 for ternary) are legal, so the bound
    is checked on the float rather than by casting to int.
    """
    try:
        b = float(bits)
    except (TypeError, ValueError):
        raise ValueError(f"bits must be a number, got {bits!r}")
    if not math.isfinite(b):
        raise ValueError(f"bits must be finite, got {b!r}")
    if not lo <= b <= hi:
        raise ValueError(f"bits must be in [{lo:g}, {hi:g}], got {b:g}")
    return max(2.0 ** (b - 1.0) - 1.0, 1.0)



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SIGIL-Edge real-weights KV-quantisation benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Outputs results_t4.json and kv_sweep.png in --out-dir.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model id. Gated models need --hf-token.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Benchmark SEVERAL models in one run. Each gets its own "
                        "results file. Overrides --model.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                   help="Hugging Face token for gated models.")
    p.add_argument("--dataset", default="",
                   help="Override the eval dataset, e.g. Salesforce/wikitext")
    p.add_argument("--dataset-config", default="",
                   help="Dataset config, e.g. wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text-file", default="",
                   help="Use a local plain-text file instead of the Hub.")
    p.add_argument("--eval-seqs", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--sweep-seqs", type=int, default=0,
                   help="Sequences per sweep config. 0 = half of --eval-seqs.")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore an existing results file and start over.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Run without a GPU. Very slow; for validation only.")
    p.add_argument("--quick", action="store_true",
                   help="Small, fast configuration (~15 min) for a smoke test.")
    p.add_argument("--dry-run", action="store_true",
                   help="Check imports, GPU and arguments, then exit.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps", "xpu"],
                   help="Force a backend. auto probes CUDA, then Apple MPS, "
                        "then Intel XPU, then CPU.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"],
                   help="Compute dtype. auto picks bf16 where supported, fp16 "
                        "on pre-Ampere GPUs, fp32 on CPU.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load weights in 4-bit (needs bitsandbytes). Lets a "
                        "model four times larger fit the same card.")
    p.add_argument("--force", action="store_true",
                   help="Start even when the pre-flight size check says the "
                        "weights will not fit.")
    p.add_argument("--allow-download", action="store_true",
                   help="Fetch model weights onto this machine even when they "
                        "exceed 1 GB and this is not a hosted notebook. Off by "
                        "default: this benchmark belongs on Colab.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the unit tests. CPU, no network, under 30s. "
                        "Run this before any long job.")
    p.add_argument("--stress", action="store_true",
                   help="Drive the real pipeline over a matrix of attention "
                        "geometries (MHA/GQA/MQA, odd head dims) on randomly "
                        "initialised models. CPU, no network.")
    return p


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str = "", width: int = 72) -> None:
    if not title:
        log("-" * width)
    else:
        log("\n" + f"-- {title} " + "-" * max(width - len(title) - 4, 0))


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

def check_env(args) -> Dict[str, Any]:
    rule("ENVIRONMENT")
    info: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch
    except ImportError:
        log("  torch is not installed.")
        log("  pip install -U torch transformers datasets accelerate sentencepiece")
        sys.exit(2)
    info["torch"] = torch.__version__
    log(f"  python {info['python']}   torch {info['torch']}")

    # `datasets` is only needed to fetch a corpus from the Hub. With
    # --text-file the Hub is never touched, so requiring it there locks the
    # script out of every offline and network-restricted machine.
    required = ["transformers"]
    optional = [] if getattr(args, "text_file", "") else ["datasets"]
    for mod in required + optional:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
            log(f"  {mod} {info[mod]}")
        except ImportError:
            log(f"  {mod} is not installed. pip install -U {mod}")
            sys.exit(2)
    if not optional:
        try:
            import datasets as _ds
            info["datasets"] = getattr(_ds, "__version__", "installed")
            log(f"  datasets {info['datasets']}  (unused: --text-file given)")
        except ImportError:
            log("  datasets not installed -- fine, --text-file bypasses the Hub")

    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 1024 ** 3, 1)
        log(f"  GPU: {p.name}, {info['gpu_mem_gb']} GiB")
    else:
        log("  No CUDA GPU detected.")
        if not args.allow_cpu and not args.dry_run:
            log("\n  On Colab: Runtime -> Change runtime type -> T4 GPU, then rerun.")
            log("  To proceed on CPU anyway (very slow): --allow-cpu")
            sys.exit(2)
        log("  Continuing on CPU (--allow-cpu). This will be extremely slow.")
    return info


# --------------------------------------------------------------------------- #
# DEVICE AND CONFIG PORTABILITY
#
# "Works on a T4" is not the same as "works". This block exists because the
# script has now failed twice on a machine that was not the one it was written
# on, and both failures were assumptions that happened to hold locally:
#
#   * dtype was pinned to fp16 the moment CUDA was seen, while detect_gpu()
#     computed supports_bf16 and nobody read it. On an Ampere+ card that threw
#     away the better dtype; the flag was dead code.
#   * every config read was cfg.<attr>, which is None on any multimodal model,
#     where the text tower lives under cfg.text_config. Bonsai 2, Gemma 3n and
#     Qwen3.5-VL all nest. The fold check would have crashed on all three.
#
# Everything below is probed. Nothing is assumed.
# --------------------------------------------------------------------------- #

def _pad_token_id(tok) -> int:
    """
    `eos_token_id` is an int on most models and a LIST on some newer ones
    (Qwen3, Llama 3.1 and anything with multiple stop tokens). Passing the list
    to generate() as pad_token_id raises deep inside the sampler.
    """
    for attr in ("pad_token_id", "eos_token_id"):
        v = getattr(tok, attr, None)
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if isinstance(v, int):
            return v
    return 0


def run_fingerprint(model_id: str, seq_len: int, shape: Dict[str, Any]) -> str:
    """
    Identify a results file so a resume cannot silently mix two different runs.
    Changing the model, the sequence length or the attention geometry
    invalidates every cached row.
    """
    import hashlib
    key = (f"{model_id}|{seq_len}|{shape.get('n_layers')}|"
           f"{shape.get('head_dim')}|{shape.get('n_kv_heads')}|"
           f"{shape.get('n_q_heads')}")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cfg_get(cfg, *names, default=None):
    """
    Read an attribute from a config that may nest its text tower.

    Multimodal configs put the language model under `text_config` (sometimes
    `llm_config` or `language_config`). Reading cfg.num_hidden_layers on one of
    those returns None and every downstream shape becomes wrong -- silently,
    which is worse than crashing.
    """
    holders = [cfg]
    for sub in ("text_config", "llm_config", "language_config", "decoder"):
        inner = getattr(cfg, sub, None)
        if inner is not None and inner is not cfg:
            holders.append(inner)
    for holder in holders:
        for name in names:
            val = getattr(holder, name, None)
            if val is not None:
                return val
    return default


def detect_accelerator(torch_mod) -> Dict[str, Any]:
    """
    Find the best available accelerator: CUDA, then Apple MPS, then Intel XPU,
    then CPU. Returns a uniform record whatever the backend, so no call site
    needs to know which one it got.
    """
    d: Dict[str, Any] = {
        "backend": "cpu", "name": "cpu", "device": "cpu",
        "total_gb": None, "free_gb": None, "capability": None,
        "supports_bf16": False, "supports_fp16_compute": False,
        "sm_count": None, "available": False,
    }
    # ---- CUDA / ROCm ---- #
    try:
        if torch_mod.cuda.is_available():
            props = torch_mod.cuda.get_device_properties(0)
            major = getattr(props, "major", 0)
            total = props.total_memory
            free = None
            try:
                free, total = torch_mod.cuda.mem_get_info()
            except Exception:
                # mem_get_info is missing on some ROCm and older builds.
                free = None
            is_rocm = bool(getattr(torch_mod.version, "hip", None))
            d.update({
                "backend": "rocm" if is_rocm else "cuda",
                "name": props.name, "device": "cuda", "available": True,
                "total_gb": round(total / 1024 ** 3, 2),
                "free_gb": (round(free / 1024 ** 3, 2) if free else None),
                "capability": f"{major}.{getattr(props, 'minor', 0)}",
                "sm_count": getattr(props, "multi_processor_count", None),
                # bf16 needs Ampere (sm_80). A T4 is 7.5 -> fp16 only. ROCm
                # gfx90a+ has bf16 but the capability numbers do not map, so
                # ask torch directly and fall back to the compute capability.
                "supports_bf16": _bf16_ok(torch_mod, major),
                "supports_fp16_compute": True,
            })
            return d
    except Exception:
        pass
    # ---- Apple Silicon ---- #
    try:
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is not None and mps.is_available() and mps.is_built():
            d.update({
                "backend": "mps", "name": "Apple MPS", "device": "mps",
                "available": True, "supports_bf16": False,
                "supports_fp16_compute": True,
                # MPS shares system RAM; there is no separate pool to query.
                "total_gb": _host_ram_gb(), "free_gb": None,
                "note": "unified memory: the GPU budget is host RAM",
            })
            return d
    except Exception:
        pass
    # ---- Intel XPU ---- #
    try:
        xpu = getattr(torch_mod, "xpu", None)
        if xpu is not None and xpu.is_available():
            d.update({"backend": "xpu", "name": "Intel XPU", "device": "xpu",
                      "available": True, "supports_bf16": True,
                      "supports_fp16_compute": True,
                      "total_gb": _host_ram_gb()})
            return d
    except Exception:
        pass
    d["total_gb"] = _host_ram_gb()
    return d


def _bf16_ok(torch_mod, major: int) -> bool:
    try:
        fn = getattr(torch_mod.cuda, "is_bf16_supported", None)
        if fn is not None:
            return bool(fn())
    except Exception:
        pass
    return major >= 8


def _host_ram_gb() -> Optional[float]:
    try:
        import os as _os
        n = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
        return round(n / 1024 ** 3, 2)
    except Exception:
        return None


def select_dtype(torch_mod, accel: Dict[str, Any], requested: str = "auto"):
    """
    Pick the compute dtype for this backend.

      auto  -> bf16 where supported, else fp16 on a GPU, else fp32
      fp32  -> always safe, always slow; the CPU default
      fp16  -> refused on CPU, where it is both slow and numerically poor

    bf16 is preferred over fp16 wherever it exists: the sweep divides by
    absmax scales, and fp16 overflows at 65504 while bf16 has fp32's range.
    """
    req = (requested or "auto").lower()
    if req == "fp32":
        return torch_mod.float32, "fp32 requested"
    if req == "bf16":
        if not accel["supports_bf16"]:
            return (torch_mod.float32 if not accel["available"]
                    else torch_mod.float16), \
                   "bf16 requested but unsupported here; downgraded"
        return torch_mod.bfloat16, "bf16 requested"
    if req == "fp16":
        if not accel["available"]:
            return torch_mod.float32, "fp16 on CPU is slow and imprecise; using fp32"
        return torch_mod.float16, "fp16 requested"
    # auto
    if not accel["available"]:
        return torch_mod.float32, "CPU: fp32"
    if accel["supports_bf16"]:
        return torch_mod.bfloat16, "bf16 available (wider range than fp16)"
    return torch_mod.float16, f"fp16 (backend {accel['backend']} lacks bf16)"


def estimate_params_from_config(cfg) -> Optional[float]:
    """
    Parameter count in billions, from config alone, BEFORE anything is loaded.

    This is what makes a fit check useful: the previous version checked whether
    the weights fit only after `.to(device)` had already OOM'd trying to put
    them there. Approximate is fine -- the decision it feeds is binary.
    """
    L = cfg_get(cfg, "num_hidden_layers", "n_layer", "num_layers")
    h = cfg_get(cfg, "hidden_size", "n_embd", "d_model")
    v = cfg_get(cfg, "vocab_size")
    if not (L and h):
        return None
    inter = cfg_get(cfg, "intermediate_size", "ffn_dim", default=4 * h)
    n_heads = cfg_get(cfg, "num_attention_heads", default=max(1, h // 64))
    n_kv = cfg_get(cfg, "num_key_value_heads", default=n_heads)
    head_dim = cfg_get(cfg, "head_dim", default=h // max(1, n_heads))
    attn = h * (n_heads * head_dim) + 2 * h * (n_kv * head_dim) + (n_heads * head_dim) * h
    mlp = 3 * h * inter                      # SwiGLU; 2*h*inter for plain MLP
    per_layer = attn + mlp
    embed = (v or 0) * h * (1 if cfg_get(cfg, "tie_word_embeddings", default=False)
                            else 2)
    return (L * per_layer + embed) / 1e9


# Environment markers of hosted notebooks, where a model download is the point.
CLOUD_ENV_MARKERS = ("COLAB_GPU", "COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE",
                     "SM_CURRENT_HOST", "PAPERSPACE_NOTEBOOK_REPO_ID",
                     "LIGHTNING_CLOUD_PROJECT_ID")
DOWNLOAD_LIMIT_GB = 1.0


def download_guard(ref: str, size_gb: Optional[float], allow: bool = False,
                   env: Optional[Dict[str, str]] = None,
                   limit_gb: float = DOWNLOAD_LIMIT_GB) -> Dict[str, Any]:
    """
    Should fetching `ref` onto THIS machine go ahead?

    This benchmark is written for a Colab T4. Run on a laptop, its first act
    was to download the model -- ~8 GB for Qwen3-4B -- which is exactly the
    complaint that produced this guard. It now goes ahead only for a local
    path, a hosted notebook, an explicit --allow-download, or a download known
    to be under 1 GB. A size that cannot be estimated counts as large.
    """
    env = dict(os.environ) if env is None else env
    if size_gb is not None:
        try:
            size_gb = float(size_gb)
        except (TypeError, ValueError):
            size_gb = None
        if size_gb is not None and (size_gb != size_gb or size_gb < 0
                                    or size_gb == float("inf")):
            size_gb = None
    local = bool(ref) and os.path.exists(str(ref))
    cloud = any(k in env for k in CLOUD_ENV_MARKERS)
    small = size_gb is not None and size_gb <= limit_gb
    ok = local or cloud or bool(allow) or small
    return {"ok": ok, "local": local, "cloud": cloud, "size_gb": size_gb,
            "reason": ("local path" if local else "hosted notebook" if cloud else
                       "--allow-download" if allow else
                       f"small (~{size_gb:.2f} GB)" if small else
                       (f"would download ~{size_gb:.1f} GB" if size_gb is not None
                        else "would download a model of unknown size")
                       + " to this machine")}


def fit_verdict(params_b: Optional[float], accel: Dict[str, Any],
                bytes_per_param: int = 2) -> Dict[str, Any]:
    """
    Will the weights alone fit, before we try? Returns advice, never an abort:
    the caller decides, and on unified-memory backends the answer is soft.
    """
    if params_b is None:
        return {"known": False, "fits": True, "advice": ""}
    need = params_b * bytes_per_param
    have = accel.get("free_gb") or accel.get("total_gb")
    if not have:
        return {"known": False, "fits": True, "weights_gb": round(need, 2),
                "advice": ""}
    budget = have * 0.80
    fits = need <= budget
    advice = "" if fits else (
        f"{need:.1f} GB of weights against a {budget:.1f} GB budget on "
        f"{accel['name']}. Options, cheapest first: --dtype fp16 if you are on "
        f"bf16; a smaller --model; --load-in-4bit if bitsandbytes is present; "
        f"or --device cpu to trade speed for room.")
    return {"known": True, "fits": fits, "weights_gb": round(need, 2),
            "budget_gb": round(budget, 2), "advice": advice}


# --------------------------------------------------------------------------- #
# AUTO-DETECTION -- probe the GPU and the model, then size the run to fit.
#
# The first two releases crashed because they ASSUMED things that were false on
# the user's actual model:
#   * assumed n_query_heads == n_kv_heads. Qwen2.5-0.5B is GQA: 14 vs 2. Crash.
#   * assumed group sizes were valid. group=128 with head_dim=64 padded a
#     non-contiguous tensor. Crash.
#   * assumed 24 sequences x 2048 tokens fits any GPU. On a smaller card it OOMs.
#
# Nothing here is assumed any more. Everything is probed and reported, and the
# run is sized to what the probe finds.
# --------------------------------------------------------------------------- #

def detect_gpu(torch_mod) -> Dict[str, Any]:
    """
    Back-compatible name. Delegates to detect_accelerator(), which also finds
    Apple MPS and Intel XPU and never raises on a backend that lacks
    mem_get_info.
    """
    return detect_accelerator(torch_mod)


def detect_model_shape(model, cfg) -> Dict[str, Any]:
    """
    Recover the attention geometry from the live model, not from assumptions.
    Reads the actual projection weights where the config is ambiguous.
    """
    # cfg_get, not getattr: on a multimodal config the language tower lives
    # under cfg.text_config and every one of these reads returns None.
    n_layers = cfg_get(cfg, "num_hidden_layers", "n_layer", "num_layers")
    n_heads = cfg_get(cfg, "num_attention_heads", "n_head")
    n_kv = cfg_get(cfg, "num_key_value_heads") or n_heads
    hidden = cfg_get(cfg, "hidden_size", "n_embd", "d_model")
    head_dim = cfg_get(cfg, "head_dim")

    # Verify against the real weights: some configs lie or omit head_dim.
    attns = [m for _, m in model.named_modules()
             if hasattr(m, "k_proj") and hasattr(m, "v_proj")
             and hasattr(m, "q_proj")]
    measured: Dict[str, Any] = {}
    if attns:
        a = attns[0]
        q_out = a.q_proj.weight.shape[0]
        k_out = a.k_proj.weight.shape[0]
        measured["q_proj_out"] = int(q_out)
        measured["k_proj_out"] = int(k_out)
        if head_dim is None and n_heads:
            head_dim = q_out // n_heads
        if head_dim:
            measured["n_q_heads_measured"] = int(q_out // head_dim)
            measured["n_kv_heads_measured"] = int(k_out // head_dim)

    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads

    n_q_real = measured.get("n_q_heads_measured", n_heads)
    n_kv_real = measured.get("n_kv_heads_measured", n_kv)
    gqa = (n_q_real // n_kv_real) if (n_q_real and n_kv_real) else 1

    return {
        "n_layers": n_layers, "hidden_size": hidden, "head_dim": head_dim,
        "n_q_heads": n_q_real, "n_kv_heads": n_kv_real,
        "gqa_ratio": gqa,
        "is_gqa": gqa > 1,
        "is_mqa": n_kv_real == 1 if n_kv_real else False,
        "n_attn_modules": len(attns),
        "measured": measured,
        "head_dim_is_pow2": bool(head_dim and (head_dim & (head_dim - 1)) == 0),
    }


def plan_run(gpu: Dict[str, Any], shape: Dict[str, Any], params_b: float,
             requested_seqs: int, requested_len: int) -> Dict[str, Any]:
    """
    Size the benchmark to the hardware actually present.

    Budget: weights + activations + KV. Leaves 25% headroom, because CUDA
    fragmentation makes a 95%-full card OOM unpredictably.
    """
    if not gpu["available"]:
        return {"eval_seqs": min(requested_seqs, 4),
                "seq_len": min(requested_len, 512),
                "reason": "CPU only -- reduced to keep the run finite",
                "adjusted": True}

    free = gpu["free_gb"] or gpu["total_gb"] or 8.0
    budget = free * 0.75

    w_gb = params_b * 2.0                      # fp16 weights
    hd, nkv, L = (shape.get("head_dim") or 64,
                  shape.get("n_kv_heads") or 8,
                  shape.get("n_layers") or 32)
    kv_per_tok = 2 * L * nkv * hd * 2 / 1024 ** 3
    # activations during a forward pass scale with seq_len
    act_per_tok = (shape.get("hidden_size") or 2048) * L * 2 * 8 / 1024 ** 3

    # The weights alone may exceed the budget, in which case NO seq_len helps.
    # An earlier version shrank seq_len to its floor and then returned a plan
    # whose own estimate exceeded the budget -- a plan that cannot work, offered
    # as though it could.
    if w_gb >= budget:
        return {
            "eval_seqs": requested_seqs, "seq_len": min(requested_len, 512),
            "weights_gb": round(w_gb, 2), "budget_gb": round(budget, 2),
            "est_peak_gb": None, "adjusted": True, "fits": False,
            "reason": (f"WILL NOT FIT: {w_gb:.1f} GB of fp16 weights alone "
                       f"exceeds the {budget:.1f} GB usable budget on "
                       f"{gpu.get('name','this GPU')}. No sequence length fixes "
                       f"this. Use a smaller model, or load in 8-bit/4-bit."),
        }

    seq_len = requested_len
    while seq_len > 128:
        if w_gb + (kv_per_tok + act_per_tok) * seq_len <= budget:
            break
        seq_len //= 2

    peak = w_gb + (kv_per_tok + act_per_tok) * seq_len
    fits = peak <= budget
    adjusted = seq_len != requested_len
    if not fits:
        reason = (f"TIGHT: even at seq_len={seq_len} the estimate is "
                  f"{peak:.1f} GB against a {budget:.1f} GB budget. Expect OOM; "
                  f"the sweep will catch it per-config and continue.")
    elif adjusted:
        reason = (f"seq_len reduced {requested_len} -> {seq_len} to fit "
                  f"{budget:.1f} GB of usable VRAM")
    else:
        reason = "requested configuration fits"
    return {
        "eval_seqs": requested_seqs, "seq_len": seq_len,
        "weights_gb": round(w_gb, 2), "budget_gb": round(budget, 2),
        "est_peak_gb": round(peak, 2), "adjusted": adjusted, "fits": fits,
        "reason": reason,
    }


def plan_sweep(shape: Dict[str, Any], seq_len: int, quick: bool) -> Dict[str, Any]:
    """
    Choose bit widths, rotations and group sizes that are VALID for this model.
    Invalid combinations are dropped with a stated reason rather than crashing.
    """
    hd = shape.get("head_dim") or 64
    bits = (4, 2) if quick else (4, 3, 2)
    raw = (None, 32) if quick else (None, 128, 64, 32, 16)

    groups, dropped = [], []
    for g in raw:
        if g is None:
            groups.append(None)
        elif g > min(seq_len, hd):
            dropped.append((g, f"exceeds min(seq_len={seq_len}, head_dim={hd})"))
        else:
            groups.append(g)

    rotate = shape.get("head_dim_is_pow2", False)
    notes = []
    if not rotate:
        notes.append(f"head_dim={hd} is not a power of two, so Hadamard "
                     f"rotation is unavailable; the granularity comparison "
                     f"still runs and is the finding that matters")
    if shape.get("is_gqa"):
        notes.append(f"GQA detected: {shape['n_q_heads']} query heads to "
                     f"{shape['n_kv_heads']} KV heads (ratio {shape['gqa_ratio']}). "
                     f"K will be repeated for the fold check.")
    for g, why in dropped:
        notes.append(f"dropped group={g}: {why}")

    return {"bits": bits, "groups": groups, "rotations": (False, True) if rotate
            else (False,), "dropped": dropped, "notes": notes,
            "n_configs": len(bits) * len(groups) * (2 if rotate else 1)}


# --------------------------------------------------------------------------- #
# quantisers and rotations
# --------------------------------------------------------------------------- #

def make_hadamard(n: int, torch_mod, seed: int = 0):
    """Randomised Hadamard: H @ diag(+-1). Requires n to be a power of two."""
    if n & (n - 1):
        raise ValueError(f"head_dim {n} is not a power of two; "
                         "rotation configs will be skipped")
    H = torch_mod.ones(1, 1, dtype=torch_mod.float64)
    while H.shape[0] < n:
        H = torch_mod.cat([torch_mod.cat([H, H], 1),
                           torch_mod.cat([H, -H], 1)], 0)
    H = H / math.sqrt(n)
    g = torch_mod.Generator().manual_seed(seed)
    s = torch_mod.randint(0, 2, (n,), generator=g, dtype=torch_mod.float64) * 2 - 1
    return H * s.unsqueeze(0)


def quant_kv(x, bits: int, is_key: bool, group, torch_mod):
    """
    Quantise a KV tensor of shape (B, S, H, D) using the KIVI convention.

    THIS IS THE CORRECTED VERSION. The first release reduced Keys over dim=-2 --
    the HEAD axis, which on a GQA model like Qwen2.5-0.5B has size 2. The scale
    was therefore the max of two values: absurdly fine granularity that no real
    deployment would use. Worse, when `group` was set the axis argument was
    ignored entirely and grouping ran along D instead, so the per-token and
    grouped rows of the sweep measured DIFFERENT SCHEMES and were not comparable.
    That is why grouping appeared 4x worse than per-token; it was an artefact.

    Correct convention:
        Keys   -> PER-CHANNEL: one scale per channel, reduced over the TOKEN axis
        Values -> PER-TOKEN:   one scale per token,   reduced over the CHANNEL axis
        group  -> subdivides THAT SAME axis into blocks, which is what makes the
                  granularity comparison meaningful.
    """
    if bits >= 16:
        return x
    qmax = _qmax(bits)
    B, S, H, D = x.shape
    axis = 1 if is_key else 3          # tokens for K, channels for V

    if not group:
        s_ = x.abs().amax(axis, keepdim=True).clamp_min(1e-8) / qmax
        return (x / s_).round().clamp(-qmax, qmax).mul(s_)

    n = x.shape[axis]
    if group >= n:                     # a group spanning the whole axis IS per-axis
        s_ = x.abs().amax(axis, keepdim=True).clamp_min(1e-8) / qmax
        return (x / s_).round().clamp(-qmax, qmax).mul(s_)

    pad = (-n) % group
    if pad:
        pad_spec = [0, 0] * (3 - axis) + [0, pad] if axis != 3 else [0, pad]
        xp = torch_mod.nn.functional.pad(x, tuple(pad_spec))
    else:
        xp = x

    # move the reduction axis last, block it, reduce, restore
    xm = xp.movedim(axis, -1).contiguous()
    lead = xm.shape[:-1]
    b = xm.reshape(*lead, -1, group)
    s_ = b.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (b / s_).round().clamp(-qmax, qmax).mul(s_).reshape(*lead, -1)
    out = q.movedim(-1, axis)
    if pad:
        out = out.narrow(axis, 0, n)
    return out.contiguous()


def valid_groups(n_tokens: int, head_dim: int, groups) -> list:
    """
    Keep only group sizes that mean something. A group larger than the axis it
    subdivides is just 'per-axis', and padding past the axis length was the
    source of the `view size is not compatible` crash.
    """
    out = []
    for g in groups:
        if g is None:
            out.append(None)
        elif g <= min(n_tokens, head_dim):
            out.append(g)
    return out



# --------------------------------------------------------------------------- #
# evaluation text
#
# datasets >= 4 requires the full `namespace/name` form, so the bare "wikitext"
# that worked for years now raises HfUriError. The canonical id is
# "Salesforce/wikitext". Several ids are tried in order so a single upstream
# rename cannot break the run again, and --text-file bypasses the Hub entirely.
# --------------------------------------------------------------------------- #

DATASET_CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),          # pre-4.x datasets
    ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    ("Salesforce/wikitext", "wikitext-2-v1"),
]


def load_eval_text(args, load_dataset_fn) -> str:
    """
    Return the evaluation corpus as one string. Tries, in order:
      1. --text-file, if given
      2. --dataset / --dataset-config, if given
      3. the known-good WikiText ids
    Raises with actionable guidance if everything fails.
    """
    if not args.text_file and load_dataset_fn is None:
        log("  `datasets` is not installed and no --text-file was given.")
        log("  Either: pip install -U datasets")
        log("      or: --text-file yourtext.txt   (any plain text file)")
        sys.exit(2)
    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            log(f"  --text-file {path} does not exist")
            sys.exit(2)
        txt = path.read_text(errors="replace")
        log(f"  using local text: {path} ({len(txt):,} chars)")
        return txt

    candidates = list(DATASET_CANDIDATES)
    if args.dataset:
        candidates.insert(0, (args.dataset, args.dataset_config or None))

    errors = []
    for name, conf in candidates:
        try:
            ds = (load_dataset_fn(name, conf, split=args.split) if conf
                  else load_dataset_fn(name, split=args.split))
            col = "text" if "text" in ds.column_names else ds.column_names[0]
            txt = "\n\n".join(str(t) for t in ds[col])
            log(f"  loaded {name} / {conf or '-'} [{args.split}] "
                f"-> {len(txt):,} chars from column '{col}'")
            return txt
        except Exception as exc:
            errors.append(f"    {name}/{conf or '-'}: "
                          f"{type(exc).__name__}: {str(exc)[:90]}")
            continue

    log("  Could not load any evaluation dataset. Tried:")
    for e in errors:
        log(e)
    log("")
    log("  Fixes, in order of ease:")
    log("    1. pip install -U datasets        (most likely fix)")
    log("    2. --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1")
    log("    3. --text-file mytext.txt         (any plain text, bypasses the Hub)")
    log("    4. --hf-token hf_xxx              (if rate-limited or gated)")
    sys.exit(2)


# --------------------------------------------------------------------------- #
# main benchmark
# --------------------------------------------------------------------------- #

def run(args) -> int:
    import contextlib
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None        # only reachable without --text-file

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results_t4.json"

    results: Dict[str, Any] = {}
    if res_path.exists() and not args.no_resume:
        try:
            results = json.loads(res_path.read_text())
            log(f"\n  Resuming from {res_path} "
                f"({len(results.get('kv_sweep', []))} configs already done)")
        except Exception:
            results = {}

    def save() -> None:
        res_path.write_text(json.dumps(results, indent=2, default=str))

    # Probe FIRST, then choose. The previous version pinned fp16 the moment it
    # saw CUDA and never read the bf16 flag it had just computed.
    accel = detect_accelerator(torch)
    if getattr(args, "device", "auto") not in ("auto", "", None):
        forced = args.device
        if forced == "cpu":
            accel = {**accel, "backend": "cpu", "device": "cpu", "name": "cpu",
                     "available": False, "supports_bf16": False}
        else:
            accel = {**accel, "device": forced}
    dev = accel["device"]
    dtype, dtype_why = select_dtype(torch, accel, getattr(args, "dtype", "auto"))
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if dev == "cpu":
        # Torch defaults to one thread in some containers; on a benchmark that
        # is the difference between 20 minutes and three hours.
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2)))
        except Exception:
            pass

    # ---- load model -------------------------------------------------------- #
    rule("MODEL")
    n_eval_req = args.eval_seqs
    model_id = args.model

    def load(mid):
        """
        transformers 5.x renamed `torch_dtype` to `dtype`. Try the new name
        first, fall back to the old one, then fall back to neither.
        """
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)  # download-ok: runs only after download_guard() passes
        # Cross the 4.x/5.x boundary: `torch_dtype` was renamed to `dtype`, and
        # `attn_implementation` is not universally accepted. Walk the matrix
        # from most-specific to least, catching TypeError (bad kwarg) AND
        # ValueError (kwarg accepted but value unsupported).
        extra = {}
        if getattr(args, "load_in_4bit", False):
            try:
                from transformers import BitsAndBytesConfig
                extra["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                extra["device_map"] = "auto"
                log("  4-bit weight loading enabled (bitsandbytes)")
            except Exception as exc:
                log(f"  --load-in-4bit unavailable ({type(exc).__name__}); "
                    f"continuing at full precision")
        attempts = []
        for attn in ({"attn_implementation": "eager"}, {}):
            for dt in ({"dtype": dtype}, {"torch_dtype": dtype}, {}):
                attempts.append({"trust_remote_code": True, **attn, **dt, **extra})
        last = None
        for kw in attempts:
            try:
                m = AutoModelForCausalLM.from_pretrained(mid, **kw)  # download-ok: runs only after download_guard() passes
                tag = ", ".join(k for k in kw if k != "trust_remote_code") or "defaults"
                log(f"  loaded with: {tag}")
                # device_map already placed the shards; moving again breaks it.
                if "device_map" not in kw:
                    m = m.to(dev)
                return m.eval(), tok
            except (TypeError, ValueError) as exc:
                last = exc
                continue
            except RuntimeError as exc:
                # Out of memory on placement is not a bad-kwarg problem, so
                # retrying the matrix cannot help. Say what to do instead.
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"{mid} does not fit on {accel['name']} at this dtype. "
                        f"Try --load-in-4bit, a smaller --model, or "
                        f"--device cpu.") from exc
                last = exc
                continue
        raise RuntimeError(f"could not load {mid}: {type(last).__name__}: {last}")

    log(f"  device: {accel['name']} [{accel['backend']}]   dtype: {dtype_why}")
    # Fit check BEFORE loading. Checking afterwards, as this script used to,
    # means the OOM has already happened by the time the check runs.
    _est = None
    try:
        from transformers import AutoConfig
        _pre = AutoConfig.from_pretrained(model_id, trust_remote_code=True)  # download-ok: config.json only, kilobytes, to size the real download first
        _est = estimate_params_from_config(_pre)
        _fit = fit_verdict(_est, accel,
                           2 if dtype in (torch.float16, torch.bfloat16) else 4)
        if _est:
            log(f"  estimated {_est:.2f}B params "
                f"(~{_fit.get('weights_gb', '?')} GB at this dtype)")
        if _fit.get("known") and not _fit["fits"]:
            log(f"  WILL NOT FIT: {_fit['advice']}")
            if not getattr(args, "force", False):
                log("  Refusing to start a run that cannot finish. "
                    "Pass --force to try anyway.")
                return 5
    except Exception as exc:
        log(f"  pre-flight size check skipped: {type(exc).__name__}")

    # bf16 checkpoint: 2 bytes per parameter is what a hub download transfers.
    _g = download_guard(model_id, None if _est is None else _est * 2.0,
                        getattr(args, "allow_download", False))
    if not _g["ok"]:
        log(f"  REFUSED: {model_id} {_g['reason']}.")
        log("  This benchmark is meant for a Colab T4: upload the script there.")
        log("  Latency on real Snapdragon needs no download at all:")
        log("      python aihub_workbench.py plan")
        log("  Already in your Hugging Face cache? --allow-download fetches nothing new.")
        return 6

    try:
        model, tok = load(model_id)
    except Exception as exc:
        log(f"  {model_id} failed to load: {type(exc).__name__}: {exc}")
        fallback = "Qwen/Qwen2.5-0.5B-Instruct"
        # Substituting a different model is only ever acceptable when the user
        # did not name one. In a --models sweep it would quietly benchmark the
        # wrong thing and put it in the comparison table under the right name.
        if getattr(args, "_in_sweep", False):
            log("  In a --models sweep: not substituting a fallback model.")
            return 3
        if model_id == fallback or args.model != build_parser().get_default("model"):
            log("  You named this model explicitly, so no fallback is applied.")
            log("  Check the id or path, then rerun.")
            return 3
        _gf = download_guard(fallback, 0.494 * 2.0, getattr(args, "allow_download", False))
        if not _gf["ok"]:
            log(f"  Not falling back: {fallback} {_gf['reason']}.")
            return 6
        log(f"  Falling back to {fallback}")
        model_id = fallback
        model, tok = load(model_id)

    cfg = model.config

    # ---- AUTO-DETECT: probe, do not assume -------------------------------- #
    rule("AUTO-DETECTION")
    # Reuse the probe taken above, which already honours --device. Re-probing
    # here ignored the override, so `--device cpu` on a CUDA box reported the
    # GPU and then sized the run against VRAM it was not going to use.
    gpu = accel
    if gpu["available"]:
        log(f"  GPU: {gpu['name']}  {gpu['total_gb']} GB total, "
            f"{gpu['free_gb']} GB free, cc {gpu['capability']}")
        log(f"  bf16 supported: {gpu['supports_bf16']}"
            + ("" if gpu["supports_bf16"] else "  (pre-Ampere -> using fp16)"))
    else:
        log("  no CUDA device; running on CPU")

    shape = detect_model_shape(model, cfg)
    head_dim = shape["head_dim"]
    n_layer = shape["n_layers"]
    n_kv = shape["n_kv_heads"]
    n_params = sum(p_.numel() for p_ in model.parameters())
    log(f"  {model_id}")
    log(f"  {n_params/1e9:.2f}B params, {n_layer} layers, head_dim {head_dim}")
    log(f"  attention: {shape['n_q_heads']} query heads, {n_kv} KV heads"
        + (f"  [GQA ratio {shape['gqa_ratio']}]" if shape["is_gqa"] else "")
        + ("  [MQA]" if shape["is_mqa"] else ""))
    log(f"  head_dim is a power of two: {shape['head_dim_is_pow2']}")

    plan = plan_run(gpu, shape, n_params / 1e9, n_eval_req, args.seq_len)
    if not plan.get("fits", True):
        log(f"  {plan['reason']}")
        if plan.get("est_peak_gb") is None:
            log("  Aborting: pick a smaller model with --model, e.g.")
            log("    --model Qwen/Qwen2.5-0.5B-Instruct")
            return 5
    elif plan["adjusted"]:
        log(f"  {plan['reason']}")
    else:
        log(f"  memory plan: ~{plan.get('est_peak_gb','?')} GB peak of "
            f"{plan.get('budget_gb','?')} GB budget")
    seq_len = plan["seq_len"]

    fp = run_fingerprint(model_id, seq_len, shape)
    if results.get("fingerprint") and results["fingerprint"] != fp:
        log(f"  cached results are for a different configuration "
            f"({results['fingerprint']} != {fp}); starting fresh")
        results = {}
    results["fingerprint"] = fp
    results.update({"model": model_id, "params_b": n_params / 1e9,
                    "n_layers": n_layer, "head_dim": head_dim,
                    "n_kv_heads": n_kv, "n_q_heads": shape["n_q_heads"],
                    "gqa_ratio": shape["gqa_ratio"], "device": dev,
                    "gpu": gpu, "shape": shape, "plan": plan})

    # ---- data -------------------------------------------------------------- #
    rule("DATA")
    text = load_eval_text(args, load_dataset)
    # The corpus is deliberately longer than the model's context; we slice it
    # into seq_len chunks below. Silence the (harmless) length warning.
    import transformers as _tf
    _prev = _tf.logging.get_verbosity() if hasattr(_tf, "logging") else None
    try:
        if _prev is not None:
            _tf.logging.set_verbosity_error()
        ids = tok(text, return_tensors="pt").input_ids
    finally:
        if _prev is not None:
            _tf.logging.set_verbosity(_prev)
    n_avail = ids.shape[1] // seq_len
    if n_avail < 1:
        # A corpus shorter than one sequence produced an empty eval list and a
        # ZeroDivisionError three functions later. Shrink instead of dying.
        new_len = max(128, 2 ** int(math.log2(max(ids.shape[1], 256))) // 2)
        log(f"  corpus has only {ids.shape[1]:,} tokens; reducing seq_len "
            f"{seq_len} -> {new_len}")
        seq_len = new_len
        n_avail = ids.shape[1] // seq_len
        if n_avail < 1:
            log(f"  Corpus is too short to evaluate even at seq_len={seq_len}. "
                f"Use a larger --text-file or drop --seq-len.")
            return 6
    n_eval = min(args.eval_seqs, n_avail)
    evals = [ids[:, i * seq_len:(i + 1) * seq_len] for i in range(n_eval)]
    log(f"  {n_eval} eval sequences of {seq_len} tokens "
        f"({n_avail} available in WikiText-2 test)")
    results.update({"seq_len": seq_len, "n_eval_seqs": n_eval})

    # ---- perplexity -------------------------------------------------------- #
    @torch.no_grad()
    def perplexity(seqs, label: str = "") -> float:
        nll, ntok = 0.0, 0
        for i, s in enumerate(seqs):
            s = s.to(dev)
            out = model(s, labels=s)
            n = s.shape[1] - 1
            if n <= 0:
                continue
            loss = out.loss.float().item()
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss on sequence {i}: {loss}")
            nll += loss * n
            ntok += n
            if label and (i + 1) % 8 == 0:
                log(f"      {label} {i+1}/{len(seqs)}")
        if ntok == 0:
            raise RuntimeError("no scorable tokens in the evaluation set")
        return math.exp(nll / ntok)

    if "ppl_fp16" not in results:
        rule("FP16 BASELINE")
        t0 = time.time()
        results["ppl_fp16"] = perplexity(evals, "fp16")
        save()
        log(f"  perplexity {results['ppl_fp16']:.4f}   ({time.time()-t0:.0f}s)")
    else:
        log(f"\n  FP16 baseline (cached): {results['ppl_fp16']:.4f}")
    ppl_fp16 = results["ppl_fp16"]

    # ---- KV quantisation hooks --------------------------------------------- #
    attns = [m for _, m in model.named_modules()
             if hasattr(m, "k_proj") and hasattr(m, "v_proj")]
    if not attns:
        log("  No k_proj/v_proj modules found; this architecture is unsupported.")
        return 4
    log(f"  found {len(attns)} attention modules for hooking")

    rot = None
    try:
        rot = make_hadamard(head_dim, torch, args.seed).to(dev, torch.float32)
        orth = (rot @ rot.T - torch.eye(head_dim, device=dev)).abs().max().item()
        log(f"  Hadamard {head_dim}x{head_dim} ready (orthogonality {orth:.2e})")
    except ValueError as exc:
        log(f"  {exc}")
        log(f"  head_dim={head_dim} is not a power of two, so the rotated half "
            f"of the sweep is SKIPPED. The granularity comparison still runs, "
            f"and that is the finding this project leads with.")

    @contextlib.contextmanager
    def kv_quant(bits: int, rotate: bool, group: Optional[int]):
        if bits >= 16:
            yield
            return
        handles = []

        def hook_factory(is_key: bool):
            def hook(mod, inp, out):
                sh = out.shape
                x = out.float().reshape(*sh[:-1], -1, head_dim)
                if rotate:
                    x = x @ rot
                x = quant_kv(x, bits, is_key, group, torch)
                if rotate:
                    x = x @ rot.T
                return x.reshape(*sh).to(out.dtype)
            return hook

        try:
            for a in attns:
                handles.append(a.k_proj.register_forward_hook(hook_factory(True)))
                handles.append(a.v_proj.register_forward_hook(hook_factory(False)))
            yield
        finally:
            for h in handles:
                h.remove()

    # Guard: the 16-bit path must be a no-op. NOT an assert -- a spurious
    # failure here would kill a 90-minute run in its second minute, and the
    # sweep is still informative even if the guard is inconclusive.
    try:
        plain = perplexity(evals[:2])
        with kv_quant(16, True, None):
            guarded = perplexity(evals[:2])
        delta = abs(guarded - plain)
        results["noop_guard_delta"] = delta
        if delta < 1e-6:
            log(f"  no-op guard passed (delta {delta:.2e})")
        else:
            log(f"  WARNING: 16-bit path is not a no-op (delta {delta:.3e}).")
            log("  The hook is altering activations it should pass through.")
            log("  Sweep numbers below are suspect -- investigate before quoting.")
    except Exception as exc:
        log(f"  no-op guard could not run: {type(exc).__name__}: {exc}")
        results["noop_guard_delta"] = None

    # ---- sweep ------------------------------------------------------------- #
    rule("KV QUANTISATION SWEEP")
    n_sweep = args.sweep_seqs or max(4, n_eval // 2)
    sweep_plan = plan_sweep(shape, seq_len, args.quick)
    for note in sweep_plan["notes"]:
        log(f"  {note}")
    bits_list = sweep_plan["bits"]
    group_list = sweep_plan["groups"]
    rot_list = sweep_plan["rotations"]
    results["sweep_plan"] = {k: v for k, v in sweep_plan.items() if k != "dropped"}
    configs = [(b, r, g) for b in bits_list for r in rot_list for g in group_list]
    log(f"  {len(configs)} valid configurations for this model")

    done = {(r["bits"], r["rotated"], r["group"])
            for r in results.get("kv_sweep", [])}
    sweep: List[Dict[str, Any]] = results.get("kv_sweep", [])
    t0 = time.time()
    for i, (bits, rotate, group) in enumerate(configs, 1):
        key = (bits, rotate, group)
        gname = str(group) if group else "per-token"
        if key in done:
            log(f"  [{i:2d}/{len(configs)}] cached  INT{bits} rot={int(rotate)} "
                f"group={gname}")
            continue
        try:
            with kv_quant(bits, rotate, group):
                ppl = perplexity(evals[:n_sweep])
            row = {"bits": bits, "rotated": rotate, "group": group,
                   "ppl": ppl, "delta_ppl": ppl - ppl_fp16}
            msg = f"ppl {ppl:.4f}"
        except Exception as exc:
            oom = "out of memory" in str(exc).lower()
            if dev == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if oom and n_sweep > 1:
                # One OOM means every later config OOMs too. Halve the work and
                # retry this config once rather than failing all of them.
                n_sweep = max(1, n_sweep // 2)
                log(f"  [{i:2d}/{len(configs)}] OOM -> retrying with "
                    f"{n_sweep} sequences")
                try:
                    with kv_quant(bits, rotate, group):
                        ppl = perplexity(evals[:n_sweep])
                    row = {"bits": bits, "rotated": rotate, "group": group,
                           "ppl": ppl, "delta_ppl": ppl - ppl_fp16,
                           "reduced_seqs": n_sweep}
                    msg = f"ppl {ppl:.4f} (on {n_sweep} seqs after OOM)"
                    sweep.append(row)
                    results["kv_sweep"] = sweep
                    save()
                    log(f"  [{i:2d}/{len(configs)}] INT{bits} "
                        f"rot={int(rotate)} group={gname:<9} {msg}")
                    continue
                except Exception as exc2:
                    exc = exc2
            row = {"bits": bits, "rotated": rotate, "group": group, "ppl": None,
                   "error": f"{type(exc).__name__}: {exc}"}
            msg = f"FAILED {row['error'][:48]}"
        sweep.append(row)
        results["kv_sweep"] = sweep
        save()
        log(f"  [{i:2d}/{len(configs)}] INT{bits} rot={int(rotate)} "
            f"group={gname:<9} {msg}")
    log(f"\n  sweep finished in {(time.time()-t0)/60:.1f} min")

    # ---- exact folding ----------------------------------------------------- #
    rule("EXACT FOLD VERIFICATION")
    if rot is None:
        log("  skipped (head_dim is not a power of two)")
    else:
        lay = attns[0]
        # cfg.hidden_size is None on a multimodal config; cfg_get finds the
        # text tower's value. Fall back to the measured projection width.
        _hidden = (cfg_get(cfg, "hidden_size", "n_embd", "d_model")
                   or getattr(lay.q_proj, "in_features", None))
        if not _hidden:
            raise RuntimeError("could not determine hidden size for fold check")
        x = torch.randn(2, 32, _hidden, device=dev, dtype=dtype)
        with torch.no_grad():
            q = lay.q_proj(x).reshape(2, 32, -1, head_dim).float()
            k = lay.k_proj(x).reshape(2, 32, -1, head_dim).float()
            # GQA: Qwen2.5-0.5B has 14 query heads and 2 KV heads, so K must be
            # repeated to the query head count exactly as attention does it.
            # The first release einsum'd them directly and crashed on 14 vs 2.
            n_q, n_k = q.shape[2], k.shape[2]
            if n_q != n_k:
                if n_q % n_k:
                    raise RuntimeError(f"{n_q} query heads is not a multiple of "
                                       f"{n_k} KV heads")
                k = k.repeat_interleave(n_q // n_k, dim=2)
                log(f"  GQA: repeated {n_k} KV heads to {n_q} query heads")
            base = torch.einsum("bthc,bshc->bhts", q, k)
            fold = torch.einsum("bthc,bshc->bhts", q @ rot, k @ rot)
            err = ((base - fold).abs().max() / base.abs().max()).item()
        results["fold_rel_err"] = err
        save()
        log(f"  relative error in attention logits after folding: {err:.3e}")
        log("  => orthogonal folding adds zero operators at inference.")

    # ---- decode + memory --------------------------------------------------- #
    rule("DECODE AND MEMORY BUDGET")
    try:
        with torch.no_grad():
            ids0 = evals[0][:, :512].to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            model.generate(ids0, max_new_tokens=args.decode_tokens,
                           do_sample=False,
                           pad_token_id=_pad_token_id(tok))
            if dev == "cuda":
                torch.cuda.synchronize()
            tps = args.decode_tokens / (time.time() - t)
        results["decode_tok_per_s"] = tps
        log(f"  decode: {tps:.1f} tok/s on {dev}")
    except Exception as exc:
        log(f"  decode benchmark failed: {type(exc).__name__}: {exc}")

    kv_bytes = 2 * 2 * n_layer * n_kv * head_dim
    results["kv_bytes_per_token_fp16"] = kv_bytes
    save()
    log(f"  KV cache: {kv_bytes/1024:.1f} KiB/token fp16")
    for ctx in (8192, 32768, 131072):
        log(f"    {ctx:>7} ctx: {kv_bytes*ctx/2**30:6.2f} GiB fp16   "
            f"{kv_bytes*ctx/4/2**30:6.2f} GiB int4")
    log("  X2 Plus has ~152 GB/s LPDDR5X; decode re-reads the whole cache per")
    log("  token, so this is the term that binds at long context.")

    # ---- table and plot ---------------------------------------------------- #
    rule("RESULTS")
    ok = [r for r in sweep if r.get("ppl")]
    log(f"  FP16 baseline: {ppl_fp16:.4f}\n")
    log(f"  {'bits':>5} {'rotation':>10} {'group':>10} {'ppl':>10} {'delta':>9}")
    for r in sorted(ok, key=lambda r: (-r["bits"], r["rotated"],
                                       r["group"] or 0)):
        log(f"  {r['bits']:>5} {'Hadamard' if r['rotated'] else 'none':>10} "
            f"{str(r['group']) if r['group'] else 'per-token':>10} "
            f"{r['ppl']:>10.4f} {r['delta_ppl']:>+9.4f}")

    if ok and not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5))
            groups = sorted({r["group"] or 0 for r in ok})
            for g in groups:
                for rt in (False, True):
                    pts = sorted([r for r in ok
                                  if (r["group"] or 0) == g and r["rotated"] == rt],
                                 key=lambda r: r["bits"])
                    if not pts:
                        continue
                    ax.plot([p["bits"] for p in pts], [p["ppl"] for p in pts],
                            marker="o", ls="-" if rt else "--",
                            label=f"group={g or 'per-token'}, "
                                  f"{'Hadamard' if rt else 'none'}")
            ax.axhline(ppl_fp16, color="k", lw=1, label="FP16")
            ax.set_xlabel("KV bits")
            ax.set_ylabel("WikiText-2 perplexity")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_title(f"{model_id} - KV quantisation")
            fig.tight_layout()
            fig.savefig(out_dir / "kv_sweep.png", dpi=140)
            log(f"\n  wrote {out_dir/'kv_sweep.png'}")
        except Exception as exc:
            log(f"  plot skipped: {type(exc).__name__}: {exc}")

    save()
    rule("DONE")
    log(f"  {res_path}")
    log("\n  Paste these into SUBMISSION.md, replacing the [FILL] markers:")
    log(f"    FP16 perplexity      {ppl_fp16:.4f}")
    if results.get("fold_rel_err") is not None:
        log(f"    fold error           {results['fold_rel_err']:.3e}")
    if results.get("decode_tok_per_s"):
        log(f"    decode               {results['decode_tok_per_s']:.1f} tok/s")
    log(f"    KV per token         {kv_bytes/1024:.1f} KiB fp16")
    log("\n  Then run the deployment exports:")
    log("    python snapdragon_engine.py export --model gemma-4-e2b-it \\")
    log('        --device "Snapdragon X2 Elite CRD"')
    return 0


# --------------------------------------------------------------------------- #
# SELF TEST AND STRESS HARNESS
#
# This section exists because of an asymmetry that cost two Colab sessions.
#
# Every other module in SIGIL-Edge carries a self-test, and none of them has
# ever broken on someone else's machine. This file -- the ONLY one that runs on
# hardware the author does not have, against library versions the author did
# not pick -- carried no tests at all, and broke twice:
#
#   crash 1  datasets >= 4 rejects the bare id "wikitext"
#            transformers 5 renamed torch_dtype -> dtype
#   crash 2  einsum on 14 query heads against 2 KV heads
#            a group of 128 viewed against a head_dim of 64
#            Keys reduced over the head axis instead of the token axis, so the
#            per-token and grouped rows measured different schemes and the
#            published "grouping is 4x worse" result was an artefact
#
# Every one of those is a five-line unit test. None needed a GPU, a download,
# or a single real weight. They are all below now.
#
#   python sigil_t4_benchmark.py --selftest    # < 30 s, CPU, no network
#   python sigil_t4_benchmark.py --stress      # ~2-4 min, CPU, no network
#
# --stress builds RANDOMLY INITIALISED models across a shape matrix -- MHA,
# GQA, MQA, non-power-of-two head dims, single-layer, short-context -- and
# drives the real pipeline over each: the real hooks, the real quantiser, the
# real fold check. Perplexity from random weights is meaningless and is not
# reported; what is tested is that the machinery survives the shape.
# --------------------------------------------------------------------------- #

STRESS_SHAPES = [
    # name,                 layers, hidden, q_heads, kv_heads, head_dim, ctx
    ("MHA square",               2,    128,       4,        4,       32,  256),
    ("GQA 4:2 (Qwen-like)",      2,    128,       4,        2,       32,  256),
    ("GQA 14:2 (the crash)",     2,    448,      14,        2,       32,  256),
    ("MQA 8:1",                  2,    256,       8,        1,       32,  256),
    ("head_dim 64 pow2",         2,    256,       4,        2,       64,  256),
    ("head_dim 48 non-pow2",     2,    192,       4,        2,       48,  256),
    ("single layer",             1,    128,       4,        1,       32,  128),
    ("wide GQA 16:4",            2,    512,      16,        4,       32,  256),
    ("tiny context 64",          2,    128,       4,        2,       32,   64),
]


def _tiny_model(torch_mod, n_layers, hidden, q_heads, kv_heads, head_dim,
                ctx, vocab=512):
    """
    A randomly initialised Llama-shaped model with an exact attention geometry.
    No network, no weights on disk, a few MB of RAM.
    """
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=n_layers, num_attention_heads=q_heads,
        num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=ctx, tie_word_embeddings=True,
    )
    torch_mod.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    return model.eval(), cfg


def _hook_roundtrip(torch_mod, model, cfg, head_dim, bits, rotate, group,
                    seq_len, vocab=512):
    """
    Drive the REAL hook path -- the same closure run() installs -- over one
    forward pass, and return the loss. This is the code that crashed in Colab;
    exercising it on a 3 MB model costs a second.
    """
    import contextlib
    attns = [m for _, m in model.named_modules()
             if hasattr(m, "k_proj") and hasattr(m, "v_proj")]
    if not attns:
        raise RuntimeError("no attention modules found")
    rot = None
    if rotate:
        rot = make_hadamard(head_dim, torch_mod, 0).to(torch_mod.float32)

    handles = []

    def hook_factory(is_key):
        def hook(mod, inp, out):
            sh = out.shape
            x = out.float().reshape(*sh[:-1], -1, head_dim)
            if rot is not None:
                x = x @ rot
            x = quant_kv(x, bits, is_key, group, torch_mod)
            if rot is not None:
                x = x @ rot.T
            return x.reshape(*sh).to(out.dtype)
        return hook

    try:
        for a in attns:
            handles.append(a.k_proj.register_forward_hook(hook_factory(True)))
            handles.append(a.v_proj.register_forward_hook(hook_factory(False)))
        # Seed the INPUT, not just the weights. Without this every call scores
        # a different random token sequence and two configs can never be
        # compared -- which made the no-op check fail against itself.
        g = torch_mod.Generator().manual_seed(1234)
        ids = torch_mod.randint(0, vocab, (1, seq_len), generator=g)
        with torch_mod.no_grad():
            out = model(ids, labels=ids)
        return float(out.loss)
    finally:
        for h in handles:
            h.remove()


def _fold_roundtrip(torch_mod, model, cfg, head_dim):
    """The GQA fold check, exactly as run() performs it."""
    attns = [m for _, m in model.named_modules()
             if hasattr(m, "k_proj") and hasattr(m, "v_proj")]
    lay = attns[0]
    hidden = (cfg_get(cfg, "hidden_size", "n_embd", "d_model")
              or getattr(lay.q_proj, "in_features", None))
    rot = make_hadamard(head_dim, torch_mod, 0).to(torch_mod.float32)
    x = torch_mod.randn(2, 16, hidden)
    with torch_mod.no_grad():
        q = lay.q_proj(x).reshape(2, 16, -1, head_dim).float()
        k = lay.k_proj(x).reshape(2, 16, -1, head_dim).float()
        n_q, n_k = q.shape[2], k.shape[2]
        if n_q != n_k:
            if n_q % n_k:
                raise RuntimeError(f"{n_q} q heads not a multiple of {n_k} kv")
            k = k.repeat_interleave(n_q // n_k, dim=2)
        base = torch_mod.einsum("bthc,bshc->bhts", q, k)
        fold = torch_mod.einsum("bthc,bshc->bhts", q @ rot, k @ rot)
        return float((base - fold).abs().max() / base.abs().max())


class _FakeTok:
    def __init__(self, eos=None, pad=None):
        self.eos_token_id, self.pad_token_id = eos, pad


class _FakeSub:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, _):
        return None


class _FakeCfg:
    """A multimodal-shaped config: the text tower is nested, as in a real VLM."""
    def __init__(self, **kw):
        self.text_config = _FakeSub(**kw)

    def __getattr__(self, _):
        return None


def selftest(verbose: bool = True) -> int:
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), str(detail)))

    def raises(fn):
        try:
            fn()
            return False
        except Exception:
            return True

    try:
        import torch
    except ImportError:
        log("  torch is required even for --selftest. pip install -U torch")
        return 2

    # ---- cfg_get: the multimodal nesting bug --------------------------------
    nested = _FakeCfg(num_hidden_layers=64, hidden_size=5120,
                      num_attention_heads=24, num_key_value_heads=4,
                      head_dim=256)
    ck("cfg_get reaches a nested text_config",
       cfg_get(nested, "num_hidden_layers") == 64)
    ck("plain getattr would have returned None here",
       getattr(nested, "num_hidden_layers", None) is None)
    ck("cfg_get honours alternative names",
       cfg_get(_FakeCfg(n_embd=768), "hidden_size", "n_embd") == 768)
    ck("cfg_get returns the default when absent",
       cfg_get(_FakeCfg(), "nope", default=7) == 7)

    # ---- dtype selection ----------------------------------------------------
    cpu_a = {"available": False, "supports_bf16": False, "backend": "cpu"}
    amp_a = {"available": True, "supports_bf16": True, "backend": "cuda"}
    t4_a = {"available": True, "supports_bf16": False, "backend": "cuda"}
    ck("CPU auto-selects fp32", select_dtype(torch, cpu_a)[0] == torch.float32)
    ck("Ampere auto-selects bf16",
       select_dtype(torch, amp_a)[0] == torch.bfloat16)
    ck("T4 auto-selects fp16 (no bf16 before Ampere)",
       select_dtype(torch, t4_a)[0] == torch.float16)
    ck("fp16 on CPU is refused, not obeyed",
       select_dtype(torch, cpu_a, "fp16")[0] == torch.float32)
    ck("bf16 request on a T4 downgrades rather than failing",
       select_dtype(torch, t4_a, "bf16")[0] == torch.float16)
    ck("fp32 is always honoured",
       select_dtype(torch, amp_a, "fp32")[0] == torch.float32)

    # ---- accelerator probe --------------------------------------------------
    acc = detect_accelerator(torch)
    ck("accelerator probe returns a uniform record",
       {"backend", "device", "supports_bf16", "available"} <= set(acc))
    ck("probe never raises without CUDA", acc["device"] in
       ("cpu", "cuda", "mps", "xpu"), acc["device"])

    # ---- pad token ----------------------------------------------------------
    ck("int eos accepted", _pad_token_id(_FakeTok(eos=2)) == 2)
    ck("LIST eos is unwrapped, not passed through",
       _pad_token_id(_FakeTok(eos=[151645, 151643])) == 151645)
    ck("pad_token_id wins over eos", _pad_token_id(_FakeTok(eos=2, pad=0)) == 0)
    ck("missing both falls back to 0", _pad_token_id(_FakeTok()) == 0)

    # ---- group validity: crash 2(b) ----------------------------------------
    ck("group larger than head_dim is dropped",
       128 not in valid_groups(2048, 64, [None, 128, 32]))
    ck("valid group is kept", 32 in valid_groups(2048, 64, [None, 128, 32]))
    ck("per-token (None) always survives",
       None in valid_groups(16, 16, [None, 32]))
    ck("group equal to head_dim is allowed",
       64 in valid_groups(2048, 64, [64]))

    # ---- quantiser: crash 2(c), the serious one -----------------------------
    x = torch.randn(1, 64, 4, 32)
    ck("16-bit path is an exact no-op",
       torch.equal(quant_kv(x, 16, True, None, torch), x))
    for bits in (2, 3, 4, 8):
        for is_key in (True, False):
            for g in (None, 8, 16):
                y = quant_kv(x, bits, is_key, g, torch)
                ck(f"quant_kv preserves shape (bits={bits}, key={is_key}, g={g})",
                   y.shape == x.shape)
                ck(f"quant_kv is finite (bits={bits}, key={is_key}, g={g})",
                   bool(torch.isfinite(y).all()))
    # The convention itself: K reduces over tokens, V over channels. If the
    # axes are swapped the errors below swap too, which is exactly the bug
    # that produced the bogus "grouping is 4x worse" headline.
    k_err = (quant_kv(x, 4, True, None, torch) - x).abs().mean().item()
    v_err = (quant_kv(x, 4, False, None, torch) - x).abs().mean().item()
    ck("K and V use different axes (errors differ)", abs(k_err - v_err) > 1e-9,
       f"K {k_err:.2e} vs V {v_err:.2e}")
    ck("more bits is never worse",
       (quant_kv(x, 8, True, None, torch) - x).abs().mean().item() <= k_err + 1e-9)
    ck("finer groups are never worse than per-axis",
       (quant_kv(x, 4, True, 8, torch) - x).abs().mean().item() <= k_err + 1e-6)
    # A group spanning the whole axis must equal the ungrouped result exactly.
    ck("group == axis length degenerates to per-axis",
       torch.allclose(quant_kv(x, 4, True, 64, torch),
                      quant_kv(x, 4, True, None, torch), atol=1e-6))
    # Non-divisible group sizes must pad and restore without changing shape.
    xo = torch.randn(1, 50, 3, 24)
    for g in (7, 16, 32):
        for is_key in (True, False):
            y = quant_kv(xo, 4, is_key, g, torch)
            ck(f"ragged axis pads and restores (g={g}, key={is_key})",
               y.shape == xo.shape and bool(torch.isfinite(y).all()))

    # ---- Hadamard -----------------------------------------------------------
    for n in (32, 64, 128):
        H = make_hadamard(n, torch, 0)
        ck(f"Hadamard {n} is orthogonal",
           float((H @ H.T - torch.eye(n, dtype=H.dtype)).abs().max()) < 1e-9)
    ck("non-power-of-two Hadamard is refused, not approximated",
       raises(lambda: make_hadamard(48, torch, 0)))
    ck("Hadamard is deterministic under a seed",
       torch.equal(make_hadamard(32, torch, 3), make_hadamard(32, torch, 3)))

    # ---- planners -----------------------------------------------------------
    shape64 = {"head_dim": 64, "n_kv_heads": 2, "n_layers": 24,
               "hidden_size": 896, "head_dim_is_pow2": True, "is_gqa": True,
               "n_q_heads": 14, "gqa_ratio": 7}
    sp = plan_sweep(shape64, 2048, False)
    ck("sweep plan drops groups wider than head_dim",
       all(g is None or g <= 64 for g in sp["groups"]), str(sp["groups"]))
    ck("sweep plan reports the GQA ratio in its notes",
       any("GQA" in n for n in sp["notes"]))
    ck("sweep plan counts configs consistently",
       sp["n_configs"] == len(sp["bits"]) * len(sp["groups"]) * len(sp["rotations"]))
    shape48 = dict(shape64, head_dim=48, head_dim_is_pow2=False)
    sp48 = plan_sweep(shape48, 2048, False)
    ck("non-pow2 head_dim disables rotation but keeps the sweep",
       sp48["rotations"] == (False,) and sp48["n_configs"] > 0)

    big = plan_run({"available": True, "free_gb": 4.0, "total_gb": 4.0,
                    "name": "small card"}, shape64, 8.0, 24, 2048)
    ck("an impossible plan is refused, not shrunk into nonsense",
       big.get("fits") is False and big.get("est_peak_gb") is None)
    okp = plan_run({"available": True, "free_gb": 14.6, "total_gb": 14.6,
                    "name": "T4"}, shape64, 0.5, 24, 2048)
    ck("a feasible plan fits", okp.get("fits") is True)
    ck("a feasible plan stays inside its own budget",
       okp["est_peak_gb"] <= okp["budget_gb"])
    cpup = plan_run({"available": False}, shape64, 0.5, 24, 2048)
    ck("CPU plan is bounded so the run terminates",
       cpup["eval_seqs"] <= 4 and cpup["seq_len"] <= 512)

    # ---- parameter estimate and fit ----------------------------------------
    est = estimate_params_from_config(
        _FakeCfg(num_hidden_layers=24, hidden_size=896, vocab_size=151936,
                 intermediate_size=4864, num_attention_heads=14,
                 num_key_value_heads=2, head_dim=64, tie_word_embeddings=True))
    ck("Qwen2.5-0.5B estimate lands near 0.5B", est and 0.3 < est < 0.8,
       f"{est:.2f}B" if est else "None")
    ck("estimate degrades to None on an empty config",
       estimate_params_from_config(_FakeCfg()) is None)
    fv = fit_verdict(8.0, {"free_gb": 14.6, "total_gb": 14.6, "name": "T4"})
    ck("8B fp16 does not fit a T4", fv["known"] and not fv["fits"])
    ck("refusal carries actionable advice", "4bit" in fv["advice"].replace("-", ""))
    ck("0.5B fits a T4",
       fit_verdict(0.5, {"free_gb": 14.6, "total_gb": 14.6, "name": "T4"})["fits"])
    ck("unknown size never blocks the run",
       fit_verdict(None, {"free_gb": 1.0})["fits"])

    # ---- fingerprint --------------------------------------------------------
    f1 = run_fingerprint("a", 2048, shape64)
    ck("fingerprint is stable", f1 == run_fingerprint("a", 2048, shape64))
    ck("fingerprint changes with the model", f1 != run_fingerprint("b", 2048, shape64))
    ck("fingerprint changes with seq_len", f1 != run_fingerprint("a", 1024, shape64))
    ck("fingerprint changes with geometry",
       f1 != run_fingerprint("a", 2048, dict(shape64, n_kv_heads=4)))

    # ---- the real pipeline on a real (tiny) model ---------------------------
    try:
        m, cfg = _tiny_model(torch, 2, 448, 14, 2, 32, 256)
        sh = detect_model_shape(m, cfg)
        ck("shape probe recovers 14 query heads", sh["n_q_heads"] == 14, sh["n_q_heads"])
        ck("shape probe recovers 2 KV heads", sh["n_kv_heads"] == 2, sh["n_kv_heads"])
        ck("shape probe computes the GQA ratio", sh["gqa_ratio"] == 7, sh["gqa_ratio"])
        ck("shape probe flags GQA", sh["is_gqa"] and not sh["is_mqa"])
        ck("shape probe finds every attention module", sh["n_attn_modules"] == 2)

        base = _hook_roundtrip(torch, m, cfg, 32, 16, False, None, 64)
        q4 = _hook_roundtrip(torch, m, cfg, 32, 4, False, None, 64)
        ck("hook path is reproducible across calls",
           _hook_roundtrip(torch, m, cfg, 32, 16, False, None, 64) == base)
        ck("16-bit hook is an exact no-op end to end",
           _hook_roundtrip(torch, m, cfg, 32, 16, False, 16, 64) == base)
        ck("16-bit hook stays a no-op under rotation",
           abs(_hook_roundtrip(torch, m, cfg, 32, 16, True, 16, 64) - base) < 1e-4)
        ck("4-bit hook actually changes the loss", abs(q4 - base) > 1e-6,
           f"delta {abs(q4 - base):.3e}")
        ck("4-bit hook produces a finite loss", math.isfinite(q4))
        ck("rotated hook runs on a GQA model",
           math.isfinite(_hook_roundtrip(torch, m, cfg, 32, 4, True, 16, 64)))
        ck("grouped hook runs on a GQA model",
           math.isfinite(_hook_roundtrip(torch, m, cfg, 32, 3, False, 16, 64)))

        err = _fold_roundtrip(torch, m, cfg, 32)
        ck("GQA fold identity holds (crash 2a)", err < 1e-4, f"{err:.2e}")
    except Exception as exc:
        ck(f"tiny-model pipeline: {type(exc).__name__}: {str(exc)[:70]}", False)

    # ---- the download guard -- the reason this file no longer fills laptops --
    _lap = {}
    ck("an 8 GB hub download is refused on a laptop",
       not download_guard("Qwen/Qwen3-4B", 8.0, env=_lap)["ok"])
    ck("the same download goes ahead on Colab",
       download_guard("Qwen/Qwen3-4B", 8.0, env={"COLAB_RELEASE_TAG": "x"})["ok"])
    ck("--allow-download overrides the refusal",
       download_guard("Qwen/Qwen3-4B", 8.0, allow=True, env=_lap)["ok"])
    ck("the ~1 GB fallback model is allowed",
       download_guard("Qwen/Qwen2.5-0.5B-Instruct", 0.494 * 2.0, env=_lap)["ok"])
    ck("a local path downloads nothing",
       download_guard(__file__, None, env=_lap)["ok"])
    ck("an unknown or non-finite size counts as large",
       not download_guard("a/b", None, env=_lap)["ok"]
       and not download_guard("a/b", float("inf"), env=_lap)["ok"])

    # ---- dataset fallbacks --------------------------------------------------
    ck("wikitext candidates use the namespaced id first",
       DATASET_CANDIDATES[0][0] == "Salesforce/wikitext")
    ck("a bare-id fallback is still offered",
       any(c[0] == "wikitext" for c in DATASET_CANDIDATES))

    if verbose:
        rule("SELF TEST")
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        if verbose and not ok:
            log(f"  [FAIL] {name}" + (f"   {detail}" if detail else ""))
    if verbose:
        for name, ok, detail in checks:
            if ok:
                log(f"  [PASS] {name}" + (f"   {detail}" if detail else ""))
        log("")
        log(f"  {npass}/{len(checks)} checks passed")
    return 0 if npass == len(checks) else 1


def stress(verbose: bool = True) -> int:
    """
    Run the real pipeline over a matrix of attention geometries.

    Randomly initialised weights, so no perplexity is reported -- the point is
    that hooks, quantiser, rotation and fold survive every shape, including the
    ones that took down two Colab sessions.
    """
    try:
        import torch
    except ImportError:
        log("  torch is required for --stress. pip install -U torch")
        return 2

    rule("STRESS TEST -- SHAPE MATRIX")
    log("  Random weights: losses are meaningless and not reported.")
    log("  What is tested is that every shape survives the real pipeline.\n")
    log(f"  {'geometry':<24}{'q:kv':>7}{'hd':>5}{'cfgs':>6}{'fold':>11}  status")

    failures, total_cfgs = [], 0
    for (name, L, hidden, qh, kvh, hd, ctx) in STRESS_SHAPES:
        try:
            m, cfg = _tiny_model(torch, L, hidden, qh, kvh, hd, ctx)
            shape = detect_model_shape(m, cfg)
            if shape["n_q_heads"] != qh or shape["n_kv_heads"] != kvh:
                raise AssertionError(
                    f"probe read {shape['n_q_heads']}:{shape['n_kv_heads']}, "
                    f"expected {qh}:{kvh}")
            seq = min(ctx, 96)
            plan = plan_sweep(shape, seq, quick=False)
            n_ok = 0
            for bits in plan["bits"]:
                for rotate in plan["rotations"]:
                    for g in valid_groups(seq, hd, plan["groups"]):
                        loss = _hook_roundtrip(torch, m, cfg, hd, bits,
                                               rotate, g, seq)
                        if not math.isfinite(loss):
                            raise AssertionError(
                                f"non-finite loss at bits={bits} "
                                f"rot={rotate} group={g}")
                        n_ok += 1
            total_cfgs += n_ok
            if shape["head_dim_is_pow2"]:
                ferr = _fold_roundtrip(torch, m, cfg, hd)
                fold = f"{ferr:.1e}"
                if ferr > 1e-4:
                    raise AssertionError(f"fold error {ferr:.2e} is too large")
            else:
                fold = "n/a"
            log(f"  {name:<24}{f'{qh}:{kvh}':>7}{hd:>5}{n_ok:>6}{fold:>11}  ok")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            log(f"  {name:<24}{f'{qh}:{kvh}':>7}{hd:>5}{'-':>6}{'-':>11}  "
                f"FAIL {type(exc).__name__}")

    # Degenerate and hostile inputs that should be refused or absorbed.
    rule("STRESS TEST -- EDGE CASES")
    edge = []

    def eck(name, fn, expect_raise=False):
        try:
            fn()
            edge.append((name, not expect_raise))
        except Exception:
            edge.append((name, expect_raise))

    x = torch.randn(1, 8, 2, 16)
    eck("group of 1", lambda: quant_kv(x, 4, True, 1, torch))
    eck("group larger than the axis", lambda: quant_kv(x, 4, True, 9999, torch))
    eck("1-bit request", lambda: quant_kv(x, 1, True, None, torch))
    eck("single token", lambda: quant_kv(torch.randn(1, 1, 2, 16), 4, True, None, torch))
    eck("single channel", lambda: quant_kv(torch.randn(1, 8, 2, 1), 4, False, None, torch))
    eck("all zeros (scale would be 0)",
        lambda: quant_kv(torch.zeros(1, 8, 2, 16), 4, True, None, torch))
    eck("large magnitudes",
        lambda: quant_kv(torch.full((1, 8, 2, 16), 1e4), 4, True, 8, torch))
    eck("tiny magnitudes",
        lambda: quant_kv(torch.full((1, 8, 2, 16), 1e-8), 4, False, 8, torch))
    eck("prime-length axis", lambda: quant_kv(torch.randn(1, 17, 3, 13), 4, True, 5, torch))
    eck("non-pow2 Hadamard refused", lambda: make_hadamard(48, torch, 0), True)
    eck("zero-size group refused", lambda: quant_kv(x, 4, True, 0, torch))

    zeros = quant_kv(torch.zeros(1, 8, 2, 16), 4, True, None, torch)
    edge.append(("all-zero input stays finite", bool(torch.isfinite(zeros).all())))
    edge.append(("all-zero input stays zero", float(zeros.abs().max()) == 0.0))

    for name, ok in edge:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    edge_fail = [n for n, ok in edge if not ok]

    rule("STRESS SUMMARY")
    log(f"  {len(STRESS_SHAPES) - len(failures)}/{len(STRESS_SHAPES)} geometries, "
        f"{total_cfgs} quantiser configurations, "
        f"{len(edge) - len(edge_fail)}/{len(edge)} edge cases")
    for name, why in failures:
        log(f"  FAILED {name}: {why}")
    for name in edge_fail:
        log(f"  FAILED edge case: {name}")
    if not failures and not edge_fail:
        log("  All shapes and edge cases pass. This is what was missing when "
            "the script broke on Colab.")
        return 0
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log()
    log("  SIGIL-Edge  KV-quantisation benchmark")
    log("  real weights, real perplexity, real memory budget")
    if args.device == "cpu":
        # Asking for CPU explicitly IS the consent that --allow-cpu asks for.
        args.allow_cpu = True
    if args.quick:
        args.eval_seqs = min(args.eval_seqs, 8)
        args.sweep_seqs = 4
        log("  --quick: reduced configuration, ~15 min")
    if args.selftest:
        return selftest()
    if args.stress:
        rc = selftest(verbose=False)
        if rc:
            log("  Self test FAILED; fix that before reading stress results.")
            log("  Run: python sigil_t4_benchmark.py --selftest")
            return rc
        log("  self test passed; starting stress matrix")
        return stress()

    check_env(args)
    if args.dry_run:
        rule("DRY RUN")
        # Say now, not ninety seconds into the real run, whether the download
        # guard will let the model onto THIS machine -- sized exactly as the
        # real run sizes it, from config.json (kilobytes), so the two verdicts
        # cannot disagree. No weights are fetched; offline, the size is unknown
        # and unknown counts as large, which is also what the real run would do.
        for ref in (getattr(args, "models", None) or [args.model]):
            _sz = None
            if not os.path.exists(str(ref)):
                try:
                    from transformers import AutoConfig
                    _c = AutoConfig.from_pretrained(ref, trust_remote_code=True)  # download-ok: config.json only, kilobytes, sizes the verdict the real run will reach
                    _e = estimate_params_from_config(_c)
                    _sz = None if _e is None else _e * 2.0
                except Exception:
                    _sz = None
            _g = download_guard(ref, _sz, getattr(args, "allow_download", False))
            if _g["ok"]:
                log(f"  download guard: {ref} may be fetched here ({_g['reason']})")
            else:
                log(f"  download guard: the real run will REFUSE {ref} here -- it "
                    f"{_g['reason']}. Run it on Colab, or pass --allow-download.")
        log("  Environment is fine. Remove --dry-run to benchmark.")
        return 0

    if args.models:
        # Multi-model sweep. Each model gets its own results file so a failure
        # on one does not lose the others, and a comparison table is written at
        # the end from whatever succeeded.
        import copy
        summary = []
        for n, mid in enumerate(args.models, 1):
            rule(f"MODEL {n}/{len(args.models)}: {mid}")
            sub = copy.copy(args)
            sub.model = mid
            sub._in_sweep = True
            # Model ids become directory names, so strip everything a
            # filesystem might object to. A local path like C:\models\foo
            # would otherwise produce a nested or invalid directory.
            stem = Path(mid).name if (os.sep in mid or "/" in mid) and \
                Path(mid).exists() else mid
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")[:60] or "model"
            sub.out_dir = str(Path(args.out_dir) / safe)
            try:
                rc = run(sub)
                rp = Path(sub.out_dir) / "results_t4.json"
                if rp.exists():
                    summary.append(json.loads(rp.read_text()))
            except Exception as exc:
                log(f"  {mid} FAILED: {type(exc).__name__}: {exc}")
                log("  continuing with the next model")
        if summary:
            rule("CROSS-MODEL COMPARISON")
            log(f"  {'model':<34}{'params':>8}{'fp16 ppl':>10}{'best INT4':>11}")
            for r in summary:
                sw = [x for x in r.get("kv_sweep", [])
                      if x.get("ppl") and x.get("bits") == 4]
                best = min((x["ppl"] for x in sw), default=None)
                log(f"  {r.get('model','?')[:33]:<34}"
                    f"{r.get('params_b',0):>8.2f}"
                    f"{r.get('ppl_fp16',0):>10.3f}"
                    f"{(f'{best:.3f}' if best else '-'):>11}")
            Path(args.out_dir, "comparison.json").write_text(
                json.dumps(summary, indent=2, default=str))
            log(f"\n  wrote {Path(args.out_dir,'comparison.json')}")
        # A run in which every model failed must not report success: a
        # scheduled job would record it as a pass and nobody would look.
        if not summary:
            log("  Every model failed; nothing was written.")
            return 7
        if len(summary) < len(args.models):
            log(f"  {len(args.models) - len(summary)} of {len(args.models)} "
                f"models failed; the table above covers the rest.")
        return 0

    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\n  interrupted; partial results were saved")
        raise SystemExit(130)
