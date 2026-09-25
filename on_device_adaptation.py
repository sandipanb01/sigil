#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 on_device_adaptation.py -- can a Snapdragon CPU actually train anything?
================================================================================

Short answer: it cannot TRAIN, but it can ADAPT, and the distinction is the
whole product.

--------------------------------------------------------------------------------
 STEP 1: THE FLOP ARITHMETIC (which turns out not to be the binding constraint)
--------------------------------------------------------------------------------
Backward is ~2x forward, so training costs ~6*P FLOPs/token against ~2*P for
inference. LoRA freezes the base, so only adapter gradients flow: ~2.2*P.

At a generous 400 G-ops/s sustained across an X2-class CPU:

    pretrain 1B from scratch (1T tokens)      476 YEARS
    full finetune 1B (1B tokens)              174 days
    LoRA finetune 1B (10M tokens)             15.3 hours
    LoRA finetune 0.5B (1M tokens)            46 minutes
    LoRA finetune 0.5B (100k tokens)          5 minutes
    personalise 45M Needle (10k tokens)       2 seconds

So FLOPs say LoRA is comfortably practical below ~1B.

--------------------------------------------------------------------------------
 STEP 2: THE CORRECTION -- ACTIVATION MEMORY, NOT FLOPS, IS THE BLOCKER
--------------------------------------------------------------------------------
"Parameter Efficiency Is Not Memory Efficiency: Rethinking Fine-Tuning for
On-Device LLM Adaptation" (arXiv:2604.22783) makes the point this module is
named after. PEFT cuts TRAINABLE PARAMETERS by orders of magnitude. It does
almost nothing for the ACTIVATIONS that backpropagation must retain:

    "fine-tuning Llama 7B requires up to 45.6 GB of on-chip memory for internal
     activations, making it impractical for most edge devices"     (arXiv:2409.15520)

That is the same lesson as this project's roofline, arriving from the training
side: **memory is the constraint, not compute.** A LoRA run whose FLOPs fit in
46 minutes still fails if its activation stack does not fit in RAM.

`adaptation_feasibility()` therefore checks BOTH, and reports which one binds.

--------------------------------------------------------------------------------
 STEP 3: THE WAY AROUND IT
--------------------------------------------------------------------------------
Zeroth-order / perturbation methods (PRGE, arXiv:2409.15520, "Enabling On-Device
Fine-Tuning of LLMs Using Only Inference Engines") estimate gradients from
forward passes alone. No backward pass means NO ACTIVATION STACK -- the blocker
disappears entirely. The cost is far more steps for the same progress, which
trades the constraint that binds (memory) for the one that does not (compute).

On a device whose NPU is busy serving inference and whose CPU is idle, that is
an unusually good trade.

--------------------------------------------------------------------------------
 STEP 4: WHAT ALREADY SHIPS -- this is not speculative
--------------------------------------------------------------------------------
  llama.cpp `finetune`        LoRA on CPU against quantised GGUF. RAM-bound:
                              roughly 16 GB trains 3B, 32 GB trains 7B.
  QVAC-fabric-llm.cpp         Tether AI, Dec 2025. Reports the FIRST successful
  (Tether)                    fine-tuning on mobile GPUs including **Adreno** --
                              Snapdragon's own GPU. Pre-built binaries.
  MobileFineTuner             arXiv:2512.08211. End-to-end C++ fine-tuning on
                              phones, native autodiff and full backprop, no
                              Python runtime.
  Federated LoRA              FedIT, FLoRA, HeLoRA, HetLoRA, PCFT-LLM -- the
                              fleet-scale version.

--------------------------------------------------------------------------------
 STEP 5: WHY THIS MATTERS FOR A SNAPDRAGON SUBMISSION
--------------------------------------------------------------------------------
This project established that decode is bandwidth-bound and the NPU does the
serving. That leaves **the CPU largely idle during inference**. On-device
adaptation is the natural use of that idle silicon, and it is the one thing a
cloud model structurally cannot do: personalise on data that never leaves the
device.

The honest pitch is NOT "we train LLMs on Snapdragon". It is:

    "Inference on the NPU. Adaptation on the idle CPU. The user's corrections
     never leave the device, and the model is measurably better tomorrow than
     it was today."

Licence: Apache-2.0. NumPy not required.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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



__all__ = ["adaptation_feasibility", "ADAPTATION_METHODS", "SHIPPING_TOOLING",
           "training_verdict", "labels_needed_estimate",
           "LABEL_EFFICIENCY_EVIDENCE"]


ADAPTATION_METHODS = {
    "pretrain": {"flops_per_param_token": 6.0, "trains_base": True,
                 "activation_multiplier": 1.0,
                 "note": "From scratch. Never on an edge device."},
    "full-finetune": {"flops_per_param_token": 6.0, "trains_base": True,
                      "activation_multiplier": 1.0,
                      "note": "All weights trainable. Optimiser state is 8 bytes"
                              "/param for Adam on top of everything else."},
    "lora": {"flops_per_param_token": 2.2, "trains_base": False,
             "activation_multiplier": 0.9,
             "note": "Freezes the base. Cuts trainable params by orders of "
                     "magnitude -- but NOT activation memory (arXiv:2604.22783)."},
    "qlora": {"flops_per_param_token": 2.2, "trains_base": False,
              "activation_multiplier": 0.9,
              "note": "4-bit base + higher-precision adapters. Cuts weight "
                      "memory; activation stack is unchanged."},
    "zeroth-order": {"flops_per_param_token": 4.0, "trains_base": False,
                     "activation_multiplier": 0.0,
                     "note": "PRGE (arXiv:2409.15520): gradients from forward "
                             "passes only. NO BACKWARD PASS means NO ACTIVATION "
                             "STACK -- removes the binding constraint entirely, "
                             "at the cost of many more steps."},
}

SHIPPING_TOOLING = {
    "llama.cpp finetune": {
        "hardware": "CPU, quantised GGUF",
        "status": "shipping",
        "note": "RAM-bound: ~16 GB trains 3B, ~32 GB trains 7B."},
    "QVAC-fabric-llm.cpp (Tether AI)": {
        "hardware": "mobile GPUs incl. ADRENO (Snapdragon), Mali, Apple",
        "status": "shipping, pre-built binaries, Dec 2025",
        "note": "Reports the first successful fine-tuning on mobile GPUs. "
                "Adreno is Snapdragon's GPU, so this is a direct path."},
    "MobileFineTuner": {
        "hardware": "phones, native C++",
        "status": "arXiv:2512.08211, Dec 2025",
        "note": "Autodiff and full backprop implemented natively; no Python."},
    "Federated LoRA (FedIT/FLoRA/HeLoRA/HetLoRA)": {
        "hardware": "fleets of heterogeneous devices",
        "status": "active research, production precedent in mobile keyboards",
        "note": "The fleet-scale version of the same idea."},
}


def adaptation_feasibility(params_b: float, method: str = "lora",
                           tokens: float = 1e6,
                           ram_gb: float = 16.0,
                           cpu_gops: float = 400.0,
                           seq_len: int = 2048,
                           batch: int = 1,
                           hidden_size: Optional[int] = None,
                           n_layers: Optional[int] = None,
                           base_bits: float = 4.0) -> Dict[str, Any]:
    """
    Is this adaptation job possible on this device, and WHICH constraint binds?

    Checks compute time AND activation memory, because the literature is
    explicit that parameter efficiency is not memory efficiency.
    """
    if method not in ADAPTATION_METHODS:
        raise ValueError(f"method must be one of {list(ADAPTATION_METHODS)}")
    if params_b <= 0 or tokens <= 0 or ram_gb <= 0:
        raise ValueError("params_b, tokens and ram_gb must be positive")

    m = ADAPTATION_METHODS[method]
    P = params_b * 1e9

    # --- compute ---
    flops = m["flops_per_param_token"] * P * tokens
    seconds = flops / (cpu_gops * 1e9)

    # --- memory: weights + optimiser + activations ---
    w_gb = P * base_bits / 8 / 1024 ** 3
    if m["trains_base"]:
        opt_gb = P * 8 / 1024 ** 3          # Adam m,v in fp32
        grad_gb = P * 2 / 1024 ** 3
    else:
        trainable = P * 0.02                # ~2% for typical LoRA rank
        opt_gb = trainable * 8 / 1024 ** 3
        grad_gb = trainable * 2 / 1024 ** 3

    # activation stack: ~ layers * seq * hidden * batch * bytes, retained for backward
    h = hidden_size or int(math.sqrt(P / 12) * 2)     # rough transformer shape
    L = n_layers or max(1, int(P / (12 * h * h)))
    act_gb = (m["activation_multiplier"] * L * seq_len * h * batch * 2
              * 6 / 1024 ** 3)              # ~6 tensors retained per layer

    total_gb = w_gb + opt_gb + grad_gb + act_gb
    usable = ram_gb * 0.70

    fits = total_gb <= usable
    if seconds < 60:
        t = f"{seconds:.0f} s"
    elif seconds < 3600:
        t = f"{seconds/60:.0f} min"
    elif seconds < 86400:
        t = f"{seconds/3600:.1f} h"
    elif seconds < 86400 * 365:
        t = f"{seconds/86400:.0f} days"
    else:
        t = f"{seconds/86400/365:.0f} years"

    # which constraint binds?
    time_ok = seconds <= 86400          # an overnight job is acceptable
    if not fits and not time_ok:
        binds = "both"
    elif not fits:
        binds = "memory"
    elif not time_ok:
        binds = "compute"
    else:
        binds = "neither"

    advice = {
        "memory": ("Activation memory is the blocker, not FLOPs. This is exactly "
                   "the arXiv:2604.22783 result. Options, in order: gradient "
                   "checkpointing (trades compute for memory), a shorter "
                   "sequence length, a smaller model, or ZEROTH-ORDER adaptation "
                   "(method='zeroth-order'), which has no activation stack at all."),
        "compute": ("FLOPs are the blocker. Reduce tokens, use a smaller model, "
                    "or run it overnight -- adaptation is not interactive."),
        "both": "Neither compute nor memory fits. Pick a substantially smaller model.",
        "neither": "Feasible on this device.",
    }[binds]

    return {
        "params_b": params_b, "method": method, "tokens": tokens,
        "compute_seconds": seconds, "compute_human": t,
        "memory_gb": {"weights": round(w_gb, 3), "optimiser": round(opt_gb, 3),
                      "gradients": round(grad_gb, 3),
                      "activations": round(act_gb, 3),
                      "total": round(total_gb, 3), "usable": round(usable, 2)},
        "fits_in_ram": fits, "feasible": fits and time_ok,
        "binding_constraint": binds, "advice": advice,
        "method_note": m["note"],
    }


def training_verdict(params_b: float, ram_gb: float = 16.0,
                     cpu_gops: float = 400.0) -> Dict[str, Any]:
    """
    Sweep every method for one model size and report what the device can do.
    This is the function that answers 'can I train on Snapdragon'.
    """
    params_b = _finite(params_b, "params_b", 1e-9, 1e6)
    ram_gb = _finite(ram_gb, "ram_gb", 1e-9)
    rows = []
    for meth, tok in (("pretrain", 1e12), ("full-finetune", 1e9),
                      ("lora", 1e6), ("qlora", 1e6), ("zeroth-order", 1e6)):
        try:
            rows.append(adaptation_feasibility(params_b, meth, tok,
                                               ram_gb, cpu_gops))
        except Exception as exc:                        # pragma: no cover
            rows.append({"method": meth, "error": str(exc)})
    feasible = [r for r in rows if r.get("feasible")]
    return {
        "params_b": params_b, "ram_gb": ram_gb,
        "results": rows,
        "feasible_methods": [r["method"] for r in feasible],
        "verdict": ("ADAPTATION ONLY -- the device can personalise, not train."
                    if feasible else
                    "NOTHING FEASIBLE at this size on this device."),
        "honest_pitch": (
            "Do not claim 'we train LLMs on Snapdragon'. Claim: inference on the "
            "NPU, adaptation on the idle CPU, and the user's data never leaves "
            "the device. That is true, defensible, and the one thing a cloud "
            "model structurally cannot do."),
    }


# --------------------------------------------------------------------------- #
# How FEW labels does on-device adaptation need? MAPA gives a hard number.
#
# "Pretraining for Sample-Efficient Neural Interfaces" (Tang, Spalding, Cogan,
# Duke; arXiv:2609.13507, Apache-2.0) is a masked autoencoder for intracranial
# neural data. Its domain is brain-computer interfaces, not language -- but the
# DEPLOYMENT PATTERN it validates is exactly the one this module argues for:
#
#     pretrain an encoder -> FREEZE it -> fit a tiny readout on very few labels
#
# The measured result: **~164 labelled trials to reach the accuracy that takes
# 3,500 without pretraining** -- a ~21x reduction -- and it sets state of the art
# on all three Neuroprobe regimes (within-session, cross-session, cross-subject)
# **without fine-tuning at all**, with the encoder frozen throughout.
#
# Why that matters here. The hard question for on-device personalisation is not
# "can the CPU do the arithmetic" -- `adaptation_feasibility()` shows it can. It
# is "will a user ever produce enough labelled corrections for adaptation to
# mean anything." MAPA answers that with a number from a different field: on the
# order of 10^2 labelled examples, not 10^4.
#
# A user correcting a document assistant produces a few hundred labels in normal
# use. That is the regime MAPA measures, and it lands on the feasible side.
#
# CAVEAT, because the transfer is by analogy: MAPA measures iEEG classification
# with a frozen encoder and a linear readout. A language model adapting via LoRA
# is a different task with a different label-efficiency curve. Treat 21x as
# evidence that the PATTERN works at small label counts, not as a number to
# quote for text.
# --------------------------------------------------------------------------- #

LABEL_EFFICIENCY_EVIDENCE = {
    "source": "MAPA, arXiv:2609.13507 (Duke, Apache-2.0)",
    "domain": "intracranial neural data (iEEG), NOT language",
    "labels_with_pretraining": 164,
    "labels_without": 3500,
    "reduction": 21.3,
    "regime": "frozen encoder + readout, no fine-tuning",
    "transfers": "The PATTERN (pretrain, freeze, tiny readout, ~10^2 labels) is "
                 "what on-device personalisation needs. The 21x figure itself is "
                 "domain-specific and should not be quoted for text tasks.",
}


def labels_needed_estimate(baseline_labels: int = 3500) -> Dict[str, Any]:
    """
    Order-of-magnitude estimate of labels needed for frozen-encoder adaptation.
    Deliberately returns a RANGE, because the supporting evidence is from a
    different domain.
    """
    if baseline_labels < 1:
        raise ValueError("baseline_labels must be positive")
    lo = max(1, int(baseline_labels / 30))
    hi = max(lo + 1, int(baseline_labels / 10))
    return {
        "baseline_labels": baseline_labels,
        "estimated_range": (lo, hi),
        "basis": LABEL_EFFICIENCY_EVIDENCE["source"],
        "confidence": "LOW -- cross-domain analogy from iEEG, not measured for "
                      "text. Use it to decide whether to TRY, not to promise.",
        "practical_read": (f"A few hundred user corrections is plausibly enough "
                           f"({lo}-{hi} labels). That is a realistic amount for "
                           f"normal use over days, not an unreachable target."),
    }


def _selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(n, c, d=""):
        checks.append((n, bool(c), d))

    pre = adaptation_feasibility(1.0, "pretrain", 1e12)
    ck("pretraining is impossible", not pre["feasible"],
       pre["compute_human"])
    ck("pretraining is compute-blocked",
       pre["binding_constraint"] in ("compute", "both"))

    ff = adaptation_feasibility(1.0, "full-finetune", 1e9)
    ck("full finetune infeasible", not ff["feasible"], ff["compute_human"])

    small = adaptation_feasibility(0.5, "lora", 1e5, ram_gb=16.0)
    ck("small LoRA is fast", small["compute_seconds"] < 600,
       small["compute_human"])

    zo = adaptation_feasibility(1.0, "zeroth-order", 1e6, ram_gb=8.0)
    lo = adaptation_feasibility(1.0, "lora", 1e6, ram_gb=8.0)
    ck("zeroth-order has no activation stack",
       zo["memory_gb"]["activations"] == 0.0)
    ck("zeroth-order uses less memory than LoRA",
       zo["memory_gb"]["total"] < lo["memory_gb"]["total"],
       f"{zo['memory_gb']['total']} vs {lo['memory_gb']['total']} GB")

    big = adaptation_feasibility(7.0, "lora", 1e6, ram_gb=8.0, seq_len=4096)
    ck("big LoRA on small RAM is memory-blocked",
       big["binding_constraint"] in ("memory", "both"), big["binding_constraint"])
    ck("memory advice names the paper's result",
       "not FLOPs" in adaptation_feasibility(
           7.0, "lora", 1e6, ram_gb=4.0, seq_len=4096)["advice"]
       or big["binding_constraint"] != "memory")

    ck("lora note says PEFT is not memory-efficient",
       "NOT activation memory" in ADAPTATION_METHODS["lora"]["note"])
    ck("zeroth-order note explains why it helps",
       "NO ACTIVATION STACK" in ADAPTATION_METHODS["zeroth-order"]["note"])

    v = training_verdict(0.5, ram_gb=16.0)
    ck("verdict is adaptation-only", "ADAPTATION ONLY" in v["verdict"])
    ck("verdict warns off the training claim",
       "Do not claim" in v["honest_pitch"])
    ck("some method is feasible at 0.5B", len(v["feasible_methods"]) > 0,
       str(v["feasible_methods"]))

    le = labels_needed_estimate()
    ck("label estimate is a range", le["estimated_range"][0] < le["estimated_range"][1])
    ck("label estimate confidence is LOW", le["confidence"].startswith("LOW"))
    ck("MAPA evidence flagged cross-domain",
       "NOT language" in LABEL_EFFICIENCY_EVIDENCE["domain"])
    ck("MAPA 21x not quotable for text",
       "should not be quoted" in LABEL_EFFICIENCY_EVIDENCE["transfers"])
    ck("labels rejects bad baseline", _raises(lambda: labels_needed_estimate(0)))

    ck("Adreno path recorded",
       "ADRENO" in SHIPPING_TOOLING["QVAC-fabric-llm.cpp (Tether AI)"]["hardware"])
    ck("four shipping tools listed", len(SHIPPING_TOOLING) == 4)

    for bad in (lambda: adaptation_feasibility(0, "lora"),
                lambda: adaptation_feasibility(1, "nope"),
                lambda: adaptation_feasibility(1, "lora", ram_gb=0)):
        pass
    ck("rejects zero params", _raises(lambda: adaptation_feasibility(0, "lora")))
    ck("rejects unknown method", _raises(lambda: adaptation_feasibility(1, "nope")))
    ck("rejects zero RAM", _raises(lambda: adaptation_feasibility(1, "lora", ram_gb=0)))

    print("\non_device_adaptation self test\n" + "=" * 68)
    n = 0
    for name, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {d}" if d else ""))
        n += ok
    print(f"\n  {n}/{len(checks)} passed\n")
    return 0 if n == len(checks) else 1


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verdict":
        pb = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        ram = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
        v = training_verdict(pb, ram)
        print(f"\n  {pb}B model, {ram} GB RAM\n")
        print(f"  {'method':<16}{'time':>12}{'mem GB':>9}{'binds':>10}  feasible")
        for r in v["results"]:
            print(f"  {r['method']:<16}{r['compute_human']:>12}"
                  f"{r['memory_gb']['total']:>9.2f}{r['binding_constraint']:>10}"
                  f"  {'yes' if r['feasible'] else 'no'}")
        print(f"\n  {v['verdict']}")
        print(f"\n  {v['honest_pitch']}\n")
    else:
        raise SystemExit(_selftest())
