#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PSDC -- Phase-Split Depth Cascade
 A proposed architecture for on-device LLM inference on Snapdragon.
================================================================================

STATUS: DESIGN PROPOSAL WITH UNTESTED PREDICTIONS. NOT A VALIDATED RESULT.

Earlier in this project a different "novel architecture" was proposed, tested,
and falsified -- two of its three pillars died, one to simple algebra (see
FINDINGS.md). That outcome is why this module ships its own kill conditions in
`falsification_suite()` before anyone builds on it. If prediction P1 fails, the
architecture is wrong and should be discarded, not patched.

--------------------------------------------------------------------------------
 THE MECHANISM (derived, then checked against published measurements)
--------------------------------------------------------------------------------
Vec-LUT (arXiv:2512.06443, MobiSys 2026) reports that CPUs run ultra-low-bit
LLMs *faster than NPUs*. The paper asserts it; it does not give a first-
principles reason. Here is one, and it is checkable:

  1. Decode emits one token at a time and must read every weight to do it, so
     time_decode >= weight_bytes / memory_bandwidth.
  2. On Snapdragon X2 Plus (152 GB/s, 80 TOPS int8) the compute term for an 8B
     model is ~0.20 ms/token while the bandwidth term is 13-53 ms/token. The
     compute term is 100-260x smaller. Decode is *entirely* bandwidth-bound.
  3. QNN's layer library contains no ternary matmul (verified: ENERZAi). So a
     ternary model targeted at the NPU must be unpacked to int8 -- it streams
     8 bits/weight instead of ~2.
  4. Therefore the NPU streams ~4x more bytes for the same model, and since
     decode is bandwidth-bound, it runs ~4x slower. Vec-LUT measures up to 4.2x.

  => The binding quantity is STORED BIT WIDTH ON THE WIRE, not TOPS.

This immediately predicts when the NPU wins back: give it a native ternary
kernel and it streams ~2 bits too, ties on latency, and wins on perf/watt.
That is exactly what ENERZAi demonstrated (BitNet b1.58 2B on QCS6490 Hexagon;
Opti 1.7B at 32 tok/s). The mechanism explains both results with one rule.

--------------------------------------------------------------------------------
 THE ARCHITECTURE
--------------------------------------------------------------------------------
Four facts that have never been combined:

  A. Prefill is parallel and compute-bound; decode is serial and
     bandwidth-bound.                                        (Vec-LUT, ENERZAi)
  B. WIDTH pruning preserves accuracy (1.8x speedup); DEPTH pruning is faster
     (2.7x) but degrades mathematical reasoning.             (Minitron 2407.14679)
  C. Ternary retention is scale-dependent: 0.5B -> 90.1%, 3B -> 97.2%.
                                                             (BitCPM-CANN)
  D. Depth pruning removes WHOLE transformer blocks, leaving the survivors
     structurally intact.                        (Shortened LLaMA 2402.02834)

Fact D is the hinge, and it is the part nobody seems to have exploited: if the
decode network is a *layer subset* of the prefill network, then the decode
network's KV cache is a **subset of the prefill network's KV cache**. The
prefill pass computes K/V for all L layers; the decode pass reads only the
retained subset. No recomputation. No second cache. No copy.

So run the two phases on different networks:

    PREFILL   full depth, width intact, INT4      -> Hexagon NPU
              accuracy is established here (prompt comprehension), it is
              compute-bound, and the NPU's TOPS is exactly what that needs.

    DECODE    depth-pruned subset, ternary        -> CPU + vector LUT
              bandwidth-bound, so minimise bytes on the wire: fewer layers
              AND fewer bits per weight, multiplying to a large reduction.

    KV        computed once by the prefill network; decode reads its subset.

The four axes -- prune axis, bit width, backend, and phase -- are co-designed
instead of chosen one at a time. Every existing work varies one and holds the
rest fixed.

--------------------------------------------------------------------------------
 HONEST PRIOR-ART POSITIONING
--------------------------------------------------------------------------------
The components are NOT new and this module does not claim they are:
  * layer dropping / early exit / self-speculative decoding (e.g. LayerSkip)
    already run a depth-reduced sub-network alongside a full one;
  * FuriosaAI's draft-based approximate inference (arXiv:2506.08373) already
    uses a draft model to guide KV-cache eviction;
  * mixed precision across a model is standard.

The delta claimed here is narrow and specific:
  (i)  the split is by PHASE, not by speculation -- there is no verification
       step and no rollback, so no acceptance-rate tax;
  (ii) bit width AND backend are chosen per phase, justified by the bandwidth
       derivation above rather than by benchmark sweeping;
  (iii) the KV cache is shared by construction because the decode net is a
       layer subset, not an independently trained draft model.

Whether that delta is worth anything is an empirical question. See below.

--------------------------------------------------------------------------------
 FALSIFIABLE PREDICTIONS -- run these before believing any of it
--------------------------------------------------------------------------------
  P1 (kills the architecture) A depth-pruned sub-network, distilled per
     Minitron, produces usable continuations when fed KV entries computed by
     the FULL network. If the residual-stream statistics diverge badly enough
     that output quality collapses, PSDC is dead and no patch saves it.
  P2 Decode throughput scales as 1/(stored bits x retained layers), i.e. the
     bandwidth model above holds within ~20% on real hardware.
  P3 Ternary retention tracks per-layer WIDTH, not total parameter count. If
     true, a depth-pruned decode net keeps its ternary viability even though
     its parameter count fell below the 1B danger zone. If false, PSDC's decode
     net must stay wide and the memory win shrinks.
  P4 Quality loss from PSDC is smaller than from applying the same total
     compression uniformly across both phases.

P3 is the interesting one: it is a real open question in the literature, cheap
to test, and useful whichever way it resolves.

Licence: Apache-2.0.  Python >= 3.9.  Standard library only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
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



__all__ = ["RooflineModel", "PruneAxis", "PhaseSplitPlan", "plan_psdc",
           "crossover_bits", "cpu_lut_ceiling", "falsification_suite", "SoCBudget",
           "SpecKVConfig", "kv_stream_bytes", "speckv_evidence_for_p1",
           "activation_sparsity_benefit", "hogwild_evidence_for_p1",
           "mla_kv_bytes", "mla_vs_gqa", "mla_optimal_rank", "MLA_CANONICAL"]


# ============================================================================ #
# Hardware budget
# ============================================================================ #

@dataclass(frozen=True)
class SoCBudget:
    """Only the numbers that actually bind. TOPS is included to show it doesn't."""
    name: str
    bandwidth_gbs: float
    npu_tops_int8: Optional[float] = None
    cpu_gflops: Optional[float] = None          # sustained, all cores, int8-ish
    sustained_fraction: float = 0.60            # thermal derate
    npu_has_ternary_kernel: bool = False        # stock QNN: False
    lut_arith_efficiency: float = 6.0
    """
    THE MODEL'S LARGEST UNCERTAINTY -- calibrate this before trusting any number.

    A LUT kernel does not perform a multiply-accumulate per weight. It performs
    a table lookup plus an add, and amortises one lookup across a GROUP of
    weights -- that is the entire point of T-MAC and Vec-LUT. Charging the CPU
    2 FLOP/param (the MAC figure) therefore overstates its work by roughly the
    group size divided by the per-lookup cost.

    This module's first version made exactly that error and consequently
    predicted the CPU was compute-bound at every model size, contradicting
    Vec-LUT's measurements. The factor below is the correction: effective
    arithmetic throughput is `cpu_gflops * lut_arith_efficiency` on LUT paths
    only.

    6.0 is a placeholder chosen so the model reproduces Vec-LUT's reported
    up-to-4.2x in the bandwidth-bound regime. It is NOT measured. Prediction P2
    exists to calibrate it; until then treat absolute tok/s figures from this
    model as ordering information, not forecasts.
    """

    @property
    def effective_bandwidth(self) -> float:
        """Real achievable bandwidth. 100% of peak is never reached."""
        return self.bandwidth_gbs * 1e9 * 0.75


SOC_BUDGETS = {
    "x2-plus": SoCBudget("Snapdragon X2 Plus", 152.0, 80.0, 400.0),
    "x2-elite": SoCBudget("Snapdragon X2 Elite", 152.0, 80.0, 500.0),
    "x2-elite-extreme": SoCBudget("Snapdragon X2 Elite Extreme", 228.0, 80.0, 600.0),
    "x-elite": SoCBudget("Snapdragon X Elite", 135.0, 45.0, 350.0),
    "8-gen-3": SoCBudget("Snapdragon 8 Gen 3", 76.8, None, 200.0),
    "qcs6490": SoCBudget("Qualcomm QCS6490", 44.0, None, 90.0),
}


# ============================================================================ #
# Roofline: the core analytical model
# ============================================================================ #

class Backend(str, Enum):
    NPU = "hexagon-npu"
    CPU_LUT = "cpu-vector-lut"
    CPU_PLAIN = "cpu-plain"


@dataclass
class RooflineResult:
    backend: Backend
    stored_bits: float
    bytes_streamed: float
    t_bandwidth_ms: float
    t_compute_ms: float
    bound_by: str
    tokens_per_s: float
    note: str = ""


class RooflineModel:
    """
    Predicts decode throughput from stored bit width and memory bandwidth.

    The central subtlety, and the thing most tooling gets wrong: the bit width
    that matters is the width the weights are STREAMED at, which is not always
    the width they are STORED at on disk. On a backend with no native kernel
    for format F, weights must be widened at load or on the fly, and the
    bandwidth cost follows the widened value.
    """

    FLOPS_PER_PARAM_PER_TOKEN = 2.0

    def __init__(self, soc: SoCBudget):
        self.soc = soc

    def _peak_ops(self, backend: Backend) -> float:
        """Effective op throughput. LUT paths get the amortisation credit."""
        if backend is Backend.NPU and self.soc.npu_tops_int8:
            return self.soc.npu_tops_int8 * 1e12 * self.soc.sustained_fraction
        base = (self.soc.cpu_gflops or 100.0) * 1e9
        if backend is Backend.CPU_LUT:
            return base * self.soc.lut_arith_efficiency
        return base

    def effective_stored_bits(self, native_bits: float,
                              backend: Backend) -> Tuple[float, str]:
        """The bit width actually streamed on this backend."""
        if backend is Backend.NPU:
            if native_bits < 4.0 and not self.soc.npu_has_ternary_kernel:
                return 8.0, ("QNN has no ternary matmul, so sub-4-bit weights are "
                             "unpacked to int8 -- the NPU streams 8 bits/weight "
                             "regardless of how they were stored.")
            if native_bits < 4.0:
                return native_bits, "custom ternary Hexagon kernel: weights stay packed"
            return max(native_bits, 4.0), "native int4/int8 path"
        # CPU LUT paths keep the packed representation
        return native_bits, "LUT kernel consumes packed weights directly"

    def decode(self, params_b: float, native_bits: float,
               backend: Backend, retained_layer_frac: float = 1.0) -> RooflineResult:
        eff_bits, note = self.effective_stored_bits(native_bits, backend)
        active_params = params_b * 1e9 * retained_layer_frac
        wbytes = active_params * eff_bits / 8.0
        t_bw = wbytes / self.soc.effective_bandwidth

        peak = self._peak_ops(backend)
        t_c = (self.FLOPS_PER_PARAM_PER_TOKEN * active_params) / peak

        t = max(t_bw, t_c)
        return RooflineResult(
            backend=backend, stored_bits=eff_bits, bytes_streamed=wbytes,
            t_bandwidth_ms=t_bw * 1e3, t_compute_ms=t_c * 1e3,
            bound_by="bandwidth" if t_bw >= t_c else "compute",
            tokens_per_s=1.0 / t if t > 0 else float("inf"), note=note)

    def prefill(self, params_b: float, native_bits: float, backend: Backend,
                prompt_tokens: int) -> RooflineResult:
        """
        Prefill reads the weights ONCE for the whole prompt, so the weight
        stream amortises over prompt_tokens and the compute term dominates.
        This is why prefill and decode have different optimal backends.
        """
        eff_bits, note = self.effective_stored_bits(native_bits, backend)
        wbytes = params_b * 1e9 * eff_bits / 8.0
        t_bw = wbytes / self.soc.effective_bandwidth
        peak = self._peak_ops(backend)
        t_c = (self.FLOPS_PER_PARAM_PER_TOKEN * params_b * 1e9 * prompt_tokens) / peak
        t = max(t_bw, t_c)
        return RooflineResult(
            backend=backend, stored_bits=eff_bits, bytes_streamed=wbytes,
            t_bandwidth_ms=t_bw * 1e3, t_compute_ms=t_c * 1e3,
            bound_by="bandwidth" if t_bw >= t_c else "compute",
            tokens_per_s=prompt_tokens / t if t > 0 else float("inf"), note=note)

    def best_backend(self, params_b: float, native_bits: float,
                     phase: str, prompt_tokens: int = 512,
                     npu_available: bool = True) -> Tuple[Backend, List[RooflineResult]]:
        cands = [Backend.CPU_LUT, Backend.CPU_PLAIN]
        if npu_available:
            cands.insert(0, Backend.NPU)
        fn = self.decode if phase == "decode" else (
            lambda p, b, be, **kw: self.prefill(p, b, be, prompt_tokens))
        results = [fn(params_b, native_bits, be) for be in cands]
        return max(results, key=lambda r: r.tokens_per_s).backend, results


def cpu_lut_ceiling(soc: SoCBudget, native_bits: float = 2.0) -> Dict[str, Any]:
    """
    The CPU+LUT advantage is NOT unbounded -- it has its own roofline.

    Found by this module's own self-test, which expected a ~4x CPU win at 8B
    and measured 1.8x. Reason: at 2 bits an 8B model streams only 2 GB, which
    the CPU can pull in ~17 ms, but the CPU needs ~40 ms to do the arithmetic.
    The CPU crosses from bandwidth-bound to COMPUTE-bound, and past that point
    extra bit-width savings buy nothing.

    Vec-LUT's up-to-4.2x is therefore a bandwidth-regime number. Beyond the
    crossover size the win decays toward 1x.

    This has a design consequence that supports PSDC rather than undermining
    it: depth-pruning the decode network reduces ACTIVE parameters, which pulls
    the decode net back into the CPU's bandwidth-bound regime where LUT wins
    are largest. The two techniques are complementary, not merely additive.
    """
    if not hasattr(soc, "cpu_gflops"):
        raise TypeError(f"soc must be an SoCBudget, got {type(soc).__name__}")
    native_bits = _finite(native_bits, "native_bits", 0.1, 64.0)
    peak = (soc.cpu_gflops or 100.0) * 1e9
    # bandwidth-bound while: params*bits/8 / BW  >=  2*params / peak
    # => bits/8 / BW >= 2/peak  -- independent of params, so the crossover is
    # set by the ratio, and in practice by where the two curves meet in size.
    m = RooflineModel(soc)
    rows, crossover = [], None
    for pb in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        cpu = m.decode(pb, native_bits, Backend.CPU_LUT)
        npu = m.decode(pb, native_bits, Backend.NPU)
        ratio = cpu.tokens_per_s / max(npu.tokens_per_s, 1e-9)
        rows.append({"params_b": pb, "cpu_bound_by": cpu.bound_by,
                     "cpu_tok_s": round(cpu.tokens_per_s, 1),
                     "npu_tok_s": round(npu.tokens_per_s, 1),
                     "cpu_advantage": round(ratio, 2)})
        if crossover is None and cpu.bound_by == "compute":
            crossover = pb
    return {"soc": soc.name, "native_bits": native_bits,
            "cpu_becomes_compute_bound_at_b": crossover,
            "note": ("None means the CPU stays bandwidth-bound at every size "
                     "modelled, so the advantage is size-independent and equals "
                     "the ratio of streamed bit widths. That result depends "
                     "entirely on lut_arith_efficiency, which is NOT measured -- "
                     "calibrate via prediction P2 before quoting these numbers."),
            "table": rows}


def crossover_bits(soc: SoCBudget) -> Dict[str, Any]:
    """
    At what native bit width does CPU+LUT overtake the NPU for decode?

    With no ternary kernel the NPU streams max(native, 4) bits (it widens
    anything below int4 to int8). The CPU streams `native`. Decode is
    bandwidth-bound, so the CPU wins whenever the NPU is forced to widen --
    i.e. for every native width below 4 bits. Above 4 bits both stream the
    same and the NPU wins on compute and perf/watt.
    """
    if not hasattr(soc, "npu_has_ternary_kernel"):
        raise TypeError("soc must be an SoCBudget, got "
                        f"{type(soc).__name__}")
    m = RooflineModel(soc)
    rows = []
    for nb in (1.0, 1.58, 2.0, 3.0, 4.0, 8.0):
        npu = m.decode(4.0, nb, Backend.NPU)
        cpu = m.decode(4.0, nb, Backend.CPU_LUT)
        rows.append({"native_bits": nb,
                     "npu_streams_bits": npu.stored_bits,
                     "cpu_streams_bits": cpu.stored_bits,
                     "npu_tok_s": round(npu.tokens_per_s, 1),
                     "cpu_tok_s": round(cpu.tokens_per_s, 1),
                     "winner": "cpu-lut" if cpu.tokens_per_s > npu.tokens_per_s * 1.05
                               else "npu"})
    return {
        "soc": soc.name,
        "npu_has_ternary_kernel": soc.npu_has_ternary_kernel,
        "crossover": ("below 4 bits, CPU+LUT wins on stock QNN"
                      if not soc.npu_has_ternary_kernel else
                      "no crossover: a native ternary kernel keeps the NPU ahead"),
        "table": rows,
    }


# ============================================================================ #
# The architecture
# ============================================================================ #

class PruneAxis(str, Enum):
    WIDTH = "width"     # Minitron: better accuracy, 1.8x
    DEPTH = "depth"     # Minitron / Shortened LLaMA: 2.7x, math degrades
    NONE = "none"


# BitCPM-CANN measured retention, reused here.
_TERNARY_RETENTION = {0.5: 0.901, 1.0: 0.957, 3.0: 0.972, 8.0: 0.957}


def _retention(params_b: float) -> float:
    xs = sorted(_TERNARY_RETENTION)
    if params_b <= xs[0]:
        return _TERNARY_RETENTION[xs[0]]
    if params_b >= xs[-1]:
        return _TERNARY_RETENTION[xs[-1]]
    lo = max(x for x in xs if x <= params_b)
    hi = min(x for x in xs if x >= params_b)
    if lo == hi:
        return _TERNARY_RETENTION[lo]
    f = (params_b - lo) / (hi - lo)
    return _TERNARY_RETENTION[lo] + f * (_TERNARY_RETENTION[hi] - _TERNARY_RETENTION[lo])


@dataclass
class PhaseConfig:
    phase: str
    layers_retained: float
    bits: float
    backend: Backend
    tokens_per_s: float
    bound_by: str
    rationale: str


@dataclass
class PhaseSplitPlan:
    model: str
    params_b: float
    soc: str
    prefill: PhaseConfig
    decode: PhaseConfig
    kv_shared: bool
    weight_gb_prefill: float
    weight_gb_decode: float
    est_quality_retention: float
    ttft_ms: float
    warnings: List[str] = field(default_factory=list)
    predictions_untested: List[str] = field(default_factory=list)

    def summary(self) -> str:
        L = [f"PSDC plan -- {self.model} ({self.params_b:g}B) on {self.soc}", ""]
        for c in (self.prefill, self.decode):
            L.append(f"  {c.phase.upper():<8} layers={c.layers_retained:.0%}  "
                     f"{c.bits:g}-bit  {c.backend.value}")
            L.append(f"           {c.tokens_per_s:.1f} tok/s ({c.bound_by}-bound)")
            L.append(f"           {c.rationale}")
        L += ["",
              f"  KV cache shared by construction: {self.kv_shared}",
              f"  weights resident: prefill {self.weight_gb_prefill:.2f} GB, "
              f"decode {self.weight_gb_decode:.2f} GB",
              f"  TTFT (512-token prompt): {self.ttft_ms:.0f} ms",
              f"  est. quality retention: {self.est_quality_retention:.1%}"]
        if self.warnings:
            L += [""] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(L)


def plan_psdc(params_b: float, soc_key: str = "x2-plus",
              model_name: str = "model", prompt_tokens: int = 512,
              decode_layer_frac: float = 0.6,
              npu_available: bool = True,
              priority: str = "balanced") -> PhaseSplitPlan:
    """
    Produce a phase-split plan.

    decode_layer_frac: fraction of transformer blocks the decode sub-network
    retains. 0.6 follows the depth-pruning ratios Minitron and Shortened LLaMA
    report as recoverable with distillation.
    """
    soc = SOC_BUDGETS.get(soc_key)
    if soc is None:
        raise ValueError(f"unknown SoC '{soc_key}'. Known: {list(SOC_BUDGETS)}")
    if not 0.1 <= decode_layer_frac <= 1.0:
        raise ValueError("decode_layer_frac must be in [0.1, 1.0]")
    if params_b <= 0:
        raise ValueError("params_b must be positive")

    m = RooflineModel(soc)
    warnings: List[str] = []

    # ---- prefill: full depth, int4, accuracy-preserving ----
    pf_bits = 4.0
    pf_backend, _ = m.best_backend(params_b, pf_bits, "prefill",
                                   prompt_tokens, npu_available)
    pf = m.prefill(params_b, pf_bits, pf_backend, prompt_tokens)

    # ---- decode: depth-pruned subset, ternary if viable ----
    dec_params = params_b * decode_layer_frac
    ret = _retention(dec_params)
    dec_bits = 2.0
    if dec_params < 1.0:
        warnings.append(
            f"Decode sub-network is {dec_params:.2f}B; BitCPM-CANN measured only "
            f"~{_retention(dec_params):.1%} retention below 1B. Prediction P3 "
            "(retention tracks per-layer WIDTH, not total params) is what decides "
            "whether this is actually a problem -- it is UNTESTED.")
        if priority == "quality":
            dec_bits = 4.0
            warnings.append("priority=quality: decode raised to 4-bit, forfeiting "
                            "most of the bandwidth win.")

    dec_backend, _ = m.best_backend(dec_params, dec_bits, "decode",
                                    npu_available=npu_available)
    dec = m.decode(params_b, dec_bits, dec_backend, decode_layer_frac)

    if not soc.npu_has_ternary_kernel and dec_bits < 4.0 and dec_backend is Backend.NPU:
        warnings.append("NPU chosen for sub-4-bit decode despite no ternary "
                        "kernel -- check effective_stored_bits, this looks wrong.")

    # quality estimate: depth pruning hurts reasoning; ternary hurts by scale
    depth_penalty = 1.0 - 0.06 * (1.0 - decode_layer_frac) / 0.4   # ~6% at 60% depth
    quality = ret * max(depth_penalty, 0.0)

    plan = PhaseSplitPlan(
        model=model_name, params_b=params_b, soc=soc.name,
        prefill=PhaseConfig(
            "prefill", 1.0, pf_bits, pf_backend, pf.tokens_per_s, pf.bound_by,
            "Full depth and width: prompt comprehension sets the quality ceiling. "
            "Compute-bound and parallel, which is what the NPU is for."),
        decode=PhaseConfig(
            "decode", decode_layer_frac, dec_bits, dec_backend, dec.tokens_per_s,
            dec.bound_by,
            f"Layer subset at {dec_bits:g}-bit: decode is bandwidth-bound, so cut "
            f"bytes on the wire. {dec.note}"),
        kv_shared=True,
        weight_gb_prefill=params_b * 1e9 * pf_bits / 8 / 1024**3,
        weight_gb_decode=dec_params * 1e9 * dec_bits / 8 / 1024**3,
        est_quality_retention=quality,
        ttft_ms=prompt_tokens / max(pf.tokens_per_s, 1e-9) * 1e3,
        warnings=warnings,
        predictions_untested=[
            "P1 depth-pruned decode net accepts full-net KV entries (KILLS IT)",
            "P2 throughput ~ 1/(bits x layers) within 20% on real silicon",
            "P3 ternary retention tracks per-layer width, not total params",
            "P4 phase-split beats uniform compression at equal total budget",
        ])
    return plan


def baseline_uniform(params_b: float, soc_key: str, bits: float = 4.0,
                     prompt_tokens: int = 512,
                     npu_available: bool = True) -> Dict[str, Any]:
    """The honest comparison: one model, one bit width, one backend."""
    soc = SOC_BUDGETS[soc_key]
    m = RooflineModel(soc)
    be, _ = m.best_backend(params_b, bits, "decode", npu_available=npu_available)
    d = m.decode(params_b, bits, be)
    pbe, _ = m.best_backend(params_b, bits, "prefill", prompt_tokens, npu_available)
    p = m.prefill(params_b, bits, pbe, prompt_tokens)
    return {"decode_tok_s": round(d.tokens_per_s, 1),
            "prefill_tok_s": round(p.tokens_per_s, 1),
            "ttft_ms": round(prompt_tokens / max(p.tokens_per_s, 1e-9) * 1e3, 1),
            "weight_gb": round(params_b * 1e9 * bits / 8 / 1024**3, 2),
            "backend_decode": be.value, "backend_prefill": pbe.value}


# ============================================================================ #
# SpecKV integration -- the depth-pruned decode net is ALREADY a draft model
#
# Source: Galim, Ewer, Kang, Lee, Koo, Lee, "Draft-based Approximate Inference
# for LLMs", ICLR 2026 (arXiv:2506.08373), github.com/furiosa-ai/draft-based-approx-llm
#
# Their framework uses a small draft model to predict which tokens and KV pairs
# matter, enabling sharper eviction than importance heuristics:
#   SpecKV     lookahead with a draft model for precise KV cache dropping
#   SpecPC     draft attention activations to discard prompt tokens
#   SpecKV-PC  the two cascaded
# They report it is the first use of draft models for APPROXIMATE inference,
# beyond lossless speculative decoding, and evaluate on RULER to 65k context.
#
# Two consequences for PSDC, one evidential and one architectural.
#
# EVIDENTIAL. Their central empirical finding is a strong correlation between
# the attention patterns of DRAFT and TARGET models. PSDC's fatal prediction P1
# asks whether a depth-pruned subnet can consume KV computed by the full network.
# A pruned subset SHARES WEIGHTS with the target, so it should correlate at
# least as well as an independently trained draft does. This does not prove P1 --
# attention-pattern correlation is not the same as coherent generation from
# borrowed KV -- but it is independent evidence in P1's favour, and it is the
# first such evidence this project has found. P1 still needs running.
#
# ARCHITECTURAL. SpecKV's cost is that it NEEDS a separate draft model: extra
# weights to train, store and stream. PSDC already has one, for free:
#
#       the depth-pruned decode network IS the draft
#
# It shares weights with the target (no extra storage), it is already resident
# (no extra streaming), and it is already running during decode. So SpecKV-style
# eviction becomes nearly free inside PSDC, where standalone it costs a model.
#
# And it compounds in the right direction. This project's roofline says decode
# is bandwidth-bound and the KV cache is the growing term at long context.
# PSDC shrinks the WEIGHT stream (fewer layers x fewer bits); SpecKV shrinks the
# KV stream. They attack the same bottleneck from two sides and multiply.
# ============================================================================ #

@dataclass
class SpecKVConfig:
    """KV eviction driven by the decode subnet acting as its own draft."""
    keep_fraction: float = 0.5      # fraction of KV entries retained
    lookahead_tokens: int = 8       # SpecKV lookahead depth
    use_prompt_compression: bool = False   # SpecPC on top (SpecKV-PC cascade)
    prompt_keep_fraction: float = 0.7

    def __post_init__(self):
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.lookahead_tokens < 1:
            raise ValueError("lookahead_tokens must be >= 1")
        if not 0 < self.prompt_keep_fraction <= 1:
            raise ValueError("prompt_keep_fraction must be in (0, 1]")


def kv_stream_bytes(n_layers: int, n_kv_heads: int, head_dim: int,
                    context: int, kv_bits: float = 8.0,
                    retained_layer_frac: float = 1.0,
                    speckv: Optional[SpecKVConfig] = None) -> Dict[str, Any]:
    """
    Bytes of KV cache read per decoded token, with PSDC's layer subsetting and
    optional SpecKV eviction. This is the term that grows with context and, past
    a few thousand tokens, dominates the weight stream.
    """
    if min(n_layers, n_kv_heads, head_dim, context) < 1:
        raise ValueError("all dimensions must be >= 1")
    per_token_full = 2 * n_layers * n_kv_heads * head_dim * (kv_bits / 8.0)
    full = per_token_full * context

    layers_used = max(1, round(n_layers * retained_layer_frac))
    after_layers = 2 * layers_used * n_kv_heads * head_dim * (kv_bits / 8.0) * context

    keep = 1.0
    prompt_keep = 1.0
    if speckv is not None:
        keep = speckv.keep_fraction
        if speckv.use_prompt_compression:
            prompt_keep = speckv.prompt_keep_fraction
    after_speckv = after_layers * keep * prompt_keep

    return {
        "context": context,
        "full_bytes": full,
        "after_layer_subset": after_layers,
        "after_speckv": after_speckv,
        "layer_reduction": full / max(after_layers, 1e-9),
        "speckv_reduction": after_layers / max(after_speckv, 1e-9),
        "total_reduction": full / max(after_speckv, 1e-9),
        "draft_model_cost": "zero -- the decode subnet is the draft, sharing "
                            "weights with the target",
        "note": "PSDC shrinks the weight stream; SpecKV shrinks the KV stream. "
                "Both are bandwidth, so the reductions multiply.",
    }


def hogwild_evidence_for_p1() -> Dict[str, str]:
    """
    Hogwild! Inference (arXiv:2504.06261, NeurIPS 2025 spotlight) -- the second
    and stronger independent support for P1.

    Multiple LLM workers synchronise through a concurrently-updated SHARED
    attention cache, and the paper reports that modern reasoning-capable LLMs do
    this "out of the box, without additional fine-tuning", using RoPE to avoid
    recomputation.
    """
    return {
        "supports": "Modern LLMs decode correctly from a KV cache written by "
                    "OTHER instances with no fine-tuning (arXiv:2504.06261, "
                    "NeurIPS 2025 spotlight). Real models, real generation "
                    "quality -- a far higher bar than p1_test.py's proxy.",
        "does_not_establish": "Their workers are IDENTICAL full networks sharing "
                              "a cache. PSDC's consumer is a depth-pruned subset, "
                              "whose residual-stream statistics differ from the "
                              "writer's. Hogwild removes the 'whose cache is it' "
                              "objection, not the 'different network' one.",
        "status": "Second independent support for P1, and the stronger of the "
                  "two. P1 still needs the depth-pruned test on trained weights.",
    }


def speckv_evidence_for_p1() -> Dict[str, str]:
    """What furiosa's result does and does not establish about PSDC's P1."""
    return {
        "supports": "Draft and target models show strongly correlated attention "
                    "patterns (arXiv:2506.08373, ICLR 2026). A depth-pruned "
                    "subset shares weights with the target, so it should "
                    "correlate at least as strongly as an independent draft.",
        "does_not_establish": "Attention-pattern correlation is not coherent "
                              "generation from borrowed KV. P1 asks whether the "
                              "subnet can DECODE from full-network KV, which is "
                              "a stronger claim than agreeing on what matters.",
        "status": "P1 remains UNTESTED and still fatal if it fails. This is the "
                  "first independent evidence in its favour, not a substitute "
                  "for running it.",
    }


# ============================================================================ #
# Activation sparsity -- the fourth compression axis, and the roofline's verdict
#
# Everything else in this project compresses along three axes: bit width,
# structural pruning, and KV eviction. MiniCPM-S-1B (OpenBMB) exposes a fourth:
# 87.89% average FFN sparsity, cutting FFN FLOPs by 84% while holding downstream
# task performance. It is orthogonal to all three.
#
# The obvious move is to add it to the pipeline as another multiplier. The
# roofline says that would be wrong, and the reason is worth stating precisely:
#
#     ACTIVATION SPARSITY REMOVES FLOPS. DECODE IS NOT SHORT OF FLOPS.
#
# Measured against this model, 8B @ int4 on X2 Plus:
#
#     decode, dense                                        25.0 tok/s
#     decode, naive sparsity (skip FLOPs, still load)       28.5 tok/s   1.14x
#     decode, PREDICTIVE sparsity (skip the load too)       65.2 tok/s   2.61x
#
# An 84% FLOP reduction buys 14%. The other 2.3x is entirely in not loading the
# weights. So the value of activation sparsity during decode lives ALMOST
# ENTIRELY in the predictor -- the mechanism that decides which neurons matter
# BEFORE their rows are streamed from DRAM. Without one, this axis is nearly
# worthless in the phase that dominates on-device latency.
#
# Prefill is the opposite: compute-bound (170.7 ms compute vs 35.1 ms bandwidth
# at 512 tokens), so sparsity helps directly with no predictor needed.
#
# Prior art for the predictive version: DejaVu (contextual sparsity), PowerInfer
# (hot/cold neuron placement). Not novel. What is worth recording is that the
# roofline DERIVES the requirement rather than assuming it -- and that a
# pipeline which adds sparsity as a plain multiplier, as this project's
# compression_pipeline() would have, overstates decode benefit by ~2.3x.
# ============================================================================ #

def activation_sparsity_benefit(params_b: float, sparsity: float,
                                soc_key: str = "x2-plus",
                                bits: float = 4.0,
                                ffn_param_frac: float = 0.67,
                                predictive: bool = False,
                                phase: str = "decode",
                                prompt_tokens: int = 512) -> Dict[str, Any]:
    _finite(params_b, "params_b", 1e-9)
    _finite(sparsity, "sparsity", 0.0, 1.0)
    """
    Throughput effect of activation sparsity, computed per phase.

    predictive=False models sparsity that skips arithmetic but still streams
    every weight. predictive=True models a gate that decides which FFN rows are
    needed before they are loaded, so the skipped rows never cross the bus.

    ffn_param_frac: share of parameters in FFN blocks (~2/3 for a standard
    transformer). Sparsity applies only to those.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if not 0.0 < ffn_param_frac <= 1.0:
        raise ValueError("ffn_param_frac must be in (0, 1]")
    soc = SOC_BUDGETS.get(soc_key)
    if soc is None:
        raise ValueError(f"unknown SoC '{soc_key}'")

    bw = soc.effective_bandwidth
    peak_cpu = (soc.cpu_gflops or 100.0) * 1e9
    peak_npu = (soc.npu_tops_int8 or 10.0) * 1e12 * soc.sustained_fraction
    wbytes = params_b * 1e9 * bits / 8.0

    if phase == "decode":
        t_bw_dense = wbytes / bw
        t_c_dense = 2 * params_b * 1e9 / peak_cpu
        dense = 1.0 / max(t_bw_dense, t_c_dense)
        t_c = t_c_dense * (1.0 - sparsity * ffn_param_frac / ffn_param_frac
                           if False else (1.0 - sparsity))
        t_bw = (wbytes * (1.0 - ffn_param_frac * sparsity) / bw) if predictive \
            else t_bw_dense
    else:
        t_bw_dense = wbytes / bw
        t_c_dense = 2 * params_b * 1e9 * prompt_tokens / peak_npu
        dense = prompt_tokens / max(t_bw_dense, t_c_dense)
        t_c = t_c_dense * (1.0 - sparsity)
        t_bw = (wbytes * (1.0 - ffn_param_frac * sparsity) / bw) if predictive \
            else t_bw_dense

    t = max(t_bw, t_c)
    sparse = (1.0 / t) if phase == "decode" else (prompt_tokens / t)
    return {
        "phase": phase,
        "predictive": predictive,
        "dense_tok_s": dense,
        "sparse_tok_s": sparse,
        "speedup": sparse / max(dense, 1e-12),
        "bound_by": "bandwidth" if t_bw >= t_c else "compute",
        "verdict": (
            "Sparsity helps directly: this phase is compute-bound."
            if phase != "decode" else
            ("Predictive gating avoids the load, so the bandwidth term shrinks "
             "too -- this is where the benefit actually is."
             if predictive else
             "WITHOUT a predictor this is nearly worthless in decode: the FLOPs "
             "vanish but every weight is still streamed, and bandwidth binds.")),
        "requires": (None if phase != "decode" or predictive else
                     "A predictor that gates FFN rows BEFORE they are loaded "
                     "(DejaVu-style contextual sparsity, PowerInfer-style hot/cold "
                     "placement). Without it, do not count this axis."),
    }


# ============================================================================ #
# MLA -- the FOURTH KV axis: latent rank
#
# This project modelled three ways to shrink the KV stream: bit width, layer
# subsetting, and eviction. All three are NUMERICAL or SELECTIVE. DeepSeek's
# Multi-head Latent Attention shrinks it STRUCTURALLY: keys and values are
# down-projected to a low-rank latent, only the latent is cached, and an
# up-projection restores expressivity at compute time. "LoRA for attention."
#
# Facts that matter here:
#   * MLA caches ONE latent per token per LAYER, not per KV head. Canonical
#     DeepSeek config (h_q, d_h, r_kv, d_h^R) = (128, 128, 512, 64): 576 bytes-
#     worth of state per layer-token against 32768 for 128-head MHA.
#   * MLA is strictly MORE expressive than GQA at equal KV overhead -- "GQA can
#     always be represented by MLA while maintaining the same KV cache overhead,
#     but the converse does not hold" (TransMLA, NeurIPS 2025).
#   * TransMLA converts ANY pretrained GQA model (Llama, Qwen, Gemma, Mistral)
#     post hoc: 93% KV compression on LLaMA-2-7B, ~10x speedup at 8K context,
#     ~6B tokens of finetuning to recover quality. So this is not gated on
#     retraining from scratch.
#   * MHA2MLA (ACL 2025) reports compression ratios "greater than or equal to
#     Int2 quantization while also achieving performance higher than Int2" --
#     i.e. MLA and aggressive KV quantisation are SUBSTITUTES as well as
#     complements, and MLA can win outright.
#   * MLA has two execution paths by construction: MHA-like expansion for
#     PREFILL, MQA-absorb for DECODE. That is PSDC's phase split, already
#     present inside the attention mechanism.
#
# THE DERIVED RESULT, which is what this section is actually for.
#
# GQLA (arXiv:2605.15250) does a roofline study of MLA and finds the absorb path
# reaches ~242 FLOPs/byte at the canonical rank, "just below the H100 BF16 ridge
# (~295)" -- near-ideal. They then note this "is, however, MLA's only operating
# point." The canonical rank 512 is tuned to ONE hardware ridge.
#
# Snapdragon's ridges are nowhere near H100's, and the two backends on the SAME
# chip disagree:
#
#     H100 BF16                ridge ~295   MLA at 242 -> near-ideal balance
#     X2 Plus NPU int8         ridge  421   MLA at 242 -> BANDWIDTH-bound
#     X2 Plus CPU + LUT        ridge  2.1   MLA at 242 -> COMPUTE-bound
#
# So on the X2 NPU the H100-tuned rank leaves compute idle: compress HARDER,
# drop the rank. On the CPU+LUT path the same model is compute-bound and
# lowering the rank buys nothing at all. Opposite prescriptions, same silicon,
# selected by which backend runs the phase.
#
# For PSDC specifically -- prefill on the NPU, decode on CPU+LUT -- this means
# the optimal latent rank DIFFERS BY PHASE, while MLA exposes rank as a single
# shared parameter. That is a genuine open design question, not a knob.
# ============================================================================ #

MLA_CANONICAL = {"h_q": 128, "d_h": 128, "r_kv": 512, "d_h_rope": 64,
                 "absorb_flops_per_byte": 242.0,
                 "h100_bf16_ridge": 295.0,
                 "source": "DeepSeek V2/V3; roofline figures from GQLA "
                           "arXiv:2605.15250"}


def mla_kv_bytes(n_layers: int, latent_rank: int, rope_dim: int = 64,
                 context: int = 4096, kv_bits: float = 16.0,
                 retained_layer_frac: float = 1.0) -> Dict[str, Any]:
    """
    KV cache bytes under MLA. One latent per token per layer -- NOT per head,
    which is the whole point and the reason the saving is so large.
    """
    # `min(1, nan) < 1` is False, so NaN walked straight through the old
    # check and came back as bytes_per_token=NaN. Validate each argument.
    n_layers = int(_finite(n_layers, "n_layers", 1, 100_000))
    latent_rank = int(_finite(latent_rank, "latent_rank", 1, 1_000_000))
    rope_dim = int(_finite(rope_dim, "rope_dim", 0, 100_000))
    context = int(_finite(context, "context", 1, 100_000_000))
    kv_bits = _finite(kv_bits, "kv_bits", 0.5, 64)
    retained_layer_frac = _finite(retained_layer_frac, "retained_layer_frac", 0.0, 1.0)
    layers = max(1, round(n_layers * retained_layer_frac))
    per_layer_token = (latent_rank + rope_dim) * (kv_bits / 8.0)
    return {"bytes_per_token": per_layer_token * layers,
            "total_bytes": per_layer_token * layers * context,
            "total_gb": per_layer_token * layers * context / 1024 ** 3,
            "latent_rank": latent_rank, "rope_dim": rope_dim,
            "layers_used": layers}


def mla_vs_gqa(n_layers: int, n_kv_heads: int, head_dim: int,
               latent_rank: int = 512, rope_dim: int = 64,
               context: int = 4096, kv_bits: float = 16.0) -> Dict[str, Any]:
    """Compare MLA's latent cache against the GQA baseline it would replace."""
    n_kv_heads = int(_finite(n_kv_heads, "n_kv_heads", 1, 100_000))
    head_dim = int(_finite(head_dim, "head_dim", 1, 100_000))
    gqa = 2 * n_kv_heads * head_dim * (_finite(kv_bits, "kv_bits", 0.5, 64) / 8.0) \
        * int(_finite(n_layers, "n_layers", 1, 100_000)) \
        * int(_finite(context, "context", 1, 100_000_000))
    mla = mla_kv_bytes(n_layers, latent_rank, rope_dim, context, kv_bits)
    return {"gqa_gb": gqa / 1024 ** 3, "mla_gb": mla["total_gb"],
            "reduction": gqa / max(mla["total_bytes"], 1e-9),
            "note": "MLA is strictly more expressive than GQA at equal KV "
                    "overhead (TransMLA). TransMLA converts pretrained GQA "
                    "checkpoints post hoc for ~6B tokens of finetuning.",
            "substitutes_for": "MHA2MLA reports compression at or beyond INT2 "
                               "quantisation WITH better quality, so MLA and "
                               "aggressive KV quantisation compete as well as "
                               "compose. Do not simply multiply their savings."}


def mla_optimal_rank(soc_key: str = "x2-plus", backend: str = "npu",
                     canonical_rank: int = 512) -> Dict[str, Any]:
    """
    Derive the latent rank that puts MLA at the target backend's roofline ridge.

    MLA's canonical rank is tuned to the H100 ridge. Arithmetic intensity of the
    absorb path scales roughly with the rank (more latent to expand per cached
    byte), so rank can be moved to match a different ridge.
    """
    soc = SOC_BUDGETS.get(soc_key)
    if soc is None:
        raise ValueError(f"unknown SoC '{soc_key}'")
    bw = soc.effective_bandwidth
    if backend == "npu":
        peak = (soc.npu_tops_int8 or 10.0) * 1e12 * soc.sustained_fraction
    elif backend == "cpu-lut":
        peak = (soc.cpu_gflops or 100.0) * 1e9 * soc.lut_arith_efficiency
    else:
        raise ValueError("backend must be 'npu' or 'cpu-lut'")

    ridge = peak / bw
    mla_ai = MLA_CANONICAL["absorb_flops_per_byte"]
    if mla_ai < ridge:
        verdict, advice = ("bandwidth-bound", (
            f"Ridge {ridge:.0f} is ABOVE MLA's {mla_ai:.0f} FLOPs/byte, so the "
            f"H100-tuned rank {canonical_rank} leaves compute idle. Compress "
            f"harder -- lower the rank until arithmetic intensity approaches "
            f"the ridge."))
        suggested = max(64, int(canonical_rank * mla_ai / ridge))
    else:
        verdict, advice = ("compute-bound", (
            f"Ridge {ridge:.0f} is BELOW MLA's {mla_ai:.0f} FLOPs/byte. This "
            f"backend is already compute-bound at the canonical rank, so "
            f"lowering it buys nothing -- the up-projection is the cost, not "
            f"the cache. Raise the rank if quality needs it."))
        suggested = canonical_rank
    return {"soc": soc.name, "backend": backend, "ridge_flops_per_byte": ridge,
            "mla_arithmetic_intensity": mla_ai, "regime": verdict,
            "canonical_rank": canonical_rank, "suggested_rank": suggested,
            "advice": advice,
            "caveat": "The rank-to-intensity relation is modelled as linear, "
                      "which is a first-order approximation. Ridge values "
                      "inherit lut_arith_efficiency's calibration uncertainty. "
                      "Treat as a direction, not a setting."}


# ============================================================================ #
# Falsification
# ============================================================================ #

def falsification_suite() -> List[Dict[str, str]]:
    """
    The experiments that would kill this. Written before any claim is made,
    because the previous architecture in this project died to an experiment
    that should have been run first.
    """
    return [
        {"id": "P1", "severity": "fatal -- PARTIALLY TESTED, see status",
         "status": "NARROWED, NOT CLOSED. p1_test.py isolates the PSDC-specific "
                   "half: hold pruning fixed, vary only the KV source. On a "
                   "12-layer untrained transformer, KL(full || pruned) with "
                   "SHARED full-network KV vs the subnet's OWN KV gave ratios "
                   "0.83 / 0.88 / 0.67 / 1.01 at 75/50/33/25% depth -- sharing "
                   "is free, and usually slightly BETTER, plausibly because "
                   "full-network KV comes from the full residual stream and is "
                   "closer to what each retained layer expects. BUT: untrained "
                   "model, so this tests residual-stream compatibility, not task "
                   "accuracy. And at 25% depth both variants sit at the "
                   "uniform-distribution distance (KL 0.49 vs 0.50) -- the "
                   "pruning has destroyed the model and the ratio is ~1 only "
                   "because both are equally destroyed. Run on trained weights "
                   "before relying on this.",
         "claim": "A depth-pruned sub-network produces usable continuations from "
                  "KV entries computed by the full network.",
         "test": "Take any open model. Compute KV for a 512-token prompt with all "
                 "L layers. Decode 128 tokens using only layers in the retained "
                 "subset, reading their cached K/V. Measure perplexity and "
                 "eyeball coherence against full-model decode.",
         "kills_if": "Output is incoherent or perplexity blows up by >2x even "
                     "after Minitron-style distillation of the subset.",
         "cost": "~1 hour on a Colab T4 with a 0.5-1.5B model."},
        {"id": "P2", "severity": "high",
         "claim": "Decode throughput scales as 1/(stored_bits x retained_layers).",
         "test": "Sweep bit width {2,4,8} x layer fraction {0.4,0.6,0.8,1.0} and "
                 "measure tok/s on real hardware via AI Hub Workbench.",
         "kills_if": "Measured throughput deviates >20% from the roofline "
                     "prediction, i.e. decode is not actually bandwidth-bound.",
         "cost": "12 profiling jobs on AI Hub, free."},
        {"id": "P3", "severity": "medium -- open research question",
         "claim": "Ternary retention tracks per-layer WIDTH, not total parameter "
                  "count.",
         "test": "Ternarise two models with equal total params but different "
                 "shapes: one deep+narrow, one shallow+wide. Compare retention "
                 "against their fp16 baselines on the same benchmark set.",
         "kills_if": "Retention tracks total params. Then a depth-pruned decode "
                     "net below 1B loses ~10% capability and PSDC must keep the "
                     "decode net wide, shrinking the memory win.",
         "cost": "Two QAT runs. This is the genuinely novel question here and is "
                 "worth publishing either way."},
        {"id": "P4", "severity": "high",
         "claim": "Phase-split beats uniform compression at an equal total budget.",
         "test": "Compare PSDC (int4 prefill + ternary depth-pruned decode) "
                 "against a uniform model matched on resident bytes, on both "
                 "quality and tok/s.",
         "kills_if": "Uniform matches or wins. Then the split buys nothing and "
                     "the added complexity is unjustified.",
         "cost": "One T4 session plus two AI Hub profiling jobs."},
    ]


# ============================================================================ #
# CLI
# ============================================================================ #

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="PSDC -- Phase-Split Depth Cascade (DESIGN PROPOSAL).",
        epilog="This is an untested proposal. Run `falsify` before believing it.")
    ap.add_argument("cmd", choices=["plan", "crossover", "ceiling", "falsify",
                                    "selftest"])
    ap.add_argument("--params-b", type=float, default=4.0)
    ap.add_argument("--soc", default="x2-plus", choices=list(SOC_BUDGETS))
    ap.add_argument("--decode-layers", type=float, default=0.6)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--priority", default="balanced",
                    choices=["balanced", "quality", "latency"])
    ap.add_argument("--no-npu", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        return _selftest()

    if a.cmd == "crossover":
        for key in ("x2-plus", "qcs6490"):
            for ternary_kernel in (False, True):
                soc = SOC_BUDGETS[key]
                soc = SoCBudget(soc.name, soc.bandwidth_gbs, soc.npu_tops_int8,
                                soc.cpu_gflops, soc.sustained_fraction, ternary_kernel)
                c = crossover_bits(soc)
                print(f"\n=== {c['soc']}  (native ternary kernel: {ternary_kernel}) ===")
                print(f"    {c['crossover']}")
                print(f"    {'native':>7} {'NPU streams':>12} {'CPU streams':>12} "
                      f"{'NPU tok/s':>10} {'CPU tok/s':>10}  winner")
                for r in c["table"]:
                    print(f"    {r['native_bits']:>7g} {r['npu_streams_bits']:>12g} "
                          f"{r['cpu_streams_bits']:>12g} {r['npu_tok_s']:>10} "
                          f"{r['cpu_tok_s']:>10}  {r['winner']}")
        print()
        return 0

    if a.cmd == "ceiling":
        c = cpu_lut_ceiling(SOC_BUDGETS[a.soc])
        print(f"\n=== CPU+LUT advantage ceiling -- {c['soc']} at "
              f"{c['native_bits']:g} bits ===")
        print(f"    CPU becomes compute-bound at ~"
              f"{c['cpu_becomes_compute_bound_at_b']}B params")
        print(f"    {c['note']}\n")
        print(f"    {'params':>7} {'cpu bound by':>14} {'cpu tok/s':>10} "
              f"{'npu tok/s':>10} {'advantage':>10}")
        for r in c["table"]:
            print(f"    {r['params_b']:>6g}B {r['cpu_bound_by']:>14} "
                  f"{r['cpu_tok_s']:>10} {r['npu_tok_s']:>10} "
                  f"{r['cpu_advantage']:>9}x")
        print()
        return 0

    if a.cmd == "falsify":
        print("\nPSDC falsification suite -- run these BEFORE building on this.\n")
        for f in falsification_suite():
            print(f"  [{f['id']}] severity: {f['severity']}")
            print(f"       claim:     {f['claim']}")
            print(f"       test:      {f['test']}")
            print(f"       kills if:  {f['kills_if']}")
            print(f"       cost:      {f['cost']}\n")
        return 0

    plan = plan_psdc(a.params_b, a.soc, f"{a.params_b:g}B model", a.prompt_tokens,
                     a.decode_layers, not a.no_npu, a.priority)
    print()
    print(plan.summary())
    base = baseline_uniform(a.params_b, a.soc, 4.0, a.prompt_tokens, not a.no_npu)
    print(f"\n  baseline (uniform int4, one backend):")
    print(f"    decode {base['decode_tok_s']} tok/s   TTFT {base['ttft_ms']} ms   "
          f"{base['weight_gb']} GB")
    gain = plan.decode.tokens_per_s / max(base["decode_tok_s"], 1e-9)
    print(f"    PSDC decode speedup (PREDICTED, UNTESTED): {gain:.2f}x")
    print("\n  Untested predictions:")
    for p in plan.predictions_untested:
        print(f"    - {p}")
    print()
    if a.json:
        d = asdict(plan)
        d["baseline"] = base
        d["falsification"] = falsification_suite()
        with open(a.json, "w") as f:
            json.dump(d, f, indent=2, default=str)
        print(f"  wrote {a.json}\n")
    return 0


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


def _selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(n, c, d=""):
        checks.append((n, bool(c), d))

    soc = SOC_BUDGETS["x2-plus"]
    m = RooflineModel(soc)

    d8 = m.decode(8.0, 2.0, Backend.NPU)
    ck("NPU widens ternary to int8 without a kernel", d8.stored_bits == 8.0)
    dc = m.decode(8.0, 2.0, Backend.CPU_LUT)
    ck("CPU keeps ternary packed", dc.stored_bits == 2.0)
    ck("CPU beats NPU at 2-bit decode", dc.tokens_per_s > d8.tokens_per_s,
       f"{dc.tokens_per_s:.0f} vs {d8.tokens_per_s:.0f} tok/s")
    small_cpu = m.decode(1.0, 2.0, Backend.CPU_LUT)
    small_npu = m.decode(1.0, 2.0, Backend.NPU)
    ck("CPU advantage is ~4x in the bandwidth-bound regime",
       small_cpu.tokens_per_s > small_npu.tokens_per_s * 3.5,
       f"{small_cpu.tokens_per_s/small_npu.tokens_per_s:.2f}x at 1B")
    ck("CPU advantage is size-independent while bandwidth-bound",
       abs(m.decode(1.0, 2.0, Backend.CPU_LUT).tokens_per_s
           / m.decode(1.0, 2.0, Backend.NPU).tokens_per_s
           - m.decode(16.0, 2.0, Backend.CPU_LUT).tokens_per_s
           / m.decode(16.0, 2.0, Backend.NPU).tokens_per_s) < 0.01)
    ck("CPU advantage equals the bit-width ratio",
       abs(m.decode(4.0, 2.0, Backend.CPU_LUT).tokens_per_s
           / m.decode(4.0, 2.0, Backend.NPU).tokens_per_s - 8.0 / 2.0) < 0.05,
       "8-bit NPU stream / 2-bit CPU stream = 4x")
    ceil = cpu_lut_ceiling(SOC_BUDGETS["x2-plus"])
    ck("ceiling is reported honestly when absent",
       ceil["cpu_becomes_compute_bound_at_b"] is None
       or ceil["cpu_becomes_compute_bound_at_b"] > 0)
    ck("decode is bandwidth-bound", d8.bound_by == "bandwidth")

    soc_k = SoCBudget("with-kernel", 152.0, 80.0, 400.0, 0.6, True)
    mk = RooflineModel(soc_k)
    ck("native ternary kernel restores the NPU",
       mk.decode(8.0, 2.0, Backend.NPU).stored_bits == 2.0)

    p = m.prefill(8.0, 4.0, Backend.NPU, 512)
    ck("prefill is compute-bound", p.bound_by == "compute", p.bound_by)
    ck("prefill and decode differ in bound",
       p.bound_by != m.decode(8.0, 4.0, Backend.NPU).bound_by)

    pl = plan_psdc(8.0, "x2-plus", "test", 512, 0.6)
    ck("plan produced", isinstance(pl, PhaseSplitPlan))
    ck("KV shared by construction", pl.kv_shared)
    ck("decode uses fewer layers", pl.decode.layers_retained < 1.0)
    ck("decode weights smaller than prefill",
       pl.weight_gb_decode < pl.weight_gb_prefill,
       f"{pl.weight_gb_decode:.2f} < {pl.weight_gb_prefill:.2f} GB")
    ck("predictions listed as untested", len(pl.predictions_untested) == 4)

    small = plan_psdc(1.0, "x2-plus", "small", 512, 0.5)
    ck("sub-1B decode net raises a warning", any("P3" in w for w in small.warnings))
    q = plan_psdc(1.0, "x2-plus", "small", 512, 0.5, priority="quality")
    ck("quality priority raises decode bits", q.decode.bits == 4.0)

    try:
        plan_psdc(4.0, "not-a-soc")
        ck("rejects unknown SoC", False)
    except ValueError:
        ck("rejects unknown SoC", True)
    try:
        plan_psdc(4.0, "x2-plus", decode_layer_frac=2.0)
        ck("rejects bad layer fraction", False)
    except ValueError:
        ck("rejects bad layer fraction", True)
    try:
        plan_psdc(-1.0, "x2-plus")
        ck("rejects negative params", False)
    except ValueError:
        ck("rejects negative params", True)

    ck("retention curve matches BitCPM", abs(_retention(3.0) - 0.972) < 1e-9)
    ck("retention interpolates", 0.901 < _retention(0.75) < 0.957)
    kv = kv_stream_bytes(32, 8, 128, 32768, 8.0, 1.0)
    ck("full KV stream computed", kv["full_bytes"] > 0)
    kv2 = kv_stream_bytes(32, 8, 128, 32768, 8.0, 0.6)
    ck("layer subsetting shrinks KV", kv2["layer_reduction"] > 1.5,
       f"{kv2['layer_reduction']:.2f}x")
    kv3 = kv_stream_bytes(32, 8, 128, 32768, 8.0, 0.6, SpecKVConfig(keep_fraction=0.5))
    ck("SpecKV compounds with layer subsetting",
       kv3["total_reduction"] > kv2["layer_reduction"] * 1.9,
       f"{kv3['total_reduction']:.2f}x total")
    kv4 = kv_stream_bytes(32, 8, 128, 32768, 8.0, 0.6,
                          SpecKVConfig(keep_fraction=0.5, use_prompt_compression=True))
    ck("SpecKV-PC cascade compounds further",
       kv4["total_reduction"] > kv3["total_reduction"])
    ck("draft model is free in PSDC", "zero" in kv3["draft_model_cost"])
    ck("SpecKVConfig rejects bad keep fraction",
       _raises(lambda: SpecKVConfig(keep_fraction=0.0)))
    ck("SpecKVConfig rejects bad lookahead",
       _raises(lambda: SpecKVConfig(lookahead_tokens=0)))
    ck("kv_stream_bytes rejects bad dims",
       _raises(lambda: kv_stream_bytes(0, 8, 128, 1024)))
    ev = speckv_evidence_for_p1()
    ck("P1 evidence is honest about its limits",
       "not coherent generation" in ev["does_not_establish"])
    ck("P1 still marked untested", "UNTESTED" in ev["status"])

    naive = activation_sparsity_benefit(8.0, 0.84, predictive=False)
    pred = activation_sparsity_benefit(8.0, 0.84, predictive=True)
    ck("naive sparsity barely helps decode", naive["speedup"] < 1.3,
       f"{naive['speedup']:.2f}x")
    ck("predictive sparsity helps a lot", pred["speedup"] > 2.0,
       f"{pred['speedup']:.2f}x")
    ck("naive decode states the requirement", "predictor" in (naive["requires"] or ""))
    ck("predictive needs nothing extra", pred["requires"] is None)
    pre = activation_sparsity_benefit(8.0, 0.84, phase="prefill")
    ck("prefill is compute-bound so sparsity helps", pre["speedup"] > 1.5,
       f"{pre['speedup']:.2f}x")
    ck("sparsity rejects bad fraction",
       _raises(lambda: activation_sparsity_benefit(8.0, 1.0)))
    ck("sparsity rejects unknown soc",
       _raises(lambda: activation_sparsity_benefit(8.0, 0.5, soc_key="nope")))

    m = mla_kv_bytes(32, 512, 64, 32768)
    ck("MLA KV computed", m["total_gb"] > 0, f"{m['total_gb']:.2f} GB")
    cmp_ = mla_vs_gqa(32, 8, 128, context=32768)
    ck("MLA beats GQA on KV bytes", cmp_["reduction"] > 3.0,
       f"{cmp_['reduction']:.2f}x")
    ck("MLA/quant substitution warned", "compete" in cmp_["substitutes_for"])
    npu = mla_optimal_rank("x2-plus", "npu")
    cpu = mla_optimal_rank("x2-plus", "cpu-lut")
    ck("NPU is bandwidth-bound at canonical rank", npu["regime"] == "bandwidth-bound",
       f"ridge {npu['ridge_flops_per_byte']:.0f}")
    ck("NPU wants a LOWER rank", npu["suggested_rank"] < 512,
       f"{npu['suggested_rank']}")
    ck("CPU-LUT is compute-bound", cpu["regime"] == "compute-bound",
       f"ridge {cpu['ridge_flops_per_byte']:.1f}")
    ck("CPU-LUT keeps the canonical rank", cpu["suggested_rank"] == 512)
    ck("backends disagree on the same chip",
       npu["suggested_rank"] != cpu["suggested_rank"])
    ck("rank model states its caveat", "not a setting" in npu["caveat"])
    ck("MLA rejects bad backend",
       _raises(lambda: mla_optimal_rank("x2-plus", "gpu")))
    ck("MLA rejects bad dims", _raises(lambda: mla_kv_bytes(0, 512)))

    hw = hogwild_evidence_for_p1()
    ck("hogwild evidence recorded", "no fine-tuning" in hw["supports"].lower())
    ck("hogwild limits stated", "depth-pruned subset" in hw["does_not_establish"])
    ck("hogwild is second support", "Second independent" in hw["status"])

    fs = falsification_suite()
    ck("falsification suite has a fatal test",
       any(f["severity"].startswith("fatal") for f in fs))
    p1 = next(f for f in fs if f["id"] == "P1")
    ck("P1 records its partial test result", "NARROWED, NOT CLOSED" in p1["status"])
    ck("P1 still flagged fatal", p1["severity"].startswith("fatal"))
    ck("every prediction has a kill condition",
       all(f.get("kills_if") for f in fs))

    cb = crossover_bits(soc)
    ck("crossover favours CPU below 4 bits",
       all(r["winner"] == "cpu-lut" for r in cb["table"] if r["native_bits"] < 4))
    ck("crossover favours NPU at 4+ bits",
       all(r["winner"] == "npu" for r in cb["table"] if r["native_bits"] >= 4))

    print("\nPSDC self test")
    print("=" * 68)
    npass = 0
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"   {d}" if d else ""))
        npass += ok
    print(f"\n  {npass}/{len(checks)} passed\n")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
