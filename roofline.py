#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 roofline.py -- the scaling-book roofline, checked against its own worked
                examples, then tested against Qualcomm's MEASURED Snapdragon X
                numbers. No GPU, no network, no downloads, no dependencies.
================================================================================

    python roofline.py book        # reproduce the book's worked answers first
    python roofline.py law         # decode time vs bytes, on Qualcomm's data
    python roofline.py bcrit       # effective critical batch = prefill/decode
    python roofline.py context     # where KV traffic overtakes weight traffic
    python roofline.py engines     # why prefill and decode want different engines
    python roofline.py runtimes    # same silicon, same weights, different software
    python roofline.py critical    # spec-sheet critical batch per SoC and format
    python roofline.py ceiling --model qwen3_4b --context 4096
    python roofline.py speculate   # free verification window per engine
    python roofline.py predict     # falsifiable predictions for X Plus 8-Core
    python roofline.py container   # the container rule, corrected for context
    python roofline.py selftest

SOURCE. "How to Scale Your Model" (jax-ml.github.io/scaling-book), Parts 1, 4
and 7: rooflines, transformer arithmetic, inference. Everything here is single
-device arithmetic -- the sharding and interconnect chapters do not apply to a
laptop and are not used.

WHY THIS FILE EXISTS. The rest of this project ASSERTS that decode on Snapdragon
is bandwidth-bound and that the stored bit width is what binds. That was argued
from spec sheets. This file does two things the argument lacked:

  1. It implements the book's formulas and then reproduces the book's OWN
     worked answers -- 240, 295, 120, 6.7 GB, 18.4B, 262 kB, 2.5 ms, 21 ms --
     before trusting itself on anything else. Same discipline as
     validate_against_paper() in residual_cascade.py.

  2. It tests the formulas against measurements taken on real Snapdragon X
     silicon by Qualcomm, which ship INSIDE the `qai_hub_models` pip package
     (models/<id>/perf.yaml). Nothing is downloaded: the numbers are embedded
     below as a verified snapshot, and when the package is installed the
     snapshot is re-checked against it.

WHAT THE MEASUREMENTS SAY (all reproducible with the commands above)

  * Decode time is LINEAR IN BYTES MOVED PER TOKEN. Across Qwen3 0.6B -> 8B on
    X Elite's NPU (GenieX QAIRT, w4a16, 4K context) R^2 >= 0.99 under every
    one of six byte-accounting assumptions, intercept within a few ms of zero,
    effective bandwidth 43-51% of the 135 GB/s peak.

  * The ENGINE barely matters for decode and matters enormously for prefill --
    the book's compute/bandwidth asymmetry, measured. On X2 Elite, Qwen3-4B at
    512 tokens decodes at 33.7 (CPU), 33.5 (GPU) and 36.2 tok/s (NPU, QAIRT),
    while prefill spans 461 -> 2307 tok/s.

  * PREFILL / DECODE IS THE EFFECTIVE CRITICAL BATCH. If prefill is
    compute-bound and decode bandwidth-bound, the ratio of their throughputs
    is exactly C_eff / W_eff x bytes/2 -- the book's B_crit -- so it must not
    depend on model size. It does not: 62.8-70.6 across three Qwen3 sizes on
    the X2 Elite NPU (CV 0.05); 13.7-17.1 on the CPU of the same chip. Read
    straight off a published table, it says how many draft tokens a
    speculative verifier checks for the price of one step on each engine.

  * Beyond ~10K tokens of context (Qwen3-4B, X2 Elite) KV traffic outweighs
    weight traffic. Measured crossovers are 0.65-1.13x what the book's KV
    formula predicts with a 16-bit cache. An 8-bit cache would need attention
    to run at under 0.6x the weight stream's efficiency to fit the same data,
    so the simplest reading is that the cache is not compressed -- and KV
    quantisation is the untaken lever for long documents.

  * Software can cost 3.2x on identical silicon and weights (Genie vs GenieX
    QAIRT, Qwen3-4B, X Elite). The fit's INTERCEPT diagnoses it: ~0 ms when a
    runtime is bandwidth-bound, 40-60 ms when per-token overhead dominates.

  * Two published pairs are UNPHYSICAL. On X2 Elite (QAIRT), Qwen3-8B is 8%
    slower than Qwen3-4B while moving 77% more bytes: a marginal 612 GB/s,
    beyond even the 228 GB/s Extreme part. On X Elite (Genie), Qwen3-4B
    decodes FASTER than Qwen3-1.7B. Both are flagged, not explained; the
    member the rest of the data disagrees with is left out of the fit, chosen
    by leave-one-out rather than by guessing. On X2 Elite that is the 4B at
    4096 tokens -- not the 8B, which was the guess -- and the same point also
    bends its own context curve upward. Under Genie it is the 1.7B.

  * Snapdragon X Plus 8-Core CRD -- the AI Hub proxy for FOUR of the seven HP
    Snapdragon machines -- is listed as supported for 221 models and measured
    for NONE. predict() turns the roofline into falsifiable numbers for it;
    aihub_workbench.py measures it without downloading anything.

A CORRECTION THIS FILE FORCED. The project previously stated that Qualcomm
publishes no Compute performance numbers at all. That was checked against the
AI Hub website, which showed empty tables. The package data carries 491
measured entries for X Elite CRD and 487 for X2 Elite CRD. The claim was wrong
for those two devices and right only for X Plus 8-Core. PROVENANCE.md records
it as the ninth prediction of absence that failed.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "critical_intensity", "critical_batch", "critical_batch_beta_alpha",
    "matmul_intensity", "kv_bytes_per_token", "dense_params", "min_step_time",
    "step_time", "attention_decode_intensity", "validate_against_book",
    "Arch", "ARCHS", "Soc", "SOCS", "bytes_per_token", "verify_geometry",
    "QUALCOMM_MEASURED", "fit_line", "bandwidth_law", "bandwidth_law_table",
    "effective_critical_batch", "context_slope", "engine_asymmetry",
    "runtime_spread", "anomalies", "decode_ceiling", "kv_crossover",
    "speculation", "predict_x_plus", "container_vs_context",
    "load_measured_from_package", "snapshot_matches_package",
]


# ============================================================================ #
# plumbing
# ============================================================================ #

def _finite(value, name: str, lo=None, hi=None) -> float:
    """
    Reject a numeric argument that is not finite, before it propagates.

    NaN and infinity do not raise -- they SPREAD, and a roofline fed NaN
    returns a confident-looking ceiling about nothing. The stress harness
    flags exactly that as the worst failure class, because nobody notices it.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got bool")
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


def _pos(value, name: str, hi=None) -> float:
    v = _finite(value, name, 0.0, hi)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


# Physical ceilings. A roofline input outside these is a typo, not hardware,
# and letting it through lets float overflow return inf -- silently, which the
# stress harness counts as the worst failure there is. The fuzzer found the
# first one: attention_decode_intensity(1e308) returned inf.
MAX_OPS_S = 1e20          # a zettaop per second
MAX_BYTES_S = 1e16        # ten petabytes per second
MAX_BYTES = 1e18          # an exabyte
MAX_DIM = 1e9


def _int(value, name: str, lo: int = 1, hi: int = 10 ** 9) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


# ============================================================================ #
# SECTION 1 -- the book's formulas
# ============================================================================ #
#
# Notation follows the book: B batch (tokens processed together), D model
# width, F FFN width, N query heads, K KV heads, H head dim, L layers, V vocab,
# T context tokens. A multiply-add counts as 2 FLOPs throughout.

TPU_V5E = {"bf16_flops": 1.97e14, "int8_ops": 3.94e14, "hbm_bytes_s": 8.2e11}
H100 = {"bf16_flops": 9.89e14, "hbm_bytes_s": 3.35e12}


def critical_intensity(peak_ops_s, bandwidth_bytes_s) -> float:
    """Part 1: the arithmetic intensity at which a chip stops being
    memory-bound. FLOPs (or ops) per byte."""
    peak = _pos(peak_ops_s, "peak_ops_s", MAX_OPS_S)
    bw = _finite(bandwidth_bytes_s, "bandwidth_bytes_s", 1.0, MAX_BYTES_S)
    return peak / bw


def matmul_intensity(B, D, F, bytes_per_elem: float = 2.0) -> float:
    """
    Part 1: [B,D] x [D,F] reads 2BD + 2DF bytes, writes 2BF, and does 2BDF
    FLOPs, so the intensity is BDF / (BD + DF + BF) -- which is ~B when
    B << D, F. That approximation is the whole reason batch-1 decode is slow.
    """
    b, d, f = (_pos(B, "B", MAX_DIM), _pos(D, "D", MAX_DIM), _pos(F, "F", MAX_DIM))
    e = _pos(bytes_per_elem, "bytes_per_elem", 16.0)
    return (2 * b * d * f) / (e * (b * d + d * f + b * f))


def critical_batch(peak_ops_s, bandwidth_bytes_s, weight_bits: float = 16.0) -> float:
    """
    Tokens per weight read at which a weight-dominated matmul turns
    compute-bound.

    Derivation. At batch B a matmul with F x D weights does 2BDF ops and moves
    DF x bytes_per_weight bytes (activations are negligible when B << D, F).
    Intensity = 2B / bytes_per_weight. Setting that equal to the chip's
    critical intensity C/W gives B_crit = (C/W) x bytes_per_weight / 2.

    `peak_ops_s` must be the peak of the dtype the MACs actually run in. The
    book's cases, reproduced in validate_against_book():
        bf16 weights, bf16 math  -> 240   (TPU v5e)
        int8 weights, bf16 math  -> 120   (half: each byte feeds 2x the FLOPs)
        int8 weights, int8 math  -> 240   (int8 peak is 2x bf16)
    """
    wb = _pos(weight_bits, "weight_bits", 64.0) / 8.0
    return critical_intensity(peak_ops_s, bandwidth_bytes_s) * wb / 2.0


def critical_batch_beta_alpha(bits_per_param, bits_per_activation,
                              alpha_hbm) -> float:
    """
    Part 7's stated form: B_crit = beta x alpha_hbm, beta = bits per param /
    bits per activation, alpha_hbm = C / W with C the bf16 peak. Kept so the
    two forms can be asserted equal on the book's cases rather than assumed.
    """
    beta = _pos(bits_per_param, "bits_per_param", 64) / _finite(
        bits_per_activation, "bits_per_activation", 1.0, 64)
    return beta * _pos(alpha_hbm, "alpha_hbm", MAX_OPS_S)


def kv_bytes_per_token(attn_layers, kv_heads, head_dim,
                       bytes_per_elem: float = 2.0) -> float:
    """
    Part 7: KV cache size = 2 x bytes x H x K x L x T, so per token
    2 x bytes x H x K x L. `attn_layers` counts FULL-attention layers only --
    a hybrid model's linear-attention layers hold a fixed-size state that does
    not grow with T (Bonsai 2: 16 of 64 layers carry KV).
    """
    return (2.0 * _pos(bytes_per_elem, "bytes_per_elem", 8)
            * _int(head_dim, "head_dim", 1, 65536)
            * _int(kv_heads, "kv_heads", 1, 4096)
            * _int(attn_layers, "attn_layers", 1, 100_000))


def dense_params(L, D, F, N, K, H, V=0, tied: bool = True) -> Dict[str, float]:
    """
    Part 4: per layer, gated MLP 3DF and attention 2DH(N + K) (Q and O are
    D x NH, K and V are D x KH). Norms are negligible and omitted. The vocab
    matrix counts once if tied, twice if not.
    """
    L_, D_, F_ = _int(L, "L", 1, 10 ** 5), _int(D, "D", 1, 10 ** 7), _int(F, "F", 1, 10 ** 8)
    N_, K_, H_ = _int(N, "N", 1, 10 ** 5), _int(K, "K", 1, 10 ** 5), _int(H, "H", 1, 10 ** 6)
    V_ = _int(V, "V", 0, 10 ** 8)
    if K_ > N_:
        raise ValueError(f"K ({K_}) cannot exceed N ({N_})")
    mlp = 3.0 * D_ * F_ * L_
    attn = 2.0 * D_ * H_ * (N_ + K_) * L_
    vocab = float(D_ * V_) * (1 if tied else 2)
    return {"mlp": mlp, "attention": attn, "vocab": vocab,
            "non_embedding": mlp + attn, "total": mlp + attn + vocab}


def min_step_time(batch, kv_bytes_per_seq, param_bytes, bandwidth_bytes_s) -> float:
    """Part 7: theoretical minimum step time, seconds, when everything is
    bandwidth-bound: (batch x KV + params) / bandwidth."""
    b = _finite(batch, "batch", 0.0, MAX_DIM)
    return ((b * _finite(kv_bytes_per_seq, "kv_bytes_per_seq", 0.0, MAX_BYTES)
             + _finite(param_bytes, "param_bytes", 0.0, MAX_BYTES))
            / _finite(bandwidth_bytes_s, "bandwidth_bytes_s", 1.0, MAX_BYTES_S))


def step_time(batch, kv_bytes_per_seq, n_params, param_bytes, flops_s,
              bandwidth_bytes_s) -> float:
    """
    Part 7's general step time, seconds:
        batch x KV / W  +  max(2 x batch x params / C,  param_bytes / W)
    The first term is attention, always bandwidth-bound; the second is the
    MLP and projections, which go compute-bound past the critical batch.
    """
    b = _finite(batch, "batch", 0.0, MAX_DIM)
    w = _finite(bandwidth_bytes_s, "bandwidth_bytes_s", 1.0, MAX_BYTES_S)
    attn = b * _finite(kv_bytes_per_seq, "kv_bytes_per_seq", 0.0, MAX_BYTES) / w
    mlp = max(2.0 * b * _finite(n_params, "n_params", 0.0, MAX_BYTES)
              / _finite(flops_s, "flops_s", 1.0, MAX_OPS_S),
              _finite(param_bytes, "param_bytes", 0.0, MAX_BYTES) / w)
    return attn + mlp


def attention_decode_intensity(q_per_kv_head, kv_bytes_per_elem: float = 2.0) -> float:
    """
    FLOPs per byte of KV read while decoding one token. Each KV head serves G
    query heads; each does a dot product with every key (2H FLOPs) and a
    weighted sum over every value (2H FLOPs), against 2H x bytes of K and V.
    Intensity = 2G / bytes. For multi-head attention in bf16 (G = 1, 2 bytes)
    that is the book's "~1": attention decode is always bandwidth-bound.
    """
    return 2.0 * _finite(q_per_kv_head, "q_per_kv_head", 1.0, 4096) / _pos(
        kv_bytes_per_elem, "kv_bytes_per_elem", 8)


def validate_against_book() -> Dict[str, Any]:
    """
    Reproduce the book's own worked answers from the formulas above. If any
    of these drifts, nothing downstream in this file is trustworthy.
    """
    rows: List[Tuple[str, float, float, float]] = []

    def row(claim, book, ours, rel_tol):
        rows.append((claim, float(book), float(ours), float(rel_tol)))

    v5e_ci = critical_intensity(TPU_V5E["bf16_flops"], TPU_V5E["hbm_bytes_s"])
    row("TPU v5e bf16 critical intensity (Part 1)", 240, v5e_ci, 0.01)
    row("H100 bf16 critical intensity (Part 1)", 295,
        critical_intensity(H100["bf16_flops"], H100["hbm_bytes_s"]), 0.01)
    row("bf16 weights, bf16 math: B_crit (Part 1)", 240,
        critical_batch(TPU_V5E["bf16_flops"], TPU_V5E["hbm_bytes_s"], 16), 0.01)
    row("int8 weights, bf16 math: B_crit (Part 1)", 120,
        critical_batch(TPU_V5E["bf16_flops"], TPU_V5E["hbm_bytes_s"], 8), 0.01)
    row("int8 weights, int8 math: B_crit (Part 7, 'returns to ~240')", 240,
        critical_batch(TPU_V5E["int8_ops"], TPU_V5E["hbm_bytes_s"], 8), 0.01)
    row("beta x alpha form, int8 params / bf16 activations (Part 7)", 120,
        critical_batch_beta_alpha(8, 16, v5e_ci), 0.01)
    row("matmul intensity ~B when B << D, F (B=8, D=F=8192)", 8,
        matmul_intensity(8, 8192, 8192), 0.01)
    row("LLaMA-2 13B KV at 8192 tokens, bf16 (Part 7), GB", 6.7,
        kv_bytes_per_token(40, 40, 128, 2) * 8192 / 1e9, 0.01)
    wm = dense_params(64, 4096, 16384, 32, 8, 256, 32128, tied=True)
    row("worked-problem model parameters (Part 7), B", 18.4,
        wm["total"] / 1e9, 0.005)
    row("worked-problem KV per token, int8 (Part 7), kB", 262.144,
        kv_bytes_per_token(64, 8, 256, 1) / 1e3, 0.001)
    row("worked-problem KV at 128k tokens, int8 (Part 7), GB", 33.5,
        kv_bytes_per_token(64, 8, 256, 1) * 128_000 / 1e9, 0.005)
    row("min step time, batch 4, 30B int8, 16 chips (Part 7), ms", 2.5,
        1e3 * min_step_time(4, 819e6, 30e9, 16 * TPU_V5E["hbm_bytes_s"]), 0.02)
    row("step time, batch 256, same setup (Part 7), ms", 21,
        1e3 * step_time(256, 819e6, 30e9, 30e9, 16 * TPU_V5E["bf16_flops"],
                        16 * TPU_V5E["hbm_bytes_s"]), 0.01)
    row("MHA KV per token, D=4096, L=64, int8 (Part 4), KiB", 512,
        kv_bytes_per_token(64, 32, 128, 1) / 1024, 0.001)
    row("[2, S, L, K, H] cache, 8k ctx, 64 layers, KH=8192, int8, GiB", 8,
        kv_bytes_per_token(64, 64, 128, 1) * 8192 / 1024 ** 3, 0.001)
    mha = dense_params(64, 4096, 16384, 32, 32, 128, 0)
    row("attention is ~1/4 of parameters when F = 4D (Part 4)", 0.25,
        mha["attention"] / mha["non_embedding"], 0.01)
    row("MHA decode attention intensity in bf16 is ~1 (Part 7)", 1.0,
        attention_decode_intensity(1, 2), 0.001)

    out = []
    for claim, book, ours, tol in rows:
        ok = abs(ours - book) <= tol * abs(book)
        out.append({"claim": claim, "book": book, "ours": round(ours, 4),
                    "rel_err": round(abs(ours - book) / abs(book), 5), "ok": ok})
    n_ok = sum(r["ok"] for r in out)
    return {"rows": out, "n": len(out), "n_ok": n_ok,
            "verdict": "FAITHFUL" if n_ok == len(out) else "DRIFTED"}


# ============================================================================ #
# SECTION 2 -- hardware and models
# ============================================================================ #

@dataclass(frozen=True)
class Soc:
    name: str
    int8_tops: float          # NPU INT8 peak, the only figure Qualcomm rates
    bw_gbs: float             # LPDDR5X peak
    bw_alt_gbs: Optional[float]
    aihub_device: Optional[str]
    engine_key: str           # key in snapdragon_engine.SOC_DB
    note: str = ""


# One source of truth: these must equal snapdragon_engine.SOC_DB, and the
# self-test asserts it whenever that module imports.
SOCS: Dict[str, Soc] = {
    "X Elite": Soc("Snapdragon X Elite", 45.0, 135.0, None,
                   "Snapdragon X Elite CRD", "snapdragon-x-elite"),
    "X Plus 8-Core": Soc("Snapdragon X Plus 8-Core", 45.0, 135.0, None,
                         "Snapdragon X Plus 8-Core CRD", "snapdragon-x-plus",
                         "same memory system and NPU as X Elite; fewer CPU cores"),
    "X2 Elite": Soc("Snapdragon X2 Elite", 80.0, 152.0, 228.0,
                    "Snapdragon X2 Elite CRD", "snapdragon-x2-elite",
                    "the CRD's SKU is not published: 152 GB/s, or 228 GB/s if "
                    "it is the Extreme part. Efficiencies are quoted against "
                    "152, so they are upper bounds."),
    "X2 Plus": Soc("Snapdragon X2 Plus", 80.0, 152.0, None, None,
                   "snapdragon-x2-plus", "no AI Hub device exists"),
}


@dataclass(frozen=True)
class Arch:
    key: str
    L: int
    D: int
    F: int
    N: int
    K: int
    H: int
    V: int
    tied: bool
    published_total_b: Optional[float] = None
    published_nonemb_b: Optional[float] = None
    attn_layers: Optional[int] = None     # None = every layer is full attention
    dense: bool = True
    weight_params_override_b: Optional[float] = None
    source: str = ""

    @property
    def kv_layers(self) -> int:
        return self.attn_layers if self.attn_layers is not None else self.L

    @property
    def group(self) -> float:
        return self.N / self.K


# Geometry cross-checked three ways: Qualcomm's own qai_hub_models constants
# (NUM_LAYERS, HIDDEN_SIZE, NUM_ATTN_HEADS, NUM_KEY_VALUE_HEADS, HEAD_DIM),
# the geometry table in snapdragon_engine.py, and the model cards' published
# parameter counts, which verify_geometry() must reproduce. FFN widths are not
# in Qualcomm's constants; the parameter-count check is what pins them.
ARCHS: Dict[str, Arch] = {a.key: a for a in (
    Arch("qwen3_0_6b", 28, 1024, 3072, 16, 8, 128, 151936, True, 0.6, 0.44,
         source="Qwen3 model card: 0.6B total, 0.44B non-embedding"),
    Arch("qwen3_1_7b", 28, 2048, 6144, 16, 8, 128, 151936, True, 1.7, 1.4,
         source="Qwen3 model card: 1.7B total, 1.4B non-embedding"),
    Arch("qwen3_4b", 36, 2560, 9728, 32, 8, 128, 151936, True, 4.0, 3.6,
         source="Qwen3 model card: 4.0B total, 3.6B non-embedding"),
    Arch("qwen3_8b", 36, 4096, 12288, 32, 8, 128, 151936, False, 8.2, 6.95,
         source="Qwen3 model card: 8.2B total, 6.95B non-embedding"),
    Arch("llama_v3_2_1b_instruct", 16, 2048, 8192, 32, 8, 64, 128256, True,
         1.23, None, source="Llama 3.2 model card: 1.23B"),
    Arch("llama_v3_2_3b_instruct", 28, 3072, 8192, 24, 8, 128, 128256, True,
         3.21, None, source="Llama 3.2 model card: 3.21B"),
    Arch("llama_v3_1_8b_instruct", 32, 4096, 14336, 32, 8, 128, 128256, False,
         8.03, None, source="Llama 3.1 model card: 8.03B"),
    Arch("phi_4_mini_instruct", 32, 3072, 8192, 24, 8, 128, 200064, True,
         3.8, None, source="Phi-4-mini model card: 3.8B"),
    # Hybrid: 48 of 64 layers are linear attention with a constant-size
    # state, so only 16 carry KV. The dense formula does not describe it;
    # its weight bytes come from the measured GGUF files instead.
    Arch("bonsai_2_27b", 64, 5120, 17408, 24, 4, 256, 0, True, 26.90, None,
         attn_layers=16, dense=False, weight_params_override_b=26.904140464,
         source="config.json via snapdragon_engine geometry; 16 of 64 layers "
                "full attention; params = 26,904,140,464"),
)}

# Bits per weight as stored, for the formats in Qualcomm's tables.
# q4_0 is llama.cpp's 32-weight block: 16 bytes of nibbles + a 2-byte scale.
WEIGHT_BITS: Dict[str, float] = {"w4a16": 4.0, "w4": 4.0, "q4_0": 4.5,
                                 "w8a16": 8.0, "w8a8": 8.0, "fp16": 16.0}

# How the LM head and the KV cache are stored is NOT published per row, so
# every fit is repeated under all six combinations and a conclusion counts
# only if it survives all of them.
ASSUMPTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (lm, kv) for lm in (4, 8, 16) for kv in (8, 16))


def _arch(a) -> Arch:
    if isinstance(a, str):
        return ARCHS[a]
    if not isinstance(a, Arch):
        raise TypeError(f"expected an Arch or a key, got {type(a).__name__}")
    return a


def nonembedding_params(a: Arch) -> float:
    a = _arch(a)
    if not a.dense:
        raise ValueError(f"{a.key} is not a dense transformer; the dense "
                         f"formula does not apply")
    return dense_params(a.L, a.D, a.F, a.N, a.K, a.H, 0)["non_embedding"]


def lm_head_params(a: Arch) -> float:
    """The LM head is read in full for every token. The embedding lookup
    reads one row and is negligible."""
    a = _arch(a)
    return float(a.V * a.D)


def total_params(a: Arch) -> float:
    a = _arch(a)
    if not a.dense:
        return float(a.weight_params_override_b) * 1e9
    return nonembedding_params(a) + lm_head_params(a) * (1 if a.tied else 2)


def verify_geometry(tol: float = 0.02) -> Dict[str, Any]:
    """Every dense architecture must reproduce its model card's published
    parameter counts. This is what pins the FFN widths."""
    rows = []
    for a in ARCHS.values():
        if not a.dense:
            continue
        tot = total_params(a) / 1e9
        r = {"arch": a.key, "total_b": round(tot, 3),
             "published_total_b": a.published_total_b}
        ok = a.published_total_b is not None and \
            abs(tot - a.published_total_b) <= tol * a.published_total_b
        if a.published_nonemb_b is not None:
            ne = nonembedding_params(a) / 1e9
            r["nonemb_b"] = round(ne, 3)
            r["published_nonemb_b"] = a.published_nonemb_b
            ok = ok and abs(ne - a.published_nonemb_b) <= tol * a.published_nonemb_b
        r["ok"] = ok
        rows.append(r)
    return {"rows": rows, "all_ok": all(r["ok"] for r in rows)}


def bytes_per_token(arch, weight_format: str, context,
                    lm_head_bits: float = 8.0, kv_bits: float = 16.0) -> float:
    """
    Bytes one decode step must move: every non-embedding weight, the LM head,
    and `context` tokens of K and V. The embedding lookup and activations are
    omitted -- they are a row and a vector.
    """
    a = ARCHS[arch] if isinstance(arch, str) else arch
    if not isinstance(a, Arch):
        raise TypeError(f"arch must be a key or an Arch, got {type(arch).__name__}")
    if weight_format not in WEIGHT_BITS:
        raise ValueError(f"unknown weight format {weight_format!r}; "
                         f"known: {sorted(WEIGHT_BITS)}")
    c = _finite(context, "context", 0.0, 1e8)
    lm = _pos(lm_head_bits, "lm_head_bits", 32)
    kvb = _pos(kv_bits, "kv_bits", 32)
    if not a.dense:
        raise ValueError(f"{a.key}: use its measured file bytes, not the "
                         f"dense byte model")
    w = nonembedding_params(a) * WEIGHT_BITS[weight_format] / 8.0
    w += lm_head_params(a) * lm / 8.0
    return w + c * kv_bytes_per_token(a.kv_layers, a.K, a.H, kvb / 8.0)


# ============================================================================ #
# SECTION 3 -- Qualcomm's measurements, embedded
# ============================================================================ #
#
# Extracted verbatim from qai_hub_models 0.62.2, models/<id>/perf.yaml, for the
# eight models whose geometry verify_geometry() pins. Columns:
#   (model, precision, device, runtime, compute unit, context,
#    decode tok/s, prefill tok/s)
# Prefill is measured over a prompt of exactly `context` tokens: it equals
# context / max time-to-first-token in the same file.

SNAPSHOT_SOURCE = ("qai_hub_models 0.62.2, qai_hub_models/models/<id>/perf.yaml "
                   "-- ships inside the pip package; nothing downloaded")

QUALCOMM_MEASURED: Tuple[Tuple[Any, ...], ...] = (
    ('llama_v3_1_8b_instruct', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 5.0161, 250.69),
    ('llama_v3_1_8b_instruct', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 10.7236, 554.24),
    ('llama_v3_1_8b_instruct', 'w4a16', 'X2 Elite', 'genie', 'npu', 4096, 21.9983, 389.41),
    ('llama_v3_1_8b_instruct', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 22.4245, 1131.87),
    ('llama_v3_2_1b_instruct', 'w4', 'X Elite', 'genie', 'npu', 4096, 8.2295, 426.08),
    ('llama_v3_2_1b_instruct', 'w4', 'X Elite', 'geniex_qairt', 'npu', 4096, 19.5217, 977.2),
    ('llama_v3_2_1b_instruct', 'w4', 'X2 Elite', 'genie', 'npu', 4096, 36.402, 747.94),
    ('llama_v3_2_1b_instruct', 'w4', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 36.6658, 2226.2),
    ('llama_v3_2_1b_instruct', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 16.8973, 834.03),
    ('llama_v3_2_1b_instruct', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 43.4773, 2095.75),
    ('llama_v3_2_1b_instruct', 'w4a16', 'X2 Elite', 'genie', 'npu', 4096, 80.9922, None),
    ('llama_v3_2_1b_instruct', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 90.5535, 4670.85),
    ('llama_v3_2_3b_instruct', 'w4', 'X Elite', 'genie', 'npu', 4096, 7.1577, 223.91),
    ('llama_v3_2_3b_instruct', 'w4', 'X Elite', 'geniex_qairt', 'npu', 4096, 8.7944, 466.89),
    ('llama_v3_2_3b_instruct', 'w4', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 16.4093, 775.38),
    ('llama_v3_2_3b_instruct', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 11.3226, 469.04),
    ('llama_v3_2_3b_instruct', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 19.8249, 988.91),
    ('llama_v3_2_3b_instruct', 'w4a16', 'X2 Elite', 'genie', 'npu', 4096, 42.4538, 656.6),
    ('llama_v3_2_3b_instruct', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 42.7519, 2068.65),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 512, 27.9884, 313.6),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 4096, 15.6171, 140.08),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 512, 27.1436, 277.56),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 4096, 16.0004, 133.47),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 512, 16.0424, 508.52),
    ('phi_4_mini_instruct', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 4096, 9.6733, 310.89),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 512, 34.224, 504.12),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 4096, 22.3581, 290.72),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 512, 28.0395, 632.95),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 4096, 21.0804, 349.76),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 512, 26.8782, 1660.22),
    ('phi_4_mini_instruct', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 4096, 18.1231, 1276.47),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 512, 117.8273, 1095.66),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 4096, 27.7167, 398.43),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 512, 59.0193, 1158.56),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 4096, 28.1124, 307.54),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 512, 43.4919, 1482.97),
    ('qwen3_0_6b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 4096, 14.7875, 1120.53),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 512, 137.0152, 2342.75),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 4096, 58.5932, 716.41),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 512, 139.6974, 2290.2),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 4096, 58.5356, 715.43),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 512, 53.3956, 3066.77),
    ('qwen3_0_6b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 4096, 27.687, 1676.52),
    ('qwen3_0_6b', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 23.8989, 815.0),
    ('qwen3_0_6b', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 89.9137, 4279.79),
    ('qwen3_0_6b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 512, 112.1466, 7917.85),
    ('qwen3_0_6b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 1024, 110.2392, 6445.76),
    ('qwen3_0_6b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 63.1877, 4288.57),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 512, 59.8382, 584.48),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 4096, 22.8101, 228.71),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 512, 34.4913, 590.16),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 4096, 19.7874, 232.75),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 512, 26.9182, 967.72),
    ('qwen3_1_7b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 4096, 11.8856, 829.11),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 512, 64.6993, 1096.11),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 4096, 36.0233, 537.38),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 512, 64.8994, 1120.35),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 4096, 36.2893, 537.28),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 512, 38.4193, 1983.2),
    ('qwen3_1_7b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 4096, 22.904, 1262.75),
    ('qwen3_1_7b', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 6.0473, 671.59),
    ('qwen3_1_7b', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 41.5319, 2547.67),
    ('qwen3_1_7b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 512, 68.4253, 4297.61),
    ('qwen3_1_7b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 1024, 65.9004, 3442.51),
    ('qwen3_1_7b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 2048, 58.6562, 3192.51),
    ('qwen3_1_7b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 3072, 53.2432, 2989.78),
    ('qwen3_1_7b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 47.6967, 2848.21),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 512, 26.2085, 120.57),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'cpu', 4096, 12.7674, 108.32),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 512, 21.6003, 238.01),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'gpu', 4096, 12.8736, 92.83),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 512, 11.4057, 490.44),
    ('qwen3_4b', 'q4_0', 'X Elite', 'geniex_llamacpp', 'npu', 4096, 5.4625, 374.88),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 512, 33.7415, 460.69),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'cpu', 4096, 20.3235, 224.51),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 512, 33.4905, 463.98),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'gpu', 4096, 20.2623, 223.92),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 512, 20.9731, 786.72),
    ('qwen3_4b', 'q4_0', 'X2 Elite', 'geniex_llamacpp', 'npu', 4096, 14.3865, 536.41),
    ('qwen3_4b', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 6.5649, 340.48),
    ('qwen3_4b', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 21.2203, 1292.52),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'genie', 'npu', 4096, 30.2685, None),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 512, 36.2322, 2306.69),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 1024, 35.7029, 1939.42),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 2048, 33.1502, 1904.94),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 3072, 30.5389, 1742.55),
    ('qwen3_4b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 26.9754, 1598.12),
    ('qwen3_8b', 'w4a16', 'X Elite', 'genie', 'npu', 4096, 4.1119, 218.72),
    ('qwen3_8b', 'w4a16', 'X Elite', 'geniex_qairt', 'npu', 4096, 13.5861, 818.82),
    ('qwen3_8b', 'w4a16', 'X2 Elite', 'geniex_qairt', 'npu', 4096, 25.0199, 1895.68),
)

# Device counts in the same package, for the correction recorded above.
PACKAGE_DEVICE_COVERAGE = {
    "Snapdragon X Elite CRD": {"listed_as_supported": 221, "measured_entries": 491},
    "Snapdragon X2 Elite CRD": {"listed_as_supported": 219, "measured_entries": 487},
    "Snapdragon X Plus 8-Core CRD": {"listed_as_supported": 221, "measured_entries": 0},
}

_DEVICE_SHORT = {"Snapdragon X Elite CRD": "X Elite",
                 "Snapdragon X2 Elite CRD": "X2 Elite",
                 "Snapdragon X Plus 8-Core CRD": "X Plus 8-Core"}


def _rows(model=None, precision=None, device=None, runtime=None, unit=None,
          context=None) -> List[Tuple[Any, ...]]:
    out = []
    for r in QUALCOMM_MEASURED:
        if model is not None and r[0] != model:
            continue
        if precision is not None and r[1] != precision:
            continue
        if device is not None and r[2] != device:
            continue
        if runtime is not None and r[3] != runtime:
            continue
        if unit is not None and r[4] != unit:
            continue
        if context is not None and r[5] != context:
            continue
        out.append(r)
    return out


def load_measured_from_package() -> Optional[List[Tuple[Any, ...]]]:
    """
    Re-read the same rows from an installed qai_hub_models, if there is one.
    Returns None when the package (or PyYAML) is absent -- that is normal on a
    laptop, and the embedded snapshot is used instead.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("qai_hub_models")
        if spec is None or not spec.submodule_search_locations:
            return None
        import yaml  # noqa: F401 -- only needed when the package is present
    except Exception:
        return None
    root = pathlib.Path(list(spec.submodule_search_locations)[0]) / "models"
    out: List[Tuple[Any, ...]] = []
    for key in ARCHS:
        p = root / key / "perf.yaml"
        if not p.exists():
            continue
        try:
            import yaml
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for prec, pd in (d.get("precisions") or {}).items():
            for _comp, cd in ((pd or {}).get("components") or {}).items():
                for dev, rd in ((cd or {}).get("performance_metrics") or {}).items():
                    if dev not in _DEVICE_SHORT or dev == "Snapdragon X Plus 8-Core CRD":
                        continue
                    for rt, rv in (rd or {}).items():
                        if not isinstance(rv, dict):
                            continue
                        for m in rv.get("llm_metrics") or []:
                            tps = m.get("tokens_per_second")
                            if not tps:
                                continue
                            pre = m.get("prefill_tokens_per_second")
                            out.append((key, prec, _DEVICE_SHORT[dev], rt,
                                        m.get("desired_compute_unit"),
                                        int(m.get("context_length") or 0),
                                        round(float(tps), 4),
                                        None if pre is None else round(float(pre), 2)))
    return sorted(out)


def snapshot_matches_package() -> Dict[str, Any]:
    """Drift check: if Qualcomm updates perf.yaml, say so rather than keep
    quoting a stale snapshot."""
    live = load_measured_from_package()
    if live is None:
        return {"status": "SKIPPED", "reason": "qai_hub_models not installed"}
    snap = sorted(QUALCOMM_MEASURED)
    if live == snap:
        return {"status": "MATCH", "rows": len(snap)}
    only_live = sorted(set(live) - set(snap))
    only_snap = sorted(set(snap) - set(live))
    return {"status": "DRIFTED", "new_or_changed": only_live[:10],
            "missing_or_changed": only_snap[:10],
            "advice": "Qualcomm's numbers changed: re-extract the snapshot"}


# ============================================================================ #
# SECTION 4 -- the tests on the measurements
# ============================================================================ #

def fit_line(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Ordinary least squares y = a + b x, with R^2. Needs >= 3 points and
    distinct x -- two points always fit perfectly and prove nothing."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys differ in length")
    if len(xs) < 3:
        raise ValueError(f"need at least 3 points, got {len(xs)}")
    x = [_finite(v, "x") for v in xs]
    y = [_finite(v, "y") for v in ys]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        raise ValueError("x values are all equal; slope undefined")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    sst = sum((yi - my) ** 2 for yi in y)
    ssr = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - ssr / sst if sst > 0 else 1.0
    return {"a": a, "b": b, "r2": r2, "n": n}


def _family(model: str) -> str:
    return model.split("_")[0]


def bandwidth_law(device: str, runtime: str, precision: str, unit: str,
                  context: int, family: Optional[str] = "qwen3",
                  exclude: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Decode milliseconds per token against bytes moved per token, one line per
    byte-accounting assumption. Bandwidth-bound means: a straight line, an
    intercept near zero, and a slope whose inverse is a plausible fraction of
    peak bandwidth. Restricting to one family keeps the export recipe fixed.
    """
    if device not in ("X Elite", "X2 Elite"):
        raise ValueError(f"no measurements for device {device!r}")
    rows = [r for r in _rows(precision=precision, device=device, runtime=runtime,
                             unit=unit, context=context)
            if (family is None or _family(r[0]) == family) and r[0] not in exclude
            and ARCHS[r[0]].dense]
    if len(rows) < 3:
        raise ValueError(f"only {len(rows)} usable rows for "
                         f"{device}/{runtime}/{precision}/{unit}/{context}")
    peak = SOCS[device].bw_gbs
    fits = []
    for lm, kv in ASSUMPTIONS:
        xs = [bytes_per_token(r[0], precision, context, lm, kv) / 1e9 for r in rows]
        ys = [1000.0 / r[6] for r in rows]
        f = fit_line(xs, ys)
        bw = 1000.0 / f["b"] if f["b"] > 0 else float("nan")
        fits.append({"lm_head_bits": lm, "kv_bits": kv,
                     "intercept_ms": round(f["a"], 2), "bw_gbs": round(bw, 1),
                     "pct_of_peak": round(100 * bw / peak, 1), "r2": round(f["r2"], 4)})
    r2s = [f["r2"] for f in fits]
    icp = [f["intercept_ms"] for f in fits]
    bws = [f["bw_gbs"] for f in fits]
    linear = min(r2s) >= 0.95
    bound = min(r2s) >= 0.98 and max(abs(v) for v in icp) <= 6.0
    # Three regimes, not two. A straight line with a large intercept is a
    # bandwidth-set SLOPE plus a fixed per-token cost -- dispatch, sync,
    # sampling -- and calling that "not bandwidth-bound" hides which half a
    # runtime author should attack.
    regime = ("bandwidth-bound" if bound else
              "bandwidth slope + fixed overhead" if linear else
              "not linear in bytes")
    return {"device": device, "runtime": runtime, "precision": precision,
            "unit": unit, "context": context,
            "models": sorted(r[0] for r in rows), "fits": fits,
            "min_r2": min(r2s), "intercept_range_ms": (min(icp), max(icp)),
            "bw_range_gbs": (min(bws), max(bws)),
            "pct_range": (round(100 * min(bws) / peak), round(100 * max(bws) / peak)),
            "bandwidth_bound": bound, "regime": regime}


def anomalies(min_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """
    Pairs of measurements that no bandwidth model can explain: within one
    group (device, runtime, precision, unit, context, family), the MARGINAL
    bandwidth between consecutive models -- extra bytes over extra time -- is
    above the device's peak. With a shared per-token overhead that is
    impossible. It is reported as a property of the published data, not
    explained here.
    """
    out = []
    groups: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
    for r in QUALCOMM_MEASURED:
        if not ARCHS[r[0]].dense:
            continue
        groups.setdefault((r[2], r[3], r[1], r[4], r[5], _family(r[0])), []).append(r)
    for (dev, rt, prec, unit, ctx, fam), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        # the smallest plausible bytes: most generous to the data
        pts = sorted(((bytes_per_token(r[0], prec, ctx, 4, 8), 1.0 / r[6], r[0])
                      for r in rs), key=lambda p: p[0])
        peak = SOCS[dev].bw_alt_gbs or SOCS[dev].bw_gbs
        for (b0, t0, m0), (b1, t1, m1) in zip(pts, pts[1:]):
            dt = t1 - t0
            db = b1 - b0
            marginal = float("inf") if dt <= 0 else db / dt / 1e9
            if marginal > peak * min_ratio:
                out.append({"device": dev, "runtime": rt, "precision": prec,
                            "unit": unit, "context": ctx, "smaller": m0,
                            "larger": m1, "extra_bytes_pct": round(100 * db / b0),
                            "extra_time_pct": round(100 * dt / t0, 1),
                            "marginal_bw_gbs": (round(marginal) if marginal != float("inf")
                                                else None),
                            "peak_gbs": peak})
    return out


def _anomalous_pairs(device: str, runtime: str, precision: str, unit: str,
                     context: int, family: str) -> List[Tuple[str, str]]:
    return [(a["smaller"], a["larger"]) for a in anomalies()
            if (a["device"], a["runtime"], a["precision"], a["unit"], a["context"])
            == (device, runtime, precision, unit, context)
            and _family(a["smaller"]) == family]


def _choose_exclusion(dev, rt, prec, unit, ctx, fam, pairs, min_models):
    """
    An unphysical pair says one of its two members is off, not which. Drop
    each in turn and keep whichever leaves the better fit for everything else
    -- the member the REST of the data disagrees with. Always dropping the
    larger model was the first version of this, and it dropped the wrong one
    in the Genie group, where the 1.7B figure is the outlier.
    """
    excl: List[str] = []
    for m0, m1 in pairs:
        best = None
        for cand in (m0, m1):
            trial = excl + [cand]
            n = len([x for x in _rows(precision=prec, device=dev, runtime=rt,
                                      unit=unit, context=ctx)
                     if _family(x[0]) == fam and x[0] not in trial])
            if n < min_models:
                continue
            law = bandwidth_law(dev, rt, prec, unit, ctx, fam, trial)
            if best is None or law["min_r2"] > best[1]:
                best = (cand, law["min_r2"])
        if best is not None:
            excl.append(best[0])
    return excl


def bandwidth_law_table(min_models: int = 3) -> List[Dict[str, Any]]:
    """Every group with enough same-family models to fit, anomalies removed."""
    seen = set()
    out = []
    for r in QUALCOMM_MEASURED:
        key = (r[2], r[3], r[1], r[4], r[5], _family(r[0]))
        if key in seen:
            continue
        seen.add(key)
        dev, rt, prec, unit, ctx, fam = key
        pairs = _anomalous_pairs(dev, rt, prec, unit, ctx, fam)
        excl = _choose_exclusion(dev, rt, prec, unit, ctx, fam, pairs, min_models)
        n = len([x for x in _rows(precision=prec, device=dev, runtime=rt, unit=unit,
                                  context=ctx)
                 if _family(x[0]) == fam and x[0] not in excl])
        if n < min_models:
            continue
        law = bandwidth_law(dev, rt, prec, unit, ctx, fam, excl)
        law["family"] = fam
        law["excluded"] = excl
        out.append(law)
    return sorted(out, key=lambda d: (d["device"], d["runtime"], d["unit"], d["context"]))


def effective_critical_batch(min_models: int = 3) -> List[Dict[str, Any]]:
    """
    prefill tok/s divided by decode tok/s, per (device, runtime, precision,
    unit, context).

    Why this IS the critical batch: prefill over a long prompt is
    compute-bound, so a token costs 2P / C_eff; decode is bandwidth-bound, so
    a token costs P x bytes / W_eff. Their ratio is C_eff / W_eff x bytes / 2
    -- the book's B_crit, with EFFECTIVE rather than peak figures. It depends
    on the engine and the precision and not on the model, so the test is that
    it stays flat across model sizes. It also says how many draft tokens a
    speculative verifier can check for the price of one decode step.
    """
    groups: Dict[Tuple[Any, ...], List[Tuple[str, float]]] = {}
    for r in QUALCOMM_MEASURED:
        if r[7] and r[6]:
            groups.setdefault((r[2], r[3], r[1], r[4], r[5]), []).append(
                (r[0], r[7] / r[6]))
    out = []
    for (dev, rt, prec, unit, ctx), vals in sorted(groups.items()):
        if len(vals) < min_models:
            continue
        v = [x for _, x in vals]
        mean = statistics.mean(v)
        cv = statistics.pstdev(v) / mean if mean else float("nan")
        out.append({"device": dev, "runtime": rt, "precision": prec, "unit": unit,
                    "context": ctx, "n": len(v), "mean": round(mean, 1),
                    "min": round(min(v), 1), "max": round(max(v), 1),
                    "cv": round(cv, 3), "per_model": {m: round(x, 1) for m, x in vals}})
    return out


def context_slope(min_points: int = 3) -> List[Dict[str, Any]]:
    """
    Decode ms per token against context length, for models measured at three
    or more contexts. The intercept is the context-free cost (weights, LM
    head, overhead); the slope is the cost of one more token of context. Their
    ratio is the measured crossover T* -- the context at which KV traffic
    equals everything else -- and it is compared with the book's formula.
    """
    groups: Dict[Tuple[Any, ...], List[Tuple[int, float]]] = {}
    for r in QUALCOMM_MEASURED:
        if r[5] and r[6] and ARCHS[r[0]].dense:
            groups.setdefault((r[0], r[2], r[3], r[1], r[4]), []).append(
                (r[5], 1000.0 / r[6]))
    out = []
    for (model, dev, rt, prec, unit), pts in sorted(groups.items()):
        if len(pts) < min_points:
            continue
        pts.sort()
        f = fit_line([c for c, _ in pts], [t for _, t in pts])
        a = ARCHS[model]
        weights = (nonembedding_params(a) * WEIGHT_BITS[prec] / 8.0
                   + lm_head_params(a) * 8 / 8.0)
        t_star = f["a"] / f["b"] if f["b"] > 0 else float("inf")
        pred16 = weights / kv_bytes_per_token(a.kv_layers, a.K, a.H, 2)
        pred8 = weights / kv_bytes_per_token(a.kv_layers, a.K, a.H, 1)
        out.append({"model": model, "device": dev, "runtime": rt,
                    "precision": prec, "unit": unit, "points": len(pts),
                    "intercept_ms": round(f["a"], 2),
                    "us_per_context_token": round(f["b"] * 1e3, 3),
                    "r2": round(f["r2"], 3), "t_star_measured": round(t_star),
                    "t_star_formula_kv16": round(pred16),
                    "t_star_formula_kv8": round(pred8),
                    "measured_over_formula_kv16": round(t_star / pred16, 2),
                    "measured_over_formula_kv8": round(t_star / pred8, 2)})
    return out


def engine_asymmetry(device: str = "X2 Elite", context: int = 512) -> List[Dict[str, Any]]:
    """
    For each model with CPU, GPU and NPU measurements on one runtime: the
    spread of decode rates versus the spread of prefill rates. The book
    predicts decode spread ~1 (all engines share one DRAM bus) and prefill
    spread >> 1 (compute differs). Also lists the best NPU runtime, because a
    weak NPU backend is a software fact, not a silicon one.
    """
    out = []
    models = sorted({r[0] for r in _rows(device=device, context=context)})
    for m in models:
        rs = _rows(model=m, device=device, runtime="geniex_llamacpp", context=context)
        by_unit = {r[4]: r for r in rs}
        if not {"cpu", "gpu", "npu"} <= set(by_unit):
            continue
        dec = {u: by_unit[u][6] for u in ("cpu", "gpu", "npu")}
        pre = {u: by_unit[u][7] for u in ("cpu", "gpu", "npu")}
        qairt = _rows(model=m, device=device, runtime="geniex_qairt", context=context)
        best_npu_decode = max([dec["npu"]] + [r[6] for r in qairt])
        best_npu_prefill = max([pre["npu"]] + [r[7] for r in qairt if r[7]])
        cpu_gpu = max(dec["cpu"], dec["gpu"]) / min(dec["cpu"], dec["gpu"])
        best_dec = [dec["cpu"], dec["gpu"], best_npu_decode]
        best_pre = [pre["cpu"], pre["gpu"], best_npu_prefill]
        out.append({"model": m, "device": device, "context": context,
                    "decode": {k: round(v, 2) for k, v in dec.items()},
                    "prefill": {k: round(v, 1) for k, v in pre.items()},
                    "qairt_npu_decode": round(qairt[0][6], 2) if qairt else None,
                    "qairt_npu_prefill": round(qairt[0][7], 1) if qairt and qairt[0][7] else None,
                    "cpu_vs_gpu_decode_spread": round(cpu_gpu, 3),
                    "best_engine_decode_spread": round(max(best_dec) / min(best_dec), 2),
                    "best_engine_prefill_spread": round(max(best_pre) / min(best_pre), 2)})
    return out


def runtime_spread() -> List[Dict[str, Any]]:
    """Same device, model, precision and context; Genie versus GenieX QAIRT.
    Identical silicon and identical weights, so the ratio is pure software."""
    out = []
    for r in QUALCOMM_MEASURED:
        if r[3] != "genie":
            continue
        q = _rows(model=r[0], precision=r[1], device=r[2], runtime="geniex_qairt",
                  unit=r[4], context=r[5])
        if q:
            out.append({"model": r[0], "precision": r[1], "device": r[2],
                        "context": r[5], "genie": round(r[6], 2),
                        "geniex_qairt": round(q[0][6], 2),
                        "ratio": round(q[0][6] / r[6], 2)})
    return out


# ============================================================================ #
# SECTION 5 -- applied to Snapdragon and the HP machines
# ============================================================================ #

def critical_batch_table() -> List[Dict[str, Any]]:
    """
    Spec-sheet B_crit per SoC and weight format, using the INT8 TOPS rating.
    That rating is for INT8 x INT8; w4a16 and fp16 MACs run slower, which only
    LOWERS B_crit -- so these are upper bounds, and the conclusion that batch-1
    decode is bandwidth-bound holds a fortiori.
    """
    out = []
    for key, s in SOCS.items():
        for fmt, bits in (("fp16", 16.0), ("int8", 8.0), ("int4", 4.0),
                          ("PTQ1_0 ternary", 1.768), ("Q1_0 1-bit", 1.131)):
            out.append({"soc": key, "format": fmt, "weight_bits": bits,
                        "b_crit": round(critical_batch(s.int8_tops * 1e12,
                                                       s.bw_gbs * 1e9, bits), 1)})
    return out


def decode_ceiling(arch, weight_format: str, context, bw_gbs,
                   lm_head_bits: float = 8.0, kv_bits: float = 16.0) -> float:
    """Tokens per second no implementation can beat at this bandwidth."""
    return _pos(bw_gbs, "bw_gbs", 1e6) * 1e9 / bytes_per_token(
        arch, weight_format, context, lm_head_bits, kv_bits)


def kv_crossover(arch, weight_format: str, kv_bits: float = 16.0,
                 lm_head_bits: float = 8.0) -> float:
    """
    Context length at which one step's KV traffic equals its weight traffic.
    Below it, weight compression (the container rule) is the lever; above it,
    KV compression is.
    """
    a = ARCHS[arch] if isinstance(arch, str) else arch
    if not isinstance(a, Arch):
        raise TypeError(f"arch must be a key or an Arch, got {type(arch).__name__}")
    kvpt = kv_bytes_per_token(a.kv_layers, a.K, a.H, _pos(kv_bits, "kv_bits", 32) / 8.0)
    if not a.dense:
        raise ValueError(f"{a.key}: pass measured file bytes to "
                         f"kv_crossover_from_bytes()")
    w = bytes_per_token(a, weight_format, 0, lm_head_bits, kv_bits)
    return w / kvpt


def kv_crossover_from_bytes(weight_bytes, arch, kv_bits: float = 16.0) -> float:
    """The same crossover, for a model whose weight bytes are measured (a
    GGUF file) rather than computed -- the hybrid Bonsai 2 case."""
    a = ARCHS[arch] if isinstance(arch, str) else arch
    if not isinstance(a, Arch):
        raise TypeError(f"arch must be a key or an Arch, got {type(arch).__name__}")
    return _pos(weight_bytes, "weight_bytes", MAX_BYTES) / kv_bytes_per_token(
        a.kv_layers, a.K, a.H, _pos(kv_bits, "kv_bits", 32) / 8.0)


def speculation(accept_rate, k: int, draft_cost_ratio, b_crit_eff) -> Dict[str, Any]:
    """
    Expected speedup of speculative decoding with k draft tokens.

    Tokens gained per verify step with i.i.d. acceptance a:
        E = (1 - a^(k+1)) / (1 - a)
    Cost per step, in units of one target decode step: the verify pass costs
    ~1 while k + 1 <= B_crit_eff (the book's point -- below the critical batch
    the extra FLOPs are free), growing as (k + 1) / B_crit_eff past it; plus k
    draft steps at `draft_cost_ratio` each (draft bytes / target bytes, since
    drafting is bandwidth-bound too).

    Independent acceptance is optimistic -- rejections cluster -- so the
    speedup is an upper bound. The ordering it produces (which engine's ridge
    binds first, and at what k) does not depend on that assumption.
    """
    a = _finite(accept_rate, "accept_rate", 0.0, 0.999)
    kk = _int(k, "k", 0, 4096)
    r = _finite(draft_cost_ratio, "draft_cost_ratio", 0.0, 10.0)
    bc = _pos(b_crit_eff, "b_crit_eff")
    tokens = (1 - a ** (kk + 1)) / (1 - a)
    verify = max(1.0, (kk + 1) / bc)
    cost = verify + kk * r
    return {"k": kk, "expected_tokens": round(tokens, 3),
            "verify_cost": round(verify, 3), "cost": round(cost, 3),
            "speedup": round(tokens / cost, 3), "verify_is_free": kk + 1 <= bc}


def best_speculation(accept_rate, draft_cost_ratio, b_crit_eff,
                     k_max: int = 64) -> Dict[str, Any]:
    best = None
    for k in range(0, _int(k_max, "k_max", 0, 4096) + 1):
        s = speculation(accept_rate, k, draft_cost_ratio, b_crit_eff)
        if best is None or s["speedup"] > best["speedup"] + 1e-12:
            best = s
    return best


def predict_x_plus() -> Dict[str, Any]:
    """
    Falsifiable predictions for Snapdragon X Plus 8-Core CRD, the AI Hub proxy
    for four of the seven HP Snapdragon machines and the one CRD Qualcomm has
    measured nothing on.

    X Plus 8-Core shares X Elite's memory system (LPDDR5X, 135 GB/s) and its
    45-TOPS Hexagon NPU; it has 8 CPU cores to X Elite's 12. The roofline
    therefore predicts:
      * NPU decode and NPU prefill: equal to X Elite's, within run-to-run noise.
      * CPU decode: equal to X Elite's -- the bus binds, not the cores.
      * CPU prefill: lower, at most by the core ratio (8/12).
    Any X Plus 8-Core measurement that breaks these falsifies the claim that
    this platform's LLM performance is set by bandwidth.
    """
    preds = []
    for r in _rows(device="X Elite"):
        m, prec, _, rt, unit, ctx, dec, pre = r
        if unit == "cpu":
            p_pre = None if pre is None else (round(pre * 8 / 12, 1), round(pre, 1))
        else:
            p_pre = None if pre is None else (round(pre, 1), round(pre, 1))
        preds.append({"model": m, "precision": prec, "runtime": rt, "unit": unit,
                      "context": ctx, "decode_tok_s": round(dec, 2),
                      "prefill_tok_s_range": p_pre})
    return {"device": "Snapdragon X Plus 8-Core CRD",
            "hp_machines": ["HP OmniBook 3 14-HZ000", "HP OmniBook 5 16-bf000",
                            "HP ProBook 4 G1q 14", "HP EliteBook Ultra G1q8 14"],
            "published_measurements": 0, "predictions": preds,
            "test_with": "python aihub_workbench.py run --device "
                         "\"Snapdragon X Plus 8-Core CRD\" --yes"}


def container_vs_context(contexts=(0, 8192, 32768, 131072, 262144),
                         kv_bits: float = 16.0) -> Dict[str, Any]:
    """
    The container rule, corrected for context. The MLX pack of Bonsai-27B is
    35% larger than the GGUF Q1_0 file, so at zero context MLX moves 35% more
    bytes per token. KV traffic is the same in both and dilutes the gap as
    context grows. Bonsai carries KV on 16 of 64 layers only, which is why the
    dilution is slow.
    """
    q1 = 3_803_452_480.0
    mlx = 5_129_115_752.0
    a = ARCHS["bonsai_2_27b"]
    kvpt = kv_bytes_per_token(a.kv_layers, a.K, a.H, _pos(kv_bits, "kv_bits", 32) / 8.0)
    rows = []
    for c in contexts:
        cc = _finite(c, "context", 0.0, 1e8)
        extra = (mlx + cc * kvpt) / (q1 + cc * kvpt) - 1.0
        rows.append({"context": int(cc), "kv_gb": round(cc * kvpt / 1e9, 2),
                     "mlx_extra_bytes_pct": round(100 * extra, 1)})
    return {"kv_bytes_per_token": kvpt, "rows": rows,
            "t_star_q1_0": round(kv_crossover_from_bytes(q1, a, kv_bits))}


# ============================================================================ #
# SECTION 6 -- self-test
# ============================================================================ #

def selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), str(detail)))

    def raises(fn, exc=Exception):
        try:
            fn()
            return False
        except exc:
            return True

    # ---- the book, reproduced before anything else is trusted ----
    v = validate_against_book()
    ck("every worked answer in the book is reproduced", v["verdict"] == "FAITHFUL",
       [r["claim"] for r in v["rows"] if not r["ok"]])
    ck("at least 15 of the book's numbers are checked", v["n"] >= 15, v["n"])
    for want in ("240", "295", "120", "6.7", "18.4", "262", "2.5", "21"):
        ck(f"the book's {want} is among the reproduced answers",
           any(abs(r["book"] - float(want.replace(" ", ""))) < 1e-9 or
               (want == "262" and abs(r["book"] - 262.144) < 1e-9)
               for r in v["rows"]))
    ck("both B_crit forms agree on int8 params / bf16 math",
       abs(critical_batch(1.97e14, 8.2e11, 8)
           - critical_batch_beta_alpha(8, 16, 1.97e14 / 8.2e11)) < 1e-9)
    ck("halving weight bits halves B_crit",
       abs(critical_batch(1e12, 1e9, 4) * 2 - critical_batch(1e12, 1e9, 8)) < 1e-9)
    ck("matmul intensity approaches B as D, F grow",
       abs(matmul_intensity(4, 1e6, 1e6) - 4) < 1e-3)
    ck("GQA raises decode attention intensity by the group size",
       abs(attention_decode_intensity(4, 2) - 4 * attention_decode_intensity(1, 2)) < 1e-12)
    ck("the MLP term turns compute-bound past B_crit",
       step_time(1000, 0, 1e9, 2e9, 1e12, 1e11) > step_time(1, 0, 1e9, 2e9, 1e12, 1e11))

    # ---- geometry pinned by published counts ----
    g = verify_geometry()
    ck("every dense architecture reproduces its model card", g["all_ok"],
       [r["arch"] for r in g["rows"] if not r["ok"]])
    ck("Qwen3-4B non-embedding is 3.63B", abs(nonembedding_params(ARCHS["qwen3_4b"])
                                              / 1e9 - 3.633) < 0.005)
    ck("Qwen3-4B KV is 147,456 bytes per token in fp16",
       kv_bytes_per_token(36, 8, 128, 2) == 147_456)
    ck("Bonsai 2 carries KV on 16 of 64 layers", ARCHS["bonsai_2_27b"].kv_layers == 16)
    ck("the dense byte model refuses the hybrid",
       raises(lambda: bytes_per_token("bonsai_2_27b", "w4a16", 0), ValueError))

    # ---- one source of truth for the hardware ----
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import snapdragon_engine as _E
        same = all(abs(_E.SOC_DB[s.engine_key].mem_bandwidth_gbs - s.bw_gbs) < 1e-9 and
                   abs(_E.SOC_DB[s.engine_key].npu_tops_int8 - s.int8_tops) < 1e-9
                   for s in SOCS.values())
        ck("SoC figures equal snapdragon_engine.SOC_DB", same)
    except Exception as exc:                                  # pragma: no cover
        ck("SoC figures equal snapdragon_engine.SOC_DB", False, repr(exc))

    # ---- the snapshot ----
    ck("snapshot carries 90 measured rows", len(QUALCOMM_MEASURED) == 90,
       len(QUALCOMM_MEASURED))
    ck("every snapshot model has pinned geometry",
       all(r[0] in ARCHS and ARCHS[r[0]].dense for r in QUALCOMM_MEASURED))
    ck("every snapshot row is on X Elite or X2 Elite",
       {r[2] for r in QUALCOMM_MEASURED} == {"X Elite", "X2 Elite"})
    ck("X Plus 8-Core has zero published measurements",
       PACKAGE_DEVICE_COVERAGE["Snapdragon X Plus 8-Core CRD"]["measured_entries"] == 0)
    drift = snapshot_matches_package()
    if drift["status"] == "SKIPPED":
        ck("snapshot matches the installed package", True, "skipped: package absent")
    else:
        ck("snapshot matches the installed package", drift["status"] == "MATCH",
           drift.get("new_or_changed"))

    # ---- the bandwidth law ----
    law = bandwidth_law("X Elite", "geniex_qairt", "w4a16", "npu", 4096)
    ck("X Elite QAIRT: decode is linear in bytes under all six assumptions",
       law["min_r2"] >= 0.99, law["min_r2"])
    ck("X Elite QAIRT: intercept within 5 ms of zero under all six",
       max(abs(x) for x in law["intercept_range_ms"]) <= 5.0, law["intercept_range_ms"])
    ck("X Elite QAIRT: effective bandwidth is 40-55% of peak",
       40 <= law["pct_range"][0] and law["pct_range"][1] <= 55, law["pct_range"])
    ck("X Elite QAIRT fit spans 0.6B to 8B",
       law["models"] == ["qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "qwen3_8b"])
    genie = bandwidth_law("X Elite", "genie", "w4a16", "npu", 4096)
    ck("Genie on the same chip is overhead-bound, not bandwidth-bound",
       not genie["bandwidth_bound"] and genie["intercept_range_ms"][0] > 30,
       genie["intercept_range_ms"])
    table = bandwidth_law_table()
    bb = [t for t in table if t["bandwidth_bound"]]
    ck("several independent groups are bandwidth-bound", len(bb) >= 5, len(bb))
    ck("no fitted group implies bandwidth above peak",
       all(t["bw_range_gbs"][1] <= (SOCS[t["device"]].bw_alt_gbs or SOCS[t["device"]].bw_gbs)
           for t in table),
       [(t["device"], t["runtime"], t["bw_range_gbs"]) for t in table
        if t["bw_range_gbs"][1] > (SOCS[t["device"]].bw_alt_gbs or SOCS[t["device"]].bw_gbs)])

    # ---- the anomalies ----
    an = {(a["device"], a["runtime"], a["smaller"], a["larger"]): a for a in anomalies()}
    ck("exactly two unphysical pairs are flagged", len(an) == 2, list(an))
    x2 = an.get(("X2 Elite", "geniex_qairt", "qwen3_4b", "qwen3_8b"))
    ck("X2 Elite QAIRT: 8B over 4B needs > 228 GB/s even on the Extreme SKU",
       x2 is not None and x2["marginal_bw_gbs"] and x2["marginal_bw_gbs"] > 228,
       x2)
    ge = an.get(("X Elite", "genie", "qwen3_1_7b", "qwen3_4b"))
    ck("X Elite Genie: the 4B decodes FASTER than the 1.7B",
       ge is not None and ge["extra_time_pct"] < 0, ge)
    ck("the second pair sits in a runtime already shown overhead-bound",
       not bandwidth_law("X Elite", "genie", "w4a16", "npu", 4096)["bandwidth_bound"])
    x2t = [t for t in table if (t["device"], t["runtime"], t["context"])
           == ("X2 Elite", "geniex_qairt", 4096) and t["family"] == "qwen3"]
    ck("leave-one-out blames the 4B@4096 on X2 Elite, not the 8B",
       x2t and x2t[0]["excluded"] == ["qwen3_4b"], x2t and x2t[0]["excluded"])
    ck("without it the rest fit a line at R^2 >= 0.99",
       x2t and x2t[0]["min_r2"] >= 0.99, x2t and x2t[0]["min_r2"])
    # Independent evidence: the same point sits ABOVE its own context trend.
    _q4 = sorted((r[5], 1000.0 / r[6]) for r in _rows(
        model="qwen3_4b", device="X2 Elite", runtime="geniex_qairt"))
    _f = fit_line([c for c, _ in _q4], [t for _, t in _q4])
    _res = {c: t - (_f["a"] + _f["b"] * c) for c, t in _q4}
    ck("and the 4B's own context curve bends up at 4096 (largest residual, positive)",
       _res[4096] > 0 and _res[4096] == max(_res.values()),
       {c: round(v, 2) for c, v in _res.items()})
    get = [t for t in table if (t["device"], t["runtime"]) == ("X Elite", "genie")
           and t["family"] == "qwen3"]
    ck("the Genie outlier removed is the 1.7B, not the larger model",
       get and get[0]["excluded"] == ["qwen3_1_7b"], get and get[0]["excluded"])

    # ---- effective critical batch ----
    ecb = {(e["device"], e["runtime"], e["unit"], e["context"]): e
           for e in effective_critical_batch()}
    x2q = ecb[("X2 Elite", "geniex_qairt", "npu", 512)]
    ck("prefill/decode is flat across model sizes on the X2 Elite NPU (CV <= 0.06)",
       x2q["cv"] <= 0.06, x2q["cv"])
    ck("X2 Elite NPU effective B_crit is 60-72", 60 <= x2q["mean"] <= 72, x2q["mean"])
    x2c = ecb[("X2 Elite", "geniex_llamacpp", "cpu", 512)]
    ck("the CPU on the same bus has a far smaller B_crit (12-18)",
       12 <= x2c["mean"] <= 18 and x2c["cv"] <= 0.10, (x2c["mean"], x2c["cv"]))
    ck("NPU sustains ~4x the CPU's compute on the same bus",
       3.5 <= x2q["mean"] / x2c["mean"] <= 5.0, round(x2q["mean"] / x2c["mean"], 2))
    xeq = ecb[("X Elite", "geniex_qairt", "npu", 4096)]
    ck("X Elite NPU effective B_crit is flat across 7 models (CV <= 0.12)",
       xeq["n"] == 7 and xeq["cv"] <= 0.12, (xeq["n"], xeq["cv"]))
    spec = critical_batch(45e12, 135e9, 4)
    ck("measured X Elite B_crit sits below the spec-sheet upper bound",
       xeq["mean"] < spec, (xeq["mean"], round(spec, 1)))

    # ---- the context slope ----
    cs = {(c["model"], c["device"]): c for c in context_slope()}
    q4 = cs[("qwen3_4b", "X2 Elite")]
    ck("Qwen3-4B on X2 Elite: KV overtakes weights near 10K tokens",
       9000 <= q4["t_star_measured"] <= 10500, q4["t_star_measured"])
    ck("measured crossovers sit within 0.6-1.2x of the 16-bit-KV formula",
       all(0.6 <= c["measured_over_formula_kv16"] <= 1.2 for c in cs.values()),
       [c["measured_over_formula_kv16"] for c in cs.values()])
    ck("an 8-bit cache would need attention at < 0.6x the weight stream's efficiency",
       all(c["measured_over_formula_kv8"] < 0.6 for c in cs.values()),
       [c["measured_over_formula_kv8"] for c in cs.values()])

    # ---- engines and runtimes ----
    ea = {e["model"]: e for e in engine_asymmetry("X2 Elite", 512)}
    e4 = ea["qwen3_4b"]
    ck("Qwen3-4B, X2 Elite: CPU and GPU decode within 1% of each other",
       e4["cpu_vs_gpu_decode_spread"] <= 1.01, e4["cpu_vs_gpu_decode_spread"])
    ck("best-engine decode spread <= 1.1x while prefill spread >= 4x",
       e4["best_engine_decode_spread"] <= 1.1 and e4["best_engine_prefill_spread"] >= 4,
       (e4["best_engine_decode_spread"], e4["best_engine_prefill_spread"]))
    ck("the same holds for every model measured on all three engines",
       all(e["best_engine_prefill_spread"] > 2 * e["best_engine_decode_spread"]
           for e in ea.values()), {m: (e["best_engine_decode_spread"],
                                       e["best_engine_prefill_spread"])
                                   for m, e in ea.items()})
    ex = {e["model"]: e for e in engine_asymmetry("X Elite", 4096)}
    ck("X Elite at 4096, with QAIRT on the NPU: prefill spread >> decode spread",
       "qwen3_4b" in ex and ex["qwen3_4b"]["best_engine_prefill_spread"]
       > 5 * ex["qwen3_4b"]["best_engine_decode_spread"],
       ex.get("qwen3_4b") and (ex["qwen3_4b"]["best_engine_decode_spread"],
                               ex["qwen3_4b"]["best_engine_prefill_spread"]))
    ck("a slow NPU decode on llama.cpp is software: QAIRT on the same NPU beats the CPU",
       "qwen3_4b" in ex and ex["qwen3_4b"]["qairt_npu_decode"] > ex["qwen3_4b"]["decode"]["cpu"],
       ex.get("qwen3_4b") and (ex["qwen3_4b"]["qairt_npu_decode"], ex["qwen3_4b"]["decode"]))
    regimes = {t["regime"] for t in table}
    ck("the law distinguishes three regimes, not two",
       {"bandwidth-bound", "bandwidth slope + fixed overhead"} <= regimes, regimes)
    rs = {(r["model"], r["device"], r["precision"]): r for r in runtime_spread()}
    ck("software alone is worth 3.2x on Qwen3-4B, X Elite",
       abs(rs[("qwen3_4b", "X Elite", "w4a16")]["ratio"] - 3.23) < 0.05,
       rs[("qwen3_4b", "X Elite", "w4a16")]["ratio"])

    # ---- applications ----
    cb = {(r["soc"], r["format"]): r["b_crit"] for r in critical_batch_table()}
    ck("spec-sheet B_crit, X Elite int4, is 83", abs(cb[("X Elite", "int4")] - 83.3) < 0.1)
    ck("batch-1 decode is >= 18x below every Snapdragon ridge",
       min(cb.values()) >= 18, min(cb.values()))
    ceil = decode_ceiling("qwen3_4b", "w4a16", 4096, 135.0)
    meas = _rows(model="qwen3_4b", device="X Elite", runtime="geniex_qairt")[0][6]
    ck("no measurement beats its own roofline ceiling", meas < ceil,
       (round(meas, 1), round(ceil, 1)))
    ck("every measured row sits below its ceiling at the device's peak",
       all(r[6] < decode_ceiling(r[0], r[1], r[5],
                                 SOCS[r[2]].bw_alt_gbs or SOCS[r[2]].bw_gbs, 4, 8)
           for r in QUALCOMM_MEASURED))
    t_star = kv_crossover("qwen3_4b", "w4a16")
    ck("formula crossover for Qwen3-4B w4a16 with fp16 KV is ~15K",
       14_000 <= t_star <= 16_000, round(t_star))
    cvc = container_vs_context()
    ck("the container rule is 35% at zero context",
       abs(cvc["rows"][0]["mlx_extra_bytes_pct"] - 34.9) < 0.2)
    ck("and decays with context, staying above 10% at 128K",
       cvc["rows"][3]["mlx_extra_bytes_pct"] > 10
       and cvc["rows"][3]["mlx_extra_bytes_pct"] < cvc["rows"][0]["mlx_extra_bytes_pct"])
    ck("hybrid attention pushes Bonsai's crossover past 50K tokens",
       cvc["t_star_q1_0"] > 50_000, cvc["t_star_q1_0"])
    s_npu = best_speculation(0.7, 0.15, 65.7)
    s_cpu = speculation(0.7, 32, 0.15, 15.6)
    ck("verification is free on the NPU for any useful k",
       speculation(0.7, 8, 0.15, 65.7)["verify_is_free"])
    ck("the CPU pays for verification past ~15 tokens", not s_cpu["verify_is_free"])
    ck("speculation helps on the NPU (speedup > 1.4 at a=0.7, r=0.15)",
       s_npu["speedup"] > 1.4, s_npu)
    cheap_npu = best_speculation(0.95, 0.01, 65.7)
    cheap_cpu = best_speculation(0.95, 0.01, 15.6)
    ck("with cheap, accurate drafts the verifier's B_crit binds: NPU beats CPU",
       cheap_npu["speedup"] > 1.2 * cheap_cpu["speedup"],
       (cheap_npu["speedup"], cheap_cpu["speedup"]))
    ck("and the NPU verifies more drafts per step than the CPU's ridge allows",
       cheap_npu["k"] + 1 > 15.6 >= cheap_cpu["k"], (cheap_npu["k"], cheap_cpu["k"]))
    px = predict_x_plus()
    ck("X Plus 8-Core predictions cover every X Elite row",
       len(px["predictions"]) == len(_rows(device="X Elite")))
    ck("the prediction names the four HP machines it stands in for",
       len(px["hp_machines"]) == 4)

    # ---- input hardening ----
    ck("NaN bandwidth is refused", raises(lambda: critical_batch(1e12, float("nan"))))
    ck("zero bandwidth is refused", raises(lambda: critical_intensity(1e12, 0)))
    ck("bool is not a number here", raises(lambda: critical_batch(True, 1e9)))
    ck("unknown weight format is refused",
       raises(lambda: bytes_per_token("qwen3_4b", "q3_k", 0), ValueError))
    ck("unknown arch key is refused", raises(lambda: bytes_per_token("nope", "w4a16", 0)))
    ck("K > N is refused", raises(lambda: dense_params(2, 8, 8, 2, 4, 4)))
    ck("a two-point fit is refused", raises(lambda: fit_line([1, 2], [1, 2])))
    ck("a degenerate fit is refused", raises(lambda: fit_line([1, 1, 1], [1, 2, 3])))
    ck("acceptance rate 1.0 is refused (series diverges)",
       raises(lambda: speculation(1.0, 4, 0.1, 50)))
    ck("unmeasured device is refused by the law",
       raises(lambda: bandwidth_law("X Plus 8-Core", "genie", "w4a16", "npu", 4096)))
    ck("huge k is bounded", raises(lambda: speculation(0.5, 10 ** 12, 0.1, 50)))

    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, det in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"   -> {det}" if det and not ok else ""))
    print(f"\n  {npass}/{len(checks)} passed")
    return 0 if npass == len(checks) else 1


# ============================================================================ #
# SECTION 7 -- CLI
# ============================================================================ #

def _print_law(t: Dict[str, Any]) -> None:
    tag = t["regime"].upper() if t["bandwidth_bound"] else t["regime"]
    print(f"\n  {t['device']:<9} {t['runtime']:<16} {t['precision']:<6} {t['unit']:<4} "
          f"ctx {t['context']:<5} {t.get('family', ''):<6} {len(t['models'])} models   {tag}")
    print(f"     R^2 >= {t['min_r2']:.4f}   intercept {t['intercept_range_ms'][0]:+.1f}"
          f"..{t['intercept_range_ms'][1]:+.1f} ms   bandwidth {t['bw_range_gbs'][0]:.0f}"
          f"-{t['bw_range_gbs'][1]:.0f} GB/s ({t['pct_range'][0]}-{t['pct_range'][1]}% of "
          f"{SOCS[t['device']].bw_gbs:.0f})")
    if t.get("excluded"):
        print(f"     excluded as anomalous: {', '.join(t['excluded'])}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Scaling-book roofline, validated on the book and tested on "
                    "Qualcomm's measured Snapdragon X data. Offline.")
    p.add_argument("cmd", choices=["book", "law", "bcrit", "context", "engines",
                                   "runtimes", "critical", "ceiling", "speculate",
                                   "predict", "container", "anomalies", "selftest"])
    p.add_argument("--model", default="qwen3_4b", choices=sorted(
        k for k, a in ARCHS.items() if a.dense))
    p.add_argument("--format", default="w4a16", choices=sorted(WEIGHT_BITS))
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--accept", type=float, default=0.7)
    p.add_argument("--draft-ratio", type=float, default=0.15)
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    print(f"\n  source: {SNAPSHOT_SOURCE}")
    if args.cmd == "book":
        v = validate_against_book()
        print(f"\n  {'claim':<66}{'book':>9}{'ours':>11}")
        for r in v["rows"]:
            print(f"  {r['claim'][:65]:<66}{r['book']:>9g}{r['ours']:>11g}  "
                  f"{'ok' if r['ok'] else 'DRIFT'}")
        print(f"\n  {v['n_ok']}/{v['n']} reproduced -- {v['verdict']}")
    elif args.cmd == "law":
        print("\n  decode ms/token = intercept + bytes/token / bandwidth, fitted under "
              "six byte-accounting assumptions")
        for t in bandwidth_law_table():
            _print_law(t)
    elif args.cmd == "bcrit":
        print(f"\n  {'device':<9}{'runtime':<17}{'prec':<7}{'unit':<5}{'ctx':>6}"
              f"{'n':>3}{'mean':>8}{'range':>14}{'CV':>7}")
        for e in effective_critical_batch():
            print(f"  {e['device']:<9}{e['runtime']:<17}{e['precision']:<7}{e['unit']:<5}"
                  f"{e['context']:>6}{e['n']:>3}{e['mean']:>8}"
                  f"{e['min']:>7}-{e['max']:<6}{e['cv']:>7}")
        print("\n  flat across model size = the ratio is a property of the engine, "
              "as B_crit must be")
    elif args.cmd == "context":
        for c in context_slope():
            print(f"\n  {c['model']} on {c['device']} ({c['runtime']}, {c['points']} contexts,"
                  f" R^2 {c['r2']})")
            print(f"     {c['intercept_ms']} ms + {c['us_per_context_token']} us per context "
                  f"token -> KV overtakes weights at {c['t_star_measured']:,} tokens")
            print(f"     formula: {c['t_star_formula_kv16']:,} (16-bit KV), "
                  f"{c['t_star_formula_kv8']:,} (8-bit KV)   measured/formula "
                  f"{c['measured_over_formula_kv16']} / {c['measured_over_formula_kv8']}")
    elif args.cmd == "engines":
        for e in engine_asymmetry("X2 Elite", 512) + engine_asymmetry("X Elite", 512):
            print(f"\n  {e['model']} on {e['device']}, {e['context']} tokens")
            print(f"     decode  tok/s  cpu {e['decode']['cpu']:>7}  gpu {e['decode']['gpu']:>7}"
                  f"  npu {e['decode']['npu']:>7}  npu-qairt {e['qairt_npu_decode']}")
            print(f"     prefill tok/s  cpu {e['prefill']['cpu']:>7}  gpu {e['prefill']['gpu']:>7}"
                  f"  npu {e['prefill']['npu']:>7}  npu-qairt {e['qairt_npu_prefill']}")
            print(f"     best-engine spread: decode {e['best_engine_decode_spread']}x, "
                  f"prefill {e['best_engine_prefill_spread']}x")
    elif args.cmd == "runtimes":
        for r in runtime_spread():
            print(f"  {r['model']:<24}{r['precision']:<7}{r['device']:<9} ctx {r['context']}"
                  f"   genie {r['genie']:>6}  qairt {r['geniex_qairt']:>6}   x{r['ratio']}")
    elif args.cmd == "critical":
        print("\n  spec-sheet B_crit (INT8 TOPS / bandwidth x bytes/2) -- upper bounds")
        for r in critical_batch_table():
            print(f"  {r['soc']:<15}{r['format']:<16}{r['b_crit']:>8}")
    elif args.cmd == "ceiling":
        a = ARCHS[args.model]
        for key, s in SOCS.items():
            c = decode_ceiling(a, args.format, args.context, s.bw_gbs)
            print(f"  {key:<15} {s.bw_gbs:>5.0f} GB/s   ceiling {c:6.1f} tok/s   "
                  f"({args.model}, {args.format}, {args.context} ctx)")
        print(f"\n  KV overtakes weights at {kv_crossover(a, args.format):,.0f} tokens "
              f"(16-bit KV)")
    elif args.cmd == "speculate":
        print("\n  best speculative speedup, verifier on each engine (measured B_crit_eff)")
        print("  The critical batch only binds when drafts are cheap and usually right --")
        print("  prompt-lookup drafting for extraction, where the answer copies the input.")
        print("  Acceptance is modelled as independent per token, which real drafts are")
        print("  not: read every speedup below as an UPPER BOUND. What is robust is the")
        print("  ordering -- where the verifier's ridge binds, and on which engine.")
        scen = [(args.accept, args.draft_ratio, "your setting"),
                (0.7, 0.15, "small draft model"),
                (0.9, 0.05, "1-bit draft model"),
                (0.95, 0.01, "prompt lookup, document extraction")]
        for a, r, label in scen:
            print(f"\n  accept {a}, draft cost {r} ({label})")
            for eng, bc in (("NPU (X2 Elite, QAIRT)", 65.7), ("CPU (X2 Elite, llama.cpp)", 15.6)):
                b = best_speculation(a, r, bc)
                print(f"     {eng:<28} B_crit_eff {bc:>5}   best k {b['k']:>2}   "
                      f"speedup {b['speedup']:>6}x   verify free at that k: {b['verify_is_free']}")
    elif args.cmd == "predict":
        px = predict_x_plus()
        print(f"\n  {px['device']} -- stands in for {', '.join(px['hp_machines'])}")
        print(f"  published measurements: {px['published_measurements']}")
        for pr in px["predictions"][:40]:
            print(f"    {pr['model']:<24}{pr['precision']:<7}{pr['runtime']:<16}"
                  f"{pr['unit']:<4} ctx {pr['context']:<5} decode ~{pr['decode_tok_s']}")
        print(f"\n  test it: {px['test_with']}")
    elif args.cmd == "container":
        c = container_vs_context()
        for r in c["rows"]:
            print(f"  context {r['context']:>7,}   KV {r['kv_gb']:>6} GB   MLX moves "
                  f"{r['mlx_extra_bytes_pct']:>5}% more bytes per token")
        print(f"\n  weights and KV traffic are equal at {c['t_star_q1_0']:,} tokens")
    elif args.cmd == "anomalies":
        for a in anomalies():
            head = (f"  {a['device']} {a['runtime']} ctx {a['context']}: {a['larger']} "
                    f"moves {a['extra_bytes_pct']}% more bytes than {a['smaller']}")
            if a["marginal_bw_gbs"] is None:
                print(f"{head} yet takes {abs(a['extra_time_pct'])}% LESS time -- "
                      f"impossible with any shared per-token overhead")
            else:
                print(f"{head} in only {a['extra_time_pct']}% more time -- marginal "
                      f"{a['marginal_bw_gbs']} GB/s, above any peak ({a['peak_gbs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
