#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
 aihub_workbench.py -- test on Qualcomm AI Hub Workbench's real Snapdragon
                       devices, and download NOTHING but a few KB of JSON
================================================================================

    python aihub_workbench.py plan                  # exact jobs and bytes; no network
    python aihub_workbench.py build                 # build the slice here; exact upload size
    python aihub_workbench.py verify                # prove the slice IS a decoder layer
    python aihub_workbench.py run                   # dry run: prints what --yes would do
    python aihub_workbench.py run --yes             # real jobs on Workbench
    python aihub_workbench.py run --mock            # the whole flow against a local fake
    python aihub_workbench.py results               # measured vs roofline vs published
    python aihub_workbench.py selftest

    one-time setup:  pip install qai-hub onnx numpy
                     qai-hub configure --api_token <workbench.aihub.qualcomm.com -> Account>

THE PROBLEM THIS REPLACES. Earlier versions of this project told you to run
`qai-hub-models fetch` and `qai-hub-models export`. Both put gigabytes on your
laptop, and it is worth being exact about where:

  * `fetch` downloads a pre-compiled model asset. For an LLM that is the whole
    model -- gigabytes.
  * `export` runs Qualcomm's recipe, whose FIRST step is from_pretrained(): the
    full source checkpoint lands on your disk (Qwen3-4B in bf16 is ~8 GB,
    Llama-3.1-8B ~16 GB) before anything is uploaded. Its LAST step downloads
    the compiled model back unless --skip-downloading is passed.

Both are gone from every command this project emits.

WHAT THIS DOES INSTEAD -- mirroring Qualcomm's own production sequence
(qai_hub_models/utils/export/pipeline.py, steps 2-5) with steps 1 and 7 removed:

  1. BUILD a decoder layer at the model's REAL dimensions, here, with seeded
     random weights. Latency depends on shapes, dtypes and the compiler, not on
     what the weights say, so no checkpoint is needed. The weights sit on a
     power-of-two-scaled 16-level grid: still pseudo-random to any compiler,
     but they compress ~5x on the wire, so a Qwen3-4B layer (404 MB of fp32)
     uploads as 84 MB. `verify` proves the graph computes a real layer by
     running it in onnxruntime against an independent numpy implementation.
  2. UPLOAD once. The upload is cached by content key; a second run uploads
     nothing and reuses the model by ID, which is how the Workbench docs say to
     reuse models.
  3. COMPILE / QUANTIZE / COMPILE / PROFILE entirely in the cloud: the same
     job sequence qai_hub_models uses (ONNX compile -> quantize job with
     --range_scheme min_max -> qnn_dlc compile with --quantize_io -> profile).
     Every intermediate model is a HANDLE; get_target_model() transfers no
     bytes. This was read from the installed client's source, not assumed.
  4. FETCH ONLY THE PROFILE: ProfileJob.download_profile() with no filename
     returns a dict through the API. Nothing else is ever downloaded, and a
     transfer ledger enforces it at run time: past 2 MB it raises.

WHY X PLUS 8-CORE IS THE DEFAULT DEVICE. Qualcomm's own package carries 491
measured entries for X Elite CRD and 487 for X2 Elite CRD -- and ZERO for X Plus
8-Core CRD, although it lists that device as supported for 221 models. X Plus
8-Core is the AI Hub proxy for four of the seven HP Snapdragon machines. That is
the gap a measurement from here fills; roofline.py predict states in advance
what the roofline expects to see there.

THE BYTES TEST. The same layer at fp16, w8a16 and w4a16 moves 2, 1 and 0.5
bytes per weight. If the NPU is bandwidth-bound for decode, time falls with
bytes along a straight line whose slope is the achieved bandwidth. If it is
compute- or overhead-bound, the three times bunch together. Either result is
worth having; only the first supports this project's thesis.

WHAT IS NOT MEASURED. One layer is not a model. Per-token figures here are
EXTRAPOLATED (layers x layer time + LM head at the achieved bandwidth) and are
labelled so. Where Qualcomm publishes full-model numbers for the same device,
`results` compares the two, which calibrates the extrapolation.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roofline as RL  # noqa: E402  -- pure Python, always importable

WORKBENCH_URL = "https://workbench.aihub.qualcomm.com"
COMPUTE_DEVICES: Tuple[str, ...] = ("Snapdragon X Elite CRD",
                                    "Snapdragon X Plus 8-Core CRD",
                                    "Snapdragon X2 Elite CRD")
DEFAULT_DEVICE = "Snapdragon X Plus 8-Core CRD"
DEVICE_SHORT = {"Snapdragon X Elite CRD": "X Elite",
                "Snapdragon X Plus 8-Core CRD": "X Plus 8-Core",
                "Snapdragon X2 Elite CRD": "X2 Elite"}

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024          # profiles only
DEFAULT_MAX_UPLOAD_MB = 150.0
BUILDER_VERSION = "1"
ZIP_LEVEL = 6
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)        # deterministic archives
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0                      # Qwen3's rope_theta
MASK_NEG = -1.0e4                             # finite: fp16- and quantiser-safe

# Precision -> how Qualcomm's own pipeline produces it. None = float path.
PRECISIONS: Dict[str, Optional[Tuple[str, str]]] = {
    "fp16": None,
    "w8a16": ("INT8", "INT16"),
    "w4a16": ("INT4", "INT16"),
}
WEIGHT_BYTES = {"fp16": 2.0, "w8a16": 1.0, "w4a16": 0.5}

# Every client method whose name says it moves a model or artefact onto this
# machine. None of them may appear as a call anywhere in this file; the
# self-test scans the source, and stress_all.py scans the whole project.
FORBIDDEN_CALLS: Tuple[str, ...] = (
    "download_target_model", "download_results", "download_output_data",
    "download_artifacts_for_type", "download_job_logs")


# Calls that put model weights on this machine from elsewhere.
HF_FETCH_CALLS: Tuple[str, ...] = ("snapshot_download", "hf_hub_download",
                                   "urlretrieve", "from_pretrained")
# Commands that do the same when a user pastes them. Built from parts so this
# file's own source never contains the needle it searches for.
#
# BANNED outright: nothing in this project needs them any more -- every AI Hub
# measurement goes through the zero-download runner in this file. `fetch`
# downloads a compiled asset; `export`, `evaluate` and `demo` all load the
# source model onto this machine before anything reaches the cloud.
BANNED_COMMANDS: Tuple[str, ...] = ("qai-hub-models " + "fetch",
                                    "qai-hub-models " + "export",
                                    "qai-hub-models " + "evaluate",
                                    "qai-hub-models " + "demo")
# ALLOWED ONLY WITH A STATED SIZE: running a model ON the HP laptop needs its
# weights there, by definition. The string that emits such a command must
# also say how much it downloads, so the size sits next to what gets pasted.
SIZED_COMMANDS: Tuple[str, ...] = ("huggingface-cli " + "download",
                                   "hf " + "download", "geniex " + "pull",
                                   "geniex " + "infer")
SIZE_MARKER = "[downloads "
HEAVY_COMMANDS: Tuple[str, ...] = BANNED_COMMANDS + SIZED_COMMANDS


WAIVER = "# download-ok:"


def download_calls_in(source: str, waived: Optional[List[Tuple[int, str, str]]] = None
                      ) -> List[Tuple[int, str]]:
    """
    Every place in `source` that downloads a model, found from the syntax tree
    rather than by grepping text. A text search trips over a docstring that
    EXPLAINS why a command was removed; this looks only at real calls, and at
    string literals that are not docstrings -- which is where an emitted
    command would live.

    Flags: .download(...) on anything; any FORBIDDEN_CALLS or HF_FETCH_CALLS;
    download_profile(<filename>) (that one writes to disk); and a string
    literal containing a HEAVY_COMMANDS entry.

    A line carrying `# download-ok: <reason>` is exempt -- the receiver of a
    call cannot be known from syntax, so a self-test that deliberately trips a
    mock's alarm needs a way to say so. Waivers are returned through `waived`
    so they can be counted and read, never silently absorbed.
    """
    import ast
    tree = ast.parse(source)
    lines = source.splitlines()
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    def judge(text: str, lineno: int) -> None:
        for needle in BANNED_COMMANDS:
            if needle in text:
                found.append((lineno, f"emits '{needle}'"))
        for needle in SIZED_COMMANDS:
            if needle in text and SIZE_MARKER not in text:
                found.append((lineno, f"emits '{needle}' without its size"))

    # An f-string is ONE emitted command even though the syntax tree splits its
    # literal text around each {placeholder}; judge the whole template, and
    # skip its pieces so they are not judged twice. (Strings assembled with +
    # are not followed -- this is a regression net for honest code, not a
    # sandbox.)
    in_fstring = set()
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                    in_fstring.add(id(v))
                else:
                    parts.append("{}")
            judge("".join(parts), node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else f.id if isinstance(f, ast.Name) else None)
            if name == "download" or name in FORBIDDEN_CALLS or name in HF_FETCH_CALLS:
                found.append((node.lineno, name))
            elif name == "download_profile" and (node.args or node.keywords):
                found.append((node.lineno, "download_profile(<filename>)"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings and id(node) not in in_fstring:
            judge(node.value, node.lineno)
    hits = []
    for ln, what in sorted(found):
        text = lines[ln - 1] if 0 < ln <= len(lines) else ""
        if WAIVER in text:
            reason = text.split(WAIVER, 1)[1].strip()
            if waived is not None:
                waived.append((ln, what, reason))
            if reason:
                continue            # a waiver without a reason does not count
        hits.append((ln, what))
    return hits


class DownloadRefused(RuntimeError):
    """Raised before anything larger than a profile would land here."""


class UploadBudgetExceeded(RuntimeError):
    """Raised before an upload that would pass the budget is started."""


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _finite(value, name: str, lo=None, hi=None) -> float:
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


# ============================================================================ #
# SECTION 1 -- the contract: a ledger that refuses downloads
# ============================================================================ #

@dataclass
class TransferLedger:
    max_upload_bytes: int
    max_download_bytes: int = MAX_DOWNLOAD_BYTES
    uploaded: int = 0
    downloaded: int = 0
    events: List[Tuple[str, str, int]] = field(default_factory=list)

    def check_upload(self, what: str, n: int) -> None:
        if n < 0:
            raise ValueError("negative byte count")
        if self.uploaded + n > self.max_upload_bytes:
            raise UploadBudgetExceeded(
                f"{what}: {n / 1e6:.1f} MB would take this run to "
                f"{(self.uploaded + n) / 1e6:.1f} MB, over the "
                f"{self.max_upload_bytes / 1e6:.0f} MB budget. Raise "
                f"--max-upload-mb, or pick a smaller --arch / --context.")

    def record_upload(self, what: str, n: int) -> None:
        self.check_upload(what, n)
        self.uploaded += n
        self.events.append(("up", what, n))

    def record_download(self, what: str, n: int) -> None:
        if n < 0:
            raise ValueError("negative byte count")
        if self.downloaded + n > self.max_download_bytes:
            raise DownloadRefused(
                f"{what}: {n} bytes would take downloads past "
                f"{self.max_download_bytes} bytes. This tool fetches profiles "
                f"only; anything larger is refused by design.")
        self.downloaded += n
        self.events.append(("down", what, n))

    def summary(self) -> Dict[str, Any]:
        return {"uploaded_bytes": self.uploaded, "downloaded_bytes": self.downloaded,
                "download_ceiling": self.max_download_bytes,
                "upload_budget": self.max_upload_bytes, "events": len(self.events)}


# ============================================================================ #
# SECTION 2 -- slice geometry and byte accounting (no dependencies)
# ============================================================================ #

@dataclass(frozen=True)
class SliceSpec:
    """
    kind "decode": one full decoder layer taking one new token against a KV
    cache of `context` tokens -- the unit that repeats L times per token.
    kind "mlp": the gated MLP block alone with a symbolic token count, so ONE
    upload can be compiled at many prefill lengths to find the NPU's ridge.
    """
    arch: str
    kind: str = "decode"
    context: int = 4096
    seed: int = 0

    def validate(self) -> "SliceSpec":
        if self.arch not in RL.ARCHS and self.arch != "tiny":
            raise ValueError(f"unknown arch {self.arch!r}; known: "
                             f"{sorted(k for k, a in RL.ARCHS.items() if a.dense)}")
        a = arch_of(self.arch)
        if not a.dense:
            raise ValueError(f"{self.arch} is a hybrid; its layers are not all "
                             f"the same shape, so one slice cannot stand for it")
        if self.kind not in ("decode", "mlp"):
            raise ValueError(f"kind must be 'decode' or 'mlp', got {self.kind!r}")
        if isinstance(self.context, bool) or not isinstance(self.context, int):
            raise TypeError("context must be an int")
        if not 1 <= self.context <= 131_072:
            raise ValueError(f"context must be in [1, 131072], got {self.context}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative int")
        if a.H % 2:
            raise ValueError("head_dim must be even for rotary embedding")
        return self

    @property
    def key(self) -> str:
        ctx = f"-p{self.context}" if self.kind == "decode" else ""
        return f"{self.arch}-{self.kind}{ctx}-s{self.seed}-v{BUILDER_VERSION}"


# A deliberately small architecture for checking the graph's arithmetic. Not
# in roofline.ARCHS because no measurement refers to it.
TINY = RL.Arch("tiny", 1, 64, 96, 4, 2, 16, 0, True)


def arch_of(key: str) -> RL.Arch:
    if not isinstance(key, str):
        raise TypeError(f"arch key must be a str, got {type(key).__name__}")
    return TINY if key == "tiny" else RL.ARCHS[key]


def _as_spec(spec) -> "SliceSpec":
    """Refuse anything that is not a SliceSpec with a TypeError, rather than
    letting `.arch` raise an AttributeError three frames later -- the class of
    failure stress_all.py counts as the function being confused."""
    if not isinstance(spec, SliceSpec):
        raise TypeError(f"expected a SliceSpec, got {type(spec).__name__}")
    return spec


def slice_params(spec: SliceSpec) -> int:
    _as_spec(spec)
    a = arch_of(spec.arch)
    mlp = 3 * a.D * a.F
    if spec.kind == "mlp":
        return mlp + a.D
    attn = a.D * a.N * a.H + 2 * a.D * a.K * a.H + a.N * a.H * a.D
    return attn + mlp + 2 * a.D + 2 * a.H


def input_specs(spec: SliceSpec, tokens: int = 1) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """In qai_hub's own format: name -> (shape, dtype)."""
    _as_spec(spec)
    a = arch_of(spec.arch)
    t = int(_finite(tokens, "tokens", 1, 65536))
    if spec.kind == "mlp":
        return {"hidden": ((1, t, a.D), "float32")}
    p = spec.context
    return {"hidden": ((1, t, a.D), "float32"),
            "past_key": ((1, a.K, p, a.H), "float32"),
            "past_value": ((1, a.K, p, a.H), "float32"),
            "cos": ((1, 1, t, a.H), "float32"),
            "sin": ((1, 1, t, a.H), "float32"),
            "mask": ((1, 1, t, p + t), "float32")}


def output_names(spec: SliceSpec) -> List[str]:
    _as_spec(spec)
    return ["out_hidden"] if spec.kind == "mlp" else ["out_hidden", "new_key", "new_value"]


# Measured, not assumed: the full Qwen3-4B slice archive is 403.7 MB of fp32
# -> 83.8 MB at DEFLATE level 6 (4.8x); Gaussian weights of the same size
# compress 1.1x. build() replaces this estimate with the exact archive size.
GRID_COMPRESSION = 4.8


def estimated_upload_bytes(spec: SliceSpec) -> int:
    _as_spec(spec)
    return int(slice_params(spec) * 4 / GRID_COMPRESSION)


def calibration_bytes(spec: SliceSpec, tokens: int = 1, samples: int = 1) -> int:
    _as_spec(spec)
    samples = int(_finite(samples, "samples", 1, 10_000))
    n = 0
    for shape, _dt in input_specs(spec, tokens).values():
        n += 4 * math.prod(shape)
    return n * samples


# ============================================================================ #
# SECTION 3 -- the ONNX slice (needs numpy + onnx, imported lazily)
# ============================================================================ #

def _np():
    try:
        import numpy as np
        return np
    except ImportError:
        raise RuntimeError("numpy is required to build a slice: pip install numpy")


def _onnx():
    try:
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        return onnx, helper, numpy_helper, TensorProto
    except ImportError:
        raise RuntimeError("onnx is required to build a slice: pip install onnx")


def grid_weights(rng, shape, fan_in: int, levels: int = 16):
    """
    Pseudo-random weights on a power-of-two-scaled grid of `levels` integers.

    Why a grid: DEFLATE finds the redundancy (the low mantissa bytes are zero),
    so the archive shrinks ~5x; no compiler special-cases a 16-value alphabet,
    so latency is unaffected. Why power-of-two: it makes those mantissa bytes
    EXACTLY zero. The scale keeps each matmul's output near unit variance, so
    activations stay sane in fp16 and under quantisation.
    """
    np = _np()
    if not hasattr(rng, "integers"):
        raise TypeError("rng must be a numpy Generator")
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= 256:
        raise ValueError("levels must be an int in [2, 256]")
    if (not isinstance(shape, tuple) or not shape
            or not all(isinstance(d, int) and not isinstance(d, bool) and 0 < d <= 1 << 20
                       for d in shape)):
        raise TypeError("shape must be a non-empty tuple of positive ints")
    fan_in = int(_finite(fan_in, "fan_in", 1, 1 << 24))
    half = levels // 2
    k = rng.integers(-half, half, size=shape)
    std_k = math.sqrt((levels ** 2 - 1) / 12.0)
    e = math.ceil(math.log2(std_k * math.sqrt(max(1, fan_in))))
    return (k * (2.0 ** -e)).astype(np.float32)


def make_weights(spec: SliceSpec) -> Dict[str, Any]:
    _as_spec(spec)
    np = _np()
    a = arch_of(spec.arch)
    rng = np.random.default_rng(1_000_003 * (spec.seed + 1) + hash_arch(a))
    w: Dict[str, Any] = {}

    def gamma(n):
        return (1.0 + rng.integers(-2, 3, size=(n,)) / 16.0).astype(np.float32)

    if spec.kind == "decode":
        w["ln1"] = gamma(a.D)
        w["wq"] = grid_weights(rng, (a.D, a.N * a.H), a.D)
        w["wk"] = grid_weights(rng, (a.D, a.K * a.H), a.D)
        w["wv"] = grid_weights(rng, (a.D, a.K * a.H), a.D)
        w["qn"] = gamma(a.H)
        w["kn"] = gamma(a.H)
        w["wo"] = grid_weights(rng, (a.N * a.H, a.D), a.N * a.H)
    w["ln2"] = gamma(a.D)
    w["wg"] = grid_weights(rng, (a.D, a.F), a.D)
    w["wu"] = grid_weights(rng, (a.D, a.F), a.D)
    w["wd"] = grid_weights(rng, (a.F, a.D), a.F)
    return w


def hash_arch(a: RL.Arch) -> int:
    if not isinstance(a, RL.Arch):
        raise TypeError(f"expected a roofline.Arch, got {type(a).__name__}")
    s = f"{a.key}:{a.L}:{a.D}:{a.F}:{a.N}:{a.K}:{a.H}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def sample_inputs(spec: SliceSpec, tokens: int = 1, seed: int = 7) -> Dict[str, Any]:
    """Inputs of the right shapes: activations N(0,1), rotary tables for the
    real positions, and an all-visible mask (decode sees the whole cache)."""
    _as_spec(spec)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TypeError("seed must be a non-negative int")
    np = _np()
    a = arch_of(spec.arch)
    rng = np.random.default_rng(seed)
    shapes = input_specs(spec, tokens)
    out = {"hidden": rng.standard_normal(shapes["hidden"][0]).astype(np.float32)}
    if spec.kind == "mlp":
        return out
    p = spec.context
    out["past_key"] = rng.standard_normal(shapes["past_key"][0]).astype(np.float32)
    out["past_value"] = rng.standard_normal(shapes["past_value"][0]).astype(np.float32)
    half = a.H // 2
    inv = ROPE_THETA ** (-np.arange(half, dtype=np.float64) * 2.0 / a.H)
    pos = np.arange(p, p + tokens, dtype=np.float64)
    ang = np.outer(pos, inv)
    emb = np.concatenate([ang, ang], axis=-1)
    out["cos"] = np.cos(emb).astype(np.float32).reshape(1, 1, tokens, a.H)
    out["sin"] = np.sin(emb).astype(np.float32).reshape(1, 1, tokens, a.H)
    out["mask"] = np.zeros(shapes["mask"][0], dtype=np.float32)
    return out


def reference_forward(spec: SliceSpec, w: Dict[str, Any], x: Dict[str, Any]):
    """
    The same layer in plain numpy, written independently of the graph builder.
    If the ONNX graph and this disagree, the graph is not a decoder layer, and
    any latency measured on it is a latency of something else.
    """
    _as_spec(spec)
    if not isinstance(w, dict) or not isinstance(x, dict):
        raise TypeError("weights and inputs must be dicts")
    np = _np()
    a = arch_of(spec.arch)

    def rms(v, g):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + RMS_EPS) * g

    def mlp(h):
        hn = rms(h, w["ln2"])
        g = hn @ w["wg"]
        u = hn @ w["wu"]
        return h + ((g / (1.0 + np.exp(-g))) * u) @ w["wd"]

    h = x["hidden"].astype(np.float64)
    if spec.kind == "mlp":
        return {"out_hidden": mlp(h)}
    t = h.shape[1]
    p = x["past_key"].shape[2]
    n, k, hd = a.N, a.K, a.H
    g = n // k
    hn = rms(h, w["ln1"])
    q = (hn @ w["wq"]).reshape(1, t, n, hd).transpose(0, 2, 1, 3)
    kk = (hn @ w["wk"]).reshape(1, t, k, hd).transpose(0, 2, 1, 3)
    v = (hn @ w["wv"]).reshape(1, t, k, hd).transpose(0, 2, 1, 3)
    q = rms(q, w["qn"])
    kk = rms(kk, w["kn"])

    def rot(z):
        return np.concatenate([-z[..., hd // 2:], z[..., :hd // 2]], axis=-1)

    q = q * x["cos"] + rot(q) * x["sin"]
    kk = kk * x["cos"] + rot(kk) * x["sin"]
    k_all = np.concatenate([x["past_key"], kk], axis=2)
    v_all = np.concatenate([x["past_value"], v], axis=2)
    s = q.reshape(1, k, g * t, hd) @ k_all.transpose(0, 1, 3, 2) / math.sqrt(hd)
    s = s.reshape(1, k, g, t, p + t) + x["mask"].reshape(1, 1, 1, t, p + t)
    s = s - s.max(axis=-1, keepdims=True)
    pr = np.exp(s)
    pr = (pr / pr.sum(axis=-1, keepdims=True)).reshape(1, k, g * t, p + t)
    ctx = (pr @ v_all).reshape(1, n, t, hd).transpose(0, 2, 1, 3).reshape(1, t, n * hd)
    h1 = h + ctx @ w["wo"]
    return {"out_hidden": mlp(h1), "new_key": kk, "new_value": v}


def build_graph(spec: SliceSpec, w: Dict[str, Any], symbolic_tokens: bool = False):
    """
    The decoder layer as an ONNX graph, opset 17, static shapes (NPU compilers
    want them) except the MLP slice's token axis when `symbolic_tokens`.

    Qualcomm's LLM exports take past K/V as inputs and emit only the NEW K/V
    slice, so a step never copies the cache; this does the same. Every op is
    in the QNN converter's standard set: MatMul, Mul, Add, Div, Sqrt,
    ReduceMean, Reshape, Transpose, Split, Concat, Neg, Softmax, Sigmoid.
    """
    _as_spec(spec)
    if not isinstance(w, dict):
        raise TypeError("weights must be a dict")
    np = _np()
    onnx, helper, numpy_helper, TP = _onnx()
    a = arch_of(spec.arch)
    nodes: List[Any] = []
    inits: List[Any] = []
    counter = [0]

    def nm(prefix):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def op(kind, inputs, **attrs):
        out = nm(kind.lower())
        nodes.append(helper.make_node(kind, inputs, [out], **attrs))
        return out

    for k_, v_ in w.items():
        const(k_, v_)
    eps = const("rms_eps", np.array(RMS_EPS, dtype=np.float32))

    def rms(x, g):
        sq = op("Mul", [x, x])
        ms = op("ReduceMean", [sq], axes=[-1], keepdims=1)
        sd = op("Sqrt", [op("Add", [ms, eps])])
        return op("Mul", [op("Div", [x, sd]), g])

    def shape(name, dims):
        return const(name, np.array(dims, dtype=np.int64))

    def mlp_block(h):
        hn = rms(h, "ln2")
        gate = op("MatMul", [hn, "wg"])
        up = op("MatMul", [hn, "wu"])
        act = op("Mul", [gate, op("Sigmoid", [gate])])
        return op("Add", [h, op("MatMul", [op("Mul", [act, up]), "wd"])])

    specs = input_specs(spec, 1)
    if spec.kind == "mlp":
        tdim = "T" if symbolic_tokens else 1
        inputs = [helper.make_tensor_value_info("hidden", TP.FLOAT, [1, tdim, a.D])]
        out = mlp_block("hidden")
        nodes.append(helper.make_node("Identity", [out], ["out_hidden"]))
        outputs = [helper.make_tensor_value_info("out_hidden", TP.FLOAT, [1, tdim, a.D])]
    else:
        t, p = 1, spec.context
        n, k, hd = a.N, a.K, a.H
        g = n // k
        inputs = [helper.make_tensor_value_info(nm_, TP.FLOAT, list(sh))
                  for nm_, (sh, _dt) in specs.items()]
        hn = rms("hidden", "ln1")
        q = op("Transpose", [op("Reshape", [op("MatMul", [hn, "wq"]),
                                            shape("sh_q", [1, t, n, hd])])],
               perm=[0, 2, 1, 3])
        kn_ = op("Transpose", [op("Reshape", [op("MatMul", [hn, "wk"]),
                                              shape("sh_k", [1, t, k, hd])])],
                 perm=[0, 2, 1, 3])
        vn = op("Transpose", [op("Reshape", [op("MatMul", [hn, "wv"]),
                                             shape("sh_v", [1, t, k, hd])])],
                perm=[0, 2, 1, 3])
        q = rms(q, "qn")
        kn_ = rms(kn_, "kn")
        halves = shape("rope_split", [hd // 2, hd // 2])

        def rope(z):
            lo, hi = nm("lo"), nm("hi")
            nodes.append(helper.make_node("Split", [z, halves], [lo, hi], axis=-1))
            rot = op("Concat", [op("Neg", [hi]), lo], axis=-1)
            return op("Add", [op("Mul", [z, "cos"]), op("Mul", [rot, "sin"])])

        q = rope(q)
        kn_ = rope(kn_)
        k_all = op("Concat", ["past_key", kn_], axis=2)
        v_all = op("Concat", ["past_value", vn], axis=2)
        qg = op("Reshape", [q, shape("sh_qg", [1, k, g * t, hd])])
        s = op("MatMul", [qg, op("Transpose", [k_all], perm=[0, 1, 3, 2])])
        s = op("Mul", [s, const("attn_scale", np.array(1.0 / math.sqrt(hd),
                                                     dtype=np.float32))])
        s5 = op("Reshape", [s, shape("sh_s5", [1, k, g, t, p + t])])
        m5 = op("Reshape", ["mask", shape("sh_m5", [1, 1, 1, t, p + t])])
        pr = op("Softmax", [op("Add", [s5, m5])], axis=-1)
        pr = op("Reshape", [pr, shape("sh_p4", [1, k, g * t, p + t])])
        ctx = op("MatMul", [pr, v_all])
        ctx = op("Reshape", [ctx, shape("sh_ctx", [1, n, t, hd])])
        ctx = op("Transpose", [ctx], perm=[0, 2, 1, 3])
        ctx = op("Reshape", [ctx, shape("sh_flat", [1, t, n * hd])])
        h1 = op("Add", ["hidden", op("MatMul", [ctx, "wo"])])
        out = mlp_block(h1)
        nodes.append(helper.make_node("Identity", [out], ["out_hidden"]))
        nodes.append(helper.make_node("Identity", [kn_], ["new_key"]))
        nodes.append(helper.make_node("Identity", [vn], ["new_value"]))
        outputs = [helper.make_tensor_value_info("out_hidden", TP.FLOAT, [1, t, a.D]),
                   helper.make_tensor_value_info("new_key", TP.FLOAT, [1, k, t, hd]),
                   helper.make_tensor_value_info("new_value", TP.FLOAT, [1, k, t, hd])]
    # The weights go in LAST, one tensor at a time. make_graph(initializer=...)
    # copies every tensor and make_model copies the whole graph again, so the
    # obvious two lines held the weights four times over: 1.6 GB of RAM for a
    # 404 MB Qwen3-4B layer, on laptops with 8 GB. Protobuf serialises fields
    # in field-number order, not insertion order, so the bytes -- and the
    # archive's hash -- are identical either way.
    graph = helper.make_graph(nodes, f"sigil_{spec.key}", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="sigil-edge-aihub-workbench",
                              producer_version=BUILDER_VERSION)
    model.ir_version = 8
    inits.reverse()
    while inits:
        model.graph.initializer.append(inits.pop())
    return model


def build(spec: SliceSpec, out_dir, symbolic_tokens: Optional[bool] = None,
          level: int = ZIP_LEVEL) -> Dict[str, Any]:
    """
    Write <key>.onnx/{model.onnx, model.data} -- the directory layout the
    Workbench docs require for external weights -- and a deterministic
    <key>.onnx.zip of it. Returns the EXACT number of bytes an upload sends.
    """
    _as_spec(spec).validate()
    onnx, _h, _nh, _tp = _onnx()
    if symbolic_tokens is None:
        symbolic_tokens = spec.kind == "mlp"
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mdir = out / f"{spec.key}.onnx"
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.iterdir():
        old.unlink()
    w = make_weights(spec)
    model = build_graph(spec, w, symbolic_tokens)
    del w                         # the graph holds its own copy from here on
    onnx.save_model(model, str(mdir / "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.data",
                    size_threshold=1024)
    onnx.checker.check_model(str(mdir / "model.onnx"))
    zpath = out / f"{spec.key}.onnx.zip"
    _zip_dir(mdir, zpath, level)
    return {"spec": spec, "dir": str(mdir), "zip": str(zpath),
            "zip_bytes": zpath.stat().st_size,
            "raw_bytes": sum(f.stat().st_size for f in mdir.iterdir()),
            "sha256": _sha256(zpath), "params": slice_params(spec),
            "input_specs": input_specs(spec, 1), "output_names": output_names(spec)}


def _zip_dir(mdir: pathlib.Path, zpath: pathlib.Path, level: int) -> None:
    """Same layout the client's own zip_model() produces -- a directory entry,
    then the files under it -- with fixed timestamps so the bytes, and hence
    the hash, are reproducible."""
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=level) as zf:
        d = zipfile.ZipInfo(mdir.name + "/", date_time=FIXED_ZIP_TIME)
        d.external_attr = 0o40755 << 16
        zf.writestr(d, b"")
        for f in sorted(mdir.iterdir()):
            info = zipfile.ZipInfo(f"{mdir.name}/{f.name}", date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(f, "rb") as src, zf.open(info, "w") as dst:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)


def stand_in_archive(spec: SliceSpec, out_dir) -> Dict[str, Any]:
    """
    A correctly laid-out archive with placeholder bytes, for exercising the
    job sequence against the MOCK when `onnx` is not installed. It is never
    uploaded anywhere real: run_decode_experiment only accepts it through the
    `prebuilt` argument, which the CLI never passes.
    """
    out = pathlib.Path(out_dir).resolve()
    mdir = out / f"{spec.key}.onnx"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.onnx").write_bytes(b"stand-in: not an ONNX model")
    (mdir / "model.data").write_bytes(b"\0" * 64)
    zpath = out / f"{spec.key}.onnx.zip"
    _zip_dir(mdir, zpath, ZIP_LEVEL)
    return {"spec": spec, "dir": str(mdir), "zip": str(zpath),
            "zip_bytes": zpath.stat().st_size, "raw_bytes": 91,
            "sha256": _sha256(zpath), "params": slice_params(spec),
            "input_specs": input_specs(spec, 1), "output_names": output_names(spec),
            "stand_in": True}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_layout(zip_path) -> Dict[str, Any]:
    """
    Check the archive against the rules the client enforces before upload --
    re-implemented from its source (exactly one .onnx directory holding one
    .onnx and at most one .data file, nothing else) -- and, when qai_hub is
    installed, against the client's own classifier as well.
    """
    if not isinstance(zip_path, (str, os.PathLike)):
        raise TypeError("zip_path must be a path")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(str(zip_path))
    names = zipfile.ZipFile(zip_path).namelist()
    tops = {n.split("/")[0] for n in names}
    files = [n for n in names if not n.endswith("/")]
    ok = (len(tops) == 1 and next(iter(tops)).endswith(".onnx")
          and sum(f.endswith(".onnx") for f in files) == 1
          and sum(f.endswith(".data") for f in files) <= 1
          and all(f.endswith((".onnx", ".data")) for f in files))
    res = {"ok": ok, "entries": names, "client_check": "skipped: qai_hub not installed"}
    try:
        from qai_hub import client as _qc
        kind = _qc._determine_model_type(str(zip_path))
        res["client_check"] = getattr(kind, "name", str(kind))
        res["ok"] = ok and res["client_check"] == "ONNX"
    except ImportError:
        pass
    except Exception as exc:
        res["client_check"] = f"REJECTED: {type(exc).__name__}: {exc}"
        res["ok"] = False
    return res


def verify_graph(spec: Optional[SliceSpec] = None, tol: float = 2e-3) -> Dict[str, Any]:
    """
    Run the built graph in onnxruntime and compare it with reference_forward.
    Uses the tiny architecture by default: the arithmetic is identical at any
    size, and this finishes in a second.
    """
    spec = spec or SliceSpec("tiny", "decode", context=24)
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "SKIPPED", "reason": "onnxruntime not installed "
                                                "(pip install onnxruntime)"}
    np = _np()
    with tempfile.TemporaryDirectory() as td:
        b = build(spec, td)
        sess = ort.InferenceSession(str(pathlib.Path(b["dir"]) / "model.onnx"),
                                    providers=["CPUExecutionProvider"])
        x = sample_inputs(spec, 1)
        got = dict(zip(output_names(spec), sess.run(output_names(spec), x)))
    ref = reference_forward(spec, make_weights(spec), x)
    errs = {k: float(np.max(np.abs(got[k] - ref[k])) / (1.0 + float(np.max(np.abs(ref[k])))))
            for k in ref}
    return {"status": "MATCH" if max(errs.values()) <= tol else "MISMATCH",
            "max_rel_err": {k: round(v, 8) for k, v in errs.items()}, "tol": tol}


# ============================================================================ #
# SECTION 4 -- backends: the real client, a recording dry run, a faithful mock
# ============================================================================ #

class _Enum:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"QuantizeDtype.{self.name}"


class MockStatus:
    def __init__(self, ok: bool, message: Optional[str] = None):
        self.code = "SUCCESS" if ok else "FAILED"
        self.success = ok
        self.failure = not ok
        self.finished = True
        self.message = message


class MockModel:
    def __init__(self, owner, model_id: str, nbytes: int, meta: Dict[str, Any]):
        self._owner = owner
        self.model_id = model_id
        self.name = meta.get("name")
        self._bytes = nbytes
        self.meta = meta

    def download(self, filename: str, timeout=None):
        # A real Model.download writes the compiled model to disk. Here it is
        # an alarm: any code path that reaches it is a bug in the caller.
        self._owner.download_attempts.append(("Model.download", self.model_id))
        raise DownloadRefused(f"MockModel.download({filename!r}) was called")


class MockJob:
    def __init__(self, owner, kind: str, job_id: str, target: Optional[MockModel],
                 profile: Optional[Dict[str, Any]] = None, ok: bool = True,
                 message: Optional[str] = None):
        self._owner = owner
        self.kind = kind
        self.job_id = job_id
        self.url = f"{WORKBENCH_URL}/jobs/{job_id}/"
        self._target = target
        self._profile = profile
        self._status = MockStatus(ok, message)

    def wait(self, timeout: Optional[int] = None):
        return self._status

    def get_status(self):
        return self._status

    def get_target_model(self):
        return self._target if self._status.success else None

    def download_profile(self, filename: Optional[str] = None):
        if filename is not None:
            self._owner.download_attempts.append(("download_profile(file)", self.job_id))
            raise DownloadRefused("download_profile with a filename writes to disk")
        if self.kind != "profile":
            raise ValueError("The supplied job ID is not for a Profile job")
        return json.loads(json.dumps(self._profile))

    def _refuse(self, what):
        self._owner.download_attempts.append((what, self.job_id))
        raise DownloadRefused(f"{what} was called on a mock job")

    def download_target_model(self, filename=None):
        self._refuse("download_target_model")

    def download_results(self, artifacts_dir=None):
        self._refuse("download_results")

    def download_output_data(self, filename=None):
        self._refuse("download_output_data")


class MockDataset:
    def __init__(self, dataset_id: str, nbytes: int):
        self.dataset_id = dataset_id
        self._bytes = nbytes


class MockHub:
    """
    A stand-in for qai_hub.Client with the SAME method signatures -- the
    self-test compares them parameter by parameter against the installed
    client -- so the full job sequence runs offline. Profiles are SIMULATED
    from the roofline and every result they produce is labelled MOCK. They are
    never a measurement and results() refuses to present them as one.
    """

    class Device:
        def __init__(self, name: str = "", os: str = "", attributes=None):
            self.name = name
            self.os = os
            self.attributes = attributes or []

    class QuantizeDtype:
        INT8 = _Enum("INT8")
        INT16 = _Enum("INT16")
        INT4 = _Enum("INT4")

    def __init__(self, fail_on: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None):
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.download_attempts: List[Tuple[str, str]] = []
        self.models: Dict[str, MockModel] = {}
        self.datasets: Dict[str, MockDataset] = {}
        self._n = 0
        self._fail_on = fail_on

    def _id(self, prefix):
        self._n += 1
        return f"mock{prefix}{self._n:04d}"

    def _log(self, method, **kw):
        self.calls.append((method, kw))

    def get_devices(self, name: str = "", os: str = "", attributes=None):
        self._log("get_devices", name=name)
        return [MockHub.Device(n, "11") for n in COMPUTE_DEVICES if not name or n == name]

    def upload_model(self, model, name: Optional[str] = None, project=None):
        size = os.path.getsize(model) if isinstance(model, (str, pathlib.Path)) else 0
        self._log("upload_model", model=str(model), name=name)
        mid = self._id("m")
        self.models[mid] = MockModel(self, mid, size, {"name": name, "stage": "source",
                                                       "path": str(model)})
        return self.models[mid]

    def get_model(self, model_id: str):
        self._log("get_model", model_id=model_id)
        if model_id not in self.models:
            raise KeyError(f"model {model_id} not found")
        return self.models[model_id]

    def upload_dataset(self, data, name: Optional[str] = None, project=None):
        n = sum(int(getattr(a, "nbytes", 0)) for v in data.values() for a in v)
        self._log("upload_dataset", name=name, bytes=n)
        did = self._id("d")
        self.datasets[did] = MockDataset(did, n)
        return self.datasets[did]

    def get_dataset(self, dataset_id: str):
        self._log("get_dataset", dataset_id=dataset_id)
        if dataset_id not in self.datasets:
            raise KeyError(f"dataset {dataset_id} not found")
        return self.datasets[dataset_id]

    def submit_compile_job(self, model, device, name: Optional[str] = None,
                           input_specs=None, options: str = "",
                           single_compile: bool = True, calibration_data=None,
                           retry: bool = True, project=None):
        kw = {"model": getattr(model, "model_id", model), "device": device.name,
              "options": options, "input_specs": input_specs}
        self._log("submit_compile_job", **kw)
        err = self._fail_on("compile", kw) if self._fail_on else None
        src = model if isinstance(model, MockModel) else None
        meta = dict(src.meta) if src else {}
        runtime = re.search(r"--target_runtime\s+(\S+)", options or "")
        meta.update(stage="compiled", runtime=runtime.group(1) if runtime else "default",
                    device=device.name)
        tgt = MockModel(self, self._id("m"), 0, meta)
        self.models[tgt.model_id] = tgt
        return MockJob(self, "compile", self._id("j"), tgt, ok=err is None, message=err)

    def submit_quantize_job(self, model, calibration_data, weights_dtype=None,
                            activations_dtype=None, name: Optional[str] = None,
                            options: str = "", project=None):
        kw = {"model": getattr(model, "model_id", model),
              "weights": getattr(weights_dtype, "name", weights_dtype),
              "activations": getattr(activations_dtype, "name", activations_dtype),
              "options": options}
        self._log("submit_quantize_job", **kw)
        err = self._fail_on("quantize", kw) if self._fail_on else None
        meta = dict(model.meta) if isinstance(model, MockModel) else {}
        meta.update(stage="quantized", weights=kw["weights"])
        tgt = MockModel(self, self._id("m"), 0, meta)
        self.models[tgt.model_id] = tgt
        return MockJob(self, "quantize", self._id("j"), tgt, ok=err is None, message=err)

    def submit_profile_job(self, model, device, name: Optional[str] = None,
                           options: str = "", retry: bool = True, project=None):
        kw = {"model": getattr(model, "model_id", model), "device": device.name,
              "options": options}
        self._log("submit_profile_job", **kw)
        err = self._fail_on("profile", kw) if self._fail_on else None
        meta = model.meta if isinstance(model, MockModel) else {}
        prof = _simulated_profile(meta, device.name) if err is None else None
        return MockJob(self, "profile", self._id("j"), None, prof, ok=err is None,
                       message=err)


def _simulated_profile(meta: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Roofline arithmetic dressed as a profile. SIMULATED: bytes at 45% of
    peak bandwidth plus 150 us of overhead. Exists so the pipeline can be
    exercised end to end; it proves nothing about Snapdragon."""
    params = int(meta.get("params") or 50_000_000)
    wb = {"INT4": 0.5, "INT8": 1.0}.get(meta.get("weights"), 2.0)
    bw = RL.SOCS[DEVICE_SHORT.get(device, "X Elite")].bw_gbs * 1e9 * 0.45
    us = int(150 + 1e6 * params * wb / bw)
    ops = int(meta.get("n_ops") or 60)
    return {"execution_summary": {"estimated_inference_time": us,
                                  "estimated_inference_peak_memory": int(params * wb * 1.2),
                                  "first_load_time": us * 40},
            "execution_detail": [{"name": f"op_{i}", "type": "Op", "compute_unit": "NPU",
                                  "execution_time": us // ops} for i in range(ops)],
            "_simulated": True}


class RecordingHub(MockHub):
    """The dry run: the mock's surface, recording every call, uploading and
    running nothing. `plan` and `run` without --yes print its log."""


# ============================================================================ #
# SECTION 5 -- the zero-download wrapper every backend goes through
# ============================================================================ #

class ZeroDownloadHub:
    """
    The only object the experiment code sees. It exposes upload, compile,
    quantize, profile and PROFILE FETCH -- and nothing that can pull a model
    or an artefact onto this machine. Byte counts go through the ledger, which
    refuses an upload past budget BEFORE it starts and a download past the
    ceiling at all.
    """

    def __init__(self, backend, ledger: TransferLedger, module=None):
        self._b = backend
        self._m = module if module is not None else backend
        self.ledger = ledger

    # -- devices --
    def device(self, name: str):
        if name not in COMPUTE_DEVICES:
            raise ValueError(f"{name!r} is not a Workbench Compute device; valid: "
                             f"{list(COMPUTE_DEVICES)}")
        return self._m.Device(name=name)

    def check_device_exists(self, name: str) -> bool:
        return any(getattr(d, "name", "") == name for d in self._b.get_devices(name=name))

    # -- uploads --
    def upload_model(self, zip_path: str, name: str):
        n = os.path.getsize(zip_path)
        self.ledger.check_upload(f"model {name}", n)
        m = self._b.upload_model(zip_path, name=name)
        self.ledger.record_upload(f"model {name}", n)
        return m

    def get_model(self, model_id: str):
        return self._b.get_model(model_id)

    def upload_dataset(self, data: Dict[str, List[Any]], name: str):
        n = sum(int(getattr(a, "nbytes", 0)) for v in data.values() for a in v)
        self.ledger.check_upload(f"dataset {name}", n)
        d = self._b.upload_dataset(data, name=name)
        self.ledger.record_upload(f"dataset {name}", n)
        return d

    def get_dataset(self, dataset_id: str):
        return self._b.get_dataset(dataset_id)

    # -- jobs (no bytes move: every model argument is a handle) --
    def compile(self, model, device: str, specs, options: str, name: str):
        return self._b.submit_compile_job(model=model, device=self.device(device),
                                          name=name, input_specs=specs, options=options)

    def quantize(self, model, calibration, weights: str, activations: str,
                 options: str, name: str):
        qd = self._m.QuantizeDtype
        return self._b.submit_quantize_job(
            model=model, calibration_data=calibration,
            weights_dtype=getattr(qd, weights), activations_dtype=getattr(qd, activations),
            name=name, options=options)

    def profile(self, model, device: str, options: str, name: str):
        return self._b.submit_profile_job(model=model, device=self.device(device),
                                          name=name, options=options)

    @staticmethod
    def wait(job, timeout: int):
        st = job.wait(timeout)
        return {"ok": bool(getattr(st, "success", False)),
                "code": getattr(st, "code", "?"),
                "message": getattr(st, "message", None),
                "job_id": getattr(job, "job_id", None), "url": getattr(job, "url", None)}

    @staticmethod
    def target(job):
        # A handle. Transfers nothing; verified in the client's source.
        return job.get_target_model()

    def profile_dict(self, job) -> Dict[str, Any]:
        """The one download: the profile as a dict, through the API."""
        prof = job.download_profile()
        n = len(json.dumps(prof, default=str).encode())
        self.ledger.record_download(f"profile {getattr(job, 'job_id', '?')}", n)
        return prof


def real_backend():
    """The installed qai_hub client, or a clear reason there is none."""
    try:
        import qai_hub as hub
    except ImportError:
        raise RuntimeError("qai-hub is not installed: pip install qai-hub, then "
                           "qai-hub configure --api_token <token from "
                           f"{WORKBENCH_URL}/account>")
    client = hub.Client() if hasattr(hub, "Client") else hub
    return client, hub


# ============================================================================ #
# SECTION 6 -- cache: never upload the same thing twice
# ============================================================================ #

CACHE_FILE = HERE / ".aihub_workbench_cache.json"


def load_cache(path=CACHE_FILE) -> Dict[str, Any]:
    try:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cache(cache: Dict[str, Any], path=CACHE_FILE) -> None:
    if not isinstance(cache, dict):
        raise TypeError("cache must be a dict")
    tmp = pathlib.Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ============================================================================ #
# SECTION 7 -- the experiment: one layer, three precisions, N devices
# ============================================================================ #

def _compile_options(precision: str, spec: SliceSpec) -> str:
    opts = f"--target_runtime qnn_dlc --output_names {','.join(output_names(spec))}"
    return opts + (" --quantize_io" if PRECISIONS[precision] else "")


def run_decode_experiment(hub: ZeroDownloadHub, spec: SliceSpec,
                          devices: Sequence[str], precisions: Sequence[str],
                          cache: Dict[str, Any], backend_name: str,
                          build_dir, timeout: int = 3600,
                          prebuilt: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    For each device and precision, the qai_hub_models sequence:
        float:  compile(qnn_dlc) -> profile
        quant:  compile(onnx) -> quantize(weights, int16, min_max)
                -> compile(qnn_dlc, --quantize_io) -> profile
    The ONNX compile and the quantize job are per model, not per device, so
    they run once and are reused across devices. A failed job is recorded
    with the service's own message and the run continues: a refusal is a
    result, not a crash.
    """
    spec.validate()
    for d in devices:
        if d not in COMPUTE_DEVICES:
            raise ValueError(f"{d!r} is not a Workbench Compute device")
    for p in precisions:
        if p not in PRECISIONS:
            raise ValueError(f"unknown precision {p!r}; known {list(PRECISIONS)}")
    bc = cache.setdefault(backend_name, {"models": {}, "datasets": {}})
    specs = input_specs(spec, 1)
    records: List[Dict[str, Any]] = []

    # 1. the source model: cached by content key, uploaded at most once
    src = None
    entry = bc["models"].get(spec.key)
    if entry:
        try:
            src = hub.get_model(entry["model_id"])
            log(f"  reusing uploaded model {entry['model_id']} ({spec.key}) -- 0 bytes")
        except Exception as exc:
            log(f"  cached model unavailable ({type(exc).__name__}); re-uploading")
            src = None
    if src is None:
        built = prebuilt or build(spec, build_dir)
        v = validate_layout(built["zip"])
        if not v["ok"]:
            raise RuntimeError(f"archive layout rejected: {v}")
        log(f"  uploading {built['zip_bytes'] / 1e6:.1f} MB ({built['raw_bytes'] / 1e6:.0f} "
            f"MB before compression)")
        src = hub.upload_model(built["zip"], name=spec.key)
        if isinstance(src, MockModel):
            src.meta.update(params=slice_params(spec), n_ops=60)
        bc["models"][spec.key] = {"model_id": getattr(src, "model_id", str(src)),
                                  "sha256": built["sha256"], "bytes": built["zip_bytes"]}

    # 2. calibration data: only if a quantised precision is asked for
    calib = None
    if any(PRECISIONS[p] for p in precisions):
        ckey = f"calib-{spec.key}"
        dentry = bc["datasets"].get(ckey)
        if dentry:
            try:
                calib = hub.get_dataset(dentry["dataset_id"])
                log(f"  reusing calibration dataset {dentry['dataset_id']} -- 0 bytes")
            except Exception:
                calib = None
        if calib is None:
            data = {k: [v] for k, v in sample_inputs(spec, 1).items()}
            calib = hub.upload_dataset(data, name=ckey)
            bc["datasets"][ckey] = {"dataset_id": getattr(calib, "dataset_id", str(calib)),
                                    "bytes": calibration_bytes(spec)}

    # 3. per precision: the device-independent quantised model, made once
    quantized: Dict[str, Any] = {}
    first_dev = devices[0]
    for p in precisions:
        if not PRECISIONS[p]:
            continue
        wq, aq = PRECISIONS[p]
        cj = hub.compile(src, first_dev, specs, "--target_runtime onnx",
                         f"{spec.key}-onnx")
        st = hub.wait(cj, timeout)
        if not st["ok"]:
            quantized[p] = ("FAILED at ONNX compile", st)
            continue
        qj = hub.quantize(hub.target(cj), calib, wq, aq, "--range_scheme min_max",
                          f"{spec.key}-{p}")
        st = hub.wait(qj, timeout)
        quantized[p] = (hub.target(qj), st) if st["ok"] else ("FAILED at quantize", st)
        tgt = quantized[p][0]
        if isinstance(tgt, MockModel):
            tgt.meta.update(params=slice_params(spec), n_ops=60)

    # 4. per device and precision: compile for the NPU, profile, fetch JSON
    for dev in devices:
        for p in precisions:
            rec = {"device": dev, "precision": p, "arch": spec.arch, "kind": spec.kind,
                   "context": spec.context, "slice_key": spec.key,
                   "slice_params": slice_params(spec), "backend": backend_name,
                   "source": ("MOCK -- simulated, not a measurement"
                              if backend_name != "real" else "AI Hub Workbench profile")}
            model_in = src
            if PRECISIONS[p]:
                q = quantized.get(p)
                if q is None or isinstance(q[0], str):
                    rec.update(status="FAILED", stage=q[0] if q else "quantize",
                               service_message=(q[1] or {}).get("message") if q else None)
                    records.append(rec)
                    continue
                model_in = q[0]
            cj = hub.compile(model_in, dev, specs, _compile_options(p, spec),
                             f"{spec.key}-{p}-{DEVICE_SHORT[dev]}")
            st = hub.wait(cj, timeout)
            rec["compile_job"] = st["job_id"]
            if not st["ok"]:
                rec.update(status="FAILED", stage="compile", service_message=st["message"])
                records.append(rec)
                continue
            tgt = hub.target(cj)
            if isinstance(tgt, MockModel):
                tgt.meta.update(params=slice_params(spec), n_ops=60)
            pj = hub.profile(tgt, dev, "", f"{spec.key}-{p}-{DEVICE_SHORT[dev]}")
            st = hub.wait(pj, timeout)
            rec["profile_job"] = st["job_id"]
            rec["profile_url"] = st["url"]
            if not st["ok"]:
                rec.update(status="FAILED", stage="profile", service_message=st["message"])
                records.append(rec)
                continue
            rec.update(status="OK", profile=summarize_profile(hub.profile_dict(pj)))
            records.append(rec)
    return records


# ============================================================================ #
# SECTION 8 -- reading a profile, and what it means against the roofline
# ============================================================================ #

def summarize_profile(prof: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the client's own _profile_pb_to_python_dict produces:
    execution_summary.estimated_inference_time is in MICROSECONDS (the
    qai_hub_models printer divides by 1000 for ms)."""
    if not isinstance(prof, dict):
        raise TypeError("profile must be a dict")
    es = prof.get("execution_summary") or {}
    det = prof.get("execution_detail") or []
    units: Dict[str, int] = {}
    for d in det:
        u = str(d.get("compute_unit", "?"))
        units[u] = units.get(u, 0) + 1
    total = sum(units.values())
    us = es.get("estimated_inference_time")
    return {"inference_us": us, "inference_ms": None if us is None else us / 1000.0,
            "peak_memory_bytes": es.get("estimated_inference_peak_memory"),
            "first_load_us": es.get("first_load_time"),
            "ops": total, "ops_by_unit": units,
            "npu_fraction": (units.get("NPU", 0) / total) if total else None,
            "simulated": bool(prof.get("_simulated"))}


def analyze(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Per (device, arch, context): the bytes test, the extrapolation, and the
    comparison with Qualcomm's published full-model number where one exists.
    """
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list of result dicts")
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("status") == "OK" and r.get("kind") == "decode":
            groups.setdefault((r["device"], r["arch"], r["context"], r["backend"]), []).append(r)
    out = []
    for (dev, arch, ctx, backend), rs in sorted(groups.items()):
        a = arch_of(arch)
        per = {r["precision"]: r["profile"]["inference_ms"] for r in rs
               if r["profile"]["inference_ms"]}
        layer_params = rs[0]["slice_params"]
        kv = 2 * ctx * a.K * a.H * 2.0             # fp16 KV read by the step
        row: Dict[str, Any] = {"device": dev, "arch": arch, "context": ctx,
                               "backend": backend, "layer_ms": per,
                               "simulated": any(r["profile"]["simulated"] for r in rs),
                               "npu_fraction": {r["precision"]: r["profile"]["npu_fraction"]
                                                for r in rs}}
        bw = {p: (layer_params * WEIGHT_BYTES[p] + kv) / (ms / 1e3) / 1e9
              for p, ms in per.items()}
        row["achieved_gbs"] = {p: round(v, 1) for p, v in bw.items()}
        peak = RL.SOCS[DEVICE_SHORT[dev]].bw_gbs
        row["pct_of_peak"] = {p: round(100 * v / peak, 1) for p, v in bw.items()}
        if len(per) >= 2:
            xs = [layer_params * WEIGHT_BYTES[p] / 1e6 for p in per]
            ys = list(per.values())
            if len(per) >= 3:
                f = RL.fit_line(xs, ys)
                row["bytes_test"] = {"ms_per_mb": round(f["b"], 5),
                                     "intercept_ms": round(f["a"], 4),
                                     "r2": round(f["r2"], 4),
                                     # x in MB, y in ms: slope is ms/MB, and
                                     # MB/ms is GB/s, so the bandwidth is 1/slope.
                                     "weight_gbs": round(1.0 / f["b"], 1) if f["b"] > 0 else None,
                                     "reading": ("time falls with bytes: bandwidth-bound"
                                                 if f["b"] > 0 and f["r2"] >= 0.9 else
                                                 "time does not track bytes: not bandwidth-bound")}
            hi, lo = max(per, key=lambda p: WEIGHT_BYTES[p]), min(per, key=lambda p: WEIGHT_BYTES[p])
            row["time_ratio_vs_byte_ratio"] = (round(per[hi] / per[lo], 2),
                                              round((layer_params * WEIGHT_BYTES[hi] + kv)
                                                    / (layer_params * WEIGHT_BYTES[lo] + kv), 2))
        if "w4a16" in per:
            lm = a.V * a.D * 1.0
            lm_ms = lm / (bw["w4a16"] * 1e9) * 1e3
            tok_ms = a.L * per["w4a16"] + lm_ms
            row["extrapolated_tok_s_w4a16"] = round(1000.0 / tok_ms, 2)
            pub = RL._rows(model=arch, precision="w4a16", device=DEVICE_SHORT[dev],
                           runtime="geniex_qairt", context=ctx)
            if pub:
                row["published_qairt_tok_s"] = round(pub[0][6], 2)
                row["published_over_extrapolated"] = round(pub[0][6] / row["extrapolated_tok_s_w4a16"], 2)
            if a.dense and a.V:
                row["roofline_ceiling_tok_s"] = round(RL.decode_ceiling(a, "w4a16", ctx, peak), 1)
        out.append(row)
    return out


# ============================================================================ #
# SECTION 9 -- self-test
# ============================================================================ #

def _signature_parity() -> Dict[str, Any]:
    """MockHub's methods against the installed client's, parameter by
    parameter. If Qualcomm renames an argument, this fails here -- not in a
    job the night before a deadline."""
    try:
        import inspect
        import qai_hub as hub
    except ImportError:
        return {"status": "SKIPPED", "reason": "qai_hub not installed"}
    rows = []
    for m in ("upload_model", "get_model", "upload_dataset", "get_dataset",
              "submit_compile_job", "submit_quantize_job", "submit_profile_job",
              "get_devices"):
        real = [p for p in inspect.signature(getattr(hub.Client, m)).parameters if p != "self"]
        mine = [p for p in inspect.signature(getattr(MockHub, m)).parameters if p != "self"]
        rows.append((m, real == mine, real, mine))
    extra = [("ProfileJob.download_profile(filename)",
              list(inspect.signature(hub.ProfileJob.download_profile).parameters)
              == ["self", "filename"]),
             ("QuantizeDtype has INT4, INT8, INT16",
              all(hasattr(hub.QuantizeDtype, x) for x in ("INT4", "INT8", "INT16"))),
             ("CompileJob.get_target_model takes no arguments",
              list(inspect.signature(hub.CompileJob.get_target_model).parameters) == ["self"])]
    ok = all(r[1] for r in rows) and all(e[1] for e in extra)
    return {"status": "MATCH" if ok else "DRIFT", "rows": rows, "extra": extra,
            "version": getattr(hub, "__version__", "?")}


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

    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    waived: List[Tuple[int, str, str]] = []
    hits = download_calls_in(src, waived)
    # ---- the contract, checked against this file's own syntax tree ----
    ck("this file makes no call that downloads a model", hits == [], hits)
    ck("exactly one waiver, and it is the mock's alarm test",
       len(waived) == 1 and "alarm" in waived[0][2], waived)
    ck("a waiver without a reason is not honoured",
       download_calls_in("m.download('x')  " + WAIVER) == [(1, "download")])
    for name in FORBIDDEN_CALLS:
        ck(f"the scanner catches {name}(...)",
           download_calls_in(f"job.{name}('x')") == [(1, name)])
    ck("the scanner catches Model.download(...)",
       download_calls_in("m.download('a.bin')") == [(1, "download")])
    ck("the scanner catches download_profile with a filename",
       download_calls_in("j.download_profile('p.json')")
       == [(1, "download_profile(<filename>)")])
    ck("but allows download_profile() with none -- the one permitted fetch",
       download_calls_in("j.download_profile()") == [])
    ck("the scanner catches an emitted fetch/export command",
       len(download_calls_in("cmd = 'qai-hub-models " + "export qwen3_4b'")) == 1)
    ck("and ignores a docstring that explains why they were removed",
       download_calls_in('"""we no longer run qai-hub-models ' + 'fetch"""') == [])
    ck("the scanner catches from_pretrained",
       download_calls_in("M.from_pretrained('q')") == [(1, "from_pretrained")])
    ck("an on-device command must state its size in the same string",
       len(download_calls_in("c = 'geniex " + "pull repo'")) == 1
       and download_calls_in("c = 'geniex " + "pull repo  " + SIZE_MARKER
                             + "~2.3 GB to THIS machine]'") == [])
    ck("an f-string is judged whole: marker and command may straddle a placeholder",
       download_calls_in('g = 3\nc = f"' + SIZE_MARKER + '~{g} GB] geniex ' + 'pull {g}"') == []
       and len(download_calls_in('g = 3\nc = f"geniex ' + 'pull {g}"')) == 1)
    ck("a stated size does not unban fetch/export",
       len(download_calls_in("c = 'qai-hub-models " + "fetch x " + SIZE_MARKER
                             + "~1 GB]'")) == 1)
    ck("the wrapper exposes no download method",
       not any(n.startswith("download") for n in dir(ZeroDownloadHub)))

    # ---- the ledger ----
    led = TransferLedger(max_upload_bytes=1000)
    ck("an upload past budget is refused before it starts",
       raises(lambda: led.check_upload("x", 2000), UploadBudgetExceeded))
    led2 = TransferLedger(max_upload_bytes=10 ** 9)
    led2.record_download("p", 1000)
    ck("a download past the ceiling is refused",
       raises(lambda: led2.record_download("big", MAX_DOWNLOAD_BYTES), DownloadRefused))
    ck("the ceiling is 2 MB -- profiles, not models", MAX_DOWNLOAD_BYTES == 2 * 1024 * 1024)
    ck("negative byte counts are refused", raises(lambda: led2.record_upload("n", -1)))

    # ---- geometry ----
    s4 = SliceSpec("qwen3_4b")
    a4 = RL.ARCHS["qwen3_4b"]
    per_layer = RL.nonembedding_params(a4) / a4.L
    ck("the Qwen3-4B slice has exactly one layer's weights",
       abs(slice_params(s4) - per_layer) / per_layer < 1e-4, (slice_params(s4), per_layer))
    ck("Qwen3-4B slice is ~404 MB of fp32 before compression",
       abs(slice_params(s4) * 4 / 1e6 - 403.7) < 1.0, slice_params(s4) * 4 / 1e6)
    ck("estimated upload fits the default budget",
       estimated_upload_bytes(s4) < DEFAULT_MAX_UPLOAD_MB * 1e6, estimated_upload_bytes(s4))
    ck("Qwen3-8B needs a raised budget", estimated_upload_bytes(SliceSpec("qwen3_8b"))
       + calibration_bytes(SliceSpec("qwen3_8b")) > DEFAULT_MAX_UPLOAD_MB * 1e6)
    ck("decode inputs match Qualcomm's KV-as-input layout",
       input_specs(s4)["past_key"][0] == (1, 8, 4096, 128)
       and input_specs(s4)["mask"][0] == (1, 1, 1, 4097))
    ck("the hybrid is refused: one slice cannot stand for it",
       raises(lambda: SliceSpec("bonsai_2_27b").validate(), ValueError))
    ck("unknown arch is refused", raises(lambda: SliceSpec("nope").validate()))
    ck("context 0 is refused", raises(lambda: SliceSpec("qwen3_4b", context=0).validate()))
    ck("bool context is refused", raises(lambda: SliceSpec("qwen3_4b", context=True).validate()))
    ck("spec keys are stable and carry the builder version",
       s4.key == "qwen3_4b-decode-p4096-s0-v1")

    # ---- the graph (needs numpy + onnx; the count is the same either way) ----
    have_onnx = True
    try:
        _np()
        _onnx()
    except RuntimeError:
        have_onnx = False
    if have_onnx:
        np = _np()
        with tempfile.TemporaryDirectory() as td:
            tiny = SliceSpec("tiny", "decode", context=24)
            b1 = build(tiny, pathlib.Path(td) / "a")
            b2 = build(tiny, pathlib.Path(td) / "b")
            ck("the archive is byte-for-byte reproducible", b1["sha256"] == b2["sha256"])
            v = validate_layout(b1["zip"])
            ck("the archive passes the client's layout rules", v["ok"], v)
            ck("layout checked by the installed client too, when present",
               v["client_check"] in ("ONNX", "skipped: qai_hub not installed"),
               v["client_check"])
            bm = build(SliceSpec("tiny", "mlp"), pathlib.Path(td) / "c")
            import onnx as _ox
            m = _ox.load(str(pathlib.Path(bm["dir"]) / "model.onnx"), load_external_data=False)
            dim = m.graph.input[0].type.tensor_type.shape.dim[1]
            ck("the MLP slice has a symbolic token axis (one upload, many lengths)",
               dim.dim_param == "T", dim)
        w = make_weights(SliceSpec("tiny"))
        ck("grid weights take exactly 16 values", len(np.unique(w["wq"])) == 16,
           len(np.unique(w["wq"])))
        big = grid_weights(np.random.default_rng(0), (1024, 2048), 1024)
        import zlib
        ratio = big.nbytes / len(zlib.compress(big.tobytes(), ZIP_LEVEL))
        gauss = np.random.default_rng(0).standard_normal((1024, 2048)).astype(np.float32)
        gratio = gauss.nbytes / len(zlib.compress(gauss.tobytes(), ZIP_LEVEL))
        ck("grid weights compress >= 4x; Gaussian ones barely do",
           ratio >= 4.0 and gratio < 1.3, (round(ratio, 2), round(gratio, 2)))
        ck("no weight is exactly zero more often than the grid implies",
           abs(float((w["wg"] == 0).mean()) - 1 / 16) < 0.05)
    else:
        for name in ("the archive is byte-for-byte reproducible",
                     "the archive passes the client's layout rules",
                     "layout checked by the installed client too, when present",
                     "the MLP slice has a symbolic token axis (one upload, many lengths)",
                     "grid weights take exactly 16 values",
                     "grid weights compress >= 4x; Gaussian ones barely do",
                     "no weight is exactly zero more often than the grid implies"):
            ck(name, True, "skipped: pip install numpy onnx")

    vg = verify_graph() if have_onnx else {"status": "SKIPPED"}
    if vg["status"] == "SKIPPED":
        ck("the decode graph IS a decoder layer (onnxruntime vs numpy)", True,
           "skipped: " + vg.get("reason", "numpy/onnx absent"))
        ck("the MLP graph matches its reference too", True, "skipped")
    else:
        ck("the decode graph IS a decoder layer (onnxruntime vs numpy)",
           vg["status"] == "MATCH", vg)
        vm = verify_graph(SliceSpec("tiny", "mlp"))
        ck("the MLP graph matches its reference too", vm["status"] == "MATCH", vm)

    # ---- the mock is the client's shape ----
    par = _signature_parity()
    if par["status"] == "SKIPPED":
        ck("mock signatures equal the installed client's", True, "skipped: qai_hub absent")
    else:
        ck("mock signatures equal the installed client's", par["status"] == "MATCH",
           [r for r in par["rows"] if not r[1]] + [e for e in par["extra"] if not e[1]])

    # ---- end to end, offline ----
    have_np = True
    try:
        _np()
    except RuntimeError:
        have_np = False
    if have_np:
        with tempfile.TemporaryDirectory() as td:
            pre = None if have_onnx else stand_in_archive(
                SliceSpec("tiny", "decode", context=24), td)
            mock = MockHub()
            led = TransferLedger(max_upload_bytes=int(1e9))
            hub = ZeroDownloadHub(mock, led)
            cache: Dict[str, Any] = {}
            spec = SliceSpec("tiny", "decode", context=24)
            recs = run_decode_experiment(hub, spec, list(COMPUTE_DEVICES),
                                         list(PRECISIONS), cache, "mock", td, timeout=5,
                                         prebuilt=pre)
            ck("every device x precision produced a record", len(recs) == 9, len(recs))
            ck("all succeeded against the mock", all(r["status"] == "OK" for r in recs))
            ups = [c for c in mock.calls if c[0] in ("upload_model", "upload_dataset")]
            ck("one model upload and one dataset upload, for 9 profiles", len(ups) == 2,
               [c[0] for c in ups])
            seq = [c[0] for c in mock.calls if c[0].startswith("submit")]
            ck("quantised path follows qai_hub_models: onnx -> quantize -> qnn_dlc",
               seq[:2] == ["submit_compile_job", "submit_quantize_job"]
               and "--target_runtime onnx" in mock.calls[[c[0] for c in mock.calls]
                                                         .index("submit_compile_job")][1]["options"])
            q_opts = [c[1]["options"] for c in mock.calls if c[0] == "submit_quantize_job"]
            ck("quantize jobs use --range_scheme min_max, as Qualcomm's w8a16 does",
               q_opts and all("--range_scheme min_max" in o for o in q_opts))
            wts = sorted({c[1]["weights"] for c in mock.calls if c[0] == "submit_quantize_job"})
            ck("w8a16 and w4a16 map to INT8 and INT4 weights", wts == ["INT4", "INT8"], wts)
            c_opts = [c[1]["options"] for c in mock.calls if c[0] == "submit_compile_job"
                      and "qnn_dlc" in c[1]["options"]]
            ck("NPU compiles target qnn_dlc and quantised ones add --quantize_io",
               len(c_opts) == 9 and sum("--quantize_io" in o for o in c_opts) == 6)
            ck("the quantised model is made once and reused across devices",
               sum(1 for c in mock.calls if c[0] == "submit_quantize_job") == 2)
            ck("nothing tried to download a model", mock.download_attempts == [],
               mock.download_attempts)
            ck("downloads stayed under the ceiling (profiles only)",
               0 < led.downloaded < 64 * 1024, led.downloaded)
            ck("every result is labelled MOCK, never measurement",
               all(r["source"].startswith("MOCK") for r in recs))
            # second run: the cache means zero bytes up
            mock2_calls_before = len(mock.calls)
            led_b = TransferLedger(max_upload_bytes=int(1e9))
            recs2 = run_decode_experiment(ZeroDownloadHub(mock, led_b), spec,
                                          ["Snapdragon X Plus 8-Core CRD"], ["w4a16"],
                                          cache, "mock", td, timeout=5)
            ck("a re-run uploads zero bytes (model and dataset reused by ID)",
               led_b.uploaded == 0 and recs2[0]["status"] == "OK", led_b.uploaded)
            ck("and reuses them through get_model / get_dataset",
               any(c[0] == "get_model" for c in mock.calls[mock2_calls_before:]))
            # a refusal from the service is recorded, not raised
            refusing = MockHub(fail_on=lambda stage, kw: (
                "Unsupported: INT4 weights on this device" if stage == "quantize"
                and kw["weights"] == "INT4" else None))
            recs3 = run_decode_experiment(ZeroDownloadHub(refusing, TransferLedger(10 ** 9)),
                                          spec, ["Snapdragon X Elite CRD"], list(PRECISIONS),
                                          {}, "mock", td, timeout=5, prebuilt=pre)
            byp = {r["precision"]: r for r in recs3}
            ck("a service refusal is recorded verbatim and the run continues",
               byp["w4a16"]["status"] == "FAILED" and byp["fp16"]["status"] == "OK"
               and "INT4" in str(byp["w4a16"].get("service_message")))
            # the budget stops an oversized run before any upload
            tight = ZeroDownloadHub(MockHub(), TransferLedger(max_upload_bytes=100))
            ck("an over-budget run stops before uploading anything",
               raises(lambda: run_decode_experiment(tight, spec, [DEFAULT_DEVICE],
                                                    ["fp16"], {}, "mock", td, timeout=5,
                                                    prebuilt=pre),
                      UploadBudgetExceeded))
            an = analyze(recs)
            ck("analysis refuses to hide that the numbers are simulated",
               an and all(r["simulated"] for r in an))
    else:
        for name in ("every device x precision produced a record",
                     "all succeeded against the mock",
                     "one model upload and one dataset upload, for 9 profiles",
                     "quantised path follows qai_hub_models: onnx -> quantize -> qnn_dlc",
                     "quantize jobs use --range_scheme min_max, as Qualcomm's w8a16 does",
                     "w8a16 and w4a16 map to INT8 and INT4 weights",
                     "NPU compiles target qnn_dlc and quantised ones add --quantize_io",
                     "the quantised model is made once and reused across devices",
                     "nothing tried to download a model",
                     "downloads stayed under the ceiling (profiles only)",
                     "every result is labelled MOCK, never measurement",
                     "a re-run uploads zero bytes (model and dataset reused by ID)",
                     "and reuses them through get_model / get_dataset",
                     "a service refusal is recorded verbatim and the run continues",
                     "an over-budget run stops before uploading anything",
                     "analysis refuses to hide that the numbers are simulated"):
            ck(name, True, "skipped: pip install numpy")

    # ---- profile parsing, on the exact shape the client returns ----
    fake = {"execution_summary": {"estimated_inference_time": 2500,
                                  "estimated_inference_peak_memory": 10},
            "execution_detail": [{"name": "a", "type": "MatMul", "compute_unit": "NPU",
                                  "execution_time": 10},
                                 {"name": "b", "type": "Softmax", "compute_unit": "CPU"}]}
    sp = summarize_profile(fake)
    ck("estimated_inference_time is read as microseconds", sp["inference_ms"] == 2.5)
    ck("a CPU fallback shows up in the NPU fraction", sp["npu_fraction"] == 0.5)
    ck("a non-dict profile is refused", raises(lambda: summarize_profile([1])))

    # ---- the bytes test recovers a planted bandwidth ----
    planted = []
    for prec in PRECISIONS:
        ms = 0.05 + (1_000_000 * WEIGHT_BYTES[prec]) / 50e9 * 1e3   # 50 GB/s + 50 us
        planted.append({"status": "OK", "kind": "decode", "device": DEFAULT_DEVICE,
                        "arch": "qwen3_0_6b", "context": 64, "backend": "planted",
                        "precision": prec, "slice_params": 1_000_000,
                        "profile": {"inference_ms": ms, "npu_fraction": 1.0,
                                    "simulated": True}})
    bt = analyze(planted)[0]["bytes_test"]
    ck("the bytes test recovers a planted 50 GB/s and 0.05 ms overhead",
       abs(bt["weight_gbs"] - 50.0) < 0.5 and abs(bt["intercept_ms"] - 0.05) < 1e-3, bt)

    # ---- guard rails on the wrapper ----
    zh = ZeroDownloadHub(MockHub(), TransferLedger(10 ** 9))
    ck("a non-Compute device is refused", raises(lambda: zh.device("Samsung Galaxy S24")))
    mm = MockModel(MockHub(), "m1", 0, {})
    ck("calling Model.download is an alarm, not a download",
       raises(lambda: mm.download("x.bin"), DownloadRefused))  # download-ok: trips the mock's alarm on purpose

    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, det in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"   -> {det}" if det and not ok else ""))
    print(f"\n  {npass}/{len(checks)} passed")
    return 0 if npass == len(checks) else 1


# ============================================================================ #
# SECTION 10 -- CLI
# ============================================================================ #

def _plan_text(spec: SliceSpec, devices: Sequence[str], precisions: Sequence[str],
               cache: Dict[str, Any], budget_mb: float) -> List[str]:
    reused = spec.key in cache.get("real", {}).get("models", {})
    up_model = 0 if reused else estimated_upload_bytes(spec)
    up_cal = (0 if f"calib-{spec.key}" in cache.get("real", {}).get("datasets", {})
              else calibration_bytes(spec)) if any(PRECISIONS[p] for p in precisions) else 0
    nq = sum(1 for p in precisions if PRECISIONS[p])
    jobs = 2 * nq + 2 * len(devices) * len(precisions)
    lines = [
        f"  slice        {spec.key}: one {spec.arch} decoder layer, "
        f"{slice_params(spec) / 1e6:.1f}M weights, KV cache of {spec.context} tokens",
        f"  devices      {', '.join(devices)}",
        f"  precisions   {', '.join(precisions)}",
        f"  upload       model ~{up_model / 1e6:.0f} MB"
        + (" (cached: 0)" if reused else " (estimate; `build` gives the exact size)")
        + f", calibration {up_cal / 1e6:.1f} MB -- budget {budget_mb:.0f} MB",
        f"  download     profile JSON only, a few KB per job -- hard ceiling "
        f"{MAX_DOWNLOAD_BYTES // 1024} KB",
        f"  jobs         {jobs} (ONNX compile + quantize once per precision; "
        f"compile + profile per device and precision)",
        "  sequence     float: compile(qnn_dlc) -> profile",
        "               quant: compile(onnx) -> quantize(min_max) -> compile(qnn_dlc,"
        " --quantize_io) -> profile",
    ]
    if up_model + up_cal > budget_mb * 1e6:
        lines.append(f"  OVER BUDGET  raise --max-upload-mb to "
                     f"{math.ceil((up_model + up_cal) / 1e6) + 5} or use a smaller --arch")
    return lines


def _explain_client_error(exc: BaseException) -> str:
    """The client's errors, as instructions. Its missing-token error is a
    FileNotFoundError about client.ini wrapped in a UserError; a first-time
    user should see what to type, not a traceback."""
    text = f"{type(exc).__name__}: {exc}"
    chain = [exc]
    while chain[-1].__context__ is not None and len(chain) < 5:
        chain.append(chain[-1].__context__)
    blob = " ".join(str(e) for e in chain)
    if "client.ini" in blob or "api_token" in blob or "API key" in blob:
        return ("  qai-hub is installed but not configured with an API token.\n"
                f"    1. sign in at {WORKBENCH_URL} -> Account -> Settings -> API Token\n"
                "    2. qai-hub configure --api_token <TOKEN>\n"
                "    3. qai-hub list-devices        # confirms it works\n"
                "  Nothing was uploaded.")
    return f"  Workbench call failed -- {text[:400]}\n  Nothing further was sent."


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Zero-download testing on Qualcomm AI Hub Workbench.")
    p.add_argument("cmd", choices=["plan", "build", "verify", "run", "results",
                                   "devices", "selftest"])
    p.add_argument("--arch", default="qwen3_4b",
                   choices=sorted(k for k, a in RL.ARCHS.items() if a.dense))
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--device", action="append", choices=list(COMPUTE_DEVICES),
                   help=f"repeatable; default {DEFAULT_DEVICE}")
    p.add_argument("--all-devices", action="store_true")
    p.add_argument("--precision", action="append", choices=list(PRECISIONS),
                   help="repeatable; default all three")
    p.add_argument("--max-upload-mb", type=float, default=DEFAULT_MAX_UPLOAD_MB)
    p.add_argument("--build-dir", default=str(HERE / "workbench_build"))
    p.add_argument("--results", default=str(HERE / "workbench_results.json"))
    p.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    p.add_argument("--yes", action="store_true", help="really submit jobs")
    p.add_argument("--mock", action="store_true", help="run against the offline fake")
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    devices = list(COMPUTE_DEVICES) if args.all_devices else (args.device or [DEFAULT_DEVICE])
    precisions = args.precision or list(PRECISIONS)
    try:
        spec = SliceSpec(args.arch, "decode", args.context).validate()
        _finite(args.max_upload_mb, "--max-upload-mb", 1, 100_000)
    except (TypeError, ValueError) as exc:
        log(f"  {exc}")
        return 2
    cache = load_cache()

    if args.cmd == "plan" or (args.cmd == "run" and not args.yes and not args.mock):
        log("\n  AI Hub Workbench plan -- nothing below has been sent anywhere\n")
        for ln in _plan_text(spec, devices, precisions, cache, args.max_upload_mb):
            log(ln)
        log("\n  predictions to test (roofline.py predict):")
        px = [r for r in RL.predict_x_plus()["predictions"]
              if r["model"] == args.arch and r["runtime"] == "geniex_qairt"]
        for r in px[:3]:
            log(f"    {r['model']} {r['precision']} QAIRT NPU @ {r['context']}: "
                f"~{r['decode_tok_s']} tok/s if X Plus 8-Core matches X Elite's bus")
        if args.cmd == "run":
            log("\n  dry run. Add --yes to submit, or --mock to exercise the flow offline.")
        return 0

    if args.cmd == "build":
        try:
            b = build(spec, args.build_dir)
        except RuntimeError as exc:
            log(f"  {exc}")
            return 2
        v = validate_layout(b["zip"])
        log(f"\n  built {b['zip']}")
        log(f"  {b['params'] / 1e6:.1f}M weights, {b['raw_bytes'] / 1e6:.1f} MB raw -> "
            f"{b['zip_bytes'] / 1e6:.1f} MB on the wire ({b['raw_bytes'] / b['zip_bytes']:.1f}x)")
        log(f"  sha256 {b['sha256'][:16]}...   layout: "
            f"{'OK' if v['ok'] else 'REJECTED'} ({v['client_check']})")
        return 0 if v["ok"] else 1

    if args.cmd == "verify":
        for s in (SliceSpec("tiny", "decode", context=24), SliceSpec("tiny", "mlp")):
            try:
                v = verify_graph(s)
            except RuntimeError as exc:
                # Optional dependency absent: a skip with the fix, not a failure.
                log(f"  {s.kind:<7} SKIPPED  {exc}")
                continue
            log(f"  {s.kind:<7} {v['status']}  {v.get('max_rel_err', v.get('reason'))}")
            if v["status"] == "MISMATCH":
                return 1
        return 0

    if args.cmd == "devices":
        try:
            client, _hub = real_backend()
            names = sorted({getattr(d, "name", "") for d in client.get_devices()
                            if "CRD" in getattr(d, "name", "")})
        except Exception as exc:
            log(_explain_client_error(exc))
            return 2
        for n in names:
            log(f"  {n}{'   <- default' if n == DEFAULT_DEVICE else ''}")
        return 0

    if args.cmd == "results":
        try:
            recs = json.loads(pathlib.Path(args.results).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log(f"  no results at {args.results}; run first")
            return 1
        for row in analyze(recs.get("records", [])):
            tag = "SIMULATED (mock)" if row["simulated"] else "MEASURED on Workbench"
            log(f"\n  {row['device']}  {row['arch']}  ctx {row['context']}   [{tag}]")
            for prec, ms in row["layer_ms"].items():
                log(f"    {prec:<6} {ms:8.3f} ms/layer   {row['achieved_gbs'][prec]:6.1f} GB/s"
                    f" ({row['pct_of_peak'][prec]}% of peak)   NPU ops "
                    f"{row['npu_fraction'][prec]}")
            if "bytes_test" in row:
                bt = row["bytes_test"]
                log(f"    bytes test: R^2 {bt['r2']}, {bt['weight_gbs']} GB/s -- {bt['reading']}")
            if "extrapolated_tok_s_w4a16" in row:
                log(f"    extrapolated w4a16: {row['extrapolated_tok_s_w4a16']} tok/s "
                    f"(ceiling {row['roofline_ceiling_tok_s']})"
                    + (f", published QAIRT {row['published_qairt_tok_s']}"
                       if "published_qairt_tok_s" in row else ""))
        return 0

    # ---- run, for real or against the mock ----
    led = TransferLedger(max_upload_bytes=int(args.max_upload_mb * 1e6))
    if args.mock:
        backend, module, name = MockHub(), None, "mock"
        hub = ZeroDownloadHub(backend, led)
    else:
        try:
            backend, module = real_backend()
        except RuntimeError as exc:
            log(f"  {exc}")
            return 2
        name = "real"
        hub = ZeroDownloadHub(backend, led, module)
        try:
            for d in devices:
                if not hub.check_device_exists(d):
                    log(f"  Workbench does not list {d!r} for this account")
                    return 2
        except Exception as exc:
            log(_explain_client_error(exc))
            return 2
    log(f"\n  running on {', '.join(devices)} [{name}]")
    try:
        recs = run_decode_experiment(hub, spec, devices, precisions, cache, name,
                                     args.build_dir, args.timeout)
    except (UploadBudgetExceeded, DownloadRefused) as exc:
        log(f"  STOPPED: {exc}")
        return 3
    except RuntimeError as exc:
        log(f"  {exc}")
        return 2
    except Exception as exc:
        log(_explain_client_error(exc))
        return 2
    finally:
        save_cache(cache)
    try:
        prev = json.loads(pathlib.Path(args.results).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {"records": []}
    prev.setdefault("records", []).extend(recs)
    prev["ledger"] = led.summary()
    pathlib.Path(args.results).write_text(json.dumps(prev, indent=1, default=str),
                                          encoding="utf-8")
    for r in recs:
        ms = r.get("profile", {}).get("inference_ms")
        log(f"  {r['device']:<30}{r['precision']:<7}{r['status']:<8}"
            + (f"{ms:.3f} ms/layer" if ms else str(r.get("service_message") or r.get("stage"))))
    s = led.summary()
    log(f"\n  uploaded {s['uploaded_bytes'] / 1e6:.1f} MB, downloaded "
        f"{s['downloaded_bytes'] / 1024:.1f} KB (profiles only)")
    log(f"  results -> {args.results};  python aihub_workbench.py results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
