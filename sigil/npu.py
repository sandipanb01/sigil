"""
sigil.npu
=========
The deployment half. Everything here exists to guarantee one property:

    SIGIL adds ZERO operators and ZERO runtime FLOPs to the deployed graph.

That is not a nice-to-have on Hexagon. The QAIRT/Genie path compiles a static
graph with fixed shapes; a novel per-token operator would mean a custom kernel,
which means it does not run on the NPU at all and silently falls back to CPU.
SIGIL sidesteps this by folding every learned rotation into weights that already
exist.

Folding identities
------------------
For an orthogonal R (R R^T = I) acting on the head dimension:

  Keys/Queries.  attention logits are q^T k. Substituting q -> R^T q and
                 k -> R^T k leaves q^T R R^T k = q^T k unchanged. So fold R into
                 BOTH W_q and W_k:  W_q <- W_q R,  W_k <- W_k R.
                 Exact, not approximate.

  Values.        the attention output is (A V) W_o. Substituting V -> V R and
                 W_o -> R^T W_o leaves the product unchanged:
                 (A V R)(R^T W_o) = (A V) W_o.
                 So  W_v <- W_v R,  W_o <- R^T W_o.  Also exact.

  RoPE caveat.   rotary embeddings are applied to q,k AFTER the projection and
                 do not commute with a general dense R on the head dimension.
                 `rope_safe_mask` returns the block structure that DOES commute
                 (rotations acting within each RoPE 2-plane pair, or
                 block-diagonal over the pairs). Use `param="butterfly"` with
                 that mask, or apply R to the V/O path only. This module refuses
                 to emit an unsafe fold rather than producing a graph that is
                 numerically wrong at long context -- which is the failure mode
                 that is easy to miss because it only shows up past a few
                 thousand tokens.

Hexagon constraints checked by `check_hexagon`
----------------------------------------------
  * static shapes, no data-dependent control flow
  * INT4/INT8/INT16 weights, FP16/BF16 activations (Hexagon v69+); the X2 series
    NPU adds FP8/BF16
  * per-channel scales are supported; per-element scales are not
  * head_dim a multiple of 32 keeps the vector units fed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["fold_qk", "fold_vo", "rope_safe_mask", "irope_foldable_layers",
           "check_hexagon",
           "HexagonReport", "aihub_export_snippet", "geniex_snippet"]


# --------------------------------------------------------------------------- #
# exact weight folding
# --------------------------------------------------------------------------- #

def _check_orthogonal(R: np.ndarray, tol: float = 1e-6) -> float:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got {R.shape}")
    err = float(np.abs(R @ R.T - np.eye(R.shape[0])).max())
    if err > tol:
        raise ValueError(f"R is not orthogonal (max |RR^T - I| = {err:.3e})")
    return err


def fold_qk(W_q: np.ndarray, W_k: np.ndarray, R: np.ndarray,
            head_dim: int, rope: bool = False):
    """
    Fold R into the query and key projections, per head.
    W_q, W_k : (d_model, n_heads * head_dim)
    Returns (W_q', W_k'). Attention logits are provably unchanged in FP.
    """
    _check_orthogonal(R)
    if R.shape[0] != head_dim:
        raise ValueError(f"R is {R.shape[0]}x{R.shape[0]} but head_dim={head_dim}")
    if rope and not _commutes_with_rope(R):
        raise ValueError(
            "R does not commute with RoPE. Use rope_safe_mask() to constrain the "
            "parameterisation, or fold on the V/O path only. Refusing to emit a "
            "graph that diverges at long context."
        )
    out = []
    for W in (W_q, W_k):
        W = np.asarray(W, dtype=np.float64)
        if W.shape[1] % head_dim:
            raise ValueError(f"W has {W.shape[1]} cols, not a multiple of head_dim {head_dim}")
        Wh = W.reshape(W.shape[0], -1, head_dim)
        out.append(np.einsum("dhc,ce->dhe", Wh, R).reshape(W.shape))
    return out[0], out[1]


def fold_vo(W_v: np.ndarray, W_o: np.ndarray, R: np.ndarray, head_dim: int):
    """
    Fold R into the value projection and its inverse into the output projection.
    W_v : (d_model, n_kv_heads * head_dim)
    W_o : (n_heads * head_dim, d_model)
    Returns (W_v', W_o'). The block output is provably unchanged in FP.
    """
    _check_orthogonal(R)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    Vh = W_v.reshape(W_v.shape[0], -1, head_dim)
    W_v2 = np.einsum("dhc,ce->dhe", Vh, R).reshape(W_v.shape)
    Oh = W_o.reshape(-1, head_dim, W_o.shape[1])
    W_o2 = np.einsum("ce,hcd->hed", R, Oh).reshape(W_o.shape)
    return W_v2, W_o2


def _commutes_with_rope(R: np.ndarray, tol: float = 1e-5) -> bool:
    """
    RoPE acts as a direct sum of 2x2 rotations on pairs (2i, 2i+1) with
    position-dependent angles. A general R commutes with all of them only if R is
    block-diagonal over those pairs with each block a rotation or reflection that
    is itself a 2x2 rotation. Test structurally.
    """
    R = np.asarray(R, dtype=np.float64)
    d = R.shape[0]
    if d % 2:
        return False
    M = np.abs(R.copy())
    for i in range(0, d, 2):
        M[i:i + 2, i:i + 2] = 0.0
    if M.max() > tol:
        return False                              # off-block mass -> no
    for i in range(0, d, 2):
        B = R[i:i + 2, i:i + 2]
        if abs(B[0, 0] - B[1, 1]) > tol or abs(B[0, 1] + B[1, 0]) > tol:
            return False                          # not a 2x2 rotation
    return True


def irope_foldable_layers(n_layers: int, rope_every: int = 2) -> Dict[str, Any]:
    """
    Which layers of an iRoPE model accept an UNRESTRICTED rotation fold.

    Llama 4 interleaves RoPE layers with NoPE layers -- "No Position Embedding"
    layers that carry no positional bias at all. The RoPE commutation problem
    guarded by `_commutes_with_rope` simply does not exist in a NoPE layer:
    with no rotary embedding applied, ANY orthogonal R folds exactly.

    So on an iRoPE model roughly half the layers can take a dense learned or
    Hadamard rotation with no block-diagonal constraint, while the RoPE layers
    stay restricted. A folding pass that applies the RoPE constraint uniformly
    is leaving the NoPE layers unnecessarily hobbled.

    rope_every=2 matches the standard alternating pattern; check the target
    model's config rather than assuming.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if rope_every < 1:
        raise ValueError("rope_every must be >= 1")
    rope = [i for i in range(n_layers) if i % rope_every == 0]
    nope = [i for i in range(n_layers) if i % rope_every != 0]
    return {
        "rope_layers": rope,
        "nope_layers": nope,
        "unrestricted_fraction": len(nope) / n_layers,
        "rope_constraint": "block-diagonal over RoPE 2-planes (rope_safe_mask)",
        "nope_constraint": "none -- any orthogonal R folds exactly",
        "note": "Applies to iRoPE models (Llama 4). On a uniform-RoPE model "
                "every layer is constrained; nope_layers will be empty if you "
                "pass rope_every=1.",
    }


def rope_safe_mask(head_dim: int) -> np.ndarray:
    """
    Boolean (head_dim, head_dim) mask of the entries a RoPE-safe rotation may
    occupy: block-diagonal over the RoPE 2-planes.
    """
    M = np.zeros((head_dim, head_dim), dtype=bool)
    for i in range(0, head_dim, 2):
        M[i:i + 2, i:i + 2] = True
    return M


# --------------------------------------------------------------------------- #
# hardware constraint audit
# --------------------------------------------------------------------------- #

@dataclass
class HexagonReport:
    ok: bool
    checks: dict
    notes: list

    def __str__(self) -> str:
        lines = [f"Hexagon deployability: {'PASS' if self.ok else 'FAIL'}"]
        for k, v in self.checks.items():
            lines.append(f"  [{'ok ' if v else 'FAIL'}] {k}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def check_hexagon(head_dim: int,
                  weight_bits: int,
                  act_dtype: str = "fp16",
                  added_ops: int = 0,
                  dynamic_shapes: bool = False,
                  per_element_scales: bool = False,
                  hexagon_version: int = 79) -> HexagonReport:
    """
    Static audit against the QAIRT / Hexagon execution model. Run this BEFORE
    submitting a compile job so failures surface locally instead of as an opaque
    compiler error.
    """
    checks = {
        "no added runtime operators": added_ops == 0,
        "static shapes only": not dynamic_shapes,
        "weight dtype in {int4,int8,int16}": weight_bits in (4, 8, 16),
        "activation dtype supported": act_dtype in ("fp16", "bf16", "int8", "int16", "fp8"),
        "scales are per-channel or per-group": not per_element_scales,
        "head_dim multiple of 32": head_dim % 32 == 0,
    }
    notes = []
    if act_dtype in ("fp8", "bf16") and hexagon_version < 79:
        notes.append("fp8/bf16 activations need the X2-series NPU (Hexagon v79+); "
                     "fall back to fp16 on X Elite / 8 Elite.")
    if head_dim % 32:
        notes.append(f"head_dim={head_dim} leaves the vector units partly idle; "
                     "pad to a multiple of 32.")
    if weight_bits not in (4, 8, 16):
        notes.append(f"weight_bits={weight_bits} has no native Hexagon path. "
                     "Sub-4-bit weights must be unpacked to int4/int8 at load "
                     "time (llama.cpp CPU/GPU path) or emulated.")
    return HexagonReport(ok=all(checks.values()), checks=checks, notes=notes)


# --------------------------------------------------------------------------- #
# export helpers (emit code, do not execute it -- no network here)
# --------------------------------------------------------------------------- #

def aihub_export_snippet(model_name: str = "sigil_folded",
                         device: str = "Snapdragon X Plus 8-Core CRD") -> str:
    """
    Ready-to-run Qualcomm AI Hub Workbench compile + profile job, ZERO download:
    the folded model is uploaded, compiled and profiled in the cloud, and only
    the profile comes back as a dict. It deliberately never calls a model
    download -- the compiled model stays on Workbench.
    """
    return f'''# pip install qai-hub
# Token: https://workbench.aihub.qualcomm.com -> Account -> Settings -> API Token
# qai-hub configure --api_token <TOKEN>
import qai_hub as hub, torch

client = hub.Client()
model = torch.export.export(folded_model.eval(), (example_inputs,))   # SIGIL-folded

compile_job = client.submit_compile_job(
    model=model,
    device=hub.Device("{device}"),
    name="{model_name}",
    input_specs=dict(input_ids=(1, 128)),
    options="--target_runtime qnn_dlc",
)
profile_job = client.submit_profile_job(
    model=compile_job.get_target_model(),     # a handle: nothing is downloaded
    device=hub.Device("{device}"),
)
print(profile_job.download_profile()["execution_summary"])   # a dict, a few KB
# For an LLM layer at real dimensions with no checkpoint at all, use
# aihub_workbench.py, which builds the slice locally and does the same.
'''


def geniex_snippet(gguf_repo: str = "prism-ml/Ternary-Bonsai-4B-gguf") -> str:
    """GenieX (github.com/qualcomm/geniex) local-serve path for the GGUF tier.
    It runs ON the target machine and pulls the GGUF there -- which is why the
    snippet says how much it downloads."""
    return f'''# pip install geniex        # BSD-3, Snapdragon only
# ON THE TARGET HP LAPTOP ONLY [downloads ~1 GB to THIS machine for the default
# 4B ternary GGUF -- never run it on a dev box]
geniex pull {gguf_repo}
geniex serve {gguf_repo} --hardware npu --port 8080
# OpenAI-compatible endpoint at http://localhost:8080/v1
'''
