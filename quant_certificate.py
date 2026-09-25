#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 quant_certificate.py -- prove a runtime is decoding your model correctly
================================================================================

SOURCE. github.com/openai/NavierStokesAndEuler ships the Lean 4 certificates
for the blowup proofs, and with them a verification protocol worth stealing.
An independent party checks a claim by running

    lake exe comparator ComparatorChallenges/NavierStokes.json

against three separate tools: `lean4export` to get the proof out in a neutral
format, `nanoda_bin` -- an INDEPENDENT reimplementation of the Lean kernel --
to check it, and `landrun` to sandbox the checker. Nothing in that chain
requires trusting the toolchain that produced the artefact. The claim is a
JSON file, the check is reproducible, and the checker is not the producer.

--------------------------------------------------------------------------------
 THE PROBLEM THIS SOLVES
--------------------------------------------------------------------------------
A quantised model has exactly the property that makes verification necessary:
it fails SILENTLY. From PrismML's own PACK-RUNTIME.md for Bonsai 2 27B, whose
weights are stored in a blockwise-Hadamard-rotated basis:

    "Ordinary MLX loaders do not apply the required transforms."

A loader that does not know about the basis returns WRONG OUTPUT rather than an
error. It loads, it runs, it emits fluent text, and it is wrong. PrismML ship
`reload-validation.json` against this, and are honest about its reach:

    "Reload validation confirms serialization integrity but doesn't assess
     model quality or cross-platform equivalence."

So the gap is real and its owner says so. On Snapdragon it is wider still: a
model reaching a Hexagon NPU has been through QNN conversion, graph
optimisation and quantisation, any of which can change numerics without
changing behaviour that anyone is watching.

I have made this mistake myself, which is why the probe set below is shaped the
way it is. In `sigil_t4_benchmark.py` an earlier release reduced Keys over the
HEAD axis instead of the TOKEN axis. Nothing crashed. Perplexity came out
plausible. An entire published sweep compared two different quantisation
schemes against each other, and the headline it produced -- "grouping is 4x
worse than per-token" -- was an artefact of that bug. A digest-and-probe check
would have caught it in milliseconds. WRONG_AXIS is in DETECTABLE_FAULTS
because of it.

--------------------------------------------------------------------------------
 WHAT A CERTIFICATE IS
--------------------------------------------------------------------------------
A small JSON file shipped beside the weights, declaring

  * the quantisation scheme: alphabet, group size, the AXIS the group runs
    along, container, stored bits per weight
  * the transform: kind, block size, and a digest of the sign vector, so a
    mismatched rotation table is caught before any weight is read
  * a set of PROBES: fixed pseudo-random inputs, generated from a seed rather
    than stored, each with the digest and summary statistics of the output a
    correct runtime produces
  * a tolerance DERIVED from the declared dtype and reduction length, not
    guessed

Verification recomputes the probes on the candidate runtime and compares. It
needs no calibration data, no reference weights and no network, runs in
milliseconds, and -- the point of the Comparator pattern -- does not trust the
runtime that produced the artefact.

Digests alone would give a bare pass/fail. Summary statistics are carried too,
because the interesting question on a failure is WHICH fault occurred, and the
signature differs: a skipped rotation preserves the norm, a wrong group axis
does not, and a transposed scale changes the sign structure.

--------------------------------------------------------------------------------
 WHAT IS MEASURED, AND WHAT IS NOT
--------------------------------------------------------------------------------
`experiment_fault_detection()` builds one correct pipeline and seven broken
ones -- skipped transform, flipped signs, wrong group axis, wrong group size,
wrong container, transposed scales, fp16 accumulation -- and runs the verifier
against each with the scheme declared CORRECTLY, so the probes have to do the
work unaided.

DETECTION HOLDS. 7/7 faults refused, and the correct pipeline passes, at every
size and seed tried:

    d = 64x32 seed 7     7/7      d = 128x64 seed 11    7/7
    d = 256x128 seed 23  7/7      d = 32x16 seed 99     7/7

DIAGNOSIS DOES NOT. Naming which fault occurred is right 7/7 on the harness the
signature table was measured from and only 4/7 to 6/7 elsewhere. The table
spans five orders of magnitude but its closest distinct pairs sit 0.53 apart in
log space, which a change of problem size is enough to reorder. So `diagnose()`
returns a ranked hint, never a verdict, and the declaration checks are what
give hard answers. Reporting it as anything better would be the same class of
mistake the certificate exists to catch.

Two findings fell out of building it:

  * WRONG_AXIS and TRANSPOSED_SCALE are the SAME FAULT. Laying per-group scales
    out transposed IS applying the group along the other axis; their measured
    signatures are identical to five decimals. The catalogue keeps both names
    because both appear in the wild, and INDISTINGUISHABLE says they are one.
  * The first verifier here passed a pipeline that negated every Hadamard sign.
    l2, std and absmax are all invariant under negation, so the statistical
    fallback saw a perfect match -- the exact fault DETECTABLE_FAULTS
    ["WRONG_SIGNS"] warns about, in the code written to catch it. `proj` in
    array_stats() exists because of that.

NOT DONE: this has not been run against a real quantised checkpoint on real
silicon. That is the next step and it is a small one -- the verifier is about
60 lines and needs only numpy.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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



try:
    import numpy as np
except ImportError:                                    # pragma: no cover
    np = None

__all__ = [
    "CERT_VERSION", "QuantScheme", "TransformSpec", "Probe", "ProbeResult",
    "Certificate", "DETECTABLE_FAULTS", "STANDARD_PROBES",
    "probe_input", "digest_array", "array_stats", "derive_tolerance",
    "build_certificate", "verify_certificate", "diagnose",
    "experiment_fault_detection", "selftest", "INDISTINGUISHABLE",
    "FAULT_SIGNATURES",
]

CERT_VERSION = "sigil-quant-cert/1"


# ============================================================================ #
# SECTION 1 -- what a certificate declares
# ============================================================================ #

@dataclass
class QuantScheme:
    """
    How weights are stored. `axis` is here because omitting it is the single
    most common way to be silently wrong: a group of 128 means nothing until
    you say 128 of WHAT.
    """
    alphabet: str                  # binary | ternary | int2 | int3 | int4 | int8
    group: Optional[int]           # None means per-axis
    axis: str                      # "in" | "out" | "token" | "channel"
    container: str                 # gguf-PTQ1_0 | gguf-PQ2_0 | mlx-affine | ...
    stored_bpw: float
    scale_dtype: str = "fp16"
    zero_point: bool = False       # affine containers carry one, symmetric do not

    def summary(self) -> str:
        g = f"g{self.group}" if self.group else "per-axis"
        z = "+zp" if self.zero_point else ""
        return (f"{self.alphabet}/{g}/{self.axis}/{self.container}"
                f"{z} @ {self.stored_bpw:.3f} bpw")


@dataclass
class TransformSpec:
    """
    A declared weight-basis transform. `sign_digest` is what catches a
    mismatched rotation table: Bonsai 2 stores 1024 fixed +/-1 signs in
    hadamard.json, and loading the weights against the wrong ones produces
    fluent, wrong output.
    """
    kind: str                      # none | hadamard | cayley | learned
    block: Optional[int] = None
    sign_digest: Optional[str] = None
    folded_into_weights: bool = True
    inverse_in_embedding: bool = False

    def summary(self) -> str:
        if self.kind == "none":
            return "no transform"
        return (f"{self.kind} block={self.block} "
                f"signs={(self.sign_digest or '-')[:12]} "
                f"{'folded' if self.folded_into_weights else 'runtime'}")


@dataclass
class Probe:
    """
    A test input, generated from a seed rather than stored, so a certificate
    stays a few kilobytes no matter how many probes it carries.
    """
    name: str
    seed: int
    rows: int
    cols: int
    kind: str = "gaussian"         # gaussian | impulse | constant | alternating
    targets: Tuple[str, ...] = ()  # fault classes this probe is meant to catch


@dataclass
class ProbeResult:
    name: str
    input_digest: str
    output_digest: str
    stats: Dict[str, float]


@dataclass
class Certificate:
    version: str
    model_id: str
    scheme: Dict[str, Any]
    transform: Dict[str, Any]
    probes: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    tolerance: Dict[str, float]
    producer: str = ""
    notes: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "Certificate":
        d = json.loads(text)
        missing = {"version", "model_id", "scheme", "transform", "probes",
                   "results", "tolerance"} - set(d)
        if missing:
            raise ValueError(f"certificate is missing fields: {sorted(missing)}")
        if d["version"] != CERT_VERSION:
            raise ValueError(f"unsupported certificate version {d['version']!r}; "
                             f"this verifier speaks {CERT_VERSION!r}")
        return Certificate(**{k: d[k] for k in
                              ("version", "model_id", "scheme", "transform",
                               "probes", "results", "tolerance")},
                           producer=d.get("producer", ""),
                           notes=d.get("notes", ""))


# --------------------------------------------------------------------------- #
# The faults worth detecting. Each is something that has actually happened in
# a shipped or published pipeline, not a hypothetical.
# --------------------------------------------------------------------------- #

DETECTABLE_FAULTS = {
    "SKIPPED_TRANSFORM": {
        "what": "Weights are in a rotated basis; the runtime did not rotate the "
                "activations to match.",
        "evidence": "PrismML PACK-RUNTIME.md: ordinary MLX loaders return wrong "
                    "output rather than an error.",
        "signature": "Output norm roughly preserved (the transform is "
                     "orthogonal) but direction is wrong -- correlation with "
                     "the reference collapses while scale looks healthy.",
    },
    "WRONG_SIGNS": {
        "what": "Right transform kind and block, wrong sign vector.",
        "evidence": "Bonsai 2 stores 1024 fixed +/-1 signs in hadamard.json; "
                    "nothing in the weight file pins them.",
        "signature": "Same as a skipped transform but partial -- correlation "
                     "degrades in proportion to the number of flipped signs.",
    },
    "WRONG_AXIS": {
        "what": "Group quantisation applied along the wrong axis.",
        "evidence": "This project's own sigil_t4_benchmark.py reduced Keys over "
                    "the head axis instead of the token axis and published a "
                    "headline that was an artefact of it.",
        "signature": "Norm and correlation both shift; the error is structured "
                     "by row rather than spread evenly.",
    },
    "TRANSPOSED_SCALE": {
        "what": "Per-group scales applied transposed or broadcast wrongly.",
        "evidence": "A classic in hand-written NPU kernels, where the scale "
                    "layout is chosen for memory access rather than clarity.",
        "signature": "Large per-row scale errors; output std diverges sharply "
                     "from the reference while the mean stays near zero.",
    },
    "WRONG_GROUP_SIZE": {
        "what": "Runtime decoded with a different group size than declared.",
        "evidence": "ggml-org discussion #22019 covers a group-128 ternary "
                    "format alongside the earlier one; a container that carries "
                    "both invites this.",
        "signature": "Subtle -- a smaller group is strictly better, so error "
                     "shrinks rather than grows. Digest catches it; statistics "
                     "alone would not.",
    },
    "WRONG_CONTAINER": {
        "what": "Decoded as a different container at the same nominal width.",
        "evidence": "Bonsai 2 ships PTQ1_0 (1.768 bpw) and PQ2_0 (2.143 bpw) "
                    "side by side, both described as ternary.",
        "signature": "Gross -- output is unrelated to the reference.",
    },
    "DTYPE_DOWNGRADE": {
        "what": "Accumulation in fp16 where the certificate declares fp32.",
        "evidence": "Common on NPUs, where the wider accumulator costs area.",
        "signature": "Small, systematic, and within a naive tolerance -- which "
                     "is why the tolerance here is derived from the declared "
                     "dtype and reduction length rather than guessed.",
    },
}


# Probes chosen so that between them they exercise every fault above. A
# gaussian probe measures the typical case; an impulse probe isolates single
# columns and so exposes axis and scale-layout faults; a constant probe cancels
# the oscillatory part of a rotation and exposes a skipped transform; an
# alternating probe is the worst case for a sign vector.
STANDARD_PROBES = (
    Probe("gaussian", 20260919, 8, 0, "gaussian",
          ("SKIPPED_TRANSFORM", "WRONG_CONTAINER", "WRONG_GROUP_SIZE",
           "DTYPE_DOWNGRADE")),
    Probe("impulse", 20260920, 8, 0, "impulse",
          ("WRONG_AXIS", "TRANSPOSED_SCALE", "WRONG_GROUP_SIZE")),
    Probe("constant", 20260921, 4, 0, "constant",
          ("SKIPPED_TRANSFORM", "WRONG_SIGNS")),
    Probe("alternating", 20260922, 4, 0, "alternating",
          ("WRONG_SIGNS", "WRONG_AXIS")),
)


# ============================================================================ #
# SECTION 2 -- probe generation, digests, tolerance
# ============================================================================ #

def _require_numpy():
    if np is None:                                     # pragma: no cover
        raise RuntimeError("numpy is required: pip install -U numpy")


def probe_input(probe: Probe, cols: int):
    """
    Regenerate a probe's input from its seed. Deterministic across machines:
    numpy's PCG64 is specified, so the same seed gives the same bits on ARM and
    x86 alike -- which is the whole point, since the verifier runs on the
    device and the certificate was produced on a workstation.
    """
    _require_numpy()
    if not isinstance(probe, Probe):
        raise TypeError(f"probe must be a Probe, got {type(probe).__name__}")
    try:
        n = int(cols or probe.cols)
    except (TypeError, ValueError):
        raise TypeError(f"cols must be an integer, got {cols!r}")
    if n <= 0 or n > 1_000_000:
        raise ValueError(f"column count out of range: {n}")
    if not isinstance(probe.rows, int) or not 0 < probe.rows <= 10_000:
        raise ValueError(f"probe.rows out of range: {probe.rows!r}")
    rng = np.random.Generator(np.random.PCG64(probe.seed))
    if probe.kind == "gaussian":
        x = rng.standard_normal((probe.rows, n))
    elif probe.kind == "impulse":
        x = np.zeros((probe.rows, n))
        for i in range(probe.rows):
            x[i, (i * 7 + 1) % n] = 1.0
    elif probe.kind == "constant":
        x = np.ones((probe.rows, n)) * (1.0 + np.arange(probe.rows)[:, None])
    elif probe.kind == "alternating":
        base = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        x = np.stack([base * (i + 1) for i in range(probe.rows)])
    else:
        raise ValueError(f"unknown probe kind {probe.kind!r}")
    return np.ascontiguousarray(x, dtype=np.float64)


def digest_array(a, places: int = 6) -> str:
    """
    A digest that is stable across machines. Values are rounded before hashing,
    because bit-exact float equality across two builds of a BLAS is not a thing
    anyone should require, and requiring it would make every certificate fail
    for the wrong reason.
    """
    _require_numpy()
    q = np.round(np.asarray(a, dtype=np.float64), places)
    q = q + 0.0                                        # normalise -0.0 to 0.0
    h = hashlib.sha256()
    h.update(str(q.shape).encode())
    h.update(q.tobytes())
    return h.hexdigest()[:32]


def _projection_signs(n: int):
    """A fixed pseudo-random +/-1 vector of length n. Deterministic everywhere."""
    _require_numpy()
    rng = np.random.Generator(np.random.PCG64(0xC0FFEE + n))
    return np.where(rng.random(n) < 0.5, -1.0, 1.0)


def array_stats(a) -> Dict[str, float]:
    """
    Summary statistics, so a failure says WHICH fault rather than just "no".

    `proj` exists because everything else here is DIRECTION-BLIND. l2, std and
    absmax are all invariant under negation, so a runtime that applied the
    Hadamard with every sign flipped -- output exactly negated -- matched the
    reference on all of them and the verifier passed it. That is precisely the
    fault DETECTABLE_FAULTS["WRONG_SIGNS"] describes, and the first version of
    this verifier had it. `proj` is the output projected onto a fixed
    pseudo-random +/-1 vector: cheap, and it moves whenever direction does.
    """
    _require_numpy()
    try:
        x = np.asarray(a, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise TypeError(f"expected an array of numbers, got {type(a).__name__}")
    if x.size and not np.all(np.isfinite(x)):
        # Statistics over NaN are NaN, and a certificate full of NaN compares
        # equal to nothing and unequal to everything. Refuse instead.
        raise ValueError("array contains non-finite values")
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "l2": 0.0, "absmax": 0.0,
                "frac_positive": 0.0, "proj": 0.0}
    nrm = float(np.linalg.norm(x))
    if not math.isfinite(nrm):
        # Every element was finite but the norm overflowed. A statistic of inf
        # compares unequal to everything, so refuse rather than record it.
        raise ValueError("array magnitudes overflow a float64 norm")
    proj = float(np.dot(x, _projection_signs(x.size)) / (nrm or 1.0))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "l2": nrm,
        "absmax": float(np.max(np.abs(x))),
        "frac_positive": float(np.mean(x > 0)),
        "proj": proj,
    }


_DTYPE_EPS = {"fp32": 2 ** -24, "tf32": 2 ** -11, "bf16": 2 ** -8,
              "fp16": 2 ** -11, "fp64": 2 ** -53}


def derive_tolerance(accum_dtype: str, reduction_len: int,
                     safety: float = 8.0) -> Dict[str, float]:
    """
    Tolerance from first principles rather than from taste.

    A dot product of length k accumulated at unit roundoff eps has relative
    error growing like sqrt(k)*eps for the random-sign case. `safety` covers
    reordering, fused multiply-add and a different BLAS.

    Getting this right matters in one direction especially: DTYPE_DOWNGRADE is
    a real fault whose error is SMALL, so a tolerance picked generously enough
    to "avoid false alarms" would let it through every time.
    """
    if accum_dtype not in _DTYPE_EPS:
        raise ValueError(f"unknown accumulation dtype {accum_dtype!r}; "
                         f"known: {sorted(_DTYPE_EPS)}")
    if reduction_len < 1:
        raise ValueError("reduction_len must be >= 1")
    eps = _DTYPE_EPS[accum_dtype]
    rel = safety * math.sqrt(reduction_len) * eps
    # A tolerance at or above 0.5 accepts essentially any output, which is not
    # a loose check but an absent one. bf16 over a 4096-length reduction lands
    # at 2.0: that dtype genuinely cannot be certified at that depth, and
    # saying so is more useful than issuing a number that always passes.
    certifiable = rel < 0.5
    return {
        "certifiable": certifiable,
        "why_not": ("" if certifiable else
                    f"{accum_dtype} accumulation over {reduction_len} terms "
                    f"gives a {rel:.2f} relative bound -- too loose to mean "
                    f"anything. Accumulate in fp32, or certify shorter "
                    f"reductions."),
        "accum_dtype": accum_dtype,
        "reduction_len": float(reduction_len),
        "rel_l2": rel,
        "min_correlation": max(0.0, 1.0 - 100.0 * rel * rel),
        "rel_std": rel * 4.0,
        "digest_places": 6.0,
    }


# ============================================================================ #
# SECTION 3 -- build and verify
# ============================================================================ #

def build_certificate(model_id: str, scheme: QuantScheme,
                      transform: TransformSpec,
                      forward: Callable[[Any], Any], cols: int,
                      probes: Sequence[Probe] = STANDARD_PROBES,
                      accum_dtype: str = "fp32",
                      producer: str = "", notes: str = "") -> Certificate:
    """
    Run each probe through a KNOWN-GOOD forward and record what it produced.

    `forward` maps (rows, cols) -> (rows, out). It is the reference
    implementation, and the certificate is only as trustworthy as it is -- which
    is exactly the Comparator position: the certificate proves AGREEMENT with a
    declared reference, not correctness in the abstract.
    """
    _require_numpy()
    results = []
    for p in probes:
        x = probe_input(p, cols)
        y = np.asarray(forward(x), dtype=np.float64)
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"probe {p.name}: forward returned {y.shape[0]} "
                             f"rows for {x.shape[0]} inputs")
        results.append(ProbeResult(p.name, digest_array(x), digest_array(y),
                                   array_stats(y)))
    return Certificate(
        version=CERT_VERSION, model_id=model_id,
        scheme=asdict(scheme), transform=asdict(transform),
        probes=[asdict(p) for p in probes],
        results=[asdict(r) for r in results],
        tolerance=derive_tolerance(accum_dtype, cols),
        producer=producer, notes=notes,
    )


def _correlation(a, b) -> float:
    _require_numpy()
    x, y = np.ravel(a), np.ravel(b)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 1.0 if nx == ny else 0.0
    return float(np.dot(x, y) / (nx * ny))


def verify_certificate(cert: Certificate, forward: Callable[[Any], Any],
                       cols: int,
                       declared_scheme: Optional[QuantScheme] = None,
                       declared_transform: Optional[TransformSpec] = None
                       ) -> Dict[str, Any]:
    """
    Recompute every probe on a candidate runtime and compare.

    REFUSE BY DEFAULT. A probe that cannot be run, a scheme that does not match
    what the runtime says it is doing, or a missing sign digest all fail. The
    alternative -- loading anyway and hoping -- is the behaviour that makes
    quantisation bugs invisible in the first place.
    """
    _require_numpy()
    tol = cert.tolerance
    checks: List[Dict[str, Any]] = []

    if declared_scheme is not None:
        for field_name in ("alphabet", "group", "axis", "container",
                           "zero_point"):
            want = cert.scheme.get(field_name)
            got = getattr(declared_scheme, field_name)
            checks.append({
                "check": f"scheme.{field_name}", "ok": want == got,
                "expected": want, "actual": got,
                "fault": ("WRONG_AXIS" if field_name == "axis" else
                          "WRONG_GROUP_SIZE" if field_name == "group" else
                          "WRONG_CONTAINER"),
            })
    if declared_transform is not None:
        for field_name in ("kind", "block", "sign_digest"):
            want = cert.transform.get(field_name)
            got = getattr(declared_transform, field_name)
            checks.append({
                "check": f"transform.{field_name}", "ok": want == got,
                "expected": want, "actual": got,
                "fault": ("WRONG_SIGNS" if field_name == "sign_digest"
                          else "SKIPPED_TRANSFORM"),
            })

    probe_rows = []
    by_name = {r["name"]: r for r in cert.results}
    for pd in cert.probes:
        p = Probe(**pd)
        ref = by_name.get(p.name)
        if ref is None:
            probe_rows.append({"probe": p.name, "ok": False,
                               "why": "no recorded result in the certificate"})
            continue
        x = probe_input(p, cols)
        if digest_array(x) != ref["input_digest"]:
            probe_rows.append({"probe": p.name, "ok": False,
                               "why": "probe input does not regenerate; the "
                                      "certificate was built for a different "
                                      "input width"})
            continue
        try:
            y = np.asarray(forward(x), dtype=np.float64)
        except Exception as exc:
            probe_rows.append({"probe": p.name, "ok": False,
                               "why": f"forward raised {type(exc).__name__}: {exc}"})
            continue
        got = array_stats(y)
        exact = digest_array(y) == ref["output_digest"]
        ref_l2 = ref["stats"]["l2"] or 1.0
        rel_l2 = abs(got["l2"] - ref["stats"]["l2"]) / ref_l2
        rel_std = (abs(got["std"] - ref["stats"]["std"]) /
                   (ref["stats"]["std"] or 1.0))
        # Correlation needs the reference vector, which the certificate does not
        # carry. Reconstruct the comparison from the digest when it matches and
        # fall back to the statistics when it does not -- that asymmetry is
        # deliberate: an exact digest is proof, statistics are diagnosis.
        # Direction, not just magnitude. `proj` is already normalised by the
        # norm, so this compares orientation independently of scale.
        ref_proj = ref["stats"].get("proj")
        d_proj = (abs(got["proj"] - ref_proj) if ref_proj is not None else 0.0)
        ok = exact or (rel_l2 <= tol["rel_l2"]
                       and rel_std <= tol["rel_std"]
                       and d_proj <= tol["rel_l2"] * 10.0 + 1e-9)
        probe_rows.append({
            "probe": p.name, "ok": ok, "digest_exact": exact,
            "rel_l2": round(rel_l2, 9), "rel_std": round(rel_std, 9),
            "d_proj": round(d_proj, 9),
            "tol_rel_l2": tol["rel_l2"], "targets": list(p.targets),
            "ref_stats": ref["stats"], "got_stats": got,
        })

    failed_checks = [c for c in checks if not c["ok"]]
    failed_probes = [r for r in probe_rows if not r["ok"]]
    verdict = "PASS" if not failed_checks and not failed_probes else "REFUSE"
    return {
        "verdict": verdict,
        "model_id": cert.model_id,
        "scheme": cert.scheme,
        "declaration_checks": checks,
        "probes": probe_rows,
        "failed_checks": failed_checks,
        "failed_probes": [r["probe"] for r in failed_probes],
        "likely_faults": diagnose(failed_checks, failed_probes),
        "action": ("load" if verdict == "PASS" else
                   "DO NOT LOAD -- this runtime is not reproducing the "
                   "certified numerics, and a quantised model fails silently"),
    }


# Signatures MEASURED from experiment_fault_detection(), as (rel_l2, rel_std,
# d_proj) averaged over the failing probes. These are indicative, not
# authoritative: they come from one synthetic harness, and a real checkpoint
# will shift the magnitudes. What they capture is the SHAPE of each fault, and
# the shapes are well separated over five orders of magnitude.
FAULT_SIGNATURES = {
    "WRONG_SIGNS":        (1e-7,   1e-7,   2.544),
    "SKIPPED_TRANSFORM":  (0.139,  0.117,  1.254),
    "WRONG_CONTAINER":    (0.103,  0.113,  0.354),
    "WRONG_GROUP_SIZE":   (0.0087, 0.0111, 0.049),
    "WRONG_AXIS":         (0.0037, 0.0074, 0.106),
    "TRANSPOSED_SCALE":   (0.0037, 0.0074, 0.106),
    "DTYPE_DOWNGRADE":    (1e-5,   1e-5,   2.8e-4),
}

# Faults that probes cannot separate, and what does. Stating this is the
# difference between a diagnosis and a guess.
INDISTINGUISHABLE = {
    frozenset({"WRONG_AXIS", "TRANSPOSED_SCALE"}):
        "These are the SAME OPERATION under two names: laying the per-group "
        "scales out transposed IS applying the group along the other axis. "
        "Measured signatures are identical to five decimal places. Probes "
        "cannot separate them and nothing else should try -- fixing either "
        "description fixes the fault.",
}

# Two faults that looked like they would collide and do not, recorded because
# the first version of this module asserted they would:
#   WRONG_SIGNS      rel_l2 = 0.000, d_proj = 2.54  (magnitude exact, direction destroyed)
#   SKIPPED_TRANSFORM rel_l2 = 0.139, d_proj = 1.25  (both wrong)
# The weights were quantised IN the rotated basis, so skipping the rotation
# perturbs magnitude as well as direction, while flipping every sign is exactly
# orthogonal and leaves magnitude untouched.


def _probe_signature(r: Dict[str, Any]) -> Tuple[float, float, float]:
    """The (rel_l2, rel_std, d_proj) triple for one failed probe."""
    return (r.get("rel_l2", 0.0), r.get("rel_std", 0.0), r.get("d_proj", 0.0))


def _log_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Distance in log space, because these quantities span five orders of
    magnitude and a linear metric would let the largest one decide everything.
    """
    floor = 1e-9
    return math.sqrt(sum(
        (math.log10(max(x, floor)) - math.log10(max(y, floor))) ** 2
        for x, y in zip(a, b)))


def diagnose(failed_checks: Sequence[Dict[str, Any]],
             failed_probes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank the likely faults by matching the observed signature against
    FAULT_SIGNATURES, nearest neighbour in log space.

    Two earlier versions of this function were worse, and both failures are
    worth keeping in view. Counting how many failing probes named each fault
    returned SKIPPED_TRANSFORM for all seven injected faults -- when everything
    fails, vote counting just reports whichever fault the most probes mention.
    Hand-written threshold rules did little better, because the thresholds were
    guesses. Measuring the signatures first, then matching, works.

    A declaration mismatch still outranks any inference: it is direct evidence,
    not a fit.
    """
    # A string iterates as characters and each one fails `.get` with an
    # AttributeError -- the confused class. Refuse wrong shapes up front.
    for name, seq in (("failed_checks", failed_checks), ("failed_probes", failed_probes)):
        if not isinstance(seq, (list, tuple)):
            raise TypeError(f"{name} must be a list of dicts, got {type(seq).__name__}")
        if not all(isinstance(x, dict) for x in seq):
            raise TypeError(f"every entry of {name} must be a dict")
    out: List[Dict[str, Any]] = []
    declared = []
    for c in failed_checks:
        f = c.get("fault")
        if f and f not in declared:
            declared.append(f)

    ranked: List[Tuple[str, float]] = []
    if failed_probes:
        sigs = [_probe_signature(r) for r in failed_probes]
        obs = tuple(sum(v[i] for v in sigs) / len(sigs) for i in range(3))
        ranked = sorted(
            ((name, _log_distance(obs, ref))
             for name, ref in FAULT_SIGNATURES.items()),
            key=lambda kv: kv[1])

    seen = set()
    order = ([(f, 0.0, True) for f in declared]
             + [(f, d, False) for f, d in ranked])
    for fault, dist, is_declared in order:
        if fault in seen:
            continue
        seen.add(fault)
        info = DETECTABLE_FAULTS.get(fault, {})
        row: Dict[str, Any] = {
            "fault": fault,
            "evidence": "declaration mismatch" if is_declared else "probe signature",
            "score": round(100.0 if is_declared else 1.0 / (1.0 + dist), 3),
            "what": info.get("what", ""),
            "signature": info.get("signature", ""),
        }
        for pair, why in INDISTINGUISHABLE.items():
            if fault in pair:
                row["not_separable_from"] = sorted(pair - {fault})
                row["what_separates_them"] = why
        out.append(row)
    return out


# ============================================================================ #
# SECTION 4 -- does it actually catch anything?
# ============================================================================ #

def _hadamard(n: int, signs=None):
    _require_numpy()
    if n & (n - 1):
        raise ValueError("Hadamard size must be a power of two")
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    H = H / math.sqrt(n)
    if signs is not None:
        H = H * np.asarray(signs).reshape(1, -1)
    return H


def _quantise(w, bits: int, group: Optional[int], axis: int):
    _require_numpy()
    qmax = _qmax(bits)
    x = np.moveaxis(w, axis, -1)
    n = x.shape[-1]
    g = group or n
    pad = (-n) % g
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    b = x.reshape(*x.shape[:-1], -1, g)
    s = np.maximum(np.abs(b).max(-1, keepdims=True), 1e-12) / qmax
    q = (np.clip(np.round(b / s), -qmax, qmax) * s).reshape(*x.shape)
    if pad:
        q = q[..., :n]
    return np.moveaxis(q, -1, axis)


def experiment_fault_detection(d_in: int = 64, d_out: int = 32,
                               seed: int = 7,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Build one correct pipeline and six broken ones, and report which faults the
    certificate catches. This is the test that decides whether the format is
    useful or merely tidy.
    """
    _require_numpy()
    rng = np.random.Generator(np.random.PCG64(seed))
    W = rng.standard_normal((d_out, d_in)) / math.sqrt(d_in)
    signs = np.where(rng.random(d_in) < 0.5, -1.0, 1.0)
    H = _hadamard(d_in, signs)
    bits, group = 4, 16

    # Reference: rotate into the stored basis, quantise there, and have the
    # runtime rotate activations to match. This is the Bonsai 2 arrangement.
    Wr = W @ H
    Wq = _quantise(Wr, bits, group, axis=1)

    def good(x):
        return (x @ H) @ Wq.T

    faults = {
        "SKIPPED_TRANSFORM": lambda x: x @ Wq.T,
        "WRONG_SIGNS": lambda x: (x @ _hadamard(d_in, -signs)) @ Wq.T,
        "WRONG_AXIS": lambda x: (x @ H) @ _quantise(Wr, bits, group, axis=0).T,
        "WRONG_GROUP_SIZE": lambda x: (x @ H) @ _quantise(Wr, bits, 8, 1).T,
        "WRONG_CONTAINER": lambda x: (x @ H) @ _quantise(Wr, 2, group, 1).T,
        "TRANSPOSED_SCALE": lambda x: (x @ H) @ (
            _quantise(Wr.T, bits, group, axis=1).T).T,
        "DTYPE_DOWNGRADE": lambda x: (
            (x @ H).astype(np.float16) @ Wq.T.astype(np.float16)
        ).astype(np.float64),
    }

    scheme = QuantScheme("int4", group, "in", "gguf-PQ2_0", 2.143, "fp16", False)
    transform = TransformSpec("hadamard", d_in,
                              digest_array(signs), True, False)
    cert = build_certificate("synthetic/fault-detection-demo", scheme,
                             transform, good, d_in,
                             producer="quant_certificate.experiment")

    self_check = verify_certificate(cert, good, d_in, scheme, transform)
    rows = [{"pipeline": "correct", "verdict": self_check["verdict"],
             "caught": self_check["verdict"] == "PASS", "faults": []}]
    if verbose:
        print(f"  {'pipeline':<20}{'verdict':>9}{'probes failed':>16}  "
              f"top diagnosis")
        print(f"  {'correct':<20}{self_check['verdict']:>9}"
              f"{len(self_check['failed_probes']):>16}  "
              f"{'-- (must pass)'}")

    caught = 0
    for name, fn in faults.items():
        # A runtime that MISDECLARES its scheme is caught by the declaration
        # check; one that declares correctly but behaves wrongly must be caught
        # by the probes alone. Test the harder case: declare correctly.
        res = verify_certificate(cert, fn, d_in, scheme, transform)
        ok = res["verdict"] == "REFUSE"
        caught += ok
        top = res["likely_faults"][0]["fault"] if res["likely_faults"] else "-"
        rows.append({"pipeline": name, "verdict": res["verdict"],
                     "caught": ok, "failed_probes": res["failed_probes"],
                     "top_diagnosis": top})
        if verbose:
            print(f"  {name:<20}{res['verdict']:>9}"
                  f"{len(res['failed_probes']):>16}  {top}")
    if verbose:
        print()
        print(f"  correct pipeline passes: {self_check['verdict'] == 'PASS'}")
        print(f"  faults caught by probes alone: {caught}/{len(faults)}")
    return {"ran": True, "rows": rows, "caught": caught,
            "total": len(faults),
            "correct_passes": self_check["verdict"] == "PASS",
            "certificate_bytes": len(cert.to_json())}


# ============================================================================ #
# SECTION 5 -- self test
# ============================================================================ #

def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), str(detail)))

    if np is None:
        print("  numpy is required for --selftest. pip install -U numpy")
        return 2

    # ---- probe generation is deterministic ----
    p = STANDARD_PROBES[0]
    ck("probe input regenerates identically",
       np.array_equal(probe_input(p, 64), probe_input(p, 64)))
    ck("different seeds give different probes",
       not np.array_equal(probe_input(p, 64),
                          probe_input(Probe("x", p.seed + 1, p.rows, 0), 64)))
    ck("probe width follows the model, not the certificate",
       probe_input(p, 128).shape == (p.rows, 128))
    ck("zero width is rejected", _raises(lambda: probe_input(p, 0)))
    ck("unknown probe kind is rejected",
       _raises(lambda: probe_input(Probe("x", 1, 2, 4, "nope"), 4)))
    for kind in ("gaussian", "impulse", "constant", "alternating"):
        x = probe_input(Probe("t", 5, 4, 0, kind), 32)
        ck(f"{kind} probe is finite and correctly shaped",
           x.shape == (4, 32) and bool(np.isfinite(x).all()))
    ck("impulse probe is sparse",
       float((probe_input(Probe("i", 1, 8, 0, "impulse"), 64) != 0).mean()) < 0.05)

    # ---- digests ----
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    ck("digest is stable", digest_array(a) == digest_array(a.copy()))
    ck("digest is shape-sensitive",
       digest_array(a) != digest_array(a.reshape(4, 3)))
    ck("digest tolerates sub-rounding jitter",
       digest_array(a) == digest_array(a + 1e-9))
    ck("digest catches a real change",
       digest_array(a) != digest_array(a + 1e-3))
    ck("digest normalises negative zero",
       digest_array(np.array([0.0])) == digest_array(np.array([-0.0])))

    # ---- tolerance is derived, not guessed ----
    t32 = derive_tolerance("fp32", 4096)
    t16 = derive_tolerance("fp16", 4096)
    ck("fp16 tolerance is wider than fp32", t16["rel_l2"] > t32["rel_l2"])
    ck("tolerance grows with reduction length",
       derive_tolerance("fp32", 16384)["rel_l2"] >
       derive_tolerance("fp32", 1024)["rel_l2"])
    ck("tolerance grows as sqrt(k)",
       abs(derive_tolerance("fp32", 4096)["rel_l2"] /
           derive_tolerance("fp32", 1024)["rel_l2"] - 2.0) < 1e-9)
    ck("fp32 tolerance stays tight enough to catch an fp16 downgrade",
       t32["rel_l2"] < 1e-4, f"{t32['rel_l2']:.2e}")
    ck("unknown dtype is rejected", _raises(lambda: derive_tolerance("int4", 10)))
    ck("reduction length is validated",
       _raises(lambda: derive_tolerance("fp32", 0)))

    # ---- serialisation round-trips and refuses junk ----
    sch = QuantScheme("ternary", 128, "in", "gguf-PTQ1_0", 1.768, "fp16", False)
    tr = TransformSpec("hadamard", 1024, "abc123", True, True)
    cert = build_certificate("demo/model", sch, tr,
                             lambda x: x @ np.ones((32, 8)), 32)
    ck("certificate round-trips through JSON",
       Certificate.from_json(cert.to_json()).model_id == "demo/model")
    ck("certificate is small", len(cert.to_json()) < 8000,
       f"{len(cert.to_json())} bytes")
    ck("a wrong version is refused", _raises(
        lambda: Certificate.from_json(json.dumps(
            {**json.loads(cert.to_json()), "version": "other/9"}))))
    ck("a truncated certificate is refused", _raises(
        lambda: Certificate.from_json(json.dumps({"version": CERT_VERSION}))))
    ck("scheme summary names the axis", "in" in sch.summary())
    ck("transform summary shows the sign digest", "abc123" in tr.summary())
    ck("a no-transform spec says so", "no transform" in
       TransformSpec("none").summary())

    # ---- verification ----
    M = np.arange(32 * 8, dtype=np.float64).reshape(32, 8) / 100.0
    fwd = lambda x: x @ M
    c2 = build_certificate("demo/verify", sch, tr, fwd, 32)
    good = verify_certificate(c2, fwd, 32, sch, tr)
    ck("a correct runtime passes", good["verdict"] == "PASS")
    ck("passing means every probe is exact",
       all(r["digest_exact"] for r in good["probes"]))
    ck("a passing verdict says to load", good["action"] == "load")
    bad = verify_certificate(c2, lambda x: x @ (M * 1.05), 32, sch, tr)
    ck("a 5% error is refused", bad["verdict"] == "REFUSE")
    ck("refusal says DO NOT LOAD", "DO NOT LOAD" in bad["action"])
    ck("a misdeclared axis is caught by the declaration check",
       verify_certificate(c2, fwd, 32,
                          QuantScheme("ternary", 128, "out", "gguf-PTQ1_0",
                                      1.768, "fp16", False), tr
                          )["verdict"] == "REFUSE")
    ck("a misdeclared axis is diagnosed as WRONG_AXIS",
       any(f["fault"] == "WRONG_AXIS" for f in verify_certificate(
           c2, fwd, 32,
           QuantScheme("ternary", 128, "out", "gguf-PTQ1_0", 1.768, "fp16",
                       False), tr)["likely_faults"]))
    ck("a wrong sign digest is caught",
       verify_certificate(c2, fwd, 32, sch,
                          TransformSpec("hadamard", 1024, "deadbeef", True, True)
                          )["verdict"] == "REFUSE")
    ck("a forward that raises is refused, not ignored",
       verify_certificate(c2, lambda x: 1 / 0, 32)["verdict"] == "REFUSE")
    ck("a forward with the wrong row count is refused",
       verify_certificate(c2, lambda x: x[:1] @ M, 32)["verdict"] == "REFUSE")
    ck("building against a bad forward raises",
       _raises(lambda: build_certificate("x", sch, tr,
                                         lambda x: x[:1] @ M, 32)))
    # The point is that a certificate carries digests and scalars, never
    # weight data. Counting numbers is the way to check that; grepping for the
    # word "weights" only found the field name `folded_into_weights`.
    _blob = json.loads(c2.to_json())
    _nums = sum(len(r["stats"]) for r in _blob["results"])
    ck("certificate carries only digests and a handful of scalars",
       _nums <= 6 * len(_blob["results"]) and len(c2.to_json()) < 8000,
       f"{_nums} scalars, {len(c2.to_json())} bytes")

    # ---- the fault catalogue ----
    ck("every fault carries evidence and a signature",
       all({"what", "evidence", "signature"} <= set(v)
           for v in DETECTABLE_FAULTS.values()))
    ck("this project's own bug is in the catalogue",
       "sigil_t4_benchmark" in DETECTABLE_FAULTS["WRONG_AXIS"]["evidence"])
    ck("every fault is targeted by at least one probe",
       set(DETECTABLE_FAULTS) <=
       {t for p in STANDARD_PROBES for t in p.targets},
       str(set(DETECTABLE_FAULTS) -
           {t for p in STANDARD_PROBES for t in p.targets}))
    ck("diagnosis weights declarations above probes",
       diagnose([{"fault": "WRONG_AXIS"}],
                [{"targets": ["SKIPPED_TRANSFORM"]}])[0]["fault"] == "WRONG_AXIS")
    ck("diagnosis of nothing is empty", diagnose([], []) == [])

    # ---- the detection experiment ----
    ex = experiment_fault_detection(verbose=False)
    ck("the correct pipeline passes its own certificate", ex["correct_passes"])
    ck("every injected fault is caught by probes alone",
       ex["caught"] == ex["total"], f"{ex['caught']}/{ex['total']}")
    ck("certificate stays a few kilobytes",
       ex["certificate_bytes"] < 8000, f"{ex['certificate_bytes']} bytes")
    # Detection must hold away from the configuration the signatures came from.
    # Diagnosis is NOT asserted here, because it does not generalise and
    # asserting it would bake in an overclaim.
    for d_in, d_out, seed in ((128, 64, 11), (32, 16, 99)):
        e2 = experiment_fault_detection(d_in, d_out, seed, verbose=False)
        ck(f"detection holds at d={d_in}x{d_out} seed={seed}",
           e2["caught"] == e2["total"] and e2["correct_passes"],
           f"{e2['caught']}/{e2['total']}")
    ck("diagnosis is returned as a ranked hint with its evidence named",
       all("evidence" in r for r in diagnose([], [{"rel_l2": 0.1, "rel_std": 0.1,
                                                   "d_proj": 1.2}])))
    ck("a declaration mismatch outranks a probe inference",
       diagnose([{"fault": "WRONG_AXIS"}],
                [{"rel_l2": 1e-5, "rel_std": 1e-5, "d_proj": 3e-4}]
                )[0]["evidence"] == "declaration mismatch")
    ck("the axis/scale collision is declared, not hidden",
       any(r.get("not_separable_from") == ["TRANSPOSED_SCALE"]
           for r in diagnose([{"fault": "WRONG_AXIS"}], [])))
    ck("signature table covers every catalogued fault",
       set(FAULT_SIGNATURES) == set(DETECTABLE_FAULTS))
    ck("log distance is symmetric and zero on itself",
       _log_distance((1, 2, 3), (1, 2, 3)) == 0.0 and
       abs(_log_distance((1, 2, 3), (4, 5, 6))
           - _log_distance((4, 5, 6), (1, 2, 3))) < 1e-12)

    # ---- honesty ----
    ck("status is stated as untested on real silicon",
       "NOT DONE" in (__doc__ or "")
       and "real quantised checkpoint" in (__doc__ or ""))
    ck("the diagnosis limitation is stated in the docstring",
       "DIAGNOSIS DOES NOT" in (__doc__ or ""))
    ck("the verifier's own sign-blindness bug is recorded",
       "invariant under negation" in (__doc__ or ""))

    print("-" * 74)
    print("  SELF TEST")
    print("-" * 74)
    npass = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"   {detail}" if detail else ""))
        npass += ok
    print()
    print(f"  {npass}/{len(checks)} passed")
    return 0 if npass == len(checks) else 1


# ============================================================================ #
# SECTION 6 -- CLI
# ============================================================================ #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Quantisation certificates: prove a runtime is decoding "
                    "your model the way you quantised it.")
    p.add_argument("cmd", choices=["faults", "demo", "experiment", "selftest"])
    p.add_argument("--out", default="", help="write the demo certificate here")
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "faults":
        print("\n  FAULTS A QUANTISED MODEL FAILS SILENTLY ON\n")
        for name, f in DETECTABLE_FAULTS.items():
            print(f"  {name}")
            print(f"    what      {f['what']}")
            print(f"    evidence  {f['evidence']}")
            print(f"    signature {f['signature']}")
            probes = [pr.name for pr in STANDARD_PROBES if name in pr.targets]
            print(f"    probes    {', '.join(probes) or 'NONE'}")
            print()
        return 0

    if args.cmd == "experiment":
        print("\n  Injecting each fault into a synthetic pipeline.\n")
        ex = experiment_fault_detection(verbose=True)
        print(f"  certificate size: {ex['certificate_bytes']} bytes\n")
        return 0 if ex["caught"] == ex["total"] and ex["correct_passes"] else 1

    if args.cmd == "demo":
        if np is None:
            print("  numpy is required")
            return 2
        rng = np.random.Generator(np.random.PCG64(1))
        d_in, d_out = 128, 64
        W = rng.standard_normal((d_out, d_in)) / math.sqrt(d_in)
        signs = np.where(rng.random(d_in) < 0.5, -1.0, 1.0)
        H = _hadamard(d_in, signs)
        Wq = _quantise(W @ H, 4, 32, axis=1)
        scheme = QuantScheme("ternary", 32, "in", "gguf-PTQ1_0", 1.768,
                             "fp16", False)
        transform = TransformSpec("hadamard", d_in, digest_array(signs),
                                  True, True)
        cert = build_certificate(
            "demo/rotated-ternary", scheme, transform,
            lambda x: (x @ H) @ Wq.T, d_in,
            producer="quant_certificate demo",
            notes="Ships beside the weights. Verify before loading; a "
                  "quantised model that decodes wrongly does not crash.")
        text = cert.to_json()
        print(text if not args.out else f"  wrote {args.out}")
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(text)
        print(f"\n  {len(text)} bytes, {len(cert.probes)} probes, "
              f"scheme: {scheme.summary()}")
        print(f"  transform: {transform.summary()}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
