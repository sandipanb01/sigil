#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 dream_search.py -- stop paying GPU hours to re-learn what you already measured
================================================================================

SOURCE. `zhengkid/Dream-RSI`, "Recursive Self-Improvement through Evolving
Worlds" (Google, Google DeepMind, University of Maryland, University of
Virginia). Its idea, in one line:

    treat accumulated discovery history as a REPLAY SIMULATOR over the
    realized search space, and evaluate candidate policies by dreaming over
    that pool instead of paying for new rollouts.

Reported: 1.74x less discovery compute and 162x fewer calls on algorithm
engineering; 4 of 4 kernels improved at 2.09x higher performance at equal
budget on GPU kernel engineering. Code is listed as being prepared for release,
so what is implemented here is the METHOD, not their implementation.

--------------------------------------------------------------------------------
 WHY THIS PROJECT HAS THE EXACT PROBLEM IT DESCRIBES
--------------------------------------------------------------------------------
`sigil_t4_benchmark.py` sweeps 18 KV-quantisation configurations. On a free
Colab T4 that is roughly 90 minutes, and it has been run more than once. Every
run writes `results_t4.json`: a complete record of (bits, rotation, group) ->
perplexity over the realized search space.

That file IS a replay simulator. A different sweep policy -- which configs to
try, in what order, when to stop -- can be scored against it for free, because
the answers are already on disk. Nothing needs a GPU. The question "would a
cheaper sweep have found the same answer?" is answerable in milliseconds, and
until now it was being answered by running the sweep again.

--------------------------------------------------------------------------------
 THE HONEST LIMIT, WHICH IS THE WHOLE DESIGN
--------------------------------------------------------------------------------
Dreaming is only valid INSIDE the realized search space. A policy that asks for
a configuration the pool never evaluated cannot be replayed -- that is a MISS,
and a dream with misses is not evidence about anything.

Every result here carries its miss count, `reliable` is False the moment a
policy leaves the pool, and `compare_policies()` refuses to rank unreliable
runs against reliable ones. A replay simulator that quietly interpolates over
gaps would produce confident numbers about configurations nobody ran, which is
the same failure mode as reporting calibration error and calling it accuracy.

--------------------------------------------------------------------------------
 WHAT IT FOUND ON REAL DATA
--------------------------------------------------------------------------------
Run `python dream_search.py compare`. Two real pools ship with the module:

  COLAB_2026_09  the user's own T4 run: 18 configs, 3 hard failures, and a
                 genuine landscape spanning 14.95 to 44,774 perplexity.
                 **Produced by the pre-fix benchmark, whose Key axis was
                 wrong.** It is therefore valid as a SEARCH LANDSCAPE and
                 invalid as quantisation guidance. Used here only for the
                 former, and labelled at every point of use.
  TINY_CPU       an 18-config sweep on a randomly initialised model, run on
                 CPU. Correct code, meaningless objective: the landscape is
                 flat (557-591). It is here as the degenerate case, because a
                 search policy that only works on easy landscapes is not a
                 policy.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

__all__ = [
    "Config", "Outcome", "TracePool", "SearchSpace", "DreamResult",
    "COLAB_2026_09", "TINY_CPU", "POOLS", "POLICIES",
    "dream", "compare_policies", "pool_is_degenerate",
    "policy_exhaustive", "policy_random",
    "policy_coordinate", "policy_greedy_neighbour", "policy_granularity_first",
    "pool_from_results_json", "selftest",
]

MISS_TOLERANCE = 0
"""
Misses allowed before a dream is marked unreliable. Zero on purpose: one
unreplayable configuration means the policy's trajectory diverged from
anything the pool can speak to, and everything after it is fiction.
"""


# ============================================================================ #
# SECTION 1 -- configurations, outcomes, pools
# ============================================================================ #

@dataclass(frozen=True, order=True)
class Config:
    """One point in the KV-quantisation search space."""
    bits: int
    rotated: bool
    group: Optional[int]          # None == per-axis

    def __post_init__(self):
        if self.bits < 1 or self.bits > 16:
            raise ValueError(f"bits out of range: {self.bits}")
        if self.group is not None and self.group < 1:
            raise ValueError(f"group must be positive or None: {self.group}")

    @property
    def label(self) -> str:
        g = str(self.group) if self.group else "per-token"
        return f"INT{self.bits} rot={int(self.rotated)} group={g}"

    def as_key(self) -> Tuple[int, bool, int]:
        return (self.bits, self.rotated, self.group or 0)


@dataclass(frozen=True)
class Outcome:
    """What a real evaluation produced. `value` is lower-is-better."""
    value: Optional[float]        # None when the evaluation failed
    cost_s: float = 0.0
    error: str = ""

    @property
    def failed(self) -> bool:
        return self.value is None or not math.isfinite(self.value)


@dataclass
class TracePool:
    """
    Accumulated discovery history, as a replay simulator.

    This is deliberately a plain lookup with no interpolation, no surrogate
    model and no extrapolation. Everything it can answer, it measured.
    """
    name: str
    records: Dict[Config, Outcome]
    source: str = ""
    caveat: str = ""
    baseline: Optional[float] = None

    def __len__(self) -> int:
        return len(self.records)

    def get(self, cfg: Config) -> Optional[Outcome]:
        return self.records.get(cfg)

    @property
    def space(self) -> "SearchSpace":
        return SearchSpace.from_pool(self)

    def succeeded(self) -> Dict[Config, Outcome]:
        return {c: o for c, o in self.records.items() if not o.failed}

    def best(self) -> Optional[Tuple[Config, Outcome]]:
        ok = self.succeeded()
        if not ok:
            return None
        cfg = min(ok, key=lambda c: ok[c].value)
        return cfg, ok[cfg]

    def total_cost_s(self) -> float:
        return sum(o.cost_s for o in self.records.values())

    def summary(self) -> Dict[str, Any]:
        b = self.best()
        return {
            "pool": self.name, "n": len(self), "n_failed":
                sum(1 for o in self.records.values() if o.failed),
            "best": b[0].label if b else None,
            "best_value": round(b[1].value, 4) if b else None,
            "baseline": self.baseline,
            "spread": (round(max(o.value for o in self.succeeded().values()) /
                             min(o.value for o in self.succeeded().values()), 1)
                       if self.succeeded() else None),
            "caveat": self.caveat,
        }


@dataclass
class SearchSpace:
    """The axes a policy may move along. Derived from the pool, never assumed."""
    bits: Tuple[int, ...]
    rotations: Tuple[bool, ...]
    groups: Tuple[Optional[int], ...]

    @staticmethod
    def from_pool(pool: TracePool) -> "SearchSpace":
        if not pool.records:
            raise ValueError(f"pool {pool.name!r} is empty")
        return SearchSpace(
            bits=tuple(sorted({c.bits for c in pool.records}, reverse=True)),
            rotations=tuple(sorted({c.rotated for c in pool.records})),
            groups=tuple(sorted({c.group for c in pool.records},
                                key=lambda g: (g is not None, g or 0))),
        )

    def all_configs(self) -> List[Config]:
        return [Config(b, r, g) for b in self.bits
                for r in self.rotations for g in self.groups]

    def __len__(self) -> int:
        return len(self.bits) * len(self.rotations) * len(self.groups)


# ============================================================================ #
# SECTION 2 -- policies
#
# A policy is online: given what it has seen, it names the next configuration
# to evaluate, or returns None to stop. That signature is what makes it
# replayable -- the policy never sees anything a real run would not have seen
# at the same point.
# ============================================================================ #

History = List[Tuple[Config, Outcome]]
Policy = Callable[[History, SearchSpace], Optional[Config]]


def _check_policy_args(history, space) -> None:
    """
    Every policy's first line. A string passed as the history iterates as
    characters and fails three frames later with an AttributeError about
    `.bits` -- the "confused" failure class stress_all.py counts as a bug,
    found by fuzzing argument PAIRS over the full hostile set.
    """
    if not isinstance(space, SearchSpace):
        raise TypeError(f"space must be a SearchSpace, got {type(space).__name__}")
    if not isinstance(history, (list, tuple)):
        raise TypeError(f"history must be a list, got {type(history).__name__}")
    for item in history:
        if (not isinstance(item, tuple) or len(item) != 2
                or not isinstance(item[0], Config) or not isinstance(item[1], Outcome)):
            raise TypeError("history entries must be (Config, Outcome) pairs")


def policy_exhaustive(history: History, space: SearchSpace) -> Optional[Config]:
    """What the benchmark does today: try everything, in a fixed order."""
    _check_policy_args(history, space)
    seen = {c for c, _ in history}
    for cfg in space.all_configs():
        if cfg not in seen:
            return cfg
    return None


def policy_random(seed: int = 0) -> Policy:
    """Uniform random without replacement. The control every policy must beat."""
    def fn(history: History, space: SearchSpace) -> Optional[Config]:
        _check_policy_args(history, space)
        seen = {c for c, _ in history}
        rest = [c for c in space.all_configs() if c not in seen]
        if not rest:
            return None
        rng = random.Random(seed * 1000003 + len(history))
        return rng.choice(rest)
    return fn


def policy_coordinate(history: History, space: SearchSpace) -> Optional[Config]:
    """
    Coordinate descent over the three axes: hold two fixed, sweep the third,
    keep the winner, move on. Classic, cheap, and it is what a careful human
    does by hand.

    It STOPS when its plan is complete. The first version of every structured
    policy here ended with a fallback loop over whatever was left, which made
    all five of them evaluate all 18 configurations and the whole comparison
    vacuous -- "fewer calls" was 1.0x across the board because nobody ever
    stopped. A search policy that cannot stop is a sweep with extra steps.
    """
    _check_policy_args(history, space)
    seen = {c for c, _ in history}
    ok = [(c, o) for c, o in history if not o.failed]
    cur = Config(space.bits[0], space.rotations[0], space.groups[0])
    if ok:
        cur = min(ok, key=lambda co: co[1].value)[0]
    for axis in ("group", "rotated", "bits"):
        if axis == "group":
            cands = [Config(cur.bits, cur.rotated, g) for g in space.groups]
        elif axis == "rotated":
            cands = [Config(cur.bits, r, cur.group) for r in space.rotations]
        else:
            cands = [Config(b, cur.rotated, cur.group) for b in space.bits]
        for c in cands:
            if c not in seen:
                return c
    return None


def policy_greedy_neighbour(history: History,
                            space: SearchSpace) -> Optional[Config]:
    """
    Start at the widest, simplest configuration and walk to the best unseen
    neighbour of the best point so far. One axis changes per step.
    """
    _check_policy_args(history, space)
    seen = {c for c, _ in history}
    if not history:
        return Config(space.bits[0], space.rotations[0], space.groups[0])
    ok = [(c, o) for c, o in history if not o.failed]
    if not ok:
        # Everything tried so far crashed; step along one axis rather than
        # giving up, but stay inside the plan.
        for c in space.all_configs():
            if c not in seen:
                return c
        return None
    cur = min(ok, key=lambda co: co[1].value)[0]
    bi, gi = space.bits.index(cur.bits), space.groups.index(cur.group)
    neigh: List[Config] = []
    for d in (-1, 1):
        if 0 <= bi + d < len(space.bits):
            neigh.append(Config(space.bits[bi + d], cur.rotated, cur.group))
        if 0 <= gi + d < len(space.groups):
            neigh.append(Config(cur.bits, cur.rotated, space.groups[gi + d]))
    for r in space.rotations:
        if r != cur.rotated:
            neigh.append(Config(cur.bits, r, cur.group))
    for c in neigh:
        if c not in seen:
            return c
    return None          # no improving neighbour left: stop


def policy_granularity_first(history: History,
                             space: SearchSpace) -> Optional[Config]:
    """
    This project's own domain finding used as a search prior: scale granularity
    dominates rotation choice. So resolve GROUP first at the widest bit width,
    then bits, and only then spend evaluations on rotation.

    This is the interesting one, because it is the point of the exercise. A
    finding is worth something if it makes the next search cheaper, and the
    pool says whether it does -- for free.
    """
    _check_policy_args(history, space)
    seen = {c for c, _ in history}
    b0, r0 = space.bits[0], space.rotations[0]
    for g in space.groups:                       # stage 1: granularity
        c = Config(b0, r0, g)
        if c not in seen:
            return c
    ok = [(c, o) for c, o in history if not o.failed]
    g_best = (min(ok, key=lambda co: co[1].value)[0].group
              if ok else space.groups[0])
    for b in space.bits:                         # stage 2: bit width
        c = Config(b, r0, g_best)
        if c not in seen:
            return c
    ok = [(c, o) for c, o in history if not o.failed]
    b_best = min(ok, key=lambda co: co[1].value)[0].bits if ok else b0
    for r in space.rotations:                    # stage 3: rotation
        c = Config(b_best, r, g_best)
        if c not in seen:
            return c
    return None          # the three stages are done: stop


POLICIES: Dict[str, Policy] = {
    "exhaustive": policy_exhaustive,
    "random": policy_random(0),
    "coordinate": policy_coordinate,
    "greedy": policy_greedy_neighbour,
    "granularity-first": policy_granularity_first,
}


# ============================================================================ #
# SECTION 3 -- dreaming
# ============================================================================ #

@dataclass
class DreamResult:
    policy: str
    pool: str
    evaluations: int
    cost_s: float
    best: Optional[Config]
    best_value: Optional[float]
    evals_to_best: Optional[int]
    misses: List[Config]
    reliable: bool
    pool_best: Optional[Config]
    pool_best_value: Optional[float]
    found_pool_best: bool
    regret: Optional[float]
    note: str = ""

    def row(self) -> Dict[str, Any]:
        return {
            "policy": self.policy, "evals": self.evaluations,
            "evals_to_best": self.evals_to_best,
            "found_optimum": self.found_pool_best,
            "regret": (round(self.regret, 4) if self.regret is not None else None),
            "misses": len(self.misses), "reliable": self.reliable,
        }


def dream(policy: Policy, pool: TracePool, budget: Optional[int] = None,
          policy_name: str = "policy",
          stop_on_miss: bool = True) -> DreamResult:
    """
    Replay a policy over a pool. No evaluation is performed; every answer comes
    from something that was already measured.

    `budget` caps evaluations. A policy that requests a configuration the pool
    does not contain records a MISS -- the dream cannot speak for it, and by
    default the replay stops there rather than pretending to continue.
    """
    if not isinstance(pool, TracePool):
        raise TypeError(f"pool must be a TracePool, got {type(pool).__name__}")
    if not callable(policy):
        raise TypeError("policy must be callable")
    if budget is not None and budget < 1:
        raise ValueError("budget must be >= 1")
    space = pool.space
    history: History = []
    misses: List[Config] = []
    cost = 0.0
    best_cfg: Optional[Config] = None
    best_val: Optional[float] = None
    evals_to_best: Optional[int] = None
    guard = len(space) * 4 + 16          # a policy that never returns None

    for step in range(guard):
        if budget is not None and len(history) >= budget:
            break
        try:
            nxt = policy(list(history), space)
        except Exception as exc:
            return DreamResult(policy_name, pool.name, len(history), cost,
                               best_cfg, best_val, evals_to_best, misses, False,
                               *_pool_best(pool), False, None,
                               f"policy raised {type(exc).__name__}: {exc}")
        if nxt is None:
            break
        if not isinstance(nxt, Config):
            return DreamResult(policy_name, pool.name, len(history), cost,
                               best_cfg, best_val, evals_to_best, misses, False,
                               *_pool_best(pool), False, None,
                               f"policy returned {type(nxt).__name__}, not Config")
        if any(nxt == c for c, _ in history):
            return DreamResult(policy_name, pool.name, len(history), cost,
                               best_cfg, best_val, evals_to_best, misses, False,
                               *_pool_best(pool), False, None,
                               f"policy re-requested {nxt.label}; a replay "
                               f"cannot charge for it twice")
        out = pool.get(nxt)
        if out is None:
            misses.append(nxt)
            if stop_on_miss:
                break
            continue
        history.append((nxt, out))
        cost += out.cost_s
        if not out.failed and (best_val is None or out.value < best_val):
            best_val, best_cfg, evals_to_best = out.value, nxt, len(history)

    pb, pbv = _pool_best(pool)
    reliable = len(misses) <= MISS_TOLERANCE
    found = bool(pb is not None and best_cfg == pb)
    regret = (best_val - pbv if (best_val is not None and pbv is not None)
              else None)
    note = ""
    if not reliable:
        note = (f"{len(misses)} configuration(s) outside the realized search "
                f"space; this dream is not evidence.")
    return DreamResult(policy_name, pool.name, len(history), cost, best_cfg,
                       best_val, evals_to_best, misses, reliable, pb, pbv,
                       found, regret, note)


def _mean_result(runs: Sequence[DreamResult], name: str,
                 pool: TracePool) -> DreamResult:
    """Average a stochastic policy over its seeds, rounding calls up."""
    ok = [r for r in runs if r.reliable]
    if not ok:
        return runs[0]
    n = len(ok)
    to_best = [r.evals_to_best for r in ok if r.evals_to_best]
    pb, pbv = _pool_best(pool)
    return DreamResult(
        policy=name, pool=pool.name,
        evaluations=int(round(sum(r.evaluations for r in ok) / n)),
        cost_s=sum(r.cost_s for r in ok) / n,
        best=None, best_value=None,
        evals_to_best=(int(math.ceil(sum(to_best) / len(to_best)))
                       if to_best else None),
        misses=[], reliable=True, pool_best=pb, pool_best_value=pbv,
        found_pool_best=all(r.found_pool_best for r in ok),
        regret=sum(r.regret for r in ok if r.regret is not None) / n,
        note=f"mean of {n} seeds")


def pool_is_degenerate(pool: TracePool) -> Dict[str, Any]:
    """
    Is a policy comparison on this pool worth anything?

    Two ways it is not. If the optimum happens to sit at the very first
    configuration every policy tries, they all "find it in 1" and the ranking
    measures the ordering, not the policies. If the landscape is flat, any
    configuration is as good as any other and stopping early costs nothing.

    COLAB_2026_09 trips the first of these: its optimum IS the first config in
    the natural order. That is reported rather than hidden, because a
    comparison that looks decisive and is not is worse than no comparison.
    """
    if not isinstance(pool, TracePool):
        raise TypeError(f"pool must be a TracePool, got {type(pool).__name__}")
    space = pool.space
    first = space.all_configs()[0]
    b = pool.best()
    spread = pool.summary()["spread"]
    reasons = []
    if b and b[0] == first:
        reasons.append("the optimum is the FIRST configuration in the natural "
                       "order, so every policy that starts there finds it "
                       "immediately and the ranking measures ordering")
    if spread is not None and spread < 1.5:
        reasons.append(f"the landscape is nearly flat (spread {spread}x), so "
                       f"stopping early costs almost nothing")
    return {"degenerate": bool(reasons), "reasons": reasons,
            "optimum_is_first": bool(b and b[0] == first), "spread": spread}


def _pool_best(pool: TracePool) -> Tuple[Optional[Config], Optional[float]]:
    b = pool.best()
    return (b[0], b[1].value) if b else (None, None)


def compare_policies(pool: TracePool,
                     policies: Optional[Dict[str, Policy]] = None,
                     budget: Optional[int] = None,
                     random_seeds: int = 8) -> Dict[str, Any]:
    """
    Score every policy on one pool. Unreliable runs are reported but never
    ranked against reliable ones.

    `random` is averaged over `random_seeds` draws, because a single random
    run is a coin toss and reporting one would be the cheapest possible way to
    mislead yourself.
    """
    if not isinstance(pool, TracePool):
        raise TypeError(f"pool must be a TracePool, got {type(pool).__name__}")
    pols = policies or POLICIES
    results = []
    for name, fn in pols.items():
        if name == "random":
            runs = [dream(policy_random(k), pool, budget, name)
                    for k in range(random_seeds)]
            results.append(_mean_result(runs, name, pool))
        else:
            results.append(dream(fn, pool, budget, name))
    reliable = [r for r in results if r.reliable]
    baseline = next((r for r in reliable if r.policy == "exhaustive"), None)
    rows = []
    for r in reliable:
        # "Fewer calls" is TOTAL evaluations against the exhaustive sweep --
        # the quantity that costs GPU hours. An earlier version divided
        # evals_to_best instead, which reported 1.0x for a policy that used 4
        # evaluations against a sweep that used 18, because both happened to
        # stumble on the optimum at step 1.
        speedup = None
        if baseline and baseline.evaluations and r.evaluations:
            speedup = baseline.evaluations / r.evaluations
        d = r.row()
        d["fewer_calls"] = (round(speedup, 2) if speedup else None)
        d["cost_of_stopping"] = (None if d["found_optimum"] else
                                 "MISSED the optimum")
        rows.append(d)
    rows.sort(key=lambda d: (not d["found_optimum"],
                             -(d["fewer_calls"] or 0)))
    return {
        "pool": pool.name,
        "pool_summary": pool.summary(),
        "degeneracy": pool_is_degenerate(pool),
        "rows": rows,
        "unreliable": [{"policy": r.policy, "misses": len(r.misses),
                        "note": r.note} for r in results if not r.reliable],
        "caveat": pool.caveat,
    }


# ============================================================================ #
# SECTION 4 -- real pools
# ============================================================================ #

def _pool(name: str, rows: Sequence[Tuple[int, int, Optional[int], Any]],
          source: str, caveat: str = "",
          baseline: Optional[float] = None) -> TracePool:
    recs: Dict[Config, Outcome] = {}
    for bits, rot, group, val in rows:
        cfg = Config(bits, bool(rot), group)
        recs[cfg] = (Outcome(None, 0.0, str(val)) if not isinstance(val, (int, float))
                     else Outcome(float(val)))
    return TracePool(name, recs, source, caveat, baseline)


COLAB_2026_09 = _pool(
    "COLAB_2026_09",
    [(4, 0, None, 14.9539), (4, 0, 128, "RuntimeError: view size"),
     (4, 0, 32, 64.5982),
     (4, 1, None, 68.2121), (4, 1, 128, 39.7911), (4, 1, 32, 49.4487),
     (3, 0, None, 19.6981), (3, 0, 128, "RuntimeError: view size"),
     (3, 0, 32, 239.4172),
     (3, 1, None, 61.8901), (3, 1, 128, 625.8235), (3, 1, 32, 2138.9822),
     (2, 0, None, 418.7574), (2, 0, 128, "RuntimeError: view size"),
     (2, 0, 32, 6893.0904),
     (2, 1, None, 697.9258), (2, 1, 128, 20590.7127), (2, 1, 32, 44774.7098)],
    source="Qwen2.5-0.5B-Instruct on a Colab T4, WikiText-2, 2026-09.",
    caveat="PRODUCED BY THE PRE-FIX BENCHMARK. Its Key axis was wrong, so "
           "these perplexities are NOT valid quantisation guidance and the "
           "three group-128 entries are the crash that motivated the fix. The "
           "pool is used here only as a SEARCH LANDSCAPE -- which it is a "
           "perfectly good one of, spanning 14.95 to 44,774.",
    baseline=14.1003)

TINY_CPU = _pool(
    "TINY_CPU",
    [(4, 0, None, 579.2197), (4, 0, 32, 579.1869), (4, 0, 16, 578.0971),
     (4, 1, None, 572.2019), (4, 1, 32, 572.1564), (4, 1, 16, 572.1183),
     (3, 0, None, 557.5045), (3, 0, 32, 557.7943), (3, 0, 16, 560.8008),
     (3, 1, None, 567.8972), (3, 1, 32, 567.2185), (3, 1, 16, 567.7660),
     (2, 0, None, 580.6989), (2, 0, 32, 582.2602), (2, 0, 16, 568.2231),
     (2, 1, None, 591.0276), (2, 1, 32, 588.9234), (2, 1, 16, 586.8530)],
    source="Randomly initialised 4-layer GQA model, CPU, correct code.",
    caveat="Correct code, MEANINGLESS objective: random weights, so the "
           "landscape is flat (557-591, a spread of 1.06x). Included as the "
           "degenerate case -- a policy that only wins on easy landscapes is "
           "not a policy.",
    baseline=575.4038)

POOLS: Dict[str, TracePool] = {p.name: p for p in (COLAB_2026_09, TINY_CPU)}


def pool_from_results_json(path: str, name: str = "") -> TracePool:
    """
    Build a pool from a real `results_t4.json`. This is the path that matters:
    every benchmark run the user has already paid for becomes a simulator they
    can test policies against for nothing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no results file at {p}")
    try:
        blob = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
    sweep = blob.get("kv_sweep")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError(f"{p} has no kv_sweep rows")
    recs: Dict[Config, Outcome] = {}
    skipped = 0
    for row in sweep:
        try:
            cfg = Config(int(row["bits"]), bool(row["rotated"]),
                         row.get("group") or None)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        ppl = row.get("ppl")
        recs[cfg] = (Outcome(float(ppl)) if isinstance(ppl, (int, float))
                     else Outcome(None, 0.0, str(row.get("error", "failed"))))
    if not recs:
        raise ValueError(f"{p} contained no usable sweep rows")
    return TracePool(
        name or f"file:{p.name}", recs,
        source=f"{p} (model {blob.get('model', '?')})",
        caveat=(f"{skipped} malformed row(s) skipped." if skipped else ""),
        baseline=blob.get("ppl_fp16"))


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

    # ---- Config / Outcome hygiene ----
    ck("config is hashable and ordered",
       len({Config(4, True, 32), Config(4, True, 32)}) == 1
       and Config(2, False, None) < Config(4, False, None))
    ck("config label names the axis",
       "group=per-token" in Config(4, False, None).label)
    ck("absurd bit width is rejected", _raises(lambda: Config(0, False, None)))
    ck("bit width above 16 is rejected", _raises(lambda: Config(99, False, None)))
    ck("negative group is rejected", _raises(lambda: Config(4, False, -8)))
    ck("None group is legal", Config(4, False, None).group is None)
    ck("failed outcome is detected", Outcome(None, 0, "boom").failed)
    ck("NaN counts as failed", Outcome(float("nan")).failed)
    ck("inf counts as failed", Outcome(float("inf")).failed)
    ck("a real value is not failed", not Outcome(12.5).failed)

    # ---- pools ----
    for pool in POOLS.values():
        s = pool.summary()
        ck(f"{pool.name} has 18 records", len(pool) == 18, str(len(pool)))
        ck(f"{pool.name} reports a best", s["best"] is not None)
        ck(f"{pool.name} carries a caveat", bool(pool.caveat))
        ck(f"{pool.name} derives a space of 18", len(pool.space) == 18,
           str(len(pool.space)))
    ck("the Colab pool records its 3 crashes",
       COLAB_2026_09.summary()["n_failed"] == 3)
    ck("the Colab pool is flagged as pre-fix",
       "PRE-FIX" in COLAB_2026_09.caveat)
    ck("the Colab landscape is genuinely wide",
       COLAB_2026_09.summary()["spread"] > 100,
       str(COLAB_2026_09.summary()["spread"]))
    ck("the tiny pool is flagged as degenerate",
       "MEANINGLESS" in TINY_CPU.caveat)
    ck("the tiny landscape is flat",
       TINY_CPU.summary()["spread"] < 1.2, str(TINY_CPU.summary()["spread"]))
    ck("an empty pool cannot yield a space",
       _raises(lambda: TracePool("empty", {}).space))
    ck("an all-failed pool has no best",
       TracePool("x", {Config(4, False, None): Outcome(None)}).best() is None)

    # ---- dreaming ----
    for name, fn in POLICIES.items():
        r = dream(fn, COLAB_2026_09, policy_name=name)
        ck(f"{name} stays inside the realized space", r.reliable,
           f"{len(r.misses)} misses")
        ck(f"{name} finds the pool optimum given the full budget",
           r.found_pool_best, r.best.label if r.best else "none")
        ck(f"{name} never exceeds the pool size", r.evaluations <= 18,
           str(r.evaluations))
        ck(f"{name} has non-zero regret only if it missed the optimum",
           (abs(r.regret) < 1e-9) == r.found_pool_best)

    ex = dream(policy_exhaustive, COLAB_2026_09, policy_name="exhaustive")
    ck("exhaustive evaluates the whole space", ex.evaluations == 18)
    ck("the Colab optimum is INT4 per-token, no rotation",
       ex.best == Config(4, False, None), ex.best.label if ex.best else "-")

    gf = dream(policy_granularity_first, COLAB_2026_09,
               policy_name="granularity-first")
    ck("granularity-first reaches the optimum sooner than exhaustive",
       (gf.evals_to_best or 99) <= (ex.evals_to_best or 99),
       f"{gf.evals_to_best} vs {ex.evals_to_best}")

    # ---- the honesty guard ----
    def off_pool(history, space):
        return Config(7, False, 999) if not history else None
    bad = dream(off_pool, COLAB_2026_09, policy_name="off-pool")
    ck("a policy leaving the pool is marked UNRELIABLE", not bad.reliable)
    ck("the miss is recorded", len(bad.misses) == 1)
    ck("an unreliable dream says it is not evidence",
       "not evidence" in bad.note)
    ck("an unreliable dream is excluded from the ranking",
       "off-pool" not in [r["policy"] for r in compare_policies(
           COLAB_2026_09, {**POLICIES, "off-pool": off_pool})["rows"]])
    ck("but it is still reported",
       "off-pool" in [u["policy"] for u in compare_policies(
           COLAB_2026_09, {**POLICIES, "off-pool": off_pool})["unreliable"]])

    # ---- adversarial policies ----
    ck("a policy that raises is caught, not propagated",
       not dream(lambda h, s: 1 / 0, COLAB_2026_09, policy_name="boom").reliable)
    ck("the raising policy's error is reported",
       "ZeroDivisionError" in dream(lambda h, s: 1 / 0, COLAB_2026_09,
                                    policy_name="boom").note)
    ck("a policy returning the wrong type is caught",
       "not Config" in dream(lambda h, s: "INT4", COLAB_2026_09,
                             policy_name="bad-type").note)
    ck("a policy repeating a configuration is caught",
       "twice" in dream(lambda h, s: Config(4, False, None), COLAB_2026_09,
                        policy_name="repeat").note)
    ck("a policy that never stops is bounded by the guard",
       dream(lambda h, s: Config(4, False, None), COLAB_2026_09,
             policy_name="loop").evaluations <= 18)
    ck("an immediately-None policy yields an empty dream",
       dream(lambda h, s: None, COLAB_2026_09, policy_name="null"
             ).evaluations == 0)

    # ---- budgets ----
    b3 = dream(policy_exhaustive, COLAB_2026_09, budget=3, policy_name="ex3")
    ck("a budget is respected exactly", b3.evaluations == 3)
    ck("a budget of 1 works",
       dream(policy_exhaustive, COLAB_2026_09, budget=1).evaluations == 1)
    ck("a zero budget is rejected",
       _raises(lambda: dream(policy_exhaustive, COLAB_2026_09, budget=0)))
    ck("a negative budget is rejected",
       _raises(lambda: dream(policy_exhaustive, COLAB_2026_09, budget=-5)))
    ck("a budget beyond the space is harmless",
       dream(policy_exhaustive, COLAB_2026_09, budget=9999).evaluations == 18)

    # ---- comparison ----
    cmp = compare_policies(COLAB_2026_09)
    ck("every policy is ranked", len(cmp["rows"]) == len(POLICIES))
    ck("the ranking puts optimum-finders first", cmp["rows"][0]["found_optimum"])
    ck("the comparison carries the pool's caveat", "PRE-FIX" in cmp["caveat"])
    ck("fewer-calls is measured on TOTAL evaluations, not time-to-best",
       next(r for r in cmp["rows"] if r["policy"] == "greedy")["fewer_calls"]
       > 1.5,
       str(next(r for r in cmp["rows"] if r["policy"] == "greedy")["fewer_calls"]))
    ck("exhaustive is exactly 1.0x against itself",
       next(r for r in cmp["rows"] if r["policy"] == "exhaustive"
            )["fewer_calls"] == 1.0)
    ck("structured policies stop instead of exhausting the space",
       all(next(r for r in cmp["rows"] if r["policy"] == n)["evals"] < 18
           for n in ("coordinate", "greedy", "granularity-first")))
    deg = pool_is_degenerate(COLAB_2026_09)
    ck("the Colab pool is flagged as unable to rank policies fairly",
       deg["degenerate"] and deg["optimum_is_first"])
    ck("the flat pool is flagged degenerate too",
       pool_is_degenerate(TINY_CPU)["degenerate"])
    ck("a policy that stops early and misses is reported as missing",
       any(r.get("cost_of_stopping") for r in compare_policies(TINY_CPU)["rows"]))
    flat = compare_policies(TINY_CPU)
    ck("the degenerate pool still ranks cleanly",
       len(flat["rows"]) == len(POLICIES)
       and all(r["reliable"] for r in flat["rows"]))

    # ---- results_t4.json ingestion ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "results_t4.json"
        good.write_text(json.dumps({
            "model": "demo", "ppl_fp16": 10.0,
            "kv_sweep": [{"bits": 4, "rotated": False, "group": None, "ppl": 11.0},
                         {"bits": 3, "rotated": True, "group": 32, "ppl": 13.0},
                         {"bits": 2, "rotated": False, "group": 32,
                          "ppl": None, "error": "OOM"}]}))
        pl = pool_from_results_json(str(good))
        ck("a real results file loads", len(pl) == 3)
        ck("its baseline is carried through", pl.baseline == 10.0)
        ck("a failed row stays failed", pl.summary()["n_failed"] == 1)
        ck("its best is the lowest perplexity", pl.best()[1].value == 11.0)

        bad_json = Path(td) / "bad.json"
        bad_json.write_text("{not json")
        ck("malformed JSON is rejected clearly",
           _raises(lambda: pool_from_results_json(str(bad_json))))
        empty = Path(td) / "empty.json"
        empty.write_text(json.dumps({"kv_sweep": []}))
        ck("an empty sweep is rejected",
           _raises(lambda: pool_from_results_json(str(empty))))
        noswp = Path(td) / "nosweep.json"
        noswp.write_text(json.dumps({"model": "x"}))
        ck("a file with no sweep is rejected",
           _raises(lambda: pool_from_results_json(str(noswp))))
        ragged = Path(td) / "ragged.json"
        ragged.write_text(json.dumps({"kv_sweep": [
            {"bits": 4, "rotated": False, "group": None, "ppl": 11.0},
            {"nonsense": True}]}))
        rp = pool_from_results_json(str(ragged))
        ck("malformed rows are skipped, not fatal", len(rp) == 1)
        ck("skipped rows are disclosed", "skipped" in rp.caveat)
        ck("a missing file raises",
           _raises(lambda: pool_from_results_json(str(Path(td) / "nope.json"))))

    # ---- honesty ----
    ck("the source is attributed in the docstring",
       "Dream-RSI" in (__doc__ or "") and "zhengkid" in (__doc__ or ""))
    ck("the realized-space limit is stated",
       "realized search space" in (__doc__ or ""))
    ck("the pre-fix provenance of the Colab pool is stated in the docstring",
       "pre-fix benchmark" in (__doc__ or ""))

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
        description="Replay-simulator search over recorded benchmark history, "
                    "after Dream-RSI.")
    p.add_argument("cmd", choices=["pools", "compare", "dream", "selftest"])
    p.add_argument("--pool", default="COLAB_2026_09")
    p.add_argument("--results", default="",
                   help="a real results_t4.json to use as the pool")
    p.add_argument("--policy", default="granularity-first",
                   choices=sorted(POLICIES))
    p.add_argument("--budget", type=int, default=None)
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    def get_pool() -> TracePool:
        if args.results:
            return pool_from_results_json(args.results)
        if args.pool not in POOLS:
            raise SystemExit(f"unknown pool {args.pool!r}; "
                             f"known: {sorted(POOLS)}")
        return POOLS[args.pool]

    if args.cmd == "pools":
        print()
        for pool in POOLS.values():
            s = pool.summary()
            print(f"  {s['pool']}   {s['n']} configs, {s['n_failed']} failed, "
                  f"spread {s['spread']}x")
            print(f"    source  {pool.source}")
            print(f"    best    {s['best']}  ->  {s['best_value']}"
                  f"   (fp16 baseline {s['baseline']})")
            for line in _wrap(pool.caveat, 68):
                print(f"    {line}")
            print()
        return 0

    pool = get_pool()

    if args.cmd == "dream":
        r = dream(POLICIES[args.policy], pool, args.budget, args.policy)
        print(f"\n  {args.policy} on {r.pool}")
        print(f"  evaluations        {r.evaluations}")
        print(f"  best found         {r.best.label if r.best else '-'}"
              f"   {r.best_value}")
        print(f"  evaluations to it  {r.evals_to_best}")
        print(f"  pool optimum       "
              f"{r.pool_best.label if r.pool_best else '-'}   {r.pool_best_value}")
        print(f"  found the optimum  {r.found_pool_best}")
        print(f"  reliable           {r.reliable}"
              + (f"   ({len(r.misses)} misses)" if r.misses else ""))
        if r.note:
            print(f"  note               {r.note}")
        print()
        return 0

    if args.cmd == "compare":
        c = compare_policies(pool, budget=args.budget)
        print(f"\n  {c['pool']}  --  {c['pool_summary']['n']} configs, "
              f"spread {c['pool_summary']['spread']}x")
        print(f"  optimum: {c['pool_summary']['best']}  "
              f"-> {c['pool_summary']['best_value']}\n")
        deg = c["degeneracy"]
        if deg["degenerate"]:
            print("  WARNING -- this pool cannot rank policies fairly:")
            for why in deg["reasons"]:
                for line in _wrap(why, 68):
                    print(f"    {line}")
            print()
        print(f"  {'policy':<20}{'evals':>7}{'to best':>9}{'found':>7}"
              f"{'fewer calls':>13}")
        for r in c["rows"]:
            fc = f"{r['fewer_calls']}x" if r["fewer_calls"] else "-"
            print(f"  {r['policy']:<20}{r['evals']:>7}{str(r['evals_to_best']):>9}"
                  f"{('yes' if r['found_optimum'] else 'NO'):>7}{fc:>13}")
        if c["unreliable"]:
            print()
            for u in c["unreliable"]:
                print(f"  UNRELIABLE  {u['policy']}: {u['note']}")
        print()
        for line in _wrap("Every number above came from history already on "
                          "disk. No GPU was used, and no configuration was "
                          "evaluated that had not already been paid for.", 72):
            print(f"  {line}")
        if c["caveat"]:
            print()
            for line in _wrap("POOL CAVEAT: " + c["caveat"], 72):
                print(f"  {line}")
        print()
        return 0
    return 1


def _wrap(text: str, width: int) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width) if text else []


if __name__ == "__main__":
    sys.exit(main())
