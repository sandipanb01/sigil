#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 audit_claims.py -- check the submission against the code
================================================================================

    python audit_claims.py

The headline numbers in SUBMISSION.md, README.md, MANIFEST.md, the deck and
the brief are asserted here against what the code actually returns, and the
documents are held to the counts the code reports. Not every figure in prose
is covered -- the fact-check of 2026-09-22 found five that were not, and each
now is. Run it before submitting and before quoting any figure out loud.

This exists because the most expensive error in this project was not a crash.
It was a catalogue entry claiming a 27B model at 1-bit was "~3.4 GB, fits a
phone" -- 12% under the real file -- guarded by a self-test whose threshold had been
calibrated to the invented number. The test passed and protected the error for
weeks. A claim is only checked when something independent recomputes it from
the source of truth.

Licence: Apache-2.0.
"""
import math
import pathlib
import re
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import snapdragon_engine as E, residual_cascade as R
import quant_certificate as Q, dream_search as D
import hp_snapdragon as HP
import roofline as RL
import aihub_workbench as W

HERE = pathlib.Path(__file__).resolve().parent

print("-" * 74)
print("  CLAIM AUDIT")
print("-" * 74)

rows = []


def ck(claim, ok, got=""):
    rows.append((claim, bool(ok), str(got)))

# --- container rule ---
q1 = E.CONTAINER_MEASUREMENTS["bonsai-27b Q1_0 (gguf)"]
mlx = E.CONTAINER_MEASUREMENTS["bonsai-27b MLX 1-bit"]
ck("Q1_0 = 3,803,452,480 B", q1[0] == 3_803_452_480, q1[0])
ck("MLX 1-bit = 5,129,115,752 B", mlx[0] == 5_129_115_752, mlx[0])
ck("Q1_0 = 1.131 bpw", abs(E.container_tax("bonsai-27b Q1_0 (gguf)")["effective_bpw"]-1.131)<0.001)
ck("MLX = 1.525 bpw", abs(E.container_tax("bonsai-27b MLX 1-bit")["effective_bpw"]-1.525)<0.001)
ratio = mlx[0]/q1[0]
ck("35% more bytes", abs(ratio-1.35)<0.005, f"{ratio:.4f}")
ck("PTQ1_0 = 1.768 bpw", abs(E.container_tax("bonsai-2-27b PTQ1_0 (gguf)")["effective_bpw"]-1.768)<0.001)
ck("PQ2_0 = 2.143 bpw", abs(E.container_tax("bonsai-2-27b PQ2_0 (gguf)")["effective_bpw"]-2.143)<0.001)
v = E.validate_container_rule()
ck("8/8 correct", v["n_correct"]==8 and v["n"]==8, f"{v['n_correct']}/{v['n']}")
ck("crossover 1008-1792", v["observed_bracket_gb_s"]==(1008.0,1792.0), v["observed_bracket_gb_s"])
ck("unpack efficiency 0.46", abs(v["unpack_efficiency"]-0.46)<0.005, v["unpack_efficiency"])
socs=[E.predict_container(s.mem_bandwidth_gbs) for s in E.SOC_DB.values() if s.mem_bandwidth_gbs]
below=[p["orders_below_bracket"] for p in socs]
ck("Snapdragon 4.4-19.7x below", abs(min(below)-4.4)<0.05 and abs(max(below)-19.7)<0.05, f"{min(below)}-{max(below)}")
ck("all Snapdragon -> PTQ1_0", all(p["container"]=="PTQ1_0" and p["confident"] for p in socs))

# --- rotation / vision tower ---
rp=E.ROTATION_PRECEDENT
ck("metadata 297,903 B", rp["metadata_bytes"]==297_903)
ck("metadata 0.0035%", abs(rp["metadata_fraction_pct"]-0.0035)<1e-6)
ck("widths 5120/6144/17408", rp["declared_widths"]==(5120,6144,17408))
ck("= 5,6,17 blocks of 1024", all(w%1024==0 for w in rp["declared_widths"]) and
   [w//1024 for w in rp["declared_widths"]]==[5,6,17])
ck("Qwen3-4B needs block 512", E.hadamard_block_feasible((2560,9728),1024)["largest_feasible_block"]==512)
vt=E.vision_tower_budget()
ck("tower 1.70% of params", abs(vt["tower_param_share_pct"]-1.70)<0.01, vt["tower_param_share_pct"])
ck("tower 13.5% of bytes", abs(vt["tower_byte_share_pct"]-13.5)<0.05, vt["tower_byte_share_pct"])
ck("saves 0.83 GB", abs(vt["saving_vs_bf16_gb"]-0.828)<0.005, vt["saving_vs_bf16_gb"])
ck("tower claim UNMEASURED", vt["caveat"].startswith("UNMEASURED"))

# --- catalogue ---
b2=E.MODEL_DB["bonsai-2-27b"]
ck("Bonsai 2 context 262144", b2.context==262_144)
ck("Bonsai 2 native 1.768 bits", abs(b2.native_bits-1.768)<0.001)
ck("LM params 26.90B", abs(E.BONSAI2_PARAMS_LM/1e9-26.90)<0.01, E.BONSAI2_PARAMS_LM)
ck("bonsai-27b corrected to 3.80", abs(E.MODEL_DB["bonsai-27b"].min_ram_gb_int4-3.80)<0.01)
ck("F16 divides to 16.0 bpw", abs(E.container_tax("bonsai-2-27b F16 (gguf)")["effective_bpw"]-16.0)<0.005)
ck("phone claim has no vendor source", E.PHONE_RESIDENCY["vendor_claim"] is None)

# --- calibration law ---
ck("c = 1.75", R.REFINEMENT_C==1.75)
ck("n/d=0.5 held-out 0.77x", abs(R.MEASURED_KEPT[0.5][1]-0.766)<0.005)
ck("n/d=0.5 calibration 2.00x", abs(R.MEASURED_KEPT[0.5][0]-1.997)<0.005)
pl=R.layer_refinement_plan("bonsai-2-27b",65_536)
ck("Bonsai2 down_proj 54% at 128x512", abs(pl["binding_kept"]-0.535)<0.01, f"{pl['binding_kept']:.3f}")
pl2=R.layer_refinement_plan("bonsai-2-27b",262_144)
ck("Bonsai2 down_proj 88% at 128x2048", abs(pl2["binding_kept"]-0.88)<0.015, f"{pl2['binding_kept']:.3f}")
attn=[m for m in pl2["matrices"] if m["matrix"].startswith("q/k")][0]
ck("attn proj ~96% at 128x2048", abs(attn["kept_fraction"]-0.966)<0.01, f"{attn['kept_fraction']:.3f}")
ck("paper cycle FAITHFUL", R.validate_against_paper()["verdict"]=="FAITHFUL")
ck("sigma floor 0.1", abs(R.analyse_cycle()["sigma_floor"]-0.1)<0.001)

# --- certificates ---
ex=Q.experiment_fault_detection(verbose=False)
ck("7/7 faults caught", ex["caught"]==7 and ex["total"]==7, f"{ex['caught']}/{ex['total']}")
ck("certificate ~3 KB", 2000<ex["certificate_bytes"]<4000, ex["certificate_bytes"])
ck("bf16@4096 uncertifiable", not Q.derive_tolerance("bf16",4096)["certifiable"])

# --- dream search ---
c=D.compare_policies(D.COLAB_2026_09)
g=[r for r in c["rows"] if r["policy"]=="greedy"][0]
ck("greedy 4 evals", g["evals"]==4, g["evals"])
ck("greedy 4.5x fewer calls", abs(g["fewer_calls"]-4.5)<0.01, g["fewer_calls"])
ck("exhaustive 18 evals", [r for r in c["rows"] if r["policy"]=="exhaustive"][0]["evals"]==18)
ck("Colab pool flagged degenerate", c["degeneracy"]["degenerate"])
ck("Colab spread 2994x", abs(c["pool_summary"]["spread"]-2994.2)<1, c["pool_summary"]["spread"])

# --- HP target and AI Hub ---
ck("seven HP Snapdragon machines", len(HP.HP_MACHINES) == 7, len(HP.HP_MACHINES))
ck("three AI Hub Compute devices", len(HP.AIHUB_COMPUTE_DEVICES) == 3)
ck("device strings match the site verbatim",
   HP.AIHUB_COMPUTE_DEVICES == ("Snapdragon X Elite CRD",
                                "Snapdragon X Plus 8-Core CRD",
                                "Snapdragon X2 Elite CRD"))
_x2p = [m for m in HP.HP_MACHINES if "X2 Plus" in m.soc][0]
ck("the X2 Plus machine has no AI Hub device",
   HP.aihub_device_for(_x2p)["device"] is None)
ck("Qualcomm has published X Elite (491) and X2 Elite (487) measurements",
   HP.PUBLISHED_COMPUTE_STATUS["published_numbers"].get("Snapdragon X Elite CRD") == 491
   and HP.PUBLISHED_COMPUTE_STATUS["published_numbers"].get("Snapdragon X2 Elite CRD") == 487)
ck("and none for X Plus 8-Core, the proxy for four HP machines",
   HP.PUBLISHED_COMPUTE_STATUS["published_numbers"].get("Snapdragon X Plus 8-Core CRD") == 0
   and sum(HP.aihub_device_for(m)["device"] == "Snapdragon X Plus 8-Core CRD"
           for m in HP.HP_MACHINES) == 4)
ck("the 8 GB machine cannot hold an 8B INT4 model",
   not HP.fits_on_machine(HP.HP_MACHINES[0], 8.0, 4.0)["fits"])
ck("the 8 GB machine can hold a 4B INT4 model",
   HP.fits_on_machine(HP.HP_MACHINES[0], 4.0, 4.0)["fits"])
ck("prefill is NPU, decode is CPU",
   HP.phase_split(HP.HP_MACHINES[-1])["prefill"]["engine"] == "Hexagon NPU"
   and "CPU" in HP.phase_split(HP.HP_MACHINES[-1])["decode"]["engine"])
ck("published-figures tier: available for two CRDs, not X Plus 8-Core",
   "NOT AVAILABLE for X Plus 8-Core" in HP.EVIDENCE_TIERS[2].strength)
_api = HP.verify_api_signatures()
ck("every AI Hub call matches the installed client",
   _api["ok"] is not False, f"qai-hub {_api.get('version','absent')}")
ck("twelve Compute LLMs listed", len(HP.AIHUB_COMPUTE_LLMS) == 12)
ck("submit defaults to a dry run",
   HP.submit_profile("m.tflite", "Snapdragon X Elite CRD")["dry_run"])

# --- the scaling book, and Qualcomm's measured X-series data ---
_book = RL.validate_against_book()
ck("the book's worked answers reproduce (17 of 17)",
   _book["verdict"] == "FAITHFUL" and _book["n"] == 17, f"{_book['n_ok']}/{_book['n']}")
_law = RL.bandwidth_law("X Elite", "geniex_qairt", "w4a16", "npu", 4096)
ck("X Elite QAIRT decode is linear in bytes, R^2 >= 0.99 under all six",
   _law["min_r2"] >= 0.99, _law["min_r2"])
ck("its intercept is within 5 ms of zero", max(abs(x) for x in _law["intercept_range_ms"]) <= 5,
   _law["intercept_range_ms"])
ck("effective bandwidth is 43-51% of the 135 GB/s peak", _law["pct_range"] == (43, 51),
   _law["pct_range"])
_ecb = {(e["device"], e["runtime"], e["unit"], e["context"]): e
        for e in RL.effective_critical_batch()}
_x2 = _ecb[("X2 Elite", "geniex_qairt", "npu", 512)]
ck("X2 Elite NPU prefill/decode 62.8-70.6, CV 0.05",
   (_x2["min"], _x2["max"]) == (62.8, 70.6) and abs(_x2["cv"] - 0.053) < 0.002,
   (_x2["min"], _x2["max"], _x2["cv"]))
_x2c = _ecb[("X2 Elite", "geniex_llamacpp", "cpu", 512)]
ck("the CPU on the same bus: 13.7-17.1", (_x2c["min"], _x2c["max"]) == (13.7, 17.1),
   (_x2c["min"], _x2c["max"]))
_e4 = {e["model"]: e for e in RL.engine_asymmetry("X2 Elite", 512)}["qwen3_4b"]
ck("Qwen3-4B on X2 Elite decodes at 33.7 / 33.5 / 36.2 tok/s (CPU/GPU/NPU)",
   (round(_e4["decode"]["cpu"], 1), round(_e4["decode"]["gpu"], 1),
    round(_e4["qairt_npu_decode"], 1)) == (33.7, 33.5, 36.2))
ck("while its prefill spans 461 -> 2307 tok/s",
   round(_e4["prefill"]["cpu"]) == 461 and round(_e4["qairt_npu_prefill"]) == 2307)
_cs = {(c["model"], c["device"]): c for c in RL.context_slope()}
ck("Qwen3-4B KV overtakes weights at ~9,770 tokens on X2 Elite",
   _cs[("qwen3_4b", "X2 Elite")]["t_star_measured"] == 9770)
ck("measured crossovers are 0.65-1.13x the 16-bit-KV formula",
   sorted(c["measured_over_formula_kv16"] for c in _cs.values()) == [0.65, 0.85, 1.13])
# The 9,770 fit keeps the 4K-context point that leave-one-out flags in the
# cross-model fit. The documents quote the fit without it too, so check that.
_p4 = sorted((r[5], 1000.0 / r[6]) for r in RL.QUALCOMM_MEASURED
             if r[:5] == ("qwen3_4b", "w4a16", "X2 Elite", "geniex_qairt", "npu")
             and r[5] != 4096)
_f4 = RL.fit_line([c for c, _ in _p4], [t for _, t in _p4])
ck("without the flagged 4K point, Qwen3-4B crosses over at ~12,700 tokens",
   round(_f4["a"] / _f4["b"], -2) == 12700, round(_f4["a"] / _f4["b"]))
_rs = {(r["model"], r["device"], r["precision"]): r["ratio"] for r in RL.runtime_spread()}
ck("software alone is 3.23x on Qwen3-4B, X Elite", _rs[("qwen3_4b", "X Elite", "w4a16")] == 3.23)
ck("exactly two published pairs are unphysical", len(RL.anomalies()) == 2)
ck("the 90-row snapshot matches the package when installed",
   RL.snapshot_matches_package()["status"] in ("MATCH", "SKIPPED"),
   RL.snapshot_matches_package()["status"])
ck("the container rule is 35% at zero context, 11% at 128K",
   [r["mlx_extra_bytes_pct"] for r in RL.container_vs_context()["rows"]][0::3] == [34.9, 10.7])
# FINDINGS says not every series is clean, and says how many are not. That
# sentence is only honest while the tool agrees with it.
_reg = [r["regime"] for r in RL.bandwidth_law_table()]
ck("of 21 published series: 8 bandwidth-bound, 12 slope + overhead, 1 not linear",
   (len(_reg), _reg.count("bandwidth-bound"),
    _reg.count("bandwidth slope + fixed overhead"),
    _reg.count("not linear in bytes")) == (21, 8, 12, 1),
   (len(_reg), _reg.count("bandwidth-bound")))

# --- zero download on AI Hub Workbench ---
ck("the Workbench runner downloads profiles only: 2 MB ceiling",
   W.MAX_DOWNLOAD_BYTES == 2 * 1024 * 1024)
ck("a Qwen3-4B layer uploads as ~84 MB (4.8x compression)",
   abs(W.estimated_upload_bytes(W.SliceSpec("qwen3_4b")) / 1e6 - 84.1) < 0.5,
   W.estimated_upload_bytes(W.SliceSpec("qwen3_4b")))
ck("its calibration data is 33.6 MB",
   abs(W.calibration_bytes(W.SliceSpec("qwen3_4b")) / 1e6 - 33.6) < 0.1)
_net = []
for _f in sorted(HERE.glob("*.py")) + sorted((HERE / "sigil").glob("*.py")):
    _net += [(_f.name,) + h for h in W.download_calls_in(_f.read_text(encoding="utf-8"))]
ck("no file in the project downloads a model", not _net, _net[:3])
ck("the default Workbench device is the unmeasured X Plus 8-Core",
   W.DEFAULT_DEVICE == "Snapdragon X Plus 8-Core CRD")
_thr = E.BONSAI2_THROUGHPUT
ck("ternary throughput is published on ten machines, for both containers on eight",
   (len(_thr), sum(1 for v in _thr.values() if v[1] is not None and v[2] is not None))
   == (10, 8))
_sbw = [so.mem_bandwidth_gbs for so in E.SOC_DB.values() if so.mem_bandwidth_gbs]
ck("nine Snapdragon parts at six bandwidths, 51-228 GB/s",
   (len(_sbw), len(set(_sbw)), round(min(_sbw)), round(max(_sbw))) == (9, 6, 51, 228),
   (len(_sbw), len(set(_sbw)), min(_sbw), max(_sbw)))
ck("the withdrawn 1-bit claim (~3.4 GB) was 12% under the 3.80 GB file",
   round(100 * (3_803_452_480 / 1e9 - 3.4) / 3.4) == 12)
ck("32 of the 61 catalogue identifiers are registry-confirmed",
   (sum(1 for m in E.MODEL_DB.values() if m.id_verified), len(E.MODEL_DB)) == (32, 61),
   (sum(1 for m in E.MODEL_DB.values() if m.id_verified), len(E.MODEL_DB)))

# --- the documented counts must match what the code actually reports ---
import subprocess
HERE = pathlib.Path(__file__).resolve().parent
SUITES = {"snapdragon_engine": 141, "sigil_t4_benchmark": 123, "dream_search": 87,
          "roofline": 83, "aihub_workbench": 64,
          "psdc": 60, "quant_certificate": 58, "residual_cascade": 49,
          "generator_quant": 47, "qat_optimizer": 43, "on_device_adaptation": 23,
          "slt_compressibility": 22, "aihub_bench": 16,
          "hp_snapdragon": 56}
# A suite that cannot run because a heavy optional dependency is missing is
# counted at its documented size and marked skipped, not failed. torch is a
# ~2 GB install; requiring it to audit a claim about a roofline would make
# this file unrunnable on exactly the machines it is meant to reassure.
# The skip is recognised from the message the module prints, so a genuinely
# broken suite still fails.
OPTIONAL_DEPS = {"sigil_t4_benchmark": ("torch is required", "torch")}

total = 0
skipped = []
for name, expected in SUITES.items():
    arg = "--selftest" if name == "sigil_t4_benchmark" else "selftest"
    text = ""
    try:
        out = subprocess.run([sys.executable, str(HERE / f"{name}.py"), arg],
                             capture_output=True, text=True, timeout=900,
                             cwd=str(HERE))
        text = out.stdout + out.stderr
        m = re.search(r"(\d+)/(\d+) (?:checks )?passed", text)
        got_pass, got_tot = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    except Exception:
        got_pass, got_tot = -1, -1
    needle, dep = OPTIONAL_DEPS.get(name, ("", ""))
    if needle and got_tot == -1 and needle in text:
        total += expected
        skipped.append(f"{name} ({dep} absent)")
        ck(f"{name} reports {expected} tests, all passing", True,
           f"skipped: {dep} not installed")
        continue
    total += max(got_tot, 0)
    ck(f"{name} reports {expected} tests, all passing",
       got_tot == expected and got_pass == got_tot, f"{got_pass}/{got_tot}")
ck("documented total is 872 self-tests", total == 872,
   f"{total}" + (f" ({'; '.join(skipped)} counted at its documented size)"
                 if skipped else ""))

# --- the documents must quote the counts the harness actually reports ---
#
# `"222" in docs` was the old check. A bare substring search is not an audit:
# it matches a page number, a byte figure, a year. It kept passing after the
# harness grew to 223 checks, because "222" was still somewhere in the file.
# Every count below is matched in context, and the stale values are named so
# they can never come back quietly.
STRESS_FULL, STRESS_NO_PKG, SELFTESTS, AUDIT_CLAIMS = 292, 255, 872, 131

docs = {f: (HERE / f).read_text(encoding="utf-8")
        for f in ("MANIFEST.md", "README.md", "SUBMISSION.md")}
man = docs["MANIFEST.md"]
ck(f"MANIFEST quotes {SELFTESTS} self-tests", f"{SELFTESTS} self-tests" in man)
ck(f"MANIFEST quotes {STRESS_FULL} adversarial checks",
   f"{STRESS_FULL}\nadversarial checks" in man or
   f"{STRESS_FULL} adversarial checks" in man)
ck(f"MANIFEST quotes {STRESS_NO_PKG} for a run without sigil/",
   f"({STRESS_NO_PKG} if the optional" in man)
ck(f"MANIFEST's stress_all row says {STRESS_FULL}",
   f"| `stress_all.py` | {STRESS_FULL} |" in man)
ck(f"README's stress_all row says {STRESS_FULL}",
   f"| `stress_all.py` | {STRESS_FULL} |" in docs["README.md"])
ck(f"SUBMISSION quotes a {STRESS_FULL}-check harness",
   f"{STRESS_FULL}-check stress harness" in docs["SUBMISSION.md"])
# The deck and the brief are generated, so their numbers live in source and go
# stale exactly like a document's. The brief shipped claiming "all 43 published
# numbers" long after the audit had grown past it; nothing was checking.
_deck = (HERE / "build_deck.py").read_text(encoding="utf-8")
_brief = (HERE / "build_brief.js").read_text(encoding="utf-8")
ck(f"the deck's stat block says {STRESS_FULL}",
   f'("{STRESS_FULL}", "adversarial checks' in _deck)
ck(f"the deck's stat block says {SELFTESTS}",
   f'("{SELFTESTS}", "self-tests' in _deck)
ck(f"the brief says a {STRESS_FULL}-check harness",
   f"{STRESS_FULL}-check adversarial" in _brief)
ck(f"the brief quotes the {AUDIT_CLAIMS}-check claim audit",
   f"{AUDIT_CLAIMS}-check claim audit" in _brief)
ck("the deck refuses to write text outside its panels",
   "refusing to write a deck with text outside its panels" in _deck)
ck(f"README's audit_claims row says {AUDIT_CLAIMS}",
   f"| `audit_claims.py` | {AUDIT_CLAIMS} |" in docs["README.md"])

# Stale values are hunted in every document a reader acts on -- and in the
# deck and brief sources, which is where "Nothing." and "all 86" outlived the
# facts. PROVENANCE.md is left out on purpose: it is the dated record, and it
# quotes each withdrawn claim verbatim so the correction can be checked.
_stale_in = dict(docs)
for _f in ("QUICKSTART.md", "RUNBOOK.md", "FINDINGS.md", "RESEARCH_AGENDA.md"):
    _stale_in[_f] = (HERE / _f).read_text(encoding="utf-8")
_stale_in["build_deck.py"], _stale_in["build_brief.js"] = _deck, _brief
STALE = ["180 adversarial", "222-check", "| `stress_all.py` | 222 |",
         "| `audit_claims.py` | 58 |", "207\nadversarial", "191 if the optional",
         "705 self-tests", "223-check", "| `stress_all.py` | 223 |",
         "192 if the optional", "publishes no Compute performance numbers at all",
         "qai-hub-models " + "fetch", "qai-hub-models " + "export",
         "290-check", "| `stress_all.py` | 290 |", "253 if the optional",
         "| `audit_claims.py` | 115 |", "Eight predictions that a source would be barren",
         "~118 MB",
         # Added 2026-09-25, the day the Workbench job succeeded: these two
         # sentences were true when they were written and false minutes later,
         # which is exactly the failure mode this list exists for.
         "No job from this project has run", "No job has been run from here yet",
         "128-check claim audit"]
for bad in STALE:
    _where = [f for f, t in _stale_in.items() if bad in t]
    ck(f"no document still quotes {bad.strip()!r}", not _where, _where)

# Last, so that len(rows) is final: this file must not misreport its own size.
# Every other count here is audited against the code; leaving this one to be
# typed by hand is how "222" survived the harness growing to 223.
ck(f"MANIFEST's audit_claims row says {AUDIT_CLAIMS}",
   f"| `audit_claims.py` | {AUDIT_CLAIMS} |" in man
   and AUDIT_CLAIMS == len(rows) + 1, f"audit really has {len(rows) + 1} claims")

npass=sum(1 for _,ok,_ in rows if ok)
for claim, ok, got in rows:
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}" + (f"   got {got}" if (got and not ok) else ""))
print()
print(f"  {npass}/{len(rows)} published claims verified against the code")
if skipped:
    print()
    print("  NOT VERIFIED HERE, and the total above says so:")
    for s in skipped:
        print(f"    - {s}: counted at its documented size, not re-measured.")
    print("    Install the dependency and re-run to close the gap.")
if npass != len(rows):
    print("  A claim and the code disagree. Fix one of them before submitting.")
sys.exit(0 if npass==len(rows) else 1)
