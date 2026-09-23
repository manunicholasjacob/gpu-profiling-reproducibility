#!/usr/bin/env python3
"""Paper 2 v3 analysis: filter to clean (non-timeout) thermally-gated cells, verify
drift is removed, refit elasticity + held-out MAPE, quantify the clock-gating yield.
Run after the clock-gated sweep completes."""
import json, glob, os, re, sys
import numpy as np
from scipy import stats

# Released artifact first, live campaign directory second. The campaign
# directory holds one timestamped subdirectory per cell and is the only place
# run order can be recovered from; it was not retained, so by default this runs
# against the released summaries and says what it cannot compute.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASED = os.path.join(HERE, "data", "v3_thermal_gated_summaries.json")

src = sys.argv[1] if len(sys.argv) > 1 else RELEASED
campaign = sys.argv[2] if len(sys.argv) > 2 else "outputs/profile_v3"
if os.path.isdir(src):                      # a campaign directory was passed
    campaign = src
    src = os.path.join(campaign, "all_summaries.json")
if not os.path.exists(src):
    sys.exit("no summaries at %s\nusage: analyze_v3.py [summaries.json|campaign_dir] "
             "[campaign_dir]" % src)

r = json.load(open(src))
print(f"source: {src}")
print(f"total cells: {len(r)}")

# Run order is reconstructed from the timestamped per-cell directory names of
# the live campaign. Those directories held the raw per-request traces and were
# not kept, so this is normally unavailable.
dirs = sorted(glob.glob(os.path.join(campaign, "2026*")))
order = {}
for i, d in enumerate(dirs):
    m = re.match(r"\d{8}_\d{6}_(.+)_c(\d+)_s(\d+)", os.path.basename(d))
    if m: order[(m.group(1), int(m.group(2)), int(m.group(3)))] = i
for c in r:
    c["order"] = order.get((c["model"], c["concurrency"], c["seed"]), np.nan)
HAVE_ORDER = bool(order)
if not HAVE_ORDER:
    print(f"run order: UNAVAILABLE (no per-cell directories under {campaign}/)")
    print("           the drift correlations below cannot be recomputed here; "
          "see data/README.md")

clean = [c for c in r if not c.get("gate_timeout")]
degraded = [c for c in r if c.get("gate_timeout")]
print(f"clean (gated OK): {len(clean)}   timed-out: {len(degraded)} ({100*len(degraded)/len(r):.0f}%)")

print("\n=== YIELD per config (clean cells) ===")
from collections import defaultdict
cfg = defaultdict(int)
for c in clean: cfg[(c["model"], c["concurrency"])] += 1
low = {k: v for k, v in cfg.items() if v < 2}
print(f"configs with <2 clean cells: {len(low)}  {dict(list(low.items())[:8])}")

print("\n=== DRIFT CHECK: corr(run_order, gpu_temp) and (order, latency) ===")
if not HAVE_ORDER:
    print("  skipped: run order was not retained with the released summaries.")
    print("  The paper's v3 value (r=+0.14, p=0.24) came from the live campaign "
          "directory.")
for name, subset in ([] if not HAVE_ORDER else [("all", r), ("clean", clean)]):
    s = [c for c in subset if not np.isnan(c["order"]) and c.get("gate_start_temp")]
    if len(s) < 10: continue
    ords = [c["order"] for c in s]
    temps = [c["gate_start_temp"] for c in s]
    # normalize latency within config
    bycfg = defaultdict(list)
    for c in s: bycfg[(c["model"], c["concurrency"])].append(c)
    rel = []
    ordr = []
    for cells in bycfg.values():
        med = np.median([c["p95_ms"] for c in cells])
        for c in cells:
            rel.append(c["p95_ms"] / med); ordr.append(c["order"])
    rt, pt = stats.pearsonr(ords, temps)
    rl, pl = stats.pearsonr(ordr, rel)
    print(f"  {name:6s}: corr(order,start_temp)={rt:+.3f} (p={pt:.3f})  "
          f"corr(order,rel_lat)={rl:+.3f} (p={pl:.3f})")
print("  (v2 had corr(order,rel_lat)=+0.57; clean v3 should be near 0 = drift removed)")

print("\n=== ELASTICITY CURVES (clean, median p95 by concurrency) ===")
for mdl in ["resnet50", "mobilenetv3", "efficientnet-b0"]:
    pts = defaultdict(list)
    for c in clean:
        if c["model"] == mdl: pts[c["concurrency"]].append(c["p95_ms"])
    xs = sorted(pts)
    curve = [np.median(pts[x]) for x in xs]
    d = np.diff(curve); nonmono = int((d < 0).sum())
    print(f"  {mdl:16s} nonmono={nonmono}/{max(len(d),1)}  {[(x, round(v,1)) for x,v in zip(xs,curve)]}")

print("\n=== SEED CV (clean) ===")
cvs = []
for (mdl, c), cells in [( (c['model'],c['concurrency']), [x for x in clean if x['model']==c['model'] and x['concurrency']==c['concurrency']]) for c in clean]:
    pass
bycfg = defaultdict(list)
for c in clean: bycfg[(c["model"], c["concurrency"])].append(c["p95_ms"])
cvs = [np.std(v)/np.mean(v) for v in bycfg.values() if len(v) >= 3]
if cvs:
    print(f"  median seed CV = {np.median(cvs):.3f}  (v1 was 0.304, partial v2 0.10, full v2 0.68)")

print("\n=== HELD-OUT PREDICTION (clean) train c={1,4,8,12,16} test c={2,6} ===")
def mape(a, p): return np.mean(np.abs((a - p) / a)) * 100
TR, TE = [1, 4, 8, 12, 16], [2, 6]
errs = []
for mdl in ["resnet50", "mobilenetv3", "efficientnet-b0"]:
    pts = defaultdict(list)
    for c in clean:
        if c["model"] == mdl: pts[c["concurrency"]].append(c["p95_ms"])
    m = {x: np.median(v) for x, v in pts.items()}
    tr = [x for x in TR if x in m]; te = [x for x in TE if x in m]
    if len(tr) < 3 or not te:
        print(f"  {mdl:16s} insufficient clean coverage")
        continue
    pred = np.interp(te, tr, [m[x] for x in tr]); act = np.array([m[x] for x in te])
    e = mape(act, pred); errs.append(e)
    print(f"  {mdl:16s} MAPE={e:5.2f}%   act={np.round(act,1)} pred={np.round(pred,1)}")
if errs: print(f"  {'OVERALL':16s} MAPE={np.mean(errs):.2f}%")
print(f"\n  clean noise floor ~{np.median(cvs)*100:.0f}% -> prediction is "
      f"{'above' if errs and np.mean(errs)>np.median(cvs)*100 else 'at/below'} noise")

print("\n=== CLOCK-GATING YIELD (the Paper 2 methodology finding) ===")
print(f"  thermal gate kept {len(clean)}/{len(r)} cells at uniform start temp (<=52C);")
print(f"  {len(degraded)} cells ({100*len(degraded)/len(r):.0f}%) could not cool in 75s due to")
print(f"  laptop chassis heat-soak over the {r[-1].get('order','?')}-cell campaign, and are excluded.")
