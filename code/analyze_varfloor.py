#!/usr/bin/env python3
"""Drift and variance-floor analysis for CPU inference profiling.

Two questions decide whether a profiling protocol is reproducible:
  1. Does run order predict latency? A nonzero correlation means the numbers
     encode when they were taken, not what was measured.
  2. Once configuration is accounted for, how much seed-to-seed spread remains?
     That is the floor no scheduling removes, and it bounds the resolution of
     any comparison built on these numbers.

Latency is normalised within each (model, concurrency) cell before the drift
test, so the correlation measures drift rather than the configuration effect.
"""

import json
import statistics
import sys
from collections import defaultdict


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main(paths):
    rows = []
    for p in paths:
        for ln in open(p):
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if not rows:
        print("no data")
        return
    print(f"# {len(rows)} cells from {', '.join(paths)}")
    print(f"# platform: {rows[0].get('machine')}  ORT {rows[0].get('ort_version')}  "
          f"gate: {rows[0].get('gate_kind')} @ {rows[0].get('gate_threshold')}")

    timeouts = [r for r in rows if not r.get("gate_ok", True)]
    print(f"# gate timeouts: {len(timeouts)}/{len(rows)} "
          f"({100*len(timeouts)/len(rows):.0f}% yield loss)")

    for metric in ("p50_ms", "p95_ms"):
        cells = defaultdict(list)
        for r in rows:
            cells[(r["model"], r["concurrency"])].append(r)
        rel, order = [], []
        for cs in cells.values():
            med = statistics.median([c[metric] for c in cs])
            for c in cs:
                rel.append(c[metric] / med)
                order.append(c["order"])
        print(f"\n=== DRIFT, {metric}: corr(run_order, within-cell relative latency) "
              f"= {pearson(order, rel):+.3f}")

        print(f"=== VARIANCE FLOOR, {metric}: seed-to-seed CV within each cell")
        print(f"{'model':<24}{'conc':>5}{'n':>4}{'median':>10}{'CV %':>8}{'max/min':>9}")
        cvs = []
        for k in sorted(cells):
            vs = [c[metric] for c in cells[k]]
            if len(vs) < 2:
                continue
            m = statistics.mean(vs)
            cv = 100 * statistics.stdev(vs) / m
            cvs.append(cv)
            print(f"{k[0]:<24}{k[1]:>5}{len(vs):>4}{statistics.median(vs):>10.3f}"
                  f"{cv:>8.2f}{max(vs)/min(vs):>9.3f}")
        if cvs:
            print(f"  -> median seed CV {statistics.median(cvs):.2f}%, "
                  f"worst {max(cvs):.2f}%")

    gb = [r.get("gate_before", -1) for r in rows if r.get("gate_before", -1) > 0]
    if gb:
        print(f"\n=== PLATFORM STATE: %processor-performance at cell start, "
              f"min {min(gb):.1f} max {max(gb):.1f} median {statistics.median(gb):.1f}")
        gvals = [r["gate_before"] for r in rows if r.get("gate_before", -1) > 0]
        gord = [r["order"] for r in rows if r.get("gate_before", -1) > 0]
        print(f"    corr(run_order, gate_before) = {pearson(gord, gvals):+.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
