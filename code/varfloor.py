#!/usr/bin/env python3
"""Variance-floor and drift measurement for CPU inference profiling.

Extends the thermally-gated GPU protocol to CPU platforms. The original gate
waits for a temperature threshold, which is unavailable on many platforms
(Windows exposes no ACPI thermal zone on this machine). We gate instead on
achieved frequency ratio, which measures the thing temperature actually causes:
throttling. That makes the protocol portable, and arguably tightens it, since
frequency ratio is one step closer to the latency being measured.

Per cell we record p50/p95/p99 latency plus the gate state before and after, so
the analysis can ask the two questions that matter for reproducibility: does run
order predict latency (drift), and how much seed-to-seed variance survives after
drift is removed (the floor).
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort


# ------------------------------------------------------------ gate ----------
class FreqGate:
    """Reads '% Processor Performance', the achieved/nominal frequency ratio."""

    WIN_CMD = ("(Get-Counter '\\Processor Information(_Total)\\% Processor Performance')"
               ".CounterSamples[0].CookedValue")

    def __init__(self):
        self.kind = "windows-perfcounter" if platform.system() == "Windows" else "linux-cpufreq"

    def read(self):
        if self.kind == "windows-perfcounter":
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", self.WIN_CMD],
                    stderr=subprocess.DEVNULL, timeout=20).decode().strip()
                return float(out.replace(",", "."))
            except Exception:
                return -1.0
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                cur = int(f.read())
            with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
                mx = int(f.read())
            return 100.0 * cur / mx
        except Exception:
            return -1.0

    def wait(self, threshold, max_wait_s, poll_s=3.0):
        """Wait until the platform is at or above `threshold`. Returns (ok, waited_s, value)."""
        t0 = time.time()
        v = self.read()
        while time.time() - t0 < max_wait_s:
            if v < 0 or v >= threshold:
                return True, time.time() - t0, v
            time.sleep(poll_s)
            v = self.read()
        return False, time.time() - t0, v


# ------------------------------------------------------------ cell ----------
def run_cell(model_path, concurrency, seed, n_iter, threads_per_session):
    """Run `concurrency` independent sessions in parallel; collect all latencies."""
    rng = np.random.default_rng(seed)
    sessions, inputs = [], []
    for _ in range(concurrency):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads_per_session
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(model_path, sess_options=so,
                                 providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        sessions.append((s, inp.name))
        inputs.append(rng.standard_normal(shape).astype(np.float32))

    lat = [[] for _ in range(concurrency)]
    barrier = threading.Barrier(concurrency)

    def worker(i):
        s, iname = sessions[i]
        x = inputs[i]
        for _ in range(10):
            s.run(None, {iname: x})
        barrier.wait()
        for _ in range(n_iter):
            t = time.perf_counter()
            s.run(None, {iname: x})
            lat[i].append((time.perf_counter() - t) * 1000.0)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    span = time.perf_counter() - t0

    flat = sorted(x for sub in lat for x in sub)
    return {
        "p50_ms": flat[len(flat) // 2],
        "p95_ms": flat[int(0.95 * len(flat))],
        "p99_ms": flat[int(0.99 * len(flat))],
        "mean_ms": statistics.mean(flat),
        "n_samples": len(flat),
        "wall_s": span,
        "throughput_ips": len(flat) / span,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--threads-per-session", type=int, default=1)
    ap.add_argument("--gate-threshold", type=float, default=0.0,
                    help="minimum %% processor performance to start a cell; 0 disables gating")
    ap.add_argument("--gate-max-wait", type=float, default=90.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gate = FreqGate()
    meta = {"platform": platform.platform(), "machine": platform.machine(),
            "ort_version": ort.__version__, "gate_kind": gate.kind,
            "gate_threshold": args.gate_threshold, "tag": args.tag}
    print("#", json.dumps(meta), flush=True)

    order = 0
    total = len(args.seeds) * len(args.models) * len(args.concurrency)
    with open(args.out, "a") as fout:
        # seed is the outer loop so run order is not confounded with configuration
        for seed in args.seeds:
            for m in args.models:
                for c in args.concurrency:
                    ok, waited, gval = (True, 0.0, gate.read())
                    if args.gate_threshold > 0:
                        ok, waited, gval = gate.wait(args.gate_threshold, args.gate_max_wait)
                    rec = {"model": os.path.splitext(os.path.basename(m))[0],
                           "concurrency": c, "seed": seed, "order": order,
                           "gate_ok": ok, "gate_wait_s": waited,
                           "gate_before": gval, **meta}
                    rec.update(run_cell(m, c, seed, args.iter, args.threads_per_session))
                    rec["gate_after"] = gate.read()
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    order += 1
                    print(f"[{order}/{total}] {rec['model']:22s} c={c} seed={seed}  "
                          f"p50={rec['p50_ms']:7.3f} p95={rec['p95_ms']:7.3f} ms  "
                          f"gate {gval:5.1f}->{rec['gate_after']:5.1f}"
                          f"{'' if ok else '  GATE TIMEOUT'}", flush=True)
    print("# done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
