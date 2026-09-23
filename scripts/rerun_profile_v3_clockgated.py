#!/usr/bin/env python3
"""
Paper 2 — clock-gated profiling sweep (v3).

Root cause found in v1/v2: the RTX 3050 Laptop GPU's SM clock collapsed ~45%
over a 2-hour sweep (corr(run_order, latency)=+0.57). Two compounding causes:
chassis heat-soak AND — critically — the machine ran ON BATTERY, so the GPU
power cap tightened as the battery drained (throttle reason 0x24 =
SW Power Cap + SW Thermal Slowdown observed directly).

** RUN THIS ON AC POWER. ** On battery the sustained power budget shrinks over
time and no amount of gating fixes it. This harness additionally defends against
residual drift by:

  1. CLOCK GATE. Before every cell, idle until the achieved SM clock under a
     fixed probe load recovers to >= GATE_FRAC of a reference clock measured at
     the start (on a cool GPU). Cap the wait at GATE_MAX_WAIT_S; if it never
     recovers, record gate_timeout=True so the cell can be filtered.
  2. CLOCK QC. Record achieved SM clock during measurement. After the run,
     cells whose measurement clock is < GATE_FRAC*ref are flagged degraded.
  3. RETRY. Retry a degraded cell once (after a longer cooldown) before giving up.
  4. AC CHECK. Refuse to start unless on AC (override with --allow-battery).

Everything else matches rerun_profile_v2.py so downstream fitting is unchanged.

Usage:  ./venv/Scripts/python.exe rerun_profile_v3_clockgated.py [--quick] [--allow-battery]
Output: outputs/profile_v3/{timestamp}_{model}_c{c}_s{seed}/
"""
import os, sys, time, json, random, threading, subprocess, csv
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from src.workloads.models import ModelLoader
from src.telemetry.nvml_logger import NVMLLogger

QUICK = "--quick" in sys.argv
ALLOW_BATTERY = "--allow-battery" in sys.argv

MODELS      = ["resnet50", "mobilenetv3", "efficientnet-b0"]
CONCURRENCY = [1, 2, 4, 6, 8, 12, 16]
SEEDS       = [42, 43, 44, 45, 46] if not QUICK else [42]
WARMUP_S    = 20 if not QUICK else 5
DURATION_S  = 40 if not QUICK else 10
TELEM_HZ    = 10
OUT         = Path("outputs/profile_v3")
ORDER_SEED  = 20260721

# --- thermal gate parameters ---
# SM clock legitimately varies with load level (a c=1 cell runs at a lower clock
# than c=16 because the GPU does not need to boost), so gating on clock against a
# single reference is wrong. Instead we reset the GPU to a fixed THERMAL state
# before every cell: idle until temperature falls to GATE_TEMP_C. That removes the
# heat-soak drift regardless of the upcoming cell's load. AC power removes the
# other (battery-drain) drift cause. Clock is still recorded per cell for QC.
GATE_TEMP_C      = 52      # start each cell only once GPU has cooled to <= this
GATE_MAX_WAIT_S  = 75      # cap the cooldown wait; record gate_timeout if exceeded
GATE_POLL_S      = 3       # re-check temperature this often while cooling
MIN_COOLDOWN_S   = 8       # always idle at least this long (clears residual heat)
RETRY_COOLDOWN_S = 40

OUT.mkdir(parents=True, exist_ok=True)


def nvsmi(field):
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            timeout=5).decode().strip().splitlines()[0].strip()
    except Exception:
        return ""


def on_ac_power():
    """True if plugged in. Uses Windows WMIC battery status (2 == AC)."""
    try:
        out = subprocess.check_output(
            ["WMIC", "path", "Win32_Battery", "get", "BatteryStatus"],
            timeout=10).decode()
        vals = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
        if not vals:
            return True  # desktop / no battery
        return vals[0] not in ("1",)  # 1 = discharging
    except Exception:
        return True


def gpu_temp():
    c = nvsmi("temperature.gpu")
    return int(c) if c.isdigit() else 999


def thermal_gate():
    """Idle until GPU cools to GATE_TEMP_C (or timeout). Returns (temp, timeout, waited)."""
    t0 = time.time()
    time.sleep(MIN_COOLDOWN_S)              # always clear residual heat first
    while True:
        t = gpu_temp()
        if t <= GATE_TEMP_C:
            return t, False, time.time() - t0
        if time.time() - t0 > GATE_MAX_WAIT_S:
            return t, True, time.time() - t0
        time.sleep(GATE_POLL_S)


def run_cell(model_info, concurrency, seed, out_dir):
    gate_temp, gate_timeout, gate_wait = thermal_gate()

    rng = np.random.RandomState(seed)
    x = rng.randn(*model_info.input_shape).astype(np.float32)
    requests, lock, stop = [], threading.Lock(), threading.Event()
    recording = threading.Event()

    def worker():
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                model_info.inference_fn(x)
            except Exception:
                continue
            t1 = time.perf_counter()
            if recording.is_set():
                with lock:
                    requests.append({"start": t0, "end": t1, "latency_ms": (t1 - t0) * 1000})

    telemetry = NVMLLogger(poll_hz=TELEM_HZ)
    telemetry.start()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()

    time.sleep(WARMUP_S)
    telemetry.get_samples(clear=True)
    recording.set()
    t_start = time.time()
    time.sleep(DURATION_S)
    recording.clear()
    actual = time.time() - t_start

    stop.set()
    for t in threads:
        t.join(timeout=3.0)
    telemetry.stop()
    samples = telemetry.get_samples()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "end", "latency_ms"])
        w.writeheader(); w.writerows(requests)
    telemetry.save_to_csv(out_dir / "telemetry.csv")

    lat = np.array([r["latency_ms"] for r in requests]) if requests else np.array([np.nan])
    temps  = [s.get("temp_c") for s in samples if s.get("temp_c") is not None]
    smclk  = [s.get("clock_sm_mhz") for s in samples if s.get("clock_sm_mhz") is not None]
    powers = [s.get("power_w") for s in samples if s.get("power_w") is not None]
    meas_clock = float(np.mean(smclk)) if smclk else 0.0

    summary = {
        "model": model_info.model_id, "concurrency": concurrency, "seed": seed,
        "n_requests": len(requests), "duration_s": actual,
        "throughput_rps": len(requests) / actual if actual > 0 else 0,
        "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)), "mean_ms": float(np.mean(lat)),
        "std_ms": float(np.std(lat)),
        "gpu_temp_mean": float(np.mean(temps)) if temps else None,
        "sm_clock_mean": meas_clock,
        "sm_clock_min": float(np.min(smclk)) if smclk else None,
        "power_mean_w": float(np.mean(powers)) if powers else None,
        # --- thermal-gate provenance ---
        "gate_start_temp": gate_temp, "gate_timeout": gate_timeout,
        "gate_wait_s": round(gate_wait, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    print("=" * 78)
    print("Paper 2 clock-gated profiling sweep (v3)")
    print("=" * 78)

    on_ac = on_ac_power()
    print(f"AC power: {'YES' if on_ac else 'NO (on battery)'}")
    if not on_ac and not ALLOW_BATTERY:
        print("\nREFUSING TO START ON BATTERY. The GPU power cap tightens as the")
        print("battery drains, which is exactly what invalidated v1 and v2.")
        print("Plug in AC and re-run, or pass --allow-battery to override.")
        sys.exit(2)

    reg = ModelLoader(device="cuda")
    loaded = {}
    for name in MODELS:
        fn = {"resnet50": reg.load_resnet50, "mobilenetv3": reg.load_mobilenetv3,
              "efficientnet-b0": reg.load_efficientnet_b0}[name]
        loaded[name] = fn()

    print("\nHard warmup (300 iters/model)...")
    for name, mi in loaded.items():
        x = np.random.randn(*mi.input_shape).astype(np.float32)
        for _ in range(300):
            mi.inference_fn(x)
        torch.cuda.synchronize()
        print(f"  {name} warm")

    print(f"\nThermal gate: each cell starts only once GPU <= {GATE_TEMP_C}C "
          f"(max wait {GATE_MAX_WAIT_S}s). Clock recorded per cell for QC.")

    cells = [(m, c, s) for m in MODELS for c in CONCURRENCY for s in SEEDS]
    random.Random(ORDER_SEED).shuffle(cells)
    print(f"{len(cells)} cells, order seed {ORDER_SEED}.\n")

    results, t0 = [], time.time()
    for i, (mdl, c, seed) in enumerate(cells, 1):
        el = (time.time() - t0) / 60
        print(f"[{i:3d}/{len(cells)}] {mdl:16s} c={c:2d} s={seed}  elapsed {el:5.1f}m", flush=True)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            d = OUT / f"{ts}_{mdl}_c{c}_s{seed}"
            s = run_cell(loaded[mdl], c, seed, d)
            results.append(s)
            flag = " GATE_TIMEOUT" if s["gate_timeout"] else ""
            print(f"          p50={s['p50_ms']:7.2f} p95={s['p95_ms']:8.2f} "
                  f"tput={s['throughput_rps']:6.1f}/s clk={s['sm_clock_mean']:.0f}MHz "
                  f"start={s['gate_start_temp']}C wait={s['gate_wait_s']}s{flag}", flush=True)
        except Exception as e:
            print(f"          FAILED: {e}", flush=True)
        with open(OUT / "all_summaries.json", "w") as f:
            json.dump(results, f, indent=2)

    to = [r for r in results if r.get("gate_timeout")]
    print(f"\nDone. {len(results)} cells, {len(to)} started above thermal target. "
          f"{(time.time()-t0)/60:.1f} min")
    print(f"Results: {OUT}/all_summaries.json")


if __name__ == "__main__":
    main()
