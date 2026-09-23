# What is in this directory, and what is not

This is a paper about reproducible measurement, so the release has to be explicit about which
of its own numbers a reader can recompute and which they have to take on trust. Three of them
they have to take on trust, and this file says which and why.

## Released

| File | Contents | Numbers it supports |
|---|---|---|
| `v3_thermal_gated_summaries.json` | 105 cells of the v3 thermally-gated GPU campaign, one record per (model, concurrency, seed) | yield 74/105 and 31 timeouts (30%); median seed CV 0.320 on p95, 0.290 on p50; held-out MAPE 15.21% on p95; one elasticity inversion per model |
| `cpu-arms/x86_ungated_clean.jsonl` | quiet CPU arm, 45 cells | corr $-0.013$, seed CV 1.23% (p50) / 5.4% (p95) |
| `cpu-arms/x86_gated_clean.jsonl` | quiet CPU arm with the 95% performance gate, 45 cells | 18% yield, corr $+0.161$, seed CV 1.14% (p50) / 6.6% (p95) |
| `cpu-arms/x86_ungated_CONTAMINATED.jsonl` | the arm run while unrelated jobs shared the machine, 27 cells (3 seeds, not 5) | corr $-0.389$, seed CV 46% (p50) / 62% (p95), corr(order, %proc-perf) $= +0.680$ |

Reproduce with:

```bash
python scripts/analyze_v3.py
```

```bash
python code/analyze_varfloor.py data/cpu-arms/x86_ungated_clean.jsonl
```

## Not released, and why

**The v1 (blocked) and v2 (randomized) GPU campaigns.** Only the v3 summaries were retained. The
three numbers that come from those campaigns, the v1 seed CV of 0.30, the v2 seed CV of 0.68, and
the v2 run-order correlation of $r = 0.57$, cannot be recomputed here. They are transcribed into
`figures/gen_fig_p2.py` from the original analysis output.

**Run order for the v3 cells.** `analyze_v3.py` reconstructed it from the timestamped per-cell
output directory names of the live campaign. Those directories held the raw per-request traces and
were not kept. Consequently the v3 drift correlation $r = 0.14$ ($p = 0.24$) also cannot be
recomputed from this directory. `analyze_v3.py` detects the absence and says so rather than
quietly reporting nothing; pass the campaign directory as a second argument if you have it.

This is the paper's own lesson turned on itself: a protocol is only as reproducible as the
artifacts it keeps, and a summary that drops run order drops the ability to test for drift. The
CPU arms, added later, carry an explicit `order` field for exactly this reason, which is why every
CPU number in the paper is recomputable and three of the GPU ones are not.

## Metric

Every latency statistic in the paper is **p95** unless labelled otherwise. It matters: on p50 the
GPU seed CV is 0.29 rather than 0.32 and the held-out MAPE is 30.8% rather than 15.2%. Set
`METRIC` in `scripts/analyze_v3.py` to `p50_ms` to see this. `code/analyze_varfloor.py` reports
both for the CPU arms.
