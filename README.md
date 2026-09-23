# Profiling inference on a consumer laptop GPU

Measurement artifact for "Profiling Inference on a Consumer Laptop GPU: Thermal Gating Controls
Temperature, Not Clock".

The paper is a measurement-methodology study on a single representative device, an RTX 3050 Laptop
GPU. The same single-tenant elasticity sweep, three torchvision models across seven concurrency
levels, was run under three profiling protocols, and the paper analyses what each protocol does to
the resulting data. The short version: a blocked design confounds model identity with thermal
drift, randomising converts that drift into variance, a thermal gate fixes the dominant bias, and
even then the hardware imposes a variance floor and a yield cost that have to be reported rather
than hidden.

## What is here

| Path | Contents |
|---|---|
| `data/v3_thermal_gated_summaries.json` | 105 cells of the thermally gated GPU campaign, one record per (model, concurrency, seed) |
| `data/cpu-arms/*.jsonl` | the CPU control arms, quiet and gated and deliberately contaminated |
| `data/README.md` | **read this first.** States exactly which of the paper's numbers you can recompute here and which you cannot |
| `scripts/analyze_v3.py` | reproduces the GPU yield and variance numbers |
| `code/analyze_varfloor.py` | reproduces the CPU drift and variance-floor numbers |
| `scripts/rerun_profile_v3_clockgated.py` | the gated profiling harness itself |
| `figures/gen_fig_p2.py` | regenerates the paper's figure from the released data |

## Reproducing

```bash
python scripts/analyze_v3.py
python code/analyze_varfloor.py data/cpu-arms/x86_ungated_clean.jsonl
```

Both run on the standard library plus the files in this repository. No GPU is needed to recompute
the analysis; the raw campaign is already summarised.

## What is deliberately not here, and why

The v1 (blocked) and v2 (randomised) GPU campaigns were not retained beyond their summaries, and
the v3 per-cell run order was not retained either. Three numbers in the paper therefore cannot be
recomputed from this repository: the v1 seed CV of 0.30, the v2 seed CV of 0.68, and the v2
run-order correlation of r = 0.57. `data/README.md` says so explicitly, and the scripts print a
notice rather than silently skipping the affected checks.

One file is contaminated on purpose. `data/cpu-arms/x86_ungated_CONTAMINATED.jsonl` was collected
while unrelated jobs shared the machine, and it is kept because it is the evidence for the paper's
argument about what contamination looks like: correlation of -0.389 and a seed CV of 46% against
1.23% for the same arm run quiet.

A paper about reproducible measurement should be explicit about the limits of its own artifact,
which is what these notes are for.
