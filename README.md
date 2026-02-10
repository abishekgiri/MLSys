# MLSys 2026 Scheduler

A high-performance scheduler for the MLSys 2026 Graph Scheduling Competition. It generates **official** schedule JSONs (subgraphs + granularities + latencies) and includes a legacy **toy** scheduler for local experimentation.

## What This Repo Does

- Reads an official MLSys problem JSON (DAG of ops + tensor shapes + memory/bandwidth)
- Builds a subgraph execution schedule
- Chooses per-subgraph granularity (tile sizes)
- Optionally retains tensors across subgraphs to reduce boundary traffic
- Writes the official schedule JSON and a run report

## Quick Start (Official Benchmarks)

```bash
# Run one benchmark
python run.py ../MLSys_official/benchmarks/mlsys-2026-1.json \
  --official --official-scheduler fuse \
  --connectivity auto --retain-budget 0.05 \
  --tag v1 --report schedules/report_v1.json

# Run all benchmarks in a directory
python run.py --all ../MLSys_official/benchmarks \
  --official --official-scheduler fuse \
  --connectivity auto --retain-budget 0.05 \
  --tag v1 --report schedules/report_v1.json
```

### Key Flags (Official Mode)

- `--official`
  Force official input format.
- `--official-scheduler {baseline,fuse}`
  `baseline` = one-op subgraphs, `fuse` = greedy subgraph fusion.
- `--connectivity {strict,loose,auto}`
  `strict` = only dependency-connected fusions.
  `loose` = boundary-connected fusions, split into connected components if needed.
  `auto` = run both and choose lower total latency.
- `--retain-budget <0.0-0.9>`
  Fraction of SRAM reserved for retained tensors between subgraphs.
- `--fuse-accept-slack <float>`
  Allow fused latency up to `(1 + slack)` of separate latency.
- `--debug-fuse` + `--debug-fuse-file`
  Write merge decisions for `mlsys-2026-17` only.
- `--tag <name>` / `--report <path>`
  Tag schedule filenames and write run summary JSON.

## Outputs

Each run produces:
- A schedule JSON under `schedules/` (official format)
- A report JSON (if `--report` is provided) with latency + retention + connectivity metadata

Official schedule JSON includes:
- `subgraphs`
- `granularities`
- `tensors_to_retain`
- `traversal_orders`
- `subgraph_latencies`
- `connectivity_requested`, `connectivity_chosen`, `latency_strict`, `latency_loose`, `loose_components_split`

## Project Layout

```
MLSys/
├── problems/                 # Local toy problems (legacy)
├── schedules/                # Generated schedules and reports
├── src/
│   ├── official_parser.py    # Official input parser
│   ├── official_scheduler.py # Fusion + granularity search
│   ├── official_evaluator.py # Local evaluator (roofline model)
│   ├── official_format.py    # Official schedule writer
│   ├── scheduler.py          # Legacy toy scheduler
│   └── utils.py              # Validation, reporting
├── run.py                    # CLI entry point
├── SUBMISSION_GUIDE.md       # Packaging / submission notes
└── writeup.md                # Technical writeup
```

## Legacy Toy Mode (Optional)

The repository still supports the original “load/store/compute” toy scheduler for quick experimentation:

```bash
python run.py problems/example_problem.json --scheduler optimized
```

This is **not** the official competition format. Use `--official` for benchmarks.

## Notes

- The local official evaluator in `src/official_evaluator.py` follows the specification in `PROBLEM.md` from the official repo.
- Traversal-order reuse is not modeled; `traversal_orders` are emitted but reuse gains are not simulated.

