# MLSys 2026 Scheduler

A practical scheduler for the MLSys 2026 Graph Scheduling Competition. It generates **official** schedule JSONs (subgraphs, granularities, latencies) and includes a Track B agent wrapper.

**Overview**
- Reads official MLSys problem JSONs (DAG + tensor shapes + hardware specs)
- Builds subgraphs and chooses per-subgraph granularity
- Optionally retains tensors across subgraphs to reduce boundary traffic
- Writes official schedule JSONs and an optional report

## Install
```bash
pip install -r requirements.txt
```

## Track B (Agent Reasoning) — Submission Entry Point
The official Track B entrypoint is `agent.py`:
```bash
export GOOGLE_API_KEY="..."
export GEMINI_MODEL="models/gemini-2.5-flash"  # optional override
python3 agent.py <input.json> <output.json>
```
- The agent makes a Gemini call to choose scheduling parameters.
- It always outputs **strict** official JSON (no extra keys).

**Track B packaging**
```bash
zip -r MLSys_trackB.zip agent.py requirements.txt prompts writeup.pdf
```

## Local Schedule Generation (Official)
```bash
python run.py --all ../MLSys_official/benchmarks \
  --official --official-scheduler fuse \
  --connectivity auto --retain-budget 0.05 \
  --tag SUBMIT --report schedules/report_SUBMIT.json \
  --submission
```
Schedules are written to `schedules/*_SUBMIT_schedule.json`.

## Quick Start (Single Benchmark)
```bash
python run.py ../MLSys_official/benchmarks/mlsys-2026-1.json \
  --official --official-scheduler fuse \
  --connectivity auto --retain-budget 0.05 \
  --tag v1 --report schedules/report_v1.json
```

## Key Flags (Official Mode)
- `--official` Force official input format.
- `--submission` Write strict official JSON (no extra metadata fields).
- `--official-scheduler {baseline,fuse}` `baseline` = one-op subgraphs, `fuse` = greedy fusion with granularity search.
- `--connectivity {strict,loose,auto}` `strict` = dependency-connected fusions only, `loose` = boundary-connected fusions with component splitting, `auto` = run both and choose lower total latency.
- `--retain-budget <0.0-0.9>` Fraction of SRAM reserved for retained tensors between subgraphs.
- `--fuse-accept-slack <float>` Allow fused latency up to `(1 + slack)` of separate latency.
- `--debug-fuse` / `--debug-fuse-file` Write merge decisions for `mlsys-2026-17` only.
- `--tag <name>` / `--report <path>` Tag schedule filenames and write a run summary JSON.

## Outputs
Each run produces:
- A schedule JSON under `schedules/` in official format
- A report JSON (if `--report` is provided) with latency, connectivity, and retention metadata

**Strict output** (when `--submission` is set) contains only:
- `subgraphs`
- `granularities`
- `tensors_to_retain`
- `traversal_orders`
- `subgraph_latencies`

## Project Layout
```
MLSys/
├── agent.py                  # Track B entrypoint
├── prompts/                  # Track B prompts
├── problems/                 # Local toy problems (legacy)
├── schedules/                # Generated schedules and reports
├── src/
│   ├── official_parser.py    # Official input parser
│   ├── official_scheduler.py # Fusion + granularity search
│   ├── official_evaluator.py # Local evaluator (roofline model)
│   ├── official_format.py    # Official schedule writer
│   ├── scheduler.py          # Legacy toy scheduler
│   └── utils.py              # Validation and reporting
├── run.py                    # CLI entry point
├── writeup.md                # Technical writeup (developer)
└── writeup.pdf               # Track B writeup
```

## Legacy Toy Mode (Optional)
The repository still supports the original load/store/compute toy scheduler:
```bash
python run.py problems/example_problem.json --scheduler optimized
```
This is **not** the official competition format. Use `--official` for benchmarks.

## Notes
- The local evaluator in `src/official_evaluator.py` follows the official `PROBLEM.md` model.
- Traversal-order reuse is not modeled; `traversal_orders` are emitted but reuse gains are not simulated.
