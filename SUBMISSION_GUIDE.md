# Submission Guide (Local)

This repo focuses on generating **official MLSys 2026 schedules** (subgraphs + granularities + latencies). Packaging requirements vary by competition track. Follow the official competition rules for final submission; use this guide for reproducible runs and artifacts.

## Reproducible Run

```bash
python run.py --all ../MLSys_official/benchmarks \
  --official --official-scheduler fuse \
  --connectivity auto --retain-budget 0.05 \
  --tag final --report schedules/report_FINAL.json
```

Artifacts:
- `schedules/*_final_schedule.json` (one per benchmark)
- `schedules/report_FINAL.json`

## Recommended Flags

- `--official` — enforce official input format
- `--official-scheduler fuse` — enable fusion + granularity search
- `--connectivity auto` — choose best of strict vs loose
- `--retain-budget 0.05` — small SRAM reservation for cross-subgraph reuse

## Debugging / Verification

- Add `--debug-fuse` to log merge decisions for `mlsys-2026-17`.
- Add `--report` to capture connectivity decisions and retention stats.

## Files to Include (Typical)

Minimum set to reproduce results:
- `run.py`
- `src/` (all Python sources)
- `requirements.txt`
- `writeup.md` (or converted PDF if required)

If the competition requires a specific binary or agent wrapper, follow the official instructions and use this repo as the implementation core.

