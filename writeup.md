# Technical Writeup: MLSys 2026 Scheduler

## Problem Summary
We are given a DAG of operations with tensor shapes, op types (MatMul / Pointwise), and hardware parameters: fast memory capacity, slow memory bandwidth, and native execution granularity. The goal is to produce an **official schedule JSON** consisting of subgraphs and per-subgraph granularities that minimize total latency under a roofline model, with subgraphs serialized.

## Approach Overview
The scheduler builds subgraphs greedily, chooses a valid granularity for each, and estimates latency using the official roofline model. It supports multiple connectivity modes and an auto-selection policy.

Key features:
- Frontier-based subgraph growth
- Granularity search over `(w, h, k)` under SRAM constraints
- Optional tensor retention across subgraphs
- Strict vs loose connectivity with an `auto` selector

## Scheduler Design

**1) Frontier-Based Growth**
We maintain a ready set of ops whose dependencies are satisfied. A subgraph starts from a seed op and grows by adding connected, ready candidates when fusion does not degrade latency beyond a configurable slack.

**2) Connectivity Modes**
- **Strict**: Only dependency-connected ops can be fused.
- **Loose**: Boundary-connected ops may be fused. A connectivity check then splits disconnected subgraphs into components for safety.
- **Auto**: Builds both strict and loose schedules and chooses the lower-latency schedule.

**3) Granularity Search**
For each candidate subgraph, we search over `(w, h, k)` derived from the native granularity by halving. A candidate is valid if its working set fits within the compute portion of SRAM (after reserving retention budget). We choose the granularity with minimum estimated latency.

**4) Tensor Retention**
We optionally retain tensors across subgraphs to reduce boundary loads/stores. Retention selection is based on next-use distance and constrained by a configurable `retain_budget` (fraction of SRAM). Retained tensors are accounted for during boundary cost estimation.

## Cost Model (Local Evaluator)
Per subgraph:
- **Compute time**: sum of op base costs, with MatMul scaled by the chosen reduction depth `k`.
- **Memory time**: boundary bytes divided by bandwidth, accounting for resident inputs and retained outputs.
- **Latency**: `max(compute_time, memory_time)` per tile, summed across tiles implied by `(w, h, k)`.

Subgraph latencies are serialized. Traversal-order reuse is not modeled.

## Complexity
Let `V` be the number of ops:
- Scheduling is roughly `O(V * C)` where `C` is the number of candidate fusions evaluated.
- Granularity search is `O(G)` per candidate, where `G` is the number of granularities tested.

Overall runtime is fast enough for all provided benchmarks.

## Practical Results
Fusion yields large improvements on most benchmarks by reducing boundary transfers and subgraph count. Auto connectivity tends to select loose mode when it yields lower latency, while still enforcing connected components for safety.

## Limitations / Future Work
- Traversal-order reuse is not modeled.
- Global search (beam/ILP) could improve results on difficult graphs.
- More advanced retention (multi-step residency planning) may further help memory-bound cases.
