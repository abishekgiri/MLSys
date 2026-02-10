# Technical Writeup: MLSys 2026 Scheduler

## Problem Summary (Official Format)

We are given a DAG of operations with tensor shapes, operation types (MatMul / Pointwise), and hardware parameters: fast memory capacity, slow memory bandwidth, and native execution granularity. The goal is to produce a **schedule JSON** consisting of subgraphs and their execution granularities to minimize total latency under a roofline model (per-subgraph latency is `max(compute_time, memory_time)`; subgraphs are serialized).

## Approach Overview

Our scheduler builds subgraphs greedily, chooses a valid granularity for each, and estimates latency using the official roofline model. The system supports multiple connectivity modes and an auto-selection policy.

**Key features:**
- Frontier-based subgraph growth
- Granularity search (w, h, k) within SRAM capacity
- Optional tensor retention across subgraphs
- Strict vs loose connectivity (with `auto` selection)

## Scheduler Design

### 1. Frontier-Based Growth

We maintain a ready set of ops whose dependencies are satisfied. A subgraph starts from a seed op and grows by adding connected, ready candidates if fusion does not degrade latency beyond the configured slack.

### 2. Connectivity Modes

- **Strict**: Only dependency-connected ops can be fused.
- **Loose**: Boundary-connected ops may be fused. A connectivity check then splits disconnected subgraphs into components for safety.
- **Auto**: Build both strict and loose schedules and choose the lower-latency schedule.

### 3. Granularity Search

For each candidate subgraph, we search over `(w, h, k)` granularities derived from native granularity by halving. A candidate is valid if its working set fits within the compute portion of SRAM (after reserving retention budget). We choose the granularity with minimum estimated latency.

### 4. Tensor Retention

We optionally retain tensors across subgraphs to reduce boundary loads/stores. Retention selection is based on next-use distance and constrained by a configurable `retain_budget` (fraction of SRAM).

## Cost Model (Local Evaluator)

Per subgraph:
- **Compute time**: sum of op base costs (MatMul scales with reduction depth)
- **Memory time**: boundary bytes / bandwidth
- **Latency**: `max(compute_time, memory_time)`

Subgraph latencies are serialized. Traversal-order reuse is not modeled.

## Complexity

Let `V` be number of ops:
- Scheduling: roughly `O(V * C)` where `C` is number of candidate fusions evaluated.
- Granularity search: `O(G)` per candidate, where `G` is the number of granularities tested.

Overall runtime is fast enough for all provided benchmarks.

## Practical Results

Fusion yields significant improvements on most benchmarks by reducing boundary transfers and subgraph count. Strict mode is safest for official evaluation; loose/auto are best-effort for performance and include safety splitting.

## Limitations / Future Work

- Traversal-order reuse is not modeled.
- Global search (beam/ILP) could further improve results on difficult graphs.
- More advanced retention (multi-subgraph residency) may improve memory-bound cases.

