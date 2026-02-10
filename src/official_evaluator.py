"""
Lightweight evaluator for the official MLSys 2026 schedule format.
Implements the model described in PROBLEM.md for single-op subgraphs and
no intra-subgraph reuse when traversal_order is null.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
from src.official_types import OfficialProblem, OfficialGranularity, OfficialSubgraph


def build_op_dependencies(problem: OfficialProblem) -> Dict[int, Set[int]]:
    producer: Dict[int, int] = {}
    for op_idx, op in enumerate(problem.ops):
        for out in op.outputs:
            producer[out] = op_idx

    deps: Dict[int, Set[int]] = {i: set() for i in range(len(problem.ops))}
    for op_idx, op in enumerate(problem.ops):
        for inp in op.inputs:
            if inp in producer:
                deps[op_idx].add(producer[inp])
    return deps


def topological_sort(problem: OfficialProblem) -> List[int]:
    deps = build_op_dependencies(problem)
    in_degree = {i: len(deps[i]) for i in deps}
    ready = [i for i, deg in in_degree.items() if deg == 0]
    order: List[int] = []
    while ready:
        op = ready.pop()
        order.append(op)
        for nxt, dset in deps.items():
            if op in dset:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)
    if len(order) != len(problem.ops):
        raise ValueError("Cycle detected in graph")
    return order


def compute_output_dims(problem: OfficialProblem, op_idx: int) -> Tuple[int, int]:
    op = problem.ops[op_idx]
    if not op.outputs:
        raise ValueError(f"Op {op_idx} has no outputs")
    out_tensor = problem.tensors[op.outputs[0]]
    return out_tensor.width, out_tensor.height


def compute_k_dimension(problem: OfficialProblem, op_idx: int) -> int:
    op = problem.ops[op_idx]
    if op.op_type != "MatMul":
        return 1
    if len(op.inputs) != 2:
        raise ValueError(f"MatMul op {op_idx} must have 2 inputs")
    lhs = problem.tensors[op.inputs[0]]
    return lhs.width


def build_tensor_producers(problem: OfficialProblem) -> Dict[int, int]:
    producer: Dict[int, int] = {}
    for op_idx, op in enumerate(problem.ops):
        for out in op.outputs:
            producer[out] = op_idx
    return producer


def build_tensor_consumers(problem: OfficialProblem) -> Dict[int, List[int]]:
    consumers: Dict[int, List[int]] = {}
    for op_idx, op in enumerate(problem.ops):
        for inp in op.inputs:
            consumers.setdefault(inp, []).append(op_idx)
    return consumers


def compute_boundary_sets(
    problem: OfficialProblem,
    subgraph_ops: List[int],
) -> Tuple[Set[int], Set[int]]:
    producers = build_tensor_producers(problem)
    consumers = build_tensor_consumers(problem)
    op_set = set(subgraph_ops)

    boundary_inputs: Set[int] = set()
    for op_idx in subgraph_ops:
        for inp in problem.ops[op_idx].inputs:
            if producers.get(inp) not in op_set:
                boundary_inputs.add(inp)

    boundary_outputs: Set[int] = set()
    for op_idx in subgraph_ops:
        for out in problem.ops[op_idx].outputs:
            for consumer in consumers.get(out, []):
                if consumer not in op_set:
                    boundary_outputs.add(out)
                    break
            if out not in consumers:
                boundary_outputs.add(out)

    return boundary_inputs, boundary_outputs


def working_set_bytes(problem: OfficialProblem, op_idx: int, gran: OfficialGranularity) -> int:
    op = problem.ops[op_idx]
    w = gran.width
    h = gran.height
    k = gran.depth if op.op_type == "MatMul" else 1
    if op.op_type == "Pointwise":
        num_inputs = len(op.inputs)
        num_outputs = len(op.outputs)
        return (num_inputs + num_outputs) * w * h
    if op.op_type == "MatMul":
        return (h * k) + (k * w) + (w * h)
    raise ValueError(f"Unknown op type: {op.op_type}")


def compute_subgraph_latency(
    problem: OfficialProblem,
    subgraph_ops: List[int],
    gran: OfficialGranularity,
    tensors_to_retain: List[int],
    traversal_order: Optional[List[int]],
    resident_tensors: Optional[Set[int]] = None,
) -> float:
    if len(subgraph_ops) == 0:
        return 0.0

    resident = set(resident_tensors or [])
    retain_set = set(tensors_to_retain or [])
    boundary_inputs_all, boundary_outputs_all = compute_boundary_sets(problem, subgraph_ops)
    boundary_inputs_all = {t for t in boundary_inputs_all if t not in resident}
    boundary_outputs_all = {t for t in boundary_outputs_all if t not in retain_set}

    # Fast path: all pointwise ops can be fused in this baseline model.
    if all(problem.ops[op_idx].op_type == "Pointwise" for op_idx in subgraph_ops):
        # Fast path if all outputs match; otherwise fall back to general model.
        base_w, base_h = compute_output_dims(problem, subgraph_ops[0])
        if all(compute_output_dims(problem, op_idx) == (base_w, base_h) for op_idx in subgraph_ops):
            w = gran.width
            h = gran.height
            tiles_x = math.ceil(base_w / w)
            tiles_y = math.ceil(base_h / h)
            spatial_tiles = tiles_x * tiles_y

            memory_in = (len(boundary_inputs_all) * w * h) / problem.slow_memory_bandwidth
            memory_out = (len(boundary_outputs_all) * w * h) / problem.slow_memory_bandwidth
            compute = sum(problem.ops[op_idx].base_cost for op_idx in subgraph_ops)
            tile_latency = max(compute, memory_in + memory_out)
            return spatial_tiles * tile_latency

    # MatMul + Pointwise fusion (single MatMul followed by one or more Pointwise ops).
    matmul_ops = [op_idx for op_idx in subgraph_ops if problem.ops[op_idx].op_type == "MatMul"]
    if len(matmul_ops) == 1 and all(
        problem.ops[op_idx].op_type in ("MatMul", "Pointwise") for op_idx in subgraph_ops
    ):
        matmul_idx = matmul_ops[0]
        if subgraph_ops[0] != matmul_idx:
            raise ValueError("MatMul must be first in fused subgraph")
        # Ensure pointwise outputs match matmul output shape; otherwise fall back.
        base_w, base_h = compute_output_dims(problem, matmul_idx)
        if all(compute_output_dims(problem, op_idx) == (base_w, base_h) for op_idx in subgraph_ops[1:]):
            w = gran.width
            h = gran.height
            k = gran.depth
            tiles_x = math.ceil(base_w / w)
            tiles_y = math.ceil(base_h / h)
            spatial_tiles = tiles_x * tiles_y

            bandwidth = problem.slow_memory_bandwidth
            if bandwidth <= 0:
                raise ValueError("slow_memory_bandwidth must be positive")

            K = compute_k_dimension(problem, matmul_idx)
            if K <= 0:
                raise ValueError("Invalid K dimension")

            matmul_inputs = set(problem.ops[matmul_idx].inputs)
            extra_inputs = boundary_inputs_all - matmul_inputs

            pointwise_cost = sum(
                problem.ops[op_idx].base_cost
                for op_idx in subgraph_ops
                if problem.ops[op_idx].op_type == "Pointwise"
            )

            k_tiles = math.ceil(K / k)
            total_per_tile = 0.0
            for step in range(k_tiles):
                k_eff = min(k, K - step * k)
                memory_in = 0.0
                if problem.ops[matmul_idx].inputs[0] in boundary_inputs_all:
                    memory_in += (h * k_eff) / bandwidth
                if problem.ops[matmul_idx].inputs[1] in boundary_inputs_all:
                    memory_in += (k_eff * w) / bandwidth

                memory_out = 0.0
                compute = problem.ops[matmul_idx].base_cost * (k_eff / K)

                if step == k_tiles - 1:
                    if extra_inputs:
                        memory_in += (len(extra_inputs) * w * h) / bandwidth
                    if boundary_outputs_all:
                        memory_out = (len(boundary_outputs_all) * w * h) / bandwidth
                    compute += pointwise_cost

                total_per_tile += max(compute, memory_in + memory_out)

            return spatial_tiles * total_per_tile

    # General mixed-op model: execute ops in order, summing per-op roofline.
    total_latency = 0.0
    bandwidth = problem.slow_memory_bandwidth
    if bandwidth <= 0:
        raise ValueError("slow_memory_bandwidth must be positive")

    for op_idx in subgraph_ops:
        op = problem.ops[op_idx]
        out_w, out_h = compute_output_dims(problem, op_idx)
        w = gran.width
        h = gran.height
        tiles_x = math.ceil(out_w / w)
        tiles_y = math.ceil(out_h / h)
        spatial_tiles = tiles_x * tiles_y

        boundary_inputs = [t for t in op.inputs if t in boundary_inputs_all]
        boundary_outputs = [t for t in op.outputs if t in boundary_outputs_all]

        if op.op_type == "Pointwise":
            memory_in = (len(boundary_inputs) * w * h) / bandwidth
            memory_out = (len(boundary_outputs) * w * h) / bandwidth
            compute = op.base_cost
            tile_latency = max(compute, memory_in + memory_out)
            total_latency += spatial_tiles * tile_latency
            continue

        if op.op_type == "MatMul":
            K = compute_k_dimension(problem, op_idx)
            if K <= 0:
                raise ValueError("Invalid K dimension")
            k = gran.depth
            k_tiles = math.ceil(K / k)
            total_per_tile = 0.0
            for step in range(k_tiles):
                k_eff = min(k, K - step * k)
                memory_in = 0.0
                if op.inputs[0] in boundary_inputs:
                    memory_in += (h * k_eff) / bandwidth
                if op.inputs[1] in boundary_inputs:
                    memory_in += (k_eff * w) / bandwidth
                memory_out = 0.0
                if step == k_tiles - 1 and boundary_outputs:
                    memory_out = (len(boundary_outputs) * w * h) / bandwidth
                compute = op.base_cost * (k_eff / K)
                total_per_tile += max(compute, memory_in + memory_out)
            total_latency += spatial_tiles * total_per_tile
            continue

        raise ValueError(f"Unknown op type: {op.op_type}")

    return total_latency

    # Unreachable: handled by early returns above.


def evaluate_solution(problem: OfficialProblem, subgraphs: List[OfficialSubgraph]) -> float:
    total = 0.0
    resident: Set[int] = set()
    for sg in subgraphs:
        total += compute_subgraph_latency(
            problem,
            sg.ops,
            sg.granularity,
            sg.tensors_to_retain,
            sg.traversal_order,
            resident_tensors=resident,
        )
        resident = set(sg.tensors_to_retain)
    return total


def _boundary_bytes_for_subgraph(
    problem: OfficialProblem,
    subgraph_ops: List[int],
    gran: OfficialGranularity,
    resident: Set[int],
    retained: Set[int],
) -> Tuple[int, int]:
    boundary_inputs_all, boundary_outputs_all = compute_boundary_sets(problem, subgraph_ops)
    boundary_inputs_all = {t for t in boundary_inputs_all if t not in resident}
    boundary_outputs_all = {t for t in boundary_outputs_all if t not in retained}

    total_in = 0
    total_out = 0
    for op_idx in subgraph_ops:
        op = problem.ops[op_idx]
        out_w, out_h = compute_output_dims(problem, op_idx)
        w = min(gran.width, out_w)
        h = min(gran.height, out_h)
        tiles_x = math.ceil(out_w / w)
        tiles_y = math.ceil(out_h / h)
        spatial_tiles = tiles_x * tiles_y

        boundary_inputs = [t for t in op.inputs if t in boundary_inputs_all]
        boundary_outputs = [t for t in op.outputs if t in boundary_outputs_all]

        if op.op_type == "Pointwise":
            total_in += len(boundary_inputs) * w * h * spatial_tiles
            total_out += len(boundary_outputs) * w * h * spatial_tiles
            continue

        if op.op_type == "MatMul":
            K = compute_k_dimension(problem, op_idx)
            if K <= 0:
                raise ValueError("Invalid K dimension")
            lhs, rhs = op.inputs
            if lhs in boundary_inputs:
                total_in += h * K * spatial_tiles
            if rhs in boundary_inputs:
                total_in += w * K * spatial_tiles
            if boundary_outputs:
                total_out += w * h * spatial_tiles
            continue

        raise ValueError(f"Unknown op type: {op.op_type}")

    return total_in, total_out


def analyze_retention(problem: OfficialProblem, subgraphs: List[OfficialSubgraph]) -> Dict[str, float]:
    resident: Set[int] = set()
    retained_bytes_total = 0
    peak_resident_bytes = 0
    load_saved_bytes = 0
    store_saved_bytes = 0

    for sg in subgraphs:
        retained = set(sg.tensors_to_retain)
        retained_bytes = sum(
            problem.tensors[t].width * problem.tensors[t].height for t in retained
        )
        retained_bytes_total += retained_bytes
        peak_resident_bytes = max(peak_resident_bytes, retained_bytes)

        base_in, base_out = _boundary_bytes_for_subgraph(
            problem,
            sg.ops,
            sg.granularity,
            resident=set(),
            retained=set(),
        )
        with_in, with_out = _boundary_bytes_for_subgraph(
            problem,
            sg.ops,
            sg.granularity,
            resident=resident,
            retained=retained,
        )
        load_saved_bytes += max(0, base_in - with_in)
        store_saved_bytes += max(0, base_out - with_out)

        resident = retained

    avg_retained_bytes = retained_bytes_total / max(1, len(subgraphs))
    return {
        'retained_bytes_total': retained_bytes_total,
        'retained_bytes_avg': avg_retained_bytes,
        'peak_resident_bytes': peak_resident_bytes,
        'load_saved_bytes': load_saved_bytes,
        'store_saved_bytes': store_saved_bytes,
    }
