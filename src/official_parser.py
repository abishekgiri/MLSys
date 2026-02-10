"""
Parser for the official MLSys 2026 problem format.
"""

import json
from typing import List
from src.official_types import (
    OfficialTensor,
    OfficialOp,
    OfficialGranularity,
    OfficialProblem,
)


def parse_official_problem(filepath: str) -> OfficialProblem:
    with open(filepath, 'r') as f:
        data = json.load(f)

    required = [
        'widths',
        'heights',
        'inputs',
        'outputs',
        'base_costs',
        'op_types',
        'fast_memory_capacity',
        'slow_memory_bandwidth',
        'native_granularity',
    ]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    widths: List[int] = data['widths']
    heights: List[int] = data['heights']
    if len(widths) != len(heights):
        raise ValueError("widths and heights length mismatch")

    tensors = [OfficialTensor(w, h) for w, h in zip(widths, heights)]

    inputs = data['inputs']
    outputs = data['outputs']
    base_costs = data['base_costs']
    op_types = data['op_types']
    if not (len(inputs) == len(outputs) == len(base_costs) == len(op_types)):
        raise ValueError("Operation fields length mismatch")

    ops = [
        OfficialOp(
            op_type=op_types[i],
            inputs=list(inputs[i]),
            outputs=list(outputs[i]),
            base_cost=base_costs[i],
        )
        for i in range(len(op_types))
    ]

    native = data['native_granularity']
    if len(native) != 2:
        raise ValueError("native_granularity must be [w, h]")

    return OfficialProblem(
        tensors=tensors,
        ops=ops,
        fast_memory_capacity=data['fast_memory_capacity'],
        slow_memory_bandwidth=data['slow_memory_bandwidth'],
        native_granularity=OfficialGranularity(
            width=native[0],
            height=native[1],
            depth=1,
        ),
    )
