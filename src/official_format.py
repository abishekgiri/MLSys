"""
Writer for the official MLSys 2026 schedule format.
"""

import json
from typing import List, Optional
from src.official_types import OfficialGranularity


def write_official_schedule(
    filepath: str,
    subgraphs: List[List[int]],
    granularities: List[OfficialGranularity],
    tensors_to_retain: List[List[int]],
    traversal_orders: List[Optional[List[int]]],
    subgraph_latencies: List[float],
    strict_output: bool = False,
    connectivity_requested: str = "strict",
    connectivity_chosen: str = "strict",
    latency_strict: Optional[float] = None,
    latency_loose: Optional[float] = None,
    loose_components_split: int = 0,
) -> None:
    output = {
        'subgraphs': subgraphs,
        'granularities': [
            [g.width, g.height, g.depth] for g in granularities
        ],
        'tensors_to_retain': tensors_to_retain,
        'traversal_orders': traversal_orders,
        'subgraph_latencies': subgraph_latencies,
    }
    if not strict_output:
        output.update({
            'connectivity_requested': connectivity_requested,
            'connectivity_chosen': connectivity_chosen,
            'latency_strict': latency_strict,
            'latency_loose': latency_loose,
            'loose_components_split': loose_components_split,
        })
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
