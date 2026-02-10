"""
Data structures for the official MLSys 2026 scheduler format.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OfficialTensor:
    width: int
    height: int


@dataclass
class OfficialOp:
    op_type: str
    inputs: List[int]
    outputs: List[int]
    base_cost: int


@dataclass
class OfficialGranularity:
    width: int
    height: int
    depth: int


@dataclass
class OfficialProblem:
    tensors: List[OfficialTensor]
    ops: List[OfficialOp]
    fast_memory_capacity: int
    slow_memory_bandwidth: int
    native_granularity: OfficialGranularity


@dataclass
class OfficialSubgraph:
    ops: List[int]
    granularity: OfficialGranularity
    tensors_to_retain: List[int]
    traversal_order: Optional[List[int]]
    subgraph_latency: float


@dataclass
class OfficialSolution:
    subgraphs: List[OfficialSubgraph]
    connectivity_requested: str = "strict"
    connectivity_chosen: str = "strict"
    latency_strict: Optional[float] = None
    latency_loose: Optional[float] = None
    loose_components_split: int = 0
