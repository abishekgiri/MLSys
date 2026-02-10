"""
Problem JSON parser for MLSys 2026 Graph Scheduling Competition.
Reads and validates problem specification files.
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Tensor:
    """Represents a tensor with its size."""
    name: str
    size: int


@dataclass
class Operation:
    """Represents a computational operation."""
    name: str
    inputs: List[str]
    outputs: List[str]
    compute_cost: int


@dataclass
class Problem:
    """Complete problem specification."""
    scratchpad_capacity: int
    tensors: Dict[str, Tensor]
    operations: Dict[str, Operation]
    edges: Dict[str, List[str]]
    load_cost: int
    store_cost: int
    outputs: List[str] = None


def parse_problem(filepath: str) -> Problem:
    """
    Parse a problem JSON file and return a Problem object.
    
    Args:
        filepath: Path to the problem JSON file
        
    Returns:
        Problem object containing all specification data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is malformed or invalid
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = ['scratchpad_capacity', 'tensors', 'operations', 'edges', 'load_cost', 'store_cost']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Parse tensors
    tensors = {}
    for name, tensor_data in data['tensors'].items():
        tensors[name] = Tensor(name=name, size=tensor_data['size'])
    
    # Parse operations
    operations = {}
    for name, op_data in data['operations'].items():
        operations[name] = Operation(
            name=name,
            inputs=op_data['inputs'],
            outputs=op_data['outputs'],
            compute_cost=op_data['compute_cost']
        )
    
    return Problem(
        scratchpad_capacity=data['scratchpad_capacity'],
        tensors=tensors,
        operations=operations,
        edges=data['edges'],
        load_cost=data['load_cost'],
        store_cost=data['store_cost'],
        outputs=data.get('outputs')
    )


def validate_problem(problem: Problem) -> bool:
    """
    Validate that a problem specification is internally consistent.
    
    Args:
        problem: Problem object to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check that all operation inputs/outputs reference valid tensors
    for op_name, op in problem.operations.items():
        for tensor_name in op.inputs + op.outputs:
            if tensor_name not in problem.tensors:
                raise ValueError(f"Operation {op_name} references unknown tensor: {tensor_name}")

    if problem.outputs:
        for tensor_name in problem.outputs:
            if tensor_name not in problem.tensors:
                raise ValueError(f"Problem outputs reference unknown tensor: {tensor_name}")
    
    # Check that scratchpad capacity is positive
    if problem.scratchpad_capacity <= 0:
        raise ValueError("Scratchpad capacity must be positive")
    
    # Check that all costs are non-negative
    if problem.load_cost < 0 or problem.store_cost < 0:
        raise ValueError("Load and store costs must be non-negative")
    
    for op in problem.operations.values():
        if op.compute_cost < 0:
            raise ValueError(f"Compute cost for {op.name} must be non-negative")
    
    return True
