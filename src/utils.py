"""
Utility functions for schedule validation and JSON I/O.
"""

import json
from typing import List, Dict, Set
from src.parser import Problem
from src.memory import MemoryTracker
from src.graph import ComputationGraph
from src.format import write_schedule
from src.cost_model import CostModel


def save_schedule(schedule: List[Dict], filepath: str) -> None:
    """
    Save a schedule to JSON file.
    
    Args:
        schedule: List of action dictionaries
        filepath: Output file path
    """
    write_schedule(schedule, filepath)


def load_schedule(filepath: str) -> List[Dict]:
    """
    Load a schedule from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of action dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['actions']


def validate_schedule(
    schedule: List[Dict],
    problem: Problem,
    allow_free: bool = False,
    require_final_outputs: bool = True,
) -> bool:
    """
    Validate that a schedule is legal.
    
    Checks:
    1. Memory never exceeds capacity
    2. Operations only compute when inputs are loaded
    3. No invalid action types
    4. Loads only from DRAM
    5. Final outputs are stored in DRAM (when required)
    
    Args:
        schedule: List of action dictionaries
        problem: Problem specification
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    memory = MemoryTracker(problem)
    computed_ops = set()

    graph = ComputationGraph(problem)
    input_tensors = graph.get_input_tensors()
    if problem.outputs:
        final_outputs = set(problem.outputs)
    else:
        final_outputs = set(
            tensor for tensor in problem.tensors
            if tensor not in graph.tensor_consumers
        )

    in_sram = {tensor: False for tensor in problem.tensors}
    in_dram = {tensor: False for tensor in problem.tensors}
    for tensor in input_tensors:
        in_dram[tensor] = True

    def _auto_drop(required: Set[str]) -> None:
        """Implicitly drop DRAM-backed tensors to make space."""
        while True:
            candidates = [
                t for t in memory.resident_tensors
                if in_dram.get(t, False) and t not in required
            ]
            if not candidates:
                return
            # Drop a largest DRAM-backed tensor first
            evict_tensor = max(candidates, key=lambda t: problem.tensors[t].size)
            memory.store(evict_tensor)
            in_sram[evict_tensor] = False

    allowed_types = {'load', 'store', 'compute'}
    if allow_free:
        allowed_types.add('free')
    
    for i, action in enumerate(schedule):
        action_type = action.get('type')

        if action_type not in allowed_types:
            raise ValueError(f"Action {i}: unknown action type '{action_type}'")
        
        if action_type == 'load':
            tensor_name = action.get('tensor')
            if not tensor_name:
                raise ValueError(f"Action {i}: load action missing 'tensor' field")
            
            if tensor_name not in problem.tensors:
                raise ValueError(f"Action {i}: unknown tensor '{tensor_name}'")
            
            if memory.is_loaded(tensor_name):
                raise ValueError(f"Action {i}: tensor '{tensor_name}' is already loaded")

            if not in_dram.get(tensor_name, False):
                raise ValueError(f"Action {i}: tensor '{tensor_name}' not available in DRAM")
            
            if not memory.can_load(tensor_name):
                _auto_drop(required=set())
                if not memory.can_load(tensor_name):
                    raise ValueError(f"Action {i}: loading '{tensor_name}' would exceed capacity")
            
            memory.load(tensor_name)
            in_sram[tensor_name] = True
        
        elif action_type == 'store':
            tensor_name = action.get('tensor')
            if not tensor_name:
                raise ValueError(f"Action {i}: store action missing 'tensor' field")
            
            if not memory.is_loaded(tensor_name):
                raise ValueError(f"Action {i}: tensor '{tensor_name}' is not loaded")
            
            memory.store(tensor_name)
            in_sram[tensor_name] = False
            in_dram[tensor_name] = True

        elif action_type == 'free':
            tensor_name = action.get('tensor')
            if not tensor_name:
                raise ValueError(f"Action {i}: free action missing 'tensor' field")

            if not memory.is_loaded(tensor_name):
                raise ValueError(f"Action {i}: tensor '{tensor_name}' is not loaded")

            memory.store(tensor_name)
            in_sram[tensor_name] = False
        
        elif action_type == 'compute':
            op_name = action.get('op')
            if not op_name:
                raise ValueError(f"Action {i}: compute action missing 'op' field")
            
            if op_name not in problem.operations:
                raise ValueError(f"Action {i}: unknown operation '{op_name}'")
            
            if op_name in computed_ops:
                raise ValueError(f"Action {i}: operation '{op_name}' already computed")
            
            op = problem.operations[op_name]
            
            # Check all inputs are loaded
            for input_tensor in op.inputs:
                if not memory.is_loaded(input_tensor):
                    raise ValueError(f"Action {i}: input '{input_tensor}' not loaded for op '{op_name}'")
            
            # Mark operation as computed
            computed_ops.add(op_name)
            
            # Add outputs to memory (they're produced by the compute)
            for output_tensor in op.outputs:
                tensor_size = problem.tensors[output_tensor].size
                
                # Check if there's space (outputs are produced, not loaded)
                if memory.get_available_space() < tensor_size:
                    _auto_drop(required=set(op.inputs))
                    if memory.get_available_space() < tensor_size:
                        raise ValueError(f"Action {i}: insufficient space for output '{output_tensor}'")
                
                memory.resident_tensors.add(output_tensor)
                memory.current_usage += tensor_size
                in_sram[output_tensor] = True
                in_dram[output_tensor] = False
        
    # Check that all operations were computed
    if len(computed_ops) != len(problem.operations):
        missing = set(problem.operations.keys()) - computed_ops
        raise ValueError(f"Not all operations were computed. Missing: {missing}")

    if require_final_outputs:
        missing_outputs = [t for t in final_outputs if not in_dram.get(t, False)]
        if missing_outputs:
            raise ValueError(f"Final outputs not stored in DRAM: {missing_outputs}")
    
    return True


def analyze_schedule(
    schedule: List[Dict],
    problem: Problem,
    allow_free: bool = False,
) -> Dict:
    """
    Analyze a schedule and return metrics for reporting.
    """
    graph = ComputationGraph(problem)
    input_tensors = graph.get_input_tensors()
    if problem.outputs:
        final_outputs = set(problem.outputs)
    else:
        final_outputs = set(
            tensor for tensor in problem.tensors
            if tensor not in graph.tensor_consumers
        )

    memory = MemoryTracker(problem)
    in_dram = {tensor: False for tensor in problem.tensors}
    for tensor in input_tensors:
        in_dram[tensor] = True

    def _auto_drop(required: Set[str]) -> None:
        while True:
            candidates = [
                t for t in memory.resident_tensors
                if in_dram.get(t, False) and t not in required
            ]
            if not candidates:
                return
            evict_tensor = max(candidates, key=lambda t: problem.tensors[t].size)
            memory.store(evict_tensor)

    load_count = 0
    store_count = 0
    compute_count = 0
    free_count = 0
    total_loaded_bytes = 0
    total_stored_bytes = 0
    required_output_stores = 0
    peak_sram_bytes = 0

    for action in schedule:
        action_type = action.get('type')

        if action_type == 'load':
            tensor_name = action['tensor']
            if not memory.can_load(tensor_name):
                _auto_drop(required=set())
            memory.load(tensor_name)
            load_count += 1
            total_loaded_bytes += problem.tensors[tensor_name].size

        elif action_type == 'store':
            tensor_name = action['tensor']
            memory.store(tensor_name)
            in_dram[tensor_name] = True
            store_count += 1
            total_stored_bytes += problem.tensors[tensor_name].size
            if tensor_name in final_outputs:
                required_output_stores += 1

        elif action_type == 'free':
            tensor_name = action['tensor']
            memory.store(tensor_name)
            free_count += 1

        elif action_type == 'compute':
            op_name = action['op']
            op = problem.operations[op_name]
            for output_tensor in op.outputs:
                tensor_size = problem.tensors[output_tensor].size
                if memory.get_available_space() < tensor_size:
                    _auto_drop(required=set(op.inputs))
                memory.resident_tensors.add(output_tensor)
                memory.current_usage += tensor_size
                in_dram[output_tensor] = False
            compute_count += 1

        peak_sram_bytes = max(peak_sram_bytes, memory.current_usage)

    spill_stores = max(0, store_count - required_output_stores)
    peak_sram_pct = (
        peak_sram_bytes / problem.scratchpad_capacity
        if problem.scratchpad_capacity > 0
        else 0.0
    )
    loaded_bytes_per_compute = (
        total_loaded_bytes / compute_count if compute_count > 0 else 0.0
    )

    cost_model = CostModel(problem, allow_free=allow_free)
    latency = cost_model.evaluate_schedule(schedule)

    return {
        'loads': load_count,
        'stores': store_count,
        'computes': compute_count,
        'free': free_count,
        'total_loaded_bytes': total_loaded_bytes,
        'total_stored_bytes': total_stored_bytes,
        'required_output_stores': required_output_stores,
        'spill_stores': spill_stores,
        'peak_sram_bytes': peak_sram_bytes,
        'peak_sram_pct': peak_sram_pct,
        'loaded_bytes_per_compute': loaded_bytes_per_compute,
        'latency': latency,
    }


def print_schedule_summary(
    schedule: List[Dict],
    problem: Problem,
    allow_free: bool = False,
) -> None:
    """
    Print a human-readable summary of a schedule.
    
    Args:
        schedule: List of action dictionaries
        problem: Problem specification
    """
    from src.cost_model import CostModel
    
    cost_model = CostModel(problem, allow_free=allow_free)
    breakdown = cost_model.get_breakdown(schedule)
    
    print("=" * 60)
    print("SCHEDULE SUMMARY")
    print("=" * 60)
    print(f"Total actions: {len(schedule)}")
    print(f"Load operations: {breakdown['load_count']}")
    print(f"Store operations: {breakdown['store_count']}")
    if breakdown.get('free_count', 0) > 0:
        print(f"Free operations: {breakdown['free_count']}")
    print(f"Compute operations: {len([a for a in schedule if a['type'] == 'compute'])}")
    print()
    print("COST BREAKDOWN")
    print("-" * 60)
    print(f"Load cost:    {breakdown['load']:>8} ({breakdown['load_count']} × {problem.load_cost})")
    print(f"Store cost:   {breakdown['store']:>8} ({breakdown['store_count']} × {problem.store_cost})")
    if breakdown.get('free_count', 0) > 0:
        print(f"Free cost:    {breakdown['free']:>8} ({breakdown['free_count']} × 0)")
    print(f"Compute cost: {breakdown['compute']:>8}")
    print("-" * 60)
    print(f"TOTAL LATENCY: {breakdown['total']}")
    print("=" * 60)
