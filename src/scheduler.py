"""
Core scheduling logic for MLSys 2026 Graph Scheduling Competition.
Implements naive baseline and optimized scheduling strategies.
"""

import heapq
from typing import List, Dict, Set
from src.parser import Problem
from src.graph import ComputationGraph
from src.memory import MemoryTracker
from src.format import emit_load, emit_store, emit_compute


class NextUseTracker:
    """
    Tracks next-use information for tensors based on a reference order.
    Uses a min-heap of consumer positions with lazy deletion.
    """

    def __init__(self, graph: ComputationGraph, op_index: Dict[str, int]):
        self.op_index = op_index
        self._heaps: Dict[str, List[int]] = {}
        self._removed: Dict[str, Dict[int, int]] = {}
        self._remaining: Dict[str, int] = {}

        for tensor_name, consumers in graph.tensor_consumers.items():
            indices = [op_index[op_name] for op_name in consumers]
            heapq.heapify(indices)
            self._heaps[tensor_name] = indices
            self._removed[tensor_name] = {}
            self._remaining[tensor_name] = len(indices)

    def mark_consumed(self, tensor_name: str, op_name: str) -> None:
        """Mark that an op has consumed a tensor."""
        if tensor_name not in self._heaps:
            return
        idx = self.op_index[op_name]
        removed_map = self._removed[tensor_name]
        removed_map[idx] = removed_map.get(idx, 0) + 1
        if self._remaining[tensor_name] > 0:
            self._remaining[tensor_name] -= 1

    def next_use_index(self, tensor_name: str) -> float:
        """Return the next use index (min) or inf if no remaining uses."""
        heap = self._heaps.get(tensor_name)
        if not heap:
            return float('inf')

        removed_map = self._removed[tensor_name]
        while heap:
            idx = heap[0]
            count = removed_map.get(idx, 0)
            if count == 0:
                return idx
            # Lazy delete one instance
            if count == 1:
                del removed_map[idx]
            else:
                removed_map[idx] = count - 1
            heapq.heappop(heap)

        return float('inf')

    def remaining_uses(self, tensor_name: str) -> int:
        """Return number of remaining consumers for a tensor."""
        return self._remaining.get(tensor_name, 0)

    def clone(self) -> "NextUseTracker":
        """Deep copy tracker state for simulation."""
        clone = NextUseTracker.__new__(NextUseTracker)
        clone.op_index = self.op_index
        clone._heaps = {name: list(heap) for name, heap in self._heaps.items()}
        clone._removed = {name: dict(removed) for name, removed in self._removed.items()}
        clone._remaining = dict(self._remaining)
        return clone


class BaseScheduler:
    """Base class for schedulers."""
    
    def __init__(self, problem: Problem):
        """
        Initialize scheduler.
        
        Args:
            problem: Problem specification
        """
        self.problem = problem
        self.graph = ComputationGraph(problem)
        self.memory = MemoryTracker(problem)
    
    def schedule(self) -> List[Dict]:
        """
        Generate a schedule.
        
        Returns:
            List of actions (dicts with 'type' and relevant fields)
        """
        raise NotImplementedError("Subclasses must implement schedule()")

    def _tensor_size(self, tensor_name: str) -> int:
        """Get size of a tensor by name."""
        return self.problem.tensors[tensor_name].size

    def _total_tensor_size(self, tensors: Set[str]) -> int:
        """Get total size of a set of tensors."""
        return sum(self._tensor_size(t) for t in tensors)


class NaiveScheduler(BaseScheduler):
    """
    Naive baseline scheduler using topological sort.
    
    Strategy:
    1. Topologically sort operations
    2. For each operation:
       - Load all missing inputs
       - Compute
       - Store all outputs immediately
    
    This will pass validation but score poorly due to:
    - Immediate eviction of outputs (no reuse)
    - No smart eviction policy
    - No consideration of future use
    """
    
    def schedule(self) -> List[Dict]:
        """Generate naive schedule."""
        actions = []
        self.memory.reset()
        
        # Get topologically sorted operations
        op_order = self.graph.topological_sort()
        
        # Track which operations have been computed
        computed_ops = set()
        
        # Process each operation (load inputs on-demand, not upfront)
        for op_idx, op_name in enumerate(op_order):
            op = self.problem.operations[op_name]

            required_inputs = set(op.inputs)
            required_input_size = self._total_tensor_size(required_inputs)
            if required_input_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} inputs ({required_input_size} bytes) exceed scratchpad "
                    f"capacity ({self.problem.scratchpad_capacity} bytes)"
                )
            future_ops = set(op_order[op_idx + 1:])
            
            # Load missing inputs
            for input_tensor in op.inputs:
                if not self.memory.is_loaded(input_tensor):
                    # Need to make space if necessary (use smart eviction)
                    while not self.memory.can_load(input_tensor):
                        candidates = self.memory.get_eviction_candidates() - required_inputs
                        if not candidates:
                            raise ValueError(
                                f"Cannot load inputs for {op_name}: insufficient scratchpad capacity"
                            )
                        
                        # Find tensors that won't be used by future operations
                        safe_to_evict = set()
                        for tensor in candidates:
                            consumers = self.graph.tensor_consumers.get(tensor, [])
                            if not any(c in future_ops for c in consumers):
                                safe_to_evict.add(tensor)
                        
                        if safe_to_evict:
                            evict_tensor = next(iter(safe_to_evict))
                        else:
                            # Evict first available (naive)
                            evict_tensor = next(iter(candidates))
                        
                        actions.append(emit_store(evict_tensor))
                        self.memory.store(evict_tensor)
                    
                    actions.append(emit_load(input_tensor))
                    self.memory.load(input_tensor)
            
            # Calculate total space needed for outputs
            output_set = set(op.outputs)
            total_output_size = self._total_tensor_size(output_set)
            if required_input_size + total_output_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} requires inputs+outputs ({required_input_size + total_output_size} bytes) "
                    f"which exceeds scratchpad capacity ({self.problem.scratchpad_capacity} bytes)"
                )
            
            # Make space for outputs by evicting tensors we no longer need
            # Priority: evict inputs that won't be used by future operations
            while self.memory.get_available_space() < total_output_size:
                candidates = self.memory.get_eviction_candidates() - required_inputs
                if not candidates:
                    raise ValueError(
                        f"Cannot make space for outputs of {op_name} without evicting required inputs"
                    )
                
                # Find inputs that are only used by this operation (can be safely evicted)
                # BUT: exclude current operation's inputs since we still need them for compute!
                safe_to_evict = set()
                for tensor in candidates:
                    # Check if this tensor is consumed by future operations
                    consumers = self.graph.tensor_consumers.get(tensor, [])
                    if not any(c in future_ops for c in consumers):
                        safe_to_evict.add(tensor)
                
                if safe_to_evict:
                    evict_tensor = next(iter(safe_to_evict))
                else:
                    # No safe candidates, evict any non-required tensor
                    evict_tensor = next(iter(candidates))
                
                actions.append(emit_store(evict_tensor))
                self.memory.store(evict_tensor)
            
            # Compute
            actions.append(emit_compute(op_name))
            computed_ops.add(op_name)
            
            # Add outputs to memory (they're now resident after compute)
            for output_tensor in output_set:
                tensor_size = self._tensor_size(output_tensor)
                self.memory.resident_tensors.add(output_tensor)
                self.memory.current_usage += tensor_size

            # Naive baseline: store outputs immediately
            seen = set()
            for output_tensor in op.outputs:
                if output_tensor in seen:
                    continue
                actions.append(emit_store(output_tensor))
                self.memory.store(output_tensor)
                seen.add(output_tensor)
        
        return actions


class OptimizedScheduler(BaseScheduler):
    """
    Optimized scheduler with smart eviction and tensor reuse.

    Improvements over naive:
    1. List scheduling with reuse-aware op reordering
    2. Belady-style eviction with lookahead next-use tracking
    3. Delayed stores (keep outputs if used soon)
    4. Lightweight input prefetching
    5. Optional two-pass ordering + beam search
    6. Optional FREE actions for dead tensors (when supported)
    """

    def __init__(
        self,
        problem: Problem,
        store_final_outputs: bool = True,
        use_two_pass: bool = False,
        beam_width: int = 1,
        beam_candidates: int = 3,
        prefetch_depth: int = 2,
        rollout_depth: int = 0,
        rollout_width: int = 2,
    ):
        super().__init__(problem)
        self.store_final_outputs = store_final_outputs
        self.use_two_pass = use_two_pass or beam_width > 1
        self.beam_width = max(1, beam_width)
        self.beam_candidates = max(self.beam_width, beam_candidates)
        self.prefetch_depth = max(0, prefetch_depth)
        self.rollout_depth = max(0, rollout_depth)
        self.rollout_width = max(1, rollout_width)

    def schedule(self) -> List[Dict]:
        """Generate optimized schedule."""
        self.memory.reset()
        if self.use_two_pass:
            op_order = self._build_op_order()
            return self._schedule_fixed_order(op_order)
        return self._schedule_dynamic_order()

    def _schedule_dynamic_order(self) -> List[Dict]:
        """Single-pass list scheduling with memory-aware choices."""
        actions: List[Dict] = []

        # Reference order for lookahead metrics
        base_order = self.graph.topological_sort()
        base_op_index = {op_name: idx for idx, op_name in enumerate(base_order)}

        # Precompute helpers
        op_successors = self._build_op_successors()
        criticality = self._compute_criticality(base_order, op_successors)
        final_outputs = self._identify_final_outputs()
        next_use_tracker = NextUseTracker(self.graph, base_op_index)
        input_tensors = self.graph.get_input_tensors()
        in_dram = {tensor: False for tensor in self.problem.tensors}
        for tensor in input_tensors:
            in_dram[tensor] = True
        consumer_counts = {
            tensor: len(consumers) for tensor, consumers in self.graph.tensor_consumers.items()
        }
        consumed_counts = {tensor: 0 for tensor in consumer_counts}

        # Initialize list scheduling state
        in_degree = {
            op_name: len(self.graph.op_dependencies.get(op_name, set()))
            for op_name in self.problem.operations
        }
        ready_ops = [op_name for op_name, deg in in_degree.items() if deg == 0]
        computed_ops: Set[str] = set()

        while ready_ops:
            score_fn = lambda name, memory=None, consumed=consumed_counts: self._score_ready_op_memory(
                name,
                criticality,
                base_op_index,
                consumer_counts,
                consumed,
                memory=memory,
            )
            if self.rollout_depth > 0 and len(ready_ops) > 1:
                op_name = self._choose_next_op_with_rollout(
                    ready_ops,
                    score_fn,
                    in_degree,
                    computed_ops,
                    next_use_tracker,
                    consumed_counts,
                    op_successors,
                    final_outputs,
                )
            else:
                op_name = self._choose_next_op(ready_ops, score_fn)
            ready_ops.remove(op_name)
            op = self.problem.operations[op_name]

            required_inputs = set(op.inputs)
            required_input_size = self._total_tensor_size(required_inputs)
            if required_input_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} inputs ({required_input_size} bytes) exceed scratchpad "
                    f"capacity ({self.problem.scratchpad_capacity} bytes)"
                )

            # Load missing inputs with smart eviction
            for input_tensor in op.inputs:
                if not self.memory.is_loaded(input_tensor):
                    while not self.memory.can_load(input_tensor):
                        evict_tensor = self._choose_eviction_victim(
                            required_inputs, next_use_tracker, final_outputs, in_dram
                        )
                        self._emit_eviction(
                            evict_tensor,
                            actions,
                            next_use_tracker=next_use_tracker,
                            final_outputs=final_outputs,
                            in_dram=in_dram,
                        )
                    if not in_dram.get(input_tensor, False):
                        raise ValueError(f"Tensor {input_tensor} not available in DRAM for load")
                    actions.append(emit_load(input_tensor))
                    self.memory.load(input_tensor)

            output_set = set(op.outputs)
            total_output_size = self._total_tensor_size(output_set)
            if required_input_size + total_output_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} requires inputs+outputs ({required_input_size + total_output_size} bytes) "
                    f"which exceeds scratchpad capacity ({self.problem.scratchpad_capacity} bytes)"
                )

            while self.memory.get_available_space() < total_output_size:
                evict_tensor = self._choose_eviction_victim(
                    required_inputs, next_use_tracker, final_outputs, in_dram
                )
                self._emit_eviction(
                    evict_tensor,
                    actions,
                    next_use_tracker=next_use_tracker,
                    final_outputs=final_outputs,
                    in_dram=in_dram,
                )

            # Compute
            actions.append(emit_compute(op_name))
            computed_ops.add(op_name)

            # Add outputs to memory
            for output_tensor in output_set:
                tensor_size = self._tensor_size(output_tensor)
                self.memory.resident_tensors.add(output_tensor)
                self.memory.current_usage += tensor_size
                in_dram[output_tensor] = False

            # Update next-use tracker (inputs consumed by this op)
            for input_tensor in required_inputs:
                next_use_tracker.mark_consumed(input_tensor, op_name)
                if input_tensor in consumed_counts:
                    consumed_counts[input_tensor] += 1

            # Release successors
            for succ in op_successors.get(op_name, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready_ops.append(succ)

            # Prefetch ready inputs when free space is available (no eviction)
            actions.extend(
                self._prefetch_ready_inputs(
                    ready_ops,
                    in_dram,
                    score_fn=score_fn,
                )
            )

        # Store any final outputs still in memory (optional)
        if self.store_final_outputs:
            for tensor_name in list(self.memory.resident_tensors):
                if tensor_name in final_outputs:
                    actions.append(emit_store(tensor_name))
                    self.memory.store(tensor_name)
                    in_dram[tensor_name] = True

        return actions

    def _schedule_fixed_order(self, op_order: List[str]) -> List[Dict]:
        """Schedule using a fixed op order with accurate next-use tracking."""
        actions: List[Dict] = []

        op_index = {op_name: idx for idx, op_name in enumerate(op_order)}
        next_use_tracker = NextUseTracker(self.graph, op_index)
        input_tensors = self.graph.get_input_tensors()
        in_dram = {tensor: False for tensor in self.problem.tensors}
        for tensor in input_tensors:
            in_dram[tensor] = True
        final_outputs = self._identify_final_outputs()
        computed_ops: Set[str] = set()

        for position, op_name in enumerate(op_order):
            op = self.problem.operations[op_name]
            required_inputs = set(op.inputs)
            required_input_size = self._total_tensor_size(required_inputs)
            if required_input_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} inputs ({required_input_size} bytes) exceed scratchpad "
                    f"capacity ({self.problem.scratchpad_capacity} bytes)"
                )

            # Load missing inputs with smart eviction
            for input_tensor in op.inputs:
                if not self.memory.is_loaded(input_tensor):
                    while not self.memory.can_load(input_tensor):
                        evict_tensor = self._choose_eviction_victim(
                            required_inputs, next_use_tracker, final_outputs, in_dram
                        )
                        self._emit_eviction(
                            evict_tensor,
                            actions,
                            next_use_tracker=next_use_tracker,
                            final_outputs=final_outputs,
                            in_dram=in_dram,
                        )
                    if not in_dram.get(input_tensor, False):
                        raise ValueError(f"Tensor {input_tensor} not available in DRAM for load")
                    actions.append(emit_load(input_tensor))
                    self.memory.load(input_tensor)

            output_set = set(op.outputs)
            total_output_size = self._total_tensor_size(output_set)
            if required_input_size + total_output_size > self.problem.scratchpad_capacity:
                raise ValueError(
                    f"Op {op_name} requires inputs+outputs ({required_input_size + total_output_size} bytes) "
                    f"which exceeds scratchpad capacity ({self.problem.scratchpad_capacity} bytes)"
                )

            while self.memory.get_available_space() < total_output_size:
                evict_tensor = self._choose_eviction_victim(
                    required_inputs, next_use_tracker, final_outputs, in_dram
                )
                self._emit_eviction(
                    evict_tensor,
                    actions,
                    next_use_tracker=next_use_tracker,
                    final_outputs=final_outputs,
                    in_dram=in_dram,
                )

            # Compute
            actions.append(emit_compute(op_name))
            computed_ops.add(op_name)

            # Add outputs to memory
            for output_tensor in output_set:
                tensor_size = self._tensor_size(output_tensor)
                self.memory.resident_tensors.add(output_tensor)
                self.memory.current_usage += tensor_size
                in_dram[output_tensor] = False

            # Update next-use tracker (inputs consumed by this op)
            for input_tensor in required_inputs:
                next_use_tracker.mark_consumed(input_tensor, op_name)

            # Opportunistically free dead inputs (only with accurate next-use)
            for input_tensor in required_inputs:
                if input_tensor in output_set:
                    continue
                if next_use_tracker.remaining_uses(input_tensor) == 0:
                    if input_tensor in self.memory.resident_tensors:
                        self._emit_eviction(
                            input_tensor,
                            actions,
                            next_use_tracker=next_use_tracker,
                            final_outputs=final_outputs,
                            in_dram=in_dram,
                        )

            # Prefetch lookahead inputs when free space is available (no eviction)
            actions.extend(
                self._prefetch_lookahead_inputs(
                    op_order,
                    position,
                    in_dram,
                )
            )

        # Store any final outputs still in memory (optional)
        if self.store_final_outputs:
            for tensor_name in list(self.memory.resident_tensors):
                if tensor_name in final_outputs:
                    actions.append(emit_store(tensor_name))
                    self.memory.store(tensor_name)
                    in_dram[tensor_name] = True

        return actions

    def _build_op_order(self) -> List[str]:
        """Build an op order using greedy or beam search list scheduling."""
        base_order = self.graph.topological_sort()
        base_op_index = {op_name: idx for idx, op_name in enumerate(base_order)}
        op_successors = self._build_op_successors()
        criticality = self._compute_criticality(base_order, op_successors)

        consumer_counts = {
            tensor: len(consumers) for tensor, consumers in self.graph.tensor_consumers.items()
        }
        op_input_sizes = {}
        op_output_sizes = {}
        op_fanout = {}
        op_release_bytes = {}

        for op_name, op in self.problem.operations.items():
            input_set = set(op.inputs)
            output_set = set(op.outputs)
            op_input_sizes[op_name] = self._total_tensor_size(input_set)
            op_output_sizes[op_name] = self._total_tensor_size(output_set)
            op_fanout[op_name] = sum(consumer_counts.get(t, 0) for t in output_set)
            op_release_bytes[op_name] = sum(
                self._tensor_size(t)
                for t in input_set
                if consumer_counts.get(t, 0) == 1
            )

        if self.beam_width <= 1:
            return self._build_op_order_greedy(
                op_successors,
                criticality,
                base_op_index,
                op_input_sizes,
                op_output_sizes,
                op_fanout,
                op_release_bytes,
            )

        return self._build_op_order_beam(
            op_successors,
            criticality,
            base_op_index,
            op_input_sizes,
            op_output_sizes,
            op_fanout,
            op_release_bytes,
        )

    def _build_op_order_greedy(
        self,
        op_successors: Dict[str, Set[str]],
        criticality: Dict[str, int],
        base_op_index: Dict[str, int],
        op_input_sizes: Dict[str, int],
        op_output_sizes: Dict[str, int],
        op_fanout: Dict[str, int],
        op_release_bytes: Dict[str, int],
    ) -> List[str]:
        """Greedy list scheduling order (memory-agnostic)."""
        in_degree = {
            op_name: len(self.graph.op_dependencies.get(op_name, set()))
            for op_name in self.problem.operations
        }
        ready_ops = [op_name for op_name, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        last_produced: Dict[str, int] = {}

        position = 0
        while ready_ops:
            op_name = self._choose_next_op(
                ready_ops,
                lambda name: self._score_ready_op_ordering(
                    name,
                    position,
                    last_produced,
                    criticality,
                    base_op_index,
                    op_input_sizes,
                    op_output_sizes,
                    op_fanout,
                    op_release_bytes,
                ),
            )
            ready_ops.remove(op_name)
            order.append(op_name)

            for output_tensor in self.problem.operations[op_name].outputs:
                last_produced[output_tensor] = position

            for succ in op_successors.get(op_name, set()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready_ops.append(succ)

            position += 1

        return order

    def _build_op_order_beam(
        self,
        op_successors: Dict[str, Set[str]],
        criticality: Dict[str, int],
        base_op_index: Dict[str, int],
        op_input_sizes: Dict[str, int],
        op_output_sizes: Dict[str, int],
        op_fanout: Dict[str, int],
        op_release_bytes: Dict[str, int],
    ) -> List[str]:
        """Beam search over op orderings (memory-agnostic)."""
        in_degree = {
            op_name: len(self.graph.op_dependencies.get(op_name, set()))
            for op_name in self.problem.operations
        }
        initial_ready = [op_name for op_name, deg in in_degree.items() if deg == 0]
        total_ops = len(self.problem.operations)

        zero_score = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        beam = [
            {
                'order': [],
                'ready': initial_ready,
                'in_degree': in_degree,
                'last_produced': {},
                'score': zero_score,
            }
        ]

        for position in range(total_ops):
            new_states = []
            for state in beam:
                ready_ops = state['ready']
                if not ready_ops:
                    continue

                ranked = sorted(
                    ready_ops,
                    key=lambda name: self._score_ready_op_ordering(
                        name,
                        position,
                        state['last_produced'],
                        criticality,
                        base_op_index,
                        op_input_sizes,
                        op_output_sizes,
                        op_fanout,
                        op_release_bytes,
                    ),
                    reverse=True,
                )
                for op_name in ranked[: self.beam_candidates]:
                    new_in_degree = dict(state['in_degree'])
                    new_ready = list(ready_ops)
                    new_ready.remove(op_name)
                    new_order = list(state['order'])
                    new_order.append(op_name)
                    new_last_produced = dict(state['last_produced'])
                    for output_tensor in self.problem.operations[op_name].outputs:
                        new_last_produced[output_tensor] = position

                    for succ in op_successors.get(op_name, set()):
                        new_in_degree[succ] -= 1
                        if new_in_degree[succ] == 0:
                            new_ready.append(succ)

                    op_score = self._score_ready_op_ordering(
                        op_name,
                        position,
                        state['last_produced'],
                        criticality,
                        base_op_index,
                        op_input_sizes,
                        op_output_sizes,
                        op_fanout,
                        op_release_bytes,
                    )
                    new_score = self._add_score(state['score'], op_score)

                    new_states.append(
                        {
                            'order': new_order,
                            'ready': new_ready,
                            'in_degree': new_in_degree,
                            'last_produced': new_last_produced,
                            'score': new_score,
                        }
                    )

            if not new_states:
                break

            new_states.sort(key=lambda s: s['score'], reverse=True)
            beam = new_states[: self.beam_width]

        return beam[0]['order'] if beam else []

    def _score_ready_op_memory(
        self,
        op_name: str,
        criticality: Dict[str, int],
        base_op_index: Dict[str, int],
        consumer_counts: Dict[str, int],
        consumed_counts: Dict[str, int],
        memory: MemoryTracker = None,
    ) -> tuple:
        """Score a ready op for list scheduling with memory pressure awareness."""
        if memory is None:
            memory = self.memory
        op = self.problem.operations[op_name]
        input_set = set(op.inputs)
        output_set = set(op.outputs)
        input_total = self._total_tensor_size(input_set)
        resident_bytes = sum(self._tensor_size(t) for t in input_set if memory.is_loaded(t))
        missing_bytes = input_total - resident_bytes
        output_bytes = self._total_tensor_size(output_set)
        consumer_count = sum(consumer_counts.get(tensor, 0) for tensor in output_set)

        release_bytes = 0
        for tensor in input_set:
            total_consumers = consumer_counts.get(tensor, 0)
            if total_consumers == 0:
                continue
            if consumed_counts.get(tensor, 0) + 1 == total_consumers:
                release_bytes += self._tensor_size(tensor)

        pressure_delta = output_bytes - release_bytes

        return (
            -missing_bytes,          # prefer fewer bytes to load
            resident_bytes,          # then prefer more inputs already resident
            -pressure_delta,         # prefer ops that reduce pressure
            -output_bytes,           # prefer smaller outputs
            consumer_count,          # prefer outputs with more reuse
            criticality[op_name],    # prefer critical ops
            -base_op_index[op_name], # deterministic tie-breaker
        )

    def _score_ready_op_ordering(
        self,
        op_name: str,
        position: int,
        last_produced: Dict[str, int],
        criticality: Dict[str, int],
        base_op_index: Dict[str, int],
        op_input_sizes: Dict[str, int],
        op_output_sizes: Dict[str, int],
        op_fanout: Dict[str, int],
        op_release_bytes: Dict[str, int],
    ) -> tuple:
        """Score a ready op for ordering (memory-agnostic)."""
        op = self.problem.operations[op_name]
        reuse_score = 0.0
        for input_tensor in op.inputs:
            if input_tensor in last_produced:
                distance = position - last_produced[input_tensor]
                if distance >= 0:
                    reuse_score += 1.0 / (1.0 + distance)

        pressure_delta = op_output_sizes[op_name] - op_release_bytes[op_name]

        return (
            reuse_score,
            -pressure_delta,
            -op_output_sizes[op_name],
            op_fanout[op_name],
            criticality[op_name],
            -op_input_sizes[op_name],
            -base_op_index[op_name],
        )

    def _choose_next_op(self, ready_ops: List[str], score_fn) -> str:
        """Choose the next op from the ready set using a scoring function."""
        best_op = None
        best_score = None
        for op_name in ready_ops:
            score = score_fn(op_name)
            if best_score is None or score > best_score:
                best_op = op_name
                best_score = score
        return best_op

    def _choose_next_op_with_rollout(
        self,
        ready_ops: List[str],
        score_fn,
        in_degree: Dict[str, int],
        computed_ops: Set[str],
        next_use_tracker: NextUseTracker,
        consumed_counts: Dict[str, int],
        op_successors: Dict[str, Set[str]],
        final_outputs: Set[str],
    ) -> str:
        """Choose the next op using limited rollout simulation."""
        ranked = sorted(ready_ops, key=score_fn, reverse=True)
        candidates = ranked[: self.rollout_width]

        best_op = None
        best_cost = None
        best_score = None

        for op_name in candidates:
            cost = self._simulate_rollout_cost(
                op_name,
                in_degree,
                ready_ops,
                computed_ops,
                next_use_tracker,
                consumed_counts,
                op_successors,
                final_outputs,
                score_fn,
            )
            score = score_fn(op_name)
            if best_cost is None or cost < best_cost or (cost == best_cost and score > best_score):
                best_op = op_name
                best_cost = cost
                best_score = score

        return best_op if best_op is not None else ranked[0]

    def _simulate_rollout_cost(
        self,
        first_op: str,
        in_degree: Dict[str, int],
        ready_ops: List[str],
        computed_ops: Set[str],
        next_use_tracker: NextUseTracker,
        consumed_counts: Dict[str, int],
        op_successors: Dict[str, Set[str]],
        final_outputs: Set[str],
        score_fn,
    ) -> int:
        """Simulate a short rollout and return estimated cost."""
        sim_memory = self._clone_memory(self.memory)
        sim_in_degree = dict(in_degree)
        sim_ready = list(ready_ops)
        sim_computed = set(computed_ops)
        sim_tracker = next_use_tracker.clone()
        sim_consumed = dict(consumed_counts)
        sim_in_dram = {tensor: False for tensor in self.problem.tensors}
        for tensor in self.graph.get_input_tensors():
            sim_in_dram[tensor] = True

        total_cost = 0

        for step in range(self.rollout_depth):
            if not sim_ready:
                break
            if step == 0:
                op_name = first_op
            else:
                op_name = self._choose_next_op(
                    sim_ready,
                    lambda name: score_fn(name, memory=sim_memory, consumed=sim_consumed),
                )

            sim_ready.remove(op_name)
            op = self.problem.operations[op_name]
            required_inputs = set(op.inputs)

            # Load inputs (simulate evictions)
            for input_tensor in op.inputs:
                if not sim_memory.is_loaded(input_tensor):
                    while not sim_memory.can_load(input_tensor):
                        evict_tensor = self._choose_eviction_victim(
                            required_inputs, sim_tracker, final_outputs, sim_in_dram, memory=sim_memory
                        )
                        if sim_in_dram.get(evict_tensor, False):
                            sim_memory.store(evict_tensor)
                        else:
                            sim_memory.store(evict_tensor)
                            total_cost += self.problem.store_cost
                            sim_in_dram[evict_tensor] = True
                    if not sim_in_dram.get(input_tensor, False):
                        return float('inf')
                    sim_memory.load(input_tensor)
                    total_cost += self.problem.load_cost

            output_set = set(op.outputs)
            total_output_size = self._total_tensor_size(output_set)
            while sim_memory.get_available_space() < total_output_size:
                evict_tensor = self._choose_eviction_victim(
                    required_inputs, sim_tracker, final_outputs, sim_in_dram, memory=sim_memory
                )
                if sim_in_dram.get(evict_tensor, False):
                    sim_memory.store(evict_tensor)
                else:
                    sim_memory.store(evict_tensor)
                    total_cost += self.problem.store_cost
                    sim_in_dram[evict_tensor] = True

            # Compute
            total_cost += op.compute_cost
            sim_computed.add(op_name)

            # Add outputs
            for output_tensor in output_set:
                sim_memory.resident_tensors.add(output_tensor)
                sim_memory.current_usage += self._tensor_size(output_tensor)
                sim_in_dram[output_tensor] = False

            # Update trackers
            for input_tensor in required_inputs:
                sim_tracker.mark_consumed(input_tensor, op_name)
                if input_tensor in sim_consumed:
                    sim_consumed[input_tensor] += 1

            # Release successors
            for succ in op_successors.get(op_name, set()):
                sim_in_degree[succ] -= 1
                if sim_in_degree[succ] == 0:
                    sim_ready.append(succ)

        return total_cost

    def _clone_memory(self, memory: MemoryTracker) -> MemoryTracker:
        """Clone memory state for simulation."""
        clone = MemoryTracker(self.problem)
        clone.resident_tensors = set(memory.resident_tensors)
        clone.current_usage = memory.current_usage
        return clone

    def _add_score(self, a: tuple, b: tuple) -> tuple:
        return tuple(x + y for x, y in zip(a, b))

    def _eviction_score(
        self,
        tensor_name: str,
        next_use_tracker: NextUseTracker,
        final_outputs: Set[str],
        in_dram: Dict[str, bool],
    ) -> tuple:
        """Score a tensor for eviction (higher is better)."""
        next_use = next_use_tracker.next_use_index(tensor_name)
        size = self._tensor_size(tensor_name)
        remaining = next_use_tracker.remaining_uses(tensor_name)
        is_final = 1 if tensor_name in final_outputs else 0
        drambacked = 1 if in_dram.get(tensor_name, False) else 0
        return (drambacked, next_use, size, -remaining, is_final)

    def _choose_eviction_victim(
        self,
        required_inputs: Set[str],
        next_use_tracker: NextUseTracker,
        final_outputs: Set[str],
        in_dram: Dict[str, bool],
        memory: MemoryTracker = None,
    ) -> str:
        """Choose which tensor to evict using lookahead + size + reuse."""
        if memory is None:
            memory = self.memory
        candidates = memory.get_eviction_candidates() - required_inputs
        if not candidates:
            raise ValueError("Cannot evict without violating required inputs")
        return max(
            candidates,
            key=lambda t: self._eviction_score(t, next_use_tracker, final_outputs, in_dram)
        )

    def _emit_eviction(
        self,
        tensor_name: str,
        actions: List[Dict],
        next_use_tracker: NextUseTracker,
        final_outputs: Set[str],
        in_dram: Dict[str, bool],
        memory: MemoryTracker = None,
    ) -> None:
        """Evict a tensor; emit store only if not DRAM-backed."""
        if memory is None:
            memory = self.memory
        if in_dram.get(tensor_name, False):
            memory.store(tensor_name)
            return

        actions.append(emit_store(tensor_name))
        memory.store(tensor_name)
        in_dram[tensor_name] = True

    def _prefetch_ready_inputs(
        self,
        ready_ops: List[str],
        in_dram: Dict[str, bool],
        score_fn,
        limit_ops: int = 2,
    ) -> List[Dict]:
        """Prefetch inputs for top ready ops when space is available."""
        actions: List[Dict] = []
        if not ready_ops:
            return actions

        sorted_ops = sorted(ready_ops, key=score_fn, reverse=True)

        for op_name in sorted_ops[:limit_ops]:
            op = self.problem.operations[op_name]
            for input_tensor in op.inputs:
                if self.memory.is_loaded(input_tensor):
                    continue
                if not in_dram.get(input_tensor, False):
                    continue
                if self.memory.can_load(input_tensor):
                    actions.append(emit_load(input_tensor))
                    self.memory.load(input_tensor)

        return actions

    def _prefetch_lookahead_inputs(
        self,
        op_order: List[str],
        position: int,
        in_dram: Dict[str, bool],
    ) -> List[Dict]:
        """Prefetch inputs for upcoming ops in the fixed order."""
        actions: List[Dict] = []
        if self.prefetch_depth <= 0:
            return actions

        for lookahead in range(1, self.prefetch_depth + 1):
            idx = position + lookahead
            if idx >= len(op_order):
                break
            op = self.problem.operations[op_order[idx]]
            for input_tensor in op.inputs:
                if self.memory.is_loaded(input_tensor):
                    continue
                if not in_dram.get(input_tensor, False):
                    continue
                if self.memory.can_load(input_tensor):
                    actions.append(emit_load(input_tensor))
                    self.memory.load(input_tensor)

        return actions

    def _build_op_successors(self) -> Dict[str, Set[str]]:
        """Build op-to-op successor mapping."""
        successors: Dict[str, Set[str]] = {name: set() for name in self.problem.operations}
        for op_name, op in self.problem.operations.items():
            for output_tensor in op.outputs:
                for consumer in self.graph.tensor_consumers.get(output_tensor, []):
                    successors[op_name].add(consumer)
        return successors

    def _compute_criticality(self, op_order: List[str], successors: Dict[str, Set[str]]) -> Dict[str, int]:
        """Compute critical path length (compute-cost based) for each op."""
        criticality: Dict[str, int] = {}
        for op_name in reversed(op_order):
            succs = successors.get(op_name, set())
            if not succs:
                criticality[op_name] = self.problem.operations[op_name].compute_cost
            else:
                criticality[op_name] = (
                    self.problem.operations[op_name].compute_cost +
                    max(criticality[s] for s in succs)
                )
        return criticality

    def _identify_final_outputs(self) -> Set[str]:
        """Identify tensors that are final outputs (never consumed)."""
        if self.problem.outputs:
            return set(self.problem.outputs)

        final_outputs = set()
        for op in self.problem.operations.values():
            for output_tensor in op.outputs:
                if output_tensor not in self.graph.tensor_consumers:
                    final_outputs.add(output_tensor)
        return final_outputs
