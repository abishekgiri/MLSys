"""
DAG (Directed Acyclic Graph) representation for computational graphs.
Handles topological sorting and dependency tracking.
"""

from typing import Dict, List, Set
from collections import defaultdict, deque
from src.parser import Problem, Operation


class ComputationGraph:
    """Represents the computational DAG with dependency tracking."""
    
    def __init__(self, problem: Problem):
        """
        Initialize the computation graph from a problem specification.
        
        Args:
            problem: Problem object containing operations and dependencies
        """
        self.problem = problem
        self.operations = problem.operations
        self.tensors = problem.tensors
        
        # Build dependency graph
        self.op_dependencies = self._build_op_dependencies()
        self.tensor_consumers = self._build_tensor_consumers()
        self.tensor_producers = self._build_tensor_producers()
    
    def _build_op_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build operation-to-operation dependencies.
        
        Returns:
            Dict mapping each operation to the set of operations it depends on
        """
        dependencies = defaultdict(set)
        
        for op_name, op in self.operations.items():
            # Find which operations produce this op's inputs
            for input_tensor in op.inputs:
                # Find the producer of this tensor
                for producer_name, producer_op in self.operations.items():
                    if input_tensor in producer_op.outputs:
                        dependencies[op_name].add(producer_name)
        
        return dict(dependencies)
    
    def _build_tensor_consumers(self) -> Dict[str, List[str]]:
        """
        Build tensor-to-operation consumer mapping.
        
        Returns:
            Dict mapping each tensor to list of operations that consume it
        """
        consumers = defaultdict(list)
        
        for op_name, op in self.operations.items():
            for input_tensor in op.inputs:
                consumers[input_tensor].append(op_name)
        
        return dict(consumers)
    
    def _build_tensor_producers(self) -> Dict[str, str]:
        """
        Build tensor-to-operation producer mapping.
        
        Returns:
            Dict mapping each tensor to the operation that produces it
        """
        producers = {}
        
        for op_name, op in self.operations.items():
            for output_tensor in op.outputs:
                if output_tensor in producers:
                    raise ValueError(f"Tensor {output_tensor} has multiple producers")
                producers[output_tensor] = op_name
        
        return producers
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort on operations using Kahn's algorithm.
        
        Returns:
            List of operation names in topologically sorted order
            
        Raises:
            ValueError: If the graph contains a cycle
        """
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for op_name in self.operations:
            in_degree[op_name] = len(self.op_dependencies.get(op_name, set()))
        
        # Initialize queue with operations that have no dependencies
        queue = deque([op for op in self.operations if in_degree[op] == 0])
        sorted_ops = []
        
        while queue:
            current = queue.popleft()
            sorted_ops.append(current)
            
            # Find operations that depend on current
            for op_name, deps in self.op_dependencies.items():
                if current in deps:
                    in_degree[op_name] -= 1
                    if in_degree[op_name] == 0:
                        queue.append(op_name)
        
        # Check for cycles
        if len(sorted_ops) != len(self.operations):
            raise ValueError("Graph contains a cycle")
        
        return sorted_ops
    
    def get_next_uses(self, tensor_name: str, current_position: int, 
                      op_order: List[str]) -> int:
        """
        Calculate the next use distance for a tensor.
        
        Args:
            tensor_name: Name of the tensor
            current_position: Current position in the operation order
            op_order: Ordered list of operations
            
        Returns:
            Position of next use, or infinity if never used again
        """
        consumers = self.tensor_consumers.get(tensor_name, [])
        
        # Find the next consumer after current position
        next_use = float('inf')
        for i in range(current_position + 1, len(op_order)):
            if op_order[i] in consumers:
                next_use = i
                break
        
        return next_use
    
    def get_input_tensors(self) -> Set[str]:
        """
        Get all tensors that are inputs (not produced by any operation).
        
        Returns:
            Set of input tensor names
        """
        all_tensors = set(self.tensors.keys())
        produced_tensors = set(self.tensor_producers.keys())
        return all_tensors - produced_tensors
