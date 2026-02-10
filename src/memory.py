"""
SRAM memory tracker for managing scratchpad memory state.
Tracks which tensors are resident and enforces capacity constraints.
"""

from typing import Set, Dict
from src.parser import Problem


class MemoryTracker:
    """Tracks SRAM state and enforces capacity constraints."""
    
    def __init__(self, problem: Problem):
        """
        Initialize memory tracker.
        
        Args:
            problem: Problem specification with capacity and tensor sizes
        """
        self.capacity = problem.scratchpad_capacity
        self.tensor_sizes = {name: tensor.size for name, tensor in problem.tensors.items()}
        self.resident_tensors: Set[str] = set()
        self.current_usage = 0
    
    def can_load(self, tensor_name: str) -> bool:
        """
        Check if a tensor can be loaded without exceeding capacity.
        
        Args:
            tensor_name: Name of tensor to check
            
        Returns:
            True if tensor can be loaded
        """
        if tensor_name in self.resident_tensors:
            return True  # Already loaded
        
        tensor_size = self.tensor_sizes[tensor_name]
        return self.current_usage + tensor_size <= self.capacity
    
    def load(self, tensor_name: str) -> None:
        """
        Load a tensor into SRAM.
        
        Args:
            tensor_name: Name of tensor to load
            
        Raises:
            ValueError: If tensor is already loaded or would exceed capacity
        """
        if tensor_name in self.resident_tensors:
            raise ValueError(f"Tensor {tensor_name} is already loaded")
        
        tensor_size = self.tensor_sizes[tensor_name]
        if self.current_usage + tensor_size > self.capacity:
            raise ValueError(f"Loading {tensor_name} would exceed capacity")
        
        self.resident_tensors.add(tensor_name)
        self.current_usage += tensor_size
    
    def store(self, tensor_name: str) -> None:
        """
        Store (evict) a tensor from SRAM.
        
        Args:
            tensor_name: Name of tensor to evict
            
        Raises:
            ValueError: If tensor is not currently loaded
        """
        if tensor_name not in self.resident_tensors:
            raise ValueError(f"Tensor {tensor_name} is not loaded")
        
        tensor_size = self.tensor_sizes[tensor_name]
        self.resident_tensors.remove(tensor_name)
        self.current_usage -= tensor_size
    
    def is_loaded(self, tensor_name: str) -> bool:
        """
        Check if a tensor is currently in SRAM.
        
        Args:
            tensor_name: Name of tensor to check
            
        Returns:
            True if tensor is loaded
        """
        return tensor_name in self.resident_tensors
    
    def get_available_space(self) -> int:
        """
        Get available SRAM space.
        
        Returns:
            Available space in bytes
        """
        return self.capacity - self.current_usage
    
    def get_eviction_candidates(self) -> Set[str]:
        """
        Get all tensors that could be evicted.
        
        Returns:
            Set of tensor names currently in SRAM
        """
        return self.resident_tensors.copy()
    
    def reset(self) -> None:
        """Reset memory state to empty."""
        self.resident_tensors.clear()
        self.current_usage = 0
