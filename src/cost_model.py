"""
Cost model for calculating schedule latency.
Evaluates total execution time based on load, store, and compute costs.
"""

from typing import Dict, List
from src.parser import Problem


class CostModel:
    """Calculates total latency for a given schedule."""
    
    def __init__(self, problem: Problem, allow_free: bool = False):
        """
        Initialize cost model.
        
        Args:
            problem: Problem specification with cost parameters
            allow_free: Whether FREE actions are allowed (otherwise charged as store)
        """
        self.load_cost = problem.load_cost
        self.store_cost = problem.store_cost
        self.allow_free = allow_free
        self.compute_costs = {name: op.compute_cost for name, op in problem.operations.items()}
    
    def evaluate_schedule(self, schedule: List[Dict]) -> int:
        """
        Calculate total latency for a schedule.
        
        Args:
            schedule: List of action dictionaries with 'type' and relevant fields
            
        Returns:
            Total latency (sum of all costs)
        """
        total_latency = 0
        
        for action in schedule:
            action_type = action['type']
            
            if action_type == 'load':
                total_latency += self.load_cost
            elif action_type == 'store':
                total_latency += self.store_cost
            elif action_type == 'free':
                total_latency += 0 if self.allow_free else self.store_cost
            elif action_type == 'compute':
                op_name = action['op']
                total_latency += self.compute_costs[op_name]
            else:
                raise ValueError(f"Unknown action type: {action_type}")
        
        return total_latency
    
    def get_breakdown(self, schedule: List[Dict]) -> Dict[str, int]:
        """
        Get detailed cost breakdown.
        
        Args:
            schedule: List of action dictionaries
            
        Returns:
            Dictionary with 'load', 'store', 'compute', and 'total' costs
        """
        load_count = 0
        store_count = 0
        free_count = 0
        compute_cost = 0
        
        for action in schedule:
            action_type = action['type']
            
            if action_type == 'load':
                load_count += 1
            elif action_type == 'store':
                store_count += 1
            elif action_type == 'free':
                free_count += 1
            elif action_type == 'compute':
                op_name = action['op']
                compute_cost += self.compute_costs[op_name]
        
        free_cost = 0 if self.allow_free else free_count * self.store_cost
        return {
            'load': load_count * self.load_cost,
            'store': store_count * self.store_cost,
            'free': free_cost,
            'compute': compute_cost,
            'total': load_count * self.load_cost + store_count * self.store_cost + free_cost + compute_cost,
            'load_count': load_count,
            'store_count': store_count,
            'free_count': free_count
        }
