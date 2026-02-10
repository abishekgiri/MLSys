#!/usr/bin/env python3
"""
Quick test script to verify the scheduler works correctly.
"""

import sys
sys.path.insert(0, '/Users/abishekkumargiri/Desktop/sellable/MLSys')

from src import (
    parse_problem,
    validate_problem,
    NaiveScheduler,
    OptimizedScheduler,
    validate_schedule,
    print_schedule_summary
)

def test_scheduler():
    """Test both schedulers on the example problem."""
    
    print("=" * 70)
    print("TESTING MLSYS SCHEDULER")
    print("=" * 70)
    print()
    
    # Load problem
    problem_path = '/Users/abishekkumargiri/Desktop/sellable/MLSys/problems/example_problem.json'
    print(f"Loading problem: {problem_path}")
    
    try:
        problem = parse_problem(problem_path)
        validate_problem(problem)
        print("✓ Problem loaded and validated\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test naive scheduler
    print("-" * 70)
    print("NAIVE SCHEDULER")
    print("-" * 70)
    try:
        naive = NaiveScheduler(problem)
        naive_schedule = naive.schedule()
        validate_schedule(naive_schedule, problem)
        print(f"✓ Generated valid schedule with {len(naive_schedule)} actions\n")
        print_schedule_summary(naive_schedule, problem, allow_free=False)
        print()
    except Exception as e:
        print(f"✗ Naive scheduler failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test optimized scheduler
    print("-" * 70)
    print("OPTIMIZED SCHEDULER")
    print("-" * 70)
    try:
        optimized = OptimizedScheduler(problem)
        opt_schedule = optimized.schedule()
        validate_schedule(opt_schedule, problem)
        print(f"✓ Generated valid schedule with {len(opt_schedule)} actions\n")
        print_schedule_summary(opt_schedule, problem, allow_free=False)
        print()
    except Exception as e:
        print(f"✗ Optimized scheduler failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare
    from src.cost_model import CostModel
    cost_model = CostModel(problem)
    
    naive_cost = cost_model.evaluate_schedule(naive_schedule)
    opt_cost = cost_model.evaluate_schedule(opt_schedule)
    improvement = ((naive_cost - opt_cost) / naive_cost) * 100
    
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Naive latency:     {naive_cost}")
    print(f"Optimized latency: {opt_cost}")
    print(f"Improvement:       {improvement:.1f}%")
    print("=" * 70)
    print()
    print("✓ ALL TESTS PASSED!")
    print()
    
    return True

if __name__ == '__main__':
    success = test_scheduler()
    sys.exit(0 if success else 1)
