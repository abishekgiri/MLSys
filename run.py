#!/usr/bin/env python3
"""
Main entry point for MLSys 2026 Graph Scheduling Competition.

Usage:
    python run.py <problem_file> [--output <schedule_file>] [--scheduler <naive|optimized>]

Example:
    python run.py problems/example_problem.json --output schedules/example_schedule.json
"""

import argparse
import sys
import json
from typing import Dict
from pathlib import Path

from src import (
    parse_problem,
    validate_problem,
    CostModel,
    NaiveScheduler,
    OptimizedScheduler,
    save_schedule,
    validate_schedule,
    print_schedule_summary,
    analyze_schedule,
    parse_official_problem,
    OfficialScheduler,
    OfficialFusionScheduler,
    analyze_retention,
    write_official_schedule,
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MLSys 2026 Graph Scheduling Competition Scheduler'
    )
    parser.add_argument(
        'problem',
        type=str,
        nargs='?',
        help='Path to problem JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output schedule JSON file (default: schedules/<problem_name>_schedule.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='schedules',
        help='Output directory for schedules when using --all (default: schedules)'
    )
    parser.add_argument(
        '--scheduler', '-s',
        type=str,
        choices=['naive', 'optimized'],
        default='optimized',
        help='Scheduler to use (default: optimized)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run scheduler on all problems in the problems/ directory'
    )
    parser.add_argument(
        '--official',
        action='store_true',
        help='Force official problem format'
    )
    parser.add_argument(
        '--retain-budget',
        type=float,
        default=0.0,
        help='Fraction of fast SRAM reserved for cross-subgraph tensor retention (0.0-0.9).'
    )
    parser.add_argument(
        '--debug-fuse',
        action='store_true',
        help='Enable fuse scheduler debug logging (merge decisions).'
    )
    parser.add_argument(
        '--debug-fuse-file',
        type=str,
        default='schedules/debug_fuse_2026-17.txt',
        help='Path to write fuse debug logs.'
    )
    parser.add_argument(
        '--fuse-accept-slack',
        type=float,
        default=0.0,
        help='Allow fused latency up to (1 + slack) * separate latency.'
    )
    parser.add_argument(
        '--official-scheduler',
        type=str,
        choices=['baseline', 'fuse'],
        default='baseline',
        help='Official scheduler variant (default: baseline)'
    )
    parser.add_argument(
        '--connectivity',
        type=str,
        choices=['strict', 'loose', 'auto'],
        default='strict',
        help='Connectivity rule for fusion (default: strict)'
    )
    parser.add_argument(
        '--two-pass',
        action='store_true',
        help='Use two-pass scheduling (order first, then schedule)'
    )
    parser.add_argument(
        '--beam-width',
        type=int,
        default=1,
        help='Beam width for ordering (only used with optimized scheduler)'
    )
    parser.add_argument(
        '--beam-candidates',
        type=int,
        default=3,
        help='Number of candidates expanded per beam step'
    )
    parser.add_argument(
        '--prefetch-depth',
        type=int,
        default=2,
        help='Lookahead depth for input prefetch (0 disables)'
    )
    parser.add_argument(
        '--rollout-depth',
        type=int,
        default=0,
        help='Rollout depth for choosing next op (0 disables)'
    )
    parser.add_argument(
        '--rollout-width',
        type=int,
        default=2,
        help='Number of candidate ops evaluated per decision in rollout mode'
    )
    parser.add_argument(
        '--experimental',
        action='store_true',
        help='Allow experimental flags that may be invalid for official evaluator'
    )
    parser.add_argument(
        '--allow-free',
        action='store_true',
        help='EXPERIMENTAL: allow FREE actions for dead tensors (if evaluator supports)'
    )
    parser.add_argument(
        '--omit-final-stores',
        action='store_true',
        help='EXPERIMENTAL: do not store final outputs at the end'
    )
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validate the problem, do not generate schedule'
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Write a JSON report with per-problem results'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Tag to append to output filenames and default report name'
    )
    
    args = parser.parse_args()
    if (args.allow_free or args.omit_final_stores) and not args.experimental:
        print("✗ Error: experimental flags require --experimental", file=sys.stderr)
        return 1

    strict_mode = not (args.allow_free or args.omit_final_stores)
    
    if not args.all and not args.problem:
        print("✗ Error: missing problem path (or use --all)", file=sys.stderr)
        return 1

    if args.all and args.output is not None:
        print("✗ Error: --output cannot be used with --all (use --output-dir)", file=sys.stderr)
        return 1

    def detect_schema(problem_path: str) -> str:
        with open(problem_path, 'r') as f:
            data = json.load(f)
        if 'widths' in data and 'heights' in data and 'op_types' in data:
            return 'official'
        return 'toy'

    def run_single(problem_path: str) -> Dict:
        print(f"Loading problem: {problem_path}")
        schema = 'official' if args.official else detect_schema(problem_path)
        if schema == 'official':
            try:
                problem = parse_official_problem(problem_path)
                print("✓ Official problem loaded successfully")
            except Exception as e:
                print(f"✗ Error loading official problem: {e}", file=sys.stderr)
                return {'problem': problem_path, 'error': str(e)}
        else:
            try:
                problem = parse_problem(problem_path)
                validate_problem(problem)
                print("✓ Problem loaded and validated successfully")
            except Exception as e:
                print(f"✗ Error loading problem: {e}", file=sys.stderr)
                return {'problem': problem_path, 'error': str(e)}

        if args.validate_only:
            return {'problem': problem_path, 'validated': True}

        # Create scheduler
        if schema == 'official':
            retain_budget = max(0.0, min(args.retain_budget, 0.9))
            if args.official_scheduler == 'baseline':
                scheduler = OfficialScheduler(
                    problem,
                    retain_budget=retain_budget,
                    problem_path=problem_path,
                    debug_fuse=args.debug_fuse,
                    debug_fuse_file=args.debug_fuse_file,
                    fuse_accept_slack=args.fuse_accept_slack,
                    connectivity=args.connectivity,
                )
                print("Using official baseline scheduler")
            else:
                scheduler = OfficialFusionScheduler(
                    problem,
                    retain_budget=retain_budget,
                    problem_path=problem_path,
                    debug_fuse=args.debug_fuse,
                    debug_fuse_file=args.debug_fuse_file,
                    fuse_accept_slack=args.fuse_accept_slack,
                    connectivity=args.connectivity,
                )
                print("Using official fusion scheduler (boundary-aware + granularity search)")
        else:
            if args.scheduler == 'naive':
                scheduler = NaiveScheduler(problem)
                print("Using naive scheduler (baseline)")
            else:
                scheduler = OptimizedScheduler(
                    problem,
                    store_final_outputs=True if strict_mode else not args.omit_final_stores,
                    use_two_pass=args.two_pass,
                    beam_width=args.beam_width,
                    beam_candidates=args.beam_candidates,
                    prefetch_depth=args.prefetch_depth,
                    rollout_depth=args.rollout_depth,
                    rollout_width=args.rollout_width,
                )
                mode = "two-pass" if scheduler.use_two_pass else "single-pass"
                print(f"Using optimized scheduler ({mode}, reordering + lookahead eviction)")
                if args.allow_free and not scheduler.use_two_pass:
                    print("Note: --allow-free is only applied in two-pass mode; free actions ignored.")
                if args.allow_free:
                    print("WARNING: --allow-free may be invalid for official evaluator.")
                if args.omit_final_stores:
                    print("WARNING: --omit-final-stores may be invalid for official evaluator.")
                if strict_mode:
                    print("Strict mode enabled: only load/store/compute and final outputs stored.")

        # Generate schedule
        print("Generating schedule...")
        try:
            schedule = scheduler.schedule()
            if schema == 'official':
                print(f"✓ Generated schedule with {len(schedule.subgraphs)} subgraphs")
            else:
                print(f"✓ Generated schedule with {len(schedule)} actions")
        except Exception as e:
            print(f"✗ Error generating schedule: {e}", file=sys.stderr)
            return {'problem': problem_path, 'error': str(e)}

        # Validate schedule
        print("Validating schedule...")
        try:
            if schema == 'official':
                # Baseline: rely on schedule generation; no extra validator yet.
                print("✓ Schedule is valid")
            else:
                validate_schedule(
                    schedule,
                    problem,
                    allow_free=False if strict_mode else args.allow_free,
                    require_final_outputs=True if strict_mode else not args.omit_final_stores,
                )
                print("✓ Schedule is valid")
        except Exception as e:
            print(f"✗ Schedule validation failed: {e}", file=sys.stderr)
            return {'problem': problem_path, 'error': str(e)}

        # Determine output path
        problem_name = Path(problem_path).stem
        tag_suffix = f"_{args.tag}" if args.tag else ""
        if args.all:
            output_path = str(Path(args.output_dir) / f"{problem_name}{tag_suffix}_schedule.json")
        elif args.output is None:
            output_path = f"schedules/{problem_name}{tag_suffix}_schedule.json"
        else:
            output_path = args.output

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save schedule
        if schema == 'official':
            write_official_schedule(
                output_path,
                [sg.ops for sg in schedule.subgraphs],
                [sg.granularity for sg in schedule.subgraphs],
                [sg.tensors_to_retain for sg in schedule.subgraphs],
                [sg.traversal_order for sg in schedule.subgraphs],
                [sg.subgraph_latency for sg in schedule.subgraphs],
                connectivity_requested=getattr(schedule, 'connectivity_requested', 'strict'),
                connectivity_chosen=getattr(schedule, 'connectivity_chosen', 'strict'),
                latency_strict=getattr(schedule, 'latency_strict', None),
                latency_loose=getattr(schedule, 'latency_loose', None),
                loose_components_split=getattr(schedule, 'loose_components_split', 0),
            )
        else:
            save_schedule(schedule, output_path)
        print(f"✓ Schedule saved to: {output_path}")

        # Print summary
        print()
        if schema == 'official':
            total_latency = sum(sg.subgraph_latency for sg in schedule.subgraphs)
            retention_stats = analyze_retention(problem, schedule.subgraphs)
            print("SCHEDULE SUMMARY (OFFICIAL)")
            print("=" * 60)
            print(f"Subgraphs: {len(schedule.subgraphs)}")
            print(f"TOTAL LATENCY: {total_latency}")
            print("RETENTION SUMMARY")
            print("-" * 60)
            print(f"Retained bytes total: {retention_stats['retained_bytes_total']}")
            print(f"Retained bytes avg:   {retention_stats['retained_bytes_avg']:.2f}")
            print(f"Peak resident bytes:  {retention_stats['peak_resident_bytes']}")
            print(f"Load bytes saved:     {retention_stats['load_saved_bytes']}")
            print(f"Store bytes saved:    {retention_stats['store_saved_bytes']}")
            breakdown = None
            analysis = None
        else:
            print_schedule_summary(schedule, problem, allow_free=not strict_mode and args.allow_free)
            cost_model = CostModel(problem, allow_free=not strict_mode and args.allow_free)
            breakdown = cost_model.get_breakdown(schedule)
            analysis = analyze_schedule(schedule, problem, allow_free=not strict_mode and args.allow_free)

        if schema == 'official':
            return {
                'problem': problem_path,
                'schedule': output_path,
                'subgraphs': len(schedule.subgraphs),
                'latency': sum(sg.subgraph_latency for sg in schedule.subgraphs),
                'connectivity_requested': getattr(schedule, 'connectivity_requested', 'strict'),
                'connectivity_chosen': getattr(schedule, 'connectivity_chosen', 'strict'),
                'latency_strict': getattr(schedule, 'latency_strict', None),
                'latency_loose': getattr(schedule, 'latency_loose', None),
                'loose_components_split': getattr(schedule, 'loose_components_split', 0),
                'retained_bytes_total': retention_stats['retained_bytes_total'],
                'retained_bytes_avg': retention_stats['retained_bytes_avg'],
                'peak_resident_bytes': retention_stats['peak_resident_bytes'],
                'load_saved_bytes': retention_stats['load_saved_bytes'],
                'store_saved_bytes': retention_stats['store_saved_bytes'],
                'mode': 'official',
                'tag': args.tag,
            }

        return {
            'problem': problem_path,
            'schedule': output_path,
            'actions': len(schedule),
            'loads': breakdown['load_count'],
            'stores': breakdown['store_count'],
            'computes': len([a for a in schedule if a['type'] == 'compute']),
            'latency': breakdown['total'],
            'total_loaded_bytes': analysis['total_loaded_bytes'],
            'total_stored_bytes': analysis['total_stored_bytes'],
            'required_output_stores': analysis['required_output_stores'],
            'spill_stores': analysis['spill_stores'],
            'peak_sram_bytes': analysis['peak_sram_bytes'],
            'peak_sram_pct': analysis['peak_sram_pct'],
            'loaded_bytes_per_compute': analysis['loaded_bytes_per_compute'],
            'mode': 'strict' if strict_mode else 'experimental',
            'tag': args.tag,
        }
    
    if args.all:
        problems_dir = Path(args.problem) if args.problem else Path('problems')
        if problems_dir.is_dir():
            problem_files = sorted(problems_dir.glob('*.json'))
        else:
            problem_files = sorted(Path('problems').glob('*.json'))

        if not problem_files:
            print("✗ Error: no problem JSON files found", file=sys.stderr)
            return 1

        results = []
        for problem_path in problem_files:
            print(f"=== {problem_path} ===")
            result = run_single(str(problem_path))
            results.append(result)

        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Report written to: {report_path}")
        elif args.tag:
            report_path = Path(args.output_dir) / f"report_{args.tag}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Report written to: {report_path}")

        return 0

    result = run_single(args.problem)
    if result.get('error'):
        return 1
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w') as f:
            json.dump([result], f, indent=2)
        print(f"✓ Report written to: {report_path}")
    elif args.tag:
        report_path = Path('schedules') / f"report_{args.tag}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w') as f:
            json.dump([result], f, indent=2)
        print(f"✓ Report written to: {report_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
