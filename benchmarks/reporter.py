#!/usr/bin/env python3
"""
Benchmark Report Generator

Generates reports from saved test results focusing on MODEL and TEST CASE analysis.

Usage:
    # Summary report
    python benchmarks/reporter.py --summary

    # Compare models
    python benchmarks/reporter.py --compare-models

    # Report by test case (use case)
    python benchmarks/reporter.py --by-test

    # Report for specific model
    python benchmarks/reporter.py --model sonnet4.5

    # Report for specific test case
    python benchmarks/reporter.py --test-id 01_how_many_pods

    # Export to JSON/CSV
    python benchmarks/reporter.py --compare-models --output report.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.config import RESULTS_DIR


def load_all_results(results_dir: Path = RESULTS_DIR) -> List[Dict[str, Any]]:
    """Load all results from JSON files."""
    results = []

    if not results_dir.exists():
        return results

    for filepath in results_dir.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return sorted(results, key=lambda r: r.get("started_at", ""), reverse=True)


def filter_results(
    results: List[Dict[str, Any]],
    model: Optional[str] = None,
    test_id: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter results by criteria."""
    filtered = results

    if model:
        filtered = [r for r in filtered if r.get("model") == model]

    if test_id:
        filtered = [r for r in filtered if r.get("test_id") == test_id]

    if status:
        filtered = [r for r in filtered if r.get("status") == status]

    if since:
        filtered = [r for r in filtered if r.get("started_at", "") >= since]

    return filtered


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate overall summary report."""
    if not results:
        return {"error": "No results found"}

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    setup_failed = sum(1 for r in results if r.get("status") == "setup_failed")
    errors = sum(1 for r in results if r.get("status") == "error")

    # Timing stats
    total_times = [r.get("total_time", 0) for r in results if r.get("total_time")]
    agent_times = [r.get("agent_time", 0) for r in results if r.get("agent_time")]

    # Unique counts
    unique_models = set(r.get("model") for r in results if r.get("model"))
    unique_tests = set(r.get("test_id") for r in results if r.get("test_id"))

    return {
        "summary": {
            "total_runs": total,
            "passed": passed,
            "failed": failed,
            "setup_failed": setup_failed,
            "errors": errors,
            "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "N/A",
        },
        "timing": {
            "avg_total_time": round(sum(total_times) / len(total_times), 2) if total_times else 0,
            "avg_agent_time": round(sum(agent_times) / len(agent_times), 2) if agent_times else 0,
            "max_total_time": round(max(total_times), 2) if total_times else 0,
            "min_total_time": round(min(total_times), 2) if total_times else 0,
        },
        "coverage": {
            "unique_models": len(unique_models),
            "unique_tests": len(unique_tests),
            "models": sorted(unique_models),
        },
        "generated_at": datetime.now().isoformat(),
    }


def generate_model_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate model comparison report."""
    model_stats = defaultdict(lambda: {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "setup_failed": 0,
        "errors": 0,
        "total_time": 0,
        "agent_time": 0,
        "tests": set(),
    })

    for r in results:
        model = r.get("model", "unknown")
        status = r.get("status", "unknown")

        model_stats[model]["total"] += 1
        model_stats[model]["tests"].add(r.get("test_id"))

        if status == "passed":
            model_stats[model]["passed"] += 1
        elif status == "failed":
            model_stats[model]["failed"] += 1
        elif status == "setup_failed":
            model_stats[model]["setup_failed"] += 1
        else:
            model_stats[model]["errors"] += 1

        model_stats[model]["total_time"] += r.get("total_time", 0)
        model_stats[model]["agent_time"] += r.get("agent_time", 0)

    # Convert to serializable format
    comparison = {}
    for model, stats in sorted(model_stats.items()):
        total = stats["total"]
        comparison[model] = {
            "total_runs": total,
            "passed": stats["passed"],
            "failed": stats["failed"],
            "setup_failed": stats["setup_failed"],
            "errors": stats["errors"],
            "pass_rate": f"{(stats['passed'] / total * 100):.1f}%" if total > 0 else "N/A",
            "pass_rate_numeric": round(stats['passed'] / total * 100, 1) if total > 0 else 0,
            "avg_total_time": round(stats["total_time"] / total, 2) if total > 0 else 0,
            "avg_agent_time": round(stats["agent_time"] / total, 2) if total > 0 else 0,
            "unique_tests": len(stats["tests"]),
        }

    return {
        "model_comparison": comparison,
        "generated_at": datetime.now().isoformat(),
    }


def generate_test_case_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate report grouped by test case (use case)."""
    test_stats = defaultdict(lambda: {
        "models": defaultdict(lambda: {
            "runs": 0,
            "passed": 0,
            "failed": 0,
            "avg_time": 0,
            "total_time": 0,
        }),
        "tags": [],
        "user_prompt": "",
    })

    for r in results:
        test_id = r.get("test_id", "unknown")
        model = r.get("model", "unknown")
        status = r.get("status", "unknown")

        test_stats[test_id]["models"][model]["runs"] += 1
        test_stats[test_id]["models"][model]["total_time"] += r.get("total_time", 0)

        if status == "passed":
            test_stats[test_id]["models"][model]["passed"] += 1
        elif status == "failed":
            test_stats[test_id]["models"][model]["failed"] += 1

        # Store test metadata
        if not test_stats[test_id]["user_prompt"]:
            test_stats[test_id]["user_prompt"] = r.get("user_prompt", "")
            test_stats[test_id]["tags"] = r.get("tags", [])

    # Convert to serializable format
    report = {}
    for test_id, stats in sorted(test_stats.items()):
        model_results = {}
        for model, model_stats in stats["models"].items():
            runs = model_stats["runs"]
            model_results[model] = {
                "runs": runs,
                "passed": model_stats["passed"],
                "failed": model_stats["failed"],
                "pass_rate": f"{(model_stats['passed'] / runs * 100):.1f}%" if runs > 0 else "N/A",
                "avg_time": round(model_stats["total_time"] / runs, 2) if runs > 0 else 0,
            }

        report[test_id] = {
            "tags": stats["tags"],
            "user_prompt": stats["user_prompt"][:100] + "..." if len(stats["user_prompt"]) > 100 else stats["user_prompt"],
            "models": model_results,
        }

    return {
        "test_case_report": report,
        "generated_at": datetime.now().isoformat(),
    }


def generate_single_test_report(
    results: List[Dict[str, Any]],
    test_id: str,
) -> Dict[str, Any]:
    """Generate detailed report for a specific test case."""
    test_results = [r for r in results if r.get("test_id") == test_id]

    if not test_results:
        return {"error": f"No results found for test: {test_id}"}

    # Group by model
    by_model = defaultdict(list)
    for r in test_results:
        by_model[r.get("model", "unknown")].append(r)

    model_performance = {}
    for model, runs in sorted(by_model.items()):
        passed = sum(1 for r in runs if r.get("status") == "passed")
        failed = sum(1 for r in runs if r.get("status") == "failed")
        total = len(runs)

        # Get latest run details
        latest = runs[0] if runs else {}

        model_performance[model] = {
            "runs": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "N/A",
            "avg_time": round(sum(r.get("total_time", 0) for r in runs) / total, 2) if total > 0 else 0,
            "latest_status": latest.get("status"),
            "latest_score": latest.get("score"),
            "latest_run_id": latest.get("run_id"),
            "latest_judge_rationale": latest.get("judge_rationale", "")[:200] if latest.get("judge_rationale") else None,
        }

    return {
        "test_id": test_id,
        "total_runs": len(test_results),
        "models_tested": sorted(by_model.keys()),
        "tags": test_results[0].get("tags", []) if test_results else [],
        "user_prompt": test_results[0].get("user_prompt") if test_results else None,
        "expected_output": test_results[0].get("expected_output") if test_results else None,
        "model_performance": model_performance,
        "generated_at": datetime.now().isoformat(),
    }


def generate_detailed_report(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate detailed report with all results."""
    return [
        {
            "test_id": r.get("test_id"),
            "model": r.get("model"),
            "run_id": r.get("run_id"),
            "status": r.get("status"),
            "score": r.get("score"),
            "total_time": r.get("total_time"),
            "agent_time": r.get("agent_time"),
            "setup_time": r.get("setup_time"),
            "judge_time": r.get("judge_time"),
            "started_at": r.get("started_at"),
            "error_message": r.get("error_message"),
        }
        for r in results
    ]


# =============================================================================
# Console Output Functions
# =============================================================================


def print_summary_report(report: Dict[str, Any]) -> None:
    """Print summary report to console."""
    summary = report.get("summary", {})
    timing = report.get("timing", {})
    coverage = report.get("coverage", {})

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 70)

    print("\nOverall Statistics:")
    print(f"  Total Runs:     {summary.get('total_runs', 0)}")
    print(f"  Passed:         {summary.get('passed', 0)} ✅")
    print(f"  Failed:         {summary.get('failed', 0)} ❌")
    print(f"  Setup Failed:   {summary.get('setup_failed', 0)} 🔧")
    print(f"  Errors:         {summary.get('errors', 0)} ⚠️")
    print(f"  Pass Rate:      {summary.get('pass_rate', 'N/A')}")

    print("\nTiming:")
    print(f"  Avg Total Time: {timing.get('avg_total_time', 0):.2f}s")
    print(f"  Avg Agent Time: {timing.get('avg_agent_time', 0):.2f}s")

    print("\nCoverage:")
    print(f"  Unique Models:  {coverage.get('unique_models', 0)}")
    print(f"  Unique Tests:   {coverage.get('unique_tests', 0)}")
    print(f"  Models:         {', '.join(coverage.get('models', []))}")

    print("=" * 70 + "\n")


def print_model_comparison(report: Dict[str, Any]) -> None:
    """Print model comparison to console."""
    comparison = report.get("model_comparison", {})

    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    # Header
    print(f"\n{'Model':<20} {'Runs':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8} {'Avg Time':>10}")
    print("-" * 70)

    for model, stats in sorted(comparison.items(), key=lambda x: -x[1].get("pass_rate_numeric", 0)):
        print(
            f"{model:<20} "
            f"{stats.get('total_runs', 0):>6} "
            f"{stats.get('passed', 0):>6} "
            f"{stats.get('failed', 0):>6} "
            f"{stats.get('pass_rate', 'N/A'):>8} "
            f"{stats.get('avg_agent_time', 0):>9.2f}s"
        )

    print("=" * 70 + "\n")


def print_test_case_report(report: Dict[str, Any]) -> None:
    """Print test case report to console."""
    test_cases = report.get("test_case_report", {})

    print("\n" + "=" * 70)
    print("TEST CASE REPORT (by Use Case)")
    print("=" * 70)

    for test_id, data in sorted(test_cases.items()):
        tags_str = f" [{', '.join(data.get('tags', []))}]" if data.get('tags') else ""
        print(f"\n{test_id}{tags_str}")
        print(f"  Prompt: {data.get('user_prompt', 'N/A')}")

        models = data.get("models", {})
        if models:
            print(f"  {'Model':<18} {'Runs':>5} {'Pass':>5} {'Rate':>8} {'Time':>8}")
            print(f"  {'-'*50}")
            for model, stats in sorted(models.items()):
                print(
                    f"  {model:<18} "
                    f"{stats.get('runs', 0):>5} "
                    f"{stats.get('passed', 0):>5} "
                    f"{stats.get('pass_rate', 'N/A'):>8} "
                    f"{stats.get('avg_time', 0):>7.2f}s"
                )

    print("\n" + "=" * 70 + "\n")


def print_single_test_report(report: Dict[str, Any]) -> None:
    """Print single test report to console."""
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    print("\n" + "=" * 70)
    print(f"TEST REPORT: {report.get('test_id', 'Unknown')}")
    print("=" * 70)

    print(f"\nTotal Runs: {report.get('total_runs', 0)}")
    print(f"Models Tested: {', '.join(report.get('models_tested', []))}")
    print(f"Tags: {', '.join(report.get('tags', []))}")

    print(f"\nPrompt: {report.get('user_prompt', 'N/A')}")

    print("\nExpected Output:")
    for exp in report.get("expected_output", []):
        print(f"  - {exp}")

    print("\nModel Performance:")
    print(f"  {'Model':<18} {'Runs':>5} {'Pass':>5} {'Rate':>8} {'Latest':>10} {'Score':>6}")
    print(f"  {'-'*60}")

    for model, perf in sorted(report.get("model_performance", {}).items()):
        print(
            f"  {model:<18} "
            f"{perf.get('runs', 0):>5} "
            f"{perf.get('passed', 0):>5} "
            f"{perf.get('pass_rate', 'N/A'):>8} "
            f"{perf.get('latest_status', 'N/A'):>10} "
            f"{perf.get('latest_score', 'N/A'):>6}"
        )

    print("\n" + "=" * 70 + "\n")


def export_to_file(data: Any, filepath: Path) -> None:
    """Export report to JSON or CSV file."""
    if filepath.suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Report saved to: {filepath}")

    elif filepath.suffix == ".csv":
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "model_comparison" in data:
            rows = [
                {"model": model, **stats}
                for model, stats in data["model_comparison"].items()
            ]
        elif isinstance(data, dict) and "test_case_report" in data:
            rows = []
            for test_id, test_data in data["test_case_report"].items():
                for model, model_stats in test_data.get("models", {}).items():
                    rows.append({
                        "test_id": test_id,
                        "model": model,
                        "tags": ", ".join(test_data.get("tags", [])),
                        **model_stats,
                    })
        else:
            print("Cannot export this report type to CSV")
            return

        if rows:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Report saved to: {filepath}")
    else:
        print(f"Unsupported file format: {filepath.suffix}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports (Model + Test Case level)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summary report
  python benchmarks/reporter.py --summary

  # Compare models side-by-side
  python benchmarks/reporter.py --compare-models

  # Report by test case (use case)
  python benchmarks/reporter.py --by-test

  # Report for specific model
  python benchmarks/reporter.py --model sonnet4.5

  # Report for specific test case
  python benchmarks/reporter.py --test-id 01_how_many_pods

  # Export to file
  python benchmarks/reporter.py --compare-models --output comparison.json
  python benchmarks/reporter.py --by-test --output tests.csv
        """,
    )

    # Report types
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary report",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare performance across models",
    )
    parser.add_argument(
        "--by-test",
        action="store_true",
        help="Report grouped by test case (use case)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results list",
    )

    # Filters
    parser.add_argument(
        "--model",
        help="Filter by model",
    )
    parser.add_argument(
        "--test-id",
        help="Report on specific test case",
    )
    parser.add_argument(
        "--status",
        choices=["passed", "failed", "setup_failed", "error"],
        help="Filter by status",
    )
    parser.add_argument(
        "--since",
        help="Filter results since date (ISO format)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (JSON or CSV)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Results directory",
    )

    args = parser.parse_args()

    # Load results
    results = load_all_results(args.results_dir)

    if not results:
        print("No results found in results directory")
        print(f"Results directory: {args.results_dir}")
        sys.exit(1)

    # Apply filters (except test_id if doing single test report)
    filtered = filter_results(
        results,
        model=args.model,
        test_id=None,  # Don't filter by test_id for filtering, handled separately
        status=args.status,
        since=args.since,
    )

    # Generate requested report
    if args.test_id:
        # Single test report
        report = generate_single_test_report(results, args.test_id)
        if args.output:
            export_to_file(report, args.output)
        else:
            print_single_test_report(report)

    elif args.compare_models:
        report = generate_model_comparison(filtered)
        if args.output:
            export_to_file(report, args.output)
        else:
            print_model_comparison(report)

    elif args.by_test:
        report = generate_test_case_report(filtered)
        if args.output:
            export_to_file(report, args.output)
        else:
            print_test_case_report(report)

    elif args.detailed:
        report = generate_detailed_report(filtered)
        if args.output:
            export_to_file(report, args.output)
        else:
            for r in report[:30]:
                status_icon = {"passed": "✅", "failed": "❌", "setup_failed": "🔧", "error": "⚠️"}.get(
                    r["status"], "?"
                )
                print(f"{status_icon} {r['model']}/{r['test_id']}: {r['status']} (score={r['score']}, time={r['total_time']:.1f}s)")
            if len(report) > 30:
                print(f"... and {len(report) - 30} more results")

    else:
        # Default to summary
        report = generate_summary_report(filtered)
        if args.output:
            export_to_file(report, args.output)
        else:
            print_summary_report(report)


if __name__ == "__main__":
    main()
