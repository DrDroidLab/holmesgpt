#!/usr/bin/env python3
"""
Benchmark Test Suite Executor

A comprehensive test suite for running LLM evaluation tests against different agents.
Each test run is saved to a JSON file in the results directory for analysis.

Usage:
    # Run single test
    python benchmarks/executor.py --agent drdroid --test-id 01_how_many_pods

    # Run multiple tests
    python benchmarks/executor.py --agent drdroid --test-id 01_how_many_pods --test-id 02_what_is_wrong_with_pod

    # Run all tests
    python benchmarks/executor.py --agent drdroid --all

    # Run tests by tag
    python benchmarks/executor.py --agent drdroid --tag kubernetes --tag easy

    # List available tests and agents
    python benchmarks/executor.py --list-tests
    python benchmarks/executor.py --list-agents
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.agent import AgentResult, TestCase, get_agent, list_agents
from benchmarks.config import (
    FIXTURES_DIR,
    RESULTS_DIR,
    Credentials,
    ensure_directories,
    load_credentials,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TestResult:
    """Complete result of a single test execution."""

    # Identifiers
    test_id: str
    agent: str
    model: str  # Model used (e.g., "sonnet4.5", "gpt5.2")
    run_id: str  # Unique ID for this run

    # Test case info
    user_prompt: str
    expected_output: List[str]
    tags: List[str]

    # Execution results
    status: str  # "passed", "failed", "setup_failed", "error"
    actual_output: Optional[str] = None
    tool_calls: List[str] = field(default_factory=list)

    # Judge evaluation
    score: Optional[float] = None
    judge_rationale: Optional[str] = None
    judge_model: Optional[str] = None

    # Timing
    setup_time: float = 0.0
    agent_time: float = 0.0
    judge_time: float = 0.0
    cleanup_time: float = 0.0
    total_time: float = 0.0

    # Agent metadata
    agent_metadata: Dict[str, Any] = field(default_factory=dict)

    # Error info
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Test Case Loading
# =============================================================================


def load_test_case(test_id: str, fixtures_path: Path = FIXTURES_DIR) -> TestCase:
    """Load a test case from YAML."""
    folder = fixtures_path / test_id
    yaml_path = folder / "test_case.yaml"

    if not yaml_path.exists():
        raise ValueError(f"Test case not found: {test_id}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    expected = data.get("expected_output", [])
    if isinstance(expected, str):
        expected = [expected]

    user_prompt = data.get("user_prompt", "")
    if isinstance(user_prompt, list):
        user_prompt = user_prompt[0]

    return TestCase(
        id=test_id,
        folder=str(folder),
        user_prompt=user_prompt,
        expected_output=expected,
        before_test=data.get("before_test"),
        after_test=data.get("after_test"),
        tags=data.get("tags", []),
        setup_timeout=data.get("setup_timeout", 300),
    )


def discover_tests(
    fixtures_path: Path = FIXTURES_DIR,
    tags: Optional[List[str]] = None,
) -> List[str]:
    """Discover all test IDs, optionally filtered by tags."""
    test_ids = []

    for item in fixtures_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            yaml_path = item / "test_case.yaml"
            if yaml_path.exists():
                # Filter by tags if specified
                if tags:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f)
                    test_tags = data.get("tags", [])
                    if not any(t in test_tags for t in tags):
                        continue

                test_ids.append(item.name)

    return sorted(test_ids)


# =============================================================================
# Setup/Cleanup Execution
# =============================================================================


def run_bash_script(
    script: str,
    cwd: str,
    timeout: int = 300,
    credentials: Optional[Credentials] = None,
    stream_output: bool = True,
) -> tuple[bool, str, float]:
    """Run a bash script with credentials applied to environment.

    Args:
        script: Bash script to run
        cwd: Working directory
        timeout: Timeout in seconds
        credentials: Credentials to apply to environment
        stream_output: If True, stream output to console in real-time
    """
    if not script or not script.strip():
        return True, "", 0.0

    # Prepare environment with credentials
    env = os.environ.copy()
    if credentials:
        env.update(credentials.to_env_vars())

    start_time = time.time()

    try:
        if stream_output:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                script,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1,  # Line buffered
            )

            output_lines = []
            try:
                # Read and print output line by line
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        print(f"  | {line.rstrip()}")
                        output_lines.append(line)

                    # Check timeout
                    if time.time() - start_time > timeout:
                        process.kill()
                        process.wait()
                        elapsed = time.time() - start_time
                        return False, f"Timeout after {timeout}s", elapsed

                process.wait()
                elapsed = time.time() - start_time
                output = "".join(output_lines)

                if process.returncode != 0:
                    return False, f"Exit code {process.returncode}\n{output}", elapsed

                return True, output, elapsed

            except Exception as e:
                process.kill()
                process.wait()
                raise e
        else:
            # Original behavior: capture output without streaming
            result = subprocess.run(
                script,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
            elapsed = time.time() - start_time
            output = f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

            if result.returncode != 0:
                return False, f"Exit code {result.returncode}\n{output}", elapsed

            return True, output, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, f"Timeout after {timeout}s", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Error: {str(e)}", elapsed


# =============================================================================
# LLM Judge
# =============================================================================


def evaluate_with_llm_judge(
    expected_elements: List[str],
    actual_output: str,
    classifier_model: str = "gpt-4.1",
) -> tuple[float, str]:
    """Evaluate output using LLM-as-judge."""
    try:
        from autoevals import LLMClassifier
    except ImportError:
        logger.error("autoevals not installed. Run: pip install autoevals")
        return 0.0, "autoevals not installed"

    expected_str = "\n- ".join(expected_elements)

    prompt_template = """
You are evaluating the correctness of an OUTPUT given by a LLM. You must return a score that
represents the correctness of that OUTPUT.

The correctness is defined by the presence of EXPECTED ELEMENTS in the OUTPUT.
Make a judgement call whether each ELEMENT sufficiently matches the OUTPUT. ELEMENTS do
not need to appear verbatim or be a perfect match but their essence should be
present in the whole OUTPUT, even if it spans multiple sentences.

# EXPECTED ELEMENTS

- {{expected}}

# OUTPUT

{{output}}


Return a choice based on the number of EXPECTED ELEMENTS present in the OUTPUT.
Possible choices:
- A: All elements are present
- B: Either no element is present or only some but not all elements are present
"""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM judge")
        return 0.0, "No API key for judge"

    try:
        classifier = LLMClassifier(
            name="Correctness",
            prompt_template=prompt_template,
            choice_scores={"A": 1, "B": 0},
            use_cot=True,
            model=classifier_model,
            api_key=api_key,
        )

        result = classifier(
            input=prompt_template,
            output=actual_output,
            expected=expected_str,
        )

        return result.score, result.metadata.get("rationale", "")

    except Exception as e:
        logger.error(f"Judge error: {e}")
        return 0.0, f"Judge error: {str(e)}"


# =============================================================================
# Result Storage
# =============================================================================


def generate_run_id(model: str, test_id: str) -> str:
    """Generate a unique run ID based on model and test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{test_id}_{timestamp}"


def save_result(result: TestResult, results_dir: Path = RESULTS_DIR) -> Path:
    """Save a test result to JSON file."""
    ensure_directories()

    filename = f"{result.run_id}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return filepath


def load_results(
    results_dir: Path = RESULTS_DIR,
    agent: Optional[str] = None,
    test_id: Optional[str] = None,
) -> List[TestResult]:
    """Load results from JSON files, optionally filtered."""
    results = []

    if not results_dir.exists():
        return results

    for filepath in results_dir.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Filter by agent
            if agent and data.get("agent") != agent:
                continue

            # Filter by test_id
            if test_id and data.get("test_id") != test_id:
                continue

            results.append(TestResult(**data))
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    return sorted(results, key=lambda r: r.started_at, reverse=True)


# =============================================================================
# Test Executor
# =============================================================================


class BenchmarkExecutor:
    """Executes benchmark tests against registered agents."""

    def __init__(
        self,
        agent: str,
        model: str,
        credentials: Optional[Credentials] = None,
        skip_setup: bool = False,
        skip_cleanup: bool = False,
        classifier_model: str = "gpt-4.1",
        quiet: bool = False,
    ):
        if not agent:
            raise ValueError(
                "Agent must be specified. Use --agent <name>. "
                f"Available agents: {', '.join(list_agents())}"
            )

        if not model:
            raise ValueError(
                "Model must be specified. Use --model <name>. "
                "Examples: sonnet4.5, gpt5.2, gpt-4.1"
            )

        self.agent_name = agent
        self.model = model
        self.agent_fn = get_agent(agent)
        self.credentials = credentials or load_credentials()
        self.skip_setup = skip_setup
        self.skip_cleanup = skip_cleanup
        self.classifier_model = classifier_model
        self.quiet = quiet  # If True, don't stream script output to console
        self.results: List[TestResult] = []

        # Apply credentials to environment
        self.credentials.apply_to_env()

        # Set model in environment for agent to use
        os.environ["DRDROID_MODEL"] = model

    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        run_id = generate_run_id(self.model, test_case.id)
        start_time = time.time()

        logger.info(f"{'='*60}")
        logger.info(f"Test: {test_case.id}")
        logger.info(f"Agent: {self.agent_name}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"{'='*60}")

        result = TestResult(
            test_id=test_case.id,
            agent=self.agent_name,
            model=self.model,
            run_id=run_id,
            user_prompt=test_case.user_prompt,
            expected_output=test_case.expected_output,
            tags=test_case.tags,
            status="error",
        )

        # 1. Run setup
        if not self.skip_setup and test_case.before_test:
            logger.info("Running setup...")
            success, output, elapsed = run_bash_script(
                test_case.before_test,
                test_case.folder,
                timeout=test_case.setup_timeout,
                credentials=self.credentials,
                stream_output=not self.quiet,  # Stream output unless --quiet
            )
            result.setup_time = elapsed

            if not success:
                logger.error(f"Setup failed:\n{output[:500]}...")
                result.status = "setup_failed"
                result.error_message = output
                result.error_type = "setup_failure"
                result.completed_at = datetime.now().isoformat()
                result.total_time = time.time() - start_time
                save_result(result)
                return result

            logger.info(f"Setup completed in {elapsed:.2f}s")

        # 2. Run agent
        try:
            logger.info(f"Running agent [{self.agent_name}]...")
            agent_start = time.time()

            agent_result = self.agent_fn(test_case)

            result.agent_time = time.time() - agent_start
            result.actual_output = agent_result.output
            result.tool_calls = agent_result.tool_calls
            result.agent_metadata = agent_result.metadata

            logger.info(f"Agent completed in {result.agent_time:.2f}s")

        except Exception as e:
            logger.error(f"Agent error: {e}")
            result.status = "error"
            result.error_message = str(e)
            result.error_type = type(e).__name__
            self._run_cleanup(test_case, result)
            result.completed_at = datetime.now().isoformat()
            result.total_time = time.time() - start_time
            save_result(result)
            return result

        # 3. Run LLM judge
        try:
            logger.info("Running LLM judge...")
            judge_start = time.time()

            score, rationale = evaluate_with_llm_judge(
                test_case.expected_output,
                result.actual_output,
                self.classifier_model,
            )

            result.judge_time = time.time() - judge_start
            result.score = score
            result.judge_rationale = rationale
            result.judge_model = self.classifier_model
            result.status = "passed" if score == 1 else "failed"

            logger.info(f"Judge score: {score} ({result.status})")

        except Exception as e:
            logger.error(f"Judge error: {e}")
            result.status = "error"
            result.error_message = f"Judge error: {str(e)}"
            result.error_type = "judge_error"

        # 4. Run cleanup
        self._run_cleanup(test_case, result)

        # Finalize
        result.completed_at = datetime.now().isoformat()
        result.total_time = time.time() - start_time

        # Save result
        filepath = save_result(result)
        logger.info(f"Result saved to: {filepath}")

        return result

    def _run_cleanup(self, test_case: TestCase, result: TestResult) -> None:
        """Run cleanup after test."""
        if not self.skip_cleanup and test_case.after_test:
            logger.info("Running cleanup...")
            success, output, elapsed = run_bash_script(
                test_case.after_test,
                test_case.folder,
                timeout=120,
                credentials=self.credentials,
                stream_output=False,  # Cleanup output is less important
            )
            result.cleanup_time = elapsed
            if not success:
                logger.warning(f"Cleanup failed (non-fatal)")
            else:
                logger.info(f"Cleanup completed in {elapsed:.2f}s")

    def run_tests(self, test_ids: List[str]) -> List[TestResult]:
        """Run multiple tests."""
        results = []

        for i, test_id in enumerate(test_ids, 1):
            logger.info(f"\n[{i}/{len(test_ids)}] Running test: {test_id}")
            try:
                test_case = load_test_case(test_id)
                result = self.run_test(test_case)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error loading test {test_id}: {e}")

        return results

    def print_summary(self) -> None:
        """Print summary of all results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        setup_failed = sum(1 for r in self.results if r.status == "setup_failed")
        errors = sum(1 for r in self.results if r.status == "error")

        total_time = sum(r.total_time for r in self.results)

        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULTS")
        print(f"Model: {self.model}")
        print("=" * 70)
        print(f"Total:        {total}")
        print(f"Passed:       {passed} ✅")
        print(f"Failed:       {failed} ❌")
        print(f"Setup Failed: {setup_failed} 🔧")
        print(f"Errors:       {errors} ⚠️")
        print(f"Pass Rate:    {(passed/total*100):.1f}%" if total > 0 else "N/A")
        print(f"Total Time:   {total_time:.2f}s")
        print("=" * 70)

        print("\nDetailed Results:")
        print("-" * 70)
        for r in self.results:
            icon = {"passed": "✅", "failed": "❌", "setup_failed": "🔧", "error": "⚠️"}.get(
                r.status, "?"
            )
            print(f"{icon} {r.test_id}: {r.status.upper()} (score={r.score}, time={r.total_time:.1f}s)")
            if r.error_message:
                print(f"   Error: {r.error_message[:80]}...")
        print()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Test Suite Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single test with model
  python benchmarks/executor.py --model sonnet4.5 --test-id 01_how_many_pods

  # Run multiple tests
  python benchmarks/executor.py --model gpt5.2 --test-id 01_how_many_pods --test-id 02_what_is_wrong_with_pod

  # Run all tests
  python benchmarks/executor.py --model sonnet4.5 --all

  # Run tests by tag
  python benchmarks/executor.py --model sonnet4.5 --tag kubernetes --tag easy

  # List available tests
  python benchmarks/executor.py --list-tests
  python benchmarks/executor.py --list-tests --tag kubernetes
        """,
    )

    # Model selection (REQUIRED)
    parser.add_argument(
        "--model",
        help="Model to use for testing (REQUIRED). Examples: sonnet4.5, gpt5.2",
    )

    # Agent selection (defaults to drdroid)
    parser.add_argument(
        "--agent",
        default="drdroid",
        help="Agent to use (default: drdroid)",
    )

    # Test selection
    parser.add_argument(
        "--test-id",
        action="append",
        dest="test_ids",
        help="Test ID(s) to run (can specify multiple)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available tests",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Filter tests by tag (can specify multiple)",
    )

    # Execution options
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip before_test scripts",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip after_test scripts",
    )
    parser.add_argument(
        "--classifier-model",
        default=os.environ.get("CLASSIFIER_MODEL", "gpt-4.1"),
        help="Model for LLM judge (default: gpt-4.1)",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Path to credentials YAML file",
    )

    # List commands
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all available test IDs",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all registered agents",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress setup/cleanup script output (don't stream to console)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle list commands
    if args.list_agents:
        print("Available agents:")
        for name in list_agents():
            print(f"  - {name}")
        return

    if args.list_tests:
        test_ids = discover_tests(tags=args.tags)
        print(f"Available tests ({len(test_ids)}):")
        for tid in test_ids:
            test_case = load_test_case(tid)
            tags_str = f" [{', '.join(test_case.tags)}]" if test_case.tags else ""
            print(f"  - {tid}{tags_str}")
        return

    # Validate model is specified for test execution
    if not args.model:
        parser.print_help()
        print("\n" + "=" * 60)
        print("ERROR: --model is required for test execution")
        print("Examples: --model sonnet4.5, --model gpt5.2")
        print("=" * 60)
        sys.exit(1)

    # Load credentials
    credentials = load_credentials(args.credentials)

    # Create executor
    executor = BenchmarkExecutor(
        agent=args.agent,
        model=args.model,
        credentials=credentials,
        skip_setup=args.skip_setup,
        skip_cleanup=args.skip_cleanup,
        classifier_model=args.classifier_model,
        quiet=args.quiet,
    )

    # Determine tests to run
    if args.all:
        test_ids = discover_tests(tags=args.tags)
    elif args.test_ids:
        test_ids = args.test_ids
    elif args.tags:
        test_ids = discover_tests(tags=args.tags)
    else:
        parser.print_help()
        print("\nError: Specify --test-id, --all, or --tag")
        sys.exit(1)

    if not test_ids:
        print("No tests found matching criteria")
        sys.exit(1)

    # Run tests
    logger.info(f"Running {len(test_ids)} test(s) with model '{args.model}'...")
    executor.run_tests(test_ids)

    # Print summary
    executor.print_summary()

    # Exit code based on results
    report = {
        "passed": sum(1 for r in executor.results if r.status == "passed"),
        "total": len(executor.results),
    }

    if report["passed"] == report["total"] and report["total"] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
