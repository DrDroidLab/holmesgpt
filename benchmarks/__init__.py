"""
Benchmark Test Suite

A comprehensive test suite for evaluating LLM agents against HolmesGPT test cases.

Modules:
    - agent.py: Agent implementations and registry
    - executor.py: Test execution engine
    - config.py: Credentials and configuration management
    - reporter.py: Report generation from results

Usage:
    # Run tests
    python benchmarks/executor.py --agent drdroid --test-id 01_how_many_pods

    # Generate reports
    python benchmarks/reporter.py --summary
    python benchmarks/reporter.py --compare-agents
"""

from benchmarks.agent import (
    AgentResult,
    TestCase,
    get_agent,
    list_agents,
    register_agent,
)
from benchmarks.config import (
    Credentials,
    load_credentials,
)
from benchmarks.executor import (
    BenchmarkExecutor,
    TestResult,
    discover_tests,
    load_test_case,
)

__all__ = [
    # Agent
    "AgentResult",
    "TestCase",
    "get_agent",
    "list_agents",
    "register_agent",
    # Config
    "Credentials",
    "load_credentials",
    # Executor
    "BenchmarkExecutor",
    "TestResult",
    "discover_tests",
    "load_test_case",
]
