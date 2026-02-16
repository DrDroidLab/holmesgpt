# Benchmarking Guide

This guide covers how to run benchmark tests using the DrDroid agent against HolmesGPT's evaluation test cases, generate reports, and review results in the Streamlit dashboard.

## Prerequisites

- Python 3.10+
- A running Kubernetes cluster with `kubectl` configured
- API keys for the LLM judge (OpenAI) and DrDroid agent
- Install extra dependencies:
  ```bash
  pip install streamlit pandas pyyaml requests
  ```

## 1. Setup Credentials

```bash
cp benchmarks/config/credentials.yaml.template benchmarks/config/credentials.yaml
```

Edit `benchmarks/config/credentials.yaml` and fill in:

```yaml
# Required for the DrDroid agent
custom:
  drdroid:
    api_url: http://your-drdroid-api-url
    api_key: your-drdroid-api-key

# Required for the LLM judge that scores results
openai:
  api_key: sk-...

judge:
  model: gpt-4.1

# Required for Kubernetes-based tests
kubernetes:
  kubeconfig: ~/.kube/config
  context: your-cluster-context
```

Alternatively, set environment variables (these override the YAML file):

```bash
export DRDROID_API_URL=http://your-drdroid-api-url
export DRDROID_API_KEY=your-drdroid-api-key
export OPENAI_API_KEY=sk-...
export CLASSIFIER_MODEL=gpt-4.1
```

## 2. Running Benchmark Tests

### List Available Tests

```bash
python benchmarks/executor.py --list-tests
```

### Run a Single Test

```bash
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --test-id 01_how_many_pods
```

The `--model` flag is **required** and labels which model the agent is using (used for tracking/comparison). The `--agent` flag selects the agent implementation (defaults to `drdroid`).

### Run Multiple Specific Tests

```bash
python benchmarks/executor.py --model sonnet4.5 --agent drdroid \
    --test-id 01_how_many_pods \
    --test-id 02_what_is_wrong_with_pod \
    --test-id 09_crashpod
```

### Run All Tests

```bash
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --all
```

### Run Tests by Tag

```bash
# Run only Kubernetes tests
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --tag kubernetes

# Run easy/regression tests
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --tag easy

# Multiple tags (OR logic)
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --tag kubernetes --tag prometheus
```

### Skip Setup or Cleanup

Useful for iterative debugging:

```bash
# Skip infrastructure setup (if resources are already running)
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --test-id 01_how_many_pods --skip-setup

# Skip cleanup (keep infrastructure running after test)
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --test-id 01_how_many_pods --skip-cleanup
```

## 3. Available Agents

| Agent        | Description                                      |
|------------- |--------------------------------------------------|
| `drdroid`    | DrDroid Investigation API                        |
| `holmes`     | HolmesGPT ToolCallingLLM                         |
| `claudecode` | Local Claude Code CLI with read-only kubectl     |
| `openai`     | Simple OpenAI completion (no tools)              |

To compare agents, run the same tests with different `--agent` flags:

```bash
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --all
python benchmarks/executor.py --model sonnet4.5 --agent holmes --all
```

## 4. Generating Reports

### CLI Summary

```bash
python benchmarks/reporter.py --summary
```

Shows overall pass rate, timing, and coverage stats.

### Model Comparison

```bash
python benchmarks/reporter.py --compare-models
```

Side-by-side comparison of pass rates across models.

### Report by Test Case

```bash
python benchmarks/reporter.py --by-test
```

Breakdown of results per test case, showing which models passed/failed each one.

### Filter Results

```bash
# Results for a specific model only
python benchmarks/reporter.py --summary --model sonnet4.5

# Results since a specific date
python benchmarks/reporter.py --summary --since 2026-01-30

# Only failed tests
python benchmarks/reporter.py --detailed --status failed

# Specific test case
python benchmarks/reporter.py --test-id 01_how_many_pods
```

### Export to File

```bash
# JSON export
python benchmarks/reporter.py --summary --output report.json

# CSV export
python benchmarks/reporter.py --compare-models --output comparison.csv
python benchmarks/reporter.py --by-test --output tests.csv
```

## 5. Reviewing Results in the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run benchmarks/dashboard.py
```

Or on a custom port:

```bash
streamlit run benchmarks/dashboard.py --server.port 8501
```

### Dashboard Sections

- **Overview** - Total runs, pass rate, cost, and tokens summary
- **Model Comparison** - Side-by-side comparison of all models tested
- **Test Cases** - Per-test-case analysis with per-model breakdown
- **Use Case x Model Matrix** - Pivot table showing status/metrics for every test-model combination
- **Raw Results** - Detailed view with filtering and drill-down into individual runs

### Dashboard Features

- **Refresh Data** button to reload latest results
- Filter by model, status, and date range
- Download CSV exports directly from the UI
- View detailed output, judge rationale, and errors for any run

## 6. Understanding Results

Each test run produces a JSON file in `benchmarks/results/`:

```
results/
  sonnet4.5_01_how_many_pods_20260130_163000.json
  sonnet4.5_02_what_is_wrong_with_pod_20260130_163100.json
  ...
```

Key fields in each result:

| Field             | Description                                 |
|-------------------|---------------------------------------------|
| `status`          | `passed`, `failed`, `setup_failed`, `error` |
| `score`           | 0.0 to 1.0 score from the LLM judge        |
| `judge_rationale` | Explanation of why the judge scored it       |
| `actual_output`   | The agent's raw response                    |
| `agent_time`      | Time the agent took to respond              |
| `setup_time`      | Time for infrastructure setup               |
| `tool_calls`      | Tools the agent invoked                     |

## 7. Adding More Tests

Test cases live in `tests/llm/fixtures/test_ask_holmes/`. Each test is a directory containing a `test_case.yaml`:

```yaml
user_prompt: "Your question here?"
expected_output:
  - "What the judge should check for"
  - "Another expected fact"
tags:
  - kubernetes
  - easy
before_test: |
  # Bash script to set up infrastructure
  kubectl apply -f manifests.yaml
after_test: |
  # Bash script to clean up
  kubectl delete -f manifests.yaml
```

After adding a test, verify it appears:

```bash
python benchmarks/executor.py --list-tests
```

Then run it:

```bash
python benchmarks/executor.py --model sonnet4.5 --agent drdroid --test-id your_new_test
```

## 8. Branch-Specific Changes

The `benchmarking-drdroid-agent` branch includes these changes over `master`:

1. **New `benchmarks/` module** - Complete benchmarking framework with executor, agent registry, reporter, config management, and Streamlit dashboard
2. **Updated test prompts** - Added "in Azure Prod cluster" context to test prompts (tests 01, 02, 04, 05, 07, 09, 10, 11, 12) for more realistic DrDroid agent evaluation
3. **Increased setup timeouts** - Longer wait times in `before_test` scripts for tests 01 and 12 to handle slower cluster environments
