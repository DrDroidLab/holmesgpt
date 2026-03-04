# Benchmark Test Suite

A comprehensive test suite for evaluating LLM models against HolmesGPT test cases.
Results are tracked by **model** and **test case (use case)** for easy comparison.

## Directory Structure

```
benchmarks/
├── config/
│   ├── credentials.yaml           # Your credentials (git-ignored)
│   └── credentials.yaml.template  # Template for credentials
├── results/                       # Test results (JSON files)
├── agent.py                       # Agent implementations
├── executor.py                    # Test execution engine
├── config.py                      # Configuration management
├── reporter.py                    # CLI report generation
├── dashboard.py                   # Streamlit dashboard
└── README.md
```

## Quick Start

```bash
# 1. Setup credentials
cp benchmarks/config/credentials.yaml.template benchmarks/config/credentials.yaml
# Edit credentials.yaml with your API keys

# 2. List available tests
python benchmarks/executor.py --list-tests

# 3. Run a test (model is REQUIRED)
python benchmarks/executor.py --model sonnet4.5 --test-id 01_how_many_pods

# 4. View results (choose one)
python benchmarks/reporter.py --summary           # CLI summary
python benchmarks/reporter.py --compare-models    # Model comparison
streamlit run benchmarks/dashboard.py             # Interactive dashboard
```

## Running Tests

### Single Test
```bash
python benchmarks/executor.py --model sonnet4.5 --test-id 01_how_many_pods
```

### Multiple Tests
```bash
python benchmarks/executor.py --model gpt5.2 \
    --test-id 01_how_many_pods \
    --test-id 02_what_is_wrong_with_pod
```

### All Tests
```bash
python benchmarks/executor.py --model sonnet4.5 --all
```

### Tests by Tag
```bash
# Run all kubernetes tests
python benchmarks/executor.py --model sonnet4.5 --tag kubernetes

# Run easy tests
python benchmarks/executor.py --model gpt5.2 --tag easy

# Multiple tags (OR logic)
python benchmarks/executor.py --model sonnet4.5 --tag kubernetes --tag prometheus
```

### Skip Setup/Cleanup
```bash
# Skip infrastructure setup (useful for debugging)
python benchmarks/executor.py --model sonnet4.5 --test-id 01_how_many_pods --skip-setup

# Skip cleanup (keep infrastructure running)
python benchmarks/executor.py --model sonnet4.5 --test-id 01_how_many_pods --skip-cleanup
```

## Available Agents

| Agent | Description |
|-------|-------------|
| `drdroid` | DrDroid Investigation API (default) |
| `claudecode` | Local Claude Code CLI with read-only kubectl |
| `holmes` | HolmesGPT ToolCallingLLM |
| `openai` | Simple OpenAI completion (no tools) |

### Claude Code Agent

The `claudecode` agent runs prompts through your local Claude Code CLI installation.
It's restricted to read-only kubectl commands for safe investigation.

```bash
# Run with Claude Code agent
python benchmarks/executor.py --model claude-sonnet --agent claudecode --test-id 01_how_many_pods

# With custom model
CLAUDE_MODEL=claude-sonnet-4-20250514 python benchmarks/executor.py --model sonnet4 --agent claudecode --all
```

**Requirements:**
- Claude Code CLI installed and authenticated (`claude` command in PATH)
- kubectl configured with appropriate cluster context

**Restrictions (enforced via system prompt):**
- Only read-only kubectl commands allowed: `get`, `describe`, `logs`, `top`, `explain`, `api-resources`, `cluster-info`
- Write commands forbidden: `apply`, `create`, `delete`, `edit`, `patch`, `exec`, etc.

### Adding a Custom Agent

Edit `benchmarks/agent.py`:

```python
@register_agent("my_agent")
def my_custom_agent(test_case: TestCase) -> AgentResult:
    """My custom agent implementation."""

    # Your agent logic here
    response = call_my_api(test_case.user_prompt)

    return AgentResult(
        output=response,
        tool_calls=["tool1", "tool2"],  # optional
        metadata={"custom": "data"},     # optional
    )
```

## Credentials Configuration

Create `benchmarks/config/credentials.yaml`:

```yaml
# Kubernetes
kubernetes:
  kubeconfig: ~/.kube/config
  context: my-cluster

# Monitoring tools
datadog:
  api_key: your-api-key
  app_key: your-app-key

prometheus:
  url: http://localhost:9090

grafana:
  url: http://localhost:3000
  api_key: your-api-key

# LLM
openai:
  api_key: sk-...

# Judge
judge:
  model: gpt-4.1

# Custom
custom:
  drdroid:
    api_url: http://localhost:8000
    api_key: your-key
```

Environment variables override file values:
- `OPENAI_API_KEY`
- `CLASSIFIER_MODEL`
- `DRDROID_API_URL`
- `DRDROID_API_KEY`
- etc.

## Results Storage

Each test run is saved to `benchmarks/results/` as a JSON file named by model:

```
results/
├── sonnet4.5_01_how_many_pods_20260130_163000.json
├── sonnet4.5_02_what_is_wrong_with_pod_20260130_163100.json
├── gpt5.2_01_how_many_pods_20260130_164000.json
└── ...
```

### Result File Format

```json
{
  "test_id": "01_how_many_pods",
  "agent": "drdroid",
  "model": "sonnet4.5",
  "run_id": "sonnet4.5_01_how_many_pods_20260130_163000",
  "status": "passed",
  "user_prompt": "How many pods are in the app-01 namespace?",
  "expected_output": ["There are 14 pods in the app-01 namespace"],
  "actual_output": "There are 14 pods running in namespace app-01.",
  "score": 1.0,
  "judge_rationale": "The output correctly states 14 pods...",
  "judge_model": "gpt-4.1",
  "setup_time": 45.2,
  "agent_time": 3.5,
  "judge_time": 2.1,
  "cleanup_time": 5.0,
  "total_time": 55.8,
  "tool_calls": ["kubectl_get_pods"],
  "agent_metadata": {
    "investigation_id": "inv-123",
    "tokens": 150
  },
  "started_at": "2026-01-30T16:30:00",
  "completed_at": "2026-01-30T16:30:55"
}
```

## Generating Reports

### Interactive Dashboard (Recommended)

```bash
streamlit run benchmarks/dashboard.py
```

The dashboard provides:

- **Overview**: Total runs, pass rate, cost, tokens summary
- **Model Comparison**: Side-by-side comparison of all models
- **Test Cases**: Analysis by test case with per-model breakdown
- **Use Case x Model Matrix**: Pivot table showing status/metrics for every combination
- **Raw Results**: Detailed view with filtering and drill-down

Features:

- Auto-refresh with "Refresh Data" button
- Filter by model, status, and date range
- Download CSV exports
- View detailed output, rationale, and errors for any run

### CLI Reports

#### Summary Report
```bash
python benchmarks/reporter.py --summary
```

Output:
```
======================================================================
BENCHMARK SUMMARY REPORT
======================================================================

Overall Statistics:
  Total Runs:     50
  Passed:         42 ✅
  Failed:         5 ❌
  Setup Failed:   2 🔧
  Errors:         1 ⚠️
  Pass Rate:      84.0%

Timing:
  Avg Total Time: 45.30s
  Avg Agent Time: 3.20s

Coverage:
  Unique Models:  2
  Unique Tests:   20
  Models:         sonnet4.5, gpt5.2
```

#### Model Comparison
```bash
python benchmarks/reporter.py --compare-models
```

Output:
```
======================================================================
MODEL COMPARISON REPORT
======================================================================

Model                Runs   Pass   Fail     Rate   Avg Time
----------------------------------------------------------------------
sonnet4.5               5      4      1    80.0%     42.50s
gpt5.2                  5      3      2    60.0%     38.20s
======================================================================
```

#### Test Case Report (by Use Case)
```bash
python benchmarks/reporter.py --by-test
```

Output:
```
======================================================================
TEST CASE REPORT (by Use Case)
======================================================================

01_how_many_pods [kubernetes, easy]
  Prompt: How many pods are in the app-01 namespace?...
  Model              Runs  Pass     Rate     Time
  --------------------------------------------------
  sonnet4.5             3     3   100.0%    42.50s
  gpt5.2                2     1    50.0%    38.20s

02_what_is_wrong_with_pod [kubernetes]
  ...
```

#### Test-Specific Report
```bash
python benchmarks/reporter.py --test-id 01_how_many_pods
```

#### Export Reports
```bash
# JSON export
python benchmarks/reporter.py --summary --output report.json

# CSV export
python benchmarks/reporter.py --compare-models --output comparison.csv
python benchmarks/reporter.py --by-test --output tests.csv
```

#### Filter Results
```bash
# Results for specific model
python benchmarks/reporter.py --summary --model sonnet4.5

# Results since a date
python benchmarks/reporter.py --summary --since 2026-01-30

# Failed tests only
python benchmarks/reporter.py --detailed --status failed
```

## Test Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Load credentials from config/credentials.yaml                    │
│ 2. Load test case from fixtures (user_prompt, expected_output)      │
│ 3. Run before_test bash script (setup infrastructure)               │
│ 4. Call agent with test_case.user_prompt (model passed via env)     │
│ 5. LLM Judge evaluates actual vs expected output                    │
│ 6. Run after_test bash script (cleanup)                             │
│ 7. Save result to results/{model}_{test_id}_{timestamp}.json        │
└─────────────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for judge | - |
| `CLASSIFIER_MODEL` | Model for LLM judge | `gpt-4.1` |
| `DRDROID_API_URL` | DrDroid API URL | `http://localhost:8000` |
| `DRDROID_API_KEY` | DrDroid API key | - |
| `DRDROID_MODEL` | Model to use (set by --model flag) | - |

## CLI Reference

### executor.py

```
python benchmarks/executor.py [OPTIONS]

Options:
  --model TEXT           Model to use (REQUIRED). Examples: sonnet4.5, gpt5.2
  --agent TEXT           Agent to use (default: drdroid)
  --test-id TEXT         Test ID(s) to run (repeatable)
  --all                  Run all tests
  --tag TEXT             Filter by tag (repeatable)
  --skip-setup           Skip before_test scripts
  --skip-cleanup         Skip after_test scripts
  --classifier-model     LLM judge model
  --credentials PATH     Path to credentials file
  --list-tests           List available tests
  --list-agents          List registered agents
  -v, --verbose          Verbose output
```

### reporter.py

```
python benchmarks/reporter.py [OPTIONS]

Options:
  --summary              Generate summary report
  --compare-models       Compare model performance
  --by-test              Report grouped by test case (use case)
  --detailed             Show detailed results
  --model TEXT           Filter by model
  --test-id TEXT         Report on specific test
  --status TEXT          Filter by status
  --since TEXT           Filter by date (ISO format)
  --output, -o PATH      Output file (JSON or CSV)
  --results-dir PATH     Results directory
```

### dashboard.py

```bash
# Launch interactive dashboard
streamlit run benchmarks/dashboard.py

# Or with custom port
streamlit run benchmarks/dashboard.py --server.port 8501
```

### agent.py

```bash
# Test an agent directly
python benchmarks/agent.py --agent drdroid --prompt "How many pods?"
```

## Integration with CI/CD

```bash
#!/bin/bash
# Run benchmarks and fail if pass rate < 80%

python benchmarks/executor.py --model sonnet4.5 --all

# Check results
python benchmarks/reporter.py --summary --output results.json

PASS_RATE=$(jq -r '.summary.pass_rate' results.json | tr -d '%')
if (( $(echo "$PASS_RATE < 80" | bc -l) )); then
    echo "Pass rate $PASS_RATE% is below threshold"
    exit 1
fi
```

## Requirements

The dashboard requires Streamlit and Pandas:

```bash
pip install streamlit pandas
```
