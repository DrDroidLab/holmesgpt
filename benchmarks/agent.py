#!/usr/bin/env python3
"""
Agent Implementations

Register your agents here. Each agent must be registered with a unique name.
The executor requires an agent type to be specified - there is no default.

Usage:
    python benchmarks/executor.py --agent drdroid --test-id 01_how_many_pods
    python benchmarks/executor.py --agent holmes --test-id 01_how_many_pods
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class AgentResult:
    """Result returned by an agent."""

    output: str  # The agent's answer (REQUIRED)
    tool_calls: List[str] = field(default_factory=list)  # Tools called
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra data


@dataclass
class TestCase:
    """Test case input to agents."""

    id: str  # e.g., "01_how_many_pods"
    folder: str  # Path to test folder
    user_prompt: str  # The question to answer
    expected_output: List[str]  # What the judge checks for
    before_test: Optional[str] = None  # Setup script
    after_test: Optional[str] = None  # Cleanup script
    tags: List[str] = field(default_factory=list)
    setup_timeout: int = 300


# =============================================================================
# Agent Registry
# =============================================================================

# Type for agent functions
AgentFunction = Callable[[TestCase], AgentResult]

# Registry of all available agents
_AGENT_REGISTRY: Dict[str, AgentFunction] = {}


def register_agent(name: str):
    """Decorator to register an agent."""

    def decorator(func: AgentFunction) -> AgentFunction:
        _AGENT_REGISTRY[name] = func
        return func

    return decorator


def get_agent(name: str) -> AgentFunction:
    """Get an agent by name."""
    if name not in _AGENT_REGISTRY:
        available = ", ".join(sorted(_AGENT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )
    return _AGENT_REGISTRY[name]


def list_agents() -> List[str]:
    """List all registered agent names."""
    return sorted(_AGENT_REGISTRY.keys())


# =============================================================================
# Agent Implementations
# =============================================================================


@register_agent("drdroid")
def drdroid_agent(test_case: TestCase) -> AgentResult:
    """
    DrDroid Investigation Agent via API.

    Environment variables:
        DRDROID_API_URL: API endpoint (default: http://localhost:8000)
        DRDROID_API_KEY: API key for authentication
        DRDROID_MODEL: Model override (optional). Options:
            - "sonnet4.5" - Claude Sonnet 4.5 via Portkey
            - "gpt5.2" - GPT 5.2 via Azure Foundry
            - "codex" - GPT 5.2 Codex via Azure Foundry
    """
    api_url = os.getenv("DRDROID_API_URL", "http://localhost:8000")
    api_key = os.getenv("DRDROID_API_KEY")
    model = os.getenv("DRDROID_MODEL")  # Optional: "sonnet4.5" or "gpt5.2"

    if not api_key:
        raise ValueError("DRDROID_API_KEY environment variable is required")

    url = f"{api_url}/api/external/investigate"

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    payload = {
        "message": test_case.user_prompt,
        "metadata": {
            "test_id": test_case.id,
            "source": "benchmark",
        },
    }

    # Add model override if specified
    if model:
        payload["model"] = model

    print(f"[drdroid] Calling API: {url}")
    print(f"[drdroid] Model: {model or 'default (credits-based)'}")
    print(f"[drdroid] Prompt: {test_case.user_prompt[:100]}...")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        result = response.json()

        output = result.get("response", "")
        investigation_id = result.get("investigation_id")
        total_tokens = result.get("total_tokens", 0)
        model_used = result.get("model")
        credits_used = result.get("credits_used")

        print(f"[drdroid] Investigation ID: {investigation_id}")
        print(f"[drdroid] Model: {model_used}")
        print(f"[drdroid] Total tokens: {total_tokens}")
        print(f"[drdroid] Credits used: {credits_used}")
        print(f"[drdroid] Output: {output[:200]}...")

        return AgentResult(
            output=output,
            metadata={
                "investigation_id": investigation_id,
                "status": result.get("status"),
                "total_tokens": total_tokens,
                "model": model_used,
                "credits_used": credits_used,
            },
        )
    except requests.exceptions.RequestException as e:
        print(f"[drdroid] Error: {str(e)}")
        return AgentResult(
            output=f"Error calling DrDroid API: {str(e)}",
            metadata={"error": str(e)},
        )


@register_agent("holmes")
def holmes_agent(test_case: TestCase) -> AgentResult:
    """
    HolmesGPT Agent using ToolCallingLLM.

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        MODEL: Model to use (default: gpt-4.1)
    """
    from holmes.config import Config
    from holmes.core.llm import DefaultLLM
    from holmes.core.tool_calling_llm import ToolCallingLLM
    from holmes.core.tools import ToolExecutor
    from holmes.core.toolset_manager import ToolsetManager

    model = os.getenv("MODEL", "gpt-4.1")

    print(f"[holmes] Using model: {model}")
    print(f"[holmes] Prompt: {test_case.user_prompt[:100]}...")

    config = Config()
    llm = DefaultLLM(model=model)

    toolset_manager = ToolsetManager(config=config)
    toolsets = toolset_manager.load_builtin_toolsets()
    toolset_manager.add_toolsets(toolsets)
    tool_executor = ToolExecutor(toolset_manager.get_enabled_toolsets())

    ai = ToolCallingLLM(llm=llm, tool_executor=tool_executor, max_steps=20)

    messages = [
        {"role": "system", "content": "You are a Kubernetes troubleshooting assistant."},
        {"role": "user", "content": test_case.user_prompt},
    ]

    result = ai.messages_call(messages=messages)

    tool_calls = []
    if result.tool_calls:
        tool_calls = [tc.description for tc in result.tool_calls]

    print(f"[holmes] Tool calls: {len(tool_calls)}")
    print(f"[holmes] Output: {(result.result or '')[:200]}...")

    return AgentResult(
        output=result.result or "",
        tool_calls=tool_calls,
        metadata={
            "num_llm_calls": result.num_llm_calls,
            "total_tokens": result.total_tokens,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost": result.cost,
            "model": model,
        },
    )


@register_agent("openai")
def openai_agent(test_case: TestCase) -> AgentResult:
    """
    Simple OpenAI chat completion (no tools).

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        MODEL: Model to use (default: gpt-4.1)
    """
    import openai

    model = os.getenv("MODEL", "gpt-4.1")

    print(f"[openai] Using model: {model}")
    print(f"[openai] Prompt: {test_case.user_prompt[:100]}...")

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": test_case.user_prompt}],
    )

    output = response.choices[0].message.content or ""

    print(f"[openai] Output: {output[:200]}...")

    return AgentResult(
        output=output,
        metadata={
            "model": model,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        },
    )


def _fetch_investigation_prompt(investigation_id: str) -> dict:
    """Fetch prompt details from Investigation API.

    Returns dict with:
        - prompt: The investigation prompt
        - context: Additional context (optional)
        - metadata: Any metadata from the API
    """
    api_url = os.getenv("INVESTIGATION_API_URL", "https://api.drdroid.io")
    api_key = os.getenv("INVESTIGATION_API_KEY", "")

    if not api_key:
        raise ValueError(
            "INVESTIGATION_API_KEY environment variable required when using investigation_id"
        )

    url = f"{api_url}/api/investigations/{investigation_id}/prompt"

    print(f"[claudecode] Fetching prompt from Investigation API...")
    print(f"[claudecode]   URL: {url}")

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"[claudecode]   Status: {response.status_code} OK")
        print(f"[claudecode]   Prompt length: {len(data.get('prompt', ''))} chars")

        return {
            "prompt": data.get("prompt", ""),
            "context": data.get("context", ""),
            "metadata": data.get("metadata", {}),
        }
    except requests.exceptions.RequestException as e:
        print(f"[claudecode]   Error fetching prompt: {e}")
        raise ValueError(f"Failed to fetch investigation prompt: {e}")


@register_agent("claudecode")
def claudecode_agent(test_case: TestCase) -> AgentResult:
    """
    Claude Code Agent - runs prompts through local Claude Code CLI.

    This agent invokes Claude Code in non-interactive mode to investigate
    issues using kubectl commands on the current cluster context.

    Environment variables:
        CLAUDE_CODE_PATH: Path to claude CLI (optional, auto-detected)
        CLAUDE_MODEL: Model to use (optional, uses Claude Code default)
        CLAUDE_MCP_CONFIG: Path to MCP servers config JSON file
        INVESTIGATION_ID: Investigation ID to fetch prompt from API (optional)
        INVESTIGATION_API_URL: Base URL for investigation API
        INVESTIGATION_API_KEY: API key for investigation API

    Requirements:
        - Claude Code CLI installed and authenticated (`claude` command available)
        - kubectl configured with appropriate cluster context
        - MCP servers configured (optional, for additional tool access)
    """
    import shutil
    import subprocess
    import time
    from pathlib import Path

    print(f"[claudecode] {'=' * 60}")
    print(f"[claudecode] CLAUDE CODE AGENT STARTING")
    print(f"[claudecode] {'=' * 60}")

    # Check if claude CLI is available
    # First check environment variable, then PATH, then common locations
    claude_path = os.getenv("CLAUDE_CODE_PATH")

    if not claude_path:
        claude_path = shutil.which("claude")

    if not claude_path:
        # Check common installation locations
        common_paths = [
            Path.home() / ".local" / "bin" / "claude",
            Path("/usr/local/bin/claude"),
            Path.home() / "bin" / "claude",
        ]
        for path in common_paths:
            if path.exists() and path.is_file():
                claude_path = str(path)
                break

    if not claude_path:
        raise ValueError(
            "Claude Code CLI not found. Checked PATH and ~/.local/bin/claude. "
            "Set CLAUDE_CODE_PATH environment variable or ensure 'claude' is in PATH. "
            "Install from: https://docs.anthropic.com/en/docs/claude-code"
        )

    model = os.getenv("CLAUDE_MODEL", "")  # Empty means use Claude Code default
    mcp_config = os.getenv("CLAUDE_MCP_CONFIG", "")  # Path to MCP servers config
    investigation_id = os.getenv("INVESTIGATION_ID", "")  # Optional investigation ID

    print(f"[claudecode] Configuration:")
    print(f"[claudecode]   CLI Path: {claude_path}")
    print(f"[claudecode]   Model: {model or 'default'}")
    print(f"[claudecode]   MCP Config: {mcp_config or 'none'}")
    print(f"[claudecode]   Investigation ID: {investigation_id or 'none'}")

    # Determine the prompt to use
    investigation_metadata = {}
    if investigation_id:
        # Fetch prompt from Investigation API
        print(f"[claudecode] {'─' * 60}")
        print(f"[claudecode] FETCHING INVESTIGATION PROMPT")
        inv_data = _fetch_investigation_prompt(investigation_id)
        user_prompt = inv_data["prompt"]
        investigation_metadata = inv_data.get("metadata", {})

        if inv_data.get("context"):
            print(f"[claudecode]   Additional context: {len(inv_data['context'])} chars")
            user_prompt = f"{inv_data['context']}\n\n{user_prompt}"

        print(f"[claudecode]   Final prompt: {user_prompt[:150]}...")
    else:
        # Use prompt from test case
        user_prompt = test_case.user_prompt
        print(f"[claudecode]   Using test case prompt: {user_prompt[:100]}...")

    # Build the prompt with read-only kubectl restrictions
    system_instructions = """You are a Kubernetes troubleshooting assistant.

IMPORTANT RESTRICTIONS:
- You may ONLY use kubectl commands that are READ-ONLY
- ALLOWED kubectl commands: get, describe, logs, top, explain, api-resources, api-versions, cluster-info, config view, config get-contexts
- FORBIDDEN kubectl commands: apply, create, delete, edit, patch, replace, scale, rollout, exec, cp, port-forward, run, set, label, annotate, taint, cordon, uncordon, drain
- If you need to run a forbidden command, explain what you would do instead of running it
- Focus on gathering information and diagnosing issues, not making changes

Investigate the following and provide your findings:"""

    full_prompt = f"{system_instructions}\n\n{user_prompt}"

    # Build claude command
    # Using --print (-p) for non-interactive mode that prints the result
    # Using --verbose to see tool calls and commands
    # Using --output-format json to get structured output with token counts
    cmd = [claude_path, "-p", full_prompt, "--verbose", "--output-format", "json"]

    # Add model override if specified
    if model:
        cmd.extend(["--model", model])

    # Add MCP config if specified
    if mcp_config:
        mcp_config_path = Path(mcp_config)
        # If relative path, resolve from benchmarks directory
        if not mcp_config_path.is_absolute():
            mcp_config_path = Path(__file__).parent / mcp_config
        if mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(mcp_config_path)])
            print(f"[claudecode]   MCP config loaded: {mcp_config_path}")

            # Log MCP servers being used
            try:
                with open(mcp_config_path, "r") as f:
                    mcp_data = json.load(f)
                    servers = mcp_data.get("mcpServers", {})
                    print(f"[claudecode]   MCP servers: {', '.join(servers.keys())}")
            except Exception as e:
                print(f"[claudecode]   Warning: Could not parse MCP config: {e}")
        else:
            print(f"[claudecode]   Warning: MCP config not found: {mcp_config_path}")

    # Add dangerously skip permissions to avoid interactive prompts
    # This is safe because we're restricting to read-only in the prompt
    cmd.append("--dangerously-skip-permissions")

    print(f"[claudecode] Running Claude Code...")
    print(f"[claudecode] Command: {' '.join(cmd[:3])}... (prompt truncated)")
    start_time = time.time()

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            cwd=test_case.folder,
            bufsize=1,  # Line buffered
        )

        output_lines = []
        tool_calls = []
        token_info = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
        }

        print(f"[claudecode] {'─' * 60}")

        # Stream output line by line
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line_stripped = line.rstrip()
                print(f"  │ {line_stripped}")
                output_lines.append(line)

                # Try to detect tool calls from output
                # Claude Code typically shows tool usage like "Running: kubectl ..." or "Tool: ..."
                line_lower = line_stripped.lower()
                if any(indicator in line_lower for indicator in [
                    "running:", "executing:", "tool:", "$ kubectl", "bash:",
                    "> kubectl", "command:", "running command"
                ]):
                    tool_calls.append(line_stripped)

                # Try to extract token information from output
                # Look for patterns like "tokens: 1234", "total_tokens", "input/output tokens"

                # Pattern: "X tokens" or "tokens: X" or "total_tokens: X"
                token_patterns = [
                    r'total[_\s]?tokens[:\s]+(\d+)',
                    r'input[_\s]?tokens[:\s]+(\d+)',
                    r'output[_\s]?tokens[:\s]+(\d+)',
                    r'prompt[_\s]?tokens[:\s]+(\d+)',
                    r'completion[_\s]?tokens[:\s]+(\d+)',
                    r'(\d+)\s+tokens?\s+used',
                    r'tokens?\s+used[:\s]+(\d+)',
                    r'cost[:\s]+\$?([\d.]+)',
                ]

                for pattern in token_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        value = match.group(1)
                        if 'total' in pattern:
                            token_info["total_tokens"] = int(value)
                        elif 'input' in pattern or 'prompt' in pattern:
                            token_info["prompt_tokens"] = int(value)
                        elif 'output' in pattern or 'completion' in pattern:
                            token_info["completion_tokens"] = int(value)
                        elif 'cost' in pattern:
                            token_info["cost"] = float(value)

            # Check timeout
            if time.time() - start_time > 300:
                process.kill()
                process.wait()
                print(f"[claudecode] {'─' * 60}")
                print(f"[claudecode] Timeout after 300s")
                return AgentResult(
                    output="Claude Code timed out after 300 seconds",
                    metadata={"error": "timeout", "timeout": 300},
                )

        process.wait()
        elapsed = time.time() - start_time

        print(f"[claudecode] {'─' * 60}")
        print(f"[claudecode] Completed in {elapsed:.2f}s")
        print(f"[claudecode] Tool calls detected: {len(tool_calls)}")

        if tool_calls:
            print(f"[claudecode] Tools used:")
            for tc in tool_calls[:10]:  # Show first 10
                print(f"  • {tc[:100]}")
            if len(tool_calls) > 10:
                print(f"  ... and {len(tool_calls) - 10} more")

        raw_output = "".join(output_lines).strip()

        # Try to parse JSON output for structured data including tokens
        output = raw_output
        try:
            # Look for JSON in the output (might be at the end or the whole output)
            json_match = re.search(r'\{[^{}]*"result"[^{}]*\}|\{[^{}]*"output"[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                # Extract result/output from JSON
                output = json_data.get("result") or json_data.get("output") or raw_output
                # Extract token info from JSON
                if "usage" in json_data:
                    usage = json_data["usage"]
                    token_info["total_tokens"] = usage.get("total_tokens", 0)
                    token_info["prompt_tokens"] = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                    token_info["completion_tokens"] = usage.get("completion_tokens") or usage.get("output_tokens", 0)
                if "cost" in json_data:
                    token_info["cost"] = json_data["cost"]
                if "total_tokens" in json_data:
                    token_info["total_tokens"] = json_data["total_tokens"]
        except (json.JSONDecodeError, AttributeError):
            pass  # Not JSON or parsing failed, use raw output

        # Print token info
        if token_info["total_tokens"] > 0:
            print(f"[claudecode] Tokens: {token_info['total_tokens']} total "
                  f"({token_info['prompt_tokens']} prompt, {token_info['completion_tokens']} completion)")
        if token_info["cost"] > 0:
            print(f"[claudecode] Cost: ${token_info['cost']:.4f}")

        print(f"[claudecode] {'=' * 60}")
        print(f"[claudecode] CLAUDE CODE AGENT COMPLETED")
        print(f"[claudecode] {'=' * 60}")

        if process.returncode != 0:
            print(f"[claudecode] Error: exit code {process.returncode}")
            return AgentResult(
                output=f"Claude Code error (exit {process.returncode}): {output}",
                tool_calls=tool_calls,
                metadata={
                    "error": f"exit code {process.returncode}",
                    "return_code": process.returncode,
                    "elapsed_time": elapsed,
                    "total_tokens": token_info["total_tokens"],
                    "prompt_tokens": token_info["prompt_tokens"],
                    "completion_tokens": token_info["completion_tokens"],
                    "cost": token_info["cost"],
                    "investigation_id": investigation_id or None,
                    "investigation_metadata": investigation_metadata,
                    "mcp_config": mcp_config or None,
                },
            )

        return AgentResult(
            output=output,
            tool_calls=tool_calls,
            metadata={
                "model": model or "claude-code-default",
                "elapsed_time": elapsed,
                "return_code": process.returncode,
                "num_tool_calls": len(tool_calls),
                "total_tokens": token_info["total_tokens"],
                "prompt_tokens": token_info["prompt_tokens"],
                "completion_tokens": token_info["completion_tokens"],
                "cost": token_info["cost"],
                "investigation_id": investigation_id or None,
                "investigation_metadata": investigation_metadata,
                "mcp_config": mcp_config or None,
            },
        )

    except subprocess.TimeoutExpired:
        print(f"[claudecode] Timeout after 300s")
        return AgentResult(
            output="Claude Code timed out after 300 seconds",
            metadata={"error": "timeout", "timeout": 300},
        )
    except Exception as e:
        print(f"[claudecode] Exception: {str(e)}")
        return AgentResult(
            output=f"Claude Code exception: {str(e)}",
            metadata={"error": str(e)},
        )


# =============================================================================
# Add your custom agents below
# =============================================================================


# @register_agent("my_agent")
# def my_custom_agent(test_case: TestCase) -> AgentResult:
#     """Your custom agent implementation."""
#     # Your code here
#     return AgentResult(output="response")


# =============================================================================
# CLI for testing agents directly
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test an agent directly")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list_agents(),
        help="Agent to test",
    )
    parser.add_argument(
        "--prompt",
        default="How many pods are in the app-01 namespace?",
        help="Test prompt",
    )

    args = parser.parse_args()

    print(f"Testing agent: {args.agent}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    test = TestCase(
        id="cli-test",
        folder=".",
        user_prompt=args.prompt,
        expected_output=["test"],
    )

    try:
        agent_fn = get_agent(args.agent)
        result = agent_fn(test)
        print("\n" + "=" * 50)
        print("RESULT:")
        print(f"Output: {result.output}")
        print(f"Tool calls: {result.tool_calls}")
        print(f"Metadata: {result.metadata}")
    except Exception as e:
        print(f"Error: {e}")
