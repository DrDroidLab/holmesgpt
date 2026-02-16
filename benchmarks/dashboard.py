#!/usr/bin/env python3
"""
Benchmark Results Dashboard

A Streamlit app to visualize benchmark test results with use-case x model level reporting.

Usage:
    streamlit run benchmarks/dashboard.py
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"


# =============================================================================
# Data Loading
# =============================================================================


@st.cache_data(ttl=30)  # Cache for 30 seconds, then refresh
def load_all_results() -> List[Dict[str, Any]]:
    """Load all results from JSON files."""
    results = []

    if not RESULTS_DIR.exists():
        return results

    for filepath in RESULTS_DIR.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                data["_filepath"] = str(filepath)
                results.append(data)
        except Exception as e:
            st.warning(f"Failed to load {filepath.name}: {e}")

    return sorted(results, key=lambda r: r.get("started_at", ""), reverse=True)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a result."""
    metadata = result.get("agent_metadata", {})

    # Build investigation URL if investigation_id exists
    investigation_id = metadata.get("investigation_id")
    investigation_url = None
    if investigation_id:
        investigation_url = f"https://aiops.drdroid.io/investigations/{investigation_id}"

    return {
        "test_id": result.get("test_id", "unknown"),
        "model": result.get("model", "unknown"),
        "agent": result.get("agent", "unknown"),
        "status": result.get("status", "unknown"),
        "score": result.get("score"),
        "total_time": result.get("total_time", 0),
        "setup_time": result.get("setup_time", 0),
        "agent_time": result.get("agent_time", 0),
        "judge_time": result.get("judge_time", 0),
        "cleanup_time": result.get("cleanup_time", 0),
        "cost": metadata.get("cost", 0) or 0,
        "total_tokens": metadata.get("total_tokens", 0) or 0,
        "prompt_tokens": metadata.get("prompt_tokens", 0) or 0,
        "completion_tokens": metadata.get("completion_tokens", 0) or 0,
        "num_llm_calls": metadata.get("num_llm_calls", 0) or 0,
        "tags": result.get("tags", []),
        "started_at": result.get("started_at", ""),
        "run_id": result.get("run_id", ""),
        "user_prompt": result.get("user_prompt", ""),
        "actual_output": result.get("actual_output", ""),
        "expected_output": result.get("expected_output", []),
        "judge_rationale": result.get("judge_rationale", ""),
        "error_message": result.get("error_message", ""),
        "investigation_id": investigation_id,
        "investigation_url": investigation_url,
    }


def build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a pandas DataFrame from results."""
    if not results:
        return pd.DataFrame()

    metrics = [extract_metrics(r) for r in results]
    df = pd.DataFrame(metrics)

    # Convert started_at to datetime
    if "started_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

    return df


# =============================================================================
# Aggregation Functions
# =============================================================================


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("model").agg(
        total_runs=("test_id", "count"),
        passed=("status", lambda x: (x == "passed").sum()),
        failed=("status", lambda x: (x == "failed").sum()),
        setup_failed=("status", lambda x: (x == "setup_failed").sum()),
        errors=("status", lambda x: (x == "error").sum()),
        avg_total_time=("total_time", "mean"),
        avg_agent_time=("agent_time", "mean"),
        total_cost=("cost", "sum"),
        avg_cost=("cost", "mean"),
        total_tokens=("total_tokens", "sum"),
        avg_tokens=("total_tokens", "mean"),
        unique_tests=("test_id", "nunique"),
    ).reset_index()

    agg["pass_rate"] = (agg["passed"] / agg["total_runs"] * 100).round(1)

    return agg.sort_values("pass_rate", ascending=False)


def aggregate_by_test(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by test case."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("test_id").agg(
        total_runs=("model", "count"),
        passed=("status", lambda x: (x == "passed").sum()),
        failed=("status", lambda x: (x == "failed").sum()),
        avg_total_time=("total_time", "mean"),
        avg_agent_time=("agent_time", "mean"),
        total_cost=("cost", "sum"),
        avg_cost=("cost", "mean"),
        total_tokens=("total_tokens", "sum"),
        models_tested=("model", "nunique"),
    ).reset_index()

    agg["pass_rate"] = (agg["passed"] / agg["total_runs"] * 100).round(1)

    return agg.sort_values("test_id")


def create_pivot_table(df: pd.DataFrame, value_col: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Create a pivot table with test_id as rows and model as columns."""
    if df.empty:
        return pd.DataFrame()

    pivot = pd.pivot_table(
        df,
        values=value_col,
        index="test_id",
        columns="model",
        aggfunc=aggfunc,
        fill_value=0,
    )

    return pivot


def create_status_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table showing pass/fail status per test x model."""
    if df.empty:
        return pd.DataFrame()

    # Get latest result for each test_id x model combination
    latest = df.sort_values("started_at", ascending=False).drop_duplicates(
        subset=["test_id", "model"]
    )

    # Create status mapping with icons
    status_map = {
        "passed": "✅",
        "failed": "❌",
        "setup_failed": "🔧",
        "error": "⚠️",
    }
    latest["status_icon"] = latest["status"].map(status_map).fillna("?")

    pivot = pd.pivot_table(
        latest,
        values="status_icon",
        index="test_id",
        columns="model",
        aggfunc="first",
        fill_value="-",
    )

    return pivot


def get_latest_results_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Get latest results organized as test_id -> model -> result_data.

    Returns a nested dict for building an interactive status matrix.
    """
    if df.empty:
        return {}

    # Get latest result for each test_id x model combination
    latest = df.sort_values("started_at", ascending=False).drop_duplicates(
        subset=["test_id", "model"]
    )

    # Status mapping with icons
    status_map = {
        "passed": "✅",
        "failed": "❌",
        "setup_failed": "🔧",
        "error": "⚠️",
    }

    # Build nested dict
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in latest.iterrows():
        test_id = row["test_id"]
        model = row["model"]

        if test_id not in matrix:
            matrix[test_id] = {}

        matrix[test_id][model] = {
            "status": row["status"],
            "status_icon": status_map.get(row["status"], "?"),
            "score": row["score"],
            "judge_rationale": row["judge_rationale"] or "No rationale available",
            "agent_time": row["agent_time"],
            "cost": row["cost"],
            "total_tokens": row["total_tokens"],
            "error_message": row["error_message"],
            "run_id": row["run_id"],
            "investigation_url": row.get("investigation_url"),
        }

    return matrix


# =============================================================================
# Streamlit App
# =============================================================================


def main():
    st.set_page_config(
        page_title="Benchmark Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Benchmark Results Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Load data
        results = load_all_results()

        if not results:
            st.warning("No results found")
            st.info(f"Results directory: {RESULTS_DIR}")
            return

        df = build_dataframe(results)

        st.metric("Total Results", len(df))
        st.metric("Unique Models", df["model"].nunique())
        st.metric("Unique Tests", df["test_id"].nunique())

        st.markdown("---")

        # Filters
        st.subheader("Filters")

        models = ["All"] + sorted(df["model"].unique().tolist())
        selected_model = st.selectbox("Model", models)

        statuses = ["All"] + sorted(df["status"].unique().tolist())
        selected_status = st.selectbox("Status", statuses)

        # Date filter
        if not df["started_at"].isna().all():
            min_date = df["started_at"].min().date()
            max_date = df["started_at"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None

    # Apply filters
    filtered_df = df.copy()

    if selected_model != "All":
        filtered_df = filtered_df[filtered_df["model"] == selected_model]

    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["status"] == selected_status]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["started_at"].dt.date >= start_date)
            & (filtered_df["started_at"].dt.date <= end_date)
        ]

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🤖 Model Comparison",
        "📋 Test Cases",
        "🔀 Use Case x Model",
        "📄 Raw Results",
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Overview")

        col1, col2, col3, col4 = st.columns(4)

        total = len(filtered_df)
        passed = (filtered_df["status"] == "passed").sum()
        failed = (filtered_df["status"] == "failed").sum()
        pass_rate = (passed / total * 100) if total > 0 else 0

        col1.metric("Total Runs", total)
        col2.metric("Passed", f"{passed} ✅")
        col3.metric("Failed", f"{failed} ❌")
        col4.metric("Pass Rate", f"{pass_rate:.1f}%")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Avg Time", f"{filtered_df['total_time'].mean():.1f}s")
        col6.metric("Total Cost", f"${filtered_df['cost'].sum():.4f}")
        col7.metric("Total Tokens", f"{filtered_df['total_tokens'].sum():,}")
        col8.metric("Avg Tokens/Run", f"{filtered_df['total_tokens'].mean():,.0f}")

        st.markdown("---")

        # Status distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Status Distribution")
            status_counts = filtered_df["status"].value_counts()
            st.bar_chart(status_counts)

        with col2:
            st.subheader("Runs by Model")
            model_counts = filtered_df["model"].value_counts()
            st.bar_chart(model_counts)

    # Tab 2: Model Comparison
    with tab2:
        st.header("Model Comparison")

        model_agg = aggregate_by_model(filtered_df)

        if not model_agg.empty:
            # Summary table
            st.subheader("Summary by Model")
            display_cols = [
                "model", "total_runs", "passed", "failed", "pass_rate",
                "avg_agent_time", "total_cost", "avg_cost", "total_tokens", "avg_tokens",
            ]
            display_df = model_agg[display_cols].copy()
            display_df.columns = [
                "Model", "Runs", "Passed", "Failed", "Pass Rate %",
                "Avg Agent Time (s)", "Total Cost ($)", "Avg Cost ($)", "Total Tokens", "Avg Tokens",
            ]

            # Format numeric columns
            display_df["Avg Agent Time (s)"] = display_df["Avg Agent Time (s)"].round(2)
            display_df["Total Cost ($)"] = display_df["Total Cost ($)"].round(4)
            display_df["Avg Cost ($)"] = display_df["Avg Cost ($)"].round(4)
            display_df["Avg Tokens"] = display_df["Avg Tokens"].round(0).astype(int)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Pass Rate by Model")
                chart_data = model_agg.set_index("model")["pass_rate"]
                st.bar_chart(chart_data)

            with col2:
                st.subheader("Avg Cost by Model")
                chart_data = model_agg.set_index("model")["avg_cost"]
                st.bar_chart(chart_data)

    # Tab 3: Test Cases
    with tab3:
        st.header("Test Case Analysis")

        test_agg = aggregate_by_test(filtered_df)

        if not test_agg.empty:
            st.subheader("Summary by Test Case")
            display_cols = [
                "test_id", "total_runs", "passed", "failed", "pass_rate",
                "models_tested", "avg_agent_time", "avg_cost",
            ]
            display_df = test_agg[display_cols].copy()
            display_df.columns = [
                "Test ID", "Runs", "Passed", "Failed", "Pass Rate %",
                "Models", "Avg Time (s)", "Avg Cost ($)",
            ]
            display_df["Avg Time (s)"] = display_df["Avg Time (s)"].round(2)
            display_df["Avg Cost ($)"] = display_df["Avg Cost ($)"].round(4)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Failing tests
            failing_tests = test_agg[test_agg["pass_rate"] < 100].sort_values("pass_rate")
            if not failing_tests.empty:
                st.subheader("Tests with Failures")
                st.dataframe(
                    failing_tests[["test_id", "total_runs", "passed", "failed", "pass_rate"]],
                    use_container_width=True,
                    hide_index=True,
                )

    # Tab 4: Use Case x Model Matrix
    with tab4:
        st.header("Use Case x Model Matrix")

        st.subheader("Status Matrix (Latest Run)")
        st.caption("Click the ℹ️ icon to view judge rationale and details")

        # Get the matrix data for interactive display
        results_matrix = get_latest_results_matrix(filtered_df)
        models = sorted(filtered_df["model"].unique().tolist())
        test_ids = sorted(results_matrix.keys())

        if results_matrix and models:
            # Build header row
            header_cols = st.columns([2] + [1] * len(models))
            header_cols[0].markdown("**Test Case**")
            for i, model in enumerate(models):
                header_cols[i + 1].markdown(f"**{model}**")

            # Build data rows
            for test_id in test_ids:
                row_cols = st.columns([2] + [1] * len(models))
                row_cols[0].write(test_id)

                for i, model in enumerate(models):
                    result_data = results_matrix.get(test_id, {}).get(model)

                    if result_data:
                        status_icon = result_data["status_icon"]

                        # Create a container with status icon and info popover
                        with row_cols[i + 1]:
                            col_status, col_info = st.columns([1, 1])
                            col_status.write(status_icon)

                            with col_info.popover("ℹ️"):
                                st.markdown(f"**Test:** {test_id}")
                                st.markdown(f"**Model:** {model}")
                                st.markdown(f"**Status:** {result_data['status']} {status_icon}")
                                st.markdown(f"**Score:** {result_data['score']}")

                                st.markdown("---")
                                st.markdown("**Timing & Cost:**")
                                st.write(f"Agent Time: {result_data['agent_time']:.2f}s")
                                st.write(f"Cost: ${result_data['cost']:.4f}")
                                st.write(f"Tokens: {result_data['total_tokens']:,}")

                                st.markdown("---")
                                st.markdown("**Judge Rationale:**")
                                st.write(result_data["judge_rationale"])

                                if result_data.get("error_message"):
                                    st.markdown("---")
                                    st.error(f"Error: {result_data['error_message']}")

                                if result_data.get("investigation_url"):
                                    st.markdown("---")
                                    st.markdown(
                                        f"[Open Investigation]({result_data['investigation_url']})"
                                    )
                    else:
                        row_cols[i + 1].write("-")

        st.markdown("---")

        # Metric selection for pivot
        metric = st.selectbox(
            "Select Metric for Heatmap",
            ["pass_rate", "agent_time", "cost", "total_tokens"],
        )

        if metric == "pass_rate":
            # Calculate pass rate per test x model
            pass_df = filtered_df.copy()
            pass_df["is_passed"] = (pass_df["status"] == "passed").astype(int)
            pivot = create_pivot_table(pass_df, "is_passed", "mean") * 100
            st.subheader("Pass Rate % (Test x Model)")
        elif metric == "agent_time":
            pivot = create_pivot_table(filtered_df, "agent_time", "mean")
            st.subheader("Avg Agent Time in seconds (Test x Model)")
        elif metric == "cost":
            pivot = create_pivot_table(filtered_df, "cost", "mean")
            st.subheader("Avg Cost in $ (Test x Model)")
        else:
            pivot = create_pivot_table(filtered_df, "total_tokens", "mean")
            st.subheader("Avg Tokens (Test x Model)")

        if not pivot.empty:
            # Round values
            pivot = pivot.round(2 if metric in ["agent_time", "cost"] else 1)
            st.dataframe(pivot, use_container_width=True)

            # Download button
            csv = pivot.to_csv()
            st.download_button(
                "Download as CSV",
                csv,
                f"benchmark_{metric}_matrix.csv",
                "text/csv",
            )

    # Tab 5: Raw Results
    with tab5:
        st.header("Raw Results")

        # Sort options
        sort_col = st.selectbox(
            "Sort by",
            ["started_at", "test_id", "model", "status", "total_time", "cost"],
        )
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        sorted_df = filtered_df.sort_values(
            sort_col, ascending=(sort_order == "Ascending")
        )

        # Create display dataframe with investigation links
        display_df = sorted_df[["test_id", "model", "status", "score", "agent_time", "cost", "total_tokens", "started_at", "investigation_url"]].copy()

        display_df.columns = [
            "Test ID", "Model", "Status", "Score", "Agent Time (s)",
            "Cost ($)", "Tokens", "Started At", "Investigation",
        ]

        # Use LinkColumn for clickable investigation links
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Investigation": st.column_config.LinkColumn(
                    "Investigation",
                    help="Open investigation in DrDroid (opens in new tab)",
                    display_text="Open",
                ),
            },
        )

        st.markdown("---")

        # Detailed view for selected result
        st.subheader("Result Details")

        result_options = sorted_df["run_id"].tolist()
        if result_options:
            selected_run = st.selectbox("Select Run", result_options)

            if selected_run:
                result_row = sorted_df[sorted_df["run_id"] == selected_run].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Test Info**")
                    st.write(f"**Test ID:** {result_row['test_id']}")
                    st.write(f"**Model:** {result_row['model']}")
                    st.write(f"**Status:** {result_row['status']}")
                    st.write(f"**Score:** {result_row['score']}")

                    # Investigation link
                    if result_row.get("investigation_url"):
                        st.markdown(
                            f"**Investigation:** [Open in DrDroid]({result_row['investigation_url']})"
                        )

                    st.markdown("**Timing**")
                    st.write(f"Setup: {result_row['setup_time']:.2f}s")
                    st.write(f"Agent: {result_row['agent_time']:.2f}s")
                    st.write(f"Judge: {result_row['judge_time']:.2f}s")
                    st.write(f"Total: {result_row['total_time']:.2f}s")

                with col2:
                    st.markdown("**Costs & Tokens**")
                    st.write(f"Cost: ${result_row['cost']:.4f}")
                    st.write(f"Total Tokens: {result_row['total_tokens']:,}")
                    st.write(f"Prompt Tokens: {result_row['prompt_tokens']:,}")
                    st.write(f"Completion Tokens: {result_row['completion_tokens']:,}")

                st.markdown("**User Prompt**")
                st.code(result_row["user_prompt"], language=None)

                st.markdown("**Expected Output**")
                expected = result_row["expected_output"]
                if isinstance(expected, list):
                    for exp in expected:
                        st.write(f"- {exp}")
                else:
                    st.write(expected)

                st.markdown("**Actual Output**")
                st.code(result_row["actual_output"] or "N/A", language=None)

                if result_row["judge_rationale"]:
                    st.markdown("**Judge Rationale**")
                    st.write(result_row["judge_rationale"])

                if result_row["error_message"]:
                    st.markdown("**Error**")
                    st.error(result_row["error_message"])


if __name__ == "__main__":
    main()
