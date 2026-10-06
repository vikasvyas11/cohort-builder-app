# flows/p_compare_export.py
# Step 6 (compare two runs) and Step 7 (export the cohort).
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.metrics_engine import compute_inter_metrics
from flows.run_results import render_run_results
from modules.splink_runner import MAX_CANDIDATE_PAIRS
from utils.helpers import (
    _metric_cards, cached_rule_patterns, render_assistant_link, render_report_download, render_waterfall_section,
    run_and_evaluate,
)
from utils.nav import _back_button, _go_to


def _mean_probability(metrics: dict):
    stats = metrics["match_prob_stats"]
    return stats["mean_match_prob"].iloc[0] if not stats.empty else None


def _rule_toggles(run1: dict) -> tuple[dict, dict]:
    """Blocking toggles for Run 2 (starting from Run 1's) and any combined rules to add."""
    fields = st.session_state["selected_fields"]
    previous = st.session_state.get("run2_blocking_toggles") or run1["run_config"]["blocking_toggles"]
    columns = st.columns(3)
    toggles = {f: columns[i % 3].toggle(f, value=previous.get(f, False), key=f"r2_{f}")
               for i, f in enumerate(fields)}
    st.session_state["run2_blocking_toggles"] = toggles

    composites = st.session_state.setdefault("r2_composite_rules", {})
    with st.expander("Combine fields into one rule (optional)"):
        st.caption("A pair is a candidate only if it agrees on every field of the rule.")
        chosen = st.multiselect("Fields", fields, key="r2_combo")
        if st.button("Add rule", key="r2_add_combo", disabled=len(chosen) < 2):
            composites["+".join(chosen)] = True
        for key in list(composites):
            left, right = st.columns([4, 1])
            left.code(" AND ".join(f'l."{p}" = r."{p}"' for p in key.split("+")))
            if right.button("Remove", key=f"r2_rm_{key}"):
                del composites[key]
                st.rerun()
    return toggles, composites


def _live_waterfall(run1: dict, toggles: dict, composites: dict) -> None:
    """Cascading waterfall of candidate pairs per rule, redrawn on every toggle or new combined rule.

    Pair counts come from the data itself (not from Run 1's predictions), so it also shows rules
    Run 1 never used. Run 1's rules are the left chart; the current Run 2 selection is the right.
    """
    st.subheader("Effect of your changes (live)")
    config = run1["run_config"]
    live = {**toggles, **{rule: True for rule in composites}}
    baseline = {rule: bool(on) for rule, on in config["blocking_toggles"].items()}
    rules = list(dict.fromkeys([*live, *baseline]))
    operation_mode = config.get("operation_mode") or st.session_state["operation_mode"]
    patterns, skipped = cached_rule_patterns(
        st.session_state["dataset_a"], st.session_state.get("dataset_b"), operation_mode, tuple(rules))
    if patterns.empty:
        st.info("None of these rules can be counted on the loaded data.")
        return
    for rule, size in skipped:
        reason = (f"would add about {size:,} pairs, over the {MAX_CANDIDATE_PAIRS:,}-pair limit"
                  if size is not None else "uses a column that is not in the data")
        st.warning(f"Rule **{rule}** is left out of the chart: it {reason}.")
    render_waterfall_section(
        patterns, {rule: live.get(rule, False) for rule in rules}, config.get("blocking_mode", "OR"),
        key_prefix="cmp_waterfall", titles=("Run 1 rules", "Run 2 selection (live)"), baseline_toggles=baseline)


def _side_by_side(run1: dict, run2: dict) -> None:
    m1, m2 = st.session_state["run1_metrics"], st.session_state["run2_metrics"]
    p1, p2 = _mean_probability(m1), _mean_probability(m2)
    c1, c2, c3 = st.columns(3)
    c1.metric("Edges", f"{m2['n_edges']:,}", delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
    c2.metric("Clusters", f"{m2['n_clusters']:,}", delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
    if p1 is not None and p2 is not None:
        c3.metric("Mean match probability", f"{p2:.4f}", delta=f"{p2 - p1:+.4f}")

    cm1, cm2 = st.session_state.get("run1_cm") or {}, st.session_state.get("run2_cm") or {}
    if cm1.get("precision") is not None and cm2.get("precision") is not None:
        st.write("**Accuracy against ground truth**")
        rows = [{"Measure": label, "Run 1": cm1[key], "Run 2": cm2[key], "Change": round(cm2[key] - cm1[key], 4)}
                for label, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    inter = compute_inter_metrics(run1["df_predict"], run2["df_predict"], run1["df_cluster"], run2["df_cluster"])
    edges = inter["edge_diff"].set_index("category")["n"].to_dict()
    st.write("**What changed between the runs**")
    st.dataframe(pd.DataFrame([
        {"Measure": "Edges in both runs", "Count": edges.get("shared", 0)},
        {"Measure": "Edges only in Run 2", "Count": edges.get("added", 0)},
        {"Measure": "Edges only in Run 1", "Count": edges.get("removed", 0)},
        {"Measure": "Clusters identical in both runs", "Count": inter["n_exact_matching_clusters"]},
        {"Measure": "Cluster pairs that partly overlap", "Count": inter["n_partial_matching_clusters"]},
    ]), hide_index=True, width="stretch")

    for key, x, y, title in (("prob_dist", "prob_bin", "n_edges", "Match probability"),
                             ("cluster_sizes", "n_nodes", "n_clusters", "Cluster size")):
        one, two = inter[f"{key}_run1"].assign(run="Run 1"), inter[f"{key}_run2"].assign(run="Run 2")
        if not one.empty and not two.empty:
            fig = px.bar(pd.concat([one, two]), x=x, y=y, color="run", barmode="group", template="simple_white",
                         title=f"{title}: Run 1 vs Run 2", color_discrete_sequence=["#1E6EC4", "#E55C30"])
            st.plotly_chart(fig.update_layout(height=320), width="stretch", key=f"cmp_{key}")


def page_comparison():
    _back_button()
    st.title("Step 6: Compare Runs")
    run1 = st.session_state.get("run1_results")
    if run1 is None:
        st.warning("No Run 1 results. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write("Change the blocking rules and run again to see what the change does to the links. "
             "Everything else stays as it was in Run 1.")
    m1 = st.session_state["run1_metrics"]
    _metric_cards([("Run 1: edges", f"{m1['n_edges']:,}"), ("Run 1: clusters", f"{m1['n_clusters']:,}")])

    st.subheader("Blocking rules for Run 2")
    toggles, composites = _rule_toggles(run1)
    _live_waterfall(run1, toggles, composites)

    if not any(toggles.values()) and not composites:
        st.error("Switch on at least one blocking rule (or add a combined rule).")
    elif st.button("Run 2", type="primary"):
        with st.spinner("Running..."):
            bundle = run_and_evaluate(
                st.session_state["dataset_a"], st.session_state.get("dataset_b"),
                st.session_state["selected_fields"], toggles, st.session_state["operation_mode"],
                st.session_state["linkage_type"], st.session_state.get("hyperparams", {}), dict(composites))
        if bundle:
            st.session_state.update({
                "run2_results": bundle["results"], "run2_metrics": bundle["metrics"], "run2_cm": bundle["cm"],
                "run2_threshold_curve": bundle["threshold_curve"], "run2_curve": bundle["curve"],
                "run2_coverage": bundle["coverage"], "run2_explorer_toggles": bundle["blocking_toggles"]})

    run2 = st.session_state.get("run2_results")
    if run2 is not None:
        st.divider()
        st.header("Run 2 results")
        render_run_results("run2")
        st.divider()
        st.header("Run 1 vs Run 2")
        _side_by_side(run1, run2)
        st.divider()
        render_assistant_link("run2")
        st.divider()
        st.subheader("Reports")
        left, right = st.columns(2)
        with left:
            render_report_download("Run 1", "run1")
        with right:
            render_report_download("Run 2", "run2")

    st.divider()
    if st.button("Continue to export", type="primary"):
        _go_to(6)


def page_export():
    _back_button()
    st.title("Step 7: Export Cohort")
    if st.session_state.get("run1_results") is None:
        st.warning("No analysis results available. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write("Download the cohort as a CSV: every input column plus `cluster_id`. Records sharing a "
             "`cluster_id` are predicted to be the same real-world entity.")
    runs = ["Run 1"] + (["Run 2"] if st.session_state.get("run2_results") is not None else [])
    chosen_run = st.radio("Export cluster assignments from:", runs, horizontal=True)
    chosen = st.session_state["run1_results" if chosen_run == "Run 1" else "run2_results"]

    dataset_a, dataset_b = st.session_state["dataset_a"], st.session_state.get("dataset_b")
    records = (dataset_a if chosen["run_config"]["operation_mode"] == "dedupe"
               else pd.concat([dataset_a, dataset_b], ignore_index=True))
    clusters = chosen["df_cluster"]
    keys = ["unique_id", "source_dataset"] if "source_dataset" in clusters.columns else ["unique_id"]
    cohort = records.merge(clusters[keys + ["cluster_id"]], on=keys, how="left")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(cohort):,}")
    c2.metric("Distinct cluster IDs", f"{cohort['cluster_id'].nunique():,}")
    c3.metric("Records with a cluster", f"{cohort['cluster_id'].notna().sum():,}")
    st.subheader("Preview (first 50 rows by cluster_id)")
    st.dataframe(cohort.sort_values("cluster_id").head(50), width="stretch")
    st.download_button(f"Download cohort CSV ({chosen_run})", cohort.to_csv(index=False).encode("utf-8"),
                       file_name=f"cohort_{chosen_run.lower().replace(' ', '_')}.csv", mime="text/csv")
