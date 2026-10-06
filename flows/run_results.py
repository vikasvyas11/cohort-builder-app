# flows/run_results.py
# The tabbed results of one linkage run. Run 1 (analysis page) and Run 2 (comparison page)
# both render through this single function, so they always show the same analysis.


import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import streamlit as st
import streamlit.components.v1 as components

from modules.splink_runner import (
    filter_predict_by_active_rules, recluster_filtered,
)
from modules.report import venn_figure
from utils.helpers import (
    _metric_cards, _plotly_bar, render_demographic_breakdowns,
    render_demographic_comparison, compute_demographic_snapshot, render_match_quality_section, split_linked_unlinked,
    render_waterfall_section, render_threshold_curve,
)


def render_run_results(slot: str) -> None:
    """Summary cards and the tabs (edges, clusters, demographics, blocking explorer, cluster
    studio, confusion matrix, raw data) for the run stored under ``slot`` ("run1" or "run2")."""
    tk, ck, thk = f"{slot}_explorer_toggles", f"{slot}_coverage", f"{slot}_explorer_threshold"
    if st.session_state.get(f"{slot}_results") is None:
        return

    results = st.session_state[f"{slot}_results"]
    metrics = st.session_state.get(f"{slot}_metrics", {})

    if results.get("n_edges", 0) == 0:
        _diag = results.get("zero_edge_diagnostic") or []
        st.error(
            "This run produced **0 candidate pairs** — every record ended up in "
            "its own singleton cluster. Below is exactly why, field by field."
        )
        if _diag:
            st.dataframe(pd.DataFrame(_diag), width="stretch", hide_index=True)
            _bad = [d["field"] for d in _diag if d.get("issue")]
            if _bad:
                st.caption(
                    f"Field(s) with no repeated values at all: {', '.join(_bad)}. "
                    "Disable these as blocking rules, or check the Dataset Profile "
                    "step to confirm they weren't corrupted during cleaning."
                )
        st.divider()

    # ── KPI headline row ──────────────────────────────────────────────────────
    st.subheader("Summary")
    _metric_cards([
        ("Records processed",         f"{results['n_input_records']:,}"),
        ("Predicted edges (matches)", f"{metrics['n_edges']:,}"),
        ("Distinct entity clusters",  f"{metrics['n_clusters']:,}"),
        ("Unique IDs with a match",   f"{metrics['n_unique_ids']:,}"),
    ])

    st.divider()

    # ── Tabbed results ────────────────────────────────────────────────────────
    (tab_edges, tab_clusters, tab_demo,
     tab_explorer, tab_studio, tab_cm, tab_data) = st.tabs([
        "Edge Metrics",
        "Cluster Metrics",
        "Demographics",
        "Blocking Explorer",    # NEW interactive explorer tab
        "Cluster Studio",
        "Confusion Matrix",
        "Raw Data",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Edge Metrics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_edges:
        st.subheader("Edge Metrics")
        lt = results["run_config"]["linkage_type"]

        prob_stats = metrics.get("match_prob_stats", pd.DataFrame())
        if not prob_stats.empty:
            st.write("**Match Probability Statistics**")
            st.dataframe(prob_stats, width="stretch")

        prob_dist = metrics.get("prob_dist", pd.DataFrame())
        if not prob_dist.empty and len(prob_dist) > 1:
            st.plotly_chart(
                _plotly_bar(prob_dist, "prob_bin", "n_edges",
                            "Match Probability Distribution"),
                width="stretch",
            )
            st.caption(
                "Bars near 1.0 indicate confident predictions. "
                "Bars spread across mid-range indicate uncertain predictions."
            )

        weight_dist = metrics.get("weight_dist", pd.DataFrame())
        if not weight_dist.empty and len(weight_dist) > 1:
            st.plotly_chart(
                _plotly_bar(weight_dist, "weight_bin", "n_edges",
                            "Match Weight Histogram", "#E55C30"),
                width="stretch",
            )
            st.caption(
                "Match weight = log2(m/u). Positive values = more likely a match. "
                "Higher values = greater confidence."
            )

        gamma_df = metrics.get("gamma_means", pd.DataFrame())
        if not gamma_df.empty and lt == "probabilistic":
            g_long = gamma_df.T.reset_index()
            g_long.columns = ["field", "mean_gamma"]
            g_long["field"] = g_long["field"].str.replace("gamma_", "", regex=False)
            st.plotly_chart(
                _plotly_bar(g_long, "field", "mean_gamma",
                            "Mean Gamma Score per Field", "#2ECC71"),
                width="stretch",
            )
            st.caption(
                "Gamma = 1: exact agreement. Gamma = 0: total disagreement. "
                "High mean gamma means matched pairs agree on this field."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Cluster Metrics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_clusters:
        st.subheader("Cluster Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Total clusters", f"{metrics['n_clusters']:,}")
        c2.metric("Cross-dataset clusters", f"{metrics['n_cross_dataset']:,}")

        s = metrics.get("singleton_stats", pd.DataFrame())
        if not s.empty:
            st.write("**Single-record vs multi-record clusters**")
            _sc1, _sc2 = st.columns([1, 1])
            with _sc1:
                st.dataframe(s, width="stretch", hide_index=True)
                st.caption(
                    "High singleton count = many records could not be linked. "
                    "Multi-record clusters = found duplicates / cross-dataset matches."
                )
            with _sc2:
                if "cluster_type" in s.columns and "n_clusters" in s.columns:
                    _s_fig = px.bar(
                        s, x="cluster_type", y="n_clusters",
                        color="cluster_type",
                        color_discrete_sequence=["#1E6EC4", "#E55C30"],
                        title="Singleton vs Multi-record",
                        template="simple_white",
                        labels={"cluster_type": "", "n_clusters": "Number of clusters"},
                    )
                    _s_fig.update_layout(
                        height=280, showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(_s_fig, width="stretch")

        cs = metrics.get("cluster_sizes", pd.DataFrame())
        if not cs.empty:
            st.plotly_chart(
                _plotly_bar(cs, "n_nodes", "n_clusters", "Cluster Size Distribution"),
                width="stretch",
            )
            st.caption(
                "A J-shaped curve (many size-1, few large clusters) is typical. "
                "Very large clusters may indicate over-linking."
            )

        venn = metrics.get("venn", {})
        op   = results["run_config"]["operation_mode"]
        if op != "dedupe" and any(venn.values()):
            st.write("**Dataset Overlap in Clusters**")
            a_only  = venn.get("a_only",  0)
            b_only  = venn.get("b_only",  0)
            both_ab = venn.get("both_ab", 0)
            vdf = pd.DataFrame([
                {"Category": "Dataset A only", "N Clusters": a_only},
                {"Category": "Both A and B",   "N Clusters": both_ab},
                {"Category": "Dataset B only", "N Clusters": b_only},
            ])
            st.dataframe(vdf, width="stretch", hide_index=True)

            # ── Venn diagram (shared drawing function — ) ──
            st.plotly_chart(venn_figure(a_only, both_ab, b_only), width="stretch", key=f"{slot}_venn")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Demographics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_demo:
        st.subheader("Demographic Breakdown")
        g = metrics.get("gender_dist", pd.DataFrame())
        c = metrics.get("city_dist",   pd.DataFrame())
        d1, d2 = st.columns(2)
        if not g.empty:
            with d1:
                st.plotly_chart(
                    px.pie(g, values="n_records", names="gender",
                           title="Gender Distribution in Clusters",
                           template="simple_white",
                           color_discrete_sequence=px.colors.qualitative.Set2),
                    width="stretch",
                )
        if not c.empty:
            with d2:
                st.plotly_chart(
                    _plotly_bar(c.head(10), "city", "n_records",
                                "Top 10 Cities in Clusters", "#9B59B6"),
                    width="stretch",
                )

        if metrics.get("demographics"):
            st.divider()
            render_demographic_breakdowns(metrics["demographics"], key_prefix=f"{slot}_demo")

        # ── Linked vs. Unlinked demographic comparison ──────────────────────
        # Does linkage outcome differ systematically by demographic group?
        # Complements (doesn't replace) the population/match-quality views
        # from the last two rounds — this answers a distinct question:
        # who ends up WITHOUT a match, and how does that group differ.
        st.divider()
        st.write("**Linked vs. Unlinked — Demographic Comparison**")
        try:
            _linked_df, _unlinked_df = split_linked_unlinked(results.get("df_cluster", pd.DataFrame()))
            _linked_snap   = compute_demographic_snapshot(_linked_df)
            _unlinked_snap = compute_demographic_snapshot(_unlinked_df)
            render_demographic_comparison(
                _linked_snap, _unlinked_snap,
                baseline_label="Linked", current_label="Unlinked",
                key_prefix=f"{slot}_linked_unlinked",
            )
            st.caption(
                f"Linked: {len(_linked_df):,} records in clusters of size > 1. "
                f"Unlinked: {len(_unlinked_df):,} records in singleton clusters."
            )
        except Exception as _lu_err:
            st.warning(f"Could not compute linked vs unlinked comparison: {_lu_err}")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Interactive Blocking Explorer
    # Layout:
    #   Left panel  – toggleable rule cards with pair counts
    #   Right panel – live df_predict table + headline stats
    # Toggling a rule updates the table in real time (Streamlit rerun).
    # "Re-cluster" button recomputes entity clusters from the filtered edges.
    # ═══════════════════════════════════════════════════════════════════════
    with tab_explorer:
        st.subheader("Interactive Blocking Explorer")
        st.write(
            "Toggle blocking rules on or off. The pairwise edge table updates "
            "to show only pairs covered by at least one active rule. "
            "If a pair is covered by multiple rules, it is kept and the "
            "'effective rule' column reflects the first active rule covering it. "
            "Click 'Re-cluster' to see how the cluster assignments change."
        )

        cov_matrix = st.session_state.get(ck)
        if cov_matrix is None or cov_matrix.empty:
            st.info("Run an analysis first to enable the interactive explorer.")
        else:
            # Initialise explorer toggles from run config if not yet set
            run_toggles = results["run_config"].get("blocking_toggles", {})
            if not st.session_state.get(tk):
                st.session_state[tk] = dict(run_toggles)

            # ── Blocking cascade waterfall (live) ───────────────────────────────
            render_waterfall_section(
                cov_matrix, st.session_state[tk],
                results["run_config"].get("blocking_mode", "OR"),
                key_prefix=f"{slot}_exp_waterfall",
            )

            # ── Two-column layout ─────────────────────────────────────────────
            col_rules, col_table = st.columns([1, 2.5], gap="large")

            with col_rules:
                st.markdown("**Blocking Rules**")

                # Select All / Clear All buttons
                sa, ca = st.columns(2)
                if sa.button("Select All", key=f"{slot}_exp_all"):
                    for f in st.session_state[tk]:
                        st.session_state[tk][f] = True
                    st.rerun()
                if ca.button("Clear All", key=f"{slot}_exp_none"):
                    for f in st.session_state[tk]:
                        st.session_state[tk][f] = False
                    st.rerun()

                # Count map: pairs originally generated by each rule
                count_map = {
                    r["rule_sql"]: r["n"]
                    for r in results.get("blocking_counts", [])
                }

                # Rule cards
                new_toggles = {}
                for field, currently_on in st.session_state[tk].items():
                    with st.container(border=True):
                        tc, ic = st.columns([1, 3])
                        new_val = tc.toggle(
                            f"Use {field} rule", value=currently_on, key=f"{slot}_exp_tog_{field}",
                            label_visibility="collapsed",
                        )
                        new_toggles[field] = new_val
                        sql = f'l."{field}" = r."{field}"'
                        n   = count_map.get(sql, 0)
                        ic.markdown(f"**{field}**")
                        ic.code(sql, language="sql")
                        # ACTIVE / INACTIVE badge
                        badge = "ACTIVE" if new_val else "INACTIVE"
                        color = "green" if new_val else "grey"
                        ic.markdown(
                            f'<span style="color:{color};font-weight:bold;'
                            f'font-size:11px">{badge}</span>'
                            f'&nbsp;&nbsp;<span style="font-size:11px">'
                            f'{n:,} pairs</span>',
                            unsafe_allow_html=True,
                        )

                # Update explorer toggles if anything changed
                if new_toggles != st.session_state[tk]:
                    st.session_state[tk] = new_toggles

            with col_table:
                # ── Filter df_predict by active explorer rules ─────────────────
                filtered_df = filter_predict_by_active_rules(
                    results["df_predict"],
                    cov_matrix,
                    st.session_state[tk],
                )

                n_orig     = len(results["df_predict"])
                n_filtered = len(filtered_df)
                n_active   = sum(1 for v in st.session_state[tk].values() if v)
                reduction  = (1 - n_filtered / n_orig) * 100 if n_orig > 0 else 0

                # ── Headline stats ─────────────────────────────────────────────
                hs1, hs2, hs3, hs4 = st.columns(4)
                hs1.metric("Candidate Pairs",  f"{n_filtered:,}")
                hs2.metric("Rules Enabled",    f"{n_active}/{len(st.session_state[tk])}")
                hs3.metric("Reduction Ratio",  f"{reduction:.1f}%")
                hs4.metric("Original Pairs",   f"{n_orig:,}")

                # ── Pair table ─────────────────────────────────────────────────
                st.write("**Pairwise Edge Table**")
                if filtered_df.empty:
                    st.warning("No pairs covered by the current active rules.")
                else:
                    # Select display columns: IDs, effective rule, scores, key gammas
                    id_cols   = [c for c in ["unique_id_l","unique_id_r",
                                              "source_dataset_l","source_dataset_r"]
                                 if c in filtered_df.columns]
                    rule_cols = ["effective_rule"] if "effective_rule" in filtered_df.columns else []
                    score_cols= [c for c in ["match_probability","match_weight"]
                                 if c in filtered_df.columns]
                    gamma_cols= [c for c in filtered_df.columns
                                 if c.startswith("gamma_")][:4]   # show first 4 gammas max

                    display_cols = id_cols + rule_cols + score_cols + gamma_cols
                    display_df   = filtered_df[display_cols].head(200).copy()

                    # match_probability as a progress bar (built into Streamlit; no matplotlib needed)
                    st.dataframe(
                        display_df,
                        column_config={"match_probability": st.column_config.ProgressColumn(
                            "match_probability", min_value=0.0, max_value=1.0, format="%.3f")},
                        width="stretch",
                        height=360,
                    )

                    st.caption(
                        f"Showing up to 200 of {n_filtered:,} filtered pairs. "
                        "The match_probability bar is full for high-confidence pairs."
                    )

            # ── Match quality by demographic group (real-time) ─────────────────
            # Edge-level view — reacts immediately to rule toggles, unlike the
            # population-count charts, since disabling a rule directly removes
            # the edges it contributed.
            st.divider()
            try:
                _exp_fields = results["run_config"].get("selected_fields", [])
                render_match_quality_section(
                    results["df_predict"], filtered_df,
                    _exp_fields, _exp_fields,
                    threshold=st.session_state.get(thk, 0.8),
                    baseline_label="Original rules", current_label="Toggled rules (live)",
                    key_prefix=f"{slot}_exp_match_quality",
                )
            except Exception as _mq_err:
                st.warning(f"Could not compute match-quality comparison: {_mq_err}")

            # ── Re-cluster button ─────────────────────────────────────────────
            st.divider()
            exp_thresh = st.slider(
                "Cluster threshold for explorer",
                0.5, 0.99,
                st.session_state.get(thk, 0.8),
                0.01,
                key=f"{slot}_exp_thresh_slider",
            )
            st.session_state[thk] = exp_thresh

            if st.button("Re-cluster with active rules", type="primary"):
                if filtered_df.empty:
                    st.warning("No pairs to cluster.")
                else:
                    with st.spinner("Re-clustering..."):
                        try:
                            new_clusters = recluster_filtered(
                                df_predict_filtered=filtered_df,
                                dataset_a=st.session_state["dataset_a"],
                                dataset_b=st.session_state.get("dataset_b"),
                                threshold=exp_thresh,
                            )
                            if not new_clusters.empty:
                                new_n_clusters = new_clusters["cluster_id"].nunique()
                                st.success(
                                    f"Re-clustered: {new_n_clusters:,} clusters "
                                    f"from {n_filtered:,} filtered edges."
                                )
                                # Side-by-side comparison
                                rc1, rc2 = st.columns(2)
                                rc1.metric(
                                    "Clusters (original rules)",
                                    f"{metrics['n_clusters']:,}",
                                )
                                rc2.metric(
                                    "Clusters (explorer rules)",
                                    f"{new_n_clusters:,}",
                                    delta=f"{new_n_clusters - metrics['n_clusters']:+,}",
                                )
                            else:
                                st.info(
                                    "Re-clustering returned no clusters. "
                                    "Try lowering the threshold or enabling more rules."
                                )
                        except Exception as e:
                            st.error(f"Re-clustering failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Cluster Studio
    # ═══════════════════════════════════════════════════════════════════════
    with tab_studio:
        st.subheader("Splink Cluster Studio")
        st.write(
            "Interactive visualisation of entity clusters. Each node is a record; "
            "edges are predicted matches. Use this to visually inspect linkage quality."
        )
        html = results.get("cluster_html", "")
        if html:
            components.html(html, height=650, scrolling=True)
        else:
            st.info("Cluster studio HTML could not be generated for this run.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Confusion Matrix
    # ═══════════════════════════════════════════════════════════════════════
    with tab_cm:
        st.subheader("Confusion Matrix and Model Accuracy")
        st.write(
            "Ground truth: the 'cluster' column in the original datasets. "
            "Records sharing the same cluster value are true matches."
        )
        cm  = st.session_state.get(f"{slot}_cm", {})
        ts  = st.session_state.get(f"{slot}_threshold_curve")
        curve = st.session_state.get(f"{slot}_curve", {})

        if not cm:
            st.info("Confusion matrix not yet available. Run analysis first.")
        elif cm.get("unavailable"):
            # Dataset does not have a 'cluster' column — show a clear explanation
            st.info(cm.get("unavailable_reason", "Confusion matrix not available."))
            st.caption(
                "To use the confusion matrix, ensure your dataset has a 'cluster' column "
                "containing integer group labels identifying which records refer to the "
                "same real-world entity. The built-in demo datasets have this column."
            )
        elif "error" in cm:
            st.info(f"Confusion matrix error: {cm['error']}")
        else:
            st.caption(
                f"Pairs count as predicted at match probability ≥ {cm.get('min_match_probability', 0):.2f} "
                "(this run's cluster threshold). The precision-recall curve below covers every threshold."
            )
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("True Positives (TP)",  f"{cm.get('tp',0):,}")
            kc2.metric("False Positives (FP)", f"{cm.get('fp',0):,}")
            kc3.metric("False Negatives (FN)", f"{cm.get('fn',0):,}")
            kc4.metric("Ground truth pairs",   f"{cm.get('n_gt_edges',0):,}")

            st.divider()
            mc1, mc2 = st.columns(2)
            with mc1:
                st.write("**Derived Metrics**")
                mdf = pd.DataFrame([
                    {"Metric":"Precision", "Value":f"{cm.get('precision',0):.4f}",
                     "Meaning":"TP / (TP+FP)"},
                    {"Metric":"Recall",    "Value":f"{cm.get('recall',0):.4f}",
                     "Meaning":"TP / (TP+FN)"},
                    {"Metric":"F1 Score",  "Value":f"{cm.get('f1',0):.4f}",
                     "Meaning":"Harmonic mean"},
                    {"Metric":"F* Score",  "Value":f"{cm.get('fstar',0):.4f}",
                     "Meaning":"TP / (TP+FP+FN)"},
                    {"Metric":"FDR",       "Value":f"{cm.get('fdr',0):.4f}",
                     "Meaning":"False Discovery Rate"},
                    {"Metric":"FNR",       "Value":f"{cm.get('fnr',0):.4f}",
                     "Meaning":"False Negative Rate"},
                ])
                st.dataframe(mdf, width="stretch", hide_index=True)

            with mc2:
                st.write("**Confusion Matrix**")
                z    = [[cm.get("tp",0), cm.get("fp",0)],
                        [cm.get("fn",0), 0]]
                text = [[f"TP<br>{cm.get('tp',0):,}",  f"FP<br>{cm.get('fp',0):,}"],
                        [f"FN<br>{cm.get('fn',0):,}",  "TN<br>(omitted)"]]
                fig_cm = gobj.Figure(data=gobj.Heatmap(
                    z=z, text=text, texttemplate="%{text}",
                    colorscale=[[0,"#B85050"],[0.5,"#CCCCCC"],[1,"#1d8a50"]],
                    showscale=False,
                ))
                fig_cm.update_layout(
                    xaxis=dict(tickvals=[0,1],
                               ticktext=["Predicted Match","Predicted Non-Match"]),
                    yaxis=dict(tickvals=[0,1],
                               ticktext=["True Non-Match","True Match"],
                               autorange="reversed"),
                    height=260, margin=dict(l=10,r=10,t=30,b=10),
                    title="Pairwise Confusion Matrix",
                )
                st.plotly_chart(fig_cm, width="stretch")

        # Precision-recall curve (probabilistic runs with ground truth only)
        if ts is not None and not ts.empty:
            st.divider()
            st.subheader("Precision-Recall Curve and Threshold Summary")
            render_threshold_curve(ts, curve, key_prefix=f"{slot}_curve_view")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Raw Data
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.subheader("Raw Tables")
        st.write("**df_predict (first 100 rows)**")
        st.dataframe(results["df_predict"].head(100), width="stretch")
        st.caption(
            "Each row is a candidate record pair. "
            "gamma_ columns show field-level agreement (1=exact, 0=disagree). "
            "match_key indicates which blocking rule generated this pair."
        )
        st.write("**df_cluster (first 100 rows)**")
        st.dataframe(results["df_cluster"].head(100), width="stretch")
