# =============================================================================
# app.py
# Splink Cohort Builder - Main Streamlit Application
#
# PURPOSE: Provide a guided, 7-page UI for non-technical users to perform
#          record linkage and deduplication using Splink + DuckDB, backed by
#          the fake1000 dataset (unique_id, first_name, surname, dob, city,
#          email, cluster, gender, postcode).
#
# HOW TO RUN:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# PAGE FLOW:
#   0. Landing          - choose dummy dataset or upload (MVP: dummy only)
#   1. Configure        - field selection + blocking rules
#   2. Operation mode   - dedupe only vs link + dedupe
#   3. Linkage type     - deterministic vs probabilistic
#   4. Analysis         - run model, view results, download PDF
#   5. Comparison       - re-run with different rules, compare metrics
#   6. Export           - prepare cohort and download CSV
#
# SESSION STATE KEYS:
#   page              int  - current page index (0-6)
#   dataset_ready     bool - whether fakea/fakeb have been built
#   fakea             pd.DataFrame
#   fakeb             pd.DataFrame
#   selected_fields   list[str]
#   blocking_toggles  dict[str, bool]
#   operation_mode    str  - "dedupe" or "link_dedupe"
#   linkage_type      str  - "deterministic" or "probabilistic"
#   run1_results      dict - output of splink_runner.run_linkage()
#   run1_metrics      dict - output of metrics_engine.compute_intra_metrics()
#   run2_results      dict
#   run2_metrics      dict
# =============================================================================

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Local modules (must be importable from the same directory)
from modules.data_builder import build_datasets, get_library_status
from modules.splink_runner import run_linkage
from modules.metrics_engine import compute_intra_metrics, compute_inter_metrics
from modules.report_gen import generate_report

# ─────────────────────────────────────────────────────────────────────────────
# APP-WIDE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Splink Cohort Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# All linkage-eligible fields in the fake1000 dataset (unique_id excluded from
# blocking/comparisons since it is the record identifier, not a comparison field)
ALL_FIELDS = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]

# Human-readable labels for the navigation sidebar
PAGE_LABELS = [
    "Dataset Selection",
    "Configure Fields and Blocking",
    "Operation Mode",
    "Linkage Type",
    "Run Analysis",
    "Compare Runs",
    "Export Cohort",
]


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Sets default values the first time the app is loaded.
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    """Initialise all session state keys with safe defaults on first load."""
    defaults = {
        "page":             0,
        "dataset_ready":    False,
        "fakea":            None,
        "fakeb":            None,
        "selected_fields":  list(ALL_FIELDS),
        "blocking_toggles": {f: True for f in ALL_FIELDS},
        "operation_mode":   None,
        "linkage_type":     None,
        "run1_results":     None,
        "run1_metrics":     None,
        "run2_results":     None,
        "run2_metrics":     None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _go_to(page: int):
    """Navigate to a specific page by updating session state."""
    st.session_state["page"] = page
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar():
    """Render the step-list navigation in the sidebar."""
    st.sidebar.title("Cohort Builder")
    st.sidebar.caption("Step-by-step linkage workflow")
    st.sidebar.divider()
    for i, label in enumerate(PAGE_LABELS):
        current = i == st.session_state["page"]
        if current:
            st.sidebar.markdown(f"**-> Step {i + 1}: {label}**")
        else:
            st.sidebar.markdown(f"Step {i + 1}: {label}")
    st.sidebar.divider()
    st.sidebar.caption(
        "This MVP is built on Splink and DuckDB. "
        "All data is processed in-memory; nothing is written to disk."
    )


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _metric_cards(metrics: list):
    """Render a row of KPI metric cards from a list of (label, value) tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)


def _plotly_bar(df, x, y, title, colour="#1E6EC4"):
    """Return a clean Plotly bar chart figure."""
    fig = px.bar(
        df, x=x, y=y, title=title,
        color_discrete_sequence=[colour],
        template="simple_white",
    )
    fig.update_layout(
        title_font_size=14,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0: LANDING / DATASET SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def page_landing():
    """Landing page: select dummy dataset or upload own data."""
    st.title("Splink Cohort Builder")
    st.write(
        "Welcome. This tool lets you perform record linkage and deduplication "
        "using Splink, a probabilistic record linkage library. "
        "Follow the steps in the sidebar to build your cohort."
    )
    st.divider()

    col_dummy, col_upload = st.columns(2, gap="large")

    with col_dummy:
        st.subheader("Use the dummy dataset")
        st.write(
            "Work with the fake1000 dataset, derived from Splink's built-in fake_1000. "
            "It contains 1000 synthetic UK records with fields: "
            "first_name, surname, dob, city, email, gender, and postcode. "
            "A second dataset (Dataset B) is automatically generated as a "
            "50% sample with realistic data quality errors introduced."
        )
        if st.button("Load dummy dataset", use_container_width=True, type="primary"):
            with st.spinner(
                "Building datasets. This includes gender assignment from names "
                "and UK postcode lookup by city, which may take a few seconds..."
            ):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"] = fakea
                    st.session_state["fakeb"] = fakeb
                    st.session_state["dataset_ready"] = True
                    lib_status = get_library_status()
                    if not lib_status["gender_guesser"]:
                        st.warning(
                            "gender-guesser is not installed. "
                            "Gender values were assigned randomly with realistic base rates. "
                            "For name-based gender inference: pip install gender-guesser"
                        )
                    if not lib_status["pgeocode"]:
                        st.warning(
                            "pgeocode is not installed. "
                            "Synthetic placeholder postcodes were used. "
                            "For real UK postcodes: pip install pgeocode"
                        )
                    st.success("Dataset loaded.")
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")

    with col_upload:
        st.subheader("Upload your own dataset")
        st.write(
            "Upload a CSV with your own records and select which fields "
            "to use for linkage and deduplication."
        )
        st.info(
            "Upload functionality is available in the full deployment version. "
            "For this MVP, please use the dummy dataset on the left."
        )

    if st.session_state["dataset_ready"]:
        st.divider()
        st.subheader("Dataset A - Preview (first 5 rows)")
        st.dataframe(st.session_state["fakea"].head(5), use_container_width=True)
        st.caption(
            f"Dataset A: {len(st.session_state['fakea']):,} records  |  "
            f"Dataset B: {len(st.session_state['fakeb']):,} records "
            f"(50% sample of A with 14% first-name typos, 9% surname typos, "
            f"5% missing DOBs, 15% email variations, 11% city abbreviations, "
            f"7% gender errors)"
        )
        st.divider()
        if st.button("Continue to field configuration", type="primary"):
            _go_to(1)

    st.divider()
    st.subheader("About this tool")
    info1, info2, info3 = st.columns(3, gap="medium")

    with info1:
        st.markdown("**How cohort building works**")
        st.write(
            "You configure a linkage model by choosing: which fields to compare, "
            "which blocking rules to apply, whether to deduplicate one dataset or "
            "link two together, and whether to use deterministic or probabilistic "
            "linkage. The model then identifies matching records and groups them "
            "into entity clusters."
        )

    with info2:
        st.markdown("**Linkage and deduplication theory**")
        st.write(
            "Record linkage identifies records across different datasets that refer "
            "to the same real-world entity (person, organisation). Deduplication "
            "does the same within a single dataset. Probabilistic linkage uses a "
            "Fellegi-Sunter model trained with Expectation-Maximisation to assign "
            "a match probability to each candidate pair."
        )

    with info3:
        st.markdown("**What you will see in the results**")
        st.write(
            "After running the analysis: the number of predicted matches (edges), "
            "match probability distributions, gamma scores per field, entity "
            "clusters, demographic breakdowns, and an interactive Splink cluster "
            "studio. You can re-run with different blocking rules to compare "
            "results, and download a PDF report and cohort CSV."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: FIELD SELECTION + BLOCKING RULES
# ─────────────────────────────────────────────────────────────────────────────
def page_configure():
    """Page for field selection and blocking rule configuration."""
    st.title("Step 2: Configure Fields and Blocking Rules")

    if not st.session_state["dataset_ready"]:
        st.warning("Please load a dataset first.")
        if st.button("Go back to dataset selection"):
            _go_to(0)
        return

    with st.expander("Dataset A - Preview (first 10 rows)", expanded=False):
        st.dataframe(st.session_state["fakea"].head(10), use_container_width=True)

    st.divider()

    # ── Field selection ────────────────────────────────────────────────────────
    st.subheader("Fields to include in comparisons")
    st.write(
        "Select which fields will be used when comparing candidate record pairs. "
        "All fields are selected by default. Deselecting a field removes it from "
        "the comparison model. Note: unique_id and cluster are excluded as they "
        "are identifiers, not linkage features."
    )

    field_cols = st.columns(2)
    selected_fields = []
    for i, field in enumerate(ALL_FIELDS):
        col = field_cols[i % 2]
        checked = col.checkbox(
            label=field,
            value=(field in st.session_state["selected_fields"]),
            key=f"field_check_{field}",
        )
        if checked:
            selected_fields.append(field)

    if not selected_fields:
        st.error("At least one field must be selected.")
        return

    st.session_state["selected_fields"] = selected_fields

    st.divider()

    # ── Blocking rules ─────────────────────────────────────────────────────────
    st.subheader("Blocking rules")
    st.write(
        "Blocking rules reduce the search space from O(n^2) comparisons to a "
        "manageable number by only comparing records that agree on at least one "
        "blocking field. Enabling more blocking rules increases recall (more "
        "candidate pairs found) but also increases computation time. "
        "All rules are enabled by default."
    )

    blocking_toggles = {}
    toggle_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        col = toggle_cols[i % 3]
        enabled = col.toggle(
            label=field,
            value=st.session_state["blocking_toggles"].get(field, True),
            key=f"block_toggle_{field}",
        )
        blocking_toggles[field] = enabled

    if not any(blocking_toggles.values()):
        st.error("At least one blocking rule must be enabled.")
        return

    st.session_state["blocking_toggles"] = blocking_toggles
    active_rules = [f for f, v in blocking_toggles.items() if v]
    st.caption(f"Active blocking rules ({len(active_rules)}): {', '.join(active_rules)}")

    st.divider()
    if st.button("Continue to operation mode", type="primary"):
        _go_to(2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: OPERATION MODE
# ─────────────────────────────────────────────────────────────────────────────
def page_operation():
    """Choose between deduplication only and link + deduplicate."""
    st.title("Step 3: Operation Mode")
    st.write("Choose how you want to process the data.")
    st.divider()

    col_dedupe, col_link = st.columns(2, gap="large")

    with col_dedupe:
        st.subheader("Deduplication only")
        st.write(
            "Examines a single dataset (Dataset A, 1000 records) and identifies "
            "records within that dataset that refer to the same person. "
            "Use this when you have one dataset and want to remove or flag internal "
            "duplicates before analysis."
        )
        st.write("**Dataset used:** Dataset A (1000 records)")
        if st.button("Select: Deduplication only", use_container_width=True, type="primary"):
            st.session_state["operation_mode"] = "dedupe"
            _go_to(3)

    with col_link:
        st.subheader("Link and deduplicate")
        st.write(
            "Links Dataset A (1000 records) with Dataset B (500 records, a 50% "
            "sample of A with controlled errors). The model identifies corresponding "
            "records across the two datasets. "
            "Use this when you have two datasets from the same population and want "
            "to find records that represent the same individual."
        )
        st.write("**Datasets used:** Dataset A (1000) + Dataset B (500)")
        if st.button("Select: Link and deduplicate", use_container_width=True, type="primary"):
            st.session_state["operation_mode"] = "link_dedupe"
            _go_to(3)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: LINKAGE TYPE
# ─────────────────────────────────────────────────────────────────────────────
def page_linkage_type():
    """Choose between deterministic and probabilistic linkage."""
    st.title("Step 4: Linkage Type")
    st.write("Choose the statistical approach for deciding whether two records match.")
    st.divider()

    col_det, col_prob = st.columns(2, gap="large")

    with col_det:
        st.subheader("Deterministic linkage")
        st.write(
            "Two records are declared a match if they satisfy at least one active "
            "blocking rule exactly. No probability model is trained; all matched "
            "pairs receive match_probability = 1.0."
        )
        st.write(
            "**When to use:** High-quality data with consistent field values. "
            "Fast to run. Easy to explain. Cannot handle typos or missing values "
            "unless blocking rules are written to accommodate them."
        )
        if st.button("Select: Deterministic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "deterministic"
            _go_to(4)

    with col_prob:
        st.subheader("Probabilistic linkage")
        st.write(
            "A Fellegi-Sunter model is trained using Expectation-Maximisation (EM). "
            "For each field comparison, the model learns the probability of agreement "
            "given a true match (m-probability) and the probability of accidental "
            "agreement (u-probability). These are combined into a match weight "
            "(log2 odds ratio) and converted to a match probability between 0 and 1."
        )
        st.write(
            "**When to use:** Data with typos, missing values, or varying formats "
            "(as in Dataset B). More accurate for real-world data. Takes 1-2 minutes "
            "for training."
        )
        if st.button("Select: Probabilistic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "probabilistic"
            _go_to(4)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_analysis():
    """Run the Splink model and display results."""
    st.title("Step 5: Run Analysis")

    # Guard: check all required settings are in place
    if not st.session_state["dataset_ready"]:
        st.warning("No dataset loaded. Please start from Step 1.")
        if st.button("Go to Step 1"):
            _go_to(0)
        return
    if st.session_state["operation_mode"] is None:
        st.warning("Operation mode not set. Please complete Step 3.")
        if st.button("Go to Step 3"):
            _go_to(2)
        return
    if st.session_state["linkage_type"] is None:
        st.warning("Linkage type not set. Please complete Step 4.")
        if st.button("Go to Step 4"):
            _go_to(3)
        return

    # Configuration summary
    with st.expander("Current run configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Operation:** {st.session_state['operation_mode'].replace('_', ' ').title()}")
        c2.write(f"**Linkage type:** {st.session_state['linkage_type'].title()}")
        c3.write(f"**Fields:** {', '.join(st.session_state['selected_fields'])}")
        active_br = [f for f, v in st.session_state["blocking_toggles"].items() if v]
        st.write(f"**Active blocking rules:** {', '.join(active_br)}")

    st.divider()

    run_btn_label = (
        "Re-run analysis with current settings"
        if st.session_state["run1_results"] is not None
        else "Run analysis"
    )

    if st.button(run_btn_label, type="primary"):
        with st.spinner(
            "Running Splink model. "
            "Probabilistic mode trains an EM model which may take 1-2 minutes..."
        ):
            try:
                results = run_linkage(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=st.session_state["blocking_toggles"],
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                )
                metrics = compute_intra_metrics(
                    results["df_predict"],
                    results["df_cluster"],
                )
                st.session_state["run1_results"] = results
                st.session_state["run1_metrics"] = metrics
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    if st.session_state["run1_results"] is None:
        return

    results = st.session_state["run1_results"]
    metrics = st.session_state["run1_metrics"]

    # KPI headline metrics
    st.subheader("Summary")
    _metric_cards([
        ("Records processed",          f"{results['n_input_records']:,}"),
        ("Predicted edges (matches)",  f"{metrics['n_edges']:,}"),
        ("Distinct entity clusters",   f"{metrics['n_clusters']:,}"),
        ("Unique IDs with a match",    f"{metrics['n_unique_ids']:,}"),
    ])

    st.divider()

    tab_edges, tab_clusters, tab_demo, tab_studio, tab_data = st.tabs([
        "Edge Metrics",
        "Cluster Metrics",
        "Demographics",
        "Cluster Studio",
        "Raw Data",
    ])

    # ── TAB: Edge metrics ─────────────────────────────────────────────────────
    with tab_edges:
        st.subheader("Edge Metrics")
        st.write(
            "An edge is a predicted pairwise match between two records. "
            "match_probability (0-1) reflects model confidence. "
            "For deterministic linkage all edges have probability = 1.0."
        )

        prob_stats = metrics.get("match_prob_stats", pd.DataFrame())
        if not prob_stats.empty:
            st.write("**Match Probability Statistics**")
            st.dataframe(prob_stats, use_container_width=True)
            st.caption(
                "A high mean (>0.9) suggests confident predictions. "
                "A wide standard deviation indicates a mix of high- and low-confidence edges."
            )

        prob_dist = metrics.get("prob_dist", pd.DataFrame())
        if not prob_dist.empty and len(prob_dist) > 1:
            fig = _plotly_bar(
                prob_dist, "prob_bin", "n_edges",
                "Match Probability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each bar shows how many edges fall in that probability band. "
                "Bars clustered near 1.0 indicate a confident model."
            )

        weight_dist = metrics.get("weight_dist", pd.DataFrame())
        if not weight_dist.empty and len(weight_dist) > 1:
            fig2 = _plotly_bar(
                weight_dist, "weight_bin", "n_edges",
                "Match Weight Distribution", colour="#E55C30"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Match weight is the log2 odds ratio. "
                "Positive values mean a pair is more likely a match than not. "
                "Higher weights indicate greater model confidence."
            )

        gamma_df = metrics.get("gamma_means", pd.DataFrame())
        if not gamma_df.empty and st.session_state["linkage_type"] == "probabilistic":
            st.write("**Mean Gamma Scores by Field**")
            g_long = gamma_df.T.reset_index()
            g_long.columns = ["field", "mean_gamma"]
            g_long["field"] = g_long["field"].str.replace("gamma_", "", regex=False)
            fig3 = _plotly_bar(
                g_long, "field", "mean_gamma",
                "Mean Gamma Score per Field", colour="#2ECC71"
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "Gamma = 1: records agree exactly on this field. "
                "Gamma = 0: complete disagreement. "
                "Intermediate values represent partial fuzzy agreement (e.g. Jaro-Winkler). "
                "High gamma on a field means matched pairs tend to agree on that field."
            )

    # ── TAB: Cluster metrics ──────────────────────────────────────────────────
    with tab_clusters:
        st.subheader("Cluster Metrics")
        st.write(
            "Clusters are groups of records predicted to represent the same entity. "
            "A singleton cluster (size 1) contains a record with no predicted matches."
        )

        c1, c2 = st.columns(2)
        c1.metric("Total clusters", f"{metrics['n_clusters']:,}")
        c2.metric("Cross-dataset clusters", f"{metrics['n_cross_dataset']:,}")

        singleton_stats = metrics.get("singleton_stats", pd.DataFrame())
        if not singleton_stats.empty:
            st.write("**Singleton vs Multi-record Clusters**")
            st.dataframe(singleton_stats, use_container_width=True)
            st.caption(
                "High singleton count: many records could not be linked (either "
                "genuinely unique, or blocking rules are too restrictive). "
                "Multi-record clusters: found duplicates or cross-dataset matches."
            )

        cluster_sizes = metrics.get("cluster_sizes", pd.DataFrame())
        if not cluster_sizes.empty:
            fig4 = _plotly_bar(
                cluster_sizes, "n_nodes", "n_clusters",
                "Cluster Size Distribution"
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(
                "A J-shaped curve (many singletons, few large clusters) is typical "
                "for real-world data. Very large clusters may indicate over-linking."
            )

        source_overlap = metrics.get("source_overlap", pd.DataFrame())
        if not source_overlap.empty and len(source_overlap) > 1:
            st.write("**Source Dataset Membership in Clusters**")
            st.dataframe(source_overlap, use_container_width=True)

    # ── TAB: Demographics ─────────────────────────────────────────────────────
    with tab_demo:
        st.subheader("Demographic Breakdown")
        st.write(
            "These charts show the demographic composition of the entity clusters. "
            "Comparing demographics between runs can reveal biases in the model."
        )

        gender_dist = metrics.get("gender_dist", pd.DataFrame())
        city_dist   = metrics.get("city_dist",   pd.DataFrame())

        dem_col1, dem_col2 = st.columns(2)

        if not gender_dist.empty:
            with dem_col1:
                fig5 = px.pie(
                    gender_dist, values="n_records", names="gender",
                    title="Gender Distribution in Clusters",
                    template="simple_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig5, use_container_width=True)

        if not city_dist.empty:
            with dem_col2:
                fig6 = _plotly_bar(
                    city_dist.head(10), "city", "n_records",
                    "Top 10 Cities in Clusters", colour="#9B59B6"
                )
                st.plotly_chart(fig6, use_container_width=True)

    # ── TAB: Cluster Studio ───────────────────────────────────────────────────
    with tab_studio:
        st.subheader("Splink Cluster Studio")
        st.write(
            "The cluster studio is an interactive visualisation of entity clusters. "
            "Each node is a record; edges between nodes are predicted matches. "
            "Use this to visually inspect the linkage quality and identify "
            "over- or under-linking issues."
        )
        cluster_html = results.get("cluster_html", "")
        if cluster_html:
            components.html(cluster_html, height=650, scrolling=True)
        else:
            st.info(
                "Cluster studio HTML could not be generated for this run. "
                "All other metrics are unaffected."
            )

    # ── TAB: Raw Data ─────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Raw Tables")
        st.write("**df_predict (first 100 rows)**")
        st.dataframe(results["df_predict"].head(100), use_container_width=True)
        st.caption(
            "Each row is a candidate record pair. "
            "match_probability = confidence that the pair is a true match. "
            "gamma_ columns show field-level agreement levels."
        )
        st.write("**df_cluster (first 100 rows)**")
        st.dataframe(results["df_cluster"].head(100), use_container_width=True)
        st.caption(
            "Each row is a record with its assigned cluster_id. "
            "Records sharing a cluster_id are predicted to be the same entity."
        )

    st.divider()

    # PDF download
    st.subheader("Download report")
    if st.button("Generate PDF report for Run 1"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_report(
                    run_label="Run 1",
                    run_config=results["run_config"],
                    metrics=metrics,
                    n_input_records=results["n_input_records"],
                )
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="linkage_report_run1.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()
    if st.button("Continue to compare runs", type="primary"):
        _go_to(5)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def page_comparison():
    """Modify blocking rules, re-run, and compare Run 1 vs Run 2."""
    st.title("Step 6: Compare Runs")

    if st.session_state["run1_results"] is None:
        st.warning("No Run 1 results available. Please complete Step 5 first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    run1 = st.session_state["run1_results"]
    m1   = st.session_state["run1_metrics"]

    st.write(
        "Modify the blocking rules below and re-run the analysis. "
        "The comparison table shows how results differ between Run 1 and Run 2."
    )
    st.divider()

    # Run 1 summary
    st.subheader("Run 1 summary")
    active_br1 = [f for f, v in run1["run_config"]["blocking_toggles"].items() if v]
    st.caption(f"Blocking rules used: {', '.join(active_br1)}")
    _metric_cards([
        ("Run 1: Edges",            f"{m1['n_edges']:,}"),
        ("Run 1: Clusters",         f"{m1['n_clusters']:,}"),
        ("Run 1: Mean match prob",
         str(m1["match_prob_stats"]["mean_match_prob"].iloc[0])
         if not m1["match_prob_stats"].empty else "N/A"),
    ])

    st.divider()
    st.subheader("Modify blocking rules for Run 2")
    st.write(
        "Toggle fields on or off. The comparison fields remain the same as Run 1; "
        "only the blocking rules change."
    )

    if "run2_blocking_toggles" not in st.session_state:
        st.session_state["run2_blocking_toggles"] = dict(
            run1["run_config"]["blocking_toggles"]
        )

    run2_toggles = {}
    t_cols = st.columns(3)
    for i, field in enumerate(st.session_state["selected_fields"]):
        col = t_cols[i % 3]
        enabled = col.toggle(
            label=field,
            value=st.session_state["run2_blocking_toggles"].get(field, True),
            key=f"run2_block_{field}",
        )
        run2_toggles[field] = enabled

    if not any(run2_toggles.values()):
        st.error("At least one blocking rule must be enabled for Run 2.")
        return

    st.session_state["run2_blocking_toggles"] = run2_toggles

    if st.button("Run analysis with updated blocking rules", type="primary"):
        with st.spinner("Running Splink model for Run 2..."):
            try:
                run2 = run_linkage(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=run2_toggles,
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                )
                m2 = compute_intra_metrics(run2["df_predict"], run2["df_cluster"])
                st.session_state["run2_results"] = run2
                st.session_state["run2_metrics"] = m2
                st.success("Run 2 complete.")
            except Exception as e:
                st.error(f"Run 2 failed: {e}")
                return

    if st.session_state["run2_results"] is None:
        return

    run2 = st.session_state["run2_results"]
    m2   = st.session_state["run2_metrics"]

    # Compute inter-model comparison metrics
    inter = compute_inter_metrics(
        df_predict_run1=run1["df_predict"],
        df_predict_run2=run2["df_predict"],
        df_cluster_run1=run1["df_cluster"],
        df_cluster_run2=run2["df_cluster"],
    )

    st.divider()
    st.subheader("Comparison: Run 1 vs Run 2")

    # KPI deltas
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.metric("Edges", f"{m2['n_edges']:,}", delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
    with kc2:
        st.metric("Clusters", f"{m2['n_clusters']:,}", delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
    with kc3:
        mp1 = (m1["match_prob_stats"]["mean_match_prob"].iloc[0]
               if not m1["match_prob_stats"].empty else 0)
        mp2 = (m2["match_prob_stats"]["mean_match_prob"].iloc[0]
               if not m2["match_prob_stats"].empty else 0)
        st.metric("Mean match probability", f"{mp2:.4f}", delta=f"{mp2 - mp1:+.4f}")

    st.divider()

    # Edge difference table
    st.subheader("Edge Changes Between Runs")
    edge_diff = inter.get("edge_diff", pd.DataFrame())
    if not edge_diff.empty:
        edge_dict = edge_diff.set_index("category")["n"].to_dict()
        comp_table = pd.DataFrame([
            {"Metric": "Shared edges (both runs)",     "Count": edge_dict.get("shared", 0)},
            {"Metric": "Edges added in Run 2",         "Count": edge_dict.get("added", 0)},
            {"Metric": "Edges removed in Run 2",       "Count": edge_dict.get("removed", 0)},
            {"Metric": "Exact matching clusters",      "Count": inter.get("n_exact_matching_clusters", 0)},
            {"Metric": "Partially matching clusters",  "Count": inter.get("n_partial_matching_clusters", 0)},
        ])
        st.dataframe(comp_table, use_container_width=True)
        st.caption(
            "Added edges in Run 2: new matches found (more permissive blocking). "
            "Removed edges in Run 2: matches lost (more restrictive blocking). "
            "Exact matching clusters: clusters whose membership is identical across runs."
        )

    # Match probability comparison
    prob_comp = inter.get("prob_comparison", pd.DataFrame())
    if not prob_comp.empty:
        st.write("**Match Probability Comparison**")
        st.dataframe(prob_comp, use_container_width=True)

    # Side-by-side probability distribution
    pd1 = inter.get("prob_dist_run1", pd.DataFrame())
    pd2 = inter.get("prob_dist_run2", pd.DataFrame())
    if not pd1.empty and not pd2.empty:
        pd1["run"] = "Run 1"
        pd2["run"] = "Run 2"
        combined_prob = pd.concat([pd1, pd2], ignore_index=True)
        fig = px.bar(
            combined_prob, x="prob_bin", y="n_edges", color="run",
            barmode="group",
            title="Match Probability Distribution: Run 1 vs Run 2",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4", "#E55C30"],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster size distribution comparison
    cs1 = inter.get("cluster_sizes_run1", pd.DataFrame())
    cs2 = inter.get("cluster_sizes_run2", pd.DataFrame())
    if not cs1.empty and not cs2.empty:
        cs1["run"] = "Run 1"
        cs2["run"] = "Run 2"
        combined_cs = pd.concat([cs1, cs2], ignore_index=True)
        fig7 = px.bar(
            combined_cs, x="n_nodes", y="n_clusters", color="run",
            barmode="group",
            title="Cluster Size Distribution: Run 1 vs Run 2",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4", "#E55C30"],
        )
        fig7.update_layout(height=350)
        st.plotly_chart(fig7, use_container_width=True)

    # Gamma comparison (probabilistic only)
    g_comp = inter.get("gamma_comparison", pd.DataFrame())
    if not g_comp.empty and st.session_state["linkage_type"] == "probabilistic":
        st.write("**Mean Gamma Score Comparison**")
        g_long = g_comp.melt(id_vars="run", var_name="field", value_name="mean_gamma")
        g_long["field"] = g_long["field"].str.replace("gamma_", "", regex=False)
        fig8 = px.bar(
            g_long, x="field", y="mean_gamma", color="run",
            barmode="group",
            title="Mean Gamma Score by Field: Run 1 vs Run 2",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4", "#E55C30"],
        )
        fig8.update_layout(height=350)
        st.plotly_chart(fig8, use_container_width=True)
        st.caption(
            "Gamma scores should not change dramatically between runs (only blocking rules "
            "differ, not the comparison functions). Large differences may indicate that "
            "blocking affects which pairs are used in EM training."
        )

    st.divider()
    if st.button("Generate PDF report for Run 2"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_report(
                    run_label="Run 2",
                    run_config=run2["run_config"],
                    metrics=m2,
                    n_input_records=run2["n_input_records"],
                )
                st.download_button(
                    label="Download Run 2 PDF",
                    data=pdf_bytes,
                    file_name="linkage_report_run2.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()
    if st.button("Continue to export", type="primary"):
        _go_to(6)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: EXPORT COHORT
# ─────────────────────────────────────────────────────────────────────────────
def page_export():
    """Build final cohort and download as CSV."""
    st.title("Step 7: Export Cohort")

    run1_available = st.session_state["run1_results"] is not None
    run2_available = st.session_state["run2_results"] is not None

    if not run1_available:
        st.warning("No analysis results available. Please complete Step 5 first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write(
        "Prepare your cohort for export. The output is a CSV containing all "
        "original record fields plus a cluster_id column. Records sharing the "
        "same cluster_id are predicted to represent the same real-world individual."
    )
    st.divider()

    # Run selection
    st.subheader("Select which run to export")
    run_options = ["Run 1"] + (["Run 2"] if run2_available else [])
    selected_run = st.radio(
        "Choose the run whose cluster assignments to export:",
        options=run_options,
        horizontal=True,
    )

    chosen_results = (
        st.session_state["run1_results"] if selected_run == "Run 1"
        else st.session_state["run2_results"]
    )

    st.divider()

    df_cluster = chosen_results["df_cluster"]
    operation  = chosen_results["run_config"]["operation_mode"]

    # Build the enriched cohort: merge cluster_id onto original record data
    if operation == "dedupe":
        raw_data = st.session_state["fakea"].copy()
    else:
        raw_data = pd.concat(
            [st.session_state["fakea"], st.session_state["fakeb"]],
            ignore_index=True,
        )

    # Merge keys: always use unique_id; include source_dataset if present
    merge_keys = (
        ["unique_id", "source_dataset"]
        if "source_dataset" in df_cluster.columns
        else ["unique_id"]
    )
    cohort_df = raw_data.merge(
        df_cluster[merge_keys + ["cluster_id"]],
        on=merge_keys,
        how="left",         # Keep all records; NaN cluster_id = no predicted match
    )

    # Cohort summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(cohort_df):,}")
    c2.metric("Distinct cluster IDs", f"{cohort_df['cluster_id'].nunique():,}")
    c3.metric("Records assigned to a cluster", f"{cohort_df['cluster_id'].notna().sum():,}")

    st.subheader("Cohort preview (first 50 rows, sorted by cluster_id)")
    st.dataframe(
        cohort_df.sort_values("cluster_id").head(50),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Download")
    csv_bytes = cohort_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download cohort CSV ({selected_run})",
        data=csv_bytes,
        file_name=f"cohort_{selected_run.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.info(
        "In the full deployment version, this step will include direct provisioning "
        "to the SAIL Databank. For this MVP, the cohort is exported as CSV only."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Entry point: initialise state, render sidebar, dispatch to current page."""
    _init_state()
    _render_sidebar()

    page_functions = {
        0: page_landing,
        1: page_configure,
        2: page_operation,
        3: page_linkage_type,
        4: page_analysis,
        5: page_comparison,
        6: page_export,
    }

    page_fn = page_functions.get(st.session_state["page"], page_landing)
    page_fn()


if __name__ == "__main__":
    main()
