# =============================================================================
# app.py — Splink Cohort Builder
# Entry point only. All page logic lives in pages/. All shared utilities in utils/.
#
# File structure:
#   app.py                     ← this file (router + config)
#   utils/state.py             ← session state init + constants
#   utils/nav.py               ← navigation helpers + sidebar
#   utils/helpers.py           ← UI helpers + analysis runner
#   pages/p_landing.py         ← landing page (three mode cards)
#   pages/p_standard.py        ← standard flow: configure, operation, linkage type
#   pages/p_advanced.py        ← advanced flow: JSON model upload
#   pages/p_upload.py          ← upload flow: setup, EDA, configure (fully dynamic)
#   pages/p_analysis.py        ← analysis page (shared by all flows)
#   pages/p_compare_export.py  ← comparison + export pages
#   modules/                   ← Splink, EDA, metrics, report modules
#
# Run:  streamlit run app.py
# =============================================================================

import streamlit as st

<<<<<<< Updated upstream
from modules.data_builder import build_datasets, get_library_status
from modules.eda_engine import (
    run_full_eda, find_high_correlation_pairs, detect_field_types,
    suggest_comparison_types, suggest_blocking_rules,
    introduce_errors_for_sample,
)
from modules.splink_runner import (
    run_linkage,
    run_linkage_from_json,
    build_coverage_matrix,
    filter_predict_by_active_rules,
    recluster_filtered,
    reconstruct_model_json,
)
from modules.metrics_engine import (
    compute_intra_metrics,
    compute_inter_metrics,
    compute_confusion_matrix,
    compute_truth_space,
    compute_crl_score,
)
from modules.report_gen import generate_report

# ─────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────────────────────────────────────
=======
>>>>>>> Stashed changes
st.set_page_config(
    page_title="Cohort Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from utils.state import _init_state
from utils.nav import _render_sidebar, _go_to

from flows.p_landing         import page_landing
from flows.p_standard        import page_configure, page_operation, page_linkage_type
from flows.p_advanced        import page_advanced_setup
from flows.p_upload          import page_upload_setup, page_eda, page_upload_configure
from flows.p_analysis        import page_analysis
from flows.p_compare_export  import page_comparison, page_export


<<<<<<< Updated upstream

# =============================================================================
# ── SESSION STATE ─────────────────────────────────────────────────────────────
# =============================================================================

def _init_state():
    """Set safe defaults for all session state keys on first load."""
    defaults = {
        # ── Navigation ──────────────────────────────────────────────────────
        "page":              0,          # Current page (int or "advanced_setup")
        "page_history":      [],         # Stack for back-navigation
        "flow":              "standard", # "standard" | "advanced"

        # ── Datasets ────────────────────────────────────────────────────────
        "dataset_ready":     False,
        "fakea":             None,
        "fakeb":             None,

        # ── Standard flow model config ───────────────────────────────────────
        "selected_fields":   list(ALL_FIELDS),
        "blocking_toggles":  {f: True for f in ALL_FIELDS},
        "composite_rules":   {},        # {"first_name+surname": True, ...}
        "operation_mode":    None,
        "linkage_type":      None,
        "hyperparams":       {          # Training hyperparameters
            "max_iterations":  25,
            "em_convergence":  0.0001,
            "recall_estimate": 0.6,
        },

        # ── Advanced flow ────────────────────────────────────────────────────
        "advanced_json":     None,      # Uploaded Splink model JSON as dict
        "advanced_op_mode":  "dedupe",  # Operation mode for advanced flow

        # ── Analysis results (shared by both flows) ──────────────────────────
        "run1_results":      None,
        "run1_metrics":      None,
        "run1_cm":           None,
        "run1_ts":           None,
        "run1_crl":          {},

        # ── Comparison (Run 2) ────────────────────────────────────────────────
        "run2_results":      None,
        "run2_metrics":      None,
        "run2_blocking_toggles": None,

        # ── Interactive blocking explorer ─────────────────────────────────────
        "coverage_matrix":   None,      # Per-pair field coverage DataFrame
        "explorer_toggles":  {},        # Which blocking rules are ON in explorer
        "explorer_threshold": 0.8,      # Cluster threshold for live re-clustering

        # ── Upload flow (new) ──────────────────────────────────────────────
        "upload_raw_df":     None,   # Original uploaded DataFrame before cleaning
        "upload_clean_df":   None,   # Cleaned DataFrame after EDA pipeline
        "upload_df_b":       None,   # Second dataset or error-introduced sample
        "upload_field_types": {},    # column → detected type string
        "upload_name_map":   {},     # original_name → cleaned_name
        "upload_eda_log":    {},     # What each EDA step did
        "upload_corr_pairs": [],     # (col_a, col_b, score) high-correlation pairs
        "upload_dropped":    set(),  # Columns user chose to drop from corr check
        "upload_id_col":     None,   # Which column is the unique identifier
        "upload_comp_types": {},     # column → comparison type string
        "upload_block_toggles": {},  # column → bool blocking rule
        "upload_comp_rules": {},     # composite blocking rules
        "upload_has_two":    False,  # Whether user uploaded two separate datasets
        "upload_two_df":     None,   # Optionally uploaded second raw dataset
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# ── NAVIGATION HELPERS ────────────────────────────────────────────────────────
# =============================================================================

def _go_to(page):
    """Navigate to a page, pushing the current page onto the history stack."""
    current = st.session_state["page"]
    history = st.session_state["page_history"]
    # Only push if we're actually moving to a different page
    if not history or history[-1] != current:
        history.append(current)
    st.session_state["page_history"] = history
    st.session_state["page"] = page
    st.rerun()


def _go_back():
    """Navigate to the previous page by popping the history stack."""
    history = st.session_state["page_history"]
    if history:
        prev = history.pop()
        st.session_state["page_history"] = history
        st.session_state["page"] = prev
        st.rerun()


def _back_button(label="Previous Step"):
    """Render a back button; only visible when there is history to go back to."""
    if st.session_state["page_history"]:
        # Use a unique key per page so Streamlit doesn't confuse multiple buttons
        if st.button(f"<- {label}", key=f"back_{st.session_state['page']}_{label}"):
            _go_back()


# =============================================================================
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# =============================================================================

def _render_sidebar():
    """Sidebar: shows flow type, step list, and a global back button."""
    flow = st.session_state.get("flow", "standard")
    st.sidebar.title("Cohort Builder")

    # Show which flow is active
    # ── Mode switcher: radio so labels always fit, no squishing ─────────────
    st.sidebar.caption("Active mode")
    _mode_map = {"standard": 0, "upload": 1, "advanced": 2}
    _mode_labels = ["Standard", "Upload Data", "Advanced (JSON)"]
    _selected_mode = st.sidebar.radio(
        "mode_switch",
        options=_mode_labels,
        index=_mode_map.get(st.session_state.get("flow", "standard"), 0),
        label_visibility="collapsed",
    )
    if _selected_mode == "Standard" and st.session_state.get("flow") != "standard":
        st.session_state["flow"] = "standard"
        st.session_state["page"] = 0
        st.session_state["page_history"] = []
        st.rerun()
    elif _selected_mode == "Upload Data" and st.session_state.get("flow") != "upload":
        st.session_state["flow"] = "upload"
        st.session_state["page"] = "upload_setup"
        st.session_state["page_history"] = []
        st.rerun()
    elif _selected_mode == "Advanced (JSON)" and st.session_state.get("flow") != "advanced":
        st.session_state["flow"] = "advanced"
        st.session_state["page"] = "advanced_setup"
        st.session_state["page_history"] = []
        st.rerun()
    st.sidebar.divider()

    if flow == "advanced":
        st.sidebar.caption("Mode: Advanced (JSON upload)")
    else:
        st.sidebar.caption("Mode: Standard (guided workflow)")

    st.sidebar.divider()
    page = st.session_state["page"]

    if flow == "advanced":
        for key, label in ADVANCED_LABELS.items():
            current = key == page
            prefix  = "-> " if current else "   "
            st.sidebar.markdown(
                f"**{prefix}{label}**" if current else f"{prefix}{label}"
            )
    elif flow == "upload":
        UPLOAD_LABELS = {
            "upload_setup":     "Upload Datasets",
            "upload_eda":       "EDA and Cleaning",
            "upload_configure": "Configure Fields",
            2: "Operation Mode",
            3: "Linkage Type",
            4: "Run Analysis",
            5: "Compare Runs",
            6: "Export Cohort",
        }
        for key, label in UPLOAD_LABELS.items():
            current = key == page
            if current:
                st.sidebar.markdown(f"**-> {label}**")
            else:
                if st.sidebar.button(label, key=f"nav_up_{key}"):
                    _go_to(key)
    else:
        for i, label in enumerate(STANDARD_LABELS):
            current = i == page
            if current:
                st.sidebar.markdown(f"**-> Step {i+1}: {label}**")
            else:
                if st.sidebar.button(f"Step {i+1}: {label}", key=f"nav_std_{i}"):
                    _go_to(i)

    st.sidebar.divider()

    # Global back button in sidebar
    if st.session_state["page_history"]:
        if st.sidebar.button("Go back"):
            _go_back()

    # Quick-jump to Export (always available once analysis is done)
    if st.session_state.get("run1_results") and page not in (6, "advanced_setup"):
        st.sidebar.divider()
        if st.sidebar.button("Jump to Export"):
            _go_to(6)

    st.sidebar.divider()
    st.sidebar.caption(
        "All data processed in memory. "
        "Nothing is written to disk."
    )


# =============================================================================
# ── SHARED UI HELPERS ─────────────────────────────────────────────────────────
# =============================================================================

def _metric_cards(metrics: list):
    """Render a row of st.metric KPI cards from [(label, value), ...] tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)


def _plotly_bar(df, x, y, title, colour="#1E6EC4"):
    """Return a clean Plotly bar chart."""
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


def _run_analysis_and_store(
    fakea, fakeb, selected_fields, blocking_toggles,
    operation_mode, linkage_type, hyperparams, composite_rules,
):
    """Run linkage (or re-run) and store all derived results in session state.
    Used by both the standard analysis page and the advanced flow analysis.
    Returns True on success, False on error (error is shown via st.error)."""
    try:
        results = run_linkage(
            fakea=fakea, fakeb=fakeb,
            selected_fields=selected_fields,
            blocking_toggles=blocking_toggles,
            operation_mode=operation_mode,
            linkage_type=linkage_type,
            hyperparams=hyperparams,
            composite_rules=composite_rules,
        )
    except Exception as e:
        st.error(f"Linkage failed: {e}")
        return False

    # Core metrics
    metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])

    # Confusion matrix (uses cluster column as ground truth)
    cm = compute_confusion_matrix(
        results["df_predict"], fakea, fakeb, operation_mode
    )

    # Truth space and CRL (probabilistic only; safe to skip for deterministic)
    if linkage_type == "probabilistic":
        ts  = compute_truth_space(results["df_predict"], fakea, fakeb, operation_mode)
        crl = compute_crl_score(ts)
    else:
        ts  = None
        crl = {}

    # Coverage matrix for the interactive explorer tab
    # Uses the field columns already present in df_predict (retain_matching_columns=True)
    active_fields  = selected_fields   # All selected fields as potential blocking fields
    cov_matrix     = build_coverage_matrix(results["df_predict"], active_fields)

    # Store everything
    st.session_state["run1_results"]     = results
    st.session_state["run1_metrics"]     = metrics
    st.session_state["run1_cm"]          = cm
    st.session_state["run1_ts"]          = ts
    st.session_state["run1_crl"]         = crl
    st.session_state["coverage_matrix"]  = cov_matrix
    # Initialise explorer toggles to match the current blocking rules
    st.session_state["explorer_toggles"] = dict(blocking_toggles)

    return True


# =============================================================================
# ── PAGE 0: LANDING ───────────────────────────────────────────────────────────
# Two cards: Standard flow (dummy dataset) | Advanced flow (upload JSON).
# Upload own CSV is present but marked as coming soon.
# =============================================================================

def page_landing():
    st.title("Splink Cohort Builder")
    st.write(
        "Choose how you want to work. Standard mode walks you through every "
        "configuration step. Advanced mode lets you upload a pre-trained Splink "
        "model JSON and jump straight to prediction and analysis."
    )
    st.divider()

    col_std, col_up, col_adv = st.columns(3, gap="large")

    # ── Standard flow card ────────────────────────────────────────────────────
    with col_std:
        st.subheader("Standard")
        st.caption("Guided workflow for non-technical users")
        st.write(
            "Work with the built-in fake1000 dataset. You will be guided through "
            "field selection, blocking rule configuration, operation mode, linkage "
            "type, and analysis one step at a time."
        )
        if st.button("Use dummy dataset", use_container_width=True, type="primary"):
            with st.spinner("Building datasets (gender + UK postcode lookup)..."):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"]        = fakea
                    st.session_state["fakeb"]        = fakeb
                    st.session_state["dataset_ready"] = True
                    st.session_state["flow"]         = "standard"
                    libs = get_library_status()
                    if not libs["gender_guesser"]:
                        st.warning("gender-guesser not installed: gender assigned randomly. "
                                   "pip install gender-guesser for name-based inference.")
                    if not libs["pgeocode"]:
                        st.warning("pgeocode not installed: synthetic postcodes used. "
                                   "pip install pgeocode for real UK postcodes.")
                    st.success("Dataset loaded.")
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")

    # ── Advanced flow card ────────────────────────────────────────────────────
    with col_adv:
        st.subheader("Advanced")
        st.caption("Upload a pre-trained Splink model JSON")
        st.write(
            "For users who have already trained a Splink model in a notebook or "
            "external system. Upload the model JSON to skip training and go directly "
            "to prediction, analysis, the interactive blocking explorer, and export."
        )
        if st.button("Upload model JSON", use_container_width=True):
            st.session_state["flow"] = "advanced"
            _go_to("advanced_setup")

    # ── Upload flow card ─────────────────────────────────────────────────────
    with col_up:
        st.subheader("Upload Your Data")
        st.caption("Upload one or two CSV datasets")
        st.write(
            "Upload your own records. The app will clean the data with automated "
            "EDA (field name standardisation, null removal, deduplication, date "
            "standardisation, correlation check), then guide you through field "
            "configuration and blocking rules before running analysis."
        )
        if st.button("Upload dataset", use_container_width=True):
            st.session_state["flow"] = "upload"
            _go_to("upload_setup")

    # ── Preview if dataset is loaded ──────────────────────────────────────────
    if st.session_state["dataset_ready"]:
        st.divider()
        st.subheader("Dataset A - Preview")
        st.dataframe(st.session_state["fakea"].head(5), use_container_width=True)
        st.caption(
            f"Dataset A: {len(st.session_state['fakea']):,} records  |  "
            f"Dataset B: {len(st.session_state['fakeb']):,} records "
            "(50% sample of A with controlled errors)"
        )
        st.divider()
        if st.button("Continue to field configuration", type="primary"):
            _go_to(1)

    st.divider()

    # ── Info panels ──────────────────────────────────────────────────────────
    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        st.markdown("**How cohort building works**")
        st.write(
            "Configure which fields to compare, which blocking rules to apply, "
            "and whether to deduplicate one dataset or link two together. "
            "The model then identifies matching records and groups them into "
            "entity clusters."
        )
    with i2:
        st.markdown("**Linkage and deduplication theory**")
        st.write(
            "Probabilistic linkage uses the Fellegi-Sunter model trained by "
            "Expectation-Maximisation to assign a match probability to each "
            "candidate pair. Deterministic linkage applies hard exact-match rules."
        )
    with i3:
        st.markdown("**What you will see**")
        st.write(
            "Match probability distributions, gamma scores, cluster metrics, "
            "a Venn diagram, a confusion matrix with Precision/Recall/F1, "
            "an interactive blocking explorer, a downloadable PDF report, "
            "and the option to save your trained model as a JSON file."
        )


# =============================================================================
# ── ADVANCED SETUP PAGE ───────────────────────────────────────────────────────
# Upload JSON, choose dataset, choose operation mode, run prediction.
# =============================================================================

def page_advanced_setup():
    _back_button("Back to landing")
    st.title("Advanced Setup: Upload Pre-trained Model")
    st.write(
        "Upload a Splink 4.x model JSON file (produced by "
        "`linker.misc.save_model_to_json()`). The model will be used for "
        "prediction directly, skipping all training steps."
    )
    st.divider()

    # ── JSON upload ──────────────────────────────────────────────────────────
    st.subheader("1. Upload Model JSON")
    uploaded = st.file_uploader(
        "Splink model JSON file",
        type=["json"],
        help="File produced by linker.misc.save_model_to_json(). "
             "Must contain trained m/u probabilities.",
    )
    if uploaded is not None:
        try:
            model_json = json.loads(uploaded.read())     # Parse the JSON
            st.session_state["advanced_json"] = model_json
            # Show a quick summary of what the JSON contains
            comparisons = model_json.get("comparisons", [])
            blocking    = model_json.get("blocking_rules_to_generate_predictions", [])
            st.success(
                f"JSON loaded successfully. "
                f"Found {len(comparisons)} comparisons and "
                f"{len(blocking)} blocking rules."
            )
            with st.expander("JSON summary", expanded=False):
                st.write(f"**Link type:** {model_json.get('link_type', 'not specified')}")
                st.write(f"**Comparison fields:** "
                         f"{', '.join(c.get('output_column_name','?') for c in comparisons)}")
                for i, br in enumerate(blocking):
                    sql = br.get("blocking_rule", str(br)) if isinstance(br, dict) else str(br)
                    st.write(f"**Blocking rule {i}:** `{sql}`")
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")

    st.divider()

    # ── Dataset selection ────────────────────────────────────────────────────
    st.subheader("2. Choose Dataset")
    if not st.session_state["dataset_ready"]:
        if st.button("Load dummy dataset (fake1000)", type="primary"):
            with st.spinner("Building datasets..."):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"]         = fakea
                    st.session_state["fakeb"]         = fakeb
                    st.session_state["dataset_ready"] = True
                    st.success("Dummy dataset loaded.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        fakea = st.session_state["fakea"]
        st.success(
            f"Dataset loaded: {len(fakea):,} records in Dataset A, "
            f"{len(st.session_state['fakeb']):,} in Dataset B."
        )

    st.divider()

    # ── Operation mode ───────────────────────────────────────────────────────
    st.subheader("3. Operation Mode")
    op_mode = st.radio(
        "How to process the data:",
        options=["dedupe", "link_dedupe"],
        format_func=lambda x: "Deduplication only (Dataset A)" if x == "dedupe"
                               else "Link and deduplicate (Dataset A + B)",
        horizontal=True,
        index=0 if st.session_state["advanced_op_mode"] == "dedupe" else 1,
    )
    st.session_state["advanced_op_mode"] = op_mode

    # ── Cluster threshold ────────────────────────────────────────────────────
    threshold = st.slider(
        "Cluster probability threshold",
        0.5, 0.99, 0.8, 0.01,
        help="Records with match_probability above this threshold are grouped into "
             "the same entity cluster.",
    )

    st.divider()

    # ── Run button ───────────────────────────────────────────────────────────
    model_json = st.session_state.get("advanced_json")
    ready      = model_json is not None and st.session_state["dataset_ready"]
    if not ready:
        st.info("Upload a JSON file and load a dataset before running.")

    if ready and st.button("Run prediction from uploaded model", type="primary"):
        with st.spinner("Running prediction (no training needed)..."):
            try:
                fakea  = st.session_state["fakea"]
                fakeb  = st.session_state["fakeb"] if op_mode == "link_dedupe" else None
                results = run_linkage_from_json(
                    model_json=model_json,
                    fakea=fakea,
                    fakeb=fakeb,
                    operation_mode=op_mode,
                    cluster_threshold=threshold,
                )
                # Compute all derived metrics (same as standard flow)
                metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])
                cm      = compute_confusion_matrix(
                    results["df_predict"], fakea, fakeb, op_mode
                )
                ts  = compute_truth_space(results["df_predict"], fakea, fakeb, op_mode)
                crl = compute_crl_score(ts)

                # Coverage matrix for the explorer
                fields     = results["run_config"]["selected_fields"]
                cov_matrix = build_coverage_matrix(results["df_predict"], fields)

                # Store results in session state (same keys as standard flow)
                st.session_state["run1_results"]       = results
                st.session_state["run1_metrics"]       = metrics
                st.session_state["run1_cm"]            = cm
                st.session_state["run1_ts"]            = ts
                st.session_state["run1_crl"]           = crl
                st.session_state["coverage_matrix"]    = cov_matrix
                st.session_state["explorer_toggles"]   = dict(
                    results["run_config"]["blocking_toggles"]
                )
                st.session_state["operation_mode"]     = op_mode
                st.session_state["linkage_type"]       = "probabilistic"

                st.success(
                    f"Prediction complete. "
                    f"{results['n_edges']:,} edges, {results['n_clusters']:,} clusters."
                )
                _go_to(4)    # Jump to analysis page
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# =============================================================================
# ── PAGE 1: CONFIGURE FIELDS, BLOCKING RULES, HYPERPARAMETERS ─────────────────
# =============================================================================

def page_configure():
    _back_button()
    st.title("Step 2: Configure Fields and Blocking Rules")

    if not st.session_state["dataset_ready"]:
        st.warning("Please load a dataset first.")
        if st.button("Go to landing"):
            _go_to(0)
        return

    # Dataset preview
    with st.expander("Dataset A preview", expanded=False):
        st.dataframe(st.session_state["fakea"].head(10), use_container_width=True)

    st.divider()

    # ── Field selection ───────────────────────────────────────────────────────
    st.subheader("Fields to include in comparisons")
    st.write(
        "Select which fields will be used when comparing candidate record pairs. "
        "unique_id and cluster are excluded as they are identifiers."
    )
    field_cols     = st.columns(2)
    selected_fields = []
    for i, field in enumerate(ALL_FIELDS):
        col = field_cols[i % 2]
        if col.checkbox(
            field,
            value=(field in st.session_state["selected_fields"]),
            key=f"field_{field}",
        ):
            selected_fields.append(field)

    if not selected_fields:
        st.error("At least one field must be selected.")
        return
    st.session_state["selected_fields"] = selected_fields

    st.divider()

    # ── Single-field blocking rules ───────────────────────────────────────────
    st.subheader("Single-field blocking rules")
    st.write(
        "Each toggle adds a blocking rule: two records are compared only if they "
        "agree exactly on that field. Each enabled field creates ONE independent "
        "rule (not a combined rule). Toggle two fields = two separate rules. "
        "Enabling more rules increases recall but increases computation time."
    )
    blocking_toggles = {}
    t_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        col = t_cols[i % 3]
        enabled = col.toggle(
            field,
            value=st.session_state["blocking_toggles"].get(field, True),
            key=f"block_{field}",
        )
        blocking_toggles[field] = enabled

    if not any(blocking_toggles.values()) and not st.session_state.get("composite_rules"):
        st.error("At least one blocking rule must be enabled.")
        return
    st.session_state["blocking_toggles"] = blocking_toggles
    active = [f for f, v in blocking_toggles.items() if v]
    st.caption(f"Active single-field rules ({len(active)}): {', '.join(active)}")

    st.divider()

    # ── Composite blocking rules ──────────────────────────────────────────────
    with st.expander("Composite blocking rules (advanced)", expanded=False):
        st.write(
            "Combine two fields into one blocking rule, e.g. "
            "`l.first_name = r.first_name AND l.dob = r.dob`. "
            "This captures pairs only when BOTH fields agree (more precise, fewer pairs)."
        )
        cb1, cb2, cb3 = st.columns([2, 2, 1])
        f1 = cb1.selectbox("Field 1", selected_fields, key="cb_f1")
        f2_opts = [f for f in selected_fields if f != f1]
        if f2_opts:
            f2 = cb2.selectbox("Field 2", f2_opts, key="cb_f2")
        else:
            f2 = None
        if cb3.button("Add composite rule") and f2:
            key = f"{f1}+{f2}"
            st.session_state["composite_rules"][key] = True

        # Show existing composite rules with remove buttons
        cr_keys = list(st.session_state.get("composite_rules", {}).keys())
        for key in cr_keys:
            parts = key.split("+")
            cr1, cr2 = st.columns([4, 1])
            cr1.code(f'l."{parts[0]}" = r."{parts[0]}" AND l."{parts[1]}" = r."{parts[1]}"')
            if cr2.button("Remove", key=f"rm_{key}"):
                del st.session_state["composite_rules"][key]

    # ── Training hyperparameters ──────────────────────────────────────────────
    with st.expander("Training hyperparameters (advanced – probabilistic mode only)",
                     expanded=False):
        st.write(
            "These settings only affect probabilistic linkage. "
            "The defaults work well for most datasets."
        )
        hp = st.session_state.get("hyperparams", {})
        new_hp = {}

        new_hp["max_iterations"] = st.number_input(
            "Max EM iterations",
            min_value=5, max_value=500, value=hp.get("max_iterations", 25), step=5,
            help="Maximum number of iterations before the EM algorithm stops.",
        )
        new_hp["em_convergence"] = st.number_input(
            "EM convergence threshold",
            min_value=1e-8, max_value=0.01, value=hp.get("em_convergence", 0.0001),
            format="%.8f",
            help="Stop EM when the largest parameter change is smaller than this value.",
        )
        new_hp["recall_estimate"] = st.slider(
            "Recall estimate for prior probability",
            min_value=0.1, max_value=0.99,
            value=hp.get("recall_estimate", 0.6), step=0.05,
            help="Fraction of true matches assumed to be captured by the prior blocking rule. "
                 "Lower values make the prior more conservative.",
        )
        st.session_state["hyperparams"] = new_hp

    st.divider()
    if st.button("Continue to operation mode", type="primary"):
        _go_to(2)


# =============================================================================
# ── PAGE 2: OPERATION MODE ────────────────────────────────────────────────────
# =============================================================================

def page_operation():
    _back_button()
    st.title("Step 3: Operation Mode")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deduplication only")
        st.write(
            "Examine Dataset A (1000 records) and identify internal duplicates. "
            "Use when you have one dataset and want to find records referring to "
            "the same entity within it."
        )
        st.write("**Dataset used:** Dataset A (1000 records)")
        if st.button("Select: Deduplication only", use_container_width=True, type="primary"):
            st.session_state["operation_mode"] = "dedupe"
            _go_to(3)

    with c2:
        st.subheader("Link and deduplicate")
        st.write(
            "Link Dataset A (1000 records) with Dataset B (500 records). "
            "Dataset B is a 50% sample of A with controlled errors. "
            "Use when you have two datasets from the same population."
        )
        st.write("**Datasets:** Dataset A (1000) + Dataset B (500)")
        if st.button("Select: Link and deduplicate", use_container_width=True, type="primary"):
            st.session_state["operation_mode"] = "link_dedupe"
            _go_to(3)


# =============================================================================
# ── PAGE 3: LINKAGE TYPE ─────────────────────────────────────────────────────
# =============================================================================

def page_linkage_type():
    _back_button()
    st.title("Step 4: Linkage Type")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deterministic")
        st.write(
            "Records are declared a match if they satisfy at least one active blocking "
            "rule exactly. No training is needed. All matched pairs receive "
            "match_probability = 1.0. Best for high-quality data."
        )
        if st.button("Select: Deterministic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "deterministic"
            _go_to(4)

    with c2:
        st.subheader("Probabilistic")
        st.write(
            "A Fellegi-Sunter model is trained via Expectation-Maximisation. "
            "Each pair gets a match_probability between 0 and 1 based on weighted "
            "comparison of all selected fields. Handles typos and missing values. "
            "Takes 1-2 minutes for training."
        )
        if st.button("Select: Probabilistic", use_container_width=True, type="primary"):
            st.session_state["linkage_type"] = "probabilistic"
            _go_to(4)


# =============================================================================
# ── PAGE 4: ANALYSIS ──────────────────────────────────────────────────────────
# Shared by both flows. Shows all tabs including the new Explorer tab.
# =============================================================================

def page_analysis():
    _back_button()
    flow = st.session_state.get("flow", "standard")

    # ── Guard: ensure required state is present ───────────────────────────────
    if flow == "standard":
        if not st.session_state["dataset_ready"]:
            st.warning("No dataset loaded. Go back to Step 1.")
            if st.button("Go to Step 1"):
                _go_to(0)
            return
        if st.session_state.get("operation_mode") is None:
            st.warning("Operation mode not set. Go back to Step 3.")
            if st.button("Go to Step 3"):
                _go_to(2)
            return
        if st.session_state.get("linkage_type") is None:
            st.warning("Linkage type not set. Go back to Step 4.")
            if st.button("Go to Step 4"):
                _go_to(3)
            return

    if flow == "advanced":
        st.title("Analysis (Advanced Flow)")
    else:
        st.title("Step 5: Run Analysis")

    # ── Configuration summary ─────────────────────────────────────────────────
    run_results = st.session_state.get("run1_results")
    if run_results:
        rc = run_results["run_config"]
        with st.expander("Run configuration", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Operation:** {rc.get('operation_mode','').replace('_',' ').title()}")
            c2.write(f"**Linkage:** {rc.get('linkage_type','').title()}")
            c3.write(f"**Fields:** {', '.join(rc.get('selected_fields',[]))}")
            if rc.get("from_json"):
                st.info("Results produced from uploaded model JSON (no EM training performed).")

    # ── Run / re-run button (standard flow only) ──────────────────────────────
    if flow in ("standard", "upload"):
        run_label = (
            "Re-run analysis with current settings"
            if run_results is not None
            else "Run analysis"
        )
        if st.button(run_label, type="primary"):
            with st.spinner(
                "Running model. Probabilistic training may take 1-2 minutes..."
            ):
                ok = _run_analysis_and_store(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=st.session_state["blocking_toggles"],
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                    hyperparams=st.session_state.get("hyperparams", {}),
                    composite_rules=st.session_state.get("composite_rules", {}),
                )
                if ok:
                    st.success("Analysis complete.")

    if st.session_state.get("run1_results") is None:
        return

    results = st.session_state["run1_results"]
    metrics = st.session_state["run1_metrics"]

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
            st.dataframe(prob_stats, use_container_width=True)

        prob_dist = metrics.get("prob_dist", pd.DataFrame())
        if not prob_dist.empty and len(prob_dist) > 1:
            st.plotly_chart(
                _plotly_bar(prob_dist, "prob_bin", "n_edges",
                            "Match Probability Distribution"),
                use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
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
            st.write("**Singleton vs Multi-record Clusters**")
            st.dataframe(s, use_container_width=True)
            st.caption(
                "High singleton count = many records could not be linked. "
                "Multi-record clusters = found duplicates / cross-dataset matches."
            )

        cs = metrics.get("cluster_sizes", pd.DataFrame())
        if not cs.empty:
            st.plotly_chart(
                _plotly_bar(cs, "n_nodes", "n_clusters", "Cluster Size Distribution"),
                use_container_width=True,
            )
            st.caption(
                "A J-shaped curve (many size-1, few large clusters) is typical. "
                "Very large clusters may indicate over-linking."
            )

        venn = metrics.get("venn", {})
        op   = results["run_config"]["operation_mode"]
        if op != "dedupe" and any(venn.values()):
            st.write("**Dataset Overlap in Clusters**")
            vdf = pd.DataFrame([
                {"Category": "Dataset A only",    "N Clusters": venn.get("a_only", 0)},
                {"Category": "Dataset B only",    "N Clusters": venn.get("b_only", 0)},
                {"Category": "Both A and B",      "N Clusters": venn.get("both_ab", 0)},
            ])
            st.dataframe(vdf, use_container_width=True, hide_index=True)

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
                    use_container_width=True,
                )
        if not c.empty:
            with d2:
                st.plotly_chart(
                    _plotly_bar(c.head(10), "city", "n_records",
                                "Top 10 Cities in Clusters", "#9B59B6"),
                    use_container_width=True,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Interactive Blocking Explorer
    # Mirrors the design from the screen.png mockup:
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

        cov_matrix = st.session_state.get("coverage_matrix")
        if cov_matrix is None or cov_matrix.empty:
            st.info("Run an analysis first to enable the interactive explorer.")
        else:
            # Initialise explorer toggles from run config if not yet set
            run_toggles = results["run_config"].get("blocking_toggles", {})
            if not st.session_state.get("explorer_toggles"):
                st.session_state["explorer_toggles"] = dict(run_toggles)

            # ── Two-column layout ─────────────────────────────────────────────
            col_rules, col_table = st.columns([1, 2.5], gap="large")

            with col_rules:
                st.markdown("**Blocking Rules**")

                # Select All / Clear All buttons
                sa, ca = st.columns(2)
                if sa.button("Select All", key="exp_all"):
                    for f in st.session_state["explorer_toggles"]:
                        st.session_state["explorer_toggles"][f] = True
                    st.rerun()
                if ca.button("Clear All", key="exp_none"):
                    for f in st.session_state["explorer_toggles"]:
                        st.session_state["explorer_toggles"][f] = False
                    st.rerun()

                # Count map: pairs originally generated by each rule
                count_map = {
                    r["rule_sql"]: r["n"]
                    for r in results.get("blocking_counts", [])
                }

                # Rule cards
                new_toggles = {}
                for field, currently_on in st.session_state["explorer_toggles"].items():
                    with st.container(border=True):
                        tc, ic = st.columns([1, 3])
                        new_val = tc.toggle(
                            "", value=currently_on, key=f"exp_tog_{field}"
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
                if new_toggles != st.session_state["explorer_toggles"]:
                    st.session_state["explorer_toggles"] = new_toggles

            with col_table:
                # ── Filter df_predict by active explorer rules ─────────────────
                filtered_df = filter_predict_by_active_rules(
                    results["df_predict"],
                    cov_matrix,
                    st.session_state["explorer_toggles"],
                )

                n_orig     = len(results["df_predict"])
                n_filtered = len(filtered_df)
                n_active   = sum(1 for v in st.session_state["explorer_toggles"].values() if v)
                reduction  = (1 - n_filtered / n_orig) * 100 if n_orig > 0 else 0

                # ── Headline stats ─────────────────────────────────────────────
                hs1, hs2, hs3, hs4 = st.columns(4)
                hs1.metric("Candidate Pairs",  f"{n_filtered:,}")
                hs2.metric("Rules Enabled",    f"{n_active}/{len(st.session_state['explorer_toggles'])}")
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

                    # Colour-code match_probability: show as bar chart column
                    if "match_probability" in display_df.columns:
                        st.dataframe(
                            display_df.style.background_gradient(
                                subset=["match_probability"],
                                cmap="RdYlGn",
                                vmin=0, vmax=1,
                            ),
                            use_container_width=True,
                            height=360,
                        )
                    else:
                        st.dataframe(display_df, use_container_width=True, height=360)

                    st.caption(
                        f"Showing up to 200 of {n_filtered:,} filtered pairs. "
                        "match_probability is colour-coded: red = low confidence, "
                        "green = high confidence."
                    )

            # ── Re-cluster button ─────────────────────────────────────────────
            st.divider()
            exp_thresh = st.slider(
                "Cluster threshold for explorer",
                0.5, 0.99,
                st.session_state.get("explorer_threshold", 0.8),
                0.01,
                key="exp_thresh_slider",
            )
            st.session_state["explorer_threshold"] = exp_thresh

            if st.button("Re-cluster with active rules", type="primary"):
                if filtered_df.empty:
                    st.warning("No pairs to cluster.")
                else:
                    with st.spinner("Re-clustering..."):
                        try:
                            new_clusters = recluster_filtered(
                                df_predict_filtered=filtered_df,
                                fakea=st.session_state["fakea"],
                                fakeb=st.session_state.get("fakeb"),
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
        cm  = st.session_state.get("run1_cm", {})
        ts  = st.session_state.get("run1_ts")
        crl = st.session_state.get("run1_crl", {})

        if not cm or "error" in cm:
            st.info(cm.get("error", "Confusion matrix not yet available."))
        else:
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
                st.dataframe(mdf, use_container_width=True, hide_index=True)

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
                st.plotly_chart(fig_cm, use_container_width=True)

        # Precision-Recall curve (probabilistic only)
        if ts is not None and not ts.empty:
            st.divider()
            st.subheader("Precision-Recall Curve and CRL Score")
            p1, p2 = st.columns(2)
            ts_pr = ts.dropna(subset=["precision_val","recall_val"])
            if not ts_pr.empty:
                with p1:
                    fig_pr = px.line(ts_pr, x="recall_val", y="precision_val",
                                     title="Precision-Recall Curve",
                                     template="simple_white",
                                     color_discrete_sequence=["#1E6EC4"])
                    fig_pr.update_layout(height=300,
                                         xaxis_range=[0,1], yaxis_range=[0,1.05])
                    st.plotly_chart(fig_pr, use_container_width=True)
            ts_fs = ts.dropna(subset=["fstar","match_probability"])
            if not ts_fs.empty:
                with p2:
                    fig_fs = px.line(ts_fs, x="match_probability", y="fstar",
                                     title="F* Score vs Threshold",
                                     template="simple_white",
                                     color_discrete_sequence=["#28A060"])
                    fig_fs.update_layout(height=300,
                                         xaxis_range=[0,1], yaxis_range=[0,1.05])
                    st.plotly_chart(fig_fs, use_container_width=True)
            if crl.get("crl_score") is not None:
                cr1,cr2,cr3,cr4 = st.columns(4)
                cr1.metric("CRL Score", f"{crl.get('crl_score',0):.6f}")
                cr2.metric("t_upper",   str(crl.get("t_upper","N/A")))
                cr3.metric("t_lower",   str(crl.get("t_lower","N/A")))
                cr4.metric("epsilon_z", str(crl.get("epsilon_z","N/A")))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Raw Data
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.subheader("Raw Tables")
        st.write("**df_predict (first 100 rows)**")
        st.dataframe(results["df_predict"].head(100), use_container_width=True)
        st.caption(
            "Each row is a candidate record pair. "
            "gamma_ columns show field-level agreement (1=exact, 0=disagree). "
            "match_key indicates which blocking rule generated this pair."
        )
        st.write("**df_cluster (first 100 rows)**")
        st.dataframe(results["df_cluster"].head(100), use_container_width=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Save trained model as JSON")
    st.write(
        "Download the trained model as a JSON file. You can upload this file "
        "later using Advanced Mode to skip training and go straight to prediction."
    )
    if st.button("Generate model JSON for download"):
        r = st.session_state.get("run1_results", {})
        mp = r.get("model_params", {})
        su = r.get("settings_used", {})
        if not su:
            st.warning("No settings available. Run the analysis first.")
        else:
            try:
                model_json = reconstruct_model_json(su, mp)
                json_bytes = json.dumps(model_json, indent=2).encode("utf-8")
                st.download_button(
                    "Download model JSON",
                    data=json_bytes,
                    file_name="splink_trained_model.json",
                    mime="application/json",
                )
                if mp.get("training_complete"):
                    st.success("Model JSON includes trained m/u probabilities. "
                               "Upload this in Advanced Mode to skip training.")
                else:
                    st.info("Model JSON contains settings only (deterministic run). "
                            "Uploading it in Advanced Mode will run prediction without EM training.")
            except Exception as e:
                st.error(f"Failed to generate model JSON: {e}")

    st.divider()
    st.subheader("Download SeRP-style PDF Report")
    if st.button("Generate PDF report"):
        with st.spinner("Generating report..."):
            try:
                ts_for_pdf = (
                    st.session_state["run1_ts"]
                    if st.session_state.get("run1_ts") is not None
                    else pd.DataFrame()
                )
                pdf_bytes = generate_report(
                    run_label="Run 1",
                    run_config=results["run_config"],
                    metrics=metrics,
                    n_input_records=results["n_input_records"],
                    model_params=results.get("model_params", {}),
                    missingness_a=results.get("missingness_a", {}),
                    missingness_b=results.get("missingness_b", {}),
                    blocking_counts=results.get("blocking_counts", []),
                    unlinkables=results.get("unlinkables", {}),
                    settings_used=results.get("settings_used", {}),
                    confusion_matrix=st.session_state.get("run1_cm", {}),
                    truth_space_df=ts_for_pdf,
                    crl_score=st.session_state.get("run1_crl", {}),
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="linkage_report_run1.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()
    if st.button("Continue to compare runs", type="primary"):
        _go_to(5)


# =============================================================================
# ── PAGE 5: COMPARISON ────────────────────────────────────────────────────────
# =============================================================================

def page_comparison():
    _back_button()
    st.title("Step 6: Compare Runs")

    if st.session_state.get("run1_results") is None:
        st.warning("No Run 1 results. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    run1 = st.session_state["run1_results"]
    m1   = st.session_state["run1_metrics"]

    st.write(
        "Modify blocking rules and re-run to compare how the results change. "
        "Both runs use the same operation mode and linkage type."
    )
    st.divider()

    # Run 1 summary
    st.subheader("Run 1 summary")
    active1 = [f for f, v in run1["run_config"]["blocking_toggles"].items() if v]
    st.caption(f"Blocking rules: {', '.join(active1)}")
    _metric_cards([
        ("Run 1: Edges",         f"{m1['n_edges']:,}"),
        ("Run 1: Clusters",      f"{m1['n_clusters']:,}"),
        ("Run 1: Mean match prob",
         str(m1["match_prob_stats"]["mean_match_prob"].iloc[0])
         if not m1["match_prob_stats"].empty else "N/A"),
    ])

    st.divider()
    st.subheader("Modify blocking rules for Run 2")

    # Initialise Run 2 toggles from Run 1 if not yet set
    if st.session_state.get("run2_blocking_toggles") is None:
        st.session_state["run2_blocking_toggles"] = dict(
            run1["run_config"]["blocking_toggles"]
        )

    r2_toggles = {}
    tc = st.columns(3)
    for i, field in enumerate(st.session_state["selected_fields"]):
        col = tc[i % 3]
        enabled = col.toggle(
            field,
            value=st.session_state["run2_blocking_toggles"].get(field, True),
            key=f"r2_{field}",
        )
        r2_toggles[field] = enabled

    if not any(r2_toggles.values()):
        st.error("At least one blocking rule must be active for Run 2.")
        return
    st.session_state["run2_blocking_toggles"] = r2_toggles

    if st.button("Run analysis with updated blocking rules", type="primary"):
        with st.spinner("Running Run 2..."):
            try:
                run2 = run_linkage(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=r2_toggles,
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                    hyperparams=st.session_state.get("hyperparams", {}),
                )
                m2 = compute_intra_metrics(run2["df_predict"], run2["df_cluster"])
                st.session_state["run2_results"] = run2
                st.session_state["run2_metrics"] = m2
                st.success("Run 2 complete.")
            except Exception as e:
                st.error(f"Run 2 failed: {e}")
                return

    if st.session_state.get("run2_results") is None:
        return

    run2 = st.session_state["run2_results"]
    m2   = st.session_state["run2_metrics"]

    from modules.metrics_engine import compute_inter_metrics
    inter = compute_inter_metrics(
        run1["df_predict"], run2["df_predict"],
        run1["df_cluster"], run2["df_cluster"],
    )

    st.divider()
    st.subheader("Comparison: Run 1 vs Run 2")

    mp1 = (m1["match_prob_stats"]["mean_match_prob"].iloc[0]
           if not m1["match_prob_stats"].empty else 0)
    mp2 = (m2["match_prob_stats"]["mean_match_prob"].iloc[0]
           if not m2["match_prob_stats"].empty else 0)

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Edges",            f"{m2['n_edges']:,}",
               delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
    kc2.metric("Clusters",         f"{m2['n_clusters']:,}",
               delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
    kc3.metric("Mean match prob",  f"{mp2:.4f}",
               delta=f"{mp2 - mp1:+.4f}")

    st.divider()

    # Edge difference table
    ed = inter.get("edge_diff", pd.DataFrame())
    if not ed.empty:
        st.write("**Edge Changes Between Runs**")
        ed_d = ed.set_index("category")["n"].to_dict()
        st.dataframe(pd.DataFrame([
            {"Metric":"Shared edges (both runs)",    "Count": ed_d.get("shared",0)},
            {"Metric":"Edges added in Run 2",        "Count": ed_d.get("added",0)},
            {"Metric":"Edges removed in Run 2",      "Count": ed_d.get("removed",0)},
            {"Metric":"Exact matching clusters",     "Count": inter.get("n_exact_matching_clusters",0)},
            {"Metric":"Partially matching clusters", "Count": inter.get("n_partial_matching_clusters",0)},
        ]), use_container_width=True, hide_index=True)

    # Side-by-side probability distribution
    pd1 = inter.get("prob_dist_run1", pd.DataFrame())
    pd2 = inter.get("prob_dist_run2", pd.DataFrame())
    if not pd1.empty and not pd2.empty:
        pd1["run"] = "Run 1"
        pd2["run"] = "Run 2"
        fig = px.bar(
            pd.concat([pd1, pd2]), x="prob_bin", y="n_edges", color="run",
            barmode="group", title="Match Probability Distribution Comparison",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4","#E55C30"],
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster size comparison
    cs1 = inter.get("cluster_sizes_run1", pd.DataFrame())
    cs2 = inter.get("cluster_sizes_run2", pd.DataFrame())
    if not cs1.empty and not cs2.empty:
        cs1["run"] = "Run 1"
        cs2["run"] = "Run 2"
        fig2 = px.bar(
            pd.concat([cs1, cs2]), x="n_nodes", y="n_clusters", color="run",
            barmode="group", title="Cluster Size Distribution Comparison",
            template="simple_white",
            color_discrete_sequence=["#1E6EC4","#E55C30"],
        )
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    # PDF for Run 2
    st.divider()
    if st.button("Generate PDF report for Run 2"):
        with st.spinner("Generating..."):
            try:
                pdf2 = generate_report(
                    run_label="Run 2",
                    run_config=run2["run_config"], metrics=m2,
                    n_input_records=run2["n_input_records"],
                    model_params=run2.get("model_params",{}),
                    missingness_a=run2.get("missingness_a",{}),
                    missingness_b=run2.get("missingness_b",{}),
                    blocking_counts=run2.get("blocking_counts",[]),
                    unlinkables=run2.get("unlinkables",{}),
                    settings_used=run2.get("settings_used",{}),
                )
                st.download_button("Download Run 2 PDF", data=pdf2,
                                   file_name="linkage_report_run2.pdf",
                                   mime="application/pdf")
            except Exception as e:
                st.error(f"PDF failed: {e}")

    st.divider()
    if st.button("Continue to export", type="primary"):
        _go_to(6)


# =============================================================================
# ── PAGE 6: EXPORT ────────────────────────────────────────────────────────────
# No longer requires Run 2 to be complete.
# =============================================================================

def page_export():
    _back_button()
    st.title("Step 7: Export Cohort")

    # Guard: need at least Run 1
    if st.session_state.get("run1_results") is None:
        st.warning("No analysis results available. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write(
        "Download the final cohort as a CSV. The output contains all original "
        "record fields plus a cluster_id column. Records sharing the same "
        "cluster_id are predicted to represent the same real-world individual."
    )
    st.divider()

    # ── Run selection (Run 2 is optional) ────────────────────────────────────
    st.subheader("Select which run to export")
    run_opts = ["Run 1"]
    if st.session_state.get("run2_results") is not None:
        run_opts.append("Run 2")
    else:
        st.caption("Run 2 is not available. Complete a comparison run to add it as an option.")

    selected_run = st.radio(
        "Export cluster assignments from:", run_opts, horizontal=True
    )
    chosen = (
        st.session_state["run1_results"]
        if selected_run == "Run 1"
        else st.session_state["run2_results"]
    )

    st.divider()

    # ── Build cohort ─────────────────────────────────────────────────────────
    df_cluster = chosen["df_cluster"]
    op_mode    = chosen["run_config"]["operation_mode"]

    if op_mode == "dedupe":
        raw = st.session_state["fakea"].copy()
    else:
        raw = pd.concat(
            [st.session_state["fakea"], st.session_state["fakeb"]],
            ignore_index=True,
        )

    # Merge cluster_id back onto original records
    merge_keys = (
        ["unique_id", "source_dataset"]
        if "source_dataset" in df_cluster.columns
        else ["unique_id"]
    )
    cohort = raw.merge(
        df_cluster[merge_keys + ["cluster_id"]],
        on=merge_keys, how="left",
    )

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records",          f"{len(cohort):,}")
    c2.metric("Distinct cluster IDs",   f"{cohort['cluster_id'].nunique():,}")
    c3.metric("Records with cluster",   f"{cohort['cluster_id'].notna().sum():,}")

    st.subheader("Cohort preview (first 50 rows, sorted by cluster_id)")
    st.dataframe(cohort.sort_values("cluster_id").head(50),
                 use_container_width=True)

    st.divider()
    csv_bytes = cohort.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download cohort CSV ({selected_run})",
        data=csv_bytes,
        file_name=f"cohort_{selected_run.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.info(
        "In the full SAIL deployment version, this page will include direct "
        "provisioning to the SAIL Databank. For this MVP, CSV download only."
    )


# =============================================================================
# ── MAIN ROUTER ───────────────────────────────────────────────────────────────
# =============================================================================

def main():
=======
def main() -> None:
>>>>>>> Stashed changes
    _init_state()
    _render_sidebar()

    flow = st.session_state.get("flow", "standard")
    page = st.session_state["page"]

    # Pages shared across all flows
    shared = {4: page_analysis, 5: page_comparison, 6: page_export}

    if flow == "advanced":
        router = {"advanced_setup": page_advanced_setup, **shared}
        router.get(page, page_advanced_setup)()
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
    elif flow == "upload":
        router = {
            "upload_setup":     page_upload_setup,
            "upload_eda":       page_eda,
            "upload_configure": page_upload_configure,
            2: page_operation,
            3: page_linkage_type,
            **shared,
        }
        router.get(page, page_upload_setup)()
<<<<<<< Updated upstream
    else:
=======

    else:  # standard
>>>>>>> Stashed changes
        router = {
            0: page_landing,
            1: page_configure,
            2: page_operation,
            3: page_linkage_type,
            **shared,
        }
        router.get(page, page_landing)()




# =============================================================================
# ── UPLOAD FLOW: PAGE 1 — DATASET UPLOAD ──────────────────────────────────────
# =============================================================================

def page_upload_setup():
    _back_button("Back to landing")
    st.title("Upload Your Dataset")
    st.write(
        "Upload one or two CSV files. If you upload one dataset the app will "
        "offer deduplication and optionally create a 30% error-introducing sample "
        "for testing linkage. If you upload two datasets both options are available."
    )
    st.divider()

    # ── Primary dataset ───────────────────────────────────────────────────────
    st.subheader("1. Upload primary dataset (Dataset A)")
    file_a = st.file_uploader(
        "Primary CSV file",
        type=["csv"],
        key="file_a_uploader",
        help="The main dataset you want to deduplicate or use as Dataset A for linking.",
    )
    if file_a is not None:
        try:
            df_a = pd.read_csv(file_a)             # Parse the uploaded CSV
            st.success(f"Dataset A loaded: {len(df_a):,} rows, {df_a.shape[1]} columns.")
            with st.expander("Preview (first 5 rows)", expanded=False):
                st.dataframe(df_a.head(5), use_container_width=True)
            st.session_state["upload_raw_df"] = df_a
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.divider()

    # ── Unique ID column selection ────────────────────────────────────────────
    df_a = st.session_state.get("upload_raw_df")
    id_col = None
    if df_a is not None:
        st.subheader("2. Select or generate a unique identifier column")
        id_options  = ["Auto-generate unique_id"] + list(df_a.columns)
        id_choice   = st.selectbox(
            "Which column uniquely identifies each record?",
            options=id_options,
            help="Splink requires every record to have a unique ID. "
                 "If your data does not have one, select 'Auto-generate'.",
        )
        if id_choice != "Auto-generate unique_id":
            id_col = id_choice
        st.session_state["upload_id_col"] = id_col
        st.divider()

    # ── Second dataset or sample ──────────────────────────────────────────────
    st.subheader("3. Second dataset (for linking)")
    upload_mode = st.radio(
        "How do you want to set up Dataset B?",
        options=[
            "Deduplication only (no Dataset B needed)",
            "Upload a second CSV as Dataset B",
            "Create a 30% sample of Dataset A with errors (for testing linking)",
        ],
        index=0,
    )

    file_b = None
    if upload_mode == "Upload a second CSV as Dataset B":
        file_b = st.file_uploader(
            "Second CSV file (Dataset B)",
            type=["csv"],
            key="file_b_uploader",
        )
        if file_b is not None:
            try:
                df_b = pd.read_csv(file_b)
                st.success(f"Dataset B loaded: {len(df_b):,} rows.")
                st.session_state["upload_two_df"] = df_b
                st.session_state["upload_has_two"]  = True
            except Exception as e:
                st.error(f"Could not read second CSV: {e}")

    st.session_state["upload_sample_mode"] = upload_mode

    st.divider()
    if df_a is not None:
        if st.button("Continue to EDA and cleaning", type="primary"):
            _go_to("upload_eda")


# =============================================================================
# ── UPLOAD FLOW: PAGE 2 — EDA AND CLEANING ────────────────────────────────────
# =============================================================================

def page_eda():
    _back_button("Back to upload")
    st.title("EDA and Data Cleaning")

    df_raw = st.session_state.get("upload_raw_df")
    if df_raw is None:
        st.warning("No dataset uploaded. Go back and upload a CSV first.")
        if st.button("Go to upload"):
            _go_to("upload_setup")
        return

    id_col = st.session_state.get("upload_id_col")

    # ── Run EDA pipeline ──────────────────────────────────────────────────────
    st.subheader("Cleaning pipeline")
    st.write(
        "The following steps are applied automatically to your dataset. "
        "Results are shown below."
    )
    if st.button("Run EDA cleaning pipeline", type="primary") or st.session_state.get("upload_clean_df") is not None:
        if st.session_state.get("upload_clean_df") is None:
            with st.spinner("Running EDA pipeline..."):
                df_clean, field_types, name_map, eda_log = run_full_eda(
                    df_raw.copy(), id_col=id_col
                )
                st.session_state["upload_clean_df"]   = df_clean
                st.session_state["upload_field_types"] = field_types
                st.session_state["upload_name_map"]    = name_map
                st.session_state["upload_eda_log"]     = eda_log

                # Find correlation pairs after cleaning
                id_type_cols = [c for c, t in field_types.items() if t == 'id']
                corr_pairs = find_high_correlation_pairs(df_clean, id_type_cols)
                st.session_state["upload_corr_pairs"] = corr_pairs

        df_clean   = st.session_state["upload_clean_df"]
        eda_log    = st.session_state["upload_eda_log"]
        field_types = st.session_state["upload_field_types"]
        name_map   = st.session_state["upload_name_map"]
        corr_pairs = st.session_state.get("upload_corr_pairs", [])
        summary    = eda_log.get("summary", {})

        # ── EDA summary KPIs ─────────────────────────────────────────────────
        st.divider()
        st.subheader("EDA Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Original rows",  f"{summary.get('original_rows',0):,}")
        k2.metric("Rows removed",   f"{summary.get('rows_removed',0):,}",
                  delta=f"-{summary.get('rows_removed',0):,}", delta_color="inverse")
        k3.metric("Remaining rows", f"{summary.get('final_rows',0):,}")
        k4.metric("Cols removed",   f"{summary.get('cols_removed',0):,}",
                  delta=f"-{summary.get('cols_removed',0):,}", delta_color="inverse")

        # Step-by-step detail
        with st.expander("Step-by-step cleaning log", expanded=False):
            # Field name changes
            changed_names = eda_log.get("field_names", {}).get("changed", {})
            if changed_names:
                st.write("**Field names standardised:**")
                for orig, new in changed_names.items():
                    st.write(f"  `{orig}` → `{new}`")
            else:
                st.write("Field names: no changes needed.")

            # Null columns
            dropped_cols = eda_log.get("null_columns_dropped", [])
            st.write(f"**Columns dropped (100% null):** {len(dropped_cols)}"
                     + (f" – {dropped_cols}" if dropped_cols else ""))

            # Null rows
            null_rows = eda_log.get("null_rows_removed", {})
            st.write(f"**Rows removed (100% null):** {null_rows.get('100%_null',0):,}")
            st.write(f"**Rows removed (n-1 null):**  {null_rows.get('n-1_null',0):,}")
            st.write(f"**Rows removed (n-2 null):**  {null_rows.get('n-2_null',0):,}")
            st.write(f"**Duplicate rows removed:**   {eda_log.get('duplicates_removed',0):,}")

            # Date standardisation
            date_fmts = eda_log.get("dates_standardised", {})
            if date_fmts:
                st.write("**Date columns standardised to YYYY-MM-DD:**")
                for col, fmt in date_fmts.items():
                    st.write(f"  `{col}` — parsed as `{fmt}`")

        # ── Detected field types ──────────────────────────────────────────────
        st.divider()
        st.subheader("Detected field types")
        st.write(
            "The app has automatically detected the semantic type of each column. "
            "These drive comparison method and blocking rule suggestions. "
            "You can override them on the next page."
        )
        type_df = pd.DataFrame([
            {"Column": col, "Detected Type": ftype,
             "Comparison Suggestion": suggest_comparison_types({col: ftype}).get(col, "ExactMatch")}
            for col, ftype in field_types.items()
        ])
        st.dataframe(type_df, use_container_width=True, hide_index=True)

        # ── Correlation check ─────────────────────────────────────────────────
        if corr_pairs:
            st.divider()
            st.subheader("High-correlation field pairs")
            st.write(
                "The following pairs of fields have very similar values (correlation >= 0.95). "
                "Keeping both in the comparison model adds little information and slows training. "
                "For each pair, choose which field to keep."
            )
            dropped = set(st.session_state.get("upload_dropped", set()))
            for col_a, col_b, score in corr_pairs:
                with st.container(border=True):
                    cc1, cc2, cc3 = st.columns([2, 2, 2])
                    cc1.write(f"**{col_a}**")
                    cc2.write(f"**{col_b}**")
                    cc3.write(f"Score: `{score:.4f}`")
                    keep = st.radio(
                        f"Keep which field? (pair: {col_a} vs {col_b})",
                        options=[col_a, col_b, "Keep both"],
                        horizontal=True,
                        key=f"corr_{col_a}_{col_b}",
                    )
                    if keep == col_a:
                        dropped.add(col_b)
                    elif keep == col_b:
                        dropped.add(col_a)
                    # "Keep both" removes neither

            st.session_state["upload_dropped"] = dropped

            if dropped:
                df_clean = df_clean.drop(
                    columns=[c for c in dropped if c in df_clean.columns]
                )
                st.session_state["upload_clean_df"] = df_clean
                # Re-sync field_types
                field_types = {c: t for c, t in field_types.items() if c not in dropped}
                st.session_state["upload_field_types"] = field_types
                st.caption(f"Columns removed from correlation check: {', '.join(dropped)}")

        # ── Cleaned data preview ──────────────────────────────────────────────
        st.divider()
        st.subheader("Cleaned dataset preview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        st.caption(f"Shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

        # ── Download clean CSV ────────────────────────────────────────────────
        csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

        st.divider()
        if st.button("Continue to field configuration", type="primary"):
            _go_to("upload_configure")


# =============================================================================
# ── UPLOAD FLOW: PAGE 3 — FIELD CONFIG AND BLOCKING RULES ─────────────────────
# =============================================================================

def page_upload_configure():
    _back_button("Back to EDA")
    st.title("Configure Fields and Blocking Rules")

    df_clean    = st.session_state.get("upload_clean_df")
    field_types = st.session_state.get("upload_field_types", {})

    if df_clean is None:
        st.warning("No cleaned dataset found. Go back to EDA first.")
        if st.button("Go to EDA"):
            _go_to("upload_eda")
        return

    # ── Unique ID configuration ───────────────────────────────────────────────
    st.subheader("1. Unique identifier")
    id_cols    = [c for c, t in field_types.items() if t == 'id']
    non_id_cols = [c for c in df_clean.columns if c not in id_cols]

    id_options = id_cols + non_id_cols
    current_id  = st.session_state.get("upload_id_col") or (id_cols[0] if id_cols else None)
    chosen_id   = st.selectbox(
        "Unique ID column (will be added automatically if not found):",
        options=["[Auto-generate]"] + id_options,
        index=0 if current_id is None else (id_options.index(current_id) + 1
                                             if current_id in id_options else 0),
    )
    if chosen_id == "[Auto-generate]":
        # Add a sequential unique_id column if user wants auto-generation
        if "unique_id" not in df_clean.columns:
            df_clean = df_clean.copy()
            df_clean.insert(0, "unique_id", range(1, len(df_clean) + 1))
            df_clean["unique_id"] = "R" + df_clean["unique_id"].astype(str)
            st.session_state["upload_clean_df"] = df_clean
        chosen_id = "unique_id"
        field_types["unique_id"] = "id"
        st.session_state["upload_field_types"] = field_types

    st.session_state["upload_id_col"] = chosen_id
    st.caption(f"Using `{chosen_id}` as the unique identifier.")

    st.divider()

    # ── Comparison field selection and type override ───────────────────────────
    st.subheader("2. Comparison fields and types")
    st.write(
        "Select which fields to include in the linkage model. "
        "Override the detected comparison type if needed."
    )
    comp_suggestions = suggest_comparison_types(field_types)
    selected_fields  = []
    comp_types       = {}
    COMP_OPTIONS     = ["NameComparison", "DateOfBirthComparison", "ExactMatch"]

    for col, suggested_type in comp_suggestions.items():
        if col == chosen_id:
            continue
        row1, row2, row3 = st.columns([2, 2, 1])
        include = row1.checkbox(col, value=True, key=f"up_inc_{col}")
        if include:
            chosen_type = row2.selectbox(
                "",
                options=COMP_OPTIONS,
                index=COMP_OPTIONS.index(suggested_type) if suggested_type in COMP_OPTIONS else 2,
                key=f"up_type_{col}",
                label_visibility="collapsed",
            )
            row3.caption(field_types.get(col, "unknown"))
            selected_fields.append(col)
            comp_types[col] = chosen_type

    st.session_state["upload_selected_fields"] = selected_fields
    st.session_state["upload_comp_types"]      = comp_types

    st.divider()

    # ── Blocking rules (suggested + toggle) ───────────────────────────────────
    st.subheader("3. Blocking rules")
    st.write(
        "Blocking rules reduce the comparison space. "
        "Suggested rules are shown enabled; toggle any off to exclude them."
    )
    block_suggestions = suggest_blocking_rules(field_types)
    block_toggles     = {}
    bt_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        col = bt_cols[i % 3]
        suggested = block_suggestions.get(field, False)
        enabled = col.toggle(
            field,
            value=st.session_state.get("upload_block_toggles", {}).get(field, suggested),
            key=f"up_block_{field}",
        )
        block_toggles[field] = enabled

    if not any(block_toggles.values()):
        st.error("At least one blocking rule must be enabled.")
        return
    st.session_state["upload_block_toggles"] = block_toggles

    # ── Composite blocking rules ──────────────────────────────────────────────
    with st.expander("Add composite blocking rules", expanded=False):
        cb1, cb2, cb3 = st.columns([2, 2, 1])
        cf1 = cb1.selectbox("Field 1", selected_fields, key="up_cb_f1")
        cf2_opts = [f for f in selected_fields if f != cf1]
        if cf2_opts:
            cf2 = cb2.selectbox("Field 2", cf2_opts, key="up_cb_f2")
        else:
            cf2 = None
        if cb3.button("Add", key="up_cb_add") and cf2:
            key = f"{cf1}+{cf2}"
            st.session_state["upload_comp_rules"][key] = True

        for key in list(st.session_state.get("upload_comp_rules", {}).keys()):
            parts = key.split("+")
            cr1, cr2 = st.columns([4, 1])
            cr1.code(f'l."{parts[0]}" = r."{parts[0]}" AND l."{parts[1]}" = r."{parts[1]}"')
            if cr2.button("Remove", key=f"up_rm_{key}"):
                del st.session_state["upload_comp_rules"][key]

    st.divider()

    # ── Handle Dataset B ──────────────────────────────────────────────────────
    st.subheader("4. Dataset B setup")
    sample_mode = st.session_state.get("upload_sample_mode", "")

    if "Create a 30% sample" in sample_mode:
        st.write(
            "A 30% sample of Dataset A will be created with controlled errors "
            "introduced (typos in names, missing dates, email variations, etc.). "
            "This simulates a second data source for testing linkage."
        )
        if st.button("Generate error-introducing sample (Dataset B)"):
            with st.spinner("Creating error sample..."):
                try:
                    df_b = introduce_errors_for_sample(
                        df_clean, field_types, sample_frac=0.3, seed=42
                    )
                    st.session_state["upload_df_b"] = df_b
                    st.success(f"Dataset B created: {len(df_b):,} records.")
                    st.dataframe(df_b.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed: {e}")
    elif "two" in sample_mode.lower() or st.session_state.get("upload_has_two"):
        raw_b = st.session_state.get("upload_two_df")
        if raw_b is not None:
            st.write("Running EDA cleaning on Dataset B...")
            if st.session_state.get("upload_df_b") is None:
                try:
                    df_b_clean, _, _, _ = run_full_eda(raw_b.copy(), id_col=None)
                    df_b_clean["source_dataset"] = "B"
                    st.session_state["upload_df_b"] = df_b_clean
                    st.success(f"Dataset B cleaned: {len(df_b_clean):,} rows.")
                except Exception as e:
                    st.error(f"Dataset B cleaning failed: {e}")
    else:
        st.write("Deduplication only — no Dataset B required.")

    st.divider()

    # ── Finalise: push datasets into shared session state ─────────────────────
    if st.button("Continue to operation mode", type="primary"):
        # Store cleaned dataset as fakea (shared key used by analysis pages)
        df_a = df_clean.copy()
        if "source_dataset" not in df_a.columns:
            df_a["source_dataset"] = "A"             # Tag source dataset

        # Ensure unique_id column exists
        if "unique_id" not in df_a.columns and chosen_id in df_a.columns:
            df_a = df_a.rename(columns={chosen_id: "unique_id"})

        st.session_state["fakea"] = df_a
        st.session_state["fakeb"] = st.session_state.get("upload_df_b")
        st.session_state["dataset_ready"]     = True

        # Push field/blocking settings into the shared keys used by analysis
        st.session_state["selected_fields"]   = selected_fields
        st.session_state["blocking_toggles"]  = block_toggles
        st.session_state["composite_rules"]   = st.session_state.get("upload_comp_rules", {})

        _go_to(2)    # Go to shared operation mode page


if __name__ == "__main__":
    main()
