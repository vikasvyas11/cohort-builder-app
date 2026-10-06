# flows/p_analysis.py
# Analysis page — shared by standard, upload, and advanced flows.
# Shows data summary, run button, tabbed results (edge metrics, cluster metrics,
# demographics, blocking explorer, cluster studio, confusion matrix, raw data).
# Also provides the model JSON and PDF download buttons.

import json
from datetime import datetime

import streamlit as st

from modules.splink_runner import (
    reconstruct_model_json,
)
from utils.helpers import (
    _run_analysis_and_store, render_assistant_link, render_report_download,
)
from flows.run_results import render_run_results
from utils.nav import _back_button, _go_to


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
    elif flow == "upload":
        st.title("Step 5: Run Analysis (Upload Data)")
    else:
        st.title("Step 5: Run Analysis")

    # ── Dataset-specific warnings ─────────────────────────────────────────────
    _linkage_type = st.session_state.get("linkage_type")
    _is_nc = (st.session_state.get("voter_field_types") is not None
              and st.session_state.get("flow") == "standard")

    if _is_nc:
        st.warning(
            "⚠️ **Voter-registry dataset detected.** "
            "Large registries can use a lot of memory during analysis, and hosted "
            "Streamlit instances have tight limits. If a run crashes, generate "
            "fewer synthetic voters or select fewer comparison fields."
        )

    if _linkage_type == "deterministic":
        st.info(
            "ℹ️ **Deterministic mode** generates all candidate pairs matching any "
            "blocking rule and assigns each pair match_probability = 1.0. "
            "With low-cardinality blocking fields (e.g. gender, county) this can "
            "produce a very large number of pairs and **may take significantly longer** "
            "than probabilistic mode. Consider using high-selectivity fields "
            "(e.g. voter_reg_num, ncid, last_name) as blocking rules."
        )

    # ══════════════════════════════════════════════════════════════════════
    # UPLOAD FLOW: show what dataset / fields are actually loaded and
    # validate them against the real data before allowing a run.
    # ══════════════════════════════════════════════════════════════════════
    if flow == "upload":
        _uf_dataset_a = st.session_state.get("dataset_a")
        _uf_sel   = st.session_state.get("selected_fields", [])
        _uf_block = st.session_state.get("blocking_toggles", {})

        if _uf_dataset_a is None:
            st.error(
                "No uploaded dataset is loaded. "
                "Complete the upload flow (upload file → EDA → configure fields) first."
            )
            if st.button("Go to Upload", key="uf_goto_upload"):
                _go_to("upload_setup")
            return

        # Compute which selected fields actually exist in the data
        _uf_actual   = set(_uf_dataset_a.columns)
        _uf_valid    = [f for f in _uf_sel if f in _uf_actual]
        _uf_missing  = [f for f in _uf_sel if f not in _uf_actual]

        if not _uf_valid:
            # No valid fields at all — redirect to configure
            st.error(
                "None of the configured comparison fields exist in the uploaded dataset. "
                "Go back to Configure Fields and select fields that match your uploaded data."
            )
            c_err1, c_err2 = st.columns(2)
            c_err1.write(f"**Configured fields:** {', '.join(_uf_sel) or 'none'}")
            c_err2.write(f"**Dataset columns:** {', '.join(sorted(_uf_actual)[:10])}{'...' if len(_uf_actual)>10 else ''}")
            if st.button("Go to Configure Fields", key="uf_goto_cfg"):
                _go_to("upload_configure")
            return

        # Data summary box — always visible so user can see what's loaded
        with st.container(border=True):
            st.markdown("**Uploaded dataset loaded**")
            ds1, ds2, ds3, ds4 = st.columns(4)
            ds1.metric("Rows (Dataset A)",   f"{len(_uf_dataset_a):,}")
            ds2.metric("Columns",            f"{_uf_dataset_a.shape[1]:,}")
            ds3.metric("Fields for linkage", f"{len(_uf_valid)}")
            _uf_fb = st.session_state.get("dataset_b")
            ds4.metric("Dataset B rows",
                       f"{len(_uf_fb):,}" if _uf_fb is not None else "None")

            st.caption(
                f"**Comparison fields:** {', '.join(_uf_valid)}"
            )
            if _uf_missing:
                st.warning(
                    f"These configured fields are NOT in the dataset and will be skipped: "
                    f"{', '.join(_uf_missing)}"
                )
            _uf_active_block = [f for f, v in _uf_block.items() if v and f in _uf_actual]
            st.caption(f"**Active blocking rules:** {', '.join(_uf_active_block) or 'none'}")

            if not _uf_active_block:
                st.error(
                    "No active blocking rules match columns in the dataset. "
                    "Go back to Configure Fields."
                )
                if st.button("Go to Configure Fields", key="uf_cfg_block"):
                    _go_to("upload_configure")
                return

        # If cached results exist from a different run (e.g. standard flow),
        # make it very clear the user needs to click Run to get upload results
        _cached = st.session_state.get("run1_results")
        if _cached:
            _prev_fields = _cached.get("run_config", {}).get("selected_fields", [])
            if set(_prev_fields) != set(_uf_valid):
                st.info(
                    "The results shown below are from a **previous run** with different "
                    "fields or data. Click **Run analysis** to run with your uploaded dataset."
                )

    # ── Configuration summary (previous run) ─────────────────────────────────
    run_results = st.session_state.get("run1_results")
    if run_results and flow != "upload":
        rc = run_results["run_config"]
        with st.expander("Run configuration", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Operation:** {rc.get('operation_mode','').replace('_',' ').title()}")
            c2.write(f"**Linkage:** {rc.get('linkage_type','').title()}")
            c3.write(f"**Fields:** {', '.join(rc.get('selected_fields',[]))}")
            if rc.get("from_json"):
                st.info("Results produced from uploaded model JSON.")

    # ── Run / re-run button ───────────────────────────────────────────────────
    if flow in ("standard", "upload"):
        run_label = (
            "Run analysis"
            if run_results is None
            else "Re-run analysis with current settings"
        )
        # For upload flow always label it clearly
        if flow == "upload":
            run_label = f"Run analysis on uploaded dataset ({len(st.session_state['dataset_a']):,} rows)"

        if st.button(run_label, type="primary"):
            with st.spinner(
                "Running model. Probabilistic training may take 1-2 minutes..."
            ):
                ok = _run_analysis_and_store(
                    dataset_a=st.session_state["dataset_a"],
                    dataset_b=st.session_state["dataset_b"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=st.session_state["blocking_toggles"],
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                    hyperparams=st.session_state.get("hyperparams", {}),
                    composite_rules=st.session_state.get("composite_rules", {}),
                )
                if ok:
                    st.success("Analysis complete.")

    render_run_results("run1")
    if st.session_state.get("run1_results") is None:
        return

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
                model_json = reconstruct_model_json(
                    su, mp,
                    linkage_type=r.get("run_config", {}).get("linkage_type", "probabilistic"),
                )
                json_bytes = json.dumps(model_json, indent=2).encode("utf-8")
                st.download_button(
                    "Download model JSON",
                    data=json_bytes,
                    file_name=f"splink_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
    render_assistant_link("run1")

    st.divider()
    st.subheader("Report")
    render_report_download("Run 1", "run1")

    st.divider()
    if st.button("Continue to compare runs", type="primary"):
        _go_to(5)


# =============================================================================
# ── PAGE 5: COMPARISON ────────────────────────────────────────────────────────
# =============================================================================

