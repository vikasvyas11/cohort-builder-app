# flows/p_standard.py
# Standard flow: Configure, Operation Mode, Linkage Type.
import streamlit as st
from modules.eda_engine import is_selective_field
from utils.state import ALL_FIELDS
from utils.nav import _back_button, _go_to



def page_configure() -> None:
    _back_button()
    st.title("Step 2: Configure Fields and Blocking Rules")

    if not st.session_state["dataset_ready"]:
        st.warning("Please load a dataset first.")
        if st.button("Go to landing"):
            _go_to(0)
        return

    with st.expander("Dataset A preview", expanded=False):
        st.dataframe(st.session_state["dataset_a"].head(10), width="stretch")
    st.divider()

    st.subheader("Fields to include in comparisons")
    st.write(
        "Select which fields to compare. unique_id and cluster are excluded "
        "as they are identifiers, not linkage features."
    )
    # Use the loaded dataset's own columns, so this page works for the voter
    # registry, uploaded datasets and the demo people file alike.
    dataset_a = st.session_state.get("dataset_a")
    EXCLUDE = {"unique_id", "cluster", "source_dataset"}
    available_fields = (
        [c for c in dataset_a.columns if c not in EXCLUDE]
        if dataset_a is not None
        else list(ALL_FIELDS)   # fallback if somehow dataset_a is missing
    )

    field_cols = st.columns(2)
    selected_fields = []
    for i, field in enumerate(available_fields):
        col = field_cols[i % 2]
        # Default: on if previously selected, or on if it was in the initial set
        default_on = st.session_state["selected_fields"] and field in st.session_state["selected_fields"]
        if col.checkbox(field, value=bool(default_on), key=f"field_{field}"):
            selected_fields.append(field)

    if not selected_fields:
        st.error("At least one field must be selected.")
        return
    st.session_state["selected_fields"] = selected_fields
    st.divider()

    st.subheader("Single-field blocking rules")
    st.write(
        "Each toggle creates one independent blocking rule. "
        "Two records are compared only if they agree exactly on at least one "
        "active blocking field."
    )
    blocking_toggles = {}
    t_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        enabled = t_cols[i % 3].toggle(
            field,
            value=st.session_state["blocking_toggles"].get(
                field, dataset_a is None or is_selective_field(dataset_a[field])),
            key=f"block_{field}",
        )
        blocking_toggles[field] = enabled

    has_single    = any(blocking_toggles.values())
    has_composite = bool(st.session_state.get("composite_rules"))
    if not has_single and not has_composite:
        st.error("At least one blocking rule (single-field or composite) must be defined.")
        return
    st.session_state["blocking_toggles"] = blocking_toggles
    st.caption(f"Active rules: {', '.join(f for f, v in blocking_toggles.items() if v)}")

    st.write("**How should active blocking rules combine?**")
    blocking_mode = st.radio(
        "Blocking mode",
        ["OR", "AND"],
        format_func=lambda x: (
            "OR — match if ANY active field agrees (default, higher recall)"
            if x == "OR" else
            "AND — match only if ALL active fields agree together (stricter, higher precision)"
        ),
        horizontal=False,
        index=0 if st.session_state.get("blocking_mode", "OR") == "OR" else 1,
        key="blocking_mode_choice",
        label_visibility="collapsed",
    )
    st.session_state["blocking_mode"] = blocking_mode
    if blocking_mode == "AND":
        st.caption(
            "⚠️ AND mode combines every currently-enabled field into a single rule. "
            "This is much stricter than OR — a pair must agree on every active field "
            "simultaneously to be considered a candidate at all. The Interactive "
            "Blocking Explorer's rule-cascade waterfall (Step 5) only applies to "
            "OR-mode runs, since AND mode has no independent rules to cascade between."
        )

    with st.expander("Composite blocking rules (advanced)", expanded=False):
        st.write(
            "Combine two or three fields into a single AND rule. "
            "Composite-only configurations (no single-field rules toggled on) are valid."
        )
        cb1, cb2, cb3, cb4 = st.columns([2, 2, 2, 1])
        f1 = cb1.selectbox("Field 1", selected_fields, key="cb_f1")
        f2_opts = [f for f in selected_fields if f != f1]
        f2 = cb2.selectbox("Field 2", f2_opts, key="cb_f2") if f2_opts else None
        f3_opts = ["(none)"] + [f for f in selected_fields if f not in (f1, f2)]
        f3_sel  = cb3.selectbox("Field 3 (optional)", f3_opts, key="cb_f3")
        f3 = None if f3_sel == "(none)" else f3_sel
        if cb4.button("Add", key="cb_add") and f2:
            rule_key = f"{f1}+{f2}" + (f"+{f3}" if f3 else "")
            st.session_state["composite_rules"][rule_key] = True
            # A composite rule only means anything if the fields inside it
            # aren't ALSO active as independent single-field rules — in OR
            # mode, a loose single-field rule matches a strict superset of
            # what the composite AND would ever match, silently making the
            # composite contribute nothing. Switch those individual toggles
            # off here so the AND condition actually takes effect.
            for f in rule_key.split("+"):
                st.session_state[f"block_{f}"] = False
            st.rerun()

        _composite_fields = set()
        for key in list(st.session_state.get("composite_rules", {}).keys()):
            parts = key.split("+")
            _composite_fields.update(parts)
            sql_parts = " AND ".join(f'l."{p}" = r."{p}"' for p in parts)
            cr1, cr2 = st.columns([4, 1])
            cr1.code(sql_parts)
            if cr2.button("Remove", key=f"rm_{key}"):
                del st.session_state["composite_rules"][key]

        _redundant = sorted(f for f in _composite_fields if blocking_toggles.get(f))
        if _redundant:
            st.warning(
                f"⚠️ {', '.join(_redundant)} still has its individual blocking rule "
                "enabled above, in addition to being part of a composite rule. "
                "In OR mode, that individual rule alone matches everything the "
                "composite AND rule would — cancelling out the composite's "
                "stricter matching. Turn off the individual toggle(s) for these "
                "fields (above) if you want the composite AND to actually apply."
            )

    with st.expander("Training hyperparameters (probabilistic mode only)", expanded=False):
        hp = st.session_state.get("hyperparams", {})
        nhp = {}
        nhp["max_iterations"] = st.number_input(
            "Max EM iterations", 5, 500,
            value=hp.get("max_iterations", 25), step=5)
        nhp["em_convergence"] = st.number_input(
            "EM convergence", 1e-8, 0.01,
            value=hp.get("em_convergence", 0.0001), format="%.8f")
        nhp["recall_estimate"] = st.slider(
            "Recall estimate for prior", 0.1, 0.99,
            value=hp.get("recall_estimate", 0.6), step=0.05)
        st.session_state["hyperparams"] = nhp

    st.divider()
    if st.button("Continue to operation mode", type="primary"):
        _go_to(2)


def page_operation() -> None:
    _back_button()
    dataset_a = st.session_state.get("dataset_a")

    # Track dynamic error configuration rules
    if "custom_error_rates" not in st.session_state:
        st.session_state["custom_error_rates"] = {}

    n_a = f"{len(dataset_a):,}" if dataset_a is not None else "N/A"

    st.title("Step 3: Operation Mode")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deduplication only")
        st.write(
            "Examine Dataset A and identify internal duplicates. "
            "Use when you have one dataset and want to remove duplicate records."
        )
        st.write(f"**Dataset used:** Dataset A ({n_a} records)")
        if st.button("Select: Deduplication only", width="stretch", type="primary"):
            st.session_state["operation_mode"] = "dedupe"
            st.session_state["dataset_b"] = None
            _go_to(3)

    with c2:
        st.subheader("Link and deduplicate")
        st.write(
            "Link Dataset A with Dataset B across two separate sources. "
            "Dataset B is a sample of Dataset A with typos, shifted dates and other "
            "realistic errors added, so a linkage has something to find."
        )

        std_fb = st.session_state.get("std_dataset_b")   # pre-built dataset_b from data_builder
        current_fb = st.session_state.get("dataset_b")

        if std_fb is not None and current_fb is None:
            # Auto-use the staged dataset_b from build_datasets()
            st.info(
                f"Dataset B available: {len(std_fb):,} records "
                "(a damaged sample of Dataset A, built with the app's corruption engine)."
            )
            st.write(
                "To customise error rates for Dataset B, go to the "
                "**EDA and Cleaning** page in the Upload flow."
            )
            if st.button("Use pre-built Dataset B", width="stretch"):
                st.session_state["dataset_b"] = std_fb
                st.rerun()

        elif current_fb is not None:
            n_b = f"{len(current_fb):,}"
            st.success(f"Dataset B ready: {n_b} records.")
        else:
            # No pre-built Dataset B — offer dynamic error-introduction generation.
            # This covers the voter-registry flow, where std_dataset_b is None.
            st.info(
                "No Dataset B available yet. "
                "Generate a derived sample from Dataset A with controlled errors below."
            )
            if dataset_a is not None:
                from modules.corruption import CorruptionSettings, make_noisy_copy
                from modules.eda_engine import infer_field_types

                EXCLUDE_B = {"unique_id", "cluster", "source_dataset"}
                _ftypes = st.session_state.get("voter_field_types") or infer_field_types(dataset_a.columns)
                _eligible = [c for c in dataset_a.columns
                             if c not in EXCLUDE_B
                             and _ftypes.get(c, "text") not in ("id",)]
                _blocking_toggles = st.session_state.get("blocking_toggles", {})

                with st.expander("Configure fields for error introduction (Dataset B)", expanded=True):
                    st.caption(
                        "Choose which fields get damaged and how hard. Fields you selected "
                        "as blocking rules in Step 2 are pre-checked. Note that exact-match "
                        "blocking on a heavily damaged field will miss most true pairs."
                    )
                    _sfrac = st.slider(
                        "Sample fraction", 0.1, 0.9, 0.3, 0.05, key="std_nc_sfrac",
                        help="Fraction of Dataset A records to include in Dataset B."
                    )
                    _rate = st.slider(
                        "Corruption rate — share of populated values changed per selected field",
                        0.05, 1.0, 0.3, 0.05, key="std_nc_rate",
                    )
                    _mc1, _mc2 = st.columns(2)
                    _letters_to_change = _mc1.slider(
                        "Error magnitude — letters changed per damaged value (text fields)",
                        1, 5, 1, key="std_nc_letters",
                        help="How many characters are replaced in each damaged text value."
                    )
                    _year_shift = _mc2.number_input(
                        "Error magnitude — year shift (year-only fields, e.g. birth_year)",
                        min_value=1, max_value=10, value=1, step=1, key="std_nc_year_shift",
                        help="Years are shifted by +/- this amount. Full dates move by a "
                             "day or two, swap day and month, or go missing."
                    )
                    _sel_all, _desel_all, _ = st.columns([1, 1, 6])
                    if _sel_all.button("Select All",   key="nc_sel_all"):
                        for _f in _eligible:
                            st.session_state[f"nc_chk_{_f}"] = True
                    if _desel_all.button("Deselect All", key="nc_desel_all"):
                        for _f in _eligible:
                            st.session_state[f"nc_chk_{_f}"] = False

                    _active_fields = []
                    _cols3 = st.columns(3)
                    for _i, _col in enumerate(_eligible):
                        _ftype = _ftypes.get(_col, "text")
                        # Auto-populate: pre-check any field the user selected as a
                        # blocking rule in Step 2, unless they've already toggled it here.
                        _default_checked = st.session_state.get(
                            f"nc_chk_{_col}", _blocking_toggles.get(_col, False)
                        )
                        _chk = _cols3[_i % 3].checkbox(
                            f"`{_col}` ({_ftype})",
                            value=_default_checked,
                            key=f"nc_chk_{_col}",
                        )
                        if _chk:
                            _active_fields.append(_col)

                    st.caption(
                        f"Errors will be introduced into: "
                        f"{', '.join(_active_fields) if _active_fields else 'no fields selected (clean sample)'}"
                    )

                if st.button("Generate Dataset B", type="primary", key="std_nc_gen_b"):
                    with st.spinner("Generating derived Dataset B…"):
                        try:
                            _df_b = make_noisy_copy(
                                dataset_a,
                                field_rates={f: _rate for f in _active_fields},
                                sample_frac=_sfrac,
                                seed=42,
                                field_types=_ftypes,
                                settings=CorruptionSettings(
                                    text_edits=_letters_to_change, year_shift=_year_shift),
                            )
                            st.session_state["dataset_b"]    = _df_b
                            st.session_state["std_dataset_b"] = _df_b
                            st.success(
                                f"Dataset B generated: {len(_df_b):,} records. "
                                f"Errors in: {', '.join(_active_fields) or 'none (clean sample)'}."
                            )
                            _post_eda_cols = st.columns(3)
                            _post_eda_cols[0].metric("Dataset B rows", f"{len(_df_b):,}")
                            _post_eda_cols[1].metric("Columns", f"{_df_b.shape[1]}")
                            _post_eda_cols[2].metric(
                                "Fields with errors", f"{len(_active_fields)}"
                            )
                        except Exception as _e:
                            st.error(f"Generation failed: {_e}")

        current_fb = st.session_state.get("dataset_b")
        btn_disabled = current_fb is None
        if st.button("Select: Link and deduplicate", width="stretch",
                     type="primary", disabled=btn_disabled):
            st.session_state["operation_mode"] = "link_dedupe"
            _go_to(3)


def page_linkage_type() -> None:
    _back_button()
    st.title("Step 4: Linkage Type")

    # ── Pre-flight candidate pair estimate ──────────────────────────────────
    # Real DuckDB join count for the current blocking configuration, run
    # BEFORE the user commits to a linkage method — turns a silent
    # out-of-memory crash mid-run into an informed choice to tighten
    # blocking first.
    try:
        from modules.splink_runner import estimate_candidate_pairs
        _dataset_a = st.session_state.get("dataset_a")
        _dataset_b = st.session_state.get("std_dataset_b") or st.session_state.get("dataset_b")
        with st.spinner("Estimating candidate pairs for the current blocking rules..."):
            _est_pairs = estimate_candidate_pairs(
                _dataset_a, _dataset_b,
                st.session_state["selected_fields"],
                st.session_state["blocking_toggles"],
                st.session_state.get("blocking_mode", "OR"),
                st.session_state["operation_mode"],
            )
        if _est_pairs > 5_000_000:
            st.error(
                f"⚠️ Estimated **{_est_pairs:,}** candidate pairs. This is very likely "
                "to exhaust available memory. Go back and tighten blocking rules "
                "(prefer AND mode, or higher-selectivity fields) before running."
            )
        elif _est_pairs > 500_000:
            st.warning(
                f"Estimated **{_est_pairs:,}** candidate pairs. This may be slow and "
                "memory-intensive — consider tightening blocking if the run struggles."
            )
        else:
            st.success(f"Estimated **{_est_pairs:,}** candidate pairs — looks manageable.")
    except Exception as _pe_err:
        st.caption(f"(Could not estimate candidate pairs: {_pe_err})")

    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Deterministic")
        st.write(
            "Records declared a match if they satisfy at least one active blocking rule. "
            "No training. All matched pairs get match_probability = 1.0. Best for high-quality data."
        )
        if st.button("Select: Deterministic", width="stretch", type="primary"):
            st.session_state["linkage_type"] = "deterministic"
            _go_to(4)

    with c2:
        st.subheader("Probabilistic")
        st.write(
            "Fellegi-Sunter model trained by EM. Each pair receives a match_probability "
            "0-1 based on field-level agreement. Handles typos and missing values. "
            "Takes 1-2 minutes for training."
        )
        if st.button("Select: Probabilistic", width="stretch", type="primary"):
            st.session_state["linkage_type"] = "probabilistic"
            _go_to(4)