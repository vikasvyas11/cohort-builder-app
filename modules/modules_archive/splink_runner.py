# =============================================================================
# modules/splink_runner.py
# PURPOSE: Wrap Splink's linkage and deduplication workflow.
#          Mirrors logic from:
#            - linkage_workflow/templates/1_train_model_deterministic.ipynb
#            - linkage_workflow/templates/1_train_model_probabilistic.ipynb
#          Enhanced to extract model parameters, missingness stats, and
#          blocking-rule comparison counts for the SeRP-style PDF report.
# =============================================================================

import io
import math
import multiprocessing
import tempfile
from typing import Optional

import duckdb
import pandas as pd

from splink import DuckDBAPI, Linker
import splink.comparison_library as cl
import splink.blocking_rule_library as brl

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → COMPARISON mapping
# Maps each dataset column to an appropriate Splink comparison strategy.
# NameComparison uses Jaro-Winkler fuzzy matching (good for typos in names).
# DateOfBirthComparison handles transpositions and date-range differences.
# ExactMatch is used for categorical fields (city, email, gender, postcode).
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_COMPARISONS = {
    "first_name": lambda: cl.NameComparison("first_name"),
    "surname":    lambda: cl.NameComparison("surname"),
    "dob":        lambda: cl.DateOfBirthComparison("dob", input_is_string=True),
    "city":       lambda: cl.ExactMatch("city"),
    "email":      lambda: cl.ExactMatch("email"),
    "gender":     lambda: cl.ExactMatch("gender"),
    "postcode":   lambda: cl.ExactMatch("postcode"),
}

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → BLOCKING RULE mapping
# Single-field blocking rules only.  Multi-field rules cause Splink 4.0.x to
# create SaltedBlockingRules that are incompatible with u-probability sampling
# on single-CPU environments.
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_BLOCKING_RULES = {
    "first_name": lambda: brl.block_on("first_name"),
    "surname":    lambda: brl.block_on("surname"),
    "dob":        lambda: brl.block_on("dob"),
    "city":       lambda: brl.block_on("city"),
    "email":      lambda: brl.block_on("email"),
    "gender":     lambda: brl.block_on("gender"),
    "postcode":   lambda: brl.block_on("postcode"),
}

DEFAULT_CLUSTER_THRESHOLD     = 0.8    # Cluster together records above this probability
DEFAULT_MATCH_WEIGHT_THRESHOLD = -5.0  # Accept most edges; clustering threshold filters


# =============================================================================
# ── DATA EXTRACTION HELPERS ──────────────────────────────────────────────────
# These are called inside run_linkage() to capture extra data for the PDF report.
# =============================================================================

def _compute_missingness(df: pd.DataFrame, fields: list) -> dict:
    """Compute per-field completeness (% non-null values) for each linkage field.
    Returns {field_name: pct_complete} where pct_complete is 0-100.
    Used in the Datasets section of the SeRP-style PDF report."""
    return {
        field: round(df[field].notna().mean() * 100, 1)  # Percentage complete
        for field in fields
        if field in df.columns    # Only include fields actually present in the DataFrame
    }


def _extract_model_params(linker: Linker) -> dict:
    """Extract trained m/u probabilities and match weights from the Splink linker.

    Called after EM training; returns a structured dict used to plot the
    Match Weights chart and Parameter Estimates chart in the PDF report.
    Returns an empty dict on any access error (deterministic mode is fine).

    Structure returned:
      {
        "comparisons": [
          {
            "field": "first_name",
            "levels": [
              {"label": "Exact match", "m_prob": 0.9, "u_prob": 0.01,
               "match_weight": 6.49, "is_null": False},
              ...
            ]
          },
          ...
        ],
        "prior_log_odds": -10.2,      # log2(lambda / (1-lambda))
        "training_complete": True,
      }
    """
    params = {
        "comparisons":       [],     # One entry per comparison field
        "prior_log_odds":    None,   # Starting match weight (prior)
        "training_complete": False,  # Flag: True only if extraction succeeded
    }

    try:
        settings = linker._settings_obj     # Splink 4 internal settings object

        # ── Extract prior match probability (lambda) ─────────────────────────
        try:
            lam = settings._probability_two_random_records_match  # P(match)
            if lam and 0 < lam < 1:
                params["prior_log_odds"] = math.log2(lam / (1.0 - lam))
            else:
                params["prior_log_odds"] = -10.0          # Safe fallback
        except Exception:
            params["prior_log_odds"] = None

        # ── Extract per-level m/u probabilities for every comparison ─────────
        for comp in settings.comparisons:
            comp_info = {
                "field":  comp._output_column_name,   # e.g. "first_name"
                "levels": [],                          # One dict per comparison level
            }
            for level in comp.comparison_levels:
                m     = getattr(level, "m_probability", None)   # P(agree | match)
                u     = getattr(level, "u_probability", None)   # P(agree | non-match)
                label = getattr(level, "label_for_charts", "Unknown level")
                null  = getattr(level, "_is_null_level", False) # True for null levels

                # Compute match weight = log2(m/u); skip null levels and zeros
                if m and u and u > 0 and not null:
                    weight = math.log2(m / u)
                else:
                    weight = None

                comp_info["levels"].append({
                    "label":        label,
                    "m_prob":       m,
                    "u_prob":       u,
                    "match_weight": weight,
                    "is_null":      null,
                })
            params["comparisons"].append(comp_info)

        params["training_complete"] = True    # Only set True on full success
    except Exception:
        pass    # Return partial dict; caller must guard on training_complete flag

    return params


def _extract_blocking_counts(df_predict: pd.DataFrame, blocking_rule_sqls: list) -> list:
    """Count pairwise comparisons generated by each blocking rule.

    Splink 4 adds a 'match_key' integer column to df_predict indicating which
    blocking rule (0-indexed) produced each candidate pair.

    Returns a list of dicts: [{"rule_index": 0, "rule_sql": "...", "n": 1234}, ...]
    Sorted by rule_index.  Returns empty list if match_key column is absent.
    """
    if "match_key" not in df_predict.columns:
        return []    # match_key not available (deterministic link may not include it)

    try:
        con = duckdb.connect()    # Temporary in-memory DuckDB connection
        con.register("df_predict", df_predict)

        # Count how many predictions each blocking rule contributed
        counts_df = con.sql("""
            SELECT CAST(match_key AS INTEGER) AS rule_index,
                   COUNT(*) AS n
            FROM df_predict
            GROUP BY rule_index
            ORDER BY rule_index
        """).df()
        con.close()

        results = []
        for _, row in counts_df.iterrows():
            idx = int(row["rule_index"])
            # Map rule index to its SQL string; fallback if index out of range
            sql = blocking_rule_sqls[idx] if idx < len(blocking_rule_sqls) else f"Rule {idx}"
            results.append({
                "rule_index": idx,
                "rule_sql":   sql,           # SQL string for the blocking rule
                "n":          int(row["n"]), # Number of comparisons from this rule
            })
        return results
    except Exception:
        return []    # Never crash; blocking counts are supplementary data


def _compute_unlinkables(df_predict: pd.DataFrame, n_records: int) -> tuple:
    """Compute the 'unlinkable records' curve (from SeRP Edge Metrics section).

    For each match-weight threshold t, the curve shows what percentage of
    input records have NO predicted edge with match_weight >= t.  A high
    unlinkable percentage at a given threshold means many records cannot
    be matched with that confidence level.

    Returns (thresholds, unlinkable_pcts) as paired lists.
    """
    if "match_weight" not in df_predict.columns or n_records == 0:
        return [], []

    # Sample thresholds from -20 to +20 in 0.5-unit steps
    thresholds = [t * 0.5 for t in range(-40, 41)]  # -20 to +20 step 0.5
    unlinkable_pcts = []

    try:
        con = duckdb.connect()
        con.register("df_predict", df_predict)

        for t in thresholds:
            # Count unique IDs (left-side) with at least one edge at this threshold
            result = con.sql(f"""
                SELECT COUNT(DISTINCT unique_id_l) AS n_linked
                FROM df_predict
                WHERE match_weight >= {t}
            """).fetchone()
            n_linked = result[0] if result else 0
            # Unlinkable = records with NO edge at or above threshold
            pct = max(0.0, (n_records - n_linked) / n_records * 100.0)
            unlinkable_pcts.append(round(pct, 1))

        con.close()
    except Exception:
        return [], []

    return thresholds, unlinkable_pcts


# =============================================================================
# ── CORE SPLINK WORKFLOW FUNCTIONS ───────────────────────────────────────────
# =============================================================================

def _build_comparisons(selected_fields: list) -> list:
    """Return a list of Splink comparison objects for the selected fields."""
    return [
        _FIELD_COMPARISONS[f]()
        for f in selected_fields
        if f in _FIELD_COMPARISONS
    ]


def _build_blocking_rules(blocking_toggles: dict) -> list:
    """Return active Splink blocking rule objects (only toggled-on fields).
    Raises ValueError if no rules are active."""
    active = [
        _FIELD_BLOCKING_RULES[field]()
        for field, enabled in blocking_toggles.items()
        if enabled and field in _FIELD_BLOCKING_RULES
    ]
    if not active:
        raise ValueError("At least one blocking rule must be enabled.")
    return active


def _build_model_settings(link_type, selected_fields, blocking_toggles) -> dict:
    """Assemble the Splink settings dict from user inputs.
    Converts comparison objects and blocking rules to dicts for Splink 4.x."""
    comparisons    = _build_comparisons(selected_fields)
    blocking_rules = _build_blocking_rules(blocking_toggles)
    return {
        "link_type":         link_type,
        "unique_id_column_name": "unique_id",
        "comparisons": [c.create_comparison_dict("duckdb") for c in comparisons],
        "blocking_rules_to_generate_predictions": [
            r.create_blocking_rule_dict("duckdb") for r in blocking_rules
        ],
        "retain_matching_columns":                True,
        "retain_intermediate_calculation_columns": True,
        "max_iterations":  25,
        "em_convergence":  0.0001,
    }


def _train_probabilistic(linker: Linker, selected_fields: list) -> None:
    """Three-step EM training for a probabilistic model.

    Uses single-field blocking rules throughout to avoid Splink 4.0.x's
    SaltedBlockingRule incompatibility on single-CPU machines.
    The cpu_count monkeypatch forces Splink to use 2 salting partitions
    (required minimum) during the u-probability random-sampling step.
    """
    PRIORITY  = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
    available = [f for f in PRIORITY if f in selected_fields]
    primary   = available[0] if available else selected_fields[0]
    secondary = available[1] if len(available) > 1 else None

    # Step 1: Prior estimate
    linker.training.estimate_probability_two_random_records_match(
        [brl.block_on(primary)], recall=0.6
    )

    # Step 2: u-probabilities via random sampling (with cpu_count patch)
    _orig = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 2    # Force >= 2 salting partitions
    try:
        linker.training.estimate_u_using_random_sampling(1e5)
    finally:
        multiprocessing.cpu_count = _orig    # Always restore

    # Step 3: EM training for m-probabilities
    linker.training.estimate_parameters_using_expectation_maximisation(
        brl.block_on(primary), fix_u_probabilities=True
    )
    if secondary:
        linker.training.estimate_parameters_using_expectation_maximisation(
            brl.block_on(secondary), fix_u_probabilities=True
        )


def _render_cluster_studio_html(linker, df_predict, df_cluster) -> str:
    """Generate Splink cluster studio HTML for embedding in Streamlit.
    Returns empty string if generation fails (never crashes the app)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name

        linker.visualisations.cluster_studio_dashboard(
            df_predict=df_predict,
            df_clustered=df_cluster,
            out_path=tmp_path,
            overwrite=True,
            return_html_as_string=True,
        )

        import os
        if os.path.exists(tmp_path):
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_str = f.read()
            os.remove(tmp_path)
            return html_str
    except Exception:
        pass
    return ""


# =============================================================================
# ── PUBLIC API ────────────────────────────────────────────────────────────────
# =============================================================================

def run_linkage(
    fakea: pd.DataFrame,
    fakeb: Optional[pd.DataFrame],
    selected_fields: list,
    blocking_toggles: dict,
    operation_mode: str,
    linkage_type: str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Run Splink linkage/deduplication and return all results + report data.

    Args:
        fakea            : Dataset A DataFrame (source_dataset='A')
        fakeb            : Dataset B DataFrame (source_dataset='B'); None for dedupe
        selected_fields  : Field names to use in comparisons
        blocking_toggles : {field: bool} – True = blocking rule enabled
        operation_mode   : 'dedupe' or 'link_dedupe'
        linkage_type     : 'deterministic' or 'probabilistic'
        cluster_threshold: Match probability above which to form entity clusters

    Returns dict with keys:
        df_predict       : pd.DataFrame – pairwise predictions
        df_cluster       : pd.DataFrame – entity cluster assignments
        cluster_html     : str          – Splink cluster studio HTML
        n_edges          : int
        n_clusters       : int
        n_input_records  : int
        settings_used    : dict         – Splink settings (for report audit trail)
        model_params     : dict         – m/u weights (probabilistic only)
        missingness_a    : dict         – {field: pct_complete} for Dataset A
        missingness_b    : dict         – {field: pct_complete} for Dataset B (link mode)
        blocking_counts  : list         – comparisons per blocking rule
        unlinkables      : dict         – {"thresholds": [...], "pcts": [...]}
        run_config       : dict         – metadata (operation/linkage type, fields, etc.)
    """
    link_type = "dedupe_only" if operation_mode == "dedupe" else "link_only"

    # ── Prepare input tables ──────────────────────────────────────────────────
    if operation_mode == "dedupe":
        df_for_dedupe = fakea.copy()
        df_for_dedupe["source_dataset"] = "A"
        input_tables    = [df_for_dedupe]
        n_input_records = len(df_for_dedupe)
    else:
        input_tables    = [fakea, fakeb]
        n_input_records = len(fakea) + len(fakeb)

    # ── Compute missingness BEFORE running Splink ─────────────────────────────
    # All fields including non-selected ones for a complete dataset overview
    all_report_fields = selected_fields
    missingness_a = _compute_missingness(fakea, all_report_fields)
    missingness_b = (_compute_missingness(fakeb, all_report_fields)
                     if fakeb is not None and operation_mode != "dedupe"
                     else {})

    # ── Build settings and Linker ─────────────────────────────────────────────
    settings = _build_model_settings(link_type, selected_fields, blocking_toggles)
    db_api   = DuckDBAPI()
    linker   = Linker(
        input_table_or_tables=input_tables,
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,
    )

    # ── Run model ─────────────────────────────────────────────────────────────
    model_params = {}    # Only populated for probabilistic runs

    if linkage_type == "deterministic":
        df_predict_raw    = linker.inference.deterministic_link()
        df_predict_pd_raw = df_predict_raw.as_pandas_dataframe()
        df_predict_pd_raw["match_probability"] = 1.0    # All deterministic pairs = certain
        df_predict_pd_raw["match_weight"]      = 100.0  # High log-odds for deterministic

        # Ensure source_dataset columns exist (may be absent in dedupe_only)
        if "source_dataset_l" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_l"] = "A"
        if "source_dataset_r" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_r"] = "A"

        # Re-register enriched DataFrame so Splink's clustering step can use it
        df_predict = linker.table_management.register_table(
            df_predict_pd_raw, "df_predict_enriched"
        )
    else:
        # Probabilistic: train EM model
        _train_probabilistic(linker, selected_fields)

        # Extract model parameters BEFORE predict (linker still has trained state)
        model_params = _extract_model_params(linker)

        df_predict = linker.inference.predict(
            threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
        )

    # ── Cluster ───────────────────────────────────────────────────────────────
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    # ── Convert to pandas ─────────────────────────────────────────────────────
    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # ── Extract blocking rule comparison counts from match_key column ─────────
    # blocking_rule_sqls maps rule index → SQL string for the report
    blocking_rule_sqls = [
        r["blocking_rule"]
        for r in settings["blocking_rules_to_generate_predictions"]
    ]
    blocking_counts = _extract_blocking_counts(df_predict_pd, blocking_rule_sqls)

    # ── Compute unlinkable records curve ─────────────────────────────────────
    thresh, pcts = _compute_unlinkables(df_predict_pd, n_input_records)
    unlinkables  = {"thresholds": thresh, "pcts": pcts}

    # ── Generate cluster studio HTML ──────────────────────────────────────────
    cluster_html = _render_cluster_studio_html(linker, df_predict, df_cluster)

    n_edges    = len(df_predict_pd)
    n_clusters = df_cluster_pd["cluster_id"].nunique()

    return {
        "df_predict":       df_predict_pd,
        "df_cluster":       df_cluster_pd,
        "cluster_html":     cluster_html,
        "n_edges":          n_edges,
        "n_clusters":       n_clusters,
        "n_input_records":  n_input_records,
        "settings_used":    settings,
        "model_params":     model_params,       # NEW: m/u weights for report
        "missingness_a":    missingness_a,      # NEW: Dataset A completeness
        "missingness_b":    missingness_b,      # NEW: Dataset B completeness
        "blocking_counts":  blocking_counts,    # NEW: comparisons per rule
        "unlinkables":      unlinkables,        # NEW: unlinkable records curve
        "run_config": {
            "operation_mode":    operation_mode,
            "linkage_type":      linkage_type,
            "selected_fields":   selected_fields,
            "blocking_toggles":  blocking_toggles,
            "cluster_threshold": cluster_threshold,
            "link_type":         link_type,
        },
    }
