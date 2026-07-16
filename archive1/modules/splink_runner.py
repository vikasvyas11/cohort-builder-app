# =============================================================================
# modules/splink_runner.py
# PURPOSE: Wrap Splink's linkage and deduplication workflow into clean
#          functions the Streamlit app can call.
#          Mirrors logic from:
#            - linkage_workflow/templates/1_train_model_deterministic.ipynb
#            - linkage_workflow/templates/1_train_model_probabilistic.ipynb
# =============================================================================

import io
import json
import multiprocessing
import tempfile
from typing import Optional

import pandas as pd

# Splink 4.x imports – the DuckDB backend is used throughout this project
from splink import DuckDBAPI, Linker
import splink.comparison_library as cl       # Pre-built comparison functions
import splink.blocking_rule_library as brl   # Pre-built blocking rule helpers

# ─────────────────────────────────────────────────────────────────────────────
# FIELD → COMPARISON mapping
# Maps each dataset column to an appropriate Splink comparison strategy.
# NameComparison uses Jaro-Winkler fuzzy matching (good for typos).
# DateOfBirthComparison handles transpositions, off-by-one, and exact matches.
# ExactMatch is used for categorical or semi-structured fields.
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
# Each field's blocking rule requires both records to match exactly on that
# field before they are even considered as a candidate pair.  Blocking reduces
# the comparison space from O(n²) to manageable size.
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

# Default match-probability threshold for clustering; records above this
# threshold are grouped into the same entity cluster.
DEFAULT_CLUSTER_THRESHOLD = 0.8

# Match weight threshold used when predicting with probabilistic model.
# Lower values return more (lower-confidence) edges; higher values are stricter.
DEFAULT_MATCH_WEIGHT_THRESHOLD = -5.0   # Accept most pairs; let clustering threshold filter


def _build_comparisons(selected_fields: list) -> list:
    """Return a list of Splink comparison objects for the given fields.
    Only fields present in _FIELD_COMPARISONS are included; unknown fields skipped."""
    return [
        _FIELD_COMPARISONS[f]()             # Instantiate each comparison object
        for f in selected_fields
        if f in _FIELD_COMPARISONS          # Skip fields not in our map
    ]


def _build_blocking_rules(blocking_toggles: dict) -> list:
    """Return a list of active Splink blocking rule objects.
    Only rules where blocking_toggles[field] == True are included.
    At least one rule must be active; raises ValueError otherwise."""
    active = [
        _FIELD_BLOCKING_RULES[field]()      # Instantiate the blocking rule object
        for field, enabled in blocking_toggles.items()
        if enabled and field in _FIELD_BLOCKING_RULES  # Only include toggled-on fields
    ]
    if not active:
        raise ValueError(
            "At least one blocking rule must be enabled. "
            "Please toggle on at least one field in the blocking rules panel."
        )
    return active


def _build_model_settings(
    link_type: str,
    selected_fields: list,
    blocking_toggles: dict,
) -> dict:
    """Assemble the Splink model settings dictionary from user inputs.

    Args:
        link_type: "dedupe_only" or "link_only" (maps to operation_mode choice)
        selected_fields: columns to use in comparisons
        blocking_toggles: dict[field → bool] indicating active blocking rules

    Returns:
        A settings dict compatible with Splink's Linker constructor.
    """
    comparisons = _build_comparisons(selected_fields)   # Build comparison objects
    blocking_rules = _build_blocking_rules(blocking_toggles)  # Build blocking rules

    # Convert comparison and blocking rule objects to dicts for Splink 4.x settings
    comparison_dicts = [c.create_comparison_dict("duckdb") for c in comparisons]
    blocking_rule_dicts = [r.create_blocking_rule_dict("duckdb") for r in blocking_rules]

    settings = {
        "link_type": link_type,                     # Whether to dedupe, link, or both
        "unique_id_column_name": "unique_id",        # Name of the ID column
        "comparisons": comparison_dicts,             # How to compare each field pair
        "blocking_rules_to_generate_predictions": blocking_rule_dicts,  # Candidate pairs
        "retain_matching_columns": True,             # Keep field columns in df_predict
        "retain_intermediate_calculation_columns": True,  # Keep gamma_ and bf_ columns
        "max_iterations": 25,                        # EM training iterations (MVP speed)
        "em_convergence": 0.0001,                    # Stop when parameter change < this
    }
    return settings


def _train_probabilistic(linker: Linker, selected_fields: list) -> None:
    """Run the three-step EM training procedure for a probabilistic model.

    Important: Splink 4.0.x creates SaltedBlockingRules for multi-field brl.block_on()
    calls which are incompatible with u-probability estimation.  All training rules
    here use single-field blocks to avoid this issue.

    Step 1: Estimate the prior probability that two random records match,
            using a single-field blocking rule to gather high-recall candidate pairs.
    Step 2: Estimate u-probabilities (chance of accidental agreement)
            via random sampling of record pairs.
    Step 3: Train m-probabilities (chance of agreement given a true match)
            via EM on up to two separate single-field blocking sessions.
    """
    # Pick the best single field for prior and EM training based on what was selected
    # Priority order: first_name > surname > dob > first available field
    PRIORITY = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]
    available = [f for f in PRIORITY if f in selected_fields]
    primary   = available[0] if available else selected_fields[0]  # Best single field
    secondary = available[1] if len(available) > 1 else None       # Second-best field

    # ── Step 1: Estimate the prior (lambda) ──────────────────────────────────
    # Single-field blocking rule avoids the SaltedBlockingRule error.
    # recall=0.6 assumes our blocking rules capture ~60% of true matches.
    prior_rule = brl.block_on(primary)
    linker.training.estimate_probability_two_random_records_match(
        [prior_rule], recall=0.6
    )

    # ── Step 2: Estimate u-probabilities via random sampling ─────────────────
    # Splink 4.0.x uses multiprocessing.cpu_count() to set salting_partitions
    # for the internal random-sample blocking rule.  On single-CPU environments
    # (cpu_count = 1) this creates an invalid SaltedBlockingRule with partitions=1,
    # raising ValueError.  Temporarily patching cpu_count to 2 is the minimal fix.
    _orig_cpu_count = multiprocessing.cpu_count   # Save original function
    multiprocessing.cpu_count = lambda: 2         # Force >= 2 so salting is valid
    try:
        linker.training.estimate_u_using_random_sampling(1e5)
    finally:
        multiprocessing.cpu_count = _orig_cpu_count  # Always restore original

    # ── Step 3: Train m-probabilities via EM ─────────────────────────────────
    # Run one EM session on the primary field; a second on secondary if available.
    # fix_u_probabilities=True keeps u-probabilities fixed during EM (standard practice).
    linker.training.estimate_parameters_using_expectation_maximisation(
        brl.block_on(primary), fix_u_probabilities=True
    )
    if secondary:
        linker.training.estimate_parameters_using_expectation_maximisation(
            brl.block_on(secondary), fix_u_probabilities=True
        )


def _render_cluster_studio_html(
    linker: Linker,
    df_predict: object,
    df_cluster: object,
) -> str:
    """Generate the Splink cluster studio HTML for embedding in the Streamlit app.
    Writes to a temp file (required by Splink API), reads back as string.
    Based on model_utils.render_cluster_studio_html from linkage-workflow repo."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name                 # Temporary file path for Splink output

        # Splink's cluster_studio_dashboard writes an interactive HTML to disk
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
                html_str = f.read()             # Read the generated HTML back
            os.remove(tmp_path)                 # Clean up the temporary file
            return html_str
    except Exception:
        pass                                    # Cluster studio is optional – never crash
    return ""                                   # Return empty string if anything fails


def run_linkage(
    fakea: pd.DataFrame,
    fakeb: Optional[pd.DataFrame],
    selected_fields: list,
    blocking_toggles: dict,
    operation_mode: str,
    linkage_type: str,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Core Splink linkage/deduplication runner.  Called from the Streamlit app.

    Args:
        fakea           : Dataset A (full fake1000 with source_dataset="A")
        fakeb           : Dataset B (50% sample with errors) – used only for
                          link_dedupe mode; None for dedupe_only
        selected_fields : List of field names to include in comparisons
        blocking_toggles: Dict mapping each field to True/False (blocking on/off)
        operation_mode  : "dedupe" → dedupe_only; "link_dedupe" → link_only
        linkage_type    : "deterministic" or "probabilistic"
        cluster_threshold: Match probability threshold for entity clustering

    Returns:
        A dict containing:
          df_predict      : pd.DataFrame – pairwise match predictions
          df_cluster      : pd.DataFrame – entity cluster assignments
          cluster_html    : str – Splink cluster studio HTML
          n_edges         : int – number of predicted edges
          n_clusters      : int – number of distinct entity clusters
          n_input_records : int – total records processed
          settings_used   : dict – model settings for audit trail
          run_config      : dict – metadata about this run
    """
    # ── Determine Splink link_type from operation_mode ───────────────────────
    # dedupe = examine one dataset for internal duplicates
    # link_dedupe = examine two datasets and also find within-dataset duplicates
    link_type = "dedupe_only" if operation_mode == "dedupe" else "link_only"

    # ── Choose which tables to pass to the Linker ────────────────────────────
    if operation_mode == "dedupe":
        # Deduplication only: use fakea alone (drop source_dataset if present,
        # then re-add a single constant value so Splink knows this is one dataset)
        df_for_dedupe = fakea.copy()
        df_for_dedupe["source_dataset"] = "A"   # Ensure source_dataset is set
        input_tables = [df_for_dedupe]           # Single table for dedupe_only
        n_input_records = len(df_for_dedupe)
    else:
        # Link and deduplicate: pass both datasets
        input_tables = [fakea, fakeb]            # Two tables for link_only
        n_input_records = len(fakea) + len(fakeb)

    # ── Build Splink model settings ──────────────────────────────────────────
    settings = _build_model_settings(link_type, selected_fields, blocking_toggles)

    # ── Instantiate the DuckDB API and Linker ────────────────────────────────
    db_api = DuckDBAPI()                         # In-memory DuckDB connection
    linker = Linker(
        input_table_or_tables=input_tables,      # Pass one or two DataFrames
        settings=settings,
        db_api=db_api,
        set_up_basic_logging=False,              # Suppress verbose Splink logging
    )

    # ── Run the model: deterministic or probabilistic ────────────────────────
    if linkage_type == "deterministic":
        # Deterministic: every pair satisfying a blocking rule is declared a match.
        # deterministic_link() does NOT include match_probability, so we add it here.
        df_predict_raw = linker.inference.deterministic_link()
        df_predict_pd_raw = df_predict_raw.as_pandas_dataframe()

        # match_probability = 1.0: all deterministic pairs are treated as certain matches
        df_predict_pd_raw["match_probability"] = 1.0
        # match_weight = high log-odds value representing a definitive match
        df_predict_pd_raw["match_weight"] = 100.0

        # Ensure source_dataset columns exist (dedupe_only mode may omit them)
        if "source_dataset_l" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_l"] = "A"
        if "source_dataset_r" not in df_predict_pd_raw.columns:
            df_predict_pd_raw["source_dataset_r"] = "A"

        # Re-register the enriched pandas DataFrame as a Splink-aware table
        # so the clustering step can consume it via the linker
        df_predict = linker.table_management.register_table(
            df_predict_pd_raw, "df_predict_enriched"
        )
    else:
        # Probabilistic: train the EM model, then predict match probabilities.
        _train_probabilistic(linker, selected_fields)  # Three-step EM training

        # Predict all candidate pairs above a very low weight threshold.
        # The cluster step will apply the cluster_threshold to finalise groupings.
        df_predict = linker.inference.predict(
            threshold_match_weight=DEFAULT_MATCH_WEIGHT_THRESHOLD
        )

    # ── Cluster pairwise predictions into entity groups ───────────────────────
    # Records connected by predicted edges above cluster_threshold are merged
    # into a single cluster (connected components algorithm).
    df_cluster = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predict,
        threshold_match_probability=cluster_threshold,
    )

    # ── Convert SplinkDataFrames to pandas for downstream use ─────────────────
    df_predict_pd = df_predict.as_pandas_dataframe()
    df_cluster_pd = df_cluster.as_pandas_dataframe()

    # ── Generate Splink cluster studio HTML ──────────────────────────────────
    cluster_html = _render_cluster_studio_html(linker, df_predict, df_cluster)

    # ── Compute summary counts ────────────────────────────────────────────────
    n_edges = len(df_predict_pd)                        # Total number of predicted edges
    n_clusters = df_cluster_pd["cluster_id"].nunique()  # Distinct entity clusters

    # ── Package results for return ────────────────────────────────────────────
    return {
        "df_predict":       df_predict_pd,      # Full predictions table
        "df_cluster":       df_cluster_pd,      # Full cluster assignments table
        "cluster_html":     cluster_html,        # Splink cluster studio HTML
        "n_edges":          n_edges,             # Quick-access edge count
        "n_clusters":       n_clusters,          # Quick-access cluster count
        "n_input_records":  n_input_records,     # Total records processed
        "settings_used":    settings,            # Settings for audit/report
        "run_config": {                          # Metadata for comparison page
            "operation_mode":   operation_mode,
            "linkage_type":     linkage_type,
            "selected_fields":  selected_fields,
            "blocking_toggles": blocking_toggles,
            "cluster_threshold": cluster_threshold,
            "link_type":        link_type,
        },
    }
