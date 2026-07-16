# =============================================================================
# modules/metrics_engine.py
# PURPOSE: Compute linkage quality metrics using DuckDB SQL queries.
#          Implements the functions from:
#            linkage-metrics/src/linkage_metrics/intra_model/edge_metrics.py
#            linkage-metrics/src/linkage_metrics/intra_model/cluster_metrics.py
#            linkage-metrics/src/linkage_metrics/inter_model/edge_metrics.py
#            linkage-metrics/src/linkage_metrics/inter_model/cluster_metrics.py
#            linkage-metrics/src/linkage_metrics/utils.py
#
# All metric functions return DuckDB query strings (as in the original repo)
# and are executed against DataFrames registered as DuckDB in-memory tables.
# =============================================================================

import duckdb
import numpy as np
import pandas as pd


# =============================================================================
# ── INTRA-MODEL EDGE METRICS ─────────────────────────────────────────────────
# These measure properties of the predicted edges within a single run.
# =============================================================================

def q_n_edges(table: str = "df_predict") -> str:
    """SQL to count total predicted pairwise edges (one row = one candidate pair)."""
    return f"SELECT COUNT(1) AS n_edges FROM {table}"


def q_n_unique_ids_with_edge(table: str = "df_predict") -> str:
    """SQL to count distinct unique_ids that appear in at least one predicted edge."""
    return f"""
    SELECT COUNT(DISTINCT uid) AS n_unique_ids_with_edge
    FROM (
        SELECT unique_id_l AS uid FROM {table}
        UNION
        SELECT unique_id_r AS uid FROM {table}
    )"""


def q_match_probability_stats(table: str = "df_predict") -> str:
    """SQL to compute match probability summary statistics for all predicted edges."""
    return f"""
    SELECT
        ROUND(AVG(match_probability), 4)    AS mean_match_prob,
        ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
        ROUND(MIN(match_probability), 4)    AS min_match_prob,
        ROUND(MAX(match_probability), 4)    AS max_match_prob,
        ROUND(STDDEV(match_probability), 4) AS stddev_match_prob
    FROM {table}"""


def q_match_weight_distribution(table: str = "df_predict", n_bins: int = 30) -> str:
    """SQL to produce a histogram of match weights across all predicted edges.
    Match weight is the log2 odds ratio: positive = more likely a match."""
    return f"""
    SELECT
        ROUND(match_weight, 1) AS weight_bin,
        COUNT(1)               AS n_edges
    FROM {table}
    GROUP BY weight_bin
    ORDER BY weight_bin"""


def q_match_probability_distribution(table: str = "df_predict") -> str:
    """SQL to produce a histogram of match probabilities in 0.05-wide bins."""
    return f"""
    SELECT
        ROUND(FLOOR(match_probability / 0.05) * 0.05, 2) AS prob_bin,
        COUNT(1) AS n_edges
    FROM {table}
    GROUP BY prob_bin
    ORDER BY prob_bin"""


def q_gamma_scores(table: str = "df_predict") -> str:
    """SQL to compute mean gamma (agreement level) for every gamma_ column.
    Gamma columns are created by Splink during comparison; gamma=1 means
    exact match, gamma=0 means no match, values in between are partial."""
    return f"""
    SELECT * FROM (
        UNPIVOT (
            SELECT {{}}_gamma_cols
            FROM {table}
            LIMIT 1
        )
        ON COLUMNS('^gamma_.*')
        INTO NAME field VALUE val
    ) -- placeholder; actual gamma query built dynamically
    """


# =============================================================================
# ── INTRA-MODEL CLUSTER METRICS ──────────────────────────────────────────────
# These measure properties of the entity clusters formed from predicted edges.
# =============================================================================

def q_n_clusters(table: str = "df_cluster") -> str:
    """SQL to count the number of distinct entity clusters."""
    return f"SELECT COUNT(DISTINCT cluster_id) AS n_clusters FROM {table}"


def q_node_counts_per_cluster(table: str = "df_cluster") -> str:
    """SQL to return each cluster_id with its record count (cluster size)."""
    return f"""
    SELECT
        cluster_id,
        COUNT(1) AS n_nodes
    FROM {table}
    GROUP BY cluster_id"""


def q_cluster_size_distribution(table: str = "df_cluster") -> str:
    """SQL to compute the frequency distribution of cluster sizes.
    e.g. how many clusters have 1 member, 2 members, 3 members, etc."""
    return f"""
    SELECT
        n_nodes,
        COUNT(1) AS n_clusters
    FROM ({q_node_counts_per_cluster(table)})
    GROUP BY n_nodes
    ORDER BY n_nodes"""


def q_singleton_vs_multi(table: str = "df_cluster") -> str:
    """SQL to count how many clusters are singletons (1 record) vs multi-record."""
    return f"""
    SELECT
        CASE WHEN n_nodes = 1 THEN 'Singleton (1 record)'
             ELSE 'Multi-record cluster (2+ records)' END AS cluster_type,
        COUNT(1) AS n_clusters,
        SUM(n_nodes) AS total_records
    FROM ({q_node_counts_per_cluster(table)})
    GROUP BY cluster_type
    ORDER BY cluster_type"""


def q_source_dataset_membership(table: str = "df_cluster") -> str:
    """SQL to compute which source datasets contribute to each cluster.
    Used to build the Venn-diagram-style overlap summary."""
    return f"""
    SELECT
        source_dataset,
        COUNT(DISTINCT cluster_id) AS n_clusters_containing_dataset
    FROM {table}
    GROUP BY source_dataset
    ORDER BY source_dataset"""


def q_cross_dataset_clusters(table: str = "df_cluster") -> str:
    """SQL to count how many clusters contain records from both datasets A and B.
    Only relevant for link_dedupe mode."""
    return f"""
    SELECT COUNT(DISTINCT cluster_id) AS n_cross_dataset_clusters
    FROM (
        SELECT
            cluster_id,
            COUNT(DISTINCT source_dataset) AS n_sources
        FROM {table}
        GROUP BY cluster_id
        HAVING n_sources > 1
    )"""


def q_demographic_breakdown(table: str = "df_cluster", col: str = "gender") -> str:
    """SQL to compute a frequency table for a demographic column in the cluster table.
    Used to compare demographics between runs."""
    return f"""
    SELECT
        {col},
        COUNT(1) AS n_records,
        ROUND(100.0 * COUNT(1) / SUM(COUNT(1)) OVER (), 1) AS pct
    FROM {table}
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    ORDER BY n_records DESC"""


# =============================================================================
# ── INTER-MODEL EDGE METRICS ─────────────────────────────────────────────────
# These compare edges between two different runs (Run 1 vs Run 2).
# Directly mirrors inter_model/edge_metrics.py from the linkage-metrics repo.
# =============================================================================

def q_edge_difference_counts(
    table_a: str = "df_predict_run1",
    table_b: str = "df_predict_run2",
) -> str:
    """SQL to count how many edges were added, removed, or shared between two runs.
    - shared: edges present in both A and B
    - added:  edges in B but not in A (new matches in run 2)
    - removed: edges in A but not in B (lost matches in run 2)
    Mirrors edge_difference_counts_of_df_predicts() from the metrics repo."""
    return f"""
    SELECT COUNT(*) AS n, 'shared' AS category
    FROM {table_a} AS a
    INNER JOIN {table_b} AS b
    USING (unique_id_l, unique_id_r, source_dataset_l, source_dataset_r)

    UNION ALL

    SELECT COUNT(*) AS n, 'added' AS category
    FROM {table_b} AS b
    WHERE NOT EXISTS (
        SELECT 1 FROM {table_a} AS a
        WHERE a.unique_id_l = b.unique_id_l
          AND a.unique_id_r = b.unique_id_r
          AND a.source_dataset_l = b.source_dataset_l
          AND a.source_dataset_r = b.source_dataset_r
    )

    UNION ALL

    SELECT COUNT(*) AS n, 'removed' AS category
    FROM {table_a} AS a
    WHERE NOT EXISTS (
        SELECT 1 FROM {table_b} AS b
        WHERE b.unique_id_l = a.unique_id_l
          AND b.unique_id_r = a.unique_id_r
          AND b.source_dataset_l = a.source_dataset_l
          AND b.source_dataset_r = a.source_dataset_r
    )"""


def q_match_prob_comparison(
    table_a: str = "df_predict_run1",
    table_b: str = "df_predict_run2",
) -> str:
    """SQL to compare mean match probabilities between two runs side by side."""
    return f"""
    SELECT
        'Run 1' AS run,
        ROUND(AVG(match_probability), 4) AS mean_match_prob,
        ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
        COUNT(1) AS n_edges
    FROM {table_a}
    UNION ALL
    SELECT
        'Run 2' AS run,
        ROUND(AVG(match_probability), 4) AS mean_match_prob,
        ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
        COUNT(1) AS n_edges
    FROM {table_b}"""


# =============================================================================
# ── INTER-MODEL CLUSTER METRICS ──────────────────────────────────────────────
# Compare cluster assignments between two runs.
# Mirrors inter_model/cluster_metrics.py from the linkage-metrics repo.
# =============================================================================

def q_exact_matching_clusters(
    table1: str = "df_cluster_run1",
    table2: str = "df_cluster_run2",
) -> str:
    """SQL to identify clusters whose exact membership (all unique_ids) is identical
    across two runs. Mirrors exact_matching_clusters() from the metrics repo."""
    return f"""
    SELECT
        a.cluster_id,
        a.unique_ids AS run1_unique_ids
    FROM (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table1} GROUP BY cluster_id
    ) AS a
    INNER JOIN (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table2} GROUP BY cluster_id
    ) AS b
    USING (cluster_id)
    WHERE a.unique_ids = b.unique_ids"""


def q_partial_matching_clusters(
    table1: str = "df_cluster_run1",
    table2: str = "df_cluster_run2",
) -> str:
    """SQL to identify clusters that share some but not all members between runs.
    Mirrors partial_matching_clusters() from the metrics repo."""
    return f"""
    SELECT
        a.cluster_id AS run1_cluster_id,
        b.cluster_id AS run2_cluster_id,
        a.unique_ids AS run1_unique_ids,
        b.unique_ids AS run2_unique_ids
    FROM (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table1} GROUP BY cluster_id
    ) AS a
    JOIN (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table2} GROUP BY cluster_id
    ) AS b
    ON list_has_any(a.unique_ids, b.unique_ids)
       AND a.unique_ids != b.unique_ids"""


# =============================================================================
# ── EXECUTION HELPERS ────────────────────────────────────────────────────────
# These functions register DataFrames, execute the metric queries, and return
# results as pandas DataFrames.
# =============================================================================

def compute_intra_metrics(df_predict: pd.DataFrame, df_cluster: pd.DataFrame) -> dict:
    """Compute all intra-model metrics for a single run.

    Args:
        df_predict : pandas DataFrame from Splink prediction step
        df_cluster : pandas DataFrame from Splink clustering step

    Returns:
        dict with keys:
          n_edges           : int
          n_unique_ids      : int
          match_prob_stats  : pd.DataFrame (1 row of summary stats)
          weight_dist       : pd.DataFrame (histogram of match weights)
          prob_dist         : pd.DataFrame (histogram of match probabilities)
          cluster_sizes     : pd.DataFrame (distribution of cluster sizes)
          singleton_stats   : pd.DataFrame (singleton vs multi-record breakdown)
          source_overlap    : pd.DataFrame (which source datasets appear in clusters)
          n_cross_dataset   : int (clusters spanning both A and B)
    """
    con = duckdb.connect()                              # Fresh in-memory DuckDB connection

    # Register DataFrames as named DuckDB tables for SQL querying
    con.register("df_predict", df_predict)
    con.register("df_cluster", df_cluster)

    results = {}

    # ── Edge counts ──────────────────────────────────────────────────────────
    results["n_edges"] = con.sql(q_n_edges()).fetchone()[0]
    results["n_unique_ids"] = con.sql(q_n_unique_ids_with_edge()).fetchone()[0]

    # ── Match probability stats ───────────────────────────────────────────────
    results["match_prob_stats"] = con.sql(q_match_probability_stats()).df()

    # ── Match weight and probability distributions ────────────────────────────
    results["weight_dist"] = con.sql(q_match_weight_distribution()).df()
    results["prob_dist"] = con.sql(q_match_probability_distribution()).df()

    # ── Cluster metrics ───────────────────────────────────────────────────────
    results["n_clusters"] = con.sql(q_n_clusters()).fetchone()[0]
    results["cluster_sizes"] = con.sql(q_cluster_size_distribution()).df()
    results["singleton_stats"] = con.sql(q_singleton_vs_multi()).df()

    # ── Source dataset overlap (only meaningful for link_dedupe mode) ─────────
    results["source_overlap"] = con.sql(q_source_dataset_membership()).df()

    # ── Cross-dataset clusters ────────────────────────────────────────────────
    try:
        results["n_cross_dataset"] = con.sql(q_cross_dataset_clusters()).fetchone()[0]
    except Exception:
        results["n_cross_dataset"] = 0    # Harmless if source_dataset col is missing

    # ── Gamma scores: compute mean per field if gamma_ columns exist ─────────
    gamma_cols = [c for c in df_predict.columns if c.startswith("gamma_")]
    if gamma_cols:
        gamma_agg = ", ".join(
            [f"ROUND(AVG({c}), 4) AS {c}" for c in gamma_cols]
        )
        results["gamma_means"] = con.sql(
            f"SELECT {gamma_agg} FROM df_predict"
        ).df()
    else:
        results["gamma_means"] = pd.DataFrame()  # Empty if no gamma columns

    # ── Demographic breakdown in clusters (gender and city if present) ────────
    cluster_cols = df_cluster.columns.tolist()
    if "gender" in cluster_cols:
        results["gender_dist"] = con.sql(q_demographic_breakdown("df_cluster", "gender")).df()
    else:
        results["gender_dist"] = pd.DataFrame()

    if "city" in cluster_cols:
        results["city_dist"] = con.sql(q_demographic_breakdown("df_cluster", "city")).df()
    else:
        results["city_dist"] = pd.DataFrame()

    con.close()     # Release the DuckDB connection
    return results


def compute_inter_metrics(
    df_predict_run1: pd.DataFrame,
    df_predict_run2: pd.DataFrame,
    df_cluster_run1: pd.DataFrame,
    df_cluster_run2: pd.DataFrame,
) -> dict:
    """Compute comparison metrics between two runs (inter-model metrics).

    Args:
        df_predict_run1 / run2: prediction DataFrames from each run
        df_cluster_run1 / run2: cluster DataFrames from each run

    Returns:
        dict with comparison metrics for the Comparison page.
    """
    con = duckdb.connect()

    # Register all four DataFrames as named DuckDB tables
    con.register("df_predict_run1", df_predict_run1)
    con.register("df_predict_run2", df_predict_run2)
    con.register("df_cluster_run1", df_cluster_run1)
    con.register("df_cluster_run2", df_cluster_run2)

    results = {}

    # ── Edge difference counts (shared / added / removed) ────────────────────
    results["edge_diff"] = con.sql(
        q_edge_difference_counts("df_predict_run1", "df_predict_run2")
    ).df()

    # ── Match probability comparison between runs ─────────────────────────────
    results["prob_comparison"] = con.sql(
        q_match_prob_comparison("df_predict_run1", "df_predict_run2")
    ).df()

    # ── Exact matching clusters ───────────────────────────────────────────────
    try:
        exact = con.sql(
            q_exact_matching_clusters("df_cluster_run1", "df_cluster_run2")
        ).df()
        results["n_exact_matching_clusters"] = len(exact)
    except Exception:
        results["n_exact_matching_clusters"] = 0  # DuckDB LIST comparison may fail gracefully

    # ── Partial matching clusters ─────────────────────────────────────────────
    try:
        partial = con.sql(
            q_partial_matching_clusters("df_cluster_run1", "df_cluster_run2")
        ).df()
        results["n_partial_matching_clusters"] = len(partial)
    except Exception:
        results["n_partial_matching_clusters"] = 0

    # ── Probability distribution for each run (for side-by-side chart) ───────
    results["prob_dist_run1"] = con.sql(
        q_match_probability_distribution("df_predict_run1")
    ).df()
    results["prob_dist_run2"] = con.sql(
        q_match_probability_distribution("df_predict_run2")
    ).df()

    # ── Cluster size distributions ────────────────────────────────────────────
    results["cluster_sizes_run1"] = con.sql(
        q_cluster_size_distribution("df_cluster_run1")
    ).df()
    results["cluster_sizes_run2"] = con.sql(
        q_cluster_size_distribution("df_cluster_run2")
    ).df()

    # ── Gamma score comparison (if gamma columns present) ─────────────────────
    gamma_cols = [c for c in df_predict_run1.columns if c.startswith("gamma_")]
    if gamma_cols:
        g_agg1 = ", ".join([f"ROUND(AVG({c}), 4) AS {c}" for c in gamma_cols])
        g_agg2 = ", ".join([f"ROUND(AVG({c}), 4) AS {c}" for c in gamma_cols])
        gdf1 = con.sql(f"SELECT 'Run 1' AS run, {g_agg1} FROM df_predict_run1").df()
        gdf2 = con.sql(f"SELECT 'Run 2' AS run, {g_agg2} FROM df_predict_run2").df()
        results["gamma_comparison"] = pd.concat([gdf1, gdf2], ignore_index=True)
    else:
        results["gamma_comparison"] = pd.DataFrame()

    con.close()
    return results
