# Architecture

## Overview

```
                      app.py  (page config, router)
                         │  session_state["flow"], ["page"]
        ┌────────────────┼──────────────────────────────┐
     Standard          Upload                        Advanced
  p_landing           p_upload                     p_advanced
  p_standard          (setup, EDA,                 (model JSON,
  (configure,          profile, configure)          profile, run)
   operation,
   linkage type)
        └────────────────┴───────────────┬──────────────┘
                                         ▼
                     p_analysis  (Run 1 results - shared)
                     p_compare_export  (Run 2, comparison, export - shared)
                                         │
              ┌──────────────────────────┼───────────────────────────┐
        utils/helpers                modules/                    utils/state, nav, safe_io
   _run_analysis_and_store   splink_runner · metrics_engine ·
   renderers, run summaries  report · eda_engine ·
                             corruption · synthetic_data ·
                             demographics · cohort_filter · data_builder
```

Pages live in `flows/` and contain layout only; anything computable without Streamlit lives in
`modules/` so it can be unit tested.

## Flows and pages

| Flow | Pages (`session_state["page"]`) |
|---|---|
| Standard | `0` Dataset selection → `"profile"` → `1` Configure fields & blocking → `2` Operation mode → `3` Linkage type → `4` Run analysis → `5` Compare runs → `6` Export |
| Upload | `"upload_setup"` → `"upload_eda"` → `"upload_profile"` → `"upload_configure"` → `2` → `3` → `4` → `5` → `6` |
| Advanced | `"advanced_setup"` → `"advanced_profile"` → `"advanced_configure"` → `4` → `5` → `6` |

Pages 4–6 are shared by every flow.

## Run lifecycle

1. The user picks fields, blocking toggles (single fields or `a+b` composites), mode (dedupe / link),
   method (deterministic / probabilistic), blocking mode (OR / AND) and hyper-parameters.
2. `helpers._run_analysis_and_store` validates fields against the loaded columns, calls
   `splink_runner.run_linkage`, then computes run metrics, the confusion matrix and (probabilistic runs)
   the threshold curve, plus the blocking **coverage matrix**, and stores everything in `session_state`.
3. `run_linkage` builds Splink settings, aligns input schemas for link mode, trains (probabilistic) or
   applies exact rules (deterministic), clusters at the threshold, and returns `df_predict`,
   `df_cluster`, model parameters, missingness, blocking counts, unlinkables and the effective settings.
4. The results page shows metrics, demographics, the blocking explorer, cluster studio, accuracy and raw
   tables, and offers model-JSON and HTML report downloads.
5. *Compare runs* offers Run 2 rule toggles and combined-field rules with a **live cascading waterfall**:
   `blocking_rule_patterns` counts, per distinct combination of rules that cover a pair, how many pairs
   there are (cached; DuckDB hash joins, within the pair budget). Toggling then only re-weights that small
   table, and `compute_blocking_waterfall` credits each pair to the first enabled rule in cascade order.
   Left chart = Run 1's rules, right = the live Run 2 selection. Then a full Run 2 and a Run 1 vs Run 2
   comparison. (Run 1's own Blocking Explorer tab keeps its edge-table filter and re-cluster button.)
6. *Export* merges `cluster_id` onto the input records and offers the cohort CSV.

## Metrics design

A pair of records is identified by an order-independent **pair key** (each record's
`"<source>|<id>"`, sorted and joined). Edge sets from different runs, and ground-truth pairs derived
from a `cluster` column, are compared directly with set operations. Recall is always measured against
all true pairs, so a true pair that blocking never generated counts as a miss.

## Deterministic linkage

`deterministic_link()` accepts any pair satisfying one blocking rule. To stop one shared common value
chaining strangers into a giant cluster, a pair is accepted only if it agrees exactly on at least two of
the fields whose blocking rules are switched on (one field if only one is on).

## Key module APIs

| Function | Purpose |
|---|---|
| `splink_runner.run_linkage(...)` | Full run (train / exact rules → predict → cluster) |
| `splink_runner.run_linkage_from_json(model, ...)` | Prediction from a saved model, no training |
| `splink_runner.reconstruct_model_json(settings, params, type)` | Re-loadable model file incl. trained m/u |
| `splink_runner.build_coverage_matrix` / `filter_predict_by_active_rules` / `compute_blocking_waterfall` / `recluster_filtered` | Blocking explorer |
| `splink_runner.blocking_rule_patterns(...)` | Per-rule candidate-pair counts straight from the data (any rule, incl. `a+b`); powers the live Compare Runs waterfall |
| `splink_runner.estimate_candidate_pairs(...)` | Pre-flight pair count for a blocking configuration |
| `metrics_engine.compute_intra_metrics` / `compute_inter_metrics` | Per-run and run-vs-run statistics |
| `metrics_engine.compute_confusion_matrix` / `compute_threshold_curve` / `summarise_threshold_curve` | Accuracy against ground truth |
| `corruption.make_noisy_copy(df, field_rates, ...)` | Build a damaged Dataset B with ground truth |
| `synthetic_data.build_demo_datasets` / `voter_data.build_voter_datasets` | Built-in data |
| `eda_engine.run_full_eda(df)` | Cleaning + type inference |
| `report.generate_report(...)` | HTML report bytes |
| `safe_io.read_remote_table` / `read_local_table` / `read_uploaded_table` | Guarded data loading |

## Security notes

User-supplied URLs go through `utils/safe_io.py` (http/https only, public addresses only, redirects
re-validated, 100 MB cap). Reading server-side file paths is disabled unless
`COHORT_BUILDER_ALLOW_LOCAL_FILES=1`. Known residual risk: DNS rebinding between the address check and
the connection.
