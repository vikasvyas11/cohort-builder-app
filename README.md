# Splink Cohort Builder

A Streamlit application for record linkage and deduplication using Splink and DuckDB. Built at Swansea University as an MVP for cohort construction workflows, targeting both non-technical and technical users.

## Access it online at https://cohort-builder.streamlit.app/


## Two workflows

**Standard mode** — guided seven-step workflow for non-technical users. Loads the fake1000 dataset, walks through field selection, blocking rules, operation mode, and linkage type, then produces analysis and an exportable cohort.

**Advanced mode** — for power users who already have a trained Splink model. Upload a model JSON, skip all training steps, and go straight to prediction, analysis, and export.  

You can save your exisiting model on Splink using the following code:
```
# Save model to JSON
linker.misc.save_model_to_json("test_splink_model.json", overwrite=True)
```

Before uploading, please check if your file has the following format to ensure consistency between runs and the app accepts the uploaded JSON file
```
{
  "link_type": "dedupe_only",
  "unique_id_column_name": "unique_id",
  "probability_two_random_records_match": 0.000812,
  "comparisons": [ ... ],
  "blocking_rules_to_generate_predictions": [ ... ]
}
```
Please ensure, comparisons contains m and u probabilities. 

---

## Features

- Probabilistic linkage via Expectation-Maximisation (Splink 4.x + DuckDB backend)
- Deterministic linkage with exact-match blocking rules
- Deduplication only, or cross-dataset linkage (Dataset A + Dataset B)
- Interactive blocking explorer: toggle rules on/off and see the pairwise prediction table update live, with one-click re-clustering
- Composite blocking rules (e.g. first_name + surname as a single rule)
- Exposed training hyperparameters: EM iterations, convergence threshold, recall estimate
- Confusion matrix with ground truth from the cluster column: TP, FP, FN, Precision, Recall, F1, F*, FDR, FNR
- Precision-Recall curve and CRL (Composite Reliability of Linkage) score
- Clickable sidebar navigation with back button and jump-to-export shortcut
- Full metrics suite covering linkage-metrics examples 0–16: match weight histogram, gamma scores, cluster size distribution, Venn diagram, inter-run edge comparison
- SeRP-style downloadable PDF report with nine sections

---

## Project structure

```
splink_cohort_builder/
├── app.py                    # Main Streamlit app (two flows, session-state navigation)
├── modules/
│   ├── data_builder.py       # Builds fake1000 with gender and UK postcode
│   ├── splink_runner.py      # Linkage workflow, JSON flow, coverage matrix, re-clustering
│   ├── metrics_engine.py     # All linkage quality metrics (examples 0-16 + confusion matrix)
│   └── report_gen.py         # SeRP-style PDF report generator
└── requirements.txt
```

---

## Installation

```
pip install -r requirements.txt
streamlit run app.py
```

Optional dependencies for higher data quality in the generated dataset:

```
pip install gender-guesser pgeocode
```

Both fall back gracefully if not installed.

---

## Core dependencies

streamlit, splink, duckdb, pandas, numpy, plotly, fpdf2, matplotlib

---

## Datasets

The built-in fake1000 dataset is derived from Splink's fake_1000, augmented with gender (inferred from first_name) and postcode (UK GeoNames lookup by city). Dataset B is a 50% sample of Dataset A with controlled errors: 14% first-name typos, 9% surname typos, 5% missing DOBs, 15% email variations, 11% city abbreviations, 7% gender errors.

---

## PDF report sections

1. Dataset information and completeness chart
2. Blocking rules and cumulative comparison count chart
3. Comparison methods
4. Model training with match weights chart and parameter estimates chart
5. Unlinkable records chart
6. Edge metrics and match weight histogram
7. Cluster metrics and dataset overlap Venn diagram
8. Confusion matrix with Precision-Recall curve and CRL score

---

## Known limitations and planned work

- Upload own CSV dataset: UI placeholder present, not yet functional
- Composite blocking rules currently limited to pairs of fields
- SAIL Databank provisioning on the export page is a placeholder
- File upload in advanced mode only accepts Splink 4.x model JSON format

---

## Related repositories

- linkage-workflow: JSON-driven Splink model configuration and notebook templates
- linkage-metrics: DuckDB SQL metric functions for intra- and inter-model linkage quality assessment
