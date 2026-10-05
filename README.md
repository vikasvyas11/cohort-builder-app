# Cohort Builder

A [Streamlit](https://streamlit.io) app for **record linkage and deduplication**, built on
[Splink](https://github.com/moj-analytical-services/splink) and [DuckDB](https://duckdb.org).
Pick or upload data, choose which fields to compare and which blocking rules generate candidate pairs,
run a deterministic or probabilistic (Fellegi–Sunter) linkage, check how accurate it is, compare a
second run, and export the resulting cohort.

It ships with **synthetic demo data** (no real people, no downloads, no API keys), so it runs out of
the box and is safe to host publicly.

## Three ways to use it

| Mode | What happens |
|---|---|
| **Standard** | Pick a built-in dataset → profile it → choose fields, blocking rules, dedupe or link, and method → run → analyse → compare → export |
| **Upload** | Upload a CSV/TSV (or give a public URL) → automatic cleaning → configure → same analysis. Can derive a damaged "Dataset B" from your data to test linking |
| **Advanced** | Upload a saved Splink model JSON, skip training, and go straight to prediction |

## What you get

- **Probabilistic linkage** (Splink, expectation-maximisation) and **deterministic linkage** (exact rules;
  a pair must agree on at least two of the fields you switched blocking on for).
- **Blocking control**: OR/AND modes, combined-field rules, and a size check that refuses a configuration
  that would create millions of pairs.
- **Live blocking explorer**: switch rules on and off and watch edges, clusters and per-group match quality
  change.
- **Accuracy against ground truth** when the data has a `cluster` column: precision, recall, F1, F\*,
  confusion matrix, precision–recall curve, average precision.
- **Run 1 vs Run 2** comparison, a self-contained **HTML report** per run (opens offline, prints to PDF),
  a saveable **model JSON**, and a **cohort CSV** with `cluster_id`.

## Quick start

**Use Python 3.12** (the only version this is tested on; `.python-version` pins it).

```bash
py -3.12 -m venv .venv          # macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Choose **Standard**, then **Use people dataset**.

## The demo data

Both datasets come with a damaged copy (Dataset B) and a hidden `cluster` ground truth, so accuracy can
be measured.

- **People**: 1,000 UK-style records for ~600 people, with realistic drift between repeat records (typos,
  transposed birth dates, new e-mails, house moves) and a few relatives sharing an address.
- **North Carolina voters**: synthetic voters whose demographics, places and common names follow the
  public NC State Board of Elections statewide voter file. The raw file lists real, named people, so
  none of it is in this repository. `tools/build_voter_profile.py` reduces it to **aggregate counts**
  (`modules/nc_voter_profile.json`, ~100 KB; every name kept is shared by at least 500 voters) and
  `modules/voter_data.py` samples brand-new voters from those counts. No row is a real person, and there
  are no addresses, phone numbers or ID numbers.

To rebuild the profile from a fresh download of the voter file:

```bash
python tools/build_voter_profile.py path/to/ncvoter_Statewide.txt
```

Never commit the raw file (it is in `.gitignore`).

### How Dataset B is damaged

`modules/corruption.py` samples rows from a clean table and damages chosen fields at a chosen rate by
field kind: text (letters replaced), dates (shifted, day/month swapped, blanked), years, e-mails, places,
postcodes and coded categories (swapped for another value seen in the column). Seeded and reproducible.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. At <https://share.streamlit.io> choose **Create app**, pick the repo, and set the main file to `app.py`.
3. Under **Advanced settings** choose **Python 3.12** (required). No secrets are needed.

Community Cloud has roughly 1 GB of memory: the demo datasets (1,000 people; 1,200 voters in A and 600 in B) are comfortable.

## Configuration

Optional environment variables:

| Variable | Default | Effect |
|---|---|---|
| `COHORT_BUILDER_ALLOW_LOCAL_FILES` | off | `1` adds a "Local file path" tab to Upload. **Leave off on any public deployment.** |
| `COHORT_BUILDER_DUCKDB_MEMORY_LIMIT` | `2GB` | Memory cap for Splink's DuckDB |
| `COHORT_BUILDER_MAX_CANDIDATE_PAIRS` | `5000000` | Largest blocking configuration the app will run |
| `COHORT_BUILDER_MAX_TRUTH_PAIRS` | `5000000` | Largest ground-truth pair set it will compute |
| `COHORT_BUILDER_MODEL_TIMEOUT_S` | `180` | Time limit for applying an uploaded model |

### Safety on a shared host

An uploaded model JSON contains SQL. It is screened (no statements, subqueries or file functions) and run
on a locked-down DuckDB connection with a memory cap, no file access, and a time limit. URLs for Upload
must be public http(s); private addresses are blocked and downloads are capped at 100 MB.

## Layout

```
app.py                  entry point and page router
flows/                  pages (layout only)
  p_landing, p_standard, p_upload, p_advanced, p_analysis, p_compare_export
modules/                logic
  splink_runner.py        linkage runs, model JSON, blocking explorer maths, sandbox
  metrics_engine.py       linkage-quality metrics
  report.py               Plotly charts and the HTML report
  eda_engine.py           cleaning and field-type inference
  corruption.py           controlled damage for test Dataset B files
  synthetic_data.py       people demo     voter_data.py   NC voter demo
  data_builder.py         cached loaders  cohort_filter.py, demographics.py
  nc_voter_profile.json   aggregate NC voter statistics
utils/                  state, navigation, shared renderers, safe_io (guarded URL/file loading)
tools/build_voter_profile.py   raw voter file → aggregate profile
tests/                  pytest suite
```

## Development

```bash
pip install -r requirements-dev.txt
pytest          # unit, end-to-end Splink runs, and headless click-throughs of every flow
ruff check .
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the run lifecycle and metrics design.

## Limitations

Splink is pinned below 5; composite rules are limited to field combinations you add by hand; Upload reads
CSV/TSV only; there is no authentication, only per-session state.

## License

MIT, see [LICENSE](LICENSE). Built on Splink (MIT), Streamlit (Apache-2.0), DuckDB (MIT), pandas, NumPy
and Plotly.

- Fellegi & Sunter (1969). A theory for record linkage. *JASA* 64(328).
- Linacre et al. (2022). Splink: Free software for probabilistic record linkage at scale. *IJPDS* 7(3).
- Hand, Christen & Kirielle (2021). F\*: an interpretable transformation of the F-measure. *Machine Learning* 110.

Independent project, not affiliated with any university or data provider. Contributions welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md).
