# =============================================================================
# modules/report_gen.py
# PURPOSE: Generate a downloadable PDF report for a linkage run.
#          Inspired by the reporting structure in:
#            linkage-workflow/src/reporting_utils/reporting.py
#          Adapted to use fpdf2 (lightweight, no font file dependencies)
#          and matplotlib for chart images embedded in the PDF.
# =============================================================================

import io
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend required for headless PDF generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF


# ─────────────────────────────────────────────────────────────────────────────
# PDF LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PAGE_WIDTH  = 210   # A4 width in mm
PAGE_HEIGHT = 297   # A4 height in mm
MARGIN      = 20    # Page margin in mm
CONTENT_W   = PAGE_WIDTH - 2 * MARGIN  # Usable content width

# Colours (RGB tuples)
COLOUR_TITLE  = (30, 30, 100)    # Dark navy for headings
COLOUR_BODY   = (50, 50, 50)     # Dark grey for body text
COLOUR_ACCENT = (0, 102, 204)    # Blue for divider lines and labels
COLOUR_LIGHT  = (230, 230, 240)  # Light grey for table row backgrounds


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS (matplotlib → PNG bytes → embed in PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_bar_png(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> bytes:
    """Render a simple bar chart as PNG bytes using matplotlib.
    Returns raw PNG bytes suitable for FPDF's image() method."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(df[x_col].astype(str), df[y_col], color="#1E6EC4", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1E1E64")
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=9)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)              # Free memory immediately after saving
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PDF CLASS
# ─────────────────────────────────────────────────────────────────────────────

class _LinkagePDF(FPDF):
    """Extended FPDF class with custom header, footer, and section helpers.
    Mirrors the chapter structure from reporting.py (cover, dataset, blocking,
    edge metrics, cluster metrics sections)."""

    def __init__(self, run_label: str = "Run 1"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.run_label = run_label          # Shown in footer and cover page
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.add_page()

    def header(self):
        """Override: Minimal header on every page (except cover page)."""
        if self.page_no() == 1:
            return                          # No header on cover page
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOUR_BODY)
        self.cell(0, 6, "Splink Cohort Builder - Linkage Report", align="L")
        self.ln(2)
        # Thin horizontal rule under header
        self.set_draw_color(*COLOUR_ACCENT)
        self.set_line_width(0.2)
        self.line(MARGIN, self.get_y(), PAGE_WIDTH - MARGIN, self.get_y())
        self.ln(4)

    def footer(self):
        """Override: Page number and run label in footer."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COLOUR_BODY)
        self.cell(0, 5, f"{self.run_label}  |  Page {self.page_no()}", align="C")

    def chapter_title(self, title: str):
        """Render a bold chapter-level heading with a coloured underline."""
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*COLOUR_TITLE)
        self.cell(0, 8, title, ln=True)
        # Horizontal rule below title
        self.set_draw_color(*COLOUR_ACCENT)
        self.set_line_width(0.5)
        self.line(MARGIN, self.get_y(), PAGE_WIDTH - MARGIN, self.get_y())
        self.ln(4)
        self.set_text_color(*COLOUR_BODY)

    def section_heading(self, heading: str):
        """Render a smaller sub-section heading."""
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*COLOUR_TITLE)
        self.cell(0, 6, heading, ln=True)
        self.ln(1)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*COLOUR_BODY)

    def body_text(self, text: str):
        """Render a body paragraph with automatic line wrapping.
        Strips non-latin-1 characters (em-dashes, smart quotes, etc.) because
        fpdf2 core Helvetica font only supports the latin-1 range."""
        # Replace common Unicode typographic characters with ASCII equivalents
        text = (text
                .replace("\u2014", "-")    # em dash -> hyphen
                .replace("\u2013", "-")    # en dash -> hyphen
                .replace("\u2018", "'")    # left single quotation mark
                .replace("\u2019", "'")    # right single quotation mark
                .replace("\u201c", '"')    # left double quotation mark
                .replace("\u201d", '"'))   # right double quotation mark
        # Encode to latin-1 replacing any remaining unmappable characters
        text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*COLOUR_BODY)
        self.multi_cell(CONTENT_W, 5.5, text)
        self.ln(2)

    def metric_row(self, label: str, value: str):
        """Render a single label: value metric line."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*COLOUR_TITLE)
        self.cell(70, 6, label + ":", ln=False)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*COLOUR_BODY)
        self.cell(0, 6, str(value), ln=True)

    def embed_chart(self, png_bytes: bytes, caption: str, w_frac: float = 0.85):
        """Embed a PNG image (from bytes) centred on the page with a caption.
        fpdf2 2.7+ returns an image info object directly; no context manager needed."""
        width_mm = CONTENT_W * w_frac               # Scale chart to fraction of page width
        self.image(
            io.BytesIO(png_bytes),
            x=(PAGE_WIDTH - width_mm) / 2,          # Centre horizontally
            w=width_mm,
        )
        self.ln(2)
        # Caption below the chart
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, caption, align="C", ln=True)
        self.ln(3)
        self.set_text_color(*COLOUR_BODY)

    def simple_table(self, headers: list, rows: list):
        """Render a simple table with alternating row shading.
        headers: list of column header strings
        rows   : list of lists (each inner list is one data row)
        """
        col_w = CONTENT_W / len(headers)            # Equal-width columns
        # ── Header row ────────────────────────────────────────────────────────
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*COLOUR_ACCENT)
        self.set_text_color(255, 255, 255)           # White text on blue header
        for h in headers:
            self.cell(col_w, 7, str(h), border=0, fill=True, align="C")
        self.ln()
        # ── Data rows ─────────────────────────────────────────────────────────
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*COLOUR_BODY)
        for i, row in enumerate(rows):
            fill = i % 2 == 0                       # Alternate row shading
            self.set_fill_color(*COLOUR_LIGHT) if fill else self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 6, str(cell), border=0, fill=fill, align="C")
            self.ln()
        self.ln(3)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _cover_page(pdf: _LinkagePDF, run_config: dict, run_label: str):
    """Build the cover page: title, run configuration summary."""
    # Large centred title
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*COLOUR_TITLE)
    pdf.ln(30)
    pdf.cell(0, 14, "Splink Cohort Builder", align="C", ln=True)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Linkage Analysis Report", align="C", ln=True)
    pdf.ln(4)

    # Run label and timestamp
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*COLOUR_BODY)
    pdf.cell(0, 8, run_label, align="C", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", ln=True)

    pdf.ln(14)
    # Horizontal rule
    pdf.set_draw_color(*COLOUR_ACCENT)
    pdf.set_line_width(0.8)
    pdf.line(MARGIN + 20, pdf.get_y(), PAGE_WIDTH - MARGIN - 20, pdf.get_y())
    pdf.ln(10)

    # Run configuration summary box
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*COLOUR_TITLE)
    pdf.cell(0, 8, "Run Configuration", ln=True)
    pdf.ln(2)

    config_items = [
        ("Operation mode",  run_config.get("operation_mode", "-").replace("_", " ").title()),
        ("Linkage type",    run_config.get("linkage_type", "-").title()),
        ("Fields used",     ", ".join(run_config.get("selected_fields", []))),
        ("Active blocking", ", ".join([
            f for f, v in run_config.get("blocking_toggles", {}).items() if v
        ])),
        ("Cluster threshold", str(run_config.get("cluster_threshold", "-"))),
    ]
    for label, value in config_items:
        pdf.metric_row(label, value)


def _dataset_section(
    pdf: _LinkagePDF, n_input_records: int, operation_mode: str
):
    """Dataset overview section explaining what data was processed."""
    pdf.add_page()
    pdf.chapter_title("1. Dataset Overview")

    if operation_mode == "dedupe":
        desc = (
            "This run performed deduplication on a single dataset (fake1000, Dataset A). "
            "The goal was to identify records within Dataset A that refer to the same entity. "
            "The fake1000 dataset is derived from Splink's built-in fake_1000 dataset, "
            "augmented with gender (inferred from first_name) and postcode (UK postcodes by city)."
        )
    else:
        desc = (
            "This run performed linkage between Dataset A (fake1000, 1000 records) and "
            "Dataset B (50% sample of Dataset A with controlled errors introduced). "
            "Dataset B contains: 14% first-name typos, 9% surname typos, 5% missing DOBs, "
            "15% email variations, 11% city abbreviations, and 7% gender assignment errors. "
            "The goal was to identify matching records across the two datasets."
        )
    pdf.body_text(desc)
    pdf.metric_row("Total records processed", str(n_input_records))


def _blocking_section(pdf: _LinkagePDF, run_config: dict):
    """Blocking rules section explaining what rules were active."""
    pdf.add_page()
    pdf.chapter_title("2. Blocking Rules")

    pdf.body_text(
        "Blocking rules define which record pairs are brought forward for comparison. "
        "Two records are only compared if they match exactly on at least one blocking field. "
        "This reduces the O(n^2) comparison space to a tractable number of candidate pairs. "
        "Turning off blocking rules increases recall but also increases computation time."
    )

    toggles = run_config.get("blocking_toggles", {})
    headers = ["Field", "Blocking Rule Active"]
    rows = [[field, "Yes" if enabled else "No"] for field, enabled in toggles.items()]
    pdf.simple_table(headers, rows)


def _edge_metrics_section(
    pdf: _LinkagePDF, metrics: dict, linkage_type: str
):
    """Edge metrics section: number of edges, match probability distribution."""
    pdf.add_page()
    pdf.chapter_title("3. Edge Metrics")

    pdf.body_text(
        "An 'edge' is a predicted pairwise link between two records. "
        "For probabilistic linkage, each edge has a match_probability between 0 and 1. "
        "Values close to 1.0 indicate near-certain matches; values near 0.5 are uncertain. "
        "For deterministic linkage, all edges are assigned match_probability = 1.0."
    )

    pdf.section_heading("Edge Counts")
    pdf.metric_row("Number of predicted edges", str(metrics.get("n_edges", 0)))
    pdf.metric_row("Unique IDs with at least one edge", str(metrics.get("n_unique_ids", 0)))

    # Match probability stats table
    prob_stats = metrics.get("match_prob_stats", pd.DataFrame())
    if not prob_stats.empty:
        pdf.section_heading("Match Probability Statistics")
        row_data = [
            ("Mean",    prob_stats["mean_match_prob"].iloc[0]),
            ("Median",  prob_stats["median_match_prob"].iloc[0]),
            ("Min",     prob_stats["min_match_prob"].iloc[0]),
            ("Max",     prob_stats["max_match_prob"].iloc[0]),
            ("Std Dev", prob_stats["stddev_match_prob"].iloc[0]),
        ]
        pdf.simple_table(["Statistic", "Value"], [[k, v] for k, v in row_data])
        pdf.body_text(
            "Interpretation: A high mean match_probability (>0.9) suggests the model "
            "is confident in its predictions. A wide standard deviation indicates a mix "
            "of high- and low-confidence edges — inspect lower-confidence edges carefully."
        )

    # Match probability distribution chart
    prob_dist = metrics.get("prob_dist", pd.DataFrame())
    if not prob_dist.empty and len(prob_dist) > 1:
        chart_png = _df_to_bar_png(
            prob_dist, "prob_bin", "n_edges",
            "Distribution of Match Probabilities"
        )
        pdf.embed_chart(
            chart_png,
            "Figure: Number of edges at each match probability band. "
            "High bars near 1.0 indicate strong model confidence."
        )

    # Gamma scores (probabilistic only)
    gamma_means = metrics.get("gamma_means", pd.DataFrame())
    if not gamma_means.empty and linkage_type == "probabilistic":
        pdf.section_heading("Gamma Score Averages")
        pdf.body_text(
            "Gamma scores measure agreement level between record pairs for each comparison field. "
            "gamma=1 means exact agreement; gamma=0 means total disagreement; "
            "intermediate values represent partial matches (e.g. Jaro-Winkler fuzzy similarity). "
            "Higher mean gammas indicate better overall agreement on that field."
        )
        g_rows = [[col.replace("gamma_", ""), f"{val:.4f}"]
                  for col, val in gamma_means.iloc[0].items() if col != "run"]
        if g_rows:
            pdf.simple_table(["Field", "Mean Gamma Score"], g_rows)


def _cluster_metrics_section(pdf: _LinkagePDF, metrics: dict):
    """Cluster metrics section: cluster counts and size distribution."""
    pdf.add_page()
    pdf.chapter_title("4. Cluster Metrics")

    pdf.body_text(
        "After pairwise prediction, connected components clustering groups records "
        "into entity clusters. Each cluster ideally represents one real-world individual. "
        "Singletons (clusters of size 1) are records with no predicted matches. "
        "Larger clusters may indicate correct multi-record entities or over-linking errors."
    )

    pdf.section_heading("Cluster Counts")
    pdf.metric_row("Total distinct clusters", str(metrics.get("n_clusters", 0)))
    pdf.metric_row("Cross-dataset clusters", str(metrics.get("n_cross_dataset", 0)))

    # Singleton vs multi-record breakdown
    singletons = metrics.get("singleton_stats", pd.DataFrame())
    if not singletons.empty:
        pdf.section_heading("Singleton vs Multi-Record Clusters")
        headers = list(singletons.columns)
        rows = [list(row) for _, row in singletons.iterrows()]
        pdf.simple_table(headers, rows)
        pdf.body_text(
            "Interpretation: A high singleton count means many records could not be linked. "
            "This may be acceptable (genuinely unique records) or may indicate blocking "
            "rules that are too restrictive. Multi-record clusters represent found duplicates."
        )

    # Cluster size distribution chart
    cluster_sizes = metrics.get("cluster_sizes", pd.DataFrame())
    if not cluster_sizes.empty and len(cluster_sizes) > 1:
        chart_png = _df_to_bar_png(
            cluster_sizes, "n_nodes", "n_clusters",
            "Distribution of Cluster Sizes"
        )
        pdf.embed_chart(
            chart_png,
            "Figure: How many clusters contain each number of records. "
            "A J-shaped curve (many singletons, few large clusters) is typical for real data."
        )

    # Demographic breakdown
    gender_dist = metrics.get("gender_dist", pd.DataFrame())
    if not gender_dist.empty:
        pdf.section_heading("Gender Distribution in Linked Records")
        pdf.simple_table(
            list(gender_dist.columns),
            [list(row) for _, row in gender_dist.iterrows()]
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_label: str,
    run_config: dict,
    metrics: dict,
    n_input_records: int,
) -> bytes:
    """Generate a complete PDF report for a single linkage run.

    Args:
        run_label      : Human-readable label e.g. "Run 1" or "Run 2"
        run_config     : Dict from splink_runner (operation_mode, linkage_type, etc.)
        metrics        : Dict from metrics_engine.compute_intra_metrics()
        n_input_records: Total records that were processed

    Returns:
        Raw PDF bytes, suitable for st.download_button().
    """
    pdf = _LinkagePDF(run_label=run_label)

    # ── Cover page ────────────────────────────────────────────────────────────
    _cover_page(pdf, run_config, run_label)

    # ── Section 1: Dataset overview ───────────────────────────────────────────
    _dataset_section(pdf, n_input_records, run_config.get("operation_mode", "dedupe"))

    # ── Section 2: Blocking rules ─────────────────────────────────────────────
    _blocking_section(pdf, run_config)

    # ── Section 3: Edge metrics ───────────────────────────────────────────────
    _edge_metrics_section(pdf, metrics, run_config.get("linkage_type", "probabilistic"))

    # ── Section 4: Cluster metrics ────────────────────────────────────────────
    _cluster_metrics_section(pdf, metrics)

    # Return the PDF as bytes for Streamlit's download_button
    return bytes(pdf.output())
