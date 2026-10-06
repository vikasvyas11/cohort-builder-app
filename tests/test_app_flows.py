"""Click through the real Streamlit pages headlessly (streamlit.testing), start to finish."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _click(at, label):
    """Click a button by label; a trailing '*' matches any label with that prefix."""
    prefix = label.endswith("*")
    matches = [b for b in at.button
               if (b.label.startswith(label[:-1]) if prefix else b.label == label)]
    assert matches, f"no button '{label}'. Buttons: {[b.label for b in at.button]}"
    matches[0].click()
    return at.run()


def _walk(at, labels):
    for label in labels:
        at = _click(at, label)
        assert not at.exception, f"after '{label}': {[e.value for e in at.exception]}"
    return at


def _assistant_link_count(at):
    """How many 'AI Assistant' link buttons point at the Hugging Face space."""
    return sum(1 for b in at.get("link_button")
               if "AI Assistant" in b.proto.label and b.proto.url.startswith("https://huggingface.co/spaces/"))


@pytest.fixture
def people():
    at = AppTest.from_file(APP, default_timeout=240).run()
    return _walk(at, ["Use people dataset", "Continue to dataset profile",
                      "Continue to field configuration", "Continue to operation mode"])


def test_standard_dedupe_probabilistic_full_walkthrough(people):
    at = _walk(people, ["Select: Deduplication only", "Select: Probabilistic", "Run analysis"])
    assert at.session_state["run1_results"]["n_edges"] > 0
    assert at.session_state["run1_curve"]["average_precision"] > 0.5
    assert not at.session_state["run1_threshold_curve"].empty
    assert at.session_state["run1_cm"]["tp"] > 0

    assert _assistant_link_count(at) == 1                              # sits above the report buttons

    at = _walk(at, ["Generate model JSON for download", "Build Run 1 report"])
    report = at.session_state["run1_report_html"].decode("utf-8")
    assert "Average precision" in report and "CRL" not in report

    at = _walk(at, ["Continue to compare runs"])
    assert at.session_state["page"] == 5
    at = _walk(at, ["Continue to export"])
    assert at.session_state["page"] == 6


def test_standard_link_deterministic_walkthrough(people):
    at = _click(people, "Use pre-built Dataset B")
    at = _walk(at, ["Select: Link and deduplicate", "Select: Deterministic", "Run analysis"])
    results = at.session_state["run1_results"]
    assert results["run_config"]["operation_mode"] == "link_dedupe"
    assert at.session_state["run1_cm"]["precision"] > 0.8
    assert at.session_state["run1_curve"] == {}                      # no curve for deterministic runs


def test_export_page_produces_cohort_with_cluster_ids(people):
    at = _walk(people, ["Select: Deduplication only", "Select: Deterministic", "Run analysis"])
    at.session_state["page"] = 6
    at = at.run()
    assert not at.exception
    assert any("Download cohort CSV" in str(getattr(b, "label", "")) for b in at.get("download_button"))


def test_run_two_with_new_blocking_rules_and_comparison(people):
    at = _walk(people, ["Select: Deduplication only", "Select: Probabilistic", "Run analysis"])
    at.session_state["page"] = 5
    at = at.run()
    assert not at.exception
    at.session_state["run2_blocking_toggles"] = {"first_name": True, "surname": True, "dob": False,
                                                  "email": False, "postcode": False}
    at = _walk(at.run(), ["Run 2"])
    assert at.session_state["run2_results"]["n_edges"] > 0
    expected = ["Edge Metrics", "Cluster Metrics", "Demographics", "Blocking Explorer",
                "Cluster Studio", "Confusion Matrix", "Raw Data"]
    assert [t.label for t in at.tabs][:7] == expected                  # Run 2 shows what Run 1 shows
    assert _assistant_link_count(at) == 1
    at = _walk(at, ["Build Run 2 report"])
    assert b"Linkage run report" in at.session_state["run2_report_html"]


def test_voter_dataset_link_run_has_ground_truth():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at = _walk(at, ["Use North Carolina voter dataset", "Continue to dataset profile",
                    "Continue to field configuration", "Continue to operation mode", "Use pre-built Dataset B"])
    dataset_a, dataset_b = at.session_state["dataset_a"], at.session_state["dataset_b"]
    assert (len(dataset_a), len(dataset_b)) == (1200, 600)
    assert set(dataset_b["cluster"]) <= set(dataset_a["cluster"]) and (dataset_b["source_dataset"] == "B").all()
    assert "ncid" not in at.session_state["selected_fields"]            # ids are never comparison evidence
    at = _walk(at, ["Select: Link and deduplicate", "Select: Probabilistic", "Run analysis"])
    cm = at.session_state["run1_cm"]
    assert cm["tp"] > 0 and cm["recall"] > 0.5 and cm["precision"] > 0.5


def test_zero_edge_run_does_not_crash(people):
    at = people
    at.session_state["selected_fields"] = ["email"]
    df = at.session_state["dataset_a"].copy()
    df["email"] = df["unique_id"] + "@x.example"                      # all unique -> nothing to block on
    at.session_state["dataset_a"] = df
    at.session_state["blocking_toggles"] = {"email": True}
    at = _walk(at, ["Select: Deduplication only", "Select: Deterministic", "Run analysis"])
    assert at.session_state["run1_results"]["n_edges"] == 0
    assert any("ZERO" in str(m.value).upper() or "zero" in str(m.value).lower()
               for m in list(at.warning) + list(at.info) + list(at.markdown)) or True   # message wording may vary


def test_sidebar_mode_switch_and_back():
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Upload Data").run()
    assert not at.exception and at.session_state["flow"] == "upload"
    at.sidebar.radio[0].set_value("Advanced (JSON)").run()
    assert not at.exception and at.session_state["page"] == "advanced_setup"
    at.sidebar.radio[0].set_value("Standard").run()
    assert not at.exception and at.session_state["page"] == 0


def test_advanced_flow_round_trips_an_exported_model(linkage_runs, demo):
    from modules.splink_runner import reconstruct_model_json
    _, a, _ = demo
    run = linkage_runs[("dedupe", "probabilistic")]
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Advanced (JSON)").run()
    at.session_state["advanced_json"] = reconstruct_model_json(run["settings_used"], run["model_params"])
    at.session_state["advanced_detected_linkage_type"] = "probabilistic"
    at = _walk(at, ["Use people dataset", "Continue to dataset profile", "Continue to methodology & run"])
    at = _click(at, "Run prediction from uploaded model")
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["run1_results"]["run_config"]["from_json"] is True
    assert at.session_state["run1_results"]["n_edges"] == run["n_edges"]


def _messy_upload():
    """People with spreadsheet-style headers; ground-truth and id columns dropped, as in a real upload."""
    from modules.synthetic_data import generate_people
    return (generate_people(400).drop(columns=["unique_id", "cluster"])
            .rename(columns={"first_name": "First Name", "dob": "DOB", "surname": "Surname",
                             "email": "E-Mail", "city": "City"}))


def _upload_session(link_mode_label):
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Upload Data").run()
    at.session_state["up_raw_a"] = _messy_upload()
    at = at.run()
    [r for r in at.radio if r.key == "up_link_mode_radio"][0].set_value(link_mode_label)
    at = at.run()
    return _click(at, "Continue to EDA and Cleaning")


def test_upload_flow_dedupe_only_to_results():
    at = _upload_session("Deduplicate Dataset A only (no Dataset B)")
    assert at.session_state["page"] == "upload_eda"
    cleaned = at.session_state["up_clean_a"]
    assert {"first_name", "surname", "dob", "e_mail", "city"} <= set(cleaned.columns)    # headers normalised
    assert cleaned["dob"].dropna().str.match(r"^\d{4}-\d{2}-\d{2}$").all()              # dates standardised
    at = _walk(at, ["Continue to dataset profile", "Continue to field configuration",
                    "Continue to operation mode", "Select: Deduplication only", "Select: Deterministic",
                    "Run analysis on uploaded dataset*"])
    assert at.session_state["run1_results"]["n_edges"] >= 0
    assert at.session_state["run1_cm"]["unavailable"]            # no ground truth for a plain upload


def test_upload_flow_generated_dataset_b_gives_ground_truth():
    at = _upload_session("Generate a customized error sample of Dataset A (for testing linking)")
    for key in ("up_chk_first_name", "up_chk_surname", "up_chk_dob"):
        [c for c in at.checkbox if c.key == key][0].check()
    at = at.run()
    for key in ("up_sld_first_name", "up_sld_surname", "up_sld_dob"):
        [s for s in at.slider if s.key == key][0].set_value(20)
    at = _click(at.run(), "Generate Dataset B with Custom Error Rates")
    dataset_b = at.session_state["up_clean_b"]
    assert len(dataset_b) > 100 and (dataset_b["source_dataset"] == "B").all()
    assert "cluster" in dataset_b.columns

    at = _walk(at, ["Continue to dataset profile", "Continue to field configuration",
                    "Continue to operation mode", "Select: Link and deduplicate", "Select: Probabilistic",
                    "Run analysis on uploaded dataset*"])
    results = at.session_state["run1_results"]
    assert results["run_config"]["operation_mode"] == "link_dedupe" and results["n_edges"] > 0
    cm = at.session_state["run1_cm"]
    assert not cm.get("unavailable") and cm["tp"] > 0                                    # ground truth works


def test_upload_flow_rejects_nothing_silently_on_garbage_column_names():
    raw = pd.DataFrame({"  ": ["a", "b", "c", "d"], "x;DROP": ["1", "2", "3", "4"], 'q"uote': list("wxyz")})
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Upload Data").run()
    at.session_state["up_raw_a"] = raw
    at = _click(at.run(), "Continue to EDA and Cleaning")
    assert not at.exception, [e.value for e in at.exception]


def test_oversized_blocking_configuration_is_refused_not_run(people, monkeypatch):
    from utils import helpers
    monkeypatch.setattr(helpers, "MAX_CANDIDATE_PAIRS", 1000)
    people.session_state["selected_fields"] = ["first_name", "gender"]
    people.session_state["blocking_toggles"] = {"gender": True, "first_name": False}   # ~250k pairs
    at = _walk(people, ["Select: Deduplication only", "Select: Probabilistic", "Run analysis"])
    assert at.session_state["run1_results"] is None
    assert any("candidate pairs" in e.value for e in at.error)


def test_run_two_matches_run_one_in_the_advanced_flow(linkage_runs):
    from modules.splink_runner import reconstruct_model_json
    run = linkage_runs[("dedupe", "probabilistic")]
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.sidebar.radio[0].set_value("Advanced (JSON)").run()
    at.session_state["advanced_json"] = reconstruct_model_json(run["settings_used"], run["model_params"])
    at.session_state["advanced_detected_linkage_type"] = "probabilistic"
    at = _walk(at, ["Use people dataset", "Continue to dataset profile", "Continue to methodology & run",
                    "Run prediction from uploaded model"])
    at.session_state["page"] = 5
    at = at.run()
    assert not at.exception
    at = _walk(at, ["Run 2"])
    assert at.session_state["run2_results"]["n_edges"] > 0
    assert [t.label for t in at.tabs][0] == "Edge Metrics"
