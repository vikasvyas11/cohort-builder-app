"""Boot the real Streamlit app headlessly and drive the landing page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_boots_without_errors(app):
    assert not app.exception
    assert "Use people dataset" in [b.label for b in app.button]


def test_people_dataset_loads(app):
    app = _click(app, "Use people dataset")
    assert not app.exception and app.session_state["dataset_ready"]
    assert len(app.session_state["dataset_a"]) == 1000


def test_voter_registry_loads_with_cohort_filter(app):
    app = _click(app, "Use North Carolina voter dataset")
    assert not app.exception
    assert app.session_state["voter_field_types"] is not None
    assert any("Cohort Filter" in e.label for e in app.expander)


def test_shipped_code_does_not_read_a_bundled_csv():
    root = Path(APP).parent
    text = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py") if "tests" not in p.parts)
    assert "voter_registry.csv" not in text
