"""Shared fixtures. Heavy fixtures are session-scoped: a Splink run takes seconds."""

import logging
import sys
import warnings

import pytest

# matplotlib is deliberately not a dependency (pandas only needs it for Styler colouring). Make any
# import of it fail here so a dev machine that happens to have it cannot hide a missing requirement.
sys.modules["matplotlib"] = None

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def demo():
    """(base, dataset_a, dataset_b) from the synthetic people generator."""
    from modules.synthetic_data import build_demo_datasets
    return build_demo_datasets()


@pytest.fixture(scope="session")
def demo_fields():
    fields = ["first_name", "surname", "dob", "city", "email", "gender"]
    toggles = {"first_name": True, "surname": True, "dob": True, "email": True, "city": False}
    return fields, toggles


@pytest.fixture(scope="session")
def linkage_runs(demo, demo_fields):
    """One finished Splink run per (mode, method) - shared by several test modules."""
    from modules.splink_runner import run_linkage
    _, a, b = demo
    fields, toggles = demo_fields
    runs = {}
    for mode, other in (("dedupe", None), ("link_dedupe", b)):
        for method in ("deterministic", "probabilistic"):
            runs[(mode, method)] = run_linkage(a, other, fields, toggles, mode, method)
    return runs
