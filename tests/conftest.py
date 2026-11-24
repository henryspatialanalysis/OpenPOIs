"""Suite-wide options: tests marked ``model`` run only with ``--model``.

A ``model`` test fits a model by MCMC. Run them when model-fitting code changes
(``src/openpois/models/``, ``calibration_bayes.py``): ``pytest --model``.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model", action = "store_true", default = False,
        help = "also run the tests that fit models by MCMC (slow)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "model: fits a model by MCMC; runs only with --model",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--model"):
        return
    skip = pytest.mark.skip(reason = "model fit; run with --model")
    for item in items:
        if item.get_closest_marker("model"):
            item.add_marker(skip)
