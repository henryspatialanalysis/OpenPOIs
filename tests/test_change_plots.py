#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""Tests for openpois.osm.change_plots."""

import numpy as np
import pandas as pd
import pytest

from openpois.osm.change_plots import (
    change_multiplot_create, change_plot_reshape_data
)


def _reshape_naive(
    observations: pd.DataFrame,
    no_change_col: str,
    change_col: str,
    final_observation_col: str,
    day_range: int,
) -> pd.DataFrame:
    """Reference implementation: the original day-by-day summation logic."""
    return pd.DataFrame({
        'no_change': [
            np.sum(day_i < observations[no_change_col])
            for day_i in range(day_range)
        ],
        'unknown': [
            np.sum(
                (observations[no_change_col] <= day_i) &
                (day_i < observations[final_observation_col])
            )
            for day_i in range(day_range)
        ],
        'change': [
            np.sum(
                (observations[change_col] <= day_i) &
                (day_i < observations[final_observation_col])
            )
            for day_i in range(day_range)
        ],
        'aged_out': [
            np.sum(observations[final_observation_col] <= day_i)
            for day_i in range(day_range)
        ],
    })


@pytest.fixture
def sample_observations() -> pd.DataFrame:
    """A mix of finite and infinite change / final-observation values."""
    rng = np.random.default_rng(20260609)
    n = 2000
    no_change = rng.integers(0, 800, size = n).astype(float)
    # Some tags change after no_change; others never change (inf).
    change = no_change + rng.integers(1, 400, size = n).astype(float)
    change[rng.random(n) < 0.4] = np.inf
    # Final observation is finite for some rows, inf (right-censored) for the rest.
    final_obs = np.maximum(no_change, change) + rng.integers(0, 200, size = n)
    final_obs[rng.random(n) < 0.5] = np.inf
    return pd.DataFrame({
        'no_change': no_change,
        'change': change,
        'final_obs': final_obs,
    })


def test_reshape_matches_naive(sample_observations: pd.DataFrame) -> None:
    """Vectorized reshape is identical to the original day-by-day logic."""
    day_range = 1000
    vectorized = change_plot_reshape_data(
        observations = sample_observations,
        no_change_col = 'no_change',
        change_col = 'change',
        final_observation_col = 'final_obs',
        day_range = day_range,
    )
    naive = _reshape_naive(
        observations = sample_observations,
        no_change_col = 'no_change',
        change_col = 'change',
        final_observation_col = 'final_obs',
        day_range = day_range,
    )
    for col in ['no_change', 'unknown', 'change', 'aged_out']:
        assert np.array_equal(vectorized[col].to_numpy(), naive[col].to_numpy()), col
    # Derived survival bounds follow from the integer counts.
    all_counts = naive['no_change'] + naive['change'] + naive['unknown']
    assert np.allclose(vectorized['ymin'], naive['no_change'] / all_counts)
    assert np.allclose(
        vectorized['ymax'], (naive['no_change'] + naive['unknown']) / all_counts
    )


def test_reshape_with_infinite_final_obs() -> None:
    """The all-inf final_obs case (used by data_viz.py) keeps every row in play."""
    obs = pd.DataFrame({
        'no_change': [10.0, 50.0, 100.0],
        'change': [20.0, np.inf, 200.0],
        'final_obs': [np.inf, np.inf, np.inf],
    })
    day_range = 300
    reshaped = change_plot_reshape_data(
        observations = obs,
        no_change_col = 'no_change',
        change_col = 'change',
        final_observation_col = 'final_obs',
        day_range = day_range,
    )
    # Nothing ages out when final_obs is infinite.
    assert (reshaped['aged_out'] == 0).all()
    # At day 0 every row is still unchanged.
    assert reshaped.loc[0, 'no_change'] == 3
    assert reshaped['all'].iloc[0] == 3


def _multiplot_observations() -> pd.DataFrame:
    """Synthetic observations with deterministic, distinct per-label counts."""
    counts = {'Park': 50, 'Restaurant': 30, 'Cafe': 20, 'Bank': 10}
    frames = []
    for label, count in counts.items():
        frames.append(pd.DataFrame({
            'shared_label': [label] * count,
            'no_change': np.linspace(0, 500, count),
            'change': np.full(count, np.inf),
            'final_obs': np.full(count, np.inf),
        }))
    return pd.concat(frames, ignore_index = True)


def test_multiplot_legend_ordered_by_count() -> None:
    """The group categorical is ordered by descending observation count."""
    fig = change_multiplot_create(
        observations = _multiplot_observations(),
        col = 'shared_label',
        top_n = 4,
        color_label = 'Amenity',
        day_range = 600,
    )
    categories = list(fig.data['group'].cat.categories)
    assert fig.data['group'].cat.ordered
    expected = [
        'Park (N = 50)',
        'Restaurant (N = 30)',
        'Cafe (N = 20)',
        'Bank (N = 10)',
    ]
    assert categories == expected
    # The extracted N values are strictly descending.
    ns = [int(c.split('N = ')[1].rstrip(')')) for c in categories]
    assert ns == sorted(ns, reverse = True)


def test_multiplot_color_label_override() -> None:
    """color_label sets the legend title instead of the title-cased column name."""
    fig = change_multiplot_create(
        observations = _multiplot_observations(),
        col = 'shared_label',
        top_n = 4,
        color_label = 'Amenity',
        day_range = 600,
    )
    assert fig.labels.color == 'Amenity'


def test_multiplot_default_color_label() -> None:
    """Without color_label, the legend title is the title-cased column name."""
    fig = change_multiplot_create(
        observations = _multiplot_observations(),
        col = 'shared_label',
        top_n = 4,
        day_range = 600,
    )
    assert fig.labels.color == 'Shared Label'


def test_multiplot_exclude_values() -> None:
    """exclude_values drops labels before the top_n ranking, which then backfills."""
    # Counts: Park 50, Restaurant 30, Cafe 20, Bank 10.
    fig = change_multiplot_create(
        observations = _multiplot_observations(),
        col = 'shared_label',
        top_n = 2,
        exclude_values = ['Park', 'Cafe'],
        day_range = 600,
    )
    categories = list(fig.data['group'].cat.categories)
    # Park + Cafe removed; the top 2 of the remainder are Restaurant then Bank.
    assert categories == ['Restaurant (N = 30)', 'Bank (N = 10)']
