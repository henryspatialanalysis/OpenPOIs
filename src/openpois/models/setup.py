"""
Data-preparation helpers for OSM turnover models.
"""

import pandas as pd


def prepare_data_for_model(
    data: pd.DataFrame,
    group_key: str | None = None,
    group_values: list[str] | None = None,
    min_value_count: int | None = None,
    t1_col: str = 'last_obs_timestamp',
    t2_col: str = 'obs_timestamp',
) -> pd.DataFrame:
    """
    Prepare an observations DataFrame for model fitting.

    Per turnover-model-methodology.md §1.2, the per-row Bernoulli-on-Poisson
    likelihood requires Δ = t_k − t_{k−1} (inter-observation), so the default
    ``t1_col`` is ``last_obs_timestamp``. Multiplying per-row Bernoullis
    telescopes to the correct individual likelihood. The previous default of
    ``last_tag_timestamp`` would have made Δ the duration since the
    individual's start — correct for one-row-per-individual but biased
    downward on multi-version POIs.

    Also emits ``is_first_interval`` — True exactly when
    ``last_obs_timestamp == last_tag_timestamp``, i.e. this row is the first
    surviving observation of its ``(POI, name-iteration)`` individual. Used
    by the ZIE δ extension (methodology §1.7).

    Finally emits ``age_start`` / ``age_end`` — the tag's age (years since
    ``last_tag_timestamp``) at the start and end of the interval. These are the
    integration bounds for time-varying λ models (e.g.
    ``constant_breakpoint``); by construction ``age_end − age_start ==
    tag_years`` and ``age_start == 0`` on first intervals. Constant-λ models
    ignore them.

    Args:
        data: Observations DataFrame as returned by format_observations.
        group_key: Column name of the grouping variable. If None, no group
            filtering is applied.
        group_values: If provided, only rows with group_key in this list are
            kept.
        min_value_count: If provided, groups with fewer than this many
            observations are dropped.
        t1_col: Name of the start-time timestamp column. Default
            ``last_obs_timestamp`` gives the inter-observation interval.
        t2_col: Name of the end-time timestamp column.

    Returns:
        Filtered DataFrame with additional ``tag_days``, ``tag_years``,
        ``is_first_interval``, ``age_start``, and ``age_end`` columns.

    Raises:
        ValueError: If ``t1_col`` or ``t2_col`` is not present in data.
    """
    if group_key is not None:
        keep_ids = data.dropna(subset = [group_key]).id.unique().tolist()  # noqa: F841
        data = data.query('id in @keep_ids')
    # If group values were set, subset to those observations
    if (group_key is not None) and (group_values is not None):
        data = (
            data
            .dropna(subset = group_key)
            .query(f'{group_key} in @group_values')
        )
    if (group_key is not None) and (min_value_count is not None):
        value_counts = data.value_counts(group_key)
        groups_over_threshold = (  # noqa: F841
            value_counts[value_counts >= min_value_count].index.tolist()
        )
        data = data.query(f'{group_key} in @groups_over_threshold')
    # Prepare timestamps
    required_cols = [t1_col, t2_col, 'last_obs_timestamp', 'last_tag_timestamp']
    if any(col not in data.columns for col in required_cols):
        raise ValueError(
            f"Required timestamp columns missing. Expected: {required_cols}"
        )
    data = data.copy()
    for timestamp_col in set(required_cols):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    tag_days = (data[t2_col] - data[t1_col]).dt.days
    # Tag age (years since the current tag value was established) at the start
    # and end of the interval — the integration bounds for time-varying λ. When
    # last_tag_timestamp is missing (e.g. a POI's first-ever observation), fall
    # back to the interval start so age_start = 0 and age_end = tag_years; this
    # keeps the row (only time-varying models read these columns).
    tag_origin = data['last_tag_timestamp'].fillna(data[t1_col])
    age_start_days = (data[t1_col] - tag_origin).dt.days
    age_end_days = (data[t2_col] - tag_origin).dt.days
    data = data.assign(
        tag_days = tag_days,
        tag_years = tag_days / 365,
        is_first_interval = (
            data['last_obs_timestamp'] == data['last_tag_timestamp']
        ),
        age_start = age_start_days / 365,
        age_end = age_end_days / 365,
    )
    data = (
        data
        .dropna(subset = ['tag_years', 'changed'])
        .query('tag_years > 1e-6')
    )
    return data
