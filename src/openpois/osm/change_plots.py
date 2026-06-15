#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module creates plots showing the stability of various OSM tags over time.
"""

import numpy as np
import pandas as pd
import plotnine as gg


def change_plot_reshape_data(
    observations: pd.DataFrame,
    no_change_col: str,
    change_col: str,
    final_observation_col: str,
    day_range: int = 365 * 10,
) -> pd.DataFrame:
    """
    Reshape data for the change plot. The data comes in with one row per POI-tag, and
    is reshaped by elapsed days since the POI-tag was added. For each elapsed day, there
    are four possibilities:

    1. Confirmed unchanged: The tag was observed unchanged on or *after* this day
    2. Confirmed changed: The tag was last observed changed on or *before* this day
    3. Unsure: The tag was last observed unchanged *before* this day, but has not yet
       been observed changed
    4. Aged out: The maximum time elapsed between when the tag was added and our data
       download is *before* this day, so we should drop it from the plot

    Args:
        observations: DataFrame with observations. Each row is an iteration of a
            tag, with the three columns described below.
        no_change_col: Column name for the days elapsed from when the tag was added to
            when it was last confirmed (observed unchanged).
        change_col: Column name for the days elapsed from when the tag was added to when
            it was changed. For tags that were unchanged, this will be infinity.
        final_observation_col: Column name for the days elapsed from when the tag was
            added to when this data was downloaded.
        day_range: Maximum elapsed time period to plot, in days

    Returns:
        DataFrame where each row is an elapsed day count with columns:
        no_change, unknown, change, aged_out, all, ymin, ymax, day, year.
    """
    # Each of the four counts below is the number of rows whose value falls in a
    # per-row integer-day interval [start, stop). Accumulating +1 at start and -1 at
    # stop into a difference array and taking the cumulative sum gives the per-day
    # counts in O(n + day_range), identical to summing the conditions day by day.
    n = len(observations)
    nc = observations[no_change_col].to_numpy(dtype = float)
    ch = observations[change_col].to_numpy(dtype = float)
    fin = observations[final_observation_col].to_numpy(dtype = float)

    def step_counts(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
        # Count, per day_i in [0, day_range), of rows with start <= day_i < stop.
        s = np.clip(
            np.nan_to_num(starts, nan = day_range, posinf = day_range), 0, day_range
        )
        e = np.clip(
            np.nan_to_num(stops, nan = day_range, posinf = day_range), 0, day_range
        )
        e = np.maximum(s, e)  # guard empty / inverted intervals -> zero-width
        diff = np.zeros(day_range + 1, dtype = np.int64)
        np.add.at(diff, s.astype(np.int64), 1)
        np.add.at(diff, e.astype(np.int64), -1)
        return np.cumsum(diff[:day_range])

    reshaped = (
        pd.DataFrame({
            'no_change': step_counts(np.zeros(n), nc),
            'unknown': step_counts(nc, fin),
            'change': step_counts(ch, fin),
            'aged_out': step_counts(fin, np.full(n, day_range)),
        })
        .assign(
            all = pd.col('no_change') + pd.col('change') + pd.col('unknown'),
            ymin = pd.col('no_change') / pd.col('all'),
            ymax = (pd.col('no_change') + pd.col('unknown')) / pd.col('all'),
            day = np.arange(day_range),
            year = pd.col('day') / 365,
        )
    )
    return reshaped


def change_plot_create(
    observations: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    no_change_col: str = 'no_change',
    change_col: str = 'change',
    final_observation_col: str = 'final_obs',
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str = '',
    y_label: str = '',
    day_range: int = 365 * 10,
) -> gg.ggplot:
    """
    Create a single change plot.

    Args:
        observations: DataFrame with observations. Each row is an iteration of a
            tag, with the three columns described below.
        predictions: DataFrame with modeled predictions.
        no_change_col: Column name for the days elapsed from when the tag was added to
            when it was last confirmed (observed unchanged).
        change_col: Column name for the days elapsed from when the tag was added to when
            it was changed. For tags that were unchanged, this will be infinity.
        final_observation_col: Column name for the days elapsed from when the tag was
            added to when this data was downloaded.
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        day_range: Maximum elapsed time period to plot, in days

    Returns:
        ggplot object
    """
    year_range = day_range / 365
    reshaped = change_plot_reshape_data(
        observations = observations,
        no_change_col = no_change_col,
        change_col = change_col,
        final_observation_col = final_observation_col,
        day_range = day_range,
    )
    if predictions is not None:
        if subtitle is not None:
            subtitle = f"{subtitle}\nModeled predictions in red"
        else:
            subtitle = "Modeled predictions in red"
    fig = (
        gg.ggplot(
            data = reshaped,
            mapping = gg.aes(x = 'year', ymin = 'ymin', ymax = 'ymax'),
        ) +
        gg.geom_ribbon(fill = 'blue', alpha = 0.25) +
        gg.geom_line(
            mapping = gg.aes(y = 'ymin'), color = 'black', linetype = 'dashed'
        ) +
        gg.geom_line(mapping = gg.aes(y = 'ymax'), color = 'black') +
        gg.labs(
            title = title,
            subtitle = subtitle,
            x = x_label,
            y = y_label,
        ) +
        gg.scale_y_continuous(
            limits = (0, 1.01),
            breaks = np.arange(0, 1.01, 0.25),
            labels = [f"{x * 100:.0f}%" for x in np.arange(0, 1.01, 0.25)],
        ) +
        gg.scale_x_continuous(
            limits = (0, year_range + 0.01),
            breaks = np.arange(year_range + 1),
            labels = [f"{x:.0f}" for x in np.arange(year_range + 1)],
        ) +
        gg.theme_bw()
    )
    if predictions is not None:
        p_renamed = predictions.assign(
            year = pd.col('t2'),
            y = pd.col('conf_mean'),
            ymin = pd.col('conf_lower'),
            ymax = pd.col('conf_upper'),
        )
        fig = fig + gg.geom_ribbon(
            data = p_renamed,
            fill = 'darkred', alpha = 0.25, linetype = 'dashed',
            color = 'darkred',
        ) + gg.geom_line(
            data = p_renamed, color = 'darkred',
            mapping = gg.aes(x = 'year', y = 'y')
        )
    return fig


def change_multiplot_create(
    observations: pd.DataFrame,
    col: str,
    top_n: int = 9,
    exclude_values: list[str] | None = None,
    no_change_col: str = 'no_change',
    change_col: str = 'change',
    final_observation_col: str = 'final_obs',
    color_label: str | None = None,
    title: str = None,
    subtitle: str = None,
    x_label: str = '',
    y_label: str = '',
    day_range: int = 365 * 10,
) -> gg.ggplot:
    """
    Create a multi-panel change plot.

    Args:
        observations: DataFrame with observations. Each row is an iteration of a
            tag, with the three columns described below.
        col: Column name for the OSM grouping tag to plot.
        top_n: Number of tags to plot, ordered by number of observations.
        exclude_values: Values of ``col`` to drop before ranking the top ``top_n``
            (e.g. catch-all "Other ..." shared labels).
        no_change_col: Column name for the days elapsed from when the tag was added to
            when it was last confirmed (observed unchanged).
        change_col: Column name for the days elapsed from when the tag was added to when
            it was changed. For tags that were unchanged, this will be infinity.
        final_observation_col: Column name for the days elapsed from when the tag was
            added to when this data was downloaded.
        color_label: Legend title for the grouping colors. Defaults to a title-cased
            version of ``col`` when not provided.
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        day_range: Maximum elapsed time period to plot, in days

    Returns:
        ggplot object
    """
    # Drop rows where the tag is missing
    # Get the top occurrences of particular tags
    obs_sub = observations.dropna(subset=[col])
    if exclude_values:
        obs_sub = obs_sub[~obs_sub[col].isin(exclude_values)]
    top_tags = obs_sub[col].value_counts().head(top_n)
    # Create a list of ggplot objects. Collect group labels in descending-count
    # order so the legend can be ordered to match the value_counts() ranking.
    reshaped_list = []
    group_order = []
    for tag, _ in top_tags.items():
        obs_sub_tag = obs_sub.query(f"{col} == @tag")
        group_label = (
            tag.replace("_", " ").title() + f" (N = {obs_sub_tag.shape[0]:,})"
        )
        group_order.append(group_label)
        reshaped_sub = (
            change_plot_reshape_data(
                observations = obs_sub_tag,
                no_change_col = no_change_col,
                change_col = change_col,
                final_observation_col = final_observation_col,
                day_range = day_range,
            )
            .assign(group = group_label)
        )
        reshaped_list.append(reshaped_sub)
    # Create a grouped change plot
    reshaped_full = pd.concat(reshaped_list)
    # Order the color legend by descending observation count.
    reshaped_full["group"] = pd.Categorical(
        reshaped_full["group"], categories = group_order, ordered = True
    )
    year_range = day_range / 365
    fig = (
        gg.ggplot(
            data = reshaped_full,
            mapping = gg.aes(x = 'year', color = 'group'),
        ) +
        gg.geom_line(mapping = gg.aes(y = 'ymin'), linetype = 'dashed') +
        gg.geom_line(mapping = gg.aes(y = 'ymax')) +
        # Use the discrete hue palette (in descending-count order) rather than the
        # sequential scale plotnine would otherwise pick for an ordered factor.
        gg.scale_color_hue() +
        gg.labs(
            title = title,
            subtitle = subtitle,
            x = x_label,
            y = y_label,
            color = color_label if color_label is not None else col.replace("_", " ").title(),
        ) +
        gg.scale_y_continuous(
            limits = (0, 1.01),
            breaks = np.arange(0, 1.01, 0.25),
            labels = [f"{x * 100:.0f}%" for x in np.arange(0, 1.01, 0.25)],
        ) +
        gg.scale_x_continuous(
            limits = (0, year_range + 0.01),
            breaks = np.arange(year_range + 1),
            labels = [f"{x:.0f}" for x in np.arange(year_range + 1)],
        ) +
        gg.theme_bw()
    )
    return fig
