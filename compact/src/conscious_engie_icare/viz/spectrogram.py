# ©, 2022, Sirris
# owner: FFNG

""" Visualisations of frequency bands over time as spectrograms.

For plotting a single frequency band, use the functions provided in
<viz/frequency_band.py>.
"""
import matplotlib.pyplot as plt
from conscious_engie_icare import util
import matplotlib.colors as colors
import re
import numpy as np


def show_spectrogram(df, timestamp_col="timestamp", frequency_cols=None,
                    ax=None, **kwargs):
    """ Shows a spectrogram of the given waveform.

    Args:
        df (pd.DataFrame): Dataframe with columns that indicate timestamp,
            location and values at specific frequencies.
        timestamp_col (string): Column title for timestamp. Default: "timestamp".
        frequency_cols (list of strings): Column titles for frequency measures.
            Default: All columns of form "d.d" where d is one or more digits.
        ax (matplotlib axis).

    Returns:
        im
        ax
    """
    # assign default values for function parameters
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    x = df.loc[:, timestamp_col]
    if frequency_cols is None:
        frequency_cols = [x for x in df.columns if re.match(r"\d+.\d+", x)]
    y = np.asarray(frequency_cols, dtype=float)
    C = df.loc[:, frequency_cols].T.to_numpy(dtype=float)
    default_kwargs = {
        "shading": "auto",
        "norm": colors.LogNorm(vmin=C.min(), vmax=C.max())
    }
    kwargs = dict(default_kwargs, **kwargs)

    # plot spectogram
    im = ax.pcolormesh(x, y, C, **kwargs)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Spectogram (n={len(df)})')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time')

    return im, ax


def show_spectrogram_grid(df, x=None, y=None):
    """ Show multiple spectrograms in a grid.

    Args:
        df (pd.DataFrame):
        x (string): column name of df to be used as categorical variable for
            columns. Example: 'pump'.
        y (string): column name of df to be used as categorical variable for
            rows. Example: 'vibration_type'.

    Returns:
        fig, ax
    """
    min_amplitude = min(util.get_min(df))
    max_amplitude = max(util.get_max(df))
    kwargs = {"norm": colors.LogNorm(vmin=min_amplitude, vmax=max_amplitude)}
    if x and y:
        fig, axes = _plot_2d(df, x, y, **kwargs)
    elif x and not y:
        fig, axes = _plot_1d(df, x, ptype='horizontal', **kwargs)
    elif not x and y:
        fig, axes = _plot_1d(df, y, ptype='vertical', **kwargs)
    else:
        raise AttributeError('Need to specify at least one of arguments x or y')
    fig.tight_layout()
    return fig, axes


def plot_frequency_band(row, ax=None):
    """Plots the frequency band for a given row (pd.Series),
    where x are frequencies and y are values."""
    if ax is None:
        fig, ax = plt.subplots()

    timestamp = row['timestamp'].strftime("%d/%m/%Y, %H:%M:%S")
    location = row['location']

    values = row.iloc[2:]
    frequency = list(map(float, values.index))
    values = list(map(float, values))

    ax.plot(frequency, values)
    ax.set_xlabel("Frequency in Hz")
    ax.set_xlim(xmin=0)
    title = f'{location} at {timestamp}'
    ax.set_title(title)
    return ax


def _plot_1d(df, val_col, ptype='horizontal', **kwargs):
    assert ptype in ['horizontal', 'vertical']
    unique_vals = sorted(df[val_col].unique())
    horizontal_subplot_kwargs = {
        'figsize': (5 * len(unique_vals), 4),
        'ncols': len(unique_vals),
    }
    vertical_subplot_kwargs = {
        'figsize': (5, 4 * len(unique_vals)),
        'nrows': len(unique_vals),
    }
    ptype_subplots_kwargs = {
        'horizontal': horizontal_subplot_kwargs,
        'vertical': vertical_subplot_kwargs

    }
    fig, axes = plt.subplots(sharex='all', sharey='all',
                             **ptype_subplots_kwargs[ptype])
    for val, ax in zip(unique_vals, axes):
        df_plot = df[df[val_col] == val]
        im, ax = show_spectrogram(df_plot, ax=ax, **kwargs)
        ax.set_title(f"{val} (n={len(df_plot)})")

    return fig, axes


def _plot_2d(df, x, y, **kwargs):
    unique_xvals = sorted(df[x].unique())
    unique_yvals = sorted(df[y].unique())
    figsize = (10 * len(unique_xvals), 10 * len(unique_yvals))
    fig, axes = plt.subplots(figsize=figsize, ncols=len(unique_xvals),
                             nrows=len(unique_yvals), sharex='all',
                             sharey='all')
    for i, xval in enumerate(unique_xvals):
        for j, yval in enumerate(unique_yvals):
            ax = axes[j][i]
            df_plot = df[(df[x] == xval) & (df[y] == yval)]
            im, ax = show_spectrogram(df_plot, ax=ax, **kwargs)
            ax.set_title(f"{xval}, {yval} (n={len(df_plot)})")
    return fig, axes
