# ©, 2022, Sirris
# owner: FFNG

"""
Utilities for visualisations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
import re
import numpy as np


def illustrate_nmf_components_for_paper(V, explained_variance_ratio, df_nmf_models, band_cols,
                                        min_explained_variance=90, vmin=0.001, vmax=0.1, order_components=None,
                                        xlims=(-1, 31), plot_x_ticks=7):
    """ Illustrate NMF components together with hyperparameter tuning and decomposition matrix.

    Improved version of illustrate_nmf_components for paper.

    :param V: Decomposition matrix as it is fed to NMF.
    :param explained_variance_ratio: Numpy array with explained variance ratio extracted with PCA.
    :param df_nmf_models: Dataframe with NMF-models for different number of components.
    :return:
    """
    def get_n_components(x, threshold):
        return (x > threshold).argmax() + 1

    # when plotting the title of the bin, we use the middle of the bin
    # e.g. if a bin ranges from 1 to 2, we plot the title at 1.5.
    # therefore we need to add 0.5 to the lower bin edge, which we call the offset
    column_names = V.columns[0]

    def extract_lower_limit(x):
        return float(re.findall(r'(?<=\_)(.*)(?=-)', x)[0])

    def extract_upper_limit(x):
        return float(re.findall(r'(?<=-)(.*)$', x)[0])

    offset = 0.5 * (extract_upper_limit(column_names) - extract_lower_limit(column_names))
    all_x_label_names = [extract_lower_limit(col)+offset for col in V.columns]

    cmap = mpl.cm.get_cmap("Blues")
    # mpl.rcParams['text.usetex'] = True
    fig, ax_row = plt.subplots(figsize=(8, 6), ncols=3, nrows=1, sharex=False, sharey=False, constrained_layout=True)

    # 1st column: show performance matrix V
    ax = ax_row[0]
    nrows = V.shape[0]
    ncols = V.shape[1]
    title_ = "Performance matrix V" + f" ({nrows} x {ncols})"
    ax.set_title(title_, fontsize=None)
    # V.columns = BAND_COLUMNS
    im = ax.imshow(
        V.sort_index(),
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        # extent=[0.25,30.25,nrows,0]
    )
    ax.tick_params(axis='y', labelrotation=90)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('Frequency [orders]')

    if isinstance(plot_x_ticks, int):
        ax.set_xticks(np.arange(0, ncols, plot_x_ticks))
        every_nth_x_label = all_x_label_names[::plot_x_ticks]
    elif isinstance(plot_x_ticks, list):
        ax.set_xticks(plot_x_ticks)
        every_nth_x_label = [all_x_label_names[idx] for idx in plot_x_ticks]
    ax.set_xticklabels(every_nth_x_label)

    # 2nd column: selection of number of components in two columns
    ax = ax_row[1]
    gridspec = ax.get_subplotspec().get_gridspec()
    ax.remove()
    subfig = fig.add_subfigure(gridspec[1])
    sub_axes = subfig.subplots(2, 1, sharex=False, sharey=False)

    # PCA
    ax = sub_axes[0]
    ax.set_ylim([30, 100])
    explained_variance = 100 * explained_variance_ratio
    ax.plot(pd.Series(dict(enumerate(explained_variance, start=1))), marker='o', markersize=4)
    threshold = 95
    n_components_95 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='green', linestyle='dotted', label=f'> {threshold}%')
    threshold = 90
    n_components_90 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='red', linestyle='dashed', label=f'> {threshold}%')
    threshold = 75
    n_components_75 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='blue', linestyle='dotted', label=f'> {threshold}%')
    ax.set_ylabel('cumulative explained variance [%]')
    ax.set_xlabel('number of components')
    ax.set_title('Cumulative explained variance')
    ax.legend()

    # NMF
    ax = sub_axes[1]
    x = df_nmf_models['n_components']
    y = df_nmf_models['reconstruction_error']
    ax.plot(pd.Series(dict(zip(x, y))), marker='o', markersize=4)
    ax.set_title('Reconstruction error of NMF')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Frobenius norm')
    # ax.set_ylim([0.00005, 0.0005])

    # 3rd column
    ax = ax_row[2]
    gridspec = ax.get_subplotspec().get_gridspec()
    ax.remove()
    subfig = fig.add_subfigure(gridspec[2])
    n_components = get_n_components(explained_variance, min_explained_variance)
    sub_axes = subfig.subplots(n_components, 1, sharex=True, sharey=True)
    subfig.suptitle('components H')
    row = df_nmf_models[df_nmf_models.n_components == n_components].iloc[0]
    H = row.H
    # alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z') + 1)))
    # y_labels = alphabet_upper_case[:H.shape[0]]
    y_labels = list(range(len(H)))

    if order_components is not None:
        H = H.T[order_components].T

    for (row, x), ax, y_label in zip(H.iterrows(), sub_axes, y_labels):
        # lower_bin = band_cols.str.extract(r'(?<=\_)(.*)(?=-)')[0]
        # x.index = lower_bin.astype(float) + offset
        x.name = 'Frequency [orders]'
        x.plot(label=row, ax=ax, markersize=3)
        # ax.set_title(y_label)
        ax.set_xlabel('Frequency [orders]')
        ax.set_ylabel(y_label)
        if isinstance(plot_x_ticks, int):
            ax.set_xticks(np.arange(0, len(x), plot_x_ticks))
            every_nth_x_label = all_x_label_names[::plot_x_ticks]
        elif isinstance(plot_x_ticks, list):
            ax.set_xticks(plot_x_ticks)
            every_nth_x_label = [all_x_label_names[idx] for idx in plot_x_ticks]
        ax.set_xticklabels(every_nth_x_label)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax.set_xlim(xlims)
    return fig, ax_row


def illustrate_nmf_components(V, explained_variance_ratio, df_nmf_models, band_cols, min_explained_variance=90,
                              vmin=0.001, vmax=0.1):
    """ Illustrate NMF components together with hyperparameter tuning and decomposition matrix.

    :param V: Decomposition matrix as it is fed to NMF.
    :param explained_variance_ratio: Numpy array with explained variance ratio extracted with PCA.
    :param df_nmf_models: Dataframe with NMF-models for different number of components.
    :return:
    """
    def get_n_components(x, threshold):
        return (x > threshold).argmax() + 1

    cmap = mpl.cm.get_cmap("Blues")
    fig, ax_row = plt.subplots(figsize=(25, 5), ncols=4, nrows=1, sharex=False, sharey=False)

    # 1st column: show feature matrix
    ax = ax_row[0]
    nrows = V.shape[0]
    ncols = V.shape[1]
    ax.set_title(f"V: {nrows} x {ncols}", fontsize=None)
    # V.columns = BAND_COLUMNS
    im = ax.imshow(
        V.sort_index(),
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        # extent=[0.25,30.25,nrows,0]
    )
    ax.tick_params(axis='y', labelrotation=90)

    # 2nd column: PCA based statistics
    ax = ax_row[1]
    ax.set_ylim([30, 100])
    explained_variance = 100 * explained_variance_ratio
    ax.plot(pd.Series(dict(enumerate(explained_variance, start=1))), marker='o', markersize=4)
    threshold = 95
    n_components_95 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='green', linestyle='dotted', label=f'> {threshold}% for {n_components_95} components')
    threshold = 90
    n_components_90 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='red', linestyle='dashed', label=f'> {threshold}% for {n_components_90} components')
    threshold = 75
    n_components_75 = get_n_components(explained_variance, threshold)
    ax.axhline(threshold, color='blue', linestyle='dotted', label=f'> {threshold}% for {n_components_75} components')
    ax.set_ylabel('cumulative explained variance [%]')
    ax.set_xlabel('number of components')
    ax.set_title('Cumulative explained variance')
    ax.legend()

    # 3rd column
    ax = ax_row[2]
    x = df_nmf_models['n_components']
    y = df_nmf_models['reconstruction_error']
    ax.plot(pd.Series(dict(zip(x, y))), marker='o', markersize=4)
    ax.set_title('Reconstruction error of NMF')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Frobenius norm')
    ax.set_ylim([0, 3.5])

    # 4th column
    ax = ax_row[3]
    gridspec = ax.get_subplotspec().get_gridspec()
    ax.remove()
    subfig = fig.add_subfigure(gridspec[3])
    n_components = get_n_components(explained_variance, min_explained_variance)
    sub_axes = subfig.subplots(n_components, 1, sharex=True, sharey=True)
    subfig.suptitle(f'{n_components} components', y=0.93)
    row = df_nmf_models[df_nmf_models.n_components == n_components].iloc[0]
    H = row.H
    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z') + 1)))
    y_labels = alphabet_upper_case[:H.shape[0]]

    for (row, x), ax, y_label in zip(H.iterrows(), sub_axes, y_labels):
        lower_bin = band_cols.str.extract(r'(?<=\_)(.*)(?=-)')[0]
        x.index = lower_bin.astype(float) + 0.25
        x.name = 'Frequency [orders]'
        x.plot(label=row, ax=ax, marker='o', markersize=3)
        # ax.set_title(y_label)
        ax.set_xlabel('Frequency [orders]')
        ax.set_ylabel(y_label)
    ax.set_xlim((-1, 31))
    return fig, ax_row


def plot_measures(data, meta_data=None):
    """Plot measures for the given meta data side-by-side."""
    measures = data[meta_data['_id']]
    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    title = meta_data['asset'] + ' @ ' + meta_data['acqend']
    fig.suptitle(title)
    plot_g(measures['g'], ax=axes[0])
    plot_fftg(measures['fftg'], ax=axes[1])
    plot_fftv(measures['fftv'], ax=axes[2])
    fig.tight_layout()
    return fig, axes


def plot_g(g, ax=None):
    """Plot acceleration against time."""
    ax = ax or plt.subplots()[1]
    sns.lineplot(data=g, x='time', y='acceleration', ax=ax, legend='brief',
                 label='acc')
    return ax


def plot_fftg(fftg, ax=None):
    """Plot acceleration against frequency."""
    ax = ax or plt.subplots()[1]
    sns.lineplot(data=fftg, x='frequency', y='acceleration', ax=ax,
                 legend='brief', label='acc')
    return ax


def plot_fftv(fftv, ax=None):
    """Plot velocity against frequency."""
    ax = ax or plt.subplots()[1]
    sns.lineplot(data=fftv, x='frequency', y='velocity', ax=ax,
                 legend='brief', label='velocity')
    return ax


def plot_harmonics(speed_rpm, vibration_data, meta_data, ts_harmonics=None,
                   min_harmonics=1, max_harmonics=20, marked_orders=None,
                   show_legend=True):
    """Plot harmonics on top of regular measures."""
    fig, axes = plot_measures(vibration_data, meta_data)
    speed_hz = speed_rpm / 60
    harmonics = {}
    for i in range(min_harmonics, max_harmonics):
        label = _get_label(i, marked_orders, max_harmonics, ts_harmonics, speed_rpm)
        i_marked = (marked_orders is not None) and (i in marked_orders)
        color = 'red' if i_marked else 'grey'
        harmonics[i] = i * speed_hz
        for j in [1, 2]:
            # vertical line indicates harmonic
            axes[j].axvline(harmonics[i], color=color, alpha=0.7, label=label)
            if show_legend:
                axes[j].legend(fontsize='small')
    return fig, axes


def _get_label(i, marked_harmonics, max_harmonics, ts_harmonics, speed_rpm):
    """Determine label for plotting harmonics."""
    if (marked_harmonics is not None) and (i == max(marked_harmonics)):
        # show label for selected harmonics
        label = 'X, '.join([str(h) for h in marked_harmonics]) + 'X'
    elif i + 1 >= max_harmonics:
        # only include label for last harmonic to avoid repetitively showing
        # the same label
        label = 'orders'
        label = label + f' @ {ts_harmonics}' if ts_harmonics else label
        label = label + f' ({round(speed_rpm, 1)} RPM)' if label is not None else None
    else:
        label = None
    return label


def plot_nr_of_entries(df):
    df = df.reset_index()

    fig, axes = plt.subplots(figsize=(10, 7), nrows=2, sharex='all')

    # plot total
    series = pd.Series([1 for _ in range(len(df))], index=df['timestamp'])
    nr_of_entries_per_day = series.resample("D").sum()
    ax = nr_of_entries_per_day.plot(ax=axes[0])
    _ = ax.set_ylabel("# entries / day")

    # plot per location
    tmp = df.groupby('location').resample("D", on='timestamp')
    entries_per_plant = tmp.size().unstack(0, fill_value=0)
    ax = sns.lineplot(data=entries_per_plant, ax=axes[1], alpha=0.66)
    ax.legend(ncol=6, title='location')
    _ = ax.set_ylabel("# entries / (day X location)")

    plt.tight_layout()

    return fig, axes


def plot_nr_of_entries2(df, ax=None, **plot_kwargs):
    # TODO refactor
    df = df.reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # plot total
    series = pd.Series([1 for _ in range(len(df))], index=df['timestamp'])
    nr_of_entries_per_day = series.resample("D").sum()
    ax = nr_of_entries_per_day.plot(ax=ax, **plot_kwargs)
    _ = ax.set_ylabel("# entries / day")

    return ax


def plot_agg_per_location(df, rule='W', hue="location", ax=None):
    # TODO refactor
    """ Plot aggregated value per location.

    Args:
        df (pd.DataFrame): Data with entries for speed, pressure or flow.
        rule (str): Aggregation rule. Default: 'W' for aggregating over week.
        ax (matplotlib axis): axis to plot on.

    Returns:
        axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    def intersection(lst1, lst2):
        """Returns intersection between two lists as set."""
        return list(set(lst1) & set(lst2))

    vars = ['speed', 'pressure', 'flow']
    var_name_list = intersection(vars, df.columns)
    if len(var_name_list) != 1:
        raise ValueError(f"df must contain exactly one of {vars}.")
    var_name = var_name_list[0]

    # prepare data
    agg_df = df.groupby('location')
    agg_df = agg_df.resample(rule, on='timestamp').mean()
    agg_df = agg_df.unstack(0, fill_value=0).fillna(0)
    unique_locations = df.location.unique().tolist()
    agg_df = agg_df[var_name].reset_index().melt(value_vars=unique_locations,
                                                 id_vars=['timestamp'])

    # plot
    ax = sns.lineplot(data=agg_df, x="timestamp", y="value", hue=hue,
                      style=hue, ax=ax, markers=True, alpha=0.66)
    _ = ax.set_title(f"Mean {var_name} per location")
    _ = ax.set_ylabel(f"mean {var_name}")

    return ax


def plot_band(ax_, axvspan_start_time_, axvspan_stop_time_):
    ax_.axvspan(pd.to_datetime(axvspan_start_time_), pd.to_datetime(axvspan_stop_time_), 
                alpha=0.2, label='surge in anomaly score', color='red')
    return ax_


def plot_process_view_for_paper(process_view_with_results, cavitation_risk, df_process,
                                pump_, start_time_, stop_time_, axvspan_start_time_, axvspan_stop_time_,
                                name_certainty_score_, step_size_=360, suptitle=True,
                                minor_day_interval=1, major_day_interval=7):
    fig, axes = plt.subplots(figsize=(5, 8.5), nrows=4, sharex=False,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    if suptitle:
        fig.suptitle(f'Process view (pump {pump_})', color='blue', size=20)

    df_ = process_view_with_results[pump_].copy()
    df_ = df_.set_index('timestamp')
    df_ = df_[(df_.index.tz_convert(None) >= start_time_) & (df_.index.tz_convert(None) <= stop_time_)]
    df_ = df_.rolling('D', min_periods=8000).mean()[::step_size_]

    df_cavitation_risk_ = cavitation_risk.set_index('timestamp')
    df_cavitation_risk_ = df_cavitation_risk_[df_cavitation_risk_.pump == pump_]
    df_cavitation_risk_.index = pd.to_datetime(df_cavitation_risk_.index).tz_convert(None)
    df_cavitation_risk_ = df_cavitation_risk_[
        (df_cavitation_risk_.index >= start_time_) & (df_cavitation_risk_.index <= stop_time_)
    ]

    df_process_ = df_process.set_index('timestamp')
    df_process_ = df_process_[df_process_.pump == pump_]
    df_process_.index = pd.to_datetime(df_process_.index).tz_convert(None)
    df_process_ = df_process_[(df_process_.index >= start_time_) & (df_process_.index <= stop_time_)]

    # First axis: certainty score
    ax = axes[0]
    ax.set_title('Certainty score', color='black', size=14)
    label_ = 'rolling mean (window width = 1 day)'
    df_[name_certainty_score_].plot(ax=ax, label=label_)
    ax.legend()
    ax = plot_band(ax, axvspan_start_time_, axvspan_stop_time_)
    ax.set_xlim(pd.to_datetime(start_time_), pd.to_datetime(stop_time_))
    ax.set_ylim([0.2, 0.8])
    ax.set_ylabel('certainty score')
    ax = x_labels(ax, minor_day_interval=minor_day_interval, major_day_interval=major_day_interval)

    # Second axis: known samples
    ax = axes[1]
    ax.set_title('Unknown operating modes', color='black', size=14)
    ax.set_ylabel('daily unknown [%]')
    unknown_om = df_[name_certainty_score_] == 0
    daily_sum_of_unknowns = unknown_om.rolling('D', min_periods=8000).sum()
    daily_sum_of_total = unknown_om.rolling('D', min_periods=8000).count()
    daily_percentage_of_unknowns = 100 * (daily_sum_of_unknowns / daily_sum_of_total)
    # daily_percentage_of_unknowns[::STEP_SIZE].plot(ax=ax)
    label_ = 'rolling window (window width = 1 day)'
    daily_percentage_of_unknowns[::step_size_].plot(ax=ax, label=label_)
    ax.legend()
    ax = plot_band(ax, axvspan_start_time_, axvspan_stop_time_)
    ax = x_labels(ax, minor_day_interval=minor_day_interval, major_day_interval=major_day_interval)
    ax.set_ylim([0, 30])
    ax.set_xlim(pd.to_datetime(start_time_), pd.to_datetime(stop_time_))

    # Third axis: cavitation risk percentile
    ax = axes[2]
    ax.set_title('Cavitation risk', color='black', size=14)
    cavitation_risk_percentile = 100 * df_cavitation_risk_['r_cav_new_v2'].rank(pct=True)
    label_ = 'rolling mean (window width = 1 day)'
    cavitation_risk_percentile.rolling('D', min_periods=8000).mean()[::step_size_].plot(ax=ax, label=label_)
    ax.legend()
    ax = plot_band(ax, axvspan_start_time_, axvspan_stop_time_)
    ax = x_labels(ax, minor_day_interval=minor_day_interval, major_day_interval=major_day_interval)
    ax.set_xlim(pd.to_datetime(start_time_), pd.to_datetime(stop_time_))
    ax.set_ylim([0, 100])
    ax.set_ylabel('cavitation risk percentile [%]')

    # Fourth axis: head
    ax = axes[3]
    ax.set_title('Head', color='black', size=14)
    label_ = 'rolling mean (window width = 1 day)'
    df_process_['head [mWC]'].rolling('D', min_periods=8000).mean()[::step_size_].plot(ax=ax, label=label_)
    ax.legend()
    # df_cavitation_risk_['cavitation_risk'].plot(ax=ax)
    ax = plot_band(ax, axvspan_start_time_, axvspan_stop_time_)
    ax = x_labels(ax, minor_day_interval=minor_day_interval, major_day_interval=major_day_interval)
    ax.set_xlim(pd.to_datetime(start_time_), pd.to_datetime(stop_time_))
    ax.set_ylim([500, 2500])
    ax.set_ylabel('head [mWC]')

    fig.tight_layout()
    return fig, axes, df_


def x_labels(ax, minor_day_interval=1, major_day_interval=7):
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=minor_day_interval))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=major_day_interval))
    #ax.xaxis.set_major_formatter(
    #    mdates.DateFormatter('%d-%m-%Y')
    #)
    #ax.xaxis.set_major_formatter(
    #    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    return ax
