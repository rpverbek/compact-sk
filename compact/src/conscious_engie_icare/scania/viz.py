# ©, 2022, Sirris
# owner: FFNG

import matplotlib.pyplot as plt
from conscious_engie_icare.scania.dataset import get_attribute_columns
import matplotlib as mpl
import os
from tqdm import tqdm
import numpy as np
from ydata_profiling import ProfileReport


FPATH_PROFILING_REPORTS = os.path.join('profiling_reports')


def plot_cumulative_timeseries(df, vehicle_ids, attribute_columns=None):
    if attribute_columns is None:
        attribute_columns = get_attribute_columns(df)
    
    unique_features = set(int(col.split('_')[0]) for col in attribute_columns)
    fig, axes = plt.subplots(figsize=(6*len(unique_features), 4*len(vehicle_ids)),
                             nrows=len(vehicle_ids), ncols=len(unique_features), sharex=True)
    if (len(vehicle_ids) > 1) and (len(unique_features) > 1):
        for vehicle_id, axes_row in zip(vehicle_ids, axes):
            df_ = df[df.vehicle_id == vehicle_id].set_index('time_step')
            for feature, ax in zip(unique_features, axes_row):
                df_plot_ = df_[df_.columns[df_.columns.str.contains(str(feature))]]
                #df_plot_.plot(ax=ax, marker='o')
                ax.plot(df_plot_, marker='o')
                ax.set_xlabel('timestamp')
                ax.set_ylabel('Cumulative count')
                ax.set_title(f'{feature} (truck {vehicle_id})')
    elif len(vehicle_ids) == 1:
        df_ = df[df.vehicle_id == vehicle_ids[0]].set_index('time_step')
        for feature, ax in zip(unique_features, axes):
            df_plot_ = df_[df_.columns[df_.columns.str.contains(str(feature))]]
            ax.plot(df_plot_, marker='o')
            ax.set_xlabel('abstract timestamp unit')
            ax.set_ylabel('cumulative count')
            ax.set_title(f'{feature} (truck {vehicle_ids[0]})')
    elif len(unique_features) == 1:
        feature = list(unique_features)[0]
        for vehicle_id, ax in zip(vehicle_ids, axes):
            df_ = df[df.vehicle_id == vehicle_id].set_index('time_step')
            df_plot_ = df_[df_.columns[df_.columns.str.contains(str(feature))]]
            ax.plot(df_plot_, marker='o')
            ax.set_xlabel('abstract timestamp unit')
            ax.set_ylabel('cumulative count')
            ax.set_title(f'{feature} (truck {vehicle_id})')
    else:
        raise NotImplementedError()
    fig.tight_layout()
    return fig, axes


def plot_distributions(df_, max_quantile=None, bins=50):
    df_ = df_.drop(columns=['vehicle_id', 'time_step'])

    # Determine the grid size for subplots
    rows = int(len(df_.columns)**0.5)
    cols = (len(df_.columns) // rows) + (len(df_.columns) % rows > 0)
    
    # Create a figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each column's distribution in a separate subplot
    for i, col in tqdm(enumerate(df_.columns)):
        series = df_[col]
        # series = series.replace([np.inf, -np.inf], 0)
        if max_quantile is not None:
            series = series[series < series.quantile(max_quantile)]
        series.plot(kind='hist', ax=axes[i], title=col, bins=bins)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    return fig, axes


def plot_V(V):
    cmap = mpl.cm.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(6, 8))
    nrows = V.shape[0]
    ncols = V.shape[1]
    title_ = "Performance matrix V" + f" ({nrows} x {ncols})"
    ax.set_title(title_, fontsize=20)
    # V.columns = BAND_COLUMNS
    im = ax.imshow(
        V,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        norm=mpl.colors.LogNorm(vmin=None, vmax=None),
        # extent=[0.25,30.25,nrows,0]
    )
    ax.tick_params(axis='y', labelrotation=90)
    return fig, ax


def generate_report(df_, fpath, bpath=FPATH_PROFILING_REPORTS, minimal=True, title=None, **kwargs):
    path = os.path.join(bpath, fpath)
    if not os.path.exists(path):
        report = ProfileReport(df_, title=title, minimal=minimal, progress_bar=True, **kwargs)
        report.to_file(path)
    else:
        print(f'Not producing report {fpath}. Report exists already.')
