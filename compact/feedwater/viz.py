# ©, 2024, Sirris
# owner: FFNG
from compact.viz.viz import get_controller, get_number_of_components, _extract_upper_limit, _extract_lower_limit
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from compact.feedwater import data
import pandas as pd
import plotly.express as px
import datetime
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from ipywidgets import interact

from matplotlib.gridspec import GridSpec

previous_value_threshold = 95


def show_fingerprints(fingerprints_):
    """ Code adapted from conscious_engie_icare/viz/viz module for showing fingerprints in interactive way. """
    possible_oms = list(fingerprints_.keys())
    controller_om = get_controller({'widget': 'Dropdown',
                                    'options': [(_om, _om) for _om in possible_oms],
                                    'value': possible_oms[0],
                                    'description': 'Select the operating mode'})
    
    def make_plot(om):
        df_ = fingerprints_[om]
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.heatmap(df_, annot=True, fmt=".3f", ax=ax, cmap='Blues', vmin=0, vmax=0.01, cbar=False)
        ax.set_title(f'Vibration fingerprint @ OM {om}')
        ax.set_xlabel('component')
        fig.show()
    
    interact(make_plot, om=controller_om)


def visualize_spider_chart(df_contextual_train_with_labels, short_form=None):
    short_form = short_form or ['v', 'p-delta-MP', 'p-in', 'p-out', 'p-delta', 'p-fwt', 't-in', 't-fwt']

    def calc_params(s):
        r = pd.concat([s, pd.Series(s.iloc[0])])
        theta = short_form + [short_form[0]]
        return {'r': r, 'theta': theta}

    om_dict = {om: group.iloc[0].pump for om, group in df_contextual_train_with_labels.groupby('cluster_kmeans')}

    nrows = 3
    ncols = 6
    CLUSTER_ORDER = df_contextual_train_with_labels.groupby('cluster_kmeans')\
        .max().sort_values(by=['pump', 'velocity <v-MP> [RPM]']).index
    titles = ["<span style='color:#ff0000;'>" + om + " (" + str(om_dict[om]) + ") </span>" for om in CLUSTER_ORDER]
    fig = make_subplots(rows=nrows, cols=ncols, specs=[[{'type': 'polar'}]*ncols]*nrows, subplot_titles=titles)

    df_ = df_contextual_train_with_labels.copy()
    scaler = MinMaxScaler()
    df_[data.CONTEXTUAL_COLUMNS] = scaler.fit_transform(df_[data.CONTEXTUAL_COLUMNS])
    for i, cluster in enumerate(CLUSTER_ORDER, start=0):
        group = df_[df_.cluster_kmeans == cluster]
        fig_row = i//ncols+1
        fig_col = i % ncols+1

        # Assign maximum value
        fig.add_trace(go.Scatterpolar(
            name=cluster,
            **calc_params(group[data.CONTEXTUAL_COLUMNS].max()),
            text=pd.DataFrame(scaler.inverse_transform(group[data.CONTEXTUAL_COLUMNS]),
                              columns=data.CONTEXTUAL_COLUMNS).max(),
            line=dict(color='red', dash='solid'),
        ), fig_row, fig_col)

        # Assign minimum value
        fig.add_trace(go.Scatterpolar(
            name=cluster,
            **calc_params(group[data.CONTEXTUAL_COLUMNS].min()),
            text=pd.DataFrame(scaler.inverse_transform(group[data.CONTEXTUAL_COLUMNS]),
                              columns=data.CONTEXTUAL_COLUMNS).min(),
            line=dict(color='blue', dash='solid'),
        ), fig_row, fig_col)

        # Assign median value
        fig.add_trace(go.Scatterpolar(
            name=cluster,
            **calc_params(group[data.CONTEXTUAL_COLUMNS].median()),
            text=pd.DataFrame(scaler.inverse_transform(group[data.CONTEXTUAL_COLUMNS]),
                              columns=data.CONTEXTUAL_COLUMNS).median(),
            line=dict(color='black', dash='solid'),
            # mode = 'markers',
            opacity=0.5,
        ), fig_row, fig_col)

    fig.update_layout(height=600, width=1100, showlegend=False, title_text="Bounding boxes per operating mode")
    fig.update_polars(angularaxis_rotation=28,
                      radialaxis=dict(range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], tickfont={'size': 8},
                                      showticklabels=True))

    fig.show(renderer='svg')


def plot_number_of_measurements(df_contextual, SPLIT_DATE='2021-10-01'):
    tmp = df_contextual.groupby(['pump']).resample('D', on='timestamp').count().iloc[:, 0]
    tmp.name = 'counts'
    df_plot_ = tmp.reset_index().sort_values('timestamp')
    fig = px.line(df_plot_, x='timestamp', y='counts', facet_row='pump',
                  title='Process data: Number of process measurement vectors per day',
                  labels={"timestamp": "Date", "counts": "Measurements"}, width=800, height=500)
    fig.update_traces(opacity=0.5)

    fig.add_vline(
        x=datetime.datetime.strptime(str(SPLIT_DATE), "%Y-%m-%d").timestamp() * 1000,
        line_width=3, line_dash="dash", line_color="green",
        row="all", col="all", annotation_text='split date', annotation_bgcolor='green'
    )
    fig.show(renderer='svg')


def visualize_scores(df_scores):
    # Illustrate
    df_ = pd.melt(
        df_scores, id_vars=['number_clusters_tested', 'X_compressed', 'y', 'n_components', 'pipeline', 'pump'],
        value_vars=['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
    )
    fig = px.line(df_, x='number_clusters_tested', y='value', facet_col='variable', facet_row='pump', width=1000,
                  height=600, markers=True)
    fig.update_yaxes(matches=None)
    fig.show(renderer='svg')


def show_distances(df_merged_all_train, df_merged_all, threshold_quantile=0.9975):

    thresholding_value = df_merged_all_train['dist_of_closest_om'].quantile(threshold_quantile)

    fig, axes = plt.subplots(figsize=(8, 3), ncols=2, sharey=True)

    # left: training set
    ax = axes[0]
    ax = df_merged_all_train['dist_of_closest_om'].plot.hist(bins=50, ax=ax)
    ax.set_title('TRAIN SET', fontsize=16, color='black')
    ax.set_xlabel('$d_{cos}$')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 0.75)
    ax.axvline(x=thresholding_value, color='red', linestyle='dashed', label=f'{threshold_quantile} quantile')

    # right: test set
    ax = axes[1]
    ax = df_merged_all['dist_of_closest_om'].plot.hist(bins=50, ax=ax)
    ax.set_title('TEST SET', fontsize=16, color='black')
    ax.set_xlabel('$d_{cos}$')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 0.75)
    ax.axvline(x=thresholding_value, color='red', linestyle='dashed', label=f'{threshold_quantile} quantile')

    fig.tight_layout()

    return thresholding_value


def plot_cumulative_anomaly_score_per_pump(df_merged_all, thresholding_value):
    RESET_THRESHOLD_1 = pd.Timedelta(value=4, unit='day')
    RESET_THRESHOLD_2 = 7
    SELECTED_PERIODS = [
        {'pump': 1, 'start': pd.to_datetime('2022-03-15'), 'stop': pd.to_datetime('2022-04-07')},
        {'pump': 3, 'start': pd.to_datetime('2022-06-16'), 'stop': pd.to_datetime('2022-06-19')},
    ]

    fig, axes = plt.subplots(figsize=(8, 8), nrows=3, sharex=True, sharey=True)
    dfs_with_global_anomaly_score_dynamic = {}
    alarms = {'1': [], '2': [], '3': []}
    for pump, ax in tqdm(zip(['1', '2', '3'], axes), total=3):
        df_anom_ = df_merged_all.copy().set_index('timestamp').sort_index()
        df_anom_ = df_anom_[df_anom_['pump'] == pump]
        df_anom_['anomaly'] = df_anom_['dist_of_closest_om'] > thresholding_value
        cumulative_reset_counts = []
        val = 0
        cumulative_anomalies = []
        ts_last_point_anomaly = df_anom_.iloc[0].name
        # time_delta = pd.Timedelta(value=0, unit='day')
        for idx, row in df_anom_.iterrows():
            time_delta = row.name - ts_last_point_anomaly
            if row['anomaly']:
                val = val + 1
                ts_last_point_anomaly = row.name
                if val > 0 and val % RESET_THRESHOLD_2 == 0:
                    val = val + 1
                    alarms[pump].append(row.name)
                    ts_last_point_anomaly = row.name
                    cumulative_anomalies.append(1)
                else:
                    cumulative_anomalies.append(0)
            elif val > 0 and time_delta >= RESET_THRESHOLD_1:
                val = 0
                # commented out following line to not set an alarm
                # alarms[pump].append(ts_last_point_anomaly + RESET_THRESHOLD_1)
                cumulative_anomalies.append(0)  # changed from 1 to 0, s.t. no alarm is thrown when reset
            else:
                cumulative_anomalies.append(0)
            # val = 0 if val >= RESET_THRESHOLD_1 else val
            cumulative_reset_counts.append(val)
        df_anom_['aggregated_anomaly_score'] = cumulative_reset_counts
        df_anom_['aggregated_anomaly_score'].plot(title=f'aggregated anomaly count (reset after {RESET_THRESHOLD_2})',
                                                  ax=ax)
        # plot red dot whenever RESET_THRESHOLD_1 is reached
        df_anom_['cumulative_anomalies'] = cumulative_anomalies
        # df_anomalies_ = df_anom_[df_anom_['cumulative_anomalies'] == 1]
        dfs_with_global_anomaly_score_dynamic[pump] = df_anom_
        # for i, (idx, row) in enumerate(df_anomalies_.iterrows()):
        #    label = 'alarm' if i == 0 else None
        #    ax.scatter(row.name, RESET_THRESHOLD_1.days-1, s=500, color='red', marker='|', label=label)
        for i, (ts) in enumerate(alarms[pump]):
            label = 'alarm' if i == 0 else None
            ax.scatter(ts, RESET_THRESHOLD_1.days - 1, s=500, color='red', marker='|', label=label)
        ax.set_title(f'Pump {pump}')
        ax.set_ylabel('Aggregated anomaly count')
        ax.set_xlabel('Timestamp')
        ax.text(0.025, 0.2, f'Number of alarms: {len(alarms[pump])}', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        for i, selected_period in enumerate(SELECTED_PERIODS):
            print(selected_period)
            label_ = 'selected period'  # if i == 0 else None
            if pump == str(selected_period['pump']):
                ax.axvspan(selected_period['start'], selected_period['stop'], alpha=0.2,
                           color='red', label=label_)
        ax.legend(loc='upper left')

    fig.tight_layout()
    fig.show()
    return dfs_with_global_anomaly_score_dynamic


def plot_example_anomaly(dfs_with_global_anomaly_score_dynamic, process_view_test_with_results, pump,
                         axvspan_start_time, axvspan_stop_time, label_size):
    fig, axes = plt.subplots(figsize=(7, 12), nrows=6, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 2]})
    fig.subplots_adjust(hspace=0.4)

    # aggregated anomaly count
    ax = axes[0]
    ax.set_ylabel('Aggregated anomaly count', color='red', size=label_size)
    df_ = dfs_with_global_anomaly_score_dynamic[str(pump)]
    df_['aggregated_anomaly_score'].plot(ax=ax, color='red', label='')
    # create annotation between 17.06. and 20.06.
    if (axvspan_start_time is not None) and (axvspan_stop_time is not None):
        ax.axvspan(pd.to_datetime(axvspan_start_time), pd.to_datetime(axvspan_stop_time), alpha=0.2,
                   label='surge in anomaly score', color='red')
    ax.legend(loc='upper left')

    # add line plot for operating mode
    ax3 = axes[2]
    # ax3.spines['right'].set_position(('outward', 60))
    df_plot_ = df_.copy()
    name_dict = dict(zip(sorted(df_['process view: closest operating mode'].unique()),
                         list(range(len(df_plot_['process view: closest operating mode'].unique())))))
    df_plot_['process view: closest operating mode'] = df_plot_['process view: closest operating mode'].replace(
        name_dict)
    # map operating mode to number
    df_plot_['process view: closest operating mode'].plot(ax=ax3, color='blue', label='Operating Mode')
    ax3.set_ylabel('Operating Mode', size=label_size)
    if (axvspan_start_time is not None) and (axvspan_stop_time is not None):
        ax3.axvspan(pd.to_datetime(axvspan_start_time), pd.to_datetime(axvspan_stop_time), alpha=0.2,
                    label='surge in anomaly score', color='red')
    ax3.tick_params(axis='y')
    # only have ticks for all operating modes
    ax3.set_yticks(list(range(len(df_plot_['process view: closest operating mode'].unique()))))
    # Replace numeric y-ticks with keys from name_dict
    yticks = ax3.get_yticks()
    yticklabels = [key for key, value in name_dict.items() if value in yticks]
    ax3.set_yticklabels(yticklabels)
    # ax3.legend(loc='upper right')

    # add line plot for operating mode certainty
    ax4 = axes[3]
    df_plot_ = df_.copy()
    df_plot_['certainty_score'].rolling('H').mean().plot(ax=ax4, color='green', label='Certainty score')
    ax4.set_ylabel('certainty score', color='black', size=label_size)
    if (axvspan_start_time is not None) and (axvspan_stop_time is not None):
        ax4.axvspan(pd.to_datetime(axvspan_start_time), pd.to_datetime(axvspan_stop_time), alpha=0.2,
                    label='highlighted period', color='red')
    ax4.tick_params(axis='y')

    # add line plot for number of timestamps with unknown operating modes
    ax5 = axes[4]
    df_plot_ = process_view_test_with_results[pump].copy()
    print((~df_plot_.OM.isna()).sum())
    df_plot_['unknown_operating_mode'] = 0  # SK TODO: calculate
    # print(df_plot_.head())
    df_plot_.set_index('timestamp')['unknown_operating_mode'].rolling('H').mean().plot(ax=ax5, color='purple',
                                                                                       label='unknown operating mode')
    ax5.set_ylabel('unknown operating mode', color='black', size=label_size)
    if (axvspan_start_time is not None) and (axvspan_stop_time is not None):
        ax5.axvspan(pd.to_datetime(axvspan_start_time), pd.to_datetime(axvspan_stop_time), alpha=0.2,
                    label='highlighted period', color='red')
    ax5.tick_params(axis='y')

    fig.suptitle(f'Example of detected anomaly (pump {pump})', size=16)

    axes[-1].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.show()


def illustrate_nmf_components_interactive(df_V_train, df_nmf_models):
    global previous_value_threshold
    previous_value_threshold = 95

    saved_values = {}

    max_n_components = len(df_nmf_models)

    def make_plot(criterium, n_components_range, **extra_options):
        global previous_value_threshold

        if 'min_explained_variance' in extra_options.keys():
            threshold = extra_options['min_explained_variance']
        else:
            threshold = previous_value_threshold

        previous_value_threshold = threshold

        n_components_variance, n_components_knee_pca, n_components_knee_nmf = get_number_of_components(
            explained_variance_ratio,
            threshold,
            df_nmf_models)

        if criterium == 'explained-variance':
            n_components = n_components_variance
        elif criterium == 'knee-point':
            n_components = max(n_components_knee_nmf, n_components_knee_pca)
        elif criterium == 'both':
            n_components = max(n_components_variance, n_components_knee_nmf, n_components_knee_pca)
        else:
            n_components = 5

        n_components = max(n_components, n_components_range[0])
        n_components = min(n_components, n_components_range[1])

        saved_values['n_components'] = n_components

        components_per_row = 3
        n_rows_first_plot = 3
        n_rows = n_rows_first_plot + int(np.ceil(n_components / components_per_row))
        n_cols = 2 * components_per_row

        fig = plt.figure(layout="constrained", figsize=(12, n_rows))

        gs = GridSpec(n_rows, n_cols, figure=fig)
        ax1 = fig.add_subplot(gs[:n_rows_first_plot, :n_cols // 2])
        ax2 = fig.add_subplot(gs[:n_rows_first_plot, n_cols // 2:])
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))

        ax1.plot(np.arange(1, max_n_components + 1), explained_variance_ratio, 'ko-')
        ax2.plot(df_nmf_models['n_components'],
                 df_nmf_models['reconstruction_error'], 'ko-')

        if criterium in ('both', 'explained-variance'):
            ax1.axhline(y=extra_options['min_explained_variance'], linestyle=':', color='g')
            ax1.axvline(x=n_components_variance, linestyle=':', color='g',
                        label=f'Explained variance PCA (N={n_components_variance})')
        if criterium in ('both', 'knee-point'):
            ax1.axvline(x=n_components_knee_pca, linestyle=':', color='b',
                        label=f'Knee point PCA (N={n_components_knee_pca})')
            ax2.axvline(x=n_components_knee_nmf, linestyle=':', color='r',
                        label=f'Knee point NMF (N={n_components_knee_nmf})')
            ax2.legend(loc='upper right', fontsize='small')
        ax1.set_xlabel("Number of components")
        ax1.set_ylabel("Cumulative explained variance")
        ax2.set_xlabel("Number of components")
        ax2.set_ylabel("Frobenium norm")

        ax1.legend(loc='lower right', fontsize='small')

        row = df_nmf_models[df_nmf_models.n_components == n_components].iloc[0]
        H = row.H

        axes_components = []

        column_names = df_V_train.columns[0]
        offset = 0.5 * (_extract_upper_limit(column_names) - _extract_lower_limit(column_names))
        all_x_label_names = [_extract_lower_limit(col) + offset for col in df_V_train.columns]
        plot_x_ticks = [0, 20, 40, 60, 80, 100, 119]
        plot_x_labels = [all_x_label_names[idx] for idx in plot_x_ticks]

        for _i_component in range(n_components):
            i_row = n_rows_first_plot + (_i_component // components_per_row)
            i_col = 2 * (_i_component % components_per_row)

            if len(axes_components) > 0:
                _ax = fig.add_subplot(gs[i_row, i_col:i_col + 2],
                                      sharey=axes_components[0], sharex=axes_components[0])
            else:
                _ax = fig.add_subplot(gs[i_row, i_col:i_col + 2])

            _ax.plot(np.arange(H.shape[1]), H.iloc[_i_component])
            text_y = 0.8 * np.max(H)
            _ax.text(0, text_y, f'Component {_i_component + 1}', fontdict={'size': 10})
            _ax.set_xticks(plot_x_ticks)
            _ax.set_xticklabels(plot_x_labels)

            axes_components.append(_ax)
            if i_row == (n_rows - 1):
                _ax.set_xlabel('Frequency [orders]')

        fig.show()

    def get_additional_options(criterium, n_components_range):
        global previous_value_threshold

        threshold = 95 if previous_value_threshold is None else previous_value_threshold
        if criterium in ('explained-variance', 'both'):
            controller_explained_variance = get_controller({'widget': 'FloatSlider',
                                                            'min': min_variance,
                                                            'max': 100,
                                                            'value': threshold,
                                                            'description': 'Explained PCA variance threshold'})
            extra_options = {'min_explained_variance': controller_explained_variance}
        else:
            extra_options = {}

        def _make_plot(**_extra_options):
            make_plot(criterium, n_components_range, **_extra_options)

        interact(_make_plot, **extra_options)

    # calculate explained variance
    pca = PCA(n_components=max_n_components, random_state=42)
    pca.fit(df_V_train)
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    explained_variance_ratio = explained_variance_ratio[:max_n_components] * 100
    min_variance = np.ceil(np.min(explained_variance_ratio))

    controller_selection_criterium = get_controller({'widget': 'Dropdown',
                                                     'options': [('Explained variance', 'explained-variance'),
                                                                 ('Knee point', 'knee-point'),
                                                                 ('Both', 'both')],
                                                     'value': 'explained-variance',
                                                     'description': 'How to select the number of components for NMF'})

    controller_n_components_range = get_controller({'widget': 'IntRangeSlider',
                                                    'min': 1,
                                                    'max': max_n_components,
                                                    'value': [2, 10],
                                                    'description': 'Range of acceptable number of components for NMF'})

    interact(get_additional_options, criterium=controller_selection_criterium,
             n_components_range=controller_n_components_range)

    return saved_values
