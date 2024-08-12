import os
import posixpath
# from elucidata.resources.pipeline import DataFrameDownload
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import colors as pltcolors
from matplotlib.ticker import LinearLocator
from matplotlib import ticker
from conscious_engie_icare import LATEST_EXPERIMENT


def preprocess_operational_features(pump, file_path):
    """
    file_name = f'operational_features_pump{pump}_{LATEST_EXPERIMENT}.csv'
    file_path = os.path.join(local_path, file_name)
    remote_path = posixpath.join(remote_path, file_name)
    df = DataFrameDownload(file_path, remote_path).make()
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pump'] = pump
    return df


def illustrate_explained_variance(df, cumulative=True, ax=None):
    # fit PCA pipe with min-max scaler
    pca_pipe = Pipeline([('scaler', MinMaxScaler()), ('pca', PCA())])
    pca_pipe.fit(df.drop(columns=['timestamp', 'pump']))
    y = pca_pipe['pca'].explained_variance_ratio_
    if cumulative:
        y = y.cumsum()
    ax.plot(range(1, len(y)+1), y, marker='o', label='minmax')

    # fit PCA pipe with standard scaler
    pca_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
    pca_pipe.fit(df.drop(columns=['timestamp', 'pump']))
    y = pca_pipe['pca'].explained_variance_ratio_
    if cumulative:
        y = y.cumsum()
    ax.plot(range(1, len(y)+1), y, marker='o', label='z-score')
    ax.set_xlabel('principal component')
    ax.set_ylabel('explained variance')
    ax.legend()
    return ax


def get_comp(pca_pipe, df, comp):
    return pca_pipe.fit_transform(df.drop(columns=['timestamp', 'pump']))[:,comp]


def find_optimal_number_of_clusters(X,
                                    dimensionality_reduction_pipe,
                                    algorithm=KMeans,
                                    min_number_clusters=2, max_number_clusters=6,
                                    base_folder=None,
                                    file_name_base=None,
                                    **clustering_param):
    """ Compare clusters with respect to specific metrics. """
    X_file_name = os.path.join(base_folder, f'X_compressed_{file_name_base}.csv')

    # extract compressed feature space
    X_compressed = dimensionality_reduction_pipe.fit_transform(X)

    # calculate internal clustering metrics for different number of clusters
    scores = []
    for n_clusters in range(min_number_clusters, max_number_clusters + 1):
        y_file_name = os.path.join(base_folder, f'y_{file_name_base}_{n_clusters}-clusters.csv')
        y = KMeans(n_clusters=n_clusters, **clustering_param).fit_predict(X_compressed)
        y = y + 1  # start with cluster 1
        s = {
            'number_clusters_tested': n_clusters,
            'silhouette_score': metrics.silhouette_score(X_compressed, y),
            'calinski_harabasz_score': metrics.calinski_harabasz_score(X_compressed, y),
            'davies_bouldin_score': metrics.davies_bouldin_score(X_compressed, y),
            'X_compressed': X_file_name,
            'y': y_file_name,
        }
        scores.append(s)
        np.savetxt(y_file_name, y, delimiter=",")
        np.savetxt(X_file_name, X_compressed, delimiter=",")
    return pd.DataFrame(scores)


def calculate_score(X, dimensionality_reduction, name,
                    n_components=None, max_number_clusters=5,
                    clustering=KMeans, base_folder='cache_scores',
                    **param_dimensionality_reduction):
    # try to load scores from cache
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    scores_name = os.path.join(base_folder, f'{name}.csv')
    try:
        df_scores = pd.read_csv(scores_name)
    except FileNotFoundError:
        print(f'Calculating scores for {scores_name}')

        # calculate scores for different number of components
        n_components = [2, 3, 4] if n_components is None else n_components
        scores = []
        for n in n_components:

            # dimensionality reduction
            dimensionality_reduction_pipe = Pipeline([
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', dimensionality_reduction(n_components=n,
                                                                      random_state=None,
                                                                      **param_dimensionality_reduction))
            ])
            s = find_optimal_number_of_clusters(X, dimensionality_reduction_pipe,
                                                max_number_clusters=max_number_clusters,
                                                algorithm=clustering,
                                                random_state=100,
                                                base_folder=base_folder,
                                                file_name_base=f'{name}_{n}-components')
            s['n_components'] = n

            # pickle pipeline
            name_pipeline = os.path.join(base_folder, name + f'_pipeline_n{n}.pickel')
            s['pipeline'] = name_pipeline
            with open(name_pipeline, 'wb') as file:
                pickle.dump(dimensionality_reduction_pipe, file, protocol=pickle.HIGHEST_PROTOCOL)

            # add information to scores
            scores.append(s)
        df_scores = pd.concat(scores)
        df_scores.to_csv(scores_name)

    return df_scores


def plot_scores_per_components(group, cols, fig, number_clusters_tested, pump, score='silhouette_score'):
    """Plot score per number of components and number of clusters."""
    col = cols[pump]
    showlegend = True if col == 1 else False

    # 2 clusters in black
    color_template = list(fig.layout['template']['layout']['colorway'])
    color_template[2] = '#000000'
    color_template[5] = color_template[7]
    color_template = tuple(color_template)

    plot = go.Scatter(x=group['n_components'], y=group[score],
                      marker_color=color_template[number_clusters_tested],
                      showlegend=showlegend,
                      name=f"{number_clusters_tested}")
    fig.add_trace(plot, row=1, col=col)
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=10))


def order_cluster_names(df, add_pump_to_cluster_name=True, col_name_clusters='cluster'):
    # rename cluster s.t. each name is unique across pumps,
    # e.g. there should be no OM 1 in pump 11 and pump 12 at the same time
    offset = {}
    tmp = 0
    for pump, group in df.groupby('pump'):
        offset[pump] = tmp
        n_unique_clusters = len(group[col_name_clusters].unique())
        tmp += n_unique_clusters
    series_offset = df['pump'].replace(offset)
    df[col_name_clusters] = df[col_name_clusters] + series_offset

    # extract mean RPM per cluster and keep track of corresponding pump
    df_grouped = df.groupby(col_name_clusters)
    mean_rpm = df_grouped.mean()['velocity <v-MP> [RPM]']
    corresponding_pump = df_grouped.first()['pump']

    # rename cluster according to asceding mean RPM,
    # i.e. lowest RPM --> Cluster0, second lowest RPM --> Cluster1
    sorted_c = mean_rpm.sort_values(ascending=True)
    om_to_pump = dict(list(zip(sorted_c.index, corresponding_pump.loc[sorted_c.index])))
    name_mapping = dict(list(zip(sorted_c.index, range(len(sorted_c)))))
    df[col_name_clusters] = df[col_name_clusters].replace(name_mapping)

    """
    # rename cluster from numeric to alphabetic
    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z')+1)))
    replace_oms = dict(list(zip(sorted(df[col_name_clusters].unique()), alphabet_upper_case)))
    om_to_pump = {replace_oms[k]: v for k, v in om_to_pump.items()}
    print(om_to_pump)
    # global_name_mapping = {k: v for k, v in om_to_pump.items()}
    df[col_name_clusters] = df[col_name_clusters].replace(to_replace=replace_oms)
    """

    # create unique name that includes pump type
    if add_pump_to_cluster_name:
        for pump, group in df.groupby('pump'):
            sorted_col_names = sorted(group[col_name_clusters].unique())
            for i, col_name in enumerate(sorted_col_names):
                name_mapping[col_name] = f'p{pump}_c{i}'
        df[col_name_clusters] = df[col_name_clusters].replace(name_mapping)

    return df, False


def plot_vibrations(df_vib, freq_values, row_id=1600, ax=None, **plot_kwargs):
    row = df_vib.iloc[row_id]
    ax = ax if ax is not None else plt.subplots(figsize=(20, 5))[1]
    ax = row[freq_values].plot(ax=ax, **plot_kwargs)
    title = f'exemplary measurement in frequency domain (pump={row.pump}, location={row.location}, direction={row.direction})'
    if 'operating_mode' in row:
        title = title[:-1] + f', operating mode = {row.operating_mode})'
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    return ax


def plot_vibrations_as_orders(df_vib, freq_values, row_id=1600, plot_x_axis_as_tuple=False, ax=None, title=None):
    row = df_vib.iloc[row_id]
    velocity = row['velocity [Hz]']
    vibrations_ = row[freq_values]
    orders = vibrations_.index.astype(float) / velocity
    if plot_x_axis_as_tuple:
        vibrations_.index = zip(vibrations_.index, orders)
    else:
        vibrations_.index = orders
    ax = ax if ax is not None else plt.subplots(figsize=(20, 5))[1]
    ax = vibrations_.plot(label='', ax=ax)
    if title is None:
        title = f'exemplary measurement in frequency domain (pump={row.pump}, location={row.location}, direction={row.direction})'
        if 'operating_mode' in row:
            title = title[:-1] + f', operating mode = {row.operating_mode})'
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel('Frequency [orders]')
    return ax


def plot_scores(df_scores, pump=11):
    df = df_scores.set_index('n_clusters')
    fig, axes = plt.subplots(figsize=(14, 3), ncols=4, sharex=True)
    axes[0].plot(df['distorsions'], marker='o')
    axes[0].set_title(f'Elbow curve (pump {pump})');
    axes[1].plot(df['chs'], marker='o')
    axes[1].set_title('Calinski Harabasz Score (larger values better)');
    axes[2].plot(df['dbs'], marker='o')
    axes[2].set_title('Davies Bouldin Score (smaller values better)');
    axes[3].plot(df['silhouette'], marker='o')
    axes[3].set_title('Silhouette Score (larger values better)');
    fig.tight_layout()
    return fig


def plot_color_map(df, x_name='p_in [barg]', y_name='global_cavitation_risk_pow_2', z_name='q [m3/h]',
                   x_bins=None, y_bins=None, ax=None, vmin=0, vmax=1, plot_type='mean'):
    x = df[x_name].to_numpy()
    y = df[y_name].to_numpy()
    z = df[z_name].to_numpy()

    if x_bins is None:
        x_bins = np.linspace(df[x_name].min(), df[x_name].max(), num=50)
    if y_bins is None:
        y_bins = np.linspace(df[y_name].min(), df[y_name].max(), num=50)

    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=z)
    H_counts, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    if plot_type == 'mean':
        norm=None
        H = H/H_counts
    elif plot_type == 'count':
        norm = pltcolors.LogNorm(vmin=1, vmax=H_counts.max())
        vmin=None
        vmax=None
        H = H_counts

    def round_to_multiple(number, multiple):
        return multiple * round(number / multiple)

    n_labels = len(H) + 1
    ax.xaxis.set_minor_locator(LinearLocator(numticks=n_labels))
    ax.yaxis.set_minor_locator(LinearLocator(numticks=n_labels))

    im = ax.imshow(H.T, origin='lower', cmap='viridis', norm=norm,
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=vmin, vmax=vmax)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    # ax.set_aspect((df[x_name].max() - df[x_name].min()) / (df[y_name].max() - df[y_name].min()))
    ax.set_aspect('auto')
    return ax, im


def add_colorbar(fig, im, **kwargs):
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
    # ax.set_xticklabels(xedges)
    fig.colorbar(im, cax=cbar_ax, **kwargs)
    return fig


def plot_exemplar_frequency_measurements(df_demonstrate, FREQ_VALUES, row_id1=1400, row_id2=900):
    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, sharex=True, sharey=True)
    ax = plot_vibrations(df_demonstrate, FREQ_VALUES, row_id=row_id1, label='measurement 1', ax=axes[0])
    ax = plot_vibrations(df_demonstrate, FREQ_VALUES, row_id=row_id2, label='measurement 2', ax=axes[1])
    ax.set_ylim(0, 0.5)


def plot_orders(ax):
    """ Postprocessing of plotting vibration as orders. """
    ax.axvline(7, color='black', alpha=0.33, label='peak at 7 orders')
    ax.axvline(10, color='orange', alpha=0.33, label='peak at 10 orders')
    ax.axvline(12, color='yellow', alpha=0.33, label='peak at 12 orders')
    ax.axvline(14, color='green', alpha=0.33, label='peak at 14 orders')
    ax.axvline(20, color='red', alpha=0.33, label='peak at 20 orders')
    ax.axvline(21, color='blue', alpha=0.33, label='peak at 21 orders')
    ax.legend()
    ax.set_ylim([0, 0.5])


def plot_exemplar_frequency_transformation(df_demonstrate, FREQ_VALUES, row_id1=1400, row_id2=900):
    """Plot frequency transformation for paper."""
    # fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=2, sharex=True, sharey=True)
    fig = plt.figure(figsize=(8, 6))
    ax_tl = fig.add_subplot(2, 2, 1)
    ax_tr = fig.add_subplot(2, 2, 2)
    ax_bl = fig.add_subplot(2, 2, 3, sharex=ax_tl, sharey=ax_tl)
    ax_br = fig.add_subplot(2, 2, 4, sharex=ax_tr, sharey=ax_tr)

    def plot_stripes_at_frequencies(row_id):
        """Plot stripes at frequencies that correspond to 7, 10, 12, 14, 20 and 21 orders."""
        row = df_demonstrate.iloc[row_id]
        velocity = row['velocity [Hz]']
        vibrations_ = row[FREQ_VALUES]
        orders = vibrations_.index.astype(float) / velocity
        x7 = len(orders[orders <= 7])
        x10 = len(orders[orders <= 10])
        x12 = len(orders[orders <= 12])
        x14 = len(orders[orders <= 14])
        x20 = len(orders[orders <= 20])
        x21 = len(orders[orders <= 21])
        ax.axvline(x7, color='black', alpha=0.33, label=f'peak at {x7} Hz')
        ax.axvline(x10, color='orange', alpha=0.33, label=f'peak at {x10} Hz')
        ax.axvline(x12, color='yellow', alpha=0.33, label=f'peak at {x12} orders')
        ax.axvline(x14, color='green', alpha=0.33, label=f'peak at {x14} orders')
        ax.axvline(x20, color='red', alpha=0.33, label=f'peak at {x20} orders')
        ax.axvline(x21, color='blue', alpha=0.33, label=f'peak at {x21} orders')
        ax.legend()

    # plot raw frequency measurement 1 in top left panel
    ax = plot_vibrations(df_demonstrate, FREQ_VALUES, label='', row_id=row_id1, ax=ax_tl)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel('acceleration')
    ax.set_title('Unprocessed frequency measurement A')
    plot_stripes_at_frequencies(row_id1)

    # plot raw frequency measurement 2 in bottom left panel
    ax = plot_vibrations(df_demonstrate, FREQ_VALUES, label='', row_id=row_id2, ax=ax_bl)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel('acceleration')
    ax.set_title('Unprocessed frequency measurement B')
    plot_stripes_at_frequencies(row_id2)

    # plot transformed frequency measurement 1 in top left panel
    ax = plot_vibrations_as_orders(df_demonstrate, FREQ_VALUES, row_id=row_id1, ax=ax_tr)
    plot_orders(ax)
    ax.set_ylim(0, 0.5)
    ax.set_title('Transformed frequency measurement A')

    # plot transformed frequency measurement 2 in bottom left panel
    ax = plot_vibrations_as_orders(df_demonstrate, FREQ_VALUES, row_id=row_id2, ax=ax_br)
    plot_orders(ax)
    ax.set_ylim(0, 0.5)
    ax.set_title('Transformed frequency measurement B')

    fig.tight_layout()

    return fig


def plot_reconstruction_with_original(model_series, indices, xlabels=None):
    """ Visualize the reconstruction of a vibration signal together with the original signal.

    Args:
        model_series: pandas series with V_normalized, W and nmf.
        indices: List of indices to plot.

    Returns:

    """
    nrows = len(indices)
    fig, axes = plt.subplots(figsize=(10, 5*nrows), nrows=nrows)
    V = model_series.V_normalized
    V_reconstructed = model_series.V_reconstructed
    R2 = model_series.R2
    MSE = model_series.MSE
    for ax, index in zip(axes, indices):
        ax.plot(V[index], marker='o', label='original')
        ax.plot(V_reconstructed[index], marker='o', label='reconstructed')
        if xlabels is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_xticklabels(xlabels, rotation=90)
        ax.legend()
        ax.set_title(f'R2 = {round(R2[index], 3)}, MSE = {round(MSE[index], 3)}')
    return fig, axes


def normalize_per_sensor(metadata, V, Scaler=MinMaxScaler, **kwargs):
    """ Standardize a dataframe per sensor.

    Args:
        metadata:
        V:
        Scaler:
        **kwargs: Key-word arguments for deployed scaler.

    Returns:

    """
    new_index = []
    V_normalized = np.empty((0, V.shape[1]))
    for (direction, location, pump), group in metadata.groupby(['direction', 'location', 'pump']):
        idcs = group.index
        new_index = new_index + idcs.tolist()
        V_group = V.loc[idcs]
        V_normalized = np.vstack((V_normalized, Scaler(**kwargs).fit_transform(V_group)))  # scale each feature vector (per sensor)
    df_V_normalized = pd.DataFrame(V_normalized, index=pd.MultiIndex.from_tuples(new_index), columns=V.columns)
    return df_V_normalized


def extract_sigma_threshold_per_band(df, factor=3):
    """ Extract a variation based threshold.
            threshold = mu + factor * sigma,
        where mu is mean of values and sigma is standard deviation per sensor.

    Args:
        df: Dataframe.
        factor: Scalar. Determines the factor of standard deviations of the threshold.

    Returns:
        df_threshold: Dataframe of same shape as df where each cell coresponds to the 3-sigma threshold.
        std_threshold: Dataframe with threshold per sensor.
    """
    # set index per sensor
    idx_sensor = ['pump', 'location', 'direction']

    # set threshold per sensor
    df_grouped = df.groupby(idx_sensor)
    mu = df_grouped.mean()
    sigma = df_grouped.std()
    std_threshold = (mu + factor * sigma)

    # create dataframe of same shape and order as df
    df_threshold = pd.merge(df, std_threshold, left_on=idx_sensor, right_index=True, how='left')
    # only include columns from right df and rename to original columns name
    threshold_columns = df_threshold.columns[df_threshold.columns.str.contains('y')]
    df_threshold = df_threshold[threshold_columns]
    df_threshold.columns = df_threshold.columns.str[:-2]
    df_threshold = df_threshold.reindex(df.index)

    return df_threshold, std_threshold


def enforce_sigma_threhsold(df, df_threshold):
    """  For each value cap at maximum value below 3-sigma threshold.

    Args:
        df: Dataframe on which the sigma threshold should be enforced.
        df_threshold: Dataframe with sigma threshold. Must be of same shape as df.

    Returns:
        Same as df, but each value above the specified threshold is replaced with the next lowest value.
    """
    df = df[df < df_threshold]
    df_with_replaced_values = []
    for idx, group in df.groupby(['pump', 'location', 'direction']):
        df_with_replaced_values.append(group.fillna(group.max().to_dict()))
    df_with_replaced_values = pd.concat(df_with_replaced_values)
    df_with_replaced_values = df_with_replaced_values.reindex(df.index)
    return df_with_replaced_values