# ©, 2023, Sirris
# owner: FFNG
""" Code related to contextual concept drift detection. """
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.metrics import pairwise


def extract_vibration_measurement_periods(df_):
    """ Given a dataframe df_ with timestamp, location and direction, extract measurement periods.

    A consistent measurement period is a period where within a 45-minute window all 6 sensor measurements were taken:
        (inboard/outboard) x (vertical/horizontal/axial)
    """
    vibration_measurement_periods = []
    for pump, group in tqdm(df_.groupby('pump'), total=len(df_.pump.unique()), desc='iterating over pumps'):
        for idx, row in group.iterrows():
            start = row.timestamp - pd.Timedelta(minutes=45)
            stop = row.timestamp + pd.Timedelta(minutes=1)
            measurement_group = group[(group.timestamp > start) & (group.timestamp < stop)]
            n_unique_sensors = len((measurement_group['location'] + measurement_group['direction']).unique())
            n_sensors = len(measurement_group)
            if (n_sensors == 6) & (n_unique_sensors == 6):
                vibration_measurement_periods.append({
                    'start': start,
                    'stop': stop,
                    'group': measurement_group
                })
    return vibration_measurement_periods


def extract_vibration_weights_per_measurement_period(measurement_periods, col_names, band_cols, normalization, model):
    Ws = []
    for period in tqdm(measurement_periods, total=len(measurement_periods)):
        band_column_names = period['group'].columns[period['group'].columns.str.contains('band_')]
        # dim(V) = 6 x 120
        period['V'] = period['group'].set_index(['location', 'direction'])[band_column_names]
        period['V_normalized'] = normalization(period['V'], band_cols)
        # dim(W) = 6 x 16
        period['W'] = model.nmf.transform(period['V_normalized'])
        period['W'] = pd.DataFrame(period['W'], columns=col_names)  # !!!
        period['W'].index = period['V'].index
        period['pump'] = period['group'].pump.iloc[0]
        Ws.append(period)
    return pd.DataFrame(Ws)


def extract_vibration_weights_per_sensor(df_bands_, band_cols, normalization, model):
    df_bands_normalized_ = normalization(df_bands_.copy(), band_cols)   # in practice this is done in the pipeline
    df_ = df_bands_.copy()[['timestamp', 'pump', 'location', 'direction']]
    tqdm.pandas()
    transform = lambda x: model.nmf.transform(x.to_numpy().reshape(1, -1))[0]
    df_['W'] = df_bands_normalized_.progress_apply(transform, axis=1)
    return df_


def calculate_distance_to_cluster_centers(pipe, x, distance_function=distance.euclidean):
    """

    Args:
        pipe:
        x:
        distance_function:

    Returns:

    """
    xt = pipe['pca'].transform(pipe['scaler'].transform(x))
    cluster_centers = pipe['kmeans'].cluster_centers_
    dist = {i: distance_function(xt, cc) for i, cc in enumerate(cluster_centers)}
    return dist


def visualize_fingerprint_for_window(df, weight_columns=None, operating_modes=None, **plot_kwargs):
    """ Create heatmap with fingerprints for specified window.

    Given a dataframe with
        (1) the weights extracted for a specific window
        (2) the asset location
        (3) the operating mode
    visualize the fingerprint as heatmap.

    Args:
        df: pandas dataframe

    Returns:
        figure
    """
    index_order = [
        ('inboard', 'axial'),
        ('inboard', 'horizontal'),
        ('inboard', 'vertical'),
        ('outboard', 'axial'),
        ('outboard', 'horizontal'),
        ('outboard', 'vertical')
    ]
    weight_columns = list(range(15)) if weight_columns is None else weight_columns
    operating_modes = list(range(5)) if operating_modes is None else operating_modes
    fingerprints = df.groupby(['operating_mode', 'location', 'direction']).mean()
    fig, axes = plt.subplots(figsize=(15, 20), nrows=len(operating_modes))
    for om, ax in zip(operating_modes, axes):
        fp = fingerprints.reset_index()
        fp_ = fp[(fp.operating_mode == om)]
        fp_ = fp_.set_index(['location', 'direction'])
        fp_ = fp_.reindex(index_order)
        ax.set_title(om)
        try:
            sns.heatmap(data=fp_[weight_columns], ax=ax, **plot_kwargs)
        except:
            print(f'om {om} not found!')
    fig.tight_layout()
    return fig, axes


def create_pivot_table(df_):
    df_pivot_ = df_.copy()
    df_pivot_ = df_pivot_.pivot(index=['idx', 'timestamp', 'pump', 'location', 'direction'], columns='om', values='cosine_distance')
    df_pivot_['closest OM'] = df_pivot_.apply(lambda x: x.idxmin(), axis=1)
    return df_pivot_


def calculate_distances_per_sensor_to_vibration_fingerprints(df_W_, component_columns, fingerprints):
    """ Calculate distances of vibration weights with all fingerprints. """
    df_dist_ = []
    for idx, row in tqdm(df_W_.iterrows(), total=len(df_W_)):
        # weights = row['W'].reshape(1, -1)
        weights = row[component_columns].to_numpy().reshape(1, -1)
        for om in fingerprints:
            fingerprint = fingerprints[om]
            fingerprint_vector = fingerprint.loc[(row.location, row.direction), :].to_numpy().reshape(1, -1)
            tmp = {
                'idx': idx, 'timestamp': row.timestamp, 'pump': row.pump, 'location': row.location, 'direction': row.direction, 
                'data': row, 'om': om,
                'cosine_distance': pairwise.cosine_distances(weights, fingerprint_vector)[0][0]
            }
            df_dist_.append(tmp)
    df_dist_ = pd.DataFrame(df_dist_)
    return df_dist_


def extract_vibration_fingerprints(df_W_train, df_contextual_train_with_labels, component_columns):
    df_W_train_with_OM = pd.merge_asof(
    df_W_train, df_contextual_train_with_labels.sort_values(by='timestamp'),
        by='pump', on='timestamp', direction='backward'
    )
    df_ = df_W_train_with_OM[component_columns].copy()
    df_[['operating_mode', 'location', 'direction']] = df_W_train_with_OM[['cluster_kmeans', 'location', 'direction']]
    fingerprints = {
        om: om_group.drop(columns=['operating_mode']).groupby(['location', 'direction']).mean()
        for om, om_group in df_.groupby('operating_mode')
    }
    return fingerprints
