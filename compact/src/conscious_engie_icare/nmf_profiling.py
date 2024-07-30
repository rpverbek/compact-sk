"""
Code related to profiling batches via non-negative matrix factorization (NMF).
"""
import plotly.express as px
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
import numpy as np
from conscious_engie_icare import distance_metrics


def extract_nmf_incremental(df_V, max_n_components, timestamps=None, verbose=True):

    V = df_V.to_numpy()
    range_components = range(1, max_n_components+1)[::-1]
    tqdm_description = 'Fitting NMF with varying number of components'
    tqdm_range_components = tqdm(range_components, desc=tqdm_description, disable=(not verbose))

    list_models = []
    previous_W = None
    previous_H = None
    for _n_components in tqdm_range_components:
        if previous_W is None or previous_H is None:
            # For the first (largest) decomposition, use the default 'nndsvd' initialization
            nmf = NMF(n_components=_n_components, init='nndsvd', max_iter=1000, random_state=42)
            W = nmf.fit_transform(V)
            H = nmf.components_
        else:
            # For subsequent decompositions, use the results of the previous decomposition for warm start
            nmf = NMF(n_components=_n_components, init='custom', max_iter=1000, random_state=42)
            W = previous_W[:, :_n_components].copy(order='C')  # Use only the first n_components columns
            H = previous_H[:_n_components, :].copy(order='C')  # Use only the first n_components rows

            W = nmf.fit_transform(V, W=W, H=H)
            H = nmf.components_
        V_reconstructed = nmf.inverse_transform(W)

        model_dict = {
            # number of components
            'n_components': _n_components,
            # trained NMF-model
            'nmf': Pipeline([('nmf', nmf)]),
            # decomposition matrix V, and approximation matrices W and H
            'V': V, 'W': W, 'H': pd.DataFrame(H),
            # timestamps for V
            'V_timestamps': timestamps,
            # reconstructed decomposition matrix
            'V_reconstructed': V_reconstructed,
            # coefficient of determination (average)
            'R2_mean': r2_score(V, V_reconstructed, multioutput='uniform_average'),
            # coefficient of determination (per sample)
            'R2': r2_score(V.T, V_reconstructed.T, multioutput='raw_values'),
            # coefficient of determination (per feature)
            'R2_feature': r2_score(V, V_reconstructed, multioutput='raw_values'),
            # mean squared error (average)
            'MSE_mean': mean_squared_error(V, V_reconstructed),
            # mean squared error (per sample)
            'MSE': mean_squared_error(V.T, V_reconstructed.T, multioutput='raw_values'),
            # reconstruction error expressed as Frobenius norm
            'reconstruction_error': nmf.reconstruction_err_
        }

        list_models.append(model_dict)

        # Update previous_W and previous_H for the next iteration
        previous_W = W
        previous_H = H
    list_models.reverse()
    df_models = pd.DataFrame(list_models)
    return df_models


def extract_nmf_per_number_of_component(df_V, n_components=60, timestamps=None, verbose=True):
    """ Perform nmf with varying number of components.

    :param df_V: Dataframe with (normalized) decomposition matrix V.
    :param n_components: Integer with maximum number of components that should be extracted for NMF.
    :param timestamps: Pandas series with timestamps corresponding to rows in feature space.
    :param verbose:
    :return: Dataframe where each row corresponds to a different number of components and the columns contain
        W and H, as well as some other model specific parameters.
    """
    V = df_V.to_numpy()
    range_components = range(1, n_components)
    tqdm_description = 'Fitting NMF with varying number of components'
    tqdm_range_components = tqdm(range_components, desc=tqdm_description, disable=(not verbose))
    list_models = [extract_NMF(V, n_components=_n_components, timestamps=timestamps.to_numpy()) for _n_components in tqdm_range_components]
    df_models = pd.DataFrame(list_models)
    return df_models


def extract_NMF(V, n_components=60, timestamps=None):
    """  Extracts statistics for non-negative matrix factorisation (NMF) on feature space provided by df.

    Given a non-negative decomposition matrix V, NMF approximates V with two matrices W and H s.t. V = W x H.

    :param V: Numpy array with (normalized) decomposition matrix V.
    :param n_components: Integer with maximum number of components that should be extracted for NMF.
    :param timestamps:
    :return: Dictionary containing W and H, as well as some other model specific parameters.
    """
    nmf = NMF(n_components=n_components, init='nndsvd', max_iter=1000, random_state=42)  # TODO: compare solvers
    model = Pipeline([('nmf', nmf)])
    W = model.fit_transform(V)
    H = pd.DataFrame(model['nmf'].components_)
    V_reconstructed = model['nmf'].inverse_transform(W)
    model_dict = {
        # number of components
        'n_components': n_components,
        # trained NMF-model
        'nmf': model,
        # decomposition matrix V, and approximation matrices W and H
        'V': V, 'W': W, 'H': H,
        # timestamps for V
        'V_timestamps': timestamps,
        # reconstructed decomposition matrix
        'V_reconstructed': V_reconstructed,
        # coefficient of determination (average)
        'R2_mean': r2_score(V, V_reconstructed, multioutput='uniform_average'),
        # coefficient of determination (per sample)
        'R2': r2_score(V.T, V_reconstructed.T, multioutput='raw_values'),
        # coefficient of determination (per feature)
        'R2_feature': r2_score(V, V_reconstructed, multioutput='raw_values'),
        # mean squared error (average)
        'MSE_mean': mean_squared_error(V, V_reconstructed),
        # mean squared error (per sample)
        'MSE': mean_squared_error(V.T, V_reconstructed.T, multioutput='raw_values'),
        # reconstruction error expressed as Frobenius norm
        'reconstruction_error': model['nmf'].reconstruction_err_
    }
    return model_dict


def plot_W_as_heatmap(W, x_labels=None):
    """ Given a component matrix W, plot its weights as heatmap.

    Args:
        W:
        x_labels:

    Returns:

    """
    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z')+1)))
    fig = px.imshow(W.T, title='W',  # text_auto=True,
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="parameters", y="components", color="Value"),
                    x=x_labels, y=alphabet_upper_case[:W.shape[1]], aspect='auto', height=600, width=1000)
    return fig


def plot_H_as_heatmap(H, x_labels=None, color_continuous_scale='RdBu_r', title='H'):
    """

    Args:
        H:
        x_labels:
        color_continuous_scale:
        title:

    Returns:

    """
    alphabet_upper_case = list(map(chr, range(ord('A'), ord('Z')+1)))
    y_labels = alphabet_upper_case[:H.shape[0]]
    fig = px.imshow(H, title=title,  # text_auto=True,
                    color_continuous_scale=color_continuous_scale,
                    labels=dict(x="parameters", y="components", color="Value"),
                    x=x_labels, y=y_labels, aspect='auto', height=600, width=1000)
    return fig


def extract_frequency_bands_for_single_sample(vibrations, velocity_in_rpm, lower_bounds=None, upper_bounds=None,
                                              window_boundaries=(9, 11), order_to_scale_peak_to=10,
                                              correct_orders=True, agg_func=np.mean):
    # set default arguments
    lower_bounds = np.arange(0.25, 30.25, 0.5) if lower_bounds is None else lower_bounds
    upper_bounds = np.arange(0.75, 30.75, 0.5) if upper_bounds is None else upper_bounds
    assert len(lower_bounds) == len(upper_bounds)

    # extract orders with velocity
    velocity_in_Hz = velocity_in_rpm / 60
    orders = vibrations.index.astype(float) / velocity_in_Hz

    # correct orders based on set peak
    if correct_orders:
        selected_frequency_band = vibrations[(orders >= window_boundaries[0]) & (orders <= window_boundaries[1])]
        if len(selected_frequency_band) > 0:
            argmax = np.argmax(selected_frequency_band)
            order_of_uncorrected_peak = orders[(orders >= window_boundaries[0]) & (orders <= window_boundaries[1])][argmax]
            c = order_to_scale_peak_to / order_of_uncorrected_peak
        else:
            print('Could not find any values in window boundaries. Going to set c to 1.')
            c = 1
        orders = c * orders

    # aggregate over frequency bands
    aggregate_over_fband = lambda lb, ub: agg_func(vibrations[(orders >= lb) & (orders <= ub)])
    frequency_bands = {f'band_{lb}-{ub}': aggregate_over_fband(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)}

    # calculate number of bins per aggregated order band
    n_per_fband = lambda lb, ub: len(vibrations[(orders >= lb) & (orders <= ub)])
    n_per_frequency_bands = {f'n_{lb}-{ub}': n_per_fband(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)}

    return frequency_bands, n_per_frequency_bands


def automated_frequency_band_extraction(df, verbose=True, freq_values=None, rpm_name='rotational speed [RPM]', min_rpm=10, expected_freq_columns=6397, **kwargs):
    """ Given a dataframe with frequency values measured in Hz and a rotational speed, convert those to orders.

    :param df: Pandas dataframe.
    :param verbose: If True, show progress bar. Default: True.
    :param freq_values: Names of columns in df that contain vibration measures. Default: Regex pattern '\d*\.\d*'.
    :param rpm_name: Name of colum with rotational speed. Default: 'rotational speed [RPM]'.
    :param min_rpm: Vibration measurements below the minimum rotational speed [RPM] are not converted. Default: 10.
    :param kwargs: Keyword arguments to be sent to `extract_frequency_bands_for_single_sample()` (see above).
    :return: Pandas dataframe with converted frequency bands.
    """
    freq_values = df.columns[df.columns.str.contains('^\d*\.\d*',na=True)] if freq_values is None else freq_values
    m  = f'We expect {expected_freq_columns} frequency values, got {len(freq_values)} instead; did the dataset change?'
    assert len(freq_values) == expected_freq_columns, m
    frequency_bands = []
    for idx, row in tqdm(df.iterrows(), total=len(df), disable=(not verbose)):
        velocity_in_rpm = row[rpm_name]
        if velocity_in_rpm > min_rpm:
            vibrations = row[freq_values]
            fbands, nfbands = extract_frequency_bands_for_single_sample(vibrations, velocity_in_rpm, **kwargs)
            frequency_bands.append({'index': idx, **fbands, **nfbands})
    frequency_bands = pd.DataFrame(frequency_bands).set_index('index')
    return frequency_bands


def derive_df_vib(data, f):
    psds = []
    rpms = []
    torques = []
    directions = []
    sample_ids = []
    unique_sample_ids = []
    for x in data:
        rpm = x['rpm']
        torque = x['torque']
        sample_id = x['sample_id']
        for key in ['psd_x', 'psd_y', 'psd_z']:
            psds.append(x[key])
            directions.append(key)
            rpms.append(rpm)
            torques.append(torque)
            sample_ids.append(sample_id)
            unique_sample_ids.append(x['unique_sample_id'])

    df_vib = pd.DataFrame(psds, columns=f)  # !!!
    df_vib['rotational speed [RPM]'] = rpms
    df_vib['torque [Nm]'] = torques
    df_vib['direction'] = directions
    df_vib['sample_id'] = sample_ids
    df_vib['unique_sample_id'] = unique_sample_ids
    return df_vib


def derive_df_orders(df_vib, setup, f, verbose=True):
    lower_bounds = np.arange(setup['start'], setup['stop'], setup['window_steps'])
    upper_bounds = np.arange((setup['start'] + setup['window_size']), (setup['stop'] + setup['window_size']), setup['window_steps'])
    n_windows = setup['n_windows']
    n_lower_bounds = len(lower_bounds)
    n_upper_bounds = len(upper_bounds)
    m = f'Number of windows ({n_windows}), lower bounds ({n_lower_bounds}) and upper bounds ({n_upper_bounds}) do not match.'
    assert n_windows == n_lower_bounds == n_upper_bounds, m
    df_orders =  automated_frequency_band_extraction(
        df_vib, verbose=verbose,
        rpm_name='rotational speed [RPM]',
        min_rpm=10, correct_orders=False,
        window_boundaries=(19, 23), order_to_scale_peak_to=21,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds,
        agg_func=np.sum, expected_freq_columns=f.shape[0]
    )
    meta_data = df_vib[['rotational speed [RPM]', 'torque [Nm]', 'direction', 'sample_id', 'unique_sample_id']]
    return df_orders, meta_data


def extract_df_dist_pivot(df_dist_, metric='cosine_distance', extract_closest_fingerprint=True):
    """ Extract a pivot dataframe with distance vectors to vibration fingerprints from a dataframe in long format.
    
    Given a dataframe in long format where each row corresponds to the distance to a specific vibration fingerprint, 
    extract a dataframe in pivot format where each row corresponds to a specific pump and the columns contain the different distances.
    """
    df_ = df_dist_[['idx', 'pump', 'om', 'start', 'stop', metric]]
    df_dist_pivot_ = df_.pivot(index=['idx', 'pump', 'start', 'stop'], columns='om', values=metric)
    df_dist_pivot_.columns.name = ''
    # for each row get arg min of distance to cluster centers
    if extract_closest_fingerprint:
        df_dist_pivot_['closest_fingerprint'] = df_dist_pivot_.apply(lambda x: x.idxmin(), axis=1)
    #df_dist_pivot_ = df_dist_pivot_.reset_index(level=[1, 2, 3])
    #df_dist_pivot_ = df_dist_pivot_.loc[df_stable_.index]
    #df_dist_pivot_['process_om'] = df_stable_[labels_].apply(lambda x: x.idxmax(), axis=1)  # (!!!) BUG PROBABLY INDUCED HERE by indexing --> not necessarily
    return df_dist_pivot_


def add_start_to_df_dist(df_dist_):
    start = []
    stop = []
    for _, row in df_dist_.iterrows():
        start.append(row.data.start)
        stop.append(row.data.stop)
    df_dist_['start'] = start
    df_dist_['stop'] = stop
    return df_dist_


def calculate_om_ratios(vibration_measurement_periods, df_endo, om_labels):
    vibration_measurement_periods_with_ratio = []
    for idx, row in tqdm(vibration_measurement_periods.iterrows(), total=len(vibration_measurement_periods)):
        start = pd.to_datetime(row.start).tz_localize('utc')
        stop = pd.to_datetime(row.stop).tz_localize('utc')
        pump = row.pump
        # create series where each index is a letter and each value is 0
        series = pd.Series(0, index=om_labels)
        series.name = 'empty series'
        operating_modes = df_endo[
            (df_endo.timestamp > start) &
            (df_endo.timestamp < stop) & 
            (df_endo.pump == pump)].prediction.value_counts()
        operating_modes.name = 'count'
        count = pd.merge(series, operating_modes, left_index=True, right_index=True, how='outer').fillna(0)['count']
        ratio = count / count.sum()
        vibration_measurement_periods_with_ratio.append(dict(row.to_dict(), **ratio.to_dict()))
    vibration_measurement_periods_with_ratio = pd.DataFrame(vibration_measurement_periods_with_ratio)

    return vibration_measurement_periods_with_ratio


def calculate_distances_per_measurement_period(measurement_period, fingerprints, verbose=False):
    # pointwise Mahalanobis distance
    fingerprint_matrix = np.array([fingerprints[om].to_numpy().flatten() for om in fingerprints])
    # calculate covariance matrix
    fingerprint_S = np.cov(fingerprint_matrix.T)
    # calculate inverse
    fingerprint_SI = np.linalg.inv(fingerprint_S)
    # calculate mu
    fingerprint_mu = fingerprint_matrix.mean(axis=0)
    df_dist_ = []
    for idx, row in measurement_period.iterrows():
        for om in fingerprints:
            weights = row['W']
            fingerprint = fingerprints[om]
            tmp = {
                'idx': idx,
                'data': row,
                'om': om,
                # 'frobenius_norm': distance_metrics.frobenius_norm(weights, fingerprint),
                # 'frobenius_norm_pow2': distance_metrics.frobenius_norm_v2(weights, fingerprint),
                # 'frobenius_norm_sqrt': distance_metrics.frobenius_norm_v3(weights, fingerprint),
                'cosine_distance': distance_metrics.cosine_distance(weights, fingerprint),
                'manhattan_distance': distance_metrics.manhattan_distance(weights, fingerprint),
            }
            df_dist_.append(tmp)
    df_dist_ = pd.DataFrame(df_dist_)
    return df_dist_
