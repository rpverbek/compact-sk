# ©, 2022, Sirris
# owner: FFNG

import csv
import os
import string
import glob

from conscious_engie_icare.config import data_dir
from conscious_engie_icare.data.phm_data_handler import FILE_NAMES_HEALTHY, BASE_PATH_HEALTHY

import pandas as pd
import numpy as np
from sklearn.metrics import auc


def get_closest_val(ts, s):
    """Given a timestamp 'ts' and a series 's' that is indexed by timestamps,
    find the closest entry to 'ts'."""
    closest_val = s.iloc[np.argmin(abs(s.index - ts))]
    return closest_val


def check_data_integrity(meta_data, dtype='fftv',
                         expected_value_separation_index=6399):
    """Check assumptions about raw data: Same frequency column and value separator."""
    assert dtype in ['fftv', 'fftg', 'g']
    local_path = os.path.join(data_dir, meta_data[dtype]["local_path"])
    with open(local_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(iter(reader)):
            if i > 0:
                # check if frequency value always the same per column
                assert last_row[2:expected_value_separation_index] \
                       == row[2:expected_value_separation_index]
            # check if "values:" column always at the same column index
            assert row.index("values:") == expected_value_separation_index
            last_row = row


def get_entries_count_per_day(df):
    """Return entries per day and location."""
    count = df.groupby('location').resample('D').count().iloc[:, 2]
    return count


def get_location_entries(df):
    """Entries per location aggregated over day."""
    # TODO: remove
    # rename locations in alphabetical order,
    # where A is most often occurring location, B second most often occurring...
    locations_value_counts = df['location'].value_counts()
    locations_value_counts = pd.DataFrame(
        locations_value_counts.rename('count'))
    locations_value_counts['alias'] = list(
        string.ascii_uppercase[:len(locations_value_counts)])
    alias_dict = locations_value_counts['alias'].to_dict()
    locations = df.replace({'location': alias_dict})
    return locations


def get_summary(df, agg_func):
    agg_values = df[df.columns[df.columns.str.contains(r"\d+.\d+")]]
    agg_values = agg_values.apply(agg_func)
    return agg_values


def get_min(df):
    return get_summary(df, lambda x: x.min())


def get_max(df):
    return get_summary(df, lambda x: x.max())


def split_in_groups(array):
    """Group consecutive elements if they have the same value."""
    return np.split(array, np.where(np.diff(array) != 0)[0] + 1)


def calculate_roc_characteristics(df_):
    df_ = df_.sort_values(by='distance_to_own_cluster_center', ascending=True)

    # Initialize variables to store ROC curve values
    fpr = []
    tpr = []

    for threshold in df_['distance_to_own_cluster_center']:
        df_['predicted_anomaly'] = df_['distance_to_own_cluster_center'] >= threshold

        # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
        true_positives = df_[(df_['pitting'] == 1) & (df_['predicted_anomaly'] == 1)].shape[0]
        false_positives = df_[(df_['pitting'] == 0) & (df_['predicted_anomaly'] == 1)].shape[0]
        true_negatives = df_[(df_['pitting'] == 0) & (df_['predicted_anomaly'] == 0)].shape[0]
        false_negatives = df_[(df_['pitting'] == 1) & (df_['predicted_anomaly'] == 0)].shape[0]

        tpr.append(true_positives / (true_positives + false_negatives))
        fpr.append(false_positives / (false_positives + true_negatives))

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def calc_tpr_at_fpr_threshold(tpr, fpr, threshold=0.1):
    # sort tpr and fpr such that they are in ascending order
    if (fpr[0] > fpr[-1]) or (tpr[0] > tpr[-1]):
        assert tpr[0] > tpr[-1] and fpr[0] > fpr[-1]
        tpr = list(reversed(tpr))
        fpr = list(reversed(fpr))
    try:
        idx = next(i for i, value in enumerate(fpr) if value > threshold)
    except StopIteration:
        idx = 0
    tpr_at_fpr = tpr[idx]
    return tpr_at_fpr


def calc_fpr_at_tpr_threshold(tpr, fpr, threshold=0.1):
    return calc_tpr_at_fpr_threshold(tpr=fpr, fpr=tpr, threshold=threshold)


def get_operating_modes():
    rpms = [int(_f.split('/')[-1].split('_')[0].strip('V')) for _f in FILE_NAMES_HEALTHY]
    torques = [int(_f.split('/')[-1].split('_')[1].strip('N')) for _f in FILE_NAMES_HEALTHY]
    df_operating_modes = pd.DataFrame(columns=sorted(np.unique(rpms)), index=sorted(np.unique(torques)), data='')
    counter = 1
    for torque in sorted(np.unique(torques)):
        for rpm in sorted(np.unique(rpms)):
            runs = glob.glob(os.path.join(BASE_PATH_HEALTHY, f'V{rpm}_{torque}N_*.txt'))
            runs = [int(run.split('/')[-1].split('_')[2].strip('.txt')) for run in runs]
            if len(runs) > 0:
                df_operating_modes.loc[torque, rpm] = f'OM {counter}'
                counter += 1
    return df_operating_modes
