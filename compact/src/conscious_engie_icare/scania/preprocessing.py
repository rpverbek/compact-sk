# ©, 2024, Sirris
# owner: FFNG

from conscious_engie_icare.scania.dataset import ATTRIBUTE_COLUMNS
from tqdm import tqdm
import pandas as pd


def differentiate_group(group):
    """ Differentiate time series and copy first value. """
    group_differentiated = group.diff()
    group_differentiated.iloc[0] = group.iloc[0]
    return group_differentiated


def differentiate(df_, attribute_columns=ATTRIBUTE_COLUMNS):
    tmp = df_.groupby('vehicle_id')[attribute_columns].progress_apply(differentiate_group)
    df_[ATTRIBUTE_COLUMNS] = tmp.reset_index(level=0, drop=True)
    return df_


def limit_ts(df_, limit=25):
    tmp = df_.groupby('vehicle_id').progress_apply(lambda group: group.iloc[:limit])
    df_ts_limited = tmp.reset_index(level=0, drop=True)
    return df_ts_limited


def forward_fill_ts(df_):
    """ Forward fill missing values for each vehicle_id based on time_step. """
    df_ts_filled = df_.sort_values(by=['vehicle_id', 'time_step'])
    tqdm.pandas()
    tmp = df_ts_filled.groupby('vehicle_id')[ATTRIBUTE_COLUMNS].progress_apply(lambda group: group.ffill())
    tmp = tmp.fillna(0)
    df_ts_filled[ATTRIBUTE_COLUMNS] = tmp.reset_index(level=0, drop=True)
    print(f'Shape of df_ts_filled: {df_ts_filled.shape}')
    return df_ts_filled


def offset_ts_v1(df_):
    first_time_step = get_first_time_step_per_vehicle_id(df_)
    dict_time_step_offset = first_time_step.set_index('vehicle_id')['time_step'].to_dict()
    df_['time_step_offset'] = df_['vehicle_id'].progress_apply(lambda x: dict_time_step_offset.get(x))
    df_['normalized_time_step'] = df_['time_step'] - df_['time_step_offset']
    df_ts_with_offset_v1 = df_
    return df_ts_with_offset_v1


def offset_ts_v2(df_):
    df_ts_with_offset_v2 = df_.sort_values(['vehicle_id', 'time_step'])
    initial_values = df_ts_with_offset_v2[df_ts_with_offset_v2['time_step'] == 0].set_index('vehicle_id')[ATTRIBUTE_COLUMNS]
    df_merged = pd.merge(df_ts_with_offset_v2, initial_values, left_on='vehicle_id', right_index=True, how='left', suffixes=("", "_offset"))
    offset_columns = df_merged.columns[df_.columns.str.contains(r"(_offset)")]
    df_merged[offset_columns] = df_merged[offset_columns].fillna(0)
    for col in ATTRIBUTE_COLUMNS:
        df_merged[col] = df_merged[col] - df_merged[col + '_offset']
    df_ts_with_offset_v2 = df_merged.copy().drop(columns=offset_columns)
    return df_ts_with_offset_v2


def offset_ts_v3(df_):
    pass


def interpolate_ts(df_, attribute_columns=ATTRIBUTE_COLUMNS):
    df_ = df_.sort_values(['vehicle_id', 'time_step'])
    tmp_grouped = df_.groupby('vehicle_id')[attribute_columns]
    tmp_grouped = tmp_grouped.apply(lambda group: group.interpolate(method='linear', limit=None))
    df_[attribute_columns] = tmp_grouped.reset_index(level=0, drop=True)
    df_ts_interpolated = df_
    return df_ts_interpolated


def get_first_time_step_per_vehicle_id(df_):
    df_ = df_.reset_index()
    df_ = df_.sort_values(by=['vehicle_id', 'time_step'])
    df_ = df_.groupby('vehicle_id').first()[['index', 'time_step']].reset_index()
    return df_


def normalize_by_total_count(df_, list_of_sensor_bins):
    """ Normalize by the total count. 

    df_: Pandas dataframe. The cumulative (imputed and preprocessed) timeseries.
    sensor_bins: list of lists. Divide each timestamp by all sensors from the given list.
    """
    # df_['sum'] = df_[sensor_bins].sum(axis=1)
    sums = []
    for sensor_bins in list_of_sensor_bins:
        sum = df_[sensor_bins].sum(axis=1)
        for sb in sensor_bins:
            df_[sb] = df_[sb] / sum
        sums.append(sum)
    return df_, sums
