# ©, 2024, Sirris
# owner: FFNG

import pandas as pd
import os
from compact import LOCAL_PATH_NMF_MODELS
from compact.normalization import normalize_1
from compact.feedwater import util

FOLDER_MODELS = os.path.join(LOCAL_PATH_NMF_MODELS, 'nmf-issue-28-02-23')

CONTEXTUAL_COLUMNS = [
    'velocity <v-MP> [RPM]', 'p_delta_pressure_inlet <deltap-filter-MP> [barg]',
    'p_in [barg]', 'p_out <pout-MP> [barg]', 'p_delta [barg]', 'p_feedwater_tank <p-FWT> [barg]',
    't_inlet <T-in-MP> [Celsius]', 't_fwt <T-fwt> [Celsius]'
]


LOCAL_PATH_DATA = os.path.join('..', '..', 'work', 'uc_feedwater_pumps')  # TODO
LOCAL_PATH_OPERATIONAL = os.path.join(LOCAL_PATH_DATA, 'operational_data')
FOLDER_MODELS = os.path.join(LOCAL_PATH_NMF_MODELS, 'nmf-issue-28-02-23')
PUMPS = [1, 2, 3]
SETUP = {
    'idx_setup': 1, 'normalization': normalize_1,  'window_size': 2, 
    'window_steps': 0.25, 'start': 0.25, 'stop': 30.25, 'n_windows': 120, 
    'peak_correction': False, 'per_sensor': False, 'path_name': 'setup-1.csv', 
    'model_name': 'model-1a.pickel'
}


def get_and_preprocess_data(pumps=PUMPS, large_operational_data=True):
    # Load contextual data
    df_contextual = pd.concat([load_and_preprocess_operational_features(
        pump, large_operational_data=large_operational_data
    ) for pump in pumps])
    meta_columns = ['timestamp', 'pump']
    endogenous_columns = [
        'velocity <v-MP> [RPM]',
        'p_delta_pressure_inlet <deltap-filter-MP> [barg]', 'p_in [barg]',
        'p_out <pout-MP> [barg]', 'p_delta [barg]', 'p_feedwater_tank <p-FWT> [barg]', 
        't_inlet <T-in-MP> [Celsius]', 't_fwt <T-fwt> [Celsius]'
    ]
    include_columns = meta_columns + endogenous_columns
    df_contextual = df_contextual[include_columns]
    
    # Load vibration data
    FNAME = os.path.join(FOLDER_MODELS, SETUP['path_name'])
    df_V = pd.read_csv(FNAME, parse_dates=['timestamp'], index_col=0)
    df_V['pump'] = df_V['pump'].replace({11: 1, 12: 2, 13: 3})

    return df_V, df_contextual


def load_and_preprocess_operational_features(pump, base_path=LOCAL_PATH_OPERATIONAL, large_operational_data=True):
    if large_operational_data:
        file_name = f'operational_features_pump{pump}.csv'
    else: 
        file_name = f'operational_features_pump{pump}-5min-granularity.csv'
    file_path = os.path.join(base_path, file_name)
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pump'] = pump
    return df


def check_criteria_custom_dataset(path_to_df_V, path_to_df_contextual):
    df_V_custom = pd.read_csv(path_to_df_V, index_col=0)
    df_contextual_custom = pd.read_csv(path_to_df_contextual, index_col=0)

    df_V_columns_expected = ['timestamp', 'pump', 'location', 'direction', 'rotational speed [RPM]', 'band_0.25-2.25',
                             'band_0.5-2.5', 'band_0.75-2.75', 'band_1.0-3.0', 'band_1.25-3.25', 'band_1.5-3.5',
                             'band_1.75-3.75', 'band_2.0-4.0', 'band_2.25-4.25', 'band_2.5-4.5', 'band_2.75-4.75',
                             'band_3.0-5.0', 'band_3.25-5.25', 'band_3.5-5.5', 'band_3.75-5.75', 'band_4.0-6.0',
                             'band_4.25-6.25', 'band_4.5-6.5', 'band_4.75-6.75', 'band_5.0-7.0', 'band_5.25-7.25',
                             'band_5.5-7.5', 'band_5.75-7.75', 'band_6.0-8.0', 'band_6.25-8.25', 'band_6.5-8.5',
                             'band_6.75-8.75', 'band_7.0-9.0', 'band_7.25-9.25', 'band_7.5-9.5', 'band_7.75-9.75',
                             'band_8.0-10.0', 'band_8.25-10.25', 'band_8.5-10.5', 'band_8.75-10.75', 'band_9.0-11.0',
                             'band_9.25-11.25', 'band_9.5-11.5', 'band_9.75-11.75', 'band_10.0-12.0',
                             'band_10.25-12.25', 'band_10.5-12.5', 'band_10.75-12.75', 'band_11.0-13.0',
                             'band_11.25-13.25', 'band_11.5-13.5', 'band_11.75-13.75', 'band_12.0-14.0',
                             'band_12.25-14.25', 'band_12.5-14.5', 'band_12.75-14.75', 'band_13.0-15.0',
                             'band_13.25-15.25', 'band_13.5-15.5', 'band_13.75-15.75', 'band_14.0-16.0',
                             'band_14.25-16.25', 'band_14.5-16.5', 'band_14.75-16.75', 'band_15.0-17.0',
                             'band_15.25-17.25', 'band_15.5-17.5', 'band_15.75-17.75', 'band_16.0-18.0',
                             'band_16.25-18.25', 'band_16.5-18.5', 'band_16.75-18.75', 'band_17.0-19.0',
                             'band_17.25-19.25', 'band_17.5-19.5', 'band_17.75-19.75', 'band_18.0-20.0',
                             'band_18.25-20.25', 'band_18.5-20.5', 'band_18.75-20.75', 'band_19.0-21.0',
                             'band_19.25-21.25', 'band_19.5-21.5', 'band_19.75-21.75', 'band_20.0-22.0',
                             'band_20.25-22.25', 'band_20.5-22.5', 'band_20.75-22.75', 'band_21.0-23.0',
                             'band_21.25-23.25', 'band_21.5-23.5', 'band_21.75-23.75', 'band_22.0-24.0',
                             'band_22.25-24.25', 'band_22.5-24.5', 'band_22.75-24.75', 'band_23.0-25.0',
                             'band_23.25-25.25', 'band_23.5-25.5', 'band_23.75-25.75', 'band_24.0-26.0',
                             'band_24.25-26.25', 'band_24.5-26.5', 'band_24.75-26.75', 'band_25.0-27.0',
                             'band_25.25-27.25', 'band_25.5-27.5', 'band_25.75-27.75', 'band_26.0-28.0',
                             'band_26.25-28.25', 'band_26.5-28.5', 'band_26.75-28.75', 'band_27.0-29.0',
                             'band_27.25-29.25', 'band_27.5-29.5', 'band_27.75-29.75', 'band_28.0-30.0',
                             'band_28.25-30.25', 'band_28.5-30.5', 'band_28.75-30.75', 'band_29.0-31.0',
                             'band_29.25-31.25', 'band_29.5-31.5', 'band_29.75-31.75', 'band_30.0-32.0',
                             'n_0.25-2.25', 'n_0.5-2.5', 'n_0.75-2.75', 'n_1.0-3.0', 'n_1.25-3.25', 'n_1.5-3.5',
                             'n_1.75-3.75', 'n_2.0-4.0', 'n_2.25-4.25', 'n_2.5-4.5', 'n_2.75-4.75', 'n_3.0-5.0',
                             'n_3.25-5.25', 'n_3.5-5.5', 'n_3.75-5.75', 'n_4.0-6.0', 'n_4.25-6.25', 'n_4.5-6.5',
                             'n_4.75-6.75', 'n_5.0-7.0', 'n_5.25-7.25', 'n_5.5-7.5', 'n_5.75-7.75', 'n_6.0-8.0',
                             'n_6.25-8.25', 'n_6.5-8.5', 'n_6.75-8.75', 'n_7.0-9.0', 'n_7.25-9.25', 'n_7.5-9.5',
                             'n_7.75-9.75', 'n_8.0-10.0', 'n_8.25-10.25', 'n_8.5-10.5', 'n_8.75-10.75', 'n_9.0-11.0',
                             'n_9.25-11.25', 'n_9.5-11.5', 'n_9.75-11.75', 'n_10.0-12.0', 'n_10.25-12.25',
                             'n_10.5-12.5', 'n_10.75-12.75', 'n_11.0-13.0', 'n_11.25-13.25', 'n_11.5-13.5',
                             'n_11.75-13.75', 'n_12.0-14.0', 'n_12.25-14.25', 'n_12.5-14.5', 'n_12.75-14.75',
                             'n_13.0-15.0', 'n_13.25-15.25', 'n_13.5-15.5', 'n_13.75-15.75', 'n_14.0-16.0',
                             'n_14.25-16.25', 'n_14.5-16.5', 'n_14.75-16.75', 'n_15.0-17.0', 'n_15.25-17.25',
                             'n_15.5-17.5', 'n_15.75-17.75', 'n_16.0-18.0', 'n_16.25-18.25', 'n_16.5-18.5',
                             'n_16.75-18.75', 'n_17.0-19.0', 'n_17.25-19.25', 'n_17.5-19.5', 'n_17.75-19.75',
                             'n_18.0-20.0', 'n_18.25-20.25', 'n_18.5-20.5', 'n_18.75-20.75', 'n_19.0-21.0',
                             'n_19.25-21.25', 'n_19.5-21.5', 'n_19.75-21.75', 'n_20.0-22.0', 'n_20.25-22.25',
                             'n_20.5-22.5', 'n_20.75-22.75', 'n_21.0-23.0', 'n_21.25-23.25', 'n_21.5-23.5',
                             'n_21.75-23.75', 'n_22.0-24.0', 'n_22.25-24.25', 'n_22.5-24.5', 'n_22.75-24.75',
                             'n_23.0-25.0', 'n_23.25-25.25', 'n_23.5-25.5', 'n_23.75-25.75', 'n_24.0-26.0',
                             'n_24.25-26.25', 'n_24.5-26.5', 'n_24.75-26.75', 'n_25.0-27.0', 'n_25.25-27.25',
                             'n_25.5-27.5', 'n_25.75-27.75', 'n_26.0-28.0', 'n_26.25-28.25', 'n_26.5-28.5',
                             'n_26.75-28.75', 'n_27.0-29.0', 'n_27.25-29.25', 'n_27.5-29.5', 'n_27.75-29.75',
                             'n_28.0-30.0', 'n_28.25-30.25', 'n_28.5-30.5', 'n_28.75-30.75', 'n_29.0-31.0',
                             'n_29.25-31.25', 'n_29.5-31.5', 'n_29.75-31.75', 'n_30.0-32.0']

    df_context_columns_expected = ['timestamp', 'pump', 'velocity <v-MP> [RPM]',
                                   'p_delta_pressure_inlet <deltap-filter-MP> [barg]', 'p_in [barg]',
                                   'p_out <pout-MP> [barg]', 'p_delta [barg]', 'p_feedwater_tank <p-FWT> [barg]',
                                   't_inlet <T-in-MP> [Celsius]', 't_fwt <T-fwt> [Celsius]']

    if not set(df_V_columns_expected) == set(df_V_custom.columns):
        raise ValueError(f'For df_V, expected dataframe with columns {df_V_columns_expected}, '
                         f'but dataframe has columns {df_V_custom.columns}')

    elif not set(df_context_columns_expected) == set(df_contextual_custom.columns):
        raise ValueError(f'For df_contextual, expected dataframe with columns {df_context_columns_expected}, '
                         f'but dataframe has columns {df_contextual_custom.columns}')
    else:
        print('Checks passed')


def get_custom_dataset(path_to_df_V, path_to_df_contextual, split_date):
    df_V_custom = pd.read_csv(path_to_df_V, index_col=0)
    df_contextual_custom = pd.read_csv(path_to_df_contextual, index_col=0)

    df_contextual_custom = df_contextual_custom[df_contextual_custom['velocity <v-MP> [RPM]'] > 100]

    df_contextual_custom_train, df_contextual_custom_test = util.contextual_train_test_split(df_contextual_custom,
                                                                                             split_date)
    df_V_custom_train, df_V_custom_test, meta_data_custom = util.vibration_train_test_split(df_V_custom, split_date)

    BAND_COLS = df_V_custom.columns[df_V_custom.columns.str.extract('(band_)').notna()[0]]

    df_V_custom_train_normalized = SETUP['normalization'](df_V_custom_train, BAND_COLS)
    df_V_custom_test_normalized = SETUP['normalization'](df_V_custom_test, BAND_COLS)

    return df_V_custom_train_normalized, df_V_custom_test_normalized, df_contextual_custom_train, \
           df_contextual_custom_test
