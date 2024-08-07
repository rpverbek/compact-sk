# ©, 2024, Sirris
# owner: FFNG

import pandas as pd
import os
from conscious_engie_icare import LOCAL_PATH_NMF_MODELS
from conscious_engie_icare.normalization import no_normalize, normalize_1, normalize_2, normalize_3

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
