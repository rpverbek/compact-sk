# TODO: REMOVE THIS FILE, AS IT ESSENTIALLY DUPLCIATES dataset.py
"""
import os
import pandas as pd

DATA_ROOT_PATH = os.path.join('..', '..', 'work', 'uc_scania')  # TODO: change this path
TRAIN_OPERATIONAL_PATH = os.path.join(DATA_ROOT_PATH, 'train_operational_readouts.csv')
TRAIN_REPAIR_PATH = os.path.join(DATA_ROOT_PATH, 'train_tte.csv')
TRAIN_SPECIFICATIONS = os.path.join(DATA_ROOT_PATH, 'train_specifications.csv')

# Function to extract attribute columns
def extract_attribute_columns(path):
    df_ts_peak = pd.read_csv(path, nrows=2)
    ATTRIBUTE_COLUMNS = df_ts_peak.columns.str.extract(r'(\d+_\d+)')[0]
    ATTRIBUTE_COLUMNS = ATTRIBUTE_COLUMNS[~ATTRIBUTE_COLUMNS.isna()].tolist()
    ATTRIBUTE_COLUMNS_397 = [c for c in ATTRIBUTE_COLUMNS if '397' in c]
    SELECTED_COLUMNS = ['vehicle_id', 'time_step'] + ATTRIBUTE_COLUMNS_397
    return ATTRIBUTE_COLUMNS, ATTRIBUTE_COLUMNS_397, SELECTED_COLUMNS

ATTRIBUTE_COLUMNS, ATTRIBUTE_COLUMNS_397, SELECTED_COLUMNS = extract_attribute_columns(TRAIN_OPERATIONAL_PATH)


def load_ts():
    return pd.read_csv(TRAIN_OPERATIONAL_PATH, dtype={'vehicle_id': str})


def train_test_split_dfs(df_ts, attribute_columns, truck_ids_train, truck_ids_test):
    df_ts = df_ts[['vehicle_id', 'time_step'] + attribute_columns]
    df_train = df_ts[df_ts.vehicle_id.isin(truck_ids_train.astype(str))]
    df_test = df_ts[df_ts.vehicle_id.isin(truck_ids_test.astype(str))]
    return df_train, df_test


def construct_decomposition_matrices(df_train, df_test):
    V_train = df_train[attribute_columns].to_numpy()
    V_test = df_test[attribute_columns].to_numpy()
    return V_train, V_test


def load_repairs():
    return
"""
    
