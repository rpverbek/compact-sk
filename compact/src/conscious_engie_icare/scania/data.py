# ©, 2024, Sirris
# owner: FFNG
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


def load_repairs():
    return
    
