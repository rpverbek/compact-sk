import os
import pandas as pd

#DATA_PATH = os.path.join('..', '..', 'data', 'IDA 2024 Industrial Challenge')
DATA_PATH = os.path.join('..', '..', 'work', 'uc_scania')
TRAIN_OPERATIONAL_PATH = os.path.join(DATA_PATH, 'train_operational_readouts.csv')
TRAIN_REPAIR_PATH = os.path.join(DATA_PATH, 'train_tte.csv')
TRAIN_SPECIFICATIONS = os.path.join(DATA_PATH, 'train_specifications.csv')

TEST_PATH = os.path.join(DATA_PATH, 'public_X_test.csv')
VARIANTS_PATH = os.path.join(DATA_PATH, 'variants.csv')

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
    return pd.read_csv(TRAIN_REPAIR_PATH)
    


def load_and_preprocess_specifications(data_path=TRAIN_SPECIFICATIONS):
    df_specifications = pd.read_csv(data_path, index_col='vehicle_id')
    df_specifications = add_column_prefix(df_specifications)
    return df_specifications


def add_column_prefix(df):
    """ Add specification prefix for truck specifications. """
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f'{col}_{x}')
    return df


def get_attribute_columns(df_ts):
    ATTRIBUTE_COLUMNS = df_ts.columns.str.extract(r'(\d+_\d+)')[0]
    ATTRIBUTE_COLUMNS = ATTRIBUTE_COLUMNS[~ATTRIBUTE_COLUMNS.isna()].tolist()
    return ATTRIBUTE_COLUMNS


def extract_list_of_attribute_columns(df_ts):
    attribute_columns = get_attribute_columns(df_ts)
    ts_columns = df_ts[attribute_columns].columns
    all_sensors = ts_columns.str.extract(r'(\d+)')[0].unique()
    list_of_attribute_columns = [[col for col in ts_columns if s in col] for s in all_sensors]
    return list_of_attribute_columns
