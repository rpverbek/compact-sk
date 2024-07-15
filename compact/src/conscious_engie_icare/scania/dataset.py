import os
import pandas as pd

DATA_PATH = os.path.join('..', '..', 'data', 'IDA 2024 Industrial Challenge')
TRAIN_OPERATIONAL_PATH = os.path.join(DATA_PATH, 'train_operational_readouts.csv')
TRAIN_REPAIR_PATH = os.path.join(DATA_PATH, 'train_tte.csv')
TRAIN_SPECIFICATIONS = os.path.join(DATA_PATH, 'train_specifications.csv')

TEST_PATH = os.path.join(DATA_PATH, 'public_X_test.csv')
VARIANTS_PATH = os.path.join(DATA_PATH, 'variants.csv')


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
