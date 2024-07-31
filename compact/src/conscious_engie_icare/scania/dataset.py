# ©, 2024, Sirris
# owner: FFNG
import os
import pandas as pd
from tqdm import tqdm

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


def load_ts(add_labels=False):
    df_ts = pd.read_csv(TRAIN_OPERATIONAL_PATH, dtype={'vehicle_id': str})
    if add_labels:
        df_repairs = load_repairs()
        delta_times, classes = construct_label_per_timestamp(df_repairs, df_ts)
        df_ts['delta_time'] = delta_times
        df_ts['class'] = classes
    return df_ts


def load_repairs():
    return pd.read_csv(TRAIN_REPAIR_PATH)


"""
def construct_label_per_timestamp(df_repairs, df_ts):
    df_ts['vehicle_id'] = df_ts['vehicle_id'].astype(str)
    df_repairs['vehicle_id'] = df_repairs['vehicle_id'].astype(str)
    delta_times = []
    classes = []
    for idx, row in tqdm(df_ts.iterrows(), total=len(df_ts), desc='extracting risk classes of trucks'):
        group = df_repairs[df_repairs['vehicle_id'] == row['vehicle_id']]
        assert len(group) == 1
        repair_entry = group.iloc[0]
        delta_time = repair_entry['length_of_study_time_step'] - row['time_step']
        assert delta_time >= 0
        delta_times.append(delta_time)
        if repair_entry['in_study_repair'] == 0:
            classes.append(0)
        else:
            if delta_time <= 6:
                classes.append(4)
            elif delta_time <= 12:
                classes.append(3)
            elif delta_time <= 24:
                classes.append(2)
            elif delta_time <= 48:
                classes.append(1)
            else: 
                classes.append(1)
    
    return delta_time, classes
"""


def construct_label_per_timestamp(df_repairs, df_ts):
    df_ts['vehicle_id'] = df_ts['vehicle_id'].astype(str)
    df_repairs['vehicle_id'] = df_repairs['vehicle_id'].astype(str)
    df_merged = df_ts.merge(df_repairs, on='vehicle_id', suffixes=('_ts', '_repair'))
    df_merged['delta_time'] = df_merged['length_of_study_time_step'] - df_merged['time_step']
    assert (df_merged['delta_time'] >= 0).all()
    
    def classify(row):
        if row['in_study_repair'] == 0:
            return 0
        elif row['delta_time'] <= 6:
            return 4
        elif row['delta_time'] <= 12:
            return 3
        elif row['delta_time'] <= 24:
            return 2
        elif row['delta_time'] <= 48:
            return 1
        else:
            return 1

    tqdm.pandas()
    df_merged['class'] = df_merged.progress_apply(classify, axis=1)
    return df_merged['delta_time'].tolist(), df_merged['class'].tolist()


def load_and_preprocess_specifications(data_path=TRAIN_SPECIFICATIONS):
    df_specifications = pd.read_csv(data_path, index_col='vehicle_id')
    df_specifications = add_column_prefix(df_specifications)
    return df_specifications


def add_column_prefix(df):
    """ Add specification prefix for truck specifications. """
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f'{col}_{x}')
    return df


def train_test_split_dfs(df_ts, attribute_columns, truck_ids_train, truck_ids_test, meta_columns=None):
    meta_columns = meta_columns or ['vehicle_id', 'time_step', 'delta_time', 'class']
    df_ts = df_ts[meta_columns + attribute_columns]
    df_train = df_ts[df_ts.vehicle_id.isin(truck_ids_train.astype(str))]
    df_test = df_ts[df_ts.vehicle_id.isin(truck_ids_test.astype(str))]
    return df_train, df_test


def construct_decomposition_matrices(df_train, df_test):
    V_train = df_train[attribute_columns].to_numpy()
    V_test = df_test[attribute_columns].to_numpy()
    return V_train, V_test


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
