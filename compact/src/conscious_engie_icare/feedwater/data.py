# ©, 2024, Sirris
# owner: FFNG

import pandas as pd
import os


LOCAL_PATH_DATA = os.path.join('..', '..', 'work', 'uc_feedwater_pumps')  # TODO
LOCAL_PATH_OPERATIONAL = os.path.join(LOCAL_PATH_DATA, 'operational_data')


def load_and_preprocess_operational_features(pump, base_path=LOCAL_PATH_OPERATIONAL):
    file_name = f'operational_features_pump{pump}.csv'
    file_path = os.path.join(base_path, file_name)
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pump'] = pump
    return df