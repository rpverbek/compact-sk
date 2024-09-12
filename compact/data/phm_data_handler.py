# ©, 2024, Sirris
# owner: FFNG

import zipfile
import os
import pandas as pd
from scipy.signal import stft, welch
import glob

# train set
BASE_PATH_HEALTHY = os.path.join('..', 'data', 'Data_Challenge_PHM2023_training_data', 'Pitting_degradation_level_0')
FILE_NAMES_HEALTHY = glob.glob(os.path.join(BASE_PATH_HEALTHY, '*.txt'))

# cached results
# previous caching folder:
# CACHING_FOLDER_NAME = os.path.join('..', 'data', 'CACHED_RESULTS_300124')
CACHING_FOLDER_NAME = os.path.join('..', 'data', 'CACHED_RESULTS_030624')
FPATH_DF_ORDERS_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'df_orders_train_folds.pkl')
FPATH_DF_ORDERS_TRAIN = os.path.join(CACHING_FOLDER_NAME, 'df_orders_train.pkl')
FPATH_META_DATA_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'meta_data_train_folds.pkl')
FPATH_META_DATA_TRAIN = os.path.join(CACHING_FOLDER_NAME, 'meta_data_train.pkl')
FPATH_DF_V_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'df_V_train_folds.pkl')
FPATH_DF_V_TRAIN = os.path.join(CACHING_FOLDER_NAME, 'df_V_train.pkl')
FPATH_UNIQUE_NAME_MAPPING_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'unique_name_mapping_folds.pkl')
FPATH_UNIQUE_NAME_MAPPING = os.path.join(CACHING_FOLDER_NAME, 'unique_name_mapping.pkl')
FPATH_FINGERPRINTS_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'fingerprints_folds.pkl')
FPATH_FINGERPRINTS = os.path.join(CACHING_FOLDER_NAME, 'fingerprints.pkl')
FPATH_MODEL_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'model_folds.pkl')
FPATH_MODEL = os.path.join(CACHING_FOLDER_NAME, 'model.pkl')
FPATH_DISTANCES_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'distance_folds')
FPATH_DISTANCES = os.path.join(CACHING_FOLDER_NAME, 'distance')

# test set
PITTING_LEVELS = [1, 2, 3, 4, 6, 8]
FPATH_DATA_HEALTHY_TEST_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'data_healthy_test_folds.pkl')
FPATH_DATA_HEALTHY_TEST = os.path.join(CACHING_FOLDER_NAME, 'data_healthy_test.pkl')
FPATH_DATA_HEALTHY_TRAIN = os.path.join(CACHING_FOLDER_NAME, 'data_healthy_train.pkl')
BASE_PATHS_TEST = [os.path.join('..', 'data', 'Data_Challenge_PHM2023_training_data',
                                f'Pitting_degradation_level_{pl}') for pl in PITTING_LEVELS]
FPATH_DF_ORDERS_TEST_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'df_orders_test_folds.pkl')
FPATH_DF_ORDERS_TEST = os.path.join(CACHING_FOLDER_NAME, 'df_orders_test.pkl')
FPATH_META_DATA_TEST_FOLDS = os.path.join(CACHING_FOLDER_NAME, 'meta_data_test_folds.pkl')
FPATH_META_DATA_TEST = os.path.join(CACHING_FOLDER_NAME, 'meta_data_test.pkl')


def fetch_and_unzip_data(fname="Data_Challenge_PHM2023_training_data", force=False):
    """ Fetch and unzip data from the remote server. """
    fname_zip = f'{fname}.zip'
    local_path_zipped = os.path.join('..', 'data', fname_zip)
    local_path_unzipped = os.path.join('..', 'data', fname)

    if not os.path.exists(local_path_unzipped) or force:
        assert os.path.exists(local_path_zipped), f'Could not find {local_path_zipped}'
        print('Unzipping data...')
        with zipfile.ZipFile(local_path_zipped, 'r') as zip_ref:
            zip_ref.extractall(os.path.join('..', 'data'))

    return local_path_unzipped


def load_train_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load training data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'V{rpm}_{torque}N_{run}.txt')
    df = pd.read_csv(path, names=['x', 'y', 'z', 'tachometer'], delimiter=' ')
    return df


def load_test_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load test data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'{run}_V{rpm}_{torque}N.txt')
    df = pd.read_csv(path, names=['x', 'y', 'z', 'tachometer'], delimiter=' ')
    return df


def extract_process_parameters(file_path, use_train_data_for_validation=True):
    parts = file_path.split('/')
    filename = parts[-1]  # Extract the filename from the path
    if use_train_data_for_validation:
        v_value, n_value, sample_number = filename.split('_')  # Extract V, N, and sample number
        return int(v_value[1:]), int(n_value[:-1]), int(sample_number.split('.')[0])
    else:
        sample_number, v_value, n_value = filename.split('_')
        return int(v_value[1:]), int(n_value.split('.')[0][:-1]), int(sample_number)


def load_data(fnames, use_train_data_for_validation=True, base_path=BASE_PATH_HEALTHY, **kwargs):
    """ Load the complete data set.

    :param fnames: List of strings. Contains all the file names which should be loaded,
        e.g., ['V100_50N_1.txt', 'V3600_100N_4.txt', ...].
    :param use_train_data_for_validation: _description_, defaults to True.
    :param base_path: _description_, defaults to BASE_PATH_HEALTHY
    :return: _description_
    """    
    data = []
    f = None
    for fn in fnames:
        rpm, torque, run = extract_process_parameters(fn, use_train_data_for_validation=use_train_data_for_validation)
        if use_train_data_for_validation:
            df = load_train_data(rpm, torque, run, base_path=base_path)
        else:
            df = load_test_data(rpm, torque, run, base_path=base_path)

        f, t, stft_x = stft(df['x'], **kwargs)
        f, t, stft_y = stft(df['y'], **kwargs)
        f, t, stft_z = stft(df['z'], **kwargs)
        f, psd_x = welch(df['x'], **kwargs)
        f, psd_y = welch(df['y'], **kwargs)
        f, psd_z = welch(df['z'], **kwargs)
        data.append({
            'rpm': rpm,
            'torque': torque, 
            'sample_id': run,
            'unique_sample_id': f'{rpm}_{torque}_{run}',  # Remove the '.txt' extension and convert to integer
            'vibration_time_domain': df, 
            'stft_x': stft_x,
            'stft_y': stft_y,
            'stft_z': stft_z,  # Remove the '.txt' extension and convert to integer
            'psd_x': psd_x,
            'psd_y': psd_y,
            'psd_z': psd_z
        })
    return data, f


def load_cached_data(fpath_df_orders_train_folds=FPATH_DF_ORDERS_TRAIN_FOLDS,
                     fpath_meta_data_train_folds=FPATH_META_DATA_TRAIN_FOLDS):
    with open(fpath_df_orders_train_folds, 'rb') as file:
        # df_orders_train_folds = pickle.load(file)
        df_orders_train_folds = pd.read_pickle(file)
    with open(fpath_meta_data_train_folds, 'rb') as file:
        # meta_data_train_folds = pickle.load(file)
        meta_data_train_folds = pd.read_pickle(file)
    return df_orders_train_folds, meta_data_train_folds
