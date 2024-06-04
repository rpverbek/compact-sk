# ©, 2024, Sirris
# owner: FFNG

import zipfile
from elucidata.resources.pipeline import DownloadFile
import posixpath
import os
import pandas as pd
import signal
from tqdm import tqdm
from scipy.signal import stft, welch
import pickle
import glob


BASE_PATH_HEALTHY = os.path.join('..', 'data', 'Data_Challenge_PHM2023_training_data', 'Pitting_degradation_level_0')
FILE_NAMES_HEALTHY = glob.glob(os.path.join(BASE_PATH_HEALTHY, '*.txt'))
# previous caching folder:
# CACHING_FOLDER_NAME = os.path.join('..', 'data', 'CACHED_RESULTS_300124')
CACHING_FOLDER_NAME = os.path.join('..', 'data', 'CACHED_RESULTS_030624')
FPATH_DF_ORDERS_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, f'df_orders_train_folds.pkl')
FPATH_META_DATA_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, f'meta_data_train_folds.pkl')
FPATH_DF_V_TRAIN_FOLDS = os.path.join(CACHING_FOLDER_NAME, f'df_V_train_folds.pkl')


def fetch_and_unzip_data(fname="Data_Challenge_PHM2023_training_data", force=False):
    """ Fetch and unzip data from the remote server. """
    fname_zip = fname + '.zip'
    local_path_zipped = os.path.join('..', 'data', fname_zip)
    local_path_unzipped = os.path.join('..', 'data', fname)

    if not os.path.exists(fname):
        # download zipped data (if not already present locally)
        remote_path = posixpath.join('outputs', 'FAIR2', 'compact', 'phm23', fname_zip)
        local_path = os.path.join('..', 'data', fname_zip)
        DownloadFile(local_path_zipped, remote_path).make(force=force)

        if not os.path.exists(os.path.join('..', 'data', fname)):
            # unzip data
            print('Unzipping data...')
            with zipfile.ZipFile(local_path_zipped, 'r') as zip_ref:
                zip_ref.extractall(os.path.join('..', 'data'))

    return local_path_unzipped


def load_train_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load training data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'V{rpm}_{torque}N_{run}.txt')
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


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def load_data(fnames, use_train_data_for_validation=True, base_path=BASE_PATH_HEALTHY, **kwargs):
    """ Load the complete data set.

    :param fnames: List of strings. Contains all the file names which should be loaded,
        e.g., ['V100_50N_1.txt', 'V3600_100N_4.txt', ...].
    :param use_train_data_for_validation: _description_, defaults to True.
    :param base_path: _description_, defaults to BASE_PATH_HEALTHY
    :return: _description_
    """    
    data = []
    for fn in tqdm(fnames):
        rpm, torque, run = extract_process_parameters(fn, use_train_data_for_validation=use_train_data_for_validation)
        try:
            with timeout(seconds=4):
                if use_train_data_for_validation:
                    df = load_train_data(rpm, torque, run, base_path=base_path) 
                else:
                    df = load_test_data(rpm, torque, run, base_path=base_path)
        except TimeoutError:
            print(f'timed out loading {fn}')
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


'''
def load_test_data(rpm, torque, run, base_path=BASE_PATH_HEALTHY):
    """ Load test data for a given run, rpm and torque. """
    path = os.path.join(base_path, f'{run}_V{rpm}_{torque}N.txt')
    df = pd.read_csv(path, names=['x', 'y', 'z', 'tachometer'], delimiter=' ')
    return df
'''
