
"""
Preprocessing
"""

from conscious_engie_icare.viz.spectrogram import plot_stft, plot_periodogram, plot_welch
from conscious_engie_icare.data.phm_data_handler import BASE_PATH_HEALTHY, FILE_NAMES_HEALTHY, load_train_data, \
    fetch_and_unzip_data, extract_process_parameters, load_data, load_cached_data, BASE_PATHS_TEST
from conscious_engie_icare.normalization import normalize_1
from conscious_engie_icare.nmf_profiling import derive_df_orders, derive_df_vib
from conscious_engie_icare.data.phm_data_handler import BASE_PATH_HEALTHY, FILE_NAMES_HEALTHY, CACHING_FOLDER_NAME, \
    fetch_and_unzip_data, load_data, load_cached_data, FPATH_DF_ORDERS_TRAIN, FPATH_META_DATA_TRAIN, \
    FPATH_DF_V_TRAIN, FPATH_DATA_HEALTHY_TEST, FPATH_DATA_HEALTHY_TRAIN, FPATH_DF_ORDERS_TEST, FPATH_META_DATA_TEST, \
    PITTING_LEVELS
import random
import os
import pandas as pd
import pickle
from tqdm import tqdm
import glob


def convert_to_frequency_domain(nperseg=10240, nfft=None, fs=20480, split=0.75):

    # if os.path.exists(FPATH_DATA_HEALTHY_TRAIN) and os.path.exists(FPATH_DATA_HEALTHY_TEST) and not recompute:
    #     data_healthy_train = pd.read_pickle(FPATH_DATA_HEALTHY_TRAIN)
    #     data_healthy_test = pd.read_pickle(FPATH_DATA_HEALTHY_TEST)

    noverlap = nperseg // 2
    data_healthy, f = load_data(FILE_NAMES_HEALTHY, nperseg=nperseg, noverlap=noverlap, nfft=nfft, fs=fs)

    split_id = int(len(data_healthy) * split)
    random.Random(0).shuffle(data_healthy)
    data_healthy_train = data_healthy[:split_id]
    data_healthy_test = data_healthy[split_id:]
    #
    # # save data_healthy_test
    # os.makedirs(os.path.dirname(FPATH_DATA_HEALTHY_TRAIN), exist_ok=True)
    # with open(FPATH_DATA_HEALTHY_TRAIN, 'wb') as file:
    #     pickle.dump(data_healthy_train, file)
    #
    # # save data_healthy_test
    # os.makedirs(os.path.dirname(FPATH_DATA_HEALTHY_TEST), exist_ok=True)
    # with open(FPATH_DATA_HEALTHY_TEST, 'wb') as file:
    #     pickle.dump(data_healthy_test, file)

    return data_healthy_train, data_healthy_test, f


def transform_orders_and_bin(data_healthy_train, f, recompute=False):
    # load transformed data (if specified)
    if os.path.exists(FPATH_DF_ORDERS_TRAIN) and os.path.exists(FPATH_META_DATA_TRAIN) and not recompute:
        df_orders_train, meta_data_train = load_cached_data(
            fpath_df_orders_train_folds=FPATH_DF_ORDERS_TRAIN,
            fpath_meta_data_train_folds=FPATH_META_DATA_TRAIN
        )

    # load train data and transform to orders
    else:
        setup = {'start': 0.5, 'stop': 100.5, 'n_windows': 50, 'window_steps': 2, 'window_size': 2}
        df_vib_train = derive_df_vib(data_healthy_train, f)
        # print(df_vib_train.shape)
        df_orders_train, meta_data_train = derive_df_orders(df_vib_train, setup, f,
                                                            verbose=False)
        df_orders_train[meta_data_train.columns] = meta_data_train

        # cache train data
        os.makedirs(os.path.dirname(FPATH_DF_ORDERS_TRAIN), exist_ok=True)
        with open(FPATH_DF_ORDERS_TRAIN, 'wb') as file:
            pickle.dump(df_orders_train, file)
        # cache test data
        os.makedirs(os.path.dirname(FPATH_META_DATA_TRAIN), exist_ok=True)
        with open(FPATH_META_DATA_TRAIN, 'wb') as file:
            pickle.dump(meta_data_train, file)

    return df_orders_train, meta_data_train


def normalize_orders(df_orders_train, recompute=False):
    if os.path.exists(FPATH_DF_V_TRAIN) and not recompute:
        df_V_train = pd.read_pickle(FPATH_DF_V_TRAIN)
    else:
        cols = df_orders_train.columns
        BAND_COLS = cols[cols.str.contains('band')].tolist()
        df_V_train = normalize_1(df_orders_train, BAND_COLS)

        with open(FPATH_DF_V_TRAIN, 'wb') as file:
            pickle.dump(df_V_train, file)
    return df_V_train


def get_and_preprocess_healthy_data(recompute=False):
    fetch_and_unzip_data()
    data_healthy_train, df_data_healthy_test, f = convert_to_frequency_domain()
    df_orders_train, meta_data_train = transform_orders_and_bin(data_healthy_train, f, recompute)
    df_V_train = normalize_orders(df_orders_train, recompute)

    return df_V_train, meta_data_train, df_data_healthy_test, f


def get_and_preprocess_unhealthy_data(df_data_healthy_test, f, recompute=False):
    # load transformed data (if possible)
    if os.path.exists(FPATH_DF_ORDERS_TEST) and os.path.exists(FPATH_META_DATA_TEST) and not recompute:
        with open(FPATH_DF_ORDERS_TEST, 'rb') as file:
            df_orders_test = pickle.load(file)
        with open(FPATH_META_DATA_TEST, 'rb') as file:
            meta_data_test = pickle.load(file)

    # transform test data to orders
    else:
        # convert pitting test samples to orders
        df_orders_test_pitting_dict = {}
        meta_data_test_pitting_dict = {}
        for lvl, path in tqdm(list(zip(PITTING_LEVELS, BASE_PATHS_TEST)),
                              desc='Extracting and order-transforming test data'):
            # load data for each level of pitting
            fnames = glob.glob(os.path.join(path, '*.txt'))
            nperseg = 10240
            noverlap = nperseg // 2
            nfft = None
            fs = 20480
            data_test, f = load_data(fnames, nperseg=nperseg, noverlap=noverlap, nfft=nfft, fs=fs, base_path=path,
                                     use_train_data_for_validation=True)

            # extract vibration data
            df_vib_test_unhealthy = derive_df_vib(data_test, f)

            # convert to orders and derive meta data
            setup = {'start': 0.5, 'stop': 100.5, 'n_windows': 50, 'window_steps': 2,
                     'window_size': 2}  # also used in 02 - data preprocessing
            df_orders_test_pitting_, meta_data_test_pitting_ = derive_df_orders(df_vib_test_unhealthy, setup, f,
                                                                                verbose=False)
            rpm = meta_data_test_pitting_['rotational speed [RPM]']
            torque = meta_data_test_pitting_['torque [Nm]']
            run = meta_data_test_pitting_['sample_id']
            meta_data_test_pitting_['unique_sample_id'] = rpm.astype(str) + '_' + torque.astype(str) + '_' + run.astype(
                str) + f'_pitting_level_{lvl}'
            df_orders_test_pitting_['unique_sample_id'] = meta_data_test_pitting_['unique_sample_id']
            df_orders_test_pitting_dict[lvl] = df_orders_test_pitting_
            meta_data_test_pitting_dict[lvl] = meta_data_test_pitting_

        # convert healthy test samples to orders
        meta_data_test_healthy_folds = []
        df_orders_test_healthy_folds = []
        df_vib_test_healthy = derive_df_vib(df_data_healthy_test, f)
        df_orders_test_healthy, meta_data_test_healthy = derive_df_orders(df_vib_test_healthy, setup, f, verbose=False)
        meta_data_test_healthy['unique_sample_id'] = meta_data_test_healthy['unique_sample_id'] + '_healthy'
        df_orders_test_healthy['unique_sample_id'] = meta_data_test_healthy['unique_sample_id']

        # concat all pitting levels samples
        df_orders_test_pitting = pd.concat(list(df_orders_test_pitting_dict.values()))
        meta_data_test_pitting = pd.concat(list(meta_data_test_pitting_dict.values()))

        # merge healthy and unhealthy samples
        # only use operating modes in the test set that are also in the training set
        om_test_healthy = meta_data_test_healthy['rotational speed [RPM]'].astype(str) + '_' + \
                          meta_data_test_healthy['torque [Nm]'].astype(str)
        om_test_pitting = meta_data_test_pitting['rotational speed [RPM]'].astype(str) + '_' + \
                          meta_data_test_pitting['torque [Nm]'].astype(str)
        new_meta_data_test_pitting_without_missing_oms = meta_data_test_pitting[
            om_test_pitting.isin(om_test_healthy)]
        new_df_orders_test_pitting_without_missing_oms = df_orders_test_pitting[
            om_test_pitting.isin(om_test_healthy)]

        # sample equal amount of samples from healthy and faulty data
        om_test_pitting_with_run = new_meta_data_test_pitting_without_missing_oms['rotational speed [RPM]'].\
                                       astype(str) + '_' \
                                   + new_meta_data_test_pitting_without_missing_oms['torque [Nm]'].astype(str) + '_' \
                                   + new_meta_data_test_pitting_without_missing_oms['sample_id'].astype(str)
        om_test_healthy_with_run = meta_data_test_healthy['rotational speed [RPM]'].astype(str) + '_' \
                                   + meta_data_test_healthy['torque [Nm]'].astype(str) + '_' \
                                   + meta_data_test_healthy['sample_id'].astype(str)
        n_samples = len(om_test_healthy_with_run.unique())
        samples = new_df_orders_test_pitting_without_missing_oms['unique_sample_id'].sample(n_samples,
                                                                                            random_state=0,
                                                                                            replace=False)
        new_meta_data_test_pitting = new_meta_data_test_pitting_without_missing_oms[
            new_meta_data_test_pitting_without_missing_oms['unique_sample_id'].isin(samples)]
        new_df_orders_test_pitting = new_df_orders_test_pitting_without_missing_oms[
            new_df_orders_test_pitting_without_missing_oms['unique_sample_id'].isin(samples)]
        df_orders_test = pd.concat([df_orders_test_healthy, new_df_orders_test_pitting]).reset_index(drop=True)
        meta_data_test = pd.concat([meta_data_test_healthy, new_meta_data_test_pitting]).reset_index(drop=True)

        # cache data
        with open(FPATH_DF_ORDERS_TEST, 'wb') as file:
            pickle.dump(df_orders_test, file)
        with open(FPATH_META_DATA_TEST, 'wb') as file:
            pickle.dump(meta_data_test, file)

    return df_orders_test, meta_data_test
