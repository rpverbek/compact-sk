from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from conscious_engie_icare.util import calc_fpr_at_tpr_threshold, calculate_roc_characteristics
from tqdm import tqdm

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

# from conscious_engie_icare.autoencoder import Autoencoder


class Benchmarking:
    def __init__(self, df_W_offline_folds, df_W_online_folds, df_cosine_folds,
                 train_vibration_measurement_periods_folds, test_vibration_measurement_periods_folds):
        self.df_W_offline_folds = df_W_offline_folds
        self.df_W_online_folds = df_W_online_folds
        self.df_cosine_folds = df_cosine_folds
        self.train_vibration_measurement_periods_folds = train_vibration_measurement_periods_folds
        self.test_vibration_measurement_periods_folds = test_vibration_measurement_periods_folds

    def run_test(self):
        approaches = [
            # {'name': 'IForest', 'function': self.train_isolation_forest},
            {'name': 'PCA (old)', 'function': self.train_pca_old, 'kwargs': {}},
            {'name': 'PCA (new, 0.999)', 'function': self.train_pca_new, 'kwargs': {}},
            {'name': 'PCA (new, 0.5)', 'function': self.train_pca_new, 'kwargs': {'n_components': 0.5}},
            {'name': 'PCA + Crossvalidation', 'function': self.train_pca_with_crossvalidation, 'kwargs': {}},
            {'name': '1cSVM+ Crossvalidation', 'function': self.train_one_class_svm_with_crossvalidation, 'kwargs': {}},
        ]
        trials = []
        for approach in approaches:
            for trial in tqdm(list(range(100)), desc=f'Approach: {approach["name"]}'):
                # calculate results
                results = approach['function'](trial, **approach['kwargs'])
                # add meta info
                results = dict({'trial': trial, 'approach': approach['name']}, **results)
                trials.append(results)
        df_roc_curves = pd.DataFrame(trials)
        return df_roc_curves

    def run_all_approaches(self, n=100, verbose=True):
        approaches = [
            {'name': 'IForest', 'function': self.train_isolation_forest},
            {'name': 'IForest+ Meta Data', 'function': self.train_isolation_forest_with_metadata},
            {'name': 'IForest+ PCA', 'function': self.train_isolation_forest_with_pca},
            {'name': 'Isolation Forest+ PCA+ Meta Data', 'function': self.train_isolation_forest_with_pca_and_metadata},
            {'name': 'IForest+ Hyperparameter tuning', 'function': self.train_isolation_forest_with_crossvalidation},
            {'name': '1cSVM', 'function': self.train_one_class_svm},
            {'name': '1cSVM+ Hyperparameter tuning', 'function': self.train_one_class_svm_with_crossvalidation},
            {'name': '1cSVM+ Meta Data', 'function': self.train_one_class_svm_with_metadata},
            {'name': '1cSVM+ PCA', 'function': self.train_one_class_svm_with_pca},
            # {'name': '1-class SVM+ PCA+ Meta Data', 'function': train_one_class_svm_with_pca_and_metadata},   # interrupts the kernel
            # {'name': 'PCA (old)', 'function': self.train_pca_old},
            {'name': 'PCA', 'function': self.train_pca_new},
            {'name': 'Our method', 'function': self.get_results_of_our_method}
        ]
        trials = []
        for approach in approaches:
            for trial in tqdm(list(range(n)), desc=f'Approach: {approach["name"]}', disable=not verbose):
                # calculate results
                results = approach['function'](trial)
                # add meta info
                results = dict({'trial': trial, 'approach': approach['name']}, **results)
                trials.append(results)
        df_roc_curves = pd.DataFrame(trials)
        return df_roc_curves

    def run_autoencoder(self, n=100, verbose=True):
        autoencoder = Autoencoder()
        results = []
        for trial in tqdm(list(range(n)), desc=f'Approach: Autoencoder', disable=not verbose):
            results.append(autoencoder.train(trial))
        df_roc_curves = pd.DataFrame(results)
        return df_roc_curves

    def train_isolation_forest(self, fold_nr, **kwargs):
        """ Detect anomalies with a standard Isolation Forest. """
        pipeline = IsolationForest()
        return self.train_standard_anomaly_detection(fold_nr, pipeline=pipeline, **kwargs)

    def train_isolation_forest_with_crossvalidation(self, fold_nr, **kwargs):
        """ Detect anomalies with a standard Isolation Forest with crossvalidation. """
        pipeline = IsolationForest()
        param_grid = {
            'n_estimators': [10, 100, 200, 500],
            'max_samples': [0.1, 0.5, 1.0, 'auto'],
            'max_features': [0.1, 0.5, 1.0, 10, 100]
        }
        return self.train_standard_anomaly_detection_with_crossvalidation(fold_nr, pipeline=pipeline,
                                                                          param_grid=param_grid, **kwargs)

    def train_isolation_forest_with_metadata(self, fold_nr, verbose=False):
        return self.train_isolation_forest(fold_nr, verbose=verbose, use_meta_data=True)

    def train_isolation_forest_with_pca(self, fold_nr, **kwargs):
        pipeline = Pipeline([('pca', PCA(n_components=0.999)), ('clf', IsolationForest(contamination=0))])
        return self.train_standard_anomaly_detection(fold_nr, pipeline=pipeline, **kwargs)

    def train_isolation_forest_with_pca_and_metadata(self, fold_nr, verbose=False):
        return self.train_isolation_forest_with_pca(fold_nr, verbose=verbose, use_meta_data=True)

    def train_one_class_svm(self, fold_nr, **kwargs):
        pipeline = OneClassSVM(kernel='rbf', gamma='auto')
        return self.train_standard_anomaly_detection(fold_nr, pipeline=pipeline, **kwargs)

    def train_one_class_svm_with_crossvalidation(self, fold_nr, **kwargs):
        pipeline = OneClassSVM()
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'nu': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        }
        return self.train_standard_anomaly_detection_with_crossvalidation(fold_nr, pipeline=pipeline,
                                                                          param_grid=param_grid, **kwargs)

    def train_one_class_svm_with_metadata(self, fold_nr, verbose=False):
        return self.train_one_class_svm(fold_nr, verbose=verbose, use_meta_data=True)

    def train_one_class_svm_with_pca(self, fold_nr, **kwargs):
        pipeline = Pipeline([('pca', PCA(n_components=0.999)), ('clf', OneClassSVM(kernel='linear', gamma='auto'))])
        return self.train_standard_anomaly_detection(fold_nr, pipeline=pipeline, **kwargs)

    def train_standard_anomaly_detection(self, fold_nr, pipeline, use_meta_data=False, verbose=False):
        binned_vibrations_train = self.train_vibration_measurement_periods_folds[fold_nr]
        binned_vibrations_test = self.test_vibration_measurement_periods_folds[fold_nr]
        df_cosine_test = self.df_cosine_folds[fold_nr]

        # frequency band columns are all columns that contain the string 'band_'
        # --> there are 50 frequency bands per sensor
        exemplary_vibration_column_names = binned_vibrations_train[0].columns
        is_vibration_column = exemplary_vibration_column_names.str.contains('band_')
        frequency_band_column_names = exemplary_vibration_column_names[is_vibration_column]
        assert len(frequency_band_column_names) == 50

        # create a matrix representation of the feature space for the train set,
        # where the three different sensors are stacked to a single vector with 150 features
        # (50 frequency bands per sensor)
        # --> this is the feature space for the clustering algorithm
        X_train = self.create_matrix_representation_train_set(binned_vibrations_train, fold_nr,
                                                              frequency_band_column_names,
                                                              use_meta_data=use_meta_data, verbose=verbose)

        # Fit Pipeline (without preprocessing the data)
        pipeline.fit(X_train)
        # y_pred_train = pipeline.predict(X_train)
        # if verbose:
        #    print(f'Number of outliers detected in training set: {np.sum(y_pred_train == -1)} (should be 0)')

        # create a matrix representation for the test set
        X_test = self.create_matrix_representation_test_set(binned_vibrations_test, fold_nr,
                                                            frequency_band_column_names,
                                                            use_meta_data=use_meta_data, verbose=verbose)

        # predict outliers in test set
        y_test = df_cosine_test['pitting'].replace({True: -1, False: 1}).to_numpy()
        # y_pred_test = pipeline.predict(X_test)
        y_score_test = pipeline.score_samples(X_test)  # the lower, the more abnormal
        # if verbose:
        #    print(f'Number of outliers detected in test set with default parameters: {np.sum(y_pred_test == -1)}',
        #          '(should be {np.sum(y_test == -1)})')

        # create ROC curve for test set
        results = self.calc_metrics_(y_test, y_score_test, verbose=verbose)
        return results

    def train_standard_anomaly_detection_with_crossvalidation(self, fold_nr, pipeline, param_grid, validation_ratio=0.5,
                                                              use_meta_data=False, verbose=False):
        # TODO: crossvalidation does not work with roc_auc
        binned_vibrations_train = self.train_vibration_measurement_periods_folds[fold_nr]
        binned_vibrations_test = self.test_vibration_measurement_periods_folds[fold_nr]
        df_cosine_test = self.df_cosine_folds[fold_nr]

        # frequency band columns are all columns that contain the string 'band_'
        # --> there are 50 frequency bands per sensor
        exemplary_vibration_column_names = binned_vibrations_train[0].columns
        is_vibration_column = exemplary_vibration_column_names.str.contains('band_')
        frequency_band_column_names = exemplary_vibration_column_names[is_vibration_column]

        # create a matrix representation of the feature space for the train set,
        # where the three different sensors are stacked to a single vector with 150 features
        # (50 frequency bands per sensor)
        # --> this is the feature space for the clustering algorithm
        X_train = self.create_matrix_representation_train_set(binned_vibrations_train, fold_nr,
                                                              frequency_band_column_names,
                                                              use_meta_data=use_meta_data, verbose=verbose)

        # create a matrix representation for the validation/test set
        X_val_test = self.create_matrix_representation_test_set(binned_vibrations_test, fold_nr,
                                                                frequency_band_column_names,
                                                                use_meta_data=use_meta_data, verbose=verbose)
        X_val = X_val_test[:int(validation_ratio * len(X_val_test))]
        X_test = X_val_test[int(validation_ratio * len(X_val_test)):]
        y_val_test = df_cosine_test['pitting'].replace({True: -1, False: 1}).to_numpy()
        y_val = y_val_test[:int(validation_ratio * len(y_val_test))]
        y_test = y_val_test[int(validation_ratio * len(y_val_test)):]

        # Gridsearch
        X_train_val = np.vstack((X_train, X_val))
        y_train_val = np.concatenate((np.ones(len(X_train)), y_val)).astype(int)
        assert len(X_train_val) == len(y_train_val)
        # train_ind = list(range(len(X_train)))
        # val_ind = list(range(len(X_train), len(X_train_val)))
        train_ind = np.ones(len(X_train), dtype=int) * -1
        val_ind = np.zeros(len(X_val), dtype=int)
        assert len(train_ind) + len(val_ind) == len(X_train_val)
        ind = np.concatenate((train_ind, val_ind))
        assert len(ind) == len(X_train_val)
        predefined_split = PredefinedSplit(test_fold=ind)
        assert predefined_split.get_n_splits() == 1
        # debugging code for scoring = 'roc_auc' (which did not work, but making a custom scorer did work)
        # split_ = next(predefined_split.split())
        # print(f'train set in predefined_split should only contain non-anomalous data: {split_}')
        # X_train_ = X_train_val[split_[0]]
        # X_val_ = X_train_val[split_[1]]
        # y_train_ = y_train_val[split_[0]]
        # y_val_ = y_train_val[split_[1]]
        # print("y_train_:", y_train_)
        # print("y_val_:", y_val_)
        # pipeline_ = pipeline
        # pipeline_.fit(X_train_val, y=y_train_val)
        # y_pred_train_ = pipeline_.predict(X_train_)
        # y_pred_val_ = pipeline_.predict(X_val_)
        # roc_auc_train_ = roc_auc_score(y_train_, y_pred_train_)
        # print("roc_auc_train_:", roc_auc_train_)
        # roc_auc_val_ = roc_auc_score(y_val_, y_pred_val_)
        # print("roc_auc_val_:", roc_auc_val_)
        # assert True, 'validation set in predefined_split should contain anomalous and non-anomalous data'
        roc_auc_scorer = make_scorer(roc_auc_score)
        grid_search_clf = GridSearchCV(pipeline, param_grid=param_grid, cv=predefined_split, scoring=roc_auc_scorer,
                                       error_score="raise")
        #print('fitting grid search')
        grid_search_clf.fit(X_train_val, y=y_train_val)
        #print('grid search fitted')
        cv_results = {'cv_results': grid_search_clf.cv_results_}

        # Fit pipeline (without preprocessing the data)
        # y_pred_train = grid_search_clf.predict(X_train)
        # if verbose:
        #    print(f'Number of outliers detected in training set: {np.sum(y_pred_train == -1)} (should be 0)')

        # predict outliers in test set
        # y_pred_test = grid_search_clf.predict(X_test)
        y_score_test = grid_search_clf.score_samples(X_test)  # the lower, the more abnormal
        # if verbose:
        #    print('Number of outliers detected in test set with default parameters:',
        #          f'{np.sum(y_pred_test == -1)} (should be {np.sum(y_test == -1)})')

        # create ROC curve for test set
        results = self.calc_metrics_(y_test, y_score_test, verbose=verbose)
        results.update(cv_results)
        return results

    def train_reconstruction_error_based_approach(self, fold_nr, pipeline, use_meta_data=False, verbose=False):
        # DEPRECATED
        binned_vibrations_train = self.train_vibration_measurement_periods_folds[fold_nr]
        binned_vibrations_test = self.test_vibration_measurement_periods_folds[fold_nr]
        df_cosine_test = self.df_cosine_folds[fold_nr]

        # frequency band columns are all columns that contain the string 'band_'
        # --> there are 50 frequency bands per sensor
        exemplary_vibration_column_names = binned_vibrations_train[0].columns
        is_vibration_column = exemplary_vibration_column_names.str.contains('band_')
        frequency_band_column_names = exemplary_vibration_column_names[is_vibration_column]
        assert len(frequency_band_column_names) == 50

        # create a matrix representation of the feature space for the train set,
        # where the three different sensors are stacked to a single vector with 150 features
        # (50 frequency bands per sensor)
        # --> this is the feature space for the clustering algorithm
        X_train = self.create_matrix_representation_train_set(binned_vibrations_train, fold_nr,
                                                              frequency_band_column_names,
                                                              use_meta_data=use_meta_data, verbose=verbose)

        # Fit Isolation Forest (without preprocessing the data)
        X_train_reconstructed = pipeline.fit_transform(X_train)
        reconstrucion_error_train = np.sum((X_train - pipeline.inverse_transform(X_train_reconstructed))**2, axis=1)
        if verbose:
            print('Reconstruction error train')
            print(pd.Series(reconstrucion_error_train).describe())

        # create a matrix representation for the test set
        X_test = self.create_matrix_representation_test_set(binned_vibrations_test, fold_nr,
                                                            frequency_band_column_names,
                                                            use_meta_data=use_meta_data, verbose=verbose)

        # predict outliers in test set
        y_test = df_cosine_test['pitting'].replace({True: -1, False: 1}).to_numpy()
        X_test_reconstructed = pipeline.transform(X_test)
        reconstrucion_error_test = np.sum((X_test - pipeline.inverse_transform(X_test_reconstructed))**2, axis=1)
        y_score_test = max(reconstrucion_error_test) - reconstrucion_error_test  # the higher, the more abnormal
        if verbose:
            print('Reconstruction error test')
            print(pd.Series(reconstrucion_error_test).describe())

        # create ROC curve for test set
        results = self.calc_metrics_(y_test, y_score_test, verbose=verbose)
        return results

    def train_pca_old(self, fold_nr, use_meta_data=False, verbose=False):
        pipeline = PCA(n_components=0.999)
        return self.train_reconstruction_error_based_approach(fold_nr, pipeline, use_meta_data=use_meta_data,
                                                              verbose=verbose)

    def train_pca_new(self, fold_nr, n_components=0.999, use_meta_data=False, verbose=False):
        pipeline = PCA_anomaly_detector(n_components=n_components)
        return self.train_standard_anomaly_detection(fold_nr, pipeline=pipeline, use_meta_data=use_meta_data,
                                                     verbose=verbose)

    def train_pca_with_crossvalidation(self, fold_nr, **kwargs):
        pipeline = PCA_anomaly_detector()
        param_grid = {
            'n_components': [1, 2, 50],
            # 'svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            # 'tol': [0.0, 0.0001, 0.001, 0.01, 0.1]
        }
        return self.train_standard_anomaly_detection_with_crossvalidation(fold_nr, pipeline=pipeline,
                                                                          param_grid=param_grid,
                                                                          **kwargs)

    def create_matrix_representation_train_set(self, binned_vibrations_train, fold_nr, frequency_band_column_names,
                                               use_meta_data=False, verbose=False):
        X_train = np.array([flatten_df(individual_measurements[frequency_band_column_names])
                            for individual_measurements in binned_vibrations_train])
        if use_meta_data:
            meta_data_train = pd.DataFrame({
                'rpm': self.df_W_offline_folds[fold_nr]['unique_sample_id'].str.extract(r'^(\d+)_')[0],
                'torque': self.df_W_offline_folds[fold_nr]['unique_sample_id'].str.extract(r'_(\d+)_')[0],
            })
            X_train = np.hstack((X_train, meta_data_train.to_numpy()))
        if verbose:
            print(f'Shape of X_train: {X_train.shape}')
        return X_train

    def create_matrix_representation_test_set(self, binned_vibrations_test, fold_nr, frequency_band_column_names,
                                              use_meta_data=False, verbose=False):
        X_test = np.array([flatten_df(individual_measurements[frequency_band_column_names])
                           for individual_measurements in binned_vibrations_test])
        if use_meta_data:
            meta_data_test = pd.DataFrame({
                'rpm': self.df_W_online_folds[fold_nr]['unique_sample_id'].str.extract(r'^(\d+)_')[0],
                'torque': self.df_W_online_folds[fold_nr]['unique_sample_id'].str.extract(r'_(\d+)_')[0],
            })
            X_test = np.hstack((X_test, meta_data_test.to_numpy()))
        if verbose:
            print(f'Shape of X_test: {X_test.shape}')
        return X_test

    def calc_metrics_(self, y_test, y_score_test, verbose=False):
        # create ROC curve for test set
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score_test)
        roc_auc = auc(fpr, tpr)
        fpr_at_tpr = calc_fpr_at_tpr_threshold(tpr, fpr, threshold=0.9)
        if verbose:
            print(f'AUC: {roc_auc:.3f}')

        results = {
            'fpr': fpr,
            'tpr': tpr,
            'fpr_at_tpr': fpr_at_tpr,
            'thresholds': thresholds,
            'roc_auc': roc_auc,
        }
        return results

    def get_results_of_our_method(self, fold_nr):
        df_cosine_ = self.df_cosine_folds[fold_nr]
        df_cosine_ = df_cosine_[df_cosine_.unique_cluster_label != -1]  # QUICK FIX !!! : removed unknown cluster labels

        # Plot the general ROC curve
        fpr, tpr, roc_auc = calculate_roc_characteristics(df_cosine_)
        fpr_at_tpr = calc_fpr_at_tpr_threshold(tpr, fpr, threshold=0.9)
        results = {
            'fpr': fpr,
            'tpr': tpr,
            'fpr_at_tpr': fpr_at_tpr,
            'thresholds': np.NaN,
            'roc_auc': roc_auc,
        }
        return results


class PCA_anomaly_detector(OutlierMixin, BaseEstimator):
    def __init__(self, n_components=None, svd_solver='auto', tol=0.0):
        self.pca = PCA(n_components=n_components, svd_solver=svd_solver, tol=tol)
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol

        # !!!
        self.q = 0.95

    def fit(self, X, y=None):
        """ Fit PCA. """
        self.X_ = X
        self.pca.fit(X)
        # extract the quantile
        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        reconstruction_error = np.sum((X - X_reconstructed)**2, axis=1)
        self.quantile = np.quantile(reconstruction_error, self.q)

        # Return the classifier
        return self

    def predict(self, X):
        # print('predict called for PCA_anomaly_detector')
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Calculate reconstruction error
        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        reconstruction_error = np.sum((X - X_reconstructed)**2, axis=1)
        # Predict outliers
        y_pred = np.ones(len(X))
        y_pred[reconstruction_error > self.quantile] = -1
        return y_pred

    def score_samples(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        # Calculate reconstruction error
        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        reconstruction_error = np.sum((X - X_reconstructed)**2, axis=1)
        # Predict outliers
        return max(reconstruction_error) - reconstruction_error  # !!!


def flatten_df(df_):
    return df_.to_numpy().flatten()
