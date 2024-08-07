# ©, 2024, Sirris
# owner: FFNG
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from conscious_engie_icare.feedwater import util
from tqdm import tqdm
from sklearn.decomposition import PCA


def extract_operating_modes(df_contextual_train, n_clusters, order_cluster_names=True):
    contextual_groups = []
    om_pipes = {}
    for pump, contextual_group in tqdm(df_contextual_train.groupby('pump')):
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=0.99)),
            ('kmeans', KMeans(n_clusters=n_clusters[pump]))
        ])
        X = contextual_group.drop(columns=['timestamp', 'pump'])
        contextual_group['cluster_kmeans'] = pipe.fit_predict(X)
        contextual_groups.append(contextual_group)
        om_pipes[pump] = pipe
    df_contextual_train_with_labels = pd.concat(contextual_groups)

    if order_cluster_names:
        # replace cluster names 1, 2, ... with letters unique letters
        df_contextual_train_with_labels, name_mapping = util.order_cluster_names(
            df_contextual_train_with_labels, col_name_clusters='cluster_kmeans'
        )

    return df_contextual_train_with_labels, om_pipes


def calculate_score(X, dimensionality_reduction, name,
                    n_components=None, max_number_clusters=5,
                    clustering=KMeans, base_folder='cache_scores',
                    **param_dimensionality_reduction):
    # try to load scores from cache
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    scores_name = os.path.join(base_folder, f'{name}.csv')
    try:
        df_scores = pd.read_csv(scores_name)
    except FileNotFoundError:
        print(f'Calculating scores for {scores_name}')

        # calculate scores for different number of components
        n_components = [2, 3, 4] if n_components is None else n_components
        scores = []
        for n in n_components:

            # dimensionality reduction
            dimensionality_reduction_pipe = Pipeline([
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', dimensionality_reduction(n_components=n,
                                                                      random_state=None,
                                                                      **param_dimensionality_reduction))
            ])
            s = find_optimal_number_of_clusters(X, dimensionality_reduction_pipe,
                                                max_number_clusters=max_number_clusters,
                                                algorithm=clustering,
                                                random_state=100,
                                                base_folder=base_folder,
                                                file_name_base=f'{name}_{n}-components')
            s['n_components'] = n

            # pickle pipeline
            name_pipeline = os.path.join(base_folder, name + f'_pipeline_n{n}.pickel')
            s['pipeline'] = name_pipeline
            with open(name_pipeline, 'wb') as file:
                pickle.dump(dimensionality_reduction_pipe, file, protocol=pickle.HIGHEST_PROTOCOL)

            # add information to scores
            scores.append(s)
        df_scores = pd.concat(scores)
        df_scores.to_csv(scores_name)

    return df_scores


def find_optimal_number_of_clusters(X,
                                    dimensionality_reduction_pipe,
                                    algorithm=KMeans,
                                    min_number_clusters=2, max_number_clusters=6,
                                    base_folder=None,
                                    file_name_base=None,
                                    **clustering_param):
    """ Compare clusters with respect to specific metrics. """
    X_file_name = os.path.join(base_folder, f'X_compressed_{file_name_base}.csv')

    # extract compressed feature space
    X_compressed = dimensionality_reduction_pipe.fit_transform(X)

    # calculate internal clustering metrics for different number of clusters
    scores = []
    for n_clusters in range(min_number_clusters, max_number_clusters + 1):
        y_file_name = os.path.join(base_folder, f'y_{file_name_base}_{n_clusters}-clusters.csv')
        y = KMeans(n_clusters=n_clusters, **clustering_param).fit_predict(X_compressed)
        y = y + 1  # start with cluster 1
        s = {
            'number_clusters_tested': n_clusters,
            'silhouette_score': metrics.silhouette_score(X_compressed, y),
            'calinski_harabasz_score': metrics.calinski_harabasz_score(X_compressed, y),
            'davies_bouldin_score': metrics.davies_bouldin_score(X_compressed, y),
            'X_compressed': X_file_name,
            'y': y_file_name,
        }
        scores.append(s)
        np.savetxt(y_file_name, y, delimiter=",")
        np.savetxt(X_file_name, X_compressed, delimiter=",")
    return pd.DataFrame(scores)
