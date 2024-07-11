from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.metrics import davies_bouldin_score
from ydata_profiling import ProfileReport
import seaborn as sns
from sklearn.metrics import pairwise_distances
import pandas as pd
# from conscious_duracell.DBCV import DBCV
# import dbcv
# from hdbscan.validity import validity_index as dbcv
from scipy.spatial import distance
from tqdm import tqdm

def plot_dbcv(X, ax, epsilons=None, dbscan_params=None, dbcv_params=None):
    assert 'epsilon' not in dbscan_params
    epsilons = epsilons or np.arange(0.001, 0.4, 0.002)
    y_dict = {e: DBSCAN(eps=e, **dbscan_params).fit_predict(X=X) for e in tqdm(epsilons)}
    scores = {e: dbcv(X, y, **dbcv_params) for e, y in tqdm(y_dict.items())}
    ax.plot(pd.Series(scores))
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('density-based cluster validation')
    return scores


def plot_cumulative_explained_variance(X):
    pca = PCA().fit(X)

    fig, ax = plt.subplots()
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), cumulative_variance, marker='o', linestyle='--')

    thresholds = [0.9, 0.95, 0.99]
    for threshold in thresholds:
        components = next(x for x, val in enumerate(cumulative_variance) if val > threshold) + 1
        ax.plot([components, components], [min(cumulative_variance), threshold], color='grey', linestyle='-', alpha=0.5, label=None)
        ax.plot([1, components], [threshold, threshold], color='grey', linestyle='-', alpha=0.5, label=f'{int(threshold*100)}% threshold')

    ax.set_title('PCA: Explained Variance by Components')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    fig.show()
    return fig, ax


def calculate_kmeans_metrics(pipeline, X, clusters_range=None):

    # Range of clusters to explore
    clusters_range = clusters_range or range(2, 25)

    # Lists to store metrics
    inertia = []
    silhouette_scores = []
    db_scores = []

    for n_clusters in clusters_range:
        # Update the 'n_clusters' of KMeans in the pipeline
        pipeline.set_params(kmeans=KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
        pipeline.fit(X)
        
        # Get the cluster labels
        labels = pipeline.named_steps['kmeans'].labels_
        
        # Calculate metrics
        inertia.append(pipeline.named_steps['kmeans'].inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot Inertia
    axs[0].plot(clusters_range, inertia, marker='o', linestyle='--')
    axs[0].set_title('KMeans Inertia')
    axs[0].set_xlabel('Number of Clusters')
    axs[0].set_ylabel('Inertia')
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))

    # Plot Silhouette Score
    axs[1].plot(clusters_range, silhouette_scores, marker='o', linestyle='--', color='r')
    axs[1].set_title('Silhouette Score')
    axs[1].set_xlabel('Number of Clusters')
    axs[1].set_ylabel('Silhouette Score')
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))

    # Plot Davies-Bouldin Score
    axs[2].plot(clusters_range, db_scores, marker='o', linestyle='--', color='g')
    axs[2].set_title('Davies-Bouldin Score')
    axs[2].set_xlabel('Number of Clusters')
    axs[2].set_ylabel('Davies-Bouldin Score')
    axs[2].xaxis.set_minor_locator(MultipleLocator(1))
    fig.tight_layout()
    return fig, axs


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def generate_reports(df, type='minimal', name='Profiling', fname='profiling', **kwargs):
    if type=='minimal':
        profile = ProfileReport(df, title=name, minimal=True, **kwargs)
        profile.to_file(f"{fname}_minimal.html")
    elif type=='complete':
        profile = ProfileReport(df, title=name, minimal=False, **kwargs)
        profile.to_file(f"{fname}_complete.html")
    else:
        return None
    return profile


def plot_correlation_matrix(df, title='', corr_method='pearson'):
    correlation_matrix = df.corr(method=corr_method)
    fig, ax = plt.subplots(figsize=(18, 12))
    hm = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    cbar = hm.collections[0].colorbar
    cbar.set_label(f'correlation coefficient ({corr_method})', rotation=270, labelpad=15)
    ax.set_title(f'Correlation Matrix ({title})')
    return fig, ax


def plot_distance_matrix(dist_matrix, ax, name):
    ax = sns.heatmap(dist_matrix, cmap='viridis', ax=ax)
    ax.set_title(f'Pairwise distance matrix ({name})')
    return ax


def plot_distribution_of_distances(dist_matrix, ax, name):
    r = np.arange(len(dist_matrix))
    mask = r[:, None]<r
    flattened_lower_triangle = dist_matrix[mask]
    ax = sns.histplot(flattened_lower_triangle, ax=ax, bins=100)
    ax.set_title(f'Distribution of pairwise distances ({name})')
    return ax


def plot_elbow(dist_matrix, ax, name):
    sorted_distance_matrix_ = np.sort(dist_matrix)
    dist_mean_5NN_ = sorted_distance_matrix_[:, :5].mean(axis=1)
    ax.plot(np.sort(dist_mean_5NN_))
    ax.grid(axis='y')
    ax.set_xlabel('Points (sample) sorted by distance')
    ax.set_ylabel('mean 5-NN distance')
    ax.set_title(f'Elbow plot ({name})')
    return ax


class RangeClassifier:
    def __init__(self, peak_ranges):
        """
        Initializes the classifier with a dictionary of ranges.
        
        Parameters:
            peak_ranges (dict): A dictionary where the keys are the labels 
                                and the values are tuples representing the range (min, max).
        """
        self.peak_ranges = peak_ranges

    def fit_predict(self, values):
        """
        Predicts labels for each value in the series based on the specified ranges.

        Parameters:
            values (list or array-like): A series of numeric values to label.
        
        Returns:
            list: The labels for each entry in values.
        """
        labels = []
        for value in values:
            label_found = False
            for label, (min_val, max_val) in self.peak_ranges.items():
                if min_val <= value < max_val:
                    labels.append(label)
                    label_found = True
                    break
            if not label_found:
                labels.append(None)  # Append None or another indicator for values outside any range
        return labels