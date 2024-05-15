# ©, 2022, Sirris
# owner: FFNG
""" Distance metrics used for anomaly detection. """
import numpy as np
from sklearn.metrics import pairwise
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns

def frobenius_norm(a, b):
    """ Frobenius norm of the differences. """
    return np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro')


def frobenius_norm_v2(a, b):
    """ Power of the Frobenius norm of the differences. """
    return np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro')**2


def frobenius_norm_v3(a, b):
    """ Square root of the Frobenius norm of the differences. """
    return np.sqrt(np.linalg.norm(a.sort_index() - b.sort_index(), ord='fro'))


def matrix_distance_(dist_function, A, B):
    """ Given two matrices A and B, transform them to vectors and calculate their pairwise distance. """
    return dist_function(
        A.to_numpy().flatten().reshape(1, -1),
        B.to_numpy().flatten().reshape(1, -1)
    )[0][0]


def cosine_distance(A, B):
    """ Cosine distance of two matrices A and B. """
    return matrix_distance_(pairwise.cosine_distances, A, B)


def manhattan_distance(A, B):
    """ Manhattan distance of two matrices A and B. """
    return matrix_distance_(pairwise.manhattan_distances, A, B)


def mahalanobis_distance(A, B, SI):
    """ Mahalanobis distance of two matrices A and B given the inverse of the co-variance matrix (SI)."""
    # unstack matrices A and B
    a = A.to_numpy().flatten().reshape(1, -1)    # !!! 96 dimensions (each sensor)
    b = B if isinstance(a, np.ndarray) else B.to_numpy()
    # b = B#.flatten().reshape(1, -1)  # !!! 16 dimensions (only mean of sensor)
    dist_ = mahalanobis(a, b, SI)
    return dist_


def visualize_distances(df_dist_train_, df_dist_online_):
    bins = np.arange(0, 1, 0.025)
    fig, axes = plt.subplots(figsize=(15, 5), ncols=4, nrows=2, sharex=True, sharey=True)
    # top row with distances in training set
    axes.flat[0].set_ylabel('Count (training set)')
    sns.histplot(df_dist_train_, x='frobenius_norm', ax=axes.flat[0], bins=bins).set_title('Frobenius norm')
    sns.histplot(df_dist_train_, x='frobenius_norm_sqrt', ax=axes.flat[1], bins=bins).set_title('Frobenius norm (sqrt)')
    sns.histplot(df_dist_train_, x='cosine_distance', ax=axes.flat[2], bins=bins).set_title('Cosine distance')
    sns.histplot(df_dist_train_, x='manhattan_distance', ax=axes.flat[3], bins=bins).set_title('Manhattan distance')

    # bottom row with distances in test set
    axes.flat[4].set_ylabel('Count (test set)')
    sns.histplot(df_dist_online_, x='frobenius_norm', ax=axes.flat[4], bins=bins).set_title('Frobenius norm')
    sns.histplot(df_dist_online_, x='frobenius_norm_sqrt', ax=axes.flat[5], bins=bins).set_title(
        'Frobenius norm (sqrt)')
    sns.histplot(df_dist_online_, x='cosine_distance', ax=axes.flat[6], bins=bins).set_title('Cosine distance')
    sns.histplot(df_dist_online_, x='manhattan_distance', ax=axes.flat[7], bins=bins).set_title('Manhattan distance')
    fig.tight_layout()

    return fig, axes


def visualize_distances_for_om(df_dist_train_, df_dist_online_, om='A'):
    fig, ax = visualize_distances(df_dist_train_[df_dist_train_.om == om], df_dist_online_[df_dist_online_.om == om])
    fig.suptitle(f'Distances for operating mode {om}')
    fig.tight_layout()
    return fig, ax


def plot_distance_to_fingerprint(df_dist_train_, idx, df_dist_online_, distribution_of_training_set=True, super_impose_online_dist=False, metric='frobenius_norm'):
    # plot underlying distribution
    df_dist_ = df_dist_train_ if distribution_of_training_set else df_dist_online_
    g = sns.displot(data=df_dist_, x=metric, col="om", col_wrap=4, height=2, aspect=4, bins=50, kind="hist");

    # superimpose vertical stripe for one given distance
    df_dist_ = df_dist_online_ if super_impose_online_dist else df_dist_train_
    for ax in g.axes:
        title = ax.get_title()
        om = title[-1]
        dist_ = df_dist_[(df_dist_.idx == idx) & (df_dist_.om == om)].iloc[0][metric]
        ax.axvline(dist_, color='red')
        ax.set_title(f'd(s_{idx}, {title}) = {round(dist_, 3)}')

    title_part_1 = "online test set" if super_impose_online_dist else "offline train set"
    title_part_2 = "train set" if distribution_of_training_set else "test set"
    g.fig.suptitle(f"{metric} for vibration measurement {idx} from " + title_part_1 + " on top of distribution of " + title_part_2, size=20)
    g.fig.tight_layout()
