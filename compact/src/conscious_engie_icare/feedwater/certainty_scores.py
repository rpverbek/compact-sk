import pandas as pd
from tqdm import tqdm
import os


def calc_certainty_score_v1(row, radius=0.1):
    """ Certainty score = 1 if exactly 1 centroid is within the radius, else 0. """
    centroids_within_radius = row.apply(lambda x: x < radius).sum()
    if centroids_within_radius == 0:
        return 0
    elif centroids_within_radius == 1:
        return 1
    else:
        return 0


def calc_certainty_score_v2(row, radius=0.1):
    """ Certainty score = 1 - dist_closest_centroid/radius if exactly 1 centroid is within the radius, else 0. """
    centroids_within_radius = row.apply(lambda x: x < radius).sum()
    if centroids_within_radius == 0:
        return 0
    elif centroids_within_radius == 1:
        return 1 - row.min()/radius
    else:
        return 0


def calc_certainty_score_v3(row, radius=0.1):
    """ Certainty score = 1 - dist_closest_centroid/radius if at least 1 centroid is within the radius, else 0. """
    centroids_within_radius = row.apply(lambda x: x < radius).sum()
    if centroids_within_radius == 0:
        return 0
    else:
        return 1 - row.min()/radius


def calc_certainty_score_svm_based(row):
    """ Certainty score based on classifier uncertainty. """
    raise NotImplementedError


def load_uncertainty_score(name, short_name, function, distances_):
    try:
        res = pd.read_csv(name, index_col=0).iloc[:, 0]
        res.name = short_name
    except FileNotFoundError:
        tqdm.pandas(desc=f'calculating {name}...')
        res = distances_.progress_apply(function, axis=1)
        os.makedirs(os.path.dirname(name), exist_ok=True)
        res.to_csv(name)
    return res
