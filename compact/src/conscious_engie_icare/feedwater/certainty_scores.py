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


def calculate_uncertainty_score(distances, name='certainty_score', function=(lambda x: calc_certainty_score_v3(x, radius=0.2))):
    """ Calculate the uncertainty score. """
    tqdm.pandas(desc=f'calculating {name}...')
    res = distances.progress_apply(function, axis=1)
    res.name = name
    return res


def load_uncertainty_score(short_name, function, distances_, file_name=None, cache_certainty_score=False):
    """ Old function to load (or calculate) uncertainty score. """
    def calc_score_():
        tqdm.pandas(desc=f'calculating {file_name}...')
        res = distances_.progress_apply(function, axis=1)
        res.name = short_name
        return res

    if cache_certainty_score:
        try:
            res = pd.read_csv(file_name, index_col=0).iloc[:, 0]
            res.name = short_name
        except FileNotFoundError:
            res = calc_score_()
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            res.to_csv(file_name)
    else:
        res = calc_score_()
    return res
