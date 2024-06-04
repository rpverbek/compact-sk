"""
Normalisation algorithms of frequency bands.
"""
import pandas as pd


def no_normalize(df_, band_cols):
    """ Dummy function for not normalizing data. """
    return df_[band_cols]


def normalize_1(df_, band_cols):
    """ The given decomposition matrix V with row v_i is normalized as v_ij' = v_ij / sum(v_i). """
    df_V_normalized = df_[band_cols]
    sum_ = df_V_normalized.sum(axis=1)
    df_V_normalized = pd.DataFrame([df_V_normalized[col].div(sum_) for col in df_V_normalized.columns]).T.dropna()
    df_V_normalized.columns = band_cols
    assert ((df_V_normalized.sum(axis=1) >= 0.9999) & (df_V_normalized.sum(axis=1) <= 1.0001)).all(), 'The frequency bands in each timestamp are expected to sum up to 1.'
    # assert len(df_V_normalized) == len(df)
    return df_V_normalized


def normalize_2(df_, band_cols):
    """ The given decomposition matrix V with row v_i is normalized as v_ij' = v_ij / n * sum(v_i). """
    df_V_normalized = df_[band_cols]
    sum_ = df_V_normalized.sum(axis=1)
    N_COLS = df_.columns[df_.columns.str.extract('(n_)').notna()[0]]
    n_ = df_[N_COLS].sum(axis=1)
    df_V_normalized = pd.DataFrame([df_V_normalized[col].div(n_ * sum_) for col in df_V_normalized.columns]).T.dropna()
    assert len(df_V_normalized) == len(df_)
    df_V_normalized.columns = band_cols
    return df_V_normalized


def normalize_3(df_, band_cols):
    """ The given decomposition matrix V with row v_i is normalized as v_ij' = v_ij / n. """
    df_V_normalized = df_[band_cols]
    N_COLS = df_.columns[df_.columns.str.extract('(n_)').notna()[0]]
    n_ = df_[N_COLS].sum(axis=1)
    df_V_normalized = pd.DataFrame([df_V_normalized[col].div(n_) for col in df_V_normalized.columns]).T.dropna()
    assert len(df_V_normalized) == len(df_)
    df_V_normalized.columns = band_cols
    return df_V_normalized
