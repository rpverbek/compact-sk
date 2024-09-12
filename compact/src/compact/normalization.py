"""
Normalisation algorithms of frequency bands.
"""
import pandas as pd


def normalize_1(df_, band_cols):
    """ The given decomposition matrix V with row v_i is normalized as v_ij' = v_ij / sum(v_i). """
    df_V_normalized = df_[band_cols]
    sum_ = df_V_normalized.sum(axis=1)
    df_V_normalized = pd.DataFrame([df_V_normalized[col].div(sum_) for col in df_V_normalized.columns]).T.dropna()
    df_V_normalized.columns = band_cols
    assert ((df_V_normalized.sum(axis=1) >= 0.9999) & (df_V_normalized.sum(axis=1) <= 1.0001)).all(), 'The frequency bands in each timestamp are expected to sum up to 1.'
    return df_V_normalized
