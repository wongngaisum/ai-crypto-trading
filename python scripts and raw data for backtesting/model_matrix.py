from numbers import Number
from typing import Sequence

import numpy as np
import pandas as pd


def get_X_columns():
    return [
        'empty',
        'sadness'
        'enthusiasm'
        '2ndneutral'
        'worry',
        'surprise',
        'love',
        'fun',
        'hate',
        'happiness',
        'boredom',
        'relief',
        'anger',
        'positive_score',
        'negative_score',
        'neutral_score',
        'bullish',
        'bearish',
        '3rdneutral',
        'VOO_-1_r',
        'QQQ_-1_r',
        'btcclose_-1_r',
        'FEAR_GREED_INDEX',
        'close_-1_r',
        'open_-1_r',
        'high_-1_r',
        'low_-1_r',
        'volume_-1_r',
        'close_5_sma',
        'rsi_6',
        'wr_6',
        'cci_6',
        'trix_6_sma',
        'vwma_5',
        'chop_5',
        'rsv_5',
        'stochrsi_5',
        'atr_5',
        'alpha002',
        'alpha003',
        'alpha004',
        'alpha005',
        'alpha006',
        'alpha007',
        'alpha008',
        'alpha009',
        'alpha010',
        'alpha011',
        'alpha012',
        'alpha013',
        'alpha014',
        'alpha015',
        'alpha016',
        'alpha017',
        'alpha018',
        'alpha019',
        'alpha020',
        'alpha021',
        'alpha022',
        #'alpha023',
        'alpha024',
        'alpha025',
        'alpha026',
        'alpha027',
        'alpha028',
        'alpha029',
        'alpha030',
        'alpha031',
        'alpha032',
        'alpha033',
        'alpha034',
        'alpha035',
        'alpha036',
        'alpha037',
        'alpha038',
        'alpha039',
        'alpha040',
        'alpha041',
        'alpha042',
        'alpha043',
        'alpha044',
        'alpha045',
        'alpha046',
        'alpha047',
        'alpha049',
        'alpha050',
        'alpha051',
        'alpha052',
        'alpha053',
        'alpha054',
        'alpha055',
        'alpha057',
        'alpha060',
        'alpha061',
        'alpha062',
        'alpha064',
        'alpha065',
        'alpha066',
        'alpha068',
        #'alpha071',
        'alpha072',
        #'alpha073',
        'alpha074',
        'alpha075',
        #'alpha077',
        'alpha078',
        'alpha081',
        'alpha083',
        'alpha085',
        'alpha086',
        #'alpha088',
        #'alpha092',
        'alpha094',
        'alpha095',
        #'alpha096',
        'alpha098',
        'alpha099',
        'alpha101',
    ]


def get_X(data, columns=None):
    # data['VOO'] = data['VOO'].fillna(method="ffill")
    # data['QQQ'] = data['QQQ'].fillna(method="ffill")
    """Return model design matrix X"""
    # print(data.filter(get_X_columns()))
    if columns is None:
        columns = get_X_columns()
    return data.filter(columns)

def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(1).shift(-1)  # Returns after roughly 1 hour
    # Devalue returns smaller than 0.4%
    y[y.between(-.004, .004)] = 0
    y[y > 0] = 1
    y[y < 0] = -1
    return y


def get_clean_Xy(df, columns=None):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df, columns)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    isnan = np.isnan(X).any(axis=1)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

def crossover_below(series1: Sequence, series2: Sequence) -> bool:
    """
    Return `True` if `series1` just crossed over (below)
    `series2`.

        >>> crossover(self.data.Close, self.sma)
        True
    """
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] > series2[-2] and series1[-1] < series2[-1]
    except IndexError:
        return False