import argparse
import json
from datetime import datetime
from os.path import join, dirname

import numpy as np
from stockstats import wrap, unwrap
import pandas as pd
from alphas import get_alpha
import os
import statsmodels.api as sm

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)

def read_file(filename, index_col=1):
    # custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=index_col, parse_dates=True, infer_datetime_format=True)


def get_data_frame():
    if os.path.exists('computed_indicators.csv'):
        print(read_file('computed_indicators.csv', 0))
        return read_file('computed_indicators.csv', 0)
    
    data = read_file('pricing_data.csv')
    data = data.rename(columns={'Volume ETH': 'volume'})
    data = data.sort_index()

    voo_df = get_voo_data_frame()
    data = data.join(voo_df, how='left')

    qqq_df = get_qqq_data_frame()
    data = data.join(qqq_df, how='left')

    fear_greed_index_df = get_fear_greed_index_data_frame()
    fear_greed_index_df.index = fear_greed_index_df.index + \
        pd.Timedelta('24 hours')
    data = data.join(fear_greed_index_df, how='left')

    # sentiment_df = get_2nd_sentiment_data_frame()
    # # sentiment_df.index = sentiment_df.index + pd.Timedelta('1 hours')
    # data = data.join(sentiment_df, how='left')
    args = parser.parse_args()  

    sentiment_df = get_1st_sentiment_data_frame()    
    sentiment_df.index = sentiment_df.index + pd.Timedelta(str(args.lag + 1) + ' hours')
    data = data.join(sentiment_df, how='left')
    
    sentiment_df = get_2nd_sentiment_data_frame()
    sentiment_df.index = sentiment_df.index + pd.Timedelta(str(args.lag + 1) + ' hours')
    data = data.join(sentiment_df, how='left')

    sentiment_df = get_3rd_sentiment_data_frame()
    sentiment_df.index = sentiment_df.index + pd.Timedelta(str(args.lag + 1) + ' hours')
    data = data.join(sentiment_df, how='left')

    btc_df = get_btc_data_frame()
    data = data.join(btc_df, how='left')
    
    
    data['VOO_-1_r'] = data['VOO'].pct_change(1)
    data['QQQ_-1_r'] = data['QQQ'].pct_change(1)
    data['btcclose_-1_r'] = data['btcclose'].pct_change(1)
    
    df = wrap(data)
    # df.init_all()
    # df['volume_delta'] = df['volume_delta']
    df['close_-1_r'] = df['close_-1_r']
    # df['close_-2_r'] = df['close_-2_r']
    df['open_-1_r'] = df['open_-1_r']
    # df['open_-2_r'] = df['open_-2_r']
    df['high_-1_r'] = df['high_-1_r']
    # df['high_-2_r'] = df['high_-2_r']
    df['low_-1_r'] = df['low_-1_r']
    # df['low_-2_r'] = df['low_-2_r']
    df['volume_-1_r'] = df['volume_-1_r']
    # # df['volume-2_r'] = df['volume_-2_r']
    # df['kdjk_3_xu_kdjd_3'] = df['kdjk_3_xu_kdjd_3']
    df['close_5_sma'] = df['close_5_sma']
    # df['close_5_sma_-1_r'] = df['close_5_sma_-1_r']
    # df['close_3_sma'] = df['close_3_sma']
    # df['close_3_sma_-1_r'] = df['close_3_sma_-1_r']
    df['rsi_6'] = df['rsi_6']
    # df['rsi_6_-1_r'] = df['rsi_6_-1_r']
    # df['rsi_3'] = df['rsi_3']
    # df['rsi_3_-1_r'] = df['rsi_3_-1_r']
    df['wr_6'] = df['wr_6']
    # df['wr_6_-1_r'] = df['wr_6_-1_r']
    # df['wr_3'] = df['wr_3']
    # df['wr_3_-1_r'] = df['wr_3_-1_r']
    df['cci_6'] = df['cci_6']
    # df['cci_6_-1_r'] = df['cci_6_-1_r']
    # df['cci_3'] = df['cci_3']
    # df['cci_3_-1_r'] = df['cci_3_-1_r']
    df['trix_6_sma'] = df['trix_6_sma']
    # df['trix_6_sma_-1_r'] = df['trix_6_sma_-1_r']
    # df['trix_3_sma'] = df['trix_3_sma']
    # df['trix_3_sma_-1_r'] = df['trix_3_sma_-1_r']
    # df['close_5_sma_xd_close_15_sma'] = df['close_5_sma_xd_close_15_sma']
    # df['close_15_sma_xd_close_5_sma'] = df['close_15_sma_xd_close_5_sma']
    df['vwma_5'] = df['vwma_5']
    # df['vwma_5_-1_r'] = df['vwma_5_-1_r']
    df['chop_5'] = df['chop_5']
    # df['chop_5_-1_r'] = df['chop_5_-1_r']
    df['rsv_5'] = df['rsv_5']
    # df['rsv_5_-1_r'] = df['rsv_5_-1_r']
    df['stochrsi_5'] = df['stochrsi_5']
    # df['stochrsi_5_-1_r'] = df['stochrsi_5_-1_r']
    df['atr_5'] = df['atr_5']
    # df['atr_5_-1_r'] = df['atr_5_-1_r']
    # df['vr_5'] = df['vr_5']
    # df['vr_5_-1_r'] = df['vr_5_-1_r']
    # df['empty_-1_r'] = df['empty_-1_r']
    # df['sadness_-1_r'] = df['sadness_-1_r']
    # df['enthusiasm_-1_r'] = df['enthusiasm_-1_r']
    # df['neutral_-1_r'] = df['neutral_-1_r']
    # df['worry_-1_r'] = df['worry_-1_r']
    # df['surprise_-1_r'] = df['surprise_-1_r']
    # df['love_-1_r'] = df['love_-1_r']
    # df['fun_-1_r'] = df['fun_-1_r']
    # df['hate_-1_r'] = df['hate_-1_r']
    # df['happiness_-1_r'] = df['happiness_-1_r']
    # df['boredom_-1_r'] = df['boredom_-1_r']
    # df['relief_-1_r'] = df['relief_-1_r']
    # df['anger_-1_r'] = df['anger_-1_r']

    # df['close_15_sma'] = df['close_15_sma']
    # df['close_5_sma'] = df['close_5_sma']
    # df['close_sma_ratio'] = df['close_15_sma']/df['close_5_sma']
    # df['volume_15_sma'] = df['volume_15_sma']
    # df['volume_5_sma'] = df['volume_5_sma']
    # df['volume_sma_ratio'] = df['volume_15_sma']/df['volume_5_sma']
    # df['atr_15_sma'] = df['atr_15_sma']
    # df['atr_5_sma'] = df['atr_5_sma']
    # df['atr_sma_ratio'] = df['atr_15_sma']/df['atr_5_sma']
    # df['adx_15_sma'] = df['adx_15_sma']
    # df['adx_5_sma'] = df['adx_5_sma']
    # df['adx_sma_ratio'] = df['adx_15_sma']/df['adx_5_sma']
    # df['kdjd_15_sma'] = df['kdjd_15_sma']
    # df['kdjd_5_sma'] = df['kdjd_5_sma']
    # df['kdjd_sma_ratio'] = df['kdjd_15_sma']/df['kdjd_5_sma']
    # df['rsi_15_sma'] = df['rsi_15_sma']
    # df['rsi_5_sma'] = df['rsi_5_sma']
    # df['rsi_sma_ratio'] = df['rsi_15_sma']/df['rsi_5_sma']

    # df['ema_diff'] = df['close_10_ema'] - df['close_50_ema']
    # df['ups'], df['downs'] = df['change'] > 0, df['change'] < 0
    # df['o/c'] = df['open']/df['close']
    # df['h/c'] = df['high']/df['close']
    # df['l/c'] = df['low']/df['close']
    df = df.rename(columns={'open': 'Open', 'high': 'High',
                   'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = unwrap(df)
    df.drop(columns=["unix", "symbol", "tradecount"], inplace=True)


    df['Returns'] = df['Close'].pct_change(1)  
    
    v = df['Volume'].values
    tp = (df['Low'] + df['Close'] + df['High']).div(3).values
    df['Vwap'] = (tp * v).cumsum() / v.cumsum()

    
    # print(df['Close'])
    alphas_df = get_alpha(df.copy())
    alphas_df['Open'] = df['Open']
    alphas_df['High'] = df['High']
    alphas_df['Low'] = df['Low']
    alphas_df['Close'] = df['Close']
    df = alphas_df
    # print(df['Close'])
    
    df.iloc[:, 1:len(df.columns)] = df.iloc[:, 1:len(
        df.columns)].apply(pd.to_numeric)    
    df = df.fillna(method="ffill")
    
    df = df.dropna()
    df = df.round(4)
    # print(df.isnull().sum())
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df.columns)

    # print(df.head())
        # print(df.tail())
    # print(df.describe())
    # print(len(df.index))
    df.to_csv('computed_indicators.csv')

    return df



def get_btc_data_frame():
    data = read_file("Binance_BTCUSDT_1h.csv")
    data = data.rename(columns={
                       'open': 'btcopen', 'high': 'btchigh', 'low': 'btclow', 'close': 'btcclose'})
    data = data.sort_index()
    data.drop(columns=["unix", "symbol", "tradecount", "Volume BTC", "Volume USDT"], inplace=True)
    return data


def get_voo_data_frame():
    data = read_file("VOO_Hourly.csv")
    data = data.rename(columns={'MIDPOINT': 'VOO'})
    data = data.sort_index()
    data.pop("IDX")
    isnan = np.isnan(data)
    data = data[~isnan]
    isnan = np.isnan(data).any(axis=1)
    data = data[~isnan]
    return data


def get_qqq_data_frame():
    data = read_file("QQQ_Hourly.csv")
    data = data.rename(columns={'MIDPOINT': 'QQQ'})
    data = data.sort_index()
    data.pop("IDX")
    isnan = np.isnan(data)
    data = data[~isnan]
    isnan = np.isnan(data).any(axis=1)
    data = data[~isnan]
    return data


def get_fear_greed_index_data_frame():
    data = pd.read_csv("crypto_fear_greed_index_hourly.csv", index_col="date")
    data = data.rename(columns={'i': 'FEAR_GREED_INDEX'})
    data = data.sort_index()
    data.index = pd.to_datetime(data.index)
    isnan = np.isnan(data)
    data = data[~isnan]
    isnan = np.isnan(data).any(axis=1)
    data = data[~isnan]
    return data


def get_1st_sentiment_data_frame():
    sentiment_date = []
    sentiment_bullish = []
    sentiment_bearish = []
    sentiment_neutral = []
    with open("./predict.json", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            sentiment_date.append(json_obj['date'].replace('+00:00', ''))
            sentiment_bullish.append(json_obj['positive_score'])
            sentiment_bearish.append(json_obj['negative_score'])
            sentiment_neutral.append(json_obj['neutral_score'])
           
    sentiment_df = pd.DataFrame({'date': sentiment_date, 'positive_score': sentiment_bullish, 'negative_score': sentiment_bearish,
                                 '1stneutral': sentiment_neutral})
    sentiment_df = sentiment_df.set_index('date')
    sentiment_df = sentiment_df.sort_index()
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    return sentiment_df

def get_2nd_sentiment_data_frame():
    sentiment_date = []
    sentiment_empty = []
    sentiment_sadness = []
    sentiment_enthusiasm = []
    sentiment_neutral = []
    sentiment_worry = []
    sentiment_surprise = []
    sentiment_love = []
    sentiment_fun = []
    sentiment_hate = []
    sentiment_happiness = []
    sentiment_boredom = []
    sentiment_relief = []
    sentiment_anger = []
    with open("./predict_bert.json", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            sentiment_date.append(json_obj['date'].replace('+00:00', ''))
            sentiment_empty.append(json_obj['empty'])
            sentiment_sadness.append(json_obj['sadness'])
            sentiment_enthusiasm.append(json_obj['enthusiasm'])
            sentiment_neutral.append(json_obj['neutral'])
            sentiment_worry.append(json_obj['worry'])
            sentiment_surprise.append(json_obj['surprise'])
            sentiment_love.append(json_obj['love'])
            sentiment_fun.append(json_obj['fun'])
            sentiment_hate.append(json_obj['hate'])
            sentiment_happiness.append(json_obj['happiness'])
            sentiment_boredom.append(json_obj['boredom'])
            sentiment_relief.append(json_obj['relief'])
            sentiment_anger.append(json_obj['anger'])

    sentiment_df = pd.DataFrame({'date': sentiment_date, 'empty': sentiment_empty, 'sadness': sentiment_sadness,
                                 'enthusiasm': sentiment_enthusiasm, '2ndneutral': sentiment_neutral,
                                 'worry': sentiment_worry,
                                 'surprise': sentiment_surprise, 'love': sentiment_love, 'fun': sentiment_fun,
                                 'hate': sentiment_hate, 'happiness': sentiment_happiness, 'boredom': sentiment_boredom,
                                 'relief': sentiment_relief, 'anger': sentiment_anger})
    sentiment_df = sentiment_df.set_index('date')
    sentiment_df = sentiment_df.sort_index()
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    return sentiment_df



def get_3rd_sentiment_data_frame():
    sentiment_date = []
    sentiment_bullish = []
    sentiment_bearish = []
    sentiment_neutral = []
    with open("./predict_rnn_sentiment.json", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            sentiment_date.append(json_obj['datetime'].replace('+00:00', ''))
            sentiment_bullish.append(json_obj['bullish'])
            sentiment_bearish.append(json_obj['bearish'])
            sentiment_neutral.append(json_obj['neutral'])
           
    sentiment_df = pd.DataFrame({'date': sentiment_date, 'bullish': sentiment_bullish, 'bearish': sentiment_bearish,
                                 '3rdneutral': sentiment_neutral})
    sentiment_df = sentiment_df.set_index('date')
    sentiment_df = sentiment_df.sort_index()
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    return sentiment_df

if __name__ == '__main__':
    get_data_frame()
