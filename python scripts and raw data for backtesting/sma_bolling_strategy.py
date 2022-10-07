import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model_matrix import get_clean_Xy, get_X, get_y, crossover_below
import argparse

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)
args = parser.parse_args()  


N_TRAIN = 5091



def std(arr: pd.Series, n: int) -> pd.Series:
    return pd.Series(arr).rolling(n).std()

# def std(arr: pd.Series, n: int) -> pd.Series:
#     first = np.empty(N_TRAIN)
#     first[:] = np.nan
#     after = pd.Series(arr[N_TRAIN:]).rolling(n).std()
#     return pd.Series(first).append(after).reset_index(inplace=False, drop=True)

# def SMA(arr: pd.Series, n: int) -> pd.Series:
#     """
#     Returns `n`-period simple moving average of array `arr`.
#     """
#     first = np.empty(N_TRAIN)
#     first[:] = np.nan
#     after = pd.Series(arr[N_TRAIN:]).rolling(n).mean()
#     return pd.Series(first).append(after)

class sma_bolling_strategy(Strategy):
    n1 = 1
    n2 = 40
    price_delta = .01
    buy_size = .2
    sell_size = .2

    def init(self):
        #df = self.data.df.iloc[:N_TRAIN]
        #X = get_X(df, ['Close'])
        close = self.data.Close
        self.mid_sma = self.I(SMA, close, 60)
        self.ma_exit = self.I(SMA, close, 90)
        self.std = self.I(std, close, 60)
        self.upper_band = self.mid_sma + self.std * 3  # todo: without multiply 2?
        self.lower_band = self.mid_sma - self.std * 3
        
        self.upper_band_1 = self.I(SMA, self.upper_band, 1)
        self.lower_band_1 = self.I(SMA, self.lower_band, 1)
        
        self.y_true = self.I(get_y, self.data.df, name='y_true')
        self.t = []
        self.p = []
        self.equities = []
        self.returns = []
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')
        self.data_len = self.data.df.shape[0]
        print(1)

    def next(self):
        try:
            self.equities.append(self.equity)
            # Skip the training, in-sample data
            if len(self.data) < N_TRAIN:
                return

            if len(self.data) > N_TRAIN:            
                self.returns.append((self.equity-self.equities[-2])/self.equities[-2])
            
            if (self.data.df.shape[0] >= self.data_len - 1):
                arr = np.array(self.returns)
                # print(np.mean(arr, axis=0), np.std(arr, axis=0))
                # print(np.mean(arr, axis=0)/np.std(arr, axis=0))
                print('Correct sharpe value: ', (np.mean(arr, axis=0)/np.std(arr, axis=0))*(np.sqrt(24*365)))
            
            # high, low, close = self.data.High, self.data.Low, self.data.Close
            # upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

            forecast = 0
            if crossover(self.ma_exit, self.lower_band):
                forecast = 1
            elif crossover_below(self.ma_exit, self.upper_band):
                forecast = -1

            if len(self.data) == N_TRAIN:
                if self.ma_exit[-1] > self.lower_band[-1]:
                    forecast = 1
                if self.ma_exit[-1] > self.upper_band[-1]:
                    forecast = -1

            if args.close_first > 0:
                if forecast == 1 and self.position.is_short:
                    self.position.close() # size=self.buy_size, tp=upper, sl=lower)
                elif forecast == 1 and not self.position.is_long:
                    self.buy()
                elif forecast == -1 and self.position.is_long:
                    self.position.close()
                elif forecast == -1 and not self.position.is_short:
                    self.sell() # size=self.sell_size, tp=lower, sl=upper)
            else:
                if forecast == 1 and not self.position.is_long:
                    self.buy() # size=self.buy_size, tp=upper, sl=lower)
                elif forecast == -1 and not self.position.is_short:
                    self.sell() # size=self.sell_size, tp=lower, sl=upper)

            self.forecasts[-1] = forecast

            self.t.append(int(self.y_true[-1]))
            self.p.append(int(forecast))
            
            print(self.data.df.shape[0], self.data_len)
            if (self.data.df.shape[0] == self.data_len - 1):
                print(accuracy_score(self.t, self.p))
                print(confusion_matrix(self.t, self.p))
                print(classification_report(self.t, self.p))
        except Exception as e:
            print(e)
            pass


class sma_bolling_walkforward_strategy(sma_bolling_strategy):
    def next(self):
        if len(self.data) < N_TRAIN:
            return
        if len(self.data) % 50:
            return super().next()
        super().next()
