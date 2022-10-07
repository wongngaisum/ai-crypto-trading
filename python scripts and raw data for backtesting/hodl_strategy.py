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


N_TRAIN = 5000


class HodlStrategy(Strategy):
    def init(self):
        self.y_true = self.I(get_y, self.data.df, name='y_true')
        self.t = []
        self.p = []
        self.equities = []
        self.returns = []
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')
        self.data_len = self.data.df.shape[0]

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
            
            if len(self.data) == N_TRAIN:
                forecast = 1

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


class HodlWalkForewardStrategy(HodlStrategy):
    def next(self):
        if len(self.data) < N_TRAIN:
            return
        if len(self.data) % 50:
            return super().next()
        super().next()
