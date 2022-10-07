import numpy as np
from backtesting import Strategy
from backtesting.test import SMA

# todo: fine tune below parameters
# we propose to use ma_period_long = 10, 30, 50, 100 and 200 hours

params = dict(
    ma_period_long=500
)
N_TRAIN = 5000


class VMAStrategy(Strategy):
    price_delta = .01
    buy_size = .2
    sell_size = .2

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.vma1 = None
        self.logP = None

    def init(self):
        self.logP = np.log10(self.data.Close)
        self.vma1 = self.I(SMA, self.logP, params['ma_period_long'])

    def next(self):
        if len(self.data) < N_TRAIN:
            return
        vma1 = self.vma1[-1]
        vma2 = np.log10(self.data.Close[-1])
        diff = vma2 - vma1
        print(diff)
        # Buy signal = if Short MA – Long MA > 0, then we would buy ETH.
        # Sell signal = if Short MA – Long MA < 0, then we would sell ETH
        high, low, close = self.data.High, self.data.Low, self.data.Close
        upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)
        if diff > 0:
            self.position.close()
            self.buy(sl=lower)
        elif diff < 0:
            self.position.close()
            self.sell(sl=upper)
