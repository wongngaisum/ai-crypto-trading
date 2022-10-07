import argparse
import numpy as np
from backtesting import Backtest
import pandas as pd
from tensorflow import keras

from cnn_strategy import CNN_Walkforward_Strategy, CNNStrategy
from combined_strategy import CombinedWalkforwardStrategy
from combined_strategy2 import CombinedWalkforwardStrategy2
from hodl_strategy import HodlWalkForewardStrategy
from price_data import get_data_frame
from knn_strategy import KNNStrategy, KNNWalkForwardStrategy
from mlp_strategy import MLPStrategy, MLPWalkForwardStrategy
from sma_bolling_strategy import sma_bolling_strategy, sma_bolling_walkforward_strategy
from sma_strategy import sma_cross_walkforward_strategy
from rf_strategy import RandomForestStrategy, RFWalkForwardStrategy
from lstm_strategy import LSTMStrategy, LSTMWalkForwardStrategy
from xgb_strategy import XGBWalkForwardStrategy
from metrics import get_metrics
from model_matrix import get_clean_Xy, get_y

from vma_stratrgy import VMAStrategy

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)
args = parser.parse_args()  

if __name__ == '__main__':
    np.random.seed(1234)
    keras.utils.set_random_seed(1234)
    bt = Backtest(get_data_frame(), HodlWalkForewardStrategy,
                  cash=1000000, commission=args.fee, # 0.1% trading fee + 0.1% silppage
                  exclusive_orders=True)
    output = bt.run()
    print(output)
    bt.plot(resample=False)