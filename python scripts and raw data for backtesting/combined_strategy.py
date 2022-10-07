import datetime
import os
import random
import traceback

import keras
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import keras_tuner
import numpy as np
import tensorflow as tf
from backtesting import Strategy
from keras.layers import Dense, Flatten, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras_tuner import RandomSearch, HyperModel, Hyperband
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from model_matrix import get_clean_Xy, get_y, get_X
import argparse
from backtesting.test import SMA
from xgboost import XGBClassifier

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)
keras.utils.set_random_seed(1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)
args = parser.parse_args()  

output_size = 1
N_TRAIN = 5000
num_of_hours_per_slice = 6
epoch = 200

WINDOW = 10

def create_dataset(X, y, time_step=1):
    dataX, dataY = [], []
    for i in range(len(X)-time_step-1):
        a = X[i:(i+time_step)]
        dataX.append(a)
        if y is not None:
            dataY.append(y[i+time_step])
    return np.array(dataX), np.array(dataY)

# Build the RNN model
def build_lstm_model(features_count):
    model=keras.models.Sequential()
    model.add(keras.layers.LSTM(50,return_sequences=True,input_shape=(WINDOW,features_count)))
    model.add(keras.layers.LSTM(50,return_sequences=True))
    # model.add(keras.layers.LSTM(50,return_sequences=True)) 
    model.add(keras.layers.LSTM(50))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def split_sequence(sequence, _y, features_count):
    steps = num_of_hours_per_slice
    X, y = list(), list()
    for start in range(len(sequence)):
        end_index = start + steps
        if end_index > len(sequence) - 1:
            break

        sequence_x, sequence_y = sequence[start: end_index], _y[end_index - 1]
        X.append(sequence_x)
        y.append(sequence_y)
    return np.array(X), np.array(y)


def build_cnn_model(features_count):
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(num_of_hours_per_slice, features_count)))
    model.add(tf.keras.layers.Conv1D(352, kernel_size=12, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv1D(32, kernel_size=9, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv1D(160, kernel_size=5, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv1D(224, kernel_size=8, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv1D(64, kernel_size=6, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(Dense(85))
    model.add(Dense(75))
    model.add(Dense(65))
    model.add(Dense(3, activation="softmax"))
    model.compile(Adam(learning_rate=0.0001), "categorical_crossentropy", metrics=["accuracy"])
    return model

def most_frequent(List):
    return max(set(List), key = List.count)

class CombinedStrategy(Strategy):
    price_delta = .004
    buy_size = .2
    sell_size = .2
    
    n1 = 1
    n2 = 90

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.forecasts = None
        self.scaler = None
        self.clf = None

    def init(self):
        random.seed(1234)
        
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)
        self.sma_diff = self.sma1 - self.sma2

        # Train the classifier in advance on the first N_TRAIN examples
        self.full_data = self.data.df
        df = self.data.df.iloc[:N_TRAIN]
        X, y = get_clean_Xy(df)
        y[y < 0] = 2

        # Normalize
        self.scaler = preprocessing.StandardScaler()
        scaled_X = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca.fit(scaled_X)
        scaled_X = self.pca.transform(scaled_X)
        self.features_count = scaled_X.shape[1]


        self.lstm_clf = build_lstm_model(self.features_count)
        self.cnn_clf = build_cnn_model(self.features_count)
        self.knn_clf = KNeighborsClassifier(7)
        self.rf_clf = RandomForestClassifier(random_state=1234, n_estimators=100, max_features=None)
        self.mlp_clf = MLPClassifier(solver='adam', alpha=1e-4, activation="relu", batch_size=64, hidden_layer_sizes=(50, 50, 50), random_state=0, max_iter=50, verbose=True)
        self.xgb_clf = XGBClassifier(random_state=1234, n_estimators=50)

        # Plot y for inspection
        self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        self.y_true = self.I(get_y, self.data.df, name='y_true')
        self.y_true[self.y_true < 0] = 2

        self.data_len = self.data.df.shape[0]

        self.t = []
        self.p = []
        self.equities = []
        self.returns = []
        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

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
            
            # Proceed only with out-of-sample data. Prepare some variables
            high, low, close = self.data.High, self.data.Low, self.data.Close
            # current_time = self.data.index[-1]


            sma_diff = self.sma_diff[-1]
            sma_forecast = 0
            if sma_diff > 0:
                sma_forecast = 1
            elif sma_diff < 0:
                sma_forecast = 2

            X = get_X(self.data.df.iloc[-1:])
            X = self.scaler.transform(X)
            X = self.pca.transform(X)
            knn_forecast = self.knn_clf.predict(X)[0]
            rf_forecast = self.rf_clf.predict(X)[0]
            mlp_forecast = self.mlp_clf.predict(X)[0]
            xgb_forecast = self.xgb_clf.predict(X)[0]


            # Update the plotted "forecast" indicator
            X = get_X(self.data.df.iloc[-WINDOW-2:])
            scaled_X = self.scaler.transform(X)
            scaled_X = self.pca.transform(scaled_X)
            scaled_X, _ = create_dataset(scaled_X, None, WINDOW)
            lstm_forecast = self.lstm_clf.predict(scaled_X, batch_size=1, verbose=0)
            lstm_forecast = np.argmax(lstm_forecast[-1])

            # Forecast the next movement
            X = get_X(self.data.df.iloc[-num_of_hours_per_slice:])
            X = self.scaler.transform(X)
            X = self.pca.transform(X)
            X = np.reshape(X, (-1, num_of_hours_per_slice, self.features_count))
            cnn_forecast = self.cnn_clf.predict(X, verbose=0)
            cnn_forecast = np.argmax(cnn_forecast, axis=1)[0]

            model_forecasts = [knn_forecast, rf_forecast, mlp_forecast, xgb_forecast, lstm_forecast, cnn_forecast]
            forecast = most_frequent(model_forecasts)

            # Update the plotted "forecast" indicator
            self.forecasts[-1] = forecast

            self.t.append(int(self.y_true[-1]))
            self.p.append(int(forecast))
            
            print(self.data.df.shape[0], self.data_len, forecast)
            if (self.data.df.shape[0] == self.data_len - 1):
                print(accuracy_score(self.t, self.p))
                print(confusion_matrix(self.t, self.p))
                print(classification_report(self.t, self.p))

            upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

            if args.close_first > 0:
                if forecast == 1 and self.position.is_short:
                    self.position.close() # size=self.buy_size, tp=upper, sl=lower)
                elif forecast == 1 and not self.position.is_long:
                    self.buy()
                elif forecast == 2 and self.position.is_long:
                    self.position.close()
                elif forecast == 2 and not self.position.is_short:
                    self.sell() # size=self.sell_size, tp=lower, sl=upper)
            else:
                if forecast == 1 and not self.position.is_long:
                    self.buy() # size=self.buy_size, tp=upper, sl=lower)
                elif forecast == 2 and not self.position.is_short:
                    self.sell() # size=self.sell_size, tp=lower, sl=upper)
        except Exception as e:
            print(traceback.format_exc())
            pass


class CombinedWalkforwardStrategy(CombinedStrategy):
    def next(self):
        if len(self.data) < N_TRAIN:
            return

        if len(self.data) % 50:
            return super().next()

        df = self.data.df[-N_TRAIN:-1]
        X, y = get_clean_Xy(df)
        y[y < 0] = 2

        # Normalzie
        scaled_X = self.scaler.fit_transform(X)
        self.pca = self.pca.fit(scaled_X)
        scaled_X = self.pca.transform(scaled_X)

        self.features_count = scaled_X.shape[1]

        train = create_dataset(scaled_X, y, WINDOW)
        lstm_X = train[0]
        lstm_y = train[1]
        
        lstm_y = to_categorical(lstm_y.astype(int), 3)

        self.lstm_clf = build_lstm_model(self.features_count)
        self.lstm_clf.fit(
            lstm_X, lstm_y, batch_size=256, epochs=40, verbose=2
        )

        cnn_X, cnn_y = split_sequence(scaled_X, np.reshape(y, (-1, 1)), self.features_count)
        cnn_X = cnn_X.reshape((-1, num_of_hours_per_slice, self.features_count))


        self.knn_clf.fit(scaled_X, y)
        self.rf_clf.fit(scaled_X, y)
        self.mlp_clf.fit(scaled_X, y)
        self.xgb_clf.fit(scaled_X, y)


        # Save the weights using the `checkpoint_path` format
        # self.clf.save_weights(checkpoint_path.format(epoch=0))
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        self.cnn_clf = build_cnn_model(self.features_count)
        # print(self.clf.summary())
        self.cnn_clf.fit(
             cnn_X, to_categorical(cnn_y, 3), epochs=epoch, verbose=2,
             validation_split=0.1,
             callbacks=[  # cp_callback,
                 es_callback,
                 # tensorboard_callback
             ]
        )

        # Now that the model is fitted,
        # proceed the same as in MLTrainOnceStrategy
        super().next()
