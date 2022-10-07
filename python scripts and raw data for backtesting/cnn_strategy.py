import datetime
import os
import random

import keras
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from model_matrix import get_clean_Xy, get_y, get_X
import argparse

np.random.seed(1234)
keras.utils.set_random_seed(1234)

parser = argparse.ArgumentParser("backtest")
parser.add_argument("lag", help="lag window of social media indicators", type=int)
parser.add_argument("fee", help="trading fee rate", type=float)
parser.add_argument("close_first", help="do not open opposite position immediately, close first", type=int)
args = parser.parse_args()  

output_size = 1
# N_TRAIN = 9911
N_TRAIN = 5000
# N_TRAIN = 667
input_dim = 71
num_of_hours_per_slice = 6
checkpoint_path = "training_cnn/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
epoch = 200

log_dir = "logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


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


def build_model(features_count):
    model = keras.Sequential()
    # keras.Input(shape=(features_count, num_of_hours_per_slice, 1)),
    # keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    # keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # # keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    # # keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.Flatten(),
    # keras.layers.Dropout(0.5),
    # keras.layers.Dense(3, activation="softmax"),
    ## test 3
    # keras.Input(shape=(features_count, num_of_hours_per_slice)),
    # keras.layers.Conv1D(128, kernel_size=8, padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.Activation("relu"),
    # keras.layers.Conv1D(256, kernel_size=5, padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.Activation("relu"),
    # keras.layers.Conv1D(128, kernel_size=3, padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.Activation("relu"),
    # keras.layers.GlobalAveragePooling1D(),
    # keras.layers.Dense(3, activation="softmax")
    ## test 4
    # tf.keras.Input(shape=(num_of_hours_per_slice, features_count)),
    # tf.keras.layers.Conv1D(32, kernel_size=num_of_hours_per_slice, padding="same"),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # # tf.keras.layers.MaxPooling1D(),
    # # tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Conv1D(32, kernel_size=features_count, padding="same"),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # tf.keras.layers.MaxPooling1D(pool_size=2),
    # tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Conv1D(16, kernel_size=2, padding="same"),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # tf.keras.layers.MaxPooling1D(pool_size=2),
    # tf.keras.layers.Conv1D(16, kernel_size=2, padding="same"),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation("relu"),
    # # tf.keras.layers.MaxPooling1D(pool_size=2),
    # # tf.keras.layers.MaxPooling1D(pool_size=2),
    # # tf.keras.layers.Flatten(),
    # # tf.keras.layers.Dropout(0.1),
    # # tf.keras.layers.GlobalAveragePooling2D(),
    # # tf.keras.layers.Dense(20),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Dense(20),
    # tf.keras.layers.Dense(20),
    # tf.keras.layers.Dense(3, activation="softmax")
    # test 5
    model.add(tf.keras.Input(shape=(num_of_hours_per_slice, features_count)))
    model.add(tf.keras.layers.Conv1D(352, kernel_size=12, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    # tf.keras.layers.MaxPooling1D(),
    # tf.keras.layers.Dropout(0.1),
    # for i in range(hp.Int("n_layers", 1, 4)):
    #     model.add(tf.keras.layers.Conv1D(hp.Int(f"conv_{i}_units", 32, 256, 32), kernel_size=3, padding="same"))
    #     model.add(tf.keras.layers.BatchNormalization())
    #     model.add(tf.keras.layers.Activation("relu"))
    #     # tf.keras.layers.MaxPooling1D(pool_size=2),
    #     # tf.keras.layers.Dropout(0.1),
    # model.add(GlobalAveragePooling1D())
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


class CNNStrategy(Strategy):
    price_delta = .004
    buy_size = .2
    sell_size = .2

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.forecasts = None
        self.scaler = None
        self.clf = None

    def init(self):
        random.seed(1234)
        # Train the classifier in advance on the first N_TRAIN examples
        self.full_data = self.data.df
        df = self.data.df.iloc[:N_TRAIN]
        X, y = get_clean_Xy(df)
        y[y < 0] = 2

        # Normalize
        self.scaler = preprocessing.StandardScaler()
        scaled_x = self.scaler.fit_transform(X)
        # self.tree = ExtraTreesClassifier(n_estimators=50, random_state=0)
        # self.tree = self.tree.fit(scaled_x, y)
        # self.tree = SelectFromModel(self.tree, prefit=True)
        # scaled_x = self.tree.transform(scaled_x)
        self.pca = PCA(n_components=0.99, svd_solver='full')
        self.pca.fit(scaled_x)
        scaled_x = self.pca.transform(scaled_x)
        self.features_count = scaled_x.shape[1]

        self.clf = build_model(self.features_count)
        # self.clf.compile(loss='mean_squared_error', optimizer='adam')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        # if latest_checkpoint is None:
        #     print('start training cnn model...')

        #     scaled_x = np.reshape(scaled_x, (-1, 1, self.features_count))
        #     y = np.reshape(y, (-1, 1))
        #     scaled_x2, y_2 = split_sequence(scaled_x, y, self.features_count)
        #     scaled_x2 = scaled_x2.reshape((-1, num_of_hours_per_slice, self.features_count))
        #     print('Shape of X is ', scaled_x2.shape)
        #     print('Shape of Y is ', y_2.shape)

        #     # Create a callback that saves the model's weights every 5 epochs
        #     cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #         filepath=checkpoint_path,
        #         verbose=1,
        #         save_weights_only=True,
        #         save_freq='epoch',
        #         period=10)

        #     # Save the weights using the `checkpoint_path` format
        #     # self.clf.save_weights(checkpoint_path.format(epoch=0))
        #     es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        #     self.clf.fit(
        #          scaled_x2, to_categorical(y_2, 3), epochs=epoch, verbose=2,
        #          validation_split=0.1,
        #          callbacks=[
        #              # cp_callback,
        #              es_callback,
        #              # tensorboard_callback
        #          ]
        #     )
        #     # self.tuner = Hyperband(
        #     #      build_model,
        #     #      objective='val_loss',
        #     #     project_name='intro_to_kt'
        #     #      #max_trials=5
        #     # )
        #     #self.tuner.search(scaled_x2, to_categorical(y_2, 3), epochs=epoch, validation_split=0.1)
        #     #print("best_model=")
        #     #print(self.tuner.get_best_models()[0])

        # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # self.clf.load_weights(latest_checkpoint)

        # Plot y for inspection
        self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        ########
        # df = self.data.df.iloc[N_TRAIN:]
        # X_test, y_test = get_clean_Xy(df)
        # y_test[y_test < 0] = 2
        # scaled_X_test = self.scaler.fit_transform(X_test)
        # scaled_X_test = self.tree.transform(scaled_X_test)
        # scaled_X_test = np.reshape(scaled_X_test, (-1, 1, self.features_count))
        # y_test = np.reshape(y_test, (-1, 1))
        # scaled_X_test, y_test = split_sequence(scaled_X_test, y_test, self.features_count)
        # scaled_X_test = scaled_X_test.reshape((-1, self.features_count, num_of_hours_per_slice))
        #
        # y_train_pred = self.clf.predict(scaled_X_test)
        # y_train_pred = np.argmax(y_train_pred, axis=1)
        # y_train_pred = np.reshape(y_train_pred, (-1, 1))
        # print(accuracy_score(y_test, y_train_pred))
        # print(confusion_matrix(y_test, y_train_pred))
        # print(classification_report(y_test, y_train_pred))
        # print(1)
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

            # Forecast the next movement
            X = get_X(self.data.df.iloc[-num_of_hours_per_slice:])

            X = self.scaler.transform(X)

            X = self.pca.transform(X)
            # X = self.tree.transform(X)

            # X = np.reshape(X, (-1, 1, self.features_count))
            # X = X.reshape((1, self.features_count, num_of_hours_per_slice))
            X = np.reshape(X, (-1, num_of_hours_per_slice, self.features_count))
            forecast = self.clf.predict(X, verbose=0)
            forecast = np.argmax(forecast, axis=1)[0]
            print(self.data.index[-1], forecast)

            # Update the plotted "forecast" indicator
            self.forecasts[-1] = forecast

            self.t.append(int(self.y_true[-1]))
            self.p.append(int(forecast))
            print(self.data.df.shape[0], self.data_len)
            print(accuracy_score(self.t, self.p))
            print(confusion_matrix(self.t, self.p))
            print(classification_report(self.t, self.p))
            print(1)
            # if self.data.df.shape[0] == self.data_len - 4:
            #     print(accuracy_score(self.t, self.p))
            #     print(confusion_matrix(self.t, self.p))
            #     print(classification_report(self.t, self.p))
            #     print(1)

            # If our forecast is upwards and we don't already hold a long position
            # place a long order for 20% of available account equity. Vice versa for short.
            # Also set target take-profit and stop-loss prices to be one price_delta
            # away from the current closing price.
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

            # # Additionally, set aggressive stop-loss on trades that have been open
            # # for more than two days
            # for trade in self.trades:
            #     if current_time - trade.entry_time > pd.Timedelta('1 hour'):
            #         if trade.is_long:
            #             trade.sl = max(trade.sl, low)
            #         else:
            #             trade.sl = min(trade.sl, high)
        except Exception as e:
            print(e)
            pass


class CNN_Walkforward_Strategy(CNNStrategy):
    def next(self):
        if len(self.data) < N_TRAIN:
            return

        if len(self.data) % 50:
            return super().next()

        df = self.data.df[-N_TRAIN:-1]
        X, y = get_clean_Xy(df)

        # Normalzie
        #self.scaler = preprocessing.StandardScaler()
        scaled_X = self.scaler.fit_transform(X)
        self.pca = self.pca.fit(scaled_X)
        scaled_X = self.pca.transform(scaled_X)
        # self.tree = ExtraTreesClassifier(n_estimators=10, random_state=0)
        # self.tree = self.tree.fit(scaled_X, y)
        # self.tree = SelectFromModel(self.tree, prefit=True)
        # scaled_X = self.tree.transform(scaled_X)

        self.features_count = scaled_X.shape[1]
        scaled_x = scaled_X # np.reshape(scaled_X, (-1, 1, self.features_count))
        y = np.reshape(y, (-1, 1))
        scaled_x2, y_2 = split_sequence(scaled_x, y, self.features_count)
        scaled_x2 = scaled_x2.reshape((-1, num_of_hours_per_slice, self.features_count))
        print('Shape of X is ', scaled_x2.shape)
        print('Shape of Y is ', to_categorical(y_2, 3).shape)


        # df2 = self.full_data[self.data.index[-1] - 39:]
        # val_X, val_y = get_clean_Xy(df2)
        # val_y[val_y < 0] = 2
        # #self.scaler = preprocessing.StandardScaler()
        # scaled_val_X = self.scaler.fit_transform(val_X)
        # #self.pca.fit(scaled_val_X)
        # scaled_val_X = self.pca.transform(scaled_val_X)
        # #scaled_val_X = np.reshape(scaled_val_X, (-1, 1, self.features_count))
        # val_y = np.reshape(y, (-1, 1))
        # scaled_val_X2, val_y2 = split_sequence(scaled_val_X, val_y, self.features_count)
        # scaled_val_X2 = scaled_val_X2.reshape((-1, num_of_hours_per_slice, self.features_count))
        # print('Shape of X is ', scaled_val_X2.shape)
        # print('Shape of Y is ', val_y.shape)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq='epoch',
            period=10)

        # Save the weights using the `checkpoint_path` format
        # self.clf.save_weights(checkpoint_path.format(epoch=0))
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        self.clf = build_model(self.features_count)
        # print(self.clf.summary())
        self.clf.fit(
             scaled_x2, to_categorical(y_2, 3), epochs=epoch, verbose=2,
             validation_split=0.1,
             callbacks=[  # cp_callback,
                 es_callback,
                 # tensorboard_callback
             ]
         )
        # self.tuner.search(scaled_x2, to_categorical(y_2, 3), epochs=epoch, validation_split=0.1)
        # print("best_model=")
        # print(self.tuner.get_best_models()[0])

        # Now that the model is fitted,
        # proceed the same as in MLTrainOnceStrategy
        super().next()
